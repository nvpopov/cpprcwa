// bench_zgeev — GPU-vs-CPU linear-algebra feasibility spike (PLAN.md Phase 9).
//
// Purpose: get hard numbers BEFORE any GPU backend work. Two questions:
//   A) Is there a GPU non-Hermitian complex eigensolver to replace LAPACK
//      zgeev?  → DECISIVE ANSWER: NO. cuSOLVER (verified CUDA 11.5 AND 12.8
//      headers) exposes only Hermitian eigensolvers (Zheevd/Zheevj) and SVD
//      (Zgesvd/Zgesvdj). There is NO cusolverDnZgeev. A GPU zgeev would need
//      third-party MAGMA (magma_zgeev) or a custom QR iteration. Per the plan
//      this descopes Phase 9 to cuFFT/cuBLAS only, keeping zgeev on CPU.
//   B) What would the descoped cuFFT/cuBLAS path actually buy? We measure the
//      two dominant pieces against their CPU counterparts:
//        - cuBLAS zgemm vs OpenBLAS zgemm at the S-matrix (2nG)³ matmul size
//        - cuFFT vs FFTW 2D transform (Epsilon_fft convolution matrices)
//
// The test matrix for the zgeev leg is the REAL RCWA patterned-layer matrix
//   M = eps2·kp − k·kᵀ   ((2nG)², non-Hermitian)
// built exactly as SolveLayerEigensystem_patterned does (TaN bar on a square
// cell, complex omega for loss, oblique incidence). Not a random matrix: the
// eigenvalue structure / conditioning is what matters.
//
// Build (CPU only):
//   g++ -O3 -march=native -std=c++17 -I/usr/include/eigen3 \
//       -o bench_zgeev bench_zgeev.cpp -lopenblas -lfftw3 -lpthread -lm
//
// Build (CPU + GPU, apt CUDA toolkits: libcublas-dev libcufft-dev libfftw3-dev):
//   g++ -O3 -march=native -std=c++17 -DUSE_CUDA -I/usr/include/eigen3 \
//       -o bench_zgeev bench_zgeev.cpp -lopenblas -lfftw3 -lcublas -lcufft \
//       -lcudart -lpthread -lm
//   (pure g++ is fine — no kernels, only cuBLAS/cuFFT library calls)
//
// Usage:
//   ./bench_zgeev [nG1 nG2 ...]     # truncation orders, default 200 500 1000
//                                   # (nG=2000 => 2nG=4000; CPU zgeev ~10-20 min)
//   OPENBLAS_NUM_THREADS=N ./bench_zgeev   # CPU thread count for zgemm/zgeev
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#endif

using complex = std::complex<double>;
using MatrixXcd = Eigen::MatrixXcd;
using VectorXcd = Eigen::VectorXcd;

extern "C" {
void zgeev_(char* jobvl, char* jobvr, int* n, std::complex<double>* a, int* lda,
            std::complex<double>* w, std::complex<double>* vl, int* ldvl,
            std::complex<double>* vr, int* ldvr, std::complex<double>* work,
            int* lwork, double* rwork, int* info);
void zgemm_(char* transa, char* transb, int* m, int* n, int* k,
            std::complex<double>* alpha, std::complex<double>* a, int* lda,
            std::complex<double>* b, int* ldb, std::complex<double>* beta,
            std::complex<double>* c, int* ldc);
}

#include <fftw3.h>

namespace {

double sinc(double x) {
    if (std::abs(x) < 1e-12) return 1.0;
    const double px = M_PI * x;
    return std::sin(px) / px;
}

// Circular reciprocal-lattice truncation (mirrors Lattice_getG method=0).
std::vector<std::pair<int, int>> make_G(int nG) {
    const double R = std::sqrt((double)nG / M_PI) + 1.0;
    const int M = (int)std::ceil(R) + 1;
    std::vector<std::pair<double, std::pair<int, int>>> cand;
    for (int i = -M; i <= M; ++i)
        for (int j = -M; j <= M; ++j) {
            const double r2 = (double)(i * i) + (double)(j * j);
            if (r2 <= R * R + 1e-9) cand.push_back({r2, {i, j}});
        }
    std::stable_sort(cand.begin(), cand.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    if (cand.size() < (size_t)nG) {
        std::fprintf(stderr, "bench_zgeev: could not build %d G vectors\n", nG);
        std::exit(1);
    }
    std::vector<std::pair<int, int>> out(nG);
    for (int k = 0; k < nG; ++k) out[k] = cand[k].second;
    return out;
}

// RCWA patterned-layer matrix M = eps2·kp − k·kᵀ  ((2nG)², non-Hermitian).
MatrixXcd build_M(int nG) {
    const double P = 3500.0;                              // cell (nm)
    const double Wx = 500.0 / P, Wy = 500.0 / P;          // TaN bar fill fraction
    const complex eps_bg(1.0, 0.0);
    const complex n_tan(0.9562, 0.0323);
    const complex d_eps = n_tan * n_tan - eps_bg;
    const double lambda = 13.5, Q = 500.0;
    const complex omega = 2.0 * M_PI / lambda * complex(1.0, 1.0 / (2.0 * Q));
    const double theta = 0.05, phi = 0.1;                 // oblique, complex k0

    const auto G = make_G(nG);
    std::vector<complex> kx(nG), ky(nG);
    for (int i = 0; i < nG; ++i) {
        kx[i] = omega * std::sin(theta) * std::cos(phi) + 2.0 * M_PI * G[i].first / P;
        ky[i] = omega * std::sin(theta) * std::sin(phi) + 2.0 * M_PI * G[i].second / P;
    }

    auto f = [&](int gi, int gj) -> complex {   // permittivity Fourier coeff
        if (gi == 0 && gj == 0) return eps_bg + d_eps * Wx * Wy;
        const double sx = (gi == 0) ? 1.0 : sinc(gi * Wx);
        const double sy = (gj == 0) ? 1.0 : sinc(gj * Wy);
        return d_eps * Wx * Wy * sx * sy;
    };

    MatrixXcd C(nG, nG);                       // convolution matrix C[i,j]=f[G_i−G_j]
    for (int i = 0; i < nG; ++i)
        for (int j = 0; j < nG; ++j)
            C(i, j) = f(G[i].first - G[j].first, G[i].second - G[j].second);

    MatrixXcd epinv = C.inverse();
    MatrixXcd eps2 = MatrixXcd::Zero(2 * nG, 2 * nG);
    eps2.topLeftCorner(nG, nG)     = C;
    eps2.bottomRightCorner(nG, nG) = C;

    MatrixXcd Jk = MatrixXcd::Zero(2 * nG, nG);  // vstack(diag(−ky), diag(kx))
    for (int i = 0; i < nG; ++i) { Jk(i, i) = -ky[i]; Jk(nG + i, i) = kx[i]; }
    MatrixXcd kp = (omega * omega) * MatrixXcd::Identity(2 * nG, 2 * nG)
                   - (Jk * epinv) * Jk.transpose();

    MatrixXcd k = MatrixXcd::Zero(2 * nG, nG);   // vstack(diag(kx), diag(ky))
    for (int i = 0; i < nG; ++i) { k(i, i) = kx[i]; k(nG + i, i) = ky[i]; }
    MatrixXcd kkT = k * k.transpose();

    return eps2 * kp - kkT;
}

double max_rel_residual(const MatrixXcd& M, const std::vector<complex>& w,
                        const MatrixXcd& VR) {
    const int n = (int)M.rows();
    double mn = 0.0;                               // ||M||_1
    for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += std::abs(M(i, j));
        mn = std::max(mn, s);
    }
    double res = 0.0;
    const int ns = std::min(5, n);
    for (int s = 0; s < ns; ++s) {
        const int idx = s * (n / ns);              // sample across the spectrum
        VectorXcd v = VR.col(idx);
        VectorXcd Mv = M * v;
        const double nr = (Mv - w[idx] * v).norm();
        res = std::max(res, nr / (mn * v.norm()));
    }
    return res;
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// CPU: LAPACK zgeev_ (OpenBLAS). Best wall time over `runs`.
double time_cpu_zgeev(const MatrixXcd& M, int runs, double& max_res) {
    int n = (int)M.rows();
    char V = 'V', N = 'N';
    std::vector<complex> w(n), vl(n * n), vr(n * n), work;
    std::vector<double> rwork(2 * n);
    int info = 0, lwork = -1;
    complex wq;
    MatrixXcd A = M;
    zgeev_(&N, &V, &n, A.data(), &n, w.data(), vl.data(), &n, vr.data(), &n,
           &wq, &lwork, rwork.data(), &info);
    lwork = (int)wq.real();
    work.resize(lwork);
    A = M;   // warmup
    zgeev_(&N, &V, &n, A.data(), &n, w.data(), vl.data(), &n, vr.data(), &n,
           work.data(), &lwork, rwork.data(), &info);

    double best = 1e300;
    for (int r = 0; r < runs; ++r) {
        A = M;
        const double t0 = now_ms();
        zgeev_(&N, &V, &n, A.data(), &n, w.data(), vl.data(), &n, vr.data(), &n,
               work.data(), &lwork, rwork.data(), &info);
        const double t1 = now_ms();
        best = std::min(best, t1 - t0);
    }
    MatrixXcd VR = Eigen::Map<const MatrixXcd>(vr.data(), n, n);
    max_res = max_rel_residual(M, w, VR);
    return best;
}

// CPU: OpenBLAS zgemm_ for an (n×n)·(n×n) product = 2n³ flops (S-matrix scale).
double bench_cpu_zgemm(int n, int runs) {
    std::vector<complex> A((size_t)n * n), B((size_t)n * n), C((size_t)n * n);
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = complex((double)(i % 7) + 1.0, 1.0);
        B[i] = complex(1.0, (double)(i % 5));
    }
    char N = 'N';
    complex alpha(1.0, 0.0), beta(0.0, 0.0);
    zgemm_(&N, &N, &n, &n, &n, &alpha, A.data(), &n, B.data(), &n, &beta,
           C.data(), &n);
    double best = 1e300;
    for (int r = 0; r < runs; ++r) {
        const double t0 = now_ms();
        zgemm_(&N, &N, &n, &n, &n, &alpha, A.data(), &n, B.data(), &n, &beta,
               C.data(), &n);
        const double t1 = now_ms();
        best = std::min(best, t1 - t0);
    }
    return best;
}

// CPU: FFTW 2D forward transform of an (nx×ny) complex grid.
double bench_cpu_fft(int nx, int ny, int runs) {
    fftw_complex* in = fftw_alloc_complex((size_t)nx * ny);
    fftw_complex* out = fftw_alloc_complex((size_t)nx * ny);
    fftw_plan p = fftw_plan_dft_2d(nx, ny, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);   // warmup
    double best = 1e300;
    for (int r = 0; r < runs; ++r) {
        const double t0 = now_ms();
        fftw_execute(p);
        const double t1 = now_ms();
        best = std::min(best, t1 - t0);
    }
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return best;
}

#if defined(USE_CUDA)
#define CHECK_CUDA(stmt) do { cudaError_t e_ = (stmt);                        \
    if (e_ != cudaSuccess) {                                                  \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),\
                     __FILE__, __LINE__); std::exit(1); } } while (0)
#define CHECK_CUBLAS(stmt) do { cublasStatus_t s_ = (stmt);                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                        \
        std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_,           \
                     __FILE__, __LINE__); std::exit(1); } } while (0)
#define CHECK_CUFFT(stmt) do { cufftResult r_ = (stmt);                       \
    if (r_ != CUFFT_SUCCESS) {                                                \
        std::fprintf(stderr, "cuFFT error %d at %s:%d\n", (int)r_,            \
                     __FILE__, __LINE__); std::exit(1); } } while (0)

// GPU: cuBLAS zgemm (n×n)·(n×n). Timing includes the host->device copies of
// A/B (honest: the S-matrix blocks live on the host in RCWA).
double bench_gpu_zgemm(int n, int runs) {
    cublasHandle_t h;
    CHECK_CUBLAS(cublasCreate(&h));
    std::vector<complex> A((size_t)n * n), B((size_t)n * n), C((size_t)n * n);
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = complex((double)(i % 7) + 1.0, 1.0);
        B[i] = complex(1.0, (double)(i % 5));
    }
    cuDoubleComplex *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(cuDoubleComplex) * A.size()));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(cuDoubleComplex) * B.size()));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(cuDoubleComplex) * C.size()));
    CHECK_CUDA(cudaMemcpy(dA, A.data(), sizeof(cuDoubleComplex) * A.size(),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), sizeof(cuDoubleComplex) * B.size(),
                          cudaMemcpyHostToDevice));
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    CHECK_CUBLAS(cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                             &alpha, dA, n, dB, n, &beta, dC, n));
    CHECK_CUDA(cudaDeviceSynchronize());
    double best = 1e300;
    for (int r = 0; r < runs; ++r) {
        const double t0 = now_ms();
        CHECK_CUBLAS(cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &alpha, dA, n, dB, n, &beta, dC, n));
        CHECK_CUDA(cudaDeviceSynchronize());
        const double t1 = now_ms();
        best = std::min(best, t1 - t0);
    }
    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return best;
}

// GPU: cuFFT 2D forward transform of an (nx×ny) complex grid.
double bench_gpu_fft(int nx, int ny, int runs) {
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan2d(&plan, nx, ny, CUFFT_Z2Z));
    cuDoubleComplex *din = nullptr, *dout = nullptr;
    CHECK_CUDA(cudaMalloc(&din, sizeof(cuDoubleComplex) * (size_t)nx * ny));
    CHECK_CUDA(cudaMalloc(&dout, sizeof(cuDoubleComplex) * (size_t)nx * ny));
    CHECK_CUFFT(cufftExecZ2Z(plan, din, dout, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());
    double best = 1e300;
    for (int r = 0; r < runs; ++r) {
        const double t0 = now_ms();
        CHECK_CUFFT(cufftExecZ2Z(plan, din, dout, CUFFT_FORWARD));
        CHECK_CUDA(cudaDeviceSynchronize());
        const double t1 = now_ms();
        best = std::min(best, t1 - t0);
    }
    cufftDestroy(plan);
    cudaFree(din); cudaFree(dout);
    return best;
}
#endif

} // namespace

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {200, 500, 1000};
    const int runs = 3;
    const int fft_nx = 512, fft_ny = 512;   // representative eps grid

    std::printf("=== A) zgeev: CPU LAPACK baseline (GPU leg — see note) ===\n");
    std::printf("matrix: RCWA patterned-layer M = eps2*kp - k*kT (2nG)^2, non-Hermitian\n");
    std::printf("residual = max rel ||M v - lambda v|| over a sample of eigenpairs\n");
    std::printf("%5s %6s %12s %10s\n", "nG", "2nG", "CPU ms", "CPU res");
    for (int nG : sizes) {
        MatrixXcd M = build_M(nG);
        double cres = 0.0;
        const double ct = time_cpu_zgeev(M, runs, cres);
        std::printf("%5d %6d %12.1f %10.2e\n", nG, 2 * nG, ct, cres);
        std::fflush(stdout);
    }
#if defined(USE_CUDA)
    int cuda_ver = 0;
    cudaRuntimeGetVersion(&cuda_ver);
    std::printf("NOTE: cusolverDnZgeev is NOT part of cuSOLVER (CUDA %d.%d headers):\n"
                "  only Hermitian (Zheevd/Zheevj) and SVD (Zgesvd/Zgesvdj) exist.\n"
                "  A GPU non-Hermitian zgeev would require MAGMA (magma_zgeev) or a\n"
                "  custom QR iteration. => Phase 9 descopes to cuFFT/cuBLAS, zgeev on CPU.\n",
                cuda_ver / 1000, (cuda_ver % 100) / 10);
#else
    std::printf("NOTE: GPU leg not compiled (add -DUSE_CUDA to build with cuBLAS/cuFFT).\n");
#endif

    std::printf("\n=== B) descoped Phase-9 components (CPU vs GPU) ===\n");
#if defined(USE_CUDA)
    std::printf("S-matrix matmul (2nG)^3:  %5s %6s %12s %12s %8s\n",
                "nG", "2nG", "OpenBLAS ms", "cuBLAS ms", "spdup");
    for (int nG : sizes) {
        const int n2 = 2 * nG;
        const double ct = bench_cpu_zgemm(n2, runs);
        const double gt = bench_gpu_zgemm(n2, runs);
        std::printf("%5d %6d %12.1f %12.1f %7.1fx\n", nG, n2, ct, gt, ct / gt);
        std::fflush(stdout);
    }
    {
        const double ct = bench_cpu_fft(fft_nx, fft_ny, runs);
        const double gt = bench_gpu_fft(fft_nx, fft_ny, runs);
        std::printf("eps FFT (%dx%d): FFTW %.3f ms  cuFFT %.3f ms  speedup %.1fx\n",
                    fft_nx, fft_ny, ct, gt, ct / gt);
    }
#else
    std::printf("(build with -DUSE_CUDA to include cuBLAS/cuFFT measurements)\n");
#endif
    std::printf("\nrule (PLAN.md Phase 9): pursue GPU only if a component is >=2x faster;\n");
    std::printf("zgeev stays on CPU either way (no cuSOLVER routine exists).\n");
    return 0;
}
