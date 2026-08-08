#include <cpprcwa/fft_funs.h>
#include "internal/lapack_wrappers.h"
#include "internal/utils.h"
#include <fftw3.h>
#include <mutex>

namespace cpprcwa {

// ── FFTWPlanCache::Impl ──────────────────────────────────────────────────
struct FFTWPlanCache::Impl {
    using Key = std::pair<int,int>;
    struct KeyHash {
        size_t operator()(const Key& k) const noexcept {
            return std::hash<long long>()((long long)k.first * 1000000LL + k.second);
        }
    };
    std::unordered_map<Key, std::pair<fftw_plan, fftw_plan>, KeyHash> plans;
    std::mutex mtx;

    std::pair<fftw_plan, fftw_plan> get(int Nx, int Ny) {
        std::lock_guard<std::mutex> lock(mtx);
        Key key{Nx, Ny};
        auto it = plans.find(key);
        if (it != plans.end()) return it->second;
        auto* fwd_buf = static_cast<complex*>(fftw_malloc(sizeof(complex) * Nx * Ny));
        auto* bwd_buf = static_cast<complex*>(fftw_malloc(sizeof(complex) * Nx * Ny));
        fftw_plan fwd = fftw_plan_dft_2d(Nx, Ny,
                                         reinterpret_cast<fftw_complex*>(fwd_buf),
                                         reinterpret_cast<fftw_complex*>(fwd_buf),
                                         FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan bwd = fftw_plan_dft_2d(Nx, Ny,
                                         reinterpret_cast<fftw_complex*>(bwd_buf),
                                         reinterpret_cast<fftw_complex*>(bwd_buf),
                                         FFTW_BACKWARD, FFTW_ESTIMATE);
        auto result = std::make_pair(fwd, bwd);
        plans.emplace(key, result);
        // Note: the FFTW plan references the buffers it was created with.
        // Our wrappers copy data into a contiguous buffer before execute,
        // so the planning buffers are owned by us for the plan's lifetime.
        // (Caller is responsible for providing their own arrays at execute.)
        return result;
    }

    ~Impl() {
        for (auto& kv : plans) {
            if (kv.second.first)  fftw_destroy_plan(kv.second.first);
            if (kv.second.second) fftw_destroy_plan(kv.second.second);
        }
    }
};

// ── FFTWPlanCache outer-class methods ─────────────────────────────────────
FFTWPlanCache::FFTWPlanCache() : impl_(new Impl()) {}

FFTWPlanCache::~FFTWPlanCache() { delete impl_; }

std::pair<fftw_plan, fftw_plan> FFTWPlanCache::get(int Nx, int Ny) {
    return impl_->get(Nx, Ny);
}

// ── FFT helpers ─────────────────────────────────────────────────────────────

namespace {

// Execute FFTW forward 2D plan in place on a row-major (Nx, Ny) buffer.
void fftw_execute_fwd(fftw_plan plan, complex* buf) {
    fftw_execute_dft(plan, reinterpret_cast<fftw_complex*>(buf),
                            reinterpret_cast<fftw_complex*>(buf));
}

void fftw_execute_bwd(fftw_plan plan, complex* buf) {
    fftw_execute_dft(plan, reinterpret_cast<fftw_complex*>(buf),
                            reinterpret_cast<fftw_complex*>(buf));
}

} // namespace

// ── Public FFT functions ────────────────────────────────────────────────────

ComplexMatrix get_conv(double dN,
                       const std::vector<complex>& s_in_flat,
                       int Nx, int Ny,
                       const IntMatrix& G) {
    int nG = G.rows();
    static FFTWPlanCache cache;
    auto [plan_f, plan_b] = cache.get(Nx, Ny);
    (void)plan_b;

    // s_grid (row-major), then forward FFT in place, scale by dN.
    std::vector<complex> buf(s_in_flat.begin(), s_in_flat.end());
    buf.resize((size_t)Nx * Ny);
    fftw_execute_fwd(plan_f, buf.data());
    for (auto& z : buf) z *= dN;

    // C[i,j] = s_fft[(G[i,0]-G[j,0]) mod Nx, (G[i,1]-G[j,1]) mod Ny]
    ComplexMatrix C(nG, nG);
    for (int i = 0; i < nG; ++i) {
        for (int j = 0; j < nG; ++j) {
            int row = ((G(i,0) - G(j,0)) % Nx + Nx) % Nx;
            int col = ((G(i,1) - G(j,1)) % Ny + Ny) % Ny;
            C(i, j) = buf[(size_t)row * Ny + col];
        }
    }
    return C;
}

ComplexVector get_fft(double dN,
                      const std::vector<complex>& s_in_flat,
                      int Nx, int Ny,
                      const IntMatrix& G) {
    int nG = G.rows();
    static FFTWPlanCache cache;
    auto [plan_f, plan_b] = cache.get(Nx, Ny);
    (void)plan_b;

    std::vector<complex> buf(s_in_flat.begin(), s_in_flat.end());
    buf.resize((size_t)Nx * Ny);
    fftw_execute_fwd(plan_f, buf.data());
    for (auto& z : buf) z *= dN;

    ComplexVector out(nG);
    for (int i = 0; i < nG; ++i) {
        int r = ((G(i,0) % Nx) + Nx) % Nx;
        int c = ((G(i,1) % Ny) + Ny) % Ny;
        out(i) = buf[(size_t)r * Ny + c];
    }
    return out;
}

GridMatrix get_ifft(double dN, int Nx, int Ny,
                    const ComplexVector& s_in,
                    const IntMatrix& G) {
    int nG = G.rows();
    static FFTWPlanCache cache;
    auto [plan_f, plan_b] = cache.get(Nx, Ny);
    (void)plan_f;

    // Scatter s_in at positions G[i] (with wrap), zero elsewhere.
    std::vector<complex> buf((size_t)Nx * Ny, complex(0.0, 0.0));
    for (int i = 0; i < nG; ++i) {
        int r = ((G(i,0) % Nx) + Nx) % Nx;
        int c = ((G(i,1) % Ny) + Ny) % Ny;
        buf[(size_t)r * Ny + c] += s_in(i);
    }
    fftw_execute_bwd(plan_b, buf.data());

    // FFTW backward doesn't normalize; divide by Nx*Ny. Also divide by dN
    // to match Python convention (s_out /= dN where dN = 1/(Nx*Ny)).
    // i.e. total scale: 1/(Nx*Ny) / dN = 1/(Nx*Ny*dN). But Python does:
    //   s_out = ifft2(s0) / dN  and numpy ifft2 already divides by N.
    // FFTW bwd = sum (no normalization) = N * (numpy ifft). So scale = (1/N) / dN.
    double scale = 1.0 / ((double)Nx * Ny * dN);
    GridMatrix out(Nx, Ny);
    for (int r = 0; r < Nx; ++r)
        for (int c = 0; c < Ny; ++c)
            out(r, c) = buf[(size_t)r * Ny + c] * scale;
    return out;
}

EpsilonFftResult Epsilon_fft(double dN,
                             const std::vector<complex>& eps_grid,
                             int Nx, int Ny,
                             const IntMatrix& G) {
    ComplexMatrix eps_fft = get_conv(dN, eps_grid, Nx, Ny, G);
    int nG = G.rows();
    EpsilonFftResult r;
    r.epsinv = internal::zinverse(eps_fft);
    r.eps2 = ComplexMatrix::Zero(2 * nG, 2 * nG);
    r.eps2.topLeftCorner(nG, nG)     = eps_fft;
    r.eps2.bottomRightCorner(nG, nG) = eps_fft;
    return r;
}

EpsilonFftResult Epsilon_fft(double dN,
                             const std::vector<std::vector<complex>>& eps_grid,
                             int Nx, int Ny,
                             const IntMatrix& G) {
    // eps_grid = [epsx, epsy, epsz]
    ComplexMatrix epsx_fft = get_conv(dN, eps_grid[0], Nx, Ny, G);
    ComplexMatrix epsy_fft = get_conv(dN, eps_grid[1], Nx, Ny, G);
    ComplexMatrix epsz_fft = get_conv(dN, eps_grid[2], Nx, Ny, G);
    int nG = G.rows();
    EpsilonFftResult r;
    r.epsinv = internal::zinverse(epsz_fft);
    r.eps2 = ComplexMatrix::Zero(2 * nG, 2 * nG);
    r.eps2.topLeftCorner(nG, nG)     = epsx_fft;
    r.eps2.bottomRightCorner(nG, nG) = epsy_fft;
    return r;
}

} // namespace cpprcwa
