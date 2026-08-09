// bench_euv_stack — end-to-end EUV absorber benchmark (quasi-1D and full-2D).
//
// Same geometry as ex_quasi1d_absorber (500 nm TaN bar, Lx=3.5 um, 40× Mo/Si
// multilayer, Si substrate). Reports best-of-N wall-clock for setup
// (Init_Setup + GridLayer_geteps: FFT convolution + patterned-layer zgeev) and
// rt_solve (S-matrix), plus the reflectivity.
//
// Usage:
//   bench_euv_stack [--quasi1d] [--theta DEG] [--phi DEG] [--runs N] nG1 nG2 ...
//   OPENBLAS_NUM_THREADS=N bench_euv_stack ...     # CPU threads (default 6)
#include <cpprcwa/cpprcwa.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using cpprcwa::complex;

namespace {
complex eps_from_n(complex n) { return n * n; }
} // namespace

int main(int argc, char** argv) {
    using namespace cpprcwa;
    using ms = std::chrono::duration<double, std::milli>;

    const complex n_mo(0.9226, 0.0064), n_si(0.9997, 0.0018);
    const complex n_ru(0.9114, 0.0171), n_tan(0.9562, 0.0323);
    const double Lx = 3500.0, wx = 500.0, lambda = 13.5, Ly = 5.0 * lambda;
    const double abs_t = 60.0;
    const int nb = 40, Nx = 1000, Ny = 4;

    bool quasi1d = false;
    double theta_deg = 0.0, phi_deg = 0.0;
    int runs = 5;
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quasi1d") == 0) quasi1d = true;
        else if (std::strcmp(argv[i], "--theta") == 0 && i + 1 < argc) theta_deg = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--phi") == 0 && i + 1 < argc) phi_deg = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = std::atoi(argv[++i]);
        else sizes.push_back(std::atoi(argv[i]));
    }
    if (sizes.empty()) sizes = {201, 500, 1000};

    // TaN bar: grid holds n (not n^2) — shared grcwa convention.
    std::vector<complex> epgrid((size_t)Nx * Ny, complex(1.0, 0.0));
    const double half = wx / (2.0 * Lx);
    for (int i = 0; i < Nx; ++i) {
        const double u = (double)i / (Nx - 1);
        if (std::fabs(u - 0.5) >= half) continue;
        for (int j = 0; j < Ny; ++j) epgrid[(size_t)i * Ny + j] = n_tan;
    }

    const bool normal = (theta_deg == 0.0);
    const char* mode =
        quasi1d ? (normal ? "QUASI-1D NORMAL" : "QUASI-1D OBLIQUE")
                : (normal ? "FULL-2D NORMAL" : "FULL-2D OBLIQUE");
    std::printf("EUV absorber benchmark — mode: %s (theta=%.1f phi=%.1f), best of %d\n",
                mode, theta_deg, phi_deg, runs);
    std::printf("%6s %7s %12s %12s %12s %9s\n",
                "nG", "2nG", "setup ms", "rt_solve ms", "total ms", "R");

    for (int nG : sizes) {
        double best_setup = 1e300, best_rt = 1e300;
        int nG_out = 0;
        double R = 0.0;
        for (int r = 0; r < runs; ++r) {
            RCWAConfig cfg;
            cfg.nG = nG;
            cfg.L1 = Eigen::Vector2d(Lx, 0.0);
            cfg.L2 = Eigen::Vector2d(0.0, Ly);
            cfg.freq = complex(1.0 / lambda, 0.0);
            cfg.theta = theta_deg * M_PI / 180.0;
            cfg.phi = phi_deg * M_PI / 180.0;
            cfg.quasi1d = quasi1d;

            RCWA solver(cfg);
            solver.Add_LayerUniform(1.0, eps_from_n(complex(1.0, 0.0)));
            solver.Add_LayerGrid(abs_t, Nx, Ny);
            solver.Add_LayerUniform(2.5, eps_from_n(n_ru));
            for (int i = 0; i < nb; ++i) {
                solver.Add_LayerUniform(2.8, eps_from_n(n_mo));
                solver.Add_LayerUniform(4.2, eps_from_n(n_si));
            }
            solver.Add_LayerUniform(1.0, eps_from_n(n_si));

            auto t0 = std::chrono::steady_clock::now();
            solver.Init_Setup();
            solver.GridLayer_geteps(epgrid);
            auto t1 = std::chrono::steady_clock::now();

            PlaneWaveExcitation exc;
            exc.p_amp = 1.0;
            solver.MakeExcitationPlanewave(exc);
            RTResult rt = solver.RT_Solve(/*normalize=*/true);
            auto t2 = std::chrono::steady_clock::now();

            best_setup = std::min(best_setup, ms(t1 - t0).count());
            best_rt = std::min(best_rt, ms(t2 - t1).count());
            nG_out = solver.nG();
            R = rt.R;
        }
        std::printf("%6d %7d %12.1f %12.1f %12.1f %9.6f\n",
                    nG_out, 2 * nG_out, best_setup, best_rt,
                    best_setup + best_rt, R);
        std::fflush(stdout);
    }
    return 0;
}
