// EUV absorber pattern on top of a Mo/Si multilayer mirror.
//
// EUV mask geometry: a periodic array of TaN absorber rectangles sits on top
// of the multilayer mirror (Ru cap + Mo/Si bilayers + substrate).
//
//   Vacuum (incident)
//   TaN absorber pattern:     300 nm period, 60×60 nm rectangle, n=0.9562+0.0323i
//   Ru cap:                   2.5 nm
//   40× [Mo 2.8 nm, Si 4.2 nm]
//   Substrate: Si
//
// Material constants at λ = 13.5 nm (ε = n²):
//   Mo: 0.9226+0.0064i  Si: 0.9997+0.0018i  Ru: 0.9114+0.0171i
//   TaN: 0.9562+0.0323i
//
// Usage:
//   ex_euv_absorber [nG] [abs_thickness_nm] [nb] [lambda_nm] [Nx]
//   ex_euv_absorber ... --field OUT       # also write reflected-field data:
//                                         #   OUT_orders.txt  per-order refl. field (ex,ey,ez)
//                                         #   OUT_grid.txt    real-space |E|² grid
//                                         #   OUT_hscan.txt   horizontal cross-section (y=center)
//                                         #   OUT_vscan.txt   vertical cross-section (x=center)
#include <cpprcwa/cpprcwa.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>

using cpprcwa::complex;

// OpenBLAS runtime thread control (guarded; only linked when BLAS is OpenBLAS).
// Many small matrices (2nG×2nG) saturate at a modest thread count; using all
// logical cores (incl. Hyper-Threading siblings) causes thread-pool overhead.
// Defaults to the physical core count, overridable via the
// OPENBLAS_NUM_THREADS environment variable or the --threads argument.
extern "C" {
void openblas_set_num_threads(int);
}
#include "blas_threads.h"

namespace {

complex eps_from_n(complex n) { return n * n; }

const complex n_vac(1.0, 0.0);
const complex n_mo (0.9226, 0.0064);
const complex n_si (0.9997, 0.0018);
const complex n_ru (0.9114, 0.0171);
const complex n_tan(0.9562, 0.0323);

const double cell = 300.0;   // lattice period / cell size (nm)
const double rect = 60.0;    // absorber rectangle side (nm)

} // namespace

int main(int argc, char** argv) {
    using namespace cpprcwa;

    // Parse positionals + optional "--field OUT"
    const int    nG_req = argc > 1 ? std::atoi(argv[1]) : 101;
    const double abs_t  = argc > 2 ? std::atof(argv[2]) : 60.0;
    const int    nb     = argc > 3 ? std::atoi(argv[3]) : 40;
    const double lambda = argc > 4 ? std::atof(argv[4]) : 13.5;
    const int    Nx     = argc > 5 ? std::atoi(argv[5]) : 300;
    const int    Ny     = Nx;
    std::string  out_prefix;
    int threads = 0;   // 0 = auto
    for (int i = 6; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], "--field") == 0) out_prefix = argv[i + 1];
        if (std::strcmp(argv[i], "--threads") == 0) threads = std::atoi(argv[i + 1]);
    }
    if (threads > 0) openblas_set_num_threads(threads);
    else openblas_set_num_threads(blas_threads::choose((int)std::thread::hardware_concurrency()));

    // Patterned layer: TaN inside the 60×60 nm rectangle, vacuum elsewhere.
    std::vector<complex> epgrid((size_t)Nx * Ny, n_vac);
    const double half = rect / (2.0 * cell);
    for (int i = 0; i < Nx; ++i) {
        double u = (double)i / (Nx - 1);
        for (int j = 0; j < Ny; ++j) {
            double v = (double)j / (Ny - 1);
            if (std::fabs(u - 0.5) < half && std::fabs(v - 0.5) < half)
                epgrid[(size_t)i * Ny + j] = n_tan;
        }
    }

    RCWAConfig cfg;
    cfg.nG = nG_req;
    cfg.L1 = Eigen::Vector2d(cell, 0.0);
    cfg.L2 = Eigen::Vector2d(0.0, cell);
    cfg.freq = complex(1.0 / lambda, 0.0);
    cfg.theta = 0.0;
    cfg.phi   = 0.0;

    RCWA solver(cfg);
    solver.Add_LayerUniform(1.0, eps_from_n(n_vac));      // incident vacuum
    solver.Add_LayerGrid(abs_t, Nx, Ny);                  // TaN absorber pattern
    solver.Add_LayerUniform(2.5, eps_from_n(n_ru));       // Ru cap
    for (int i = 0; i < nb; ++i) {
        solver.Add_LayerUniform(2.8, eps_from_n(n_mo));
        solver.Add_LayerUniform(4.2, eps_from_n(n_si));
    }
    solver.Add_LayerUniform(1.0, eps_from_n(n_si));       // substrate

    auto t0 = std::chrono::steady_clock::now();
    solver.Init_Setup();
    solver.GridLayer_geteps(epgrid);
    auto t1 = std::chrono::steady_clock::now();

    PlaneWaveExcitation exc;
    exc.p_amp = 1.0;
    solver.MakeExcitationPlanewave(exc);

    RTResult rt = solver.RT_Solve(/*normalize=*/true);
    auto t2 = std::chrono::steady_clock::now();
    using ms = std::chrono::duration<double, std::milli>;
    const double setup_ms = ms(t1 - t0).count();   // Init_Setup + eps/eig
    const double solve_ms = ms(t2 - t1).count();   // RT_Solve (S-matrix + flux)

    const double fill = (rect * rect) / (cell * cell);
    std::printf("EUV absorber on mirror: nG=%d, absorber %gx%g nm in %g nm cell, "
                "t_abs=%.1f nm, nb=%d, lambda=%.2f nm\n",
                solver.nG(), rect, rect, cell, abs_t, nb, lambda);
    std::printf("absorber areal fill factor = %.1f%%\n", 100.0 * fill);
    std::printf("Reflectivity R = %.6f\n", rt.R);
    std::printf("Transmission T = %.6g\n", rt.T);
    std::printf("R + T = %.6f  (rest absorbed in TaN/Mo/Si)\n", rt.R + rt.T);
    std::printf("perf_ms: setup=%.1f rt_solve=%.1f total=%.1f\n",
                setup_ms, solve_ms, setup_ms + solve_ms);

    if (!out_prefix.empty()) {
        // ── Backward-propagating field in layer 0 (air) at z=0 = reflected field ──
        FieldFourier refl = solver.BackwardPropagatedFieldFourier(0, 0.0);
        const IntMatrix& G = solver.G();
        const int nG = solver.nG();

        // Per-order reflected-field coefficients.
        std::string orders_path = out_prefix + "_orders.txt";
        std::FILE* fo = std::fopen(orders_path.c_str(), "w");
        for (int i = 0; i < nG; ++i)
            std::fprintf(fo, "%d %d %d %.17g %.17g %.17g %.17g %.17g %.17g\n",
                         i, G(i,0), G(i,1),
                         refl.ex(i).real(), refl.ex(i).imag(),
                         refl.ey(i).real(), refl.ey(i).imag(),
                         refl.ez(i).real(), refl.ez(i).imag());
        std::fclose(fo);
        std::printf("wrote %s (%d orders)\n", orders_path.c_str(), nG);

        // Real-space reflected field on the (Nx, Ny) grid.
        double dN = 1.0 / ((double)Nx * Ny);
        GridMatrix Ex = get_ifft(dN, Nx, Ny, refl.ex, G);
        GridMatrix Ey = get_ifft(dN, Nx, Ny, refl.ey, G);
        GridMatrix Ez = get_ifft(dN, Nx, Ny, refl.ez, G);

        std::string grid_path = out_prefix + "_grid.txt";
        std::FILE* fg = std::fopen(grid_path.c_str(), "w");
        std::fprintf(fg, "# Nx=%d Ny=%d cell=%.1f lambda=%.2f\n", Nx, Ny, cell, lambda);
        std::fprintf(fg, "# i j x_nm y_nm |E|^2 Re(Ey)\n");
        std::vector<double> hscan((size_t)Ny), vscan((size_t)Nx);
        const int ic = Nx / 2, jc = Ny / 2;
        for (int i = 0; i < Nx; ++i) {
            double x_nm = (double)i / (Nx - 1) * cell;
            for (int j = 0; j < Ny; ++j) {
                double y_nm = (double)j / (Ny - 1) * cell;
                double absE2 = (std::norm(Ex(i,j)) + std::norm(Ey(i,j)) + std::norm(Ez(i,j)));
                std::fprintf(fg, "%d %d %.6f %.6f %.17g %.17g\n",
                             i, j, x_nm, y_nm, absE2, Ey(i,j).real());
                if (i == ic) hscan[(size_t)j] = absE2;   // horizontal (y sweep at x=center)
                if (j == jc) vscan[(size_t)i] = absE2;   // vertical   (x sweep at y=center)
            }
        }
        std::fclose(fg);
        std::printf("wrote %s (%dx%d)\n", grid_path.c_str(), Nx, Ny);

        auto write_scan = [&](const std::string& path, const std::vector<double>& data,
                              bool horizontal) {
            std::FILE* f = std::fopen(path.c_str(), "w");
            for (size_t k = 0; k < data.size(); ++k) {
                double pos = (double)k / (data.size() - 1) * cell;
                std::fprintf(f, "%.6f %.17g\n", pos, data[k]);
            }
            std::fclose(f);
            std::printf("wrote %s (%s through center)\n", path.c_str(),
                        horizontal ? "horizontal" : "vertical");
        };
        write_scan(out_prefix + "_hscan.txt", hscan, true);
        write_scan(out_prefix + "_vscan.txt", vscan, false);

        // Full complex reflected field along both cross-section lines:
        //   pos |E|^2 Re(ex) Im(ex) Re(ey) Im(ey) Re(ez) Im(ez)
        // horizontal: x = center, y varies; vertical: y = center, x varies.
        auto write_field_scan = [&](const std::string& path, bool horizontal) {
            std::FILE* f = std::fopen(path.c_str(), "w");
            const int n = horizontal ? Ny : Nx;
            for (int k = 0; k < n; ++k) {
                double pos = (double)k / (n - 1) * cell;
                int ii = horizontal ? ic : k;
                int jj = horizontal ? k  : jc;
                complex ex = Ex(ii, jj), ey = Ey(ii, jj), ez = Ez(ii, jj);
                std::fprintf(f, "%.6f %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                             pos, std::norm(ex) + std::norm(ey) + std::norm(ez),
                             ex.real(), ex.imag(), ey.real(), ey.imag(),
                             ez.real(), ez.imag());
            }
            std::fclose(f);
            std::printf("wrote %s (%s through center)\n", path.c_str(),
                        horizontal ? "horizontal" : "vertical");
        };
        write_field_scan(out_prefix + "_hfield.txt", true);
        write_field_scan(out_prefix + "_vfield.txt", false);

        // Performance summary for the comparison script.
        std::string perf_path = out_prefix + "_perf.txt";
        std::FILE* fp = std::fopen(perf_path.c_str(), "w");
        std::fprintf(fp, "setup_ms %.3f\nrt_solve_ms %.3f\ntotal_ms %.3f\n",
                     setup_ms, solve_ms, setup_ms + solve_ms);
        std::fclose(fp);
        std::printf("wrote %s\n", perf_path.c_str());
    }
    return 0;
}
