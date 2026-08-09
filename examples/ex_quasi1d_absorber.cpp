// Quasi-1D EUV absorber: a 1D line grating on top of a Mo/Si multilayer mirror.
//
// Same EUV mask stack as ex_euv_absorber, but the TaN absorber is a line/bar
// that is *uniform in y* (quasi-1D geometry): the periodic direction is x only.
//
//   Vacuum (incident)
//   TaN absorber bar:   Lx=3.5 um period, 500 nm wide (x), spans full y cell
//   Ru cap:             2.5 nm
//   40× [Mo 2.8 nm, Si 4.2 nm]
//   Substrate: Si
//
// Material constants at λ = 13.5 nm (ε = n²):
//   Mo: 0.9226+0.0064i  Si: 0.9997+0.0018i  Ru: 0.9114+0.0171i
//   TaN: 0.9562+0.0323i
//
// The cell is Lx × Ly with Ly = nly·λ ("several wavelengths"). Because the
// structure is y-invariant, all y≠0 Fourier coefficients vanish and the y≠0
// diffraction orders are never excited: the harmonic set reduces to the 1D
// row (i, 0) as long as nG stays below the first y-harmonic radius. The
// example prints the propagating x-order diffraction efficiencies (the fan
// of reflected orders of the line grating).
//
// Usage:
//   ex_quasi1d_absorber [nG] [abs_thickness_nm] [nb] [nly] [Nx]
//   ex_quasi1d_absorber ... --field OUT       # also write reflected-field data:
//                                             #   OUT_orders.txt  per-order refl. field
//                                             #   OUT_effic.txt   per-order efficiency+angle
//                                             #   OUT_grid.txt    real-space |E|² grid
//                                             #   OUT_vscan.txt   vertical cross-section (x sweep)
//                                             #   OUT_perf.txt
//   ex_quasi1d_absorber --quasi1d ...         # restrict to x-only harmonics
//   ex_quasi1d_absorber --theta DEG --phi DEG # oblique incidence
//   ex_quasi1d_absorber --pol p|s             # polarization (default p)
#include <cpprcwa/cpprcwa.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>

using cpprcwa::complex;

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

const double Lx = 3500.0;   // grating period / cell size in x (nm)
const double wx = 500.0;    // absorber bar width in x (nm)

} // namespace

int main(int argc, char** argv) {
    using namespace cpprcwa;

    // Positionals and --flags may appear in any order.
    std::vector<double> pos;
    std::string  out_prefix;
    int threads = 0;   // 0 = auto
    bool quasi1d = false;   // restrict to x-only harmonics (exact: y-invariant bar)
    double theta_deg = 0.0;
    double phi_deg   = 0.0;
    char pol = 'p';        // 'p' or 's' polarization
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--field") == 0 && i + 1 < argc) out_prefix = argv[++i];
        else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--quasi1d") == 0) quasi1d = true;
        else if (std::strcmp(argv[i], "--theta") == 0 && i + 1 < argc) theta_deg = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--phi") == 0 && i + 1 < argc) phi_deg = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--pol") == 0 && i + 1 < argc) pol = argv[++i][0];
        else pos.push_back(std::atof(argv[i]));
    }
    const int    nG_req = pos.size() > 0 ? (int)pos[0] : 201;
    const double abs_t  = pos.size() > 1 ? pos[1] : 60.0;
    const int    nb     = pos.size() > 2 ? (int)pos[2] : 40;
    const double nly    = pos.size() > 3 ? pos[3] : 5.0;  // Ly = nly·λ
    const double lambda = pos.size() > 4 ? pos[4] : 13.5;
    const int    Nx     = pos.size() > 5 ? (int)pos[5] : 1000;
    const int    Ny     = 4;
    if (threads > 0) openblas_set_num_threads(threads);
    else openblas_set_num_threads(blas_threads::choose((int)std::thread::hardware_concurrency()));

    const double Ly = nly * lambda;

    // Patterned layer: TaN inside the 500 nm bar (full y span), vacuum elsewhere.
    std::vector<complex> epgrid((size_t)Nx * Ny, n_vac);
    const double half = wx / (2.0 * Lx);
    for (int i = 0; i < Nx; ++i) {
        double u = (double)i / (Nx - 1);
        if (std::fabs(u - 0.5) >= half) continue;
        for (int j = 0; j < Ny; ++j)
            epgrid[(size_t)i * Ny + j] = n_tan;
    }

    RCWAConfig cfg;
    cfg.nG = nG_req;
    cfg.L1 = Eigen::Vector2d(Lx, 0.0);
    cfg.L2 = Eigen::Vector2d(0.0, Ly);
    cfg.freq = complex(1.0 / lambda, 0.0);
    cfg.theta = theta_deg * M_PI / 180.0;
    cfg.phi   = phi_deg   * M_PI / 180.0;
    cfg.quasi1d = quasi1d;

    RCWA solver(cfg);
    solver.Add_LayerUniform(1.0, eps_from_n(n_vac));      // incident vacuum
    solver.Add_LayerGrid(abs_t, Nx, Ny);                  // TaN line grating
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
    if (pol == 's') exc.s_amp = 1.0;
    else            exc.p_amp = 1.0;
    solver.MakeExcitationPlanewave(exc);

    RTResult rt = solver.RT_Solve(/*normalize=*/true);
    auto t2 = std::chrono::steady_clock::now();
    using ms = std::chrono::duration<double, std::milli>;
    const double setup_ms = ms(t1 - t0).count();
    const double solve_ms = ms(t2 - t1).count();

    // ── Harmonic-set characterization (quasi-1D demonstration) ──
    const IntMatrix& G = solver.G();
    const int nG = solver.nG();
    int nx_only = 0;
    for (int i = 0; i < nG; ++i)
        if (G(i, 1) == 0) ++nx_only;

    const double fill = wx / Lx;
    std::printf("Quasi-1D EUV absorber: nG=%d (x-only %d)%s, cell %g x %g nm "
                "(%g um x %.1f nm = %.1f lambda), t_abs=%.1f nm, nb=%d, lambda=%.2f nm\n",
                nG, nx_only, quasi1d ? " [1D harmonic set]" : "", Lx, Ly,
                Lx * 1e-3, Ly, nly, abs_t, nb, lambda);
    std::printf("absorber line duty cycle = %.1f%%\n", 100.0 * fill);
    std::printf("Reflectivity R = %.6f\n", rt.R);
    std::printf("Transmission T = %.6g\n", rt.T);
    std::printf("R + T = %.6f  (rest absorbed in TaN/Mo/Si)\n", rt.R + rt.T);
    std::printf("perf_ms: setup=%.1f rt_solve=%.1f total=%.1f\n",
                setup_ms, solve_ms, setup_ms + solve_ms);

    // ── Propagating reflected x-orders + efficiencies ──
    FieldFourier refl = solver.BackwardPropagatedFieldFourier(0, 0.0);
    const ComplexVector& kx = solver.kx();
    const ComplexVector& ky = solver.ky();
    const double k0 = 2.0 * M_PI / lambda;
    double total_flux = 0.0;
    std::vector<double> flux(nG);
    for (int i = 0; i < nG; ++i) {
        // Per-order reflected z-flux: 0.5 Re(conj(Ex)·Hy - conj(Ey)·Hx).
        flux[i] = 0.5 * (std::conj(refl.ex(i)) * refl.hy(i)
                         - std::conj(refl.ey(i)) * refl.hx(i)).real();
        total_flux += flux[i];
    }
    std::printf("propagating reflected x-orders (eff = R_m, angle from normal):\n");
    int nprop = 0;
    for (int i = 0; i < nG; ++i) {
        double kx_n = (kx(i) / complex(k0, 0.0)).real();
        double ky_n = (ky(i) / complex(k0, 0.0)).real();
        double sint2 = kx_n * kx_n + ky_n * ky_n;
        if (sint2 >= 1.0) continue;   // evanescent
        double eff = std::fabs(total_flux) > 0.0 ? flux[i] * (rt.R / total_flux) : 0.0;
        double deg = std::asin(std::sqrt(sint2)) * 180.0 / M_PI;
        std::printf("  m=%4d  (i=%3d, j=%2d)  theta=%.4f deg  eff=%.6g\n",
                    i, G(i, 0), G(i, 1), deg, eff);
        ++nprop;
    }
    std::printf("  %d propagating reflected orders (of nG=%d)\n", nprop, nG);

    if (!out_prefix.empty()) {
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

        std::string eff_path = out_prefix + "_effic.txt";
        std::FILE* fe = std::fopen(eff_path.c_str(), "w");
        for (int i = 0; i < nG; ++i) {
            double kx_n = (kx(i) / complex(k0, 0.0)).real();
            double ky_n = (ky(i) / complex(k0, 0.0)).real();
            double sint2 = kx_n * kx_n + ky_n * ky_n;
            double deg = sint2 < 1.0 ? std::asin(std::sqrt(sint2)) * 180.0 / M_PI : -1.0;
            double eff = std::fabs(total_flux) > 0.0 ? flux[i] * (rt.R / total_flux) : 0.0;
            std::fprintf(fe, "%d %d %d %.6f %.17g\n", i, G(i,0), G(i,1), deg, eff);
        }
        std::fclose(fe);
        std::printf("wrote %s\n", eff_path.c_str());

        // Real-space reflected field on an (Nx, Ny) grid.
        double dN = 1.0 / ((double)Nx * Ny);
        GridMatrix Ex = get_ifft(dN, Nx, Ny, refl.ex, G);
        GridMatrix Ey = get_ifft(dN, Nx, Ny, refl.ey, G);
        GridMatrix Ez = get_ifft(dN, Nx, Ny, refl.ez, G);

        std::string grid_path = out_prefix + "_grid.txt";
        std::FILE* fg = std::fopen(grid_path.c_str(), "w");
        std::fprintf(fg, "# Nx=%d Ny=%d Lx=%.1f Ly=%.1f lambda=%.2f\n", Nx, Ny, Lx, Ly, lambda);
        std::fprintf(fg, "# i j x_nm y_nm |E|^2 Re(Ey)\n");
        std::vector<double> vscan((size_t)Nx);
        const int jc = Ny / 2;
        for (int i = 0; i < Nx; ++i) {
            double x_nm = (double)i / (Nx - 1) * Lx;
            for (int j = 0; j < Ny; ++j) {
                double y_nm = (double)j / (Ny - 1) * Ly;
                double absE2 = std::norm(Ex(i,j)) + std::norm(Ey(i,j)) + std::norm(Ez(i,j));
                std::fprintf(fg, "%d %d %.6f %.6f %.17g %.17g\n",
                             i, j, x_nm, y_nm, absE2, Ey(i,j).real());
                if (j == jc) vscan[(size_t)i] = absE2;   // x sweep at y=center
            }
        }
        std::fclose(fg);
        std::printf("wrote %s (%dx%d)\n", grid_path.c_str(), Nx, Ny);

        std::string scan_path = out_prefix + "_vscan.txt";
        std::FILE* fs = std::fopen(scan_path.c_str(), "w");
        for (size_t k = 0; k < vscan.size(); ++k) {
            double pos = (double)k / (vscan.size() - 1) * Lx;
            std::fprintf(fs, "%.6f %.17g\n", pos, vscan[k]);
        }
        std::fclose(fs);
        std::printf("wrote %s (x sweep through center)\n", scan_path.c_str());

        std::string perf_path = out_prefix + "_perf.txt";
        std::FILE* fp = std::fopen(perf_path.c_str(), "w");
        std::fprintf(fp, "setup_ms %.3f\nrt_solve_ms %.3f\ntotal_ms %.3f\n",
                     setup_ms, solve_ms, setup_ms + solve_ms);
        std::fclose(fp);
        std::printf("wrote %s\n", perf_path.c_str());
    }
    return 0;
}
