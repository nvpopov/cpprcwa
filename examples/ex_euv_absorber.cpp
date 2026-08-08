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
#include <cpprcwa/cpprcwa.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using cpprcwa::complex;

namespace {

complex eps_from_n(complex n) { return n * n; }

const complex n_vac(1.0, 0.0);
const complex n_mo (0.9226, 0.0064);
const complex n_si (0.9997, 0.0018);
const complex n_ru (0.9114, 0.0171);
const complex n_tan(0.9562, 0.0323);

} // namespace

int main(int argc, char** argv) {
    using namespace cpprcwa;

    const int    nG_req   = argc > 1 ? std::atoi(argv[1]) : 101;
    const double abs_t    = argc > 2 ? std::atof(argv[2]) : 60.0;  // absorber height nm
    const int    nb       = argc > 3 ? std::atoi(argv[3]) : 40;
    const double lambda   = argc > 4 ? std::atof(argv[4]) : 13.5;
    const int    Nx       = argc > 5 ? std::atoi(argv[5]) : 300;
    const int    Ny       = Nx;

    const double cell  = 300.0;   // lattice period / cell size (nm)
    const double rect  = 60.0;    // absorber rectangle side (nm)

    // Patterned layer: TaN inside the 60×60 nm rectangle, vacuum elsewhere.
    // Grid coordinate u = i/(Nx-1), so the rectangle is |u-0.5| < rect/(2·cell).
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
    solver.Init_Setup();
    solver.GridLayer_geteps(epgrid);

    PlaneWaveExcitation exc;
    exc.p_amp = 1.0;
    solver.MakeExcitationPlanewave(exc);

    RTResult rt = solver.RT_Solve(/*normalize=*/true);

    const double fill = (rect * rect) / (cell * cell);    // absorber areal fraction
    std::printf("EUV absorber on mirror: nG=%d, absorber %gx%g nm in %g nm cell, "
                "t_abs=%.1f nm, nb=%d, lambda=%.2f nm\n",
                solver.nG(), rect, rect, cell, abs_t, nb, lambda);
    std::printf("absorber areal fill factor = %.1f%%\n", 100.0 * fill);
    std::printf("Reflectivity R = %.6f\n", rt.R);
    std::printf("Transmission T = %.6g\n", rt.T);
    std::printf("R + T = %.6f  (rest absorbed in TaN/Mo/Si)\n", rt.R + rt.T);
    return 0;
}
