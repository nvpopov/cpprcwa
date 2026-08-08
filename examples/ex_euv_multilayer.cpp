// EUV multilayer mirror reflectivity (Mo/Si multilayer with Ru cap).
//
// Structure (λ = 13.5 nm design):
//   Vacuum (incident)
//   Ru cap:            2.5 nm
//   N × [Mo 2.8 nm, Si 4.2 nm]     (bilayer period d = 7.0 nm, Γ ≈ 0.4)
//   Substrate: Si (or SiO2 glass)
//
// Material permittivities at 13.5 nm: ε = n², n = 1−δ + iβ
//   Mo: 0.9226 + 0.0064i     Si: 0.9997 + 0.0018i
//   Ru: 0.9114 + 0.0171i     SiO2: 0.9782 + 0.0107i
//
// Only the specular (G=0) order propagates for a planar stack, so the
// truncated reciprocal lattice reduces to a single order.
//
// Usage:
//   ex_euv_multilayer [nb] [lambda_nm] [substrate=si|sio2]   single wavelength
//   ex_euv_multilayer --scan [nb] [lmin] [lmax] [dlam]       wavelength scan
#include <cpprcwa/cpprcwa.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

using cpprcwa::complex;

namespace {

complex eps_from_n(complex n) { return n * n; }

const complex n_mo  (0.9226, 0.0064);
const complex n_si  (0.9997, 0.0018);
const complex n_ru  (0.9114, 0.0171);
const complex n_sio2(0.9782, 0.0107);

// Build the multilayer and return the normalized reflectivity at `lam` nm.
double reflectivity(int nb, double lam, const complex& n_sub) {
    using namespace cpprcwa;
    RCWAConfig cfg;
    cfg.nG = 2;                                    // → nG_out = 1 (specular only)
    cfg.L1 = Eigen::Vector2d(1.0, 0.0);
    cfg.L2 = Eigen::Vector2d(0.0, 1.0);
    cfg.freq = complex(1.0 / lam, 0.0);            // freq·t = t/λ
    cfg.theta = 0.0;
    cfg.phi   = 0.0;

    RCWA solver(cfg);
    solver.Add_LayerUniform(1.0, eps_from_n(complex(1.0, 0.0)));   // incident vacuum
    solver.Add_LayerUniform(2.5, eps_from_n(n_ru));                // Ru cap
    for (int i = 0; i < nb; ++i) {
        solver.Add_LayerUniform(2.8, eps_from_n(n_mo));
        solver.Add_LayerUniform(4.2, eps_from_n(n_si));
    }
    solver.Add_LayerUniform(1.0, eps_from_n(n_sub));               // substrate

    solver.Init_Setup();
    PlaneWaveExcitation exc;
    exc.p_amp = 1.0;
    solver.MakeExcitationPlanewave(exc);
    return solver.RT_Solve(/*normalize=*/true).R;
}

} // namespace

int main(int argc, char** argv) {
    using namespace cpprcwa;

    bool scan = argc > 1 && std::strcmp(argv[1], "--scan") == 0;
    int  nb   = scan ? (argc > 2 ? std::atoi(argv[2]) : 40)
                     : (argc > 1 ? std::atoi(argv[1]) : 40);
    if (scan) {
        double lmin = argc > 3 ? std::atof(argv[3]) : 12.5;
        double lmax = argc > 4 ? std::atof(argv[4]) : 14.5;
        double dlam = argc > 5 ? std::atof(argv[5]) : 0.1;
        const complex n_sub = n_si;
        std::printf("EUV multilayer scan: nb=%d, lmin=%.2f, lmax=%.2f, dlam=%.2f nm\n",
                    nb, lmin, lmax, dlam);
        std::printf("lambda[nm]\tR\n");
        double best_l = lmin, best_R = -1.0;
        for (double lam = lmin; lam <= lmax + 1e-9; lam += dlam) {
            double R = reflectivity(nb, lam, n_sub);
            std::printf("%.2f\t%.6f\n", lam, R);
            if (R > best_R) { best_R = R; best_l = lam; }
        }
        std::printf("peak: R=%.6f at lambda=%.2f nm\n", best_R, best_l);
        return 0;
    }

    double lam = argc > 2 ? std::atof(argv[2]) : 13.5;
    std::string sub = argc > 3 ? argv[3] : "si";
    const complex n_sub = (sub == "sio2") ? n_sio2 : n_si;

    double R = reflectivity(nb, lam, n_sub);
    std::printf("EUV multilayer mirror: nb=%d, lambda=%.2f nm, period=%.2f nm, substrate=%s\n",
                nb, lam, 7.0, sub.c_str());
    std::printf("Reflectivity R = %.6f\n", R);
    return 0;
}
