// Port of example/ex4.py — hexagonal lattice of a hole (non-orthogonal coords).
#include <cpprcwa/cpprcwa.h>
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    using namespace cpprcwa;

    int nG = 101;
    double angle = M_PI / 3;
    Eigen::Vector2d L1(0.1, 0.0);
    Eigen::Vector2d L2(0.1 * std::cos(angle), 0.1 * std::sin(angle));
    double freq = 1.0, theta = 0.0, phi = 0.0;
    double Qabs = std::numeric_limits<double>::infinity();
    complex freqcmp(freq * (1.0 + 1.0 / (2.0 * Qabs)), 0.0);
    int Nx = 1000, Ny = 1000;

    complex ep0(1.0, 0.0), epp(4.0, 0.0), epbkg(1.0, 0.0), epN(1.0, 0.0);
    double thick0 = 1.0, thickp = 0.4, thickN = 1.0;

    // eps grid in non-orthogonal (u, v) coords, transformed to Cartesian.
    std::vector<complex> epgrid((size_t)Nx * Ny, epp);
    double radius = 0.3;
    double uc0 = 0.5, vc0 = 0.5;
    double xc0 = uc0 + vc0 * std::cos(angle);
    double yc0 = vc0 * std::sin(angle);
    for (int i = 0; i < Nx; ++i) {
        double u = (double)i / (Nx - 1);
        for (int j = 0; j < Ny; ++j) {
            double v = (double)j / (Ny - 1);
            double x = u + v * std::cos(angle);
            double y = v * std::sin(angle);
            double dx = x - xc0, dy = y - yc0;
            if (dx*dx + dy*dy < radius*radius)
                epgrid[(size_t)i * Ny + j] = epbkg;
        }
    }

    RCWAConfig cfg;
    cfg.nG = nG;
    cfg.L1 = L1;
    cfg.L2 = L2;
    cfg.freq = freqcmp;
    cfg.theta = theta;
    cfg.phi = phi;

    RCWA solver(cfg);
    solver.Add_LayerUniform(thick0, ep0);
    solver.Add_LayerGrid(thickp, Nx, Ny);
    solver.Add_LayerUniform(thickN, epN);
    solver.Init_Setup();
    std::printf("Total nG = %d\n", solver.nG());

    PlaneWaveExcitation exc;
    exc.p_amp = 1.0;
    exc.s_amp = 0.0;
    solver.MakeExcitationPlanewave(exc);
    solver.GridLayer_geteps(epgrid);

    RTResult rt = solver.RT_Solve(/*normalize=*/true);
    std::printf("R=%f, T=%f, R+T=%f\n", rt.R, rt.T, rt.R + rt.T);
    return 0;
}
