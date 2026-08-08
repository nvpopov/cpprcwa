// Port of example/ex1.py — square lattice of holes.
#include <cpprcwa/cpprcwa.h>
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    using namespace cpprcwa;

    // Truncation order (actual number may be smaller)
    int nG = 301;
    Eigen::Vector2d L1(1.5, 0.0), L2(0.0, 1.5);
    double freq = 1.0, theta = 0.0, phi = 0.0;
    double Qabs = std::numeric_limits<double>::infinity();
    complex freqcmp(freq * (1.0 + 1.0 / (2.0 * Qabs)), 0.0);
    int Nx = 400, Ny = 400;

    double thick0 = 1.0, thickp = 0.2, thickN = 1.0;
    complex ep0(1.0, 0.0), epp(4.0, 0.0), epbkg(1.0, 0.0), epN(1.0, 0.0);

    double radius = 0.3;
    std::vector<complex> epgrid((size_t)Nx * Ny, epp);
    for (int i = 0; i < Nx; ++i) {
        double x = (double)i / (Nx - 1);
        for (int j = 0; j < Ny; ++j) {
            double y = (double)j / (Ny - 1);
            double dx = x - 0.5, dy = y - 0.5;
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
