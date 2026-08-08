// Port of example/ex2.py — two patterned layers at oblique incidence.
#include <cpprcwa/cpprcwa.h>
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    using namespace cpprcwa;

    Eigen::Vector2d L1(0.2, 0.0), L2(0.0, 0.2);
    int nG = 101;
    double freq = 1.0, theta = M_PI / 10, phi = 0.0;

    const int Np = 2;
    const int Nx = 100, Ny = 100;
    double thick0 = 0.1, thickN = 0.4;
    double pthick[2] = {0.2, 0.3};
    complex ep0(2.0, 0.0), epN(3.0, 0.0);

    RCWAConfig cfg;
    cfg.nG = nG;
    cfg.L1 = L1;
    cfg.L2 = L2;
    cfg.freq = complex(freq, 0.0);
    cfg.theta = theta;
    cfg.phi = phi;

    RCWA solver(cfg);
    solver.Add_LayerUniform(thick0, ep0);
    for (int i = 0; i < Np; ++i)
        solver.Add_LayerGrid(pthick[i], Nx, Ny);
    solver.Add_LayerUniform(thickN, epN);
    solver.Init_Setup();
    std::printf("Total nG = %d\n", solver.nG());

    double radius = 0.5, a = 0.5;
    complex ep1(4.0, 0.0), ep2(6.0, 0.0), epbkg(1.0, 0.0);

    // Layer 1: circle
    std::vector<complex> epgrid1((size_t)Nx * Ny, ep1);
    for (int i = 0; i < Nx; ++i) {
        double x = (double)i / (Nx - 1);
        for (int j = 0; j < Ny; ++j) {
            double y = (double)j / (Ny - 1);
            if ((x - 0.5)*(x - 0.5) + (y - 0.5)*(y - 0.5) < radius*radius)
                epgrid1[(size_t)i * Ny + j] = epbkg;
        }
    }
    // Layer 2: square
    std::vector<complex> epgrid2((size_t)Nx * Ny, ep2);
    for (int i = 0; i < Nx; ++i) {
        double x = (double)i / (Nx - 1);
        for (int j = 0; j < Ny; ++j) {
            double y = (double)j / (Ny - 1);
            if (std::abs(x - 0.5) < a/2 && std::abs(y - 0.5) < a/2)
                epgrid2[(size_t)i * Ny + j] = epbkg;
        }
    }

    std::vector<complex> ep_all(epgrid1.begin(), epgrid1.end());
    ep_all.insert(ep_all.end(), epgrid2.begin(), epgrid2.end());
    solver.GridLayer_geteps(ep_all);

    PlaneWaveExcitation exc;
    exc.p_amp = 0.0;
    exc.s_amp = 1.0;
    solver.MakeExcitationPlanewave(exc);

    RTResult rt = solver.RT_Solve(/*normalize=*/true);
    std::printf("R=%f, T=%f, R+T=%f\n", rt.R, rt.T, rt.R + rt.T);
    return 0;
}
