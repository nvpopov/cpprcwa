// End-to-end RT_Solve timing at a configurable truncation order.
#include <cpprcwa/cpprcwa.h>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <vector>

int main(int argc, char** argv) {
    using namespace cpprcwa;
    int nG = argc > 1 ? std::atoi(argv[1]) : 101;
    int Nx = argc > 2 ? std::atoi(argv[2]) : 200;
    int Ny = Nx;

    std::vector<complex> epgrid((size_t)Nx * Ny, complex(4.0, 0.0));
    double r = 0.3;
    for (int i = 0; i < Nx; ++i) {
        double x = (double)i / (Nx - 1);
        for (int j = 0; j < Ny; ++j) {
            double y = (double)j / (Ny - 1);
            if ((x - 0.5)*(x - 0.5) + (y - 0.5)*(y - 0.5) < r*r)
                epgrid[(size_t)i * Ny + j] = complex(1.0, 0.0);
        }
    }

    RCWAConfig cfg;
    cfg.nG = nG;
    cfg.L1 = Eigen::Vector2d(1.5, 0.0);
    cfg.L2 = Eigen::Vector2d(0.0, 1.5);
    cfg.freq = complex(1.0, 0.0);

    auto t0 = std::chrono::steady_clock::now();
    RCWA solver(cfg);
    solver.Add_LayerUniform(1.0, complex(1.0, 0.0));
    solver.Add_LayerGrid(0.2, Nx, Ny);
    solver.Add_LayerUniform(1.0, complex(1.0, 0.0));
    solver.Init_Setup();
    auto t1 = std::chrono::steady_clock::now();
    solver.GridLayer_geteps(epgrid);
    auto t2 = std::chrono::steady_clock::now();
    PlaneWaveExcitation exc; exc.p_amp = 1.0;
    solver.MakeExcitationPlanewave(exc);
    RTResult rt = solver.RT_Solve(true);
    auto t3 = std::chrono::steady_clock::now();

    using ms = std::chrono::duration<double, std::milli>;
    std::printf("nG=%d setup=%.1fms eps_fft+eig=%.1fms rt_solve=%.1fms  R=%f T=%f\n",
                solver.nG(),
                ms(t1 - t0).count(), ms(t2 - t1).count(), ms(t3 - t2).count(),
                rt.R, rt.T);
    return 0;
}
