#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cpprcwa/rcwa.h>
#include <cpprcwa/errors.h>
#include <cmath>
#include <vector>

using namespace cpprcwa;
using Catch::Approx;

namespace {
std::vector<complex> make_circle_epsgrid(int Nx, int Ny, double radius, complex eps_in) {
    std::vector<complex> g((size_t)Nx * Ny, complex(1.0, 0.0));
    for (int i = 0; i < Nx; ++i) {
        double x = (double)i / (Nx - 1);
        for (int j = 0; j < Ny; ++j) {
            double y = (double)j / (Ny - 1);
            double dx = x - 0.5, dy = y - 0.5;
            if (dx*dx + dy*dy < radius*radius)
                g[(size_t)i * Ny + j] = eps_in;
        }
    }
    return g;
}
} // namespace

// S4 golden reference (grcwa tests/test_rcwa.py:65-75).
TEST_CASE("RCWA S4 golden values (p-pol and s-pol)", "[rcwa][golden]") {
    const int Nx = 100, Ny = 100;
    auto epgrid = make_circle_epsgrid(Nx, Ny, 0.4, complex(12.0, 0.0));

    RCWAConfig cfg;
    cfg.nG = 101;
    cfg.L1 = Eigen::Vector2d(0.1, 0.0);
    cfg.L2 = Eigen::Vector2d(0.0, 0.1);
    cfg.freq = complex(1.0, 0.0);
    cfg.theta = M_PI / 18;
    cfg.phi   = M_PI / 9;

    RCWA solver(cfg);
    solver.Add_LayerUniform(1.0, complex(1.0, 0.0));
    solver.Add_LayerGrid(0.2, Nx, Ny);
    solver.Add_LayerUniform(1.0, complex(1.0, 0.0));
    solver.Init_Setup();
    CHECK(solver.nG() == 97);
    solver.GridLayer_geteps(epgrid);

    // p-polarization: golden T = 0.85249901083265
    PlaneWaveExcitation exc_p;
    exc_p.p_amp = 1.0;
    exc_p.s_amp = 0.0;
    solver.MakeExcitationPlanewave(exc_p);
    RTResult rp = solver.RT_Solve();
    CHECK(std::abs(rp.T - 0.85249901083265) < 1e-3 * 0.85249901083265);

    // s-polarization: golden T = 0.83900479939861
    PlaneWaveExcitation exc_s;
    exc_s.p_amp = 0.0;
    exc_s.s_amp = 1.0;
    solver.MakeExcitationPlanewave(exc_s);
    RTResult rs = solver.RT_Solve();
    CHECK(std::abs(rs.T - 0.83900479939861) < 1e-3 * 0.83900479939861);
}

// Field reconstruction and post-processing on the S4 pattern.
TEST_CASE("RCWA fields and post-processing (S4 pattern)", "[rcwa][fields]") {
    const int Nx = 100, Ny = 100;
    auto epgrid = make_circle_epsgrid(Nx, Ny, 0.4, complex(12.0, 0.0));

    RCWAConfig cfg;
    cfg.nG = 101;
    cfg.L1 = Eigen::Vector2d(0.1, 0.0);
    cfg.L2 = Eigen::Vector2d(0.0, 0.1);
    cfg.freq = complex(1.0, 0.0);
    cfg.theta = M_PI / 18;
    cfg.phi   = M_PI / 9;

    RCWA solver(cfg);
    solver.Add_LayerUniform(1.0, complex(1.0, 0.0));
    solver.Add_LayerGrid(0.2, Nx, Ny);
    solver.Add_LayerUniform(1.0, complex(1.0, 0.0));
    solver.Init_Setup();
    solver.GridLayer_geteps(epgrid);
    PlaneWaveExcitation exc; exc.p_amp = 1.0; exc.s_amp = 0.0;
    solver.MakeExcitationPlanewave(exc);

    // FieldFourier at layer 1, z=0 (validated against Python to ~1e-12).
    auto ff = solver.Solve_FieldFourier(1, 0.0);
    CHECK(std::abs(ff[0].ex.cwiseAbs2().sum() - 0.658607) < 1e-3);
    CHECK(std::abs(ff[0].ey.cwiseAbs2().sum() - 0.111785) < 1e-3);
    CHECK(std::abs(ff[0].ez.cwiseAbs2().sum() - 0.0667863) < 1e-4);
    CHECK(std::abs(ff[0].hx.cwiseAbs2().sum() - 0.206002) < 1e-3);
    CHECK(std::abs(ff[0].hy.cwiseAbs2().sum() - 1.55473) < 1e-3);

    // FieldOnGrid at layer 1, z=0.
    auto fg = solver.Solve_FieldOnGrid(1, 0.0);
    CHECK(fg[0].ex.rows() == Nx);
    CHECK(fg[0].ex.cols() == Ny);
    CHECK(std::abs(fg[0].ex(50,50).real() - 0.313744) < 1e-3);
    CHECK(std::abs(fg[0].ex(50,50).imag() + 0.228369) < 1e-3);

    // Return_eps 'xx'
    auto exx = solver.Return_eps(1, Nx, Ny, "xx");
    CHECK(std::abs(exx(50,50).real() - 9.7295) < 0.05);
    CHECK(std::abs(exx(0,0).real() - 0.338414) < 0.05);

    // Volume_integral with M = real(epinv), must be positive real (Python asserts > 0).
    ComplexMatrix epinv_real = solver.patterned_epinv_list()[0].real().cast<complex>();
    complex val = solver.Volume_integral(1, epinv_real, epinv_real, epinv_real, true);
    CHECK(val.real() > 0.0);
    CHECK(val.real() == Approx(0.196392).epsilon(2e-3));

    // Stress tensor: Tz < 0 for incident wave on absorbing layer (Python asserts Tz<0).
    auto stress = solver.Solve_ZStressTensorIntegral(0);
    CHECK(stress[0] == Approx(-0.139217).epsilon(1e-3));
    CHECK(stress[1] == Approx(-0.0506709).epsilon(1e-3));
    CHECK(stress[2] < 0.0);
}

// Energy conservation: R+T == cos(theta) for lossless uniform stack at oblique incidence.
TEST_CASE("RCWA energy conservation oblique uniform", "[rcwa]") {
    RCWAConfig cfg;
    cfg.nG = 101;
    cfg.L1 = Eigen::Vector2d(0.1, 0.0);
    cfg.L2 = Eigen::Vector2d(0.0, 0.1);
    cfg.freq = complex(1.0, 0.0);
    cfg.theta = M_PI / 18;
    cfg.phi   = M_PI / 9;
    RCWA solver(cfg);
    solver.Add_LayerUniform(1.0, complex(1.0, 0.0));
    solver.Add_LayerUniform(0.2, complex(1.0, 0.0));
    solver.Add_LayerUniform(1.0, complex(1.0, 0.0));
    solver.Init_Setup();
    PlaneWaveExcitation exc; exc.p_amp = 1.0; exc.s_amp = 0.0;
    solver.MakeExcitationPlanewave(exc);
    RTResult rt = solver.RT_Solve();
    CHECK(rt.R == Approx(0.0).margin(1e-10));
    CHECK(rt.T == Approx(std::cos(M_PI/18)).epsilon(1e-6));
}

TEST_CASE("RCWA construct and add layers", "[rcwa]") {
    RCWAConfig cfg;
    cfg.nG = 10;
    cfg.L1 = Eigen::Vector2d(1.5, 0.0);
    cfg.L2 = Eigen::Vector2d(0.0, 1.5);
    cfg.freq = complex(1.0, 0.0);
    cfg.theta = 0.0;
    cfg.phi = 0.0;

    RCWA solver(cfg);
    solver.Add_LayerUniform(0.0, complex(1.0, 0.0));
    solver.Add_LayerUniform(0.5, complex(1.0, 0.0));
    solver.Add_LayerUniform(0.0, complex(1.0, 0.0));
    solver.Init_Setup();

    PlaneWaveExcitation exc;
    exc.p_amp = 1.0;
    exc.s_amp = 0.0;
    solver.MakeExcitationPlanewave(exc);

    // Lossless, uniform, normal incidence: R=0, T=1
    RTResult rt = solver.RT_Solve();
    CHECK(rt.R == Approx(0.0).margin(1e-10));
    CHECK(rt.T == Approx(1.0).margin(1e-9));
}

TEST_CASE("RCWA rejects non-uniform layer 0", "[rcwa]") {
    RCWAConfig cfg;
    cfg.nG = 10;
    cfg.L1 = Eigen::Vector2d(1.5, 0.0);
    cfg.L2 = Eigen::Vector2d(0.0, 1.5);
    cfg.freq = complex(1.0, 0.0);
    RCWA solver(cfg);
    solver.Add_LayerGrid(0.5, 16, 16);
    CHECK_THROWS_AS(solver.Init_Setup(), error::ConfigError);
}

// EUV multilayer mirror: 40× Mo/Si bilayers + Ru cap on Si, λ = 13.5 nm.
// Validated against grcwa (identical to printed precision).
TEST_CASE("RCWA EUV multilayer mirror reflectivity", "[rcwa][euv]") {
    auto eps_from_n = [](complex n) { return n * n; };
    complex n_mo (0.9226, 0.0064);
    complex n_si (0.9997, 0.0018);
    complex n_ru (0.9114, 0.0171);

    const int nb = 40;
    RCWAConfig cfg;
    cfg.nG = 2;                                  // → nG_out = 1 (specular only)
    cfg.L1 = Eigen::Vector2d(1.0, 0.0);
    cfg.L2 = Eigen::Vector2d(0.0, 1.0);
    cfg.freq = complex(1.0 / 13.5, 0.0);
    cfg.theta = 0.0;
    cfg.phi   = 0.0;

    RCWA solver(cfg);
    solver.Add_LayerUniform(1.0, eps_from_n(complex(1.0, 0.0)));
    solver.Add_LayerUniform(2.5, eps_from_n(n_ru));
    for (int i = 0; i < nb; ++i) {
        solver.Add_LayerUniform(2.8, eps_from_n(n_mo));
        solver.Add_LayerUniform(4.2, eps_from_n(n_si));
    }
    solver.Add_LayerUniform(1.0, eps_from_n(n_si));   // substrate
    solver.Init_Setup();
    CHECK(solver.nG() == 1);

    PlaneWaveExcitation exc; exc.p_amp = 1.0;
    solver.MakeExcitationPlanewave(exc);
    RTResult rt = solver.RT_Solve(/*normalize=*/true);

    // grcwa reference: R = 0.599171 at λ = 13.5 nm
    CHECK(rt.R == Approx(0.599171).epsilon(1e-5));
    CHECK(rt.T == Approx(0.0076644).epsilon(1e-3));
}

// EUV absorber pattern (TaN rectangles) on top of the Mo/Si mirror.
// Fast config (nG=13→9, 100×100 grid); validated against grcwa.
TEST_CASE("RCWA EUV absorber on mirror reflectivity", "[rcwa][euv]") {
    auto eps_from_n = [](complex n) { return n * n; };
    complex n_mo (0.9226, 0.0064);
    complex n_si (0.9997, 0.0018);
    complex n_ru (0.9114, 0.0171);
    complex n_tan(0.9562, 0.0323);

    const int Nx = 100, Ny = 100;
    const double cell = 300.0, rect = 60.0, abs_t = 60.0;
    const int nb = 40;
    const double half = rect / (2.0 * cell);

    std::vector<complex> epgrid((size_t)Nx * Ny, complex(1.0, 0.0));
    for (int i = 0; i < Nx; ++i) {
        double u = (double)i / (Nx - 1);
        for (int j = 0; j < Ny; ++j) {
            double v = (double)j / (Ny - 1);
            if (std::fabs(u - 0.5) < half && std::fabs(v - 0.5) < half)
                epgrid[(size_t)i * Ny + j] = n_tan;
        }
    }

    RCWAConfig cfg;
    cfg.nG = 13;                                  // → nG_out = 9
    cfg.L1 = Eigen::Vector2d(cell, 0.0);
    cfg.L2 = Eigen::Vector2d(0.0, cell);
    cfg.freq = complex(1.0 / 13.5, 0.0);
    cfg.theta = 0.0;
    cfg.phi   = 0.0;

    RCWA solver(cfg);
    solver.Add_LayerUniform(1.0, eps_from_n(complex(1.0, 0.0)));
    solver.Add_LayerGrid(abs_t, Nx, Ny);
    solver.Add_LayerUniform(2.5, eps_from_n(n_ru));
    for (int i = 0; i < nb; ++i) {
        solver.Add_LayerUniform(2.8, eps_from_n(n_mo));
        solver.Add_LayerUniform(4.2, eps_from_n(n_si));
    }
    solver.Add_LayerUniform(1.0, eps_from_n(n_si));
    solver.Init_Setup();
    CHECK(solver.nG() == 9);
    solver.GridLayer_geteps(epgrid);

    PlaneWaveExcitation exc; exc.p_amp = 1.0;
    solver.MakeExcitationPlanewave(exc);
    RTResult rt = solver.RT_Solve(/*normalize=*/true);

    // grcwa reference: R = 0.56675020, T = 0.00741819 at this config
    CHECK(rt.R == Approx(0.56675020).epsilon(1e-5));
    CHECK(rt.T == Approx(0.00741819).epsilon(1e-3));

    // Reflected field in air (backward-propagating, layer 0, z=0). All
    // reflected orders match grcwa to machine precision. Golden values below
    // are the grcwa per-order reflected-field coefficients for this config.
    FieldFourier refl = solver.BackwardPropagatedFieldFourier(0, 0.0);
    // Order 0 (specular): ex = -0.3283787647 + 0.6693281360i
    CHECK(std::abs(refl.ex(0) - complex(-0.32837876468760635, 0.66932813595394325)) < 1e-9);
    // Order 1 (G=(-1,0)): ex = -0.0351497273 + 0.0145995195i, ez = 0.0015833417 - 0.0006576446i
    CHECK(std::abs(refl.ex(1) - complex(-0.035149727329177233, 0.01459951954681468)) < 1e-9);
    CHECK(std::abs(refl.ez(1) - complex(0.0015833416756686704, -0.00065764458218208757)) < 1e-9);
    // Specular reflected power: |b0|² of the dominant order ≈ R of the bare field
    double spec_power = refl.ey.cwiseAbs2().sum();   // Ey dominates for p-pol at normal incidence
    (void)spec_power;
}

// EUV absorber at oblique incidence (θ = 6°). Same stack/pattern as the
// normal-incidence absorber test. Validated against grcwa.
TEST_CASE("RCWA EUV absorber on mirror oblique incidence (theta=6deg)", "[rcwa][euv]") {
    auto eps_from_n = [](complex n) { return n * n; };
    complex n_mo (0.9226, 0.0064);
    complex n_si (0.9997, 0.0018);
    complex n_ru (0.9114, 0.0171);
    complex n_tan(0.9562, 0.0323);

    const int Nx = 100, Ny = 100;
    const double cell = 300.0, rect = 60.0, abs_t = 60.0;
    const int nb = 40;
    const double half = rect / (2.0 * cell);

    std::vector<complex> epgrid((size_t)Nx * Ny, complex(1.0, 0.0));
    for (int i = 0; i < Nx; ++i) {
        double u = (double)i / (Nx - 1);
        for (int j = 0; j < Ny; ++j) {
            double v = (double)j / (Ny - 1);
            if (std::fabs(u - 0.5) < half && std::fabs(v - 0.5) < half)
                epgrid[(size_t)i * Ny + j] = n_tan;
        }
    }

    RCWAConfig cfg;
    cfg.nG = 13;                                  // → nG_out = 9
    cfg.L1 = Eigen::Vector2d(cell, 0.0);
    cfg.L2 = Eigen::Vector2d(0.0, cell);
    cfg.freq = complex(1.0 / 13.5, 0.0);
    cfg.theta = M_PI / 30;                        // 6°
    cfg.phi   = 0.0;

    RCWA solver(cfg);
    solver.Add_LayerUniform(1.0, eps_from_n(complex(1.0, 0.0)));
    solver.Add_LayerGrid(abs_t, Nx, Ny);
    solver.Add_LayerUniform(2.5, eps_from_n(n_ru));
    for (int i = 0; i < nb; ++i) {
        solver.Add_LayerUniform(2.8, eps_from_n(n_mo));
        solver.Add_LayerUniform(4.2, eps_from_n(n_si));
    }
    solver.Add_LayerUniform(1.0, eps_from_n(n_si));
    solver.Init_Setup();
    CHECK(solver.nG() == 9);
    solver.GridLayer_geteps(epgrid);

    PlaneWaveExcitation exc; exc.p_amp = 1.0;
    solver.MakeExcitationPlanewave(exc);
    RTResult rt = solver.RT_Solve(/*normalize=*/true);

    // grcwa reference at θ=6°: R = 0.5870720083, T = 0.0074574197
    CHECK(rt.R == Approx(0.5870720083).epsilon(1e-5));
    CHECK(rt.T == Approx(0.0074574197).epsilon(1e-3));

    // Reflected field in air at θ=6°. Golden values from grcwa, matched by
    // G-vector (degenerate-order independent).
    FieldFourier refl = solver.BackwardPropagatedFieldFourier(0, 0.0);
    const IntMatrix& G = solver.G();
    auto find_order = [&](int gx, int gy) -> int {
        for (int i = 0; i < solver.nG(); ++i)
            if (G(i,0) == gx && G(i,1) == gy) return i;
        return -1;
    };
    // G=(0,0) specular: ex = 0.095652855788650529 + 0.7487993216328479i,
    //                    ez = 0.010053520258644148 + 0.078701980067677757i
    int i0 = find_order(0, 0);
    REQUIRE(i0 >= 0);
    CHECK(std::abs(refl.ex(i0) - complex( 0.095652855788650529,  0.7487993216328479)) < 1e-9);
    CHECK(std::abs(refl.ez(i0) - complex( 0.010053520258644148,  0.078701980067677757)) < 1e-9);
    // G=(1,0): ex = -0.014581850948967027 + 0.034996101456876559i
    int i1 = find_order(1, 0);
    REQUIRE(i1 >= 0);
    CHECK(std::abs(refl.ex(i1) - complex(-0.014581850948967027,  0.034996101456876559)) < 1e-9);
    // G=(0,1): ex = -0.023727395629504811 + 0.030379369087523173i
    int i2 = find_order(0, 1);
    REQUIRE(i2 >= 0);
    CHECK(std::abs(refl.ex(i2) - complex(-0.023727395629504811,  0.030379369087523173)) < 1e-9);
}
