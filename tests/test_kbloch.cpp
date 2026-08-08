#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cpprcwa/kbloch.h>

using namespace cpprcwa;
using Catch::Approx;

TEST_CASE("Lattice_Reciprocate square", "[kbloch]") {
    Eigen::Vector2d L1(1.5, 0.0), L2(0.0, 1.5);
    auto [Lk1, Lk2] = Lattice_Reciprocate(L1, L2);
    // d = 2.25; Lk1 = (1/1.5, 0), Lk2 = (0, 1/1.5)
    CHECK(Lk1[0] == Approx(1.0 / 1.5));
    CHECK(Lk1[1] == Approx(0.0));
    CHECK(Lk2[0] == Approx(0.0));
    CHECK(Lk2[1] == Approx(1.0 / 1.5));
}

TEST_CASE("Lattice_Reciprocate hexagonal", "[kbloch]") {
    // Hexagonal: L1=(1,0), L2=(cos(60), sin(60))=(0.5, sqrt(3)/2)
    double s = std::sqrt(3.0) / 2.0;
    Eigen::Vector2d L1(1.0, 0.0), L2(0.5, s);
    auto [Lk1, Lk2] = Lattice_Reciprocate(L1, L2);
    // d = sqrt(3)/2
    // Lk1 = (L2[1]/d, -L2[0]/d) = (1, -1/sqrt(3))
    // Lk2 = (-L1[1]/d, L1[0]/d)  = (0, 2/sqrt(3))
    CHECK(Lk1[0] == Approx(1.0));
    CHECK(Lk1[1] == Approx(-1.0 / std::sqrt(3.0)));
    CHECK(Lk2[0] == Approx(0.0).margin(1e-15));
    CHECK(Lk2[1] == Approx(2.0 / std::sqrt(3.0)));
}

TEST_CASE("Lattice_getG square circular nG_out<=nG", "[kbloch]") {
    Eigen::Vector2d L1(1.5, 0.0), L2(0.0, 1.5);
    auto [Lk1, Lk2] = Lattice_Reciprocate(L1, L2);
    int nG_req = 20;
    auto [G, nG_out] = Lattice_getG(nG_req, Lk1, Lk2, 0);
    CHECK(nG_out <= nG_req);
    CHECK(G.rows() == nG_out);
    CHECK(G.cols() == 2);
    // First entry should be (0,0)
    CHECK(G(0, 0) == 0);
    CHECK(G(0, 1) == 0);
}

TEST_CASE("Lattice_getG parallelogramic gives odd NGroot^2", "[kbloch]") {
    Eigen::Vector2d L1(1.5, 0.0), L2(0.0, 1.5);
    auto [Lk1, Lk2] = Lattice_Reciprocate(L1, L2);
    int nG_req = 25;
    auto [G, nG_out] = Lattice_getG(nG_req, Lk1, Lk2, 1);
    // sqrt(25)=5 (odd), so nG_out should be 25
    CHECK(nG_out == 25);
    CHECK(G.rows() == 25);
}

TEST_CASE("Lattice_getG symmetry: -G present whenever G present", "[kbloch]") {
    Eigen::Vector2d L1(1.5, 0.0), L2(0.0, 1.5);
    auto [Lk1, Lk2] = Lattice_Reciprocate(L1, L2);
    auto [G, nG_out] = Lattice_getG(50, Lk1, Lk2, 0);
    // For every (gx, gy) in G (except 0), (-gx, -gy) must also be present.
    for (int i = 0; i < nG_out; ++i) {
        int gx = G(i, 0), gy = G(i, 1);
        if (gx == 0 && gy == 0) continue;
        bool found = false;
        for (int j = 0; j < nG_out; ++j) {
            if (G(j, 0) == -gx && G(j, 1) == -gy) { found = true; break; }
        }
        CHECK(found);
    }
}

TEST_CASE("Lattice_SetKs normal incidence", "[kbloch]") {
    Eigen::Vector2d L1(1.5, 0.0), L2(0.0, 1.5);
    auto [Lk1, Lk2] = Lattice_Reciprocate(L1, L2);
    auto [G, nG_out] = Lattice_getG(10, Lk1, Lk2, 0);
    complex kx0(0.0, 0.0), ky0(0.0, 0.0);
    ComplexVector kx, ky;
    Lattice_SetKs(G, kx0, ky0, Lk1, Lk2, kx, ky);
    // kx[0] should be 0 (the G=0 mode at normal incidence)
    CHECK(kx(0).real() == Approx(0.0).margin(1e-12));
    CHECK(ky(0).real() == Approx(0.0).margin(1e-12));
    // |kx|^2 + |ky|^2 should match 4*pi^2*|G|^2
    for (int i = 0; i < nG_out; ++i) {
        complex gx = Lk1[0] * G(i, 0) + Lk2[0] * G(i, 1);
        complex gy = Lk1[1] * G(i, 0) + Lk2[1] * G(i, 1);
        CHECK(std::abs(kx(i) - 2.0 * M_PI * gx) < 1e-12);
        CHECK(std::abs(ky(i) - 2.0 * M_PI * gy) < 1e-12);
    }
}
