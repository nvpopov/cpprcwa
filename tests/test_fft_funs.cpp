#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cpprcwa/fft_funs.h>

using namespace cpprcwa;
using Catch::Approx;

TEST_CASE("FFT round-trip: fft of constant gives DC peak", "[fft_funs]") {
    int Nx = 16, Ny = 16;
    std::vector<complex> s_in((size_t)Nx * Ny, complex(3.14, 0.0));
    IntMatrix G(1, 2);
    G << 0, 0;
    double dN = 1.0 / (Nx * Ny);
    ComplexVector s_fft = get_fft(dN, s_in, Nx, Ny, G);
    // FFT of constant 3.14 at DC should be 3.14 (after dN scaling, sum * dN = 3.14)
    CHECK(s_fft(0).real() == Approx(3.14).margin(1e-12));
    CHECK(s_fft(0).imag() == Approx(0.0).margin(1e-12));
}

TEST_CASE("FFT convolution preserves Parseval for impulse", "[fft_funs]") {
    int Nx = 8, Ny = 8;
    // Impulse at origin: fft gives all-ones matrix (scaled by dN).
    std::vector<complex> s_in((size_t)Nx * Ny, complex(0.0, 0.0));
    s_in[0] = complex(1.0, 0.0);
    IntMatrix G(3, 2);
    G << 0, 0,  1, 0,  0, 1;
    double dN = 1.0 / (Nx * Ny);
    ComplexMatrix C = get_conv(dN, s_in, Nx, Ny, G);
    // Convolution with an impulse is identity-like: C[i,j] = delta_{i,j} * dN * N = dN
    // Actually for impulse at origin, fft = 1 everywhere, so C[i,j] = dN for all i,j.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            CHECK(C(i, j).real() == Approx(dN).margin(1e-12));
}

TEST_CASE("Epsilon_fft isotropic DC component equals spatial average", "[fft_funs]") {
    int Nx = 32, Ny = 32;
    IntMatrix G(5, 2);
    G << 0, 0,  1, 0,  0, 1,  -1, 0,  0, -1;
    // Constant epsilon = 2.0 everywhere
    std::vector<complex> eps((size_t)Nx * Ny, complex(2.0, 0.0));
    double dN = 1.0 / (Nx * Ny);
    auto r = Epsilon_fft(dN, eps, Nx, Ny, G);
    // eps_fft[0,0] should equal spatial mean = 2.0
    CHECK(r.eps2(0, 0).real() == Approx(2.0).margin(1e-12));
    CHECK(r.eps2(0, 0).imag() == Approx(0.0).margin(1e-12));
    // epsinv[0,0] should equal 1/2
    CHECK(r.epsinv(0, 0).real() == Approx(0.5).margin(1e-12));
}
