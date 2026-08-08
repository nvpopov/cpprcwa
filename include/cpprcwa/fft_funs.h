#pragma once

#include <fftw3.h>
#include <utility>
#include <cpprcwa/types.h>

namespace cpprcwa {

// ── FFT plan cache (§5.2.1) ───────────────────────────────────────────────
// FFTW plans are expensive (~10-100ms). Cache one (forward, backward) pair
// per (Nx, Ny). Thread-safe plan lookup; reuse requires FFTW threads support
// or per-thread caches (§6.11).
class FFTWPlanCache {
public:
    FFTWPlanCache();
    ~FFTWPlanCache();

    // Returns (forward, backward) plans for the given grid size. The cache
    // owns the plans; caller must NOT destroy them.
    std::pair<fftw_plan, fftw_plan> get(int Nx, int Ny);

    FFTWPlanCache(const FFTWPlanCache&) = delete;
    FFTWPlanCache& operator=(const FFTWPlanCache&) = delete;
private:
    struct Impl;
    Impl* impl_;
};

// ── FFT operations (§5.2) ────────────────────────────────────────────────

// Build nG×nG convolution matrix: C[i,j] = s_fft[(G[i]-G[j]) mod (Nx,Ny)]
ComplexMatrix get_conv(double dN,
                       const std::vector<complex>& s_in_flat,
                       int Nx, int Ny,
                       const IntMatrix& G);

// Extract Fourier coefficients at the G points: s_fft[G[:,0], G[:,1]]
ComplexVector get_fft(double dN,
                      const std::vector<complex>& s_in_flat,
                      int Nx, int Ny,
                      const IntMatrix& G);

// Reconstruct real-space grid from nG Fourier coefficients at positions G.
GridMatrix get_ifft(double dN, int Nx, int Ny,
                    const ComplexVector& s_in,
                    const IntMatrix& G);

// ── Epsilon_fft (§5.2.5) ──────────────────────────────────────────────────
// Isotropic: eps_grid is a single flat array (length Nx*Ny).
//   Returns (epsinv = inv(fft_conv(eps)), eps2 = block_diag(eps_fft, eps_fft)).
// Anisotropic: eps_grid holds 3 flat arrays [epsx, epsy, epsz].
//   Returns (epsinv = inv(fft_conv(epsz)), eps2 = block_diag(epsx_fft, epsy_fft)).

struct EpsilonFftResult {
    ComplexMatrix epsinv;   // nG × nG
    ComplexMatrix eps2;     // 2nG × 2nG
};

EpsilonFftResult Epsilon_fft(double dN,
                             const std::vector<complex>& eps_grid,   // isotropic
                             int Nx, int Ny,
                             const IntMatrix& G);

EpsilonFftResult Epsilon_fft(double dN,
                             const std::vector<std::vector<complex>>& eps_grid, // anisotropic
                             int Nx, int Ny,
                             const IntMatrix& G);

} // namespace cpprcwa
