#pragma once

#include <array>
#include <complex>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace cpprcwa {

using complex = std::complex<double>;
using ComplexVector = Eigen::VectorXcd;
using ComplexMatrix = Eigen::MatrixXcd;
using RealVector    = Eigen::VectorXd;
using IntMatrix     = Eigen::MatrixXi;

// Row-major grid for real-space (Nx, Ny) fields. Required for FFTW interop:
// fftw_plan_dft_2d expects row-major data. Using this alias everywhere grids
// appear avoids transpose bugs (see PLAN.md §5.2.2).
using GridMatrix = Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// ── Configuration ──────────────────────────────────────────────────────────

struct RCWAConfig {
    int nG;                              // Requested truncation order
    Eigen::Vector2d L1;                  // Direct lattice vector 1
    Eigen::Vector2d L2;                  // Direct lattice vector 2
    complex freq;                        // COMPLEX frequency (caller folds in Qabs; §6.1)
    double theta = 0.0;                  // Polar angle (radians)
    double phi   = 0.0;                  // Azimuthal angle (radians)
    int verbose  = 1;
};

// ── Layers ─────────────────────────────────────────────────────────────────

enum class LayerType : int {
    Uniform = 0,
    Grid    = 1,
    Fourier = 2,   // stub — throws NotImplementedError if used
};

struct LayerUniform {
    double thickness;
    complex epsilon;
};

struct LayerGrid {
    double thickness;
    int Nx, Ny;
};

// ── Excitation ─────────────────────────────────────────────────────────────

enum class Direction : int { Forward = 0, Backward = 1 };

struct PlaneWaveExcitation {
    double p_amp = 0.0, p_phase = 0.0;
    double s_amp = 0.0, s_phase = 0.0;
    int    order = 0;
    Direction direction = Direction::Forward;
};

// ── Field/Result containers ────────────────────────────────────────────────

struct FieldFourier {
    ComplexVector ex, ey, ez;            // Length nG each
    ComplexVector hx, hy, hz;
};

struct FieldGrid {
    GridMatrix ex, ey, ez;               // Shape (Nx, Ny)
    GridMatrix hx, hy, hz;
};

struct RTResult {
    double R = 0.0, T = 0.0;
    ComplexVector R_per_order;           // Populated if byorder=true
    ComplexVector T_per_order;
};

} // namespace cpprcwa
