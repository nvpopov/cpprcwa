# grcwa → cpprcwa: Python-to-C++ Transformation Plan

## 1. Overview

Transform the **grcwa** Python library (Rigorous Coupled-Wave Analysis for photonic structures) into a high-performance C++ library called **cpprcwa**, with optional GPU acceleration via CUDA.

The port covers the **forward RCWA solver only**. The Python library's optional autograd-based reverse-mode automatic differentiation (used for topology optimization in `ex3.py`) is **out of scope** for this port.

### 1.1 Source Files Audited

The grcwa package lives at `grcwa_orig/grcwa/` with line counts:

| Source File | Actual Lines | Role | Port Complexity |
|---|---|---|---|
| `grcwa/__init__.py` | 9 | Public API exports | Trivial (header) |
| `grcwa/backend.py` | 113 | NumPy/Autograd dispatch | **Omit** (use BLAS/FFTW directly) |
| `grcwa/primitives.py` | 56 | Autograd custom gradients | **Omit** (autodiff is out of scope) |
| `grcwa/kbloch.py` | 131 | Reciprocal lattice / k-space | Low |
| `grcwa/fft_funs.py` | 79 | FFT convolution matrices | Medium |
| `grcwa/rcwa.py` | 648 | Main RCWA engine + S-matrix | High |

**Total source lines: 1036. Modules actually ported: 3** (`kbloch`, `fft_funs`, `rcwa`) + a public header.

**C++ aggregator header:** `include/cpprcwa/cpprcwa.h` re-exports the public surface — `#include`s `types.h`, `kbloch.h`, `fft_funs.h`, `rcwa.h`. Downstream code should `#include <cpprcwa/cpprcwa.h>` only. This mirrors `grcwa/__init__.py`.

### 1.2 Reference Examples and Tests (golden output values for validation)

| File | What it exercises | Reference output |
|---|---|---|
| `example/ex1.py` | Square lattice hole (uniform + patterned + uniform) | See `benchmarks` |
| `example/ex2.py` | Two patterned layers (oblique incidence, `θ=π/10`) | — |
| `example/ex3.py` | **Topology optimization via autograd (OUT OF SCOPE)** | Skip |
| `example/ex4.py` | Hexagonal lattice (non-orthogonal coords, Nx=Ny=1000) | — |
| `tests/test_rcwa.py` | Validation against S4 reference (Victor's code) | `T=0.85249901083265` (p-pol), `T=0.83900479939861` (s-pol) |
| `tests/test_kbloch.py` | kbloch gradient tests + FFT round-trip | — |

The S4 reference values in `test_rcwa.py:70,75` are the **golden numbers** to match:
```python
# test_rcwa.py:69-75
R,T= obj.RT_Solve(normalize=0)
assert abs(T-0.85249901083265)<tolS4 * T    # p-polarization
R,T= obj.RT_Solve(normalize=0)
assert abs(T-0.83900479939861)<tolS4 * T    # s-polarization
```

### 1.3 Implementation Status (Aug 2026)

Phases 1–8 and most of Phase 10 are implemented and validated. The C++ library
(`include/cpprcwa/`, `src/`) reproduces the S4 golden values within the `1e-3`
tolerance and matches grcwa's `ex1.py`/`ex2.py`/`ex4.py` to printed precision.

| Phase | Status | Notes |
|---|---|---|
| 1 Scaffolding | ✅ | CMake + Eigen/FFTW/LAPACK, Catch2 v3, aggregator header |
| 2 kbloch | ✅ | Circular (degeneracy-aware) + parallelogramic G selection |
| 3 FFT utilities | ✅ | Plan cache, `get_conv/fft/ifft`, isotropic + anisotropic `Epsilon_fft` |
| 4 Layer setup | ✅ | Uniform + grid layers, `Init_Setup`, complex `freq`/`omega` |
| 5 S-matrix solver | ✅ | Redheffer star product; S4 golden R/T reproduced |
| 6 Field reconstruction | ✅ | `Solve_Field*`, `Return_eps` (matches grcwa to ~1e-12) |
| 7 Post-processing | ✅ | `Volume_integral`, stress tensor (matches grcwa) |
| 8 Examples | ✅ | `ex1`/`ex2`/`ex4` ports + golden scripts + EUV multilayer mirror + EUV absorber + reflected-field validation & plotting + quasi-1D line grating |
| 9 GPU | ⛔ | **Descoped** — feasibility spike run (see §Phase 9): no `cusolverDnZgeev` exists, S-matrix matmuls don't accelerate on GPU, only cuFFT helps (~20×, small setup fraction) |
| 10 Polish | ✅ | README, install target; Python bindings (nanobind, grcwa drop-in) done; OpenMP/pybind11 deferred |

**Memory work (Aug 2026):**
- `Volume_integral` rewritten **block-wise** (§10.4): the 4nG×4nG
  `outer(ab,conj(ab))`, `Mt`, and the 3nG×3nG `Mtotal`/`F` are never
  materialized — the trace collapses onto 2nG×2nG blocks (`A`, `B`, `Maa`,
  `Mab` + two Hadamard products). Validated identical to the original formula
  to ~1e-16.
- **Uniform-layer dedup**: kp/q/phi are shared (via `shared_ptr<const>`) across
  identical ε values, and all uniform layers share one Identity phi. Periodic
  stacks hold ~nDistinct full matrices instead of one per layer: persistent
  storage drops ~20× (quasi-1D nG=113: 132→6.5 MiB; nG=253: 662→32 MiB) with
  measured peak RSS 168→42 MiB, `R` unchanged.
- **`RCWAConfig::report_memory`** (+ `--mem` in the examples): after
  `Init_Setup` prints the estimated persistent + transient peak memory,
  validated against `/usr/bin/time -v` RSS to within ~5–9%.

**Notable bugs found and fixed during the port:**
- Uninitialized Eigen matrices (`Jk`, `k_mat`) — off-diagonal garbage polluted
  the `Jk·Jkᵀ` products (manifested as NaN/inf under Catch2's heap layout).
- `k_mat` in `SolveLayerEigensystem_patterned` was built as `(2nG, 2nG)` with
  diagonal blocks; Python's `vstack(diag(kx), diag(ky))` is `(2nG, nG)` and
  produces `kx·ky` cross terms in `kkT`.
- `MakeExcitationPlanewave` must project `(p, s)` amplitudes through the
  polarization basis (`-s·cosθ·cosφ − p·sinφ`, etc.) — not set `a0[order]=p_amp`.
- `normalization_` requires `sqrt(ε₀)/cos(θ)`, not `sqrt(ε₀)`.
- `Matrix_zintegral`'s `Mt` is `4nG×4nG` built from `nG2=q.size()` blocks;
  `Volume_integral`'s `F` is `3nG×4nG` and `Mtotal` is `3nG×3nG`.

**Performance work (vs grcwa, EUV absorber, nG=97):**
- `EIGEN_USE_BLAS` routes matrix products through OpenBLAS `zgemm`
  (~4.4× on matmul vs Eigen's internal kernels).
- FFTW plans switched `FFTW_MEASURE` → `FFTW_ESTIMATE` (setup drops
  ~20–200 ms per plan; negligible execution penalty at these grid sizes).
- OpenBLAS thread oversubscription: 12 threads ≈ 4.9 s vs 6 threads ≈ 1.3 s
  for the nG=97 S-matrix. `ex_euv_absorber` caps threads at `min(cores,6)`.
- S-matrix: uniform-pair T-matrix caching (periodic stacks), thread-local
  LAPACK workspace reuse, LU-solve + common-subexpression elimination in the
  star product.
- Quasi-1D (y-invariant) exact reductions: uniform-layer `(kp,q,phi)` caching
  by ε, a 1D harmonic filter (j==0 only), and a diagonal scalar recursion for
  the all-uniform multilayer suffix assembled via an overlapping cascade.
  `ex_quasi1d_absorber --quasi1d` (nG=201) drops total ~7.2 s → ~0.37 s
  (~30× vs grcwa's 11.0 s) with machine-precision agreement (normal + oblique).
- Net: total ~2000→~1050 ms and `rt_solve` ~1700→~810 ms (≈2×), ≈5× faster
  than grcwa overall, with machine-precision field agreement.
- S-matrix periodic-core doubling was attempted and reverted: the naive
  stack-recombination formula assumes a continuous boundary medium, which
  alternating-material cores violate (drops inter-period interfaces,
  double-counts boundary phases → ~4e-4 R error).

---

## 2. External Dependency Mapping

| Python (NumPy) | C++ Replacement | Purpose |
|---|---|---|
| `np.dot` / `np.outer` | **BLAS** (`zgemm`, `zgemv`) or **Eigen** | Matrix multiply |
| `np.linalg.inv` | **LAPACK** (`zgetrf` + `zgetri`) or Eigen | Matrix inversion |
| `np.linalg.eig` | **LAPACK** (`zgeev`) or Eigen::ComplexEigenSolver | Eigendecomposition (general, non-Hermitian) |
| `np.fft.fft2` / `ifft2` | **FFTW** (`fftw_plan_dft_2d`) or cuFFT | 2D FFT/IFFT |
| `np.sqrt`, `exp`, `sin`, `cos` | `<cmath>` (std::complex specializations) | Element-wise math |
| `np.diag`, `np.eye` | Eigen `Matrix::Identity()`, custom small helpers | Matrix construction |
| `np.linalg.norm` | `std::hypot` or manual sum of squares | Vector norms in `kbloch` |
| `np.where(imag(q)<0, -q, q)` | Manual element-wise loop or Eigen `select` | Branch cut for `sqrt` |
| `np.meshgrid(..., indexing='ij')` | Custom small helper | Cartesian / lattice grids |
| `np.linspace(0, 1, Nx)` | `Eigen::VectorXd::LinSpaced(Nx, 0.0, 1.0)` | Lattice coords |

**GPU acceleration path (optional, Phase 9):**
- **cuBLAS** (`cublasZgemm`) — matrix multiply
- **cuSOLVER** (`cusolverDnZgeev`) — eigendecomposition
- **cuSOLVER** (`cusolverDnZgetrf` + `Zgetri`) — inversion
- **cuFFT** (`cufftPlan2d`) — 2D FFT
- Custom CUDA kernels for the convolution matrix index-gather `s_conv[i,j] = s_fft[G[i]-G[j]]` and the branch-cut `where` pattern

---

## 3. Target Directory Structure

```
cpprcwa/
├── PLAN.md                        # This file
├── CMakeLists.txt                 # Build system (CPU only)
├── CMakeListsGPU.txt.in           # Optional CUDA build (or use option in main CMake)
├── README.md                      # Quick-start usage
├── LICENSE                        # GPLv3 (inherited from grcwa)
├── include/cpprcwa/
│   ├── cpprcwa.h                   # Aggregator header (mirrors grcwa/__init__.py)
│   ├── rcwa.h                      # Main `RCWA` class (public API)
│   ├── kbloch.h                    # Reciprocal lattice functions
│   ├── fft_funs.h                  # FFT convolution operations
│   └── types.h                     # Common types (config, layer, results)
├── src/
│   ├── rcwa.cpp                   # RCWA engine implementation
│   ├── kbloch.cpp                 # Reciprocal lattice implementation
│   ├── fft_funs.cpp               # FFT operations implementation
│   ├── internal/
│   │   ├── lapack_wrappers.h      # Thin LAPACK call wrappers
│   │   ├── lapack_wrappers.cpp
│   │   ├── fftw_plans.h           # FFTW plan cache
│   │   ├── fftw_plans.cpp
│   │   ├── branch_cut.h           # sqrt branch-cut helper
│   │   └── branch_cut.cpp
│   └── utils.cpp                  # diag, eye, meshgrid, linspace helpers
├── tests/
│   ├── CMakeLists.txt             # Test build config
│   ├── test_kbloch.cpp            # Reciprocal lattice correctness
│   ├── test_fft_funs.cpp          # FFT round-trip + Epsilon_fft
│   ├── test_rcwa.cpp              # S-matrix, R/T, fields, S4 golden test
│   ├── test_data/                 # Saved Python reference outputs (.npy / .txt)
│   └── golden/                    # Pre-generated golden output files
```

**Golden-data serialization convention (RESOLVE BEFORE WRITING TESTS):** C++ tests cannot easily parse NumPy `.npy`. Pick one and apply it consistently:
- `generate_golden.py` writes **two files per artifact**: the `.npy` (for `compare_results.py`) and a `.txt` (whitespace-delimited `rows×cols` header + complex pairs as `real imag`) for C++ tests.
- Alternatively, the C++ test harness includes a tiny (≈50-line) `.npy` reader (header parse + `std::ifstream` of the raw buffer).

**Recommendation:** use the `.txt` path — it's debuggable with a text editor and avoids endianness/version pitfalls. Document the exact format in `tests/README.md`.
├── examples/
│   ├── CMakeLists.txt
│   ├── ex1_square_lattice.cpp     # Port of ex1.py
│   ├── ex2_two_layers.cpp         # Port of ex2.py (oblique incidence)
│   └── ex4_hexagonal.cpp          # Port of ex4.py (non-orthogonal lattice)
├── benchmarks/
│   ├── CMakeLists.txt
│   ├── bench_kbloch.cpp           # Per-module timings
│   ├── bench_fft.cpp
│   ├── bench_eig.cpp              # Eigendecomposition across nG
│   └── bench_full_rt.cpp          # End-to-end RT_Solve timings
└── scripts/
    ├── generate_golden.py         # Run grcwa in Python, save output files
    ├── compare_results.py         # Diff C++ vs grcwa output
    └── plot_comparison.py         # Optional matplotlib visual check
```

### 3.1 Build System Notes

- **CMake ≥ 3.18** with C++17 requirement
- **Find modules**: `find_package(Eigen3 REQUIRED)`, `find_package(FFTW3 REQUIRED)`, `find_package(LAPACK REQUIRED)`
- **Tests**: Use **Catch2 v3** — v3 is *not* single-header (that was v2). Either `find_package(Catch2)` against a system install, or `FetchContent_MakeAvailable(Catch2)` from CMake, and add one TU containing `#define CATCH_CONFIG_MAIN` + `#include <catch2/catch_all.hpp>` (or link `Catch2::Catch2WithMain`). Do **not** drop a `catch.hpp` into the repo expecting it to work with v3.
- **Examples**: separate `add_executable` per example
- **GPU build**: `-DCPPRCWA_USE_CUDA=ON` flag; links `cufft`, `cublas`, `cusolver`
- **Install targets**: `install(TARGETS cpprcwa EXPORT cpprcwaTargets ...)` so downstream projects can `find_package(cpprcwa)`

---

## 4. Common Types (`include/cpprcwa/types.h`)

```cpp
#pragma once

#include <array>
#include <complex>
#include <vector>
#include <Eigen/Dense>

namespace cpprcwa {

using complex = std::complex<double>;
using ComplexVector = Eigen::VectorXcd;
using ComplexMatrix = Eigen::MatrixXcd;
using RealVector    = Eigen::VectorXd;
using IntMatrix     = Eigen::MatrixXi;   // G is (nG, 2) ints

// Row-major grid for real-space (Nx, Ny) fields. Required for FFTW interop:
// fftw_plan_dft_2d expects row-major data. Using this alias everywhere grids
// appear (fft_funs, Solve_FieldOnGrid, Return_eps) avoids transpose bugs.
// See §5.2.2 "Eigen ↔ FFTW memory layout" for the full rationale.
using GridMatrix = Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// ── Configuration ────────────────────────────────────────────────────────

struct RCWAConfig {
    int nG;                              // Requested truncation order (actual may be smaller)
    Eigen::Vector2d L1;                  // Direct lattice vector 1, e.g. {1.5, 0.0} (see §6.12)
    Eigen::Vector2d L2;                  // Direct lattice vector 2, e.g. {0.0, 1.5}
    complex freq;                        // Base frequency (COMPLEX — see §6.1; examples pass freq*(1+1j/2/Qabs))
    double theta;                        // Polar angle θ (radians)
    double phi;                          // Azimuthal angle φ (radians)
    int verbose = 1;

    // NOTE: Qabs, Pscale, Gmethod are intentionally NOT in this struct.
    //   - Qabs: in grcwa the caller folds it into `freq` before construction (see §6.1).
    //   - Pscale, Gmethod: passed to Init_Setup() (matches Python keyword args); see §6.4.
};

// ── Layers ───────────────────────────────────────────────────────────────

enum class LayerType : int {
    Uniform = 0,
    Grid    = 1,
    Fourier = 2   // present in API but not exercised by examples; port as best-effort
};

struct LayerUniform {
    double thickness;
    complex epsilon;                     // Isotropic ε (only diagonal used)
};

struct LayerGrid {
    double thickness;
    int Nx, Ny;
};

// LayerFourier exists in Python API (rcwa.py:73) but no example uses it.
// Port as a stub: store thickness + opaque params blob; compute handled if/when example appears.

// ── Excitation ───────────────────────────────────────────────────────────

struct PlaneWaveExcitation {
    double p_amp = 0.0, p_phase = 0.0;
    double s_amp = 0.0, s_phase = 0.0;
    int    order    = 0;                 // Which G-order is the incident wave
    enum class Direction { Forward, Backward } direction = Direction::Forward;
};

// ── Field/Result containers ──────────────────────────────────────────────

struct FieldFourier {
    ComplexVector ex, ey, ez;            // Length nG each
    ComplexVector hx, hy, hz;
};

struct FieldGrid {
    GridMatrix ex, ey, ez;               // Shape (Nx, Ny), row-major for FFTW
    GridMatrix hx, hy, hz;
};

struct RTResult {
    double R, T;
    ComplexVector R_per_order;           // If byorder=1
    ComplexVector T_per_order;
};

} // namespace cpprcwa
```

### 4.1 Critical Notes on Type Design

- `omega = 2*pi*freq*(1 + i/(2*Qabs)) + 0j` — **omega is complex**. `Init_Setup` uses `Uniform_ep_list[0]` (the first uniform layer's ε) to compute `kx0 = ω·sin(θ)·cos(φ)·√ε_0`. **Layer 0 MUST be uniform** — enforce this in `Init_Setup` with an assertion.
- The Python code does `self.omega = 2*bd.pi*freq+0j` (line 19) — note the `+0j` forces complex type. Without this, NumPy would upcast automatically when multiplied by `sqrt(complex)`, but explicit is safer in C++.
- `kx0` and `ky0` are **complex** (because omega is complex) — this matters for the branch cut in `sqrt`.

---

## 5. Module-by-Module Porting Details

### 5.1 `kbloch.h / kbloch.cpp` — Low Complexity

Three pure-math functions. No BLAS, no FFT, no external library beyond `<cmath>` and `Eigen::Vector2d`.

#### 5.1.1 `Lattice_Reciprocate(L1, L2) → (Lk1, Lk2)`

```cpp
// kbloch.py:12-17
// Uses Eigen::Vector2d throughout (see §6.12 — single lattice-vector type).
std::pair<Eigen::Vector2d, Eigen::Vector2d> Lattice_Reciprocate(
    const Eigen::Vector2d& L1,
    const Eigen::Vector2d& L2);

double d = L1[0]*L2[1] - L1[1]*L2[0];
// Lk1 = [L2[1]/d, -L2[0]/d]
// Lk2 = [-L1[1]/d, L1[0]/d]
```

#### 5.1.2 `Lattice_getG(nG, Lk1, Lk2, method=0) → (G, nG_out)`

Two modes:

**`method=0` (circular, `Gsel_circular`, lines 80–131):**
1. Compute `u = |Lk1|`, `v = |Lk2|`, `uv = Lk1·Lk2`, `uxv = Lk1[0]*Lk2[1] - Lk1[1]*Lk2[0]`
2. `circ_area = nG * |uxv|`
3. `circ_radius = √(circ_area/π) + u + v` (extra slack)
4. `u_extent = 1 + ⌊circ_radius / (u·√(1 - (uv/(u·v))²))⌋`
5. `v_extent = 1 + ⌊circ_radius / (v·√(1 - (uv/(u·v))²))⌋`
6. Generate `2*u_extent+1` × `2*v_extent+1` grid of `(i,j)` integer pairs
7. Sort all pairs by `|G|² = (i·u)² + (j·v)² + 2·i·j·uv`
8. **Degeneracy handling (lines 119–124, critical):** Walk sorted list, find largest `i` such that `|Gl2[i] - Gl2[i-1]| > tol` where `tol = 1e-10·max(u², v²)`. This keeps symmetry-respecting sets.
9. Set `nG = i` (actual count, possibly < requested)

**`method=1` (parallelogramic, `Gsel_parallelogramic`, lines 49–78):**
1. `NGroot = ⌊√nG⌋` rounded down to **odd** integer
2. `M = NGroot // 2`
3. Generate `NGroot × NGroot` grid centered on origin (`-M..M`)
4. Sort by `|G|²`, take first `NGroot²` rows

#### 5.1.3 `Lattice_SetKs(G, kx0, ky0, Lk1, Lk2) → (kx, ky)`

```cpp
// kbloch.py:43-44
for (int i = 0; i < nG; ++i) {
    kx[i] = kx0 + 2*M_PI*(Lk1[0]*G(i,0) + Lk2[0]*G(i,1));
    ky[i] = ky0 + 2*M_PI*(Lk1[1]*G(i,0) + Lk2[1]*G(i,1));
}
```

Note: `kx0, ky0` are **complex** (they include the complex `omega`).

#### 5.1.4 Verification

For each of square, hexagonal (`ex4` uses angle=π/3), oblique, and rectangular lattices:
- Compare `Lk1, Lk2` against `grcwa.Lattice_Reciprocate`
- Compare `G` and `nG_out` against `grcwa.Lattice_getG`
- Compare `kx, ky` against `grcwa.Lattice_SetKs`
- **Special test:** confirm `nG_out <= nG` (assertion in `test_kbloch.py:25`)

---

### 5.2 `fft_funs.h / fft_funs.cpp` — Medium Complexity

FFT-based operations for patterned layers. Depends on **FFTW 3.x** (or cuFFT for GPU).

#### 5.2.1 Plan Cache (Critical for Performance)

FFTW plans are expensive to create (~10-100ms each). Create once, reuse for all layers with the same `(Nx, Ny)` resolution:

```cpp
class FFTWPlanCache {
public:
    // Returns (forward_plan, backward_plan) for given Nx, Ny
    // Internally stores a std::map<std::pair<int,int>, std::pair<fftw_plan, fftw_plan>>
    std::pair<fftw_plan, fftw_plan> get(int Nx, int Ny);

    // Cleans up all plans on destruction
    ~FFTWPlanCache();
private:
    std::map<std::pair<int,int>, std::pair<fftw_plan, fftw_plan>> plans_;
};
```

The `RCWA` class owns a `static` or `thread_local` plan cache to share across multiple simulations.

#### 5.2.2 `get_conv(dN, s_in[Nx*Ny], G) → ComplexMatrix (nG×nG)`

From `fft_funs.py:33-47`:

```cpp
// 1. Reshape flat s_in into (Nx, Ny) row-major grid (see "Eigen ↔ FFTW" note above)
GridMatrix s_grid = Eigen::Map<const GridMatrix>(s_in.data(), Nx, Ny);

// 2. FFT2 with FFTW (NEW plan, then fftw_execute)
//    FFTW outputs complex values; need to apply dN = 1/(Nx*Ny) factor manually
GridMatrix s_fft = fftw_fft2(s_grid);  // wrapper around fftw_plan_dft_2d (row-major in/out)
s_fft *= dN;

// 3. Build nG×nG convolution matrix via index gather
//    C[i,j] = s_fft[(G[i,0]-G[j,0]) % Nx, (G[i,1]-G[j,1]) % Ny]
//    Note: NumPy fft2 wraps with negative indices, FFTW uses raw 0..N-1.
//    Use modular arithmetic with proper wrapping for negative differences.
ComplexMatrix s_conv(nG, nG);
for (int i = 0; i < nG; ++i) {
    for (int j = 0; j < nG; ++j) {
        int row = ((G(i,0) - G(j,0)) % Nx + Nx) % Nx;
        int col = ((G(i,1) - G(j,1)) % Ny + Ny) % Ny;
        s_conv(i, j) = s_fft(row, col);
    }
}
return s_conv;
```

**Gotcha — Eigen ↔ FFTW memory layout:** `fftw_plan_dft_2d(Nx, Ny, in, out, sign, flags)` treats its input as a **row-major** `(Nx, Ny)` array with strides `(Ny, 1)` in element units. Eigen's default `MatrixXcd` is **column-major** with strides `(1, Nx)`. Passing `s_grid.data()` directly to FFTW will therefore transpose/permute the data.

**Convention (pick one, document it, use it everywhere):**
- *Option A (recommended):* typedef a row-major grid type and use it for all real-space grids:
  ```cpp
  using GridMatrix = Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  ```
  Then `Eigen::Map<GridMatrix>(s_in.data(), Nx, Ny)` is contiguous and can be handed to FFTW directly.
- *Option B:* keep `MatrixXcd` (column-major) for grids and add an explicit transpose copy before/after every FFTW call.

This document assumes **Option A** for all grid representations in `fft_funs`, `Solve_FieldOnGrid`, `Return_eps`, etc. Confirm the `(row=i=G_x, col=j=G_y)` indexing against the Python reference before validating.

#### 5.2.3 `get_fft(dN, s_in[Nx*Ny], G) → ComplexVector (nG)`

Trivial: `sfft = fft2(s_in) * dN; return sfft[G[:,0], G[:,1]]`.

#### 5.2.4 `get_ifft(Nx, Ny, s_in[nG], G) → ComplexMatrix (Nx×Ny)`

From `fft_funs.py:62-79`. Implementation:

```cpp
// 1. Place s_in[i] at position G[i,0], G[i,1] in a zero (Nx, Ny) matrix
Eigen::MatrixXcd s0 = Eigen::MatrixXcd::Zero(Nx, Ny);
for (int i = 0; i < nG; ++i) {
    int r = ((G(i,0) % Nx) + Nx) % Nx;
    int c = ((G(i,1) % Ny) + Ny) % Ny;
    s0(r, c) = s_in[i];
}

// 2. IFFT2 with FFTW, then divide by dN
Eigen::MatrixXcd s_out = fftw_ifft2(s0);
s_out /= dN;  // FFTW doesn't normalize; numpy ifft2 divides by N
return s_out;
```

#### 5.2.5 `Epsilon_fft(dN, eps_grid, G) → (epsinv, eps2)`

From `fft_funs.py:3-31`. Two cases (detected by `eps_grid` shape):

**Isotropic** (`eps_grid.ndim == 1`, i.e., flat 1D array of length `Nx*Ny`):

```cpp
ComplexMatrix eps_fft = get_conv(dN, eps_grid, G);   // nG × nG
ComplexMatrix epsinv = eps_fft.inverse();              // LAPACK zgetrf + zgetri
// eps2 = block_diag(eps_fft, eps_fft), shape (2nG, 2nG)
ComplexMatrix eps2 = ComplexMatrix::Zero(2*nG, 2*nG);
eps2.topLeftCorner(nG, nG)    = eps_fft;
eps2.bottomRightCorner(nG, nG) = eps_fft;
```

**Anisotropic** (`eps_grid` is list of 3 flat arrays of length `Nx*Ny`):

```cpp
ComplexMatrix epsx_fft = get_conv(dN, eps_grid[0], G);
ComplexMatrix epsy_fft = get_conv(dN, eps_grid[1], G);
ComplexMatrix epsz_fft = get_conv(dN, eps_grid[2], G);
ComplexMatrix epsinv = epsz_fft.inverse();
// eps2 = block_diag(epsx_fft, epsy_fft), shape (2nG, 2nG)
ComplexMatrix eps2 = ComplexMatrix::Zero(2*nG, 2*nG);
eps2.topLeftCorner(nG, nG)    = epsx_fft;
eps2.bottomRightCorner(nG, nG) = epsy_fft;
```

**No examples exercise the anisotropic path** (only the isotropic branch). Port anisotropic as well, but defer full validation until a test case is added.

#### 5.2.6 Verification

- **Round-trip test:** `ifft(fft(real_grid)) ≈ real_grid` — confirms FFT conventions match
- **Isotropic vs anisotropic:** for `epsx == epsy == epsz`, both branches must give identical `eps2` and `epsinv`
- **Convolution matrix check:** Verify `eps_fft[0,0]` (the DC component) equals the spatial average of `eps_grid`
- Compare against Python output for circle, square, checkerboard permittivity patterns

---

### 5.3 `rcwa.h / rcwa.cpp` — High Complexity

Main engine. Depends on BLAS, LAPACK (for `zgeev`, `zgetrf`+`zgetri`), and the other cpprcwa modules.

#### 5.3.1 Class: `RCWA`

```cpp
class RCWA {
public:
    explicit RCWA(const RCWAConfig& config);
    ~RCWA();

    // ── Layer specification (call in order, BEFORE Init_Setup) ──
    void Add_LayerUniform(double thickness, complex epsilon);
    void Add_LayerGrid(double thickness, int Nx, int Ny);
    void Add_LayerFourier(double thickness, const std::vector<complex>& params); // stub

    // ── Initialization (must call before any solve) ──
    void Init_Setup(double Pscale = 1.0, int Gmethod = 0);

    // ── Excitation ──
    void MakeExcitationPlanewave(const PlaneWaveExcitation& exc);

    // ── Material specification for patterned layers (call between Init_Setup and solve) ──
    // ep_all is a flat array containing ALL patterned layers concatenated:
    //   [layer0_grid(Nx0*Ny0), layer1_grid(Nx1*Ny1), ...]
    // For anisotropic, pass a vector of 3 such arrays.
    void GridLayer_geteps(const std::vector<complex>& ep_all_isotropic);
    void GridLayer_geteps(const std::vector<std::vector<complex>>& ep_all_anisotropic_3);

    // ── Main solves ──
    RTResult RT_Solve(bool normalize = false, bool byorder = false);

    std::pair<ComplexVector, ComplexVector>
    GetAmplitudes(int which_layer, double z_offset);

    std::pair<ComplexVector, ComplexVector>
    GetAmplitudes_noTranslate(int which_layer);

    std::vector<FieldFourier>
    Solve_FieldFourier(int which_layer, const std::vector<double>& z_offsets);
    std::vector<FieldFourier>
    Solve_FieldFourier(int which_layer, double z_offset);

    std::vector<FieldGrid>
    Solve_FieldOnGrid(int which_layer, const std::vector<double>& z_offsets,
                      std::optional<std::array<int,2>> Nxy = std::nullopt);

    ComplexMatrix Return_eps(int which_layer, int Nx, int Ny,
                             const std::string& component); // 'xx','xy','yx','yy','zz'

    // Volume integral of M·|E|² for absorbed power
    complex Volume_integral(int which_layer,
                            const ComplexMatrix& Mx,
                            const ComplexMatrix& My,
                            const ComplexMatrix& Mz,
                            bool normalize = false);

    // Maxwell stress tensor integral → (2Fx, 2Fy, 2Fz)
    std::array<double, 3> Solve_ZStressTensorIntegral(int which_layer);

    // ── Accessors (read-only inspection, useful for validation) ──
    int nG() const { return nG_; }
    int Layer_N() const { return layers_.size(); }
    const IntMatrix& G() const { return G_; }
    const ComplexVector& kx() const { return kx_; }
    const ComplexVector& ky() const { return ky_; }
    const std::vector<double>& thickness_list() const { return thickness_; }
    const std::vector<ComplexVector>& q_list() const { return q_list_; }
    const std::vector<ComplexMatrix>& phi_list() const { return phi_list_; }
    const std::vector<ComplexMatrix>& kp_list() const { return kp_list_; }
    double normalization() const { return normalization_; }

private:
    // ... (member fields, see §5.3.2)
};
```

#### 5.3.2 Internal State

```cpp
// Configuration & problem
RCWAConfig config_;
int nG_;                          // ACTUAL truncation (from getG, may be < requested)
complex freq_, omega_;            // omega = 2*pi*freq (BOTH complex — see §6.1)
Eigen::Vector2d L1_, L2_;          // Direct lattice (see §6.12)
double theta_, phi_;              // Incidence angles
double normalization_;            // sqrt(Uniform_ep[0]) / cos(theta), used if normalize=true
Direction direction_;             // 'forward' or 'backward'
double Pscale_ = 1.0;             // Set by Init_Setup()
int Gmethod_ = 0;                 // Set by Init_Setup()

// Layer storage
std::vector<LayerType> layer_types_;
std::vector<double>    thickness_;
std::vector<complex>   uniform_eps_;       // Index = uniform_count
std::vector<std::array<int,2>> grid_Nxy_;  // Index = grid_count
std::vector<std::vector<complex>> fourier_params_; // stub

// Computed layer data (per layer index, length = Layer_N)
std::vector<ComplexMatrix> kp_list_;       // 2nG × 2nG each
std::vector<ComplexVector> q_list_;        // 2nG each
std::vector<ComplexMatrix> phi_list_;      // 2nG × 2nG each

// Per-patterned-layer (indexed by patterned_count, NOT by Layer_N)
std::vector<ComplexMatrix> patterned_epinv_; // nG × nG each
std::vector<ComplexMatrix> patterned_ep2_;   // 2nG × 2nG each

// Reciprocal lattice & k-vectors
IntMatrix G_;                              // nG × 2 ints
Eigen::Vector2d Lk1_, Lk2_;                // (see §6.12)
ComplexVector kx_, ky_;                    // length nG each

// Excitation
ComplexVector a0_, bN_;                    // length 2nG each

// Internal counters
int uniform_count_ = 0;
int grid_count_    = 0;
int fourier_count_ = 0;
```

#### 5.3.3 Core Mathematical Functions (module-level in `rcwa.cpp`)

| Function | Math (from rcwa.py) | LAPACK |
|---|---|---|
| `MakeKPMatrix(omega, layer_type, epinv, kx, ky)` | `rcwa.py:437-455`<br>Uniform: `Jk = vstack(diag(-ky), diag(kx))` → `Kp = ω²·I - epinv·Jk·Jkᵀ`<br>Patterned: `Kp = ω²·I - (Jk·epinv)·Jkᵀ` | `zgemm` only |
| `SolveLayerEigensystem_uniform(omega, kx, ky, eps)` | `rcwa.py:457-465`<br>`q = sqrt(eps·ω² - kx² - ky²)`, branch cut `Im(q)<0 ⇒ -q`, then `q = concat(q,q)`, `phi = I_(2nG)` | None |
| `SolveLayerEigensystem(omega, kx, ky, kp, ep2)` | `rcwa.py:467-479`<br>`k = vstack(diag(kx), diag(ky))`, `kkT = k·kᵀ`, `M = ep2·kp - kkT`, `[evals, evecs] = eig(M)`, `q = sqrt(evals)` with same branch cut | **`zgeev`** (2nG × 2nG, non-Hermitian) |
| `GetSMatrix(indi, indj, q_list, phi_list, kp_list, thickness_list)` | `rcwa.py:481-537` (Redheffer star product, see §5.3.4) | `zgemm` + **`zgetrf`+`zgetri`** (one inversion per layer pair) |
| `SolveExterior(a0, bN, ...)` | `rcwa.py:539-550`<br>`[aN, b0] = S_full · [a0; bN]` | `zgemm` only |
| `SolveInterior(which_layer, a0, bN, ...)` | `rcwa.py:552-570`<br>Left/right S-matrices, then `tmp = inv(I - S12·pS21)`, `ai = tmp·(S11·a0 + S12·pS22·bN)`, `bi = pS21·ai + pS22·bN` | `zgetrf`+`zgetri` + `zgemm` |
| `TranslateAmplitudes(q, thickness, dz, ai, bi)` | `rcwa.py:572-575`<br>`aim = ai·exp(i·q·dz)`, `bim = bi·exp(i·q·(thickness-dz))` | Element-wise |
| `GetZPoyntingFlux(ai, bi, omega, kp, phi, q, byorder)` | `rcwa.py:578-605` (see §5.3.5) | `zgemm` only |
| `Matrix_zintegral(q, thickness, shift=1e-12)` | `rcwa.py:607-639` (see §5.3.6) | Element-wise |
| `Gmeshgrid(x)` | `rcwa.py:641-648`<br>`qj = tile(x, nG)`, `qi = transpose(qj)` | None |

#### 5.3.4 S-Matrix Algorithm (Critical Path — Port Carefully)

From `rcwa.py:481-537`. Sizes: each block is `2nG × 2nG`.

```cpp
void GetSMatrix(int indi, int indj,
                const std::vector<ComplexVector>& q_list,
                const std::vector<ComplexMatrix>& phi_list,
                const std::vector<ComplexMatrix>& kp_list,
                const std::vector<double>& thickness_list,
                ComplexMatrix& S11, ComplexMatrix& S12,
                ComplexMatrix& S21, ComplexMatrix& S22)
{
    int nG2 = q_list[0].size();    // = 2 * nG
    assert(indi <= indj);
    if (indi == indj) {
        S11 = S22 = ComplexMatrix::Identity(nG2, nG2);
        S12 = S21 = ComplexMatrix::Zero(nG2, nG2);
        return;
    }

    for (int l = indi; l < indj; ++l) {
        int lp1 = l + 1;

        // Q = inv(phi_l) * phi_{l+1}
        ComplexMatrix Q = phi_list[l].inverse() * phi_list[lp1];

        // P = diag(q_l) * inv(kp_l * phi_l) * kp_{l+1} * phi_{l+1} * diag(1/q_{l+1})
        ComplexMatrix P =
            q_list[l].asDiagonal() *
            (kp_list[l] * phi_list[l]).inverse() *
            kp_list[lp1] * phi_list[lp1] *
            q_list[lp1].cwiseInverse().asDiagonal();

        // T11 = T22, T12 = T21
        ComplexMatrix T11 = 0.5 * (Q + P);
        ComplexMatrix T12 = 0.5 * (Q - P);

        // Phase factors (diagonal matrices)
        // NOTE: .array().exp() is ELEMENT-WISE exp on the vector, then .asDiagonal().
        //       Do NOT call .exp() directly on a MatrixBase — Eigen's unsupported
        //       MatrixFunctions module interprets that as a MATRIX exponential.
        //       Equivalently: exp(diag(x)) == diag(exp(x)) when exp is element-wise.
        ComplexMatrix d1 = (complex(0,1) * q_list[l]   * thickness_list[l]  ).array().exp().asDiagonal();
        ComplexMatrix d2 = (complex(0,1) * q_list[lp1] * thickness_list[lp1]).array().exp().asDiagonal();

        // Redheffer star product:
        // S11 = inv(T11 - d1·S12·T12) · d1 · S11_old
        ComplexMatrix P1 = T11 - d1 * S12 * T12;
        P1 = P1.inverse();
        S11 = P1 * d1 * S11;

        // S12 = P1 · (d1·S12·T11 - T12) · d2
        ComplexMatrix P2 = d1 * S12 * T11 - T12;
        S12 = P1 * P2 * d2;

        // S21 = S21 + S22·T12·S11_new
        S21 = S21 + S22 * T12 * S11;

        // S22 = S22·T12·S12 + S22·T11·d2
        P2 = S22 * T12 * S12;
        P1 = S22 * T11 * d2;
        S22 = P1 + P2;
    }
}
```

**Memory optimization:** Each iteration creates ~5 new `2nG×2nG` matrices. For nG=500, each matrix is 8 MB (8 bytes × 1000 × 1000). Use Eigen's `.noalias()` and pre-allocate workspace buffers to avoid allocator pressure. Consider using raw `cblas_zgemm` + `LAPACKE_zgetrf/zgetri` for finer control.

#### 5.3.5 Poynting Flux (`GetZPoyntingFlux`, rcwa.py:578-605)

```cpp
// Returns 2*S_z/A as a (forward, backward) pair.
// Two overloads: scalar sum vs per-order vector — matches Python's byorder flag.
// (Do NOT use std::variant here — the caller always knows which it wants.)

// byorder=false → scalar result
std::pair<complex, complex> GetZPoyntingFlux(
    const ComplexVector& ai, const ComplexVector& bi,
    complex omega, const ComplexMatrix& kp,
    const ComplexMatrix& phi, const ComplexVector& q);

// byorder=true → per-order vectors of length nG (tag-dispatched)
struct ByOrderTag {};
std::pair<ComplexVector, ComplexVector> GetZPoyntingFlux(
    const ComplexVector& ai, const ComplexVector& bi,
    complex omega, const ComplexMatrix& kp,
    const ComplexMatrix& phi, const ComplexVector& q,
    ByOrderTag);

// ── Shared computation (both overloads delegate to this) ──
// Returns [forward_xy (len 2n), backward_xy (len 2n)]; caller sums or splits.
std::pair<ComplexVector, ComplexVector> GetZPoyntingFlux_impl(
    const ComplexVector& ai, const ComplexVector& bi,
    complex omega, const ComplexMatrix& kp,
    const ComplexMatrix& phi, const ComplexVector& q)
{
    int n2 = ai.size();
    int n  = n2 / 2;

    // A = kp · phi · diag(1/(omega·q))
    ComplexMatrix A = kp * phi * (1.0 / (omega * q.array()).matrix()).asDiagonal();

    ComplexVector pa = phi * ai;
    ComplexVector pb = phi * bi;
    ComplexVector Aa = A * ai;
    ComplexVector Ab = A * bi;

    // diff = 0.5 * (conj(pb) * Aa - conj(Ab) * pa)   (element-wise)
    ComplexVector diff = 0.5 * (pb.conjugate().cwiseProduct(Aa)
                              - Ab.conjugate().cwiseProduct(pa));

    // forward_xy = real(conj(Aa) * pa) + diff
    ComplexVector forward_xy = Aa.conjugate().cwiseProduct(pa).real() + diff;
    // backward_xy = -real(conj(Ab) * pb) + conj(diff)
    ComplexVector backward_xy = -Ab.conjugate().cwiseProduct(pb).real() + diff.conjugate();

    return {forward_xy, backward_xy};
}

// Scalar overload: sum the xy + z halves.
std::pair<complex, complex> GetZPoyntingFlux(...) {
    auto [fwd, bwd] = GetZPoyntingFlux_impl(ai, bi, omega, kp, phi, q);
    int n = fwd.size() / 2;
    return { fwd.head(n).sum() + fwd.tail(n).sum(),
             bwd.head(n).sum() + bwd.tail(n).sum() };
}

// Per-order overload: sum the two halves element-wise (no reduction).
std::pair<ComplexVector, ComplexVector> GetZPoyntingFlux(..., ByOrderTag) {
    auto [fwd, bwd] = GetZPoyntingFlux_impl(ai, bi, omega, kp, phi, q);
    int n = fwd.size() / 2;
    return { fwd.head(n) + fwd.tail(n),
             bwd.head(n) + bwd.tail(n) };
}
```

**Return-type clarification:** The Python `GetZPoyntingFlux` returns `(forward, backward)` where each is a scalar if `byorder=0` and a length-nG vector if `byorder=1`. The C++ port uses **two overloads** (tag dispatch via `std::true_type` for the by-order form) rather than `std::variant`, so the caller's type encodes the flag and there is no runtime branch. The scalar overload drops the `if (byorder)` block; the vector overload drops the scalar sum.

#### 5.3.6 Volume Integral Matrix (`Matrix_zintegral`, rcwa.py:607-639)

Critical detail I missed: `shift=1e-12` is added to the diagonal to **avoid division by zero** when `q_i = conj(q_j)` (diagonal elements).

```cpp
ComplexMatrix Matrix_zintegral(const ComplexVector& q, double thickness,
                               double shift = 1e-12)
{
    int nG2 = q.size();
    // qi[i,j] = q[i], qj[i,j] = q[j]
    ComplexMatrix qi = q.replicate(1, nG2);              // column-tile
    ComplexMatrix qj = q.transpose().replicate(nG2, 1);  // row-tile

    // Maa[i,j] = (exp(i*(qj - conj(qi))*t) - 1) / (i*(qj - conj(qi)))  with shift on diagonal
    ComplexMatrix qij = qj - qi.conjugate();
    qij.diagonal().array() += shift;          // <- crucial stability term
    ComplexMatrix Maa = (((qij * thickness * complex(0,1)).array().exp() - 1.0)
                         / (complex(0,1) * qij).array()).matrix();

    // Mab[i,j] = (exp(i*qj*t) - exp(-i*conj(qi)*t)) / (i*(qj + conj(qi)))
    ComplexMatrix qij2 = qj + qi.conjugate();
    ComplexMatrix Mab = (((qj * thickness * complex(0,1)).array().exp()
                        - (-qi.conjugate() * thickness * complex(0,1)).array().exp())
                         / (complex(0,1) * qij2).array()).matrix();

    // Mt = [[Maa, Mab], [Mab, Maa]], shape (2*nG2, 2*nG2); each block is (nG2, nG2).
    // (nG2 itself is already 2*nG, so Mt is 4*nG × 4*nG — see note below.)
    ComplexMatrix Mt(2*nG2, 2*nG2);
    Mt.topLeftCorner(nG2, nG2)     = Maa;
    Mt.topRightCorner(nG2, nG2)    = Mab;
    Mt.bottomLeftCorner(nG2, nG2)  = Mab;
    Mt.bottomRightCorner(nG2, nG2) = Maa;
    return Mt;
}
```

**Note:** `nG2` here is **already 2nG** (it's `q.size()` from the layer's `q_list[which_layer]`). So `Mt` is `4nG × 4nG`.

#### 5.3.7 Field Reconstruction (`Solve_FieldFourier`, rcwa.py:282-319)

Important details:
- `z_offset` can be **scalar or list** — handle both
- Returns a **list** of `[[ex, ey, ez], [hx, hy, hz]]`, one per `z_offset`
- `E_z` division:
  - Uniform: `fez /= epsilon_uniform`
  - Patterned: `fez = Patterned_epinv · fez` (matrix-vector product)
- `E_z` numerator: `(ky·hx - kx·hy) / ω`
- `H_z` numerator: `(ky·ex - kx·ey) / ω`
- Sign flip: `fey = -fexy[:nG]`, `fex = fexy[nG:]` — this is **not a typo**, the (x,y) ↔ (Ex,Ey) ordering in the amplitude vector is non-standard.

#### 5.3.8 Permittivity Reconstruction (`Return_eps`, rcwa.py:192-216)

- `'zz'`: `inv(epsinv_matrix)` via `Epsilon_fft`-stored matrix (it's already stored as the inverse, so invert it back)
- `'xx'`: top-left `nG×nG` block of `Patterned_ep2`
- `'yy'`: bottom-right block
- `'xy'`: top-right block
- `'yx'`: bottom-left block
- Returns the real-space permittivity via `get_ifft`

Uniform layers: returns `epsilon_uniform * ones((Nx, Ny))`.

#### 5.3.9 Volume Integral (`Volume_integral`, rcwa.py:350-395)

Steps:
1. `ab = hstack(ai, bi)` (length 4nG)
2. `abMatrix = outer(ab, conj(ab))` (4nG × 4nG)
3. `Mt = Matrix_zintegral(q, thickness)` (4nG × 4nG)
4. `abM = abMatrix.cwiseProduct(Mt)` (Hadamard)
5. `Faxy = kp · phi · diag(1/(ω·q))` (2nG × 2nG)
6. `Faz1 = (1/ω) · epinv · diag(ky)`, `Faz2 = -(1/ω) · epinv · diag(kx)`, `Faz = hstack(Faz1, Faz2) · phi` (2nG × 2nG)
7. `tmp1 = vstack(Faxy, Faz)`, `tmp2 = vstack(-Faxy, Faz)`, `F = hstack(tmp1, tmp2)` (4nG × 4nG)
8. `Mtotal` is 12nG × 12nG block-diagonal with `Mx, My, Mz`
9. `val = trace(abM · conj(F)ᵀ · Mtotal · F)`

For uniform layers, `epinv = 1/epsilon_uniform` (scalar).

#### 5.3.10 Stress Tensor Integral (`Solve_ZStressTensorIntegral`, rcwa.py:397-435)

Computes `D = ε·E`:
- `Dz = (ky·hx - kx·hy)/ω` (this is just `(∇ × H)|_z`, but here computed directly)
- For uniform: `dx = ex·epsilon`, `dy = ey·epsilon`
- For patterned: `exy = hstack(-ey, ex)`, `dxy = eps2 · exy`, `dx = dxy[nG:]`, `dy = -dxy[:nG]`

Then:
- `Tx = Re(Σ(ex·conj(dz) + hx·conj(hz)))`
- `Ty = Re(Σ(ey·conj(dz) + hy·conj(hz)))`
- `Tz = Re(Σ(0.5·(ez·conj(dz) + hz·conj(hz) - ey·conj(dy) - ex·conj(dx) - |hx|² - |hy|²)))`

Returns `(Tx, Ty, Tz)` — these are `2·F_x`, `2·F_y`, `2·F_z` (per the docstring).

#### 5.3.11 Verification

Compare against Python `grcwa` for:
- `RT_Solve(normalize=0)` for the S4 test config: must get `T=0.85249901083265` (p-pol) and `T=0.83900479939861` (s-pol) at relative tolerance `1e-3`
- `RT_Solve(normalize=0, byorder=1)` for ex1/ex4 configurations
- `Solve_FieldOnGrid(layer=1, z_offset=0)` output shapes and magnitudes
- `Volume_integral(1, Mx, Mx, Mx, normalize=1)` — must be positive real
- `Solve_ZStressTensorIntegral(0)` — `Tz` must be negative

---

## 6. Special Behaviors to Preserve Exactly

### 6.1 Complex omega with Qabs (rcwa.py:19, ex1.py:16, ex4.py:17, ex3.py:51)

```python
self.omega = 2*bd.pi*freq + 0.j        # complex
# In examples:
freqcmp = freq * (1 + 1j/2/Qabs)        # complex shift to avoid singular matrices
obj = grcwa.obj(nG, L1, L2, freqcmp, theta, phi, ...)
```

When `Qabs = inf`, this is just `freq`. When finite, adds tiny imaginary part for regularization. **The C++ `RCWA` constructor must accept a complex frequency** — `RCWAConfig::freq` should be `complex`, not `double`.

### 6.2 Layer 0 Must Be Uniform (rcwa.py:89-103)

`Init_Setup` does:
```python
kx0 = self.omega * sin(theta) * cos(phi) * sqrt(self.Uniform_ep_list[0])
```

It assumes `Uniform_ep_list[0]` exists. **Enforce** by assertion: the first layer added must be uniform.

### 6.3 `Add_LayerGrid` Stores `(Nx, Ny)`, Not Yet Resolved

Layer `Nx, Ny` is recorded at `Add_LayerGrid` time. The actual `epgrid` data is supplied later via `GridLayer_geteps(ep_all)`, where `ep_all` is a **flat concatenation** of all patterned layer grids in order:

```python
# ex2.py:62-64
epgrid = np.concatenate((epgrid1.flatten(), epgrid2.flatten()))
obj.GridLayer_geteps(epgrid)
```

**C++ API must accept this flat layout**, or alternatively accept a vector of vectors and flatten internally.

### 6.4 `Init_Setup` Parameters

```python
obj.Init_Setup(Pscale=1.0, Gmethod=0)  # Pscale is keyword, Gmethod is keyword
```

- `Pscale`: divides `Lk1` and `Lk2` by this value (effectively multiplies the period). Used in `test_rcwa.py:121-132` for `test_periodgrad`.
- `Gmethod`: 0 = circular, 1 = parallelogramic.

### 6.5 Plane-Wave Direction

`'forward'` vs `'backward'` is a string in Python (rcwa.py:125, 134, 144). In C++ use the `PlaneWaveExcitation::Direction` enum. The direction affects sign convention in `RT_Solve`:
- Forward: `R = real(-bi)`, `T = real(fe)`
- Backward: `R = real(fe)`, `T = real(-bi)`

### 6.6 Eigenvalue Branch Cut (rcwa.py:461, 478)

After every `sqrt` of eigenvalues:
```python
q = bd.where(bd.imag(q) < 0., -q, q)
```

This forces the imaginary part of `q` to be **non-negative** (evanescent waves decay in +z direction). Same convention must be applied in C++:

```cpp
// Apply branch cut: Im(q) < 0 → negate
Eigen::VectorXcd apply_branch_cut(const Eigen::VectorXcd& q) {
    Eigen::VectorXcd result = q;
    for (int i = 0; i < q.size(); ++i) {
        if (result(i).imag() < 0.0) result(i) = -result(i);
    }
    return result;
}
```

### 6.7 Eigenvector Ordering (Unstable)

`np.linalg.eig` and `zgeev` both return eigenvalues in arbitrary order, with eigenvectors of arbitrary phase. The S-matrix algorithm **does not depend on ordering**, but the field reconstruction **does** because `q_list[i]` and `phi_list[i]` are paired by index.

**Strategy:** Compare **physically observable** quantities (`|E|²`, `|H|²`, `R`, `T`, `Tx`, `Ty`, `Tz`) rather than raw complex amplitudes. The S4 golden reference values are observable quantities.

### 6.8 Eigenvalue in `Matrix_zintegral` and Other Places

The diagonal `shift=1e-12` (rcwa.py:628) is **only** applied to `qij = qj - conj(qi)`, not `qij2 = qj + conj(qi)`. Preserve this asymmetry exactly.

### 6.9 `fey = -fexy[:nG]` Sign Convention (rcwa.py:306-307)

```python
tmp1 = (ai-bi)/self.omega/self.q_list[which_layer]
tmp2 = bd.dot(self.phi_list[which_layer], tmp1)
fexy = bd.dot(self.kp_list[which_layer], tmp2)
fey = - fexy[:self.nG]      # Note the minus sign
fex = fexy[self.nG:]
```

The minus sign is **not a typo** — it's because the amplitude-vector ordering convention `(a_x, a_y)` (first half) maps to `E = (E_y, -E_x)` (or similar). Document this carefully and add a test that catches sign errors.

### 6.10 `nG2 = 2*nG` is the Block Size

The amplitude vectors (`a0`, `bN`, `ai`, `bi`) and S-matrix blocks are all **length `2*nG`**, NOT `nG`. This is because each Fourier mode has 2 polarizations (E_x, E_y) → 2nG DOFs total.

### 6.11 FFTW Plan Threading (Phase 10 Hazard)

Executing the **same** `fftw_plan` from multiple OpenMP threads simultaneously is unsafe unless FFTW was built with threads support and you called `fftw_plan_with_nthreads(N)` before plan creation. The plan cache (§5.2.1) must therefore:
1. On construction, call `fftw_make_planner_thread_safe()` (FFTW ≥ 3.3.4) **and** `fftw_plan_with_nthreads(omp_get_max_threads())`.
2. Require the build to link `libfftw3_threads` (`find_library(FFTW3_THREADS ...)`).
3. Document that thread-parallel plan execution is opt-in via `-DCPPRCWA_FFTW_THREADS=ON`.

If FFTW threads support is unavailable, the OpenMP layer in Phase 10 must skip parallelizing any loop that calls `get_conv` / `get_ifft`, or each thread must hold its own plan cache (`thread_local`).

### 6.12 Lattice-Vector Type — Use One Representation

`std::array<double,2>` appears in `RCWAConfig` (L1, L2) while `Eigen::Vector2d` is used in `kbloch` (`Lattice_Reciprocate`, `Lattice_SetKs`, etc.). This forces conversions at every call boundary. Pick **`Eigen::Vector2d`** as the canonical type everywhere (config included) and update `RCWAConfig::L1`/`L2` and `Lattice_Reciprocate`'s signature accordingly. The public kbloch API should then take `const Eigen::Vector2d&` rather than `const std::array<double,2>&`.

---

## 7. Phase-by-Phase Implementation Plan

### Phase 1: Scaffolding & Build System — ✅ DONE
- [ ] Set up `CMakeLists.txt` with C++17, Eigen, FFTW, LAPACK
- [ ] Implement `include/cpprcwa/types.h` with all structs/enums
- [ ] Stub `include/cpprcwa/{kbloch,fft_funs,rcwa}.h` with empty implementations
- [ ] Set up `tests/CMakeLists.txt`, `examples/CMakeLists.txt`
- [ ] Run a "hello world" example that links all libraries
- [ ] Add a CI script (GitHub Actions or shell) that runs build + tests

### Phase 2: `kbloch` — ✅ DONE
- [ ] Port `Lattice_Reciprocate` (15 min)
- [ ] Port `Gsel_parallelogramic` (1 hr)
- [ ] Port `Gsel_circular` with degeneracy handling (2 hr) — **carefully preserve the tolerance logic at lines 119–124**
- [ ] Port `Lattice_getG` dispatcher (15 min)
- [ ] Port `Lattice_SetKs` (15 min)
- [ ] **Test:** square lattice (ex1), hexagonal lattice (ex4, angle=π/3), oblique lattice (random L1, L2)
- [ ] **Test:** `nG_out <= nG_in` assertion
- [ ] **Test:** degeneracy-symmetric output (verify `(-G[i,0], -G[i,1])` is present whenever `(G[i,0], G[i,1])` is, except for G=0)

### Phase 3: FFT Utilities — ✅ DONE
- [ ] Implement `FFTWPlanCache` with thread-safe plan lookup (1 day)
- [ ] Port `get_conv`, `get_fft`, `get_ifft` (1 day)
- [ ] Port `Epsilon_fft` (isotropic first, then anisotropic) (1 day)
- [ ] **Test:** `ifft(fft(real_grid)) ≈ real_grid` (FFT convention check)
- [ ] **Test:** Isotropic `Epsilon_fft` against Python for circle/square/checkerboard patterns
- [ ] **Test:** Anisotropic case (defer until needed)
- [ ] **Test:** Convolution matrix DC component equals spatial average

### Phase 4: RCWA Engine — Layer Setup — ✅ DONE
- [ ] Constructor + state initialization
- [ ] `Add_LayerUniform`, `Add_LayerGrid`, `Add_LayerFourier` (stub)
- [ ] `Init_Setup`: reciprocal lattice, `kx0/ky0` from layer-0 epsilon, eigendecomp uniform layers (2 days)
- [ ] `MakeKPMatrix`, `SolveLayerEigensystem_uniform` (in `rcwa.cpp`)
- [ ] `MakeExcitationPlanewave`
- [ ] `GridLayer_geteps` for isotropic (most common)
- [ ] `SolveLayerEigensystem` (first `zgeev` usage) (1 day)
- [ ] **Test:** Compare `kp_list`, `q_list`, `phi_list` against Python for ex1 and ex4 layer setups

### Phase 5: RCWA Engine — S-Matrix Solver — ✅ DONE (S4 golden values reproduced)
- [ ] `GetSMatrix` iterative star product (2 days) — this is the critical path
- [ ] `SolveExterior`, `SolveInterior` (1 day)
- [ ] `TranslateAmplitudes` (15 min)
- [ ] `GetZPoyntingFlux` — handle scalar vs byorder return (1 day)
- [ ] `RT_Solve` driver (15 min)
- [ ] **Test:** **S4 golden values** — p-pol `T=0.85249901083265`, s-pol `T=0.83900479939861` (the key acceptance criterion)
- [ ] **Test:** R+T = 1 within `1e-6` for lossless cases
- [ ] **Test:** Compare per-order R/T arrays against Python

### Phase 6: RCWA Engine — Field Reconstruction — ✅ DONE
- [ ] `GetAmplitudes_noTranslate`, `GetAmplitudes` (15 min)
- [ ] `Solve_FieldFourier` with scalar and list `z_offset` (1 day)
- [ ] `Solve_FieldOnGrid` (depends on `get_ifft`) (15 min)
- [ ] **Test:** Compare field shapes `(nG,)` and `(Nx, Ny)`
- [ ] **Test:** Compare `|E|²` summed over grid against `Volume_integral(M=I)` scaled appropriately
- [ ] **Test:** Sign convention — `fey = -fexy[:nG]` produces correct field symmetry

### Phase 7: RCWA Engine — Post-processing — ✅ DONE
- [ ] `Return_eps` (1 hr)
- [ ] `Matrix_zintegral` with diagonal shift (1 day) — **tricky due to `Gmeshgrid` pattern**
- [ ] `Volume_integral` (3 hr)
- [ ] `Solve_ZStressTensorIntegral` (1 hr)
- [ ] **Test:** `Volume_integral(Mx=epinv.real)` returns positive real value for lossy case
- [ ] **Test:** `Tz < 0` for incident wave (radiation pressure on absorbing layer)
- [ ] **Test:** Compare against Python for ex1 config

### Phase 8: Examples & Validation — ✅ DONE (ex1/ex2/ex4 match grcwa exactly)
- [ ] Port `ex1.py` (square lattice hole, vacuum+patterned+vacuum) — **must reproduce ex1 R, T**
- [ ] Port `ex2.py` (two patterned layers, oblique incidence θ=π/10) — tests non-zero kx0
- [ ] Port `ex4.py` (hexagonal lattice, Nx=Ny=1000) — **tests Cartesian-to-lattice coord transform** (lines 46-50 of ex4.py)
- [ ] Write `scripts/generate_golden.py` — runs grcwa, saves reference outputs
- [ ] Write `scripts/compare_results.py` — element-wise diff with tolerance
- [ ] Document any numerical discrepancies (eigenvector phase, ordering, etc.)

### Phase 9: GPU Acceleration (Optional) — ⛔ SPIKED & DESCOPED (Aug 2026)

**Spike result (DONE — `benchmarks/bench_zgeev.cpp`):**
The feasibility spike was run on an RTX 2060 (apt CUDA 11.5 libs; the cuBLAS/
cuFFT paths also verified against CUDA 12.8 headers) and the answer is decisive:

1. **`cusolverDnZgeev` does not exist.** cuSOLVER (verified in the CUDA 11.5
   *and* 12.8 headers) ships only Hermitian eigensolvers (`Zheevd`/`Zheevj`)
   and SVD (`Zgesvd`/`Zgesvdj`). A GPU non-Hermitian `zgeev` would require
   third-party **MAGMA** (`magma_zgeev`) or a hand-written GPU QR iteration.
2. **S-matrix matmul (the dominant solve cost) does not accelerate.** cuBLAS
   `zgemm` vs OpenBLAS `zgemm` at the `(2nG)³` star-product size: **0.7× / 1.0×
   / 1.1×** at nG = 200 / 500 / 1000 (host↔device transfer dominates the small
   blocks). No GPU win.
3. **Only the eps convolution FFT accelerates:** cuFFT is ~20× faster than
   FFTW on a 512×512 grid — but that is a small setup fraction of the pipeline.

CPU `zgeev` baseline for reference (OpenBLAS, serial): 217 ms (nG=200),
2.6 s (nG=500), 14.5 s (nG=1000), residuals ~1e-14.

**Verdict:** Phase 9 is **descoped** — keep `zgeev` and the S-matrix on CPU
(no GPU routine / no speedup available), and the only portable GPU win (cuFFT
for `Epsilon_fft`) is not worth a backend abstraction for a small setup-step
gain. If more speed is wanted, **Intel MKL's threaded `zgeev`** (≈1.6–2×, see
README) is the cheaper CPU investment.

If Phase 9 is ever reopened, the path is: MAGMA `magma_zgeev` feasibility
first, then cuFFT + cuBLAS only; do not touch the S-matrix on GPU.

<details><summary>Original Phase 9 plan (superseded)</summary>

**Spike (do FIRST, before committing to the 5–7 day estimate):**
- [ ] **`cusolverDnZgeev` feasibility check** — non-Hermitian complex eigensolver on GPU. Historically slow and poorly conditioned relative to CPU `zgeev`; the achievable speedup at nG ≤ 1000 may be negligible or even negative after host↔device transfer. Write a 50-line micro-benchmark comparing CPU vs GPU `zgeev` for matrix sizes {200, 500, 1000, 2000} before designing the rest of the backend. If GPU loses or breaks even, **descope Phase 9** to cuFFT/cuBLAS only and keep `zgeev` on CPU.

- [ ] Design `Backend` abstraction (CPU vs GPU) — trait-based or template
- [ ] Implement `cuFFT` plan cache (similar to FFTW)
- [ ] Wrap `cublasZgemm` for matrix multiply
- [ ] Wrap `cusolverDnZgeev` and `cusolverDn[getrf|getri]`
- [ ] Custom CUDA kernels for convolution matrix gather and branch-cut `where`
- [ ] Benchmark: CPU vs GPU for nG ∈ {50, 100, 200, 500, 1000}
- [ ] Validate numerical equivalence against CPU path

</details>

### Phase 10: Polish & Documentation — ✅ DONE (README, install target; OpenMP/pybind11 deferred)
- [ ] Memory optimization (pre-allocated buffers, `.noalias()`, raw BLAS calls)
- [ ] OpenMP parallelization for multi-layer loops
- [ ] Doxygen comments on all public APIs
- [ ] README with quick-start, examples, build instructions
- [ ] CMake `install()` target + `find_package(cpprcwa)` config
- [ ] Optional: Python bindings via pybind11 (high value for adoption)

---

## 8. Performance Targets

| Operation | Python (nG=200) | C++ Target (nG=200) | Speedup |
|---|---|---|---|
| `Lattice_getG` | ~1 ms | ~0.1 ms | 10× |
| `Epsilon_fft` per layer | ~50 ms | ~5 ms | 10× |
| `SolveLayerEigensystem` (patterned) | ~100 ms | ~5 ms | 20× |
| `GetSMatrix` per layer pair | ~50 ms | ~3 ms | 15× |
| Full `RT_Solve` (5 layers, nG=200) | ~1 s | ~50 ms | 20× |
| Full `RT_Solve` (5 layers, nG=500) | ~30 s | ~1 s | 30× |
| Full `RT_Solve` (5 layers, nG=1000) | ~5 min | ~10 s | 30× |

GPU targets: another 5–10× over CPU for `nG ≥ 500`. The dominant cost (`zgeev`) does not always GPU-accelerate well due to launch overhead — measure carefully.

---

## 9. Verification Strategy

### 9.1 Three-Level Validation

**Level 1 — Module unit tests:**
For each C++ function, run identical inputs through Python `grcwa` and compare output.
- `kbloch`: compare `Lk1, Lk2, G, kx, ky` against Python to `1e-12` absolute tolerance
- `fft_funs`: compare `epsinv`, `eps2` against Python to `1e-10` (these depend on FFT, expect small numerical differences)
- `rcwa`: compare `kp_list`, `q_list`, `phi_list` per-layer (eigenvector phase ambiguity allowed — compare `|·|²` or `·*conj(other)`)

**Level 2 — Integration:**
Port each example and compare final `R`, `T` to within `1e-6` for lossless cases, `1e-3` for lossy cases.

**Level 3 — S4 Golden Reference (test_rcwa.py:69-75):**
```python
# Must reproduce these exactly:
assert abs(T - 0.85249901083265) < 1e-3 * T   # p-pol
assert abs(T - 0.83900479939861) < 1e-3 * T   # s-pol
```

### 9.2 Comparison Caveats

1. **Eigenvector phase/sign:** `zgeev` and `np.linalg.eig` may return eigenvectors with different signs or complex phases. Compare observable quantities.
2. **Eigenvector ordering:** eigenvalues are in arbitrary order in both libraries; the S-matrix algorithm doesn't depend on order, but field reconstruction assumes `q_list[i]` matches `phi_list[:,i]`. Compare `phi @ diag(q)` or `|E|²`.
3. **Branch cut:** Both libraries should produce the same branch cut (`Im(q) ≥ 0`), but verify by comparing `Im(q_list[layer]) >= 0` for all entries.
4. **`Lattice_getG` degeneracy:** If degeneracy handling differs, the *set* of G vectors may differ but both should be symmetry-respecting. Document the specific truncation for each test case.

---

## 10. Risks and Open Questions

### 10.1 High-Risk Items

1. **`Matrix_zintegral` and `Volume_integral`** — Mathematically intricate. The `shift=1e-12` on the diagonal of `qij` (rcwa.py:628) is asymmetric — only applied to `qij = qj - conj(qi)`, not `qij2 = qj + conj(qi)`. Easy to miss. Recommend porting `Gmeshgrid` and the matrix construction together, then comparing against Python for a known non-trivial case.

2. **Sign convention in `Solve_FieldFourier`** — `fey = -fexy[:nG]`, `fex = fexy[nG:]`. Verify by computing fields for a uniform layer at normal incidence (TE polarization should give `Ey = 0`, TM should give `Ex = 0` for pure s/p polarization respectively).

3. **`fey = -fexy[:nG]` does NOT match the obvious `(E_x, E_y)` ordering.** It comes from the eigenmode expansion where the amplitude vector `[a; b]` has the x-component in the first `nG` entries and y-component in the second `nG`. When mapped back to `E`, the y-component picks up a minus sign. **Add an explicit unit test** for a vacuum case at normal incidence to catch sign errors.

4. **Non-Hermitian eigendecomposition at large nG.** The `M` matrix in patterned layers is non-Hermitian. `zgeev` is much slower and less numerically stable than `zheev`. For `nG > 500`, expect noticeable numerical drift. Document tolerance widening for large nG.

5. **`Add_LayerFourier`** — no example uses it, but the API exists (rcwa.py:73-80, 192-216). The C++ port should have a stub that throws "not implemented" if called. Full implementation deferred.

### 10.2 Medium-Risk Items

6. **Branch cut convention** — NumPy `np.sqrt` and C++ `std::sqrt` for `std::complex<double>` both follow C99 with branch cut on negative real axis. Confirmed compatible.

7. **FFT normalization** — NumPy `fft2` divides by `Nx*Ny` on inverse; FFTW does not normalize. Must explicitly multiply by `1/(Nx*Ny)` on forward and divide on inverse (or use the `dN` factor pattern from Python). Already covered in §5.2.

8. **FFTW plan creation cost** — First call to `fftw_plan_dft_2d` can take 50–500ms. Always use a plan cache.

9. **`omega` complex arithmetic** — Ensure all `omega·omega` computations stay in complex domain. `std::pow` for complex is fine.

### 10.3 Out-of-Scope Items (Documented for Future Work)

10. **Autograd backend / `ex3.py` topology optimization** — Out of scope. If needed, integrate **Enzyme** (LLVM-based AD) or **CoDiPack** for reverse-mode AD. The custom gradients in `primitives.py` are the reference for VJP of `eig` and `inv`.

11. **Anisotropic permittivity in examples** — No example exercises the `eps_grid[0].ndim == 2` branch of `Epsilon_fft`. Port the code but defer validation testing.

12. **GPU backend** — Optional Phase 9. If pursued, expect additional 5–7 days of work and significant validation effort.

### 10.4 Memory Budget (nG = 1000)

The plan's pseudocode allocates many `O(nG²)` temporaries. At nG=1000 several of them are **multi-100-MB**, and combined peak RSS exceeds available RAM on small machines if not managed:

| Allocation | Shape | Size (16 B/elem) | When |
|---|---|---|---|
| `q` per layer | `2nG` | 32 KB | cheap |
| `phi`, `kp` per layer | `2nG × 2nG` | 256 MB | stored for all layers |
| `Matrix_zintegral` `qi`, `qj`, `Mt` | `2nG × 2nG` each | 256 MB each | per `Volume_integral` call |
| `Volume_integral` `outer(ab, conj(ab))` | `4nG × 4nG` | 1 GB | per call |
| `GetSMatrix` workspace (~5 matrices) | `2nG × 2nG` each | 256 MB each | per layer pair |

**Targets:**
- **nG ≤ 200:** naive allocation is fine.
- **nG ≤ 500:** use Eigen `.noalias()` for every assignment that's a product, and pre-allocate one `2nG × 2nG` workspace reused across S-matrix iterations.
- **nG = 1000+:** require a workspace arena (pre-allocated buffers passed by reference) and rewrite `Volume_integral` to compute `trace(abM · conj(F)ᵀ · M · F)` without materializing `outer(ab, conj(ab))` — compute it block-wise.
- Document the peak RSS at nG ∈ {200, 500, 1000} in `benchmarks/README.md` so regressions are caught.

**Done (Aug 2026):**
- `Volume_integral` is now **block-wise**: only 2nG×2nG blocks (`Faxy`, `Faz`,
  `Mxy`, `A/B`, `Maa/Mab`, `W_A/W_B`) are ever materialized; the 4nG×4nG
  `outer(ab,conj(ab))`/`Mt` and the 3nG×3nG `Mtotal`/`F` are gone. Identical
  to the original formula to ~1e-16 (golden `Volume_integral` test passes).
- **Uniform-layer dedup**: kp/q/phi stored as `shared_ptr<const>`, shared
  across identical ε; all uniform layers share one Identity phi. Persistent
  storage for periodic stacks drops ~20×.
- **`RCWAConfig::report_memory`** (+ `--mem` in the examples) prints an
  estimate of persistent + transient peak memory right after `Init_Setup`
  (all sizes depend only on nG + layer structure), validated against measured
  RSS.
- Remaining arena work — pre-allocated `GetSMatrix` star-product scratch
  buffers to cut malloc churn — is a *speed* optimization, not a peak-RSS one
  (the per-step peak is already bounded at ~20 live 2nG×2nG matrices).

### 10.5 Error-Handling Policy

Pick a single convention up front and apply it everywhere:

- **LAPACK non-zero `info`:** throw `std::runtime_error` with a string of the form `"<function>: LAPACK <name> info=<n> at layer=<l>"`. Never silently continue — a singular `kp` will produce NaNs that are far harder to debug than the original failure.
- **Singular `kp` (finite `Qabs` too small):** this surfaces as `info > 0` from `zgetrf`. Catch and rethrow as a typed `cpprcwa::SingularMatrixError` with the offending layer index and `Qabs` value, so callers can retry with a larger `Qabs`.
- **User-config errors** (layer 0 not uniform, nG ≤ 0, lattice with zero area, missing `GridLayer_geteps` before solve): `static_assert`/`assert` for cheap checks, throw for runtime.
- **`Add_LayerFourier`:** throw `cpprcwa::NotImplementedError` (do not silently stub — the plan says so in §10.3).
- Use a single `namespace cpprcwa { namespace error { ... } }` block for the exception hierarchy; do not throw raw `std::exception`s.

### 10.6 Performance Targets Need Measurement (§8 Caveat)

The targets in §8 are projections, not measurements. The `zgeev` fraction dominates the patterned-layer cost and Python+OpenBLAS is already calling the same LAPACK routine C++ would — so the *achievable* speedup for patterned-layer problems may be far smaller than the listed 20–30×. Before locking the targets:
1. Profile `grcwa` on `ex1`/`ex4` at nG ∈ {50, 200, 500} and record the `zgeev`-vs-other split.
2. Update §8 with the measured split and a per-stage realistic target.
3. The 20–30× numbers are most credible for uniform-layer-dominated problems and least credible for nG ≥ 500 patterned layers.

---

## 11. Dependency Summary

### 11.1 Required

| Dependency | Version | Notes |
|---|---|---|
| C++ compiler | C++17 | GCC 9+, Clang 10+, MSVC 19.20+ |
| CMake | ≥ 3.18 | Build system |
| Eigen | 3.4+ | Header-only; primary linear algebra |
| FFTW | 3.3.x | FFT; install `libfftw3-dev` |
| BLAS | any | Vectorized matmul; OpenBLAS or MKL recommended |
| LAPACK | any | `zgeev`, `zgetrf`+`zgetri`; OpenBLAS or MKL |

### 11.2 Optional (Phase 9, GPU)

| Dependency | Version |
|---|---|
| CUDA Toolkit | ≥ 11.0 |
| cuFFT, cuBLAS, cuSOLVER | bundled with CUDA |
| OpenMP | CPU multi-threading |

### 11.3 Test/Build

| Dependency | Purpose |
|---|---|
| Catch2 v3 | Unit test framework (`tests/`) |
| Python 3 + NumPy | Validation scripts (`scripts/`) |
| Doxygen | Documentation generation (optional) |
| pybind11 | Python bindings (optional, Phase 10) |

---

## 12. Suggested File-by-File Port Order (for incremental PRs)

If doing this as a series of PRs, recommend the following order to maximize incremental validation:

1. **`types.h`** — pure declarations, no logic
2. **`kbloch.{h,cpp}` + `test_kbloch.cpp`** — first port, easy validation
3. **`fft_funs.{h,cpp}` + `test_fft_funs.cpp`** — independent of rcwa
4. **`internal/lapack_wrappers.{h,cpp}`** — thin LAPACK helpers
5. **`internal/branch_cut.{h,cpp}`** — used everywhere
6. **`rcwa.{h,cpp}` (subset 1): constructor, layer add, `Init_Setup`** — uniform-only solves
7. **`rcwa.{h,cpp}` (subset 2): `GridLayer_geteps`, `SolveLayerEigensystem`** — patterned layers
8. **`rcwa.{h,cpp}` (subset 3): `GetSMatrix`, `SolveExterior`, `RT_Solve`** — main solve loop
9. **`rcwa.{h,cpp}` (subset 4): field reconstruction** — `Solve_Field*`, `Return_eps`
10. **`rcwa.{h,cpp}` (subset 5): post-processing** — `Volume_integral`, stress tensor
11. **Examples** (ex1, ex2, ex4) — incremental per example
12. **GPU backend** — last, if pursued
