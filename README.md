# cpprcwa

A high-performance C++ port of the **grcwa** Python library (Rigorous
Coupled-Wave Analysis for photonic structures). The forward RCWA solver is
ported; autograd-based reverse-mode differentiation (grcwa `ex3.py`) is out
of scope.

## Features

- Circular / parallelogramic reciprocal-lattice truncation (`kbloch`)
- FFT-based epsilon convolution matrices with a thread-safe plan cache (`fft_funs`)
- Full S-matrix engine: `RT_Solve`, amplitudes, fields (Fourier + real-space),
  `Return_eps`, `Volume_integral`, stress-tensor integral (`rcwa`)
- Uniform and patterned (grid) layers, isotropic and anisotropic epsilons
  (anisotropic validated only at the module level)
- Validated against grcwa / S4: reproduces the S4 golden reference
  `T=0.85249901083265` (p-pol) and `T=0.83900479939861` (s-pol) within `1e-3`
  relative tolerance, and matches grcwa's `ex1.py`, `ex2.py`, `ex4.py` exactly.
- Python bindings (nanobind) expose the grcwa `obj` API as a drop-in
  replacement — grcwa's own `ex1.py` / `ex2.py` / `ex4.py` run unchanged via
  `import grcwa` (shim) or `import cpprcwa`, with machine-precision agreement.

## Requirements

- CMake ≥ 3.18
- A C++17 compiler (GCC 9+, Clang 10+, MSVC 19.20+)
- [Eigen](https://eigen.tuxfamily.org) 3.4+
- [FFTW3](https://www.fftw.org) 3.3.x (`libfftw3-dev`)
- BLAS + LAPACK (OpenBLAS recommended)
- Catch2 v3 (for tests; system package or FetchContent fallback)
- Python 3 + nanobind + numpy (for the Python bindings; `CPPRCWA_BUILD_PYTHON`)

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Or install the Python bindings as a package (scikit-build-core + nanobind):

```bash
pip install .
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `CPPRCWA_BUILD_TESTS` | `ON` | Unit tests (`tests/`) |
| `CPPRCWA_BUILD_EXAMPLES` | `ON` | Example programs (`examples/`) |
| `CPPRCWA_BUILD_BENCHMARKS` | `OFF` | Timed benchmarks (`benchmarks/`) |
| `CPPRCWA_USE_CUDA` | `OFF` | GPU backend (not yet implemented) |
| `CPPRCWA_USE_NATIVE` | `ON` | Release: `-march=native -mtune=native` (CPU-specific SIMD) |
| `CPPRCWA_USE_LTO` | `ON` | Release: `-flto` (link-time optimization) |
| `CPPRCWA_USE_FASTMATH` | `OFF` | Release: `-ffast-math` (opt-in; alters FP rounding) |
| `CPPRCWA_BUILD_PYTHON` | `ON` | Python bindings (nanobind) |

Release builds also add `-funroll-loops -fno-math-errno -ffp-contract=fast`
(GCC/Clang). The heavy `zgemm`/`zgeev` work runs in prebuilt OpenBLAS/LAPACK,
so these flags mainly speed up the surrounding C++ scalar code (~5–10%
wall-clock); they are verified to leave R and the grcwa field agreement at
machine precision.

### Intel MKL backend

OpenBLAS is the default. Intel MKL can give a significant speedup of the
patterned-layer eigen-solve (`zgeev`), which OpenBLAS runs single-threaded:

- `zgeev` (2nG×2nG, n=266): OpenBLAS ~126 ms vs MKL ~80 ms at 6 threads
  (~1.6×); the gap grows with matrix size (~2× at n=530).
- End-to-end oblique quasi-1D (nG=187): ~1.5× faster total
  (1268 ms → ~830 ms).

To build with MKL (threaded, Intel OpenMP):

```bash
cmake -S . -B build-mkl -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=Intel10_64lp
cmake --build build-mkl -j
```

`-DBLA_VENDOR=Intel10_64lp_seq` selects sequential MKL instead. Control the
thread count at runtime with `MKL_NUM_THREADS` / `OMP_NUM_THREADS`.

**Known caveats (Ubuntu `intel-mkl` packages, 2020.x):**

- Loading MKL into a Python process (e.g. the `cpprcwa` bindings) can fail
  with `undefined symbol: mkl_sparse_optimize_bsr_trsm_i8` because the
  dispatcher's `libmkl_avx2.so` is incomplete. Workaround:
  ```bash
  export LD_PRELOAD="$(dpkg -L intel-mkl | grep 'libiomp5.so$') \
  $(dpkg -L intel-mkl | grep 'libmkl_intel_thread.so$') \
  $(dpkg -L intel-mkl | grep 'libmkl_core.so$') \
  $(dpkg -L intel-mkl | grep 'libmkl_intel_lp64.so$') \
  $(dpkg -L intel-mkl | grep 'libmkl_avx2.so$')"
  ```
- Do **not** raise the OpenBLAS thread count to speed up `zgeev`: OpenBLAS's
  `zgeev` is LAPACK-serial, so extra threads only add pool overhead (it got
  *slower*: 126 ms → 393 ms at n=266). MKL threads `zgeev` internally and
  benefits from `MKL_NUM_THREADS`.

## Python bindings (grcwa drop-in)

When `CPPRCWA_BUILD_PYTHON=ON`, the build produces `cpprcwa.cpython-*.so`
plus a `grcwa/` shim package in the build directory. Use it in place of
grcwa:

```bash
PYTHONPATH=<build-dir> python3 your_script.py     # import cpprcwa
PYTHONPATH=<build-dir> python3 grcwa_script.py    # import grcwa (shim)
```

The `obj` class mirrors `grcwa.obj` (constructor `(nG, L1, L2, freq, theta,
phi, verbose=1, quasi1d=False)`, `Add_Layer*`, `Init_Setup`,
`GridLayer_geteps`, `MakeExcitationPlanewave`, `RT_Solve`, `GetAmplitudes`,
`Solve_FieldFourier/OnGrid`, `Return_eps`, `Volume_integral`,
`Solve_ZStressTensorIntegral`, and the `G/kx/ky/q_list/...` attributes).
Module-level `Lattice_*`, `get_fft`, `get_ifft`, `Epsilon_fft` are exported
too. grcwa's own `ex1.py`, `ex2.py`, `ex4.py` run unchanged and match the
original to machine precision. Anisotropic `GridLayer_geteps` and
`Add_LayerFourier` are not implemented (raise `NotImplementedError`).

## Performance notes

- Matrix products use the BLAS backend (`EIGEN_USE_BLAS`, i.e. OpenBLAS
  `zgemm`); LAPACK is used for `zgeev` and matrix inversions.
- FFTW plans are cached per `(Nx, Ny)` and created with `FFTW_ESTIMATE`.
- OpenBLAS can oversubscribe on small S-matrix blocks: `ex_euv_absorber`
  caps it at `min(cores, 6)` threads at runtime (overridable with
  `--threads N` or `OPENBLAS_NUM_THREADS`).
- The S-matrix caches interface matrices for repeated uniform layer pairs
  (helps periodic multilayer stacks), reuses LAPACK workspaces, and uses
  LU-solve + common-subexpression elimination in the star product.
- Uniform layers share `(kp, q, phi)` across identical ε values (cached in
  `Init_Setup`), turning 80 eigen/matrix setups into ~5 for periodic stacks.
- **Quasi-1D structures** (`cfg.quasi1d`): exact reductions for y-invariant
  geometries — the harmonic set is filtered to the x-only row, and the
  all-uniform multilayer suffix is solved with a per-harmonic recursion
  (scalar when `ky0=0`, exact 2×2 Ex–Ey blocks for oblique incidence) combined
  via an overlapping cascade. Valid for any (θ, φ) — results match full-2D to
  machine precision. Measured ~30× faster than grcwa
  (`ex_quasi1d_absorber --quasi1d`, nG=201), and ~40–90× faster than the
  general 2D path at oblique incidence (nG=400).
- Measured on the EUV absorber (nG=97): ~2× faster than the naive build, and
  ≈5× faster than grcwa overall, with machine-precision field agreement.

## Quick start

```cpp
#include <cpprcwa/cpprcwa.h>
using namespace cpprcwa;

RCWAConfig cfg;
cfg.nG = 101;                       // truncation order
cfg.L1 = Eigen::Vector2d(0.1, 0.0); // direct lattice vector 1
cfg.L2 = Eigen::Vector2d(0.0, 0.1); // direct lattice vector 2
cfg.freq = complex(1.0, 0.0);       // COMPLEX frequency (fold in Qabs loss here)
cfg.theta = 0.0;                    // polar angle
cfg.phi = 0.0;                      // azimuthal angle

RCWA solver(cfg);
solver.Add_LayerUniform(1.0, complex(1.0, 0.0));
solver.Add_LayerGrid(0.2, 100, 100);           // patterned layer (Nx×Ny)
solver.Add_LayerUniform(1.0, complex(1.0, 0.0));
solver.Init_Setup();                            // sets up G, kx/ky, uniform eigensystems

// Provide permittivity for all patterned layers (flat concatenation).
std::vector<complex> ep(100 * 100, complex(4.0, 0.0));
// ... set ep to the desired pattern ...
solver.GridLayer_geteps(ep);

PlaneWaveExcitation exc;
exc.p_amp = 1.0;                    // p-polarized incident plane wave
solver.MakeExcitationPlanewave(exc);

RTResult rt = solver.RT_Solve(/*normalize=*/true);
std::printf("R=%f T=%f\n", rt.R, rt.T);
```

## Validation against grcwa

Run the S4 golden test:

```bash
ctest --test-dir build --output-on-failure -R "golden"
```

Regenerate reference files from the Python library:

```bash
python3 scripts/generate_golden.py --example s4
```

## Layout

```
include/cpprcwa/  public headers (cpprcwa.h aggregator, types.h, kbloch.h,
                   fft_funs.h, rcwa.h, errors.h)
src/              implementations (rcwa.cpp, kbloch.cpp, fft_funs.cpp,
                   internal/{lapack_wrappers,branch_cut,utils}.{h,cpp})
tests/            Catch2 unit tests + golden reference outputs
examples/         ex1 (square holes), ex2 (two layers), ex4 (hexagonal),
                  ex_euv_multilayer (Mo/Si mirror), ex_euv_absorber (TaN pattern),
                  ex_quasi1d_absorber (quasi-1D line grating)
benchmarks/       timed benchmarks
scripts/          golden-file generation / comparison helpers
```

## License

GPLv3 (inherited from grcwa).
