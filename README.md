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

## Requirements

- CMake ≥ 3.18
- A C++17 compiler (GCC 9+, Clang 10+, MSVC 19.20+)
- [Eigen](https://eigen.tuxfamily.org) 3.4+
- [FFTW3](https://www.fftw.org) 3.3.x (`libfftw3-dev`)
- BLAS + LAPACK (OpenBLAS recommended)
- Catch2 v3 (for tests; system package or FetchContent fallback)

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `CPPRCWA_BUILD_TESTS` | `ON` | Unit tests (`tests/`) |
| `CPPRCWA_BUILD_EXAMPLES` | `ON` | Example programs (`examples/`) |
| `CPPRCWA_BUILD_BENCHMARKS` | `OFF` | Timed benchmarks (`benchmarks/`) |
| `CPPRCWA_USE_CUDA` | `OFF` | GPU backend (not yet implemented) |

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
