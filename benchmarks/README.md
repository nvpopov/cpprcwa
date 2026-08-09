# cpprcwa benchmarks

Harnesses in this directory (build with `-DCPPRCWA_BUILD_BENCHMARKS=ON`):

| tool | purpose |
|---|---|
| `bench_full_rt` | end-to-end `RT_Solve` on a 3-layer hole pattern |
| `bench_euv_stack` | EUV absorber (500 nm TaN bar, Lx=3.5 µm, 40× Mo/Si) — quasi-1D and full-2D, best-of-N timing |
| `bench_zgeev` | GPU (`cusolverDnZgeev`) vs CPU (`zgeev`) feasibility spike → **concluded no GPU zgeev exists** (see PLAN.md §Phase 9) |
| `bench_grcwa.py` | same geometry timed through the grcwa nanobind shim |

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCPPRCWA_BUILD_BENCHMARKS=ON
cmake --build build -j --target bench_euv_stack bench_zgeev
./build/benchmarks/bench_euv_stack --quasi1d 201 500 1000       # quasi-1D, best of 5
./build/benchmarks/bench_euv_stack 201                          # full-2D
OPENBLAS_NUM_THREADS=6 ./build/benchmarks/bench_euv_stack ...   # 6 threads recommended
PYTHONPATH=build python3 benchmarks/bench_grcwa.py --quasi1d 201
```

---

## Results (Aug 2026, OpenBLAS 6 threads, best-of-N)

> **Mode legend — always explicit:** `QUASI-1D NORMAL` = exact y-invariant
> reduction, θ=0°, φ=0° (block-diagonal eig applies). `QUASI-1D OBLIQUE` =
> quasi-1D, θ≠0°, φ≠0° (conical: full 2nG eig, 2×2 per-harmonic suffix).
> `FULL-2D NORMAL` / `FULL-2D OBLIQUE` = general 2D solve.

### MODE: QUASI-1D NORMAL (θ=0°, φ=0°)

Full TE/TM block-decoupled path (block-diagonal eig + block-decoupled S-matrix).

| nG | 2nG | setup ms | rt_solve ms | total ms | R |
|---|---|---|---|---|---|
| 113 | 226 | 49 | 35 | **84** | 0.528927 |
| 187 | 374 | 122 | 102 | 223 | 0.529043 |
| 253 | 506 | 244 | 202 | 446 | 0.529086 |
| 359 | 718 | 579 | 452 | 1031 | 0.529114 |

### MODE: QUASI-1D OBLIQUE (θ=6°, φ=45° — conical, full 2nG eig)

| nG | 2nG | setup ms | rt_solve ms | total ms | R |
|---|---|---|---|---|---|
| 113 | 226 | 71 | 74 | **145** | 0.547546 |
| 187 | 374 | 229 | 217 | 445 | 0.547773 |
| 253 | 506 | 605 | 482 | 1087 | 0.547820 |

### MODE: FULL-2D NORMAL (θ=0°, φ=0°)

| nG | 2nG | setup ms | rt_solve ms | total ms | R |
|---|---|---|---|---|---|
| 199 | 398 | 228 | 1309 | **1536** | 0.528927 |
| 497 | 994 | 2526 | 14402 | 16928 | 0.529043 |

### MODE: FULL-2D OBLIQUE (θ=6°, φ=45°)

| nG | 2nG | setup ms | rt_solve ms | total ms | R |
|---|---|---|---|---|---|
| 199 | 398 | 257 | 1189 | **1446** | 0.547546 |

### Memory (estimated peak RSS from `--mem`, validated vs `/usr/bin/time`; all NORMAL incidence)

| config | persistent | RT transient | peak RSS |
|---|---|---|---|
| QUASI-1D NORMAL nG=113 | 6.5 MiB | 15.6 MiB | 28 MiB |
| QUASI-1D NORMAL nG=253 | 32 MiB | 78 MiB | 142 MiB |
| FULL-2D NORMAL nG=199 | 20 MiB | 48 MiB | 88 MiB |

### Speedup summary (mode-explicit)

| case | before | after | factor |
|---|---|---|---|
| FULL-2D NORMAL nG=97 `rt_solve` | 963 ms | ~300 ms | 3.2× |
| FULL-2D NORMAL nG=201 `rt_solve` | 6420 ms | ~1500 ms | 4.2× |
| QUASI-1D NORMAL nG=201 total | ~360 ms | 84 ms | 4.3× |
| QUASI-1D NORMAL nG=253 `rt_solve` | 559 ms | 202 ms | 2.8× |

Historical grcwa (Python) comparison from SUMMARY.md: grcwa EUV absorber nG=97
≈ 5.8 s (cpprcwa ~1.05 s → ~5×); grcwa QUASI-1D NORMAL nG=201 ≈ 11.0 s
(cpprcwa 136 ms → **~80×**).

`rt_solve` scales ~cubically with nG (S-matrix is (2nG)³-bound). The
periodic-core binary exponentiation and quasi-1D fast path are what keep
multilayer stacks tractable.

## Benchmarking notes

- The machine runs other load (browser/IDE); use best-of-N (the harness does
  this, default 5) rather than single runs.
- OpenBLAS `zgeev` is single-threaded; `zgemm` threads. 6 threads is the
  measured sweet spot (more causes pool overhead on the small S-matrix blocks).
- MKL's threaded `zgeev` is ~1.6–2× faster than OpenBLAS's serial `zgeev` (see
  README.md MKL section) — the cheapest single-component speedup available.
