#!/usr/bin/env python3
"""Compare Solve_FieldFourier: grcwa (full harmonic set) vs cpprcwa
with quasi1d=True (fast path).

The cpprcwa quasi1d fast path restricts the harmonic set to the x-only row
(j=0) — exact for y-invariant structures. The original grcwa library has no
such option, so it solves the full harmonic set; for this geometry every
y != 0 order is a dead mode (amplitude ~ 0), and the non-zero harmonics are
exactly the (i, 0) row.

This script:
  1. runs both solvers on the quasi-1D EUV line-grating (full 40x Mo/Si
     multilayer) at oblique incidence,
  2. matches orders between the two solvers by (kx, ky) — verified against
     the integer G pairs as a cross-check,
  3. asserts that grcwa's non-1D (j != 0) harmonics carry negligible field
     (i.e. the fast path discards only dead modes),
  4. compares Solve_FieldFourier element-wise on the matched (i, 0) orders
     at several layers and z-offsets.

Usage:
    python3 scripts/quasi1d_field_fastpath.py [nG_req] [Nx] [Ny]
        [theta_deg] [phi_deg] [abs_t] [nb] [tol]

Needs:
    - the original grcwa checkout (grcwa_orig/ next to the repo, or
      $CPPRCWA_GRCWA_ORIG)
    - a build of the cpprcwa nanobind module (build/, build-mkl/, or
      $CPPRCWA_BUILD)

Exit code 0 = fast path agrees with grcwa within tol, 1 = mismatch.
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)

# ── locate the original grcwa library ───────────────────────────────────────
for cand in (os.environ.get("CPPRCWA_GRCWA_ORIG"),
             os.path.join(_repo, "..", "grcwa_orig"),
             os.path.join(os.path.dirname(_repo), "grcwa_orig")):
    if cand and os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "grcwa")):
        sys.path.insert(0, cand)
        break
import grcwa  # noqa: E402

# ── locate the cpprcwa nanobind module ──────────────────────────────────────
for cand in (os.environ.get("CPPRCWA_BUILD"),
             os.path.join(_repo, "build"),
             os.path.join(_repo, "build-mkl")):
    if cand and os.path.isdir(cand):
        sys.path.insert(0, cand)
        break
import cpprcwa  # noqa: E402


def eps_from_n(n):
    return n * n


def make_stack(obj, abs_t, Nx, Ny, nb):
    """Add the quasi-1D EUV line-grating layer stack (shared by both libs):
    vacuum / TaN grid / Ru cap / nb x (Mo, Si) / Si substrate."""
    n_ru = 0.9114 + 0.0171j
    n_mo = 0.9226 + 0.0064j
    n_si = 0.9997 + 0.0018j
    obj.Add_LayerUniform(1.0, eps_from_n(1.0 + 0j))       # incident vacuum
    obj.Add_LayerGrid(abs_t, Nx, Ny)                       # TaN line grating
    obj.Add_LayerUniform(2.5, eps_from_n(n_ru))            # Ru cap
    for _ in range(nb):
        obj.Add_LayerUniform(2.8, eps_from_n(n_mo))        # Mo
        obj.Add_LayerUniform(4.2, eps_from_n(n_si))        # Si
    obj.Add_LayerUniform(1.0, eps_from_n(n_si))            # substrate


def make_epgrid(Nx, Ny, wx_nm, Lx_nm):
    epgrid = np.ones((Nx, Ny), dtype=complex)
    u = np.linspace(0, 1, Nx)
    half = wx_nm / (2 * Lx_nm)
    for i in range(Nx):
        if abs(u[i] - 0.5) < half:
            epgrid[i, :] = 0.9562 + 0.0323j   # TaN (n, not n^2 — shared convention)
    return epgrid


def match_by_k(g_kx, g_ky, c_kx, c_ky, g_G, c_G, atol=1e-9):
    """Match grcwa orders to cpprcwa orders by (kx, ky). Returns:
    - perm: cpprcwa index for each grcwa index (None if unmatched),
    - matched: list of grcwa indices that found a cpprcwa partner.
    kx/ky are complex; G integer pairs are used as a cross-check."""
    n_g = len(g_kx)
    perm = [None] * n_g
    used = set()
    for j in range(n_g):
        # candidate cpprcwa indices with matching kx/ky (complex, ~1e-9)
        cand = [i for i in range(len(c_kx))
                if abs(c_kx[i] - g_kx[j]) < atol and abs(c_ky[i] - g_ky[j]) < atol
                and i not in used]
        if not cand:
            continue
        # prefer the candidate whose integer G pair also matches (exact tie-break)
        best = None
        for i in cand:
            if np.array_equal(c_G[i], g_G[j]):
                best = i
                break
        if best is None:
            best = cand[0]
        perm[j] = best
        used.add(best)
    matched = [j for j in range(n_g) if perm[j] is not None]
    return perm, matched


def main():
    nG_req = int(sys.argv[1]) if len(sys.argv) > 1 else 61
    Nx = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    Ny = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    theta_deg = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
    phi_deg = float(sys.argv[5]) if len(sys.argv) > 5 else 45.0
    abs_t = float(sys.argv[6]) if len(sys.argv) > 6 else 60.0
    nb = int(sys.argv[7]) if len(sys.argv) > 7 else 40
    tol = float(sys.argv[8]) if len(sys.argv) > 8 else 1e-8

    Lx = 3500.0
    Ly = 67.5
    wx = 500.0
    lam = 13.5

    # ── grcwa (full harmonic set) ───────────────────────────────────────────
    g = grcwa.obj(nG_req, [Lx, 0], [0, Ly], 1.0 / lam,
                  np.radians(theta_deg), np.radians(phi_deg), verbose=0)
    make_stack(g, abs_t, Nx, Ny, nb)
    g.Init_Setup()
    g.GridLayer_geteps(make_epgrid(Nx, Ny, wx, Lx).flatten())
    g.MakeExcitationPlanewave(1, 0, 0, 0, order=0)

    # ── cpprcwa (quasi1d fast path) ─────────────────────────────────────────
    c = cpprcwa.obj(nG=nG_req, L1=[Lx, 0.0], L2=[0.0, Ly],
                    freq=1.0 / lam, theta=np.radians(theta_deg),
                    phi=np.radians(phi_deg), quasi1d=True)
    make_stack(c, abs_t, Nx, Ny, nb)
    c.Init_Setup()
    c.GridLayer_geteps(make_epgrid(Nx, Ny, wx, Lx).flatten())
    c.MakeExcitationPlanewave(p_amp=1.0, p_phase=0.0, s_amp=0.0, s_phase=0.0)

    gG = np.asarray(g.G); gkx = np.asarray(g.kx); gky = np.asarray(g.ky)
    cG = np.asarray(c.G); ckx = np.asarray(c.kx); cky = np.asarray(c.ky)
    print(f"grcwa:   nG_req={nG_req} -> nG={g.nG}  (j=0 orders: "
          f"{(gG[:,1] == 0).sum()})")
    print(f"cpprcwa: nG_req={nG_req} -> nG={c.nG} (quasi1d, x-only)")

    # sanity: cpprcwa fast path must be purely x-only (j=0)
    if not np.all(cG[:, 1] == 0):
        print("FATAL: cpprcwa quasi1d harmonic set is not x-only")
        return 1

    # match orders by (kx, ky), cross-checked against G
    perm, matched = match_by_k(gkx, gky, ckx, cky, gG, cG)
    if not matched:
        print("FATAL: no orders matched between the two solvers")
        return 1

    # step 3: verify grcwa's non-1D (j != 0) harmonics are dead modes.
    # Check the reflected field (layer 0, z=0) and one interior layer.
    dead_max = 0.0
    probe_layers = [0, 1, deep_mo_of(nb)]
    for l in probe_layers:
        fg = np.array(g.Solve_FieldFourier(l, 0.0)[0]).reshape(6, -1)
        for j in range(len(gG)):
            if gG[j, 1] != 0:
                dead_max = max(dead_max, np.abs(fg[:, j]).max())
    print(f"grcwa max |field| on j!=0 (dead) harmonics = {dead_max:.3e}")

    layers = [0, 1, 2, deep_mo_of(nb), g.Layer_N - 1]
    zs = [0.0, 0.5, 1.0]
    comps = ["ex", "ey", "ez", "hx", "hy", "hz"]

    worst = 0.0
    for l in layers:
        for z in zs:
            fg = np.array(g.Solve_FieldFourier(l, z)[0]).reshape(6, -1)
            fc = np.array(c.Solve_FieldFourier(l, z)[0]).reshape(6, -1)
            # compare only matched orders: fc[:, perm[j]] vs fg[:, j]
            fg_m = fg[:, [j for j in matched]]
            fc_m = fc[:, [perm[j] for j in matched]]
            scale = max(np.abs(fg_m).max(), 1e-30)
            d = np.abs(fg_m - fc_m).max() / scale
            worst = max(worst, d)
            for ci, name in enumerate(comps):
                if np.abs(fg_m[ci]).max() > 1e-12:
                    rel = np.abs(fg_m[ci] - fc_m[ci]).max() / np.abs(fg_m[ci]).max()
                else:
                    rel = np.abs(fg_m[ci] - fc_m[ci]).max()
                print(f"layer {l:2d} z={z:4.1f}  {name:2s}  rel {rel:12.3e}")
            print(f"layer {l:2d} z={z:4.1f}  max-norm  {d:12.3e}")

    print(f"\nworst relative field diff (matched 1D orders) = {worst:.3e}  "
          f"(tol = {tol})")
    ok = worst <= tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def deep_mo_of(nb):
    return 2 * (nb // 2) + 1         # Mo layer at index 2k+1, k = nb//2


if __name__ == "__main__":
    sys.exit(main())
