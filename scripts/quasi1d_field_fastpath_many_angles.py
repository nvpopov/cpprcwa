#!/usr/bin/env python3
"""Compare Solve_FieldFourier across many incidence angles:
grcwa (full set) vs cpprcwa quasi1d=True fast path.

Same methodology as quasi1d_field_fastpath.py, but sweeps a grid of
(theta, phi) incidence angles instead of a single one. For each angle it
  - solves grcwa (full harmonic set) and cpprcwa (quasi1d, x-only),
  - matches orders by (kx, ky) (cross-checked against integer G),
  - checks grcwa's j != 0 harmonics are dead modes (they are ~0),
  - compares Solve_FieldFourier on the matched (i, 0) orders at several
    layers and z-offsets,
and reports the worst relative field diff over the whole sweep.

Use a modest nG (default 61): grcwa's full-set solve over the 84-layer
multilayer is ~4 s per angle, so ~12-20 angles take ~1-2 min.

Usage:
    python3 scripts/quasi1d_field_fastpath_many_angles.py \
        [nG_req] [Nx] [Ny] [abs_t] [nb] [tol] \
        --thetas 0,10,20,30 --phis 0,45,90

Needs:
    - the original grcwa checkout (grcwa_orig/ next to the repo, or
      $CPPRCWA_GRCWA_ORIG)
    - a build of the cpprcwa nanobind module (build/, build-mkl/, or
      $CPPRCWA_BUILD)

Exit code 0 = fast path agrees with grcwa within tol at every angle, 1 = else.
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
    """vacuum / TaN grid / Ru cap / nb x (Mo, Si) / Si substrate."""
    n_ru = 0.9114 + 0.0171j
    n_mo = 0.9226 + 0.0064j
    n_si = 0.9997 + 0.0018j
    obj.Add_LayerUniform(1.0, eps_from_n(1.0 + 0j))
    obj.Add_LayerGrid(abs_t, Nx, Ny)
    obj.Add_LayerUniform(2.5, eps_from_n(n_ru))
    for _ in range(nb):
        obj.Add_LayerUniform(2.8, eps_from_n(n_mo))
        obj.Add_LayerUniform(4.2, eps_from_n(n_si))
    obj.Add_LayerUniform(1.0, eps_from_n(n_si))


def make_epgrid(Nx, Ny, wx_nm, Lx_nm):
    epgrid = np.ones((Nx, Ny), dtype=complex)
    u = np.linspace(0, 1, Nx)
    half = wx_nm / (2 * Lx_nm)
    for i in range(Nx):
        if abs(u[i] - 0.5) < half:
            epgrid[i, :] = 0.9562 + 0.0323j   # TaN (n, not n^2)
    return epgrid


def match_by_k(g_kx, g_ky, c_kx, c_ky, g_G, c_G, atol=1e-9):
    """Match grcwa orders to cpprcwa orders by (kx, ky). Returns
    (perm, matched): perm[j] = cpprcwa index for grcwa index j (None if
    unmatched), matched = list of grcwa indices with a partner."""
    n_g = len(g_kx)
    perm = [None] * n_g
    used = set()
    for j in range(n_g):
        cand = [i for i in range(len(c_kx))
                if abs(c_kx[i] - g_kx[j]) < atol and abs(c_ky[i] - g_ky[j]) < atol
                and i not in used]
        if not cand:
            continue
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


def solve_compare_one(theta_deg, phi_deg, nG_req, Nx, Ny, abs_t, nb,
                      Lx, Ly, wx, lam, layers, zs):
    """Solve both solvers at one angle and return (worst_rel, dead_max)."""
    # ── grcwa (full set) ────────────────────────────────────────────────────
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

    if not np.all(cG[:, 1] == 0):
        raise RuntimeError("cpprcwa quasi1d harmonic set is not x-only")

    perm, matched = match_by_k(gkx, gky, ckx, cky, gG, cG)
    if not matched:
        raise RuntimeError("no orders matched")

    # dead-mode check: grcwa j != 0 harmonics must carry ~zero field
    dead_max = 0.0
    probe = {0, 1, 2, layers[-2], layers[-1]}
    for l in probe:
        fg = np.array(g.Solve_FieldFourier(l, 0.0)[0]).reshape(6, -1)
        for j in range(len(gG)):
            if gG[j, 1] != 0:
                dead_max = max(dead_max, np.abs(fg[:, j]).max())

    # field comparison on matched (i, 0) orders
    worst = 0.0
    for l in layers:
        for z in zs:
            fg = np.array(g.Solve_FieldFourier(l, z)[0]).reshape(6, -1)
            fc = np.array(c.Solve_FieldFourier(l, z)[0]).reshape(6, -1)
            fg_m = fg[:, [j for j in matched]]
            fc_m = fc[:, [perm[j] for j in matched]]
            worst = max(worst,
                        np.abs(fg_m - fc_m).max() / max(np.abs(fg_m).max(), 1e-30))
    return worst, dead_max, g.nG, c.nG


def parse_deg_list(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def main():
    args = sys.argv[1:]

    thetas = None
    phis = None
    pos = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--thetas" and i + 1 < len(args):
            thetas = parse_deg_list(args[i + 1]); i += 2
        elif a == "--phis" and i + 1 < len(args):
            phis = parse_deg_list(args[i + 1]); i += 2
        else:
            pos.append(a); i += 1

    nG_req = int(pos[0]) if len(pos) > 0 else 61
    Nx = int(pos[1]) if len(pos) > 1 else 80
    Ny = int(pos[2]) if len(pos) > 2 else 4
    abs_t = float(pos[3]) if len(pos) > 3 else 60.0
    nb = int(pos[4]) if len(pos) > 4 else 40
    tol = float(pos[5]) if len(pos) > 5 else 1e-8

    if thetas is None:
        thetas = [0.0, 10.0, 20.0, 30.0]
    if phis is None:
        phis = [0.0, 45.0, 90.0]

    Lx = 3500.0
    Ly = 67.5
    wx = 500.0
    lam = 13.5

    deep_mo = 2 * (nb // 2) + 1
    substrate = 2 + 2 * nb + 1          # vacuum, grid, Ru, 2*nb multilayers, sub
    layers = [0, 1, 2, deep_mo, substrate]
    zs = [0.0, 0.5, 1.0]
    comps = ["ex", "ey", "ez", "hx", "hy", "hz"]

    print(f"angle sweep: thetas={thetas} deg, phis={phis} deg")
    print(f"nG_req={nG_req} Nx={Nx} Ny={Ny} abs_t={abs_t} nb={nb} tol={tol} "
          f"layers={layers}")
    print(f"{'theta':>7} {'phi':>6} {'nG_g':>5} {'nG_c':>5} "
          f"{'dead_max':>10} {'worst_rel':>10}  verdict")
    print("-" * 70)

    worst_overall = 0.0
    n_fail = 0
    for th in thetas:
        for ph in phis:
            worst, dead, ng, nc = solve_compare_one(
                th, ph, nG_req, Nx, Ny, abs_t, nb, Lx, Ly, wx, lam,
                layers, zs)
            worst_overall = max(worst_overall, worst)
            ok = worst <= tol
            if not ok:
                n_fail += 1
            print(f"{th:7.1f} {ph:6.1f} {ng:5d} {nc:5d} "
                  f"{dead:10.3e} {worst:10.3e}  {'PASS' if ok else 'FAIL'}")

    print("-" * 70)
    print(f"\nworst relative field diff over {len(thetas) * len(phis)} angles = "
          f"{worst_overall:.3e}  (tol = {tol})")
    ok_all = n_fail == 0 and worst_overall <= tol
    print("PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
