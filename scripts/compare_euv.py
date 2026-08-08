#!/usr/bin/env python3
"""Compare EUV multilayer / absorber reflectivity: grcwa vs cpprcwa.

Runs the Mo/Si multilayer (optionally with a TaN absorber pattern on top)
through grcwa and prints the reflectivity, which the C++ examples reproduce.

Usage:
    python3 scripts/compare_euv.py [nb] [lambda_nm]                 # bare mirror
    python3 scripts/compare_euv.py --absorber [nG] [abs_t_nm] [Nx]  # with pattern
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grcwa_orig"))
_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)
for cand in (os.path.join(_repo, "..", "grcwa_orig"),
             os.path.join(os.path.dirname(_repo), "grcwa_orig")):
    if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "grcwa")):
        sys.path.insert(0, cand)
        break
import grcwa  # noqa: E402


def eps_from_n(n):
    return n * n


def build(nb, lam, absorber=None):
    """absorber = (nG, abs_t_nm, Nx) or None for a bare planar mirror."""
    n_mo = 0.9226 + 0.0064j
    n_si = 0.9997 + 0.0018j
    n_ru = 0.9114 + 0.0171j
    n_tan = 0.9562 + 0.0323j

    if absorber:
        nG, abs_t, Nx = absorber
        cell = 300.0
        rect = 60.0
        obj = grcwa.obj(nG, [cell, 0], [0, cell], 1.0 / lam, 0.0, 0.0, verbose=0)
        epgrid = np.ones((Nx, Nx), dtype=complex)
        u0 = np.linspace(0, 1, Nx)
        u, v = np.meshgrid(u0, u0, indexing="ij")
        half = rect / (2 * cell)
        epgrid[(np.abs(u - 0.5) < half) & (np.abs(v - 0.5) < half)] = n_tan
        obj.Add_LayerUniform(1.0, eps_from_n(1.0 + 0j))
        obj.Add_LayerGrid(abs_t, Nx, Nx)
        obj.Add_LayerUniform(2.5, eps_from_n(n_ru))
        for _ in range(nb):
            obj.Add_LayerUniform(2.8, eps_from_n(n_mo))
            obj.Add_LayerUniform(4.2, eps_from_n(n_si))
        obj.Add_LayerUniform(1.0, eps_from_n(n_si))
        obj.Init_Setup()
        obj.GridLayer_geteps(epgrid.flatten())
    else:
        obj = grcwa.obj(2, [1.0, 0], [0, 1.0], 1.0 / lam, 0.0, 0.0, verbose=0)
        obj.Add_LayerUniform(1.0, eps_from_n(1.0 + 0j))
        obj.Add_LayerUniform(2.5, eps_from_n(n_ru))
        for _ in range(nb):
            obj.Add_LayerUniform(2.8, eps_from_n(n_mo))
            obj.Add_LayerUniform(4.2, eps_from_n(n_si))
        obj.Add_LayerUniform(1.0, eps_from_n(n_si))
        obj.Init_Setup()

    obj.MakeExcitationPlanewave(1, 0, 0, 0, order=0)
    R, T = obj.RT_Solve(normalize=1)
    return obj.nG, R, T


def main():
    absorber = None
    args = [a for a in sys.argv[1:]]
    if "--absorber" in args:
        absorber = (int(args[1]), float(args[2]), int(args[3]))
        nb = 40
        lam = 13.5
    else:
        nb = int(args[0]) if args else 40
        lam = float(args[1]) if len(args) > 1 else 13.5

    nG, R, T = build(nb, lam, absorber)
    if absorber:
        nG_req, abs_t, Nx = absorber
        print(f"grcwa: absorber nG_req={nG_req}->nG={nG} t_abs={abs_t} Nx={Nx}  "
              f"R={R:.6f}  T={T:.6g}  R+T={R + T:.6f}")
        print(f"cpprcwa:  ./examples/ex_euv_absorber {nG_req} {abs_t} {nb} {lam:.2f} {Nx}")
    else:
        print(f"grcwa:  nb={nb} lambda={lam:.2f} nm  R={R:.6f}  T={T:.6g}  R+T={R + T:.6f}")
        print("cpprcwa:  ./examples/ex_euv_multilayer %d %.2f" % (nb, lam))


if __name__ == "__main__":
    main()
