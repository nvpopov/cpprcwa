#!/usr/bin/env python3
"""Compare reflected field (in air) between cpprcwa and grcwa for the EUV
absorber-on-mirror example, and visualize the real-space reflected field.

Computes in grcwa: per-order reflected-field Fourier coefficients and the
real-space reconstructed field in the incident medium at z=0 (backward-only,
ai=0). Loads the cpprcwa outputs written by ex_euv_absorber --field.

Usage:
    python3 scripts/plot_reflected_field.py OUT [nG] [abs_t] [nb] [lambda] [Nx]
        OUT: prefix of the cpprcwa output files (OUT_orders.txt, OUT_grid.txt,
             OUT_hscan.txt, OUT_vscan.txt)

Outputs:
    - prints per-order comparison (max abs / relative error)
    - OUT_plot.png : 2D |E|² map (cpprcwa) + horizontal & vertical cross-sections
                     (cpprcwa vs grcwa) through the domain center
"""
import os
import sys
import time

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
from grcwa.rcwa import TranslateAmplitudes  # noqa: E402
from grcwa.fft_funs import get_ifft  # noqa: E402


def eps_from_n(n):
    return n * n


def run_grcwa(nG_req, abs_t, nb, lam, Nx):
    """Build the absorber stack and return the object."""
    cell, rect = 300.0, 60.0
    n_mo, n_si, n_ru, n_tan = (0.9226 + 0.0064j, 0.9997 + 0.0018j,
                               0.9114 + 0.0171j, 0.9562 + 0.0323j)
    epgrid = np.ones((Nx, Nx), dtype=complex)
    u0 = np.linspace(0, 1, Nx)
    u, v = np.meshgrid(u0, u0, indexing="ij")
    half = rect / (2 * cell)
    epgrid[(np.abs(u - 0.5) < half) & (np.abs(v - 0.5) < half)] = n_tan

    obj = grcwa.obj(nG_req, [cell, 0], [0, cell], 1.0 / lam, 0.0, 0.0, verbose=0)
    obj.Add_LayerUniform(1.0, eps_from_n(1.0 + 0j))
    obj.Add_LayerGrid(abs_t, Nx, Nx)
    obj.Add_LayerUniform(2.5, eps_from_n(n_ru))
    for _ in range(nb):
        obj.Add_LayerUniform(2.8, eps_from_n(n_mo))
        obj.Add_LayerUniform(4.2, eps_from_n(n_si))
    obj.Add_LayerUniform(1.0, eps_from_n(n_si))
    obj.Init_Setup()
    obj.GridLayer_geteps(epgrid.flatten())
    obj.MakeExcitationPlanewave(1, 0, 0, 0, order=0)
    return obj


def reflected_orders(obj):
    """Reflected-field Fourier coefficients (backward-only, layer 0, z=0)."""
    ai, bi = obj.GetAmplitudes_noTranslate(0)          # ai=a0, bi=b0
    aim, bim = TranslateAmplitudes(obj.q_list[0], obj.thickness_list[0], 0.0,
                                   np.zeros_like(ai), bi)
    fhxy = obj.phi_list[0] @ (aim + bim)
    fhx, fhy = fhxy[:obj.nG], fhxy[obj.nG:]
    tmp1 = (aim - bim) / obj.omega / obj.q_list[0]
    tmp2 = obj.phi_list[0] @ tmp1
    fexy = obj.kp_list[0] @ tmp2
    fey, fex = -fexy[:obj.nG], fexy[obj.nG:]
    fhz = (obj.kx * fey - obj.ky * fex) / obj.omega
    fez = (obj.ky * fhx - obj.kx * fhy) / obj.omega
    fez = fez / obj.Uniform_ep_list[0]                 # layer 0 is uniform
    return fex, fey, fez


def load_cpp_orders(path):
    data = np.loadtxt(path)
    gx = data[:, 1].astype(int)
    gy = data[:, 2].astype(int)
    ex = data[:, 3] + 1j * data[:, 4]
    ey = data[:, 5] + 1j * data[:, 6]
    ez = data[:, 7] + 1j * data[:, 8]
    return gx, gy, ex, ey, ez


def load_cpp_grid(path):
    lines = open(path).read().splitlines()
    data = np.loadtxt(lines[2:])                       # skip two header lines
    return data


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out = sys.argv[1]
    nG_req = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    abs_t = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    lam = float(sys.argv[5]) if len(sys.argv) > 5 else 13.5
    Nx = int(sys.argv[6]) if len(sys.argv) > 6 else 100

    t_py0 = time.perf_counter()
    obj = run_grcwa(nG_req, abs_t, nb, lam, Nx)
    R_py, T_py = obj.RT_Solve(normalize=1)
    t_py = time.perf_counter() - t_py0     # whole grcwa pipeline (build+setup+solve)

    perf = {"total_ms": None, "setup_ms": None, "rt_solve_ms": None}
    perf_path = out + "_perf.txt"
    if os.path.isfile(perf_path):
        for line in open(perf_path):
            k, _, v = line.partition(" ")
            perf[k.strip()] = float(v)
    print("performance (wall clock, whole solve pipeline):")
    if perf["total_ms"]:
        print(f"  cpprcwa: setup={perf['setup_ms']:.1f} ms  rt_solve={perf['rt_solve_ms']:.1f} ms  "
              f"total={perf['total_ms']:.1f} ms")
    else:
        print("  cpprcwa: (no OUT_perf.txt found; rerun with --field to get timing)")
    print(f"  grcwa:   total={t_py * 1e3:.1f} ms")
    if perf["total_ms"]:
        speedup = (t_py * 1e3) / perf["total_ms"]
        print(f"  cpprcwa/grcwa speedup = {speedup:.1f}x")

    # ── Per-order comparison ──
    gx_p, gy_p, ex_p, ey_p, ez_p = load_cpp_orders(out + "_orders.txt")
    ex_g, ey_g, ez_g = reflected_orders(obj)

    assert len(ex_p) == obj.nG, f"nG mismatch: cpp={len(ex_p)} grcwa={obj.nG}"
    # Align cpp orders to grcwa G ordering (both use the same G set, but
    # degenerate entries may be in a different order).
    G = obj.G
    idx = {tuple(G[i]): i for i in range(obj.nG)}
    cpp = np.empty_like(ex_g)
    for i in range(len(gx_p)):
        k = idx.get((gx_p[i], gy_p[i]))
        if k is not None:
            cpp[k] = ex_p[i] + ey_p[i] + ez_p[i]  # combined for total-field check
    ref = ex_g + ey_g + ez_g

    print(f"nG = {obj.nG}")
    print("order comparison: |cpp - grcwa| / max|grcwa|")
    denom = np.max(np.abs(ref))
    err = np.abs(cpp - ref) / denom
    # Report per-order relative error vs the per-order magnitude
    print(f"  max abs-err over orders = {np.max(np.abs(cpp - ref)):.3e}")
    print(f"  max rel-err (vs peak order) = {np.max(err):.3e}")
    # Recompute per-order component-wise error (ex, ey, ez separately)
    ex_c = np.empty_like(ex_g); ey_c = np.empty_like(ey_g); ez_c = np.empty_like(ez_g)
    for i in range(len(gx_p)):
        k = idx.get((gx_p[i], gy_p[i]))
        if k is not None:
            ex_c[k], ey_c[k], ez_c[k] = ex_p[i], ey_p[i], ez_p[i]
    for name, a, b in [("ex", ex_c, ex_g), ("ey", ey_c, ey_g), ("ez", ez_c, ez_g)]:
        rel = np.max(np.abs(a - b)) / np.max(np.abs(b))
        print(f"  order err {name}: max_abs={np.max(np.abs(a - b)):.3e} rel={rel:.3e}")

    # ── Real-space grid comparison ──
    Ex_g = get_ifft(Nx, Nx, ex_g, G)
    Ey_g = get_ifft(Nx, Nx, ey_g, G)
    Ez_g = get_ifft(Nx, Nx, ez_g, G)
    absE2_g = np.abs(Ex_g) ** 2 + np.abs(Ey_g) ** 2 + np.abs(Ez_g) ** 2
    data = load_cpp_grid(out + "_grid.txt")
    i = data[:, 0].astype(int); j = data[:, 1].astype(int)
    absE2_c = np.empty((Nx, Nx))
    for k in range(len(data)):
        absE2_c[i[k], j[k]] = data[k, 4]
    grid_err = np.max(np.abs(absE2_c - absE2_g))
    print(f"real-space |E|^2 grid: max diff cpp vs grcwa = {grid_err:.3e}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return 0

    h_c = np.loadtxt(out + "_hscan.txt")   # col0 pos, col1 |E|²
    v_c = np.loadtxt(out + "_vscan.txt")
    cell = 300.0
    x = np.linspace(0, cell, Nx)
    ic, jc = Nx // 2, Nx // 2
    h_g = absE2_g[ic, :]                     # y sweep at x=center
    v_g = absE2_g[:, jc]                     # x sweep at y=center

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    im = axes[0].imshow(absE2_c.T, origin="lower", extent=[0, cell, 0, cell],
                        aspect="equal", cmap="inferno")
    axes[0].axhline(cell * jc / (Nx - 1), color="w", lw=0.8, ls="--")
    axes[0].axvline(cell * ic / (Nx - 1), color="w", lw=0.8, ls="--")
    axes[0].set_title("cpprcwa reflected |E|²")
    axes[0].set_xlabel("x [nm]"); axes[0].set_ylabel("y [nm]")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].plot(x, h_g, "b-", label="grcwa")
    axes[1].plot(h_c[:, 0], h_c[:, 1], "ro", ms=3, label="cpprcwa")
    axes[1].set_title("horizontal cross-section (x = center)")
    axes[1].set_xlabel("y [nm]"); axes[1].set_ylabel("|E|²")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(x, v_g, "b-", label="grcwa")
    axes[2].plot(v_c[:, 0], v_c[:, 1], "ro", ms=3, label="cpprcwa")
    axes[2].set_title("vertical cross-section (y = center)")
    axes[2].set_xlabel("x [nm]"); axes[2].set_ylabel("|E|²")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    png = out + "_plot.png"
    fig.savefig(png, dpi=150)
    print(f"saved {png}")

    # ── Absolute field value |E| = sqrt(|E|^2) ──
    y = np.linspace(0, cell, Nx)
    absE_c = np.sqrt(np.maximum(absE2_c, 0.0))   # |E| map (cpprcwa)
    absE_g = np.sqrt(np.maximum(absE2_g, 0.0))   # |E| map (grcwa)
    figa, axesa = plt.subplots(1, 3, figsize=(15, 4.2))
    ima = axesa[0].imshow(absE_c.T, origin="lower", extent=[0, cell, 0, cell],
                          aspect="equal", cmap="inferno")
    axesa[0].axhline(cell * jc / (Nx - 1), color="w", lw=0.8, ls="--")
    axesa[0].axvline(cell * ic / (Nx - 1), color="w", lw=0.8, ls="--")
    axesa[0].set_title("cpprcwa reflected |E| (absolute value)")
    axesa[0].set_xlabel("x [nm]"); axesa[0].set_ylabel("y [nm]")
    figa.colorbar(ima, ax=axesa[0], fraction=0.046)

    axesa[1].plot(y, absE_g[ic, :], "b-", label="grcwa")
    axesa[1].plot(h_c[:, 0], np.sqrt(np.maximum(h_c[:, 1], 0.0)), "ro",
                  ms=3, label="cpprcwa")
    axesa[1].set_title("|E| — horizontal cross-section (x = center)")
    axesa[1].set_xlabel("y [nm]"); axesa[1].set_ylabel("|E|")
    axesa[1].legend(); axesa[1].grid(alpha=0.3)

    axesa[2].plot(x, absE_g[:, jc], "b-", label="grcwa")
    axesa[2].plot(v_c[:, 0], np.sqrt(np.maximum(v_c[:, 1], 0.0)), "ro",
                  ms=3, label="cpprcwa")
    axesa[2].set_title("|E| — vertical cross-section (y = center)")
    axesa[2].set_xlabel("x [nm]"); axesa[2].set_ylabel("|E|")
    axesa[2].legend(); axesa[2].grid(alpha=0.3)

    print(f"|E| grid agreement: max |cpp - grcwa| = "
          f"{np.max(np.abs(absE_c - absE_g)):.3e}")
    figa.tight_layout()
    pnga = out + "_absE.png"
    figa.savefig(pnga, dpi=150)
    print(f"saved {pnga}")

    # ── Field-component cross-sections (Ex, Ey, Ez) through the center ──
    # Files (written by ex_euv_absorber --field):
    #   pos |E|^2 Re(ex) Im(ex) Re(ey) Im(ey) Re(ez) Im(ez)
    # horizontal: x = center, y varies; vertical: y = center, x varies.
    try:
        hf = np.loadtxt(out + "_hfield.txt")
        vf = np.loadtxt(out + "_vfield.txt")
    except OSError:
        print("no *_hfield.txt/_vfield.txt found; skipping cross-section field plot")
        return 0

    # grcwa cross-sections (complex field components).
    Exh = Ex_g[ic, :]; Eyh = Ey_g[ic, :]; Ezh = Ez_g[ic, :]   # y sweep at x=center
    Exv = Ex_g[:, jc]; Eyv = Ey_g[:, jc]; Ezv = Ez_g[:, jc]   # x sweep at y=center

    def parse(p):
        return (p[:, 0], p[:, 1],
                p[:, 2] + 1j * p[:, 3], p[:, 4] + 1j * p[:, 5], p[:, 6] + 1j * p[:, 7])

    def field_plot(ax, pos_c, E_c, pos_g, E_g, title, ylab, draw_cpp):
        ax.plot(pos_g, E_g.real, "-", color="b", lw=1.2, label="grcwa Re")
        ax.plot(pos_g, E_g.imag, "--", color="c", lw=1.2, label="grcwa Im")
        if draw_cpp:
            ax.plot(pos_c, E_c.real, "o", ms=3, color="r", label="cpprcwa Re")
            ax.plot(pos_c, E_c.imag, "s", ms=3, color="m", label="cpprcwa Im")
        ax.set_title(title)
        ax.set_xlabel(ylab); ax.set_ylabel("field")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    pos_hc, _, exh, eyh, ezh = parse(hf)
    pos_vc, _, exv, eyv, ezv = parse(vf)
    y = np.linspace(0, cell, Nx)
    x = np.linspace(0, cell, Nx)

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 7.5))
    field_plot(axes2[0, 0], pos_hc, exh, y, Exh, "Ex — horizontal (x=center)", "y [nm]", True)
    field_plot(axes2[0, 1], pos_hc, eyh, y, Eyh, "Ey — horizontal (x=center)", "y [nm]", True)
    field_plot(axes2[0, 2], pos_hc, ezh, y, Ezh, "Ez — horizontal (x=center)", "y [nm]", True)
    field_plot(axes2[1, 0], pos_vc, exv, x, Exv, "Ex — vertical (y=center)", "x [nm]", True)
    field_plot(axes2[1, 1], pos_vc, eyv, x, Eyv, "Ey — vertical (y=center)", "x [nm]", True)
    field_plot(axes2[1, 2], pos_vc, ezv, x, Ezv, "Ez — vertical (y=center)", "x [nm]", True)

    # Report component-wise agreement along both cross-sections.
    print("cross-section field agreement (max |cpp - grcwa|):")
    for name, c, g in [("Ex,h", exh, Exh), ("Ey,h", eyh, Eyh), ("Ez,h", ezh, Ezh),
                       ("Ex,v", exv, Exv), ("Ey,v", eyv, Eyv), ("Ez,v", ezv, Ezv)]:
        print(f"  {name}: {np.max(np.abs(c - g)):.3e}")

    fig2.tight_layout()
    png2 = out + "_crossections.png"
    fig2.savefig(png2, dpi=150)
    print(f"saved {png2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
