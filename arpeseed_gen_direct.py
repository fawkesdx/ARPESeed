"""ARPESeed -- first-gamma corpus generator (direct angle-grid physics).

Replaces the Aurelia-based generator. Tight-binding bands are evaluated directly on the
detector angle grid, so there is no Make_angle_conv / griddata interpolation step and no
quantization of the Gamma offset onto a discrete k mesh. See PROJECT_STATUS.md section 3.

Parameter distributions deliberately mirror generate_full_corpus_fast.py (the generator
behind the trained best_gamma_model.pth) so that retraining isolates one single change:
the angle-to-momentum mapping. The old generator offset Gamma in ANGLE before the sine,

    kx = A*sin(phi - phi0)                      # old, unphysical

which makes the pattern a rigid angular translation toward the label. The correct mapping
offsets in MOMENTUM and keeps the cos(theta) projection,

    kx = A*sin(phi)*cos(theta) - A*sin(phi0)*cos(theta0)
    ky = A*sin(theta)          - A*sin(theta0)

so the pattern warps as well as translates, which is what a tilted sample really produces.

Output layout is drop-in compatible with ML_pipeline/supervised/first_gamma/dataset_loader.py:

    <base>/<regime>/npy/sample_XXXXX.npy    float32, shape (3, 240, 300)
    <base>/<regime>/labels.csv              filename,x_gamma,y_gamma

x_gamma is phi (slit angle, image X axis), y_gamma is theta (deflection, image Y axis).

Usage:
    python3 arpeseed_gen_direct.py verify [--out grid.png]
    python3 arpeseed_gen_direct.py generate [--workers 32] [--per-regime 16667]
"""

import argparse
import csv
import os

import numpy as np

DEFAULT_BASE = os.environ.get(
    "ARPESEED_CORPUS_DIR",
    os.path.join(os.environ.get("ARPESEED_DATA_ROOT", "data"), "corpus_first_gamma_direct"),
)

N_TH, N_PH = 240, 300
TH_LIM_DEG, PH_LIM_DEG = 12.0, 15.0
K_SCALE = 0.512  # sqrt(2 m_e)/hbar in Angstrom^-1 eV^-1/2

REGIMES = [
    ("range_20_70", 20.0, 70.0),
    ("range_60_150", 60.0, 150.0),
    ("range_350_1000", 350.0, 1000.0),
]

_TH = np.linspace(np.radians(-TH_LIM_DEG), np.radians(TH_LIM_DEG), N_TH)
_PH = np.linspace(np.radians(-PH_LIM_DEG), np.radians(PH_LIM_DEG), N_PH)
PH_GRID, TH_GRID = np.meshgrid(_PH, _TH)


def draw_params(rng, hv_min, hv_max, hard_mix=False):
    """Sample synthetic material + geometry. hard_mix=True blends OOD-style diversity."""
    base_hv = rng.uniform(hv_min, hv_max - (12.0 if hard_mix else 30.0))
    gamma_lim = 10.0 if hard_mix else 8.0

    if hard_mix and rng.random() < 0.15:
        lattice = "oblique"
        a = rng.uniform(2.0, 7.5)
        b = rng.uniform(2.0, 7.5)
        gamma_angle = rng.uniform(65.0, 115.0)
    else:
        lattice = rng.choice(["square", "hexagonal", "rectangular"])
        a = rng.uniform(2.0, 7.0)
        b = rng.uniform(2.0, 7.0) if lattice == "rectangular" else a
        gamma_angle = 90.0

    return dict(
        th0_deg=rng.uniform(-gamma_lim, gamma_lim),
        ph0_deg=rng.uniform(-gamma_lim, gamma_lim),
        energies=(
            base_hv,
            base_hv + (rng.uniform(2.0, 6.0) if hard_mix else rng.uniform(5.0, 15.0)),
            base_hv + (rng.uniform(6.0, 12.0) if hard_mix else rng.uniform(15.0, 30.0)),
        ),
        lattice=lattice,
        a=a,
        b=b,
        gamma_angle=gamma_angle,
        c=rng.uniform(3.0, 15.0),
        tx=rng.uniform(0.4, 2.2),
        ty=rng.uniform(0.4, 2.2),
        tz=rng.uniform(0.05, 1.6),
        two_band=bool(hard_mix and rng.random() < 0.25),
        band_split=rng.uniform(0.1, 0.6),
        e_cut=rng.uniform(-1.8, 1.8),
        cut_width=rng.uniform(0.05, 0.4) if hard_mix else rng.uniform(0.2, 0.8),
        lineshape=rng.choice(["lorentzian", "gaussian"]) if hard_mix else "gaussian",
        v0=rng.uniform(8.0, 24.0) if hard_mix else rng.uniform(10.0, 20.0),
        work_function=rng.uniform(3.8, 5.4) if hard_mix else rng.uniform(4.0, 5.0),
        background=float(rng.choice([2.0, 5.0, 10.0])) if hard_mix else 5.0,
    )


def render_channel(p, hv):
    """Clean isoenergy detector image at photon energy hv, normalized to peak 1.0."""
    e_kin = hv - p["work_function"]
    k_par = K_SCALE * np.sqrt(e_kin)

    th0, ph0 = np.radians(p["th0_deg"]), np.radians(p["ph0_deg"])

    # Momentum-space offset of the Gamma point, and the emission-sphere projection.
    kx = k_par * (np.sin(PH_GRID) * np.cos(TH_GRID) - np.sin(ph0) * np.cos(th0))
    ky = k_par * (np.sin(TH_GRID) - np.sin(th0))

    cos2 = np.clip(1.0 - np.sin(TH_GRID) ** 2 - np.sin(PH_GRID) ** 2, 0.0, 1.0)
    kz = K_SCALE * np.sqrt(e_kin * cos2 + p["v0"])

    a, b, c = p["a"], p["b"], p["c"]
    tx, ty, tz = p["tx"], p["ty"], p["tz"]

    if p["lattice"] == "hexagonal":
        val = tx * np.cos(kx * a) + ty * 2.0 * np.cos(kx * a / 2.0) * np.cos(
            np.sqrt(3.0) * ky * a / 2.0
        )
    elif p["lattice"] == "oblique":
        g = np.radians(p["gamma_angle"])
        val = tx * np.cos(kx * a) + ty * np.cos((kx * np.cos(g) + ky * np.sin(g)) * b)
    else:
        val = tx * np.cos(kx * a) + ty * np.cos(ky * b)
    val = val + tz * np.cos(kz * c)

    if p.get("lineshape") == "lorentzian":
        w = p["cut_width"]
        band = w ** 2 / ((val - p["e_cut"]) ** 2 + w ** 2)
        if p.get("two_band"):
            band = band + 0.7 * w ** 2 / (
                (val - p["e_cut"] - p["band_split"]) ** 2 + w ** 2
            )
        clean = band
    else:
        clean = np.exp(-((val - p["e_cut"]) ** 2) / p["cut_width"])
        if p.get("two_band"):
            clean = clean + 0.7 * np.exp(
                -((val - p["e_cut"] - p["band_split"]) ** 2) / p["cut_width"]
            )
    clean[~np.isfinite(clean)] = 0.0
    return clean


def render_sample(p, rng):
    """Stack of 3 noisy channels, shape (3, 240, 300) float32.

    Noise model is the locked-in ultra-fast 1-second scan: peak band signal 1.0 count on a
    5.0 count background, pure Poisson.
    """
    bg = p.get("background", 5.0)
    channels = [
        rng.poisson(render_channel(p, hv) * 1.0 + bg) for hv in p["energies"]
    ]
    return np.asarray(channels, dtype=np.float32)


def _seed(regime_idx, sample_idx):
    return np.random.SeedSequence([20260831, regime_idx, sample_idx])


def _worker(job):
    regime_idx, folder_name, hv_min, hv_max, index, base, hard_mix = job
    out_dir = os.path.join(base, folder_name, "npy")
    name = f"sample_{index:05d}.npy"
    path = os.path.join(out_dir, name)

    rng = np.random.default_rng(_seed(regime_idx, index))
    p = draw_params(rng, hv_min, hv_max, hard_mix=hard_mix)

    if not os.path.exists(path):
        np.save(path, render_sample(p, rng))
    return folder_name, name, p["ph0_deg"], p["th0_deg"]


def generate(base, per_regime, workers, hard_mix=False):
    import multiprocessing as mp

    jobs = []
    for regime_idx, (folder_name, hv_min, hv_max) in enumerate(REGIMES):
        os.makedirs(os.path.join(base, folder_name, "npy"), exist_ok=True)
        jobs += [
            (regime_idx, folder_name, hv_min, hv_max, i, base, hard_mix)
            for i in range(per_regime)
        ]

    print(f"[arpeseed] {len(jobs)} samples, {workers} workers -> {base}", flush=True)

    records = {name: [] for name, _, _ in REGIMES}
    with mp.Pool(workers) as pool:
        for n, (folder_name, name, x_gamma, y_gamma) in enumerate(
            pool.imap_unordered(_worker, jobs, chunksize=16), start=1
        ):
            records[folder_name].append((name, x_gamma, y_gamma))
            if n % 1000 == 0:
                print(f"[arpeseed] {n}/{len(jobs)}", flush=True)

    for folder_name, rows in records.items():
        rows.sort()
        with open(os.path.join(base, folder_name, "labels.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "x_gamma", "y_gamma"])
            for name, x_gamma, y_gamma in rows:
                w.writerow([name, f"{x_gamma:.4f}", f"{y_gamma:.4f}"])
        print(f"[arpeseed] wrote {folder_name}/labels.csv ({len(rows)} rows)", flush=True)

    print("[arpeseed] done", flush=True)


def verify(out_path, n_samples=3):
    """Clean physics over noisy input, Gamma crosshair overlaid, for visual inspection.

    The 1-count signal on a 5-count background is invisible by design, so geometry must be
    judged on the clean row; the noisy row only confirms the count statistics.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [-PH_LIM_DEG, PH_LIM_DEG, -TH_LIM_DEG, TH_LIM_DEG]
    fig, axes = plt.subplots(2 * n_samples, 3, figsize=(15, 3.6 * 2 * n_samples))

    for s in range(n_samples):
        rng = np.random.default_rng(_seed(1, 10_000 + s))
        p = draw_params(rng, 60.0, 150.0)
        noisy = render_sample(p, rng)

        for ch, hv in enumerate(p["energies"]):
            clean = render_channel(p, hv)

            for row_off, img, tag in ((0, clean, "clean"), (1, noisy[ch], "noisy")):
                ax = axes[2 * s + row_off, ch]
                ax.imshow(
                    img, origin="lower", extent=extent, aspect="auto", cmap="viridis"
                )
                ax.plot(
                    p["ph0_deg"], p["th0_deg"], "g+", markersize=22, markeredgewidth=3
                )
                ax.set_title(f"{tag}  hv={hv:.0f} eV", fontsize=9)
                ax.set_xlabel("phi / slit (deg)")
                if ch == 0:
                    ax.set_ylabel(
                        f"{p['lattice']}  a={p['a']:.1f} b={p['b']:.1f} c={p['c']:.1f}\n"
                        f"tz={p['tz']:.2f} Ecut={p['e_cut']:.2f}\n"
                        f"Gamma=({p['ph0_deg']:.2f}, {p['th0_deg']:.2f})",
                        fontsize=8,
                    )

    fig.suptitle(
        "ARPESeed direct-physics generator -- green cross = true Gamma label\n"
        "check: pattern contracts toward the cross as hv increases (clean rows)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"[arpeseed] wrote {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify")
    v.add_argument("--out", default="arpeseed_verify.png")
    v.add_argument("--n", type=int, default=3)

    g = sub.add_parser("generate")
    g.add_argument("--base", default=DEFAULT_BASE)
    g.add_argument("--per-regime", type=int, default=16667)
    g.add_argument("--workers", type=int, default=32)
    g.add_argument(
        "--hard-mix",
        action="store_true",
        help="OOD-style diversity: oblique, two-band, Lorentzian, varied background",
    )

    args = ap.parse_args()
    if args.cmd == "verify":
        verify(args.out, args.n)
    else:
        generate(args.base, args.per_regime, args.workers, hard_mix=args.hard_mix)
