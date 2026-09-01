"""ARPESeed -- held-out out-of-distribution benchmark for the first-gamma task.

The point of this benchmark is that it is NOT drawn from the training generator. Ekahana et
al. section 9 argues shared models are only useful if independent benchmarks exist and domain
shift is documented, so the released test set deliberately violates the training
distribution along several axes at once:

  axis                  training corpus                 this benchmark
  --------------------  ------------------------------  ---------------------------------
  lattice families      square / rect / hexagonal       + oblique (non-orthogonal axes)
  band count            single band                     + two-band (split / folded)
  spectral lineshape    Gaussian cut, width 0.2-0.8     Lorentzian cut, width 0.05-0.4
  background counts     5.0 (fixed)                     2.0 / 5.0 / 10.0
  Gamma range           +/- 8 deg                       +/- 10 deg (extrapolation band)
  hv steps              +5..15, +15..30 eV              +2..6 eV (small steps, tight budget)
  detector artifacts    none                            slit shadowing + dead stripes

The angle-to-momentum mapping is the physically correct one (same as
arpeseed_gen_direct.py); only the *material and instrument* distributions shift. A model that
learned photoemission kinematics should degrade gracefully here. A model that memorized the
training generator's quirks should fall apart.

Publish this test set and its labels. Do not train on it.

Usage:
    python3 arpeseed_benchmark.py generate [--n 500] [--base <dir>]
    python3 arpeseed_benchmark.py preview  [--out arpeseed_benchmark_preview.png]
"""

import argparse
import csv
import os

import numpy as np

from arpeseed_gen_direct import (
    K_SCALE,
    PH_GRID,
    PH_LIM_DEG,
    TH_GRID,
    TH_LIM_DEG,
)

DEFAULT_BASE = os.environ.get(
    "ARPESEED_BENCHMARK_DIR",
    os.path.join(os.environ.get("ARPESEED_DATA_ROOT", "data"), "benchmark_first_gamma_v1"),
)
BENCHMARK_SEED = 987_654_321

# Same three regimes as the training corpus, so per-regime scores are comparable.
REGIMES = [
    ("range_20_70", 20.0, 70.0),
    ("range_60_150", 60.0, 150.0),
    ("range_350_1000", 350.0, 1000.0),
]


def draw_params(rng, hv_min, hv_max):
    base_hv = rng.uniform(hv_min, hv_max - 12.0)
    lattice = rng.choice(["square", "rectangular", "hexagonal", "oblique"])
    a = rng.uniform(2.0, 7.5)
    return dict(
        th0_deg=rng.uniform(-10.0, 10.0),
        ph0_deg=rng.uniform(-10.0, 10.0),
        # Small photon-energy steps: the realistic tight-beamtime-budget case.
        energies=(
            base_hv,
            base_hv + rng.uniform(2.0, 6.0),
            base_hv + rng.uniform(6.0, 12.0),
        ),
        lattice=lattice,
        a=a,
        b=(rng.uniform(2.0, 7.5) if lattice in ("rectangular", "oblique") else a),
        gamma_angle=(rng.uniform(65.0, 115.0) if lattice == "oblique" else 90.0),
        c=rng.uniform(3.0, 16.0),
        tx=rng.uniform(0.4, 2.2),
        ty=rng.uniform(0.4, 2.2),
        tz=rng.uniform(0.05, 1.6),
        two_band=bool(rng.random() < 0.35),
        band_split=rng.uniform(0.1, 0.6),
        e_cut=rng.uniform(-1.8, 1.8),
        cut_width=rng.uniform(0.05, 0.4),
        v0=rng.uniform(8.0, 24.0),
        work_function=rng.uniform(3.8, 5.4),
        background=float(rng.choice([2.0, 5.0, 10.0])),
        slit_shadow=bool(rng.random() < 0.4),
        dead_stripes=int(rng.integers(0, 3)),
    )


def render_channel(p, hv):
    """Clean isoenergy image, peak-normalized. Correct momentum-space Gamma offset."""
    e_kin = hv - p["work_function"]
    k_par = K_SCALE * np.sqrt(e_kin)

    th0, ph0 = np.radians(p["th0_deg"]), np.radians(p["ph0_deg"])
    kx = k_par * (np.sin(PH_GRID) * np.cos(TH_GRID) - np.sin(ph0) * np.cos(th0))
    ky = k_par * (np.sin(TH_GRID) - np.sin(th0))

    cos2 = np.clip(1.0 - np.sin(TH_GRID) ** 2 - np.sin(PH_GRID) ** 2, 0.0, 1.0)
    kz = K_SCALE * np.sqrt(e_kin * cos2 + p["v0"])

    a, b, c = p["a"], p["b"], p["c"]
    tx, ty, tz = p["tx"], p["ty"], p["tz"]

    if p["lattice"] == "hexagonal":
        band = tx * np.cos(kx * a) + ty * 2.0 * np.cos(kx * a / 2.0) * np.cos(
            np.sqrt(3.0) * ky * a / 2.0
        )
    elif p["lattice"] == "oblique":
        # Non-orthogonal in-plane axes: a1 along x, a2 rotated by gamma_angle.
        g = np.radians(p["gamma_angle"])
        band = tx * np.cos(kx * a) + ty * np.cos(
            (kx * np.cos(g) + ky * np.sin(g)) * b
        )
    else:
        band = tx * np.cos(kx * a) + ty * np.cos(ky * b)
    band = band + tz * np.cos(kz * c)

    def lineshape(offset):
        w = p["cut_width"]
        return w**2 / ((band - p["e_cut"] - offset) ** 2 + w**2)

    clean = lineshape(0.0)
    if p["two_band"]:
        clean = clean + 0.7 * lineshape(p["band_split"])

    clean[~np.isfinite(clean)] = 0.0
    return clean / (clean.max() + 1e-9)


def apply_detector_artifacts(clean, p, rng):
    """Slit shadowing (cos^4 vignetting) and dead detector stripes."""
    out = clean
    if p["slit_shadow"]:
        vign = np.cos(PH_GRID) ** 4 * np.cos(TH_GRID) ** 2
        out = out * (vign / vign.max())
    for _ in range(p["dead_stripes"]):
        col = int(rng.integers(0, out.shape[1]))
        width = int(rng.integers(1, 4))
        out = out.copy()
        out[:, col : col + width] = 0.0
    return out


def render_sample(p, rng):
    channels = []
    for hv in p["energies"]:
        clean = apply_detector_artifacts(render_channel(p, hv), p, rng)
        channels.append(rng.poisson(clean * 1.0 + p["background"]))
    return np.asarray(channels, dtype=np.float32)


def _seed(regime_idx, i):
    return np.random.SeedSequence([BENCHMARK_SEED, regime_idx, i])


def generate(base, n_per_regime):
    for regime_idx, (folder_name, hv_min, hv_max) in enumerate(REGIMES):
        npy_dir = os.path.join(base, folder_name, "npy")
        os.makedirs(npy_dir, exist_ok=True)
        rows = []
        for i in range(n_per_regime):
            rng = np.random.default_rng(_seed(regime_idx, i))
            p = draw_params(rng, hv_min, hv_max)
            name = f"sample_{i:05d}.npy"
            np.save(os.path.join(npy_dir, name), render_sample(p, rng))
            rows.append(
                [name, f"{p['ph0_deg']:.4f}", f"{p['th0_deg']:.4f}", p["lattice"]]
            )

        with open(os.path.join(base, folder_name, "labels.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "x_gamma", "y_gamma"])
            for r in rows:
                w.writerow(r[:3])

        # Side-car metadata for per-family error breakdown; not read by the loader.
        with open(os.path.join(base, folder_name, "meta.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "x_gamma", "y_gamma", "lattice"])
            w.writerows(rows)

        print(f"[benchmark] {folder_name}: {len(rows)} samples", flush=True)
    print(f"[benchmark] done -> {base}", flush=True)


def preview(out_path, n_samples=3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [-PH_LIM_DEG, PH_LIM_DEG, -TH_LIM_DEG, TH_LIM_DEG]
    fig, axes = plt.subplots(2 * n_samples, 3, figsize=(15, 3.6 * 2 * n_samples))

    for s in range(n_samples):
        rng = np.random.default_rng(_seed(1, s))
        p = draw_params(rng, 60.0, 150.0)
        noisy = render_sample(p, rng)
        rng2 = np.random.default_rng(_seed(1, s))
        draw_params(rng2, 60.0, 150.0)

        for ch, hv in enumerate(p["energies"]):
            clean = apply_detector_artifacts(render_channel(p, hv), p, rng2)
            for row_off, img, tag in ((0, clean, "clean"), (1, noisy[ch], "noisy")):
                ax = axes[2 * s + row_off, ch]
                ax.imshow(img, origin="lower", extent=extent, aspect="auto", cmap="magma")
                ax.plot(p["ph0_deg"], p["th0_deg"], "c+", markersize=22, markeredgewidth=3)
                ax.set_title(f"{tag}  hv={hv:.0f} eV", fontsize=9)
                ax.set_xlabel("phi / slit (deg)")
                if ch == 0:
                    ax.set_ylabel(
                        f"{p['lattice']} g={p['gamma_angle']:.0f}deg\n"
                        f"two_band={p['two_band']} bkg={p['background']:.0f}\n"
                        f"shadow={p['slit_shadow']} stripes={p['dead_stripes']}\n"
                        f"Gamma=({p['ph0_deg']:.2f}, {p['th0_deg']:.2f})",
                        fontsize=8,
                    )

    fig.suptitle(
        "ARPESeed first-gamma BENCHMARK v1 (out-of-distribution) -- cyan cross = label",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"[benchmark] wrote {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--base", default=DEFAULT_BASE)
    g.add_argument("--n", type=int, default=500, help="samples per regime")

    p_ = sub.add_parser("preview")
    p_.add_argument("--out", default="arpeseed_benchmark_preview.png")
    p_.add_argument("--n", type=int, default=3)

    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a.base, a.n)
    else:
        preview(a.out, a.n)
