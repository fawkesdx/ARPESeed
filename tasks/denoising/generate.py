"""Generate clean/noisy paired stacks for ARPESeed denoising task."""

import argparse
import os
from multiprocessing import Pool

import numpy as np

from arpeseed_gen_direct import draw_params, render_channel, REGIMES, _seed

DEFAULT_BASE = "corpus_denoising/dataset"


def _worker(job):
    regime_idx, folder, hv_min, hv_max, idx, base = job
    rng = np.random.default_rng(_seed(regime_idx, idx + 500_000))
    p = draw_params(rng, hv_min, hv_max)
    clean_ch, noisy_ch = [], []
    for hv in p["energies"]:
        clean = render_channel(p, hv)
        bg = p.get("background", 5.0)
        noisy = rng.poisson(clean * 1.0 + bg)
        clean_ch.append(clean)
        noisy_ch.append(noisy)
    name = f"sample_{idx:05d}"
    np.save(os.path.join(base, folder, "clean", name + ".npy"), np.asarray(clean_ch, np.float32))
    np.save(os.path.join(base, folder, "noisy", name + ".npy"), np.asarray(noisy_ch, np.float32))
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--n", type=int, default=2000, help="samples per regime")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = []
    for regime_idx, (folder, hv_min, hv_max) in enumerate(REGIMES):
        for sub in ("clean", "noisy"):
            os.makedirs(os.path.join(args.base, folder, sub), exist_ok=True)
        jobs += [(regime_idx, folder, hv_min, hv_max, i, args.base) for i in range(args.n)]

    print(f"[denoise] {len(jobs)} pairs -> {args.base}", flush=True)
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_worker, jobs, chunksize=8), 1):
            if i % 500 == 0:
                print(f"[denoise] {i}/{len(jobs)}", flush=True)
    print("[denoise] done", flush=True)


if __name__ == "__main__":
    main()
