"""Pre-compute per-channel z-score normalized copies of a first-gamma corpus.

Training normally loads raw Poisson .npy and normalizes on every epoch — disk-bound.
This writes a parallel tree with normalization done once.

Usage:
    python3 arpeseed_cache_normalized.py \\
        --src data/corpus_first_gamma_direct \\
        --dst data/corpus_first_gamma_direct_cached \\
        --workers 16
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np

REGIMES = ["range_20_70", "range_60_150", "range_350_1000"]


def normalize_stack(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        std = out[c].std()
        out[c] = (out[c] - out[c].mean()) / (std if std > 0 else 1.0)
    return out


def _job(args):
    src_path, dst_path = args
    if os.path.exists(dst_path):
        return "skip"
    arr = normalize_stack(np.load(src_path))
    np.save(dst_path, arr)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    jobs = []
    for regime in REGIMES:
        src_npy = os.path.join(args.src, regime, "npy")
        dst_npy = os.path.join(args.dst, regime, "npy")
        os.makedirs(dst_npy, exist_ok=True)
        for name in os.listdir(src_npy):
            if not name.endswith(".npy"):
                continue
            jobs.append(
                (os.path.join(src_npy, name), os.path.join(dst_npy, name))
            )
        # copy labels
        import shutil

        src_labels = os.path.join(args.src, regime, "labels.csv")
        dst_labels = os.path.join(args.dst, regime, "labels.csv")
        if os.path.exists(src_labels):
            shutil.copy2(src_labels, dst_labels)

    print(f"[cache] {len(jobs)} files src={args.src} dst={args.dst}", flush=True)
    with Pool(args.workers) as pool:
        done = 0
        for status in pool.imap_unordered(_job, jobs, chunksize=32):
            done += 1
            if done % 2000 == 0:
                print(f"[cache] {done}/{len(jobs)}", flush=True)
    print("[cache] done", flush=True)


if __name__ == "__main__":
    main()
