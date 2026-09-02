"""ARPESeed -- first-gamma training and cross-corpus evaluation.

Two corpora differ ONLY in the angle-to-momentum mapping (see PROJECT_STATUS.md section 1):

  old : generate_full_corpus_fast.py   kx = A*sin(phi - phi0)                      (bug)
  new : arpeseed_gen_direct.py         kx = A*(sin(phi)cos(theta) - sin(phi0)cos(theta0))

Cross-evaluating both models on both corpora separates "the model learned Gamma-finding"
from "the model learned the rigid-translation shortcut the old generator accidentally
provided". A model that relies on the shortcut collapses when moved to the corrected corpus.

The 80/20 split is deterministic (seeded permutation) so every number below is comparable;
the original dataset_loader.py used an unseeded random_split.

Usage:
    # one GPU (cuda:0 only)
    python3 arpeseed_train_eval.py train --corpus new --epochs 40

    # both V100s — recommended on Einstein (see docstring at bottom of file)
    torchrun --standalone --nproc_per_node=2 arpeseed_train_eval.py train \\
        --corpus new --epochs 40 --batch-size 32 --amp

    python3 arpeseed_train_eval.py eval  --model <path> --corpus old
    python3 arpeseed_train_eval.py cross --old-model <path> --new-model <path>
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import ResNet18_Weights, resnet18

REGIMES = ["range_20_70", "range_60_150", "range_350_1000"]

_DATA_ROOT = os.environ.get("ARPESEED_DATA_ROOT", "data")
_CORPUS = {
    "old": "corpus_finding_first_gamma",
    "new": "corpus_first_gamma_direct",
    "new_cached": "corpus_first_gamma_direct_cached",
    "hardmix": "corpus_first_gamma_hardmix",
    "hardmix_cached": "corpus_first_gamma_hardmix_cached",
    "mixed_cached": [
        "corpus_first_gamma_direct_cached",
        "corpus_first_gamma_hardmix_cached",
    ],
    "benchmark": "benchmark_first_gamma_v1",
}


def resolve_corpus(corpus_key):
    spec = _CORPUS[corpus_key]
    if isinstance(spec, list):
        return [os.path.join(_DATA_ROOT, name) for name in spec]
    return os.path.join(_DATA_ROOT, spec)


CORPORA = {k: resolve_corpus(k) for k in _CORPUS}

SPLIT_SEED = 12345
VAL_FRACTION = 0.2


def setup_distributed():
    """Return (rank, world_size, local_rank, device). Single-GPU if LOCAL_RANK unset."""
    if "LOCAL_RANK" not in os.environ:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return 0, 1, 0, device

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank, device


def is_main(rank):
    return rank == 0


def unwrap(model):
    """Strip DDP wrapper so state_dict matches a plain ResNet18 checkpoint."""
    return model.module if isinstance(model, DDP) else model


def log(rank, msg):
    if is_main(rank):
        print(msg, flush=True)


class FirstGammaDataset(Dataset):
    """Per-channel z-score normalized (3, 240, 300) stacks with (x_gamma, y_gamma) labels.

    Normalization matches the original dataset_loader.py so checkpoints stay comparable.
    """

    def __init__(self, corpus_root, indices=None, already_normalized=False):
        self.already_normalized = already_normalized
        self.samples = []
        roots = corpus_root if isinstance(corpus_root, list) else [corpus_root]
        for root in roots:
            for regime in REGIMES:
                folder = os.path.join(root, regime)
                csv_path = os.path.join(folder, "labels.csv")
                npy_dir = os.path.join(folder, "npy")
                if not os.path.exists(csv_path):
                    continue
                with open(csv_path) as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        path = os.path.join(npy_dir, row[0])
                        if os.path.exists(path):
                            self.samples.append((path, float(row[1]), float(row[2])))
        self.samples.sort()
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, x_gamma, y_gamma = self.samples[idx]
        images = np.load(path).astype(np.float32)
        if not self.already_normalized:
            for c in range(images.shape[0]):
                std = images[c].std()
                images[c] = (images[c] - images[c].mean()) / (std if std > 0 else 1.0)
        return (
            torch.from_numpy(images),
            torch.tensor([x_gamma, y_gamma], dtype=torch.float32),
        )


def split_indices(corpus_root):
    n = len(FirstGammaDataset(corpus_root))
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    n_val = int(VAL_FRACTION * n)
    return perm[n_val:].tolist(), perm[:n_val].tolist()


def build_model(weights_path=None, device="cpu", use_imagenet_init=True):
    pretrained = None
    if weights_path is None and use_imagenet_init:
        pretrained = ResNet18_Weights.DEFAULT
    model = resnet18(weights=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device)


def augment(images, labels):
    """Random horizontal/vertical flips with matching label sign inversion."""
    b = images.size(0)
    h = torch.rand(b) > 0.5
    v = torch.rand(b) > 0.5
    if h.any():
        images[h] = torch.flip(images[h], dims=[3])
        labels[h, 0] = -labels[h, 0]
    if v.any():
        images[v] = torch.flip(images[v], dims=[2])
        labels[v, 1] = -labels[v, 1]
    return images, labels


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns MSE plus the interpretable radial angular error in degrees."""
    model.eval()
    sq_err = 0.0
    n = 0
    radial = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred = model(images)
        sq_err += float(((pred - labels) ** 2).sum())
        radial.append(torch.linalg.norm(pred - labels, dim=1).cpu().numpy())
        n += images.size(0)
    radial = np.concatenate(radial)
    return {
        "n": int(n),
        "mse": sq_err / (2 * n),
        "mean_radial_deg": float(radial.mean()),
        "median_radial_deg": float(np.median(radial)),
        "p95_radial_deg": float(np.percentile(radial, 95)),
    }


def make_loader(corpus_root, indices, batch_size, workers, shuffle, rank=0, world_size=1, already_normalized=False):
    dataset = FirstGammaDataset(corpus_root, indices, already_normalized=already_normalized)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=world_size > 1 and shuffle,
    ), world_size > 1


def train(corpus, epochs, batch_size, workers, out_prefix, lr, amp=False, early_stop_patience=0):
    rank, world_size, local_rank, device = setup_distributed()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    root = CORPORA[corpus]
    train_idx, val_idx = split_indices(root)
    eff_batch = batch_size * world_size
    log(
        rank,
        f"[arpeseed] corpus={corpus} train={len(train_idx)} val={len(val_idx)} "
        f"gpus={world_size} batch/gpu={batch_size} effective_batch={eff_batch} "
        f"amp={amp} device={device}",
    )

    already_norm = (
        corpus.endswith("_cached")
        or corpus == "mixed_cached"
    )
    train_loader, train_ddp = make_loader(
        root, train_idx, batch_size, workers, True, rank, world_size, already_norm
    )
    if is_main(rank):
        val_loader, _ = make_loader(
            root, val_idx, batch_size * 2, workers, False, already_normalized=already_norm
        )
    else:
        val_loader = None

    # Both ranks build the same architecture. ImageNet weights hit disk cache after
    # rank 0's first download, so rank 1 loads instantly — no split-init broadcast needed.
    model = build_model(device=device, use_imagenet_init=True)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    ckpt = f"{out_prefix}.pth"
    log_rows = []
    best = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        if train_ddp:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = augment(images, labels)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = criterion(model(images), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
            seen += images.size(0)

        if world_size > 1:
            stats = torch.tensor([running, seen], device=device, dtype=torch.float64)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            running, seen = stats.tolist()
        train_mse = running / seen

        metrics = None
        if is_main(rank):
            metrics = evaluate(unwrap(model), val_loader, device)
            scheduler.step()

            log(
                rank,
                f"[arpeseed] epoch {epoch}/{epochs} train_mse={train_mse:.4f} "
                f"val_mse={metrics['mse']:.4f} mean_radial={metrics['mean_radial_deg']:.3f} deg",
            )
            log_rows.append([epoch, train_mse, metrics["mse"], metrics["mean_radial_deg"]])

            if metrics["mse"] < best:
                best = metrics["mse"]
                epochs_no_improve = 0
                torch.save(unwrap(model).state_dict(), ckpt)
                log(rank, f"[arpeseed]   saved {ckpt}")
            else:
                epochs_no_improve += 1
                if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                    log(
                        rank,
                        f"[arpeseed] early stop at epoch {epoch} "
                        f"(no val improvement for {early_stop_patience} epochs)",
                    )

        stop_flag = torch.zeros(1, device=device, dtype=torch.int32)
        if is_main(rank) and early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            stop_flag[0] = 1
        if world_size > 1:
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            if world_size > 1:
                dist.barrier()
            break
        elif world_size > 1:
            # Keep LR scheduler in sync on all ranks (StepLR has no state beyond epoch count).
            scheduler.step()

        if world_size > 1:
            dist.barrier()

    if is_main(rank):
        with open(f"{out_prefix}_log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_mse", "val_mse", "val_mean_radial_deg"])
            w.writerows(log_rows)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ep = [r[0] for r in log_rows]
            plt.figure(figsize=(10, 6))
            plt.plot(ep, [r[1] for r in log_rows], label="Train MSE", linewidth=2)
            plt.plot(ep, [r[2] for r in log_rows], label="Val MSE", linewidth=2)
            plt.xlabel("Epoch", fontsize=13)
            plt.ylabel("MSE (deg^2)", fontsize=13)
            plt.title(
                f"ARPESeed first-gamma -- corpus '{corpus}' ({world_size} GPU)",
                fontsize=15,
            )
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.savefig(f"{out_prefix}_loss_curve.png", dpi=150, bbox_inches="tight")
        except Exception as exc:
            log(rank, f"[arpeseed] plot skipped: {exc}")

        log(rank, f"[arpeseed] best val MSE {best:.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()

    return ckpt if is_main(rank) else None


def eval_one(model_path, corpus, batch_size, workers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = CORPORA[corpus]
    if corpus == "benchmark":
        indices = list(range(len(FirstGammaDataset(root))))
    else:
        _, indices = split_indices(root)
    already_norm = corpus.endswith("_cached") or corpus == "mixed_cached"
    val_loader, _ = make_loader(
        root, indices, batch_size, workers, False, already_normalized=already_norm
    )
    model = build_model(model_path, device)
    m = evaluate(model, val_loader, device)
    print(f"[arpeseed] {os.path.basename(model_path)} on '{corpus}' -> {json.dumps(m)}", flush=True)
    return m


def cross(old_model, new_model, batch_size, workers, out_path):
    results = {}
    for model_name, path in (("old_model", old_model), ("new_model", new_model)):
        for corpus in ("old", "new"):
            results[f"{model_name}|{corpus}_corpus"] = eval_one(
                path, corpus, batch_size, workers
            )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== ARPESeed cross-evaluation (validation splits, radial error in degrees) ===")
    header = f"{'':<12}{'old corpus':>26}{'new corpus':>26}"
    print(header)
    for model_name in ("old_model", "new_model"):
        cells = []
        for corpus in ("old", "new"):
            m = results[f"{model_name}|{corpus}_corpus"]
            cells.append(f"MSE {m['mse']:.3f} / {m['mean_radial_deg']:.3f} deg")
        print(f"{model_name:<12}{cells[0]:>26}{cells[1]:>26}")
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--corpus", choices=list(CORPORA), default="new")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--workers", type=int, default=10)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--out-prefix", default="arpeseed_first_gamma_new")
    t.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="stop if val MSE does not improve for N epochs (0=disabled; try 5)",
    )
    t.add_argument("--amp", action="store_true", help="mixed-precision training on CUDA")

    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--corpus", choices=list(CORPORA), required=True)
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--workers", type=int, default=10)

    c = sub.add_parser("cross")
    c.add_argument("--old-model", required=True)
    c.add_argument("--new-model", required=True)
    c.add_argument("--batch-size", type=int, default=64)
    c.add_argument("--workers", type=int, default=10)
    c.add_argument("--out", default="arpeseed_cross_eval.json")

    a = ap.parse_args()
    if a.cmd == "train":
        train(a.corpus, a.epochs, a.batch_size, a.workers, a.out_prefix, a.lr, a.amp, a.early_stop_patience)
    elif a.cmd == "eval":
        eval_one(a.model, a.corpus, a.batch_size, a.workers)
    else:
        cross(a.old_model, a.new_model, a.batch_size, a.workers, a.out)
