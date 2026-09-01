"""Train a small CNN denoiser on synthetic ARPES isoenergy pairs."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class DenoiseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 3, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class PairDataset(Dataset):
    def __init__(self, base, regimes):
        self.base = base
        self.items = []
        for regime in regimes:
            clean_dir = os.path.join(base, regime, "clean")
            for name in os.listdir(clean_dir):
                if name.endswith(".npy"):
                    self.items.append((regime, name))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        regime, name = self.items[idx]
        base = os.path.join(self.base, regime)
        noisy = np.load(os.path.join(base, "noisy", name)).astype(np.float32)
        clean = np.load(os.path.join(base, "clean", name)).astype(np.float32)
        for arr in (noisy, clean):
            for c in range(3):
                std = arr[c].std()
                arr[c] = (arr[c] - arr[c].mean()) / (std if std > 0 else 1.0)
        return torch.from_numpy(noisy), torch.from_numpy(clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="corpus_denoising/dataset")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="models/denoising/arpeseed_denoise_v1.pth")
    args = ap.parse_args()

    regimes = [d for d in os.listdir(args.base) if d.startswith("range_")]
    ds = PairDataset(args.base, regimes)
    n_val = max(1, len(ds) // 5)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = DenoiseCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best = float("inf")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            opt.zero_grad()
            pred = model(noisy)
            loss = loss_fn(pred, clean)
            loss.backward()
            opt.step()
            tr += loss.item() * noisy.size(0)
        tr /= len(train_idx)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                va += loss_fn(model(noisy), clean).item() * noisy.size(0)
        va /= len(val_idx)
        print(f"epoch {epoch}/{args.epochs} train_mse={tr:.4f} val_mse={va:.4f}", flush=True)
        if va < best:
            best = va
            torch.save(model.state_dict(), args.out)
            print(f"  saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
