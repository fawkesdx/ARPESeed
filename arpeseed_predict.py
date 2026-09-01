"""ARPESeed first-gamma inference — load v2 weights and predict Γ from a .npy scan.

Input: float32 array shape (3, 240, 300) — three photon-energy channels, same layout as training.
Output: (phi_gamma, theta_gamma) in degrees.

Usage:
    python3 arpeseed_predict.py scan.npy
    python3 arpeseed_predict.py scan.npy --weights models/first_gamma_v2/arpeseed_first_gamma_v2.pth
    python3 arpeseed_predict.py scan.npy --plot out.png
"""

import argparse
import sys

import numpy as np
import torch
from torchvision.models import resnet18


def load_model(weights_path, device="cpu"):
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess(stack: np.ndarray) -> torch.Tensor:
    x = stack.astype(np.float32)
    if x.shape != (3, 240, 300):
        raise ValueError(f"expected shape (3, 240, 300), got {x.shape}")
    for c in range(3):
        std = x[c].std()
        x[c] = (x[c] - x[c].mean()) / (std if std > 0 else 1.0)
    return torch.from_numpy(x)


def predict(stack: np.ndarray, model, device="cpu"):
    tensor = preprocess(stack).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor).cpu().numpy()[0]
    return float(out[0]), float(out[1])  # phi, theta


def main():
    ap = argparse.ArgumentParser(description="ARPESeed first-gamma v2 inference")
    ap.add_argument("npy", help="path to (3,240,300) float scan")
    ap.add_argument(
        "--weights",
        default="models/first_gamma_v2/arpeseed_first_gamma_v2.pth",
    )
    ap.add_argument("--plot", metavar="PNG", help="optional diagnostic figure")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    stack = np.load(args.npy)
    model = load_model(args.weights, device)
    phi, theta = predict(stack, model, device)

    print(f"x_gamma (phi, slit deg): {phi:.4f}")
    print(f"y_gamma (theta, defl deg): {theta:.4f}")
    print(f"radial from origin: {np.hypot(phi, theta):.4f} deg")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, ax in enumerate(axes):
            ax.imshow(
                stack[i],
                origin="lower",
                extent=[-15, 15, -12, 12],
                aspect="auto",
                cmap="viridis",
            )
            ax.plot(phi, theta, "c+", markersize=18, markeredgewidth=2)
            ax.set_title(f"channel {i}")
            ax.set_xlabel("phi (deg)")
            ax.set_ylabel("theta (deg)")
        fig.suptitle(f"ARPESeed prediction: ({phi:.2f}, {theta:.2f}) deg")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120, bbox_inches="tight")
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
