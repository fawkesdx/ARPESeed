"""ARPESeed first-gamma inference — load v2 weights and predict Γ from a .npy scan.

Input: float32 array —
  (3, H, W) preferred — three different photon-energy isoenergy maps
  (1, H, W) or (H, W) — single map; duplicated to 3 channels with a warning
Any spatial size is bilinear-resized to the training grid (3, 240, 300).
Output: (phi_gamma, theta_gamma) in degrees (training FOV units: φ ±15°, θ ±12°).

Usage:
    python3 arpeseed_predict.py scan.npy
    python3 arpeseed_predict.py scan.npy --weights models/first_gamma_v2/arpeseed_first_gamma_v2.pth
    python3 arpeseed_predict.py scan.npy --plot out.png
"""

import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18

MODEL_HW = (240, 300)  # (N_theta, N_phi) — training spatial size
SINGLE_CHANNEL_MSG = (
    "only one isoenergy map given; model was trained on three different "
    "photon energies (band pattern scales with hν). Duplicating this image "
    "into a 3-channel stack so inference can run — expect lower accuracy. "
    "Prefer three hν channels when possible."
)


def load_model(weights_path, device="cpu"):
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def ensure_three_channels(stack: np.ndarray) -> np.ndarray:
    """Map (H,W) / (1,H,W) / (3,H,W) → (3,H,W). Duplicate if single map."""
    x = np.asarray(stack, dtype=np.float32)
    if x.ndim == 2:
        warnings.warn(SINGLE_CHANNEL_MSG, stacklevel=2)
        return np.stack([x, x, x], axis=0)
    if x.ndim == 3 and x.shape[0] == 1:
        warnings.warn(SINGLE_CHANNEL_MSG, stacklevel=2)
        return np.concatenate([x, x, x], axis=0)
    if x.ndim == 3 and x.shape[0] == 3:
        return x
    raise ValueError(
        f"expected (H, W), (1, H, W), or (3, H, W), got {x.shape}"
    )


def to_model_grid(stack: np.ndarray) -> np.ndarray:
    """Ensure (3, 240, 300). Accepts 1 or 3 channels; bilinear resize if needed."""
    x = ensure_three_channels(stack)
    if x.shape[1:] == MODEL_HW:
        return x
    warnings.warn(
        f"resizing input {tuple(np.asarray(stack).shape)} → "
        f"(3, {MODEL_HW[0]}, {MODEL_HW[1]}) to match training grid",
        stacklevel=2,
    )
    t = torch.from_numpy(x).unsqueeze(0)  # (1, 3, H, W)
    t = F.interpolate(t, size=MODEL_HW, mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


def preprocess(stack: np.ndarray) -> torch.Tensor:
    x = to_model_grid(stack)
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
    ap = argparse.ArgumentParser(description="ARPESeed first-gamma inference")
    ap.add_argument("npy", help="path to (3,H,W), (1,H,W), or (H,W) float scan")
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

    print(f"input shape: {tuple(np.asarray(stack).shape)} → model grid (3, {MODEL_HW[0]}, {MODEL_HW[1]})")
    print(f"x_gamma (phi, slit deg): {phi:.4f}")
    print(f"y_gamma (theta, defl deg): {theta:.4f}")
    print(f"radial from origin: {np.hypot(phi, theta):.4f} deg")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grid = to_model_grid(stack)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, ax in enumerate(axes):
            ax.imshow(
                grid[i],
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
