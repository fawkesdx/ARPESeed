# Model Card — `arpeseed/first-gamma-v2.1` (recommended)

> **Status:** trained and evaluated on synthetic data.
> **Supersedes:** v2 / v2-es for OOD and varied beamline conditions.
> **Retired:** `arpeseed/first-gamma-v1` — tombstone in `archive/first_gamma_v1_geometry_bug/`.

## Summary

| | |
| --- | --- |
| **Project** | ARPESeed — task-oriented mini models for ARPES, distributed as weights |
| **Task** | Locate the first Brillouin-zone Γ point in noisy, ultra-fast ARPES isoenergy maps |
| **Type** | Supervised 2D coordinate regression |
| **Architecture** | ResNet18 (ImageNet init), `fc` → `Linear(512, 2)` |
| **Parameters** | 11.2 M — ~44.8 MB |
| **Recommended checkpoint** | **`arpeseed_first_gamma_v2_1.pth`** — hard-mix training, best OOD |
| **Alternate (easy val)** | `arpeseed_first_gamma_v2.pth` — 1.79° on easy-corpus val |
| **Trained** | 2026-09-01, 2× Tesla V100, DDP, AMP, early stop |
| **Generator** | `arpeseed_gen_direct.py --hard-mix` |
| **License** | TBD |
| **Contact** | Sandy Adhitia Ekahana — sekahana@lbl.gov |

## Intended use

Coarse live alignment from a 1-second test scan (three photon energies). **Prefer v2.1** when
sample geometry, lineshape, or background may differ from the easy training set. Use v2 if you
know scans match the original square/hex/rect Gaussian training distribution.

Not a substitute for full Fermi-surface mapping. No uncertainty estimate.

## Inputs and outputs

`float32` `(3, 240, 300)`, φ ∈ ±15°, θ ∈ ±12°, per-channel z-score normalization.
Output `(x_gamma, y_gamma)` in degrees (φ, θ).

```python
import numpy as np, torch
from torchvision.models import resnet18

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("arpeseed_first_gamma_v2_1.pth", map_location="cpu"))
model.eval()

x = np.load("scan.npy").astype(np.float32)
for c in range(3):
    x[c] = (x[c] - x[c].mean()) / (x[c].std() or 1.0)

with torch.no_grad():
    phi_gamma, theta_gamma = model(torch.from_numpy(x)[None]).numpy()[0]
```

Or: `python arpeseed_predict.py scan.npy --weights arpeseed_first_gamma_v2_1.pth`

## Training data

50,001 synthetic samples from `arpeseed_gen_direct.py --hard-mix`:

- Oblique lattices (~15%), two-band spectra (~25%), Lorentzian + Gaussian lineshapes
- Background 2 / 5 / 10 counts, ±10° Γ, small hν steps
- Same correct momentum-space Γ offset as v2

## Reported performance

### Which checkpoint?

| Checkpoint | Training corpus | Val mean radial | **OOD mean radial** |
| --- | --- | --- | --- |
| **v2.1** | hard-mix | 4.36° (hard val) | **4.82°** |
| v2 | easy | **1.79°** | 5.41° |
| v2-es | easy (cached) | 1.89° | 5.36° |

**Use v2.1 by default** — best on the independent OOD benchmark (~0.6° better than v2).

Full numbers: [BENCHMARK_RESULTS.json](BENCHMARK_RESULTS.json).

### v2.1 — OOD benchmark (`benchmark_first_gamma_v1`, 1500 samples)

| Metric | v2.1 | v2 (previous) |
| --- | --- | --- |
| Mean radial | **4.82°** | 5.41° |
| Median | **3.96°** | 4.68° |
| p95 | **11.61°** | 13.38° |
| MSE | **18.28 deg²** | 23.66 deg² |

OOD set includes oblique lattices, two-band spectra, Lorentzian lineshapes, varied background,
±10° Γ, small hν steps, slit vignetting, dead stripes. Training hard-mix covers material/instrument
axes but not yet slit shadows / dead stripes.

### v2.1 — in-distribution validation (hard-mix corpus, epoch 14)

| Metric | Value |
| --- | --- |
| Val MSE | 15.69 deg² |
| Mean radial | 4.36° |

Not comparable to v2's 1.79° — harder training distribution by design.

### v2 — easy-corpus validation (reference)

| Metric | Best (epoch 19) |
| --- | --- |
| Val MSE | 3.96 deg² |
| Mean radial | **1.79°** |

## Known limitations

- Synthetic only — no real beamline validation yet
- OOD ~4.8° mean — improved but not sub-degree on hard cases
- Detector artifacts (slit shadow, dead stripes) in benchmark but not in training generator yet
- No confidence score

## Reproducibility

| Component | File |
| --- | --- |
| Hard-mix generator | `arpeseed_gen_direct.py --hard-mix` |
| Training | `arpeseed_train_eval.py --corpus hardmix_cached` |
| Cached corpus | `arpeseed_cache_normalized.py` |
| OOD benchmark | `arpeseed_benchmark.py` |
| Cross-eval (v1/v2) | `archive/first_gamma_v1_geometry_bug/cross_eval.json` |
