# Model Card — `arpeseed/first-gamma-v2` (current)

> **Status:** trained and evaluated on synthetic data.
> **Supersedes:** `arpeseed/first-gamma-v1` (retired — see tombstone in `archive/first_gamma_v1_geometry_bug/`).

## Summary

| | |
| --- | --- |
| **Project** | ARPESeed — task-oriented mini models for ARPES, distributed as weights |
| **Task** | Locate the first Brillouin-zone Γ point in noisy, ultra-fast ARPES isoenergy maps |
| **Type** | Supervised 2D coordinate regression |
| **Architecture** | ResNet18 (ImageNet init), `fc` → `Linear(512, 2)` |
| **Parameters** | 11.2 M — ~44.8 MB |
| **Checkpoint** | `arpeseed_first_gamma_v2.pth` (not in git; train or request weights) |
| **Trained** | 2026-09-01, 2× Tesla V100, DDP, AMP, 40 epochs (best epoch 19) |
| **Generator** | `arpeseed_gen_direct.py` — correct momentum-space Γ offset |
| **License** | TBD |
| **Contact** | Sandy Adhitia Ekahana — sekahana@lbl.gov |

## Intended use

Coarse live alignment from a 1-second test scan (three photon energies). Not a substitute for
a full Fermi-surface map or final momentum calibration. Always sanity-check predictions;
model returns a coordinate with no uncertainty estimate.

## Inputs and outputs

`float32` `(3, 240, 300)`, φ ∈ ±15°, θ ∈ ±12°, per-channel z-score normalization.
Output `(x_gamma, y_gamma)` in degrees (φ, θ).

```python
import numpy as np, torch
from torchvision.models import resnet18

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("arpeseed_first_gamma_v2.pth", map_location="cpu"))
model.eval()

x = np.load("scan.npy").astype(np.float32)
for c in range(3):
    x[c] = (x[c] - x[c].mean()) / (x[c].std() or 1.0)

with torch.no_grad():
    phi_gamma, theta_gamma = model(torch.from_numpy(x)[None]).numpy()[0]
```

Or use `arpeseed_predict.py`.

## Training data

50,001 synthetic samples from `arpeseed_gen_direct.py` — correct angle-to-momentum mapping.
Poisson noise: peak 1.0 count, background 5.0.

## Reported performance

**In-distribution validation** (seeded 80/20 split, `SPLIT_SEED=12345`):

| Metric | Best (epoch 19) |
| --- | --- |
| Val MSE | 3.96 deg² |
| Mean radial error | **1.79°** |
| Median radial error | 0.83° |
| p95 radial error | 6.56° |

**Cross-corpus** (`archive/first_gamma_v1_geometry_bug/cross_eval.json`):

| Corpus | Mean radial |
| --- | --- |
| Corrected training corpus (v2) | **1.79°** |
| Retired buggy corpus (v1) | 1.86° |

**OOD benchmark** — see [BENCHMARK_RESULTS.json](BENCHMARK_RESULTS.json):

| Split | Mean radial | Median | p95 |
| --- | --- | --- | --- |
| In-distribution val | **1.79°** | 0.83° | 6.56° |
| OOD benchmark | **5.41°** | 4.68° | 13.38° |

Train MSE at epoch 40: **0.73** vs val **4.08** — overfitting after epoch ~19.

### Comparison to v1 (why the headline changed)

| Model | Corpus | Mean error |
| --- | --- | --- |
| v1 | buggy synthetic | 0.26° (misleading) |
| v1 | corrected synthetic | 2.50° |
| **v2** | **corrected synthetic** | **1.79°** |

## Known limitations

- Synthetic only — no real beamline validation yet
- Overfitting gap (train 0.73 vs val 4.0)
- OOD benchmark ~5.4° mean — oblique lattices, detector artifacts not in training set
- No confidence score

## Reproducibility

| Component | File |
| --- | --- |
| Corpus generator | `arpeseed_gen_direct.py` |
| Training | `arpeseed_train_eval.py` |
| Cross-eval | `archive/first_gamma_v1_geometry_bug/cross_eval.json` |
