# Model Card — `arpeseed/first-gamma-v2.2` (recommended)

> **Status:** trained and evaluated on synthetic data only.
> **Supersedes:** v2, v2-es, v2.1.
> **Retired:** `arpeseed/first-gamma-v1` — tombstone in `archive/first_gamma_v1_geometry_bug/`.

## Summary

| | |
| --- | --- |
| **Project** | ARPESeed — task-oriented mini models for ARPES, distributed as weights |
| **Task** | Locate the first Brillouin-zone Γ point in noisy, ultra-fast ARPES isoenergy maps |
| **Type** | Supervised 2D coordinate regression |
| **Architecture** | ResNet18 (ImageNet init), `fc` → `Linear(512, 2)` |
| **Parameters** | 11.2 M — ~44.8 MB |
| **Recommended checkpoint** | **`arpeseed_first_gamma_v2_2.pth`** |
| **Trained** | 2026-09-02, 2× Tesla V100, DDP, AMP, early stop (epoch 13) |
| **Training data** | Mixed easy (50k) + hard-mix w/ detector artifacts (50k), cached |
| **Generator** | `arpeseed_gen_direct.py` + `--hard-mix` |
| **License** | TBD |
| **Contact** | Sandy Adhitia Ekahana — sekahana@lbl.gov |

## Intended use

Coarse live alignment from a 1-second test scan (three photon energies). **v2.2 is the default**
checkpoint — best reported easy-val and OOD numbers among released models.

**Synthetic training only.** Not validated on real beamline data. Fine-tune on local scans before
production use. No uncertainty estimate.

## Inputs and outputs

Training grid: `float32` `(3, 240, 300)`, φ ∈ ±15°, θ ∈ ±12°, per-channel z-score.
Output `(x_gamma, y_gamma)` in degrees (φ, θ).

**Any `(3, H, W)` accepted** — `arpeseed_predict.py` bilinear-resizes to `(240, 300)`
before normalization (no shape error). Prefer that path over hand-rolled loading.

**Single isoenergy map** `(H, W)` or `(1, H, W)` also accepted: image is duplicated into
three identical channels with a warning. Prefer three different photon energies — the
model uses hν-dependent pattern scaling; single-map mode is a degraded fallback.

```python
from arpeseed_predict import load_model, predict
import numpy as np

model = load_model("arpeseed_first_gamma_v2_2.pth")
phi, theta = predict(np.load("scan.npy"), model)  # any (3, H, W)
```

Or: `python arpeseed_predict.py scan.npy --weights arpeseed_first_gamma_v2_2.pth`

## Training data

100k synthetic samples total — mixed training run:

| Corpus | N | Contents |
| --- | --- | --- |
| Easy | 50,001 | Square / hex / rect, Gaussian, bg=5 |
| Hard-mix | 50,001 | Oblique, two-band, Lorentzian, bg 2/5/10, ±10° Γ, slit shadow, dead stripes |

## Reported performance

Full lineage: [MODEL_HISTORY.md](MODEL_HISTORY.md) · [BENCHMARK_RESULTS.json](BENCHMARK_RESULTS.json)

### v2.2 vs previous releases

| Checkpoint | Easy val mean | OOD mean | OOD median |
| --- | --- | --- | --- |
| v2 | 1.79° | 5.41° | 4.68° |
| v2.1 | 4.36° (hard only) | 4.82° | 3.96° |
| **v2.2** | **1.66°** | **4.42°** | **3.21°** |

### v2.2 — easy-corpus validation (`corpus_first_gamma_direct`, n=10k)

| Metric | Value |
| --- | --- |
| Mean radial | **1.66°** |
| Median | 0.95° |
| p95 | 5.55° |
| MSE | 3.03 deg² |

### v2.2 — OOD benchmark (`benchmark_first_gamma_v1`, n=1500)

| Metric | Value |
| --- | --- |
| Mean radial | **4.42°** |
| Median | 3.21° |
| p95 | 11.28° |
| MSE | 16.61 deg² |

### v2.2 — mixed-corpus validation (training split, epoch 13)

| Metric | Value |
| --- | --- |
| Mean radial | 2.87° |
| MSE | 8.81 deg² |

## Known limitations

- Synthetic only — no real beamline validation
- OOD ~4.4° mean — improved but not sub-degree on hardest cases
- No confidence score

## Reproducibility

| Component | File |
| --- | --- |
| Generator | `arpeseed_gen_direct.py --hard-mix` |
| Training | `arpeseed_train_eval.py --corpus mixed_cached` |
| Cached corpora | `arpeseed_cache_normalized.py` |
| OOD benchmark | `arpeseed_benchmark.py` |
