# ARPESeed

Task-oriented mini models for ARPES — synthetic-trained weights for beamline micro-tasks.

Part of the [TensorSpec](https://github.com/fawkesdx/TensorSpec) ecosystem. Training data and checkpoints are **not** in this repo (too large); only code, model cards, and benchmark numbers.

## Tasks

| Task | Artifact | Status |
|------|----------|--------|
| [First Γ finder](tasks/first_gamma/) | `arpeseed/first-gamma-v2` (default), `v2-es` (alternate) | trained, evaluated |
| [Denoising](tasks/denoising/) | `arpeseed/denoise-v1` | scaffold |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-arpeseed.txt

# inference (weights downloaded separately)
python arpeseed_predict.py scan.npy --weights path/to/arpeseed_first_gamma_v2.pth
```

## Core scripts

| Script | Purpose |
|--------|---------|
| `arpeseed_gen_direct.py` | Synthetic corpus generator (correct Γ geometry) |
| `arpeseed_train_eval.py` | Train / eval ResNet18 Γ regressor |
| `arpeseed_predict.py` | Local inference CLI |
| `arpeseed_benchmark.py` | OOD benchmark generation + scoring |
| `arpeseed_cache_normalized.py` | Pre-z-score corpus for faster training |

Set `ARPESEED_DATA_ROOT` to your corpus parent directory (default in scripts is a placeholder).

## Archive

`archive/first_gamma_v1_geometry_bug/` — tombstone for the retired v1 model (geometry bug in old generator).

## Citation

Sandy Adhitia Ekahana, LBNL. Contact: sekahana@lbl.gov
