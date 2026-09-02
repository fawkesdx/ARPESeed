# ARPESeed first-gamma v2.2

## ⚠️ Synthetic training only

**Trained exclusively on simulated ARPES stacks** — not real beamline data. Fine-tune locally before production use.

See [MODEL_HISTORY.md](https://github.com/fawkesdx/ARPESeed/blob/main/tasks/first_gamma/MODEL_HISTORY.md) for full version lineage (v1 → v2 → v2.1 → **v2.2**).

---

## Download

**`arpeseed_first_gamma_v2_2.pth`** — artifact `arpeseed/first-gamma-v2.2`

Older releases: [v2.1](https://github.com/fawkesdx/ARPESeed/releases/tag/first-gamma-v2.1)

## Metrics

| | Easy val | OOD benchmark |
|---|---|---|
| **v2.2** | **1.66°** | **4.42°** |
| v2.1 | — | 4.82° |
| v2 | 1.79° | 5.41° |

Training: mixed 100k synthetic samples (easy + hard-mix with detector artifacts).

## Inference

```bash
python arpeseed_predict.py scan.npy --weights arpeseed_first_gamma_v2_2.pth
```

## Citation

Sandy Adhitia Ekahana, LBNL — sekahana@lbl.gov  
https://github.com/fawkesdx/ARPESeed
