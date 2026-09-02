# ARPESeed first-gamma — GitHub Release

## ⚠️ Synthetic training only

**All checkpoints in this release were trained exclusively on simulated ARPES isoenergy stacks** (Poisson-noisy tight-binding-style physics). They have **not** been validated on real beamline data.

- Use for research, coarse synthetic benchmarking, or as a **starting point for fine-tuning** on your local scans.
- Do **not** claim beamline-ready autonomous alignment without local validation.
- See [MODEL_HISTORY.md](https://github.com/fawkesdx/ARPESeed/blob/main/tasks/first_gamma/MODEL_HISTORY.md) for the full version lineage.

---

## Recommended download

| File | Artifact ID | When to use |
|------|-------------|-------------|
| **`arpeseed_first_gamma_v2_1.pth`** | `arpeseed/first-gamma-v2.1` | Default — best OOD benchmark (4.82° mean radial) |
| `arpeseed_first_gamma_v2.pth` | `arpeseed/first-gamma-v2` | Easy-distribution val baseline (1.79°) |
| `arpeseed_first_gamma_v2_es.pth` | `arpeseed/first-gamma-v2-es` | Alternate training run |

## Inference

```bash
pip install -r requirements-arpeseed.txt
python arpeseed_predict.py scan.npy --weights arpeseed_first_gamma_v2_1.pth
```

Input: `float32` numpy `(3, 240, 300)` — three isoenergy channels.

## Metrics summary

| Version | Val mean radial | OOD mean radial |
|---------|-----------------|-----------------|
| v2 | 1.79° (easy) | 5.41° |
| v2-es | 1.89° | 5.36° |
| **v2.1** | 4.36° (hard) | **4.82°** |

OOD benchmark: 1500 held-out synthetic samples (oblique lattices, two-band, Lorentzian, detector artifacts, etc.).

## Citation

Sandy Adhitia Ekahana, Lawrence Berkeley National Laboratory.  
Repository: https://github.com/fawkesdx/ARPESeed  
Contact: sekahana@lbl.gov
