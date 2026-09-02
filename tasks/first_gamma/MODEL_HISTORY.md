# first-gamma model history

Citable lineage for papers and Zenodo. **All checkpoints are trained on synthetic data only** — no real beamline scans in training. Fine-tune on local data before production alignment.

## How to cite

| Use case | Artifact ID | Checkpoint |
|----------|-------------|------------|
| **Recommended (varied / OOD conditions)** | `arpeseed/first-gamma-v2.1` | `arpeseed_first_gamma_v2_1.pth` |
| Easy-distribution val baseline | `arpeseed/first-gamma-v2` | `arpeseed_first_gamma_v2.pth` |
| Fast-training variant | `arpeseed/first-gamma-v2-es` | `arpeseed_first_gamma_v2_es.pth` |
| Retired (do not use) | `arpeseed/first-gamma-v1` | tombstone only |

GitHub releases: https://github.com/fawkesdx/ARPESeed/releases

---

## Version table

| Version | Artifact ID | Date | Training data | Best epoch | Val mean radial | OOD mean radial | Status |
|---------|-------------|------|---------------|------------|-----------------|-----------------|--------|
| **v1** | `first-gamma-v1` | 2026-08 | Buggy generator (`sin(φ−φ₀)` shortcut) | — | 0.26° (misleading) | — | **Retired** — [tombstone](../../archive/first_gamma_v1_geometry_bug/) |
| **v2** | `first-gamma-v2` | 2026-09-01 | 50k easy corpus, correct Γ geometry | 19 | **1.79°** (easy val) | 5.41° | Available |
| **v2-es** | `first-gamma-v2-es` | 2026-09-01 | Easy corpus, cached + early stop | 17 | 1.89° | 5.36° | Available |
| **v2.1** | `first-gamma-v2.1` | 2026-09-01 | 50k `--hard-mix` corpus | 14 | 4.36° (hard val) | **4.82°** | **Recommended** |
| **v2.2** | `first-gamma-v2.2` | TBD | Mixed easy + hard-mix w/ detector artifacts | TBD | TBD | TBD | In progress |

---

## Training data definitions

| Corpus | Generator | Contents |
|--------|-----------|----------|
| Easy | `arpeseed_gen_direct.py` | Square / hex / rect, Gaussian bands, bg=5, ±8° Γ |
| Hard-mix | `arpeseed_gen_direct.py --hard-mix` | + oblique, two-band, Lorentzian, bg 2/5/10, ±10° Γ, small Δhν |
| Hard-mix + artifacts (v2.2+) | `--hard-mix` (updated) | + slit vignetting (~40%), dead stripes (0–2) |
| Mixed (v2.2+) | Both cached corpora | 50k easy + 50k hard-mix in one training run |
| OOD benchmark | `arpeseed_benchmark.py` | Held-out test set — **never train on this** |

---

## Per-version notes

### v1 — retired

Geometry bug made Γ a rigid angular translation. Cross-eval on corrected corpus: **2.50°**. Archived for reproducibility only.

### v2

First model with correct momentum-space Γ offset. Best easy-corpus validation. Use when scans match the easy synthetic distribution.

### v2-es

Same data as v2; pre-z-scored cache + early stopping. Marginal OOD gain; mainly a training-speed experiment.

### v2.1

Trained entirely on hard-mix synthetic data. **Best OOD benchmark** among released models. Val on hard-mix is ~4.4° — not comparable to v2's 1.79° easy val.

### v2.2 — planned

Goals: (1) add detector artifacts to hard-mix generator, (2) train on **mixed** easy + hard-mix to recover easy-val accuracy while keeping OOD gains.

---

## Synthetic-data disclaimer (all versions)

> These weights were trained exclusively on Poisson-noisy synthetic isoenergy stacks from tight-binding-style simulators. They have **not** been validated on real ARPES beamline data. Reported errors are on synthetic validation or the synthetic OOD benchmark. Expect additional domain shift on experimental scans; fine-tuning on ≥200 labeled local scans is recommended before autonomous alignment.

---

## File naming convention

```
arpeseed_first_gamma_v2.pth      → arpeseed/first-gamma-v2
arpeseed_first_gamma_v2_es.pth   → arpeseed/first-gamma-v2-es
arpeseed_first_gamma_v2_1.pth    → arpeseed/first-gamma-v2.1
arpeseed_first_gamma_v2_2.pth    → arpeseed/first-gamma-v2.2 (future)
```

When uploading to Zenodo, bundle: checkpoint + this file + `MODEL_CARD.md` + `BENCHMARK_RESULTS.json`.
