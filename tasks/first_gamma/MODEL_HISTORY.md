# first-gamma model history

Citable lineage for papers and Zenodo. **All checkpoints are trained on synthetic data only** — no real beamline scans in training. Fine-tune on local data before production alignment.

## How to cite

| Use case | Artifact ID | Checkpoint |
|----------|-------------|------------|
| **Recommended (default)** | `arpeseed/first-gamma-v2.2` | `arpeseed_first_gamma_v2_2.pth` |
| Previous OOD-focused | `arpeseed/first-gamma-v2.1` | `arpeseed_first_gamma_v2_1.pth` |
| Original easy baseline | `arpeseed/first-gamma-v2` | `arpeseed_first_gamma_v2.pth` |
| Training I/O experiment | `arpeseed/first-gamma-v2-es` | `arpeseed_first_gamma_v2_es.pth` |
| Retired (do not use) | `arpeseed/first-gamma-v1` | tombstone only |

GitHub releases: https://github.com/fawkesdx/ARPESeed/releases

---

## Version table

| Version | Artifact ID | Date | Training data | Best epoch | Easy val | OOD mean | Status |
|---------|-------------|------|---------------|------------|----------|----------|--------|
| **v1** | `first-gamma-v1` | 2026-08 | Buggy generator | — | 0.26°† | — | **Retired** |
| **v2** | `first-gamma-v2` | 2026-09-01 | 50k easy | 19 | 1.79° | 5.41° | Available |
| **v2-es** | `first-gamma-v2-es` | 2026-09-01 | Easy, cached | 17 | 1.89° | 5.36° | Available |
| **v2.1** | `first-gamma-v2.1` | 2026-09-01 | 50k hard-mix | 14 | — | 4.82° | Available |
| **v2.2** | `first-gamma-v2.2` | 2026-09-02 | Mixed easy + hard-mix‡ | 13 | **1.66°** | **4.42°** | **Recommended** |

† Misleading — geometry bug. ‡ Hard-mix includes detector artifacts (slit shadow, dead stripes).

---

## Per-version notes

### v2.2 — current default

Mixed 100k cached corpus (50k easy + 50k hard-mix with detector artifacts). Beats v2 on easy val **and** beats v2.1 on OOD. Use this unless reproducing an older paper number.

### v2.1

Hard-mix only — good OOD but poor easy-val comparability. Superseded by v2.2.

### v2 / v2-es

First correct-geometry models on easy corpus. Kept for ablation and historical comparison.

### v1 — retired

[tombstone](../../archive/first_gamma_v1_geometry_bug/)

---

## Synthetic-data disclaimer (all versions)

> These weights were trained exclusively on Poisson-noisy synthetic isoenergy stacks. **Not validated on real beamline data.** Fine-tune on ≥200 labeled local scans before autonomous alignment.

---

## Zenodo bundle checklist

Per version deposit:

1. `arpeseed_first_gamma_v2_X.pth`
2. `MODEL_CARD.md` (this version section)
3. `MODEL_HISTORY.md` (full lineage)
4. `BENCHMARK_RESULTS.json`
