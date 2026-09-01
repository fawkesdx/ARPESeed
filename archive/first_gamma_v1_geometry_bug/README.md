# Tombstone — first-gamma corpus v1 (geometry bug)

**Status:** RETIRED — do not train on this data  
**Created:** August 2025  
**Retired:** 1 September 2026  
**Project:** ARPESeed (`arpeseed/first-gamma-v1`)

---

## What this was

50,148 synthetic ARPES isoenergy maps for training a ResNet18 to locate the first Brillouin-zone
Γ point from three noisy channels (three photon energies). Ultra-fast scan simulation: peak band
signal 1.0 Poisson count on a 5.0 count background.

A model was trained on this corpus and reported validation MSE 0.265 (~±0.51° radial error) on
50k samples using a Tesla V100. That number looked excellent. It was misleading.

---

## The mistake — do not repeat

The generator `generate_full_corpus_fast.py` mapped sample tilt to momentum **incorrectly**:

```python
# WRONG (v1 generator)
X_rad = np.radians(X_deg - x_gamma)
Y_rad = np.radians(Y_deg - y_gamma)
kx    = 0.512 * np.sqrt(E_kin) * np.sin(X_rad)
ky    = 0.512 * np.sqrt(E_kin) * np.sin(Y_rad)
```

Two separate errors:

1. **Angle-space offset before the sine** — uses `sin(φ − φ₀)` instead of offsetting in
   momentum. The band pattern becomes a **rigid translation** in angle space that scales
   isotropically toward the label. The network only has to find a translate-and-scale centre,
   not the warping a real tilted sample produces.

2. **Missing `cos θ` in `kx`** — the horizontal slit projection requires
   `kx ∝ sin φ · cos θ`. Even at Γ = (0°, 0°) this shifts `kx` by ~2–3% at the detector
   corner (φ = 15°, θ = 12°).

**Correct mapping** (used from v2 onward in `arpeseed_gen_direct.py`):

```python
kx = A * (np.sin(PH) * np.cos(TH) - np.sin(ph0) * np.cos(th0))
ky = A * (np.sin(TH) - np.sin(th0))
```

---

## Consequences

| What happened | Why it matters |
| --- | --- |
| ±0.51° val error on 50k noisy samples | Measured on the **same buggy generator** as training — in-distribution, not generalization |
| Pattern rigidly follows label | Task was easier than real beamline alignment |
| Full 41 GB corpus deleted after archival | Cannot recover all 50k files — generator used **unseeded** `random`; bit-identical regeneration impossible |
| Model weights kept | Historical artifact + baseline for cross-corpus comparison |

---

## What we kept (this folder)

| File | Purpose |
| --- | --- |
| `README.md` | This tombstone |
| `MANIFEST.json` | Machine-readable metadata, sample list, cross-eval results |
| `generate_full_corpus_fast.py` | The buggy generator — read before writing any new ARPES synthetic code |
| `labels_sample.csv` | Labels for forensic `.npy` files kept here |
| `npy/` | 40 example samples (~40 MB) spanning three photon-energy regimes |
| `training_log_final_50k.csv` | Epoch-by-epoch log from the V100 training run |
| `loss_curve_final_50k.png` | Loss curve figure |
| `best_gamma_model.pth` | Checkpoint trained on the buggy corpus (44.8 MB) |
| `cross_eval.json` | 2×2 model×corpus evaluation run before deletion (if present) |

---

## Replacement

| | v1 (this tombstone) | v2 (current) |
| --- | --- | --- |
| Generator | `generate_full_corpus_fast.py` | `arpeseed_gen_direct.py` |
| Corpus path (Einstein) | ~~`.../corpus_finding_first_gamma/`~~ deleted | `/mnt/data/sandy/tensorspec_heavy/corpus_first_gamma_direct/` |
| Geometry | `sin(φ−φ₀)`, no `cos θ` | Momentum-space offset, full slit projection |
| Model artifact | `arpeseed/first-gamma-v1` (buggy, kept for history) | `arpeseed/first-gamma-v2` (train on v2 corpus) |

---

## Checklist before writing a new ARPES synthetic generator

- [ ] Γ offset applied in **momentum**, not by subtracting angles before `sin`
- [ ] Horizontal slit: `kx` includes `cos θ`
- [ ] `kz` from emission sphere + inner potential if using 3-channel hv stacks
- [ ] Label convention documented: `x_gamma = φ`, `y_gamma = θ`, degrees, `origin='lower'`
- [ ] Fixed random seed OR save generator config per sample if corpus must be reproducible
- [ ] Visual verify: green cross on clean physics row; pattern **contracts** toward cross as hν increases
- [ ] Held-out benchmark from a **different** generator before publishing accuracy numbers

---

## References in this repo

- Full audit: `PROJECT_STATUS.md` §1 (geometry bug table)
- Model card (Known Issues): `MODEL_CARD.md`
- Ideas log snapshot: `IDEAS_LOG.md`
