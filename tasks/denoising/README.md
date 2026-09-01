# ARPESeed task 2 — denoising (synthetic clean/noisy pairs)

Ultra-fast 1-second ARPES isoenergy stacks: predict clean band from noisy input.

| | |
| --- | --- |
| Input | `(3, 240, 300)` noisy Poisson counts |
| Target | `(3, 240, 300)` clean physics (pre-noise) |
| Loss | MSE per pixel, all channels |

Data lands in gitignored `corpus_denoising/dataset/` (see `.gitignore`).

```bash
python3 tasks/denoising/generate.py --n 2000 --workers 16
python3 tasks/denoising/train.py --epochs 20
```

Scaffold only — no released model card yet.
