# Exponential Moving Average of Weights (EMA / SWA)

## Design Philosophy

The parameters that a network visits in the last epochs of SGD oscillate around a better solution than any single iterate. Averaging them — either as an **exponential moving average** kept alongside training, or as a uniform average of a cyclic/Late-phase trajectory (**SWA**) — gives flatter minima and better generalization essentially for free: no extra forward passes, just extra memory.

## Functionality

`θ_ema ← decay · θ_ema + (1 − decay) · θ` after each optimizer step (or every `k` steps).

- `decay` typically `0.999–0.9999`; some implementations use `(1 + step) / (10 + step)` style ramps so early noisy iterates contribute little.
- The EMA copy is **not** trained — it is evaluated/exported; optionally copied back into the model at the end.
- **SWA**: average snapshots taken with a constant or cyclic LR; **LAWA/Model Soup**: average the last `k` checkpoints.
- Cost: one extra copy of the parameters (memory) plus one cheap update per step.

## Used By

| Model | Role |
|-------|------|
| Diffusion models (DDPM, EDM, DiT) | EMA of denoiser weights — standard for sample quality |
| GAN training | Stabilizes generator outputs |
| BYOL / MoCo / DINO (self-supervised) | The momentum encoder *is* an EMA of the online encoder |
| YOLO / detection finetunes, segmentation models | EMA weights for final checkpoint quality |

## Features

- **Better generalization** at zero training-time cost; almost always a small win.
- Smooths out the noise of the final high-LR phase; pairs naturally with cosine schedules.
- **Memory cost**: a full extra parameter copy (often kept in fp32).
- Needs care with BatchNorm: EMA weights with stale batch statistics can be worse — re-compute BN stats on the EMA model before export.

## Evolution

- **Predecessor**: Polyak averaging / iterate averaging in stochastic optimization.
- **Itself**: EMA of weights (common practice; formalized in BYOL's momentum encoder); SWA (Izmailov et al., 2018).
- **Successors**: LAWA (last-k weighting), Model Soups (average independently trained models), EMA of optimizer states, and schedule-free optimizers that build averaging into the update rule.
