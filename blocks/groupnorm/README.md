# GroupNorm

## Design Philosophy

BatchNorm is the wrong normalization when the batch is small: its statistics depend on other samples in the batch, and with detection/segmentation/video models the per-device batch may be 1–2 items. GroupNorm normalizes **within a single sample** — split channels into groups, normalize over (group channels × spatial) — so its behavior is identical at batch size 1 and 64, and identical in train and eval.

## Functionality

```
reshape: (N, C, H, W) -> (N, G, C//G * H, W)
normalize per (n, g): mean/var over the group's channel*spatial elements
y = gamma * (x - mean) / sqrt(var + eps) + beta
```

- **G** is the only hyperparameter: G=1 ≈ LayerNorm over C,H,W; G=C ≈ InstanceNorm; the usual default is **G=32** (and `min(G, C)` for small channel counts).
- Per-channel learnable **affine** (gamma/beta), same as BatchNorm.
- **No running statistics**, no batch coupling → nothing to synchronize (see `syncbatchnorm`) and nothing to freeze in eval.
- Deterministic per sample → stable with mixed precision and with gradient accumulation.
- Cost: one reduction over the group; no cross-device communication.

## Used By

| Architecture | How it is used |
|---|---|
| Mask R-CNN / FPN (Facebook "Group Normalization" paper) | replaces BN for detection; the canonical use case |
| Segmentation (DeepLabV3+, Mask2Former) | small batches with large images |
| Video models, 3D CNNs | temporal clips make batches tiny |
| Diffusion U-Nets (Stable Diffusion, DDPM) | GroupNorm is the default normalization |
| Transformers with conv stems | GN in the stem/upsampling paths, LN in attention blocks |

## Features

- **Batch-size independent**: the defining property; works at batch size 1.
- **No sync needed**: no all-reduce, unlike SyncBatchNorm.
- **Train/eval identical**: no running stats to update or freeze.
- **Slight accuracy gap**: on large-batch image classification, BN (or sync BN) can still be a bit better; the gap shrinks as tasks get heavier per sample.
- **Group size matters**: too small → noisy statistics like InstanceNorm; too large → approaches LayerNorm and loses per-channel structure.

## Evolution

- **BatchNorm (2015)** → **LayerNorm (2016)** (per-sample, for sequences) → **InstanceNorm (2016)** (style transfer) → **GroupNorm (2018)**: "a middle ground between LayerNorm and InstanceNorm".
- **Weight Standardization + GN (2019)**: closes most of the remaining gap with BN on small-batch training.
- **Modern**: GN is the default in diffusion models and in heavy-per-sample vision tasks; **RMSNorm/LayerNorm** dominate transformers.
- **Related**: **SyncBatchNorm** (keep BN but sync stats), **Adaptive LayerNorm (adaLN)** for conditioning, **Filter Response Normalization**.

## See Also

`batch-norm`, `layer-norm`, `syncbatchnorm`, `mixed-precision-amp`
