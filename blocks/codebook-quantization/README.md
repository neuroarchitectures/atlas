# Codebook Quantization of Latents

## Design Philosophy

Self-supervised speech needs *discrete* prediction targets, but hand-labels are unavailable. Quantize the continuous latent against a learned codebook — Gumbel-softmax product quantization over two codebook groups — so masked prediction (contrastive here) has stable discrete targets that co-train with the encoder.

## Functionality

- Latent z → nearest entries from two codebooks (320 entries, 8 groups product-quantized); straight-through gradient path.
- Quantized codes serve as masked-prediction targets for the contrastive objective.

## Used By

| Model | Role |
|-------|------|
| Wav2Vec2 base | Codebook targets for masked contrastive prediction over conv-stem latents |

## Features

- **Learned discrete units** — no k-means pre-clustering needed (vs. HuBERT's offline units).
- **Straight-through Gumbel** — differentiable discrete selection during training.

## Evolution

- **Predecessor**: VQ-VAE quantization; HuBERT's masked-cluster-id offline pipeline.
- **Related**: residual-vector-quantization (EnCodec) — hierarchical codebooks for reconstruction; here codebooks are targets, not reconstruction.
