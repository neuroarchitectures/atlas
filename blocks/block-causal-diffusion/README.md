# Block-Causal Diffusion Generation

## Design Philosophy

Pure autoregression is slow; pure diffusion can't do unbounded sequences. Block Diffusion interpolates: generate *blocks* autoregressively (conditioned on cached previous blocks) while denoising tokens *within* a block in parallel via discrete diffusion. The block-causal attention mask — causal across blocks, bidirectional within — enables exact KV caching of finished blocks.

## Functionality

- Denoiser outputs per block: logits + K_b, V_b for cache; attention mask = block-causal.
- Two-pass training (KV-cache pass + denoising pass); block size L' is the AR/parallelism dial. CLDLM applies the same scheme over latent blocks of a text VAE with a flow-matching prior.

## Used By

| Model | Role |
|-------|------|
| Block Diffusion | Interpolates AR ↔ diffusion for discrete sequence generation |
| CLDLM | Block-causal DiT prior over latent text blocks + flow matching |

## Features

- **KV-cache compatible diffusion** — the property pure diffusion LMs lack.
- **Tunable sequence parallelism** — larger blocks, more parallelism.

## Evolution

- **Predecessor**: Diffusion-LM, MaskGiT-style parallel decoding; AR transformers.
- **Related**: megabyte's global-local-patch-decoder — hierarchical AR alternative to intra-block diffusion.
