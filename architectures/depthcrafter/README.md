# DepthCrafter

## Overview

DepthCrafter turns a **pretrained video diffusion model** into a video depth estimator. Rather than training a depth network from scratch, it repurposes Stable Video Diffusion's U-Net as a strong generative prior, adds **temporal layers** for long-range consistency, and handles arbitrarily long videos by **segment-wise estimation with seamless stitching**.

- **Year:** 2024
- **Authors:** Hu et al.
- **Source:** arXiv:2409.02095 — *DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos*
- **Category:** DL/Depth Estimation

## Key Characteristics

- **Generative prior** — inherits open-world generalization from a video diffusion model, so it works on internet video instead of only on the training domain.
- **Temporal layers** — long-range temporal attention in the U-Net is what produces temporally consistent depth instead of per-frame flicker.
- **Long sequences** — segment-wise processing (roughly 110 frames per segment as reported) with stitching extends a fixed-length model to arbitrarily long videos.
- **Fine-grained detail** — diffusion-based estimation keeps thin structures and sharp boundaries better than regression baselines.
- **Zero-shot transfer** — no metric scale; output is affine-invariant depth, which is the standard limitation of monocular video depth.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Hu_et_al._2024_2409.02095.md`](references/papers/Hu_et_al._2024_2409.02095.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (Midas / DepthAnything → Video Depth Anything → DepthCrafter; siblings: Marigold, DepthFM, Lotus (diffusion-based depth), Video Depth Anything; downstream: video effects, 3D/4D reconstruction).
