# Depth Anything V2

## Overview

Depth Anything V2 is a **monocular depth estimation** model whose paper is deliberately about *findings, not tricks*: relative depth quality is limited less by architecture than by **label quality and data scaling**. Replacing V1's noisy pseudo-labels with high-precision **synthetic** images, and using a powerful **DINOv2** encoder, yields a model that is simultaneously more robust, more detailed, faster, and smaller.

- **Year:** 2024
- **Authors:** Yang et al. (HKU / TikTok)
- **Source:** arXiv:2406.09414 — *Depth Anything V2*
- **Category:** DL/Depth

## Key Characteristics

- **DINOv2-based encoder** — a strong pretrained representation is the single biggest lever for fine-grained depth detail.
- **Synthetic training data with precise labels** beats large real images with pseudo-labels; the paper also shows how to keep the model robust by using real images only for auxiliary constraints.
- **Three key findings** motivate the design: replace labelled real images with synthetic ones, enlarge the *model* capacity rather than increasing noisy data, and use large-scale pseudo-labelled real images mainly as a bridge.
- **Latency/accuracy family**: the Small variant is ~25M parameters at ~60 ms (V100), the Large variant ~335M at ~213 ms — i.e. a usable latency ladder for on-device and server use.
- **DA-2K benchmark**: a densely annotated, hard-case evaluation set introduced by the paper to measure fine-grained and robust relative depth.
- Output is **relative (affine-invariant) depth**, not metric depth; metric variants exist downstream.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Yang_et_al._2024_2406.09414.md`](references/papers/Yang_et_al._2024_2406.09414.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (MiDaS / DPT → Depth Anything V1 → V2; competitors: Marigold, DepthFM; siblings: MoGe, UniDepth, Metric3D).
