# StarNet

## Overview

Why do the "star operations" (element-wise multiplication) used in some recent efficient networks work so well? StarNet answers that question and then builds the simplest possible network on the answer: the star operation maps inputs into an **exceedingly high-dimensional, non-linear feature space without widening the network**, behaving like a polynomial kernel, so a compact model can get the benefits of huge implicit dimensionality.

- **Year:** 2024
- **Authors:** Ma et al. (Northeastern University, China; University of Sydney)
- **Source:** arXiv:2403.19967 — *Rewrite the Stars*
- **Category:** DL/Efficient CNN

## Key Characteristics

- **The star operation is a kernel** — pairwise multiplication of features across channels, analogous to **polynomial kernel functions**; a single star operation yields about (d/√2)² linearly independent dimensions.
- **Implicit dimensions instead of width** — traditional networks increase channel count; the star operation gains dimensionality within a **compact feature space**, and stacked layers increase implicit complexity exponentially.
- **Better suited to small networks** — the paper's inference is that this mechanism helps efficient, compact models more than large ones, and StarNet is built to test it.
- **Deliberately plain** — StarNet has no sophisticated designs and no fine-tuned hyper-parameters, which is the point: the gain comes from the operation, not from architecture search.
- **Beats carefully designed efficient models** — StarNet-S4 surpasses EdgeViT-XS by 0.9% top-1 on ImageNet-1K while running 3× faster on iPhone13/CPU and 2× faster on GPU.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Ma_et_al._2024_2403.19967.md`](references/papers/Ma_et_al._2024_2403.19967.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (MobileNetV3, EdgeViT, FasterNet → StarNet; siblings: MobileNetV4, RepViT, EfficientViT; contrast: width-increasing efficient networks).
