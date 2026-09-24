# UniFormer

## Overview

Video networks face two structural problems at once. **3D convolution** captures local patterns efficiently but **struggles to capture long-range dependency** because of its limited receptive field. **Self-attention** captures long-range dependency but is **inefficient at encoding low-level features** — Video Swin applies self-attention in a local 3D window, and that inefficiency hinders the model's potential. UniFormer's answer is to use **3D convolution for local features and self-attention for global**, both inside a concise transformer format.

- **Year:** 2022
- **Authors:** Li et al. (University of Science and Technology of China, MSRA, Beijing Institute of Technology, University of Sydney)
- **Source:** arXiv:2201.04676 — *UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning*
- **Category:** DL/Video Recognition

## Key Characteristics

- **Unified transformer format** — a basic transformer format specially designed for efficient and effective spatiotemporal representation learning; the point is that convolution and attention share one format.
- **Three key modules per block** — **Dynamic Position Embedding (DPE)**, **Multi-Head Relation Aggregator (MHRA)**, and **Feed-Forward Network (FFN)**.
- **Local MHRA = 3D convolution** — encodes local features in a concise transformer format, instead of applying self-attention in a local 3D window as Video Swin does.
- **Global MHRA = self-attention** — for long-range dependency, kept where it is worth the cost.
- **Hierarchical stack** — blocks stacked hierarchically following prior hierarchical designs.
- **Targets the two named problems** — spatiotemporal **redundancy** and **dependency**; the paper's stated motivation is overcoming both.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Li_et_al._2022_2201.04676.md`](references/papers/Li_et_al._2022_2201.04676.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (2D/3D CNNs, ViT video variants, MViT, Video Swin → UniFormer; siblings: VideoMAE, TimeSformer, ViViT, SlowFast; contrast: Video Swin, which applies self-attention even in local 3D windows).
