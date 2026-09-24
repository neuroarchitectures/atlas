# FoundationStereo

## Overview

Stereo networks were trained on ~40K synthetic pairs (Scene Flow) and fine-tuned per target domain — none worked as an off-the-shelf model, unlike foundation models elsewhere in vision. FoundationStereo is a **large stereo foundation model with strong zero-shot generalization and no per-domain fine-tuning**: 1M high-fidelity synthetic pairs with self-curation, plus monocular priors distilled in from DepthAnythingV2 through a side-tuning backbone.

- **Year:** 2025
- **Authors:** Wen et al. (NVIDIA, University of Washington)
- **Source:** arXiv:2501.09898 — *FoundationStereo: Zero-Shot Stereo Matching*
- **Category:** DL/Stereo Depth

## Key Characteristics

- **1M-pair synthetic dataset with self-curation** — a large-scale, high-diversity, photorealistic set, plus an automatic self-curation pipeline that prunes ambiguous samples inevitably produced by domain randomization.
- **Side-tuning feature backbone** — adapts internet-scale monocular priors from DepthAnythingV2 (trained on real images) into the stereo setup, closing part of the sim-to-real gap.
- **Attentive Hybrid Cost Volume (AHCF)** — two parts: **3D Axial-Planar Convolution (APC)**, which decouples standard 3D convolution into spatial- and disparity-oriented 3D convolutions to enlarge receptive fields; and a **Disparity Transformer (DT)** doing self-attention over the entire disparity space for global reasoning.
- **Zero-shot, off-the-shelf** — comparable to or better than prior works that were fine-tuned on the target domain; notably better on in-the-wild data.
- **Also improves initialization and refinement** — the AHCF features give both a better disparity initialization and stronger features for iterative refinement.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Wen_et_al._2025_2501.09898.md`](references/papers/Wen_et_al._2025_2501.09898.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (RAFT-Stereo / IGEV / CREStereo → FoundationStereo; siblings: Fast-FoundationStereo (real-time variant); contrast: fine-tuned-per-domain stereo networks).
