# VideoMAE

## Overview

Masked autoencoding had transformed image pre-training, but video looked like a bad fit: video is **temporally redundant**, so a masked cube is easily recovered from neighbouring frames, and the correspondence between frames risks **information leakage** — the model finds shortcut features that do not generalize. VideoMAE's fix is a masking strategy rather than a new architecture: mask tubes, at an extremely high ratio.

- **Year:** 2022
- **Authors:** Tong et al. (Nanjing University, Shanghai AI Laboratory, Shenzhen Institute of Advanced Technology CAS, University of Science and Technology of China)
- **Source:** arXiv:2203.12602 — *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*
- **Category:** DL/Video Representation

## Key Characteristics

- **Extremely high masking ratio** — because frames are redundant, most cubes are dropped from the downsampled clips; this improves pre-training performance *and* greatly reduces compute via the asymmetric encoder-decoder.
- **Tube masking** — masking whole spatiotemporal cubes rather than random patches, which relieves the leakage risk for cubes with no or negligible motion during reconstruction.
- **Plain ViT backbones** — claimed as the first masked video pre-training framework using simple vanilla ViT backbones, with no video-specific backbone design.
- **Data-efficient** — trains successfully on relatively small video datasets (Something-Something, UCF101, HMDB51), significantly outperforming the previous state of the art without extra data.
- **Motivated by two named problems** — temporal redundancy and temporal correlation, each addressed by one of the two masking choices.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Tong_et_al._2022_2203.12602.md`](references/papers/Tong_et_al._2022_2203.12602.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (image MAE → VideoMAE; siblings: TimeSformer, ViViT, SlowFast, Uniformer; contrast: I-JEPA / V-JEPA, which predict representations rather than pixels).
