# V-JEPA 2

## Overview

V-JEPA 2 is a **self-supervised video model** trained by masked **latent prediction**: predict the representations of masked spatio-temporal regions from the visible ones, with a target encoder updated by EMA. Scaling this recipe to over a million hours of video yields representations that support understanding, prediction, and — with an **action-conditioned** variant — planning and robot control.

- **Year:** 2025
- **Authors:** Assran et al. (Meta FAIR)
- **Source:** arXiv:2506.09985 — *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*
- **Category:** DL/World-Model

## Key Characteristics

- **Predict in representation space, not pixel space**: the loss is on latent features from an EMA target encoder, which avoids modeling photometric detail and is far cheaper than generative video prediction.
- **Masked spatio-temporal modeling**: large regions of the video are masked and predicted by a lightweight predictor on the context encoder's output.
- **Scale**: trained on a video+image dataset of **over 1 million hours**, with the usual self-supervised scaling behavior on downstream understanding tasks.
- **V-JEPA 2-AC**: an **action-conditioned world model** trained with **under 62 hours of unlabeled robot video**, which was used **zero-shot on Franka arms in two labs** for picking and placing — the paper's main claim about planning.
- Pairs with an LLM for video question answering; no labels are needed for the pretraining stage.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Assran_et_al._2025_2506.09985.md`](references/papers/Assran_et_al._2025_2506.09985.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (I-JEPA → V-JEPA → V-JEPA 2; siblings: VideoMAE, DINO-world models, VGGT-World; competitors: generative video world models).
