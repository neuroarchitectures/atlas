# VideoPrism

## Overview

VideoPrism is Google DeepMind's **foundational visual encoder for video understanding**: a single **frozen** video encoder that serves web video QA, classification, retrieval, and CV-for-science tasks. It is pretrained in two stages — video-text contrastive learning on 36M high-quality video-caption pairs, then masked video modeling on 582M clips with noisy parallel text — with **cascade masking**, **token shuffling**, and **global-local distillation** as the key techniques. Evaluated on 33 benchmarks, it set SOTA on 31 of them.

- **Year:** 2024
- **Authors:** Zhao et al. (Google DeepMind)
- **Source:** arXiv:2402.13217 — *VideoPrism: A Foundational Visual Encoder for Video Understanding* (ICML 2024)
- **Category:** DL/Video-Foundation

## Key Characteristics

- **One frozen encoder, many tasks**: the same frozen video backbone drives classification, retrieval, QA, localization, and science benchmarks — task-specific adaptation is optional.
- **Two-stage pretraining**: (1) video-text contrastive alignment on 36M curated pairs; (2) masked video modeling on 582M clips with noisy parallel text — designed so the model learns mostly from video, using text as a signal rather than a crutch.
- **Cascade masking**: per-sample mask ratios cascade across a range — from low (contrastive-like global alignment) to very high (pure masked modeling) — unifying the two regimes in one stage instead of trading them off.
- **Token shuffling**: spatio-temporal token order is shuffled to break low-level shortcuts in masked modeling.
- **Global-local distillation**: the global (CLS) embedding is distilled into local patch tokens, so frozen local features inherit global semantics.
- **Models**: ViT-B (114M) / ViT-L (354M) video encoders, plus dual-tower **LvT** video-text variants (248M / 580M) for retrieval. Input 288×288, 16×16 per-frame token grid.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Zhao_et_al._2024_2402.13217.md`](references/papers/Zhao_et_al._2024_2402.13217.md); official code at [google-deepmind/videoprism](https://github.com/google-deepmind/videoprism) (Apache-2.0)

## Related Architectures

See `architecture.md` → Evolution section (CLIP → video contrastive encoders → VideoPrism; sibling: V-JEPA 2 — latent prediction without pixels or text crutches).
