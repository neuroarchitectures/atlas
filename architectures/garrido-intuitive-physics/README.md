# Garrido Intuitive Physics

## Overview

This work investigates whether JEPA representations truly **understand** the physical world, or merely memorize visual patterns. Using a **violation-of-expectation** framework — if a model exhibits higher "surprise" (prediction error) for physically impossible events, it understands the corresponding physical law — the study tests three intuitive physics properties: **object permanence** (objects don't disappear when occluded), **shape consistency** (objects don't deform), and **spatial continuity** (objects don't pass through walls). The key finding: **latent-space prediction (V-JEPA) > pixel-space prediction > multimodal LLMs**. V-JEPA shows above-chance performance on multiple intuitive physics tests, while pixel-space predictors and text-reasoning MLLMs perform near chance. Even models trained on just one week of unique video show above-chance performance, challenging the assumption that "core knowledge" requires hardcoding.

- **Year:** 2025
- **Authors:** Luis Garrido et al. (Meta FAIR / NYU)
- **Source:** arXiv:2502.11831 — *Intuitive physics understanding emerges from self-supervised pretraining on natural videos*
- **Category:** DL/World-Model

## Key Characteristics

- **Violation-of-expectation framework**: models that show higher prediction error for physically impossible events are judged to "understand" the corresponding physical law.
- **Three intuitive physics properties tested**: object permanence, shape consistency, spatial continuity.
- **Latent > pixel > LLM**: V-JEPA (latent-space prediction) shows above-chance performance; pixel-space predictors and text-reasoning MLLMs perform near chance.
- **Emergent, not hardcoded**: even models trained on one week of unique video show above-chance performance — core knowledge emerges from self-supervised prediction, not from built-in modules.
- **Predictive coding connection**: jointly learning abstract representations and predicting missing sensory input (predictive coding) is sufficient for intuitive physics understanding.
- **Challenge to core knowledge hypothesis**: the results challenge the assumption that core knowledge needs to be hardcoded.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Garrido_et_al._2025_2502.11831.md`](references/papers/Garrido_et_al._2025_2502.11831.md)

## Related Architectures

See `architecture.md` → Evolution section (V-JEPA → Garrido evaluation; sibling: LeWM validates physical structure probing; this work bridges representation learning and physical understanding).
