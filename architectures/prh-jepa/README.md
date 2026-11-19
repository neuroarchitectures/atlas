# PRH-JEPA

## Overview

PRH-JEPA (Proving and Improving the Platonic Representation Hypothesis with JEPAs) is the **theoretically deepest** work in the LeCun JEPA series. It transforms the "Platonic Representation Hypothesis" — the conjecture that different models/modalities converge to the same representation — from an empirical observation into a **mathematical theorem**. **Theorem 1 (Good Views)**: any number of JEPAs whose positive pairs are two observations of the same scene ("good views") learn representations that differ only by an orthogonal rotation. **Theorem 2 (Bad Views)**: next-token prediction is a "bad views" JEPA — positive pairs are different parts of the same document, not two observations of the same scene — explaining why language models need massive scale to align with vision. The paper demonstrates 42× sample efficiency for good-view JEPAs over next-token prediction, and shows that only orthogonal-rotation alignment transfers across representations.

- **Year:** 2026
- **Authors:** Randall Balestriero, Yann LeCun (Meta FAIR / NYU)
- **Source:** arXiv:2610.39302 — *Proving and Improving the Platonic Representation Hypothesis with JEPAs*
- **Category:** DL/Theory

## Key Characteristics

- **Platonic Representation Hypothesis formalized**: different models observing the same world converge to the same representation, up to rotation.
- **Theorem 1 (Good Views)**: JEPAs with good-view positive pairs (two observations of the same scene) learn representations differing only by an orthogonal rotation.
- **Theorem 2 (Bad Views)**: next-token prediction is a bad-view JEPA — explains why LLMs need scale to align with vision.
- **42× sample efficiency**: good-view JEPA with 21.6M samples matches next-token prediction with 900M samples (.109 vs .111 alignment).
- **Scale advantage**: 109M parameters, 116B tokens, outperforms models 660× larger and 100× more data.
- **Only orthogonal rotations transfer**: COCO CIDEr .806 (orthogonal) vs .635 (unconstrained linear) vs .585 (two-layer MLP).
- **Zero-shot unpaired alignment**: ImageNet-100 top-5 64.4%, exceeding benchmark upper bound (≤56%) up to DINOv3-7B / Qwen3-8B.
- **Alignment scaling law**: 33 cells fit RMSE .0067, predict unseen cells RMSE .0115.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Balestriero_LeCun_2026_2610.39302.md`](references/papers/Balestriero_LeCun_2026_2610.39302.md)

## Related Architectures

See `architecture.md` → Evolution section (LeJEPA → PRH-JEPA; sibling: LeJEPA Identifiability provides linear recovery guarantee; PRH-JEPA provides cross-representation alignment guarantee).
