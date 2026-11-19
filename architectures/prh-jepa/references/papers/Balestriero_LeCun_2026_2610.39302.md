# Proving and Improving the Platonic Representation Hypothesis with JEPAs

**Authors:** Randall Balestriero, Yann LeCun
**arXiv:** 2610.39302 (2026)

## Abstract

The Platonic Representation Hypothesis posits that different models trained on different data converge to similar representations of the world. We prove this for Joint-Embedding Predictive Architectures (JEPAs). **Theorem 1 (Good Views)**: any number of JEPAs whose positive pairs are two observations of the same scene ("good views") learn representations that differ only by an orthogonal rotation, under mild conditions (whitened second-order anti-collapse, spectral gap). **Theorem 2 (Bad Views)**: next-token prediction is a "bad views" JEPA — positive pairs are different parts of the same document, not two observations of the same scene — explaining why language models require massive scale to align with vision. Good-view JEPAs achieve 42× sample efficiency over next-token prediction (21.6M vs 900M samples for equivalent alignment, .109 vs .111). A 109M-parameter text JEPA trained on 116B tokens outperforms models 660× larger and trained on 100× more data. Only orthogonal-rotation alignment transfers across representations (COCO CIDEr .806 orthogonal vs .635 unconstrained linear vs .585 two-layer MLP). Zero-shot unpaired alignment achieves ImageNet-100 top-5 64.4%, exceeding the benchmark upper bound (≤56%) up to DINOv3-7B / Qwen3-8B.
