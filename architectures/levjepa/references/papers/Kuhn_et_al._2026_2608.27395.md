# LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics

**Authors:** Julian Kuhn, Florian Maes, Jonathan Serra, Quentin Le Lidec, Yann LeCun, Randall Balestriero, Jan Buettner
**arXiv:** 2608.27395 (2026)

## Abstract

Video joint-embedding predictive architectures (V-JEPA, V-JEPA 2) achieve strong self-supervised video representations but rely on costly heuristics: EMA target encoders, predictor networks, stop-gradient, and structured masking. We introduce LeVJEPA, which eliminates all of these by using SIGReg (Sketched Isotropic Gaussian Regularization) as the sole collapse-prevention mechanism. The architecture reduces to a single block-causal ViT encoder with a small projector. LeVJEPA matches or exceeds V-JEPA 2's performance with 5.6–20.8× less compute across ViT-S/B/L, and exceeds the strongest video baseline by 7.6 points on ImageNet-1K at equal total FLOP budget. We show that random token dropping (up to 95%) acts as augmentation rather than approximation, improving performance from 33.9% to 47.6%. Block-causal attention enables streaming inference and autoregressive world models, with frozen probing performance matching full bidirectional attention. A ViT-Tiny variant achieves non-trivial ImageNet accuracy in 12 hours on a single consumer GPU.
