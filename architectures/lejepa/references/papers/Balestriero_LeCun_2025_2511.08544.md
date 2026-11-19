# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics

**Authors:** Randall Balestriero, Yann LeCun
**arXiv:** 2511.08544 (2025)

## Abstract

Joint-Embedding Predictive Architectures (JEPA) have emerged as a powerful framework for self-supervised representation learning, but they rely on a collection of empirically-derived heuristics — exponential moving average target encoders, stop-gradients, carefully tuned masking strategies, and predictor networks — that are poorly understood and require extensive hyperparameter tuning. In this work, we introduce LeJEPA, a provably optimal and scalable JEPA variant that eliminates all such heuristics. Our key contribution is SIGReg (Sketched Isotropic Gaussian Regularization), which constrains the embedding distribution to match an isotropic Gaussian distribution. We prove that an isotropic Gaussian is the optimal embedding distribution for minimizing downstream prediction risk under the JEPA alignment objective. The implementation uses the Cramér-Wold theorem to reduce high-dimensional distribution matching to a set of one-dimensional normality tests along random directions, achieving linear complexity. LeJEPA achieves 79% linear probing accuracy on ImageNet-1K with ViT-H/14, and stably operates across 10+ datasets and 60+ architectures without any per-dataset tuning.
