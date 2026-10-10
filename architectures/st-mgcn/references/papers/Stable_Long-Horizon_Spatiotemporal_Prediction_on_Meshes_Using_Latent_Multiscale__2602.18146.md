# Paper (Salesses et al. 2026)

> Source: `https://arxiv.org/abs/2602.18146`

---

**Stable Long-Horizon Spatiotemporal Prediction on Meshes Using Latent Multiscale Recurrent Graph Neural Networks**

Lionel Salesses, Larbi Arbaoui, Tariq Benamara, Arnaud Francois, Caroline Sainvitu

**Abstract**

Accurate long-horizon prediction of spatiotemporal fields on complex geometries is a fundamental challenge in scientific machine learning, with applications such as additive manufacturing where temperature histories govern defect formation and mechanical properties. High-fidelity simulations are accurate but computationally costly, and despite recent advances, machine learning methods remain challenged by long-horizon temperature and gradient prediction. We propose a deep learning framework for predicting full temperature histories directly on meshes, conditioned on geometry and process parameters, while maintaining stability over thousands of time steps and generalizing across heterogeneous geometries. The framework adopts a temporal multiscale architecture composed of two coupled models operating at complementary time scales. Both models rely on a latent recurrent graph neural network to capture spatiotemporal dynamics on meshes, while a variational graph autoencoder provides a compact latent representation that reduces memory usage and improves training stability. Experiments on simulated powder bed fusion data demonstrate accurate and temporally stable long-horizon predictions across diverse geometries, outperforming existing baseline. Although evaluated in two dimensions, the framework is general and extensible to physics-driven systems with multiscale dynamics and to three-dimensional geometries.

---

- **arXiv ID**: `2602.18146`
- **Published**: 2026-02-20
- **Categories**: cs.LG, cond-mat.mtrl-sci, physics.app-ph
- **PDF**: https://arxiv.org/pdf/2602.18146
