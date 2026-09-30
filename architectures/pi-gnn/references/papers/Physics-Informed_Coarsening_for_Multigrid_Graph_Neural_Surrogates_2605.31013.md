# Paper (Bazzi et al. 2026)

> Source: `https://arxiv.org/abs/2605.31013`

---

**Physics-Informed Coarsening for Multigrid Graph Neural Surrogates**

Amir Bazzi, David Cardinaux, Ramy Nemer, Jose Alaves, Arjun Kalkur Matpadi Raghavendra, Elie Hachem

**Abstract**

Learning-based surrogates for partial differential equations have recently matched the accuracy of classical solvers while achieving orders-of-magnitude speedups, predominantly in fluid settings and structured geometries. In contrast, robust surrogates for deformable solids remain underexplored, despite the presence of nonlinear elasticity, plasticity, and transient behavior that challenge standard architectures. We introduce a multigrid graph neural network for solid mechanics that couples an encoder-processor-decoder backbone with a physics-informed coarsening strategy. Instead of downsampling via geometric heuristics, our method scores nodes using a residual-based measure of local physical activity and preferentially retains regions of high strain or stress concentration, allocating multiscale capacity where it is most needed. This preserves long-range interactions through hierarchical message passing while improving stability over long rollouts. We evaluate on multiple datasets covering linear, nonlinear, and transient regimes, and observe consistent gains in accuracy and rollout stability compared to standard sampling baselines. Our results highlight the importance of physics-informed coarsening for scalable surrogate modeling in solid mechanics.

---

- **arXiv ID**: `2605.31013`
- **Published**: 2026-05-29
- **Categories**: cs.LG
- **PDF**: https://arxiv.org/pdf/2605.31013
