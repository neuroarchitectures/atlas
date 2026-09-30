# Paper (Shivottam et al. 2026)

> Source: `https://arxiv.org/abs/2605.06588`

---

**Towards Metric-Faithful Neural Graph Matching**

Jyotirmaya Shivottam, Subhankar Mishra

**Abstract**

Graph Edit Distance (GED) is a fundamental, albeit NP-hard, metric for structural graph similarity. Recent neural graph matching architectures approximate GED by first encoding graphs with a Graph Neural Network (GNN) and then applying either a graph-level regression head or a matching-based alignment module. Despite substantial architectural progress, the role of encoder geometry in neural GED estimation remains poorly understood. In this paper, we develop a theoretical framework that connects encoder geometry to GED estimation quality for two broad classes of neural GED estimators: graph similarity predictors and alignment-based methods. On fixed graph collections, where the doubly-stochastic metric $d_{\mathrm{DS}}$ is comparable to GED, we show that graph-level bi-Lipschitz encoders yield controlled GED surrogates and improved ranking stability; for matching-based estimators, node-level bi-Lipschitz geometry propagates to encoder-induced alignment costs and the resulting optimized alignment objective. We instantiate this perspective using FSW-GNN, a bi-Lipschitz WL-equivalent encoder, as a drop-in replacement in representative neural GED architectures. Across representative baselines and benchmark datasets, the resulting geometry-aware variants significantly improve GED prediction and ranking metrics. A faithfulness case study of untrained encoders, together with ablations and transfer experiments, supports the view that these gains arise from improved representation geometry, positioning encoder geometry as a useful design principle for neural graph matching.

---

- **arXiv ID**: `2605.06588`
- **Published**: 2026-05-07
- **Categories**: cs.LG, cs.AI
- **PDF**: https://arxiv.org/pdf/2605.06588
