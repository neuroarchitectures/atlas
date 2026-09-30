# Paper (Li et al. 2026)

> Source: `https://arxiv.org/abs/2602.13282`

---

**GraFSTNet: Graph-based Frequency SpatioTemporal Network for Cellular Traffic Prediction**

Ziyi Li, Hui Ma, Fei Xing, Chunjiong Zhang, Ming Yan

**Abstract**

With rapid expansion of cellular networks and the proliferation of mobile devices, cellular traffic data exhibits complex temporal dynamics and spatial correlations, posing challenges to accurate traffic prediction. Previous methods often focus predominantly on temporal modeling or depend on predefined spatial topologies, which limits their ability to jointly model spatio-temporal dependencies and effectively capture periodic patterns in cellular traffic. To address these issues, we propose a cellular traffic prediction framework that integrates spatio-temporal modeling with time-frequency analysis. First, we construct a spatial modeling branch to capture inter-cell dependencies through an attention mechanism, minimizing the reliance on predefined topological structures. Second, we build a time-frequency modeling branch to enhance the representation of periodic patterns. Furthermore, we introduce an adaptive-scale LogCosh loss function, which adjusts the error penalty based on traffic magnitude, preventing large errors from dominating the training process and helping the model maintain relatively stable prediction accuracy across different traffic intensities. Experiments on three open-sourced datasets demonstrate that the proposed method achieves prediction performance superior to state-of-the-art approaches.

---

- **arXiv ID**: `2602.13282`
- **Published**: 2026-02-06
- **Categories**: cs.NI, cs.AI
- **PDF**: https://arxiv.org/pdf/2602.13282
