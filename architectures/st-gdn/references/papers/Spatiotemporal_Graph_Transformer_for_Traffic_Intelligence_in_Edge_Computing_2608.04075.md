# Paper (Ale et al. 2026)

> Source: `https://arxiv.org/abs/2608.04075`

---

**Spatiotemporal Graph Transformer for Traffic Intelligence in Edge Computing**

Laha Ale, Letian Lin, Na Cao, Zheng Ma, Peng Yu

**Abstract**

Accurate traffic forecasting is essential for proactive resource management in edge computing, where service demand evolves dynamically across both space and time. In practical cellular edge systems, traffic exhibits strong spatial correlations among neighboring service regions and long-range temporal dependencies driven by user mobility and application behavior. Existing recurrent forecasting approaches can capture short-term dynamics but often struggle to model long-horizon traffic evolution under non-stationary conditions. To address this challenge, we propose a spatiotemporal graph Transformer framework that jointly models spatial interactions and temporal dependencies for traffic forecasting in edge computing. The framework employs graph neural networks to capture spatial correlations among service regions and leverages Transformer-based self-attention to learn long-range temporal patterns from historical traffic observations. By decoupling spatial representation learning from temporal reasoning, the proposed approach provides an effective mechanism for large-scale spatiotemporal traffic modeling. Extensive experiments on a real-world cellular network dataset demonstrate that the proposed graph Transformer consistently outperforms recurrent graph-based baselines, including GCN-RNN, GCN-LSTM, and GCN-GRU models, across multiple forecasting horizons. The resulting forecasts enable more effective proactive resource provisioning and reduce overload risk compared with reactive management strategies. These results highlight the potential of graph-enhanced attention mechanisms for building intelligent and adaptive edge computing systems.

---

- **arXiv ID**: `2608.04075`
- **Published**: 2026-08-04
- **Categories**: cs.LG, cs.AI
- **PDF**: https://arxiv.org/pdf/2608.04075
