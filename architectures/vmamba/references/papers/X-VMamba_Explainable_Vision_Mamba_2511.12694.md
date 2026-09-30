# Paper (Mabrok et al. 2025)

> Source: `https://arxiv.org/abs/2511.12694`

---

**X-VMamba: Explainable Vision Mamba**

Mohamed A. Mabrok, Yalda Zafari

**Abstract**

State Space Models (SSMs), particularly the Mamba architecture, have recently emerged as powerful alternatives to Transformers for sequence modeling, offering linear computational complexity while achieving competitive performance. Yet, despite their effectiveness, understanding how these Vision SSMs process spatial information remains challenging due to the lack of transparent, attention-like mechanisms. To address this gap, we introduce a controllability-based interpretability framework that quantifies how different parts of the input sequence (tokens or patches) influence the internal state dynamics of SSMs. We propose two complementary formulations: a Jacobian-based method applicable to any SSM architecture that measures influence through the full chain of state propagation, and a Gramian-based approach for diagonal SSMs that achieves superior speed through closed-form analytical solutions. Both methods operate in a single forward pass with linear complexity, requiring no architectural modifications or hyperparameter tuning. We validate our framework through experiments on three diverse medical imaging modalities, demonstrating that SSMs naturally implement hierarchical feature refinement from diffuse low-level textures in early layers to focused, clinically meaningful patterns in deeper layers. Our analysis reveals domain-specific controllability signatures aligned with diagnostic criteria, progressive spatial selectivity across the network hierarchy, and the substantial influence of scanning strategies on attention patterns. Beyond medical imaging, we articulate applications spanning computer vision, natural language processing, and cross-domain tasks. Our framework establishes controllability analysis as a unified, foundational interpretability paradigm for SSMs across all domains. Code and analysis tools will be made available upon publication

---

- **arXiv ID**: `2511.12694`
- **Published**: 2025-11-16
- **Categories**: cs.CV, cs.LG, math.DS
- **PDF**: https://arxiv.org/pdf/2511.12694
