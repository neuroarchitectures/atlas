# Paper (Lu et al. 2025)

> Source: `https://arxiv.org/abs/2509.19633`

---

**Mamba Modulation: On the Length Generalization of Mamba**

Peng Lu, Jerry Huang, Qiuhao Zeng, Xinyu Wang, Boxing Chen, Philippe Langlais, Yufei Cui

**Abstract**

The quadratic complexity of the attention mechanism in Transformer models has motivated the development of alternative architectures with sub-quadratic scaling, such as state-space models. Among these, Mamba has emerged as a leading architecture, achieving state-of-the-art results across a range of language modeling tasks. However, Mamba's performance significantly deteriorates when applied to contexts longer than those seen during pre-training, revealing a sharp sensitivity to context length extension. Through detailed analysis, we attribute this limitation to the out-of-distribution behaviour of its state-space dynamics, particularly within the parameterization of the state transition matrix $\mathbf{A}$. Unlike recent works which attribute this sensitivity to the vanished accumulation of discretization time steps, $\exp(-\sum_{t=1}^NΔ_t)$, we establish a connection between state convergence behavior as the input length approaches infinity and the spectrum of the transition matrix $\mathbf{A}$, offering a well-founded explanation of its role in length extension. Next, to overcome this challenge, we propose an approach that applies spectrum scaling to pre-trained Mamba models to enable robust long-context generalization by selectively modulating the spectrum of $\mathbf{A}$ matrices in each layer. We show that this can significantly improve performance in settings where simply modulating $Δ_t$ fails, validating our insights and providing avenues for better length generalization of state-space models with structured transition matrices.

---

- **arXiv ID**: `2509.19633`
- **Published**: 2025-09-23
- **Categories**: cs.LG, cs.AI, stat.ML
- **PDF**: https://arxiv.org/pdf/2509.19633
