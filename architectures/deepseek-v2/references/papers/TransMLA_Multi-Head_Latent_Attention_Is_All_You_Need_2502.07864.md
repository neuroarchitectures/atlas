# Paper (Meng et al. 2025)

> Source: `https://arxiv.org/abs/2502.07864`

---

**TransMLA: Multi-Head Latent Attention Is All You Need**

Fanxu Meng, Pingzhi Tang, Xiaojuan Tang, Zengwei Yao, Xing Sun, Muhan Zhang

**Abstract**

In this paper, we present TransMLA, a framework that seamlessly converts any GQA-based pre-trained model into an MLA-based model. Our approach enables direct compatibility with DeepSeek's codebase, allowing these models to fully leverage DeepSeek-specific optimizations such as vLLM and SGlang. By compressing 93% of the KV cache in LLaMA-2-7B, TransMLA achieves a 10.6x inference speedup at an 8K context length while preserving meaningful output quality. Additionally, the model requires only 6 billion tokens for fine-tuning to regain performance on par with the original across multiple benchmarks. TransMLA offers a practical solution for migrating GQA-based models to the MLA structure. When combined with DeepSeek's advanced features, such as FP8 quantization and Multi-Token Prediction, even greater inference acceleration can be realized.

---

- **arXiv ID**: `2502.07864`
- **Published**: 2025-02-11
- **Categories**: cs.LG, cs.AI
- **PDF**: https://arxiv.org/pdf/2502.07864
