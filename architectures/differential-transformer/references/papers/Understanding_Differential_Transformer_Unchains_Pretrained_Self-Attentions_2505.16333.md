# Paper (Kong et al. 2025)

> Source: `https://arxiv.org/abs/2505.16333`

---

**Understanding Differential Transformer Unchains Pretrained Self-Attentions**

Chaerin Kong, Jiho Jang, Nojun Kwak

**Abstract**

Differential Transformer has recently gained significant attention for its impressive empirical performance, often attributed to its ability to perform noise canceled attention. However, precisely how differential attention achieves its empirical benefits remains poorly understood. Moreover, Differential Transformer architecture demands large-scale training from scratch, hindering utilization of open pretrained weights. In this work, we conduct an in-depth investigation of Differential Transformer, uncovering three key factors behind its success: (1) enhanced expressivity via negative attention, (2) reduced redundancy among attention heads, and (3) improved learning dynamics. Based on these findings, we propose DEX, a novel method to efficiently integrate the advantages of differential attention into pretrained language models. By reusing the softmax attention scores and adding a lightweight differential operation on the output value matrix, DEX effectively incorporates the key advantages of differential attention while remaining lightweight in both training and inference. Evaluations confirm that DEX substantially improves the pretrained LLMs across diverse benchmarks, achieving significant performance gains with minimal adaptation data (< 0.01%).

---

- **arXiv ID**: `2505.16333`
- **Published**: 2025-05-22
- **Categories**: cs.LG
- **PDF**: https://arxiv.org/pdf/2505.16333
