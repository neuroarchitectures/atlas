# Paper (Li et al. 2024)

> Source: `https://arxiv.org/abs/2402.08961`

---

**HyCubE: Efficient Knowledge Hypergraph 3D Circular Convolutional Embedding**

Zhao Li, Xin Wang, Jun Zhao, Wenbin Guo, Jianxin Li

**Abstract**

Knowledge hypergraph embedding models are usually computationally expensive due to the inherent complex semantic information. However, existing works mainly focus on improving the effectiveness of knowledge hypergraph embedding, making the model architecture more complex and redundant. It is desirable and challenging for knowledge hypergraph embedding to reach a trade-off between model effectiveness and efficiency. In this paper, we propose an end-to-end efficient knowledge hypergraph embedding model, HyCubE, which designs a novel 3D circular convolutional neural network and the alternate mask stack strategy to enhance the interaction and extraction of feature information comprehensively. Furthermore, our proposed model achieves a better trade-off between effectiveness and efficiency by adaptively adjusting the 3D circular convolutional layer structure to handle n-ary knowledge tuples of different arities with fewer parameters. In addition, we use a knowledge hypergraph 1-N multilinear scoring way to accelerate the model training efficiency further. Finally, extensive experimental results on all datasets demonstrate that our proposed model consistently outperforms state-of-the-art baselines, with an average improvement of 8.22% and a maximum improvement of 33.82% across all metrics. Meanwhile, HyCubE is 6.12x faster, GPU memory usage is 52.67% lower, and the number of parameters is reduced by 85.21% compared with the average metric of the latest state-of-the-art baselines.

---

- **arXiv ID**: `2402.08961`
- **Published**: 2024-02-14
- **Categories**: cs.AI
- **PDF**: https://arxiv.org/pdf/2402.08961
