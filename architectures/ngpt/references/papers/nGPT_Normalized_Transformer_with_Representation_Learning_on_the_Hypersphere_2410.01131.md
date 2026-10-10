# Paper (Loshchilov et al. 2024)

> Source: `https://arxiv.org/abs/2410.01131`

---

**nGPT: Normalized Transformer with Representation Learning on the Hypersphere**

Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, Boris Ginsburg

**Abstract**

We propose a novel neural network architecture, the normalized Transformer (nGPT) with representation learning on the hypersphere. In nGPT, all vectors forming the embeddings, MLP, attention matrices and hidden states are unit norm normalized. The input stream of tokens travels on the surface of a hypersphere, with each layer contributing a displacement towards the target output predictions. These displacements are defined by the MLP and attention blocks, whose vector components also reside on the same hypersphere. Experiments show that nGPT learns much faster, reducing the number of training steps required to achieve the same accuracy by a factor of 4 to 20, depending on the sequence length.

---

- **arXiv ID**: `2410.01131`
- **Published**: 2024-10-01
- **Categories**: cs.LG, cs.AI
- **PDF**: https://arxiv.org/pdf/2410.01131
