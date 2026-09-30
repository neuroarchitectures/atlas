# Paper (Ruiz et al. 2020)

> Source: `https://arxiv.org/abs/2002.01038`

---

**Gated Graph Recurrent Neural Networks**

Luana Ruiz, Fernando Gama, Alejandro Ribeiro

**Abstract**

Graph processes exhibit a temporal structure determined by the sequence index and and a spatial structure determined by the graph support. To learn from graph processes, an information processing architecture must then be able to exploit both underlying structures. We introduce Graph Recurrent Neural Networks (GRNNs) as a general learning framework that achieves this goal by leveraging the notion of a recurrent hidden state together with graph signal processing (GSP). In the GRNN, the number of learnable parameters is independent of the length of the sequence and of the size of the graph, guaranteeing scalability. We prove that GRNNs are permutation equivariant and that they are stable to perturbations of the underlying graph support. To address the problem of vanishing gradients, we also put forward gated GRNNs with three different gating mechanisms: time, node and edge gates. In numerical experiments involving both synthetic and real datasets, time-gated GRNNs are shown to improve upon GRNNs in problems with long term dependencies, while node and edge gates help encode long range dependencies present in the graph. The numerical results also show that GRNNs outperform GNNs and RNNs, highlighting the importance of taking both the temporal and graph structures of a graph process into account.

---

- **arXiv ID**: `2002.01038`
- **Published**: 2020-02-03
- **Categories**: eess.SP, cs.LG
- **PDF**: https://arxiv.org/pdf/2002.01038
