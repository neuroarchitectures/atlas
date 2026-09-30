# Paper (Chen et al. 2021)

> Source: `https://arxiv.org/abs/2112.14900`

---

**Motif Graph Neural Network**

Xuexin Chen, Ruichu Cai, Yuan Fang, Min Wu, Zijian Li, Zhifeng Hao

**Abstract**

Graphs can model complicated interactions between entities, which naturally emerge in many important applications. These applications can often be cast into standard graph learning tasks, in which a crucial step is to learn low-dimensional graph representations. Graph neural networks (GNNs) are currently the most popular model in graph embedding approaches. However, standard GNNs in the neighborhood aggregation paradigm suffer from limited discriminative power in distinguishing \emph{high-order} graph structures as opposed to \emph{low-order} structures. To capture high-order structures, researchers have resorted to motifs and developed motif-based GNNs. However, existing motif-based GNNs still often suffer from less discriminative power on high-order structures. To overcome the above limitations, we propose Motif Graph Neural Network (MGNN), a novel framework to better capture high-order structures, hinging on our proposed motif redundancy minimization operator and injective motif combination. First, MGNN produces a set of node representations w.r.t. each motif. The next phase is our proposed redundancy minimization among motifs which compares the motifs with each other and distills the features unique to each motif. Finally, MGNN performs the updating of node representations by combining multiple representations from different motifs. In particular, to enhance the discriminative power, MGNN utilizes an injective function to combine the representations w.r.t. different motifs. We further show that our proposed architecture increases the expressive power of GNNs with a theoretical analysis. We demonstrate that MGNN outperforms state-of-the-art methods on seven public benchmarks on both node classification and graph classification tasks.

---

- **arXiv ID**: `2112.14900`
- **Published**: 2021-12-30
- **Categories**: cs.LG
- **PDF**: https://arxiv.org/pdf/2112.14900
