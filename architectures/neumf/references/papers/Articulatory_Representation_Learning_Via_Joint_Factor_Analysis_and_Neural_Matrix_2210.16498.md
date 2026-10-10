# Paper (Lian et al. 2022)

> Source: `https://arxiv.org/abs/2210.16498`

---

**Articulatory Representation Learning Via Joint Factor Analysis and Neural Matrix Factorization**

Jiachen Lian, Alan W Black, Yijing Lu, Louis Goldstein, Shinji Watanabe, Gopala K. Anumanchipalli

**Abstract**

Articulatory representation learning is the fundamental research in modeling neural speech production system. Our previous work has established a deep paradigm to decompose the articulatory kinematics data into gestures, which explicitly model the phonological and linguistic structure encoded with human speech production mechanism, and corresponding gestural scores. We continue with this line of work by raising two concerns: (1) The articulators are entangled together in the original algorithm such that some of the articulators do not leverage effective moving patterns, which limits the interpretability of both gestures and gestural scores; (2) The EMA data is sparsely sampled from articulators, which limits the intelligibility of learned representations. In this work, we propose a novel articulatory representation decomposition algorithm that takes the advantage of guided factor analysis to derive the articulatory-specific factors and factor scores. A neural convolutive matrix factorization algorithm is then employed on the factor scores to derive the new gestures and gestural scores. We experiment with the rtMRI corpus that captures the fine-grained vocal tract contours. Both subjective and objective evaluation results suggest that the newly proposed system delivers the articulatory representations that are intelligible, generalizable, efficient and interpretable.

---

- **arXiv ID**: `2210.16498`
- **Published**: 2022-10-29
- **Categories**: eess.AS, cs.SD
- **PDF**: https://arxiv.org/pdf/2210.16498
