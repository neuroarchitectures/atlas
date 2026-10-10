# Paper (Tay et al. 2018)

> Source: `https://arxiv.org/abs/1806.06446`

---

**Self-Attentive Neural Collaborative Filtering**

Yi Tay, Shuai Zhang, Luu Anh Tuan, Siu Cheung Hui

**Abstract**

This paper has been withdrawn as we discovered a bug in our tensorflow implementation that involved accidental mixing of vectors across batches. This lead to different inference results given different batch sizes which is completely strange. The performance scores still remain the same but we concluded that it was not the self-attention that contributed to the performance. We are withdrawing the paper because this renders the main claim of the paper false. Thanks to Guan Xinyu from NUS for discovering this issue in our previously open source code.

---

- **arXiv ID**: `1806.06446`
- **Published**: 2018-06-17
- **Categories**: cs.IR, cs.AI, cs.LG, cs.NE
- **PDF**: https://arxiv.org/pdf/1806.06446
