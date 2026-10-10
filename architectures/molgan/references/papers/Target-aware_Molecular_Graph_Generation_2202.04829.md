# Paper (Tan et al. 2022)

> Source: `https://arxiv.org/abs/2202.04829`

---

**Target-aware Molecular Graph Generation**

Cheng Tan, Zhangyang Gao, Stan Z. Li

**Abstract**

Generating molecules with desired biological activities has attracted growing attention in drug discovery. Previous molecular generation models are designed as chemocentric methods that hardly consider the drug-target interaction, limiting their practical applications. In this paper, we aim to generate molecular drugs in a target-aware manner that bridges biological activity and molecular design. To solve this problem, we compile a benchmark dataset from several publicly available datasets and build baselines in a unified framework. Building on the recent advantages of flow-based molecular generation models, we propose SiamFlow, which forces the flow to fit the distribution of target sequence embeddings in latent space. Specifically, we employ an alignment loss and a uniform loss to bring target sequence embeddings and drug graph embeddings into agreements while avoiding collapse. Furthermore, we formulate the alignment into a one-to-many problem by learning spaces of target sequence embeddings. Experiments quantitatively show that our proposed method learns meaningful representations in the latent space toward the target-aware molecular graph generation and provides an alternative approach to bridge biology and chemistry in drug discovery.

---

- **arXiv ID**: `2202.04829`
- **Published**: 2022-02-10
- **Categories**: cs.LG, cs.AI
- **PDF**: https://arxiv.org/pdf/2202.04829
