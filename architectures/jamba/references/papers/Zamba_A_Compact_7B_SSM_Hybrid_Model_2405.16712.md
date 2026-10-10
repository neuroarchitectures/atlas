# Paper (Glorioso et al. 2024)

> Source: `https://arxiv.org/abs/2405.16712`

---

**Zamba: A Compact 7B SSM Hybrid Model**

Paolo Glorioso, Quentin Anthony, Yury Tokpanov, James Whittington, Jonathan Pilault, Adam Ibrahim, Beren Millidge

**Abstract**

In this technical report, we present Zamba, a novel 7B SSM-transformer hybrid model which achieves competitive performance against leading open-weight models at a comparable scale. Zamba is trained on 1T tokens from openly available datasets and is the best non-transformer model at this scale. Zamba pioneers a unique architecture combining a Mamba backbone with a single shared attention module, thus obtaining the benefits of attention at minimal parameter cost. Due to its architecture, Zamba is significantly faster at inference than comparable transformer models and requires substantially less memory for generation of long sequences. Zamba is pretrained in two phases: the first phase is based on existing web datasets, while the second one consists of annealing the model over high-quality instruct and synthetic datasets, and is characterized by a rapid learning rate decay. We open-source the weights and all checkpoints for Zamba, through both phase 1 and annealing phases.

---

- **arXiv ID**: `2405.16712`
- **Published**: 2024-05-26
- **Categories**: cs.LG, cs.AI, cs.CL
- **PDF**: https://arxiv.org/pdf/2405.16712
