# Paper (Zhang et al. 2024)

> Source: `https://arxiv.org/abs/2407.19239`

---

**MaTrRec: Uniting Mamba and Transformer for Sequential Recommendation**

Shun Zhang, Runsen Zhang, Zhirong Yang

**Abstract**

Sequential recommendation systems aim to provide personalized recommendations by analyzing dynamic preferences and dependencies within user behavior sequences. Recently, Transformer models can effectively capture user preferences. However, their quadratic computational complexity limits recommendation performance on long interaction sequence data. Inspired by the State Space Model (SSM)representative model, Mamba, which efficiently captures user preferences in long interaction sequences with linear complexity, we find that Mamba's recommendation effectiveness is limited in short interaction sequences, with failing to recall items of actual interest to users and exacerbating the data sparsity cold start problem. To address this issue, we innovatively propose a new model, MaTrRec, which combines the strengths of Mamba and Transformer. This model fully leverages Mamba's advantages in handling long-term dependencies and Transformer's global attention advantages in short-term dependencies, thereby enhances predictive capabilities on both long and short interaction sequence datasets while balancing model efficiency. Notably, our model significantly improves the data sparsity cold start problem, with an improvement of up to 33% on the highly sparse Amazon Musical Instruments dataset. We conducted extensive experimental evaluations on five widely used public datasets. The experimental results show that our model outperforms the current state-of-the-art sequential recommendation models on all five datasets. The code is available at https://github.com/Unintelligentmumu/MaTrRec.

---

- **arXiv ID**: `2407.19239`
- **Published**: 2024-07-27
- **Categories**: cs.IR
- **PDF**: https://arxiv.org/pdf/2407.19239
