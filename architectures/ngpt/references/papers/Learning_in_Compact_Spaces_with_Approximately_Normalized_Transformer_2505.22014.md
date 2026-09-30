# Paper (Franke et al. 2025)

> Source: `https://arxiv.org/abs/2505.22014`

---

**Learning in Compact Spaces with Approximately Normalized Transformer**

Jörg K. H. Franke, Urs Spiegelhalter, Marianna Nezhurina, Jenia Jitsev, Frank Hutter, Michael Hefenbrock

**Abstract**

The successful training of deep neural networks requires addressing challenges such as overfitting, numerical instabilities leading to divergence, and increasing variance in the residual stream. A common solution is to apply regularization and normalization techniques that usually require tuning additional hyperparameters. An alternative is to force all parameters and representations to lie on a hypersphere. This removes the need for regularization and increases convergence speed, but comes with additional costs. In this work, we propose a more holistic, approximate normalization via simple scalar multiplications motivated by the tight concentration of the norms of high-dimensional random vectors. Additionally, instead of applying strict normalization for the parameters, we constrain their norms. These modifications remove the need for weight decay and learning rate warm-up as well, but do not increase the total number of normalization layers. Our experiments with transformer architectures show up to 40% faster convergence compared to GPT models with QK normalization, with only 3% additional runtime cost. When deriving scaling laws, we found that our method enables training with larger batch sizes while preserving the favorable scaling characteristics of classic GPT architectures.

---

- **arXiv ID**: `2505.22014`
- **Published**: 2025-05-28
- **Categories**: cs.LG
- **PDF**: https://arxiv.org/pdf/2505.22014
