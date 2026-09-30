# Wide & Deep Learning for Recommender Systems

> Source: `https://arxiv.org/abs/8450.29884`

---

**Authors:** Heng-Tze Cheng, Levent Koç, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg S. Corrado, Wei Chai, Mustafa Ispir et al.

**Published:** 2016

**Concepts:** Computer science, Feature engineering, Artificial intelligence, Generalization, Feature (linguistics)

## Abstract

Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. In this paper, we present Wide & Deep learning---jointly trained wide linear models and deep neural networks---to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow.

**PDF:** https://arxiv.org/pdf/8450.29884