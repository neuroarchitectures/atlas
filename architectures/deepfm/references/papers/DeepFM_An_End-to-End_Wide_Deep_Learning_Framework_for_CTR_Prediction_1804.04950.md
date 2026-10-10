# Paper (Guo et al. 2018)

> Source: `https://arxiv.org/abs/1804.04950`

---

**DeepFM: An End-to-End Wide & Deep Learning Framework for CTR Prediction**

Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He, Zhenhua Dong

**Abstract**

Learning sophisticated feature interactions behind user behaviors is critical in maximizing CTR for recommender systems. Despite great progress, existing methods have a strong bias towards low- or high-order interactions, or rely on expertise feature engineering. In this paper, we show that it is possible to derive an end-to-end learning model that emphasizes both low- and high-order feature interactions. The proposed framework, DeepFM, combines the power of factorization machines for recommendation and deep learning for feature learning in a new neural network architecture. Compared to the latest Wide & Deep model from Google, DeepFM has a shared raw feature input to both its "wide" and "deep" components, with no need of feature engineering besides raw features. DeepFM, as a general learning framework, can incorporate various network architectures in its deep component. In this paper, we study two instances of DeepFM where its "deep" component is DNN and PNN respectively, for which we denote as DeepFM-D and DeepFM-P. Comprehensive experiments are conducted to demonstrate the effectiveness of DeepFM-D and DeepFM-P over the existing models for CTR prediction, on both benchmark data and commercial data. We conduct online A/B test in Huawei App Market, which reveals that DeepFM-D leads to more than 10% improvement of click-through rate in the production environment, compared to a well-engineered LR model. We also covered related practice in deploying our framework in Huawei App Market.

---

- **arXiv ID**: `1804.04950`
- **Published**: 2018-04-12
- **Categories**: cs.IR, cs.LG, stat.ML
- **PDF**: https://arxiv.org/pdf/1804.04950
