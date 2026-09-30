# Paper (Jiang et al. 2023)

> Source: `https://arxiv.org/abs/2307.09193`

---

**ESMC: Entire Space Multi-Task Model for Post-Click Conversion Rate via Parameter Constraint**

Zhenhao Jiang, Biao Zeng, Hao Feng, Jin Liu, Jicong Fan, Jie Zhang, Jia Jia, Ning Hu, Xingyu Chen, Xuguang Lan

**Abstract**

Large-scale online recommender system spreads all over the Internet being in charge of two basic tasks: Click-Through Rate (CTR) and Post-Click Conversion Rate (CVR) estimations. However, traditional CVR estimators suffer from well-known Sample Selection Bias and Data Sparsity issues. Entire space models were proposed to address the two issues via tracing the decision-making path of "exposure_click_purchase". Further, some researchers observed that there are purchase-related behaviors between click and purchase, which can better draw the user's decision-making intention and improve the recommendation performance. Thus, the decision-making path has been extended to "exposure_click_in-shop action_purchase" and can be modeled with conditional probability approach. Nevertheless, we observe that the chain rule of conditional probability does not always hold. We report Probability Space Confusion (PSC) issue and give a derivation of difference between ground-truth and estimation mathematically. We propose a novel Entire Space Multi-Task Model for Post-Click Conversion Rate via Parameter Constraint (ESMC) and two alternatives: Entire Space Multi-Task Model with Siamese Network (ESMS) and Entire Space Multi-Task Model in Global Domain (ESMG) to address the PSC issue. Specifically, we handle "exposure_click_in-shop action" and "in-shop action_purchase" separately in the light of characteristics of in-shop action. The first path is still treated with conditional probability while the second one is treated with parameter constraint strategy. Experiments on both offline and online environments in a large-scale recommendation system illustrate the superiority of our proposed methods over state-of-the-art models. The real-world datasets will be released.

---

- **arXiv ID**: `2307.09193`
- **Published**: 2023-07-18
- **Categories**: cs.AI, cs.IR
- **PDF**: https://arxiv.org/pdf/2307.09193
