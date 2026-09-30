# Paper (Xu et al. 2026)

> Source: `https://arxiv.org/abs/2603.12578`

---

**Bridging Sequential and Contextual Features with a Dual-View of Fine-grained Core-Behaviors and Global Interest-Distribution**

Yi Xu, Chaofan Fan, Moyu Zhang, Jinxin Hu, Jiahao Wang, Hao Zhang, Shizhun Wang, Yu Zhang, Xiaoyi Zeng

**Abstract**

Click-through rate (CTR) prediction tasks typically estimate the probability of a user clicking on a candidate item by modeling both user behavior sequence features and the item's contextual features, where the user behavior sequence is particularly critical as it dynamically reflects real-time shifts in user interest. Traditional CTR models often aggregate this dynamic sequence into a single vector before interacting it with contextual features. This approach, however, not only leads to behavior information loss during aggregation but also severely limits the model's capacity to capture interactions between contextual features and specific user behaviors, ultimately impairing its ability to capture fine-grained behavioral details and hindering models' prediction accuracy. Conversely, a naive approach of directly interacting with each user action with contextual features is computationally expensive and introduces significant noise from behaviors irrelevant to the candidate item. This noise tends to overwhelm the valuable signals arising from interactions involving more behaviors relevant to the candidate item. Therefore, to resolve the above issue, we propose a Core-Behaviors and Distributional-Compensation Dual-View Interaction Network (CDNet), which bridges the gap between sequential and contextual feature interactions from two complementary angles: a fine-grained interaction involving the most relevant behaviors and contextual features, and a coarse-grained interaction that models the user's overall interest distribution against the contextual features. By simultaneously capturing important behavioral details without forgoing the holistic user interest, CDNet effectively models the interplay between sequential and contextual features without imposing a significant computational burden. Ultimately, extensive experiments validate the effectiveness of CDNet.

---

- **arXiv ID**: `2603.12578`
- **Published**: 2026-03-13
- **Categories**: cs.IR
- **PDF**: https://arxiv.org/pdf/2603.12578
