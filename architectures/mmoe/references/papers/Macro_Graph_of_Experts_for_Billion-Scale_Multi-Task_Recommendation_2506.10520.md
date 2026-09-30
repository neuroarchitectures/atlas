# Paper (Yao et al. 2025)

> Source: `https://arxiv.org/abs/2506.10520`

---

**Macro Graph of Experts for Billion-Scale Multi-Task Recommendation**

Hongyu Yao, Zijin Hong, Hao Chen, Zhiqing Li, Qijie Shen, Zuobin Ying, Qihua Feng, Huan Gong, Feiran Huang

**Abstract**

Graph-based multi-task learning at billion-scale presents a significant challenge, as different tasks correspond to distinct billion-scale graphs. Traditional multi-task learning methods often neglect these graph structures, relying solely on individual user and item embeddings. However, disregarding graph structures overlooks substantial potential for improving performance. In this paper, we introduce the Macro Graph of Experts (MGOE) framework, the first approach capable of leveraging macro graph embeddings to capture task-specific macro features while modeling the correlations between task-specific experts. Specifically, we propose the concept of a Macro Graph Bottom, which, for the first time, enables multi-task learning models to incorporate graph information effectively. We design the Macro Prediction Tower to dynamically integrate macro knowledge across tasks. MGOE has been deployed at scale, powering multi-task learning for a leading billion-scale recommender system, Alibaba. Extensive offline experiments conducted on three public benchmark datasets demonstrate its superiority over state-of-the-art multi-task learning methods, establishing MGOE as a breakthrough in multi-task graph-based recommendation. Furthermore, online A/B tests confirm the superiority of MGOE in billion-scale recommender systems.

---

- **arXiv ID**: `2506.10520`
- **Published**: 2025-06-12
- **Categories**: cs.IR, cs.LG
- **PDF**: https://arxiv.org/pdf/2506.10520
