# Paper (Zhao et al. 2021)

> Source: `https://arxiv.org/abs/2104.06312`

---

**A Non-sequential Approach to Deep User Interest Model for CTR Prediction**

Keke Zhao, Xing Zhao, Qi Cao, Linjian Mo

**Abstract**

Click-Through Rate (CTR) prediction plays an important role in many industrial applications, and recently a lot of attention is paid to the deep interest models which use attention mechanism to capture user interests from historical behaviors. However, most current models are based on sequential models which truncate the behavior sequences by a fixed length, thus have difficulties in handling very long behavior sequences. Another big problem is that sequences with the same length can be quite different in terms of time, carrying completely different meanings. In this paper, we propose a non-sequential approach to tackle the above problems. Specifically, we first represent the behavior data in a sparse key-vector format, where the vector contains rich behavior info such as time, count and category. Next, we enhance the Deep Interest Network to take such rich information into account by a novel attention network. The sparse representation makes it practical to handle large scale long behavior sequences. Finally, we introduce a multidimensional partition framework to mine behavior interactions. The framework can partition data into custom designed time buckets to capture the interactions among information aggregated in different time buckets. Similarly, it can also partition the data into different categories and capture the interactions among them. Experiments are conducted on two public datasets: one is an advertising dataset and the other is a production recommender dataset. Our models outperform other state-of-the-art models on both datasets.

---

- **arXiv ID**: `2104.06312`
- **Published**: 2021-04-05
- **Categories**: cs.IR
- **PDF**: https://arxiv.org/pdf/2104.06312
