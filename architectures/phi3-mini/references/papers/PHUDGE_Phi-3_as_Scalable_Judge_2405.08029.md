# Paper (Deshwal et al. 2024)

> Source: `https://arxiv.org/abs/2405.08029`

---

**PHUDGE: Phi-3 as Scalable Judge**

Mahesh Deshwal, Apoorva Chawla

**Abstract**

In this paper cum technical report, we present PHUDGE A fine tuned Phi3 model that achieved SOTA results in 4 tasks as Feedback Test, Feedback OOD, MT Human, Preference Test surpassing each and every existing model in latency and throughput. It shows very strong correlation not only with GPT4 but with Human annotators too in unseen data as well as in both absolute and relative grading tasks. We have not only addressed the usage of small LMs for cost effective production grade systems but have also shown that Causal modelling is not only slow in nature but sometimes it can hinder models learning capabilities and should be replaced by simpler tasks whenever we can to make the overall system faster and better. We show that by following systematic ML experimentation, thoughtful data augmentation and re purposing the problem itself, we can even beat 10x bigger models even with lesser training data. To the best of our knowledge, we are re the first one to experiment and showcase the usage of generalised version of Earth Movers Distance AKA Wasserstein distance by using Minkowski Distance with a penalty to control loss smoothing and can be used as a loss function instead of Cross Entropy to get stable training and better results for grading tasks.

---

- **arXiv ID**: `2405.08029`
- **Published**: 2024-05-12
- **Categories**: cs.LG, cs.AI
- **PDF**: https://arxiv.org/pdf/2405.08029
