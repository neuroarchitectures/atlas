# ViTPose++: Vision Transformer for Generic Body Pose Estimation

> Source: `https://arxiv.org/abs/2023.33300`

---

**Authors:** Yufei Xu, Jing Zhang, Qiming Zhang, Dacheng Tao

**Published:** 2023

**Concepts:** Computer science, Scalability, Artificial intelligence, Transformer, Inference

## Abstract

In this paper, we show the surprisingly good properties of plain vision transformers for body pose estimation from various aspects, namely simplicity in model structure, scalability in model size, flexibility in training paradigm, and transferability of knowledge between models, through a simple baseline model dubbed ViTPose. ViTPose employs the plain and non-hierarchical vision transformer as an encoder to encode features and a lightweight decoder to decode body keypoints in either a top-down or a bottom-up manner. It can be scaled to 1B parameters by taking the advantage of the scalable model capacity and high parallelism, setting a new Pareto front for throughput and performance. Besides, ViTPose is very flexible regarding the attention type, input resolution, and pre-training and fine-tuning strategy. Based on the flexibility, a novel ViTPose++ model is proposed to deal with heterogeneous body keypoint categories via knowledge factorization, i.e., adopting task-agnostic and task-specific feed-forward networks in the transformer. We also demonstrate that the knowledge of large ViTPose models can be easily transferred to small ones via a simple knowledge token. Our largest single model ViTPose-G sets a new record on the MS COCO test set without model ensemble. Furthermore, our ViTPose++ model achieves state-of-the-art performance simultaneously on a series of body pose estimation tasks, including MS COCO, AI Challenger, OCHuman, MPII for human keypoint detection, COCO-Wholebody for whole-body keypoint detection, as well as AP-10K and APT-36K for animal keypoint detection, without sacrificing inference speed.

**PDF:** https://arxiv.org/pdf/2023.33300