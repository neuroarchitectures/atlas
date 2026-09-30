# Time Interval Aware Self-Attention for Sequential Recommendation

> Source: `https://arxiv.org/abs/6191.33717`

---

**Authors:** Jiacheng Li, Yujie Wang, Julian McAuley

**Published:** 2020

**Concepts:** Timestamp, Computer science, Exploit, Interval (graph theory), Context (archaeology)

## Abstract

Sequential recommender systems seek to exploit the order of users' interactions, in order to predict their next action based on the context of what they have done recently. Traditionally, Markov Chains(MCs), and more recently Recurrent Neural Networks (RNNs) and Self Attention (SA) have proliferated due to their ability to capture the dynamics of sequential patterns. However a simplifying assumption made by most of these models is to regard interaction histories as ordered sequences, without regard for the time intervals between each interaction (i.e., they model the time-order but not the actual timestamp). In this paper, we seek to explicitly model the timestamps of interactions within a sequential modeling framework to explore the influence of different time intervals on next item prediction. We propose TiSASRec (Time Interval aware Self-attention based sequential recommendation), which models both the absolute positions of items as well as the time intervals between them in a sequence. Extensive empirical studies show the features of TiSASRec under different settings and compare the performance of self-attention with different positional encodings. Furthermore, experimental results show that our method outperforms various state-of-the-art sequential models on both sparse and dense datasets and different evaluation metrics.

**PDF:** https://arxiv.org/pdf/6191.33717