# Learning Graph Structures with Transformer for Multivariate Time Series Anomaly Detection in IoT

> Source: `https://arxiv.org/abs/2104.03466`

---

**Authors:** Zekai Chen, Dingshuo Chen, Xiao Zhang, Zixuan Yuan, Xiuzhen Cheng

**Published:** 2021

**Concepts:** Computer science, Anomaly detection, Time series, Multivariate statistics, Transformer

## Abstract

Many real-world IoT systems, which include a variety of internet-connected sensory devices, produce substantial amounts of multivariate time series data. Meanwhile, vital IoT infrastructures like smart power grids and water distribution networks are frequently targeted by cyber-attacks, making anomaly detection an important study topic. Modeling such relatedness is, nevertheless, unavoidable for any efficient and effective anomaly detection system, given the intricate topological and nonlinear connections that are originally unknown among sensors. Furthermore, detecting anomalies in multivariate time series is difficult due to their temporal dependency and stochasticity. This paper presented GTA, a new framework for multivariate time series anomaly detection that involves automatically learning a graph structure, graph convolution, and modeling temporal dependency using a Transformer-based architecture. The connection learning policy, which is based on the Gumbel-softmax sampling approach to learn bi-directed links among sensors directly, is at the heart of learning graph structure. To describe the anomaly information flow between network nodes, we introduced a new graph convolution called Influence Propagation convolution. In addition, to tackle the quadratic complexity barrier, we suggested a multi-branch attention mechanism to replace the original multi-head self-attention method. Extensive experiments on four publicly available anomaly detection benchmarks further demonstrate the superiority of our approach over alternative state-of-the-arts. Codes are available at https://github.com/ZEKAICHEN/GTA.

**PDF:** https://arxiv.org/pdf/2104.03466