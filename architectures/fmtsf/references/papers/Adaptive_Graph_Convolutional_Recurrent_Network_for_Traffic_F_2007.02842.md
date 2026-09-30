# Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting

> Source: `https://arxiv.org/abs/2007.02842`

---

**Authors:** Lei Bai, Lina Yao, Can Li, Xianzhi Wang, Can Wang

**Published:** 2020

**Concepts:** Computer science, Graph, Margin (machine learning), Convolutional neural network, Data mining

## Abstract

Modeling complex spatial and temporal correlations in the correlated time series data is indispensable for understanding the traffic dynamics and predicting the future status of an evolving traffic system. Recent works focus on designing complicated graph neural network architectures to capture shared patterns with the help of pre-defined graphs. In this paper, we argue that learning node-specific patterns is essential for traffic forecasting while the pre-defined graph is avoidable. To this end, we propose two adaptive modules for enhancing Graph Convolutional Network (GCN) with new capabilities: 1) a Node Adaptive Parameter Learning (NAPL) module to capture node-specific patterns; 2) a Data Adaptive Graph Generation (DAGG) module to infer the inter-dependencies among different traffic series automatically. We further propose an Adaptive Graph Convolutional Recurrent Network (AGCRN) to capture fine-grained spatial and temporal correlations in traffic series automatically based on the two modules and recurrent networks. Our experiments on two real-world traffic datasets show AGCRN outperforms state-of-the-art by a significant margin without pre-defined graphs about spatial connections.

**PDF:** https://arxiv.org/pdf/2007.02842