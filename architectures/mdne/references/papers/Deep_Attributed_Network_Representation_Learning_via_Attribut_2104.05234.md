# Deep Attributed Network Representation Learning via Attribute Enhanced Neighborhood

> Source: `https://arxiv.org/abs/2104.05234`

---

**Authors:** Cong Li, Min Shi, Qu, Bo, Xiang Li

**Published:** 2021

**Concepts:** Computer science, Homophily, Robustness (evolution), Autoencoder, Artificial intelligence

## Abstract

Attributed network representation learning aims at learning node embeddings by integrating network structure and attribute information. It is a challenge to fully capture the microscopic structure and the attribute semantics simultaneously, where the microscopic structure includes the one-step, two-step and multi-step relations, indicating the first-order, second-order and high-order proximity of nodes, respectively. In this paper, we propose a deep attributed network representation learning via attribute enhanced neighborhood (DANRL-ANE) model to improve the robustness and effectiveness of node representations. The DANRL-ANE model adopts the idea of the autoencoder, and expands the decoder component to three branches to capture different order proximity. We linearly combine the adjacency matrix with the attribute similarity matrix as the input of our model, where the attribute similarity matrix is calculated by the cosine similarity between the attributes based on the social homophily. In this way, we preserve the second-order proximity to enhance the robustness of DANRL-ANE model on sparse networks, and deal with the topological and attribute information simultaneously. Moreover, the sigmoid cross-entropy loss function is extended to capture the neighborhood character, so that the first-order proximity is better preserved. We compare our model with the state-of-the-art models on five real-world datasets and two network analysis tasks, i.e., link prediction and node classification. The DANRL-ANE model performs well on various networks, even on sparse networks or networks with isolated nodes given the attribute information is sufficient.

**PDF:** https://arxiv.org/pdf/2104.05234