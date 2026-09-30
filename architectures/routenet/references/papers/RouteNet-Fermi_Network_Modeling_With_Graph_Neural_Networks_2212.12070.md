# RouteNet-Fermi: Network Modeling With Graph Neural Networks

> Source: `https://arxiv.org/abs/2212.12070`

---

**Authors:** Miquel Ferriol-Galmés, Jordi Paillissé, José Suárez‐Varela, Krzysztof Rusek, Shihan Xiao, Xiang Shi, Xiangle Cheng, Pere Barlet‐Ros, Albert Cabellos‐Aparicio

**Published:** 2023

**Concepts:** Computer science, Testbed, Network packet, Queueing theory, Computer network

## Abstract

Network models are an essential block of modern networks. For example, they are widely used in network planning and optimization. However, as networks increase in scale and complexity, some models present limitations, such as the assumption of Markovian traffic in queuing theory models, or the high computational cost of network simulators. Recent advances in machine learning, such as Graph Neural Networks (GNN), are enabling a new generation of network models that are data-driven and can learn complex non-linear behaviors. In this paper, we present RouteNet-Fermi, a custom GNN model that shares the same goals as Queuing Theory, while being considerably more accurate in the presence of realistic traffic models. The proposed model predicts accurately the delay, jitter, and packet loss of a network. We have tested RouteNet-Fermi in networks of increasing size (up to 300 nodes), including samples with mixed traffic profiles — e.g., with complex non-Markovian models — and arbitrary routing and queue scheduling configurations. Our experimental results show that RouteNet-Fermi achieves similar accuracy as computationally-expensive packet-level simulators and scales accurately to larger networks. Our model produces delay estimates with a mean relative error of 6.24% when applied to a test dataset of 1,000 samples, including network topologies one order of magnitude larger than those seen during training. Finally, we have also evaluated RouteNet-Fermi with measurements from a physical testbed and packet traces from a real-life network.

**PDF:** https://arxiv.org/pdf/2212.12070