# Stochastic Graph Neural Networks

> Source: `https://arxiv.org/abs/2006.02684`

---

**Authors:** Zhan Gao, Elvin Isufi, Alejandro Ribeiro

**Published:** 2021

**Concepts:** Computer science, Randomness, Graph, Artificial neural network, Variance (accounting)

## Abstract

Graph neural networks (GNNs) model nonlinear representations in graph data with applications in distributed agent coordination, control, and planning among others. Current GNN architectures assume ideal scenarios and ignore link fluctuations that occur due to environment, human factors, or external attacks. In these situations, the GNN fails to address its distributed task if the topological randomness is not considered accordingly. To overcome this issue, we put forth the stochastic graph neural network (SGNN) model: a GNN where the distributed graph convolution module accounts for the random network changes. Since stochasticity brings in a new learning paradigm, we conduct a statistical analysis on the SGNN output variance to identify conditions the learned filters should satisfy for achieving robust transference to perturbed scenarios, ultimately revealing the explicit impact of random link losses. We further develop a stochastic gradient descent (SGD) based learning process for the SGNN and derive conditions on the learning rate under which this learning process converges to a stationary point. Numerical results corroborate our theoretical findings and compare the benefits of SGNN robust transference with a conventional GNN that ignores graph perturbations during learning.

**PDF:** https://arxiv.org/pdf/2006.02684