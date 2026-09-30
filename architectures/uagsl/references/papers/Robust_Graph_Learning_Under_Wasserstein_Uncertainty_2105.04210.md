# Paper (Zhang et al. 2021)

> Source: `https://arxiv.org/abs/2105.04210`

---

**Robust Graph Learning Under Wasserstein Uncertainty**

Xiang Zhang, Yinfei Xu, Qinghe Liu, Zhicheng Liu, Jian Lu, Qiao Wang

**Abstract**

Graphs are playing a crucial role in different fields since they are powerful tools to unveil intrinsic relationships among signals. In many scenarios, an accurate graph structure representing signals is not available at all and that motivates people to learn a reliable graph structure directly from observed signals. However, in real life, it is inevitable that there exists uncertainty in the observed signals due to noise measurements or limited observability, which causes a reduction in reliability of the learned graph. To this end, we propose a graph learning framework using Wasserstein distributionally robust optimization (WDRO) which handles uncertainty in data by defining an uncertainty set on distributions of the observed data. Specifically, two models are developed, one of which assumes all distributions in uncertainty set are Gaussian distributions and the other one has no prior distributional assumption. Instead of using interior point method directly, we propose two algorithms to solve the corresponding models and show that our algorithms are more time-saving. In addition, we also reformulate both two models into Semi-Definite Programming (SDP), and illustrate that they are intractable in the scenario of large-scale graph. Experiments on both synthetic and real world data are carried out to validate the proposed framework, which show that our scheme can learn a reliable graph in the context of uncertainty.

---

- **arXiv ID**: `2105.04210`
- **Published**: 2021-05-10
- **Categories**: cs.LG, eess.SP
- **PDF**: https://arxiv.org/pdf/2105.04210
