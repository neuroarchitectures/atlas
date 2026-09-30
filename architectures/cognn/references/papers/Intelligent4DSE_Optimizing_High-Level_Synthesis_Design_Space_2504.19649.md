# Intelligent4DSE: Optimizing High-Level Synthesis Design Space Exploration with Graph Neural Networks and Large Language Models

> Source: `https://arxiv.org/abs/2504.19649`

---

**Authors:** Lei Xu, Shanshan Wang, Emmanuel Casseau, Chenglong Xiao

**Published:** 2025

**Concepts:** Computer science, Artificial neural network, Design space exploration, Set (abstract data type), Artificial intelligence

## Abstract

High-Level Synthesis (HLS) Design Space Exploration (DSE) is essential for generating hardware designs that balance performance, power, and area (PPA). To optimize this process, existing works often employs message-passing neural networks (MPNNs) to predict quality of results (QoR). These predictors serve as evaluators in the DSE process, effectively bypassing the time-consuming estimations traditionally required by HLS tools. However, existing models based on MPNNs struggle with over-smoothing and limited expressiveness. Additionally, while meta-heuristic algorithms are widely used in DSE, they typically require extensive domain-specific knowledge to design operators and time-consuming tuning. To address these limitations, we propose ECoGNNs-LLMMHs, a framework that integrates graph neural networks with task-adaptive message passing and large language model-enhanced meta-heuristic algorithms. Compared with state-of-the-art works, ECoGNN exhibits lower prediction error in the post-HLS prediction task, with the error reduced by 57.27\%. For post-implementation prediction tasks, ECoGNN demonstrates the lowest prediction errors, with average reductions of 17.6\% for flip-flop (FF) usage, 33.7\% for critical path (CP) delay, 26.3\% for power consumption, 38.3\% for digital signal processor (DSP) utilization, and 40.8\% for BRAM usage. LLMMH variants can generate superior Pareto fronts compared to meta-heuristic algorithms in terms of average distance from the reference set (ADRS) with average improvements of 87.47\%, respectively. Compared with the SOTA DSE approaches GNN-DSE and IRONMAN-PRO, LLMMH can reduce the ADRS by 68.17\% and 63.07\% respectively.

**PDF:** https://arxiv.org/pdf/2504.19649