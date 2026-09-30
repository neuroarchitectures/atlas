# Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective

> Source: `https://arxiv.org/abs/2412.09805`

---

**Authors:** Gu, Ming, Zheng, Zhuonan, Sheng Zhou, Liu, Meihan, Jiawei Chen, Qiao, Tanyu, Liangcheng Li, Jiajun Bu

**Published:** 2024

**Concepts:** Generalization, Dilemma, Smoothness, Mathematical economics, Computer science

## Abstract

Graph Neural Networks (GNNs) have achieved great success but are often considered to be challenged by varying levels of homophily in graphs. Recent \textit{empirical} studies have surprisingly shown that homophilic GNNs can perform well across datasets of different homophily levels with proper hyperparameter tuning, but the underlying theory and effective architectures remain unclear. To advance GNN universality across varying homophily, we theoretically revisit GNN message passing and uncover a novel \textit{smoothness-generalization dilemma}, where increasing hops inevitably enhances smoothness at the cost of generalization. This dilemma hinders learning in high-order homophilic neighborhoods and all heterophilic ones, where generalization is critical due to complex neighborhood class distributions that are sensitive to shifts induced by noise or sparsity. To address this, we introduce the Inceptive Graph Neural Network (IGNN) built on three simple yet effective design principles, which alleviate the dilemma by enabling distinct hop-wise generalization alongside improved overall generalization with adaptive smoothness. Benchmarking against 30 baselines demonstrates IGNN's superiority and reveals notable universality in certain homophilic GNN variants. Our code and datasets are available at \href{https://github.com/galogm/IGNN}{https://github.com/galogm/IGNN}.

**PDF:** https://arxiv.org/pdf/2412.09805