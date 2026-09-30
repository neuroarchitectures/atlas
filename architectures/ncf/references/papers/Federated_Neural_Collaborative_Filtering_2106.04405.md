# Paper (Perifanis et al. 2021)

> Source: `https://arxiv.org/abs/2106.04405`

---

**Federated Neural Collaborative Filtering**

Vasileios Perifanis, Pavlos S. Efraimidis

**Abstract**

In this work, we present a federated version of the state-of-the-art Neural Collaborative Filtering (NCF) approach for item recommendations. The system, named FedNCF, enables learning without requiring users to disclose or transmit their raw data. Data localization preserves data privacy and complies with regulations such as the GDPR. Although federated learning enables model training without local data dissemination, the transmission of raw clients' updates raises additional privacy issues. To address this challenge, we incorporate a privacy-preserving aggregation method that satisfies the security requirements against an honest but curious entity. We argue theoretically and experimentally that existing aggregation algorithms are inconsistent with latent factor model updates. We propose an enhancement by decomposing the aggregation step into matrix factorization and neural network-based averaging. Experimental validation shows that FedNCF achieves comparable recommendation quality to the original NCF system, while our proposed aggregation leads to faster convergence compared to existing methods. We investigate the effectiveness of the federated recommender system and evaluate the privacy-preserving mechanism in terms of computational cost.

---

- **arXiv ID**: `2106.04405`
- **Published**: 2021-06-02
- **Categories**: cs.IR, cs.CR, cs.LG
- **PDF**: https://arxiv.org/pdf/2106.04405
