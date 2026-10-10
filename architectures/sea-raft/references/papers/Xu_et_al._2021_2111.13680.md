# Paper (Xu et al. 2021)

> Source: `https://arxiv.org/abs/2111.13680v4`

---

GMFlow: Learning Optical Flow via Global Matching

Haofei Xu
Jing Zhang
Jianfei Cai
Hamid Rezatofighi
Dacheng Tao

Published: 2021-11-26T18:59:56Z

Categories: cs.CV

Comment: CVPR 2022, Oral

PDF: https://arxiv.org/pdf/2111.13680v4

Abstract
Learning-based optical flow estimation has been dominated with the pipeline of cost volume with convolutions for flow regression, which is inherently limited to local correlations and thus is hard to address the long-standing challenge of large displacements. To alleviate this, the state-of-the-art framework RAFT gradually improves its prediction quality by using a large number of iterative refinements, achieving remarkable performance but introducing linearly increasing inference time. To enable both high accuracy and efficiency, we completely revamp the dominant flow regression pipeline by reformulating optical flow as a global matching problem, which identifies the correspondences by directly comparing feature similarities. Specifically, we propose a GMFlow framework, which consists of three main components: a customized Transformer for feature enhancement, a correlation and softmax layer for global feature matching, and a self-attention layer for flow propagation. We further introduce a refinement step that reuses GMFlow at higher feature resolution for residual flow prediction. Our new framework outperforms 31-refinements RAFT on the challenging Sintel benchmark, while using only one refinement and running faster, suggesting a new paradigm for accurate and efficient optical flow estimation. Code is available at https://github.com/haofeixu/gmflow.
