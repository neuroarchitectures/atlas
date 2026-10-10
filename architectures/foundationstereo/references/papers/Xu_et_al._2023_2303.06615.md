# Paper (Xu et al. 2023)

> Source: `https://arxiv.org/abs/2303.06615v2`

---

Iterative Geometry Encoding Volume for Stereo Matching

Gangwei Xu
Xianqi Wang
Xiaohuan Ding
Xin Yang

Published: 2023-03-12T09:11:14Z

Categories: cs.CV

Comment: Accepted to CVPR 2023

PDF: https://arxiv.org/pdf/2303.06615v2

Abstract
Recurrent All-Pairs Field Transforms (RAFT) has shown great potentials in matching tasks. However, all-pairs correlations lack non-local geometry knowledge and have difficulties tackling local ambiguities in ill-posed regions. In this paper, we propose Iterative Geometry Encoding Volume (IGEV-Stereo), a new deep network architecture for stereo matching. The proposed IGEV-Stereo builds a combined geometry encoding volume that encodes geometry and context information as well as local matching details, and iteratively indexes it to update the disparity map. To speed up the convergence, we exploit GEV to regress an accurate starting point for ConvGRUs iterations. Our IGEV-Stereo ranks $1^{st}$ on KITTI 2015 and 2012 (Reflective) among all published methods and is the fastest among the top 10 methods. In addition, IGEV-Stereo has strong cross-dataset generalization as well as high inference efficiency. We also extend our IGEV to multi-view stereo (MVS), i.e. IGEV-MVS, which achieves competitive accuracy on DTU benchmark. Code is available at https://github.com/gangweiX/IGEV.
