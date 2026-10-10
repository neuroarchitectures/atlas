# Paper (Aydemir et al. 2025)

> Source: `https://arxiv.org/abs/2507.09217v1`

---

Online Long-term Point Tracking in the Foundation Model Era

Görkay Aydemir

Published: 2025-07-12T09:24:28Z

Categories: cs.CV

Comment: arXiv admin note: substantial text overlap with arXiv:2501.18487

PDF: https://arxiv.org/pdf/2507.09217v1

Abstract
Point tracking aims to identify the same physical point across video frames and serves as a geometry-aware representation of motion. This representation supports a wide range of applications, from robotics to augmented reality, by enabling accurate modeling of dynamic environments. Most existing long-term tracking approaches operate in an offline setting, where future frames are available to refine predictions and recover from occlusions. However, real-world scenarios often demand online predictions: the model must operate causally, using only current and past frames. This constraint is critical in streaming video and embodied AI, where decisions must be made immediately based on past observations. Under such constraints, viewpoint invariance becomes essential. Visual foundation models, trained on diverse large-scale datasets, offer the potential for robust geometric representations. While they lack temporal reasoning on their own, they can be integrated into tracking pipelines to enrich spatial features. In this thesis, we address the problem of long-term point tracking in an online setting, where frames are processed sequentially without access to future information or sliding windows. We begin by evaluating the suitability of visual foundation models for this task and find that they can serve as useful initializations and be integrated into tracking pipelines. However, to enable long-term tracking in an online setting, a dedicated design is still required. In particular, maintaining coherence over time in this causal regime requires memory to propagate appearance and context across frames. To address this, we introduce Track-On, a transformer-based model that treats each tracked point as a query and processes video frames one at a time. Track-On sets a new state of the art across seven public benchmarks, demonstrating the feasibility of long-term tracking without future access.
