# Paper (Ranftl et al. 2019)

> Source: `https://arxiv.org/abs/1907.01341v3`

---

Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer

René Ranftl
Katrin Lasinger
David Hafner
Konrad Schindler
Vladlen Koltun

Published: 2019-07-02T13:16:52Z

Categories: cs.CV

Comment: To appear in TPAMI (accepted August 2020)

PDF: https://arxiv.org/pdf/1907.01341v3

Abstract
The success of monocular depth estimation relies on large and diverse training sets. Due to the challenges associated with acquiring dense ground-truth depth across different environments at scale, a number of datasets with distinct characteristics and biases have emerged. We develop tools that enable mixing multiple datasets during training, even if their annotations are incompatible. In particular, we propose a robust training objective that is invariant to changes in depth range and scale, advocate the use of principled multi-objective learning to combine data from different sources, and highlight the importance of pretraining encoders on auxiliary tasks. Armed with these tools, we experiment with five diverse training datasets, including a new, massive data source: 3D films. To demonstrate the generalization power of our approach we use zero-shot cross-dataset transfer}, i.e. we evaluate on datasets that were not seen during training. The experiments confirm that mixing data from complementary sources greatly improves monocular depth estimation. Our approach clearly outperforms competing methods across diverse datasets, setting a new state of the art for monocular depth estimation. Some results are shown in the supplementary video at https://youtu.be/D46FzVyL9I8
