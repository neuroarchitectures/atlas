# Paper (Ke et al. 2023)

> Source: `https://arxiv.org/abs/2312.02145v2`

---

Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation

Bingxin Ke
Anton Obukhov
Shengyu Huang
Nando Metzger
Rodrigo Caye Daudt
Konrad Schindler

Published: 2023-12-04T18:59:13Z

Categories: cs.CV

Comment: CVPR 2024 camera ready

PDF: https://arxiv.org/pdf/2312.02145v2

Abstract
Monocular depth estimation is a fundamental computer vision task. Recovering 3D depth from a single image is geometrically ill-posed and requires scene understanding, so it is not surprising that the rise of deep learning has led to a breakthrough. The impressive progress of monocular depth estimators has mirrored the growth in model capacity, from relatively modest CNNs to large Transformer architectures. Still, monocular depth estimators tend to struggle when presented with images with unfamiliar content and layout, since their knowledge of the visual world is restricted by the data seen during training, and challenged by zero-shot generalization to new domains. This motivates us to explore whether the extensive priors captured in recent generative diffusion models can enable better, more generalizable depth estimation. We introduce Marigold, a method for affine-invariant monocular depth estimation that is derived from Stable Diffusion and retains its rich prior knowledge. The estimator can be fine-tuned in a couple of days on a single GPU using only synthetic training data. It delivers state-of-the-art performance across a wide range of datasets, including over 20% performance gains in specific cases. Project page: https://marigoldmonodepth.github.io.
