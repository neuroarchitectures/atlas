# Paper (Yamada et al. 2022)

> Source: `https://arxiv.org/abs/2204.03934v1`

---

Does Robustness on ImageNet Transfer to Downstream Tasks?

Yutaro Yamada
Mayu Otani

Published: 2022-04-08T08:55:34Z

Categories: cs.CV, cs.LG

Comment: CVPR 2022

PDF: https://arxiv.org/pdf/2204.03934v1

Abstract
As clean ImageNet accuracy nears its ceiling, the research community is increasingly more concerned about robust accuracy under distributional shifts. While a variety of methods have been proposed to robustify neural networks, these techniques often target models trained on ImageNet classification. At the same time, it is a common practice to use ImageNet pretrained backbones for downstream tasks such as object detection, semantic segmentation, and image classification from different domains. This raises a question: Can these robust image classifiers transfer robustness to downstream tasks? For object detection and semantic segmentation, we find that a vanilla Swin Transformer, a variant of Vision Transformer tailored for dense prediction tasks, transfers robustness better than Convolutional Neural Networks that are trained to be robust to the corrupted version of ImageNet. For CIFAR10 classification, we find that models that are robustified for ImageNet do not retain robustness when fully fine-tuned. These findings suggest that current robustification techniques tend to emphasize ImageNet evaluations. Moreover, network architecture is a strong source of robustness when we consider transfer learning.
