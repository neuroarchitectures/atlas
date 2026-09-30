# Paper (Yu et al. 2023)

> Source: `https://arxiv.org/abs/2303.16900v3`

---

InceptionNeXt: When Inception Meets ConvNeXt

Weihao Yu
Pan Zhou
Shuicheng Yan
Xinchao Wang

Published: 2023-03-29T17:59:58Z

Categories: cs.CV, cs.AI, cs.LG

Comment: CVPR 2024. Code: https://github.com/sail-sg/inceptionnext

PDF: https://arxiv.org/pdf/2303.16900v3

Abstract
Inspired by the long-range modeling ability of ViTs, large-kernel convolutions are widely studied and adopted recently to enlarge the receptive field and improve model performance, like the remarkable work ConvNeXt which employs 7x7 depthwise convolution. Although such depthwise operator only consumes a few FLOPs, it largely harms the model efficiency on powerful computing devices due to the high memory access costs. For example, ConvNeXt-T has similar FLOPs with ResNet-50 but only achieves ~60% throughputs when trained on A100 GPUs with full precision. Although reducing the kernel size of ConvNeXt can improve speed, it results in significant performance degradation, which poses a challenging problem: How to speed up large-kernel-based CNN models while preserving their performance. To tackle this issue, inspired by Inceptions, we propose to decompose large-kernel depthwise convolution into four parallel branches along channel dimension, i.e., small square kernel, two orthogonal band kernels, and an identity mapping. With this new Inception depthwise convolution, we build a series of networks, namely IncepitonNeXt, which not only enjoy high throughputs but also maintain competitive performance. For instance, InceptionNeXt-T achieves 1.6x higher training throughputs than ConvNeX-T, as well as attains 0.2% top-1 accuracy improvement on ImageNet-1K. We anticipate InceptionNeXt can serve as an economical baseline for future architecture design to reduce carbon footprint. Code is available at https://github.com/sail-sg/inceptionnext.
