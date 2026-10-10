# Paper (Steiner et al. 2021)

> Source: `https://arxiv.org/abs/2106.10270v2`

---

How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers

Andreas Steiner
Alexander Kolesnikov
Xiaohua Zhai
Ross Wightman
Jakob Uszkoreit
Lucas Beyer

Published: 2021-06-18T17:58:20Z

Categories: cs.CV, cs.AI, cs.LG

Journal Reference: Transactions on Machine Learning Research (05/2022)

Comment: Andreas, Alex, Xiaohua and Lucas contributed equally. We release more than 50'000 ViT models trained under diverse settings on various datasets. Available at https://github.com/google-research/big_vision, https://github.com/google-research/vision_transformer and https://github.com/rwightman/pytorch-image-models TMLR review at https://openreview.net/forum?id=4nPswr1KcP

PDF: https://arxiv.org/pdf/2106.10270v2

Abstract
Vision Transformers (ViT) have been shown to attain highly competitive performance for a wide range of vision applications, such as image classification, object detection and semantic image segmentation. In comparison to convolutional neural networks, the Vision Transformer's weaker inductive bias is generally found to cause an increased reliance on model regularization or data augmentation ("AugReg" for short) when training on smaller training datasets. We conduct a systematic empirical study in order to better understand the interplay between the amount of training data, AugReg, model size and compute budget. As one result of this study we find that the combination of increased compute and AugReg can yield models with the same performance as models trained on an order of magnitude more training data: we train ViT models of various sizes on the public ImageNet-21k dataset which either match or outperform their counterparts trained on the larger, but not publicly available JFT-300M dataset.
