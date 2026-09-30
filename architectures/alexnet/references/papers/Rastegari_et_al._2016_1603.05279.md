# Paper (Rastegari et al. 2016)

> Source: `https://arxiv.org/abs/1603.05279v4`

---

XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks

Mohammad Rastegari
Vicente Ordonez
Joseph Redmon
Ali Farhadi

Published: 2016-03-16T20:56:21Z

Categories: cs.CV

PDF: https://arxiv.org/pdf/1603.05279v4

Abstract
We propose two efficient approximations to standard convolutional neural networks: Binary-Weight-Networks and XNOR-Networks. In Binary-Weight-Networks, the filters are approximated with binary values resulting in 32x memory saving. In XNOR-Networks, both the filters and the input to convolutional layers are binary. XNOR-Networks approximate convolutions using primarily binary operations. This results in 58x faster convolutional operations and 32x memory savings. XNOR-Nets offer the possibility of running state-of-the-art networks on CPUs (rather than GPUs) in real-time. Our binary networks are simple, accurate, efficient, and work on challenging visual tasks. We evaluate our approach on the ImageNet classification task. The classification accuracy with a Binary-Weight-Network version of AlexNet is only 2.9% less than the full-precision AlexNet (in top-1 measure). We compare our method with recent network binarization methods, BinaryConnect and BinaryNets, and outperform these methods by large margins on ImageNet, more than 16% in top-1 accuracy.
