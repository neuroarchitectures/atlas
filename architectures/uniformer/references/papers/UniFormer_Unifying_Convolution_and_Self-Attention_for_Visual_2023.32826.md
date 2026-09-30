# UniFormer: Unifying Convolution and Self-Attention for Visual Recognition

> Source: `https://arxiv.org/abs/2023.32826`

---

**Authors:** Kunchang Li, Yali Wang, Junhao Zhang, Peng Gao, Guanglu Song, Yu Liu, Hongsheng Li, Yu Qiao

**Published:** 2023

**Concepts:** Computer science, Convolution (computer science), Artificial intelligence, Pattern recognition (psychology), Computer vision

## Abstract

It is a challenging task to learn discriminative representation from images and videos, due to large local redundancy and complex global dependency in these visual data. Convolution neural networks (CNNs) and vision transformers (ViTs) have been two dominant frameworks in the past few years. Though CNNs can efficiently decrease local redundancy by convolution within a small neighborhood, the limited receptive field makes it hard to capture global dependency. Alternatively, ViTs can effectively capture long-range dependency via self-attention, while blind similarity comparisons among all the tokens lead to high redundancy. To resolve these problems, we propose a novel Unified transFormer (UniFormer), which can seamlessly integrate the merits of convolution and self-attention in a concise transformer format. Different from the typical transformer blocks, the relation aggregators in our UniFormer block are equipped with local and global token affinity respectively in shallow and deep layers, allowing tackling both redundancy and dependency for efficient and effective representation learning. Finally, we flexibly stack our blocks into a new powerful backbone, and adopt it for various vision tasks from image to video domain, from classification to dense prediction. Without any extra training data, our UniFormer achieves 86.3 top-1 accuracy on ImageNet-1 K classification task. With only ImageNet-1 K pre-training, it can simply achieve state-of-the-art performance in a broad range of downstream tasks. It obtains 82.9/84.8 top-1 accuracy on Kinetics-400/600, 60.9/71.2 top-1 accuracy on Something-Something V1/V2 video classification tasks, 53.8 box AP and 46.4 mask AP on COCO object detection task, 50.8 mIoU on ADE20 K semantic segmentation task, and 77.4 AP on COCO pose estimation task. Moreover, we build an efficient UniFormer with a concise hourglass design of token shrinking and recovering, which achieves 2-4[Formula: see text] higher throughput than the recent lightweight models.

**PDF:** https://arxiv.org/pdf/2023.32826