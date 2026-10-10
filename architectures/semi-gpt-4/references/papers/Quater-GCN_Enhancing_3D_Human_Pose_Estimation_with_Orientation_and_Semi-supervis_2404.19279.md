# Paper (Song et al. 2024)

> Source: `https://arxiv.org/abs/2404.19279`

---

**Quater-GCN: Enhancing 3D Human Pose Estimation with Orientation and Semi-supervised Training**

Xingyu Song, Zhan Li, Shi Chen, Kazuyuki Demachi

**Abstract**

3D human pose estimation is a vital task in computer vision, involving the prediction of human joint positions from images or videos to reconstruct a skeleton of a human in three-dimensional space. This technology is pivotal in various fields, including animation, security, human-computer interaction, and automotive safety, where it promotes both technological progress and enhanced human well-being. The advent of deep learning significantly advances the performance of 3D pose estimation by incorporating temporal information for predicting the spatial positions of human joints. However, traditional methods often fall short as they primarily focus on the spatial coordinates of joints and overlook the orientation and rotation of the connecting bones, which are crucial for a comprehensive understanding of human pose in 3D space. To address these limitations, we introduce Quater-GCN (Q-GCN), a directed graph convolutional network tailored to enhance pose estimation by orientation. Q-GCN excels by not only capturing the spatial dependencies among node joints through their coordinates but also integrating the dynamic context of bone rotations in 2D space. This approach enables a more sophisticated representation of human poses by also regressing the orientation of each bone in 3D space, moving beyond mere coordinate prediction. Furthermore, we complement our model with a semi-supervised training strategy that leverages unlabeled data, addressing the challenge of limited orientation ground truth data. Through comprehensive evaluations, Q-GCN has demonstrated outstanding performance against current state-of-the-art methods.

---

- **arXiv ID**: `2404.19279`
- **Published**: 2024-04-30
- **Categories**: cs.CV
- **PDF**: https://arxiv.org/pdf/2404.19279
