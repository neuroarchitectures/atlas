# Paper (Wang et al. 2025)

> Source: `https://arxiv.org/abs/2503.11651v1`

---

VGGT: Visual Geometry Grounded Transformer

Jianyuan Wang
Minghao Chen
Nikita Karaev
Andrea Vedaldi
Christian Rupprecht
David Novotny

Published: 2025-03-14T17:59:47Z

Categories: cs.CV

Comment: CVPR 2025, Project Page: https://vgg-t.github.io/

PDF: https://arxiv.org/pdf/2503.11651v1

Abstract
We present VGGT, a feed-forward neural network that directly infers all key 3D attributes of a scene, including camera parameters, point maps, depth maps, and 3D point tracks, from one, a few, or hundreds of its views. This approach is a step forward in 3D computer vision, where models have typically been constrained to and specialized for single tasks. It is also simple and efficient, reconstructing images in under one second, and still outperforming alternatives that require post-processing with visual geometry optimization techniques. The network achieves state-of-the-art results in multiple 3D tasks, including camera parameter estimation, multi-view depth estimation, dense point cloud reconstruction, and 3D point tracking. We also show that using pretrained VGGT as a feature backbone significantly enhances downstream tasks, such as non-rigid point tracking and feed-forward novel view synthesis. Code and models are publicly available at https://github.com/facebookresearch/vggt.
