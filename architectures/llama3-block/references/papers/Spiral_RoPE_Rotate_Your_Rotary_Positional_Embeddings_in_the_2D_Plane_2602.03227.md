# Paper (Liu et al. 2026)

> Source: `https://arxiv.org/abs/2602.03227`

---

**Spiral RoPE: Rotate Your Rotary Positional Embeddings in the 2D Plane**

Haoyu Liu, Sucheng Ren, Tingyu Zhu, Peng Wang, Cihang Xie, Alan Yuille, Zeyu Zheng, Feng Wang

**Abstract**

Rotary Position Embedding (RoPE) is the de facto positional encoding in large language models due to its ability to encode relative positions and support length extrapolation. When adapted to vision transformers, the standard axial formulation decomposes two-dimensional spatial positions into horizontal and vertical components, implicitly restricting positional encoding to axis-aligned directions. We identify this directional constraint as a fundamental limitation of the standard axial 2D RoPE, which hinders the modeling of oblique spatial relationships that naturally exist in natural images. To overcome this limitation, we propose Spiral RoPE, a simple yet effective extension that enables multi-directional positional encoding by partitioning embedding channels into multiple groups associated with uniformly distributed directions. Each group is rotated according to the projection of the patch position onto its corresponding direction, allowing spatial relationships to be encoded beyond the horizontal and vertical axes. Across a wide range of vision tasks including classification, segmentation, and generation, Spiral RoPE consistently improves performance. Qualitative analysis of attention maps further show that Spiral RoPE exhibits more concentrated activations on semantically relevant objects and better respects local object boundaries, highlighting the importance of multi-directional positional encoding in vision transformers.

---

- **arXiv ID**: `2602.03227`
- **Published**: 2026-02-03
- **Categories**: cs.CV
- **PDF**: https://arxiv.org/pdf/2602.03227
