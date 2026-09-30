# VGGT-Geo: Probabilistic Geometric Fusion of Visual Geometry Grounded Transformer Priors for Robust Dense Indoor SLAM

> Source: `https://doi.org/https://doi.org/10.3390/ijgi15020085`

---

**Authors:** Kai Qin, Jing Li, Sisi Zlatanova, Haitao Wu, Zhigang Wu, Yin Han Gao, Dingjie Zhou, Yuchen Li, Sizhe Shen, Xiangjun Qu et al.

**Published:** 2026

**Concepts:** Artificial intelligence, Prior probability, Computer science, Robustness (evolution), Probabilistic logic

## Abstract

With the rapid evolution of Digital Twins and Embodied AI, achieving fast, dense, and high-precision 3D perception in unknown environments has become paramount. However, existing Visual SLAM paradigms face a critical dilemma: geometry-based methods often fail in texture-less areas due to feature scarcity, while learning-based approaches frequently suffer from scale drift and unphysical deformations. To bridge this gap, we propose VGGT-Geo, a novel SLAM system that synergizes generative priors from Large Foundation Models with multi-modal geometric optimization. Distinguishing itself from simple cascaded architectures, we construct a Probabilistic Geometric Fusion framework, consisting of (1) Generative Warm-start, leveraging the holistic scene understanding capabilities of the VGGT, (2) Confidence-Aware Optimization to extract dense features via DINOv3 and predict their confidence map, and (3) a Multi-Modal Constraint Closure that fuses point-line features and metric depth priors to constrain rotational Degrees of Freedom in Manhattan Worlds. We conducted systematic evaluations on TUM, Replica, Tanks and Temples, and a challenging self-collected dataset featuring extreme lighting and texture-less walls. Experimental results demonstrate that VGGT-Geo exhibits superior robustness and accuracy in unseen environments. On our most challenging dataset, it achieves an Absolute Trajectory Error of 4–5 cm and a Relative Rotation Error of 0.79°, outperforming current state-of-the-art methods by approximately 50% in trajectory accuracy. This study validates that synergizing the intuition of Large Foundation Models with geometric rigor is a viable path toward next-generation robust SLAM.

**DOI:** https://doi.org/https://doi.org/10.3390/ijgi15020085
