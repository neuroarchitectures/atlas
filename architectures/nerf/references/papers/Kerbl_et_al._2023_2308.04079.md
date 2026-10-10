# Paper (Kerbl et al. 2023)

> Source: `https://arxiv.org/abs/2308.04079v1`

---

3D Gaussian Splatting for Real-Time Radiance Field Rendering

Bernhard Kerbl
Georgios Kopanas
Thomas Leimkühler
George Drettakis

Published: 2023-08-08T06:37:06Z

Categories: cs.GR, cs.CV

Journal Reference: ACM Transactions on Graphics, volume 42(4), July 2023

Comment: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

PDF: https://arxiv.org/pdf/2308.04079v1

Abstract
Radiance Field methods have recently revolutionized novel-view synthesis of scenes captured with multiple photos or videos. However, achieving high visual quality still requires neural networks that are costly to train and render, while recent faster methods inevitably trade off speed for quality. For unbounded and complete scenes (rather than isolated objects) and 1080p resolution rendering, no current method can achieve real-time display rates. We introduce three key elements that allow us to achieve state-of-the-art visual quality while maintaining competitive training times and importantly allow high-quality real-time (>= 30 fps) novel-view synthesis at 1080p resolution. First, starting from sparse points produced during camera calibration, we represent the scene with 3D Gaussians that preserve desirable properties of continuous volumetric radiance fields for scene optimization while avoiding unnecessary computation in empty space; Second, we perform interleaved optimization/density control of the 3D Gaussians, notably optimizing anisotropic covariance to achieve an accurate representation of the scene; Third, we develop a fast visibility-aware rendering algorithm that supports anisotropic splatting and both accelerates training and allows realtime rendering. We demonstrate state-of-the-art visual quality and real-time rendering on several established datasets.
