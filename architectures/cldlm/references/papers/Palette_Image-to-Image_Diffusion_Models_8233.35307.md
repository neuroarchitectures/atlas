# Palette: Image-to-Image Diffusion Models

> Source: `https://arxiv.org/abs/8233.35307`

---

**Authors:** Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan B. Ho, Tim Salimans, David J. Fleet, Mohammad Norouzi

**Published:** 2022

**Concepts:** Computer science, Inpainting, Artificial intelligence, Image (mathematics), Task (project management)

## Abstract

This paper develops a unified framework for image-to-image translation based on conditional diffusion models and evaluates this framework on four challenging image-to-image translation tasks, namely colorization, inpainting, uncropping, and JPEG restoration. Our simple implementation of image-to-image diffusion models outperforms strong GAN and regression baselines on all tasks, without task-specific hyper-parameter tuning, architecture customization, or any auxiliary loss or sophisticated new techniques needed. We uncover the impact of an L2 vs. L1 loss in the denoising diffusion objective on sample diversity, and demonstrate the importance of self-attention in the neural architecture through empirical studies. Importantly, we advocate a unified evaluation protocol based on ImageNet, with human evaluation and sample quality scores (FID, Inception Score, Classification Accuracy of a pre-trained ResNet-50, and Perceptual Distance against original images). We expect this standardized evaluation protocol to play a role in advancing image-to-image translation research. Finally, we show that a generalist, multi-task diffusion model performs as well or better than task-specific specialist counterparts. Check out https://diffusion-palette.github.io/ for an overview of the results and code.

**PDF:** https://arxiv.org/pdf/8233.35307