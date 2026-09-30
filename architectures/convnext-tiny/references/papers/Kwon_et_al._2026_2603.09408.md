# Paper (Kwon et al. 2026)

> Source: `https://arxiv.org/abs/2603.09408v1`

---

Reviving ConvNeXt for Efficient Convolutional Diffusion Models

Taesung Kwon
Lorenzo Bianchi
Lennart Wittke
Felix Watine
Fabio Carrara
Jong Chul Ye
Romann Weber
Vinicius Azevedo

Published: 2026-03-10T09:24:30Z

Categories: cs.CV, cs.AI, cs.LG

Comment: CVPR 2026. Official implementation: https://github.com/star-kwon/FCDM

PDF: https://arxiv.org/pdf/2603.09408v1

Abstract
Recent diffusion models increasingly favor Transformer backbones, motivated by the remarkable scalability of fully attentional architectures. Yet the locality bias, parameter efficiency, and hardware friendliness--the attributes that established ConvNets as the efficient vision backbone--have seen limited exploration in modern generative modeling. Here we introduce the fully convolutional diffusion model (FCDM), a model having a backbone similar to ConvNeXt, but designed for conditional diffusion modeling. We find that using only 50% of the FLOPs of DiT-XL/2, FCDM-XL achieves competitive performance with 7$\times$ and 7.5$\times$ fewer training steps at 256$\times$256 and 512$\times$512 resolutions, respectively. Remarkably, FCDM-XL can be trained on a 4-GPU system, highlighting the exceptional training efficiency of our architecture. Our results demonstrate that modern convolutional designs provide a competitive and highly efficient alternative for scaling diffusion models, reviving ConvNeXt as a simple yet powerful building block for efficient generative modeling.
