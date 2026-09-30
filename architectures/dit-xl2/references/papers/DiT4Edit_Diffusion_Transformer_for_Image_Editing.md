# DiT4Edit: Diffusion Transformer for Image Editing

> Source: `https://doi.org/https://doi.org/10.1609/aaai.v39i3.32304`

---

**Authors:** Kun Feng, Yue Ma, Bingyuan Wang, Chenyang Qi, H.F. Chen, Qifeng Chen, Zeyu Wang

**Published:** 2025

**Concepts:** Transformer, Computer science, Computer graphics (images), Electrical engineering, Engineering

## Abstract

Despite recent advances in UNet-based image editing, methods for shape-aware object editing in high-resolution images are still lacking. Compared to UNet, Diffusion Transformers (DiT) demonstrate superior capabilities to effectively capture the long-range dependencies among patches, leading to higher-quality image generation. In this paper, we propose DiT4Edit, the first Diffusion Transformer-based image editing framework. Specifically, DiT4Edit uses the DPM-Solver inversion algorithm to obtain the inverted latents, reducing the number of steps compared to the DDIM inversion algorithm commonly used in UNet-based frameworks. Additionally, we design unified attention control and patch merging, tailored for transformer computation streams. This integration allows our framework to generate higher-quality edited images faster. Our design leverages the advantages of DiT, enabling it to surpass UNet structures in image editing, especially in high-resolution and arbitrary-size images. Extensive experiments demonstrate the strong performance of DiT4Edit in various editing scenarios, highlighting the potential of diffusion transformers for image editing.

**DOI:** https://doi.org/https://doi.org/10.1609/aaai.v39i3.32304
