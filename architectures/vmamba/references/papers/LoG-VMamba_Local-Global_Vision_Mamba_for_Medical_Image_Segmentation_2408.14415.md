# Paper (Dang et al. 2024)

> Source: `https://arxiv.org/abs/2408.14415`

---

**LoG-VMamba: Local-Global Vision Mamba for Medical Image Segmentation**

Trung Dinh Quoc Dang, Huy Hoang Nguyen, Aleksei Tiulpin

**Abstract**

Mamba, a State Space Model (SSM), has recently shown competitive performance to Convolutional Neural Networks (CNNs) and Transformers in Natural Language Processing and general sequence modeling. Various attempts have been made to adapt Mamba to Computer Vision tasks, including medical image segmentation (MIS). Vision Mamba (VM)-based networks are particularly attractive due to their ability to achieve global receptive fields, similar to Vision Transformers, while also maintaining linear complexity in the number of tokens. However, the existing VM models still struggle to maintain both spatially local and global dependencies of tokens in high dimensional arrays due to their sequential nature. Employing multiple and/or complicated scanning strategies is computationally costly, which hinders applications of SSMs to high-dimensional 2D and 3D images that are common in MIS problems. In this work, we propose Local-Global Vision Mamba, LoG-VMamba, that explicitly enforces spatially adjacent tokens to remain nearby on the channel axis, and retains the global context in a compressed form. Our method allows the SSMs to access the local and global contexts even before reaching the last token while requiring only a simple scanning strategy. Our segmentation models are computationally efficient and substantially outperform both CNN and Transformers-based baselines on a diverse set of 2D and 3D MIS tasks. The implementation of LoG-VMamba is available at \url{https://github.com/Oulu-IMEDS/LoG-VMamba}.

---

- **arXiv ID**: `2408.14415`
- **Published**: 2024-08-26
- **Categories**: cs.CV, cs.LG
- **PDF**: https://arxiv.org/pdf/2408.14415
