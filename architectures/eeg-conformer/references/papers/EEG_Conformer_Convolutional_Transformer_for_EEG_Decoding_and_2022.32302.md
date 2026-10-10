# EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization

> Source: `https://arxiv.org/abs/2022.32302`

---

**Authors:** Yonghao Song, Qingqing Zheng, Bingchuan Liu, Xiaorong Gao

**Published:** 2022

**Concepts:** Electroencephalography, Decoding methods, Transformer, Computer science, Visualization

## Abstract

Due to the limited perceptual field, convolutional neural networks (CNN) only extract local temporal features and may fail to capture long-term dependencies for EEG decoding. In this paper, we propose a compact Convolutional Transformer, named EEG Conformer, to encapsulate local and global features in a unified EEG classification framework. Specifically, the convolution module learns the low-level local features throughout the one-dimensional temporal and spatial convolution layers. The self-attention module is straightforwardly connected to extract the global correlation within the local temporal features. Subsequently, the simple classifier module based on fully-connected layers is followed to predict the categories for EEG signals. To enhance interpretability, we also devise a visualization strategy to project the class activation mapping onto the brain topography. Finally, we have conducted extensive experiments to evaluate our method on three public datasets in EEG-based motor imagery and emotion recognition paradigms. The experimental results show that our method achieves state-of-the-art performance and has great potential to be a new baseline for general EEG decoding. The code has been released in https://github.com/eeyhsong/EEG-Conformer.

**PDF:** https://arxiv.org/pdf/2022.32302