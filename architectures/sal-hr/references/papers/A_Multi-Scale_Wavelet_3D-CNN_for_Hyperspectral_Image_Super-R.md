# A Multi-Scale Wavelet 3D-CNN for Hyperspectral Image Super-Resolution

> Source: `https://doi.org/https://doi.org/10.3390/rs11131557`

---

**Authors:** Jingxiang Yang, Yongqiang Zhao, Jonathan Cheung-Wai Chan, Liang Xiao

**Published:** 2019

**Concepts:** Wavelet, Artificial intelligence, Subnet, Computer science, Pattern recognition (psychology)

## Abstract

Super-resolution (SR) is significant for hyperspectral image (HSI) applications. In single-frame HSI SR, how to reconstruct detailed image structures in high resolution (HR) HSI is challenging since there is no auxiliary image (e.g., HR multispectral image) providing structural information. Wavelet could capture image structures in different orientations, and emphasis on predicting high-frequency wavelet sub-bands is helpful for recovering the detailed structures in HSI SR. In this study, we propose a multi-scale wavelet 3D convolutional neural network (MW-3D-CNN) for HSI SR, which predicts the wavelet coefficients of HR HSI rather than directly reconstructing the HR HSI. To exploit the correlation in the spectral and spatial domains, the MW-3D-CNN is built with 3D convolutional layers. An embedding subnet and a predicting subnet constitute the MW-3D-CNN, the embedding subnet extracts deep spatial-spectral features from the low resolution (LR) HSI and represents the LR HSI as a set of feature cubes. The feature cubes are then fed to the predicting subnet. There are multiple output branches in the predicting subnet, each of which corresponds to one wavelet sub-band and predicts the wavelet coefficients of HR HSI. The HR HSI can be obtained by applying inverse wavelet transform to the predicted wavelet coefficients. In the training stage, we propose to train the MW-3D-CNN with L1 norm loss, which is more suitable than the conventional L2 norm loss for penalizing the errors in different wavelet sub-bands. Experiments on both simulated and real spaceborne HSI demonstrate that the proposed algorithm is competitive with other state-of-the-art HSI SR methods.

**DOI:** https://doi.org/https://doi.org/10.3390/rs11131557
