# VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning

**Authors:** Adrien Bardes, Jean Ponce, Yann LeCun
**arXiv:** 2105.04906 (2021, ICLR 2022)

## Abstract

Self-supervised joint embedding methods based on non-contrastive losses and without negative samples can suffer from representation collapse. In this paper, we propose VICReg (Variance-Invariance-Covariance Regularization), a method that combines a variance term with an invariance term and a covariance term to prevent collapse while maintaining invariance between the representations of two views of an image. The variance term maintains the variance of each embedding dimension above a threshold, the invariance term forces the representations of two views of an image to be similar, and the covariance term decorrelates the embedding dimensions, preventing the collapse of multiple dimensions to the same value. VICReg matches or outperforms state-of-the-art self-supervised methods on ImageNet-1K with standard linear and semi-supervised evaluation protocols, and is competitive on downstream tasks such as object detection and semantic segmentation.
