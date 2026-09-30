# Paper (Li et al. 2018)

> Source: `https://arxiv.org/abs/1804.00607v4`

---

MegaDepth: Learning Single-View Depth Prediction from Internet Photos

Zhengqi Li
Noah Snavely

Published: 2018-04-02T16:03:34Z

Categories: cs.CV

Comment: updated paper for 'MegaDepth: Learning Single-View Depth Prediction from Internet Photos', CVPR, 2018

PDF: https://arxiv.org/pdf/1804.00607v4

Abstract
Single-view depth prediction is a fundamental problem in computer vision. Recently, deep learning methods have led to significant progress, but such methods are limited by the available training data. Current datasets based on 3D sensors have key limitations, including indoor-only images (NYU), small numbers of training examples (Make3D), and sparse sampling (KITTI). We propose to use multi-view Internet photo collections, a virtually unlimited data source, to generate training data via modern structure-from-motion and multi-view stereo (MVS) methods, and present a large depth dataset called MegaDepth based on this idea. Data derived from MVS comes with its own challenges, including noise and unreconstructable objects. We address these challenges with new data cleaning methods, as well as automatically augmenting our data with ordinal depth relations generated using semantic segmentation. We validate the use of large amounts of Internet data by showing that models trained on MegaDepth exhibit strong generalization-not only to novel scenes, but also to other diverse datasets including Make3D, KITTI, and DIW, even when no images from those datasets are seen during training.
