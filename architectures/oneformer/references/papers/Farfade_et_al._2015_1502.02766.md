# Paper (Farfade et al. 2015)

> Source: `https://arxiv.org/abs/1502.02766v3`

---

Multi-view Face Detection Using Deep Convolutional Neural Networks

Sachin Sudhakar Farfade
Mohammad Saberian
Li-Jia Li

Published: 2015-02-10T03:15:21Z

Categories: cs.CV

Comment: in International Conference on Multimedia Retrieval 2015 (ICMR)

PDF: https://arxiv.org/pdf/1502.02766v3

Abstract
In this paper we consider the problem of multi-view face detection. While there has been significant research on this problem, current state-of-the-art approaches for this task require annotation of facial landmarks, e.g. TSM [25], or annotation of face poses [28, 22]. They also require training dozens of models to fully capture faces in all orientations, e.g. 22 models in HeadHunter method [22]. In this paper we propose Deep Dense Face Detector (DDFD), a method that does not require pose/landmark annotation and is able to detect faces in a wide range of orientations using a single model based on deep convolutional neural networks. The proposed method has minimal complexity; unlike other recent deep learning object detection methods [9], it does not require additional components such as segmentation, bounding-box regression, or SVM classifiers. Furthermore, we analyzed scores of the proposed face detector for faces in different orientations and found that 1) the proposed method is able to detect faces from different angles and can handle occlusion to some extent, 2) there seems to be a correlation between dis- tribution of positive examples in the training set and scores of the proposed face detector. The latter suggests that the proposed methods performance can be further improved by using better sampling strategies and more sophisticated data augmentation techniques. Evaluations on popular face detection benchmark datasets show that our single-model face detector algorithm has similar or better performance compared to the previous methods, which are more complex and require annotations of either different poses or facial landmarks.
