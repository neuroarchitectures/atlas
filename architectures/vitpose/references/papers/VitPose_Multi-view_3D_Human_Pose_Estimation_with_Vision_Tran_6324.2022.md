# VitPose: Multi-view 3D Human Pose Estimation with Vision Transformer

> Source: `https://arxiv.org/abs/6324.2022`

---

**Authors:** Hailun Xia, Qiang Zhang

**Published:** 2022

**Concepts:** Artificial intelligence, Computer science, Pose, Transformer, Computer vision

## Abstract

Fusion of features from multi-view is one of the effective means to improve multi-view 3D pose estimation. It relies on robust and accurate 2D pose estimation, and effective fusion methods. If we treat the same fusion weights for predictions with different accuracy. In this case, “Bad-prediction” can have a bad effect on “Well-prediction” during the fusion process. To address these issues, inspired by previous vision transformer work, we propose a transformer framework for multi-view 3D pose estimation named VitPose, aiming to strengthen the mutual constraint of human articulation points by enhancing the large-scale information of image prediction, rather than calibrating after prediction. In addition, we design simple feedback for training fusion weights, which is used to avoid the interference of “Bad-prediction” to “Well-prediction”. We add multi-view geometric calibration that introduces the spatial information of the view into the transform structure, which is used to strengthen the connection between two views. We conducted extensive experiments on Human3.6M, which showed that our approach achieved competitive results. Specifically, We achieve 17.0 mm Mean Per Joint Position Error(MPJPE) on Human3.6M on 384×384 resolution, which is the State-of-The-Art(SOTA) method with vanilla triangulation.

**PDF:** https://arxiv.org/pdf/6324.2022