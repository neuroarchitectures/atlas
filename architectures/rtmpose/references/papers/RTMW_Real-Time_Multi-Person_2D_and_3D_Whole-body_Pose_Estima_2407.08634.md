# RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation

> Source: `https://arxiv.org/abs/2407.08634`

---

**Authors:** Tao Jiang, Xinchen Xie, Yining Li

**Published:** 2024

**Concepts:** Pose, Estimation, Computer science, Artificial intelligence, Computer vision

## Abstract

Whole-body pose estimation is a challenging task that requires simultaneous prediction of keypoints for the body, hands, face, and feet. Whole-body pose estimation aims to predict fine-grained pose information for the human body, including the face, torso, hands, and feet, which plays an important role in the study of human-centric perception and generation and in various applications. In this work, we present RTMW (Real-Time Multi-person Whole-body pose estimation models), a series of high-performance models for 2D/3D whole-body pose estimation. We incorporate RTMPose model architecture with FPN and HEM (Hierarchical Encoding Module) to better capture pose information from different body parts with various scales. The model is trained with a rich collection of open-source human keypoint datasets with manually aligned annotations and further enhanced via a two-stage distillation strategy. RTMW demonstrates strong performance on multiple whole-body pose estimation benchmarks while maintaining high inference efficiency and deployment friendliness. We release three sizes: m/l/x, with RTMW-l achieving a 70.2 mAP on the COCO-Wholebody benchmark, making it the first open-source model to exceed 70 mAP on this benchmark. Meanwhile, we explored the performance of RTMW in the task of 3D whole-body pose estimation, conducting image-based monocular 3D whole-body pose estimation in a coordinate classification manner. We hope this work can benefit both academic research and industrial applications. The code and models have been made publicly available at: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose

**PDF:** https://arxiv.org/pdf/2407.08634