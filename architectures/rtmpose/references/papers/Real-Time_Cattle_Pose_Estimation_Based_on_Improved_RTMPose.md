# Real-Time Cattle Pose Estimation Based on Improved RTMPose

> Source: `https://doi.org/https://doi.org/10.3390/agriculture13101938`

---

**Authors:** Xiaowu Li, Kun Sun, Hongbo Fan, Zihan He

**Published:** 2023

**Concepts:** Block (permutation group theory), Computer science, Pose, Artificial intelligence, Inference

## Abstract

Accurate cattle pose estimation is essential for Precision Livestock Farming (PLF). Computer vision-based, non-contact cattle pose estimation technology can be applied for behaviour recognition and lameness detection. Existing methods still face challenges in achieving fast cattle pose estimation in complex scenarios. In this work, we introduce the FasterNest Block and Depth Block to enhance the performance of cattle pose estimation based on the RTMPose model. First, the accuracy of cattle pose estimation relies on the capture of high-level image features. The FasterNest Block, with its three-branch structure, effectively utilizes high-level feature map information, significantly improving accuracy without a significant decrease in inference speed. Second, large kernel convolutions can increase the computation cost of the model. Therefore, the Depth Block adopts a method based on depthwise separable convolutions to replace large kernel convolutions. This addresses the insensitivity to semantic information while reducing the model’s parameter. Additionally, the SimAM module enhances the model’s spatial learning capabilities without introducing extra parameters. We conducted tests on various datasets, including our collected complex scene dataset (cattle dataset) and the AP-10K public dataset. The results demonstrate that our model achieves the best average accuracy with the lowest model parameters and computational requirements, achieving 82.9% on the cattle test set and 72.0% on the AP-10K test set. Furthermore, in conjunction with the object detection model RTMDet-m, our model reaches a remarkable inference speed of 39FPS on an NVIDIA GTX 2080Ti GPU using the PyTorch framework, making it the fastest among all models. This work provides adequate technical support for fast and accurate cattle pose estimation in complex farm environments.

**DOI:** https://doi.org/https://doi.org/10.3390/agriculture13101938
