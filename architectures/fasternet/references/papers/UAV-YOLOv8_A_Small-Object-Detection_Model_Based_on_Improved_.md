# UAV-YOLOv8: A Small-Object-Detection Model Based on Improved YOLOv8 for UAV Aerial Photography Scenarios

> Source: `https://doi.org/https://doi.org/10.3390/s23167190`

---

**Authors:** Gang Wang, Yanfei Chen, Pei Wen An, Hanyu Hong, Jinghu Hu, Tiange Huang

**Published:** 2023

**Concepts:** Computer science, Object detection, Block (permutation group theory), Feature (linguistics), Artificial intelligence

## Abstract

Unmanned aerial vehicle (UAV) object detection plays a crucial role in civil, commercial, and military domains. However, the high proportion of small objects in UAV images and the limited platform resources lead to the low accuracy of most of the existing detection models embedded in UAVs, and it is difficult to strike a good balance between detection performance and resource consumption. To alleviate the above problems, we optimize YOLOv8 and propose an object detection model based on UAV aerial photography scenarios, called UAV-YOLOv8. Firstly, Wise-IoU (WIoU) v3 is used as a bounding box regression loss, and a wise gradient allocation strategy makes the model focus more on common-quality samples, thus improving the localization ability of the model. Secondly, an attention mechanism called BiFormer is introduced to optimize the backbone network, which improves the model's attention to critical information. Finally, we design a feature processing module named Focal FasterNet block (FFNB) and propose two new detection scales based on this module, which makes the shallow features and deep features fully integrated. The proposed multiscale feature fusion network substantially increased the detection performance of the model and reduces the missed detection rate of small objects. The experimental results show that our model has fewer parameters compared to the baseline model and has a mean detection accuracy higher than the baseline model by 7.7%. Compared with other mainstream models, the overall performance of our model is much better. The proposed method effectively improves the ability to detect small objects. There is room to optimize the detection effectiveness of our model for small and feature-less objects (such as bicycle-type vehicles), as we will address in subsequent research.

**DOI:** https://doi.org/https://doi.org/10.3390/s23167190
