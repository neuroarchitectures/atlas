# Efficient Vision Transformer YOLOv5 for Accurate and Fast Traffic Sign Detection

> Source: `https://doi.org/https://doi.org/10.3390/electronics13050880`

---

**Authors:** Guang Zeng, Zhizhou Wu, Lipeng Xu, Yunyi Liang

**Published:** 2024

**Concepts:** Computer science, Traffic sign, Algorithm, Object detection, Minimum bounding box

## Abstract

Accurate and fast detection of traffic sign information is vital for autonomous driving systems. However, the YOLOv5 algorithm faces challenges with low accuracy and slow detection when it is used for traffic sign detection. To address these shortcomings, this paper introduces an accurate and fast traffic sign detection algorithm–YOLOv5-Efficient Vision TransFormer(EfficientViT)). The algorithm focuses on improving both the accuracy and speed of the model by replacing the CSPDarknet backbone of the YOLOv5(s) model with the EfficientViT network. Additionally, the algorithm incorporates the Convolutional Block Attention Module(CBAM) attention mechanism to enhance feature layer information extraction and boost the accuracy of the detection algorithm. To mitigate the adverse effects of low-quality labels on gradient generation and enhance the competitiveness of high-quality anchor frames, a superior gradient gain allocation strategy is employed. Furthermore, the strategy introduces the Wise-IoU (WIoU), a dynamic non-monotonic focusing mechanism for bounding box loss, to further enhance the accuracy and speed of the object detection algorithm. The algorithm’s effectiveness is validated through experiments conducted on the 3L-TT100K traffic sign dataset, showcasing a mean average precision (mAP) of 94.1% in traffic sign detection. This mAP surpasses the performance of the YOLOv5(s) algorithm by 4.76% and outperforms the baseline algorithm. Additionally, the algorithm achieves a detection speed of 62.50 frames per second, which is much better than the baseline algorithm.

**DOI:** https://doi.org/https://doi.org/10.3390/electronics13050880
