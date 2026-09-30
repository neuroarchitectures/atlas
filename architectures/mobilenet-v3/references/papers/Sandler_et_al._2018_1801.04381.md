# Paper (Sandler et al. 2018)

> Source: `https://arxiv.org/abs/1801.04381v4`

---

MobileNetV2: Inverted Residuals and Linear Bottlenecks

Mark Sandler
Andrew Howard
Menglong Zhu
Andrey Zhmoginov
Liang-Chieh Chen

Published: 2018-01-13T04:46:26Z

Categories: cs.CV

Journal Reference: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510-4520

PDF: https://arxiv.org/pdf/1801.04381v4

Abstract
In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. We also describe efficient ways of applying these mobile models to object detection in a novel framework we call SSDLite. Additionally, we demonstrate how to build mobile semantic segmentation models through a reduced form of DeepLabv3 which we call Mobile DeepLabv3. The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain representational power. We demonstrate that this improves performance and provide an intuition that led to this design. Finally, our approach allows decoupling of the input/output domains from the expressiveness of the transformation, which provides a convenient framework for further analysis. We measure our performance on Imagenet classification, COCO object detection, VOC image segmentation. We evaluate the trade-offs between accuracy, and number of operations measured by multiply-adds (MAdd), as well as the number of parameters
