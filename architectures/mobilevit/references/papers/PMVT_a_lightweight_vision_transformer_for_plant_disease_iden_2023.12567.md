# PMVT: a lightweight vision transformer for plant disease identification on mobile devices

> Source: `https://arxiv.org/abs/2023.12567`

---

**Authors:** Guoqiang Li, Yuchao Wang, Qing Qi Zhao, Peiyan Yuan, Baofang Chang

**Published:** 2023

**Concepts:** Computer science, Artificial intelligence, Mobile device, Residual, Encoder

## Abstract

Due to the constraints of agricultural computing resources and the diversity of plant diseases, it is challenging to achieve the desired accuracy rate while keeping the network lightweight. In this paper, we proposed a computationally efficient deep learning architecture based on the mobile vision transformer (MobileViT) for real-time detection of plant diseases, which we called plant-based MobileViT (PMVT). Our proposed model was designed to be highly accurate and low-cost, making it suitable for deployment on mobile devices with limited resources. Specifically, we replaced the convolution block in MobileViT with an inverted residual structure that employs a 7×7 convolution kernel to effectively model long-distance dependencies between different leaves in plant disease images. Furthermore, inspired by the concept of multi-level attention in computer vision tasks, we integrated a convolutional block attention module (CBAM) into the standard ViT encoder. This integration allows the network to effectively avoid irrelevant information and focus on essential features. The PMVT network achieves reduced parameter counts compared to alternative networks on various mobile devices while maintaining high accuracy across different vision tasks. Extensive experiments on multiple agricultural datasets, including wheat, coffee, and rice, demonstrate that the proposed method outperforms the current best lightweight and heavyweight models. On the wheat dataset, PMVT achieves the highest accuracy of 93.6% using approximately 0.98 million (M) parameters. This accuracy is 1.6% higher than that of MobileNetV3. Under the same parameters, PMVT achieved an accuracy of 85.4% on the coffee dataset, surpassing SqueezeNet by 2.3%. Furthermore, out method achieved an accuracy of 93.1% on the rice dataset, surpassing MobileNetV3 by 3.4%. Additionally, we developed a plant disease diagnosis app and successfully used the trained PMVT model to identify plant disease in different scenarios.

**PDF:** https://arxiv.org/pdf/2023.12567