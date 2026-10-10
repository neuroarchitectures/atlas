# Paper (Xiong et al. 2023)

> Source: `https://arxiv.org/abs/2312.00863v1`

---

EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything

Yunyang Xiong
Bala Varadarajan
Lemeng Wu
Xiaoyu Xiang
Fanyi Xiao
Chenchen Zhu
Xiaoliang Dai
Dilin Wang
Fei Sun
Forrest Iandola
Raghuraman Krishnamoorthi
Vikas Chandra

Published: 2023-12-01T18:31:00Z

Categories: cs.CV

PDF: https://arxiv.org/pdf/2312.00863v1

Abstract
Segment Anything Model (SAM) has emerged as a powerful tool for numerous vision applications. A key component that drives the impressive performance for zero-shot transfer and high versatility is a super large Transformer model trained on the extensive high-quality SA-1B dataset. While beneficial, the huge computation cost of SAM model has limited its applications to wider real-world applications. To address this limitation, we propose EfficientSAMs, light-weight SAM models that exhibits decent performance with largely reduced complexity. Our idea is based on leveraging masked image pretraining, SAMI, which learns to reconstruct features from SAM image encoder for effective visual representation learning. Further, we take SAMI-pretrained light-weight image encoders and mask decoder to build EfficientSAMs, and finetune the models on SA-1B for segment anything task. We perform evaluations on multiple vision tasks including image classification, object detection, instance segmentation, and semantic object detection, and find that our proposed pretraining method, SAMI, consistently outperforms other masked image pretraining methods. On segment anything task such as zero-shot instance segmentation, our EfficientSAMs with SAMI-pretrained lightweight image encoders perform favorably with a significant gain (e.g., ~4 AP on COCO/LVIS) over other fast SAM models.
