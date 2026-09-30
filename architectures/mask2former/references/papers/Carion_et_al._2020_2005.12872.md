# Paper (Carion et al. 2020)

> Source: `https://arxiv.org/abs/2005.12872v3`

---

End-to-End Object Detection with Transformers

Nicolas Carion
Francisco Massa
Gabriel Synnaeve
Nicolas Usunier
Alexander Kirillov
Sergey Zagoruyko

Published: 2020-05-26T17:06:38Z

Categories: cs.CV

PDF: https://arxiv.org/pdf/2005.12872v3

Abstract
We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster RCNN baseline on the challenging COCO object detection dataset. Moreover, DETR can be easily generalized to produce panoptic segmentation in a unified manner. We show that it significantly outperforms competitive baselines. Training code and pretrained models are available at https://github.com/facebookresearch/detr.
