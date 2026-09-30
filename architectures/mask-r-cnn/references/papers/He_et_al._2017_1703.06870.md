# Paper (He et al. 2017)

> Source: `https://arxiv.org/abs/1703.06870v3`

---

Mask R-CNN

Kaiming He
Georgia Gkioxari
Piotr Dollár
Ross Girshick

Published: 2017-03-20T17:53:38Z

Categories: cs.CV

Comment: open source; appendix on more results

PDF: https://arxiv.org/pdf/1703.06870v3

Abstract
We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition. Code has been made available at: https://github.com/facebookresearch/Detectron
