# Paper (Mao et al. 2023)

> Source: `https://arxiv.org/abs/2305.09523v1`

---

SCTracker: Multi-object tracking with shape and confidence constraints

Huan Mao
Yulin Chen
Zongtan Li
Feng Chen
Pingping Chen

Published: 2023-05-16T15:18:42Z

Categories: cs.CV

PDF: https://arxiv.org/pdf/2305.09523v1

Abstract
Detection-based tracking is one of the main methods of multi-object tracking. It can obtain good tracking results when using excellent detectors but it may associate wrong targets when facing overlapping and low-confidence detections. To address this issue, this paper proposes a multi-object tracker based on shape constraint and confidence named SCTracker. In the data association stage, an Intersection of Union distance with shape constraints is applied to calculate the cost matrix between tracks and detections, which can effectively avoid the track tracking to the wrong target with the similar position but inconsistent shape, so as to improve the accuracy of data association. Additionally, the Kalman Filter based on the detection confidence is used to update the motion state to improve the tracking performance when the detection has low confidence. Experimental results on MOT 17 dataset show that the proposed method can effectively improve the tracking performance of multi-object tracking.
