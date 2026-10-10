# Paper (OSheaWheller et al. 2026)

> Source: `https://arxiv.org/abs/2609.37801v1`

---

ByteTraX: Enhancing the ByteTrack Architecture with Optimised Thresholding

Thomas A. O'Shea-Wheller

Published: 2026-09-29T15:23:06Z

Categories: cs.CV

PDF: https://arxiv.org/pdf/2609.37801v1

Abstract
The ByteTrack algorithm is a widely used and computationally efficient multi-object tracking architecture. Its core innovation lies in the combination of lenient bounding box associations with tracklet similarity matching to robustly deal with object occlusions. However, this strategy is nevertheless vulnerable to erroneous track reclassification and identity switching, as detection confidence scores dictate association priority. To address this, I present a simple enhancement of the ByteTrack architecture, named ByteTraX, that optimises track continuity via a single unified matching threshold, while penalising identity switches through stringent track initiation criteria. This approach achieves consistently improved performance across a range of diverse benchmarks including GMOT-40, LC-MOT, SportsMOT, TeamTrack, DAMUNT, and DeepSea-MOT, while simultaneously increasing processing speed by >10%. Specifically, results demonstrate a >40% reduction in identity switches, accompanied by mean increases in HOTA of 3.6, IDF1 of 5.6, and FPS of 6.3. As such, adoption of the ByteTraX algorithm has the potential to substantially enhance tracking performance over the ByteTrack baseline, while retaining the efficiency needed for real-time deployment. To facilitate usage, I provide the source code, integration functionality for the YOLO family of object detection models, and deployment instructions via an open source repository.
