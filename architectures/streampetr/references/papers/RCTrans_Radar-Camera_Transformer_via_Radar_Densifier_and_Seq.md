# RCTrans: Radar-Camera Transformer via Radar Densifier and Sequential Decoder for 3D Object Detection

> Source: `https://doi.org/https://doi.org/10.1609/aaai.v39i5.32535`

---

**Authors:** Yiheng Li, Yang Yang, Zhen Lei

**Published:** 2025

**Concepts:** Radar, Computer science, Transformer, Computer vision, Artificial intelligence

## Abstract

In radar-camera 3D object detection, the radar point clouds are sparse and noisy, which causes difficulties in fusing camera and radar modalities. To solve this, we introduce a novel query-based detection method named Radar-Camera Transformer (RCTrans). Specifically, we first design a Radar Dense Encoder to enrich the sparse valid radar tokens, and then concatenate them with the image tokens. By doing this, we can fully explore the 3D information of each interest region and reduce the interference of empty tokens during the fusing stage. We then design a Pruning Sequential Decoder to predict 3D boxes based on the obtained tokens and random initialized queries. To alleviate the effect of elevation ambiguity in radar point clouds, we gradually locate the position of the object via a sequential fusion structure. It helps to get more precise and flexible correspondences between tokens and queries. A pruning training strategy is adopted in the decoder, which can save much time during inference and inhibit queries from losing their distinctiveness. Extensive experiments on the large-scale nuScenes dataset prove the superiority of our method, and we also achieve new state-of-the-art radar-camera 3D detection results.

**DOI:** https://doi.org/https://doi.org/10.1609/aaai.v39i5.32535
