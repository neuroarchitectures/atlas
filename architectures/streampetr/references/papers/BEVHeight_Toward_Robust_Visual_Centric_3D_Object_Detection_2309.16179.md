# BEVHeight++: Toward Robust Visual Centric 3D Object Detection

> Source: `https://arxiv.org/abs/2309.16179`

---

**Authors:** Lei Yang, Tao Tang, Jun Li, Peng Chen, Kun Yuan, Wang, Li, Yi Huang, Xinyu Zhang, Kaicheng Yu

**Published:** 2023

**Concepts:** Computer vision, Artificial intelligence, Leverage (statistics), Computer science, Perception

## Abstract

While most recent autonomous driving system focuses on developing perception methods on ego-vehicle sensors, people tend to overlook an alternative approach to leverage intelligent roadside cameras to extend the perception ability beyond the visual range. We discover that the state-of-the-art vision-centric bird's eye view detection methods have inferior performances on roadside cameras. This is because these methods mainly focus on recovering the depth regarding the camera center, where the depth difference between the car and the ground quickly shrinks while the distance increases. In this paper, we propose a simple yet effective approach, dubbed BEVHeight++, to address this issue. In essence, we regress the height to the ground to achieve a distance-agnostic formulation to ease the optimization process of camera-only perception methods. By incorporating both height and depth encoding techniques, we achieve a more accurate and robust projection from 2D to BEV spaces. On popular 3D detection benchmarks of roadside cameras, our method surpasses all previous vision-centric methods by a significant margin. In terms of the ego-vehicle scenario, our BEVHeight++ possesses superior over depth-only methods. Specifically, it yields a notable improvement of +1.9% NDS and +1.1% mAP over BEVDepth when evaluated on the nuScenes validation set. Moreover, on the nuScenes test set, our method achieves substantial advancements, with an increase of +2.8% NDS and +1.7% mAP, respectively.

**PDF:** https://arxiv.org/pdf/2309.16179