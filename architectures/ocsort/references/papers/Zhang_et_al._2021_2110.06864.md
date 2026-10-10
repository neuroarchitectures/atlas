# Paper (Zhang et al. 2021)

> Source: `https://arxiv.org/abs/2110.06864v3`

---

ByteTrack: Multi-Object Tracking by Associating Every Detection Box

Yifu Zhang
Peize Sun
Yi Jiang
Dongdong Yu
Fucheng Weng
Zehuan Yuan
Ping Luo
Wenyu Liu
Xinggang Wang

Published: 2021-10-13T17:01:26Z

Categories: cs.CV

PDF: https://arxiv.org/pdf/2110.06864v3

Abstract
Multi-object tracking (MOT) aims at estimating bounding boxes and identities of objects in videos. Most methods obtain identities by associating detection boxes whose scores are higher than a threshold. The objects with low detection scores, e.g. occluded objects, are simply thrown away, which brings non-negligible true object missing and fragmented trajectories. To solve this problem, we present a simple, effective and generic association method, tracking by associating almost every detection box instead of only the high score ones. For the low score detection boxes, we utilize their similarities with tracklets to recover true objects and filter out the background detections. When applied to 9 different state-of-the-art trackers, our method achieves consistent improvement on IDF1 score ranging from 1 to 10 points. To put forwards the state-of-the-art performance of MOT, we design a simple and strong tracker, named ByteTrack. For the first time, we achieve 80.3 MOTA, 77.3 IDF1 and 63.1 HOTA on the test set of MOT17 with 30 FPS running speed on a single V100 GPU. ByteTrack also achieves state-of-the-art performance on MOT20, HiEve and BDD100K tracking benchmarks. The source code, pre-trained models with deploy versions and tutorials of applying to other trackers are released at https://github.com/ifzhang/ByteTrack.
