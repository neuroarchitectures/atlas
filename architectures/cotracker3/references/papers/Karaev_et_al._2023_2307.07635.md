# Paper (Karaev et al. 2023)

> Source: `https://arxiv.org/abs/2307.07635v3`

---

CoTracker: It is Better to Track Together

Nikita Karaev
Ignacio Rocco
Benjamin Graham
Natalia Neverova
Andrea Vedaldi
Christian Rupprecht

Published: 2023-07-14T21:13:04Z

Categories: cs.CV

Comment: Code and model weights are available at: https://co-tracker.github.io/

PDF: https://arxiv.org/pdf/2307.07635v3

Abstract
We introduce CoTracker, a transformer-based model that tracks a large number of 2D points in long video sequences. Differently from most existing approaches that track points independently, CoTracker tracks them jointly, accounting for their dependencies. We show that joint tracking significantly improves tracking accuracy and robustness, and allows CoTracker to track occluded points and points outside of the camera view. We also introduce several innovations for this class of trackers, including using token proxies that significantly improve memory efficiency and allow CoTracker to track 70k points jointly and simultaneously at inference on a single GPU. CoTracker is an online algorithm that operates causally on short windows. However, it is trained utilizing unrolled windows as a recurrent network, maintaining tracks for long periods of time even when points are occluded or leave the field of view. Quantitatively, CoTracker substantially outperforms prior trackers on standard point-tracking benchmarks.
