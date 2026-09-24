# ByteTrack

## Overview

Tracking-by-detection throws away detection boxes below a score threshold — but a low score often means *occlusion*, not absence. ByteTrack associates **every** detection box in two rounds: high-score boxes first, then low-score boxes, which recover occluded objects and remove background false positives. Combined with a YOLOX-class detector it reached 80.3 MOTA / 77.3 IDF1 on MOT17 at 30 FPS.

- **Year:** 2021
- **Authors:** Zhang et al.
- **Source:** arXiv:2110.06864 — *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*
- **Category:** DL/Tracking

## Key Characteristics

- **Associate every box** — the central claim: discarding low-score detections causes irreversible missing objects and fragmented trajectories.
- **Two-stage association** — high-score boxes to tracklets first (IoU + Kalman), then low-score boxes to the *unmatched* tracklets, recovering occluded objects.
- **Detector-agnostic** — the association method is a wrapper: applied to 9 different trackers it improved IDF1 by 1–10 points.
- **No appearance model** — pure motion + IoU association, which is why it runs at 30 FPS; the cost is weaker re-identification after long occlusion.
- **Real-time** — 80.3 MOTA, 77.3 IDF1, 63.1 HOTA on MOT17 at 30 FPS on one V100.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Zhang_et_al._2021_2110.06864.md`](references/papers/Zhang_et_al._2021_2110.06864.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (SORT / DeepSORT → ByteTrack; siblings: OC-SORT, BoT-SORT, StrongSORT; successors: ByteTrackv2, hybrid motion+appearance trackers).
