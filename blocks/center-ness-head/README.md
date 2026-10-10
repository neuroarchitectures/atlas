# Center-Ness Head (FCOS)

## Design Philosophy

A fully convolutional, anchor-free detector predicts a box at every foreground location, which produces many low-quality boxes far from the object centre. A cheap auxiliary branch that scores *how centered* a location is suppresses these outliers without a separate objectness head.

## Functionality

For each location `(x, y)` inside a predicted box, define

```
center-ness = sqrt( min(l*, r*) / max(l*, r*) * min(t*, b*) / max(t*, b*) )
```

where `l*, r*, t*, b*` are the distances to the box's four sides. The branch is a small parallel conv (one channel) trained with BCE; at inference the classification score is multiplied by the center-ness score before NMS.

## Used By

FCOS

## Features

- **Down-weights off-center boxes**: score decays smoothly toward the box edges.
- **Anchor-free**: no priors, no IoU-based assignment.
- **One extra channel**: negligible cost alongside cls/reg heads.
- **NMS-friendly**: sharpens the score ranking, so standard NMS works.

## Evolution

Predecessor: anchor-based detection heads (RPN, SSD). Sibling: objectness branches and quality-aware query selection in DETR-family detectors. Successor: center sampling / center-prior assignment in later anchor-free detectors.
