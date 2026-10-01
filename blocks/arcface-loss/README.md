# ArcFace Loss

## Design Philosophy

Additive angular margin loss that enforces inter-class discrimination and intra-class compactness by adding a margin penalty in the angular space of the cosine similarity.

## Functionality

L = -log(exp(s * cos(theta_y + m)) / (exp(s * cos(theta_y + m)) + sum_{j!=y} exp(s * cos(theta_j)))). The margin m is added to the angle of the target class in the arc-cosine space.

## Used By

ArcFace (face recognition) | Search engines | Person re-identification

## Features

- **Angular margin**: Margin in the angular space, geometrically motivated.
- **No hyperplane collapse**: Unlike softmax, doesn't suffer from weight norm dominance.
- **Combined with feature normalization**: Both weights and features are L2-normalized.

## Evolution

Predecessor: Center loss, CosFace. Successor: MagFace, AdaFace.
