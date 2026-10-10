# Center Loss

## Design Philosophy

Cross-entropy learns separable features but not discriminative ones. Center loss adds a penalty for distance from class centers, pulling features toward their class mean.

## Functionality

Maintain a center c_y for each class y. Loss = CE + lambda/2 * ||x - c_y||^2. The centers are updated by the average of features in each class per batch.

## Used By

Face recognition | Person re-identification | Fine-grained classification

## Features

- **Intra-class compactness**: Pulls features toward class centers.
- **Inter-class separation**: Combined with CE for separability.
- **Learnable centers**: Updated during training, not fixed.

## Evolution

Predecessor: Cross-entropy, triplet loss. Successor: ArcFace, CosFace, AM-Softmax.
