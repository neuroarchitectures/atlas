# Triplet Loss

## Design Philosophy

Learn embeddings by pulling positive pairs together and pushing negative pairs apart. For an anchor, a positive (same class), and a negative (different class), the loss is max(0, d(a,p) - d(a,n) + margin), where d is a distance metric.

## Functionality

1. Select anchor a, positive p (same label), negative n (different label). 2. Compute d(a,p) = ||a - p||_2 and d(a,n) = ||a - n||_2. 3. Loss = max(0, d(a,p) - d(a,n) + margin). The margin is a hyperparameter controlling how far negatives should be from the anchor.

## Used By

FaceNet (face recognition) | Sentence embeddings | Person re-identification | Image retrieval

## Features

- **Metric learning**: Learns a distance metric directly from data.
- **Margin**: Controls the minimum gap between positive and negative distances.
- **Hard negative mining**: Selecting the hardest negatives (closest to anchor) is critical for convergence.

## Evolution

Predecessor: Contrastive loss (pairwise). Successor: Semi-hard mining, circle loss, multi-similarity loss.
