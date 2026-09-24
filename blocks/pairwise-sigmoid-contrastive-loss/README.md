# Pairwise Sigmoid Contrastive Loss

## Design Philosophy

Softmax contrastive loss couples every sample to the whole batch — batch size becomes a hyperparameter and small-batch training destabilizes. SigLIP treats every image-text pair as an *independent binary problem*: sigmoid(s_ij) predicting "is this the matching pair" — no batch-global normalization, so training works at any batch size.

## Functionality

- `L = −Σ_ij log σ(z_ij · (s_ij − b))` with learnable temperature z and bias b, applied over the full similarity matrix (each pair an independent sigmoid).

## Used By

| Model | Role |
|-------|------|
| SigLIP base | Image-text pretraining replacing CLIP's softmax InfoNCE |

## Features

- **Batch-size free** — the practical advantage that made SigLIP the default for small-batch training.
- **Per-pair calibration** — the bias term learns the decision point explicitly.

## Evolution

- **Predecessor**: CLIP's contrastive-dual-encoder + InfoNCE softmax.
- **Successor**: SigLIP 2 (loss refinements); sigmoid ranking objectives throughout retrieval.
