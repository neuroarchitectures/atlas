# Continuous Next-Embedding Prediction

## Design Philosophy

Language models output distributions over a vocabulary; but concepts (sentences, ideas) live in continuous spaces. LCM's head predicts the *next element as a continuous vector* — MSE regression, diffusion, or quantized codebook variants — making sequence modeling language- and modality-agnostic.

## Functionality

- Transformer core over SONAR sentence embeddings (1024-d); next-concept head variants: regression (MSE), diffusion-based sampling, or codebook quantization of the continuous target.

## Used By

| Model | Role |
|-------|------|
| LCM (Large Concept Model) | Autoregressive next-concept prediction in embedding space |

## Features

- **Modality-free sequence modeling** — the same head works for text, speech, video embeddings.
- **Semantic units** — the sequence length is measured in concepts, not tokens.

## Evolution

- **Predecessor**: token-level LM heads; sequence-to-sequence translation.
- **Related**: multi-token-prediction (still vocabulary-bound); vggt-world's continuous latents (geometry instead of semantics).
