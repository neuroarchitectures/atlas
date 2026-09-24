# Embedding Distillation

## Design Philosophy

A large teacher's *encoder embedding* is the valuable asset — not its outputs. Distill student embeddings directly against frozen teacher embeddings, without backprop through the teacher's decoder; the sampling/weighting strategy of the pairs is where variants differ.

## Functionality

- MobileSAM: MSE between tiny ViT student and frozen ViT-H image-encoder embeddings (~11k images, <1% of full-SAM compute); teacher's prompt encoder/mask decoder reused unchanged.
- MiniLM (all-minilm-l6): student's self-attention distributions mimic a deep teacher layer's.
- EdgeSAM: disagreement-driven prompt sampling — dynamically sample prompts where teacher and student masks disagree, as training-only dynamic tokens.
- VideoPrism: global-local distillation — the contrastive global (CLS) embedding distilled into every patch token so a frozen encoder serves dense heads.

## Used By

| Model | Role |
|-------|------|
| MobileSAM | Decoupled encoder-only embedding MSE distillation |
| all-MiniLM-L6-v2 | Deep self-attention distillation into a 6-layer/384-dim student |
| EdgeSAM | Prompt-in-the-loop disagreement sampling distillation |
| VideoPrism | Global (CLS) → patch token distillation |

## Features

- **No teacher backprop** — embeddings are targets, not gradients.
- **Prompt/sampling strategy as the variant axis**.

## Evolution

- **Predecessor**: BERT-style KD, TinyBERT (attention + hidden state matching).
- **Related**: gram-anchoring-loss (similarity-geometry distillation); go-lsd-self-distillation (self, cross-layer).
