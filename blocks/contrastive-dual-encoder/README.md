# Contrastive Dual-Encoder (CLIP)

## Design Philosophy

Two towers (image + text), each ending in a linear projection to a **shared embedding space**, trained so matching pairs score high by dot product. The philosophy: contrastive learning on 400M image-text pairs aligns the modalities without any cross-attention — the towers never interact during the forward pass; alignment happens only through the loss. This underpins modern multimodality.

## Functionality

- **Vision tower**: ViT image encoder → linear projection → 512-dim (normalized).
- **Text tower**: Transformer text encoder (causal) → linear projection → 512-dim (normalized).
- **Similarity**: `logit_{i,j} = (v_i · t_j) / τ`, where `τ` is a learned temperature.
- **Loss**: Symmetric cross-entropy over the batch (InfoNCE) — the diagonal (matching pairs) should be high, off-diagonal low.

## Used By

| Model | Role |
|-------|------|
| CLIP ViT-B/32 | The defining model (88M vision + 63M text) |
| CLIP ViT-L/14 | Larger variant, used by LLaVA, Stable Diffusion |
| SigLIP | Sigmoid loss variant |
| E5, BGE (text-only) | Same dual-encoder pattern for text retrieval |

## Features

- **No cross-attention**: Towers are independent at inference — batched retrieval is a dot product.
- **Shared space**: Image and text live in the same 512-dim space.
- **Zero-shot**: Classify by comparing to text prompts — no task-specific head.

## Evolution

- **Predecessor**: Image-text retrieval (VSE++); contrastive losses (InfoNCE).
- **Successor**: SigLIP (sigmoid loss); OpenCLIP; the text tower is the text encoder of Stable Diffusion.
