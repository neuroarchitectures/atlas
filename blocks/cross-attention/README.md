# Cross-Attention Block

## Design Philosophy

Attention where queries come from one sequence and keys/values from another. The philosophy: the decoder (or image latents, or a multimodal bridge) *queries* an external memory (encoder output, text embeddings) to retrieve relevant information. This is the mechanism that makes seq2seq, multimodal fusion, and conditioning work.

## Functionality

- `Q = W_Q · x_decoder` (queries from the decoder/primary stream)
- `K = W_K · x_encoder`, `V = W_V · x_encoder` (keys/values from the encoder/condition stream)
- `CrossAttn = softmax(QK^T / √d_k) V` — the decoder retrieves from the encoder.
- Wrapped in residual + norm, like self-attention.

## Used By

| Model | Role |
|-------|------|
| Transformer (original) | Decoder cross-attends encoder output |
| Whisper | Decoder cross-attends audio encoder states |
| T5 | Encoder-decoder cross-attention |
| Diffusion U-Net (SD) | Image latents cross-attend CLIP text embeddings |
| BLIP-2 | Q-Former queries cross-attend frozen image features |
| Flamingo | Gated cross-attention to vision features |

## Features

- **Asymmetric**: Q from one stream, K/V from another — the retrieval mechanism.
- **Conditioning**: The standard way to inject text/audio/image conditioning into a model.
- **Fixed memory**: The K/V stream is often cached (encoder output) and reused across decoding steps.

## Evolution

- **Predecessor**: Bahdanau attention (additive, over RNN states).
- **Successor**: Gated cross-attention (Flamingo, tanh-gated for stable training); prefix-token injection (LLaVA, no cross-attn).
