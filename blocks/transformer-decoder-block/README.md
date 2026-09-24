# Transformer Decoder Block

## Design Philosophy

The encoder block extended with two additions: (1) **masked self-attention** so positions can't see the future (autoregressive), and (2) **encoder-decoder cross-attention** so the decoder can attend to the encoder's output. The philosophy: a sequence-to-sequence model where the decoder generates output one token at a time, conditioned on the encoder's representation of the input.

## Functionality

Three sub-layers, each with residual + LayerNorm:

1. **Masked multi-head self-attention**: Same as encoder MHA, but a causal mask zeros out future positions.
2. **Encoder-decoder cross-attention**: Queries from the decoder; keys/values from the encoder output.
3. **Position-wise FFN**: Same as encoder.

Original (post-norm): `x → [masked-MHA → add → LN] → [cross-attn → add → LN] → [FFN → add → LN]`.

## Used By

| Model | Role |
|-------|------|
| Transformer (original) | 6 decoder blocks + 6 encoder blocks |
| T5 | Decoder blocks (with relative position bias) |
| Whisper | 12 decoder blocks, each cross-attends encoder |
| BART | Decoder (with encoder for denoising) |

## Features

- **Autoregressive masking**: The causal mask preserves the "generate one at a time" property during training.
- **Cross-attention**: The bridge from encoder to decoder; the decoder *queries* the encoder memory.
- **KV cache**: At inference, past K/V are cached so each new token only computes one row.

## Evolution

- **Predecessor**: Encoder block + seq2seq RNN decoders.
- **Successor**: Decoder-only block (GPT family) — drops cross-attention, keeps only masked self-attention + FFN.
