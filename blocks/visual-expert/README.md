# Visual Expert (CogVLM)

## Design Philosophy

Injecting image features only at the input of an LLM forces the language layers to carry visual reasoning implicitly. A "visual expert" instead gives image tokens their own dedicated parameters inside every transformer layer, so vision and language can be processed by separate, specialised pathways that interact through attention.

## Functionality

Inside each decoder layer, route tokens by modality:

- Text tokens use the standard QKV and FFN weights.
- Image tokens use a parallel set of QKV and FFN weights (the expert).

```
h = Attn( x ; W_text if text else W_vision )
h = FFN( h ; W_text^ffn if text else W_vision^ffn )
```

Both pathways share the same attention computation, so image and text tokens still attend to each other.

## Used By

CogVLM | CogVLM2

## Features

- **Modality-specific capacity**: vision gets its own parameters at every layer.
- **Deep fusion**: image features influence every layer, not just the input.
- **No degradation**: the text pathway stays a pure LLM for text-only inputs.
- **Parameter-efficient**: only the expert weights are added to the frozen LLM.

## Evolution

Predecessor: input-only visual projection (LLaVA, BLIP-2). Sibling: gated cross-attention (Flamingo), decoupled cross-attention (IP-Adapter). Successor: modality-aware experts in later VLMs.
