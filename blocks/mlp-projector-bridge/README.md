# MLP Projector Bridge (LLaVA-style)

## Design Philosophy

Map image-patch features into the LLM's token embedding space with a tiny **2-layer MLP**, then prepend the visual tokens to the text tokens. The philosophy: the simplest possible bridge — no cross-attention, no resampler, just a linear projection. The unmodified LLM decoder treats the image as a prefix. Simple, and it defined the open multimodal-LLM playbook.

## Functionality

- **Vision encoder** (frozen CLIP ViT-L/14): produces `N` patch features of dim `D_v`.
- **MLP projector**: `D_v → D_llm` (2-layer MLP, the only trained bridge).
- **Token prepending**: Projected visual tokens are concatenated *in front of* the text tokens.
- **LLM** (Llama): attends over image + text tokens jointly, unmodified.
- Only the projector (and the LLM, optionally) are trained; the vision encoder stays frozen.

## Used By

| Model | Role |
|-------|------|
| LLaVA-1.5-7B | The canonical recipe |
| Qwen-VL, InternVL | Variants of the prefix-token + projector pattern |

## Features

- **Minimal**: A 2-layer MLP is the whole bridge.
- **Prefix tokens**: Visual tokens prepended to text — the LLM needs no modification.
- **Open playbook**: Most open MLLMs follow this pattern.

## Evolution

- **Predecessor**: Flamingo (cross-attention bridge, more complex); BLIP-2 (Q-Former bridge).
- **Successor**: LLaVA-NeXT (higher-res, more visual tokens); the projector pattern is now the open default.
