# Relative Position Bias

## Design Philosophy

What matters for attention is the *offset* between tokens, not their absolute indices. Add bias terms to attention logits as a function of relative distance — bucketed learned biases (T5) or continuous parameterized biases (Swin-V2's log-spaced form) — removing absolute position encodings entirely.

## Functionality

- T5: bucket relative distances (log-scaled buckets for far pairs) → learned per-head bias added to logits.
- Swin-V2: continuous log-spaced parameterization of relative position bias so pretrained (low-res) biases transfer to higher-resolution inputs without relearning.

## Used By

| Model | Role |
|-------|------|
| T5-Small | 8 heads, bucketed biases, no absolute PE |
| Swin-V2 | Log-spaced continuous bias in all window-attention blocks |

## Features

- **Translation invariance built in** — logits depend only on distance.
- **Resolution transfer** (Swin-V2 form) — pretrain low-res, deploy high-res.

## Evolution

- **Predecessor**: Shaw et al. 2018 relative attention; Transformer-XL.
- **Successor**: rope — rotation encodes relative offsets multiplicatively; ALiBi as distance-proportional bias without parameters.
