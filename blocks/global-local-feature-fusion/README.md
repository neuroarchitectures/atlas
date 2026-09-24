# Global-Local Feature Fusion (HQ-Features)

## Design Philosophy

ViT image encoders give rich semantics at the last layer but blur fine boundaries. SAM-HQ learns a *fusion* of an early high-resolution layer with the final layer into an "HQ-Features" map carrying both detail and semantics — beating FPN-style alternatives in ablation.

## Functionality

- Extract early ViT features + final features → fusion convs → HQ-Features map.
- A dedicated HQ-Output token in the mask decoder attends over HQ-Features and routes to a small conv upsampling path producing the high-quality mask (added to the standard mask).

## Used By

| Model | Role |
|-------|------|
| SAM-HQ | Only trainable parts: fusion layers + HQ token + upsampling path (rest of SAM frozen) |

## Features

- **Boundary fidelity** — the largest source of SAM-HQ's mask quality gains.
- **Minimal training budget** — SAM's encoder/decoder stay frozen.

## Evolution

- **Predecessor**: SAM's single final-layer image embedding.
- **Related**: multi-resolution-parallel-streams (HRNet) — persistent high-res as the conv-side counterpart.
