# MambaVision Mixer

## Design Philosophy

Mamba blocks handle local texture well but lag attention on long-range spatial dependencies. MambaVision redesigns the SSM block for vision (symmetric branch, conv path) and — critically — *integrates* it: CNN residual blocks at high resolution, Mamba mixers in the middle, self-attention only in the final stages where global reasoning pays.

## Functionality

- Mixer: parallel branches — depthwise conv + SSM path — merged, then MLP; residual around both.
- Hierarchy: conv stem → mixer+MLP stages → transformer blocks at final stages (integration pattern from ablations).

## Used By

| Model | Role |
|-------|------|
| MambaVision | Multi-resolution vision backbone; attention placed only at the last stages |

## Features

- **Vision-tuned SSM block** — the conv branch compensates SSM's weakness on fine 2D structure.
- **Hybrid placement rule** — where each mixer type earns its cost.

## Evolution

- **Predecessor**: VMamba's ss2d-selective-scan (pure-SSM vision); Vim's bidirectional-ssm.
- **Successor**: MambaVision-T2; the hybrid-placement lesson generalizes to jamba's ssm-attention-hybrid.
