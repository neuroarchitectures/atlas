# Cosine Learning Rate Schedule

## Design Philosophy

The learning rate should be **high early (to explore) and tiny late (to settle into a sharp minimum)**. A cosine decay does exactly that: it drops slowly at first, then rapidly in the middle, then flattens near zero — a shape that empirically beats step decay for most supervised training, and needs no hand-picked drop epochs.

## Functionality

`η_t = η_min + 0.5 · (η_max − η_min) · (1 + cos(π · t / T))`, where `t` is the current step and `T` the total number of steps.

- Usually combined with **linear warmup** for the first 1–5% of steps.
- `η_min` is often 0 (or a small fraction such as `η_max/100`).
- Multiplicative vs. additive: applied on top of per-parameter-group scaling (layer-wise LR decay).
- Restarts (SGDR / cosine annealing with restarts) reuse the same curve for cyclical training.

## Used By

| Model | Role |
|-------|------|
| ResNet / EfficientNet supervised training | Standard 300-epoch schedule |
| DETR / Deformable DETR / RT-DETR / Grounding DINO | Detection finetuning |
| Depth Anything V2, MoGe, MASt3R, VGGT | Vision / 3D model training |
| Almost every supervised vision recipe | Default decay shape |

## Features

- **Smooth, no drop epochs** — no discrete staircase, no manual milestones.
- Strong final accuracy; the long low-LR tail acts like averaging.
- Requires knowing the **total number of steps in advance** — awkward for open-ended or continued training; **WSD (warmup–stable–decay)** addresses this by keeping a constant phase and decaying only when you decide to stop.

## Evolution

- **Predecessors**: fixed LR, step decay (e.g. ×0.1 at epochs 30/60), exponential decay, polynomial decay.
- **Itself**: cosine annealing (Loshchilov & Hutter, SGDR, 2017).
- **Successors**: WSD / warmup-stable-decay (MiniCPM), OneCycle (super-convergence with a rising then falling LR), and schedule-free / constant-LR-with-averaging optimizers.
