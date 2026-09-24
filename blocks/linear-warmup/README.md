# Linear LR Warmup

## Design Philosophy

At step 0 the optimizer's moment estimates are all zero and the gradients are large, so taking a full-size Adam step can immediately derail training (especially with large batches, mixed precision, or newly initialized attention layers). Warmup ramps the learning rate **linearly from ~0 to the target value over the first few percent of training**, letting the statistics settle before large updates are taken.

## Functionality

`η_t = η_target · t / T_warmup` for `t < T_warmup`, after which the main schedule (usually cosine) takes over.

- `T_warmup` in practice: ~1k–10k steps, or 1–5% of total steps; scales with batch size (larger batch → longer warmup).
- Variants: **constant warmup** (fixed small LR), **gradual warmup** (linear, the standard), and warmup applied per-parameter-group.
- Often paired with **gradient clipping** in the early phase; in fp16 training, warmup also mitigates scale-overflow issues.

## Used By

| Model | Role |
|-------|------|
| BERT / GPT / LLaMA pretraining | Mandatory with AdamW + large batch |
| ViT / DINOv2 / SAM 2 / Depth Anything V2 | Vision pretraining and finetuning |
| DETR family (DETR, Deformable DETR, RT-DETR) | Stabilizes bipartite-matching training |
| VGGT, MASt3R, MoGe | 3D foundation-model training |

## Features

- **Prevents early divergence** without changing the final schedule.
- Nearly free: one scalar multiplication per step.
- Interacts with the main decay — warmup + cosine is the de facto default recipe.

## Evolution

- **Predecessor**: no warmup (fine for SGD with momentum on CNNs, failed for Adam + large batch).
- **Itself**: gradual warmup (Goyal et al., 2017 — "Accurate, Large Minibatch SGD").
- **Related**: warmup-stable-decay (WSD), which keeps a stable phase after warmup and only decays at the end; warmup of other quantities (weight decay, EMA decay, or MoE temperature).
