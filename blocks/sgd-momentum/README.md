# SGD with Momentum (Heavy-Ball / Nesterov)

## Design Philosophy

The oldest optimizer that still wins on vision: update along the gradient, but carry a **velocity** term so that consistent directions accumulate and oscillating directions cancel. It adds one buffer per parameter (same size as the parameter) and no per-parameter adaptive state — which is exactly why it remains the default for CNNs, detection heads, and fine-tuning where AdamW's adaptivity costs generalization.

## Functionality

```
v_t = mu * v_{t-1} + g_t            # heavy-ball
theta_t = theta_{t-1} - lr * v_t    # standard
# Nesterov variant: look ahead before computing the gradient
v_t = mu * v_{t-1} + grad(theta - lr * mu * v_{t-1})
```

- `mu` (momentum): 0.9 is the default; 0.99 appears with large-batch training.
- Two forms: **heavy-ball** (gradient at the current point) and **Nesterov** (gradient at the look-ahead point). Nesterov usually converges a bit faster; heavy-ball is more common in vision code.
- Often combined with weight decay (L2) — note that with SGD, decoupled vs. coupled weight decay is a real choice (AdamW decouples it; SGD's classic `weight_decay` is coupled).
- Buffer size: 1× parameters (vs. 2× for Adam/AdamW, 3× with master weights in mixed precision).

## Used By

| Architecture | How it is used |
|---|---|
| ResNet / VGG / EfficientNet | the canonical SGD+momentum training recipe |
| Faster R-CNN / Mask R-CNN / RetinaNet | standard detection training (SGD+momentum, 0.9) |
| YOLOv5–v11 | SGD or AdamW selectable; SGD common for the largest models |
| DETR-family fine-tuning | frequently SGD for backbone-only fine-tuning |
| MoCo / SimCLR / BYOL | self-supervised baselines use SGD + momentum (plus EMA encoder) |

## Features

- **Generalization**: on image classification and detection, well-tuned SGD+momentum frequently beats AdamW on final accuracy.
- **Memory**: one velocity buffer; no second moment — the cheapest stateful optimizer.
- **Scale-invariance of learning rate**: none — learning rate must be tuned for batch size (linear scaling rule: `lr *= batch_size / base_batch`).
- **Nesterov**: available as a one-line flag; useful when SGD under-performs.
- **Interaction with BatchNorm**: SGD + BN is the classic pairing; BN's noise makes adaptive methods less necessary.

## Evolution

- **SGD (1951–)**: Robbins–Monro stochastic approximation.
- **Heavy-ball / momentum (Polyak 1964)**: add velocity.
- **Nesterov accelerated gradient (1983)**: look-ahead gradient.
- **Adam / AdamW (2015/2019)**: adaptive per-parameter scaling; dominates transformers.
- **LARS / LAMB (2017/2019)**: layer-wise lr scaling for huge batches — SGD adapted to large-batch pretraining.
- **Current practice**: AdamW for transformers and pretraining; **SGD+momentum remains the strong choice for CNN fine-tuning and detection**, and is what most "best final accuracy" recipes in vision use.

## See Also

`adamw-optimizer`, `lion-optimizer`, `kaiming-init`, `linear-warmup`, `cosine-lr-schedule`
