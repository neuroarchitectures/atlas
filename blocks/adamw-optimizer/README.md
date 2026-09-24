# AdamW (Adam with Decoupled Weight Decay)

## Design Philosophy

Adam is scale-invariant and fast, but L2 regularization added to the *loss* interacts badly with Adam's per-parameter normalization: the weight-decay gradient gets rescaled by the adaptive denominator, so the effective regularization is far weaker than intended (and adaptive-gradient methods generalize worse than SGD). AdamW fixes this by **decoupling weight decay from the gradient update** — decay is applied directly to the parameters.

## Functionality

```
m_t = β1·m_{t-1} + (1−β1)·g_t            # first moment (momentum)
v_t = β2·v_{t-1} + (1−β2)·g_t²           # second moment (uncentered variance)
m̂_t = m_t / (1−β1^t),  v̂_t = v_t / (1−β2^t)   # bias correction
θ_t = θ_{t-1} − η · m̂_t / (√v̂_t + ε)
θ_t = θ_t − η · λ · θ_{t-1}               # decoupled weight decay
```

- Defaults: `β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8`; weight decay `λ` typically `0.01–0.1`.
- Requires a learning-rate **schedule** (warmup + decay) to perform well; Adam alone with a constant LR plateaus.
- Memory: two optimizer states per parameter (m and v), i.e. 2× model size in fp32.

## Used By

| Model | Role |
|-------|------|
| BERT / RoBERTa / GPT / LLaMA / Qwen | Standard pretraining optimizer |
| ViT, DINOv2, SAM / SAM 2, Depth Anything V2 | Standard vision pretraining optimizer |
| DETR / Deformable DETR / RT-DETR / Grounding DINO | Detection finetuning (with per-group decay) |
| VGGT, MASt3R, MoGe | 3D model training |

## Features

- **Correct weight decay**: regularization strength is independent of the adaptive scaling.
- **Per-parameter adaptive step size** — tolerant of poorly scaled losses and sparse gradients.
- **No- LR-warmup needed**: warmup is required or early training is unstable.
- Detectors commonly use **layer-wise decay** and **no decay on norms/biases**.

## Evolution

- **Predecessors**: SGD with momentum; AdaGrad; RMSProp; Adam (Kingma & Ba, 2015).
- **Itself**: AdamW (Loshchilov & Hutter, 2019).
- **Variants/successors**: AdamW with decoupled schedules (WSD), LAMB (large-batch BERT), Adafactor / 8-bit Adam (memory), Lion and Sophia (second-order-ish, cheaper), Muon (orthogonalized momentum for hidden weights).
