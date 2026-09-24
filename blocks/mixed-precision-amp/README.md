# Mixed Precision (AMP / BF16 / FP16)

## Design Philosophy

Modern accelerators compute FP16/BF16 matrix products several times faster than FP32, and halving the dtype halves memory traffic. Mixed precision keeps a **FP32 master copy of weights** and runs the math in low precision: it is a throughput and memory technique that must not change convergence — hence the loss-scaling and master-weight machinery.

## Functionality

```
weights: FP32 master copy
forward : cast weights -> FP16/BF16, compute in low precision
loss    : multiply by scale S (FP16 only), backward in low precision
grads   : unscale by S, cast to FP32, clip, optimizer step on FP32 weights
```

- **FP16**: needs **loss scaling** (dynamic or fixed) to prevent small gradients flushing to zero; narrower range (max 65504) → risk of overflow/underflow.
- **BF16**: same exponent range as FP32, so **no loss scaling needed** and far fewer NaN problems — the default on Ampere+/TPU; slightly less mantissa precision.
- **AMP (automatic mixed precision)**: per-op cast lists (matmul/conv in low precision, reductions/norms/losses in FP32).
- **FP8** (Hopper+): next step, with per-tensor scaling; used for inference quantization-aware training and large-scale pretraining.
- Keep in FP32: LayerNorm/Softmax/CrossEntropy reductions, optimizer state, master weights.

## Used By

| Architecture | How it is used |
|---|---|
| ResNet / EfficientNet training | canonical NVIDIA APEX/AMP recipes (FP16 + dynamic loss scaling) |
| ViT / BERT / GPT / LLaMA | BF16 training (bfloat16 preferred for stability at scale) |
| DETR / RT-DETR / YOLO | AMP is on by default in most modern training loops |
| Diffusion / video models | BF16 or FP16 for activation-heavy U-Nets; attention in FP32 where unstable |
| On-device deployment | training-side precursor to INT8/FP16 inference quantization |

## Features

- **Speed**: 1.5–3× on tensor-core hardware; larger speedups when memory-bound.
- **Memory**: ~half activations → larger batches, longer sequences, higher resolution.
- **Master weights**: the essential detail — the optimizer step happens in FP32 even though compute is low precision.
- **Loss scaling**: required for FP16, unnecessary for BF16.
- **Numerics**: watch for overflow (FP16), silent underflow, and ops that must stay FP32 (reductions, norms).
- **Determinism/reproducibility**: low-precision reductions make bitwise reproduction unrealistic.

## Evolution

- **FP32 training** (baseline).
- **FP16 + loss scaling (2017–2018, NVIDIA APEX, Micikevicius et al.)**: first practical mixed precision.
- **BF16 (2018+, TPU/Ampere)**: no loss scaling, range-safe; the current default for large models.
- **AMP in PyTorch/TF (2019)**: automatic cast lists and dynamic scaler.
- **FP8 (2022+, H100)**: per-tensor/amax-history scaling for training and inference.
- **Related**: `syncbatchnorm` (batch stats must be computed in FP32), Kahan summation for BF16 optimizer steps.

## See Also

`gradient-checkpointing`, `adamw-optimizer`, `batch-norm`, `syncbatchnorm`
