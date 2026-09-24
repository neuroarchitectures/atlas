# Gradient Checkpointing (Activation Recomputation)

## Design Philosophy

Training memory is dominated by **activations** stored for the backward pass, not by parameters. Gradient checkpointing trades compute for memory: store only a subset of activations, and **recompute the discarded ones during backward**. It is the standard way to fit a batch, a long sequence, or a high-resolution input into a fixed memory budget.

## Functionality

```
forward : save only checkpointed boundaries (e.g. every block's input), drop intermediates
backward: recompute the segment's forward from its boundary, then propagate gradients
```

- **Selective checkpointing**: checkpoint expensive/rare layers (attention, transformer blocks) and keep cheap ones (norms, activations) — the usual compromise.
- **Full vs. block**: checkpointing every transformer block costs ~+33% forward compute (one extra forward) for a large memory reduction.
- Where to apply: transformer blocks, long-sequence attention, video models (temporal depth), high-resolution CNNs, diffusion U-Nets.
- Interactions: must be applied consistently with other memory tricks (mixed precision keeps master weights; checkpointing does not affect them); incompatible-by-default with some custom autograd functions that assume stored tensors.
- **Not** the same as gradient accumulation (which reduces batch memory by splitting the batch) — checkpointing reduces *activation* memory per sample.

## Used By

| Architecture | How it is used |
|---|---|
| GPT / LLaMA pretraining | checkpoint every transformer block; enables long context |
| ViT / Swin at high resolution | block-level checkpointing |
| Video models (V-JEPA, VideoMAE) | essential: temporal depth multiplies activations |
| BEVFormer / multi-view 3D | per-view + transformer block checkpointing to fit 6-camera batches |
| Diffusion U-Nets | encoder-block checkpointing |

## Features

- **Memory**: activation memory drops from O(depth) to O(√depth) or O(1) per segment, at the cost of recompute.
- **Throughput**: ~20–35% slower per step in exchange for fitting a much larger batch — often a net throughput *win* because the batch can grow.
- **Selective application**: checkpoint only the memory-heavy sub-modules to keep the slowdown small.
- **Orthogonal to precision**: combine with `mixed-precision-amp` and gradient accumulation; the three are independent knobs.
- **Correctness**: numerically identical (up to recompute nondeterminism) — it does not change the math.

## Evolution

- **Martens & Sutskever-style recompute / Chen et al. (2016)**: "Training Deep Nets with Sublinear Memory Cost" formalized checkpointing.
- **PyTorch `torch.utils.checkpoint`, TF `recompute_grad`**: standard library support.
- **Selective / block checkpointing (2019+)**: per-module control; becomes default in LLM stacks.
- **Modern**: **fully-sharded + checkpointing**, **CPU offloading**, **sequence/selective-scan recompute** — the same idea applied to state-space and video models.

## See Also

`mixed-precision-amp`, `stochastic-depth`, `adamw-optimizer`
