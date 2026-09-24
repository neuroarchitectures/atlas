# SyncBatchNorm

## Design Philosophy

BatchNorm's statistics are computed **over the batch**, so with data parallelism each replica normalizes with its own *local* batch — effectively a much smaller batch than you configured. When the per-GPU batch gets small (large images, video, detection, segmentation), those statistics become noisy and accuracy drops. SyncBatchNorm computes mean/variance **across all replicas** (all-reduce before normalization), so the normalization batch equals the global batch.

## Functionality

```
local:  mean, var over the replica's own N samples
sync:   all_reduce(mean, var) across replicas -> global mean/var
then:   normalize locally with the global statistics
backward: gradients of the loss w.r.t. the global stats are also all-reduced
```

- Requires a communication step per BN layer per forward (and one in backward) — that is the cost.
- Statistics are still **batch-dependent** (unlike GroupNorm/LayerNorm), so it does not solve the small-batch problem in an absolute sense — it makes the batch *as large as you configured*.
- Standard in DDP: `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)`.
- Buffers (running mean/var) are updated with the global statistics; they must be kept in FP32 under mixed precision.
- Gotcha: frozen BN / eval-mode BN still uses running stats; sync only matters in training mode.

## Used By

| Architecture | How it is used |
|---|---|
| Object detection (Faster R-CNN, RetinaNet, DETR-family, YOLO multi-GPU) | per-GPU batch is 1–2 images; sync BN is required for the reported numbers |
| Semantic segmentation (DeepLab, Mask2Former) | large images → tiny per-GPU batch |
| Video / 3D models | clips are memory-heavy, batch per GPU is small |
| Contrastive / self-supervised learning | MoCo/SimCLR-style: shard BN or sync BN to avoid the "info leak" of local stats |

## Features

- **Restores the intended batch size** for normalization — the whole point.
- **Cost**: an all-reduce per BN layer; noticeable on many-GPU jobs with many BN layers (mitigated by fusing or by replacing BN with GN).
- **Consistency**: makes single-node and multi-node training behave the same way.
- **Not a fix for batch-size 1**: if the global batch is tiny, use GroupNorm / LayerNorm instead.
- **Mixed precision**: compute batch statistics in FP32 even under AMP.

## Evolution

- **BatchNorm (2015)**: per-device batch statistics.
- **Data-parallel BN**: each replica computes its own → mismatch with the nominal batch size.
- **SyncBatchNorm (2018, PyTorch/MegDet)**: cross-replica all-reduce of statistics; became standard for detection/segmentation.
- **Shuffle/Sharding BN variants**: normalize within a subset for contrastive learning to prevent leakage.
- **Alternatives**: **GroupNorm (2018)** — batch-independent, the Mask R-CNN/Video choice when batches are small; **LayerNorm** in transformers; **Ghost BN / accumulating stats** for very small batches.

## See Also

`batch-norm`, `groupnorm`, `layer-norm`, `mixed-precision-amp`
