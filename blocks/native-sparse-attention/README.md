# Native Sparse Attention (NSA)

## Design Philosophy

Block-sparse attention where each query attends only to a selected **top-k of blocks**. The philosophy: long-range reach without the full n×n cost, and *hardware-aligned* (block-granular) so it is fast in practice. Unlike a fixed window, selected blocks can be anywhere in the sequence, preserving genuine long-range links. NSA makes the sparsity natively trainable (learned during pretraining, not applied post-hoc).

## Functionality

Three parallel branches, outputs combined via learned gates:

1. **Compression branch**: Aggregate blocks of keys/values into coarse representations; attention over compressed tokens captures global patterns.
2. **Selection branch**: Score each block's importance (using compressed attention as proxy); keep top-n blocks; fine-grained attention only over those.
3. **Sliding window branch**: Local context, always attended.
4. **Gated combination**: `o = Σ g_c · Attn_branch_c`, gates from an MLP + sigmoid.

- **Block granularity**: Selection and compute align to GPU tiles — fast where token-level sparsity is not.
- **Natively trainable**: Sparsity learned during pretraining, full gradient flow.

## Used By

| Model | Role |
|-------|------|
| NSA (DeepSeek) | The defining mechanism |
| DeepSeek-V3 (partial) | Sparse attention variants |
- Compatible with GQA/MQA (blockwise design aligns with shared KV).

## Features

- **Long-range + efficient**: Top-k blocks anywhere in the sequence, O(n · topBlocks) cost.
- **Hardware-aligned**: Block granularity → Tensor Core utilization.
- **Three complementary branches**: Compression (global), selection (fine important), window (local) — no single branch suffices.

## Evolution

- **Predecessor**: Sliding window (local only); H2O / Quest (post-hoc KV eviction, not trainable).
- **Successor**: An active direction toward natively sparse LLMs.
