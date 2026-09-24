# References: Native Sparse Attention (NSA)

## Papers

- [`Native_Sparse_Attention_Gao_Dai_Luo_etal_2025.md`](papers/Native_Sparse_Attention_Gao_Dai_Luo_etal_2025.md) — primary source paper (from PaperVault `DL-Architectures/01-transformers/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **arXiv:** https://arxiv.org/abs/2502.11089
- **Institution:** DeepSeek-AI

## Related

- **FlashAttention** — Block-based attention implementation that NSA's hardware-aligned design builds upon for contiguous memory access.
- **H2O** — KV-cache eviction method; example of phase-restricted sparsity that NSA improves upon.
- **Quest** — Blockwise KV-cache selection; example of methods incompatible with GQA that NSA addresses.
- **HashAttention** — Token-granular selection with non-contiguous memory access; NSA's blockwise selection resolves this.
- **ClusterKV / MagicPIG** — Methods with non-trainable components (k-means, SimHash) that prevent gradient flow; NSA enables end-to-end training.
- **Multi-Query Attention (MQA) / Grouped-Query Attention (GQA)** — Modern decoding-efficient architectures that NSA is designed to be compatible with.
- **Transformer (Full Attention)** — The baseline that NSA matches or exceeds while achieving substantial speedups.
