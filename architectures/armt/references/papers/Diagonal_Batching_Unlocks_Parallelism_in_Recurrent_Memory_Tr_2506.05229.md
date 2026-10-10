# Diagonal Batching Unlocks Parallelism in Recurrent Memory Transformers for Long Contexts

> Source: `https://arxiv.org/abs/2506.05229`

---

**Authors:** Danil Sivtsov, Ivan Rodkin, Gleb Kuzmin, Yuri Kuratov, Ivan Valer'evich Oseledets

**Published:** 2025

**Concepts:** Speedup, Computer science, Inference, Diagonal, Computation

## Abstract

Transformer models struggle with long-context inference due to their quadratic time and linear memory complexity. Recurrent Memory Transformers (RMTs) offer a solution by reducing the asymptotic cost to linear time and constant memory usage. However, their memory update mechanism leads to sequential execution, causing a performance bottleneck. We introduce Diagonal Batching, a scheduling scheme that unlocks parallelism across segments in RMTs while preserving exact recurrence. This approach eliminates the sequential constraint, enabling efficient GPU inference even for single long-context inputs without complex batching and pipelining techniques. Because the technique is purely a run-time computation reordering, existing RMT models adopt it with no retraining. Applied to a LLaMA-1B ARMT model, Diagonal Batching yields a 3.3x speedup over standard full-attention LLaMA-1B and a 1.8x speedup over the sequential RMT implementation on 131,072-token sequences. By removing sequential bottleneck, Diagonal Batching reduces inference cost and latency, thereby strengthening RMTs as a practical solution for real-world, long-context applications.

**PDF:** https://arxiv.org/pdf/2506.05229