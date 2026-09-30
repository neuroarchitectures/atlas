# Extending LLM Context via Associative Recurrent Memory

> Source: `https://arxiv.org/abs/2607.11614`

---

**Authors:** Gleb Kuzmin, Ivan Rodkin, Aydar Bulatov, Yuri Kuratov, Lyudmila Rvanova, Mikhail Katkov, Ilia Sochenkov, Misha V. Tsodyks, Timothy Baldwin, Mikhail Burtsev et al.

**Published:** 2026

**Concepts:** Computer science, Associative property, Transformer, Artificial intelligence, Content-addressable memory

## Abstract

Extending the context length of large language models (LLMs) is critical for many real-world applications, yet standard transformers remain constrained by quadratic compute and linear memory scaling. In this work, we investigate the Associative Recurrent Memory Transformer (ARMT) as a practical approach for enabling long-context processing in LLMs, constant memory scaling, and better efficiency. We make three main contributions. First, we construct two domain-specific long-context datasets designed to evaluate realistic workloads, focusing on narrow-domain fine-tuning scenarios. Second, we propose a comprehensive training recipe for ARMT-based context extension, combining continued pre-training, synthetic long-context data generation, curriculum learning, and selective integration of associative memory into chosen model layers. Third, we present an extensive experimental study demonstrating that ARMT-augmented models: (i) process inputs well beyond their original context limits without degrading performance relative to in-limit baselines; (ii) generalize more effectively to out-of-distribution context lengths; and (iii) need 30% less FLOPs while preserving baseline performance within the original context window.

**PDF:** https://arxiv.org/pdf/2607.11614