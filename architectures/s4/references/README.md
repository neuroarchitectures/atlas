# References: S4

## Papers

- [`Efficiently_Modeling_Long_Sequences_with_Structured_State_Sp_Gu_Goel_2021.md`](papers/Efficiently_Modeling_Long_Sequences_with_Structured_State_Sp_Gu_Goel_2021.md) — primary source paper (from PaperVault `DL-Architectures/01-transformers/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **Code:** https://github.com/HazyResearch/state-spaces
- **arXiv:** https://arxiv.org/abs/2111.00396

## Related

- **Linear State Space Layer (LSSL)** — Direct predecessor; showed SSMs can address LRDs but was computationally infeasible. S4 solves its bottleneck.
- **HiPPO framework** — Provides the special `A` matrices for continuous-time memorization; foundational to S4's parameterization.
- **Mamba (S6)** — Successor; introduces selective state spaces making the SSM input-dependent, achieving Transformer-level language modeling.
- **H3** — Hybrid attention-SSM architecture for language modeling building on S4.
- **S5** — Simplified S4 using multi-input multi-output (MIMO) SSM.
- **Transformer** — S4 matches or exceeds Transformers on LRD tasks while being more efficient (`O(N+L)` vs `O(n²)`).
- **RNN/LSTM** — S4's recurrent view provides efficient inference like RNNs but with principled LRD handling.
- **Long Range Arena (LRA)** — Benchmark where S4 first solved the Path-X task (length 16384).
