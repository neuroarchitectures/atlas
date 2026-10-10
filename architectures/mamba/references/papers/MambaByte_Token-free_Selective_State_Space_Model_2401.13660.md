# Paper (Wang et al. 2024)

> Source: `https://arxiv.org/abs/2401.13660`

---

**MambaByte: Token-free Selective State Space Model**

Junxiong Wang, Tushaar Gangavarapu, Jing Nathan Yan, Alexander M. Rush

**Abstract**

Token-free language models learn directly from raw bytes and remove the inductive bias of subword tokenization. Operating on bytes, however, results in significantly longer sequences. In this setting, standard autoregressive Transformers scale poorly as the effective memory required grows with sequence length. The recent development of the Mamba state space model (SSM) offers an appealing alternative approach with a fixed-sized memory state and efficient decoding. We propose MambaByte, a token-free adaptation of the Mamba SSM trained autoregressively on byte sequences. In terms of modeling, we show MambaByte to be competitive with, and even to outperform, state-of-the-art subword Transformers on language modeling tasks while maintaining the benefits of token-free language models, such as robustness to noise. In terms of efficiency, we develop an adaptation of speculative decoding with tokenized drafting and byte-level verification. This results in a $2.6\times$ inference speedup to the standard MambaByte implementation, showing similar decoding efficiency as the subword Mamba. These findings establish the viability of SSMs in enabling token-free language modeling.

---

- **arXiv ID**: `2401.13660`
- **Published**: 2024-01-24
- **Categories**: cs.CL, cs.LG
- **PDF**: https://arxiv.org/pdf/2401.13660
