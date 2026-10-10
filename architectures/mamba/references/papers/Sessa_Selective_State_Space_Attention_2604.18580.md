# Paper (Horbatko et al. 2026)

> Source: `https://arxiv.org/abs/2604.18580`

---

**Sessa: Selective State Space Attention**

Liubomyr Horbatko

**Abstract**

Modern sequence modeling is dominated by two families: Transformers, whose self-attention can access arbitrary elements of the visible sequence, and structured state-space models, which propagate information through an explicit recurrent state. These mechanisms face different limitations on long contexts: when attention is diffuse, the influence of individual tokens is diluted across the effective support, while recurrent state propagation can lose long-range sensitivity unless information is actively preserved. As a result, both mechanisms face challenges in preserving and selectively retrieving information over long contexts. We propose Sessa, a decoder that places attention inside a recurrent feedback path. This creates many attention-based paths through which past tokens can influence future states, rather than relying on a single attention read or a single recurrent chain. We prove that, under explicit assumptions and matched regimes, Sessa admits power-law memory tails $O(\ell^{-β})$ for $0 < β< 1$, with slower decay than in the corresponding Transformer and Mamba-style baselines. We further give an explicit construction that achieves this power-law rate. Under the same assumptions, Sessa is the only model class among those considered that realizes flexible selective retrieval, including profiles whose influence does not decay with distance. Consistent with this theoretical advantage, across matched experiments, Sessa achieves the strongest performance on long-context benchmarks while remaining competitive with Transformer and Mamba-style baselines on short-context language modeling.

---

- **arXiv ID**: `2604.18580`
- **Published**: 2026-04-20
- **Categories**: cs.LG, cs.AI, cs.CL
- **PDF**: https://arxiv.org/pdf/2604.18580
