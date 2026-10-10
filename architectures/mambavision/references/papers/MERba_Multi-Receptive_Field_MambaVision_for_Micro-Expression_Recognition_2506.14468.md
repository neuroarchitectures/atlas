# Paper (Mao et al. 2025)

> Source: `https://arxiv.org/abs/2506.14468`

---

**MERba: Multi-Receptive Field MambaVision for Micro-Expression Recognition**

Xinglong Mao, Shifeng Liu, Sirui Zhao, Tong Xu, Hanchao Wang, Baozhi Jia, Enhong Chen

**Abstract**

Micro-expressions (MEs) are brief, involuntary facial movements that reveal genuine emotions, offering valuable insights for psychological assessment and criminal investigations. Despite significant progress in automatic ME recognition (MER), existing methods still struggle to simultaneously capture localized muscle activations and global facial dependencies, both essential for decoding subtle emotional cues. To address this challenge, we propose MERba, a hierarchical multi-receptive field architecture specially designed for MER, which incorporates a series of Local-Global Feature Integration stages. Within each stage, detailed intra-window motion patterns are captured using MERba Local Extractors, which integrate MambaVision Mixers with a tailored asymmetric multi-scanning strategy to enhance local spatial sensitivity. These localized features are then aggregated through lightweight self-attention layers that explicitly model inter-window relationships, enabling effective global context construction. Furthermore, to mitigate the challenge of high inter-class similarity among negative MEs, we introduce a Dual-Granularity Classification Module that decomposes the recognition task into a coarse-to-fine paradigm. Extensive experiments on three benchmark datasets demonstrate that MERba consistently outperforms existing methods, with ablation studies confirming the effectiveness of each proposed component.

---

- **arXiv ID**: `2506.14468`
- **Published**: 2025-06-17
- **Categories**: cs.HC
- **PDF**: https://arxiv.org/pdf/2506.14468
