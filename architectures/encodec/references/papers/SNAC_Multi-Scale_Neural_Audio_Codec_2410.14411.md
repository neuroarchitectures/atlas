# Paper (Siuzdak et al. 2024)

> Source: `https://arxiv.org/abs/2410.14411`

---

**SNAC: Multi-Scale Neural Audio Codec**

Hubert Siuzdak, Florian Grötschla, Luca A. Lanzendörfer

**Abstract**

Neural audio codecs have recently gained popularity because they can represent audio signals with high fidelity at very low bitrates, making it feasible to use language modeling approaches for audio generation and understanding. Residual Vector Quantization (RVQ) has become the standard technique for neural audio compression using a cascade of VQ codebooks. This paper proposes the Multi-Scale Neural Audio Codec, a simple extension of RVQ where the quantizers can operate at different temporal resolutions. By applying a hierarchy of quantizers at variable frame rates, the codec adapts to the audio structure across multiple timescales. This leads to more efficient compression, as demonstrated by extensive objective and subjective evaluations. The code and model weights are open-sourced at https://github.com/hubertsiuzdak/snac.

---

- **arXiv ID**: `2410.14411`
- **Published**: 2024-10-18
- **Categories**: cs.SD, cs.LG, eess.AS
- **PDF**: https://arxiv.org/pdf/2410.14411
