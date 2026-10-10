# Paper (Liu et al. 2024)

> Source: `https://doi.org/https://doi.org/10.1109/jstsp.2024.3506286`

---

**SemantiCodec: An Ultra Low Bitrate Semantic Audio Codec for General Sound**

Haohe Liu, Xuenan Xu, Yi Chen Yuan, Mengyue Wu, Wenwu Wang, Mark D. Plumbley

**Abstract**

Large languagemodels (LLMs) have significantly advanced audio processing through audio codecs that convert audio into discrete tokens, enabling the application of language modelling techniques to audio data. However, traditional codecs often operate at high bitrates or within narrow domains such as speech and lack the semantic clues required for efficient language modelling. Addressing these challenges, we introduce SemantiCodec, a novel codec designed to compress audio into fewer than a hundred tokens per second across diverse audio types, including speech, general sound, and music, without compromising quality. SemantiCodec features a dual-encoder architecture: a semantic encoder using a self-supervised pre-trained Audio Masked Autoencoder (AudioMAE), discretized using k-means clustering on extensive audio data, and an acoustic encoder to capture the remaining details. The semantic and acoustic encoder outputs are used to reconstruct audio via a diffusion-model-based decoder. SemantiCodec is presented in three variants with token rates of 25, 50, and 100 per second, supporting a range of ultra-low bit rates between 0.31 kbps and 1.40 kbps. Experimental results demonstrate that SemantiCodec significantly outperforms the state-of-the-art Descript codec on reconstruction quality. Our results also suggest that SemantiCodec contains significantly richer semantic information than all evaluated state-of-the-art audio codecs, even at significantly lower bitrates.

---

- **arXiv ID**: `N/A`
- **Year**: 2024
- **DOI**: https://doi.org/10.1109/jstsp.2024.3506286
- **PDF**: N/A
