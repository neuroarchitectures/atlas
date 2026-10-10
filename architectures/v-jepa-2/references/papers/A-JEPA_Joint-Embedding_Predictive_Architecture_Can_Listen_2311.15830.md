# A-JEPA: Joint-Embedding Predictive Architecture Can Listen

> Source: `https://arxiv.org/abs/2311.15830`

---

**Authors:** Zhengcong Fei, Mingyuan Fan, Junshi Huang

**Published:** 2023

**Concepts:** Computer science, Spectrogram, Encoder, Embedding, Speech recognition

## Abstract

This paper presents that the masked-modeling principle driving the success of large foundational vision models can be effectively applied to audio by making predictions in a latent space. We introduce Audio-based Joint-Embedding Predictive Architecture (A-JEPA), a simple extension method for self-supervised learning from the audio spectrum. Following the design of I-JEPA, our A-JEPA encodes visible audio spectrogram patches with a curriculum masking strategy via context encoder, and predicts the representations of regions sampled at well-designed locations. The target representations of those regions are extracted by the exponential moving average of context encoder, \emph{i.e.}, target encoder, on the whole spectrogram. We find it beneficial to transfer random block masking into time-frequency aware masking in a curriculum manner, considering the complexity of highly correlated in local time and frequency in audio spectrograms. To enhance contextual semantic understanding and robustness, we fine-tune the encoder with a regularized masking on target datasets, instead of input dropping or zero. Empirically, when built with Vision Transformers structure, we find A-JEPA to be highly scalable and sets new state-of-the-art performance on multiple audio and speech classification tasks, outperforming other recent models that use externally supervised pre-training.

**PDF:** https://arxiv.org/pdf/2311.15830