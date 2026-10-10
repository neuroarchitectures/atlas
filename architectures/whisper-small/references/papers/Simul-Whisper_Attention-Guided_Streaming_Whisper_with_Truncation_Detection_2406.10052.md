# Paper (Wang et al. 2024)

> Source: `https://arxiv.org/abs/2406.10052`

---

**Simul-Whisper: Attention-Guided Streaming Whisper with Truncation Detection**

Haoyu Wang, Guoqiang Hu, Guodong Lin, Wei-Qiang Zhang, Jian Li

**Abstract**

As a robust and large-scale multilingual speech recognition model, Whisper has demonstrated impressive results in many low-resource and out-of-distribution scenarios. However, its encoder-decoder structure hinders its application to streaming speech recognition. In this paper, we introduce Simul-Whisper, which uses the time alignment embedded in Whisper's cross-attention to guide auto-regressive decoding and achieve chunk-based streaming ASR without any fine-tuning of the pre-trained model. Furthermore, we observe the negative effect of the truncated words at the chunk boundaries on the decoding results and propose an integrate-and-fire-based truncation detection model to address this issue. Experiments on multiple languages and Whisper architectures show that Simul-Whisper achieves an average absolute word error rate degradation of only 1.46% at a chunk size of 1 second, which significantly outperforms the current state-of-the-art baseline.

---

- **arXiv ID**: `2406.10052`
- **Published**: 2024-06-14
- **Categories**: cs.SD, cs.CL, eess.AS
- **PDF**: https://arxiv.org/pdf/2406.10052
