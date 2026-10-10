# SSE-PT: Sequential Recommendation Via Personalized Transformer

> Source: `https://arxiv.org/abs/3313.34122`

---

**Authors:** Liwei Wu, Shuqing Li, Cho‐Jui Hsieh, James Sharpnack

**Published:** 2020

**Concepts:** Computer science, Personalization, Transformer, Regularization (linguistics), Artificial intelligence

## Abstract

Temporal information is crucial for recommendation problems because user preferences are naturally dynamic in the real world. Recent advances in deep learning, especially the discovery of various attention mechanisms and newer architectures in addition to widely used RNN and CNN in natural language processing, have allowed for better use of the temporal ordering of items that each user has engaged with. In particular, the SASRec model, inspired by the popular Transformer model in natural languages processing, has achieved state-of-the-art results. However, SASRec, just like the original Transformer model, is inherently an un-personalized model and does not include personalized user embeddings. To overcome this limitation, we propose a Personalized Transformer (SSE-PT) model, outperforming SASRec by almost 5% in terms of [email protected] on 5 real-world datasets. Furthermore, after examining some random users’ engagement history, we find our model not only more interpretable but also able to focus on recent engagement patterns for each user. Moreover, our SSE-PT model with a slight modification, which we call SSE-PT++, can handle extremely long sequences and outperform SASRec in ranking results with comparable training speed, striking a balance between performance and speed requirements. Our novel application of the Stochastic Shared Embeddings (SSE) regularization is essential to the success of personalization. Code and data are open-sourced at https://github.com/wuliwei9278/SSE-PT.

**PDF:** https://arxiv.org/pdf/3313.34122