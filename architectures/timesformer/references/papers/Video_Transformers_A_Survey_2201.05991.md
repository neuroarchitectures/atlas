# Video Transformers: A Survey

> Source: `https://arxiv.org/abs/2201.05991`

---

**Authors:** Javier Selva, Anders Skaarup Johansen, Sérgio Escalera, Kamal Nasrollahi, Thomas Baltzer Moeslund, Albert Clapés

**Published:** 2023

**Concepts:** Computer science, Curse of dimensionality, Transformer, Machine learning, Artificial intelligence

## Abstract

Transformer models have shown great success handling long-range interactions, making them a promising tool for modeling video. However, they lack inductive biases and scale quadratically with input length. These limitations are further exacerbated when dealing with the high dimensionality introduced by the temporal dimension. While there are surveys analyzing the advances of Transformers for vision, none focus on an in-depth analysis of video-specific designs. In this survey, we analyze the main contributions and trends of works leveraging Transformers to model video. Specifically, we delve into how videos are handled at the input level first. Then, we study the architectural changes made to deal with video more efficiently, reduce redundancy, re-introduce useful inductive biases, and capture long-term temporal dynamics. In addition, we provide an overview of different training regimes and explore effective self-supervised learning strategies for video. Finally, we conduct a performance comparison on the most common benchmark for Video Transformers (i.e., action classification), finding them to outperform 3D ConvNets even with less computational complexity.

**PDF:** https://arxiv.org/pdf/2201.05991