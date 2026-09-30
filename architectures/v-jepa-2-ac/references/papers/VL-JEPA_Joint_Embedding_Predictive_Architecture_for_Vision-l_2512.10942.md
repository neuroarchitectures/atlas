# VL-JEPA: Joint Embedding Predictive Architecture for Vision-language

> Source: `https://arxiv.org/abs/2512.10942`

---

**Authors:** Delong Chen, Mustafa Shukor, Théo Moutakanni, Willy Chung, Yu, Jade, Tejaswi Kasarla, Bang, Yejin, Allen Bolourchi, Yann LeCun, Pascale Fung

**Published:** 2025

**Concepts:** Embedding, Computer science, Encoder, Discriminative model, Decoding methods

## Abstract

We introduce VL-JEPA, a vision-language model built on a Joint Embedding Predictive Architecture (JEPA). Instead of autoregressively generating tokens as in classical VLMs, VL-JEPA predicts continuous embeddings of the target texts. By learning in an abstract representation space, the model focuses on task-relevant semantics while abstracting away surface-level linguistic variability. In a strictly controlled comparison against standard token-space VLM training with the same vision encoder and training data, VL-JEPA achieves stronger performance while having 50% fewer trainable parameters. At inference time, a lightweight text decoder is invoked only when needed to translate VL-JEPA predicted embeddings into text. We show that VL-JEPA natively supports selective decoding that reduces the number of decoding operations by 2.85x while maintaining similar performance compared to non-adaptive uniform decoding. Beyond generation, the VL-JEPA's embedding space naturally supports open-vocabulary classification, text-to-video retrieval, and discriminative VQA without any architecture modification. On eight video classification and eight video retrieval datasets, the average performance VL-JEPA surpasses that of CLIP, SigLIP2, and Perception Encoder. At the same time, the model achieves comparable performance as classical VLMs (InstructBLIP, QwenVL) on four VQA datasets: GQA, TallyQA, POPE and POPEv2, despite only having 1.6B parameters.

**PDF:** https://arxiv.org/pdf/2512.10942