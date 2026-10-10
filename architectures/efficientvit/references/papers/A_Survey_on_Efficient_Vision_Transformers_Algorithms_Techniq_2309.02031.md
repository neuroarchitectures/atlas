# A Survey on Efficient Vision Transformers: Algorithms, Techniques, and Performance Benchmarking

> Source: `https://arxiv.org/abs/2309.02031`

---

**Authors:** Lorenzo Papa, Paolo Russo, Irene Amerini, Luping Zhou

**Published:** 2024

**Concepts:** Benchmarking, Computer science, Artificial intelligence, Computer vision, Machine vision

## Abstract

Vision Transformer (ViT) architectures are becoming increasingly popular and widely employed to tackle computer vision applications. Their main feature is the capacity to extract global information through the self-attention mechanism, outperforming earlier convolutional neural networks. However, ViT deployment and performance have grown steadily with their size, number of trainable parameters, and operations. Furthermore, self-attention's computational and memory cost quadratically increases with the image resolution. Generally speaking, it is challenging to employ these architectures in real-world applications due to many hardware and environmental restrictions, such as processing and computational capabilities. Therefore, this survey investigates the most efficient methodologies to ensure sub-optimal estimation performances. More in detail, four efficient categories will be analyzed: compact architecture, pruning, knowledge distillation, and quantization strategies. Moreover, a new metric called Efficient Error Rate has been introduced in order to normalize and compare models' features that affect hardware devices at inference time, such as the number of parameters, bits, FLOPs, and model size. Summarizing, this paper first mathematically defines the strategies used to make Vision Transformer efficient, describes and discusses state-of-the-art methodologies, and analyzes their performances over different application scenarios. Toward the end of this paper, we also discuss open challenges and promising research directions.

**PDF:** https://arxiv.org/pdf/2309.02031