# STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning

> Source: `https://arxiv.org/abs/2310.09615`

---

**Authors:** Weipu Zhang, Gang Wang, Jian Feng Sun, Yetian Yuan, Gao Ming Huang

**Published:** 2023

**Concepts:** Reinforcement learning, Computer science, Sprite (computer graphics), Artificial intelligence, Transformer

## Abstract

Recently, model-based reinforcement learning algorithms have demonstrated remarkable efficacy in visual input environments. These approaches begin by constructing a parameterized simulation world model of the real environment through self-supervised learning. By leveraging the imagination of the world model, the agent's policy is enhanced without the constraints of sampling from the real environment. The performance of these algorithms heavily relies on the sequence modeling and generation capabilities of the world model. However, constructing a perfectly accurate model of a complex unknown environment is nearly impossible. Discrepancies between the model and reality may cause the agent to pursue virtual goals, resulting in subpar performance in the real environment. Introducing random noise into model-based reinforcement learning has been proven beneficial. In this work, we introduce Stochastic Transformer-based wORld Model (STORM), an efficient world model architecture that combines the strong sequence modeling and generation capabilities of Transformers with the stochastic nature of variational autoencoders. STORM achieves a mean human performance of $126.7\%$ on the Atari $100$k benchmark, setting a new record among state-of-the-art methods that do not employ lookahead search techniques. Moreover, training an agent with $1.85$ hours of real-time interaction experience on a single NVIDIA GeForce RTX 3090 graphics card requires only $4.3$ hours, showcasing improved efficiency compared to previous methodologies.

**PDF:** https://arxiv.org/pdf/2310.09615