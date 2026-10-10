# Paper (Madha et al. 2026)

> Source: `https://arxiv.org/abs/2608.00011`

---

**DLLM-TTS: Block Discrete Diffusion Language Model for Text-to-Speech Synthesis**

Wasim Madha, Nityanand Mathur, Hamees Sayed, Apoorv Singh, Sameer Khurana, Akshat Mandloi, Sudarshan Kamath

**Abstract**

Current text-to-speech systems face a trade-off: autoregres- sive codec language models produce highly intelligible speech but require large-scale models and training data and decode tokens sequentially, while non-autoregressive approaches im- prove speed at the cost of linguistic accuracy. We present DLLM-TTS, a framework that formulates TTS as conditional block discrete diffusion over X-Codec2 neural audio codec to- kens. The model decomposes sequences into blocks and applies masked diffusion within each block while processing blocks se- quentially, learning both local acoustic coherence and global text-speech alignment. During inference, parallel token pre- diction within blocks enables efficient generation with a real- time factor (RTF) of 0.15. A 0.6B-parameter model trained on 20K hours achieves competitive performance on the Seed- TTS-eval benchmark, demonstrating that block discrete diffu- sion language models enable practical and data-efficient speech synthesis with parallel generation.

---

- **arXiv ID**: `2608.00011`
- **Published**: 2026-06-19
- **Categories**: cs.CL, cs.AI
- **PDF**: https://arxiv.org/pdf/2608.00011
