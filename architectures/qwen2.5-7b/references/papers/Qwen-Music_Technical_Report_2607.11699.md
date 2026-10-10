# Paper (Xu et al. 2026)

> Source: `https://arxiv.org/abs/2607.11699`

---

**Qwen-Music Technical Report**

Jin Xu, Kangdi Wang, Ruibin Yuan, Shun Lei, Xiong Wang, Xize Cheng, Xueyao Zhang, Yang Zhang, Yiheng Chen, Yongqi Wang et al.

**Abstract**

We introduce Qwen-Music, a music generation model that produces high-fidelity songs with complete vocals. It supports text-to-music generation from descriptions, lyrics, and musical attributes, and cover song generation with different styles and vocal characteristics. Qwen-Music comprises three components: Qwen-Music-Tokenizer, Qwen-Music-LLM, and Qwen-Music-Render. The tokenizer compresses audio into a 25 Hz single-codebook stream of Music Semantic Tokens that preserve semantic and melodic information. The LLM performs autoregressive modeling with a melody-token-based chain-of-thought (Melody-CoT) mechanism that plans melodies before full-song generation, improving musicality, structural coherence, and reference-melody preservation. The renderer enriches discrete semantic tokens with acoustic details to produce high-fidelity stereo waveforms. We train the LLM using a quality-aware pre-training curriculum followed by progressive post-training with supervised initialization, offline DPO, and online GSPO to improve musicality and instruction following. On 600 evaluation inputs, Qwen-Music achieves state-of-the-art results in 13 of 16 objective musicality and audio-quality metrics. Professional evaluators also prefer Qwen-Music over leading proprietary systems. For cover generation, Qwen-Music preserves reference melodies more accurately than Suno V5.5, Suno V5, and MiniMax Cover on the AI-generated reference set, and outperforms MiniMax Cover on most metrics on the real-world popular-song reference set.

---

- **arXiv ID**: `2607.11699`
- **Published**: 2026-07-13
- **Categories**: cs.SD
- **PDF**: https://arxiv.org/pdf/2607.11699
