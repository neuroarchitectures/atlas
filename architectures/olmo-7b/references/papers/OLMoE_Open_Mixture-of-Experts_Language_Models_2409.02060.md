# Paper (Muennighoff et al. 2024)

> Source: `https://arxiv.org/abs/2409.02060`

---

**OLMoE: Open Mixture-of-Experts Language Models**

Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert et al.

**Abstract**

We introduce OLMoE, a fully open, state-of-the-art language model leveraging sparse Mixture-of-Experts (MoE). OLMoE-1B-7B has 7 billion (B) parameters but uses only 1B per input token. We pretrain it on 5 trillion tokens and further adapt it to create OLMoE-1B-7B-Instruct. Our models outperform all available models with similar active parameters, even surpassing larger ones like Llama2-13B-Chat and DeepSeekMoE-16B. We present various experiments on MoE training, analyze routing in our model showing high specialization, and open-source all aspects of our work: model weights, training data, code, and logs.

---

- **arXiv ID**: `2409.02060`
- **Published**: 2024-09-03
- **Categories**: cs.CL, cs.AI, cs.LG
- **PDF**: https://arxiv.org/pdf/2409.02060
