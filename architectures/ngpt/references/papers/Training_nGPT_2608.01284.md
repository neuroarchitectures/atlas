# Paper (Loshchilov et al. 2026)

> Source: `https://arxiv.org/abs/2608.01284`

---

**Training nGPT**

Ilya Loshchilov, Boris Ginsburg

**Abstract**

The normalized Transformer (nGPT) realizes hyperspherical representation learning by constraining model parameter vectors and activation vectors to the unit hypersphere. In this paper, we describe a practical training recipe for nGPT and evaluate it on modern hybrid Mamba-2--Transformer Mixture-of-Experts (MoE) models. The recipe introduces Logit Gradient Preconditioning, Logarithmic Learning Rate Decay, GatedAdamW, angular update control, and optional exploration mechanisms. Compared with an unnormalized model of the same hybrid MoE architecture trained with AdamW, the 30B-total-parameter nGPT model reaches the same validation loss using approximately half as many training tokens. The recipe scales across the models considered, which contain up to 30B total parameters.

---

- **arXiv ID**: `2608.01284`
- **Published**: 2026-08-02
- **Categories**: cs.LG, cs.AI
- **PDF**: https://arxiv.org/pdf/2608.01284
