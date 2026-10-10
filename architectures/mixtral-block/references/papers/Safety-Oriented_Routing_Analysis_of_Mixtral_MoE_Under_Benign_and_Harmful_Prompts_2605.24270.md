# Paper (Siddiky et al. 2026)

> Source: `https://arxiv.org/abs/2605.24270`

---

**Safety-Oriented Routing Analysis of Mixtral MoE Under Benign and Harmful Prompts**

Md Nurul Absar Siddiky

**Abstract**

Sparse mixture-of-experts (MoE) language models activate only a small subset of parameters for each token, making router behavior a central part of model computation. This paper studies routing behavior of Mixtral 8x7B-Instruct under benign and harmful prompts using two complementary signals: activation-based routing scores derived from expert selection frequencies and gradient-based scores derived from router-gate sensitivities. We analyze expert- and layer-level routing behavior and conduct expert-suppression interventions. The results show that activation-based expert usage is broad and long-tailed, whereas gradient-based importance is concentrated. At expert level, benign and harmful prompt groups remain close under both signals with modest separation. At layer level, activation-based routing is most selective around layers 8-15, while gradient-based importance is concentrated in final layers. Expert classification shows most experts are shared across benign and harmful prompts, though a limited subset shows clear group preference. Top-ranked expert sets show stronger benign-malicious overlap under gradient scores than activation scores, suggesting concentration on a common late-layer expert set. In intervention experiments, suppressing top five benign-dominant experts from activation scores reduces restricted responses from 24 to 14 over 100 prompts, while suppressing gradient-derived experts reduces them from 34 to 22 with fewer unintended reversals. Overall, safety-relevant routing in Mixtral is subtle, depth-dependent, and distributed rather than dominated by a fixed set of experts.

---

- **arXiv ID**: `2605.24270`
- **Published**: 2026-05-22
- **Categories**: cs.AI, cs.CR
- **PDF**: https://arxiv.org/pdf/2605.24270
