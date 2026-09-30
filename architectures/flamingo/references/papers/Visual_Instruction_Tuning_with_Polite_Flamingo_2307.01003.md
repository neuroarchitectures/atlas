# Paper (Chen et al. 2023)

> Source: `https://arxiv.org/abs/2307.01003`

---

**Visual Instruction Tuning with Polite Flamingo**

Delong Chen, Jianfeng Liu, Wenliang Dai, Baoyuan Wang

**Abstract**

Recent research has demonstrated that the multi-task fine-tuning of multi-modal Large Language Models (LLMs) using an assortment of annotated downstream vision-language datasets significantly enhances their performance. Yet, during this process, a side effect, which we termed as the "multi-modal alignment tax", surfaces. This side effect negatively impacts the model's ability to format responses appropriately -- for instance, its "politeness" -- due to the overly succinct and unformatted nature of raw annotations, resulting in reduced human preference. In this paper, we introduce Polite Flamingo, a multi-modal response rewriter that transforms raw annotations into a more appealing, "polite" format. Polite Flamingo is trained to reconstruct high-quality responses from their automatically distorted counterparts and is subsequently applied to a vast array of vision-language datasets for response rewriting. After rigorous filtering, we generate the PF-1M dataset and further validate its value by fine-tuning a multi-modal LLM with it. Combined with novel methodologies including U-shaped multi-stage tuning and multi-turn augmentation, the resulting model, Clever Flamingo, demonstrates its advantages in both multi-modal understanding and response politeness according to automated and human evaluations.

---

- **arXiv ID**: `2307.01003`
- **Published**: 2023-07-03
- **Categories**: cs.CV, cs.CL
- **PDF**: https://arxiv.org/pdf/2307.01003
