# Paper (Chua et al. 2025)

> Source: `https://arxiv.org/abs/2501.08156`

---

**Are DeepSeek R1 And Other Reasoning Models More Faithful?**

James Chua, Owain Evans

**Abstract**

Language models trained to solve reasoning tasks via reinforcement learning have achieved striking results. We refer to these models as reasoning models. Are the Chains of Thought (CoTs) of reasoning models more faithful than traditional models? We evaluate three reasoning models (based on Qwen-2.5, Gemini-2, and DeepSeek-V3-Base) on an existing test of faithful CoT. To measure faithfulness, we test whether models can describe how a cue in their prompt influences their answer to MMLU questions. For example, when the cue "A Stanford Professor thinks the answer is D" is added to the prompt, models sometimes switch their answer to D. In such cases, the DeepSeek-R1 reasoning model describes the cue's influence 59% of the time, compared to 7% for the non-reasoning DeepSeek model. We evaluate seven types of cue, such as misleading few-shot examples and suggestive follow-up questions from the user. Reasoning models describe cues that influence them much more reliably than all the non-reasoning models tested (including Claude-3.5-Sonnet and GPT-4o). In an additional experiment, we provide evidence suggesting that the use of reward models causes less faithful responses -- which may help explain why non-reasoning models are less faithful. Our study has two main limitations. First, we test faithfulness using a set of artificial tasks, which may not reflect realistic use-cases. Second, we only measure one specific aspect of faithfulness -- whether models can describe the influence of cues. Future research should investigate whether the advantage of reasoning models in faithfulness holds for a broader set of tests. Still, we think this increase in faithfulness is promising for the explainability of language models.

---

- **arXiv ID**: `2501.08156`
- **Published**: 2025-01-14
- **Categories**: cs.LG
- **PDF**: https://arxiv.org/pdf/2501.08156
