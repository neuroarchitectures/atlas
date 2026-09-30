# Paper (Sun et al. 2023)

> Source: `https://arxiv.org/abs/2310.12751`

---

**Character-level Chinese Backpack Language Models**

Hao Sun, John Hewitt

**Abstract**

The Backpack is a Transformer alternative shown to improve interpretability in English language modeling by decomposing predictions into a weighted sum of token sense components. However, Backpacks' reliance on token-defined meaning raises questions as to their potential for languages other than English, a language for which subword tokenization provides a reasonable approximation for lexical items. In this work, we train, evaluate, interpret, and control Backpack language models in character-tokenized Chinese, in which words are often composed of many characters. We find that our (134M parameter) Chinese Backpack language model performs comparably to a (104M parameter) Transformer, and learns rich character-level meanings that log-additively compose to form word meanings. In SimLex-style lexical semantic evaluations, simple averages of Backpack character senses outperform input embeddings from a Transformer. We find that complex multi-character meanings are often formed by using the same per-character sense weights consistently across context. Exploring interpretability-through control, we show that we can localize a source of gender bias in our Backpacks to specific character senses and intervene to reduce the bias.

---

- **arXiv ID**: `2310.12751`
- **Published**: 2023-10-19
- **Categories**: cs.CL
- **PDF**: https://arxiv.org/pdf/2310.12751
