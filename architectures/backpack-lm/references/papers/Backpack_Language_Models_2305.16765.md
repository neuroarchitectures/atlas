# Paper (Hewitt et al. 2023)

> Source: `https://arxiv.org/abs/2305.16765`

---

**Backpack Language Models**

John Hewitt, John Thickstun, Christopher D. Manning, Percy Liang

**Abstract**

We present Backpacks: a new neural architecture that marries strong modeling performance with an interface for interpretability and control. Backpacks learn multiple non-contextual sense vectors for each word in a vocabulary, and represent a word in a sequence as a context-dependent, non-negative linear combination of sense vectors in this sequence. We find that, after training, sense vectors specialize, each encoding a different aspect of a word. We can interpret a sense vector by inspecting its (non-contextual, linear) projection onto the output space, and intervene on these interpretable hooks to change the model's behavior in predictable ways. We train a 170M-parameter Backpack language model on OpenWebText, matching the loss of a GPT-2 small (124Mparameter) Transformer. On lexical similarity evaluations, we find that Backpack sense vectors outperform even a 6B-parameter Transformer LM's word embeddings. Finally, we present simple algorithms that intervene on sense vectors to perform controllable text generation and debiasing. For example, we can edit the sense vocabulary to tend more towards a topic, or localize a source of gender bias to a sense vector and globally suppress that sense.

---

- **arXiv ID**: `2305.16765`
- **Published**: 2023-05-26
- **Categories**: cs.CL
- **PDF**: https://arxiv.org/pdf/2305.16765
