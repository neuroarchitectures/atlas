# Paper (Chen et al. 2026)

> Source: `https://arxiv.org/abs/2604.14180`

---

**Internal Knowledge Without External Expression: Probing the Generalization Boundary of a Classical Chinese Language Model**

Jiuting Chen, Yuan Lian, Hao Wu, Tianqi Huang, Hiroshi Sasaki, Makoto Kouno, Jongil Choi

**Abstract**

We train a 318M-parameter Transformer language model from scratch on a curated corpus of 1.56 billion tokens of pure Classical Chinese, with zero English characters or Arabic numerals. Through systematic out-of-distribution (OOD) testing, we ask whether the model distinguishes known from unknown inputs, and whether it expresses that distinction in its generated text. We find a clear dissociation between internal and external uncertainty. Internally, the model exhibits a perplexity jump ratio of 2.39x between real and fabricated historical events (p = 8.9e-11, n = 92 per group), with semi-fabricated events (real figures + fictional actions) showing the highest perplexity (4.24x, p = 1.1e-16), demonstrating genuine factual encoding beyond syntactic pattern matching. Externally, however, the model never learns to express uncertainty: classical Chinese epistemic markers appear at lower rates for OOD questions (3.5%) than in-distribution ones (8.3%, p = 0.023), reflecting rhetorical conventions in the training data rather than genuine metacognition. We test both findings across three languages (Classical Chinese, English, Japanese), three writing systems, and eight models from 110M to 1.56B. The internal factual-encoding effect replicates in six of the eight models, emerging with scale (the two smallest Japanese models do not yet separate real from fabricated history), while the external absence of uncertainty expression holds across all eight. We further show that uncertainty expression frequency is determined entirely by training data conventions -- not epistemic states -- with Classical Chinese models showing a "humility paradox" (more hedging for known topics), while Japanese models almost never hedge. We argue that metacognitive expression -- the ability to say "I don't know" -- does not emerge from language modeling alone and requires explicit training signals such as RLHF.

---

- **arXiv ID**: `2604.14180`
- **Published**: 2026-03-31
- **Categories**: cs.CL, cs.AI
- **PDF**: https://arxiv.org/pdf/2604.14180
