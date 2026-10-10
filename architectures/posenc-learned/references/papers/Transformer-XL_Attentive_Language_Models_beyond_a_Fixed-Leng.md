# Transformer-XL: Attentive Language Models beyond a Fixed-Length Context

> Source: `https://doi.org/https://doi.org/10.18653/v1/p19-1285`

---

**Authors:** Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc Viet Le, Ruslan Salakhutdinov

**Published:** 2019

**Concepts:** Perplexity, Computer science, Language model, Transformer, Treebank

## Abstract

Transformers have a potential of learning longer-term dependency, but are limited by a fixed-length context in the setting of language modeling.We propose a novel neural architecture Transformer-XL that enables learning dependency beyond a fixed length without disrupting temporal coherence.It consists of a segment-level recurrence mechanism and a novel positional encoding scheme.Our method not only enables capturing longer-term dependency, but also resolves the context fragmentation problem.As a result, Transformer-XL learns dependency that is 80% longer than RNNs and 450% longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1,800+ times faster than vanilla Transformers during evaluation.Notably, we improve the state-ofthe-art results of bpc/perplexity to 0.99 on en-wiki8, 1.08 on text8, 18.3 on WikiText-103, 21.8 on One Billion Word, and 54.5 on Penn Treebank (without finetuning).When trained only on WikiText-103, Transformer-XL manages to generate reasonably coherent, novel text articles with thousands of tokens.Our code, pretrained models, and hyperparameters are available in both Tensorflow and PyTorch 1 .

**DOI:** https://doi.org/https://doi.org/10.18653/v1/p19-1285
