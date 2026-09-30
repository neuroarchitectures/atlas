# Paper (Li et al. 2023)

> Source: `https://arxiv.org/abs/2301.04761`

---

**NarrowBERT: Accelerating Masked Language Model Pretraining and Inference**

Haoxin Li, Phillip Keung, Daniel Cheng, Jungo Kasai, Noah A. Smith

**Abstract**

Large-scale language model pretraining is a very successful form of self-supervised learning in natural language processing, but it is increasingly expensive to perform as the models and pretraining corpora have become larger over time. We propose NarrowBERT, a modified transformer encoder that increases the throughput for masked language model pretraining by more than $2\times$. NarrowBERT sparsifies the transformer model such that the self-attention queries and feedforward layers only operate on the masked tokens of each sentence during pretraining, rather than all of the tokens as with the usual transformer encoder. We also show that NarrowBERT increases the throughput at inference time by as much as $3.5\times$ with minimal (or no) performance degradation on sentence encoding tasks like MNLI. Finally, we examine the performance of NarrowBERT on the IMDB and Amazon reviews classification and CoNLL NER tasks and show that it is also comparable to standard BERT performance.

---

- **arXiv ID**: `2301.04761`
- **Published**: 2023-01-11
- **Categories**: cs.CL, cs.LG
- **PDF**: https://arxiv.org/pdf/2301.04761
