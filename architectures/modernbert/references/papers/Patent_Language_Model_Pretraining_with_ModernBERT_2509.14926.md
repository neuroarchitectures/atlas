# Paper (Yousefiramandi et al. 2025)

> Source: `https://arxiv.org/abs/2509.14926`

---

**Patent Language Model Pretraining with ModernBERT**

Amirhossein Yousefiramandi, Ciaran Cooney

**Abstract**

Transformer-based language models such as BERT have become foundational in NLP, yet their performance degrades in specialized domains like patents, which contain long, technical, and legally structured text. Prior approaches to patent NLP have primarily relied on fine-tuning general-purpose models or domain-adapted variants pretrained with limited data. In this work, we pretrain 3 domain-specific masked language models for patents, using the ModernBERT architecture and a curated corpus of over 60 million patent records. Our approach incorporates architectural optimizations, including FlashAttention, rotary embeddings, and GLU feed-forward layers. We evaluate our models on four downstream patent classification tasks. Our model, ModernBERT-base-PT, consistently outperforms the general-purpose ModernBERT baseline on three out of four datasets and achieves competitive performance with a baseline PatentBERT. Additional experiments with ModernBERT-base-VX and Mosaic-BERT-large demonstrate that scaling the model size and customizing the tokenizer further enhance performance on selected tasks. Notably, all ModernBERT variants retain substantially faster inference over - 3x that of PatentBERT - underscoring their suitability for time-sensitive applications. These results underscore the benefits of domain-specific pretraining and architectural improvements for patent-focused NLP tasks.

---

- **arXiv ID**: `2509.14926`
- **Published**: 2025-09-18
- **Categories**: cs.CL, cs.AI, cs.LG
- **PDF**: https://arxiv.org/pdf/2509.14926
