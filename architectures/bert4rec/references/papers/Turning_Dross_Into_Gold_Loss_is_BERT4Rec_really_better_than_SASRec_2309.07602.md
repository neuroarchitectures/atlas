# Paper (Klenitskiy et al. 2023)

> Source: `https://arxiv.org/abs/2309.07602`

---

**Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?**

Anton Klenitskiy, Alexey Vasilev

**Abstract**

Recently sequential recommendations and next-item prediction task has become increasingly popular in the field of recommender systems. Currently, two state-of-the-art baselines are Transformer-based models SASRec and BERT4Rec. Over the past few years, there have been quite a few publications comparing these two algorithms and proposing new state-of-the-art models. In most of the publications, BERT4Rec achieves better performance than SASRec. But BERT4Rec uses cross-entropy over softmax for all items, while SASRec uses negative sampling and calculates binary cross-entropy loss for one positive and one negative item. In our work, we show that if both models are trained with the same loss, which is used by BERT4Rec, then SASRec will significantly outperform BERT4Rec both in terms of quality and training speed. In addition, we show that SASRec could be effectively trained with negative sampling and still outperform BERT4Rec, but the number of negative examples should be much larger than one.

---

- **arXiv ID**: `2309.07602`
- **Published**: 2023-09-14
- **Categories**: cs.IR, cs.AI, cs.LG
- **PDF**: https://arxiv.org/pdf/2309.07602
