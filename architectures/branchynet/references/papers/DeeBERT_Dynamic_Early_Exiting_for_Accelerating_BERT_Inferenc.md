# DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference

> Source: `https://doi.org/https://doi.org/10.18653/v1/2020.acl-main.204`

---

**Authors:** Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, Jimmy Lin

**Published:** 2020

**Concepts:** Inference, Computer science, Transformer, Language model, Redundancy (engineering)

## Abstract

Large-scale pre-trained language models such as BERT have brought significant improvements to NLP applications.However, they are also notorious for being slow in inference, which makes them difficult to deploy in realtime applications.We propose a simple but effective method, DeeBERT, to accelerate BERT inference.Our approach allows samples to exit earlier without passing through the entire model.Experiments show that DeeBERT is able to save up to ∼40% inference time with minimal degradation in model quality.Further analyses show different behaviors in the BERT transformer layers and also reveal their redundancy.Our work provides new ideas to efficiently apply deep transformer-based models to downstream tasks.

**DOI:** https://doi.org/https://doi.org/10.18653/v1/2020.acl-main.204
