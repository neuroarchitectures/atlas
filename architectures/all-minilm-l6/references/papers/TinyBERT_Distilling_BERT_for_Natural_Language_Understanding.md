# TinyBERT: Distilling BERT for Natural Language Understanding

> Source: `https://doi.org/https://doi.org/10.18653/v1/2020.findings-emnlp.372`

---

**Authors:** Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu

**Published:** 2020

**Concepts:** Computer science, Transformer, Distillation, Inference, Benchmark (surveying)

## Abstract

Language model pre-training, such as BERT, has significantly improved the performances of many natural language processing tasks.However, pre-trained language models are usually computationally expensive, so it is difficult to efficiently execute them on resourcerestricted devices.To accelerate inference and reduce model size while maintaining accuracy, we first propose a novel Transformer distillation method that is specially designed for knowledge distillation (KD) of the Transformer-based models.By leveraging this new KD method, the plenty of knowledge encoded in a large "teacher" BERT can be effectively transferred to a small "student" Tiny-BERT.Then, we introduce a new two-stage learning framework for TinyBERT, which performs Transformer distillation at both the pretraining and task-specific learning stages.This framework ensures that TinyBERT can capture the general-domain as well as the task-specific knowledge in BERT.TinyBERT 4 1 with 4 layers is empirically effective and achieves more than 96.8% the performance of its teacher BERT BASE on GLUE benchmark, while being 7.5x smaller and 9.4x faster on inference.TinyBERT 4 is also significantly better than 4-layer state-of-the-art baselines on BERT distillation, with only ∼28% parameters and ∼31% inference time of them.Moreover, TinyBERT 6 with 6 layers performs on-par with its teacher BERT BASE .

**DOI:** https://doi.org/https://doi.org/10.18653/v1/2020.findings-emnlp.372
