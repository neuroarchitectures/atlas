# TernaryBERT: Distillation-aware Ultra-low Bit BERT

> Source: `https://doi.org/https://doi.org/10.18653/v1/2020.emnlp-main.37`

---

**Authors:** Wei Zhang, Lu Hou, Yichun Yin, Lifeng Shang, Xiao Chen, Xin Jiang, Qun Liu

**Published:** 2020

**Concepts:** Computer science, Transformer, Leverage (statistics), Computation, Distillation

## Abstract

Transformer-based pre-training models like BERT have achieved remarkable performance in many natural language processing tasks.However, these models are both computation and memory expensive, hindering their deployment to resource-constrained devices.In this work, we propose TernaryBERT, which ternarizes the weights in a fine-tuned BERT model.Specifically, we use both approximation-based and loss-aware ternarization methods and empirically investigate the ternarization granularity of different parts of BERT.Moreover, to reduce the accuracy degradation caused by the lower capacity of low bits, we leverage the knowledge distillation technique (Jiao et al., 2019) in the training process.Experiments on the GLUE benchmark and SQuAD show that our proposed TernaryBERT outperforms the other BERT quantization methods, and even achieves comparable performance as the fullprecision model while being 14.9x smaller.

**DOI:** https://doi.org/https://doi.org/10.18653/v1/2020.emnlp-main.37
