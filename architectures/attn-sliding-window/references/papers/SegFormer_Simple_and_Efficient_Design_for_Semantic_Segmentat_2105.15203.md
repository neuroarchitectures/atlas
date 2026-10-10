# SegFormer: Simple and Efficient Design for Semantic Segmentation with\n Transformers

> Source: `https://arxiv.org/abs/2105.15203`

---

**Authors:** Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Álvarez, Ping Luo

**Published:** 2021

**Concepts:** Computer science, Transformer, Encoder, Segmentation, Robustness (evolution)

## Abstract

We present SegFormer, a simple, efficient yet powerful semantic segmentation\nframework which unifies Transformers with lightweight multilayer perception\n(MLP) decoders. SegFormer has two appealing features: 1) SegFormer comprises a\nnovel hierarchically structured Transformer encoder which outputs multiscale\nfeatures. It does not need positional encoding, thereby avoiding the\ninterpolation of positional codes which leads to decreased performance when the\ntesting resolution differs from training. 2) SegFormer avoids complex decoders.\nThe proposed MLP decoder aggregates information from different layers, and thus\ncombining both local attention and global attention to render powerful\nrepresentations. We show that this simple and lightweight design is the key to\nefficient segmentation on Transformers. We scale our approach up to obtain a\nseries of models from SegFormer-B0 to SegFormer-B5, reaching significantly\nbetter performance and efficiency than previous counterparts. For example,\nSegFormer-B4 achieves 50.3% mIoU on ADE20K with 64M parameters, being 5x\nsmaller and 2.2% better than the previous best method. Our best model,\nSegFormer-B5, achieves 84.0% mIoU on Cityscapes validation set and shows\nexcellent zero-shot robustness on Cityscapes-C. Code will be released at:\ngithub.com/NVlabs/SegFormer.\n

**PDF:** https://arxiv.org/pdf/2105.15203