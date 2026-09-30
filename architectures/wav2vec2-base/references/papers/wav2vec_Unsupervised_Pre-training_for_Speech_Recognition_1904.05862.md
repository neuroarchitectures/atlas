# Paper (Schneider et al. 2019)

> Source: `https://arxiv.org/abs/1904.05862`

---

**wav2vec: Unsupervised Pre-training for Speech Recognition**

Steffen Schneider, Alexei Baevski, Ronan Collobert, Michael Auli

**Abstract**

We explore unsupervised pre-training for speech recognition by learning representations of raw audio. wav2vec is trained on large amounts of unlabeled audio data and the resulting representations are then used to improve acoustic model training. We pre-train a simple multi-layer convolutional neural network optimized via a noise contrastive binary classification task. Our experiments on WSJ reduce WER of a strong character-based log-mel filterbank baseline by up to 36% when only a few hours of transcribed data is available. Our approach achieves 2.43% WER on the nov92 test set. This outperforms Deep Speech 2, the best reported character-based system in the literature while using two orders of magnitude less labeled training data.

---

- **arXiv ID**: `1904.05862`
- **Published**: 2019-04-11
- **Categories**: cs.CL
- **PDF**: https://arxiv.org/pdf/1904.05862
