# Paper (Grill et al. 2020)

> Source: `https://arxiv.org/abs/2006.07733v3`

---

Bootstrap your own latent: A new approach to self-supervised Learning

Jean-Bastien Grill
Florian Strub
Florent Altché
Corentin Tallec
Pierre H. Richemond
Elena Buchatskaya
Carl Doersch
Bernardo Avila Pires
Zhaohan Daniel Guo
Mohammad Gheshlaghi Azar
Bilal Piot
Koray Kavukcuoglu
Rémi Munos
Michal Valko

Published: 2020-06-13T22:35:21Z

Categories: cs.LG, cs.CV, stat.ML

PDF: https://arxiv.org/pdf/2006.07733v3

Abstract
We introduce Bootstrap Your Own Latent (BYOL), a new approach to self-supervised image representation learning. BYOL relies on two neural networks, referred to as online and target networks, that interact and learn from each other. From an augmented view of an image, we train the online network to predict the target network representation of the same image under a different augmented view. At the same time, we update the target network with a slow-moving average of the online network. While state-of-the art methods rely on negative pairs, BYOL achieves a new state of the art without them. BYOL reaches $74.3\%$ top-1 classification accuracy on ImageNet using a linear evaluation with a ResNet-50 architecture and $79.6\%$ with a larger ResNet. We show that BYOL performs on par or better than the current state of the art on both transfer and semi-supervised benchmarks. Our implementation and pretrained models are given on GitHub.
