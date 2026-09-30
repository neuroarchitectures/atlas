# Paper (Kadambari et al. 2020)

> Source: `https://arxiv.org/abs/2001.09407`

---

**Fast Graph Convolutional Recurrent Neural Networks**

Sai Kiran Kadambari, Sundeep Prabhakar Chepuri

**Abstract**

This paper proposes a Fast Graph Convolutional Neural Network (FGRNN) architecture to predict sequences with an underlying graph structure. The proposed architecture addresses the limitations of the standard recurrent neural network (RNN), namely, vanishing and exploding gradients, causing numerical instabilities during training. State-of-the-art architectures that combine gated RNN architectures, such as Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) with graph convolutions are known to improve the numerical stability during the training phase, but at the expense of the model size involving a large number of training parameters. FGRNN addresses this problem by adding a weighted residual connection with only two extra training parameters as compared to the standard RNN. Numerical experiments on the real 3D point cloud dataset corroborates the proposed architecture.

---

- **arXiv ID**: `2001.09407`
- **Published**: 2020-01-26
- **Categories**: eess.SP
- **PDF**: https://arxiv.org/pdf/2001.09407
