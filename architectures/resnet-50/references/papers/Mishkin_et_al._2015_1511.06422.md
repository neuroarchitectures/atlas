# Paper (Mishkin et al. 2015)

> Source: `https://arxiv.org/abs/1511.06422v7`

---

All you need is a good init

Dmytro Mishkin
Jiri Matas

Published: 2015-11-19T22:19:15Z

Categories: cs.LG

Comment: Published as a conference paper at ICLR 2016

PDF: https://arxiv.org/pdf/1511.06422v7

Abstract
Layer-sequential unit-variance (LSUV) initialization - a simple method for weight initialization for deep net learning - is proposed. The method consists of the two steps. First, pre-initialize weights of each convolution or inner-product layer with orthonormal matrices. Second, proceed from the first to the final layer, normalizing the variance of the output of each layer to be equal to one. Experiment with different activation functions (maxout, ReLU-family, tanh) show that the proposed initialization leads to learning of very deep nets that (i) produces networks with test accuracy better or equal to standard methods and (ii) is at least as fast as the complex schemes proposed specifically for very deep nets such as FitNets (Romero et al. (2015)) and Highway (Srivastava et al. (2015)). Performance is evaluated on GoogLeNet, CaffeNet, FitNets and Residual nets and the state-of-the-art, or very close to it, is achieved on the MNIST, CIFAR-10/100 and ImageNet datasets.
