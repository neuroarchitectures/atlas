# SparseSwin: Swin transformer with sparse transformer block

> Source: `https://arxiv.org/abs/2309.05224`

---

**Authors:** Krisna Pinasthika, Blessius Sheldo Putra Laksono, Riyandi Banovbi Putera Irsal, Syifa’ Hukma Shabiyya, Novanto Yudistira

**Published:** 2024

**Concepts:** Transformer, Computer science, Electrical engineering, Engineering, Voltage

## Abstract

Advancements in computer vision research have put transformer architecture as the state-of-the-art in computer vision tasks . One of the known drawbacks of the transformer architecture is the high number of parameters, this can lead to a more complex and inefficient algorithm. This paper aims to reduce the number of parameters and in turn, made the transformer more efficient. We present Sparse Transformer (SparTa) Block, a modified transformer block with an addition of a sparse token converter that reduces the dimension of high-level features to the number of latent tokens. We implemented the SparTa Block within the Swin-T architecture (SparseSwin) to leverage Swin's proficiency in extracting low-level features and enhance its capability to extract information from high-level features while reducing the number of parameters. The proposed SparseSwin model outperforms other state-of-the-art models in image classification with an accuracy of 87.26%, 97.43%, and 85.35% on the ImageNet100, CIFAR10, and CIFAR100 datasets respectively. Despite its fewer parameters, the result highlights the potential of a transformer architecture using a sparse token converter with a limited number of tokens to optimize the use of the transformer and improve its performance. The code is available at https://github.com/KrisnaPinasthika/SparseSwin .

**PDF:** https://arxiv.org/pdf/2309.05224