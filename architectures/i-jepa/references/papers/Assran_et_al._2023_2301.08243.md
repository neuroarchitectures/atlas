# Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture

This paper demonstrates an approach for learning highly semantic image representations without relying on hand-crafted data-augmentations. We introduce the Image-based Joint-Embedding Predictive Architecture (I-JEPA), a non-generative approach for self-supervised learning from images. The idea behind I-JEPA is simple: from a single context block, predict the representations of various target blocks in the same image. A core design choice to guide I-JEPA towards producing semantic representations is the masking strategy; specifically, it is crucial to (a) sample target blocks with sufficiently large scale (semantic), and to (b) use a sufficiently informative (spatially distributed) context block. Empirically, when combined with Vision Transformers, we find I-JEPA to be highly scalable. For instance, we train a ViT-Huge/14 on ImageNet using 16 A100 GPUs in under 72 hours to achieve strong downstream performance across a wide range of tasks, from linear classification to object counting and depth prediction.

## 1 Introduction

In computer vision, there are two common families of approaches for self-supervised learning from images: invariance-based methods [4, 37, 17, 24, 35, 18, 74, 10, 1] and generative methods [28, 8, 57, 36].

Invariance-based pretraining methods optimize an encoder to produce similar embeddings for two or more views of the same image [15, 20], with image views typically constructed using a set of hand-crafted data augmentations, such as random scaling, cropping, and color jittering [20], amongst others [35]. These pretraining methods can produce representations of a high semantic level [18, 4], but they also introduce strong biases that may be detrimental for certain downstream tasks or even for pretraining tasks with different data distributions [2]. Often, it is unclear how to generalize these biases for tasks requiring different levels of abstraction. For example, image classification and instance segmentation do not require the same invariances [11]. Additionally, it is not straightforward to generalize these image-specific augmentations to other modalities such as audio.

Cognitive learning theories have suggested that a driving mechanism behind representation learning in biological systems is the adaptation of an internal model to predict sensory input responses [59, 31]. This idea is at the core of self-supervised generative methods, which remove or corrupt portions of the input and learn to predict the corrupted content [67, 57, 36, 9, 71, 68]. In particular, mask-denoising approaches learn representations by reconstructing randomly masked patches from an input, either at the pixel or token level. Masked pretraining tasks require less prior knowledge than view-invariance approaches and easily generalize beyond the image modality [8]. However, the resulting representations are typically of a lower semantic level and underperform invariance-based pretraining in off-the-shelf evaluations (e.g., linear-probing) and in transfer settings with limited supervision for semantic classification tasks [4]. Consequently, a more involved adaptation mechanism (e.g., end-to-end fine-tuning) is required to reap the full advantage of these methods.

In this work, we explore how to improve the semantic level of self-supervised representations without using extra prior knowledge encoded through image transformations. To that end, we introduce a joint-embedding predictive architecture [48] for images (I-JEPA). An illustration of the method is provided in Figure 3. The idea behind I-JEPA is to predict missing information in an abstract representation space; e.g., given a single context block, predict the representations of various target blocks in the same image, where target representations are computed by a learned target-encoder network.

Compared to generative methods that predict in pixel/token space, I-JEPA makes use of abstract prediction targets for which unnecessary pixel-level details are potentially eliminated, thereby leading the model to learn more semantic features. Another core design choice to guide I-JEPA towards producing semantic representations is the proposed multi-block masking strategy. Specifically, we demonstrate the importance of predicting sufficiently large target blocks in the image, using an informative (spatially distributed) context block.

Through an extensive empirical evaluation, we demonstrate that:

- • I-JEPA learns strong off-the-shelf representations without the use of hand-crafted view augmentations (cf. Fig.1). I-JEPA outperforms pixel-reconstruction methods such as MAE [36] on ImageNet-1K linear probing, semi-supervised 1% ImageNet-1K, and semantic transfer tasks.
I-JEPA learns strong off-the-shelf representations without the use of hand-crafted view augmentations (cf. Fig.1). I-JEPA outperforms pixel-reconstruction methods such as MAE [36] on ImageNet-1K linear probing, semi-supervised 1% ImageNet-1K, and semantic transfer tasks.

- • I-JEPA is competitive with view-invariant pretraining approaches on semantic tasks and achieves better performance on low-level visions tasks such as object counting and depth prediction (Sections 5 and 6). By using a simpler model with less rigid inductive bias, I-JEPA is applicable to a wider set of tasks.
I-JEPA is competitive with view-invariant pretraining approaches on semantic tasks and achieves better performance on low-level visions tasks such as object counting and depth prediction (Sections 5 and 6). By using a simpler model with less rigid inductive bias, I-JEPA is applicable to a wider set of tasks.

- • I-JEPA is also scalable and efficient (Section 7). Pre-training a ViT-H/14 on ImageNet requires less than 1200 GPU hours, which is over 2.5×2.5\times faster than a ViT-S/16 pretrained with iBOT[79] and over 10×10\times more efficient than a ViT-H/14 pretrained with MAE. Predicting in representation space significantly reduces the total computation needed for self-supervised pretraining.
I-JEPA is also scalable and efficient (Section 7). Pre-training a ViT-H/14 on ImageNet requires less than 1200 GPU hours, which is over 2.5×2.5\times faster than a ViT-S/16 pretrained with iBOT[79] and over 10×10\times more efficient than a ViT-H/14 pretrained with MAE. Predicting in representation space significantly reduces the total computation needed for self-supervised pretraining.

## 2 Background

Self-supervised learning is an approach to representation learning in which a system learns to capture the relationships between its inputs. This objective can be readily described using the framework of Energy-Based Models (EBMs) [49] in which the self-supervised objective is to assign a high energy to incompatible inputs, and to assign a low energy to compatible inputs. Many existing generative and non-generative approaches to self-supervised learning can indeed be cast in this framework; see Figure 2.

Invariance-based pretraining can be cast in the framework of EBMs using a Joint-Embedding Architecture (JEA), which learns to output similar embeddings for compatible inputs, 𝒙,𝒚{\bm{x}},{\bm{y}}, and dissimilar embeddings for incompatible inputs; see Figure 2(a). In the context of image-based pretraining, compatible 𝒙,𝒚{\bm{x}},{\bm{y}} pairs are typically constructed by randomly applying hand-crafted data augmentations to the same input image [20].

The main challenge with JEAs is representation collapse, wherein the energy landscape is flat (i.e., the encoder produces a constant output regardless of the input). During the past few years, several approaches have been investigated to prevent representation collapse, such as contrastive losses that explicitly push apart embeddings of negative examples [15, 37, 24], non-contrastive losses that minimize the informational redundancy across embeddings [74, 10], and clustering-based approaches that maximize the entropy of the average embedding [18, 5, 4]. There are also heuristic approaches that leverage an asymmetric architectural design between the xx-encoder and yy-encoder to avoid collapse [24, 35, 8].

Reconstruction-based methods for self-supervised learning can also be cast in the framework of EBMs using Generative Architectures; see Figure 2(b). Generative Architectures learn to directly reconstruct a signal 𝒚{\bm{y}} from a compatible signal 𝒙{\bm{x}}, using a decoder network that is conditioned on an additional (possibly latent) variable 𝒛{\bm{z}} to facilitate reconstruction. In the context of image-based pretraining, one common approach in computer vision is to produce compatible 𝒙,𝒚{\bm{x}},{\bm{y}} pairs using masking [38, 9] where 𝒙{\bm{x}} is a copy of the image 𝒚{\bm{y}}, but with some of the patches masked. The conditioning variable 𝒛{\bm{z}} then corresponds to a set of (possibly learnable) mask and position tokens, that specifies to the decoder which image patches to reconstruct. Representation collapse is not a concern with these architectures as long as the informational capacity of 𝒛{\bm{z}} is low compared to the signal 𝒚{\bm{y}}.

As shown in Figure 2(c), Joint-Embedding Predictive Architectures [48] are conceptually similar to Generative Architectures; however, a key difference is that the loss function is applied in embedding space, not input space. JEPAs learn to predict the embeddings of a signal 𝒚{\bm{y}} from a compatible signal 𝒙{\bm{x}}, using a predictor network that is conditioned on an additional (possibly latent) variable 𝒛{\bm{z}} to facilitate prediction. Our proposed I-JEPA provides an instantiation of this architecture in the context of images using masking; see Figure 3.

In contrast to Joint-Embedding Architectures, JEPAs do not seek representations invariant to a set of hand-crafted data augmentations, but instead seek representations that are predictive of each other when conditioned on additional information 𝒛{\bm{z}}. However, as with Joint-Embedding Architectures, representation collapse is also a concern with JEPAs; we leverage an asymmetric architecture between the 𝒙{\bm{x}}- and 𝒚{\bm{y}}-encoders to avoid representation collapse.

## 3 Method

We now describe the proposed Image-based Joint-Embedding Predictive Architecture (I-JEPA), illustrated in Figure 3. The overall objective is as follows: given a context block, predict the representations of various target blocks in the same image. We use a Vision Transformer [63, 29] (ViT) architecture for the context-encoder, target-encoder, and predictor. A ViT is composed of a stack of transformer layers, each consisting of a self-attention [66] operation followed by a fully-connected MLP. Our encoder/predictor architecture is reminiscent of the generative masked autoencoders (MAE) [36] method. However, one key difference is that the I-JEPA method is non-generative and the predictions are made in representation space.

We first describe how we produce the targets in the I-JEPA framework: in I-JEPA, the targets correspond to the representations of image blocks. Given an input image 𝒚{\bm{y}}, we convert it into a sequence of NN non-overlapping patches, and feed this through the target-encoder fθ¯f_{\bar{\theta}} to obtain a corresponding patch-level representation 𝒔y={𝒔y1,…,𝒔yN}{\bm{s}}_{y}=\{{\bm{s}}_{y_{1}},\dots,{\bm{s}}_{y_{N}}\} where 𝒔yk{\bm{s}}_{y_{k}} is the representation associated with the kthk^{\text{th}} patch. To obtain the targets for our loss, we randomly sample MM (possibly overlapping) blocks from the target representations 𝒔y{\bm{s}}_{y}. We denote by BiB_{i} the mask corresponding of the ithi^{\text{th}} block and by 𝒔y​(i)={𝒔yj}j∈Bi{\bm{s}}_{y}(i)=\{{\bm{s}}_{y_{j}}\}_{j\in B_{i}} its patch-level representation. Typically, we set MM equal to 4, and sample the blocks with a random aspect ratio in the range (0.75,1.5)(0.75,1.5) and random scale in the range (0.15,0.2)(0.15,0.2). Note that the target blocks are obtained by masking the output of the target-encoder, not the input. This distinction is crucial to ensure target representations of a high semantic level; see, e.g., [8].

Recall, the goal behind I-JEPA is to predict the target block representations from a single context block. To obtain the context in I-JEPA, we first sample a single block 𝒙{\bm{x}} from the image with a random scale in the range (0.85,1.0)(0.85,1.0) and unit aspect ratio. We denote by BxB_{x} the mask associated with the context block 𝒙{\bm{x}}. Since the target blocks are sampled independently from the context block, there may be significant overlap. To ensure a non-trivial prediction task, we remove any overlapping regions from the context block. Figure 4 shows examples of various context and target blocks in practice. Next, the masked context block, 𝒙{\bm{x}}, is fed through the context encoder fθf_{\theta} to obtain a corresponding patch-level representation 𝒔x={𝒔xj}j∈Bx{\bm{s}}_{x}=\{{\bm{s}}_{x_{j}}\}_{j\in B_{x}}.

Given the output of the context encoder, 𝒔x{\bm{s}}_{x}, we wish to predict the MM target block representations 𝒔y​(1),…,𝒔y​(M){\bm{s}}_{y}(1),\ldots,{\bm{s}}_{y}(M). To that end, for a given target block 𝒔y​(i){\bm{s}}_{y}(i) corresponding to a target mask BiB_{i}, the predictor gϕ​(⋅,⋅)g_{\phi}(\cdot,\cdot) takes as input the output of the context encoder 𝒔x{\bm{s}}_{x} and a mask token for each patch we wish to predict, {𝒎j}j∈Bi\{{\bm{m}}_{j}\}_{j\in B_{i}}, and outputs a patch-level prediction 𝒔^y​(i)={𝒔^yj}j∈Bi=gϕ​(𝒔x,{𝒎j}j∈Bi)\hat{{\bm{s}}}_{y}(i)=\{\hat{{\bm{s}}}_{y_{j}}\}_{j\in B_{i}}=g_{\phi}({\bm{s}}_{x},\{{\bm{m}}_{j}\}_{j\in B_{i}}). The mask tokens are parameterized by a shared learnable vector with an added positional embedding. Since we wish to make predictions for MM target blocks, we apply our predictor MM times, each time conditioning on the mask tokens corresponding to the target-block locations we wish to predict, and obtain predictions 𝒔^y​(1),…,𝒔^y​(M)\hat{{\bm{s}}}_{y}(1),\ldots,\hat{{\bm{s}}}_{y}(M).

The loss is simply the average L2L_{2} distance between the predicted patch-level representations 𝒔^y​(i)\hat{{\bm{s}}}_{y}(i) and the target patch-level representation 𝒔y​(i){\bm{s}}_{y}(i); i.e.,

|  | 1M​∑i=1MD⁡(𝒔^y​(i),𝒔y​(i))=1M​∑i=1M∑j∈Bi∥𝒔^yj−𝒔yj∥22.\frac{1}{M}\sum^{M}_{i=1}{D}\left(\hat{{\bm{s}}}_{y}(i),{\bm{s}}_{y}(i)\right)=\frac{1}{M}\sum^{M}_{i=1}\sum_{j\in B_{i}}\lVert\hat{{\bm{s}}}_{y_{j}}-{\bm{s}}_{y_{j}}\rVert^{2}_{2}. |  |
|---|---|---|

The parameters of the predictor, ϕ\phi, and context encoder, θ\theta, are learned through gradient-based optimization, while the parameters of the target encoder θ¯\bar{\theta} are updated via an exponential moving average of the context-encoder parameters. The use of an exponential moving average target-encoder has proven essential for training JEAs with Vision Transformers [25, 79, 18], we find the same to be true for I-JEPA.

## 4 Related Work

A long line of work has explored visual representation learning by predicting the values of missing or corrupted sensory inputs. Denoising autoencoders use random noise as input corruption [67]. Context encoders regress an entire image region based on its surrounding [57]. Other works cast image colorization as a denoising task [77, 46, 47].

The idea of image denoising has recently been revisited in the context of masked image modelling [36, 71, 9], where a Vision Transformer [29] is used to reconstruct missing input patches. The work on Masked Autoencoders (MAE) [36] proposed an efficient architecture that only requires the encoder to process visible image patches. By reconstructing missing patches in pixels space, MAE achieves strong performance when fine-tuned end-to-end on large labeled datasets and exhibits good scaling properties. BEiT [9] predicts the value of missing patches in a tokenized space; specifically, tokenizing image patches using a frozen discreteVAE, which is trained on a dataset containing 250 million images [58]. Yet, pixel-level pre-training has been shown to outperform BEiT for fine-tuning [36]. Another work, SimMIM [71], explores reconstruction targets based on the classic Histogram of Gradients [27] feature space, and demonstrates some advantage over pixel space reconstruction. Different from those works, our representation space is learned during training through a Joint-Embedding Predictive Architecture. Our goal is to learn semantic representations that do not require extensive fine-tuning on downstream tasks.

| Method | Arch. | Epochs | Top-1 |
|---|---|---|---|
| Methods without view data augmentations |  |  |  |
| data2vec [8] | ViT-L/16 | 1600 | 77.3 |
| MAE [36] | ViT-B/16 | 1600 | 68.0 |
| ViT-L/16 | 1600 | 76.0 |  |
| ViT-H/14 | 1600 | 77.2 |  |
| CAE [22] | ViT-B/16 | 1600 | 70.4 |
| ViT-L/16 | 1600 | 78.1 |  |
| I-JEPA | ViT-B/16 | 600 | 72.9 |
| ViT-L/16 | 600 | 77.5 |  |
| ViT-H/14 | 300 | 79.3 |  |
|  | ViT-H/1644816_{448} | 300 | 81.1 |
| Methods using extra view data augmentations |  |  |  |
| SimCLR v2 [21] | RN152 (2×2\times) | 800 | 79.1 |
| DINO [18] | ViT-B/8 | 300 | 80.1 |
| iBOT [79] | ViT-L/16 | 250 | 81.0 |

Closest to our work is data2vec [8] and Context Autoencoders [25]. The data2vec method learns to predict the representation of missing patches computed through an online target encoder; by avoiding handcrafted augmentations, the method can be applied to diverse modalities with promising results in vision, text and speech. Context Autoencoders use an encoder/decoder architecture optimized via the sum of a reconstruction loss and an alignment constraint, which enforces predictability of missing patches in representation space. Compared to these methods, I-JEPA exhibits significant improvements in computational efficiency and learns more semantic off-the-shelf representations. Concurrent to our work, data2vec-v2 [7] explores efficient architectures for learning with various modalities.

We also compare I-JEPA with various methods based on joint-embedding architectures; e.g., DINO [18], MSN [4] and iBOT [79]. Theses methods rely on hand-crafted data augmentations during pretraining to learn semantic image representations. The work on MSN [4], uses masking as an additional data-augmentation during pretraining, while iBOT combines a data2vec-style patch-level reconstruction loss with the DINO view-invariance loss. Common to these approaches is the need to process multiple user-generated views of each input image, thereby hindering scalability. By contrast, I-JEPA only requires processing a single view of each image. We find that a ViT-Huge/14 trained with I-JEPA requires less computational effort than a ViT-Small/16 trained with iBOT.

## 5 Image Classification

| Method | Arch. | Epochs | Top-1 |
|---|---|---|---|
| Methods without view data augmentations |  |  |  |
| data2vec [8] | ViT-L/16 | 1600 | 73.3 |
| MAE [36] | ViT-L/16 | 1600 | 67.1 |
| ViT-H/14 | 1600 | 71.5 |  |
| I-JEPA | ViT-L/16 | 600 | 69.4 |
| ViT-H/14 | 300 | 73.3 |  |
| ViT-H/1644816_{448} | 300 | 77.3 |  |
| Methods using extra view data augmentations |  |  |  |
| iBOT [79] | ViT-B/16 | 400 | 69.7 |
| DINO [18] | ViT-B/8 | 300 | 70.0 |
| SimCLR v2 [35] | RN151 (2×2\times) | 800 | 70.2 |
| BYOL [35] | RN200 (2×2\times) | 800 | 71.2 |
| MSN [4] | ViT-B/4 | 300 | 75.7 |

To demonstrate that I-JEPA learns high-level representations without relying on hand-crafted data-augmentations, we report results on various image classification tasks using the linear probing and partial fine-tuning protocols. In this section, we consider self-supervised models that have been pretrained on the ImageNet-1K dataset [60]. Pretraining and evaluation implementation details are described in the Appendix A. All I-JEPA models are trained at resolution 224×224224\times 224 pixels, unless stated otherwise.

Table 1 shows performance on the common ImageNet-1K linear-evaluation benchmark. After self-supervised pretraining, the model weights are frozen and a linear classifier is trained on top using the full ImageNet-1K training set. Compared to popular methods such as Masked Autoencoders (MAE) [36], Context Autoencoders (CAE) [22], and data2vec [8], which also do not rely on extensive hand-crafted data-augmentations during pretraining, we see that I-JEPA significantly improves linear probing performance, while using less computational effort (see section 7). By leveraging the improved efficiency of I-JEPA, we can train larger models that outperform the best CAE model while using a fraction of the compute. I-JEPA also benefits from scale; in particular, a ViT-H/16 trained at resolution 448×448448\times 448 pixels matches the performance of view-invariant approaches such as iBOT [79], despite avoiding the use of hand-crafted data-augmentations.

Table 2 shows performance on the 1% ImageNet benchmark. Here the idea is to adapt the pretrained models for ImageNet classification using only 1% of the available ImageNet labels, corresponding to roughly 12 or 13 images per class. Models are adapted via fine-tuning or linear-probing, depending on whichever works best for each respective method. I-JEPA outperforms MAE while requiring less pretraining epochs when using a similar encoder architecture. I-JEPA, using a ViT-H/14 architecture, matches the performance of a ViT-L/16 pretrained with data2vec [8], while using significantly less computational effort (see Section 7). By increasing the image input resolution, I-JEPA outperforms previous methods including joint-embedding methods that do leverage extra hand-crafted data-augmentations during pretraining, such as MSN [4], DINO [17], and iBOT [79].

Table 3 shows performance on various downstream image classification tasks using a linear probe. I-JEPA significantly outperforms previous methods that do not use augmentations (MAE and data2vec), and decreases the gap with the best view-invariance-based methods, which leverage hand-crafted data augmentations during pretraining, even surpassing the popular DINO [18] on CIFAR100 and Place205 with a linear probe.

| Method | Arch. | CIFAR100 | Places205 | iNat18 |
|---|---|---|---|---|
| Methods without view data augmentations |  |  |  |  |
| data2vec [8] | ViT-L/16 | 81.6 | 54.6 | 28.1 |
| MAE [36] | ViT-H/14 | 77.3 | 55.0 | 32.9 |
| I-JEPA | ViT-H/14 | 87.5 | 58.4 | 47.6 |
| Methods using extra view data augmentations |  |  |  |  |
| DINO [18] | ViT-B/8 | 84.9 | 57.9 | 55.9 |
| iBOT [79] | ViT-L/16 | 88.3 | 60.4 | 57.3 |

## 6 Local Prediction Tasks

As demonstrated in Section 5, I-JEPA learns semantic image representations that significantly improve the downstream image classification performance of previous methods, such as MAE and data2vec. Additionally, I-JEPA benefits from scale and can close the gap, and even surpass, view-invariance based methods that leverage extra hand-crafted data augmentations. In this section, we find that I-JEPA also learns local image features and surpasses view-invariance based methods on low-level and dense prediction tasks, such as object counting and depth prediction.

Table 4 shows performance on various low-level tasks using a linear probe. After pretraining, the encoder weights are frozen and a linear model is trained on top to perform object-counting and depth prediction on the Clevr dataset [43]. Compared to view-invariance methods such as DINO and iBOT, the I-JEPA method effectively captures low-level image features during pretraining and outperforms them in object counting (Clevr/Count) and (by a large margin) depth prediction (Clevr/Dist).

| Method | Arch. | Clevr/Count | Clevr/Dist |
|---|---|---|---|
| Methods without view data augmentations |  |  |  |
| data2vec [8] | ViT-L/16 | 85.3 | 71.3 |
| MAE [36] | ViT-H/14 | 90.5 | 72.4 |
| I-JEPA | ViT-H/14 | 86.7 | 72.4 |
| Methods using extra data augmentations |  |  |  |
| DINO [18] | ViT-B/8 | 86.6 | 53.4 |
| iBOT [79] | ViT-L/16 | 85.7 | 62.8 |

| Pretrain | Arch. | CIFAR100 | Place205 | INat18 | Clevr/Count | Clevr/Dist |
|---|---|---|---|---|---|---|
| IN1k | ViT-H/14 | 87.5 | 58.4 | 47.6 | 86.7 | 72.4 |
| IN22k | ViT-H/14 | 89.5 | 57.8 | 50.5 | 88.6 | 75.0 |
| IN22k | ViT-G/16 | 89.5 | 59.1 | 55.3 | 86.7 | 73.0 |

## 7 Scalability

I-JEPA is highly scalable compared to previous approaches. Figure 5 shows semi-supervised evaluation on 1% ImageNet-1K as a function of GPU hours. I-JEPA requires less compute than previous methods and achieves strong performance without relying on hand-crafted data-augmentations. Compared to reconstruction-based methods, such as MAE, which directly use pixels as targets, I-JEPA introduces extra overhead by computing targets in representation space (about 7% slower time per iteration). However, since I-JEPA converges in roughly 5×5\times fewer iterations, we still see significant compute savings in practice. Compared to view-invariance based methods, such as iBOT, which rely on hand-crafted data augmentations to create and process multiple views of each image, I-JEPA also runs significantly faster. In particular, a huge I-JEPA model (ViT-H/14) requires less compute than a small iBOT model (ViT-S/16).

We also find I-JEPA to benefit from pretraining with larger datasets. Table 5 shows transfer learning performance on semantic and low level tasks when increasing the size of the pretraining dataset (IN1K versus IN22K). Transfer learning performance on these conceptually different tasks improves when pretraining on a larger more diverse dataset.

Table 5 also shows that I-JEPA benefit from larger model size when pretraining on IN22K. Pretraining a ViT-G/16 significantly improves the downstream performances on image classification tasks such as Place205 and INat18 compared to a ViT-H/14 model, but does not improve performance on low-level downstream tasks — the ViT-G/16 uses larger input patches, which can be detrimental for the local prediction tasks.

|  | Targets | Context |  |  |  |
|---|---|---|---|---|---|
| Mask | Type | Freq. | Type | Avg. Ratio∗ | Top-1 |
| multi-block | Block(0.15,0.2)(0.15,0.2) | 4 | Block(0.85,1.0)(0.85,1.0) ×\times Complement | 0.25 | 54.2 |
| rasterized | Quadrant | 3 | Complement | 0.25 | 15.5 |
| block | Block(0.6)(0.6) | 1 | Complement | 0.4 | 20.2 |
| random | Random(0.6) | 1 | Complement | 0.4 | 17.6 |
| ∗Avg. Ratio is the average number of patches in the context block relative to the total number of patches in the image. |  |  |  |  |  |

## 8 Predictor Visualizations

The role of the predictor in I-JEPA is to take the output of the context encoder and, conditioned on positional mask tokens, to predict the representations of a target black at the location specified by the mask tokens. One natural question is whether the predictor conditioned on the positional mask tokens is learning to correctly capture positional uncertainty in the target. To qualitatively investigate this question, we visualize the outputs of the predictor. We use the following visualization approach to enable the research community to independently reproduce our findings. After pretraining, we freeze the context-encoder and predictor weights, and train a decoder following the RCDM framework [13] to map the average-pool of the predictor outputs back to pixel space. Figure 6 shows decoder outputs for various random seeds. Qualities that are common across samples represent information that is contained in the average-pooled predictor representation. The I-JEPA predictor correctly captures positional uncertainty and produces high-level object parts with the correct pose (e.g., back of the bird and top of the car).

## 9 Ablations

| Targets | Arch. | Epochs | Top-1 |
|---|---|---|---|
| Target-Encoder Output | ViT-L/16 | 500 | 66.9 |
| Pixels | ViT-L/16 | 800 | 40.7 |

Table 7 compares low-shot performance on 1% ImageNet-1K using a linear probe when the loss is computed in pixel-space versus representation space. We conjecture that a crucial component of I-JEPA is that the loss is computed entirely in representation space, thereby giving the target encoder the ability to produce abstract prediction targets, for which irrelevant pixel-level details are eliminated. From Table 7, it is clear that predicting in pixel-space leads to a significant degradation in the linear probing performance.

Table 6 compare our multi-block masking with other masking strategies such as rasterized masking, where the image is split into four large quadrants, and the goal is to use one quadrant as a context to predict the other three quadrants, and the traditional block and random masking typically used in reconstruction-based methods. In block masking, the target is a single image block and the context is the image complement. In random masking, the target is a set of random patches and the context is the image complement. Note that there is no overlap between the context and target blocks in all considered strategies. We find multi-block masking helpful for guiding I-JEPA to learning semantic representations. Additional ablations on multi-block masking can be found in Appendix C.

## 10 Conclusion

We proposed I-JEPA, a simple and efficient method for learning semantic image representations without relying on hand-crafted data augmentations. We show that by predicting in representation space, I-JEPA converges faster than pixel reconstruction methods and learns representations of a high semantic level. In contrast to view-invariance based methods, I-JEPA highlights a path for learning general representations with joint-embedding architectures, without relying on hand-crafted view augmentations.

## Appendix A Implementation Details

### A.1 Pretraining

For I-JEPA pretraining, we use Vision Transformer [29] (ViT) architectures for the context-encoder, target-encoder, and the predictor. While the context-encoders and target-encoders correspond to standard ViT architectures, the predictor is designed as a light-weight (narrow) ViT architecture. Specifically, we fix the embedding dimension of the predictor to 384, while keeping the number of self-attention heads equal to that of the backbone context-encoder. For the smaller ViT-B/16 context-encoder, we set the depth of the predictor to 6. For ViT-L/16, ViT-H/16, and ViT-H/14 context-encoders, we set the depth of the predictor to 12. Finally, the ViT-G/16 uses a predictor of depth 16. I-JEPA is pretrained without a [cls] token. We use the target-encoder for evaluation and average pool its output to produce a global image representation.

We use AdamW [51] to optimize the context-encoder and predictor weights. Our default batch-size is 2048, and the learning rate is linearly increased from 10−410^{-4} to 10−310^{-3} during the first 15 epochs of pretraining, and decayed to 10−610^{-6} following a cosine schedule thereafter. Following [18, 4], the weight-decay is linearly increased from 0.040.04 to 0.40.4 throughout pretraining. The target-encoder weights are identical to the context-encoder weights at initialization, and updated via an exponential moving average thereafter [61, 37, 23, 35, 18, 4]. We use a momentum value of 0.9960.996, and linearly increase this value to 1.01.0 throughout pretraining, following [18, 4].

By default, we sample 44 possibly overlapping target blocks masks with random scale in the range (015,0.2)(015,0.2) and aspect ratio in the range (0.75,1.5)(0.75,1.5). We sample 11 context block mask with random scale in the range (0.85,1.0)(0.85,1.0) and unit aspect ratio. We subsequently eliminate any regions in the context block mask that overlap with any of the 44 target block masks. The context-block mask and target-block masks are sampled independently for each image in the mini-batch. To ensure efficient batch processing, we restrict the size of all context masks on a co-located GPU to be identical. Similarly, we restrict the size of all target masks on a co-located GPU to be identical. The mask-sampler is efficiently implemented in only a few lines of code in PyTorch [56] using a batch-collator function, which runs in the data loader processes. In short, in each iteration, the data loader returns a mini-batch of images and a set of context and target masks for each image, identifying the patch indices to keep for the context and target views.

### A.2 Downstream Tasks

When evaluating methods such as iBOT [79], DINO [18] or MAE [36], which leverage Vision Transformers [29] with an additional [cls] token, we use the default configurations of VISSL [34] to evaluate all the models on iNaturalist18 [65], CIFAR100 [45], Clevr/Count [42, 75], Clevr/Dist [42, 75], and Places205 [78]. We freeze the encoder and return the best number among the following representations: 1) the [cls] token representation of the last layer, 2) the concatenation of the last 44 layers of the [cls] token. For each representation, we try two different heads: 1) a linear head, or 2) a linear head preceded by a batch normalization, and return the best number. We use the default data augmentations of VISSL [34]: random resize cropping and horizontal flipping, with the exception of Clevr/Count and Clevr/Dist, where we only use center crop and horizontal flipping, as random cropping interferes with the capability of counting objects and estimating distance, removing critical objects from the scene. For CIFAR100, we resize the images to 224×224224\times 224 pixels, so as to keep the number of patches equal to that used during pretraining.

Because our I-JEPA implementation uses Vision Transformer architectures without a [cls] token, we adapt the default VISSL evaluation recipe to utilize the average-pooled patch representation instead of the [cls] token. We therefore report the best linear evaluation number among the following representations: 1) the average-pooled patch representation of the last layer, 2) the concatenation of the last 44 layers of the average-pooled patch representations. We otherwise keep the linear-probing recipe identical.

To evaluate the I-JEPA on ImageNet [60], we adapt the VISSL recipe to use average pooled representations instead of the [cls] token. Following MAE [36], we use the LARS [72] optimizer with a batch-size of 1638416384, and train the linear probe for 50 epochs. We use a learning rate with a step-wise decay, dividing it by a factor of 1010 every 1515 epochs, and sweep three different reference learning rates [0.01,0.05,0.001][0.01,0.05,0.001], and two weight decay values [0.0005,0.0][0.0005,0.0].

To evaluate our model on the ImageNet-1% low-shot task, we adapt the fine-tuning protocol of MAE [36].We fine-tune our ViT-L/H models for 50 epochs on ImageNet-1% with the AdamW optimizer and a cosine learning rate scheduler. We use a batch size of 512, a learning rate layer decay of 0.750.75 and 0.10.1 label smoothing. We use the default randaugment data-augmentations as in MAE. In contrast to the fine-tuning done with MAE, we do not use mixup, cutmix, random erasing or drop path. For the I-JEPA, we use a learning rate /weight decay of 3e-5/5e-2 for the ViT-L/16, 3e-5/4e-1 for the ViT-H/14 and 3e-5/4e-1 for the ViT-H/16448. Similar fine-tuning strategy for low-shot learning has been explored by Semi-VIT in the context of semi-supervised learning [16].

## Appendix B Broader Related Work

Self-supervised learning of visual representations with joint-embedding architectures is an active line of research [69, 37, 24, 35, 23, 18, 10, 79, 12, 54, 3]. These approaches train a pair of encoders to output similar embeddings for two or more views of the same image. To avoid pathological solution, many popular joint-embedding approaches use explicit regularization [20, 18, 10, 5] or architectural constraints [35, 24]. Collapse-prevention based on architectural constraints leverage specific network design choices to avoid collapse, for example, by stopping the gradient flow in one of the joint-embedding branches [20], using a momentum encoder in one of the joint-embedding branches [35], or using an asymmetric prediction head [35, 20, 8]. Recent work [62] attempts to theoretically understand (in certain simplified settings) how joint-embedding methods with architectural constraints avoid representation collapse without explicit regularization.

Typical regularization-based approaches to collapse prevention in joint-embedding architectures try to maximize the volume of space occupied by the representations. This is often motivated through the InfoMax [52] principle. Indeed, a longstanding conviction in unsupervised representation learning is that the resulting representations should be both maximally informative about the inputs, while also satisfying certain simplicity constraints [50, 33]. The former objective is often referred to as the information-maximization principle (InfoMax), while the latter is sometimes referred to as the parsimony principle [52]. Such approaches to representation learning have been proposed for decades (e.g., [14]), where, historically, simplicity constraints were enforced by encouraging the learned representations to be sparse, low-dimensional, or disentangled, i.e., the individual dimensions of the representation vector should be statistically independent [33]. Modern approaches enforce the simplicity constraints coupled with InfoMax regularization through self-supervised loss terms [40, 64, 6, 44, 41, 55]. One example is the widespread view-invariance penalty [53], often coupled with with independence [74, 10] or low-dimensionality constraints, e.g., by projecting representations on the unit hypersphere [20, 37, 35]. However, despite its proliferation, there have also been many criticisms of the InfoMax principle, especially since it is does not discriminate between different types of information (e.g, noise and semantics) [2]. Indeed, the sets of features we wish the model to capture are not always those with the highest marginal entropy (maximal information content).

Orthogonal to the contributions of invariance-based pretraining, another line of work attempts to learn representations by artificially masking parts of the input and training a network to reconstruct the hidden content [67]. Autoregressive models, and denoising autoencoders in particular, predict clean visual inputs from noisy views [19, 67, 36, 9, 8]. Typically, the goal is to predict missing inputs at a pixel level [29, 36, 70], or at a patch token-level, using a tokenizer [9, 68]. While these works demonstrate impressive scalability, they usually learn features at a low-level of semantic abstraction compared to joint-embedding approaches [4].

More recently, a set of approaches attempt to combine both joint-embedding architectures and reconstruction based approaches [30], wherein they combine an invariance pretraining loss with a patch-level reconstruction loss, as in the iBOT method [79]. Since view-invariance based approaches are typically biased towards learning global image representations, thereby limiting their applicability to other computer vision tasks, the idea is that adding local loss terms can improve performance on other popular tasks in computer vision [26, 32, 11]. The framework of contrastive predictive coding [55] is also closely related to this line of work on local loss terms. In the context of images [39], here the idea is to use a contrastive objective combined with a convolutional network to discriminate between overlapping image patch representations. Specifically, the goal is to encourage the representations of an image patch to be predictable of the image patches directly below it, while pushing away the representations of other patch views. In contrast to that work, the proposed I-JEPA method is non-contrastive and does not seek to discriminate between image patches. Rather, the goal is to predict the representations of various target blocks from a single context block. This is achieved with a Joint-Embedding Predictive Architecture, using a predictor network that is conditioned on positional embeddings corresponding to the location of the target block in the image. Qualitative experiments in Section 8 show that the predictor network in our architecture learns to correctly perform this local-to-local region feature mapping, and learns to correctly capture positional uncertainty in the image.

## Appendix C Additional Ablations

This section follows the same experimental protocol as Section 9. We report the result of a linear probe with a frozen backbone, trained on the low-shot 1% ImageNet-1K benchmark.

We present an extended ablation of the multiblock masking strategy where we change the targets block scale (Table 8), the context scale (Table 9) and the number of target blocks (Table 10). We train a ViT-B/16 for 300 epochs using I-JEPA with various multi-block settings and compare performance on the 1% ImageNet-1K benchmark using a linear probe. In short, we find that it is important to predict several relatively large (semantic) target blocks, and to use a sufficiently informative (spatially distributed) context block.

| Targets | Context |  |  |
|---|---|---|---|
| Scale | Freq. | Scale | Top-1 |
| (0.075, 0.2) | 4 | (0.85, 1.0) | 19.2 |
| (0.1, 0.2) | 4 | (0.85, 1.0) | 39.2 |
| (0.125, 0.2) | 4 | (0.85, 1.0) | 42.4 |
| (0.15, 0.2) | 4 | (0.85, 1.0) | 54.2 |
| (0.2, 0.25) | 4 | (0.85, 1.0) | 38.9 |
| (0.2, 0.3) | 4 | (0.85, 1.0) | 33.6 |

| Targets | Context |  |  |
|---|---|---|---|
| Scale | Freq. | Scale | Top-1 |
| (0.15, 0.2) | 4 | (0.40, 1.0) | 31.2 |
| (0.15, 0.2) | 4 | (0.65, 1.0) | 47.1 |
| (0.15, 0.2) | 4 | (0.75, 1.0) | 49.3 |
| (0.15, 0.2) | 4 | (0.85, 1.0) | 54.2 |

| Targets | Context |  |  |
|---|---|---|---|
| Scale | Freq. | Scale | Top-1 |
| (0.15, 0.2) | 1 | (0.85, 1.0) | 9.0 |
| (0.15, 0.2) | 2 | (0.85, 1.0) | 22.0 |
| (0.15, 0.2) | 3 | (0.85, 1.0) | 48.5 |
| (0.15, 0.2) | 4 | (0.85, 1.0) | 54.2 |

An important important design choice in I-JEPA is that the target blocks are obtained by masking the output of the target-encoder, not the input. Table 11 shows the effect of this design choice on the semantic level of the learned representations when pretraining a ViT-H/16 using I-JEPA for 300 epochs. In the case where masking is applied to the input, we forward-propagate through the target-encoder once for each target region. Masking the output of the target-encoder during pretraining results in more semantic prediction targets and improves linear probing performance.

| Target Masking | Arch. | Epochs | Top-1 |
|---|---|---|---|
| Output | ViT-H/16 | 300 | 67.3 |
| Input | ViT-H/16 | 300 | 56.1 |

We examine the impact of the predictor depth on the downstream low-shot performance in Table 12. We pretrain a ViT-L/16 for 500 epochs using either a 6-layer predictor network or a 12-layer predictor network. The model pretrained using a deeper predictor shows a significant improvement in downstream low-shot performance compared to the model pretrained with a shallower predictor.

| Predictor Depth | Arch. | Epochs | Top-1 |
|---|---|---|---|
| 6 | ViT-L/16 | 500 | 64.0 |
| 12 | ViT-L/16 | 500 | 66.9 |

In Table 13, we evaluate the impact of weight-decay during pretraining. We explore two weight decay strategies: linearly increase the weight-decay from 0.040.04 to 0.40.4 or use a fix weight-decay of 0.050.05. Using a smaller weight decay during pretraining improves the downstream performance on ImageNet-1% when fine-tuning. However, this also leads to a degradation of performance in linear evaluation. In the main paper, we use the first weight decay strategy as it improves the performances in linear evaluation downstream tasks.

| Weight Decay | Arch. | Epochs | ImageNet-1% | ImageNet Linear-Eval |
|---|---|---|---|---|
| 0.04→0.40.04\rightarrow 0.4 | ViT-L/16 | 600 | 69.4 | 77.8 |
| 0.050.05 | ViT-L/16 | 600 | 70.7 | 76.4 |

We explore the impact of the predictor width in Table 14. We compare I-JEPA using a ViT-L encoder and a predictor with 386386 channels to a similar model using a predictor with 10241024 channels. Note that the ViT-L encoder has 10241024 channels. Using a bottleneck in the predictor width improves the downstream performance on ImageNet 1%.

| Predictor Width | Arch. | Epochs | Top-1 |
|---|---|---|---|
| 384 | ViT-L/16 | 600 | 70.7 |
| 1024 | ViT-L/16 | 600 | 68.4 |

## Appendix D Finetuning on the full ImageNet

In this section, we report performance on I-JEPA when fine-tuning on the full ImageNet dataset. We focus on the ViT-H/16448 as this architecture achieves state-of-art performance with MAE [36].

We use a fine-tuning protocol similar to MAE. Specifically, we fine-tune our model for 5050 epochs using AdamW and a cosine learning rate schedule. The base learning rate is set to 10−410^{-4} and the batch size to 528528. We train using mixup [76] set to 0.80.8, cutmix [73] set to 1.01.0, a drop path probability of 0.250.25 and a weight decay set to 0.040.04. We also use a layer decay of 0.750.75. Finally, we use the same rand-augment data-augmentations as MAE,

Table 15 reports the fine-tuning results. I-JEPA achieves 87.187.1 top-1 accuracy. Its performance is less than 1%1\% away from the best MAE model despite I-JEPA being trained for 5.35.3 times less epochs than MAE. This result demonstrates that I-JEPA is competitive when fine-tuning on the full ImageNet dataset.

| Method | Arch. | Epochs | Top-1 |
|---|---|---|---|
| MAE [36] | ViT-H/14448 | 1600 | 87.8 |
| I-JEPA | ViT-H/16448 | 300 | 87.1 |

## Appendix E RCDM Visualizations

To visualize the representations of a pretrained neural network in pixel space, we use the RCDM framework [13]. The RCDM framework trains a decoder network hωh_{\omega}, comprising a generative diffusion model, to reconstruct an image 𝒙{\bm{x}} from the representation vector of that image 𝒔x{\bm{s}}_{x} and a noisy version of that image 𝒙^≔𝒙+ϵ\hat{{\bm{x}}}\coloneqq{\bm{x}}+\epsilon, where ϵ\epsilon is an additive noise vector. Concretely, the decoder objective is to minimize the loss function ∥hω​(𝒙^,𝒔x)−ϵ∥\lVert h_{\omega}(\hat{{\bm{x}}},{\bm{s}}_{x})-\epsilon\rVert. We train each RCDM network for 300,000 iterations using the default hyperparameters [13]. After training the decoder, one can subsequently feed the representation vector of an unseen test image 𝒔y{\bm{s}}_{y} into the decoder along with various random noise vectors to generate several pixel-level visualizations of the representation, thus providing insight into the features captured in the representations of the pretrained network. Qualities that are common across samples represent information that is contained in the representation. On the other hand, qualities that vary across samples represent information that is not contained in the representations

In Figure 6, the visualizations are obtained by feeding the average-pooled output of the predictor, conditioned on a specific target region, into the decoder network, along with various random noise vectors. In Figures 7 and 8, the visualizations are obtained by feeding the average-pooled output of the target-encoder into the decoder network, along with various random noise vectors.

### E.1 Encoder Visualization

In Figure 7, we visualize the average-pooled I-JEPA representations at the output of our ViT-H/14 target-encoder. The first column contains the original image, while subsequent columns contain synthetic samples obtained by feeding the average-pooled representation of the image into the decoder along with various random noise vectors. Figure 7 suggests that the I-JEPA target-encoder is able to correctly capture the high-level information regarding objects and their poses, while discarding low-level image details and background information.

Figure 8 shows similar visualizations, but when using an MSN [4] pretrained ViT-L/7 target-encoder to compute the image representations. The MSN method trains a context- and target-encoder using a Joint-Embedding Architecture to enforce invariance of global image representations to various hand crafted data augmentations and missing patches. While the MSN pretrained network is able to capture high level semantic information about the image in the first column, it also exhibits higher variability in the generated samples, e.g., variability in the object pose, object scale, and number of instances. In short, the MSN pretrained discards much of the local structure in the image, which is in stark contrast to I-JEPA, which retains information about much of the local structure in the input image.
