# Training data-efficient image transformers & distillation through attention

Recently, neural networks purely based on attention were shown to address image understanding tasks such as image classification. These high-performing vision transformers are pre-trained with hundreds of millions of images using a large infrastructure, thereby limiting their adoption.

In this work, we produce competitive convolution-free transformers by training on Imagenet only. We train them on a single computer in less than 3 days. Our reference vision transformer (86M parameters) achieves top-1 accuracy of 83.1% (single-crop) on ImageNet with no external data.

More importantly, we introduce a teacher-student strategy specific to transformers. It relies on a distillation token ensuring that the student learns from the teacher through attention. We show the interest of this token-based distillation, especially when using a convnet as a teacher. This leads us to report results competitive with convnets for both Imagenet (where we obtain up to 85.2% accuracy) and when transferring to other tasks. We share our code and models.

## 1 Introduction

Convolutional neural networks have been the main design paradigm for image understanding tasks, as initially demonstrated on image classification tasks. One of the ingredient to their success was the availability of a large training set, namely Imagenet [13, 42]. Motivated by the success of attention-based models in Natural Language Processing [14, 52], there has been increasing interest in architectures leveraging attention mechanisms within convnets [2, 34, 61]. More recently several researchers have proposed hybrid architecture transplanting transformer ingredients to convnets to solve vision tasks [6, 43].

The vision transformer (ViT) introduced by Dosovitskiy et al. [15] is an architecture directly inherited from Natural Language Processing [52], but applied to image classification with raw image patches as input. Their paper presented excellent results with transformers trained with a large private labelled image dataset (JFT-300M [46], 300 millions images). The paper concluded that transformers “do not generalize well when trained on insufficient amounts of data”, and the training of these models involved extensive computing resources.

In this paper, we train a vision transformer on a single 8-GPU node in two to three days (53 hours of pre-training, and optionally 20 hours of fine-tuning) that is competitive with convnets having a similar number of parameters and efficiency. It uses Imagenet as the sole training set. We build upon the visual transformer architecture from Dosovitskiy et al. [15] and improvements included in the timm library [55]. With our Data-efficient image Transformers (DeiT), we report large improvements over previous results, see Figure 1. Our ablation study details the hyper-parameters and key ingredients for a successful training, such as repeated augmentation.

We address another question: how to distill these models? We introduce a token-based strategy, specific to transformers and denoted by DeiT, and show that it advantageously replaces the usual distillation.

In summary, our work makes the following contributions:

- • We show that our neural networks that contains no convolutional layer can achieve competitive results against the state of the art on ImageNet with no external data. They are learned on a single node with 4 GPUs in three days11 1 We can accelerate the learning of the larger model DeiT-B by training it on 8 GPUs in two days.. Our two new models DeiT-S and DeiT-Ti have fewer parameters and can be seen as the counterpart of ResNet-50 and ResNet-18.
We show that our neural networks that contains no convolutional layer can achieve competitive results against the state of the art on ImageNet with no external data. They are learned on a single node with 4 GPUs in three days11 1 We can accelerate the learning of the larger model DeiT-B by training it on 8 GPUs in two days.. Our two new models DeiT-S and DeiT-Ti have fewer parameters and can be seen as the counterpart of ResNet-50 and ResNet-18.

- • We introduce a new distillation procedure based on a distillation token, which plays the same role as the class token, except that it aims at reproducing the label estimated by the teacher. Both tokens interact in the transformer through attention. This transformer-specific strategy outperforms vanilla distillation by a significant margin.
We introduce a new distillation procedure based on a distillation token, which plays the same role as the class token, except that it aims at reproducing the label estimated by the teacher. Both tokens interact in the transformer through attention. This transformer-specific strategy outperforms vanilla distillation by a significant margin.

- • Interestingly, with our distillation, image transformers learn more from a convnet than from another transformer with comparable performance.
Interestingly, with our distillation, image transformers learn more from a convnet than from another transformer with comparable performance.

- • Our models pre-learned on Imagenet are competitive when transferred to different downstream tasks such as fine-grained classification, on several popular public benchmarks: CIFAR-10, CIFAR-100, Oxford-102 flowers, Stanford Cars and iNaturalist-18/19.
Our models pre-learned on Imagenet are competitive when transferred to different downstream tasks such as fine-grained classification, on several popular public benchmarks: CIFAR-10, CIFAR-100, Oxford-102 flowers, Stanford Cars and iNaturalist-18/19.

This paper is organized as follows: we review related works in Section 2, and focus on transformers for image classification in Section 3. We introduce our distillation strategy for transformers in Section 4. The experimental section 5 provides analysis and comparisons against both convnets and recent transformers, as well as a comparative evaluation of our transformer-specific distillation. Section 6 details our training scheme. It includes an extensive ablation of our data-efficient training choices, which gives some insight on the key ingredients involved in DeiT. We conclude in Section 7.

## 2 Related work

is so core to computer vision that it is often used as a benchmark to measure progress in image understanding. Any progress usually translates to improvement in other related tasks such as detection or segmentation. Since 2012’s AlexNet [32], convnets have dominated this benchmark and have become the de facto standard. The evolution of the state of the art on the ImageNet dataset [42] reflects the progress with convolutional neural network architectures and learning [32, 44, 48, 50, 51, 57].

Despite several attempts to use transformers for image classification [7], until now their performance has been inferior to that of convnets. Nevertheless hybrid architectures that combine convnets and transformers, including the self-attention mechanism, have recently exhibited competitive results in image classification [56], detection [6, 28], video processing [45, 53], unsupervised object discovery [35], and unified text-vision tasks [8, 33, 37].

Recently Vision transformers (ViT) [15] closed the gap with the state of the art on ImageNet, without using any convolution. This performance is remarkable since convnet methods for image classification have benefited from years of tuning and optimization [22, 55]. Nevertheless, according to this study [15], a pre-training phase on a large volume of curated data is required for the learned transformer to be effective. In our paper we achieve a strong performance without requiring a large training dataset, i.e., with Imagenet1k only.

introduced by Vaswani et al. [52] for machine translation are currently the reference model for all natural language processing (NLP) tasks. Many improvements of convnets for image classification are inspired by transformers. For example, Squeeze and Excitation [2], Selective Kernel [34] and Split-Attention Networks [61] exploit mechanism akin to transformers self-attention (SA) mechanism.

(KD), introduced by Hinton et al. [24], refers to the training paradigm in which a student model leverages “soft” labels coming from a strong teacher network. This is the output vector of the teacher’s softmax function rather than just the maximum of scores, wich gives a “hard” label. Such a training improves the performance of the student model (alternatively, it can be regarded as a form of compression of the teacher model into a smaller one – the student). On the one hand the teacher’s soft labels will have a similar effect to labels smoothing [58]. On the other hand as shown by Wei et al. [54] the teacher’s supervision takes into account the effects of the data augmentation, which sometimes causes a misalignment between the real label and the image. For example, let us consider image with a “cat” label that represents a large landscape and a small cat in a corner. If the cat is no longer on the crop of the data augmentation it implicitly changes the label of the image. KD can transfer inductive biases [1] in a soft way in a student model using a teacher model where they would be incorporated in a hard way. For example, it may be useful to induce biases due to convolutions in a transformer model by using a convolutional model as teacher. In our paper we study the distillation of a transformer student by either a convnet or a transformer teacher. We introduce a new distillation procedure specific to transformers and show its superiority.

## 3 Vision transformer: overview

In this section, we briefly recall preliminaries associated with the vision transformer [15, 52], and further discuss positional encoding and resolution.

The attention mechanism is based on a trainable associative memory with (key, value) vector pairs. A query vector q∈ℝdq\in\mathbb{R}^{d} is matched against a set of kk key vectors (packed together into a matrix K∈ℝk×dK\in\mathbb{R}^{k\times d}) using inner products. These inner products are then scaled and normalized with a softmax function to obtain kk weights. The output of the attention is the weighted sum of a set of kk value vectors (packed into V∈ℝk×dV\in\mathbb{R}^{k\times d}). For a sequence of NN query vectors (packed into Q∈ℝN×dQ\in\mathbb{R}^{N\times d}), it produces an output matrix (of size N×dN\times d):

|  | Attention⁡(Q,K,V)=Softmax⁡(Q​K⊤/d)​V,\mathrm{Attention}(Q,K,V)=\mathrm{Softmax}(QK^{\top}/\sqrt{d})V, |  | (1) |
|---|---|---|---|

where the Softmax\mathrm{Softmax} function is applied over each row of the input matrix and the d\sqrt{d} term provides appropriate normalization. In [52], a Self-attention layer is proposed. Query, key and values matrices are themselves computed from a sequence of NN input vectors (packed into X∈ℝN×DX\in\mathbb{R}^{N\times D}): Q=X​WQQ=XW_{\mathrm{Q}}, K=X​WKK=XW_{\mathrm{K}}, V=X​WVV=XW_{\mathrm{V}}, using linear transformations WQ,WK,WVW_{\mathrm{Q}},W_{\mathrm{K}},W_{\mathrm{V}} with the constraint k=Nk=N, meaning that the attention is in between all the input vectors. Finally, Multi-head self-attention layer (MSA) is defined by considering hh attention “heads”, ie hh self-attention functions applied to the input. Each head provides a sequence of size N×dN\times d. These hh sequences are rearranged into a N×d​hN\times dh sequence that is reprojected by a linear layer into N×DN\times D.

To get a full transformer block as in [52], we add a Feed-Forward Network (FFN) on top of the MSA layer. This FFN is composed of two linear layers separated by a GeLu activation [23]. The first linear layer expands the dimension from DD to 4​D4D, and the second layer reduces the dimension from 4​D4D back to DD. Both MSA and FFN are operating as residual operators thank to skip-connections, and with a layer normalization [3].

In order to get a transformer to process images, our work builds upon the ViT model [15]. It is a simple and elegant architecture that processes input images as if they were a sequence of input tokens. The fixed-size input RGB image is decomposed into a batch of NN patches of a fixed size of 16×1616\times 16 pixels (N=14×14N=14\times 14). Each patch is projected with a linear layer that conserves its overall dimension 3×16×16=7683\times 16\times 16=768. The transformer block described above is invariant to the order of the patch embeddings, and thus does not consider their relative position. The positional information is incorporated as fixed [52] or trainable [18] positional embeddings. They are added before the first transformer block to the patch tokens, which are then fed to the stack of transformer blocks.

is a trainable vector, appended to the patch tokens before the first layer, that goes through the transformer layers, and is then projected with a linear layer to predict the class. This class token is inherited from NLP [14], and departs from the typical pooling layers used in computer vision to predict the class. The transformer thus process batches of (N+1)(N+1) tokens of dimension DD, of which only the class vector is used to predict the output. This architecture forces the self-attention to spread information between the patch tokens and the class token: at training time the supervision signal comes only from the class embedding, while the patch tokens are the model’s only variable input.

Touvron et al. [50] show that it is desirable to use a lower training resolution and fine-tune the network at the larger resolution. This speeds up the full training and improves the accuracy under prevailing data augmentation schemes. When increasing the resolution of an input image, we keep the patch size the same, therefore the number NN of input patches does change. Due to the architecture of transformer blocks and the class token, the model and classifier do not need to be modified to process more tokens. In contrast, one needs to adapt the positional embeddings, because there are NN of them, one for each patch. Dosovitskiy et al. [15] interpolate the positional encoding when changing the resolution and demonstrate that this method works with the subsequent fine-tuning stage.

## 4 Distillation through attention

In this section we assume we have access to a strong image classifier as a teacher model. It could be a convnet, or a mixture of classifiers. We address the question of how to learn a transformer by exploiting this teacher. As we will see in Section 5 by comparing the trade-off between accuracy and image throughput, it can be beneficial to replace a convolutional neural network by a transformer. This section covers two axes of distillation: hard distillation versus soft distillation, and classical distillation versus the distillation token.

[24, 54] minimizes the Kullback-Leibler divergence between the softmax of the teacher and the softmax of the student model.

Let ZtZ_{\mathrm{t}} be the logits of the teacher model, ZsZ_{\mathrm{s}} the logits of the student model. We denote by τ\tau the temperature for the distillation, λ\lambda the coefficient balancing the Kullback–Leibler divergence loss (KL\mathrm{KL}) and the cross-entropy (ℒCE\mathcal{L}_{\mathrm{CE}}) on ground truth labels yy, and ψ\psi the softmax function. The distillation objective is

|  | ℒglobal=(1−λ)​ℒCE​(ψ⁡(Zs),y)+λ​τ2​KL​(ψ⁡(Zs/τ),ψ⁡(Zt/τ)).\mathcal{L}_{\mathrm{global}}=(1-\lambda)\mathcal{L}_{\mathrm{CE}}(\psi(Z_{\mathrm{s}}),y)+\lambda\tau^{2}\mathrm{KL}(\psi(Z_{\mathrm{s}}/\tau),\psi(Z_{\mathrm{t}}/\tau)). |  | (2) |
|---|---|---|---|

We introduce a variant of distillation where we take the hard decision of the teacher as a true label. Let yt=argmaxc​Zt​(c)y_{\mathrm{t}}=\mathrm{argmax}_{c}Z_{\mathrm{t}}(c) be the hard decision of the teacher, the objective associated with this hard-label distillation is:

|  | ℒglobalhardDistill=12​ℒCE​(ψ⁡(Zs),y)+12​ℒCE​(ψ⁡(Zs),yt).\mathcal{L}_{\mathrm{global}}^{\mathrm{hardDistill}}=\frac{1}{2}\mathcal{L}_{\mathrm{CE}}(\psi(Z_{s}),y)+\frac{1}{2}\mathcal{L}_{\mathrm{CE}}(\psi(Z_{s}),y_{\mathrm{t}}). |  | (3) |
|---|---|---|---|

For a given image, the hard label associated with the teacher may change depending on the specific data augmentation. We will see that this choice is better than the traditional one, while being parameter-free and conceptually simpler: The teacher prediction yty_{\mathrm{t}} plays the same role as the true label yy.

Note also that the hard labels can also be converted into soft labels with label smoothing [47], where the true label is considered to have a probability of 1−ε1-\varepsilon, and the remaining ε\varepsilon is shared across the remaining classes. We fix this parameter to ε=0.1\varepsilon=0.1 in our all experiments that use true labels.

We now focus on our proposal, which is illustrated in Figure 2. We add a new token, the distillation token, to the initial embeddings (patches and class token). Our distillation token is used similarly as the class token: it interacts with other embeddings through self-attention, and is output by the network after the last layer. Its target objective is given by the distillation component of the loss. The distillation embedding allows our model to learn from the output of the teacher, as in a regular distillation, while remaining complementary to the class embedding.

Interestingly, we observe that the learned class and distillation tokens converge towards different vectors: the average cosine similarity between these tokens equal to 0.06. As the class and distillation embeddings are computed at each layer, they gradually become more similar through the network, all the way through the last layer at which their similarity is high (cos=0.93), but still lower than 1. This is expected since as they aim at producing targets that are similar but not identical.

We verified that our distillation token adds something to the model, compared to simply adding an additional class token associated with the same target label: instead of a teacher pseudo-label, we experimented with a transformer with two class tokens. Even if we initialize them randomly and independently, during training they converge towards the same vector (cos=0.999), and the output embedding are also quasi-identical. This additional class token does not bring anything to the classification performance. In contrast, our distillation strategy provides a significant improvement over a vanilla distillation baseline, as validated by our experiments in Section 5.2.

We use both the true label and teacher prediction during the fine-tuning stage at higher resolution. We use a teacher with the same target resolution, typically obtained from the lower-resolution teacher by the method of Touvron et al [50]. We have also tested with true labels only but this reduces the benefit of the teacher and leads to a lower performance.

At test time, both the class or the distillation embeddings produced by the transformer are associated with linear classifiers and able to infer the image label. Yet our referent method is the late fusion of these two separate heads, for which we add the softmax output by the two classifiers to make the prediction. We evaluate these three options in Section 5.

## 5 Experiments

This section presents a few analytical experiments and results. We first discuss our distillation strategy. Then we comparatively analyze the efficiency and accuracy of convnets and vision transformers.

### 5.1 Transformer models

As mentioned earlier, our architecture design is identical to the one proposed by Dosovitskiy et al. [15] with no convolutions. Our only differences are the training strategies, and the distillation token. Also we do not use a MLP head for the pre-training but only a linear classifier. To avoid any confusion, we refer to the results obtained in the prior work by ViT, and prefix ours by DeiT. If not specified, DeiT refers to our referent model DeiT-B, which has the same architecture as ViT-B. When we fine-tune DeiT at a larger resolution, we append the resulting operating resolution at the end, e.g, DeiT-B↑\uparrow384. Last, when using our distillation procedure, we identify it with an alembic sign as DeiT.

The parameters of ViT-B (and therefore of DeiT-B) are fixed as D=768D=768, h=12h=12 and d=D/h=64d=D/h=64. We introduce two smaller models, namely DeiT-S and DeiT-Ti, for which we change the number of heads, keeping dd fixed. Table 1 summarizes the models that we consider in our paper.

| Model | ViT model | embedding | #heads | #layers | #params | training | throughput |
|---|---|---|---|---|---|---|---|
|  |  | dimension |  |  |  | resolution | (im/sec) |
| DeiT-Ti | N/A | 0192 | 03 | 12 | 005M | 224 | 2536 |
| DeiT-S | N/A | 0384 | 06 | 12 | 022M | 224 | 0940 |
| DeiT-B | ViT-B | 0768 | 12 | 12 | 086M | 224 | 0292 |

### 5.2 Distillation

Our distillation method produces a vision transformer that becomes on par with the best convnets in terms of the trade-off between accuracy and throughput, see Table 5. Interestingly, the distilled model outperforms its teacher in terms of the trade-off between accuracy and throughput. Our best model on ImageNet-1k is 85.2% top-1 accuracy outperforms the best Vit-B model pre-trained on JFT-300M at resolution 384 (84.15%). For reference, the current state of the art of 88.55% achieved with extra training data was obtained by the ViT-H model (600M parameters) trained on JFT-300M at resolution 512. Hereafter we provide several analysis and observations.

We have observed that using a convnet teacher gives better performance than using a transformer. Table 2 compares distillation results with different teacher architectures. The fact that the convnet is a better teacher is probably due to the inductive bias inherited by the transformers through distillation, as explained in Abnar et al. [1]. In all of our subsequent distillation experiments the default teacher is a RegNetY-16GF [40] (84M parameters) that we trained with the same data and same data-augmentation as DeiT. This teacher reaches 82.9%82.9\% top-1 accuracy on ImageNet.

| Teacher | Student: DeiT-B |  |  |
|---|---|---|---|
| Models | acc. | pretrain | ↑\uparrow384 |
| DeiT-B | 81.8 | 81.9 | 83.1 |
| RegNetY-4GF | 80.0 | 82.7 | 83.6 |
| RegNetY-8GF | 81.7 | 82.7 | 83.8 |
| RegNetY-12GF | 82.4 | 83.1 | 84.1 |
| RegNetY-16GF | 82.9 | 83.1 | 84.2 |

We compare the performance of different distillation strategies in Table 3. Hard distillation significantly outperforms soft distillation for transformers, even when using only a class token: hard distillation reaches 83.0% at resolution 224×\times224, compared to the soft distillation accuracy of 81.8%. Our distillation strategy from Section 4 further improves the performance, showing that the two tokens provide complementary information useful for classification: the classifier on the two tokens is significantly better than the independent class and distillation classifiers, which by themselves already outperform the distillation baseline.

The distillation token gives slightly better results than the class token. It is also more correlated to the convnets prediction. This difference in performance is probably due to the fact that it benefits more from the inductive bias of convnets. We give more details and an analysis in the next paragraph. The distillation token has an undeniable advantage for the initial training.

As discussed above, the architecture of the teacher has an important impact. Does it inherit existing inductive bias that would facilitate the training? While we believe it difficult to formally answer this question, we analyze in Table 4 the decision agreement between the convnet teacher, our image transformer DeiT learned from labels only, and our transformer DeiT.

Our distilled model is more correlated to the convnet than with a transformer learned from scratch. As to be expected, the classifier associated with the distillation embedding is closer to the convnet that the one associated with the class embedding, and conversely the one associated with the class embedding is more similar to DeiT learned without distillation. Unsurprisingly, the joint class+distil classifier offers a middle ground.

|  | Supervision | ImageNet top-1 (%) |  |  |  |  |
|---|---|---|---|---|---|---|
| method ↓\downarrow | label | teacher | Ti 224 | S 224 | B 224 | B↑\uparrow384 |
| DeiT– no distillation | ✓ | ✗ | 72.2 | 79.8 | 81.8 | 83.1 |
| DeiT– usual distillation | ✗ | soft | 72.2 | 79.8 | 81.8 | 83.2 |
| DeiT– hard distillation | ✗ | hard | 74.3 | 80.9 | 83.0 | 84.0 |
| DeiT: class embedding | ✓ | hard | 73.9 | 80.9 | 83.0 | 84.2 |
| DeiT: distil. embedding | ✓ | hard | 74.6 | 81.1 | 83.1 | 84.4 |
| DeiT: class+distillation | ✓ | hard | 74.5 | 81.2 | 83.4 | 84.5 |

Increasing the number of epochs significantly improves the performance of training with distillation, see Figure 3. With 300 epochs, our distilled network DeiT-B is already better than DeiT-B. But while for the latter the performance saturates with longer schedules, our distilled network clearly benefits from a longer training time.

|  | groundtruth | no distillation | DeiT student (of the convnet) |  |  |  |
|---|---|---|---|---|---|---|
|  | convnet | DeiT | class | distillation | DeiT |  |
| groundtruth | 0.000 | 0.171 | 0.182 | 0.170 | 0.169 | 0.166 |
| convnet (RegNetY) | 0.171 | 0.000 | 0.133 | 0.112 | 0.100 | 0.102 |
| DeiT | 0.182 | 0.133 | 0.000 | 0.109 | 0.110 | 0.107 |
| DeiT– class only | 0.170 | 0.112 | 0.109 | 0.000 | 0.050 | 0.033 |
| DeiT– distil. only | 0.169 | 0.100 | 0.110 | 0.050 | 0.000 | 0.019 |
| DeiT– class+distil. | 0.166 | 0.102 | 0.107 | 0.033 | 0.019 | 0.000 |

### 5.3 Efficiency vs accuracy: a comparative study with convnets

In the literature, the image classificaton methods are often compared as a compromise between accuracy and another criterion, such as FLOPs, number of parameters, size of the network, etc.

We focus in Figure 1 on the tradeoff between the throughput (images processed per second) and the top-1 classification accuracy on ImageNet. We focus on the popular state-of-the-art EfficientNet convnet, which has benefited from years of research on convnets and was optimized by architecture search on the ImageNet validation set.

Our method DeiT is slightly below EfficientNet, which shows that we have almost closed the gap between vision transformers and convnets when training with Imagenet only. These results are a major improvement (+6.3% top-1 in a comparable setting) over previous ViT models trained on Imagenet1k only [15]. Furthermore, when DeiT benefits from the distillation from a relatively weaker RegNetY to produce DeiT, it outperforms EfficientNet. It also outperforms by 1% (top-1 acc.) the Vit-B model pre-trained on JFT300M at resolution 384 (85.2% vs 84.15%), while being significantly faster to train.

Table 5 reports the numerical results in more details and additional evaluations on ImageNet V2 and ImageNet Real, that have a test set distinct from the ImageNet validation, which reduces overfitting on the validation set. Our results show that DeiT-B and DeiT-B ↑\uparrow384 outperform, by some margin, the state of the art on the trade-off between accuracy and inference time on GPU.

|  |  | image | throughput | ImNet | Real | V2 |
|---|---|---|---|---|---|---|
| Network | #param. | size | (image/s) | top-1 | top-1 | top-1 |
| Convnets |  |  |  |  |  |  |
| ResNet-18 [21] | 12M | 2242224^{2} | 4458.4 | 69.8 | 77.3 | 57.1 |
| ResNet-50 [21] | 25M | 2242224^{2} | 1226.1 | 76.2 | 82.5 | 63.3 |
| ResNet-101 [21] | 45M | 2242224^{2} | 0753.6 | 77.4 | 83.7 | 65.7 |
| ResNet-152 [21] | 60M | 2242224^{2} | 0526.4 | 78.3 | 84.1 | 67.0 |
| RegNetY-4GF [40]⋆\star | 21M | 2242224^{2} | 1156.7 | 80.0 | 86.4 | 69.4 |
| RegNetY-8GF [40]⋆\star | 39M | 2242224^{2} | 0591.6 | 81.7 | 87.4 | 70.8 |
| RegNetY-16GF [40]⋆\star | 84M | 2242224^{2} | 0334.7 | 82.9 | 88.1 | 72.4 |
| EfficientNet-B0 [48] | 5M | 2242224^{2} | 2694.3 | 77.1 | 83.5 | 64.3 |
| EfficientNet-B1 [48] | 8M | 2402240^{2} | 1662.5 | 79.1 | 84.9 | 66.9 |
| EfficientNet-B2 [48] | 9M | 2602260^{2} | 1255.7 | 80.1 | 85.9 | 68.8 |
| EfficientNet-B3 [48] | 12M | 3002300^{2} | 0732.1 | 81.6 | 86.8 | 70.6 |
| EfficientNet-B4 [48] | 19M | 3802380^{2} | 0349.4 | 82.9 | 88.0 | 72.3 |
| EfficientNet-B5 [48] | 30M | 4562456^{2} | 0169.1 | 83.6 | 88.3 | 73.6 |
| EfficientNet-B6 [48] | 43M | 5282528^{2} | 0096.9 | 84.0 | 88.8 | 73.9 |
| EfficientNet-B7 [48] | 66M | 6002600^{2} | 0055.1 | 84.3 | _ | _ |
| EfficientNet-B5 RA [12] | 30M | 4562456^{2} | 0096.9 | 83.7 | _ | _ |
| EfficientNet-B7 RA [12] | 66M | 6002600^{2} | 0055.1 | 84.7 | _ | _ |
| KDforAA-B8 | 87M | 8002800^{2} | 0025.2 | 85.8 | _ | _ |
| Transformers |  |  |  |  |  |  |
| ViT-B/16 [15] | 86M | 3842384^{2} | 0085.9 | 77.9 | 83.6 | _ |
| ViT-L/16 [15] | 307M | 3842384^{2} | 0027.3 | 76.5 | 82.2 | _ |
| DeiT-Ti | 5M | 2242224^{2} | 2536.5 | 72.2 | 80.1 | 60.4 |
| DeiT-S | 22M | 2242224^{2} | 0940.4 | 79.8 | 85.7 | 68.5 |
| DeiT-B | 86M | 2242224^{2} | 0292.3 | 81.8 | 86.7 | 71.5 |
| DeiT-B↑\uparrow384 | 86M | 3842384^{2} | 0085.9 | 83.1 | 87.7 | 72.4 |
| DeiT-Ti | 6M | 2242224^{2} | 2529.5 | 74.5 | 82.1 | 62.9 |
| DeiT-S | 22M | 2242224^{2} | 0936.2 | 81.2 | 86.8 | 70.0 |
| DeiT-B | 87M | 2242224^{2} | 0290.9 | 83.4 | 88.3 | 73.2 |
| DeiT-Ti/ 1000 epochs | 6M | 2242224^{2} | 2529.5 | 76.6 | 83.9 | 65.4 |
| DeiT-S/ 1000 epochs | 22M | 2242224^{2} | 0936.2 | 82.6 | 87.8 | 71.7 |
| DeiT-B/ 1000 epochs | 87M | 2242224^{2} | 0290.9 | 84.2 | 88.7 | 73.9 |
| DeiT-B ↑\uparrow384 | 87M | 3842384^{2} | 0085.8 | 84.5 | 89.0 | 74.8 |
| DeiT-B ↑\uparrow384 / 1000 epochs | 87M | 3842384^{2} | 0085.8 | 85.2 | 89.3 | 75.2 |

### 5.4 Transfer learning: Performance on downstream tasks

Although DeiT perform very well on ImageNet it is important to evaluate them on other datasets with transfer learning in order to measure the power of generalization of DeiT. We evaluated this on transfer learning tasks by fine-tuning on the datasets in Table 6. Table 7 compares DeiT transfer learning results to those of ViT [15] and state of the art convolutional architectures [48]. DeiT is on par with competitive convnet models, which is in line with our previous conclusion on ImageNet.

| Dataset | Train size | Test size | #classes |
|---|---|---|---|
| ImageNet [42] | 1,281,167 | 50,000 | 1000 |
| iNaturalist 2018 [26] | 437,513 | 24,426 | 8,142 |
| iNaturalist 2019 [27] | 265,240 | 3,003 | 1,010 |
| Flowers-102 [38] | 2,040 | 6,149 | 102 |
| Stanford Cars [30] | 8,144 | 8,041 | 196 |
| CIFAR-100 [31] | 50,000 | 10,000 | 100 |
| CIFAR-10 [31] | 50,000 | 10,000 | 10 |

| Model | ImageNet | CIFAR-10 | CIFAR-100 | Flowers | Cars | iNat-18 | iNat-19 | im/sec |
|---|---|---|---|---|---|---|---|---|
| Grafit ResNet-50 [49] | 79.6 | _ | _ | 98.2 | 92.5 | 69.8 | 75.9 | 1226.1 |
| Grafit RegNetY-8GF [49] | _ | _ | _ | 99.0 | 94.0 | 76.8 | 80.0 | 591.6 |
| ResNet-152 [10] | _ | _ | _ | _ | _ | 69.1 | _ | 526.3 |
| EfficientNet-B7 [48] | 84.3 | 98.9 | 91.7 | 98.8 | 94.7 | _ | _ | 55.1 |
| ViT-B/32 [15] | 73.4 | 97.8 | 86.3 | 85.4 | _ | _ | _ | 394.5 |
| ViT-B/16 [15] | 77.9 | 98.1 | 87.1 | 89.5 | _ | _ | _ | 85.9 |
| ViT-L/32 [15] | 71.2 | 97.9 | 87.1 | 86.4 | _ | _ | _ | 124.1 |
| ViT-L/16 [15] | 76.5 | 97.9 | 86.4 | 89.7 | _ | _ | _ | 27.3 |
| DeiT-B | 81.8 | 99.1 | 90.8 | 98.4 | 92.1 | 73.2 | 77.7 | 292.3 |
| DeiT-B↑\uparrow384 | 83.1 | 99.1 | 90.8 | 98.5 | 93.3 | 79.5 | 81.4 | 085.9 |
| DeiT-B | 83.4 | 99.1 | 91.3 | 98.8 | 92.9 | 73.7 | 78.4 | 290.9 |
| DeiT-B ↑\uparrow384 | 84.4 | 99.2 | 91.4 | 98.9 | 93.9 | 80.1 | 83.0 | 085.9 |

We investigate the performance when training from scratch on a small dataset, without Imagenet pre-training. We get the following results on the small CIFAR-10, which is small both w.r.t. the number of images and labels:

| Method | RegNetY-16GF | DeiT-B | DeiT-B |
|---|---|---|---|
| Top-1 | 98.0 | 97.5 | 98.5 |

For this experiment, we tried we get as close as possible to the Imagenet pre-training counterpart, meaning that (1) we consider longer training schedules (up to 7200 epochs, which corresponds to 300 Imagenet epochs) so that the network has been fed a comparable number of images in total; (2) we re-scale images to 224×224224\times 224 to ensure that we have the same augmentation. The results are not as good as with Imagenet pre-training (98.5% vs 99.1%), which is expected since the network has seen a much lower diversity. However they show that it is possible to learn a reasonable transformer on CIFAR-10 only.

## 6 Training details & ablation

|  | top-1 accuracy |  |  |  |  |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ablation on ↓\downarrow | Pre-training | Fine-tuning | Rand-Augment | AutoAug | Mixup | CutMix | Erasing | Stoch. Depth | Repeated Aug. | Dropout | Exp. Moving Avg. | pre-trained 2242224^{2} | fine-tuned 3842384^{2} |
| none: DeiT-B | adamw | adamw | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 81.8 ±0.2\pm 0.2 | 83.1 ±0.1\pm 0.1 |
| optimizer | SGD | adamw | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 74.5 | 77.3 |
| adamw | SGD | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 81.8 | 83.1 |  |
| data augmentation | adamw | adamw | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 79.6 | 80.4 |
| adamw | adamw | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 81.2 | 81.9 |  |
| adamw | adamw | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 78.7 | 79.8 |  |
| adamw | adamw | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | 80.0 | 80.6 |  |
| adamw | adamw | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | 75.8 | 76.7 |  |
| regularization | adamw | adamw | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | 04.3* | 00.1 |
| adamw | adamw | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | 03.4* | 00.1 |  |
| adamw | adamw | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | 76.5 | 77.4 |  |
| adamw | adamw | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 81.3 | 83.1 |  |
| adamw | adamw | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | 81.9 | 83.1 |  |

Pre-training

Fine-tuning

Rand-Augment

AutoAug

Mixup

CutMix

Erasing

Stoch. Depth

Repeated Aug.

Dropout

Exp. Moving Avg.

pre-trained 2242224^{2}

fine-tuned 3842384^{2}

In this section we discuss the DeiT training strategy to learn vision transformers in a data-efficient manner. We build upon PyTorch [39] and the timm library [55]22 2 The timm implementation already included a training procedure that improved the accuracy of ViT-B from 77.91% to 79.35% top-1, and trained on Imagenet-1k with a 8xV100 GPU machine.. We provide hyper-parameters as well as an ablation study in which we analyze the impact of each choice.

Transformers are relatively sensitive to initialization. After testing several options in preliminary experiments, some of them not converging, we follow the recommendation of Hanin and Rolnick [20] to initialize the weights with a truncated normal distribution.

Table 9 indicates the hyper-parameters that we use by default at training time for all our experiments, unless stated otherwise. For distillation we follow the recommendations from Cho et al. [9] to select the parameters τ\tau and λ\lambda. We take the typical values τ=3.0\tau=3.0 and λ=0.1\lambda=0.1 for the usual (soft) distillation.

| Methods | ViT-B [15] | DeiT-B |
|---|---|---|
| Epochs | 300 | 300 |
| Batch size | 4096 | 1024 |
| Optimizer | AdamW | AdamW |
| learning rate | 0.003 | 0.0005×batchsize5120.0005\times\frac{\textrm{batchsize}}{512} |
| Learning rate decay | cosine | cosine |
| Weight decay | 0.3 | 0.05 |
| Warmup epochs | 3.4 | 5 |
| Label smoothing ε\varepsilon | ✗ | 0.1 |
| Dropout | 0.1 | ✗ |
| Stoch. Depth | ✗ | 0.1 |
| Repeated Aug | ✗ | ✓ |
| Gradient Clip. | ✓ | ✗ |
| Rand Augment | ✗ | 9/0.5 |
| Mixup prob. | ✗ | 0.8 |
| Cutmix prob. | ✗ | 1.0 |
| Erasing prob. | ✗ | 0.25 |

Compared to models that integrate more priors (such as convolutions), transformers require a larger amount of data. Thus, in order to train with datasets of the same size, we rely on extensive data augmentation. We evaluate different types of strong data augmentation, with the objective to reach a data-efficient training regime.

Auto-Augment [11], Rand-Augment [12], and random erasing [62] improve the results. For the two latter we use the timm [55] customizations, and after ablation we choose Rand-Augment instead of AutoAugment. Overall our experiments confirm that transformers require a strong data augmentation: almost all the data-augmentation methods that we evaluate prove to be useful. One exception is dropout, which we exclude from our training procedure.

We have considered different optimizers and cross-validated different learning rates and weight decays. Transformers are sensitive to the setting of optimization hyper-parameters. Therefore, during cross-validation, we tried 3 different learning rates (5.10−4,3.10−4,5.10−55.10^{-4},3.10^{-4},5.10^{-5}) and 3 weight decay (0.030.03, 0.040.04, 0.050.05). We scale the learning rate according to the batch size with the formula: lrscaled=lr512×batchsize\mathrm{lr}_{\mathrm{scaled}}=\frac{\mathrm{lr}}{512}\times\mathrm{batchsize}, similarly to Goyal et al. [19] except that we use 512 instead of 256 as the base value.

The best results use the AdamW optimizer with the same learning rates as ViT [15] but with a much smaller weight decay, as the weight decay reported in the paper hurts the convergence in our setting.

We have employed stochastic depth [29], which facilitates the convergence of transformers, especially deep ones [16, 17]. For vision transformers, they were first adopted in the training procedure by Wightman [55]. Regularization like Mixup [60] and Cutmix [59] improve performance. We also use repeated augmentation [4, 25], which provides a significant boost in performance and is one of the key ingredients of our proposed training procedure.

We evaluate the EMA of our network obtained after training. There are small gains, which vanish after fine-tuning: the EMA model has an edge of is 0.1 accuracy points, but when fine-tuned the two models reach the same (improved) performance.

We adopt the fine-tuning procedure from Touvron et al. [51]: our schedule, regularization and optimization procedure are identical to that of FixEfficientNet but we keep the training-time data augmentation (contrary to the dampened data augmentation of Touvron et al. [51]). We also interpolate the positional embeddings: In principle any classical image scaling technique, like bilinear interpolation, could be used. However, a bilinear interpolation of a vector from its neighbors reduces its ℓ2\ell_{2}-norm compared to its neighbors. These low-norm vectors are not adapted to the pre-trained transformers and we observe a significant drop in accuracy if we employ use directly without any form of fine-tuning. Therefore we adopt a bicubic interpolation that approximately preserves the norm of the vectors, before fine-tuning the network with either AdamW [36] or SGD. These optimizers have a similar performance for the fine-tuning stage, see Table 8.

By default and similar to ViT [15] we train DeiT models with at resolution 224224 and we fine-tune at resolution 384384. We detail how to do this interpolation in Section 3. However, in order to measure the influence of the resolution we have finetuned DeiT at different resolutions. We report these results in Table 10.

| image | throughput | Imagenet [42] | Real [5] | V2 [41] |
|---|---|---|---|---|
| size | (image/s) | acc. top-1 | acc. top-1 | acc. top-1 |
| 1602160^{2} | 609.31 | 79.9 | 84.8 | 67.6 |
| 2242224^{2} | 291.05 | 81.8 | 86.7 | 71.5 |
| 3202320^{2} | 134.13 | 82.7 | 87.2 | 71.9 |
| 3842384^{2} | 085.87 | 83.1 | 87.7 | 72.4 |

A typical training of 300 epochs takes 37 hours with 2 nodes or 53 hours on a single node for the DeiT-B.As a comparison point, a similar training with a RegNetY-16GF [40] (84M parameters) is 20% slower. DeiT-S and DeiT-Ti are trained in less than 3 days on 4 GPU. Then, optionally we fine-tune the model at a larger resolution. This takes 20 hours on a single node (8 GPU) to produce a FixDeiT-B model at resolution 384×\times384, which corresponds to 25 epochs. Not having to rely on batch-norm allows one to reduce the batch size without impacting performance, which makes it easier to train larger models. Note that, since we use repeated augmentation [4, 25] with 3 repetitions, we only see one third of the images during a single epoch33 3 Formally it means that we have 100 epochs, but each is 3x longer because of the repeated augmentations. We prefer to refer to this as 300 epochs in order to have a direct comparison on the effective training time with and without repeated augmentation..

## 7 Conclusion

In this paper, we have introduced DeiT, which are image transformers that do not require very large amount of data to be trained, thanks to improved training and in particular a novel distillation procedure. Convolutional neural networks have optimized, both in terms of architecture and optimization during almost a decade, including through extensive architecture search that is prone to overfiting, as it is the case for instance for EfficientNets [51]. For DeiT we have started the existing data augmentation and regularization strategies pre-existing for convnets, not introducing any significant architectural beyond our novel distillation token. Therefore it is likely that research on data-augmentation more adapted or learned for transformers will bring further gains.

Therefore, considering our results, where image transformers are on par with convnets already, we believe that they will rapidly become a method of choice considering their lower memory footprint for a given accuracy.

We provide an open-source implementation of our method. It is available at https://github.com/facebookresearch/deit.

### Acknowledgements

Many thanks to Ross Wightman for sharing his ViT code and bootstrapping training method with the community, as well as for valuable feedback that helped us to fix different aspects of this paper. Thanks to Vinicius Reis, Mannat Singh, Ari Morcos, Mark Tygert, Gabriel Synnaeve, and other colleagues at Facebook for brainstorming and some exploration on this axis. Thanks to Ross Girshick and Piotr Dollar for constructive comments.
