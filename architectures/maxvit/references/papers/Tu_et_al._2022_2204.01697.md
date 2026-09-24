# MaxViT: Multi-Axis Vision Transformer

Transformers have recently gained significant attention in the computer vision community. However, the lack of scalability of self-attention mechanisms with respect to image size has limited their wide adoption in state-of-the-art vision backbones. In this paper we introduce an efficient and scalable attention model we call multi-axis attention, which consists of two aspects: blocked local and dilated global attention. These design choices allow global-local spatial interactions on arbitrary input resolutions with only linear complexity. We also present a new architectural element by effectively blending our proposed attention model with convolutions, and accordingly propose a simple hierarchical vision backbone, dubbed MaxViT, by simply repeating the basic building block over multiple stages. Notably, MaxViT is able to “see” globally throughout the entire network, even in earlier, high-resolution stages. We demonstrate the effectiveness of our model on a broad spectrum of vision tasks. On image classification, MaxViT achieves state-of-the-art performance under various settings: without extra data, MaxViT attains 86.5% ImageNet-1K top-1 accuracy; with ImageNet-21K pre-training, our model achieves 88.7% top-1 accuracy. For downstream tasks, MaxViT as a backbone delivers favorable performance on object detection as well as visual aesthetic assessment. We also show that our proposed model expresses strong generative modeling capability on ImageNet, demonstrating the superior potential of MaxViT blocks as a universal vision module. The source code and trained models will be available at https://github.com/google-research/maxvit.

## 1 Introduction

Convolutional Neural Networks (ConvNets) have been the dominant architectural design choice for computer vision [48, 29, 75, 76] since AlexNet [48]. ConvNets continue to excel on numerous vision problems by going deeper [75], wider [76, 74], adding dense connections [37], efficient separable convolutions [35, 70], atrous convolutions [9], using encoder-decoder frameworks [67], and even introducing modern micro-design components [57]. Meanwhile, as inspired by the evolution of self-attention models like Transformers [85] in natural language processing [20, 100, 49, 63], numerous researchers have started to introduce attention mechanisms into vision [88, 6]. The Vision Transformer (ViT) [22] is perhaps the first fully Transformer-based architecture for vision, whereby image patches are simply regarded as sequences of words and a transformer encoder is applied on these visual tokens. When pre-trained on large-scale datasets [73], ViT can achieve compelling results on image recognition.

However, it has been observed that without extensive pre-training [22, 81] ViT underperforms on image recognition. This is due to the strong model capacity of Transformers, that is imbued with less inductive bias, which leads to overfitting. To properly regularize the model capacity and improve its scalability, numerous subsequent efforts have studied sparse Transformer models tailored for vision tasks such as local attention [56, 99, 50, 16]. These methods typically re-introduce hierarchical architectures to compensate for the loss of non-locality. The Swin Transformer [56] is one such successful attempt to modify Transformers by applying self-attention on shifted non-overlapping windows. For the first time, this approach outperformed ConvNets on the ImageNet benchmark with a pure vision Transformer. Despite having more flexibility and generalizability than the full attention used in ViT, window-based attention has been observed to have limited model capacity due to the loss of non-locality, and henceforth scales unfavorably on larger data regimes such as ImageNet-21K and JFT [19]. However, acquiring global interactions via full-attention at early or high-resolution stages in a hierarchical network is computationally heavy, as the attention operator requires quadratic complexity. How to efficiently incorporate global and local interactions to balance the model capacity and generalizability under a computation budget still remains challenging.

In this paper, we present a new type of Transformer module, called multi-axis self-attention (Max-SA), that capably serves as a basic architecture component which can perform both local and global spatial interactions in a single block. Compared to full self-attention, Max-SA enjoys greater flexibility and efficiency, i.e., naturally adaptive to different input lengths with linear complexity; in contrast to (shifted) window/local attention, Max-SA allows for stronger model capacity by proposing a global receptive field. Moreover, with merely linear complexity, Max-SA can be used as a general stand-alone attention module in any layer of a network, even in earlier, high-resolution stages.

To demonstrate its effectiveness and universality, we further design a simple but effective vision backbone called Multi-axis Vision Transformer (MaxViT) by hierarchically stacking repeated blocks composed of Max-SA and convolutions. While our proposed model belongs to the category of hybrid vision Transformers, MaxViT distinguishes from previous approaches [94, 19] in that we strive for simplicity, by designing a basic block unifying convolution, local, and global attention, then simply repeating it. Our experiments shows that the MaxViT significantly improves upon state-of-the-art (SOTA) performance under all data regimes for a broad range of visual tasks including classification, object detection and segmentation, image aesthetics assessment, and image generation. Specifically, as Figure 1 shows, MaxViT outperforms all recent Transformer-based models in regards to both accuracy vs. FLOPs and accuracy vs. parameter curves. Our contributions are:

- • A generic strong Transformer backbone, MaxViT, that can capture both local and global spatial interactions throughout every stage of the network.
A generic strong Transformer backbone, MaxViT, that can capture both local and global spatial interactions throughout every stage of the network.

- • A novel stand-alone multi-axis attention module composed of blocked local and dilated global attention, enjoying global perception in linear complexity.
A novel stand-alone multi-axis attention module composed of blocked local and dilated global attention, enjoying global perception in linear complexity.

- • We demonstrate large amounts of design choices including number of layers, layouts, the use of MBConv, etc. with extensive ablation studies, that eventually converge towards our final modular design, the MaxViT-Block.
We demonstrate large amounts of design choices including number of layers, layouts, the use of MBConv, etc. with extensive ablation studies, that eventually converge towards our final modular design, the MaxViT-Block.

- • Our extensive experiments show that MaxViT achieves SOTA results under various data regimes for a broad range of tasks including image classification, object detection, image aesthetic assessment, and image generation.
Our extensive experiments show that MaxViT achieves SOTA results under various data regimes for a broad range of tasks including image classification, object detection, image aesthetic assessment, and image generation.

## 2 Related work

Convolutional networks. Since AlexNet [48], convolutional neural networks (ConvNets) have been used as de facto solutions to almost all vision tasks [29, 8, 37, 104, 13, 51, 90, 78, 89] before the “Roaring 20s” [57]. Phenomenal architectural improvements have been made in the past decade: residual [29] and dense connections [37], fully-convolutional networks [58], encoder-decoder schemes [67], feature pyramids [52], increased depths and widths [75], spatial- and channel-wise attention models [36, 91], non-local interactions [88], to name a few. A remarkable recent work ConvNeXt [57] has re-introduced core designs of vision Transformers and shown that a ‘modernized’ pure ConvNet can achieve performance comparable to Transformers on broad vision tasks.

Transformers in vision. Transformers were originally proposed for natural language processing [85]. The debut of the Vision Transformer (ViT) [22] in 2020 showed that pure Transformer-based architectures are also effective solutions for vision problems. The elegantly novel view of ViT that treats image patches as visual words has stimulated explosive research interest in visual Transformers. To account for locality and 2D nature of images, the Swin Transformer aggregates attention in shifted windows in a hierarchical architecture [56]. More recent works have been focused on improving model and data efficiency, including sparse attention [21, 99, 64, 86, 96, 1], improved locality [101, 27], pyramidal designs [87, 24, 97], improved training strategies [81, 82, 105, 3], etc. We refer readers to dedicated surveys [44, 44] of vision Transformers for a comprehensive review.

Hybrid models. Pure Transformer-based vision models have been observed to generalize poorly due to relatively less inductive bias [22, 19, 81]. Vision Transformers also exhibit substandard optimizability [94]. An intriguingly simple improvement is to adopt a hybrid design of Transformer and convolution layers such as using a few convolutions to replace the coarse patchify stem [94, 19]. A broad range of works fall into this category, either explicitly hybridized [93, 23, 19, 94, 24, 98, 4] or in an implicit fashion [56, 16].

Transformer for GANs. Transformers have also proven effective in generative adversarial networks (GANs) [26]. TransGAN [40] built a pure Transformer GAN with a careful design of local attention and upsampling layers, demonstrating effectiveness on small scale datasets [47, 18]. GANformer [38] explored efficient global attention mechanisms to improve on StyleGAN [42] generator. HiT [103] presents an efficient Transformer generator based on local-global attention that can scale up to 1K high-resolution image generation.

## 3 Method

Inspired by the sparse approaches presented in [103, 83], we introduce a new type of attention module, dubbed blocked multi-axis self-attention (Max-SA), by decomposing the fully dense attention mechanisms into two sparse forms – window attention and grid attention – which reduces the quadratic complexity of vanilla attention to linear, without any loss of non-locality. Our sequential design offers greater simplicity and flexibility, while performing even better than previous methods – each individual module can be used either standalone or combined in any order (Tables 10-10), whereas parallel designs [103, 83] offer no such benefits. Because of the flexibility and scalability of Max-SA, we are able to build a novel vision backbone, which we call MaxViT, by simply stacking alternative layers of Max-SA with MBConv [35] in a hierarchical architecture, as shown in Figure 2. MaxViT benefits from global and local receptive fields throughout the entire network, from shallow to deep stages, demonstrating superior performance in regards to both model capacity and generalization abilities.

### 3.1 Attention

Self-attention allows for spatial mixing of entire spatial (or sequence) locations while also benefiting from content-dependent weights based on normalized pairwise similarity. The standard self-attention defined in [85, 22] is location-unaware, i.e., non-translation equivariant, an important inductive bias imbued in ConvNets. Relative self-attention [56, 19, 71, 40] has been proposed to improve on vanilla attention by introducing a relative learned bias added to the attention weights, which has been shown to consistently outperform original attention on many vision tasks [56, 19, 40]. In this work, we mainly adopt the pre-normalized relative self-attention defined in [19] as the key operator in MaxViT.

### 3.2 Multi-axis Attention

Global interaction is one of the key advantages of self-attention as compared to local convolution. However, directly applying attention along the entire space is computationally infeasible as the attention operator requires quadratic complexity. To tackle this problem, we present a multi-axis approach to decompose the full-size attention into two sparse forms – local and global – by simply decomposing the spatial axes. Let X∈ℝH×W×CX\in\mathbb{R}^{H\times W\times C} be an input feature map. Instead of applying attention on the flattened spatial dimension H​WHW, we block the feature into a tensor of shape (HP×WP,P×P,C)(\frac{H}{P}\times\frac{W}{P},P\times P,C), representing partitioning into non-overlapping windows, each of size P×PP\times P. Applying self-attention on the local spatial dimension i.e., P×PP\times P, is equivalent to attending within a small window [56]. We will use this block attention to conduct local interactions.

Despite bypassing the notoriously heavy computation of full self-attention, local-attention models have been observed to underfit on huge-scale datasets [19, 22]. Inspired by block attention, we present a surprisingly simple but effective way to gain sparse global attention, which we call grid attention. Instead of partitioning feature maps using fixed window size, we grid the tensor into the shape (G×G,HG×WG,C)(G\times G,\frac{H}{G}\times\frac{W}{G},C) using a fixed G×GG\times G uniform grid, resulting in windows having adaptive size HG×WG\frac{H}{G}\times\frac{W}{G}. Employing self-attention on the decomposed grid axis i.e., G×GG\times G, corresponds to dilated, global spatial mixing of tokens. By using the same fixed window and grid sizes (we use P=G=7P=G=7 following Swin [56]), we can fully balance the computation between local and global operations, both having only linear complexity with respect to spatial size or sequence length. Note that our proposed Max-SA module can be a drop-in replacement of the Swin attention module [56] with exactly the same number of parameters and FLOPs. Yet it enjoys global interaction capability without requiring masking, padding, or cyclic-shifting, making it more implementation friendly, preferable to the shifted window scheme [56]. For instance, the multi-axis attention can be easily implemented with einops [66] without modifying the original attention operation (see Appendix). It is worth mentioning that our proposed multi-axis attention (Max-SA) is fundamentally different from the axial-attention models [86, 33]. Please see Appendix for a detailed comparison.

MaxViT block. We sequentially stack the two types of attentions to gain both local and global interactions in a single block, as shown in Figure 3. Note that we also adopt typical designs in Transformers [22, 56], including LayerNorm [2], Feedforward networks (FFNs) [22, 56], and skip-connections. We also add a MBConv block [35] with squeeze-and-excitation (SE) module [36] prior to the multi-axis attention, as we have observed that using MBConv together with attention further increases the generalization as well as the trainability of the network [94]. Using MBConv layers prior to attention offers another advantage, in that depthwise convolutions can be regarded as conditional position encoding (CPE) [17], making our model free of explicit positional encoding layers. Note that our proposed stand-alone multi-axis attention may be used together or in isolation for different purposes – block attention for local interaction, and grid attention for global mixing. These elements can be easily plugged into many vision architectures, especially on high-resolution tasks that can benefit by global interactions with affordable computation.

### 3.3 Architecture Variants

We designed a series of extremely simple architectural variants to explore the effectiveness of our proposed MaxViT block, as shown in Figure 2. We use a hierarchical backbone similar to common ConvNet practices [29, 57, 19, 80] where the input is first downsampled using Conv3x3 layers in stem stage (S0). The body of the network contains four stages (S1-S4), with each stage having half the resolution of the previous one with a doubled number of channels (hidden dimension). In our network, we employ identical MaxViT blocks throughout the entire backbone. We apply downsampling in the Depthwise Conv3x3 layer of the first MBConv block in each stage. The expansion and shrink rates for inverted bottleneck [35] and squeeze-excitation (SE) [36] are 4 and 0.25 by default. We set the attention head size to be 32 for all attention blocks. We scale up the model by increasing block numbers per stage BB and the channel dimension CC. We summarize the architectural configurations of the MaxViT variants in Table 1.

| Stage | Size | MaxViT-T | MaxViT-S | MaxViT-B | MaxViT-L | MaxViT-XL |
|---|---|---|---|---|---|---|
| S0: Conv-stem | 1/2\nicefrac{{1}}{{2}} | B=2 C=64 | B=2 C=64 | B=2 C=64 | B=2 C=128 | B=2 C=192 |
| S1: MaxViT-Block | 1/4\nicefrac{{1}}{{4}} | B=2 C=64 | B=2 C=96 | B=2 C=96 | B=2 C=128 | B=2 C=192 |
| S2: MaxViT-Block | 1/8\nicefrac{{1}}{{8}} | B=2 C=128 | B=2 C=192 | B=6 C=192 | B=6 C=256 | B=6 C=384 |
| S3: MaxViT-Block | 1/16\nicefrac{{1}}{{16}} | B=5 C=256 | B=5 C=384 | B=14 C=384 | B=14 C=512 | B=14 C=768 |
| S4: MaxViT-Block | 1/32\nicefrac{{1}}{{32}} | B=2 C=512 | B=2 C=768 | B=2 C=768 | B=2 C=1024 | B=2 C=1536 |

## 4 Experiments

We validated the efficacy of our proposed model on various vision tasks: ImageNet classification [48], image object detection and instance segmentation [53], image aesthetics/quality assessment [61], and unconditional image generation [26]. More experimental details can be found in the Appendix.

|  | Model | Eval size | Params | FLOPs | Throughput (image/s) | IN-1K top-1 acc. |
|---|---|---|---|---|---|---|
| ConvNets | ∙\bulletEffNet-B6 [79] | 528 | 43M | 19.0G | 96.9 | 84.0 |
| ∙\bulletEffNet-B7 [79] | 600 | 66M | 37.0G | 55.1 | 84.3 |  |
| ∙\bulletRegNetY-16 [62] | 224 | 84M | 16.0G | 334.7 | 82.9 |  |
| ∙\bulletNFNet-F0 [5] | 256 | 72M | 12.4G | 533.3 | 83.6 |  |
| ∙\bulletNFNet-F1 [5] | 320 | 132M | 35.5G | 228.5 | 84.7 |  |
| ∙\bulletEffNetV2-S [80] | 384 | 24M | 8.8G | 666.6 | 83.9 |  |
| ∙\bulletEffNetV2-M [80] | 480 | 55M | 24.0G | 280.7 | 85.1 |  |
| ∙\bulletConvNeXt-S [57] | 224 | 50M | 8.7G | 447.1 | 83.1 |  |
| ∙\bulletConvNeXt-B [57] | 224 | 89M | 15.4G | 292.1 | 83.8 |  |
| ∙\bulletConvNeXt-L [57] | 224 | 198M | 34.4G | 146.8 | 84.3 |  |
| ViTs | ∘\circViT-B/32 [22] | 384 | 86M | 55.4G | 85.9 | 77.9 |
| ∘\circViT-B/16 [22] | 384 | 307M | 190.7G | 27.3 | 76.5 |  |
| ∘\circDeiT-B [81] | 384 | 86M | 55.4G | 85.9 | 83.1 |  |
| ∘\circCaiT-M24 [82] | 224 | 186M | 36.0G | - | 83.4 |  |
| ∘\circCaiT-M24 [82] | 384 | 186M | 116.1G | - | 84.5 |  |
| ∘\circDeepViT-L [105] | 224 | 55M | 12.5G | - | 83.1 |  |
| ∘\circT2T-ViT-24 [101] | 224 | 64M | 15.0G | - | 82.6 |  |
| ∘\circSwin-S [56] | 224 | 50M | 8.7G | 436.9 | 83.0 |  |
| ∘\circSwin-B [56] | 384 | 88M | 47.0G | 84.7 | 84.5 |  |
| ∘\circCSwin-B [21] | 224 | 78M | 15.0G | 250 | 84.2 |  |
| ∘\circCSwin-B [21] | 384 | 78M | 47.0G | - | 85.4 |  |
| ∘\circFocal-S [99] | 224 | 51M | 9.1G | - | 83.5 |  |
| ∘\circFocal-B [99] | 224 | 90M | 16.0G | - | 83.8 |  |
| Hybrid | ⋄\diamondCvT-21 [93] | 384 | 32M | 24.9G | - | 83.3 |
| ⋄\diamondCoAtNet-2 [19] | 224 | 75M | 15.7G | 247.7 | 84.1 |  |
| ⋄\diamondCoAtNet-3 [19] | 224 | 168M | 34.7G | 163.3 | 84.5 |  |
| ⋄\diamondCoAtNet-3 [19] | 384 | 168M | 107.4G | 48.5 | 85.8 |  |
| ⋄\diamondCoAtNet-3 [19] | 512 | 168M | 203.1G | 22.4 | 86.0 |  |
| ⋄\diamondMaxViT-T | 224 | 31M | 5.6G | 349.6 | 83.62 |  |
| ⋄\diamondMaxViT-S | 224 | 69M | 11.7G | 242.5 | 84.45 |  |
| ⋄\diamondMaxViT-B | 224 | 120M | 23.4G | 133.6 | 84.95 |  |
| ⋄\diamondMaxViT-L | 224 | 212M | 43.9G | 99.4 | 85.17 |  |
| ⋄\diamondMaxViT-T | 384 | 31M | 17.7G | 121.9 | 85.24 |  |
| ⋄\diamondMaxViT-S | 384 | 69M | 36.1G | 82.7 | 85.74 |  |
| ⋄\diamondMaxViT-B | 384 | 120M | 74.2G | 45.8 | 86.34 |  |
| ⋄\diamondMaxViT-L | 384 | 212M | 133.1G | 34.3 | 86.40 |  |
| ⋄\diamondMaxViT-T | 512 | 31M | 33.7G | 63.8 | 85.72 |  |
| ⋄\diamondMaxViT-S | 512 | 69M | 67.6G | 43.3 | 86.19 |  |
| ⋄\diamondMaxViT-B | 512 | 120M | 138.5G | 24.0 | 86.66 |  |
| ⋄\diamondMaxViT-L | 512 | 212M | 245.4G | 17.8 | 86.70 |  |

|  | Model | Eval size | Params | FLOPs | IN-1K top-1 acc. |  |
|---|---|---|---|---|---|---|
|  | 21K→\rightarrow1K | JFT→\rightarrow1K |  |  |  |  |
| ConvNets | ∙\bulletBiT-R-101x3 [46] | 384 | 388M | 204.6G | 84.4 | - |
| ∙\bulletBiT-R-152x4 [46] | 480 | 937M | 840.5G | 85.4 | - |  |
| ∙\bulletEffNetV2-L [80] | 480 | 121M | 53.0G | 86.8 | - |  |
| ∙\bulletEffNetV2-XL [80] | 512 | 208M | 94.0G | 87.3 | - |  |
| ∙\bulletConvNeXt-L [57] | 384 | 198M | 101.0G | 87.5 | - |  |
| ∙\bulletConvNeXt-XL [57] | 384 | 350M | 179.0G | 87.8 | - |  |
| ∙\bulletNFNet-F4+ [5] | 512 | 527M | 367G | - | 89.20 |  |
| ViTs | ∘\circViT-B/16 [22] | 384 | 87M | 55.5G | 84.0 | - |
| ∘\circViT-L/16 [22] | 384 | 305M | 191.1G | 85.2 |  |  |
| ∘\circViT-L/16 [22] | 512 | 305M | 364G | - | 87.76 |  |
| ∘\circViT-H/14 [22] | 518 | 632M | 1021G | - | 88.55 |  |
| ∘\circHaloNet-H4 [84] | 512 | 85M | - | 85.8 | - |  |
| ∘\circSwinV2-B [56] | 384 | 88M | - | 87.1 | - |  |
| ∘\circSwinV2-L [56] | 384 | 197M | - | 87.7 | - |  |
| Hybrid | ⋄\diamondCvT-W24 [93] | 384 | 277M | 193.2G | 87.7 | - |
| ⋄\diamondR+ViT-L/16 [22] | 384 | 330M | - | - | 87.12 |  |
| ⋄\diamondCoAtNet-3 [19] | 384 | 168M | 107.4G | 87.6 | 88.52 |  |
| ⋄\diamondCoAtNet-3 [19] | 512 | 168M | 214G | 87.9 | 88.81 |  |
| ⋄\diamondCoAtNet-4 [19] | 512 | 275M | 360.9G | 88.1 | 89.11 |  |
| ⋄\diamondCoAtNet-5 [19] | 512 | 688M | 812G | - | 89.77 |  |
| ⋄\diamondMaxViT-B | 384 | 119M | 74.2G | 88.24 | 88.69 |  |
| ⋄\diamondMaxViT-L | 384 | 212M | 128.7G | 88.32 | 89.12 |  |
| ⋄\diamondMaxViT-XL | 384 | 475M | 293.7G | 88.51 | 89.36 |  |
| ⋄\diamondMaxViT-B | 512 | 119M | 138.3G | 88.38 | 88.82 |  |
| ⋄\diamondMaxViT-L | 512 | 212M | 245.2G | 88.46 | 89.41 |  |
|  | ⋄\diamondMaxViT-XL | 512 | 475M | 535.2G | 88.70 | 89.53 |

### 4.1 Image Classification on ImageNet-1K

ImageNet-1K. We show in Table 2 the performance comparisons on ImageNet-1K classification. Under the basic 224×\times224 setting, MaxViT outperformed the most recent strong hybrid model CoAtNet by a large margin across the entire FLOPs spectrum, as shown in Figure 1(a). The MaxViT-L model sets a new performance record of 85.17% at 224×224224\times 224 training without extra training strategies, outperforming CoAtNet-3 by 0.67%. In regards to throughput-accuracy trade-offs at 2242224^{2}, MaxViT-S obtains 84.45% top-1 accuracy, 0.25% higher than CSWin-B and 0.35% higher than CoAtNet-2 with comparable throughput.

When fine-tuned at higher resolutions (384/512), MaxViT continues to deliver high performance compared to strong ConvNet and Transformer competitors: (1) at 3842384^{2}, MaxViT-B attains 86.34% top-1 accuracy, outperforming EfficientNetV2-L by 0.64%; (2) when fine-tuned at 5122512^{2}, our MaxViT-L (212M) achieves top-1 accuracy 86.7% , setting new SOTA performance on ImageNet-1K under the normal training setting. As Figure 1 shows, MaxViT scales much better than SOTA vision Transformers on the ImageNet-1K trained model scale.

ImageNet-21K. Table 3 shows the results of models pre-trained on ImageNet-21K. Remarkably, the MaxViT-B model achieves 88.38% accuracy, outperforming the previous best model CoAtNet-4 by 0.28% using only 43% of parameter count and 38% of FLOPs, demonstrating greater parameter and computing efficiency. Figure 4(a) visualizes the model size comparison – MaxViT scales significantly better than previous attention-based models of similar complexities, across the board. Additionally, the MaxViT-XL model achieves new SOTA performance, an accuracy of 88.70% when fine-tuned at resolution 512×512512\times 512.

JFT-300M. We also trained our model on a larger-scale proprietary dataset JFT-300M which contains ∼\sim300 million weakly labeled images. As shown in Table 3 and Figure 4(b), our model is also scalable to massive scale training data – MaxViT-XL achieves a high accuracy of 89.53% with 475 million parameters, outperforming previous models under comparable model sizes. Due to resource limitations, we leave experiments on billion-parameter-scale models on planet-scale datasets (e.g., JFT-3B [102]) as future work.

### 4.2 Object Detection and Instance Segmentation

Setting. We evaluated the MaxViT architectures on the COCO2017 [53] object bounding box detection and instance segmentation tasks with a two-stage framework [65]. On the object detection task, a feature-pyramid architecture [52] was employed to boost different levels of objectiveness. In the instance segmentation task, a well-known Cascade Mask-RCNN framework [28] was employed. The dataset contains 118K training and 5K validation samples. For all the compared models, the backbones are first pretrained using ImageNet-1K. The pretrained models are then used to finetune on the detection and segmentation tasks.

Results on COCO. As shown in Table 4, A​PAP, A​P50AP_{50}, and A​P75AP_{75} are reported for comparison. The parameters and FLOPs are also reported as a reference for model complexity. The MaxViT backbone models, used in object detection and segmentation tasks, outperform all other backbones by large margins, including Swin, ConvNeXt, and UViT at various model sizes with respect to both accuracy and efficiency. Note that MaxViT-S outperforms other base-level models (e.g., Swin-B, UViT-B), with about 40% less computational cost.

| Backbone | Resolution | AP | AP50 | AP75 | APm | AP50m{}^{m}_{50} | AP75m{}^{m}_{75} | FLOPs | Pars. |
|---|---|---|---|---|---|---|---|---|---|
| ∙\bulletResNet-50 [29] | 1280×\times800 | 46.3 | 64.3 | 50.5 | 40.1 | 61.7 | 43.4 | 739G | 82M |
| ∙\bulletX101-32 [95] | 1280×\times800 | 48.1 | 66.5 | 52.4 | 41.6 | 63.9 | 45.2 | 819G | 101M |
| ∙\bulletX101-64 [95] | 1280×\times800 | 48.3 | 66.4 | 52.3 | 41.7 | 64.0 | 45.1 | 972G | 140M |
| ∙\bulletConvNeXt-T [57] | 1280×\times800 | 50.4 | 69.1 | 54.8 | 43.7 | 66.5 | 47.3 | 741G | - |
| ∙\bulletConvNeXt-S [57] | 1280×\times800 | 51.9 | 70.8 | 56.5 | 45.0 | 68.4 | 49.1 | 827G | - |
| ∙\bulletConvNeXt-B [57] | 1280×\times800 | 52.7 | 71.3 | 57.2 | 45.6 | 68.9 | 49.5 | 964G | - |
| ∘\circSwin-T [56] | 1280×\times800 | 50.4 | 69.2 | 54.7 | 43.7 | 66.6 | 47.3 | 745G | 86M |
| ∘\circSwin-S [56] | 1280×\times800 | 51.9 | 70.7 | 56.3 | 45.0 | 68.2 | 48.8 | 838G | 107M |
| ∘\circSwin-B [56] | 1280×\times800 | 51.9 | 70.5 | 56.4 | 45.0 | 68.1 | 48.9 | 982G | 145M |
| ∘\circUViT-T [14] | 896×\times896 | 51.1 | 70.4 | 56.2 | 43.6 | 67.7 | 47.2 | 613G | 47M |
| ∘\circUViT-S [14] | 896×\times896 | 51.4 | 70.8 | 56.2 | 44.1 | 68.2 | 48.0 | 744G | 54M |
| ∘\circUViT-B [14] | 896×\times896 | 52.5 | 72.0 | 57.6 | 44.3 | 68.7 | 48.3 | 975G | 74M |
| ∘\circAs-ViT-L [15] | 1024×\times1024 | 52.7 | 72.3 | 57.9 | 45.2 | 69.7 | 49.8 | 1094G | 139M |
| ⋄\diamondMaxViT-T | 896×\times896 | 52.1 | 71.9 | 56.8 | 44.6 | 69.1 | 48.4 | 475G | 69M |
| ⋄\diamondMaxViT-S | 896×\times896 | 53.1 | 72.5 | 58.1 | 45.4 | 69.8 | 49.5 | 595G | 107M |
| ⋄\diamondMaxViT-B | 896×\times896 | 53.4 | 72.9 | 58.1 | 45.7 | 70.3 | 50.0 | 856G | 157M |

### 4.3 Image Aesthetic Assessment.

Setting. We train and evaluate the MaxViT model on the AVA benchmark [61] which contains 255K images with aesthetics scores rated by amateur photographers. Similar to [77], we split the dataset into 80%/20% training and test sets. We followed [77] and used the normalized Earth Mover’s Distance as our training loss. We trained MaxViT at three different input resolutions: 2242224^{2}, 3842384^{2} and 5122512^{2}, initialized with ImageNet-1K pre-trained weights.

Results on AVA. To evaluate and compare our model against existing methods, we present a summary of our results in Table 6. For similar input resolutions, the proposed MaxViT-T model outperforms existing image aesthetic assessment methods. As the input resolution increases, the performance improves, benefiting from its strong non-local capacity. Also, MaxViT shows better linear correlation compared to the SOTA method [43] which uses multi-resolution inputs.

### 4.4 Image Generation

Setting. We evaluate the generative ability of MaxViT blocks to generate images of 128x128 resolution on ImageNet-1K. We choose the unconditional image generation to focus on the performance of different generators in GANs. We use the Inception Score (IS) [69] and the Fréchet Inception Distance (FID) [32] as quantitative evaluation metrics. 50,000 samples were randomly generated to calculate the FID and IS scores. We compared MaxViT against HiT [103], a SOTA generative Transformer model, which uses attention at low resolutions (e.g., 32, 64), and using implicit neural functions at high resolutions (e.g., 128). By contrast, MaxViT uses the proposed MaxViT block at every resolution. Note that we use an inverse block order (GA-BA-Conv) as we found it to perform better (see Table 10). Since Batch Normalization [39, 103] achieves better results on image generation, we replaced all Layer Norm with Batch Norm under this setting.

Results on ImageNet-1K. The results are shown in Table 6. Our MaxViT achieved better FID and IS with significantly lower number of parameters. These results demonstrate the effectiveness of MaxViT blocks for generation tasks. More details of the generative experiment can be found in Appendix.

| Table 5: Image aesthetic assessment results on the AVA benchmark [61]. PLCC and SRCC represent the Pearson’s linear and Spearman’s rank correlation coefficients. Model Res. Pars. PLCC↑\uparrow SRCC↑\uparrow ∙\bulletNIMA [77] 224224 56M 0.636 0.612 ∙\bulletEffNet-B0 [79] 224224 5.3M 0.642 0.620 ∙\bulletAFDC[10] 224224 44.5M 0.671 0.649 ∘\circViT-S/32 [43] 384384 22M 0.665 0.656 ∘\circViT-B/32 [43] 384384 88M 0.664 0.664 ∘\circMUSIQ [43] ∼512224\!\sim\!512 27M 0.720 0.706 ⋄\diamondMaxViT-T 224224 31M 0.707 0.685 ⋄\diamondMaxViT-T 384384 31M 0.736 0.699 ⋄\diamondMaxViT-T 512512 31M 0.745 0.708 | Model | Res. | Pars. | PLCC↑\uparrow | SRCC↑\uparrow | ∙\bulletNIMA [77] | 224224 | 56M | 0.636 | 0.612 | ∙\bulletEffNet-B0 [79] | 224224 | 5.3M | 0.642 | 0.620 | ∙\bulletAFDC[10] | 224224 | 44.5M | 0.671 | 0.649 | ∘\circViT-S/32 [43] | 384384 | 22M | 0.665 | 0.656 | ∘\circViT-B/32 [43] | 384384 | 88M | 0.664 | 0.664 | ∘\circMUSIQ [43] | ∼512224\!\sim\!512 | 27M | 0.720 | 0.706 | ⋄\diamondMaxViT-T | 224224 | 31M | 0.707 | 0.685 | ⋄\diamondMaxViT-T | 384384 | 31M | 0.736 | 0.699 | ⋄\diamondMaxViT-T | 512512 | 31M | 0.745 | 0.708 | Table 6: Comparison of image generation on ImageNet. ‡\ddagger used a pre-trained ImageNet classifier. Model FID↓\downarrow IS↑\uparrow ∙\bulletGAN [26] 54.17 14.01 ∙\bulletPacGAN2 [54] 57.51 13.50 ∙\bulletMGAN [34] 50.90 14.44 ∙\bulletLogoGAN [68]‡\ddagger 38.41 18.86 ∙\bulletSS-GAN [12] 43.87 - ∙\bulletSC GAN [55] 40.30 15.82 ∙\bulletConvNet-R1R_{1} [103] 37.18 19.55 ∘\circHiT [103] (32.9M) 30.83 21.64 ⋄\diamondMaxViT (18.6M) 30.77 22.58 | Model | FID↓\downarrow | IS↑\uparrow | ∙\bulletGAN [26] | 54.17 | 14.01 | ∙\bulletPacGAN2 [54] | 57.51 | 13.50 | ∙\bulletMGAN [34] | 50.90 | 14.44 | ∙\bulletLogoGAN [68]‡\ddagger | 38.41 | 18.86 | ∙\bulletSS-GAN [12] | 43.87 | - | ∙\bulletSC GAN [55] | 40.30 | 15.82 | ∙\bulletConvNet-R1R_{1} [103] | 37.18 | 19.55 | ∘\circHiT [103] (32.9M) | 30.83 | 21.64 | ⋄\diamondMaxViT (18.6M) | 30.77 | 22.58 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Model | Res. | Pars. | PLCC↑\uparrow | SRCC↑\uparrow |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∙\bulletNIMA [77] | 224224 | 56M | 0.636 | 0.612 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∙\bulletEffNet-B0 [79] | 224224 | 5.3M | 0.642 | 0.620 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∙\bulletAFDC[10] | 224224 | 44.5M | 0.671 | 0.649 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∘\circViT-S/32 [43] | 384384 | 22M | 0.665 | 0.656 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∘\circViT-B/32 [43] | 384384 | 88M | 0.664 | 0.664 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∘\circMUSIQ [43] | ∼512224\!\sim\!512 | 27M | 0.720 | 0.706 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ⋄\diamondMaxViT-T | 224224 | 31M | 0.707 | 0.685 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ⋄\diamondMaxViT-T | 384384 | 31M | 0.736 | 0.699 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ⋄\diamondMaxViT-T | 512512 | 31M | 0.745 | 0.708 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Model | FID↓\downarrow | IS↑\uparrow |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∙\bulletGAN [26] | 54.17 | 14.01 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∙\bulletPacGAN2 [54] | 57.51 | 13.50 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∙\bulletMGAN [34] | 50.90 | 14.44 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∙\bulletLogoGAN [68]‡\ddagger | 38.41 | 18.86 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∙\bulletSS-GAN [12] | 43.87 | - |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∙\bulletSC GAN [55] | 40.30 | 15.82 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∙\bulletConvNet-R1R_{1} [103] | 37.18 | 19.55 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ∘\circHiT [103] (32.9M) | 30.83 | 21.64 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ⋄\diamondMaxViT (18.6M) | 30.77 | 22.58 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

| Model | Res. | Pars. | PLCC↑\uparrow | SRCC↑\uparrow |
|---|---|---|---|---|
| ∙\bulletNIMA [77] | 224224 | 56M | 0.636 | 0.612 |
| ∙\bulletEffNet-B0 [79] | 224224 | 5.3M | 0.642 | 0.620 |
| ∙\bulletAFDC[10] | 224224 | 44.5M | 0.671 | 0.649 |
| ∘\circViT-S/32 [43] | 384384 | 22M | 0.665 | 0.656 |
| ∘\circViT-B/32 [43] | 384384 | 88M | 0.664 | 0.664 |
| ∘\circMUSIQ [43] | ∼512224\!\sim\!512 | 27M | 0.720 | 0.706 |
| ⋄\diamondMaxViT-T | 224224 | 31M | 0.707 | 0.685 |
| ⋄\diamondMaxViT-T | 384384 | 31M | 0.736 | 0.699 |
| ⋄\diamondMaxViT-T | 512512 | 31M | 0.745 | 0.708 |

| Model | FID↓\downarrow | IS↑\uparrow |
|---|---|---|
| ∙\bulletGAN [26] | 54.17 | 14.01 |
| ∙\bulletPacGAN2 [54] | 57.51 | 13.50 |
| ∙\bulletMGAN [34] | 50.90 | 14.44 |
| ∙\bulletLogoGAN [68]‡\ddagger | 38.41 | 18.86 |
| ∙\bulletSS-GAN [12] | 43.87 | - |
| ∙\bulletSC GAN [55] | 40.30 | 15.82 |
| ∙\bulletConvNet-R1R_{1} [103] | 37.18 | 19.55 |
| ∘\circHiT [103] (32.9M) | 30.83 | 21.64 |
| ⋄\diamondMaxViT (18.6M) | 30.77 | 22.58 |

### 4.5 Ablation Studies.

In this section, we ablate important design choices in MaxViT on ImageNet-1K image classification. We use the MaxViT-T model trained for 300 epochs by default and report top-1 accuracy on ImageNet-1K. Except for the ablated design choice, we used the same training configurations, unless stated otherwise.

| Table 7: Effects of global grid-attention. Ablate-S1 means we remove grid-attention in stage 1 while Replace-S1 means replacing grid-attention with block-attention. Model Pars. FLOPs Top-1 Acc. MaxViT-T 30.9M 5.6G 83.62 Ablate-S1 30.8M 5.3G 83.36(-0.26) Ablate-S2 30.5M 5.3G 83.38(-0.24) Ablate-S3 26.9M 4.9G 83.00(-0.62) Replace-S1 30.9M 5.6G 83.49(-0.13) Replace-S2 30.9M 5.6G 83.41(-0.22) Replace-S3 30.9M 5.6G 83.40(-0.23) | Model | Pars. | FLOPs | Top-1 Acc. | MaxViT-T | 30.9M | 5.6G | 83.62 | Ablate-S1 | 30.8M | 5.3G | 83.36(-0.26) | Ablate-S2 | 30.5M | 5.3G | 83.38(-0.24) | Ablate-S3 | 26.9M | 4.9G | 83.00(-0.62) | Replace-S1 | 30.9M | 5.6G | 83.49(-0.13) | Replace-S2 | 30.9M | 5.6G | 83.41(-0.22) | Replace-S3 | 30.9M | 5.6G | 83.40(-0.23) | Table 8: Block order study. C, BA, GA represent MBConv, block-, and grid-attention respectively. Model Pars. FLOPs Top-1 acc. C-BA-GA 30.9M 5.6G 83.62 C-GA-BA 30.9M 5.6G 83.54(-0.08) BA-C-GA 31.1M 5.3G 83.07(-0.55) BA-GA-C 31.1M 5.3G 83.02(-0.60) GA-C-BA 31.1M 5.3G 83.08(-0.54) GA-BA-C 31.1M 5.3G 83.03(-0.59) GAN experiments Model Pars. FID↓\downarrow IS↑\uparrow GA-BA-C 18.6M 30.77 22.68 C-BA-GA 18.6M 31.40 21.49(-1.19) | Model | Pars. | FLOPs | Top-1 acc. | C-BA-GA | 30.9M | 5.6G | 83.62 | C-GA-BA | 30.9M | 5.6G | 83.54(-0.08) | BA-C-GA | 31.1M | 5.3G | 83.07(-0.55) | BA-GA-C | 31.1M | 5.3G | 83.02(-0.60) | GA-C-BA | 31.1M | 5.3G | 83.08(-0.54) | GA-BA-C | 31.1M | 5.3G | 83.03(-0.59) | GAN experiments | Model | Pars. | FID↓\downarrow | IS↑\uparrow | GA-BA-C | 18.6M | 30.77 | 22.68 | C-BA-GA | 18.6M | 31.40 | 21.49(-1.19) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Model | Pars. | FLOPs | Top-1 Acc. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MaxViT-T | 30.9M | 5.6G | 83.62 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Ablate-S1 | 30.8M | 5.3G | 83.36(-0.26) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Ablate-S2 | 30.5M | 5.3G | 83.38(-0.24) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Ablate-S3 | 26.9M | 4.9G | 83.00(-0.62) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Replace-S1 | 30.9M | 5.6G | 83.49(-0.13) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Replace-S2 | 30.9M | 5.6G | 83.41(-0.22) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Replace-S3 | 30.9M | 5.6G | 83.40(-0.23) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Model | Pars. | FLOPs | Top-1 acc. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| C-BA-GA | 30.9M | 5.6G | 83.62 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| C-GA-BA | 30.9M | 5.6G | 83.54(-0.08) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| BA-C-GA | 31.1M | 5.3G | 83.07(-0.55) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| BA-GA-C | 31.1M | 5.3G | 83.02(-0.60) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GA-C-BA | 31.1M | 5.3G | 83.08(-0.54) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GA-BA-C | 31.1M | 5.3G | 83.03(-0.59) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GAN experiments |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Model | Pars. | FID↓\downarrow | IS↑\uparrow |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GA-BA-C | 18.6M | 30.77 | 22.68 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| C-BA-GA | 18.6M | 31.40 | 21.49(-1.19) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Table 9: Ablation of MBConv. Ablate-S1 means we delete MBConv layers in stage 1. Note that the network will also be smaller if we ablate MBConv layers in some stage. Model Pars. FLOPs Top-1 acc. MaxViT-T 30.9M 5.6G 83.62 Ablate-S1 30.8M 5.2G 83.24(-0.38) Ablate-S2 30.5M 5.4G 83.02(-0.60) Ablate-S3 27.6M 5.1G 82.65(-0.97) Ablate-S4 25.7M 5.4G 83.09(-0.53) | Model | Pars. | FLOPs | Top-1 acc. | MaxViT-T | 30.9M | 5.6G | 83.62 | Ablate-S1 | 30.8M | 5.2G | 83.24(-0.38) | Ablate-S2 | 30.5M | 5.4G | 83.02(-0.60) | Ablate-S3 | 27.6M | 5.1G | 82.65(-0.97) | Ablate-S4 | 25.7M | 5.4G | 83.09(-0.53) | Table 10: Sequential vs. parallel. We compared our model with modified parallel multi-axis scheme P​a​r​a​lParal-⋆\star. Model Pars. FLOPs Top-1 acc. MaxViT-T 30.9M 5.6G 83.62 P​a​r​a​lParal-T 34.5M 6.2G 82.64(-0.98) MaxViT-S 68.9M 11.7G 84.45 P​a​r​a​lParal-S 76.9M 13.0G 83.45(-1.00) MaxViT-B 119.4M 24.2G 84.95 P​a​r​a​lParal-B 133.4M 26.9G 83.70(-1.25) MaxViT-L 211.8M 43.9G 85.17 P​a​r​a​lParal-L 236.6M 48.8G 83.54(-1.63) | Model | Pars. | FLOPs | Top-1 acc. | MaxViT-T | 30.9M | 5.6G | 83.62 | P​a​r​a​lParal-T | 34.5M | 6.2G | 82.64(-0.98) | MaxViT-S | 68.9M | 11.7G | 84.45 | P​a​r​a​lParal-S | 76.9M | 13.0G | 83.45(-1.00) | MaxViT-B | 119.4M | 24.2G | 84.95 | P​a​r​a​lParal-B | 133.4M | 26.9G | 83.70(-1.25) | MaxViT-L | 211.8M | 43.9G | 85.17 | P​a​r​a​lParal-L | 236.6M | 48.8G | 83.54(-1.63) |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Model | Pars. | FLOPs | Top-1 acc. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MaxViT-T | 30.9M | 5.6G | 83.62 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Ablate-S1 | 30.8M | 5.2G | 83.24(-0.38) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Ablate-S2 | 30.5M | 5.4G | 83.02(-0.60) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Ablate-S3 | 27.6M | 5.1G | 82.65(-0.97) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Ablate-S4 | 25.7M | 5.4G | 83.09(-0.53) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Model | Pars. | FLOPs | Top-1 acc. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MaxViT-T | 30.9M | 5.6G | 83.62 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| P​a​r​a​lParal-T | 34.5M | 6.2G | 82.64(-0.98) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MaxViT-S | 68.9M | 11.7G | 84.45 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| P​a​r​a​lParal-S | 76.9M | 13.0G | 83.45(-1.00) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MaxViT-B | 119.4M | 24.2G | 84.95 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| P​a​r​a​lParal-B | 133.4M | 26.9G | 83.70(-1.25) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MaxViT-L | 211.8M | 43.9G | 85.17 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| P​a​r​a​lParal-L | 236.6M | 48.8G | 83.54(-1.63) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

| Model | Pars. | FLOPs | Top-1 Acc. |
|---|---|---|---|
| MaxViT-T | 30.9M | 5.6G | 83.62 |
| Ablate-S1 | 30.8M | 5.3G | 83.36(-0.26) |
| Ablate-S2 | 30.5M | 5.3G | 83.38(-0.24) |
| Ablate-S3 | 26.9M | 4.9G | 83.00(-0.62) |
| Replace-S1 | 30.9M | 5.6G | 83.49(-0.13) |
| Replace-S2 | 30.9M | 5.6G | 83.41(-0.22) |
| Replace-S3 | 30.9M | 5.6G | 83.40(-0.23) |

| Model | Pars. | FLOPs | Top-1 acc. |
|---|---|---|---|
| C-BA-GA | 30.9M | 5.6G | 83.62 |
| C-GA-BA | 30.9M | 5.6G | 83.54(-0.08) |
| BA-C-GA | 31.1M | 5.3G | 83.07(-0.55) |
| BA-GA-C | 31.1M | 5.3G | 83.02(-0.60) |
| GA-C-BA | 31.1M | 5.3G | 83.08(-0.54) |
| GA-BA-C | 31.1M | 5.3G | 83.03(-0.59) |
| GAN experiments |  |  |  |
| Model | Pars. | FID↓\downarrow | IS↑\uparrow |
| GA-BA-C | 18.6M | 30.77 | 22.68 |
| C-BA-GA | 18.6M | 31.40 | 21.49(-1.19) |

| Model | Pars. | FLOPs | Top-1 acc. |
|---|---|---|---|
| MaxViT-T | 30.9M | 5.6G | 83.62 |
| Ablate-S1 | 30.8M | 5.2G | 83.24(-0.38) |
| Ablate-S2 | 30.5M | 5.4G | 83.02(-0.60) |
| Ablate-S3 | 27.6M | 5.1G | 82.65(-0.97) |
| Ablate-S4 | 25.7M | 5.4G | 83.09(-0.53) |

| Model | Pars. | FLOPs | Top-1 acc. |
|---|---|---|---|
| MaxViT-T | 30.9M | 5.6G | 83.62 |
| P​a​r​a​lParal-T | 34.5M | 6.2G | 82.64(-0.98) |
| MaxViT-S | 68.9M | 11.7G | 84.45 |
| P​a​r​a​lParal-S | 76.9M | 13.0G | 83.45(-1.00) |
| MaxViT-B | 119.4M | 24.2G | 84.95 |
| P​a​r​a​lParal-B | 133.4M | 26.9G | 83.70(-1.25) |
| MaxViT-L | 211.8M | 43.9G | 85.17 |
| P​a​r​a​lParal-L | 236.6M | 48.8G | 83.54(-1.63) |

Global grid-attention. One of our main contributions is the grid-attention module, which allows for sparse global interactions at linear time, enabling our model to capture global information at all stages. We conducted two ablations to understand its gain: 1) completely removed global attention at each stage; 2) replaced grid attention with block attention to retain the same parameter count and FLOPs. As Table 10 shows, enabling global attention at earlier stages can further boost performance over using only local attention or convolutions.

MBConv layer. We also ablated the usage of MBConv layers in MaxViT by removing all MBConv in each stage. Note that we should also consider the reduction of parameter count and FLOPs when removing the MBConv layers. Plus, Stage 3 has 5 blocks whereas other stages have only 2. As Table 10 shows, the usage of MBConv layers in MaxViT significantly boosts performance.

Block order study. We present three different modules to build the MaxViT block – MBConv, block-, and grid-attention – which captures spatial interactions from local to global. To investigate the most effective way to combine them, we evaluated the MaxViT-T model using all 6 permutations. We always apply downsampling in the first layer, which might cause a minor model size difference. We can observe from Table 10 that placing MBConv before attention layers is almost always better than other combinations. The reason might be that it is more suitable to get local features/patterns in early layers, then aggregate them globally, which is aligned with existing hybrid models [19, 94], which puts Conv layers in front of attention. In generative experiments (Section 4.4), however, we found the best order to be from global to local: GA-BA-C. We hypothesize that it may be advantageous for generation tasks to first obtain the overall structures correct with global processing blocks (i.e., grid-attention layers), then fill in finer details using local processing blocks (i.e., MBConv).

Sequential vs. parallel. In our approach, we sequentially stack the multi-axis attention modules following [56, 86], while there also exist other models that adopt a parallel design [103, 83]. In this ablation, we compare our sequential Max-SA against parallel branches containing block- and grid-attention respectively. Note that we use an input projection to double the channels, then split the heads to feed the two branches in order to remain similar complexity to MaxViT, and an output projection that reduces the concatenated branches. We did rough parameter tuning and found that an initial learning rate of 10−310^{-3} performs significantly better than 3×10−33\times 10^{-3} for parallel models. We use all the same parameters except the learning rate. As Table 10 shows, our sequential approach remarkably outperforms parallel counterparts with fewer parameters and computation. The reason may be that the parallel designs learn complementary cues with less interactions between them, whereas our sequential stack is able to learn more powerful fusions between local and global layers.

Vertical layout. We further examine our vertical layout design, i.e., the number of blocks each stage. We compared our design against the choice of Swin/ConvNeXt [56, 57]. We change MaxViT-T and -S to blocks B=(2,2,6,2)B=(2,2,6,2), and MaxViT-B, -L to have blocks B=(2,2,18,2)B=(2,2,18,2) strictly following the stage ratio of Swin [56]. It may be seen from Figure 5 that our layout performed comparably to Swin for small models, but scales significantly better for larger models.

## 5 Discussion and Conclusion

While recent works in the 2020s have arguably shown that ConvNets and vision Transformers can achieve similar performance on image recognition, our work presents a unified design that takes advantages of the best of both worlds – efficient convolution and sparse attention – and demonstrates that a model built on top, namely MaxViT, can achieve state-of-the-art performance on a variety of vision tasks, and more importantly, scale extremely well to massive scale data sizes. Even though we present our model in the context of vision tasks, the proposed multi-axis approach can easily extend to language modeling to capture both local and global dependencies in linear time. We also look forward to studying other forms of sparse attention in higher-dimensional or multi-modal signals such as videos, point clouds, and vision-languages.

Societal impact. Investigating the performance and scalability of large model designs would consume considerable computing resources. These efforts can contribute to increased carbon emissions, which could hence raise environmental concerns. However, the proposed model offers strong modular candidates that expand the network’s design space for future efforts on automated architectural design. If trained improperly, the proposed model may express bias and fairness issues. The proposed generative model can be abused to generate misleading media and fake news. These issues demand caution in future related research.

Acknowledgment. We thank Xianzhi Du and Wuyang Chen for extensive help on experiments. We also thank Hanxiao Liu, Zihang Dai, Anurag Arnab, Huiwen Chang, Junjie Ke, Mauricio Delbracio, Sungjoon Choi, and Irene Zhu for valuable discussions and help.

## Appendix

In this Appendix we provide the following material:

- • Sec. 0.A describes the detailed architectures of MaxViT for image classification (Sec. 0.A.1), object detection and segmentation (Sec. 0.A.2), image aesthetics assessment (Sec. 0.A.3), and image generation (Sec. 0.A.4).
Sec. 0.A describes the detailed architectures of MaxViT for image classification (Sec. 0.A.1), object detection and segmentation (Sec. 0.A.2), image aesthetics assessment (Sec. 0.A.3), and image generation (Sec. 0.A.4).

- • Sec. 0.B presents complete training settings and hyperparameters for image classification (Sec. 0.B.1), object detection and segmentation (Sec. 0.B.2), image aesthetics assessment (Sec. 0.B.3), and image generation (Sec. 0.B.4).
Sec. 0.B presents complete training settings and hyperparameters for image classification (Sec. 0.B.1), object detection and segmentation (Sec. 0.B.2), image aesthetics assessment (Sec. 0.B.3), and image generation (Sec. 0.B.4).

- • Sec. 0.C demonstrates comprehensive experimental results, including image classification on ImageNet-1K (Table 13), ImageNet-21K and JFT (Table 14), as well as more image generation visualizations on ImageNet-1K (Figure 8).
Sec. 0.C demonstrates comprehensive experimental results, including image classification on ImageNet-1K (Table 13), ImageNet-21K and JFT (Table 14), as well as more image generation visualizations on ImageNet-1K (Figure 8).

## Appendix 0.A Model Details

### 0.A.1 Backbone Details

MaxViT leverages the MBConv block [70, 79] as the main convolution operator. We also adopt a pre-activation structure [30, 19] to promote homogeneity between MBConv and Transformer blocks. Specifically, assume 𝐱\mathbf{x} to be the input feature, the MBConv block without downsampling is formulated as:

|  | 𝐱←𝐱+𝖯𝗋𝗈𝗃⁡(𝖲𝖤⁡(𝖣𝖶𝖢𝗈𝗇𝗏⁡(𝖢𝗈𝗇𝗏⁡(𝖭𝗈𝗋𝗆⁡(𝐱))))),\mathbf{x}\leftarrow\mathbf{x}+\mathsf{Proj}(\mathsf{SE}(\mathsf{DWConv}(\mathsf{Conv}(\mathsf{Norm}(\mathbf{x}))))), |  | (1) |
|---|---|---|---|

where 𝖭𝗈𝗋𝗆\mathsf{Norm} is 𝖡𝖺𝗍𝖼𝗁𝖭𝗈𝗋𝗆\mathsf{BatchNorm} [39], 𝖢𝗈𝗇𝗏\mathsf{Conv} is the expansion Conv1x1 followed by 𝖡𝖺𝗍𝖼𝗁𝖭𝗈𝗋𝗆\mathsf{BatchNorm} and 𝖦𝖤𝖫𝖴\mathsf{GELU} [31] activation, a typical choice for Transformer-based models. 𝖣𝖶𝖢𝗈𝗇𝗏\mathsf{DWConv} is the Depthwise Conv3x3 followed by 𝖡𝖺𝗍𝖼𝗁𝖭𝗈𝗋𝗆\mathsf{BatchNorm} and 𝖦𝖤𝖫𝖴\mathsf{GELU}. 𝖲𝖤\mathsf{SE} is the Squeeze-Excitation layer [36], while 𝖯𝗋𝗈𝗃\mathsf{Proj} is the shrink Conv1x1 to down-project the number of channels. Note that for the first MBConv block in every stage, the downsampling is done by applying stride-2 Depthwise Conv3x3 while the shortcut branch should also apply pooling and channel projection:

|  | 𝐱←𝖯𝗋𝗈𝗃⁡(𝖯𝗈𝗈𝗅𝟤𝖣⁡(𝐱))+𝖯𝗋𝗈𝗃⁡(𝖲𝖤⁡(𝖣𝖶𝖢𝗈𝗇𝗏↓(𝖢𝗈𝗇𝗏⁡(𝖭𝗈𝗋𝗆⁡(𝐱))))).\mathbf{x}\leftarrow\mathsf{Proj}(\mathsf{Pool2D}(\mathbf{x}))+\mathsf{Proj}(\mathsf{SE}(\mathsf{DWConv}\!\downarrow\!(\mathsf{Conv}(\mathsf{Norm}(\mathbf{x}))))). |  | (2) |
|---|---|---|---|

Relative attention has been explored in several previous studies for both NLP [71, 92] and vision [56, 84, 19, 40]. Here to simplify the presentation, we present our model using only a single head of the multi-head self-attention. In the actual implementation, we always use multi-head attention with the same head dimension. The relative attention can be defined as:

|  | 𝖱𝖾𝗅𝖠𝗍𝗍𝖾𝗇𝗍𝗂𝗈𝗇⁡(Q,K,V)=𝗌𝗈𝖿𝗍𝗆𝖺𝗑⁡(Q​KT/d+B)​V,\mathsf{RelAttention}(Q,K,V)=\mathsf{softmax}(QK^{T}/\sqrt{d}+B)V, |  | (3) |
|---|---|---|---|

where Q,K,V∈ℝ(H×W)×CQ,K,V\in\mathbb{R}^{(H\times W)\times C} are the query, key, and value matrices and dd is the hidden dimension. The attention weights are co-decided by a learned static location-aware matrix BB and the scaled input-adaptive attention Q​KT/dQK^{T}/\sqrt{d}. Considering the differences in 2D coordinates, the relative position bias BB is parameterized by a matrix B^∈ℝ(2​H−1)​(2​W−1)\hat{B}\in\mathbb{R}^{(2H-1)(2W-1)}. Following typical practices [56, 19], when fine-tuned at a higher resolution e.g., H′×W′H^{\prime}\times W^{\prime}, we use bilinear interpolation to map the relative positional bias from ℝ(2​H−1)​(2​W−1)\mathbb{R}^{(2H-1)(2W-1)} to ℝ(2​H′−1)​(2​W′−1)\mathbb{R}^{(2H^{\prime}-1)(2W^{\prime}-1)}. This relative attention benefits from input-adaptivity, translation equivariance, and global interactions, which is a preferred choice over the vanilla self-attention on 2D vision tasks. In our model, all the attention operators use this relative attention defined in Eq. 3 by default.

We assume the relative attention operator in Eq. 3 follows the convention for 1D input sequences i.e., always regards the second last dimension of an input (…,L,C)(...,L,C) as the spatial axis where L,C{L},C represent sequence length and channels. The proposed Multi-Axis Attention can be implemented without modification to the self-attention operation. To start with, we first define the 𝖡𝗅𝗈𝖼𝗄⁡(⋅)\mathsf{Block}(\cdot) operator with parameter PP as partitioning the input image/feature 𝐱∈ℝH×W×C\mathbf{x}\in\mathbb{R}^{H\times W\times C} into non-overlapping blocks with each block having size P×PP\times P. Note that after window partition, the block dimensions are gathered onto the spatial dimension (i.e., -2 axis):

|  | 𝖡𝗅𝗈𝖼𝗄:(H,W,C)→(HP×P,WP×P,C)→(H​WP2,P2,C).\mathsf{Block}:(H,W,C)\rightarrow(\frac{H}{P}\times P,\frac{W}{P}\times P,C)\rightarrow(\frac{HW}{P^{2}},P^{2},C). |  | (4) |
|---|---|---|---|

We denote the 𝖴𝗇𝖻𝗅𝗈𝖼𝗄⁡(⋅)\mathsf{Unblock}(\cdot) operation as the reverse of the above block partition procedure. Similarly, we define the 𝖦𝗋𝗂𝖽⁡(⋅)\mathsf{Grid}(\cdot) operation with parameter GG as dividing the input feature into a uniform G×GG\times G grid, with each lattice having adaptive size HG×WG\frac{H}{G}\times\frac{W}{G}. Unlike the 𝖻𝗅𝗈𝖼𝗄\mathsf{block} operator, we need to apply an extra 𝖳𝗋𝖺𝗇𝗌𝗉𝗈𝗌𝖾\mathsf{Transpose} to place the grid dimension in the assumed spatial axis (i.e., -2 axis):

|  | 𝖦𝗋𝗂𝖽:(H,W,C)→(G×HG,G×WG,C)→(G2,H​WG2,C)→(H​WG2,G2,C)⏟swapaxes(axis1=-2,axis2=-3)\mathsf{Grid}:(H,W,C)\rightarrow(G\times\frac{H}{G},G\times\frac{W}{G},C)\rightarrow\underbrace{(G^{2},\frac{HW}{G^{2}},C)\rightarrow(\frac{HW}{G^{2}},G^{2},C)}_{\text{swapaxes(axis1=-2,axis2=-3)}} |  | (5) |
|---|---|---|---|

with its inverse operation 𝖴𝗇𝗀𝗋𝗂𝖽⁡(⋅)\mathsf{Ungrid}(\cdot) that reverses the gridded input back to the normal 2D feature space.

To this end, we are ready to explain the multi-axis attention module. Given an input tensor 𝐱∈ℝH×W×C\mathbf{x}\in\mathbb{R}^{H\times W\times C}, the local Block Attention can be expressed as:

|  | 𝐱←𝐱+𝖴𝗇𝖻𝗅𝗈𝖼𝗄⁡(𝖱𝖾𝗅𝖠𝗍𝗍𝖾𝗇𝗍𝗂𝗈𝗇⁡(𝖡𝗅𝗈𝖼𝗄⁡(𝖫𝖭⁡(𝐱))))𝐱←𝐱+𝖬𝖫𝖯⁡(𝖫𝖭⁡(𝐱))\displaystyle\begin{split}\mathbf{x}&\leftarrow\mathbf{x}+\mathsf{Unblock}(\mathsf{RelAttention}(\mathsf{Block}(\mathsf{LN}(\mathbf{x}))))\\ \mathbf{x}&\leftarrow\mathbf{x}+\mathsf{MLP}(\mathsf{LN}(\mathbf{x}))\\ \end{split} |  | (6) |
|---|---|---|---|

while the global, dilated Grid Attention module is formulated as:

|  | 𝐱←𝐱+𝖴𝗇𝗀𝗋𝗂𝖽⁡(𝖱𝖾𝗅𝖠𝗍𝗍𝖾𝗇𝗍𝗂𝗈𝗇⁡(𝖦𝗋𝗂𝖽⁡(𝖫𝖭⁡(𝐱))))𝐱←𝐱+𝖬𝖫𝖯⁡(𝖫𝖭⁡(𝐱))\displaystyle\begin{split}\mathbf{x}&\leftarrow\mathbf{x}+\mathsf{Ungrid}(\mathsf{RelAttention}(\mathsf{Grid}(\mathsf{LN}(\mathbf{x}))))\\ \mathbf{x}&\leftarrow\mathbf{x}+\mathsf{MLP}(\mathsf{LN}(\mathbf{x}))\\ \end{split} |  | (7) |
|---|---|---|---|

where we omit the Q​K​VQKV input format in the 𝖱𝖾𝗅𝖠𝗍𝗍𝖾𝗇𝗍𝗂𝗈𝗇\mathsf{RelAttention} operation for simplicity. 𝖫𝖭\mathsf{LN} denotes the Layer Normalization [2], where 𝖬𝖫𝖯\mathsf{MLP} is a standard MLP network [22, 56] consisting of two linear layers: 𝐱←W2​𝖦𝖤𝖫𝖴​(W1​𝐱)\mathbf{x}\leftarrow W_{2}\mathsf{GELU}(W_{1}\mathbf{x}).

It should be noted that our proposed multi-axis attention (Max-SA) module is completely different from the axial attention proposed in [33, 86]. As shown in Figure 6(a), Axial attention proposes to first apply column-wise attention then row-wise, which achieves a global receptive field with 𝒪⁡(N​N)\mathcal{O}(N\sqrt{N}) complexity (assuming NN equals to the number of pixels). On the contrary, our proposed Max-SA shown in Figure 6(b) first employs local attention, then sparse global attention, enjoying global receptive fields with only 𝒪⁡(N)\mathcal{O}(N) linear complexity. Moreover, we deem the proposed Max-SA a more natural approach for vision since the design of attended regions account for the 2D structure of images, e.g., mixing tokens in a spatially-local small window.

We demonstrate in Algo. 1 an einops-style pseudocode of the MaxViT block which contains MBConv, block attention, and grid attention.

Instead of using the [cls] token [22], we simply apply global average pooling to the output of the last stage (S4) to obtain the feature representation, followed by the final classification head.

Finally, we present detailed architectural specifications for the MaxViT model family (T/S/B/L) in Table 11.

|  | dsp. rate (out size) | MaxViT-T | MaxViT-S |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
| stem | 2×2\times (×112112\!\times\!112) | 2×2\times | (×112112\!\times\!112) | 3×\times3, 64, stride 2 3×\times3, 64, stride 1 | 3×\times3, 64, stride 2 | 3×\times3, 64, stride 1 | 3×\times3, 64, stride 2 3×\times3, 64, stride 1 | 3×\times3, 64, stride 2 | 3×\times3, 64, stride 1 |
| 2×2\times |  |  |  |  |  |  |  |  |  |
| (×112112\!\times\!112) |  |  |  |  |  |  |  |  |  |
| 3×\times3, 64, stride 2 |  |  |  |  |  |  |  |  |  |
| 3×\times3, 64, stride 1 |  |  |  |  |  |  |  |  |  |
| 3×\times3, 64, stride 2 |  |  |  |  |  |  |  |  |  |
| 3×\times3, 64, stride 1 |  |  |  |  |  |  |  |  |  |
| S1 | 4×4\times (56×5656\times 56) | 4×4\times | (56×5656\times 56) | [MBConv, 64, E 4, R 4Rel-MSA, P 7×7, H 2Rel-MSA, G 7×7, H 2]×2\left[\begin{array}[]{c}\text{MBConv, 64, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 2}\\ \text{Rel-MSA, G 7$\times$7, H 2}\end{array}\right]\times 2 | [MBConv, 96, E 4, R 4Rel-MSA, P 7×7, H 3Rel-MSA, G 7×7, H 3]×2\left[\begin{array}[]{c}\text{MBConv, 96, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 3}\\ \text{Rel-MSA, G 7$\times$7, H 3}\end{array}\right]\times 2 |  |  |  |  |
| 4×4\times |  |  |  |  |  |  |  |  |  |
| (56×5656\times 56) |  |  |  |  |  |  |  |  |  |
| S2 | 8×8\times (28×2828\times 28) | 8×8\times | (28×2828\times 28) | [MBConv, 128, E 4, R 4Rel-MSA, P 7×7, H 4Rel-MSA, G 7×7, H 4]×2\left[\begin{array}[]{c}\text{MBConv, 128, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 4}\\ \text{Rel-MSA, G 7$\times$7, H 4}\end{array}\right]\times 2 | [MBConv, 192, E 4, R 4Rel-MSA, P 7×7, H 6Rel-MSA, G 7×7, H 6]×2\left[\begin{array}[]{c}\text{MBConv, 192, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 6}\\ \text{Rel-MSA, G 7$\times$7, H 6}\end{array}\right]\times 2 |  |  |  |  |
| 8×8\times |  |  |  |  |  |  |  |  |  |
| (28×2828\times 28) |  |  |  |  |  |  |  |  |  |
| S3 | 16×16\times (14×1414\times 14) | 16×16\times | (14×1414\times 14) | [MBConv, 256, E 4, R 4Rel-MSA, P 7×7, H 8Rel-MSA, G 7×7, H 8]×5\left[\begin{array}[]{c}\text{MBConv, 256, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 8}\\ \text{Rel-MSA, G 7$\times$7, H 8}\end{array}\right]\times 5 | [MBConv, 384, E 4, R 4Rel-MSA, P 7×7, H 12Rel-MSA, G 7×7, H 12]×5\left[\begin{array}[]{c}\text{MBConv, 384, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 12}\\ \text{Rel-MSA, G 7$\times$7, H 12}\end{array}\right]\times 5 |  |  |  |  |
| 16×16\times |  |  |  |  |  |  |  |  |  |
| (14×1414\times 14) |  |  |  |  |  |  |  |  |  |
| S4 | 32×32\times (7×77\times 7) | 32×32\times | (7×77\times 7) | [MBConv, 512, E 4, R 4Rel-MSA, P 7×7, H 16Rel-MSA, G 7×7, H 16]×2\left[\begin{array}[]{c}\text{MBConv, 512, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 16}\\ \text{Rel-MSA, G 7$\times$7, H 16}\end{array}\right]\times 2 | [MBConv, 768, E 4, R 4Rel-MSA, P 7×7, H 24Rel-MSA, G 7×7, H 24]×2\left[\begin{array}[]{c}\text{MBConv, 768, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 24}\\ \text{Rel-MSA, G 7$\times$7, H 24}\end{array}\right]\times 2 |  |  |  |  |
| 32×32\times |  |  |  |  |  |  |  |  |  |
| (7×77\times 7) |  |  |  |  |  |  |  |  |  |
|  | dsp. rate (out size) | MaxViT-B | MaxViT-L |  |  |  |  |  |  |
| stem | 2×2\times (×112112\!\times\!112) | 2×2\times | (×112112\!\times\!112) | 3×\times3, 64, stride 2 3×\times3, 64, stride 1 | 3×\times3, 64, stride 2 | 3×\times3, 64, stride 1 | 3×\times3, 128, stride 2 3×\times3, 128, stride 1 | 3×\times3, 128, stride 2 | 3×\times3, 128, stride 1 |
| 2×2\times |  |  |  |  |  |  |  |  |  |
| (×112112\!\times\!112) |  |  |  |  |  |  |  |  |  |
| 3×\times3, 64, stride 2 |  |  |  |  |  |  |  |  |  |
| 3×\times3, 64, stride 1 |  |  |  |  |  |  |  |  |  |
| 3×\times3, 128, stride 2 |  |  |  |  |  |  |  |  |  |
| 3×\times3, 128, stride 1 |  |  |  |  |  |  |  |  |  |
| S1 | 4×4\times (56×5656\times 56) | 4×4\times | (56×5656\times 56) | [MBConv, 96, E 4, R 4Rel-MSA, P 7×7, H 3Rel-MSA, G 7×7, H 3]×2\left[\begin{array}[]{c}\text{MBConv, 96, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 3}\\ \text{Rel-MSA, G 7$\times$7, H 3}\end{array}\right]\times 2 | [MBConv, 128, E 4, R 4Rel-MSA, P 7×7, H 4Rel-MSA, G 7×7, H 4]×2\left[\begin{array}[]{c}\text{MBConv, 128, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 4}\\ \text{Rel-MSA, G 7$\times$7, H 4}\end{array}\right]\times 2 |  |  |  |  |
| 4×4\times |  |  |  |  |  |  |  |  |  |
| (56×5656\times 56) |  |  |  |  |  |  |  |  |  |
| S2 | 8×8\times (28×2828\times 28) | 8×8\times | (28×2828\times 28) | [MBConv, 192, E 4, R 4Rel-MSA, P 7×7, H 6Rel-MSA, G 7×7, H 6]×6\left[\begin{array}[]{c}\text{MBConv, 192, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 6}\\ \text{Rel-MSA, G 7$\times$7, H 6}\end{array}\right]\times 6 | [MBConv, 256, E 4, R 4Rel-MSA, P 7×7, H 8Rel-MSA, G 7×7, H 8]×6\left[\begin{array}[]{c}\text{MBConv, 256, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 8}\\ \text{Rel-MSA, G 7$\times$7, H 8}\end{array}\right]\times 6 |  |  |  |  |
| 8×8\times |  |  |  |  |  |  |  |  |  |
| (28×2828\times 28) |  |  |  |  |  |  |  |  |  |
| S3 | 16×16\times (14×1414\times 14) | 16×16\times | (14×1414\times 14) | [MBConv, 384, E 4, R 4Rel-MSA, P 7×7, H 12Rel-MSA, G 7×7, H 12]×14\left[\begin{array}[]{c}\text{MBConv, 384, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 12}\\ \text{Rel-MSA, G 7$\times$7, H 12}\end{array}\right]\!\times\!14 | [MBConv, 512, E 4, R 4Rel-MSA, P 7×7, H 16Rel-MSA, G 7×7, H 16]×14\left[\begin{array}[]{c}\text{MBConv, 512, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 16}\\ \text{Rel-MSA, G 7$\times$7, H 16}\end{array}\right]\times 14 |  |  |  |  |
| 16×16\times |  |  |  |  |  |  |  |  |  |
| (14×1414\times 14) |  |  |  |  |  |  |  |  |  |
| S4 | 32×32\times (7×77\times 7) | 32×32\times | (7×77\times 7) | [MBConv, 768, E 4, R 4Rel-MSA, P 7×7, H 24Rel-MSA, G 7×7, H 24]×2\left[\begin{array}[]{c}\text{MBConv, 768, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 24}\\ \text{Rel-MSA, G 7$\times$7, H 24}\end{array}\right]\times 2 | [MBConv, 1024, E 4, R 4Rel-MSA, P 7×7, H 32Rel-MSA, G 7×7, H 32]×2\left[\begin{array}[]{c}\text{MBConv, 1024, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 32}\\ \text{Rel-MSA, G 7$\times$7, H 32}\end{array}\right]\times 2 |  |  |  |  |
| 32×32\times |  |  |  |  |  |  |  |  |  |
| (7×77\times 7) |  |  |  |  |  |  |  |  |  |

| 2×2\times |
|---|
| (×112112\!\times\!112) |

| 3×\times3, 64, stride 2 |
|---|
| 3×\times3, 64, stride 1 |

| 3×\times3, 64, stride 2 |
|---|
| 3×\times3, 64, stride 1 |

| 4×4\times |
|---|
| (56×5656\times 56) |

| 8×8\times |
|---|
| (28×2828\times 28) |

| 16×16\times |
|---|
| (14×1414\times 14) |

| 32×32\times |
|---|
| (7×77\times 7) |

| 2×2\times |
|---|
| (×112112\!\times\!112) |

| 3×\times3, 64, stride 2 |
|---|
| 3×\times3, 64, stride 1 |

| 3×\times3, 128, stride 2 |
|---|
| 3×\times3, 128, stride 1 |

| 4×4\times |
|---|
| (56×5656\times 56) |

| 8×8\times |
|---|
| (28×2828\times 28) |

| 16×16\times |
|---|
| (14×1414\times 14) |

| 32×32\times |
|---|
| (7×77\times 7) |

### 0.A.2 Detection and Segmentation Models

We follow the settings of the cascaded Faster-RCNN [65] and Mask-RCNN [28], but replace the feature extraction backbone with our MaxViT backbone. We also applied FPN [52] in the feature map generation, where the S2, S3, S4 (multi-scale features of targeted resolution 1/81/8, 1/161/16, 1/321/32 in MaxViT, respectively) are used. Then the generated feature maps are fed into the detection head. For fair comparison, we follow the original implementation without adopting any system-level strategies to further boost the final performance, such as the HTC framework [7], instaboost [25], etc. used in Swin [56]. We show the results of MaxViT-T/S/B on these two tasks to compare it against recent strong models at similar model complexity.

### 0.A.3 Image Aesthetics Model

This task requires incorporating both local and global information of an image to accurately predict human perceptual preference. To this end, the model needs to have the capacity to learn pixel-level quality aspects such as sharpness, noisiness and contrast as well as semantic-level aspects such as composition and depth-of-field. We follow [77] and use the normalized Earth Mover’s Distance as our training loss. Given the ground truth and predicted probability mass functions p and p^\widehat{\textbf{p}} representing the histogram of scores, the normalized Earth Mover’s Distance can be expressed as:

|  | EMD​(p,p^)=(1N​∑k=1N|CDFp​(k)−CDFp^​(k)|r)1/r\mbox{EMD}(\textbf{p},\widehat{\textbf{p}})=\left(\frac{1}{N}\sum_{k=1}^{N}|\mbox{CDF}_{\textbf{p}}(k)-\mbox{CDF}_{\widehat{\textbf{p}}}(k)|^{r}\right)^{1/r} |  | (8) |
|---|---|---|---|

where CDFp​(k)\mbox{CDF}_{\textbf{p}}(k) is the cumulative distribution function as ∑i=1kpi\sum_{i=1}^{k}\textbf{p}_{i}, and N=10N=10 represents the number score bins. In our experiments we set r=2r=2. We remove the classification head used in MaxViT, and instead append a fully-connected layer with 10 neurons followed by 𝗌𝗈𝖿𝗍𝗆𝖺𝗑\mathsf{softmax}.

### 0.A.4 GAN Model

The above image recognition tasks can validate the power of our proposed MaxViT block used in downsampling (contracting) models. For this GAN experiment, we would like to demonstrate its effectiveness in upsampling (expanding) architectures. The MaxViT-GAN model for image generation is illustrated in Figure 7. For unconditional image generation, MaxViT-GAN first takes a latent code z∼𝒩⁡(𝟎,𝐈)z\sim\mathcal{N}(\mathbf{0},\mathbf{I}) as input, then progressively generates an image of target resolution through a hierarchically upsampling structure. We start by linearly projecting the input to a feature with spatial dimension 8×88\times 8. During the generation, the feature will go through five stages consisting of identical GAN blocks with gradually increased spatial resolution, similar to the design of our main model. Similar to [103], we apply a cross-attention layer before the MaxViT block as a memory-efficient form of self-modulation in every stage, which has been shown to stabilize GAN training and also improve mode coverage [11, 103]. We use pixel shuffle [72] for upsampling in the end of each stage.

## Appendix 0.B Experimental Settings

### 0.B.1 ImageNet Classification

We provide ImageNet-1K experimental settings of MaxViT models for both pre-training and fine-tuning in Table 12. All the MaxViT variants used similar hyperparameters except that we mainly customize the stochastic depth rate to regularize each model separately.

| Hyperparameter | ImageNet-1K | ImageNet-21K | JFT-300M |  |  |  |
|---|---|---|---|---|---|---|
| Pre-training | Fine-tuning | Pre-training | Fine-tuning | Pre-training | Fine-tuning |  |
| (MaxViT-T/S/B/L) | (MaxViT-B/L/XL) | (MaxViT-B/L/XL) |  |  |  |  |
| Stochastic depth | 0.2/0.3/0.4/0.60.2/0.3/0.4/0.6 | 0.3/0.4/0.60.3/0.4/0.6 | 0.4/0.5/0.90.4/0.5/0.9 | 0.0/0.0/0.00.0/0.0/0.0 | 0.1/0.2/0.20.1/0.2/0.2 |  |
| Center crop | True | False | True | False | True | False |
| RandAugment | 2, 15 | 2, 15 | 2, 5 | 2, 15 | 2, 5 | 2, 15 |
| Mixup alpha | 0.8 | 0.8 | None | None | None | None |
| Loss type | Softmax | Softmax | Sigmoid | Softmax | Sigmoid | Softmax |
| Label smoothing | 0.1 | 0.1 | 0.0001 | 0.1 | 0 | 0.1 |
| Train epochs | 300 | 30 | 90 | 30 | 14 | 30 |
| Train batch size | 4096 | 512 | 4096 | 512 | 4096 | 512 |
| Optimizer type | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW |
| Peak learning rate | 3e-3 | 5e-5 | 1e-3 | 5e-5 | 1e-3 | 5e-5 |
| Min learning rate | 1e-5 | 5e-5 | 1e-5 | 5e-5 | 1e-5 | 5e-5 |
| Warm-up | 10K steps | None | 5 epochs | None | 20K steps | None |
| LR decay schedule | Cosine | None | Linear | None | Linear | None |
| Weight decay rate | 0.05 | 1e-8 | 0.01 | 1e-8 | 0.01 | 1e-8 |
| Gradient clip | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| EMA decay rate | None | 0.9999 | None | 0.9999 | None | 0.9999 |

### 0.B.2 Coco Detection and Segmentation

We evaluated MaxViT on the COCO2017 [53] object bounding box detection and instance segmentation tasks. The dataset contains 118K training and 5K validation samples. All the MaxViT backbones used are pretrained on ImageNet-1k at resolution 224×224224\times 224. These pretrained checkpoints are then used as the warm-up weights for fine-tuning the detection and segmentation tasks. For both tasks, the input images are resized to 896×896896\times 896. The training is conducted with a batch size of 256, using the AdamW [59] optimizer with learning rate of 1e-3, 3e-3, 3e-3, and stochastic depth of 0.8,0.3,0.30.8,0.3,0.3 for MaxViT-T/S/B, respectively.

### 0.B.3 Image Aesthetics Assessment

We trained and evaluated the MaxViT model on the AVA benchmark [61]. This dataset consists of 255K images rated by armature photographers through photography contests. Each image is rated by an average of 200 human raters, assigning a score from 1 to 10 to images. The higher the score, the better the visual aesthetic quality of the image. Each image in the dataset has a histogram of scores associated with it, which we use as the ground truth label. Similar to [77, 43], we split the dataset into train and test sets, such that 20% of the data is used for testing. We train MaxViT for three different input resolutions: 224×224224\times 224, 384×384384\times 384 and 512×512512\times 512. We initialized the model with ImageNet-1K 224×\times224 pre-trained weights. The weight and bias momentums are set to 0.9, and a dropout rate of 0.75 is applied on the last layer of the baseline network. We use an initial learning rate of 1e-3, exponentially decayed with decay factor 0.9 every 10 epochs. We set the stochastic depth rate to 0.5.

### 0.B.4 Image Generation

We use a ResNet-based discriminator following [42]. To train the model, we also used the standard non-saturating logistic GAN loss with R​1R1 gradient penalty [60] applied to the discriminator with the gradient penalty weight set to 10. We employ the Adam [45] optimizer with a learning rate of 1e-4 for both generator and discriminator. The model is trained on TPU for one million steps with batch size 256. Notably, we do not employ extra GAN training tricks such as pixel norm, noise injection, progressive growing, etc. on which recent state-of-the-art models are heavily relied to attain good results [41, 42]. The overall objectives of the GAN training are defined as:

|  | ℒG\displaystyle\mathcal{L}_{G} | =−𝔼z∼Pz[log(D(G(z))],\displaystyle=-\mathbb{E}_{z\sim P_{z}}[\log(D(G(z))], |  | (9) |
|---|---|---|---|---|
|  | ℒD\displaystyle\mathcal{L}_{D} | =−𝔼x∼Px​[log⁡(D⁡(x))]−𝔼z∼Pz​[log⁡(1−D⁡(G⁡(z)))]+γ​𝔼x∼Px​[‖∇xD​(x)‖22],\displaystyle=-\mathbb{E}_{x\sim P_{x}}[\log(D(x))]-\mathbb{E}_{z\sim P_{z}}[\log(1-D(G(z)))]+\gamma\mathbb{E}_{x\sim P_{x}}[\|\nabla_{x}D(x)\|_{2}^{2}], |  | (10) |

where γ\gamma denotes the R1R_{1} gradient penalty weight.

## Appendix 0.C Complete Experimental Results

We provide complete experiment comparisons for ImageNet-1K, Image-21K, and JFT datasets in Table 13 and Table 14, respectively. We also provide more visual results for unconditional image generation on ImageNet-1K in Figure 8.

|  | Model | Eval size | Params | FLOPs | throughput (img/s) | ImageNet top-1 acc. |
|---|---|---|---|---|---|---|
| ConvNets | ∙\bulletEffNet-B3 [79] | 300 | 12M | 1.8G | 732.1 | 81.6 |
| ∙\bulletEffNet-B4 [79] | 380 | 19M | 4.2G | 349.4 | 82.9 |  |
| ∙\bulletEffNet-B5 [79] | 456 | 30M | 9.9G | 169.1 | 83.6 |  |
| ∙\bulletEffNet-B6 [79] | 528 | 43M | 19.0G | 96.9 | 84.0 |  |
| ∙\bulletEffNet-B7 [79] | 600 | 66M | 37.0G | 55.1 | 84.3 |  |
| ∙\bulletRegNetY-8GF [62] | 224 | 39M | 8.0G | 591.6 | 81.7 |  |
| ∙\bulletRegNetY-16GF [62] | 224 | 84M | 16.0G | 334.7 | 82.9 |  |
| ∙\bulletNFNet-F0 [5] | 256 | 72M | 12.4G | 533,.3 | 83.6 |  |
| ∙\bulletNFNet-F1 [5] | 320 | 132M | 35.5G | 228.5 | 84.7 |  |
| ∙\bulletNFNet-F2 [5] | 352 | 194M | 62.6G | 129.0 | 85.1 |  |
| ∙\bulletNFNet-F3 [5] | 416 | 255M | 114.7G | 78.8 | 85.7 |  |
| ∙\bulletNFNet-F4 [5] | 512 | 316M | 215.2G | 51.7 | 85.9 |  |
| ∙\bulletNFNet-F5 [5] | 544 | 377M | 289.8G | - | 86.0 |  |
| ∙\bulletEffNetV2-S [80] | 384 | 24M | 8.8G | 666.6 | 83.9 |  |
| ∙\bulletEffNetV2-M [80] | 380 | 55M | 24.0G | 280.7 | 85.1 |  |
| ∙\bulletEffNetV2-L [80] | 480 | 121M | 53.0G | 163.2 | 85.7 |  |
| ∙\bulletConvNeXt-T [57] | 224 | 29M | 4.5G | 774.7 | 82.1 |  |
| ∙\bulletConvNeXt-S [57] | 224 | 50M | 8.7G | 447.1 | 83.1 |  |
| ∙\bulletConvNeXt-B [57] | 224 | 89M | 15.4G | 292.1 | 83.8 |  |
| ∙\bulletConvNeXt-L [57] | 384 | 198M | 101.0G | 50.4 | 85.5 |  |
| ViTs | ∘\circViT-B/32 [22] | 384 | 86M | 55.4G | 85.9 | 77.9 |
| ∘\circViT-B/16 [22] | 384 | 307M | 190.7G | 27.3 | 76.5 |  |
| ∘\circDeiT-S [81] | 224 | 22M | 4.6G | 940.4 | 79.8 |  |
| ∘\circDeiT-B [81] | 224 | 86M | 17.5G | 292.3 | 81.8 |  |
| ∘\circDeiT-B [81] | 384 | 86M | 55.4G | 85.9 | 83.1 |  |
| ∘\circCaiT-S36 [82] | 224 | 68M | 13.9G | - | 83.3 |  |
| ∘\circCaiT-M24 [82] | 224 | 186M | 36.0G | - | 83.4 |  |
| ∘\circCaiT-M24 [82] | 384 | 186M | 116.1G | - | 84.5 |  |
| ∘\circDeepViT-S [105] | 224 | 27M | 6.2G | - | 82.3 |  |
| ∘\circDeepViT-L [105] | 224 | 55M | 12.5G | - | 83.1 |  |
| ∘\circT2T-ViT-14 [101] | 224 | 22M | 6.1G | - | 81.7 |  |
| ∘\circT2T-ViT-19 [101] | 224 | 39M | 9.8G | - | 82.2 |  |
| ∘\circT2T-ViT-24 [101] | 224 | 64M | 15.0G | - | 82.6 |  |
| ∘\circSwin-T [56] | 224 | 29M | 4.5G | 755.2 | 81.3 |  |
| ∘\circSwin-S [56] | 224 | 50M | 8.7G | 436.9 | 83.0 |  |
| ∘\circSwin-B [56] | 384 | 88M | 47.0G | 84.7 | 84.5 |  |
| ∘\circCSwin-B [21] | 224 | 78M | 15.0G | 250 | 84.2 |  |
| ∘\circCSwin-B [21] | 384 | 78M | 47.0G | - | 85.4 |  |
| ∘\circFocal-S [99] | 224 | 51M | 9.1G | - | 83.5 |  |
| ∘\circFocal-B [99] | 224 | 90M | 16.0G | - | 83.8 |  |
| Hybrid | ⋄\diamondCvT-13 [93] | 224 | 20M | 4.5G | - | 81.6 |
| ⋄\diamondCvT-21 [93] | 224 | 32M | 7.1G | - | 82.5 |  |
| ⋄\diamondCvT-21 [93] | 384 | 32M | 24.9G | - | 83.3 |  |
| ⋄\diamondCoAtNet-0 [19] | 224 | 25M | 4.2G | 534.5 | 81.6 |  |
| ⋄\diamondCoAtNet-1 [19] | 224 | 42M | 8.4G | 336.5 | 83.3 |  |
| ⋄\diamondCoAtNet-2 [19] | 224 | 75M | 15.7G | 247.6 | 84.1 |  |
| ⋄\diamondCoAtNet-3 [19] | 384 | 168M | 107.4G | 48.5 | 85.8 |  |
| ⋄\diamondCoAtNet-3 [19] | 512 | 168M | 203.1G | 22.4 | 86.0 |  |
| ⋄\diamondMaxViT-T | 224 | 31M | 5.6G | 349.6 | 83.62 |  |
| ⋄\diamondMaxViT-S | 224 | 69M | 11.7G | 242.5 | 84.45 |  |
| ⋄\diamondMaxViT-B | 224 | 120M | 23.4G | 133.6 | 84.95 |  |
| ⋄\diamondMaxViT-L | 224 | 212M | 43.9G | 99.4 | 85.17 |  |
| ⋄\diamondMaxViT-T | 384 | 31M | 17.7G | 121.9 | 85.24 |  |
| ⋄\diamondMaxViT-S | 384 | 69M | 36.1G | 82.7 | 85.74 |  |
| ⋄\diamondMaxViT-B | 384 | 120M | 74.2G | 45.8 | 86.34 |  |
| ⋄\diamondMaxViT-L | 384 | 212M | 133.1G | 34.3 | 84.40 |  |
| ⋄\diamondMaxViT-T | 512 | 31M | 33.7G | 63.8 | 85.72 |  |
|  | ⋄\diamondMaxViT-S | 512 | 69M | 67.6G | 43.3 | 86.19 |
|  | ⋄\diamondMaxViT-B | 512 | 120M | 138.5G | 24.0 | 86.66 |
|  | ⋄\diamondMaxViT-L | 512 | 212M | 245.4G | 17.8 | 86.70 |

|  | Model | Eval size | Params | FLOPs | IN-1K top-1 acc. |  |
|---|---|---|---|---|---|---|
|  | 21K→\rightarrow1K | JFT→\rightarrow1K |  |  |  |  |
| ConvNets | ∙\bulletBiT-R-101x3 [46] | 384 | 388M | 204.6G | 84.4 |  |
| ∙\bulletBiT-R-152x4 [46] | 480 | 937M | 840.5G | 85.4 |  |  |
| ∙\bulletEffNetV2-S [80] | 384 | 24M | 8.8G | 85.0 |  |  |
| ∙\bulletEffNetV2-M [80] | 480 | 55M | 24.0G | 86.1 |  |  |
| ∙\bulletEffNetV2-L [80] | 480 | 121M | 53.0G | 86.8 |  |  |
| ∙\bulletEffNetV2-XL [80] | 512 | 208M | 94.0G | 87.3 |  |  |
| ∙\bulletNFNet-F4+ [5] | 512 | 527M | 367G | - | 89.20 |  |
| ∙\bulletConvNeXt-B [57] | 384 | 89M | 45.1G | 86.8 |  |  |
| ∙\bulletConvNeXt-L [57] | 384 | 198M | 101.0G | 87.5 |  |  |
| ∙\bulletConvNeXt-XL [57] | 384 | 350M | 179.0G | 87.8 |  |  |
| ViTs | ∘\circViT-B/16 [22] | 384 | 87M | 55.5G | 84.0 |  |
| ∘\circViT-L/16 [22] | 384 | 305M | 191.1G | 85.2 |  |  |
| ∘\circViT-L/16 [22] | 512 | 305M | 364G | - | 87.76 |  |
| ∘\circViT-H/14 [22] | 518 | 632M | 1021G | - | 88.55 |  |
| ∘\circHaloNet-H4 [84] | 384 | 85M | - | 85.6 |  |  |
| ∘\circHaloNet-H4 [84] | 512 | 85M | - | 85.8 |  |  |
| ∘\circSwin-B [56] | 384 | 88M | 47.0G | 86.4 |  |  |
| ∘\circSwin-L [56] | 384 | 197M | 103.9G | 87.3 |  |  |
| ∘\circSwinV2-B [56] | 384 | 88M | - | 87.1 |  |  |
| ∘\circSwinV2-L [56] | 384 | 197M | - | 87.7 |  |  |
| ∘\circCSwin-B [21] | 384 | 78M | 47.0G | 87.0 |  |  |
| ∘\circCSwin-L [21] | 384 | 173M | 96.8G | 87.5 |  |  |
| Hybrid | ⋄\diamondCvT-13 [93] | 384 | 20M | 16.0G | 83.3 |  |
| ⋄\diamondCvT-21 [93] | 384 | 32M | 25.0G | 84.9 |  |  |
| ⋄\diamondCvT-W24 [93] | 384 | 277M | 193.2G | 87.7 |  |  |
| ⋄\diamondResNet+ViT-L/16 [22] | 384 | 330M | - | - | 87.12 |  |
| ⋄\diamondCoAtNet-2 [19] | 384 | 75M | 49.8G | 87.1 |  |  |
| ⋄\diamondCoAtNet-3 [19] | 384 | 168M | 107.4G | 87.6 |  |  |
| ⋄\diamondCoAtNet-4 [19] | 384 | 275M | 189.5G | 87.9 |  |  |
| ⋄\diamondCoAtNet-2 [19] | 512 | 75M | 96.7G | 87.3 |  |  |
| ⋄\diamondCoAtNet-3 [19] | 512 | 168M | 203.1G | 87.9 | 88.81 |  |
| ⋄\diamondCoAtNet-4 [19] | 512 | 275M | 360.9G | 88.1 | 89.11 |  |
| ⋄\diamondCoAtNet-5 [19] | 512 | 688M | 812G | - | 89.77 |  |
| ⋄\diamondMaxViT-B | 384 | 119M | 74.2G | 88.24 | 88.69 |  |
| ⋄\diamondMaxViT-L | 384 | 212M | 128.7G | 88.32 | 89.12 |  |
| ⋄\diamondMaxViT-XL | 384 | 475M | 293.7G | 88.51 | 89.36 |  |
| ⋄\diamondMaxViT-B | 512 | 119M | 138.3G | 88.38 | 88.82 |  |
| ⋄\diamondMaxViT-L | 512 | 212M | 245.2G | 88.46 | 89.41 |  |
| ⋄\diamondMaxViT-XL | 512 | 475M | 535.2G | 88.70 | 89.53 |  |
