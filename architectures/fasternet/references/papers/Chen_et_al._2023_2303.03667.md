# Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks

To design fast neural networks, many works have been focusing on reducing the number of floating-point operations (FLOPs). We observe that such reduction in FLOPs, however, does not necessarily lead to a similar level of reduction in latency. This mainly stems from inefficiently low floating-point operations per second (FLOPS). To achieve faster networks, we revisit popular operators and demonstrate that such low FLOPS is mainly due to frequent memory access of the operators, especially the depthwise convolution. We hence propose a novel partial convolution (PConv) that extracts spatial features more efficiently, by cutting down redundant computation and memory access simultaneously. Building upon our PConv, we further propose FasterNet, a new family of neural networks, which attains substantially higher running speed than others on a wide range of devices, without compromising on accuracy for various vision tasks. For example, on ImageNet-1k, our tiny FasterNet-T0 is 2.8×2.8\times, 3.3×3.3\times, and 2.4×2.4\times faster than MobileViT-XXS on GPU, CPU, and ARM processors, respectively, while being 2.9% more accurate. Our large FasterNet-L achieves impressive 83.5% top-1 accuracy, on par with the emerging Swin-B, while having 36% higher inference throughput on GPU, as well as saving 37% compute time on CPU. Code is available at https://github.com/JierunChen/FasterNet.

## 1 Introduction

Neural networks have undergone rapid development in various computer vision tasks such as image classification, detection and segmentation. While their impressive performance has powered many applications, a roaring trend is to pursue fast neural networks with low latency and high throughput for great user experiences, instant responses, safety reasons, etc.

How to be fast? Instead of asking for more costly computing devices, researchers and practitioners prefer to design cost-effective fast neural networks with reduced computational complexity, mainly measured in the number of floating-point operations (FLOPs)11 1 We follow a widely adopted definition of FLOPs, as the number of multiply-adds [84, 42].. MobileNets [25, 54, 24], ShuffleNets [84, 46] and GhostNet [17], among others, leverage the depthwise convolution (DWConv) [55] and/or group convolution (GConv) [31] to extract spatial features. However, in the effort to reduce FLOPs, the operators often suffer from the side effect of increased memory access. MicroNet [33] further decomposes and sparsifies the network to push its FLOPs to an extremely low level. Despite its improvement in FLOPs, this approach experiences inefficient fragmented computation. Besides, the above networks are often accompanied by additional data manipulations, such as concatenation, shuffling, and pooling, whose running time tends to be significant for tiny models.

Apart from the above pure convolutional neural networks (CNNs), there is an emerging interest in making vision transformers (ViTs) [12] and multilayer perceptrons (MLPs) architectures [64] smaller and faster. For example, MobileViTs [48, 49, 70] and MobileFormer [6] reduce the computational complexity by combining DWConv with a modified attention mechanism. However, they still suffer from the aforementioned issue with DWConv and also need dedicated hardware support for the modified attention mechanism. The use of advanced yet time-consuming normalization and activation layers may also limit their speed on devices.

All these issues together lead to the following question: Are these “fast” neural networks really fast? To answer this, we examine the relationship between latency and FLOPs, which is captured by

|  | L​a​t​e​n​c​y=F​L​O​P​sF​L​O​P​S,Latency=\frac{FLOPs}{FLOPS}, |  | (1) |
|---|---|---|---|

where FLOPS is short for floating-point operations per second, as a measure of the effective computational speed. While there are many attempts to reduce FLOPs, they seldom consider optimizing FLOPS at the same time to achieve truly low latency. To better understand the situation, we compare the FLOPS of typical neural networks on an Intel CPU. The results in Fig. 2 show that many existing neural networks suffer from low FLOPS, and their FLOPS is generally lower than the popular ResNet50. With such low FLOPS, these “fast” neural networks are actually not fast enough. Their reduction in FLOPs cannot be translated into the exact amount of reduction in latency. In some cases, there is no improvement, and it even leads to worse latency. For example, CycleMLP-B1 [5] has half of FLOPs of ResNet50 [20] but runs more slowly (i.e., CycleMLP-B1 vs. ResNet50: 116.1ms vs. 73.0ms). Note that this discrepancy between FLOPs and latency has also been noticed in previous works [46, 48] but remains unresolved partially because they employ the DWConv/GConv and various data manipulations with low FLOPS. It is deemed there are no better alternatives available.

This paper aims to eliminate the discrepancy by developing a simple yet fast and effective operator that maintains high FLOPS with reduced FLOPs. Specifically, we reexamine existing operators, particularly DWConv, in terms of the computational speed – FLOPS. We uncover that the main reason causing the low FLOPS issue is frequent memory access. We then propose a novel partial convolution (PConv) as a competitive alternative that reduces the computational redundancy as well as the number of memory access. Fig. 1 illustrates the design of our PConv. It takes advantage of redundancy within the feature maps and systematically applies a regular convolution (Conv) on only a part of the input channels while leaving the remaining ones untouched. By nature, PConv has lower FLOPs than the regular Conv while having higher FLOPS than the DWConv/GConv. In other words, PConv better exploits the on-device computational capacity. PConv is also effective in extracting spatial features as empirically validated later in the paper.

We further introduce FasterNet, which is primarily built upon our PConv, as a new family of networks that run highly fast on various devices. In particular, our FasterNet achieves state-of-the-art performance for classification, detection, and segmentation tasks while having much lower latency and higher throughput. For example, our tiny FasterNet-T0 is 2.8×2.8\times, 3.3×3.3\times, and 2.4×2.4\times faster than MobileViT-XXS [48] on GPU, CPU, and ARM processors, respectively, while being 2.9% more accurate on ImageNet-1k. Our large FasterNet-L achieves 83.5% top-1 accuracy, on par with the emerging Swin-B [41], while offering 36% higher throughput on GPU and saving 37% compute time on CPU. To summarize, our contributions are as follows:

- • We point out the importance of achieving higher FLOPS beyond simply reducing FLOPs for faster neural networks.
We point out the importance of achieving higher FLOPS beyond simply reducing FLOPs for faster neural networks.

- • We introduce a simple yet fast and effective operator called PConv, which has a high potential to replace the existing go-to choice, DWConv.
We introduce a simple yet fast and effective operator called PConv, which has a high potential to replace the existing go-to choice, DWConv.

- • We introduce FasterNet which runs favorably and universally fast on a variety of devices such as GPU, CPU, and ARM processors.
We introduce FasterNet which runs favorably and universally fast on a variety of devices such as GPU, CPU, and ARM processors.

- • We conduct extensive experiments on various tasks and validate the high speed and effectiveness of our PConv and FasterNet.
We conduct extensive experiments on various tasks and validate the high speed and effectiveness of our PConv and FasterNet.

## 2 Related Work

We briefly review prior works on fast and efficient neural networks and differentiate this work from them.

CNN. CNNs are the mainstream architecture in the computer vision field, especially when it comes to deployment in practice, where being fast is as important as being accurate. Though there have been numerous studies [55, 56, 7, 8, 83, 33, 21, 86] to achieve higher efficiency, the rationale behind them is more or less to perform a low-rank approximation. Specifically, the group convolution [31] and the depthwise separable convolution [55] (consisting of depthwise and pointwise convolutions) are probably the most popular ones. They have been widely adopted in mobile/edge-oriented networks, such as MobileNets [25, 54, 24], ShuffleNets [84, 46], GhostNet [17], EfficientNets [61, 62], TinyNet [18], Xception [8], CondenseNet [27, 78], TVConv [4], MnasNet[60], and FBNet [74]. While they exploit the redundancy in filters to reduce the number of parameters and FLOPs, they suffer from increased memory access when increasing the network width to compensate for the accuracy drop. By contrast, we consider the redundancy in feature maps and propose a partial convolution to reduce FLOPs and memory access simultaneously.

ViT, MLP, and variants. There is a growing interest in studying ViT ever since Dosovitskiy et al. [12] expanded the application scope of transformers [69] from machine translation [69] or forecasting [73] to the computer vision field. Many follow-up works have attempted to improve ViT in terms of training setting [65, 66, 58] and model design [41, 40, 72, 15, 85]. One notable trend is to pursue a better accuracy-latency trade-off by reducing the complexity of the attention operator [1, 68, 29, 45, 63], incorporating convolution into ViTs [10, 6, 57], or doing both [3, 34, 52, 49]. Besides, other studies [64, 35, 5] propose to replace the attention with simple MLP-based operators. However, they often evolve to be CNN-like [39]. In this paper, we focus on analyzing the convolution operations, particularly DWConv, due to the following reasons: First, the advantage of attention over convolution is unclear or debatable [71, 42]. Second, the attention-based mechanism generally runs slower than its convolutional counterparts and thus becomes less favorable for the current industry [48, 26]. Finally, DWConv is still a popular choice in many hybrid models, so it is worth a careful examination.

## 3 Design of PConv and FasterNet

In this section, we first revisit DWConv and analyze the issue with its frequent memory access. We then introduce PConv as a competitive alternative operator to resolve the issue. After that, we introduce FasterNet and explain its details, including design considerations.

### 3.1 Preliminary

DWConv is a popular variant of Conv and has been widely adopted as a key building block for many neural networks. For an input 𝐈∈ℝc×h×w\mathbf{I}\in\mathbb{R}^{c\times h\times w}, DWConv applies cc filters 𝐖∈ℝk×k\mathbf{W}\in\mathbb{R}^{k\times k} to compute the output 𝐎∈ℝc×h×w\mathbf{O}\in\mathbb{R}^{c\times h\times w}. As shown in Fig. 1(b), each filter slides spatially on one input channel and contributes to one output channel. This depthwise computation makes DWConv have as low FLOPs as h×w×k2×ch\times w\times k^{2}\times c compared to a regular Conv with h×w×k2×c2h\times w\times k^{2}\times c^{2}. While effective in reducing FLOPs, a DWConv, which is typically followed by a pointwise convolution, or PWConv, cannot be simply used to replace a regular Conv as it would incur a severe accuracy drop. Thus, in practice the channel number cc (or the network width) of DWConv is increased to c′​(c′>c)c^{\prime}\left(c^{\prime}>c\right) to compensate the accuracy drop, e.g., the width is expanded by six times for the DWConv in the inverted residual blocks [54]. This, however, results in much higher memory access that can cause non-negligible delay and slow down the overall computation, especially for I/O-bound devices. In particular, the number of memory access now escalates to

|  | h×w×2​c′+k2×c′≈h×w×2​c′,h\times w\times 2c^{\prime}+k^{2}\times c^{\prime}\approx h\times w\times 2c^{\prime}, |  | (2) |
|---|---|---|---|

which is higher than that of a regular Conv, i.e.,

|  | h×w×2​c+k2×c2≈h×w×2​c.h\times w\times 2c+k^{2}\times c^{2}\approx h\times w\times 2c. |  | (3) |
|---|---|---|---|

Note that the h×w×2​c′h\times w\times 2c^{\prime} memory access is spent on the I/O operation, which is deemed to be already the minimum cost and hard to optimize further.

### 3.2 Partial convolution as a basic operator

We below demonstrate that the cost can be further optimized by leveraging the feature maps’ redundancy. As visualized in Fig. 3, the feature maps share high similarities among different channels. This redundancy has also been covered in many other works [17, 82], but few of them make full use of it in a simple yet effective way.

Specifically, we propose a simple PConv to reduce computational redundancy and memory access simultaneously. The bottom-left corner in Fig. 4 illustrates how our PConv works. It simply applies a regular Conv on only a part of the input channels for spatial feature extraction and leaves the remaining channels untouched. For contiguous or regular memory access, we consider the first or last consecutive cpc_{p} channels as the representatives of the whole feature maps for computation. Without loss of generality, we consider the input and output feature maps to have the same number of channels. Therefore, the FLOPs of a PConv are only

|  | h×w×k2×cp2.h\times w\times k^{2}\times c_{p}^{2}. |  | (4) |
|---|---|---|---|

With a typical partial ratio r=cpc=14r\!=\!\frac{c_{p}}{c}\!=\!\frac{1}{4}, the FLOPs of a PConv is only 116\frac{1}{16} of a regular Conv. Besides, PConv has a smaller amount of memory access, i.e.,

|  | h×w×2​cp+k2×cp2≈h×w×2​cp,h\times w\times 2c_{p}+k^{2}\times c_{p}^{2}\approx h\times w\times 2c_{p}, |  | (5) |
|---|---|---|---|

which is only 14\frac{1}{4} of a regular Conv for r=14r=\frac{1}{4}.

Since there are only cpc_{p} channels utilized for spatial feature extraction, one may ask if we can simply remove the remaining (c−cp)(c-c_{p}) channels? If so, PConv would degrade to a regular Conv with fewer channels, which deviates from our objective to reduce redundancy. Note that we keep the remaining channels untouched instead of removing them from the feature maps. It is because they are useful for a subsequent PWConv layer, which allows the feature information to flow through all channels.

### 3.3 PConv followed by PWConv

To fully and efficiently leverage the information from all channels, we further append a pointwise convolution (PWConv) to our PConv. Their effective receptive field together on the input feature maps looks like a T-shaped Conv, which focuses more on the center position compared to a regular Conv uniformly processing a patch, as shown in Fig. 5. To justify this T-shaped receptive field, we first evaluate the importance of each position by calculating the position-wise Frobenius norm. We assume that a position tends to be more important if it has a larger Frobenius norm than other positions. For a regular Conv filter 𝐅∈ℝk2×c\mathbf{F}\in\mathbb{R}^{k^{2}\times c}, the Frobenius norm at position ii is calculated by ‖𝐅i‖=∑j=1c|fi​j|2,\left\|\mathbf{F}_{i}\right\|\!=\!\sqrt{\sum_{j=1}^{c}|f_{ij}|^{2}}, for i=1,2,3​…,k2i=1,2,3...,k^{2}. We consider a salient position to be the one with the maximum Frobenius norm. We then collectively examine each filter in a pre-trained ResNet18, find out their salient positions, and plot a histogram of the salient positions. Results in Fig. 6 show that the center position turns out to be the salient position most frequently among the filters. In other words, the center position weighs more than its surrounding neighbors. This is consistent with the T-shaped computation which concentrates on the center position.

While the T-shaped Conv can be directly used for efficient computation, we show that it is better to decompose the T-shaped Conv into a PConv and a PWConv because the decomposition exploits the inter-filter redundancy and further saves FLOPs. For the same input 𝐈∈ℝc×h×w\mathbf{I}\in\mathbb{R}^{c\times h\times w} and output 𝐎∈ℝc×h×w\mathbf{O}\in\mathbb{R}^{c\times h\times w}, a T-shaped Conv’s FLOPs can be calculated as

|  | h×w×(k2×cp×c+c×(c−cp)),h\times w\times\left(k^{2}\times c_{p}\times c+c\times\left(c-c_{p}\right)\right), |  | (6) |
|---|---|---|---|

which is higher than the FLOPs of a PConv and a PWConv, i.e.,

|  | h×w×(k2×cp2+c2),h\times w\times(k^{2}\times c_{p}^{2}+c^{2}), |  | (7) |
|---|---|---|---|

where (k2−1)​c>k2​cp,(k^{2}-1)c>k^{2}c_{p}, e.g. when cp=c4c_{p}=\frac{c}{4} and k=3.k=3. Besides, we can readily leverage the regular Conv for the two-step implementation.

### 3.4 FasterNet as a general backbone

Given our novel PConv and off-the-shelf PWConv as the primary building operators, we further propose FasterNet, a new family of neural networks that runs favorably fast and is highly effective for many vision tasks. We aim to keep the architecture as simple as possible, without bells and whistles, to make it hardware-friendly in general.

We present the overall architecture in Fig. 4. It has four hierarchical stages, each of which is preceded by an embedding layer (a regular Conv 4×44\times 4 with stride 4) or a merging layer (a regular Conv 2×22\times 2 with stride 2) for spatial downsampling and channel number expanding. Each stage has a stack of FasterNet blocks. We observe that the blocks in the last two stages consume less memory access and tend to have higher FLOPS, as empirically validated in Tab. 1. Thus, we put more FasterNet blocks and correspondingly assign more computations to the last two stages. Each FasterNet block has a PConv layer followed by two PWConv (or Conv 1×11\times 1) layers. Together, they appear as inverted residual blocks where the middle layer has an expanded number of channels, and a shortcut connection is placed to reuse the input features.

In addition to the above operators, the normalization and activation layers are also indispensable for high-performing neural networks. Many prior works [20, 54, 17], however, overuse such layers throughout the network, which may limit the feature diversity and thus hurt the performance. It can also slow down the overall computation. By contrast, we put them only after each middle PWConv to preserve the feature diversity and achieve lower latency. Besides, we use the batch normalization (BN) [30] instead of other alternative ones [2, 67, 75]. The benefit of BN is that it can be merged into its adjacent Conv layers for faster inference while being as effective as the others. As for the activation layers, we empirically choose GELU [22] for smaller FasterNet variants and ReLU [51] for bigger FasterNet variants, considering both running time and effectiveness. The last three layers, i.e. a global average pooling, a Conv 1×11\times 1, and a fully-connected layer, are used together for feature transformation and classification.

To serve a wide range of applications under different computational budgets, we provide tiny, small, medium, and large variants of FasterNet, referred to as FasterNet-T0/1/2, FasterNet-S, FasterNet-M, and FasterNet-L, respectively. They share a similar architecture but vary in depth and width. Detailed architecture specifications are provided in the appendix.

## 4 Experimental Results

We first examine the computational speed of our PConv and its effectiveness when combined with a PWConv. We then comprehensively evaluate the performance of our FasterNet for classification, detection, and segmentation tasks. Finally, we conduct a brief ablation study.

To benchmark the latency and throughput, we choose the following three typical processors, which cover a wide range of computational capacity: GPU (2080Ti), CPU (Intel i9-9900X, using a single thread), and ARM (Cortex-A72, using a single thread). We report their latency for inputs with a batch size of 1 and throughput for inputs with a batch size of 32. During inference, the BN layers are merged to their adjacent layers wherever applicable.

### 4.1 PConv is fast with high FLOPS

| Operator | Feature map size | FLOPs (M), ×\times10 layers | GPU | CPU | ARM |  |  |  |
|---|---|---|---|---|---|---|---|---|
| Throughput (fps) | FLOPS (G/s) | Latency (ms) | FLOPS (G/s) | Latency (ms) | FLOPS (G/s) |  |  |  |
| Conv 3×\times3 | 96×\times56×\times56 | 2601 | 3010 | 7824 | 35.67 | 72.90 | 779.57 | 3.33 |
| 192×\times28×\times28 | 2601 | 4893 | 12717 | 28.41 | 91.53 | 619.64 | 4.19 |  |
| 384×\times14×\times14 | 2601 | 4558 | 11854 | 31.85 | 81.66 | 595.09 | 4.37 |  |
| 768×\times7×\times7 | 2601 | 3159 | 8212 | 62.71 | 41.47 | 662.17 | 3.92 |  |
| Average | - | 10151 | - | 71.89 | - | 3.95 |  |  |
| GConv 3×\times3 (16 groups) | 96×\times56×\times56 | 162 | 2888 | 469 | 21.90 | 7.42 | 166.30 | 0.97 |
| 192×\times28×\times28 | 162 | 10811 | 1754 | 7.58 | 21.44 | 96.22 | 1.68 |  |
| 384×\times14×\times14 | 162 | 15534 | 2514 | 4.40 | 36.88 | 63.57 | 2.55 |  |
| 768×\times7×\times7 | 162 | 16000 | 2598 | 4.28 | 37.97 | 65.20 | 2.49 |  |
| Average | - | 1833 | - | 25.93 | - | 1.92 |  |  |
| DWConv 3×\times3 | 96×\times56×\times56 | 27.09 | 11940 | 323 | 3.59 | 7.52 | 108.70 | 0.24 |
| 192×\times28×\times28 | 13.54 | 23358 | 315 | 1.97 | 6.86 | 82.01 | 0.16 |  |
| 384×\times14×\times14 | 6.77 | 46377 | 313 | 1.06 | 6.35 | 94.89 | 0.07 |  |
| 768×\times7×\times7 | 3.38 | 88889 | 302 | 0.68 | 4.93 | 150.89 | 0.02 |  |
| Average | - | 313 | - | 6.42 | - | 0.12 |  |  |
| PConv 3×\times3 (ours, with r=14r=\frac{1}{4}) | 96×\times56×\times56 | 162 | 9215 | 1493 | 5.46 | 29.67 | 85.30 | 1.90 |
| 192×\times28×\times28 | 162 | 14360 | 2326 | 3.09 | 52.43 | 66.46 | 2.44 |  |
| 384×\times14×\times14 | 162 | 24408 | 3954 | 3.58 | 45.25 | 49.98 | 3.24 |  |
| 768×\times7×\times7 | 162 | 32866 | 5324 | 5.02 | 32.27 | 48.30 | 3.35 |  |
| Average | - | 3274 | - | 39.91 | - | 2.73 |  |  |

We below show that our PConv is fast and better exploits the on-device computational capacity. Specifically, we stack 10 layers of pure PConv and take feature maps of typical dimensions as inputs. We then measure FLOPs and latency/throughput on GPU, CPU, and ARM processors, which also allow us to further compute FLOPS. We repeat the same procedure for other convolutional variants and make comparisons.

Results in Tab. 1 show that PConv is overall an appealing choice for high FLOPS with reduced FLOPs. It has only 116\frac{1}{16} FLOPs of a regular Conv and achieves 10.5×10.5\times, 6.2×6.2\times, and 22.8×22.8\times higher FLOPS than the DWConv on GPU, CPU, and ARM, respectively. We are unsurprised to see that the regular Conv has the highest FLOPS as it has been constantly optimized for years. However, its total FLOPs and latency/throughput are unaffordable. GConv and DWConv, despite their significant reduction in FLOPs, suffer from a drastic decrease in FLOPS. In addition, they tend to increase the number of channels to compensate for the performance drop, which, however, increase their latency.

### 4.2 PConv is effective together with PWConv

We next show that a PConv followed by a PWConv is effective in approximating a regular Conv to transform the feature maps. To this end, we first build four datasets by feeding the ImageNet-1k val split images into a pre-trained ResNet50, and extract the feature maps before and after the first Conv 3×33\times 3 in each of the four stages. Each feature map dataset is further spilt into the train (70%), val (10%), and test (20%) subsets. We then build a simple network consisting of a PConv followed by a PWConv and train it on the feature map datasets with a mean squared error loss. For comparison, we also build and train networks for DWConv + PWConv and GConv + PWConv under the same setting.

Tab. 2 shows that PConv + PWConv achieve the lowest test loss, meaning that they better approximate a regular Conv in feature transformation. The results also suggest that it is sufficient and efficient to capture spatial features from only a part of the feature maps. PConv shows a great potential to be the new go-to choice in designing fast and effective neural networks.

| Stage | DWConv+PWConv | GConv+PWConv (16 groups) | GConv+PWConv | (16 groups) | PConv+PWConv r=14r=\frac{1}{4} | PConv+PWConv | r=14r=\frac{1}{4} |
|---|---|---|---|---|---|---|---|
| GConv+PWConv |  |  |  |  |  |  |  |
| (16 groups) |  |  |  |  |  |  |  |
| PConv+PWConv |  |  |  |  |  |  |  |
| r=14r=\frac{1}{4} |  |  |  |  |  |  |  |
| 1 | 0.0089 | 0.0065 | 0.0069 |  |  |  |  |
| 2 | 0.0158 | 0.0137 | 0.0136 |  |  |  |  |
| 3 | 0.0214 | 0.0202 | 0.0172 |  |  |  |  |
| 4 | 0.0130 | 0.0128 | 0.0115 |  |  |  |  |
| Average | 0.0148 | 0.0133 | 0.0123 |  |  |  |  |

| GConv+PWConv |
|---|
| (16 groups) |

| PConv+PWConv |
|---|
| r=14r=\frac{1}{4} |

### 4.3 FasterNet on ImageNet-1k classification

| Network | Params (M) | Params | (M) | FLOPs (G) | FLOPs | (G) | Throughput on GPU (fps) ↑\uparrow | Throughput | on GPU | (fps) ↑\uparrow | Latency on CPU (ms) ↓\downarrow | Latency | on CPU | (ms) ↓\downarrow | Latency on ARM (ms) ↓\downarrow | Latency | on ARM | (ms) ↓\downarrow | Acc. (%) | Acc. | (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Params |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (M) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| FLOPs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (G) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Throughput |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| on GPU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (fps) ↑\uparrow |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Latency |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| on CPU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (ms) ↓\downarrow |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Latency |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| on ARM |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (ms) ↓\downarrow |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Acc. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (%) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ShuffleNetV2 ×\times1.5 [46] | 3.5 | 0.30 | 4878 | 12.1 | 266 | 72.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MobileNetV2 [54] | 3.5 | 0.31 | 4198 | 12.2 | 442 | 72.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MobileViT-XXS [48] | 1.3 | 0.42 | 2393 | 30.8 | 348 | 69.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| EdgeNeXt-XXS [47] | 1.3 | 0.26 | 2765 | 15.7 | 239 | 71.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| FasterNet-T0 | 3.9 | 0.34 | 6807 | 9.2 | 143 | 71.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GhostNet ×\times1.3 [17] | 7.4 | 0.24 | 2988 | 17.9 | 481 | 75.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ShuffleNetV2 ×\times2 [46] | 7.4 | 0.59 | 3339 | 17.8 | 403 | 74.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MobileNetV2 ×\times1.4 [54] | 6.1 | 0.60 | 2711 | 22.6 | 650 | 74.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MobileViT-XS [48] | 2.3 | 1.05 | 1392 | 40.8 | 648 | 74.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| EdgeNeXt-XS [47] | 2.3 | 0.54 | 1738 | 24.4 | 434 | 75.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PVT-Tiny [72] | 13.2 | 1.94 | 1266 | 55.6 | 708 | 75.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| FasterNet-T1 | 7.6 | 0.85 | 3782 | 17.7 | 285 | 76.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| CycleMLP-B1 [5] | 15.2 | 2.10 | 865 | 116.1 | 892 | 79.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PoolFormer-S12 [79] | 11.9 | 1.82 | 1439 | 49.0 | 665 | 77.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MobileViT-S [48] | 5.6 | 2.03 | 1039 | 56.7 | 941 | 78.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| EdgeNeXt-S [47] | 5.6 | 1.26 | 1128 | 39.2 | 743 | 79.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ResNet50 [20, 42] | 25.6 | 4.11 | 959 | 73.0 | 1131 | 78.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| FasterNet-T2 | 15.0 | 1.91 | 1991 | 33.5 | 497 | 78.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| CycleMLP-B2 [5] | 26.8 | 3.90 | 528 | 186.3 | 1502 | 81.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PoolFormer-S24 [79] | 21.4 | 3.41 | 748 | 92.8 | 1261 | 80.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PoolFormer-S36 [79] | 30.9 | 5.00 | 507 | 138.0 | 1860 | 81.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ConvNeXt-T [42] | 28.6 | 4.47 | 657 | 86.3 | 1889 | 82.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Swin-T [41] | 28.3 | 4.51 | 609 | 122.2 | 1424 | 81.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PVT-Small [72] | 24.5 | 3.83 | 689 | 89.6 | 1345 | 79.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PVT-Medium [72] | 44.2 | 6.69 | 438 | 143.6 | 2142 | 81.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| FasterNet-S | 31.1 | 4.56 | 1029 | 71.2 | 1103 | 81.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PoolFormer-M36 [79] | 56.2 | 8.80 | 320 | 215.0 | 2979 | 82.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ConvNeXt-S [42] | 50.2 | 8.71 | 377 | 153.2 | 3484 | 83.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Swin-S [41] | 49.6 | 8.77 | 348 | 224.2 | 2613 | 83.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PVT-Large [72] | 61.4 | 9.85 | 306 | 203.4 | 3101 | 81.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| FasterNet-M | 53.5 | 8.74 | 500 | 129.5 | 2092 | 83.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PoolFormer-M48 [79] | 73.5 | 11.59 | 242 | 281.8 | OOM | 82.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ConvNeXt-B [42] | 88.6 | 15.38 | 253 | 257.1 | OOM | 83.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Swin-B [41] | 87.8 | 15.47 | 237 | 349.2 | OOM | 83.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| FasterNet-L | 93.5 | 15.52 | 323 | 219.5 | OOM | 83.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

| Params |
|---|
| (M) |

| FLOPs |
|---|
| (G) |

| Throughput |
|---|
| on GPU |
| (fps) ↑\uparrow |

| Latency |
|---|
| on CPU |
| (ms) ↓\downarrow |

| Latency |
|---|
| on ARM |
| (ms) ↓\downarrow |

| Acc. |
|---|
| (%) |

| Backbone | Params (M) | Params | (M) | FLOPs (G) | FLOPs | (G) | Latency on GPU (ms) | Latency on | GPU (ms) | A​PbAP^{b} | A​P50bAP^{b}_{50} | A​P75bAP^{b}_{75} | A​PmAP^{m} | A​P50mAP^{m}_{50} | A​P75mAP^{m}_{75} |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Params |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (M) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| FLOPs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (G) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Latency on |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GPU (ms) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ResNet50 [20] | 44.2 | 253 | 54.9 | 38.0 | 58.6 | 41.4 | 34.4 | 55.1 | 36.7 |  |  |  |  |  |  |
| PoolFormer-S24 [79] | 41.0 | 233 | 111.0 | 40.1 | 62.2 | 43.4 | 37.0 | 59.1 | 39.6 |  |  |  |  |  |  |
| PVT-Small [72] | 44.1 | 238 | 89.5 | 40.4 | 62.9 | 43.8 | 37.8 | 60.1 | 40.3 |  |  |  |  |  |  |
| FasterNet-S | 49.0 | 258 | 54.3 | 39.9 | 61.2 | 43.6 | 36.9 | 58.1 | 39.7 |  |  |  |  |  |  |
| ResNet101 [20] | 63.2 | 329 | 68.9 | 40.4 | 61.1 | 44.2 | 36.4 | 57.7 | 38.8 |  |  |  |  |  |  |
| ResNeXt101-32×\times4d [77] | 62.8 | 333 | 80.5 | 41.9 | 62.5 | 45.9 | 37.5 | 59.4 | 40.2 |  |  |  |  |  |  |
| PoolFormer-S36 [79] | 50.5 | 266 | 146.9 | 41.0 | 63.1 | 44.8 | 37.7 | 60.1 | 40.0 |  |  |  |  |  |  |
| PVT-Medium [72] | 63.9 | 295 | 117.3 | 42.0 | 64.4 | 45.6 | 39.0 | 61.6 | 42.1 |  |  |  |  |  |  |
| FasterNet-M | 71.2 | 344 | 71.4 | 43.0 | 64.4 | 47.4 | 39.1 | 61.5 | 42.3 |  |  |  |  |  |  |
| ResNeXt101-64×\times4d [77] | 101.9 | 487 | 112.9 | 42.8 | 63.8 | 47.3 | 38.4 | 60.6 | 41.3 |  |  |  |  |  |  |
| PVT-Large [72] | 81.0 | 358 | 152.2 | 42.9 | 65.0 | 46.6 | 39.5 | 61.9 | 42.5 |  |  |  |  |  |  |
| FasterNet-L | 110.9 | 484 | 93.8 | 44.0 | 65.6 | 48.2 | 39.9 | 62.3 | 43.0 |  |  |  |  |  |  |

| Params |
|---|
| (M) |

| FLOPs |
|---|
| (G) |

| Latency on |
|---|
| GPU (ms) |

| Ablation | Variant | Throughput on GPU (fps) | Throughput | on GPU | (fps) | Latency on CPU (ms) | Latency | on CPU | (ms) | Latency on ARM (ms) | Latency | on ARM | (ms) | Acc. (%) | Acc. | (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Throughput |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| on GPU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (fps) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Latency |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| on CPU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (ms) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Latency |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| on ARM |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (ms) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Acc. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (%) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Partial ratio | T0∗\text{T0}^{*} w/ r=1/2r=1/2 | 6626 | 9.6 | 145 | 71.7 |  |  |  |  |  |  |  |  |  |  |  |
| T0 w/ r=1/4r=1/4 | 6807 | 9.2 | 143 | 71.9 |  |  |  |  |  |  |  |  |  |  |  |  |
| T0∗\text{T0}^{*} w/ r=1/8r=1/8 | 6204 | 8.9 | 140 | 71.3 |  |  |  |  |  |  |  |  |  |  |  |  |
| Normalization | T0 w/ BN | 6807 | 9.2 | 143 | 71.9 |  |  |  |  |  |  |  |  |  |  |  |
| T0 w/ LN | 5515 | 10.7 | 159 | 71.9 |  |  |  |  |  |  |  |  |  |  |  |  |
| Activation | T0 w/ ReLU | 6929 | 8.2 | 114 | 71.3 |  |  |  |  |  |  |  |  |  |  |  |
| T0∗\text{T0}^{*} w/ ReLU | 5866 | 9.3 | 143 | 71.7 |  |  |  |  |  |  |  |  |  |  |  |  |
| T0 w/ GELU | 6807 | 9.2 | 143 | 71.9 |  |  |  |  |  |  |  |  |  |  |  |  |
| T2 w/ ReLU | 1991 | 33.5 | 497 | 78.9 |  |  |  |  |  |  |  |  |  |  |  |  |
| T2 w/ GELU | 1985 | 35.4 | 557 | 78.7 |  |  |  |  |  |  |  |  |  |  |  |  |

| Throughput |
|---|
| on GPU |
| (fps) |

| Latency |
|---|
| on CPU |
| (ms) |

| Latency |
|---|
| on ARM |
| (ms) |

| Acc. |
|---|
| (%) |

To verify the effectiveness and efficiency of our FasterNet, we first conduct experiments on the large-scale ImageNet-1k classification dataset [53]. It covers 1k categories of common objects and contains about 1.3M labeled images for training and 50k labeled images for validation. We train our models for 300 epochs using AdamW optimizer [44]. We set the batch size to 2048 for the FasterNet-M/L and 4096 for other variants. We use cosine learning rate scheduler [43] with a peak value of 0.001⋅batch size/10240.001\cdot\text{batch size}/1024 and a 20-epoch linear warmup. We apply commonly-used regularization and augmentation techniques, including Weight Decay [32], Stochastic Depth [28], Label Smoothing [59], Mixup [81], Cutmix [80] and Rand Augment [9], with varying magnitudes for different FasterNet variants. To reduce the training time, we use 192×192192\times 192 resolution for the first 280 training epochs and 224×224224\times 224 for the remaining 20 epochs. For fair comparison, we do not use knowledge distillation [23] and neural architecture search [87]. We report our top-1 accuracy on the validation set with a center crop at 224×224224\times 224 resolution and a 0.9 crop ratio. Detailed training and validation settings are provided in the appendix.

Fig. 7 and Tab. 3 demonstrate the superiority of our FasterNet over state-of-the-art classification models. The trade-off curves in Fig. 7 clearly show that FasterNet sets the new state-of-the-art in balancing accuracy and latency/throughput among all the networks examined. From another perspective, FasterNet runs faster than various CNN, ViT and MLP models on a wide range of devices, when having similar top-1 accuracy. As quantitatively shown in Tab. 3, FasterNet-T0 is 2.8×2.8\times, 3.3×3.3\times, and 2.4×2.4\times faster than MobileViT-XXS [48] on GPU, CPU, and ARM processors, respectively, while being 2.9% more accurate. Our large FasterNet-L achieves 83.5% top-1 accuracy, comparable to the emerging Swin-B [41] and ConvNeXt-B [42] while having 36% and 28% higher inference throughput on GPU, as well as saving 37% and 15% compute time on CPU. Given such promising results, we highlight that our FasterNet is much simpler than many other models in terms of architectural design, which showcases the feasibility of designing simple yet powerful neural networks.

### 4.4 FasterNet on downstream tasks

To further evaluate the generalization ability of FasterNet, we conduct experiments on the challenging COCO dataset [36] for object detection and instance segmentation. As a common practice, we employ the ImageNet pre-trained FasterNet as a backbone and equip it with the popular Mask R-CNN detector [19]. To highlight the effectiveness of the backbone itself, we simply follow PoolFormer [79] and adopt an AdamW optimizer, a 1×1\times training schedule (12 epochs), a batch size of 16, and other training settings without further hyper-parameter tuning.

Tab. 4 shows the results for comparison between FasterNet and representative models. FasterNet consistently outperforms ResNet and ResNext by having higher average precision (AP) with similar latency. Specifically, FasterNet-S yields +1.9+1.9 higher box AP and +2.4+2.4 higher mask AP compared to the standard baseline ResNet50. FasterNet is also competitive against the ViT variants. Under similar FLOPs, FasterNet-L reduces PVT-Large’s latency by 38%, i.e., from 152.2 ms to 93.8 ms on GPU, and achieves +1.1+1.1 higher box AP and +0.4+0.4 higher mask AP.

### 4.5 Ablation study

We conduct a brief ablation study on the value of partial ratio rr and the choices of activation and normalization layers. We compare different variants in terms of ImageNet top-1 accuracy and on-device latency/throughput. Results are summarized in Tab. 5. For the partial ratio rr, we set it to 14\frac{1}{4} for all FasterNet variants by default, which achieves higher accuracy, higher throughput, and lower latency at similar complexity. A too large partial ratio rr would make PConv degrade to a regular Conv, while a too small value would render PConv less effective in capturing the spatial features. For the normalization layers, we choose BatchNorm over LayerNorm because BatchNorm can be merged into its adjacent convolutional layers for faster inference while it is as effective as LayerNorm in our experiment. For the activation function, interestingly, we empirically found that GELU fits FasterNet-T0/T1 models more efficiently than ReLU. It, however, becomes opposite for FasterNet-T2/S/M/L. Here we only show two examples in Tab. 5 due to space constraint. We conjecture that GELU strengthens FasterNet-T0/T1 by having higher non-linearity, while the benefit fades away for larger FasterNet variants.

## 5 Conclusion

In this paper, we have investigated the common and unresolved issue that many established neural networks suffer from low floating-point operations per second (FLOPS). We have revisited a bottleneck operator, DWConv, and analyzed its main cause for a slowdown – frequent memory access. To overcome the issue and achieve faster neural networks, we have proposed a simple yet fast and effective operator, PConv, that can be readily plugged into many existing networks. We have further introduced our general-purpose FasterNet, built upon our PConv, that achieves state-of-the-art speed and accuracy trade-off on various devices and vision tasks. We hope that our PConv and FasterNet would inspire more research on simple yet effective neural networks, going beyond academia to impact the industry and community directly.

Acknowledgement This work was supported, in part, by Hong Kong General Research Fund under grant number 16200120. The work of C.-H. Lee was supported, in part, by the NSF under Grant IIS-2209921.

## Appendix

In this appendix, we provide further details on the experimental settings, full comparison plots, architectural configurations, PConv implementations, comparisons with related work, limitations, and future work.

## Appendix A ImageNet-1k experimental settings

We provide ImageNet-1k training and evaluation settings in Tab. 6. They can be used for reproducing our main results in Tab. 3 and Fig. 7. Different FasterNet variants vary in the magnitude of regularization and augmentation techniques. The magnitude increases as the model becomes larger to alleviate overfitting and improve accuracy. Note that most of the compared works in Tab. 3 and Fig. 7, e.g., MobileViT, EdgeNext, PVT, CycleMLP, ConvNeXt, Swin, etc., also adopt such advanced training techniques (ADT). Some even heavily rely on the hyper-parameter search. For others w/o ADT, i.e., ShuffleNetV2, MobileNetV2, and GhostNet, though the comparison is not totally fair, we include them for reference.

## Appendix B Downstream tasks experimental settings

For object detection and instance segmentation on the COCO2017 dataset, we equip our FasterNet backbone with the popular Mask R-CNN detector. We use ImageNet-1k pre-trained weights to initialize the backbone and Xavier to initialize the add-on layers. Detailed settings are summarized in Tab. 7.

## Appendix C Full comparison plots on ImageNet-1k

Fig. 8 shows the full comparison plots on ImageNet-1k, which is the extension of Fig. 7 in the main paper with a larger range of latency. Fig. 8 shows consistent results that FasterNet strikes better trade-offs than others in balancing accuracy and latency/throughput on GPU, CPU, and ARM processors.

## Appendix D Detailed architectural configurations

We present the detailed architectural configurations in Tab. 8. While different FasterNet variants share a unified architecture, they vary in the network width (the number of channels) and network depth (the number of FasterNet blocks at each stage). The classifier at the end of the architecture is used for classification tasks but removed for other downstream tasks.

| Variants | T0 | T1 | T2 | S | M | L |
|---|---|---|---|---|---|---|
| Train Res | 192 for epoch 1∼\sim280, 224 for epoch 281∼\sim300 |  |  |  |  |  |
| Test Res | 224 |  |  |  |  |  |
| Epochs | 300 |  |  |  |  |  |
| # of forward pass | 188k |  |  |  |  |  |
| Batch size | 4096 | 4096 | 4096 | 4096 | 2048 | 2048 |
| Optimizer | AdamW |  |  |  |  |  |
| Momentum | 0.9/0.999 |  |  |  |  |  |
| LR | 0.004 | 0.004 | 0.004 | 0.004 | 0.002 | 0.002 |
| LR decay | cosine |  |  |  |  |  |
| Weight decay | 0.005 | 0.01 | 0.02 | 0.03 | 0.05 | 0.05 |
| Warmup epochs | 20 |  |  |  |  |  |
| Warmup schedule | linear |  |  |  |  |  |
| Label smoothing | 0.1 |  |  |  |  |  |
| Dropout | ✗ |  |  |  |  |  |
| Stoch. Depth | ✗ | 0.02 | 0.05 | 0.1 | 0.2 | 0.3 |
| Repeated Aug | ✗ |  |  |  |  |  |
| Gradient Clip. | ✗ | ✗ | ✗ | ✗ | 1 | 0.01 |
| H. flip | ✓ |  |  |  |  |  |
| RRC | ✓ |  |  |  |  |  |
| Rand Augment | ✗ | 3/0.5 | 5/0.5 | 7/0.5 | 7/0.5 | 7/0.5 |
| Auto Augment | ✗ |  |  |  |  |  |
| Mixup alpha | 0.05 | 0.1 | 0.1 | 0.3 | 0.5 | 0.7 |
| Cutmix alpha | 1.0 |  |  |  |  |  |
| Erasing prob. | ✗ |  |  |  |  |  |
| Color Jitter | ✗ |  |  |  |  |  |
| PCA lighting | ✗ |  |  |  |  |  |
| SWA | ✗ |  |  |  |  |  |
| EMA | ✗ |  |  |  |  |  |
| Layer scale | ✗ |  |  |  |  |  |
| CE loss | ✓ |  |  |  |  |  |
| BCE loss | ✗ |  |  |  |  |  |
| Mixed precision | ✓ |  |  |  |  |  |
| Test crop ratio | 0.9 |  |  |  |  |  |
| Top-1 acc. (%) | 71.9 | 76.2 | 78.9 | 81.3 | 83.0 | 83.5 |

| Variants | S | M | L |
|---|---|---|---|
| Train and test Res | shorter side == 800, longer side ≤\leq 1333 |  |  |
| Batch size | 16 (2 on each GPU) |  |  |
| Optimizer | AdamW |  |  |
| Train schedule | 1×\times schedule (12 epochs) |  |  |
| Weight decay | 0.0001 |  |  |
| Warmup schedule | linear |  |  |
| Warmup iterations | 500 |  |  |
| LR decay | StepLR at epoch 8 and 11 with decay rate 0.1 |  |  |
| LR | 0.0002 | 0.0001 | 0.0001 |
| Stoch. Depth | 0.15 | 0.2 | 0.3 |

## Appendix E More comparisons with related work

Improving FLOPS. There are a few other works [11, 76] also looking into the FLOPS issue and trying to improve it. They generally follow existing operators and try to find their proper configurations, e.g., RepLKNet [11] simply increases the kernel size while TRT-ViT [76] reorders different blocks in the architecture. By contrast, this paper advances the field by proposing a novel and efficient PConv, opening up new directions and potentially larger room for FLOPS improvement.

| Name | Output size | Layer specification | T0 | T1 | T2 | S | M | L |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Embedding | h4×w4\frac{h}{4}\times\frac{w}{4} | Conv_4_cc_4, BN | Conv_4_cc_4, | BN | # Channels cc | 40 | 64 | 96 | 128 | 144 | 192 |  |  |
| Conv_4_cc_4, |  |  |  |  |  |  |  |  |  |  |  |  |  |
| BN |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Stage 1 | h4×w4\frac{h}{4}\times\frac{w}{4} | [c2cc]×b1\left[\text{\begin{tabular}[c]{@{}c@{}}PConv\_3\_$c$\_1\_1/4,\\ Conv\_1\_$2c$\_1,\\ BN, Acti,\\ Conv\_1\_$c$\_1\end{tabular}}\right]\times b_{1} | # Blocks b1b_{1} | 1 | 1 | 1 | 1 | 3 | 3 |  |  |  |  |
| Merging | h8×w8\frac{h}{8}\times\frac{w}{8} | Conv_2_2​c2c_2, BN | Conv_2_2​c2c_2, | BN | # Channels 2​c2c | 80 | 128 | 192 | 256 | 288 | 384 |  |  |
| Conv_2_2​c2c_2, |  |  |  |  |  |  |  |  |  |  |  |  |  |
| BN |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Stage 2 | h8×w8\frac{h}{8}\times\frac{w}{8} | [2c4c2c]×b2\left[\text{\begin{tabular}[c]{@{}c@{}}PConv\_3\_$2c$\_1\_1/4,\\ Conv\_1\_$4c$\_1,\\ BN, Acti,\\ Conv\_1\_$2c$\_1\end{tabular}}\right]\times b_{2} | # Blocks b2b_{2} | 2 | 2 | 2 | 2 | 4 | 4 |  |  |  |  |
| Merging | h16×w16\frac{h}{16}\times\frac{w}{16} | Conv_2_4​c4c_2, BN | Conv_2_4​c4c_2, | BN | # Channels 4​c4c | 160 | 256 | 384 | 512 | 576 | 768 |  |  |
| Conv_2_4​c4c_2, |  |  |  |  |  |  |  |  |  |  |  |  |  |
| BN |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Stage 3 | h16×w16\frac{h}{16}\times\frac{w}{16} | [4c8c4c]×b3\left[\text{\begin{tabular}[c]{@{}c@{}}PConv\_3\_$4c$\_1\_1/4,\\ Conv\_1\_$8c$\_1,\\ BN, Acti,\\ Conv\_1\_$4c$\_1\end{tabular}}\right]\times b_{3} | # Blocks b3b_{3} | 8 | 8 | 8 | 13 | 18 | 18 |  |  |  |  |
| Merging | h32×w32\frac{h}{32}\times\frac{w}{32} | Conv_2_8​c8c_2, BN | Conv_2_8​c8c_2, | BN | # Channels 8​c8c | 320 | 512 | 768 | 1024 | 1152 | 1536 |  |  |
| Conv_2_8​c8c_2, |  |  |  |  |  |  |  |  |  |  |  |  |  |
| BN |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Stage 4 | h32×w32\frac{h}{32}\times\frac{w}{32} | [8c16c8c]×b4\left[\text{\begin{tabular}[c]{@{}c@{}}PConv\_3\_$8c$\_1\_1/4,\\ Conv\_1\_$16c$\_1,\\ BN, Acti,\\ Conv\_1\_$8c$\_1\end{tabular}}\right]\times b_{4} | # Blocks b4b_{4} | 2 | 2 | 2 | 2 | 3 | 3 |  |  |  |  |
| Classifier | 1×11\times 1 | Global average pool, Conv_1_1280_1, Acti, FC_1000 | Global average pool, | Conv_1_1280_1, | Acti, | FC_1000 | Acti | GELU | GELU | ReLU | ReLU | ReLU | ReLU |
| Global average pool, |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Conv_1_1280_1, |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Acti, |  |  |  |  |  |  |  |  |  |  |  |  |  |
| FC_1000 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| FLOPs (G) | 0.34 | 0.85 | 1.90 | 4.55 | 8.72 | 15.49 |  |  |  |  |  |  |  |
| Params (M) | 3.9 | 7.6 | 15.0 | 31.1 | 53.5 | 93.4 |  |  |  |  |  |  |  |

| Conv_4_cc_4, |
|---|
| BN |

| Conv_2_2​c2c_2, |
|---|
| BN |

| Conv_2_4​c4c_2, |
|---|
| BN |

| Conv_2_8​c8c_2, |
|---|
| BN |

| Global average pool, |
|---|
| Conv_1_1280_1, |
| Acti, |
| FC_1000 |

PConv vs. GConv. PConv is schematically equivalent to a modified GConv [31] that operates on a single group and leaves other groups untouched. Though simple, such a modification remains unexplored before. It’s also significant in the sense that it prevents the operator from excessive memory access and is computationally more efficient. From the perspective of low-rank approximations, PConv improves GConv by further reducing the intra-filter redundancy beyond the inter-filter redundancy [16].

FasterNet vs. ConvNeXt. Our FasterNet appears similar to ConvNeXt [42] after substituting DWConv with our PConv. However, they are different in motivations. While ConvNeXt searches for a better structure by trial and error, we append PWConv after PConv to better aggregate information from all channels. Moreover, ConvNeXt follows ViT to use fewer activation functions, while we intentionally remove them from the middle of PConv and PWConv, to minimize their error in approximating a regular Conv.

Other paradigms for efficient inference. Our work focuses on efficient network design, orthogonal to the other paradigms, e.g., neural architecture search (NAS) [13], network pruning [50], and knowledge distillation [23]. They can be applied in this paper for better performance. However, we opt not to do so to keep our core idea centered and to make the performance gain clear and fair.

Other partial/masked convolution works. There are several works [38, 14, 37] sharing similar names with our PConv. However, they differ a lot in objectives and methods. For example, they apply filters on partial pixels to exclude invalid patches [38], enable self-supervised learning [14], or synthesize novel images [37], while we target at the channel dimension for efficient inference.

## Appendix F Limitations and future work

We have demonstrated that PConv and FasterNet are fast and effective, being competitive with existing operators and networks. Yet there are some minor technical limitations of this paper. For one thing, PConv is designed to apply a regular convolution on only a part of the input channels while leaving the remaining ones untouched. Thus, the stride of the partial convolution should always be 1, in order to align the spatial resolution of the convolutional output and that of the untouched channels. Note that it is still feasible to down-sample the spatial resolution as there can be additional downsampling layers in the architecture. And for another, our FasterNet is simply built upon convolutional operators with a possibly limited receptive field. Future efforts can be made to enlarge its receptive field and combine it with other operators to pursue higher accuracy.
