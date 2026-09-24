# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction

High-resolution dense prediction enables many appealing real-world applications, such as computational photography, autonomous driving, etc. However, the vast computational cost makes deploying state-of-the-art high-resolution dense prediction models on hardware devices difficult. This work presents EfficientViT, a new family of high-resolution vision models with novel multi-scale linear attention. Unlike prior high-resolution dense prediction models that rely on heavy softmax attention, hardware-inefficient large-kernel convolution, or complicated topology structure to obtain good performances, our multi-scale linear attention achieves the global receptive field and multi-scale learning (two desirable features for high-resolution dense prediction) with only lightweight and hardware-efficient operations. As such, EfficientViT delivers remarkable performance gains over previous state-of-the-art models with significant speedup on diverse hardware platforms, including mobile CPU, edge GPU, and cloud GPU. Without performance loss on Cityscapes, our EfficientViT provides up to 13.9×\times and 6.2×\times GPU latency reduction over SegFormer and SegNeXt, respectively. For super-resolution, EfficientViT delivers up to 6.4×\times speedup over Restormer while providing 0.11dB gain in PSNR. For Segment Anything, EfficientViT delivers 48.9×\times higher throughput on A100 GPU while achieving slightly better zero-shot instance segmentation performance on COCO.

## 1 Introduction

High-resolution dense prediction is a fundamental task in computer vision and has broad applications in the real world, including autonomous driving, medical image processing, computational photography, etc. Therefore, deploying state-of-the-art (SOTA) high-resolution dense prediction models on hardware devices can benefit many use cases.

However, there is a large gap between the computational cost required by SOTA high-resolution dense prediction models and the limited resources of hardware devices. It makes using these models in real-world applications impractical. In particular, high-resolution dense prediction models require high-resolution images and strong context information extraction ability to work well [1, 2, 3, 4, 5, 6]. Therefore, directly porting efficient model architectures from image classification is unsuitable for high-resolution dense prediction.

This work introduces EfficientViT, a new family of vision transformer models for efficient high-resolution dense prediction. The core of EfficientViT is a new multi-scale linear attention module that enables the global receptive field and multi-scale learning with hardware-efficient operations. Our module is motivated by prior SOTA high-resolution dense prediction models. They demonstrate that the multi-scale learning [3, 4] and global receptive field [7] are critical in improving models’ performances. However, they do not consider hardware efficiency when designing their models, which is essential for real-world applications. For example, SegFormer [7] introduces softmax attention [8] into the backbone to have a global receptive field. However, its computational complexity is quadratic to the input resolution, making it unable to handle high-resolution images efficiently. SegNeXt [9] proposes a multi-branch module with large-kernel convolutions (kernel size up to 21) to enable a large receptive field and multi-scale learning. However, large-kernel convolution requires exceptional support on hardware to achieve good efficiency [10, 11], which is usually unavailable on hardware devices.

Hence, the design principle of our module is to enable these two critical features while avoiding hardware-inefficient operations. Specifically, we propose substituting the inefficient softmax attention with lightweight ReLU linear attention [12] to have the global receptive field. By leveraging the associative property of matrix multiplication, ReLU linear attention can reduce the computational complexity from quadratic to linear while preserving functionality. In addition, it avoids hardware-inefficient operations like softmax, making it more suitable for hardware deployment (Figure 4).

However, ReLU linear attention alone has limited capacity due to the lack of local information extraction and multi-scale learning ability. Therefore, we propose to enhance ReLU linear attention with convolution and introduce the multi-scale linear attention module to address the capacity limitation of ReLU linear attention. Specifically, we aggregate nearby tokens with small-kernel convolutions to generate multi-scale tokens. We perform ReLU linear attention on multi-scale tokens (Figure 2) to combine the global receptive field with multi-scale learning. We also insert depthwise convolutions into FFN layers to further improve the local feature extraction capacity.

We extensively evaluate EfficientViT on two popular high-resolution dense prediction tasks: semantic segmentation and super-resolution. EfficientViT provides significant performance boosts over prior SOTA high-resolution dense prediction models. More importantly, EfficientViT does not involve hardware-inefficient operations, so our #FLOPs reduction can easily translate to latency reduction on hardware devices (Figure 1).

In addition to these conventional high-resolution dense prediction tasks, we apply EfficientViT to Segment Anything [13], an emerging promptable segmentation task that allows zero-shot transfer to many vision tasks. EfficientViT achieves 48.9×\times acceleration on A100 GPU than SAM-ViT-Huge [13] without performance loss. We summarize our contributions as follows:

- • We introduce a new multi-scale linear attention module for efficient high-resolution dense prediction. It achieves the global receptive field and multi-scale learning while maintaining good efficiency on hardware. To the best of our knowledge, our work is the first to demonstrate the effectiveness of linear attention for high-resolution dense prediction.
We introduce a new multi-scale linear attention module for efficient high-resolution dense prediction. It achieves the global receptive field and multi-scale learning while maintaining good efficiency on hardware. To the best of our knowledge, our work is the first to demonstrate the effectiveness of linear attention for high-resolution dense prediction.

- • We design EfficientViT, a new family of high-resolution vision models, based on the proposed multi-scale linear attention module.
We design EfficientViT, a new family of high-resolution vision models, based on the proposed multi-scale linear attention module.

- • Our model demonstrates remarkable speedup on semantic segmentation, super-resolution, Segment Anything, and ImageNet classification on diverse hardware platforms (mobile CPU, edge GPU, and cloud GPU) over prior SOTA models.
Our model demonstrates remarkable speedup on semantic segmentation, super-resolution, Segment Anything, and ImageNet classification on diverse hardware platforms (mobile CPU, edge GPU, and cloud GPU) over prior SOTA models.

## 2 Method

This section first introduces the multi-scale linear attention module. Unlike prior works, our multi-scale linear attention simultaneously achieves the global receptive field and multi-scale learning with only hardware-efficient operations. Then, based on the multi-scale linear attention, we present a new family of vision transformer models named EfficientViT for high-resolution dense prediction.

### 2.1 Multi-Scale Linear Attention

Our multi-scale linear attention balances two crucial aspects of efficient high-resolution dense prediction, i.e., performance and efficiency. Specifically, the global receptive field and multi-scale learning are essential from the performance perspective. Previous SOTA high-resolution dense prediction models provide strong performances by enabling these features but fail to provide good efficiency. Our module tackles this issue by trading slight capacity loss for significant efficiency improvements.

An illustration of the proposed multi-scale linear attention module is provided in Figure 2 (right). In particular, we propose to use ReLU linear attention [12] to enable the global receptive field instead of the heavy softmax attention [8]. While ReLU linear attention [12] and other linear attention modules [14, 15, 16, 17] have been explored in other domains, it has never been successfully applied to high-resolution dense prediction. To the best of our knowledge, EfficientViT is the first work demonstrating ReLU linear attention’s effectiveness in high-resolution dense prediction. In addition, our work introduces novel designs to address its capacity limitation.

Given input x∈ℝN×fx\in\mr^{N\times f}, the generalized form of softmax attention can be written as:

|  | Oi=∑j=1NS​i​m​(Qi,Kj)∑j=1NS​i​m​(Qi,Kj)​Vj,\small O_{i}=\sum_{j=1}^{N}\frac{Sim(Q_{i},K_{j})}{\sum_{j=1}^{N}{Sim(Q_{i},K_{j})}}V_{j}, |  | (1) |
|---|---|---|---|

where Q=x​WQQ=xW_{Q}, K=x​WKK=xW_{K}, V=x​WVV=xW_{V} and WQ/WK/WV∈ℝf×dW_{Q}/W_{K}/W_{V}\in\mr^{f\times d} is the learnable linear projection matrix. OiO_{i} represents the i-th row of matrix OO. S​i​m​(⋅,⋅)Sim(\cdot,\cdot) is the similarity function. When using the similarity function S​i​m​(Q,K)=exp⁡(Q​KTd)Sim(Q,K)=\exp(\frac{QK^{T}}{\sqrt{d}}), Eq. (1) becomes the original softmax attention [8].

Apart from exp⁡(Q​KTd)\exp(\frac{QK^{T}}{\sqrt{d}}), we can use other similarity functions. In this work, we use ReLU linear attention [12] to achieve both the global receptive field and linear computational complexity. In ReLU linear attention, the similarity function is defined as

|  | S​i​m​(Q,K)=ReLU​(Q)​ReLU​(K)T.\small Sim(Q,K)=\text{ReLU}(Q)\text{ReLU}(K)^{T}. |  | (2) |
|---|---|---|---|

With S​i​m​(Q,K)=ReLU​(Q)​ReLU​(K)TSim(Q,K)=\text{ReLU}(Q)\text{ReLU}(K)^{T}, Eq. (1) can be rewritten as:

|  | Oi\displaystyle O_{i} | =∑j=1NReLU​(Qi)​ReLU​(Kj)T∑j=1NReLU​(Qi)​ReLU​(Kj)T​Vj\displaystyle=\sum_{j=1}^{N}\frac{\text{ReLU}(Q_{i})\text{ReLU}(K_{j})^{T}}{\sum_{j=1}^{N}{{\color[rgb]{1,0,0}\text{ReLU}(Q_{i})}\text{ReLU}(K_{j})^{T}}}V_{j} |  |
|---|---|---|---|
|  |  | =∑j=1N(ReLU​(Qi)​ReLU​(Kj)T)​VjReLU​(Qi)​∑j=1NReLU​(Kj)T.\displaystyle=\frac{\sum_{j=1}^{N}(\text{ReLU}(Q_{i})\text{ReLU}(K_{j})^{T})V_{j}}{{\color[rgb]{1,0,0}\text{ReLU}(Q_{i})}\sum_{j=1}^{N}\text{ReLU}(K_{j})^{T}}. |  |

Then, we can leverage the associative property of matrix multiplication to reduce the computational complexity and memory footprint from quadratic to linear without changing its functionality:

|  | Oi\displaystyle O_{i} | =∑j=1N[ReLU​(Qi)​ReLU​(Kj)T]​VjReLU​(Qi)​∑j=1NReLU​(Kj)T\displaystyle=\frac{\sum_{j=1}^{N}{\color[rgb]{1,0,0}[\text{ReLU}(Q_{i})\text{ReLU}(K_{j})^{T}]}V_{j}}{\text{ReLU}(Q_{i})\sum_{j=1}^{N}\text{ReLU}(K_{j})^{T}} |  |  |
|---|---|---|---|---|
|  |  | =∑j=1NReLU​(Qi)​[(ReLU​(Kj)T​Vj)]ReLU​(Qi)​∑j=1NReLU​(Kj)T\displaystyle=\frac{\sum_{j=1}^{N}{\color[rgb]{0,0,1}\text{ReLU}(Q_{i})}{\color[rgb]{1,0,0}[(\text{ReLU}(K_{j})^{T}V_{j})]}}{\text{ReLU}(Q_{i})\sum_{j=1}^{N}\text{ReLU}(K_{j})^{T}} |  |  |
|  |  | =ReLU​(Qi)​(∑j=1NReLU​(Kj)T​Vj)ReLU​(Qi)​(∑j=1NReLU​(Kj)T).\displaystyle=\frac{{\color[rgb]{0,0,1}\text{ReLU}(Q_{i})}(\sum_{j=1}^{N}\text{ReLU}(K_{j})^{T}V_{j})}{\text{ReLU}(Q_{i})(\sum_{j=1}^{N}\text{ReLU}(K_{j})^{T})}. |  | (3) |

As shown in Eq. (3), we only need to compute (∑j=1NReLU​(Kj)T​Vj)(\sum_{j=1}^{N}\text{ReLU}(K_{j})^{T}V_{j}) ∈ℝd×d\in\mr^{d\times d} and (∑j=1NReLU​(Kj)T)(\sum_{j=1}^{N}\text{ReLU}(K_{j})^{T}) ∈ℝd×1\in\mr^{d\times 1} once, then can reuse them for each query, thereby only requires 𝒪⁡(N)\mathcal{O}(N) computational cost and 𝒪⁡(N)\mathcal{O}(N) memory.

Another key merit of ReLU linear attention is that it does not involve hardware-unfriendly operations like softmax, making it more efficient on hardware. For example, Figure 4 shows the latency comparison between softmax attention and ReLU linear attention. With similar computation, ReLU linear attention is significantly faster than softmax attention on the mobile CPU.

Although ReLU linear attention is superior to softmax attention in terms of computational complexity and hardware latency, ReLU linear attention has limitations. Figure 3 visualizes the attention maps of softmax attention and ReLU linear attention. Because of the lack of the non-linear similarity function, ReLU linear attention cannot generate concentrated attention maps, making it weak at capturing local information.

To mitigate its limitation, we propose to enhance ReLU linear attention with convolution. Specifically, we insert a depthwise convolution in each FFN layer. An overview of the resulting building block is illustrated in Figure 2 (left), where the ReLU linear attention captures context information and the FFN+DWConv captures local information.

Furthermore, we propose to aggregate the information from nearby Q/K/V tokens to get multi-scale tokens to enhance the multi-scale learning ability of ReLU linear attention. This information aggregation process is independent for each Q, K, and V in each head. We only use small-kernel depthwise-separable convolutions [18] for information aggregation to avoid hurting hardware efficiency. In the practical implementation, independently executing these aggregation operations is inefficient on GPU. Therefore, we take advantage of the group convolution to reduce the number of total operations. Specifically, all DWConvs are fused into a single DWConv while all 1x1 Convs are combined into a single 1x1 group convolution (Figure 2 right) where the number of groups is 3×#heads3\times\text{\#heads} and the number of channels in each group is d. After getting multi-scale tokens, we perform ReLU linear attention upon them to extract multi-scale global features. Finally, we concatenate the features along the head dimension and feed them to the final linear projection layer to fuse the features.

### 2.2 EfficientViT Architecture

We build a new family of vision transformer models based on the proposed multi-scale linear attention module. The core building block (denoted as ‘EfficientViT Module’) is illustrated in Figure 2 (left). The macro architecture of EfficientViT is demonstrated in Figure 5. We use the standard backbone-head/encoder-decoder architecture design.

- • Backbone. The backbone of EfficientViT also follows the standard design, which consists of the input stem and four stages with gradually decreased feature map size and gradually increased channel number. We insert the EfficientViT module in Stages 3 and 4. For downsampling, we use an MBConv with stride 2.
Backbone. The backbone of EfficientViT also follows the standard design, which consists of the input stem and four stages with gradually decreased feature map size and gradually increased channel number. We insert the EfficientViT module in Stages 3 and 4. For downsampling, we use an MBConv with stride 2.

- • Head. P2, P3, and P4 denote the outputs of Stages 2, 3, and 4, forming a pyramid of feature maps. For simplicity and efficiency, we use 1x1 convolution and standard upsampling operation (e.g., bilinear/bicubic upsampling) to match their spatial and channel size and fuse them via addition. Since our backbone already has a strong context information extraction capacity, we adopt a simple head design that comprises several MBConv blocks and the output layers (i.e., prediction and upsample). In the experiments, we empirically find this simple head design is sufficient for achieving SOTA performances. In addition to dense prediction, our model can be applied to other vision tasks, such as image classification, by combining the backbone with task-specific heads.
Head. P2, P3, and P4 denote the outputs of Stages 2, 3, and 4, forming a pyramid of feature maps. For simplicity and efficiency, we use 1x1 convolution and standard upsampling operation (e.g., bilinear/bicubic upsampling) to match their spatial and channel size and fuse them via addition. Since our backbone already has a strong context information extraction capacity, we adopt a simple head design that comprises several MBConv blocks and the output layers (i.e., prediction and upsample). In the experiments, we empirically find this simple head design is sufficient for achieving SOTA performances.

In addition to dense prediction, our model can be applied to other vision tasks, such as image classification, by combining the backbone with task-specific heads.

Following the same macro architecture, we design a series of models with different sizes to satisfy various efficiency constraints. We name these models EfficientViT-B0, EfficientViT-B1, EfficientViT-B2, and EfficientViT-B3, respectively. In addition, we designed the EfficientViT-L series for the cloud platforms. Detailed configurations of these models are provided in our official GitHub repository11 1 https://github.com/mit-han-lab/efficientvit.

## 3 Experiments

### 3.1 Setups

| Components | mIoU ↑\uparrow | Params ↓\downarrow | MACs ↓\downarrow |  |
|---|---|---|---|---|
| Multi-scale | Global att. |  |  |  |
|  |  | 68.1 | 0.7M | 4.4G |
| ✓ |  | 72.3 | 0.7M | 4.4G |
|  | ✓ | 72.2 | 0.7M | 4.4G |
| ✓ | ✓ | 74.5 | 0.7M | 4.4G |

| Models | Top1 Acc ↑\uparrow | Top5 Acc ↑\uparrow | Params ↓\downarrow | MACs ↓\downarrow | Latency ↓\downarrow | Throughput ↑\uparrow |  |
|---|---|---|---|---|---|---|---|
| Nano(bs1) | Orin(bs1) | A100 (image/s) |  |  |  |  |  |
| CoAtNet-0 [19] | 81.6 | - | 25M | 4.2G | 95.8ms | 4.5ms | 3011 |
| ConvNeXt-T [20] | 82.1 | - | 29M | 4.5G | 87.9ms | 3.8ms | 3303 |
| EfficientViT-B2 (r256) | 82.7 | 96.1 | 24M | 2.1G | 58.5ms | 2.8ms | 5325 |
| Swin-B [21] | 83.5 | - | 88M | 15G | 240ms | 6.0ms | 2236 |
| CoAtNet-1 [19] | 83.3 | - | 42M | 8.4G | 171ms | 8.3ms | 1512 |
| ConvNeXt-S [20] | 83.1 | - | 50M | 8.7G | 146ms | 6.5ms | 2081 |
| EfficientViT-B3 (r224) | 83.5 | 96.4 | 49M | 4.0G | 101ms | 4.4ms | 3797 |
| CoAtNet-2 [19] | 84.1 | - | 75M | 16G | 254ms | 10.3ms | 1174 |
| ConvNeXt-B [20] | 83.8 | - | 89M | 15G | 211ms | 7.8ms | 1579 |
| EfficientViT-B3 (r288) | 84.2 | 96.7 | 49M | 6.5G | 141ms | 5.6ms | 2372 |
| CoAtNet-3 [19] | 84.5 | - | 168M | 35G | - | 15.4ms | 642 |
| ConvNeXt-L [20] | 84.3 | - | 198M | 34G | - | 11.5ms | 1032 |
| EfficientNetV2-S [22] | 83.9 | - | 22M | 8.8G | - | 4.3ms | 2869 |
| EfficientViT-L1 (r224) | 84.5 | 96.9 | 53M | 5.3G | - | 2.6ms | 6207 |
| EfficientNetV2-M [22] | 85.2 | - | 54M | 25G | - | 9.2ms | 1160 |
| FasterViT-4 [23] | 85.4 | 97.3 | 425M | 37G | - | 13.0ms | 1382 |
| EfficientViT-L2 (r288) | 85.6 | 97.4 | 64M | 11G | - | 4.3ms | 3102 |
| FasterViT-6 [23] | 85.8 | 97.4 | 1360M | 142G | - | - | 594 |
| EfficientNetV2-L [22] | 85.7 | - | 120M | 53G | - | - | 696 |
| EfficientViT-L2 (r384) | 86.0 | 97.5 | 64M | 20G | - | - | 1784 |

We evaluate the effectiveness of EfficientViT on three representative high-resolution dense prediction tasks, including semantic segmentation, super-resolution, and Segment Anything.

For semantic segmentation, we use two popular benchmark datasets: Cityscapes [24] and ADE20K [25]. In addition, we evaluate EfficientViT under two settings for super-resolution: lightweight super-resolution (SR) and high-resolution SR. We train models on DIV2K [26] for lightweight SR and test on BSD100 [27]. For high-resolution SR, we train models on the first 3000 training images of FFHQ [28] and test on the first 500 validation images of FFHQ22 2 https://rb.gy/7je1a.

Apart from dense prediction, we also study the effectiveness of EfficientViT for image classification using the ImageNet dataset [29].

We measure the mobile latency on Qualcomm Snapdragon 8Gen1 CPU with Tensorflow-Lite33 3 https://www.tensorflow.org/lite, batch size 1 and fp32. We use TensorRT44 4 https://docs.nvidia.com/deeplearning/tensorrt/ and fp16 to measure the latency on edge GPU and cloud GPU. The data transfer time is included in the reported latency/throughput results.

We implement our models using Pytorch [30] and train them on GPUs. We use the AdamW optimizer with cosine learning rate decay for training our models. For multi-scale linear attention, we use a two-branch design for the best trade-off between performance and efficiency, where 5x5 nearby tokens are aggregated to generate multi-scale tokens.

For semantic segmentation experiments, we use the mean Intersection over Union (mIoU) as our evaluation metric. The backbone is initialized with weights pretrained on ImageNet and the head is initialized randomly, following the common practice.

For super-resolution, we use PSNR and SSIM on the Y channel as the evaluation metrics, same as previous work [31]. The models are trained with random initialization.

### 3.2 Ablation Study

We conduct ablation study experiments on Cityscapes to study the effectiveness of two key design components of our EfficientViT module, i.e., multi-scale learning and global attention. To eliminate the impact of pre-training, we train all models from random initialization. In addition, we rescale the width of the models so that they have the same #MACs. The results are summarized in Table 1. We can see that removing either global attention or multi-scale learning will significantly hurt the performances. It shows that all of them are essential for achieving a better trade-off between performance and efficiency.

To understand the effectiveness of EfficientViT’s backbone in image classification, we train our models on ImageNet following the standard training strategy. We summarize the results and compare our models with SOTA image classification models in Table 2.

Though EfficientViT is designed for high-resolution dense prediction, it achieves highly competitive performances on ImageNet classification. In particular, EfficientViT-L2-r384 obtains 86.0 top1 accuracy on ImageNet, providing +0.3 accuracy gain over EfficientNetV2-L and 2.6x speedup on A100 GPU.

| Models | mIoU ↑\uparrow | Params ↓\downarrow | MACs ↓\downarrow | Latency ↓\downarrow | Throughput ↑\uparrow |  |
|---|---|---|---|---|---|---|
| Nano(bs1) | Orin(bs1) | A100(image/s) |  |  |  |  |
| DeepLabV3plus-Mbv2 [32] | 75.2 | 15M | 555G | - | 83.5ms | 102 |
| EfficientViT-B0 | 75.7 | 0.7M | 4.4G | 0.28s | 9.9ms | 263 |
| SegFormer-B1 [7] | 78.5 | 14M | 244G | 5.6s | 146ms | 49 |
| SegNeXt-T [9] | 79.8 | 4.3M | 51G | 2.2s | 93.2ms | 95 |
| EfficientViT-B1 | 80.5 | 4.8M | 25G | 0.82s | 24.3ms | 175 |
| SegFormer-B3 [7] | 81.7 | 47M | 963G | - | 407ms | 18 |
| SegNeXt-S [9] | 81.3 | 14M | 125G | 3.4s | 127ms | 70 |
| EfficientViT-B2 | 82.1 | 15M | 74G | 1.7s | 46.5ms | 112 |
| SegFormer-B5 [7] | 82.4 | 85M | 1460G | - | 638ms | 12 |
| SegNeXt-B [9] | 82.6 | 28M | 276G | - | 228ms | 41 |
| EfficientViT-B3 | 83.0 | 40M | 179G | - | 81.8ms | 70 |
| EfficientViT-L1 | 82.7 | 40M | 282G | - | 45.9ms | 122 |
| SegNeXt-L [9] | 83.2 | 49M | 578G | - | 374ms | 26 |
| EfficientViT-L2 | 83.2 | 53M | 396G | - | 60.0ms | 102 |

| Models | mIoU ↑\uparrow | Params ↓\downarrow | MACs ↓\downarrow | Latency ↓\downarrow | Throughput ↑\uparrow |  |
|---|---|---|---|---|---|---|
| Nano(bs1) | Orin(bs1) | A100(image/s) |  |  |  |  |
| SegFormer-B1 [7] | 42.2 | 14M | 16G | 389ms | 12.3ms | 542 |
| SegNeXt-T [9] | 41.1 | 4.3M | 6.6G | 281ms | 12.4ms | 842 |
| EfficientViT-B1 | 42.8 | 4.8M | 3.1G | 110ms | 4.0ms | 1142 |
| SegNeXt-S [9] | 44.3 | 14M | 16G | 428ms | 17.2ms | 592 |
| EfficientViT-B2 | 45.9 | 15M | 9.1G | 212ms | 7.3ms | 846 |
| Mask2Former [33] | 47.7 | 47M | 74G | - | - | - |
| MaskFormer [34] | 46.7 | 42M | 55G | - | - | - |
| SegFormer-B2 [7] | 46.5 | 28M | 62G | 920ms | 24.3ms | 345 |
| SegNeXt-B [9] | 48.5 | 28M | 35G | 806ms | 32.9ms | 347 |
| EfficientViT-B3 | 49.0 | 39M | 22G | 411ms | 12.5ms | 555 |
| EfficientViT-L1 | 49.2 | 40M | 36G | - | 7.2ms | 947 |
| SegFormer-B4 [7] | 50.3 | 64M | 96G | - | 44.9ms | 212 |
| EfficientViT-L2 | 50.7 | 51M | 45G | - | 9.0ms | 758 |

| Model | FFHQ (512x512 →\rightarrow 1024x1024) | BSD100 (160x240 →\rightarrow 320x480) |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
| PSNR ↑\uparrow | SSIM ↑\uparrow | A100(bs1) ↓\downarrow | Speedup ↑\uparrow | PSNR ↑\uparrow | SSIM ↑\uparrow | A100(bs1) ↓\downarrow | Speedup ↑\uparrow |  |
| Restormer [35] | 43.43 | 0.9806 | 92.0ms | 1x | 32.31 | 0.9021 | 15.1ms | 1x |
| SwinIR [31] | 43.49 | 0.9807 | 61.2ms | 1.5x | 32.31 | 0.9012 | 9.7ms | 1.6x |
| VapSR [36] | - | - | - | - | 32.27 | 0.9011 | 4.8ms | 3.1x |
| BSRN [37] | - | - | - | - | 32.24 | 0.9006 | 4.5ms | 3.4x |
| EfficientViT w0.75 | 43.54 | 0.9809 | 14.3ms | 6.4x | 32.31 | 0.9016 | 2.8ms | 5.4x |
| EfficientViT | 43.58 | 0.9810 | 17.8ms | 5.2x | 32.33 | 0.9019 | 3.2ms | 4.7x |

### 3.3 Semantic Segmentation

Table 3 reports the comparison between EfficientViT and SOTA semantic segmentation models on Cityscapes. EfficientViT achieves remarkable efficiency improvements over prior SOTA semantic segmentation models without sacrificing performances. Specifically, compared with SegFormer, EfficientViT obtains up to 13x #MACs saving and up to 8.8x latency reduction on the edge GPU (Jetson AGX Orin) with higher mIoU. Compared with SegNeXt, EfficientViT provides up to 2.0x MACs reduction and 3.8x speedup on the edge GPU (Jetson AGX Orin) while maintaining higher mIoU. On A100 GPU, EfficientViT delivers up to 3.9x higher throughput than SegNeXt and 10.2x higher throughput than SegFormer while achieving the same or higher mIoU. Having similar computational cost, EfficientViT also yields significant performance gains over previous SOTA models. For example, EfficientViT-B3 delivers +4.5 mIoU gain over SegFormer-B1 with lower MACs.

In addition to the quantitative results, we visualize EfficientViT and the baseline models qualitatively on Cityscapes. The results are shown in Figure 6. We can find that EfficientViT can better recognize boundaries and small objects than the baseline models while achieving lower latency on GPU.

Table 4 summarizes the comparison between EfficientViT and SOTA semantic segmentation models on ADE20K. Like Cityscapes, we can see that EfficientViT also achieves significant efficiency improvements on ADE20K. For example, with +0.6 mIoU gain, EfficientViT-B1 provides 5.2x MACs reduction and up to 3.5x GPU latency reduction than SegFormer-B1. With +1.6 mIoU gain, EfficientViT-B2 requires 1.8x fewer computational costs and runs 2.4x faster on Jetson AGX Orin GPU than SegNeXt-S.

### 3.4 Super-Resolution

Table 5 presents the comparison of EfficientViT with SOTA ViT-based SR methods (SwinIR [31] and Restormer [35]) and SOTA CNN-based SR methods (VapSR [36] and BSRN [37]). EfficientViT provides a better latency-performance trade-off than all compared methods.

On lightweight SR, EfficientViT provides up to 0.09dB gain in PSNR on BSD100 while maintaining the same or lower GPU latency compared with SOTA CNN-based SR methods. Compared with SOTA ViT-based SR methods, EfficientViT provides up to 5.4×\times speedup on GPU and maintains the same PSNR on BSD100.

On high-resolution SR, the advantage of EfficientViT over previous ViT-based SR methods becomes more significant. Compared with Restormer, EfficientViT achieves up to 6.4×\times speedup on GPU and provides 0.11dB gain in PSNR on FFHQ.

|  | Params ↓\downarrow | MACs ↓\downarrow | Throughput ↑\uparrow | COCO | LVIS |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|
|  | A100(image/s) | mAP\rm mAP | APS\rm AP^{S} | APM\rm AP^{M} | APL\rm AP^{L} | mAP\rm mAP | APS\rm AP^{S} | APM\rm AP^{M} | APL\rm AP^{L} |  |  |
| SAM-ViT-H [13] | 641M | 2973G | 11 | 46.5 | 30.8 | 51.0 | 61.7 | 44.2 | 31.8 | 57.1 | 65.3 |
| EfficientViT-SAM-L0 | 35M | 35G | 762 | 45.7 | 28.2 | 49.5 | 63.4 | 41.8 | 28.8 | 53.4 | 64.7 |
| EfficientViT-SAM-L1 | 48M | 49G | 638 | 46.2 | 28.7 | 50.4 | 64.0 | 42.1 | 29.1 | 54.3 | 65.0 |
| EfficientViT-SAM-L2 | 61M | 69G | 538 | 46.6 | 28.9 | 50.8 | 64.2 | 42.7 | 29.4 | 55.1 | 65.5 |
| EfficientViT-SAM-XL0 | 117M | 185G | 278 | 47.5 | 30.0 | 51.5 | 64.6 | 43.9 | 31.2 | 56.2 | 65.9 |
| EfficientViT-SAM-XL1 | 203M | 322G | 182 | 47.8 | 30.5 | 51.8 | 64.7 | 44.4 | 31.6 | 57.0 | 66.4 |

|  | COCO | LVIS |  |  |  |  |
|---|---|---|---|---|---|---|
|  | 1 click | 3 click | 5 click | 1 click | 3 click | 5 click |
| SAM-ViT-H [13] | 58.4 | 69.6 | 71.4 | 59.2 | 66.0 | 66.8 |
| EfficientViT-SAM-XL1 | 59.8 | 71.3 | 75.3 | 56.6 | 67.0 | 71.7 |

### 3.5 Segment Anything

We build EfficientViT-SAM, a new family of accelerated segment anything models, by leveraging EfficientViT to replace SAM’s image encoder. Meanwhile, we retain SAM’s lightweight prompt encoder and mask decoder. The training process consists of two phases. First, we train the image encoder of EfficientViT-SAM using SAM’s image encoder as the teacher. Second, we train EfficientViT-SAM end-to-end using the whole SA-1B dataset [13].

We thoroughly test EfficientViT-SAM on various zero-shot benchmarks to verify its effectiveness. Table 6 demonstrates the zero-shot instance segmentation results on COCO [38] and LVIS [39], prompted with the predicted bounding boxes from ViTDet [40]. EfficientViT-SAM provides superior performance/efficiency compared with SAM-ViT-H [13]. In particular, EfficientViT-SAM-XL1 outperforms SAM-ViT-H on COCO and LVIS while having 16.5×\times higher throughput on A100 GPU.

Figure 7 shows the comparison between EfficientViT-SAM and prior SAM models. EfficientViT-SAM is the first accelerated SAM model that matches/outperforms SAM-ViT-H’s [13] zero-shot performance, delivering the SOTA performance-efficiency trade-off.

Apart from box-prompted instance segmentation, we also evaluate EfficientViT-SAM on point-prompted segmentation. The results are summarized in Table 7. EfficientViT-SAM-XL1 outperforms SAM-ViT-H in most cases, especially when more points are given. On LVIS, when given a single point, we find SAM-ViT-H performs better than EfficientViT-SAM-XL1. This might be because we do not have the interactive segmentation setup during the end-to-end training phase. Further investigation is needed to improve the performance of the single-point setting.

## 4 Related Work

Dense prediction targets producing predictions for each pixel given the input image. It can be viewed as an extension of image classification from per-image prediction to per-pixel predictions. Extensive studies have been done to improve the performance of CNN-based high-resolution dense prediction models [1, 2, 3, 4, 5, 6].

In addition, there are also some works targeting improving the efficiency of high-resolution dense prediction models [41, 42, 43, 44]. While these models provide good efficiency, their performances are far behind SOTA high-resolution dense prediction models.

Compared to these works, our models provide a better trade-off between performance and efficiency by enabling a global receptive field and multi-scale learning with lightweight operations.

While ViT provides impressive performances in the high-computation region, it is usually inferior to previous efficient CNNs [45, 46, 47, 48] when targeting the low-computation region. To close the gap, MobileViT [49] proposes to combine the strength of CNN and ViT by replacing local processing in convolutions with global processing using transformers. MobileFormer [50] proposes to parallelize MobileNet and Transformer with a two-way bridge in between for feature fusing. NASViT [51] proposes to leverage neural architecture search to search for efficient ViT architectures.

However, these models mainly focus on image classification and still rely on softmax attention with quadratic computational complexity, thus unsuitable for high-resolution dense prediction.

Our work is also related to efficient deep learning, which aims at improving the efficiency of deep neural networks so that we can deploy them on hardware platforms with limited resources, such as mobile phones and IoT devices. Typical technologies in efficient deep learning include network pruning [52, 53, 54], quantization [55], efficient model architecture design [18, 56], and training techniques [57, 58, 59]. In addition to manual designs, many recent works use AutoML techniques [60, 61, 62] to automatically design [47], prune [63] and quantize [64] neural networks.

## 5 Conclusion

In this work, we studied efficient architecture design for high-resolution dense prediction. We introduced a lightweight multi-scale attention module that simultaneously achieves a global receptive field, and multi-scale learning with lightweight and hardware-efficient operations, thus providing significant speedup on diverse hardware devices without performance loss than SOTA high-resolution dense prediction models. For future work, we will explore applying EfficientViT to other vision tasks and further scaling up our EfficientViT models.

## Acknowledgments

We thank MIT-IBM Watson AI Lab, MIT AI Hardware Program, Amazon and MIT Science Hub, Qualcomm Innovation Fellowship, National Science Foundation for supporting this research.
