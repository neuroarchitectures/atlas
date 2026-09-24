# YOLOv12: Attention-Centric Real-Time Object Detectors

Enhancing the network architecture of the YOLO framework has been crucial for a long time, but has focused on CNN-based improvements despite the proven superiority of attention mechanisms in modeling capabilities. This is because attention-based models cannot match the speed of CNN-based models. This paper proposes an attention-centric YOLO framework, namely YOLOv12, that matches the speed of previous CNN-based ones while harnessing the performance benefits of attention mechanisms.

YOLOv12 surpasses all popular real-time object detectors in accuracy with competitive speed. For example, YOLOv12-N achieves 40.6%40.6\% mAP with an inference latency of 1.641.64 ms on a T4 GPU, outperforming advanced YOLOv10-N / YOLOv11-N by 2.1%/1.2%2.1\%/1.2\% mAP with a comparable speed. This advantage extends to other model scales. YOLOv12 also surpasses end-to-end real-time detectors that improve DETR, such as RT-DETR / RT-DETRv2: YOLOv12-S beats RT-DETR-R18 / RT-DETRv2-R18 while running 42%42\% faster, using only 36%36\% of the computation and 45%45\% of the parameters. More comparisons are shown in Figure 1.

## 1 Introduction

Real-time object detection has consistently attracted significant attention due to its low-latency characteristics, which provide substantial practicality [24, 28, 4, 17]. Among them, the YOLO series [45, 47, 46, 3, 29, 32, 57, 24, 58, 53, 28] has effectively established an optimal balance between latency and accuracy, thus dominating the field. Although improvements in YOLO have focused on areas such as loss functions [8, 68, 44, 43, 67, 35, 48], label assignment [23, 34, 59, 22, 69], network architecture design has remained a critical research priority [32, 57, 24, 58, 28]. Although attention-centric vision transformer (ViT) architectures have been proven to possess stronger modeling capabilities, even in small models [25, 20, 21, 50], most architectural designs continue to focus primarily on CNNs.

The primary reason for this situation lies in the inefficiency of the attention mechanism, which comes from two main factors: quadratic computational complexity and inefficient memory access operations of the attention mechanism (the latter being the main issue addressed by FlashAttention [14, 13]). As a result, under a similar computational budget, CNN-based architectures outperform attention-based ones by a factor of ∼3×\sim 3\times [38], which significantly limits the adoption of attention mechanisms in YOLO systems where high inference speed is critical.

This paper aims to address these challenges and further builds an attention-centric YOLO framework, namely YOLOv12. We introduce three key improvements. First, we propose a simple yet efficient area attention module (A2), which maintains a large receptive field while reducing the computational complexity of attention in a very simple way, thus enhancing speed. Second, we introduce the residual efficient layer aggregation networks (R-ELAN) to address the optimization challenges introduced by attention (primarily large-scale models). R-ELAN introduces two improvements based on the original ELAN [57]: (i) a block-level residual design with scaling techniques and (ii) a redesigned feature aggregation method. Third, we make some architectural improvements beyond the vanilla attention to fit the YOLO system. We upgrade traditional attention-centric architectures including: introducing FlashAttention to conquer the memory access issue of attention, removing designs such as positional encoding to make the model fast and clean, adjusting the MLP ratio from 44 to 1.21.2 to balance the computation between attention and feed forward network for better performance, reducing the depth of stacked blocks to facilitate optimization, and making use of convolution operators as much as possible to leverage their computational efficiency.

Based on the designs outlined above, we develop a new family of real-time detectors with 55 model scales: YOLOv12-N, S, M, L, and X. We perform extensive experiments on standard object detection benchmarks following YOLOv11 [28] without any additional tricks, demonstrating that YOLOv12 provides significant improvements over previous popular models in terms of latency-accuracy and FLOPs-accuracy trade-offs across these scales, as illustrated in Figure 1. For example, YOLOv12-N achieves 40.6%40.6\% mAP, outperforming YOLOv10-N [53] by 2.1%2.1\% mAP while maintaining a faster inference speed, and YOLOv11-N [28] by 1.2%1.2\% mAP with a comparable speed. This advantage remains consistent across other scale models. Compared to RT-DETR-R18 [66] / RT-DETRv2-R18 [40], YOLOv12-S is 1.5%/0.1%1.5\%/0.1\% mAP better, while reports 42%/42%42\%/42\% faster latency speed, requiring only 36%/36%36\%/36\% of their computations and 45%/45%45\%/45\% of their parameters.

In summary, the contributions of YOLOv12 are two-fold: 1) it establishes an attention-centric, simple yet efficient YOLO framework that, through methodological innovation and architectural improvements, breaks the dominance of CNN models in YOLO series. 2) without relying on additional techniques such as pretraining, YOLOv12 achieves state-of-the-art results with fast inference speed and higher detection accuracy, demonstrating its potential.

## 2 Related Work

Real-time Object Detectors. Real-time object detectors have consistently attracted the community’s attention due to their significant practical value. The YOLO series [45, 47, 46, 3, 29, 32, 57, 54, 9, 24, 58, 53, 28] has emerged as a leading framework for real-time object detection. The early YOLO systems [45, 47, 46] establish the framework for the YOLO series, primarily from a model design perspective. YOLOv4 [3] and YOLOv5 [29] add CSPNet [55], data augmentation, and multiple feature scales to the framework. YOLOv6 [32] further advance these with BiC and SimCSPSPPF modules for the backbone and neck, alongside anchor-aided training. YOLOv7 [57] introduce E-ELAN [56] (efficient layer aggregation networks) for improved gradient flow and various bag-of-freebies, while YOLOv8 [24] integrate a efficient C2f block for enhanced feature extraction. In recent iterations, YOLOv9 [58] introduce GELAN for architecture optimization and PGI for training improvement, while YOLOv10 [53] apply NMS-free training with dual assignments for efficiency gains. YOLOv11 [28] further reduces latency and increases accuracy by adopting the C3K2 module (a specification of GELAN [58]) and lightweight depthwise separable convolution in the detection head. Recently, an end-to-end object detection method, namely RT-DETR [66], improved traditional end-to-end detectors [7, 37, 71, 42, 33] to meet real-time requirements by designing an efficient encoder and an uncertainty-minimal query selection mechanism. RT-DETRv2 [40] further enhances it with bag-of-freebies. Unlike previous YOLO series, this study aims to build a YOLO framework centered around attention to leverage the superiority of the attention mechanism.

Efficient Vision Transformers. Reducing computational costs from global self-attention is crucial to effectively applying vision transformers in downstream tasks. PVT [61] addresses this using multi-resolution stages and downsampling features. Swin Transformer [39] limits self-attention to local windows and adjusts the window partitioning style to connect non-overlapping windows, balancing communication needs with memory and computation demands. Other methods, such as axial self-attention [26] and criss-cross attention [27], calculate attention within horizontal and vertical windows. CSWin transformer [16] builds on this by introducing cross-shaped window self-attention, computing attention along horizontal and vertical stripes in parallel. In addition, local-global relations are established in works such as [12, 64], improving efficiency by reducing reliance on global self-attention. Fast-iTPN [50] improves downstream task inference speed with token migration and token gathering mechanisms. Some approaches [49, 60, 31, 62] use linear attention to decrease the complexity of the attention. Although Mamba-based vision models [70, 38] aim for linear complexity, they still fall short of real-time speeds [38]. FlashAttention [14, 13] identifies high bandwidth memory bottlenecks that lead to inefficient attention computation and addresses them through I/O optimization, reducing memory access to enhance computational efficiency. In this study, we discard complex designs and propose a simple area attention mechanism to reduce the complexity of attention. Additionally, we employ FlashAttention to overcome inherent memory accessing problems of the attention mechanism [14, 13].

## 3 Approach

This section introduces YOLOv12, an innovation in the YOLO framework from the perspective of network architecture with attention mechanism.

### 3.1 Efficiency Analysis

The attention mechanism, while highly effective in capturing global dependencies and facilitating tasks such as natural language processing [5, 15] and computer vision [19, 39], is inherently slower than convolution neural networks (CNN). Two primary factors contribute to this discrepancy in speed.

Complexity. First, the computational complexity of the self-attention operation scales quadratically with the input sequence length LL. Specifically, for an input sequence with length LL and feature dimension dd, the computation of the attention matrix requires 𝒪⁡(L2​d)\mathcal{O}(L^{2}d) operations, since each token attends to every other token. In contrast, the complexity of convolution operations in CNNs scales linearly with respect to the spatial or temporal dimension, i.e., 𝒪⁡(k​L​d)\mathcal{O}(kLd), where kk is the kernel size and is typically much smaller than LL. As a result, self-attention becomes computationally prohibitive, especially for large inputs such as high-resolution images or long sequences.

Moreover, another significant factor is that most attention-based vision transformers, due to their complex designs (e.g., window partitioning/reversing in Swin transformer [39]) and the introduction of additional modules (e.g., positional encoding), gradually accumulate speed overhead, resulting in an overall slower speed compared to CNN architectures [38]. In this paper, the design modules utilize simple and clean operations to implement attention, ensuring efficiency to the greatest extent.

Computation. Second, in the attention computation process, memory access patterns are less efficient compared to CNNs [14, 13]. Specifically, during self-attention, intermediate maps such as the attention map (Q​KT)(QK^{T}) and the softmax map (L×LL\times L) need to be stored from high-speed GPU SRAM (the actual location of the computation) to high bandwidth GPU memory (HBM) and later retrieved during the computation, and the read and write speed of the former is more than 1010 times that of the latter, thus resulting in significant memory accessing overhead and increased wall-clock time11 1 This problem has been solved by FlashAttention [14, 13], which will be directly adopted during model design. Additionally, irregular memory access patterns in attention introduce further latency compared to CNNs, which utilize structured and localized memory access. CNNs benefit from spatially constrained kernels, enabling efficient memory caching and reduced latency due to their fixed receptive fields and sliding-window operations.

These two factors, quadratic computational complexity and inefficient memory accessing, together render attention mechanisms slower than CNNs, particularly in real-time or resource-constrained scenarios. Addressing these limitations has become a critical area of research, with approaches such as sparse attention mechanisms and memory-efficient approximations (e.g., Linformer [60] or Performer [11]) aiming to mitigate the quadratic scaling.

| Method | FLOPs | #Param. | APv​a​l50:95\text{AP}^{val}_{50:95} | AP50v​a​l\text{AP}^{val}_{50} | AP75v​a​l\text{AP}^{val}_{75} | Latency |
|---|---|---|---|---|---|---|
| (G) | (M) | (%) | (%) | (%) | (ms) |  |
| YOLOv6-3.0-N [32] | 11.4 | 4.7 | 37.0 | 52.7 | – | 2.69 |
| Gold-YOLO-N [54] | 12.1 | 5.6 | 39.6 | 55.7 | – | 2.92 |
| YOLOv8-N [24] | 8.7 | 3.2 | 37.4 | 52.6 | 40.5 | 1.77 |
| YOLOv10-N [53] | 6.7 | 2.3 | 38.5 | 53.8 | 41.7 | 1.84 |
| YOLO11-N [28] | 6.5 | 2.6 | 39.4 | 55.3 | 42.8 | 1.5 |
| YOLOv12-N (Ours) | 6.5 | 2.6 | 40.6 | 56.7 | 43.8 | 1.64 |
| YOLOv6-3.0-S [32] | 45.3 | 18.5 | 44.3 | 61.2 | – | 3.42 |
| Gold-YOLO-S [54] | 46.0 | 21.5 | 45.4 | 62.5 | – | 3.82 |
| YOLOv8-S [24] | 28.6 | 11.2 | 45.0 | 61.8 | 48.7 | 2.33 |
| RT-DETR-R18 [66] | 60.0 | 20.0 | 46.5 | 63.8 | – | 4.58 |
| RT-DETRv2-R18 [41] | 60.0 | 20.0 | 47.9 | 64.9 | – | 4.58 |
| YOLOv9-S [58] | 26.4 | 7.1 | 46.8 | 63.4 | 50.7 | – |
| YOLOv10-S [53] | 21.6 | 7.2 | 46.3 | 63.0 | 50.4 | 2.49 |
| YOLO11-S [28] | 21.5 | 9.4 | 46.9 | 63.9 | 50.6 | 2.5 |
| YOLOv12-S (Ours) | 21.4 | 9.3 | 48.0 | 65.0 | 51.8 | 2.61 |
| YOLOv6-3.0-M [32] | 85.8 | 34.9 | 49.1 | 66.1 | – | 5.63 |
| Gold-YOLO-M [54] | 87.5 | 41.3 | 49.8 | 67.0 | – | 6.38 |
| YOLOv8-M [24] | 78.9 | 25.9 | 50.3 | 67.2 | 54.7 | 5.09 |
| RT-DETR-R34 [66] | 100.0 | 36.0 | 48.9 | 66.8 | – | 6.32 |
| RT-DETRv2-R34 [41] | 100.0 | 36.0 | 49.9 | 67.5 | – | 6.32 |
| YOLOv9-M [58] | 76.3 | 20.0 | 51.4 | 68.1 | 56.1 | – |
| YOLOv10-M [53] | 59.1 | 15.4 | 51.1 | 68.1 | 55.8 | 4.74 |
| YOLO11-M [28] | 68.0 | 20.1 | 51.5 | 68.5 | 55.7 | 4.7 |
| YOLOv12-M (Ours) | 67.5 | 20.2 | 52.5 | 69.6 | 57.1 | 4.86 |
| YOLOv6-3.0-L [32] | 150.7 | 59.6 | 51.8 | 69.2 | – | 9.02 |
| Gold-YOLO-L [54] | 151.7 | 75.1 | 51.8 | 68.9 | – | 10.65 |
| YOLOv8-L [24] | 165.2 | 43.7 | 53.0 | 69.8 | 57.7 | 8.06 |
| RT-DETR-R50 [66] | 136.0 | 42.0 | 53.1 | 71.3 | – | 6.90 |
| RT-DETRv2-R50 [41] | 136.0 | 42.0 | 53.4 | 71.6 | – | 6.90 |
| YOLOv9-C [58] | 102.1 | 25.3 | 53.0 | 70.2 | 57.8 | – |
| YOLOv10-B [53] | 92.0 | 19.1 | 52.5 | 69.6 | 57.2 | 5.74 |
| YOLOv10-L [53] | 120.3 | 24.4 | 53.2 | 70.1 | 58.1 | 7.28 |
| YOLO11-L [28] | 86.9 | 25.3 | 53.3 | 70.1 | 58.2 | 6.2 |
| YOLOv12-L (Ours) | 88.9 | 26.4 | 53.7 | 70.7 | 58.5 | 6.77 |
| YOLOv8-X [24] | 257.8 | 68.2 | 54.0 | 71.0 | 58.8 | 12.83 |
| RT-DETR-R101 [66] | 259.0 | 76.0 | 54.3 | 72.7 | – | 13.5 |
| RT-DETRv2-R101 [41] | 259.0 | 76.0 | 54.3 | 72.8 | – | 13.5 |
| YOLOv10-X [53] | 160.4 | 29.5 | 54.4 | 71.3 | 59.3 | 10.70 |
| YOLO11-X [28] | 194.9 | 56.9 | 54.6 | 71.6 | 59.5 | 11.3 |
| YOLOv12-X (Ours) | 199.0 | 59.1 | 55.2 | 72.0 | 60.2 | 11.79 |

### 3.2 Area Attention

A simple approach to reduce the computational cost of vanilla attention is to use the linear attention mechanism [49, 60], which reduces the complexity of vanilla attention from quadratic to linear. For a visual feature ff with dimensions (n,h,d)(n,h,d), where nn is the number of tokens, hh is the number of heads and dd is the head size, linear attention reduces the complexity from 2​n2​h​d2n^{2}hd to 2​n​h​d22nhd^{2}, decreasing the computational cost since n>dn>d. However, linear attention suffers from global dependency degradation [30], instability [11], and distribution sensitivity [63]. Furthermore, due to the low-rank bottleneck [10, 2], it offers only limited speed advantages when applied to YOLO with input resolution of 640×640640\times 640.

An alternative approach to effectively reduce complexity is the local attention mechanism (e.g., Shift window [39], criss-cross attention [27], and axial attention [16]), as shown in Figure 2, which transforms global attention into local, thus reducing computational costs. However, partitioning the feature map into windows can introduce overhead or reduce the receptive field, impacting both speed and accuracy. In this study, we propose the simple yet efficient area attention module. As illustrated in Figure 2, the feature map with the resolution of (H,W)(H,W) is divided into ll segments of size (Hl,W)(\frac{H}{l},W) or (H,Wl)(H,\frac{W}{l}). This eliminates explicit window partitioning, requiring only a simple reshape operation, and thus achieves faster speed. We empirically set the default value of ll to 44, reducing the receptive field to 14\frac{1}{4} of the original, yet it still maintains a large receptive field. With this approach, the computational cost of the attention mechanism is reduced from 2​n2​h​d2n^{2}hd to 12​n2​h​d\frac{1}{2}n^{2}hd. We show that despite the complexity n2n^{2}, this is still efficient enough to meet the real-time requirements of the YOLO system when nn is fixed at 640640 (it increases nn if the input resolution increases). Interestingly, we find that this modification has only a slight impact on performance but significantly improves speed.

### 3.3 Residual Efficient Layer Aggregation Networks

Efficient layer aggregation networks (ELAN) [57] are designed to improve feature aggregation. As shown in Figure 3 (b), ELAN splits the output of a transition layer (a 1×11\times 1 convolution), processes one split through multiple modules, then concatenates the all the outputs and applies another transition layer (a 1×11\times 1 convolution) to align dimensions. However, as analyzed by [57], this architecture can introduce instability. We argue that such a design causes gradient blocking and lacks residual connections from input to output. Furthermore, we build the network around the attention mechanism, which presents additional optimization challenges. Empirically, L- and X-scale models either fail to converge or remain unstable, even when using Adam or AdamW optimizers.

To address this problem, we propose residual efficient layer aggregation networks (R-ELAN), Figure 3 (d). In contrast, we introduce a residual shortcut from input to output throughout the block with a scaling factor (default to 0.010.01). This design is similar to layer scaling [52], which is introduced to build a deep vision transformer. However, applying layer scaling for each area attention will not conquer the optimization challenge and introduce slowdown on latency. This shows that the introduction of the attention mechanism is not the only reason for convergence but the ELAN architecture itself, which verifies the rationale behind our R-ELAN design.

We also design a new aggregation approach as shown in Figure 3 (d). The original ELAN layer processes the input of the module by first passing it through a transition layer, which then splits it into two parts. One part is further processed by subsequent blocks, and finally both parts are concatenated to produce the output. In contrast, our design applies a transition layer to adjust the channel dimensions and produces a single feature map. This feature map is then processed through subsequent blocks followed by the concatenation, forming a bottleneck structure. This approach not only preserves the original feature integration capability, but also reduces both computational cost and parameter / memory usage.

### 3.4 Architectural Improvements

In this section, we will introduce the overall architecture and some improvements over the vanilla attention mechanism. Some of them are not initially proposed by us.

Many attention-centric vision transformers are designed with the plain-style architectures [18, 51, 1, 25, 19, 21], while we retain the hierarchical design of the previous YOLO systems [45, 47, 46, 3, 29, 32, 57, 24, 58, 53, 28] and will demonstrate the necessity of this. We remove the design of stacking three blocks in the last stage of the backbone, which is present in recent versions [24, 58, 53, 28]. Instead, we retain only a single R-ELAN block, reducing the total number of blocks and contributing to optimization. We inherit the first two stages of the backbone from YOLOv11 [28] and do not use the proposed R-ELAN.

Additionally, we modify several default configurations in the vanilla attention mechanism to better suit the YOLO system. These modifications include adjusting the MLP ratio from 44 to 1.21.2 (or 22 for the N- / S- / M-scale models) is used to better allocate computational resources for better performance, adopting nn.Conv2d+BN instead of nn.Linear+LN to fully exploit the efficiency of convolution operators, removing positional encoding, and introduce a large separable convolution (7×77\times 7) (namely position perceiver) to help the area attention perceive position information. The effectiveness of these modifications will be validated in Section 4.5.

## 4 Experiment

| Model | Vanilla | Re-Aggre. | Resi. | Scaling | Convergence | FLOPs (G) | #Param. (M) | APv​a​l50:95\text{AP}^{val}_{50:95} (%) |
|---|---|---|---|---|---|---|---|---|
| YOLOv12-N | ✔ | ✗ | ✗ | – | ✔ | 6.9 | 2.7 | 40.8 |
| ✗ | ✔ | ✗ | – | ✔ | 6.5 | 2.6 | 40.6 |  |
| ✗ | ✗ | ✔ | 0.1 | ✔ | 6.5 | 2.6 | 40.3 |  |
| YOLOv12-L | ✔ | ✗ | ✗ | – | ✗ | – | – | – |
| ✗ | ✔ | ✗ | – | ✗ | – | – | – |  |
| ✗ | ✔ | ✔ | 0.1 | ✔ | 88.9 | 26.4 | 53.3 |  |
| ✗ | ✔ | ✔ | 0.01 | ✔ | 88.9 | 26.4 | 53.7 |  |
| ✗ | ✗ | ✔ | 0.01 | ✔ | 94.3 | 27.8 | 53.8 |  |
| YOLOv12-X | ✔ | ✗ | ✗ | – | ✗ | – | – | – |
| ✗ | ✔ | ✗ | – | ✗ | – | – | – |  |
| ✗ | ✔ | ✔ | 0.1 | ✗ | – | – | – |  |
| ✗ | ✔ | ✔ | 0.01 | ✔ | 199.0 | 59.1 | 55.2 |  |
| ✗ | ✗ | ✔ | 0.01 | ✔ | 211.3 | 62.3 | 55.3 |  |

This section is divided into four parts: experimental setup, a systematic comparison with popular methods, ablation studies to validate our approach, and an analysis with visualizations to further explore YOLOv12.

### 4.1 Experimental Setup

We validate the proposed method on the MSCOCO 2017 dataset [36]. The YOLOv12 family includes 55 variants: YOLOv12-N, YOLOv12-S, YOLOv12-M, YOLOv12-L, and YOLOv12-X. All models are trained for 600600 epochs using the SGD optimizer with an initial learning rate of 0.010.01, which remains the same as YOLOv11 [28]. We adopt a linear learning rate decay schedule and perform a linear warm-up for the first 33 epochs. Following the approach in [53, 66], the latencies of all models are tested on a T4 GPU with TensorRT FP16.

Baseline. We choose the previous version of YOLOv11 [28] as our baseline. The model scaling strategy is also consistent with it. We use several of its proposed C3K2 blocks (that is a special case of GELAN [58]). We do not use any more tricks beyond YOLOv11 [28].

| Model | Area Attention | CUDA | CPU |  |
|---|---|---|---|---|
| FP32 | FP16 |  |  |  |
| YOLOv12-N | ✗ | 2.7/2.5 | 1.5/1.5 | 62.9 |
| ✔ | 2.0/2.0 | 1.3/1.3 | 31.4 |  |
| YOLOv12-S | ✗ | 5.1/4.4 | 2.5/2.2 | 130.0 |
| ✔ | 3.5/3.1 | 1.7/1.7 | 78.4 |  |
| YOLOv12-X | ✗ | 26.4/21.9 | 11.1/10.4 | 804.2 |
| ✔ | 18.2/14.3 | 7.1/6.7 | 512.5 |  |

| Model | Scale | FLOPs | RTX 3080 | A5000 | A6000 |
|---|---|---|---|---|---|
| (G) |  |  |  |  |  |
| YOLOv9 [58] | T | 8.2 | 2.4/1.5 | 2.4/1.6 | 2.3/1.7 |
| S | 26.4 | 3.7/1.9 | 3.4/2.0 | 3.5/1.9 |  |
| M | 76.3 | 6.5/2.8 | 5.5/2.6 | 5.2/2.6 |  |
| C | 102.1 | 8.0/2.9 | 6.4/2.7 | 6.0/2.7 |  |
| E | 189.0 | 17.2/6.7 | 14.2/6.3 | 13.1/5.9 |  |
| YOLOv10 [53] | N | 6.7 | 1.6/1.0 | 1.6/1.0 | 1.6/1.0 |
| S | 21.6 | 2.8/1.4 | 2.4/1.4 | 2.4/1.3 |  |
| M | 59.1 | 5.7/2.5 | 4.5/2.4 | 4.2/2.2 |  |
| B | 92.0 | 6.8/2.9 | 5.5/2.6 | 5.2/2.8 |  |
| YOLOv11 [28] | N | 6.5 | 1.6/1.0 | 1.6/1.0 | 1.5/0.9 |
| S | 21.5 | 2.8/1.3 | 2.4/1.4 | 2.4/1.3 |  |
| M | 68.0 | 5.6/2.3 | 4.5/2.2 | 4.4/2.1 |  |
| L | 86.9 | 7.4/3.0 | 5.9/2.7 | 5.8/2.7 |  |
| X | 194.9 | 15.2/5.3 | 10.7/4.7 | 9.1/4.0 |  |
| YOLOv12 | N | 6.5 | 1.7/1.1 | 1.7/1.0 | 1.7/1.1 |
| S | 21.4 | 2.9/1.5 | 2.5/1.5 | 2.5/1.4 |  |
| M | 67.5 | 5.8/1.5 | 4.6/2.4 | 4.4/2.2 |  |
| L | 88.9 | 7.9/3.3 | 6.2/3.1 | 6.0/3.0 |  |
| X | 199.0 | 15.6/5.6 | 11.0/5.2 | 9.5/4.4 |  |

### 4.2 Comparison with State-of-the-arts

We present a performance comparison between YOLOv12 and other popular real-time detectors in Table 1.

For N-scale models, YOLOv12-N outperforms YOLOv6-3.0-N [32], YOLOv8-N [58], YOLOv10-N [53], and YOLOv11 [28] by 3.6%3.6\%, 3.3%3.3\%, 2.1%2.1\%, and 1.2%1.2\% in mAP, respectively, while maintaining similar or even fewer computations and parameters, and achieving the fast latency speed of 1.641.64 ms/image.

For S-scale models, YOLOv12-S, with 21.421.4G FLOPs and 9.39.3M parameters, achieves 48.048.0 mAP with 2.612.61 ms/image latency. It outperforms YOLOv8-S [24], YOLOv9-S [58], YOLOv10-S [53], and YOLOv11-S [28] by 3.0%3.0\%, 1.2%1.2\%, 1.7%1.7\%, and 1.1%1.1\%, respectively, while maintaining similar or fewer computations. Compared to the end-to-end detectors RT-DETR-R18 [66] / RT-DETRv2-R18 [41], YOLOv12-S achieves beatable performance but with much better inference speed and less computational cost and fewer parameters.

For M-scale models, YOLOv12-M, with 67.567.5G FLOPs and 20.220.2M parameters, achieves 52.552.5 mAP performance and 4.864.86 ms/image speed. Compared to Gold-YOLO-M [54], YOLOv8-M [24], YOLOv9-M [58], YOLOv10 [53], YOLOv11 [28], and RT-DETR-R34 [66] / RT-DETRv2-R34 [40], YOLOv12-S enjoys superiority.

For L-scale models, YOLOv12-L even surpasses YOLOv10-L [53] with 31.431.4G fewer FLOPs. YOLOv12-L beats YOLOv11 [28] by 0.4%0.4\% mAP with comparable FLOPs and parameters. YOLOv12-L also outperforms RT-DERT-R50 [66] / RT-DERTv2-R50 [41] with faster speed, fewer FLOPs (34.6%34.6\%) and fewer parameters (37.1%37.1\%).

For X-scale models, YOLOv12-X significantly outperforms YOLOv10-X [53] / YOLOv11-X [28] by 0.8%0.8\% and 0.6%0.6\%, respectively, with comparable speed, FLOPs, and parameters. YOLOv12-X again beats RT-DETR-R101 [66] / RT-DETRv2-R101 [40] with faster speed, fewer FLOPs (23.4%23.4\%) and fewer parameters (22.2%22.2\%).

In particular, if the L- / X-scale models are evaluated using FP32 precision (which requires saving the model separately in FP32 format), YOLOv12 will achieve an improvement of ∼\sim 0.2%0.2\% mAP. This means that YOLOv12-L/X will report 33.9%33.9\%/55.4%55.4\% mAP.

| Method | APv​a​l50:95\text{AP}^{val}_{50:95} | Latency |
|---|---|---|
| Linear+LN{}_{+\text{LN}} | 40.5 | 1.68 |
| Linear+BN{}_{+\text{BN}} | 39.5 | 1.70 |
| Conv+BN{}_{+\text{BN}} | 40.6 | 1.64 |
| Conv+LN{}_{+\text{LN}} | 40.3 | 1.66 |

| Method | APv​a​l50:95\text{AP}^{val}_{50:95} | Latency |
|---|---|---|
| N/A | 38.3 | 1.60 |
| S1 | 40.1 | 1.63 |
| S4 | 39.8 | 1.71 |
| Ours | 40.6 | 1.64 |

| Epochs | APv​a​l\text{AP}^{val} (N) | APv​a​l\text{AP}^{val} (S) |
|---|---|---|
| 300 | 39.5 | 47.0 |
| 500 | 40.3 | 47.8 |
| 600 | 40.6 | 48.0 |
| 800 | 40.4 | 47.7 |

| kernel | APv​a​l50:95\text{AP}^{val}_{50:95} | Latency |
|---|---|---|
| 3×33\times 3 | 40.4 | 1.60 |
| 5×55\times 5 | 40.4 | 1.61 |
| 7×77\times 7 | 40.6 | 1.64 |
| 9×99\times 9 | 40.7 | 1.79 |

| Pos. | APv​a​l50:95\text{AP}^{val}_{50:95} | Latency |
|---|---|---|
| RPE | 40.3 | 1.76 |
| APE | 40.5 | 1.69 |
| N/A | 40.6 | 1.64 |

| Area | APv​a​l50:95\text{AP}^{val}_{50:95} | Latency |
|---|---|---|
| ✗ | 40.8 | 1.70 |
| ✔ | 40.6 | 1.64 |

| Ratio (L) | APv​a​l50:95\text{AP}^{val}_{50:95} | Latency |
|---|---|---|
| 1.2 | 53.8 | 6.77 |
| 2.0 | 53.6 | 6.75 |
| 4.0 | 53.1 | 6.68 |

| FA | Latency (N) | Latency (S) |
|---|---|---|
| ✗ | 1.92 | 3.02 |
| ✔ | 1.64 | 2.61 |

### 4.3 Ablation Studies

∙\bullet R-ELAN. Table 2 evaluates the effectiveness of the proposed residual efficient layer networks (R-ELAN) using YOLOv12-N/L/X models. The results reveal two key findings: (i) For small models like YOLOv12-N, residual connections do not impact convergence but degrade performance. In contrast, for larger models (YOLOv12-L/X), they are essential for stable training. In particular, YOLOv12-X requires a minimal scaling factor (0.010.01) to ensure convergence. (ii) The proposed feature integration method effectively reduces the complexity of the model in terms of FLOPs and parameters while maintaining comparable performance with only a marginal decrease.

∙\bullet Area Attention. We conduct ablation experiments to validate the effectiveness of area attention, with results presented in Table 3. Evaluations are performed on YOLOv12-N/S/X models, measuring the inference speed on both the GPU (CUDA) and CPU. CUDA results are obtained using RTX 3080 and A5000, while CPU performance is measured on an Intel Core i7-10700K @ 3.80GHz. The results demonstrate a significant speedup with area attention (✔). For example, with FP32 on RTX 3080, YOLOv12-N achieves a 0.70.7ms reduction in inference time. This performance gain is consistently observed across different models and hardware configurations. We do not use FlashAttention [14, 13] in this experiment because it would significantly reduce the speed difference.

### 4.4 Speed Comparison

Table 4 presents a comparative analysis of inference speed across different GPUs, evaluating YOLOv9 [58], YOLOv10 [53], YOLOv11 [28], and our YOLOv12 on RTX 3080, RTX A5000, and RTX A6000 with FP32 and FP16 precision. To ensure consistency, all results are obtained on the same hardware and YOLOv9 [58] and YOLOv10 [53] are evaluated using the integrated codebase of ultralytics [28]. The results indicate that YOLOv12 achieves a significantly higher inference speed than YOLOv9 [58] while remaining on par with YOLOv10 [53] and YOLOv11 [28]. For example, on RTX 3080, YOLOv9 reports 2.42.4 ms (FP32) and 1.51.5 ms (FP16), while YOLOv12-N achieves 1.71.7 ms (FP32) and 1.11.1 ms (FP16). Similar trends hold across other configurations.

Figure 4 presents additional comparisons. The left subfigure illustrates the accuracy-parameter trade-off comparison with popular methods, where YOLOv12 establishes a dominant boundary beyond the counterparts, surpassing even YOLOv10, A YOLO version characterized by significantly fewer parameters, showcasing the efficacy of YOLOv12. We compare the inference latency of YOLOv12 with previous YOLO versions on a CPU in the right subfigure (All results were measured on an Intel Core i7-10700K @ 3.80GHz). As shown, YOLOv12 surpasses other competitors with a more advantageous boundary, highlighting its efficiency across diverse hardware platforms.

### 4.5 Diagnosis & Visualization

We diagnose the YOLOv12 designs in Tables 5(a) to 5(h). Unless otherwise specified, we perform these diagnostics on YOLOv12-N, with a default training of 600600 epochs from scratch.

∙\bullet Attention Implementation: Table 5(a). We examine two approaches to implementing attention. The convolution-based approach is faster than the linear-based approach due to the computational efficiency of convolution. Additionally, we explore two normalization methods (layer normalization (LN) and batch normalization (BN)) and find the result: although layer normalization is commonly used in attention mechanisms, it performs worse than batch normalization when used with convolution. Notably, this has already been used in the PSA module [53] and our finding is in line with its design.

∙\bullet Hierarchical Design: Table 5(b). Unlike other detection systems such as Mask R-CNN [25, 1], where the architectures of the plain vision transformers can produce strong results, YOLOv12 exhibits a different behavior. When using a plain vision transformer (N/A), the detector’s performance drops significantly, achieving only 38.3%38.3\% mAP. A more moderate adjustment, such as omitting the first (S1) or fourth stage (S4), while maintaining similar FLOPs by adjusting the feature dimensions, results in a slight performance degradation of 0.5%0.5\% mAP and 0.8%0.8\% mAP, respectively. Consistent with previous YOLO models, the hierarchical design remains the most effective, yielding the best performance in YOLOv12.

∙\bullet Training Epochs: Table 5(c). We examine how varying the number of training epochs impacts performance (training from scratch). Although some existing YOLO detectors achieve optimal results after roughly 500500 training epochs [24, 58, 53], YOLOv12 requires a more extended training period (about 600600 epochs) to achieve peak performance, keeping the same configuration used in YOLOv11 [28].

∙\bullet Position Perceiver: Tables 5(d). In the attention mechanism, we apply a separable convolution with a large kernel to the attention value vv, adding its output to v​@​attnv@\text{attn}. We refer to this component as the Position Perceiver, as the smoothing effect of convolution preserves the original positions of image pixels, it helps the attention mechanism perceive positional information (This has already been used in the PSA module [53], but we expand the convolution kernel, achieving performance improvement without affecting speed). As shown in the table, increasing the convolution kernel size improves performance but gradually reduces speed. When the kernel size reaches 9×99\times 9, the slowdown becomes significant. Therefore, we set 7×77\times 7 as the default kernel size.

|  | APv​a​l50:95\text{AP}^{val}_{50:95} (%) | AP50v​a​l\text{AP}^{val}_{50}(%) | AP75v​a​l\text{AP}^{val}_{75} (%) | APs​m​a​l​lv​a​l\text{AP}^{val}_{small} (%) | APm​e​d​i​u​mv​a​l\text{AP}^{val}_{medium} (%) | APl​a​r​g​ev​a​l\text{AP}^{val}_{large} (%) |
|---|---|---|---|---|---|---|
| YOLOv12-N | 40.6 | 56.7 | 43.8 | 20.2 | 45.2 | 58.4 |
| YOLOv12-S | 48.0 | 65.0 | 51.8 | 29.8 | 53.2 | 65.6 |
| YOLOv12-M | 52.5 | 69.6 | 57.1 | 35.7 | 58.2 | 68.8 |
| YOLOv12-L | 53.7 | 70.7 | 58.5 | 36.9 | 59.5 | 69.9 |
| YOLOv12-X | 55.2 | 72.0 | 60.2 | 39.6 | 60.7 | 70.9 |

∙\bullet Position Embedding: Tables 5(e). We examine the impact of commonly used positional embeddings in most attention-based models (RPE: relative positional embedding; APE: absolute positional encoding) on performance. Interestingly, the best-performing configuration is achieved without any positional embedding, which brings cleaner architecture and faster inference latency.

∙\bullet Area Attention: Tables 5(f). In this table, we use the FlashAttention technique by default. This causes that while the area attention mechanism increases computational complexity (leading to performance gains), the resulting slowdown remains minimal. For further validation of area attention effectiveness, refer to Table 3.

∙\bullet MLP Ratio: Tables 5(g). In traditional vision transformers, the MLP ratio within the attention module is generally set to 4.04.0. However, we observe different behavior with YOLOv12. In the table, varying the MLP ratio impacts the model size, so we adjust the feature dimensions to maintain overall model consistency. In particular, YOLOv12 achieves better performance with an MLP ratio of 1.21.2, diverging from conventional practices. This adjustment shifts the computational load more toward the attention mechanism, highlighting the importance of area attention.

∙\bullet FlashAttention: Tables 5(h). This table validates the role of FlashAttention in YOLOv12. It shows that FlashAttention accelerates YOLOv12-N by approximately 0.30.3ms and YOLOv12-S by around 0.40.4ms without other costs.

Visualization: Heat Map Comparison. Figure 5 compares the heat maps of YOLOv12 with state-of-the-art YOLOv10 [53] and YOLOv11 [28]. These heat maps, extracted from the third stage of the backbones of X-scale models, highlight the regions activated by the model, reflecting its object perception capability. As illustrated, compared to YOLOv10 and YOLOv11, YOLOv12 produces clearer object contours and more precise foreground activation, indicating improved perception. Our explanation is that this improvement comes from the area attention mechanism, which has a larger receptive field than convolutional networks and is therefore considered better at capturing the overall context, leading to more precise foreground activation. We believe that this characteristic gives YOLOv12 a performance advantage.

| Hyperparameters | YOLOv12-N/S/M/L/X |
|---|---|
| Training Configuration |  |
| Epochs | 600600 |
| Optimizer | SGD |
| Momentum | 0.9370.937 |
| Batch size | 32×832\times 8 |
| Weight decay | 5×10−45\times 10^{-4} |
| Warm-up epochs | 33 |
| Warm-up momentum | 0.80.8 |
| Warm-up bias learning rate | 0.00.0 |
| Initial learning rate | 10−210^{-2} |
| Final learning rate | 10−410^{-4} |
| Learning rate schedule | Linear decay |
| Loss Parameters |  |
| Box loss gain | 7.57.5 |
| Class loss gain | 0.50.5 |
| DFL loss gain | 1.51.5 |
| Augmentation Parameters |  |
| HSV saturation augmentation | 0.70.7 |
| HSV value augmentation | 0.40.4 |
| HSV hue augmentation | 0.0150.015 |
| Translation augmentation | 0.10.1 |
| Scale augmentation | 0.5/0.9/0.9/0.9/0.90.5/0.9/0.9/0.9/0.9 |
| Mosaic augmentation | 1.01.0 |
| Mixup augmentation | 0.0/0.05/0.15/0.15/0.20.0/0.05/0.15/0.15/0.2 |
| Copy-paste augmentation | 0.1/0.15/0.4/0.5/0.60.1/0.15/0.4/0.5/0.6 |
| Close mosaic epochs | 1010 |

## 5 Conclusion

This study introduces YOLOv12, which successfully adopts an attention-centric design that traditionally is considered inefficient for real-time requirements, into the YOLO framework, achieving a state-of-the-art latency-accuracy trade-off. To enable efficient inference, we propose a novel network that leverages area attention to reduce computational complexity and residual efficient layer aggregation networks (R-ELAN) to enhance feature aggregation. Furthermore, we refine key components of the vanilla attention mechanism to better align with YOLO’s real-time constraints while maintaining high-speed performance. As a result, YOLOv12 achieves state-of-the-art performance by effectively combining area attention, R-ELAN, and architectural optimizations, leading to significant improvements in both accuracy and efficiency. Comprehensive ablation studies further validate the effectiveness of these innovations. This study challenges the dominance of CNN-based designs in YOLO systems and advances the integration of attention mechanisms for real-time object detection, paving the way for a more efficient and powerful YOLO system.

## 6 Limitations

YOLOv12 requires FlashAttention [14, 13], which currently supports Turing, Ampere, Ada Lovelace, or Hopper GPUs (e.g., T4, Quadro RTX series, RTX20 series, RTX30 series, RTX40 series, RTX A5000/6000, A30/40, A100, H100, etc.).

## 7 More Details

Fine-tuning Details. By default, all YOLOv12 models are trained using the SGD optimizer for 600600 epochs. Following previous works [57, 24, 58, 53], the SGD momentum and weight decay are set to 0.9370.937 and 5×10−45\times 10^{-4}, respectively. The initial learning rate is set to 1×10−21\times 10^{-2} and decays linearly to 1×10−41\times 10^{-4} throughout the training process. Data augmentations, including Mosaic [3, 57], Mixup [71], and copy-paste augmentation [65], are applied to enhance training. Following YOLOv11 [28], we adopt the Albumentations library [6]. Detailed hyperparameters are presented in Table 7. All models are trained on 8×8\times NVIDIA A6000 GPUs. Following established conventions [24, 58, 53, 28], we report the standard mean average precision (mAP) on different object scales and IoU thresholds. In addition, we report the average latency in all images. We recommend reviewing more details at the official code: https://github.com/sunsmarterjie/yolov12.

Result Details. We report more details of the results in Table 6 including APv​a​l50:95\text{AP}^{val}_{50:95}, AP50v​a​l\text{AP}^{val}_{50}, AP75v​a​l\text{AP}^{val}_{75}, APs​m​a​l​lv​a​l\text{AP}^{val}_{small}, APm​e​d​i​u​mv​a​l\text{AP}^{val}_{medium}, APl​a​r​g​ev​a​l\text{AP}^{val}_{large}.
