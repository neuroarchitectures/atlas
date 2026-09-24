# MobileNetV4: Universal Models for the Mobile Ecosystem

We present the latest generation of MobileNets: MobileNetV4 (MNv4). They feature universally-efficient architecture designs for mobile devices. We introduce the Universal Inverted Bottleneck (UIB) search block, a unified and flexible structure that merges Inverted Bottleneck (IB), ConvNext, Feed Forward Network (FFN), and a novel Extra Depthwise (ExtraDW) variant. Alongside UIB, we present Mobile MQA, an attention block for mobile accelerators, delivering a significant 39% speedup. An optimized neural architecture search (NAS) recipe is also introduced which improves MNv4 search effectiveness. The integration of UIB, Mobile MQA and the refined NAS recipe results in a new suite of MNv4 models that are mostly Pareto optimal across mobile CPUs, DSPs, GPUs, as well as accelerators like Apple Neural Engine and Google Pixel EdgeTPU. This performance uniformity is not found in any other models tested. We introduce performance modeling and analysis techniques to explain how this performance is achieved. Finally, to further boost accuracy, we introduce a novel distillation technique. Enhanced by this technique, our MNv4-Hybrid-Large model delivers 87% ImageNet-1K accuracy, with a Pixel 8 EdgeTPU runtime of 3.8ms.

## 1 Introduction

Efficient on-device neural networks not only enable fast, real-time and interactive experiences, but also avoid streaming private data through the public internet. However, the computational constraints of mobile devices pose the significant challenge of balancing accuracy and efficiency. To this end, we introduce UIB and Mobile MQA, two innovative building blocks integrated via a refined NAS recipe to create a series of universally mostly-Pareto-optimal mobile models.11 1 All MobileNetV4 models are available at https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py Additionally, we present a distillation technique that further improves efficiency.

Our Universal Inverted Bottleneck (UIB) block improves the Inverted Bottleneck block [38] by incorporating two optional depthwise convolutions [20]. Despite its simplicity, UIB unifies prominent micro-architectures - Inverted Bottleneck (IB), ConvNext [34], and FFN [13] - and introduces the Extra Depthwise (ExtraDW) IB block. UIB offers flexibility in spatial and channel mixing, the option to extend the receptive field, and enhanced computational efficiency.

Our optimized Mobile MQA block achieves over a 39% inference speedup on mobile accelerators with respect to Multi-Head Attention [47].

Our two-phase NAS approach, separating coarse and fine-grained searches, significantly boosts search efficiency and facilitates the creation of models that are significantly larger than previous state-of-the-art models [43]. Additionally, incorporating an offline distillation dataset reduces noise in NAS reward measurements, resulting in improved model quality.

By integrating UIB, MQA, and an improved NAS recipe, we present the MNv4 suite of models which achieve mostly Pareto optimal performance across diverse hardware platforms, including CPUs, DSPs, GPUs, and accelerators. Our models range from the extremely compact MNv4-Conv-S to the MNv4-Hybrid-L high-end variant that establishes a new reference for mobile model accuracy. MNv4-Conv-S achieves 73.8% top-1 ImageNet-1K accuracy with 3.8M parameters, 0.2G MACs and 2.4 ms of Pixel 6 CPU latency. MNv4-Hybrid-L gets 83.4% top-1 within 3.8 ms on Pixel 8 EdgeTPU. Our novel distillation recipe mixes datasets with different augmentations and adds balanced in-class data, enhancing generalization and increasing accuracy. With these techniques, MNv4-Hybrid-L achieves a 87% top-1 accuracy on ImageNet-1K: 0.5% less than its teacher, despite having 39x less MACs.

## 2 Related Work

Optimizing models for both accuracy and efficiency is a well studied problem.

Mobile Convolutional Networks: Key work includes MobileNetV1 [21] with depthwise-separable convolutions for better efficiency, MobileNetV2 [38] introducing linear bottlenecks and inverted residuals, GhostNet [17] increasing the relative frequency of depthwise convolutions, MnasNet [42] integrating lightweight attention in bottlenecks, and MobileOne [46] adding and re-parameterizing linear branches in inverted bottlenecks at inference time.

Efficient Hybrid Networks: This research combines convolutions and attention. MobileViT [35] merges CNN strengths with ViT [13] through global attention blocks. GhostNetV2 [44] uses FC layers to capture long-range dependencies. MobileFormer [7] parallelizes a MobileNet and a Transformer with a two-way bridge in between for feature fusing. FastViT [45] adds attention to the last stage with large convolutional kernels instead of early stage self-attention.

Efficient Attention: Research has focused on enhancing MHSA [47] efficiency. EfficientViT [14] and MobileViTv2 [36] introduce self-attention approximations for linear complexity with minor accuracy impacts. EfficientFormerV2 [29] downsamples Q, K, and V for efficiency, while CMT [16] and NextViT [28] downsample only K and V.

Hardware-aware Neural Architecture Search (NAS): Another common technique is to automate the model design process using hardware-aware Neural Architecture Search (NAS). NetAdapt [52] uses empirical latency tables to optimize the accuracy of a model under a target latency constraint. MnasNet [42] also uses latency tables, but applies reinforcement learning to do hardware-aware NAS. FBNet [50] accelerates multi-task hardware-aware search via differentiable NAS. MobileNetV3 [19] is tuned to mobile phone CPUs through a combination of hardware-aware NAS, the NetAdapt algorithm, and architecture advances. MobileNet MultiHardware [9] optimizes a single model for multiple hardware targets. Once-for-all [6] separates training and search for efficiency.

## 3 Hardware-Independent Pareto Efficiency

The Roofline Model: For a model to be universally efficient, it must perform well on hardware targets with vastly different bottlenecks that limit the model’s performance. These bottlenecks are largely determined by the hardware’s peak computational throughput and its peak memory bandwidth.

To this end, we use the Roofline Model [49] which estimates the performance of a given workload and predicts whether it is memory-bottlenecked or compute-bottlenecked. In short, it abstracts away specific hardware details and only considers a workload’s operational intensity (LayerMACsi/(WeightBytesi+ActivationBytesi)\text{LayerMACs}_{i}/(\text{WeightBytes}_{i}+\text{ActivationBytes}_{i})) vs. the theoretical limits of the hardware’s processor and memory system. Memory and compute operations happen roughly in parallel, so the slower of the two approximately determines the latency bottleneck. To apply the Roofline Model to neural networks with layers indexed by ii, we can calculate the model inference latency, ModelTime, as follows:

|  | ModelTime\displaystyle\text{ModelTime} | =∑imax⁡(MACTimei,MemTimei)\displaystyle=\sum_{i}\max(\text{MACTime}_{i},\text{MemTime}_{i}) |  | (1) |
|---|---|---|---|---|
|  | MACTimei=LayerMACsiPeakMACs,\displaystyle\text{MACTime}_{i}=\frac{\text{LayerMACs}_{i}}{\text{PeakMACs}}, | MemTimei=WeightBytesi+ActivationBytesiPeakMemBW\displaystyle\text{MemTime}_{i}=\frac{\text{WeightBytes}_{i}+\text{ActivationBytes}_{i}}{\text{PeakMemBW}} |  |  |

In the roofline model, hardware behavior is summarized by the Ridge Point (RP)—the ratio of a hardware’s PeakMACs to PeakMemBW i.e. the minimum operational intensity required to achieve maximum performance. 22 2 The common practice of using a model’s total MACs to proxy latency is the same as targeting a roofline model with a Ridge Point (RP)=0\text{(RP)}=0. This is equivalent to infinite bytes per MAC so, ∀i,MemTimei=0\forall i,\text{MemTime}_{i}=0 and ModelTime=∑iMACTimei\text{ModelTime}=\sum_{i}\text{MACTime}_{i}. In order to optimize for hardware with a wide range of bottlenecks, as seen in Fig. 2 and Fig. 3, we analyze our algorithms’ latency while sweeping the RP from its lowest expected value (0 MAC/byte) to its highest expected value (500 MACs/byte)—see Appendix 0.F for more details. Roofline Models only depend on the ratio of data transfer to compute, so all hardware with the same RP will rank workloads the same by latency.33 3 The Roofline Model assumes that software implementation has no impact on workload performance. This means techniques with complex memory access (e.g. pruning) perform much better on a Roofline Model than on a real device. This means that swept-RP roofline analysis (see next paragraph) applies to future hardware and software if the RP of the new targets is contained in the swept range.44 4 Andrew Lavin independently proposed a similar framework in the context of analyzing strategies for performance modeling and kernel execution [26].

Ridge Point Sweep Analysis: As seen in Fig. 2 and Fig. 3, the roofline model sheds light on how MobileNetV4 models achieve hardware-independent mostly-Pareto-optimal performance against other convolutional MobileNets. On low-RP hardware (e.g. CPUs), models are more likely to be compute-bound than memory-bound. So, to improve latency, you minimize the total number of MACs even at the cost of increased memory complexity (MobileNetV3Large-1.5x). Data movement is the bottleneck on high-RP hardware, so MACs do not meaningfully slow down the model but can increase model capacity (MobileNetV1-1.5x). So models optimized for low-RPs run slowly at high-RPs because memory-intensive and low-MAC fully-connected (FC) layers are bottlenecked on memory bandwidth and can’t take advantage of the high available PeakMACs.

MobileNetV4 Design: MobileNetV4 balances investing MACs and memory bandwidth where they will provide the maximum return for the cost, paying particular attention to the start and end of the network. At the beginning of the network, MobileNetV4 uses large and expensive initial layers to substantially improve the models’ capacity and downstream accuracy. These initial layers are dominated by a high number of MACs, so they are only expensive on low-RP hardware. At the end of the network, all MobileNetV4 variants use the same size final FC layers to maximize accuracy, even though this causes smaller MNV4 variants to suffer higher FC latency on high-RP hardware. Since large initial Conv layers are expensive on low-RP hardware but not high-RP hardware while the final FC layers are expensive on high-RP hardware but not low-RP hardware, MobileNetV4 models will never see both slowdowns at the same time. In other words, MNv4 models are able to use expensive layers that disproportionately improve accuracy but do not suffer the simultaneous combined costs of the layers, resulting in mostly Pareto-optimal performance at all ridge points.

## 4 Universal Inverted Bottlenecks

With an established foundation of roofline modeling and operational intensity, we proceed to discuss our architectural blocks. First is the Universal Inverted Bottleneck (UIB) Block, a building block for efficient network design that can adapt to a variety of optimization targets while remaining simple enough to use with Neural Architecture Search (NAS). Figure 4 shows the UIB block structure.

UIB extends the MobileNet Inverted Bottleneck (IB) block (introduced in MobileNetV2 [38]), which has become the standard building block for efficient networks [19, 43, 34, 13]. We introduce an optional DW before the expansion layer and also make the DW between the expansion and projection layer optional. The NAS procedure selects which DW ops to include, resulting in novel architectures. Despite the simplicity of this modification, our new building block unifies important existing blocks: the original IB block, ConvNext block, and the FFN block in ViT. Additionally, UIB introduces a novel variant: the Extra DepthWise IB (ExtraDW) block. The NAS SuperNet size is manageable because the pointwise expansion and projection components of each block are shared between instantions and the depthwise ops are searchable options. In a SuperNet-based NAS algorithm, this approach shares >>95% of the parameters between instantiations so NAS remains efficient. We further use FusedIBs to improve the efficiency: A k×kk{\times}k FusedIB is a k×kk{\times}k Conv2D into a 1×11{\times}1 Conv2D [1]. FusedIBs are used in all MNV4 model stems (Appendix 0.D, Tabs. 11 - 15).

UIB Instantiations: The two optional depthwise convolutions in the UIB block have four possible instantiations (Fig. 4), resulting in different tradeoffs.

MobileNet Inverted Bottleneck (IB) performs spatial mixing on the expanded features’ activations for greater model capacity at increased cost.

ConvNext-Like allows for a cheaper spatial mixing with larger kernel size by performing the spatial mixing before the expansion.

ExtraDW inexpensively increases the network depth and receptive field, combining the benefits of ConvNext-Like and IB. 55 5 ExtraDW could be seen as a MobileNetV1-style factorization of two standard convolutional blocks.

FFN is a stack of two 1x1 pointwise convolutions (PW) with activation and normalization layers in between. PW is very accelerator-friendly but works best with other blocks.

At each network stage, UIB provides flexibility to: (1) Strike an ad-hoc spatial and channel mixing tradeoff. (2) Enlarge the receptive field as needed. (3) Maximize the computational utilization. Tab. 1 shows impact on accuracy and latency across three searches.

| Block | Top-1 | GMACs | MParams | P8 EdgeTPU |
|---|---|---|---|---|
| UIB | 83.3% | 6.2 | 33.0 | 2.68 ms |
| CN | 83.2% | 6.9 | 35.1 | 2.69 ms |
| IB | 82.3% | 6.1 | 32.4 | 2.61 ms |

## 5 Mobile MQA

In this section we present Mobile MQA, a novel accelerator-optimized attention block which speeds up attention by >>39%.

Importance of Operational Intensity: Vision model research has largely focused on improving efficiency by reducing MACs. Since accelerators greatly increase computational capabilities without proportionally increasing memory bandwidth, many models are bottlenecked by memory access and solely minimizing MACs will not improve performance. Instead we must consider the Operational Intensity—the ratio of arithmetic operations to memory access.

MQA is efficient in hybrid models: MHSA [47] projects the queries, keys, and values into multiple spaces to capture different aspects of the information. Multi-Query Attention (MQA) [39] simplifies this by sharing keys and values across all heads. While large language models require multiple query heads, they can share a single head for keys and values without sacrificing accuracy [8] [27]. When the number of batched tokens is small compared to the feature dimensions, sharing one head across keys and values reduces memory bandwidth requirements—significantly improving Operational Intensity. In hybrid mobile vision models, the tokens are often small compared to features because attention is only used in the low-resolution later stages with high feature dimensions and because batch size one operation is common. Our experiments confirm MQA’s advantage in hybrid models. As shown in Tab. 2 MQA achieves >>39% acceleration on EdgeTPUs and Samsung S23 GPU with negligible quality loss (-0.03%) compared to MHSA. MQA also reduces MACs and model parameters by >>25%. To our knowledge, we are the first to use MQA for mobile vision. Furthermore, we introduce an additional Einsum optimization (see Appendix 0.G), specifically tailored for accelerated inference on hardware accelerators.

| model | Top-1 | MACs | Params | EdgeTPU | Samsung S23 |  |
|---|---|---|---|---|---|---|
|  | Acc(%) | (G) | (M) | Pixel 7 | Pixel 8 | GPU |
| base model | 84.88 | 6.0 | 30.9 | 4.31 ms | 2.35 ms | 13.15 ms |
| +3 MHSA | 85.27 | 6.7 | 36.0 | 9.69 ms | 2.76 ms | 16.46 ms |
| +3 MQA | 85.24 | 6.5 | 34.7 | 5.16 ms | 2.60 ms | 15.10ms |
|  | (-0.03%) | (-28.6%) | (-25.5%) | (-84.2%) | (-39.0%) | (-41.1%) |

Incorporate asymmetric spatial down-sampling: Drawing inspiration from MQA, which utilizes asymmetric computation across queries, keys, and values, we add Spatial Reduction Attention (SRA) [48] to our optimized MQA block to downscale key and value resolution while retaining high-resolution queries. This strategy is motivated by the observed correlation between spatially adjacent tokens in hybrid models attributed to spatial mixing convolution filters in early layers. Unlike [48], our method replaces AvgPooling with a stride-2 3x3 DW for spatial reduction—a cost-effective way to boost model capacity.

Mobile MQA Here we present our Mobile MQA block:

|  | Mobile_MQA​(𝐗)\displaystyle\text{Mobile\_MQA}(\mathbf{X}) | =Concat​(attention1,…,attentionn)​𝐖O\displaystyle=\text{Concat}(\text{attention}_{1},\dots,\text{attention}_{n})\mathbf{W}^{O} |  | (2) |
|---|---|---|---|---|
|  | where​attentionj\displaystyle\text{where}\ \text{attention}_{j} | =softmax​((𝐗𝐖Qj)​(S​R​(𝐗)​𝐖K)Tdk)​(S​R​(𝐗)​𝐖V)\displaystyle=\text{softmax}\left(\frac{(\mathbf{X}\mathbf{W}^{Q_{j}})(SR(\mathbf{X})\mathbf{W}^{K})^{T}}{\sqrt{d_{k}}}\right)(SR(\mathbf{X})\mathbf{W}^{V}) |  |  |

where S​RSR denotes either spatial reduction, our stride-2 DW, or, if spatial reduction isn’t used, the identity function. As shown in Tab. 3, asymmetric spatial down-sampling adds >20% efficiency with minimal accuracy loss (-0.06%).

| down-sampling on KV | Top-1 Acc | MACs (G) | CPU (ms) | GPU (ms) |
|---|---|---|---|---|
| No | 80.77 | 1.285 | 15.8 | 7.4 |
| Yes | 80.71 | 1.245 | 12.8 | 5.9 |
| Efficiency Gain | - | +3% | +23% | +25% |

## 6 Design of MNv4 Models

Our Design Philosophy: Simplicity Meets Efficiency. In developing the latest MobileNets, our core goal was Pareto optimality across diverse mobile platforms. To achieve this, we started by conducting extensive correlation analyses on existing models and hardware. Through empirical examination, we found a set of components and parameters that ensure high correlations between cost models (the prediction of cost of latency) across various devices while approaching the Pareto frontier in performance. Our investigation unveiled critical insights:

Multi-path efficiency concerns: Group convolutions [56] and similar multi-path designs, despite lower MAC counts, can be less efficient due to memory access complexity.

Hardware support matters: Advanced modules like Squeeze and Excite (SE) [22], GELU [18], and LayerNorm [2] are not well supported on DSPs, with LayerNorm also lagging behind BatchNorm [24], and SE is slow on accelerators.

The Power of Simplicity: Conventional components – depthwise and pointwise convolutions, ReLU [37], BatchNorm, and simple attention (e.g., MHSA) – demonstrate superior efficiency and hardware compatibility.

Based on these findings, we established a set of design principles:

- • Standard Components: We prioritize widely supported elements for seamless deployment and hardware efficiency.
Standard Components: We prioritize widely supported elements for seamless deployment and hardware efficiency.

- • Flexible UIB Blocks: Our searchable UIB block lets NAS tune spatial and channel mixing, adjust receptive fields, and improve hardware utilization.
Flexible UIB Blocks: Our searchable UIB block lets NAS tune spatial and channel mixing, adjust receptive fields, and improve hardware utilization.

- • Employ Straightforward Attention: Our Mobile MQA mechanism prioritizes simplicity for optimal performance.
Employ Straightforward Attention: Our Mobile MQA mechanism prioritizes simplicity for optimal performance.

These principles allow MobileNetV4 to be mostly Pareto-optimal on all hardware evaluated. In the following, we detail our refined NAS recipe for UIB model search, outline specific search configurations for various MNv4-Conv model sizes, and explain the construction of hybrid models.

### 6.1 Refining NAS for Enhanced Architectures

To effectively instantiate the UIB blocks, we adopt TuNAS [4] with tailored enhancements for improved performance. We use use per-size searches and search spaces instead of using fixed scaling rules such as in EfficientNet[43].

Enhanced Search Strategy: Our approach mitigates TuNAS’s bias towards smaller filters and expansion factors, attributed to parameter sharing, by implementing a two-stage search. This strategy addresses the variance in parameter counts between UIB’s depthwise layers and other search options.

Coarse-Grained Search: Initially, we focus on determining optimal filter sizes while maintaining fixed parameters: an inverted bottleneck block with a default expansion factor of 4 and a 3x3 depthwise kernel.

Fine-Grained Search: Building on the initial search’s outcomes, we search the configuration of UIB’s two depthwise layers (including their presence and kernel size of either 3x3 or 5x5), keeping the expansion factor constant at 4.

Tab. 4 demonstrates the enhanced efficiency and model quality achieved through our two-stage search compared to a conventional one-stage search, where a unified search space was explored in a single TuNAS pass.

| Search Method | Top-1 Acc (Val) | Top-1 Acc (Train) | Pixel 6 EdgeTPU (ms) |
|---|---|---|---|
| One-stage | 81.26 | 74.64 | 3.85 |
| Two-stage | 81.48 (+0.22) | 78.24 (+3.60) | 3.67 (-4.68%) |

Enhancing TuNAS with Robust Training: The success of TuNAS hinges on accurately evaluating architecture quality, crucial for reward calculation and policy learning. Originally, TuNAS leveraged ImageNet-1k for training the SuperNet, but ImageNet performance is notably affected by data augmentation, regularization, and hyper-parameter choices. Given TuNAS’s evolving architecture samples, finding a stable set of hyper-parameters is challenging.

We address this with an offline distillation dataset, eliminating the need for extra augmentations and reducing sensitivity to regularization and optimization settings. The JFT distillation dataset, as detailed in Sec. 8, serves as our training set for TuNAS, with notable improvements shown in Tab. 5. Acknowledging that depth-scaling surpasses width-scaling in extended training sessions [3], we extend TuNAS training to 750 epochs, yielding deeper, higher-quality models.

| NAS Dataset | Top-1 Acc (Val/Train) | MACs | Params | Pixel 4 GPU | Pixel 6 CPU |
|---|---|---|---|---|---|
| ImageNet | 82.4 / 72.9 | 7.2G | 43.5M | 59.2ms | 70.4ms |
| JFT distill | 82.3 / 74.0 | 6.2G | 34.4M | 51.0ms | 67.3ms |
| Gain | -0.1 / +1.1 | +13.9% | +20.9% | +13.9% | +4.4% |

### 6.2 Optimization of MNv4 Models

We constructed MNv4-Conv models from NAS-optimized UIB blocks, tailoring them for specific resource constraints. More details are given in Appendix 0.A. In line with other hybrid models, we found that adding attention to the last stages of convolution models is most effective. In MNv4-Hybrid models, we interlace Mobile MQA blocks with UIB blocks for enhanced performance. For comprehensive model specifications, refer to Appendix 0.D.

## 7 Results

In this section, we demonstrate the mostly Pareto-optimal performance of MobileNetV4 (MNv4) on ImageNet-1K classification and COCO object detection.

### 7.1 ImageNet classification

Experimental Setup: To assess model architecture performance, we train exclusively with the ImageNet-1k [12] training split and measure Top-1 accuracy on its validation split. Our latency analysis includes a representative selection of mobile hardware, including ARM Cortex CPUs (Pixel 6, Samsung S23), Qualcomm Hexagon DSP (Pixel 4), ARM Mali GPU (Pixel 7), Qualcomm Snapdragon (S23 GPU), Apple Neural Engine, and Google EdgeTPU. Our complete training recipe is detailed in the Appendix 0.C.

We benchmark our models against the leading efficient models, including hybrid (MiT-EfficientViT [14], FastViT [45], NextViT [28]) and convolutional models (MobileOne [46], ConvNext [34], and previous MobileNets [20] [38] [19]) based on their reported Top-1 Accuracies and our latency evaluations. We used modern training recipes to improve MobileNetV1-V3 accuracy: a +3.4% (to 74.0%) for V1, +1.4% (to 73.4%) for V2, and +0.3% (to 75.5%) for V3. These new figures are used throughout the paper to isolate architectural advancements.

|  |  |  |  | Latency (ms) |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|
|  |  | Params | MACs | Pixel 6 | Pixel 8 | iPhone 13 | Pixel 4 | Pixel 7 | Samsung S23 |  |
| Model | Top-1 | (M) | (G) | CPU | EdgeTPU | CoreML | Hexagon | GPU | CPU | GPU |
| MobileNet-V2-0.5x [38] | 66 | 2.0 | 0.1 | 2.4 | 0.7 | 0.5 | 2.9 | 8.3 | 1.8 | 1.9 |
| MobileNet-V3L-0.5x [19] | 69.2 | 2.7 | 0.1 | 2.4 | 0.8 | 0.45 | 3.5 | 9.9 | 2.0 | 2.1 |
| MobileOne-S0 [46] | 71.4 | 2.1 | 0.3 | 4.2 | 0.7 | 0.5 | 2.9 | 10.7 | 3.3 | 1.7 |
| MobileNet-V2 [38] | 73.4 | 3.5 | 0.3 | 5.0 | 0.7 | 0.7 | 3.9 | 13.6 | 4.1 | 2.5 |
| MNv4-Conv-S | 73.8 | 3.8 | 0.2 | 2.4 | 0.7 | 0.6 | 2.4 | 8.4 | 1.8 | 2.0 |
| MobileNet-V1 [21] | 74.0 | 4.2 | 0.6 | 6.1 | 0.8 | 0.7 | 3.2 | 13.0 | 4.6 | 2.1 |
| FastViT-T8† [45] | 75.6 | 3.6 | 0.7 | 49.3 | 1.3 | 0.7 | Failed | 40.7 | 43.6 | 24.7 |
| MobileNet-V2-1.5x [38] | 76.8 | 6.8 | 0.7 | 9.3 | 0.9 | 1.0 | 5.6 | 16.4 | 7.3 | 3.3 |
| MultiHardware-MAX-1.5x [9] | 77.9 | 8.9 | 0.8 | 9.8 | 1.0 | - | 5.7 | 23.2 | - | 4.1 |
| MultiHardware-AVG-1.5x [9] | 78.2 | 10.0 | 1.0 | 12.0 | 1.1 | - | 6.1 | 20.3 | - | 4.5 |
| MobileNet-V2-2.0x [38] | 78.4 | 11.2 | 1.1 | 13.9 | 1.1 | 1.5 | 6.9 | 19.1 | 10.6 | 4.2 |
| MobileOne-S4 [46] | 79.4 | 14.8 | 1.5 | 26.7 | 1.7 | 1.5 | 9.0 | 28.6 | 19.4 | 5.9 |
| FastViT-S12† [45] | 79.8 | 8.8 | 1.8 | 83.0 | 1.8 | 1.6 | Failed | 75.0 | 69.2 | 47.0 |
| MIT-EfficientViT-B1-r224 [14] | 79.4 | 9.1 | 0.5 | - | - | 2.4 | - | - | 18.1 | 5.0 |
| MNv4-Conv-M | 79.9 | 9.2 | 1.0 | 11.4 | 1.1 | 1.1 | 7.3 | 18.1 | 8.6 | 4.1 |
| FastViT-SA12 [45] | 80.6 | 10.9 | 1.9 | 86.5 | 2.0 | 1.6 | Failed | 79.6 | 69.5 | 52.1 |
| MNv4-Hybrid-M | 80.7 | 10.5 | 1.2 | 14.3 | 1.5 | - | Failed | 17.9 | 10.8 | 5.9 |
| FastViT-SA24 [45] | 82.6 | 20.6 | 3.8 | 171.6 | 3.2 | 2.4 | Failed | 131.9 | 136.3 | 107.5 |
| MIT-EfficientViT-B2-r256 [14] | 82.7 | 24.0 | 2.1 | - | - | 5.4 | - | - | 64.9 | 9.5 |
| MNv4-Conv-L | 82.9 | 31 | 5.9 | 59.9 | 2.4 | 3.0 | 20.8 | 37.6 | 43.0 | 13.2 |
| ConvNext-S [34] | 83.1 | 50 | 8.7 | 314.9 | 3.7 | - | Failed | 45.2 | 243.9 | 18.5 |
| NextViT-B [28] | 83.2 | 44.8 | 8.3 | - | - | - | - | - | - | - |
| MNv4-Hybrid-L | 83.4 | 35.9 | 7.2 | 87.6 | 3.8 | - | Failed | 61.3 | 61.8 | 18.1 |
| MIT-EfficientViT-B3-r224 [14] | 83.5 | 49.0 | 4.0 | - | - | 12.2 | - | - | 125.9 | 18.4 |
| FastViT-SA36 [45] | 83.6 | 30.4 | 5.6 | 241.6 | 4.3 | - | Failed | 186.5 | 206.3 | 138.1 |

Results: Our results, seen in Fig. 1 and Tab. 6, demonstrate that MNv4 models are mostly Pareto-optimal across a range of accuracies and mobile targets, including CPUs, DSPs, GPUs, and accelerators like the Apple Neural Engine and Google EdgeTPU.

MNv4 performs notably well on CPU—roughly 2x faster than MobileNetV3 and substantially faster than iso-accuracy models. On EdgeTPUs, MNv4 models are as accurate as MobileNetV3 and 2x as fast. MNv4-Conv-M is >50% faster than MobileOne-S4 and FastViT-S12 and has +1.5% more Top-1 accuracy than MobileNetV2 at comparable latency. On S23 GPU and iPhone 13 CoreML (ANE), MNv4 is mostly at the Pareto front. MIT-EfficientViT—which has the closest performance on S23 GPU—has >2x the latency as MNv4 on CoreML at the same accuracy. FastViT—optimized for Apple Neural Engine—is 2nd on CoreML but has >5x the latency of MNv4 on S23 GPU. While some models, such as EfficientViT, reach the same accuracy with fewer MACs, MobileNetV4 models are optimized for high accuracy and minimal latency on the most hardware possible. Increasing MACs often decreases memory bandwidth and op complexity which is often more important for achieving this goal. Like many hybrid models, MNv4-hybrid models are not compatible with DSPs. MNv4-Conv models remain the top performers on DSP, emphasizing the compatibility and efficiency across diverse hardware provided by our UIB block, NAS recipe, and search spaces. MNv4-Hybrid performs well on CPUs and accelerators which demonstrates the broad efficiency of Mobile MQA.

Mobile models should perform well on diverse hardware, but we show that many models fail to meet this requirement. MobileNetV3 performs well on CPUs but not on EdgeTPU, DSPs, and GPUs. FastViT performs well on ANE but not on CPUs and GPUs. EfficientViT has good performance on GPUs but not on ANE. In contrast, MNv4-Conv models achieves mostly-Pareto-optimal performance across CPUs, GPUs, DSPs, the Apple Neural Engine, and Google EdgeTPUs. This versatility ensures MNv4-Conv models can be easily deployed across the mobile ecosystem and sets a new benchmark for mobile model universality.

### 7.2 COCO Object Detection

Experimental Setup: We evaluate the effectiveness of MNv4 backbones for object detection tasks on the COCO 17 [33] dataset. We compare MNv4 medium backbones against SOTA backbones with a MAC count. For each backbone, we build a detector using the RetinaNet [32] framework. We attach a 256256-d FPN [31] decoder to the P33 - P77 endpoints, as well as a 256256-d prediction head with 44 convolutional layers. As usual for mobile detectors, we use depth-separable convolutions for an efficient FPN decoder and box prediction head. We train all models on COCO 17 [33] for 600600-epochs. Images are resized to 384​p​x384px and augmented with random horizontal flip, random scale, and Randaug [10]. We exclude Shear and Rotate from Randaug, as those deteriorate small-object detection AP. The models are trained with a 20482048 batch size, Adam [25], and a 0.000030.00003 L2 weight decay, plus a cosine LR schedule with 2424 epochs warm-up. The learning rate is tuned per-model. For all baselines, filter multipliers are tuned to similar MACs. Following classification, MobileNetV4 backbones are trained using a 0.20.2 stochastic drop [23]. MobileNet baselines were from Tensorflow Model Garden [53] implementation. EfficientFormer was reimplemented in Tensorflow.

Results: Results are reported in Tab. 7. Parameters, MACs and benchmarks are computed using the entire detector at the 384​p​x384px input resolution. The MNv4-Conv-M detector achieves 32.6% AP, similar to MobileNetMultiAvg and MobileNetV2. However, this model is 1212% faster than MobileNetMultiAvg and 2323% faster than MobileNetV2 on Pixel 6 CPU. The MNv4-Hybrid-M detector gets +1.6+1.6% AP over MNv4-Conv-M while running 1818% slower on Pixel 6 CPU. This demonstrates the effectiveness of MNv4 hybrid models on tasks like object detection.

|  | COCO | MACs | Params | Pixel 6 CPU |
|---|---|---|---|---|
| Backbone | Val AP | (G) | (M) | latency (ms) |
| EfficientFormer L1 [30] | 29.5 | 6.54 | 12.77 | 84.3 |
| MobileNet v1 @ 1.5 [20] | 31.0 | 6.68 | 9.05 | 66.4 |
| MNv4-Conv-M | 32.6 | 5.06 | 9.79 | 51.3 |
| MobileNet Multi-AVG @ 1.5 [9] | 32.7 | 5.42 | 9.51 | 58.1 |
| MobileNet v2 @ 2.0 [38] | 32.9 | 5.81 | 10.15 | 66.4 |
| MobileNet v3 Large [19] @ 2.0 | 33.2 | 4.99 | 17.92 | 59.9 |
| MNv4-Hybrid-M | 34.0 | 5.62 | 11.15 | 60.5 |

## 8 Enhanced distillation recipe

Complementing architectural innovation, distillation is a powerful tool for enhancing machine learning efficiency. This is particularly true for mobile models where distillation can greatly increase accuracy without increasing latency. Building upon the Patient Teacher distillation baseline [5], we introduce two novel techniques to further boost performance.

Dynamic Dataset Mixing: Data augmentation is crucial for distillation performance. While prior methods rely on a fixed augmentation sequence, we find that dynamically mixing multiple datasets with diverse augmentation strategies improves distillation. Our experiments use three distillation datasets:

: Inception Crop [41] followed by RandAugment [11] l2m9 applied to 500 ImageNet-1k replicas.

: Inception Crop followed by extreme Mixup [55] applied to 1000 ImageNet-1k replicas (mirroring the Patient Teacher approach).

: A dynamic mixture of 𝒟1\mathcal{D}_{1} and 𝒟2\mathcal{D}_{2} during training.

Our results (Tab. 8) show that 𝒟2\mathcal{D}_{2} outperforms 𝒟1\mathcal{D}_{1} (84.1% vs. 83.8% student accuracy), but a dynamic mixture of the two (𝒟1\mathcal{D}_{1} + 𝒟2\mathcal{D}_{2}) elevates accuracy to 84.4% (+0.3%). This suggests that mixing expands the augmented image space, increases difficulty and diversity, and leads to improved student performance.

JFT Data Augmentation: To increase training data volume, we add in-domain, class-balanced data by resampling the JFT-300M [40] dataset to 130K images per class (130M total). Following Noisy Student[51] and using EfficientNet-B0 trained on ImageNet-1K, we select images with a relevance threshold above 0.3. For classes with abundant data, we choose the top 130K images; for rare classes, we replicate images for balance. This dataset is replicated 10x. Due to JFT’s complexity, we apply weaker augmentations (Inception Crop + RandAugment l2m5). This is dataset 𝒟3\mathcal{D}_{3}. Tab. 8 shows that using solely 𝒟3\mathcal{D}_{3} drops accuracy by 2%. However, combining ImageNet and JFT data (𝒟1\mathcal{D}_{1} + 𝒟2\mathcal{D}_{2} + 𝒟3\mathcal{D}_{3}) raises accuracy by +0.6%. The additional data improves generalization.

Our distillation recipe: Our combined distillation recipe dynamically mixes datasets 𝒟1\mathcal{D}_{1}, 𝒟2\mathcal{D}_{2}, and 𝒟3\mathcal{D}_{3} for diverse augmentations and leverages class-balanced JFT data. As shown in Tab. 8 and Tab. 9, our method improves top-1 accuracy >>0.8% over the previous SOTA [5]. Training an MNv4-Conv-L student model for 2000 epochs yields 85.9% top-1 accuracy. Our approach is effective: the student has 15x fewer parameters and 48x fewer MACs than its teacher (EfficientNet-L2), but is only 1.6% less accurate. MNv4-Conv-Hybrid reaches 87.0% top-1 accuracy by combining this distillation with pretraining on JFT. More details of our distillation recipe can be found in Appendix 0.H.

| Dataset | Data source | Augmentations | Mixing | Top-1 Acc (Val/Train) |  |
|---|---|---|---|---|---|
|  |  |  | Ratio | 400 epochs | 2000 epochs |
| 𝒟1\mathcal{D}_{1} | 1000×\times | Inception Crop & | - | 83.8/86.6 | - |
|  | ImageNet-1k | RandAug l2m9 |  |  |  |
| 𝒟2\mathcal{D}_{2} | 1000×\times | Inception Crop & | - | 84.1/85.6 | - |
| (SOTA [5]) | ImageNet-1k | Extreme Mixup |  |  |  |
| 𝒟3\mathcal{D}_{3} | 10×\times | Inception Crop & | - | 81.8/84.1 | - |
|  | JFT subset | RandAug l2m5 |  |  |  |
| Ours: 𝒟1\mathcal{D}_{1} + 𝒟2\mathcal{D}_{2} |  |  | 1:1 | 84.4/85.0 (++0.3) | - |
| Ours: 𝒟2\mathcal{D}_{2} + 𝒟3\mathcal{D}_{3} |  |  | 1:1 | 84.7/82.7 (++0.6) | - |
| Ours: 𝒟1\mathcal{D}_{1} + 𝒟2\mathcal{D}_{2} + 𝒟3\mathcal{D}_{3} |  |  | 1:1:2 | 84.9/82.6 (++0.8) | 85.9/85.5 (++1.8) |

| Model | IN-1k Only | SOTA | Our | Our Gain Over |
|---|---|---|---|---|
|  | Only | Distill[5] | Distill | IN-1k / SOTA |
| MNv4-Conv-S | 73.8 | - | 75.5 | +1.7 / - |
| MNv4-Conv-M | 79.9 | 81.5 | 82.7 | +2.8 / +1.2 |
| MNv4-Hybrid-M | 80.7 | 82.7 | 83.7 | +3.0 / +1.0 |
| MNv4-Conv-L | 82.9 | 84.4 | 85.9 | +3.0 / +1.5 |
| MNv4-Hybrid-L | 83.4 | 85.7 | 86.6 | +3.2 / +0.9 |

## 9 Conclusion

In this paper, we presented MobileNetV4, a series of universal, efficient models that run efficiently across the mobile ecosystem. Multiple advances make MobileNetV4 mostly-Pareto-optimal on all mobile CPUs, GPUs, DSPs and specialized accelerators, a characteristic not found in any other models tested. We introduced the Universal Inverted Bottleneck and Mobile MQA layers and combined them with improved NAS recipes. With these and a novel, SOTA distillation approach, we achieve 87% ImageNet-1K accuracy at 3.8ms Pixel 8 EdgeTPU latency, setting a new state-of-the-art. Finally, we introduced a framework for understanding model universality on heterogeneous devices. We hope the novel contributions and analysis further spur advances in mobile computer vision.

## Acknowledgements

We appreciate Tammo Spalink, Yeqing Li, Sage Stevens, Bob Muniz, Liviu Panait, David Wood, Lynn Nguyen, and Lucas Beyer for their support while developing this work. We also thank the MLPerf Mobile working group for their feedback and collaboration.66 6 MobileNetV4-Conv-L was used for v4.0 of https://mlcommons.org/benchmarks/inference-mobile/ We particularly thank Ross Wightman for his reimplementation of MobileNetV4 for timm and training recipe improvements.77 7 MobileNetV4 models are available in timm at https://huggingface.co/collections/timm/mobilenetv4-pretrained-weights-6669c22cda4db4244def9637

## Appendix 0.A Search space details

The following is how we construct the search space for NAS.

Search Space Construction:

- • Fixed Initial Layers: We started with a Conv2D layer (3x3 kernel, stride 2) in the first stage for quick resolution reduction, followed by NAS-optimized FusedIB blocks (stride 2) in the second stage to balance between efficiency and accuracy.
Fixed Initial Layers: We started with a Conv2D layer (3x3 kernel, stride 2) in the first stage for quick resolution reduction, followed by NAS-optimized FusedIB blocks (stride 2) in the second stage to balance between efficiency and accuracy.

- • NAS-Driven Optimization: The NAS process precisely determined the ideal number of UIB blocks and parameter instantiations across the remaining four stages, ensuring an optimal structure for performance.
NAS-Driven Optimization: The NAS process precisely determined the ideal number of UIB blocks and parameter instantiations across the remaining four stages, ensuring an optimal structure for performance.

- • Fixed Head Layers: We use the same head layer configuration as MobileNet V3.
Fixed Head Layers: We use the same head layer configuration as MobileNet V3.

Observing that pointwise convolutions within UIB blocks tend to exhibit low operational intensity at higher resolutions, we prioritized operations with higher computational density in the initial layers to balance efficiency and accuracy.

Our optimization targets:

- • MNv4-Conv-S: Dual goals—285M MACs and 0.2ms latency (Pixel 6 EdgeTPU, 224px inputs).
MNv4-Conv-S: Dual goals—285M MACs and 0.2ms latency (Pixel 6 EdgeTPU, 224px inputs).

- • MNv4-Conv-M: 0.6ms latency (Pixel 6 EdgeTPU, 256px inputs).
MNv4-Conv-M: 0.6ms latency (Pixel 6 EdgeTPU, 256px inputs).

- • MNv4-Conv-L: Dual latency goals of 2.3ms (Pixel 6 EdgeTPU) and 2.0ms (Pixel 7 EdgeTPU) with 384px inputs.
MNv4-Conv-L: Dual latency goals of 2.3ms (Pixel 6 EdgeTPU) and 2.0ms (Pixel 7 EdgeTPU) with 384px inputs.

To be noted, by restricting our search space to components with well-correlated cost models across devices, we found that EdgeTPU latency optimization directly yields universally efficient models, as demonstrated in later sections.

## Appendix 0.B Benchmarking methodology

We applied a consistent benchmarking strategy across various mobile platforms, with an exception for the Apple Neural Engine. To enhance efficiency, models were converted to TensorFlow Lite format and quantized to INT8 for mobile CPUs, Hexagon, and EdgeTPUs, while FP16 was used for mobile GPUs. We run each model roughly 1000 times and take the mean latency of those runs. We then repeat that process 5 times for each model and report the median of means. To optimize performance, we set the CPU affinity to the fastest core and use the XNNPACK backend for CPU evaluations. In contrast, for benchmarks on the Apple Neural Engine (conducted on an iPhone 13 with iOS 16.6.1, CoreMLTools 7.1, and Xcode 15.0.1 for profiling), PyTorch models were converted to CoreML’s MLProgram format in Float16 precision, with float16 MultiArray inputs to minimize input copying.

## Appendix 0.C Training setup for ImageNet-1k classification

To enhance model performance, our training recipe incorporates widely adopted data augmentation techniques and regularization methods. For data augmentation, we use Inception Crop [41], horizontal flip, RandAugment [10], Mixup [55], and CutMix [54]. For regularization, we apply L2 normalization and stochastic depth drop [23]. The intensity of augmentation and regularization is adjusted according to model size, as detailed in Tab. 10.

|  | Conv-S | Conv-M | Hybrid-M | Conv-L | Hybrid-L |
|---|---|---|---|---|---|
| Batch size | 4096 | 4096 | 16384 | 16384 | 16384 |
| Peak learning rate | 0.002 | 0.004 | 0.016 | 0.004 | 0.01 |
| Cosine decay alpha | 0.0 | 0.0 | 0.0 | 0.0 | 0.001 |
| Cosine decay epochs | 9600 | 500 | 500 | 500 | 500 |
| Warm-up epochs | 5 | 5 | 20 | 20 | 20 |
| Training epochs | 9600 | 500 | 500 | 500 | 500 |
| AdamW weight decay | 0.01 | 0.1 | 0.1 | 0.2 | 0.2 |
| AdamW β1\beta_{1} | 0.6 | 0.9 | 0.9 | 0.9 | 0.9 |
| AdamW β2\beta_{2} | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| AdamW ϵ\epsilon | 10−610^{-6} | 10−710^{-7} | 10−710^{-7} | 10−710^{-7} | 10−710^{-7} |
| EMA decay | 0.9999 | - | - | - | - |
| L2-regularization | 10−510^{-5} | - | - | - | - |
| Gradient clipping | - | - | - | - | - |
| Label smoothing | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| Dropout | 0.3 | 0.2 | 0.2 | 0.2 | 0.2 |
| Peak Stochastic Depth drop rate | 0 | 0.075 | 0.075 | 0.35 | 0.35 |
| RandAugment probability | 0.5 | 0.7 | 0.7 | 1.0 | 1.0 |
| RandAugment layers | 2 | 2 | 2 | 2 | 2 |
| RandAugment magnitude | 9 | 15 | 15 | 15 | 15 |
| RandAugment excluded ops | Cutout | Cutout | Cutout | Cutout | Cutout |
| Mixup/Cutmix probability | - | - | - | 0.3 | 0.3 |
| Mixup α\alpha | - | - | - | 0.8 | 0.8 |
| Cutmix α\alpha | - | - | - | 1.0 | 1.0 |
| Mixup/Cutmix switch probability | - | - | - | 0.5 | 0.5 |

## Appendix 0.D Model details

The architecture details of our MNv4 models are described from Tab. 11 to Tab. 15.

Now, let’s examine the details of the TuNAS-optimized MNv4-Conv models. TuNAS optimized macro architecture strategically combines four UIB instantiations: Extra DW, ConvNext, IB, and FFN. This combination demonstrates the flexibility of UIB and the importance of using different instantiation blocks in different stages of the network. Specifically, at the start of each searchable stage, where the spatial resolution significantly drops, ExtraDW emerges as the preferred choice. The design of duo depthwise layers in ExtraDW helps to enlarge the receptive field, enhances spatial mixing, and effectively mitigates resolution loss. Similarly, ExtraDW is frequently selected in the early stages of MNv4-Conv models for similar reasons. For the final layers, where preceding layers have conducted substantial spatial mixing, FFN and ConvNext are chosen because channel mixing provides a larger incremental gain.

| Input | Block | DW K1K_{1} | DW K2K_{2} | Expanded Dim | Output Dim | Stride |
|---|---|---|---|---|---|---|
| 2242×3224^{2}\times 3 | Conv2D | - | 3×33\times 3 | - | 32 | 2 |
| 1122×32112^{2}\times 32 | FusedIB | - | 3×33\times 3 | 32 | 32 | 2 |
| 562×3256^{2}\times 32 | FusedIB | - | 3×33\times 3 | 96 | 64 | 2 |
| 282×6428^{2}\times 64 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 192 | 96 | 2 |
| 142×9614^{2}\times 96 | IB | - | 3×33\times 3 | 192 | 96 | 1 |
| 142×9614^{2}\times 96 | IB | - | 3×33\times 3 | 192 | 96 | 1 |
| 142×9614^{2}\times 96 | IB | - | 3×33\times 3 | 192 | 96 | 1 |
| 142×9614^{2}\times 96 | IB | - | 3×33\times 3 | 192 | 96 | 1 |
| 142×9614^{2}\times 96 | ConvNext | 3×33\times 3 | - | 384 | 96 | 1 |
| 142×9614^{2}\times 96 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 576 | 128 | 2 |
| 72×1287^{2}\times 128 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 512 | 128 | 1 |
| 72×1287^{2}\times 128 | IB | - | 5×55\times 5 | 512 | 128 | 1 |
| 72×1287^{2}\times 128 | IB | - | 5×55\times 5 | 384 | 128 | 1 |
| 72×1287^{2}\times 128 | IB | - | 3×33\times 3 | 512 | 128 | 1 |
| 72×1287^{2}\times 128 | IB | - | 3×33\times 3 | 512 | 128 | 1 |
| 72×1287^{2}\times 128 | Conv2D | - | 1×11\times 1 | - | 960 | 1 |
| 72×9607^{2}\times 960 | AvgPool | - | 7×77\times 7 | - | 960 | 1 |
| 12×9601^{2}\times 960 | Conv2D | - | 1×11\times 1 | - | 1280 | 1 |
| 12×12801^{2}\times 1280 | Conv2D | - | 1×11\times 1 | - | 1000 | 1 |

| Input | Block | DW K1K_{1} | DW K2K_{2} | Expanded Dim | Output Dim | Stride |
|---|---|---|---|---|---|---|
| 2562×3256^{2}\times 3 | Conv2D | - | 3×33\times 3 | - | 32 | 2 |
| 1282×32128^{2}\times 32 | FusedIB | - | 3×33\times 3 | 128 | 48 | 2 |
| 642×4864^{2}\times 48 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 192 | 80 | 2 |
| 322×8032^{2}\times 80 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 160 | 80 | 1 |
| 322×8032^{2}\times 80 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 480 | 160 | 2 |
| 162×16016^{2}\times 160 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 640 | 160 | 1 |
| 162×16016^{2}\times 160 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 640 | 160 | 1 |
| 162×16016^{2}\times 160 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 640 | 160 | 1 |
| 162×16016^{2}\times 160 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 640 | 160 | 1 |
| 162×16016^{2}\times 160 | ConvNext | 3×33\times 3 | - | 640 | 160 | 1 |
| 162×16016^{2}\times 160 | FFN | - | - | 320 | 160 | 1 |
| 162×16016^{2}\times 160 | ConvNext | 3×33\times 3 | - | 640 | 160 | 1 |
| 162×16016^{2}\times 160 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 960 | 256 | 2 |
| 82×2568^{2}\times 256 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | FFN | - | - | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | ConvNext | 3×33\times 3 | - | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 512 | 256 | 1 |
| 82×2568^{2}\times 256 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | FFN | - | - | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | FFN | - | - | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | ConvNext | 5×55\times 5 | - | 512 | 256 | 1 |
| 82×2568^{2}\times 256 | Conv2D | - | 1×11\times 1 | - | 960 | 1 |
| 82×9608^{2}\times 960 | AvgPool | - | 8×88\times 8 | - | 960 | 1 |
| 12×9601^{2}\times 960 | Conv2D | - | 1×11\times 1 | - | 1280 | 1 |
| 12×12801^{2}\times 1280 | Conv2D | - | 1×11\times 1 | - | 1000 | 1 |

| Input | Block | DW K1K_{1} | DW K2K_{2} | Expanded Dim | Output Dim | Stride |
|---|---|---|---|---|---|---|
| 2562×3256^{2}\times 3 | Conv2D | - | 3×33\times 3 | - | 32 | 2 |
| 1282×32128^{2}\times 32 | FusedIB | - | 3×33\times 3 | 128 | 48 | 2 |
| 642×4864^{2}\times 48 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 192 | 80 | 2 |
| 322×8032^{2}\times 80 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 160 | 80 | 1 |
| 322×8032^{2}\times 80 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 480 | 160 | 2 |
| 162×16016^{2}\times 160 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 640 | 160 | 1 |
| 162×16016^{2}\times 160 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 640 | 160 | 1 |
| 162×16016^{2}\times 160 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 640 | 160 | 1 |
| 162×16016^{2}\times 160 | Mobile-MQA | - | - | - | 160 | 1 |
| 162×16016^{2}\times 160 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 640 | 160 | 1 |
| 162×16016^{2}\times 160 | Mobile-MQA | - | - | - | 160 | 1 |
| 162×16016^{2}\times 160 | ConvNext | 3×33\times 3 | - | 640 | 160 | 1 |
| 162×16016^{2}\times 160 | Mobile-MQA | - | - | - | 160 | 1 |
| 162×16016^{2}\times 160 | FFN | - | - | 640 | 160 | 1 |
| 162×16016^{2}\times 160 | Mobile-MQA | - | - | - | 160 | 1 |
| 162×16016^{2}\times 160 | ConvNext | 3×33\times 3 | - | 640 | 160 | 1 |
| 162×16016^{2}\times 160 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 960 | 256 | 2 |
| 82×2568^{2}\times 256 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | FFN | - | - | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | ConvNext | 3×33\times 3 | - | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 512 | 256 | 1 |
| 82×2568^{2}\times 256 | Mobile-MQA | - | - | - | 256 | 1 |
| 82×2568^{2}\times 256 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | Mobile-MQA | - | - | - | 256 | 1 |
| 82×2568^{2}\times 256 | FFN | - | - | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | Mobile-MQA | - | - | - | 256 | 1 |
| 82×2568^{2}\times 256 | FFN | - | - | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | Mobile-MQA | - | - | - | 256 | 1 |
| 82×2568^{2}\times 256 | ConvNext | 5×55\times 5 | - | 1024 | 256 | 1 |
| 82×2568^{2}\times 256 | Conv2D | - | 1×11\times 1 | - | 960 | 1 |
| 82×9608^{2}\times 960 | AvgPool | - | 8×88\times 8 | - | 960 | 1 |
| 12×9601^{2}\times 960 | Conv2D | - | 1×11\times 1 | - | 1280 | 1 |
| 12×12801^{2}\times 1280 | Conv2D | - | 1×11\times 1 | - | 1000 | 1 |

| Input | Block | DW K1K_{1} | DW K2K_{2} | Expanded Dim | Output Dim | Stride |
|---|---|---|---|---|---|---|
| 3842×3384^{2}\times 3 | Conv2D | - | 3×33\times 3 | - | 24 | 2 |
| 1922×24192^{2}\times 24 | FusedIB | - | 3×33\times 3 | 96 | 48 | 2 |
| 962×4896^{2}\times 48 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 192 | 96 | 2 |
| 482×9648^{2}\times 96 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 384 | 96 | 1 |
| 482×9648^{2}\times 96 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 384 | 192 | 2 |
| 242×19224^{2}\times 192 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ConvNext | 3×33\times 3 | - | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 768 | 512 | 2 |
| 122×51212^{2}\times 512 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ConvNext | 5×55\times 5 | - | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ConvNext | 5×55\times 5 | - | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ConvNext | 5×55\times 5 | - | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ConvNext | 5×55\times 5 | - | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ConvNext | 5×55\times 5 | - | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ConvNext | 5×55\times 5 | - | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | Conv2D | - | 1×11\times 1 | - | 960 | 1 |
| 122×96012^{2}\times 960 | AvgPool | - | 12×1212\times 12 | - | 960 | 1 |
| 12×9601^{2}\times 960 | Conv2D | - | 1×11\times 1 | - | 1280 | 1 |
| 12×12801^{2}\times 1280 | Conv2D | - | 1×11\times 1 | - | 1000 | 1 |

| Input | Block | DW K1K_{1} | DW K2K_{2} | Expanded Dim | Output Dim | Stride |
|---|---|---|---|---|---|---|
| 3842×3384^{2}\times 3 | Conv2D | - | 3×33\times 3 | - | 24 | 2 |
| 1922×24192^{2}\times 24 | FusedIB | - | 3×33\times 3 | 96 | 48 | 2 |
| 962×4896^{2}\times 48 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 192 | 96 | 2 |
| 482×9648^{2}\times 96 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 384 | 96 | 1 |
| 482×9648^{2}\times 96 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 384 | 192 | 2 |
| 242×19224^{2}\times 192 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 3×33\times 3 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 3×33\times 3 | 5×55\times 5 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | Mobile-MQA | - | - | - | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | Mobile-MQA | - | - | - | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | Mobile-MQA | - | - | - | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | Mobile-MQA | - | - | - | 192 | 1 |
| 242×19224^{2}\times 192 | ConvNext | 3×33\times 3 | - | 768 | 192 | 1 |
| 242×19224^{2}\times 192 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 768 | 512 | 2 |
| 122×51212^{2}\times 512 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ConvNext | 5×55\times 5 | - | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ConvNext | 5×55\times 5 | - | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ConvNext | 5×55\times 5 | - | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ExtraDW | 5×55\times 5 | 3×33\times 3 | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | ExtraDW | 5×55\times 5 | 5×55\times 5 | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | Mobile-MQA | - | - | - | 512 | 1 |
| 122×51212^{2}\times 512 | ConvNext | 5×55\times 5 | - | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | Mobile-MQA | - | - | - | 512 | 1 |
| 122×51212^{2}\times 512 | ConvNext | 5×55\times 5 | - | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | Mobile-MQA | - | - | - | 512 | 1 |
| 122×51212^{2}\times 512 | ConvNext | 5×55\times 5 | - | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | Mobile-MQA | - | - | - | 512 | 1 |
| 122×51212^{2}\times 512 | ConvNext | 5×55\times 5 | - | 2048 | 512 | 1 |
| 122×51212^{2}\times 512 | Conv2D | - | 1×11\times 1 | - | 960 | 1 |
| 122×96012^{2}\times 960 | AvgPool | - | 12×1212\times 12 | - | 960 | 1 |
| 12×9601^{2}\times 960 | Conv2D | - | 1×11\times 1 | - | 1280 | 1 |
| 12×12801^{2}\times 1280 | Conv2D | - | 1×11\times 1 | - | 1000 | 1 |

## Appendix 0.E Larger Pareto curve

## Appendix 0.F Additional Roofline Analysis

This extends the analysis from Fig. 3 to include MobileNetV4-Conv-Small (Fig. 7), MobileNetV4-Conv-Medium (Fig. 8), and MobileNetV4-Conv-Large (Fig. 9). These figures also break out each order of magnitude in the sweep from a 0.0 MACs/byte ridge point (MACs-only, infinite memory bandwidth) to a 500.0 MACs/byte ridge point (Accelerator-like, bottlenecked on memory bandwidth). Also included is a correlation analysis between measured latencies, empirically-fit roofline models, and counting MACs (Tab. 16 and Fig. 6).

|  | Ridge Point |  |  |
|---|---|---|---|
| Execution Target | (MACs/B) | rsr_{s}-Roofline | rsr_{s}-MAC |
| Pixel 6 CPU (Int8) | 31.2 | 0.973 | 0.962 |
| Samsung Galaxy S23 CPU (Int8) | 39.7 | 0.962 | 0.940 |
| Pixel 4 DSP (Int8) | 347.3 | 0.962 | 0.758 |
| Pixel 8 EdgeTPU (Int8) | 433.8 | 0.973 | 0.857 |

## Appendix 0.G Einsum Optimization

MQA, while faster than MHSA, is still 12x slower than UIB for the same MACs. In the following, we investigate MQA’s implementation to uncovers performance issues, and introduce Mobile MQA to improve efficiency.

Einstein summation (Einsum), extensively used in MQA and MHSA implementations across TensorFlow Keras, PyTorch, and JAX, can obscure underlying computational inefficiencies. When executed on-device, Einsum operations are decomposed into sequences of tensor transposes, reshapes, and batched matrix multiplications, aiming to minimize MACs. However, transposes, despite not involving MACs, are highly resource-intensive due to requiring complete tensor reads and writes in memory, significantly impacting performance. Key inefficiencies in Einsum execution include:

Contracted and non-contracted indices are not contiguous in the input: Mobile inference generally does not use a batch dimension. Accordingly an Einsum with two inputs can be translated to a matrix multiplication if both the contracted and non-contracted indices from each input are adjacent to each other. This is because adjacent indices can be very cheaply reshaped to a single index for the purpose of running the matrix multiplication and then very cheaply reshaped back. If the indices are not cleanly split into a set of contracting indices and non-contracting indices, then transposition operations are needed to bring them into this form for matrix multiplication. In the example below, for the slow implementation, two transposes must be introduced: one to transpose OO, and one to transpose PoP_{o}.

Interleaving non-contracting indices or reordering indices in the output: In the slow implementation of the following example, the non-contracted indices hh and kk from PqP_{q} are interleaved with index nn from XX. As a result, two transposes will be introduced, one before the matrix multiplication to transpose PqP_{q}, and one afterwards to swap the indices in the output tensor.

In Figure 11, we provide the pseudo-code details of our optimized Mobile MQA and compare them with the original MQA in Figure 10. Mobile MQA achieves significant efficiency improvements by addressing the two main inefficiency issues identified earlier. It requires only a few line of code changes to reorder the indices in the EinSum computations and weight tensor definitions in MQA to eliminate unnecessary transposes. Table 17 demonstrates that we reduced the latency of MQA by threefold on Pixel 7 and by 13.8% on Pixel 8. Additionally, we observed an almost threefold training speedup on Google Tensor v4 using the tf.keras training framework.

| model | Top-1 | MACs | Params | EdgeTPU |  |
|---|---|---|---|---|---|
|  | Acc(%) | (G) | (M) | Pixel 7 | Pixel 8 |
| base model | 84.88 | 6.0 | 30.9 | 4.31 ms | 2.35 ms |
| +3 MQA | 85.22 | 6.5 | 34.7 | 7.00 ms | 2.64 ms |
| (w/o Einsum optimization) |  |  |  |  |  |
| +3 Mobile MQA | 85.24 | 6.5 | 34.7 | 5.16 ms | 2.60 ms |
| (w/ Einsum optimization) |  |  |  | (-68.4%) | (-13.8%) |

## Appendix 0.H Implementation details of our distillation recipe

To create the offline distillation dataset, we first augment a candidate image and then compress it using JPEG encoding. We then decode the image and run inference using the teacher. The teacher’s predicted probability (the softmax output) of the 1k classes is recorded and will be used later as soft labels to train the student. We use EfficientNet L2 [51] as the teacher model. This teacher model has 480 M parameters and 290 G MACs. It achieves 87.5% top-1 accuracy on ImageNet 1k.

During student training, we disable all data augmentation except left-right flip. We also apply dropout in the final fully connected layer as the only regularization method. We use cross-entropy between the teacher’s soft labels and the student’s predicted class probability vectors as our training loss. We train the student with the AdamW optimizer. We use a cosine learning rate schedule with warm-up.

Besides the significant boost in model accuracy, there are other benefits of using distillation training. For example, we found that distillation training is 2.5x faster than training on ImageNet-1k. With a batch size of 16k and 400 training epochs, distillation training on MNv4-Conv-Large takes only 2.3 hours on 128 TPU v5e. Finally, distillation training provides significant relief from hyper-parameter tuning.
