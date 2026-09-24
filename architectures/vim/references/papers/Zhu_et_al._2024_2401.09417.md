# Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model

Recently the state space models (SSMs) with efficient hardware-aware designs, i.e., the Mamba deep learning model, have shown great potential for long sequence modeling. Meanwhile building efficient and generic vision backbones purely upon SSMs is an appealing direction. However, representing visual data is challenging for SSMs due to the position-sensitivity of visual data and the requirement of global context for visual understanding. In this paper, we show that the reliance on self-attention for visual representation learning is not necessary and propose a new generic vision backbone with bidirectional Mamba blocks (Vim), which marks the image sequences with position embeddings and compresses the visual representation with bidirectional state space models. On ImageNet classification, COCO object detection, and ADE20k semantic segmentation tasks, Vim achieves higher performance compared to well-established vision transformers like DeiT, while also demonstrating significantly improved computation & memory efficiency. For example, Vim is 2.8×\times faster than DeiT and saves 86.8% GPU memory when performing batch inference to extract features on images with a resolution of 1248×\times1248. The results demonstrate that Vim is capable of overcoming the computation & memory constraints on performing Transformer-style understanding for high-resolution images and it has great potential to be the next-generation backbone for vision foundation models. Code and models are released at https://github.com/hustvl/Vim

## 1 Introduction

Recent research advancements have led to a surge of interest in the state space model (SSM). Originating from the classic Kalman filter model (Kalman, 1960), modern SSMs excel at capturing long-range dependencies and benefit from parallel training. Some SSM-based methods, such as the linear state-space layers (LSSL) (Gu et al., 2021b), structured state space sequence model (S4) (Gu et al., 2021a), diagonal state space (DSS) (Gupta et al., 2022), and S4D (Gu et al., 2022), are proposed to process sequence data across a wide range of tasks and modalities, particularly on modeling long-range dependencies. They are efficient in processing long sequences because of convolutional computation and near-linear computation. 2-D SSM (Baron et al., 2023), SGConvNeXt (Li et al., 2022b), and ConvSSM (Smith et al., 2023a) combine SSM with CNN or Transformer architecture to process 2-D data. The recent work, Mamba (Gu & Dao, 2023), incorporates time-varying parameters into the SSM and proposes a hardware-aware algorithm to enable very efficient training and inference. The superior scaling performance of Mamba indicates that it is a promising alternative to Transformer in language modeling. Nevertheless, a generic pure-SSM-based backbone network has not been explored for processing visual data, such as images and videos.

Vision Transformers (ViTs) have achieved great success in visual representation learning, excelling in large-scale self-supervised pre-training and high performance on downstream tasks. Compared with convolutional neural networks, the core advantage lies in that ViT can provide each image patch with data/patch-dependent global context through self-attention. This differs from convolutional networks that use the same parameters, i.e., the convolutional filters, for all positions. Another advantage is the modality-agnostic modeling by treating an image as a sequence of patches without 2D inductive bias, which makes it the preferred architecture for multimodal applications (Bavishi et al., 2023; Li et al., 2023; Liu et al., 2023). At the same time, the self-attention mechanism in Transformers poses challenges in terms of speed and memory usage when dealing with long-range visual dependencies, e.g., processing high-resolution images.

Motivated by the success of Mamba in language modeling, it is appealing that we can also transfer this success from language to vision, i.e., to design a generic and efficient visual backbone with the advanced SSM method. However, there are two challenges for Mamba, i.e., unidirectional modeling and lack of positional awareness. To address these challenges, we propose the Vision Mamba (Vim) model, which incorporates the bidirectional SSMs for data-dependent global visual context modeling and position embeddings for location-aware visual recognition. We first split the input image into patches and linearly project them as vectors to Vim. Image patches are treated as the sequence data in Vim blocks, which efficiently compresses the visual representation with the proposed bidirectional selective state space. Furthermore, the position embedding in Vim block provides the awareness for spatial information, which enables Vim to be more robust in dense prediction tasks. In the current stage, we train the Vim model on the supervised image classification task using the ImageNet dataset and then use the pretrained Vim as the backbone to perform sequential visual representation learning for downstream dense prediction tasks, i.e., semantic segmentation, object detection, and instance segmentation. Like Transformers, Vim can be pretrained on large-scale unsupervised visual data for better visual representation. Thanks to the better efficiency of Mamba, the large-scale pretraining of Vim can be achieved with lower computational cost.

Compared with other SSM-based models for vision tasks, Vim is a pure-SSM-based method and models images in a sequence manner, which is more promising for a generic and efficient backbone. Thanks to the bidirectional compressing modeling with positional awareness, Vim is the first pure-SSM-based model to handle dense prediction tasks. Compared with the most convincing Transformer-based model, i.e., DeiT (Touvron et al., 2021a), Vim achieves superior performance on ImageNet classification. Furthermore, Vim is more efficient in terms of GPU memory and inference time for high-resolution images. The efficiency in terms of memory and speed empowers Vim to directly perform sequential visual representation learning without relying on 2D priors (such as the 2D local window in ViTDet (Li et al., 2022c)) for high-resolution visual understanding tasks while achieving higher accuracy than DeiT.

Our main contributions can be summarized as follows:

- • We propose Vision Mamba (Vim), which incorporates bidirectional SSM for data-dependent global visual context modeling and position embeddings for location-aware visual understanding.
We propose Vision Mamba (Vim), which incorporates bidirectional SSM for data-dependent global visual context modeling and position embeddings for location-aware visual understanding.

- • Without the need of attention, the proposed Vim has the same modeling power as ViT while it only has subquadratic-time computation and linear memory complexity. Specifically, Vim is 2.8×\times faster than DeiT and saves 86.8% GPU memory when performing batch inference to extract features on images at the resolution of 1248×\times1248.
Without the need of attention, the proposed Vim has the same modeling power as ViT while it only has subquadratic-time computation and linear memory complexity. Specifically, Vim is 2.8×\times faster than DeiT and saves 86.8% GPU memory when performing batch inference to extract features on images at the resolution of 1248×\times1248.

- • We conduct extensive experiments on ImageNet classification and dense prediction downstream tasks. The results demonstrate that Vim achieves superior performance compared to the well-established and highly-optimized plain vision Transformer, i.e., DeiT.
We conduct extensive experiments on ImageNet classification and dense prediction downstream tasks. The results demonstrate that Vim achieves superior performance compared to the well-established and highly-optimized plain vision Transformer, i.e., DeiT.

## 2 Related Work

Architectures for generic vision backbone. In the early eras, ConvNet (LeCun et al., 1998) serves as the de-facto standard network design for computer vision. Many convolutional neural architectures (Krizhevsky et al., 2012; Szegedy et al., 2015; Simonyan & Zisserman, 2014; He et al., 2016; Tan & Le, 2019; Wang et al., 2020a; Huang et al., 2017; Xie et al., 2017; Tan & Le, 2021; Radosavovic et al., 2020) have been proposed as the vision backbone for various visual applications. The pioneering work, Vision Transformer (ViT) (Dosovitskiy et al., 2020) changes the landscape. It treats an image as a sequence of flattened 2D patches and directly applies a pure Transformer architecture. The surprising results of ViT on image classification and its scaling ability encourage a lot of follow-up works (Touvron et al., 2021b; Tolstikhin et al., 2021; Touvron et al., 2022; Fang et al., 2022). One line of works focuses on hybrid architecture designs by introducing 2D convolutional priors into ViT (Wu et al., 2021; Dai et al., 2021; d’Ascoli et al., 2021; Dong et al., 2022). PVT (Wang et al., 2021) proposes a pyramid structure Transformer. Swin Transformer (Liu et al., 2021) applies self-attention within shift windows. Another line of works focuses on improving traditional 2D ConvNets with more advanced settings (Wang et al., 2023b; Liu et al., 2022a). ConvNeXt (Liu et al., 2022b) reviews the design space and proposes pure ConvNets, which can be scalable as ViT and its variants. RepLKNet (Ding et al., 2022) proposes to scale up the kernel size of existing ConvNets to bring improvements.

Though these dominant follow-up works demonstrate superior performance and better efficiency on ImageNet (Deng et al., 2009) and various downstream tasks (Lin et al., 2014; Zhou et al., 2019) by introducing 2D priors, with the surge of large-scale visual pretraining (Bao et al., 2022; Fang et al., 2023; Caron et al., 2021) and multi-modality applications (Radford et al., 2021; Li et al., 2022a; Li et al., 2023; Liu et al., 2023; Bavishi et al., 2023; Jia et al., 2021), vanilla Transformer-style model strikes back to the center stage of computer vision. The advantages of larger modeling capacity, unified multi-modality representation, being friendly to self-supervised learning etc., make it the preferred architecture. However, the number of visual tokens is limited due to the quadratic complexity of Transformer. There are plenty of works (Choromanski et al., 2021; Wang et al., 2020b; Kitaev et al., 2020; Child et al., 2019; Ding et al., 2023; Qin et al., 2023; Sun et al., 2023) to address this long-standing and prominent challenge, but few of them focus on visual applications. Recently, LongViT (Wang et al., 2023c) built an efficient Transformer architecture for computational pathology applications via dilated attention. The linear computation complexity of LongViT allows it to encode the extremely long visual sequence. In this work, we draw inspiration from Mamba (Gu & Dao, 2023) and explore building a pure-SSM-based model as a generic vision backbone without using attention, while preserving the sequential, modality-agnostic modeling merit of ViT.

State space models for long sequence modeling. (Gu et al., 2021a) proposes a Structured State-Space Sequence (S4) model, a novel alternative to CNNs or Transformers, to model the long-range dependency. The promising property of linearly scaling in sequence length attracts further explorations. (Wang et al., 2022) proposes Bidirectional Gated SSM to replicate BERT (Devlin et al., 2018) results without attention. (Smith et al., 2023b) proposes a new S5 layer by introducing MIMO SSM and efficient parallel scan into S4 layer. (Fu et al., 2023) designs a new SSM layer, H3, that nearly fills the performance gap between SSMs and Transformer attention in language modeling. (Mehta et al., 2023) builds the Gated State Space layer on S4 by introducing more gating units to improve the expressivity. Recently, (Gu & Dao, 2023) proposes a data-dependent SSM layer and builds a generic language model backbone, Mamba, which outperforms Transformers at various sizes on large-scale real data and enjoys linear scaling in sequence length. In this work, we explore transferring the success of Mamba to vision, i.e., building a generic vision backbone purely upon SSM without attention.

State space models for visual applications. (Islam & Bertasius, 2022) uses 1D S4 to handle the long-range temporal dependencies for video classification. (Nguyen et al., 2022) further extends 1D S4 to handle multi-dimensional data including 2D images and 3D videos. (Islam et al., 2023) combines the strengths of S4 and self-attention to build TranS4mer model, achieving state-of-the-art performance for movie scene detection. (Wang et al., 2023a) introduces a novel selectivity mechanism to S4, largely improving the performance of S4 on long-form video understanding with a much lower memory footprint. (Yan et al., 2023) supplants attention mechanisms with a more scalable SSM-based backbone to generate high-resolution images and process fine-grained representation under affordable computation. (Ma et al., 2024) proposes U-Mamba, a hybrid CNN-SSM architecture, to handle the long-range dependencies in biomedical image segmentation. The above works (Xing et al., 2024; Ma et al., 2024; Yan et al., 2023; Wang et al., 2023a; Islam et al., 2023; Nguyen et al., 2022; Islam & Bertasius, 2022) either apply SSM to specific visual applications or build a hybrid architecture by combining SSM with convolution or attention. Different from them, we build a pure-SSM-based model, which can be adopted as a generic vision backbone. It is noteworthy that VMamba (Liu et al., 2024), a concurrent work with our method, has demonstrated impressive results in visual recognition by incorporating Mamba with multi-directional scanning and a hierarchical network architecture. In contrast, Vim primarily concentrates on visual sequence learning and boasts a unified representation for multi-modality data.

## 3 Method

The goal of Vision Mamba (Vim) is to introduce the advanced state space model (SSM), i.e., Mamba (Gu & Dao, 2023), to computer vision. This section begins with a description of the preliminaries of SSM. It is followed by an overview of Vim. We then detail how the Vim block processes input token sequences and proceed to illustrate the architecture details of Vim. The section concludes with an analysis of the efficiency of the proposed Vim.

### 3.1 Preliminaries

The SSM-based models, i.e., structured state space sequence models (S4) and Mamba are inspired by the continuous system, which maps a 1-D function or sequence x⁡(t)∈ℝ↦y⁡(t)∈ℝx(t)\in\mathbb{R}\mapsto y(t)\in\mathbb{R} through a hidden state h⁡(t)∈ℝ𝙽h(t)\in\mathbb{R}^{\mathtt{N}}. This system uses 𝐀∈ℝ𝙽×𝙽\mathbf{A}\in\mathbb{R}^{\mathtt{N}\times\mathtt{N}} as the evolution parameter and 𝐁∈ℝ𝙽×1\mathbf{B}\in\mathbb{R}^{\mathtt{N}\times 1}, 𝐂∈ℝ1×𝙽\mathbf{C}\in\mathbb{R}^{1\times\mathtt{N}} as the projection parameters. The continuous system works as follows: h′​(t)=𝐀​h​(t)+𝐁​x​(t)h^{\prime}(t)=\mathbf{A}h(t)+\mathbf{B}x(t) and y⁡(t)=𝐂​h​(t)y(t)=\mathbf{C}h(t).

The S4 and Mamba are the discrete versions of the continuous system, which include a timescale parameter 𝚫\mathbf{\Delta} to transform the continuous parameters 𝐀\mathbf{A}, 𝐁\mathbf{B} to discrete parameters 𝐀¯\mathbf{\overline{A}}, 𝐁¯\mathbf{\overline{B}}. The commonly used method for transformation is zero-order hold (ZOH), which is defined as follows:

|  | 𝐀¯\displaystyle\mathbf{\overline{A}} | =exp⁡(𝚫​𝐀),\displaystyle=\exp{(\mathbf{\Delta}\mathbf{A})}, |  | (1) |
|---|---|---|---|---|
|  | 𝐁¯\displaystyle\mathbf{\overline{B}} | =(𝚫​𝐀)−1​(exp⁡(𝚫​𝐀)−𝐈)⋅𝚫​𝐁.\displaystyle=(\mathbf{\Delta}\mathbf{A})^{-1}(\exp{(\mathbf{\Delta}\mathbf{A})}-\mathbf{I})\cdot\mathbf{\Delta}\mathbf{B}. |  |  |

After the discretization of 𝐀¯\mathbf{\overline{A}}, 𝐁¯\mathbf{\overline{B}}, the discretized version using a step size 𝚫\mathbf{\Delta} can be rewritten as:

|  | ht\displaystyle h_{t} | =𝐀¯​ht−1+𝐁¯​xt,\displaystyle=\mathbf{\overline{A}}h_{t-1}+\mathbf{\overline{B}}x_{t}, |  | (2) |
|---|---|---|---|---|
|  | yt\displaystyle y_{t} | =𝐂​ht.\displaystyle=\mathbf{C}h_{t}. |  |  |

At last, the models compute output through a global convolution.

|  | 𝐊¯\displaystyle\mathbf{\overline{K}} | =(𝐂​𝐁¯,𝐂​𝐀¯​𝐁¯,…,𝐂​𝐀¯𝙼−1​𝐁¯),\displaystyle=(\mathbf{C}\mathbf{\overline{B}},\mathbf{C}\mathbf{\overline{A}}\mathbf{\overline{B}},\dots,\mathbf{C}\mathbf{\overline{A}}^{\mathtt{M}-1}\mathbf{\overline{B}}), |  | (3) |
|---|---|---|---|---|
|  | 𝐲\displaystyle\mathbf{y} | =𝐱∗𝐊¯,\displaystyle=\mathbf{x}*\mathbf{\overline{K}}, |  |  |

where 𝙼\mathtt{M} is the length of the input sequence 𝐱\mathbf{x}, and 𝐊¯∈ℝ𝙼\overline{\mathbf{K}}\in\mathbb{R}^{\mathtt{M}} is a structured convolutional kernel.

### 3.2 Vision Mamba

An overview of the proposed Vim is shown in Fig. 2. The standard Mamba is designed for the 1-D sequence. To process the vision tasks, we first transform the 2-D image 𝐭∈ℝ𝙷×𝚆×𝙲\mathbf{t}\in\mathbb{R}^{\mathtt{H}\times\mathtt{W}\times\mathtt{C}} into the flattened 2-D patches 𝐱𝐩∈ℝ𝙹×(𝙿2⋅𝙲)\mathbf{x_{p}}\in\mathbb{R}^{\mathtt{J}\times(\mathtt{P}^{2}\cdot\mathtt{C})}, where (𝙷,𝚆)(\mathtt{H},\mathtt{W}) is the size of input image, 𝙲\mathtt{C} is the number of channels, 𝙿\mathtt{P} is the size of image patches. Next, we linearly project the 𝐱𝐩\mathbf{x_{p}} to the vector with size 𝙳\mathtt{D} and add position embeddings 𝐄p​o​s∈ℝ(𝙹+1)×𝙳\mathbf{E}_{pos}\in\mathbb{R}^{(\mathtt{J}+1)\times\mathtt{D}}, as follows:

|  | 𝐓0\displaystyle\mathbf{T}_{0} | =[𝐭c​l​s;𝐭p1​𝐖;𝐭p2​𝐖;⋯;𝐭p𝙹​𝐖]+𝐄p​o​s,\displaystyle=[\mathbf{t}_{cls};\mathbf{t}_{p}^{1}\mathbf{W};\mathbf{t}_{p}^{2}\mathbf{W};\cdots;\mathbf{t}_{p}^{\mathtt{J}}\mathbf{W}]+\mathbf{E}_{pos}, |  | (4) |
|---|---|---|---|---|

where 𝐭p𝚓\mathbf{t}_{p}^{\mathtt{j}} is the 𝚓\mathtt{j}-th patch of 𝐭\mathbf{t}, 𝐖∈ℝ(𝙿2⋅𝙲)×𝙳\mathbf{W}\in\mathbb{R}^{(\mathtt{P}^{2}\cdot\mathtt{C})\times\mathtt{D}} is the learnable projection matrix. Inspired by ViT (Dosovitskiy et al., 2020) and BERT (Kenton & Toutanova, 2019), we also use class token to represent the whole patch sequence, which is denoted as 𝐭c​l​s\mathbf{t}_{cls}. We then send the token sequence (𝐓𝚕−1\mathbf{T}_{\mathtt{l}-1}) to the 𝚕\mathtt{l}-th layer of the Vim encoder, and get the output 𝐓𝚕\mathbf{T}_{\mathtt{l}}. Finally, we normalize the output class token 𝐓𝙻0\mathbf{T}_{\mathtt{L}}^{0} and feed it to the multi-layer perceptron (MLP) head to get the final prediction p^\hat{p}, as follows: 𝐓l=𝐕𝐢𝐦⁡(𝐓𝚕−1)+𝐓𝚕−1\mathbf{T}_{l}=\mathbf{Vim{}}(\mathbf{T}_{\mathtt{l}-1})+\mathbf{T}_{\mathtt{l}-1}, 𝐟=𝐍𝐨𝐫𝐦⁡(𝐓𝙻0)\mathbf{f}=\mathbf{Norm}(\mathbf{T}_{\mathtt{L}}^{0}), and p^=𝐌𝐋𝐏⁡(𝐟)\hat{p}=\mathbf{MLP}(\mathbf{f}), where 𝐕𝐢𝐦\mathbf{Vim{}} is the proposed vision mamba block, 𝙻\mathtt{L} is the number of layers, and 𝐍𝐨𝐫𝐦\mathbf{Norm} is the normalization layer.

### 3.3 Vim Block

The original Mamba block is designed for the 1-D sequence, which is not suitable for vision tasks requiring spatial-aware understanding. In this section, we introduce the Vim block, which incorporates the bidirectional sequence modeling for the vision tasks. The Vim block is shown in Fig. 2.

Specifically, we present the operations of Vim block in Algo. 29. The input token sequence 𝐓𝚕−1\mathbf{T}_{\mathtt{l}-1} is first normalized by the normalization layer. Next, we linearly project the normalized sequence to the 𝐱\mathbf{x} and 𝐳\mathbf{z} with dimension size EE. Then, we process the 𝐱\mathbf{x} from the forward and backward directions. For each direction, we first apply the 1-D convolution to the 𝐱\mathbf{x} and get the 𝐱o′\mathbf{x}^{\prime}_{o}. We then linearly project the 𝐱o′\mathbf{x}^{\prime}_{o} to the 𝐁o\mathbf{B}_{o}, 𝐂o\mathbf{C}_{o}, 𝚫o\mathbf{\Delta}_{o}, respectively. The 𝚫o\mathbf{\Delta}_{o} is then used to transform the 𝐀¯o\overline{\mathbf{A}}_{o}, 𝐁¯o\overline{\mathbf{B}}_{o}, respectively. Finally, we compute the 𝐲f​o​r​w​a​r​d\mathbf{y}_{forward} and 𝐲b​a​c​k​w​a​r​d\mathbf{y}_{backward} through the SSM. The 𝐲f​o​r​w​a​r​d\mathbf{y}_{forward} and 𝐲b​a​c​k​w​a​r​d\mathbf{y}_{backward} are then gated by the 𝐳\mathbf{z} and added together to get the output token sequence 𝐓𝚕\mathbf{T}_{\mathtt{l}}.

### 3.4 Architecture Details

In summary, the hyper-parameters of our architecture are listed as follows: 𝙻\mathtt{L} denotes the number of blocks, 𝙳\mathtt{D} denotes the hidden state dimension, 𝙴\mathtt{E} denotes the expanded state dimension, and 𝙽\mathtt{N} denotes the SSM dimension. Following ViT (Dosovitskiy et al., 2020) and DeiT (Touvron et al., 2021b), we first employ 16×\times16 kernel size projection layer to get a 1-D sequence of non-overlapping patch embeddings. Subsequently, we directly stack 𝙻\mathtt{L} Vim blocks. By default, we set the number of blocks 𝙻\mathtt{L} to 24, SSM dimension 𝙽\mathtt{N} to 16. To align with the model sizes of DeiT series, we set the hidden state dimension 𝙳\mathtt{D} to 192 and expanded state dimension 𝙴\mathtt{E} to 384 for the tiny-size variant. For the small-size variant, we set 𝙳\mathtt{D} to 384 and 𝙴\mathtt{E} to 768.

### 3.5 Efficiency Analysis

Traditional SSM-based methods leverage the fast Fourier transform to boost the convolution operation as shown in Eq. (3). For data-dependent methods, such as Mamba, the SSM operation in Line 11 of Algo. 29 is no longer equivalent to convolution. To address this problem, Mamba and the proposed Vim choose a modern-hardware-friendly way to ensure efficiency. The key idea of this optimization is to avoid the IO-bound and memory-bound of modern hardware accelerators (GPUs).

IO-Efficiency. The high bandwidth memory (HBM) and SRAM are two important components for GPUs. Among them, SRAM has a larger bandwidth and HBM has a bigger memory size. The standard implementation of Vim’s SSM operation with HBM requires the number of memory IO on the order of O⁡(𝙱𝙼𝙴𝙽)O(\mathtt{B}\mathtt{M}\mathtt{E}\mathtt{N}). Inspired by Mamba, Vim first reads in O⁡(𝙱𝙼𝙴+𝙴𝙽)O(\mathtt{B}\mathtt{M}\mathtt{E}+\mathtt{E}\mathtt{N}) bytes of memory (𝚫𝐨,𝐀𝐨,𝐁𝐨,𝐂𝐨)(\mathbf{\Delta_{o}},\mathbf{{A}_{o}},\mathbf{{B}_{o}},\mathbf{C_{o})} from slow HBM to fast SRAM. Then, Vim gets the discrete 𝐀¯𝐨\mathbf{\overline{A}_{o}}, 𝐁¯𝐨\mathbf{\overline{B}_{o}} of a size of (𝙱,𝙼,𝙴,𝙽)(\mathtt{B},\mathtt{M},\mathtt{E},\mathtt{N}) in SRAM. Last, Vim performs SSM operations in SRAM and writes the output of a size of (𝙱,𝙼,𝙴)(\mathtt{B},\mathtt{M},\mathtt{E}) back to HBM. This method can help to reduce IOs from O⁡(𝙱𝙼𝙴𝙽)O(\mathtt{B}\mathtt{M}\mathtt{E}\mathtt{N}) to O⁡(𝙱𝙼𝙴+𝙴𝙽)O(\mathtt{B}\mathtt{M}\mathtt{E}+\mathtt{E}\mathtt{N}).

Memory-Efficiency. To avoid out-of-memory problems and achieve lower memory usage when dealing with long sequences, Vim chooses the same recomputation method as Mamba. For the intermediate states of size (𝙱,𝙼,𝙴,𝙽)(\mathtt{B},\mathtt{M},\mathtt{E},\mathtt{N}) to calculate the gradient, Vim recomputes them at the network backward pass. For intermediate activations such as the output of activation functions and convolution, Vim also recomputes them to optimize the GPU memory requirement, as the activation values take a lot of memory but are fast for recomputation.

Computation-Efficiency. SSM in Vim block (Line 11 in Algo.29) and self-attention in Transformer both play a key role in providing global context adaptively. Given a visual sequence 𝐓∈R1×𝙼×𝙳\mathbf{T}\in R^{1\times\mathtt{M}\times\mathtt{D}} and the default setting 𝙴=2​𝙳\mathtt{E}=2\mathtt{D}, the computation complexity of a global self-attention and SSM are:

|  |  | Ω⁡(self-attention)=4​𝙼𝙳2+2​𝙼2​𝙳,\displaystyle\Omega(\text{self-attention})=4\mathtt{M}\mathtt{D}^{2}+2\mathtt{M}^{2}\mathtt{D}, |  | (5) |
|---|---|---|---|---|
|  |  | Ω⁡(SSM)=3​𝙼​(2​𝙳)​𝙽+𝙼⁡(2​𝙳)​𝙽,\displaystyle\Omega(\text{SSM})=3\mathtt{M}(2\mathtt{D})\mathtt{N}+\mathtt{M}(2\mathtt{D})\mathtt{N}, |  | (6) |

where self-attention is quadratic to sequence length 𝙼\mathtt{M}, and SSM is linear to sequence length 𝙼\mathtt{M} (𝙽\mathtt{N} is a fixed parameter, set to 16 by default). The computational efficiency makes Vim scalable for gigapixel applications with large sequence lengths.

## 4 Experiment

| Method | image size | image | size | #param. | ImageNet top-1 acc. | ImageNet | top-1 acc. |
|---|---|---|---|---|---|---|---|
| image |  |  |  |  |  |  |  |
| size |  |  |  |  |  |  |  |
| ImageNet |  |  |  |  |  |  |  |
| top-1 acc. |  |  |  |  |  |  |  |
| Convnets |  |  |  |  |  |  |  |
| ResNet-18 | 2242224^{2} | 12M | 69.8 |  |  |  |  |
| ResNet-50 | 2242224^{2} | 25M | 76.2 |  |  |  |  |
| ResNet-101 | 2242224^{2} | 45M | 77.4 |  |  |  |  |
| ResNet-152 | 2242224^{2} | 60M | 78.3 |  |  |  |  |
| ResNeXt50-32×\times4d | 2242224^{2} | 25M | 77.6 |  |  |  |  |
| RegNetY-4GF | 2242224^{2} | 21M | 80.0 |  |  |  |  |
| Transformers |  |  |  |  |  |  |  |
| ViT-B/16 | 3842384^{2} | 86M | 77.9 |  |  |  |  |
| ViT-L/16 | 3842384^{2} | 307M | 76.5 |  |  |  |  |
| DeiT-Ti | 2242224^{2} | 6M | 72.2 |  |  |  |  |
| DeiT-S | 2242224^{2} | 22M | 79.8 |  |  |  |  |
| DeiT-B | 2242224^{2} | 86M | 81.8 |  |  |  |  |
| SSMs |  |  |  |  |  |  |  |
| S4ND-ViT-B | 2242224^{2} | 89M | 80.4 |  |  |  |  |
| Vim-Ti | 2242224^{2} | 7M | 76.1 |  |  |  |  |
| Vim-Ti† | 2242224^{2} | 7M | 78.3 +2.2 |  |  |  |  |
| Vim-S | 2242224^{2} | 26M | 80.3 |  |  |  |  |
| Vim-S† | 2242224^{2} | 26M | 81.4 +1.1 |  |  |  |  |
| Vim-B | 2242224^{2} | 98M | 81.9 |  |  |  |  |
| Vim-B† | 2242224^{2} | 98M | 83.2 +1.3 |  |  |  |  |

| image |
|---|
| size |

| ImageNet |
|---|
| top-1 acc. |

### 4.1 Image Classification

Settings. We benchmark Vim on the ImageNet-1K dataset (Deng et al., 2009), which contains 1.28M training images and 50K validation images from 1,000 categories. All models are trained on the training set, and top-1 accuracy on the validation set is reported. For fair comparisons, our training settings mainly follow DeiT (Touvron et al., 2021b). Specifically, we apply random cropping, random horizontal flipping, label-smoothing regularization, mixup, and random erasing as data augmentations. When training on 2242224^{2} input images, we employ AdamW (Loshchilov & Hutter, 2019) with a momentum of 0.90.9, a total batch size of 10241024, and a weight decay of 0.050.05 to optimize models. We train the Vim models for 300300 epochs using a cosine schedule, 1×1\times10−310^{-3} initial learning rate, and EMA. During testing, we apply a center crop on the validation set to crop out 2242224^{2} images. Experiments are performed on 8 A800 GPUs.

| Method | Backbone | image size | image | size | #param. | v​a​lval mIoU | v​a​lval | mIoU |
|---|---|---|---|---|---|---|---|---|
| image |  |  |  |  |  |  |  |  |
| size |  |  |  |  |  |  |  |  |
| v​a​lval |  |  |  |  |  |  |  |  |
| mIoU |  |  |  |  |  |  |  |  |
| DeepLab v3+ | ResNet-101 | 5122512^{2} | 63M | 44.1 |  |  |  |  |
| UperNet | ResNet-50 | 5122512^{2} | 67M | 41.2 |  |  |  |  |
| UperNet | ResNet-101 | 5122512^{2} | 86M | 44.9 |  |  |  |  |
| UperNet | DeiT-Ti | 5122512^{2} | 11M | 39.2 |  |  |  |  |
| UperNet | DeiT-S | 5122512^{2} | 43M | 44.0 |  |  |  |  |
| UperNet | Vim-Ti | 5122512^{2} | 13M | 41.0 |  |  |  |  |
| UperNet | Vim-S | 5122512^{2} | 46M | 44.9 |  |  |  |  |

| image |
|---|
| size |

| v​a​lval |
|---|
| mIoU |

Long Sequence Fine-tuning To make full use of the efficient long sequence modeling power of Vim, we continue to fine-tune Vim with a long sequence setting for 30 epochs after pretraining. Specifically, we set a patch extraction stride of 88 while keeping the patch size unchanged, a constant learning rate of 10−510^{-5}, and a weight decay of 10−810^{-8}.

Results. Tab. 1 compares Vim with ConvNet-based, Transformer-based and SSM-based backbone networks. Compared to ConvNet-based ResNet (He et al., 2016), Vim demonstrates superior performance. For example, when the parameters are roughly similar, the top-1 accuracy of Vim-Small reaches 80.3, which is 4.1 points higher than that of ResNet50. Compared with the conventional self-attention-based ViT (Dosovitskiy et al., 2020), Vim outperforms it by considerable margins in terms of both parameter numbers and classification accuracy. When compared to the highly-optimized ViT-variant, i.e., DeiT (Touvron et al., 2021b), Vim surpasses it at different scales with comparable parameter numbers: 3.9 points higher for Vim-Tiny over DeiT-Tiny, 0.5 points higher for Vim-Small over DeiT-Small, and 0.1 points higher for Vim-Base over DeiT-Base. Compared with SSM-based S4ND-ViT-B (Nguyen et al., 2022), Vim achieves similar top-1 accuracy with 3×\times fewer parameters. After long sequence fine-tuning, Vim-Tiny†, Vim-S†, and Vim-B† all achieve higher results. Among them, Vim-S† even achieves similar results with DeiT-B. The results demonstrate that Vim can be adapted to longer sequence modeling easily and extract stronger visual representation.

Fig. 1 (b) and (c) compare the FPS and GPU memory of tiny-size Vim and DeiT. Vim demonstrates better efficiency in speed and memory as image resolution grows. Specifically, when the image size is 512×\times512, Vim achieves similar FPS and memory as DeiT. As the image size grows to 1248×\times1248, Vim is 2.8×\times faster than DeiT and saves 86.8% GPU memory. The pronounced superiority of Vim’s linear scaling in sequence length makes it ready for high-resolution downstream vision applications and long-sequence multi-modality applications.

| Backbone | APbox{}^{\text{box}} | AP50box{}^{\text{box}}_{\text{50}} | AP75box{}^{\text{box}}_{\text{75}} | APsbox{}^{\text{box}}_{\text{s}} | APmbox{}^{\text{box}}_{\text{m}} | APlbox{}^{\text{box}}_{\text{l}} |
|---|---|---|---|---|---|---|
| DeiT-Ti | 44.4 | 63.0 | 47.8 | 26.1 | 47.4 | 61.8 |
| Vim-Ti | 45.7 | 63.9 | 49.6 | 26.1 | 49.0 | 63.2 |
| Backbone | APmask{}^{\text{mask}} | AP50mask{}^{\text{mask}}_{\text{50}} | AP75mask{}^{\text{mask}}_{\text{75}} | APsmask{}^{\text{mask}}_{\text{s}} | APmmask{}^{\text{mask}}_{\text{m}} | APlmask{}^{\text{mask}}_{\text{l}} |
| DeiT-Ti | 38.1 | 59.9 | 40.5 | 18.1 | 40.5 | 58.4 |
| Vim-Ti | 39.2 | 60.9 | 41.7 | 18.2 | 41.8 | 60.2 |

### 4.2 Semantic Segmentation

Settings. We conduct experiments for semantic segmentation on the ADE20K (Zhou et al., 2019) and use UperNet (Xiao et al., 2018b) as the segmentation framework. We provide detailed settings in Sec. B.

Results. As shown in Tab. 2, Vim consistently outperforms DeiT across different scales: 1.8 mIoU higher for Vim-Ti over DeiT-Ti, and 0.9 mIoU higher for Vim-S over DeiT-S. Compared to the ResNet-101 backbone, our Vim-S achieves the same segmentation performance with nearly 2×\times fewer parameters.

To further evaluate the efficiency for downstream tasks, i.e., segmentation, detection, and instance segmentation, we combine the backbones with a commonly used feature pyramid network (FPN) module and benchmark their FPS and GPU memory. As shown in Fig. 3 and Fig. 4, the efficiency curves demonstrate similar comparison results of the pure backbone (Fig. 1), though we append a heavy FPN on the backbones. The exceptional linear scaling performance is attributed to our proposed efficient backbone Vim, which builds the foundation for learning gigapixel-level visual representation in an end-to-end manner without the need for multi-stage encoding (e.g., aerial image, medical image, and computational pathology).

### 4.3 Object Detection and Instance Segmentation

Settings. We conduct experiments for object detection and instance segmentation on the COCO 2017 dataset (Lin et al., 2014) and use ViTDet (Xiao et al., 2018b) as the basic framework. We provide detailed settings in Sec. B.

Results. Tab. 3 compares Vim-Ti with DeiT-Ti using Cascade Mask R-CNN framework (Cai & Vasconcelos, 2019). Vim-Ti surpasses DeiT-Ti by 1.3 box AP and 1.1 mask AP. For the middle-size and large-size objects, Vim-Ti outperforms DeiT-Ti by 1.6 APmbox{}^{\text{box}}_{\text{m}}/1.3 APmmask{}^{\text{mask}}_{\text{m}} and 1.4 APlbox{}^{\text{box}}_{\text{l}}/1.8 APlmask{}^{\text{mask}}_{\text{l}}, demonstrating better long-range context learning than DeiT (Fig. 5).

We highlight that the accuracy superiority is non-trivial since DeiT is equipped with window attention while Vim works in a pure sequence modeling manner. Specifically, to perform representation learning on high-resolution images (i.e., 1024×\times1024), we follow ViTDet (Li et al., 2022c) and modify the DeiT backbone with the use of 2D window attention, which injects 2D prior and breaks the sequential modeling nature of Transformer. Thanks to the efficiency illustrated in Sec. 3.5, Fig. 1 and Fig. 4, we can directly apply Vim on 1024×\times1024 input images and learn sequential visual representation for object detection and instance segmentation without need for 2D priors in the backbone.

| Bidirectional strategy | ImageNet top-1 acc. | ImageNet | top-1 acc. | ADE20K mIoU | ADE20K | mIoU |
|---|---|---|---|---|---|---|
| ImageNet |  |  |  |  |  |  |
| top-1 acc. |  |  |  |  |  |  |
| ADE20K |  |  |  |  |  |  |
| mIoU |  |  |  |  |  |  |
| None | 73.2 | 32.3 |  |  |  |  |
| Bidirectional Layer | 70.9 | 33.6 |  |  |  |  |
| Bidirectional SSM | 72.8 | 33.2 |  |  |  |  |
| Bidirectional SSM + Conv1d | 73.9 | 35.9 |  |  |  |  |

| ImageNet |
|---|
| top-1 acc. |

| ADE20K |
|---|
| mIoU |

### 4.4 Ablation Study

Bidirectional SSM. We ablate the key bidirectional design of Vim, using ImageNet-1K classification and the Segmenter (Strudel et al., 2021) semantic segmentation framework on ADE20K. To fully evaluate the power of learned representation on ImageNet, we use a simple Segmenter head with only 2 layers to perform transfer learning on semantic segmentation. We study the following bidirectional strategies. None: We directly adopt the Mamba block to process visual sequence with only the forward direction. Bidirectional Sequence: During training, we randomly flip the visual sequence. This works like data augmentation. Bidirectional Block: We pair the stacked blocks. The first block of each pair processes visual sequence in the forward direction and the second block of each pair processes in the backward direction. Bidirectional SSM: We add an extra SSM for each block to process the visual sequence in the backward direction. Bidirectional SSM + Conv1d: Based on Bidirectional SSM, we further add a backward Conv1d before the backward SSM (Fig. 2).

As shown in Tab. 4, directly adopting the Mamba block achieves good performance in classification. However, the unnatural unidirectional manner poses challenges in downstream dense prediction. Specifically, the preliminary bidirectional strategy of using Bidirectional Block achieves 7 points lower top-1 accuracy on classification. Yet, it outperforms the vanilla unidirectional Mamba block by 1.3 mIoU on semantic segmentation. By adding extra backward SSM and Conv1d, we achieve superior classification accuracy (73.9 top-1 acc vs. 73.2 top-1 acc) and exceptional segmentation superiority (35.9 mIoU vs. 32.3 mIoU). We use the strategy of Bidirectional SSM + Conv1d as the default setting in our Vim block.

Classification Design. We ablate the classification design of Vim, benchmarking on ImageNet-1K classification. We study the following classification strategies. Mean pool: We adopt mean pooling on the output feature from the last Vim block and perform classification on this pooled feature. Max pool: We first adapt the classification head on each token of the visual sequence and then perform max pooling on the sequence to get the classification prediction result. Head class token: Following DeiT (Touvron et al., 2021b), we concatenate the class token at the head of the visual sequence and perform classification. Double class token: Based on the head class token strategy, we additionally add a class token at the tail of the visual sequence. Middle class token: We add a class token at the middle of the visual sequence and then perform classification on the final middle class token.

| Classification strategy | ImageNet top-1 acc. |
|---|---|
| Mean pool | 73.9 |
| Max pool | 73.4 |
| Head class token | 75.2 |
| Double class token | 74.3 |
| Middle class token | 76.1 |

As shown in Tab. 5, experiments show that the middle class token strategy can fully exploit the recurrent nature of SSM and the central object prior in ImageNet, demonstrating the best top-1 accuracy of 76.1.

## 5 Conclusion and Future Work

We have proposed Vision Mamba (Vim) to explore the very recent efficient state space model, i.e., Mamba, as generic vision backbones. Unlike prior state space models for vision tasks which use hybrid architecture or equivalent global 2D convolutional kernel, Vim learns visual representation in the sequence modeling manner and does not introduce image-specific inductive biases. Thanks to the proposed bidirectional state space modeling, Vim achieves data-dependent global visual context and enjoys the same modeling power as Transformer, while having lower computation complexity. Benefiting from the hardware-aware designs of Mamba, the inference speed and memory usage of Vim are significantly better than ViTs when processing high-resolution images. Experiment results on standard computer vision benchmarks have verified the modeling power and high efficiency of Vim, showing that Vim has great potential to be the next-generation vision backbone.

In future works, Vim with the bidirectional SSM modeling with position embeddings is suitable for unsupervised tasks such as mask image modeling pretraining and the similar architecture with Mamba enables multimodal tasks such as CLIP-style pretraining. Based on the pretrained Vim weights, exploring the usefulness of Vim for analyzing high-resolution medical images, remote sensing images, and long videos, which can be regarded as downstream tasks, is very straightforward.

## Impact Statement

We advance the efficiency of the generic vision backbone. Any societal consequences or impacts that typically relate to work focused on increased efficiency also apply here, as such work necessarily improves the practicality of vision backbone for an array of visual applications with high-resolution input images.

## Acknowledgement

This work was partially supported by the National Science and Technology Major Project under Grant No. 2023YFF0905400 and National Natural Science Foundation of China (NSFC) under Grant No. 62276108.

We would like to acknowledge Tianheng Cheng, Yuxin Fang, Shusheng Yang, Bo Jiang, and Jingfeng Yao for their helpful feedback on the draft.

## Appendix A Visualization

## Appendix B Additional Setting

Settings for Semantic Segmentation. We conduct experiments for semantic segmentation on the ADE20K (Zhou et al., 2019) dataset. ADE20K contains 150 fine-grained semantic categories, with 20K, 2K, and 3K images for training, validation, and testing, respectively. We choose UperNet (Xiao et al., 2018a) as our base framework. In training, we employ AdamW with a weight decay of 0.010.01, and a total batch size of 1616 to optimize models. The employed training schedule uses an initial learning rate of 66×\times10−510^{-5}, linear learning rate decay, a linear warmup of 1,5001,500 iterations, and a total training of 160160K iterations. The data augmentations follow common settings, including random horizontal flipping, random re-scaling within the ratio range [0.5,2.0][0.5,2.0], and random photometric distortion. During evaluation, we rescale the image to have a shorter side of 512512.

Settings for Object Detection and Instance Segmentation. We conduct experiments for object detection and instance segmentation on the COCO 2017 dataset (Lin et al., 2014). The COCO 2017 dataset contains 118K images for training, 5K images for validating, and 20K images for testing. We use the canonical Cascade Mask R-CNN (Cai & Vasconcelos, 2019) as the base framework. For ViT-based backbones, we apply extra configurations (e.g., interleaved window & global attention) to handle the high-resolution images following ViTDet (Li et al., 2022c). For SSM-based Vim, we directly use it without any modifications. Other training and evaluation settings are just the same. During training, we employ AdamW with a weight decay of 0.10.1, and a total batch size of 6464 to optimize models. The employed training schedule uses an initial learning rate of 11×\times10−410^{-4}, linear learning rate decay, and a total training of 380380K iterations. The data augmentations use large-scale jitter data augmentation (Ghiasi et al., 2021) to 1024×\times1024 input images. During evaluation, we rescale the image to have a shorter side of 1024.

## Appendix C Extended Comparison on Hierarchical Architecture

To further compare with hierarchical architectures, we propose another variant Hier-Vim by replacing shifted local window attention in SwinTransformer with the proposed global bidirectional SSM. We detail the configuration in Tab. 6

| Model | #Blocks | #Channels | Params |
|---|---|---|---|
| Hier-Vim-T | [2, 2, 5, 2] | [96, 192, 384, 768] | 30M |
| Hier-Vim-S | [2, 2, 15, 2] | [96, 192, 384, 768] | 50M |
| Hier-Vim-B | [2, 2, 15, 2] | [128, 256, 512, 1024] | 89M |

| Method | image size | image | size | #param. | ImageNet top-1 acc. | ImageNet | top-1 acc. |
|---|---|---|---|---|---|---|---|
| image |  |  |  |  |  |  |  |
| size |  |  |  |  |  |  |  |
| ImageNet |  |  |  |  |  |  |  |
| top-1 acc. |  |  |  |  |  |  |  |
| Swin-T (Liu et al., 2021) | 2242224^{2} | 28M | 81.2 |  |  |  |  |
| FocalTransformer-T (Yang et al., 2021) | 2242224^{2} | 29M | 82.2 |  |  |  |  |
| CVT-21 (Wu et al., 2021) | 2242224^{2} | 32M | 82.5 |  |  |  |  |
| MetaFormer-S35 (Yu et al., 2022) | 2242224^{2} | 31M | 81.4 |  |  |  |  |
| GFNet-H-S (Rao et al., 2021) | 2242224^{2} | 32M | 81.5 |  |  |  |  |
| Hier-Vim-T | 2242224^{2} | 30M | 82.5 |  |  |  |  |
| Swin-S (Liu et al., 2021) | 2242224^{2} | 50M | 83.2 |  |  |  |  |
| FocalTransformer-S (Yang et al., 2021) | 2242224^{2} | 51M | 83.5 |  |  |  |  |
| MetaFormer-S35 (Yu et al., 2022) | 2242224^{2} | 73M | 82.5 |  |  |  |  |
| GFNet-H-B (Rao et al., 2021) | 2242224^{2} | 54M | 82.9 |  |  |  |  |
| Hier-Vim-S | 2242224^{2} | 50M | 83.4 |  |  |  |  |
| Swin-B (Liu et al., 2021) | 2242224^{2} | 88M | 83.5 |  |  |  |  |
| FocalTransformer-B (Yang et al., 2021) | 2242224^{2} | 90M | 83.8 |  |  |  |  |
| Hier-Vim-B | 2242224^{2} | 89M | 83.9 |  |  |  |  |

| image |
|---|
| size |

| ImageNet |
|---|
| top-1 acc. |

Classification on ImageNet. Following the standard training and validation protocols (Liu et al., 2021; Liu et al., 2024), we compare Hier-Vim with popular hierarchical architectures across tiny, small, and base model sizes in Tab. 7. The results indicate that Hier-Vim outperforms Swin Transformer by 1.3% at the tiny size, 0.2% at the small size, and 0.4% at the base size, demonstrating competitive performance against well-established and highly-optimized modern hierarchical architectures.

langley00
