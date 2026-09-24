# MonoSplat: Generalizable 3D Gaussian Splatting from Monocular Depth Foundation Models

Recent advances in generalizable 3D Gaussian Splatting have demonstrated promising results in real-time high-fidelity rendering without per-scene optimization, yet existing approaches still struggle to handle unfamiliar visual content during inference on novel scenes due to limited generalizability. To address this challenge, we introduce MonoSplat, a novel framework that leverages rich visual priors from pre-trained monocular depth foundation models for robust Gaussian reconstruction. Our approach consists of two key components: a Mono-Multi Feature Adapter that transforms monocular features into multi-view representations, coupled with an Integrated Gaussian Prediction module that effectively fuses both feature types for precise Gaussian generation. Through the Adapter’s lightweight attention mechanism, features are seamlessly aligned and aggregated across views while preserving valuable monocular priors, enabling the Prediction module to generate Gaussian primitives with accurate geometry and appearance. Through extensive experiments on diverse real-world datasets, we convincingly demonstrate that MonoSplat achieves superior reconstruction quality and generalization capability compared to existing methods while maintaining computational efficiency with minimal trainable parameters. Codes are available at https://github.com/CUHK-AIM-Group/MonoSplat.

## 1 Introduction

The ability to faithfully reconstruct 3D scenes and generate photorealistic renderings represents a cornerstone challenge in computer vision and graphics, with far-reaching implications for virtual reality experiences [49], digital content creation [27], and large-scale visualization systems [39]. While the field has witnessed significant evolution in scene representation approaches - progressing from scene representation networks (SRN) [33] to neural radiance fields (NeRF) [21], light field networks (LFN) [35], and most recently 3D Gaussian Splatting (3DGS) [14] - practical deployment still faces critical barriers. These include substantial computational overhead during rendering [42, 51] and time-consuming per-scene optimization procedures [26, 56], which collectively hinder the widespread adoption of these promising technologies.

Recent advances in generalizable 3D Gaussian Splatting [1, 6, 45, 17] represent a paradigm shift in addressing traditional reconstruction challenges. These methods leverage feed-forward networks for direct 3D Gaussian prediction, allowing for real-time high-fidelity rendering while eliminating the need for per-scene optimization. Various architectural innovations have emerged, including epipolar geometry-based depth estimation in pixelSplat [1] and latentSplat [45], as well as cost volume-based multi-view aggregation in MVSplat [6] and MVSGaussian [17]. Subsequent works, FreeSplat [43] and eFreeSplat [22], further explore free-view reconstruction and training without epipolar constraints. However, these approaches often encounter limitations when processing scenes with unfamiliar visual content and layouts, as their understanding of the visual world is constrained by training data distribution and hampered by challenges in zero-shot domain generalization. This observation motivates our investigation into leveraging the rich visual priors embedded in state-of-the-art monocular depth foundation models to achieve more robust and generalizable Gaussian reconstruction.

The basis of our approach originates from the observation: Comtemporary monocular depth foundation models (e.g., MiDaS [28], Depth Anything Model [50]), trained on extensive datasets, exhibit exceptional proficiency in predicting monocular depths across diverse visual domains. If a thorough and extensive understanding of the visual world is fundamental to generalizable Gaussian reconstruction, then it should be feasible to develop a widely applicable reconstruction framework using a pretrained monocular depth estimator. To this end, we present MonoSplat, a novel Gaussian reconstruction framework built upon monocular depth foundation models. The MonoSplat framework comprises two key architectural components: First, a Mono-Multi Feature Adapter that transforms the monocular features from the frozen depth foundation model into multi-view features with cross-view awareness. Second, an Integrated Gaussian Prediction module that synergistically integrates monocular and multi-view features to generate precise Gaussian primitives. Through such an elegant design philosophy, the architecture achieves remarkable efficiency with minimal trainable parameters. Most significantly, MonoSplat’s novel integration of monocular depth priors enables unprecedented zero-shot generalization capabilities, establishing new performance standards across diverse real-world scenarios. Our primary contributions are:

- • We propose a simple yet effective framework to convert a pre-trained monocular depth foundation model into a Gaussian reconstruction model;
We propose a simple yet effective framework to convert a pre-trained monocular depth foundation model into a Gaussian reconstruction model;

- • We introduce a Mono-Multi Feature Adapter that converts monocular features from the depth foundation model into multi-view features with cross-view awareness;
We introduce a Mono-Multi Feature Adapter that converts monocular features from the depth foundation model into multi-view features with cross-view awareness;

- • We present an Integrated Gaussian Prediction module that effectively combines monocular and multi-view features to produce accurate Gaussian primitives;
We present an Integrated Gaussian Prediction module that effectively combines monocular and multi-view features to produce accurate Gaussian primitives;

- • MonoSplat, a state-of-the-art, generalizable Gaussian reconstruction model that offers excellent performance across a wide variety of natural images.
MonoSplat, a state-of-the-art, generalizable Gaussian reconstruction model that offers excellent performance across a wide variety of natural images.

## 2 Related work

3D Reconstruction and View Synthesis. The landscape of 3D reconstruction from posed images has been fundamentally transformed by breakthroughs in neural rendering [41] and neural fields [21, 46, 34]. Modern approaches construct scene representations through differentiable rendering by optimizing image-space photometric errors. While early methods explored voxel-based structures [25, 32, 20], the field witnessed a paradigm shift towards neural fields with volume rendering [21, 46] as the dominant framework. To address computational inefficiencies inherent in these approaches, researchers investigated discrete data structures [10, 3, 16, 24], though the challenge of achieving real-time high-resolution rendering persisted. The advent of 3D Gaussian splatting [14] introduced efficient rasterization capabilities, yet its reliance on extensive image sets for quality synthesis remained a limitation. Our work pushes the boundaries of this domain by introducing neural architectures capable of estimating 3D Gaussian primitives through single forward passes.

Generalizable Neural Radiance Fields. The field of generalizable reconstruction has witnessed remarkable progress since the introduction of Neural Radiance Fields (NeRF) [21], enabling advancements in both object-centric [7, 11, 13, 18, 30, 42, 51] and scene-level reconstruction [7, 8, 2, 5, 47]. Following PixelNeRF’s [51] groundbreaking work in image-based radiance field reconstruction using pixel-aligned features, the field has seen continuous innovation through enhanced feed-forward models incorporating sophisticated matching mechanisms [2, 5], transformer-based architectures [31, 8, 23], and novel volumetric representations [2, 47]. While MuRF [47] currently sets the state-of-the-art by leveraging target view frustum volumes with (2+1)D CNN architecture, a fundamental challenge persists across these approaches: the computational burden of per-pixel volume sampling during rendering. The requirement for extensive sampling points along each ray to achieve high-quality reconstruction results in prohibitive computational costs and rendering times, significantly impeding practical deployment.

Generalizable 3D Gaussian Splatting. 3D Gaussian Splatting [14, 4] introduced an efficient rasterization-based rendering paradigm, sparking significant research interest in feed-forward models [38, 37, 1, 54, 6, 17, 45, 52, 40, 9]. Initial single-view approaches, such as Splatter Image [38] and Flash3D [37], demonstrated promising results in simple scenes but exhibited limitations when handling complex geometries. Then, pixelSplat [1] pioneered the use of epipolar transformers and probabilistic depth estimation, while MVSplat [6] and MVSGaussians [17] advanced the field through cost volume-based depth prediction. LatentSplat [45] further expanded the capabilities with variational Gaussians for view extrapolation. Subsequently, FreeSplat [43] addressed reconstruction from extended sequences, and eFreeSplat [22] introduced a novel architecture that eliminates dependency on epipolar priors. Despite these advances, existing methods have largely overlooked the potential of foundation models’ visual priors, limiting their generalizability due to constrained visual knowledge. The concurrent work DepthSplat [48] makes initial strides in leveraging monocular features from Depth Anything Model [50]. However, their dual-branch network architecture introduces redundant model parameters. In contrast, our framework achieves superior efficiency by directly utilizing a frozen depth foundation model and elegantly converting monocular features into multi-view representations, resulting in a more efficient and effective solution.

## 3 Method

Given an image sequence ℐ={𝑰i}i=1N{\mathcal{I}}=\{{\bm{I}}_{i}\}_{i=1}^{N} (𝑰i∈ℝH×W×3{\bm{I}}_{i}\in\mathbb{R}^{H\times W\times 3}) and their corresponding camera projection matrices 𝒫={𝑷i|𝑷i=𝑲i​[𝑹i|𝒕i]}i=1N{\mathcal{P}}=\{{\bm{P}}_{i}|{\bm{P}}_{i}={\bm{K}}_{i}[{\bm{R}}_{i}|{\bm{t}}_{i}]\}_{i=1}^{N}, derived from camera intrinsics 𝑲i{\bm{K}}_{i}, rotation matrices 𝑹i{\bm{R}}_{i}, and translation vectors 𝒕i{\bm{t}}_{i}, our objective is to learn a mapping function f𝜽f_{\bm{\theta}} that transforms input images to 3D Gaussian primitives:

|  | f𝜽:{𝑰i,𝑷i}i=1N→{𝝁j,αj,𝚺j,𝒄j}j=1N×H×W,f_{\bm{\theta}}:\{{\bm{I}}_{i},{\bm{P}}_{i}\}_{i=1}^{N}\rightarrow\{\bm{\mu}_{j},\alpha_{j},\mathbf{\Sigma}_{j},{\bm{c}}_{j}\}_{j=1}^{N\times H\times W}, |  | (1) |
|---|---|---|---|

where f𝜽f_{\bm{\theta}} is realized through a feed-forward network with learnable parameters 𝜽\bm{\theta}, trained on large-scale datasets. Each Gaussian primitive is characterized by its spatial position 𝝁j\bm{\mu}_{j}, opacity value αj\alpha_{j}, covariance matrix 𝚺j\mathbf{\Sigma}_{j}, and spherical harmonics-based color coefficients 𝒄j{\bm{c}}_{j}. To achieve robust and generalizable reconstruction from multiple input views, we propose an architecture that builds upon monocular depth foundation models, as illustrated in Figure 2.

The MonoSplat framework consists of two principal components: (1) a Mono-Multi Feature Adapter that transforms monocular features from foundation models into scene-aware multi-view representations (Sec. 3.1), and (2) an Integrated Gaussian Modeling module that synergistically combines monocular and multi-view features to predict precise Gaussian primitives (Sec. 3.2). The comprehensive training methodology and optimization strategy are elaborated in Sec. 3.3.

### 3.1 Mono-Multi Feature Adapter

Depth foundation models, with Depth Anything Model [50] as a prominent example, have achieved unprecedented cross-domain generalization through their extensive training on large-scale datasets. However, harnessing these models for generalized Gaussian reconstruction introduces a fundamental challenge: transforming the view-specific monocular representations into geometrically coherent multi-view features that are essential for precise 3D reconstruction. To bridge this gap, we propose a two-stage feature adapter architecture. The first stage consolidates multi-scale encoder features into a unified representation, capturing both fine-grained details and global context. The second stage leverages a multi-view transformer architecture that enables comprehensive cross-view feature exchange, establishing geometric consistency across different viewpoints. Besides, we extract complementary monocular features from the foundation model, providing rich depth priors that serve as a crucial foundation for accurate Gaussian parameter inference.

Our feature extraction and fusion pipeline consists of two main stages: multi-scale feature extraction with unified aggregation, and cross-view geometric reasoning.

In the first stage, for each image 𝑰i{\bm{I}}_{i} in the sequence ℐ={𝑰i}i=1N{\mathcal{I}}=\{{\bm{I}}_{i}\}_{i=1}^{N} (𝑰i∈ℝH×W×3{\bm{I}}_{i}\in\mathbb{R}^{H\times W\times 3}), we employ a hierarchical feature extraction strategy to obtain multi-scale representations {𝑭i1,…,𝑭iS}\{{\bm{F}}_{i}^{1},...,{\bm{F}}_{i}^{S}\} from intermediate layers of the monocular depth encoder, where SS denotes the number of feature scales. It is worth noting that the depth encoder is deliberately maintained in a frozen state during training, serving dual purposes: preserving the valuable 3D geometric priors acquired through extensive pre-training on large-scale datasets, and substantially reducing the number of trainable parameters. Then, to effectively consolidate these multi-scale features {𝑭i1,…,𝑭iS}\{{\bm{F}}_{i}^{1},...,{\bm{F}}_{i}^{S}\} into a unified representation, we adopt the DPT architecture [29], which progressively fuses the feature hierarchy through:

|  | 𝑭i=ℱD​P​T(𝑭i1,…,𝑭iS), 𝑭i∈ℝH/4×W/4×C,{\bm{F}}_{i}=\mathcal{F}_{DPT}({{\bm{F}}_{i}^{1},...,{\bm{F}}_{i}^{S}}),\text{ }{\bm{F}}_{i}\in\mathbb{R}^{H/4\times W/4\times C}, |  | (2) |
|---|---|---|---|

where CC represents the feature dimension. Our choice of DPT is motivated by its specialized design for Vision Transformer (ViT) encoders in dense prediction tasks, making it particularly well-suited for integration with depth foundation models. This multi-scale feature fusion is fundamental to our approach, as it enables the aggregated features 𝑭i{\bm{F}}_{i} to simultaneously capture fine-grained geometric details and comprehensive scene-level structural information.

In the second stage, we enhance the multi-view geometric understanding through an efficient multi-view Transformer architecture. For view ii, the cross-view feature refinement can be expressed as:

|  | 𝑭im​v=ℱc​r​o​s​s(𝑭i,𝑭jj∈𝒩​i), 𝑭im​v∈ℝH/4×W/4×Cm​v{\bm{F}}_{i}^{mv}=\mathcal{F}_{cross}({\bm{F}}_{i},{{\bm{F}}_{j}}_{j\in\mathcal{N}i}),\text{ }{\bm{F}}_{i}^{mv}\in\mathbb{R}^{H/4\times W/4\times C_{mv}} |  | (3) |
|---|---|---|---|

where 𝒩i\mathcal{N}_{i} represents the MM nearest neighboring views of view ii, and ℱc​r​o​s​s\mathcal{F}_{cross} denotes the Transformer operations incorporating both self-attention and cross-attention mechanisms. To achieve computational efficiency without sacrificing effectiveness, we implement the local window attention mechanism introduced in Swin Transformer [19]. For sequences containing more than two views, we introduce an efficient attention computation strategy where each view only performs cross-attention operations with its MM nearest neighbors. This approach significantly reduces computational complexity while maintaining the quality of feature interactions across views. The resulting cross-view aware features 𝑭im​v{\bm{F}}_{i}^{mv} effectively encode rich geometric relationships and correspondences across multiple viewpoints.

In addition to the geometry-aware multi-view features 𝑭im​v{\bm{F}}_{i}^{mv}, we also leverage the rich monocular priors by extracting features 𝑭im​o​n​o{\bm{F}}_{i}^{mono} from the frozen depth foundation model. Rather than utilizing features from the encoder’s final layer, we strategically extract features from the decoder’s last layer. This design choice is motivated by two considerations: First, the decoder features have undergone comprehensive multi-scale feature aggregation, enabling more precise depth estimation. Second, these features have been refined through the entire decoder hierarchy, incorporating both local details and global context, thus providing more direct and informative depth cues for subsequent Gaussian prediction. This approach harnesses the foundation model’s pre-trained knowledge while maintaining computational efficiency through feature reuse.

### 3.2 Integrated Gaussian Prediction

Although multi-view features 𝑭im​v{\bm{F}}^{mv}_{i} enable sophisticated cost volume construction for Gaussian parameter estimation, they face inherent limitations in challenging scenarios such as occlusions, texture-less regions, and specular surfaces. To address these fundamental limitations, we propose an innovative approach that integrates rich monocular features 𝑭im​o​n​o{\bm{F}}^{mono}_{i} from the foundation model into both the cost volume computation and subsequent parameter prediction stages. This strategic integration allows our framework to leverage the comprehensive geometric priors acquired through large-scale pre-training, enabling robust Gaussian prediction across diverse and challenging conditions.

Our framework employs a plane-sweep stereo approach to encode cross-view feature-matching information. For each view ii, we first sample a set of uniform depth values {dm}m=1D\{d_{m}\}_{m=1}^{D} between predefined near and far bounds. The warping process proceeds by back-projecting pixels from the reference view ii to 3D points using each sampled depth dmd_{m}, followed by projecting these 3D points onto neighboring views jj through their respective camera matrices. This geometric transformation enables us to warp features from neighboring views 𝑭jm​v{\bm{F}}_{j}^{mv} to align with the reference view ii. We then compute feature correlations 𝑪j→i{\bm{C}}_{j\rightarrow i} through dot product operations between the warped features {𝑭j→im​v}m=1D\{{\bm{F}}^{mv}_{j\rightarrow i}\}_{m=1}^{D} and reference view features 𝑭im​v{\bm{F}}_{i}^{mv}. The final cost volume 𝑪i∈ℝH/4×W/4×D{\bm{C}}_{i}\in\mathbb{R}^{H/4\times W/4\times D} is constructed by aggregating correlation estimates from MM neighboring views, enhancing the robustness of our feature matching process.

The key innovation lies in our integration strategy, where we strategically combine both monocular and multi-view features 𝑭im​o​n​o{\bm{F}}_{i}^{mono} and 𝑭im​v{\bm{F}}_{i}^{mv} with the initial cost volume 𝑪i{\bm{C}}_{i}. This integration is particularly powerful as the monocular features, derived from foundation models pretrained on diverse datasets, provide rich geometric priors that complement traditional multi-view matching. These priors are especially valuable in challenging scenarios where multi-view matching often fails. The integrated representation undergoes refinement through a dedicated network ℱc​o​s​t\mathcal{F}_{cost}:

|  | 𝑪i^=ℱc​o​s​t​([𝑪i,𝑭m​o​n​o,𝑭m​v]).\hat{{\bm{C}}_{i}}=\mathcal{F}_{cost}([{\bm{C}}_{i},{\bm{F}}_{mono},{\bm{F}}_{mv}]). |  | (4) |
|---|---|---|---|

By leveraging both data-driven priors from monocular features and geometric constraints from multi-view matching, our integrated cost volume achieves more robust and accurate depth estimation across diverse real-world scenes.

The refined cost volume 𝑪i^\hat{{\bm{C}}_{i}} undergoes probabilistic processing through a softmax operation, yielding probability distributions 𝑷i∈ℝH/4×W/4×D{\bm{P}}_{i}\in\mathbb{R}^{H/4\times W/4\times D}. These probabilities enable the computation of depth maps 𝑫i∈ℝH/4×W/4{\bm{D}}_{i}\in\mathbb{R}^{H/4\times W/4} through weighted averaging of depth candidates for each view. While cost volume-based approaches inherently face limitations in resolution and discrete matching characteristics, we address these challenges through a refinement strategy that leverages both geometric and learned priors.

Our refinement process begins by upsampling both the depth maps and features through bilinear interpolation. Crucially, we integrate the interpolated monocular features alongside multi-view features and depth maps through a specialized refinement network:

|  | 𝑭i^=ℱf​e​a​t​u​r​e​([i​n​t​(𝑫i),i​n​t​(𝑭im​o​n​o),i​n​t​(𝑭m​v)]),\hat{{\bm{F}}_{i}}=\mathcal{F}_{feature}([int({\bm{D}}_{i}),int({\bm{F}}^{mono}_{i}),int({\bm{F}}^{mv})]), |  | (5) |
|---|---|---|---|

where i​n​t​(⋅)int(\cdot) represents our interpolation function that upsamples inputs to match the image resolution. This integration of monocular features is particularly beneficial for accurate Gaussian parameter prediction, as it provides rich geometric priors that help resolve ambiguities in-depth estimation and local geometry characterization.

The refined features 𝑭i^∈ℝH×W×D\hat{{\bm{F}}_{i}}\in\mathbb{R}^{H\times W\times D} are then processed by two specialized decoder heads: ℱd​e​p​t​h\mathcal{F}_{depth} for precise inference of Gaussian positions 𝝁i\bm{\mu}_{i} and opacities αi\alpha_{i}, and ℱr​a​w\mathcal{F}_{raw} for determining Gaussian covariances 𝚺i\mathbf{\Sigma}_{i} and colors 𝒄i{\bm{c}}_{i}:

|  | 𝝁i,αi=ℱd​e​p​t​h(𝑭i^), 𝚺i,𝒄i=ℱr​a​w(𝑭i^).\bm{\mu}_{i},\alpha_{i}=\mathcal{F}_{depth}(\hat{{\bm{F}}_{i}}),\text{ }\mathbf{\Sigma}_{i},{\bm{c}}_{i}=\mathcal{F}_{raw}(\hat{{\bm{F}}_{i}}). |  | (6) |
|---|---|---|---|

Finally, we aggregate Gaussian attributes from all input images {𝑰i}i=1N\{{\bm{I}}_{i}\}_{i=1}^{N} following established methodologies [1, 6]. The unified merging operation yields a consolidated Gaussian representation:

|  | 𝑮=⋃{𝝁i,αi,𝚺i,𝒄i}i=1N{\bm{G}}={\bigcup\{\bm{\mu}_{i},\alpha_{i},\mathbf{\Sigma}_{i},{\bm{c}}_{i}\}_{i=1}^{N}} |  | (7) |
|---|---|---|---|

This merged representation, enhanced by the integration of monocular priors throughout our pipeline, captures detailed geometric and appearance information that enables high-fidelity novel view synthesis even in challenging scenarios where traditional multi-view approaches might struggle.

### 3.3 Optimization

Our model generates 3D Gaussian primitives for novel view synthesis through a differentiable rendering pipeline. During training, we optimize the network parameters using a combination of reconstruction losses against ground truth RGB images. Specifically, following [1, 6], we employ a weighted sum of mean squared error (ℒm​s​e\mathcal{L}_{mse}) and perceptual LPIPS [53] losses:

|  | ℒ=ℒm​s​e+λl​p​i​p​s​ℒl​p​i​p​s\mathcal{L}=\mathcal{L}_{mse}+\lambda_{lpips}\mathcal{L}_{lpips} |  | (8) |
|---|---|---|---|

where λl​p​i​p​s\lambda_{lpips} balances the contribution of pixel-wise and perceptual supervision.

| Method | RealEstate10k [55] | ACID [15] | Time (s) ↓\downarrow | Params (M) ↓\downarrow | Mem. (GB)↓\downarrow |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
| PSNR ↑\uparrow | SSIM ↑\uparrow | LPIPS ↓\downarrow | PSNR ↑\uparrow | SSIM ↑\uparrow | LPIPS ↓\downarrow |  |  |  |  |
| Du et al.[8] | 24.78 | 0.820 | 0.213 | 26.88 | 0.799 | 0.218 | 1.325 | 125.1 | 19.604 |
| GPNR[36] | 24.11 | 0.793 | 0.255 | 25.28 | 0.764 | 0.332 | 13.340 | 9.6 | 19.441 |
| pixelNeRF [51] | 20.43 | 0.589 | 0.550 | 20.97 | 0.547 | 0.533 | 5.299 | 28.2 | 3.962 |
| MuRF [47] | 26.10 | 0.858 | 0.143 | 28.09 | 0.841 | 0.155 | 0.186 | 5.3 | N/A |
| pixelSplat [1] | 26.09 | 0.863 | 0.136 | 28.27 | 0.843 | 0.146 | 0.104 | 125.4 | 3.002 |
| latentSplat [45] | 23.07 | 0.825 | 0.182 | 24.95 | 0.782 | 0.207 | 0.110 | 187.0 | 3.161 |
| MVSplat [6] | 26.39 | 0.869 | 0.128 | 28.25 | 0.843 | 0.144 | 0.044 | 12.0 | 0.860 |
| eFreeSplat [22] | 26.45 | 0.865 | 0.126 | 28.30 | 0.851 | 0.140 | 0.061 | N/A | N/A |
| Ours | 26.68 | 0.875 | 0.123 | 28.63 | 0.864 | 0.138 | 0.051 | 30.3 | 0.857 |

## 4 Experiments

We present our experimental evaluation as follows: we first describe our experimental setup (Sec. 4.1), then benchmark our method against state-of-the-art baselines (Sec. 4.2), and finally analyze the effectiveness of individual components through ablation studies (Sec. 4.3).

### 4.1 Experimental Setup

Datasets. Our experiments utilize three diverse datasets: RealEstate10K [55], featuring indoor scenes from YouTube real estate tours (67,477 training, 7,289 test sequences); ACID [15], containing aerial landscape footage (11,075 training, 1,972 test sequences); and DTU [12], comprising object-centric captures under controlled conditions. Following [1, 6], we evaluate on 16 DTU scenes from 4 viewpoints each. Both RealEstate10K and ACID include precise camera pose annotations, while DTU helps validate cross-domain generalization capability.

Metrics. We assess the quality of synthesized views through a comprehensive metric suite. The evaluation framework combines pixel-wise assessment (PSNR), structural analysis (SSIM [44]), and perceptual similarity measures (LPIPS [53]). Alongside these quality metrics, we benchmark model efficiency by measuring inference speed and memory requirements. Following established protocols [1, 6], we standardize our evaluation at 256×256256\times 256 resolution to enable direct comparisons.

Implementation details. Our method is implemented in PyTorch with a custom CUDA backend optimized for 3D Gaussian Splatting operations. The framework’s foundation builds upon the ViT-s variant of Depth Anything Model [50], from which we extract intermediate features at strategic network layers following the official implementation protocol. The architecture incorporates DPT heads comprising four layers with 64-dimensional outputs, complemented by a three-layer cross-view transformer for feature integration. For geometric reasoning, our cost volume construction spans a depth range of dn​e​a​r=0.5d_{near}=0.5 to df​a​r=100.0d_{far}=100.0, subdivided into D=128D=128 discrete planes for comprehensive scene sampling. The training objective balances multiple loss terms through carefully tuned coefficients (λl​p​i​p​s=0.05\lambda_{lpips}=0.05), with optimization conducted over 300,000 iterations using a batch size of 14 on a single A100 GPU. Comprehensive ablation studies exploring different ViT variants and detailed architectural specifications are presented in the supplementary materials.

### 4.2 Results

Baselines. We benchmark our approach against several leading methods in scene-level novel view synthesis, spanning three categories: i) Light Field Network approaches, including GPNR [36] and AttnRend [8], ii) NeRF-based solutions represented by pixelNeRF [51] and MuRF [47], and iii) state-of-the-art 3D Gaussian Splatting methods, notably pixelSplat [1], latentSplat [45], MVSplat [6], and eFreeSplat [22].

Assessing image quality. As demonstrated in Table 1, MonoSplat achieves state-of-the-art performance across all visual quality metrics on both RealEstate10K [55] and ACID [15] benchmarks, attributed to our innovative incorporation of depth foundation models into the Gaussian reconstruction framework. Qualitative comparisons with leading open-sourced models in Figure 4 further substantiate our method’s superiority. Notably, MonoSplat demonstrates exceptional performance in challenging scenarios: maintaining fidelity in extreme close-up views (“sofa corner” in 1st row), preserving geometric consistency under large viewpoint variations (“lamp” in 2nd row), and accurately rendering regions with substantial illumination changes (“desktop” in 3rd row). While baseline approaches exhibit visible artifacts in these demanding cases, MonoSplat maintains visual consistency. These results underscore the significance of leveraging pre-trained geometric priors for enhancing the quality and robustness of novel view synthesis across diverse real-world scenarios.

| Method | Re10k→\rightarrowDTU | Re10k→\rightarrowACID |  |  |  |  |
|---|---|---|---|---|---|---|
| PSNR | SSIM | LPIPS | PSNR | SSIM | LPIPS |  |
| pixelSplat [1] | 12.89 | 0.382 | 0.560 | 27.64 | 0.830 | 0.160 |
| MVSplat [6] | 13.94 | 0.473 | 0.385 | 28.15 | 0.841 | 0.147 |
| Ours | 15.25 | 0.605 | 0.291 | 28.24 | 0.848 | 0.145 |

Assessing cross-dataset generalization. Our method demonstrates exceptional generalization capabilities on out-of-distribution novel scenes. To rigorously evaluate this advantage, we conduct comprehensive cross-dataset experiments, utilizing models trained exclusively on RealEstate10K indoor scenes for inference on both ACID outdoor scenes and DTU object-centric captures. As evidenced in Figure 3, MonoSplat consistently produces high-fidelity novel views despite substantial variations in camera distributions and scene characteristics. In comparison, pixelSplat exhibits significant performance degradation, attributable to its feature aggregation strategy’s limitations in domain adaptation, while MVSplat demonstrates reduced effectiveness without robust feature injection mechanisms. These qualitative observations are conclusively validated by quantitative metrics presented in Table 2, where MonoSplat achieves substantial improvements in both PSNR and LPIPS metrics across datasets. Notably, the performance disparity between our method and existing approaches becomes increasingly pronounced as the domain gap widens from ACID to DTU datasets, underscoring our method’s superior cross-domain generalization capabilities.

Assessing model efficiency. As illustrated in Table 1, MonoSplat demonstrates superior performance across both quality metrics and computational efficiency metrics compared to existing approaches. Our method achieves state-of-the-art image quality while maintaining competitive inference speeds - with an encoding time of 0.051s that closely matches MVSplat’s 0.042s. While our total parameter count (30.3M) exceeds MVSplat’s (12.0M) due to the incorporation of depth foundation models, MonoSplat actually requires fewer trainable parameters during training (10.3M). This efficiency stems from our architectural design choice to leverage frozen pre-trained depth models, allowing the majority of parameters to remain fixed. Such design not only reduces memory requirements below those of MVSplat but also enables efficient fine-tuning for downstream tasks with minimal computational and memory overhead.

Assessing geometry reconstruction. Figure 5 showcases MonoSplat’s superior capability in generating high-quality 3D Gaussian primitives compared to state-of-the-art methods pixelSplat [1] and MVSplat [6], achieving exceptional geometry reconstruction with purely photometric supervision. While pixelSplat delivers acceptable 2D rendering quality, its 3D reconstruction exhibits significant artifacts, particularly floating Gaussians that fail to adhere to the true geometric structure. Similarly, MVSplat demonstrates limitations in depth estimation accuracy, especially in regions with complex geometric details. In contrast, MonoSplat produces notably more precise and coherent 3D Gaussians, leveraging our effective integration of monocular features to provide robust geometric priors..

### 4.3 Ablations

We conduct thorough ablations on RealEstate10K [55] to analyze MonoSplat, where all models are trained for 20, 000 iterations with a batch size of 14. Results are shown in Tab. 3 and 6, and we discuss them in detail next.

Importance of frozening depth foundation model. We hypothesize that the strong geometric priors from the depth foundation model are crucial for superior generalization in Gaussian reconstruction. To validate this, we compare our frozen pre-trained foundation model against a variant “Not frozen” where the backbone is fine-tuned during training. As shown in Table 3, allowing backbone updates leads to notable performance degradation, particularly in generalization capability (2.61dB decrement). This occurs because fine-tuning on limited scene data causes the model to overfit and lose the rich geometric priors. Keeping the foundation model frozen preserves these valuable priors for better cross-scenario reconstruction, as demonstrated by the qualitative results in Figure 6, where the fine-tuned model shows more errors on out-of-domain data.

Importance of DPT. The DPT component plays a crucial role in aggregating multi-scale features from the frozen encoder, which is essential for subsequent feature matching. When we remove DPT and directly utilize features from the final encoder layer in our experiments, we observe a dramatic decline in reconstruction quality, as shown in Table 3 and Figure 6. This performance degradation can be attributed to the limitation of single-scale features in supporting effective feature matching within the cost volume. These results validate the efficacy of incorporating DPT for robust feature aggregation.

Importance of cross-view aggregation. Cross-view aggregation enables monocular features to incorporate information from other viewpoints, establishing accurate pixel correspondences. This transformation is critical for subsequent cost volume matching, as the cost volume relies on pixel correspondences to reason about depth. To validate its importance, we design a variant “w/o cross aggregation” by removing the cross-view aggregation module. As shown in Table 3, this leads to degraded reconstruction performance of 0.98 dB decrement, primarily due to the cost volume’s inability to establish reliable correspondences.

Importance of monocular feature integration. As described in Sec. 3.2, we integrate monocular features into the cost volume and also the feature refinement network. To validate its effectiveness, we design three variants with “w/o MF” that removes monocular features, “MF in cost volume” and “MF in refinement” that only integrates monocular features during the cost volume construction and feature refinement stages, respectively. As shown in Table 3 and Figure 6, removing monocular features entirely leads to extreme generalization performance drop, indicating that integrating monocular features is crucial to the generalizability. In addition, integrating monocular features into only one stage can also cause degraded reconstruction performance. These results demonstrate not only the effectiveness of monocular features but also the importance of incorporating them throughout the pipeline to maximize their utility.

| Method | Re10k | Re10k→\rightarrowDTU |  |  |
|---|---|---|---|---|
| PSNR ↑\uparrow | SSIM↑\uparrow | PSNR↑\uparrow | SSIM↑\uparrow |  |
| Not frozen | 25.87 | 0.860 | 12.63 | 0.328 |
| w/o DPT | 25.25 | 0.846 | 14.69 | 0.480 |
| w/o cross aggregation | 25.48 | 0.858 | 10.28 | 0.267 |
| w/o MF | 26.33 | 0.870 | 10.33 | 0.268 |
| MF in cost volume | 26.20 | 0.867 | 15.06 | 0.466 |
| MF in refinement | 26.06 | 0.863 | 14.93 | 0.527 |
| Full model | 26.50 | 0.870 | 15.24 | 0.604 |

## 5 Conclusion

In this paper, we present MonoSplat, a framework that improves 3D Gaussian reconstruction by leveraging pretrained monocular depth models. Our approach transforms monocular priors into accurate Gaussian primitives while maintaining cross-view consistency. MonoSplat achieves state-of-the-art performance on diverse real-world scenes and demonstrates strong zero-shot generalization with minimal computational overhead, highlighting the value of incorporating geometric priors from foundation models.

## 6 Acknowledgment

This work was supported by Hong Kong Research Grants Council (RGC) General Research Fund 11211221 and 14204321, and conclusions contained herein reflect the opinions and conclusions of its authors and no other entity.

Supplementary Material

## Appendix A More Details

### A.1 Data

During training, we employ our custom data loaders for all methods and progressively increase the spacing between reference views. Specifically, we implement a linear increase in view distances over the first 150,000 training steps: the minimum distance between reference views increases from 25 to 45, while the maximum distance expands from 45 to 192. To ensure a fair comparison with previous works [1, 6], input images are resized to a resolution of 256×256. Our pre-processing filters out invalid images, including those with misaligned sizes and images where the maximum field of view exceeds 100 degrees.

### A.2 Model

For the frozen depth encoder, we adopt three variants of Depth Anything V2 [50], which are based on ViT-S, ViT-B, and ViT-L, respectively. For these variants, we extract features of intermediate layers of [2, 5, 8, 11], [2, 5, 8, 11], and [4, 11, 17, 23], respectively, and feed these features to the following DPT decoder and the original depth decoder. For DPT, we set the final output dimension as 64 to balance the efficiency and effectiveness. Following DPT, the multi-view transformer adds position encoding for different views and outputs features of dimension 64.

For the cost volume construction, we use 128 planes and calculate the 2D cost volume, similar to [6], followed by the UNet-style refinement. The cost volume UNet uses a base feature dimension of 128, which was empirically found to balance model capacity and efficiency well. We maintain consistent channel dimensions across three downsampling stages using multipliers [1,1,1]. Self-attention is applied at 1/4 resolution to enhance feature correlation.

After obtaining the predicted depths, we combine them with features for further Gaussian parameter prediction, achieved by a depth UNet. The depth UNet employs a base feature dimension of 32, with channel multipliers [1,1,1,1,1] across five downsampling stages. Attention mechanisms are incorporated at resolution 1/16 to capture long-range dependencies. Finally, we constrain the Gaussian scale within the range [0.5, 15.0]. The minimum scale of 0.5 ensures sufficient detail capture at fine levels. The maximum scale of 15.0 prevents overly large Gaussians while allowing coverage of broader regions. We use spherical harmonics of degree 4 to represent view-dependent appearance.

### A.3 Training

Our default model training is conducted on a single A100 GPU with a batch size of 14. Each batch comprises one training scene consisting of two input views and four target views. Following pixelSplat [1] and MVSplat [6], we progressively increase the frame distance between input views throughout the training process. The near and far depth planes are empirically set to 0.5 and 100 for both RealEstate10K and ACID datasets. For the DTU dataset, we utilize the depth bounds of 2.125 and 4.525.

## Appendix B More Experimental Analysis

All experiments in this section follow the same settings as in Sec. 4.1 unless otherwise specified, which are trained and tested on RealEstate10K [55]. To investigate our hypothesis regarding the advantages of monocular depth foundation model features for generalizable Gaussian reconstruction, we conducted experiments with various frozen backbones, including DINOv2 (used in pixelSplat [1]) and UniMatch (employed in MVSplat [6]). The results, as presented in the Table 4, show substantial deterioration in both in-domain and out-of-domain generalization performance, validating the essential role of depth foundation models. Furthermore, our exploration of different Depth Anything v2 variants revealed a positive correlation between model size and performance, with larger models achieving better reconstruction quality and generalization capabilities.

| Method | Re10k | Re10k→\rightarrowDTU |  |  |
|---|---|---|---|---|
| PSNR ↑\uparrow | SSIM↑\uparrow | PSNR↑\uparrow | SSIM↑\uparrow |  |
| DINOv2 | 26.14 | 0.872 | 13.45 | 0.342 |
| UniMatch | 26.03 | 0.868 | 14.92 | 0.465 |
| DAMv2-S | 26.50 | 0.870 | 15.24 | 0.604 |
| DAMv2-B | 26.83 | 0.875 | 15.62 | 0.620 |
| DAMv2-L | 27.12 | 0.878 | 15.95 | 0.608 |

## Appendix C More Visual Comparisons

In this section, we provide more visual comparisons of geometry reconstruction in Figure 7 and cross-dataset generalization results in Figure 8.
