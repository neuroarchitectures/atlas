# UniDepth: Universal Monocular Metric Depth Estimation

Accurate monocular metric depth estimation (MMDE) is crucial to solving downstream tasks in 3D perception and modeling. However, the remarkable accuracy of recent MMDE methods is confined to their training domains. These methods fail to generalize to unseen domains even in the presence of moderate domain gaps, which hinders their practical applicability. We propose a new model, UniDepth, capable of reconstructing metric 3D scenes from solely single images across domains. Departing from the existing MMDE methods, UniDepth directly predicts metric 3D points from the input image at inference time without any additional information, striving for a universal and flexible MMDE solution. In particular, UniDepth implements a self-promptable camera module predicting dense camera representation to condition depth features. Our model exploits a pseudo-spherical output representation, which disentangles camera and depth representations. In addition, we propose a geometric invariance loss that promotes the invariance of camera-prompted depth features. Thorough evaluations on ten datasets in a zero-shot regime consistently demonstrate the superior performance of UniDepth, even when compared with methods directly trained on the testing domains. Code and models are available at: github.com/lpiccinelli-eth/unidepth.

## 1 Introduction

The precise pixel-wise depth estimation is crucial to understanding the geometric scene structure, with applications in 3D modeling [10], robotics [63, 11], and autonomous vehicles [51, 38]. However, delivering reliable metric scaled depth outputs is necessary to perform 3D reconstruction effectively, thus motivating the challenging and inherently ill-posed task of Monocular Metric Depth Estimation (MMDE).

While existing MMDE methods [14, 16, 3, 43, 40, 61, 41] have demonstrated remarkable accuracy across different benchmarks, they require training and testing on datasets with similar camera intrinsics and scene scales. Moreover, the training datasets typically have a limited size and contain little diversity in scenes and cameras. These characteristics result in poor generalization to real-world inference scenarios [52], where images are captured in uncontrolled, arbitrarily structured environments and cameras with arbitrary intrinsics.

Only a few methods [59, 21] have addressed the challenging task of generalizable MMDE. However, these methods assume controlled setups at test time, including camera intrinsics. While this assumption simplifies the task, it has two notable drawbacks. Firstly, it does not address the full application spectrum, e.g. in-the-wild video processing and crowd-sourced image analysis. Secondly, the inherent camera parameter noise is directly injected into the model, leading to large inaccuracies in the high-noise case.

In this work, we address the more demanding task of generalizable MMDE without any reliance on additional external information, such as camera parameters, thus defining the universal MMDE task. Our approach, named UniDepth, is the first that attempts to solve this challenging task without restrictions on scene composition and setup and distinguishes itself through its general and adaptable nature. Unlike existing methods, UniDepth delivers metric 3D predictions for any scene solely from a single image, waiving the need for extra information about scene or camera. Furthermore, UniDepth flexibly allows for the incorporation of additional camera information at test time.

Our design introduces a camera module that outputs a non-parametric, i.e. dense camera representation, serving as the prompt to the depth module. However, relying only on this single additional module clearly results in challenges related to training stability and scale ambiguity. We propose an effective pseudo-spherical representation of the output space to disentangle the camera and depth dimensions of this space. This representation employs azimuth and elevation angle components for the camera and a radial component for the depth, forming a perfect orthogonal space between the camera plane and the depth axis. Moreover, the camera components are embedded through Laplace spherical harmonic encoding. Figure 1 depicts our camera self-prompting mechanism and the output space. Additionally, we introduce a geometric invariance loss to enhance the robustness of depth estimation. The underlying idea is that the camera-conditioned depth features from two views of the same image should exhibit reciprocal consistency. In particular, we sample two geometric augmentations, creating a pair of different views for each training image, thus simulating different apparent cameras for the original scene.

Our overall contribution is the first universal MMDE method, UniDepth, that predicts a point in metric 3D space for each pixel without any input other than a single image. In particular, first, we design a promptable camera module, an architectural component that learns a dense camera representation and allows for non-parametric camera conditioning. Second, we propose a pseudo-spherical representation of the output space, thus solving the intertwined nature of camera and depth prediction. In addition, we introduce a geometric invariance loss to disentangle the camera information from the underlying 3D geometry of the scene. Moreover, we extensively test UniDepth and re-evaluate seven MMDE State-of-the-Art (SotA) methods on ten different datasets in a fair and comparable zero-shot setup to lay the ground for the generalized MMDE task. Owing to its design, UniDepth consistently sets the new state of the art even compared with non-zero-shot methods, ranking first in the competitive official KITTI Depth Prediction Benchmark.

## 2 Related Work

Metric and Scale-agnostic Depth Estimation. It is crucial to distinguish Monocular Metric Depth Estimation (MMDE) from scale-agnostic, namely up-to-a-scale, monocular depth estimation. MMDE SotA approaches typically confine training and testing to the same domain. However, challenges arise, such as overfitting to the training scenario leading to considerable performance drops in the presence of minor domain gaps, often overlooked in benchmarks like NYU-Depthv2 [35] (NYU) and KITTI [18]. On the other hand, scale-agnostic depth methods, including MiDaS [42], OmniData [13], and LeReS [58], show robust generalization by training on extensive datasets. Their limitation lies in the absence of a metric output, hindering practical usage in downstream applications.

Monocular Metric Depth Estimation. The introduction of end-to-end trainable neural networks in MMDE, pioneered by [14], marked a significant milestone, also introducing the optimization process through the Scale-Invariant log loss (SIlog\mathrm{SI}_{\log}). Subsequent developments witnessed the emergence of advanced networks, ranging from convolution-based architectures [16, 27, 31, 40] to transformer-based approaches [57, 3, 61, 41]. Despite impressive achievements on established benchmarks, MMDE models face challenges in zero-shot scenarios, revealing the need for robust generalization against domain shifts in appearance and geometry.

General Monocular Metric Depth Estimation. Recent efforts focus on developing MMDE models [4, 21, 59] for general depth prediction across diverse domains. These models often leverage camera awareness, either by directly incorporating external camera parameters into computations [15, 21] or by normalizing the shape or output depth based on intrinsic properties, as seen in [28, 1, 59].

However, these generalizable MMDE methods often adopt specific strategies to enhance performance, e.g. geometric pretraining [4] or dataset-specific prior like reshaping [59]. In addition, these methods assume access to noiseless camera intrinsics both at training and test time, also limiting their applicability to pinhole camera models. Additionally, SotA methods depend on a predefined backprojection operation, blurring the distinction between learning depth and the 3D scene. In contrast, our approach aims to overcome these limitations, presenting a more demanding perspective, e.g. universal MMDE. Universal MMDE involves directly predicting the 3D scene from the input image without any additional information other than the latter. Notably, we do not require any additional prior information at test time, such as access to camera information.

## 3 UniDepth

MMDE SotA methods typically assume access to the camera intrinsics, thus blurring the line between pure depth estimation and actual 3D estimation. In contrast, UniDepth aims to create a universal MMDE model deployable in diverse scenarios without relying on any other external information, such as camera intrinsic, thus leading to 3D space estimation by design. However, attempting to directly predict 3D points from a single image without a proper internal representation neglects geometric prior knowledge, i.e. perspective geometry, burdening the learning process with re-learning laws of perspective projection from data.

Sec. 3.1 introduces a pseudo-spherical representation of the output space to inherently disentangle camera rays’ angles from depth. In addition, our preliminary studies indicate that depth prediction clearly benefits from prior information on the acquisition sensor, leading to the introduction of a self-prompting camera operation in Sec. 3.2. Further disentanglement at the level of internal depth features is achieved through a geometric invariance loss, outlined in Sec. 3.3. This loss ensures depth features remain invariant when conditioned on the bootstrapped camera predictions, promoting robust camera-aware depth predictions. The overall architecture and the resulting optimization induced by the combination of design choices are detailed in Sec. 3.4.

### 3.1 3D Representation

The general purpose nature of our MMDE method requires inferring both depth and camera intrinsics to make 3D predictions based only on imagery observations. We design the 3D output space presenting a natural disentanglement of the two sub-tasks, namely depth estimation and camera calibration. In particular, we exploit the pseudo-spherical representation where the basis is defined by azimuth, elevation, and log-depth, i.e. (θ\theta,ϕ\phi,zlogz_{\log}), in contrast to the Cartesian representation (xx,yy,zz). The strength of the proposed pseudo-spherical representation lies in the decoupling of camera (θ\theta,ϕ\phi) and depth (zlogz_{\log}) components, ensuring their orthogonality by design, in contrast to the entanglement present in Cartesian representation.

It is worth highlighting that in this output space, the non-parametric dense representation of the camera is mathematically represented as a tensor 𝐂∈ℝH×W×2\mathbf{C}\in\mathbb{R}^{H\times W\times 2}, where HH and WW are the height and width of the input image and the last dimension corresponds to azimuth and elevation values. While in the typical Cartesian space, the backprojection involves the multiplication of homogeneous camera rays and depth, the backprojection operation in the proposed representation space accounts for the concatenation of camera and depth representations. The pencil of rays are defined as (𝐫1,𝐫2,𝐫3)=𝐊−1​[𝐮,𝐯,𝟏]T(\mathbf{r}_{1},\mathbf{r}_{2},\mathbf{r}_{3})=\mathbf{K}^{-1}[\mathbf{u},\mathbf{v},\mathbf{1}]^{T}, where 𝐊\mathbf{K} is the calibration matrix, 𝐮\mathbf{u} and 𝐯\mathbf{v} are pixel positions in pixel coordinates, and 𝟏\mathbf{1} is a vector of ones. Therefore, the homogeneous camera rays (𝐫x,𝐫y)(\mathbf{r}_{x},\mathbf{r}_{y}) correspond to (𝐫1𝐫3,𝐫2𝐫3)(\frac{\mathbf{r}_{1}}{\mathbf{r}_{3}},\frac{\mathbf{r}_{2}}{\mathbf{r}_{3}}).

Moreover, the angular dense representation can be embedded via the Laplace Spherical Harmonic Encoding (SHE). The camera embedding tensor is defined as 𝐄=SHE⁡(𝐂),𝐄∈ℝH×W×d\mathbf{E}=\mathrm{SHE}(\mathbf{C}),\mathbf{E}\in\mathbb{R}^{H\times W\times d}, where dd is the number of harmonics chosen. SHE⁡(⋅)\mathrm{SHE}(\cdot) computes the set of spherical harmonics, i.e., {𝒴}l,m\{\mathcal{Y}\}_{l,m} with degree ll and order mm, and concatenating along the channel dimension, with 𝐘ml\mathbf{Y}^{l}_{m} as

|  | Yml​(θ,ϕ)=αml​𝒫ml​(cos⁡θ)​ei​m​ϕ,Y^{l}_{m}(\theta,\phi)=\alpha^{l}_{m}\mathcal{P}^{l}_{m}(\cos\theta)e^{im\phi}, |  | (1) |
|---|---|---|---|

where 𝒫ml\mathcal{P}^{l}_{m} is the associated Legendre polynomial of degree ll and order mm, and αml\alpha^{l}_{m} is a normalizing constant. In particular, the spherical harmonics on the unit sphere form an orthogonal basis of the spherical manifold and preserve inner products. The total number of harmonics utilized is 8181, resulting from capping the degree ll to 8. SHE is utilized as a mathematic sounder choice compared to, e.g. the Fourier Transform, to produce the camera embeddings.

### 3.2 Self-Promptable Camera

The camera module plays a crucial role in the final 3D predictions since its angular dense output accounts for two dimensions of the output space, namely azimuth and elevation. Most importantly, these embeddings prompt the depth module to ensure a bootstrapped prior knowledge of the input scene’s global depth scale. The prompting is fundamental to avoid mode collapse in the scene scale and to alleviate the depth module from the burden of predicting depth from scratch as the scale is already modeled by camera output.

Nonetheless, the internal representation of the camera module is based on a pinhole parameterization, namely via focal length (fxf_{x}, fyf_{y}) and principal point (cxc_{x}, cyc_{y}). The four tokens conceptually corresponding to the intrinsics are then projected to scalar values, i.e., Δ​fx\Delta f_{x}, Δ​fy\Delta f_{y}, Δ​cx\Delta c_{x}, Δ​cy\Delta c_{y}. However, they do not directly represent the camera parameters, but the multiplicative residuals to a pinhole camera initialization, namely H2\frac{H}{2} for y-components and W2\frac{W}{2} for x-components, leading to fx=Δ​fx​W2f_{x}=\frac{\Delta f_{x}W}{2}, fy=Δ​fy​H2f_{y}=\frac{\Delta f_{y}H}{2}, cx=Δ​cx​W2c_{x}=\frac{\Delta c_{x}W}{2}, cy=Δ​cy​H2c_{y}=\frac{\Delta c_{y}H}{2}, leading to invariance towards input image sizes.

Subsequently, a backprojection operation based on the intrinsic parameters is applied to every pixel coordinate to produce the corresponding rays. The rays are normalized and thus represent vectors on a unit sphere. The critical step involves extracting azimuth and elevation from the backprojected rays, effectively creating a “dense” angular camera representation. This dense representation undergoes SHE to produce the embeddings 𝐄\mathbf{E}. The embedded representations are then seamlessly passed to the depth module as a prompt, where they play a vital role as a conditioning factor. The conditioning is enforced via a cross-attention layer between the initialized feature of Depth Module 𝐃∈ℝh×w×C\mathbf{D}\in\mathbb{R}^{h\times w\times C} and the camera embeddings 𝐄\mathbf{E} where (h,w)=(H/16,W/16)(h,w)=(H/16,W/16). The camera-prompted depth features 𝐃|𝐄∈ℝh×w×C\mathbf{D|E}\in\mathbb{R}^{h\times w\times C} are defined as

|  | 𝐃|𝐄=MLP⁡(CA⁡(𝐃,𝐄)),\mathbf{D|E}=\mathrm{MLP}(\mathrm{CA}(\mathbf{D},\mathbf{E})), |  | (2) |
|---|---|---|---|

where CA\mathrm{CA} is a cross-attention block and MLP\mathrm{MLP} is a MultiLayer Perceptron with one 4​C4C-channel hidden layer.

Figure 3 illustrates one of the main benefits of our camera module. In particular, in high-noise intrinsics or camera-agnostic scenarios, UniDepth can bootstrap the camera prediction, thus displaying total noise insensitivity. However, we can substitute the camera module output to improve 3D reconstruction peak performance if any external dense camera representation is provided. This adaptability enhances the model’s versatility, allowing it to operate seamlessly in diverse setups. Moreover, Figure 3 suggests that training with noisy self-prompts enhances the robustness of UniDepth to noisier external intrinsics if given at test time.

### 3.3 Geometric Invariance Loss

The spatial locations from the same scene captured by different cameras should correspond when the depth module is conditioned on the specific camera. To this end, we propose a geometric invariance loss to enforce the consistency of camera-prompted depth features of the same scene from different acquisition sensors. In particular, consistency is enforced on features extracted from identical 3D locations.

For each image, we perform NN distinct geometrical augmentations, denoted as {𝒯i}i=1N\{\mathcal{T}_{i}\}_{i=1}^{N}, with N=2N=2 in our experiments. This operation involves involves sampling a rescaling factor r∼2𝒰[−1,1]r\sim 2^{\mathcal{U}_{[-1,1]}} and a relative translation on the xx-axis t∼𝒰[−0.1,0.1]t\sim\mathcal{U}_{[-0.1,0.1]}, then cropping it to the network’s input shape. This is analogous to sampling a pair of images from the same scene and extrinsic parameters but captured by different cameras. Let 𝐂i\mathbf{C}_{i} and 𝐃𝐢|𝐄𝐢\mathbf{D_{i}|E_{i}} describe the predicted camera representation and camera-prompted depth features, respectively, corresponding to augmentation 𝒯i\mathcal{T}_{i}. It is evident that the camera representations differ when two diverse geometric augmentations are applied, i.e., 𝐂i≠𝐂j\mathbf{C}_{i}\neq\mathbf{C}_{j} if 𝒯i≠𝒯j\mathcal{T}_{i}\neq\mathcal{T}_{j}. Therefore, the geometric invariance loss can be expressed as

|  | ℒcon(𝐃1|𝐄1,𝐃2|𝐄2)=‖𝒯2∘𝒯1−1∘(𝐃1|𝐄1)−sg⁡(𝐃2|𝐄2)‖1,\begin{split}&\mathcal{L}_{\mathrm{con}}(\mathbf{D}_{1}|\mathbf{E}_{1},\mathbf{D}_{2}|\mathbf{E}_{2})=\\ &\left\|\mathcal{T}_{2}\circ\mathcal{T}^{-1}_{1}\circ(\mathbf{D}_{1}|\mathbf{E}_{1})-\mathrm{sg}(\mathbf{D}_{2}|\mathbf{E}_{2})\right\|_{1},\end{split} |  | (3) |
|---|---|---|---|

where 𝐃i|𝐄i\mathbf{D}_{i}|\mathbf{E}_{i} represents the depth feature after being conditioned by camera prompt 𝐄i\mathbf{E}_{i}, as outlined in Sec. 3.2, and sg⁡(⋅)\mathrm{sg}(\cdot) corresponds to the stop-gradient detach operation needed to exploit 𝐃2|𝐄2\mathbf{D}_{2}|\mathbf{E}_{2} as pseudo ground-truth (GT). The bi-directional loss can be computed as: 12(ℒcon(𝐃1|𝐄1,𝐃2|𝐄2)+ℒcon(𝐃2|𝐄2,𝐃1|𝐄1))\frac{1}{2}(\mathcal{L}_{\mathrm{con}}(\mathbf{D}_{1}|\mathbf{E}_{1},\mathbf{D}_{2}|\mathbf{E}_{2})+\mathcal{L}_{\mathrm{con}}(\mathbf{D}_{2}|\mathbf{E}_{2},\mathbf{D}_{1}|\mathbf{E}_{1})). It is necessary to apply the geometric invariance loss after the features are conditioned on the viewing information, i.e., camera. Otherwise, the loss would enforce consistency across features that inherently carry distinct camera information.

### 3.4 Network Design

Architecture. Our network, described in Fig. 2, comprises an Encoder Backbone, a Camera Module, and a Depth Module. The encoder can be either convolutional or ViT-based [12], producing features at different “scales”, i.e. 𝐅∈ℝh×w×C×B\mathbf{F}\in\mathbb{R}^{h\times w\times C\times B}, where (h,w)=(H16,W16)(h,w)=(\frac{H}{16},\frac{W}{16}) and B=4B=4.

The Camera Module parameters are initialized class tokens for ViT-style or pooled feature maps for convolutional-style backbones. The encoded features from the Encoder Backbone are passed to the Camera Module as a stack of detached tokens, the encoder class tokens are utilized as camera parameters initialization. The features are processed to obtain the final dense representation 𝐂\mathbf{C} as detailed in Sec. 3.2, and further embedded to 𝐄\mathbf{E} via SHE⁡(⋅)\mathrm{SHE}(\cdot) outlined in Sec. 3.1. Note that the stop-gradient operation is necessary because of the low variety of effective cameras compared to the image diversity. In fact, the Camera Module component easily overfits and clearly dominates the overall backbone gradient.

The Depth Module is fed with the encoder features to condition the initial latent features 𝐋∈ℝh×w×C\mathbf{L}\in\mathbb{R}^{h\times w\times C} via one cross-attention layer to obtain the initial depth features, 𝐃\mathbf{D}. The latent feature tensor 𝐋\mathbf{L} is obtained as the average of the features 𝐅\mathbf{F} along the BB dimension. Furthermore, the depth features are conditioned on the camera prompts 𝐄\mathbf{E} to obtain 𝐃|𝐄\mathbf{D|E} as described in Sec. 3.2. The camera-prompted depth features are further processed via self-attention layers where the positional encoding utilized is 𝐄\mathbf{E} and upsampled to produce a multi-scale output. The log-depth prediction 𝐙log∈ℝH×W×1\mathbf{Z}_{\log}\in\mathbb{R}^{H\times W\times 1} corresponds to the mean of the interpolated intermediate representations. The final 3D output 𝐎∈ℝH×W×3\mathbf{O}\in\mathbb{R}^{H\times W\times 3} is the concatenation of predicted rays and depth, 𝐎=𝐂||𝐙\mathbf{O}=\mathbf{C}||\mathbf{Z}, with 𝐙\mathbf{Z} as element-wise exponentiation of 𝐙log\mathbf{Z}_{\log}.

Optimization. The optimization process is guided by a re-formulation of the Mean Squared Error (MSE) loss in the final 3D output space (θ\theta,ϕ\phi,zlogz_{\log}) from Sec. 3.1 as:

|  | ℒλ​MSE​(𝜺)=‖𝕍⁡[𝜺]‖1+𝝀T​(𝔼⁡[𝜺]⊙𝔼⁡[𝜺]),\vskip-4.0pt\begin{split}\mathcal{L}_{\lambda\mathrm{MSE}}(\bm{\varepsilon})=\|\mathbb{V}[\bm{\varepsilon}]\|_{1}+\bm{\lambda}^{T}(\mathbb{E}[\bm{\varepsilon}]\odot\mathbb{E}[\bm{\varepsilon}]),\end{split} |  | (4) |
|---|---|---|---|

where 𝜺=𝐨^−𝐨∗∈ℝ3\bm{\varepsilon}=\hat{\mathbf{o}}-\mathbf{o}^{*}\in\mathbb{R}^{3}, 𝐨^=(θ^,ϕ^,z^log)\hat{\mathbf{o}}=(\hat{\theta},\hat{\phi},\hat{z}_{\log}) is the predicted 3D output, 𝐨∗=(θ∗,ϕ∗,zlog∗)\mathbf{o}^{*}=(\theta^{*},\phi^{*},z_{\log}^{*}) is the GT 3D value, and 𝝀=(λθ,λϕ,λz)∈ℝ3\bm{\lambda}=(\lambda_{\theta},\lambda_{\phi},\lambda_{z})\in\mathbb{R}^{3} is a vector of weights for each dimension of the output. 𝕍⁡[𝜺]\mathbb{V}[\bm{\varepsilon}] and 𝔼⁡[𝜺]\mathbb{E}[\bm{\varepsilon}] are computed as the vectors of empirical variances and means for each of the three output dimensions over all pixels, i.e. {𝜺(i)}i=1N\{\bm{\varepsilon}^{(i)}\}_{i=1}^{N}. Note that if λd=1\lambda_{d}=1 for a dimension dd, the loss represents the standard MSE loss for that dimension. If λd<1\lambda_{d}<1, a scale-invariant loss term is added to that dimension if it is expressed in log space, e.g. for the depth dimension zlogz_{\log} and a shift-invariant loss term is added if that output is expressed in linear space. In particular, if only the last output dimension is considered, i.e., the one corresponding to depth, and λz=0.15\lambda_{z}=0.15 is utilized, the corresponding loss is the standard SIlog\mathrm{SI}_{\log}. In our experiments, we set λθ=λϕ=1\lambda_{\theta}=\lambda_{\phi}=1 and λz=0.15\lambda_{z}=0.15. Therefore, the final optimization loss is defined as

|  | ℒ=ℒλ​MSE+α​ℒcon, with ​α=0.1.\vskip-5.0pt\mathcal{L}=\mathcal{L}_{\lambda\mathrm{MSE}}+\alpha\mathcal{L}_{\mathrm{con}},\text{ with }\alpha=0.1. |  | (5) |
|---|---|---|---|

The loss defined here serves as a motivation for the designed output representation. Specifically, employing a Cartesian representation and applying the loss directly to the output space would result in backpropagation through (xx, yy), and zlogz_{\log} errors. However, xx and yy components are derived as rx⋅zr_{x}\cdot z and ry⋅zr_{y}\cdot z as detailed in Sec. 3.1. Consequently, the gradients of camera components, expressed by (rxr_{x}, ryr_{y}), and of depth become intertwined, leading to suboptimal optimization as discussed in Sec. 4.3.

## 4 Experiments

| Method | NuScenes | DDAD | ETH3D | Diode (Indoor) | SUN-RGBD | VOID | IBims-1 | HAMMER |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow |  |
| BTS [28] | 33.733.7 | 68.068.0 | 37.537.5 | 43.043.0 | 40.840.8 | 40.540.5 | 26.826.8 | 29.929.9 | 27.427.4 | 19.219.2 | 22.822.8 | 31.631.6 | 76.176.1 | 14.614.6 | 64.864.8 | 47.447.4 | 25.825.8 | 64.564.5 | 53.153.1 | 17.517.5 | 57.257.2 | 3.893.89 | 20.920.9 | 22.822.8 |
| AdaBins [3] | 33.333.3 | 61.461.4 | 35.235.2 | 37.737.7 | 44.444.4 | 35.635.6 | 24.324.3 | 28.328.3 | 25.225.2 | 17.417.4 | 21.621.6 | 28.728.7 | 77.777.7 | 13.913.9 | 65.465.4 | 50.550.5 | 23.823.8 | 65.065.0 | 55.055.0 | 15.615.6 | 57.857.8 | 7.217.21 | 21.521.5 | 27.727.7 |
| NeWCRF [61] | 44.244.2 | 49.449.4 | 42.242.2 | 45.645.6 | 34.934.9 | 41.641.6 | 35.735.7 | 26.126.1 | 32.332.3 | 20.120.1 | 18.518.5 | 35.335.3 | 75.375.3 | 11.911.9 | 61.661.6 | 53.153.1 | 22.322.3 | 67.967.9 | 53.653.6 | 14.714.7 | 59.259.2 | 1.431.43 | 14.914.9 | 20.820.8 |
| iDisc [41] | 39.439.4 | 37.137.1 | 34.534.5 | 28.428.4 | 32.232.2 | 25.825.8 | 35.635.6 | 27.527.5 | 31.431.4 | 23.823.8 | 15.815.8 | 33.433.4 | 83.783.7 | 12.412.4 | 71.071.0 | 55.355.3 | 20.320.3 | 68.668.6 | 48.948.9 | 13.213.2 | 55.455.4 | 2.582.58 | 14.014.0 | 32.632.6 |
| ZoeDepth [4] | 28.328.3 | 31.531.5 | 26.026.0 | 27.227.2 | 31.731.7 | 21.121.1 | 35.035.0 | 17.617.6 | 26.426.4 | 36.936.9 | 12.812.8 | 40.540.5 | 86.786.7 | 9.589.58 | 75.675.6 | 63.463.4 | 15.915.9 | 72.472.4 | 58.058.0 | 10.910.9 | 59.659.6 | 0.720.72 | 9.789.78 | 21.021.0 |
| Metric3D† [59] | 72.372.3 | 29.029.0 | 53.953.9 | −- | −- | −- | 45.6¯\underline{45.6} | 18.918.9 | 35.9\mathbf{35.9} | 39.239.2 | 11.111.1 | 42.142.1 | 15.415.4 | 13.413.4 | 14.414.4 | 65.965.9 | 16.216.2 | 70.470.4 | 79.7\mathbf{79.7} | 10.110.1 | 68.5\mathbf{68.5} | 3.403.40 | 12.112.1 | 29.029.0 |
| UniDepth-C | 83.3¯\underline{83.3} | 22.9¯\underline{22.9} | 62.362.3 | 83.2¯\underline{83.2} | 21.4¯\underline{21.4} | 59.359.3 | 49.8\mathbf{49.8} | 13.213.2 | 33.7¯\underline{33.7} | 60.260.2 | 9.039.03 | 50.050.0 | 94.8¯\underline{94.8} | 8.108.10 | 81.4¯\underline{81.4} | 86.686.6 | 12.8¯\underline{12.8} | 85.185.1 | 79.7\mathbf{79.7} | 8.928.92 | 66.7¯\underline{66.7} | 20.2\mathbf{20.2} | 8.78¯\underline{8.78} | 57.1\mathbf{57.1} |
| UniDepth-V | 86.2\mathbf{86.2} | 21.7\mathbf{21.7} | 64.2\mathbf{64.2} | 86.4\mathbf{86.4} | 20.3\mathbf{20.3} | 61.8\mathbf{61.8} | 32.632.6 | 11.6¯\underline{11.6} | 24.324.3 | 77.1¯\underline{77.1} | 6.38¯\underline{6.38} | 59.4\mathbf{59.4} | 96.6\mathbf{96.6} | 7.05\mathbf{7.05} | 81.9\mathbf{81.9} | 89.4¯\underline{89.4} | 10.9\mathbf{10.9} | 85.7¯\underline{85.7} | 23.923.9 | 7.22¯\underline{7.22} | 37.137.1 | 13.3¯\underline{13.3} | 7.41\mathbf{7.41} | 55.9¯\underline{55.9} |
| UniDepth-C‡ | 83.3¯\underline{83.3} | 22.9¯\underline{22.9} | 60.960.9 | 83.183.1 | 21.4¯\underline{21.4} | 57.357.3 | 22.922.9 | 13.113.1 | 25.425.4 | 60.460.4 | 9.019.01 | 49.949.9 | 92.392.3 | 8.278.27 | 75.275.2 | 86.586.5 | 12.8¯\underline{12.8} | 85.085.0 | 79.4¯\underline{79.4} | 8.888.88 | 64.264.2 | 12.712.7 | 9.309.30 | 54.854.8 |
| UniDepth-V‡ | 86.2\mathbf{86.2} | 21.7\mathbf{21.7} | 63.0¯\underline{63.0} | 86.4\mathbf{86.4} | 20.3\mathbf{20.3} | 60.4¯\underline{60.4} | 17.617.6 | 11.4\mathbf{11.4} | 21.421.4 | 77.4\mathbf{77.4} | 6.36\mathbf{6.36} | 58.6¯\underline{58.6} | 94.8¯\underline{94.8} | 7.17¯\underline{7.17} | 75.975.9 | 90.2\mathbf{90.2} | 10.9\mathbf{10.9} | 86.2\mathbf{86.2} | 17.517.5 | 7.20\mathbf{7.20} | 36.536.5 | 2.562.56 | 8.358.35 | 53.853.8 |

### 4.1 Experimental Setup

In-domain training datasets. The training dataset utilized is the ensemble of Argoverse2 [53], Waymo [49], DrivingStereo [56], Cityscapes [7], BDD100K [60], Mapillary-PSD [1], A2D2 [19], ScanNet [8], and Taskonomy [62]. The resulting dataset amounts roughly to 3M real-world images with different cameras and domains, compared to, e.g. Metric3D [59] and ZeroDepth [21] which exploit 8M and 17M training images, respectively.

Zero-shot testing datasets. We evaluate the generalizability of the compared models by testing them on ten datasets not seen during training. More precisely, each method is tested on validation splits from SUN-RGBD [48] without NYU split, Diode Indoor [50] , IBims-1 [26], VOID [54] HAMMER [25], ETH-3D [44], nuScenes [5], and DDAD [20] with split proposed in [41] and evaluated with official masks. Also, UniDepth and the models from [59, 21] are zero-shot-tested on NYU-Depth V2 [35] and KITTI [18]. In particular, KITTI testing is performed on the corrected Eigen-split test set [14] with the Garg evaluation mask [17], while NYU testing uses the evaluation mask from [28].

Evaluation Details. All methods have been re-evaluated with a fair and consistent pipeline. In particular, we do not exploit any test-time augmentations. We use training image shapes for zero-shot testing and evaluate on the same validation splits and masks. Unfortunately, ZeroDepth lacks full code reproducibility, thus we report results from the original paper only, and for visualization, we utilize their provided code and weights. When methods do not report the configuration for a specific test dataset, we use the settings of NYU and KITTI for indoor and outdoor testing, respectively. We utilize common depth estimation evaluation metrics: root mean square error (RMS\mathrm{RMS}) and its log variant (RMSlog\mathrm{RMS_{log}}), absolute mean relative error (A.Rel\mathrm{A.Rel}), the percentage of inlier pixels (δi\mathrm{\delta}_{i}) with threshold 1.25i1.25^{i}, scale-invariant error in log-scale (SIlog\mathrm{SI_{log}}): 100​Var⁡(εlog)100\sqrt{\mathrm{Var}(\varepsilon_{\log})}. In addition, we report point-cloud-based metrics proposed in [37], namely Chamfer Distance (CD\mathrm{CD}) and F-score (FA\mathrm{F_{A}}), with the latter aggregated as the area under the curve up to 1/201/20 of the datasets’ maximum depth. All methods exploit GT intrinsics during evaluation. Nonetheless, we present results both with and without GT intrinsics for UniDepth.

Implementation Details. UniDepth is implemented in PyTorch [39] and CUDA [36]. For training, we use the AdamW [34] optimizer (β1=0.9\beta_{1}=0.9, β2=0.999\beta_{2}=0.999) with an initial learning rate of 0.00010.0001. The learning rate is divided by a factor of 10 for the backbone weights for every experiment and weight decay is set to 0.10.1. As the learning rate scheduler, we exploit Cosine Annealing to one-tenth starting from 30% of the training. We run 1M optimization iterations with a batch size of 128, each training dataset is uniformly represented in each batch. In particular, we sample 64 images and then we sample two different augmented views of the same image for consistency loss. The augmentations include both geometric and appearance (random brightness, gamma, saturation, hue shift, and grayscale) augmentations. ViT-L [12] backbone is initialized with weights from DINO-pre-trained [6] models, and ConvNext-L [33] is ImageNet [9]-pre-trained. The required training time amounts to roughly 12 days on 8 NVIDIA A100. Ablations are conducted with three different seeds and for 100k training iterations, using a randomly sampled subset with a size equal to 20% of the original training set.

| Method | δ0.5\mathrm{\delta}_{{0.5}} | δ1\mathrm{\delta}_{1} | FA\mathrm{F}_{A} | A.Rel\mathrm{A.Rel} | RMS\mathrm{RMS} | RMSlog\mathrm{RMS}_{\log} | CD\mathrm{CD} | SIlog\mathrm{SI}_{\log} |
|---|---|---|---|---|---|---|---|---|
| Higher is better | Lower is better |  |  |  |  |  |  |  |
| BTS [28] | 66.166.1 | 88.588.5 | 74.074.0 | 10.910.9 | 0.3910.391 | 0.1410.141 | 0.1600.160 | 11.511.5 |
| AdaBins [3] | 68.168.1 | 90.190.1 | 74.774.7 | 10.310.3 | 0.3650.365 | 0.1310.131 | 0.1560.156 | 10.610.6 |
| NeWCRF [61] | 69.669.6 | 92.192.1 | 75.875.8 | 9.569.56 | 0.3330.333 | 0.1190.119 | 0.1470.147 | 9.169.16 |
| iDisc [41] | 74.574.5 | 93.893.8 | 78.278.2 | 8.618.61 | 0.3130.313 | 0.1100.110 | 0.1330.133 | 8.858.85 |
| ZoeDepth† [4] | 78.478.4 | 95.295.2 | 80.180.1 | 7.707.70 | 0.2780.278 | 0.0970.097 | 0.1250.125 | 7.197.19 |
| ZeroDepth [21] | −- | 90.190.1 | −- | 10.010.0 | 0.3800.380 | −- | −- | −- |
| Metric3D [59] | 76.376.3 | 92.692.6 | 77.877.8 | 9.389.38 | 0.3370.337 | 0.1200.120 | 0.1460.146 | 9.139.13 |
| UniDepth-C | 85.4¯\underline{85.4} | 97.2¯\underline{97.2} | 84.3¯\underline{84.3} | 6.26¯\underline{6.26} | 0.232¯\underline{0.232} | 0.082¯\underline{0.082} | 0.101¯\underline{0.101} | 6.41¯\underline{6.41} |
| UniDepth-V | 88.6\mathbf{88.6} | 98.4\mathbf{98.4} | 85.9\mathbf{85.9} | 5.78\mathbf{5.78} | 0.201\mathbf{0.201} | 0.073\mathbf{0.073} | 0.092\mathbf{0.092} | 5.27\mathbf{5.27} |

### 4.2 Comparison with the State of the Art

Our method consistently outperforms previous SotA methods as shown in Table 1. We particularly excel in the scale-invariant aspect, represented by SIlog\mathrm{SI}_{\log}, with an average 34.0% improvement, and an average 12.3% improvement for δ1\mathrm{\delta}_{1} and FA\mathrm{F}_{A}. However, UniDepth could fail to capture the specific scene scales in certain cases, e.g. in ETH3D and IBims-1. This pitfall is demonstrated by the drop in scale-dependent metrics, e.g. FA\mathrm{F}_{A} drop is 11.8% and 31.4%, respectively, although having a clear scale-invariant improvement of 36.9% and 28.5%. Therefore, we speculate that our method would still greatly benefit from domain-specific fine-tuning.

| Method | δ0.5\mathrm{\delta}_{{0.5}} | δ1\mathrm{\delta}_{1} | FA\mathrm{F}_{A} | A.Rel\mathrm{A.Rel} | RMS\mathrm{RMS} | RMSlog\mathrm{RMS}_{\log} | CD\mathrm{CD} | SIlog\mathrm{SI}_{\log} |
|---|---|---|---|---|---|---|---|---|
| Higher is better | Lower is better |  |  |  |  |  |  |  |
| BTS [28] | 86.986.9 | 96.296.2 | 82.082.0 | 5.635.63 | 2.432.43 | 0.0890.089 | 0.420.42 | 8.188.18 |
| AdaBins [3] | 86.286.2 | 96.396.3 | 81.581.5 | 5.855.85 | 2.382.38 | 0.0890.089 | 0.4290.429 | 8.108.10 |
| NeWCRF [61] | 88.988.9 | 97.597.5 | 82.782.7 | 5.205.20 | 2.072.07 | 0.0780.078 | 0.3880.388 | 7.007.00 |
| iDisc [41] | 89.289.2 | 97.597.5 | 83.183.1 | 5.095.09 | 2.072.07 | 0.0770.077 | 0.3800.380 | 7.117.11 |
| ZoeDepth† [4] | 87.487.4 | 96.596.5 | 82.182.1 | 5.765.76 | 2.392.39 | 0.0890.089 | 0.4310.431 | 7.477.47 |
| ZeroDepth [21] | −- | 89.289.2 | −- | 10.210.2 | 4.384.38 | 0.1960.196 | −- | −- |
| Metric3D [59] | 88.988.9 | 97.597.5 | 82.982.9 | 5.335.33 | 2.262.26 | 0.0810.081 | 0.3920.392 | 7.287.28 |
| UniDepth-C | 91.1¯\underline{91.1} | 97.9¯\underline{97.9} | 83.9¯\underline{83.9} | 4.69¯\underline{4.69} | 2.00¯\underline{2.00} | 0.072¯\underline{0.072} | 0.371¯\underline{0.371} | 6.71¯\underline{6.71} |
| UniDepth-V | 93.4\mathbf{93.4} | 98.6\mathbf{98.6} | 85.0\mathbf{85.0} | 4.21\mathbf{4.21} | 1.75\mathbf{1.75} | 0.064\mathbf{0.064} | 0.338\mathbf{0.338} | 5.84\mathbf{5.84} |

| KITTI |  |  |  |  |  |  |
|---|---|---|---|---|---|---|
| NYUv2 |  |  |  |  |  |  |
| Diode |  |  |  |  |  |  |
|  | RGB & GT | ZoeDepth† [4] | ZeroDepth [21] | Metric3D [59] | UniDepth | Meters || A.Rel\mathrm{A.Rel} |

The last two rows in Table 1 present UniDepth in its whole design, namely functioning with solely the input image by self-prompting the predicted dense camera representation, as detailed in Eq. 3. Experiments show that not only is the performance preserved for most of the test sets, but UniDepth with the bootstrapped camera can also outperform models with GT camera, e.g. SIlog\mathrm{SI}_{\log} in ETH3D and IBims-1. On the other hand, in cases with particularly out-of-domain camera types, such as ETH3D or HAMMER, bootstrapping camera prediction results in additional noise for scaled depth prediction, thus worsening results for δ1\mathrm{\delta}_{1}.

Table 2 and Table 3 display results on the two popular benchmark NYU [35] and KITTI [18] Eigen-split. UniDepth sets the state of the art in these two benchmarks despite being compared with models trained on the same domain. Importantly, the KITTI Depth Prediction Benchmark, which provides a perfectly fair evaluation, underscores the excellent zero-shot performance of our method and its robustness compared to the current MMDE SotA methods, as UniDepth ranks first on this benchmark at the time of submission, with a 15.5% improvement in SIlog\mathrm{SI}_{\log} over the second-best method. Performance disparities are not solely attributed to dataset characteristics, as observed in the comparison with Metric3D and ZeroDepth. Despite being trained on a smaller dataset, UniDepth outperforms both of these methods. In particular, UniDepth improves in δ1\mathrm{\delta}_{1} over Metric3D and ZeroDepth by 5.8% and 7.3%, respectively, on NYU (Table 2) and by 1.1% and 9.4%, respectively, on KITTI (Table 3). Moreover, ZoeDepth, which has a capacity similar to our ViT-based approach and is pre-trained on the diverse MiDaS dataset [42], shows limitations in general zero-shot scenarios in Table 1, exhibiting performance comparable to traditional MMDE methods especially on scale-invariant metrics.

| Method | KITTI | NYU |  |  |  |  |
|---|---|---|---|---|---|---|
| δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow |  |
| iDisc [41] | 93.493.4 | 8.368.36 | 78.078.0 | 92.192.1 | 8.82¯\underline{8.82} | 75.075.0 |
| Metric3D [59] | 97.5¯\underline{97.5} | 7.28¯\underline{7.28} | 82.9¯\underline{82.9} | 92.6¯\underline{92.6} | 9.139.13 | 77.8¯\underline{77.8} |
| UniDepth | 97.9\mathbf{97.9} | 6.66\mathbf{6.66} | 83.8\mathbf{83.8} | 97.1\mathbf{97.1} | 6.69\mathbf{6.69} | 84.3\mathbf{84.3} |

For the sake of fair comparison, we provide in Table 4 a comparison between Metric3D, iDisc, and UniDepth where the latter two are retrained on a strict subset of Metric3D’s data, namely accounting for one-quarter of the original Metric3D dataset, with same framework detailed in Sec. 4.1. The results are two-fold: they demonstrate how UniDepth still surpasses Metric3D with a subsplit of the training set, and how MMDE SotA methods designed for single-domain can not fully exploit the training diversity. Qualitative results in Fig. 4 emphasize how the method excels in capturing the overall scale and scene complexity in a zero-shot setup.

### 4.3 Ablation Study

|  | Ablation | In-Domain | Out-of-Domain |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
|  | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | ARelC↓\mathrm{ARel}_{C}\downarrow | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | ARelC↓\mathrm{ARel}_{C}\downarrow |  |
| 1 | Oracle† | 89.06±0.0389.06\scriptstyle\pm 0.03 | 13.15±0.0213.15\scriptstyle\pm 0.02 | 65.45±0.1365.45\scriptstyle\pm 0.13 | n/a | 68.11±0.1768.11\scriptstyle\pm 0.17 | 14.78±0.0114.78\scriptstyle\pm 0.01 | 57.17±0.09{\color[rgb]{0.5,0.5,0.5}57.17\scriptstyle\pm 0.09} | n/a |
| 2 | Full | 88.89±0.1088.89\scriptstyle\pm 0.10 | 13.13±0.0113.13\scriptstyle\pm 0.01 | 63.52±0.0863.52\scriptstyle\pm 0.08 | 2.05±0.012.05\scriptstyle\pm 0.01 | 57.06±1.4857.06\scriptstyle\pm 1.48 | 14.83±0.0414.83\scriptstyle\pm 0.04 | 49.71±0.5549.71\scriptstyle\pm 0.55 | 13.54±0.8513.54\scriptstyle\pm 0.85 |
| 3 | – Camera† | 87.42±0.0487.42\scriptstyle\pm 0.04 | 13.49±0.0813.49\scriptstyle\pm 0.08 | 63.78±0.0263.78\scriptstyle\pm 0.02 | n/a | 48.38±0.9748.38\scriptstyle\pm 0.97 | 15.55±0.1515.55\scriptstyle\pm 0.15 | 45.21±0.8645.21\scriptstyle\pm 0.86 | n/a |
| 4 | – Spherical | 61.30±1.0061.30\scriptstyle\pm 1.00 | 19.36±0.0919.36\scriptstyle\pm 0.09 | 17.89±0.1117.89\scriptstyle\pm 0.11 | 48.29±4.0348.29\scriptstyle\pm 4.03 | 37.09±1.3737.09\scriptstyle\pm 1.37 | 22.49±0.1622.49\scriptstyle\pm 0.16 | 21.78±0.1421.78\scriptstyle\pm 0.14 | 87.51±11.187.51\scriptstyle\pm 11.1 |
| 5 | – ℒcon\mathcal{L}_{\mathrm{con}} | 88.53±0.0788.53\scriptstyle\pm 0.07 | 13.24±0.0113.24\scriptstyle\pm 0.01 | 60.89±0.1560.89\scriptstyle\pm 0.15 | 2.65±0.062.65\scriptstyle\pm 0.06 | 52.89±0.2152.89\scriptstyle\pm 0.21 | 14.85±0.0114.85\scriptstyle\pm 0.01 | 45.17±0.3245.17\scriptstyle\pm 0.32 | 14.27±0.4114.27\scriptstyle\pm 0.41 |
| 6 | – Dense | 87.62±0.1187.62\scriptstyle\pm 0.11 | 13.41±0.0513.41\scriptstyle\pm 0.05 | 61.33±0.5461.33\scriptstyle\pm 0.54 | 1.91±0.041.91\scriptstyle\pm 0.04 | 55.65±0.1855.65\scriptstyle\pm 0.18 | 15.04±0.0415.04\scriptstyle\pm 0.04 | 43.19±0.2443.19\scriptstyle\pm 0.24 | 16.61±0.4116.61\scriptstyle\pm 0.41 |
| 7 | – Detach | 88.16±0.1288.16\scriptstyle\pm 0.12 | 13.48±0.0613.48\scriptstyle\pm 0.06 | 64.19±0.1764.19\scriptstyle\pm 0.17 | 0.93±0.020.93\scriptstyle\pm 0.02 | 46.60±0.2546.60\scriptstyle\pm 0.25 | 15.26±0.1015.26\scriptstyle\pm 0.10 | 43.85±2.0143.85\scriptstyle\pm 2.01 | 18.99±1.0018.99\scriptstyle\pm 1.00 |
| 8 | Baseline | 77.36±0.2277.36\scriptstyle\pm 0.22 | 21.17±0.2821.17\scriptstyle\pm 0.28 | 16.29±0.2616.29\scriptstyle\pm 0.26 | n/a | 48.19±1.0248.19\scriptstyle\pm 1.02 | 23.05±0.4523.05\scriptstyle\pm 0.45 | 14.29±0.3614.29\scriptstyle\pm 0.36 | n/a |
| 9 | Baseline++ | 82.41±0.1382.41\scriptstyle\pm 0.13 | 16.31±0.0516.31\scriptstyle\pm 0.05 | 41.98±0.1241.98\scriptstyle\pm 0.12 | n/a | 51.22±0.3551.22\scriptstyle\pm 0.35 | 18.14±0.0518.14\scriptstyle\pm 0.05 | 38.27±0.0238.27\scriptstyle\pm 0.02 | n/a |

The importance of each component introduced in UniDepth in Sec. 3 is evaluated by ablating the method in Table 5. All ablations exploit the predicted camera representation, if not stated otherwise. The first distinction involves the Oracle model, which operates under ideal conditions with known camera information during training and testing, addressing a task similar to [21, 59]. On the other hand, Baseline is a straightforward encoder-decoder implementation with a (xx,yy,zz) output, as outlined at the beginning of Sec. 3, while Baseline++ exploits the proposed pseudo-spherical representation. Modules’ architectures are consistent across experiments. The In-Domain column reflects testing on validation splits of training domains, while Out-of-Domain corresponds to zero-shot testing, as detailed in Sec. 4.1. Notably, In-Domain results exhibit a higher degree of homogeneity compared to Out-of-Domain, which is noisier yet more informative for gauging expected performances in downstream applications and in-the-wild deployment.

Architecture. The Oracle model demonstrates more robust scale-dependent performance during zero-shot testing compared to the Full model, highlighting how the proposed task is inherently more demanding. The Baseline model illustrates an approach to the problem without utilizing external information and lacking a proper design for both internal and output space. This approach yields markedly inferior results for both In-Domain and Out-of-Domain scenarios in terms of depth and 3D reconstruction metrics.

Camera Module. In Table 5, row 3, the benefit of the Camera Module becomes apparent, revealing a substantial disparity in the effect of this module on scale-invariant and scale-dependent metrics for in- and out-of-domain testing. This disparity stems from the absence of prior knowledge of the model regarding scale, impeding its optimal utilization of the diverse training set. Concentrating solely on predicting depth, rather than a complete 3D output, proves advantageous in averting convergence issues during training. This is evident in comparison with methods predicting 3D, either without reliance on camera information (rows 8 and 9) or influenced by intertwined optimization (row 4), as elucidated in Sec. 3. Refraining from relying on the camera also constrains the model’s capacity to recover a multimodal distribution for out-of-domain samples. The lack of a (bootstrapped) prior prevents the depth module from serving as a corrective mechanism based on an initial scale estimation and imposes an unnecessary computational burden, i.e. recovering the depth values from scratch. This limitation is underscored by the marked variability observed for test sets strongly out-of-distribution, such as KITTI, when comparing the utilization or absence of camera information (rows 2 and 3, respectively). In particular, Full achieves 95.2% in δ1\mathrm{\delta}_{1} in KITTI, while “– Camera” obtains 58.9% for the same test set, despite a mere 2% difference between the two versions on nuScenes and DDAD.

Optimization and Output Representation. All ablations employ the same loss ℒλ​M​S​E\mathcal{L}_{\lambda MSE}, but across different output spaces. In row 4, a Cartesian output space is used instead of a pseudo-spherical from Sec. 3.1, which results in substantially inferior performance due to the respective intertwined formulation of camera and depth output spaces. The Baseline (row 8) also employs a Cartesian representation, but the negative impact of this choice is less pronounced in this model because of the absence of a camera module. More specifically, the decoder of Baseline is not conditioned on inaccurate prior camera and scale information as in row 4. Moreover, row 9 corresponds to Baseline with pseudo-spherical representation. Comparison between row 8 and row 9 shows that when predicting directly the 3D outputs, the choice of the output representation is still relevant in defining a better internal representation and optimization. Row 5 demonstrates the positive impact of the geometric invariance loss. This loss contributes to enhanced in-domain and out-of-domain performance by promoting the invariance of depth features to appearance variations owing to different camera intrinsics. Furthermore, stopping the gradient from propagating from the Camera Module to the Encoder (row 7), as described in Sec. 3.4, proves particularly beneficial in avoiding scale and camera overfitting in zero-shot testing, and stabilizes the training. The more stable training is obtained by limiting the dominant effect that camera supervision has on the gradient of the Encoder weights compared to depth supervision.

Camera Representation. In row 6, the model incorporates a sparse camera representation, specifically the pinhole camera model with (fxf_{x}, fyf_{y}, cxc_{x}, cyc_{y}), leading to sparse camera prompting and scalar supervision; the camera module still predicts the residual components as outlined in Sec. 3.2. This approach hurts generalization, as evidenced by ARelC\mathrm{ARel}_{C} in the out-of-domain evaluation, despite the slight improvement in in-domain ARelC\mathrm{ARel}_{C}. We speculate that the four prompts convey less robust information to the depth module than their dense counterpart, resulting in inferior performance for depth metrics compared to Full for both in- and out-of-domain.

## 5 Conclusion

In this work, we propose UniDepth to predict metric 3D points in diverse scenes relying solely on a single input image. Through meticulous ablation studies, we systematically address the challenges inherent in universal MMDE tasks, underscoring the pivotal contributions of our work. The designed self-prompting camera allows camera-free test time application and renders the model more robust against camera noise. The introduced pseudo-spherical output space representation adequately disentangles the camera and depth of the optimization process. Furthermore, the proposed geometric invariance loss effectively ensures camera-aware depth consistency. Extensive validations unequivocally exhibit how UniDepth sets the new state of the art across multiple benchmarks in a zero-shot regime, even surpassing in-domain trained methods. This attests to the robustness and efficacy of our model and, most importantly, outlines its potential to propel the field of MMDE to new frontiers.

Acknowledgment. This work is funded by Toyota Motor Europe via the research project TRACE-Zürich.

Supplementary Material

This supplementary material offers further insights into our work. In Sec. A we provide results on the official KITTI benchmark, and standard metric evaluation on KITTI and NYU validation set. Moreover, Sec. B includes additional ablations, namely with VIT backbone and a comparison of different pseudo-spherical representations. In addition, differences between convolutional and ViT-based backbones regarding generalization are discussed. In Sec. C, we describe the datasets used for training and testing and how we propose to amend Diode [50] artifacts at boundaries present in ground-truth depth. We analyze the complexity of UniDepth and compare it with other methods in Sec. D. Furthermore, we describe in Sec. E the network architecture in more detail, necessarily Sec. E overlaps with Sec. 3. Eventually, additional visualizations are provided in Sec. F.

## A Results

KITTI benchmark [18]. Table 6 clearly shows the compelling performance of UniDepth on the official KITTI private test set. Results of the latest published methods are reported. The table is fetched from the official KITTI leaderboard for depth prediction. In particular, UniDepth ranks first in the KITTI benchmark at the time of submission among all methods, published and not.

| Method | SIlog\mathrm{SI_{\log}} | Sq.Rel\mathrm{Sq.Rel} | A.Rel\mathrm{A.Rel} | iRMS\mathrm{iRMS} |
|---|---|---|---|---|
| Lower is better |  |  |  |  |
| MG [29] | 9.93 | 1.68 % | 7.99 % | 10.63 |
| URCDC-Depth [45] | 10.03 | 1.74 % | 8.24 % | 10.71 |
| iDisc [41] | 9.89 | 1.77 % | 8.11 % | 10.73 |
| VA-DepthNet [30] | 9.84 | 1.66 % | 7.96 % | 10.44 |
| IEBins [47] | 9.63 | 1.60 % | 7.82 % | 10.68 |
| NDDepth [46] | 9.62 | 1.59 % | 7.75 % | 10.62 |
| UniDepth | 8.13 | 1.09% | 6.54 % | 8.24 |

KITTI Eigen-split and NYUv2-Depth. For the sake of completeness, we report the “standard” metrics results in Table 7 and Table 8 on KITTI Eigen-split and NYU validation set, respectively. It is worth noting that the typical metrics δ2\delta_{2} and, especially, δ3\delta_{3} are saturated, thus not informative. Therefore, we advocate our choice of not reporting them in the main paper and prefer to report δ0.5\delta_{0.5}. Moreover, we suggest in future works the use of the area under the curve of the δ\delta metrics as a more informative and comprehensive metric, instead of the values at fixed thresholds, i.e. {1.25i}i=13\{1.25^{i}\}^{3}_{i=1}.

| Method | δ1\mathrm{\delta}_{1} | δ2\mathrm{\delta}_{2} | δ3\mathrm{\delta}_{3} | FA\mathrm{F}_{A} | A.Rel\mathrm{A.Rel} | RMS\mathrm{RMS} | RMSlog\mathrm{RMS}_{\log} | CD\mathrm{CD} | SIlog\mathrm{SI}_{\log} |
|---|---|---|---|---|---|---|---|---|---|
| Higher is better | Lower is better |  |  |  |  |  |  |  |  |
| BTS [28] | 96.296.2 | 99.499.4 | 99.8¯\underline{99.8} | 82.082.0 | 5.635.63 | 2.432.43 | 0.0890.089 | 0.420.42 | 8.188.18 |
| AdaBins [3] | 96.396.3 | 99.599.5 | 99.8¯\underline{99.8} | 81.581.5 | 5.855.85 | 2.382.38 | 0.0890.089 | 0.4290.429 | 8.108.10 |
| NeWCRF [61] | 97.597.5 | 99.7¯\underline{99.7} | 99.9\mathbf{99.9} | 82.782.7 | 5.205.20 | 2.072.07 | 0.0780.078 | 0.3880.388 | 7.007.00 |
| iDisc [41] | 97.597.5 | 99.7¯\underline{99.7} | 99.9\mathbf{99.9} | 83.183.1 | 5.095.09 | 2.072.07 | 0.0770.077 | 0.3800.380 | 7.117.11 |
| ZoeDepth [4] | 96.596.5 | 99.199.1 | 99.499.4 | 82.182.1 | 5.765.76 | 2.392.39 | 0.0890.089 | 0.4310.431 | 7.477.47 |
| Metric3D [59] | 97.597.5 | 99.599.5 | 99.8¯\underline{99.8} | 82.982.9 | 5.335.33 | 2.262.26 | 0.0810.081 | 0.3920.392 | 7.287.28 |
| Ours-C | 97.8¯\underline{97.8} | 99.7¯\underline{99.7} | 99.9\mathbf{99.9} | 83.9¯\underline{83.9} | 4.69¯\underline{4.69} | 2.00¯\underline{2.00} | 0.073¯\underline{0.073} | 0.371¯\underline{0.371} | 6.72¯\underline{6.72} |
| Ours-V | 98.6\mathbf{98.6} | 99.8\mathbf{99.8} | 99.9\mathbf{99.9} | 85.0\mathbf{85.0} | 4.21\mathbf{4.21} | 1.75\mathbf{1.75} | 0.064\mathbf{0.064} | 0.338\mathbf{0.338} | 5.84\mathbf{5.84} |
| Ours-C ‡ | 97.8\mathbf{97.8} | 99.7¯\underline{99.7} | 99.9\mathbf{99.9} | 80.880.8 | 4.774.77 | 2.002.00 | 0.0730.073 | 0.4270.427 | 6.726.72 |
| Ours-V ‡ | 98.6\mathbf{98.6} | 99.8\mathbf{99.8} | 99.9\mathbf{99.9} | 82.782.7 | 4.21\mathbf{4.21} | 1.75\mathbf{1.75} | 0.064\mathbf{0.064} | 0.3810.381 | 5.84\mathbf{5.84} |

| Method | δ1\mathrm{\delta}_{1} | δ2\mathrm{\delta}_{2} | δ3\mathrm{\delta}_{3} | FA\mathrm{F}_{A} | A.Rel\mathrm{A.Rel} | RMS\mathrm{RMS} | Log10\mathrm{Log}_{10} | CD\mathrm{CD} | SIlog\mathrm{SI}_{\log} |
|---|---|---|---|---|---|---|---|---|---|
| Higher is better | Lower is better |  |  |  |  |  |  |  |  |
| BTS [28] | 88.588.5 | 97.8\mathbf{97.8} | 99.499.4 | 74.074.0 | 10.910.9 | 0.3910.391 | 0.0460.046 | 0.1600.160 | 11.511.5 |
| AdaBins [3] | 90.190.1 | 98.398.3 | 99.699.6 | 74.774.7 | 10.310.3 | 0.3650.365 | 0.0440.044 | 0.1560.156 | 10.610.6 |
| NeWCRF [61] | 92.192.1 | 99.199.1 | 99.8¯\underline{99.8} | 75.875.8 | 9.569.56 | 0.3330.333 | 0.0400.040 | 0.1470.147 | 9.169.16 |
| iDisc [41] | 93.893.8 | 99.299.2 | 99.8¯\underline{99.8} | 78.278.2 | 8.618.61 | 0.3130.313 | 0.0370.037 | 0.1330.133 | 8.858.85 |
| ZoeDepth [4] | 95.295.2 | 99.599.5 | 99.8¯\underline{99.8} | 80.180.1 | 7.707.70 | 0.2780.278 | 0.0330.033 | 0.1250.125 | 7.197.19 |
| Metric3D [59] | 92.692.6 | 97.997.9 | 99.199.1 | 77.877.8 | 9.389.38 | 0.3370.337 | 0.0380.038 | 0.1460.146 | 9.139.13 |
| Ours-C | 97.297.2 | 99.6¯\underline{99.6} | 99.9\mathbf{99.9} | 84.484.4 | 6.226.22 | 0.2310.231 | 0.0260.026 | 0.1010.101 | 6.396.39 |
| Ours-V | 98.4\mathbf{98.4} | 99.7¯\underline{99.7} | 99.9\mathbf{99.9} | 85.9\mathbf{85.9} | 5.78\mathbf{5.78} | 0.201\mathbf{0.201} | 0.024\mathbf{0.024} | 0.092\mathbf{0.092} | 5.27\mathbf{5.27} |
| Ours-C‡ | 97.297.2 | 99.6¯\underline{99.6} | 99.9\mathbf{99.9} | 84.184.1 | 6.336.33 | 0.2320.232 | 0.0270.027 | 0.1030.103 | 6.406.40 |
| Ours-V‡ | 98.3¯\underline{98.3} | 99.7¯\underline{99.7} | 99.9\mathbf{99.9} | 85.5¯\underline{85.5} | 6.04¯\underline{6.04} | 0.205¯\underline{0.205} | 0.025¯\underline{0.025} | 0.094¯\underline{0.094} | 5.28¯\underline{5.28} |

## B Ablations

|  | Ablation | In-Domain | Out-of-Domain |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
|  | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | ARelC↓\mathrm{ARel}_{C}\downarrow | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | ARelC↓\mathrm{ARel}_{C}\downarrow |  |
| 1 | Oracle | 91.46±0.0991.46\scriptstyle\pm 0.09 | 12.12±0.0212.12\scriptstyle\pm 0.02 | 68.35±0.1468.35\scriptstyle\pm 0.14 | n/a | 72.17±0.4472.17\scriptstyle\pm 0.44 | 13.07±0.0113.07\scriptstyle\pm 0.01 | 59.84±0.1859.84\scriptstyle\pm 0.18 | n/a |
| 2 | Full | 91.43±0.0591.43\scriptstyle\pm 0.05 | 12.06±0.0612.06\scriptstyle\pm 0.06 | 65.44±0.8465.44\scriptstyle\pm 0.84 | 2.19±0.142.19\scriptstyle\pm 0.14 | 64.45±0.5264.45\scriptstyle\pm 0.52 | 13.0±0.0213.0\scriptstyle\pm 0.02 | 52.46±0.2952.46\scriptstyle\pm 0.29 | 12.31±0.6112.31\scriptstyle\pm 0.61 |
| 3 | – Camera | 89.33±0.0489.33\scriptstyle\pm 0.04 | 12.54±0.0412.54\scriptstyle\pm 0.04 | 66.02±0.2766.02\scriptstyle\pm 0.27 | n/a | 60.67±0.2260.67\scriptstyle\pm 0.22 | 13.4±0.0713.4\scriptstyle\pm 0.07 | 52.43±0.0852.43\scriptstyle\pm 0.08 | n/a |
| 4 | – ℒcon\mathcal{L}_{\mathrm{con}} | 90.27±0.1390.27\scriptstyle\pm 0.13 | 12.21±0.0112.21\scriptstyle\pm 0.01 | 63.28±0.6663.28\scriptstyle\pm 0.66 | 1.92±0.311.92\scriptstyle\pm 0.31 | 61.98±0.4161.98\scriptstyle\pm 0.41 | 13.24±0.0413.24\scriptstyle\pm 0.04 | 50.91±0.1650.91\scriptstyle\pm 0.16 | 13.11±0.3613.11\scriptstyle\pm 0.36 |
| 5 | – Spherical | 32.92±0.1832.92\scriptstyle\pm 0.18 | 18.11±0.0818.11\scriptstyle\pm 0.08 | 33.62±0.0733.62\scriptstyle\pm 0.07 | 21.64±0.221.64\scriptstyle\pm 0.2 | 48.43±1.2748.43\scriptstyle\pm 1.27 | 18.53±0.3518.53\scriptstyle\pm 0.35 | 42.85±1.1842.85\scriptstyle\pm 1.18 | 17.16±0.7917.16\scriptstyle\pm 0.79 |
| 6 | – Dense | 90.16±0.1590.16\scriptstyle\pm 0.15 | 12.23±0.0112.23\scriptstyle\pm 0.01 | 64.19±0.0364.19\scriptstyle\pm 0.03 | 1.83±0.181.83\scriptstyle\pm 0.18 | 62.44±0.1962.44\scriptstyle\pm 0.19 | 13.36±0.0413.36\scriptstyle\pm 0.04 | 49.34±0.2849.34\scriptstyle\pm 0.28 | 13.68±0.6113.68\scriptstyle\pm 0.61 |
| 7 | – Detach | 89.93±0.0289.93\scriptstyle\pm 0.02 | 12.58±0.0412.58\scriptstyle\pm 0.04 | 66.30±0.3566.30\scriptstyle\pm 0.35 | 0.94±0.030.94\scriptstyle\pm 0.03 | 51.77±0.0951.77\scriptstyle\pm 0.09 | 13.45±0.0213.45\scriptstyle\pm 0.02 | 49.91±0.0149.91\scriptstyle\pm 0.01 | 14.87±0.2214.87\scriptstyle\pm 0.22 |
| 8 | Baseline | 21.26±0.2321.26\scriptstyle\pm 0.23 | 23.43±0.4523.43\scriptstyle\pm 0.45 | 29.19±0.0929.19\scriptstyle\pm 0.09 | n/a | 34.15±0.7434.15\scriptstyle\pm 0.74 | 20.39±0.4220.39\scriptstyle\pm 0.42 | 40.14±0.5240.14\scriptstyle\pm 0.52 | n/a |
| 9 | Baseline++ | 88.84±0.1188.84\scriptstyle\pm 0.11 | 12.93±0.1112.93\scriptstyle\pm 0.11 | 42.72±0.1042.72\scriptstyle\pm 0.10 | n/a | 59.31±0.5859.31\scriptstyle\pm 0.58 | 14.04±0.0314.04\scriptstyle\pm 0.03 | 44.12±0.1044.12\scriptstyle\pm 0.10 | n/a |

### B.1 Ablations with ViT backbone

Ablations with ViT backbone are provided in Table 9. The trend in Table 9 is consistent with the one outlined for the convolutional backbone. More specifically, the ablated components contribute similarly between ViT-L [12] and ConvNext-L [33] backbones. However, utilizing a ViT backbone shows a larger variability for out-of-domain results, also showing a stronger effect of the usage of pseudo-spherical representation both for the Baseline and Full. The increased susceptibility of the scene’s depth scale to domain shift is also related to the backbone comparison in Table 1. In particular, zero-shot results suggest that the convolutional architecture exhibits superior resilience to scale-related domain shifts, although showing relative disadvantage in handling appearance-related domain shifts. SIlog\mathrm{SI}_{\log} consistently favors ViT over convolutional methods, emphasizing the latter’s diminished performance in appearance domain shifts. However, scale-dependent metrics do not consistently favor ViT, indicating that the constrained receptive field of convolutional methods yields higher robustness to domain shifts associated with scale.

### B.2 Alternative pseudo-spherical representation

| Ablation | Backbone | In-Domain | Out-of-Domain |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
| δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | ARelC↓\mathrm{ARel}_{C}\downarrow | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | ARelC↓\mathrm{ARel}_{C}\downarrow |  |  |
| UniDepth | ViT-L [12] | 91.43±0.0591.43\scriptstyle\pm 0.05 | 12.06±0.0612.06\scriptstyle\pm 0.06 | 65.44±0.8465.44\scriptstyle\pm 0.84 | 2.19±0.142.19\scriptstyle\pm 0.14 | 64.45±0.5264.45\scriptstyle\pm 0.52 | 13.00±0.0213.00\scriptstyle\pm 0.02 | 52.46±0.2952.46\scriptstyle\pm 0.29 | 12.31±0.6112.31\scriptstyle\pm 0.61 |
| UniDepth rays | ViT-L [12] | 90.93±0.0290.93\scriptstyle\pm 0.02 | 12.19±0.0612.19\scriptstyle\pm 0.06 | 64.70±0.0564.70\scriptstyle\pm 0.05 | 2.44±0.112.44\scriptstyle\pm 0.11 | 65.50±0.8165.50\scriptstyle\pm 0.81 | 13.03±0.0113.03\scriptstyle\pm 0.01 | 53.12±0.0253.12\scriptstyle\pm 0.02 | 11.82±0.9911.82\scriptstyle\pm 0.99 |
| UniDepth | ConvNext-L [33] | 88.89±0.1088.89\scriptstyle\pm 0.10 | 13.13±0.0113.13\scriptstyle\pm 0.01 | 63.52±0.0863.52\scriptstyle\pm 0.08 | 2.05±0.012.05\scriptstyle\pm 0.01 | 57.06±1.4857.06\scriptstyle\pm 1.48 | 14.83±0.0414.83\scriptstyle\pm 0.04 | 49.71±0.5549.71\scriptstyle\pm 0.55 | 13.54±0.8513.54\scriptstyle\pm 0.85 |
| UniDepth rays | ConvNext-L [33] | 88.55±0.3188.55\scriptstyle\pm 0.31 | 13.24±0.1013.24\scriptstyle\pm 0.10 | 62.58±1.1162.58\scriptstyle\pm 1.11 | 2.74±0.132.74\scriptstyle\pm 0.13 | 55.10±0.3955.10\scriptstyle\pm 0.39 | 14.91±0.0114.91\scriptstyle\pm 0.01 | 46.38±0.6146.38\scriptstyle\pm 0.61 | 15.00±0.3615.00\scriptstyle\pm 0.36 |

Sec. 3 focuses on describing the pseudo-spherical representation chosen to disentangle the two sub-tasks, namely calibration and depth estimation, and ablations studies confirm the effectiveness of disentangling the sub-tasks. In particular, UniDepth exploits an angular pseudo-spherical representation, namely based on azimuth, elevation angle, and log-depth, i.e. (θ\theta, ϕ\phi, zlogz_{\log}). Nevertheless, an alternative solution to disentangle the two different sub-tasks, namely calibration and depth estimation, is to exploit the bearing vector and log-depth. More specifically, a bearing vector corresponds to the unit-length ray represented by (rxr_{x}, ryr_{y}, rzr_{z}) ∈𝕊2\in\mathbb{S}^{2}, with 𝕊2\mathbb{S}^{2} corresponding to the unit-sphere manifold. The bearing vectors are obtained as the unprojection of image coordinates based on the (pinhole) camera model. With this design, the output is represented by the tuple (rxr_{x}, ryr_{y}, rzr_{z}, zlogz_{\log}) and the loss ℒλ​MSE\mathcal{L}_{\mathrm{\lambda MSE}} is applied seamlessly as depicted in Sec. 3, but with λrx=λry=λrz=1\lambda_{r_{x}}=\lambda_{r_{y}}=\lambda_{r_{z}}=1 and λz=0.15\lambda_{z}=0.15.

However, the disentanglement in rays and log-depth can be viewed as an alternative pseudo-spherical representation, in fact, rays and angles share a direct relationship θ=arctan⁡(rxrz)\theta=\mathrm{arctan}(\frac{r_{x}}{r_{z}}) and ϕ=arccos⁡(ry)\phi=\mathrm{arccos}(r_{y}). Table 10 explores the effectiveness of this alternative representation and compares to the one presented in Sec. 3. The ablation study reported in Table 10 highlights how the difference between the two representations is marginal and, in most cases, within the uncertainty range, thus proving their similarity. The main difference lies in the output space dimensionality. In principle, the bearing vectors would span the entire ℝ3\mathbb{R}^{3} space. However, the space is constrained to the unit-sphere manifold by L2\mathrm{L}_{2} normalization.

Furthermore, we ablate our camera prompting with respect to CAMConvs [15] in Table 11

| Ablation | In-Domain | Out-of-Domain |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
| δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | ARelC↓\mathrm{ARel}_{C}\downarrow | δ1↑\mathrm{\delta}_{1}\uparrow | SIlog↓\mathrm{SI}_{\log}\downarrow | FA↑\mathrm{F}_{A}\uparrow | ARelC↓\mathrm{ARel}_{C}\downarrow |  |
| w/ CAMConvs | 87.8187.81 | 13.4913.49 | 60.9060.90 | 2.552.55 | 54.6554.65 | 15.3715.37 | 43.0943.09 | 16.1116.11 |
| Full | 88.89\mathbf{88.89} | 13.13\mathbf{13.13} | 63.52\mathbf{63.52} | 2.05\mathbf{2.05} | 57.06\mathbf{57.06} | 14.83\mathbf{14.83} | 49.71\mathbf{49.71} | 13.54\mathbf{13.54} |

|  | Dataset | Images | Scene | Acquisition |
|---|---|---|---|---|
| Training Set | A2D2 [19] | 78k | Outdoor | LiDAR |
| Argoverse2 [53] | 403k | Outdoor | LiDAR |  |
| BDD100k [60] | 270k | Outdoor | SfM |  |
| CityScapes [7] | 24k | Outdoor | MVS |  |
| DrivingStereo [56] | 63k | Outdoor | MVS |  |
| Mapillary PSD [1] | 742k | Outdoor | SfM |  |
| ScanNet [8] | 83k | Indoor | RGB-D |  |
| Taskonomy [62] | 1940k | Indoor | RGB-D |  |
| Waymo [49] | 223k | Outdoor | LiDAR |  |
| Testing Set | DDAD [20] | 1002 | Outdoor | LiDAR |
| Diode [50] | 325 | Indoor | LiDAR |  |
| ETH3D [44] | 454 | Outdoor | RGB-D |  |
| HAMMER [25] | 496 | Indoor | Mix |  |
| IBims-1 [26] | 100 | Indoor | RGB-D |  |
| KITTI [18] | 652 | Outdoor | LiDAR |  |
| NuScenes [5] | 3k | Outdoor | LiDAR |  |
| NYU [35] | 654 | Indoor | RGB-D |  |
| SUN-RGBD [48] | 4.4k | Indoor | RGB-D |  |
|  | VOID [54] | 800 | Indoor | RGB-D |

## C Datasets

### C.1 Datasets details

Details of training and testing datasets are presented in Table 12. The training datasets are processed in a way that the interval between two consecutive RGB and GT depth frames is not smaller than one second. We do not apply any post-processing apart from the aforementioned subsampling. The total amount of training samples accounts for 3’743’000 samples. SUN-RGBD [48] validation set involves also NYU [35] test set. Therefore, we removed the samples corresponding to NYU test set to avoid any overlap between test sets. As per standard practice, KITTI Eigen-split corresponds to the corrected and accumulated GT depth maps with 45 images with inaccurate GT discarded from the original 697 images.

### C.2 Diode Indoor ground-truth correction

Diode [50] ground-truth depth is not perfectly accurate on boundaries, in particular, a simple inspection shows how depth in boundaries presents low values, but greater than zero. These artifacts present in the GT affect the validation pipeline and results. Therefore, we design a simple image processing algorithm, outlined in Algorithm 1, that, first, detects the aforementioned boundary artifacts and, second, masks the depth in the corresponding neighborhoods. Thanks to masking those boundaries, the corresponding regions are ignored during validation.

## D Model Complexity

Table 13 displays the parameters and inference complexity of UniDepth and other SotA methods. UniDepth with ViT-L backbone is comparable to ZoeDepth in terms of efficiency and model parameters; however UniDepth surpasses it in terms of performance as stated in Sec. 4. Metric3D displays an improved efficiency due to the fully convolutional and relatively low dimensionality designed in the decoder. It is worth highlighting how ZeroDepth presents a low efficiency although based on ResNet-18, we argue that this is due to the expensive full-resolution cross-attention in the decoder. The last two rows in Table 13 analyze separately the complexity of the single Camera and Depth Module. The Camera Module is a lightweight component accounting for 13.4M parameters. On the other hand, the Depth Module amounts to more than half of the total latency, despite the limited memory consumption. The Depth Module’s high latency is due to the several (6) self-attention layers in the decoder.

| Method | Backbone | Latency (ms) | Throughput (FPS) | Parameters (M) |
|---|---|---|---|---|
| BTS [28] | D161 | 28.5 | 35.1 | 47.0 |
| Adains [3] | EN-B5 | 33.2 | 30.1 | 78.3 |
| NewCRF [61] | SWin-L [32] | 53.1 | 18.8 | 280.0 |
| iDisc [41] | SWin-L [32] | 81.1 | 12.3 | 209.2 |
| ZoeDepth [4] | BEiT-L | 144.8 | 6.91 | 345.9 |
| ZeroDepth [21] | R18 | 955.6 | 1.05 | 232.6 |
| Metric3D [59] | CNXT-L | 40.3 | 24.8 | 203.2 |
| UniDepth | CNXT-L | 86.6 | 11.5 | 238.9 |
| UniDepth | ViT-L [12] | 146.4 | 6.83 | 347.0 |
| Camera Module | - | 5.1 | - | 13.4 |
| Depth Module | - | 49.2 | - | 26.6 |

## E Network Architecture

Encoder. We show the effectiveness of our method with different encoders, both convolutional and transformer-based ones, e.g., ConvNext [33] and ViT [12]. However, all of them follow the same structure: the feature maps are extracted at each layer and the features map corresponding to a “scale” is obtained as the pixel-wise average. For ConvNext, we obtain the class tokens as the average pooled feature maps. All backbones utilized are originally designed for classification, thus we remove the last 3 layers, i.e., the pooling layer, fully connected layer, and softmax\mathrm{softmax} layer. The feature maps are flattened, then LayerNorm [2] (LN) and a linear layer are applied. The linear layer projects the features to a common channel dimension of 512. The projected feature maps are interpolated to a common shape, namely (h,w)=(H16,W16)(h,w)=(\frac{H}{16},\frac{W}{16}), with HH, WW as input height and width, respectively. Two independent projections are utilized for the features maps, i.e. 𝐅∈ℝh×w×C×B\mathbf{F}\in\mathbb{R}^{h\times w\times C\times B} with BB corresponding to the four scales, and CC set to 512 as mentioned above, and the class tokens ∈ℝC×B\in\mathbb{R}^{C\times B}, the latter fed to the Camera Module only.

Camera Module. The camera parameters are initialized with the four class tokens extracted from the Encoder. The flattened and stacked feature maps from the encoder are detached and used as keys and values in one cross-attention layer, where the queries correspond to the four camera parameters. The output is processed by a MultiLayer Perceptron (MLP) with one hidden layer with dimension of 2048 and non-linear activation Gaussian Error Linear Unit (GELU) [23]. The cross-attention and the MLP present a residual connection. The four tokens are further processed with two additional self-attention layers, projected to dimension one and then exponentiated. The camera parameters are obtained as fx=Δ​fx​W2f_{x}=\frac{\Delta f_{x}W}{2}, fy=Δ​fy​H2f_{y}=\frac{\Delta f_{y}H}{2}, cx=Δ​cx​W2c_{x}=\frac{\Delta c_{x}W}{2}, cy=Δ​cy​H2c_{y}=\frac{\Delta c_{y}H}{2}. The dense camera representation 𝐂\mathbf{C} is obtained by backprojecting with the predicted camera parameters: (𝐫x,𝐫y,𝐫z)=𝐊−1​[𝐮,𝐯,𝟏]T(\mathbf{r}_{x},\mathbf{r}_{y},\mathbf{r}_{z})=\mathbf{K}^{-1}[\mathbf{u},\mathbf{v},\mathbf{1}]^{T} and calculating the azimuth and elevation angles, θ\theta and ϕ\phi, as in Sec. B. The angular representation is embedded through the Laplace Spherical Harmonics Embedding (SHE) leading to 81 channels, resulting in 𝐄∈ℝh×w×81\mathbf{E}\in\mathbb{R}^{h\times w\times 81}.

Depth Module. The depth latents are initialized as the average of the features 𝐅\mathbf{F} along the BB dimension. Then, the latents are conditioned on the original feature tensor 𝐅\mathbf{F} via one cross-attention layer where two projections of 𝐅\mathbf{F} account for keys and values and 𝐋\mathbf{L} as queries. In addition, one MLP is applied, seamlessly as in the Camera Module. Furthermore, the depth features are conditioned on the camera prompts 𝐄\mathbf{E} with one additional cross-attention layer, where keys and values are two projections of camera embeddings 𝐄\mathbf{E}, and one MLP as above. The features are decoded in three consecutive stages. The first stage applies three self-attention layers with 𝐄\mathbf{E} as positional encoding. The features are then processed with one ConvNext [33] layer, upsampled by a factor of two, and the channels are halved. The second and third stages are similar, although the second stage presents two self-attention layers and the third only one. In the second and third stages, MLP’s hidden channel dimension is sequentially halved, too, from the initial aforementioned value of 2048. Each stage’s output is projected to a dimension one. Therefore, the three output maps are interpolated to a common shape, i.e. (H2\frac{H}{2}, W2\frac{W}{2}), and pixel-wise averaged. The final log-depth output 𝐙log\mathbf{Z}_{\log} is obtained by upsampling the obtained tensor to the input shape (HH, WW). The final depth is element-wise exponentiation of 𝐙log\mathbf{Z}_{\log}.

## F Visualization

We provide here twenty more qualitative comparisons, two for each zero-shot test set: KITTI, NYU, Diode, ETH3D in Fig. 5, DDAD, NuScenes, SUN-RGBD, IBims-1 in Fig. 6, and Fig. 7 displays VOID and HAMMER. The error maps are shown after applying median-based rescaling. The rescaling was deemed necessary to avoid some of the error maps being completely red and not informative. Due to sparsity, DDAD and Nuscenes GT and error maps are dilated by a factor of 5, leading to visible GT depth and error maps.

| KITTI |  |  |  |  |  |  |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |
| NYU |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
| Diode |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
| ETH3D |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  | RGB & GT | ZoeDepth† [4] | ZeroDepth [21] | Metric3D [59] | UniDepth | Meters || A.Rel\mathrm{A.Rel} |

| DDAD |  |  |  |  |  |  |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |
| NuScenes |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
| SUN-RGBD |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
| IBims-1 |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  | RGB & GT | ZoeDepth [4] | ZeroDepth [21] | Metric3D† [59] | UniDepth | Meters || A.Rel\mathrm{A.Rel} |

| VOID |  |  |  |  |  |  |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |
| HAMMER |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  | RGB & GT | ZoeDepth [4] | ZeroDepth [21] | Metric3D [59] | UniDepth | Meters || A.Rel\mathrm{A.Rel} |
