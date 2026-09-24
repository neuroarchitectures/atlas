# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

We present a method that achieves state-of-the-art results for synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views. Our algorithm represents a scene using a fully-connected (non-convolutional) deep network, whose input is a single continuous 5D coordinate (spatial location (x,y,z)(x,y,z) and viewing direction (θ,ϕ)(\theta,\phi)) and whose output is the volume density and view-dependent emitted radiance at that spatial location. We synthesize views by querying 5D coordinates along camera rays and use classic volume rendering techniques to project the output colors and densities into an image. Because volume rendering is naturally differentiable, the only input required to optimize our representation is a set of images with known camera poses. We describe how to effectively optimize neural radiance fields to render photorealistic novel views of scenes with complicated geometry and appearance, and demonstrate results that outperform prior work on neural rendering and view synthesis. View synthesis results are best viewed as videos, so we urge readers to view our supplementary video for convincing comparisons.

## 1 Introduction

In this work, we address the long-standing problem of view synthesis in a new way by directly optimizing parameters of a continuous 5D scene representation to minimize the error of rendering a set of captured images.

We represent a static scene as a continuous 5D function that outputs the radiance emitted in each direction (θ,ϕ)(\theta,\phi) at each point (x,y,z)(x,y,z) in space, and a density at each point which acts like a differential opacity controlling how much radiance is accumulated by a ray passing through (x,y,z)(x,y,z). Our method optimizes a deep fully-connected neural network without any convolutional layers (often referred to as a multilayer perceptron or MLP) to represent this function by regressing from a single 5D coordinate (x,y,z,θ,ϕ)(x,y,z,\theta,\phi) to a single volume density and view-dependent RGB color. To render this neural radiance field (NeRF) from a particular viewpoint we: 1) march camera rays through the scene to generate a sampled set of 3D points, 2) use those points and their corresponding 2D viewing directions as input to the neural network to produce an output set of colors and densities, and 3) use classical volume rendering techniques to accumulate those colors and densities into a 2D image. Because this process is naturally differentiable, we can use gradient descent to optimize this model by minimizing the error between each observed image and the corresponding views rendered from our representation. Minimizing this error across multiple views encourages the network to predict a coherent model of the scene by assigning high volume densities and accurate colors to the locations that contain the true underlying scene content. Figure 2 visualizes this overall pipeline.

We find that the basic implementation of optimizing a neural radiance field representation for a complex scene does not converge to a sufficiently high-resolution representation and is inefficient in the required number of samples per camera ray. We address these issues by transforming input 5D coordinates with a positional encoding that enables the MLP to represent higher frequency functions, and we propose a hierarchical sampling procedure to reduce the number of queries required to adequately sample this high-frequency scene representation.

Our approach inherits the benefits of volumetric representations: both can represent complex real-world geometry and appearance and are well suited for gradient-based optimization using projected images. Crucially, our method overcomes the prohibitive storage costs of discretized voxel grids when modeling complex scenes at high-resolutions. In summary, our technical contributions are:

- – An approach for representing continuous scenes with complex geometry and materials as 5D neural radiance fields, parameterized as basic MLP networks.
An approach for representing continuous scenes with complex geometry and materials as 5D neural radiance fields, parameterized as basic MLP networks.

- – A differentiable rendering procedure based on classical volume rendering techniques, which we use to optimize these representations from standard RGB images. This includes a hierarchical sampling strategy to allocate the MLP’s capacity towards space with visible scene content.
A differentiable rendering procedure based on classical volume rendering techniques, which we use to optimize these representations from standard RGB images. This includes a hierarchical sampling strategy to allocate the MLP’s capacity towards space with visible scene content.

- – A positional encoding to map each input 5D coordinate into a higher dimensional space, which enables us to successfully optimize neural radiance fields to represent high-frequency scene content.
A positional encoding to map each input 5D coordinate into a higher dimensional space, which enables us to successfully optimize neural radiance fields to represent high-frequency scene content.

We demonstrate that our resulting neural radiance field method quantitatively and qualitatively outperforms state-of-the-art view synthesis methods, including works that fit neural 3D representations to scenes as well as works that train deep convolutional networks to predict sampled volumetric representations. As far as we know, this paper presents the first continuous neural scene representation that is able to render high-resolution photorealistic novel views of real objects and scenes from RGB images captured in natural settings.

## 2 Related Work

A promising recent direction in computer vision is encoding objects and scenes in the weights of an MLP that directly maps from a 3D spatial location to an implicit representation of the shape, such as the signed distance [6] at that location. However, these methods have so far been unable to reproduce realistic scenes with complex geometry with the same fidelity as techniques that represent scenes using discrete representations such as triangle meshes or voxel grids. In this section, we review these two lines of work and contrast them with our approach, which enhances the capabilities of neural scene representations to produce state-of-the-art results for rendering complex realistic scenes.

A similar approach of using MLPs to map from low-dimensional coordinates to colors has also been used for representing other graphics functions such as images [44], textured materials [12, 31, 36, 37], and indirect illumination values [38].

Recent work has investigated the implicit representation of continuous 3D shapes as level sets by optimizing deep networks that map x​y​zxyz coordinates to signed distance functions [15, 32] or occupancy fields [11, 27]. However, these models are limited by their requirement of access to ground truth 3D geometry, typically obtained from synthetic 3D shape datasets such as ShapeNet [3]. Subsequent work has relaxed this requirement of ground truth 3D shapes by formulating differentiable rendering functions that allow neural implicit shape representations to be optimized using only 2D images. Niemeyer et al. [29] represent surfaces as 3D occupancy fields and use a numerical method to find the surface intersection for each ray, then calculate an exact derivative using implicit differentiation. Each ray intersection location is provided as the input to a neural 3D texture field that predicts a diffuse color for that point. Sitzmann et al. [42] use a less direct neural 3D representation that simply outputs a feature vector and RGB color at each continuous 3D coordinate, and propose a differentiable rendering function consisting of a recurrent neural network that marches along each ray to decide where the surface is located.

Though these techniques can potentially represent complicated and high-resolution geometry, they have so far been limited to simple shapes with low geometric complexity, resulting in oversmoothed renderings. We show that an alternate strategy of optimizing networks to encode 5D radiance fields (3D volumes with 2D view-dependent appearance) can represent higher-resolution geometry and appearance to render photorealistic novel views of complex scenes.

Given a dense sampling of views, photorealistic novel views can be reconstructed by simple light field sample interpolation techniques [21, 5, 7]. For novel view synthesis with sparser view sampling, the computer vision and graphics communities have made significant progress by predicting traditional geometry and appearance representations from observed images. One popular class of approaches uses mesh-based representations of scenes with either diffuse [48] or view-dependent [2, 8, 49] appearance. Differentiable rasterizers [4, 10, 23, 25] or pathtracers [22, 30] can directly optimize mesh representations to reproduce a set of input images using gradient descent. However, gradient-based mesh optimization based on image reprojection is often difficult, likely because of local minima or poor conditioning of the loss landscape. Furthermore, this strategy requires a template mesh with fixed topology to be provided as an initialization before optimization [22], which is typically unavailable for unconstrained real-world scenes.

Another class of methods use volumetric representations to address the task of high-quality photorealistic view synthesis from a set of input RGB images. Volumetric approaches are able to realistically represent complex shapes and materials, are well-suited for gradient-based optimization, and tend to produce less visually distracting artifacts than mesh-based methods. Early volumetric approaches used observed images to directly color voxel grids [19, 40, 45]. More recently, several methods [9, 13, 17, 28, 33, 43, 46, 52] have used large datasets of multiple scenes to train deep networks that predict a sampled volumetric representation from a set of input images, and then use either alpha-compositing [34] or learned compositing along rays to render novel views at test time. Other works have optimized a combination of convolutional networks (CNNs) and sampled voxel grids for each specific scene, such that the CNN can compensate for discretization artifacts from low resolution voxel grids [41] or allow the predicted voxel grids to vary based on input time or animation controls [24]. While these volumetric techniques have achieved impressive results for novel view synthesis, their ability to scale to higher resolution imagery is fundamentally limited by poor time and space complexity due to their discrete sampling — rendering higher resolution images requires a finer sampling of 3D space. We circumvent this problem by instead encoding a continuous volume within the parameters of a deep fully-connected neural network, which not only produces significantly higher quality renderings than prior volumetric approaches, but also requires just a fraction of the storage cost of those sampled volumetric representations.

## 3 Neural Radiance Field Scene Representation

We represent a continuous scene as a 5D vector-valued function whose input is a 3D location 𝐱=(x,y,z)\mathbf{x}=(x,y,z) and 2D viewing direction (θ,ϕ)(\theta,\phi), and whose output is an emitted color 𝐜=(r,g,b)\mathbf{c}=(r,g,b) and volume density σ\sigma. In practice, we express direction as a 3D Cartesian unit vector 𝐝\mathbf{d}. We approximate this continuous 5D scene representation with an MLP network FΘ:(𝐱,𝐝)→(𝐜,σ)F_{\mathrm{\Theta}}:(\mathbf{x},\mathbf{d})\to(\mathbf{c},\sigma) and optimize its weights Θ\mathrm{\Theta} to map from each input 5D coordinate to its corresponding volume density and directional emitted color.

We encourage the representation to be multiview consistent by restricting the network to predict the volume density σ\sigma as a function of only the location 𝐱\mathbf{x}, while allowing the RGB color 𝐜\mathbf{c} to be predicted as a function of both location and viewing direction. To accomplish this, the MLP FΘF_{\mathrm{\Theta}} first processes the input 3D coordinate 𝐱\mathbf{x} with 8 fully-connected layers (using ReLU activations and 256 channels per layer), and outputs σ\sigma and a 256-dimensional feature vector. This feature vector is then concatenated with the camera ray’s viewing direction and passed to one additional fully-connected layer (using a ReLU activation and 128 channels) that output the view-dependent RGB color.

See Fig. 3 for an example of how our method uses the input viewing direction to represent non-Lambertian effects. As shown in Fig. 4, a model trained without view dependence (only 𝐱\mathbf{x} as input) has difficulty representing specularities.

## 4 Volume Rendering with Radiance Fields

Our 5D neural radiance field represents a scene as the volume density and directional emitted radiance at any point in space. We render the color of any ray passing through the scene using principles from classical volume rendering [16]. The volume density σ⁡(𝐱)\sigma(\mathbf{x}) can be interpreted as the differential probability of a ray terminating at an infinitesimal particle at location 𝐱\mathbf{x}. The expected color C⁡(𝐫)C(\mathbf{r}) of camera ray 𝐫⁡(t)=𝐨+t​𝐝\mathbf{r}(t)=\mathbf{o}+t\mathbf{d} with near and far bounds tnt_{n} and tft_{f} is:

|  | OPENC⁡(𝐫)=∫tntfT⁡(t)​σ​(𝐫⁡(t))​𝐜​(𝐫⁡(t),𝐝)​𝑑t, where ​T​(t)=exp⁡(−∫tntσ(𝐫(s))ds).C(\mathbf{r})=\int_{t_{n}}^{t_{f}}T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})dt\,,\textrm{ where }T(t)=\exp\mathopen{}\mathclose{{\left(-\int_{t_{n}}^{t}\sigma(\mathbf{r}(s))ds}}\right)\,. |  | (1) |
|---|---|---|---|

The function T⁡(t)T(t) denotes the accumulated transmittance along the ray from tnt_{n} to tt, i.e., the probability that the ray travels from tnt_{n} to tt without hitting any other particle. Rendering a view from our continuous neural radiance field requires estimating this integral C⁡(𝐫)C(\mathbf{r}) for a camera ray traced through each pixel of the desired virtual camera.

We numerically estimate this continuous integral using quadrature. Deterministic quadrature, which is typically used for rendering discretized voxel grids, would effectively limit our representation’s resolution because the MLP would only be queried at a fixed discrete set of locations. Instead, we use a stratified sampling approach where we partition [tn,tf][t_{n},t_{f}] into NN evenly-spaced bins and then draw one sample uniformly at random from within each bin:

|  | ti∼𝒰[tn+i−1N(tf−tn),tn+iN(tf−tn)].t_{i}\sim\mathcal{U}\mathopen{}\mathclose{{\left[t_{n}+\frac{i-1}{N}(t_{f}-t_{n}),\,\,t_{n}+\frac{i}{N}(t_{f}-t_{n})}}\right]\,. |  | (2) |
|---|---|---|---|

Although we use a discrete set of samples to estimate the integral, stratified sampling enables us to represent a continuous scene representation because it results in the MLP being evaluated at continuous positions over the course of optimization. We use these samples to estimate C⁡(𝐫)C(\mathbf{r}) with the quadrature rule discussed in the volume rendering review by Max [26]:

|  | OPENOPENC^​(𝐫)=∑i=1NTi​(1−exp⁡(−σi​δiCLOSE))​𝐜i, where ​Ti=exp⁡(−∑j=1i−1σjδj),\hat{C}(\mathbf{r})=\sum_{i=1}^{N}T_{i}(1-\exp\mathopen{}\mathclose{{\left(-\sigma_{i}\delta_{i}}}\right))\mathbf{c}_{i}\,,\textrm{ where }T_{i}=\exp\mathopen{}\mathclose{{\left(-\sum_{j=1}^{i-1}\sigma_{j}\delta_{j}}}\right)\,, |  | (3) |
|---|---|---|---|

where δi=ti+1−ti\delta_{i}=t_{i+1}-t_{i} is the distance between adjacent samples. This function for calculating C^​(𝐫)\hat{C}(\mathbf{r}) from the set of (𝐜i,σi)(\mathbf{c}_{i},\sigma_{i}) values is trivially differentiable and reduces to traditional alpha compositing with alpha values OPENαi=1−exp⁡(−σi​δiCLOSE)\alpha_{i}=1-\exp\mathopen{}\mathclose{{\left(-\sigma_{i}\delta_{i}}}\right).

## 5 Optimizing a Neural Radiance Field

In the previous section we have described the core components necessary for modeling a scene as a neural radiance field and rendering novel views from this representation. However, we observe that these components are not sufficient for achieving state-of-the-art quality, as demonstrated in Section 6.4). We introduce two improvements to enable representing high-resolution complex scenes. The first is a positional encoding of the input coordinates that assists the MLP in representing high-frequency functions, and the second is a hierarchical sampling procedure that allows us to efficiently sample this high-frequency representation.

|  |  |  |  |
|---|---|---|---|
| Ground Truth | Complete Model | No View Dependence | No Positional Encoding |

### 5.1 Positional encoding

Despite the fact that neural networks are universal function approximators [14], we found that having the network FΘF_{\mathrm{\Theta}} directly operate on x​y​z​θ​ϕxyz\theta\phi input coordinates results in renderings that perform poorly at representing high-frequency variation in color and geometry. This is consistent with recent work by Rahaman et al. [35], which shows that deep networks are biased towards learning lower frequency functions. They additionally show that mapping the inputs to a higher dimensional space using high frequency functions before passing them to the network enables better fitting of data that contains high frequency variation.

We leverage these findings in the context of neural scene representations, and show that reformulating FΘF_{\mathrm{\Theta}} as a composition of two functions FΘ=FΘ′∘γ{F_{\mathrm{\Theta}}=F^{\prime}_{\mathrm{\Theta}}\circ\gamma}, one learned and one not, significantly improves performance (see Fig. 4 and Table 2). Here γ\gamma is a mapping from ℝ\mathbb{R} into a higher dimensional space ℝ2​L\mathbb{R}^{2L}, and FΘ′F^{\prime}_{\mathrm{\Theta}} is still simply a regular MLP. Formally, the encoding function we use is:

|  | OPENγ⁡(p)=(OPENsin⁡(20​π​pCLOSE),OPENcos⁡(20​π​pCLOSE),⋯,OPENsin⁡(2L−1​π​pCLOSE),OPENcos⁡(2L−1​π​pCLOSE)).\gamma(p)=\mathopen{}\mathclose{{\left(\begin{array}[]{ccccc}\sin\mathopen{}\mathclose{{\left(2^{0}\pi p}}\right),&\cos\mathopen{}\mathclose{{\left(2^{0}\pi p}}\right),&\cdots,&\sin\mathopen{}\mathclose{{\left(2^{L-1}\pi p}}\right),&\cos\mathopen{}\mathclose{{\left(2^{L-1}\pi p}}\right)\end{array}}}\right)\,. |  | (4) |
|---|---|---|---|

This function γ⁡(⋅)\gamma(\cdot) is applied separately to each of the three coordinate values in 𝐱\mathbf{x} (which are normalized to lie in [−1,1][-1,1]) and to the three components of the Cartesian viewing direction unit vector 𝐝\mathbf{d} (which by construction lie in [−1,1][-1,1]). In our experiments, we set L=10L=10 for γ⁡(𝐱)\gamma(\mathbf{x}) and L=4L=4 for γ⁡(𝐝)\gamma(\mathbf{d}).

A similar mapping is used in the popular Transformer architecture [47], where it is referred to as a positional encoding. However, Transformers use it for a different goal of providing the discrete positions of tokens in a sequence as input to an architecture that does not contain any notion of order. In contrast, we use these functions to map continuous input coordinates into a higher dimensional space to enable our MLP to more easily approximate a higher frequency function. Concurrent work on a related problem of modeling 3D protein structure from projections [51] also utilizes a similar input coordinate mapping.

### 5.2 Hierarchical volume sampling

Our rendering strategy of densely evaluating the neural radiance field network at NN query points along each camera ray is inefficient: free space and occluded regions that do not contribute to the rendered image are still sampled repeatedly. We draw inspiration from early work in volume rendering [20] and propose a hierarchical representation that increases rendering efficiency by allocating samples proportionally to their expected effect on the final rendering.

Instead of just using a single network to represent the scene, we simultaneously optimize two networks: one “coarse” and one “fine”. We first sample a set of NcN_{c} locations using stratified sampling, and evaluate the “coarse” network at these locations as described in Eqns. 2 and 3. Given the output of this “coarse” network, we then produce a more informed sampling of points along each ray where samples are biased towards the relevant parts of the volume. To do this, we first rewrite the alpha composited color from the coarse network C^c​(𝐫)\hat{C}_{c}(\mathbf{r}) in Eqn. 3 as a weighted sum of all sampled colors cic_{i} along the ray:

|  | C^c(𝐫)=∑i=1Ncwici,wi=Ti(1−exp(−σi​δiCLOSE)).\hat{C}_{c}(\mathbf{r})=\sum_{i=1}^{N_{c}}w_{i}c_{i}\,,\quad\,\,w_{i}=T_{i}(1-\exp\mathopen{}\mathclose{{\left(-\sigma_{i}\delta_{i}}}\right))\,. |  | (5) |
|---|---|---|---|

Normalizing these weights as w^i=wi/∑j=1Ncwj\hat{w}_{i}=\nicefrac{{w_{i}}}{{\sum_{j=1}^{N_{c}}w_{j}}} produces a piecewise-constant PDF along the ray. We sample a second set of NfN_{f} locations from this distribution using inverse transform sampling, evaluate our “fine” network at the union of the first and second set of samples, and compute the final rendered color of the ray C^f​(𝐫)\hat{C}_{f}(\mathbf{r}) using Eqn. 3 but using all Nc+NfN_{c}+N_{f} samples. This procedure allocates more samples to regions we expect to contain visible content. This addresses a similar goal as importance sampling, but we use the sampled values as a nonuniform discretization of the whole integration domain rather than treating each sample as an independent probabilistic estimate of the entire integral.

### 5.3 Implementation details

We optimize a separate neural continuous volume representation network for each scene. This requires only a dataset of captured RGB images of the scene, the corresponding camera poses and intrinsic parameters, and scene bounds (we use ground truth camera poses, intrinsics, and bounds for synthetic data, and use the COLMAP structure-from-motion package [39] to estimate these parameters for real data). At each optimization iteration, we randomly sample a batch of camera rays from the set of all pixels in the dataset, and then follow the hierarchical sampling described in Sec. 5.2 to query NcN_{c} samples from the coarse network and Nc+NfN_{c}+N_{f} samples from the fine network. We then use the volume rendering procedure described in Sec. 4 to render the color of each ray from both sets of samples. Our loss is simply the total squared error between the rendered and true pixel colors for both the coarse and fine renderings:

|  | ℒ=∑𝐫∈ℛ[‖C^c(𝐫)−C(𝐫)‖22+‖C^f(𝐫)−C(𝐫)‖22]\mathcal{L}=\sum_{\mathbf{r}\in\mathcal{R}}\mathopen{}\mathclose{{\left[\mathopen{}\mathclose{{\left\lVert\hat{C}_{c}(\mathbf{r})-C(\mathbf{r})}}\right\rVert_{2}^{2}+\mathopen{}\mathclose{{\left\lVert\hat{C}_{f}(\mathbf{r})-C(\mathbf{r})}}\right\rVert_{2}^{2}}}\right] |  | (6) |
|---|---|---|---|

where ℛ\mathcal{R} is the set of rays in each batch, and C⁡(𝐫)C(\mathbf{r}), C^c​(𝐫)\hat{C}_{c}(\mathbf{r}), and C^f​(𝐫)\hat{C}_{f}(\mathbf{r}) are the ground truth, coarse volume predicted, and fine volume predicted RGB colors for ray 𝐫\mathbf{r} respectively. Note that even though the final rendering comes from C^f​(𝐫)\hat{C}_{f}(\mathbf{r}), we also minimize the loss of C^c​(𝐫)\hat{C}_{c}(\mathbf{r}) so that the weight distribution from the coarse network can be used to allocate samples in the fine network.

In our experiments, we use a batch size of 4096 rays, each sampled at Nc=64N_{c}=64 coordinates in the coarse volume and Nf=128N_{f}=128 additional coordinates in the fine volume. We use the Adam optimizer [18] with a learning rate that begins at 5×10−45\times 10^{-4} and decays exponentially to 5×10−55\times 10^{-5} over the course of optimization (other Adam hyperparameters are left at default values of β1=0.9\beta_{1}=0.9, β2=0.999\beta_{2}=0.999, and ϵ=10−7\epsilon=10^{-7}). The optimization for a single scene typically take around 100–300k iterations to converge on a single NVIDIA V100 GPU (about 1–2 days).

## 6 Results

We quantitatively (Tables 1) and qualitatively (Figs. 8 and 6) show that our method outperforms prior work, and provide extensive ablation studies to validate our design choices (Table 2). We urge the reader to view our supplementary video to better appreciate our method’s significant improvement over baseline methods when rendering smooth paths of novel views.

### 6.1 Datasets

We first show experimental results on two datasets of synthetic renderings of objects (Table 1, “Diffuse Synthetic 360​°360\degree” and “Realistic Synthetic 360​°360\degree”). The DeepVoxels [41] dataset contains four Lambertian objects with simple geometry. Each object is rendered at 512×512512\times 512 pixels from viewpoints sampled on the upper hemisphere (479 as input and 1000 for testing). We additionally generate our own dataset containing pathtraced images of eight objects that exhibit complicated geometry and realistic non-Lambertian materials. Six are rendered from viewpoints sampled on the upper hemisphere, and two are rendered from viewpoints sampled on a full sphere. We render 100 views of each scene as input and 200 for testing, all at 800×800800\times 800 pixels.

| Ship |  |  |  |  |  |
|---|---|---|---|---|---|
| Lego |  |  |  |  |  |
| Microphone |  |  |  |  |  |
| Materials |  |  |  |  |  |
|  | Ground Truth | NeRF (ours) | LLFF [28] | SRN [42] | NV [24] |

We show results on complex real-world scenes captured with roughly forward-facing images (Table 1, “Real Forward-Facing”). This dataset consists of 8 scenes captured with a handheld cellphone (5 taken from the LLFF paper and 3 that we capture), captured with 20 to 62 images, and hold out 1/8\nicefrac{{1}}{{8}} of these for the test set. All images are 1008×7561008\times 756 pixels.

| Fern |  |  |  |  |
|---|---|---|---|---|
| T-Rex |  |  |  |  |
| Orchid |  |  |  |  |
|  | Ground Truth | NeRF (ours) | LLFF [28] | SRN [42] |

|  | Diffuse Synthetic 360∘360^{\circ} [41] | Realistic Synthetic 360∘360^{\circ} | Real Forward-Facing [28] |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
| Method | PSNR↑\uparrow | SSIM↑\uparrow | LPIPS↓\downarrow | PSNR↑\uparrow | SSIM↑\uparrow | LPIPS↓\downarrow | PSNR↑\uparrow | SSIM↑\uparrow | LPIPS↓\downarrow |
| SRN [42] | 33.2033.20 | 0.9630.963 | 0.0730.073 | 22.2622.26 | 0.8460.846 | 0.1700.170 | 22.8422.84 | 0.6680.668 | 0.3780.378 |
| NV [24] | 29.6229.62 | 0.9290.929 | 0.0990.099 | 26.0526.05 | 0.8930.893 | 0.1600.160 | - | - | - |
| LLFF [28] | 34.3834.38 | 0.9850.985 | 0.0480.048 | 24.8824.88 | 0.9110.911 | 0.1140.114 | 24.1324.13 | 0.7980.798 | 0.212\mathbf{0.212} |
| Ours | 40.15\mathbf{40.15} | 0.991\mathbf{0.991} | 0.023\mathbf{0.023} | 31.01\mathbf{31.01} | 0.947\mathbf{0.947} | 0.081\mathbf{0.081} | 26.50\mathbf{26.50} | 0.811\mathbf{0.811} | 0.2500.250 |

### 6.2 Comparisons

To evaluate our model we compare against current top-performing techniques for view synthesis, detailed below. All methods use the same set of input views to train a separate network for each scene except Local Light Field Fusion [28], which trains a single 3D convolutional network on a large dataset, then uses the same trained network to process input images of new scenes at test time.

synthesizes novel views of objects that lie entirely within a bounded volume in front of a distinct background (which must be separately captured without the object of interest). It optimizes a deep 3D convolutional network to predict a discretized RGBα\alpha voxel grid with 1283128^{3} samples as well as a 3D warp grid with 32332^{3} samples. The algorithm renders novel views by marching camera rays through the warped voxel grid.

represent a continuous scene as an opaque surface, implicitly defined by a MLP that maps each (x,y,z)(x,y,z) coordinate to a feature vector. They train a recurrent neural network to march along a ray through the scene representation by using the feature vector at any 3D coordinate to predict the next step size along the ray. The feature vector from the final step is decoded into a single color for that point on the surface. Note that SRN is a better-performing followup to DeepVoxels [41] by the same authors, which is why we do not include comparisons to DeepVoxels.

LLFF is designed for producing photorealistic novel views for well-sampled forward facing scenes. It uses a trained 3D convolutional network to directly predict a discretized frustum-sampled RGBα\alpha grid (multiplane image or MPI [52]) for each input view, then renders novel views by alpha compositing and blending nearby MPIs into the novel viewpoint.

### 6.3 Discussion

We thoroughly outperform both baselines that also optimize a separate network per scene (NV and SRN) in all scenarios. Furthermore, we produce qualitatively and quantitatively superior renderings compared to LLFF (across all except one metric) while using only their input images as our entire training set.

The SRN method produces heavily smoothed geometry and texture, and its representational power for view synthesis is limited by selecting only a single depth and color per camera ray. The NV baseline is able to capture reasonably detailed volumetric geometry and appearance, but its use of an underlying explicit 1283128^{3} voxel grid prevents it from scaling to represent fine details at high resolutions. LLFF specifically provides a “sampling guideline” to not exceed 64 pixels of disparity between input views, so it frequently fails to estimate correct geometry in the synthetic datasets which contain up to 400-500 pixels of disparity between views. Additionally, LLFF blends between different scene representations for rendering different views, resulting in perceptually-distracting inconsistency as is apparent in our supplementary video.

The biggest practical tradeoffs between these methods are time versus space. All compared single scene methods take at least 12 hours to train per scene. In contrast, LLFF can process a small input dataset in under 10 minutes. However, LLFF produces a large 3D voxel grid for every input image, resulting in enormous storage requirements (over 15GB for one “Realistic Synthetic” scene). Our method requires only 5 MB for the network weights (a relative compression of 3000×3000\times compared to LLFF), which is even less memory than the input images alone for a single scene from any of our datasets.

### 6.4 Ablation studies

|  |  | Input | #Im. | LL | (Nc,Nf)(\,N_{c}\,,\,N_{f}\,) | PSNR↑\uparrow | SSIM↑\uparrow | LPIPS↓\downarrow |
|---|---|---|---|---|---|---|---|---|
| 1) | No PE, VD, H | x​y​zxyz | 100 | - | (256, - ) | 26.6726.67 | 0.9060.906 | 0.1360.136 |
| 2) | No Pos. Encoding | x​y​z​θ​ϕxyz\theta\phi | 100 | - | (64, 128) | 28.7728.77 | 0.9240.924 | 0.1080.108 |
| 3) | No View Dependence | x​y​zxyz | 100 | 10 | (64, 128) | 27.6627.66 | 0.9250.925 | 0.1170.117 |
| 4) | No Hierarchical | x​y​z​θ​ϕxyz\theta\phi | 100 | 10 | (256, - ) | 30.0630.06 | 0.9380.938 | 0.1090.109 |
| 5) | Far Fewer Images | x​y​z​θ​ϕxyz\theta\phi | 25 | 10 | (64, 128) | 27.7827.78 | 0.9250.925 | 0.1070.107 |
| 6) | Fewer Images | x​y​z​θ​ϕxyz\theta\phi | 50 | 10 | (64, 128) | 29.7929.79 | 0.9400.940 | 0.0960.096 |
| 7) | Fewer Frequencies | x​y​z​θ​ϕxyz\theta\phi | 100 | 5 | (64, 128) | 30.5930.59 | 0.9440.944 | 0.0880.088 |
| 8) | More Frequencies | x​y​z​θ​ϕxyz\theta\phi | 100 | 15 | (64, 128) | 30.8130.81 | 0.9460.946 | 0.0960.096 |
| 9) | Complete Model | x​y​z​θ​ϕxyz\theta\phi | 100 | 10 | (64, 128) | 31.01{\mathbf{31.01}} | 0.947{\mathbf{0.947}} | 0.081{\mathbf{0.081}} |

We validate our algorithm’s design choices and parameters with an extensive ablation study in Table 2. We present results on our “Realistic Synthetic 360​°360\degree” scenes. Row 9 shows our complete model as a point of reference. Row 1 shows a minimalist version of our model without positional encoding (PE), view-dependence (VD), or hierarchical sampling (H). In rows 2–4 we remove these three components one at a time from the full model, observing that positional encoding (row 2) and view-dependence (row 3) provide the largest quantitative benefit followed by hierarchical sampling (row 4). Rows 5–6 show how our performance decreases as the number of input images is reduced. Note that our method’s performance using only 25 input images still exceeds NV, SRN, and LLFF across all metrics when they are provided with 100 images (see supplementary material). In rows 7–8 we validate our choice of the maximum frequency LL used in our positional encoding for 𝐱\mathbf{x} (the maximum frequency used for 𝐝\mathbf{d} is scaled proportionally). Only using 5 frequencies reduces performance, but increasing the number of frequencies from 10 to 15 does not improve performance. We believe the benefit of increasing LL is limited once 2L2^{L} exceeds the maximum frequency present in the sampled input images (roughly 1024 in our data).

## 7 Conclusion

Our work directly addresses deficiencies of prior work that uses MLPs to represent objects and scenes as continuous functions. We demonstrate that representing scenes as 5D neural radiance fields (an MLP that outputs volume density and view-dependent emitted radiance as a function of 3D location and 2D viewing direction) produces better renderings than the previously-dominant approach of training deep convolutional networks to output discretized voxel representations.

Although we have proposed a hierarchical sampling strategy to make rendering more sample-efficient (for both training and testing), there is still much more progress to be made in investigating techniques to efficiently optimize and render neural radiance fields. Another direction for future work is interpretability: sampled representations such as voxel grids and meshes admit reasoning about the expected quality of rendered views and failure modes, but it is unclear how to analyze these issues when we encode scenes in the weights of a deep neural network. We believe that this work makes progress towards a graphics pipeline based on real world imagery, where complex scenes could be composed of neural radiance fields optimized from images of actual objects and scenes.

Acknowledgements We thank Kevin Cao, Guowei Frank Yang, and Nithin Raghavan for comments and discussions. RR acknowledges funding from ONR grants N000141712687 and N000142012529 and the Ronald L. Graham Chair. BM is funded by a Hertz Foundation Fellowship, and MT is funded by an NSF Graduate Fellowship. Google provided a generous donation of cloud compute credits through the BAIR Commons program. We thank the following Blend Swap users for the models used in our realistic synthetic dataset: gregzaal (ship), 1DInc (chair), bryanajones (drums), Herberhold (ficus), erickfree (hotdog), Heinzelnisse (lego), elbrujodelatribu (materials), and up3d.de (mic).

## Appendix 0.A Additional Implementation Details

Fig. 7 details our simple fully-connected architecture.

Our method renders views by querying the neural radiance field representation at continuous 5D coordinates along camera rays. For experiments with synthetic images, we scale the scene so that it lies within a cube of side length 2 centered at the origin, and only query the representation within this bounding volume. Our dataset of real images contains content that can exist anywhere between the closest point and infinity, so we use normalized device coordinates to map the depth range of these points into [−1,1][-1,1]. This shifts all the ray origins to the near plane of the scene, maps the perspective rays of the camera to parallel rays in the transformed volume, and uses disparity (inverse depth) instead of metric depth, so all coordinates are now bounded.

For real scene data, we regularize our network by adding random Gaussian noise with zero mean and unit variance to the output σ\sigma values (before passing them through the ReLU) during optimization, finding that this slightly improves visual performance for rendering novel views. We implement our model in Tensorflow [1].

To render new views at test time, we sample 6464 points per ray through the coarse network and 64+128=19264+128=192 points per ray through the fine network, for a total of 256256 network queries per ray. Our realistic synthetic dataset requires 640k rays per image, and our real scenes require 762k rays per image, resulting in between 150 and 200 million network queries per rendered image. On an NVIDIA V100, this takes approximately 30 seconds per frame.

## Appendix 0.B Additional Baseline Method Details

We use the NV code open-sourced by the authors at https://github.com/facebookresearch/neuralvolumes and follow their procedure for training on a single scene without time dependence.

We use the SRN code open-sourced by the authors at https://github.com/vsitzmann/scene-representation-networks and follow their procedure for training on a single scene.

We use the pretrained LLFF model open-sourced by the authors at https://github.com/Fyusion/LLFF.

The SRN implementation published by the authors requires a significant amount of GPU memory, and is limited to an image resolution of 512×512512\times 512 pixels even when parallelized across 4 NVIDIA V100 GPUs. We compute quantitative metrics for SRN at 512×512512\times 512 pixels for our synthetic datasets and 504×376504\times 376 pixels for the real datasets, in comparison to 800×800800\times 800 and 1008×7521008\times 752 respectively for the other methods that can be run at higher resolutions.

## Appendix 0.C NDC ray space derivation

We reconstruct real scenes with “forward facing” captures in the normalized device coordinate (NDC) space that is commonly used as part of the triangle rasterization pipeline. This space is convenient because it preserves parallel lines while converting the zz axis (camera axis) to be linear in disparity.

Here we derive the transformation which is applied to rays to map them from camera space to NDC space. The standard 3D perspective projection matrix for homogeneous coordinates is:

|  | M=(nr0000nt0000−(f+n)f−n−2​f​nf−n00−10)M=\begin{pmatrix}\frac{n}{r}&0&0&0\\ 0&\frac{n}{t}&0&0\\ 0&0&\frac{-(f+n)}{f-n}&\frac{-2fn}{f-n}\\ 0&0&-1&0\end{pmatrix} |  | (7) |
|---|---|---|---|

where n,fn,f are the near and far clipping planes and rr and tt are the right and top bounds of the scene at the near clipping plane. (Note that this is in the convention where the camera is looking in the −z-z direction.) To project a homogeneous point (x,y,z,1)⊤(x,y,z,1)^{\top}, we left-multiply by M and then divide by the fourth coordinate:

|  | (nr0000nt0000−(f+n)f−n−2​f​nf−n00−10)​(xyz1)\displaystyle\begin{pmatrix}\frac{n}{r}&0&0&0\\ 0&\frac{n}{t}&0&0\\ 0&0&\frac{-(f+n)}{f-n}&\frac{-2fn}{f-n}\\ 0&0&-1&0\end{pmatrix}\begin{pmatrix}x\\ y\\ z\\ 1\end{pmatrix} | =(nr​xnt​y−(f+n)f−n​z−−2​f​nf−n−z)\displaystyle=\begin{pmatrix}\frac{n}{r}x\\ \frac{n}{t}y\\ \frac{-(f+n)}{f-n}z-\frac{-2fn}{f-n}\\ -z\end{pmatrix} |  | (8) |
|---|---|---|---|---|
|  | project | →(nr​x−znt​y−z(f+n)f−n−2​f​nf−n​1−z)\displaystyle\rightarrow\begin{pmatrix}\frac{n}{r}\frac{x}{-z}\\ \frac{n}{t}\frac{y}{-z}\\ \frac{(f+n)}{f-n}-\frac{2fn}{f-n}\frac{1}{-z}\end{pmatrix} |  | (9) |

The projected point is now in normalized device coordinate (NDC) space, where the original viewing frustum has been mapped to the cube [−1,1]3[-1,1]^{3}.

Our goal is to take a ray 𝐨+t​𝐝\mathbf{o}+t\mathbf{d} and calculate a ray origin 𝐨′\mathbf{o}^{\prime} and direction 𝐝′\mathbf{d}^{\prime} in NDC space such that for every tt, there exists a new t′t^{\prime} for which π⁡(𝐨+t​𝐝)=𝐨′+t′​𝐝′\pi(\mathbf{o}+t\mathbf{d})=\mathbf{o}^{\prime}+t^{\prime}\mathbf{d}^{\prime} (where π\pi is projection using the above matrix). In other words, the projection of the original ray and the NDC space ray trace out the same points (but not necessarily at the same rate).

Let us rewrite the projected point from Eqn. 9 as (ax​x/z,ay​y/z,az+bz/z)⊤(a_{x}x/z,a_{y}y/z,a_{z}+b_{z}/z)^{\top}. The components of the new origin 𝐨′\mathbf{o}^{\prime} and direction 𝐝′\mathbf{d}^{\prime} must satisfy:

|  | (ax​ox+t​dxoz+t​dzay​oy+t​dyoz+t​dzaz+bzoz+t​dz)=(ox′+t′​dx′oy′+t′​dy′oz′+t′​dz′).\displaystyle\begin{pmatrix}a_{x}\frac{o_{x}+td_{x}}{o_{z}+td_{z}}\\[6.0pt] a_{y}\frac{o_{y}+td_{y}}{o_{z}+td_{z}}\\[6.0pt] a_{z}+\frac{b_{z}}{o_{z}+td_{z}}\end{pmatrix}=\begin{pmatrix}o_{x}^{\prime}+t^{\prime}d_{x}^{\prime}\\ o_{y}^{\prime}+t^{\prime}d_{y}^{\prime}\\ o_{z}^{\prime}+t^{\prime}d_{z}^{\prime}\end{pmatrix}\,. |  | (10) |
|---|---|---|---|

To eliminate a degree of freedom, we decide that t′=0t^{\prime}=0 and t=0t=0 should map to the same point. Substituting t=0t=0 and t′=0t^{\prime}=0 Eqn. 10 directly gives our NDC space origin 𝐨′\mathbf{o}^{\prime}:

|  | 𝐨′=(ox′oy′oz′)=(ax​oxozay​oyozaz+bzoz)=π⁡(𝐨).\displaystyle\mathbf{o}^{\prime}=\begin{pmatrix}o_{x}^{\prime}\\ o_{y}^{\prime}\\ o_{z}^{\prime}\end{pmatrix}=\begin{pmatrix}a_{x}\frac{o_{x}}{o_{z}}\\[6.0pt] a_{y}\frac{o_{y}}{o_{z}}\\[6.0pt] a_{z}+\frac{b_{z}}{o_{z}}\end{pmatrix}=\pi(\mathbf{o})\,. |  | (11) |
|---|---|---|---|

This is exactly the projection π⁡(𝐨)\pi(\mathbf{o}) of the original ray’s origin. By substituting this back into Eqn. 10 for arbitrary tt, we can determine the values of t′t^{\prime} and 𝐝′\mathbf{d}^{\prime}:

|  | (t′​dx′t′​dy′t′​dz′)\displaystyle\begin{pmatrix}t^{\prime}d_{x}^{\prime}\\ t^{\prime}d_{y}^{\prime}\\ t^{\prime}d_{z}^{\prime}\end{pmatrix} | =(ax​ox+t​dxoz+t​dz−ax​oxozay​oy+t​dyoz+t​dz−ay​oyozaz+bzoz+t​dz−az−bzoz)\displaystyle=\begin{pmatrix}a_{x}\frac{o_{x}+td_{x}}{o_{z}+td_{z}}-a_{x}\frac{o_{x}}{o_{z}}\\[6.0pt] a_{y}\frac{o_{y}+td_{y}}{o_{z}+td_{z}}-a_{y}\frac{o_{y}}{o_{z}}\\[6.0pt] a_{z}+\frac{b_{z}}{o_{z}+td_{z}}-a_{z}-\frac{b_{z}}{o_{z}}\end{pmatrix} |  | (12) |
|---|---|---|---|---|
|  |  | =(ax​oz​(ox+t​dx)−ox​(oz+t​dz)(oz+t​dz)​ozay​oz​(oy+t​dy)−oy​(oz+t​dz)(oz+t​dz)​ozbz​oz−(oz+t​dz)(oz+t​dz)​oz)\displaystyle=\begin{pmatrix}a_{x}\frac{o_{z}(o_{x}+td_{x})-o_{x}(o_{z}+td_{z})}{(o_{z}+td_{z})o_{z}}\\[6.0pt] a_{y}\frac{o_{z}(o_{y}+td_{y})-o_{y}(o_{z}+td_{z})}{(o_{z}+td_{z})o_{z}}\\[6.0pt] b_{z}\frac{o_{z}-(o_{z}+td_{z})}{(o_{z}+td_{z})o_{z}}\end{pmatrix} |  | (13) |
|  |  | =(OPENax​t​dzoz+t​dz​(dxdz−oxozCLOSE)OPENay​t​dzoz+t​dz​(dydz−oyozCLOSE)−bz​t​dzoz+t​dz​1oz)\displaystyle=\begin{pmatrix}a_{x}\frac{td_{z}}{o_{z}+td_{z}}\mathopen{}\mathclose{{\left(\frac{d_{x}}{d_{z}}-\frac{o_{x}}{o_{z}}}}\right)\\[6.0pt] a_{y}\frac{td_{z}}{o_{z}+td_{z}}\mathopen{}\mathclose{{\left(\frac{d_{y}}{d_{z}}-\frac{o_{y}}{o_{z}}}}\right)\\[6.0pt] -b_{z}\frac{td_{z}}{o_{z}+td_{z}}\frac{1}{o_{z}}\end{pmatrix} |  | (14) |

Factoring out a common expression that depends only on tt gives us:

|  | t′\displaystyle t^{\prime} | =t​dzoz+t​dz=1−ozoz+t​dz\displaystyle=\frac{td_{z}}{o_{z}+td_{z}}=1-\frac{o_{z}}{o_{z}+td_{z}} |  | (15) |
|---|---|---|---|---|
|  | 𝐝′\displaystyle\mathbf{d}^{\prime} | =(OPENax​(dxdz−oxozCLOSE)OPENay​(dydz−oyozCLOSE)−bz​1oz).\displaystyle=\begin{pmatrix}a_{x}\mathopen{}\mathclose{{\left(\frac{d_{x}}{d_{z}}-\frac{o_{x}}{o_{z}}}}\right)\\[6.0pt] a_{y}\mathopen{}\mathclose{{\left(\frac{d_{y}}{d_{z}}-\frac{o_{y}}{o_{z}}}}\right)\\[6.0pt] -b_{z}\frac{1}{o_{z}}\end{pmatrix}\,. |  | (16) |

Note that, as desired, t′=0t^{\prime}=0 when t=0t=0. Additionally, we see that t′→1t^{\prime}\to 1 as t→∞t\to\infty. Going back to the original projection matrix, our constants are:

|  | ax\displaystyle a_{x} | =−nr\displaystyle=-\frac{n}{r} |  | (17) |
|---|---|---|---|---|
|  | ay\displaystyle a_{y} | =−nt\displaystyle=-\frac{n}{t} |  | (18) |
|  | az\displaystyle a_{z} | =f+nf−n\displaystyle=\frac{f+n}{f-n} |  | (19) |
|  | bz\displaystyle b_{z} | =2​f​nf−n\displaystyle=\frac{2fn}{f-n} |  | (20) |

Using the standard pinhole camera model, we can reparameterize as:

|  | ax\displaystyle a_{x} | =−fc​a​mW/2\displaystyle=-\frac{f_{cam}}{W/2} |  | (21) |
|---|---|---|---|---|
|  | ay\displaystyle a_{y} | =−fc​a​mH/2\displaystyle=-\frac{f_{cam}}{H/2} |  | (22) |

where WW and HH are the width and height of the image in pixels and fc​a​mf_{cam} is the focal length of the camera.

In our real forward facing captures, we assume that the far scene bound is infinity (this costs us very little since NDC uses the zz dimension to represent inverse depth, i.e., disparity). In this limit the zz constants simplify to:

|  | az\displaystyle a_{z} | =1\displaystyle=1 |  | (23) |
|---|---|---|---|---|
|  | bz\displaystyle b_{z} | =2​n.\displaystyle=2n\,. |  | (24) |

Combining everything together:

|  | 𝐨′\displaystyle\mathbf{o}^{\prime} | =(−fc​a​mW/2​oxoz−fc​a​mH/2​oyoz1+2​noz)\displaystyle=\begin{pmatrix}-\frac{f_{cam}}{W/2}\frac{o_{x}}{o_{z}}\\[6.0pt] -\frac{f_{cam}}{H/2}\frac{o_{y}}{o_{z}}\\[6.0pt] 1+\frac{2n}{o_{z}}\end{pmatrix} |  | (25) |
|---|---|---|---|---|
|  | 𝐝′\displaystyle\mathbf{d}^{\prime} | =(OPEN−fc​a​mW/2​(dxdz−oxozCLOSE)OPEN−fc​a​mH/2​(dydz−oyozCLOSE)−2​n​1oz).\displaystyle=\begin{pmatrix}-\frac{f_{cam}}{W/2}\mathopen{}\mathclose{{\left(\frac{d_{x}}{d_{z}}-\frac{o_{x}}{o_{z}}}}\right)\\[6.0pt] -\frac{f_{cam}}{H/2}\mathopen{}\mathclose{{\left(\frac{d_{y}}{d_{z}}-\frac{o_{y}}{o_{z}}}}\right)\\[6.0pt] -2n\frac{1}{o_{z}}\end{pmatrix}\,. |  | (26) |

One final detail in our implementation: we shift 𝐨\mathbf{o} to the ray’s intersection with the near plane at z=−nz=-n (before this NDC conversion) by taking 𝐨n=𝐨+tn​𝐝\mathbf{o}_{n}=\mathbf{o}+t_{n}\mathbf{d} for tn=−(n+oz)/dzt_{n}=-(n+o_{z})/d_{z}. Once we convert to the NDC ray, this allows us to simply sample t′t^{\prime} linearly from 00 to 11 in order to get a linear sampling in disparity from nn to ∞\infty in the original space.

## Appendix 0.D Additional Results

Tables 3, 4, 5, and 6 include a breakdown of the quantitative results presented in the main paper into per-scene metrics. The per-scene breakdown is consistent with the aggregate quantitative metrics presented in the paper, where our method quantitatively outperforms all baselines. Although LLFF achieves slightly better LPIPS metrics, we urge readers to view our supplementary video where our method achieves better multiview consistency and produces fewer artifacts than all baselines.

| Pedestal |  |  |  |  |  |
|---|---|---|---|---|---|
| Cube |  |  |  |  |  |
|  | Ground Truth | NeRF (ours) | LLFF [28] | SRN [42] | NV [24] |

|  | PSNR↑\uparrow | SSIM↑\uparrow | LPIPS↓\downarrow |  |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  | Chair | Pedestal | Cube | Vase | Chair | Pedestal | Cube | Vase | Chair | Pedestal | Cube | Vase |
| DeepVoxels [41] | 33.4533.45 | 32.3532.35 | 28.4228.42 | 27.9927.99 | 0.990.99 | 0.970.97 | 0.970.97 | 0.960.96 | −- | −- | −- | −- |
| SRN [42] | 36.6736.67 | 35.9135.91 | 28.7428.74 | 31.4631.46 | 0.9820.982 | 0.9570.957 | 0.9440.944 | 0.9690.969 | 0.0930.093 | 0.0810.081 | 0.0740.074 | 0.0440.044 |
| NV [24] | 35.1535.15 | 36.4736.47 | 26.4826.48 | 20.3920.39 | 0.9800.980 | 0.9630.963 | 0.9160.916 | 0.8570.857 | 0.0960.096 | 0.0690.069 | 0.1130.113 | 0.1170.117 |
| LLFF [28] | 36.1136.11 | 35.8735.87 | 32.5832.58 | 32.9732.97 | 0.992\mathbf{0.992} | 0.9830.983 | 0.9830.983 | 0.9830.983 | 0.0510.051 | 0.0390.039 | 0.0640.064 | 0.0390.039 |
| Ours | 42.65\mathbf{42.65} | 41.44\mathbf{41.44} | 39.19\mathbf{39.19} | 37.32\mathbf{37.32} | 0.991{0.991} | 0.986\mathbf{0.986} | 0.996\mathbf{0.996} | 0.992\mathbf{0.992} | 0.047\mathbf{0.047} | 0.024\mathbf{0.024} | 0.006\mathbf{0.006} | 0.017\mathbf{0.017} |

| PSNR↑\uparrow |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
|  | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
| SRN [42] | 26.9626.96 | 17.1817.18 | 20.7320.73 | 26.8126.81 | 20.8520.85 | 18.0918.09 | 26.8526.85 | 20.6020.60 |
| NV [24] | 28.3328.33 | 22.5822.58 | 24.7924.79 | 30.7130.71 | 26.0826.08 | 24.2224.22 | 27.7827.78 | 23.9323.93 |
| LLFF [28] | 28.7228.72 | 21.1321.13 | 21.7921.79 | 31.4131.41 | 24.5424.54 | 20.7220.72 | 27.4827.48 | 23.2223.22 |
| Ours | 33.00\mathbf{33.00} | 25.01\mathbf{25.01} | 30.13\mathbf{30.13} | 36.18\mathbf{36.18} | 32.54\mathbf{32.54} | 29.62\mathbf{29.62} | 32.91\mathbf{32.91} | 28.65\mathbf{28.65} |

| SSIM↑\uparrow |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
|  | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
| SRN [42] | 0.9100.910 | 0.7660.766 | 0.8490.849 | 0.9230.923 | 0.8090.809 | 0.8080.808 | 0.9470.947 | 0.7570.757 |
| NV [24] | 0.9160.916 | 0.8730.873 | 0.9100.910 | 0.9440.944 | 0.8800.880 | 0.8880.888 | 0.9460.946 | 0.7840.784 |
| LLFF [28] | 0.9480.948 | 0.8900.890 | 0.8960.896 | 0.9650.965 | 0.9110.911 | 0.8900.890 | 0.9640.964 | 0.8230.823 |
| Ours | 0.967{\mathbf{0.967}} | 0.925{\mathbf{0.925}} | 0.964{\mathbf{0.964}} | 0.974{\mathbf{0.974}} | 0.961{\mathbf{0.961}} | 0.949{\mathbf{0.949}} | 0.980{\mathbf{0.980}} | 0.856{\mathbf{0.856}} |

| LPIPS↓\downarrow |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
|  | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
| SRN [42] | 0.1060.106 | 0.2670.267 | 0.1490.149 | 0.1000.100 | 0.2000.200 | 0.1740.174 | 0.0630.063 | 0.2990.299 |
| NV [24] | 0.1090.109 | 0.2140.214 | 0.1620.162 | 0.1090.109 | 0.1750.175 | 0.1300.130 | 0.1070.107 | 0.2760.276 |
| LLFF [28] | 0.0640.064 | 0.1260.126 | 0.1300.130 | 0.061\mathbf{0.061} | 0.1100.110 | 0.1170.117 | 0.0840.084 | 0.2180.218 |
| Ours | 0.046\mathbf{0.046} | 0.091\mathbf{0.091} | 0.044\mathbf{0.044} | 0.121{0.121} | 0.050\mathbf{0.050} | 0.063\mathbf{0.063} | 0.028\mathbf{0.028} | 0.206\mathbf{0.206} |

| PSNR↑\uparrow |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
|  | Room | Fern | Leaves | Fortress | Orchids | Flower | T-Rex | Horns |
| SRN [42] | 27.2927.29 | 21.3721.37 | 18.2418.24 | 26.6326.63 | 17.3717.37 | 24.6324.63 | 22.8722.87 | 24.3324.33 |
| LLFF [28] | 28.4228.42 | 22.8522.85 | 19.5219.52 | 29.4029.40 | 18.5218.52 | 25.4625.46 | 24.1524.15 | 24.7024.70 |
| Ours | 32.70\mathbf{32.70} | 25.17\mathbf{25.17} | 20.92\mathbf{20.92} | 31.16\mathbf{31.16} | 20.36\mathbf{20.36} | 27.40\mathbf{27.40} | 26.80\mathbf{26.80} | 27.45\mathbf{27.45} |

| SSIM↑\uparrow |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
|  | Room | Fern | Leaves | Fortress | Orchids | Flower | T-Rex | Horns |
| SRN [42] | 0.8830.883 | 0.6110.611 | 0.5200.520 | 0.6410.641 | 0.4490.449 | 0.7380.738 | 0.7610.761 | 0.7420.742 |
| LLFF [28] | 0.9320.932 | 0.7530.753 | 0.697{\mathbf{0.697}} | 0.8720.872 | 0.5880.588 | 0.844{\mathbf{0.844}} | 0.8570.857 | 0.840{\mathbf{0.840}} |
| Ours | 0.948{\mathbf{0.948}} | 0.792{\mathbf{0.792}} | 0.6900.690 | 0.881{\mathbf{0.881}} | 0.641{\mathbf{0.641}} | 0.8270.827 | 0.880{\mathbf{0.880}} | 0.8280.828 |

| LPIPS↓\downarrow |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
|  | Room | Fern | Leaves | Fortress | Orchids | Flower | T-Rex | Horns |
| SRN [42] | 0.2400.240 | 0.4590.459 | 0.4400.440 | 0.4530.453 | 0.4670.467 | 0.2880.288 | 0.2980.298 | 0.3760.376 |
| LLFF [28] | 0.155\mathbf{0.155} | 0.247\mathbf{0.247} | 0.216\mathbf{0.216} | 0.1730.173 | 0.313\mathbf{0.313} | 0.174\mathbf{0.174} | 0.222\mathbf{0.222} | 0.193\mathbf{0.193} |
| Ours | 0.1780.178 | 0.2800.280 | 0.3160.316 | 0.171\mathbf{0.171} | 0.3210.321 | 0.2190.219 | 0.2490.249 | 0.2680.268 |

|  |  | PSNR↑\uparrow |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
|  |  | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
| 1) | No PE, VD, H | 28.4428.44 | 23.1123.11 | 25.1725.17 | 32.2432.24 | 26.3826.38 | 24.6924.69 | 28.1628.16 | 25.1225.12 |
| 2) | No Pos. Encoding | 30.3330.33 | 24.5424.54 | 29.3229.32 | 33.1633.16 | 27.7527.75 | 27.7927.79 | 30.7630.76 | 26.5526.55 |
| 3) | No View Dependence | 30.0630.06 | 23.4123.41 | 25.9125.91 | 32.6532.65 | 29.9329.93 | 24.9624.96 | 28.6228.62 | 25.7225.72 |
| 4) | No Hierarchical | 31.3231.32 | 24.5524.55 | 29.2529.25 | 35.2435.24 | 31.4231.42 | 29.2229.22 | 31.7431.74 | 27.7327.73 |
| 5) | Far Fewer Images | 30.9230.92 | 22.6222.62 | 24.3924.39 | 32.7732.77 | 27.9727.97 | 26.5526.55 | 30.4730.47 | 26.5726.57 |
| 6) | Fewer Images | 32.1932.19 | 23.7023.70 | 27.4527.45 | 34.9134.91 | 31.5331.53 | 28.5428.54 | 32.3332.33 | 27.6727.67 |
| 7) | Fewer Frequencies | 32.1932.19 | 25.29{\mathbf{25.29}} | 30.73{\mathbf{30.73}} | 36.0636.06 | 30.7730.77 | 29.77{\mathbf{29.77}} | 31.6631.66 | 28.2628.26 |
| 8) | More Frequencies | 32.8732.87 | 24.6524.65 | 29.9229.92 | 35.7835.78 | 32.5032.50 | 29.5429.54 | 32.8632.86 | 28.3428.34 |
| 9) | Complete Model | 33.00{\mathbf{33.00}} | 25.0125.01 | 30.1330.13 | 36.18{\mathbf{36.18}} | 32.54{\mathbf{32.54}} | 29.6229.62 | 32.91{\mathbf{32.91}} | 28.65{\mathbf{28.65}} |

|  |  | SSIM↑\uparrow |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
|  |  | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
| 1) | No PE, VD, H | 0.9190.919 | 0.8960.896 | 0.9260.926 | 0.9550.955 | 0.8820.882 | 0.9050.905 | 0.9550.955 | 0.8100.810 |
| 2) | No Pos. Encoding | 0.9380.938 | 0.9180.918 | 0.9530.953 | 0.9560.956 | 0.9030.903 | 0.9330.933 | 0.9680.968 | 0.8240.824 |
| 3) | No View Dependence | 0.9480.948 | 0.9060.906 | 0.9380.938 | 0.9610.961 | 0.9470.947 | 0.9120.912 | 0.9620.962 | 0.8280.828 |
| 4) | No Hierarchical | 0.9510.951 | 0.9140.914 | 0.9560.956 | 0.9690.969 | 0.9510.951 | 0.9440.944 | 0.9730.973 | 0.8440.844 |
| 5) | Far Fewer Images | 0.9560.956 | 0.8950.895 | 0.9220.922 | 0.9660.966 | 0.9300.930 | 0.9250.925 | 0.9720.972 | 0.8320.832 |
| 6) | Fewer Images | 0.9630.963 | 0.9110.911 | 0.9480.948 | 0.9710.971 | 0.9570.957 | 0.9410.941 | 0.9790.979 | 0.8470.847 |
| 7) | Fewer Frequencies | 0.9590.959 | 0.928{\mathbf{0.928}} | 0.965{\mathbf{0.965}} | 0.9720.972 | 0.9470.947 | 0.952{\mathbf{0.952}} | 0.9730.973 | 0.8530.853 |
| 8) | More Frequencies | 0.9670.967 | 0.9210.921 | 0.9620.962 | 0.9730.973 | 0.9610.961 | 0.9480.948 | 0.980{\mathbf{0.980}} | 0.8530.853 |
| 9) | Complete Model | 0.967{\mathbf{0.967}} | 0.9250.925 | 0.9640.964 | 0.974{\mathbf{0.974}} | 0.961{\mathbf{0.961}} | 0.9490.949 | 0.9800.980 | 0.856{\mathbf{0.856}} |

|  |  | LPIPS↓\downarrow |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
|  |  | Chair | Drums | Ficus | Hotdog | Lego | Materials | Mic | Ship |
| 1) | No PE, VD, H | 0.0950.095 | 0.1680.168 | 0.0840.084 | 0.1040.104 | 0.1780.178 | 0.1110.111 | 0.0840.084 | 0.2610.261 |
| 2) | No Pos. Encoding | 0.0760.076 | 0.1040.104 | 0.0500.050 | 0.1240.124 | 0.1280.128 | 0.0790.079 | 0.0410.041 | 0.2610.261 |
| 3) | No View Dependence | 0.0750.075 | 0.1480.148 | 0.1130.113 | 0.1120.112 | 0.0880.088 | 0.1020.102 | 0.0730.073 | 0.2200.220 |
| 4) | No Hierarchical | 0.0650.065 | 0.1770.177 | 0.0560.056 | 0.1300.130 | 0.0720.072 | 0.0800.080 | 0.0390.039 | 0.2490.249 |
| 5) | Far Fewer Images | 0.0580.058 | 0.1730.173 | 0.0820.082 | 0.1230.123 | 0.0810.081 | 0.0790.079 | 0.0350.035 | 0.2290.229 |
| 6) | Fewer Images | 0.0510.051 | 0.1660.166 | 0.0570.057 | 0.1210.121 | 0.0550.055 | 0.0680.068 | 0.0290.029 | 0.2230.223 |
| 7) | Fewer Frequencies | 0.0550.055 | 0.1430.143 | 0.038{\mathbf{0.038}} | 0.087{\mathbf{0.087}} | 0.0710.071 | 0.060{\mathbf{0.060}} | 0.0290.029 | 0.2190.219 |
| 8) | More Frequencies | 0.0470.047 | 0.1580.158 | 0.0450.045 | 0.1160.116 | 0.0500.050 | 0.0640.064 | 0.027{\mathbf{0.027}} | 0.2610.261 |
| 9) | Complete Model | 0.046{\mathbf{0.046}} | 0.091{\mathbf{0.091}} | 0.0440.044 | 0.1210.121 | 0.050{\mathbf{0.050}} | 0.0630.063 | 0.0280.028 | 0.206{\mathbf{0.206}} |
