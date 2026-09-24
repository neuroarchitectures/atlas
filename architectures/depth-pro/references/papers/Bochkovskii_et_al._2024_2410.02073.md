# Depth Pro: Sharp Monocular Metric Depth in Less Than a Second

We present a foundation model for zero-shot metric monocular depth estimation. Our model, Depth Pro, synthesizes high-resolution depth maps with unparalleled sharpness and high-frequency details. The predictions are metric, with absolute scale, without relying on the availability of metadata such as camera intrinsics. And the model is fast, producing a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. These characteristics are enabled by a number of technical contributions, including an efficient multi-scale vision transformer for dense prediction, a training protocol that combines real and synthetic datasets to achieve high metric accuracy alongside fine boundary tracing, dedicated evaluation metrics for boundary accuracy in estimated depth maps, and state-of-the-art focal length estimation from a single image. Extensive experiments analyze specific design choices and demonstrate that Depth Pro outperforms prior work along multiple dimensions. We release code & weights at https://github.com/apple/ml-depth-pro

## 1 Introduction

| Input |  |  |  |
|---|---|---|---|
| Depth Pro |  |  |  |
| Marigold |  |  |  |
| Depth Anything v2 |  |  |  |
| Metric3D v2 |  |  |  |

Zero-shot monocular depth estimation underpins a growing variety of applications, such as advanced image editing, view synthesis, and conditional image generation. Inspired by MiDaS (Ranftl et al., 2022) and many follow-up works (Ranftl et al., 2021; Ke et al., 2024; Yang et al., 2024a; Piccinelli et al., 2024; Hu et al., 2024), applications increasingly leverage the ability to derive a dense pixelwise depth map for any image.

Our work is motivated in particular by novel view synthesis from a single image, an exciting application that has been transformed by advances in monocular depth estimation (Hedman et al., 2017; Shih et al., 2020; Jampani et al., 2021; Khan et al., 2023). Applications such as view synthesis imply a number of desiderata for monocular depth estimation. First, the depth estimator should work zero-shot on any image, not restricted to a specific domain (Ranftl et al., 2022; Yang et al., 2024a). Furthermore, the method should ideally produce metric depth maps in this zero-shot regime, to accurately reproduce object shapes, scene layouts, and absolute scales (Guizilini et al., 2023; Hu et al., 2024). For the broadest applicability ‘in the wild’, the method should produce metric depth maps with absolute scale even if no camera intrinsics (such as focal length) are provided with the image (Piccinelli et al., 2024). This enables view synthesis scenarios such as “Synthesize a view of this scene from 63 mm away” for essentially arbitrary single images (Dodgson, 2004).

Second, for the most compelling results, the monocular depth estimator should operate at high resolution and produce fine-grained depth maps that closely adhere to image details such as hair, fur, and other fine structures (Miangoleh et al., 2021; Ke et al., 2024; Li et al., 2024a). One benefit of producing sharp depth maps that accurately trace intricate details is the elimination of “flying pixels”, which can degrade image quality in applications such as view synthesis (Jampani et al., 2021).

Third, for many interactive application scenarios, the depth estimator should operate at low latency, processing a high-resolution image in less than a second to support interactive view synthesis “queries” on demand. Low latency is a common characteristic of methods that reduce zero-shot monocular depth estimation to a single forward pass through a neural network (Ranftl et al., 2021; Yang et al., 2024a; Piccinelli et al., 2024), but it is not always shared by methods that employ more computationally demanding machinery at test time (Ke et al., 2024; Li et al., 2024a).

In this work, we present a foundation model for zero-shot metric monocular depth estimation that meets all of these desiderata. Our model, Depth Pro, produces metric depth maps with absolute scale on arbitrary images ‘in the wild’ without requiring metadata such as camera intrinsics. It operates at high resolution, producing 2.25-megapixel depth maps (with a native output resolution of 1536×15361536\times 1536 before optional further upsampling) in 0.3 seconds on a V100 GPU. Fig. 1 shows some representative results. Depth Pro dramatically outperforms all prior work in sharp delineation of object boundaries, including fine structures such as hair, fur, and vegetation. As shown in Fig. 2, Depth Pro offers unparalleled boundary tracing, outperforming all prior work by a multiplicative factor in boundary recall. (See Sec. 4 for additional detail.) Compared to the prior state of the art in boundary accuracy (Ke et al., 2024; Li et al., 2024a), Depth Pro is one to two orders of magnitude faster, yields much more accurate boundaries, and provides metric depth maps with absolute scale.

Depth Pro is enabled by a number of technical contributions. First, we derive a new set of metrics that enable leveraging highly accurate matting datasets for quantifying the accuracy of boundary tracing in evaluating monocular depth maps. We analyze the effect of typical output resolutions and find that a high resolution is necessary but not sufficient to improve boundary accuracy. Second, we design an efficient multi-scale ViT-based architecture for capturing the global image context while also adhering to fine structures at high resolution. Third, we devise a set of loss functions and a training curriculum that promote sharp depth estimates while training on real-world datasets that provide coarse and inaccurate supervision around boundaries, along with synthetic datasets that offer accurate pixelwise ground truth but limited realism. Fourth, we contribute zero-shot focal length estimation from a single image that dramatically outperforms the prior state of the art.

## 2 Related work

Early work on monocular depth estimation focused on training on individual datasets recorded with a single camera (Saxena et al., 2009; Eigen et al., 2014; Eigen & Fergus, 2015). Although this setup directly enabled metric depth predictions, it was limited to single datasets and narrow domains.

Zero-shot depth estimation. MegaDepth (Li & Snavely, 2018) demonstrated that training on a diverse dataset allows generalizing monocular depth prediction beyond a specific domain. MiDaS (Ranftl et al., 2022) advanced this idea by training on a large mix of diverse datasets with a scale-and-shift-invariant loss. Follow-up works applied this recipe to transformer-based architectures (Ranftl et al., 2021; Birkl et al., 2023) and further expanded the set of feasible datasets through self-supervision (Spencer et al., 2023; Yang et al., 2024a). A line of work uses self-supervision to learn from unlabeled image and video data (Petrovai & Nedevschi, 2022; Yang et al., 2024a). A number of recent approaches (Ke et al., 2024; Gui et al., 2025) harness diffusion models to synthesize relative depth maps. Although some of these methods demonstrated excellent generalization, their predictions are ambiguous in scale and shift, which precludes downstream applications that require accurate shapes, sizes, or distances.

Zero-shot metric depth. A line of work sought to improve metric depth prediction through a global distribution of depth values (Fu et al., 2018; Bhat et al., 2021; Bhat et al., 2022; Li et al., 2024b) and further conditioning on scene type (Bhat et al., 2023). A different approach directly takes into account camera intrinsics. Cam-Convs (Fácil et al., 2019) conditioned convolutions on the camera intrinsics. LeReS (Yin et al., 2021) trains a separate network for undistorting point clouds to recover scale and shift, Metric3D (Yin et al., 2023) scales images or depth maps to a canonical space and remaps estimated depth given the focal length, and ZeroDepth (Guizilini et al., 2023) learns camera-specific embedddings in a variational framework. DMD (Saxena et al., 2023) conditions a diffusion model on the field of view. Metric3D v2 (Hu et al., 2024) leverages surface normals as an auxilliary output to improve metric depth. All of these methods require the camera intrinsics to be known and accurate. More recent works attempt to reason about unknown camera intrinsics either through a separate network (Spencer et al., 2024) or by predicting a camera embedding for conditioning its depth predictions in a spherical space (Piccinelli et al., 2024). Akin to these recent approaches, our method does not require the focal length to be provided as input. We propose to directly estimate the field of view from intermediate features of the depth prediction network, and show that this substantially outperforms the prior state of the art in the task of cross-domain focal length estimation.

Sharp occluding contours. SharpNet (Ramamonjisoa & Lepetit, 2019) incorporates normals and occluding contour constraints, but requires additional contour and normal supervision during training. BoostingDepth (Miangoleh et al., 2021) obtains detailed predictions from a low-resolution network by applying it independently to image patches. Since the patches lack global context, BoostingDepth fuses them through a sophisticated multi-step pipeline. PatchFusion (Li et al., 2024a) refines this concept through image-adaptive patch sampling and tailored modules that enable end-to-end training. A recent line of work (Gui et al., 2025; Ke et al., 2024) leverages diffusion priors to enhance the sharpness of occlusion boundaries. These approaches predominantly focus on predicting relative (rather than metric) depth. We propose a simpler architecture without task-specific modules or diffusion priors and demonstrate that even sharper and more accurate results can be obtained while producing metric depth maps and reducing runtime by more than two orders of magnitude.

Guided depth super-resolution uses the input image to upsample low-resolution depth predictions (Metzger et al., 2023; Zhong et al., 2023). SMDNet (Tosi et al., 2021) predicts bimodal mixture densities to sharpen occluding contours. And Ramamonjisoa et al. (Ramamonjisoa et al., 2020) introduce a module for learning to sharpen depth boundaries of a pretrained network. These works are orthogonal to ours and could be applied to further upsample our high-resolution predictions.

To evaluate boundary tracing in predicted depth maps, Koch et al. (2018) introduce the iBims dataset with manual annotations of occluding contours and corresponding metrics. The need for manual annotation and highly accurate depth ground truth constrain the benchmark to a small set of indoor scenes. We contribute metrics based on segmentation and matting datasets that provide a complementary view by enabling evaluation on complex, dynamic environments or scenes with exceedingly fine detail for which ground-truth depth is impossible to obtain.

Multi-scale vision transformers. Vision transformers (ViTs) have emerged as the dominant general-purpose architecture for perception tasks but operate at low resolution (Dosovitskiy et al., 2021). The computational complexity of their attention layers prohibits naïvely scaling them to higher resolutions and several works proposed alternatives (Zhu et al., 2021; Liu et al., 2021; Li et al., 2022c; Chu et al., 2021; Liu et al., 2022a; Liu et al., 2023; Cai et al., 2023; Jaegle et al., 2022). Another line of work modified the ViT architecture to produce a hierarchy of features (Fan et al., 2021; Xie et al., 2021; Yuan et al., 2021; Ranftl et al., 2021; Chen et al., 2021; Lee et al., 2022).

Rather than modifying the ViT architecture, which requires computationally expensive retraining, we propose an architecture that applies a plain ViT backbone at multiple scales and fuses predictions into a single high-resolution output. This design benefits from ongoing improvements in ViT pretraining, as new variants can be easily swapped in (Oquab et al., 2024; Peng et al., 2022b; Sun et al., 2023).

Pretrained vision transformers have been adapted for semantic segmenation and object detection. ViT-Adapter (Chen et al., 2023) and ViT-CoMer (Xia et al., 2024) supplement a pretrained ViT with a convolutional network for dense prediction, whereas ViT-Det (Li et al., 2022b) builds a feature pyramid on top of a pretrained ViT. Distinct from these, we fuse features from the ViT applied at multiple scales to learn global context together with local detail.

## 3 Method

### 3.1 Network

The key idea behind our network is to apply plain ViT encoders (Dosovitskiy et al., 2021) on patches extracted at multiple scales and fuse their predictions into a single high-resolution dense depth prediction in an end-to-end trainable model (see Fig. 3). As the patch encoder shares weights across all scales, it may intuitively learn a scale-invariant representation. The image encoder anchors the patch predictions in a global context. It is applied to the whole input image, downsampled to the base input resolution of the chosen encoder backbone (in our case 384×\times384).

The whole network operates at a fixed resolution of 1536×\times1536, which was chosen as a multiple of the ViT’s 384×\times384. This guarantees a sufficiently large receptive field and constant runtimes for any image while preventing out-of-memory errors (which we repeatedly observed for variable-resolution approaches on large images). Confirming this design choice, our results in Sec. 4 and Tab. 5 demonstrate that Depth Pro is consistently orders of magnitude faster than variable-resolution approaches while being more accurate and producing sharper boundaries. A key benefit of assembling our architecture from plain ViT encoders over custom encoders is the abundance of pretrained ViT-based backbones that can be harnessed (Oquab et al., 2024; Peng et al., 2022b; Sun et al., 2023). We evaluate several pretrained backbones and compare our architecture to other high-resolution architectures in the appendices (Tab. 9 and Sec. B.3).

After initial downsampling to 1536×\times1536, the input image is split into patches of 384×384384\times 384 at each scale. For the two finest scales, we let patches overlap to avoid seams, which yields 25 and 9 patches, respectively. In total, we extract 35 patches, concatenate them along the batch dimension to allow efficient batch processing, and feed them to the patch encoder. This yields a feature tensor at resolution 24×2424\times 24 per input patch (Features 3 – 6 in Fig. 3). At the finest scale we further extract intermediate features (Features 1 & 2 in Fig. 3) to capture finer-grained details, yielding additional 25+25=5025+25=50 feature patches. We merge the feature patches into maps (detailed in Sec. C.1), which are fed into the decoder module, which resembles the DPT decoder (Ranftl et al., 2021).

In addition to sharing representations across scales, the patch-based application of the encoder network allows trivial parallelization as patches can be processed independently. Another source of computational efficiency comes from the lower computational complexity of patch-based processing in comparison to scaling up the ViT to higher resolutions. The reason is multi-head self-attention (Vaswani et al., 2017), whose computational complexity scales quadratically with the number of input pixels, and thus quartically in image dimension.

### 3.2 Sharp monocular depth estimation

For each input image II, our network ff predicts a canonical inverse depth image C=f⁡(I)C=f(I). To obtain a dense metric depth map DmD_{m}, we scale by the horizontal field of view, represented by the focal length f𝑝𝑥f_{\mathit{px}} and the width ww (Yin et al., 2023): Dm=f𝑝𝑥w​C.D_{m}=\tfrac{f_{\mathit{px}}}{wC}.

We train with several objectives, all based on canonical inverse depth, because this prioritizes areas close to the camera over farther areas or the whole scene, and thus supports visual quality in applications such as novel view synthesis (see Sec. B.4). Let C^\hat{C} be the ground-truth canonical inverse depth. For all metric datasets we compute the mean absolute error (ℒ𝑀𝐴𝐸\mathcal{L}_{\mathit{MAE}}, Eq. 1) per pixel ii, and discard pixels with an error in the top 20%20\% per image for real-world (as opposed to synthetic) datasets:

|  | ℒ𝑀𝐴𝐸​(C^,C)=1N​∑iN|C^i−Ci|.\displaystyle\mathcal{L}_{\mathit{MAE}}(\hat{C},C)=\frac{1}{N}\sum^{N}_{i}|\hat{C}_{i}-C_{i}|. |  | (1) |
|---|---|---|---|

For all non-metric datasets (i.e., those without reliable camera intrinsics or inconsistent scale), we normalize predictions and ground truth via the mean absolute deviation from the median (Ranftl et al., 2022) before applying a loss. We further compute errors on the first and second derivatives of (canoncial) inverse depth maps at multiple scales. Let ∇∗\nabla_{*} indicate a spatial derivative operator ∗*, such as Scharr (S) (Scharr et al., 1997) or Laplace (L), and pp the error norm. We define the multi-scale derivative loss over MM scales as

|  | ℒ∗,p,M​(C,C^)=1M​∑jM1Nj​∑iNj|∇∗Cij−∇∗C^ij|p,\displaystyle\mathcal{L}_{*,p,M}(C,\hat{C})=\frac{1}{M}\sum^{M}_{j}\frac{1}{N_{j}}\sum_{i}^{N_{j}}|\nabla_{*}C^{j}_{i}-\nabla_{*}\hat{C}^{j}_{i}|^{p}, |  | (2) |
|---|---|---|---|

where the scales jj are computed by blurring and downsampling the inverse depth maps by a factor of 2 per scale. As shorthands we define the Mean Absolute Gradient Error ℒ𝑀𝐴𝐺𝐸=ℒS,1,6\mathcal{L}_{\mathit{MAGE}}=\mathcal{L}_{S,1,6}, the Mean Absolute Laplace Error ℒ𝑀𝐴𝐿𝐸=ℒL,1,6\mathcal{L}_{\mathit{MALE}}=\mathcal{L}_{L,1,6}, and the Mean Squared Gradient Error ℒ𝑀𝑆𝐺𝐸=ℒS,2,6\mathcal{L}_{\mathit{MSGE}}=\mathcal{L}_{S,2,6}.

We propose a training curriculum motivated by the following observations. First, training on a large mix of real-world and synthetic datasets improves generalization as measured by zero-shot accuracy (Ranftl et al., 2022; Ranftl et al., 2021; Yang et al., 2024a; Hu et al., 2024). Second, synthetic datasets provide pixel-accurate ground truth, whereas real-world datasets often contain missing areas, mismatched depth, or false measurements on object boundaries. Third, predictions get sharper over the course of training.

Based on these observations, we design a two-stage training curriculum. In the first stage, we aim to learn robust features that allow the network to generalize across domains. To that end, we train on a mix of all labeled training sets. Specifically, we minimize ℒ𝑀𝐴𝐸\mathcal{L}_{\mathit{MAE}} on metric datasets and its normalized version on non-metric datasets. ℒ𝑀𝐴𝐸\mathcal{L}_{\mathit{MAE}} is chosen for its robustness in handling potentially corrupted real-world ground truth. To steer the network towards sharp boundaries, we aim to also supervise on gradients of the predictions. Done naïvely, however, this can hinder optimization and slow down convergence. We found that a scale-and-shift-invariant loss on gradients, applied only to synthetic datasets, worked best. Controlled experiments are reported in Sec. B.5 of the appendices.

The second stage of training is designed to sharpen boundaries and reveal fine details in the predicted depth maps. To minimize the effect of inaccurate ground truth, at this stage we only train on synthetic datasets that provide high-quality pixel-accurate ground truth. (Note that this inverts the common practice of first training on synthetic data and then fine-tuning on real data (Gaidon et al., 2016; Gómez et al., 2023; Sun et al., 2021).) Specifically, we again minimize the ℒ𝑀𝐴𝐸\mathcal{L}_{\mathit{MAE}} and supplement it with a selection of losses on the first- and second-order derivatives: ℒ𝑀𝐴𝐺𝐸\mathcal{L}_{\mathit{MAGE}}, ℒ𝑀𝐴𝐿𝐸\mathcal{L}_{\mathit{MALE}}, and ℒ𝑀𝑆𝐺𝐸\mathcal{L}_{\mathit{MSGE}}. We provide a detailed specification of the loss functions that are applied at each stage in the appendices.

Applications such as novel view synthesis require depth maps to adhere to object boundaries. This is particularly challenging for thin structures. Misaligned or blurry boundaries can make objects appear distorted or split into parts. Common benchmarks for monocular depth prediction rarely take boundary sharpness into account. This may be attributed in part to the lack of diverse and realistic datasets with precise pixel-accurate ground-truth depth. To address this shortcoming, we propose a new set of metrics specifically for the evaluation of depth boundaries. Our key observation is that we can leverage existing high-quality annotations for matting, saliency, or segmentation as ground truth for depth boundaries. We treat annotations for these tasks as binary maps, which define a foreground/background relationship between an object and its environment. (This relationship may not hold in every case, especially for segmentation masks. However, we can easily discard such problematic cases through manual inspection. It is much easier to filter out a segmentation mask than to annotate it.) To ensure that the relationship holds, we only consider pixels around edges in the binary map.

We first define the metrics for depth maps and later derive the formulation for binary segmentation masks. Motivated by the ranking loss (Chen et al., 2016), we use the pairwise depth ratio of neighboring pixels to define a foreground/background relationship. Let i,ji,j be the locations of two neighboring pixels. We then define an occluding contour cdc_{d} derived from a depth map dd as cd(i,j)=[d⁡(j)d⁡(i)>(1+t100)]c_{d}(i,j)=\left[\tfrac{d(j)}{d(i)}>(1+\tfrac{t}{100})\right], where [⋅][\cdot] is the Iverson bracket. Intuitively, this indicates the presence of an occluding contour between pixels ii and jj if their corresponding depth differs by more than t%t\%. For all pairs of neighboring pixels, we can then compute the precision (PP) and recall (RR) as

|  | P​(t)=∑i,j∈N⁡(i)cd​(i,j)∧cd^​(i,j)∑i,j∈N⁡(i)cd​(i,j)​ and R​(t)=∑i,j∈N⁡(i)cd​(i,j)∧cd^​(i,j)∑i,j∈N⁡(i)cd^​(i,j).\text{P}(t)=\frac{\sum_{i,j\in N(i)}c_{d}(i,j)\wedge c_{\hat{d}}(i,j)}{\sum_{i,j\in N(i)}c_{d}(i,j)}\text{ and }\text{R}(t)=\frac{\sum_{i,j\in N(i)}c_{d}(i,j)\wedge c_{\hat{d}}(i,j)}{\sum_{i,j\in N(i)}c_{\hat{d}}(i,j)}. |  | (3) |
|---|---|---|---|

Note that both PP and RR are scale-invariant. In our experiments, we report the F1 score. To account for multiple relative depth ratios, we further perform a weighted averaging of the F1 values with thresholds that range linearly from tm​i​n=5t_{min}=5 to tm​a​x=25t_{max}=25, with stronger weights towards high threshold values. Compared to other edge-based metrics (such as the edge accuracy and completion from iBims (Koch et al., 2018)), our metric does not require any manual edge annotation, but simply pixelwise ground truth, which is easily obtained for synthetic datasets.

Similarly, we can also identify occluding contours from binary label maps that can be derived from real-world segmentation, saliency, and matting datasets. Given a binary mask bb over the image, we define the presence of an occluding contour cbc_{b} between pixels i,ji,j as cb​(i,j)=b⁡(i)∧¬b⁡(j)c_{b}(i,j)=b(i)\land\lnot b(j). With this definition at hand, we compute the recall R​(t)\text{R}(t) by replacing the occluding contours from depth maps in Eq. 3 with those from binary maps. Since the binary maps commonly label whole objects, we cannot obtain ground-truth occluding contours that do not align with object silhouettes. Thus the boundary annotation is incomplete – some but not all occluding contours are identified by this procedure. Therefore we can only compute the recall but not the precision for binary maps.

To penalize blurry edges, we suppress non-maximum values of cd^c_{\hat{d}} within the valid bounds of cd^​(i,j)c_{\hat{d}}(i,j) connected components. We report additional experiments and qualitative results in Sec. A.4.

### 3.3 Focal length estimation

To handle images that may have inaccurate or missing EXIF metadata, we supplement our network with a focal length estimation head. A small convolutional head ingests frozen features from the depth estimation network and task-specific features from a separate ViT image encoder to predict the horizontal angular field-of-view. We use ℒ2\mathcal{L}_{2} as the training loss. We train the focal length head and the ViT encoder after the depth estimation training. Separating the focal length training has several benefits over joint training with the depth network. It avoids the necessity of balancing the depth and focal length training objectives. It also allows training the focal length head on a different set of datasets, excluding some narrow-domain single-camera datasets that are used in training the depth estimation network, and adding large-scale image datasets that provide focal length supervision but no depth supervision. Further details are provided in Sec. C.2.

## 4 Experiments

This section summarizes the key results. Additional details and experiments are reported in the appendices, including details on datasets, hyperparameters, experimental protocols, and the comparison of runtimes, which is summarized in Fig. 2. The appendices also report controlled experiments, including controlled studies on network architectures, training objectives, and training curricula.

Here we summarize a number of key comparisons of Depth Pro to state-of-the-art metric monocular depth estimation systems. One challenge in conducting such a comparison is that many leading recent systems are trained on bespoke combinations of datasets. Some systems use proprietary datasets that are not publicly available, and some use datasets that are only available under restrictive licenses. Some recent systems also train on unlabeled datasets or incorporate pretrained models (e.g., diffusion models) that were trained on additional massive datasets. This rules out the possibility of a comparison that controls for training data (e.g., only comparing to systems that use the same datasets we do). At this stage of this research area, the only feasible comparison to other leading cross-domain monocular depth estimation models is on a full system-to-system basis. Fully trained models (each trained on a large, partially overlapping and partially distinct collection of datasets) are compared to each other zero-shot on datasets that none of the compared systems trained on.

Zero-shot metric depth. We evaluate our method’s ability to predict zero-shot metric depth and compare against the state of the art in Tab. 1. Our baselines include Depth Anything (Yang et al., 2024a), Metric3D (Yin et al., 2023), PatchFusion (Li et al., 2024a), UniDepth (Piccinelli et al., 2024), ZeroDepth (Guizilini et al., 2023) and ZoeDepth (Bhat et al., 2023). We also report results for the very recent Depth Anything v2 (Yang et al., 2024b) and Metric3D v2 (Hu et al., 2024).

As an overall summary measure of metric depth accuracy, Tab. 1 uses the δ1\delta_{1} metric (Ladicky et al., 2014), which is commonly used for this purpose (Yin et al., 2023; Yang et al., 2024a; Piccinelli et al., 2024). It is defined as the percentage of inlier pixels, for which the predicted and ground-truth depths are within 25% of each other. We picked this metric for its robustness, with the strictest threshold found in the literature (25%25\%).

| Method | Booster | ETH3D | Middlebury | NuScenes | Sintel | Sun-RGBD | Avg. Rank↓\downarrow |
|---|---|---|---|---|---|---|---|
| DepthAnything (Yang et al., 2024a) | 52.3 | 9.3 | 39.3 | 35.4 | 6.9 | 85.0 | 5.7 |
| DepthAnything v2 (Yang et al., 2024b) | 59.5 | 36.3 | 37.2 | 17.7 | 5.9 | 72.4 | 5.8 |
| Metric3D (Yin et al., 2023) | 4.7 | 34.2 | 13.6 | 64.4 | 17.3 | 16.9 | 5.8 |
| Metric3D v2 (Hu et al., 2024) | 39.4 | 87.7 | 29.9 | 82.6 | 38.3 | 75.6 | 3.7 |
| PatchFusion (Li et al., 2024a) | 22.622.6 | 51.8 | 49.9 | 20.420.4 | 14.014.0 | 53.653.6 | 5.25.2 |
| UniDepth (Piccinelli et al., 2024) | 27.627.6 | 25.325.3 | 31.931.9 | 83.6 | 16.516.5 | 95.8 | 4.2 |
| ZeroDepth (Guizilini et al., 2023) | OOM | OOM | 46.546.5 | 64.364.3 | 12.912.9 | OOM | 4.64.6 |
| ZoeDepth (Bhat et al., 2023) | 21.621.6 | 34.234.2 | 53.8 | 28.128.1 | 7.87.8 | 85.7 | 5.35.3 |
| Depth Pro (Ours) | 46.6 | 41.5 | 60.5 | 49.149.1 | 40.0 | 89.0 | 2.5 |

Corresponding tables for additional metrics can be found in Sec. A.2 of the appendices, including 𝐴𝑏𝑠𝑅𝑒𝑙\mathit{AbsRel} (Ladicky et al., 2014), 𝐿𝑜𝑔10\mathit{Log}_{10}, δ2\delta_{2} and δ3\delta_{3} scores, as well as point-cloud metrics (Spencer et al., 2022). Tab. 1 also reports the average rank of each method across datasets, a common way to summarize cross-dataset performance (Ranftl et al., 2022).

We report results on Booster (Ramirez et al., 2024), Middlebury (Scharstein et al., 2014), Sun-RGBD (Song et al., 2015), ETH3D (Schöps et al., 2017), nuScenes (Caesar et al., 2020), and Sintel (Butler et al., 2012), because, to our knowledge, they were never used in training any of the evaluated systems. Despite our best efforts, we were not able to run ZeroDepth on Booster, Middlebury, or Sun-RGBD as it consistently ran out of memory due to the high image resolutions. More details on our evaluation setup can be found in Sec. C of the appendix.

The results in Tab. 1 confirm the findings of Piccinelli et al. (2024), who observed considerable domain bias in some of the leading metric depth estimation models. Notably, Depth Anything v1 & v2 focus on relative depth estimation; for metric depth, they provide different models for different domains, fine-tuned either for indoor or for outdoor scenes. Metric3D v1 & v2 provide domain-invariant models, but their performance depends strongly on careful selection of the crop size at test time, which is performed per domain in their experiments and thus violates the zero-shot premise. We tried setting the crop size automatically based on the aspect ratio of the image, but this substantially degraded the performance of Metric3D; for this reason, we use the recommended non-zero-shot protocol, with the recommended per-domain crop sizes. Since domain-specific models and crop sizes violate the strict zero-shot premise we (and other baselines) operate under, we mark the Depth Anything and Metric3D results in gray in Tab. 1.

We find that Depth Pro demonstrates the strongest generalization by consistently scoring among the top approaches per dataset and obtaining the best average rank across all datasets.

|  | Method | Sintel F1↑\uparrow | Spring F1↑\uparrow | iBims F1↑\uparrow | AM R↑\uparrow | P3M R↑\uparrow | DIS R↑\uparrow |
|---|---|---|---|---|---|---|---|
| Absolute | DPT (Ranftl et al., 2021) | 0.181 | 0.029 | 0.113 | 0.055 | 0.075 | 0.018 |
| Metric3D (Yin et al., 2023) | 0.037 | 0.000 | 0.055 | 0.003 | 0.003 | 0.001 |  |
| Metric3D v2 (Hu et al., 2024) | 0.321 | 0.024 | 0.096 | 0.024 | 0.013 | 0.006 |  |
| ZoeDepth (Bhat et al., 2023) | 0.027 | 0.001 | 0.035 | 0.008 | 0.004 | 0.002 |  |
| PatchFusion (Li et al., 2024a) | 0.312 | 0.032 | 0.134 | 0.061 | 0.109 | 0.068 |  |
| UniDepth (Piccinelli et al., 2024) | 0.316 | 0.000 | 0.039 | 0.001 | 0.003 | 0.000 |  |
| Rel. | DepthAnything (Yang et al., 2024a) | 0.261 | 0.045 | 0.127 | 0.058 | 0.094 | 0.023 |
| DepthAnything v2 (Yang et al., 2024b) | 0.228 | 0.056 | 0.111 | 0.107 | 0.131 | 0.056 |  |
|  | Marigold (Ke et al., 2024) | 0.068 | 0.032 | 0.149 | 0.064 | 0.101 | 0.049 |
|  | Depth Pro (Ours) | 0.409 | 0.079 | 0.176 | 0.173 | 0.168 | 0.077 |

|  |  |  |  |  |  |
|---|---|---|---|---|---|
| Image | Alpha Matte | Depth Pro (Ours) | DepthAnything v2 | PatchFusion | Marigold |

Zero-shot boundaries. Tab. 2 summarizes the evaluation of boundary accuracy for Depth Pro and several baselines. This evaluation is conducted in a zero-shot setting: models are only evaluated on datasets that were not seen during training. Since our boundary metrics are scale-invariant, our baselines here also include methods that only predict relative (rather than absolute metric) depth. Our absolute baselines include Metric3D (Yin et al., 2023), Metric3D v2 (‘giant’ model) (Hu et al., 2024), PatchFusion (Li et al., 2024a), UniDepth (Piccinelli et al., 2024), and ZoeDepth (Bhat et al., 2023). We also report results for the relative variants of Depth Anything v1 & v2 (Yang et al., 2024a; Yang et al., 2024b) because they yield sharper boundaries than their metric counterparts. Lastly, we include Marigold (Ke et al., 2024), a recent diffusion-based relative depth model that became popular due to its high-fidelity predictions. We use the boundary metrics introduced in Sec. 3.2, and report the average boundary F1 score for datasets with ground-truth depth, and boundary recall (RR) for datasets with matting or segmentation annotations. For image matting datasets, a pixel is marked as occluding when the value of the alpha matte is above 0.10.1.

The datasets include Sintel (Butler et al., 2012) and Spring (Mehl et al., 2023), which are synthetic. We also include the iBims dataset (Koch et al., 2018) which is often used specifically to evaluate depth boundaries, despite having low resolution. We refer to the appendices for a full slate of iBims-specific metrics. To evaluate high-frequency structures encountered in natural images (such as hair or fur), we use AM-2k (Li et al., 2022a) and P3M-10k (Li et al., 2021), which are high-resolution image matting datasets that were used to evaluate image matting models (Li et al., 2023). Additionally, we further report results on the DIS-5k (Qin et al., 2022) image segmentation dataset. This is an object segmentation dataset that provides highly accurate binary masks across diverse images. We manually remove samples in which the segmented object is occluded by foreground objects. Fig. 2 visually summarizes the boundary recall metric on the AM-2k dataset, as a function of runtime.

We find that Depth Pro produces more accurate boundaries than all baselines on all datasets, by a significant margin. As can be observed in Fig. 1, in the images in Tab. 2, and the additional results in Sec. A, the competitive metric accuracy of Metric3D v2 and Depth Anything v2 does not imply sharp boundaries. Depth Pro has a consistently higher recall for thin structures like hair and fur and yields sharper boundaries. This is also true in comparison to the diffusion-based Marigold, as well as PatchFusion, which operates at variable resolution. Note that Depth Pro is orders of magnitude faster than Marigold and PatchFusion (see Fig. 2 & Tab. 5). Fig. 4 demonstrates the benefits of sharp boundary prediction for novel view synthesis from a single image.

| Input |  |  |  |
|---|---|---|---|
| Depth Pro |  |  |  |
| Depth Anything v2 |  |  |  |
| Marigold |  |  |  |
| Metric3D v2 |  |  |  |

Focal length estimation. Prior work (Piccinelli et al., 2024; Kocabas et al., 2021; Baradad & Torralba, 2020) does not provide comprehensive systematic evaluations of focal length estimators on in-the-wild images. To address this, we curated a zero-shot test dataset. To this end, we selected diverse datasets with intact EXIF data, enabling reliable assessment of focal length estimation accuracy. FiveK (Bychkovsky et al., 2011), DDDP (Abuolaim & Brown, 2020), and RAISE (Dang-Nguyen et al., 2015) contribute professional-grade photographs taken with SLR cameras. SPAQ (Fang et al., 2020) provides casual photographs from mobile phones. PPR10K (Liang et al., 2021) provides high-quality portrait images. Finally, ZOOM (Zhang et al., 2019) includes sets of scenes captured at various optical zoom levels.

|  | DDDP | FiveK | PPR10K | RAISE | SPAQ | ZOOM |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  | δ25%\delta_{25\%} | δ50%\delta_{50\%} | δ25%\delta_{25\%} | δ50%\delta_{50\%} | δ25%\delta_{25\%} | δ50%\delta_{50\%} | δ25%\delta_{25\%} | δ50%\delta_{50\%} | δ25%\delta_{25\%} | δ50%\delta_{50\%} | δ25%\delta_{25\%} | δ50%\delta_{50\%} |
| UniDepth (Piccinelli et al., 2024) | 6.8 | 40.3 | 24.8 | 56.2 | 13.8 | 44.2 | 35.4 | 74.8 | 44.2 | 77.4 | 20.4 | 45.4 |
| SPEC (Kocabas et al., 2021) | 14.6 | 46.3 | 30.2 | 56.6 | 34.6 | 67.0 | 49.2 | 78.6 | 50.0 | 82.2 | 23.2 | 43.6 |
| im2pcl (Baradad & Torralba, 2020) | 7.3 | 29.6 | 28.0 | 60.0 | 24.2 | 61.4 | 51.8 | 75.2 | 26.6 | 55.0 | 22.4 | 42.8 |
| Depth Pro (Ours) | 66.9 | 85.8 | 74.2 | 92.4 | 64.6 | 88.8 | 84.2 | 96.4 | 68.4 | 85.2 | 69.8 | 91.6 |

Tab. 3 compares Depth Pro against state-of-the-art focal length estimators and shows the percentage of images with relative estimation error under 25% and 50%, respectively. Depth Pro is the most accurate across all datasets. For example, on PPR10K, a dataset of human portraits, our method leads with 64.6% of the images having a focal length error below 25%, while the second-best method, SPEC, only achieves 34.6% on this metric. We attribute this superior performance to our network design and training protocol, which decouple training of the focal length estimator from the depth network, enabling us to use different training sets for these two tasks. Further controlled experiments are reported in Sec. B.7 of the appendices.

## 5 Conclusion & limitations

Depth Pro produces high-resolution metric depth maps with high-frequency detail at sub-second runtimes. Our model achieves state-of-the-art zero-shot metric depth estimation accuracy without requiring metadata such as camera intrinsics, and traces out occlusion boundaries in unprecedented detail, facilitating applications such as novel view synthesis from single images ‘in the wild’. While Depth Pro outperforms prior work along multiple dimensions, it is not without limitations. For example, the model is limited in dealing with translucent surfaces and volumetric scattering, where the definition of pixelwise depth is ill-posed and ambiguous.

## Supplemental material

In Section A, we provide additional results and experiments. Sec. A.1 presents further qualitative comparisons to baselines, Sec.A.2 presents a more detailed zero-shot evaluation, Sec.A.3 lists runtimes for all evaluated methods, and Sec.A.4 presents additional experiments on boundary accuracy. Section B showcases a selection of controlled experiments on Depth Pro that helped guide architectural choices (Sec.B.2, Sec.B.3, and Sec.B.7), training objective design (Sec.B.5), and curriculum training (Sec.B.6). In Section C, we provide additional implementation, training and evaluation details, including a complete summary of the datasets that were involved in this paper. Finally, Section D provides additional material on downstream applications.

## Appendix A Additional Results

### A.1 Qualitative results

| Input Image | Depth Pro (Ours) | Depth Anything v2 | Marigold | Metric3D v2 |
|---|---|---|---|---|
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

| Input Image | Depth Pro (Ours) | Depth Anything v2 | Marigold | Metric3D v2 |
|---|---|---|---|---|
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

| Input Image | Depth Pro (Ours) | Depth Anything v2 | Marigold | Metric3D v2 |
|---|---|---|---|---|
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

We provide additional qualitative results of Depth Pro, Marigold (Ke et al., 2024), Metric3D v2 (Hu et al., 2024), and Depth Anything v2 (Yang et al., 2024b) on in-the-wild images from AM-2k (Li et al., 2022a), DIS-5k (Qin et al., 2022), and Unsplash11 1 https://www.unsplash.com in Fig. 5, Fig. 6, and Fig. 7. Fine details are repeatedly missed by Metric3D v2 and Depth Anything v2. Marigold reproduces finer details than Metric3D v2 and Depth Anything v2 but commonly yields noisy predictions.

### A.2 Zero-shot metric depth

Expanding on the summary in Tab.1, we provide additional results for zero-shot metric depth estimation in Tab. 4. We report results on Booster (Ramirez et al., 2024), Middlebury (Scharstein et al., 2014), Sun-RGBD (Song et al., 2015), ETH3D (Schöps et al., 2017), nuScenes (Caesar et al., 2020), and Sintel (Butler et al., 2012). Our baselines include Depth Anything (Yang et al., 2024a) and Depth Anything v2 (Yang et al., 2024b), Metric3D (Yin et al., 2023) and Metric3D v2 (Hu et al., 2024), PatchFusion (Li et al., 2024a), UniDepth (Piccinelli et al., 2024), ZeroDepth (Guizilini et al., 2023), and ZoeDepth (Bhat et al., 2023). To preserve the zero-shot setting, we do not report results for models that were trained on the same dataset as the evaluation dataset. We report commonly used metrics in the depth estimation literature, namely 𝐴𝑏𝑠𝑅𝑒𝑙\mathit{AbsRel}, 𝐿𝑜𝑔10\mathit{Log}_{10} (Saxena et al., 2009), δ1\delta_{1}, δ2\delta_{2} and δ3\delta_{3} scores (Ladicky et al., 2014), as well as point-cloud metrics (Spencer et al., 2022). Due to the high resolution of Booster images, we were not able to obtain point-cloud metrics in reasonable time.

| NuScenes | AbsRel↓\downarrow | Log10↓{}_{10}\downarrow | δ2↑\delta_{2}\uparrow | δ3↑\delta_{3}\uparrow | SI-Log↓\downarrow | PC-CD↓\downarrow | PC-F↑\uparrow | PC-IoU↑\uparrow |
|---|---|---|---|---|---|---|---|---|
| DepthAnything (Yang et al., 2024a) | 0.453 | 0.151 | 73.876 | 90.301 | 28.153 | 24.146 | 0.007 | 0.004 |
| DepthAnything v2 (Yang et al., 2024b) | 0.614 | 0.326 | 31.837 | 47.265 | 29.737 | 37.516 | 0.008 | 0.004 |
| Metric3D (Yin et al., 2023) | 0.422 | 0.132 | 77.220 | 83.605 | 33.827 | 29.284 | 0.007 | 0.004 |
| Metric3D v2 (Hu et al., 2024) | 0.197 | 0.080 | 93.252 | 95.736 | 27.032 | 14.876 | 0.008 | 0.004 |
| PatchFusion (Li et al., 2024a) | 0.392 | 0.226 | 48.742 | 76.035 | 31.171 | 20.836 | 0.006 | 0.003 |
| UniDepth (Piccinelli et al., 2024) | 0.138 | 0.060 | 93.006 | 96.415 | 21.801 | 11.629 | 0.009 | 0.004 |
| ZeroDepth (Guizilini et al., 2023) | 0.237 | 0.121 | 82.596 | 89.908 | 30.703 | 23.348 | 0.007 | 0.004 |
| ZoeDepth (Bhat et al., 2023) | 0.498 | 0.182 | 64.947 | 82.704 | 31.501 | 39.183 | 0.006 | 0.003 |
| Depth Pro (Ours) | 0.287 | 0.164 | 73.836 | 84.252 | 29.548 | 22.480 | 0.010 | 0.005 |

| Sintel | AbsRel↓\downarrow | Log10↓{}_{10}\downarrow | δ2↑\delta_{2}\uparrow | δ3↑\delta_{3}\uparrow | SI-Log↓\downarrow | PC-CD↓\downarrow | PC-F↑\uparrow | PC-IoU↑\uparrow |
|---|---|---|---|---|---|---|---|---|
| DepthAnything (Yang et al., 2024a) | 3.973 | 0.559 | 15.418 | 27.281 | 35.771 | 38.592 | 0.057 | 0.030 |
| DepthAnything v2 (Yang et al., 2024b) | 2.226 | 0.494 | 18.696 | 33.820 | 41.923 | 54.931 | 0.057 | 0.031 |
| Metric3D (Yin et al., 2023) | 1.733 | 0.387 | 32.375 | 44.793 | 48.605 | 45.858 | 0.056 | 0.031 |
| Metric3D v2 (Hu et al., 2024) | 0.370 | 0.216 | 62.915 | 76.866 | 25.312 | 34.790 | 0.091 | 0.051 |
| PatchFusion (Li et al., 2024a) | 0.617 | 0.391 | 35.515 | 51.443 | 36.806 | 44.615 | 0.077 | 0.045 |
| UniDepth (Piccinelli et al., 2024) | 0.869 | 0.301 | 35.722 | 57.256 | 42.837 | 32.338 | 0.098 | 0.057 |
| ZeroDepth (Guizilini et al., 2023) | 0.703 | 0.491 | 25.629 | 37.076 | 50.839 | 76.274 | 0.052 | 0.029 |
| ZoeDepth (Bhat et al., 2023) | 0.946 | 0.392 | 22.698 | 44.969 | 40.217 | 52.301 | 0.085 | 0.049 |
| Depth Pro (Ours) | 0.508 | 0.230 | 59.247 | 71.138 | 27.494 | 41.968 | 0.121 | 0.073 |

| Sun-RGBD | AbsRel↓\downarrow | Log10↓{}_{10}\downarrow | δ2↑\delta_{2}\uparrow | δ3↑\delta_{3}\uparrow | SI-Log↓\downarrow | PC-CD↓\downarrow | PC-F↑\uparrow | PC-IoU↑\uparrow |
|---|---|---|---|---|---|---|---|---|
| DepthAnything (Yang et al., 2024a) | 0.114 | 0.053 | 98.811 | 99.770 | 8.038 | 0.034 | 0.160 | 0.090 |
| DepthAnything v2 (Yang et al., 2024b) | 0.182 | 0.070 | 97.645 | 99.462 | 8.390 | 0.045 | 0.169 | 0.096 |
| Metric3D (Yin et al., 2023) | 1.712 | 0.382 | 26.999 | 34.116 | 20.262 | 0.506 | 0.060 | 0.032 |
| Metric3D v2 (Hu et al., 2024) | 0.156 | 0.076 | 96.348 | 99.548 | 7.433 | 0.025 | 0.179 | 0.102 |
| PatchFusion (Li et al., 2024a) | 0.466 | 0.961 | 60.145 | 60.651 | 69.647 | 331.477 | 0.052 | 0.027 |
| UniDepth (Piccinelli et al., 2024) | 0.087 | 0.037 | 99.330 | 99.804 | 6.968 | 0.020 | 0.294 | 0.183 |
| ZoeDepth (Bhat et al., 2023) | 0.123 | 0.053 | 97.954 | 99.505 | 8.964 | 0.048 | 0.135 | 0.075 |
| Depth Pro (Ours) | 0.113 | 0.049 | 98.506 | 99.547 | 7.841 | 0.039 | 0.179 | 0.103 |

| ETH3D | AbsRel↓\downarrow | Log10↓{}_{10}\downarrow | δ2↑\delta_{2}\uparrow | δ3↑\delta_{3}\uparrow | SI-Log↓\downarrow | PC-CD↓\downarrow | PC-F↑\uparrow | PC-IoU↑\uparrow |
|---|---|---|---|---|---|---|---|---|
| DepthAnything (Yang et al., 2024a) | 1.682 | 0.380 | 19.784 | 31.057 | 10.903 | 0.072 | 0.172 | 0.114 |
| DepthAnything v2 (Yang et al., 2024b) | 0.370 | 0.173 | 64.657 | 86.256 | 9.683 | 0.042 | 0.330 | 0.233 |
| Metric3D (Yin et al., 2023) | 0.859 | 0.240 | 49.291 | 57.573 | 14.541 | 0.072 | 0.303 | 0.219 |
| Metric3D v2 (Hu et al., 2024) | 0.124 | 0.053 | 99.553 | 99.900 | 6.197 | 0.083 | 0.466 | 0.358 |
| PatchFusion (Li et al., 2024a) | 0.256 | 0.106 | 88.378 | 97.306 | 11.023 | 0.042 | 0.209 | 0.135 |
| UniDepth (Piccinelli et al., 2024) | 0.457 | 0.186 | 57.670 | 81.483 | 7.729 | 0.031 | 0.409 | 0.305 |
| ZoeDepth (Bhat et al., 2023) | 0.500 | 0.176 | 64.452 | 81.434 | 13.250 | 0.078 | 0.127 | 0.082 |
| Depth Pro (Ours) | 0.327 | 0.193 | 61.309 | 71.228 | 10.170 | 0.094 | 0.487 | 0.398 |

| Middlebury | AbsRel↓\downarrow | Log10↓{}_{10}\downarrow | δ2↑\delta_{2}\uparrow | δ3↑\delta_{3}\uparrow | SI-Log↓\downarrow | PC-CD↓\downarrow | PC-F↑\uparrow | PC-IoU↑\uparrow |
|---|---|---|---|---|---|---|---|---|
| DepthAnything (Yang et al., 2024a) | 0.273 | 0.149 | 69.619 | 86.060 | 12.420 | 0.102 | 0.103 | 0.055 |
| DepthAnything v2 (Yang et al., 2024b) | 0.262 | 0.141 | 72.074 | 90.549 | 9.639 | 0.063 | 0.127 | 0.069 |
| Metric3D (Yin et al., 2023) | 1.251 | 0.305 | 37.528 | 58.733 | 12.091 | 0.069 | 0.069 | 0.036 |
| Metric3D v2 (Hu et al., 2024) | 0.450 | 0.152 | 73.321 | 88.610 | 5.519 | 0.022 | 0.215 | 0.122 |
| PatchFusion (Li et al., 2024a) | 0.250 | 0.108 | 87.166 | 98.154 | 14.641 | 0.319 | 0.084 | 0.044 |
| UniDepth (Piccinelli et al., 2024) | 0.324 | 0.127 | 80.047 | 99.621 | 7.379 | 0.113 | 0.221 | 0.129 |
| ZeroDepth (Guizilini et al., 2023) | 0.377 | 0.179 | 67.060 | 78.952 | 14.482 | 0.232 | 0.052 | 0.027 |
| ZoeDepth (Bhat et al., 2023) | 0.214 | 0.115 | 77.683 | 90.860 | 10.448 | 0.069 | 0.114 | 0.062 |
| Depth Pro (Ours) | 0.251 | 0.089 | 93.169 | 96.401 | 8.610 | 0.107 | 0.161 | 0.091 |

| Booster | AbsRel↓\downarrow | Log10↓{}_{10}\downarrow | δ2↑\delta_{2}\uparrow | δ3↑\delta_{3}\uparrow | SI-Log↓\downarrow |
|---|---|---|---|---|---|
| DepthAnything (Yang et al., 2024a) | 0.317 | 0.114 | 79.615 | 95.228 | 10.507 |
| DepthAnything v2 (Yang et al., 2024b) | 0.315 | 0.110 | 76.239 | 94.276 | 7.056 |
| Metric3D (Yin et al., 2023) | 1.332 | 0.346 | 13.073 | 33.975 | 10.631 |
| Metric3D v2 (Hu et al., 2024) | 0.417 | 0.140 | 75.783 | 92.833 | 3.932 |
| PatchFusion (Li et al., 2024a) | 0.719 | 0.213 | 49.387 | 72.892 | 14.128 |
| UniDepth (Piccinelli et al., 2024) | 0.500 | 0.166 | 60.904 | 89.213 | 7.436 |
| ZoeDepth (Bhat et al., 2023) | 0.610 | 0.195 | 52.655 | 75.508 | 10.551 |
| Depth Pro (Ours) | 0.336 | 0.118 | 79.429 | 96.524 | 4.616 |

### A.3 Runtime

To assess the latency of our approach in comparison to baselines, we test all approaches on images of varying sizes and report results in Tab. 5. We pick common image resolutions (VGA: 640×\times480, HD: 1920×\times1080, 4K: 4032×\times3024) and measure each method’s average runtime for processing an image of the given size. All reported runtimes are reproduced in our environment and include preprocessing, eventual resizing (for methods operating at a fixed internal resolution), and inference of each model. We further report the parameter counts and flops (at HD resolution) for each method as measured with the fvcore package.

Among all approaches with a fixed output resolution, Depth Pro has the highest native output resolution, processing more than 3 times as many pixels as the next highest, Metric3D v2 (Hu et al., 2024). Yet Depth Pro has less than half the parameter count and requires only a third of the runtime compared to Metric3D v2.

The variable-resolution approaches (PatchFusion (Li et al., 2024a) and ZeroDepth (Guizilini et al., 2023)) have considerably larger runtime, with the faster model, ZeroDepth, taking almost 4 times as long as Depth Pro, even for small VGA images.

| Method | Parameter count | FlopsHD↓\downarrow | Native output resolution ↑\!\uparrow | t𝑉𝐺𝐴t_{\mathit{VGA}} (ms) ↓\downarrow | t𝐻𝐷t_{\mathit{HD}} (ms) ↓\downarrow | t4​Kt_{\mathit{4K}} (ms) ↓\downarrow |  |
|---|---|---|---|---|---|---|---|
| DPT | 123M | - | 384 ×\times 384 | = 0.15 MP | 33.2 | 30.6 | 27.8 |
| ZoeDepth | 340M | - | 384 ×\times 512 | = 0.20 MP | 235.7 | 235.1 | 235.4 |
| DepthAnything v2 | 335M | 1827G | 518 ×\times 518 | = 0.27 MP | 90.9 | 91.1 | 91.2 |
| UniDepth | 347M | 630G | 462 ×\times 616 | = 0.28 MP | 178.5 | 183.0 | 198.1 |
| Metric3D | 203M | 477G | 480 ×\times 1216 | = 0.58 MP | 217.9 | 263.8 | 398.1 |
| Marigold | 949M | - | 768 ×\times 768 | = 0.59 MP | 5174.3 | 4433.6 | 4977.6 |
| Metric3D v2 | 1.378G | 6830G | 616 ×\times 1064 | = 0.66 MP | 1299.6 | 1299.7 | 1390.2 |
| PatchFusion | 203M | - | Original (tile-based) | 84012.0 | 84029.9 | 84453.9 |  |
| ZeroDepth | 233M | 10862G | Original | 1344.3 | 8795.7 | 34992.2 |  |
| Depth Pro | 504M | 4370G | 1536 ×\times 1536 | = 2.36 MP | 341.3 | 341.3 | 341.3 |

### A.4 Boundary experiments

Boundary metrics empirical study. To illustrate how our boundary metrics work, we report additional qualitative edge metric results in Fig. 8. In particular, we show the occluding contours derived from the ground-truth and predicted depth, which illustrate how incorrect depth boundary predictions can impact the metric. Furthermore, to illustrate the behavior of the boundary precision and recall measurements under various image perturbations we also provide an empirical study in Fig. 9. We report both quantitative and qualitative results on samples from the UnrealStereo4K dataset (Tosi et al., 2021). Our results empirically demonstrate the correlation between erroneous depth edge predictions and low precision and recall values.

|  | Image | dd | d^\hat{d} | ⋃(i,j)∈Ncd​(i,j)\bigcup\limits_{(i,j)\in N}c_{d}(i,j) | ⋃(i,j)∈Ncd^​(i,j)\bigcup\limits_{(i,j)\in N}c_{\hat{d}}(i,j) |
|---|---|---|---|---|---|
| Sintel (Butler et al., 2012) |  |  |  |  |  |
| Spring (Mehl et al., 2023) |  |  |  |  |  |
| AM-2k (Li et al., 2022a) |  |  |  |  |  |
| P3M-10k (Li et al., 2021) |  |  |  |  |  |
| DIS-5k (Qin et al., 2022) |  |  |  |  |  |

Sintel (Butler et al., 2012)

Spring (Mehl et al., 2023)

AM-2k (Li et al., 2022a)

P3M-10k (Li et al., 2021)

DIS-5k (Qin et al., 2022)

dd d^\hat{d} ⋃(i,j)∈Ncd​(i,j)\bigcup\limits_{(i,j)\in N}c_{d}(i,j) ⋃(i,j)∈Ncd^​(i,j)\bigcup\limits_{(i,j)\in N}c_{\hat{d}}(i,j) Gaussian Blur Pixel-Shift Up-Down (nearest) Up-Down (bilinear) Up-Down (bicubic)

Results on the iBims dataset (Koch et al., 2018). We supplement our boundary evaluation by results on the iBims dataset, which is commonly used for evaluating depth boundaries. iBims consists of images of indoor scenes that have been laser-scanned. The images are at 640×480640\times 480 resolution and have been supplemented with manually annotated occlusion boundary maps to facilitate evaluation. The iBims benchmark uses Depth Directed Errors (DDE), which evaluate overall metric depth accuracy, Depth Boundary Errors (DBE), which are similar in spirit to our proposed boundary metric but require manual annotation, and Planar Errors, which evaluate the accuracy of planes derived from the depth maps.

We find that Depth Pro is on par with the state of the art according to the DDE and PE metrics, and significantly outperforms all prior work according to the boundary metrics.

| Method | DDE (in %) | DBE (in px) | PE (in m/∘) |  |  |  |  |
|---|---|---|---|---|---|---|---|
|  | εDDE0\varepsilon^{0}_{\text{DDE}}↑\uparrow | εDDE−\varepsilon^{-}_{\text{DDE}}↓\downarrow | εDDE+\varepsilon^{+}_{\text{DDE}}↓\downarrow | εDBEacc\varepsilon^{\text{acc}}_{\text{DBE}}↓\downarrow | εDBEcomp\varepsilon^{\text{comp}}_{\text{DBE}}↓\downarrow | εPEplan\varepsilon^{\text{plan}}_{\text{PE}}↓\downarrow | εPEorie\varepsilon^{\text{orie}}_{\text{PE}}↓\downarrow |
| DPT (Ranftl et al., 2021) | 58.744 | 41.255 | 0.000 | 3.580 | 39.372 | 0.138 | 31.837 |
| Metric3D (Yin et al., 2023) | 88.608 | 1.337 | 10.054 | 2.073 | 19.011 | 0.100 | 22.451 |
| Metric3D v2 (Hu et al., 2024) | 84.721 | 0.546 | 14.732 | 1.843 | 10.062 | 0.095 | 19.561 |
| ZoeDepth (Bhat et al., 2023) | 85.600 | 13.874 | 0.525 | 1.960 | 18.166 | 0.103 | 20.108 |
| Depth Anything (Yang et al., 2024a) | 88.951 | 10.741 | 0.308 | 2.081 | 19.172 | 0.106 | 20.680 |
| Depth Anything v2 (Yang et al., 2024b) | 91.773 | 1.619 | 6.607 | 1.959 | 8.350 | 0.095 | 19.406 |
| PatchFusion (Li et al., 2024a) | 85.765 | 12.602 | 1.633 | 1.711 | 20.722 | 0.117 | 23.926 |
| Marigold (Ke et al., 2024) | 58.738 | 41.261 | 0.000 | 1.855 | 12.742 | 0.168 | 33.734 |
| UniDepth (Piccinelli et al., 2024) | 73.020 | 0.041 | 26.939 | 1.999 | 14.234 | 0.098 | 19.114 |
| Depth Pro (Ours) | 89.725 | 1.809 | 8.464 | 1.680 | 10.138 | 0.095 | 18.776 |

## Appendix B Controlled Experiments

We conduct several controlled experiments to investigate the impact of various components and design decisions in Depth Pro. Specifically, we aim to assess the contribution of the native output resolution, key components in the network architecture (Sec. B.2), the depth representation, training objectives (Sec. B.5), training curriculum (Sec. B.6), and the focal length estimation head (Sec. B.7).

### B.1 Native output resolution

To assess the importance of the native output resolution, we conduct a simple experiment on UnrealStereo4K. We downsample the ground truth depth maps to several common resolutions found in the literature as well the output resolution of Depth Pro. We then upsample again to the original resolution and evaluate depth and boundary metrics. Results are listed in Tab. 7. We find that the native output resolution has a strong effect on boundary accuracy. More specifically, doubling the resolution may improve boundary accuracy by a factor of 3. It is important to note however, that these results represent an upper bound. The zero-shot comparison to the state of the art (Tab. 1 and 2) includes approaches that predict depth at the full input resolution, namely PatchFusion and ZeroDepth. Although PatchFusion and ZeroDepth predict at a higher resolution than e.g., Metric3D v2 or DepthAnything v2, they trail the lower resolution approaches in performance. We conclude that predicting at a high native output resolution is necessary but not sufficient for predicting accurate metric depth with sharp boundaries.

| Output resolution | Example | Approximate optimum |  |  |
|---|---|---|---|---|
| Log10↓\downarrow | AbsRel↓\downarrow | F1↑\uparrow |  |  |
| 1536×\times1536 | Depth Pro | 0.019 | 0.004 | 0.311 |
| 768×\times768 | Marigold (Ke et al., 2024) | 0.048 | 0.010 | 0.131 |
| 518×\times518 | Depth Anything v2 (Yang et al., 2024a) | 0.084 | 0.016 | 0.065 |
| 384×\times384 | DPT (Ranftl et al., 2021) | 0.123 | 0.024 | 0.044 |

### B.2 Network backbone

To assess the effect of the network architecture, we begin by evaluating various candidate image encoder backbones within our network architecture. To assess their performance, we conduct a comparative analysis utilizing off-the-shelf models available from the TIMM library (Wightman, 2019). Using the pretrained weights, we train each backbone at 384 ×\times 384 resolution across five RGB-D datasets (Keystone, HRWSI, RedWeb, TartanAir, and Hypersim) and evaluate their performance in terms of metric depth accuracy across multiple datasets, including Booster, Hypersim, Middlebury, and NYUv2, utilizing metrics such as 𝐴𝑏𝑠𝑅𝑒𝑙\mathit{AbsRel} for affine-invariant depth and 𝐿𝑜𝑔10\mathit{Log}_{10} for metric depth in Tab. 8. We find that ViT-L DINOv2 (Oquab et al., 2024) outperforms all other backbones by a significant margin and conclude that the combination of backbone and pretraining strategy considerably affects downstream performance. Following this analysis, we pick ViT-L DINOv2 for our encoder backbones.

| Backbone | Flops (G) | Params (M) | AbsRel ↓\downarrow | Log10↓{}_{10}\downarrow |  |
|---|---|---|---|---|---|
| ViT-L DINOv2-reg4 (Oquab et al., 2024) | 248 | 345 | 0.039 | 0.138 |  |
| ViT-L DINOv2 (Oquab et al., 2024) | 247 | 345 | 0.040 | 0.129 |  |
| ViT-L MAE (He et al., 2022) | 247 | 343 | 0.041 | 0.150 |  |
| ViT-L BeiTv2 (Peng et al., 2022b) | 242 | 336 | 0.042 | 0.134 |  |
| ViT-L BeiT (Bhat et al., 2023) | 259 | 336 | 0.048 | 0.147 |  |
| ViT-L SO400m-siglip (Zhai et al., 2023) | 311 | 471 | 0.051 | 0.174 |  |
| ViT-L CLIP-quickgelu (Radford et al., 2021) | 247 | 344 | 0.053 | 0.166 |  |
| ViT-L CLIP (Radford et al., 2021) | 247 | 345 | 0.057 | 0.156 |  |
| ViT-L (Dosovitskiy et al., 2021) | 247 | 345 | 0.061 | 0.163 |  |
| ConvNext-XXL (Liu et al., 2022b) | 514 | 867 | 0.075 | 0.216 |  |
| ViT-L DeiT-3 (Touvron et al., 2022) | 247 | 345 | 0.078 | 0.176 |  |
| ConvNext-L-mlp (Liu et al., 2022b) | 162 | 214 | 0.081 | 0.222 |  |
| ConvNextv2-H (Woo et al., 2023) | 405 | 680 | 0.085 | 0.242 |  |
| SegAnything ViT-L (Kirillov et al., 2023) | 245 | 330 | 0.087 | 0.311 |  |
| SWINv2-L (Liu et al., 2022a) | 177 | 212 | 0.091 | 0.240 |  |
| CAFormer-B36 (Yu et al., 2024) | 124 | 108 | 0.091 | 0.248 |  |
| EfficientViT-L3 (Liu et al., 2023) | – | – | 0.109 | 0.303 |  |

### B.3 High-resolution alternatives

We further evaluate alternative high-resolution 1536×\times1536 network structures and different pretrained weights (Tab. 9). To do this, we test generalization accuracy by training on a train split of some datasets and testing on a val or test split of other datasets, following the Stage 1 protocol for all models in accordance with Tab. 16 and Tab. 17. All ViT models use a patch size of 16×\times16. For weights pretrained with a patch size of 14×\times14 we apply bicubic interpolation to the weights of the convolutional patch embedding layer and scale these weights inversely to the number of pixels (i.e., the weights are reduced by a factor of 1.3). All ViT models use resolution 1536×\times1536, for this we apply bicubic interpolation to positional embeddings prior to training. The Depth Pro approach in all cases uses ViT with resolution 384×\times384 and patch size 16×\times16 for both the patch encoder and the image encoder. SWINv2 and convolutional models are pretrained on ImageNet (Deng et al., 2009). Other models use different pretraining approaches described in their papers: CLIP (Radford et al., 2021), MAE (He et al., 2022), BeiTv2 (Peng et al., 2022b), and DINOv2 (Oquab et al., 2024). For the Segment Anything model we use publicly available pretrained weights, which were initialized using MAE pretraining (He et al., 2022) and subsequently trained for segmentation as described in their paper (Kirillov et al., 2023).

|  |  |  | Metric depth accuracy | Boundary accuracy |  |  |
|---|---|---|---|---|---|---|
|  | Method | Latency, ms ↓\downarrow | NYUv2 δ1↑\delta_{1}\!\uparrow | iBims δ1↑\delta_{1}\!\uparrow | iBims F1↑\uparrow | DIS R↑\uparrow |
| Conv. | EfficientNetV2-XL (Tan & Le, 2021) | 118 | 4.4 | 7.0 | 0.005 | 0.000 |
| ConvNext-XXL (Liu et al., 2022b) | 304 | 68.0 | 38.3 | 0.134 | 0.031 |  |
|  | ConvNextv2-H (Woo et al., 2023) | 287 | 70.0 | 56.6 | 0.131 | 0.044 |
| Trans. | S. Anything (Kirillov et al., 2023) (ViT-L) | 349 | 53.2 | 38.9 | 0.140 | 0.051 |
| S. Anything (Kirillov et al., 2023) (ViT-H) | 365 | 51.7 | 41.1 | 0.146 | 0.050 |  |
|  | SWINv2-L (Liu et al., 2022a) (window=24) | 272 | 58.4 | 33.1 | 0.117 | 0.028 |
| ViT | ViT-L CLIP (Radford et al., 2021) | 384 | 92.2 | 81.9 | 0.157 | 0.052 |
| ViT-L BeiTv2 (Peng et al., 2022b) | OOM | 90.4 | 86.5 | 0.149 | 0.042 |  |
| ViT-L MAE (He et al., 2022) | 390 | 92.7 | 84.7 | 0.163 | 0.065 |  |
| ViT-L DINOv2 (Oquab et al., 2024) | 392 | 96.5 | 90.3 | 0.161 | 0.065 |  |
|  | Depth Pro | 341 | 96.1 | 91.3 | 0.177 | 0.080 |

We find that the presented Depth Pro approach is faster and more accurate for object boundaries than the plain ViT, with comparable metric depth accuracy. In comparison to other transformer-based and convolutional models, Depth Pro has comparable latency, several times lower metric depth error, and several times higher recall accuracy for object boundaries.

Importantly, our proposed architecture performs significantly better than straight-forward scaling up the ViT architecture with DINOv2 pretraining. On DIS5K for instance, our architecture improves the boundary recall by relative 23%23\% over DINOv2.

### B.4 Depth representation

To assess the effect of the predicted depth representation, we train a ViT encoder and DPT decoder on Hypersim to predict inverse depth, log-depth, or depth. All configurations in this experiment are supervised with a only mean absolute error. Tab. 10 lists the d​e​l​t​a1delta_{1} error computed over several depth ranges. We find that supervising the network on depth directly leads to worse results than log-depth or inverse depth. Overall, predicting inverse depth works best, with the largest difference in the regions close to the camera. This makes inverse depth the representation of choice for downstream tasks like novel view synthesis, which benefit particularly from higher accuracy close to the camera.

|  | Hypersim δ1↑\delta_{1}\!\uparrow |  |  |  |  |  |
|---|---|---|---|---|---|---|
| Training objective | 0-1m | 1-2m | 2-4m | 4-8m | 8-16m | >>16m |
| Inverse-depth | 0.730 | 0.833 | 0.896 | 0.921 | 0.922 | 0.922 |
| Log-depth | 0.700 | 0.807 | 0.892 | 0.919 | 0.920 | 0.920 |
| Depth | 0.657 | 0.716 | 0.819 | 0.853 | 0.850 | 0.850 |

### B.5 Training objectives

To assess the efficacy of our training curriculum, we compare it to alternative training schedules. We first examine the different stages individually and then compare full curricula.

|  | HRWSI | Hypersim | Apolloscape |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
| Cond. | AbsRel↓\downarrow | δ1↑\delta_{1}\!\uparrow | Log10↓{}_{10}\!\downarrow | AbsRel↓\downarrow | δ1↑\delta_{1}\!\uparrow | F1↑\uparrow | Log10↓{}_{10}\!\downarrow | AbsRel↓\downarrow | δ1↑\delta_{1}\!\uparrow |
| 1A | 0.166 | 82.1 | 0.083 | 0.259 | 75.4 | 0.221 | 0.156 | 0.339 | 45.6 |
| 1D | 0.138 | 85.1 | 0.077 | 0.246 | 78.4 | 0.391 | 0.128 | 0.424 | 60.6 |
| 1B | 0.156 | 83.3 | 0.078 | 0.249 | 77.3 | 0.388 | 0.152 | 0.300 | 47.3 |
| 1C | 0.150 | 83.7 | 0.074 | 0.235 | 79.9 | 0.442 | 0.084 | 0.235 | 75.6 |

Stage 1 training objectives. We first evaluate loss combinations for the first stage and report results in Tab. 11. Condition 1A only applies a mean absolute error loss to all datasets. For non-metric datasets, we use the scale-and-shift-invariant version. Condition 1B adds gradient losses to all synthetic datasets. We again use the scale-and-shift-invariant version for non-metric datasets. Following our observations from Sec. 3, we propose to apply an appropriate mean absolute error loss as in other conditions depending on a dataset being metric, but apply a scale-and-shift-invariant gradient loss irrespective of a dataset being metric or not (C). We find that loss combinations minimizing gradients (1B, 1C, 1D) consistently outperform just applying an absolute error (1A). Besides improving relative and metric depth estimates, they strongly improve boundary metrics, here up to a factor of 2. Interestingly, minimizing gradients on all datasets outperforms minimizing gradients on just synthetic data. This suggests that the added diversity from real-world datasets more than balances out their potentially noisy ground truth, even for minimizing gradients, which emphasize the noise. The best performance however, is achieved by applying a scale-and-shift-invariant gradient loss on the synthetic datasets (1C). We found that this setting improves convergence and overall performs best.

|  |  |  |  |  | HRWSI | Hypersim | Apolloscape |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Condition | ℒ𝑀𝑆𝐸\mathcal{L}_{\mathit{MSE}} | ℒ𝑀𝐴𝐺𝐸\mathcal{L}_{\mathit{MAGE}} | ℒ𝑀𝑆𝐺𝐸\mathcal{L}_{\mathit{MSGE}} | ℒ𝑀𝐴𝐿𝐸\mathcal{L}_{\mathit{MALE}} | AbsRel↓\downarrow | δ1↑\delta_{1}\!\uparrow | Log10↓{}_{10}\!\downarrow | AbsRel↓\downarrow | δ1↑\delta_{1}\!\uparrow | F1↑\uparrow | Log10↓{}_{10}\!\downarrow | AbsRel↓\downarrow | δ1↑\delta_{1}\!\uparrow |
| 2A | ✓ | ✓ | ✓ | ✓ | 0.149 | 83.6 | 0.072 | 0.235 | 81.3 | 0.465 | 0.092 | 0.303 | 72.9 |
| 2B | ✓ | ✓ | ✓ |  | 0.148 | 83.7 | 0.072 | 0.230 | 81.0 | 0.463 | 0.092 | 0.299 | 73.1 |
| 2C | ✓ | ✓ |  |  | 0.150 | 83.7 | 0.072 | 0.235 | 80.8 | 0.468 | 0.091 | 0.300 | 73.2 |
| 2D | ✓ |  |  |  | 0.150 | 83.4 | 0.074 | 0.239 | 79.8 | 0.461 | 0.096 | 0.349 | 72.8 |
| 2E |  |  |  |  | 0.159 | 82.7 | 0.074 | 0.242 | 80.6 | 0.459 | 0.096 | 0.346 | 73.3 |

Condition

ℒ𝑀𝑆𝐸\mathcal{L}_{\mathit{MSE}}

ℒ𝑀𝐴𝐺𝐸\mathcal{L}_{\mathit{MAGE}}

ℒ𝑀𝑆𝐺𝐸\mathcal{L}_{\mathit{MSGE}}

ℒ𝑀𝐴𝐿𝐸\mathcal{L}_{\mathit{MALE}}

Stage 2 training objectives. The second stage of our training curriculum focuses on sharpening depth boundaries while retaining high metric depth accuracy. To that end, we only employ synthetic datasets due to their high quality ground truth. The obvious strategy for sharpening predictions is the application of gradient losses. We evaluate our combination of multi-scale derivative-based losses in an ablation study. Condition 2A uses all of the losses, namely ℒ𝑀𝐴𝐸\mathcal{L}_{\mathit{MAE}}, ℒ𝑀𝑆𝐸\mathcal{L}_{\mathit{MSE}} ,ℒ𝑀𝐴𝐺𝐸\mathcal{L}_{\mathit{MAGE}}, ℒ𝑀𝐴𝐿𝐸\mathcal{L}_{\mathit{MALE}}, and ℒ𝑀𝑆𝐺𝐸\mathcal{L}_{\mathit{MSGE}}. See Tab. 12. 2B removes the second-order loss ℒ𝑀𝐴𝐿𝐸\mathcal{L}_{\mathit{MALE}}. 2C further removes the squared first order losses ℒ𝑀𝑆𝐺𝐸\mathcal{L}_{\mathit{MSGE}}. 2D removes all derivative-based losses. 2E applies the ℒ𝑀𝐴𝐸\mathcal{L}_{\mathit{MAE}} to all datasets. Removing ℒ𝑀𝐴𝐿𝐸\mathcal{L}_{\mathit{MALE}} improves results on Apolloscape. Our combination of 0th- to 2nd-order derivative losses (2A) performs best across metrics and datasets in aggregate (e.g., in terms of the average rank across metrics).

### B.6 Full curricula

|  | HRWSI | Hypersim | Apolloscape |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
| Cond. | AbsRel↓\downarrow | δ1↑\delta_{1}\!\uparrow | Log10↓{}_{10}\!\downarrow | AbsRel↓\downarrow | δ1↑\delta_{1}\!\uparrow | F1↑\uparrow | Log10↓{}_{10}\!\downarrow | AbsRel↓\downarrow | δ1↑\delta_{1}\!\uparrow |
| 3A (Ours) | 0.149 | 83.6 | 0.072 | 0.235 | 81.3 | 0.465 | 0.092 | 0.303 | 72.9 |
| 3B | 0.148 | 83.9 | 0.073 | 0.245 | 81.3 | 0.478 | 0.095 | 0.292 | 72.1 |
| 3C | 0.153 | 83.6 | 0.166 | 0.386 | 37.1 | 0.095 | 0.586 | 0.712 | 0.5 |

We assess the efficacy of our complete training curriculum in comparison to alternatives. Condition 3A represents our two-stage curriculum. Condition 3B trains in a single stage and applies all the second-stage gradient losses throughout the whole training. Condition 3C reverses our two stages and represents the established strategy of pretraining on synthetic data first and fine-tuning with real-world data. We find that training with a single stage is a reasonable default strategy and works much better than first training on synthetic data and then fine-tuning on real data. Our proposed strategy however, yields further improvements on metric depth with slightly worse boundary accuracy.

### B.7 Focal length estimation

Additional analysis of zero-shot focal length estimation accuracy. In Fig. 10, we present a more comprehensive analysis of our focal length predictor’s performance compared to baseline models. To that end, we plot the percentage of samples below a certain absolute relative error for each method and dataset in our zero-shot evaluation set up. Depth Pro outperforms all approaches on all datasets.

| Architecture | δ25%↑\delta_{25\%}\uparrow |
|---|---|
| Encoder for depth only | 60.0 |
| Encoder for focal length only | 74.4 |
| Encoder for depth and refinement network | 63.6 |
| Parallel encoders for depth and focal length | 78.2 |

Controlled evaluation of network structures. We evaluate a number of choices for the focal length estimation head and report results in Tab. 14. The models are evaluated on 500 images randomly sampled from Flickr (Thomee et al., 2016). As the first condition, we extract features from the frozen image encoder trained for depth estimation and add a small convolutional head. As the second condition, we train a separate ViT-based image encoder (Dosovitskiy et al., 2021). As the third condition, we train a ViT-based encoder on extracted features from the frozen image encoder for depth estimation. The final condition represents our chosen architecture depicted in Fig. 3, which utilizes frozen features from the depth network and task-specific features from a separate ViT image encoder in parallel.

We observe that refining depth features performs on par with just using the frozen depth features, suggesting that adding more computation on top of the frozen DPT features in addition to our small convolutional head does not provide extra benefits despite the increased computation. Training a separate image encoder from scratch improves performance by 14.6 percentage points, which indicates that accurate focal length prediction requires extra task-specific knowledge in addition to depth information. Furthermore, the two encoders in parallel outperform using just a single image encoder for focal length prediction, which highlights the importance of features from the pretrained depth network for obtaining a high-performing focal length estimator.

## Appendix C Implementation, Training and Evaluation Details

In this section we provide additional details on the datasets used for training and evaluation, hyperparameter settings for our method, and details on the evaluation setup.

### C.1 Merge operation

We merge overlapping feature patches to feature maps by generating a Voronoi partition of the desired feature map. To generate the partition, we use the patch centers as seeds and obtain a Voronoi cell per feature patch. The area of the patch covered by the Voronoi cell is copied to the feature map, the remaining area discarded. By overlapping patches we ensure that the receptive field of the patch encoder partially covers neighboring patches. Fig. 11 illustrates the approach for merging 3×33\times 3 patches into the Feature 4 feature map.

### C.2 Focal length head

The focal length head consists of a three-layer convolutional network with kernel sizes 3,3,6, and strides 2,2,1. Each layer halves the channel dimension and is followed by a rectified linear unit. The first layer starts from 128 channels and the last reduces channels to a single focal length value per image.

### C.3 Datasets

Tab. 15 provides a comprehensive summary of the datasets utilized in our study, detailing their respective licenses and specifying their roles (e.g., training or testing).

| Dataset | URL | License | Usage |
|---|---|---|---|
| 3D Ken Burns (Niklaus et al., 2019) | https://github.com/sniklaus/3d-ken-burns | CC-BY-NC-SA 4.0 | Train |
| AM-2K (Li et al., 2022a) | https://github.com/JizhiziLi/GFM | Custom | Testing |
| Apolloscape (Huang et al., 2020) | https://apolloscape.auto/ | Custom | Val |
| ARKitScenes (Dehghan et al., 2021) | https://github.com/apple/ARKitScenes | Custom | Train |
| Bedlam (Black et al., 2023) | https://bedlam.is.tue.mpg.de/#data | Custom | Train |
| BlendedMVG (Yao et al., 2020) | https://github.com/YoYo000/BlendedMVS | CC BY 4.0 | Train |
| Booster (Ramirez et al., 2024) | https://cvlab-unibo.github.io/booster-web/ | CC BY NC 4.0 | Test |
| DDAD (Guizilini et al., 2020) | https://github.com/TRI-ML/DDAD | CC-BY-NC-SA 4.0 | Testing |
| DIML (indoor) (Kim et al., 2016) | https://dimlrgbd.github.io/ | Custom | Train |
| DIS5K (Qin et al., 2022) | https://xuebinqin.github.io/dis/index.html | Custom | Test |
| DPDD (Abuolaim & Brown, 2020) | https://github.com/Abdullah-Ab…pixel | MIT | Testing |
| Dynamic Replica (Karaev et al., 2023) | https://github.com/facebookres…stereo | CC BY-NC 4.0 | Train |
| EDEN (Le et al., 2021) | https://lhoangan.github.io/eden/ | Custom | Train |
| ETH3D (Schöps et al., 2017) | https://www.eth3d.net/ | CC-BY-NC-SA 4.0 | Testing |
| FiveK (Bychkovsky et al., 2011) | https://data.csail.mit.edu/graphics/fivek/ | Custom | Testing |
| HRWSI (Xian et al., 2020) | https://kexianhust.github….Ranking-Loss/ | Custom | Train, Val |
| Hypersim (Roberts et al., 2021) | https://github.com/apple/ml-hypersim | Custom | Train, Val |
| iBims (Koch et al., 2018) | https://www.asg.ed.tum.de/lmf/ibims1/ | Custom | Test |
| IRS (Wang et al., 2019) | https://github.com/HKBU-HPML/IRS | Custom | Train |
| KITTI (Geiger et al., 2013) | https://www.cvlibs.net/datasets/kitti/ | CC-BY-NC-SA 3.0 | Testing |
| Middlebury (Scharstein et al., 2014) | https://vision.middlebury.edu/stereo/data/ | Custom | Testing |
| MVS-Synth (Huang et al., 2018) | https://phuang17….mvs-synth.html | Custom | Train |
| NYUv2 (Silberman et al., 2012) | https://cs.nyu.edu/…v2.html | Custom | Testing |
| nuScenes (Caesar et al., 2020) | https://www.nuscenes.org/ | Custom | Testing |
| P3M-10k (Li et al., 2021) | https://github.com/JizhiziLi/P3M | Custom | Testing |
| PPR10K (Liang et al., 2021) | https://github.com/csjliang/PPR10K | Apache 2.0 | Testing |
| RAISE (Dang-Nguyen et al., 2015) | http://loki…download.html | Custom | Testing |
| ReDWeb (Xian et al., 2018) | https://sites.google.com/site/redwebcvpr18/ | Custom | Train |
| SAILVOS3D (Hu et al., 2021) | https://sailvos.web.illin…index.html | Custom | Train |
| ScanNet (Dai et al., 2017) | http://www.scan-net.org/ | Custom | Train |
| Sintel (Butler et al., 2012) | http://sintel.is.tue.mpg.de/ | Custom | Testing |
| SmartPortraits (Kornilova et al., 2022) | https://mobile…SmartPortraits/ | Custom | Train |
| SPAQ (Fang et al., 2020) | https://github.com/h4nwei/SPAQ | Custom | Testing |
| Spring (Mehl et al., 2023) | https://spring-benchmark.org/ | CC BY 4.0 | Testing |
| Sun-RGBD (Song et al., 2015) | https://rgbd.cs.princeton.edu/ | Custom | Testing |
| Synscapes (Wrenninge & Unger, 2018) | https://synscapes.on.liu.se/ | Custom | Train |
| TartanAir (Wang et al., 2020) | https://theairlab.org/tartanair-dataset/ | CC BY 4.0 | Train |
| UASOL (Bauer et al., 2019) | https://osf.io/64532/ | CC BY 4.0 | Train |
| UnrealStereo4K (Tosi et al., 2021) | https://github.com/fabiotosi92/SMD-Nets | Custom | Train |
| Unsplash | https://unsplash.com/data | Custom | Testing |
| UrbanSyn (Gómez et al., 2023) | https://www.urbansyn.org/ | CC BY-SA 4.0 | Train |
| VirtualKITTI2 (Gaidon et al., 2016) | https://europe.naverlabs.com…-worlds/ | CC BY-NC-SA 3.0 | Train |
| ZOOM (Zhang et al., 2019) | https://github.com/ceciliav…inference | - | Testing |

### C.4 Training hyperparameters

We specify the training hyperparameters in Tab. 16 and Tab. 17.

|  | Stage 1 Stage 2 | Stage 1 | Stage 2 |
|---|---|---|---|
| Stage 1 | Stage 2 |  |  |
| Epochs | 250 100 | 250 | 100 |
| 250 | 100 |  |  |
| Epoch length | 72000 |  |  |
| Schedule | 1 %1\text{\,}\mathrm{\%} warmup, 80 %80\text{\,}\mathrm{\%} constant LR, 19 %19\text{\,}\mathrm{\%} ×\times0.1 LR |  |  |
| LR for Encoder | 1.28e-5 |  |  |
| LR for Decoder | 1.28e-4 |  |  |
| Batch size | 128 |  |  |
| Optimizer | Adam |  |  |
| Weight decay | 0 |  |  |
| Clip gradient norm | 0.2 |  |  |
| Pretrained LayerNorm | Frozen |  |  |
| Random color change probability | 75 %75\text{\,}\mathrm{\%} |  |  |
| Random blur probability | 30 %30\text{\,}\mathrm{\%} |  |  |
| Center crop probability for FOV-augmentation | 50 %50\text{\,}\mathrm{\%} |  |  |
| Metric depth normalization | CSTM-label (Yin et al., 2023) |  |  |
| Number of channels for Decoder | 256 |  |  |
| Resolution | 1536×\times1536 |  |  |
| DepthPro model structure: |  |  |  |
| Image-Encoder resolution | 384×\times384 |  |  |
| Patch-Encoder resolution | 384×\times384 |  |  |
| Number of 384×\times384 patches in DepthPro | 35 |  |  |
| Intersection of 384×\times384 patches in DepthPro | 25 %25\text{\,}\mathrm{\%} |  |  |

| Stage 1 | Stage 2 |
|---|---|

| 250 | 100 |
|---|---|

| Loss function | Datasets |
|---|---|
| Stage 1 |  |
|  |  |
| MAE SSI-MAGE | Hypersim, Tartanair, Synscapes, Urbansyn, Dynamic Replica, Bedlam, IRS, Virtual Kitti2, Sailvos3d |
|  |  |
| MAE (trimmed = 20 %20\text{\,}\mathrm{\%}) | ARKitScenes, Diml Indoor, Scannet, Smart Portraits |
|  |  |
| SSI-MAE SSI-MAGE | UnrealStereo4k, 3D Ken Burns, Eden, MVS Synth |
|  |  |
| SSI-MAE (trimmed = 20 %20\text{\,}\mathrm{\%}) | HRWSI, BlendedMVG |
|  |  |
| Stage 2 |  |
|  |  |
| MAE, MSE, MAGE, MALE, MSGE | Hypersim, Tartanair, Synscapes, Urbansyn, Dynamic Replica, Bedlam, IRS, Virtual Kitti2, Sailvos3d |

| Dataset | Minimum distance (m) | Maximum distance (m) | Number of Samples | Depth Resolution (px) |
|---|---|---|---|---|
| Booster | 0.0010.001 | 1010 | 228 | 3008×41123008\times 4112 |
| ETH3D | 0.10.1 | 200200 | 454 | 4032×60484032\times 6048 |
| iBims | 0.10.1 | 1010 | 100 | 480×640480\times 640 |
| Middlebury | 0.0010.001 | 1010 | 15 | 1988×29521988\times 2952 |
| NuScenes | 0.0010.001 | 8080 | 881 | 900×1600900\times 1600 |
| Sintel | 0.010.01 | 8080 | 1064 | 436×1024436\times 1024 |
| Sun-RGBD | 0.0010.001 | 1010 | 5050 | 530×730530\times 730 |

### C.5 Baselines

Below we provide further details on the setup of the baselines.

DepthAnything. Depth Anything v1 and v2 each released a general model for relative depth, but their metric depth models are tailored to specific domains (indoor vs. outdoor). For the metric depth evaluation, we match these models to datasets according to their domain, and for datasets containing both indoor and outdoor images, we select the model with the best performance. For qualitative results and the (scale and shift invariant) zero-shot boundary evaluation, we employ the relative depth models, since they yield better qualitative results and sharper boundaries than the metric models.

Metric3D. For Metric3D v1 and v2, we found that the crop size parameter strongly affects metric scale accuracy. In fact, using a fixed crop size consistently yielded very poor results on at least one metric dataset. In order to obtain acceptable results, we used different crop sizes for indoor (512, 1088) and outdoor (512, 992) datasets. As in the case of Depth Anything, we mark these results in gray to indicate that they are not strictly zero-shot. For Metric 3D v2, we use the largest (‘giant’) model.

UniDepth. For UniDepth, we use the ViT-L version, which performs best on average among the UniDepth variants.

ZoeDepth. We use the model finetuned on both indoor and outdoor data (denoted ZoeD_NK).

### C.6 Evaluation setup

In evaluating our approach and baselines, we found the range of valid depth values, the depth map resolution used for computing metrics, the resizing approach used for matching the resolution of the ground truth depth maps, and the choice of intrinsics to affect results, sometimes strongly. This is why we made an effort to set up and evaluate each baseline in the same fair evaluation setup, which we detail below.

Tab. 18 lists our evaluation datasets, the range of depth values used for evaluation, the number of samples, and the resolution of the ground truth depth maps. In case a method predicted depth maps at a different resolution, we resized predictions bilinearly to match the ground truth resolution.

Since several factors outlined above can affect the reported accuracy of a method, few baselines report sufficient detail on their evaluation setup, and the exact evaluation setups may differ across baselines, it is generally impossible to exactly reproduce reported results while guaranteeing fairness. We prioritized fair comparison and tried to evaluate all baselines in the same environment. We were able to match most reported results, with the following three notable differences. ZeroDepth reported better results on nuScenes, which we attribute to the use of a different validation set in their evaluation. UniDepth reported different results on ETH3D, which we attribute to the handling of raw images; specifically, in our setup, we use the raw images without any post-processing, and take the intrinsics from the accompanying EXIF data; we believe this best adheres to the zero-shot premise for single-image depth estimation. Finally, on SUN-RGBD, Depth Anything fairs better in our evaluation setup than in the evaluation reported in the original paper.

Evaluation metric for sharp boundaries. For both our depth-based and mask-based boundary metrics, we apply the same weighted-averaging strategy to account for multiple relative depth ratios. F1 values (depth-based metrics) and recall values (mask-based metrics) are averaged across thresholds that range linearly from tm​i​n=5t_{min}=5 to tm​a​x=25t_{max}=25. Weights are computed as the normalized range of threshold values between tm​i​nt_{min} and tm​a​xt_{max}, such that stronger weights are given towards high threshold values.

## Appendix D Applications

Metric, sharp, and fast monocular depth estimation enables a variety of downstream applications. We showcase the utility of Depth Pro in two additional contexts beyond novel view synthesis: conditional image synthesis with ControlNet (Zhang et al., 2023b) and synthetic depth of field (Peng et al., 2022a).

Depth-conditioned image synthesis. In this application we stylize an image through a text prompt via ControlNet (Zhang et al., 2023b). To retain the structure of the input image, we predict a depth map from the input image and use it for conditioning the image synthesis through a pretrained depth-to-image ControlNet SD 1.5 model. Figure 12 shows the input image, prompt, and predicted depth maps and synthesis results for Depth Pro, Deoth Anything v2, Marigold, and Metric3D v2. We find that only Depth Pro accurately predicts the cables and sky region, resulting in a stylized image that retains the structure of the input image. Baselines either miss cables, causing the cable car to float in mid-air (Depth Anything v2), or add a gradient to the sky (Marigold).

| Input |  | Prompt: A watercolor image of a cable-car. |
|---|---|---|
|  | Predicted depth | Synthesized image |
| Depth Pro |  |  |
| Depth Anything v2 |  |  |
| Marigold |  |  |
| Metric3D v2 |  |  |

Synthetic depth of field. Synthetic depth of field can be used to highlight the primary subject in a photo by deliberately blurring the surrounding areas. BokehMe (Peng et al., 2022a) introduces a hybrid rendering framework that marries a neural renderer with a classical physically motivated renderer. This framework takes a single image along with a depth map as input. In this context, it is essential for the depth map to delineate objects well, such that the photo’s subject is kept correctly in focus while other content is correctly blurred out. Furthermore, the depth map should correctly trace out the details of the subject, to keep these (and only these) details correctly in focus. Figure 13 shows the advantage afforded by Depth Pro in this application. (We keep the most salient object in focus by setting the refocused disparity (disp_focus) hyperparameter of BokehMe as the disparity of the object.)
