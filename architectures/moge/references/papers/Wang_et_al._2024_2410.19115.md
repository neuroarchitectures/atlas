## MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision

### Ruicheng Wang¹

*⋆*

### Sicheng Xu² Cassie Dai³

*⋆*

### Jianfeng Xiang⁴

*⋆*

### Yu Deng² Xin Tong² Jiaolong Yang²

*†* 1

### USTC

### Microsoft Research

### Harvard

### Tsinghua University

*{*t-ruiwang,sichengxu,t-jxiang,dengyu,jiaoyan*}*@microsoft.com *{*cccassied,xtong.gfx*}*@gmail.com

# arXiv:2410.19115v3 [cs.CV] 15 Apr 2025

Figure 1. Given any image, our method reconstructs an affine-invariant 3D point map of the scene and can also produce a depth map and

the camera focal length. The model yields high-quality shapes and generalizes well to open-domain images. (*Best viewed with zoom*)

*⋆* Work done during internship at Microsoft Research *†* Corresponding author

Scale-inv. ★**Ours**

### Abstract

point ★UniDepth ★DUSt3R ★LeReS *We present MoGe, a powerful model for recovering 3D* *geometry from monocular open-domain images. Given a* *single image, our model directly predicts a 3D point map of* *the captured scene with an affine-invariant representation,* *which is agnostic to true global scale and shift. This new* *representation precludes ambiguous supervision in train-* *ing and facilitates effective geometry learning. Further-* *more, we propose a set of novel global and local geome-* *try supervision techniques that empower the model to learn* *high-quality geometry. These include a robust, optimal, and* *efficient point cloud alignment solver for accurate global* *shape learning, and a multi-scale local geometry loss pro-* *moting precise local geometry supervision. We train our* *model on a large, mixed dataset and demonstrate its strong* *generalizability and high accuracy. In our comprehen-* *sive evaluation on diverse unseen datasets, our model sig-* *nificantly outperforms state-of-the-art methods across all* *tasks, including monocular estimation of 3D point map,* *depth map, and camera field of view. Code and models can* *be found on ourproject page.*

### 1. Introduction

Estimating the 3D geometry of general scenes is a fun- damental task in computer vision. While 3D reconstruc- tion from a set of images has been extensively studied with Structure-from-Motion (SfM) [1,48] and Multi-View Stereo (MVS) [49,66] techniques, recovering the 3D ge- ometry from a single image in an arbitrary domain remains a significant challenge due to its highly ill-posed nature. For monocular geometry estimation (MGE), a common approach involves first estimating a depth map, subject to an unknown scale (and shift), and then combining it with cam- era intrinsics to recover the 3D shape via unprojection [42, 71]. For the former, significant advancements in monocular depth estimation (MDE) have been achieved in recent years by training on large-scale data [26,43,67,68,73] or lever- aging powerful pretrained models [20,23,29]. However, inferring camera intrinsics (*e.g*., focal length) from single images remains challenging due to the high degree of am- biguity when strong geometric cues are absent. Inaccurate camera parameters can lead to significant geometry distor- tions even when used with ground truth depth maps. Re- cently, DUSt3R [59] proposed predicting a 3D point map from an image by mapping each pixel to a free 3D point, thereby bypassing camera parameter estimation during ge- ometry recovery. Although the model can estimate monoc- ular geometry, it is primarily designed for multi-view sce- narios and is trained on image pair inputs. In this paper, we introduce a new *direct* 3D geometry es- timation method designed for single images in the open do-

Wild Camera★ ★ ★ZoeDepth Depth Anything V2 MiDaS V3.1★ ★Metric3D V2

★Marigold

Affine depth -inv.

Figure 2. Accuracy ranking of existing methods and ours in our ex-

tensive evaluation on monocular 3D point map (scale-invariant and affine-invariant), depth map (scale-invariant depth, affine-invariant depth, and affine-invariant disparity), and camera field of view. Outer methods rank higher. See our experiments for details.

main. Our model architecture is simple and straightforward: it directly predicts point maps from images, which can fur- ther derive depth map and camera focal length or FOV if needed. Different from DUSt3R [59] which uses scale- invariant point maps, our method predicts *affine-invariant* point maps, where the 3D points are subject to an unknown global scale as well as a 3D shift. This alteration is impor- tant as it eliminates the focal-distance ambiguity which is detrimental to the network training – we provide intuitive explanations with illustration as well as empirical evidence to support this advancement. More importantly, our findings indicate that *the design* *of training supervisions is crucial* to achieve robust and ac- curate point map prediction from single images. Similar to previous MDE approaches, a global scaling factor and translation are required to align predictions with ground truth during training. However, existing methods for this global alignment calculation are either sensitive to outliers or solved with coarse approximations, leading to subopti- mal supervision. We propose a *Robust, Optimal, and Ef-* *ficient (ROE)* global alignment solver to resolve scale and shift for the affine-invariant point map loss, which substan- tially improves the training effectiveness and final accuracy. On the other hand, the effective learning of local, region- specific geometry has been largely neglected previously. In monocular geometry estimation, the relative distances be- tween different objects can be ambiguous, which hinders precise local geometry learning when only a global align- ment is applied. In light of this, we propose a *multi-scale* *local geometry loss*, which penalizes local discrepancies in 3D point clouds under independent affine alignments. This design significantly boosted the accuracy of our local ge- ometry prediction. We train our model on large-scale data sourced from var- ious existing datasets. Our model demonstrates strong gen-

eralizability and accuracy in monocular geometry estima- tion for open-domain images (examples in Fig.1). Zero- shot evaluation on eight unseen datasets shows a significant improvement over previous MGE methods that output point clouds (over **35%** error reduction compared to the previous best). Moreover, our method significantly outperforms prior approaches that focus on the sub-tasks of MDE and camera FOV (**20%***∼***30%** error reduction for the former and over **20%** for the latter). Figure2shows that our method ranks top across all tasks and metrics in the evaluation. Our contributions are summarized as follows: 1) We pro- pose a new direct MGE method for open-domain images using affine-invariant point maps. 2) We establish a set of novel and effective global and local supervisions for robust and precise geometry recovery. 3) We demonstrate the sub- stantial enhancement in performance of our model over ex- isting methods on MGE, MDE, and camera FOV estimation across diverse datasets. We hope that our method serves as a versatile foundational model for monocular geometry tasks, facilitating applications such as 3D-aware image edit- ing, depth-to-image synthesis, novel view synthesis, and 3D scene understanding, as well as providing initial geometri- cal priors for video or multiview-based 3D reconstruction.

### 2. Related Work

**Monocular depth estimation.** As a crucial precursor to monocular geometry estimation, monocular depth estima- tion is a long-standing task and has been extensively studied in the past. A number of methods [4,5,18,26,36,42,73] have been developed to estimate depth with metric scale, which often rely heavily on data from specific sensors such as RGBD cameras, LiDAR, and calibrated stereo cameras. This dependence restricts their applicability to specific do- mains, such as indoor and driving scenes. In contrast, rela- tive depth estimation has garnered significant attention due to its ability to utilize a much wider variety of data, thereby exhibiting enhanced generalizability. A prevalent approach is to predict relative depth in an affine-invariant manner, ei- ther through direct regression [10,11,22,35,43,67,68,70] or generative modeling [20,23,29,56]. Despite the re- cent advancements in monocular depth estimation, recov- ering 3D shape from depth information always necessitates known camera intrinsics.

**Monocular point map estimation.** Compared to depth maps, 3D point map offers a more straightforward repre- sentation of 3D geometry. Monocular point map estimation aims to recover free 3D points for each pixel in a single im- age. Several approaches address this by predicting camera parameters alongside depth. For example, LeReS [71,72] introduced a two-stage pipeline with affine-invariant depth prediction followed by a point cloud module to recover shift and camera focal length. UniDepth [42] proposed a

self-promptable camera module that predicts dense camera representation to condition the subsequent depth estimation module. DUSt3R [59] employs an end-to-end model to di- rectly map two-view images to camera space point maps, adaptable to monocular scenarios by using two identical in- put images. However, its scale-invariant point map can suf- fer from the focal-distance ambiguity. In contrast, our ap- proach is designed to handle monocular input and we em- ploy affine-invariant point map with meticulously designed training supervisions for more effective geometry learning.

**Camera intrinsics estimation.** Estimating camera intrin- sics is a fundamental task in 3D vision. While traditional multiview camera calibration using special patterns is well studied [75], estimating camera intrinsics from single im- ages is ill-posed and remains challenging. For the latter, earlier works utilized known 3D shapes [61] or vanishing points [13]. More recently, learning-based approaches have been explored [7,28,33,63,77] to handle in-the-wild im- ages, but the results are far from satisfactory. In this work, we demonstrate that our camera parameter estimation de- rived from point maps can achieve state-of-the-art accuracy and offer remarkable generalizability.

**Large-scale data training for monocular geometry.** Re- cently, there has been an emerging trend [5,17,26,42,43, 59,67,68,73] in monocular geometry estimation and depth estimation that leverages large-scale datasets coupled with advanced network backbones [15,41,44] to enhance per- formance. MiDaS [43] marks an important milestone by mixing large datasets from various domains for training. Depth Anything [67] leverages both labeled and extensive unlabeled data to improve generalization. Its successive work [68] further highlights the importance of high-quality synthetic data for capturing rich shape details in depth pre- diction. These advancements underscore the vital impor- tance of large-scale training data in improving the perfor- mance of monocular geometry estimation.

### 3. Approach

Our method takes a single image as input and directly pre- dicts the geometry of the scene represented by 3D points. An overview of the method is shown in Fig.3.

#### 3.1. Affine-invariant point map

For an image I *∈* R *H×W ×*3, our model *Fθ*infers the 3D coordinates of the image pixels, represented by a point map P *∈* R *H×W ×*3, *i.e*.,:

#### Fθ: I 7→ P. (1)

The *X* and *Y* axes of the coordinate frame of P are aligned with the *u* and *v* axes in the image space respectively. Monocular geometry estimation often suffers from focal- distance ambiguity, as shown in Fig.4(a). To deal with

**Inference Post-processing**

Shift ො𝒕 **ViT** **Conv** Recover … … **Dec.** Focal 𝑓መ Input 𝐈 Affine-invariant

|Input 𝐈||Affine-invariant point map 𝐏|Mask 𝐌 ෡|𝐌 ෡ ⊙ 𝐏 ෡|||Camera-space point map|Depth map|
|---|---|---|---|---|---|---|---|---|
|||𝓛 = 𝝀|Training 𝓛 + 𝝀 𝓛𝑺 + 𝝀|+ 𝝀 𝓛𝑵 + 𝝀|𝓛𝑴||||
||Global||Local|||Normal||Mask|
||𝐌 ⊙ 𝐏 ෡|𝓢 ⊙ 𝐏 ෡|𝓢 ⊙ 𝐏 ෡|𝓢 ⊙ 𝐏|෡|𝒏 ෝ||𝐌 ෡|
|Prediction|𝓛|𝓛|𝓛|𝓛||𝓛||𝓛|
||𝐌 ⊙ 𝐏|𝓢 ⊙ 𝐏|𝓢 ⊙ 𝐏|𝓢 ⊙ 𝐏||𝒏||¬𝐌|
|Ground truth|||||||||

෡ Mask 𝐌෡ 𝐌෡ ⊙ 𝐏෡ Camera-space Depth map

𝑮 𝓛𝑮 + 𝝀𝑺𝟏 𝑺𝟏 𝑺𝟐 𝟐 𝑺𝟑𝓛𝑺𝟑 𝑵 𝑴

1 2 3 𝑗 𝑗 𝑗 𝒊

𝑮 𝑺𝟏 𝑺𝟐 𝑺𝟑 𝑵 𝑴

𝑗1 𝑗2 𝑗3 𝒊 𝐢𝐧𝐟

Figure 3. Method overview. Our model consists of a ViT encoder and a convolutional decoder. It predicts an affine-invariant point map as

well as a mask that excludes regions with undefined geometry (*e.g*., infinity). Depth, camera shift, and focal length can be further derived from the model output. For training, we design robust and effective supervisions focusing on both the global and local geometry.

**Input images Predictions GT geometry & cameras Scale-invariant Affine-invariant**

Similar images Similar predictions Distinct focal lengths & distances ✘Inconsistent supervision Consistent supervision

Definitive local shapes

**(a) (b) (c)**
✘Ambiguous relative distance

Figure 4. The focal-distance ambiguity and effects of different 3D point representations. (a) For similar Figure 5. The relative dis-

images captured with varying camera focal length and distance to the objects, perceiving their true camera tance ambiguity, where dis- setup is challenging and models often produce similar geometries. (b) Inconsistent supervision signals occur tances between objects can with only scale alignment. (c) Consistent geometry supervision with an additional translation alignment. be hard to predict.

this issue, we propose to predict *affine-invariant* point map *t* *′z* by minimizing the projection error: P, *i.e*., P is agnostic to the global scale *s ∈* R and offset*N* 2 2 3 *∼* X <u>fxifyi</u> t *∈* R and thus P = *s*P + t*, ∀s, ∀*t. Compared to the min *− ui*+ *− vi,* (2) scale-invariant supervision [59] shown in Fig.4(b), the dis-*f,t* *′z* *i*=1 *z* *i*+ *t* *′zz* *i*+ *t* *′z*

crepancies in focal lengths and camera positions between*′* where *tz*= *t /sz*. Eq.2can be efficiently solved with a fast the predictions and ground truth are amended by a transla- iterative optimization algorithm, which typically converges tion, as shown in Fig.4(c). This ensures consistent geom- within 10 iterations, taking approximately 3ms. More de- etry supervision when focal length is ambiguous, enabling tails about the solution can be found in the *suppl. materials*. more effective geometry learning. Following accepted practice, we assume that the camera With *t′z*recovered, the scale-invariant depth map and principal point coincides with image center and pixels are point map in the camera space can be obtained by adding square. In this case, t can be simplified with a *Z*-axis shift it to the *z* coordinates. *t* *z*(*i.e*., *tx*= *ty*= 0).

**Recovering camera focal and shift.** The affine-invariant point map representation can be used to recover both cam-**3.2. Training Objectives** era shift and focal length. Given the predicted 3D points (*xi,yi,zi*) and their corresponding 2D pixels (*ui,vi*), we **Global point map supervision.** Let pˆ*i*denote the pre- solve for camera focal length prediction *f* and *Z*-axis shift dicted 3D point for the *i*-th pixel, and p*i*the corresponding

ground truth. The global point map loss is defined as: X <u>1</u> *LG* = *∥s*pˆ*i*+ t *−* p*i∥*1*,* (3) *z* *i* *i∈M*

where *∥·∥*1denotes the *L₁* norm, *s* and t are the alignment parameters that transform the predicted affine-invariant point map to the ground-truth camera space, and *M* is the mask for regions with labels. The weighting term *z* <u>1</u> *i*, where *z* *i*is the *z*-coordinate of p*i*, is applied to balance the super- vision signal across extreme depths variations. To apply the global loss *LG* for training, we need to first determine *s* and t. Previous affine-invariant depth esti- mation methods often use rough alignment approximations, like using the median depth of two sets as an anchor to com- pute shift, followed by scaling [43,67,68], or normalizing depth range [20,29]. These simple strategies yield subopti- mal alignment and could lead to unsatisfactory supervision. In this work, we propose a solver to the *optimal* align- ment parameters. Specifically, we determine *s* *∗* and t *∗* via X <u>1</u> (*s* *∗* *,* t *∗* ) = argmin *∥s*pˆ*i*+ t *−* p*i∥*1*,* (4) *s,*t *i∈M* *z* *i*

where *tx*and *ty*are 0. One approach to solve this equation is to frame it as an absolute residuals optimization problem and use linear programming methods such as simplex algo- rithms and the interior-point method [8]. However, these methods are known to have high computational complexity, 3 often exceeding *O*(*N*), making them inefficient for net- work training with thousands of points, where computation can take seconds. Instead, we develop an efficient paral- lelized searching algorithm. With the fact that the optimum must occur when *sz*ˆ*k*+*tz*= *zk*for some index*k*, we replace *t* *z*with *s* and break the problem into a series of parallel one- dimensional subproblems. This reduces the complexity to 2 *O*(*N* log*N*) and enables efficient GPU-based training. A high-level description of this solution is presented in Algo- rithm1with more details in the *suppl. materials*. In practice, we found that Eq.4is still sensitive to out- liers occasionally even with the *L₁* error norm. If the model mistakenly predicts a close foreground edge pixel as a dis- tant background point, the objective would be dominated by this incorrect prediction. To further improve the robust- ness, we apply truncation min(*·,τ*) to the absolute resid- uals. While this truncation introduces non-convexity, our solver remains applicable with minor adjustments. In summary, our alignment solver is made *Robust*, *Opti-* *mal*, and *Efficient*, and hence we call it the *ROE* solver.

**Multi-scale local geometry loss.** In monocular geometry estimation, the relative distance between different objects can be ambiguous and difficult to predict, as illustrated in Fig.5. This hinders precise local geometry learning when a global alignment computed with all objects is applied. To enhance the supervision for local geometry, we propose a

**Algorithm 1** Overview of ROE alignment **function** SOLVESUBPROBLEM() Enumerate breakpoints of the piecewise linear function. Find extrema among the breakpoints by their derivatives. **return** *s*

(*k*) with smallest objective value *l*
(*k*) at extrema.
**end function**

**for** index *k* = 1 to *N* **do** *▷ parallel computation*

(*k*)
Formulate subproblem by substituting *tz* with *zk− s z*ˆ .*k*

(*k*) (*k*)
Solve scale *s* and *l* via SOLVESUBPROBLEM(). Obtain translation *t*

(*k*) as *z −s*
(*k*) *z*ˆ.
*z k k* **end for** Select optimal *s* *∗* and *t* *∗* *z*with smallest function value *l*

(*k*).
loss function which measures the accuracy of local sphere regions with *independent* alignment. Specifically, given a ground-truth 3D point p*j*as the an- chor, we first select the point set within the spherical region centered at p*j*, defined as:

*Sj*= *{i |∥*p*i−* p*j∥≤ rj,i ∈M},* (5) *√* <u>W2+H2</u> where*rj*is the radius. We set*rj*= *α·zj·* 2*·f* where*zj* is the *z*-coordinate of *pj*, *f* is the ground truth focal length, and *W* and *H* are image width and height. This way, the hyper-parameter *α ∈* (0*,*1) approximates the proportion of the projected sphere’s diameter to the image diagonal. Then, we apply the aforementioned ROE alignment solver to align the point maps with optimal (*s* *∗* *,* t *∗* ) and *j j* compute error. For each sphere scale parameter *α*, we sam- ple a set of anchor points *Hα*and compute the loss as X X X <u>1</u> *∗ ∗* *LS*(*α*)= *lSj*= *sj*pˆ*i*+ t*j−* p*i.* (6) *z* *i* 1 *j∈Hα j∈Hαi∈Sj*

In practice, we use three *α* scales from coarse to fine: 1*/*4, 1*/*16, and 1*/*64, denoted as *LS*1, *LS*2and *LS*3, respectively.

**Normal loss.** For better surface quality, we additionally supervise the normal computed from the predicted point map with the ground truth: X *LN* = ∠(nˆ*i,* n*i*)*,* (7) *i∈M*

where the normal nˆ*i*of pixel *i* is obtained from the cross product of its adjacent edges on the image grid, n*i*is the ground truth, and ∠(*·, ·*) measures the angle difference be- tween the two vectors.

**Mask loss.** The infinity region (*e.g*., sky) in outdoor scenes and plain backgrounds in object-only images have undefined geometry. We use a single-channel output head for predicting the mask Mˆ *∈* R *H×W* of valid points:

ˆ2 *LM*= *∥*M *−* (1 *−* Minf)*∥ ,*2(8)

where Minfis the infinity mask label. For synthetic data, the ground truth masks are readily available, while for real

outdoor scenes, we use SegFormer [64] to obtain sky masks. At inference time, the predicted mask is binarized with a threshold of 0*.*5. The recovery of focal and shift involves masked points, excluding the regions of invalid geometry.

#### 3.3. Training data

We collected 21 publicly available datasets from various do- mains [3,12,14,19,21,25,27,34,35,37,40,45,47,51, 57,58,60,62,69,74,76], totaling around 9 million frames for training our model. Since the quality of data labels are varied, we apply tailored loss function combinations for dif- ferent data sources. Higher-quality and less noisy data are assigned with finer-level loss functions. To address imper- fections and potential outliers in real labels, we exclude the top 5% of per-pixel loss values for real data. Following DI- NOv2 [41], we balance the weights of collected datasets via image retrieval from a curated image set. Details of training datasets can be found in the *suppl. materials*.

### 4. Experiments

**Implementation details.** We use a ViT [15] encoder pre- trained with DINOv2 [41] and a lightweight CNN-based upsampler as the decoder. The encoder and decoder have initial learning rates of 5 *×* 10 *−*6 and 5 *×* 10 *−*5, with a de- cay factor of 5 every 100K iterations. The training batch size is 256. Our model is trained on images with aspect ratios uniformly sampled between 1 : 2 and 2 : 1, and pixel counts from 250K to 500K. We apply several image augmentations, including color jittering, Gaussian blurring, JPEG compression-decompression, and random cropping. The principal point of the cropped image is aligned with its center using a perspective crop.

#### 4.1. Quantitative Evaluations.

We assess the zero-shot performance of our model and com- pare it to several state-of-the-art methods for monocular point map estimation [42,59,71] and depth map estima- tion [5,6,20,26,29,67,68]. To ensure a fair compari- son, all methods utilize ViT-Large [15] as backbone, except for LeReS (using ResNeXt101 [65]) as well as GeoWizard and Marigold (using StableDiffusion V1 [46]). The scale and shift of the predicted point maps and depth maps are aligned with the ground truth when necessary. We also evaluate the FOV prediction results against several recent methods [28,77]. As detailed below, *our method achieves* 1 *superior results across various tasks and benchmarks*.

**Point map estimation.** We evaluate the accuracy of point map estimation on eight diverse datasets: NYUv2 [39], KITTI [53], ETH3D [50], iBims-1 [31], Sintel [9],

1 In this revision, we corrected evaluation numbers due to a minor er- ror in the evaluation data. The overall rankings and conclusions are not affected. All results are reproducible using the released code.

Google Scanned Objects (GSO) [16], DDAD [24], and DIODE [54]. These datasets represent a wide range of do- mains, including indoor scenes, street views, object scans, and synthetic movies. The raw datasets are processed for reliable evaluation (*e.g*., sky regions in Sintel dataset and boundary artifacts in the DIODE are removed; see the *suppl. material* for more preprocessing details). For eval- uation metrics, we use the relative point error Rel p, *i.e*., p *∥*pˆ *−* p*∥*2*/∥*p*∥*2, and the percentage of inliers *δ₁* with *∥*pˆ *−* p*∥*2*/*min(*∥*p*∥*2*, ∥*pˆ*∥*2) *<* 0*.*25. We compare our method with LeReS [71], UniDepth [42], and DUSt3R [59], evaluating both camera- space scale-invariant point maps and affine-invariant point maps. Table1demonstrates that our method outperforms existing methods, achieving the lowest average Rel p and highest *δ* p 1for both point map representations. To further assess local geometry accuracy, we evaluate affine-invariant point maps within regions defined by ob- ject segmentation masks, sourced from dataset labels or Segment Anything [30]. Due to the need for high-quality ground truth and scenes with multiple objects, NYUv2, KITTI, and GSO are excluded. Our method achieves a substantial reduction in region-wise Rel p, lowering the er- ror from 7*.*97 to 5*.*50—an approximately 30% improvement over previous methods.

**Depth map estimation.** We employ the same bench- marks used for point map evaluation to assess the depth estimation accuracy. We adopt absolute relative error Rel d, *i.e*., *|z − z*ˆ*|/z*, and percentage of inliers *δ₁* *d* with max(*z/z,*ˆ *z/z* ˆ) *<* 1*.*25 as the evaluation metrics. For a comprehensive comparison with different monoc- ular depth estimation methods, we evaluate scale-invariant depth [5,26,42,59,71], affine-invariant depth [20,29], and affine-invariant disparity [6,67,68]. The *z* coordinates in our predicted point maps represent affine-invariant depth, which can be converted to scale-invariant depth using the shift derived from Eq.2. For affine-invariant disparity, we take the inverse of our scale-invariant depth predictions. As shown in Table2, our method exhibits the lowest average Rel d for all depth representations. Since existing methods with scale-invariant depth representations can be converted to affine-invariant depth and disparity, we provide an exten- sive comparison in the *suppl. materials*, where our method also significantly outperforms the others.

**Camera FOV estimation.** To evaluate the accuracy of the estimated camera FOV, we select benchmarks with vertical FOV *≥* 45 *◦*, including NYUv2 [39], ETH3D [50] and iBims-1 [31], and augment the FOV by randomly center cropping 50%*∼*100% from the original images. All input images are undistorted with a centered principal point. We compare our model to point map estimation meth- ods [42,59,71] as well as two learning-based camera cal-

|Method|||||GSO|Sintel|||Average||
|---|---|---|---|---|---|---|---|---|---|---|
||RelNYUv2 δ|RelKITTI δ|RelETH3D δ|ReliBims-1 δ Scale-invariant point map|Rel δ|Rel δ|RelDDAD δ|RelDIODE δ|Rel δ|Rank|
|LeReS|16.9 76.0|31.6 28.4|17.1 75.8|18.5 72.2|14.7 76.0|38.6 30.6|32.0 39.4|27.6 46.4|24.6 55.6|3.94|
|DUSt3R|5.53 97.1|15.2 87.9|10.7 90.6|6.18 95.4|4.54 99.3|34.8 50.3|21.4 70.1|12.4 86.7|13.8 84.7|2.75|
|UniDepth|5.33 98.4|5.96 98.5|18.5 77.6|5.29 97.4|6.58 99.6|33.0 48.9|11.4 90.2|12.3 91.0|12.3 87.7|2.09|
|Ours|4.86 98.4|5.47 97.4|4.58 98.9|4.63 97.1 Affine-invariant point map|2.58 100|22.3 69.5|12.3 90.3|6.58 94.5|7.91 93.3|1.22|
|LeReS|9.51 91.4|26.1 49.1|14.7 79.6|11.0 88.6|8.91 95.2|29.7 55.5|29.4 46.7|15.1 80.1|18.1 73.3|3.94|
|DUSt3R|4.45 97.4|12.7 83.3|7.27 95.0|5.04 96.0|3.07 99.6|30.3 56.6|19.7 71.2|8.97 88.7|11.4 86.0|2.94|
|UniDepth|3.93 98.4|4.29 98.6|12.2 89.6|4.65 98.0|2.99 99.8|28.5 58.4|10.3 90.5|8.56 90.9|9.43 90.5|1.81|
|Ours|3.68 98.3|4.86 97.2|3.57 99.0|3.61 97.3 Local point map|1.14 100|16.8 77.8|10.5 91.4|4.37 96.4|6.07 94.7|1.31|
|LeReS|--|--|9.32 91.9|8.57 93.2|--|13.3 84.8|10.7 88.9|11.6 88.2|10.7 89.4|3.80|
|DUSt3R|--|--|6.05 94.8|5.44 95.9|--|11.8 87.0|9.24 90.8|7.32 93.1|7.97 92.3|2.30|
|UniDepth|--|--|8.61 92.6|5.92 96.0|--|13.4 84.3|8.18 92.0|9.95 90.0|9.21 91.0|2.90|
|Ours|--|--|3.21 98.1|4.16 96.8|--|8.63 92.7|6.74 94.3|4.78 96.3|5.50 95.6|1.00|

p *↓* p *↑* p *↓* p *↑* p *↓* p *↑* p *↓* p *↑* p *↓* p *↑* p *↓* p *↑* p *↓* p *↑* p *↓* p *↑* p *↓* p *↑ ↓* 1 1 1 1 1 1 1 1 1

Table 1. Quantitative results for point map estimation. Rel

p and *δ₁* p are in percentage. The best values are highlighted in **bold**, and the second-best ones are underlined. Local point map accuracy is evaluated on affine-invariant point maps within local object regions

|Method||||iBims-1|GSO|Sintel|||Average||
|---|---|---|---|---|---|---|---|---|---|---|
||RelNYUv2 δ|Rel.KITTI δ|RelETH3D δ|Rel. δ Scale-invariant depth map|Rel δ|Rel δ|RelDDAD δ|RelDIODE δ|Rel δ|Rank|
|LeReS|12.1 82.6|19.2 64.8|14.2 78.4|14.0 78.8|13.6 77.9|30.5 52.1|26.5 52.0|18.2 69.6|18.5 69.5|7.31|
|ZoeDepth|5.6296.3|7.2791.9|10.4 87.3|7.45 93.2|3.23 99.9|27.4 61.8|17.0 72.8|11.3 85.2|11.286.15.50||
|DUSt3R|4.40 97.1|7.81 90.6|6.04 95.7|4.98 95.8|3.27 99.5|31.1 57.2|18.6 73.3|8.91 88.8|10.6 87.2|5.00|
|Metric3D V2|4.69 97.4|4.00 98.5|3.84 98.5|4.23 97.7|2.46 99.9|20.7 69.8|7.4194.6|3.29 98.4|6.3394.32.07||
|UniDepth|3.86 98.4|3.73 98.6|5.67 97.0|4.79 97.4|4.18 99.7|28.3 58.8|10.1 90.5|6.83 92.8|8.43 91.6|3.00|
|DA V1|4.7797.5|5.6195.6|9.41 88.9|5.53 95.8|5.49 99.3|28.3 56.7|13.2 81.5|10.3 87.5|10.387.95.67||
|DA V2|5.03 97.3|7.23 93.7|6.12 95.5|4.32 97.9|4.38 99.3|23.0 65.2|14.7 78.0|7.95 90.0|9.09 89.6|4.06|
|Ours|3.44 98.4|4.25 97.8|3.36 98.9|3.46 97.0 Affine-invariant depth map|1.47 100|19.3 73.4|9.17 90.5|4.89 94.7|6.17 93.8|1.62|
|Marigold|4.63 97.3|7.29 93.8|6.08 96.3|4.35 97.2|2.78 99.9|21.2 75.0|14.6 80.5|6.34 94.3|8.41 91.8|2.25|
|GeoWizard|4.69 97.4|8.14 92.5|6.90 94.0|4.50 97.1|2.00 99.9|17.8 76.2|16.5 75.7|7.03 92.7|8.44 90.7|2.69|
|Ours|2.92 98.6|3.94 98.0|2.69 99.2|2.74 97.9 Affine-invariant disparity map|0.94 100|13.0 83.2|8.40 92.1|3.16 97.5|4.72 95.8|1.00|
|MiDaS V3.1|4.58 98.1|6.25 94.7|5.77 96.8|4.73 97.4|1.86 100|21.3 73.1|14.5 82.6|6.05 94.9|8.13 92.2|3.69|
|DA V1|4.20 98.4|5.40 97.0|4.68 98.2|4.18 97.6|1.54 100|20.1 77.6|12.7 86.9|5.69 95.7|7.31 93.9|2.31|
|DA V2|4.14 98.3|5.61 96.7|4.71 97.9|3.47 98.5|1.24 100|21.4 72.8|13.1 86.4|5.29 96.1|7.37 93.3|2.56|
|Ours|3.38 98.6|4.05 98.1|3.11 98.9|3.23 98.0|0.96 100|18.4 79.5|8.99 91.5|3.98 97.2|5.76 95.2|1.06|

d *↓* d *↑ ↓* d *↑* d *↓* d *↑ ↓* d *↑* d *↓* d *↑* d *↓* d *↑* d *↓* d *↑* d *↓* d *↑* d *↓* d *↑ ↓* 1 1 1 1 1 1 1 1 1

Table 2. Quantitative results for depth map estimation.Gray numbersdenote models trained on respective benchmarks and thus excluded

from ranking. A more extensive comparison can be found in the *suppl. materials*.

|||||Point||Depth||Disparity|
|---|---|---|---|---|---|---|---|---|
||NYUv2|Ablation|Scale-inv.|Affine-inv.|Local|Scale-inv.|Affine-inv.|Affine-inv.|
|Method|||||||||
|||SI-Log depth|11.2 88.7|9.09 90.6|9.19 91.2|8.94 90.1|7.27 92.6|8.23 92.1|
|Perspective 5.38 4.39 13.6 11.9 10.6 9.30 9.86 8.53 5.00||Affine-inv. depth|29.9 51.4|29.0 52.7|12.2 86.0|28.9 52.7|6.18 93.9|15.9 76.6|
|WildCam|3.82 3.20 7.70 5.81 9.48 9.08 7.00 6.03 3.00|ROE scale-inv.|10.3 89.8|8.34 91.6|8.59 91.9|8.27 90.9|6.73 93.2|7.90 92.6|
|LeReS|19.4 19.6 8.26 7.19 18.4 17.5 15.4 14.8 5.53|L2 affine-inv.|13.5 84.2|10.3 88.2|9.48 91.0|11.1 85.7|8.03 91.2|9.37 90.5|
|DUSt3R|2.57 1.86 5.77 3.60 3.83 2.53 4.06 2.66 1.67|Med. affine-inv.|10.9 89.0|8.97 90.7|9.44 90.7|9.10 89.8|7.50 92.4|8.74 91.8|
|UniDepth|7.56 4.31 10.7 9.96 11.9 5.96 10.1 6.74 4.50|ROE affine-inv.|9.84 90.3|7.88 92.1|7.62 93.3|7.91 91.2|6.29 93.7|7.43 93.2|
|||Full w/o trunc.|9.81 90.5|7.91 91.7|7.12 93.8|7.92 91.3|6.31 93.5|7.45 93.1|
|Ours|3.41 3.21 2.50 1.54 2.81 1.89 2.91 2.21 1.50|Full w/o LS|9.98 90.3|7.94 92.1|7.47 93.4|7.94 91.2|6.30 93.6|7.47 93.2|
|||Full|9.78 90.6|7.83 92.1|7.16 93.8|7.82 91.3|6.20 93.7|7.30 93.3|

ETH3D iBims-1 Average Mean*↓* Med.*↓* Mean Med.*↓* Mean*↓* Med.*↓* Mean*↓* Med.*↓* Rank*↓* Relp*↓ δ*1p *↑* Relp*↓ δ*1p *↑* Relp*↓ δ*1p *↑* Reld*↓ δ*1d *↑* Reld*↓ δ*1d *↑* Reld*↓ δ*1d *↑*

Table 3. Evaluation results for camera FOV in de-

grees. Table 4. Quantitative ablation study results. All experiments are conducted with a ViT-Base encoder. The first six rows are trained with global loss only.

ibration methods: Perspective Fields [28] and Wild Cam-tion and point map estimation methods. era [77]. For fair zero-shot evaluation, we utilize their pre- trained checkpoints on in-the-wild datasets. Table3reports **4.2. Qualitative comparison** mean and median FOV errors. Our method achieves a mean We show visual results of disparity and point maps of dif- error of 2*.*91 *◦* and a median error of 2*.*21 *◦* on average of the ferent methods in Fig.7. For depth prediction methods three FOV benchmarks, outperforming both camera calibra- without camera parameters estimation, we lift their results

|w/o ℒ𝑆|a large value for ∞|w/ mask & ℒ𝑀|
|---|---|---|
|(a)|(b)||

input image *w/o* ℒ𝑆 *&* ℒ𝑁 full *w/o* mask

Figure 6. Qualitative ablation study results. All experiments use the ViT-Base backbone for the encoder. **(a)** Surface visualization of the

impact of *LS* and *LN*. Removing either leads to noisy surfaces and poor geometry. **(b)** Depth visualization for the ablation of valid region mask prediction. Our method correctly predicts the sky regions, for which the predicted point values will be erroneous if the masks are removed. Supervising infinity regions by assigning a large distance label can negatively affect the foreground prediction accuracy.

to 3D points using ground-truth intrinsics when available. Upon visual inspection, our method produces significantly less-distorted point maps compared to others, demonstrat- ing its superior generalizability and accuracy. For example, as shown in the third row of Fig.7, other methods show noticeable stretching and deformation of the car shape. Al- though [68] produces visually sharp disparity maps, dis- torted 3D geometry is evident even with ground-truth cam- era intrinsics. In contrast, our method yields a more regular and well-formed car structure. More comparisons can be found in the *suppl. materials*.

#### 4.3. Ablation Study

We conducted comprehensive ablation studies on our point map representation, alignment strategy and loss function designs. ViT-base backbones are used here for efficiency.

**Point map representation.** We compare our affine- invariant point map representation with three variants: (1) a scale-invariant depth map and a dense ray map for cam- era parameters, (2) an affine-invariant depth map with a shift scalar and a dense ray map, and (3) a scale-invariant point map in camera space. The first variant uses SI-log loss [18] for the depth map and MSE for the ray map, fol- lows UniDepth [42]. The second employs ROE alignment with *L₁* loss for the affine depth, *L₁* loss for shift, and MSE for the ray map. The third applies the ROE alignment to solve the scale parameter only. Table4(row 1-3) shows that our affine-invariant representation outperforms the oth- ers across various tasks. Notably, while the affine-invariant depth representation slightly improves specific metrics, all others drop dramatically. Our representation excels by ef- fectively resolving focal-distance ambiguity, a key issue that hampers effectively training of the other three variants.

**ROE alignment.** To assess the effectiveness of our op- timal alignment, we compare it to least-squares alignment for *L₂* loss and a previously-used median-based align- ment [43] for *L₁* loss. The former has an analytic solution, while the latter normalizes predicted and ground-truth point maps by their median: (p *−* t(P))*/s*(P), where t(P) = P*N* (0*,*0*,*median(Z)), and *s*(P) = *N i*=1 *∥*p*i−*t(P)*∥*1. As

shown in the Table4(row 4-6), our proposed alignment method consistently outperforms the other two strategies across all tasks by large margins, highlighting its critical role. We also ablate the truncation of alignment objective residuals, which degrades performance (Table4row 7).

**Multi-scale local geometry loss.** As shown in Table4 (row 8), the quantitative results without *LS* suffer a signif- icant drop on local geometry metrics. This demonstrates the effectiveness of *LS* in enhancing local geometry accu- racy and compensating for the insufficient supervision of the global loss due to the ambiguous relative positions be- tween distant scene objects. Fig.6(a) further demonstrates the effectiveness of the proposed local geometry loss.

**Infinity mask.** Our method predicts a valid region mask to handle infinity regions. As shown in Fig.6(b), it cor- rectly identifies sky regions, where the output point values would be erroneous if the masks were removed. Another strategy for addressing infinity is to assign a large value, *i.e*., 1*,*000*×* the average distance, to these regions. Fig.6(b) shows that this strategy compromises the foreground accu- racy as the network struggles with large values for infinity. Incorporating a separate mask prediction effectively han- dles infinity while retaining accurate foreground geometry.

### 5. Conclusion

We have presented a method for accurate monocular geom- etry estimation of open-domain images. Our key insight is to design effective supervision which is largely neglected by previous methods. We introduce an affine-invariant point map representation which precludes ambiguous supervision during training. A robust, optimal, and efficient point map alignment solver is introduced for effective shape learning, along with a multi-scale geometry loss for precise local su- pervision. Trained on a large dataset collection, our model demonstrates strong generalizability and significantly out- performs previous methods across various tasks and bench- marks. We believe our method marks a significant advance- ment in monocular 3D geometry estimation and can serve as a robust foundational model for various applications.

Input Ground truth Ours LeReS UniDepth DUSt3R Metric3D V2* DA V2*

*Not* *available*

*Not* *available*

*Not* *available*

*Not* *available*

Figure 7. Qualitative comparison of point and disparity maps. The first three cases are from our test datasets, while the last two rows are

random “in-the-wild” images. *: for methods without camera intrinsics prediction, we use ground-truth camera intrinsics (and disparity shifts) to lift their results into 3D points when available, or use our intrinsics prediction for the “in-the-wild” images. ***Best view with zoom.***

### References Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe,

Daniel Kurz, Arik Schwartz, and Elad Shulman. ARK- [1]Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Si- itscenes-a diverse real-world dataset for 3d indoor scene mon, Brian Curless, Steven M Seitz, and Richard Szeliski. understanding using mobile RGB-d data. In *Thirty-fifth Con-* Building rome in a day. *Communications of the ACM*, 54 *ference on Neural Information Processing Systems Datasets* (10):105–112, 2011.2 *and Benchmarks Track (Round 1)*, 2021.6,4 [2]Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge [4]Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. on single image super-resolution: Dataset and study. In *The* Adabins: Depth estimation using adaptive bins. In *Proceed-* *IEEE Conference on Computer Vision and Pattern Recogni-ings of the IEEE/CVF conference on computer vision and* *tion (CVPR) Workshops*, 2017.5 *pattern recognition*, pages 4009–4018, 2021.3 [3]Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, [5]Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter

Wonka, and Matthias Muller. ¨ Zoedepth: Zero-shot trans- fer by combining relative and metric depth. *arXiv preprint* *arXiv:2302.12288*, 2023.3,6 [6]Reiner Birkl, Diana Wofk, and Matthias Muller. ¨ Midas v3. 1–a model zoo for robust monocular relative depth estima- tion. *arXiv preprint arXiv:2307.14460*, 2023.6 [7]Oleksandr Bogdan, Viktor Eckstein, Francois Rameau, and Jean-Charles Bazin. Deepcalib: A deep learning approach for automatic intrinsic calibration of wide field-of-view cam- eras. In *Proceedings of the 15th ACM SIGGRAPH European* *Conference on Visual Media Production*, pages 1–10, 2018. 3 [8]Stephen Boyd and Lieven Vandenberghe. *Convex Optimiza-* *tion*. Cambridge University Press, New York, NY, USA,

2004.5
[9]D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black. A naturalistic open source movie for optical flow evaluation. In *European Conf. on Computer Vision (ECCV)*, pages 611–

625. Springer-Verlag, 2012.6,4
[10]Weifeng Chen, Zhao Fu, Dawei Yang, and Jia Deng. Single- image depth perception in the wild. In *Advances in Neu-* *ral Information Processing Systems*. Curran Associates, Inc.,

2016.3
[11]Weifeng Chen, Shengyi Qian, David Fan, Noriyuki Ko- jima, Max Hamilton, and Jia Deng. Oasis: A large-scale dataset for single image 3d in the wild. In *Proceedings of* *the IEEE/CVF Conference on Computer Vision and Pattern* *Recognition (CVPR)*, 2020.3 [12]Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In *Proceedings of* *the IEEE/CVF Conference on Computer Vision and Pattern* *Recognition*, pages 13142–13153, 2023.6,4 [13]Jonathan Deutscher, Michael Isard, and John MacCormick. Automatic camera calibration from a single manhattan im- age. In *Computer Vision—ECCV 2002: 7th European* *Conference on Computer Vision Copenhagen, Denmark,* *May 28–31, 2002 Proceedings, Part IV 7*, pages 175–188. Springer, 2002.3 [14]Digital Image Media Laboratory (DIML) and Computer Vi- sion Laboratory (CVL). Diml/cvl rgb-d dataset: 2m rgb-d images of natural indoor and outdoor scenes. https:// dimlrgbd.github.io/downloads/technical_ report.pdf.6,4 [15]Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl- vain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*, 2021.3,6 [16]Laura Downs, Anthony Francis, Nate Koenig, Brandon Kin- man, Ryan Hickman, Krista Reymann, Thomas B. McHugh, and Vincent Vanhoucke. Google scanned objects: A high- quality dataset of 3d scanned household items, 2022.6,4 [17]Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. Omnidata: A scalable pipeline for making multi- task mid-level vision datasets from 3d scans. In *Proceedings*

*of the IEEE/CVF International Conference on Computer Vi-* *sion*, pages 10786–10796, 2021.3 [18]David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep net- work. *Advances in neural information processing systems*, 27, 2014.3,8,4 [19]Michael Fonder and Marc Van Droogenbroeck. Mid-air: A multi-modal dataset for extremely low altitude drone flights. In *Conference on Computer Vision and Pattern Recognition* *Workshop (CVPRW)*, 2019.6,4 [20]Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowiz- ard: Unleashing the diffusion priors for 3d geometry estima- tion from a single image. *arXiv preprint arXiv:2403.12013*,

2024.2,3,5,6
[21]Jakob Geyer, Yohannes Kassahun, Mentar Mahmudi, Xavier Ricou, Rupesh Durgesh, Andrew S. Chung, Lorenz Hauswald, Viet Hoang Pham, Maximilian Muhlegg, Sebas-¨ tian Dorn, Tiffany Fernandez, Martin Janicke, Sudesh Mi-¨ rashi, Chiragkumar Savani, Martin Sturm, Oleksandr Voro- biov, Martin Oelker, Sebastian Garreis, and Peter Schuberth. A2D2: Audi Autonomous Driving Dataset. 2020.6,4 [22]Clement Godard, ´ Oisin Mac Aodha, Michael Firman, and Gabriel J Brostow. Digging into self-supervised monocular depth estimation. In *Proceedings of the IEEE/CVF inter-* *national conference on computer vision*, pages 3828–3838,

2019.3
[23]Ming Gui, Johannes S Fischer, Ulrich Prestel, Pingchuan Ma, Dmytro Kotovenko, Olga Grebenkova, Stefan Andreas Baumann, Vincent Tao Hu, and Bjorn Ommer. ¨ Depthfm: Fast monocular depth estimation with flow matching. *arXiv* *preprint arXiv:2403.13788*, 2024.2,3 [24]Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raven- tos, and Adrien Gaidon. 3d packing for self-supervised monocular depth estimation. In *IEEE Conference on Com-* *puter Vision and Pattern Recognition (CVPR)*, 2020.6,4 [25]Jose L. Gomez, ´ Manuel Silva, Antonio Seoane, Agnes` Borras, ´ Mario Noriega, German´ Ros, Jose A. Iglesias- Guitian, and Antonio M. Lopez. ´ All for one, and one for all: Urbansyn dataset, the third musketeer of synthetic driv- ing scenes, 2023.6,4 [26]Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geomet- ric foundation model for zero-shot metric depth and surface normal estimation. *arXiv preprint arXiv:2404.15506*, 2024. 2,3,6,5 [27]Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra Ahuja, and Jia-Bin Huang. Deepmvs: Learning multi-view stereopsis. In *IEEE Conference on Computer Vision and Pat-* *tern Recognition (CVPR)*, 2018.6,4 [28]Linyi Jin, Jianming Zhang, Yannick Hold-Geoffroy, Oliver Wang, Kevin Blackburn-Matzen, Matthew Sticha, and David F Fouhey. Perspective fields for single image cam- era calibration. In *Proceedings of the IEEE/CVF Conference* *on Computer Vision and Pattern Recognition*, pages 17307– 17316, 2023.3,6,7

[29]Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Met- zger, Rodrigo Caye Daudt, and Konrad Schindler. Repurpos- ing diffusion-based image generators for monocular depth estimation. In *Proceedings of the IEEE/CVF Conference* *on Computer Vision and Pattern Recognition*, pages 9492– 9502, 2024.2,3,5,6 [30]Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White- head, Alexander C Berg, Wan-Yen Lo, et al. Segment any- thing. In *Proceedings of the IEEE/CVF International Con-* *ference on Computer Vision*, pages 4015–4026, 2023.6 [31]Tobias Koch, Lukas Liebel, Marco Korner, ¨ and Friedrich Fraundorfer. Comparison of monocular depth estimation methods using geometrically relevant metrics on the ibims-1 dataset. *Computer Vision and Image Understanding (CVIU)*, 191:102877, 2020.6,4 [32]Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Ui- jlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. *IJCV*, 2020.3 [33]Jinwoo Lee, Hyunsung Go, Hyunjoon Lee, Sunghyun Cho, Minhyuk Sung, and Junho Kim. Ctrl-c: Camera calibration transformer with line-classification. In *Proceedings of the* *IEEE/CVF International Conference on Computer Vision*, pages 16228–16237, 2021.3 [34]Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhen- zhi Wang, Dahua Lin, and Bo Dai. Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond. In *Proceedings of the IEEE/CVF International Conference on* *Computer Vision*, pages 3205–3215, 2023.6,4 [35]Zhengqi Li and Noah Snavely. Megadepth: Learning single- view depth prediction from internet photos. In *Computer* *Vision and Pattern Recognition (CVPR)*, 2018.3,6,4 [36]Zhenyu Li, Xuyang Wang, Xianming Liu, and Junjun Jiang. Binsformer: Revisiting adaptive bins for monocular depth estimation. *IEEE Transactions on Image Processing*, 2024. 3 [37]Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nali- vayko, and Andres Bruhn. ´ Spring: A high-resolution high- detail dataset and benchmark for scene flow, optical flow and stereo. In *Proc. IEEE/CVF Conference on Computer Vision* *and Pattern Recognition (CVPR)*, 2023.6,4 [38]Jorge J More.´ The levenberg-marquardt algorithm: imple- mentation and theory. In *Numerical analysis: proceedings* *of the biennial Conference held at Dundee, June 28–July 1,* *1977*, pages 105–116. Springer, 2006.1 [39]Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In *ECCV*, 2012.6,4 [40]Simon Niklaus, Long Mai, Jimei Yang, and Feng Liu. 3d ken burns effect from a single image. *ACM Transactions on* *Graphics*, 38(6):184:1–184:15, 2019.6,4 [41]Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy ´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.

Dinov2: Learning robust visual features without supervision. *arXiv preprint arXiv:2304.07193*, 2023.3,6 [42]Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In *Proceed-* *ings of the IEEE/CVF Conference on Computer Vision and* *Pattern Recognition*, pages 10106–10116, 2024.2,3,6,8,5 [43]Rene´ Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. *IEEE transactions on pattern analysis and machine* *intelligence*, 44(3):1623–1637, 2020.2,3,5,8 [44]Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. ´ Vi- sion transformers for dense prediction. In *Proceedings of* *the IEEE/CVF international conference on computer vision*, pages 12179–12188, 2021.3 [45]Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic syn- thetic dataset for holistic indoor scene understanding. In *International Conference on Computer Vision (ICCV) 2021*,

2021.6,4
[46]Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn-¨ thesis with latent diffusion models, 2021.6 [47]German Ros, Laura Sellart, Joanna Materzynska, David Vazquez, and Antonio M. Lopez. The synthia dataset: A large collection of synthetic images for semantic segmenta- tion of urban scenes. In *The IEEE Conference on Computer* *Vision and Pattern Recognition (CVPR)*, 2016.6,4 [48]Johannes L Schonberger and Jan-Michael Frahm. Structure- from-motion revisited. In *Proceedings of the IEEE con-* *ference on computer vision and pattern recognition*, pages 4104–4113, 2016.2 [49]Johannes L Schonberger, ¨ Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. Pixelwise view selection for unstructured multi-view stereo. In *Computer Vision–ECCV* *2016: 14th European Conference, Amsterdam, The Nether-* *lands, October 11-14, 2016, Proceedings, Part III 14*, pages 501–518. Springer, 2016.2 [50]Thomas Schops, Torsten Sattler, and Marc Pollefeys. ¨ BAD SLAM: Bundle adjusted direct RGB-D SLAM. In *Confer-* *ence on Computer Vision and Pattern Recognition (CVPR)*,

2019.6,4
[51]Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Et- tinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In *Proceedings of the IEEE/CVF Conference* *on Computer Vision and Pattern Recognition (CVPR)*, 2020. 6,4 [52]Michal Tolgyessy, Martin Dekan, L’ubo ¨ s Chovanec, and Pe-ˇ ter Hubinsky. Evaluation of the azure kinect and its compar-` ison to kinect v1 and kinect v2. *Sensors*, 21(2):413, 2021.

[53]Jonas Uhrig, Nick Schneider, Lukas Schneider, Uwe Franke, Thomas Brox, and Andreas Geiger. Sparsity invariant cnns. In *International Conference on 3D Vision (3DV)*, 2017.6,4 [54]Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z. Dai, Andrea F. Daniele, Moham- madreza Mostajabi, Steven Basart, Matthew R. Walter, and Gregory Shakhnarovich. DIODE: A Dense Indoor and Out- door DEpth Dataset. *CoRR*, abs/1908.00463, 2019.6,4 [55]Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stefan J. van der Walt, Matthew Brett, Joshua Wil-´ son, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, ˙Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, De- nis Laxalde, Josef Perktold, Robert Cimrman, Ian Henrik- sen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antonio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, ˆ and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algo- rithms for Scientific Computing in Python. *Nature Methods*, 17:261–272, 2020.1 [56]Qiang Wan, Zilong Huang, Bingyi Kang, Jiashi Feng, and Li Zhang. Harnessing diffusion models for visual perception with meta prompts. *arXiv preprint arXiv:2312.14733*, 2023. 3 [57]Kaixuan Wang and Shaojie Shen. Flow-motion and depth network for monocular stereo and beyond. *CoRR*, abs/1909.05452, 2019.6,4 [58]Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng, Kaiyong Zhao, and Xiaowen Chu. IRS: A large synthetic indoor robotics stereo dataset for disparity and surface nor- mal estimation. *CoRR*, abs/1912.09678, 2019.6,4 [59]Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vi- sion made easy. In *CVPR*, 2024.2,3,4,6,5 [60]Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Se- bastian Scherer. Tartanair: A dataset to push the limits of visual slam. 2020.6,4 [61]Marta Wilczkowiak, Edmond Boyer, and Peter Sturm. Cam- era calibration and 3d reconstruction from single images us- ing parallelepipeds. In *IEEE International Conference on* *Computer Vision*, pages 142–148. IEEE, 2001.3 [62]Benjamin Wilson, William Qi, Tanmay Agarwal, John Lam- bert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan, Rat- nesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, Deva Ramanan, Peter Carr, and James Hays. Argoverse 2: Next generation datasets for self-driving perception and fore- casting. In *Proceedings of the Neural Information Process-* *ing Systems Track on Datasets and Benchmarks (NeurIPS* *Datasets and Benchmarks 2021)*, 2021.6,4 [63]Scott Workman, Connor Greenwell, Menghua Zhai, Ryan Baltenberger, and Nathan Jacobs. Deepfocal: A method for direct focal length estimation. In *2015 IEEE International* *Conference on Image Processing (ICIP)*, pages 1369–1373. IEEE, 2015.3 [64]Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and

efficient design for semantic segmentation with transform- ers. *Advances in neural information processing systems*, 34: 12077–12090, 2021.6 [65]Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, and ´ Kaiming He. Aggregated residual transformations for deep neural networks. In *Proceedings of the IEEE conference on* *computer vision and pattern recognition*, pages 1492–1500,

2017.6
[66]Jiayu Yang, Wei Mao, Jose M Alvarez, and Miaomiao Liu. Cost volume pyramid based depth inference for multi-view stereo. In *Proceedings of the IEEE/CVF conference on* *computer vision and pattern recognition*, pages 4877–4886,

2020.2
[67]Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In *Proceedings of* *the IEEE/CVF Conference on Computer Vision and Pattern* *Recognition*, pages 10371–10381, 2024.2,3,5,6 [68]Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiao- gang Xu, Jiashi Feng, and Hengshuang Zhao. Depth any- thing v2. *arXiv preprint arXiv:2406.09414*, 2024.2,3,5,6, 8 [69]Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A large-scale dataset for generalized multi-view stereo net- works. *Computer Vision and Pattern Recognition (CVPR)*,

2020.6,4
[70]Wei Yin, Xinlong Wang, Chunhua Shen, Yifan Liu, Zhi Tian, Songcen Xu, Changming Sun, and Dou Renyin. Di- versedepth: Affine-invariant depth prediction using diverse data. *arXiv preprint arXiv:2002.00569*, 2020.3 [71]Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, and Chunhua Shen. Learning to recover 3d scene shape from a single image. *CoRR*, abs/2012.09365, 2020.2,3,6,5 [72]Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Si- mon Chen, Yifan Liu, and Chunhua Shen. Towards accurate reconstruction of 3d scene shape from a single monocular image. *IEEE Transactions on Pattern Analysis and Machine* *Intelligence*, 45(5):6480–6494, 2022.3 [73]Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3d: Towards zero-shot metric 3d prediction from a single image. In *Proceedings of the IEEE/CVF International Conference* *on Computer Vision*, pages 9043–9053, 2023.2,3 [74]Amir R Zamir, Alexander Sax,, William B Shen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In *2018 IEEE Confer-* *ence on Computer Vision and Pattern Recognition (CVPR)*. IEEE, 2018.6,4 [75]Zhengyou Zhang. A flexible new technique for camera cali- bration. *IEEE Transactions on Pattern Analysis and Machine* *Intelligence*, 22(11):1330–1334, 2000.3 [76]Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3d: A large photo-realistic dataset for structured 3d modeling. In *Proceedings of The* *European Conference on Computer Vision (ECCV)*, 2020.6,

[77]Shengjie Zhu, Abhinav Kumar, Masa Hu, and Xiaoming Liu. Tame a wild camera: in-the-wild monocular camera calibra- tion. *Advances in Neural Information Processing Systems*, 36, 2024.3,6,7

## MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision

## Supplementary Material

### A. Algorithm Details

|Algorithm 2|ROE alignment subproblem|w/o|truncation|
|---|---|---|---|
||ˆ [1|||
|input:|arrays X ..n], X [1 ∗|..n], W [1 ..n]|∗|
|output:|optimal scale s|and objective value|l to Eq.11|
|||ˆ,||
|function sort arrays|SOLVESUBPROBLEM( ˆ, X X, W by|X X, W ˆ [ X [ i] /X i]|)|
|Q|[1 ..n] ← accumulated sum of|W ∗ X ˆ||
||||n|
|D|[0 ..n] ← {−Q [ n] }∪{|2 · Q [ i] − Q [ n] }|i =1|
||∗|||
|i s|← the first i s.t. D [ ∗ ∗ ˆ [ ∗ ← X [ i] /X i] ∗|i − 1] ≤ 0 ≤ D [ i] ∗||
|l return end function|← objective function value at ∗ ∗ s, l.|s.||

#### A.1. Recovering Shift and Camera Focal

We assume a simple pinhole camera model with isotropic focal length and centered principal point. The 2D image plane is parameterized with the center as (0*,*0). The im- age plane coordinate of pixel *i* is denoted as (*ui,vi*), cor- responding to its predicted 3D point p*i*= (*xi,yi,zi*). The focal length and shift is obtained by minimizing the projec- tion error,

X <u>fx</u> <u>i</u> 2 <u>fyi</u> 2 min *′* *− ui*+ *′* *− vi,* (9) *f,tz zi*+ *tzzi*+ *tz* *i∈M*

which can be further reduced to have a single variable *t* *′z* by substituting *f* with its close-form solution with respect to *′z* *t*, P <u>x</u> <u>i</u> P <u>y</u> <u>i</u> <u>i∈M zi +t</u>*′*<u>z</u>*ui*+<u>i∈M zi +t</u>*′*<u>z</u>*vi* *f* = P2P2

*.* (10)
<u>x</u> <u>i</u> + <u>y</u> <u>i</u> *i∈M zi* +*t′zi∈M zi* +*t′z*

We use a numerical solver for this least squares problem with Levenberg-Marquardt algorithm [38] implemented by SciPy [55] package. For efficiency, the point map is resized to low resolution (64 *×* 64) for running this algorithm. In our practice, it typically converges within 10 iterations in around 3ms.

#### A.2. ROE Alignment

We will first introduce an algorithm to a simpler subprob- lem then derive the solution to either with or without the constraint of *tx*= *ty*= 0 (1D-shift case or 3D-shift case, respectively).

**Subproblem (*w/o* truncation).** Consider the optimiza- tion objective with respect to scale *s* only, denoted as *l₀*(*s*). We omit the mask *M* for simplicity and denote *N* as the number of valid points:

|N|w x −τ|x|
|---|---|---|
|s|i x w x ˆ|w x +x ˆ τ|
|i=1|x ˆ w τ i x wx x ˆ− x ˆ|w x ˆ x w x ˆ τ wx x ˆ+|

X min*l₀*(*s*) = min *wi|sx*ˆ*i− xi|,* (11) *s*

where *wi>* 0 and *x*ˆ*i>* 0 without loss of generality. The objective, as a summation of convex functions, is also con- vex obviously. The minimum occurs where its left-hand *l₀* *′−*

(*s*) derivative and right-hand derivative *l₀*
*′*+

(*s*) have op-
posite signs or one of them is zero,

*′−* X X X X*N* *l₀* (*s*) = *wix*ˆ*i− wix*ˆ*i*= 2 *wix*ˆ*i− wix*ˆ*i,* *xi* *<s s≤* *xi xi* *<s i*=1 *x*ˆ *i x*ˆ*i x*ˆ*i* *N* X X X X *l* *′*+

(*s*) = *w x*ˆ *− w x*ˆ = 2 *w x*ˆ *− w x*ˆ*.*
0 *i i i i i i i i* *x* *x* ˆ *i* *i* *≤s s<x* *x* ˆ *i* *i x* *x*ˆ *i* *i* *≤s i*=1 (12) *′− ′*+ <u>x x</u> *l* (*s*) and *l* (*s*) differ at *{* <u>i</u> *}*. First, we sort *{* <u>i</u> *}* *N* and 0 0 *x*ˆ*i x*ˆ*ii*=1 compute the prefix summations of *{wix*ˆ*i}* *N* *i*=1. This allows us to evaluate the derivatives in *O*(1) time for each point in *{* *x* *i* *}* *N*. Finally, *x*ˆ *i* such that *l* *′−* ( *x* *i*

) *≤* 0 *≤ l* *′*+ (
*x* *i* ) is the *x*ˆ *ii*=1 *xi*0 *x*ˆ*i*0 *x*ˆ*i* minimum point. The solution is outlined in Algorithm2.

**Subproblem (*w/* truncation).** We truncate each residual term to suppress outliers. The truncated objective is

X *N* min*l₁*(*s*) = min min(*τ,w |sxi*ˆ*i− x |i*)*,* (13) *s s* *i*=1 where *τ* is set to 1 in all our experiments. For each item *l₁,i*(*s*) = min(*τ,wi|sx*ˆ*i− xi|*) in the equation, the one- sided derivatives are  <u>i i i i</u> *′−*  *−wix*ˆ*i i< s ≤i* *l* (*s*) = *wix*ˆ*i* <u>i</u> *< s ≤* <u>i i i</u> *,* 1*,i* *i i i*  0 otherwise (14)  *−wix*ˆ <u>i</u> *i* <u>i</u> *i* <u>i</u> *≤ s <* *i* <u>i</u>

*l₁* *′*+ *,i*(*s*) = *wix*ˆ*i* <u>i</u> *i* *≤ s <* <u>i</u> *i* <u>i</u> *i* <u>i</u> *.*  otherwise

Therefore, the one-sided derivatives of *l₀*(*s*) are

X *N*X X *l₁* *′−*

(*s*) = *l₁* *′−* *,i*(*s*) = *wix*ˆ*i− wix*ˆ*i*

|Algorithm 3|ROE alignment subproblem|w/|truncation|
|---|---|---|---|
||ˆ|||
|input:|arrays X [1 ..n], X [1|..n], W [1 ..n], float ∗|τ ∗|
|output:|the optimal scale|s, objective value|l to Eq.13|
|function A|SOLVESUBPROBLEM( [1 ..n] ← X/X ˆ|X ˆ, X, W,|τ )|
|B|[1 ..n] ← ( W ∗ X − τ|) / ( W ∗ X ˆ||
|C|[1 ..n] ← ( W ∗ X + τ|) / ( W ∗ X ˆ) )||
|for|each array A in {A,B,C} sort A and obtain sorted indices|do I A [1 ..n|]|
||||ˆ n|
|end for|QA [1 ..n] ← accumulated sum of|{W X|[ IA [ i]] } i =1|
|Initialize|IE as empty set|||
|for|i = 1 to n do for each array A in {A,B,C} − j ← the last j s.t.|▷ parallel computation do A [ j] < X [ i] /X|ˆ [ i]|
||A + j ← the last j s.t.|A [ j] ≤ X [ i] /X|ˆ [ i]|
||A end for − − d ← 2 · QA [ j] − QB|− [ j] − QC [ j|−]|
||A + + d ← 2 · QA [ j] − QB A|B + [ j] − QC [ j B C|C +]|
|end for Initialize|− + if d < 0 ≤ d then l [1 ..N] with ∞|append i to IE||
|for end for|i in IE do s ← X [ i] /X ˆ [ i] l [ i] ← objective function value at|▷ parallel computation s||
||∗|||
|i|← index of the minimum in ∗ ∗ ˆ [ ∗ ∗|l [ i] ,i ∈ IE ∗||
|s|← X [ i] /X i], l ← l ∗ ∗|[ i]||
|return end function|s, l|||
 *i*=1*x*
*x*ˆ *i* *<s≤* *wwixix* ˆ+ *τ wwixix* ˆ *−τ* *<s≤* *x* *x*ˆ *i* *i i i i i i* X X X = 2 *wix*ˆ*i− wix*ˆ*i− wix*ˆ*i,* *xi* *<swixi −τ wixi*+*τ* *x*ˆ *i wix*ˆ*i<swix*ˆ*i<s* (15) X *N*X X *l₁* *′*+

(*s*) = *l₁* *′*+ *,i*(*s*) = *wix*ˆ*i− wix*ˆ*i*
*x ww τ ww −τ x* *i*=1 *x*ˆ *i* *≤s<ixix*ˆ+ *ixix*ˆ*≤s<x*ˆ*i* *i i i i i i* X X X = 2 *wix*ˆ*i− wix*ˆ*i− wix*ˆ*i.* *xi* *≤swixi −τ wixi*+*τ* *x*ˆ *i wix*ˆ*i≤swix*ˆ*i≤s* (16)

Lemma1shows that the minimum of the objective func- tion in Eq.13is still achieved at one of the points in the set <u>x</u> <u>i</u> *N* *{* *x*ˆ *i* *}* *i*=1, despite the function is non-convex and may con- tain local minima. Solving the subproblem requires two steps, as outlined in Algorithm3. The first step is to identify all extrema in *{* *x* *x* ˆ *i* *i* *}* *N* *i*=1that satisfy *l₀* *′−* ( *x* *x* ˆ *i* *i*

) *<* 0 *≤ l₀* *′*+
( *x* *x* ˆ *i* *i* ) by eval- uating the derivative values. This can be done efficiently through first binary searching on the sorted arrays *{* <u>x</u> *x*ˆ <u>i</u> *i* *}* *N* *i*=1, *{* <u>w</u> *w* <u>i</u> <u>x</u> *i* <u>i</u> *x*ˆ <u>−</u> *i* <u>τ</u> *}* *N* *i*=1and *{* <u>w</u> *w* <u>i</u> <u>x</u> *i* <u>i</u> *x*ˆ <u>+</u> *i* <u>τ</u> *}* *N* *i*=1and then indexing the prefix summations of *{wix*ˆ*i}* in the associated orders. This step has a complexity of *O*(*N* log*N*). The second step involves computing the objective values at these extrema and deter- mining the minimum, which takes *O*(*Nne*) time, where *ne* is the number of extrema. As *ne*approximates the number of outliers, it is typically a small constant in practice.

**Lemma 1.** *There exists at least one pair of* (*k* *∗* *,s* *∗* ) *such* *that s* *∗* *x*ˆ*k∗ − xk∗* = 0 *and s* *∗* *minimizes Eq.13.*

*Proof.* The minimum of *l₁*(*s*) must exist, because *l₁*(*s*) is continuous, piece-wisely linear and bounded in [0*,Nτ*]. *∗ ∗* We first prove that there must exist *s* such that *l₁*(*s*) = *′*+ *′− ∗* min*l₁*(*s*) and *l₁* (*s*) *> l₁* (*s*). Otherwise, for all *s* such *∗ ′*+ *−* that *l₀*(*s*) = min*l₁*(*s*), there will be *l₁* (*s*) = *l₁* (*s*) = 0, hence the value of *l₁*(*s*) in the linear interval where the minimum locates is constant. As a consequence, all neigh- boring intervals will be constant until the boundary where min*l₁*(*s*) = *l₁*(*−∞*) = *Nτ*, which contradicts the obvious fact that min*l₁*(*s*) *≤ l₁*(*x₁/x*ˆ₁) *< Nτ*. Given *l₀* *′*+ (*s* *∗* ) *> l₀* *′−* (*s* *∗* ), there exists an index *k* *∗* such that *s* *∗* = ˆ*xk∗ /xk∗*, because

0 *<l₁* *′*+ (*s* *∗* ) *− l₁* *′−* (*s* *∗* ) X X X =2 *wix*ˆ*i− wix*ˆ*i− wix*ˆ*i* *xi* =*s∗ wixi −τ ∗ wixi*+*τ ∗* *x*ˆ *i wix*ˆ*i*=*swix*ˆ*i*=*s* (17) X *≤ wix*ˆ*i.* <u>xi</u> =*s∗* *x*ˆ *i*

**Alignment with 1D shift.** Recall the alignment objective and let *wi*be 1*/zi*. We rewrite it as follows:

X *n* min [*w |sxi*ˆ*i− x |i*+ *w |syi*ˆ*i−y |i*+ *w |szi*ˆ +*itz− z |i*]*,* *s,tz* *i*=1 (18)

or apply truncation to each absolute residual term

X *n* min *s,t* [min(*τ,wi|sx*ˆ*i− xi|*)+min(*τ,wi|sy*ˆ*i−yi|*) *z* *i*=1 (19)

#### + min(τ,wi|szˆi+ tz− zi|)].

The proposed solution is outlined in Algorithm4, with proof as follows. The corresponding subproblem solver is selected based on whether truncation is applied.

**Lemma 2.** *There exists at least one triplet of* (*k* *∗* *,s* *∗* *,t* *∗* ) *∗ ∗ ∗ ∗* *z* *such that s z*ˆ*k∗* + *tz− zk∗* = 0 *and* (*s ,tz*) *minimizes the* *objective of Equation19.*

*Proof.* Denote the objective as *l₂*(*s,tz*),

X *n* *l₂*(*s,tz*) = [min (*τ,wi|sx*ˆ*i− xi|*) + min (*τ,wi|sy*ˆ*i− yi|*)] *i*=1 X *N* + min (*τ,wi|tz−* (*zi− sz*ˆ*i*)*|*)]*.* *i*=1 (20)

Given arbitrary *s*, using Lemma1, there exists at least one

P pair ( *n* *t* *z* *,k*) such that *tz−* (*zk− sz*ˆ*k*) = 0 and *tz*minimizes

*i*=1min (*τ,wi|sz*ˆ*i*+ *tz− zi|*), hence minimizes *l₂*(*s,tz*) as the rest parts are constant with regard to *tz*. Therefore, a *∗ ∗ ∗* solution *s* is always associated with corresponding (*tz,k*<u>)</u> *∗ ∗* such that that *s z*ˆ*k∗* + *tz− zk∗* = 0.

Lemma2allows us to reduce Eq.19to the subproblem with respect to some index *k*. For each possible index *k*, the objective is formed as:

X *n* min min(*τ,wi|sx*ˆ*i−sxi|*)+min(*τ,wi|sy*ˆ*i−yi|*) *s*(21) *i*=1

|Algorithm 5|ROE alignment|w/|3-D shift|
|---|---|---|---|
|||ˆ||
|input:|point arrays weight array|P [1 ..N, 1 W [1 ..N ∗|.. 3], P [1 ..N, 1 .. 3],] ∗|
|output:|the optimal scale|s, shift t|,|
||objective value|∗ l to Eq.22||
|W [1|.. 3 N] ← repeat each element in||W 3 times|
|Initialize arrays for k|s [1 = 1 to N do ˆ|..N], l [1 ..N ˆ|], t[1 ..N, 3] ▷ parallel computation ˆ|
|X|[1 ... 3 N] ← FLATTEN(|P|[1 ..N, 1 .. 3] − P [ k, 1 .. 3])|
|X|[1 ... 3 N] ← FLATTEN(|P|[1 ..N, 1 .. 3] − P [ k, 1 .. 3])|
||||ˆ,|
|(|s [ k] ,l [ k]) ← SOLVESUBPROBLEM(||X X, W )|
|||ˆ[||
|t[ end for ∗ k ←|k] ← P [ k] − s index of the minimum in|[ k] · P k|] l [1 ..N]|
|∗|∗ ∗|∗ ∗|∗|
|s ← s return|[ k], l ← l [ ∗ ∗ ∗ s, t, l|k], t ←|t[ k]|

+ min(*τ,wi|s*(ˆ*zi− z*ˆ*k*) *−* (*zi− zk*)*|*)*,*

which is solvable in *O*(*N* log*N*) complexity. We enumer- ate all possible indices for *k* and find the minimum. There- fore, the total time complexity is *O*(*N²* log*N*). In our implementation, the point map is resized to low resolution (64*×*64) for alignment, with *N* = 4096 at most. The algorithm is further parallelized with tensor operations on GPUs.

X *n* min *s,tz* min(*τ,wi|sx*ˆ*i*+ *tx− xi|*) *i*=1 (22) + min(*τ,wi|sy*ˆ*i*+ *ty− yi|*)

#### + min(τ,wi|szˆi+ tz− zi|).

Similarly to the proof of Lemma2, there exists at least one group (*k₁* *∗* *,k₂* *∗* *,k₃* *∗* *,s* *∗* *,* t *∗* ) such that *s* *∗* *x*ˆ*k* 1 *∗* + *t∗* *x− x*1 *∗* = *∗∗∗∗∗∗∗∗∗k ∗* 0, *s y*ˆ*k* 2 + *ty− yk* 2 = 0, *s z*ˆ*k* 3 + *tz− zk* 3 = 0, and (*s ,* t ) minimizes the objective. However, the *O*(*N⁴* log*N*) time complexity of a brute-force search is prohibitive. Mo- tivated by the strong locality of surface points within a 3D sphere, we introduce a reasonable assumption, *k₁* = *k₂* = *k₃*, to obtain an approximately optimal solution with *O*(*N²* log*N*) complexity. This assumption posits that the predicted and ground truth patches can be well aligned un- der the condition that one corresponding pair of points co- incides. The effectiveness of the approximated solution has been empirically validated.

|Algorithm 4|ROE alignment|w/|1-D shift|
|---|---|---|---|
|||ˆ[1||
|input:|point arrays weight array|P ..N, 1 W [1 ..N ∗|.. 3], P [1 ..N, 1 .. 3],] ∗|
|output:|the optimal scale|s, shift|t z,|
||objective value|∗ l to Eq.18or Eq.19||
|W [1|.. 3 N] ← repeat each element in||W 3 times|
|Initialize arrays for k|s [1 = 1 to N do|..N], l [1 ..N|], t z [1 ..N] ▷ parallel computation|
|X|ˆ [1 ... 3 N] ← FLATTEN(|P ˆ[1|..N, 1 .. 3] −{ 0, 0, P ˆ[ k, 3] } )|
|X|[1 ... 3 N] ← FLATTEN(|P [1|..N, 1 .. 3] −{ 0, 0 ,P [ k, 3] } )|
||||ˆ,|
|(|s [ k] ,l [ k]) ← SOLVESUBPROBLEM(||X X, W )|
|||ˆ[||
|t end for ∗ k ←|z [ k] ← P [ k, 3] index of the minimum in|− s [ k] · P|k, 3] l [1 ..N]|
|∗|∗ ∗|∗ ∗|∗|
|s ← s return|[ k], l ← l [ ∗ ∗ ∗ s, t z, l|k], t z ← t|z [ k]|

**Alignment with 3D shift.** We apply truncation and rewrite the objective as follows:

### B. Experiment Details

#### B.1. Training Data

The datasets used in our training are listed in Table5. The number of frames may slightly differ from that of the origi- nal data because some invalid frames are dropped. To assign balanced weights to the datasets for training, we compute the retrieval probability of each dataset relative to OpenImagesV7 [32], a large and diverse natural image dataset. Specifically, we leverage DINOv2 [41] to extract feature vectors and calculate the probability that the near- est neighbor of a randomly selected image from OpenIm- agesV7 is found in each respective training dataset.

Image Raw Processed

|Name|Domain|#Frames|Type Weight||
|---|---|---|---|---|
|A2D2[21]|Outdoor/Driving|196K|C 0.8%||
|Argoverse2[62]|Outdoor/Driving|1.1M|C 7.4%||
|ARKitScenes[3]|Indoor|449K|B 8.6%||
|DIML-indoor[14]|Indoor|894K|D 4.8%||
|BlendedMVS[69]|In-the-wild|115K|B 12.0%||
|MegaDepth[35]|Outdoor/In-the-wild|92K|B 5.6%||
|Taskonomy[74]|Indoor|3.6M|B 14.1%||
|Waymo[51]|Outdoor/Driving|788K|C 6.4%||
|GTA-SfM[57]|Outdoor/In-the-wild|19K|A 2.8%||
|Hypersim[45]|Indoor|75K|A 5.0%||
|IRS[58]|Indoor|101K|A 5.6%||
|KenBurns[40]|In-the-wild|76K|A 1.6%||
|MatrixCity[34]|Outdoor/Driving|390K|A 1.3%||
|MidAir[19]|Outdoor/In-the-wild|423K|A 4.0%||
|MVS-Synth[27]|Outdoor/Driving|12K|A 1.2%||
|Spring[37]|In-the-wild|5K|A 0.7%||
|Structured3D[76]|Indoor|77K|A 4.8%||
|Synthia[47]|Outdoor/Driving|96K|A 1.2%||
|TartanAir[60]|In-the-wild|306K|A 5.0%||
|UrbanSyn[25]|Outdoor/Driving|7K|A 2.1%||
|ObjaverseV1[12]|Object|167K|A 4.8%||
|Type|Range|LG LSApplied losses|LS LS LN LM||
|A. Synthetic|Perfect ∞|✓ ✓|✓ ✓|✓ ✓|
|B. SfM/MV Recon|High ∞|✓ ✓|✓|✓|
|C. LiDAR/Laser|High ∼ 100m|✓ ✓||✓|
|D. Kinect|Medium ∼ 10m|✓||✓|

(a)
(b)
(c)
AccuracyLabel qualityDensity1 2 3 Dense Dense&Partial Sparse Dense

Table 5. Datasets used for training and tailored loss combination.

#### B.2. Evaluation Data

The raw evaluation datasets are processed accordingly for reliable evaluation and fair comparison. We report the de- tails as follows.

- **NYUv2 [39].** We use the official test split of 654 samples. Due to the inaccuracy of ground truth values captured by Kinect V1 near boundaries, we filter and remove bound- ary regions by a simple edge detection method. Specif- ically, we set a threshold for the difference between the minimum and maximum depth values within a local win- dow. Depth values beyond 5 meters are excluded because they are unreliable due to the limited sensor range [52]. Additionally, we manually mask out areas with reflective and transparent surfaces, such as mirrors and glass, which cannot be accurately captured by the sensor.
- **KITTI [53].** We utilize the test split of 652 images of Eigen *et al*. [18] following previous works. The original resolution of 1242 *×* 375 does not match our training as- pect ratio (ranging from 1 : 2 to 2 : 1), so we apply center cropping to obtain a resolution of 750*×*375 from the raw images.
- **ETH3D [50].** All 454 images are included. The im- ages are undistorted with the official calibration data and downsized from the original resolution of 6202*×*4135 to 2048 *×* 1365.
- **iBims-1 [31].** All 100 images are included at an original
Figure 8. Examples of evaluation data preprocessing: (a) Re-

moving mirror and boundary artifacts from the ground truth depth in NYUv2. (b) Excluding sky regions in Sintel. (c) Removing boundary artifacts from the ground truth depth in DIODE.

resolution of 640 *×* 480.

- **GSO [16].** The dataset contains 1,030 objects. For each object, we render a single view at 512 *×* 512 resolution. The view is randomly sampled with a FOV ranging from 30 *◦* to 60 *◦*
. The object is centered in the image, and its bounding box occupies approximately 70% of the im- age’s size.

- **Sintel [9].** We use all 1,064 frames and center-crop the images to 872 *×* 436 from the original 1024 *×* 436 res- olution to fit our aspect ratio range. The sky regions are manually masked out because evaluating models with sky depth included is not meaningful.
- **DDAD [24].** We randomly select 1,000 samples from the validation set. The dataset was collected using multiple cameras and LiDAR sensors mounted on a moving vehi- cle. Some cameras inadvertently capture parts of the ve- hicle, causing discrepancies with the sensor’s depth data. To address this issue, we crop the regions that are not ob- structed by the vehicle itself.
- **DIODE [54].** We utilize the official validation split,

which includes 325 indoor images and 446 outdoor im- ages at an original resolution of 1024 *×* 768. Due to ar- tifacts in ground truth depth values near the boundaries in this dataset, we identify and remove these boundary regions using a similar approach as described above.

#### B.3. Evaluation Protocol

For all our models and baselines, predictions and ground truth are aligned in scale (and shift, if applicable) for each image before measuring errors. To clarify the notations in this section:

- pˆ*i*and p*i*are the predicted and ground-truth points, re- spectively.
- *z*ˆ*i*and *zi*are the predicted and ground-truth depths, which are the *Z*-coordinate of corresponding points.
- *M* is the mask of valid ground-truth.
- *a* and *b* denote the scale and shift used to align predictions with the ground truth for evaluation, to avoid confusion with similar symbols used in the training objectives.
*∗*

- **Scale-invariant point map.** The scale *a* to align predic- tion with ground truth is computed as:
X <u>1</u> *a* *∗* = argmin *∥a*pˆ*i−* p*i∥*1*,* (23) *a* *i∈M* *z* *i*

- **Affine-invariant point map.** The scale *a*
*∗* and shift b *∗*

are computed as: X <u>1</u> (*a* *∗* *,* b *∗* ) = argmin *∥a*pˆ*i*+ b *−* p*i∥*1*.* (24) *a,*b *i∈M* *z* *i*

- **Scale-invariant depth map**, the scale *a*
*∗* is computed as X <u>1</u> *a* *∗* = argmin *|az*ˆ*i− zi|.* (25) *s* *i∈M* *z* *i*

- **Affine-invariant depth map.** The scale *a*
*∗* and shift *b* *∗*

are computed as X <u>1</u> (*a* *∗* *,b* *∗* ) = argmin *|az*ˆ*i*+ *b − zi|.* (26) *s* *i∈M* *z* *i*

- **Affine-invariant disparity map.** We follow the estab- lished protocol for affine disparity alignment [43], using least-squares to align predictions in disparity space:
X (*a* *∗* *,b* *∗* ) = argmin (*ad*ˆ*i*+ *b − di*) 2 *,* (27) *s* *i∈M*

where *d*ˆ*i*is the predicted disparity and *di*is the ground truth, defined as *di*= 1*/zi*. To prevent aligned dispari- ties from taking excessively small or negative values, the aligned disparity is truncated by the inverted maximum depth 1*/z*maxbefore inversion. The final aligned depth *z*ˆ*i∗* is computed as:

*z*ˆ *i∗*:=*.* (28) max(*a∗d*ˆ*i*+ *b∗,*1*/z*max)

### C. More Results

**Full table of depth estimation results** In Table6, we present detailed results for depth estimation where methods that predict metric or scale-invariant depth are also evalu- ated on affine-invariant depth and disparity for a fair com- parison.

**More qualitative comparisons** Fig.10and Fig.11 present additional visual comparisons on zero-shot evalu- ation datasets and in-the-wild images. Our method is com- pared with LeReS [71], UniDepth [42], DUSt3R [59], Met- ric3D V2 [26] and Depth Anything V2 [68]. Since Met- ric3D V2 and Depth Anything V2 predict depth map and require ground truth camera focal to obtain 3D points cloud results, we visualize them using our estimated focal lengths. In the supplementary videos, we present *extensive and* *uncurated comparisons* using the first 100 images from the DIV2K[2] dataset.

**More visual results** In Fig.12and Fig.13, we demon- strate more reconstruction results of our method for more open-domain images.

### D. Limitations and Future Work

While our model demonstrates strong performance, accu- rately capturing thin structures remains a significant chal- lenge. This difficulty arises from the network’s limited ca- pacity and the presence of noisy real-world training data. As illustrated in Fig.9, our model may fail to recover these intricate structures. Additionally, while monocular video reconstruction holds great promise as an application, achieving temporal consistency presents substantial challenges. Our model, de- signed for single-image input, cannot inherently maintain temporal coherence due to the ambiguity of the task. Ad- dressing this issue would require non-trivial solutions, such as global optimization techniques. Given the rapid advance- ments in video depth estimation, we believe that an end-to- end model for monocular video reconstruction could sig- nificantly benefit from our proposed techniques. Exploring this direction is a compelling avenue for future work.

Figure 9. A failure case. Our model fails to capture the thin struc-

ture of the fence, leading to a flattened geometry.

NYUv2 KITTI ETH3D iBims-1 GSO Sintel DDAD DIODE Average **Method**d d d d d d d d d d d d d d d d Rel *↓ δ₁↑* Rel.*↓ δ₁↑* Rel *↓ δ₁↑* Rel.*↓ δ₁↑* Rel *↓ δ₁↑* Rel *↓ δ₁↑* Rel *↓ δ₁↑* Rel *↓ δ₁↑* Rel *↓ δ₁↑* Rank*↓* Scale-invariant depth

|DA V1|4.7797.5|5.6195.6|9.41 88.9|5.53 95.8|5.49 99.3|28.3 56.7|13.2 81.5|10.3 87.5|10.387.95.67|
|---|---|---|---|---|---|---|---|---|---|
|-metric indoor|4.7797.5|15.4 73.6|9.41 88.9|5.53 95.8|5.49 99.3|28.3 56.7|24.2 57.4|10.3 87.5|12.982.1|
|-metric outdoor|15.9 72.3|5.6195.6|8.77 92.4|13.8 78.8|8.59 93.6|28.1 54.8|13.2 81.5|13.0 81.4|13.481.3|
|-metric indoor|5.03 97.3|7.61 90.9|6.12 95.5|4.32 97.9|4.38 99.3|23.0 65.2|16.6 73.4|7.95 90.0|9.38 88.7|
|-metric outdoor|15.3 72.3|7.23 93.7|9.30 89.6|10.6 84.9|9.62 92.5|28.6 57.3|14.7 78.0|12.2 83.2|13.4 81.4|

|LeReS|12.1 82.6|19.2 64.8|14.2 78.4|14.0 78.8|13.6 77.9|30.5 52.1|26.5 52.0|18.2 69.6|18.5 69.5|7.31|
|---|---|---|---|---|---|---|---|---|---|---|
|ZoeDepth|5.6296.3|7.2791.9|10.4 87.3|7.45 93.2|3.23 99.9|27.4 61.8|17.0 72.8|11.3 85.2|11.286.15.50||
|DUSt3R|4.40 97.1|7.81 90.6|6.04 95.7|4.98 95.8|3.27 99.5|31.1 57.2|18.6 73.3|8.91 88.8|10.6 87.2|5.00|
|Metric3D V2|4.69 97.4|4.00 98.5|3.84 98.5|4.23 97.7|2.46 99.9|20.7 69.8|7.4194.6|3.29 98.4|6.3394.32.07||
|UniDepth|3.86 98.4|3.73 98.6|5.67 97.0|4.79 97.4|4.18 99.7|28.3 58.8|10.1 90.5|6.83 92.8|8.43 91.6|3.00|

- - DA V2 5.03 97.3 7.23 93.7 6.12 95.5 4.32 **97.9** 4.38 99.3 23.0 65.2 14.7 78.0 7.95 90.0 9.09 89.6 4.06 - -

|DA V1|3.8298.3|5.0496.4|6.23 95.2|4.23 97.3|1.98 100|20.1 71.8|11.3 86.1|6.75 92.6|7.4392.24.83|
|---|---|---|---|---|---|---|---|---|---|
|-metric indoor|3.8298.3|9.95 86.5|6.23 95.2|4.23 97.3|1.98 100|20.1 71.8|17.0 74.0|6.75 92.6|8.7689.5|
|-metric outdoor|7.68 93.8|5.0496.4|6.21 96.6|7.00 94.2|2.77 99.8|20.6 70.0|11.3 86.1|7.03 93.2|8.4591.3|
|-metric indoor|4.16 97.9|7.09 92.3|4.63 97.2|3.44 98.3|1.44 100|17.1 76.6|14.3 79.8|5.41 94.6|7.20 92.1|
|-metric outdoor|8.65 91.0|6.77 94.3|7.24 93.5|6.80 93.5|2.29 100|22.4 67.1|13.4 81.8|8.19 90.7|9.47 89.0|

|Ours|3.44 98.4|4.25 97.8|3.36 98.9|3.46 97.0|1.47 100|19.3 73.4|9.17 90.5|4.89 94.7|6.17 93.8|1.62|
|---|---|---|---|---|---|---|---|---|---|---|
|||||Affine-invariant depth|||||||
|LeReS|6.21 95.4|8.28 90.3|8.95 90.8|6.68 94.5|4.03 99.4|24.0 64.8|16.2 75.8|9.99 88.1|10.5 87.4|8.81|
|ZoeDepth|4.7697.3|5.5995.1|7.27 94.2|5.85 95.7|2.54 99.9|21.8 69.2|14.2 80.1|7.80 90.9|8.7390.37.33||
|DUSt3R|3.73 97.8|7.30 91.6|4.96 96.4|3.94 96.6|2.55 99.6|25.4 64.2|16.9 76.2|6.68 92.6|8.93 89.4|6.62|
|Metric3D V2|3.94 97.6|3.50 98.4|3.24 99.0|3.28 98.3|2.10 99.4|26.6 71.7|7.1594.8|2.75 98.7|6.5794.73.64||
|UniDepth V1|3.40 98.6|3.55 98.7|4.92 97.5|3.76 98.2|2.48 99.9|24.9 64.1|9.46 90.8|4.90 96.2|7.17 93.0|3.62|
|Marigold|4.63 97.3|7.29 93.8|6.08 96.3|4.35 97.2|2.78 99.9|21.2 75.0|14.6 80.5|6.34 94.3|8.41 91.8|5.69|
|Geowizard|4.69 97.4|8.14 92.5|6.90 93.9|4.50 97.1|2.00 99.9|17.8 76.2|16.5 75.7|7.03 92.7|8.45 90.7|6.44|

- -

|DA V2|4.16 97.9|6.77 94.3|4.63 97.2|3.44 98.3|1.44 100|17.1 76.6|13.4 81.8|5.41 94.6|7.04 92.6|-|
|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||-|
|||||||||||2.94|
|Ours|2.92 98.6|3.94 98.0|2.69 99.2|2.74 97.9 Affine-invariant disparity|0.94 100|13.0 83.2|8.40 92.1|3.16 97.5|4.72 95.8|1.56|
|LeReS|7.31 95.5|12.2 87.1|10.2 90.1|8.44 92.9|4.33 99.7|28.9 59.6|23.4 73.0|10.7 88.3|13.2 85.8|8.25|
|ZoeDepth|5.2197.7|5.8495.6|8.07 94.0|6.19 96.1|2.60 99.9|26.9 66.3|14.1 81.7|8.17 92.0|9.6390.46.75||
|DUSt3R|4.24 98.1|7.72 92.1|5.60 96.2|4.49 96.6|2.63 99.8|40.0 56.7|17.4 76.2|7.10 92.8|11.1 88.6|6.75|
|Metric3D V2|13.4 81.5|3.76 98.2|4.30 97.7|8.55 92.3|1.80 100|21.8 72.4|7.3594.1|7.70 90.2|8.5890.85.29||
|UniDepth V1|3.78 98.7|3.64 98.7|5.34 97.2|4.06 98.1|2.56 99.9|28.6 60.7|9.94 89.1|5.95 95.5|7.98 92.2|3.62|
|MiDaS V3.1|4.58 98.1|6.25 94.7|5.77 96.8|4.73 97.4|1.86 100|21.3 73.1|14.5 82.6|6.05 94.9|8.13 92.2|5.00|
|DA V1|4.20 98.4|5.40 97.0|4.68 98.2|4.18 97.6|1.54 100|20.1 77.6|12.7 86.9|5.69 95.7|7.31 93.9|3.00|
|DA V2|4.14 98.3|5.61 96.7|4.71 97.9|3.47 98.5|1.24 100|21.4 72.8|13.1 86.4|5.29 96.1|7.37 93.3|3.12|
|Ours|3.38 98.6|4.05 98.1|3.11 98.9|3.23 98.0|0.96 100|18.4 79.5|8.99 91.5|3.98 97.2|5.76 95.2|1.44|

Table 6. Full table of comparison for depth map estimation. * methods have multiple model versions available for respective benchmarks,

among which the best for each benchmark is chosen for ranking, followed by the detailed results in smaller text size for each version.Gray numbersdenote models trained on respective benchmarks.

Input Ground truth Ours LeReS UniDepth DUSt3R Metric3D V2* DA V2*

Figure 10. Additional qualitative comparisons from the *evaluation datasets*. *: for methods without camera intrinsics prediction, ground-

truth camera intrinsics (and disparity shifts) were used to lift their results into 3D points. ***Best viewed with zoom.***

Input Ours LeReS UniDepth DUSt3R Metric3D V2* DA V2*

Figure 11. Additional qualitative comparisons for *in-the-wild* images from the DIV2K dataset. *: for methods without camera intrinsics

prediction, our camera intrinsics prediction were used to lift their results into 3D points. Supplementary videos contain more results. ***Best*** ***viewed with zoom.***

Figure 12. Additional visual results for open-domain images of our model (page 1 of 2). The columns from left to right are the input images,

reconstructed disparity maps, reconstructed surface geometry viewed from the source view, and three novel-view images, respectively.

Figure 13. Additional visual results for open-domain images of our model (page 2 of 2). The columns from left to right are the input images,

reconstructed disparity maps, reconstructed surface geometry viewed from the source view, and three novel-view images, respectively.
