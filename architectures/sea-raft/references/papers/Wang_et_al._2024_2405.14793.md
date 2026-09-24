# SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow

We introduce SEA-RAFT, a more simple, efficient, and accurate RAFT for optical flow. Compared with RAFT, SEA-RAFT is trained with a new loss (mixture of Laplace). It directly regresses an initial flow for faster convergence in iterative refinements and introduces rigid-motion pre-training to improve generalization. SEA-RAFT achieves state-of-the-art accuracy on the Spring benchmark with a 3.69 endpoint-error (EPE) and a 0.36 1-pixel outlier rate (1px), representing 22.9% and 17.8% error reduction from best published results. In addition, SEA-RAFT obtains the best cross-dataset generalization on KITTI and Spring. With its high efficiency, SEA-RAFT operates at least 2.3×\times faster than existing methods while maintaining competitive performance. The code is publicly available at https://github.com/princeton-vl/SEA-RAFT.

## 1 Introduction

Optical flow is a fundamental task in low-level vision and aims to estimate per-pixel 2D motion between video frames. It is useful for various downstream tasks including action recognition [49, 39, 67], video in-painting [22, 60, 10], frame interpolation [61, 27, 15], 3D reconstruction and synthesis [33, 69].

Although traditionally formulated as an optimization problem [13, 62, 5], almost all recent methods are based on deep learning [50, 44, 48, 14, 43, 57, 56, 8, 66, 45, 68, 63, 55, 24, 11, 29, 6, 54, 42]. In particular, many state-of-the-art methods [50, 44, 14, 43, 66, 29] have adopted architectures based on RAFT [50], which uses a recurrent network to iteratively refine a flow field.

In this paper, we introduce SEA-RAFT, a new variant of RAFT that is more efficient and accurate. When compared against all existing approaches, SEA-RAFT has the best accuracy-efficiency Pareto frontier (Fig. 1):

- • Accuracy: On Spring [35], SEA-RAFT achieves a new state of the art, outperforming the next best by a large margin: 18% error reduction on 1px-outlier rate (3.686 vs. 4.482) and 24% error reduction on endpoint-error (0.363 vs. 0.471). On Sintel [3] and KITTI [36], it outperforms all other methods that have similar computational costs.
Accuracy: On Spring [35], SEA-RAFT achieves a new state of the art, outperforming the next best by a large margin: 18% error reduction on 1px-outlier rate (3.686 vs. 4.482) and 24% error reduction on endpoint-error (0.363 vs. 0.471). On Sintel [3] and KITTI [36], it outperforms all other methods that have similar computational costs.

- • Efficiency: On each benchmark tested, SEA-RAFT runs at least 2.3×2.3\times faster than existing methods that have comparable accuracy. Our smallest model, which still outperforms all other methods on Spring, can run at 21fps when processing 1080p images on an RTX3090, 3×\times faster than the original RAFT.
Efficiency: On each benchmark tested, SEA-RAFT runs at least 2.3×2.3\times faster than existing methods that have comparable accuracy. Our smallest model, which still outperforms all other methods on Spring, can run at 21fps when processing 1080p images on an RTX3090, 3×\times faster than the original RAFT.

We achieve this by introducing a combination of improvements over the original RAFT:

- • Mixture of Laplace Loss: Instead of the standard L1L_{1} loss, we train the network to predict parameters of a mixture of Laplace distributions to maximize the log-likelihood of the ground truth flow. As we will demonstrate, this new loss reduces overfitting to ambiguous cases and improves generalization.
Mixture of Laplace Loss: Instead of the standard L1L_{1} loss, we train the network to predict parameters of a mixture of Laplace distributions to maximize the log-likelihood of the ground truth flow. As we will demonstrate, this new loss reduces overfitting to ambiguous cases and improves generalization.

- • Directly Regressed Initial Flow: Instead of initializing the flow field to zero before iterative refinement, we directly predict the initial flow by reusing the existing context encoder and feeding it the stacked input frames. This simple change introduces minimal overhead but is surprisingly effective in reducing the number of iterations and improving efficiency.
Directly Regressed Initial Flow: Instead of initializing the flow field to zero before iterative refinement, we directly predict the initial flow by reusing the existing context encoder and feeding it the stacked input frames. This simple change introduces minimal overhead but is surprisingly effective in reducing the number of iterations and improving efficiency.

- • Rigid-Flow Pre-Training: We find that pre-training on TartanAir [52], which can significantly improve generalization, despite the limited diversity of flow, which is induced purely by camera motion in a static scene.
Rigid-Flow Pre-Training: We find that pre-training on TartanAir [52], which can significantly improve generalization, despite the limited diversity of flow, which is induced purely by camera motion in a static scene.

These improvements are novel in the context of RAFT-style methods for optical flow. Moreover, they are orthogonal to the improvements proposed in existing RAFT-style methods, which focus on replacing certain blocks with newer designs, such as replacing convolutional blocks with transformers.

Besides the main improvements above, SEA-RAFT also incorporates architectural changes that greatly simplify the original RAFT. In particular, we find that certain custom designs of the original RAFT are unnecessary and can be replaced with standard off-the-shelf modules. For example, the original feature encoder and context encoder were custom-designed and must use different normalization layers for stable training; we replaced each with a standard ResNet. In addition, we replace the original convolutional GRU with a simple RNN consisting entirely of ConvNext blocks. Such simplifications make it easy for SEA-RAFT to incorporate new neural building blocks and scale to larger datasets.

We perform extensive experiments to evaluate SEA-RAFT on standard benchmarks including Spring, Sintel, and KITTI. We also validate the effectiveness of our improvements through ablation studies.

## 2 Related Works

Estimating Optical Flow Classical approaches treated optical flow as an optimization problem that maximizes visual similarity between corresponding pixels, with strong regularization.[13, 62, 5]. Current methods [17, 9, 45, 65, 16, 50, 44, 48, 6, 14, 43, 54, 55, 56, 57, 24, 42, 18, 19, 30, 68, 66, 31] are mostly based on deep learning. FlowNets [9, 17] regarded optical flow as a dense regression problem and used stacked convolution blocks for prediction. DCNet [58] and PWC-Net [45] introduced 4D cost-volume to explicitly model pixel correspondence. RAFT [50] further combined multi-scale 4D cost-volume with recurrent iterative refinements, achieving large improvements and spawning many follow-ups [14, 43, 48, 66, 30, 31, 68, 19, 44].

Our method is a new variant of RAFT [50] with several improvements including a new loss function, direct regression of initial flow, rigid-flow pre-training, and architectural simplifications. All of these improvements are new compared to existing RAFT variants. In particular, our direct regression of initial flow is new compared to existing efficient RAFT variants [11, 37, 6], which mainly focus on efficient implementations of RAFT modules. This direct regression is a simple change with minimal overhead, but substantially reduces the number of RAFT iterations needed.

Data for Optical Flow FlyingChairs and FlyingThings3D [9, 34] are commonly used datasets for optical flow. They provide a large amount of synthetic data but have limited realism. Sintel [3], VIPER [41], Infinigen [40], and Spring [35] are more realistic, using open-source 3D animations, games or procedurally generated scenes. Besides synthetic data, Middlebury, KITTI, and HD1K [12, 23, 36, 1] provide annotations for real-world image pairs. These datasets are limited in both quantity and diversity due to the difficulty of accurately annotating optical flow in the real world. To leverage more data, several methods [8, 54, 55, 42] pre-train their models on different tasks. MatchFlow [8] pre-trains on geometric image matching (GIM) using MegaDepth [26]. Croco-Flow [54, 55], DDVM [42], and Flowformer++ [43] pre-train on unlabeled data. We pre-train SEA-RAFT on rigid flow using TartanAir [52]. Though TartanAir [52] has been used in other methods such as DDVM [42] and CroCo-Flow [54, 55], our adoption of rigid-flow pre-training is new in the context of RAFT-style methods.

Predicting Probability Distributions Predicting probability distributions is a common practice in computer vision [32, 25, 2, 51, 53, 47, 4, 64]. In tasks closely related to optical flow such as keypoint matching [47, 4, 64, 51], the variance of the probability distribution reflects uncertainty of predictions and therefore is useful for many applications. For example, LoFTR [47] filters out uncertain matching pairs. Aspanformer [4] adjusts the look-up radius based on uncertainty.

To handle the ambiguity caused by heavy occlusion, SEA-RAFT predicts a mixture of Laplace (MoL) distribution. Although MoL has been used in keypoint matching methods such as PDC-Net+ [51], our use of MoL is new in the context of RAFT-style methods. In addition, our formulation is different in that we require one mixture component to have a constant variance, making it equivalent to the L1L_{1} loss that aligns better with the optical flow evaluation metrics. This difference is crucial for achieving competitive performance in optical flow, where every pixel needs accurate correspondence, unlike keypoint matching, where a subset of reliable matches suffices.

## 3 Method

In this section, we first describe the iterative refinement in RAFT and then introduce the improvements that lead to SEA-RAFT.

### 3.1 Iterative refinement

Given two adjacent RGB frames, RAFT predicts a field of pixel-wise 2D vectors through iterative refinement that consists of two parts: (1) feature and context encoders, which transform images into lower-resolution dense features, and (2) an RNN unit, which iteratively refines the predictions.

Given two images I1,I2∈ℝH×W×3I_{1},I_{2}\in\mathbb{R}^{H\times W\times 3}, the feature encoder FF takes I1I_{1}, I2I_{2} as inputs separately and outputs a lower-resolution feature F⁡(I1),F⁡(I2)∈ℝh×w×DF(I_{1}),F(I_{2})\in\mathbb{R}^{h\times w\times D}. The context encoder CC takes source image I1I_{1} as input and outputs a context feature C⁡(I1)∈ℝh×w×DC(I_{1})\in\mathbb{R}^{h\times w\times D}. A multi-scale 4D correlation volume {Vk}\{V_{k}\} is then built with the features from feature encoder FF:

|  | Vk=F⁡(I1)∘AvgPool​(F⁡(I2),2k)⊤∈ℝh×w×h2k×w2k,\displaystyle V_{k}=F(I_{1})\circ\texttt{AvgPool}(F(I_{2}),2^{k})^{\top}\in\mathbb{R}^{h\times w\times\frac{h}{2^{k}}\times\frac{w}{2^{k}}}, |  |
|---|---|---|

where ∘\circ represents the correlation operator, which computes similarities (as dot products of feature vectors) between all pairs of pixels across two feature maps.

Several works [19, 18] have explored the optimal choices of the number of levels in the cost volume(kk) and the feature resolution (h,w)(h,w). In SEA-RAFT, we simply follow the original setting in RAFT [50]: (h,w)=18​(H,W),k=4(h,w)=\frac{1}{8}(H,W),k=4.

RAFT iteratively refines a flow prediction μ\mu. Initially, μ\mu is set to be all zeros. Each refinement step uses the current flow prediction μ\mu to fetch a DMD_{M}-dim motion feature MM from the multi-scale correlation volume {Vk}\{V_{k}\} with a look-up radius rr:

|  | M=MotionEncoder​(LookUp​({Vk},μ,r))∈ℝh×w×DM,\displaystyle M=\texttt{MotionEncoder}(\texttt{LookUp}(\{V_{k}\},\mu,r))\in\mathbb{R}^{h\times w\times D_{M}}, |  |
|---|---|---|

where the Lookup operator returns a motion feature vector for each pixel in I1I_{1}, consisting of similarities between the pixel in I1I_{1} and its current correspondence’s neighboring pixels in I2I_{2} within the radius rr. The motion feature vector is further transformed by a motion encoder.

Existing works [4, 11, 21] have explored dynamic radius and look-up when obtaining the motion features from {Vk}\{V_{k}\}. For simplicity of design, SEA-RAFT follows the original RAFT and sets the look-up radius r=4r=4 to a fixed constant. The motion feature MM is fed into the RNN cell along with hidden state hh and context feature C⁡(I1)C(I_{1}). From the new hidden state h′h^{\prime}, the residual flow Δ​μ\Delta\mu is regressed by a 2-layer FlowHead:

|  | h′\displaystyle h^{\prime} | =RNN​(h,M,C⁡(I1))\displaystyle=\texttt{RNN}(h,M,C(I_{1})) |  |
|---|---|---|---|
|  | Δ​μ\displaystyle\Delta\mu | =FlowHead​(h′)\displaystyle=\texttt{FlowHead}(h^{\prime}) |  |

Methods using RAFT-Style iterative refinement [50, 14] usually need many iterations: 1212 in training and as many as 3232 in inference. As a result, RNN-based iterative refinement is a significant bottleneck in latency. Though there have been attempts [6, 11] to reduce the number of iterations, the performance drastically drops with fewer iterations. In contrast, SEA-RAFT only needs 44 iterations in training and up to 1212 iterations in inference to achieve competitive performance.

### 3.2 Mixture-of-Laplace Loss

Most prior works are supervised using an endpoint-error loss on all pixels. However, optical flow training data often contains ambiguous, unpredictable samples, which can dominate this loss empirically.

Ambiguous Cases Ambiguous cases of optical flow can arise with heavy occlusion Fig. 3. While in many cases the motion of occluded pixels can be predicted, sometimes the ambiguity can be too large to predict a single outcome. We examined 10 samples with the highest endpoint-error in the training and validation sets of FlyingChairs [9] and found that ambiguous cases dominate the error.

Review of Probabilistic Regression Prior works for image-matching have proposed probabilistic losses to enable their model to express aleatoric or epistemic uncertainty [47, 51, 4, 51, 55, 53, 64]. These approaches regress the parameters of the probabilistic model and maximize the log-likelihood of the ground truth during training.

Given an image pair {I1,I2}\{I_{1},I_{2}\} and the flow ground truth μg​t\mu_{gt}, the training loss is

|  | ℒp​r​o​b=−log⁡pθ​(μ=μg​t|I1,I2)\displaystyle\mathcal{L}_{prob}=-\log{p_{\theta}(\mu=\mu_{gt}|I_{1},I_{2})} |  |
|---|---|---|

where the probability density function pθp_{\theta} is parameterized by the network. Prior work has formulated pθp_{\theta} as a Gaussian or a Laplace distribution with a predicted mean and variance. For example, we can formulate a naive version of probabilistic regression by assuming: (1) pθp_{\theta} is Laplace with mean μ∈ℝH×W×2\mu\in\mathbb{R}^{H\times W\times 2} and scale b∈ℝH×W×1b\in\mathbb{R}^{H\times W\times 1} predicted by the network, (2) the flow distribution is pixel-wisely independent, and (3) the x-direction flow and the y-direction flow are independent but share the same scale parameter bb:

|  | ℒL​a​p\displaystyle\mathcal{L}_{Lap} | =1H​W​∑u∑v(log⁡2​b​(u,v)+‖μg​t​(u,v)−μ⁡(u,v)‖12​b​(u,v))\displaystyle=\frac{1}{HW}\sum_{u}\sum_{v}{(\log{2b(u,v)}+\frac{\|\mu_{gt}(u,v)-\mu(u,v)\|_{1}}{2b(u,v)})} |  | (1) |
|---|---|---|---|---|

where u,vu,v are indices to the pixels. The Laplace loss can be regarded as an extended version of L1L_{1} loss with an extra penalty term bb. During inference, μ\mu represents the flow prediction, and the scale factor bb provides an estimation of uncertainty. However, we find this naive probabilistic regression does not work well on optical flow, which has also been pointed out by prior work [64].

Mixture of Laplace One reason that naive probabilistic regression performs poorly is numerical instability as the loss contains a log term. To address this issue, we regress b⁡(u,v)b(u,v) directly in log-space. This approach makes training more stable compared to previous approaches which clamp bb to [ϵ,∞)[\epsilon,\infty), where ϵ\epsilon is a small positive number.

Another reason that naive probabilistic regression performs poorly is that it deviates from the standard endpoint-error metric, which only cares about the L1L_{1} difference, but not the uncertainty estimation. Thus, we propose to use a mixture of two Laplace distributions: one for ordinary cases, and the other for ambiguous cases, with mixing coefficient α∈[0,1]\alpha\in[0,1]:

|  | M​i​x​L​a​p​(x,α,β1,β2,μ)\displaystyle MixLap(x;\alpha,\beta_{1},\beta_{2},\mu) | =α⋅e−|x−μ|eβ12​eβ1+(1−α)⋅e−|x−μ|eβ22​eβ2\displaystyle=\alpha\cdot\frac{e^{-\frac{|x-\mu|}{e^{\beta_{1}}}}}{2e^{\beta_{1}}}+(1-\alpha)\cdot\frac{e^{-\frac{|x-\mu|}{e^{\beta_{2}}}}}{2e^{\beta_{2}}} |  |
|---|---|---|---|

Intuitively, at each pixel, we want the first component of the mixture to be aligned with the endpoint-error metric, and the second component to account for ambiguous cases. To explicitly enforce this, we fix β1=0\beta_{1}=0, such that the network is encouraged to optimize for the L1 loss when possible. This leads to the following Mixture-of-Laplace (MoL) loss:

|  | ℒM​o​L\displaystyle\mathcal{L}_{MoL} | =−12​H​W∑u∑v∑d∈{x,y}log[MixLap(μg​t(u,v)d;α(u,v),0,β2(u,v),μ(u,v)d)]\displaystyle=\scalebox{0.85}{\mbox{$\displaystyle-\frac{1}{2HW}\sum_{u}\sum_{v}\sum_{d\in\{x,y\}}{\log{[MixLap(\mu_{gt}(u,v)_{d};\alpha(u,v),0,\beta_{2}(u,v),\mu(u,v)_{d})]}}$}} |  | (2) |
|---|---|---|---|---|

where dd indexes the axe of the flow vector (the xx direction or yy direction).

The free parameters α\alpha, β2\beta_{2}, μ\mu of ℒM​o​L\mathcal{L}_{MoL} are predicted by the network. Intuitively, a higher α\alpha means the flow prediction of this pixel is more “ordinary” instead of “ambiguous”. Mathematically, a higher α\alpha makes ℒm​o​L\mathcal{L}_{moL} behave like an L1L_{1} loss. In Sec. 4.3, we then show that this property leads to better accuracy.

Note that though the mixture model has been used in keypoint matching [47, 4, 51], its application to optical flow requires a different formulation because the goal is substantially different. In keypoint matching, the goal is to identify a subset of reliable matches for downstream applications such as camera pose estimation. Predicting uncertainty serves to filter out unreliable matches, and there is no explicit penalty for predicting few correspondences. As a result, it is not essential for them to align a mixing component to L1L_{1} loss. In optical flow, we are evaluated on the flow prediction for every pixel.

Implementation Details We set an upper bound for β\beta to 1010 in the loss to make the training more stable. We also re-predict α\alpha and β\beta every update iteration. We can similarly define the probabilistic sequence loss as:

|  | ℒa​l​l=∑i=1NγN−i​ℒM​o​Li\displaystyle\mathcal{L}_{all}=\sum_{i=1}^{N}{\gamma^{N-i}\mathcal{L}_{MoL}^{i}} |  | (3) |
|---|---|---|---|

where ℒm​i​xi\mathcal{L}_{mix}^{i} denotes the probabilistic loss in iteration ii, NN denotes the number of iterations, and γ<1\gamma<1 exponentially downweights the early iterations. We empirically observe that our method significantly reduces the number of update iterations needed in inference. In fact, N=4N=4 is sufficient for SEA-RAFT to take first place on the Spring [35] benchmark. We provide detailed ablations in Tab. 4.

### 3.3 Direct Regression of Initial Flow

RAFT-style iterative refinements [48, 14, 37, 8, 31, 66] typically zero-initialize the flow field. However, zero-initialization may deviate substantially from the ground truth, thus needing many iterations. In SEA-RAFT, we borrow an idea from the FlowNet family of methods [17, 9] to predict an initial estimate of optical flow from the context encoder, given both frames as input. We also predict an associated MoL (see Sec. 3.2).

This simple modification also significantly improves the convergence speed of the iterative refinement framework, allowing one to use fewer iterations during inference. Detailed ablations are shown in Tab. 4.

### 3.4 Large-Scale Rigid-Flow Pre-Training

Most prior works train on a small number of datasets with limited size, diversity and realism [9, 34]. To improve generationalization, we pre-train SEA-RAFT on TartanAir [52], which provides optical flow annotations between a pair of (non-rectified) stereo cameras. This type of motion field is a special case of optical flow due to viewpoint change in a rigid static scene. Despite its limited motion diversity, it enables SEA-RAFT to train on data with higher realism and scene diversity, leading to better generalization.

### 3.5 Simplifications

We also provide a few architecture changes that greatly simplify the original RAFT [50]. First, we adopt truncated, ImageNet [7] pre-trained ResNets for the backbones. We also substitute the ConvGRU in RAFT with two ConvNeXt [28] blocks, which we show provides better efficiency and training stability. The detailed ablations of these changes are shown in Tab. 4.

## 4 Experiments

We evaluate SEA-RAFT on Spring [35], KITTI [12], and Sintel [3]. Following previous works, we also incorporate FlyingChairs [9], FlyingThings [34], and HD1K [23] into our training pipeline. To verify the effectiveness of TartanAir [52] rigid-flow pre-training, we provide the performance gain from it in different settings.

Model Details SEA-RAFT is implemented in PyTorch [38]. There are three different types of SEA-RAFT and we denote them as SEA-RAFT(S/M/L) . The only differences among them are the backbone choices and the number of iterations in inference. Specifically, SEA-RAFT(S) uses the first 6 layers of ResNet-18 as the feature/context encoder, and SEA-RAFT(M) uses the first 13 layers of ResNet-34. The pre-trained weights we use are downloaded from torchvision. SEA-RAFT(S) and SEA-RAFT(M) use the same architecture for the recurrent units and keep the number of iterations N=4N=4 in both training and inference. SEA-RAFT(L) can be regarded as an extension based on SEA-RAFT(M): they share the same weights, but SEA-RAFT(L) uses N=12N=12 iterations in inference. Following RAFT [50], we stop the gradient for μ\mu when computing μ′=μ+Δ​μ\mu^{\prime}=\mu+\Delta\mu and only propagate the gradient for residual flow Δ​μ\Delta\mu.

Training Details As mentioned in Sec. 3.4, We pre-train SEA-RAFT on TartanAir [52] for 300k steps with a batch size of 32, input resolution 480×640480\times 640 and learning rate 4×10−44\times 10^{-4}. Similar to RAFT [50], MaskFlowNet [65] and PWC-Net+ [45], we then train our models on FlyingChairs [9] for 100k steps with a batch size of 16, input resolution 368×496368\times 496, learning rate 2.5×10−42.5\times 10^{-4} and FlyingThings3D [34] for 120k steps with a batch size of 32, input resolution 432×960432\times 960, learning rate 4×10−44\times 10^{-4} (denoted as "C+T" following previous works). For the submissions on Sintel [3] benchmark, we fine-tune the model from "C+T" on a mixture of Sintel [3], FlyingThings3D clean pass [34], KITTI [12] and HD1K [23] for 300k steps with a batch size of 32, input resolution 432×960432\times 960 and learning rate 4×10−44\times 10^{-4} (denoted as "C+T+S+K+H" following previous works). Different from previous methods, we reduce the percentage of Sintel [3] in the mixture dataset, which is usually more than 70% in previous papers. Details will be mentioned in the supplementary material. For KITTI [12] submissions, we fine-tune our models from "C+T+S+K+H" on the KITTI training set for extra 10k steps with a batch size of 16, input resolution 432×960432\times 960 and learning rate 10−410^{-4}. For Spring [35] submissions, we fine-tune our models from "C+T+S+K+H" on the Spring training set for extra 120k steps with a batch size of 32, input resolution 540×960540\times 960 and learning rate 4×10−44\times 10^{-4}.

Metrics We adopt the widely used metrics in this study: endpoint-error (EPE), 1-pixel outlier rate (1px), Fl-score and WAUC error. Definitions can be found in [41, 35, 12].

| Extra Data | Method | Spring (train) | Spring (test) |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
| Fine-tune | 1px↓\downarrow | EPE↓\downarrow | 1px↓\downarrow | EPE↓\downarrow | Fl↓\downarrow | WAUC↑\uparrow |  |  |
|  | PWC-Net [45] | ✗ | - | - | 82.27∗ | 2.288∗ | 4.889∗ | 45.670∗ |
|  | FlowNet2 [17] | ✗ | - | - | 6.710∗ | 1.040∗ | 2.823∗ | 90.907∗ |
|  | RAFT [50] | ✗ | 4.788 | 0.448 | 6.790∗ | 1.476∗ | 3.198∗ | 90.920∗ |
|  | GMA [20] | ✗ | 4.763 | 0.443 | 7.074∗ | 0.914∗ | 3.079∗ | 90.722∗ |
|  | RPKNet [37] | ✗ | 4.472 | 0.416 | 4.809 | 0.657 | 1.756 | 92.638 |
|  | DIP [68] | ✗ | 4.273 | 0.463 | - | - | - | - |
|  | SKFlow [48] | ✗ | 4.521 | 0.408 | - | - | - | - |
|  | GMFlow [56] | ✗ | 29.49 | 0.930 | 10.355∗ | 0.945∗ | 2.952∗ | 82.337∗ |
|  | GMFlow++ [57] | ✗ | 4.292 | 0.433 | - | - | - | - |
|  | Flowformer [14] | ✗ | 4.508 | 0.470 | 6.510∗ | 0.723∗ | 2.384∗ | 91.679∗ |
|  | CRAFT [44] | ✗ | 4.803 | 0.448 | - | - | - | - |
|  | SEA-RAFT(S) | ✗ | 4.077 | 0.415 | - | - | - | - |
|  | SEA-RAFT(M) | ✗ | 4.060 | 0.406 | - | - | - | - |
| MegaDepth [26] | MatchFlow(G) [8] | ✗ | 4.504 | 0.407 | - | - | - | - |
| YouTube-VOS [59] | Flowformer+⁣+++[43] | ✗ | 4.482 | 0.447 | - | - | - | - |
| VIPER [41] | MS-RAFT+ [19] | ✗ | 3.577 | 0.397 | 5.724∗ | 0.643∗ | 2.189∗ | 92.888∗ |
| TartanAir [52] | SEA-RAFT(S) | ✗ | 4.161 | 0.410 | - | - | - | - |
| TartanAir [52] | SEA-RAFT(M) | ✗ | 3.888 | 0.406 | - | - | - | - |
| CroCo-Pretrain | CroCoFlow [55] | ✓ | - | - | 4.565 | 0.498 | 1.508 | 93.660 |
| CroCo-Pretrain | Win-Win [24] | ✓ | - | - | 5.371 | 0.475 | 1.621 | 92.270 |
| TartanAir [52] | SEA-RAFT(S) | ✓ | - | - | 3.904 | 0.377 | 1.389 | 94.182 |
| TartanAir [52] | SEA-RAFT(M) | ✓ | - | - | 3.686 | 0.363 | 1.347 | 94.534 |

SEA-RAFT(S)

SEA-RAFT(M)

4.060

0.406

3.577

0.397

SEA-RAFT(S)

SEA-RAFT(M)

SEA-RAFT(S)

SEA-RAFT(M)

3.686

0.363

1.347

94.534

### 4.1 Results on Spring

Zero-Shot Evaluation We compare several representative existing methods with SEA-RAFT using the checkpoints and configurations for Sintel [3] submission on the Spring [35] training split. For fair comparisons, we remove the test-time optimizations such as tiling in this setting, which will significantly slow down the inference speed. All experiments follow the same downsample-upsample protocol: We first downsample the 1080p images by 2×2\times, do inference, and then bi-linearly upsample the flow field back to 1080p, which ensures the input resolution in inference is similar to their training resolution in "C+T+S+K+H". As shown in Tab. 1, SEA-RAFT achieves the best results among representative existing methods without using extra data, which demonstrates the superiority of our mixture loss and architecture design. When allowed to use extra data, SEA-RAFT falls slightly behind MS-RAFT+ [19] but is 24×24\times faster and 11×11\times smaller as mentioned in Fig. 1.

Fine-Tuning Test SEA-RAFT ranks 1st on the public test benchmark: SEA-RAFT(M) outperforms all other methods by at least 22.9%22.9\% on average EPE(endpoint-error) and 17.8%17.8\% on 1px (1-pixel outlier rate), and SEA-RAFT(S) outperforms other methods by at least 20.0%20.0\% on EPE and 12.8%12.8\% on 1px. Besides the strong performance, our method is notably fast. SEA-RAFT(S) is at least 2.3×2.3\times faster than existing methods which can achieve similar performance. As we still follow the downsample-upsample protocol without using any test-time optimizations in submissions, the inference latency directly reflects our speed in handling 1080p images, which means over 20fps on a single RTX3090.

### 4.2 Results on Sintel and KITTI

| Extra Data | Method | Sintel | KITTI |  |  |
|---|---|---|---|---|---|
| Clean↓\downarrow | Final↓\downarrow | Fl-epe↓\downarrow | Fl-all↓\downarrow |  |  |
|  | PWC-Net [45] | 2.55 | 3.93 | 10.4 | 33.7 |
|  | RAFT [50] | 1.43 | 2.71 | 5.04 | 17.4 |
|  | GMA [20] | 1.30 | 2.74 | 4.69 | 17.1 |
|  | SKFlow [48] | 1.22 | 2.46 | 4.27 | 15.5 |
|  | FlowFormer [14] | 1.01 | 2.40 | 4.09† | 14.7† |
|  | DIP [68] | 1.30 | 2.82 | 4.29 | 13.7 |
|  | EMD-L [6] | 0.88 | 2.55 | 4.12 | 13.5 |
|  | CRAFT [44] | 1.27 | 2.79 | 4.88 | 17.5 |
|  | RPKNet [37] | 1.12 | 2.45 | - | 13.0 |
|  | GMFlowNet [66] | 1.14 | 2.71 | 4.24 | 15.4 |
|  | SEA-RAFT(M) | 1.21 | 4.04 | 4.29 | 14.2 |
|  | SEA-RAFT(L) | 1.19 | 4.11 | 3.62 | 12.9 |
|  | GMFlow [56] | 1.08 | 2.48 | 11.2∗ | 28.7∗ |
| Tartan | GMFlow [56] | - | - | 8.70 (-22%)∗ | 24.4 (-15%)∗ |
|  | SEA-RAFT(S) | 1.27 | 4.32 | 4.61 | 15.8 |
| Tartan | SEA-RAFT(S) | 1.27 | 3.74 (-13%) | 4.43 | 15.1 |
| K+H | SEA-RAFT(S) | 1.32 | 2.95 (-32%) | - | - |
| Tartan+K+H | SEA-RAFT(S) | 1.30 | 2.79 (-35%) | - | - |

2.40

0.88

SEA-RAFT(M)

SEA-RAFT(L)

3.62

12.9

SEA-RAFT(S)

SEA-RAFT(S)

SEA-RAFT(S)

SEA-RAFT(S)

| Extra Data | Method | Sintel | KITTI | Inference Cost |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
| Clean↓\downarrow | Final↓\downarrow | Fl-all↓\downarrow | Fl-bg↓\downarrow | Fl-fg↓\downarrow | #MACs | Latency |  |  |
|  | PWC-Net+ [46] | 3.45 | 4.60 | 7.72 | 7.69 | 7.88 | 101.3G | 23.82ms |
|  | RAFT [50] | 1.61⋆ | 2.86⋆ | 5.10 | 4.74 | 6.87 | 938.2G | 140.7ms |
|  | GMA [20] | 1.39⋆ | 2.47⋆ | 5.15 | - | - | 1352G | 183.3ms |
|  | DIP [68] | 1.44⋆ | 2.83⋆ | 4.21 | 3.86 | 5.96 | 3068G | 498.9ms |
|  | GMFlowNet [66] | 1.39 | 2.65 | 4.79 | 4.39 | 6.84 | 1094G | 244.3ms |
|  | GMFlow [56] | 1.74 | 2.90 | 9.32 | 9.67 | 7.57 | 602.6G | 138.5ms |
|  | CRAFT [44] | 1.45⋆ | 2.42⋆ | 4.79 | 4.58 | 5.85 | 2274G | 483.4ms |
|  | FlowFormer [14] | 1.20 | 2.12 | 4.68† | 4.37† | 6.18† | 1715G | 335.6ms |
|  | SKFlow [48] | 1.28⋆ | 2.23⋆ | 4.85 | 4.55 | 6.39 | 1453G | 331.9ms |
|  | GMFlow+ [57] | 1.03 | 2.37 | 4.49 | 4.27 | 5.60 | 1177G | 249.6ms |
|  | EMD-L [6] | 1.32 | 2.51 | 4.49 | 4.16 | 6.15 | 1755G | OOM |
|  | RPKNet [37] | 1.31 | 2.65 | 4.64 | 4.63 | 4.69 | 137.0G | 183.3ms |
| VIPER [41] | CCMR+ [18] | 1.07 | 2.10 | 3.86 | 3.39 | 6.21 | 12653G | OOM |
| MegaDepth [26] | MatchFlow(G) [8] | 1.16⋆ | 2.37⋆ | 4.63† | 4.33† | 6.11† | 1669G | 290.6ms |
| YouTube-VOS [59] | Flowformer++[43] | 1.07 | 1.94 | 4.52† | - | - | 1713G | 373.4ms |
| CroCo-Pretrain | CroCoFlow [55] | 1.09† | 2.44† | 3.64† | 3.18† | 5.94† | 57343G† | 6422ms† |
| DDVM-Pretrain | DDVM [42] | 1.75† | 2.48† | 3.26 † | 2.90 † | 5.05† | - | - |
| TartanAir [52] | SEA-RAFT(M) | 1.44 | 2.86 | 4.64 | 4.47 | 5.49 | 486.9G | 70.96ms |
| TartanAir [52] | SEA-RAFT(L) | 1.31 | 2.60 | 4.30 | 4.08 | 5.37 | 655.1G | 108.0ms |

101.3G

23.82ms

1.03

4.69

1.94

3.26

2.90

SEA-RAFT(M)

SEA-RAFT(L)

Zero-Shot Evaluation Following previous works, we evaluate the zero-shot performance of SEA-RAFT given training schedule "C+T" on Sintel(train) [3] and KITTI(train) [36]. The results are provided in Tab. 2. On KITTI(train), SEA-RAFT outperforms all prior works by a large margin, improving Fl-epe from 4.094.09 to 3.623.62 and Fl-all from 13.713.7 to 12.912.9. On Sintel(train), SEA-RAFT achieves competitive results on the clean pass but, for reasons unclear to us, underperforms existing methods on the final pass. Note that although this “C+T” zero-shot setting is standard, it is of limited relevance to real-world applications, which do not need to restrict the training data to only C+T. Indeed, we show that by adding a small amount of high-quality real-world data (KITTI + HD1K, about 1.2k image pairs compared with 80k image pairs in FlyingThings3D [34]), the performance gap on the Sintel(train) final pass can be remarkably reduced.

Fine-Tuning Test Results are shown in Tab. 3. Compared with RAFT [50], SEA-RAFT achieves 19.9%19.9\% improvements on the Sintel clean pass, 4.2%4.2\% improvements on the Sintel final pass, and 15.7%15.7\% improvements on KITTI Fl-all score. SEA-RAFT is also competitive among all existing methods in terms of performance-speed trade-off: It is the only method that can achieve results better than RAFT [50] with latency around 7070ms. On Sintel(test), methods with similar performance are at least 1.8×1.8\times slower than us. On KITTI(test), methods with similar performance are at least 4.6×4.6\times slower than us.

### 4.3 Ablations and Analysis

| Experiment | Init. | Pre-Training | RNN | Loss Design | #MACs | EPE |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
| Img [7] | Tar [52] | GRU | #blocks | Type | Params |  |  |  |  |
| SEA-RAFT (w/o Tar.) | ✓ | ✓ | - | - | 2 | Mixture-of-Laplace | β1=0,β2∈[0,10]\beta_{1}=0,\beta_{2}\in[0,10] | 284.7G | 0.187 |
| SEA-RAFT (w/ Tar.) | ✓ | ✓ | ✓ | - | 2 | Mixture-of-Laplace | β1=0,β2∈[0,10]\beta_{1}=0,\beta_{2}\in[0,10] | 284.7G | 0.179 |
| w/o Img. | ✓ | - | - | - | 2 | Mixture-of-Laplace | β1=0,β2∈[0,10]\beta_{1}=0,\beta_{2}\in[0,10] | 284.7G | 0.194 |
| w/o Direct Reg. | ✗ | ✓ | - | - | 2 | Mixture-of-Laplace | β1=0,β2∈[0,10]\beta_{1}=0,\beta_{2}\in[0,10] | 277.3G | 0.201 |
| RAFT GRU | ✓ | ✓ | - | ✓ | - | Mixture-of-Laplace | β1=0,β2∈[0,10]\beta_{1}=0,\beta_{2}\in[0,10] | 297.9G | 0.189 |
| More ConvNeXt Blocks | ✓ | ✓ | - | - | 4 | Mixture-of-Laplace | β1=0,β2∈[0,10]\beta_{1}=0,\beta_{2}\in[0,10] | 314.7G | 0.189 |
| Naive Laplace | ✓ | ✓ | - | - | 2 | Naive Single Laplace | β∈[−10,10]\beta\in[-10,10] | 284.7G | 0.217 |
| Naive Mixture-of-Laplace | β1,β2∈[−10,10]\beta_{1},\beta_{2}\in[-10,10] | 0.248 |  |  |  |  |  |  |  |
| L1 | ✓ | ✓ | - | - | 2 | L1L_{1} | ✗ | 284.7G | 0.206 |
| Gaussian | ✓ | ✓ | - | - | 2 | Mixture-of-Gaussian | σ1=1,σ2=eβ2,β2∈[0,10]\displaystyle\sigma_{1}=1,\sigma_{2}=e^{\beta_{2}},\beta_{2}\in[0,10] | 284.7G | 0.210 |

σ1=1,σ2=eβ2,β2∈[0,10]\displaystyle\sigma_{1}=1,\sigma_{2}=e^{\beta_{2}},\beta_{2}\in[0,10]

Ablation experiments are conducted on the Spring [35] dataset based on SEA-RAFT(S). We separate a subval set (sequence 0045 and 0047) from the original training set, train our model on the remaining training data and evaluate the performance on subval. The model is trained with a batch size of 32, input resolution 540×960540\times 960, and tested following "downsample-upsample" protocol mentioned in Sec. 4.1. We describe the details of ablation studies in the following and show the results in Tab. 4:

Pretraining We test the performance of TartanAir [52] rigid-flow pre-training on different datasets(see Tabs. 1, 4 and 2 for details). Without TartanAir, SEA-RAFT already provide strong performance, and the rigid-flow pre-training makes it better. We also show that ImageNet pre-trained weights are effective.

RNN Design Our new RNN designs can reduce the computation without performance loss compared with the GRU used in RAFT [50]. We also show that on Spring subval, 4 ConvNeXt blocks do not work better than 2 ConvNeXt blocks.

Loss Design We see that naive Laplace regression does worse than the original L1L_{1} loss. We also see that it is important to set β1\beta_{1} to 00 in the MoL loss, which aligns the MoL loss to L1L_{1} for ordinary cases. Besides, we find that the mixture of Gaussian loss does not work well for optical flow, even though it has been found to be useful for image matching [4].

Direct Regression of Initial Flow We see that the regressed flow initialization significantly improves accuracy without introducing much overhead.

Inference Time Breakdown In Fig. 7, we show how the computational cost increases when we add more refinements. The cost bottleneck for SEA-RAFT is no longer iterative refinements ( Tab. 5), which allows us to use larger backbones given the same computational cost constraint as RAFT [50].

## 5 Conclusion

| Method | #Iters | Latency (ms) |  |
|---|---|---|---|
| Total | Iter. |  |  |
| RAFT [50] | 24 (K) | 111 | 90.3 (82%) |
| 32 (S) | 141 | 120 (86%) |  |
| SEA-RAFT | 4 (S) | 47.5 | 18.5 (39%) |
| 4 (M) | 70.9 | 18.5 (26%) |  |
| 12 (L) | 108 | 55.5 (51%) |  |

We have introduced SEA-RAFT, a simpler, more efficient and accurate variant of RAFT. It achieves high accuracy across a diverse range of datasets, strong cross-dataset generalization, and state-of-the-art accuracy-speed trade-offs, making it useful for real-world high-resolution optical flow.

## Acknowledgements

This work was partially supported by the National Science Foundation.
