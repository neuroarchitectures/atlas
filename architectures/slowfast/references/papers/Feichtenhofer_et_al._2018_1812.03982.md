# SlowFast Networks for Video Recognition

We present SlowFast networks for video recognition. Our model involves (i) a Slow pathway, operating at low frame rate, to capture spatial semantics, and (ii) a Fast pathway, operating at high frame rate, to capture motion at fine temporal resolution. The Fast pathway can be made very lightweight by reducing its channel capacity, yet can learn useful temporal information for video recognition. Our models achieve strong performance for both action classification and detection in video, and large improvements are pin-pointed as contributions by our SlowFast concept. We report state-of-the-art accuracy on major video recognition benchmarks, Kinetics, Charades and AVA. Code has been made available at: https://github.com/facebookresearch/SlowFast.

## 1 Introduction

It is customary in the recognition of images I⁡(x,y)I(x,y) to treat the two spatial dimensions xx and yy symmetrically. This is justified by the statistics of natural images, which are to a first approximation isotropic—all orientations are equally likely—and shift-invariant [41, 26]. But what about video signals I⁡(x,y,t)I(x,y,t)? Motion is the spatiotemporal counterpart of orientation [2], but all spatiotemporal orientations are not equally likely. Slow motions are more likely than fast motions (indeed most of the world we see is at rest at a given moment) and this has been exploited in Bayesian accounts of how humans perceive motion stimuli [58]. For example, if we see a moving edge in isolation, we perceive it as moving perpendicular to itself, even though in principle it could also have an arbitrary component of movement tangential to itself (the aperture problem in optical flow). This percept is rational if the prior favors slow movements.

If all spatiotemporal orientations are not equally likely, then there is no reason for us to treat space and time symmetrically, as is implicit in approaches to video recognition based on spatiotemporal convolutions [49, 5]. We might instead “factor” the architecture to treat spatial structures and temporal events separately. For concreteness, let us study this in the context of recognition. The categorical spatial semantics of the visual content often evolve slowly. For example, waving hands do not change their identity as “hands” over the span of the waving action, and a person is always in the “person” category even though he/she can transit from walking to running. So the recognition of the categorical semantics (as well as their colors, textures, lighting etc.) can be refreshed relatively slowly. On the other hand, the motion being performed can evolve much faster than their subject identities, such as clapping, waving, shaking, walking, or jumping. It can be desired to use fast refreshing frames (high temporal resolution) to effectively model the potentially fast changing motion.

Based on this intuition, we present a two-pathway SlowFast model for video recognition (Fig. 1). One pathway is designed to capture semantic information that can be given by images or a few sparse frames, and it operates at low frame rates and slow refreshing speed. In contrast, the other pathway is responsible for capturing rapidly changing motion, by operating at fast refreshing speed and high temporal resolution. Despite its high temporal rate, this pathway is made very lightweight, e.g., ∼\scriptstyle\sim20% of total computation. This is because this pathway is designed to have fewer channels and weaker ability to process spatial information, while such information can be provided by the first pathway in a less redundant manner. We call the first a Slow pathway and the second a Fast pathway, driven by their different temporal speeds. The two pathways are fused by lateral connections.

Our conceptual idea leads to flexible and effective designs for video models. The Fast pathway, due to its lightweight nature, does not need to perform any temporal pooling—it can operate on high frame rates for all intermediate layers and maintain temporal fidelity. Meanwhile, thanks to the lower temporal rate, the Slow pathway can be more focused on the spatial domain and semantics. By treating the raw video at different temporal rates, our method allows the two pathways to have their own expertise on video modeling.

There is another well known architecture for video recognition which has a two-stream design [44], but provides conceptually different perspectives. The Two-Stream method [44] has not explored the potential of different temporal speeds, a key concept in our method. The two-stream method adopts the same backbone structure to both streams, whereas our Fast pathway is more lightweight. Our method does not compute optical flow, and therefore, our models are learned end-to-end from the raw data. In our experiments we observe that the SlowFast network is empirically more effective.

Our method is partially inspired by biological studies on the retinal ganglion cells in the primate visual system [27, 37, 8, 14, 51], though admittedly the analogy is rough and premature. These studies found that in these cells, ∼\scriptstyle\sim80% are Parvocellular (P-cells) and ∼\scriptstyle\sim15-20% are Magnocellular (M-cells). The M-cells operate at high temporal frequency and are responsive to fast temporal changes, but not sensitive to spatial detail or color. P-cells provide fine spatial detail and color, but lower temporal resolution, responding slowly to stimuli. Our framework is analogous in that: (i) our model has two pathways separately working at low and high temporal resolutions; (ii) our Fast pathway is designed to capture fast changing motion but fewer spatial details, analogous to M-cells; and (iii) our Fast pathway is lightweight, similar to the small ratio of M-cells. We hope these relations will inspire more computer vision models for video recognition.

We evaluate our method on the Kinetics-400 [30], Kinetics-600 [3], Charades [43] and AVA [20] datasets. Our comprehensive ablation experiments on Kinetics action classification demonstrate the efficacy contributed by SlowFast. SlowFast networks set a new state-of-the-art on all datasets with significant gains to previous systems in the literature.

## 2 Related Work

Actions can be formulated as spatiotemporal objects and captured by oriented filtering in spacetime, as done by HOG3D [31] and cuboids [10]. 3D ConvNets [48, 49, 5] extend 2D image models [32, 45, 47, 24] to the spatiotemporal domain, handling both spatial and temporal dimensions similarly. There are also related methods focusing on long-term filtering and pooling using temporal strides [52, 13, 55, 62], as well as decomposing the convolutions into separate 2D spatial and 1D temporal filters [12, 50, 61, 39].

Beyond spatiotemporal filtering or their separable versions, our work pursuits a more thorough separation of modeling expertise by using two different temporal speeds.

There is a classical branch of research focusing on hand-crafted spatiotemporal features based on optical flow. These methods, including histograms of flow [33], motion boundary histograms [6], and trajectories [53], had shown competitive performance for action recognition before the prevalence of deep learning.

In the context of deep neural networks, the two-stream method [44] exploits optical flow by viewing it as another input modality. This method has been a foundation of many competitive results in the literature [12, 13, 55]. However, it is methodologically unsatisfactory given that optical flow is a hand-designed representation, and two-stream methods are often not learned end-to-end jointly with the flow.

## 3 SlowFast Networks

SlowFast networks can be described as a single stream architecture that operates at two different framerates, but we use the concept of pathways to reflect analogy with the biological Parvo- and Magnocellular counterparts. Our generic architecture has a Slow pathway (Sec. 3.1) and a Fast pathway (Sec. 3.2), which are fused by lateral connections to a SlowFast network (Sec. 3.3). Fig. 1 illustrates our concept.

### 3.1 Slow pathway

The Slow pathway can be any convolutional model (e.g., [12, 49, 5, 56]) that works on a clip of video as a spatiotemporal volume. The key concept in our Slow pathway is a large temporal stride τ\tau on input frames, i.e., it processes only one out of τ\tau frames. A typical value of τ\tau we studied is 16—this refreshing speed is roughly 2 frames sampled per second for 30-fps videos. Denoting the number of frames sampled by the Slow pathway as TT, the raw clip length is T×τT\times\tau frames.

### 3.2 Fast pathway

In parallel to the Slow pathway, the Fast pathway is another convolutional model with the following properties.

Our goal here is to have a fine representation along the temporal dimension. Our Fast pathway works with a small temporal stride of τ/α\tau/\alpha, where α>1\alpha>1 is the frame rate ratio between the Fast and Slow pathways. The two pathways operate on the same raw clip, so the Fast pathway samples α​T\alpha T frames, α\alpha times denser than the Slow pathway. A typical value is α=8\alpha=8 in our experiments.

The presence of α\alpha is in the key of the SlowFast concept (Fig. 1, time axis). It explicitly indicates that the two pathways work on different temporal speeds, and thus drives the expertise of the two subnets instantiating the two pathways.

Our Fast pathway not only has a high input resolution, but also pursues high-resolution features throughout the network hierarchy. In our instantiations, we use no temporal downsampling layers (neither temporal pooling nor time-strided convolutions) throughout the Fast pathway, until the global pooling layer before classification. As such, our feature tensors always have α​T\alpha T frames along the temporal dimension, maintaining temporal fidelity as much as possible.

Our Fast pathway also distinguishes with existing models in that it can use significantly lower channel capacity to achieve good accuracy for the SlowFast model. This makes it lightweight.

In a nutshell, our Fast pathway is a convolutional network analogous to the Slow pathway, but has a ratio of β\beta (β<1\beta<1) channels of the Slow pathway. The typical value is β=1/8\beta=1/8 in our experiments. Notice that the computation (floating-number operations, or FLOPs) of a common layer is often quadratic in term of its channel scaling ratio. This is what makes the Fast pathway more computation-effective than the Slow pathway. In our instantiations, the Fast pathway typically takes ∼\scriptstyle\sim20% of the total computation. Interestingly, as mentioned in Sec. 1, evidence suggests that ∼\scriptstyle\sim15-20% of the retinal cells in the primate visual system are M-cells (that are sensitive to fast motion but not color or spatial detail).

The low channel capacity can also be interpreted as a weaker ability of representing spatial semantics. Technically, our Fast pathway has no special treatment on the spatial dimension, so its spatial modeling capacity should be lower than the Slow pathway because of fewer channels. The good results of our model suggest that it is a desired tradeoff for the Fast pathway to weaken its spatial modeling ability while strengthening its temporal modeling ability.

Motivated by this interpretation, we also explore different ways of weakening spatial capacity in the Fast pathway, including reducing input spatial resolution and removing color information. As we will show by experiments, these versions can all give good accuracy, suggesting that a lightweight Fast pathway with less spatial capacity can be made beneficial.

### 3.3 Lateral connections

The information of the two pathways is fused, so one pathway is not unaware of the representation learned by the other pathway. We implement this by lateral connections, which have been used to fuse optical flow-based, two-stream networks [12, 13]. In image object detection, lateral connections [35] are a popular technique for merging different levels of spatial resolution and semantics.

Similar to [12, 35], we attach one lateral connection between the two pathways for every “stage" (Fig. 1). Specifically for ResNets [24], these connections are right after pool1, res2, res3, and res4. The two pathways have different temporal dimensions, so the lateral connections perform a transformation to match them (detailed in Sec. 3.4). We use unidirectional connections that fuse features of the Fast pathway into the Slow one (Fig. 1). We have experimented with bidirectional fusion and found similar results.

Finally, a global average pooling is performed on each pathway’s output. Then two pooled feature vectors are concatenated as the input to the fully-connected classifier layer.

| stage | Slow pathway | Fast pathway | output sizes TT×\timesS2S^{2} |
|---|---|---|---|
| raw clip | - | - | 64×\times2242{}^{\text{2}} |
| data layer | stride 16, 12{}^{\text{2}} | stride 2, 12{}^{\text{2}} | Slow:4×2242Fast:32×2242\begin{array}[]{c}\text{\emph{Slow}}:\text{{4}$\times$224${}^{\text{2}}$}\\[-0.85005pt] \text{\emph{Fast}}:\text{{\color[rgb]{0.4727,0.6992,0.5}{32}}$\times$224${}^{\text{2}}$}\end{array} |
| conv1 | 1×\times72{}^{\text{2}}, 64 | 5×\times72{}^{\text{2}}, 8 | Slow:4×1122Fast:32×1122\begin{array}[]{c}\text{\emph{Slow}}:\text{{4}$\times$112${}^{\text{2}}$}\\[-0.85005pt] \text{\emph{Fast}}:\text{{\color[rgb]{0.4727,0.6992,0.5}{32}}$\times$112${}^{\text{2}}$}\end{array} |
| stride 1, 22{}^{\text{2}} | stride 1, 22{}^{\text{2}} |  |  |
| pool1 | 1×\times32{}^{\text{2}} max | 1×\times32{}^{\text{2}} max | Slow:4×562Fast:32×562\begin{array}[]{c}\text{\emph{Slow}}:\text{{4}$\times$56${}^{\text{2}}$}\\[-0.85005pt] \text{\emph{Fast}}:\text{{\color[rgb]{0.4727,0.6992,0.5}{32}}$\times$56${}^{\text{2}}$}\end{array} |
| stride 1, 22{}^{\text{2}} | stride 1, 22{}^{\text{2}} |  |  |
| res2 | [1×12, 641×32, 641×12, 256]\left[\begin{array}[]{c}\text{1$\times$1${}^{\text{2}}$, {64}}\\[-0.85005pt] \text{1$\times$3${}^{\text{2}}$, {64}}\\[-0.85005pt] \text{1$\times$1${}^{\text{2}}$, {256}}\end{array}\right]×\times3 | [3×12, 81×32, 81×12, 32]\left[\begin{array}[]{c}\text{\lx@text@underline{3$\times$1${}^{\text{2}}$}, {\color[rgb]{1,0.5,0}8}}\\[-0.85005pt] \text{1$\times$3${}^{\text{2}}$, {\color[rgb]{1,0.5,0}8}}\\[-0.85005pt] \text{1$\times$1${}^{\text{2}}$, {\color[rgb]{1,0.5,0}32}}\end{array}\right]×\times3 | Slow:4×562Fast:32×562\begin{array}[]{c}\text{\emph{Slow}}:\text{{4}$\times$56${}^{\text{2}}$}\\[-0.85005pt] \text{\emph{Fast}}:\text{{\color[rgb]{0.4727,0.6992,0.5}{32}}$\times$56${}^{\text{2}}$}\end{array} |
| res3 | [1×12, 1281×32, 1281×12, 512]\left[\begin{array}[]{c}\text{1$\times$1${}^{\text{2}}$, {128}}\\[-0.85005pt] \text{1$\times$3${}^{\text{2}}$, {128}}\\[-0.85005pt] \text{1$\times$1${}^{\text{2}}$, {512}}\end{array}\right]×\times4 | [3×12, 161×32, 161×12, 64]\left[\begin{array}[]{c}\text{\lx@text@underline{3$\times$1${}^{\text{2}}$}, {\color[rgb]{1,0.5,0}16}}\\[-0.85005pt] \text{1$\times$3${}^{\text{2}}$, {\color[rgb]{1,0.5,0}16}}\\[-0.85005pt] \text{1$\times$1${}^{\text{2}}$, {\color[rgb]{1,0.5,0}64}}\end{array}\right]×\times4 | Slow:4×282Fast:32×282\begin{array}[]{c}\text{\emph{Slow}}:\text{{4}$\times$28${}^{\text{2}}$}\\[-0.85005pt] \text{\emph{Fast}}:\text{{\color[rgb]{0.4727,0.6992,0.5}{32}}$\times$28${}^{\text{2}}$}\end{array} |
| res4 | [3×12, 2561×32, 2561×12, 1024]\left[\begin{array}[]{c}\text{\lx@text@underline{3$\times$1${}^{\text{2}}$}, {256}}\\[-0.85005pt] \text{1$\times$3${}^{\text{2}}$, {256}}\\[-0.85005pt] \text{1$\times$1${}^{\text{2}}$, {1024}}\end{array}\right]×\times6 | [3×12, 321×32, 321×12, 128]\left[\begin{array}[]{c}\text{\lx@text@underline{3$\times$1${}^{\text{2}}$}, {\color[rgb]{1,0.5,0}32}}\\[-0.85005pt] \text{1$\times$3${}^{\text{2}}$, {\color[rgb]{1,0.5,0}32}}\\[-0.85005pt] \text{1$\times$1${}^{\text{2}}$, {\color[rgb]{1,0.5,0}128}}\end{array}\right]×\times6 | Slow:4×142Fast:32×142\begin{array}[]{c}\text{\emph{Slow}}:\text{{4}$\times$14${}^{\text{2}}$}\\[-0.85005pt] \text{\emph{Fast}}:\text{{\color[rgb]{0.4727,0.6992,0.5}{32}}$\times$14${}^{\text{2}}$}\end{array} |
| res5 | [3×12, 5121×32, 5121×12, 2048]\left[\begin{array}[]{c}\text{\lx@text@underline{3$\times$1${}^{\text{2}}$}, {512}}\\[-0.85005pt] \text{1$\times$3${}^{\text{2}}$, {512}}\\[-0.85005pt] \text{1$\times$1${}^{\text{2}}$, {2048}}\end{array}\right]×\times3 | [3×12, 641×32, 641×12, 256]\left[\begin{array}[]{c}\text{\lx@text@underline{3$\times$1${}^{\text{2}}$}, {\color[rgb]{1,0.5,0}64}}\\[-0.85005pt] \text{1$\times$3${}^{\text{2}}$, {\color[rgb]{1,0.5,0}64}}\\[-0.85005pt] \text{1$\times$1${}^{\text{2}}$, {\color[rgb]{1,0.5,0}256}}\end{array}\right]×\times3 | Slow:4×72Fast:32×72\begin{array}[]{c}\text{\emph{Slow}}:\text{{4}$\times$7${}^{\text{2}}$}\\[-0.85005pt] \text{\emph{Fast}}:\text{{\color[rgb]{0.4727,0.6992,0.5}{32}}$\times$7${}^{\text{2}}$}\end{array} |
| global average pool, concate, fc | # classes |  |  |

### 3.4 Instantiations

Our idea of SlowFast is generic, and it can be instantiated with different backbones (e.g., [45, 47, 24]) and implementation specifics. In this subsection, we describe our instantiations of the network architectures.

An example SlowFast model is specified in Table 1. We denote spatiotemporal size by TT×\timesS2S^{2} where TT is the temporal length and SS is the height and width of a square spatial crop. The details are described next.

The Slow pathway in Table 1 is a temporally strided 3D ResNet, modified from [12]. It has TT == 4 frames as the network input, sparsely sampled from a 64-frame raw clip with a temporal stride τ\tau == 16. We opt to not perform temporal downsampling in this instantiation, as doing so would be detrimental when the input stride is large.

Unlike typical C3D / I3D models, we use non-degenerate temporal convolutions (temporal kernel size >> 1, underlined in Table 1) only in res4 and res5; all filters from conv1 to res3 are essentially 2D convolution kernels in this pathway. This is motivated by our experimental observation that using temporal convolutions in earlier layers degrades accuracy. We argue that this is because when objects move fast and the temporal stride is large, there is little correlation within a temporal receptive field unless the spatial receptive field is large enough (i.e., in later layers).

Table 1 shows an example of the Fast pathway with α=8\alpha=8 and β=1/8\beta=1/8. It has a much higher temporal resolution (green) and lower channel capacity (orange).

The Fast pathway has non-degenerate temporal convolutions in every block. This is motivated by the observation that this pathway holds fine temporal resolution for the temporal convolutions to capture detailed motion. Further, the Fast pathway has no temporal downsampling layers by design.

Our lateral connections fuse from the Fast to the Slow pathway. It requires to match the sizes of features before fusing. Denoting the feature shape of the Slow pathway as {\{TT, S2S^{2}, CC}\}, the feature shape of the Fast pathway is {\{α​T\alpha T, S2S^{2}, β​C\beta C}\}. We experiment with the following transformations in the lateral connections:

(i) Time-to-channel: We reshape and transpose {\{α​T\alpha T, S2S^{2}, β​C\beta C}\} into {\{TT, S2S^{2}, α​β​C\alpha\beta C}\}, meaning that we pack all α\alpha frames into the channels of one frame.

(ii) Time-strided sampling: We simply sample one out of every α\alpha frames, so {\{α​T\alpha T, S2S^{2}, β​C\beta C}\} becomes {\{TT, S2S^{2}, β​C\beta C}\}.

(iii) Time-strided convolution: We perform a 3D convolution of a 5×\times12{}^{\text{2}} kernel with 2​β​C2\beta C output channels and stride == α\alpha.

The output of the lateral connections is fused into the Slow pathway by summation or concatenation.

## 4 Experiments: Action Classification

We evaluate our approach on four video recognition datasets using standard evaluation protocols. For the action classification experiments, presented in this section we consider the widely used Kinetics-400 [30], the recent Kinetics-600 [3], and Charades [43]. For action detection experiments in Sec. 5, we use the challenging AVA dataset [20].

Our models on Kinetics are trained from random initialization (“from scratch”), without using ImageNet [7] or any pre-training. We use synchronized SGD training following the recipe in [19]. See details in Appendix.

For the temporal domain, we randomly sample a clip (of α​T\alpha T×\timesτ\tau frames) from the full-length video, and the input to the Slow and Fast pathways are respectively TT and α​T\alpha T frames; for the spatial domain, we randomly crop 224×\times224 pixels from a video, or its horizontal flip, with a shorter side randomly sampled in [256, 320] pixels [45, 56].

Following common practice, we uniformly sample 10 clips from a video along its temporal axis. For each clip, we scale the shorter spatial side to 256 pixels and take 3 crops of 256×\times256 to cover the spatial dimensions, as an approximation of fully-convolutional testing, following the code of [56]. We average the softmax scores for prediction.

We report the actual inference-time computation. As existing papers differ in their inference strategy for cropping/clipping in space and in time. When comparing to previous work, we report the FLOPs per spacetime “view" (temporal clip with spatial crop) at inference and the number of views used. Recall that in our case, the inference-time spatial size is 2562 (instead of 2242 for training) and 10 temporal clips each with 3 spatial crops are used (30 views).

Kinetics-400 [30] consists of ∼\scriptstyle\sim240k training videos and 20k validation videos in 400 human action categories. Kinetics-600 [3] has ∼\scriptstyle\sim392k training videos and 30k validation videos in 600 classes. We report top-1 and top-5 classification accuracy (%). We report the computational cost (in FLOPs) of a single, spatially center-cropped clip.

Charades [43] has ∼\scriptstyle\sim9.8k training videos and 1.8k validation videos in 157 classes in a multi-label classification setting of longer activities spanning ∼\scriptstyle\sim30 seconds on average. Performance is measured in mean Average Precision (mAP).

### 4.1 Main Results

Table 2 shows the comparison with state-of-the-art results for our SlowFast instantiations using various input samplings (TT×\timesτ\tau) and backbones: ResNet-50/101 (R50/101) [24] and Nonlocal (NL) [56].

In comparison to the previous state-of-the-art [56] our best model provides 2.1% higher top-1 accuracy. Notably, all our results are substantially better than existing results that are also without ImageNet pre-training. In particular, our model (79.8%) is 5.9% absolutely better than the previous best result of this kind (73.9%). We have experimented with ImageNet pretraining for SlowFast networks and found that they perform similar (±\pm0.3%) for both the pre-trained and the train from scratch (random initialization) variants.

| model | flow | pretrain | top-1 | top-5 | GFLOPs×\timesviews |
|---|---|---|---|---|---|
| I3D [5] |  | ImageNet | 72.1 | 90.3 | 108 ×\times N/A |
| Two-Stream I3D [5] | ✓ | ImageNet | 75.7 | 92.0 | 216 ×\times N/A |
| S3D-G [61] | ✓ | ImageNet | 77.2 | 93.0 | 143 ×\times N/A |
| Nonlocal R50 [56] |  | ImageNet | 76.5 | 92.6 | 282 ×\times 30 |
| Nonlocal R101 [56] |  | ImageNet | 77.7 | 93.3 | 359 ×\times 30 |
| R(2+1)D Flow [50] | ✓ | - | 67.5 | 87.2 | 152 ×\times 115 |
| STC [9] |  | - | 68.7 | 88.5 | N/A ×\times N/A |
| ARTNet [54] |  | - | 69.2 | 88.3 | 23.5 ×\times 250 |
| S3D [61] |  | - | 69.4 | 89.1 | 66.4 ×\times N/A |
| ECO [63] |  | - | 70.0 | 89.4 | N/A ×\times N/A |
| I3D [5] | ✓ | - | 71.6 | 90.0 | 216 ×\times N/A |
| R(2+1)D [50] |  | - | 72.0 | 90.0 | 152 ×\times 115 |
| R(2+1)D [50] | ✓ | - | 73.9 | 90.9 | 304 ×\times 115 |
| SlowFast 4×\times16, R50 |  | - | 75.6 | 92.1 | 36.1 ×\times 30 |
| SlowFast 8×\times8, R50 |  | - | 77.0 | 92.6 | 65.7 ×\times 30 |
| SlowFast 8×\times8, R101 |  | - | 77.9 | 93.2 | 106 ×\times 30 |
| SlowFast 16×\times8, R101 |  | - | 78.9 | 93.5 | 213 ×\times 30 |
| SlowFast 16×\times8, R101+NL |  | - | 79.8 | 93.9 | 234 ×\times 30 |

Our results are achieved at low inference-time cost. We notice that many existing works (if reported) use extremely dense sampling of clips along the temporal axis, which can lead to >>100 views at inference time. This cost has been largely overlooked. In contrast, our method does not require many temporal clips, due to the high temporal resolution yet lightweight Fast pathway. Our cost per spacetime view can be low (e.g., 36.1 GFLOPs), while still being accurate.

The SlowFast variants from Table 2 (with different backbones and sample rates) are compared in Fig. 2 the with their corresponding Slow-only pathway to assess the improvement brought by the Fast pathway. The horizontal axis measures model capacity for a single input clip of 2562 spatial size, which is proportional to 1//30 of the overall inference cost. Fig. 2 shows that for all variants the Fast pathway is able to consistently improve the performance of the Slow counterpart at comparatively low cost. The next subsection provides a more detailed analysis on Kinetics-400.

is relatively new, and existing results are limited. So our goal is mainly to provide results for future reference in Table 3. Note that the Kinetics-600 validation set overlaps with the Kinetics-400 training set [3], and therefore we do not pre-train on Kinetics-400. The winning entry [21] of the latest ActivityNet Challenge 2018 [15] reports a best single-model, single-modality accuracy of 79.0%. Our variants show good performance with the best model at 81.8%. SlowFast results on the recent Kinetics-700 [4] are in [11].

| model | pretrain | top-1 | top-5 | GFLOPs×\timesviews |
|---|---|---|---|---|
| I3D [3] | - | 71.9 | 90.1 | 108 ×\times N/A |
| StNet-IRv2 RGB [21] | ImgNet+Kin400 | 79.0 | N/A | N/A |
| SlowFast 4×\times16, R50 | - | 78.8 | 94.0 | 36.1 ×\times 30 |
| SlowFast 8×\times8, R50 | - | 79.9 | 94.5 | 65.7 ×\times30 |
| SlowFast 8×\times8, R101 | - | 80.4 | 94.8 | 106 ×\times 30 |
| SlowFast 16×\times8, R101 | - | 81.1 | 95.1 | 213 ×\times 30 |
| SlowFast 16×\times8, R101+NL | - | 81.8 | 95.1 | 234 ×\times 30 |

| model | pretrain | mAP | GFLOPs×\timesviews |
|---|---|---|---|
| CoViAR, R-50 [59] | ImageNet | 21.9 | N/A |
| Asyn-TF, VGG16 [42] | ImageNet | 22.4 | N/A |
| MultiScale TRN [62] | ImageNet | 25.2 | N/A |
| Nonlocal, R101 [56] | ImageNet+Kinetics400 | 37.5 | 544 ×\times 30 |
| STRG, R101+NL [57] | ImageNet+Kinetics400 | 39.7 | 630 ×\times 30 |
| our baseline (Slow-only) | Kinetics-400 | 39.0 | 187 ×\times 30 |
| SlowFast | Kinetics-400 | 42.1 | 213 ×\times 30 |
| SlowFast, +NL | Kinetics-400 | 42.5 | 234 ×\times 30 |
| SlowFast, +NL | Kinetics-600 | 45.2 | 234 ×\times 30 |

|  | lateral | top-1 | top-5 | GFLOPs |
|---|---|---|---|---|
| Slow-only | - | 72.6 | 90.3 | 27.3 |
| Fast-only | - | 51.7 | 78.5 | 6.4 |
| SlowFast | - | 73.5 | 90.3 | 34.2 |
| SlowFast | TtoC, sum | 74.5 | 91.3 | 34.2 |
| SlowFast | TtoC, concat | 74.3 | 91.0 | 39.8 |
| SlowFast | T-sample | 75.4 | 91.8 | 34.9 |
| SlowFast | T-conv | 75.6 | 92.1 | 36.1 |

|  | top-1 | top-5 | GFLOPs |
|---|---|---|---|
| Slow-only | 72.6 | 90.3 | 27.3 |
| β\beta == 1/41/4 | 75.6 | 91.7 | 54.5 |
| 1/61/6 | 75.8 | 92.0 | 41.8 |
| 1/81/8 | 75.6 | 92.1 | 36.1 |
| 1/121/12 | 75.2 | 91.8 | 32.8 |
| 1/161/16 | 75.1 | 91.7 | 30.6 |
| 1/321/32 | 74.2 | 91.3 | 28.6 |

| Fast pathway | spatial | top-1 | top-5 | GFLOPs |
|---|---|---|---|---|
| RGB | - | 75.6 | 92.1 | 36.1 |
| RGB, β\beta=1/41/4 | half | 74.7 | 91.8 | 34.4 |
| gray-scale | - | 75.5 | 91.9 | 34.1 |
| time diff | - | 74.5 | 91.6 | 34.2 |
| optical flow | - | 73.8 | 91.3 | 35.1 |

[43] is a dataset with longer range activities. Table 4 shows our SlowFast results on it. For fair comparison, our baseline is the Slow-only counterpart that has 39.0 mAP. SlowFast increases over this baseline by 3.1 mAP (to 42.1), while the extra NL leads to an additional 0.4 mAP. We also achieve 45.2 mAP when pre-trained on Kinetics-600. Overall, our SlowFast models in Table 4 outperform the previous best number (STRG [57]) by solid margins, at lower cost.

### 4.2 Ablation Experiments

This section provides ablation studies on Kinetics-400 comparing accuracy and computational complexity.

We first aim to explore the SlowFast complementarity by changing the sample rate (TT×\timesτ\tau) of the Slow pathway. Therefore, this ablation studies α\alpha, the frame rate ratio between the Fast and Slow paths. Fig. 2 shows the accuracy vs. complexity tradeoff for various instantiations of Slow and SlowFast models. It is seen that doubling the number of frames in the Slow pathway increases performance (vertical axis) at double computational cost (horizontal axis), while SlowFast significantly extends the performance of all variants at small increase of computational cost, even if the Slow pathways operates on higher frame rate. Green arrows illustrate the gain of adding the Fast pathway to the corresponding Slow-only architecture. The red arrow illustrates that SlowFast provides higher accuracy and reduced cost.

Next, Table 5 shows a series of ablations on the Fast pathway design, using the default SlowFast, TT×\timesτ\tau == 4×\times16, R-50 instantiation (specified in Table 1), analyzed in turn.

The first two rows in Table 5a show the results for using the structure of one individual pathway alone. The default instantiations of the Slow and Fast pathway are very lightweight with only 27.3 and 6.4 GFLOPs, 32.4M and 0.53M parameters, producing 72.6% and 51.7% top-1 accuracy, respectively. The pathways are designed with their special expertise if they are used jointly, as is ablated next.

Table 5a shows various ways of fusing the Slow and Fast pathways. As a naïve fusion baseline, we show a variant using no lateral connection: it only concatenates the final outputs of the two pathways. This variant has 73.5% accuracy, slightly better than the Slow counterpart by 0.9%.

Next, we ablate SlowFast models with various lateral connections: time-to-channel (TtoC), time-strided sampling (T-sample), and time-strided convolution (T-conv). For TtoC, which can match channel dimensions, we also report fusing by element-wise summation (TtoC, sum). For all other variants concatenation is employed for fusion.

Table 5a shows that these SlowFast models are all better than the Slow-only pathway. With the best-performing lateral connection of T-conv, the SlowFast network is 3.0% better than Slow-only. We employ T-conv as our default.

Interestingly, the Fast pathway alone has only 51.7% accuracy (Table 5a). But it brings in up to 3.0% improvement to the Slow pathway, showing that the underlying representation modeled by the Fast pathway is largely complementary. We strengthen this observation by the next set of ablations.

A key intuition for designing the Fast pathway is that it can employ a lower channel capacity for capturing motion without building a detailed spatial representation. This is controlled by the channel ratio β\beta. Table 5b shows the effect of varying β\beta.

The best-performing β\beta values are 1/61/6 and 1/81/8 (our default). Nevertheless, it is surprising to see that all values from β\beta==1/321/32 to 1/41/4 in our SlowFast model can improve over the Slow-only counterpart. In particular, with β\beta==1/321/32, the Fast pathway only adds as small as 1.3 GFLOPs (∼\scriptstyle\sim5% relative), but leads to 1.6% improvement.

| model | pre-train | top-1 | top-5 | GFLOPs |
|---|---|---|---|---|
| 3D R-50 [56] | ImageNet | 73.4 | 90.9 | 36.7 |
| 3D R-50, recipe in [56] | - | 69.4 | 88.6 | 36.7 |
| 3D R-50, our recipe | - | 73.5 | 90.8 | 36.7 |

Further, we experiment with using different weaker spatial inputs to the Fast pathway in our SlowFast model. We consider: (i) a half spatial resolution (112×\times112), with β\beta==1/41/4 (vs. default 1/81/8) to roughly maintain the FLOPs; (ii) gray-scale input frames; (iii) “time difference" frames, computed by subtracting the current frame with the previous frame; and (iv) using optical flow as the input to the Fast pathway.

Table 5c shows that all these variants are competitive and are better than the Slow-only baseline. In particular, the gray-scale version of the Fast pathway is nearly as good as the RGB variant, but reduces FLOPs by ∼\scriptstyle\sim5%. Interestingly, this is also consistent with the M-cell’s behavior of being insensitive to colors [27, 37, 8, 14, 51].

We believe both Table 5b and Table 5c convincingly show that the lightweight but temporally high-resolution Fast pathway is an effective component for video recognition.

Our models are trained from scratch, without ImageNet training. To draw fair comparisons, it is helpful to check the potential impacts (positive or negative) of training from scratch. To this end, we train the exact same 3D ResNet-50 architectures specified in [56], using our large-scale SGD recipe trained from scratch.

Table 6 shows the comparisons using this 3D R-50 baseline architecture. We observe, that our training recipe achieves comparably good results as the ImageNet pre-training counterpart reported by [56], while the recipe in [56] is not well tuned for directly training from scratch. This suggests that our training system, as the foundation of our experiments, has no loss for this baseline model, despite not using ImageNet for pre-training.

## 5 Experiments: AVA Action Detection

The AVA dataset [20] focuses on spatiotemporal localization of human actions. The data is taken from 437 movies. Spatiotemporal labels are provided for one frame per second, with every person annotated with a bounding box and (possibly multiple) actions. Note the difficulty in AVA lies in action detection, while actor localization is less challenging [20]. There are 211k training and 57k validation video segments in AVA v2.1 which we use. We follow the standard protocol [20] of evaluating on 60 classes (see Fig. 3). The performance metric is mean Average Precision (mAP) over 60 classes, using a frame-level IoU threshold of 0.5.

Our detector is similar to Faster R-CNN [40] with minimal modifications adapted for video. We use the SlowFast network or its variants as the backbone. We set the spatial stride of res5 to 1 (instead of 2), and use a dilation of 2 for its filters. This increases the spatial resolution of res5 by 2×\times. We extract region-of-interest (RoI) features [17] at the last feature map of res5. We first extend each 2D RoI at a frame into a 3D RoI by replicating it along the temporal axis, similar to the method presented in [20]. Subsequently, we compute RoI features by RoIAlign [22] spatially, and global average pooling temporally. The RoI features are then max-pooled and fed to a per-class, sigmoid-based classifier for multi-label prediction.

We follow previous works that use pre-computed proposals [20, 46, 29]. Our region proposals are computed by an off-the-shelf person detector, i.e., that is not jointly trained with the action detection models. We adopt a person-detection model trained with Detectron [18]. It is a Faster R-CNN with a ResNeXt-101-FPN [60, 35] backbone. It is pre-trained on ImageNet and the COCO human keypoint images [36]. We fine-tune this detector on AVA for person (actor) detection. The person detector produces 93.9 AP@50 on the AVA validation set. Then, the region proposals for action detection are detected person boxes with a confidence of >> 0.8, which has a recall of 91.1% and a precision of 90.7% for the person class.

We initialize the network weights from the Kinetics-400 classification models. We use step-wise learning rate, reducing the learning rate 10×\times when validation error saturates. We train for 14k iterations (68 epochs for ∼\scriptstyle\sim211k data), with linear warm-up [19] for the first 1k iterations. We use a weight decay of 10-7. All other hyper-parameters are the same as in the Kinetics experiments. Ground-truth boxes are used as the samples for training. The input is instantiation-specific α​T\alpha T×\timesτ\tau frames of size 224×\times224.

We perform inference on a single clip with α​T\alpha T×\timesτ\tau frames around the frame that is to be evaluated. We resize the spatial dimension such that its shorter side is 256 pixels. The backbone feature extractor is computed fully convolutionally, as in standard Faster R-CNN [40].

| model | flow | video pretrain | val mAP | test mAP |
|---|---|---|---|---|
| I3D [20] |  | Kinetics-400 | 14.5 | - |
| I3D [20] | ✓ | Kinetics-400 | 15.6 | - |
| ACRN, S3D [46] | ✓ | Kinetics-400 | 17.4 | - |
| ATR, R50+NL [29] |  | Kinetics-400 | 20.0 | - |
| ATR, R50+NL [29] | ✓ | Kinetics-400 | 21.7 | - |
| 9-model ensemble [29] | ✓ | Kinetics-400 | 25.6 | 21.1 |
| I3D [16] |  | Kinetics-600 | 21.9 | 21.0 |
| SlowFast |  | Kinetics-400 | 26.3 | - |
| SlowFast |  | Kinetics-600 | 26.8 | - |
| SlowFast, +NL |  | Kinetics-600 | 27.3 | 27.1 |
| SlowFast*, +NL |  | Kinetics-600 | 28.2 | - |

### 5.1 Main Results

We compare with previous results on AVA in Table 7. An interesting observation is on the potential benefit of using optical flow (see column ‘flow’ in Table 7). Existing works have observed mild improvements: +1.1 mAP for I3D in [20], and +1.7 mAP for ATR in [29]. In contrast, our baseline improves by the Fast pathway by +5.2 mAP (see Table 9 in our ablation experiments in the next section). Moreover, two-stream methods using optical flow can double the computational cost, whereas our Fast pathway is lightweight.

As system-level comparisons, our SlowFast model has 26.3 mAP using only Kinetics-400 pre-training. This is 5.6 mAP higher than the previous best number under similar settings (21.7 of ATR [29], single-model), and 7.3 mAP higher than that using no optical flow (Table 7).

The work in [16] pre-trains on the larger Kinetics-600 and achieves 21.9 mAP. For fair comparison, we observe an improvement from 26.3 mAP to 26.8 mAP for using Kinetics-600. Augmenting SlowFast with NL blocks [56] increases this to 27.3 mAP. We train this model on train+val (and by 1.5×\times longer) and submit it to the AVA v2.1 test server [34]. It achieves 27.1 mAP single crop test set accuracy.

By using predicted proposals overlapping with ground-truth boxes by IoU >> 0.9, in addition to the ground truth boxes, for training we achieve 28.2 mAP single crop validation accuracy, a new state-of-the-art on AVA.

| model | flow | video pretrain | val mAP | test mAP |
|---|---|---|---|---|
| SlowFast, 8×\times8 |  | Kinetics-600 | 29.0 | - |
| SlowFast, 16×\times8 |  | Kinetics-600 | 29.8 | - |
| SlowFast++, 16×\times8 |  | Kinetics-600 | 30.7 | - |
| SlowFast++, ensemble |  | Kinetics-600 | - | 34.3 |

Using the AVA v2.2 dataset (which provides more consistent annotations) improves this number to 29.0 mAP (Table 8). The longer-term SlowFast, 16×\times8 model produces 29.8 mAP and using multiple spatial scales and horizontal flip for testing, this number is increased to 30.7 mAP.

Finally, we create an ensemble of 7 models and submit it to the official test server for the ActivityNet challenge 2019 [1]. As shown in Table 8 this entry (SlowFast++, ensemble) achieved 34.3 mAP accuracy on the test set, ranking first in the AVA action detection challenge 2019. Further details on our winning solution are provided in the corresponding technical report [11].

### 5.2 Ablation Experiments

Table 9 compares a Slow-only baseline with its SlowFast counterpart, with the per-category AP shown in Fig. 3. Our method improves massively by 5.2 mAP (relative 28%) from 19.0 to 24.2. This is solely contributed by our SlowFast idea.

Category-wise (Fig. 3), our SlowFast model improves in 57 out of 60 categories, vs. its Slow-only counterpart. The largest absolute gains are observed for “hand clap" (+27.7 AP), “swim" (+27.4 AP), “run/jog" (+18.8 AP), “dance" (+15.9 AP), and “eat” (+12.5 AP). We also observe large relative increase in “jump/leap”, “hand wave”, “put down”, “throw”, “hit” or “cut”. These are categories where modeling dynamics are of vital importance. The SlowFast model is worse in only 3 categories: “answer phone" (-0.1 AP), “lie/sleep" (-0.2 AP), “shoot" (-0.4 AP), and their decrease is relatively small vs. others’ increase.

## 6 Conclusion

The time axis is a special dimension. This paper has investigated an architecture design that contrasts the speed along this axis. It achieves state-of-the-art accuracy for video action classification and detection. We hope that this SlowFast concept will foster further research in video recognition.

| model | T×τ{T\times\tau} | α\alpha | mAP |
|---|---|---|---|
| Slow-only, R-50 | 4×\times16 | - | 19.0 |
| SlowFast, R-50 | 4×\times16 | 8 | 24.2 |

## Appendix A Appendix

We study backbones including ResNet-50 and the deeper ResNet-101 [24], optionally augmented with non-local (NL) blocks [56]. For models involving R-101, we use a scale jittering range of [256, 340]. The TT×\timesτ\tau == 16×\times8 models are initilaized from the 8×\times8 counterparts and trained for half the training epochs to reduce training time. For all models involving NL, we initialize them with the counterparts that are trained without NL, to facilitate convergence. We only use NL on the (fused) Slow features of res4 (instead of res3+res4 [56]).

On Kinetics, we adopt synchronized SGD training in 128 GPUs following the recipe in [19], and we found its accuracy is as good as typical training in one 8-GPU machine but it scales out well. The mini-batch size is 8 clips per GPU (so the total mini-batch size is 1024). We use the initialization method in [23]. We train with Batch Normalization (BN) [28] with BN statistics computed within each 8 clips. We adopt a half-period cosine schedule [38] of learning rate decaying: the learning rate at the nn-th iteration is η⋅0.5​[cos⁡(nnmax​π)+1]\eta\cdot 0.5[\cos(\frac{n}{n_{\text{max}}}\pi)+1], where nmaxn_{\text{max}} is the maximum training iterations and the base learning rate η\eta is set as 1.6. We also use a linear warm-up strategy [19] in the first 8k iterations. For Kinetic-400, we train for 256 epochs (60k iterations with a total mini-batch size of 1024, in ∼\scriptstyle\sim240k Kinetics videos) when T≤T\leq 4 frames, and 196 epochs when T>T> 4 frames: it is sufficient to train shorter when a clip has more frames. We use momentum of 0.9 and weight decay of 10-4{}^{\text{-4}}. Dropout [25] of 0.5 is used before the final classifier layer.

For Kinetics-600, we extend the training epochs (and schedule) by 2×\times and set the base learning rate η\eta to 0.8.

For Charades, we fine-tune the Kinetics models. A per-class sigmoid output is used to account for the mutli-class nature. We train on a single machine for 24k iterations using a batch size of 16 and a base learning rate of 0.0375 (Kinetics-400 pre-trained) and 0.02 (Kinetics-600 pre-trained) with 10×\times step-wise decay if the validation error saturates. For inference, we temporally max-pool scores [56].
