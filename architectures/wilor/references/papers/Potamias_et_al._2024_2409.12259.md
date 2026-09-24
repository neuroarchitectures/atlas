# WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild

In recent years, 3D hand pose estimation methods have garnered significant attention due to their extensive applications in human-computer interaction, virtual reality, and robotics. In contrast, there has been a notable gap in hand detection pipelines, posing significant challenges in constructing effective real-world multi-hand reconstruction systems. In this work, we present a data-driven pipeline for efficient multi-hand reconstruction in the wild. The proposed pipeline is composed of two components: a real-time fully convolutional hand localization and a high-fidelity transformer-based 3D hand reconstruction model. To tackle the limitations of previous methods and build a robust and stable detection network, we introduce a large-scale dataset with over than 2M in-the-wild hand images with diverse lighting, illumination, and occlusion conditions. Our approach outperforms previous methods in both efficiency and accuracy on popular 2D and 3D benchmarks. Finally, we showcase the effectiveness of our pipeline to achieve smooth 3D hand tracking from monocular videos, without utilizing any temporal components. Code, models, and dataset are available on our project page.

## 1 Introduction

Hand detection and reconstruction has been a long-studied problem due to its numerous applications, ranging from virtual reality [23] to sign language [2, 108, 31] and human behaviour recognition [68]. Given the large variations in hand appearance and articulation [67] along with heavy occlusions and motion blur [25, 27] that are usually present in hand interactions, the task of hand pose estimation is considerably challenging. Over the years, several methods have been proposed to tackle 3D hand pose estimation [51, 12, 44, 63]. However, despite producing credible results, these methods primarily focus on images containing a fixed number of hands and hence cannot generalize to in-the-wild images.

In the closely related fields of 3D human body and face reconstruction, state-of-the-art methods [8, 18, 24, 22] employ bottom-up pipelines founded on top of high-performance detection models that initially localize human body and face within the image, enabling their generalization to in-the-wild images. Despite the numerous methods that have been proposed to solve the task of human body and face detection, there has been a notable lack in real-time hand detection methods. The importance of hand detectors is further emphasized considering that current 3D hand pose estimation frameworks operate on tight crops around the hand regions [11, 103]. Consequently, hand detectors are essential for the generalisation of such methods in in-the-wild scenarios. Popular hand detection and localisation methods [97, 8] fail significantly to detect multiple hands and challenging poses, while more recent methods [54, 55, 41] albeit producing reasonable results struggle to operate in real-time. Motivated by the lack of accurate hand detection frameworks, we propose a robust single-state anchor-free detector that can operate in over than 100 frames-per-second (fps). As we experimentally show, robust detections can enforce more stable 4D reconstructions and overcome jittering artifacts which is currently one of the main limitations of 3D frame-based pose estimation methods.

In contrast to the relatively unexplored hand detection and localization problem, 3D hand pose estimation has received significantly more attention. Initial 3D pose reconstruction methods have focused on traditional convolution-based backbones to process and extract image features [5, 38, 51, 12]. Following the success of transformers and their ability to consume large amounts of data [88, 15, 39, 73, 86, 99], several methods have paved the way of utilising transformer architectures scaling up the 3D human body and hand recovery [43, 44, 63]. Recently, Pavlakos et al. [63] showcased the effectiveness of vision transformers (ViT) using a simple yet powerful framework trained on a large-scale dataset. The key to the success of this method lies in the scale of its architecture, composed of more than 0.5 billion parameters, enabling effective consumption of large amount of data. However, as shown in the literature [57, 82, 42, 98], regressing the hand parameters from a single image results in poor alignment and incorrect poses. Currently, methods that aim to achieve better image alignment rely on sub-optimal solutions, such as intermediate heatmap representations [36, 11, 103]. To tackle this, we propose a high-fidelity 3D pose estimation method that decomposes 3D hand reconstruction into two stages. In particular, the decoder first predicts a rough hand estimation that is used to extract multi-scale image-aligned features from our refinement module. By leveraging the rough hand estimation, we can extract meaningful spatial features that lead to better image alignment and state-of-the-art performance on FreiHand [107] and HO3D [25] benchmark datasets. Additionally, in contrast to vertex regression methods [38, 43, 44] that directly regress 3D vertices, our method predicts MANO parameters [74], ensuring both explainable and plausible hand poses.

In this paper, we propose a high fidelity full stack method that can reconstruct 3D hands in real-time. Specifically:

- • Based on the limitations of current hand detection benchmark datasets, we collect a large-scale dataset of in-the-wild images that contain multiple hands and introduce a challenging benchmark for hand detection. We make the dataset, along with the corresponding 2D and 3D annotations, publicly available.
Based on the limitations of current hand detection benchmark datasets, we collect a large-scale dataset of in-the-wild images that contain multiple hands and introduce a challenging benchmark for hand detection. We make the dataset, along with the corresponding 2D and 3D annotations, publicly available.

- • We propose a real-time hand detection method trained on the collected large-scale dataset that outperforms previous hand detection methods by a large margin in both accuracy and efficiency.
We propose a real-time hand detection method trained on the collected large-scale dataset that outperforms previous hand detection methods by a large margin in both accuracy and efficiency.

- • We propose a transformer-based method that facilitates high fidelity 3D reconstructions and tackles the architectural limitations of previous methods using a novel refinement module. The proposed method, apart from highly efficient, achieves state-of-the-art performance in both Freihand and HO3D benchmark datasets.
We propose a transformer-based method that facilitates high fidelity 3D reconstructions and tackles the architectural limitations of previous methods using a novel refinement module. The proposed method, apart from highly efficient, achieves state-of-the-art performance in both Freihand and HO3D benchmark datasets.

## 2 Related Work

Hand Detection and Tracking. Object detection has been extensively studied in the literature achieving remarkable advancements [49, 46, 70] and setting the foundations for human body [83, 65, 20, 60, 18, 94] and face detection [13, 105] pipelines. In contrast, despite two decades of research efforts [72], hand detection has not yet achieved comparable breakthroughs. Initial approaches used controlled conditions and depth cameras [91, 104, 81, 90, 80] to detect and track human hands. Several efforts have been made to boost hand detection under different skin tones and backgrounds using multi-stage frameworks [50, 64], however, they fail to generalize in challenging environments. Following the success in object detection, several methods have adopted fully convolutional architectures for hand detection [30, 75, 14, 97, 76]. Simon et al. [78] introduced a multi-view bootstrapping procedure to annotate in-the-wild data and train a real-time convolutional detector network. Recently, Narasimhaswamy et al. [54] proposed an extension of MaskRCNN [29] network to detect in-the-wild hands and identifying their corresponding contact points [55] and body associations [56]. Nevertheless, despite the extensive efforts in the literature, most methods rely on slow backbones and struggle with challenging images. The primary issue is the lack of large-scale training data featuring multiple levels of occlusions and motion blur from in-the-wild scenes. To tackle such limitation, we propose a lightweight hand detector that is 45 ×\times faster compared to previous state-of-the-art detectors, trained on 2M in-the-wild images with diverse environments and occlusion.

3D Hand Pose Estimation. Similar to hand detection, initial approaches for hand pose estimation relied on depth cameras [59, 84, 19] to reconstruct 3D hands. Boukhayma et al. [5] introduced the first fully learnable pipeline that directly regresses the parameters of the MANO hand model [74] from RGB images. In a similar manner, several follow-up works used heatmaps [100] and iterative refinement [1] to enforce 2D alignment. Kulon et al. [37, 38] introduced an alternative regression method that directly regresses 3D vertices using spiral graph neural networks [6, 66], which significantly outperformed previous methods. Various approaches have been proposed to improve task-specific challenges of 3D pose estimation, including robustness to occlusions [61] and motion blur [58] and reducing inference speed [11, 103]. Recently, Pavlakos et al. [63] highlighted the importance of scaling up both the training data and the capacity of the model. Specifically, building on the success of the Vision Transformer (ViT) backbones for body pose estimation [42, 21, 7], they demonstrated that using a simple yet effective large-scale transformer architecture can achieve state-of-the-art performance when trained on a diverse collection of datasets. However, directly regressing MANO parameters from the image in one go may introduce misalignments and incorrect poses. To tackle this, we propose a novel refinement layer that deforms hand pose using mesh-aligned multi-scale features.

## 3 WHIM Dataset

A vital cause behind the lack of high-fidelity hand detection systems lies in the limited amount of in-the-wild datasets with multiple hand annotations. To build a robust hand detection framework, we collected a large-scale dataset with millions of in-the-wild hands (WHIM) with diverse poses, illuminations, occlusions, and skin tones.

To collect the proposed dataset, we devised a pipeline to automatically annotate YouTube videos from diverse and challenging in-the-wild scenarios. In particular, we selected more than 1,400 YouTube videos containing hand activities including sign language, cooking, everyday activities, sports, and games with ego- and exo-centric viewpoints, motion blur, different hand scales, and interactions. To accurately detect and annotate the hands on each frame we used a combination of ensemble networks. First, we used VitPose [94] and AlphaPose [18] to detect all humans in the frame and selected the bounding boxes with confidence bigger than 0.65. We then cropped the bounding boxes and fed them to an ensemble hand detection pipeline that consists of MediaPipe [97], OpenPose [8] and ContactHands [54] models. To localize the hand, we used a weighted average between the bounding box positions 𝐛i\mathbf{b}_{i} of the three detectors did_{i}, scaled from their corresponding confidence P⁡(𝐛i|di)P(\mathbf{b}_{i}|d_{i}):

|  | y^=∑iP⁡(𝐛i|di)​𝐛i∑iP⁡(𝐛i|di)\hat{y}=\frac{\sum_{i}P(\mathbf{b}_{i}|d_{i})\mathbf{b}_{i}}{\sum_{i}P(\mathbf{b}_{i}|d_{i})} |  | (1) |
|---|---|---|---|

where 𝐛i{\mathbf{b}_{i}} denotes the estimated hand bounding box.

In addition to the bounding box, we used the estimated 2D landmarks [97, 8], to fit a 3D parametric hand model ℳ\mathcal{M} [74]. More specifically, we optimized shape β\mathbf{\beta} and pose θ\mathbf{\theta} parameters to minimize the re-projection loss ℒp​r​o​j\mathcal{L}_{proj} between the regressed 𝐉ℳ{\mathbf{J}_{\mathcal{M}}} and the estimated landmarks 𝐉^𝐬\mathbf{\hat{J}_{s}}:

|  | ℒp​r​o​j=‖𝐉ℳ−π⁡(𝐉^𝐬,K)‖1,\mathcal{L}_{proj}=||\mathbf{J}_{\mathcal{M}}-\pi(\mathbf{\hat{J}_{s}},K)||_{1}, |  | (2) |
|---|---|---|---|

where π⁡(⋅)\pi(\cdot) denotes the weak perspective projection transform and KK the estimated intrinsic camera matrix.

Given the degrees of freedom of the human hand, optimizing the hand model using joint terms usually results in unnatural poses. To tackle the ambiguities during the optimization process, we followed [79] and included bio-mechanical losses to constrain the optimization. In particular, apart from the re-projection error, we enhanced the fitting process using loss functions that constrain the bone lengths and the angle rotations to feasible ranges, as defined in [79]:

|  | ℒB​M​C=ℒB​L+ℒA\mathcal{L}_{BMC}=\mathcal{L}_{BL}+\mathcal{L}_{A} |  | (3) |
|---|---|---|---|

where ℒB​L\mathcal{L}_{BL} and ℒA\mathcal{L}_{A} denote bone length and joint angle loss terms, respectively. For additional details of the bio-mechanical constraints we refer the reader to [79].

Finally, given that the bio-mechanical prior acts mainly on the joint space, we followed [2] and trained a PCA model on ARCTIC dataset [17] acting as a 3D prior, modeling the distribution of feasible hand poses. We formulated the prior loss as the reconstruction error of the 3D mesh 𝐗\mathbf{X} projected and reconstructed from the PCA space 𝐔\mathbf{U}, as:

|  | ℒp​r​i​o​r=‖𝐗−[(𝐗−𝝁)​𝐔T]​𝐔+𝝁‖2,\mathcal{L}_{prior}=||\mathbf{X}-[(\mathbf{X}-\boldsymbol{\mu})\mathbf{U}^{T}]\mathbf{U}+\boldsymbol{\mu}||_{2}, |  | (4) |
|---|---|---|---|

where 𝐔∈ℝd×N⋅3\mathbf{U}\in\mathbb{R}^{d\times N\cdot 3} denotes the eigenvector basis of dd components and 𝝁\boldsymbol{\mu} the mean mesh. Fig. 2 includes several examples of the proposed WHIM dataset.

## 4 Method

### 4.1 Hand Detection and Localization

Over the past years, fully convolutional networks (FCNs) have shown remarkable efficiency in human detection [89] and object detection [71]. Building on their success, we employ an FCN architecture to achieve both accurate and real-time hand localization. Similar to object detection frameworks, given an image 𝐈∈ℝH×W×3\mathbf{I}\in\mathbb{R}^{H\times W\times 3} our goal is to detect the bounding boxes 𝐁={𝐛j∈ℝ4:0≤j≤n}\mathbf{B}=\{\mathbf{b}_{j}\in\mathbb{R}^{4}:0\leq j\leq n\} of the nn hands present in the image along with their hand side label 𝐲j\mathbf{y}_{j}. We follow the commonly used one-stage backbone-neck-head formulation and we built upon the powerful and efficient DarkNet backbone [70]. We extract the last three feature maps {C​3,C​4,C​5}\{C3,C4,C5\} of the backbone to generate a multi-scale feature pyramid in the neck module. To enable our model to effectively capture multi-scale features using both top-down and bottom-up pathways across different feature maps, we utilized Path Aggregation Network (PANet) [47], an extension of Feature Pyramid Network [45] that facilitates fine-grained information flow using a bottom-up path augmentation. Finally, we use three detection heads to predict the bounding boxes 𝐛j\mathbf{b}_{j} and hand side labels 𝐲j\mathbf{y}_{j} at different anchor resolutions. Following [16], we adopt an anchor-free design to enhance the flexibility of our localization method and directly predict bounding box coordinates without relying on predefined anchor boxes. An overview of the proposed detection network is visualized in Fig. 3. Similar to [13], we observed that joint keypoint supervision significantly improved the performance and the robustness of the detector. The full training objective can be defined as:

|  | ℒ=λ0​ℒB​C​E+λ1​ℒD​F​L+λ2​ℒC​I​o​U+λ3​ℒk​p​t​s\mathcal{L}=\lambda_{0}\mathcal{L}_{BCE}+\lambda_{1}\mathcal{L}_{DFL}+\lambda_{2}\mathcal{L}_{CIoU}+\lambda_{3}\mathcal{L}_{kpts} |  | (5) |
|---|---|---|---|

where ℒB​C​E\mathcal{L}_{BCE} is the binary cross entropy loss between the predicted and the ground truth box labels, ℒD​F​L\mathcal{L}_{DFL} denotes the distributional focal loss [40] which measures the difference between the predicted and the ground truth bounding box distributions, ℒC​I​o​U\mathcal{L}_{CIoU} measures the discrepancy between the predicted and the ground truth bounding box [101], ℒk​p​t​s\mathcal{L}_{kpts} denotes an L​2L2 loss on the and λ0,λ1,λ2,λ3\lambda_{0},\lambda_{1},\lambda_{2},\lambda_{3} are weights that balance the losses.

### 4.2 Hand Reconstruction

Given an image 𝐈𝐡∈ℝH×W×3\mathbf{I_{h}}\in\mathbb{R}^{H\times W\times 3} that contains a human hand, tightly cropped around the hand detectors bounding box, the proposed 3D hand reconstruction method estimates the corresponding hand pose θ∈ℝ48\theta\in\mathbb{R}^{48} and shape β∈ℝ10\beta\in\mathbb{R}^{10} MANO [74] parameters along with the camera parameters 𝐊c​a​m={𝐭c​a​m,𝐬c​a​m}\mathbf{K}_{cam}=\{\mathbf{t}_{cam},\mathbf{s}_{cam}\} to obtain a 3D hand.

To build a powerful 3D pose estimation network that can scale on large amounts of data, we follow [42, 7, 63], and built our backbone using a pre-trained ViT encoder [88, 94]. The image 𝐈𝐡\mathbf{I_{h}} is first split into MM-size patches 𝐏∈ℝH​WM2×(M2×3)\mathbf{P}\in\mathbb{R}^{\frac{HW}{M^{2}}\times(M^{2}\times 3)} and then embedded to high dimensional tokens 𝐓i​m​g∈ℝH​WM2×C\mathbf{T}_{img}\in\mathbb{R}^{\frac{HW}{M^{2}}\times C}. To uniquely encode their spatial location, positional embeddings 𝐏e\mathbf{P}_{e} are added to the image tokens 𝐓i​m​g\mathbf{T}_{img} [88]. In addition to the image tokens, we explicitly model hand pose, shape, and camera parameters with three distinct tokens 𝐓p​o​s​e,𝐓s​h​a​p​e,𝐓c​a​m\mathbf{T}_{pose},\mathbf{T}_{shape},\mathbf{T}_{cam}. We then feed the concatenated tokens to the ViT transformer encoder to obtain a set of updated feature tokens 𝐓′i​m​g,𝐓′p​o​s​e,𝐓′s​h​a​p​e,𝐓′c​a​m\mathbf{T^{\prime}}_{img},\mathbf{T^{\prime}}_{pose},\mathbf{T^{\prime}}_{shape},\mathbf{T^{\prime}}_{cam}. Using a set of MLP layers we regress a rough estimation of pose θc\theta^{c} and shape βc\beta^{c} parameters of the MANO model, which will serve as a prior for the refinement network. Similarly, we regress the camera translation and scale parameters 𝐊c​a​m={𝐭c​a​m,𝐬c​a​m}\mathbf{K}_{cam}=\{\mathbf{t}_{cam},\mathbf{s}_{cam}\} from the camera token features.

Multi-Scale Pose Refinement Module. In order to get better image alignment and more accurate hand pose, we introduce a fully differentiable refinement module that predicts pose and shape residuals of the rough hand estimation. To achieve this, we utilize image features extracted from the ViT backbone as 2D feature cues within our refinement module. In particular, we reshape image feature tokens 𝐓′i​m​g\mathbf{T^{\prime}}_{img} to form a low resolution feature map 𝐅0∈ℝHM×WM×C\mathbf{F}_{0}\in\mathbb{R}^{\frac{H}{M}\times\frac{W}{M}\times C} and project the rough hand estimation ℳl\mathcal{M}_{l} to feature map using the estimated 𝐭c​a​m,𝐬c​a​m\mathbf{t}_{cam},\mathbf{s}_{cam} camera parameters. Then, using bilinear interpolation we sample from 𝐅0\mathbf{F}_{0} a feature vector 𝐟0𝐯\mathbf{f}^{\mathbf{v}}_{0} for each projected vertex 𝐯\mathbf{v}:

|  | 𝐟0𝐯=π⁡(𝐯,𝐊c​a​m)\mathbf{f}^{\mathbf{v}}_{0}=\pi(\mathbf{v},\mathbf{K}_{cam}) |  | (6) |
|---|---|---|---|

where π⁡(⋅)\pi(\cdot) denotes the weak perspective projection.

Note that we project the whole hand mesh ℳl\mathcal{M}_{l} to the feature map, instead of just the hand joints, as we aim to acquire better shape and pose image alignment. The image-aligned vertex features are then aggregated to form a global feature vector that is used to regress pose and shape residuals:

|  | Δ​β\displaystyle\Delta\beta | =M​L​Pβ​(□𝐯∈ℳl​𝐟0𝐯)\displaystyle=MLP_{\beta}(\square_{\mathbf{v}\in\mathcal{M}_{l}}\mathbf{f}^{\mathbf{v}}_{0}) |  | (7) |
|---|---|---|---|---|
|  | Δ​θ\displaystyle\Delta\theta | =M​L​Pθ​(□𝐯∈ℳl​𝐟0𝐯)\displaystyle=MLP_{\theta}(\square_{\mathbf{v}\in\mathcal{M}_{l}}\mathbf{f}^{\mathbf{v}}_{0}) |  |  |

where □\square denotes the aggregation function, e.g., mean, max, sum.

Given that the initial feature map is very low-dimensional, we use a set of deconvolutional layers to upsample 𝐅0\mathbf{F}_{0} to multiple higher resolution feature maps 𝐅0,𝐅1,…,𝐅n{\mathbf{F}_{0},\mathbf{F}_{1},...,\mathbf{F}_{n}} that will serve as multi-scale features for the proposed refinement module. Intuitively, low-dimensional feature maps will provide global and structural residuals of the hand shape while more high-resolution features provide finer details of the hand pose.

Loss function. The proposed model is trained with supervision for 3D vertices 𝐕^3​D\hat{\mathbf{V}}_{3D}, 2D joints 𝐉^2​D\hat{\mathbf{J}}_{2D} as well as MANO parameters θ^,β^\hat{\theta},\hat{\beta}, when available. Additionally, following [35, 63], we utilize a discriminator network DD to enforce plausible hand poses and shapes and penalize irregular articulations. The full loss function can be defined as:

|  |  | ℒ=ℒ3​D+ℒ2​D+ℒm​a​n​o+ℒa​d​v,\displaystyle\mathcal{L}=\mathcal{L}_{3D}+\mathcal{L}_{2D}+\mathcal{L}_{mano}+\mathcal{L}_{adv}, |  | (8) |
|---|---|---|---|---|
|  |  | ℒ3​D=‖𝐕3​D−𝐕^3​D‖1,\displaystyle\mathcal{L}_{3D}=||\mathbf{V}_{3D}-\hat{\mathbf{V}}_{3D}||_{1}, |  |  |
|  |  | ℒ2​D=‖π⁡(𝐉3​D,𝐊c​a​m)−𝐉^2​D‖1\displaystyle\mathcal{L}_{2D}=||\pi(\mathbf{J}_{3D},\mathbf{K}_{cam})-\hat{\mathbf{J}}_{2D}||_{1} |  |  |
|  |  | ℒm​a​n​o=‖θ−θ^‖22+‖β−β^‖22,\displaystyle\mathcal{L}_{mano}=||\theta-\hat{\theta}||_{2}^{2}+||\beta-\hat{\beta}||_{2}^{2}, |  |  |
|  |  | ℒadv=‖D⁡(θ,β)−1‖2.\displaystyle\mathcal{L}_{\texttt{adv}}=||D(\theta,\beta)-1||_{2}. |  |  |

## 5 Experiments

In this section we first evaluate the proposed hand detection network using established benchmarks to assess its performance. Next, we conduct an extensive qualitative and quantitative analysis of the proposed 3D hand pose estimation method. Finally, we demonstrate the critical role of precise hand localization in the accuracy of 4D hand reconstruction.

### 5.1 Evaluation of Hand Detection and Localization

Training. We train the proposed hand detection network using the curated WHIM dataset that consists of over 2M in-the-wild images of multiple hands and scales. To further boost the generalization and robustness of our network, we follow several data augmentations during training. Particularly, we introduce random rotations in the range of [−60∘,60∘][-60^{\circ},60^{\circ}] and translations in the range of [−0.1,0.1][-0.1,0.1] along with random masking and cropping of the image. Additionally, in each training batch, we follow mosaic and mixup augmentation, which significantly affects the robustness to diverse hand scales.

Evaluation. To compare our network, we employ popular baselines such as OpenPose [8] and Mediapipe [97], which are widely used across the community [69, 67], along with more recent hand detection pipelines such as ContactHands [54] and ViTDet [41]. All methods are evaluated under three criteria: i) the inference speed in terms of frames per second (FPS), ii) the detection performance in terms of average precision (AP) at IoU = 0.5 and mean AP at different IoU=0.5:0.05:0.95 thresholds and iii) the model size measured in Mb. An optimal hand detection system should be lightweight to ensure compatibility with mobile devices, operate in real-time to avoid impacting the runtime of a 3D pose estimation pipeline while achieving precise detections.

In Tab. 1, we evaluate the proposed and the baseline methods on three datasets: the proposed WHIM dataset along with the benchmark Coco-WholeBody [33] and Oxford-Hands [50] dataset. All experiments were conducted on a NVIDIA RTX 4090 GPU. As can be easily seen, the proposed medium size detector, Proposed-M, can run at more than 130 FPS while the small version, Proposed-S, can achieve up to 175 FPS, improving the mAP metric, on average, by 26% compared to previous state-of-the-art model. In addition, compared to previous state-of-the-art model, ContactHands [54], the proposed detector is 45×45\times faster and has 32×32\times reduced model size which enables the utilization of the proposed detector in mobile applications and heavy pipelines without posing any significant overhead. It is important to note that despite the varying resolutions and hand scales across the three datasets, the proposed model consistently outperforms the baseline methods. This is particularly evident on the COCO-WholeBody dataset [33], an extension of the COCO dataset that includes full-body images, where the hands are relatively small compared to the overall image size.

|  |  |  | Coco-Whole | Oxford-Hands | WHIM |  |  |  |
|---|---|---|---|---|---|---|---|---|
| Method | Size (Mb)↓\downarrow | FPS ↑\uparrow | AP0.5 ↑\uparrow | mAP ↑\uparrow | AP0.5 ↑\uparrow | mAP ↑\uparrow | AP0.5 ↑\uparrow | mAP↑\uparrow |
| MediaPipe [97] | 25 | 25 | 15.43 | 3.72 | 8.72 | 1.80 | 53.09 | 12.01 |
| OpenPose [8] | 141 | 29 | 37.05 | 9.06 | 20.74 | 4.41 | 76.8 | 34.25 |
| ContactHands [55] | 819 | 3 | 50.29 | 16.67 | 70.02 | 36.41 | 93.42 | 49.44 |
| ViTDet [41] | 1400 | 1 | 41.64 | 13.21 | 67.56 | 29.77 | 84.76 | 35.42 |
| Proposed-S | 7(×3.5↓\times 3.5\downarrow) | 175(×6↑\times 6\uparrow) | 46.96 | 18.56 | 75.21 | 38.16 | 91.80 | 46.50 |
| Proposed-M | 25 | 138 | 62.48 | 25.97 | 82.64 | 48.98 | 96.06 | 53.79 |

Ablation. The efficiency and accuracy of the proposed hand detection method are mainly attributed to the selection of the backbone architecture and the utilization of a large-scale training dataset. We further evaluate the contribution of each component using an ablation study on OxfordHands [50] and WHIM datasets where we utilized different backbone networks and training datasets. As can be observed from Tab. 2, training the detection network with different backbones, apart from achieving similar detection performance, significantly degrades the inference speed of the network. The importance of the proposed large-scale in-the-wild WHIM dataset is also validated in Tab. 2, where we can observe a significant performance drop when the model was trained with significantly less data, e.g., Proposed w. 0.25M, Proposed w. 0.5M, Proposed w. 1M. An interesting observation highlighting the versatility of the WHIM dataset is that the proposed model achieves better performance when trained on WHIM compared to the OxfordHands dataset. Finally, we evaluate the contribution of the proposed augmentation strategy and the use of landmark regression loss. The augmentation strategy significantly contributes to cross-dataset generalization, achieving 14% increase on mAP. Similarly, we can observe that incorporating landmark regression loss enhances the detector’s precision, leading to more robust detections.

|  |  |  | OxfordHands | WHIM |  |  |
|---|---|---|---|---|---|---|
| Method | Size (Mb) ↓\downarrow | FPS ↑\uparrow | AP0.5 ↑\uparrow | mAP ↑\uparrow | AP0.5 ↑\uparrow | mAP ↑\uparrow |
| Proposed-w. 0.25M | - | - | 49.15 | 26.69 | 75.75 | 38.48 |
| Proposed-w. 0.5M | - | - | 58.32 | 35.15 | 83.03 | 42.11 |
| Proposed-w. 1M | - | - | 69.21 | 43.04 | 88.37 | 47.92 |
| Proposed-w.OxfordHands | - | - | 68.15 | 40.34 | 70.14 | 35.29 |
| Proposed-w. ResNet50 | 118 | 34 | 74.13 | 47.34 | 95.43 | 51.85 |
| Proposed-w. HRNet | 132 | 30 | 84.82 | 49.83 | 97.23 | 54.12 |
| Proposed-w/o Augmentations | - | - | 70.76 | 42.17 | 91.13 | 49.93 |
| Proposed-w/o Landmark Loss | - | - | 72.57 | 45.96 | 92.43 | 51.44 |
| Proposed-M | 25 (↓5×\downarrow 5\times) | 138 (↑4×\uparrow 4\times) | 82.64 | 48.98 | 96.06 | 53.79 |

### 5.2 Evaluation of 3D Hand Pose Estimation

Training. Following [7, 42, 63], we trained the proposed hand regressor using a combination of datasets to improve robustness to diverse poses, illuminations and occlusions. Particularly, we utilized a set of datasets containing both 2D and 3D annotations namely FreiHAND [107], HO3D [25], MTC [92], RHD [106], InterHand2.6M [52], H2O3D [25], DEX YCB [9], COCO WholeBody [33], Halpe [18] MPII NZSL [78], BEDLAM [4], ARCTIC [17], Re:InterHand [53] and Hot3D [3]. In total we utilized 4.2M images, 55% more than previous state-of-the-art.

Evaluation. To compare the proposed method we employ with state-of-the-art methods including METRO [43], Mesh Graphormer [44], AMVUR [32], MobRecon [11], HaMeR [63] and SimpleHand [103]. In Tab. 3 and Tab. 4 we report the reconstruction results the popular benchmark FreiHAND [107] and HO3Dv2 [25] datasets. Following the common protocol [107], we measure the reconstruction performance in terms of Procrustes Aligned Mean per Joint and Vertex Error (PA-MPJPE, PA-MPVPE) along with the fraction of poses with less than 5mm and 15mm error (F@5, F@15). Additionally, we report Area Under the Curve for 3D joints and vertices (AUCJ\textrm{AUC}_{\textrm{J}}, AUCV\textrm{AUC}_{\textrm{V}}) for HO3D dataset. The proposed method achieves state-of-the-art performance and outperforms previous methods under all metrics on both benchmark datasets, which can be further validated qualitatively in Fig. 6. Leveraging the image-aligned features of the refinement module, WiLoR achieves high fidelity reconstructions even in challenging articulations.

| Method | PA-MPJPE ↓\downarrow | PA-MPVPE ↓\downarrow | F@5 ↑\uparrow | F@15 ↑\uparrow |
|---|---|---|---|---|
| I2L-MeshNet [51] | 7.4 | 7.6 | 0.681 | 0.973 |
| Pose2Mesh [12] | 7.7 | 7.8 | 0.674 | 0.969 |
| I2UV-HandNet [10] | 6.7 | 6.9 | 0.707 | 0.977 |
| METRO [43] | 6.5 | 6.3 | 0.731 | 0.984 |
| Tang et al. [85] | 6.7 | 6.7 | 0.724 | 0.981 |
| Mesh Graphormer [44] | 5.9 | 6.0 | 0.764 | 0.986 |
| MobRecon [11] | 5.7 | 5.8 | 0.784 | 0.986 |
| AMVUR [32] | 6.2 | 6.1 | 0.767 | 0.987 |
| HaMeR [63] | 6.0 | 5.7 | 0.785 | 0.990 |
| Proposed | 5.5 | 5.1 | 0.825 | 0.993 |

| Method | AUCJ\textrm{AUC}_{\textrm{J}} ↑\uparrow | PA-MPJPE ↓\downarrow | AUCV\textrm{AUC}_{\textrm{V}} ↑\uparrow | PA-MPVPE ↓\downarrow | F@5 ↑\uparrow | F@15 ↑\uparrow |
|---|---|---|---|---|---|---|
| Liu et al. [48] | 0.803 | 9.9 | 0.810 | 9.5 | 0.528 | 0.956 |
| HandOccNet [62] | 0.819 | 9.1 | 0.819 | 8.8 | 0.564 | 0.963 |
| I2UV-HandNet [10] | 0.804 | 9.9 | 0.799 | 10.1 | 0.500 | 0.943 |
| Hampali et al. [25] | 0.788 | 10.7 | 0.790 | 10.6 | 0.506 | 0.942 |
| Hasson et al. [28] | 0.780 | 11.0 | 0.777 | 11.2 | 0.464 | 0.939 |
| ArtiBoost [95] | 0.773 | 11.4 | 0.782 | 10.9 | 0.488 | 0.944 |
| Pose2Mesh [12] | 0.754 | 12.5 | 0.749 | 12.7 | 0.441 | 0.909 |
| I2L-MeshNet [51] | 0.775 | 11.2 | 0.722 | 13.9 | 0.409 | 0.932 |
| METRO [43] | 0.792 | 10.4 | 0.779 | 11.1 | 0.484 | 0.946 |
| MobRecon[11] | - | 9.2 | - | 9.4 | 0.538 | 0.957 |
| Keypoint Trans [26] | 0.786 | 10.8 | - | - | - | - |
| AMVUR [32] | 0.835 | 8.3 | 0.836 | 8.2 | 0.608 | 0.965 |
| HaMeR | 0.846 | 7.7 | 0.841 | 7.9 | 0.635 | 0.980 |
| Proposed | 0.851 | 7.5 | 0.846 | 7.7 | 0.646 | 0.983 |

| Method | PA-MPJPE ↓\downarrow | PA-MPVPE ↓\downarrow | F@5 ↑\uparrow | F@15 ↑\uparrow |
|---|---|---|---|---|
| Proposed w. FastViT | 6.5 | 6.3 | 0.741 | 0.967 |
| Proposed w/o ViTPose | 5.9 | 5.7 | 0.795 | 0.989 |
| Proposed w. Single-Scale | 6.0 | 5.9 | 0.793 | 0.991 |
| Proposed w/o Refinement | 6.1 | 5.8 | 0.795 | 0.991 |
| Proposed w. FreiHAND [107] | 6.1 | 5.8 | 0.793 | 0.990 |
| Proposed w. Datasets [63] | 5.9 | 5.7 | 0.805 | 0.992 |
| Proposed Full | 5.5 | 5.1 | 0.825 | 0.993 |

Ablation. To further investigate the contributions of each component of the proposed method we conducted an ablation study. In Tab. 5, we assess the contribution of the backbone architecture, the training datasets used along with the refinement module. As can be observed, swapping the ViT backbone with the recent efficient FastViT [87] architecture (Proposed w. FastViT) results in significant degradation of performance despite the runtime efficiency. Similarly, training the backbone from scratch without using the pre-trained weights of ViTPose [94] (Proposed w/o ViTPose) also results in a performance drop. To evaluate the effect of the proposed refinement module we trained a model that directly regresses the MANO and camera parameters from the ViT output tokens without using any refinement module (Proposed w/o Refinement). Additionally, we trained a model with a single-scale refinement module that samples features from a single feature map (Proposed w. Single-Scale). Both architectural choices deteriorate the reconstruction performance of the proposed model which highlights the effect of the proposed multi-scale refinement module. Finally, we examine the effect of the large-scale training set by training two derivatives of the proposed model using only the FreiHAND dataset [107] (Proposed w. FreiHAND [107]), similar to [43, 44, 103, 11] and a model trained on the datasets used in [63] (Proposed w. Datasets [63]).

### 5.3 Evaluation of Dynamic Reconstruction

A key challenge for 3D pose estimation methods is to achieve stable and robust 4D reconstructions without being trained using a dynamic setting [77]. Traditionally, methods for 3D pose estimation from single image suffer from low temporal coherence and jittering effects across frames, setting a huge burden on their generalization to real-world video reconstruction. To effectively evaluate the temporal coherence of the proposed method, we reconstructed frame-wise a 4D sequence and measure the jittering between frames. In particular, we calculate the mean per frame Euclidean distance of the 3D vertices (MPFVE) and joints (MPFJE) between consecutive frames. Additionally, similar to [77], we measure the jerk (Jitter) of the 3D hand joints motion along with the global Root Translation Error (RTE) that measures the displacement of the wrist across frames. In Tab. 6, we report the reconstruction results for the best performing methods on HO3D [25] dataset. WiLoR outperforms baseline methods in temporal coherence without relying on any temporal module based on the robust stability of the detections. We refer the reader to the supplementary material for qualitative video results.

| Method | MPFVE (×100)↓(\times 100)\downarrow | MPFJE (×100)↓(\times 100)\downarrow | Jitter ↓\downarrow | RTE ↓\downarrow |
|---|---|---|---|---|
| MeshGraphormer [44] | 21.86 | 4.99 | 41.16 | 7.92 |
| MobRecon [11] | 22.18 | 6.09 | 40.25 | 8.03 |
| SimpleHand [103] | 19.72 | 5.12 | 38.53 | 6.04 |
| HaMeR [63] | 10.60 | 1.768 | 20.43 | 2.92 |
| Proposed | 4.43 | 0.762 | 5.92 | 0.07 |

## 6 Conclusion

In this work, we present WiLoR, the first full-stack hand detection and 3D pose estimation framework. Using a large-scale in-the-wild dataset we train a light-weight yet highly accurate hand detector model that can robustly detect hands under different occlusions and illuminations at over 130 FPS. Additionally, we propose a high fidelity 3D hand pose estimation model built on top of our novel refinement module, that overcomes the limitations of previous methods and mitigates the alignment issues of previous methods. Under a series of experiments, we showcase that WiLoR outperforms previous state-of-the-art methods on two benchmark datasets and show robust performance on challenging cases. WiLoR establishes a comprehensive solution for multi-hand detection, localization and 3D reconstruction.

Acknowledgements S. Zafeiriou was supported by Turing AI Fellowship (EP/Z534699/1) and GNOMON (EP/X011364). R.A. Potamias was supported by EPSRC Project GNOMON (EP/X011364).

Supplementary Material

## 7 Implementation Details

In this section we report the training details of the hand detection and the hand pose estimation models.

### 7.1 Hand Detection and Localization

To train the detector model we use WHIM dataset that comprises of over 2M in the wild images from daily activities. We train WiLoR detector with Adam optimizer for 200 epochs with early stopping if there is no loss decrease for over 30 epochs. Initiate the training with a learning rate of 0.01 and linearly decrease to 1​e−61e-6 for the last 30 epochs of the training. We trained the model for three weeks using two NVIDIA RTX 4090 and a batch size of 256. To weight the different losses we set λ0=0.5\lambda_{0}=0.5 for the classification loss, λ1=1.5\lambda_{1}=1.5 for the distribution focal length loss, λ2=15\lambda_{2}=15 for the bounding box loss, λ3=10\lambda_{3}=10 for the keypoints loss. We use random mosaic augmentations with probability 0.7, random rotations between [−60o,60o][-60^{o},60^{o}] and random image scaling between [0.5, 1].

### 7.2 Hand Pose Estimation

We build our hand pose estimation method on top of a ViT-Large backbone with pre-trained weights from ViTPose [94], with a hidden dimension of 1280. Apart from the image patches, we use three additional learnable tokens that correspond to hand pose, shape and camera translation and scale. We initialize the tokens with the mean pose, shape and camera parameters from the training set. Using a set of fully-connected layers, we map the output tokens to 96 MANO pose parameters (15 joint rotations + 1 global orientation represented in 6d rotation format [102]), 10 MANO shape parameters and a 3D camera translation. We then reshape the output image tokens to a 16×1216\times 12 image form, and perform two sets of upsamplings using deconvolutions. At each upsampling step we reduce the feature by 2 times. Using the initially estimated camera parameters we project the rough MANO estimation to the feature maps and sample a set of multi-scale per-vertex image-aligned features. The concatenated set of features is then aggregated and regressed from a set of fully-connected layers that predict the pose, shape and camera residuals. We train the model for 1000 epochs using Adam optimizer with an initial learning rate of 1e-5 and a weight decay of 1e-4. Similar to the hand detector, we apply random scaling, rotations and color jitter during training. Similar to [63], to balance the losses we set λ3​D=0.05\lambda_{3D}=0.05, λ2​D=0.01\lambda_{2D}=0.01, λp​o​s​e=0.001\lambda_{pose}=0.001, λs​h​a​p​e=0.0005\lambda_{shape}=0.0005 and λa​d​v=0.0005\lambda_{adv}=0.0005.

## 8 Comparison with existing datasets

In contrast to 3D hand pose estimation methods that utilizes images of tightly cropped hands, to train a powerful hand detector network, it is required to create a dataset that contains images with multiple hands under different occlusions, views, illuminations and skin tones. Bellow we compare WHIM with such available datasets. WHIM is 100×\times larger than previous in-the-wild multi-hand datasets.

| Dataset | #Img | Annotations | Egocentric | Third-Person | Objects | Real | 3D |
|---|---|---|---|---|---|---|---|
| OxfordHands | 13K | Manual | ✘ | ✔ | ✘ | ✔ | ✘ |
| MPI-HP | 25K | Manual | ✘ | ✔ | ✘ | ✔ | ✘ |
| Coco-Whole | 200K | Manual | ✘ | ✔ | ✘ | ✔ | ✘ |
| BEDLAM | 380K | GT | ✘ | ✔ | ✘ | ✘ | ✔ |
| AGORA | 18K | GT | ✘ | ✔ | ✘ | ✘ | ✔ |
| ContactHands | 21K | Manual | ✘ | ✔ | ✔ | ✔ | ✘ |
| CocoHands | 25K | Manual | ✘ | ✔ | ✔ | ✔ | ✘ |
| BodyHands | 20K | Manual | ✘ | ✔ | ✘ | ✔ | ✘ |
| WHIM | 2M | Auto | ✔ | ✔ | ✔ | ✔ | ✔ |

## 9 Limitations

Although achieving state-of-the-art performance on both 3D hand pose estimation and hand detection tasks, WiLoR still fails to recover challenging cases. Despite being trained on a large-scale dataset, the data distribution is still limited to ‘common’ hand poses and appearances, failing to generalize to samples far from the trained distribution. As can be seen in Fig. 7 WiLoR can fail under extreme finger poses and can also fail to detect hands in crowded environments. Creating a synthetic dataset with diverse hand poses and photorealistic hands could help mitigate these issues [67]. Additionally, since WiLoR employs a bottom-up reconstruction strategy, interactions and contacts between hands may not be adequately captured in 3D space. In scenarios where accurate hand contact estimation is crucial [2, 108], incorporating additional interaction constraints [93] may be necessary. Finally, WiLoR estimates 3D hand poses in camera space, which may lead to inaccurate assumptions about the overall 3D scene. Adapting WiLoR with a 3D metric foundational model [96, 99] could enable more accurate 3D reconstruction in world space.

## 10 Training Datasets

To train our hand pose estimation module we use a combination of datasets to enforce the generalization of the model. In particular, we use 14 datasets with both 2D and 3D annotations, from three major categories: controlled environment hand images, hand-object interaction, in-the-wild and synthetic datasets, resulting in 4.2M images total:

- • FreiHAND [107] is a common 3D hand pose estimation dataset composed of 132K images of indoor, outdoor, and synthetic scenes. It provides both 3D hand and 2D keypoint annotations.
FreiHAND [107] is a common 3D hand pose estimation dataset composed of 132K images of indoor, outdoor, and synthetic scenes. It provides both 3D hand and 2D keypoint annotations.

- • MTC [92] is a subset of Panoptic Studio Dataset [34] that contains 360K multi-view images in a studio environment. The dataset provides both 3D hand and 2D keypoint annotations.
MTC [92] is a subset of Panoptic Studio Dataset [34] that contains 360K multi-view images in a studio environment. The dataset provides both 3D hand and 2D keypoint annotations.

- • InterHand2.6M [52] is a large scale environment from light stage environment that contains hand articulations from 27 different subjects and 80 different cameras. The dataset provides both 3D hand and 2D keypoint annotations.
InterHand2.6M [52] is a large scale environment from light stage environment that contains hand articulations from 27 different subjects and 80 different cameras. The dataset provides both 3D hand and 2D keypoint annotations.

To increase the generalization of WiLoR under severe occlusions we include several datasets where hands interact with objects.

- • HO3D [25] provides a hand-object dataset with over than 120K images from multi-view cameras of hands interacting with objects. It is used as one of the main benchmarks for hand and object reconstruction. Images were captured in an lab environment setting. It provides both 3D hand and 2D keypoint annotations.
HO3D [25] provides a hand-object dataset with over than 120K images from multi-view cameras of hands interacting with objects. It is used as one of the main benchmarks for hand and object reconstruction. Images were captured in an lab environment setting. It provides both 3D hand and 2D keypoint annotations.

- • H2O3D [25] contains over 60K images from five multi-view cameras of hands interacting with objects. In contrast to HO3D, each subject is interacting with the object using two hands which increases the occlusions of the hands. It provides both 3D hand and 2D keypoint annotations.
H2O3D [25] contains over 60K images from five multi-view cameras of hands interacting with objects. In contrast to HO3D, each subject is interacting with the object using two hands which increases the occlusions of the hands. It provides both 3D hand and 2D keypoint annotations.

- • DEX YCB [9] similar to HO3D, DEX YCB is a benchmark dataset for hand object reconstruction. It contains over than 500K multi-view images from 10 objects grasping objects. The dataset provides both 3D hand and 2D keypoint annotations.
DEX YCB [9] similar to HO3D, DEX YCB is a benchmark dataset for hand object reconstruction. It contains over than 500K multi-view images from 10 objects grasping objects. The dataset provides both 3D hand and 2D keypoint annotations.

- • ARCTIC [17] is a large scale dataset of bimanual hand-object manipulations containing over than 400K images from both egocentric and third-person views. It contains both single and dual hand manipulations along with accurate 3D and 2D hand annotations.
ARCTIC [17] is a large scale dataset of bimanual hand-object manipulations containing over than 400K images from both egocentric and third-person views. It contains both single and dual hand manipulations along with accurate 3D and 2D hand annotations.

- • Hot3D [3] is a recent egocentric dataset from daily activities that includes a high degree of occlusions and can significantly enhance the performance of WiLoR in egocentric scenarios. It contains both 3D hand and 2D keypoint annotations.
Hot3D [3] is a recent egocentric dataset from daily activities that includes a high degree of occlusions and can significantly enhance the performance of WiLoR in egocentric scenarios. It contains both 3D hand and 2D keypoint annotations.

Additionally, we include four synthetic datasets that provide accurate ground truth 2D and 3D annotations:

- • RHD [106] is amongst the first synthetic datasets of hands rendered under different illumination patterns. The dataset is composed of 62K images with accurate 3D and 2D annotations.
RHD [106] is amongst the first synthetic datasets of hands rendered under different illumination patterns. The dataset is composed of 62K images with accurate 3D and 2D annotations.

- • Re:InterHand [53] is synthetic dataset that extends InterHand2.6M by rendering hands under different illuminations and environments to bridge the gap between studio setup and in-the-wild images. The dataset provides both 3D hand and 2D keypoint annotations.
Re:InterHand [53] is synthetic dataset that extends InterHand2.6M by rendering hands under different illuminations and environments to bridge the gap between studio setup and in-the-wild images. The dataset provides both 3D hand and 2D keypoint annotations.

- • BEDLAM [4] is a large scale full body synthetic dataset, that has proven extremely effective in whole body reconstruction tasks [7]. We use BEDLAM and randomly crop regions around the human hands to augment the training data. We use over 500K image crops. The dataset provides both 3D hand and 2D keypoint annotations.
BEDLAM [4] is a large scale full body synthetic dataset, that has proven extremely effective in whole body reconstruction tasks [7]. We use BEDLAM and randomly crop regions around the human hands to augment the training data. We use over 500K image crops. The dataset provides both 3D hand and 2D keypoint annotations.

Finally, we include in-the-wild datasets that contain only 2D information, but can effectively boost the generalization performance of WiLoR in in the while scenarios.

- • COCO WholeBody [33] is a subset of COCO dataset and one of the main benchmarks for body pose estimation. It contains over than 100K in the wild images with humans participating in different activities. It provides 2D hand keypoint annotations.
COCO WholeBody [33] is a subset of COCO dataset and one of the main benchmarks for body pose estimation. It contains over than 100K in the wild images with humans participating in different activities. It provides 2D hand keypoint annotations.

- • Halpe [18] is a full body in-the-wild dataset, composed of over than 40K images will 2D keypoint annotations. The dataset contains 21 keypoints for each hand.
Halpe [18] is a full body in-the-wild dataset, composed of over than 40K images will 2D keypoint annotations. The dataset contains 21 keypoints for each hand.

- • MPII NZSL [78] is a common benchmark for human body pose estimation, containing a set of in-the-wild, synthetic and lab environment images. The dataset contain 15K images with 2D keypoint hand annotations.
MPII NZSL [78] is a common benchmark for human body pose estimation, containing a set of in-the-wild, synthetic and lab environment images. The dataset contain 15K images with 2D keypoint hand annotations.

## 11 Temporal Coherence

In the project page we provide several videos that demonstrate the temporal coherence of WiLoR in challenging scenarios such as kneading dough or playing guitar. Despite being trained on single images, WiLoR can provide smooth reconstructions given its stable and robust detections.
