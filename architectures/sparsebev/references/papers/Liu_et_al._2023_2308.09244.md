# SparseBEV: High-Performance Sparse 3D Object Detection from Multi-Camera Videos

Camera-based 3D object detection in BEV (Bird’s Eye View) space has drawn great attention over the past few years. Dense detectors typically follow a two-stage pipeline by first constructing a dense BEV feature and then performing object detection in BEV space, which suffers from complex view transformations and high computation cost. On the other side, sparse detectors follow a query-based paradigm without explicit dense BEV feature construction, but achieve worse performance than the dense counterparts. In this paper, we find that the key to mitigate this performance gap is the adaptability of the detector in both BEV and image space. To achieve this goal, we propose SparseBEV, a fully sparse 3D object detector that outperforms the dense counterparts. SparseBEV contains three key designs, which are (1) scale-adaptive self attention to aggregate features with adaptive receptive field in BEV space, (2) adaptive spatio-temporal sampling to generate sampling locations under the guidance of queries, and (3) adaptive mixing to decode the sampled features with dynamic weights from the queries. On the test split of nuScenes, SparseBEV achieves the state-of-the-art performance of 67.5 NDS. On the val split, SparseBEV achieves 55.8 NDS while maintaining a real-time inference speed of 23.5 FPS. Code is available at https://github.com/MCG-NJU/SparseBEV.

## 1 Introduction

Camera-based 3D Object Detection [13, 54, 31, 11, 24, 25, 40] has witnessed great progress over the past few years. Compared with the LiDAR-based counterparts [19, 56, 4, 36], camera-based approaches have lower deployment cost and can detect long-range objects.

Previous methods can be divided into two paradigms. BEV (Bird’s Eye View)-based methods [13, 11, 25, 24, 40] follow a two-stage pipeline by first constructing an explicit dense BEV feature from multi-view features and then performing object detection in BEV space. Although achieving remarkable progress, they suffer from high computation cost and rely on complex view transformation operators. Another line of work [54, 31, 32] explores the sparse query-based paradigm by initializing a set of sparse reference points in 3D space. Specifically, DETR3D [54] links the queries to image features using 3D-to-2D projection. It has simpler structure and faster speed, but its performance still lags far behind the dense ones. PETR series [31, 32] uses dense global attention for the interaction between query and image feature, which is computationally expensive and buries the advantage of the sparse paradigm. Thus, a natural question arises whether fully sparse detectors achieve similar accuracy to the dense ones?

In this paper, we find that the key to obtain high performance in sparse 3D object detection is the adaptability of the detector in both BEV and image space. In BEV space, the detector should be able to aggregate multi-scale features adaptively. Dense BEV-based detectors typically use a BEV encoder to encode multi-scale BEV features. It can be a stack of residual blocks with FPN [26] (e.g. BEVDet [26]), or a transformer encoder with multi-scale deformable attention [60] (e.g. BEVFormer [25]). For sparse detectors such as DETR3D, we argue that the multi-head self attention (MHSA) [49] among queries can play the role of the BEV encoder, as queries are defined in BEV space. However, the vanilla MHSA has a global receptive field, lacking an explicit multi-scale design. In image space, the detector should be adaptive to different objects with different sizes and categories. This is because although the objects have similar sizes in 3D space, they might vary greatly in images. However, the single-point sampling in DETR3D has a fixed local receptive field and the sampled feature is processed by static linear layers, hindering its performance.

To this end, we present SparseBEV, a fully sparse 3D object detector that matches or even outperforms the dense counterparts. Our SparseBEV detector contains three key designs, which are (1) scale-adaptive self attention to aggregate features with adaptive receptive field in BEV space, (2) adaptive spatio-temporal sampling to generate sampling locations under the guidance of queries, and (3) adaptive mixing to decode the sampled features with dynamic weights from the queries. We also propose to use pillars instead of reference points as the formulation of query, since pillars introduce better spatial priors.

We conduct comprehensive experiments on the nuScenes dataset. As shown in Fig. 1, our SparseBEV achieves the performance of 55.8 NDS and the speed of 23.5 FPS on the val split, surpassing all previous methods in both speed and accuracy. Besides, we can flexibly adjust of the inference speed by reducing the number of decoder layers without re-training. On test split, SparseBEV with V2-99 [20] backbone achieves 63.6 NDS without using future frames or test-time augmentation. By further utilizing future frames, SparseBEV achieves 67.5 NDS, outperforming the previous state-of-the-art BEVFormerV2 [55] by 2.7 NDS.

## 2 Related Work

### 2.1 Query-based 2D Object Detection

Recently, Transformer [49] with its attention blocks has been widely applied in the computer vision tasks [5, 8, 2]. In object detection, DETR [2] was the first model to predict objects based on learnable queries and treat the detection as a set prediction problem. A lot of works [60, 45, 30, 37, 7, 21, 57] were then proposed to accelerate the convergence of DETR by using the sampled features instead of using the global ones. For example, Deformable-DETR [60] samples image features based on sparse reference points, and then applies the deformable attention on the features. Sparse R-CNN [45] uses the ROIAlign [9] to obtain local features and then performs the dynamic convolution. AdaMixer [7] combines the advantages of the sampling points and the dynamic convolution to further boost the convergence. DN-DETR [21] devises a denoising mechanism that feeds ground-truth bounding boxes with noises into decoder and trains the model to reconstruct the original boxes. DINO [57] furtherly optimizes DN-DETR [21] and DAB-DETR [30] by proposing a contrastive denoising training strategy and a mixed query selection method. There are also several methods [3, 14, 46] designed to tackle the instability of the training procedure. Our work follows the query-based detection paradigm and extends it to 3D space with temporal information.

### 2.2 Monocular 3D Object Detection

Monocular 3D object detection takes one single image as input and outputs predicted 3D bounding boxes of objects. A significant challenge in monocular 3D object detection is how to transfer 2D features to 3D space. Several works [53, 43, 51, 39] incorporates depth information to deal with this problem. Pseudo LiDAR [53] first estimates the depth of input images and constructs pseudo point clouds. The pseudo point clouds are then sent to a LiDAR-based detection module to predict the 3D boxes of interested objects. CaDDN [43] further proposes a fully differentiable end-to-end network which learns pixel-wise categorical depth distributions to predict appropriate depth intervals in 3D space. Inspired by FCOS [47], FCOS3D [51] projects 3D coordinates onto 2D images and decouples them as 2D attributes (centerness and classification) and 3D attributes (depths, sizes, and orientations).

### 2.3 3D Object Detection in BEV

Bird’s-eye-view (BEV) object detection [38, 42, 13, 54, 31, 58, 25, 55, 40] aims at detecting objects in BEV space given either single-view or multi-view 2D images. Early works [44, 53, 43] typically transformed 2D features to BEV space based on single-view images and conducted monocular 3D object detection. LSS [42] takes six 2D images of different views as input and transforms them into 3D space based on depth estimation. Based on LSS, BEVDet [13] lifts 2D feature to BEV space and uses a BEV encoder with residual blocks and FPN to further encode the BEV features. BEVFormer [25] proposes a spatio-temporal transformer encoder that projects multi-view and multi-timestamp input to BEV representations. To ease the optimization, BEVFormerV2 [55] introduces perspective view supervision and supervises monocular 3D object detection parallel with BEV object detection. SOLOFusion [40] defines the localization potential for depth estimation and fuses short-term, high-resolution and long-term, low-resolution temporal stereo.

Inspired by DETR, another line of works [54, 31, 32] explore the sparse query-based paradigm. DETR3D [54] proposes a top-down framework starting from a learnable sparse set of reference points and refining them iteratively via 3D-to-2D queries. However, such 3D-to-2D projection hinders the receptive field of the query. To handle this, PETR series [31, 32] use global attention for the interaction betweeen queries and image features, and introduce 3D positional embeddings to encode 2D features into 3D representation without explicit projection. Although achieving notable improvements, the expensive dense global attention buries the advantages of the sparse paradigm and makes it difficult to utilize long-term temporal information efficiently. In contrast, we keep the fully sparse design of DETR3D and boost the performance by enhancing the adaptability of the detector.

## 3 SparseBEV

As shown in Fig. 2, SparseBEV is a query-based one-stage detector with LL decoder layers. The input multi-camera videos are processed frame-by-frame using image backbone and FPN. Next, a set of sparse pillar queries are initialized in BEV space and aggregated by scale-adaptive self attention. These queries then interact with the image features via adaptive spatio-temporal sampling and adaptive mixing to make 3D object predictions. We also propose a dual-branch version of SparseBEV to further enhance the temporal modeling.

### 3.1 Query Formulation

We first define a set of learnable queries, where each of them is represented by its translation [x,y,z][x,y,z], dimension [w,l,h][w,l,h], rotation θ\theta and velocity [vx,vy][v_{x},v_{y}]. The queries are initialized to be pillars in BEV space where zz is set to 0 and hh is set to ∼\sim4m. The initial velocity is set to [vx,vy]=𝟎[v_{x},v_{y}]=\mathbf{0}. Other parameters (x,y,w,l,θx,y,w,l,\theta) are drawn from random gaussian distributions. Following Sparse R-CNN[45], we attach a DD-dim query feature to each query box to encode the rich instance characteristics.

### 3.2 Scale-adaptive Self Attention

As mentioned above, dense BEV-based methods typically use a BEV encoder to encode multi-scale BEV features. However, in our method, since we do not explicitly build a BEV feature, how to aggregate multi-scale features in BEV space is a challenge.

In this work, we argue that the self attention can play the role of BEV encoder, since queries are defined in BEV space. The vanilla multi-head self attention has global receptive field and lacks the ability of local multi-scale aggregation. Thus, we propose scale-adaptive self attention (SASA), which learns appropriate receptive fields under the guidance of queries. First, we compute the all-pair distances D∈ℝN×ND\in\mathbb{R}^{N\times N} (NN is the number of queries) between the query centers in BEV space:

|  | Di,j=(xi−xj)2+(yi−yj)2,D_{i,j}=\sqrt{(x_{i}-x_{j})^{2}+(y_{i}-y_{j})^{2}}, |  | (1) |
|---|---|---|---|

where xix_{i} and yiy_{i} denotes the center of the ii-th query. The attention considers not only the similarity between query features, but the distance between them as well. A toy example below shows how it works:

|  | Attn​(Q,K,V)\displaystyle\text{Attn}(Q,K,V) | =Softmax​(Q​KTd−τ​D)​V,\displaystyle=\text{Softmax}(\frac{QK^{T}}{\sqrt{d}}-\tau D)V, |  | (2) |
|---|---|---|---|---|

where Q,K,V∈ℝN×dQ,K,V\in\mathbb{R}^{N\times d} is the query itself and dd is the channel dimension. τ\tau is a scalar to control the receptive field for each query. When τ=0\tau=0, it degrades to the vanilla self attention with global receptive field. As τ\tau grows, the attention weights for distant queries becomes smaller and the receptive field narrows.

In practice, the receptive field controller τ\tau is adaptive to each query and specific to each head. Supposing there are HH heads, we use a linear transformation to generate head-specific τ1,τ2,…,τH\tau_{1},\tau_{2},...,\tau_{H} from the give query 𝐪∈ℝd\mathbf{q}\in\mathbb{R}^{d}:

|  | [τ1,τ2,…,τH]=Lineard→H​(𝐪)∈ℝH,[\tau_{1},\tau_{2},...,\tau_{H}]=\text{Linear}_{d\to H}(\mathbf{q})\in\mathbb{R}^{H}, |  | (3) |
|---|---|---|---|

where the weights are shared across different queries.

In our experiments, we surprisingly find that the τ\tau for each head is learnt to uniformly distribute in a certain range regardless of the initialization. In Fig. 3, we sort the heads according to τ\tau and visualize the attention weights for the distance part. As we can see, each head learns a different receptive field from each other, enabling the network to aggregate features in a multi-scale manner like FPN.

Scale-adaptive self attention (SASA) demonstrates the necessity of FPN, while being more flexible as it learns the scales adaptively from the query. We also find an interesting phenomenon that different categories of queries have different sizes of receptive field. For example, queries representing the bus have larger receptive field than those representing the pedestrians. More details can be found in the ablation studies.

### 3.3 Adaptive Spatio-temporal Sampling

For each frame, we use a linear layer to generate a set of sampling offsets {(Δ​xi,Δ​yi,Δ​zi)}\{(\Delta x_{i},\Delta y_{i},\Delta z_{i})\} adaptively from the query feature. These offsets are transformed to 3D sampling points based on the query pillar:

|  | [xiyizi]=[cos⁡θ−sin⁡θ0sin⁡θcos⁡θ0001]​[w⋅Δ​xil⋅Δ​yih⋅Δ​zi]+[xyz].\displaystyle\left[\!\begin{array}[]{c}x_{i}\\ y_{i}\\ z_{i}\\ \end{array}\!\right]\!=\!\left[\!\!\begin{array}[]{cccc}\cos\theta&-\!\sin\theta&0\\ \sin\theta&\cos\theta&0\\ 0&0&1\\ \end{array}\!\!\right]\!\!\left[\!\!\begin{array}[]{c}w\!\cdot\!\Delta x_{i}\\ l\!\cdot\!\Delta y_{i}\\ h\!\cdot\!\Delta z_{i}\\ \end{array}\!\!\right]\!+\!\left[\!\begin{array}[]{c}x\\ y\\ z\\ \end{array}\!\right]. |  |
|---|---|---|

Compared with the deformable attention in BEVFormer, our sampling points are adaptive to both query pillar and query feature, thus better covering objects with varying sizes. Besides, these points are not restricted to the query, since we do not limit the range of the sampling offsets.

Next, we perform temporal alignment by warping the sampling points according to motions. In autonomous driving, there are two types of motion: one is ego-motion and the other is object motion. Ego-motion describes the motion of the car from its own perspective as it navigates through the environment, while object motion refers to the movement of other objects in the environment as they move around the self-driving car.

As mentioned above, instantaneous velocity can be equal to average velocity for a short time window in self-driving. Thus, we adaptively warp the sampling points to previous timestamps using the velocity vector [vx,vy][v_{x},v_{y}] from the query:

|  | xt,i\displaystyle x_{t,i} | =xi+vx⋅(Tt−T0)\displaystyle=x_{i}+v_{x}\cdot(T_{t}-T_{0}) |  | (16) |
|---|---|---|---|---|
|  | yt,i\displaystyle y_{t,i} | =yi+vy⋅(Tt−T0),\displaystyle=y_{i}+v_{y}\cdot(T_{t}-T_{0}), |  | (17) |

where TtT_{t} denotes the timestamp of previous frame tt (T0T_{0} denotes the current timestamp). Note that zt,iz_{t,i} is identical to ziz_{i} because the velocity vector is defined in BEV plane.

Next, we warp the sampling points based on the ego pose provided by the dataset. Points are first transformed to the global coordinate system and then to the local coordinate system of frame tt:

|  | [xt,i′yt,i′zt,i′​ 1]T=Et​E0−1​[xt,iyt,izt,i​ 1]T,[x_{t,i}^{\prime}\ \ \ y_{t,i}^{\prime}\ \ \ z_{t,i}^{\prime}\ \ \ 1]^{T}=E_{t}E_{0}^{-1}[x_{t,i}\ \ y_{t,i}\ \ z_{t,i}\ \ 1]^{T}, |  | (18) |
|---|---|---|---|

where Et=[𝐑|𝐭]E_{t}=[\mathbf{R}|\mathbf{t}] is the ego pose of frame tt.

| Method | Backbone | Input Size | Epochs | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|---|---|---|---|---|---|---|---|---|---|---|
| PETRv2 [32] | ResNet50 | 704 ×\times 256 | 60 | 45.6 | 34.9 | 0.700 | 0.275 | 0.580 | 0.437 | 0.187 |
| BEVStereo [22] | ResNet50 | 704 ×\times 256 | 90 ‡\ddagger | 50.0 | 37.2 | 0.598 | 0.270 | 0.438 | 0.367 | 0.190 |
| BEVPoolv2 [12] | ResNet50 | 704 ×\times 256 | 90 ‡\ddagger | 52.6 | 40.6 | 0.572 | 0.275 | 0.463 | 0.275 | 0.188 |
| SOLOFusion [40] | ResNet50 | 704 ×\times 256 | 90 ‡\ddagger | 53.4 | 42.7 | 0.567 | 0.274 | 0.511 | 0.252 | 0.181 |
| Sparse4Dv2 [29] | ResNet50 | 704 ×\times 256 | 100 | 53.9 | 43.9 | 0.598 | 0.270 | 0.475 | 0.282 | 0.179 |
| StreamPETR †\dagger [50] | ResNet50 | 704 ×\times 256 | 60 | 55.0 | 45.0 | 0.613 | 0.267 | 0.413 | 0.265 | 0.196 |
| SparseBEV | ResNet50 | 704 ×\times 256 | 36 | 54.5 | 43.2 | 0.606 | 0.274 | 0.387 | 0.251 | 0.186 |
| SparseBEV †\dagger | ResNet50 | 704 ×\times 256 | 36 | 55.8 | 44.8 | 0.581 | 0.271 | 0.373 | 0.247 | 0.190 |
| DETR3D †\dagger [54] | ResNet101-DCN | 1600 ×\times 900 | 24 | 43.4 | 34.9 | 0.716 | 0.268 | 0.379 | 0.842 | 0.200 |
| BEVFormer †\dagger [25] | ResNet101-DCN | 1600 ×\times 900 | 24 | 51.7 | 41.6 | 0.673 | 0.274 | 0.372 | 0.394 | 0.198 |
| BEVDepth [24] | ResNet101 | 1408 ×\times 512 | 90 ‡\ddagger | 53.5 | 41.2 | 0.565 | 0.266 | 0.358 | 0.331 | 0.190 |
| Sparse4D †\dagger [28] | ResNet101-DCN | 1600 ×\times 900 | 48 | 55.0 | 44.4 | 0.603 | 0.276 | 0.360 | 0.309 | 0.178 |
| SOLOFusion [40] | ResNet101 | 1408 ×\times 512 | 90 ‡\ddagger | 58.2 | 48.3 | 0.503 | 0.264 | 0.381 | 0.246 | 0.207 |
| SparseBEV †\dagger | ResNet101 | 1408 ×\times 512 | 24 | 59.2 | 50.1 | 0.562 | 0.265 | 0.321 | 0.243 | 0.195 |

For each timestamp, we project the warped sampling points {(xt,i′,yt,i′,zt,i′)}\{(x_{t,i}^{\prime},y_{t,i}^{\prime},z_{t,i}^{\prime})\} to each view using the provided camera intrinsics and extrinsics. Since there are overlaps between adjacent views, the projected point may hit one or more views, which are termed as 𝒱\mathcal{V}. For each hit view kk, we have multi-scale feature maps {Fk,j|j∈{1,2,…,Nfeat}}\{F_{k,j}|j\in\{1,2,...,N_{\text{feat}}\}\} from the image backbone. Features are first sampled by bilinear interpolation 𝐁\mathbf{B} in the image plane and then weighted over the scale axis:

|  | ft,i=1|𝒱|​∑k∈𝒱∑j=1Nfeatwi​j​𝐁​(Fk,j,𝐏k​(xt,i′,yt,i′,zt,i′))f_{t,i}=\frac{1}{|\mathcal{V}|}\sum_{k\in\mathcal{V}}\sum_{j=1}^{N_{\text{feat}}}w_{ij}\mathbf{B}(F_{k,j},\mathbf{P}_{k}(x_{t,i}^{\prime},y_{t,i}^{\prime},z_{t,i}^{\prime})) |  | (19) |
|---|---|---|---|

where NfeatN_{\text{feat}} is the number of the multi-scale feature maps and 𝐏k\mathbf{P}_{k} is the projection function for view kk. wi​jw_{ij} is the weight for the ii-th point on the jj-th feature map and is generated from the query feature by linear transformation.

### 3.4 Adaptive Mixing

Given the sampled features from different timestamps and locations, the key is how to adaptively decode it under the guidance of queries. Inspired by AdaMixer [7], we introduce a simple but effective approach to decode and aggregate the spatio-temporal features with dynamic convolutions [15] and MLP-Mixer [48]. Supposing there are TT frames in total and SS sampling points per frame, we first stack them to a total of P=T×SP=T\times S points. Thus, the sampled features are organized to f∈ℝP×Cf\in\mathbb{R}^{P\times C}.

We first perform channel mixing on ff to enhance object semantics. The dynamic weights are generated from the query feature 𝐪\mathbf{q}:

|  | Wc\displaystyle W_{c} | =Linear​(𝐪)∈ℝC×C\displaystyle=\text{Linear}(\mathbf{q})\in\mathbb{R}^{C\times C} |  | (20) |
|---|---|---|---|---|
|  | Mc​(f)\displaystyle\text{M}_{c}(f) | =ReLU​(LayerNorm​(f​Wc)),\displaystyle=\text{ReLU}(\text{LayerNorm}(fW_{c})), |  | (21) |

where WcW_{c} is the dynamic weights and is shared across different frames and different sampling points.

Next, we then transpose the feature and apply the dynamic weights to the point dimension of it:

|  | Wp\displaystyle W_{p} | =Linear​(𝐪)∈ℝP×P\displaystyle=\text{Linear}(\mathbf{q})\in\mathbb{R}^{P\times P} |  | (22) |
|---|---|---|---|---|
|  | Mp​(f)\displaystyle\text{M}_{p}(f) | =ReLU​(LayerNorm​(fT​Wp)),\displaystyle=\text{ReLU}(\text{LayerNorm}(f^{T}W_{p})), |  | (23) |

where WpW_{p} is the dynamic weights and is shared across different channels.

After channel mixing and point mixing, the spatio-temporal features are flattened and aggregated by a linear layer. The final regression and classification predictions are computed by two MLPs respectively.

| Method | Backbone | Epochs | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|---|---|---|---|---|---|---|---|---|---|
| DETR3D [54] | V2-99 | 24 | 47.9 | 41.2 | 0.641 | 0.255 | 0.394 | 0.845 | 0.133 |
| PETR [31] | V2-99 | 24 | 50.4 | 44.1 | 0.593 | 0.249 | 0.383 | 0.808 | 0.132 |
| UVTR [23] | V2-99 | 24 | 55.1 | 47.2 | 0.577 | 0.253 | 0.391 | 0.508 | 0.123 |
| BEVFormer [25] | V2-99 | 24 | 56.9 | 48.1 | 0.582 | 0.256 | 0.375 | 0.378 | 0.126 |
| BEVDet4D [11] | Swin-B [33] | 90‡ | 56.9 | 45.1 | 0.511 | 0.241 | 0.386 | 0.301 | 0.121 |
| PolarFormer [16] | V2-99 | 24 | 57.2 | 49.3 | 0.556 | 0.256 | 0.364 | 0.440 | 0.127 |
| PETRv2 [32] | V2-99 | 24 | 59.1 | 50.8 | 0.543 | 0.241 | 0.360 | 0.367 | 0.118 |
| Sparse4D [28] | V2-99 | 48 | 59.5 | 51.1 | 0.533 | 0.263 | 0.369 | 0.317 | 0.124 |
| BEVDepth [24] | V2-99 | 90‡ | 60.0 | 50.3 | 0.445 | 0.245 | 0.378 | 0.320 | 0.126 |
| BEVStereo [22] | V2-99 | 90‡ | 61.0 | 52.5 | 0.431 | 0.246 | 0.358 | 0.357 | 0.138 |
| SOLOFusion [40] | ConvNeXt-B [34] | 90‡ | 61.9 | 54.0 | 0.453 | 0.257 | 0.376 | 0.276 | 0.148 |
| SparseBEV | V2-99 | 24 | 62.7 | 54.3 | 0.502 | 0.244 | 0.324 | 0.251 | 0.126 |
| SparseBEV (dual-branch) | V2-99 | 24 | 63.6 | 55.6 | 0.485 | 0.244 | 0.332 | 0.246 | 0.117 |
| BEVFormerV2 †\dagger [55] | InternImage-XL [52] | 24 | 64.8 | 58.0 | 0.448 | 0.262 | 0.342 | 0.238 | 0.128 |
| BEVDet-Gamma †\dagger [12] | Swin-B | 90‡ | 66.4 | 58.6 | 0.375 | 0.243 | 0.377 | 0.174 | 0.123 |
| SparseBEV †\dagger | V2-99 | 24 | 67.5 | 60.3 | 0.425 | 0.239 | 0.311 | 0.172 | 0.116 |

### 3.5 Dual-branch SparseBEV

Inspired by SlowFast [6], we further enhance the long-term temporal modeling by dividing the input video into a slow stream and a fast stream, resulting in a dual-branch SparseBEV. The slow stream is designed to capture fine-grained appearance details and it operates at low frame rates and high resolutions. The fast stream is responsible for capturing long-term temporal stereo and it operates at high frame rates and low resolutions. Sampling points are projected to the two streams respectively and the sampled features are stacked before adaptive mixing.

In this way, we decouple the learning of static appearance and temporal motion, leading to better performance. Besides, the computation cost is significantly reduced since only a small fraction of frames needs to be processed at high resolution. However, since this dual-branch design makes the framework a little complex, we do not use it unless otherwise stated. The ablations of this part can be found in the supplementary material.

## 4 Experiments

### 4.1 Implementation Details

We implement our model using PyTorch [41]. Following previous methods [54, 25, 32, 40], we adopt common image backbones including ResNet [10] and V2-99 [20]. The decoder consists of 6 layers and weights are shared across different layers. By default, we use T=8T=8 frames in total and the interval between adjacent frames is about 0.5s.

During training, we adopt the Hungarian algorithm [18] for label assignment between ground-truth objects and predictions. Focal loss [27] is used for classification and L1 loss is used for 3D bounding box regression. We use the AdamW [35] optimizer with a global batch size of 8. The initial learning rate is set to 2×10−42\times 10^{-4} and is decayed with cosine annealing policy.

Recently, we follow the training setting of the very recent work StreamPETR [50] and refresh our results for fair comparison. For the regression loss, we change the loss weight of xx and yy to 2.0 and leave the others to 1.0. Query denoising [21] is also introduced to stablize training and speedup convergence.

### 4.2 Datasets and Metrics

We evaluate our model on the nuScenes dataset [1], which consists of large-scale multimodal data collected from 6 surround-view cameras, 1 lidar and 5 radars. The dataset has 1000 videos and is split into 700/150/150 videos for training/validation/testing. Each video has roughly 20s duration and the key samples are annotated every 0.5s.

For 3D object detection, there are up to 1.4M annotated 3D bounding boxes of 10 classes. The official evaluation metrics include the well-known mean Average Precision (mAP) and five true positive (TP) metrics, including ATE, ASE, AOE, AVE, and AAE for measuring translation, scale, orientation, velocity, and attribute errors respectively. The overall performance is measured by the nuScenes Detection Score (NDS), which is the composite of the above metrics.

### 4.3 Comparison with the State-of-the-art Methods

In Tab. 1, we compare SparseBEV with previous state-of-the-art methods on the validation split of nuScenes. Unless otherwise stated, the image backbone is pretrained on ImageNet-1k [17] and the number of queries is set to 900. To keep the simplicity of our approach, the dual-branch design is not used here. When adopting ResNet50 as the backbone and 704 ×\times 256 as the input size, SparseBEV outperforms the previous state-of-the-art method SOLOFusion by 1.1 NDS and 0.5 mAP. By further adopting nuImages [1] pretraining and reducing the number queries to 400, SparseBEV sets a new record of 55.8 NDS while maintaining a real-time inference speed of 23.5 FPS (corresponding to Fig. 1). Next, we upgrade the backbone to ResNet101 and scale the input size up to 1408 ×\times 512. Under this setting, SparseBEV still surpasses SOLOFusion by 1.8 mAP and 1.0 NDS, demonstrating the scalability of our method.

We submit our method to the website of nuScenes and report the leaderboard in Tab. 2. Using the V2-99 [20] backbone pretrained by DD3D [39], SparseBEV achieves 62.7 NDS and 54.3 mAP without bells and whistles. Built on top of this, the dual-branch design gives us extra bonus on the performance, hitting 63.6 NDS and 55.6 mAP. We also follow the offline setting of BEVFormerV2 [55] which leverages both past and future frames to assist the detection. Remarkably, our method (single branch only) surpasses BEVFormerV2 by 2.8 mAP and 2.2 NDS. In addition, BEVFormerV2 uses InternImage-XL [52] with over 300M parameters as the image backbone, while our V2-99 backbone is much more lightweight with only ∼\sim70M parameters.

### 4.4 Ablation Studies

In this section, we conduct ablations on the validation split of nuScenes. For all experiments, we use ResNet50 pretrained on nuImages [1] as the image backbone. The input contains 8 frames with 704 ×\times 256 resolution. We use 900 queries in the decoder and the model is trained for 24 epochs without CBGS [59].

| Query Formulation | NDS | mAP |
|---|---|---|
| 3D Reference Points | 55.1 | 44.0 |
| BEV Pillars | 55.6 | 45.4 |

In Tab. 3, we ablate different formulations of the query. The top row uses a set of reference points distributing uniformly in 3D space (e.g. DETR3D and PETR). By replacing the reference points with pillars in BEV space, we observe performance improvements (+0.5 NDS) over the baseline. This is because pillars introduce better spatial priors than reference points.

| Self Attention | Distance Function | NDS | mAP |
|---|---|---|---|
| MHSA | - | 53.4 | 41.4 |
| SASA | τ​D2\tau D^{2} | 54.3 | 43.8 |
| τ​D\tau D | 55.6 | 45.4 |  |
| τ​D\tau\sqrt{D} | 55.1 | 44.3 |  |

We study the effect of scale-adaptive self attention (SASA) in Tab. 4. Compared with the vanilla multi-head self attention, SASA achieves +4.0 mAP and +2.2 NDS improvements over the baseline. We further ablate different distance functions and find that L2L_{2} distance works best. Besides, we also observe two interesting phenomena in SASA. First, different heads learn a different receptive field (see Fig. 3), enabling the model to aggregate multi-scale features in BEV space. Second, queries representing larger objects tend to have larger receptive field. In Fig. 4, we average the receptive field coefficient τ\tau over all queries and all heads for each class. As we can see, the receptive field of large objects (such as bus and truck) is larger than small objects (such as pedestrian and traffic cone), demonstrating the effectiveness of our adaptive design.

In Fig. 5, we ablate the number of frames and sampling points. The performance continues to improve as the number of frames increases, proving that SparseBEV can benefit from the long-term history. Here, we use 8 frames for fair comparison with previous methods. As for the number of sampling points, we observe that dispatching 16 points for each frame leads to the best performance. We also provide the visualization of the sampling points from the last stage of the decoder in Fig. 6. Our sampling scheme has adaptive regions of interest and achieves good temporal alignments for both static and moving objects across different frames.

| Ego Align | Obj. Align | NDS | mAP | mAVE |
|---|---|---|---|---|
| - | - | 44.4 | 34.1 | 0.510 |
| √\surd | - | 54.2 | 43.5 | 0.281 |
| √\surd | √\surd | 55.6 | 45.4 | 0.243 |

In Tab. 5, we validate the necessity of temporal alignment in spatio-temporal sampling. In our method, ego motion is aligned with the provided ego pose, while object motion is aligned with a simple constant velocity motion model. As we can see from the table, both of them contribute to the performance.

In Tab. 6, we validate the design of the adaptive mixing. The top row is a baseline that uses attention weights to aggregate sampled features (as done in DETR3D). Static mixing and adaptive mixing improve the baseline by 2.7 NDS and 6.5 NDS respectively, demonstrating the necessity of the adaptive design. Next, we explore different combinations of channel and point mixing. Channel mixing followed by point mixing leads to better performance, proving that the object semantics should be enhanced before point mixing.

### 4.5 Limitations and Future Work

One limitation of SparseBEV is the heavy reliance on ego pose. As we can see from Tab. 5, the performance drops about 10 NDS without ego-based temporal alignment. However, in the real world, the ego pose provided by IMU may be unreliable and inaccurate, seriously affecting the robustness. Another limitation is that the inference latency increases linearly with the number of frames, since we stack the sampled features along the temporal dimension.

In the future, we will explore a more elegant and concise way of decoupling spatial appearance and temporal motion. We will also try to extend SparseBEV to other 3D perception tasks such as BEV segmentation, occupancy prediction and lane detection.

| Method | Details | NDS | mAP |
|---|---|---|---|
| W/o Mixing | - | 49.1 | 38.6 |
| Static Mixing | Channel →\rightarrow Point | 51.8 | 41.9 |
| Adaptive Mixing | Channel Only | 53.1 | 42.7 |
| Point Only | 53.6 | 43.3 |  |
| Point →\rightarrow Channel | 55.4 | 44.6 |  |
| Channel →\rightarrow Point | 55.6 | 45.4 |  |

## 5 Conclusion

In this paper, we have proposed a query-based one-stage 3D object detector, named SparseBEV, which can enjoy the benefits of the BEV space without explicitly constructing a dense BEV feature. SparseBEV improves the adaptability of the decoder by three key modules: scale-adaptive self attention, adaptive spatio-temporal sampling and adaptive mixing. We further introduce a dual-branch design to enhance the long-term temporal modeling. Experiments show that SparseBEV achieves the state-of-the-art performance on the dataset of nuScenes for both speed and accuracy. We hope this exciting result will attract more attention to the fully sparse BEV detection paradigm.

This work is supported by National Key R&\&D Program of China (No. 2022ZD0160900), National Natural Science Foundation of China (No. 62076119, No. 61921006), Fundamental Research Funds for the Central Universities (No. 020214380091, No. 020214380099), and Collaborative Innovation Center of Novel Software Technology and Industrialization. Besides, the authors would like to thank Ziteng Gao, Zhiqi Li and Ruopeng Gao for their help.

## Appendix A Details of Dual-branch SparseBEV

In this section, we provide detailed explanations and ablations on the dual branch design. As shown in Fig. 7, the input multi-camera videos are divided into a high-resolution “slow” stream and a low-resolution “fast” stream. Sampling points are projected to the two streams respectively and the sampled features are stacked before adaptive mixing. Experiments are conducted with a V2-99 backbone pretrained by FCOS3D [51] on the training split of nuScenes. 11 1 Note that the experiment setting used here is different from that in the main paper, since the experiments are conducted before the submission of ICCV 2023. After submission, we further improve our implementation to refesh our results. The conclusion is consistent between these different implmentations.

In Fig. 8, we compare our dual branch design with single branch baselines. If we use a single branch of 1600 ×\times 640 (orange curve) resolution, adding more frames does not provide as much benefit as it does at 800 ×\times 320 resolution (green curve). By using dual branch of 1600 ×\times 640 and 640 ×\times 256 resolution with 1:2 ratio, we decouple spatial appearance and temporal motion, unlocking better performance. As we can see from the blue curve, the longer the frame sequence, the more gain the dual branch design brings.

In Tab. 7, we provide detailed quanlitative results. Under the setting of 8 frames (∼\sim 4 seconds), our dual branch design with only two high resolution (HR) frames surpasses the baseline with eight HR frames. By increasing the number of HR frames to 4, we further improve the performance by 0.8 mAP and 0.5 NDS. Moreover, increasing the resolution of the LR frames does not bring any improvement, which clearly demonstrates that appearance detail and temporal motion are decoupled to different branches.

| Method | Setting | mAP | NDS |
|---|---|---|---|
| Single branch | 8f ×\times 1600 | 48.9 | 57.3 |
| Dual branch | 2f ×\times 1600 + 8f ×\times 640 | 49.4 | 57.9 |
| Dual branch | 4f ×\times 1600 + 8f ×\times 640 | 50.2 | 58.4 |
| Dual branch | 4f ×\times 1600 + 8f ×\times 800 | 50.1 | 58.0 |

Since the dual-branch design also enlarges the receptive field (smaller resolution provides larger receptive field) which may improve performance, we further analyse where the improvement comes from in Tab. 8. The first row is our baseline which takes 8 frames with a single branch of 1600×6401600\times 640 as input. We first try to increase the receptive field by adding an extra C6C_{6} feature map (Row 2), and observe that the performance is slightly improved. This demonstrates that a larger receptive field is required for high-resolution and long-term inputs. However, the spatial appearance and temporal motion is still coupling, limiting the performance. By using dual branches of 1600 ×\times 640 and 640 ×\times 256 with 1:2 ratio (Row 3), we decouple spatial appearance and temporal motion, leading to better performance. Moreover, the training cost is also reduced by 1/3. This experiment demonstrates that we not only need larger receptive fields, but also decouple spatial appearance and temporal motion.

| Method | Feature Maps | Train. Cost | mAP |
|---|---|---|---|
| Single branch | C2,C3,C4,C5C_{2},C_{3},C_{4},C_{5} | 2d 17h | 48.9 |
| Single branch | C2,C3,C4,C5,C6C_{2},C_{3},C_{4},C_{5},C_{6} | 2d 18h | 49.3 |
| Dual branch | C2,C3,C4,C5C_{2},C_{3},C_{4},C_{5} | 1d 19h | 50.2 |

## Appendix B Study on Scale-adaptive Self Attention

| Self Attention | Distance Function | NDS | mAP |
|---|---|---|---|
| SASA-beta | τ​D\tau D | 55.2 | 44.8 |
| SASA | τ​D\tau D | 55.6 | 45.4 |

In this section, we’ll talk about how we came up with scale-adaptive self attention (SASA). In the main paper, the receptive field coefficient τ\tau is specific to each head and adaptive to each query. In the development of SASA, there is an intermediate version (dubbed SASA-beta for convenience): the τ\tau for each head is simply a learnable parameter shared by all queries.

In Fig. 9, we take a closer look at how τ\tau changes with training. We surprisingly find that regardless of the initialization, each head learns a different τ\tau from the others and all of them are distributed in range [0,2][0,2], enabling the network to aggregate local and multi-scale features from multiple heads.

Next, we improve SASA-beta by generating the τ\tau adaptively from the query, which corresponds to the version in the main paper. Compared with SASA-beta, SASA not only has the ability of multi-scale feature aggregation, but generates adaptive receptive field for each query as well. The quanlitative comparison between SASA-beta and SASA is shown in Tab. 9.

## Appendix C More Visualizations

In Fig. 10, we provide more visualizations of the sampling points from different stages. In the initial stage, the sampling points have the shape of pillars. In later stages, they are refined to cover objects with different sizes.
