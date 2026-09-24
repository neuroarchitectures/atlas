# Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection

In this paper, we propose a long-sequence modeling framework, named StreamPETR, for multi-view 3D object detection. Built upon the sparse query design in the PETR series, we systematically develop an object-centric temporal mechanism. The model is performed in an online manner and the long-term historical information is propagated through object queries frame by frame. Besides, we introduce a motion-aware layer normalization to model the movement of the objects. StreamPETR achieves significant performance improvements only with negligible computation cost, compared to the single-frame baseline. On the standard nuScenes benchmark, it is the first online multi-view method that achieves comparable performance (67.6% NDS & 65.3% AMOTA) with lidar-based methods. The lightweight version realizes 45.0% mAP and 31.7 FPS, outperforming the state-of-the-art method (SOLOFusion) by 2.3% mAP and 1.8×\times faster FPS. Code has been available at https://github.com/exiawsh/StreamPETR.git.

## 1 Introduction

Camera-only 3D detection is crucial for autonomous driving because of the low deployment cost and ease of detecting road elements. Recently, multi-view object detection has made remarkable progress by leveraging temporal information [27, 16, 31, 25, 39, 29]. The historical features facilitate the detection of occlusion objects and greatly improve the performance. According to the differences between temporal representations, existing methods can be roughly divided into BEV temporal and perspective temporal methods.

BEV temporal methods [27, 16, 25, 39] explicitly warp BEV features from historical to current frame, as illustrated in Fig. 1 (a), where BEV features serve as an efficient intermediate representation for temporal modeling. However, the highly structured BEV features limit the modeling of moving objects. This paradigm requires a large receptive field to alleviate this problem [16, 39, 27].

Different from these approaches, perspective temporal methods [31, 29] are mainly based on DETR [4, 60]. The sparse query design facilitates the modeling of moving objects [29]. However, the sparse object queries need to interact with multi-frame image features for long-term temporal dependence (see Fig. 1 (b)), leading to multiple computations. Thus, existing works are either stuck in solving the moving objects or introducing multiple computation costs.

Based on the above analysis, we suppose it is possible to employ sparse queries as the hidden states of temporal propagation. In this way, we can utilize object queries to model moving objects while keeping high efficiency. Therefore, we introduce a new paradigm: object-centric temporal modeling and design an efficient framework, termed StreamPETR, as shown in Fig. 1 (c). StreamPETR directly performs frame-by-frame 3D predictions on streaming video. It is effective for motion modeling and is able to build long-term spatial-temporal interaction.

Specifically, a memory queue is first built to store the historical object queries. Then a propagation transformer conducts long-range temporal and spatial interaction with current object queries. The updated object queries are used to generate 3D bounding boxes and pushed into the memory queue. Besides, a motion-aware layer normalization (MLN) is introduced to implicitly encode the motion of the ego vehicle and surrounding objects at different time stamps.

Compared with existing temporal methods, the proposed object-centric temporal modeling brings several advantages. StreamPETR only processes a small number of object queries instead of dense feature maps at each time stamp, consuming negligible computational burden (as shown in Fig. 2). For moving objects, MLN alleviates the cumulative error in video streaming. Except for the location prior used in previous methods, StreamPETR additionally considers the semantic similarity by global attention, which facilitates the detection in motion scenes. To summarize, our contributions are:

- • We pull out the key of streaming multi-view 3D detection and systematically design an object-centric temporal modeling paradigm. The long-term historical information is propagated through object queries frame by frame.
We pull out the key of streaming multi-view 3D detection and systematically design an object-centric temporal modeling paradigm. The long-term historical information is propagated through object queries frame by frame.

- • We develop an object-centric temporal modeling framework, termed StreamPETR. It models moving objects and long-term spatial-temporal interaction simultaneously, consuming negligible storage and computation costs.
We develop an object-centric temporal modeling framework, termed StreamPETR. It models moving objects and long-term spatial-temporal interaction simultaneously, consuming negligible storage and computation costs.

- • On the standard nuScenes dataset, StreamPETR outperforms all online camera-only algorithms. Extensive experiment shows that it can be well generalized to other sparse query-based methods, e.g. DETR3D [47].
On the standard nuScenes dataset, StreamPETR outperforms all online camera-only algorithms. Extensive experiment shows that it can be well generalized to other sparse query-based methods, e.g. DETR3D [47].

## 2 Related Work

### 2.1 Multi-view 3D Object Detection

Multi-view 3D detection is an important task in autonomous driving, which needs to continuously process multi-camera images and predict 3D bounding boxes over time. Pioneer’s works [47, 30, 18, 27, 19, 48] focus on the efficient transformation from multiple perspective views to a unified 3D space in a single frame. The transformation can be divided into BEV-based methods [18, 27, 49, 15, 25, 19] and sparse query based methods [47, 30, 29, 5, 48]. To alleviate the occlusion problem and ease the difficulty of speed prediction, recent works additionally introduce temporal information to extend these two paradigms.

It is relatively intuitive to extend the single-frame BEV methods for temporal modeling. BEVFormer [27] first introduces sequential temporal modeling into multi-view 3D object detection and applies temporal self-attention. BEVDet series [16, 25, 23] use concatenate operation to fuse the adjacent BEV features and achieve remarkable results. Furthermore, SOLOFusion [39] extends BEVStereo [23] to long-term memory and reaches a promising performance. Without an intermediate feature representation, the temporal modeling of query-based methods is more challenging. PETRv2 [31] performs the global cross-attention, while DETR4D [34] and Sparse4D [29] apply sparse attention to model the interaction between multi-frames, which introduce multiple computations. However, the sparse query design is convenient to model the moving objects [29]. In order to combine the advantages of the two paradigms, we utilize sparse object queries as the intermediate representation, which can model moving objects and efficiently propagate long-term temporal information.

### 2.2 Query Propagation

Since DETR [4] is proposed in 2D object detection, the object query has been applied in many downstream tasks [54, 35, 57, 56, 12] to model the temporal interaction. For video object detection, LWDN [20] adopts a brain-inspired memory mechanism to propagate and update the memory feature. QueryProp [12] performs query interaction to reduce the computational cost on non-key frames. It achieves significant improvements and maintains high efficiency. 3D-MAN [52] has a similar idea and extends a single-frame Lidar detector to multi-frames, which effectively combines the features coming from different perspectives of a scene. In object tracking, MOTR [54] and TrackFormer [35] propose the track query to model the object association across frames. MeMOT [2] employs a memory bank to build long temporal dependence, which further boosts performance. MOTRv2 [57] eases the conflict between the detection and association tasks by incorporating an extra detector. MUTR [56] and PF-Track [36] extend MOTR [54] into multi-view 3D object tracking and achieve a promising result.

## 3 Delving into Temporal Modeling

To facilitate our study, we present a generalized formulation for various temporal modeling designs. Given the perspective view features F2​d={F2​d0⋯F2​dt}F_{2d}=\{F^{0}_{2d}\cdots F^{t}_{2d}\}, dense BEV features Fb​e​v={Fb​e​v0⋯Fb​e​vt}F_{bev}=\{F^{0}_{bev}\cdots F^{t}_{bev}\} and sparse object features Fo​b​j={Fo​b​j0⋯Fo​b​jt}F_{obj}=\{F^{0}_{obj}\cdots F^{t}_{obj}\}. The dominant temporal modeling methods can be formulated as:

|  | F~o​u​t=φ⁡(F2​d,Fb​e​v,Fo​b​j)\displaystyle\tilde{F}_{out}\!=\!\varphi(F_{2d},F_{bev},F_{obj}) |  | (1) |
|---|---|---|---|

where φ\varphi is the temporal fusion operation, F~o​u​t\tilde{F}_{out} is the output feature that includes temporal information. We first describe the existing temporal modeling from BEV and perspective view. After that, the proposed object-centric temporal modeling is elaborated.

BEV Temporal Modeling uses the grid-structured BEV features to perform the temporal fusion. To compensate for the ego vehicle motion, the last frame feature Fb​e​vt−1F^{t-1}_{bev} is usually aligned to the current frame.

|  | F~b​e​vt=φ⁡(Fb​e​vt−1,Fb​e​vt)\displaystyle\tilde{F}^{t}_{bev}=\varphi(F^{t-1}_{bev},F^{t}_{bev}) |  | (2) |
|---|---|---|---|

Then a temporal fusion function φ\varphi (concatenation [16, 25] or deformable attention [27]) can be applied for intermediate temporal representation F~b​e​vt\tilde{F}^{t}_{bev}. Extending the above process to long temporal modeling, there are two main routes. The first one is to align the historical kk BEV features and concatenate them with the current frame.

|  | F~b​e​vt=φ⁡(Fb​e​vt−k,⋯,Fb​e​vt−1,Fb​e​vt)\displaystyle\tilde{F}^{t}_{bev}=\varphi(F^{t-k}_{bev},\cdots,F^{t-1}_{bev},F^{t}_{bev}) |  | (3) |
|---|---|---|---|

For another one, the long-term historical information is propagated through the hidden states of BEV features F~b​e​vt−1\tilde{F}^{t-1}_{bev} in a recurrent manner.

|  | F~b​e​vt=φ⁡(F~b​e​vt−1,Fb​e​vt)\displaystyle\tilde{F}^{t}_{bev}=\varphi(\tilde{F}^{t-1}_{bev},F^{t}_{bev}) |  | (4) |
|---|---|---|---|

However, the BEV temporal fusion only considers the static BEV features and ignores the movement of the objects, leading to spatial dislocation.

Perspective Temporal Modeling is mainly performed via interactions between object queries and perspective features. The temporal function φ\varphi is usually achieved by the spatial cross-attention [31, 29, 34]:

|  | F~to​b​j=φ(Ft−k2​d,Fto​b​j)⋯+φ(Ft2​d,Fto​b​j)\displaystyle\tilde{F}^{t}_{obj}=\varphi(F^{t-k}_{2d},{F}^{t}_{obj})\cdots+\varphi(F^{t}_{2d},F^{t}_{obj}) |  | (5) |
|---|---|---|---|

The cross-attention between object query and multi-frame perspective view requires repeated feature aggregation. Simply extending to long-term temporal modeling greatly increases the computation cost.

Object-centric Temporal Modeling is our proposed object-centric solution, which models the temporal interaction by object queries. Through object queries, the motion compensation can be conveniently applied based on estimated states Fo​b​jt−1{F}^{t-1}_{obj}.

|  | F~o​b​jt−1=μ⁡(Fo​b​jt−1,M)\displaystyle\tilde{F}^{t-1}_{obj}=\mu({F}^{t-1}_{obj},M) |  | (6) |
|---|---|---|---|

where μ\mu is an explicit linear velocity model or implicit function to encode motion attributes MM (including the relative time interval △​t\triangle t, estimated velocity vv, and ego-pose matrix EE, which are the same definition in Sec. 4). Further, a global attention φ\varphi is constructed to propagate temporal information through object queries frame by frame:

|  | F~o​b​jt=φ⁡(F~o​b​jt−1,Fo​b​jt)\displaystyle\tilde{F}^{t}_{obj}=\varphi(\tilde{F}^{t-1}_{obj},F^{t}_{obj}) |  | (7) |
|---|---|---|---|

## 4 Method

### 4.1 Overall Architecture

As illustrated in Fig. 3, StreamPETR is built upon end-to-end sparse query-based 3D object detectors [30, 47]. It consists of an image encoder, a recursively updated memory queue, and a propagation transformer [43]. The image encoder is a standard 2D backbone, which is applied to extract semantic features from multi-view images. Then the extracted features, information in the memory queue, and object queries are fed into the propagation transformer to perform the spatial-temporal interaction. The main difference between StreamPETR and single-frame baseline is the memory queue, which recursively updates the temporal information of object queries. Combined with the propagation transformer, the memory queue can propagate temporal priors from previous to current frames efficiently.

### 4.2 Memory Queue

We design a memory queue of N×KN\times K for effective temporal modeling. NN is the number of stored frames and KK is the number of objects stored per frame. According to the experience, we set N=4N=4 and K=256K=256 (ensuring high recall in complex scenarios). After the preset time interval τ\tau, the relative time interval △​t\triangle t, context embedding QcQ_{c}, object center QpQ_{p}, velocity vv, and ego-pose matrix EE of selected object queries are stored in memory queue. Specifically, the above information, corresponding to foreground objects (with top-KK highest classification score), is selected and pushed into the memory queue. The entrance and exit of the memory queue follow the first-in, first-out (FIFO) rule. When information from a new frame is added to the memory queue, the oldest is discarded. Actually, the proposed memory queue is highly flexible and customized, users can freely control the maximal memory size N×KN\times K and saving interval τ\tau during both training and inference.

### 4.3 Propagation Transformer

As illustrated in Fig. 4, the propagation transformer consists of three main components: (1) the motion-aware layer normalization module implicitly updates the object state according to the context embedding and motion information recorded in the memory queue; (2) the hybrid attention replaces the default self-attention operation. It plays the role of temporal modeling and removing duplicated predictions; (3) the cross-attention is adopted for feature aggregation. It can be replaced with an arbitrary spatial operation to build the relationship between image tokens and 3D object queries, such as global attention in PETR [30] or sparse projective attention in DETR3D [47].

Motion-aware Layer Normalization is designed to model the movement of objects. For simplicity, we take the transformation process from the last frame t−1t-1 as the example and adopt the same operation for other previous frames. Given the ego pose matrix from the last frame Et−1E_{t-1} and current frame EtE_{t}, the ego transformation Et−1tE_{t-1}^{t} can be calculated as:

|  | Et−1t=Eti​n​v⋅Et−1E_{t-1}^{t}=\ E_{t}^{inv}\cdot\ E_{t-1} |  | (8) |
|---|---|---|---|

Assume that objects are static, 3D centers Qpt−1Q_{p}^{t-1} in memory queue can be explicitly aligned to the current frame, which is formulated as:

|  | Q~pt=Et−1t⋅Qpt−1\tilde{Q}_{p}^{t}=E_{t-1}^{t}\cdot Q_{p}^{t-1} |  | (9) |
|---|---|---|---|

| Methods | Backbone | Image Size | Frames | mAP↑\uparrow | NDS↑\uparrow | mATE↓\downarrow | mASE↓\downarrow | mAOE↓\downarrow | mAVE↓\downarrow | mAAE↓\downarrow | FPS↑\uparrow |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BEVDet [18] | ResNet50 | 256 ×\times 704 | 1 | 0.298 | 0.379 | 0.725 | 0.279 | 0.589 | 0.860 | 0.245 | 16.7 |
| BEVDet4D [16] | ResNet50 | 256 ×\times 704 | 2 | 0.322 | 0.457 | 0.703 | 0.278 | 0.495 | 0.354 | 0.206 | 16.7 |
| PETRv2 [31] | ResNet50 | 256 ×\times 704 | 2 | 0.349 | 0.456 | 0.700 | 0.275 | 0.580 | 0.437 | 0.187 | 18.9 |
| BEVDepth [25] | ResNet50 | 256 ×\times 704 | 2 | 0.351 | 0.475 | 0.639 | 0.267 | 0.479 | 0.428 | 0.198 | 15.7 |
| BEVStereo [23] | ResNet50 | 256 ×\times 704 | 2 | 0.372 | 0.500 | 0.598 | 0.270 | 0.438 | 0.367 | 0.190 | 12.2 |
| BEVFormerv2 [50] †∗{}^{\dagger}\ast | ResNet50 | - | - | 0.423 | 0.529 | 0.618 | 0.273 | 0.413 | 0.333 | 0.188 | - |
| SOLOFusion [39] | ResNet50 | 256 ×\times 704 | 16+1 | 0.427 | 0.534 | 0.567 | 0.274 | 0.511 | 0.252 | 0.181 | 11.4 |
| BEVPoolv2 [17] | ResNet50 | 256 ×\times 704 | 8+1 | 0.406 | 0.526 | 0.572 | 0.275 | 0.463 | 0.275 | 0.188 | 16.6 |
| StreamPETR | ResNet50 | 256 ×\times 704 | 8 | 0.432 | 0.540 | 0.581 | 0.272 | 0.413 | 0.295 | 0.195 | 27.1 |
| StreamPETR∗‡{}^{\ast}\ddagger | ResNet50 | 256 ×\times 704 | 8 | 0.450 | 0.550 | 0.613 | 0.267 | 0.413 | 0.265 | 0.196 | 31.7 |
| DETR3D [47]∗ | ResNet101-DCN | 900 ×\times 1600 | 1 | 0.349 | 0.434 | 0.716 | 0.268 | 0.379 | 0.842 | 0.200 | 3.7 |
| Focal-PETR [44] | ResNet101-DCN | 512 ×\times 1408 | 1 | 0.390 | 0.461 | 0.678 | 0.263 | 0.395 | 0.804 | 0.202 | 6.6 |
| PETR [30]∗ | ResNet101-DCN | 512 ×\times 1408 | 1 | 0.366 | 0.441 | 0.717 | 0.267 | 0.412 | 0.834 | 0.190 | 5.7 |
| BEVFormer [27]∗ | ResNet101-DCN | 900 ×\times 1600 | 4 | 0.416 | 0.517 | 0.673 | 0.274 | 0.372 | 0.394 | 0.198 | 3.0 |
| PolarDETR [5]-T∗ | ResNet101-DCN | 900 ×\times 1600 | 2 | 0.383 | 0.488 | 0.707 | 0.269 | 0.344 | 0.518 | 0.196 | 3.5 |
| Sparse4D [29]∗ | ResNet101-DCN | 900 ×\times 1600 | 4 | 0.436 | 0.541 | 0.633 | 0.279 | 0.363 | 0.317 | 0.177 | 4.3 |
| BEVDepth | ResNet101 | 512 ×\times 1408 | 2 | 0.412 | 0.535 | 0.565 | 0.266 | 0.358 | 0.331 | 0.190 | - |
| SOLOFusion | ResNet101 | 512 ×\times 1408 | 16+1 | 0.483 | 0.582 | 0.503 | 0.264 | 0.381 | 0.246 | 0.207 | - |
| StreamPETR∗ | ResNet101 | 512 ×\times 1408 | 8 | 0.504 | 0.592 | 0.569 | 0.262 | 0.315 | 0.257 | 0.199 | 6.4 |

where Q~pt\tilde{Q}_{p}^{t} is the aligned centers. Motivated by the task-specific control in generative model [9, 40, 55], we adopt a conditional layer normalization to model the movement of the objects. As shown in Fig. 4, the default affine transformation in layer normalization (LN) is closed. The motion attributes (Et−1t,v,△​tE_{t-1}^{t},v,\triangle t) are flattened and converted to affine vectors γ\gamma and β\beta by two linear layers (ξ1\xi_{1}, ξ2\xi_{2}):

|  | γ\displaystyle\gamma | =ξ1​(Et−1t,v,△​t),\displaystyle=\xi_{1}(E_{t-1}^{t},v,\triangle t), |  | (10) |
|---|---|---|---|---|
|  | β\displaystyle\beta | =ξ2​(Et−1t,v,△​t)\displaystyle=\xi_{2}(E_{t-1}^{t},v,\triangle t) |  |  |

Afterward, the affine transformation is performed to get the motion-aware context embedding Q~ct\tilde{Q}_{c}^{t} and motion-aware position encoding Q~p​et\tilde{Q}_{pe}^{t}.

|  | Q~p​et\displaystyle\tilde{Q}_{pe}^{t} | =γ⋅L​N​(ψ⁡(Q~pt))+β,\displaystyle=\gamma\cdot LN(\psi(\tilde{Q}_{p}^{t}))+\beta, |  | (11) |
|---|---|---|---|---|
|  | Q~ct\displaystyle\tilde{Q}_{c}^{t} | =γ⋅L​N​(Qct)+β\displaystyle=\gamma\cdot LN(Q_{c}^{t})+\beta |  |  |

where ψ\psi is a multi-layer perceptron (MLP) that converted the 3D sampled points Q~pt\tilde{Q}_{p}^{t} into position encoding Q~p​et\tilde{Q}_{pe}^{t}. For the sake of unification, the MLN is also adopted into current object queries. The velocity vv and time interval △​t\triangle t of the current frame are zero-initialized.

Hybrid Attention layer. The self-attention in DETR [4] contributes to duplicated prediction removal. We replace it with hybrid attention, which additionally introduces temporal interaction. As shown in Fig. 4, all stored object queries in the memory queue are concatenated with current queries to obtain the hybrid queries. The hybrid queries are regard as the k​e​ykey and v​a​l​u​evalue in multi-head attention. Since the number of hybrid queries is small (about 2kk, which is far less than image tokens in the cross-attention), the hybrid attention layer brings negligible computation cost.

Following PETR [30], the q​u​e​r​yquery can be defined as a randomly initialized 3D anchor. To fully utilize the spatial and context priors in streaming video, some object queries in the memory queue are directly propagated into the current frame. In our implementation, queries from the last frame are concatenated with randomly initialized queries. For a fair comparison, the number of randomly initialized queries and propagated queries are set to 644 and 256 respectively.

| Methods | Modality | Backbone | Image / Voxel | TTA | mAP↑\uparrow | NDS↑\uparrow | mATE↓\downarrow | mASE↓\downarrow | mAOE↓\downarrow | mAVE↓\downarrow | mAAE↓\downarrow |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CenterPoint [53] | L | - | 0.075×\times0.075×\times0.2 | ✘ | 0.603 | 0.673 | 0.262 | 0.239 | 0.361 | 0.288 | 0.136 |
| FCOS3D [46] | C | R101-DCN | 900 ×\times 1600 | ✔ | 0.358 | 0.428 | 0.690 | 0.249 | 0.452 | 1.434 | 0.124 |
| DETR3D [47] | C | V2-99 | 900 ×\times 1600 | ✔ | 0.412 | 0.479 | 0.641 | 0.255 | 0.394 | 0.845 | 0.133 |
| MV2D [48] | C | V2-99 | 640 ×\times 1600 | ✘ | 0.463 | 0.514 | 0.542 | 0.247 | 0.403 | 0.857 | 0.127 |
| UVTR [24] | C | V2-99 | 900 ×\times 1600 | ✘ | 0.472 | 0.551 | 0.577 | 0.253 | 0.391 | 0.508 | 0.123 |
| BEVFormer [27] | C | V2-99 | 900 ×\times 1600 | ✘ | 0.481 | 0.569 | 0.582 | 0.256 | 0.375 | 0.378 | 0.126 |
| PETRv2 [31] | C | V2-99 | 640 ×\times 1600 | ✘ | 0.490 | 0.582 | 0.561 | 0.243 | 0.361 | 0.343 | 0.120 |
| PolarFormer [19] | C | V2-99 | 900 ×\times 1600 | ✘ | 0.493 | 0.572 | 0.556 | 0.256 | 0.364 | 0.439 | 0.127 |
| BEVStereo [23] | C | V2-99 | 640 ×\times 1600 | ✘ | 0.525 | 0.610 | 0.431 | 0.246 | 0.358 | 0.357 | 0.138 |
| StreamPETR | C | V2-99 | 640 ×\times 1600 | ✘ | 0.550 | 0.636 | 0.479 | 0.239 | 0.317 | 0.241 | 0.119 |
| BEVDet4D [16] | C | Swin-B [32] | 900 ×\times 1600 | ✔ | 0.451 | 0.569 | 0.511 | 0.241 | 0.386 | 0.301 | 0.121 |
| BEVDepth [25] | C | ConvNeXt-B | 640 ×\times 1600 | ✘ | 0.520 | 0.609 | 0.445 | 0.243 | 0.352 | 0.347 | 0.127 |
| AeDet [10] | C | ConvNeXt-B | 640 ×\times 1600 | ✔ | 0.531 | 0.620 | 0.439 | 0.247 | 0.344 | 0.292 | 0.130 |
| PETRv2 | C | RevCol-L [3] | 640 ×\times 1600 | ✘ | 0.512 | 0.592 | 0.547 | 0.242 | 0.360 | 0.367 | 0.126 |
| SOLOFusion [39] | C | ConvNeXt-B | 640 ×\times 1600 | ✘ | 0.540 | 0.619 | 0.453 | 0.257 | 0.376 | 0.276 | 0.148 |
| StreamPETR | C | ViT-L | 800 ×\times 1600 | ✘ | 0.620 | 0.676 | 0.470 | 0.241 | 0.258 | 0.236 | 0.134 |

## 5 Experiments

### 5.1 Dataset and Metrics

We evaluate our approach on the large-scale NuScenes dataset [1] and Waymo Open dataset [42].

The nuScenes Dataset includes 1000 scenes, which are 20 seconds in length and annotated at 2Hz. The camera rig covers the full 360° field of view (FOV). The annotations contain up to 1.4M 3D bounding boxes, and 10 common classes are used for evaluation: car, truck, bus, trailer, construction vehicle, pedestrian, motorcycle, bicycle, barrier, and traffic cone. We compare the methods with the following metrics, the nuScenes Detection Score (NDS), mean Average Precision (mAP), and 5 kinds of True Positive (TP) metrics including average translation error (ATE), average scale error (ASE), average orientation error (AOE), average velocity error (AVE), average attribute error (AAE). Following the standard evaluation metrics, we report the average multi-object tracking accuracy (AMOTA), average multi-object tracking precision (AMOTP), recall (RECALL), multi-object tracking accuracy (MOTA) and ID switch (IDS) for 3D object tracking task.

Waymo Open Dataset collects camera data only spanning a horizontal FOV of 230 degrees. The ground truth bounding boxes are annotated to a maximum range of 75 meters. The longitudinal error tolerant metrics LET-3D-AP, LET-3D-AP-H and LET-3D-APL are used for evaluation. Noting that we only use 20% of training data for fair comparison according to common practice.

| Methods | AMOTA↑\uparrow | AMOTP↓\downarrow | RECALL↑\uparrow | IDS↓\downarrow |
|---|---|---|---|---|
| CenterPoint [53] | 0.638 | 0.555 | 67.5% | 760 |
| SimpleTrack [37] | 0.668 | 0.550 | 70.3% | 575 |
| QD3DT [14] | 0.217 | 1.550 | 37.5% | 6856 |
| MUTR3D [56] | 0.270 | 1.494 | 41.1% | 6018 |
| CC-3DT [11] | 0.410 | 1.274 | 57.8% | 3334 |
| PolarDETR [5] | 0.273 | 1.185 | 40.4% | 2170 |
| UVTR [24] | 0.519 | 1.125 | 59.9% | 2204 |
| QTrack [51] | 0.480 | 1.100 | 59.7% | 1484 |
| Sparse4D [29] | 0.519 | 1.078 | 63.3% | 1090 |
| ByteTrackv2 [58] | 0.564 | 1.005 | 63.5% | 704 |
| PF-Track [36] | 0.434 | 1.252 | 53.8% | 249 |
| StreamPETR | 0.653 | 0.876 | 73.3% | 1037 |

### 5.2 Implementation Details

We conduct experiments with ResNet50 [13], ResNet101, V2-99 [21] and ViT [8] backbones under different pre-training. Following previous methods [27, 30, 39], the performance of ResNet50 and ResNet101 models with pre-trained weights ImageNet [7] and nuImages [1] are provided on the nuScenes val set. To scale up our method, we also report results on the nuScenes test set with V2-99 initialized from DD3D [38] checkpoint and ViT-Large [8]. Following BEVFormerv2 [50], the ViT-Large [8] is pre-trained on Objects365 [41] and COCO [28] dataset.

StreamPETR is trained by AdamW [33] optimizer with a batch size of 16. The base learning rate is set to 4e-4 and the cosine annealing policy is employed. Only key frames are used during both training and inference. All experiments are conducted without CBGS [59] strategy. Our implementation is mainly based on Focal-PETR [44], which introduces auxiliary 2D supervision. The models in the ablation study are trained for 24 epochs, while trained for 60 epochs when compared with others. In particular, we only train 24 epochs for ViT-L [8] to prevent over-fitting. For image and BEV data augmentation, we adopt the same methods as PETR [18, 30]. We randomly skip 1 frame during the training sequence for temporal data augmentation [27].

| Methods | Backbone | mAPL↑\uparrow | mAP↑\uparrow | mAPH↑\uparrow |
|---|---|---|---|---|
| BEVFormer++ [26] | ResNet101-DCN | 0.361 | 0.522 | 0.481 |
| MV-FCOS3D++ [45] | ResNet101-DCN | 0.379 | 0.522 | 0.484 |
| PETR-DN [30] | ResNet101 | 0.358 | 0.502 | 0.462 |
| PETRv2 [31] | ResNet101 | 0.366 | 0.519 | 0.479 |
| StreamPETR∗ | ResNet101 | 0.399 | 0.553 | 0.517 |
| StreamPETR‡ | ResNet101 | 0.395 | 0.551 | 0.518 |

### 5.3 Main Results

NuScenes Dataset. We compare the proposed StreamPETR with previous state-of-the-art vision-based 3D detectors on the nuScenes val and test set. As shown in Tab. 1, StreamPETR shows superior performance on mAP, NDS, mASE, and mAOE metrics when adopting ResNet101 backbone with nuImages pretraining. Compared with the single frame baseline Focal-PETR, StreamPETR has considerable improvements of 11.4% mAP and 13.1% NDS. The mATE of StreamPETR is 10.9% better than Focal-PETR, indicating that our object-centric temporal modeling is able to improve both the accuracy of localization. With image resolutions of 256×\times704 and adopting ResNet50 backbone, StreamPETR exceeds the state-of-the-art method (SOLOFusion) by 0.5 % mAP and 0.6 % NDS. When we reduce the number of queries and apply nuImages pre-training, our method has 2.3 % and 1.6 % advantages in mAP and NDS. At the same time, the inference speed of StreamPETR is 1.8×\times faster.

When we compare the performance on the test set in Tab. 2 and adopt a smaller V2-99 backbone, StreamPETR can surpass SOLOFusion with ConvNext-Base backbone by 1.0% mAP and 1.7% NDS. Scaling up the backbone to ViT-Large [8], StreamPETR achieves 62.0% of mAP, 67.6% of NDS, and 25.8% of mAOE. Note that it is the first online multi-view method that achieves comparable performance with CenterPoint.

For 3D multi-object tracking task, we simply extend the multi-object tracking of CenterPoint [53] to the multi-view 3D setting. Owing to the exceptional detection and velocity estimation performance, StreamPETR significantly outperforms ByteTrackv2 [58] with an impressive margin of +8.9% AMOTA in Tab. 3. Furthermore, StreamPETR excels over CenterPoint [53] in AMOTA, and demonstrates superior benefits in RECALL.

Waymo Open Dataset. In this section, we provide experimental results on the Waymo val set, as shown in Tab. 4. Our model has trained 24 epochs and the saving interval of the memory queue is set to 5. It can be seen that our method shows superiority in official metrics compared with the dense BEV methods, e.g. BEVFormer++ [26] and MV-FCOS3D++ [45]. The Waymo open dataset has a larger evaluation range than nuScenes, our object-centric modeling method still shows obvious advantages in localization capability and longitudinal prediction. We also re-implemented PETR-DN and PETRv2 (all with query denoising [22]) as baseline models. StreamPETR outperforms the single-frame PETR-DN with a margin of 4.1% mAPL, 5.1% mAP, and 5.5% mAP-H. The Waymo open dataset covers part of the horizontal FOV, while object-centric temporal modeling still brings significant improvement. When we adopt the checkpoint and adjust saving interval τ\tau to 1 during testing, StreamPETR has slight performance degradation, proving the adaptability on sensor frequency.

### 5.4 Ablation Study & Analysis

Impact of Training Sequence Length. StreamPETR is trained in local sliding windows and tested in online streaming video. To analyze the inconsistency between training and testing, we conduct experiments with varying numbers of training frames and show results in Tab. 5. When adding more training frames, the performance of StreamPETR continues to grow, and the performance gap between sliding windows and online video decreases obviously. It is worth noting that when the number of training frames increases to 8, video testing (40.2% mAP, 50.5% NDS) shows superior performance than the sliding window (39.6% mAP, 50.1% NDS), which proves that our method has a good potential to build long-term temporal dependency. Expanding to 12 frames brings limited performance improvement, so we train our models on 8 frames for experimental efficiency.

Effect of Motion-aware Layer Normalization. We compare the different designs for decoupling the ego vehicle and moving objects in Tab. 6. It can be seen that the performance does not improve when adopting explicit motion compensation (MC). We argue that the explicit way may cause error propagation in the early training phase. The MLN implicitly encodes and decouples the movements of the ego vehicle and moving objects. Specifically, implicit encoding of ego poses has achieved significant improvements, among which mAP increases by 2.0% and NDS increases by 1.8%. Besides, the encoding of relative time offset △​t\triangle t and object velocity vv can further boost the performance. Both mAP and NDS are increased by 0.4%, which indicates that dynamic properties have a beneficial effect on the temporal interaction between object queries.

| Training frames | Test | mAP↑\uparrow | NDS↑\uparrow | mATE↓\downarrow | mAVE↓\downarrow |
|---|---|---|---|---|---|
| 1 | - | 0.317 | 0.372 | 0.770 | 0.885 |
| 2 | W | 0.328 | 0.410 | 0.742 | 0.726 |
| 2 | V | 0.315 | 0.401 | 0.738 | 0.767 |
| 4 | W | 0.377 | 0.483 | 0.683 | 0.385 |
| 4 | V | 0.366 | 0.475 | 0.685 | 0.392 |
| 8 | W | 0.396 | 0.501 | 0.664 | 0.324 |
| 8 | V | 0.402 | 0.505 | 0.660 | 0.316 |
| 12 | W | 0.403 | 0.507 | 0.649 | 0.325 |
| 12 | V | 0.402 | 0.509 | 0.645 | 0.316 |

| MC | LN | Ego Pose | Time | Velocity | mAP↑\uparrow | NDS↑\uparrow | mATE↓\downarrow | mAVE↓\downarrow |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  | 0.378 | 0.483 | 0.697 | 0.354 |
| ✔ |  |  |  |  | 0.380 | 0.481 | 0.693 | 0.379 |
|  | ✔ |  |  |  | 0.375 | 0.481 | 0.702 | 0.370 |
|  | ✔ | ✔ |  |  | 0.398 | 0.501 | 0.667 | 0.316 |
|  | ✔ |  | ✔ |  | 0.381 | 0.488 | 0.697 | 0.354 |
|  | ✔ |  |  | ✔ | 0.386 | 0.489 | 0.690 | 0.373 |
|  | ✔ | ✔ | ✔ | ✔ | 0.402 | 0.505 | 0.660 | 0.316 |

| number frames | mAP↑\uparrow | NDS↑\uparrow | mATE↓\downarrow | mAVE↓\downarrow | FPS↑\uparrow |
|---|---|---|---|---|---|
| 0 | 0.317 | 0.372 | 0.770 | 0.885 | 27.7 |
| 1 | 0.394 | 0.501 | 0.669 | 0.324 | 27.7 |
| 2 | 0.401 | 0.505 | 0.660 | 0.314 | 27.4 |
| 3 | 0.400 | 0.504 | 0.663 | 0.322 | 27.3 |
| 4 | 0.402 | 0.505 | 0.660 | 0.316 | 27.1 |

Number of Frames for Long-term Fusion. In Tab. 7, we analyze the impacts of memory size on hybrid attention. We can find that the mAP and NDS are improved with the increase of the memory size and begin to saturate when reaching 2 frames (nearly 1 second). The object query in StreamPETR is propagated and updated recursively, so even without a large-capacity memory queue, our method can still build a long-term spatial-temporal dependency. Since increasing the memory queue brings negligible computing costs, we use 4 frames to alleviate forgetting and obtain more stable results.

Perspective v.s. Object-Centric. StreamPETR achieves efficient temporal modeling through the interaction of sparse object queries. An alternative solution is to build temporal interaction via the perspective memory [31]. As shown in Tab. 8, the query-based temporal modeling has superior performance than perspective-based both on speed and accuracy. The combination of the query and perspective memory does not further improve the performance, implying that the temporal propagation of global query interaction is sufficient to achieve leading performance. Besides, concatenating current object queries with the queries of the last frame improves 0.7% mAP and 0.9% NDS.

Analysis of Moving Objects. In this section, we detailed analyze the performance of StreamPETR on perceiving static and moving objects respectively. For fair comparisons, all models are trained with 24 epochs without CBGS [59] and evaluated on the nuScenes [1] val set. The detection performance of moving objects still lags behind that of static objects to a large margin even with temporal modeling. Compared with dense BEV paradigms [23, 17], StreamPETR∗* has reached promising performance on both static and moving objects. This proves the superiority of object-centric temporal modeling, which has global temporal and spatial receptive fields. Applying the implicit encoding for motion information, the performance of StreamPETR can be further improved.

| Perspective | Object | Propagated | mAP↑\uparrow | NDS↑\uparrow | mATE↓\downarrow | mAVE↓\downarrow | FPS↑\uparrow |
|---|---|---|---|---|---|---|---|
|  |  |  | 0.317 | 0.372 | 0.770 | 0.885 | 27.7 |
| ✔ |  | ✔ | 0.361 | 0.459 | 0.731 | 0.374 | 18.9 |
|  | ✔ |  | 0.395 | 0.496 | 0.703 | 0.363 | 27.1 |
|  | ✔ | ✔ | 0.402 | 0.505 | 0.660 | 0.316 | 27.1 |
| ✔ | ✔ | ✔ | 0.402 | 0.503 | 0.662 | 0.341 | 18.6 |

### 5.5 Failure Cases

We show the detection results of a challenging scene in Fig. 6. StreamPETR shows impressive results on crowded objects within the detection range of 30m. However, our method has many False Positives on remote objects. It is a common phenomenon of camera-based methods. In a complex urban scene, the duplicated predictions on remote objects can be tolerable and cause relatively little impact.

## 6 Conclusion

In this paper, we propose StreamPETR, an effective long-sequence 3D object detector. Different from the previous works, our method explores an object-centric paradigm that propagates temporal information through object queries frame by frame. In addition, a motion-aware layer normalization is adopted to introduce the motion information. StreamPETR achieves leading performance improvements while introducing negligible storage and computation cost. It is the first online multi-view method that achieves comparable performance with lidar-based methods. We hope StreamPETR can provide some new insights into long-sequence modeling for the community.

## A. Appendix

### A.1. Algorithm Workflow

Input: Multi-view 2D features from streaming video F2​d={F2​d1,F2​d2,⋯,F2​dT}\textbf{F}_{2d}=\{F_{2d}^{1},F_{2d}^{2},\cdots,F_{2d}^{T}\}. A set of learnable reference points Qpi​n​i​tQ_{p}^{init}. Memory Queue: Memory queue of NN historical frames X={Xt−N,⋯,Xt−2,Xt−1}\textbf{X}=\{X^{t-N},\cdots,X^{t-2},X^{t-1}\}. For each time stamp t−k{t-k}, we maintain the query states with motion information, Xt−k={△​tt−k,Qct−k,Qpt−k,vt−k,Et−k}X^{t-k}=\{\triangle t^{t-k},Q_{c}^{t-k},Q_{p}^{t-k},v^{t-k},E^{t-k}\}. Output: 3D bounding boxes prediction with classification scores bt=(x,y,z,l,w,h,θ,vx,vy,c​l​s)\textbf{b}^{t}=(x,y,z,l,w,h,\theta,v_{x},v_{y},cls).

The workflow of our proposed Propagation Transformer is shown in Alg. 1, which is divided into four stages:

(1) The motion compensation takes the information of the memory queue as input (including the relative time interval △​t\triangle t, context embedding QcQ_{c}, object center QpQ_{p}, velocity vv, and ego-pose matrix EE). The object 3D centers Qpt−N:t−1{Q}_{p}^{t-N:t-1} in the memory queue are explicitly aligned to the current frame according to the ego pose E​g​oEgo. Then the aligned centers Q~pt−N:t−1\tilde{Q}_{p}^{t-N:t-1} are used to generate position encoding of object query by a single MLP layer ψ\psi. Afterward, we apply the Motion aware Layer Normalization (MLN) to encode motion information MM into the historical queries.

(2) The generation of the object queries is mainly based on the learnable query embedding [Qp​ei​n​i​t,Qci​n​i​t][Q_{pe}^{init},Q_{c}^{init}] and the obtained propagated query embedding [Q~p​et−1,Q~ct−1][\tilde{Q}_{pe}^{t-1},\tilde{Q}_{c}^{t-1}] from the last frame t−1t-1.

(3) The spatial-temporal interaction of the Propagation Transformer has stacked L layers of the hybrid attention (H​y​b​r​i​d​_​A​t​t​nHybrid\_Attn) and cross attention (C​r​o​s​s​_​A​t​t​nCross\_Attn). For each layer, the hybrid attention performs the interaction of current queries [Qci,Qp​ei][Q_{c}^{i},Q_{pe}^{i}] and historical queries [Q~ch​y​b​r​i​d,Q~p​eh​y​b​r​i​d][\tilde{Q}_{c}^{hybrid},\tilde{Q}_{pe}^{hybrid}], and the cross attention performs the interaction of current queries and image tokens [F2​dt,F3​d​_​p​et][F_{2d}^{t},F_{3d\_pe}^{t}]. F3​d​_​p​etF_{3d\_pe}^{t} is the 3D position encoding proposed in PETR [30].

(4) After the layer-by-layer refinement, a 3D Head (H​e​a​dHead) are conducted to generate the predictions. Then we select top-K foreground objects according to the classification scores and push the information of the selected objects to the memory queue X.

| Flash Attn | A100 Training Time (s/iter) ↓\downarrow | GPU Memory (G)↓\downarrow |
|---|---|---|
| ✔ | 1.51 | 27G |
| ✘ | 1.68 | 61G |

### A.2. Additional Details

Qualitative results of StreamPETR are provided in video.

Here we provide more details for reproducing the results. First, we detach the gradient of the first 6 frames and compute the gradient and losses of the last 2 frames, which can accelerate the convergence. We additionally adopt Flash Attention [6] to further save the GPU memory, as shown in Tab. 9. The query denoising [22] is also conducted following PETRv2 [31]. When measuring the inference speed, we close the Flash Attention.

| Method | mAP↑\uparrow | NDS↑\uparrow | mATE↓\downarrow | mAVE↓\downarrow | FPS↑\uparrow |
|---|---|---|---|---|---|
| DETR3D | 0.347 | 0.422 | 0.765 | 0.876 | 6.3 |
| Stream-DETR3D | 0.396 | 0.490 | 0.723 | 0.487 | 6.2 |

### A.3. Extension of Our Method

To verify the extensibility of our method, we conduct experiments on another sparse query base model DETR3D [47]. We use ResNet101-DCN as the backbone, without additional augmentation and CBGS [59]. Results in Tab. 10 show that StreamDETR3D brings 4.9% and 6.8% improvements on mAP and NDS, while the inference speed is little impacted. Compared with the PETR paradigm, the improvement of DETR3D is relatively small. One possible reason is that the local spatial attention adopted by DETR3D limits the performance.
