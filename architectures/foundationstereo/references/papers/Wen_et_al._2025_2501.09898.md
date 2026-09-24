# FoundationStereo: Zero-Shot Stereo Matching

Tremendous progress has been made in deep stereo matching to excel on benchmark datasets through per-domain fine-tuning. However, achieving strong zero-shot generalization — a hallmark of foundation models in other computer vision tasks — remains challenging for stereo matching. We introduce FoundationStereo, a foundation model for stereo depth estimation designed to achieve strong zero-shot generalization. To this end, we first construct a large-scale (1M stereo pairs) synthetic training dataset featuring large diversity and high photorealism, followed by an automatic self-curation pipeline to remove ambiguous samples. We then design a number of network architecture components to enhance scalability, including a side-tuning feature backbone that adapts rich monocular priors from vision foundation models to mitigate the sim-to-real gap, and long-range context reasoning for effective cost volume filtering. Together, these components lead to strong robustness and accuracy across domains, establishing a new standard in zero-shot stereo depth estimation. Project page: https://nvlabs.github.io/FoundationStereo/

## 1 Introduction

Since the advent of the first stereo matching algorithm nearly half a century ago [42], we have come a long way. Recent stereo algorithms can achieve amazing results, almost saturating the most challenging benchmarks—thanks to the proliferation of training datasets and advances in deep neural network architectures. Yet, fine-tuning on the dataset of the target domain is still the method of choice to get competitive results. Given the zero-shot generalization ability shown on other problems within computer vision via the scaling law [32, 46, 78, 79], what prevents stereo matching algorithms from achieving a similar level of generalization?

Leading stereo networks [73, 80, 53, 41, 54, 11] construct cost volumes from the unary features and leverage 3D CNNs for cost filtering. Refinement-based methods [36, 86, 60, 21, 34, 27, 67, 14] iteratively refine the disparity map based on recurrent modules such as Gated Recurrent Units (GRU). Despite their success on public benchmarks under per-domain fine-tuning setup, however, they struggle to gather non-local information to effectively scale to larger datasets. Other methods [35, 68] explore transformer architectures for unary feature extraction, while lacking the specialized structure afforded by cost volumes and iterative refinement to achieve high accuracy.

Such limitations have, to date, hindered the development of a stereo network that generalizes well to other domains. While it is true that cross-domain generalization has been explored by some prior works [82, 84, 17, 10, 49, 37], such approaches have not achieved results that are competitive with those obtained by fine-tuning on the target domain, either due to insufficient structure in the network architecture, impoverished training data, or both. These networks are generally experimented on Scene Flow [43], a rather small dataset with only 40K annotated training image pairs. As a result, none of these methods can be used as an off-the-shelf solution, as opposed to the strong generalizability of vision foundation models that have emerged in other tasks.

To address these limitations, we propose FoundationStereo, a large foundation model for stereo depth estimation that achieves strong zero-shot generalization without per-domain fine-tuning. We train the network on a large-scale (1M image pairs) high-fidelity synthetic training dataset with high diversity and photorealism. An automatic self-curation pipeline is developed to eliminate the ambiguous samples that are inevitably introduced during the domain randomized data generation process, improving both the dataset quality and model robustness over iterate updates. To mitigate the sim-to-real gap, we propose a side-tuning feature backbone that adapts internet-scale rich priors from DepthAnythingV2 [79] that is trained on real monocular images to the stereo setup. To effectively leverage these rich monocular priors embedded into the 4D cost volume, we then propose an Attentive Hybrid Cost Volume (AHCF) module, consisting of 3D Axial-Planar Convolution (APC) filtering that decouples standard 3D convolution into two separate spatial- and disparity-oriented 3D convolutions, enhancing the receptive fields for volume feature aggregation; and a Disparity Transformer (DT) that performs self-attention over the entire disparity space within the cost volume, providing long range context for global reasoning. Together, these innovations significantly enhance the representation, leading to better disparity initialization, as well as more powerful features for the subsequent iterative refinement process.

Our contributions can be summarized as follows:

- ∙\bullet We present FoundationStereo, a zero-shot generalizable stereo matching model that achieves comparable or even more favorable results to prior works fine-tuned on a target domain; it also significantly outperforms existing methods when applied to in-the-wild data.
We present FoundationStereo, a zero-shot generalizable stereo matching model that achieves comparable or even more favorable results to prior works fine-tuned on a target domain; it also significantly outperforms existing methods when applied to in-the-wild data.

- ∙\bullet We create a large-scale (1M) high-fidelity synthetic dataset for stereo learning with high diversity and photorealism; and a self-curation pipeline to ensure that bad samples are pruned.
We create a large-scale (1M) high-fidelity synthetic dataset for stereo learning with high diversity and photorealism; and a self-curation pipeline to ensure that bad samples are pruned.

- ∙\bullet To harness internet-scale knowledge containing rich semantic and geometric priors, we propose a Side-Tuning Adapter (STA) that adapts the ViT-based monocular depth estimation model [79] to the stereo setup.
To harness internet-scale knowledge containing rich semantic and geometric priors, we propose a Side-Tuning Adapter (STA) that adapts the ViT-based monocular depth estimation model [79] to the stereo setup.

- ∙\bullet We develop Attentive Hybrid Cost Filtering (AHCF), which includes an hourglass module with 3D Axial-Planar Convolution (APC), and a Disparity Transformer (DT) module that performs full self-attention over the disparity dimension.
We develop Attentive Hybrid Cost Filtering (AHCF), which includes an hourglass module with 3D Axial-Planar Convolution (APC), and a Disparity Transformer (DT) module that performs full self-attention over the disparity dimension.

## 2 Related Work

Deep Stereo Matching. Recent advances in stereo matching have been driven by deep learning, significantly enhancing accuracy and generalization. Cost volume aggregation methods construct cost volumes from unary features and perform 3D CNN for volume filtering [73, 80, 53, 41, 54, 11], though the high memory consumption prevents direct application to high resolution images. Iterative refinement methods, inspired by RAFT [57], bypasses the costly 4D volume construction and filtering by recurrently refining the disparity [36, 86, 60, 21, 34, 27, 67, 14]. While they generalize well to various disparity range, the recurrent updates are often time-consuming, and lack long-range context reasoning. Recent works [71, 72] thus combine the strengths of cost filtering and iterative refinement. With the tremendous progress made by vision transformers, another line of research [35, 23, 68] introduces transformer architecture to stereo matching, particularly in the unary feature extraction stage. Despite their success on per-domain fine-tuning setup, zero-shot generalization still remains challenging. To tackle this problem, [82, 84, 17, 10, 49, 37] explore learning domain-invariant features for cross-domain generalization, with a focus on training on Scene Flow [43] dataset. Concurrent work [3] achieves remarkable zero-shot generalization with monocular prior enhanced correlation volumes. However, the strong generalizability of vision foundation models emerged in other tasks that is supported by scaling law has yet to be fully realized in stereo matching for practical applications.

| Properties | Sintel [6] | Sceneflow [43] | CREStereo [34] | IRS [64] | TartanAir [66] | FallingThings [61] | UnrealStereo4K [59] | Spring [44] | FSD (Ours) |  |
|---|---|---|---|---|---|---|---|---|---|---|
| Scenarios | Flying Objects | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Indoor | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |  |
| Outdoor | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ |  |
| Driving | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |  |
| Movie | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |  |
| Simulator | Blender | Blender | Blender | Unreal Engine | Unreal Engine | Unreal Engine | Unreal Engine | Blender | NVIDIA Omniverse |  |
| Rendering Realism | High | Low | High | High | High | High | High | High | High |  |
| Scenes | 10 | 9 | 0 | 4 | 18 | 3 | 8 | 47 | 12 |  |
| Layout Realism | Medium | Low | Low | High | High | Medium | High | High | High |  |
| Stereo Pairs | 1K† | 40K† | 200K | 103K† | 306K† | 62K | 7.7K | 6K† | 1000K |  |
| Resolution | 1024×4361024\times 436 | 960×540960\times 540 | 1920×10801920\times 1080 | 960×540960\times 540 | 640×480640\times 480 | 960×540960\times 540 | 3840×21603840\times 2160 | 1920×10801920\times 1080 | 1280×7201280\times 720 |  |
| Reflections | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |  |
| Camera Params | Constant | Constant | Constant | Constant | Constant | Constant | Constant | Constant baseline, varying intrinsics | Varying baseline and intrinsics |  |

Stereo Matching Training Data. Training data is essential for deep learning models. KITTI 12 [20] and KITTI 15 [45] provide hundreds of training pairs on driving scenarios. DrivingStereo [76] further scales up to 180K stereo pairs. Nevertheless, the sparse ground-truth disparity obtained by LiDAR sensors hinders learning accurate and dense stereo matching. Middlebury [51] and ETH3D [52] develop a low number of training data covering both indoor and outdoor scenarios beyond driving. Booster [48] presents a real-world dataset focusing on transparent objects. InStereo2K [2] presents a larger training dataset consisting of 2K stereo pairs with denser ground-truth disparity obtained with structured light system. However, challenges of scarce data size, imperfect ground-truth disparity and lack of collection scalability in real-world have driven the widespread adoption of synthetic data for training. This includes Scene Flow [43], Sintel [6], CREStereo [34], IRS [64], TartanAir [66], FallingThings [61], Virtual KITTI 2 [7], CARLA HR-VS [75], Dynamic Replica [28]. In Tab. 1, we compare our proposed FoundationStereo dataset (FSD) with commonly used synthetic training datasets for stereo matching. Our dataset encompasses a wide range of scenarios, features the largest data volume to date, includes diverse 3D assets, captures stereo images under diversely randomized camera parameters, and achieves high fidelity in both rendering and spatial layouts.

Vision Foundation Models. Vision foundation models have significantly advanced across various vision tasks in 2D, 3D and multi-modal alignment. CLIP [47] leverages large-scale image-text pair training to align visual and textual modalities, enabling zero-shot classification and facilitating cross-modal applications. DINO series [8, 46, 38] employ self-supervised learning for dense representation learning, effectively capturing detailed features critical for segmentation and recognition tasks. SAM series [32, 50, 77] demonstrate high versatility in segmentation driven by various prompts such as points, bounding boxes, language. Similar advancements also appear in 3D vision tasks. DUSt3R [65] and MASt3R [33] present generalizable frameworks for dense 3D reconstruction from uncalibrated and unposed cameras. FoundationPose [69] develops a unified framework of 6D object pose estimation and tracking for novel objects. More closely related to this work, a number of efforts [79, 78, 29, 4] demonstrated strong generalization in monocular depth estimation task and multi-view stereo [26]. Together, these approaches exemplify under the scaling law, how foundation models in vision are evolving to support robust applications across diverse scenarios without tedious per-domain fine-tuning.

## 3 Approach

The overall network architecture is shown in Fig. 2. The rest of this section describes the various components.

### 3.1 Monocular Foundation Model Adaptation

To mitigate the sim-to-real gap when the stereo network is primarily trained on synthetic dataset, we leverage the recent advancements on monocular depth estimation trained on internet-scale real data [79, 5]. We use a CNN network to adapt the ViT-based monocular depth estimation network to the stereo setup, thus synergizing the strengths of both CNN and ViT architectures.

We explored multiple design choices for combining CNN and ViT approaches, as outlined in Fig. 3 (left). In particular, (a) directly uses the feature pyramids from the DPT head in a frozen DepthAnythingV2 [79] without using CNN features. (b) resembles ViT-Adapter [12] by exchanging features between CNN and ViT. (c) applies a 4×44\times 4 convolution with stride 4 to downscale the feature before the DepthAnythingV2 final output head. The feature is then concatenated with the same level CNN feature to obtain a hybrid feature at 1/4 scale. The side CNN network is thus learned to adapt the ViT features [83] to stereo matching task. Surprisingly, while being simple, we found (c) significantly surpasses the alternative choices on the stereo matching task, as shown in the experiments (Sec. 4.5). As a result, we adopt (c) as the main design of STA module.

Formally, given a pair of left and right images Il,Ir∈ℝH×W×3I_{l},I_{r}\in\mathbb{R}^{H\times W\times 3}, we employ EdgeNeXt-S [40] as the CNN module within STA to extract multi-level pyramid features, where the 1/4 level feature is equipped with DepthAnythingV2 feature: fl(i),fr(i)∈ℝCi×Hi×Wif_{l}^{(i)},f_{r}^{(i)}\in\mathbb{R}^{C_{i}\times\frac{H}{i}\times\frac{W}{i}}, i∈{4,8,16,32}i\in\{4,8,16,32\}. EdgeNeXt-S [40] is chosen for its memory efficiency and because larger CNN backbones did not yield additional benefits in our investigation. When forwarding to DepthAnythingV2, we first resize the image to be divisible by 14, to be consistent with its pretrained patch size. The STA weights are shared when applied to Il,IrI_{l},I_{r}.

Similarly, we employ STA to extract context feature, with the difference that the CNN module is designed with a sequence of residual blocks [25] and down-sampling layers. It generates context features of multiple scales: fc(i)∈ℝCi×Hi×Wif_{c}^{(i)}\in\mathbb{R}^{C_{i}\times\frac{H}{i}\times\frac{W}{i}}, i∈{4,8,16}i\in\{4,8,16\}, as in [36]. fcf_{c} participates in initializing the hidden state of the ConvGRU block and inputting to the ConvGRU block at each iteration, effectively guiding the iterative process with progressively refined contextual information.

Fig. 3 visualizes the power of rich monocular prior that helps to reliably predict on ambiguous regions which is challenging to deal with by naive correspondence search along the epipolar line. Instead of using the raw monocular depth from DepthAnythingV2 which has scale ambiguity, we use its latent feature as geometric priors extracted from both stereo images and compared through cost filtering as described next.

### 3.2 Attentive Hybrid Cost Filtering

Hybrid Cost Volume Construction. Given unary features at 1/4 scale fl4,fr4f_{l}^{4},f_{r}^{4} extracted from previous step, we construct the cost volume 𝐕𝐂∈ℝC×D4×H4×W4\mathbf{V_{C}}\in\mathbb{R}^{C\times\frac{D}{4}\times\frac{H}{4}\times\frac{W}{4}} with a combination of group-wise correlation and concatenation [24]:

|  |  | 𝐕gwc​(g,d,h,w)=⟨f^l,g(4)​(h,w),f^r,g(4)​(h,w−d)⟩,\displaystyle\mathbf{V_{\text{gwc}}}(g,d,h,w)=\left\langle\widehat{f}_{l,g}^{(4)}(h,w),\widehat{f}_{r,g}^{(4)}(h,w-d)\right\rangle, |  |  |
|---|---|---|---|---|
|  |  | 𝐕cat​(d,h,w)=[Conv​(fl(4))​(h,w),Conv​(fr(4))​(h,w−d)],\displaystyle\mathbf{V_{\text{cat}}}(d,h,w)=\left[\text{Conv}(f_{l}^{(4)})(h,w),\text{Conv}(f_{r}^{(4)})(h,w-d)\right], |  |  |
|  |  | 𝐕𝐂​(d,h,w)=[𝐕gwc​(d,h,w),𝐕cat​(d,h,w)]\displaystyle\mathbf{V_{C}}(d,h,w)=\left[\mathbf{V_{\text{gwc}}}(d,h,w),\mathbf{V_{\text{cat}}}(d,h,w)\right] |  | (1) |

where f^\widehat{f} denotes L2L_{2} normalized feature for better training stability; ⟨⋅,⋅⟩\left\langle\cdot,\cdot\right\rangle represents dot product; g∈{1,2,…,G}g\in\{1,2,...,G\} is the group index among the total G=8G=8 feature groups that we evenly divide the total features into; d∈{1,2,…,D4}d\in\{1,2,...,\frac{D}{4}\} is the disparity index. [⋅,⋅][\cdot,\cdot] denotes concatenation along channel dimension. The group-wise correlation 𝐕gwc\mathbf{V_{\text{gwc}}} harnesses the strengths of conventional correlation-based matching costs, offering a diverse set of similarity measurement features from each group. 𝐕cat\mathbf{V_{\text{cat}}} preserves unary features including the rich monocular priors by concatenating left and right features at shifted disparity. To reduce memory consumption, we linearly downsize the unary feature dimension to 14 using a convolution of kernel size 1 (weights are shared between fl4f_{l}^{4} and fr4f_{r}^{4}) before concatenation. Next, we describe two sub-modules for effective cost volume filtering.

Axial-Planar Convolution (APC) Filtering. An hourglass network consisting of 3D convolutions, with three down-sampling blocks and three up-sampling blocks with residual connections, is leveraged for cost volume filtering [71, 1]. While 3D convolutions of kernel size 3×3×33\times 3\times 3 are commonly used for relatively small disparity sizes [71, 24, 9], we observe it struggles with larger disparities when applied to high resolution images, especially since the disparity dimension is expected to model the probability distribution for the initial disparity prediction. However, it is impractical to naively increase the kernel size, due to the intensive memory consumption. In fact, even when setting kernel size to 5×5×55\times 5\times 5 we observe unmanageable memory usage on an 80 GB GPU. This drastically limits the model’s representation power when scaling up with large amount of training data. We thus develop “Axial-Planar Convolution” which decouples a single 3×3×33\times 3\times 3 convolution into two separate convolutions: one over spatial dimensions (kernel size Ks×Ks×1K_{s}\times K_{s}\times 1) and the other over disparity (1×1×Kd1\times 1\times K_{d}), each followed by BatchNorm and ReLU. APC can be regarded as a 3D version of Separable Convolution [16] with the difference that we only separate the spatial and disparity dimensions without subdividing the channel into groups which sacrifices representation power. The disparity dimension is specially treated due to its uniquely encoded feature comparison within the cost volume. We use APC wherever possible in the hourglass network except for the down-sampling and up-sampling layers.

Disparity Transformer (DT). While prior works [68, 35] introduced transformer architecture to unary feature extraction step to scale up stereo training, the cost filtering process is often overlooked, which remains an essential step in achieving accurate stereo matching by encapsulating correspondence information. Therefore, we introduce DT to further enhance the long-range context reasoning within the 4D cost volume. Given 𝐕𝐂\mathbf{V_{C}} obtained in Eq. (1), we first apply a 3D convolution of kernel size 4×4×44\times 4\times 4 with stride 4 to downsize the cost volume. We then reshape the volume into a batch of token sequences, each with length of disparity. We apply position encoding before feeding it to a series (4 in our case) of transformer encoder blocks, where FlashAttention [18] is leveraged to perform multi-head self-attention [63]. The process can be written as:

|  |  | 𝐐𝟎=PE​(𝐑⁡(Conv4×4×4​(𝐕𝐂)))∈ℝ(H16×W16)×C×D16\displaystyle\mathbf{Q_{0}}=\text{PE}\left(\mathbf{R}\left(\text{Conv}_{4\times 4\times 4}(\mathbf{V_{C}})\right)\right)\in\mathbb{R}^{\left(\frac{H}{16}\times\frac{W}{16}\right)\times C\times\frac{D}{16}} |  |
|---|---|---|---|
|  |  | MultiHead​(𝐐,𝐊,𝐕)=[head1,…,headh]​𝐖O\displaystyle\text{MultiHead}(\mathbf{Q},\mathbf{K},\mathbf{V})=[\text{head}_{1},\ldots,\text{head}_{h}]\mathbf{W}_{O} |  |
|  |  | where headi=FlashAttention​(𝐐i,𝐊i,𝐕i)\displaystyle\quad\text{where }\text{head}_{i}=\text{FlashAttention}(\mathbf{Q}_{i},\mathbf{K}_{i},\mathbf{V}_{i}) |  |
|  |  | 𝐐𝟏=Norm​(MultiHead​(𝐐𝟎,𝐐𝟎,𝐐𝟎)+𝐐𝟎)\displaystyle\mathbf{Q_{1}}=\text{Norm}\left(\text{MultiHead}(\mathbf{Q_{0}},\mathbf{Q_{0}},\mathbf{Q_{0}})+\mathbf{Q_{0}}\right) |  |
|  |  | 𝐐𝟐=Norm​(FFN​(𝐐𝟏)+𝐐𝟏)\displaystyle\mathbf{Q_{2}}=\text{Norm}\left(\text{FFN}(\mathbf{Q_{1}})+\mathbf{Q_{1}}\right) |  |

where 𝐑⁡(⋅)\mathbf{R}(\cdot) denotes reshape operation; PE​(⋅)\text{PE}(\cdot) represents position encoding; [⋅,⋅][\cdot,\cdot] denotes concatenation along the channel dimension; 𝐖O\mathbf{W}_{O} is linear weights. The number of heads is h=4h=4 in our case. Finally, the DT output is up-sampled to the same size as 𝐕𝐂\mathbf{V_{C}} using trilinear interpolation and summed with hourglass output, as shown in Fig. 2.

Initial Disparity Prediction. We apply soft-argmin [30] to the filtered volume 𝐕𝐂′\mathbf{V_{C}^{\prime}} to produce an initial disparity:

|  | d0=∑d=0D4−1d⋅Softmax​(𝐕𝐂′)​(d)\displaystyle d_{0}=\sum_{d=0}^{\frac{D}{4}-1}d\cdot\text{Softmax}(\mathbf{V_{C}^{\prime}})(d) |  | (2) |
|---|---|---|---|

where d0d_{0} is at 1/4 scale of the original image resolution.

### 3.3 Iterative Refinement

Given d0d_{0}, we perform iterative GRU updates to progressively refine disparity, which helps to avoid local optimum and accelerate convergence [71]. In general, the k-th update can be formulated as:

|  |  | 𝐕corr​(w′,h,w)=⟨fl(4)​(h,w),fr(4)​(h,w′)⟩\displaystyle\mathbf{V}_{\text{corr}}(w^{\prime},h,w)=\left\langle f_{l}^{(4)}(h,w),f_{r}^{(4)}(h,w^{\prime})\right\rangle |  | (3) |
|---|---|---|---|---|
|  |  | 𝐅𝐕​(h,w)=[𝐕𝐂′​(dk,h,w),𝐕corr​(w−dk,h,w)]\displaystyle\mathbf{F_{V}}(h,w)=[\mathbf{V_{C}^{\prime}}(d_{k},h,w),\mathbf{V}_{\text{corr}}(w-d_{k},h,w)] |  | (4) |
|  |  | xk=[Convv​(𝐅𝐕),Convd​(dk),dk,c]\displaystyle x_{k}=[\text{Conv}_{v}(\mathbf{F_{V}}),\text{Conv}_{d}(d_{k}),d_{k},c] |  | (5) |
|  |  | zk=σ⁡(Convz​([hk−1,xk]))\displaystyle z_{k}=\sigma\left(\text{Conv}_{z}([h_{k-1},x_{k}])\right) |  | (6) |
|  |  | rk=σ⁡(Convr​([hk−1,xk]))\displaystyle r_{k}=\sigma\left(\text{Conv}_{r}([h_{k-1},x_{k}])\right) |  | (7) |
|  |  | h^k=tanh​(Convh​([rk⊙hk−1,xk]))\displaystyle\hat{h}_{k}=\text{tanh}\left(\text{Conv}_{h}([r_{k}\odot h_{k-1},x_{k}])\right) |  | (8) |
|  |  | hk=(1−zk)⊙hk−1+zk⊙h^k\displaystyle h_{k}=(1-z_{k})\odot h_{k-1}+z_{k}\odot\hat{h}_{k} |  | (9) |
|  |  | dk+1=dk+ConvΔ​(hk)\displaystyle d_{k+1}=d_{k}+\text{Conv}_{\Delta}(h_{k}) |  | (10) |

where ⊙\odot denotes element-wise product; σ\sigma denotes sigmoid; 𝐕corr∈ℝW4×H4×W4\mathbf{V}_{\text{corr}}\in\mathbb{R}^{\frac{W}{4}\times\frac{H}{4}\times\frac{W}{4}} is the pair-wise correlation volume; 𝐅𝐕\mathbf{F_{V}} represents the looked up volume features using latest disparity; c=ReLU​(fc)c=\text{ReLU}(f_{c}) encodes the context feature from left image, including STA adapted features (Sec. 3.1) which effectively guide the refinement process leveraging rich monocular priors.

We use three levels of GRU blocks to perform coarse-to-fine hidden state update in each iteration, where the initial hidden states are produced from context features h0(i)=tanh​(fc(i)),i∈{4,8,16}h_{0}^{(i)}=\text{tanh}(f_{c}^{(i)}),i\in\{4,8,16\}. At each level, attention-based selection mechanism [67] is leveraged to capture information at different frequencies. Finally, dkd_{k} is up-sampled to the full resolution using convex sampling [57].

### 3.4 Loss Function

The model is trained with the following objective:

|  | ℒ=|d0−d¯|smooth+∑k=1KγK−k​‖dk−d¯‖1\displaystyle\mathcal{L}=\left|d_{0}-\overline{d}\right|_{\text{smooth}}+\sum_{k=1}^{K}\gamma^{K-k}\left\|d_{k}-\overline{d}\right\|_{1} |  | (11) |
|---|---|---|---|

where d¯\overline{d} represents ground-truth disparity; |⋅|smooth\left|\cdot\right|_{\text{smooth}} denotes smooth L1L_{1} loss; kk is the iteration number; γ\gamma is set to 0.9, and we apply exponentially increasing weights [36] to supervise the iteratively refined disparity.

### 3.5 Synthetic Training Dataset

We created a large scale synthetic training dataset with NVIDIA Omniverse. This FoundationStereo Dataset (FSD) accounts for crucial stereo matching challenges such as reflections, low-texture surfaces, and severe occlusions. We perform domain randomization [58] to augment dataset diversity, including random stereo baseline, focal length, camera perspectives, lighting conditions and object configurations. Meanwhile, high-quality 3D assets with abundant textures and path-tracing rendering are leveraged to enhance realism in rendering and layouts. Fig. 4 displays some samples from our dataset including both structured indoor and outdoor scenarios, as well as more diversely randomized flying objects with various geometries and textures under complex yet realistic lighting. See the appendix for details.

Iterative Self-Curation. While synthetic data generation in theory can produce unlimited amount of data and achieve large diversity through randomization, ambiguities can be inevitably introduced especially for less structured scenes with flying objects, which confuses the learning process. To eliminate those samples, we design an automatic iterative self-curation strategy. Fig. 4 demonstrates this process and detected ambiguous samples. We start with training an initial version of FoundationStereo on FSD, after which it is evaluated on FSD. Samples where BP-2 (Sec. 4.2) is larger than 60% are regarded as ambiguous samples and replaced by regenerating new ones. The training and curation processes are alternated to iteratively (twice in our case) update both FSD and FoundationStereo.

## 4 Experiments

### 4.1 Implementation Details

We implement FoundationStereo in PyTorch. The foundation model is trained on a mixed dataset consisting of our proposed FSD, together with Scene Flow [43], Sintel [6], CREStereo [34], FallingThings [61], InStereo2K [2] and Virtual KITTI 2 [7]. We train FoundationStereo using AdamW optimizer [39] for 200K steps with a total batch size of 128 evenly distributed over 32 NVIDIA A100 GPUs. The learning rate starts at 1e-4 and decays by 0.1 at 0.8 of the entire training process. Images are randomly cropped to 320×\times736 before feeding to the network. Data augmentations similar to [36] are performed. During training, 22 iterations are used in GRU updates. In the following, unless otherwise mentioned, we use the same foundation model for zero-shot inference with 32 refinement iterations and 416 for maximum disparity.

### 4.2 Benchmark Datasets and Metric

Datasets. We consider five commonly used public datasets for evaluation: Scene Flow [43] is a synthetic dataset including three subsets: FlyingThings3D, Driving, and Monkaa. Middlebury [51] consists of indoor stereo image pairs with high-quality ground-truth disparity captured via structured light. Unless otherwise mentioned, evaluations are performed on half resolution and non-occluded regions. ETH3D [52] provides grayscale stereo image pairs covering both indoor and outdoor scenarios. KITTI 2012 [20] and KITTI 2015 [45] datasets feature real-world driving scenes, where sparse ground-truth disparity maps are provided, which are derived from LIDAR sensors.

Metrics. “EPE” computes average per-pixel disparity error. “BP-X” computes the percentage of pixels where the disparity error is larger than X pixels. “D1” computes the percentage of pixels whose disparity error is larger than 3 pixels and 5% of the ground-truth disparity.

### 4.3 Zero-Shot Generalization Comparison

| Methods | Middlebury | ETH3D | KITTI-12 | KITTI-15 |
|---|---|---|---|---|
| BP-2 | BP-1 | D1 | D1 |  |
| CREStereo++ [27] | 14.8 | 4.4 | 4.7 | 5.2 |
| DSMNet [82] | 13.8 | 6.2 | 6.2 | 6.5 |
| Mask-CFNet [49] | 13.7 | 5.7 | 4.8 | 5.8 |
| HVT-RAFT [10] | 10.4 | 3.0 | 3.7 | 5.2 |
| RAFT-Stereo [36] | 12.6 | 3.3 | 4.7 | 5.5 |
| Selective-IGEV [67] | 9.2 | 5.7 | 4.5 | 5.6 |
| IGEV [36] | 8.8 | 4.0 | 5.2 | 5.7 |
| Former-RAFT-DAM [84] | 8.1 | 3.3 | 3.9 | 5.1 |
| IGEV++ [72] | 7.8 | 4.1 | 5.1 | 5.9 |
| NMRF [22] | 7.5 | 3.8 | 4.2 | 5.1 |
| Ours (Scene Flow) | 5.5 | 1.8 | 3.2 | 4.9 |
| Selective-IGEV* [67] | 7.5 | 3.4 | 3.2 | 4.5 |
| Ours | 1.1 | 0.5 | 2.3 | 2.8 |

Benchmark Evaluation. Tab. 2 exhibits quantitative comparison of zero-shot generalization results on four public real-world datasets. Even when trained solely on Scene Flow, our method outperforms the comparison methods consistently across all datasets, thanks to the efficacy of adapting rich monocular priors from vision foundation models. We further evaluate in a more realistic setup, allowing methods to train on any available dataset while excluding the target domain, to achieve optimal zero-shot inference results as required in practical applications.

| Method | LEAStereo [15] | GANet [81] | ACVNet [70] | IGEV-Stereo [71] | NMRF [22] | MoCha-Stereo [14] | Selective-IGEV [67] | Ours |
|---|---|---|---|---|---|---|---|---|
| EPE | 0.78 | 0.84 | 0.48 | 0.47 | 0.45 | 0.41 | 0.44 | 0.34 |

| Method | Zero-Shot | BP-0.5 | BP-1.0 | EPE |
|---|---|---|---|---|
| GMStereo [74] | ✗ | 5.94 | 1.83 | 0.19 |
| HITNet [56] | ✗ | 7.83 | 2.79 | 0.20 |
| EAI-Stereo [85] | ✗ | 5.21 | 2.31 | 0.21 |
| RAFT-Stereo [36] | ✗ | 7.04 | 2.44 | 0.18 |
| CREStereo [34] | ✗ | 3.58 | 0.98 | 0.13 |
| IGEV-Stereo [71] | ✗ | 3.52 | 1.12 | 0.14 |
| CroCo-Stereo [68] | ✗ | 3.27 | 0.99 | 0.14 |
| MoCha-Stereo [14] | ✗ | 3.20 | 1.41 | 0.13 |
| Selective-IGEV [67] | ✗ | 3.06 | 1.23 | 0.12 |
| Ours (finetuned) | ✗ | 1.26 | 0.26 | 0.09 |
| Ours | ✓ | 2.31 | 1.52 | 0.13 |

In-the-Wild Generalization. We compare our foundation model against recent approaches that released their checkpoints trained on a mixture of datasets, to resemble the practical zero-shot application on in-the-wild images. Comparison methods include CroCo v2 [68], CREStereo [34], IGEV [71] and Selective-IGEV [67]. For each method, we select the best performing checkpoint from their public release. In this evaluation, the four real-world benchmark datasets [51, 52, 20, 45] have been used for training comparison methods, whereas they are not used in our fixed foundation model. Fig. 5 displays qualitative comparison on various scenarios, including a robot scene from DROID [31] dataset and custom captures covering indoor and outdoor.

### 4.4 In-Domain Comparison

Tab. 3 presents quantitative comparison on Scene Flow, where all methods are following the same officially divided train and test split. Our FoundationStereo model outperforms the comparison methods by a large margin, reducing the previous best EPE from 0.41 to 0.33. Although in-domain training is not the focus of this work, the results reflect the effectiveness of our model design.

Tab. 4 exhibits quantitative comparison on ETH3D leaderboard (test set). For our approach, we perform evaluations in two settings. First, we fine-tune our foundation model on a mixture of the default training dataset (Sec. 4.1) and ETH3D training set for another 50K steps, using the same learning rate schedule and data augmentation. Our model significantly surpasses the previous best approach by reducing more than half of the error rates and ranks 1st on leaderboard at the time of submission. This indicates great potential of transferring capability from our foundation model if in-domain fine-tuning is desired. Second, we also evaluated our foundation model without using any data from ETH3D. Remarkably, our foundation model’s zero-shot inference achieves comparable or even better results than leading approaches that perform in-domain training.

In addition, our finetuned model also ranks 1st on the Middlebury leaderboard. See appendix for details.

### 4.5 Ablation Study

| Row | Variations | BP-2 |
|---|---|---|
| 1 | DINOv2-L [46] | 2.46 |
| 2 | DepthAnythingV2-S [79] | 2.22 |
| 3 | DepthAnythingV2-B [79] | 2.11 |
| 4 | DepthAnythingV2-L [79] | 1.97 |
| 5 | STA (a) | 6.48 |
| 6 | STA (b) | 2.22 |
| 7 | STA (c) | 1.97 |
| 8 | Unfreeze ViT | 3.94 |
| 9 | Freeze ViT | 1.97 |

| Row | Variations | BP-2 |  | Row | Variations | BP-2 |
|---|---|---|---|---|---|---|
| 1 | RoPE | 2.19 |  | 10 | (3,3,1), (1,1,5) | 2.10 |
| 2 | Cosine | 1.97 |  | 11 | (3,3,1), (1,1,9) | 2.06 |
| 3 | 1/32 | 2.06 |  | 12 | (3,3,1), (1,1,13) | 2.01 |
| 4 | 1/16 | 1.97 |  | 13 | (3,3,1), (1,1,17) | 1.97 |
| 5 | Full | 2.25 |  | 14 | (3,3,1), (1,1,21) | 1.98 |
| 6 | Disparity | 1.97 |  | 15 | (7,7,1), (1,1,17) | 1.99 |
| 7 | Pre-hourglass | 2.06 |  |  |  |  |
| 8 | Post-hourglass | 2.20 |  |  |  |  |
| 9 | Parallel | 1.97 |  |  |  |  |

We investigate different design choices for our model and dataset. Unless otherwise mentioned, we train on a randomly subsampled version (100K) of FSD to make the experiment scale more affordable. Given Middlebury dataset’s high quality ground-truth, results are evaluated on its training set to reflect zero-shot generalization. Since the focus of this work is to build a stereo matching foundation model with strong generalization, we do not deliberately limit model size while pursuing better performance.

STA Design Choices. As shown in Tab. 5, we first compare different vision foundation models for adapting rich monocular priors, including different model sizes of DepthAnythingV2 [79] and DINOv2-Large [46]. While DINOv2 previously exhibited promising results in correspondence matching [19], it is not as effective as DepthAnythingV2 in the stereo matching task, possibly due to its less task-relevance and its limited resolution to reason high-precision pixel-level correspondence. We then study different design choices from Fig. 3. Surprisingly, while being simple, we found (c) significantly surpasses the alternatives. We hypothesize the latest feature before the final output head preserves high-resolution and fine-grained semantic and geometric priors that are suitable for subsequent cost volume construction and filtering process. We also experimented whether to freeze the adapted ViT model. As expected, unfreezing ViT corrupts the pretrained monocular priors, leading to degraded performance.

| Row | STA | AHCF | BP2 |  | Row | FSD | BP2 |  |
|---|---|---|---|---|---|---|---|---|
| APC | DT |  | 1 | ✗ | 2.34 |  |  |  |
| 1 |  |  |  | 2.48 |  | 2 | ✓ | 1.15 |
| 2 | ✓ |  |  | 2.21 |  |  |  |  |
| 3 | ✓ | ✓ |  | 2.16 |  |  |  |  |
| 4 | ✓ |  | ✓ | 2.05 |  |  |  |  |
| 5 | ✓ | ✓ | ✓ | 1.97 |  |  |  |  |

AHCF Design Choices. As shown in Tab. 6, for DT module we study different position embedding (row 1-2); different feature scale to perform transformer (row 3-4); transformer over the full cost-volume or only along the disparity dimension (row 5-6); different placements of DT module relative to the hourglass network (row 7-9). Specifically, RoPE [55] encodes relative distances between tokens instead of absolute positions, making it more adaptive to varying sequence lengths. However, it does not outperform cosine position embedding, probably due to the constant disparity size in 4D cost volume. While in theory, full volume attention provides larger receptive field, it is less effective than merely applying over the disparity dimension of the cost volume. We hypothesize the extremely large space of 4D cost volume makes it less tractable, whereas attention over disparity provides sufficient context for a better initial disparity prediction and subsequent volume feature lookup during GRU updates. Next, we compare different kernel sizes in APC (row 10-15), where the last dimension in each parenthesis corresponds to disparity dimension. We observe increasing benefits when enlarging disparity kernel size until it saturates at around 17.

Effects of Proposed Modules. The quantitative effects are shown in Tab. 7 (left). STA leverages rich monocular priors which greatly enhances generalization to real images for ambiguous regions. DT and APC effectively aggregate cost volume features along spatial and disparity dimensions, leading to improved context for disparity initialization and subsequent volume feature look up during GRU updates. Fig. 3 further visualizes the resulting effects.

Effects of FoundationStereo Dataset. We study whether to include FSD dataset with the existing public datasets for training our foundation model described in Sec. 4.1. Results are shown in Tab. 7 (right).

## 5 Conclusion

We introduced FoundationStereo, a foundation model for stereo depth estimation that achieves strong zero-shot generalization across various domains without fine-tuning. We envision such a foundation model can facilitate broader adoption of stereo estimation models in practical applications. Despite its remarkable generalization, it has several limitations. First, our model is not yet optimized for efficiency, which takes 0.7s on image size of 375×\times1242 on NVIDIA A100 GPU. Future work could explore adapting distillation and pruning techniques applied to other vision foundation models [87, 13]. Second, our dataset FSD includes a limited collection of transparent objects. Robustness could be further enhanced by augmenting with a larger diversity of fully transparent objects during training.

Supplementary Material

## 6 ETH3D Leaderboard

At the time of submission, our fine-tuned model ranks 1st on the ETH3D leaderboard, significantly outperforming both published and unpublished works. The screenshot is shown in Fig. 6.

## 7 Middlebury Leaderboard

At the time of submission, our fine-tuned model ranks 1st on the Middlebury leaderboard, significantly outperforming both published and unpublished works. The screenshot is shown in Fig. 8.

## 8 More Ablation Study on Synthetic Data

Effects of Self-Curation. We study the effectiveness of self-curation pipeline introduced in Sec. 3.5. When disabling the self-curation while keeping the same data size, the synthetic dataset involves ambiguous samples that confuse the learning process, leading to slight performance drop when evaluated on Middlebury [51] dataset.

| Variation | BP2 |
|---|---|
| W/ self-curation | 1.15 |
| W/o self-curation | 1.27 |

Effects of FSD for Other Methods. Table 2 (main paper) indicates benefits by introducing FSD for FoundationStereo model. To answer the question of whether FSD can benefit other methods beyond FoundationStereo, we now train representative works IGEV and Selective-IGEV on FSD and compare with their counterparts trained on Scene Flow. As shown in the table below, for both methods, our proposed FSD effectively boosts the performance compared to the commonly used Scene Flow dataset.

| Methods | Train data | Middlebury | ETH3D | KITTI-12 | KITTI-15 |
|---|---|---|---|---|---|
| BP-2 | BP-1 | D1 | D1 |  |  |
| IGEV | Scene Flow | 8.8 | 4.0 | 5.2 | 5.7 |
| IGEV | FSD | 7.8 | 3.5 | 3.2 | 4.7 |
| Selective-IGEV | Scene Flow | 9.2 | 5.7 | 4.5 | 5.6 |
| Selective-IGEV | FSD | 7.9 | 3.5 | 3.0 | 4.4 |

## 9 Results on Translucent Objects

We evaluate on Booster [48] (half resolution), which is a challenging dataset consisting of specular and transparent objects. We compare with the most competitive methods from Fig. 5 (main paper) in the zero-shot setting. The quantitative and qualitative results are shown below.

| Methods | Half |  |  |  |
|---|---|---|---|---|
| BP1 | BP-2 | BP-3 | EPE |  |
| Selective-IGEV | 23.8 | 15.0 | 12.0 | 6.6 |
| IGEV | 30.8 | 22.3 | 19.0 | 22.7 |
| Ours | 19.0 | 9.6 | 6.7 | 2.2 |

## 10 More Results on Middlebury Dataset

We compare with competitive methods that released their public weights in zero-shot on Middlebury, shown in below table. Since NMRF [22] did not report their evaluated Middlebury resolution, we rerun their released weights on all resolutions. At full resolution, maximum disparity 320 is used for FoundationStereo. Across all resolutions, ours significantly outperforms baselines. We also report the peak memory usage and running time averaged across the dataset on the same hardware, particularly single GPU 3090. On half and quarter resolutions, our peak memory occurs at STA module. On full resolution, it occurs at DT module. Despite the speed limitation which is not the focus when developing this work, ours can successfully run on a desktop GPU. Pruning or distillation remains an interesting future work to improve speed and memory footprint.

| Methods | Full | Half | Quarter |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
| BP-2 | peak mem (G) | time (s) | BP-2 | peak mem (G) | time (s) | BP-2 | peak mem (G) | time (s) |  |
| Selective- IGEV[61] | 12.9 | 6.9 | 2.52 | 9.2 | 1.7 | 0.72 | 7.0 | 0.5 | 0.25 |
| IGEV[33] | 13.1 | 6.3 | 2.06 | 8.8 | 1.6 | 0.53 | 6.4 | 0.5 | 0.18 |
| IGEV++[66] | 12.7 | 13.1 | 2.12 | 7.8 | 3.4 | 0.50 | 6.3 | 0.9 | 0.15 |
| NMRF[20] | 35.3 | 8.1 | 0.95 | 10.9 | 1.8 | 0.20 | 5.0 | 0.5 | 0.05 |
| Ours | 4.8 | 18.5 | 8.14 | 1.1 | 10.5 | 2.97 | 1.3 | 2.3 | 0.55 |

## 11 More Details of Synthetic Data Generation

Tooling and Assets. The dataset generation is built on NVIDIA Omniverse. We use RTX path-tracing with 32 to 128 samples per pixel for high-fidelity photorealistic rendering. The data generation is performed across 48 NVIDIA A40 GPUs for 10 days. There are more than 5K object assets collected from varying sources including artist designs and 3D scanning with high-frequency geometry details. Object assets are divided into the groups of: furniture, open containers, vehicles, robots, floor tape, free-standing walls, stairs, plants, forklifts, dynamically animated digital humans, other obstacles and distractors. Each group is defined with a separate randomization range for sampling locations, scales and appearances. In addition, we curated 12 large scene models (Fig. 7), 16 skybox images, more than 150 materials, and 400 textures for tiled wrapping on object geometries for appearance augmentation. These textures are obtained from real-world photos and procedurally generated random patterns.

Camera Configuration. For each data sample, we first randomly sample the stereo baseline camera focal length to diversify the coverage of field-of-views and disparity distributions. Next, objects are spawned into the scene in two different methods to randomize the scene configuration: 1) camera is spawned in a random pose, and objects are added relative to the camera at random locations; 2) objects are spawned near a random location, and the camera is spawned nearby and oriented to the center of mass of the object clutter.

Layout Configuration. We generate layouts in two kinds of styles: chaotic and realistic. Such combination of the more realistic structured layouts with the more randomized setups with flying objects has been shown to benefit sim-to-real generalization [62]. Specifically, chaotic-style scenes involve large number of flying distractors and simple scene layouts which consists of infinitely far skybox and a background plane. The lighting and object appearances (texture and material) are highly randomized. The realistic-style data uses indoor and outdoor scene models where the camera is restricted to locate at predefined areas. Object assets are dropped and applied with physical properties for collision. The simulation is performed randomly between 0.25 to 2 seconds to create physically realistic layouts with no penetration, involving both settled and falling objects. Materials and scales native to object assets are maintained and more natural lighting is applied. Among the realistic-style data, we further divide the scenes into three types which determine what categories of objects are selected to compose the scene for more consistent semantics:

- ∙\bullet Navigation - camera poses are often in parallel to the ground and objects are often spawned further away. Objects such as free-standing walls, furniture, and digital humans are sampled with higher probability.
Navigation - camera poses are often in parallel to the ground and objects are often spawned further away. Objects such as free-standing walls, furniture, and digital humans are sampled with higher probability.

- ∙\bullet Driving - camera is often in parallel to the ground above the ground and objects are often spawned further away. Objects such as vehicles, digital humans, poles, signs and speed bumps are sampled with higher probability.
Driving - camera is often in parallel to the ground above the ground and objects are often spawned further away. Objects such as vehicles, digital humans, poles, signs and speed bumps are sampled with higher probability.

- ∙\bullet Manipulation - camera is oriented to face front or downward as in ego-centric views and objects are often spawned in closer range to resemble interaction scenarios. Objects such as household or grocery items, open containers, robotic arms are sampled with higher probability.
Manipulation - camera is oriented to face front or downward as in ego-centric views and objects are often spawned in closer range to resemble interaction scenarios. Objects such as household or grocery items, open containers, robotic arms are sampled with higher probability.

Lighting Configuration. Light types include global illumination, directed sky rays, lights baked-into 3D scanned assets, and light spheres which add dynamic lighting when spawned near to surfaces. Light colors, intensities and directions are randomized. Lighting vibes such as daytime, dusk and night are included within the random sampling ranges.

Disparity Distribution. Fig. 9 shows the disparity distribution of our FSD dataset.

## 12 Acknowledgement

We would like to thank Gordon Grigor, Jack Zhang, Xutong Ren, Karsten Patzwaldt, Hammad Mazhar and other NVIDIA Isaac team members for their tremendous engineering support and valuable discussions.
