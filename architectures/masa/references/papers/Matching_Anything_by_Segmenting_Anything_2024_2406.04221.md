# Matching Anything by Segmenting Anything

The robust association of the same objects across video frames in complex scenes is crucial for many applications, especially Multiple Object Tracking (MOT). Current methods predominantly rely on labeled domain-specific video datasets, which limits the cross-domain generalization of learned similarity embeddings. We propose MASA, a novel method for robust instance association learning, capable of matching any objects within videos across diverse domains without tracking labels. Leveraging the rich object segmentation from the Segment Anything Model (SAM), MASA learns instance-level correspondence through exhaustive data transformations. We treat the SAM outputs as dense object region proposals and learn to match those regions from a vast image collection. We further design a universal MASA adapter which can work in tandem with foundational segmentation or detection models and enable them to track any detected objects. Those combinations present strong zero-shot tracking ability in complex domains. Extensive tests on multiple challenging MOT and MOTS benchmarks indicate that the proposed method, using only unlabeled static images, achieves even better performance than state-of-the-art methods trained with fully annotated in-domain video sequences, in zero-shot association. Our code is available at github.com/siyuanliii/masa.

### 1 Introduction

Multiple Object Tracking (MOT) is one of the fundamental problems in computer vision. It plays a pivotal role in numerous robotics systems such as autonomous driving. Tracking requires both detecting the objects of interest in videos and associating them across frames. While recent advancements in vision foundation models [35, 78, 40, 70, 33, 47] have demonstrated an exceptional ability to detect, segment, and perceive depth for any objects, associating those objects in videos remains challenging. Recent successful multiple object tracking approaches [36, 66] have emphasized the importance of learning discriminative instance embeddings for accurate association. Some [46] even argued that it is the only necessary tracking component besides detection.

However, learning effective object association usually requires a significant amount of annotated data. While collecting detection labels on a diverse set of images is laborious, obtaining tracking labels on videos is even more challenging. Consequently, current MOT datasets mostly focus on objects from a specific domain with a small number of fixed categories or a limited number of labeled frames.

Training on those datasets limits the generalizability of tracking models to different domains and novel concepts. Although recent studies [40, 78, 35] have made successful attempts to address the model generalization issue for object detection and segmentation, the path to learning a universal association model for tracking any objects is still unclear.

Our goal is to develop a method capable of matching any objects or regions. We aim to integrate this generalizable tracking capability with any detection and segmentation methods to help them track any object they have detected. A primary challenge is acquiring matching supervision for general objects across diverse domains, without incurring substantial labelling costs.

To this end, we propose the Matching Anything by Segmenting Anything (MASA) pipeline to learn object-level associations from unlabeled images of any domain. Figure 1 presents an overview of our MASA pipeline. We leverage the rich object appearance and shape information encoded by the foundation segmentation SAM, combined with extensive data transformation, to establish strong instance correspondence.

Applying different geometric transformations to the same image gives automatic pixel-level correspondence in two views from the same image. SAM’s segmentation ability allows for the automatic grouping of pixels from the same instance, facilitating the conversion of pixel-level to instance-level correspondence. This process creates a self-supervision signal for learning discriminative object representation, utilizing dense similarity learning between view pairs. Our training strategy enables us to use a rich collection of raw images from diverse domains, demonstrating that such automatic self-training on diverse raw images provides excellent zero-shot multiple object tracking performance, even surpassing models reliant on in-domain video annotations for association learning.

Beyond the self-training pipeline, we further build a universal tracking adapter — MASA adapter, to empower any existing open-world segmentation and detection foundation models such as SAM [35], Detic [78] and Grounding-DINO [40] for tracking any objects they have detected. To preserve their original segmentation and detection ability, we freeze their original backbone and add the MASA adapter on the top.

Moreover, we propose a multi-task training pipeline that jointly performs the distillation of SAM’s detection knowledge and instance similarity learning. This approach allows us to learn the object’s location, shape and appearance prior of SAM, and simulate real detection proposals during contrastive similarity learning. This pipeline further improves the generalization capabilities of our tracking features. Additionally, our learned detection head speeds up the original SAM dense uniform point proposals for segmenting everything by over tenfold, crucial for tracking applications.

We evaluate MASA on multiple challenging benchmarks, including TAO MOT [17], Open-vocabulary MOT [37], MOT and MOTS on BDD100K [71], and UVO [55]. Extensive experiments indicate that compared with state-of-the-art object tracking approaches trained on thoroughly in-domain labeled videos, our method achieves on-par or even better association performance, using a single model with the same model parameters and testing in zero-shot association settings.

### 2 Related Work

Learning robust instance-level correspondence is crucial to object tracking. Existing approaches can be divided into self-supervised [58] and supervised [46, 63, 66, 36, 72, 76, 9, 44, 57, 65] strategies. Specifically, as a representative self-supervised method, UniTrack [58] attempts to directly use off-the-shelf self-supervised representations [11, 64] for association. Despite competitive results on some benchmarks [45], these methods cannot fully exploit instance-level training data, limiting their performance in challenging scenarios. In contrast, supervised methods train discriminative instance embeddings on frame pairs, by contrastive learning. Although achieving superior performance on challenging benchmarks [71, 17, 50, 41, 37], these methods rely on tremendous in-domain labeled video data. Several methods [2, 20, 77, 37, 79] learn tracking signals from static images but still require substantial fine-grained instance annotations in specific domains or post-hoc test-time adaptation [53], limiting their ability for cross-domain generalization. To tackle these problems, we exploit the exhaustive object shape, and appearance information encoded by SAM to learn universal instance matching, purely from unlabeled images. Our learned representation shows exceptional zero-shot association ability across diverse domains.

Deva [14], TAM [67] and SAM-Track [15] integrate SAM [35] with video object segmentation (VOS) approaches (such as XMem [13] and DeAOT [69]) to enable an interactive pipeline for tracking any object, where SAM is mainly used for mask initialization/correction and XMem/DeAOT handle the tracking and prediction. SAM-PT [49] combines SAM with point-tracking methods such as [29, 24, 54] to perform tracking. However, all those approaches face limitations, such as poor mask propagation quality due to domain gaps and the inability to handle multiple diverse objects or rapid objects entry and exit, common in scenarios like autonomous driving. Our work focuses on a different direction. Instead of building an interactive tracking pipeline or using off-the-shelf VOS or point-based trackers, we focus on learning universal association modules by leveraging SAM’s rich instance segmentation knowledge.

### 3 Method

SAM [35] is composed of three modules: (a) Image encoder: A heavy ViT-based backbone for feature extraction. (b) Prompt encoder: Modeling the positional information from the interactive points, box, or mask prompts. (c) Mask decoder: A transformer-based decoder takes both the extracted image embedding with the concatenated output and prompt tokens for final mask prediction. To generate all potential mask proposals, SAM adopts densely sampled regular grids as point anchors and generates mask predictions for each point prompt. The complete pipeline includes patch cropping with greedy box-based NMS, three-step filtering, and heavy post-processing on masks. For more details on SAM’s everything mode, we refer readers to [35].

Our method consists of two key components. First, based on SAM, we develop a new pipeline: MASA (Section 3.2.1). With this pipeline, we construct exhaustive supervision for dense instance-level correspondence from a rich collection of unlabeled images. It enables us to learn strong discriminative instance representations to track any objects, without requiring any video annotations. Second, we introduce a universal MASA adapter (Section 3.2.2) to effectively transform the features from a frozen detection or segmentation backbone for learning generalizable instance appearance representations. As a byproduct, the distillation branch of the MASA adapter can also significantly improve the efficiency of segmenting everything. Besides, we also construct a unified model to jointly detect / segment and track anything (Section 3.2.3). Our complete training pipeline is shown in Figure 2.

To learn instance-level correspondence, previous works [46, 65, 36, 76, 66] heavily relied on manually labeled in-domain video data. However, current video datasets [45, 6, 71] contain only a limited range of fixed categories. This limited diversity in datasets leads to learning appearance embeddings that are tailored to specific domains, posing challenges in their universal generalization.

UniTrack [58] demonstrates that universal appearance features can be learned through contrastive self-supervised learning techniques [64, 11, 8] from raw images or videos. These representations, harnessing the diversity of a large volume of unlabeled images, can generalize across different tracking domains. However, they often depend on clean, object-centered images, such as those in ImageNet [52], or videos like DAVIS17 [48], and focus on frame-level similarities. This focus causes them to fail in fully leveraging instance information, leading to difficulties in learning discriminative instance representations in complex domains with multiple instances, as demonstrated in Table 7.

To address these issues, we propose the MASA training pipeline. Our core idea is to increase diversity from two perspectives: training image diversity and instance diversity. As shown in Figure 1, we first construct a rich collection of raw images from diverse domains to prevent learning domain-specific features. These images also contain a rich number of instances in complex environments to enhance instance diversity. Given an image II, we simulate appearance changes in videos by adopting two different augmentations on the same image. By applying strong data augmentations φ⁡(I)\varphi(I) and ϕ⁡(I)\phi(I), we construct two different views V1V_{1} and V2V_{2} of II, thereby automatically obtaining pixel-level correspondence.

If the image is clean and contains only one instance, such as those in ImageNet, frame-level similarity can be applied as in [11, 74, 64]. However, with multiple instances, we need to further mine the instance information contained in such raw images. The foundational segmentation model SAM [35] offers us this capability. SAM automatically groups pixels belonging to the same instances and also provides the shape and boundary information of detected instances, valuable for learning discriminative features.

Since we construct the dataset by selecting images with multiple instances, SAM’s exhaustive segmentation of the entire images automatically yields a dense and diverse collection of instance proposals QQ. With pixel-level correspondences established, applying the same ϕ⁡(⋅)\phi(\cdot) and φ⁡(⋅)\varphi(\cdot) to QQ transfers pixel-level correspondence to dense instance-level correspondence. This self-supervision signal enables us to use the contrastive learning formula from [34, 46, 36] to learn a discriminative contrastive embedding space:

|  | ℒ𝒞=−∑q∈Qlogesim⁡(q,q+)τesim⁡(q,q+)τ+∑q−∈Q−esim⁡(q,q−)τ,\displaystyle\mathcal{L_{C}}=-\sum_{q\in Q}{\mathrm{log}\frac{e^{\frac{\mathrm{sim}(q,q^{+})}{\tau}}}{e^{\frac{\mathrm{sim}(q,q^{+})}{\tau}}+\sum_{q^{-}\in Q^{-}}e^{\frac{\mathrm{sim}(q,q^{-})}{\tau}}}}, |  |
|---|---|---|

Here, q+q^{+} and q−q^{-} denote the positive and negative samples to qq, respectively. Positive samples are the same instance proposals being applied different ϕ⁡(⋅)\phi(\cdot) and φ⁡(⋅)\varphi(\cdot). Negative samples are from different instances. Furthermore, sim⁡(⋅)\mathrm{sim(\cdot)} denotes the cosine similarity and τ\tau is a temperature parameter, set to 0.07 in our experiments.

This contrastive learning formula pushes object embeddings belonging to the same instance closer while distancing embeddings from different instances. As demonstrated by existing works [10, 46], negative samples are crucial for learning discriminative representations. Under the contrastive learning paradigm, the dense proposals generated by SAM naturally provide more negative samples, thus enhancing learning better instance representation for association.

We introduce the MASA adapter, designed to extend the open-world segmentation and detection models (such as SAM [35], Detic [78], and Grounding-DINO [40]) to track any detected objects. The MASA adapter operates in conjunction with frozen backbone features from these foundational models, ensuring their original detection and segmentation capabilities are preserved. However, as not all pre-trained features are inherently discriminative for tracking, we first transform these frozen backbone features into new features more suitable for tracking.

Given the diversity in shapes and sizes of objects, we construct a multi-scale feature pyramid. For hierarchical backbones like the Swin Transformer [42] in Detic and Grounding DINO, we directly employ FPN [39]. For SAM, which utilizes a plain ViT [18] backbone, we use Transpose Convolution and MaxPooling to upsample and downsample the single-scale features of stride 16×16\times to produce hierarchical features with scale ratios of 14,18,116,132{\frac{1}{4},\frac{1}{8},\frac{1}{16},\frac{1}{32}}. To effectively learn discriminative features for different instances, it’s essential that objects in one location are aware of the appearances of instances in other locations. Hence, we use deformable convolution to generate dynamic offsets and aggregate information across spatial locations and feature levels as [16]:

|  | F(p)=1L∑j=1L∑k=1Kwk⋅Fj​(p+pk+Δ​pkj)⋅Δ​mkj,\displaystyle\begin{split}F(p)=\frac{1}{L}\sum_{j=1}^{L}\sum_{k=1}^{K}&w_{k}\cdot F^{j}(p+p_{k}+\Delta p_{k}^{j})\cdot\Delta m_{k}^{j},\end{split} |  | (1) |
|---|---|---|---|

where LL represents the feature level, KK is the number of sampling locations for a convolutional kernel, wkw_{k} and pkp_{k} are the weight and predefined offset for the kk-th location, respectively, and Δ​pkj\Delta p_{k}^{j} and Δ​mkj\Delta m_{k}^{j} are the learnable offset and modulation factor for the kk-th location at the jj-th feature level. For SAM-based models, we additionally use task-aware attention and scale-aware attention from Dyhead [16], since the detection performance is important for accurate auto mask generation as in Figure 3 (b). After acquiring the transformed feature map, we extract instance-level features by applying RoI-Align [26] to the visual features FF, followed by processing with a lightweight track head comprising 44 convolutional layers and 11 fully connected layer to generate instance embeddings.

Additionally, we introduce an object prior distillation branch as an auxiliary task during training. This branch employs a standard RCNN [51] detection head to learn bounding boxes that tightly encompass SAM’s mask predictions for each instance. It effectively learns exhaustive object location and shape knowledge from SAM and distils this information into the transformed feature representations. This design not only strengthens the features of the MASA adapter, resulting in improved association performance but also accelerates SAM’s everything mode by directly providing the predicted box prompts.

The MASA adapter is optimized using a combination of detection and contrastive losses as defined in Section 3.2.1: ℒ=ℒdet+ℒC\mathcal{L}=\mathcal{L}_{\text{det}}+\mathcal{L}_{C}. The detection loss is identical to that in [51].

Figure 3 shows the test pipeline with our unified models.

Detect and Track Anything When we integrate the MASA adapter with object detectors, we remove the MASA detection head that was learned during training. The MASA adapter then solely serves as a tracker. The detectors predict the bounding boxes, and then they are utilized to prompt the MASA adapter, which retrieves corresponding tracking features for instance matching. We use a simple bi-softmax nearest neighbor search for accurate instance matching, as illustrated in Section J.4 of the Appendix.

Segment and Track Anything With SAM, we keep the detection head. We use it to predict all potential objects within a scene, forwarding box predictions as prompts to both the SAM mask decoder and the MASA adapter for segmenting and tracking everything. The predicted box prompts omit the need for the heavy post-processing illustrated in the original SAM’s everything mode, therefore, significantly speeding up the auto mask generation of SAM.

Testing with Given Observations When detections are obtained from sources other than the one the MASA adapter is built upon, our MASA adapter serves as a tracking feature provider. We directly utilize the provided bounding boxes as prompts to extract tracking features from our MASA adapter through the ROI-Align [26] operation.

### 4 Experiments

We perform experiments on multiple challenging MOT/MOTS benchmarks with diverse domains.

| Method | TETA | LocA | AssocA | ClsA |
|---|---|---|---|---|
| Fully-supervised, in-domain |  |  |  |  |
| SORT [5] | 24.9 | 48.1 | 14.3 | 12.1 |
| Tracktor [3] | 24.2 | 47.4 | 13.0 | 12.1 |
| Tracktor++ [3] | 28.0 | 49.0 | 22.8 | 12.1 |
| DeepSORT [59] | 26.0 | 48.4 | 17.5 | 12.1 |
| AOA [19] | 25.3 | 23.4 | 30.6 | 21.9 |
| QDTrack [46] | 30.0 | 50.5 | 27.4 | 12.1 |
| TETer [36]† | 34.6 | 52.1 | 36.7 | 15.0 |
| Self-supervised, zero-shot |  |  |  |  |
| Ours-Detic† | 34.7 | 51.9 | 36.4 | 15.8 |
| Ours-Grounding-DINO† | 34.9 | 51.8 | 37.6 | 15.4 |
| Ours-SAM-B† | 34.5 | 51.8 | 36.6 | 15.1 |
| Ours-SAM-H† | 34.5 | 51.8 | 36.4 | 15.4 |
| Ours-Detic | 46.3 | 65.8 | 44.1 | 28.9 |

| Method | Base | Novel |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
| TETA | LocA | AssocA | ClsA | TETA | LocA | AssocA | ClsA |  |
| DeepSORT [59] | 28.4 | 52.5 | 15.6 | 17.0 | 24.5 | 49.2 | 15.3 | 9.0 |
| Tracktor++ [3] | 29.6 | 52.4 | 19.6 | 16.9 | 25.7 | 50.1 | 18.9 | 8.1 |
| Bytetrack [75] | 29.5 | 51.7 | 19.7 | 17.2 | 25.4 | 49.4 | 18.1 | 8.7 |
| OC-SORT [7] | 30.0 | 53.3 | 23.5 | 13.3 | 26.7 | 51.5 | 21.6 | 7.1 |
| OVTrack [37] | 36.3 | 53.9 | 36.3 | 18.7 | 32.0 | 51.4 | 33.2 | 11.4 |
| Ours-Detic | 47.0 | 66.0 | 44.5 | 30.5 | 40.8 | 64.4 | 41.2 | 17.0 |

| Method | Track mAP50 | Track mAP75 | Track mAP |
|---|---|---|---|
| Fully-supervised, in-domain |  |  |  |
| SORT-TAO [17] | 13.2 | - | - |
| QDTrack [46] | 15.9 | 5.0 | 10.6 |
| TAC [61] | 17.7 | 5.8 | 7.3 |
| BIV [60] | 21.6 | 10.4 | 16.1 |
| GTR†\dagger [79] | 22.5 | - | - |
| Self-supervised, zero-shot |  |  |  |
| Ours-Detic†\dagger | 22.0 | 12.2 | 17.1 |
| Ours-Grounding-DINO†\dagger | 22.8 | 12.3 | 17.6 |
| Ours-SAM-B†\dagger | 23.9 | 13.0 | 18.4 |
| Ours-SAM-H†\dagger | 22.9 | 12.1 | 17.5 |
| Ours-Detic | 30.9 | 18.0 | 24.4 |

| Method | mIDF1↑\uparrow | AssocA↑\uparrow | TETA↑\uparrow | mMOTSA↑\uparrow | mHOTA↑\uparrow |
|---|---|---|---|---|---|
| Fully-supervised, in-domain |  |  |  |  |  |
| MaskTrackRCNN [68] | 26.2 | - | - | 12.3 | - |
| STEm-Seg [1] | 25.4 | - | - | 12.2 | - |
| QDTrack-mots [46] | 40.8 | - | - | 22.5 | - |
| PCAN [30] | 45.1 | 46.7 | 46.8 | 27.4 | 35.9 |
| VMT [31] | 45.7 | 47.3 | 47.1 | 28.7 | 36.6 |
| Unicorn [65] | 44.2 | - | - | 29.6 | - |
| UNINEXT-H [66]† | 48.5 | 53.2 | 53.6 | 35.7 | 40.6 |
| Self-supervised, zero-shot |  |  |  |  |  |
| Ours-Detic† | 49.5 | 53.5 | 54.4 | 36.4 | 40.2 |
| Ours-Grounding-DINO† | 48.6 | 52.3 | 54.0 | 36.1 | 40.0 |
| Ours-SAM-B† | 49.2 | 53.9 | 54.8 | 35.2 | 40.7 |
| Ours-SAM-H† | 49.7 | 54.5 | 54.7 | 35.8 | 40.8 |

| Method | mIDF1↑\uparrow | IDF1↑\uparrow | TETA↑\uparrow | AssocA↑\uparrow | mMOTA↑\uparrow |
|---|---|---|---|---|---|
| Fully-supervised, in-domain |  |  |  |  |  |
| QDTrack [46] | 50.8 | 71.5 | 47.8 | 48.5 | 36.6 |
| TETer [36] | 53.3 | 71.1 | 50.8 | 52.9 | 39.1 |
| MOTR [72] | 54.0 | 65.8 | - | - | 32.3 |
| Unicorn [65] | 54.0 | 71.3 | - | - | 41.2 |
| UNINEXT-H [66] | 56.7 | 69.9 | - | - | 44.2 |
| ByteTrack [75]† | 54.8 | 70.4 | 55.7 | 51.5 | 45.5 |
| Self-supervised, zero-shot |  |  |  |  |  |
| Ours-Detic† | 55.8 | 71.3 | 54.4 | 52.9 | 44.6 |
| Ours-Grounding-DINO† | 55.6 | 71.7 | 54.5 | 52.7 | 44.5 |
| Ours-SAM-B† | 55.6 | 71.6 | 54.0 | 52.6 | 44.1 |
| Ours-SAM-H† | 55.3 | 71.7 | 54.2 | 51.9 | 44.5 |

TAO MOT TAO dataset [17] is designed to track a diverse range of objects, encompassing over 800 categories, making it the most diverse MOT dataset with the largest class collection to date. It contains 500, 988, and 1,419 videos annotated at 1 FPS in the train, validation, and test sets, respectively. We report performances on the validation set. TAO comprises several benchmarks, each highlighting different characteristics and requirements. The TAO TETA benchmark [36] emphasizes association by rewarding trackers that produce clean trajectories with no overlaps. Conversely, the TAO Track mAP benchmark [17] values particularly the classification of trajectories, and does not heavily penalize overlapping trajectories. The open-vocabulary MOT benchmark [37] requires trackers to avoid training with annotations from novel classes, focusing on the generalization ability to track novel categories.

BDD100K MOT [71] requires trackers to track common objects in autonomous driving scenarios. The dataset is annotated at 5 FPS with 200 videos in the validation set.

BDD100K MOTS Different from BDD100K MOT, BDD100K MOTS [71] requires trackers to track and segment objects simultaneously, evaluating tracking performance on masks. There are 154 videos for training, 32 videos for validation, and 37 videos for testing.

UVO [55] is a challenging benchmark for open-world instance segmentation in videos. Compared with previous video-level object segmentation datasets [68], it annotates much more diverse instances. UVO has two evaluation tracks, an image track, and a video track. We evaluate all methods on the UVOv0.5 validation set.

Evaluation Metrics As analyzed in previous works [36], traditional tracking metrics like mMOTA [71], and track mAP [17] can be misleading, particularly in long-tail scenarios, due to their high sensitivity to classification. To address this issue, [36] introduced TETA, a new tracking metric that decomposes into three separate components: AssocA, LocA, and ClsA, reflecting the accuracy of association, localization, and classification, respectively. In standard MOT benchmarks, to ensure a fair comparison of trackers’ association abilities, we adopt the same detection observations used by leading state-of-the-art trackers. Therefore, our focus is primarily on association-related metrics like AssocA, mIDF1, and IDF1. Additionally, when evaluating our unified models, we consider the full spectrum of metrics to capture their comprehensive capabilities. Particularly for open-world segmentation on UVO, our emphasis is on AR100 and Track AR100 metrics in the image and video levels. This is due to the fact that SAM often segments every part of an object, whereas UVO lacks such detailed annotations, making traditional AP evaluations less accurate.

Training Data SA-1B [35] consists of 11M diverse, high-resolution images, containing diverse scenarios with multiple object interactions in complex environments. We sub-sample the SA-1B raw images to construct a training set of 500K images, SA-1B-500K.

Implementation Details For our models, we utilize the official weights of SAM [35], Detic, and Grounding-DINO, ensuring that all components of these models remain frozen during the training phase. Specifically, we employ SAM with both ViT-Base and ViT-Huge backbones, and Detic and Grounding-DINO are used with the SwinB backbone. We train the models with bootstrapping sampling for 200,000 images per epoch, with a batch size of 128. We use SGD with an initial learning rate of 0.04, coupled with a step policy for learning rate decay. Momentum and weight decay parameters are set to 0.9 and 1e-4. Our training spans 12 epochs, with the learning rate being reduced at the 8th and 11th epochs. For data augmentation, we use random affine, MixUp [73], and Large-scale Jittering [21], in addition to standard practices like flipping, color jittering, and random cropping. More details are provided in Section J of the Appendix.

We evaluate our methods in two ways. Firstly, to accurately assess the association ability of our method, we always provide the same detection observations as current state-of-the-art methods in standard MOT benchmarks. Secondly, to evaluate the integrated abilities of our unified models, we follow this protocol: for SAM-based models, we evaluate on the open-world video segmentation dataset UVO. For the detectors-based models, we evaluated on the open-vocabulary MOT benchmark [37]. We also report the scores on TAO TETA and TAO TrackmAP benchmarks. Note that we perform zero-shot association tests for all our variants, and use the same weights across all benchmarks.

TAO TETA We use the same observations as TETer-SwinT [36]. As shown in Table 1, our method with Grounding-DINO’s backbone performs the best, in the zero-shot setting, without training on any in-domain labeled videos, on both AssocA and TETA. We also test our unified Detic model which jointly outputs the detection and tracking results. It outperforms all other methods significantly and achieves the new state-of-the-art. It demonstrates our method can couple well with current detection foundation models and transfer their strong detection ability into tracking.

Open-vocabulary MOT Similar to the open-vocabulary object detection task [22], open-vocabulary MOT [37] stipulates that methods should only use the frequent and common classes annotations from LVIS [23] for training, treating the rare classes as novel. We evaluated our unified ’detect and track anything’ model Detic, which was trained exclusively with base class annotations. Table 2 shows our unified Detic model outperforms existing models on all metrics across both base and novel splits, and it achieves this significant lead despite our tracker being trained solely with out-of-domain, unlabeled images.

TAO Track mAP We use the same observations as GTR [79]. As shown in Table 3, our method with SAM-B performs the best (Track mAP50 of 23.9) given the same detections. Most of our models outperform the current state-of-the-art GTR, which is an offline method that utilizes future information for association. In contrast, our methods conduct tracking in an online fashion and test in a zero-shot setting. Our unified Detic model again, achieves the new state-of-the-art by outperforming GTR by a large margin.

BDD100K MOTS We use the same observations as the state-of-the-art method, UNINEXT-H [66] and perform zero-shot association test on BDD100K MOTS benchmark. As shown in Table 4, our method achieves the best association performance (mIDF1 of 49.7 and AssocA of 54.5) among all approaches. This demonstrates the superiority of the instance embeddings learned by our method.

BDD100K MOT As shown in Table 5, given the same observations as ByteTrack [75], our method achieves the best IDF1 of 71.7 and AssocA 52.9. Compared with state-of-the-art ByteTrack [75], our method also achieves better association performance, being about 1.4% higher on both IDF1 and AssocA, without using any BDD images for training. ByteTrack additionally selects low-confidence boxes and adds them to the tracklets, resulting in a better mMOTA score which prioritizes detection performance [43].

UVO VIS We perform zero-shot tests for our unified ’segment and track anything’ model based on SAM. We directly use the box prompts from the MASA detection head for faster segmenting everything. As shown in Figure 4a, our method achieves the best performance on both image and video tracks, outperforming its counterparts by a large margin. Besides, we also compare our method with SAM’s default auto mask segmentation. As shown in Figure 4b, as the inference time increases, AR100 of our method grows much faster than SAM due to the distillate detection branch. The upper bound AR100 of our method with ViT-Base backbone even surpasses SAM by 10%. Besides, when achieving the same AR100, our method is about 10×10\times faster than SAM. This stems from the fact that our method learns a strong object prior to capturing potential objects with a small number of sparse proposals. However, to segment everything, SAM has to sample about 1​k1k points evenly, which is inflexible and inefficient, while also relying on hand-crafted complex post-processing methods.

Compare with VOS Methods We evaluated the VOS-based method Deva [14], which integrates XMem [13] for tracking multiple objects and SAM-PT [49], which uses point-tracking. To ensure a fair comparison, we provide the same observations on BDD MOTS, TAO TETA and UVO benchmarks. For UVO, we use SAM’s auto-mask generation to generate masks first, then we resolve the overlapping masks following the heuristic in Deva [14] and use Deva to generate per-frame observations.

Table 6 shows that our method outperforms Deva across all benchmarks. Notably, on the autonomous driving BDD100K benchmark, where objects frequently enter and exit the scene, VOS-based methods like Deva are prone to a significant increase in false positives. This is reflected in the TETA scores, where such errors are heavily penalized. Additionally, Deva struggles with overlapping predictions, a common issue with current detection models. We provide a more in-depth analysis in Section H of the Appendix.

Compare with Self-supervised Methods We further compare our approach with self-supervised methods aimed at learning universal appearance features from raw images or videos. To ensure a fair comparison, we train all methods using a mix of BDD and COCO raw images. Specifically, for VFS, we utilize raw videos from BDD. We employ a ResNet-50 model for VFS [64] and MoCov2 [11], and a ViT-B model for DINO [8], following the association tracking strategy outlined in UniTrack [58]. Additionally, we ensure that detection observations are identical across all models. Table 7 demonstrates that our methods significantly outperform other self-supervised approaches. This advantage stems from the fact that traditional self-supervised learning primarily focuses on frame-level similarities, which limits their effectiveness in leveraging instance information and causes struggles when training with images containing multiple objects. Further analysis of this is provided in Section G of the Appendix.

| (a) Quantitative results on UVO [55] dataset. |  |  |
|---|---|---|
| Track | Method | AR100 |
| image | Mask R-CNN [26] | 41.3 |
| Ours-SAM-B | 43.7 |  |
| Ours-SAM-H | 50.8 |  |
|  | Fully-supervised, in-domain |  |
| video | MaskTrack R-CNN [68] | 17.2 |
| QDTrack [46] | 20.9 |  |
| SeqFormer [62] | 21.3 |  |
| Zero-shot test |  |  |
| Ours-SAM-B | 24.9 |  |
| Ours-SAM-H | 28.4 |  |

| Method | BDD MOTS | TAO | UVO |  |
|---|---|---|---|---|
| AssocA | TETA | AssocA | AR100 |  |
| SAM-PT [49] | - | - | - | 31.8 |
| Deva† [14] | 46.8 | 32.1 | 22.4 | 36 |
| Ours-SAM-H† | 54.5 | 54.7 | 36.4 | 37.5 |

| Method | Video | BDD MOT | TAO | BDD MOTS |  |  |
|---|---|---|---|---|---|---|
| AssocA | mIDF1 | AssocA | AssocA | mIDF1 |  |  |
| Train on BDD & COCO |  |  |  |  |  |  |
| VFS [64] | ✓ | 29.2 | 35.0 | 19.1 | 30.7 | 30.1 |
| MoCov2 [11] | ✗ | 42.7 | 46.7 | 30.7 | 51 | 45.3 |
| DINO [8] | ✗ | 23.1 | 16.8 | 12.9 | 20.2 | 22.2 |
| Ours-SAM-B | ✗ | 51.9 | 54.9 | 35.8 | 53.7 | 49.1 |

To reduce the training costs, we bootstrap fewer raw images (40K) for training for the ablation experiments. Unless specified we train the model with an image collection containing 70k raw images from [71] and 110k images from [38] training set respectively. We employ the Ours-SAM-B model and test on BDD MOT and TAO TETA benchmarks.

Effect of Training Strategies and Model Architectures Table 8 illustrates that directly using the off-the-shelf SAM features (row 1) for association yields poor results. The primary reason is that SAM’s original features are optimized for segmentation, not for instance-level discrimination. However, integrating our MASA training approach and adding a lightweight track head significantly enhances performance, yielding improvements of 15.6%15.6\% in AssocA and 14.4%14.4\% in mIDF1 on BDD MOT. This underscores the efficacy of our training strategy. Incorporating a dynamic feature fusion block further enhances performance by 1.6%1.6\%. Additionally, joint training with the object prior distillation branch leads to an increase of 1.8%1.8\% in AssocA and 1.6%1.6\% in mIDF1, showing the effect of these architectural designs.

| MASA training | Dynamic feature fusion | Object prior distillation | AssocA | mIDF1 |
|---|---|---|---|---|
|  |  |  | 32.9 | 37.3 |
| ✓ |  |  | 48.5 | 51.7 |
| ✓ | ✓ |  | 50.1 | 53.3 |
| ✓ | ✓ | ✓ | 51.9 | 54.9 |

| (a) The effect of proposal quality. |  |  |
|---|---|---|
|  | AssocA | AssocA |
| Mask2Former | 46.4 | 29.8 |
| SAM | 50.9 | 34.1 |

| (b) The effect of proposal number. |  |  |
|---|---|---|
| Instance Number | BDD | TAO |
| AssocA | AssocA |  |
| 64 | 47.1 | 31.5 |
| 128 | 50.9 | 34.6 |
| 256 | 51.9 | 35.8 |

| (c) The effect of data augmentation. |  |  |  |  |  |  |
|---|---|---|---|---|---|---|
| # | Affine | Mixup | LSJ | BDD MOT | TAO |  |
| mIDF1 | AssocA | AssocA |  |  |  |  |
| 1 |  |  |  | 48.2 | 43.9 | 28.5 |
| 2 | ✓ |  |  | 53.0 | 49.1 | 33.3 |
| 3 |  | ✓ |  | 52.8 | 49.9 | 31.5 |
| 4 |  |  | ✓ | 52.9 | 49.3 | 32.3 |
| 5 | ✓ | ✓ | ✓ | 54.9 | 51.9 | 35.8 |

Effect of Proposal Diversity We evaluate different proposal generation mechanisms in association learning. We use only raw images from the training set of the BDD detection task for training. By substituting SAM in our MASA pipeline with Mask2former-SwinL [12], pre-trained on COCO. As shown in Table 9a, we found that the model trained with SAM’s proposals significantly enhanced both in-domain performance on BDD and zero-shot tracking on TAO. This underscores the importance of SAM’s dense, diverse object proposals for superior contrastive similarity learning.

Effect of Proposal Quantity Investigating the impact of SAM’s proposal quantity on learning, we experimented with different upper bounds of 64, 128, and 256 proposals per batch. Table 9b shows consistent improvements in AssocA on BDD and TAO with increasing proposal numbers, indicating that a rich collection of instances fosters more discriminative tracking features.

Effect of Data Augmentations As shown in Table 9c, the combination of random affine, Mixup [73] and LSJ [21] gives the best performance. Method 1 represents basic data augmentation including flipping, resizing, color jitter and random cropping. If there is no strong augmentation (method 1), its mIDF1 on BDD MOT drops by 6.7%, being much worse than that with method 5. These results illustrate the necessity of strong augmentations in training only on static images.

Qualitative Results In Figure 14, we present the qualitative results of our unified methods, Grounding-DINO and SAM-H. Our methods accurately detect, segment, and track multiple objects and even their parts across diverse domains. This includes animated movie scenes featuring many similar-looking characters and driving scenes within complex environments.

### 5 Conclusion

We present MASA, a novel method that exploits the extensive instance-level shape and appearance information from SAM to learn generalizable instance associations from unlabeled images. MASA demonstrates exceptional zero-shot association performance across various benchmarks, eliminating the need for expensive domain-specific labels. Moreover, our universal MASA adapter can be added to any existing detection and segmentation models, enabling them to efficiently track any objects across diverse domains.

### 6 Acknowledgments

We would like to express our gratitude to Ruolan Xiang for her help with editing, and to Bin Yan for both editing and the valuable discussions.

## Appendix

In this supplementary material, we provide additional ablation studies and qualitative results of our fast proposal generation and of our association. We also elaborate on our experimental setup, method details, and training and inference hyper-parameters.

The supplementary material is structured as follows:

- • Section A: Effectiveness on other backbones.
Section A: Effectiveness on other backbones.

- • Section B: Zero-shot evaluation on YoutubeVIS.
Section B: Zero-shot evaluation on YoutubeVIS.

- • Section C: Visualization of instance embeddings.
Section C: Visualization of instance embeddings.

- • Section D: Domain gap and adaptation.
Section D: Domain gap and adaptation.

- • Section E: Impact of additional photometric augmentation.
Section E: Impact of additional photometric augmentation.

- • Section F: More detailed comparison of proposal diversity.
Section F: More detailed comparison of proposal diversity.

- • Section G: Compare with self-supervised methods.
Section G: Compare with self-supervised methods.

- • Section H: Comparison with VOS-based methods.
Section H: Comparison with VOS-based methods.

- • Section I: More qualitative results.
Section I: More qualitative results.

- • Section J: More implementation details.
Section J: More implementation details.

- • Section K: Limitations.
Section K: Limitations.

### A Effectiveness on Other Backbones

In our main paper, we introduced four method variants, each building upon foundational detection and segmentation models: SAM-ViT-B, SAM-ViT-H, Grounding-DINO, and Detic. Notably, the latter two variants leverage the Swin-B backbone. Our MASA training pipeline and adapter have shown great adaptability to a range of variables, including variations in backbone structures, pre-training methods (such as detection or segmentation), and the diverse datasets employed in training these foundational models.

A critical observation from our study is the reliance of these variants on large, complex backbones and their pre-training on extensive datasets. This reliance poses an important question about scalability and efficiency: Can our method sustain its effectiveness when applied to smaller, more streamlined backbones, like the ResNet-50, especially with standard ImageNet pre-training? To explore this, we devised a new variant, "Ours-R50," which integrates the ResNet-50 backbone pre-trained on the ImageNet classification task (IN-Sup R50). This new variant maintains the MASA adapter architecture from our main research, adhering to the identical training protocol established by our initial four variants.

We have assessed the performance of Ours-R50 across various benchmarks, including BDD MOTS, BDD MOT, and TAO TETA. The quantitative results, detailed in Tables 10, 11, and 12, demonstrate the efficacy of Ours-R50. These findings are significant as they suggest that our approach can be effectively adapted to smaller backbones, offering the potential for more efficient and scalable solutions in detection and segmentation tasks.

BDD MOTS: For BDD MOTS (Table 10), Ours-R50, equipped with the ResNet-50 backbone and our MASA training approach, not only outperforms the UNINEXT-H model with a +0.2 mIDF1 and +0.4 AssocA but also shows minimal performance drop compared to the strongest variant, Ours-SAM-H (-1 mIDF1 and -0.9 AssocA). This highlights our method’s ability to yield competitive instance embeddings, even without the advanced features provided by larger, specialized models.

BDD MOT: In the BDD MOT benchmark (Table 11), Ours-R50 surpasses ByteTrack in terms of IDF1 (+0.9) and AssocA (+0.2) scores. Its performance is on par with our other variants, showing only a slight decrease compared to Ours-Detic (-1 mIDF1 and -1.2 AssocA). These results reaffirm the adaptability of our method across various backbone architectures and pre-training environments.

TAO TETA: Evaluating on TAO TETA (Table 12), Ours-R50, with its standard ResNet-50 backbone, continues to perform robustly. It closely matches the fully supervised TETer model, with only a slight decrease in AssocA (-1). This performance, consistent with our other variants, further validates the generalizability of our MASA approach across different backbones and pre-training methodologies.

| Method | mIDF1↑\uparrow | AssocA↑\uparrow | TETA↑\uparrow | mMOTSA↑\uparrow | mHOTA↑\uparrow |
|---|---|---|---|---|---|
| Fully-supervised, in-domain |  |  |  |  |  |
| UNINEXT-H [66]† | 48.5 | 53.2 | 53.6 | 35.7 | 40.6 |
| Self-supervised, zero-shot |  |  |  |  |  |
| Ours-Detic† | 49.5 | 53.5 | 54.4 | 36.4 | 40.2 |
| Ours-Grounding-DINO† | 48.6 | 52.3 | 54.0 | 36.1 | 40.0 |
| Ours-SAM-B† | 49.2 | 53.9 | 54.8 | 35.2 | 40.7 |
| Ours-SAM-H† | 49.7 | 54.5 | 54.7 | 35.8 | 40.8 |
| Ours-R50† | 48.7 | 53.6 | 54.7 | 35.2 | 40.4 |

| Method | mIDF1↑\uparrow | IDF1↑\uparrow | TETA↑\uparrow | AssocA↑\uparrow | mMOTA↑\uparrow |
|---|---|---|---|---|---|
| Fully-supervised, in-domain |  |  |  |  |  |
| ByteTrack [75]† | 54.8 | 70.4 | 55.7 | 51.5 | 45.5 |
| Self-supervised, zero-shot |  |  |  |  |  |
| Ours-Detic† | 55.8 | 71.3 | 54.4 | 52.9 | 44.6 |
| Ours-Grounding-DINO† | 55.6 | 71.7 | 54.5 | 52.7 | 44.5 |
| Ours-SAM-B† | 55.6 | 71.6 | 54.0 | 52.6 | 44.1 |
| Ours-SAM-H† | 55.3 | 71.7 | 54.2 | 51.9 | 44.5 |
| Ours-R50† | 54.8 | 71.3 | 54.0 | 51.7 | 44.2 |

| Method | AssocA | TETA | LocA | ClsA |
|---|---|---|---|---|
| Fully-supervised, in-domain |  |  |  |  |
| TETer [36]† | 36.7 | 34.6 | 52.1 | 15.0 |
| Self-supervised, zero-shot |  |  |  |  |
| Ours-Detic† | 36.4 | 34.7 | 51.9 | 15.8 |
| Ours-Grounding-DINO† | 37.6 | 34.9 | 51.8 | 15.4 |
| Ours-SAM-B† | 36.6 | 34.5 | 51.9 | 15.1 |
| Ours-SAM-H† | 36.4 | 34.5 | 51.8 | 15.4 |
| Ours-R50† | 35.7 | 34.1 | 52.1 | 15.0 |

### B Zero-shot Evaluation on YoutubeVIS

In this section, we evaluate our association method in a zero-shot setting on the Youtube-VIS 2019 [68] benchmark. To be specific, we test our MASA adapter with SAM-ViT-B as the base model directly on Youtube-VIS 2019 for association. Our method uses the same object detection observations as the state-of-the-art VIS method UNINEXT-R50 [66]. As shown in Table 13, our method achieves comparable performance with SOTA UNINEXT trained with the in-domain YoutubeVIS data, while outperforming all other approaches significantly. This outcome underlines the robust zero-shot association capabilities of our method, highlighting its effectiveness in scenarios without domain-specific training.

| Method | Zero-shot | Association Label | Video | VIS2019 val |  |
|---|---|---|---|---|---|
| AP\rm AP | AP75\rm AP_{75} |  |  |  |  |
| VisTR [56] | ✗ | ✓ | ✓ | 36.2 | 36.9 |
| MaskProp [4] | ✗ | ✓ | ✓ | 40.0 | 42.9 |
| IFC [28] | ✗ | ✓ | ✓ | 42.8 | 46.8 |
| SeqFormer [62] | ✗ | ✓ | ✓ | 47.4 | 51.8 |
| IDOL [63] | ✗ | ✓ | ✓ | 49.5 | 52.9 |
| MFVIS [32] | ✗ | ✓ | ✓ | 46.6 | 49.7 |
| VITA [27] | ✗ | ✓ | ✓ | 49.8 | 54.5 |
| UNINEXT-R50 [66]† | ✗ | ✓ | ✓ | 53.0 | 59.1 |
| Ours-SAM-B† | ✓ | ✗ | ✗ | 51.8 | 58.1 |

### C Visualization of Instance Embeddings

In Figure 6, we use t-SNE to visualize instance embeddings learned in different ways. We compare self-supervised approaches such as MoCo-v2 [11], VFS [64], and DINO [8], alongside two base models: SAM ViT-B [35], originally pre-trained on SA-1B for segmentation tasks, and IN-Sup R50 [25], initially pre-trained on ImageNet for image classification. Additionally, we present embeddings from fully supervised in-domain video models [36] and the same base models enhanced with our MASA adapters. In these visualizations, instances that share the same ground-truth ID are represented in the same colors. We use the BDD100K sequence as the data source.

Our observations indicate that the embeddings from the original SAM, IN-Sup R50, as well as the self-supervised methods like MoCo, VFS, and DINO, do not consistently separate different instances within certain complex scenarios, as highlighted by the instances marked in green, orange, and yellow. In contrast, by applying our MASA adapter to the original SAM ViT-B and IN-Sup R50 features, the resulting adapted embeddings exhibit a successful delineation of distinct instances. This performance is comparable to that of fully supervised methods that have been trained on labeled in-domain videos. Significantly, our method achieves these results without any labeled in-domain video data, demonstrating its considerable potential for robust instance-level correspondence learning.

### D Domain Gap and Adaptation

Except for previously mentioned applications, MASA can also serve as a useful domain adaption method for instance association. To be specific, due to the domain gaps such as object categories, scenarios, and lighting conditions, trackers trained on data of domain A may suffer from performance drop when evaluating on domain B. For example, compared with BDD [71], TAO [17] covers much more diverse scenarios and object categories. Thus, we choose BDD100K [71] as the source domain and TAO [17] as the target domain. Then we train two separate models with the same architecture as TETer [36] using labeled data of BDD and LVIS+ TAO [23, 17] respectively. These two models are represented by the blue and green bars in Figure 7. Please note that when evaluating their associating ability on TAO, they use the same object detection observations. As shown in Figure 7, directly applying embeddings trained on BDD to TAO (blue bar) leads to poor AssocA, which is 6.5% lower than the model trained on in-domain TAO (green bar). To alleviate this performance gap, we fine-tune the track head of the original TETer model represented by the blue bar with the MASA training pipeline, while freezing all other parameters (orange bar). Specifically, we only fine-tune the model using unlabeled images of LVIS and TAO, while not using any original TAO annotation. As shown in Figure 7, compared with the blue bar, the orange bar achieves an improvement of 3.9% on AssocA, reducing the domain gap by 60%. This demonstrates that MASA can effectively improve the association performance in out-of-domain scenarios, only requiring unlabeled images from the target domain.

### E Impact of Photometric Augmentation

In Section 4.3 of our main paper, we focused on various geometric augmentations, including random affine transformations, and large-scale jittering. We also use MixUp to enhance the instance diversity and simulate the occlusion effect. This section delves into the impact of additional photometric augmentation. We specifically examine the effects of motion blur, Gaussian noise, snow, fog, and brightness adjustments. Photometric augmentations are characterized by their ability to modify pixel values in an image. These alterations often mimic changes in environmental factors such as lighting and weather, impacting how scenes are captured by cameras. Unlike geometric augmentations that change the spatial arrangement of pixels through rotation, scaling, or cropping, photometric augmentations do not alter the structural integrity of objects within an image. Figure 8 illustrates these augmentations visually.

We maintained the same training regimen as our ablation study in the main paper, and using SAM-ViT-B as the foundational model for our experiments. Table 14 presents the results, indicating that the inclusion of photometric augmentation yields only modest improvements. We observed a marginal increase of +0.1 mIDF1 and +0.2 AssocA on the BDD dataset and +0.1 AssocA on the TAO dataset. Consequently, these augmentations are not included as a default in our methodology to achieve a better balance between performance improvement and the potential increase in computational complexity.

| Standard Aug | Strong Aug | Photometric Aug | BDD MOT | TAO |  |
|---|---|---|---|---|---|
| mIDF1 | AssocA | AssocA |  |  |  |
| ✓ |  |  | 48.2 | 43.9 | 28.5 |
| ✓ | ✓ |  | 54.9 | 51.9 | 35.8 |
| ✓ | ✓ | ✓ | 55 | 52.1 | 35.9 |

### F Comparison of Proposal Diversity

In our main paper, we assessed different proposal generation mechanisms within the context of association learning. Specifically, we focused on training using raw images from the BDD dataset. We experimented by replacing SAM in our MASA pipeline with Mask2former-SwinL, pre-trained on the COCO dataset (see [12]). As detailed in Table 9c of the main paper, the model utilizing SAM’s proposals demonstrated enhanced performance. This was evident both in in-domain tracking on the BDD dataset and in zero-shot tracking scenarios on the TAO dataset. Such findings highlight the crucial role of SAM’s dense and diverse object proposals in facilitating effective contrastive similarity learning.

Further, we present visual comparisons of the proposals generated by Mask2former and SAM in Figure 9. These comparative visualizations distinctly showcase the superior diversity in SAM’s proposals relative to those generated by Mask2former. SAM exhibits an enhanced ability to identify a wider array of instances within raw images, providing proposals with greater diversity. This diversity is pivotal in instance similarity learning and significantly contributes to the out-of-domain generalization capabilities of the learned instance representations.

### G Compare with Self-Supervised Methods

The task of extracting meaningful information from purely unlabeled images is notably challenging. UniTrack [58] has showcased the potential of self-supervised trained representations, such as MoCo [11] and VFS [64], in generalizing to various tracking tasks across different domains. However, as depicted in Figure 10, current self-supervised methods predominantly employ contrastive training with clean, object-centered images or videos. In particular, VFS trains on the Kinetics dataset, while MoCo and DINO utilize ImageNet.

However, these approaches primarily focus on frame-level similarities and fail to leverage instance information effectively. Consequently, they struggle to learn accurate instance representations in complex domains with multiple instances appearing together, demonstrating a notable weakness in extracting robust and generalized representations.

Visualization of Object-Centered Training Data We visualize the training data of VFS [64], the Kinetics dataset, and compare it with the driving videos from BDD100K in Figure 11. Kinetics, being an action recognition dataset, ensures the presence of instances throughout its videos by focusing on contained actions. Centred entities in Kinetics videos usually remain consistent over time, making VFS’s sampling strategy suitable for Kinetics. In contrast, BDD100K driving videos present a more dynamic and unpredictable environment. These videos frequently feature objects that enter and exit the frame, leading to a significant variation in the presence of instances across different frames. This characteristic of BDD100K poses a challenge as two frames sampled from the same video may not share the same instances, highlighting a fundamental difference in the nature of training data between the two datasets.

Training with Different Data Sources For a fair comparison, when comparing our method with other self-supervised counterparts in Table 6 of the main paper, we train all methods using the same raw training images (BDD and COCO), which are not object-centered and usually contain multiple instances in complex environments. In this section, we also present the tracking performance of those self-supervised methods using their original object-centered training data. As shown in the table below, the AssocA of MoCo trained on images from BDD and COCO remains relatively stable compared to its original version trained on ImageNet, with only a slight drop on the BDD MOT dataset. However, for VFS, training on images with multiple instances leads to a significant performance drop of 15.9 AssocA on BDD MOT and 12.7 AssocA on TAO, respectively. The reason is as follows: VFS considers frames from the same video as positive samples and frames from different videos as negative samples. This strategy is reasonable for Kinetics but not for BDD, as demonstrated in Figure 11. Specifically, centred entities in Kinetics videos usually do not change over time, but in BDD videos, objects frequently move in and out of frames. Two frames from the same BDD video may not contain the same instances at all. Lastly, in DINO’s training process, it forces representations of two augmented views from the same image to be similar without explicitly using negative samples. However, for images in BDD and COCO, two augmented views may contain many different instances, considering the complex scenes of these two datasets. This training strategy may cause the learned embeddings to be less discriminative.

Our approach, which leverages instance-level knowledge from the pre-trained SAM, moves beyond frame-level similarity to embrace a more nuanced instance-level similarity. The strong results obtained underscore the effectiveness of our proposed methods in learning robust representations for tracking purposes.

| Method | Video | BDD MOT | TAO | BDD MOTS |  |  |
|---|---|---|---|---|---|---|
| AssocA | mIDF1 | AssocA | AssocA | mIDF1 |  |  |
| Train on object-centred data |  |  |  |  |  |  |
| VFS | ✓ | 45.1 | 49.9 | 31.8 | 50.3 | 44.9 |
| MoCov2 | ✗ | 44.1 | 48.6 | 31.1 | 50.5 | 45.6 |
| DINO | ✗ | 41.7 | 46.5 | 26 | 46 | 40.5 |
| Train on BDD & COCO |  |  |  |  |  |  |
| VFS | ✓ | 29.2 | 35.0 | 19.1 | 30.7 | 30.1 |
| MoCov2 | ✗ | 42.7 | 46.7 | 30.7 | 51 | 45.3 |
| DINO | ✗ | 23.1 | 16.8 | 12.9 | 20.2 | 22.2 |
| Ours-SAM-B | ✗ | 51.9 | 54.9 | 35.8 | 53.7 | 49.1 |

### H Comparison with VOS-based Methods

The recent segmentation foundation model, SAM, has demonstrated exceptional ability in segmenting any object. However, simultaneously tracking all instances generated by SAM in videos remains a challenging task. Current methods typically employ SAM as a mask generator for the first frame of a video, then apply off-the-shelf video object segmentation (VOS) methods to propagate the initialized mask to subsequent frames [14, 15, 69, 13]. One notable method, Deva [14], utilizes XMem [13] for mask propagation to track multiple instances simultaneously. However, these methods encounter several key disadvantages.

Inadequate Mask Propagation Quality: Trained on relatively small-scale video segmentation datasets, these methods experience substantial domain gaps when tasked with tracking any object in any domain, resulting in inadequate mask propagation quality. Our main paper illustrates that our method significantly outperforms Deva [14] in zero-shot testing across various multiple object tracking benchmarks, especially in driving scenes, which are out-of-domain for both Deva and our method. We further provide a qualitative comparison in Figure 12. Additional video comparisons can be found in the provided video file. Testing Deva in the driving domain, which differs significantly from its training data, results in poor mask quality and accumulating errors over time. Moreover, there is no effective mechanism to handle the rapid entry and exit of objects in a scene, a common occurrence in real-world applications like autonomous driving. In contrast, our method exhibits stable performance in such scenarios.

Difficulty in Managing Multiple Granularities of Pixels: Furthermore, these methods are primarily developed for video object segmentation (VOS) tasks, which typically involve videos and annotations of single, rather than multiple, diverse objects. As a result, most VOS-based approaches are designed to track only one instance at a time. While recent advancements like those in [14, 69] allow for the simultaneous tracking of multiple instances, they often work on the premise that each pixel is part of a single instance. This overlooks complexities in pixel granularity, where a pixel may be part of multiple instances depending on the level of granularity—a common situation in the outputs of SAM, as depicted in Figure 13. This issue is further illustrated using the UVO dataset, which contains only coarse object-level annotations, often omitting finer details of object parts.

We apply SAM to generate mask predictions for each frame in the UVO dataset for both methods. To track objects segmented by SAM, a VOS-based method like Deva has to resolve overlaps by assigning each pixel to a unique instance. For example, if a group of pixels belongs to a part of an object, it must decide whether to track the part or the whole object. Assigning pixels to a part implies that the corresponding object is partially excluded, as shown with the cars in Figure 13. Conversely, assigning pixels to the object results in the removal of the part mask. We present the quantitative results of these scenarios on the UVO dataset in Table 16. Tracking parts leads to an incomplete representation of object masks on UVO, thus affecting performance negatively. In contrast, our method, capable of handling multiple granularities, tracks both entire objects and their parts without compromising performance on the UVO dataset.

. Track Method AR100 video Zero-shot test Deva-SAM-H: track all instances (with parts) [14] 19.4 Deva-SAM-H: track only whole objects (no parts) [14] 36.0 Ours-SAM-H: track all instances (with parts) 37.5

### I More Qualitative Results

We provide a video file containing our qualitative tracking results on multiple domains. Here we provide some visualization results regarding fast proposal generation and dense object association.

In Figure 14, we compare the segmentation quality of our fast proposal generation with SAM’s original everything mode on raw images from COCO validation set. By default, we output 300 bounding boxes per image, and use a bounding box NMS with 0.5 threshold as the only post-processing during inference. The results show that our fast proposal generation can achieve similar segmentation quality to the everything mode of SAM, despite using much less time.

We show qualitative results of open-vocabulary tracking in Figure 15. We observe that our method does well on tracking, and is able to generalize even to very exotic classes, such as minions. More results can be found in the provided video.

We provide qualitative results on our joint segmentation and tracking models. Since we learn proposal generation and association in a joint way, it makes our model capable of segmenting and associating anything in videos. Figure 16 shows the qualitative association performance using our self-generated proposals. We notice that although we can learn strong associations using MASA, it is very difficult to generate consistent proposals across frames. For example, we can see the missing segmentation for the building on the left in the second row. Those inconsistent detections will lead to severe flickering effects when visualising the results on videos. This indicates we still need further efforts on consistent proposal generation for robust detecting objects in videos.

### J Implementation Details

We provide more details regarding our model architecture, training, and inference.

MASA Adapter The MASA Adapter comprises two main parts. The first part involves the construction of a feature pyramid and dynamic feature fusion. The second part is the FasterRCNN-based detection head for the object prior to distillation and the track head for producing tracking features. The construction process for the feature pyramid varies depending on the backbone used. These variations are detailed in the respective sections for each model. The dynamic feature fusion employs standard deformable convolution, as outlined in [80], to aggregate information across spatial locations and feature levels. Additionally, task-aware attention and scale-aware attention from [16] are utilized for SAM-based models. In total, three fusion blocks are established for the feature fusion process.

The FasterRCNN-based detection head includes a region proposal network and a class-agnostic box regression head. The track head comprises four convolutional layers and one fully connected layer, used to generate instance embeddings.

Ours-Detic We utilize the pre-trained Detic [78] model with Swin-B [42] as the backbone. The pre-trained model adheres to the open-vocabulary object detection setup described in [22], where rare classes from LVIS are excluded from training. We freeze the Detic Swin-B backbone and employ the standard FPN for constructing the feature pyramid. Specifically, we extract features from the 4t​h4^{th}, 22n​d22^{nd}, and 24t​h24^{th} blocks of the Swin-B backbone. Subsequently, we integrate the dynamic feature fusion atop the feature pyramid to learn tracking features through detection distillation and instance contrastive learning.

Ours-Grounding-DINO We employ the pre-trained Grounding-DINO [40] model with Swin-B [42] as the backbone. The Swin-B backbone is frozen, and we use the standard FPN to construct the feature pyramid. Apart from the differing pre-training and window sizes for the Swin backbone, all learnable components are identical to Ours-Detic.

Ours-SAM-B This model is based on SAM, with all original SAM components frozen. To obtain multi-level hierarchical features from the plain ViT backbone of SAM, we extract feature maps from the outputs of the 3r​d3^{rd}, 6t​h6^{th}, 9t​h9^{th}, and 12t​h12^{th} blocks. Transposed Convolutions are used to upscale the feature map from the 3r​d3^{rd} block by 4×4\times and from the 6t​h6^{th} block by 2×2\times. We maintain the 9t​h9^{th} feature map as is, and downscale the feature map from the 12t​h12^{th} block by 1/21/2 using MaxPooling. This approach yields hierarchical features with scale ratios of 14,18,116,132{\frac{1}{4},\frac{1}{8},\frac{1}{16},\frac{1}{32}}. The remainder of the model mirrors the two models mentioned above.

Ours-SAM-H The learnable portion is largely similar to Ours-SAM-B. The sole distinction is that we extract features from the outputs of the 8t​h8^{th}, 16t​h16^{th}, 24t​h24^{th}, and 32n​d32^{nd} blocks to construct the feature pyramid.

For SAM-based models, we turn off MixUp augmentation in the last two epochs. After that, we finetune the track heads of the SAM-based models while freezing the other parts with all augmentations for 6 epochs.

For training our model with any raw image collection, the following pipeline is utilized. Initially, the ’everything’ mode of SAM is employed to generate training data on raw images offline, using the SAM-ViTH model to ensure higher quality. We adhere to the default SAM settings, which involve using 32 sampling points along each side of an image. Additionally, an Intersection over Union (IoU) prediction threshold of 0.88 is applied to filter out low-quality predictions. Subsequently, small disconnected regions and holes in masks are removed. Bounding box Non-Maximum Suppression (NMS) is also used to eliminate overlapping predictions with a threshold of 0.7. In our ablation studies, this pipeline is applied to generate data on raw COCO and BDD100K images.

Notably, during testing on UVO, in addition to using proposals generated by our fast-segmenting everything mode, we also incorporate the same per-frame mask observation as employed in [14]. This inclusion aims to minimize the temporal inconsistency in SAM’s mask predictions on videos.

Overall, our inference scheme is illustrated in Algorithm 1. In terms of similarity computation, we provide the formula that we use:

|  | s1​(τ,r)=12​[exp​(qr⋅qτ)∑r′∈Pexp​(qr′⋅qτ)+exp​(qr⋅qτ)∑τ′∈𝒯exp​(qr⋅qτ′)]s2​(τ,r)=qr⋅qτ‖qr‖​‖qτ‖s⁡(τ,r)=12​(s1​(τ,r)+s2​(τ,r))\begin{gathered}\textbf{$s_{1}$}(\tau,r)=\frac{1}{2}\left[\frac{\text{exp}(\textbf{q}_{r}\cdot\textbf{q}_{\tau})}{\sum_{r^{\prime}\in P}\text{exp}(\textbf{q}_{r^{\prime}}\cdot\textbf{q}_{\tau})}+\frac{\text{exp}(\textbf{q}_{r}\cdot\textbf{q}_{\tau})}{\sum_{\tau^{\prime}\in\mathcal{T}}\text{exp}(\textbf{q}_{r}\cdot\textbf{q}_{\tau^{\prime}})}\right]\\ \textbf{$s_{2}$}(\tau,r)=\frac{\textbf{q}_{r}\cdot\textbf{q}_{\tau}}{\|\textbf{q}_{r}\|\|\textbf{q}_{\tau}\|}\\ s(\tau,r)=\frac{1}{2}(s_{1}(\tau,r)+s_{2}(\tau,r))\end{gathered} |  | (2) |
|---|---|---|---|

where s​(τ,r)\textbf{s}(\tau,r) represents the similarity score between a track τ\tau and an object candidate rr. Here, qr\textbf{q}_{r} denotes the detection embedding of the object candidate rr, encapsulating its appearance features, while qτ\textbf{q}_{\tau} represents the track embedding for track τ\tau, capturing the features of the tracked object. The s1​(τ,r)\textbf{$s_{1}$}(\tau,r) employs an exponential function to compute the dot product of these embeddings, reflecting the degree of similarity between the object candidate and the track. This similarity score is normalized twice: firstly, across all object candidates r′r^{\prime} in the set PP for a given track τ\tau, and secondly, across all tracks τ′\tau^{\prime} in the set 𝒯\mathcal{T} for a given object candidate rr. This dual normalization ensures a balanced and comprehensive assessment of similarity, facilitating accurate object association in dynamic video sequences. s2​(τ,r)\textbf{$s_{2}$}(\tau,r) computes the cosine similarity. The final s⁡(τ,r)s(\tau,r) score is the average between s1​(τ,r)\textbf{$s_{1}$}(\tau,r) and s2​(τ,r)\textbf{$s_{2}$}(\tau,r).

### K Limitations

One key limitation of our approach is handling temporal inconsistencies in detection or segmentation results across video frames. This issue, common in open-world object detection and segmentation models like SAM, is evident when an object detected in one frame is missed in the next, causing flickering effects in video visualization, as seen in our demonstrations. While our MASA adapter excels in learning associations, it cannot rectify foundational models’ detection or segmentation errors. The challenge of generating consistent proposals across frames highlights an important area for future research to enhance the robustness and stability of object detection in dynamic video environments.

Another limitation is the lack of a long-term memory system, which is crucial for handling occlusions. We only learn a universal instance appearance model, which can be used directly by different detectors. However, the tracking strategy and memory management are still done by the bi-softmax matching and a queue to store each instance’s appearance embeddings. This simple strategy is prone to failure in cases of severe occlusions. Developing a more sophisticated long-term memory system and improved tracking strategies will be essential to address this limitation.

| tt | t+1t+1 | t+2t+2 | t+3t+3 | t+4t+4 |
|---|---|---|---|---|
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
