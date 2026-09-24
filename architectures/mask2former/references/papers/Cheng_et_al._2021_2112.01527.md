# Masked-attention Mask Transformer for Universal Image Segmentation

Image segmentation groups pixels with different semantics, e.g., category or instance membership. Each choice of semantics defines a task. While only the semantics of each task differ, current research focuses on designing specialized architectures for each task. We present Masked-attention Mask Transformer (Mask2Former), a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic). Its key components include masked attention, which extracts localized features by constraining cross-attention within predicted mask regions. In addition to reducing the research effort by at least three times, it outperforms the best specialized architectures by a significant margin on four popular datasets. Most notably, Mask2Former sets a new state-of-the-art for panoptic segmentation (57.8 PQ on COCO), instance segmentation (50.1 AP on COCO) and semantic segmentation (57.7 mIoU on ADE20K).

## 1 Introduction

Image segmentation studies the problem of grouping pixels. Different semantics for grouping pixels, e.g., category or instance membership, have led to different types of segmentation tasks, such as panoptic, instance or semantic segmentation. While these tasks differ only in semantics, current methods develop specialized architectures for each task. Per-pixel classification architectures based on Fully Convolutional Networks (FCNs) [37] are used for semantic segmentation, while mask classification architectures [24, 5] that predict a set of binary masks each associated with a single category, dominate instance-level segmentation. Although such specialized architectures [37, 10, 24, 6] have advanced each individual task, they lack the flexibility to generalize to the other tasks. For example, FCN-based architectures struggle at instance segmentation, leading to the evolution of different architectures for instance segmentation compared to semantic segmentation. Thus, duplicate research and (hardware) optimization effort is spent on each specialized architecture for every task.

To address this fragmentation, recent work [14, 62] has attempted to design universal architectures, that are capable of addressing all segmentation tasks with the same architecture (i.e., universal image segmentation). These architectures are typically based on an end-to-end set prediction objective (e.g., DETR [5]), and successfully tackle multiple tasks without modifying the architecture, loss, or the training procedure. Note, universal architectures are still trained separately for different tasks and datasets, albeit having the same architecture. In addition to being flexible, universal architectures have recently shown state-of-the-art results on semantic and panoptic segmentation [14]. However, recent work still focuses on advancing specialized architectures [39, 45, 20], which raises the question: why haven’t universal architectures replaced specialized ones?

Although existing universal architectures are flexible enough to tackle any segmentation task, as shown in Figure 1, in practice their performance lags behind the best specialized architectures. For instance, the best reported performance of universal architectures [14, 62], is currently lower (>9>9 AP) than the SOTA specialized architecture for instance segmentation [6]. Beyond the inferior performance, universal architectures are also harder to train. They typically require more advanced hardware and a much longer training schedule. For example, training MaskFormer [14] takes 300 epochs to reach 40.1 AP and it can only fit a single image in a GPU with 32G memory. In contrast, the specialized Swin-HTC++ [6] obtains better performance in only 72 epochs. Both the performance and training efficiency issues hamper the deployment of universal architectures.

In this work, we propose a universal image segmentation architecture named Masked-attention Mask Transformer (Mask2Former) that outperforms specialized architectures across different segmentation tasks, while still being easy to train on every task. We build upon a simple meta architecture [14] consisting of a backbone feature extractor [25, 36], a pixel decoder [33] and a Transformer decoder [51]. We propose key improvements that enable better results and efficient training. First, we use masked attention in the Transformer decoder which restricts the attention to localized features centered around predicted segments, which can be either objects or regions depending on the specific semantic for grouping. Compared to the cross-attention used in a standard Transformer decoder which attends to all locations in an image, our masked attention leads to faster convergence and improved performance. Second, we use multi-scale high-resolution features which help the model to segment small objects/regions. Third, we propose optimization improvements such as switching the order of self and cross-attention, making query features learnable, and removing dropout; all of which improve performance without additional compute. Finally, we save 3×3\times training memory without affecting the performance by calculating mask loss on few randomly sampled points. These improvements not only boost the model performance, but also make training significantly easier, making universal architectures more accessible to users with limited compute.

We evaluate Mask2Former on three image segmentation tasks (panoptic, instance and semantic segmentation) using four popular datasets (COCO [35], Cityscapes [16], ADE20K [65] and Mapillary Vistas [42]). For the first time, on all these benchmarks, our single architecture performs on par or better than specialized architectures. Mask2Former sets the new state-of-the-art of 57.8 PQ on COCO panoptic segmentation [28], 50.1 AP on COCO instance segmentation [35] and 57.7 mIoU on ADE20K semantic segmentation [65] using the exact same architecture.

## 2 Related Work

Specialized semantic segmentation architectures typically treat the task as a per-pixel classification problem. FCN-based architectures [37] independently predict a category label for every pixel. Follow-up methods find context to play an important role for precise per-pixel classification and focus on designing customized context modules [7, 8, 63] or self-attention variants [55, 21, 61, 26, 64, 45].

Specialized instance segmentation architectures are typically based upon “mask classification.” They predict a set of binary masks each associated with a single class label. The pioneering work, Mask R-CNN [24], generates masks from detected bounding boxes. Follow-up methods either focus on detecting more precise bounding boxes [4, 6], or finding new ways to generate a dynamic number of masks, e.g., using dynamic kernels [49, 56, 3] or clustering algorithms [29, 11]. Although the performance has been advanced in each task, these specialized innovations lack the flexibility to generalize from one to the other, leading to duplicated research effort. For instance, although multiple approaches have been proposed for building feature pyramid representations [33], as we show in our experiments, BiFPN [47] performs better for instance segmentation while FaPN [39] performs better for semantic segmentation.

Panoptic segmentation has been proposed to unify both semantic and instance segmentation tasks [28]. Architectures for panoptic segmentation either combine the best of specialized semantic and instance segmentation architectures into a single framework [60, 27, 11, 31] or design novel objectives that equally treat semantic regions and instance objects [5, 52]. Despite those new architectures, researchers continue to develop specialized architectures for different image segmentation tasks [45, 20]. We find panoptic architectures usually only report performance on a single panoptic segmentation task [52], which does not guarantee good performance on other tasks (Figure 1). For example, panoptic segmentation does not measure architectures’ abilities to rank predictions as instance segmentations. Thus, we refrain from referring to architectures that are only evaluated for panoptic segmentation as universal architectures. Instead, here, we evaluate our Mask2Former on all studied tasks to guarantee generalizability.

Universal architectures have emerged with DETR [5] and show that mask classification architectures with an end-to-end set prediction objective are general enough for any image segmentation task. MaskFormer [14] shows that mask classification based on DETR not only performs well on panoptic segmentation but also achieves state-of-the-art on semantic segmentation. K-Net [62] further extends set prediction to instance segmentation. Unfortunately, these architectures fail to replace specialized models as their performance on particular tasks or datasets is still worse than the best specialized architecture (e.g., MaskFormer [14] cannot segment instances well). To our knowledge, Mask2Former is the first architecture that outperforms state-of-the-art specialized architectures on all considered tasks and datasets.

## 3 Masked-attention Mask Transformer

We now present Mask2Former. We first review a meta architecture for mask classification that Mask2Former is built upon. Then, we introduce our new Transformer decoder with masked attention which is the key to better convergence and results. Lastly, we propose training improvements that make Mask2Former efficient and accessible.

### 3.1 Mask classification preliminaries

Mask classification architectures group pixels into NN segments by predicting NN binary masks, along with NN corresponding category labels. Mask classification is sufficiently general to address any segmentation task by assigning different semantics, e.g., categories or instances, to different segments. However, the challenge is to find good representations for each segment. For example, Mask R-CNN [24] uses bounding boxes as the representation which limits its application to semantic segmentation. Inspired by DETR [5], each segment in an image can be represented as a CC-dimensional feature vector (“object query”) and can be processed by a Transformer decoder, trained with a set prediction objective. A simple meta architecture would consist of three components. A backbone that extracts low-resolution features from an image. A pixel decoder that gradually upsamples low-resolution features from the output of the backbone to generate high-resolution per-pixel embeddings. And finally a Transformer decoder that operates on image features to process object queries. The final binary mask predictions are decoded from per-pixel embeddings with object queries. One successful instantiation of such a meta architecture is MaskFormer [14], and we refer readers to [14] for more details.

### 3.2 Transformer decoder with masked attention

Mask2Former adopts the aforementioned meta architecture, with our proposed Transformer decoder (Figure 2 right) replacing the standard one. The key components of our Transformer decoder include a masked attention operator, which extracts localized features by constraining cross-attention to within the foreground region of the predicted mask for each query, instead of attending to the full feature map. To handle small objects, we propose an efficient multi-scale strategy to utilize high-resolution features. It feeds successive feature maps from the pixel decoder’s feature pyramid into successive Transformer decoder layers in a round robin fashion. Finally, we incorporate optimization improvements that boost model performance without introducing additional computation. We now discuss these improvements in detail.

Context features have been shown to be important for image segmentation [7, 8, 63]. However, recent studies [46, 22] suggest that the slow convergence of Transformer-based models is due to global context in the cross-attention layer, as it takes many training epochs for cross-attention to learn to attend to localized object regions [46]. We hypothesize that local features are enough to update query features and context information can be gathered through self-attention. For this we propose masked attention, a variant of cross-attention that only attends within the foreground region of the predicted mask for each query.

Standard cross-attention (with residual path) computes

|  | 𝐗l=softmax​(𝐐l​𝐊lT)​𝐕l+𝐗l−1.\displaystyle\mathbf{X}_{l}=\text{softmax}(\mathbf{Q}_{l}\mathbf{K}^{\text{T}}_{l})\mathbf{V}_{l}+\mathbf{X}_{l-1}. |  | (1) |
|---|---|---|---|

Here, ll is the layer index, 𝐗l∈ℝN×C\mathbf{X}_{l}\in\mathbb{R}^{N\times C} refers to NN CC-dimensional query features at the lthl^{\text{th}} layer and 𝐐l=fQ​(𝐗l−1)∈ℝN×C\mathbf{Q}_{l}=f_{Q}(\mathbf{X}_{l-1})\in\mathbb{R}^{N\times C}. 𝐗0\mathbf{X}_{0} denotes input query features to the Transformer decoder. 𝐊l,𝐕l∈ℝHl​Wl×C\mathbf{K}_{l},\mathbf{V}_{l}\in\mathbb{R}^{H_{l}W_{l}\times C} are the image features under transformation fK​(⋅)f_{K}(\cdot) and fV​(⋅)f_{V}(\cdot) respectively, and HlH_{l} and WlW_{l} are the spatial resolution of image features that we will introduce next in Section 3.2.2. fQf_{Q}, fKf_{K} and fVf_{V} are linear transformations.

Our masked attention modulates the attention matrix via

|  | 𝐗l=softmax​(ℳ↕−∞+𝒬↕​𝒦↕T)​𝒱↕+𝒳↕−∞.\displaystyle\mathbf{X}_{l}=\text{softmax}(\mathbfcal{M}_{l-1}+\mathbf{Q}_{l}\mathbf{K}^{\text{T}}_{l})\mathbf{V}_{l}+\mathbf{X}_{l-1}. |  | (2) |
|---|---|---|---|

Moreover, the attention mask ℳ↕−∞\mathbfcal{M}_{l-1} at feature location (x,y)(x,y) is

|  | ℳ↕−∞​(§,†)={′if ​ℳ↕−∞​(§,†)=∞−∞otherwise.\displaystyle\mathbfcal{M}_{l-1}(x,y)=\left\{\begin{array}[]{ll}0&\text{if~}\mathbf{M}_{l-1}(x,y)=1\\ -\infty&\text{otherwise}\end{array}\right.. |  |
|---|---|---|

Here, 𝐌l−1∈{0,1}N×Hl​Wl\mathbf{M}_{l-1}\in\{0,1\}^{N\times H_{l}W_{l}} is the binarized output (thresholded at 0.50.5) of the resized mask prediction of the previous (l−1l-1)-th Transformer decoder layer. It is resized to the same resolution of 𝐊l\mathbf{K}_{l}. 𝐌0\mathbf{M}_{0} is the binary mask prediction obtained from 𝐗0\mathbf{X}_{0}, i.e., before feeding query features into the Transformer decoder.

High-resolution features improve model performance, especially for small objects [5]. However, this is computationally demanding. Thus, we propose an efficient multi-scale strategy to introduce high-resolution features while controlling the increase in computation. Instead of always using the high-resolution feature map, we utilize a feature pyramid which consists of both low- and high-resolution features and feed one resolution of the multi-scale feature to one Transformer decoder layer at a time.

Specifically, we use the feature pyramid produced by the pixel decoder with resolution 1/321/32, 1/161/16 and 1/81/8 of the original image. For each resolution, we add both a sinusoidal positional embedding epos∈ℝHl​Wl×Ce_{\text{pos}}\in\mathbb{R}^{H_{l}W_{l}\times C}, following [5], and a learnable scale-level embedding elvl∈ℝ1×Ce_{\text{lvl}}\in\mathbb{R}^{1\times C}, following [66]. We use those, from lowest-resolution to highest-resolution for the corresponding Transformer decoder layer as shown in Figure 2 left. We repeat this 3-layer Transformer decoder LL times. Our final Transformer decoder hence has 3​L3L layers. More specifically, the first three layers receive a feature map of resolution H1=H/32H_{1}=H/32, H2=H/16H_{2}=H/16, H3=H/8H_{3}=H/8 and W1=W/32W_{1}=W/32, W2=W/16W_{2}=W/16, W3=W/8W_{3}=W/8, where HH and WW are the original image resolution. This pattern is repeated in a round robin fashion for all following layers.

A standard Transformer decoder layer [51] consists of three modules to process query features in the following order: a self-attention module, a cross-attention and a feed-forward network (FFN). Moreover, query features (𝐗0\mathbf{X}_{0}) are zero initialized before being fed into the Transformer decoder and are associated with learnable positional embeddings. Furthermore, dropout is applied to both residual connections and attention maps.

To optimize the Transformer decoder design, we make the following three improvements. First, we switch the order of self- and cross-attention (our new “masked attention”) to make computation more effective: query features to the first self-attention layer are image-independent and do not have signals from the image, thus applying self-attention is unlikely to enrich information. Second, we make query features (𝐗0\mathbf{X}_{0}) learnable as well (we still keep the learnable query positional embeddings), and learnable query features are directly supervised before being used in the Transformer decoder to predict masks (𝐌0\mathbf{M}_{0}). We find these learnable query features function like a region proposal network [43] and have the ability to generate mask proposals. Finally, we find dropout is not necessary and usually decreases performance. We thus completely remove dropout in our decoder.

### 3.3 Improving training efficiency

One limitation of training universal architectures is the large memory consumption due to high-resolution mask prediction, making them less accessible than the more memory-friendly specialized architectures [24, 6]. For example, MaskFormer [14] can only fit a single image in a GPU with 32G memory. Motivated by PointRend [30] and Implicit PointRend [13], which show a segmentation model can be trained with its mask loss calculated on KK randomly sampled points instead of the whole mask, we calculate the mask loss with sampled points in both the matching and the final loss calculation. More specifically, in the matching loss that constructs the cost matrix for bipartite matching, we uniformly sample the same set of KK points for all prediction and ground truth masks. In the final loss between predictions and their matched ground truths, we sample different sets of KK points for different pairs of prediction and ground truth using importance sampling [30]. We set K=12544K=12544, i.e., 112×112112\times 112 points. This new training strategy effectively reduces training memory by 3×3\times, from 18GB to 6GB per image, making Mask2Former more accessible to users with limited computational resources.

## 4 Experiments

We demonstrate Mask2Former is an effective architecture for universal image segmentation through comparisons with specialized state-of-the-art architectures on standard benchmarks. We evaluate our proposed design decisions through ablations on all three tasks. Finally we show Mask2Former generalizes beyond the standard benchmarks, obtaining state-of-the-art results on four datasets.

| method | backbone | query type | epochs | PQ | PQTh{}^{\text{Th}} | PQSt{}^{\text{St}} | APpanTh{}^{\text{Th}}_{\text{pan}} | mIoUpan{}_{\text{pan}} | #params. | FLOPs | fps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DETR [5] | R50 | 100 queries | 500+25 | 43.4 | 48.2 | 36.3 | 31.1 | - | - | - | - |
| MaskFormer [14] | R50 | 100 queries | 300 | 46.5 | 51.0 | 39.8 | 33.0 | 57.8 | 045M | 0181G | 17.6 |
| Mask2Former (ours) | R50 | 100 queries | 50 | 51.9 | 57.7 | 43.0 | 41.7 | 61.7 | 044M | 0226G | 08.6 |
| DETR [5] | R101 | 100 queries | 500+25 | 45.1 | 50.5 | 37.0 | 33.0 | - | - | - | - |
| MaskFormer [14] | R101 | 100 queries | 300 | 47.6 | 52.5 | 40.3 | 34.1 | 59.3 | 064M | 0248G | 14.0 |
| Mask2Former (ours) | R101 | 100 queries | 50 | 52.6 | 58.5 | 43.7 | 42.6 | 62.4 | 063M | 0293G | 07.2 |
| Max-DeepLab [52] | Max-L | 128 queries | 216 | 51.1 | 57.0 | 42.2 | - | - | 451M | 3692G | - |
| MaskFormer [14] | Swin-L†{}^{\text{\textdagger}} | 100 queries | 300 | 52.7 | 58.5 | 44.0 | 40.1 | 64.8 | 212M | 0792G | 05.2 |
| K-Net [62] | Swin-L†{}^{\text{\textdagger}} | 100 queries | 36 | 54.6 | 60.2 | 46.0 | - | - | - | - | - |
| Mask2Former (ours) | Swin-L†{}^{\text{\textdagger}} | 200 queries | 100 | 57.8 | 64.2 | 48.1 | 48.6 | 67.4 | 216M | 0868G | 04.0 |

Datasets. We study Mask2Former using four widely used image segmentation datasets that support semantic, instance and panoptic segmentation: COCO [35] (80 “things” and 53 “stuff” categories), ADE20K [65] (100 “things” and 50 “stuff” categories), Cityscapes [16] (8 “things” and 11 “stuff” categories) and Mapillary Vistas [42] (37 “things” and 28 “stuff” categories). Panoptic and semantic segmentation tasks are evaluated on the union of “things” and “stuff” categories while instance segmentation is only evaluated on the “things” categories.

Evaluation metrics. For panoptic segmentation, we use the standard PQ (panoptic quality) metric [28]. We further report APpanTh{}^{\text{Th}}_{\text{pan}}, which is the AP evaluated on the “thing” categories using instance segmentation annotations, and mIoUpan{}_{\text{pan}}, which is the mIoU for semantic segmentation by merging instance masks from the same category, of the same model trained only with panoptic segmentation annotations. For instance segmentation, we use the standard AP (average precision) metric [35]. For semantic segmentation, we use mIoU (mean Intersection-over-Union) [19].

| method | backbone | query type | epochs | AP | APS{}^{\text{S}} | APM{}^{\text{M}} | APL{}^{\text{L}} | APboundary{}^{\text{boundary}} | #params. | FLOPs | fps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MaskFormer [14] | R50 | 100 queries | 300 | 34.0 | 16.4 | 37.8 | 54.2 | 23.0 | 045M | 0181G | 19.2 |
| Mask R-CNN [24] | R50 | dense anchors | 36 | 37.2 | 18.6 | 39.5 | 53.3 | 23.1 | 044M | 0201G | 15.2 |
| Mask R-CNN [24, 23, 18] | R50 | dense anchors | 400 | 42.5 | 23.8 | 45.0 | 60.0 | 28.0 | 046M | 0358G | 10.3 |
| Mask2Former (ours) | R50 | 100 queries | 50 | 43.7 | 23.4 | 47.2 | 64.8 | 30.6 | 044M | 0226G | 09.7 |
| Mask R-CNN [24] | R101 | dense anchors | 36 | 38.6 | 19.5 | 41.3 | 55.3 | 24.5 | 063M | 0266G | 10.8 |
| Mask R-CNN [24, 23, 18] | R101 | dense anchors | 400 | 43.7 | 24.6 | 46.4 | 61.8 | 29.1 | 065M | 0423G | 08.6 |
| Mask2Former (ours) | R101 | 100 queries | 50 | 44.2 | 23.8 | 47.7 | 66.7 | 31.1 | 063M | 0293G | 07.8 |
| QueryInst [20] | Swin-L†{}^{\text{\textdagger}} | 300 queries | 50 | 48.9 | 30.8 | 52.6 | 68.3 | 33.5 | - | - | 03.3 |
| Swin-HTC++ [36, 6] | Swin-L†{}^{\text{\textdagger}} | dense anchors | 72 | 49.5 | 31.0 | 52.4 | 67.2 | 34.1 | 284M | 1470G | - |
| Mask2Former (ours) | Swin-L†{}^{\text{\textdagger}} | 200 queries | 100 | 50.1 | 29.9 | 53.9 | 72.1 | 36.2 | 216M | 0868G | 04.0 |

### 4.1 Implementation details

We adopt settings from [14] with the following differences:

Pixel decoder. Mask2Former is compatible with any existing pixel decoder module. In MaskFormer [14], FPN [33] is chosen as the default for its simplicity. Since our goal is to demonstrate strong performance across different segmentation tasks, we use the more advanced multi-scale deformable attention Transformer (MSDeformAttn) [66] as our default pixel decoder. Specifically, we use 6 MSDeformAttn layers applied to feature maps with resolution 1/81/8, 1/161/16 and 1/321/32, and use a simple upsampling layer with lateral connection on the final 1/81/8 feature map to generate the feature map of resolution 1/41/4 as the per-pixel embedding. In our ablation study, we show that this pixel decoder provides best results across different segmentation tasks.

Transformer decoder. We use our Transformer decoder proposed in Section 3.2 with L=3L=3 (i.e., 9 layers total) and 100 queries by default. An auxiliary loss is added to every intermediate Transformer decoder layer and to the learnable query features before the Transformer decoder.

Loss weights. We use the binary cross-entropy loss (instead of focal loss [34] in [14]) and the dice loss [41] for our mask loss: ℒmask=λce​ℒce+λdice​ℒdice\mathcal{L}_{\text{mask}}=\lambda_{\text{ce}}\mathcal{L}_{\text{ce}}+\lambda_{\text{dice}}\mathcal{L}_{\text{dice}}. We set λce=5.0\lambda_{\text{ce}}=5.0 and λdice=5.0\lambda_{\text{dice}}=5.0. The final loss is a combination of mask loss and classification loss: ℒmask+λcls​ℒcls\mathcal{L}_{\text{mask}}+\lambda_{\text{cls}}\mathcal{L}_{\text{cls}} and we set λcls=2.0\lambda_{\text{cls}}=2.0 for predictions matched with a ground truth and 0.10.1 for the “no object,” i.e., predictions that have not been matched with any ground truth.

Post-processing. We use the exact same post-processing as [14] to acquire the expected output format for panoptic and semantic segmentation from pairs of binary masks and class predictions. Instance segmentation requires additional confidence scores for each prediction. We multiply class confidence and mask confidence (i.e., averaged foreground per-pixel binary mask probability) for a final confidence.

### 4.2 Training settings

Panoptic and instance segmentation. We use Detectron2 [57] and follow the updated Mask R-CNN [24] baseline settings11 1 https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#new-baselines-using-large-scale-jitter-and-longer-training-schedule for the COCO dataset. More specifically, we use AdamW [38] optimizer and the step learning rate schedule. We use an initial learning rate of 0.00010.0001 and a weight decay of 0.050.05 for all backbones. A learning rate multiplier of 0.10.1 is applied to the backbone and we decay the learning rate at 0.9 and 0.95 fractions of the total number of training steps by a factor of 10. If not stated otherwise, we train our models for 50 epochs with a batch size of 16. For data augmentation, we use the large-scale jittering (LSJ) augmentation [23, 18] with a random scale sampled from range 0.1 to 2.0 followed by a fixed size crop to 1024×10241024\times 1024. We use the standard Mask R-CNN inference setting where we resize an image with shorter side to 800 and longer side up-to 1333. We also report FLOPs and fps. FLOPs are averaged over 100 validation images (COCO images have varying sizes). Frames-per-second (fps) is measured on a V100 GPU with a batch size of 1 by taking the average runtime on the entire validation set including post-processing time.

Semantic segmentation. We follow the same settings as [14] to train our models, except: 1) a learning rate multiplier of 0.1 is applied to both CNN and Transformer backbones instead of only applying it to CNN backbones in [14], 2) both ResNet and Swin backbones use an initial learning rate of 0.00010.0001 and a weight decay of 0.050.05, instead of using different learning rates in [14].

### 4.3 Main results

Panoptic segmentation. We compare Mask2Former with state-of-the-art models for panoptic segmentation on the COCO panoptic [28] dataset in Table 1. Mask2Former consistently outperforms MaskFormer by more than 5 PQ across different backbones while converging 6×6\times faster. With Swin-L backbone, our Mask2Former sets a new state-of-the-art of 57.8 PQ, outperforming existing state-of-the-art [14] by 5.1 PQ and concurrent work, K-Net [62], by 3.2 PQ. Mask2Former even outperforms the best ensemble models with extra training data in the COCO challenge (see Appendix A.1 for test set results).

Beyond the PQ metric, our Mask2Former also achieves higher performance on two other metrics compared to DETR [5] and MaskFormer: APpanTh{}^{\text{Th}}_{\text{pan}}, which is the AP evaluated on the 80 “thing” categories using instance segmentation annotation, and mIoUpan{}_{\text{pan}}, which is the mIoU evaluated on the 133 categories for semantic segmentation converted from panoptic segmentation annotation. This shows Mask2Former’s universality: trained only with panoptic segmentation annotations, it can be used for instance and semantic segmentation.

| method | backbone | crop size | mIoU (s.s.) | mIoU (m.s.) |
|---|---|---|---|---|
| MaskFormer [14] | R50 | 512512 | 44.5 | 46.7 |
| Mask2Former (ours) | R50 | 512512 | 47.2 | 49.2 |
| Swin-UperNet [36, 58] | Swin-T†{}^{\text{\textdagger}} | 512512 | - | 46.1 |
| MaskFormer [14] | Swin-T | 512512 | 46.7 | 48.8 |
| Mask2Former (ours) | Swin-T | 512512 | 47.7 | 49.6 |
| MaskFormer [14] | Swin-L†{}^{\text{\textdagger}} | 640640 | 54.1 | 55.6 |
| FaPN-MaskFormer [39, 14] | Swin-L-FaPN†{}^{\text{\textdagger}} | 640640 | 55.2 | 56.7 |
| BEiT-UperNet [2, 58] | BEiT-L†{}^{\text{\textdagger}} | 640640 | - | 57.0 |
| Mask2Former (ours) | Swin-L†{}^{\text{\textdagger}} | 640640 | 56.1 | 57.3 |
| Swin-L-FaPN†{}^{\text{\textdagger}} | 640640 | 56.4 | 57.7 |  |

|  | AP | PQ | mIoU | FLOPs |
|---|---|---|---|---|
| Mask2Former (ours) | 43.7 | 51.9 | 47.2 | 226G |
| −- masked attention | 37.8 (-5.9) | 47.1 (-4.8) | 45.5 (-1.7) | 213G |
| −- high-resolution features | 41.5 (-2.2) | 50.2 (-1.7) | 46.1 (-1.1) | 218G |

|  | AP | PQ | mIoU | FLOPs |
|---|---|---|---|---|
| Mask2Former (ours) | 43.7 | 51.9 | 47.2 | 226G |
| −- learnable query features | 42.9 (-0.8) | 51.2 (-0.7) | 45.4 (-1.8) | 226G |
| −- cross-attention first | 43.2 (-0.5) | 51.6 (-0.3) | 46.3 (-0.9) | 226G |
| −- remove dropout | 43.0 (-0.7) | 51.3 (-0.6) | 47.2 (-0.0) | 226G |
| −- all 3 components above | 42.3 (-1.4) | 50.8 (-1.1) | 46.3 (-0.9) | 226G |

|  | AP | PQ | mIoU | FLOPs |
|---|---|---|---|---|
| cross-attention | 37.8 | 47.1 | 45.5 | 213G |
| SMCA [22] | 37.9 | 47.2 | 46.6 | 213G |
| mask pooling [62] | 43.1 | 51.5 | 46.0 | 217G |
| masked attention | 43.7 | 51.9 | 47.2 | 226G |

|  | AP | PQ | mIoU | FLOPs |
|---|---|---|---|---|
| single scale (1/321/32) | 41.5 | 50.2 | 46.1 | 218G |
| single scale (1/161/16) | 43.0 | 51.5 | 46.5 | 222G |
| single scale (1/81/8) | 44.0 | 51.8 | 47.4 | 239G |
| naïve m.s. (3 scales) | 44.0 | 51.9 | 46.3 | 247G |
| efficient m.s. (3 scales) | 43.7 | 51.9 | 47.2 | 226G |

|  | AP | PQ | mIoU | FLOPs |
|---|---|---|---|---|
| FPN [33] | 41.5 | 50.7 | 45.6 | 195G |
| Semantic FPN [27] | 42.1 | 51.2 | 46.2 | 258G |
| FaPN [39] | 42.4 | 51.8 | 46.8 | - |
| BiFPN [47] | 43.5 | 51.8 | 45.6 | 204G |
| MSDeformAttn [66] | 43.7 | 51.9 | 47.2 | 226G |

Instance segmentation. We compare Mask2Former with state-of-the-art models on the COCO [35] dataset in Table 2. With ResNet [25] backbone, Mask2Former outperforms a strong Mask R-CNN [24] baseline using large-scale jittering (LSJ) augmentation [23, 18] while requiring 8×8\times fewer training iterations. With Swin-L backbone, Mask2Former outperforms the state-of-the-art HTC++ [6]. Although we only observe +0.6+0.6 AP improvement over HTC++, the Boundary AP [12] improves by 2.12.1, suggesting that our predictions have a better boundary quality thanks to the high-resolution mask predictions. Note that for a fair comparison, we only consider single-scale inference and models trained with only COCO train2017 set data.

With a ResNet-50 backbone Mask2Former improves over MaskFormer on small objects by 7.0 APS{}^{\text{S}}, while overall the highest gains come from large objects (+10.6+10.6 APL{}^{\text{L}}). The performance on APS{}^{\text{S}} still lags behind other state-of-the-art models. Hence there still remains room for improvement on small objects, e.g., by using dilated backbones like in DETR [5], which we leave for future work.

Semantic segmentation. We compare Mask2Former with state-of-the-art models for semantic segmentation on the ADE20K [65] dataset in Table 3. Mask2Former outperforms MaskFormer [14] across different backbones, suggesting that the proposed improvements even boost semantic segmentation results where [14] was already state-of-the-art. With Swin-L as backbone and FaPN [39] as pixel decoder, Mask2Former sets a new state-of-the-art of 57.7 mIoU. We also report the test set results in Appendix A.3.

### 4.4 Ablation studies

We now analyze Mask2Former through a series of ablation studies using a ResNet-50 backbone [25]. To test the generality of the proposed components for universal image segmentation, all ablations are performed on three tasks.

Transformer decoder. We validate the importance of each component by removing them one at a time. As shown in Table 4(a), masked attention leads to the biggest improvement across all tasks. The improvement is larger for instance and panoptic segmentation than for semantic segmentation. Moreover, using high-resolution features from the efficient multi-scale strategy is also important. Table 4(b) shows additional optimization improvements further improve the performance without extra computation.

Masked attention. Concurrent work has proposed other variants of cross-attention [22, 40] that aim to improve the convergence and performance of DETR [5] for object detection. Most recently, K-Net [62] replaced cross-attention with a mask pooling operation that averages features within mask regions. We validate the importance of our masked attention in Table 4(c). While existing cross-attention variants may improve on a specific task, our masked attention performs the best on all three tasks.

Feature resolution. Table 4(d) shows that Mask2Former benefits from using high-resolution features (e.g., a single scale of 1/81/8) in the Transformer decoder. However, this introduces additional computation. Our efficient multi-scale (efficient m.s.) strategy effectively reduces the FLOPs without affecting the performance. Note that, naively concatenating multi-scale features as input to every Transformer decoder layer (naïve m.s.) does not yield additional gains.

Pixel decoder. As shown in Table 4(e), Mask2Former is compatible with any existing pixel decoder. However, we observe different pixel decoders specialize in different tasks: while BiFPN [47] performs better on instance-level segmentation, FaPN [39] works better for semantic segmentation. Among all studied pixel decoders, the MSDeformaAttn [66] consistently performs the best across all tasks and thus is selected as our default. This set of ablations also suggests that designing a module like a pixel decoder for a specific task does not guarantee generalization across segmentation tasks. Mask2Former, as a universal model, could serve as a testbed for a generalizable module design.

| matching lossmatching loss | matching losstraining loss | AP (COCO) | PQ (COCO) | mIoU (ADE20K) | memory (COCO) |
|---|---|---|---|---|---|
| mask (ours) | mask | 41.0 | 50.3 | 45.9 | 18G |
|  | point | 41.0 | 50.8 | 45.9 | 06G |
| point (ours) | mask | 43.1 | 51.4 | 47.3 | 18G |
|  | point (ours) | 43.7 | 51.9 | 47.2 | 06G |

Calculating loss with points vs. masks. In Table 5 we study the performance and memory implications when calculating the loss based on either mask or sampled points. Calculating the final training loss with sampled points reduces training memory by 3×3\times without affecting the performance. Additionally, calculating the matching loss with sampled points improves performance across all three tasks.

Learnable queries as region proposals. Region proposals [50, 1], either in the form of boxes or masks, are regions that are likely to be “objects.” With learnable queries being supervised by the mask loss, predictions from learnable queries can serve as mask proposals. In Figure 3 top, we visualize mask predictions of selected learnable queries before feeding them into the Transformer decoder (the proposal generation process is shown in Figure 3 bottom right). In Figure 3 bottom left, we further perform a quantitative analysis on the quality of these proposals by calculating the class-agnostic average recall with 100 predictions (AR@100) on COCO val2017. We find these learnable queries already achieve good AR@100 compared to the final predictions of Mask2Former after the Transformer decoder layers, i.e., layer 9, and AR@100 consistently improves with more decoder layers.

|  |  | panoptic model | semantic model |  |  |  |
|---|---|---|---|---|---|---|
| method | backbone | PQ | APpanTh{}^{\text{Th}}_{\text{pan}} | mIoUpan{}_{\text{pan}} | mIoU (s.s.) | (m.s.) |
| Panoptic FCN [31] | Swin-L†{}^{\text{\textdagger}} | 65.9 | - | - | - | - |
| Panoptic-DeepLab [11] | SWideRNet [9] | 66.4 | 40.1 | 82.2 | - | - |
| Panoptic-DeepLab [11] | SWideRNet [9] | 67.5∗ | 43.9∗ | 82.9∗ | - | - |
| SETR [64] | ViT-L†{}^{\text{\textdagger}} [17] | - | - | - | - | 82.2 |
| SegFormer [59] | MiT-B5 [59] | - | - | - | - | 84.0 |
| Mask2Former (ours) | R50 | 62.1 | 37.3 | 77.5 | 79.4 | 82.2 |
| Swin-B†{}^{\text{\textdagger}} | 66.1 | 42.8 | 82.7 | 83.3 | 84.5 |  |
| Swin-L†{}^{\text{\textdagger}} | 66.6 | 43.6 | 82.9 | 83.3 | 84.3 |  |

### 4.5 Generalization to other datasets

To show our Mask2Former can generalize beyond the COCO dataset, we further perform experiments on other popular image segmentation datasets. In Table 6, we show results on Cityscapes [16]. Please see Appendix B for detailed training settings on each dataset as well as more results on ADE20K [65] and Mapillary Vistas [42].

We observe that our Mask2Former is competitive to state-of-the-art methods on these datasets as well. It suggests Mask2Former can serve as a universal image segmentation model and results generalize across datasets.

|  | PQ | AP | mIoU |
|---|---|---|---|
| panoptic | 51.9 | 41.7 | 61.7 |
| instance | - | 43.7 | - |
| semantic | - | - | 61.5 |

| PQ | AP | mIoU |
|---|---|---|
| 39.7 | 26.5 | 46.1 |
| - | 26.4 | - |
| - | - | 47.2 |

| PQ | AP | mIoU |
|---|---|---|
| 62.1 | 37.3 | 77.5 |
| - | 37.4 | - |
| - | - | 79.4 |

### 4.6 Limitations

Our ultimate goal is to train a single model for all image segmentation tasks. In Table 7, we find Mask2Former trained on panoptic segmentation only performs slightly worse than the exact same model trained with the corresponding annotations for instance and semantic segmentation tasks across three datasets. This suggests that even though Mask2Former can generalize to different tasks, it still needs to be trained for those specific tasks. In the future, we hope to develop a model that can be trained only once for multiple tasks and even for multiple datasets.

Furthermore, as seen in Tables 2 and 4(d), even though it improves over baselines, Mask2Former struggles with segmenting small objects and is unable to fully leverage multi-scale features. We believe better utilization of the feature pyramid and designing losses for small objects are critical.

## 5 Conclusion

We present Mask2Former for universal image segmentation. Built upon a simple meta framework [14] with a new Transformer decoder using the proposed masked attention, Mask2Former obtains top results in all three major image segmentation tasks (panoptic, instance and semantic) on four popular datasets, outperforming even the best specialized models designed for each benchmark while remaining easy to train. Mask2Former saves 3×3\times research effort compared to designing specialized models for each task, and it is accessible to users with limited computational resources. We hope to attract interest in universal model design.

Ethical considerations: While our technical innovations do not appear to have any inherent biases, the models trained with our approach on real-world datasets should undergo ethical review to ensure the predictions do not propagate problematic stereotypes, and the approach is not used for applications including but not limited to illegal surveillance.

Acknowledgments: Thanks to Nicolas Carion and Xingyi Zhou for helpful feedback. BC and AS are supported in part by NSF #1718221, 2008387, 2045586, 2106825, MRI #1725729, NIFA 2020-67021-32799 and Cisco Systems Inc. (CG 1377144 - thanks for access to Arcetri).

Appendix

We first provide more results for Mask2Former with different backbones as well as test-set performance on standard benchmarks (Appendix A): We use COCO panoptic [28] for panoptic, COCO [35] for instance, and ADE20K [65] for semantic segmentation. Then, we provide more detailed results on additional datasets (Appendix B). Finally, we provide additional ablation studies (Appendix C) and visualization of Mask2Former predictions for all three segmentation tasks (Appendix D).

|  | method | backbone | search space | epochs | PQ | PQTh{}^{\text{Th}} | PQSt{}^{\text{St}} | APpanTh{}^{\text{Th}}_{\text{pan}} | mIoUpan{}_{\text{pan}} | #params. | FLOPs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN backbones | DETR [5] | R50 | 100 queries | 500+25 | 43.4 | 48.2 | 36.3 | 31.1 | - | - | - |
| R101 | 100 queries | 500+25 | 45.1 | 50.5 | 37.0 | 33.0 | - | - | - |  |  |
| K-Net [62] | R50 | 100 queries | 36 | 47.1 | 51.7 | 40.3 | - | - | - | - |  |
| Panoptic SegFormer [32] | R50 | 400 queries | 50 | 50.0 | 56.1 | 40.8 | - | - | 47M | 0246G |  |
| MaskFormer [14] | R50 | 100 queries | 300 | 46.5 | 51.0 | 39.8 | 33.0 | 57.8 | 045M | 0181G |  |
| R101 | 100 queries | 300 | 47.6 | 52.5 | 40.3 | 34.1 | 59.3 | 064M | 0248G |  |  |
| Mask2Former (ours) | R50 | 100 queries | 50 | 51.9 | 57.7 | 43.0 | 41.7 | 61.7 | 044M | 0226G |  |
| R101 | 100 queries | 50 | 52.6 | 58.5 | 43.7 | 42.6 | 62.4 | 063M | 0293G |  |  |
| Transformer backbones | Max-DeepLab [52] | Max-S | 128 queries | 216 | 48.4 | 53.0 | 41.5 | - | - | 062M | 0324G |
| Max-L | 128 queries | 216 | 51.1 | 57.0 | 42.2 | - | - | 451M | 3692G |  |  |
| Panoptic SegFormer [32] | PVTv2-B5 [54] | 400 queries | 50 | 54.1 | 60.4 | 44.6 | - | - | 101M | 0391G |  |
| K-Net [62] | Swin-L†{}^{\text{\textdagger}} | 100 queries | 36 | 54.6 | 60.2 | 46.0 | - | - | - | - |  |
| MaskFormer [14] | Swin-T | 100 queries | 300 | 47.7 | 51.7 | 41.7 | 33.6 | 60.4 | 042M | 0179G |  |
| Swin-S | 100 queries | 300 | 49.7 | 54.4 | 42.6 | 36.1 | 61.3 | 063M | 0259G |  |  |
| Swin-B | 100 queries | 300 | 51.1 | 56.3 | 43.2 | 37.8 | 62.6 | 102M | 0411G |  |  |
| Swin-B†{}^{\text{\textdagger}} | 100 queries | 300 | 51.8 | 56.9 | 44.1 | 38.5 | 63.6 | 102M | 0411G |  |  |
| Swin-L†{}^{\text{\textdagger}} | 100 queries | 300 | 52.7 | 58.5 | 44.0 | 40.1 | 64.8 | 212M | 0792G |  |  |
| Mask2Former (ours) | Swin-T | 100 queries | 50 | 53.2 | 59.3 | 44.0 | 43.3 | 63.2 | 047M | 0232G |  |
| Swin-S | 100 queries | 50 | 54.6 | 60.6 | 45.7 | 44.7 | 64.2 | 069M | 0313G |  |  |
| Swin-B | 100 queries | 50 | 55.1 | 61.0 | 46.1 | 45.2 | 65.1 | 107M | 0466G |  |  |
| Swin-B†{}^{\text{\textdagger}} | 100 queries | 50 | 56.4 | 62.4 | 47.3 | 46.3 | 67.1 | 107M | 0466G |  |  |
| Swin-L†{}^{\text{\textdagger}} | 200 queries | 100 | 57.8 | 64.2 | 48.1 | 48.6 | 67.4 | 216M | 0868G |  |  |

| method | backbone | PQ | PQTh{}^{\text{Th}} | PQSt{}^{\text{St}} | SQ | RQ |
|---|---|---|---|---|---|---|
| Max-DeepLab [52] | Max-L | 51.3 | 57.2 | 42.4 | 82.5 | 61.3 |
| Panoptic FCN [31] | Swin-L | 52.7 | 59.4 | 42.5 | - | - |
| MaskFormer [14] | Swin-L | 53.3 | 59.1 | 44.5 | 82.0 | 64.1 |
| Panoptic SegFormer [32] | PVTv2-B5 [54] | 54.4 | 61.1 | 44.3 | 83.3 | 64.6 |
| K-Net [62] | Swin-L | 55.2 | 61.2 | 46.2 | - | - |
| Megvii (challenge winner) | - | 54.7 | 64.6 | 39.8 | 83.6 | 64.3 |
| Mask2Former (ours) | Swin-L | 58.3 | 65.1 | 48.1 | 84.1 | 68.6 |

|  | method | backbone | search space | epochs | AP | APS{}^{\text{S}} | APM{}^{\text{M}} | APL{}^{\text{L}} | APboundary{}^{\text{boundary}} | #params. | FLOPs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN backbones | Mask R-CNN [24] | R50 | dense anchors | 36 | 37.2 | 18.6 | 39.5 | 53.3 | 23.1 | 044M | 0201G |
| R50 | dense anchors | 400 | 42.5 | 23.8 | 45.0 | 60.0 | 28.0 | 046M | 0358G |  |  |
| R101 | dense anchors | 36 | 38.6 | 19.5 | 41.3 | 55.3 | 24.5 | 063M | 0266G |  |  |
| R101 | dense anchors | 400 | 43.7 | 24.6 | 46.4 | 61.8 | 29.1 | 065M | 0423G |  |  |
| Mask2Former (ours) | R50 | 100 queries | 50 | 43.7 | 23.4 | 47.2 | 64.8 | 30.6 | 044M | 0226G |  |
| R101 | 100 queries | 50 | 44.2 | 23.8 | 47.7 | 66.7 | 31.1 | 063M | 0293G |  |  |
| Transformer backbones | QueryInst [20] | Swin-L†{}^{\text{\textdagger}} | 300 queries | 50 | 48.9 | 30.8 | 52.6 | 68.3 | 33.5 | - | - |
| Swin-HTC++ [36, 6] | Swin-B†{}^{\text{\textdagger}} | dense anchors | 36 | 49.1 | - | - | - | - | 160M | 1043G |  |
| Swin-L†{}^{\text{\textdagger}} | dense anchors | 72 | 49.5 | 31.0 | 52.4 | 67.2 | 34.1 | 284M | 1470G |  |  |
| Mask2Former (ours) | Swin-T | 100 queries | 50 | 45.0 | 24.5 | 48.3 | 67.4 | 31.8 | 047M | 0232G |  |
| Swin-S | 100 queries | 50 | 46.3 | 25.3 | 50.3 | 68.4 | 32.9 | 069M | 0313G |  |  |
| Swin-B | 100 queries | 50 | 46.7 | 26.1 | 50.5 | 68.8 | 33.2 | 107M | 0466G |  |  |
| Swin-B†{}^{\text{\textdagger}} | 100 queries | 50 | 48.1 | 27.8 | 52.0 | 71.1 | 34.4 | 107M | 0466G |  |  |
| Swin-L†{}^{\text{\textdagger}} | 200 queries | 100 | 50.1 | 29.9 | 53.9 | 72.1 | 36.2 | 216M | 0868G |  |  |

| method | backbone | AP | AP50 | AP75 | APS{}^{\text{S}} | APM{}^{\text{M}} | APL{}^{\text{L}} |
|---|---|---|---|---|---|---|---|
| QueryInst [20] | Swin-L | 49.1 | 74.2 | 53.8 | 31.5 | 51.8 | 63.2 |
| Swin-HTC++ [36, 6] | Swin-L | 50.2 | - | - | - | - | - |
| Swin-HTC++ [36, 6] (multi-scale) | Swin-L | 51.1 | - | - | - | - | - |
| Megvii (challenge winner) | - | 53.1 | 76.8 | 58.6 | 36.6 | 56.5 | 67.7 |
| Mask2Former (ours) | Swin-L | 50.5 | 74.9 | 54.9 | 29.1 | 53.8 | 71.2 |

|  | method | backbone | crop size | mIoU (s.s.) | mIoU (m.s.) | #params. | FLOPs |
|---|---|---|---|---|---|---|---|
| CNN | MaskFormer [14] | R50 | 512×512512\times 512 | 44.5 | 46.7 | 041M | 053G |
| R101 | 512×512512\times 512 | 45.5 | 47.2 | 060M | 073G |  |  |
| Mask2Former (ours) | R50 | 512×512512\times 512 | 47.2 | 49.2 | 044M | 071G |  |
| R101 | 512×512512\times 512 | 47.8 | 50.1 | 063M | 090G |  |  |
| Transformer backbones | Swin-UperNet [36, 58] | Swin-L†{}^{\text{\textdagger}} | 640×640640\times 640 | - | 53.5 | 234M | 647G |
| FaPN-MaskFormer [39, 14] | Swin-L†{}^{\text{\textdagger}} | 640×640640\times 640 | 55.2 | 56.7 | - | - |  |
| BEiT-UperNet [2, 58] | BEiT-L†{}^{\text{\textdagger}} | 640×640640\times 640 | - | 57.0 | 502M | - |  |
| MaskFormer [14] | Swin-T | 512×512512\times 512 | 46.7 | 48.8 | 042M | 055G |  |
| Swin-S | 512×512512\times 512 | 49.8 | 51.0 | 063M | 079G |  |  |
| Swin-B | 640×640640\times 640 | 51.1 | 52.3 | 102M | 195G |  |  |
| Swin-B†{}^{\text{\textdagger}} | 640×640640\times 640 | 52.7 | 53.9 | 102M | 195G |  |  |
| Swin-L†{}^{\text{\textdagger}} | 640×640640\times 640 | 54.1 | 55.6 | 212M | 375G |  |  |
| Mask2Former (ours) | Swin-T | 512×512512\times 512 | 47.7 | 49.6 | 047M | 074G |  |
| Swin-S | 512×512512\times 512 | 51.3 | 52.4 | 069M | 098G |  |  |
| Swin-B | 640×640640\times 640 | 52.4 | 53.7 | 107M | 223G |  |  |
| Swin-B†{}^{\text{\textdagger}} | 640×640640\times 640 | 53.9 | 55.1 | 107M | 223G |  |  |
| Swin-L†{}^{\text{\textdagger}} | 640×640640\times 640 | 56.1 | 57.3 | 215M | 403G |  |  |
| Swin-L-FaPN†{}^{\text{\textdagger}} | 640×640640\times 640 | 56.4 | 57.7 | 217M | - |  |  |

| method | backbone | P.A. | mIoU | score |
|---|---|---|---|---|
| SETR [64] | ViT-L | 78.35 | 45.03 | 61.69 |
| Swin-UperNet [36, 58] | Swin-L | 78.42 | 47.07 | 62.75 |
| MaskFormer [14] | Swin-L | 79.36 | 49.67 | 64.51 |
| Mask2Former (ours) | Swin-L-FaPN | 79.80 | 49.72 | 64.76 |

|  |  | panoptic model | instance model | semantic model |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
| method | backbone | PQ (s.s.) | PQ (m.s.) | APpanTh{}^{\text{Th}}_{\text{pan}} | mIoUpan{}_{\text{pan}} | AP | AP50 | mIoU (s.s.) | mIoU (m.s.) |
| Panoptic-DeepLab [11] | R50 | 60.3 | - | 32.1 | 78.7 | - | - | - | - |
| X71 [15] | 63.0 | 64.1 | 35.3 | 80.5 | - | - | - | - |  |
| SWideRNet [9] | 66.4 | 67.5 | 40.1 | 82.2 | - | - | - | - |  |
| Panoptic FCN [31] | Swin-L†{}^{\text{\textdagger}} | 65.9 | - | - | - | - | - | - | - |
| Segmenter [45] | ViT-L†{}^{\text{\textdagger}} | - | - | - | - | - | - | - | 81.3 |
| SETR [64] | ViT-L†{}^{\text{\textdagger}} | - | - | - | - | - | - | - | 82.2 |
| SegFormer [59] | MiT-B5 | - | - | - | - | - | - | - | 84.0 |
| Mask2Former (ours) | R50 | 62.1 | - | 37.3 | 77.5 | 37.4 | 61.9 | 79.4 | 82.2 |
| R101 | 62.4 | - | 37.7 | 78.6 | 38.5 | 63.9 | 80.1 | 81.9 |  |
| Swin-T | 63.9 | - | 39.1 | 80.5 | 39.7 | 66.9 | 82.1 | 83.0 |  |
| Swin-S | 64.8 | - | 40.7 | 81.8 | 41.8 | 70.4 | 82.6 | 83.6 |  |
| Swin-B†{}^{\text{\textdagger}} | 66.1 | - | 42.8 | 82.7 | 42.0 | 68.8 | 83.3 | 84.5 |  |
| Swin-L†{}^{\text{\textdagger}} | 66.6 | - | 43.6 | 82.9 | 43.7 | 71.4 | 83.3 | 84.3 |  |

|  |  | panoptic model | instance model | semantic model |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|
| method | backbone | PQ | APpanTh{}^{\text{Th}}_{\text{pan}} | mIoUpan{}_{\text{pan}} | AP | APS{}^{\text{S}} | APM{}^{\text{M}} | APL{}^{\text{L}} | mIoU (s.s.) | mIoU (m.s.) |
| MaskFormer [14] | R50 | 34.7 | - | - | - | - | - | - | - | - |
| Panoptic-DeepLab [11] | SWideRNet [9] | 37.9∗ | - | 50.0∗ | - | - | - | - | - | - |
| Swin-UperNet [36, 58] | Swin-L†{}^{\text{\textdagger}} | - | - | - | - | - | - | - | - | 53.5 |
| MaskFormer [14] | Swin-L†{}^{\text{\textdagger}} | - | - | - | - | - | - | - | 54.1 | 55.6 |
| FaPN-MaskFormer [39, 14] | Swin-L†{}^{\text{\textdagger}} | - | - | - | - | - | - | - | 55.2 | 56.7 |
| BEiT-UperNet [2, 58] | BEiT-L†{}^{\text{\textdagger}} | - | - | - | - | - | - | - | - | 57.0 |
| Mask2Former (ours) | R50 | 39.7 | 26.5 | 46.1 | 26.4 | 10.4 | 28.9 | 43.1 | 47.2 | 49.2 |
| Swin-L†{}^{\text{\textdagger}} | 48.1 | 34.2 | 54.5 | 34.9 | 16.3 | 40.0 | 54.7 | 56.1 | 57.3 |  |
| Swin-L-FaPN†{}^{\text{\textdagger}} | 46.2 | 33.2 | 55.4 | 33.4 | 14.6 | 37.6 | 54.6 | 56.4 | 57.7 |  |

|  |  | panoptic model | semantic model |  |  |
|---|---|---|---|---|---|
| method | backbone | PQ | mIoUpan{}_{\text{pan}} | mIoU (s.s.) | mIoU (m.s.) |
| Panoptic-DeepLab [11] | ensemble | 42.2∗ | 58.7∗ | - | - |
| SWideRNet [9] | 43.7 | 59.4 | - | - |  |
| SWideRNet [9] | 44.8∗ | 60.0∗ | - | - |  |
| Panoptic FCN [31] | Swin-L†{}^{\text{\textdagger}} | 45.7 | - | - | - |
| MaskFormer [14] | R50 | - | - | 53.1 | 55.4 |
| HMSANet [48] | HRNet [53] | - | - | - | 61.1 |
| Mask2Former (ours) | R50 | 36.3 | 50.7 | 57.4 | 59.0 |
| Swin-L†{}^{\text{\textdagger}} | 45.5 | 60.8 | 63.2 | 64.7 |  |

## Appendix A Additional results

Here, we provide more results of Mask2Former with different backbones on COCO panoptic [28] for panoptic segmentation, COCO [35] for instance segmentation and ADE20K [65] for semantic segmentation. More specifically, for each benckmark, we evaluate Mask2Former with ResNet [25] with 50 and 101 layers, as well as Swin [36] Tiny, Small, Base and Large variants as backbones. We use ImageNet [44] pre-trained checkpoints to initialize backbones.

### A.1 Panoptic segmentation.

In Table I, we report Mask2Former with various backbones on COCO panoptic val2017. Mask2Former outperforms all existing panoptic segmentation models with various backbones. Our best model sets a new state-of-the-art of 57.857.8 PQ.

In Table II, we further report the best Mask2Former model on the test-dev set. Note that Mask2Former trained only with the standard train2017 data, achieves the absolute new state-of-the-art performance on both validation and test set. Mask2Former even outperforms the best COCO competition entry which uses extra training data and test-time augmentation.

### A.2 Instance segmentation.

In Table III, we report Mask2Former results obtained with various backbones on COCO val2017. Mask2Former outperforms the best single-scale model, HTC++ [36, 6]. Note that it is non-trivial to do multi-scale inference for instance-level segmentation tasks without introducing complex post-processing like non-maximum suppression. Thus, we only compare Mask2Former with other single-scale inference models. We believe multi-scale inference can further improve Mask2Former performance and it remains an interesting future work.

In Table IV, we further report the best Mask2Former model on the test-dev set. Mask2Former achieves the absolute new state-of-the-art performance on both validation and test set. On the one hand, Mask2Former is extremely good at segmenting large objects: we can even outperform the challenge winner (which uses extra training data, model ensemble, etc.) on APL{}^{\text{L}} by a large margin without any bells-and-whistles. On the other hand, the poor performance on small objects leaves room for further improvement in the future.

### A.3 Semantic segmentation.

In Table V, we report Mask2Former results obtained with various backbones on ADE20K val. Mask2Former outperforms all existing semantic segmentation models with various backbones. Our best model sets a new state-of-the-art of 57.757.7 mIoU.

In Table VI, we further report the best Mask2Former model on the test set. Following [14], we train Mask2Former on the union of ADE20K train and val set with ImageNet-22K pre-trained checkpoint and use multi-scale inference. Mask2Former is able to outperform previous state-of-the-art methods on all metrics.

## Appendix B Additional datasets

We study Mask2Former on three image segmentation tasks (panoptic, instance and semantic segmentation) using four datasets. Here we report additional results on Cityscapes [16], ADE20K [65] and Mapillary Vistas [42] as well as more detailed training settings.

### B.1 Cityscapes

Cityscapes is an urban egocentric street-view dataset with high-resolution images (1024×20481024\times 2048 pixels). It contains 2975 images for training, 500 images for validation and 1525 images for testing with a total of 19 classes.

Training settings. For all three segmentation tasks: we use a crop size of 512×1024512\times 1024, a batch size of 16 and train all models for 90k iterations. During inference, we operate on the whole image (1024×20481024\times 2048). Other implementation details largely follow Section 4.1 (panoptic and instance segmentation follow semantic segmentation training settings), except that we use 200 queries for panoptic and instance segmentation models with Swin-L backbone. All other backbones or semantic segmentation models use 100 queries.

Results. In Table VII, we report Mask2Former results obtained with various backbones on Cityscapes for three segmentation tasks and compare it with other state-of-the-art methods without using extra data. For panoptic segmentation, Mask2Former with Swin-L backbone outperforms the state-of-the-art Panoptic-DeepLab [11] with SWideRnet [9] using single-scale inference. For semantic segmentation, Mask2Former with Swin-B backbone outperforms the state-of-the-art SegFormer [59].

### B.2 ADE20K

Training settings. For panoptic and instance segmentation, we use the exact same training parameters as we used for semantic segmentation, except that we always use a crop size of 640×640640\times 640 for all backbones. Other implementation details largely follow Section 4.1 , except that we use 200 queries for panoptic and instance segmentation models with Swin-L backbone. All other backbones or semantic segmentation models use 100 queries.

Results. In Table VIII, we report the results of Mask2Former obtained with various backbones on ADE20K for three segmentation tasks and compare it with other state-of-the-art methods. Mask2Former with Swin-L backbone sets a new state-of-the-art performance on ADE20K for panoptic segmentation. As there are few papers reporting results on ADE20K, we hope this experiment could set up a useful benchmark for future research.

### B.3 Mapillary Vistas

Mapillary Vistas is a large-scale urban street-view dataset with 18k, 2k and 5k images for training, validation and testing. It contains images with a variety of resolutions, ranging from 1024×7681024\times 768 to 4000×60004000\times 6000. We only report panoptic and semantic segmentation results for this dataset.

Training settings. For both panoptic and semantic segmentation, we follow the same data augmentation of [14]: standard random scale jittering between 0.5 and 2.0, random horizontal flipping, random cropping with a crop size of 1024×10241024\times 1024 as well as random color jittering. We train our model for 300k iterations with a batch size of 16 using the “poly” learning rate schedule [7]. During inference, we resize the longer side to 2048 pixels. Our panoptic segmentation model with a Swin-L backbone uses 200 queries. All other backbones or semantic segmentation models use 100 queries.

Results. In Table IX, we report Mask2Former results obtained with various backbones on Mapillary Vistas for panoptic and semantic segmentation tasks and compare it with other state-of-the-art methods. Our Mask2Former is very competitive compared to state-of-the art specialized models even if it is not designed for Mapillary Vistas.

## Appendix C Additional ablation studies

We perform additional ablation studies of Mask2Former using the same settings that we used in the main paper: a single ResNet-50 backbone [25].

### C.1 Convergence analysis

We train Mask2Former with 12, 25, 50 and 100 epochs with either standard scale augmentation (Standard Aug.) [57] or the more recent large-scale jittering augmentation (LSJ Aug.) [23, 18]. As shown in Figure IV, Mask2Former converges in 25 epochs using standard augmentation and almost converges in 50 epochs using large-scale jittering augmentation. This shows that Mask2Former with our proposed Transformer decoder converges faster than models using the standard Transformer decoder: e.g., DETR [5] and MaskFormer [14] require 500 epochs and 300 epochs respectively.

### C.2 Masked attention analysis

|  | 1/321/32 | 1/161/16 | 1/81/8 | average |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
|  | fg | bg | fg | bg | fg | bg | fg | bg |
| cross-attention | 0.23 | 0.77 | 0.23 | 0.77 | 0.15 | 0.85 | 0.20 | 0.80 |
| masked attention | 0.53 | 0.47 | 0.61 | 0.39 | 0.64 | 0.36 | 0.59 | 0.41 |

We quantitatively and qualitatively analyzed the COCO panoptic model with the R50 backbone. First, we visualize the last three attention maps of our model using cross-attention (Figure 10(a) top) and masked attention (Figure 10(a) bottom) of a single query that predicts the “cat.” With cross-attention, the attention map spreads over the entire image and the region with highest response is outside the object of interest. We believe this is because the softmax used in cross-attention never attains zero, and small attention weights on large background regions start to dominate. Instead, masked attention limits the attention weights to focus on the object. We validate this hypothesis in Table 10(b): we compute the cumulative attention weights on foreground (defined by the matching ground truth to each prediction) and background for all queries on the entire COCO val set. On average, only 20%20\% of the attention weights in cross-attention focus on the foreground while masked attention increases this ratio to almost 60%60\%.

Second, we plot the panoptic segmentation performance using output from each Transformer decoder layer (Figure II). We find masked attention with a single Transformer decoder layer already outperforms cross-attention with 9 layers. We hope the effectiveness of masked attention, together with this analysis, leads to better attention design.

### C.3 Object query analysis

Object queries play an important role in Mask2Former. We ablate different design choices of object queries including the number of queries and making queries learnable.

Number of queries. We study the effect of different number of queries for three image segmentation tasks in Table 11(a). For instance and semantic segmentation, using 100 queries achieves the best performance, while using 200 queries can further improve panoptic segmentation results. As panoptic segmentation is a combination of instance and semantic segmentation, it has more segments per image than the other two tasks. This ablation suggests that picking the number of queries for Mask2Former may depend on the number of segments per image for a particular task or dataset.

Learnable queries. An object query consists of two parts: object query features and object query positional embeddings. Object query features are only used as the initial input to the Transformer decoder and are updated through decoder layers; whereas query positional embeddings are added to query features in every Transformer decoder layer when computing the attention weights. In DETR [5], query features are zero-initialized and query positional embeddings are learnable. Furthermore, there is no direct supervision on these query features before feeding them into the Transformer (since they are zero vectors). In our Mask2Former, we still make query positional embeddings learnable. In addition, we make query features learnable as well and directly apply losses on these learnable query features before feeding them into the Transformer decoder.

In Table 11(b), we compare our learnable query features with zero-initialized query features in DETR. We find it is important to directly supervise object queries even before feeding them into the Transformer decoder. Learnable queries without supervision perform similarly well as zero-initialized queries in DETR.

### C.4 MaskFormer vs. Mask2Former

Mask2Former builds upon the same meta architecture as MaskFormer [14] with two major differences: 1) We use more advanced training parameters summarized in Table 12(a); and 2) we propose a new Transformer decoder with masked attention, instead of using the standard Transformer decoder, as well as some optimization improvements summarized in Table 12(b). To better understand Mask2Former’s improvements over MaskFormer, we perform ablation studies on training parameter improvements and Transformer decoder improvements in isolation.

In Table 12(c), we study our new training parameters. We train the MaskFormer model with either its original training parameters in [14] or our new training parameters. We observe significant improvements of using our new training parameters for MaskFormer as well. This shows the new training parameters are also generally applicable to other models.

In Table 12(d), we study our new Transformer decoder. We train a MaskFormer model and a Mask2Former model with the exact same backbone, i.e., a ResNet-50; pixel decoder, i.e., a FPN; and training parameters. That is, the only difference is in the Transformer decoder, summarized in Table 12(b). We observe improvements for all three tasks, suggesting that the new Transformer decoder itself is indeed better than the standard Transformer decoder.

While computational efficiency was not our primary goal, we find that Mask2Former actually has a better compute-performance trade-off compared to MaskFormer (Figure III). Even the lightest instantiation of Mask2Former outperforms the heaviest MaskFormer instantiation, using 14th\frac{1}{4}^{\text{th}} the FLOPs.

## Appendix D Visualization

We visualize sample predictions of the Mask2Former model with Swin-L [36] backbone on three tasks: COCO panoptic val2017 set for panoptic segmentation (57.8 PQ) in Figure V, COCO val2017 set for instance segmentation (50.1 AP) in Figure VI and ADE20K validation set for semantic segmentation (57.7 mIoU, multi-scale inference) in Figure VII.

|  | AP (COCO) | PQ (COCO) | mIoU (ADE20K) | FLOPs (COCO) |
|---|---|---|---|---|
| 50 | 42.4 | 50.5 | 46.2 | 217G |
| 100 | 43.7 | 51.9 | 47.2 | 226G |
| 200 | 43.5 | 52.2 | 47.0 | 246G |
| 300 | 43.5 | 52.1 | 46.5 | 265G |
| 1000 | 40.3 | 50.7 | 44.8 | 405G |

|  | AP (COCO) | PQ (COCO) | mIoU (ADE20K) | FLOPs (COCO) |
|---|---|---|---|---|
| zero-initialized (DETR [5]) | 42.9 | 51.2 | 45.5 | 226G |
| learnable w/o supervision | 42.9 | 51.2 | 47.0 | 226G |
| learnable w/ supervision | 43.7 | 51.9 | 47.2 | 226G |

| training parameters | MaskFormer | Mask2Former (ours) |
|---|---|---|
| learning rate | 0.0001 | 0.0001 |
| weight decay | 0.0001 | 0.05 |
| batch size | 16∗ | 16 |
| epochs | 75∗ | 50 |
| data augmentation | standard scale aug. w/ crop | LSJ aug. |
| λcls\lambda_{\text{cls}} | 1.0 | 2.0 |
| λfocal\lambda_{\text{focal}} / λce\lambda_{\text{ce}} | 20.0 / - | - / 5.0 |
| λdice\lambda_{\text{dice}} | 1.0 | 5.0 |
| mask loss | mask | 12544 sampled points |

| Transformer decoder | MaskFormer | Mask2Former (ours) |
|---|---|---|
| # of layers | 6 | 9 |
| single layer | SA-CA-FFN | MA-SA-FFN |
| dropout | 0.1 | 0.0 |
| feature resolution | {1/32}×6\{1/32\}\times 6 | {1/32,1/16,1/8}×3\{1/32,1/16,1/8\}\times 3 |
| input query features | zero init. | learnable |
| query p.e. | learnable | learnable |

| modelmodelmodel | modelmodeltraining params. | AP (COCO) | PQ (COCO) | mIoU (ADE20K) |
|---|---|---|---|---|
| MaskFormer | MaskFormer | 34.0 | 46.5 | 44.5 |
| MaskFormer | Mask2Former | 37.8 (+3.8) | 48.2 (+1.7) | 45.3 (+0.8) |

| modelmodelTransformer decoder | modelmodelpixel decoder | AP (COCO) | PQ (COCO) | mIoU (ADE20K) |
|---|---|---|---|---|
| MaskFormer | FPN | 37.8 | 48.2 | 45.3 |
| Mask2Former | FPN | 41.5 (+3.7) | 50.7 (+2.5) | 45.6 (+0.3) |
