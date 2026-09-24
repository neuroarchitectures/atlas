# DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection

We present DINO (DETR with Improved deNoising anchOr boxes), a state-of-the-art end-to-end object detector. DINO improves over previous DETR-like models in performance and efficiency by using a contrastive way for denoising training, a mixed query selection method for anchor initialization, and a look forward twice scheme for box prediction. DINO achieves 49.449.4AP in 1212 epochs and 51.351.3AP in 2424 epochs on COCO with a ResNet-50 backbone and multi-scale features, yielding a significant improvement of +6.0AP and +2.7AP, respectively, compared to DN-DETR, the previous best DETR-like model. DINO scales well in both model size and data size. Without bells and whistles, after pre-training on the Objects365 dataset with a SwinL backbone, DINO obtains the best results on both COCO val2017 (63.2AP) and test-dev (63.3AP). Compared to other models on the leaderboard, DINO significantly reduces its model size and pre-training data size while achieving better results. Our code will be available at https://github.com/IDEACVR/DINO.

## 1 Introduction

Object detection is a fundamental task in computer vision. Remarkable progress has been accomplished by classical convolution-based object detection algorithms [31, 35, 19, 2, 12]. Despite that such algorithms normally include hand-designed components like anchor generation and non-maximum suppression (NMS), they yield the best detection models such as DyHead [7], Swin [23] and SwinV2 [22] with HTC++ [4], as evidenced on the COCO test-dev leaderboard [1].

In contrast to classical detection algorithms, DETR [3] is a novel Transformer-based detection algorithm. It eliminates the need of hand-designed components and achieves comparable performance with optimized classical detectors like Faster RCNN [31]. Different from previous detectors, DETR models object detection as a set prediction task and assigns labels by bipartite graph matching. It leverages learnable queries to probe the existence of objects and combine features from an image feature map, which behaves like soft ROI pooling [21].

Despite its promising performance, the training convergence of DETR is slow and the meaning of queries is unclear. To address such problems, many methods have been proposed, such as introducing deformable attention [41], decoupling positional and content information [25], providing spatial priors [11, 39, 37], etc. Recently, DAB-DETR [21] proposes to formulate DETR queries as dynamic anchor boxes (DAB), which bridges the gap between classical anchor-based detectors and DETR-like ones. DN-DETR [17] further solves the instability of bipartite matching by introducing a denoising (DN) technique. The combination of DAB and DN makes DETR-like models competitive with classical detectors on both training efficiency and inference performance.

The best detection models nowadays are based on improved classical detectors like DyHead [8] and HTC [4]. For example, the best result presented in SwinV2 [22] was trained with the HTC++ [4, 23] framework. Two main reasons contribute to the phenomenon: 1) Previous DETR-like models are inferior to the improved classical detectors. Most classical detectors have been well studied and highly optimized, leading to a better performance compared with the newly developed DETR-like models. For instance, the best performing DETR-like models nowadays are still under 5050 AP on COCO. 2) The scalability of DETR-like models has not been well studied. There is no reported result about how DETR-like models perform when scaling to a large backbone and a large-scale data set. We aim to address both concerns in this paper.

Specifically, by improving the denoising training, query initialization, and box prediction, we design a new DETR-like model based on DN-DETR [17], DAB-DETR [21], and Deformable DETR [41]. We name our model as DINO (DETR with Improved deNoising anchOr box). As shown in Fig. 1, the comparison on COCO shows the superior performance of DINO. In particular, DINO demonstrates a great scalability, setting a new record of 63.363.3 AP on the COCO test-dev leaderboard [1]

As a DETR-like model, DINO contains a backbone, a multi-layer Transformer encoder, a multi-layer Transformer decoder, and multiple prediction heads. Following DAB-DETR [21], we formulate queries in decoder as dynamic anchor boxes and refine them step-by-step across decoder layers. Following DN-DETR [17], we add ground truth labels and boxes with noises into the Transformer decoder layers to help stabilize bipartite matching during training. We also adopt deformable attention [41] for its computational efficiency. Moreover, we propose three new methods as follows. First, to improve the one-to-one matching, we propose a contrastive denoising training by adding both positive and negative samples of the same ground truth at the same time. After adding two different noises to the same ground truth box, we mark the box with a smaller noise as positive and the other as negative. The contrastive denoising training helps the model to avoid duplicate outputs of the same target. Second, the dynamic anchor box formulation of queries links DETR-like models with classical two-stage models. Hence we propose a mixed query selection method, which helps better initialize the queries. We select initial anchor boxes as positional queries from the output of the encoder, similar to [41, 39]. However, we leave the content queries learnable as before, encouraging the first decoder layer to focus on the spatial prior. Third, to leverage the refined box information from later layers to help optimize the parameters of their adjacent early layers, we propose a new look forward twice scheme to correct the updated parameters with gradients from later layers.

We validate the effectiveness of DINO with extensive experiments on the COCO [20] detection benchmarks. As shown in Fig. 1, DINO achieves 49.449.4AP in 1212 epochs and 51.351.3AP in 2424 epochs with ResNet-50 and multi-scale features, yielding a significant improvement of +6.0AP and +2.7AP, respectively, compared to the previous best DETR-like model. In addition, DINO scales well in both model size and data size. After pre-training on the Objects365 [33] data set with a SwinL [23] backbone, DINO achieves the best results on both COCO val2017 (63.2AP) and test-dev (63.3AP) benchmarks, as shown in Table 3. Compared to other models on the leaderboard [1], we reduce the model size to 1/15 compared to SwinV2-G [22]. Compared to Florence [40], we reduce the pre-training detection dataset to 1/5 and backbone pre-training dataset to 1/60 while achieving better results.

We summarize our contributions as follows.

- 1. We design a new end-to-end DETR-like object detector with several novel techniques, including contrastive DN training, mixed query selection, and look forward twice for different parts of the DINO model.
We design a new end-to-end DETR-like object detector with several novel techniques, including contrastive DN training, mixed query selection, and look forward twice for different parts of the DINO model.

- 2. We conduct intensive ablation studies to validate the effectiveness of different design choices in DINO. As a result, DINO achieves 49.449.4AP in 1212 epochs and 51.351.3AP in 2424 epochs with ResNet-50 and multi-scale features, significantly outperforming the previous best DETR-like models. In particular, DINO trained in 1212 epochs shows a more significant improvement on small objects, yielding an improvement of +7.5AP.
We conduct intensive ablation studies to validate the effectiveness of different design choices in DINO. As a result, DINO achieves 49.449.4AP in 1212 epochs and 51.351.3AP in 2424 epochs with ResNet-50 and multi-scale features, significantly outperforming the previous best DETR-like models. In particular, DINO trained in 1212 epochs shows a more significant improvement on small objects, yielding an improvement of +7.5AP.

- 3. We show that, without bells and whistles, DINO can achieve the best performance on public benchmarks. After pre-training on the Objects365 [33] dataset with a SwinL [23] backbone, DINO achieves the best results on both COCO val2017 (63.2AP) and test-dev (63.3AP) benchmarks. To the best of our knowledge, this is the first time that an end-to-end Transformer detector outperforms state-of-the-art (SOTA) models on the COCO leaderboard [1].
We show that, without bells and whistles, DINO can achieve the best performance on public benchmarks. After pre-training on the Objects365 [33] dataset with a SwinL [23] backbone, DINO achieves the best results on both COCO val2017 (63.2AP) and test-dev (63.3AP) benchmarks. To the best of our knowledge, this is the first time that an end-to-end Transformer detector outperforms state-of-the-art (SOTA) models on the COCO leaderboard [1].

## 2 Related Work

### 2.1 Classical Object Detectors

Early convolution-based object detectors are either two-stage or one-stage models, based on hand-crafted anchors or reference points. Two-stage models [30, 13] usually use an region proposal network (RPN) [30] to propose potential boxes, which are then refined in the second stage. One-stage models such as YOLO v2 [28] and YOLO v3 [29] directly output offsets relative to predefined anchors. Recently, some convolution-based models such as HTC++ [4] and Dyhead [7] have achieved top performance on the COCO 2017 dataset [20]. The performance of convolution-based models, however, relies on the way they generate anchors. Moreover, they need hand-designed components like NMS to remove duplicate boxes, and hence cannot perform end-to-end optimization.

### 2.2 DETR and Its Variants

Carion et al. [3] proposed a Transformer-based end-to-end object detector named DETR (DEtection TRansformer) without using hand-designed components like anchor design and NMS. Many follow-up papers have attempted to address the slow training convergence issue of DETR introduced by decoder cross-attention. For instance, Sun et al. [34] designed an encoder-only DETR without using a decoder. Dai et al. [7] proposed a dynamic decoder to focus on important regions from multiple feature levels.

Another line of works is towards a deeper understanding of decoder queries in DETR. Many papers associate queries with spatial position from different perspectives. Deformable DETR [41] predicts 22D anchor points and designs a deformable attention module that only attends to certain sampling points around a reference point. Efficient DETR [39] selects top K positions from encoder’s dense prediction to enhance decoder queries. DAB-DETR [21] further extends 22D anchor points to 44D anchor box coordinates to represent queries and dynamically update boxes in each decoder layer. Recently, DN-DETR [17] introduces a denoising training method to speed up DETR training. It feeds noise-added ground-truth labels and boxes into the decoder and trains the model to reconstruct the original ones. Our work of DINO in this paper is based on DAB-DETR and DN-DETR, and also adopts deformable attention for its computational efficiency.

### 2.3 Large-scale Pre-training for Object Detection

Large-scale pre-training have had a big impact on both natural language processing [10] and computer vision [27]. The best performance detectors nowadays are mostly achieved with large backbones pre-trained on large-scale data. For example, Swin V2 [22] extends its backbone size to 3.03.0 Billion parameters and pre-trains its models with 7070M privately collected images. Florence [40] first pre-trains its backbone with 900900M privately curated image-text pairs and then pre-trains its detector with 99M images with annotated or pseudo boxes. In contrast, DINO achieves the SOTA result with a publicly available SwinL [23] backbone and a public dataset Objects365 [33] (1.7M annotated images) only.

## 3 DINO: DETR with Improved DeNoising Anchor Boxes

### 3.1 Preliminaries

As studied in Conditional DETR [25] and DAB-DETR [21], it becomes clear that queries in DETR [3] are formed by two parts: a positional part and a content part, which are referred to as positional queries and content queries in this paper. DAB-DETR [21] explicitly formulates each positional query in DETR as a 4D anchor box (x,y,w,h)(x,y,w,h), where xx and yy are the center coordinates of the box and ww and hh correspond to its width and height. Such an explicit anchor box formulation makes it easy to dynamically refine anchor boxes layer by layer in the decoder.

DN-DETR [17] introduces a denoising (DN) training method to accelerate the training convergence of DETR-like models. It shows that the slow convergence problem in DETR is caused by the instability of bipartite matching. To mitigate this problem, DN-DETR proposes to additionally feed noised ground-truth (GT) labels and boxes into the Transformer decoder and train the model to reconstruct the ground-truth ones. The added noise (Δ​x,Δ​y,Δ​w,Δ​h)(\Delta x,\Delta y,\Delta w,\Delta h) is constrained by |Δ​x|<λ​w2|\Delta x|<\frac{\lambda w}{2}, |Δ​y|<λ​h2|\Delta y|<\frac{\lambda h}{2}, |Δ​w|<λ​w|\Delta w|<\lambda w, and |Δ​y|<λ​h|\Delta y|<\lambda h, where (x,y,w,h)(x,y,w,h) denotes a GT box and λ\lambda11 1 The DN-DETR paper [17] uses λ1\lambda_{1} and λ2\lambda_{2} to denote noise scales of center shifting and box scaling, but sets λ1=λ2\lambda_{1}=\lambda_{2}. In this paper, we use λ\lambda in place of λ1​and​λ2\lambda_{1}\text{and}\lambda_{2} for simplicity. is a hyper-parameter to control the scale of noise. Since DN-DETR follows DAB-DETR to view decoder queries as anchors, a noised GT box can be viewed as a special anchor with a GT box nearby as λ\lambda is usually small. In addition to the orginal DETR queries, DN-DETR adds a DN part which feeds noised GT labels and boxes into the decoder to provide an auxiliary DN loss. The DN loss effectively stabilizes and speeds up the DETR training and can be plugged into any DETR-like models.

Deformable DETR [41] is another early work to speed up the convergence of DETR. To compute deformable attention, it introduces the concept of reference point so that deformable attention can attend to a small set of key sampling points around a reference. The reference point concept makes it possible to develop several techniques to further improve the DETR performance. The first technique is query selection22 2 Also named as “two-stage” in the Deformable DETR paper. As the “two-stage” name might confuse readers with classical detectors, we use the term “query selection” instead in our paper., which selects features and reference boxes from the encoder as inputs to the decoder directly. The second technique is iterative bounding box refinement with a careful gradient detachment design between two decoder layers. We call this gradient detachment technique “look forward once” in our paper.

Following DAB-DETR and DN-DETR, DINO formulates the positional queries as dynamic anchor boxes and is trained with an extra DN loss. Note that DN-DETR also adopts several techniques from Deformable DETR to achieve a better performance, including its deformable attention mechanism and “look forward once” implementation in layer parameter update. DINO further adopts the query selection idea from Deformable DETR to better initialize the positional queries. Built upon this strong baseline, DINO introduces three novel methods to further improve the detection performance, which will be described in Sec. 3.3, Sec. 3.4, and Sec. 3.5, respectively.

### 3.2 Model Overview

As a DETR-like model, DINO is an end-to-end architecture which contains a backbone, a multi-layer Transformer [36] encoder, a multi-layer Transformer decoder, and multiple prediction heads. The overall pipeline is shown in Fig. 2. Given an image, we extract multi-scale features with backbones like ResNet [14] or Swin Transformer [23], and then feed them into the Transformer encoder with corresponding positional embeddings. After feature enhancement with the encoder layers, we propose a new mixed query selection strategy to initialize anchors as positional queries for the decoder. Note that this strategy does not initialize content queries but leaves them learnable. More details of mixed query selection are available in Sec. 3.4. With the initialized anchors and the learnable content queries, we use the deformable attention [41] to combine the features of the encoder outputs and update the queries layer-by-layer. The final outputs are formed with refined anchor boxes and classification results predicted by refined content features. As in DN-DETR [17], we have an extra DN branch to perform denoising training. Beyond the standard DN method, we propose a new contrastive denoising training approach by taking into account hard negative samples, which will be presented in Sec. 3.3. To fully leverage the refined box information from later layers to help optimize the parameters of their adjacent early layer, a novel look forward twice method is proposed to pass gradients between adjacent layers, which will be described in Sec. 3.5.

### 3.3 Contrastive DeNoising Training

DN-DETR is very effective in stabilizing training and accelerating convergence. With the help of DN queries, it learns to make predictions based on anchors which have GT boxes nearby. However, it lacks a capability of predicting “no object” for anchors with no object nearby. To address this issue, we propose a Contrastive DeNoising (CDN) approach to rejecting useless anchors.

Implementation: DN-DETR has a hyper-parameter λ\lambda to control the noise scale. The generated noises are no larger than λ\lambda as DN-DETR wants the model to reconstruct the ground truth (GT) from moderately noised queries. In our method, we have two hyper-parameters λ1\lambda_{1} and λ2\lambda_{2}, where λ1<λ2\lambda_{1}<\lambda_{2}. As shown in the concentric squares in Fig. 3, we generate two types of CDN queries: positive queries and negative queries. Positive queries within the inner square have a noise scale smaller than λ1\lambda_{1} and are expected to reconstruct their corresponding ground truth boxes. Negative queries between the inner and outer squares have a noise scale larger than λ1\lambda_{1} and smaller than λ2\lambda_{2}. They are are expected to predict “no object”. We usually adopt small λ2\lambda_{2} because hard negative samples closer to GT boxes are more helpful to improve the performance. As shown in Fig. 3, each CDN group has a set of positive queries and negative queries. If an image has nn GT boxes, a CDN group will have 2×n2\times n queries with each GT box generating a positive and a negative queries. Similar to DN-DETR, we also use multiple CDN groups to improve the effectiveness of our method. The reconstruction losses are l1l_{1} and GIOU losses for box regression and focal loss [19] for classification. The loss to classify negative samples as background is also focal loss.

Analysis: The reason why our method works is that it can inhibit confusion and select high-quality anchors (queries) for predicting bounding boxes. The confusion happens when multiple anchors are close to one object. In this case, it is hard for the model to decide which anchor to choose. The confusion may lead to two problems. The first is duplicate predictions. Although DETR-like models can inhibit duplicate boxes with the help of set-based loss and self-attention [3], this ability is limited. As shown in the left figure of Fig. 8, when replacing our CDN queries with DN queries, the boy pointed by the arrow has 33 duplicate predictions. With CDN queries, our model can distinguish the slight difference between anchors and avoid duplicate predictions as shown in the right figure of Fig. 8. The second problem is that an unwanted anchor farther from a GT box might be selected. Although denoising training [17] has improved the model to choose nearby anchors, CDN further improves this capability by teaching the model to reject farther anchors.

Effectiveness: To demonstrate the effectiveness of CDN, we define a metric called Average Top-K Distance (ATD(kk)) and use it to evaluate how far anchors are from their target GT boxes in the matching part. As in DETR, each anchor corresponds to a prediction which may be matched with a GT box or background. We only consider those matched with GT boxes here. Assume we have NN GT bounding boxes b0,b2,…,bN−1{b_{0},b_{2},...,b_{N-1}} in a validation set, where bi=(xi,yi,wi,hi)b_{i}=(x_{i},y_{i},w_{i},h_{i}). For each bib_{i}, we can find its corresponding anchor and denote it as ai=(xi′,yi′,wi′,hi′)a_{i}=(x^{\prime}_{i},y^{\prime}_{i},w^{\prime}_{i},h^{\prime}_{i}). aia_{i} is the initial anchor box of the decoder whose refined box after the last decoder layer is assigned to bib_{i} during matching. Then we have

|  | A​T​D​(k)=1k​∑{t​o​p​K⁡({∥b0−a0∥1,∥b1−a1∥1,…,∥bN−1−aN−1∥1},k)}ATD(k)=\frac{1}{k}\sum\left\{\mathop{topK}\left(\left\{\lVert b_{0}-a_{0}\rVert_{1},\lVert b_{1}-a_{1}\rVert_{1},...,\lVert b_{N-1}-a_{N-1}\rVert_{1}\right\},k\right)\right\} |  | (1) |
|---|---|---|---|

where ∥bi−ai∥1\lVert b_{i}-a_{i}\rVert_{1} is the l1l_{1} distance between bib_{i} and aia_{i} and t​o​p​K⁡(𝐱,k)\mathop{topK}(\mathbf{x},k) is a function that returns the set of kk largest elements in 𝐱\mathbf{x}. The reason why we select the top-K elements is that the confusion problem is more likely to happen when the GT box is matched with a farther anchor. As shown in (a) and (b) of Fig. 4, DN is good enough for selecting a good anchor overall. However, CDN finds better anchors for small objects. Fig. 4 (c) shows that CDN queries lead to an improvement of +1.3+1.3 AP over DN queries on small objects in 1212 epochs with ResNet-50 and multi-scale features.

### 3.4 Mixed Query Selection

In DETR [3] and DN-DETR [17], decoder queries are static embeddings without taking any encoder features from an individual image, as shown in Fig. 5 (a). They learn anchors (in DN-DETR and DAB-DETR) or positional queries (in DETR) from training data directly and set the content queries as all 00 vectors. Deformable DETR [41] learns both the positional and content queries, which is another implementation of static query initialization. To further improve the performance, Deformable DETR [41] has a query selection variant (called ”two-stage” in [41]), which select top K encoder features from the last encoder layer as priors to enhance decoder queries. As shown in Fig. 5 (b), both the positional and content queries are generated by a linear transform of the selected features. In addition, these selected features are fed to an auxiliary detection head to get predicted boxes, which are used to initialize reference boxes. Similarly, Efficient DETR [39] also selects top K features based on the objectiveness (class) score of each encoder feature.

The dynamic 44D anchor box formulation of queries in our model makes it closely related to decoder positional queries, which can be improved by query selection. We follow the above practice and propose a mixed query selection approach. As shown in Fig. 5 (c), we only initialize anchor boxes using the position information associated with the selected top-K features, but leave the content queries static as before. Note that Deformable DETR [41] utilizes the top-K features to enhance not only the positional queries but also the content queries. As the selected features are preliminary content features without further refinement, they could be ambiguous and misleading to the decoder. For example, a selected feature may contain multiple objects or be only part of an object. In contrast, our mixed query selection approach only enhances the positional queries with top-K selected features and keeps the content queries learnable as before. It helps the model to use better positional information to pool more comprehensive content features from the encoder.

### 3.5 Look Forward Twice

We propose a new way to box prediction in this section. The iterative box refinement in Deformable DETR [41] blocks gradient back propagation to stabilize training. We name the method look forward once since the parameters of layer ii are updated based on the auxiliary loss of boxes bib_{i} only, as shown in Fig. 6 (a). However, we conjecture that the improved box information from a later layer could be more helpful to correct the box prediction in its adjacent early layer. Hence we propose another way called look forward twice to perform box update, where the parameters of layer-ii are influenced by losses of both layer-ii and layer-(i+1)(i+1), as shown in Fig. 6 (b). For each predicted offset Δ​bi\Delta b_{i}, it will be used to update box twice, one for bi′b_{i}^{\prime} and another for bi+1(p​r​e​d)b_{i+1}^{(pred)}, hence we name our method as look forward twice.

The final precision of a predicted box bi(pred)b_{i}^{(\mathrm{pred})} is determined by two factors: the quality of the initial box bi−1b_{i-1} and the predicted offset of the box Δ​bi\Delta b_{i}. The look forward once scheme optimizes the latter only, as the gradient information is detached from layer-ii to layer-(i−1)(i-1). In contrast, we improve both the initial box bi−1b_{i-1} and the predicted box offset Δ​bi\Delta b_{i}. A simple way to improving the quality is supervising the final box bi′b_{i}^{\prime} of layer ii with the output of the next layer Δ​bi+1\Delta b_{i+1}. Hence we use the sum of bi′b_{i}^{\prime} and Δ​bi+1\Delta b_{i+1} as the predicted box of layer-(i+1)(i+1).

More specifically, given an input box bi−1b_{i-1} for the ii-th layer, we obtain the final prediction box bi(pred)b_{i}^{(\mathrm{pred})} by:

|  | Δ​bi\displaystyle\Delta b_{i} | =Layeri(bi−1),\displaystyle=\mathrm{Layer_{i}}(b_{i-1}),\quad\quad | bi′\displaystyle b_{i}^{\prime} | =Update⁡(bi−1,Δ​bi),\displaystyle=\mathrm{Update}(b_{i-1},\Delta b_{i}), |  | (2) |
|---|---|---|---|---|---|---|
|  | bi\displaystyle b_{i} | =Detach(bi′),\displaystyle=\mathrm{Detach}(b_{i}^{\prime}),\quad\quad | bi(pred)\displaystyle b_{i}^{(\mathrm{pred})} | =Update⁡(bi−1′,Δ​bi),\displaystyle=\mathrm{Update}(b_{i-1}^{\prime},\Delta b_{i}), |  |  |

where bi′b_{i}^{\prime} is the undetached version of bib_{i}. The term Update⁡(⋅,⋅)\mathrm{Update}(\cdot,\cdot) is a function that refines the box bi−1b_{i-1} by the predicted box offset Δ​bi\Delta b_{i}. We adopt the same way for box update33 3 We use normalized forms of boxes in our model, hence each value of a box is a float between 00 and 11. Given two boxes, we sum them after inverse sigmoid and then transform the summation by sigmoid. Refer to Deformable DETR [41] Sec. A.3 for more details. as in Deformable DETR [41].

## 4 Experiments

### 4.1 Setup

Dataset and Backbone: We conduct evaluation on the COCO 2017 object detection dataset [20], which is split into train2017 and val2017 (also called minival). We report results with two different backbones: ResNet-50 [14] pre-trained on ImageNet-1k [9] and SwinL [23] pre-trained on ImageNet-22k [9]. DINO with ResNet-50 is trained on train2017 without extra data, while DINO with SwinL is first pre-trained on Object365 [33] and then fine-tuned on train2017. We report the standard average precision (AP) result on val2017 under different IoU thresholds and object scales. We also report the test-dev results for DINO with SwinL. Implementation Details: DINO is composed of a backbone, a Transformer encoder, a Transformer decoder, and multiple prediction heads. In appendix 0.D, we provide more implementation details, including all the hyper-parameters and engineering techniques used in our models for those who want to reproduce our results. We will release the code after the blind review.

### 4.2 Main Results

| Model | Epochs | AP | AP50 | AP75 | APS | APM | APL | GFLOPS | Params | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| Faster-RCNN(5scale) [30] | 1212 | 37.937.9 | 58.858.8 | 41.141.1 | 22.422.4 | 41.141.1 | 49.149.1 | 207207 | 4040M | 21∗21^{*} |
| DETR(DC5) [3] | 1212 | 15.515.5 | 29.429.4 | 14.514.5 | 4.34.3 | 15.115.1 | 26.726.7 | 225225 | 4141M | 2020 |
| Deformable DETR(4scale)[41] | 1212 | 41.141.1 | −- | −- | −- | −- |  | 196196 | 4040M | 24 |
| DAB-DETR(DC5)† [21] | 1212 | 38.038.0 | 60.360.3 | 39.839.8 | 19.219.2 | 40.940.9 | 55.455.4 | 256256 | 4444M | 1717 |
| Dynamic DETR(5scale) [8] | 1212 | 42.942.9 | 61.061.0 | 46.346.3 | 24.624.6 | 44.944.9 | 54.454.4 | −- | 5858M | −- |
| Dynamic Head(5scale) [7] | 1212 | 43.043.0 | 60.760.7 | 46.846.8 | 24.724.7 | 46.446.4 | 53.953.9 | −- | −- | −- |
| HTC(5scale) [4] | 1212 | 42.342.3 | −- | −- | −- | −- | −- | 441441 | 8080M | 5∗5^{*} |
| DN-Deformable-DETR(4scale)† [17] | 1212 | 43.443.4 | 61.961.9 | 47.247.2 | 24.824.8 | 46.846.8 | 59.459.4 | 265265 | 4848M | 2323 |
| DINO-4scale† | 12 | 49.0(+5.6){(+5.6)} | 66.6 | 53.5 | 32.0(+7.2){(+7.2)} | 52.3 | 63.0 | 279279 | 4747M | 24 |
| DINO-5scale† | 12 | 49.4(+6.0){(+6.0)} | 66.9 | 53.8 | 32.3(+7.5){(+7.5)} | 52.5 | 63.9 | 860860 | 4747M | 1010 |

12-epoch setting: With our improved anchor box denoising and training losses, the training process can be significantly accelerated. As shown in Table 1, we compare our method with strong baselines including both convolution-based methods [30, 4, 7] and DETR-like methods [3, 41, 8, 21, 17]. For a fair comparison, we report both GFLOPS and FPS tested on the same A100 NVIDIA GPU for all the models listed in Table 1. All methods except for DETR and DAB-DETR use multi-scale features. For those without multi-scale features, we report their results with ResNet-DC5 which has a better performance for its use of a dilated larger resolution feature map. Since some methods adopt 55 scales of feature maps and some adopt 44, we report our results with both 44 and 55 scales of feature maps.

As shown in Table 1, our method yields an improvement of +5.6+5.6 AP under the same setting using ResNet-50 with 44-scale feature maps and +6.0+6.0 AP with 55-scale feature maps. Our 44-scale model does not introduce much overhead in computation and the number of parameters. Moreover, our method performs especially well for small objects, gaining +7.2+7.2 AP with 44 scales and +7.5+7.5 AP with 55 scales. Note that the results of our models with ResNet-50 backbone are higher than those in the first version of our paper due to engineering techniques. Comparison with the best models with a ResNet-50 backbone:

| Model | Epochs | AP | AP50 | AP75 | APS | APM | APL |
|---|---|---|---|---|---|---|---|
| Faster-RCNN [30] | 108108 | 42.042.0 | 62.462.4 | 44.244.2 | 20.520.5 | 45.845.8 | 61.161.1 |
| DETR(DC5) [41] | 500500 | 43.343.3 | 63.163.1 | 45.945.9 | 22.522.5 | 47.347.3 | 61.161.1 |
| Deformable DETR [41] | 5050 | 46.246.2 | 65.265.2 | 50.050.0 | 28.828.8 | 49.249.2 | 61.761.7 |
| SMCA-R [11] | 5050 | 43.743.7 | 63.663.6 | 47.247.2 | 24.224.2 | 47.047.0 | 60.460.4 |
| TSP-RCNN-R [34] | 9696 | 45.045.0 | 64.564.5 | 49.649.6 | 29.729.7 | 47.747.7 | 58.058.0 |
| Dynamic DETR(5scale) [7] | 5050 | 47.2{47.2} | 65.965.9 | 51.151.1 | 28.628.6 | 49.349.3 | 59.159.1 |
| DAB-Deformable-DETR [21] | 5050 | 46.946.9 | 66.066.0 | 50.850.8 | 30.130.1 | 50.450.4 | 62.562.5 |
| DN-Deformable-DETR [17] | 50 | 48.6 | 67.467.4 | 52.752.7 | 31.031.0 | 52.052.0 | 63.763.7 |
| DINO-4scale | 24 | 50.4(+1.8) | 68.368.3 | 54.854.8 | 33.333.3 | 53.753.7 | 64.864.8 |
| DINO-5scale | 24 | 51.3(+2.7) | 69.169.1 | 56.056.0 | 34.534.5 | 54.254.2 | 65.865.8 |
| DINO-4scale | 36 | 50.9(+2.3) | 69.069.0 | 55.355.3 | 34.634.6 | 54.154.1 | 64.664.6 |
| DINO-5scale | 36 | 51.2(+2.6) | 69.069.0 | 55.855.8 | 35.035.0 | 54.354.3 | 65.365.3 |

To validate the effectiveness of our method in improving both convergence speed and performance, we compare our method with several strong baselines using the same ResNet-50 backbone. Despite the most common 5050-epoch setting, we adopt the 2424 (2×2\times) and 3636 (3×3\times) epoch settings since our method converges faster and yields only a smaller additional gain with 5050-epoch training. The results in Table 2 show that, using only 2424 epochs, our method achieves an improvement of +1.8+1.8 AP and +2.7+2.7 AP with 44 and 55 scales, respectively. Moreover, using 3636 epochs in the 3×3\times setting, the improvement increases to +2.3+2.3 and +2.6+2.6 AP with 44 and 55 scales, respectively. The detailed convergence curve comparison is shown in Fig. 7.

| Method | Params | Backbone Pre-training Dataset | Detection Pre-training Dataset | Use Mask | End-to-end | val2017 (AP) | test-dev (AP) |  |  |
|---|---|---|---|---|---|---|---|---|---|
| w/o TTA | w/ TTA | w/o TTA | w/ TTA |  |  |  |  |  |  |
| SwinL [23] | 284284M | IN-22K-14M | O365 | ✓ |  | 57.157.1 | 58.058.0 | 57.757.7 | 58.758.7 |
| DyHead [7] | ≥284\geq 284M | IN-22K-14M | Unknown* |  |  | −- | 58.458.4 | −- | 60.660.6 |
| Soft Teacher+SwinL [38] | 284284M | IN-22K-14M | O365 | ✓ |  | 60.160.1 | 60.760.7 | −- | 61.361.3 |
| GLIP [18] | ≥284\geq 284M | IN-22K-14M | FourODs [18],GoldG+ [18, 15] |  |  | −- | 60.860.8 | −- | 61.561.5 |
| Florence-CoSwin-H[40] | ≥637\geq 637M | FLD-900M [40] | FLD-9M [40] |  |  | −- | 62.062.0 | −- | 62.462.4 |
| SwinV2-G [22] | 3.03.0B | IN-22K-ext-70M [22] | O365 | ✓ |  | 61.961.9 | 62.562.5 | −- | 63.163.1 |
| DINO-SwinL(Ours) | 218M | IN-22K-14M | O365 |  | ✓ | 63.1 | 63.2 | 63.2 | 63.3 |

### 4.3 Comparison with SOTA Models

To compare with SOTA results, we use the publicly available SwinL [23] backbone pre-trained on ImageNet-22K. We first pre-train DINO on the Objects365 [33] dataset and then fine-tune it on COCO. As shown in Table 3, DINO achieves the best results of 63.263.2AP and 63.363.3AP on COCO val2017 and test-dev, which demonstrate its strong scalability to larger model size and data size. Note that all the previous SOTA models in Table 3 do not use Transformer decoder-based detection heads (HTC++ [4] and DyHead [7]). It is the first time that an end-to-end Transformer detector is established as a SOTA model on the leaderboard [1]. Compared with the previous SOTA models, we use a much smaller model size (1/151/15 parameters compared with SwinV2-G [22]), backbone pre-training data size (1/601/60 images compared with Florence), and detection pre-training data size (1/51/5 images compared with Florence), while achieving better results. In addition, our reported performance without test time augmentation (TTA) is a neat result without bells and whistles. These results effectively show the superior detection performance of DINO compared with traditional detectors.

### 4.4 Ablation

| #Row | QS | CDN | LFT | AP | AP50 | AP75 | APS | APM | APL |
|---|---|---|---|---|---|---|---|---|---|
| 1. DN-DETR [17] | No |  |  | 43.443.4 | 61.961.9 | 47.247.2 | 24.824.8 | 46.846.8 | 59.459.4 |
| 2. Optimized DN-DETR | No |  |  | 44.944.9 | 62.862.8 | 48.648.6 | 26.926.9 | 48.248.2 | 60.060.0 |
| 3. Strong baseline (Row2+pure query selection) | Pure |  |  | 46.546.5 | 64.264.2 | 50.450.4 | 29.629.6 | 49.849.8 | 61.061.0 |
| 4. Row3+mixed query selection | Mixed |  |  | 47.047.0 | 64.264.2 | 51.051.0 | 31.131.1 | 50.150.1 | 61.561.5 |
| 5. Row4+look forward twice | Mixed |  | ✓ | 47.447.4 | 64.864.8 | 51.651.6 | 29.929.9 | 50.850.8 | 61.961.9 |
| 6. DINO (ours, Row5+contrastive DN) | Mixed | ✓ | ✓ | 47.9 | 65.3 | 52.1 | 31.2 | 50.9 | 61.9 |

Effectiveness of New Algorithm Components: To validate the effectiveness of our proposed methods, we build a strong baseline with optimized DN-DETR and pure query selection as described in section 3.1. We include all the pipeline optimization and engineering techniques (see section 4.1 and Appendix 0.D) in the strong baseline. The result of the strong baseline is available in Table 4 Row 3. We also present the result of optimized DN-DETR without pure query selection from Deformable DETR [41] in Table 4 Row 2. While our strong baseline outperforms all previous models, our three new methods in DINO further improve the performance significantly.

## 5 Conclusion

In this paper, we have presented a strong end-to-end Transformer detector DINO with contrastive denoising training, mixed query selection, and look forward twice, which significantly improves both the training efficiency and the final detection performance. As a result, DINO outperforms all previous ResNet-50-based models on COCO val2017 in both the 1212-epoch and the 3636-epoch settings using multi-scale features. Motivated by the improvement, we further explored to train DINO with a stronger backbone on a larger dataset and achieved a new state of the art, 63.363.3 AP on COCO 2017 test-dev. This result establishes DETR-like models as a mainstream detection framework, not only for its novel end-to-end detection optimization, but also for its superior performance.

## Appendix 0.A Test Time Augmentations (TTA)

We aim to build an end-to-end detector that is free from hand-crafted components. However, to compare with traditional detection models, we also explore the use of TTA in DETR-like models. We only use it in our large model with the SwinL backbone. Our TTA does not obtain an inspiring gain compared with traditional detectors, but we hope our exploration may provide some insights for future studies.

We adopt multi-scale test and horizontal flip as TTA. However, the way of ensembling different augmentations in our method is different from that in traditional methods which usually output duplicate boxes. In traditional methods, the ensembling is done by first gathering predictions from all augmentations and ranked by a confidence score. Then, duplicate boxes are found and eliminated by NMS or box voting. The reason why predictions from all augmentations are gathered first is that duplicate boxes appear not only among different augmentations but also within one augmentation. This ensembling method decreases the performance for our method since DETR-like methods are not prone to output duplicate boxes since their set-based prediction loss inhibits duplicate predictions and ensembling may incorrectly remove true positive predictions [3]. To address this issue, we designed a one-to-one ensembling method. Assume we have nn augmentations A​u​g0,A​u​g1,…,A​u​gn−1Aug_{0},Aug_{1},...,Aug_{n-1}, where A​u​giAug_{i} has predictions 𝐎i\mathbf{O}^{i} and a pre-defined hyper-parameter weight wiw^{i}. 𝐎i={(b0i,l0i,s0i),(b1i,l1i,s1i),…,(bm−1i,lm−1i,sm−1i)}\mathbf{O}^{i}=\left\{(b^{i}_{0},l^{i}_{0},s^{i}_{0}),(b^{i}_{1},l^{i}_{1},s^{i}_{1}),...,(b^{i}_{m-1},l^{i}_{m-1},s^{i}_{m-1})\right\} where bji,ljib^{i}_{j},l^{i}_{j} and sjis^{i}_{j} denote the jj-th boundbox, label and score, respectively. We let A​u​g0Aug_{0} be the main augmentation which is the most reliable one. For each prediction in 𝐎0\mathbf{O}^{0}, we select the prediction with the highest I​O​UIOU from predictions of each of other augmentations 𝐎1,…,𝐎n−1\mathbf{O}^{1},...,\mathbf{O}^{n-1} and make sure the I​O​UIOU is higher than a predefined threshold. Finally, we ensemble the selected boxes through weighted average as follows

|  | b=1∑Ii​∑i=on−1Ii​wi​si​d​x​(i)i​bi​d​x​(i)ib=\frac{1}{\sum I^{i}}\sum_{i=o}^{n-1}I^{i}w^{i}s^{i}_{idx(i)}b^{i}_{idx(i)} |  | (3) |
|---|---|---|---|

where Ii=1I^{i}=1 when there is at least one box in 𝐎i\mathbf{O}^{i} with IOU higher than the threshold and Ii=0I^{i}=0 otherwise. i​d​x​(i)idx(i) denotes the index of the selected box in 𝐎i\mathbf{O}^{i}.

## Appendix 0.B Training Efficiency

We provide the GPU memory and training time for our base model in Table 5. All results are reported on 8 Nvidia A100 GPUs with ResNet-50 [14]. The results demonstrate that our models are not only effective but also efficient for training.

| Model | Batch Size per GPU | Traning Time | GPU Mem. | Epoch | AP |
|---|---|---|---|---|---|
| Faster RCNN [30]* | 8 | ∼60\sim 60min/ep | 1313GB | 108108 | 42.042.0 |
| DETR [3] | 8 | ∼16\sim 16min/ep | 2626GB | 300300 | 41.241.2 |
| Deformable DETR [41]⋆ | 2 | ∼55\sim 55min/ep | 1616GB | 5050 | 45.445.4 |
| DINO(Ours) | 2 | ∼55\sim 55min/ep | 1616GB | 12 | 47.9 |

## Appendix 0.C Additional Analysis on our Model Components

| # Encoder/Decoder | 6/66/6 | 4/64/6 | 3/63/6 | 2/62/6 | 6/4 | 6/2 | 2/42/4 | 2/22/2 |
|---|---|---|---|---|---|---|---|---|
| AP | 47.447.4 | 46.246.2 | 45.845.8 | 45.445.4 | 46.046.0 | 44.444.4 | 44.144.1 | 41.241.2 |

Analysis on the Number of Encoder and Decoder Layers: We also investigate the influence of varying numbers of encoder and decoder layers. As shown in Table 6, decreasing the number of decoder layers hurts the performance more significantly. For example, using the same 66 encoder layers while decreasing the number of decoder layers from 66 to 22 leads to a 3.03.0 AP drop. This performance drop is expected as the boxes are dynamically updated and refined through each decoder layer to get the final results. Moreover, we also observe that compared with other DETR-like models like Dynamic DETR [7] whose performance drops by 13.813.8AP (29.129.1 vs 42.942.9) when decreasing the number of decoder layers to 22, the performance drop of DINO is much smaller. This is because our mixed query selection approach feeds the selected boxes from the encoder to enhance the decoder queries. Therefore, the decoder queries are well initialized and not deeply coupled with decoder layer refinement.

| # Denoising query | 100 CDN | 1000 DN | 200 DN | 100 DN | 50 DN | 10 DN | No DN |
|---|---|---|---|---|---|---|---|
| AP | 47.947.9 | 47.647.6 | 47.447.4 | 47.447.4 | 46.746.7 | 46.046.0 | 45.145.1 |

Analysis on Query Denoising: We continue to investigate the influence of query denoising by varying the number of denoising queries. We use the optimized dynamic denoising group (detailed in Appendix 0.D.1). As shown in Table 7, when we use less than 100100 denoising queries, increasing the number can lead to a significant performance improvement. However, continuing to increase the DN number after 100100 yields only a small additional or even worse performance improvement. We also analysis the effect of the number of encoder and decoder Layers in Appendix 0.C.

## Appendix 0.D More Implementation Details

### 0.D.1 Dynamic DN groups

In DN-DETR, all the GT objects (label+box) in one image are collected as one GT group for denoising. To improve the DN training efficiency, multiple noised versions of the GT group in an image are used during training. In DN-DETR, the number of groups is set to five or ten according to different model sizes. As DETR-like models adopt mini-batch training, the total number of DN queries for each image in one batch is padded to the largest one in the batch. Considering that the number of objects in one image in COCO dataset ranges from 11 to 8080, this design is inefficient and results in excessive memory consumption. To address this problem, we propose to fix the number of DN queries and dynamically adjust the number of groups for each image according to its number of objects.

### 0.D.2 Large-Scale Model Pre-trianing

Objects365 [33] is a large-scale detection data set with over 1.7​M1.7M annotated images for training and 80,00080,000 annotated images for validation. To use the data more efficiently, We select the first 5,0005,000 out of 80,00080,000 validation images as our validation set and add the others to training. We pre-train DINO on Objects365 for 2626 epochs using 6464 Nvidia A100 GPUs and fine-tune the model on COCO for 1818 epochs using 1616 Nvidia A100 GPUS. Each GPU has a local batch size of 11 image only. In the fine-tuning stage, we enlarge the image size to 1.5×1.5\times (i.e., with max size 1200×20001200\times 2000). This adds around 0.50.5 AP to the final result. To reduce the GPU memory usage, we leverage checkpointing [6] and mixed precision [26] during training. Moreover, we use 10001000 DN queries for this large model.

### 0.D.3 Other Implementation Details

For hyper-parameters, as in DN-DETR, we use a 66-layer Transformer encoder and a 66-layer Transformer decoder and 256256 as the hidden feature dimension. We set the initial learning rate (lr) as 1×10−41\times 10^{-4} and adopt a simple lr scheduler, which drops lr at the 1111-th, 2020-th, and 3030-th epoch by multiplying 0.1 for the 1212, 2424, and 3636 epoch settings with RestNet50, respectively. We use the AdamW [16, 24] optimizer with weight decay of 1×10−41\times 10^{-4} and train our model on Nvidia A100 GPUs with batch size 1616. Since DN-DETR [17] adopts 300300 decoder queries and 33 patterns [37], we use 300×3=900300\times 3=900 decoder queries with the same computation cost. Learning schedules of our DINO with SwinL are available in the appendix.

We use the L1 loss and GIOU [32] loss for box regression and focal loss [19] with α=0.25,γ=2\alpha=0.25,\gamma=2 for classification. As in DETR [3], we add auxiliary losses after each decoder layer. Similar to Deformable DETR [41], we add extra intermediate losses after the query selection module, with the same components as for each decoder layer. We use the same loss coefficients as in DAB-DETR [21] and DN-DETR [17], that is, 1.01.0 for classification loss, 5.05.0 for L1 loss, and 2.02.0 for GIOU loss.

We also optimize the detection pipeline used in DAB-DETR [21] and DN-DETR [17]. Following DN-Deformable-DETR [17], we use the same multi-scale approach as in Deformable DETR [41] and adopt the deformable attention. DN-DETR uses different prediction heads with unshared parameters in different decoder layers. In addition, we introduce dynamic denoising group to increase denoising training efficiency and alleviate memory overhead (see Appendix 0.D.1). In this work, we find that using a shared prediction head will add additional performance improvement. This also leads to a reduction of about one million parameters. In addition, we find the conditional queries [25] used in DAB-DETR does not suit our model and we do not include them in our final model.

We use the same random crop and scale augmentation during training following DETR [3]. For example, we randomly resize an input image with its shorter side between 480480 and 800800 pixels and its longer side at most 13331333. For DINO with SwinL, we pre-train the model using the default setting, but finetune using 1.5×1.5\times larger scale (shorter side between 720720 and 12001200 pixels and longer side at most 20002000 pixels) to compare with models on the leaderboard [1]. Without using any other tricks, we achieve the result of 63.163.1 on val2017 and 63.263.2 on test-dev without test time augmentation (TTA) (see Appendix 0.A), outperforming the previous state-of-the-art result 63.163.1 achieved by SwinV2 [22] with a much neater solution.

For our 44-scale models, we extract features from stages 22, 33, and 44 of the backbone and add an extra feature by down-sampling the output of the stage 44. An additional feature map of the backbone stage 11 is used for our 55-scale models. For hyper-parameters, we set λ1=1.0\lambda_{1}=1.0 and λ2=2.0\lambda_{2}=2.0 and use 100100 CDN pairs which contain 100100 positive queries and 100100 negative queries.

### 0.D.4 Detailed Hyper-parameters

We list the hyper-parameters for those who want to reproduce our results in Table 8.

| Item | Value |
|---|---|
| lr | 0.0001 |
| lr_backbone | 1e-05 |
| weight_decay | 0.0001 |
| clip_max_norm | 0.1 |
| pe_temperature | 20 |
| enc_layers | 6 |
| dec_layers | 6 |
| dim_feedforward | 2048 |
| hidden_dim | 256 |
| dropout | 0.0 |
| nheads | 8 |
| num_queries | 900 |
| enc_n_points | 4 |
| dec_n_points | 4 |
| transformer_activation | “relu” |
| batch_norm_type | “FrozenBatchNorm2d” |
| set_cost_class | 2.0 |
| set_cost_bbox | 5.0 |
| set_cost_giou | 2.0 |
| cls_loss_coef | 1.0 |
| bbox_loss_coef | 5.0 |
| giou_loss_coef | 2.0 |
| focal_alpha | 0.25 |
| dn_box_noise_scale | 0.4 |
| dn_label_noise_ratio | 0.5 |
