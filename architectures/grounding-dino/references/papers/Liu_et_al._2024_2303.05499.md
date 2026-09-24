## Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

Shilong Liu 1,2*⋆*, Zhaoyang Zeng², Tianhe Ren², Feng Li 2, 3, Hao Zhang 2, 3, Jie Yang 2, 4, Qing Jiang 2, 6 Chunyuan Li⁵, Jianwei Yang⁵, Hang Su¹, Jun Zhu¹ *⋆⋆*, Lei Zhang² *⋆⋆*.

1 Dept. of Comp. Sci. and Tech., BNRist Center, State Key Lab for Intell. Tech. & Sys., Institute for AI, Tsinghua-Bosch Joint Center for ML, Tsinghua University 2 International Digital Economy Academy (IDEA) 3 The Hong Kong University of Science and Technology 4 The Chinese University of Hong Kong (Shenzhen) 5 Microsoft Research, Redmond 6 South China University of Technology liusl20@mails.tsinghua.edu.cn, leizhang@idea.edu.cn

Object localization Text understanding Collaborate with stable diffusion. COCO pre-defined categories Human-input novel categories Human-input reference sentences

bench ear, lion, bench The left lion Prompt (modify detected objects): Dog

person worldcup The bottom man with his head up Prompt (modify background): All people Standard Object Detection Zero-Shot Transfer to Referring Object Detection around the world cheer with a worldcup. Novel Categories (Referring Expression Comprehension)

**(a) Closed-Set Object Detection (b) Open-Set Object Detection (c) Application: Image Editing**
Fig. 1: (a) Closed-set object detection requires models to detect objects of pre-defined

categories. (b) We evaluate models on novel objects and standard Referring expression comprehension (REC) benchmarks for model generalizations on novel objects with attributes. (c) We present an image editing application by combining Grounding DINO and Stable Diffusion [41]. Best viewed in colors.

Abstract. In this paper, we develop an open-set object detector, called Grounding DINO, by marrying Transformer-based detector DINO with grounded pre-training, which can detect arbitrary objects with human

# arXiv:2303.05499v5 [cs.CV] 19 Jul 2024

inputs such as category names or referring expressions. The key solution of open-set object detection is introducing language to a closed-set detector for open-set concept generalization. To effectively fuse language and vision modalities, we conceptually divide a closed-set detector into three phases and propose a tight fusion solution, which includes a feature enhancer, a language-guided query selection, and a cross-modality decoder for modal- ities fusion. We first pre-train Grounding DINO on large-scale datasets, *⋆* This work was done when Shilong Liu, Feng Li, Hao Zhang, Jie Yang, and Qing Jiang were interns at IDEA. *⋆⋆* Corresponding authors.

2 S. Liu et al.

including object detection data, grounding data, and caption data, and evaluate the model on both open-set object detection and referring object detection benchmarks. Grounding DINO performs remarkably well on all three settings, including benchmarks on COCO, LVIS, ODinW, and RefCOCO/+/g. Grounding DINO achieves a 52*.*5 AP on the COCO zero- shot¹ detection benchmark. It sets a new record on the ODinW zero-shot benchmark with a mean 26*.*1 AP. We release some checkpoints and infer- ence codes at [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).

Keywords: Object Detection · Image Grounding · Multi-modal learning

### 1 Introduction

A key indicator of an Artificial General Intelligence (AGI) system’s capability is its proficiency in handling open-world scenarios. In this paper, we aim to develop a strong system to detect arbitrary objects specified by human language inputs, a task commonly referred to as *open-set object detection²*. The task has wide applications for its great potential as a generic object detector. For example, we can cooperate with generative models for image editing (as shown in Fig. 1 (b)). In pursuit of this goal, we design the strong open-set object detector Ground- ing DINO by following the two principles: tight modality fusion based on DINO [57] and large-scale grounded pre-train for concept generalization. Tight modality fusion based on DINO. The key to open-set detection is introducing language for unseen object generalization [1,7,25]. Most existing open-set detectors are developed by extending closed-set detectors to open-set scenarios with language information. As shown in Fig. 2, a closed-set detector typically has three important modules, a backbone for feature extraction, a neck for feature enhancement, and a head for region refinement (or box prediction). A closed-set detector can be generalized to detect novel objects by learning language-aware region embeddings so that each region can be classified into novel categories in a language-aware semantic space. The key to achieving this goal is using contrastive loss between region outputs and language features at the neck and/or head outputs.Text Features **Open-Set** To help a model **Detector** Text Encoder align cross-modality Feature Contrastive Feature Feature Contrastive information some work Fusion A Loss A Fusion B Fusion C Loss B

tried to fuse features(ResNet,Swin Backbone,…) (DyHead,Encoder Neck ,…) (ROIHeadHead, Decoder, …) before the final loss **Closed-Set** Image Refined Image Query Init Regions Output **Detector** Features Features stage. We summarize the modulized design Fig. 2: Extending closed-set detectors to open-set scenarios.

1 In this paper, ‘zero-shot’ refers to scenarios where the training split of the test dataset is not utilized in the training process. 2 We view the terms *open-set object detection*, *open-world object detection*, and *open-* *vocabulary object detection* the same task in this paper. To avoid confusion, we always use *open-set object detection* in our paper.

Grounding DINO 3

of object detectors in Fig. 2. Feature fusion can be performed in three phases: neck (phase A), query initialization (phase B), and head (phase C). For example, GLIP [25] performs early fusion in the neck module (phase A), and OV-DETR [55] uses language-aware queries as head inputs (phase B). We argue that introducing more feature fusion into the pipeline can facilitate better alignment between different modality features, thereby achieving better performance. Although conceptually simple, it is hard for previous work to perform feature fusion in all three phases. The design of classical detectors like Faster RCNN makes it hard to interact with language information in most blocks. Unlike classical detectors, the Transformer-based detector method such as DINO has a consistent structure with language blocks. The layer-by-layer design enables it to interact with language information easily. Under this principle, we design three feature fusion approaches in the neck, query initialization, and head phases. More specifically, we design a feature enhancer by stacking self-attention, text-to-image cross-attention, and image-to-text cross-attention as the neck module. We then develop a language-guided query selection method to initialize queries for the detection head. We also design a cross-modality decoder for the head phase with image and text cross-attention layers to boost query representations. Large-scale grounded pre-train for zero-shot transfer. Most existing open-set models [14,21] rely on pre-trained CLIP models for concept generaliza- tion. Nevertheless, the efficacy of CLIP, specifically pre-trained on image-text pairs, is limited for region-text pair detection tasks, as identified in the Region- CLIP study by RegionCLIP [61]. In contrast, GLIP [25] presents a different way by reformulating object detection as a phrase grounding task and introducing con- trastive training between object regions and language phrases on large-scale data. It shows great flexibility for heterogeneous datasets and remarkable performance on closed-set and open-set detection. We have adopted and refined the grounded training methodology. GLIP’s approach involves concatenating all categories into a sentence in a random order. However, the direct category names concatenation does not consider the potential influence of unrelated categories on each other when extracting features. To mitigate this issue and improve model performance during grounded training, we introduce a technique that utilizes sub-sentence level text features. It removes the attention between unrelated categories during word feature extractions. Further elaboration on this technique can be found in Section 3.4. We pre-train the Grounding DINO on a large-scale dataset and evaluate the performance on mainstream object detection benchmarks like COCO [29]. While some studies have examined open-set detection models under a "partial label" framework—training on a subset of data (e.g., base categories) and testing on additional categories—we advocate for a fully zero-shot approach to enhance practical applicability. Moreover, we extend the model to another important scenario Referring Expression Comprehension (REC) [30,34] 3, where objects are described with attributes.

3 We use the term *Referring Expression Comprehension (REC)* and *Referring (Object)* *Detection* exchangeable in this paper.

We conduct experiments on all three settings, including closed-set detection, open-set detection, and referring object detection, as shown in Fig. 1, to compre- hensively evaluate open-set detection performance. Grounding DINO outperforms competitors by a large margin. For example, Grounding DINO reaches a 52*.*5 AP on COCO minival without any COCO training data. It also establishes a new state of the art on the ODinW [23] zero-shot benchmark with a 26*.*1 mean AP.

Model Base DetectorModel Design Fusion (Fig. 2) CLIP ViLD [14] Mask R-CNN-✓ RegionCLIP [61] Faster RCNN-✓ FindIt [21] Faster RCNN A MDETR [18] DETR A,C DQ-DETR [45] DETR A,C GLIP [25] DyHead A GLIPv2 [58] DyHead A OV-DETR [55] Deformable DETR B ✓ OWL-ViT [35]--✓ DetCLIP [52] ATSS-✓ OmDet [60] Sparse R-CNN C ✓ Grounding DINO (Ours) DINO A,B,C

Represent. Level (Sec. 3.4) Text Prompt Closed-Set Settings COCO COCO Zero-Shot Transfer LVIS ODinW Referring Detection RefCOCO/+/g sentence ✓ partial label partial label sentence ✓ partial label partial label sentence ✓ partial label fine-tune word fine-tune zero-shot fine-tune word ✓ zero-shot fine-tune word ✓ zero-shot zero-shot zero-shot word ✓ zero-shot zero-shot zero-shot sentence ✓ partial label partial label sentence ✓ partial label partial label zero-shot sentence zero-shot zero-shot sentence ✓ zero-shot sub-sentence ✓ zero-shot zero-shot zero-shot zero-shot

and evaluated on other cases. [56]

### 2 Related Work

Table 1: A comparison of previous open-set object detectors. Our summarization is

based on the experiments in their paper, but not the ability to extend their models to other tasks. It is worth noting that some related works may not (only) be designed for the open-set object detection initially, like MDETR [18] and GLIPv2 [58], but we list them here for a comprehensive comparison with existing work. We use the term “partial label” for the settings, where models are trained on partial data (e.g. base categories)

Detection Transformers. Grounding DINO is built upon the DETR-like model DINO [57], which is an end-to-end Transformer-based detector. DETR was first proposed in [2] and then has been improved from many directions [4,5,13,17,33,48,64] in the past few years. DAB-DETR [31] introduces anchor boxes as DETR queries for more accurate box prediction. DN-DETR [24] proposes a query-denoising approach to stabilizing the bipartite matching. DINO [57] further develops several techniques including contrastive de-noising and sets a new record on the COCO object detection benchmark. However, such detectors mainly focus on closed-set detection and are difficult to generalize to novel classes because of the limited pre-defined categories. Open-Set Object Detection. Open-set object detection is trained using existing bounding box annotations and aims at detecting arbitrary classes with the help of language generalization. OV-DETR [56] uses image and text embedding encoded by a CLIP model as queries to decode the category-specified boxes in the DETR framework [2]. ViLD [14] distills knowledge from a CLIP teacher model into a R-CNN-like detector so that the learned region embeddings contain the semantics of language. GLIP [12] formulates object detection as a grounding problem and leverages additional grounding data to help learn aligned semantics

Contrastive loss Localization loss A cat sets on a table. cat. 1 Text 1 person. mouse. Features 1 1 1

Model Outputs Keys& Values Cross-Modality Decoder Cross-Modality Queries Language-guide Query Selection Text Image Features Features

Feature Enhancer Vanilla Text Vanilla Image Features Features Text Image Backbone Backbone

cat. person. mouse. A cat sets on a table.

Input Text Input Image

Updated A Cross-Modality Decoder Layer CrossQuery-Modality

Text Features FFN K,V Text Cross -Attention Q K,V Image Cross-Attention Q

Image Features

Self-Attention Q,K,V Cross-Modality Query

#### help extend the generalization ability.

Table 1.

### 3 Grounding DINO

Updated Text Updated Image Features Features

FFN FFN Text-to-image Cross-Attention Q K,V Image-to-text Cross-Attention K,V Q Self-Attention Deformable Q,K,V Self-Attention Q,K,V Text Image Features Features

Fig. 3: The framework of Grounding DINO. We present the overall framework, a feature

enhancer layer, and a decoder layer in block 1, block 2, and block 3, respectively.

at phrase and region levels. It shows that such a formulation can even achieve stronger performance on fully-supervised detection benchmarks. DetCLIP [52] involves large-scale image captioning datasets and uses the generated pseudo labels to expand the knowledge database. The generated pseudo labels effectively

However, previous works only fuse multi-modal information in partial phases, which may lead to sub-optimal language generalization ability. For example, GLIP only considers fusion in the feature enhancement (phase A) and OV-DETR only injects language information at the decoder inputs (phase B). Moreover, the REC task is normally overlooked in evaluation, which is an important scenario for open-set detection. We compare our model with other open-set methods in

Grounding DINO outputs multiple pairs of object boxes and noun phrases for a given (Image, Text) pair. For example, as shown in Fig. 3, the model locates a cat and a table from the input image and extracts word cat and table from the input text as corresponding labels. Both object detection and REC tasks can be aligned with the pipeline. Following GLIP [25], we concatenate all category names as input texts for object detection tasks. REC requires a bounding box for

each text input. We use the output object with the largest scores as the output for the REC task. Grounding DINO is a dual-encoder-single-decoder architecture. It contains an image backbone for image feature extraction, a text backbone for text feature extraction, a feature enhancer for image and text feature fusion (Sec. 3.1), a language-guided query selection module for query initialization (Sec. 3.2), and a cross-modality decoder for box refinement (Sec. 3.3). For each (Image, Text) pair, we first extract vanilla image features and vanilla text features using an image backbone and a text backbone, respectively. The two vanilla features are fed into a feature enhancer module for cross-modality feature fusion. After obtaining cross-modality text and image features, we use a language-guided query selection module to select cross-modality queries from image features. Like the object queries in most DETR-like models, these cross- modality queries will be fed into a cross-modality decoder to probe desired features from the two modal features and update themselves. The output queries of the last decoder layer will be used to predict object boxes and extract corresponding phrases.

Per-sentence features Per-word features Per-word features

Encoder Encoder Encoder Encoder Encoder Encoder cat. baseball glove. A cat is sleeping on a table. cat. baseball glove. A cat is sleeping on a table. cat. baseball glove. A cat is sleeping on a table.

|cat. baseball glove. A cat is sleeping on a table.|||
|---|---|---|
|(a) Sentence level|(b) Word level|(c) Sub-sentence level|

Fig. 4: Comparisons of text representations.

3.1 Feature Extraction and Enhancer Given an (Image, Text) pair, we extract multi-scale image features with an image backbone like Swin Transformer [32], and text features with a text backbone like BERT [8]. Following previous DETR-like detectors [57,64], multi-scale features are extracted from the outputs of different blocks. After extracting vanilla image and text features, we fed them into a feature enhancer for cross-modality feature fusion. The feature enhancer includes multiple feature enhancer layers. We illustrate a feature enhancer layer in Fig. 3 block 2. We leverage the Deformable self-attention to enhance image features and the vanilla self-attention for text feature enhancers. Inspired by GLIP [25], we add an image-to-text and a text- to-image cross-attention modules for feature fusion. These modules help align features of different modalities.
3.2 Language-Guided Query Selection Grounding DINO aims to detect objects from an image specified by an input text. To effectively leverage the input text to guide object detection, we design a

language-guided query selection module to select features that are more relevant to the input text as decoder queries. Let’s denote the image feature as X*I∈* R *NI ×d* and the text features as X*T∈* R *NT ×d*. Here, *NI* represents the number of image tokens, *NT* indicates the number of text tokens, and *d* corresponds to the feature dimension. In our experiments, we specifically utilize a feature dimension of *d* = 256. Typically, in our models, the value of *NI* exceeds 10*,*000, while *NT* remains below 256. Our objective is to extract *Nq*queries from the encoder’s image features to be used as inputs for the decoder. In alignment with the DINO method, we set *Nq*to be

900. The top *Nq*query indices for the image feature, denoted as I*Nq*, are selected using the following expression:
I *Nq*= Top*Nq*(Max (*−*1) (X*I*X ⊺

*T*))*.* (1)
In this expression, Top*N* *q* represents the operation to pick the top *Nq*indices. The function Max (*−*1) executes the max operation along the *−*1 dimension, and the symbol ⊺ denotes matrix transposition. We present the query selection process in Algorithm 1 in PyTorch style. The language-guided query selection module outputs *Nq*indices. We can extract features based on the selected indices to initialize queries. Following DINO [57], we use mixed query selection to initialize decoder queries. Each decoder query contains two parts: content part and positional part [33], respectively. We formulate the positional part as dynamic anchor boxes [31], which are initialized with encoder outputs. The other part, the content queries, are set to be learnable during training.

3.3 Cross-Modality Decoder We develop a cross-modality decoder to combine image and text modality features, as shown in Fig. 3 block 3. Each cross-modality query is fed into a self-attention layer, an image cross-attention layer to combine image features, a text cross- attention layer to combine text features, and an FFN layer in each cross-modality decoder layer. Each decoder layer has an extra text cross-attention layer compared with the DINO decoder layer, as we need to inject text information into queries for better modality alignment.
3.4 Sub-Sentence Level Text Feature Two kinds of text prompts are explored in previous works, which we named as sentence level representation and word level representation, as shown in Fig. 4. Sentence level representation [35,52] encodes a whole sentence to one feature. If some sentences in phrase grounding data have multiple phrases, it extracts these phrases and discards other words. In this way, it removes the influence between words while losing fine-grained information in sentences. Word level representation [12,18] enables encoding multiple category names with one forward but introduces unnecessary dependencies among categories, especially when the input text is a concatenation of multiple category names in an arbitrary order.

|Grounding DINO T (Ours)|Swin-T|O365|46.7|56.9 / -|
|---|---|---|---|---|
|Grounding DINO T (Ours)|Swin-T|O365,GoldG|48.1|57.1 / -|
|Grounding DINO T (Ours)|Swin-T|O365,GoldG,Cap4M|48.4|57.2 / -|
|Grounding DINO L (Ours)|Swin-L|O365,OI [19],GoldG|52.5|62.6 / 62.7 (63.0 / 63.0)*|
|Grounding DINO L (Ours)|Swin-L|O365,OI,GoldG,Cap4M,COCO,RefC|60.7‡|62.6 / -|

|Model|Backbone|Pre-Training Data|Zero-Shot|Fine-Tuning|
|---|---|---|---|---|
||||2017val|2017val/test-dev|
|Faster R-CNN|RN50-FPN|-|-|40.2 / -|
|Faster R-CNN|RN101-FPN|-|-|42.0 / -|
|DyHead-T [5]|Swin-T|-|-|49.7 / -|
|DyHead-L [5]|Swin-L|-|-|58.4 / 58.7|
|DyHead-L [5]|Swin-L|O365,ImageNet21K|-|60.3 / 60.6|
|SoftTeacher [50]|Swin-L|O365,SS-COCO|-|60.7 / 61.3|
|DINO(Swin-L) [57]|Swin-L|O365|-|62.5 / -|
|DyHead-T† [5]|Swin-T|O365|43.6|53.3 / -|
|GLIP-T (B) [25]|Swin-T|O365|44.9|53.8 / -|
|GLIP-T (C) [25]|Swin-T|O365,GoldG|46.7|55.1 / -|
|GLIP-L [25]|Swin-L|FourODs,GoldG,Cap24M|49.8|60.8 / 61.0|
|DINO(Swin-T)† [57]|Swin-T|O365|46.2|56.9 / -|

Table 2: Zero-shot domain transfer and fine-tuning on COCO. * The results in brackets
 are trained with 1.5*×* image sizes, i.e., with a maximum image size of 2000. †The models map a subset of O365 categories to COCO for zero-shot evaluations. ‡This is not a real zero-shot performance as we add COCO data during model training. We list the result here for a reference. As shown in Fig. 4 (b), some unrelated words interact during attention. To avoid unwanted word interactions, we introduce attention masks to block attentions among unrelated category names, named “sub-sentence” level representation. It eliminates the influence between different category names while keeping per-word features for fine-grained understanding.
3.5 Loss Function Following previous DETR-like works [2,24,31,33,57,64], we use the L1 loss and the GIOU [40] loss for bounding box regressions. We follow GLIP [25] and use contrastive loss between predicted objects and language tokens for classification. Specifically, we dot product each query with text features to predict logits for each text token and then compute focal loss [28] for each logit. Box regression and classification costs are first used for bipartite matching between predictions and ground truths. We then calculate final losses between ground truths and matched predictions with the same loss components. Following DETR-like models, we add auxiliary loss after each decoder layer and after the encoder outputs.
### 4 Experiments

We conduct extensive experiments on three settings: a closed-set setting on the COCO detection benchmark (Sec. C.2), an open-set setting on zero-shot COCO, LVIS, and ODinW (Sec. 4.2), and a referring detection setting on RefCOCO/+/g (Sec. 4.3). Ablations are then conducted to show the effectiveness of our model design (Sec. 4.5). We also explore a way to transfer a well-trained DINO to the

open-set scenario by training a few plug-in modules in Sec. C.1. The test of our model efficiency is presented in Sec. C.4.

4.1 Implementation Details We trained two model variants, Grounding DINO T with Swin-T [32], and Grounding DINO L with Swin-L [32] as an image backbone, respectively. We leveraged BERT-base [8] from Hugging Face [49] as text backbones. As we focus more on the model performance on novel classes, we list zero-shot transfer and referring detection results in the main text. By default, we use 900 queries in our model following DINO. We set the maximum text token number as 256. Using BERT as our text encoder, we follow BERT to tokenize texts with a BPE scheme [42]. We use six feature enhancer layers in the feature enhancer module. The cross-modality decoder is composed of six decoder layers as well. We leverage deformable attention [64] in image cross-attention layers. Both matching costs and final losses include classification losses (or contrastive losses), box L1 losses, and GIOU [40] losses. Following DINO, we set the weight of classification costs, box L1 costs, and GIOU costs as 2.0, 5.0, and 2.0, respectively, during Hungarian matching. The corresponding loss weights are 1.0, 5.0, and 2.0 in the final loss calculation. Our Swin Transformer Tiny models are trained on 16 Nvidia V100 GPUs with a total batch size of 32. We extract three image feature scales, from 8*×* to 32*×*. It is named “4scale” in DINO since we downsample the 32*×* feature map to 64*×* as an extra feature scale. For the model with Swin Transformer Large, we extract four image feature scales from backbones, from 4*×* to 32*×*. The model is trained on 64 Nvidia A100 GPUs with a total batch size of 64.
4.2 Zero-Shot Transfer of Grounding DINO In this setting, we pre-train models on large-scale datasets and directly evaluate models on new datasets. We also list some fine-tuned results for a more thorough comparison of our model with prior works. *COCO Benchmark* We compare Grounding DINO with GLIP and DINO in
Table 2. We pre-train models on large-scale datasets and directly evaluate our
 model on the COCO benchmark. As the O365 dataset [43] has (nearly⁴) covered all categories in COCO, we evaluate an O365 pre-trined DINO on COCO as a zero-shot baseline. The result shows that DINO performs better on the COCO zero-shot transfer than DyHead. Grounding DINO outperforms all previous models on the zero-shot transfer setting, with +0*.*5AP and +1*.*8AP compared with DINO and GLIP under the same setting. Grounding data is still helpful for Grounding DINO, introducing more than 1AP (48.1 vs. 46.7) on the zero-shot 4 It is not an exact mapping between O365 and COCO categories. We made some approximations during evaluation.

transfer setting. With stronger backbones and larger data, Grounding DINO sets a new record of 52*.*5 AP on the COCO object detection benchmark without seeing any COCO images during training. Grounding DINO obtains a 62*.*6 AP on COCO minival, outperforming DINO’s 62*.*5 AP. When enlarging the input images by 1*.*5*×*, the benefits reduce. We suspect that the text branch enlarges the gap between models with different input images. Even though the performance plateaus with larger input size, Grounding DINO gets an impressive 63*.*0 AP on COCO test-dev with fine-tuning on the COCO dataset(See the number in brackets of Table 2).

*LVIS Benchmark* LVIS [15] is a dataset for long-tail objects. It contains more than 1000 categories for evaluation. We use LVIS as a downstream task to test the zero-shot abilities of our model. We use GLIP and DetCLIPv2 as baselines for our models. The results are shown in Table 3.

|Grounding DINO T|Swin-T|O365,GoldG|25.6|14.4/19.6/32.2|
|---|---|---|---|---|
|Grounding DINO T|Swin-T|O365,GoldG,Cap4M|27.4|18.1/23.3/32.7|
|Grounding DINO L|Swin-L|O365,OI,GoldG,Cap4M, COCO,RefC|33.9|22.2/30.7/38.8|

|Model|Backbone|Pre-Training Data|AP MiniVal [18] APr/APc/APf|
|---|---|---|---|
|||Zero-Shot Setting||
|GLIP-T (C)|Swin-T|O365,GoldG|24.9 17.7/19.5/31.0|
|GLIP-T|Swin-T|O365,GoldG,Cap4M|26.0 20.8/21.4/31.0|
|DetCLIPv2|Swin-T|O365,GoldG,CC15M|40.4 36.0/41.7/40.0|

*Fine-Tune Setting* MDETR RN101 GoldG,RefC 24.2 20.9/24.9/24.3

|MDETR|RN101|GoldG,RefC|24.2 20.9/24.9/24.3|
|---|---|---|---|
|Mask R-CNN|RN101|-|33.3 26.3/34.0/33.9|
|DetCLIPv2 [51]|Swin-T|O365,GoldG,CC15M|50.7 44.3/52.4/50.3|
|Grounding DINO T|Swin-T|O365,GoldG|52.1 35.4/51.3/55.7|

Table 3: Model results on LVIS.

We found two interesting phenomena in the results. First, Grounding DINO works better than common objects than GLIP, but worse on rare categories. We reviewed DETR-like models on LVIS and noted these models often exhibit lower rare category AP despite similar overall AP, like Table 2 of [9] and Table 6 of [18]. To our knowledge, no existing DETR-like models effectively address the rarity challenge in LVIS without extra training data, which may be a characteristic limitation of the architecture. The other phenomenon is that Grounding DINO has larger gains with more data than GLIP. For example, Grounding DINO introduces +1*.*8 AP gains with the caption data Cap4M, whereas GLIP has only +1*.*1 AP. We believe that Grounding DINO has better scalability compared with GLIP. A larger-scale training will be left as our future work. Although achieving better results than GLIP, we found that Grounding DINO is inferior to DetCLIPv2, which is trained on a larger scale data. This performance difference might be attributed to the disparity in data distribution between the training dataset and the LVIS dataset.

To unveil the full potential of Grounding DINO, we fine-tuned it on the LVIS dataset. Table 3 highlights the commendable capability of our model. Remarkably, despite being pre-trained only on the O365 and GoldG datasets, Grounding DINO outperforms DetCLIPv2-T by a margin of 1*.*5 AP. This result shows that Grounding DINO might have learned a better object-level representation which helps yield a better performance after fine-tuning (aligning with the target dataset). In our future work, we will perform more studies, including varying the semantic concept coverage of the training data and increasing the scale of the training data, to further improve the zero-shot generalization performance.

Test Model Language Input Backbone Model Size Pre-Training Data AP*average* AP*median* *Zero-Shot Setting* MDETR [18] ✓ ENB5 [46] 169M GoldG,RefC 10.7 3.0 OWL-ViT [35] ✓ ViT L/14(CLIP) *>*1243M O365, VG 18.8 9.8 GLIP-T [25] ✓ Swin-T 232M O365,GoldG,Cap4M 19.6 5.1 OmDet [60] ✓ ConvNeXt-B 230M COCO,O365,LVIS,PhraseCut 19.7 10.8 GLIPv2-T [59] ✓ Swin-T 232M O365,GoldG,Cap4M 22.3 8.9 DetCLIP [52] ✓ Swin-L 267M O365,GoldG,YFCC1M 24.9 18.3 Florence [54] ✓ CoSwinH *≈*841M FLD900M,O365,GoldG 25.8 14.3

|Grounding DINO T(Ours)|✓|Swin-T|172M|O365,GoldG|20.0|9.5|
|---|---|---|---|---|---|---|
|Grounding DINO T(Ours)|✓|Swin-T|172M|O365,GoldG,Cap4M|22.3|11.9|
|Grounding DINO L(Ours)|✓|Swin-L|341M|O365,OI,GoldG,Cap4M,COCO,RefC|26.1|18.4|

DyHead-T [5] GLIP-T [25] DINO-Swin-T [57] OmDet [60] Grounding DINO T(Ours)

GLIP-T [25] DyHead-T [5] DINO-Swin-T [57] OmDet [60] DINO-Swin-L [57] Grounding DINO T(Ours)

✗ ✓ ✗ ✓ ✓

✓ ✗ ✗ ✓ ✗ ✓

*Few-Shot Setting* Swin-T *≈*100M Swin-T 232M Swin-T 49M ConvNeXt-B 230M Swin-T 172M *Full-Shot Setting* Swin-T 232M Swin-T *≈*100M Swin-T 49M ConvNeXt-B 230M Swin-L 218M Swin-T 172M

O365 37.5 36.7 O365,GoldG,Cap4M 38.9 33.7 O365 41.2 41.1 COCO,O365,LVIS,PhraseCut 42.4 41.7 O365,GoldG 46.4 51.1

O365,GoldG,Cap4M 62.6 62.1 O365 63.2 64.9 O365 66.7 68.5 COCO,O365,LVIS,PhraseCut 67.1 71.2 O365 68.8 70.7 O365,GoldG 70.7 76.2

setting.

Table 4: Model results on the ODinW benchmark.

*ODinW Benchmark* ODinW (Object Detection in the Wild) [23] is a more challenging benchmark to test model performance under real-world scenarios. It collects more than 35 datasets for evaluation. We report three settings, zero-shot, few-shot, and full-shot results in Table 4. Grounding DINO performs well on this benchmark. With only O365 and GoldG for pre-train, Grounding DINO T outperforms DINO on few-shot and full-shot settings. Impressively, Grounding DINO with a Swin-T backbone outperforms DINO with Swin-L on the full-shot

Grounding DINO outperforms GLIP under the same backbone for the zero- shot setting. Grounding DINO and GLIPv2-T show similar *APaverage*. However, a key distinction lies in the *APmedian*, where Grounding DINO significantly outperforms GLIPv2-T (11.9 vs 8.9). This suggests that while GLIPv2 may exhibit larger performance variance across different datasets, Grounding DINO maintains a more consistent performance level. GLIPv2 incorporates advanced techniques like masked text training and cross-instance contrastive learning, making it

more complex than our Grounding DINO model. Moreover, our model is more compact (172M parameters) compared to GLIPv2 (232M parameters). These factors combined—performance consistency, model complexity, and size—should address concerns about our model’s capability in true open-set scenarios. Grounding DINO L set a new record on ODinW zero-shot with a 26*.*1 AP, even outperforming the giant Florence models [54]. The results show the generalization and scalability of Grounding DINO.

4.3 Referring Object Detection Settings We further explore our models’ performances on the REC task. We leverage GLIP [25] as our baseline. We evaluate the model performance on RefCOCO/+/g 5 directly. The results are shown in Table 5. Grounding DINO outperforms GLIP under the same setting. Nevertheless, both GLIP and Grounding DINO perform not well without REC data. More training data like caption data or larger models help the final performance, but quite minor. After injecting RefCOCO/+/g data into training, Grounding DINO obtains significant gains. The results reveal that most nowadays open-set object detectors need to pay more attention for a more fine-grained detection.
RefCOCOg Method Backbone Pre-Training Data Fine-tuning valRefCOCO testA testB valRefCOCO+ testA testB val test MAttNet [53] R101 None ✓ 76.65 81.14 69.99 65.33 71.62 56.02 66.58 67.27 VGTR [10] R101 None ✓ 79.20 82.32 73.78 63.91 70.09 56.51 65.73 67.23 TransVG [7] R101 None ✓ 81.02 82.72 78.35 64.82 70.70 56.94 68.67 67.73 VILLA_L*∗*[11] R101 CC, SBU, COCO, VG ✓ 82.39 87.48 74.84 76.17 81.54 66.84 76.18 76.71 RefTR [26] R101 VG ✓ 85.65 88.73 81.16 77.55 82.26 68.99 79.25 80.01 MDETR [18] R101 GoldG,RefC ✓ 86.75 89.58 81.41 79.52 84.09 70.62 81.64 80.89 DQ-DETR [45] R101 GoldG,RefC ✓ 88.63 91.04 83.51 81.66 86.15 73.21 82.76 83.44 GLIP-T(B) Swin-T O365,GoldG 49.96 54.69 43.06 49.01 53.44 43.42 65.58 66.08 GLIP-T Swin-T O365,GoldG,Cap4M 50.42 54.30 43.83 49.50 52.78 44.59 66.09 66.89

|Grounding DINO T (Ours)|Swin-T|O365,GoldG||50.41|57.24|43.21|51.40|57.59|45.81|67.46|67.13|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Grounding DINO T (Ours)|Swin-T|O365,GoldG,RefC||73.98|74.88|59.29|66.81|69.91|56.09|71.06|72.07|
|Grounding DINO T (Ours)|Swin-T|O365,GoldG,RefC|✓|89.19|91.86|85.99|81.09|87.40|74.71|84.15|84.94|
|Grounding DINO L (Ours)*|Swin-L|O365,OI,GoldG,Cap4M,COCO,RefC|✓||||90.56 93.19 88.24 82.75 88.95 75.92 86.13 87.02|||||

Table 5: Top-1 accuracy comparison on the referring expression comprehension task.

We mark the best results in bold. All models are trained with a ResNet-101 backbone. We use the notations “CC”, “SBU”, “VG”, “OI”, “O365”, and “YFCC” for Conceptual Captions [44], SBU Captions [36], Visual Genome [20], OpenImage [22], Objects365 [63], YFCC100M [47] respectively. The term “RefC” is used for RefCOCO, RefCOCO+, and RefCOCOg three datasets. * There might be a data leak since COCO includes validation images in RefC. But the annotations of the two datasets are different.

4.4 Effects of RefC and COCO Data We add the RefCOCO/+/g (we note it as “RefC” in tables) and COCO into training in some settings. We explore the influence of these data in Table 6. 5 We used the official released code and checkpoints in [https://github.com/](https://github.com/) microsoft/GLIP.

The results show that RefC helps improve the COCO zero-shot and fine-tuning performance but hurts the LVIS and ODinW results. With COCO introduced, the COCO results are greatly improved. It shows that COCO brings marginal improvements in LVIS and slight decreases on ODinW.

COCO minival LVIS minival ODinW Model Pre-Train Zero-Shot Fine-Tune Zero-Shot Zero-Shot Grounding DINO T O365,GoldG 48.1 57.1 25.6 20.0 Grounding DINO T O365,GoldG,RefC 48.5 57.3 21.9 17.7 Grounding DINO T O365,GoldG,RefC,COCO 56.1 57.5 22.3 17.4

Table 6: Impacts of RefC and COCO data for open-set settings. All models are trained

with a Swin Transformer Tiny backbone.

4.5 Ablations We conduct ablation studies in this section. We propose a tight fusion grounding model for open-set object detection and a sub-sentence level text prompt. To verify the effectiveness of the model design, we remove some fusion blocks for different variants. Results are shown in Table 7. All models are pre-trained on O365 with a Swin-T backbone. The results show that encoder fusion significantly improves model performance on both COCO and LVIS datasets. The results from comparing model #1 with the baseline model #0 validate this observation. Other techniques, such as language- guided query selection, text cross-attention, and sub-sentence text prompt, also contribute positively to the LVIS performance, yielding significant gains of +3.0 AP, +1.8 AP, and +0.5 AP, respectively. Additionally, these methods enhance the COCO zero-shot performance, further underscoring their effectiveness. However, we observed that language-guided query selection and sub-sentence text prompt had minimal impact on the COCO fine-tune performance. This outcome is reasonable, given that these methods do not alter model parameters or add computational burdens. Text cross-attention, while introducing fewer parameters than encoder fusion, showed less performance improvement compared to encoder fusion (+0.6 vs. +0.8). This finding suggests that fine-tuning performance is predominantly influenced by the model’s parameters, indicating that scaling models is a promising direction for enhancing performance.
### 5 Conclusion

We have presented a Grounding DINO model in this paper. Grounding DINO extends DINO to open-set object detection, enabling it to detect arbitrary objects given texts as queries. We review open-set object detector designs and propose a tight fusion approach to better fusing cross-modality information. We propose a sub-sentence level representation to use detection data for text prompts in a more reasonable way. The results show the effectiveness of our model design and

|||COCO minival||LVIS minival|
|---|---|---|---|---|
|#ID|Model|Zero-Shot|Fine-Tune|Zero-Shot|
|0|Grounding DINO (Full Model)|46.7|56.9|16.1|
|1|w/o encoder fusion|45.8|56.1|13.1|
|2|static query selection|46.3|56.6|13.6|
|3|w/o text cross-attention|46.1|56.3|14.3|
|4|word-level text prompt|46.4|56.6|15.6|

Table 7: Ablations for our model. All models are trained on the O365 dataset with a

Swin Transformer Tiny backbone.

fusion approach. Moreover, we extend open-set object detection to REC tasks and perform evaluation accordingly. We show that existing open-set detectors do not work well for REC data without fine-tuning. Hence we call extra attention to REC zero-shot performance in future studies. Limitations: Despite the great performance on open-set object detection settings, Grounding DINO cannot be used for segmentation tasks like GLIPv2. Our training data is less than the largest GLIP model, which may limit our final performance. Moreover, we find that our model will produce false positive results in some cases, which may need more techniques or data to reduce the hallucination. Social Impacts: The use of deep learning models, such as this one, exposes them to vulnerabilities through adversarial attacks. Additionally, the accuracy and correctness of the model’s outputs cannot be guaranteed. There is also the risk that the open-set detection capabilities of the model could be exploited for unlawful purposes.

### Acknowledgement

We thank the authors of GLIP [25]: Liunian Harold Li, Pengchuan Zhang, and Haotian Zhang for their helpful discussions and instructions. We also thank Tiancheng Zhao, the author of OmDet [60], and Jianhua Han, the author of DetCLIP [52], for their response to their model details. We thank He Cao of The Hong Kong University of Science and Technology for his help on diffusion models.

### References

1. Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., Zhang,
L.: Bottom-up and top-down attention for image captioning and visual question answering. computer vision and pattern recognition (2017)
2. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: End- to-end object detection with transformers. In: European Conference on Computer Vision. pp. 213–229. Springer (2020)
3. Chen, K., Pang, J., Wang, J., Xiong, Y., Li, X., Sun, S., Feng, W., Liu, Z., Shi, J., Ouyang, W., et al.: Hybrid task cascade for instance segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4974–4983 (2019)

4. Chen, Q., Chen, X., Wang, J., Feng, H., Han, J., Ding, E., Zeng, G., Wang, J.: Group DETR: Fast detr training with group-wise one-to-many assignment (2022)
5. Dai, X., Chen, Y., Xiao, B., Chen, D., Liu, M., Yuan, L., Zhang, L.: Dynamic head: Unifying object detection heads with attentions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7373–7382 (2021)
6. Dai, X., Chen, Y., Yang, J., Zhang, P., Yuan, L., Zhang, L.: Dynamic detr: End-to- end object detection with dynamic attention. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 2988–2997 (October
2021)
7. Deng, J., Yang, Z., Chen, T., Zhou, W., Li, H.: Transvg: End-to-end visual grounding with transformers. arXiv: Computer Vision and Pattern Recognition (2021)
8. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidi- rectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018)
9. Dong, N., Zhang, Y., Ding, M., Lee, G.H.: Boosting long-tailed object detection via step-wise learning on smooth-tail data (2023), [https://arxiv.org/abs/2305](https://arxiv.org/abs/2305). 12833
10.Du, Y., Fu, Z., Liu, Q., Wang, Y.: Visual grounding with transformers. (2021)
11. Gan, Z., Chen, Y.C., Li, L., Zhu, C., Cheng, Y., Liu, J.: Large-scale adversar- ial training for vision-and-language representation learning. neural information processing systems (2020)
12. Gao, P., Geng, S., Zhang, R., Ma, T., Fang, R., Zhang, Y., Li, H., Qiao, Y.: Clip-adapter: Better vision-language models with feature adapters. arXiv preprint arXiv:2110.04544 (2021)
13. Gao, P., Zheng, M., Wang, X., Dai, J., Li, H.: Fast convergence of DETR with spatially modulated co-attention. arXiv preprint arXiv:2101.07448 (2021)
14. Gu, X., Lin, T.Y., Kuo, W., Cui, Y.: Open-vocabulary object detection via vision and language knowledge distillation. Learning (2021)
15. Gupta, A., Dollar, P., Girshick, R.: Lvis: A dataset for large vocabulary instance segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5356–5364 (2019)
16. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 770–778 (2016)
17. Jia, D., Yuan, Y., He, H., Wu, X., Yu, H., Lin, W., Sun, L., Zhang, C., Hu, H.: DETRs with hybrid matching (2022)
18.Kamath, A., Singh, M., LeCun, Y., Synnaeve, G., Misra, I., Carion, N.: MDETR- modulated detection for end-to-end multi-modal understanding. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1780–1790 (2021)
19. Krasin, I., Duerig, T., Alldrin, N., Ferrari, V., Abu-El-Haija, S., Kuznetsova, A., Rom, H., Uijlings, J., Popov, S., Veit, A., et al.: Openimages: A public dataset for large-scale multi-label and multi-class image classification. Dataset available from [https://github](https://github). com/openimages 2(3), 18 (2017)
20. Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.J., Shamma, D.A., Bernstein, M.S., Fei-Fei, L.: Visual genome: Connecting language and vision using crowdsourced dense image annotations. International Journal of Computer Vision (2017)
21. Kuo, W., Bertsch, F., Li, W., Piergiovanni, A., Saffar, M., Angelova, A.: Findit: Generalized localization with natural language queries (2022)

22. Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali,
S., Popov, S., Malloci, M., Kolesnikov, A., Duerig, T., Ferrari, V.: The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. arXiv: Computer Vision and Pattern Recognition (2018)
23. Li, C., Liu, H., Li, L.H., Zhang, P., Aneja, J., Yang, J., Jin, P., Lee, Y.J., Hu, H., Liu,
Z., Gao, J.: Elevater: A benchmark and toolkit for evaluating language-augmented visual models (2022)
24. Li, F., Zhang, H., Liu, S., Guo, J., Ni, L.M., Zhang, L.: DN-DETR: Accelerate DETR training by introducing query denoising. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13619–13627 (2022)
25. Li, L.H., Zhang, P., Zhang, H., Yang, J., Li, C., Zhong, Y., Wang, L., Yuan,
L., Zhang, L., Hwang, J.N., et al.: Grounded language-image pre-training. arXiv preprint arXiv:2112.03857 (2021)
26. Li, M., Sigal, L.: Referring transformer: A one-step approach to multi-task visual grounding. arXiv: Computer Vision and Pattern Recognition (2021)
27. Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., Lee, Y.J.: GLIGEN: Open-set grounded text-to-image generation. CVPR (2023)
28. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal loss for dense object detection. In: Proceedings of the IEEE international conference on computer vision. pp. 2980–2988 (2017)
29. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014)
30. Liu, J., Wang, L., Yang, M.H.: Referring expression generation and comprehension via attributes. international conference on computer vision (2017)
31. Liu, S., Li, F., Zhang, H., Yang, X., Qi, X., Su, H., Zhu, J., Zhang, L.: DAB- DETR: Dynamic anchor boxes are better queries for DETR. In: International Conference on Learning Representations (2022), [https://openreview.net/forum?](https://openreview.net/forum?) id=oMI9PjOb9Jl
32. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. arXiv preprint arXiv:2103.14030 (2021)
33. Meng, D., Chen, X., Fan, Z., Zeng, G., Li, H., Yuan, Y., Sun, L., Wang, J.: Conditional DETR for fast training convergence. arXiv preprint arXiv:2108.06152 (2021)
34. Miao, P., Su, W., Wang, L., Fu, Y., Li, X.: Referring expression comprehension via cross-level multi-modal fusion. ArXiv abs/2204.09957 (2022)
35. Minderer, M., Gritsenko, A., Stone, A., Neumann, M., Weissenborn, D., Dosovitskiy,
A., Mahendran, A., Arnab, A., Dehghani, M., Shen, Z., Wang, X., Zhai, X., Kipf,
T., Houlsby, N.: Simple open-vocabulary object detection with vision transformers (2022)
36. Ordonez, V., Kulkarni, G., Berg, T.L.: Im2text: Describing images using 1 million captioned photographs. neural information processing systems (2011)
37. Plummer, B.A., Wang, L., Cervantes, C.M., Caicedo, J.C., Hockenmaier, J., Lazeb- nik, S.: Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In: Proceedings of the IEEE international conference on computer vision. pp. 2641–2649 (2015)
38. Plummer, B.A., Wang, L., Cervantes, C.M., Caicedo, J.C., Hockenmaier, J., Lazeb- nik, S.: Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. International Journal of Computer Vision (2015)

39. Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. IEEE Transactions on Pattern Analysis and Machine Intelligence 39(6), 1137–1149 (2017)
40. Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., Savarese, S.: Gener- alized intersection over union: A metric and a loss for bounding box regression. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 658–666 (2019)
41. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models (2021)
42. Sennrich, R., Haddow, B., Birch, A.: Neural machine translation of rare words with subword units. meeting of the association for computational linguistics (2015)
43. Shao, S., Li, Z., Zhang, T., Peng, C., Yu, G., Zhang, X., Li, J., Sun, J.: Objects365: A large-scale, high-quality dataset for object detection. In: Proceedings of the IEEE international conference on computer vision. pp. 8430–8439 (2019)
44. Sharma, P., Ding, N., Goodman, S., Soricut, R.: Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. meeting of the association for computational linguistics (2018)
45. Shilong, L., Yaoyuan, L., Shijia, H., Feng, L., Hao, Z., Hang, S., Jun, Z., Lei, Z.: DQ-DETR: Dual query detection transformer for phrase extraction and grounding. In: Proceedings of the AAAI Conference on Artificial Intelligence (2023)
46. Tan, M., Le, Q.V.: Efficientnet: Rethinking model scaling for convolutional neural networks. international conference on machine learning (2019)
47. Thomee, B., Shamma, D.A., Friedland, G., Elizalde, B., Ni, K., Poland, D.N., Borth,
D., Li, L.J.: Yfcc100m: the new data in multimedia research. Communications of The ACM (2016)
48. Wang, Y., Zhang, X., Yang, T., Sun, J.: Anchor DETR: Query design for transformer- based detector. national conference on artificial intelligence (2021)
49. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al.: Huggingface’s transformers: State-of-the- art natural language processing. arXiv preprint arXiv:1910.03771 (2019)
50. Xu, M., Zhang, Z., Hu, H., Wang, J., Wang, L., Wei, F., Bai, X., Liu, Z.: End-to-end semi-supervised object detection with soft teacher. arXiv preprint arXiv:2106.09018 (2021)
51. Yao, L., Han, J., Liang, X., Xu, D., Zhang, W., Li, Z., Xu, H.: DetCLIPv2: Scalable Open-Vocabulary Object Detection Pre-training via Word-Region Alignment (2023)
52. Yao, L., Han, J., Wen, Y., Liang, X., Xu, D., Zhang, W., Li, Z., Xu, C., Xu, H.: Detclip: Dictionary-enriched visual-concept paralleled pre-training for open-world detection (2022)
53. Yu, L., Lin, Z., Shen, X., Yang, J., Lu, X., Bansal, M., Berg, T.L.: Mattnet: Modular attention network for referring expression comprehension. computer vision and pattern recognition (2018)
54.Yuan, L., Chen, D., Chen, Y.L., Codella, N., Dai, X., Gao, J., Hu, H., Huang, X., Li, B., Li, C., Liu, C., Liu, M., Liu, Z., Lu, Y., Shi, Y., Wang, L., Wang, J., Xiao,
B., Xiao, Z., Yang, J., Zeng, M., Zhou, L., Zhang, P.: Florence: A new foundation model for computer vision (2022)
55. Zang, Y., Li, W., Zhou, K., Huang, C., Loy, C.C.: Open-vocabulary DETR with conditional matching (2022)
56. Zareian, A., Rosa, K.D., Hu, D.H., Chang, S.F.: Open-vocabulary object detection using captions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14393–14402 (2021)

57. Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L.M., Shum, H.Y.: DINO: DETR with improved denoising anchor boxes for end-to-end object detection (2022)
58. Zhang, H., Zhang, P., Hu, X., Chen, Y.C., Li, L.H., Dai, X., Wang, L., Yuan,
L., Hwang, J.N., Gao, J.: GLIPv2: Unifying Localization and Vision-Language Understanding (2022)
59. Zhang, H., Zhang, P., Hu, X., Chen, Y.C., Li, L.H., Dai, X., Wang, L., Yuan, L., Hwang, J.N., Gao, J.: Glipv2: Unifying localization and vision-language understand- ing (2022)
60. Zhao, T., Liu, P., Lu, X., Lee, K.: Omdet: Language-aware object detection with large-scale vision-language multi-dataset pre-training (2022)
61. Zhong, Y., Yang, J., Zhang, P., Li, C., Codella, N., Li, L.H., Zhou, L., Dai, X., Yuan, L., Li, Y., Gao, J.: Regionclip: Region-based language-image pretraining (2022)
62. Zhou, X., Girdhar, R., Joulin, A., Krähenbühl, P., Misra, I.: Detecting twenty- thousand classes using image-level supervision. In: ECCV (2022)
63. Zhou, X., Wang, D., Krähenbühl, P.: Objects as points. arXiv preprint arXiv:1904.07850 (2019)
64. Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J.: Deformable DETR: Deformable transformers for end-to-end object detection. In: ICLR 2021: The Ninth International Conference on Learning Representations (2021)

### A More Implementation Details

A.1 Hyperparameters
Table 8 presents the hyperparameters used in our main experiments.

|Item|Value|
|---|---|
|optimizer|AdamW|
|lr|1e-4|
|lr of image backbone|1e-5|
|lr of text backbone|1e-5|
|weight decay|0.0001|
|clip max norm|0.1|
|number of encoder layers|6|
|number of decoder layers|6|
|dim feedforward|2048|
|hidden dim|256|
|dropout|0.0|
|nheads|8|
|number of queries|900|
|set cost class|1.0|
|set cost bbox|5.0|
|set cost giou|2.0|
|ce loss coef|2.0|
|bbox loss coef|5.0|
|giou loss coef|2.0|

Table 8: Hyper-parameters used in our pre-trained models.

A.2 Pseudo Code Language-Guided Query Selection We present the pseudo-code of the Language-Guided Query Selection module in Algorithm 1. The variables image_feat and text_feat are used for image and text features, respectively. num_query is the number of queries in the decoder, which is set to 900 in our implementation. We use bs and ndim for batch size and feature dimension in the pseudo-code. num_img_tokens and num_text_tokens are used for the number of image and text tokens, respectively.
### B Data Usage

We use three types of data in our model pre-train.

Algorithm 1: Pseudocode of Language-guided Query Selection in PyTorch-like style.

|"""|||
|---|---|---|
|Input:|num_query:int|image_feat:(bs,num_img_tokens,ndim) text_feat:(bs,num_text_tokens,ndim)|
|Output:|topk_idx:(bs,num_query)||
|"""|||
|logits|= num_text_tokens logits_per_img_feat|torch.einsum("bic,btc->bit", image_feat, text_feat)#bs,num_img_tokens, = logits.max(-1)[0]#bs,num_img_tokens|
|topk_idx|=|torch.topk(logits_per_img_feature, num_query, dim=1)[1]#bs,num_query|

1 2 3 4 5 6 7 8 9 10 11 12 13

1. Detection data. Following GLIP [25], we reformulate the object detection task to a phrase grounding task by concatenating the category names into text prompts. We use COCO [29], O365 [43], and OpenImage(OI) [19] for our model pretrain. To simulate different text inputs, we randomly sampled category names from all categories in a dataset on the fly during training.
2. Grounding data. We use the GoldG and RefC data as grounding data. Both GoldG and RefC are preprocessed by MDETR [18]. These data can be fed into Grounding DINO directly. GoldG contains images in Flickr30k entities [37,38] and Visual Genome [20]. RefC contains images in RefCOCO, RefCOCO+, and RefCOCOg.
3. Caption data. To enhance the model performance on novel categories, we feed the semantic-rich caption data to our model. Following GLIP, we use the pseudo-labeled caption data for model training. In our experiments, we use the same data with GLIP under comparable settings. More specifically, we use GLIP-T annotated caption data for Grounding DINO T, while GLIP-L annotated caption data for Grounding DINO L. There are two versions of the O365 dataset, which we termed O365v1 and O365v2, respectively. O365v1 is a subset of O365v2. O365v1 contains about 600K images, while O365v2 contains about 1.7M images. Following previous works [25,52], we pre-train the Grounding DINO T on O365v1 for a fair comparison. The Grounding DINO L is pre-trained on O365v2 for a better result.
### C More Experiment Results

C.1 Transfer from DINO to Grounding DINO Recent work has presented many large-scale image models for detection with DINO architecture
6. It is computationally expensive to train a Grounding DINO

6 See model instances at [https://github.com/IDEA-Research/detrex](https://github.com/IDEA-Research/detrex)

model from scratch. However, the cost can be significantly reduced if we leverage pre-trained DINO weights. Hence, we conduct some experiments to transfer pre-trained DINO to Grounding DINO models. We freeze the modules co-existing in DINO and Grounding DINO and fine-tune the other parameters only. (We compare DINO and Grounding DINO in Sec. C.1.) The results are available in

Table 9.

Model DINO Pre-Train Pre-Train Data Grounded Fine-Tune COCO minival Zero-Shot LVIS minival Zero-Shot Zero-Shot ODinW

Grounding DINO T-O365 46.7 16.2 14.5 (from scratch)-O365,GoldG 48.1 25.6 20.0 Grounding DINO T O365 O365 46.5 17.9 13.6 (from pre-trained DINO) O365 O365,GoldG 46.4 26.1 18.5

Table 9: Transfer pre-trained DINO to Grounding DINO. We freeze shared modules

between DINO and Grounding DINO during grounded fine-tuning. All models are trained with a Swin Transformer Tiny backbone.

It shows that we can achieve similar performances with Grounding DINO- Training only text and fusion blocks using a pre-trained DINO. Interestingly, the DINO-pre-trained Grounding DINO outperforms the standard Grounding DINO on LVIS under the same setting. The results show that there might be much room for model training improvement, which will be our future work to explore.

Table 10: Comparison between two Grounding DINO variants: Training from scratch

and transfer from DINO-pretrained models. The models are trained on O365 and evaluated on COCO.

With a pre-trained DINO initialization, the model converges faster than Grounding DINO from scratch, as shown in Fig. 10. Notably, we use the results without exponential moving average (EMA) for the curves in Fig. 10, which results in a different final performance that in Table 9. As the model trained from scratch need more training time, we only show results of early epochs. Comparison between DINO and Grounding DINO To illustrate the difference between DINO and Grounding DINO, we compare DINO and

Grounding DINO in Fig. 5. We mark the DINO blocks in gray, while the newly proposed modules are shaded in blue.

C.2 Detailed Results on COCO Detection Benchmarks COCO Detection Results under the 1*×* Setting Model Epochs AP AP₅₀ AP₇₅ AP*S* AP*M* AP*L* Faster-RCNN(5scale) [39] 12 37*.*9 58*.*8 41*.*1 22*.*4 41*.*1 49*.*1

|Faster-RCNN(5scale) [39]|12 37.9|58.8 41.1|22.4 41.1|49.1|
|---|---|---|---|---|
|DETR(DC5) [2]|12 15.5|29.4 14.5|4.3 15.1|26.7|
|Deformable DETR(4scale) [64]|12 41.1|− −|− −||
|DAB-DETR(DC5)|12 38.0|60.3 39.8|19.2 40.9|55.4|
|Dynamic DETR(5scale) [6]|12 42.9|61.0 46.3|24.6 44.9|54.4|
|Dynamic Head(5scale) [5]|12 43.0|60.7 46.8|24.7 46.4|53.9|
|HTC(5scale) [3]|12 42.3|− −|− −|−|
|DN-Deformable-DETR(4scale) [24]|12 43.4|61.9 47.2|24.8 46.8|59.4|
|DINO-4scale [57]|12 49.0|66.6 53.5|32.0 52.3|63.0|
|Grounding DINO (4scale)|12 48.1|65.8 52.3|30.4 51.3|62.3|

*†* [31]

Table 11: Results for Grounding DINO and other detection models with the ResNet50

backbone on COCO val2017 trained with 12 epochs (the so called 1*×* setting).

We present the performance of Grounding DINO on standard COCO detection benchmark in Table 11. All models are trained with a ResNet-50 [16] backbone for 12 epochs. Grounding DINO achieves 48.1 AP under the research setting, which shows that Grounding DINO is a strong closed-set detector. However, it is inferior compared with the original DINO. We suspect that the new components may make the model harder to optimize than DINO.

C.3 Detailed Results on ODinW Benchmarks We present detailed results of Grounding DINO on ODinW35 [23] in Table 12,
Table 13, and Table 14.

C.4 Model Efficiency We compare the model size and efficiency between Grounding DINO T and GLIP-T in Table 20. The results show that our model has a smaller parameter size and better efficiency than GLIP.
C.5 Ablations for More Decoder Queries To verify the model performance with more decoder queries, we conducted additional experiments with 1200 and 1500 queries on the COCO and LVIS datasets, detailed in Table 15 below.

1. Model Overall **Contrastive loss** Localization loss
cat **Text** dog desk 1 **Features**person dog mouse table sets 1 1

Model Outputs

Values Cross-Modality Decoder Keys& Cross-Modality Queries Language-guide Query Selection **Text** Image **Features** Features

Feature Enhancer **Vanilla Text** Vanilla Image **Features** Features Text Image Backbone Backbone

cat. person. mouse. A cat sets on a table.

**Input Text** Input Image

3. A Decoder Layer
Cross Updated -Modality A Cross-Modality Decoder Layer Query FFN **Text Features** K,V Text Cross-Attention Q K,V Image Cross-Attention Q

Image Features

Self-Attention Q,K,V Cross-Modality Query

2. A Feature Enhancer Layer **Updated Text** Updated Image **Features** Features FFN FFN Text-to-image Cross-Attention Q K,V Image-to-text Cross-Attention K,V Q Self-Attention Self-Attention
Deformable Q,K,V Q,K,V **Text** Image **Features** Features

tions in blue. Best view in color.

#### the benefits of additional queries.

18 epochs.

Fig. 5: Comparison between DINO and our Grounding DINO. We mark the modifica-

The results indicate that models with 1200 and 1500 queries slightly out- perform the 900-query version on LVIS rare classes, suggesting better coverage of rare classes. However, the improvement is marginal, as the 900-query model already sufficiently covers all objects in both COCO and LVIS. Additionally, introducing more queries exacerbates data imbalance during training, as the model is trained on objects from sampled categories. This imbalance could offset

C.6 Results with Different Language Encoder for REC To verify the impacts of language encoders with different sizes, we conducted experiments using two variants of BERT: bert-base-uncased (BERT-B) and bert-large-uncased (BERT-L). These models were trained on a combined dataset consisting of RefCOCO, RefCOCO+, and RefCOCOg. We removed the leaked data in the combined dataset for a fair comparison. It’s important to note that training on this combined dataset, as opposed to tuning each dataset separately, might result in slightly lower performance. For a fair comparison, we initialized the models with the O365+GoldG+Cap4M checkpoint, except for the BERT parameters. Limited by time and resources, the training duration was capped at

Interestingly, our results showed that Grounding DINO with BERT-B outper- formed or matched the BERT-L variant in most metrics (as shown in Table 16). This suggests that our default use of BERT-B during the pretrain stage may have contributed to its better performance. Moreover, we didn’t observe significant improvements in the late stages of training, indicating that both models were nearing their optimal performance. This outcome suggests that the main limitation in enhancing REC performance lies within the detection branch, rather than the language processing module. A dedicated model for REC data might be helpful for REC tasks.

C.7 Detic Pseudo-labeled Data for LVIS To illustrate the potential of our model under similar conditions, we conducted oracle experiments as shown in Table 17. We pseudo-labeled the ImageNet dataset using a pre-trained Detic [62] model and filtered out LVIS-related images for training, creating a new pseudo-labeled dataset named IN22K-LVIS-1M. It contains about 1M pseudo-labeled images for training. Note that our model under the setting may be not a real zero-shot setting, as our pseudo labeler Detic is trained with LVIS data. The results from these experiments suggest that Grounding DINO can achieve promising LVIS performance, even without direct LVIS training data. A similarly distributed dataset can help the model to generalize well.
C.8 Comparison between Grounding DINO and GLIP on ODinW In our comparison of Grounding DINO and GLIP across various datasets in ODinW, as presented in Table 18, we observe that Grounding DINO underper- forms on certain uncommon datasets where both models generally show limited effectiveness. For instance, in the PlantDoc dataset, Grounding DINO scores
0.36 compared to GLIP’s 1.1. This dataset includes infrequent categories such as "Tomato leaf mosaic virus," which are not well-represented in the training data. These findings highlight the need for improving data quality to enhance overall model performance.
C.9 Training Grounding DINO on RefCOCO/+/g from scratch To demonstrate our model’s capabilities, we trained a Grounding DINO from scratch on the RefCOCO/+/g datasets, as shown in Table 19. Despite time limitations that restricted us to 9 epochs of training, we achieved comparable per- formance to SOTA REC models. This underscores the potential of our approach to handling REC tasks effectively.

### D Visualizations

D.1 Detection Visualizations We present some visualizations in Fig. 6. Our model presents great generalization on different scenes and text inputs. For example, Grounding DINO accurately locates man in blue and child in red in the last image.
Fig. 6: Visualizations of model outputs.

D.2 Physical meaning of language-guided query We visualize the top 900 queries (boxes) in language-guided query selection across various prompts in Figure 7. It shows our model can have dynamic queries during inference. We will incorporate the analyses into the revised paper.
Input Image building river ferris wheel

Fig. 7: Top queries in language-guided query selection.

D.3 Comparison of RefCOCO and Grounding Data The RefCOCO has a different formulation with grounded training, resulting in a big performance gap without the RefCOCO dataset. As shown in Fig. 8, each RefCOCO text prompt corresponds to only *one box*, while our model tends to predict *multiple objects*.

pred: left gt: rightman on far left on screen woman with glasses guy with cow

Fig. 8: Our model predictions and ground-truths in RefCOCO.

D.4 Marry Grounding DINO with Stable Diffusion for Object Detection and Inpainting We present an image editing application in Fig. 1 (b). The results in Fig. 1 (b) are generated by two processes. First, we detect objects with Grounding DINO and generate masks by masking out the detected objects or backgrounds. After that, we feed original images, image masks, and generation prompts to an inpainting model (typical Stable Diffusion [41]) to render new images. We use the released checkpoints in [https://github.com/Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion) for new image generation. More results are available in Figure 9. The “detection prompt” is the language input for Grounding DINO, while the “generation prompt” is for the inpainting model.
D.5 Marry Grounding DINO with Stable Diffusion for Object Detection and Grounded Generation To enable fine-grained image editing, we combine the Grounding DINO with GLIGEN [27]. We use the “phrase prompt” in Figure 10 as the input phrases of each box for GLIGEN. GLIGEN supports grounding results as inputs and can generate objects on specific positions. We can assign each bounding box an object with GLIGEN, as shown in Figure 10 (c) (d). Moreover, GLIGEN can full fill each bounding box, which results in better visualization, as that in Figure 10 (a) (b). For example, we use the same generative prompt in Figure 9 (b) and Figure 10 (b). The GLIGEN results ensure each bounding box with an object and fulfills the detected regions.

|Model|params|GFLOPS|FPS|
|---|---|---|---|
|GLIP-T [25]|232M|488G|6.11|
|Grounding DINO T (Ours)|172M|464G|8.37|

Table 20: Comparison of model size and model efficiency between GLIP and Grounding
 DINO.

Inputs Detection Edited Images Results

(a) Detection Prompt: green mountain Generation Prompt: red mountain.
(b) Detection Prompt: pandas Generation Prompt: dogs and birthday cakes
(c) Detection Prompt: black cat Generation Prompt: cats and apples
(d) Detection Prompt: the running girl Generation Prompt (modify background): The Wandering Earth
(e)* Detection Prompt: face Generation Prompt (modify background): a girl with short hair
Fig. 9: Combination of Grounding DINO and Stable Diffusion. We first detect objects

with Grounding DINO and then perform image inpainting with Stable Diffusion. “De- tection Prompt” and “Generation Prompt” are inputs for Grounding DINO and Stable Diffusion, respectively. *The input human face in the row (e) is generated by StyleGAN.

|28|S. Liu et al.||
|---|---|---|
||Inputs|Detection Results|
|Fig. 10:|Combination of Grounding DINO and GLIGEN. We first detect objects||

Edited Images

(a) Detection Prompt: sign Generation Prompt: flying birds. Phrase Prompt: flying birds.
(b) Detection Prompt: pandas Generation Prompt: dogs and birthday cakes. Phrase Prompt*: a dog; a cake.
(c) Detection Prompt: dog, cat Generation Prompt: a cake and a phone Phrase Prompt*: a cake; a phone.
(d) Detection Prompt: a sketch person Generation Prompt: a woman and a man are talking Phrase Prompt*: a woman; a man.
with Grounding DINO and then perform image inpainting with GLIGEN. “Detection Prompt” and “Generation Prompt” are inputs for Grounding DINO and Stable Diffusion, respectively. “Phrase Prompt” are language inputs for each bounding box. The phrase prompts are separated by semicolons. *We assign phrase prompts to bounding boxes randomly.

Dataset AP AP₅₀ AP₇₅ AP*S* AP*M* AP*L* AerialMaritimeDrone_large 9.48 15.61 8.35 8.72 10.28 2.91

|AerialMaritimeDrone_large|9.48 15.61 8.35|||8.72 10.28 2.91|||
|---|---|---|---|---|---|---|
|AerialMaritimeDrone_tiled|17.56 26.35 13.89|||0|1.61|28.7|
|AmericanSignLanguageLetters Aquarium|1.45 18.83 34.32 18.19 10.65 20.64 21.52|2.21|1.39|-1|-1|1.81|
|BCCD_BCCD|6.17 11.31 6.04|||1.27|9.09|6.89|
|ChessPiece|6.99 11.13 9.03|||-1|-1|8.11|
|CottontailRabbits|71.93 85.05 85.05|||-1|70|73.58|
|DroneControl_Drone_Control EgoHands_generic|6.15 10.95 6.23 48.07 75.06 56.52 1.48 11.42 51.84|||2.08|6.91|6.16|
|EgoHands_specific|0.66|1.25|0.64|0|0.02|0.92|
|HardHatWorkers|2.39|9.17|1.07|2.13|4.32|4.6|
|MaskWearing|0.58|1.43|0.56|0.12|0.51|4.66|
|MountainDewCommercial|18.22 29.73 21.33|||0|23.23 49.8||
|NorthAmericaMushrooms|65.48 71.26 66.18|||-1|-1|65.49|
|OxfordPets_by-breed|0.27|0.6|0.21|-1|1.38|0.33|
|OxfordPets_by-species|1.66|5.02|1|-1|0.65|1.89|
|PKLot_640|0.08|0.26|0.02|0.14|0.79|0.11|
|Packages PascalVOC|56.34 68.65 68.65 47.21 57.59 51.28 16.53 39.51 58.5|||-1|-1|56.34|
|Raccoon_Raccoon|44.82 76.44 46.16|||-1|17.08 48.56||
|ShellfishOpenImages|23.08 32.21 26.94|||-1|18.82 23.28||
|ThermalCheetah|12.9 19.65 14.72|||0|8.35 50.15||
|UnoCards VehiclesOpenImages WildfireSmoke|0.87 59.24 71.88 64.69 7.42 32.38 72.21 25.6 43.96 25.34 5.03 18.85 42.59|1.52|0.96|2.91|2.18|-1|
|boggleBoards|0.81|2.92|0.12|2.96|1.13|-1|
|brackishUnderwater|1.3|1.88|1.4|0.99|1.75 11.39||
|dice_mediumColor|0.16|0.72|0.07|0.38|3.3|2.23|
|openPoetryVision pistols|0.18 46.4 66.47 47.98 4.51 22.94 55.03|0.5|0.06|-1|0.25|0.17|
|plantdoc pothole|0.34 19.87 28.94 22.23 12.49 15.6 28.78|0.51|0.35|-1|0.28|0.86|
|selfdrivingCa thermalDogsAndPeople|9.46 19.13 8.19 72.67 86.65 79.98 33.93 30.2 86.71|||0.85|6.82 16.51||
|websiteScreenshots|1.51|2.8|1.42|0.85|2.06|2.59|

Table 12: Detailed results on 35 datasets in ODinW of Grounding DINO with Swin-T

pre-trained on O365 and GoldG.

|Dataset|AP AP₅₀ AP₇₅|AP|AP AP|
|---|---|---|---|
|AerialMaritimeDrone_large|10.3 18.17 9.21|8.92|11.2 7.35|
|AerialMaritimeDrone_tiled|17.5 28.04 18.58|0|3.64 24.16|
|AmericanSignLanguageLetters Aquarium|0.78 1.17 18.64 35.27 17.29 11.33 17.8 21.34|0.76-1|-1 1.02|
|BCCD_BCCD|11.96 22.77 8.65|0.16|5.02 13.15|
|ChessPiece|15.62 22.02 20.19|-1|-1 15.72|
|CottontailRabbits|67.61 78.82 78.82|-1|70 68.09|
|DroneControl_Drone_Control EgoHands_generic|4.99 8.76 57.64 90.18 66.78 3.74 24.67 61.33|5 0.65|5.03 8.61|
|EgoHands_specific|0.69 1.37|0.63 0|0.02 1.03|
|HardHatWorkers|4.05 13.16 1.96|2.29|7.55 9.81|
|MaskWearing|0.25 0.81|0.15 0.09|0.13 2.78|
|MountainDewCommercial|25.46 39.08 28.89|0|32.53 58.38|
|NorthAmericaMushrooms|68.18 72.89 69.75|-1|-1 68.62|
|OxfordPets_by-breed|0.21 0.42|0.22-1|2.91 0.17|
|OxfordPets_by-species|1.3 3.95|0.71-1|0.28 1.62|
|PKLot_640|0.06 0.18|0.02 0.03|0.59 0.15|
|Packages PascalVOC|60.53 76.24 76.24 55.65 66.51 60.47 19.61 44.25 67.21|-1|-1 60.53|
|Raccoon_Raccoon|60.07 84.81 66.5|-1|11.23 65.86|
|ShellfishOpenImages ThermalCheetah|29.56 38.08 33.5 17.72 25.93 19.61 1.04 20.02 63.69|-1|6.38 29.95|
|UnoCards VehiclesOpenImages WildfireSmoke|0.81 1.3 58.49 71.56 63.64 8.22 28.03 71.1 20.04 39.74 22.49 4.13 15.71 30.41|1 2.6|1.01-1|
|boggleBoards|0.29 1.15|0.04 1.8|0.57-1|
|brackishUnderwater|1.47 2.34|1.58 2.32|3.31 9.96|
|dice_mediumColor|0.33 1.38|0.15 0.03|1.05 12.57|
|openPoetryVision pistols|0.05 0.19 66.99 86.34 72.65 16.25 39.24 75.98|0-1|0.09 0.21|
|plantdoc pothole|0.36 0.47 25.21 38.21 26.01 8.94 18.45 39.28|0.39-1|0.24 0.82|
|selfdrivingCa thermalDogsAndPeople|9.95 20.55 8.28 67.89 80.85 78.66 45.05 30.24 85.56|1.36|7.27 15.46|
|websiteScreenshots|1.3 2.26|1.21 0.95|1.81 2.23|

*S M L*

Table 13: Detailed results on 35 datasets in ODinW of Grounding DINO with Swin-T

pre-trained on O365, GoldG, and Cap4M.

Dataset AP AP₅₀ AP₇₅ AP*S* AP*M* AP*L* AerialMaritimeDrone_large 12.64 18.44 14.75 9.15 19.16 0.98 AerialMaritimeDrone_tiled 20.47 34.81 12.79 0 7.61 26.93

|AerialMaritimeDrone_tiled|20.47 34.81 12.79|||0|7.61 26.93||
|---|---|---|---|---|---|---|
|AmericanSignLanguageLetters Aquarium|3.94 28.14 45.47 30.97 12.1 24.71 39.42|4.84|4|-1|-1|4.48|
|BCCD_BCCD|23.85 36.92 28.88|||0.3|10.8 24.43||
|ChessPiece|18.44 26.3 23.33|||-1|-1|18.62|
|CottontailRabbits|71.66 88.48 88.48|||-1|66|73.04|
|DroneControl_Drone_Control EgoHands_generic|7.16 11.56 7.67 52.08 81.57 59.15 1.12 31.78 55.46|||2.29|10.6|7.68|
|EgoHands_specific|1.22|2.28|1.2|0|0.05|1.5|
|HardHatWorkers|9.14 23.64||5.6|5.09 15.34 13.59|||
|MaskWearing|1.64|4.69|1.18|0.44|1.05|8.67|
|MountainDewCommercial|33.28 53.59 32.76|||0|35.86|80|
|NorthAmericaMushrooms|72.33 73.18 73.18|||-1|-1|72.39|
|OxfordPets_by-breed|0.58|1.05|0.59|-1|4.46|0.6|
|OxfordPets_by-species|1.64|4.8|0.87|-1|1.51|1.8|
|PKLot_640|0.25|0.71|0.05|0.31|1.44|0.4|
|Packages PascalVOC|63.86 76.24 76.24 66.01 76.65 71.8 32.01 55.7 75.37|||-1|-1|63.86|
|Raccoon_Raccoon|65.81 90.39 69.93|||-1|26|68.97|
|ShellfishOpenImages ThermalCheetah|62.47 74.25 70.07 21.33 26.11 24.92 2.39 15.84 75.34|||-1|26|63.06|
|UnoCards VehiclesOpenImages WildfireSmoke|0.52 62.74 75.15 67.23 10.66 47.46 76.36 23.66 45.72 25.06 1.58 22.22 35.27|0.84|0.66|3.02|0.92|-1|
|boggleBoards|0.28|1.04|0.05|5.64|0.7|-1|
|brackishUnderwater|2.41|3.39|2.79|4.43|3.88 21.22||
|dice_mediumColor|0.26|1.15|0.03|0|1.09|4.07|
|openPoetryVision pistols|0.08 71.4 90.69 77.21 18.74 39.58 80.78|0.35|0.01|-1|0.15|0.11|
|plantdoc pothole|2.02 30.4 44.22 33.84 12.27 18.84 48.57|2.64|2.37|-1|0.5|2.82|
|selfdrivingCa thermalDogsAndPeople|9.25 17.72 8.39 72.02 86.02 79.47 29.16 68.05 86.75|||1.93|7.03 13.02||
|websiteScreenshots|1.32|2.64|1.16|0.79|1.8|2.46|

Table 14: Detailed results on 35 datasets in ODinW of Grounding DINO with Swin-L

pre-trained on O365, OI, GoldG, Cap4M, COCO, and RefC.

|Pretrain Data Query Num||COCO||LVIS||
|---|---|---|---|---|---|
|||AP AP|AP AP|AP AP|AP|
|O365|900|46.7 32.1|50.0 61.3 15.8|9.4 22.9 28.4||
|O365|1200|46.7 32.3|49.9 61.1 15.7|9.0 23.1 29.1||
|O365|1500|46.9 32.7|50.3 61.3 15.8|9.2 23.0 29.0||

*S M L* AP *r c f*

Table 15: Results for Grounding DINO Tiny with more decoder queries

|Model|RefCOCO||RefCOCO+||RefCOCOg||
|---|---|---|---|---|---|---|
||val testA testB||val testA testB||val|test|
|Grounding DINO T (BERT-B) 87.4 91.6||84.2|78.6 86.5|73.4|81.6|83.3|
|Grounding DINO T (BERT-L) 87.0 91.4||84.0|78.1 86.4|73.0|81.7|83.3|

Table 16: Results for Grounding DINO with different language encoders

|Model|Pre-train Data|LVIS MiniVal-AP(r/c/f)|
|---|---|---|
|||Zero-shot|
|DetCLIPv2-T Grounding DINO T OG+IN22K-LVIS-1M 40.6 (38.5 / 41.1 / 40.4) 54.5 (47.3 / 53.9 / 56.1)|OG + CC15M|40.4 (36.0 / 41.7 / 40.0) 50.7 (44.3 / 52.4 / 50.3)|

Fine-tune

Table 17: Oracle experiments on LVIS. Note that our model under the setting may be

not a real zero-shot setting, as our pseudo labeler Detic is trained with LVIS data.

Metric

||GLIP-T Grounding DINO T|
|---|---|
|19.6|22.3|
|5.1|11.9|
|13.70|10.30|
|12.60|17.50|
||0.78|
|18.30|18.64|
|1.00|11.96|
|10.00|15.62|
|69.70|67.61|
|5.10|4.99|
|50.00|57.64|
|0.80|0.69|
|3.00|4.05|
|1.10|0.25|
|21.60|25.46|
|75.10|68.18|
|0.40|0.21|
|1.10|1.30|
|0.00|0.06|
|72.30|60.53|
|56.10|55.65|
|57.80|60.07|
|25.90|29.56|
|2.70|17.72|
|0.20|0.81|
|56.00|58.49|
|2.30|20.04|
|0.00|0.29|
|3.70|1.47|
|1.10|0.33|
|0.00|0.05|
|49.80|66.99|
|1.10|0.36|
|17.20|25.21|
|8.00|9.95|
|43.70|67.89|
|0.50|1.30|

Average Score (↑) Median Score (↑) AerialMaritimeDrone_large (↑) AerialMaritimeDrone_tiled (↑) AmericanSignLanguageLetters_American_Sign_Language_Letters (↑) 2.50 Aquarium_Aquarium_Combined (↑) BCCD_BCCD (↑) ChessPieces_Chess_Pieces (↑) CottontailRabbits (↑) DroneControl_Drone_Control (↑) EgoHands_generic (↑) EgoHands_specific (↑) HardHatWorkers (↑) MaskWearing (↑) MountainDewCommercial (↑) NorthAmericaMushrooms_North_American_Mushrooms (↑) OxfordPets_by-breed (↑) OxfordPets_by-species (↑) PKLot_640 (↑) Packages_Raw (↑) PascalVOC (↑) Raccoon_Raccoon (↑) ShellfishOpenImages (↑) ThermalCheetah (↑) UnoCards (↑) VehiclesOpenImages (↑) WildfireSmoke (↑) boggleBoards_416x416AutoOrient_export (↑) brackishUnderwater (↑) dice_mediumColor_export (↑) openPoetryVision (↑) pistols_export (↑) plantdoc (↑) pothole (↑) selfdrivingCar_fixedLarge_export (↑) thermalDogsAndPeople (↑) websiteScreenshots (↑)

Table 18: Comparison of Grounding DINO and GLIP on ODinW. Both models are

trained on O365, GoldG, and Cap4M with Swin-Tiny backbones.

refcoco (val/testA/testB) refcoco+ (val/testA/testB) refcocog (val/test)

|TransVG|R101 81.02/82.72/78.35|64.82/70.70/56.94|68.67/67.73|
|---|---|---|---|
|Grounding DINO T (9ep) SwinT 81.42/85.28/76.75||69.14/74.79/60.17|73.96/74.37|

Table 19: Training on RefCOCO from scratch.
