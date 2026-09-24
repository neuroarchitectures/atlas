# OneFormer: One Transformer to Rule Universal Image Segmentation

Universal Image Segmentation is not a new concept. Past attempts to unify image segmentation in the last decades include scene parsing, panoptic segmentation, and, more recently, new panoptic architectures. However, such panoptic architectures do not truly unify image segmentation because they need to be trained individually on the semantic, instance, or panoptic segmentation to achieve the best performance. Ideally, a truly universal framework should be trained only once and achieve SOTA performance across all three image segmentation tasks. To that end, we propose OneFormer, a universal image segmentation framework that unifies segmentation with a multi-task train-once design. We first propose a task-conditioned joint training strategy that enables training on ground truths of each domain (semantic, instance, and panoptic segmentation) within a single multi-task training process. Secondly, we introduce a task token to condition our model on the task at hand, making our model task-dynamic to support multi-task training and inference. Thirdly, we propose using a query-text contrastive loss during training to establish better inter-task and inter-class distinctions. Notably, our single OneFormer model outperforms specialized Mask2Former models across all three segmentation tasks on ADE20k, Cityscapes, and COCO, despite the latter being trained on each of the three tasks individually with three times the resources. With new ConvNeXt and DiNAT backbones, we observe even more performance improvement. We believe OneFormer is a significant step towards making image segmentation more universal and accessible. To support further research, we open-source our code and models at https://github.com/SHI-Labs/OneFormer.

## 1 Introduction

Image Segmentation is the task of grouping pixels into multiple segments. Such grouping can be semantic-based (e.g., road, sky, building), or instance-based (objects with well-defined boundaries). Earlier segmentation approaches [40, 7, 23] tackled these two segmentation tasks individually, with specialized architectures and therefore separate research effort into each. In a recent effort to unify semantic and instance segmentation, Kirillov et al. [29] proposed panoptic segmentation, with pixels grouped into an amorphous segment for amorphous background regions (labeled “stuff”) and distinct segments for objects with well-defined shape (labeled “thing”). This effort, however, led to new specialized panoptic architectures [11] instead of unifying the previous tasks (see Fig. 1a). More recently, the research trend shifted towards unifying image segmentation with new panoptic architectures, such as K-Net [61], MaskFormer [13], and Mask2Former [12]. Such panoptic architectures can be trained on all three tasks and obtain high performance without changing architecture. They do need to, however, be trained individually on each task to achieve the best performance (see Fig. 1b). The individual training policy requires extra training time and produces different sets of model weights for each task. In that regard, they can only be considered a semi-universal approach. For example, Mask2Former [12] is trained for 160K iterations on ADE20K [15] for each of the semantic, instance, and panoptic segmentation tasks to obtain the best performance for each task, yielding a total of 480k iterations in training, and three models to store and host for inference.

In an effort to truly unify image segmentation, we propose a multi-task universal image segmentation framework (OneFormer), which outperforms existing state-of-the-arts on all three image segmentation tasks (see Fig. 1c), by only training once on one panoptic dataset. Through this work, we aim to answer the following questions:

(i) Why are existing panoptic architectures [12, 13] not successful with a single training process or model to tackle all three tasks? We hypothesize that existing methods need to train individually on each segmentation task due to the absence of task guidance in their architectures, making it challenging to learn the inter-task domain differences when trained jointly or with a single model. To tackle this challenge, we introduce a task input token in the form of text: “the task is {task}”, to condition the model on the task in focus, making our architecture task-guided for training, and task-dynamic for inference, all with a single model. We uniformly sample {task} from {panoptic, instance, semantic} and the corresponding ground truth during our joint training process to ensure our model is unbiased in terms of tasks. Motivated by the ability of panoptic [29] data to capture both semantic and instance information, we derive the semantic and instance labels from the corresponding panoptic annotations during training. Consequently, we only need panoptic data during training. Moreover, our joint training time, model parameters, and FLOPs are comparable to the existing methods, decreasing training time and storage requirements up to 3×, making image segmentation less resource intensive and more accessible.

(ii) How can the multi-task model better learn inter-task and inter-class differences during the single joint training process? Following the recent success of transformer frameworks [3, 27, 12, 60, 38, 22, 21] in computer vision, we formulate our framework as a transformer-based approach, which can be guided through the use of query tokens. To add task-specific context to our model, we initialize our queries as repetitions of the task token (obtained from the task input) and compute a query-text contrastive loss [57, 43] with the text derived from the corresponding ground-truth label for the sampled task as shown in Fig. 2. We hypothesize that a contrastive loss on the queries helps guide the model to be more task-sensitive. Furthermore, it also helps reduce the category mispredictions to a certain extent.

We evaluate OneFormer on three major segmentation datasets: ADE20K [15], Cityscapes [14], and COCO [34], each with all three (semantic, instance, and panoptic) segmentation tasks. OneFormer sets the new state of the arts for all three tasks with a single jointly trained model. To summarize, our main contributions are:

- • We propose OneFormer, the first multi-task universal image segmentation framework based on transformers that need to be trained only once with a single universal architecture, a single model, and on a single dataset, to outperform existing frameworks across semantic, instance, and panoptic segmentation tasks, despite the latter need to be trained separately on each task using multiple times of the resources.
We propose OneFormer, the first multi-task universal image segmentation framework based on transformers that need to be trained only once with a single universal architecture, a single model, and on a single dataset, to outperform existing frameworks across semantic, instance, and panoptic segmentation tasks, despite the latter need to be trained separately on each task using multiple times of the resources.

- • OneFormer uses a task-conditioned joint training strategy, uniformly sampling different ground truth domains ( semantic, instance, or panoptic) by deriving all labels from panoptic annotations to train its multi-task model. Thus, OneFormer actually achieves the orignial unification goal of panoptic segmentation [29].
OneFormer uses a task-conditioned joint training strategy, uniformly sampling different ground truth domains ( semantic, instance, or panoptic) by deriving all labels from panoptic annotations to train its multi-task model. Thus, OneFormer actually achieves the orignial unification goal of panoptic segmentation [29].

- • We validate OneFormer through extensive experiments on three major benchmarks: ADE20K [15], Cityscapes [14], and COCO [34]. OneFormer sets a new state-of-the-art performance on all three segmentation tasks compared with methods using the standard Swin-L [38] backbone, and improves even more with new ConvNeXt [39] and DiNAT [21] backbones.
We validate OneFormer through extensive experiments on three major benchmarks: ADE20K [15], Cityscapes [14], and COCO [34]. OneFormer sets a new state-of-the-art performance on all three segmentation tasks compared with methods using the standard Swin-L [38] backbone, and improves even more with new ConvNeXt [39] and DiNAT [21] backbones.

## 2 Related Work

### 2.1 Image Segmentation

Image segmentation is one of the most fundamental tasks in image processing and computer vision. Traditional works usually tackle one of the three image segmentation tasks with specialized network architectures (Fig. 1a).

Semantic Segmentation. Semantic segmentation was long tackled as a pixel classification problem with CNNs [40, 6, 7, 10]. More recent works [26, 44, 56, 27] have shown the success of transformer-based methods in semantic segmentation following its success in language and vision [49, 3]. Among them, MaskFormer [13] treated semantic segmentation as a mask classification problem following early works [20, 4, 16], through using a transformer decoder with object queries [3]. We also formulate semantic segmentation as a mask classification problem.

Instance Segmentation. Traditional instance segmentation methods [23, 2, 5] are also formulated as mask classifiers, which predict binary masks and a class label for each mask. We also formulate instance segmentation as a mask classification problem.

Panoptic Segmentation. Panoptic Segmentation [29] was proposed to unify instance and semantic segmentation. One of the earliest architectures in this scope was Panoptic-FPN [28], which introduced separate instance and semantic task branches. Works that followed significantly improved performance with transformer-based architectures [50, 59, 60, 51, 12, 13]. Despite the progress made so far, panoptic segmentation models are still behind in performance compared to individual instance and semantic segmentation models, therefore not living up to their full unification potential. Motivated by this, we design our OneFormer to be trained with panoptic annotations only.

### 2.2 Universal Image Segmentation

The concept of universal image segmentation has existed for some time, starting with image and scene parsing [58, 48, 47], followed by panoptic segmentation as an effort to unify semantic and instance segmentation [29]. More recently, promising architectures [61, 12, 13] designed specifically for panoptic segmentation have emerged which also perform well on semantic and instance segmentation tasks. K-Net [61], a CNN, uses dynamic learnable instance and semantic kernels with bipartite matching. MaskFormer [13] is a transformer-based architecture, serving as a mask classifier. It was inspired by DETR’s [3] reformulation of object detection in the scope of transformers, where the image is fed to the encoder, and the decoder produces proposals based on queries. Mask2Former [12] improved upon MaskFormer with learnable queries, deformable multi-scale attention [64] in the decoder, a masked cross-attention and set the new state of the art on all three tasks. Unfortunately, it requires training the model individually on each task to achieve the best performance. Therefore, there remains a gap in truly unifying the three segmentation tasks. To the best of our knowledge, OneFormer is the first framework to beat state of the art on all three image segmentation tasks with a single universal model.

### 2.3 Transformer-based Architectures

Architectures based on the transformer encoder-decoder structure [3, 64, 31, 36] have proved effective in object detection since the introduction of DETR [3]. Mask2Former [13, 12] demonstrated the effectiveness of such architectures for image segmentation with a mask classification formulation. Inspired by this success, we also formulate our framework as a query-based mask classification task. Additionally, we claim that calculating a query-text contrastive loss [57, 43] on the task-guided queries can help the model learn inter-task differences and reduce the category mispredictions in the model outputs. Concurrent to our work, LMSeg [1] uses text derived from multiple datasets’ taxonomy to calculate a query-text contrastive loss and tackle the multi-dataset segmentation training challenge. Unlike LMSeg [1], our work focuses on multiple tasks and uses the classes present in the training sample’s ground-truth label to calculate the query-text contrastive loss.

## 3 Method

In this section, we introduce OneFormer, a universal image segmentation framework jointly trained on the panoptic, semantic, and instance segmentation and outperforms individually trained models. We provide an overview of OneFormer in Fig. 2. OneFormer uses two inputs: sample image and task input of the form “the task is {task}”. During our single joint training process, the task is uniformly sampled from {panoptic, instance, semantic} for each image. Firstly, we extract multi-scale features from the input image using a backbone and a pixel decoder. We tokenize the task input to obtain a 1-D task token used to condition the object queries and, consequently, our model on the task for each input. Additionally, we create a text list representing the number of binary masks for each class present in the GT label and map it to text query representations. Note that the text list depends on the input image and the {task}. For supervision of the model’s task-dynamic predictions, we derive the corresponding ground-truths from panoptic annotations. As the ground truth is task-dependent, we calculate a query-text contrastive loss between the object and text queries to ensure there is task distinction in the object queries. The object queries and multi-scale features are fed into a transformer decoder to produce final predictions. We provide more details in the following sections.

### 3.1 Task Conditioned Joint Training

Existing semi-universal architectures for image segmentation [61, 13, 12] face a significant drop in performance when jointly trained on all three segmentation tasks (Tab. 7). We attribute their failure to tackle the multi-task challenge to the absence of task-conditioning in their architecture.

We tackle the multi-task train-once challenge for image segmentation using a task-conditioned joint training strategy. Particularly, we first uniformly sample the task from {panoptic, semantic, instance} for the GT label. We realize the unification potential of panoptic annotations [29] by deriving the task-specific labels from the panoptic annotations, thus, using only one set of annotations.

Next, we extract a set of binary masks for each category present in the image from the task-specific GT label, i.e., semantic task guarantees only one amorphous binary mask for each class present in the image, whereas, instance task signifies non-overlapping binary masks for only thing classes, ignoring the stuff regions. Panoptic task denotes a single amorphous mask for stuff classes and non-overlapping masks for thing classes as shown in Fig. 3. Subsequently, we iterate over the set of masks to create a list of text (𝐓list\mathbf{T}_{\text{list}}) with a template “a photo with a {CLS}”, where CLS is the class name for the corresponding binary mask. The number of binary masks per sample varies over the dataset. Therefore, we pad 𝐓list\mathbf{T}_{\text{list}} with “a/an {task} photo” entries to obtain a padded list (𝐓pad\mathbf{T}_{\text{pad}}) of constant length NtextN_{\text{text}}, with padded entries representing no-object masks. We later use 𝐓pad\mathbf{T}_{\text{pad}} for computing a query-text contrastive loss (Sec. 3.3).

We condition our architecture on the task using a task input (𝐈task\mathbf{I}_{\text{task}}) with the template “the task is {task}”, which is tokenized and mapped to a task-token (𝐐task\mathbf{Q}_{\text{task}}). We use 𝐐task\mathbf{Q}_{\text{task}} to condition OneFormer on the task (Sec. 3.2).

### 3.2 Query Representations

During training, we use two sets of queries in our architecture: text queries (𝐐text\mathbf{Q}_{\text{text}}) and object queries (𝐐\mathbf{Q}). 𝐐text\mathbf{Q}_{\text{text}} is the text-based representation for the segments in the image, while 𝐐\mathbf{Q} is the image-based representation.

To obtain 𝐐text\mathbf{Q}_{\text{text}}, we first tokenize the text entries 𝐓pad\mathbf{T}_{\text{pad}} and pass the tokenized representations through a text-encoder [57], which is a 6-layer transformer [49]. The encoded NtextN_{\text{text}} text embeddings represent the number of binary masks and their corresponding classes in the input image. We further concatenate a set of NctxN_{\text{ctx}} learnable text context embeddings (𝐐ctx\mathbf{Q}_{\text{ctx}}) to the encoded text embeddings to obtain the final NN text queries (𝐐text\mathbf{Q}_{\text{text}}), as shown in Fig. 4. Our motivation behind using 𝐐ctx\mathbf{Q}_{\text{ctx}} is to learn a unified textual context [63, 62] for a sample image. We only use the text queries during training; therefore, we can drop the text mapper module during inference to reduce the model size.

To obtain 𝐐\mathbf{Q}, we first initialize the object queries (𝐐′\mathbf{Q^{\prime}}) as a N−1N-1 times repetitions of the task-token (𝐐task\mathbf{Q}_{\text{task}}). Then, we update 𝐐′\mathbf{Q^{\prime}} with guidance from the flattened 1/41/4-scale features inside a 2-layer transformer [3, 49]. The updated 𝐐′\mathbf{Q^{\prime}} from the transformer (rich with image-contextual information) is concatenated with 𝐐task\mathbf{Q}_{\text{task}} to obtain a task-conditioned representation of NN queries, 𝐐\mathbf{Q}. Unlike the vanilla all-zeros or random initialization [3], the task-guided initialization of the queries and the concatenation with 𝐐task\mathbf{Q}_{\text{task}} is critical for the model to learn multiple segmentation tasks (Sec. 4.3).

### 3.3 Task Guided Contrastive Queries

Developing a single model for all three segmentation tasks is challenging due to the inherent differences among the three tasks. The meaning of the object queries, 𝐐\mathbf{Q}, is task-dependent. Should the queries focus only on the thing classes (instance segmentation), or should the queries predict only one amorphous object for each class present in the image (semantic segmentation) or a mix of both (panoptic segmentation)? Existing query-based architectures [13, 12] do not take such differences into account and hence, fail at effectively training a single model on all three tasks.

To this end, we propose to calculate a query-text contrastive loss using 𝐐\mathbf{Q} and 𝐐text\mathbf{Q}_{\text{text}}. We use 𝐓pad\mathbf{T}_{\text{pad}} to obtain the text queries representation, 𝐐t​e​x​t\mathbf{Q}_{text}, where 𝐓p​a​d\mathbf{T}_{pad} is a list of textual representations for each mask-to-be-detected in a given image with “a/an {task} photo” representing the no-object detections in 𝐐\mathbf{Q} [3]. Thus, the text queries align with the purpose of object queries, representing the objects/segments present [3] in an image. Therefore, we can successfully learn the inter-task distinctions in the query representations using a contrastive loss between the ground truth-derived text and object queries. Moreover, contrastive learning on the queries enables us to attend to inter-class differences and reduce category misclassifications.

|  | ℒ𝐐→𝐐text=−1B∑i=1Blogexp⁡(qio​b​j⊙qit​x​t/τ)∑j=1Bexp⁡(qio​b​j⊙qjt​x​t/τ),ℒ𝐐text→𝐐=−1B∑i=1Blogexp⁡(qit​x​t⊙qio​b​j/τ)∑j=1Bexp⁡(qit​x​t⊙qjo​b​j/τ)ℒ𝐐↔𝐐text=ℒ𝐐→𝐐text+ℒ𝐐text→𝐐\begin{split}\mathcal{L}_{\mathbf{Q}\rightarrow{\mathbf{Q}_{\text{text}}}}&=-\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(q_{i}^{obj}{\odot}q_{i}^{txt}/\tau)}{\sum_{j=1}^{B}\exp(q_{i}^{obj}{\odot}q_{j}^{txt}/\tau)},\\ \mathcal{L}_{{\mathbf{Q}_{\text{text}}}\rightarrow{\mathbf{Q}}}&=-\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(q_{i}^{txt}{\odot}q_{i}^{obj}/\tau)}{\sum_{j=1}^{B}\exp(q_{i}^{txt}{\odot}q_{j}^{obj}/\tau)}\\ \mathcal{L}_{\mathbf{Q}\leftrightarrow{\mathbf{Q}_{\text{text}}}}&=\mathcal{L}_{\mathbf{Q}\rightarrow{\mathbf{Q}_{\text{text}}}}+\mathcal{L}_{{\mathbf{Q}_{\text{text}}}\rightarrow{\mathbf{Q}}}\end{split} |  | (1) |
|---|---|---|---|

Considering that we have a batch of BB object-text query pairs {(qio​b​j,xit​x​t)}i=1B\{(q_{i}^{obj},x_{i}^{txt})\}_{i=1}^{B}, where qio​b​jq_{i}^{obj} and qit​x​tq_{i}^{txt} are the corresponding object and text queries, respectively, of the ii-th pair, we measure the similarity between the queries by calculating a dot product. The total contrastive loss is composed of two losses [57]: (i) an object-to-text contrastive loss (ℒ𝐐→𝐐text\mathcal{L}_{\mathbf{Q}\rightarrow{\mathbf{Q}_{\text{text}}}}) and; (ii) a text-to-object contrastive loss (ℒ𝐐text→𝐐\mathcal{L}_{{\mathbf{Q}_{\text{text}}}\rightarrow{\mathbf{Q}}}) as shown in Eq. 1. τ\tau is a learnable temperature parameter to scale the contrastive logits.

| Method | Backbone | #Params | #FLOPs | #Queries | Crop Size | Iters | Iters | PQ | AP | mIoU (s.s.) | mIoU | (s.s.) | mIoU (m.s.) | mIoU | (m.s.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Iters |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mIoU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (s.s.) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mIoU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (m.s.) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Individual Training |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| UPerNet‡ [55] | UPerNet‡ [55] | SwinV2-L† [37] | — | — | — | ×640640\!\times\!640 | 40k | — | — | — | — | —- | —- | 55.9 |  |
| UPerNet‡ [55] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| —- |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeMask Mask2Former [27] | SeMask Mask2Former [27] | SeMask Swin-L† [27] | 223M | 426G | 200 | ×640640\!\times\!640 | 160k | — | — | — | — | 56.4 | 56.4 | 57.5 |  |
| SeMask Mask2Former [27] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 56.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| UPerNet + K-Net [61] | UPerNet + K-Net [61] | Swin-L† [38] | — | — | — | ×640640\!\times\!640 | 160k | — | — | — | — | — | — | 54.3 |  |
| UPerNet + K-Net [61] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MaskFormer [13] | MaskFormer [13] | Swin-L† [38] | 212M | 375G | 100 | ×640640\!\times\!640 | 160k | — | — | — | — | 54.1 | 54.1 | 55.6 |  |
| MaskFormer [13] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 54.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Panoptic∗ [12] | Mask2Former-Panoptic∗ [12] | Swin-L† [38] | 216M | 413G | 200 | ×640640\!\times\!640 | 160k | 48.7 | 48.7 | 34.2 | 34.2 | 54.5 | 54.5 | — | — |
| Mask2Former-Panoptic∗ [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 34.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 54.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Instance [12] | Mask2Former-Instance [12] | Swin-L† [38] | 216M | 411G | 200 | ×640640\!\times\!640 | 160k | — | — | 34.9 | 34.9 | — | — | — | — |
| Mask2Former-Instance [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 34.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Semantic [12] | Mask2Former-Semantic [12] | Swin-L† [38] | 215M | 403G | 100 | ×640640\!\times\!640 | 160k | — | — | — | — | 56.1 | 56.1 | 57.3 | 57.3 |
| Mask2Former-Semantic [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 56.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kMaX-DeepLab‡‡ [60] | kMaX-DeepLab‡‡ [60] | ConvNeXt-L† [39] | 232M | 333G | 256 | ×641641\!\times\!641 | 100k | 48.7 | 48.7 | — | — | 54.8 | 54.8 | — |  |
| kMaX-DeepLab‡‡ [60] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 54.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kMaX-DeepLab‡‡ [60] | kMaX-DeepLab‡‡ [60] | ConvNeXt-L† [39] | 232M | 1302G | 256 | ×12811281\!\times\!1281 | 100k | 50.9 | 50.9 | — | — | 55.2 | 55.2 | — |  |
| kMaX-DeepLab‡‡ [60] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 50.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 55.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| UPerNet‡‡ [55] | UPerNet‡‡ [55] | SwinV2-G†† [37] | >>3B | — | — | ×640640\!\times\!640 | 80k | — | — | — | — | 59.1 | 59.1 | — |  |
| UPerNet‡‡ [55] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 59.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former‡‡ [12] | Mask2Former‡‡ [12] | BEiT-3†† [52] | 1.9B | — | — | ×896896\!\times\!896 | — | — | — | — | — | 62.0 | 62.0 | 62.8 |  |
| Mask2Former‡‡ [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 62.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Joint Training |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | Swin-L† [38] | 219M | 436G | 250 | ×640640\!\times\!640 | 160k | 49.8 | 49.8 | 35.9 | 35.9 | 57.0 | 57.0 | 57.7 | 57.7 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 49.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 35.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | Swin-L† [38] | 219M | 801G | 250 | ×896896\!\times\!896 | 160k | 51.1 | 51.1 | 37.6 | 37.6 | 57.4 | 57.4 | 58.3 | 58.3 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 51.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 37.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | Swin-L† [38] | 219M | 1597G | 250 | ×12801280\!\times\!1280 | 160k | 51.4 | 51.4 | 37.8 | 37.8 | 57.0 | 57.0 | 57.7 | 57.7 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 51.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 37.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | ConvNeXt-L† [39] | 220M | 389G | 250 | ×640640\!\times\!640 | 160k | 50.0 | 50.0 | 36.2 | 36.2 | 56.6 | 56.6 | 57.4 | 57.4 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 50.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 36.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 56.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | ConvNeXt-XL† [39] | 372M | 607G | 250 | ×640640\!\times\!640 | 160k | 50.1 | 50.1 | 36.3 | 36.3 | 57.4 | 57.4 | 58.8 | 58.8 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 50.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 36.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | DiNAT-L† [21] | 223M | 359G | 250 | ×640640\!\times\!640 | 160k | 50.5 | 50.5 | 36.0 | 36.0 | 58.3 | 58.3 | 58.4 | 58.4 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 50.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 36.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | DiNAT-L† [21] | 223M | 678G | 250 | ×896896\!\times\!896 | 160k | 51.2 | 51.2 | 36.8 | 36.8 | 58.1 | 58.1 | 58.6 | 58.6 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 51.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 36.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | DiNAT-L† [21] | 223M | 1369G | 250 | ×12801280\!\times\!1280 | 160k | 51.5 | 51.5 | 37.1 | 37.1 | 58.2 | 58.2 | 58.7 | 58.7 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 51.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 37.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

| Iters |
|---|

| mIoU |
|---|
| (s.s.) |

| mIoU |
|---|
| (m.s.) |

| UPerNet‡ [55] |
|---|

| — |
|---|

| — |
|---|

| —- |
|---|

| SeMask Mask2Former [27] |
|---|

| — |
|---|

| — |
|---|

| 56.4 |
|---|

| UPerNet + K-Net [61] |
|---|

| — |
|---|

| — |
|---|

| — |
|---|

| MaskFormer [13] |
|---|

| — |
|---|

| — |
|---|

| 54.1 |
|---|

| Mask2Former-Panoptic∗ [12] |
|---|

| 48.7 |
|---|

| 34.2 |
|---|

| 54.5 |
|---|

| — |
|---|

| Mask2Former-Instance [12] |
|---|

| — |
|---|

| 34.9 |
|---|

| — |
|---|

| — |
|---|

| Mask2Former-Semantic [12] |
|---|

| — |
|---|

| — |
|---|

| 56.1 |
|---|

| 57.3 |
|---|

| kMaX-DeepLab‡‡ [60] |
|---|

| 48.7 |
|---|

| — |
|---|

| 54.8 |
|---|

| kMaX-DeepLab‡‡ [60] |
|---|

| 50.9 |
|---|

| — |
|---|

| 55.2 |
|---|

| UPerNet‡‡ [55] |
|---|

| — |
|---|

| — |
|---|

| 59.1 |
|---|

| Mask2Former‡‡ [12] |
|---|

| — |
|---|

| — |
|---|

| 62.0 |
|---|

| OneFormer |
|---|

| 49.8 |
|---|

| 35.9 |
|---|

| 57.0 |
|---|

| 57.7 |
|---|

| OneFormer |
|---|

| 51.1 |
|---|

| 37.6 |
|---|

| 57.4 |
|---|

| 58.3 |
|---|

| OneFormer |
|---|

| 51.4 |
|---|

| 37.8 |
|---|

| 57.0 |
|---|

| 57.7 |
|---|

| OneFormer |
|---|

| 50.0 |
|---|

| 36.2 |
|---|

| 56.6 |
|---|

| 57.4 |
|---|

| OneFormer |
|---|

| 50.1 |
|---|

| 36.3 |
|---|

| 57.4 |
|---|

| 58.8 |
|---|

| OneFormer |
|---|

| 50.5 |
|---|

| 36.0 |
|---|

| 58.3 |
|---|

| 58.4 |
|---|

| OneFormer |
|---|

| 51.2 |
|---|

| 36.8 |
|---|

| 58.1 |
|---|

| 58.6 |
|---|

| OneFormer |
|---|

| 51.5 |
|---|

| 37.1 |
|---|

| 58.2 |
|---|

| 58.7 |
|---|

### 3.4 Other Architecture Components

Backbone and Pixel Decoder: We use the widely used ImageNet [30] pre-trained backbones [38, 39, 21] to extract multi-scale feature representations from the input image. Our pixel decoder aids the feature modeling by gradually upsampling the backbone features. Motivated by the recent success of multi-scale deformable attention [64, 12], we use the same Multi-Scale Deformable Transformer (MSDeformAttn) based architecture for our pixel decoder.

Transformer Decoder: We use a multi-scale strategy [12] to utilize the higher resolution maps inside our transformer decoder. Specifically, we feed the object queries (𝐐\mathbf{Q}) and the multi-scale outputs from the pixel decoder (FiF_{i}), i∈{1/4,1/8,1/16,1/32}i\in\{1/4,1/8,1/16,1/32\} as inputs. We use the features with resolution 1/81/8, 1/161/16 and 1/321/32 of the original image alternatively to update 𝐐\mathbf{Q} using a masked cross-attention (CA) operation [12], followed by a self-attention (SA) and finally a feed-forward network (FFN). We perform these sets of alternate operations LL times inside the transformer decoder.

The final query outputs from the transformer decoder are mapped to a K+1K+1 dimensional space for class predictions, where KK denotes the number of classes and an extra +1+1 for the no-object predictions. To obtain the final masks, we decode the pixel features (F1/4F_{1/4}) at 1/41/4 resolution of the original image with the help of an einsum operation between QQ and F1/4F_{1/4}. During inference, we follow the same post-processing technique as [12] to obtain the final panoptic, semantic, and instance segmentation predictions. We only keep predictions with scores above a threshold of 0.5, 0.8, and 0.8 during post-processing for panoptic segmentation on the ADE20K [15], Cityscapes [14] and COCO [34] datasets, respectively.

### 3.5 Losses

In addition to the contrastive loss on the queries, we calculate the standard classification CE-loss (ℒcls\mathcal{L}_{\text{cls}}) over the class predictions. Following [12], we use a combination of binary cross-entropy (ℒbce\mathcal{L}_{\text{bce}}) and dice loss (ℒdice\mathcal{L}_{\text{dice}}) over the mask predictions. Therefore, our final loss function is a weighted sum of the four losses (Eq. 2). We empirically set λ𝐐↔𝐐text=0.5\lambda_{\mathbf{Q}\leftrightarrow{\mathbf{Q}_{\text{text}}}}=0.5, λcls=2\lambda_{\text{cls}}=2, λbce=5\lambda_{\text{bce}}=5 and λdice=5\lambda_{\text{dice}}=5. To find the least cost assignment, we use bipartite matching [3, 13] between the set predictions and the ground truths. We set λcls\lambda_{\text{cls}} as 0.10.1 for the no-object predictions [12].

|  | ℒfinal=λ𝐐↔𝐐text​ℒ𝐐↔𝐐text+λcls​ℒc​l​s+λbce​ℒb​c​e+λdice​ℒd​i​c​e\begin{split}\mathcal{L}_{\text{final}}&=\lambda_{\mathbf{Q}\leftrightarrow{\mathbf{Q}_{\text{text}}}}\mathcal{L}_{\mathbf{Q}\leftrightarrow{\mathbf{Q}_{\text{text}}}}+\lambda_{\text{cls}}\mathcal{L}_{cls}\\ &+\lambda_{\text{bce}}\mathcal{L}_{bce}+\lambda_{\text{dice}}\mathcal{L}_{dice}\end{split} |  | (2) |
|---|---|---|---|

## 4 Experiments

We illustrate that OneFormer, when trained only once with our task-conditioned joint-training strategy, generalizes well to all three image segmentation tasks on three widely used datasets. Furthermore, we provide extensive ablations to demonstrate the significance of OneFormer’s components. Due to space constraints, we provide implementation details in the appendix.

| Method | Backbone | #Params | #FLOPs | #Queries | Crop Size | Iters | Iters | PQ | AP | mIoU (s.s.) | mIoU | (s.s.) | mIoU (m.s.) | mIoU | (m.s.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Iters |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mIoU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (s.s.) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mIoU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (m.s.) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Individual Training |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| CMT-DeepLab‡ [59] | CMT-DeepLab‡ [59] | MaX-S† [50] | — | — | — | ×20491025\!\times\!2049 | 60k | 64.6 | 64.6 | — | — | 81.4 | 81.4 | — |  |
| CMT-DeepLab‡ [59] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 81.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Axial-DeepLab-L‡ [51] | Axial-DeepLab-L‡ [51] | Axial ResNet-L† [51] | 45M | 687G | — | ×20491025\!\times\!2049 | 60k | 63.9 | 63.9 | 35.8 | 35.8 | 81.0 | 81.0 | 81.5 |  |
| Axial-DeepLab-L‡ [51] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 63.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 35.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 81.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Axial-DeepLab-XL‡ [51] | Axial-DeepLab-XL‡ [51] | Axial ResNet-XL† [51] | 173M | 2447G | — | ×20491025\!\times\!2049 | 60k | 64.4 | 64.4 | 36.7 | 36.7 | 80.6 | 80.6 | 81.1 |  |
| Axial-DeepLab-XL‡ [51] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 36.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 80.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Panoptic-DeepLab‡ [11] | Panoptic-DeepLab‡ [11] | SWideRNet† [8] | 536M | 10365G | — | ×20491025\!\times\!2049 | 60k | 66.4 | 66.4 | 40.1 | 40.1 | 82.2 | 82.2 | 82.9 |  |
| Panoptic-DeepLab‡ [11] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 66.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 40.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 82.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Panoptic [12] | Mask2Former-Panoptic [12] | Swin-L† [38] | 216M | 514G | 200 | ×1024512\!\times\!1024 | 90k | 66.6 | 66.6 | 43.6 | 43.6 | 82.9 | 82.9 | — | — |
| Mask2Former-Panoptic [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 66.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 43.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 82.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Instance [12] | Mask2Former-Instance [12] | Swin-L† [38] | 216M | 507G | 200 | ×1024512\!\times\!1024 | 90k | — | — | 43.7 | 43.7 | — | — | — | — |
| Mask2Former-Instance [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 43.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Semantic [12] | Mask2Former-Semantic [12] | Swin-L† [38] | 215M | 494G | 100 | ×1024512\!\times\!1024 | 90k | — | — | — | — | 83.3 | 83.3 | 84.3 | 84.3 |
| Mask2Former-Semantic [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 83.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kMaX-DeepLab‡ [60] | kMaX-DeepLab‡ [60] | ConvNeXt-L† [39] | 232M | 1673G | 256 | ×20491025\!\times\!2049 | 60k | 68.4 | 68.4 | 44.0 | 44.0 | 83.5 | 83.5 | — |  |
| kMaX-DeepLab‡ [60] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 44.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 83.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Joint Training |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | Swin-L† [38] | 219M | 543G | 250 | ×1024512\!\times\!1024 | 90k | 67.2 | 67.2 | 45.6 | 45.6 | 83.0 | 83.0 | 84.4 | 84.4 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 67.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 45.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 83.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | ConvNeXt-L† [39] | 220M | 497G | 250 | ×1024512\!\times\!1024 | 90k | 68.5 | 68.5 | 46.5 | 46.5 | 83.0 | 83.0 | 84.0 | 84.0 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 46.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 83.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | ConvNeXt-XL† [39] | 372M | 775G | 250 | ×1024512\!\times\!1024 | 90k | 68.4 | 68.4 | 46.7 | 46.7 | 83.6 | 83.6 | 84.6 | 84.6 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 46.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 83.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | DiNAT-L† [21] | 223M | 450G | 250 | ×1024512\!\times\!1024 | 90k | 67.6 | 67.6 | 45.6 | 45.6 | 83.1 | 83.1 | 84.0 | 84.0 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 67.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 45.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 83.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

| Iters |
|---|

| mIoU |
|---|
| (s.s.) |

| mIoU |
|---|
| (m.s.) |

| CMT-DeepLab‡ [59] |
|---|

| 64.6 |
|---|

| — |
|---|

| 81.4 |
|---|

| Axial-DeepLab-L‡ [51] |
|---|

| 63.9 |
|---|

| 35.8 |
|---|

| 81.0 |
|---|

| Axial-DeepLab-XL‡ [51] |
|---|

| 64.4 |
|---|

| 36.7 |
|---|

| 80.6 |
|---|

| Panoptic-DeepLab‡ [11] |
|---|

| 66.4 |
|---|

| 40.1 |
|---|

| 82.2 |
|---|

| Mask2Former-Panoptic [12] |
|---|

| 66.6 |
|---|

| 43.6 |
|---|

| 82.9 |
|---|

| — |
|---|

| Mask2Former-Instance [12] |
|---|

| — |
|---|

| 43.7 |
|---|

| — |
|---|

| — |
|---|

| Mask2Former-Semantic [12] |
|---|

| — |
|---|

| — |
|---|

| 83.3 |
|---|

| 84.3 |
|---|

| kMaX-DeepLab‡ [60] |
|---|

| 68.4 |
|---|

| 44.0 |
|---|

| 83.5 |
|---|

| OneFormer |
|---|

| 67.2 |
|---|

| 45.6 |
|---|

| 83.0 |
|---|

| 84.4 |
|---|

| OneFormer |
|---|

| 68.5 |
|---|

| 46.5 |
|---|

| 83.0 |
|---|

| 84.0 |
|---|

| OneFormer |
|---|

| 68.4 |
|---|

| 46.7 |
|---|

| 83.6 |
|---|

| 84.6 |
|---|

| OneFormer |
|---|

| 67.6 |
|---|

| 45.6 |
|---|

| 83.1 |
|---|

| 84.0 |
|---|

| Method | Backbone | #Params | #FLOPs | #Queries | Epochs | Epochs | PQ | PQTh{}^{\text{Th}} | PQSt{}^{\text{St}} | AP | APinstance{}^{\text{instance}} | mIoU | mIoU |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Epochs |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mIoU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Individual Training |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MaskFormer [13] | MaskFormer [13] | Swin-L† [38] | 212M | 792G | 100 | 300 | 52.7 | 58.5 | 58.5 | 44.0 | 44.0 | — | — | 64.8 |  |  |  |
| MaskFormer [13] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 44.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| K-Net [61] | K-Net [61] | Swin-L† [38] | — | — | 100 | 36 | 54.6 | 60.2 | 60.2 | 46.0 | 46.0 | — | — | — |  |  |  |
| K-Net [61] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 60.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 46.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Panoptic SegFormer [33] | Panoptic SegFormer [33] | Swin-L† [38] | 221M | 816G | 353 | 24 | 55.8 | 61.7 | 61.7 | 46.9 | 46.9 | — | — | — |  |  |  |
| Panoptic SegFormer [33] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 61.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 46.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Panoptic [12] | Mask2Former-Panoptic [12] | Swin-L† [38] | 216M | 875G | 200 | 100 | 57.8 | 57.8 | 64.2 | 64.2 | 48.1 | 48.1 | 48.7 | 48.7 | 48.6 | 67.4 | 67.4 |
| Mask2Former-Panoptic [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 67.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Instance [12] | Mask2Former-Instance [12] | Swin-L† [38] | 216M | 868G | 200 | 100 | — | — | — | — | — | — | 49.1 | 49.1 | 50.1 | — | — |
| Mask2Former-Instance [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 49.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Semantic‡ [12] | Mask2Former-Semantic‡ [12] | Swin-L† [38] | 216M | 891G | 200 | 100 | — | — | — | — | — | — | — | — | — | 67.2 | 67.2 |
| Mask2Former-Semantic‡ [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 67.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kMaX-DeepLab∗ [60] | kMaX-DeepLab∗ [60] | ConvNeXt-L† [39] | 232M | 749G | 128 | 81 | 57.9 | 64.0 | 64.0 | 48.6 | 48.6 | — | — | — |  |  |  |
| kMaX-DeepLab∗ [60] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kMaX-DeepLab∗ [60] | kMaX-DeepLab∗ [60] | ConvNeXt-L† [39] | 232M | 749G | 256 | 81 | 58.0 | 64.2 | 64.2 | 48.6 | 48.6 | — | — | — |  |  |  |
| kMaX-DeepLab∗ [60] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Joint Training |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | Swin-L† [38] | 219M | 891G | 150 | 100 | 57.9 | 57.9 | 64.4 | 64.4 | 48.0 | 48.0 | 49.0 | 49.0 | 48.9 | 67.4 | 67.4 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 49.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 67.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | DiNAT-L† [21] | 223M | 736G | 150 | 100 | 58.0 | 58.0 | 64.3 | 64.3 | 48.4 | 48.4 | 49.2 | 49.2 | 49.2 | 68.1 | 68.1 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 49.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

| Epochs |
|---|

| mIoU |
|---|

| MaskFormer [13] |
|---|

| 58.5 |
|---|

| 44.0 |
|---|

| K-Net [61] |
|---|

| 60.2 |
|---|

| 46.0 |
|---|

| Panoptic SegFormer [33] |
|---|

| 61.7 |
|---|

| 46.9 |
|---|

| Mask2Former-Panoptic [12] |
|---|

| 57.8 |
|---|

| 64.2 |
|---|

| 48.1 |
|---|

| 48.7 |
|---|

| 67.4 |
|---|

| Mask2Former-Instance [12] |
|---|

| — |
|---|

| — |
|---|

| — |
|---|

| 49.1 |
|---|

| — |
|---|

| Mask2Former-Semantic‡ [12] |
|---|

| — |
|---|

| — |
|---|

| — |
|---|

| — |
|---|

| 67.2 |
|---|

| kMaX-DeepLab∗ [60] |
|---|

| 64.0 |
|---|

| 48.6 |
|---|

| kMaX-DeepLab∗ [60] |
|---|

| 64.2 |
|---|

| 48.6 |
|---|

| OneFormer |
|---|

| 57.9 |
|---|

| 64.4 |
|---|

| 48.0 |
|---|

| 49.0 |
|---|

| 67.4 |
|---|

| OneFormer |
|---|

| 58.0 |
|---|

| 64.3 |
|---|

| 48.4 |
|---|

| 49.2 |
|---|

| 68.1 |
|---|

### 4.1 Datasets and Evaluation Metrics

Datasets. We experiment on three widely used datasets that support all three: semantic, instance, and panoptic segmentation tasks. Cityscapes [14] consists of a total 19 (11 “stuff” and 8 “thing”) classes with 2,975 training, 500 validation and 1,525 test images. ADE20K [15] is another benchmark dataset with 150 (50 “stuff” and 100 “thing”) classes among the 20,210 training and 2,000 validation images. COCO [34] has 133 (53 “stuff” and 80 “thing”) classes with 118k training and 5,000 validation images.

Evaluation Metrics. For all three image segmentation tasks, we report the PQ [29], AP [34], and mIoU [18] scores. Since we only have a single model for all three tasks, we use the value of the task token to decide the scores to consider. For e.g., when task is panoptic, we report the PQ score and similarly we report AP and mIoU scores when task is instance and semantic, respectively.

### 4.2 Main Results

ADE20K. We compare OneFormer with the existing state-of-the-art pseudo-universal and specialized architectures on the ADE20K [15] val dataset in Tab. 1. With the standard Swin-L† backbone, OneFormer, while being trained only once, outperforms Mask2Former’s [12] individually trained models on all three image segmentation tasks and sets a new state-of-the-art performance when compared with other methods using the same backbone.

Cityscapes. We compare OneFormer with the existing state-of-the-art pseudo-universal and specialized architectures on the Cityscapes [15] val dataset in Tab. 2. With Swin-L† backbone, OneFormer outperforms Mask2Former with a +0.6%+\mathbf{0.6}\% and +1.9%+\mathbf{1.9}\% improvement on the PQ and AP metrics, respectively. Additionally, with ConvNeXt-L† and ConvNeXt-XL† backbone, OneFormer sets a new state-of-the-art of 68.5%68.5\% PQ and 46.7%46.7\% AP, respectively.

COCO. We compare OneFormer with the existing state-of-the-art pseudo-universal and specialized architectures on the COCO [34] val2017 dataset in Tab. 3. With Swin-L† backbone, OneFormer performs on-par with the individually trained Mask2Former [12] with a +0.1%+0.1\% improvement in the PQ score. Due to the discrepancies between the panoptic and instance annotations in COCO [34], we evaluate the AP score using the instance ground truths derived from the panoptic annotations. We provide more information in the appendix. Following [12], we evaluate mIoU on semantic ground truths derived from panoptic annotations.

### 4.3 Ablation Studies

We analyze OneFormer’s components through a series of ablation studies. Unless stated otherwise, we ablate with Swin-L† OneFormer on the Cityscapes [14] dataset.

Task-Conditioned Architecture. We validate the importance of the task token (𝐐task\mathbf{Q}_{\text{task}}), initializing the queries with repetitions of the task token (task-guided query init.) and the learnable text context (𝐐ctx\mathbf{Q}_{\text{ctx}}) by removing each component one at a time in Tab. 4. Without the task token, we observe a significant drop in the AP score (−2.7%-2.7\%). Furthermore, using a learnable text context (𝐐ctx\mathbf{Q}_{\text{ctx}}) leads to an improvement of +4.5%+4.5\% in the PQ score, proving its significance. Lastly, initializing the queries as repetitions of the task token (task-guided query init.) instead of using an all-zeros initialization [3] leads to an improvement of +1.4%+1.4\% in the PQ and +1.1%+1.1\% in the AP score, indicating the importance of task-conditioning the initialization of the queries.

Contrastive Query Loss. We report results without the query-text contrastive loss (ℒ𝐐↔𝐐text\mathcal{L}_{\mathbf{Q}\leftrightarrow{\mathbf{Q}_{\text{text}}}}) in Tab. 5. We observe that the contrastive loss significantly benefits the PQ (+8.4%+8.4\%) and AP (+3.2%+3.2\%) scores. We also conduct experiments substituting our query-text contrastive loss with a classification loss (ℒcls\mathcal{L}_{\text{cls}}) on the queries. ℒcls\mathcal{L}_{\text{cls}} can be regarded as a straightforward alternative for ℒ𝐐↔𝐐text\mathcal{L}_{\mathbf{Q}\leftrightarrow{\mathbf{Q}_{\text{text}}}} as the both provide supervision for the number of masks for each class present in the image. However, we observe significant drops on all the metrics (−0.8%-0.8\% PQ, −0.9%-0.9\% AP and −0.4%-0.4\% mIoU) using the classification loss instead of the contrastive loss. We attribute the drops to the inability of the classification loss to capture the inter-task differences effectively.

|  | PQ | AP | mIoU |
|---|---|---|---|
| OneFormer (ours) | 67.2 | 45.6 | 83.0 |
| −- task-token (𝐐task\mathbf{Q}_{\text{task}}) | 66.5 (-0.7) | 43.3 (-2.3) | 82.9 (-0.1) |
| −- learnable text context (𝐐ctx\mathbf{Q}_{\text{ctx}}) | 62.7 (-4.5) | 45.0 (-0.6) | 82.8 (-0.2) |
| −- task-guided query init. | 65.8 (-1.4) | 44.5 (-1.1) | 83.1 (+0.1) |

|  | PQ | AP | mIoU. | #param. |
|---|---|---|---|---|
| contrastive-loss (ours) | 67.2 | 45.6 | 83.0 | 219M |
| query classification-loss | 66.4 (-0.8) | 44.7 (-0.9) | 82.6 (-0.4) | 219M |
| no contrastive-loss | 58.8 (-8.4) | 42.4 (-3.2) | 82.5 (-0.5) | 219M |

|  | PQ | AP | mIoU |
|---|---|---|---|
| “a photo with a {CLS}” (ours) | 67.2 | 45.6 | 83.0 |
| “a photo with a {CLS} {TYPE}” | 65.4 (-1.8) | 44.5 (-1.1) | 82.8 (-0.2) |
| “{CLS}” | 66.6 (-0.6) | 44.7 (-0.9) | 82.5 (-0.5) |

Input Text Template. We study the importance of the template choice for the entries in the text list (𝐓list\mathbf{T}_{\text{list}}) in Tab. 6. We experiment with “a photo with a {CLS} {TYPE}” template for our text entries where CLS is the class name for the object mask and TYPE is the task-dependent class-type: “stuff” for amorphous masks (panoptic and semantic task) and “thing” for all distinct object masks. We also experiment with the identity template “{CLS}”. Our choice of the template: “a photo with a {CLS}” gives a strong performance as a baseline. We believe more exploration in the text template space could help in improving the performance further.

Task Conditioned Joint Training. As a baseline for comparison, we train a Swin-L† Mask2Former-Joint with our joint training strategy, i.e., uniformly sampling each task’s GT on the ADE20K [15] dataset. We compare the Mask2Former-Joint baseline with our Swin-L† OneFormer in Tab. 7. We train both models for 160k iterations with a batch size of 16. Our OneFormer achieves a +1.1%+1.1\%, +2.2%+2.2\%, and +0.8%+0.8\% improvement on the PQ, AP and mIoU metrics, respectively, proving the importance of our architecture design for practical multi-task joint training.

|  | PQ | AP | mIoU | #param. |
|---|---|---|---|---|
| OneFormer (ours) | 49.8 | 35.9 | 57.0 | 219M |
| Mask2Former-Joint | 48.7 (-1.1) | 33.7 (-2.2) | 56.2 (-0.8) | 216M |

| Task Token Input | PQ | PQTh{}^{\text{Th}} | PQSt{}^{\text{St}} | AP | mIoU |
|---|---|---|---|---|---|
| the task is panoptic | 49.3 | 49.6 | 50.2 | 35.8 | 57.0 |
| the task is instance | 33.1 | 48.8 | 1.5 | 35.9 | 26.4 |
| the task is semantic | 40.4 | 35.5 | 50.2 | 25.3 | 57.0 |

Task Token Input. We demonstrate that our framework is sensitive to the task token input by setting the value of {task} during inference as panoptic, instance, or semantic in Tab. 8. We report results with our Swin-L† OneFormer trained on ADE20K [15] dataset. We observe a significant drop in the PQ and mIoU metrics when task is instance compared to panoptic. Moreover, the PQSt{}^{\text{St}} drops to 1.5%1.5\%, and there is only a −0.8%-0.8\% drop on PQTh{}^{\text{Th}} metric, proving that the network learns to focus majorly on the distinct “thing” instances when the task is instance. Similarly, there is a sizable drop in the PQ, PQTh{}^{\text{Th}} and AP metrics for the semantic task with PQSt{}^{\text{St}} staying the same, showing that our framework can segment out amorphous masks for “stuff” regions but does not predict different masks for “thing” objects. Therefore, OneFormer dynamically learns the inter-task distinctions which is critical for a train-once multi-task architecture. We include qualitative analysis on the task dynamic nature of OneFormer in the appendix.

Reduced Category Misclassifications. Our query-text contrastive loss helps OneFormer learn the inter-task distinctions and reduce the number of category misclassifications in the predictions. Mask2Former incorrectly predicts “wall” as “fence” in the first row, “vegetation” as “terrain”, and “terrain” as “sidewalk”. At the same time, our OneFormer produces more accurate predictions in regions (inside blue boxes) with similar classes, as shown in Fig. 5.

## 5 Conclusion

In this work, we present OneFormer, a new multi-task universal image segmentation framework with transformers and task-guided queries to unify semantic, instance, and panoptic segmentation with a single universal architecture, a single model, and training on a single dataset. Our jointly trained single OneFormer model outperforms the individually trained specialized Mask2Former models, the previous single-architecture state of the art, on all three segmentation tasks across major datasets. Consequently, OneFormer can cut training time, weight storage, and inference hosting requirements down to a third, making image segmentation more accessible. We believe OneFormer is a significant step towards making image segmentation more universal and accessible and will support further research in this direction by open-sourcing our codes and models.

We thank Intelligence Advanced Research Projects Activity (IARPA), University of Oregon, University of Illinois at Urbana-Champaign, and Picsart AI Research (PAIR) for their generous support that made this work possible.

Appendix

## Appendix A Implementation Details

We implement our framework using the Detectron2 [54] library.

Multi-Scale Feature Modeling. We adopt the settings from [12] for modeling the image pixel-level features. More specifically, we use 6 MSDeformAttn [64] inside our pixel decoder, applied to feature maps with resolutions 1/81/8, 1/161/16, and 1/321/32 of the original image. We use lateral connections and upsampling to aggregate the multi-scale features to a final 1/41/4 resolution scale. We map all the features to a hidden dimension of 256256.

Unified Task-Conditioned Query Formulation. We initialize the N−1N-1 queries as repetitions of task-token, 𝐐task\mathbf{Q}_{\text{task}}. Unless stated otherwise, we set N=250N=250 and Nctx=16N_{\text{ctx}}=16. Our text tokenizer and text encoder are the same as [57]. We use a single linear layer to project the tokenized task input, followed by a layer-norm to obtain 𝐐task\mathbf{Q}_{\text{task}}.

Task-Dynamic Mask and Class Prediction Formation. Following [12], we set L=3L=3 inside the transformer decoder. Therefore, we have a total of 3​L3L (9) stages inside our transformer decoder. We also calculate an auxiliary loss on each intermediate class and mask predictions after every transformer decoder stage [12].

Training Settings. We train our model with a batch size of 16. When training on ADE20K [15] and Cityscapes [14], we use the AdamW [41] optimizer with a base learning rate of 0.00010.0001, poly learning rate decay and weight decay 0.10.1. We use a crop size of ×512512\!\times\!512 and ×1024512\!\times\!1024 on ADE20K and Cityscapes, respectively. We train for 90k and 160k iterations on Cityscapes and ADE20K, respectively. For data augmentation, we use shortest edge resizing, fixed size cropping, and color jittering followed by a random horizontal flip.

When training on COCO [34], we use a step learning rate schedule along with the AdamW [41] optimizer, a base learning rate of 0.00010.0001, 1010 warmup iterations, and a weight decay of 0.050.05. We decay the learning rate at 0.90.9 and 0.950.95 fractions of the total number of training steps by a factor of 1010. We train for a total of 100 epochs with LSJ augmentation [19, 17] with a random scale sampled from the range 0.10.1 to 2.02.0 followed by a fixed size crop to ×10241024\!\times\!1024 resolution.

Evaluation Settings. We follow the same evaluation settings as Mask2Former [12]. Unless stated otherwise, we report results for the single-scale inference setting. Unlike the training stage, during evaluation, we use the ground-truth annotations from the respective task GT labels to calculate the metric scores instead of deriving the labels from the panoptic annotations. Additionally, we set the value of task in “the task is {task}” as panoptic, instance and semantic to obtain the corresponding task predictions.

## Appendix B Additional Ablations

Ablation on Number of Queries. We study the effect of the different number of queries on the COCO dataset in Tab. I. We conduct experiments using the ResNet-50 (R50) [24] backbone and train for 50 epochs. We find that N=150N=150 performs the best.

Additionally, we tune the number of queries on the Swin-L† backbone separately. During our experiments, we found that N=250N=250 is the best setting with Swin-l† on ADE20K [15] and Cityscapes [14] datasets. On COCO [34], N=150N=150 gives the best performance with Swin-L†. We also noticed that with smaller backbones like R50 [24], N=150N=150 is the optimal setting on the ADE20K [15] dataset.

| #queries | PQ | AP | mIoU | #param. |
|---|---|---|---|---|
| 100 | 51.3 | 41.9 | 60.8 | 47M |
| 120 | 51.0 | 42.0 | 60.8 | 47M |
| 150 | 51.5 | 42.5 | 61.2 | 47M |
| 200 | 51.3 | 42.5 | 60.0 | 47M |

| NctxN_{\text{ctx}} | PQ | AP | mIoU | #param. |
|---|---|---|---|---|
| 0 | 41.7 | 27.5 | 46.5 | 47M |
| 8 | 41.0 | 27.2 | 46.5 | 47M |
| 16 | 41.9 | 27.3 | 47.3 | 47M |
| 32 | 41.7 | 27.5 | 46.8 | 47M |

| contrastive-loss weight | PQ | AP | mIoU |
|---|---|---|---|
| λ𝐐↔𝐐text=0.0\lambda_{\mathbf{Q}\leftrightarrow{\mathbf{Q}_{\text{text}}}}=0.0 | 51.1 | 42.1 | 60.2 |
| λ𝐐↔𝐐text=0.5\lambda_{\mathbf{Q}\leftrightarrow{\mathbf{Q}_{\text{text}}}}=0.5 | 51.5 | 42.5 | 61.2 |
| λ𝐐↔𝐐text=1.0\lambda_{\mathbf{Q}\leftrightarrow{\mathbf{Q}_{\text{text}}}}=1.0 | 50.7 | 42.0 | 60.5 |

Ablation on Contrastive Loss’ Weight. We run ablations on the weight for the contrastive loss’ weight on the COCO dataset in Tab. III. We conduct our experiments using the ResNet-50 (R50) [24] backbone and train for 50 epochs. We find that λ𝐐↔𝐐text=0.5\lambda_{\mathbf{Q}\leftrightarrow{\mathbf{Q}_{\text{text}}}}=0.5 is the optimal weight setting.

Ablation on Number of Learnable Text Context Embeddings. We study the effect of different number of learnable text context embeddings on the ADE20K [15] dataset in Tab. II. We conduct our experiments using the ResNet-50 (R50) [24] backbone and train for 160k iterations. We find that Nctx=16N_{\text{ctx}}=16 performs best.

## Appendix C Individual Training

In this section, we analyze our OneFormer’s performance with individual training on the panoptic, instance, and semantic segmentation task. For this study, we conduct experiments with the ResNet-50 (R50) [24] backbone on the ADE20K [15] dataset. We train all models for 160k iterations with a batch size of 16.

As shown in Tab. IV, OneFormer outperforms Mask2Former [12] (the previous SOTA pseudo-universal image segmentation method) with every training strategy. Furthermore, with joint training, Mask2Former [12] suffers a significant drop in performance, and OneFormer achieves the highest PQ, AP and mIoU scores.

In order to train OneFormer on a single task, we set the value of task as that of the corresponding task in our task token input: “the task is {task}” for the samples during training. Therefore, under Panoptic Training, only panoptic ground truth labels will be used, and similarly, for Semantic and Instance Training, only semantic and instance ground truth labels shall be used, respectively. The joint training strategy remains the same as described in Sec 3.1 (main text) with uniform sampling for each task-specific ground truth label. Note that for training OneFormer, we derive all ground truth labels from the panoptic annotations.

## Appendix D Analysis on the Task-Dynamic Nature of OneFormer

We analyze OneFormer’s ability to capture the inter-task differences by changing the value of {task} in the task token input: “the task is {task}” as panoptic, instance, or semantic, during inference. We report quantitative report results with our Swin-L† OneFormer trained on Cityscapes [14] dataset in Tab. V. When we set task as “instance”, we observe that PQSt{}^{\text{St}} drops to 0.0%0.0\%, and there is only a −0.2%-0.2\% drop on PQTh{}^{\text{Th}} metric as compared to the setting when task is panoptic. This observation proves that OneFormer learns to change its feed-forward output depending on the task dynamically. Similarly, there is a sizable drop in the PQ, PQTh{}^{\text{Th}} and AP metrics for the semantic task with PQSt{}^{\text{St}} improving by +0.2%+0.2\% showing that our framework can segment out amorphous masks for “stuff” regions but does not predict different masks for “thing” objects.

| training strategy | method | PQ | AP | mIoU |  |
|---|---|---|---|---|---|
| Panoptic Training | Mask2Former [12] | Mask2Former [12] | 40.7 | 25.2 | 45.6 |
| Mask2Former [12] |  |  |  |  |  |
| OneFormer (ours) | OneFormer (ours) | 41.4 (+0.7) | 27.0 (+1.8) | 46.1 (+0.5) |  |
| OneFormer (ours) |  |  |  |  |  |
| Instance Training | Mask2Former [12] | Mask2Former [12] | — | 26.4 | — |
| Mask2Former [12] |  |  |  |  |  |
| OneFormer (ours) | OneFormer (ours) | — | 26.7 (+0.3) | — |  |
| OneFormer (ours) |  |  |  |  |  |
| Semantic Training | Mask2Former [12] | Mask2Former [12] | — | — | 47.2 |
| Mask2Former [12] |  |  |  |  |  |
| OneFormer (ours) | OneFormer (ours) | — | — | 47.3 (+0.1) |  |
| OneFormer (ours) |  |  |  |  |  |
| Joint Training | Mask2Former† [12] | Mask2Former† [12] | 40.8 | 25.7 | 46.6 |
| Mask2Former† [12] |  |  |  |  |  |
| OneFormer (ours) | OneFormer (ours) | 41.9 (+1.1) | 27.3 (+1.6) | 47.3 (+0.7) |  |
| OneFormer (ours) |  |  |  |  |  |

| Mask2Former [12] |
|---|

| OneFormer (ours) |
|---|

| Mask2Former [12] |
|---|

| OneFormer (ours) |
|---|

| Mask2Former [12] |
|---|

| OneFormer (ours) |
|---|

| Mask2Former† [12] |
|---|

| OneFormer (ours) |
|---|

| Task Token Input | PQ | PQTh{}^{\text{Th}} | PQSt{}^{\text{St}} | AP | mIoU |
|---|---|---|---|---|---|
| the task is panoptic | 67.2 | 61.0 | 71.7 | 45.3 | 83.0 |
| the task is instance | 25.6 | 60.8 | 0.0 | 45.6 | 6.3 |
| the task is semantic | 56.9 | 36.2 | 71.9 | 27.2 | 83.0 |

| Method | Backbone | #Params | Crop Size | Extra Data | Extra Data | PQ | AP | mIoU (s.s.) | mIoU | (s.s.) | mIoU (m.s.) | mIoU | (m.s.) |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Extra Data |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mIoU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (s.s.) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mIoU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (m.s.) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Individual Training |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former [12] | Mask2Former [12] | BEiT-3 [52] | 1.9B | ×896896\!\times\!896 | ✓ | — | — | — | 62.0 | 62.0 | 62.8 |  |  |  |
| Mask2Former [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 62.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| UPerNet [55] | UPerNet [55] | FD-SwinV2-G [53] | >>3B | ×896896\!\times\!896 | ✓ | — | — | — | — | — | — | 61.4 |  |  |
| UPerNet [55] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask DINO [32] | Mask DINO [32] | Swin-L [38] | 223M | ×896896\!\times\!896 | ✓ | — | — | — | — | 59.5 | 59.5 | 60.8 |  |  |
| Mask DINO [32] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 59.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former [12] | Mask2Former [12] | ViT-Adapter-L [9] | 568M | ×896896\!\times\!896 | ✓ | — | — | — | — | 59.4 | 59.4 | 60.5 |  |  |
| Mask2Former [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 59.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| UPerNet [55] | UPerNet [55] | SwinV2-G [37] | >>3B | ×896896\!\times\!896 | ✓ | — | — | — | — | 59.3 | 59.3 | 59.9 |  |  |
| UPerNet [55] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 59.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| UPerNet [55] | UPerNet [55] | ViT-Adapter-L [9] | 571M | ×640640\!\times\!640 | ✗ | — | — | — | — | 58.0 | 58.0 | 58.4 |  |  |
| UPerNet [55] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MSFaPN-Mask2Former [27] | MSFaPN-Mask2Former [27] | SeMask Swin-L† [27] | — | ×640640\!\times\!640 | ✗ | — | — | — | — | 57.0 | 57.0 | 58.2 |  |  |
| MSFaPN-Mask2Former [27] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| FaPN-Mask2Former [25] | FaPN-Mask2Former [25] | Swin-L [38] | — | ×640640\!\times\!640 | ✗ | ✗ | — | — | — | — | 56.4 | 56.4 | 57.7 | 57.7 |
| FaPN-Mask2Former [25] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ✗ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 56.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeMask Mask2Former [27] | SeMask Mask2Former [27] | SeMask Swin-L† [27] | — | ×640640\!\times\!640 | ✗ | — | — | — | — | 56.4 | 56.4 | 57.5 |  |  |
| SeMask Mask2Former [27] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 56.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Semantic [25] | Mask2Former-Semantic [25] | Swin-L [38] | 216M | ×640640\!\times\!640 | ✗ | ✗ | — | — | — | — | 56.1 | 56.1 | 57.3 | 57.3 |
| Mask2Former-Semantic [25] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ✗ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 56.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Panoptic [12] | Mask2Former-Panoptic [12] | Swin-L [38] | 216M | ×640640\!\times\!640 | ✗ | 48.1 | 48.1 | 34.2 | 34.2 | 54.5 | 54.5 | — | — |  |
| Mask2Former-Panoptic [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 34.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 54.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kMaX-DeepLab [60] | kMaX-DeepLab [60] | ConvNeXt-L† [39] | 232M | ×641641\!\times\!641 | ✗ | 48.7 | 48.7 | — | — | 54.8 | 54.8 | — |  |  |
| kMaX-DeepLab [60] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 54.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Instance [12] | Mask2Former-Instance [12] | Swin-L [38] | 216M | ×640640\!\times\!640 | ✗ | — | — | 34.9 | 34.9 | — | — | — | — |  |
| Mask2Former-Instance [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 34.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kMaX-DeepLab [60] | kMaX-DeepLab [60] | ConvNeXt-L† [39] | 232M | ×12811281\!\times\!1281 | ✗ | 50.9 | 50.9 | — | — | 55.2 | 55.2 | — |  |  |
| kMaX-DeepLab [60] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 50.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 55.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Joint Training |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | Swin-L [38] | 219M | ×640640\!\times\!640 | ✗ | 49.8 | 49.8 | 35.9 | 35.9 | 57.0 | 57.0 | 57.7 | 57.7 |  |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 49.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 35.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | Swin-L [38] | 219M | ×896896\!\times\!896 | ✗ | 51.1 | 51.1 | 37.6 | 37.6 | 57.4 | 57.4 | 58.3 | 58.3 |  |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 51.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 37.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | Swin-L [38] | 219M | ×12801280\!\times\!1280 | ✗ | 51.4 | 51.4 | 37.8 | 37.8 | 57.0 | 57.0 | 57.7 | 57.7 |  |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 51.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 37.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | ConvNeXt-L [39] | 220M | ×640640\!\times\!640 | ✗ | 50.0 | 50.0 | 36.2 | 36.2 | 56.6 | 56.6 | 57.4 | 57.4 |  |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 50.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 36.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 56.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | ConvNeXt-XL [39] | 372M | ×640640\!\times\!640 | ✗ | 50.1 | 50.1 | 36.3 | 36.3 | 57.4 | 57.4 | 58.8 | 58.8 |  |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 50.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 36.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | DiNAT-L [21] | 223M | ×640640\!\times\!640 | ✗ | 50.5 | 50.5 | 36.0 | 36.0 | 58.3 | 58.3 | 58.4 | 58.4 |  |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 50.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 36.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | DiNAT-L [21] | 223M | ×896896\!\times\!896 | ✗ | 51.2 | 51.2 | 36.8 | 36.8 | 58.1 | 58.1 | 58.6 | 58.6 |  |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 51.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 36.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | DiNAT-L [21] | 223M | ×12801280\!\times\!1280 | ✗ | 51.5 | 51.5 | 37.1 | 37.1 | 58.2 | 58.2 | 58.7 | 58.7 |  |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 51.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 37.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

| Extra Data |
|---|

| mIoU |
|---|
| (s.s.) |

| mIoU |
|---|
| (m.s.) |

| Mask2Former [12] |
|---|

| — |
|---|

| 62.0 |
|---|

| UPerNet [55] |
|---|

| — |
|---|

| — |
|---|

| — |
|---|

| Mask DINO [32] |
|---|

| — |
|---|

| — |
|---|

| 59.5 |
|---|

| Mask2Former [12] |
|---|

| — |
|---|

| — |
|---|

| 59.4 |
|---|

| UPerNet [55] |
|---|

| — |
|---|

| — |
|---|

| 59.3 |
|---|

| UPerNet [55] |
|---|

| — |
|---|

| — |
|---|

| 58.0 |
|---|

| MSFaPN-Mask2Former [27] |
|---|

| — |
|---|

| — |
|---|

| 57.0 |
|---|

| FaPN-Mask2Former [25] |
|---|

| ✗ |
|---|

| — |
|---|

| — |
|---|

| 56.4 |
|---|

| 57.7 |
|---|

| SeMask Mask2Former [27] |
|---|

| — |
|---|

| — |
|---|

| 56.4 |
|---|

| Mask2Former-Semantic [25] |
|---|

| ✗ |
|---|

| — |
|---|

| — |
|---|

| 56.1 |
|---|

| 57.3 |
|---|

| Mask2Former-Panoptic [12] |
|---|

| 48.1 |
|---|

| 34.2 |
|---|

| 54.5 |
|---|

| — |
|---|

| kMaX-DeepLab [60] |
|---|

| 48.7 |
|---|

| — |
|---|

| 54.8 |
|---|

| Mask2Former-Instance [12] |
|---|

| — |
|---|

| 34.9 |
|---|

| — |
|---|

| — |
|---|

| kMaX-DeepLab [60] |
|---|

| 50.9 |
|---|

| — |
|---|

| 55.2 |
|---|

| OneFormer |
|---|

| 49.8 |
|---|

| 35.9 |
|---|

| 57.0 |
|---|

| 57.7 |
|---|

| OneFormer |
|---|

| 51.1 |
|---|

| 37.6 |
|---|

| 57.4 |
|---|

| 58.3 |
|---|

| OneFormer |
|---|

| 51.4 |
|---|

| 37.8 |
|---|

| 57.0 |
|---|

| 57.7 |
|---|

| OneFormer |
|---|

| 50.0 |
|---|

| 36.2 |
|---|

| 56.6 |
|---|

| 57.4 |
|---|

| OneFormer |
|---|

| 50.1 |
|---|

| 36.3 |
|---|

| 57.4 |
|---|

| 58.8 |
|---|

| OneFormer |
|---|

| 50.5 |
|---|

| 36.0 |
|---|

| 58.3 |
|---|

| 58.4 |
|---|

| OneFormer |
|---|

| 51.2 |
|---|

| 36.8 |
|---|

| 58.1 |
|---|

| 58.6 |
|---|

| OneFormer |
|---|

| 51.5 |
|---|

| 37.1 |
|---|

| 58.2 |
|---|

| 58.7 |
|---|

We further provide qualitative evidence in Fig. II. As demonstrated by the first example in Fig. II, the rider and bicycle regions are detected. However, the other “stuff” regions are misclassified in the semantic inference output when task=“instance”. Similarly, the people are detected in the second example, and the other “stuff” regions are misclassified. In further evidence, in both examples, the distinct “thing” objects are segmented into a single amorphous mask in the panoptic and instance inference outputs when task=“semantic”. Therefore, the differences in the qualitative results demonstrate OneFormer’s ability to output task-dependent class and mask predictions, which our task token input can guide.

## Appendix E Comparison to SOTA Methods at System-Level for Image Segmentation

In this section, we compare OneFormer to other SOTA systems for panoptic, instance, and semantic segmentation tasks on the ADE20K val [15], Cityscapes val [14], and COCO val2017 [34] datasets. As shown in Fig. I, our single OneFormer model outperforms Mask2Former for the three image segmentation tasks on all three datasets. Note that we are comparing the same OneFormer models referenced in our main text to other systems without applying additional system-level training techniques or using additional data and huge backbones.

### E.1 SOTA Systems on ADE20K val

As shown in Tab. VI, without using any extra training data, Swin-L OneFormer sets new state-of-the-art performance on instance segmentation with 37.8% AP, and DiNat-L OneFormer sets new state-of-the-art performance on panoptic segmentation with 51.5% PQ beating the previous state-of-the-art Swin-L Mask2Former’s [12] 34.9% AP and ConvNeXt-L KMaX-DeepLab’s [60] 50.9% PQ, respectively. Furthermore, DiNAT-L OneFormer and ConvNeXt-L OneFormer achieve the new-state-of-the-art single-scale and multi-scale mIoU scores of 58.3% and 58.8%, respectively, compared to other systems that do not use extra data during training.

### E.2 SOTA Systems on Cityscapes val

Without any extra data during training, our ConvNeXt-L OneFormer sets the new state-of-the-art performance on panoptic segmentation with 68.5% PQ with single-scale inference. Similarly, ConvNeXt-XL OneFormer achieves a new state-of-the-art 46.7% AP score with single-scale inference as shown in Tab. VII.

### E.3 SOTA Systems on COCO val

Without using any extra training data, DiNAT-L OneFormer matches the previous state-of-the-art KMaX-DeepLab [60] with 58.0% PQ score. Swin-L OneFormer achieves the best PQTh{}^{\text{Th}} score of 64.4%. For evaluating on the semantic segmentation task, we generate semantic GT annotations from the corresponding panoptic annotations. As shown in Tab. VIII, DiNAT-L OneFormer achieves an impressive 68.1% mIoU.

While analyzing the COCO dataset, we found serious discrepancies between the GT panoptic and instance annotations. Therefore, for fair comparison, during evaluation, we generate the instance annotations from the panoptic annotations for calculating the AP scores as only use panoptic annotations during training. We provide more information about the discrepancies in Appendix F. DiNAT-L OneFormer achieves 49.2% AP outperforming Mask2Former-Instance [12].

| Method | Backbone | #Params | Crop Size | Extra Data | Extra Data | MS (PQ & AP) | MS | (PQ & AP) | PQ | AP | mIoU (s.s.) | mIoU | (s.s.) | mIoU (m.s.) | mIoU | (m.s.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Extra Data |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| MS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (PQ & AP) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mIoU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (s.s.) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mIoU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (m.s.) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Individual Training |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| HRNetV2-OCR+PSA [35] | HRNetV2-OCR+PSA [35] | HRNetV2-W48 [45] | — | ×20481024\!\times\!2048 | ✓ | ✗ | ✗ | — | — | — | — | — | — | 86.9 |  |  |
| HRNetV2-OCR+PSA [35] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ✗ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| HRNetV2-OCR [35] | HRNetV2-OCR [35] | HRNetV2-W48 [45] | — | ×20481024\!\times\!2048 | ✓ | ✗ | ✗ | — | — | — | — | — | — | 86.3 |  |  |
| HRNetV2-OCR [35] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ✗ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former [12] | Mask2Former [12] | ViT-Adapter-L [9] | 571M | ×896896\!\times\!896 | ✓ | ✗ | ✗ | — | — | — | — | 84.9 | 84.9 | 85.8 |  |  |
| Mask2Former [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ✗ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former [12] | Mask2Former [12] | SeMask Swin-L [27] | 223M | ×1024512\!\times\!1024 | ✗ | ✗ | ✗ | — | — | — | — | 84.0 | 84.0 | 85.0 |  |  |
| Mask2Former [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ✗ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Semantic [12] | Mask2Former-Semantic [12] | Swin-L [38] | 215M | ×1024512\!\times\!1024 | ✗ | ✗ | ✗ | — | — | — | 83.3 | 83.3 | 84.3 | 84.3 |  |  |
| Mask2Former-Semantic [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ✗ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 83.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Panoptic-DeepLab [11] | Panoptic-DeepLab [11] | SWideRNet [8] | — | ×20491025\!\times\!2049 | ✓ | ✓ | 69.6 | 69.6 | 46.8 | 46.8 | — | — | 85.3 |  |  |  |
| Panoptic-DeepLab [11] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 69.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 46.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Axial-DeepLab-XL [51] | Axial-DeepLab-XL [51] | Axial ResNet-XL [51] | 173M | ×20491025\!\times\!2049 | ✓ | ✓ | 68.5 | 68.5 | 44.2 | 44.2 | — | — | 84.6 |  |  |  |
| Axial-DeepLab-XL [51] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 44.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| EfficientPS [42] | EfficientPS [42] | EfficientNet [46] | — | ×20491025\!\times\!2049 | ✓ | ✓ | 67.5 | 67.5 | 43.5 | 43.5 | — | — | 82.1 |  |  |  |
| EfficientPS [42] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 67.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 43.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Panoptic-DeepLab [11] | Panoptic-DeepLab [11] | SWideRNet [8] | — | ×20491025\!\times\!2049 | ✓ | ✗ | 68.5 | 68.5 | 42.8 | 42.8 | 84.6 | 84.6 | 85.3 |  |  |  |
| Panoptic-DeepLab [11] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 42.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Axial-DeepLab-XL [51] | Axial-DeepLab-XL [51] | Axial ResNet-XL [51] | 173M | ×20491025\!\times\!2049 | ✓ | ✗ | 67.8 | 67.8 | 41.9 | 41.9 | 84.2 | 84.2 | — |  |  |  |
| Axial-DeepLab-XL [51] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 67.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 41.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kMaX-DeepLab [60] | kMaX-DeepLab [60] | ConvNeXt-L [39] | 232M | ×20491025\!\times\!2049 | ✗ | ✗ | 68.4 | 68.4 | 44.0 | 44.0 | 83.5 | 83.5 | — |  |  |  |
| kMaX-DeepLab [60] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 44.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 83.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Panoptic-DeepLab [11] | Panoptic-DeepLab [11] | SWideRNet [8] | — | ×20491025\!\times\!2049 | ✗ | ✗ | 66.4 | 66.4 | 40.1 | 40.1 | 82.2 | 82.2 | 82.9 |  |  |  |
| Panoptic-DeepLab [11] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 66.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 40.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 82.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Axial-DeepLab-XL [51] | Axial-DeepLab-XL [51] | Axial ResNet-XL [51] | 173M | ×20491025\!\times\!2049 | ✗ | ✗ | 64.4 | 64.4 | 36.7 | 36.7 | 80.6 | 80.6 | 81.1 |  |  |  |
| Axial-DeepLab-XL [51] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 36.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 80.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Panoptic [12] | Mask2Former-Panoptic [12] | Swin-L [38] | 216M | ×1024512\!\times\!1024 | ✗ | ✗ | 66.6 | 66.6 | 43.6 | 43.6 | 82.9 | 82.9 | — | — |  |  |
| Mask2Former-Panoptic [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 66.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 43.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 82.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Instance [12] | Mask2Former-Instance [12] | Swin-L [38] | 216M | ×1024512\!\times\!1024 | ✗ | ✗ | — | — | 43.7 | 43.7 | — | — | — | — |  |  |
| Mask2Former-Instance [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 43.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Joint Training |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | Swin-L [38] | 219M | ×1024512\!\times\!1024 | ✗ | ✗ | 67.2 | 67.2 | 45.6 | 45.6 | 83.0 | 83.0 | 84.4 | 84.4 |  |  |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 67.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 45.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 83.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | ConvNeXt-L [39] | 220M | ×1024512\!\times\!1024 | ✗ | ✗ | 68.5 | 68.5 | 46.5 | 46.5 | 83.0 | 83.0 | 84.0 | 84.0 |  |  |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 46.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 83.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | ConvNeXt-XL [39] | 372M | ×1024512\!\times\!1024 | ✗ | ✗ | 68.4 | 68.4 | 46.7 | 46.7 | 83.6 | 83.6 | 84.6 | 84.6 |  |  |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 46.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 83.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | DiNAT-L [21] | 223M | ×1024512\!\times\!1024 | ✗ | ✗ | 67.6 | 67.6 | 45.6 | 45.6 | 83.1 | 83.1 | 84.0 | 84.0 |  |  |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 67.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 45.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 83.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 84.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

| Extra Data |
|---|

| MS |
|---|
| (PQ & AP) |

| mIoU |
|---|
| (s.s.) |

| mIoU |
|---|
| (m.s.) |

| HRNetV2-OCR+PSA [35] |
|---|

| ✗ |
|---|

| — |
|---|

| — |
|---|

| — |
|---|

| HRNetV2-OCR [35] |
|---|

| ✗ |
|---|

| — |
|---|

| — |
|---|

| — |
|---|

| Mask2Former [12] |
|---|

| ✗ |
|---|

| — |
|---|

| — |
|---|

| 84.9 |
|---|

| Mask2Former [12] |
|---|

| ✗ |
|---|

| — |
|---|

| — |
|---|

| 84.0 |
|---|

| Mask2Former-Semantic [12] |
|---|

| ✗ |
|---|

| — |
|---|

| 83.3 |
|---|

| 84.3 |
|---|

| Panoptic-DeepLab [11] |
|---|

| 69.6 |
|---|

| 46.8 |
|---|

| — |
|---|

| Axial-DeepLab-XL [51] |
|---|

| 68.5 |
|---|

| 44.2 |
|---|

| — |
|---|

| EfficientPS [42] |
|---|

| 67.5 |
|---|

| 43.5 |
|---|

| — |
|---|

| Panoptic-DeepLab [11] |
|---|

| 68.5 |
|---|

| 42.8 |
|---|

| 84.6 |
|---|

| Axial-DeepLab-XL [51] |
|---|

| 67.8 |
|---|

| 41.9 |
|---|

| 84.2 |
|---|

| kMaX-DeepLab [60] |
|---|

| 68.4 |
|---|

| 44.0 |
|---|

| 83.5 |
|---|

| Panoptic-DeepLab [11] |
|---|

| 66.4 |
|---|

| 40.1 |
|---|

| 82.2 |
|---|

| Axial-DeepLab-XL [51] |
|---|

| 64.4 |
|---|

| 36.7 |
|---|

| 80.6 |
|---|

| Mask2Former-Panoptic [12] |
|---|

| 66.6 |
|---|

| 43.6 |
|---|

| 82.9 |
|---|

| — |
|---|

| Mask2Former-Instance [12] |
|---|

| — |
|---|

| 43.7 |
|---|

| — |
|---|

| — |
|---|

| OneFormer |
|---|

| 67.2 |
|---|

| 45.6 |
|---|

| 83.0 |
|---|

| 84.4 |
|---|

| OneFormer |
|---|

| 68.5 |
|---|

| 46.5 |
|---|

| 83.0 |
|---|

| 84.0 |
|---|

| OneFormer |
|---|

| 68.4 |
|---|

| 46.7 |
|---|

| 83.6 |
|---|

| 84.6 |
|---|

| OneFormer |
|---|

| 67.6 |
|---|

| 45.6 |
|---|

| 83.1 |
|---|

| 84.0 |
|---|

| Method | Backbone | #Params | Extra Data | PQ | PQTh{}^{\text{Th}} | PQSt{}^{\text{St}} | AP | APinstance{}^{\text{instance}} | mIoU | mIoU |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mIoU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Individual Training |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask DINO [32] | Mask DINO [32] | Swin-L [38] | 223M | ✓ | 59.4 | 59.4 | — | — | — | — | 54.5 | 54.5 | — |  |  |
| Mask DINO [32] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 59.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 54.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kMaX-DeepLab [60] | kMaX-DeepLab [60] | ConvNeXt-L [39] | 232M | ✓ | 58.1 | 64.3 | 64.3 | 48.8 | 48.8 | — | — | — |  |  |  |
| kMaX-DeepLab [60] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| kMaX-DeepLab [60] | kMaX-DeepLab [60] | ConvNeXt-L [39] | 232M | ✗ | 58.0 | 64.2 | 64.2 | 48.6 | 48.6 | — | — | — |  |  |  |
| kMaX-DeepLab [60] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Panoptic [12] | Mask2Former-Panoptic [12] | Swin-L [38] | 216M | ✗ | 57.8 | 57.8 | 64.2 | 64.2 | 48.1 | 48.1 | 48.7 | 48.7 | 48.6 | 67.4 | 67.4 |
| Mask2Former-Panoptic [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 67.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Panoptic SegFormer [33] | Panoptic SegFormer [33] | Swin-L [38] | 221M | ✗ | 55.8 | 61.7 | 61.7 | 46.9 | 46.9 | — | — | — |  |  |  |
| Panoptic SegFormer [33] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 61.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 46.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mask2Former-Instance [12] | Mask2Former-Instance [12] | Swin-L [38] | 216M | ✗ | — | — | — | — | — | — | 49.1 | 49.1 | 50.1 | — | — |
| Mask2Former-Instance [12] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 49.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| — |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Joint Training |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | Swin-L [38] | 219M | ✗ | 57.9 | 57.9 | 64.4 | 64.4 | 48.0 | 48.0 | 49.0 | 49.0 | 48.9 | 67.4 | 67.4 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 57.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 49.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 67.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| OneFormer | OneFormer | DiNAT-L [21] | 223M | ✗ | 58.0 | 58.0 | 64.3 | 64.3 | 48.4 | 48.4 | 49.2 | 49.2 | 49.2 | 68.1 | 68.1 |
| OneFormer |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 58.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 64.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 48.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 49.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 68.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

| mIoU |
|---|

| Mask DINO [32] |
|---|

| 59.4 |
|---|

| — |
|---|

| 54.5 |
|---|

| kMaX-DeepLab [60] |
|---|

| 64.3 |
|---|

| 48.8 |
|---|

| kMaX-DeepLab [60] |
|---|

| 64.2 |
|---|

| 48.6 |
|---|

| Mask2Former-Panoptic [12] |
|---|

| 57.8 |
|---|

| 64.2 |
|---|

| 48.1 |
|---|

| 48.7 |
|---|

| 67.4 |
|---|

| Panoptic SegFormer [33] |
|---|

| 61.7 |
|---|

| 46.9 |
|---|

| Mask2Former-Instance [12] |
|---|

| — |
|---|

| — |
|---|

| — |
|---|

| 49.1 |
|---|

| — |
|---|

| OneFormer |
|---|

| 57.9 |
|---|

| 64.4 |
|---|

| 48.0 |
|---|

| 49.0 |
|---|

| 67.4 |
|---|

| OneFormer |
|---|

| 58.0 |
|---|

| 64.3 |
|---|

| 48.4 |
|---|

| 49.2 |
|---|

| 68.1 |
|---|

## Appendix F Analysis on Discrepancy between Instance and Panoptic Annotations in COCO

During our joint training, we derive the semantic and instance ground-truth labels from the corresponding panoptic annotations. Unlike, Cityscapes [14] and ADE20K [15] datasets, which combine the semantic and instance annotations to generate the corresponding panoptic annotations while preparing the data, COCO [34] has separate sets of panoptic and instance annotations. As expected, there are no discrepancies between the panoptic and instance annotations in the Cityscapes [14] and ADE20K [15] datasets. However, because COCO [34] has separately developed panoptic and instance annotations, we discover significant discrepancies in the COCO train2017 and val2017 [34] datasets as shown in Fig. III and Fig. IV, respectively.

In Fig. III, the instance annotations merge the “tie” object into the “person” object. In another example, instance annotations merge the “dog” and “boat” into a single instance, while the panoptic annotations segment the two instances correctly.

In Fig. IV, the instance annotations skip multiple “person” and “motorcycle” objects in different images, while the panoptic annotations include them all. In another example, instance annotations leave out a group of “person” object instances in the background, and panoptic annotations merge those instances into a single object mask.

These discrepancies are a significant barrier to developing and evaluating a unified image segmentation model. As demonstrated in Fig. III and Fig. IV, our predictions match the panoptic annotations much more than the instance annotations which is expected from our training strategy involving only panoptic annotations. Therefore, while comparing our Swin-L† OneFormer to other SOTA methods in Tab. 3 (main text), we evaluate the AP score on instance GTs derived from the panoptic annotations.
