# Paper (Chen et al. 2023)

> Source: `https://arxiv.org/abs/2312.14238`

---

InternVL: Scaling up Vision Foundation Models and Aligning
for Generic Visual-Linguistic Tasks
Zhe Chen2,1†, Jiannan Wu3,1†, Wenhai Wang1,4, Weijie Su6,1†, Guo Chen2,1†, Sen Xing5, Muyan Zhong5,
Qinglong Zhang1, Xizhou Zhu5,7,1, Lewei Lu7,1, Bin Li6, Ping Luo3, Tong Lu2, Yu Qiao1, Jifeng Dai5,1B
1OpenGVLab, Shanghai AI Laboratory
2Nanjing University
3The University of Hong Kong
4The Chinese University of Hong Kong
5Tsinghua University
6University of Science and Technology of China
7SenseTime Research
https://github.com/OpenGVLab/InternVL
contras(ve
image
text
vision
encoder
text
encoder
contras(ve
image
text
generative
scaling up
vision
encoder
to 6B #params
large
language
model
large
language
model
prompt
shared
weights
image
classes
vision
encoder
(a) Supervised pre-training
(b) Contrastive pre-training
(c) InternVL: Scaling up vision encoder and aligning with LLM (ours)
Figure 1. Comparisons of different vision and vision-language foundation models. (a) indicates the traditional vision foundation model,
e.g. ResNet [57] pre-trained on classification tasks. (b) represents the vision-language foundation models, e.g. CLIP [117] pre-trained on
image-text pairs. (c) is our InternVL, which presents a workable way to align the large-scale vision foundation model (i.e., InternViT-6B)
with the large language model and is versatile for both contrastive and generative tasks.
Abstract
The exponential growth of large language models
(LLMs) has opened up numerous possibilities for multi-
modal AGI systems. However, the progress in vision and
vision-language foundation models, which are also critical
elements of multi-modal AGI, has not kept pace with LLMs.
In this work, we design a large-scale vision-language foun-
dation model (InternVL), which scales up the vision foun-
dation model to 6 billion parameters and progressively
aligns it with the LLM, using web-scale image-text data
from various sources. This model can be broadly applied
to and achieve state-of-the-art performance on 32 generic
visual-linguistic benchmarks including visual perception
tasks such as image-level or pixel-level recognition, vision-
language tasks such as zero-shot image/video classification,
zero-shot image/video-text retrieval, and link with LLMs to
create multi-modal dialogue systems. It has powerful visual
capabilities and can be a good alternative to the ViT-22B.
We hope that our research could contribute to the develop-
ment of multi-modal large models.
† This work is done when they are interns at Shanghai AI Laboratory;
B corresponding author (daijifeng@tsinghua.edu.cn)
1. Introduction
Large language models (LLMs) largely promote the devel-
opment of artificial general intelligence (AGI) systems with
their impressive capabilities in open-world language tasks,
and their model scale and performance are still increasing
at a fast pace.
Vision large language models (VLLMs)
[3, 5, 21, 23, 34, 92, 115, 147, 187], which leverage
LLMs, have also achieved significant breakthroughs, en-
abling sophisticated vision-language dialogues and interac-
tions. However, the progress of vision and vision-language
foundation models, which are also crucial for VLLMs, has
lagged behind the rapid growth of LLMs.
To bridge vision models with LLMs, existing VLLMs
[5, 81, 131, 177, 187] commonly employ lightweight “glue”
layers, such as QFormer [81] or linear projection [92], to
align features of vision and language models. Such align-
ment contains several limitations: (1) Disparity in param-
eter scales. The large LLMs [48] now boosts up to 1000
billion parameters, while the widely-used vision encoders
of VLLMs are still around one billion. This gap may lead
to the under-use of LLM’s capacity. (2) Inconsistent rep-
resentation. Vision models, trained on pure-vision data or
1
arXiv:2312.14238v3  [cs.CV]  15 Jan 2024

87.8
89.6
78.4
75.9
87.8
66.4
82.0
82.1
94.5
75.7
71.6
79.6
59.6
54.6
37.0
56.8
69.8
69.3
63.4
93.9
80.4
68.8
51.4
88.9
74.5
70.0
69.2
94.6
117.7
1531.3
85.9
322.5
88.2
90.4
79.9
77.5
89.8
69.1
83.2
83.8
95.5
77.3
73.9
80.6
64.5
61.5
44.9
65.7
76.1
75.5
67.5
95.7
85.0
74.9
58.6
92.9
77.7
71.4
73.8
96.6
128.2
1586.4
87.6
327.6
5
5
5
5
5
5
5
IN-1K
IN-ReaL
IN-V2
IN-A
IN-R
IN-Sketch
IN-1K
IN-A
IN-R
IN-V2
IN-Sketch
ObjectNet
IN-1K (ZH)
IN-1K (JP)
IN-1K (AR)
IN-1K (IT)
Kinetics-400
Kinetics-600
Kinetics-700
Flickr30K I2T
Flickr30K T2I
COCO I2T
COCO T2I
Flickr30K-CN I2T
Flickr30K-CN T2I
COCO-CN I2T
COCO-CN T2I
XTD R@10 I2T
ZS COCO Caption
MME
POPE
Tiny LVLM
Pervious SOTA
Our Performance
Dialogue
Linear-Probe Image Classification
Zero-Shot Image & Video Classification
Zero-Shot Image-Text Retrieval
Figure 2. Comparison results on various generic visual-linguistic tasks, including image classification, video classification, image-text
retrieval, image captioning, and multi-modal dialogue. The proposed InternVL achieves the best performance on all these tasks. Note that
only the models trained on public data are included. “IN” is an abbreviation for ImageNet [38].
aligned with the BERT series [39, 70, 93], often exhibit
representation inconsistencies with LLMs. (3) Inefficient
connection. The “glue” layers are usually lightweight and
randomly initialized, which may not capture the rich cross-
modal interactions and dependencies that are crucial for
multi-modal understanding and generation.
These limitations reveal a large gap in both parameter
scale and feature representation ability between the vision
encoder and the LLM. To bridge this gap, our inspiration
lies in elevating the vision encoder to align with the param-
eter scale of the LLM and subsequently harmonizing their
representations. However, the training of such large-scale
models necessitates a vast amount of image-text data ob-
tained from the Internet. The significant heterogeneity and
quality variations within this data pose considerable chal-
lenges to the training process. To enhance the efficacy of
the training, generative supervision is considered as a com-
plementary approach to contrastive learning, as depicted in
Figure 1. This strategy aims to provide additional guidance
to the model during training. Yet, the suitability of low-
quality data for generative training remains a concern. Be-
sides, how to effectively represent the users’ commands and
align the representations between the vision encoder and
LLM is another open question.
To address these issues, we formulate the InternVL, a
large-scale vision-language foundation model, which aligns
the representation of the scaled-up vision encoder with the
LLM and achieves state-of-the-art performance on various
visual and vision-language tasks. As shown in Figure 1 (c),
InternVL has three key designs: (1) Parameter-balanced vi-
sion and language components: It includes a vision encoder
scaled up to 6 billion parameters and an LLM middleware
with 8 billion parameters, where the middleware functions
as a substantial “glue” layer to reorganize visual features
based on user commands. Unlike prior vision-only (Fig-
ure 1 (a)) or dual-tower (Figure 1 (b)) structures, our vi-
sion encoder and middleware offer flexible combinations
for both contrastive and generative tasks. (2) Consistent
representations: To maintain the consistency of represen-
tations between the vision encoder and LLM, we employ a
pre-trained multilingual LLaMA [32], to initialize the mid-
dleware and align the vision encoder with it. (3) Progressive
image-text alignment: We leverage image-text data from di-
verse sources, ensuring training stability through a progres-
sive alignment strategy. This strategy initiates contrastive
learning on large-scale noisy image-text data and subse-
quently transitions to generative learning on fine-grained
data. This approach ensures a consistent enhancement of
model performance and task scope.
These designs endow our model with several advantages:
(1) Versatile. It functions as a standalone vision encoder for
perception tasks, or collaborates with the language middle-
ware for vision-language tasks and multi-modal dialogue
systems.
The language middleware bridges the gap be-
tween the vision encoder and the LLM decoder. (2) Strong.
By leveraging the training strategy, large-scale parameters,
and web-scale data, our model has a powerful represen-
tation that helps to achieve state-of-the-art results on var-
ious vision and vision-language tasks, as shown in Fig-
ure 2. (3) LLM-friendly. Due to the aligned feature space
with LLMs, our model can smoothly integrate with exist-
ing LLMs, such as LLaMA series [138, 139], Vicuna [184],
and InternLM [135]. These features distinguish our model
from the previous approaches and establish a leading vision-
language foundation model for various applications.
In summary, our contribution has three folds:
(1) We present a large-scale vision-language foundation
model—InternVL, which aligns the large-scale vision en-
coder with LLMs for the first time. The model demonstrates
strong performance on a wide range of generic visual-
linguistic tasks, including visual perception tasks, vision-
language tasks, and multi-modal dialogue.
(2) We introduce a progressive image-text alignment
strategy for the efficient training of large-scale vision-
language foundation models. This strategy maximizes the
utilization of web-scale noisy image-text data for con-
trastive learning and fine-grained, high-quality data for gen-
erative learning.
2

(3) We extensively compare the proposed model with
the current state-of-the-art vision foundation models and
VLLMs.
The results indicate that InternVL achieves
leading performance on a broad range of generic visual-
linguistic tasks, including image classification (ImageNet),
semantic segmentation (ADE20K), video classification (Ki-
netics), image-text retrieval (Flickr30K & COCO), video-
text retrieval (MSR-VTT), and image captioning (COCO &
Flickr30K & NoCaps). Meanwhile, it is also effective for
multi-modal dialogue (MME & POPE & Tiny LVLM).
2. Related Work
2.1. Vision Foundation Models
The past decade has witnessed significant development in
foundation models within the field of computer vision.
Starting with the pioneering AlexNet [73], a variety of con-
volutional neural networks (CNNs) have emerged, continu-
ously refreshing the ImageNet benchmark [33, 40, 57, 62,
65, 95, 148, 160]. In particular, the introduction of residual
connections [57] effectively addressed the problem of van-
ishing gradients. This breakthrough led to an era of “big &
deep” neural networks, signifying that, with adequate train-
ing and data, larger and deeper models can achieve better
performance. In other words, scaling up matters.
In recent years, ViT [42] has opened up new possibilities
for network architectures in the computer vision field. ViT
and its variants [15, 25, 37, 46, 94, 117, 144, 145, 178, 179]
have significantly increased their capacity and excelled in
various important visual tasks. In the LLM era, these vi-
sion foundation models often connect with LLMs through
some lightweight “glue” layers [80, 92, 187]. However, a
gap exists as these models primarily derive from visual-only
datasets like ImageNet [38] or JFT [173], or are aligned
with the BERT series [39, 70, 93] using image-text pairs,
lacking direct alignment with LLMs.
Additionally, the
prevalent vision models employed to connect with LLMs
are still limited to around 1 billion parameters [46, 67],
which also constrains the performance of VLLMs.
2.2. Large Language Models
Large language models (LLMs) have revolutionized the
field of artificial intelligence, enabling natural language pro-
cessing tasks that were previously thought exclusive to hu-
mans [110, 138, 153].
The emergence of GPT-3 [153]
brought a significant leap in capabilities, particularly in few-
shot and zero-shot learning, highlighting the immense po-
tential of LLMs. This promise was further realized with the
advancements of ChatGPT and GPT-4 [110]. The progress
in the field has been further accelerated by the emergence of
open-source LLMs, including the LLaMA series [138, 139],
Vicuna [184], InternLM [135], MOSS [132], ChatGLM
[44], Qwen [4], Baichuan [6], and Falcon [114], among oth-
ers [32, 134, 154]. However, in real scenarios, interactions
are not limited to natural language. The vision modality
can bring additional information, which means more pos-
sibilities. Therefore, exploring how to utilize the excellent
capabilities of LLMs for multi-modal interactions is poised
to become the next research trend.
2.3. Vision Large Language Models
Recent advancements have seen the creation of vision large
language models (VLLMs) [3, 23, 75, 79, 82, 88, 131, 156,
165, 168, 175, 177, 180, 181, 188], which aim to enhance
language models with the capability to process and inter-
pret visual information. Flamingo [3] uses the visual and
language inputs as prompts and shows remarkable few-shot
performance for visual question answering. Subsequently,
GPT-4 [110], LLaVA series [91, 92, 100] and MiniGPT-4
[187] have brought in visual instruction tuning, to improve
the instruction-following ability of VLLMs. Concurrently,
models such as VisionLLM [147], KOSMOS-2 [115], and
Qwen-VL et al. [5, 21, 149] have improved VLLMs with
visual grounding capabilities, facilitating tasks such as re-
gion description and localization. Many API-based meth-
ods [96, 97, 125, 133, 155, 163, 166] have also attempted to
integrate vision APIs with LLMs for solving vision-centric
tasks. Additionally, PaLM-E [43] and EmbodiedGPT [108]
represent advanced efforts in adapting VLLMs for em-
bodied applications, significantly expanding their poten-
tial applications. These works showcase that VLLMs have
achieved significant breakthroughs. However, the progress
of vision and vision-language foundation models, equally
essential for VLLMs, has not kept pace.
3. Proposed Method
3.1. Overall Architecture
As depicted in Figure 3, unlike traditional vision-only back-
bones [57, 94, 148] and dual-encoder models [67, 117,
130], the proposed InternVL is designed with a vision en-
coder InternViT-6B and a language middleware QLLaMA.
Specifically, InternViT-6B is a vision transformer with 6 bil-
lion parameters, customized to achieve a favorable trade-
off between performance and efficiency.
QLLaMA is a
language middleware with 8 billion parameters, initialized
with a multilingual-enhanced LLaMA [32]. It could pro-
vide robust multilingual representation for image-text con-
trastive learning, or serve as a bridge to connect the vision
encoder and the off-the-shelf LLM decoder.
To align the two large-scale components with substan-
tial gaps in modalities and structures, we introduce a pro-
gressive alignment training strategy.
The training strat-
egy is conducted progressively, beginning with contrastive
learning on large-scale noisy data, and gradually moving
towards generative learning on exquisite and high-quality
3

contrastive loss
supported tasks:
1. zero-shot image classification (new)
2. zero-shot image-text retrieval (new)
stage 1: contrastive pre-training
supported tasks: 1. zero-shot image classification
2. zero-shot image-text retrieval
3. zero-shot image captioning (new)
stage 2: generative pre-training
1. matching loss
2. contrastive loss
3. generative loss
supported tasks: 4. multi-modal dialogue (new)
5. visual question answering (new)
stage 3: supervised fine-tuning
generative loss
MLP
MLP
frozen weights
trainable weights
shared weights
cross
attention
cross
attention
InternViT-6B
LLaMA-7B
InternViT-6B
QLLaMA
InternViT-6B
QLLaMA
Vicuna-13B
/
Figure 3. The training strategy of the proposed InternVL model.
It consists of three progressive stages, including vision-language
contrastive training, vision-language generative training, and supervised fine-tuning. These stages effectively leverage public data from
diverse sources, ranging from noisy image-text pairs on the web to high-quality caption, VQA, and multi-modal dialogue datasets.
name
width
depth
MLP
#heads
#param (M)
ViT-G [173]
1664
48
8192
16
1843
ViT-e [23]
1792
56
15360
16
3926
EVA-02-ViT-E [130]
1792
64
15360
16
4400
ViT-6.5B [128]
4096
32
16384
32
6440
ViT-22B [37]
6144
48
24576
48
21743
InternViT-6B (ours)
3200
48
12800
25
5903
Table 1. Architecture details of the InternViT-6B model.
data. In this way, we ensure the effective organization and
full utilization of web-scale image-text data from a variety
of sources. Then, equipped with the aligned vision encoder
and language middleware, our model functions like a Swiss
Army knife. It boasts a flexible composition that can be
adapted for a wide array of generic visual-linguistic tasks.
These tasks range from visual perception and image/video-
text retrieval to image captioning, visual question answer-
ing, and multi-modal dialogue, among others.
3.2. Model Design
Large-Scale Vision Encoder: InternViT-6B. We imple-
ment the vision encoder of InternVL with vanilla vision
transformer (ViT) [42]. To match the scale of LLMs, we
scale up the vision encoder to 6 billion parameters, result-
ing in the InternViT-6B model. To obtain a good trade-off
between accuracy, speed, and stability, we conduct a hy-
perparameter search for InternViT-6B. We vary the model
depth within {32, 48, 64, 80}, the head dimension within
{64, 128}, and the MLP ratio within {4, 8}. The model
width and the head number are calculated based on the
given model scale and other hyperparameters.
We employ contrastive learning on a 100M subset of the
LAION-en dataset [120] to measure the accuracy, speed,
and stability of InternViT-6B variants with different config-
urations. We report the following findings: (1) Speed. For
different model settings, when computation is not saturated,
the models with smaller depths exhibit faster speed per im-
age. However, as the GPU computation is fully utilized, the
speed difference becomes negligible; (2) Accuracy. With
the same number of parameters, the depth, head dimension,
and MLP ratio have little impact on the performance. Based
on these findings, we identified the most stable configura-
tion for our final model, as shown in Table 1.
Language Middleware: QLLaMA. The language mid-
dleware QLLaMA is proposed to align visual and linguis-
tic features.
As shown in Figure 3, QLLaMA is devel-
oped based on the pre-trained multilingual LLaMA [32],
and newly added 96 learnable queries and cross-attention
layers (1 billion parameters) that are randomly initialized.
This manner allows QLLaMA to smoothly integrate visual
elements into the language model, thereby enhancing the
coherence and effectiveness of the combined features.
Compared to recently popular approaches [81, 92] that
use lightweight “glue” layers, such as QFormer [81] and
linear layers [92] to connect vision encoder and LLMs, our
method has three advantages: (1) By initializing with the
pre-trained weights of [32], QLLaMA can transform im-
age tokens generated by InternViT-6B into the representa-
tion that is aligned with the LLMs; (2) QLLaMA has 8 bil-
lion parameters for vision-language alignment, which are
42 times larger than the QFormer. Therefore, even with a
frozen LLM decoder, InternVL can achieve promising per-
formance on multi-modal dialogue tasks. (3) It can also be
4

a cute panda [EOS]
image
text
(a) InternVL-C
attention pooling
[EOS]
similarity
InternViT-6B
QLLaMA
a cute panda [EOS]
image
text
(b) InternVL-G
[EOS]
query
attention pooling
similarity
InternViT-6B
QLLaMA
what is this?
image
text
(d) InternVL-Chat (w/ QLLaMA)  
query
InternViT-6B
Vicuna-13B
<image><query> what is this?
image + query + text
a cute panda
QLLaMA
<image> what is this?
image
image + text
(c) InternVL-Chat (w/o QLLaMA)  
Vicuna-13B
InternViT-6B
a cute panda
Figure 4. Different ways to use InternVL. By flexibly combining the vision encoder and the language middleware, InternVL can support
various vision-language tasks, including contrastive tasks, generative tasks, and multi-modal dialogue.
characteristics
stage 1
stage 2
dataset
language
original
cleaned
remain
cleaned
remain
LAION-en [120]
English
2.3B
1.94B
84.3%
91M
4.0%
LAION-COCO [121]
663M
550M
83.0%
550M
83.0%
COYO [14]
747M
535M
71.6%
200M
26.8%
CC12M [20]
12.4M
11.1M
89.5%
11.1M
89.5%
CC3M [124]
3.0M
2.6M
86.7%
2.6M
86.7%
SBU [112]
1.0M
1.0M
100%
1.0M
100%
Wukong [55]
Chinese
100M
69.4M
69.4%
69.4M
69.4%
LAION-multi [120]
Multi
2.2B
1.87B
85.0%
100M
4.5%
Total
Multi
6.03B
4.98B
82.6%
1.03B
17.0%
Table 2. Details of the training data for InternVL in stage 1
and stage 2. Among them, LAION-en [120], LAION-multi [120],
COYO [14], and Wukong [55] are web-scale image-text pairs data.
LAION-COCO [121] is a synthetic dataset with high-quality cap-
tions from LAION-en. CC12M [20], CC3M [124], SBU [112] are
academic caption datasets. “Multi” means multilingual.
applied to contrastive learning, providing a powerful text
representation for image-text alignment tasks, such as zero-
shot image classification and image-text retrieval.
“Swiss Army Knife” Model: InternVL. By flexibly com-
bining the vision encoder and the language middleware, In-
ternVL can support various vision or vision-language tasks.
(1) For visual perception tasks, the vision encoder of In-
ternVL, i.e. InternViT-6B, can be used as the backbone for
vision tasks. Given an input image I ∈RH×W ×3, our
model can generate a feature map F ∈RH/14×W/14×D for
dense prediction tasks, or work with global average pooling
and linear projection to make image classification.
(2) For contrastive tasks, as shown in Figure 4 (a) (b), we in-
troduce two inference modes: InternVL-C and InternVL-
G, using the vision encoder or the combination of InternViT
and QLLaMA to encode visual features. Specifically, we
apply attention pooling to the visual features of InternViT
or the query features of QLLaMA, to calculate the global
visual feature If. Besides, we encode text as Tf by ex-
tracting the feature from the [EOS] token of QLLaMA. By
computing similarity scores between If and Tf, we support
various contrastive tasks such as image-text retrieval.
(3) For generative tasks, unlike QFormer [80], QLLaMA
inherently has promising image captioning abilities thanks
to its scaled-up parameters. The queries of QLLaMA re-
organize the visual representations from InternViT-6B and
play as the prefix texts for QLLaMA. The subsequent text
tokens are generated one by one sequentially.
task
#samples
dataset
Captioning
588K
COCO Caption [22], TextCaps [126]
VQAv2 [54], OKVQA [104], A-OKVQA [122],
VQA
1.1M
IconQA [99], AI2D [71], GQA [64]
OCR-VQA [107], ChartQA [105], DocVQA [29],
ST-VQA [12], EST-VQA [150], InfoVQA [106],
OCR
294K
LLaVAR [182]
Grounding
323K
RefCOCO/+/g [103, 170], Toloka [140]
Grounded Cap.
284K
RefCOCO/+/g [103, 170]
LLaVA-150K [92], SVIT [183], VisDial [36],
Conversation
1.4M
LRV-Instruction [90], LLaVA-Mix-665K [91]
Table 3. Details of the training data for InternVL in stage 3.
We collect a wide range of high-quality instruction data, totaling
approximately 4 million samples. For a fair comparison, we only
use the training split of these datasets.
(4) For multi-modal dialogue, we introduce InternVL-
Chat, leveraging InternVL as the visual component to con-
nect with LLMs. For this purpose, we have two distinct
configurations. One option is to employ the InternViT-6B
independently, as shown in Figure 4 (c). The alternative
is to employ the complete InternVL model concurrently, as
illustrated in Figure 4 (d).
3.3. Alignment Strategy
As shown in Figure 3, the training of InternVL consists
of three progressive stages, including vision-language con-
trastive training, vision-language generative training, and
supervised fine-tuning. These stages effectively leverage
public data from diverse sources, ranging from noisy image-
text pairs on the web to high-quality caption, VQA, and
multi-modal dialogue datasets.
Vision-Language Contrastive Training. In the first stage,
we conduct contrastive learning to align InternViT-6B with
a multilingual LLaMA-7B [32] on web-scale, noisy image-
text pairs. The data are all publicly available and comprise
multilingual content, including LAION-en [120], LAION-
multi [120], LAION-COCO [121], COYO [14], Wukong
[55], etc. We use the combination of these datasets and fil-
ter out some extremely low-quality data to train our model.
As summarized in Table 2, the original dataset contains
6.03 billion image-text pairs, and 4.98 billion remains af-
ter cleaning. More details about data preparation will be
provided in the supplementary materials.
During training, we adopt the LLaMA-7B to encode the
text as Tf, and use InternViT-6B to extract the visual fea-
5

method
#param IN-1K IN-ReaL IN-V2 IN-A IN-R IN-Ske avg.
OpenCLIP-H [67]
0.6B
84.4
88.4
75.5
−
−
−
−
OpenCLIP-G [67]
1.8B
86.2
89.4
77.2
63.8 87.8
66.4
78.5
DINOv2-g [111]
1.1B
86.5
89.6
78.4
75.9 78.8
62.5
78.6
EVA-01-CLIP-g [46]
1.1B
86.5
89.3
77.4
70.5 87.7
63.1
79.1
MAWS-ViT-6.5B [128]
6.5B
87.8
–
–
–
–
–
–
ViT-22B∗[37]
21.7B
89.5
90.9
83.2
83.8 87.4
−
−
InternViT-6B (ours)
5.9B
88.2
90.4
79.9
77.5 89.8
69.1
82.5
Table 4. Linear evaluation on image classification. We report the
top-1 accuracy on ImageNet-1K [38] and its variants [10, 60, 61,
119, 141]. ∗ViT-22B [37] uses the private JFT-3B dataset [173].
method
#param
crop size
1/16
1/8
1/4
1/2
1
ViT-L [137]
0.3B
5042
36.1
41.3
45.6
48.4
51.9
ViT-G [173]
1.8B
5042
42.4
47.0
50.2
52.4
55.6
ViT-22B [37]
21.7B
5042
44.7
47.2
50.6
52.5
54.9
InternViT-6B (ours)
5.9B
5042
46.5
50.0
53.3
55.8
57.2
(a) Few-shot semantic segmentation with limited training data. Following
ViT-22B [37], we fine-tune the InternViT-6B with a linear classifier.
method
decoder
#param (train/total)
crop size
mIoU
OpenCLIP-Gfrozen [67]
Linear
0.3M / 1.8B
5122
39.3
ViT-22Bfrozen [37]
Linear
0.9M / 21.7B
5042
34.6
InternViT-6Bfrozen (ours)
Linear
0.5M / 5.9B
5042
47.2
ViT-22Bfrozen [37]
UperNet
0.8B / 22.5B
5042
52.7
InternViT-6Bfrozen (ours)
UperNet
0.4B / 6.3B
5042
54.9
ViT-22B [37]
UperNet
22.5B / 22.5B
5042
55.3
InternViT-6B (ours)
UperNet
6.3B / 6.3B
5042
58.9
(b) Semantic segmentation performance in three different settings, from
top to bottom: linear probing, head tuning, and full-parameter tuning.
Table 5. Semantic segmentation on ADE20K. Results show that
InternViT-6B has better pixel-level perceptual capacity.
ture If. Following the objective function of CLIP [117],
we minimize a symmetric cross-entropy loss on the simi-
larity scores of image-text pairs in a batch. This stage al-
lows InternVL to excel on contrastive tasks like zero-shot
image classification and image-text retrieval, and the vision
encoder of this stage can also perform well on visual per-
ception tasks like semantic segmentation.
Vision-Language Generative Training.
In the second
stage of training, we connect InternViT-6B with QLLaMA
and adopt a generative training strategy. Specifically, QL-
LaMA inherits the weights of LLaMA-7B in the first stage.
We keep both InternViT-6B and QLLaMA frozen and only
train the newly added learnable queries and cross-attention
layers with filtered, high-quality data. Table 2 summarizes
the datasets for the second stage. It can be seen that we fur-
ther filtered out data with low-quality captions, reducing it
from 4.98 billion in the first stage to 1.03 billion.
Following the loss function of BLIP-2 [81], the loss
in this stage is computed as the sum of three compo-
nents: image-text contrastive (ITC) loss, image-text match-
ing (ITM) loss, and image-grounded text generation (ITG)
loss. This enables the queries to extract powerful visual rep-
resentations, and further align feature space with LLMs, at-
tributable to the effective training objectives and the utiliza-
tion of our large-scale, LLM-initialized QLLaMA.
Supervised Fine-tuning. To demonstrate the benefits of
InternVL in creating multi-modal dialogue systems, we
connect it with an off-the-shelf LLM decoder (e.g., Vi-
cuna [184] or InternLM [135]) through an MLP layer, and
conduct supervised fine-tuning (SFT). As detailed in Table
3, we collect a wide range of high-quality instruction data,
totaling approximately 4 million samples. For non-dialogue
datasets, we follow the method described in [91] for con-
version. Owing to the similar feature space of QLLaMA
and LLMs, we can achieve robust performance even when
freezing the LLM decoder, choosing to train just the MLP
layer or both the MLP layer and QLLaMA. This approach
not only expedites the SFT process but also maintains the
original language capabilities of the LLMs.
4. Experiments
4.1. Implementation Details
Stage 1. In this stage, the image encoder InternViT-6B is
randomly initialized [7], and the text encoder LLaMA-7B
is initialized with the pre-trained weights from [32]. All
parameters are fully trainable.
Stage 2.
In this stage, InternViT-6B and QLLaMA in-
herit their weights from the first stage, while the new learn-
able queries and cross-attention layers in QLLaMA are ran-
domly initialized. Benefiting from the powerful representa-
tions learned in the first stage, we keep both InternViT-6B
and QLLaMA frozen and only train the new parameters.
Stage 3. At this stage, we have two different configura-
tions. One is to use InternViT-6B separately, as shown in
Figure 4 (c). The other is to use the entire InternVL model
simultaneously, as shown in Figure 4 (d). More details will
be provided in the supplementary materials.
4.2. Visual Perception Benchmarks
First of all, we validate the visual perception capabilities of
InternViT-6B, the most core component of InternVL.
Transfer to Image Classification. We evaluate the qual-
ity of visual representation produced by InternViT-6B using
the ImageNet-1K [38] dataset. Following common prac-
tices [37, 58, 111], we adopt the linear probing evalua-
tion, i.e. training a linear classifier while keeping the back-
bone frozen. In addition to the ImageNet-1K validation set,
we also report performance metrics on several ImageNet
variants [10, 60, 61, 119, 141], to benchmark the domain
generalization capability. As shown in Table 4, InternViT-
6B achieves a very significant improvement over previous
state-of-the-art methods [46, 67, 111] on linear probing. To
our knowledge, this represents the currently best linear eval-
uation results without the JFT dataset [173].
Transfer to Semantic Segmentation. To investigate the
pixel-level perceptual capacity of InternViT-6B, we con-
duct extensive experiments of semantic segmentation on the
ADE20K [185] dataset. Following ViT-22B [37], we be-
6

method
IN-1K
IN-A
IN-R
IN-V2
IN-Sketch
ObjectNet
∆↓
avg.
OpenCLIP-H [67]
78.0
59.3
89.3
70.9
66.6
69.7
5.7
72.3
OpenCLIP-g [67]
78.5
60.8
90.2
71.7
67.5
69.2
5.5
73.0
OpenAI CLIP-L+ [117]
76.6
77.5
89.0
70.9
61.0
72.0
2.1
74.5
EVA-01-CLIP-g [130]
78.5
73.6
92.5
71.5
67.3
72.3
2.5
76.0
OpenCLIP-G [67]
80.1
69.3
92.1
73.6
68.9
73.0
3.9
76.2
EVA-01-CLIP-g+ [130]
79.3
74.1
92.5
72.1
68.1
75.3
2.4
76.9
MAWS-ViT-2B [128]
81.9
–
–
–
–
–
–
–
EVA-02-CLIP-E+ [130]
82.0
82.1
94.5
75.7
71.6
79.6
1.1
80.9
CoCa∗[169]
86.3
90.2
96.5
80.7
77.6
82.7
0.6
85.7
LiT-22B∗[37, 174]
85.9
90.1
96.0
80.9
−
87.6
−
−
InternVL-C (ours)
83.2
83.8
95.5
77.3
73.9
80.6
0.8
82.4
(a) ImageNet variants [38, 60, 61, 119, 141] and ObjectNet [8].
method
EN
ZH
JP
AR
IT
avg.
M-CLIP [16]
−
−
−
−
20.2
−
CLIP-Italian [11]
−
−
−
−
22.1
−
Japanese-CLIP-ViT-B [102]
−
−
54.6
−
−
−
Taiyi-CLIP-ViT-H [176]
−
54.4
−
−
−
−
WuKong-ViT-L-G [55]
−
57.5
−
−
−
−
CN-CLIP-ViT-H [162]
−
59.6
−
−
−
−
AltCLIP-ViT-L [26]
74.5
59.6
−
−
−
−
EVA-02-CLIP-E+ [130]
82.0
3.6
5.0
0.2
41.2
−
OpenCLIP-XLM-R-B [67]
62.3
42.7
37.9
26.5
43.7
42.6
OpenCLIP-XLM-R-H [67]
77.0
55.7
53.1
37.0
56.8
55.9
InternVL-C (ours)
83.2
64.5
61.5
44.9
65.7
64.0
(b) Multilingual ImageNet-1K [38, 76].
Table 6. Comparison of zero-shot image classification performance. “∆↓”: The gap between the averaged top-1 accuracy and the IN-1K
top-1 accuracy. ∗CoCa [169] and LiT-22B [37] use the private JFT-3B dataset [173] during training. Multilingual evaluation involves 5
languages, including English (EN), Chinese (ZH), Japanese (JP), Arabic (AR), and Italian (IT).
Flickr30K (English, 1K test set) [116]
COCO (English, 5K test set) [22]
multi-
Image →Text
Text →Image
Image →Text
Text →Image
method
lingual
R@1
R@5
R@10
R@1
R@5
R@10
R@1
R@5
R@10
R@1
R@5
R@10
avg.
Florence [171]
×
90.9
99.1
−
76.7
93.6
−
64.7
85.9
−
47.2
71.4
−
−
ONE-PEACE [143]
×
90.9
98.8
99.8
77.2
93.5
96.2
64.7
86.0
91.9
48.0
71.5
79.6
83.2
OpenCLIP-H [67]
×
90.8
99.3
99.7
77.8
94.1
96.6
66.0
86.1
91.9
49.5
73.4
81.5
83.9
OpenCLIP-g [67]
×
91.4
99.2
99.6
77.7
94.1
96.9
66.4
86.0
91.8
48.8
73.3
81.5
83.9
OpenCLIP-XLM-R-H [67]
✓
91.8
99.4
99.8
77.8
94.1
96.5
65.9
86.2
92.2
49.3
73.2
81.5
84.0
EVA-01-CLIP-g+ [130]
×
91.6
99.3
99.8
78.9
94.5
96.9
68.2
87.5
92.5
50.3
74.0
82.1
84.6
CoCa [169]
×
92.5
99.5
99.9
80.4
95.7
97.7
66.3
86.2
91.8
51.2
74.2
82.0
84.8
OpenCLIP-G [67]
×
92.9
99.3
99.8
79.5
95.0
97.1
67.3
86.9
92.6
51.4
74.9
83.0
85.0
EVA-02-CLIP-E+ [130]
×
93.9
99.4
99.8
78.8
94.2
96.8
68.8
87.8
92.8
51.1
75.0
82.7
85.1
BLIP-2† [81]
×
97.6
100.0
100.0
89.7
98.1
98.9
−
−
−
−
−
−
−
InternVL-C (ours)
✓
94.7
99.6
99.9
81.7
96.0
98.2
70.6
89.0
93.5
54.1
77.3
84.6
86.6
InternVL-G (ours)
✓
95.7
99.7
99.9
85.0
97.0
98.6
74.9
91.3
95.2
58.6
81.3
88.0
88.8
method
Flickr30K-CN (Chinese, 1K test set) [77]
COCO-CN (Chinese, 1K test set) [84]
avg.
WuKong-ViT-L [55]
×
76.1
94.8
97.5
51.7
78.9
86.3
55.2
81.0
90.6
53.4
80.2
90.1
78.0
R2D2-ViT-L [159]
×
77.6
96.7
98.9
60.9
86.8
92.7
63.3
89.3
95.7
56.4
85.0
93.1
83.0
Taiyi-CLIP-ViT-H [176]
×
−
−
−
−
−
−
−
−
−
60.0
84.0
93.3
−
AltCLIP-ViT-H [26]
✓
88.9
98.5
99.5
74.5
92.0
95.5
−
−
−
−
−
−
−
CN-CLIP-ViT-H [162]
×
81.6
97.5
98.8
71.2
91.4
95.5
63.0
86.6
92.9
69.2
89.9
96.1
86.1
OpenCLIP-XLM-R-H [67]
✓
86.1
97.5
99.2
71.0
90.5
94.9
70.0
91.5
97.0
66.1
90.8
96.0
87.6
InternVL-C (ours)
✓
90.3
98.8
99.7
75.1
92.9
96.4
68.8
92.0
96.7
68.9
91.9
96.5
89.0
InternVL-G (ours)
✓
92.9
99.4
99.8
77.7
94.8
97.3
71.4
93.9
97.7
73.8
94.4
98.1
90.9
Table 7.
Comparison of zero-shot image-text retrieval performance.
We evaluate the retrieval capability in English using the
Flickr30K [116] and COCO [22], as well as in Chinese using Flickr30K-CN [77] and COCO-CN [84].
†BLIP-2 [81] is finetuned on
COCO and zero-shot transferred to Flickr30K, contributing to the enhanced zero-shot performance on Flickr30K.
gin with few-shot learning experiments, i.e. fine-tuning the
backbone with a linear head on a limited dataset. As in-
dicated in Table 5a, InternViT-6B consistently outperforms
ViT-22B across five experiments with varying proportions
of training data. Additionally, Table 5b presents our fur-
ther verification in three distinct settings, including linear
probing, head tuning [158], and full-parameter tuning. No-
tably, in the case of linear probing, InternViT-6B attains
47.2 mIoU, a substantial +12.6 mIoU improvement over
ViT-22B. These results underscore the strong out-of-the-
box pixel-level perceptual capacity of our InternViT-6B.
4.3. Vision-Language Benchmarks
In this section, we evaluate the inherent capabilities of In-
ternVL on various vision-language tasks.
Zero-Shot Image Classification.
We conduct thorough
validation of the zero-shot image classification capabil-
K400 [17]
K600 [18]
K700 [19]
method
#F
top-1
avg.
top-1
avg.
top-1
avg.
OpenCLIP-g [67]
1
−
63.9
−
64.1
−
56.9
OpenCLIP-G [67]
1
−
65.9
−
66.1
−
59.2
EVA-01-CLIP-g+ [130]
1
−
66.7
−
67.0
−
60.9
EVA-02-CLIP-E+ [130]
1
−
69.8
−
69.3
−
63.4
InternVL-C (ours)
1
65.9
76.1
65.5
75.5
56.8
67.5
ViCLIP [152]
8
64.8
75.7
62.2
73.5
54.3
66.4
InternVL-C (ours)
8
69.1
79.4
68.9
78.8
60.6
71.5
Table 8. Comparison of zero-shot video classification results on
Kinetics 400/600/700. We report the top-1 accuracy and the mean
of top-1 and top-5 accuracy. “#F” denotes the number of frames.
ity of InternVL-C. As depicted in Table 6a, InternVL-
C attains leading performance on various ImageNet vari-
ants [38, 60, 61, 119, 141] and ObjectNet [8]. Compared
to EVA-02-CLIP-E+ [130], it exhibits stronger robustness
to distribution shift, manifesting in a more consistent accu-
racy across ImageNet variants. Additionally, as shown in
7

visual
glue
train.
image captioning
visual question answering
dialogue
method
encoder
layer
LLM
Res.
PT
SFT
param
COCO
Flickr NoCaps
VQAv2
GQA
VizWiz VQAT
MME
POPE
InstructBLIP [34]
EVA-g
QFormer
Vicuna-7B
224
129M
1.2M
188M
–
82.4
123.1
–
49.2
34.5
50.1
–
–
BLIP-2 [81]
EVA-g
QFormer
Vicuna-13B
224
129M
–
188M
–
71.6
103.9
41.0
41.0
19.6
42.5
1293.8
85.3
InstructBLIP [34]
EVA-g
QFormer
Vicuna-13B
224
129M
1.2M
188M
–
82.8
121.9
–
49.5
33.4
50.7
1212.8
78.9
InternVL-Chat (ours) IViT-6B
QLLaMA
Vicuna-7B
224
1.0B
4.0M
64M
141.4∗
89.7
120.5
72.3∗
57.7∗
44.5
42.1
1298.5
85.2
InternVL-Chat (ours) IViT-6B
QLLaMA
Vicuna-13B
224
1.0B
4.0M
90M
142.4∗
89.9
123.1
71.7∗
59.5∗
54.0
49.1
1317.2
85.4
Shikra [21]
CLIP-L
Linear
Vicuna-13B
224
600K
5.5M
7B
117.5∗
73.9
–
77.4∗
–
–
–
–
–
IDEFICS-80B [66]
CLIP-H
Cross-Attn
LLaMA-65B
224
1.6B
–
15B
91.8∗
53.7
65.0
60.0
45.2
36.0
30.9
–
–
IDEFICS-80B-I [66]
CLIP-H
Cross-Attn
LLaMA-65B
224
353M
6.7M
15B
117.2∗
65.3
104.5
37.4
–
26.0
–
–
–
Qwen-VL [5]
CLIP-G
VL-Adapter Qwen-7B
448
1.4B†
50M†
9.6B
–
85.8
121.4
78.8∗
59.3∗
35.2
63.8
–
–
Qwen-VL-Chat [5]
CLIP-G
VL-Adapter Qwen-7B
448
1.4B†
50M†
9.6B
–
81.0
120.2
78.2∗
57.5∗
38.9
61.5
1487.5
–
LLaVA-1.5 [91]
CLIP-L336
MLP
Vicuna-7B
336
558K
665K
7B
–
–
–
78.5∗
62.0∗
50.0
58.2
1510.7
85.9
LLaVA-1.5 [91]
CLIP-L336
MLP
Vicuna-13B
336
558K
665K
13B
–
–
–
80.0∗
63.3∗
53.6
61.3
1531.3
85.9
InternVL-Chat (ours) IViT-6B
MLP
Vicuna-7B
336
558K
665K
7B
–
–
–
79.3∗
62.9∗
52.5
57.0
1525.1
86.4
InternVL-Chat (ours) IViT-6B
MLP
Vicuna-13B
336
558K
665K
13B
–
–
–
80.2∗
63.9∗
54.6
58.7
1546.9
87.1
InternVL-Chat (ours) IViT-6B
QLLaMA
Vicuna-13B
336
1.0B
4.0M
13B
146.2∗
92.2
126.2
81.2∗
66.6∗
58.5
61.5
1586.4
87.6
Table 9. Comparison with SoTA methods on 9 benchmarks. Image captioning datasets include: COCO Karpathy test [22], Flickr30K
Karpathy test [116], NoCaps val [2]. VQA datasets include: VQAv2 test-dev [54], GQA test-balanced [64], VizWiz test-dev [56], and
TextVQA val [127]. ∗The training annotations of the datasets are observed during training. “IViT-6B” represents our InternViT-6B.
method
glue layer
LLM decoder
COCO Flickr30K NoCaps
Flamingo-9B [3]
Cross-Attn
Chinchilla-7B
79.4
61.5
–
Flamingo-80B [3]
Cross-Attn
Chinchilla-70B
84.3
67.2
–
KOSMOS-2 [115]
Linear
KOSMOS-1
–
66.7
–
PaLI-X-55B [24]
Linear
UL2-32B
–
–
126.3
BLIP-2 [81]
QFormer
Vicuna-13B
–
71.6
103.9
InstructBLIP [34]
QFormer
Vicuna-13B
–
82.8
121.9
Shikra-13B [21]
Linear
Vicuna-13B
–
73.9
–
ASM [149]
QFormer
Husky-7B
–
87.7
117.2
Qwen-VL [5]
VL-Adapter Qwen-7B
–
85.8
121.4
Qwen-VL-Chat [5]
VL-Adapter Qwen-7B
–
81.0
120.2
Emu [131]
QFormer
LLaMA-13B
112.4
–
–
Emu-I [131]
QFormer
LLaMA-13B
117.7
–
–
DreamLLM [41]
Linear
Vicuna-7B
115.4
–
–
InternVL-G (ours)
Cross-Attn
QLLaMA
128.2
79.2
113.7
Table 10. Comparison of zero-shot image captioning. QLLaMA
inherently possesses promising zero-shot captioning capabilities
thanks to its scaled-up parameters and datasets.
Table 6b, our model showcases robust multilingual capabil-
ities, outperforming competing models [16, 26, 67, 162] on
the multilingual ImageNet-1K benchmark.
Zero-Shot Video Classification. Following previous meth-
ods [117, 130, 152], we report the top-1 accuracy and the
mean of top-1 and top-5 accuracy on Kinetics-400/600/700
[17–19]. As shown in Table 8, when sampling only a sin-
gle center frame in each video, our method achieves an av-
erage accuracy of 76.1%, 75.5%, and 67.5% on the three
datasets, surpassing EVA-02-CLIP-E+ [130] by +6.3, +6.2,
and +4.1 points, respectively. Additionally, when uniformly
sampling 8 frames in each video, we obtain at least 3.3
points of improvement compared to the single-frame set-
ting, outperforming ViCLIP [152] trained using web-scale
video data. In summary, InternVL-C exhibits remarkable
generalization capabilities in video classification.
Zero-Shot Image-Text Retrieval.
InternVL exhibits a
powerful multilingual image-text retrieval capability. In Ta-
ble 7, we evaluate these capabilities in English using the
Flickr30K [116] and COCO [22] datasets, as well as in
Chinese using the Flickr30K-CN [77] and COCO-CN [84].
Additionally, we leverage the XTD dataset [1] to evalu-
ate the multilingual image-text retrieval capability across
8 languages (see supplementary materials). In summary,
InternVL-C achieves state-of-the-art performance across
most retrieval metrics, and with the second stage of pre-
training, InternVL-G further enhances zero-shot image-text
retrieval performance.
These improvements in retrieval
tasks suggest a more effective alignment between visual and
linguistic features, through additional image encoding using
the language middleware–QLLaMA.
Zero-Shot Image Captioning.
Benefiting from vision-
language generative training on a vast collection of high-
quality image-text pairs, our QLLaMA possesses promis-
ing capability in zero-shot image captioning.
As shown
in Table 10, QLLaMA surpasses other models in zero-shot
performance on the COCO Karpathy test set [22]. It also
achieves comparable results to current state-of-the-art mod-
els on both the Flickr30K Karpathy test [116] and the No-
Caps val set [2]. When InternVL is linked with an LLM
(e.g., Vicuna-7B/13B [184]) and subjected to SFT, a notable
enhancement in zero-shot performance is observed for both
Flickr30K and NoCaps, as shown in Table 9.
4.4. Multi-Modal Dialogue Benchmarks
Beyond the traditional multi-modal tasks, the emergence
of ChatGPT [110] has led to a growing focus on evaluat-
ing the performance of multi-modal models in real usage
scenarios, specifically within the realm of multi-modal di-
alogue. We conducted testing of InternVL-Chat models on
two prominent multi-modal dialogue benchmarks, includ-
ing MME [50] and POPE [86].
MME is a comprehen-
sive benchmark that includes 14 sub-tasks focusing on the
model’s perception and cognition capabilities. POPE is a
popular dataset used to evaluate object hallucination. As
shown in Table 9, it clearly demonstrates that our models
exhibit superior performance compared with previous meth-
8

name
width
depth
MLP
#heads
#param
FLOPs
throughput
zs IN
variant 1
3968
32
15872
62
6051M
1571G
35.5 / 66.0
65.8
variant 2
3200
48
12800
50
5903M
1536G
28.1 / 64.9
66.1
variant 3
3200
48
12800
25
5903M
1536G
28.0 / 64.6
66.2
variant 4
2496
48
19968
39
5985M
1553G
28.3 / 65.3
65.9
variant 5
2816
64
11264
44
6095M
1589G
21.6 / 61.4
66.2
variant 6
2496
80
9984
39
5985M
1564G
16.9 / 60.1
66.2
Table 11. Comparison of hyperparameters in InternViT-6B.
The throughput (img/s) and GFLOPs are measured at 224×224 in-
put resolution, with a batch size of 1 or 128 on a single A100 GPU.
Flash Attention [35] and bf16 precision are used during testing.
“zs IN” denotes the zero-shot top-1 accuracy on the ImageNet-1K
validation set [38]. The final selected model is marked in gray .
visual
glue
LLM
dataset
dialogue caption visual question answering
encoder layer
MME
NoCaps OKVQA VizWizval GQA
EVA-E MLP
V-7B
665K [91]
970.5
75.1
40.1
25.5
41.3
IViT-6B MLP
V-7B
665K [91] 1022.3
80.8
42.9
28.3
45.8
IViT-6B QLLaMA V-7B
665K [91] 1227.5
94.5
51.0
38.4
57.4
IViT-6B QLLaMA V-7B
Ours
1298.5
120.5
51.8
44.9
57.7
IViT-6B QLLaMA V-13B
Ours
1317.2
123.1
55.5
55.7
59.5
Table 12. Ablation studies of using InternVL to build multi-
modal dialogue system. V-7B and V-13B denote Vicuna-7B/13B
[184], respectively. “IViT-6B” represents our InternViT-6B.
ods, under the condition of fair trainable parameter counts.
4.5. Ablation Study
Hyperparameters of InternViT-6B. As discussed in Sec-
tion 3.2, we explored variations in model depth {32, 48,
64, 80}, head dimension {64, 128}, and MLP ratio {4,
8}, resulting in 16 distinct models.
In selecting the op-
timal model, we initially narrowed down our focus to 6
models, chosen based on their throughput, as listed in Ta-
ble 11. These models underwent further evaluation using
contrastive learning on a 100M subset of LAION-en [120]
over 10K iterations. For the experimental setup, the primary
difference was the use of a randomly initialized text encoder
from CLIP-L [117], in order to speed up the training. For
the sake of accuracy, inference speed, and training stability,
we ultimately chose variant 3 as the final InternViT-6B.
Consistency of Feature Representation. In this study, we
validate the consistency of the feature representation of In-
ternVL with off-the-shelf LLMs. We adopt a minimalist
setting, i.e. conducting a single-stage SFT using only the
LLaVA-Mix-665K [85] dataset. Moreover, only the MLP
layers are trainable, thereby confirming the inherent align-
ment level among features from various vision foundation
models and LLMs. The results are shown in Table 12. We
observed that compared to EVA-E [130], our InternViT-6B
achieves better performance under this simple setup. Addi-
tionally, it is noteworthy that performance across all three
tasks saw significant improvement when using QLLaMA
as the “glue layer”. These significant improvements clearly
delineate that the feature representation of InternVL is more
consistent with the off-the-shelf LLM.
5. Conclusion
In this paper, we present InternVL, a large-scale vision-
language foundation model that scales up the vision founda-
tion model to 6 billion parameters and is aligned for generic
visual-linguistic tasks.
Specifically, we design a large-
scale vision foundation model InternViT-6B, progressively
align it with an LLM-initialized language middleware QL-
LaMA, and leverage web-scale image-text data from vari-
ous sources for efficient training. It bridges the gap between
vision foundation models and LLMs, and demonstrates pro-
ficiency in a wide range of generic visual-linguistic tasks,
such as image/video classification, image/video-text re-
trieval, image captioning, visual question answering, and
multi-modal dialogue. We hope this work could contribute
to the development of the VLLM community.
Acknowledgement
We thank Shenglong Zhang, Beitong Zhou, Xinyue Zhang,
Dongxing Shi, Weigao Sun, Xingcheng Zhang, and Zhifeng
Yue for their contributions to the optimization of the train-
ing framework. We thank Zhenhang Huang for his assis-
tance in data preparation.
9

A. Supplementary Materials
A.1. More Experiments
Zero-Shot Image Classification on 20 Datasets. In this
section, we expand our examination to showcase the effec-
tiveness and robustness of InternVL in 20 different zero-
shot image classification benchmarks. As indicated in Ta-
ble 16, InternVL registers an average performance of 78.1%
across all 20 benchmarks. This performance notably ex-
ceeds that of the previously leading method, EVA-02-CLIP-
E+ [47], by a margin of 1.0 points. This underscores that,
beyond ImageNet [38] and its variants, InternVL possesses
robust generalization capabilities across a variety of differ-
ent domains in zero-shot image classification.
Zero-Shot Image-Text Retrieval on XTD. Table 13 re-
ports the results of InternVL on the multilingual image-text
retrieval dataset XTD [1], spanning eight languages. As can
be seen, InternVL-C achieves an average recall@10 score
of 95.1% across these languages. The second stage model,
InternVL-G, further improves retrieval performance. It at-
tains the highest scores in each individual language and es-
tablishes a new record for average performance at 96.6%.
Zero-Shot Video Retrieval. In Table 14, we present our
results of zero-shot video-text retrieval on the MSR-VTT
dataset [161] using our InternVL models, i.e. InternVL-C
and InternVL-G. In the 1-frame setting, we select a sin-
gle central frame from each video.
In the 8-frame set-
ting, we uniformly extract 8 frames from each video, treat
them as independent images for encoding, and then average
the embeddings. The results showcase consistent improve-
ment across various metrics such as R@1, R@5, R@10,
and the average score. Importantly, both models exhibit
promising outcomes in single-frame and multi-frame con-
figurations, with InternVL-G achieving slightly higher per-
formance than InternVL-C, especially in the multi-frame
setting. These results underscore the effectiveness of QL-
LaMA in harmonizing visual and linguistic features.
Fine-tuned Image-Text Retrieval. In Table 15, we report
the fine-tuned image-text retrieval results of InternVL, on
both the English and Chinese versions of the Flickr30K
dataset [77, 116]. The specific hyperparameters for fine-
tuning are shown in Table 21. As can be seen, our mod-
els obtain competitive performance, with InternVL-G-FT
marginally surpassing InternVL-C-FT in both datasets. No-
tably, in the highly challenging Flickr30K-CN, both models
show a promising ability to handle cross-lingual retrieval
tasks. These results demonstrate the effectiveness of our
language middleware, especially in the retrieval tasks.
Tiny LVLM. Tiny LVLM [123] is an ability-level bench-
mark for evaluating the performance of multimodal dia-
logue models. It provides a systematic assessment of five
categories of multimodal capabilities, including visual per-
ception, visual knowledge acquisition, visual reasoning, vi-
method
EN
ES
FR
ZH
IT
KO
RU
JP
avg.
mUSE m3 [164]
85.3 78.9 78.9 76.7 73.6 67.8 76.1 70.7 76.0
M-CLIP [16]
92.4 91.0 90.0 89.7 91.1 85.2 85.8 81.9 88.4
MURAL [69]
−
92.9
−
89.7 91.8 88.1 87.2
−
−
AltCLIP [26]
95.4 94.1 92.9 95.1 94.2 94.4 91.8 91.7 93.7
OpenCLIP-XLM-R-B [67] 95.8 94.4 92.5 91.8 94.4 86.3 89.9 90.7 92.0
OpenCLIP-XLM-R-H [67] 97.3 96.1 94.5 94.7 96.0 90.2 93.9 94.0 94.6
InternVL-C (ours)
97.3 95.7 95.1 95.6 96.0 92.2 93.3 95.5 95.1
InternVL-G (ours)
98.6 97.7 96.5 96.7 96.9 95.1 94.8 96.1 96.6
Table 13. Comparison of zero-shot multilingual image-text re-
trieval performance on the XTD dataset. Multiple languages
include English (EN), Spanish (ES), French (FR), Chinese (ZH),
Italian (IT), Korean (KO), Russian (RU), and Japanese (JP). We
follow M-CLIP [16] to report the recall@10 on Image-to-Text.
MSR-VTT (1K test set) [161]
Video →Text
Text →Video
method
#F
R@1
R@5
R@10
R@1
R@5
R@10
avg.
OpenAI CLIP-L [117]
1
27.8
49.4
58.0
29.0
50.5
59.2
45.7
InternVL-C (ours)
1
35.3
56.6
66.6
37.5
60.9
70.9
54.6
InternVL-G (ours)
1
36.6
58.3
67.7
39.1
61.7
70.7
55.7
OpenAI CLIP-L [117]
8
26.6
50.8
61.8
30.7
54.4
64.0
48.1
Florence [171]
8
–
–
–
37.6
63.8
72.6
–
InternVideo† [151]
8
39.6
–
–
40.7
–
–
–
UMT-L† [83]
8
38.6
59.8
69.6
42.6
64.4
73.1
58.0
LanguageBind† [186]
8
40.9
66.4
75.7
44.8
70.0
78.7
62.8
InternVL-C (ours)
8
40.2
63.1
74.1
44.7
68.2
78.4
61.5
InternVL-G (ours)
8
42.4
65.9
75.4
46.3
70.5
79.6
63.4
Table 14.
Comparison of zero-shot video-text retrieval per-
formance on MSR-VTT. “#F” denotes the number of frames.
† These models are trained with temporal attention layers.
Flickr30K (English, 1K test set) [116]
Image →Text
Text →Image
method
R@1
R@5
R@10
R@1
R@5
R@10
avg.
ALIGN [70]
95.3
99.8
100.0
84.9
97.4
98.6
96.0
FILIP [167]
96.6
100.0
100.0
87.1
97.7
99.1
96.8
Florence [171]
97.2
99.9
−
87.9
98.1
−
−
BLIP [80]
97.4
99.8
99.9
87.6
97.7
99.0
96.9
OmniVL [142]
97.3
99.9
100.0
87.9
97.8
99.1
97.0
BEiT-3 [146]
97.5
99.9
100.0
89.1
98.6
99.3
97.4
ONE-PEACE [143]
97.6
100.0
100.0
89.6
98.0
99.1
97.4
InternVL-C-FT (ours)
97.2
100.0
100.0
88.5
98.4
99.2
97.2
InternVL-G-FT (ours)
97.9
100.0
100.0
89.6
98.6
99.2
97.6
method
Flickr30K-CN (Chinese, 1K test set) [77]
avg.
Wukong-ViT-L [55]
92.7
99.1
99.6
77.4
94.5
97.0
93.4
CN-CLIP-ViT-H [162]
95.3
99.7
100.0
83.8
96.9
98.6
95.7
R2D2-ViT-L [159]
95.6
99.8
100.0
84.4
96.7
98.4
95.8
InternVL-C-FT (ours)
96.5
99.9
100.0
85.2
97.0
98.5
96.2
InternVL-G-FT (ours)
96.9
99.9
100.0
85.9
97.1
98.7
96.4
Table 15. Comparison of fine-tuned image-text retrieval per-
formance. We evaluate English and Chinese image-text retrieval
using Flickr30K [116] and Flickr30K-CN [77], with separate fine-
tuning for each to prevent data leakage.
sual commonsense, and object hallucination. We report our
results on Tiny LVLM in Table 17.
A.2. More Ablation Studies
Compatibility with Other LLM. In this experiment, we
test the compatibility of InternVL with LLMs other than
Vicuna [184].
The experimental setup used here is the
same as in Table 9 of the main paper. As shown in Table
10

method
CIFAR-10 [74]
CIFAR-100 [74]
MNIST [78]
Caltech-101 [49]
SUN397 [157]
FGVC Aircraft [101]
Country-211 [117]
Stanford Cars [72]
Birdsnap [9]
DTD [28]
Eurosat [59]
FER2013 [52]
Flowers-102 [109]
Food-101 [13]
GTSRB [129]
Pets [113]
Rendered SST2 [117]
Resisc45 [27]
STL10 [30]
VOC2007 [45]
avg. top-1 acc.
OpenAI CLIP-L+ [117]
94.9
74.4
79.0
87.2
68.7
33.4
34.5
79.3
41.0
56.0
61.5
49.1
78.6
93.9
52.4
93.8
70.7
65.4
99.4
78.1
69.6
EVA-01-CLIP-g [130]
98.3
88.7
62.3
87.7
74.2
32.4
28.6
91.7
50.0
61.3
73.6
52.2
74.5
93.5
49.1
94.2
58.4
70.3
98.9
83.2
71.2
OpenCLIP-g [67]
98.2
84.7
71.9
88.1
74.1
44.6
30.9
94.0
51.0
68.7
64.7
55.8
81.0
92.4
49.7
93.9
56.7
69.6
98.9
81.6
72.5
OpenCLIP-H [67]
97.4
84.7
72.9
85.0
75.2
42.8
30.0
93.5
52.9
67.8
72.7
52.0
80.1
92.7
58.4
94.5
64.3
70.5
98.5
77.7
73.2
EVA-02-CLIP-L+ [130]
98.9
89.8
64.3
89.5
74.8
37.5
33.6
91.6
45.8
64.5
71.4
51.0
77.2
94.2
57.6
94.2
64.6
69.8
99.7
82.7
72.6
EVA-01-CLIP-g+ [130]
99.1
90.1
71.8
88.1
74.3
39.4
30.8
90.7
52.6
67.3
73.2
56.0
79.7
93.7
66.5
94.8
58.6
71.4
99.5
82.9
74.0
OpenCLIP-G [67]
98.2
87.5
71.6
86.4
74.5
49.7
33.8
94.5
54.5
69.0
70.0
59.5
81.5
93.1
62.5
95.2
65.2
72.6
98.5
80.7
74.9
EVA-02-CLIP-E [130]
99.3
92.5
76.7
89.0
76.5
47.9
34.7
94.4
56.3
68.2
77.6
55.1
82.5
95.2
67.1
95.6
61.1
73.5
99.2
83.0
76.3
EVA-02-CLIP-E+ [130]
99.3
93.1
74.7
90.5
75.1
54.1
35.7
94.6
58.1
68.2
75.8
58.6
84.5
94.9
67.7
95.8
61.4
75.6
99.2
85.6
77.1
InternVL-C (ours)
99.4
93.2
80.6
89.5
76.0
52.7
34.1
94.2
72.0
70.7
79.4
56.2
86.1
95.3
65.5
96.0
67.9
74.2
99.5
80.0
78.1
Table 16. Comparison of zero-shot image classification performance on 20 other datasets. These results indicate that, in addition to
ImageNet [38], InternVL also possesses good generalization capabilities in zero-shot image classification across various domains.
method
LLM
VR
VP VKA VC
OH Overall
MiniGPT-4 [187]
Vicuna-7B
37.6 37.8 17.6 49.0 50.7
192.6
LLaVA [92]
Vicuna-7B
41.6 38.3 18.7 49.4 49.0
197.0
VisualGLM [44]
ChatGLM-6B 37.3 36.3 46.9 37.6 54.0
211.9
Otter [79]
Otter-9B
41.6 37.0 15.1 52.4 74.0
216.4
LLaMA-Adapter-V2 [51]
LLaMA-7B
43.5 46.8 22.3 56.0 60.7
229.2
Lynx [172]
Vicuna-7B
52.2 65.8 17.6 57.4 86.3
279.2
BLIP-2 [81]
FlanT5xl
44.9 49.0 64.1 44.0 82.7
284.7
InstructBLIP [34]
Vicuna-7B
46.7 48.0 61.7 59.2 85.0
300.6
LLaVA-1.5 [91]
Vicuna-7B
55.6 49.0 57.0 57.2 88.3
307.2
Qwen-VL-Chat [5]
Qwen-7B
62.4 54.5 55.1 54.8 90.0
316.8
Bard [53]
Bard
64.2 57.0 68.1 59.6 70.7
319.6
InternLM-XComposer [177] InternLM-7B
55.8 53.8 64.1 61.8 87.0
322.5
InternVL-Chat (ours)
Vicuna-13B
56.4 52.3 68.0 62.0 89.0
327.6
Table 17. Evaluation of Tiny LVLM test set. Here we report
five categories of multimodal capabilities, including visual rea-
soning (VR), visual perception (VP), visual knowledge acquisition
(VKA), visual commonsense (VC), and object hallucination (OH).
18, InternLM-7B [135] achieves slightly better performance
than Vicuna-7B [184]. This indicates that our InternVL ex-
hibits promising compatibility with various LLMs.
Efficiency Analysis. In this study, we analyze the com-
putational efficiency of InternVL in encoding image-text
pairs. The entire encoding process consists of two parts:
image encoding and text encoding. The analysis covered
two models (InternVL-C and InternVL-G) and their per-
formance across three different image sizes (224, 336, and
448). The results are shown in Table 19.
From these results, we find that: (1) As the image size
increases, the encoding time also significantly increases,
leading directly to a decrease in frame rate; (2) InternVL-G
slightly increased the encoding time due to the introduc-
tion of QLLaMA for secondary image encoding, but it still
maintains a reasonable frame rate across all image sizes;
(3) Even though we scale up the text encoder, the addi-
tional cost of text encoding is not significant, as the main
time expenditure lies in image encoding. In summary, when
choosing between InternVL-C and InternVL-G, one should
weigh the trade-off between computational efficiency and
visual
glue
visual question answering
dialogue
encoder
layer LLM
VQAv2 GQA VizWiz VQAT
MME
POPE
IViT-6B MLP Vicuna-7B
79.3
62.9
52.5
57.0
1525.1
86.4
IViT-6B MLP InternLM-7B
79.7
63.2
53.1
58.0
1532.8
86.4
Table 18. Compatibility with other LLM. Here we use InternLM
[135] as an example to verify the compatibility of InternVL with
LLMs other than Vicuna [184]. The experimental settings used
here are the same as in Table 9 of the main paper.
image
encode image (ms)
encode text (ms)
total
method
size
InternViT-6B
QLLaMA
QLLaMA
time
FPS
InternVL-C
224
15.5
–
4.9
20.4
48.9
InternVL-C
336
35.2
–
4.9
40.1
24.9
InternVL-C
448
66.9
–
4.9
71.8
13.9
InternVL-G
224
15.5
8.2
4.9
28.6
35.0
InternVL-G
336
35.2
10.3
4.9
50.4
19.8
InternVL-G
448
66.9
12.8
4.9
84.6
11.8
Table 19. Efficiency analysis of InternVL for encoding image-
text pairs. The total time to encode an image-text pair includes
both the image encoding part and the text encoding part. We mea-
sure the time cost with a batch size of 128 on a single A100 GPU.
Flash Attention [35] and bf16 precision are used during testing.
potential performance improvements based on specific re-
quirements. Additionally, these results were measured us-
ing PyTorch with Flash Attention [35] and bf16 precision,
and there is still considerable room for optimization, such
as using model quantization and TensorRT.
A.3. Detailed Training Settings
Settings of Stage 1. As shown in Table 20, in this stage, the
image encoder InternViT-6B is randomly initialized using
the BEiT’s initialization method [7], and the text encoder
LLaMA-7B is initialized with the pre-trained weights from
[32], a multilingual LLaMA-7B. All parameters are fully
trainable. We employ the AdamW optimizer [98] with β1 =
0.9, β2 = 0.95, weight decay at 0.1, and a cosine learning
rate schedule starting at 1e-3 and 1e-4 for the image and
text encoders, respectively. We adopt a uniform drop path
11

config
stage 1
stage 2
image enc. weight init.
random init. [7]
from stage 1
text enc. weight init.
from [32]
from stage 1
image enc. peak learning rate
1e-3
frozen
text enc. peak learning rate
1e-4
frozen
cross attn peak learning rate
–
5e-5
learning rate schedule
cosine decay
cosine decay
optimizer
AdamW [98]
AdamW [98]
optimizer hyper-parameters
β1, β2 = 0.9, 0.95
β1, β2 = 0.9, 0.98
weight decay
0.1
0.05
input resolution
1962 →2242
2242
patch size
14
14
total batch size
164K
20K
warm-up iterations
5K
2K
total iterations
175K
80K
samples seen
28.7B
1.6B
drop path rate [63]
uniform (0.2)
0.0
data augmentation
random resized crop
random resized crop
numerical precision
DeepSpeed bf16 [118]
DeepSpeed bf16 [118]
trainable / total parameters
13B / 13B
1B / 14B
GPUs for training
640×A100 (80G)
160×A100 (80G)
Table 20. Training settings of InternVL’s stage 1 and stage 2.
“1962 →2242” means we initially train at a 196×196 resolution,
and later switch to 224×224 resolution for the final 0.5 billion
samples, for higher training efficiency.
rate of 0.2. The training involves a total batch size of 164K
across 640 A100 GPUs, extending over 175K iterations to
process about 28.7 billion samples. To enhance efficiency,
we initially train at a 196×196 resolution, masking 50% of
image tokens [87], and later switch to 224×224 resolution
without masking for the final 0.5 billion samples.
Settings of Stage 2. In this stage, InternViT-6B and QL-
LaMA inherit their weights from the first stage, while the
learnable queries and cross-attention layers in QLLaMA
are randomly initialized. Benefiting from the powerful en-
coding capabilities learned in the first stage, we keep both
InternViT-6B and QLLaMA frozen and only train the newly
added parameters. The input images are processed at a res-
olution of 224×224. For optimization, the AdamW opti-
mizer [98] is employed with β1 = 0.9, β2 = 0.98, weight
decay set at 0.05, and a total batch size of 20K. The training
extends over 80K steps across 160 A100 GPUs, inclusive of
2K warm-up steps, and is governed by a cosine learning rate
schedule with a peak learning rate of 5e-5. More detailed
training settings are listed in Table 20.
Settings of Stage 3. At this stage, we have two different
configurations. One is to use InternViT-6B separately, as
shown in Figure 4 (c). The other is to use the entire In-
ternVL model simultaneously, as shown in Figure 4 (d).
(1) InternVL-Chat (w/o QLLaMA): For this setup, we
follow the training recipes of LLaVA-1.5 [91].
We use
the same hyperparameters and datasets for supervised fine-
tuning, i.e. we first train the MLP layers with the LGS-558K
[92] dataset, and then train the LLM with the LLaVA-Mix-
665K [91] dataset, both for one epoch.
(2) InternVL-Chat (w/ QLLaMA): For this more ad-
vanced setup, we also conducted the training in two steps.
We first train the MLP layers with our custom SFT dataset
config
retrieval fine-tuning
image-text data
Flickr30K [116] / Flickr30K-CN [77]
peak learning rate
1e-6
layer-wise lr decay rate
InternViT-6B (0.9), QLLaMA (0.9)
learning rate schedule
cosine decay
optimizer
AdamW [98]
optimizer hyper-parameters
β1, β2 = 0.9, 0.999
weight decay
0.05
input resolution
3642
patch size
14
total batch size
1024
warm-up iterations
100
training epochs
10
drop path rate [63]
0.3
data augmentation
random resized crop & flip
numerical precision
DeepSpeed bf16 [118]
trainable / total parameters
14B / 14B
GPUs for training
32×A100 (80G)
Table 21. Training settings of retrieval fine-tuning. We fine-
tune InternVL on Flickr30K and Flickr30K-CN separately.
config
ImageNet linear probing
peak learning rate
0.2
learning rate schedule
cosine decay
optimizer
SGD
optimizer momentum
0.9
weight decay
0.0
input resolution
2242
patch size
14
total batch size
1024
warm-up epochs
1
training epochs
10
data augmentation
random resized crop & flip
GPUs for training
8×A100 (80G)
Table 22. Training settings of ImageNet linear probing.
and then fine-tune the LLM with it. Due to the expansion of
the dataset, we increased the batch size to 512.
Settings of Retrieval Fine-tuning. In this experiment, all
parameters of InternVL are set to be trainable. We conduct
separate fine-tuning on the Flickr30K [116] and Flickr30K-
CN [77]. Following common practice [81], a 364×364 res-
olution is adopted for fine-tuning.
To avoid over-fitting,
we apply a layer-wise learning rate decay of 0.9 to both
InternViT-6B and QLLaMA, along with a drop path rate
of 0.3 for InternViT-6B. The AdamW optimizer [98] is uti-
lized, with a total batch size of 1024, for fine-tuning the In-
ternVL model across 10 epochs. For more detailed training
settings, please refer to Table 21.
Settings of ImageNet Linear Probing.
We follow the
common practices of linear probing in previous methods
[37, 58, 111]. Specifically, we employ an additional Batch-
Norm [68] to normalize the pre-trained backbone features
during training.
Besides, we concatenate the average-
pooled patch token features with the class token. The linear
head is trained using the SGD optimizer for 10 epochs on
ImageNet-1K [38], with a total batch size of 1024, a peak
learning rate of 0.2, 1 epoch warm-up, and no weight de-
cay. Data augmentation involves random-resized-crop and
flip. For more training details, please see Table 22.
Settings of ADE20K Semantic Segmentation. In Table
12

(a) Training Data for Stage 1 & 2
(b) Testing Datasets for Image Classification
LAION-COCO
ImageNet-ReaL
Country-211
FER2013
Rendered SST2
CIFAR-100
COYO
SBU
ImageNet-V2
ImageNet-Sketch
Stanford Cars
Flowers-102
Resisc45
MNIST
LAION-multi
CC3M
ObjectNet
ImageNet-A
Birdsnap
Food-101
STL10
Caltech-101
Wukong
CC12M
Multilingual IN-1K
ImageNet-R
DTD
GTSRB
VOC2007
SUN397
LAION-en
ImageNet-1K
RGVC Aircraft
Eurosat
Pets
CIFAR-10
Flickr30K
COCO-CN
Flickr30K-CN
XTD
COCO
Kinetics 400
Kinetics 600
Kinetics 700
(d) Testing Datasets for Image-Text Retrieval
(c) Testing Datasets for Video Classification
ADE20K
COCO
Flickr30K
NoCaps
MSR-VTT
(e) Testing Dataset for Video-Text Retrieval
(f) Testing Datasets for Image Captioning
(g) Testing Dataset for Segmentation
Training Sets (English)
Training Sets (Multilingual)
ImageNet-1K
Zero-Shot Test Sets (English)
Zero-Shot Test Sets (Multilingual)
Datasets for Transfer Learning
Figure 5. Panoramic overview of the datasets used in InternVL’s stage 1 and stage 2. During the training of stage 1 and stage 2, we
utilize web-scale image-text data from a variety of sources to train our InternVL model, as shown in (a). To assess InternVL’s capabilities
in handling generic visual-linguistic tasks, we conducted extensive validations across a range of tasks and datasets, including (b) image
classification, (c) video classification, (d) image-text retrieval, (e) video-text retrieval, (f) image captioning, and (g) semantic segmentation.
config
linear probing / head tuning / full tuning
peak learning rate
4e-5
layer-wise lr decay rate
– / – / 0.95
learning rate schedule
polynomial decay
optimizer
AdamW [98]
optimizer hyper-parameters
β1, β2 = 0.9, 0.999
weight decay
0.0 / 0.05 / 0.05
input resolution
5042
patch size
14
total batch size
16
warm-up iterations
1.5K
total iterations
80K
drop path rate [63]
0.0 / 0.0 / 0.4
data augmentation
default augmentation in MMSeg [31]
numerical precision
DeepSpeed bf16 [118]
GPUs for training
8×A100 (80G)
Table 23. Training settings of ADE20K semantic segmentation.
We list the hyperparameters for three different configurations, in-
cluding linear probing, head tuning, and full-parameter tuning.
23, we have listed the hyperparameters for three different
configurations in ADE20K semantic segmentation, includ-
ing linear probing, head tuning, and full-parameter tuning.
A.4. Data Preparation for Pre-training
Training Data for Stage 1 & Stage 2. During the first
and second stages, we employed a vast collection of image-
text pair data (see Figure 5 (a)), such as LAION-en [120],
LAION-multi [120], LAION-COCO [121], COYO [14],
Wukong [55], among others [20, 112, 124]. A detailed in-
troduction to these datasets is provided in Table 24.
Training Data Cleaning for Stage 1 & Stage 2. To fully
utilize web-scale image-text data, we adopted different data
filtering strategies in stage 1 and stage 2.
(1) Stage 1: In the first stage, we applied only minor data
filtering, thus retaining the vast majority of the data. We
considered six factors: CLIP similarity, watermark proba-
bility, unsafe probability, aesthetic score, image resolution,
and caption length, to remove extreme data points and avoid
disrupting training stability. Additionally, we removed data
that was duplicated with ImageNet-1K/22K [38], Flickr30K
[116], and COCO [89] to ensure the reliability of our zero-
shot evaluations. Due to download failures and the use of
our data filtering pipeline, the total amount of data retained
in the first stage was 4.98 billion.
(2) Stage 2: In the second stage, we implemented a more
stringent data filtering strategy. With generative supervision
included, we deleted most of the low-quality data based on
the captions, mainly considering the length, completeness,
readability, and whether they were gibberish or boilerplate
(like menus, error messages, or duplicate text), contained
offensive language, placeholder text, or source code. We
retained only 1.03 billion entries.
Testing Datasets for Image Classification. We conducted
extensive validation on image classification tasks (see Fig-
ure 5 (b)), including the linear probing performance of
InternViT-6B and the zero-shot performance of InternVL-
C. These datasets used are listed in Table 24.
Testing Datasets for Video Classification. As shown in
Figure 5 (c), to evaluate the capabilities of video classifi-
cation, we utilize the following Kinetics datasets: Kinetics
400 [17], Kinetics 600 [18], and Kinetics 700 [19].
13

Testing Datasets for Image-Text Retrieval. We use five
datasets (see Figure 5 (d)) to evaluate InternVL’s zero-shot,
multilingual image-text retrieval capabilities. A detailed in-
troduction to these datasets is provided in Table 25.
Testing Dataset for Video-Text Retrieval. As shown in
Figure 5 (e), we use the MSR-VTT [161] dataset to evaluate
our InternVL in zero-shot video-text retrieval.
Testing Dataset for Image Captioning. As illustrated in
Figure 5 (f), we use three image captioning datasets to
test our InternVL model. A detailed introduction to these
datasets is provided in Table 26.
Testing Dataset for Semantic Segmentation. We use the
ADE20K [185] dataset to study the pixel-level perceptual
capacity of InternViT-6B, as shown in Figure 5 (g). A de-
tailed introduction to this dataset is provided in Table 26.
A.5. Data Preparation for SFT
Training Data for SFT. In this stage, we collect a wide
range of high-quality instruction data.
For non-dialogue
datasets, we follow the method described in [91] for con-
version. A detailed introduction is provided in Table 27.
Testing Datasets for SFT. We validate the effectiveness of
our supervised fine-tuned InternVL-Chat models on three
tasks, including image captioning, visual question answer-
ing, and multi-modal dialogue. There datasets are listed in
Table 28. For most of these datasets, we employ the same
response formatting prompt as for LLaVA-1.5 [91].
14

dataset
introduction
Training Data for Stage 1 & Stage 2.
LAION-en [120]
LAION-en is a part of the LAION-5B dataset, containing 2.32 billion English-only image-text pairs.
LAION-multi [120]
LAION-multi is another segment of LAION-5B, featuring 2.26 billion image-text pairs across more than
100 languages, and is ideal for multilingual studies.
Laion-COCO [121]
Laion-COCO comprises 663 million synthetic captions for web images, generated using a blend of BLIP-
L/14 [80] and CLIP models [117].
COYO [14]
COYO-700M is a large-scale dataset that contains 747 million image-text pairs as well as many other
meta-attributes to increase the usability to train various models. It follows a similar strategy to previous
vision-language datasets, collecting many informative pairs of alt-text and its associated image in HTML
documents.
Wukong [55]
Wukong is a large-scale Chinese image-text dataset for benchmarking different multi-modal pre-training
methods. It contains 100 million Chinese image-text pairs from the web.
CC3M [124]
This dataset consists of approximately 3 million images, each annotated with a caption.
CC12M [20]
CC12M is a dataset with 12 million image-text pairs. It is larger and covers a much more diverse set of
visual concepts than the CC3M [124].
SBU [112]
The SBU Captioned Photo Dataset is a collection of over 1 million images with associated text descriptions
extracted from Flicker.
Testing Datasets for Image Classification.
ImageNet-1K [38]
A large-scale dataset commonly used in image classification, consisting of over 1 million images across 1K
different classes.
ImageNet-ReaL [10]
It contains ImageNet val images augmented with a new set of “re-assessed” labels. These labels are col-
lected using an enhanced protocol, resulting in multi-label and more accurate annotations.
ImageNet-V2 [119]
A dataset created to test the robustness of models trained on ImageNet-1K, containing new test images
collected following the original methodology.
ImageNet-A [61]
It consists of real-world, unmodified, and naturally occurring examples that are misclassified by ResNet
models [57]. It’s designed to highlight the challenges of adversarial examples in natural settings.
ImageNet-R [60]
A set of images labeled with ImageNet labels obtained by collecting art, cartoons, deviantart, graffiti, em-
broidery, graphics, origami, paintings, patterns, plastic objects, plush objects, sculptures, sketches, tattoos,
toys, and video game renditions of ImageNet classes. It has renditions of 200 ImageNet classes resulting in
30K images.
ImageNet-Sketch [141]
It consists of 51K images, approximately 50 images for each of the ImageNet classes. It is constructed
using Google Image queries with the standard class name followed by “sketch of”.
ObjectNet [8]
ObjectNet is a crowd-sourced test set of 50K images featuring objects in unusual poses and cluttered scenes,
designed to challenge recognition performance. It includes controls for rotation, background, and view-
point, and covers 313 object classes, with 113 overlapping with ImageNet [38].
Multilingual IN-1K [76]
An adaptation of ImageNet-1K supporting multilingual annotations, facilitating research in cross-lingual
image classification.
CIFAR-10/100 [74]
It comprises 60K 32×32 images in 10 classes (CIFAR-10) or 100 classes (CIFAR-100).
MNIST [78]
A classic dataset containing 70K 28×28 gray-scale images of handwritten digits.
Caltech-101 [49]
The dataset comprises images of objects from 101 classes and a background clutter class, each labeled with
a single object. It contains about 40 to 800 images per class, totaling approximately 9K images.
SUN397 [157]
The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of 397 categories
with 109K images.
FGVC Aircraft [101]
The dataset contains 10K images of aircraft, with 100 images for each of 102 different aircraft model
variants, most of which are airplanes.
Country-211 [117]
It is a dataset released by OpenAI, designed to assess the geolocation capability of visual representations.
It filters the YFCC100M [136] dataset to find 211 countries that have at least 300 photos with GPS coordi-
nates. OpenAI built a balanced dataset with 211 categories, by sampling 200 photos for training and 100
photos for testing, for each country.
Stanford Cars [72]
This dataset consists of 196 classes of cars with a total of 16K images, taken from the rear. The data is
divided into almost a 50-50 train/test split with 8K training images and 8K testing images.
Table 24. Introduction of datasets used in InternVL’s stage 1 and stage 2. In summary, we utilize a vast amount of image-text data for
pre-training and conduct comprehensive evaluation across a wide range of generic visual-linguistic tasks.
15

dataset
introduction
Testing Datasets for Image Classification.
Birdsnap [9]
Birdsnap is a large bird dataset consisting of 49,829 images from 500 bird species with 47,386 images used
for training and 2,443 images used for testing. Due to broken links, we are only able to download 1,845 out
of the 2,443 testing images.
DTD [28]
The Describable Textures Dataset (DTD) contains 5,640 texture images in the wild. They are annotated
with human-centric attributes inspired by the perceptual properties of textures.
Eurosat [59]
This dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes
with 27K labeled and geo-referenced samples.
FER2013 [52]
This dataset includes around 30K RGB facial images, categorized into seven expressions: angry, disgust,
fear, happy, sad, surprise, and neutral.
Flowers-102 [109]
It is consistent with 102 flower categories commonly occurring in the United Kingdom. Each class consists
of between 40 and 258 images.
Food-101 [13]
The Food-101 dataset consists of 101 food categories with 750 training and 250 test images per category,
making a total of 101K images.
GTSRB [129]
The German Traffic Sign Recognition Benchmark (GTSRB) contains 43 classes of traffic signs, split into
39,209 training images and 12,630 test images.
Pets [113]
The Oxford-IIIT Pet Dataset is a 37-category pet dataset with roughly 200 images for each class created by
the Visual Geometry Group at Oxford.
Rendered SST2 [117]
This dataset is used to evaluate the model’s capability on optical character recognition. It was generated by
rendering sentences in the Standford Sentiment Treebank v2 dataset.
Resisc45 [30]
This is a dataset for remote sensing scene classification. It contains 31,500 RGB images divided into 45
scene classes, each class containing 700 images.
STL10 [109]
The STL-10 dataset, inspired by CIFAR-10 [74], includes 10 classes with 500 training and 800 test color
images each, sized 96×96 pixels.
VOC2007 [45]
The Pascal VOC 2007 dataset focuses on recognizing objects in realistic scenarios and contains 20 object
classes across 9,963 images with 24,640 labeled objects. The data has been divided into 50% for train-
ing/validation and 50% for testing. Following common practice, we conduct zero-shot image classification
by cropping images to isolate objects using bounding boxes.
Testing Datasets for Video Classification.
Kinetics 400 [17]
A large-scale dataset containing around 400 human action classes with at least 400 video clips for each
class, sourced from YouTube.
Kinetics 600 [18]
An expansion of Kinetics 400, this dataset includes 600 action classes and provides an increased diversity
in video representation.
Kinetics 700 [19]
The latest in the series, Kinetics 700 offers an even broader range with 700 action categories, further chal-
lenging the robustness of retrieval models.
Testing Datasets for Image-Text Retrieval.
COCO [22]
The COCO Caption dataset contains diverse images with detailed captions, widely used for image-text
retrieval and image captioning tasks.
COCO-CN [84]
COCO-CN is a bilingual image description dataset enriching COCO with manually written Chinese sen-
tences and tags. The new dataset can be used for multiple tasks including image tagging, captioning, and
retrieval, all in a cross-lingual setting.
Flickr30K [116]
This dataset comprises 31,000 images sourced from Flickr, each annotated with five captions, making it
suitable for image-text retrieval.
Flickr30K-CN [77]
Flickr30K-CN offers Chinese captions for the images, enabling studies in cross-lingual and multi-modal
retrieval tasks.
XTD [1]
A newly developed 1K multilingual test set, featuring COCO images annotated in various languages.
Testing Dataset for Video-Text Retrieval.
MSR-VTT [161]
This is a large-scale dataset for open-domain video captioning and video-text retrieval, comprising 10,000
video clips across 20 categories. Each clip is annotated with 20 English sentences, totaling about 29,000
distinct words in all captions. The standard division of the dataset allocates 6,513 clips for training, 497 for
validation, and 2,990 for testing purposes.
Table 25. Introduction of datasets used in InternVL’s stage 1 and stage 2. In summary, we utilize a vast amount of image-text data for
pre-training and conduct comprehensive evaluation across a wide range of generic visual-linguistic tasks.
16

dataset
introduction
Testing Datasets for Image Captioning.
COCO [22]
We use the Karpathy test set for testing.
Flickr30K [116]
We use the Karpathy test set for testing.
NoCaps [2]
NoCaps stands out for testing models’ capabilities in open-ended caption generation, using images that go
beyond the training data’s domain. We report the performance on the NoCaps val set.
Testing Dataset for Semantic Segmentation.
ADE20K [185]
ADE20K contains more than 20K scene-centric images exhaustively annotated with pixel-level objects and
object parts labels. There are a total of 150 semantic categories, which include stuffs like sky, road, grass,
and discrete objects like person, car, bed. We report the performance on the ADE20K val set.
Table 26. Introduction of datasets used in InternVL’s stage 1 and stage 2. In summary, we utilize a vast amount of image-text data for
pre-training and conduct comprehensive evaluation across a wide range of generic visual-linguistic tasks.
dataset
introduction
Training Data for SFT.
COCO Caption [22]
It contains over 0.5 million captions describing over 110K images. Following common practice, we use
the Karpathy training set for training. We transform it into a dialogue dataset using the response formatting
prompt: “Provide a one-sentence caption for the provided image.”
TextCaps [126]
TextCaps contains 145K captions for 28K images. It challenges a model to recognize text, relate it to its
visual context, and decide what part of the text to copy or paraphrase. OCR tokens are used during training.
We transform it into a dialogue dataset using the response formatting prompt: “Provide a one-sentence
caption for the provided image.”
VQAv2 [54]
VQAv2, the second version of the VQA dataset, features open-ended questions related to images. Answer-
ing these questions demands a grasp of vision, language, and common sense. We convert it into a dialogue
dataset using the prompt: “Answer the question using a single word or phrase.”
OKVQA [104]
A dataset with over 14K questions requiring external knowledge for answers, focusing on knowledge-based
visual question answering. We transform it into a dialogue dataset using the response formatting prompt:
“Answer the question using a single word or phrase.”
A-OKVQA [122]
An augmented successor of OKVQA [104] and contains 25K questions requiring a broad base of common-
sense and world knowledge to answer. We transform it into a dialogue dataset using the response formatting
prompt: “Answer with the option’s letter from the given choices directly.”
IconQA [99]
A dataset with 107K questions across three sub-tasks, focusing on abstract diagram recognition and com-
prehensive visual reasoning. We convert it into a dialogue dataset using these prompts: “Answer with the
option’s letter from the given choices directly.” and “Answer the question using a single word or phrase.”
AI2D [71]
AI2D features over 5K grade school science diagrams with rich annotations and 15K multiple-choice ques-
tions for diagram understanding research. We convert it into a dialogue dataset using the prompt: “Please
answer the question based on the options mentioned before.”
GQA [64]
GQA is a large-scale dataset with more than 110K images and 22 million questions, combining real images
with balanced question-answer pairs for visual reasoning. We transform it into a dialogue dataset using the
prompt: “Answer the question using a single word or phrase.”
OCR-VQA [107]
The OCR-VQA dataset contains 207,572 images of book covers and more than 1 million question-answer
pairs about these images. We convert it into a dialogue dataset using the response formatting prompt:
“Answer the question using a single word or phrase.”
ChartQA [105]
ChartQA is a dataset for question answering about charts, focusing on visual and logical reasoning. It com-
prises 9.6K human-written questions and 23.1K questions generated from human-written chart summaries.
We convert it using the prompt: “Answer the question using a single word or phrase.”
DocVQA [29]
The DocVQA dataset consists of 50,000 questions defined on over 12,000 document images. We convert it
into a dialogue dataset using the prompt: “Answer the question using a single word or phrase.”
ST-VQA [12]
The ST-VQA dataset contains a total of 31,791 questions over 23,038 images. The training set alone
consists of 26,308 questions based on 19,027 images. We convert it into a dialogue dataset using the
response formatting prompt: “Answer the question using a single word or phrase.”
Table 27. Introduction of datasets used in InternVL’s stage 3. We collect a wide range of high-quality instruction data. For non-dialogue
datasets, we follow the response formatting prompts described in [91] for conversion. Note that only the training set is used for training.
17

dataset
introduction
Training Data for SFT.
EST-VQA [150]
The EST-VQA dataset provides questions, images, and answers, but also a bounding box for each question
that indicates the area of the image that informs the answer. We convert it into a dialogue dataset using the
response formatting prompt: “Answer the question using a single word or phrase.”
InfoVQA [106]
This dataset includes a diverse collection of infographics with natural language questions and answers. It
focuses on reasoning over document layout, textual content, graphical elements, and data visualizations. We
convert it into a dialogue dataset using the prompt: “Answer the question using a single word or phrase.”
LLaVAR [182]
The LLaVAR dataset advances visual instruction tuning for Large Language Models by focusing on text-
rich images. It incorporates 422K images processed with OCR and 16K GPT-4 generated conversations,
enhancing text-based VQA performance and human interaction capabilities in diverse scenarios. Note that,
we only use the 20K high-quality data for fine-tuning of LLaVAR.
RefCOCO [103, 170]
A mixed dataset of RefCOCO [170], RefCOCO+[170], and RefCOCO-g [103]. We convert it into a dialogue
dataset following LLaVA-1.5 [91].
Toloka [140]
The TolokaVQA dataset comprises images with associated textual questions, each marked with a bounding
box indicating the visual answer. It’s sourced from a licensed subset of the COCO dataset and labeled on the
Toloka platform. We convert it into a dialogue dataset following LLaVA-1.5 [91].
LLaVA-150K [92]
This is a set of GPT-generated multi-modal instruction-following data, constructed for visual instruction
tuning and building large multi-modal models towards GPT-4 vision/language capability. It includes 158K
unique language-image instruction-following samples.
SVIT [183]
This dataset includes 3.2 million visual instruction tuning data, with 1.6M conversation QA pairs, 1.6M
complex reasoning QA pairs, and 106K detailed image descriptions. It is designed to improve multi-modal
performance in visual perception, reasoning, and planning. For this dataset, we merge the QA pairs from the
same training image into a single conversation.
VisDial [36]
A dataset based on the COCO images, featuring dialogues created by two Amazon Mechanical Turk workers.
One plays the ‘questioner’, seeing only an image’s text description, and the other, the ‘answerer’, sees the
image. They engage in a 10-round Q&A session about the image.
LRV-Instruction [90]
The LRV-Instruction dataset is designed to combat hallucination in large multi-modal models. It comprises
120K GPT-4-generated visual instructions for 16 vision-and-language tasks, including both positive and neg-
ative instructions for robust tuning. Negative instructions focus on Nonexistent and Existent Element Manip-
ulation. This dataset helps improve accuracy and consistency in multi-modal tasks.
LLaVA-Mix-665K [91]
LLaVA-Mix-665K is an instruction-following dataset mixed from 10 academically oriented datasets.
Testing Dataset for SFT (Image Captioning).
COCO [22]
Karpathy test set is used for testing. The prompt is: “Provide a one-sentence caption for the provided image.”
Flickr30K [116]
Karpathy test set is used for testing. The prompt is: “Provide a one-sentence caption for the provided image.”
NoCaps [2]
NoCaps val set is used for testing. The prompt is: “Provide a one-sentence caption for the provided image.”
Testing Dataset for SFT (Visual Question Answering).
VQAv2 [54]
VQAv2 test-dev set is used for testing. The prompt is: “Answer the question using a single word or phrase.”
GQA [64]
GQA test-balanced set is used. The prompt is: “Answer the question using a single word or phrase.”
VizWiz [56]
VizWiz test-dev set is used for testing. The prompt is: “When the provided information is insufficient,
respond with ‘Unanswerable’. Answer the question using a single word or phrase.”
TextVQA [127]
TextVQA val set is used for testing. The prompt is: “Answer the question using a single word or phrase.”
Testing Dataset for SFT (Multi-Modal Dialogue).
MME [50]
MME is a comprehensive evaluation benchmark for multi-modal large language models. It measures both
perception and cognition abilities on a total of 14 subtasks, including existence, count, position, color, poster,
celebrity, scene, landmark, artwork, OCR, commonsense reasoning, numerical calculation, text translation,
and code reasoning. The prompt for this dataset is: “Answer the question using a single word or phrase.”
POPE [86]
POPE is a popular dataset used to evaluate object hallucination. The response formatting prompt used for
this dataset is: “Answer the question using a single word or phrase.”
Table 28. Introduction of datasets used in InternVL’s stage 3. We collect a wide range of high-quality instruction data. For non-dialogue
datasets, we follow the response formatting prompts described in [91] for conversion. Note that only the training set is used for training.
We evaluate our InternVL-Chat models on three tasks, including image captioning, VQA, and multi-modal dialogue. For these datasets,
we employ the same response formatting prompts as for LLaVA-1.5 [91].
18

References
[1] Pranav Aggarwal and Ajinkya Kale.
Towards zero-
shot
cross-lingual
image
retrieval.
arXiv
preprint
arXiv:2012.05107, 2020. 8, 10, 16
[2] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen,
Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh,
Stefan Lee, and Peter Anderson. Nocaps: Novel object cap-
tioning at scale. In ICCV, pages 8948–8957, 2019. 8, 17,
18
[3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, An-
toine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur
Mensch, Katherine Millican, Malcolm Reynolds, et al.
Flamingo: a visual language model for few-shot learning.
NeurIPS, 35:23716–23736, 2022. 1, 3, 8
[4] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xi-
aodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang,
Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Day-
iheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin
Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi
Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei
Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang,
Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen
Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan
Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jin-
gren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen tech-
nical report. arXiv preprint arXiv:2309.16609, 2023. 3
[5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan
Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren
Zhou. Qwen-vl: A frontier large vision-language model
with versatile abilities. arXiv preprint arXiv:2308.12966,
2023. 1, 3, 8, 11
[6] Baichuan. Baichuan 2: Open large-scale language models.
arXiv preprint arXiv:2309.10305, 2023. 3
[7] Hangbo Bao, Li Dong, and Furu Wei.
Beit: Bert pre-
training of image transformers. In ICLR, 2022. 6, 11, 12
[8] Andrei Barbu, David Mayo, Julian Alverio, William Luo,
Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and
Boris Katz.
Objectnet:
A large-scale bias-controlled
dataset for pushing the limits of object recognition models.
NeurIPS, 32, 2019. 7, 15
[9] Thomas Berg, Jiongxin Liu, Seung Woo Lee, Michelle L
Alexander, David W Jacobs, and Peter N Belhumeur. Bird-
snap:
Large-scale fine-grained visual categorization of
birds. In CVPR, pages 2011–2018, 2014. 11, 16
[10] Lucas Beyer, Olivier J H´enaff, Alexander Kolesnikov, Xi-
aohua Zhai, and A¨aron van den Oord. Are we done with
imagenet? arXiv preprint arXiv:2006.07159, 2020. 6, 15
[11] Federico Bianchi, Giuseppe Attanasio, Raphael Pisoni, Sil-
via Terragni, Gabriele Sarti, and Sri Lakshmi. Contrastive
language-image pre-training for the italian language. arXiv
preprint arXiv:2108.08688, 2021. 7
[12] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez,
Marc¸al Rusinol, Ernest Valveny, CV Jawahar, and Dimos-
thenis Karatzas. Scene text visual question answering. In
ICCV, pages 4291–4301, 2019. 5, 17
[13] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool.
Food-101–mining discriminative components with random
forests. In ECCV, pages 446–461, 2014. 11, 16
[14] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun
Lee, Woonhyuk Baek, and Saehoon Kim.
Coyo-700m:
Image-text pair dataset, 2022. 5, 13, 15
[15] Yuxuan Cai, Yizhuang Zhou, Qi Han, Jianjian Sun, Xiang-
wen Kong, Jun Li, and Xiangyu Zhang. Reversible column
networks. arXiv preprint arXiv:2212.11696, 2022. 3
[16] Fredrik Carlsson, Philipp Eisen, Faton Rekathati, and Mag-
nus Sahlgren. Cross-lingual and multilingual clip. In Pro-
ceedings of the Thirteenth Language Resources and Evalu-
ation Conference, pages 6848–6854, 2022. 7, 8, 10
[17] Joao Carreira and Andrew Zisserman. Quo vadis, action
recognition?
a new model and the kinetics dataset.
In
CVPR, pages 6299–6308, 2017. 7, 8, 13, 16
[18] Joao Carreira, Eric Noland, Andras Banki-Horvath, Chloe
Hillier, and Andrew Zisserman. A short note about kinetics-
600. arXiv preprint arXiv:1808.01340, 2018. 7, 13, 16
[19] Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zis-
serman.
A short note on the kinetics-700 human action
dataset. arXiv preprint arXiv:1907.06987, 2019. 7, 8, 13,
16
[20] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu
Soricut. Conceptual 12m: Pushing web-scale image-text
pre-training to recognize long-tail visual concepts.
In
CVPR, pages 3558–3568, 2021. 5, 13, 15
[21] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang,
Feng Zhu, and Rui Zhao.
Shikra:
Unleashing multi-
modal llm’s referential dialogue magic.
arXiv preprint
arXiv:2306.15195, 2023. 1, 3, 8
[22] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna
Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence
Zitnick. Microsoft coco captions: Data collection and eval-
uation server. arXiv preprint arXiv:1504.00325, 2015. 5, 7,
8, 16, 17, 18
[23] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergio-
vanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman,
Adam Grycner, Basil Mustafa, Lucas Beyer, et al.
Pali:
A jointly-scaled multilingual language-image model.
In
ICLR, 2022. 1, 3, 4
[24] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa,
Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Se-
bastian Goodman, Xiao Wang, Yi Tay, et al. Pali-x: On
scaling up a multilingual vision and language model. arXiv
preprint arXiv:2305.18565, 2023. 8
[25] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong
Lu, Jifeng Dai, and Yu Qiao. Vision transformer adapter for
dense predictions. In ICLR, 2022. 3
[26] Zhongzhi Chen, Guang Liu, Bo-Wen Zhang, Fulong Ye,
Qinghong Yang, and Ledell Wu. Altclip: Altering the lan-
guage encoder in clip for extended language capabilities.
arXiv preprint arXiv:2211.06679, 2022. 7, 8, 10
[27] Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sens-
ing image scene classification: Benchmark and state of the
art. Proceedings of the IEEE, 105(10):1865–1883, 2017.
11
19

[28] Mircea Cimpoi,
Subhransu Maji,
Iasonas Kokkinos,
Sammy Mohamed, and Andrea Vedaldi. Describing tex-
tures in the wild. In CVPR, pages 3606–3613, 2014. 11,
16
[29] Christopher Clark and Matt Gardner. Simple and effective
multi-paragraph reading comprehension.
arXiv preprint
arXiv:1710.10723, 2017. 5, 17
[30] Adam Coates, Andrew Ng, and Honglak Lee. An analysis
of single-layer networks in unsupervised feature learning.
In AISTAT, pages 215–223, 2011. 11, 16
[31] MMSegmentation Contributors. Mmsegmentation: Open-
mmlab semantic segmentation toolbox and benchmark,
2020. 13
[32] Yiming Cui, Ziqing Yang, and Xin Yao. Efficient and ef-
fective text encoding for chinese llama and alpaca. arXiv
preprint arXiv:2304.08177, 2023. 2, 3, 4, 5, 6, 11, 12
[33] Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong
Zhang, Han Hu, and Yichen Wei.
Deformable convolu-
tional networks. In ICCV, pages 764–773, 2017. 3
[34] Wenliang Dai, Junnan Li, Dongxu Li, AnthonyMeng Huat,
Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and
Steven Hoi. Instructblip: Towards general-purpose vision-
language models with instruction tuning.
arXiv preprint
arXiv:2305.06500, 2023. 1, 8, 11
[35] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christo-
pher R´e. Flashattention: Fast and memory-efficient exact
attention with io-awareness.
NeurIPS, 35:16344–16359,
2022. 9, 11
[36] Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh,
Deshraj Yadav, Jos´e MF Moura, Devi Parikh, and Dhruv
Batra. Visual dialog. In CVPR, pages 326–335, 2017. 5, 18
[37] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr
Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter
Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdul-
mohsin, et al. Scaling vision transformers to 22 billion pa-
rameters. In ICML, pages 7480–7512, 2023. 3, 4, 6, 7,
12
[38] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical im-
age database. In CVPR, pages 248–255, 2009. 2, 3, 6, 7, 9,
10, 11, 12, 13, 15
[39] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova.
Bert:
Pre-training of deep bidirectional
transformers for language understanding.
arXiv preprint
arXiv:1810.04805, 2018. 2, 3
[40] Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong
Han, Guiguang Ding, and Jian Sun. Repvgg: Making vgg-
style convnets great again. In CVPR, pages 13733–13742,
2021. 3
[41] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng
Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou,
Haoran Wei, et al. Dreamllm: Synergistic multimodal com-
prehension and creation. arXiv preprint arXiv:2309.11499,
2023. 8
[42] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold,
Sylvain Gelly, et al. An image is worth 16x16 words: Trans-
formers for image recognition at scale. In ICLR, 2020. 3,
4
[43] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch,
Aakanksha Chowdhery,
Brian Ichter,
Ayzaan Wahid,
Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e:
An embodied multimodal language model. arXiv preprint
arXiv:2303.03378, 2023. 3
[44] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong
Qiu, Zhilin Yang, and Jie Tang. Glm: General language
model pretraining with autoregressive blank infilling.
In
ACL, pages 320–335, 2022. 3, 11
[45] Mark Everingham, SM Ali Eslami, Luc Van Gool, Christo-
pher KI Williams, John Winn, and Andrew Zisserman.
The pascal visual object classes challenge: A retrospective.
IJCV, 111:98–136, 2015. 11, 16
[46] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu,
Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue
Cao. Eva: Exploring the limits of masked visual represen-
tation learning at scale. arXiv preprint arXiv:2211.07636,
2022. 3, 6
[47] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang,
Xinlong Wang, and Yue Cao. Eva-02: A visual represen-
tation for neon genesis. arXiv preprint arXiv:2303.11331,
2023. 10
[48] William Fedus, Barret Zoph, and Noam Shazeer. Switch
transformers: Scaling to trillion parameter models with
simple and efficient sparsity.
JMLR, 23(1):5232–5270,
2022. 1
[49] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning gener-
ative visual models from few training examples: An incre-
mental bayesian approach tested on 101 object categories.
In CVPRW, pages 178–178, 2004. 11, 15
[50] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin,
Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui
Yang, Xiawu Zheng, et al. Mme: A comprehensive eval-
uation benchmark for multimodal large language models.
arXiv preprint arXiv:2306.13394, 2023. 8, 18
[51] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie
Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xi-
angyu Yue, et al.
Llama-adapter v2: Parameter-efficient
visual instruction model. arXiv preprint arXiv:2304.15010,
2023. 11
[52] Ian J Goodfellow, Dumitru Erhan, Pierre Luc Carrier,
Aaron Courville, Mehdi Mirza, Ben Hamner, Will Cukier-
ski, Yichuan Tang, David Thaler, Dong-Hyun Lee, et al.
Challenges in representation learning: A report on three
machine learning contests.
In ICONIP, pages 117–124,
2013. 11, 16
[53] Google. Google bard. https://bard.google.com/,
2023. 11
[54] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv
Batra, and Devi Parikh. Making the v in vqa matter: El-
evating the role of image understanding in visual question
answering. In CVPR, pages 6904–6913, 2017. 5, 8, 17, 18
[55] Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Niu
Minzhe, Xiaodan Liang, Lewei Yao, Runhui Huang, Wei
20

Zhang, Xin Jiang, et al. Wukong: A 100 million large-scale
chinese cross-modal pre-training benchmark. NeurIPS, 35:
26418–26431, 2022. 5, 7, 10, 13, 15
[56] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi
Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham.
Vizwiz grand challenge: Answering visual questions from
blind people. In CVPR, pages 3608–3617, 2018. 8, 18
[57] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In CVPR,
pages 770–778, 2016. 1, 3, 15
[58] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr
Doll´ar, and Ross Girshick. Masked autoencoders are scal-
able vision learners. In CVPR, pages 16000–16009, 2022.
6, 12
[59] Patrick Helber, Benjamin Bischke, Andreas Dengel, and
Damian Borth. Eurosat: A novel dataset and deep learn-
ing benchmark for land use and land cover classification.
IEEE Journal of Selected Topics in Applied Earth Obser-
vations and Remote Sensing, 12(7):2217–2226, 2019. 11,
16
[60] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kada-
vath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu,
Samyak Parajuli, Mike Guo, et al. The many faces of ro-
bustness: A critical analysis of out-of-distribution general-
ization. In ICCV, pages 8340–8349, 2021. 6, 7, 15
[61] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Stein-
hardt, and Dawn Song. Natural adversarial examples. In
CVPR, pages 15262–15271, 2021. 6, 7, 15
[62] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation
networks. In CVPR, pages 7132–7141, 2018. 3
[63] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kil-
ian Q Weinberger. Deep networks with stochastic depth. In
ECCV, pages 646–661, 2016. 12, 13
[64] Drew A Hudson and Christopher D Manning. Gqa: A new
dataset for real-world visual reasoning and compositional
question answering. In CVPR, pages 6700–6709, 2019. 5,
8, 17, 18
[65] Forrest Iandola, Matt Moskewicz, Sergey Karayev, Ross
Girshick, Trevor Darrell, and Kurt Keutzer. Densenet: Im-
plementing efficient convnet descriptor pyramids.
arXiv
preprint arXiv:1404.1869, 2014. 3
[66] IDEFICS.
Introducing idefics:
An open reproduction
of state-of-the-art visual language model.
https://
huggingface.co/blog/idefics, 2023. 8
[67] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade
Gordon, Nicholas Carlini, Rohan Taori, Achal Dave,
Vaishaal Shankar, Hongseok Namkoong, John Miller, Han-
naneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Open-
clip.
Zenodo. Version 0.1. https://doi.org/10.
5281/zenodo.5143773, 2021.
DOI: 10.5281/zen-
odo.5143773. 3, 6, 7, 8, 10, 11
[68] Sergey Ioffe and Christian Szegedy. Batch normalization:
Accelerating deep network training by reducing internal co-
variate shift. In ICML, pages 448–456, 2015. 12
[69] Aashi Jain, Mandy Guo, Krishna Srinivasan, Ting Chen,
Sneha Kudugunta, Chao Jia, Yinfei Yang, and Jason
Baldridge. Mural: multimodal, multitask retrieval across
languages. arXiv preprint arXiv:2109.05125, 2021. 10
[70] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana
Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li,
and Tom Duerig.
Scaling up visual and vision-language
representation learning with noisy text supervision.
In
ICML, pages 4904–4916, 2021. 2, 3, 10
[71] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon
Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is
worth a dozen images. In ECCV, pages 235–251, 2016. 5,
17
[72] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei.
3d object representations for fine-grained categorization. In
ICCVW, pages 554–561, 2013. 11, 15
[73] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton.
Imagenet classification with deep convolutional neural net-
works. NeurIPS, 25, 2012. 3
[74] Alex Krizhevsky et al. Learning multiple layers of features
from tiny images. 2009. 11, 15, 16
[75] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui
Yuan, Shu Liu, and Jiaya Jia.
Lisa:
Reasoning seg-
mentation via large language model.
arXiv preprint
arXiv:2308.00692, 2023. 3
[76] LAION-AI.
Clip benchmark:
Clip-like model evalua-
tion.
https://github.com/LAION-AI/CLIP_
benchmark, 2023. 7, 15
[77] Weiyu Lan, Xirong Li, and Jianfeng Dong. Fluency-guided
cross-lingual image captioning. In ACM MM, pages 1549–
1557, 2017. 7, 8, 10, 12, 16
[78] Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick
Haffner.
Gradient-based learning applied to document
recognition. Proceedings of the IEEE, 86(11):2278–2324,
1998. 11, 15
[79] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang,
Jingkang Yang, and Ziwei Liu.
Otter: A multi-modal
model with in-context instruction tuning.
arXiv preprint
arXiv:2305.03726, 2023. 3, 11
[80] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi.
Blip: Bootstrapping language-image pre-training for uni-
fied vision-language understanding and generation.
In
ICML, pages 12888–12900, 2022. 3, 5, 10, 15
[81] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.
Blip-2: Bootstrapping language-image pre-training with
frozen image encoders and large language models. arXiv
preprint arXiv:2301.12597, 2023. 1, 4, 6, 7, 8, 11, 12
[82] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wen-
hai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu
Qiao. Videochat: Chat-centric video understanding. arXiv
preprint arXiv:2305.06355, 2023. 3
[83] Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He,
Limin Wang, and Yu Qiao. Unmasked teacher: Towards
training-efficient video foundation models. arXiv preprint
arXiv:2303.16058, 2023. 10
[84] Xirong Li, Chaoxi Xu, Xiaoxu Wang, Weiyu Lan, Zhengx-
iong Jia, Gang Yang, and Jieping Xu. Coco-cn for cross-
lingual image tagging, captioning, and retrieval. TMM, 21
(9):2347–2360, 2019. 7, 8, 16
[85] Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Man-
galam, Bo Xiong, Jitendra Malik, and Christoph Feichten-
21

hofer. Improved multiscale vision transformers for classi-
fication and detection. arXiv preprint arXiv:2112.01526,
2021. 9
[86] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin
Zhao, and Ji-Rong Wen.
Evaluating object hallucina-
tion in large vision-language models.
arXiv preprint
arXiv:2305.10355, 2023. 8, 18
[87] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feicht-
enhofer, and Kaiming He.
Scaling language-image pre-
training via masking. In CVPR, pages 23390–23400, 2023.
12
[88] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang,
Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai.
Monkey: Image resolution and text label are important
things for large multi-modal models.
arXiv preprint
arXiv:2311.06607, 2023. 3
[89] Tsung-Yi Lin, Michael Maire, Serge Belongie, James
Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and
C Lawrence Zitnick. Microsoft coco: Common objects in
context. In ECCV, pages 740–755, 2014. 13
[90] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser
Yacoob, and Lijuan Wang.
Aligning large multi-modal
model with robust instruction tuning.
arXiv preprint
arXiv:2306.14565, 2023. 5, 18
[91] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee.
Improved baselines with visual instruction tuning. arXiv
preprint arXiv:2310.03744, 2023. 3, 5, 6, 8, 9, 11, 12, 14,
17, 18
[92] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee. Visual instruction tuning. NeurIPS, 2023. 1, 3, 4, 5,
11, 12, 18
[93] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
dar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke
Zettlemoyer, and Veselin Stoyanov.
Roberta:
A ro-
bustly optimized bert pretraining approach. arXiv preprint
arXiv:1907.11692, 2019. 2, 3
[94] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng
Zhang, Stephen Lin, and Baining Guo. Swin transformer:
Hierarchical vision transformer using shifted windows. In
ICCV, pages 10012–10022, 2021. 3
[95] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feicht-
enhofer, Trevor Darrell, and Saining Xie. A convnet for the
2020s. arXiv preprint arXiv:2201.03545, 2022. 3
[96] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi
Wang, Shoufa Chen, Qinglong Zhang, Zeqiang Lai, Yang
Yang, Qingyun Li, Jiashuo Yu, et al. Interngpt: Solving
vision-centric tasks by interacting with chatgpt beyond lan-
guage. arXiv preprint arXiv:2305.05662, 2023. 3
[97] Zhaoyang Liu, Zeqiang Lai, Zhangwei Gao, Erfei Cui,
Xizhou Zhu, Lewei Lu, Qifeng Chen, Yu Qiao, Jifeng
Dai, and Wenhai Wang. Controlllm: Augment language
models with tools by searching on graphs. arXiv preprint
arXiv:2310.17796, 2023. 3
[98] Ilya Loshchilov and Frank Hutter. Decoupled weight decay
regularization. arXiv preprint arXiv:1711.05101, 2017. 11,
12, 13
[99] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao,
Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu.
Iconqa: A new benchmark for abstract diagram under-
standing and visual language reasoning.
arXiv preprint
arXiv:2110.13214, 2021. 5, 17
[100] Yadong Lu, Chunyuan Li, Haotian Liu, Jianwei Yang, Jian-
feng Gao, and Yelong Shen. An empirical study of scal-
ing instruct-tuned large multimodal models. arXiv preprint
arXiv:2309.09958, 2023. 3
[101] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew
Blaschko, and Andrea Vedaldi. Fine-grained visual classi-
fication of aircraft. arXiv preprint arXiv:1306.5151, 2013.
11, 15
[102] Kei Sawada Makoto Shiin, Tianyu Zhao.
Construction
and public release of language image pretraining models in
japanese. In The 25th Meeting on Image Recognition and
Understanding, 2022. 7
[103] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana
Camburu, Alan L Yuille, and Kevin Murphy. Generation
and comprehension of unambiguous object descriptions. In
CVPR, pages 11–20, 2016. 5, 18
[104] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and
Roozbeh Mottaghi. Ok-vqa: A visual question answering
benchmark requiring external knowledge. In CVPR, pages
3195–3204, 2019. 5, 17
[105] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty,
and Enamul Hoque. Chartqa: A benchmark for question
answering about charts with visual and logical reasoning.
arXiv preprint arXiv:2203.10244, 2022. 5, 17
[106] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis
Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa.
In WACV, pages 1697–1706, 2022. 5, 18
[107] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and
Anirban Chakraborty. Ocr-vqa: Visual question answering
by reading text in images. In ICDAR, pages 947–952. IEEE,
2019. 5, 17
[108] Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang,
Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao,
and Ping Luo.
Embodiedgpt:
Vision-language pre-
training via embodied chain of thought.
arXiv preprint
arXiv:2305.15021, 2023. 3
[109] Maria-Elena Nilsback and Andrew Zisserman. Automated
flower classification over a large number of classes.
In
ICVGIP, pages 722–729, 2008. 11, 16
[110] OpenAI. Gpt-4 technical report, 2023. 3, 8
[111] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.
Dinov2: Learning robust visual features without supervi-
sion. arXiv preprint arXiv:2304.07193, 2023. 6, 12
[112] Vicente Ordonez, Girish Kulkarni, and Tamara Berg.
Im2text: Describing images using 1 million captioned pho-
tographs. In NeurIPS, 2011. 5, 13, 15
[113] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and
CV Jawahar. Cats and dogs. In CVPR, pages 3498–3505,
2012. 11, 16
[114] Guilherme Penedo, Quentin Malartic, Daniel Hesslow,
Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobei-
dli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Lau-
22

nay. The RefinedWeb dataset for Falcon LLM: outperform-
ing curated corpora with web data, and web data only. arXiv
preprint arXiv:2306.01116, 2023. 3
[115] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan
Huang, Shuming Ma, and Furu Wei. Kosmos-2: Ground-
ing multimodal large language models to the world. arXiv
preprint arXiv:2306.14824, 2023. 1, 3, 8
[116] Bryan A Plummer, Liwei Wang, Chris M Cervantes,
Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazeb-
nik. Flickr30k entities: Collecting region-to-phrase corre-
spondences for richer image-to-sentence models. In ICCV,
pages 2641–2649, 2015. 7, 8, 10, 12, 13, 16, 17, 18
[117] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn-
ing transferable visual models from natural language super-
vision. In ICML, pages 8748–8763, 2021. 1, 3, 6, 7, 8, 9,
10, 11, 15, 16
[118] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and
Yuxiong He.
Deepspeed: System optimizations enable
training deep learning models with over 100 billion param-
eters. In SIGKDD, pages 3505–3506, 2020. 12, 13
[119] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and
Vaishaal Shankar. Do imagenet classifiers generalize to im-
agenet? In ICML, pages 5389–5400, 2019. 6, 7, 15
[120] Christoph Schuhmann, Romain Beaumont, Richard Vencu,
Cade Gordon,
Ross Wightman,
Mehdi Cherti,
Theo
Coombes, Aarush Katta, Clayton Mullis, Mitchell Worts-
man, et al.
Laion-5b: An open large-scale dataset for
training next generation image-text models. NeurIPS, 35:
25278–25294, 2022. 4, 5, 9, 13, 15
[121] Christoph Schuhmann, Andreas K¨opf, Richard Vencu,
Theo
Coombes,
and
Romain
Beaumont.
Laion
coco:
600m
synthetic
captions
from
laion2b-en.
https://laion.ai/blog/laion-coco/, 2022. 5, 13, 15
[122] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark,
Kenneth Marino, and Roozbeh Mottaghi.
A-okvqa: A
benchmark for visual question answering using world
knowledge. In ECCV, pages 146–162, 2022. 5, 17
[123] Wenqi Shao, Yutao Hu, Peng Gao, Meng Lei, Kaipeng
Zhang, Fanqing Meng, Peng Xu, Siyuan Huang, Hong-
sheng Li, Yu Qiao, et al. Tiny lvlm-ehub: Early multimodal
experiments with bard. arXiv preprint arXiv:2308.03729,
2023. 10
[124] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu
Soricut. Conceptual captions: A cleaned, hypernymed, im-
age alt-text dataset for automatic image captioning. In ACL,
2018. 5, 13, 15
[125] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li,
Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving
ai tasks with chatgpt and its friends in huggingface. arXiv
preprint arXiv:2303.17580, 2023. 3
[126] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and
Amanpreet Singh. Textcaps: a dataset for image caption-
ing with reading comprehension. In ECCV, pages 742–758,
2020. 5, 17
[127] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang,
Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus
Rohrbach. Towards vqa models that can read. In CVPR,
pages 8317–8326, 2019. 8, 18
[128] Mannat Singh, Quentin Duval, Kalyan Vasudev Alwala,
Haoqi Fan, Vaibhav Aggarwal, Aaron Adcock, Armand
Joulin, Piotr Doll´ar, Christoph Feichtenhofer, Ross Gir-
shick, et al. The effectiveness of mae pre-pretraining for
billion-scale pretraining. arXiv preprint arXiv:2303.13496,
2023. 4, 6, 7
[129] Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and
Christian Igel. Man vs. computer: Benchmarking machine
learning algorithms for traffic sign recognition. Neural net-
works, 32:323–332, 2012. 11, 16
[130] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue
Cao.
Eva-clip: Improved training techniques for clip at
scale. arXiv preprint arXiv:2303.15389, 2023. 3, 4, 7, 8, 9,
11
[131] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong
Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun
Huang, and Xinlong Wang. Generative pretraining in mul-
timodality. arXiv preprint arXiv:2307.05222, 2023. 1, 3,
8
[132] Tianxiang Sun, Xiaotian Zhang, Zhengfu He, Peng Li,
Qinyuan Cheng, Hang Yan, Xiangyang Liu, Yunfan Shao,
Qiong Tang, Xingjian Zhao, Ke Chen, Yining Zheng, Zhe-
jian Zhou, Ruixiao Li, Jun Zhan, Yunhua Zhou, Linyang
Li, Xiaogui Yang, Lingling Wu, Zhangyue Yin, Xuanjing
Huang, and Xipeng Qiu. Moss: Training conversational
language models from synthetic data. 2023. 3
[133] D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. Vipergpt:
Visual inference via python execution for reasoning. arXiv
preprint arXiv:2303.08128, 2023. 3
[134] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois,
Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B
Hashimoto.
Alpaca:
A strong, replicable instruction-
following model. Stanford Center for Research on Founda-
tion Models. https://crfm. stanford. edu/2023/03/13/alpaca.
html, 3(6):7, 2023. 3
[135] InternLM Team. Internlm: A multilingual language model
with progressively enhanced capabilities.
https://
github.com/InternLM/InternLM, 2023. 2, 3, 6,
11
[136] Bart Thomee, David A Shamma, Gerald Friedland, Ben-
jamin Elizalde, Karl Ni, Douglas Poland, Damian Borth,
and Li-Jia Li. Yfcc100m: The new data in multimedia re-
search. Communications of the ACM, 59(2):64–73, 2016.
15
[137] Hugo Touvron, Matthieu Cord, and Herv´e J´egou. Deit iii:
Revenge of the vit. In ECCV, pages 516–533, 2022. 6
[138] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Bap-
tiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar,
et al. Llama: Open and efficient foundation language mod-
els. arXiv preprint arXiv:2302.13971, 2023. 2, 3
[139] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert,
Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov,
Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.
Llama 2: Open foundation and fine-tuned chat models.
arXiv preprint arXiv:2307.09288, 2023. 2, 3
23

[140] Dmitry Ustalov, Nikita Pavlichenko, Sergey Koshelev,
Daniil Likhobaba, and Alisa Smirnova.
Toloka vi-
sual question answering benchmark.
arXiv preprint
arXiv:2309.16511, 2023. 5, 18
[141] Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P
Xing. Learning robust global representations by penalizing
local predictive power. NeurIPS, 32, 2019. 6, 7, 15
[142] Junke Wang, Dongdong Chen, Zuxuan Wu, Chong Luo,
Luowei Zhou, Yucheng Zhao, Yujia Xie, Ce Liu, Yu-Gang
Jiang, and Lu Yuan. Omnivl: One foundation model for
image-language and video-language tasks.
NeurIPS, 35:
5696–5710, 2022. 10
[143] Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiao-
huan Zhou, Jingren Zhou, Xinggang Wang, and Chang
Zhou.
One-peace:
Exploring one general representa-
tion model toward unlimited modalities.
arXiv preprint
arXiv:2305.11172, 2023. 7, 10
[144] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao
Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao.
Pyramid vision transformer: A versatile backbone for dense
prediction without convolutions. In ICCV, pages 568–578,
2021. 3
[145] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao
Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao.
Pvtv2:
Improved baselines with pyramid vision trans-
former. CVMJ, pages 1–10, 2022. 3
[146] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhil-
iang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mo-
hammed, Saksham Singhal, Subhojit Som, et al. Image as
a foreign language: Beit pretraining for vision and vision-
language tasks. In CVPR, pages 19175–19186, 2023. 10
[147] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu,
Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou,
Yu Qiao, et al. Visionllm: Large language model is also
an open-ended decoder for vision-centric tasks. NeurIPS,
2023. 1, 3
[148] Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang,
Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu,
Hongsheng Li, et al. Internimage: Exploring large-scale vi-
sion foundation models with deformable convolutions. In
CVPR, pages 14408–14419, 2023. 3
[149] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhen-
hang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu,
Zhiguo Cao, et al. The all-seeing project: Towards panop-
tic visual recognition and understanding of the open world.
arXiv preprint arXiv:2308.01907, 2023. 3, 8
[150] Xinyu Wang, Yuliang Liu, Chunhua Shen, Chun Chet Ng,
Canjie Luo, Lianwen Jin, Chee Seng Chan, Anton van den
Hengel, and Liangwei Wang. On the general value of evi-
dence, and bilingual scene-text visual question answering.
In CVPR, pages 10126–10135, 2020. 5, 18
[151] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun
Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun
Wang, et al. Internvideo: General video foundation models
via generative and discriminative learning. arXiv preprint
arXiv:2212.03191, 2022. 10
[152] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu,
Xin Ma, Xinyuan Chen, Yaohui Wang, Ping Luo, Ziwei
Liu, et al. Internvid: A large-scale video-text dataset for
multimodal understanding and generation. arXiv preprint
arXiv:2307.06942, 2023. 7, 8
[153] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.
Chain-of-thought prompting elicits reasoning in large lan-
guage models. NeurIPS, 35:24824–24837, 2022. 3
[154] Tianwen Wei, Liang Zhao, Lichang Zhang, Bo Zhu, Lijie
Wang, Haihua Yang, Biye Li, Cheng Cheng, Weiwei L¨u,
Rui Hu, et al. Skywork: A more open bilingual foundation
model. arXiv preprint arXiv:2310.19341, 2023. 3
[155] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong
Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talk-
ing, drawing and editing with visual foundation models.
arXiv preprint arXiv:2303.04671, 2023. 3
[156] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-
Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv
preprint arXiv:2309.05519, 2023. 3
[157] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva,
and Antonio Torralba.
Sun database: Large-scale scene
recognition from abbey to zoo. In CVPRW, pages 3485–
3492, 2010. 11, 15
[158] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and
Jian Sun. Unified perceptual parsing for scene understand-
ing. In ECCV, pages 418–434, 2018. 7
[159] Chunyu Xie, Jincheng Li, Heng Cai, Fanjing Kong, Xiaoyu
Wu, Jianfei Song, Henrique Morimitsu, Lin Yao, Dexin
Wang, Dawei Leng, et al. Zero and r2d2: A large-scale chi-
nese cross-modal benchmark and a vision-language frame-
work. arXiv preprint arXiv:2205.03860, 2022. 7, 10
[160] Saining Xie, Ross Girshick, Piotr Doll´ar, Zhuowen Tu, and
Kaiming He. Aggregated residual transformations for deep
neural networks. In CVPR, pages 1492–1500, 2017. 3
[161] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large
video description dataset for bridging video and language.
In CVPR, pages 5288–5296, 2016. 10, 14, 16
[162] An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang
Zhang, Jingren Zhou, and Chang Zhou.
Chinese clip:
Contrastive vision-language pretraining in chinese. arXiv
preprint arXiv:2211.01335, 2022. 7, 8, 10
[163] Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge,
Xiu Li, and Ying Shan.
Gpt4tools: Teaching large lan-
guage model to use tools via self-instruction. arXiv preprint
arXiv:2305.18752, 2023. 3
[164] Yinfei Yang, Daniel Cer, Amin Ahmad, Mandy Guo, Jax
Law, Noah Constant, Gustavo Hernandez Abrego, Steve
Yuan, Chris Tar, Yun-Hsuan Sung, et al. Multilingual uni-
versal sentence encoder for semantic retrieval.
In ACL,
pages 87–94, 2020. 10
[165] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang,
Chung-Ching Lin, Zicheng Liu, and Lijuan Wang.
The
dawn of lmms:
Preliminary explorations with gpt-4v
(ision). arXiv preprint arXiv:2309.17421, 9, 2023. 3
[166] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin,
Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu,
Michael Zeng, and Lijuan Wang.
Mm-react: Prompting
chatgpt for multimodal reasoning and action. arXiv preprint
arXiv:2303.11381, 2023. 3
24

[167] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe
Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang,
and Chunjing Xu. Filip: Fine-grained interactive language-
image pre-training. In ICLR, 2021. 10
[168] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan,
Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Jun-
feng Tian, Qian Qi, Ji Zhang, and Fei Huang.
mplug-
docowl: Modularized multimodal large language model for
document understanding, 2023. 3
[169] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung,
Mojtaba Seyedhosseini, and Yonghui Wu.
Coca: Con-
trastive captioners are image-text foundation models. arXiv
preprint arXiv:2205.01917, 2022. 7
[170] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg,
and Tamara L Berg. Modeling context in referring expres-
sions. In ECCV, pages 69–85, 2016. 5, 18
[171] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella,
Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang,
Boxin Li, Chunyuan Li, et al.
Florence:
A new
foundation model for computer vision.
arXiv preprint
arXiv:2111.11432, 2021. 7, 10
[172] Yan Zeng, Hanbo Zhang, Jiani Zheng, Jiangnan Xia, Guo-
qiang Wei, Yang Wei, Yuchen Zhang, and Tao Kong. What
matters in training a gpt4-style language model with mul-
timodal inputs?
arXiv preprint arXiv:2307.02469, 2023.
11
[173] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and
Lucas Beyer. Scaling vision transformers. In CVPR, pages
12104–12113, 2022. 3, 4, 6, 7
[174] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner,
Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer.
Lit: Zero-shot transfer with locked-image text tuning. In
CVPR, pages 18123–18133, 2022. 7
[175] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An
instruction-tuned audio-visual language model for video
understanding. arXiv preprint arXiv:2306.02858, 2023. 3
[176] Jiaxing Zhang, Ruyi Gan, Junjie Wang, Yuxiang Zhang,
Lin Zhang, Ping Yang, Xinyu Gao, Ziwei Wu, Xiaoqun
Dong, Junqing He, et al.
Fengshenbang 1.0: Being the
foundation of chinese cognitive intelligence. arXiv preprint
arXiv:2209.02970, 2022. 7
[177] Pan Zhang, Xiaoyi Dong Bin Wang, Yuhang Cao, Chao Xu,
Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang
Zhang, Haodong Duan, Hang Yan, et al.
Internlm-
xcomposer: A vision-language large model for advanced
text-image comprehension and composition. arXiv preprint
arXiv:2309.15112, 2023. 1, 3, 11
[178] Qinglong Zhang and Yu-Bin Yang. Rest: An efficient trans-
former for visual recognition. NeurIPS, 34:15475–15485,
2021. 3
[179] Qinglong Zhang and Yu-Bin Yang. Rest v2: simpler, faster
and stronger. NeurIPS, 35:36440–36452, 2022. 3
[180] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu,
Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao.
Llama-adapter: Efficient fine-tuning of language models
with zero-init attention. arXiv preprint arXiv:2303.16199,
2023. 3
[181] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi
Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi:
Instruction tuning large language model on region-of-
interest. arXiv preprint arXiv:2307.03601, 2023. 3
[182] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou,
Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced
visual instruction tuning for text-rich image understanding.
arXiv preprint arXiv:2306.17107, 2023. 5, 18
[183] Bo Zhao, Boya Wu, and Tiejun Huang. Svit: Scaling up
visual instruction tuning. arXiv preprint arXiv:2307.04087,
2023. 5, 18
[184] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuo-
han Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-
judge with mt-bench and chatbot arena.
arXiv preprint
arXiv:2306.05685, 2023. 2, 3, 6, 8, 9, 10, 11
[185] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela
Barriuso, and Antonio Torralba.
Scene parsing through
ade20k dataset. In CVPR, pages 633–641, 2017. 6, 14,
17
[186] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui,
HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang,
Zongwei Li, et al.
Languagebind:
Extending video-
language pretraining to n-modality by language-based se-
mantic alignment. arXiv preprint arXiv:2310.01852, 2023.
10
[187] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mo-
hamed Elhoseiny. Minigpt-4: Enhancing vision-language
understanding with advanced large language models. arXiv
preprint arXiv:2304.10592, 2023. 1, 3, 11
[188] Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Wei-
jie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu, Xiao-
gang Wang, et al. Ghost in the minecraft: Generally capable
agents for open-world enviroments via large language mod-
els with text-based knowledge and memory. arXiv preprint
arXiv:2305.17144, 2023. 3
25
