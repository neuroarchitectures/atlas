
# Uni-Perceiver: Pre-training Unified Architecture for Generic Perception for Zero-shot and Few-shot Tasks

Uni-Perceiver: Pre-training Uniﬁed Architecture for Generic Perception for
Zero-shot and Few-shot Tasks
Xizhou Zhu1∗, Jinguo Zhu2∗†, Hao Li3∗†, Xiaoshi Wu3∗†
Xiaogang Wang3, Hongsheng Li3, Xiaohua Wang2, Jifeng Dai1B
1SenseTime Research
2Xi’an Jiaotong University
3CUHK-SenseTime Joint Laboratory, The Chinese University of Hong Kong
{zhuwalter, daijifeng}@sensetime.com
lechatelia@stu.xjtu.edu.cn, {haoli, wuxiaoshi}@link.cuhk.edu.hk
{xgwang, hsli}@ee.cuhk.edu.hk, xhw@mail.xjtu.edu.cn
Abstract
Biological intelligence systems of animals perceive the
world by integrating information in different modalities and
processing simultaneously for various tasks. In contrast,
current machine learning research follows a task-speciﬁc
paradigm, leading to inefﬁcient collaboration between tasks
and high marginal costs of developing perception models
for new tasks. In this paper, we present a generic perception
architecture named Uni-Perceiver, which processes a vari-
ety of modalities and tasks with uniﬁed modeling and shared
parameters. Speciﬁcally, Uni-Perceiver encodes different
task inputs and targets from arbitrary modalities into a uni-
ﬁed representation space with a modality-agnostic Trans-
former encoder and lightweight modality-speciﬁc tokeniz-
ers. Different perception tasks are modeled as the same
formulation, that is, ﬁnding the maximum likelihood target
for each input through the similarity of their representa-
tions. The model is pre-trained on several uni-modal and
multi-modal tasks, and evaluated on a variety of down-
stream tasks, including novel tasks that did not appear in
the pre-training stage. Results show that our pre-trained
model without any tuning can achieve reasonable perfor-
mance even on novel tasks. The performance can be im-
proved to a level close to state-of-the-art methods by con-
ducting prompt tuning on 1% of downstream task data.
Full-data ﬁne-tuning further delivers results on par with or
better than state-of-the-art results. Code shall be released.
*Equal contribution. †This work is done when Jinguo Zhu, Hao Li, and
Xiaoshi Wu are interns at SenseTime Research. BCorresponding author.
Model A
Model B
Model C
Image
Video
Text
Text
Task head 1 
Task head 2 
Task head 3 
Task head 4
Video Classification
VQA
Image Retrieval
Sentence Classification
Unified
Model
VQA
Video Caption
Image Retrieval
Video
Image
Image
Text
Text
Video
Classname
Classname
Image
Answer
Word
Text
Video
Video Classification
Input
Target
…
…
Unified
Model
…
Text
Previous Task-specific Perception Models
Image Classification
Video Retrieval
Uni-Perceiver (Ours)
share weight for various modalities and tasks
applicable to various tasks, e.g., 
Joint Probability Estimation 
Figure 1.
Comparing previous task-speciﬁc perception models
with our proposed Uni-Perceiver, which processes various modal-
ities and tasks with a single siamese model and shared parameters.
1. Introduction
Biological intelligence systems of animals perceive the
world by receiving information in different modalities, inte-
grating with the complex central nervous system, and pro-
cessing simultaneously for different tasks. However, de-
signing a generic artiﬁcial perception model that handles
multiple modalities and numerous tasks has always been
considered too difﬁcult.
To simplify this problem, pre-
vious machine learning research has focused on develop-
ing specialized models for inputs from certain restricted
modality, e.g., Convolutional Neural Networks [39] for vi-
sual recognition and Transformers [69] for natural language
processing. Recently, Transformers have been proved to
arXiv:2112.01522v1  [cs.CV]  2 Dec 2021

have competitive performance in more scenarios such as
image [9, 17, 45, 66, 68, 71, 73, 76] and video [4, 5, 74]
recognition, which triggers a new paradigm of designing
uniﬁed architectures for different modalities. Following this
paradigm, recent works [1, 22, 28, 56] adopt Transform-
ers as the backbone for multi-modal applications such as
visual-linguistic recognition. They convert the inputs from
different modalities into uniﬁed input token sequences with
modality-speciﬁc tokenizers. Models are pre-trained with
large-scale multi-modal datasets, and then adapted to down-
stream tasks with ﬁne-tuning.
Despite the ability of processing multi-modal informa-
tion with uniﬁed architectures, current methods still require
speciﬁc design and training for different tasks. This limita-
tion is caused by two reasons. First, the input of a particular
model is the combination of speciﬁc modalities required by
its target task. Second, previous works require prediction
heads speciﬁcally designed and trained for the target tasks.
We argue that this task-speciﬁc paradigm conﬂicts
with the objective of designing generic perceptual mod-
els.
Speciﬁcally, during pre-training, the specialised de-
signs for different tasks hinder the collaboration between
tasks, which may hurt the representational capacity. Mean-
while, when a pre-trained model is applied to a new task, the
input format and the prediction head need to be re-designed
and ﬁne-tuned on sufﬁcient downstream data. Considerable
effort in collecting and annotating data is required. Also,
all parameters need to be copied and maintained for each
downstream task, which becomes inefﬁcient and inconve-
nient as the number of tasks and the model size grow. On the
other hand, when ﬁne-tuning is performed with insufﬁcient
training data, it may forget the pre-trained knowledge that
is beneﬁcial to the downstream task, thereby hurting gen-
eralization performance [13]. All of these issues increase
the marginal cost of developing perception models for new
tasks and limit the capability to meet the rapidly grow-
ing demands of diverse scenarios, indicating task-speciﬁc
paradigm is not suitable for generic perceptual modeling.
Our core idea is to replace task-speciﬁc designs by
encoding different task inputs and targets from arbitrary
modalities into a uniﬁed representation space, and model
the joint probability of inputs and targets through the sim-
ilarity of their representations. This design eliminates the
gap between the formulations of different perception tasks,
and therefore encourages the collaboration between differ-
ent modalities and tasks in representation learning. More-
over, by aligning the formulations of pre-training and down-
stream tasks, the knowledge can be better transferred when
applying the pre-trained model to the target tasks.
The
model can even conduct zero-shot inference on novel tasks
that do not appear in the pre-training stage.
In this paper, we propose a uniﬁed architecture named
Uni-Perceiver, which processes various modalities and
tasks with a single siamese model and shared parameters.
Speciﬁcally, the task inputs and targets from arbitrary com-
binations of modalities are ﬁrst converted into uniﬁed to-
ken sequences with lightweight modality-speciﬁc tokeniz-
ers. The sequences are then encoded by a modality-agnostic
Transformer encoder into a uniﬁed representation space.
Different perception tasks are modeled as the same formu-
lation, ﬁnding the maximum likelihood target for each input
through the similarity of their representations, so as to facil-
itate the generic perceptual modeling.
Uni-Perceiver is pre-trained on various uni-modal tasks
such as image / video classiﬁcation and language model-
ing, and multi-modal tasks such as image-text retrieval and
language modeling with image clues.
When applied to
downstream tasks, thanks to the generic modeling of per-
ception tasks, the pre-trained model shows the ability of
zero-shot inference on novel tasks that did not appear in the
pre-training stage. Moreover, the performance can be fur-
ther boosted with additional task-speciﬁc data. For the few-
shot scenario, we adapt the model to downstream tasks with
prompt tuning [41], where only a small amount of addi-
tional parameters are optimized for speciﬁc tasks. The per-
formance of our model can be further improved with full-
model ﬁne-tuning on sufﬁcient downstream training data.
We pre-train our model on several uni-modal and multi-
modal tasks, and evaluate its performance on a variety of
downstream tasks, including novel tasks that did not ap-
pear in the pre-training stage. Results show that our pre-
trained model without any tuning can achieve reasonable
performance even on novel tasks. Its performance can be
boosted to a level close to state-of-the-art methods by con-
ducting prompt tuning with 1% of the downstream task data.
When ﬁne-tuning the pre-trained model with 100% of the
target data, our model achieves result on par with or better
than state-of-the-art methods on almost all the tasks, which
demonstrates the strong representation ability.
2. Related Works
Architecture. For visual recognition, Convolutional Neu-
ral Networks (CNN) [39] used to be the main architec-
ture paradigm. Motivated by the success of Transformers
in natural language processing [7, 16, 31, 34, 44, 69], at-
tempts have been made to apply Transformers to image and
video modalities. For image recognition, vision Transform-
ers [9, 17, 45, 66, 68, 71, 73, 76] replace CNNs by an image
patch tokenizer and a transformer encoder, which have been
proved to obtain competitive performance as CNNs. [4, 5,
74] make attempts to apply Transformers on video recog-
nition in a convolution-free fashion. For visual-linguistic
recognition, recent works [14, 36, 37, 46, 53, 62, 65, 78]
also adopt Transformers as the backbone, while they usu-
ally take regional features as inputs, which are typically
extracted by off-the-shelf object detectors (e.g., Faster R-

CNN [57] pre-trained on Visual Genome [30]). [24] at-
tempts to eliminate the need for object detectors by di-
rectly extracting features from the raw pixels with CNNs.
[1, 22, 28, 56] take a further step by applying Transform-
ers to raw image patches and word tokens. Transformers
have enabled a uniﬁed architecture paradigm for different
modalities, which only need the modality-speciﬁc tokeniz-
ers to convert inputs from different modalities into uniﬁed
input token sequences.
Nevertheless, previous architecture requires prediction
heads speciﬁcally designed and trained for different per-
ception tasks. Instead, we replace the task-speciﬁc design
by encoding different task inputs and targets into a uniﬁed
representation space, and model their relationship by repre-
sentational similarity. This modiﬁcation enables our model
to conduct zero-shot inference even on novel downstream
tasks that did not appear in the pre-training stage.
Pre-training. Large-scale pre-training has achieved great
success in the ﬁeld of deep learning, which can alleviate
the data-hungry challenge and improve the performance of
downstream tasks [72]. For image recognition, pre-training
is usually performed on image classiﬁcation datasets, e.g.,
ImageNet [15].
Video recognition networks are either
pre-trained on image classiﬁcation or video classiﬁcation
datasets, e.g., Moments in Time [49] and Kinetics [27].
In natural language processing, self-supervised language
modeling [7, 16, 31, 34, 44] is adopted for pre-training on
large-scale unlabeled corpora [54]. Speciﬁcally, GPT [7]
performs the auto-regressive pre-training, which optimizes
the probability of the next word conditioned on previous
words. BERT [16] uses masked language modeling (MLM)
and next sentence prediction (NSP) for pre-training. These
pre-trained models can serve as robust feature extractors for
downstream tasks with small architecture modiﬁcations.
Recent years have witnessed interest in large-scale cross-
modal pre-training [72].
Compared with uni-modal pre-
training, cross-modal pre-training needs to align informa-
tion from different modalities. Such pre-training is usu-
ally performed on image-text pairs collected from Inter-
net [8, 26, 50, 60] and manual annotated visual-linguistic
datasets [30, 40, 50]. Moreover, various pre-training ob-
jectives are also proposed to utilize these datasets effec-
tively. The most widely used objectives are image-text re-
trieval [2, 37, 47, 55, 63, 64, 65], masked language model-
ing with image clues [2, 37, 47, 62, 63, 64, 65], and masked
region modeling [14, 47, 62, 63, 65]. Among them, masked
region modeling requires regional features extracted by off-
the-shelf object detectors. More recently, CLIP [55] has
veriﬁed the effectiveness of only performing image-text re-
trieval pre-training on huge webly collected data.
Previous multi-task pre-training requires task-speciﬁc
heads, which hinders the collaboration among different
tasks. Instead, we encode different task inputs and targets
into a uniﬁed representation space, and model their relation-
ship by a uniﬁed representational similarity, which enables
the collaboration between different modalities and tasks.
Our pre-training tasks include image and video classiﬁca-
tion, language modeling with and without image clues, and
image-text retrieval. We do not use regional features and
the corresponding pre-training tasks.
Prompt Tuning. As an alternative solution to ﬁne-tuning,
prompt tuning has recently been proposed in the NLP com-
munity, which originated from prompting methods [41]. In
prompting, specially designed natural language tokens, or
namely prompts, are inserted into the input sequence as
hints for the target tasks. These prompt inputs are used
to query a large language model (e.g., GPT-3 [7]). Meth-
ods [25, 61] have been proposed to automate the prompt en-
gineering process. The prompting process does not tune any
of the parameters, which is empirically sub-optimal com-
pared to ﬁne-tuning [43].
Prompt tuning [33] is proposed to replace hard language
prompts with learnable prompt tokens that can be updated
through gradient back-propagation, while other parameters
are still kept ﬁxed. Other than adding learnable input to-
kens, Preﬁx-Tuning [38] adds learnable prompts to each
layer of the Transformer to boost the model capacity. For
few-shot scenario, [20] proves that prompt tuning can be
much better than traditional ﬁne-tuning. When the train-
ing data is sufﬁcient, prompt tuning performs slightly worse
than ﬁne-tuning [43]. However, the performance gap from
full-model ﬁne-tuning closes up as the pre-trained model
gets larger [33, 42]. Inspired by the success of prompt tun-
ing in NLP, [77] applies prompt tuning to visual-linguistic
pre-trained models (e.g., CLIP [55]) to perform few-shot
image classiﬁcation. [51, 58] further apply a residual fea-
ture adapter to improve the few-shot performance.
In this paper, we focus on the zero-shot and few-shot sce-
narios, where the downstream tasks may not even appear in
the pre-training stage. For few-shot learning, we adapt the
model with prompt tuning proposed by [41]. The perfor-
mance of our model can be further improved by ﬁne-tuning
the whole model with sufﬁcient downstream training data.
3. Method
3.1. Uniﬁed Architecture for Generic Perception
In this section, we will describe our uniﬁed architecture
for various modalities and tasks. Fig. 2 illustrates the archi-
tecture. Speciﬁcally, the model ﬁrst converts different task
inputs and targets from arbitrary combinations of modalities
into token sequences with modality-speciﬁc tokenizers. A
modality-agnostic Transformer encoder, which shares pa-
rameters for different input modalities and target tasks, is
then employed to encode different token sequences into a
shared representation space. Any perception task can be

𝑃𝑥, 𝑦∝exp{cos(𝑓𝑥, 𝑓(𝑦))/𝜏}
Transformer Encoder
Transformer Encoder
share
weight
<SPE>
𝑥!
"
𝑥#
"
Text
Tokenizer
Text
𝑓(𝑥)
…
𝑥!
$
𝑥%
$
Image
Tokenizer
Image
…
𝑥!
&
𝑥'
&
Video
Tokenizer
Video
…
Text
Tokenizer
Text
Image
Tokenizer
Image
Video
Tokenizer
Video
𝑓(𝑦)
𝒙∈𝓧
…
…
…
𝒚∈𝓨
Joint Probability Distribution
Input
Target
Any modality combinations can be applied 
Any modality combinations can be applied 
<my>
1
text tokens
text pos embed
text type embed
+
+
<dog>
<is>
<cute>
<EOT>
2
3
4
5
<T>
<T>
<T>
<T>
<T>
+
+
+
+
+
+
+
+
+
+
𝑥!
"
𝑥#
"
𝑥$
"
𝑥%
"
𝑥&
"
Text
Tokenizer
Linear Projection + Layer Norm
<SPE>
𝑦!
"
𝑦#!
"
𝑦!
$
𝑦%!$
𝑦!
&
𝑦'!
&
1
image patches
image pos embed
vision type embed
+
+
+
+
𝑥!'
<V>
2
+
+
<V>
3
+
+
<V>
4
+
+
<V>
8
+
+
<V>
9
+
+
<V>
𝑥#
'
𝑥$
'
𝑥%'
𝑥('
𝑥)'
…
…
…
…
Image
Tokenizer
Linear Projection + Layer Norm
1
frame patches
image pos embed
vision type embed
+
+
+
+
𝑥!
*
<V>
temporal pos embed
+
1
+
1
+
1
+
+
<V>
9
+
1
+
…
…
…
…
1
+
+
<V>
1
+
2
+
1
+
+
<V>
9
+
2
+
…
…
…
…
1
+
+
<V>
1
+
3
+
1
+
+
<V>
9
+
3
+
…
…
…
…
𝑥)
*
…
𝑥!+
*
𝑥!(
*
…
𝑥!)
*
𝑥#,
*
…
Video
Tokenizer
Linear Projection + Layer Norm
Figure 2. Overview of our uniﬁed architecture for generic perception. Different task inputs and targets from arbitrary modalities are
converted into uniﬁed token sequences with modality-speciﬁc tokenizers. A modality-agnostic weight-sharing Transformer encoder is then
applied to encode these token sequences into the shared representation space. Any perception task can be modeled as ﬁnding the maximum
likelihood target for each input through the similarity of their representations.
modeled in a single uniﬁed formulation, which ﬁnds the
maximum likelihood target for each input through the sim-
ilarity of their representations.
Tokenization. Given the raw inputs from text, image, and
video modalities, modality-speciﬁc tokenizers are applied
to generate the input token sequences for the Transformer
encoder.
Here, we use the BPE tokenizer [59] for text
modality, the image patch tokenizer [17] for image modal-
ity, and the temporal frame patch tokenizer [6] for video
modality. These outputted tokens are attached with addi-
tional modality type embeddings to identify which modal-
ity the raw input belongs to. Details of the modality-speciﬁc
tokenizers are described in the Appendix.
As illustrated in Fig. 2, depending on the task require-
ments, the input sequence x of the Transformer encoder
can be composed of different combinations of text token
sequence xT , image token sequence xI, and video to-
ken sequence xV .
At the beginning of the sequence x,
a special token <SPE> is always inserted.
For exam-
ple, x = [<SPE>, xI, xT ] for image-text pair inputs, and
x = [<SPE>, xV ] for video-only inputs, where [ ] denotes
the sequence concatenation. The feature of <SPE> at the
encoder output serves as the representation of the input.
Generic Modeling of Perception Tasks.
We model dif-
ferent perception tasks with a uniﬁed architecture, whose
parameters are shared for all target tasks. Each task is de-
ﬁned with a set of inputs X and a set of candidate targets Y.
Given an input x ∈X, the task is formulated as ﬁnding the
maximum likelihood target y ∈Y as
ˆy = arg max
y∈Y P(x, y),
(1)
where P(x, y) is the joint probability distribution. The joint
probability is estimated through calculating the cosine sim-
ilarity between the representation of x and y as
P(x, y) ∝exp
 cos
 f(x), f(y)

/τ

,
(2)
where f(·) is the Transformer encoder, and τ > 0 is a learn-
able temperature parameter.
To obtain generic modeling capability, our uniﬁed archi-
tecture is pre-trained on a variety of multi-modal tasks si-
multaneously. Suppose a series of pre-training tasks is de-
noted as {X1, Y1}, {X2, Y2}, ..., {Xn, Yn}, where Xi and
Yi is the input set and target set of the i-th task, respectively.
Then the pre-training loss is deﬁned as
L =
n
X
i=1
E
{x,y}∈{Xi,Yi}

−log
P(x, y)
P
z∈Yi P(x, z)

,
(3)
where E is the mathematical expectation, and {x, y} ∈
{Xi, Yi} indicates a ground-truth input-target pair sampled
from the dataset of the i-th task.
Our uniﬁed architecture is suitable for any task, as long
as its input set X and target set Y are composed of images,
texts, and videos. For example, the target set Y in classiﬁca-
tion tasks can be a set of class names, a set of class descrip-
tions, or even a set of images with handwritten numbers
representing class indexes. Detailed instances of X and Y
will be introduced in the next subsection. Note that we cur-
rently focus on text, image, and video modalities, but more
modalities are also applicable, as long as the corresponding
tokenizers are applied.
Relation to Previous Perception Models.
Our method
shares the same goal of learning multi-modal representa-
tions as previous perception models.
However, existing
works follow a task-speciﬁc paradigm, while our method
is designed for generic perceptual modeling. The main dif-
ference lies in two parts:
1) Previous works focus on inputs from certain combina-
tions of modalities required by their target tasks, while our

𝑓(𝑥)
<SPE>
Image
Classification
𝑥!
"
Image
𝑥#
"
𝑥$
"
𝑥%
"
𝑥&
"
𝑓(𝑦)
<SPE>
𝑦!
'
𝑦#
'
𝑦$
'
𝑦%
'
𝑦&
'
Class label
𝑓(𝑥)
<SPE>
Video
Classification
𝑥!
(
Video
𝑥#
(
𝑥$
(
𝑥%
(
𝑥&
(
𝑓(𝑦)
<SPE>
𝑦!
'
𝑦#
'
𝑦$
'
𝑦%
'
𝑦&
'
Class label
𝑓(𝑥)
<SPE>
Language
Modeling
𝑥!
'
Sentence with masked tokens
𝑥#
'
𝑥$
'
𝑥&
'
𝑓(𝑦)
<SPE>
𝑦!
'
Vocabulary
<SPE>
𝑓(𝑥)
<SPE>
Language 
Modeling with 
Image Clues
𝑥!
"
Image + Sentence with masked tokens
𝑥#
"
𝑥!
'
𝑥$
'
𝑓(𝑦)
<SPE>
𝑦!
'
Vocabulary
<SPE>
𝑓(𝑥)
<SPE>
Text QA 
Retrieval
(Q →A)
𝑥!
'
Question
𝑥#
'
𝑥$
'
𝑥%
'
𝑥&
'
𝑓(𝑦)
<SPE>
𝑦!
'
𝑦#
'
𝑦$
'
𝑦%
'
𝑦&
'
Answer
𝑓(𝑥)
<SPE>
Text QA 
Retrieval
(A →Q)
𝑥!
'
Answer
𝑥#
'
𝑥$
'
𝑥%
'
𝑥&
'
𝑓(𝑦)
<SPE>
𝑦!
'
𝑦#
'
𝑦$
'
𝑦%
'
𝑦&
'
Question
𝑓(𝑥)
<SPE>
Image-Text 
Retrieval
(I → T)
𝑥!
"
Image
𝑥#
"
𝑥$
"
𝑥%
"
𝑥&
"
𝑓(𝑦)
<SPE>
𝑦!
'
𝑦#
'
𝑦$
'
𝑦%
'
𝑦&
'
Caption
𝑓(𝑥)
<SPE>
Image-Text 
Retrieval
(T → I)
𝑥!
'
Caption
𝑥#
'
𝑥$
'
𝑥%
'
𝑥&
'
𝑓(𝑦)
<SPE>
𝑦!
"
𝑦#
"
𝑦$
"
𝑦%
"
𝑦&
"
Image
Figure 3. Input and target formats of pre-training tasks. For each
task, the left column represents the format of input sequence x, and
the right column represents the format of the target sequence y.
f(x) and f(y) indicate the representations used for calculating the
joint probability distribution as in Eq. (2). Here, we have omitted
the tokenizer and encoder for concision.
method handles arbitrary combinations of modalities with a
uniﬁed architecture and shared parameters.
2) Previous works require prediction heads speciﬁcally
designed and trained for each perception task, while our
method models different tasks with the same formulation
and processes them with uniﬁed modeling.
Therefore, when transferred to a new task, previous
methods need to re-design their input formats and predic-
tion heads accordingly. Models require ﬁne-tuning on sufﬁ-
cient task-speciﬁc data, resulting in remarkable human and
computational costs. In contrast, our method can directly
conduct zero-shot inference on novel tasks that do not ap-
pear in the pre-training stage. The performance can be fur-
ther boosted with prompt tuning on few-shot downstream
data and ﬁne-tuning on sufﬁcient downstream data.
3.2. Pre-training on Multi-Modal Tasks
Our model is pre-trained on a variety of tasks simulta-
neously to learn the multi-modal generic representations.
The pre-training tasks are illustrated in Fig. 3.
Speciﬁ-
cally, for uni-modal pre-training tasks, we adopt the most
widely-used image classiﬁcation, video classiﬁcation, and
language modeling tasks. To further enhance the relation-
ships between different modalities, some cross-modal tasks
are also employed, such as language modeling with image
clues and image-text retrieval tasks. Note that for image and
video classiﬁcation tasks, we regard each class name (e.g.,
tigershark) as a text sequence. This provides weak su-
pervision for bridging the gap among the representations of
images, videos, and texts.
Image and Video Classiﬁcation. In image and video clas-
siﬁcation tasks, X denotes the set of all possible images or
videos in the training dataset, and Y consists of candidate
class labels in each dataset. Each class name is regarded as
a text sequence to provide weak supervision of the relation-
ship to texts. Both the input x ∈X and target y ∈Y start
with an <SPE> token, whose feature at the encoder output
represents the corresponding sequence.
Language Modeling with and without Image Clues. The
language modeling task aims to predict the masked words
according to the context. Both auto-regressive [7] and auto-
encoding [16] language modeling are adopted. When inputs
have no image, the auto-regressive and auto-encoding tasks
correspond to the text generation and the masked language
modeling tasks, respectively. When inputs have images, the
auto-regressive and auto-encoding tasks correspond to the
image caption and the masked language modeling with im-
age clues tasks, respectively.
For auto-encoding language modeling, we follow the
practice in BERT [16] to mask out 15% words from the
text randomly. The model predicts each masked word based
on all inputs. For auto-regressive language modeling, the
model predicts each word based on its previous text and im-
age (if any). Please refer to the Appendix for an efﬁcient
implementation of auto-regressive language modeling.
In this task, X consists of language sentences or image-
text pairs. Y denotes the set of all words in the vocabulary,
where each word is regarded as a single text sequence. Each
word that needs to be predicted in x ∈X is replaced by a
<SPE> token, whose feature at the encoder output is used
to match the words in the vocabulary Y.
Image and Text Retrieval.
For image-text retrieval, the
input sets X and Y are composed of images and text se-
quences respectively, or vice versa. For text-only retrieval,
the input sets X and Y are both text sequences. Each se-
quence in X and Y has a special token <SPE> at the begin-
ning, whose feature at the output of the encoder serves as
the ﬁnal representation.
3.3. Zero-shot, Prompt Tuning and Fine-tuning
During the pre-training stage, our uniﬁed architecture
learns to model the joint distribution of input and target se-
quences from arbitrary modalities. Thanks to the generic
perceptual modeling, our pre-trained model can perform
zero-shot inference on completely novel tasks that do not
appear in the pre-training stage. Our model can be further

adapted to downstream tasks with task-speciﬁc additional
training data. For the few-shot scenario, we employ the
prompt tuning [41] scheme, which only adds a few addi-
tional task-speciﬁc parameters to the model. The perfor-
mance on speciﬁc tasks can be further improved by ﬁne-
tuning the whole model on sufﬁcient downstream data.
Zero-shot Inference on Novel Tasks. Our model has the
potential to perform zero-shot inference on any perception
task that can be modeled by a joint probability distribu-
tion. For a task with input x ∈X and a candidate target
y ∈Y, we ﬁrstly tokenize x and y into two sequences. The
joint probability P(x, y) is then estimated following Eq. (2).
Zero-shot inference can be conducted by maximum likeli-
hood estimation, as described in Eq. (1). Performance can
also be improved through prompt engineering, similar to
the prompting [41] for language models such as GPT-3 [7],
where network training is not required.
Prompt Tuning.
For the few-shot scenario with limited
training data, we adopt prompt tuning, which is memory-
efﬁcient and has been proved to be better than the ﬁne-
tuning scheme in few-shot NLP [20]. In prompt tuning,
most pre-trained parameters are ﬁxed, leaving only a small
portion of task-speciﬁc parameters to be optimized. Specif-
ically, following P-Tuning v2 [42], learnable prompt tokens
with random initialization are added at each layer of the
Transformer encoder, and class labels with linear heads are
added for classiﬁcation tasks. The <SPE> token and layer
norm parameters are also tuned. We refer the readers to the
Appendix for more details.
Fine-Tuning.
For downstream tasks with sufﬁcient train-
ing data, our model can also be ﬁne-tuned to further im-
prove its performance. During ﬁne-tuning, our model can
serve as a joint probability estimator (same as our proposed
generic perceptual modeling), or a feature extractor (same
as traditional pre-trained models). Under the setting of joint
probability estimation, the downstream tasks are formulated
in the same uniﬁed manner as in pre-training. On the other
hand, similar to previous perception models, our model can
also be used as a feature extractor by adding a task-speciﬁc
head on the top of the encoder. We empirically ﬁnd these
two schemes achieve very similar performance, and hence
the scheme of joint probability distribution estimator is used
by default for consistency.
4. Experiments
4.1. Datasets
Our model is pre-trained on a variety of tasks, whose
statistics are listed in Tab. 1. We pre-train image classiﬁ-
cation on ImageNet-21k [15]. For video classiﬁcation, we
pre-train on Kinetics-700 [27] and Moments in Time [49].
We pre-train language modeling on BookCorpora [79] &
Dataset
#Images
#Videos
#Text
ImageNet-21k [15]
14.2M
0
21K
Kinetics-700 [27]
0
542K
700
Moments in Time [49]
0
792K
339
Books&Wiki [79]
0
0
101M
PAQ [35]
0
0
65M
CC3M [60]
3.0M
0
3.0M
CC12M [8]
11.1M
0
11.1M
COCO Caption [11]
113K
0
567K
Visual Genome [30]
108K
0
5.41M
SBU [50]
830K
0
830K
YFCC* [26]
14.8M
0
14.8M
Table 1. Pre-training dataset statistics. #Images, #Videos and
#Text represent the number of images, video clips, and textual
sentences (or phrases), respectively.
English Wikipedia (Books&Wiki) and PAQ [35].
For
language modeling with image clues and image-text re-
trieval, we use a combination of COCO Caption [12], SBU
Captions (SBU) [50], Visual Genome [30], CC3M [60],
CC12M [8] and YFCC [26]. To evaluate the effectiveness of
our method and verify the generalization of our pre-trained
model, we also use several novel datasets that did not appear
in pre-training, i.e., Flickr30k [52], MSVD [10], VQA [21],
and GLUE [70]. See Appendix for the details of datasets.
4.2. Implementation Details
The Transformer encoder used for experiments is of the
same conﬁguration with BERTBASE [16]. It is a 12-layer
encoder with the embedding dimension of 768 and the at-
tention head number of 12. The hidden dimension size in
FFN is 3072. We pre-train the model with multiple tasks
simultaneously. In each iteration, each GPU independently
samples a single task and dataset. The gradients of different
GPUs are synchronized after the gradient back-propagation.
We use AdamW [29] optimizer with a base learning rate of
0.0002 and a weight decay of 0.05. Gradient clipping with
5.0 is used to stabilize training. We also use drop path [32]
with a probability of 0.1 during training. The model is pre-
trained on 128 Tesla V100 GPUs in a distributed fashion
for 500k iterations. We use the cosine learning rate sched-
ule with 50k iterations of linear warmup. See Appendix for
more implementation details.
4.3. Evaluation on Pre-training Tasks
We ﬁrst evaluate our pre-trained model on tasks that have
been involved in the pre-training stage, while the datasets
might be different. The widely used Imagenet-1k [15] and
Kinetics-400 [27] are used for evaluating the image and
video classiﬁcation tasks, respectively. COCO Caption and
Flickr30k are the typical datasets used to evaluate the per-
formance on image caption and image-text retrieval.
Results.
Tab. 2, Tab. 3, and Tab. 4 present the evaluation
results of our models on four pre-training tasks, i.e., image

Task
ImageNet-1k
Kinetics-400
Acc
Acc
DeiT [67]
81.8
-
TimeSformer [6]
-
75.5
Ours w/o Tuning
78.0
73.5
Ours PT (0.1%)
79.4
73.6
Ours FT (0.1%)
78.8
73.5
Ours PT (1%)
80.2
73.6
Ours FT (1%)
80.2
73.6
Ours FT (100%)
83.8
75.8
Table 2. Image and video classiﬁcation performance under dif-
ferent tuning settings. PT means prompt-tuning, and FT means
ﬁne-tuning. The percentage of data used in tuning is noted.
classiﬁcation, video classiﬁcation, image-text retrieval, and
image caption. We compare our model with task-speciﬁc
SOTA methods having the similar model size.
Results show that without any tuning, our pre-trained
model reaches reasonable performance on these tasks. Al-
though the performance is slightly worse than the SOTA
methods. We speculate that the performance gap is due to
the limited capacity of our model, which may have a neg-
ative impact on the representation ability. Note that our
method shares a similar model size with other methods,
but need to simultaneously process much more pre-training
tasks from various datasets and modalities.
By conducting prompt tuning on each task with 1%
downstream data, the performance is boosted to a level
close to SOTA performance. It’s worth noting that all pa-
rameters of other methods are speciﬁcally trained on the tar-
get tasks. While for our prompt tuning, only a small amount
of parameters are tuned, and the encoder is still ﬁxed and
shared among different tasks, indicating that our method
can handle different tasks with low marginal cost.
We further ﬁne-tune the pre-trained model with 100% of
the downstream data. With full-data ﬁne-tuning, our model
achieves performance on-par with or better than the SOTA
methods on all these tasks, which proves our model has
learned high-quality representations. We also compare the
performance of prompt tuning and ﬁne-tuning in the sce-
nario of few-shot learning. On all of these tasks, prompt
tuning shows a consistently better performance than ﬁne-
tuning with the same amount of data, which demonstrates
its superiority under few-shot scenarios.
4.4. Generalization to Novel Tasks
Thanks to the generic perceptual modeling, our pre-
trained model can generalize to novel tasks by convert-
ing the tasks into our uniﬁed task formulation. We eval-
uate zero-shot inference on tasks that did not appear in
pre-training, i.e., video caption, video-text retrieval, visual
question answering, and natural language understanding.
Video Caption and Video-Text Retrieval. Our pre-trained
model is evaluated on MSVD [10] dataset. Speciﬁcally, for
video caption, X1 consists of the concatenation of video and
language sequences that have been predicted, and X2 de-
notes the set of all words in the vocabulary. For the video-
text retrieval, the input sets X1 and X2 consist of possible
video and text sequences, or vice versa.
Visual Question Answering.
In visual question answer-
ing, the model is asked to answer a question w.r.t a refer-
ence image from a list of answer candidates. We evaluate
our pre-trained model on VQA [21] dataset. X1 is a set
of image-text sequence, where the text is the question to-
kens followed by a <SPE> token used to predict the an-
swers. Each x2 ∈X2 is an answer sequence beginning with
<SPE>. Inference is achieved by computing the similarity
between output features of <SPE> in x1 and x2.
Natural Language Understanding.
Six language-only
tasks are chosen from GLUE benchmark [70] to evaluate the
natural language understanding ability of our pre-trained
model. These tasks are either single sentence classiﬁcation
or sentence-pair classiﬁcation tasks. We follow [19] to con-
struct the textual class labels for each dataset. Here, the
input sequence x1 ∈X1 denotes the input single sentence
or the sentence-pair, and the sequence x2 ∈X2 represents
the class labels in each dataset.
Result. Tab. 5, Tab. 6 and Tab. 7 show the results on video
caption and video-text retrieval and visual question answer-
ing, respectively.
Our pre-trained model can obtain rea-
sonable zero-shot performance on these novel tasks. Note
that none of previous works can perform this type of zero-
shot inference at all. From Tab. 7, we note that our model
shows unsatisfactory zero-shot performance on “Yes/No”
and “Number” subsets in VQA. We speculate that it may be
due to the distribution difference between those answers and
our pre-training corpora. We futher conduct prompt tuning
on these tasks with only 1% data, which brings our model
to a level close to the SOTA results. By further ﬁne-tuning
with 100% downstream data, our model can achieve results
on par with or better than the SOTA methods.
On the GLUE benchmark, our model can achieve com-
parable performance with [3] in zero-shot evaluation. When
ﬁne-tuning the pre-trained model with 100% downstream
data, our model performs slightly worse than BERTBASE.
Since our model has the same number of parameters as
BERTBASE, but need to process much more tasks from var-
ious datasets and modalities, we speculate that the perfor-
mance drop is due to the limited capacity of the model.
5. Conclusion
In this paper, we propose a uniﬁed perception architec-
ture that processes various modalities and tasks with a sin-
gle model and shared parameters. With pre-training on uni-
modal and multi-modal tasks, our model shows the ability

Task
Text Retrieval
Image Retrieval
Flickr30k
COCO Caption
Flickr30k
COCO Caption
R@1
R@5
R10
R@1
R@5
R10
R@1
R@5
R10
R@1
R@5
R10
ImageBERT [53] w/o Tuning
70.7
90.2
94.0
44.0
71.2
80.4
54.3
79.6
87.5
32.3
59.0
70.2
UNITER-B [14] w/o Tuning
80.7
95.7
98.0
-
-
-
66.2
88.4
92.9
-
-
-
ViLT [28] w/o Tuning
73.2
93.6
96.5
56.5
82.6
89.6
55.0
82.5
89.8
40.4
70.0
81.1
Unicoder-VL [23]
86.2
96.3
99.0
62.3
87.1
92.8
71.5
91.2
95.2
48.4
76.7
85.9
UNITER-B
85.9
97.1
98.8
64.4
87.4
93.1
72.5
92.4
96.1
50.3
78.5
87.2
ViLT
83.5
96.7
98.6
61.5
86.3
92.7
64.4
88.7
93.8
42.7
72.9
83.1
Ours w/o Tuning
74.8
94.8
98.2
57.7
85.6
92.3
65.8
88.8
93.6
46.3
75.0
84.0
Ours PT (1%)
84.4
97.8
99.2
61.4
86.7
93.2
71.1
91.6
95.1
47.0
75.3
84.3
Ours FT (1%)
78.4
95.7
97.8
60.2
85.1
90.6
61.0
85.7
91.0
43.6
70.9
80.5
Ours PT (10%)
86.4
98.2
99.5
61.6
87.0
93.2
72.5
92.3
95.7
47.2
75.4
84.3
Ours FT (10%)
84.9
97.4
98.3
60.9
85.5
92.1
67.9
89.4
92.9
45.6
73.4
82.6
Ours FT (100%)
87.9
98.2
99.1
64.7
87.8
93.7
74.9
93.5
96.0
48.3
75.9
84.5
Table 3. Image-text retrieval performance under different tuning settings. PT means prompt-tuning, and FT means ﬁne-tuning. The
percentage of data used in tuning is noted.
Task
COCO Caption
Flickr30k
B@4
M
C
S
B@4
M
C
S
Uniﬁed VLP [78]
36.5
28.4
116.9
21.2
30.1
23.0
67.4
17.0
Ours w/o Tuning
33.6
27.0
109.8
20.3
17.0
16.2
41.2
11.2
Ours PT (1%)
34.3
27.2
109.6
21.2
28.1
21.6
59.1
15.6
Ours FT (1%)
28.0
26.8
100.1
20.2
18.9
19.7
45.3
14.3
Ours PT (10%)
35.0
27.9
114.1
21.3
28.8
22.1
61.7
16.8
Ours FT (10%)
32.7
27.5
109.0
21.1
26.9
21.6
52.1
14.5
Ours FT (100%)
35.6
28.1
116.5
21.5
30.1
24.5
72.7
18.2
Table 4. Image caption performance under different tuning set-
tings. B@4, M, C, S stand for BLEU-4, METEOR, CIDEr, and
SPICE scores, respectively.
Task
MSVD
B@4
M
R
C
S
ORG-TRL [75]
54.3
36.4
73.9
95.2
-
Ours w/o Tuning
20.3
25.8
52.1
45.7
6.5
Ours PT (1%)
54.8
38.9
74.7
104.8
6.6
Ours FT (1%)
47.3
35.8
66.2
80.1
6.2
Ours PT (10%)
57.2
39.1
75.6
112.1
6.8
Ours FT (10%)
56.7
38.7
70.0
88.2
6.7
Ours FT (100%)
61.5
42.3
79.0
131.0
7.7
Table 5.
Video caption (novel task) performance under differ-
ent tuning settings.
Note that this task did not appear in our
pre-training. The only task related to video modality in our pre-
training is video classiﬁcation.
of zero-shot inference on novel tasks, and can reach the per-
formance close to SOTA results by prompt tuning with only
a small amount of downstream data. The performance can
be further improved to be on par with or superior to SOTA
results by full-data ﬁne-tuning.
Limitations. Our method is currently only applicable when
the target set is discrete, such as classiﬁcation and retrieval.
Whether our model can be extended to regression tasks is
still questionable. Future work may explore the uniﬁed per-
Task
Text Retrieval
Video Retrieval
MSVD
MSVD
R@1
R@5
R10
R@1
R@5
R@10
CLIP4clip [48]
56.6
79.7
84.3
46.2
76.1
84.6
CLIP2video [18]
58.7
85.6
91.6
47.0
76.8
85.9
Ours w/o Tuning
42.7
69.1
79.6
34.6
64.5
75.4
Ours PT (1%)
61.2
83.7
89.0
42.6
73.3
82.5
Ours FT (1%)
49.6
75.8
83.7
37.5
68.2
79.3
Ours PT (10%))
61.3
84.8
90.9
43.1
74.2
83.4
Ours FT (10%))
59.1
81.9
87.4
41.7
71.6
81.3
Ours FT (100%)
61.5
83.5
90.2
45.4
75.8
85.0
Table 6. Video-text retrieval (novel task) performance under dif-
ferent tuning settings. Note that this task did not appear in our
pre-training.
Task
VQA v2 test-dev
Yes/No
Numbers
Others
Uniﬁed VLP [78]
87.2
52.1
60.3
Ours w/o Tuning
0.9
3.0
25.5
Ours PT (0.1%)
63.0
31.8
49.6
Ours FT (0.1%)
63.0
30.6
49.1
Ours PT (1%)
70.8
41.3
57.7
Ours FT (1%)
71.0
42.4
57.5
Ours FT (100%)
84.8
47.4
61.8
Table 7. Visual question answering (novel task) performance un-
der different tuning settings. Note that this task did not appear in
our pre-training.
ception model of both discrete and continuous target sets.
Potential Negative Societal Impact. This work may share
the common negative impacts of large-scale training, which
may consume lots of electricity and result in increased car-
bon emissions. This method also learns from a large number
of datasets that may contain data biases. Future work may
seek for more efﬁcient and unbiased training.

Task
GLUE
MNLI
QNLI
QQP
RTE
SST-2
MRPC
(Acc)
(Acc)
(F1)
(Acc)
(Acc)
(F1)
PLM [3] w/o tuning
49.4
50.7
46.6
53.8
70.6
44.2
BERTBASE [70]
84.6
92.7
71.2
66.4
93.5
88.9
Ours w/o Tuning
49.6
51.0
53.6
55.6
70.6
76.1
Ours PT (1%)
60.1
76.0
70.2
56.3
80.9
80.3
Ours FT (1%)
47.3
60.6
68.9
49.1
69.7
72.3
Ours PT (10%)
68.5
83.2
77.0
58.2
83.4
83.2
Ours FT (10%)
60.5
71.5
71.4
50.5
79.1
80.6
Ours FT (100%)
81.7
89.9
87.1
64.3
90.2
86.6
Table 8. Natural language understanding (novel task) performance
under different tuning settings. Note that this task did not appear
in our pre-training.
Acknowledgments
The work is supported by the Na-
tional Key R&D Program of China (2020AAA0105200),
Beijing Academy of Artiﬁcial Intelligence.
References
[1] Hassan Akbari, Li Yuan, Rui Qian, Wei-Hong Chuang, Shih-
Fu Chang, Yin Cui, and Boqing Gong. Vatt: Transformers
for multimodal self-supervised learning from raw video, au-
dio and text. arXiv preprint arXiv:2104.11178, 2021. 2, 3
[2] Chris Alberti, Jeffrey Ling, Michael Collins, and David Re-
itter. Fusion of detected objects in text for visual question
answering. arXiv preprint arXiv:1908.05054, 2019. 3
[3] Anonymous. Are BERT families zero-shot learners? a study
on their potential and limitations.
In Submitted to ICLR,
2022. under review. 7, 9
[4] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen
Sun, Mario Luˇci´c, and Cordelia Schmid. Vivit: A video vi-
sion transformer. arXiv preprint arXiv:2103.15691, 2021. 2
[5] Gedas Bertasius, Heng Wang, and Lorenzo Torresani.
Is
space-time attention all you need for video understanding?
In ICML, July 2021. 2
[6] Gedas Bertasius, Heng Wang, and Lorenzo Torresani.
Is
space-time attention all you need for video understanding?
arXiv preprint arXiv:2102.05095, 2021. 4, 7, 12, 14
[7] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Sub-
biah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakan-
tan, Pranav Shyam, Girish Sastry, Amanda Askell, Sand-
hini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. J.
Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler,
Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen,
Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess,
Jack Clark, Christopher Berner, Sam McCandlish, Alec Rad-
ford, Ilya Sutskever, and Dario Amodei.
Language mod-
els are few-shot learners. arXiv preprint arXiv:2005.14165,
2020. 2, 3, 5, 6
[8] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu
Soricut.
Conceptual 12M: Pushing web-scale image-text
pre-training to recognize long-tail visual concepts. In CVPR,
2021. 3, 6, 14
[9] Chun-Fu Chen, Quanfu Fan, and Rameswar Panda. Crossvit:
Cross-attention multi-scale vision transformer for image
classiﬁcation. arXiv preprint arXiv:2103.14899, 2021. 2
[10] David Chen and William B Dolan. Collecting highly paral-
lel data for paraphrase evaluation. In ACL, pages 190–200,
2011. 6, 7
[11] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedan-
tam, Saurabh Gupta, Piotr Doll´ar, and C. Lawrence Zit-
nick. Microsoft coco captions: Data collection and evalu-
ation server. arXiv preprint arXiv:1504.00325, 2015. 6, 14
[12] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedan-
tam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick.
Microsoft coco captions:
Data collection and evaluation
server. arXiv preprint arXiv:1504.00325, 2015. 6
[13] Xinyang Chen, Sinan Wang, Bo Fu, Mingsheng Long, and
Jianmin Wang. Catastrophic forgetting meets negative trans-
fer: Batch spectral shrinkage for safe transfer learning. In
NeurIPS, 2019. 2
[14] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy,
Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter:
Universal image-text representation learning.
In ECCV,
pages 104–120. Springer, 2020. 2, 3, 8
[15] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In CVPR, pages 248–255. Ieee, 2009. 3, 6, 14
[16] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova.
Bert:
Pre-training of deep bidirectional
transformers for language understanding.
arXiv preprint
arXiv:1810.04805, 2018. 2, 3, 5, 6
[17] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, et al. An image is worth 16x16 words: Trans-
formers for image recognition at scale.
arXiv preprint
arXiv:2010.11929, 2020. 2, 4, 12
[18] Han Fang, Pengfei Xiong, Luhui Xu, and Yu Chen.
Clip2video: Mastering video-text retrieval via image clip.
arXiv preprint arXiv:2106.11097, 2021. 8
[19] Tianyu Gao, Adam Fisch, and Danqi Chen.
Making pre-
trained language models better few-shot learners.
arXiv
preprint arXiv:2012.15723, 2020. 7
[20] Tianyu Gao, Adam Fisch, and Danqi Chen.
Making pre-
trained language models better few-shot learners. In ACL,
2021. 3, 6
[21] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Ba-
tra, and Devi Parikh. Making the V in VQA matter: Ele-
vating the role of image understanding in Visual Question
Answering. In CVPR, 2017. 6, 7
[22] Ronghang Hu and Amanpreet Singh. Unit: Multimodal mul-
titask learning with a uniﬁed transformer.
arXiv preprint
arXiv:2102.10772, 2021. 2, 3
[23] Haoyang Huang, Yaobo Liang, Nan Duan, Ming Gong, Lin-
jun Shou, Daxin Jiang, and Ming Zhou. Unicoder: A uni-
versal language encoder by pre-training with multiple cross-
lingual tasks. arXiv preprint arXiv:1909.00964, 2019. 8
[24] Zhicheng Huang, Zhaoyang Zeng, Bei Liu, Dongmei Fu,
and Jianlong Fu.
Pixel-bert: Aligning image pixels with
text by deep multi-modal transformers.
arXiv preprint
arXiv:2004.00849, 2020. 3
[25] Zhengbao Jiang, Frank F. Xu, J. Araki, and Graham Neubig.
How can we know what language models know?
TACL,
8:423–438, 2020. 3

[26] Sebastian Kalkowski, Christian Schulze, Andreas Dengel,
and Damian Borth. Real-time analysis and visualization of
the yfcc100m dataset. In Proceedings of the 2015 workshop
on community-organized multimodal mining: opportunities
for novel solutions, pages 25–30, 2015. 3, 6, 14
[27] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang,
Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola,
Tim Green, Trevor Back, Paul Natsev, et al. The kinetics hu-
man action video dataset. arXiv preprint arXiv:1705.06950,
2017. 3, 6, 14
[28] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Vision-
and-language transformer without convolution or region su-
pervision. arXiv preprint arXiv:2102.03334, 2021. 2, 3, 8
[29] Diederik P Kingma and Jimmy Ba. Adam: A method for
stochastic optimization.
arXiv preprint arXiv:1412.6980,
2014. 6
[30] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson,
Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalan-
tidis, Li-Jia Li, David A Shamma, et al.
Visual genome:
Connecting language and vision using crowdsourced dense
image annotations. IJCV, 123(1):32–73, 2017. 3, 6, 14
[31] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin
Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite
bert for self-supervised learning of language representations.
arXiv preprint arXiv:1909.11942, 2019. 2, 3
[32] Gustav
Larsson,
Michael
Maire,
and
Gregory
Shakhnarovich.
Fractalnet:
Ultra-deep neural networks
without residuals. arXiv preprint arXiv:1605.07648, 2017.
6
[33] Brian Lester, Rami Al-Rfou, and Noah Constant. The power
of scale for parameter-efﬁcient prompt tuning. In EMNLP,
2021. 3
[34] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvinine-
jad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and
Luke Zettlemoyer. Bart: Denoising sequence-to-sequence
pre-training for natural language generation, translation, and
comprehension. arXiv preprint arXiv:1910.13461, 2019. 2,
3
[35] Patrick Lewis, Yuxiang Wu, Linqing Liu, Pasquale Min-
ervini, Heinrich K¨uttler, Aleksandra Piktus, Pontus Stene-
torp, and Sebastian Riedel. Paq: 65 million probably-asked
questions and what you can do with them. arXiv preprint
arXiv:2102.07033, 2021. 6, 14
[36] Gen Li, Nan Duan, Yuejian Fang, Daxin Jiang, and Ming
Zhou. Unicoder-vl: A universal encoder for vision and lan-
guage by cross-modal pre-training. In AAAI, 2020. 2
[37] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh,
and Kai-Wei Chang.
Visualbert:
A simple and perfor-
mant baseline for vision and language.
arXiv preprint
arXiv:1908.03557, 2019. 2, 3
[38] Xiang Lisa Li and Percy Liang. Preﬁx-tuning: Optimizing
continuous prompts for generation, 2021. 3
[39] Zewen Li, Fan Liu, Wenjie Yang, Shouheng Peng, and Jun
Zhou. A survey of convolutional neural networks: Analysis,
applications, and prospects. IEEE Transactions on Neural
Networks and Learning Systems, pages 1–21, 2021. 1, 2
[40] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence
Zitnick. Microsoft coco: Common objects in context. In
ECCV, 2014. 3
[41] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hi-
roaki Hayashi, and Graham Neubig. Pre-train, prompt, and
predict: A systematic survey of prompting methods in nat-
ural language processing. arXiv preprint arXiv:2107.13586,
2021. 2, 3, 6
[42] Xiao Liu, Kaixuan Ji, Yicheng Fu, Zhengxiao Du, Zhilin
Yang, and Jie Tang. P-tuning v2: Prompt tuning can be com-
parable to ﬁne-tuning universally across scales and tasks.
arXiv preprint arXiv:2110.07602, 2021. 3, 6
[43] Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie
Qian, Zhilin Yang, and Jie Tang. Gpt understands, too. arXiv
preprint arXiv:2103.10385, 2021. 3
[44] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar
Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettle-
moyer, and Veselin Stoyanov. Roberta: A robustly optimized
bert pretraining approach. arXiv preprint arXiv:1907.11692,
2019. 2, 3
[45] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei,
Zheng Zhang, Stephen Lin, and Baining Guo. Swin trans-
former: Hierarchical vision transformer using shifted win-
dows. ICCV, 2021. 2
[46] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert:
Pretraining task-agnostic visiolinguistic representations for
vision-and-language tasks. In NeurIPS, 2019. 2
[47] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee.
Vilbert: Pretraining task-agnostic visiolinguistic represen-
tations for vision-and-language tasks.
arXiv preprint
arXiv:1908.02265, 2019. 3
[48] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei,
Nan Duan, and Tianrui Li. Clip4clip: An empirical study
of clip for end to end video clip retrieval. arXiv preprint
arXiv:2104.08860, 2021. 8
[49] Mathew Monfort, Alex Andonian, Bolei Zhou, Kandan Ra-
makrishnan, Sarah Adel Bargal, Tom Yan, Lisa Brown,
Quanfu Fan, Dan Gutfreund, Carl Vondrick, et al. Moments
in time dataset: one million videos for event understanding.
TPAMI, 42(2):502–508, 2019. 3, 6, 14
[50] Vicente Ordonez,
Girish Kulkarni,
and Tamara Berg.
Im2text: Describing images using 1 million captioned pho-
tographs. NeurIPS, 24:1143–1151, 2011. 3, 6, 14
[51] Gao Peng, Geng Shijie, Zhang Renrui, Ma Teli, Fang
Rongyao, Zhang Yongfeng, Li Hongsheng, and Qiao Yu.
Clip-adapter: Better vision-language models with feature
adapters. arXiv preprint arXiv:2110.04544, 2021. 3
[52] Bryan A Plummer, Liwei Wang, Chris M Cervantes,
Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazeb-
nik. Flickr30k entities: Collecting region-to-phrase corre-
spondences for richer image-to-sentence models. In ICCV,
pages 2641–2649, 2015. 6
[53] Di Qi, Lin Su, Jianwei Song, Edward Cui, Taroon Bharti,
and Arun Sacheti. Imagebert: Cross-modal pre-training with
large-scale weak-supervised image-text data. arXiv preprint
arXiv:2001.07966, 2020. 2, 8
[54] Xipeng Qiu, Tianxiang Sun, Yige Xu, Yunfan Shao, Ning
Dai, and Xuanjing Huang. Pre-trained models for natural
language processing: A survey. Science China Technological
Sciences, pages 1–26, 2020. 3
[55] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya

Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn-
ing transferable visual models from natural language super-
vision. arXiv preprint arXiv:2103.00020, 2021. 3
[56] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever.
Learning transferable visual
models from natural language supervision. In ICML, 2021.
2, 3
[57] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
Faster r-cnn: Towards real-time object detection with region
proposal networks. NeurIPS, 28:91–99, 2015. 3
[58] Zhang Renrui, Fang Rongyao, Gao Peng, Zhang Wei,
Li Kunchang, Dai Jifeng, Qiao Yu, and Li Hongsheng.
Tip-adapter:
Training-free clip-adapter for better vision-
language modeling. arXiv preprint arXiv:2111.03930, 2021.
3
[59] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural
machine translation of rare words with subword units. In
ACL, 2016. 4, 12
[60] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu
Soricut. Conceptual captions: A cleaned, hypernymed, im-
age alt-text dataset for automatic image captioning. In ACL,
pages 2556–2565, 2018. 3, 6, 14
[61] Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric
Wallace, and Sameer Singh. AutoPrompt: Eliciting knowl-
edge from language models with automatically generated
prompts. In EMNLP, 2020. 3
[62] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu
Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visual-
linguistic representations. arXiv preprint arXiv:1908.08530,
2019. 2, 3
[63] Chen Sun, Fabien Baradel, Kevin Murphy, and Cordelia
Schmid.
Learning video representations using contrastive
bidirectional transformer. arXiv preprint arXiv:1906.05743,
2019. 3
[64] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and
Cordelia Schmid. Videobert: A joint model for video and
language representation learning.
In CVPR, pages 7464–
7473, 2019. 3
[65] Hao Tan and Mohit Bansal.
Lxmert:
Learning cross-
modality encoder representations from transformers. arXiv
preprint arXiv:1908.07490, 2019. 2, 3
[66] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco
Massa, Alexandre Sablayrolles, and Herve Jegou. Training
data-efﬁcient image transformers & distillation through at-
tention.
In ICML, volume 139, pages 10347–10357, July
2021. 2
[67] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco
Massa, Alexandre Sablayrolles, and Herv´e J´egou. Training
data-efﬁcient image transformers & distillation through at-
tention. In ICML, pages 10347–10357. PMLR, 2021. 7, 14
[68] Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles,
Gabriel Synnaeve, and Herv´e J´egou. Going deeper with im-
age transformers. arXiv preprint arXiv:2103.17239, 2021.
2
[69] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
Polosukhin. Attention is all you need. In NeurIPS, pages
5998–6008, 2017. 1, 2
[70] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill,
Omer Levy, and Samuel R Bowman. Glue: A multi-task
benchmark and analysis platform for natural language un-
derstanding. arXiv preprint arXiv:1804.07461, 2018. 6, 7,
9
[71] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao
Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyra-
mid vision transformer: A versatile backbone for dense pre-
diction without convolutions. In ICCV, 2021. 2
[72] Han Xu, Zhang Zhengyan, Ding Ning, Gu Yuxian, Liu Xiao,
Huo Yuqi, Qiu Jiezhong, Zhang Liang, Han Wentao, Huang
Minlie, et al. Pre-trained models: Past, present and future.
arXiv preprint arXiv:2106.07139, 2021. 3
[73] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi,
Zi-Hang Jiang, Francis E.H. Tay, Jiashi Feng, and Shuicheng
Yan. Tokens-to-token vit: Training vision transformers from
scratch on imagenet.
In ICCV, pages 558–567, October
2021. 2
[74] Yanyi Zhang, Xinyu Li, Chunhui Liu, Bing Shuai, Yi Zhu,
Biagio Brattoli, Hao Chen, Ivan Marsic, and Joseph Tighe.
Vidtr: Video transformer without convolutions.
In ICCV,
pages 13577–13587, October 2021. 2
[75] Ziqi Zhang, Yaya Shi, Chunfeng Yuan, Bing Li, Peijin Wang,
Weiming Hu, and Zheng-Jun Zha. Object relational graph
with teacher-recommended learning for video captioning. In
CVPR, pages 13278–13288, 2020. 8
[76] Daquan Zhou, Bingyi Kang, Xiaojie Jin, Linjie Yang, Xi-
aochen Lian, Qibin Hou, and Jiashi Feng. Deepvit: Towards
deeper vision transformer. arXiv preprint arXiv:2103.11886,
2021. 2
[77] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei
Liu. Learning to prompt for vision-language models. arXiv
preprint arXiv:2109.01134, 2021. 3
[78] Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Ja-
son Corso, and Jianfeng Gao. Uniﬁed vision-language pre-
training for image captioning and vqa. In AAAI, 2020. 2,
8
[79] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov,
Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Align-
ing books and movies: Towards story-like visual explana-
tions by watching movies and reading books. In ICCV, pages
19–27, 2015. 6, 14

A. Tokenizer
Given the raw inputs from text, image, and video modali-
ties, modality-speciﬁc tokenizers are applied to generate the
input token sequences for the Transformer encoder. Here,
we use the BPE tokenizer [59] for text modality, the im-
age patch tokenizer [17] for image modality, and the tem-
poral frame patch tokenizer [6] for video modality. These
outputted tokens are attached with additional modality type
embeddings to identify which modality the raw input be-
longs to. The tokenizers are illustrated in Fig. 2.
Text Tokenizer. The BPE Tokenizer [59] is employed for
text modality. The text inputs are split into sub-words and
projected by a linear embedding layer. A learnable 1D posi-
tional embedding for text with a max length of 256 is added
to the word embeddings. To specify the input modality,
an additional trainable textual modality embedding <T> is
added to each token embedding. Note that all the text inputs
of our model share the same vocabulary.
Image Tokenizer.
The image patch tokenizer [17] is uti-
lized as the image tokenizer. The input images are resized to
224×224, and ﬂattened to a sequence of image patches with
shapes 16×16, which are further mapped by a linear projec-
tion. A sequence of learnable image positional embeddings
with a ﬁxed length 14 × 14 = 196, and an additional visual
modality embedding <V> are also added.
Video Tokenizer. The temporal frame patch tokenizer [6]
is used for video tokenization.
Each frame of the input
video is ﬂattened to image patches with shape 16×16. For
a video with N frames (N = 8 by default), the number
of tokens would be N × 14 × 14. The spatial positional
embeddings for images as well as a 1D temporal position
embedding with max length N = 8 are added to the video
embeddings. Besides, the additional visual modality em-
bedding <V> is added to each token embedding.
B. Implementation Details for Auto-regressive
Language Modeling
In the formulation of autoregressive tasks from previous
works, each input token attends to all previous tokens, in-
cluding itself, to predict the next word. Unlike previous
works, we use <SPE> to predict the current word. Fig. 4
shows how we achieve this. In the inference stage, <SPE>
is appended to the end of the predicted tokens as the in-
put. The corresponding output is used for prediction. In the
training stage, we append several <SPE> tokens after the
input sequence. Each <SPE> token is trained to predict a
word in the input sequence. The attention mask is designed
to make sure that the word tokens do not attend to <SPE>
tokens, and each <SPE> token only attends to itself and the
previous word tokens. In this way, the training stage and the
inference stage are aligned.
I
am
ok
SPE
I
am
ok
SPE
Query
Key & Value
Attend
Do not 
attend
Training
Stage
SPE
SPE
SPE
SPE
I
am
SPE
I
am
SPE
Query
Key & Value
Inference
Stage
Predicts
ok
Figure 4. The attention mask of autoregressive language modeling.
C. Prompt Tuning
Four groups of parameters are learnable in prompt-
tuning. They are <SPE>, layer normalization parameters,
prompt tokens, and linear heads. This section introduces
the usage and the number of parameters of each parameter
group.
Details Similar to the pre-training stage, <SPE> token is
shared for both inputs and targets.
Layer normalization
weights and biases in each Transformer layer and tokenizer
are tuned. Learnable prompts are added to each layer of
the Transformer encoder. Speciﬁcally, the input prompts
of each layer do not come from the output of the previ-
ous layer. They are random initialized learnable parame-
ters. For all prompt-tuning experiments, we use 10 learn-
able prompts for each layer on both inputs. The linear heads
only apply for classiﬁcation tasks. It takes the feature that
is to be classiﬁed as input, and returns a classiﬁcation logit.
The output probability is a linear combination of the proba-
bility from the similarity score and the linear head:
P(x, y) ∝α exp
 cos
 f(x), f(y)

/τ

+w⊤f(x)+b (4)

Number of Parameters
<SPE>
768
Layer Norm
41,472
Prompt
184,320
Linear head
768 * num classes
Table 9. The number of learnable parameters in the prompt-tuning
stage.
Note that the linear head only applies for classiﬁcation
tasks.
𝑓(𝑥)
<SPE>
Video
caption
𝑥!
"
Video + Sentence with masked tokens
𝑥#
"
𝑥!
$
𝑥%
$
𝑓(𝑦)
<SPE>
𝑦!
$
Vocabulary
<SPE>
<SPE>
VQA
𝑥!
&
Image + Question with masked tokens
𝑥#
&
𝑥!
$
𝑥#
$
𝑓(𝑦)
<SPE>
𝑦!
$
𝑦#
$
𝑦%
$
𝑦'
$
𝑦(
$
Answer
𝑓(𝑥)
<SPE>
Video-Text 
Retrieval
(V → T)
Video
𝑓(𝑦)
<SPE>
𝑦!
$
𝑦#
$
𝑦%
$
𝑦'
$
𝑦(
$
Caption
𝑓(𝑥)
<SPE>
Video-Text 
Retrieval
(T → V)
𝑥!
$
Caption
𝑥#
$
𝑥%
$
𝑥'
$
𝑥(
$
𝑓(𝑦)
<SPE>
Video
𝑥!
"
𝑥#
"
𝑥%
"
𝑥'
"
𝑥(
"
𝑥!
"
𝑥#
"
𝑥%
"
𝑥'
"
𝑥(
"
𝑓(𝑥)
<SPE>
𝑓(𝑥)
<SPE>
Natural 
Language 
Understanding 
𝑥!
$
𝑥#
$
𝑥%
$
𝑥'
$
𝑥(
$
𝑓(𝑦)
<SPE>
𝑦!
$
𝑦#
$
𝑦%
$
𝑦'
$
𝑦(
$
Single sentence or sentence pair
Class label
Class label
Figure 5. Input and target formats of our novel tasks. For each
task, the left column represents the format of input sequence x,
and the right column represents the format of the target sequence
y. f(x) and f(y) are used to calculate the joint probability dis-
tribution. Here, we have omitted the tokenizer and encoder for
concision.
where w and b are the weights and bias of the linear, and α
is a learnable scalar. w and b are initialized with 0 and α is
initialized with 1.
Number of Parameters Tab. 9 shows the number of param-
eters of each learnable component in prompt-tuning. For
tasks other than classiﬁcation, the number of parameters
is 227K in total. When the linear head is added, e.g., in
ImageNet-1k classiﬁcation task, the number is 995K, which
is still less than 1% of all the parameters in the pre-training
stage.
D. Formulation of Novel Tasks
The generic perceptual modeling makes it easy to con-
vert existing tasks into the uniﬁed task formulation of our
Uni-Perceiver. Fig. 5 illustrates the input and the output
formulations of our novel tasks, which we will describe in
details.
Video Caption.
Similar to image caption, video caption
is modeled as the autoregressive language modeling task
with video clues. In this task, the model predicts each word
based on the video and its previous words. The input set
X consists of the sequence concatenation of video and the
words that have been predicted, followed with a <SPE> to-
ken. The output features of the <SPE> token is used to
calculate the joint probability with each words in the the
vocabulary set Y. Then the word with the highest probabil-
ity is the predicted word at the current location. Addition-
ally, video caption follows the efﬁcient implementation of
autoregressive training introduced in Sec. B.
Video-Text Retrieval.
The Video-Text retrieval follows
the formulation of Image-Text retrieval task, except that the
image sequence is replaced by the video sequence. Speciﬁ-
cally, the input sets X and Y are composed of video and text
sequences respectively. Each sequence in X and Y also has
a <SPE> token at the beginning. We use the output feature
at the <SPE> token as the ﬁnal representation of the input
video or the text to calculate the joint probability distribu-
tion.
Visual Question Answering. We formulate the VQA task
as a special case of masked language modeling with image
clues. The input x ∈X is the combination of image and
question sequences. It should be noted that each question
ends with a “?” mark and a <SPE> token is appended to the
end of the question sequence. The Y set consists of candi-
date answer sequences, in which each begins with a <SPE>
token too. The joint probability distribution between X and
Y can be calculated by using the output feature from <SPE>
tokens.
Natural Language Understanding.
The formulation of
natural language understanding task is similar to that of
image classiﬁcation task.
X denotes the set of the in-
put single sentence or the sentence-pair, and Y is the set
contains the textual class label.
For example, in SST-2
[87], x denotes the sentence sequence of movie review and
y ∈Y = {great, terrible} is the sentiment label. In
MRPC [81], x instead is the sequence combination of sen-
tence pairs extracted from news, and y ∈Y = {Yes, No}
is the label to indicate whether the sentence pair are seman-
tically equivalent. We also add <SPE> tokens at the begin-
ning of the sequences x and y, of which output features are
used to computed joint probability.
E. Extra pre-training details
Sampling Weight & Batch Size & Loss
Tab. 10 lists the
batch size and sampling weight of each task and dataset in
the pre-training stage. We use cross-entropy loss for lan-
guage modeling tasks.
The other tasks are trained with
cross-entropy loss with 0.1 label smoothing.
The loss
weight of video classiﬁcation is 0.05, which helps stabilize
training in our experiments. The other loss weights are 1.0
by default.
For retrieval tasks like image-text retrieval, we use train-

Task
Dataset
Batch Size
Sampling Weight
Image Classiﬁcation
ImageNet-21k [15]
64
0.333
Video Classiﬁcation
Kinetics-700 [27]
4
0.0925
Moments in Time [49]
24
0.0185
Auto-encoding LM
Books&Wiki [79]
64
0.07775
YFCC [26]
64
0.02778
CC12M [8]
64
0.02778
CC3M [60]
64
0.01389
Visual Genome [30]
64
0.01389
COCO Caption [11]
64
0.01389
SBU [50]
64
0.01389
PAQ [35]
512
0.0222
Auto-regressive LM
Books&Wiki [79]
64
0.07775
YFCC [26]
56
0.02778
CC12M [8]
56
0.02778
CC3M [60]
56
0.01389
Visual Genome [30]
56
0.01389
COCO Caption [11]
56
0.01389
SBU [50]
56
0.01389
PAQ [35]
400
0.0222
Retrieval
YFCC [26]
128
0.02778
CC12M [8]
128
0.02778
CC3M [60]
128
0.01389
Visual Genome [30]
128
0.01389
COCO Caption [11]
128
0.01389
SBU [50]
128
0.01389
PAQ [35]
512
0.0222
Table 10. Ingredients and hyper-parameters for our pre-training.
ing samples in the same batch as negative samples, whose
typical size is 127 except the PAQ dataset. Note that we
do not use memory bank and do not gather feature across
GPU devices to provide more negative samples, which may
further promote the performance of retrieval tasks.
Data Augmentation We apply augmentation techniques to
image and video modalities to avoid overﬁtting. For images
in ImageNet-21k dataset, we apply augmentation same as
[67]. Rand-Aug[80], random erasing [91], mixup [90] and
cutmix [89] are used simultaneously. For images in other
datasets, we resize the images to the short edge size of 256,
and then a 224 × 224 region is cropped randomly from the
resized images during training. During inference, the ran-
dom crop is replaced with center crop operation. For all
Video inputs, we apply the same augmentations used in [6].
We use clips of size 8 × 224 × 224 for Kinetics-700 and
Kinetics-400, and 3 × 224 × 224 for Moments in Time.
The temporal sample rate is 32. We use a single temporal
clip. During training, the start frame is randomly picked if
the video is longer than the clip. In the training stage, we
ﬁrst resize the shorter side of the video to a random value
in [256, 320], then we randomly sample a 224 × 224 crop.
In the test stage, the short side of the video is resized to
224. For classiﬁcation tasks, We use 3 spatial crops with
size 224 × 224 to cover a larger range of content and av-
erage their logits for evaluation. For video captioning task,
we use center crop after resizing.
Data Parallel for Vocabulary and Class Labels
Naive
implementation of language modeling and ImageNet-21k
classiﬁcation is impractical due to memory limitation. In
language modeling, we need to compare the feature of a
<SPE> token in a sequence to the feature of each token in
our tokenizer. Since the vocabulary size is large, we apply
data parallel for the vocabulary set. A similar method is
used for class labels of ImageNet-21k.
Removing Overlap
For the training set of K700 partici-
pating in pre-training, we remove those videos overlapping
with validation set of K400.
F. Licences of Datasets
ImageNet-21K [15] is subject to the ImageNet terms of use
[88].
Kinetics-700 [86] & Kinetics-400 [27] The kinetics dataset
is licensed by Google Inc. under a Creative Commons At-
tribution 4.0 International License.
BooksCorpus [79] Replicate Toronto BookCorpus is open-
source and licensed under GNU GPL, Version 3.
Wikipedia Most of Wikipedia’s text is co-licensed un-
der the Creative Commons Attribution-ShareAlike 3.0 Un-
ported License (CC BY-SA) and the GNU Free Documen-
tation License (GFDL) (unversioned, with no invariant sec-
tions, front-cover texts, or back-cover texts).
Some text
has been imported only under CC BY-SA and CC BY-SA-
compatible license and cannot be reused under GFDL.
YFCC [26] All the photos and videos provided in YFCC
dataset are licensed under one of the Creative Commons
copyright licenses.
CC12M [8] is licensed under the Terms of Use of Concep-
tual 12M [84].
CC3M [60] is licensed under the Conceptual Captions
Terms of Use [85].
Visual Genome [30] is licensed under a Creative Commons
Attribution 4.0 International License [83].
COCO Caption [11] The images are subject to the Flickr
terms of use [82].
SBU Caption [50] The images are subject to the Flickr
terms of use [82].
PAQ [35] is licensed under the Attribution-NonCommercial
4.0 International License.
Appendix References
[80] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V
Le. Randaugment: Practical automated data augmentation
with a reduced search space. In CVPR Workshops, pages
702–703, 2020.

[81] William B Dolan and Chris Brockett. Automatically con-
structing a corpus of sentential paraphrases.
In Proceed-
ings of the Third International Workshop on Paraphrasing
(IWP2005), 2005.
[82] Inc. Flickr. Flickr terms & conditions of use. https://
www.flickr.com/help/terms.
[83] Ranjay Krishna. Visual genome terms & conditions of use.
https://visualgenome.org/about.
[84] Google LLC. Conceptual 12m terms & conditions of use.
https : / / github . com / google - research -
datasets / conceptual - 12m / blob / main /
LICENSE.
[85] Google LLC.
Conceptual captions terms & condi-
tions of use.
https://github.com/google-
research - datasets / conceptual - captions /
blob/master/LICENSE.
[86] Lucas Smaira, Jo˜ao Carreira, Eric Noland, Ellen Clancy,
Amy Wu, and Andrew Zisserman.
A short note on the
kinetics-700-2020 human action dataset.
arXiv preprint
arXiv:2010.10864, 2020.
[87] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang,
Christopher D Manning, Andrew Y Ng, and Christopher
Potts. Recursive deep models for semantic compositional-
ity over a sentiment treebank. In Proceedings of the 2013
conference on empirical methods in natural language pro-
cessing, pages 1631–1642, 2013.
[88] Princeton University and Stanford University.
Imagenet
terms & conditions of use. https://image-net.org/
download.
[89] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk
Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regu-
larization strategy to train strong classiﬁers with localizable
features. In ICCV, pages 6023–6032, 2019.
[90] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and
David Lopez-Paz. mixup: Beyond empirical risk minimiza-
tion. arXiv preprint arXiv:1710.09412, 2017.
[91] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and
Yi Yang. Random erasing data augmentation. In AAAI, vol-
ume 34, pages 13001–13008, 2020.
