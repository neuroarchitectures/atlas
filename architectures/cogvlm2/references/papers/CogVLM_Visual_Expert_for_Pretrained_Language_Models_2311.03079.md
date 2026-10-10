# Paper (Wang et al. 2023)

> Source: `https://arxiv.org/abs/2311.03079`

---

CogVLM: Visual Expert for Pretrained Language Models
Weihan Wang * 1 ‡ Qingsong Lv * 2 Wenmeng Yu 2 Wenyi Hong 1 ‡ Ji Qi 1 ‡ Yan Wang 2 Junhui Ji 2
Zhuoyi Yang 1 ‡ Lei Zhao 2 Xixuan Song 1 ‡ Jiazheng Xu 1 ‡ Keqin Chen 3 ‡ Bin Xu 1 Juanzi Li 1 Yuxiao Dong 1
Ming Ding 2 Jie Tang 1
Abstract
We introduce CogVLM, a powerful open-source
visual language foundation model.
Different
from the popular shallow alignment method
which maps image features into the input space
of language model, CogVLM bridges the gap
between the frozen pretrained language model
and image encoder by a trainable visual ex-
pert module in the attention and FFN layers.
As a result, CogVLM enables a deep fusion
of vision language features without sacrificing
any performance on NLP tasks. CogVLM-17B
achieves state-of-the-art performance on 17 clas-
sic cross-modal benchmarks, including 1) im-
age captioning datasets: NoCaps, Flicker30k, 2)
VQA datasets: OKVQA, TextVQA, OCRVQA,
ScienceQA, 3) LVLM benchmarks:
MM-
Vet, MMBench, SEED-Bench, LLaVABench,
POPE, MMMU, MathVista, 4) visual grounding
datasets: RefCOCO, RefCOCO+, RefCOCOg,
Visual7W. Codes and checkpoints are available at
https://github.com/THUDM/CogVLM.
1. Introduction
Vision language models are versatile and powerful. Many
vision and cross-modality tasks can be formulated as next
token prediction, e.g., image captioning (Agrawal et al.,
2019), visual question answering (Antol et al., 2015), visual
grounding (Yu et al., 2016) and even segmentation (Chen
et al., 2022a). Useful abilities like in-context learning (Tsim-
poukelli et al., 2021; Sun et al., 2023a; Alayrac et al., 2022)
also emerge along with the improvement of downstream
tasks when scaling up VLMs. However, to train a large
language model is already non-trivial, and it is more chal-
lenging to train a VLM from scratch with the same NLP
*Equal contribution
1Tsinghua University 2Zhipu AI ‡Done
as intern at Zhipu AI 3Beihang University.
Correspondence
to:
Ming Ding <ming.ding@zhipuai.cn>, Jie Tang < ji-
etang@tsinghua.edu.cn>.
LLaVA-Bench
POPE
SEED
MMBench
MM-Vet
NoCaps
ScienceQA
OKVQA
TextVQA
VQAv2
Flickr
RefCOCOg
RefCOCO+
RefCOCO
MathVista
MMMU
41.75
59.5
77.25
84.62
86.25
87.88
57.5
65.0
72.5
45.5
61.0
76.5
31.5
41.0
50.5
122.0
125.0
128.0
67.0
79.0
91.0
51.5
58.0
64.5
53.5
62.0
70.5
60.75
71.5
82.25
75.0
85.0
95.0
80.0
85.0
90.0
79.5
84.0
88.5
86.25
89.5
92.75
26.25
30.5
34.75
30.25
35.5
40.75
N/A
Shikra
Qwen-VL
LLaVA-1.5
InstructBLIP
mPLUG-Owl2
Unified-IO2
Emu2
CogVLM
Figure 1. The performance of CogVLM on a broad range of multi-
modal tasks in comparison with existing approaches.
performance as well-trained pure language models like
LLaMA2 (Touvron et al., 2023). Therefore, it is natural
to investigate how to train a VLM from an off-the-shelf
pretrained language model.
The popular shallow alignment methods represented by In-
structBLIP (Li et al., 2023b) and MiniGPT-4 (Zhu et al.,
2023) connect a frozen pretrained vision encoder and lan-
guage model via a trainable Q-Former or a linear layer,
mapping the image features into the input embedding space
of the language model. This method converges rapidly, but
its performance is noticeably inferior to that of LLaVA-
1.5 with trainable language parameters, despite their model
sizes and training datasets being almost identical.
The primary challenge in the performance of shallow align-
ment methods within VLMs can be attributed to the lack
of deep fusion between visual and linguistic data. Shallow
alignment methods struggle because they rely on ‘frozen’
language model weights, which are intrinsically trained to
process text tokens. This presents a significant mismatch
1
arXiv:2311.03079v2  [cs.CV]  4 Feb 2024

CogVLM: Visual Expert for Pretrained Language Models
Referring Expression Comprehension
 Q: Describe this image in detail.
CogVLM: The image showcases a railway platform labeled 
'Platform 2'. A green train is approaching, with steam 
emanating from its engine. On the platform, there are 
several people, including a woman in a pink shirt, a child 
with a backpack, and a man in a green shirt. There's a 
chalkboard sign on the platform that reads: 'Please Use The 
Footbridge To Cross The Line If You Require Assistance. 
Please Ask The Station Staff.' In the background, there's a 
signal post and some trees.
Detailed Description
 Q: Can you explain this meme?
CogVLM: This meme is a humorous take on the 
idea that certain drugs, like cocaine, are believed 
to have no side effects. The meme uses a ﬁctional 
meeting of various mythical creatures, each with a 
unique and exaggerated appearance, to represent 
the different effects of cocaine. The text at the top 
of the image, 'Publicly, we are still saying there 
are no side effects,' is a statement often made by 
pharmaceutical companies or health organizations 
about certain drugs. The creatures in the meeting, 
with their various and often comical expressions 
and postures, humorously represent the different 
'side effects' of cocaine, suggesting that the drug 
has a wide range of unintended effects.
Meme Understanding
 Q: From the perspective of a photographer, 
analyze why this picture was taken well?
CogVLM: From a photographer's perspective, several 
elements contribute to the quality of this image. The 
composition is balanced, with the hot air balloon taking up 
a signiﬁcant portion of the frame, yet still allowing the 
surrounding environment to be visible. The use of lighting, 
especially the soft glow from the balloon and the car's 
headlights, creates a warm and inviting atmosphere. The 
depth of ﬁeld is shallow, focusing on the subjects in the 
foreground while softly blurring the background, drawing 
attention to the main elements of the scene.
Visual Reasoning
 Q: What is the difference of largest and 
smallest bar? Answer it step by step.
CogVLM: The largest bar represents Iceland with a 
percentage of 100%. The smallest bar represents 
Cambodia with a percentage of 52.4%. To ﬁnd the 
difference, subtract the smallest percentage from the 
largest: 100% - 52.4% = 47.6%.
Chart Understanding & Reasoning
 Q: How many goals did the person in the 
picture score in the 2018 World Cup?
CogVLM: The person in the picture is Cristiano Ronaldo. 
He is the captain of the Portuguese team and has 
participated in the 2018 World Cup. In the 2018 World 
Cup, Cristiano Ronaldo scored 4 goals.
World Knowledge
 Q1: What did Musk say?
CogVLM: Elon Musk in the tweet said: "Fight Recap: I 
joked on X about ﬁghting Zuck, Zuck then said, 'Send ME 
LOCATION.' Italy graciously offered a Colosseum, but 
Zuck declined. I suggested his home as a "safe space," but 
tragically, he was already "traveling." Is there anywhere he 
will ﬁght?"
 Q2: Do you think Zuck is really traveling?
CogVLM: Based on the content of the tweet, it seems that 
Zuck is not really traveling, but rather pretending to be 
traveling in order to avoid the conﬂict.
OCR-Free Reasoning
CogVLM: A piggy bank [[302,198,694,857]] with a 
smile [[400,520,582,643]] on it is surrounded by coins 
[[028,820,986,997]], toy houses [[052,697,174,892;
526,763,636,952;168,697,286,883;738,697,852,895]], 
and greenery.
 Q: Can you provide a description of the image and include 
the coordinates [[x0,y0,x1,y1]] for each mentioned object?
Grounding with Caption
CogVLM: Find the ﬂowers [[328,519,436,780]]. Check the 
girl holding it, got the girl [[116,069,450,927]]. Check the 
clothes of it, got the clothes [[134,309,418,852]]. The 
clothes [[134,309,418,852]] are blue. So the answer is blue.
 Q1: What color are the clothes of the girl whose 
hands are holding ﬂowers? Let's think step by step.
CogVLM: Find the not(ﬂowers) hands [[654,637,710,728]]. 
Check the girl with it, got the girl [[526,109,880,946]]. 
Check the jeans of it, got the jeans [[530,613,880,871]]. The 
jeans [[530,613,880,871]] are gray. So the answer is gray.
 Q2: What color are the jeans of the girl whose hands 
are not holding ﬂowers? Let's think step by step.
Grounding Visual Question Answering
Figure 2. Samples generated by CogVLM.
2

CogVLM: Visual Expert for Pretrained Language Models
Figure 3. MMLU score and training loss over multimodal pre-
training phase. When directly training the language part of the
VLM using the LAION dataset, the model’s score on the pure text
dataset MMLU rapidly decreases, dropping to 24.9 at 2500 steps.
Figure 4. The architecture of CogVLM. (a) The illustration about
the input, where an image is processed by a pretrained ViT and
mapped into the same space as the text features. (b) The Trans-
former block in the language model. The image features have a
different QKV matrix and FFN. Only the purple parts are trainable.
issue, as visual features lack a direct equivalent in the tex-
tual input space. Consequently, when these visual features
undergo multi-layer transformations, they tend to deviate
from the expected input distribution of the deeper language
model layers. This misalignment is particularly evident in
tasks like image captioning, where the specificity of a task
– such as writing style and caption length – can only be
superficially encoded into visual features through shallow
methods.
A common strategy, as seen in PaLI (Chen et al., 2022b)
and Qwen-VL (Bai et al., 2023), involves direct training
of LLM during the pre-training or supervised fine-tuning
(SFT) phase. However, this approach can compromise the
models’ generalizability, particularly for tasks focused on
textual outputs. Conventionally, LLMs are pretrained on
extensive text-only datasets (Raffel et al., 2020), leading to
a significant divergence in data distribution when compared
to image-text pair datasets like LAION (Schuhmann et al.,
2022) and COYO (Byeon et al., 2022). This shift often
results in catastrophic forgetting, a phenomenon where the
model’s proficiency in its original domain deteriorates. This
issue is evident in Figure 3, which shows a marked decline in
MMLU (Hendrycks et al., 2020) score as the model becomes
more attuned to the LAION dataset, thus validating our
hypothesis. This trend is not isolated; similar effects have
been observed in models like PaLM-E (Driess et al., 2023)
and Flamingo (Alayrac et al., 2022). For instance, adapting
an 8B parameter language model for VLM pretraining can
lead to an 87.3% reduction in natural language generation
(NLG) performance (Driess et al., 2023).
The discussion above raises an important question: is it
possible to retain the NLP capabilities of the large language
model while adding top-notch visual understanding abilities
to it?
CogVLM gives a “yes” answer. CogVLM instead adds
a trainable visual expert to the language model. In each
layer, the image features in the sequence use a new QKV
matrix and MLP layer with the text features. Visual ex-
pert doubles the number of parameters while keeping the
FLOPs the same. Since all the parameters in the original lan-
guage model are fixed, the behaviors are the same as in the
original language model if the input sequence contains no
image. This inspiration arises from the comparison between
P-Tuning (Liu et al., 2023f) and LoRA (Hu et al., 2021) in
efficient finetuning, where p-tuning learns a task prefix em-
bedding in the input while LoRA adapts the model weights
in each layer via a low-rank matrix. As a result, LoRA
performs better and more stable. A similar phenomenon
might also exist in VLM, because in the shallow alignment
methods, the image features act like the prefix embedding
in P-Tuning.
Our contributions in this work are as follows:
• We introduce the CogVLM model, which deeply inte-
grates visual and linguistic features while retaining the
full capabilities of a pretrained large language model.
CogVLM-17B, trained from Vicuna-7B, achieves state-
of-the-art across 17 classic cross-modal benchmarks.
• Through extensive ablation studies, we validated the
effectiveness of our proposed visual expert module and
the importance of deep fusion. We further delved into
multiple critical factors in multimodal pertaining, in-
cluding the scale of visual encoder, variants of attention
mask, the most impactful parameters in VLMs, and the
necessity of incorporating self-supervised image loss,
etc.
3

CogVLM: Visual Expert for Pretrained Language Models
• We have made the weights of CogVLM and the dataset
used in the SFT phase available to the public. We
anticipate that the open sourcing of CogVLM will sig-
nificantly contribute to the research and industrial ap-
plication of visual understanding.
2. Method
2.1. Architecture
CogVLM model comprises four fundamental components:
a vision transformer (ViT) encoder, an MLP adapter, a pre-
trained large language model (GPT), and a visual expert
module. Figure 4 shows an overview of the CogVLM ar-
chitecture. The components’ design and implementation
details are provided below:
ViT encoder. We utilize pretrained EVA2-CLIP-E (Sun
et al., 2023b) in CogVLM-17B. Note that the final layer of
ViT encoder is removed because it specializes in aggregating
the [CLS] features for contrastive learning.
MLP adapter. To map the output of ViT into the same
space as the text features from word embedding, we use an
MLP adapter, a two-layer MLP (SwiGLU (Shazeer, 2020)).
For implementation convenience, all image features share
the same position id in the language model.
Pretrained large language model. CogVLM’s model de-
sign is compatible with any off-the-shelf GPT-style pre-
trained large language model. Specifically, CogVLM-17B
adopts Vicuna1.5-7B (Chiang et al., 2023) for further train-
ing. A causal mask is applied to all the attention operations,
including the attention between image features.
Visual expert module. We add a visual expert module to
each layer to enable deep visual-language feature alignment.
Specifically, the visual expert module in each layer consists
of a QKV matrix and an MLP in each layer. The shapes
of the QKV matrix and MLP are identical to those in the
pretrained language model and initialized from them. The
motivation is that each attention head in the language model
captures a certain aspect of semantic information, while a
trainable visual expert can transform the image features
to align with the different heads, therefore enabling deep
fusion.
Formally, suppose that the input hidden states of an attention
layer are X ∈RB×H×(LI+LT )×D, where B is the batch
size, LI and LT are the lengths of image and text sequences,
H is the number of attention heads, and D is the hidden
size. In the attention with visual expert, X is first split as
image hidden states XI and text hidden states XT , and the
attention is computed as:
Attention(X, WI, WT ) = softmax(Tril(QKT )
√
D
)V,
(1)
Q = concat(XIW Q
I , XT W Q
T ),
(2)
K = concat(XIW K
I , XT W K
T ),
(3)
V = concat(XIW V
I , XT W V
T ),
(4)
where WI, WT are the QKV matrices of the visual ex-
pert and original language model, and Tril(·) means lower-
triangular mask. The visual expert in FFN layers performs
similarly,
FFN(X) = concat(FFNI(XI), FFNT (XT )),
(5)
where FFNI and FFNT are the FFN of the visual expert and
original language model.
Position embedding. In the RoPE within LLM, we allow
all visual tokens to share a single position id, as they already
encapsulate positional information when inputted into the
ViT. This approach mitigates the impact of remote atten-
uation between tokens in the LLM. Given that an image
can occupy hundreds to thousands of tokens, and a typical
input sequence is structured as ‘<image embed> query’,
using conventional positional encoding would result in ex-
cessively lengthy encoding sequences. Moreover, it would
lead the query to focus more on the image sequences closer
to it, namely the lower part of an image.
2.2. Pretraining
Data. The image-text pairs for pretraining are all publicly
available, including LAION-2B and COYO-700M. After
removing the broken URLs, NSFW images, images with
noisy captions, images with political bias and images with
an aspect ratio > 6 or < 1/6, about 1.5B images are left for
pretraining.
We also crafted a visual grounding dataset of 40M images.
Each noun in the image caption is associated with bounding
boxes to indicate the positions in the image. The construc-
tion process basically follows (Peng et al.), which extracts
nouns via spaCy (Honnibal & Johnson, 2015) and predicts
the bounding boxes using GLIPv2 (Zhang et al., 2022). The
image-text pairs are sampled from LAION-115M, a subset
of LAION-400M filtered by (Li et al., 2023b). We filter and
retain a subset of 40 million images to ensure that over 75%
of images contain at least two bounding boxes.
Training. The first stage of pretraining is for image cap-
tioning loss, i.e. next token prediction in the text part. We
train the CogVLM-17B model on the 1.5B image-text pairs
introduced above for 120,000 iterations with a batch size
of 8,192. The second stage of pretraining is a mixture of
image captioning and Referring Expression Comprehension
(REC). REC is a task to predict the bounding box in the im-
age given the text description of an object, which is trained
in the form of VQA, i.e., Question: Where is the object?
and Answer: [[x0, y0, x1, y1]]. Both x and y coordinates
4

CogVLM: Visual Expert for Pretrained Language Models
range from 000 to 999, meaning the normalized position
in the image. We only consider the loss of the next token
prediction in the “Answer” part. We pretrain the second
stage for 60,000 iterations with a batch size of 1,024 on the
text-image pairs and visual grounding datasets introduced
above. During the final 30,000 iterations, we change the
input resolution from 224 × 224 to 490 × 490. The total
number of trainable parameters is 6.5B.
2.3. Alignment
In the instruction alignment phase, we trained two gen-
eralist models: CogVLM-Chat and CogVLM-Grounding.
CogVLM-Chat accepts natural language inputs and outputs,
while CogVLM-Grounding accepts inputs and outputs with
bounding boxes.
CogVLM-Chat. In our study, we integrated data from a va-
riety of open-source visual question-answering datasets, in-
cluding VQAv2 (Antol et al., 2015), OKVQA (Marino et al.,
2019), TextVQA (Singh et al., 2019), OCRVQA (Mishra
et al., 2019), ScienceQA (Lu et al., 2022), as well as
datasets formatted as multi-turn dialogues such as LLaVA-
Instruct (Liu et al., 2023c), LRV-Instruction (Liu et al.,
2023a), LLaVAR (Zhang et al., 2023). We then conducted
unified instruction-supervised fine-tuning (SFT) across these
diverse datasets. The integrity and quality of SFT data are
crucial; notably, the LLaVA-Instruct dataset, initially gen-
erated through a language-only GPT-4 pipeline, contained
certain inaccuracies. We meticulously corrected these errors
through manual inspection and annotation to ensure data
quality.
VQA datasets typically feature concise, often one-word an-
swers, contrasting with the dialogue datasets that provide
detailed responses with extensive reasoning. To accommo-
date this variability, we employed prompts formatted as
Question: Short answer: for concise responses and Ques-
tion: Answer: for extended discourse in the SFT phase.
During training, the model underwent 6000 iterations with
a learning rate of 1e-5 and a batch size of 1024. To en-
hance and ensure the stability of the training, we activated
the visual encoder’s parameters and adjusted its learning
rate to be one-tenth of that used for the remaining training
parameters.
CogVLM-Grounding. In order to endow our model with
consistent, interactive visual grounding capabilities, we col-
lect a high-quality dataset covering 4 types of grounding
data: (1) Grounded Captioning (GC) - image captioning
datasets where each noun phrase within the caption is fol-
lowed by the corresponding referential bounding boxes; (2)
Referring Expression Generation (REG) - image-oriented
datasets that each bounding box in the image is annotated
with a descriptive textual expression that accurately charac-
terizes and refers to the content within the specific region;
(3) Referring Expression Comprehension (REC) - text-
oriented datasets that each textual description is annotated
with multiple referential links associating the phrases with
corresponding boxes; (4) Grounded Visual Question An-
swering (GroundedVQA) - VQA-style datasets where the
questions may contain region references in a given image.
The sources of grounding data are all publicly available,
including Flickr30K Entities (Plummer et al., 2015), Ref-
COCO (Kazemzadeh et al., 2014; Mao et al., 2016; Yu et al.,
2016), Visual7W (Zhu et al., 2016), VisualGenome (Krishna
et al., 2017) and Grounded CoT-VQA (Chen et al., 2023a).
[box] in this section is in the format of [[x0, y0, x1, y1]].
It is noteworthy that the curated datasets exhibit a versa-
tility of visual grounding capabilities, and many datasets
can be adapted and repurposed across different tasks. For
instance, grounded captioning datasets can be reformulated
to suit REG and REC tasks. Taking the example of “A
man [box1] and a woman [box2] are walking together.”, this
can be reframed into question answering pairs like (“De-
scribe this region [box2].”, “A woman.”) and (“Where is
the man?”, “[box1]”). Similarly, REC datasets can be trans-
lated into REG tasks by switching the input and output, and
vice versa. However, certain conversions might lead to am-
biguities. For example, when presented with the isolated
query “Where is another man?” from the caption “A man
[box1] is running, while another man [box2] is looking.”,
the distinction between [box1] and [box2] becomes unclear,
potentially leading to errors.
3. Experiments
To rigorously validate the superior performance and robust
generalization of our base model, we conduct quantitative
evaluations on an array of multi-modal benchmarks. These
benchmarks can be categorized into three broad areas cover-
ing a comprehensive range of measurement1:
• Image Captioning. The main purpose of these tasks
is to generate textual captions summarizing the major
content of a given image. We utilize prominent datasets
including NoCaps (Agrawal et al., 2019), COCO (Lin
et al., 2014), Flickr30K (Plummer et al., 2015), and
TextCaps (Sidorov et al., 2020) for evaluation.
• Visual Question Answering. The VQA tasks require
models to answer questions that may focus on distinct
visual contents based on the given image. Our assess-
ment covers diverse datasets, including VQAv2 (An-
tol et al., 2015), OKVQA (Marino et al., 2019),
TextVQA (Singh et al., 2019), OCRVQA (Mishra et al.,
2019) and ScienceQA (Lu et al., 2022).
1Detailed summary of all benchmarks and corresponding met-
rics are available at Appendix A.2.
5

CogVLM: Visual Expert for Pretrained Language Models
Table 1. Performance on Image Captioning benchmarks. All tasks use CIDEr as the evaluation metric. OOD refers to out-of-domain
test set. Karp. refers to the Karpathy test split.
Method
Train
Data
NoCaps val
NoCaps test
Flickr
COCO
TextCaps
OOD
overall
OOD
overall
Karp.
Karp.
test
Human
-
95.7
87.1
91.6
85.3
-
-
125.1
VinVL (Zhang et al., 2021)
8.9M
83.8
94.3
78.0
92.5
-
130.8
-
SimVLM (Wang et al., 2021)
1.8B
115.2
112.2
109.5
110.3
-
143.3
-
CoCa (Yu et al., 2022)
4.8B
-
122.4
-
120.6
-
143.6
-
LEMON (Hu et al., 2022)
2B
120.2
117.3
110.1
114.3
-
139.1
-
Flamingo (Alayrac et al., 2022)
2.3B
-
-
-
-
67.2
138.1
-
Prismer (Liu et al., 2023d)
12.7M
113.5
112.9
-
110.8
-
136.5
-
BLIP-2 (Li et al., 2023b)
129M
124.8
121.6
-
-
-
144.5
-
InstructBLIP (Dai et al., 2023)
129M
-
123.1
-
-
82.4
-
-
UniversalCap (Cornia et al., 2021)
35M
123.4
122.1
114.3
119.3
-
143.4
-
GIT (Wang et al., 2022a)
0.8B
127.1
125.5
122.0
123.4
49.6
144.8
138.2
GIT2 (Wang et al., 2022a)
12.9B
130.6
126.9
122.3
124.8
50.7
145.0
145.0
Qwen-VL (Bai et al., 2023)
1.4B
-
121.4
-
-
85.8
-
-
PaLI-17B (Chen et al., 2022b)
1.6B
-
127.0
-
124.4
-
149.1
135.4
PaLI-X-55B (Chen et al., 2023b)
-
-
126.3
-
124.3
-
149.2
147.0
CogVLM (ours)
1.5B
132.6
128.3
128.0
126.4
94.9
148.7
144.9
• LVLM Benchmarks. LVLM benchmarks are primar-
ily employed to assess the advanced capabilities of
large multimodal models, such as object recognition
and localization, OCR, visual description, and visual
knowledge reasoning. We conduct multidimensional
evaluations of the models on datasets including MM-
Vet (Yu et al., 2023), MMBench (Liu et al., 2023g),
SEED-Bench (Li et al., 2023a), LLaVA-Bench (Liu
et al., 2023c), POPE (Li et al., 2023c), MMMU (Yue
et al., 2023) and MathVista (Lu et al., 2023).
• Visual Grounding. Visual grounding involves a set
of tasks that establish referential links between textual
mentions in a sentence and specific regions in an im-
age. We evaluate our model on the typical datasets,
including Visual7w (Zhu et al., 2016), RefCOCO (Liu
et al., 2017), RefCOCO+, and RefCOCOg to ensure
completeness.
3.1. Image Captioning
We evaluate the image captioning capability of our pre-
trained base model on the aforementioned four benchmarks.
In a zero-shot evaluation on the Nocaps and Flickr datasets,
we assess the precision of our model in describing long-
tail visual concepts. Additionally, we present results from
finetuning on the COCO and TextCaps datasets.
The detailed performance is shown in Table 1. Overall, our
model achieves the SOTA or compatible performance across
the board. Specifically, on the NoCaps benchmark, our base
model outperforms the previous best method, GIT2, across
four splits with a maximum of 5.7 points in the out-domain
set while only consuming 10% of the pretraining data (1.5B
vs 12.9B). On the Flickr benchmark, our model achieves a
SOTA score of 94.9 surpassing the concurrently released
Qwen-VL model by 9.1 points. These results demonstrate
the remarkable capability and robustness of our pretrained
model on the image captioning task. We also evaluate our
model on the COCO (Lin et al., 2014) and TextCaps, where
the latter is specifically designed to integrate the textual in-
formation of the given image into captions. Though training
without the dedicated OCR data, encouragingly, our base
model reveals a significant text-reading ability and obtains
a competitive performance with PaLI-X-55B, and outper-
forms the previous best model of the same scale, PaLI-17B,
by 9.1 points score.
3.2. Visual Question Answering
As illustrated in Table 2, our CogVLM model demon-
strates outstanding performance and a significant lead
over models of similar parameter scale across a variety
of tasks, including daily-life image question-answering
dataset VQAv2, text-intensive image question-answering
datasets such as TextVQA and OCRVQA, and knowledge-
demanding datasets like OKVQA and ScienceQA. This
success showcases the model’s robust generalization capa-
bilities and potential across diverse domains.
3.3. LVLM Benchmarks
Our findings, detailed in Table 2, demonstrate that CogVLM
achieved state-of-the-art results in all 7 LVLM-benchmarks,
markedly surpassing all other models.
It also outper-
formed multimodal models that utilized larger language
models, such as LLava1.5 with Vicuna-13B and Emu-2 with
6

CogVLM: Visual Expert for Pretrained Language Models
Table 2. Generalist performance on VQA and LVLM benchmarks. * donates the dataset has been trained during SFT stage. We
compared with the latest state-of-the-art generalist models, including MiniGPT-4 (Zhu et al., 2023), IDEFICS-Instruct (Laurenc¸on et al.,
2023), OpenFlamingo (Awadalla et al., 2023), DreamLLM (Dong et al., 2023), InstructBLIP (Dai et al., 2023), Fuyu (Bavishi et al., 2023),
Qwen-VL (Bai et al., 2023), LLaVA-1.5 (Liu et al., 2023b), mPLUG-Owl2 (Ye et al., 2023), SPHINX (Lin et al., 2023b), Emu2 (Sun
et al., 2023a).
Method
LLM
VQA
LVLM-Benchmark
VQAv2 OKVQA TextVQA OCRVQA ScienceQA MM-Vet SEED MMBench LLaVA POPE MMMU MathVista
MiniGPT-4
Vicuna-7B
-
-
0.6
-
39.6
22.1
47.4
23.0
45.1
-
-
23.1
IDEFICS-Instruct LLaMA-65B
37.4
36.9
32.9
-
61.8
39.7
53.2
54.5
56.9
-
-
26.2
OpenFlamingo
MPT-7B
53.0
38.3
28.3
-
44.8
24.8
42.7
5.7
34.2
-
26.3
18.6
DreamLLM
Vicuna-7B
56.6
44.3
34.9
-
-
35.9
-
49.9
-
-
-
-
InstructBLIP
Vicuna-7B
-
-
50.1
-
60.5
26.2
58.8
33.9
59.8
53.8
-
25.3
Fuyu
Fuyu-8B
74.2*
60.6*
-
-
-
-
-
-
-
-
27.4
-
Qwen-VL-Chat
Qwen-7B
78.2*
56.6*
61.5*
70.5*
68.8
-
65.4
61.8
67.7
-
32.9
33.8
LLaVA-1.5
Vicuna-7B
78.5*
-
58.2*
-
66.8
30.5
58.6
64.3
60.7
85.9
-
23.6
mPLUG-Owl2
LLaMA2-7B
79.4*
57.7*
58.2*
-
68.7
36.2
64.1
64.5
25.0
86.2
32.1
25.3
Unified-IO2
UIO-2XXL
79.4*
55.5*
-
-
86.2*
-
65.6
71.5
-
87.7
-
-
LLaVA-1.5
Vicuna-13B
80.0*
-
61.3*
-
71.6
35.4
61.6
67.7
64.6
85.9
33.6
26.1
SPHINX-2k
LLaMA2 13B 80.7*
62.6*
61.2*
67.8*
70.6
40.2
71.6
65.9
-
87.2
32.9
27.8
Emu2-Chat
LLaMA-33B
84.9*
64.8*
66.6*
-
-
48.5
62.8
63.6
56.4
-
34.1
-
CogVLM-Chat
Vicuna-7B
82.3*
64.8*
70.4*
73.8*
91.2*
51.1
72.5
77.6
77.8
87.9
41.1
34.5
Table 3. Results on Referring Expression Comprehension and Grounded Visual Question Answering.
Type
Model
RefCOCO
RefCOCO+
RefCOCOg
Visual7W
val
test-A
test-B
val
test-A
test-B
val
test
test
Generalist
OFA-L* (Wang et al., 2022b)
79.96
83.67
76.39
68.29
76.00
61.75
67.57
67.58
-
VisionLLM-H (Wang et al., 2023b)
-
86.70
-
-
-
-
-
-
-
Shikra-7B (Chen et al., 2023a)
87.01
90.61
80.24
81.60
87.36
72.12
82.27
82.19
-
Shikra-13B (Chen et al., 2023a)
87.83
91.11
81.81
82.89
87.79
74.41
82.64
83.16
85.33
Qwen-VL (Bai et al., 2023)
89.36
92.26
85.34
83.12
88.25
77.21
85.58
85.48
-
Ferret-13B (You et al., 2023)
89.48
92.41
84.36
82.81
88.14
75.17
85.83
86.34
-
CogVLM-Grounding
92.76
94.75
88.99
88.68
92.91
83.39
89.75
90.79
91.05
Specialist
G-DINO-L (Liu et al., 2023e)
90.56
93.19
88.24
82.75
88.95
75.92
86.13
87.02
-
UNINEXT-H (Lin et al., 2023a)
92.64
94.33
91.46
85.24
89.63
79.79
88.73
89.37
-
ONE-PEACE (Wang et al., 2023a)
92.58
94.18
89.26
88.77
92.21
83.23
89.22
89.27
-
LLAMA-33B, leading by 15.7 and 2.6 points on MM-vet,
9.9 and 14.0 points on MMBench, respectively. Compared
to IDEFICS-Instruct trained on LLaMA-65B, CogVLM’s
scores exceeded by 19.3, 23.1, and 20.9 points on Seed-
Bench, MMBench, and LLaVA-Bench, respectively. Fur-
thermore, CogVLM achieved a score of 41.1 on the MMMU
dataset, and also scored 87.9 on the hallucination assessment
dataset POPE, along with 35.2 on the multimodal mathe-
matical reasoning benchmark MathVista. These impressive
results not only showcase its robust reasoning abilities and
multi-task generalization capabilities but also clearly demon-
strate that CogVLM is significantly outpacing other models
in these domains. Notably, shallow fusion models such as
InstructBLIP and MiniGPT-4 underperformed across most
benchmarks, despite InstructBLIP’s extensive training on
instructional data, underscoring the necessity of deep fusion
for enhanced performance.
3.4. Visual Grounding
Table 3 shows the result on the standard visual grounding
benchmarks. We find that our generalist model achieves
state-of-the-art performance across the board, with a sig-
nificant advantage over the previous or concurrent models.
As shown in the bottom part of Table 3, our model even
surpasses models that are specifically trained for individ-
ual tasks, achieving SOTA performance on 5 of 9 splits.
For instance, in the RefCOCO val subset, our model at-
tains a score of 92.76, surpassing UNINEXT-H’s 92.64; in
the RefCOCO+ test-A subset, it scores 92.91, exceeding
ONE-PEACE’s 92.21; and in the RefCOCOg test subset, it
achieves 90.79, outperforming UNINEXT-H’s 89.27. These
results suggest a remarkable visual grounding capability of
our model incorporating our training paradigm.
7

CogVLM: Visual Expert for Pretrained Language Models
Table 4. Ablation studies for various components and training settings. VE refers to visual expert.
Ablated Aspects
Original Setting
Ablated Setting
Trainable
COCO
NoCaps
OKVQA
TextVQA
VQAv2
params
CIDEr↑
CIDEr↑
top1↑
top1↑
top1↑
Tuned parameters
VE-full every layer
+
MLP Adapter
MLP Adapter
140M
131.2
111.5
55.1
40.7
73.8
LLM+MLP Adapter
6.9B
140.3
118.5
56.8
44.7
78.9
VE-full every 4th layer
1.7B
138.7
117.4
58.9
44.1
77.6
VE-FFN every layer
4.4B
140.0
118.7
58.2
45.1
78.6
Init method
From LLM
Random init
6.6B
138.0
117.9
55.9
44.0
79.1
Visual attention mask
Causal mask
Full mask
6.6B
141.0
117.2
57.4
45.1
79.6
Image SSL loss
%
!(clip feature)
6.6B
142.9
119.8
58.7
45.9
79.7
Visual encoder
EVA2-E
EVA2-L
6.6B
141.4
122.5
59.2
42.8
79.0
EMA
!
%
6.6B
143.1
119.2
57.1
43.8
79.4
CogVLM (ours)
—
—
6.6B
142.8
120.1
59.3
45.3
80.0
3.5. Ablation Study
To understand the impact of various components and set-
tings on our model’s performance, we conduct an extensive
ablation study for 6,000 iterations and a batch size of 8,192.
Table 4 summarizes the results about the following aspects:
Model structure and tuned parameters. To investigate
the effectiveness of CogVLM’s model, we conduct ablation
studies on several structure variants and tuning strategies,
including: 1) tuning only the MLP Adapter layer; 2) tuning
all LLM parameters and the Adapter without adding visual
expert; 3) only adding visual expert at every 4th LLM layer;
and 4) only add visual expert to FFNs at all layers.
From the results, we can see that shallow vision-language
alignment, i.e. only tuning the adapter layer (similar to the
method used in BLIP-2), results in a significantly inferior
performance. Also, the performance of training the visual
expert is higher than that of training the LLM, especially on
the datasets that require external knowledge, even though
the training parameters are roughly the same. We also com-
pare with other variants of adding visual expert, including a.
inserting an expert module every 4 layers and b. removing
the attention part from the expert. Both of them result in a
certain degree of performance decline, but within an accept-
able range, which provides some guidance for balancing
computational overhead and model performance.
Initialization Method. As for visual expert’s initializa-
tion method, we compare initialization with weights from
LLM to random initialization. Our results across various
datasets demonstrate that initialization with LLM’s weights
consistently achieves superior performance. This indicates
that the transformer architecture pre-trained on language
data possesses a certain capability to process visual tokens.
Moreover, it can serve as a more effective starting point for
multimodal pre-training initialization.
Visual Attention Mask. We empirically find that using a
causal mask on visual tokens yields a better result in com-
parison with a full mask. This is slightly counterintuitive, as
using a bidirectional attention mask allows access to more
information than a causal mask. We hypothesize the possi-
ble explanation for this phenomenon is that the causal mask
better fits the inherent structure of LLMs.
Image SSL Loss. We also investigated the self-supervised
learning loss on image features, where each visual feature
predicts the CLIP feature of the next position for visual self-
supervision. Align with the observation from PaLI-X (Chen
et al., 2023b), we find it brings no improvement on down-
stream tasks, although we indeed observed improvements
in small models in our early experiments.
Visual Encoder. we substituted the 300M-parameter EVA2-
L model for the 4.4B-parameter EVA2-E to investigate the
impact of visual encoder parameters on various tasks. The
results indicated that there was only a slight decrease in per-
formance across most benchmarks. However, a notable ex-
ception was observed in the text-oriented dataset TextVQA,
where we recorded a decline of 2.5.
EMA. We utilize EMA (Exponential Moving Average) dur-
ing pretraining. The ablation results show that EMA often
brings improvements across various tasks compared to not
using it.
4. Conclusion
In this paper, we introduce CogVLM, an open visual lan-
guage foundation model. CogVLM shifts the paradigm
for VLM training from shallow alignment to deep fusion,
achieving state-of-the-art performance on 17 classic multi-
modal benchmarks.
The VLM training is still in its infancy, and there are many
directions to explore, for example, better SFT alignment,
RLHF and anti-hallucination. Since the previous famous
VLMs are mostly closed-source, we believe CogVLM will
be a solid foundation for future multi-modal research.
8

CogVLM: Visual Expert for Pretrained Language Models
References
Agrawal, H., Desai, K., Wang, Y., Chen, X., Jain, R., John-
son, M., Batra, D., Parikh, D., Lee, S., and Anderson, P.
Nocaps: Novel object captioning at scale. In Proceedings
of the IEEE/CVF international conference on computer
vision, pp. 8948–8957, 2019.
Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I.,
Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds,
M., et al. Flamingo: a visual language model for few-shot
learning. Advances in Neural Information Processing
Systems, 35:23716–23736, 2022.
Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D.,
Zitnick, C. L., and Parikh, D. Vqa: Visual question
answering. In Proceedings of the IEEE international
conference on computer vision, pp. 2425–2433, 2015.
Awadalla, A., Gao, I., Gardner, J., Hessel, J., Hanafy, Y.,
Zhu, W., Marathe, K., Bitton, Y., Gadre, S., Sagawa, S.,
Jitsev, J., Kornblith, S., Koh, P. W., Ilharco, G., Worts-
man, M., and Schmidt, L.
Openflamingo: An open-
source framework for training large autoregressive vision-
language models.
arXiv preprint arXiv:2308.01390,
2023.
Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J.,
Zhou, C., and Zhou, J. Qwen-vl: A frontier large vision-
language model with versatile abilities. arXiv preprint
arXiv:2308.12966, 2023.
Bavishi, R., Elsen, E., Hawthorne, C., Nye, M., Odena,
A., Somani, A., and Tas¸ırlar, S. Introducing our mul-
timodal models, 2023. URL https://www.adept.
ai/blog/fuyu-8b.
Byeon,
M.,
Park,
B.,
Kim,
H.,
Lee,
S.,
Baek,
W., and Kim, S.
Coyo-700m:
Image-text pair
dataset.
https://github.com/kakaobrain/
coyo-dataset, 2022.
Cadene, R., Ben-Younes, H., Cord, M., and Thome, N.
Murel: Multimodal relational reasoning for visual ques-
tion answering. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pp.
1989–1998, 2019.
Chen, K., Zhang, Z., Zeng, W., Zhang, R., Zhu, F., and
Zhao, R. Shikra: Unleashing multimodal llm’s referential
dialogue magic. arXiv preprint arXiv:2306.15195, 2023a.
Chen, T., Li, L., Saxena, S., Hinton, G., and Fleet, D. J.
A generalist framework for panoptic segmentation of
images and videos. arXiv preprint arXiv:2210.06366,
2022a.
Chen, X., Wang, X., Changpinyo, S., Piergiovanni, A.,
Padlewski, P., Salz, D., Goodman, S., Grycner, A.,
Mustafa, B., Beyer, L., et al.
Pali: A jointly-scaled
multilingual language-image model.
arXiv preprint
arXiv:2209.06794, 2022b.
Chen, X., Djolonga, J., Padlewski, P., Mustafa, B., Chang-
pinyo, S., Wu, J., Ruiz, C. R., Goodman, S., Wang, X.,
Tay, Y., et al. Pali-x: On scaling up a multilingual vision
and language model. arXiv preprint arXiv:2305.18565,
2023b.
Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang,
H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E.,
et al. Vicuna: An open-source chatbot impressing gpt-4
with 90%* chatgpt quality. See https://vicuna. lmsys. org
(accessed 14 April 2023), 2023.
Cornia, M., Baraldi, L., Fiameni, G., and Cucchiara, R. Uni-
versal captioner: Long-tail vision-and-language model
training through content-style separation. arXiv preprint
arXiv:2111.12727, 1(2):4, 2021.
Dai, W., Li, J., Li, D., Tiong, A. M. H., Zhao, J., Wang,
W., Li, B., Fung, P., and Hoi, S. Instructblip: Towards
general-purpose vision-language models with instruction
tuning, 2023.
Dong, R., Han, C., Peng, Y., Qi, Z., Ge, Z., Yang, J., Zhao,
L., Sun, J., Zhou, H., Wei, H., et al. Dreamllm: Syn-
ergistic multimodal comprehension and creation. arXiv
preprint arXiv:2309.11499, 2023.
Driess, D., Xia, F., Sajjadi, M. S., Lynch, C., Chowdhery,
A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T.,
et al. Palm-e: An embodied multimodal language model.
arXiv preprint arXiv:2303.03378, 2023.
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika,
M., Song, D., and Steinhardt, J.
Measuring mas-
sive multitask language understanding. arXiv preprint
arXiv:2009.03300, 2020.
Honnibal, M. and Johnson, M. An improved non-monotonic
transition system for dependency parsing. In Proceedings
of the 2015 conference on empirical methods in natural
language processing, pp. 1373–1378, 2015.
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang,
S., Wang, L., and Chen, W. Lora: Low-rank adaptation of
large language models. arXiv preprint arXiv:2106.09685,
2021.
Hu, X., Gan, Z., Wang, J., Yang, Z., Liu, Z., Lu, Y., and
Wang, L. Scaling up vision-language pre-training for
image captioning. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pp.
17980–17989, 2022.
9

CogVLM: Visual Expert for Pretrained Language Models
Kafle, K. and Kanan, C. An analysis of visual question
answering algorithms. In Proceedings of the IEEE inter-
national conference on computer vision, pp. 1965–1973,
2017.
Kazemzadeh, S., Ordonez, V., Matten, M., and Berg, T.
Referitgame: Referring to objects in photographs of natu-
ral scenes. In Proceedings of the 2014 conference on em-
pirical methods in natural language processing (EMNLP),
pp. 787–798, 2014.
Krasin, I., Duerig, T., Alldrin, N., Ferrari, V., Abu-El-Haija,
S., Kuznetsova, A., Rom, H., Uijlings, J., Popov, S., Veit,
A., et al. Openimages: A public dataset for large-scale
multi-label and multi-class image classification. Dataset
available from https://github. com/openimages, 2(3):18,
2017.
Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K.,
Kravitz, J., Chen, S., Kalantidis, Y., Li, L.-J., Shamma,
D. A., et al. Visual genome: Connecting language and
vision using crowdsourced dense image annotations. In-
ternational journal of computer vision, 123:32–73, 2017.
Laurenc¸on, H., Saulnier, L., Tronchon, L., Bekman, S.,
Singh, A., Lozhkov, A., Wang, T., Karamcheti, S., Rush,
A. M., Kiela, D., Cord, M., and Sanh, V. Obelics: An
open web-scale filtered dataset of interleaved image-text
documents, 2023.
Li, B., Wang, R., Wang, G., Ge, Y., Ge, Y., and Shan, Y.
Seed-bench: Benchmarking multimodal llms with gener-
ative comprehension. arXiv preprint arXiv:2307.16125,
2023a.
Li, J., Li, D., Savarese, S., and Hoi, S.
Blip-2: Boot-
strapping language-image pre-training with frozen im-
age encoders and large language models. arXiv preprint
arXiv:2301.12597, 2023b.
Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen,
J.-R.
Evaluating object hallucination in large vision-
language models.
arXiv preprint arXiv:2305.10355,
2023c.
Lin, F., Yuan, J., Wu, S., Wang, F., and Wang, Z. Uninext:
Exploring a unified architecture for vision recognition.
arXiv preprint arXiv:2304.13700, 2023a.
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ra-
manan, D., Doll´ar, P., and Zitnick, C. L. Microsoft coco:
Common objects in context. In Computer Vision–ECCV
2014: 13th European Conference, Zurich, Switzerland,
September 6-12, 2014, Proceedings, Part V 13, pp. 740–
755. Springer, 2014.
Lin, Z., Liu, C., Zhang, R., Gao, P., Qiu, L., Xiao, H., Qiu,
H., Lin, C., Shao, W., Chen, K., et al. Sphinx: The
joint mixing of weights, tasks, and visual embeddings
for multi-modal large language models. arXiv preprint
arXiv:2311.07575, 2023b.
Liu, F., Lin, K., Li, L., Wang, J., Yacoob, Y., and Wang, L.
Aligning large multi-modal model with robust instruction
tuning. arXiv preprint arXiv:2306.14565, 2023a.
Liu, H., Li, C., Li, Y., and Lee, Y. J.
Improved base-
lines with visual instruction tuning.
arXiv preprint
arXiv:2310.03744, 2023b.
Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction
tuning. arXiv preprint arXiv:2304.08485, 2023c.
Liu, J., Wang, L., and Yang, M.-H. Referring expression
generation and comprehension via attributes. In Proceed-
ings of the IEEE International Conference on Computer
Vision, pp. 4856–4864, 2017.
Liu, S., Fan, L., Johns, E., Yu, Z., Xiao, C., and Anand-
kumar, A. Prismer: A vision-language model with an
ensemble of experts. arXiv preprint arXiv:2303.02506,
2023d.
Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C.,
Yang, J., Su, H., Zhu, J., et al. Grounding dino: Marry-
ing dino with grounded pre-training for open-set object
detection. arXiv preprint arXiv:2303.05499, 2023e.
Liu, X., Zheng, Y., Du, Z., Ding, M., Qian, Y., Yang, Z.,
and Tang, J. Gpt understands, too. AI Open, 2023f.
Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W.,
Yuan, Y., Wang, J., He, C., Liu, Z., et al. Mmbench: Is
your multi-modal model an all-around player?
arXiv
preprint arXiv:2307.06281, 2023g.
Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu,
S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to
explain: Multimodal reasoning via thought chains for
science question answering. Advances in Neural Infor-
mation Processing Systems, 35:2507–2521, 2022.
Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi,
H., Cheng, H., Chang, K.-W., Galley, M., and Gao,
J.
Mathvista: Evaluating mathematical reasoning of
foundation models in visual contexts. arXiv preprint
arXiv:2310.02255, 2023.
Mao, J., Huang, J., Toshev, A., Camburu, O., Yuille, A. L.,
and Murphy, K. Generation and comprehension of unam-
biguous object descriptions. In Proceedings of the IEEE
conference on computer vision and pattern recognition,
pp. 11–20, 2016.
10

CogVLM: Visual Expert for Pretrained Language Models
Marino, K., Rastegari, M., Farhadi, A., and Mottaghi, R.
Ok-vqa: A visual question answering benchmark requir-
ing external knowledge. In Proceedings of the IEEE/cvf
conference on computer vision and pattern recognition,
pp. 3195–3204, 2019.
Mishra, A., Shekhar, S., Singh, A. K., and Chakraborty, A.
Ocr-vqa: Visual question answering by reading text in
images. In 2019 international conference on document
analysis and recognition (ICDAR), pp. 947–952. IEEE,
2019.
Peng, Z., Wang, W., Dong, L., Hao, Y., Huang, S., Ma,
S., and Wei, F.
Kosmos-2: Grounding multimodal
large language models to the world.
arXiv preprint
arXiv:2306.14824.
Plummer, B. A., Wang, L., Cervantes, C. M., Caicedo, J. C.,
Hockenmaier, J., and Lazebnik, S. Flickr30k entities:
Collecting region-to-phrase correspondences for richer
image-to-sentence models. In Proceedings of the IEEE
international conference on computer vision, pp. 2641–
2649, 2015.
Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S.,
Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring
the limits of transfer learning with a unified text-to-text
transformer. The Journal of Machine Learning Research,
21(1):5485–5551, 2020.
Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C.,
Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis,
C., Wortsman, M., et al. Laion-5b: An open large-scale
dataset for training next generation image-text models.
Advances in Neural Information Processing Systems, 35:
25278–25294, 2022.
Shazeer, N.
Glu variants improve transformer.
arXiv
preprint arXiv:2002.05202, 2020.
Shrestha, R., Kafle, K., and Kanan, C. Answer them all!
toward universal visual question answering models. In
Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pp. 10472–10481, 2019.
Sidorov, O., Hu, R., Rohrbach, M., and Singh, A. Textcaps:
a dataset for image captioning with reading comprehen-
sion. In Computer Vision–ECCV 2020: 16th European
Conference, Glasgow, UK, August 23–28, 2020, Proceed-
ings, Part II 16, pp. 742–758. Springer, 2020.
Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X.,
Batra, D., Parikh, D., and Rohrbach, M. Towards vqa
models that can read. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
pp. 8317–8326, 2019.
Sun, Q., Cui, Y., Zhang, X., Zhang, F., Yu, Q., Luo, Z.,
Wang, Y., Rao, Y., Liu, J., Huang, T., et al. Generative
multimodal models are in-context learners. arXiv preprint
arXiv:2312.13286, 2023a.
Sun, Q., Fang, Y., Wu, L., Wang, X., and Cao, Y. Eva-
clip: Improved training techniques for clip at scale. arXiv
preprint arXiv:2303.15389, 2023b.
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi,
A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P.,
Bhosale, S., et al. Llama 2: Open foundation and fine-
tuned chat models. arXiv preprint arXiv:2307.09288,
2023.
Tsimpoukelli, M., Menick, J. L., Cabi, S., Eslami, S.,
Vinyals, O., and Hill, F.
Multimodal few-shot learn-
ing with frozen language models. Advances in Neural
Information Processing Systems, 34:200–212, 2021.
Wang, J., Yang, Z., Hu, X., Li, L., Lin, K., Gan, Z., Liu,
Z., Liu, C., and Wang, L. Git: A generative image-to-
text transformer for vision and language. arXiv preprint
arXiv:2205.14100, 2022a.
Wang, P., Yang, A., Men, R., Lin, J., Bai, S., Li, Z., Ma, J.,
Zhou, C., Zhou, J., and Yang, H. Ofa: Unifying architec-
tures, tasks, and modalities through a simple sequence-
to-sequence learning framework. In International Con-
ference on Machine Learning, pp. 23318–23340. PMLR,
2022b.
Wang, P., Wang, S., Lin, J., Bai, S., Zhou, X., Zhou, J.,
Wang, X., and Zhou, C. One-peace: Exploring one gen-
eral representation model toward unlimited modalities.
arXiv preprint arXiv:2305.11172, 2023a.
Wang, W., Chen, Z., Chen, X., Wu, J., Zhu, X., Zeng, G.,
Luo, P., Lu, T., Zhou, J., Qiao, Y., et al. Visionllm: Large
language model is also an open-ended decoder for vision-
centric tasks. arXiv preprint arXiv:2305.11175, 2023b.
Wang, Z., Yu, J., Yu, A. W., Dai, Z., Tsvetkov, Y., and Cao,
Y. Simvlm: Simple visual language model pretraining
with weak supervision. arXiv preprint arXiv:2108.10904,
2021.
Ye, Q., Xu, H., Ye, J., Yan, M., Liu, H., Qian, Q., Zhang,
J., Huang, F., and Zhou, J. mplug-owl2: Revolution-
izing multi-modal large language model with modality
collaboration. arXiv preprint arXiv:2311.04257, 2023.
You, H., Zhang, H., Gan, Z., Du, X., Zhang, B., Wang,
Z., Cao, L., Chang, S.-F., and Yang, Y. Ferret: Refer
and ground anything anywhere at any granularity. arXiv
preprint arXiv:2310.07704, 2023.
11

CogVLM: Visual Expert for Pretrained Language Models
Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini,
M., and Wu, Y. Coca: Contrastive captioners are image-
text foundation models. arXiv preprint arXiv:2205.01917,
2022.
Yu, L., Poirson, P., Yang, S., Berg, A. C., and Berg, T. L.
Modeling context in referring expressions. In Computer
Vision–ECCV 2016: 14th European Conference, Amster-
dam, The Netherlands, October 11-14, 2016, Proceed-
ings, Part II 14, pp. 69–85. Springer, 2016.
Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang,
X., and Wang, L.
Mm-vet: Evaluating large multi-
modal models for integrated capabilities. arXiv preprint
arXiv:2308.02490, 2023.
Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G.,
Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. Mmmu:
A massive multi-discipline multimodal understanding
and reasoning benchmark for expert agi. arXiv preprint
arXiv:2311.16502, 2023.
Zhang, H., Zhang, P., Hu, X., Chen, Y.-C., Li, L., Dai, X.,
Wang, L., Yuan, L., Hwang, J.-N., and Gao, J. Glipv2:
Unifying localization and vision-language understanding.
Advances in Neural Information Processing Systems, 35:
36067–36080, 2022.
Zhang, P., Li, X., Hu, X., Yang, J., Zhang, L., Wang, L.,
Choi, Y., and Gao, J. Vinvl: Revisiting visual representa-
tions in vision-language models. In Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, pp. 5579–5588, 2021.
Zhang, Y., Zhang, R., Gu, J., Zhou, Y., Lipka, N., Yang, D.,
and Sun, T. Llavar: Enhanced visual instruction tuning
for text-rich image understanding, 2023.
Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M.
Minigpt-4: Enhancing vision-language understanding
with advanced large language models. arXiv preprint
arXiv:2304.10592, 2023.
Zhu, Y., Groth, O., Bernstein, M., and Fei-Fei, L. Visual7w:
Grounded question answering in images. In Proceedings
of the IEEE conference on computer vision and pattern
recognition, pp. 4995–5004, 2016.
12

CogVLM: Visual Expert for Pretrained Language Models
A. Appendix
A.1. Details of Training Settings
We report the details of parameter settings during pre-training and multitask training in Table 5 and Table 6.
Table 5. Hyperparameters for pre-training model.
Hyperparameters Stage 1
Stage 2
Total steps
120, 000
60, 000
Warmup steps
12, 000
1, 200
Batch size
8, 192
1, 024
Learning rate
1e−4
1e−5
Learning rate decay
Cosine
Weight decay
0.05
Dropout ratio
0.1
Adam ϵ
1e−8
Adam β
(0.9, 0.95)
Textual encoder
Vicuna-1.5-7B
Visual encoder
EVA2-CLIP-E
Patch size
14
Input resolution
2242
2242 →4902
Table 6. Hyperparameters for multitask finetuning CogVLM.
Hyperparameters Multitask
Learning rate
1e−5
Total steps
6,000
Batch size
1,024
AdamW ϵ
1e−8
AdamW β
(0.9, 0.95)
Weight decay
0.1
Dropout ratio
0.1
Input resolution
4902
A.2. Details of Associated Datasets
In this section, we introduce the details of datasets and their use in our evaluation process for all associated benchmarks.
A.2.1. IMAGE CAPTIONING
• COCO (Lin et al., 2014) The Captions in COCO dataset are collected using Amazon’s Mechanical Turk (AMT)
workers who are given instructions to control the quality. The dataset contains 330K images, where the train, validation
and test sets contain 413,915 captions for 82,783 images, 202,520 captions for 40,504 images, and 379,249 captions
for 40,775 images respectively.
• NoCaps (Agrawal et al., 2019). NoCaps is a large-scale benchmark for novel object captioning, containing nearly
400 novel object classes compared to COCO. The validation and test set comprised of 4,500 and 10,600 images,
respectively, sourced from the Open Images (Krasin et al., 2017) and annotated with 11 human-generated captions per
13

CogVLM: Visual Expert for Pretrained Language Models
Table 7. Summary of the evaluation benchmarks.
Task
Dataset
Description
Split
Metrics
Image Caption
NoCaps
Captioning of natural images.
val
CIDEr (↑)
Flickr
Captioning of natural images.
karpathy-test
CIDEr (↑)
COCO
Captioning of natural images.
karpathy-test
CIDEr (↑)
TextCaps
Captioning of natural images containing text.
test
CIDEr (↑)
General VQA
VQAv2
VQA on natural images.
test-dev
VQA Score(↑)
OK-VQA
VQA on natural images requiring outside knowledge.
val
VQA Score (↑)
ScienceQA
Multi-choice VQA on a diverse set of science topics
test
Accuracy (↑)
TDIUC
VQA on natural images with detailed question types.
val
VQA Score (↑)
Text-oriented VQA
OCR-VQA
VQA on images of book covers.
test
EM (↑)
TextVQA
VQA on natural images containing text.
val
VQA Score (↑)
LVLM Benchmarks
MM-Vet
Open-ended VQA on a diverse set of topics
test
GPT4 Score(↑)
SEED-Bench
Multi-choice VQA on a diverse set of topics
IMG
Accuracy (↑)
MMBench
Multi-choice VQA on a diverse set of topics
test
Accuracy (↑)
LLaVA-Bench
Open-ended VQA for testing instruction following abilities
In-the-Wild
GPT4 Score(↑)
POPE
Multi-choice VQA for testing hallucinations
overall
Accuracy (↑)
MMMU
VQA on a diverse set of topics
test
Accuracy (↑)
MathVista
VQA for Measuring Mathematical Abilities
test-mini
Accuracy (↑)
Grounding
RefCOCO
Refer grounding on natural images.
overall
Accuracy (↑)
RefCOCO+
Refer grounding on natural images.
overall
Accuracy (↑)
RefCOCOg
Refer grounding on natural images.
overall
Accuracy (↑)
Visual7W
VQA with referential regions selection.
val
Accuracy (↑)
image, and each set is subdivided into three domains: “in”, “near”, and “out”, with objects in the “out-domain” never
appearing in the COCO dataset.
• Flickr30K (Plummer et al., 2015). Flickr30K is a high-quality dataset consists of 31,783 images of everyday
life activities, envets and scenes (all harvested from the online website Flickr) and 158,915 captions (obtained via
crodsourcing). Each image in this dataset is described independently by five annotators who are not familiar with the
specific entities and circumstances depicted in them.
• TextCaps (Sidorov et al., 2020) Textcaps is a dataset with 145k captions for 28k images. The design purpose of the
TextCaps dataset is to effectively integrate textual information with visual context into captions, requiring the model to
have both excellent OCR capabilities and strong captioning abilities.
A.2.2. GENERAL VQA
• VQAv2 (Antol et al., 2015) VQAv2 encompasses over 200,000 images, paired with more than 1.1 million questions
that have collectively garnered over 11 million answers. Questions span various types, including yes/no, counting, and
open-ended queries.
• OKVQA (Marino et al., 2019) The OK-VQA (Outside Knowledge Visual Question Answering) dataset is specifically
designed to probe visual question answering capabilities that necessitate external knowledge or common sense beyond
image content. It has 14,055 open-ended questions and 5 ground truth answers per question.
• ScienceQA (Lu et al., 2022) The ScienceQA dataset comprises 21,208 multimodal multiple-choice questions spanning
three diverse subjects: natural science, language science, and social science. Each question is annotated with
explanations linked to relevant lectures.
• TDIUC (Shrestha et al., 2019) The TDIUC dataset features 1.6M questions across 170K images from MS COCO and
Visual Genome. Categorized into 12 distinct question types, it ranges from basic tasks like identifying objects or colors
to more advanced reasoning like counting or positional discernment.
14

CogVLM: Visual Expert for Pretrained Language Models
A.2.3. TEXT-ORIENTED VQA
• OCRVQA (Mishra et al., 2019) OCR-VQA consists of 207,572 book cover images with over 1 million question-answer
pairs.
• TextVQA (Singh et al., 2019) TextVQA is a dataset with 45,336 questions on 28,408 images that challenges models to
detect, read, and reason about text within images to provide answers.
A.3. LVLM Benchmarks
• MM-Vet (Yu et al., 2023) MM-Vet defines six core VL capabilities and examines 16 integrations of interest derived
from the combinations of these capabilities. It employs an evaluator based on LLMs for open-ended outputs, capable
of assessing across different question types and answer styles, thus deriving a unified scoring metric.
• SEED-Bench (Li et al., 2023a) SEED-Bench is a dataset comprising 19K multiple-choice questions with precise
human annotations, covering 12 evaluation dimensions, including understanding of image and video modalities. It
obtains accurate answer options through manual annotations, enabling objective and efficient assessment of model
performance.
• MMBench (Liu et al., 2023g) MMBench comprises approximately 3000 multiple-choice questions, covering 20
different capability dimensions, aimed at evaluating various abilities of visual-language models. MMBench adopts a
hierarchical capability dimension structure, including two high-level capability dimensions: perception and reasoning,
as well as fine-grained capability dimensions such as object localization and attribute inference.
• LLaVA-Bench (Liu et al., 2023c) LLaVA-Bench (In-the-Wild) is a benchmark dataset comprising 60 questions,
designed to evaluate the multimodal instruction following capabilities of LMMs. It includes indoor and outdoor scenes,
memes, paintings, sketches, etc., and is equipped with highly detailed, manually curated descriptions and appropriate
question selections.
• POPE (Li et al., 2023c) The POPE dataset is a binary classification query dataset specifically designed to evaluate
object hallucination issues in LMMs. The random, popular, and adversarial subsets within the POPE dataset are
constructed through different sampling strategies, totaling 8,910 entries.
• MMMU (Yue et al., 2023) The MMMU dataset is a large-scale, multidisciplinary multimodal understanding and
reasoning benchmark set, containing 11.5K questions. It covers 6 major disciplines, 30 topics, and 183 subfields, with
question types including multiple-choice and open-ended questions. The dataset includes 30 types of images, such
as charts, tables, chemical structures, photographs, paintings, musical scores, etc., testing the multimodal perception
capabilities of models and their performance in expert-level tasks.
• MathVista (Lu et al., 2023) MathVista is a new benchmark dataset that combines mathematical and visual understand-
ing, comprising 31 existing multimodal datasets and 3 newly created datasets, totaling 6141 examples. These datasets
encompass a diverse range of mathematical reasoning abilities, including seven types: algebra, arithmetic, geometry,
logic, numerical common sense, science, and statistics. The goal is to comprehensively evaluate the capabilities of
existing foundational models in mathematical reasoning and visual understanding.
A.3.1. GROUNDING
• RefCOCO/RefCOCO+ (Liu et al., 2017) RefCOCO and RefCOCO+ evolved from the ReferItGame. Both subsets
focus on images with two or more similar objects. RefCOCO, with 142,209 expressions across 19,994 images, places
no linguistic constraints. Conversely, RefCOCO+ emphasizes appearance-centric descriptions, omitting locational
terms, and comprises 141,564 expressions over 19,992 images.
• RefCOCOg (Mao et al., 2016) The RefCOCOg subset was amassed through Amazon Mechanical Turk, where workers
penned natural referring expressions for objects in MSCOCO images; it boasts 85,474 referring expressions spanning
26,711 images, each containing 2 to 4 objects of the same category.
• Visual7W (Zhu et al., 2016). The Visual7W dataset is predominantly designed for VQA tasks, with a dedicated
subset crafted for grounded VQA. In this subset, models are presented with an image accompanied by a “which”-type
15

CogVLM: Visual Expert for Pretrained Language Models
question, such as “Which is the small computer in the corner?”. Participants are then given four bounding boxes within
the image, from which they must select the correct one as the answer. The grounded Visual7W part consists of 25,733
images and 188,068 questions.
• Flickr30K-Entities (Plummer et al., 2015). The Flickr30K Entities dataset, a precursor in the realm of grounded
captioning, encompasses a collection of 31,783 images accompanied by 158k captioning annotations. Every caption in
this dataset has been meticulously annotated such that each noun phrase is linked with a manually delineated referential
bounding box. In total, there are 276k such annotated bounding boxes provided within this dataset.
• VisualGenome (Krishna et al., 2017). The VisualGenome dataset stands as a cornerstone in understanding the
multifaceted relationships present within images. With a collection of over 100k images, each image is annotated
in detail, capturing an average of 21 objects, 18 attributes, and 18 inter-object relationships. A unique aspect of this
dataset is the alignment of objects, attributes, relationships, and region descriptions with standardized terminologies
from WordNet. Specifically tailored for the REG and REC tasks, each annotated region in an image comes with a
corresponding descriptive text, making it a rich resource for image understanding and semantic modeling. We use the
subset with around 86k images and 3.6 million region-caption pairs for visual grounding.
0
20
40
60
80
100
Absurd
Average
Counting
Object Presence
Color Attribute
Other Attributes
Sport Recognition.
Object Recognition
Positional Reasoning
Scene Classification
Activity Recognition
Utility and Affordances
Sentiment Understanding
Previous SOTA
CogVLM-17B
Figure 5. Performance on TDIUC benchmark with fine-grained questions classes.
B. Additional Fine-grained Experiments
To comprehensively investigate the proposed model on specific topics and question types, we further conduct extensive
experiments on a representative benchmark, TDIUC (Kafle & Kanan, 2017). We use the publicly available split of val set as
evaluation data, and the VQA accuracy calculated from their official scripts as the evaluation metric.
The experimental results on TDIUC compare our model against the specialist SOTA method MUREL (Cadene et al., 2019)
are shown in Figure 5. From the experimental result, we can see that our model consistently outperforms the previous model
on 12 specific question types, resulting in a 94.0 accuracy score compared to the previous SOTA of 88.2 on the overall
dataset. These results demonstrate that our model exhibits comprehensive problem-solving skills on general VQA tasks.
C. Computational Efficiency
In this section, we compare the computational efficiency of our model with other state-of-the-art models, considering both
pretraining and finetuning data from datasets such as VQAv2 and TextVQA. Owing to an optimized architecture and the
16

CogVLM: Visual Expert for Pretrained Language Models
utilization of high-quality pretraining data, our model demonstrates a marked reduction in resource consumption during
training relative to models with comparable parameter magnitudes.
Table 8. Comparison of different models based on their computational efficiency. We use PFLOPS*days as metrics.
Model
Pretraining Data
Pretraining compute
VQAv2 finetuning
TextVQA finetuning
PaLI-3B
1.6B
56
1.1
0.2
PaLI-17B
1.6B
453
4.5
0.9
Flamingo-80B
2.3B
1381*
N/A
N/A
GIT2-5.1B
12.9B
5513*
N/A
N/A
CogVLM
1.5B
230.1
1.2
0.13
17
