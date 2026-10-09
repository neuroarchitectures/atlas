# Paper (Wu et al. 2021)

> Source: `https://arxiv.org/abs/2103.15808`

---

CvT: Introducing Convolutions to Vision Transformers
Haiping Wu1,2*
Bin Xiao2†
Noel Codella2
Mengchen Liu2
Xiyang Dai2
Lu Yuan2
Lei Zhang2
1McGill University
2Microsoft Cloud + AI
haiping.wu2@mail.mcgill.ca, {bixi, ncodella, mengcliu, xidai, luyuan, leizhang}@microsoft.com
Abstract
We present in this paper a new architecture, named Con-
volutional vision Transformer (CvT), that improves Vision
Transformer (ViT) in performance and efﬁciency by intro-
ducing convolutions into ViT to yield the best of both de-
signs. This is accomplished through two primary modiﬁca-
tions: a hierarchy of Transformers containing a new convo-
lutional token embedding, and a convolutional Transformer
block leveraging a convolutional projection. These changes
introduce desirable properties of convolutional neural net-
works (CNNs) to the ViT architecture (i.e. shift, scale,
and distortion invariance) while maintaining the merits of
Transformers (i.e. dynamic attention, global context, and
better generalization). We validate CvT by conducting ex-
tensive experiments, showing that this approach achieves
state-of-the-art performance over other Vision Transform-
ers and ResNets on ImageNet-1k, with fewer parame-
ters and lower FLOPs.
In addition, performance gains
are maintained when pretrained on larger datasets (e.g.
ImageNet-22k) and ﬁne-tuned to downstream tasks. Pre-
trained on ImageNet-22k, our CvT-W24 obtains a top-1 ac-
curacy of 87.7% on the ImageNet-1k val set. Finally, our
results show that the positional encoding, a crucial com-
ponent in existing Vision Transformers, can be safely re-
moved in our model, simplifying the design for higher res-
olution vision tasks.
Code will be released at https:
//github.com/leoxiaobin/CvT.
1. Introduction
Transformers [31, 10] have recently dominated a wide
range of tasks in natural language processing (NLP) [32].
The Vision Transformer (ViT) [11] is the ﬁrst computer vi-
sion model to rely exclusively on the Transformer archi-
tecture to obtain competitive image classiﬁcation perfor-
mance at large scale. The ViT design adapts Transformer
*This work is done when Haiping Wu was an intern at Microsoft.
†Corresponding author
CvT
ViT
BiT
(a)
78
80
82
84
86
88
ImageNet top-1 accuracy (%)
20M
32M
277M
86M
307M
25M
928M
CvT
ViT
BiT
20
40
60
80
Model Paramters (M)
(b)
80.0
80.5
81.0
81.5
82.0
82.5
ImageNet top-1 accuracy (%)
CvT
DeiT
T2T
PVT
TNT
Figure 1: Top-1 Accuracy on ImageNet validation com-
pared to other methods with respect to model parame-
ters. (a) Comparison to CNN-based model BiT [18] and
Transformer-based model ViT [11], when pretrained on
ImageNet-22k. Larger marker size indicates larger archi-
tectures. (b) Comparison to concurrent works: DeiT [30],
T2T [41], PVT [34], TNT [14] when pretrained on
ImageNet-1k.
architectures [10] from language understanding with mini-
mal modiﬁcations. First, images are split into discrete non-
overlapping patches (e.g. 16 × 16). Then, these patches are
treated as tokens (analogous to tokens in NLP), summed
with a special positional encoding to represent coarse spa-
tial information, and input into repeated standard Trans-
former layers to model global relations for classiﬁcation.
Despite the success of vision Transformers at large scale,
the performance is still below similarly sized convolutional
neural network (CNN) counterparts (e.g., ResNets [15])
when trained on smaller amounts of data. One possible rea-
son may be that ViT lacks certain desirable properties in-
herently built into the CNN architecture that make CNNs
uniquely suited to solve vision tasks.
For example, im-
ages have a strong 2D local structure: spatially neighbor-
ing pixels are usually highly correlated. The CNN archi-
1
arXiv:2103.15808v1  [cs.CV]  29 Mar 2021

Method
Needs Position Encoding (PE)
Token Embedding
Projection for Attention
Hierarchical Transformers
ViT [11], DeiT [30]
yes
non-overlapping
linear
no
CPVT [6]
no (w/ PE Generator)
non-overlapping
linear
no
TNT [14]
yes
non-overlapping (patch+pixel)
linear
no
T2T [41]
yes
overlapping (concatenate)
linear
partial (tokenization)
PVT [34]
yes
non-overlapping
spatial reduction
yes
CvT (ours)
no
overlapping (convolution)
convolution
yes
Table 1: Representative works of vision Transformers.
tecture forces the capture of this local structure by using
local receptive ﬁelds, shared weights, and spatial subsam-
pling [20], and thus also achieves some degree of shift,
scale, and distortion invariance. In addition, the hierarchi-
cal structure of convolutional kernels learns visual patterns
that take into account local spatial context at varying levels
of complexity, from simple low-level edges and textures to
higher order semantic patterns.
In this paper, we hypothesize that convolutions can be
strategically introduced to the ViT structure to improve
performance and robustness, while concurrently maintain-
ing a high degree of computational and memory efﬁciency.
To verify our hypothesises, we present a new architecture,
called the Convolutional vision Transformer (CvT), which
incorporates convolutions into the Transformer that is in-
herently efﬁcient, both in terms of ﬂoating point operations
(FLOPs) and parameters.
The CvT design introduces convolutions to two core sec-
tions of the ViT architecture. First, we partition the Trans-
formers into multiple stages that form a hierarchical struc-
ture of Transformers. The beginning of each stage consists
of a convolutional token embedding that performs an over-
lapping convolution operation with stride on a 2D-reshaped
token map (i.e., reshaping ﬂattened token sequences back
to the spatial grid), followed by layer normalization. This
allows the model to not only capture local information, but
also progressively decrease the sequence length while si-
multaneously increasing the dimension of token features
across stages, achieving spatial downsampling while con-
currently increasing the number of feature maps, as is per-
formed in CNNs [20]. Second, the linear projection prior
to every self-attention block in the Transformer module is
replaced with our proposed convolutional projection, which
employs a s × s depth-wise separable convolution [5] oper-
ation on an 2D-reshaped token map. This allows the model
to further capture local spatial context and reduce seman-
tic ambiguity in the attention mechanism. It also permits
management of computational complexity, as the stride of
convolution can be used to subsample the key and value ma-
trices to improve efﬁciency by 4× or more, with minimal
degradation of performance.
In summary, our proposed Convolutional vision Trans-
former (CvT) employs all the beneﬁts of CNNs: local re-
ceptive ﬁelds, shared weights, and spatial subsampling,
while keeping all the advantages of Transformers: dynamic
attention, global context fusion, and better generalization.
Our results demonstrate that this approach attains state-of-
art performance when CvT is pre-trained with ImageNet-
1k, while being lightweight and efﬁcient: CvT improves the
performance compared to CNN-based models (e.g. ResNet)
and prior Transformer-based models (e.g. ViT, DeiT) while
utilizing fewer FLOPS and parameters. In addition, CvT
achieves state-of-the-art performance when evaluated at
larger scale pretraining (e.g. on the public ImageNet-22k
dataset). Finally, we demonstrate that in this new design, we
can drop the positional embedding for tokens without any
degradation to model performance. This not only simpliﬁes
the architecture design, but also makes it readily capable of
accommodating variable resolutions of input images that is
critical to many vision tasks.
2. Related Work
Transformers that exclusively rely on the self-attention
mechanism to capture global dependencies have dominated
in natural language modelling [31, 10, 25]. Recently, the
Transformer based architecture has been viewed as a viable
alternative to the convolutional neural networks (CNNs) in
visual recognition tasks, such as classiﬁcation [11, 30], ob-
ject detection [3, 45, 43, 8, 28], segmentation [33, 36], im-
age enhancement [4, 40], image generation [24], video pro-
cessing [42, 44] and 3D point cloud processing [12].
Vision Transformers.
The Vision Transformer (ViT) is
the ﬁrst to prove that a pure Transformer architecture can
attain state-of-the-art performance (e.g. ResNets [15], Ef-
ﬁcientNet [29]) on image classiﬁcation when the data is
large enough (i.e. on ImageNet-22k, JFT-300M). Speciﬁ-
cally, ViT decomposes each image into a sequence of tokens
(i.e. non-overlapping patches) with ﬁxed length, and then
applies multiple standard Transformer layers, consisting of
Multi-Head Self-Attention module (MHSA) and Position-
wise Feed-forward module (FFN), to model these tokens.
DeiT [30] further explores the data-efﬁcient training and
distillation for ViT. In this work, we study how to combine
2

Figure 2: The pipeline of the proposed CvT architecture. (a) Overall architecture, showing the hierarchical multi-stage
structure facilitated by the Convolutional Token Embedding layer. (b) Details of the Convolutional Transformer Block,
which contains the convolution projection as the ﬁrst layer.
CNNs and Transformers to model both local and global de-
pendencies for image classiﬁcation in an efﬁcient way.
In order to better model local context in vision Trans-
formers, some concurrent works have introduced design
changes.
For example, the Conditional Position encod-
ings Visual Transformer (CPVT) [6] replaces the prede-
ﬁned positional embedding used in ViT with conditional
position encodings (CPE), enabling Transformers to pro-
cess input images of arbitrary size without interpolation.
Transformer-iN-Transformer (TNT) [14] utilizes both an
outer Transformer block that processes the patch embed-
dings, and an inner Transformer block that models the re-
lation among pixel embeddings, to model both patch-level
and pixel-level representation. Tokens-to-Token (T2T) [41]
mainly improves tokenization in ViT by concatenating mul-
tiple tokens within a sliding window into one token. How-
ever, this operation fundamentally differs from convolutions
especially in normalization details, and the concatenation
of multiple tokens greatly increases complexity in compu-
tation and memory. PVT [34] incorporates a multi-stage
design (without convolutions) for Transformer similar to
multi-scales in CNNs, favoring dense prediction tasks.
In contrast to these concurrent works, this work aims
to achieve the best of both worlds by introducing convolu-
tions, with image domain speciﬁc inductive biases, into the
Transformer architecture. Table 1 shows the key differences
in terms of necessity of positional encodings, type of token
embedding, type of projection, and Transformer structure in
the backbone, between the above representative concurrent
works and ours.
Introducing Self-attentions to CNNs.
Self-attention
mechanisms have been widely applied to CNNs in vision
tasks. Among these works, the non-local networks [35] are
designed for capturing long range dependencies via global
attention. The local relation networks [17] adapts its weight
aggregation based on the compositional relations (similar-
ity) between pixels/features within a local window, in con-
trast to convolution layers which employ ﬁxed aggrega-
tion weights over spatially neighboring input feature. Such
an adaptive weight aggregation introduces geometric pri-
ors into the network which are important for the recogni-
tion tasks. Recently, BoTNet [27] proposes a simple yet
powerful backbone architecture that just replaces the spa-
tial convolutions with global self-attention in the ﬁnal three
bottleneck blocks of a ResNet and achieves a strong per-
formance in image recognition. Instead, our work performs
an opposite research direction: introducing convolutions to
Transformers.
Introducing Convolutions to Transformers.
In NLP
and speech recognition, convolutions have been used to
modify the Transformer block, either by replacing multi-
head attentions with convolution layers [38], or adding
additional convolution layers in parallel
[39] or sequen-
tially [13], to capture local relationships. Other prior work
[37] proposes to propagate attention maps to succeeding
layers via a residual connection, which is ﬁrst transformed
by convolutions. Different from these works, we propose
to introduce convolutions to two primary parts of the vi-
sion Transformer: ﬁrst, to replace the existing Position-wise
Linear Projection for the attention operation with our Con-
volutional Projection, and second, to use our hierarchical
multi-stage structure to enable varied resolution of 2D re-
shaped token maps, similar to CNNs. Our unique design
affords signiﬁcant performance and efﬁciency beneﬁts over
3

prior works.
3. Convolutional vision Transformer
The overall pipeline of the Convolutional vision Trans-
former (CvT) is shown in Figure 2.
We introduce two
convolution-based operations into the Vision Transformer
architecture, namely the Convolutional Token Embedding
and Convolutional Projection. As shown in Figure 2 (a), a
multi-stage hierarchy design borrowed from CNNs [20, 15]
is employed, where three stages in total are used in this
work.
Each stage has two parts.
First, the input image
(or 2D reshaped token maps) are subjected to the Convo-
lutional Token Embedding layer, which is implemented as a
convolution with overlapping patches with tokens reshaped
to the 2D spatial grid as the input (the degree of overlap
can be controlled via the stride length). An additional layer
normalization is applied to the tokens. This allows each
stage to progressively reduce the number of tokens (i.e. fea-
ture resolution) while simultaneously increasing the width
of the tokens (i.e. feature dimension), thus achieving spa-
tial downsampling and increased richness of representation,
similar to the design of CNNs. Different from other prior
Transformer-based architectures [11, 30, 41, 34], we do not
sum the ad-hod position embedding to the tokens. Next,
a stack of the proposed Convolutional Transformer Blocks
comprise the remainder of each stage. Figure 2 (b) shows
the architecture of the Convolutional Transformer Block,
where a depth-wise separable convolution operation [5],
referred as Convolutional Projection, is applied for query,
key, and value embeddings respectively, instead of the stan-
dard position-wise linear projection in ViT [11]. Addition-
ally, the classiﬁcation token is added only in the last stage.
Finally, an MLP (i.e. fully connected) Head is utilized upon
the classiﬁcation token of the ﬁnal stage output to predict
the class.
We ﬁrst elaborate on the proposed Convolutional Token
Embedding layer. Next we show how to perform Convolu-
tional Projection for the Multi-Head Self-Attention module,
and its efﬁcient design for managing computational cost.
3.1. Convolutional Token Embedding
This convolution operation in CvT aims to model local
spatial contexts, from low-level edges to higher order se-
mantic primitives, over a multi-stage hierarchy approach,
similar to CNNs.
Formally, given a 2D image or a 2D-reshaped output to-
ken map from a previous stage xi−1 ∈RHi−1×Wi−1×Ci−1
as the input to stage i, we learn a function f(·) that maps
xi−1 into new tokens f(xi−1) with a channel size Ci, where
f(·) is 2D convolution operation of kernel size s × s, stride
s −o and p padding (to deal with boundary conditions).
The new token map f(xi−1) ∈RHi×Wi×Ci has height and
width
Hi =
Hi−1 + 2p −s
s −o
+ 1

, Wi =
Wi−1 + 2p −s
s −o
+ 1

.
(1)
f(xi−1) is then ﬂattened into size HiWi × Ci and normal-
ized by layer normalization [1] for input into the subsequent
Transformer blocks of stage i.
The Convolutional Token Embedding layer allows us to
adjust the token feature dimension and the number of to-
kens at each stage by varying parameters of the convolution
operation. In this manner, in each stage we progressively
decrease the token sequence length, while increasing the
token feature dimension. This gives the tokens the ability
to represent increasingly complex visual patterns over in-
creasingly larger spatial footprints, similar to feature layers
of CNNs.
3.2. Convolutional Projection for Attention
The goal of the proposed Convolutional Projection layer
is to achieve additional modeling of local spatial context,
and to provide efﬁciency beneﬁts by permitting the under-
sampling of K and V matrices.
Fundamentally, the proposed Transformer block with
Convolutional Projection is a generalization of the origi-
nal Transformer block. While previous works [13, 39] try
to add additional convolution modules to the Transformer
Block for speech recognition and natural language process-
ing, they result in a more complicated design and addi-
tional computational cost. Instead, we propose to replace
the original position-wise linear projection for Multi-Head
Self-Attention (MHSA) with depth-wise separable convo-
lutions, forming the Convolutional Projection layer.
3.2.1
Implementation Details
Figure 3 (a) shows the original position-wise linear projec-
tion used in ViT [11] and Figure 3 (b) shows our proposed
s × s Convolutional Projection. As shown in Figure 3 (b),
tokens are ﬁrst reshaped into a 2D token map. Next, a Con-
volutional Projection is implemented using a depth-wise
separable convolution layer with kernel size s. Finally, the
projected tokens are ﬂattened into 1D for subsequent pro-
cess. This can be formulated as:
xq/k/v
i
= Flatten (Conv2d (Reshape2D(xi), s)) ,
(2)
where xq/k/v
i
is the token input for Q/K/V matrices at
layer i, xi is the unperturbed token prior to the Convolu-
tional Projection, Conv2d is a depth-wise separable con-
volution [5] implemented by: Depth-wise Conv2d →
BatchNorm2d →Point-wise Conv2d, and s refers
to the convolution kernel size.
The resulting new Transformer Block with the Convo-
lutional Projection layer is a generalization of the original
4

Figure 3: (a) Linear projection in ViT [11]. (b) Convolutional projection. (c) Squeezed convolutional projection. Unless
otherwise stated, we use (c) Squeezed convolutional projection by default.
Transformer Block design. The original position-wise lin-
ear projection layer could be trivially implemented using a
convolution layer with kernel size of 1 × 1.
3.2.2
Efﬁciency Considerations
There are two primary efﬁciency beneﬁts from the design
of our Convolutional Projection layer.
First, we utilize efﬁcient convolutions. Directly using
standard s×s convolutions for the Convolutional Projection
would require s2C2 parameters and O(s2C2T) FLOPs,
where C is the token channel dimension, and T is the num-
ber of tokens for processing. Instead, we split the standard
s × s convolution into a depth-wise separable convolution
[16]. In this way, each of the proposed Convolutional Pro-
jection would only introduce an extra of s2C parameters
and O(s2CT) FLOPs compared to the original position-
wise linear projection, which are negligible with respect to
the total parameters and FLOPs of the models.
Second, we leverage the proposed Convolutional Projec-
tion to reduce the computation cost for the MHSA opera-
tion. The s × s Convolutional Projection permits reducing
the number of tokens by using a stride larger than 1. Fig-
ure 3 (c) shows the Convolutional Projection, where the key
and value projection are subsampled by using a convolu-
tion with stride larger than 1. We use a stride of 2 for key
and value projection, leaving the stride of 1 for query un-
changed. In this way, the number of tokens for key and
value is reduced 4 times, and the computational cost is re-
duced by 4 times for the later MHSA operation. This comes
with a minimal performance penalty, as neighboring pix-
els/patches in images tend to have redundancy in appear-
ance/semantics. In addition, the local context modeling of
the proposed Convolutional Projection compensates for the
loss of information incurred by resolution reduction.
3.3. Methodological Discussions
Removing Positional Embeddings:
The introduction of
Convolutional Projections for every Transformer block,
combined with the Convolutional Token Embedding, gives
us the ability to model local spatial relationships through the
network. This built-in property allows dropping the position
embedding from the network without hurting performance,
as evidenced by our experiments (Section 4.4), simplifying
design for vision tasks with variable input resolution.
Relations to Concurrent Work:
Recently, two more re-
lated concurrent works also propose to improve ViT by in-
corporating elements of CNNs to Transformers. Tokens-
to-Token ViT [41] implements a progressive tokenization,
and then uses a Transformer-based backbone in which the
length of tokens is ﬁxed. By contrast, our CvT implements
a progressive tokenization by a multi-stage process – con-
taining both convolutional token embeddings and convolu-
tional Transformer blocks in each stage. As the length of
tokens are decreased in each stage, the width of the tokens
(dimension of feature) can be increased, allowing increased
richness of representations at each feature spatial resolu-
tion. Additionally, whereas T2T concatenates neighboring
tokens into one new token, leading to increasing the com-
plexity of memory and computation, our usage of convolu-
tional token embedding directly performs contextual learn-
ing without concatenation, while providing the ﬂexibility
of controlling stride and feature dimension. To manage the
complexity, T2T has to consider a deep-narrow architecture
design with smaller hidden dimensions and MLP size than
ViT in the subsequent backbone. Instead, we changed pre-
vious Transformer modules by replacing the position-wise
linear projection with our convolutional projection
Pyramid Vision Transformer (PVT) [34] overcomes the
difﬁculties of porting ViT to various dense prediction tasks.
In ViT, the output feature map has only a single scale with
low resolution. In addition, computations and memory cost
are relatively high, even for common input image sizes. To
address this problem, both PVT and our CvT incorporate
pyramid structures from CNNs to the Transformers struc-
ture. Compared with PVT, which only spatially subsam-
ples the feature map or key/value matrices in projection, our
CvT instead employs convolutions with stride to achieve
this goal. Our experiments (shown in Section 4.4) demon-
5

Output Size
Layer Name
CvT-13
CvT-21
CvT-W24
Stage1
56 × 56
Conv. Embed.
7 × 7, 64, stride 4
7 × 7, 192, stride 4
56 × 56
Conv. Proj.
MHSA
MLP


3 × 3, 64
H1 = 1, D1 = 64
R1 = 4

× 1


3 × 3, 64
H1 = 1, D1 = 64
R1 = 4

× 1


3 × 3, 192
H1 = 3, D1 = 192
R1 = 4

× 2
Stage2
28 × 28
Conv. Embed.
3 × 3, 192, stride 2
3 × 3, 768, stride 2
28 × 28
Conv. Proj.
MHSA
MLP


3 × 3, 192
H2 = 3, D2 = 192
R2 = 4

× 2


3 × 3, 192
H2 = 3, D2 = 192
R2 = 4

× 4


3 × 3, 768
H2 = 12, D2 = 768
R2 = 4

× 2
Stage3
14 × 14
Conv. Embed.
3 × 3, 384, stride 2
3 × 3, 1024, stride 2
14 × 14
Conv. Proj.
MHSA
MLP


3 × 3, 384
H3 = 6, D3 = 384
R3 = 4

× 10


3 × 3, 384
H3 = 6, D3 = 384
R3 = 4

× 16


3 × 3, 1024
H3 = 16, D3 = 1024
R3 = 4

× 20
Head
1 × 1
Linear
1000
Params
19.98 M
31.54 M
276.7 M
FLOPs
4.53 G
7.13 G
60.86 G
Table 2: Architectures for ImageNet classiﬁcation. Input image size is 224 × 224 by default. Conv. Embed.: Convolutional
Token Embedding. Conv. Proj.: Convolutional Projection. Hi and Di is the number of heads and embedding feature
dimension in the ith MHSA module. Ri is the feature dimension expansion ratio in the ith MLP layer.
strate that the fusion of local neighboring information plays
an important role on the performance.
4. Experiments
In this section, we evaluate the CvT model on large-scale
image classiﬁcation datasets and transfer to various down-
stream datasets. In addition, we perform through ablation
studies to validate the design of the proposed architecture.
4.1. Setup
For evaluation, we use the ImageNet dataset, with 1.3M
images and 1k classes, as well as its superset ImageNet-22k
with 22k classes and 14M images [9]. We further trans-
fer the models pretrained on ImageNet-22k to downstream
tasks, including CIFAR-10/100 [19], Oxford-IIIT-Pet [23],
Oxford-IIIT-Flower [22], following [18, 11].
Model Variants
We instantiate models with different pa-
rameters and FLOPs by varying the number of Transformer
blocks of each stage and the hidden feature dimension used,
as shown in Table 2. Three stages are adapted. We de-
ﬁne CvT-13 and CvT-21 as basic models, with 19.98M and
31.54M paramters. CvT-X stands for Convolutional vision
Transformer with X Transformer Blocks in total. Addition-
ally, we experiment with a wider model with a larger token
dimension for each stage, namely CvT-W24 (W stands for
Wide), resulting 298.3M parameters, to validate the scaling
ability of the proposed architecture.
Training
AdamW [21] optimizer is used with the weight
decay of 0.05 for our CvT-13, and 0.1 for our CvT-21 and
CvT-W24.
We train our models with an initial learning
rate of 0.02 and a total batch size of 2048 for 300 epochs,
with a cosine learning rate decay scheduler. We adopt the
same data augmentation and regularization methods as in
ViT [30]. Unless otherwise stated, all ImageNet models are
trained with an 224 × 224 input size.
Fine-tuning
We adopt ﬁne-tuning strategy from ViT [30].
SGD optimizor with a momentum of 0.9 is used for ﬁne-
tuning. As in ViT [30], we pre-train our models at resolu-
tion 224 × 224, and ﬁne-tune at resolution of 384 × 384.
We ﬁne-tune each model with a total batch size of 512,
for 20,000 steps on ImageNet-1k, 10,000 steps on CIFAR-
10 and CIFAR-100, and 500 steps on Oxford-IIIT Pets and
Oxford-IIIT Flowers-102.
4.2. Comparison to state of the art
We compare our method with state-of-the-art classiﬁca-
tion methods including Transformer-based models and rep-
resentative CNN-based models on ImageNet [9], ImageNet
Real [2] and ImageNet V2 [26] datasets in Table 3.
Compared to Transformer based models, CvT achieves
a much higher accuracy with fewer parameters and FLOPs.
CvT-21 obtains a 82.5% ImageNet Top-1 accuracy, which
is 0.5% higher than DeiT-B with the reduction of 63% pa-
rameters and 60% FLOPs. When comparing to concurrent
works, CvT still shows superior advantages. With fewer
paramerters, CvT-13 achieves a 81.6% ImageNet Top-1 ac-
curacy, outperforming PVT-Small [34], T2T-ViTt-14 [41],
TNT-S [14] by 1.7%, 0.8%, 0.2% respectively.
Our architecture designing can be further improved in
terms of model parameters and FLOPs by neural architec-
ture search (NAS) [7]. In particular, we search the proper
stride for each convolution projection of key and value
(stride = 1, 2) and the expansion ratio for each MLP
layer (ratioMLP = 2, 4).
Such architecture candidates
with FLOPs ranging from 2.59G to 4.03G and the num-
6

#Param.
image
FLOPs
ImageNet
Real
V2
Method Type
Network
(M)
size
(G)
top-1 (%)
top-1 (%)
top-1 (%)
Convolutional Networks
ResNet-50 [15]
25
2242
4.1
76.2
82.5
63.3
ResNet-101 [15]
45
2242
7.9
77.4
83.7
65.7
ResNet-152 [15]
60
2242
11
78.3
84.1
67.0
Transformers
ViT-B/16 [11]
86
3842
55.5
77.9
83.6
–
ViT-L/16 [11]
307
3842
191.1
76.5
82.2
–
DeiT-S [30][arxiv 2020]
22
2242
4.6
79.8
85.7
68.5
DeiT-B [30][arxiv 2020]
86
2242
17.6
81.8
86.7
71.5
PVT-Small [34][arxiv 2021]
25
2242
3.8
79.8
–
–
PVT-Medium [34][arxiv 2021]
44
2242
6.7
81.2
–
–
PVT-Large [34][arxiv 2021]
61
2242
9.8
81.7
–
–
T2T-ViTt-14 [41][arxiv 2021]
22
2242
6.1
80.7
–
–
T2T-ViTt-19 [41][arxiv 2021]
39
2242
9.8
81.4
–
–
T2T-ViTt-24 [41][arxiv 2021]
64
2242
15.0
82.2
–
–
TNT-S [14][arxiv 2021]
24
2242
5.2
81.3
–
–
TNT-B [14][arxiv 2021]
66
2242
14.1
82.8
–
–
Convolutional Transformers
Ours: CvT-13
20
2242
4.5
81.6
86.7
70.4
Ours: CvT-21
32
2242
7.1
82.5
87.2
71.3
Ours: CvT-13↑384
20
3842
16.3
83.0
87.9
71.9
Ours: CvT-21↑384
32
3842
24.9
83.3
87.7
71.9
Ours: CvT-13-NAS
18
2242
4.1
82.2
87.5
71.3
Convolution Networks22k
BiT-M↑480 [18]
928
4802
837
85.4
–
–
Transformers22k
ViT-B/16↑384 [11]
86
3842
55.5
84.0
88.4
–
ViT-L/16↑384 [11]
307
3842
191.1
85.2
88.4
–
ViT-H/16↑384 [11]
632
3842
–
85.1
88.7
–
Convolutional Transformers22k
Ours: CvT-13↑384
20
3842
16
83.3
88.7
72.9
Ours: CvT-21↑384
32
3842
25
84.9
89.8
75.6
Ours: CvT-W24↑384
277
3842
193.2
87.7
90.6
78.8
Table 3: Accuracy of manual designed architecture on ImageNet [9], ImageNet Real [2] and ImageNet V2 matched fre-
quency [26]. Subscript 22k indicates the model pre-trained on ImageNet22k [9], and ﬁnetuned on ImageNet1k with the input
size of 384 × 384, except BiT-M [18] ﬁnetuned with input size of 480 × 480.
ber of model parameters ranging from 13.66M to 19.88M
construct the search space. The NAS is evaluated directly
on ImageNet-1k. The searched CvT-13-NAS, a bottleneck-
like architecture with stride = 2, ratioMLP = 2 at the ﬁrst
and last stages, and stride = 1, ratioMLP = 4 at most lay-
ers of the middle stage, reaches to a 82.2% ImageNet Top-1
accuracy with fewer model parameters than CvT-13.
Compared to CNN-based models, CvT further closes the
performance gap of Transformer-based models. Our small-
est model CvT-13 with 20M parameters and 4.5G FLOPs
surpasses the large ResNet-152 model by 3.2% on Ima-
geNet Top-1 accuracy, while ResNet-151 has 3 times the
parameters of CvT-13.
Furthermore, when more data are involved, our wide
model CvT-W24* pretrained on ImageNet-22k reaches to
87.7% Top-1 Accuracy on ImageNet without extra data
(e.g. JFT-300M), surpassing the previous best Transformer
based models ViT-L/16 by 2.5% with similar number of
model parameters and FLOPs.
4.3. Downstream task transfer
We further investigate the ability of our models to trans-
fer by ﬁne-tuning models on various tasks, with all models
being pre-trained on ImageNet-22k. Table 4 shows the re-
sults. Our CvT-W24 model is able to obtain the best per-
formance across all the downstream tasks considered, even
when compared to the large BiT-R152x4 [18] model, which
has more than 3× the number of parameters as CvT-W24.
4.4. Ablation Study
We design various ablation experiments to investigate
the effectiveness of the proposed components of our archi-
tecture. First, we show that with our introduction of con-
volutions, position embeddings can be removed from the
7

Model
Param
(M)
CIFAR
10
CIFAR
100
Pets
Flowers
102
BiT-M [18]
928
98.91
92.17
94.46
99.30
ViT-B/16 [11]
86
98.95
91.67
94.43
99.38
ViT-L/16 [11]
307
99.16
93.44
94.73
99.61
ViT-H/16 [11]
632
99.27
93.82
94.82
99.51
Ours: CvT-13
20
98.83
91.11
93.25
99.50
Ours: CvT-21
32
99.16
92.88
94.03
99.62
Ours: CvT-W24
277
99.39
94.09
94.73
99.72
Table 4: Top-1 accuracy on downstream tasks. All the mod-
els are pre-trained on ImageNet-22k data
Method
Model
Param
(M)
Pos. Emb.
ImageNet
Top-1 (%)
a
DeiT-S
22
Default
79.8
b
DeiT-S
22
N/A
78.0
c
CvT-13
20
Every stage
81.5
d
CvT-13
20
First stage
81.4
e
CvT-13
20
Last stage
81.4
f
CvT-13
20
N/A
81.6
Table 5: Ablations on position embedding.
model. Then, we study the impact of each of the proposed
Convolutional Token Embedding and Convolutional Projec-
tion components.
Removing Position Embedding
Given that we have in-
troduced convolutions into the model, allowing local con-
text to be captured, we study whether position embed-
ding is still needed for CvT. The results are shown in Ta-
ble 5, and demonstrate that removing position embedding
of our model does not degrade the performance. There-
fore, position embeddings have been removed from CvT
by default. As a comparison, removing the position em-
bedding of DeiT-S would lead to 1.8% drop of ImageNet
Top-1 accuracy, as it does not model image spatial relation-
ships other than by adding the position embedding. This
further shows the effectiveness of our introduced convolu-
tions. Position Embedding is often realized by ﬁxed-length
learn-able vectors, limiting the trained model adaptation of
variable-length input. However, a wide range of vision ap-
plications take variable image resolutions.
Recent work
CPVT [6] tries to replace explicit position embedding of
Vision Transformers with a conditional position encodings
module to model position information on-the-ﬂy. CvT is
able to completely remove the positional embedding, pro-
viding the possibility of simplifying adaption to more vision
tasks without requiring a re-designing of the embedding.
Method
Conv.
Embed.
Pos.
Embed.
#Param
(M)
ImageNet
top-1 (%)
a
19.5
80.7
b

19.9
81.1
c


20.3
81.4
d

20.0
81.6
Table 6: Ablations on Convolutional Token Embedding.
Method
Conv. Proj. KV.
stride
Params
(M)
FLOPs
(G)
ImageNet
top-1 (%)
a
1
20
6.55
82.3
b
2
20
4.53
81.6
Table 7: Ablations on Convolutional Projection with differ-
ent strides for key and value projection. Conv. Proj. KV.:
Convolutional Projection for key and value. We apply Con-
volutional Projection in all Transformer blocks.
Method
Conv. Projection
Imagenet
top-1 (%)
Stage 1
Stage 2
Stage 3
a
80.6
b

80.8
c


81.0
d



81.6
#Blocks
1
2
10
Table 8:
Ablations on Convolutional Projection v.s.
Position-wise Linear Projection.
 indicates the use of
Convolutional Projection, otherwise use Position-wise Lin-
ear Projection.
Convolutional Token Embedding
We study the effec-
tiveness of the proposed Convolutional Token Embedding,
and Table 6 shows the results.
Table 6d is the CvT-13
model. When we replace the Convolutional Token Embed-
ding with non-overlapping Patch Embedding [11], the per-
formance drops 0.8% (Table 6a v.s. Table 6d). When po-
sition embedding is used, the introduction of Convolutional
Token Embedding still obtains 0.3% improvement (Table 6b
v.s.
Table 6c).
Further, when using both Convolutional
Token Embedding and position embedding as Table 6d, it
slightly drops 0.1% accuracy. These results validate the in-
troduction of Convolutional Token Embedding not only im-
proves the performance, but also helps CvT model spatial
relationships without position embedding.
Convolutional Projection
First, we compare the pro-
posed Convolutional Projection with different strides in Ta-
ble 7. By using a stride of 2 for key and value projection,
we observe a 0.3% drop in ImageNet Top-1 accuracy, but
8

with 30% fewer FLOPs. We choose to use Convolutional
Projection with stride 2 for key and value as default for less
computational cost and memory usage.
Then, we study how the proposed Convolutional Pro-
jection affects the performance by choosing whether to use
Convolutional Projection or the regular Position-wise Lin-
ear Projection for each stage. The results are shown in Ta-
ble 8. We observe that replacing the original Position-wise
Linear Projection with the proposed Convolutional Projec-
tion improves the Top-1 Accuracy on ImageNet from 80.6%
to 81.5%. In addition, performance continually improves as
more stages use the design, validating this approach as an
effective modeling strategy.
5. Conclusion
In this work, we have presented a detailed study of in-
troducing convolutions into the Vision Transformer archi-
tecture to merge the beneﬁts of Transformers with the ben-
eﬁts of CNNs for image recognition tasks. Extensive ex-
periments demonstrate that the introduced convolutional to-
ken embedding and convolutional projection, along with the
multi-stage design of the network enabled by convolutions,
make our CvT architecture achieve superior performance
while maintaining computational efﬁciency. Furthermore,
due to the built-in local context structure introduced by con-
volutions, CvT no longer requires a position embedding,
giving it a potential advantage for adaption to a wide range
of vision tasks requiring variable input resolution.
References
[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton.
Layer normalization, 2016. 4
[2] Lucas Beyer, Olivier J H´enaff, Alexander Kolesnikov, Xi-
aohua Zhai, and A¨aron van den Oord. Are we done with
imagenet? arXiv preprint arXiv:2006.07159, 2020. 6, 7
[3] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas
Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-
end object detection with transformers. In European Confer-
ence on Computer Vision, pages 213–229. Springer, 2020.
2
[4] Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping
Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, and
Wen Gao. Pre-trained image processing transformer. arXiv
preprint arXiv:2012.00364, 2020. 2
[5] Franc¸ois Chollet. Xception: Deep learning with depthwise
separable convolutions.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
1251–1258, 2017. 2, 4
[6] Xiangxiang Chu, Bo Zhang, Zhi Tian, Xiaolin Wei, and
Huaxia Xia. Do we really need explicit position encodings
for vision transformers? arXiv preprint arXiv:2102.10882,
2021. 3, 8
[7] Xiyang Dai, Dongdong Chen, Mengchen Liu, Yinpeng
Chen, and Lu YUan. Da-nas: Data adapted pruning for efﬁ-
cient neural architecture search. In European Conference on
Computer Vision, 2020. 6
[8] Zhigang Dai, Bolun Cai, Yugeng Lin, and Junying Chen.
Up-detr: Unsupervised pre-training for object detection with
transformers. arXiv preprint arXiv:2011.09094, 2020. 2
[9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In 2009 IEEE conference on computer vision and
pattern recognition, pages 248–255. Ieee, 2009. 6, 7
[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. BERT: Pre-training of deep bidirectional trans-
formers for language understanding. In Proceedings of the
2019 Conference of the North American Chapter of the As-
sociation for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers), pages
4171–4186, Minneapolis, Minnesota, 2019. Association for
Computational Linguistics. 1, 2
[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, et al. An image is worth 16x16 words: Trans-
formers for image recognition at scale.
arXiv preprint
arXiv:2010.11929, 2020. 1, 2, 4, 5, 6, 7, 8
[12] Nico Engel, Vasileios Belagiannis, and Klaus Dietmayer.
Point transformer.
arXiv preprint arXiv:011.00931, 2020.
2
[13] Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Par-
mar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zheng-
dong Zhang, Yonghui Wu, et al. Conformer: Convolution-
augmented transformer for speech recognition.
arXiv
preprint arXiv:2005.08100, 2020. 3, 4
[14] Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu,
and Yunhe Wang. Transformer in transformer. arXiv preprint
arXiv:2103.00112, 2021. 1, 3, 6, 7
[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 770–778, 2016. 1, 2, 4, 7
[16] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry
Kalenichenko, Weijun Wang, Tobias Weyand, Marco An-
dreetto, and Hartwig Adam. Mobilenets: Efﬁcient convolu-
tional neural networks for mobile vision applications. arXiv
preprint arXiv:1704.04861, 2017. 5
[17] Han Hu, Zheng Zhang, Zhenda Xie, and Stephen Lin. Lo-
cal relation networks for image recognition. arXiv preprint
arXiv:1904.11491, 2019. 3
[18] Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan
Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby.
Big transfer (bit): General visual representation learning.
arXiv preprint arXiv:1912.11370, 6(2):8, 2019. 1, 6, 7
[19] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple
layers of features from tiny images. 2009. 6
[20] Yann Lecun, Patrick Haffner, L´eon Bottou, and Yoshua Ben-
gio.
Object recognition with gradient-based learning.
In
Contour and Grouping in Computer Vision. Springer, 1999.
2, 4
[21] Ilya Loshchilov and Frank Hutter. Decoupled weight decay
regularization. arXiv preprint arXiv:1711.05101, 2017. 6
9

[22] Maria-Elena Nilsback and Andrew Zisserman. Automated
ﬂower classiﬁcation over a large number of classes. In In-
dian Conference on Computer Vision, Graphics and Image
Processing, Dec 2008. 6
[23] Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and
C. V. Jawahar. Cats and dogs. In IEEE Conference on Com-
puter Vision and Pattern Recognition, 2012. 6
[24] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz
Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Im-
age transformer. In International Conference on Machine
Learning, pages 4055–4064. PMLR, 2018. 2
[25] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya
Sutskever. Improving language understanding by generative
pre-training. 2018. 2
[26] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and
Vaishaal Shankar. Do imagenet classiﬁers generalize to im-
agenet? In International Conference on Machine Learning,
pages 5389–5400. PMLR, 2019. 6, 7
[27] Aravind Srinivas, Tsung-Yi Lin, Niki Parmar, Jonathon
Shlens, Pieter Abbeel, and Ashish Vaswani.
Bottle-
neck transformers for visual recognition.
arXiv preprint
arXiv:2101.11605, 2021. 3
[28] Zhiqing Sun, Shengcao Cao, Yiming Yang, and Kris Kitani.
Rethinking transformer-based set prediction for object detec-
tion. arXiv preprint arXiv:2011.10881, 2020. 2
[29] Mingxing Tan and Quoc Le. Efﬁcientnet: Rethinking model
scaling for convolutional neural networks. In International
Conference on Machine Learning, pages 6105–6114. PMLR,
2019. 2
[30] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco
Massa, Alexandre Sablayrolles, and Herv´e J´egou. Training
data-efﬁcient image transformers & distillation through at-
tention. arXiv preprint arXiv:2012.12877, 2020. 1, 2, 4, 6,
7
[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia
Polosukhin. Attention is all you need. In Isabelle Guyon,
Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob
Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors,
Advances in Neural Information Processing Systems 30: An-
nual Conference on Neural Information Processing Systems
2017, December 4-9, 2017, Long Beach, CA, USA, pages
5998–6008, 2017. 1, 2
[32] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill,
Omer Levy, and Samuel R. Bowman. GLUE: A multi-task
benchmark and analysis platform for natural language un-
derstanding. In 7th International Conference on Learning
Representations, ICLR 2019, New Orleans, LA, USA, May
6-9, 2019. OpenReview.net, 2019. 1
[33] Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille,
and Liang-Chieh Chen. Max-deeplab: End-to-end panop-
tic segmentation with mask transformers.
arXiv preprint
arXiv:2012.00759, 2020. 2
[34] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao
Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao.
Pyramid vision transformer:
A versatile backbone for
dense prediction without convolutions.
arXiv preprint
arXiv:2102.12122, 2021. 1, 3, 4, 5, 6, 7
[35] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaim-
ing He. Non-local neural networks. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 7794–7803, 2018. 3
[36] Yuqing Wang, Zhaoliang Xu, Xinlong Wang, Chunhua Shen,
Baoshan Cheng, Hao Shen, and Huaxia Xia.
End-to-
end video instance segmentation with transformers. arXiv
preprint arXiv:2011.14503, 2020. 2
[37] Yujing Wang, Yaming Yang, Jiangang Bai, Mingliang
Zhang, Jing Bai, Jing Yu, Ce Zhang, Gao Huang, and Yunhai
Tong. Evolving attention with residual convolutions. arXiv
preprint arXiv:2102.12895, 2021. 3
[38] Felix Wu, Angela Fan, Alexei Baevski, Yann N Dauphin,
and Michael Auli. Pay less attention with lightweight and dy-
namic convolutions. arXiv preprint arXiv:1901.10430, 2019.
3
[39] Zhanghao Wu, Zhijian Liu, Ji Lin, Yujun Lin, and Song
Han. Lite transformer with long-short range attention. arXiv
preprint arXiv:2004.11886, 2020. 3, 4
[40] Fuzhi Yang, Huan Yang, Jianlong Fu, Hongtao Lu, and Bain-
ing Guo. Learning texture transformer network for image
super-resolution. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pages
5791–5800, 2020. 2
[41] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi,
Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens-
to-token vit: Training vision transformers from scratch on
imagenet. arXiv preprint arXiv:2101.11986, 2021. 1, 3, 4,
5, 6, 7
[42] Yanhong Zeng, Jianlong Fu, and Hongyang Chao. Learning
joint spatial-temporal transformations for video inpainting.
In European Conference on Computer Vision, pages 528–
543. Springer, 2020. 2
[43] Minghang Zheng, Peng Gao, Xiaogang Wang, Hongsheng
Li, and Hao Dong. End-to-end object detection with adaptive
clustering transformer.
arXiv preprint arXiv:2011.09315,
2020. 2
[44] Luowei Zhou, Yingbo Zhou, Jason J. Corso, Richard Socher,
and Caiming Xiong. End-to-end dense video captioning with
masked transformer. In Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), June
2018. 2
[45] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang
Wang, and Jifeng Dai. Deformable detr: Deformable trans-
formers for end-to-end object detection.
arXiv preprint
arXiv:2010.04159, 2020. 2
10
