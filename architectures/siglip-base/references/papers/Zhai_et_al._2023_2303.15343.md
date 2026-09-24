# Paper (Zhai et al. 2023)

> Source: `https://arxiv.org/abs/2303.15343`

---

Sigmoid Loss for Language Image Pre-Training

Xiaohua Zhai(cid:63) Basil Mustafa Alexander Kolesnikov Lucas Beyer(cid:63)
Google DeepMind, Z¨urich, Switzerland
{xzhai, basilm, akolesnikov, lbeyer}@google.com

Abstract

We propose a simple pairwise Sigmoid loss
for
Language-Image Pre-training (SigLIP). Unlike standard
contrastive learning with softmax normalization, the sig-
moid loss operates solely on image-text pairs and does not
require a global view of the pairwise similarities for nor-
malization. The sigmoid loss simultaneously allows fur-
ther scaling up the batch size, while also performing bet-
ter at smaller batch sizes. Combined with Locked-image
Tuning, with only four TPUv4 chips, we train a SigLiT
model that achieves 84.5% ImageNet zero-shot accuracy
in two days. The disentanglement of the batch size from
the loss further allows us to study the impact of exam-
ples vs pairs and negative to positive ratio. Finally, we
push the batch size to the extreme, up to one million, and
ﬁnd that the beneﬁts of growing batch size quickly dimin-
ish, with a more reasonable batch size of 32 k being suf-
ﬁcient. We release our models at https://github.
com/google-research/big_vision and hope our
research motivates further explorations in improving the
quality and efﬁciency of language-image pre-training.

1. Introduction

Contrastive pre-training using weak supervision from
image-text pairs found on the web is becoming the go-to
method for obtaining generic computer vision backbones,
slowly replacing pre-training on large labelled multi-class
datasets. The high-level idea is to simultaneously learn
an aligned representation space for images and texts using
paired data. Seminal works CLIP [36] and ALIGN [23] es-
tablished the viability of this approach at a large scale, and
following their success, many large image-text datasets be-
came available privately [59, 13, 21, 49] and publicly [40,
6, 15, 7, 41].

The standard recipe to pre-train such models leverages
the image-text contrastive objective. It aligns the image and

(cid:63)equal contribution

Table 1: SigLiT and SigLIP results. Sigmoid loss is mem-
ory efﬁcient, allows larger batch sizes (BS) that unlocks
language image pre-training with a small number of chips.
SigLiT model with a frozen public
B/8 checkpoint [42],
trained on the LiT image-text dataset [59] using four TPU-
v4 chips for one day, achieves 79.7% 0-shot accuracy on
ImageNet. The same setup with a g/14 checkpoint [58]
leads to 84.5% accuracy, trained for two days. With a pub-
lic unlocked
B/16 image checkpoint [42], trained on the
WebLI dataset [13], SigLIP achieves 71.0% 0-shot accu-
racy using 16 TPU-v4 chips for three days. The last two
rows show results with randomly initialized models.

Image Text

BS #TPUv4 Days

INet-0

SigLiT
SigLiT

SigLIP
SigLIP
SigLIP

B/8
g/14

B/16
B/16
B/16

L∗
L

32 k
20 k


79.8
84.5

16 k
32 k
32 k

B
B
B
∗ We use a variant of the L model with 12 layers.

71.0
72.1
73.4


text embeddings for matching (positive) image-text pairs
while making sure that unrelated (negative) image-text pairs
are dissimilar in the embedding space. This is achieved via a
batch-level softmax-based contrastive loss, applied twice to
normalize the pairwise similarity scores across all images,
then all texts. A naive implementation of the softmax is
numerically unstable; it is usually stabilized by subtracting
the maximum input value before applying the softmax [18],
which requires another pass over the full batch.

In this paper, we propose a simpler alternative: the sig-
moid loss. It does not require any operation across the full
batch and hence greatly simpliﬁes the distributed loss im-
plementation and boosts efﬁciency. Additionally, it con-
ceptually decouples the batch size from the deﬁnition of
the task. We compare the proposed sigmoid loss with the
standard softmax loss across multiple setups.
In partic-
ular, we investigate sigmoid-based loss with two promi-


nent approaches for image-text learning: CLIP [36] and
LiT [59], which we call sigmoid language image pre-
training (SigLIP) and sigmoid LiT (SigLiT), respectively.
We ﬁnd that the sigmoid loss performs signiﬁcantly better
than the softmax loss when the batch size is smaller than
16 k. As the train batch size grows, the gap closes. Impor-
tantly, the sigmoid loss is symmetric, requires just a single
pass, and a typical implementation requires less memory
than the softmax loss. This enables successful training of a
SigLiT model at a batch size of one million. However, we
ﬁnd that the performance saturates with growing batch size,
both for softmax and sigmoid. The good news is that a rea-
sonable batch size, i.e. 32 k, is sufﬁcient for image-text pre-
training. This conclusion also holds for multilingual SigLIP
training on over 100 languages.

In Table 1, we present setups for image-text pre-training
that require a moderate amount of TPUv4 chips for training.
SigLiT is surprisingly efﬁcient, reaching 79.7% zero-shot
accuracy on ImageNet in just a single day on four chips.
SigLIP’s more demanding from-scratch training reaches
73.4% zero-shot accuracy in 5 days with 32 TPUv4 chips.
This compares favorably to prior works such as FLIP [30]
and CLIP [36], which require approximately 5 and 10 days
respectively on 256 TPUv3 cores. When ﬁne-tuning a pre-
in Table 1,
trained vision backbone in SigLIP, denoted as
we found that disabling the weight decay on the pre-trained
backbone leads to better results (see Figure 4 for details).
We hope our work paves the way for making the nascent
language-image pre-training ﬁeld more accessible.

2. Related Work

Contrastive learning with the sigmoid loss. One prior
work proposes a similar sigmoid loss for the task of unsu-
pervised dimensionality reduction [19]; in the scope of con-
trastive image-text learning, the vast majority of works rely
on the softmax-based InfoNCE loss as popularized by [46].
In supervised classiﬁcation, the sigmoid loss has already
been shown to be slightly more effective and robust than
the softmax loss [3, 51].

Contrastive language-image pre-training has become
popular since CLIP [36] and ALIGN [23] applied softmax
contrastive learning [60, 46, 10, 24] to large-scale image-
text datasets. Both models perform very well on zero-shot
transfer tasks, including classiﬁcation and retrieval. Follow-
up works show that contrastively pre-trained models pro-
duce good representations for ﬁne-tuning [53, 16], linear
regression [23], object detection [31], semantic segmenta-
tion [33] and video tasks [57].

Generative language-image pre-training Besides soft-
max contrastive pre-training, various alternatives have been
proposed. GIT [49], SimVLM [50], and LEMON [21] suc-
cessfully pre-train models using a generative text decoder

Algorithm 1 Sigmoid loss pseudo-implementation.

: image model embedding [n, dim]
: text model embedding [n, dim]
: learnable temperature and bias
: mini-batch size

1 # img_emb
2 # txt_emb
3 # t_prime, b
4 # n
6 t = exp(t_prime)
7 zimg = l2_normalize(img_emb)
8 ztxt = l2_normalize(txt_emb)
9 logits = dot(zimg, ztxt.T) * t + b
10 labels = 2 * eye(n) - ones(n) # -1 with diagonal 1
11 l = -sum(log_sigmoid(labels * logits)) / n

instead, while CoCa [56] adds such a decoder to the dis-
criminative CLIP/ALIGN setup, thus combining the pros
and cons of both approaches into a single very capable
model. BLIP [28] further proposes CapFilt which uses the
generative decoder to create better captions and the discrim-
inative part of the model to ﬁlter pairs. Language-Image
pre-training is a very active ﬁeld and surveys [8] rapidly be-
come outdated.

Efﬁcient language-image pre-training On the other hand,
few works have tried making language image pre-training
more efﬁcient. LiT [59] and FLIP [30] are notable attempts,
the former requires a pre-trained and locked backbone, and
the latter sacriﬁces quality by randomly dropping visual to-
kens. BASIC [35] and LAION [52] look at scaling batch-
size but only go up to 16 k and 160 k respectively, by using
many hundreds of chips, and for the former also mixing in a
large private classiﬁcation dataset [35, 55]. The recent Lion
optimizer [12] claims to be able to reduce the training cost
to reach similar quality.

3. Method

In this section, we ﬁrst review the widely-used softmax-
based contrastive loss. We then introduce the pairwise sig-
moid loss and discuss its efﬁcient implementation.

Given a mini-batch B = {(I1, T1), (I2, T2), . . . } of
image-text pairs, the contrastive learning objective encour-
ages embeddings of matching pairs (Ii, Ti) to align with
each other, while pushing embeddings of unmatched pairs
(Ii, Tj(cid:54)=i) apart. For practical purposes, it is assumed that
for all images i, the text associated with a different image j
is not related to i, and vice-versa. This assumption is usu-
ally noisy and imperfect.

3.1. Softmax loss for language image pre-training

When using the softmax loss to formalize this objective,
an image model f (·) and a text model g(·) are trained to


(a) Initially each device holds 4
image and 4 text representations.
Each device needs to see the rep-
resentations from other devices
to calculate the full loss.

(b) They each compute the com-
ponent of the loss (highlighted)
for their representations, which
includes the positives.

(c) Texts are swapped across the
devices, so device 1 now has I1:4
and T5:8 etc. The new loss is
computed and accumulated with
the previous.

(d) This repeats till every image
& text pair have interacted, e.g.
device 1 has the loss of I1:4 and
T1:12. A ﬁnal cross-device sum
brings everything together.

Figure 1: Efﬁcient loss implementation demonstrated via a mock setup with 3 devices and a global batch size of 12. There
are no all-gathers, and at any point in time only the bright yellow square (size 4 × 4) is materialized in memory.

minimize the following objective:

−

2|B|

|B|
(cid:88)

i=1


(cid:122)


log



image→text softmax
(cid:125)(cid:124)
etxi·yi
j=1 etxi·yj

(cid:80)|B|

(cid:123)

+

(cid:122)
log

text→image softmax
(cid:125)(cid:124)
etxi·yi
j=1 etxj ·yi

(cid:80)|B|


(cid:123)





and yi = g(Ti)
where xi = f (Ii)
. In this paper, we
(cid:107)g(Ti)(cid:107)2
(cid:107)f (Ii)(cid:107)2
adopt the vision transformer architecture [17] for images
and the transformer architecture [47] for texts. Note that
due to the asymmetry of the softmax loss, the normalization
is independently performed two times: across images and
across texts [36]. The scalar t is parametrized as exp(t(cid:48)),
where t(cid:48) is a global freely learnable parameter.

3.2. Sigmoid loss for language image pre-training

Instead of the softmax-based contrastive loss, we pro-
pose a simpler alternative that does not require computing
global normalization factors. The sigmoid-based loss pro-
cesses every image-text pair independently, effectively turn-
ing the learning problem into the standard binary classiﬁca-
tion on the dataset of all pair combinations, with a positive
labels for the matching pairs (Ii, Ti) and negative labels for
all other pairs (Ii, Tj(cid:54)=i). It is deﬁned as follows:

−

|B|

|B|
(cid:88)

|B|
(cid:88)

i=1

j=1

log

(cid:124)

1 + ezij (−txi·yj +b)
(cid:125)

(cid:123)(cid:122)
Lij

ization, the heavy imbalance coming from the many nega-
tives dominates the loss, leading to large initial optimization
steps attempting to correct this bias. To alleviate this, we
introduce an additional learnable bias term b similar to the
temperature t. We initialize t(cid:48) and b to log 10 and −10 re-
spectively. This makes sure the training starts roughly close
to the prior and does not require massive over-correction.
Algorithm 1 presents a pseudocode implementation of the
proposed sigmoid loss for language image pre-training.

3.3. Efﬁcient “chunked” implementation

Contrastive training typically utilizes data parallelism.
Computing the loss when data is split across D devices
necessitates gathering all embeddings [59] with expensive
all-gathers and, more importantly, the materialization of a
memory-intensive |B| × |B| matrix of pairwise similarities.
The sigmoid loss, however, is particularly amenable to
a memory efﬁcient, fast, and numerically stable implemen-
tation that ameliorates both these issues. Denoting the per-
device batch size as b = |B|

D , the loss is reformulated as:

−

|B|

B: swap negs
across devices
(cid:122)(cid:125)(cid:124)(cid:123)
D
(cid:88)

C: per device
loss
(cid:125)(cid:124)
b(dj +1)
(cid:88)

(cid:122)
b(di+1)
(cid:88)

D
(cid:88)

(cid:123)

Lij

di=1
(cid:124)(cid:123)(cid:122)(cid:125)
A: ∀ device di

dj =1

i=bdi
(cid:124) (cid:123)(cid:122) (cid:125)
all local
positives

j=bdj
(cid:124) (cid:123)(cid:122) (cid:125)
negs from
next device

where zij is the label for a given image and text input, which
equals 1 if they are paired and −1 otherwise. At initial-

This is particularly simple for the sigmoid loss as each pair
is an independent term in the loss. Figure 1 illustrates this


Device 1Device 2Device 3I₁I₂I₃I₄I₅I₆I₇I₈I₉I₁₀I₁₁I₁₂Device 1T₁T₂T₃T₄Device 2T₅T₆T₇T₈Device 3T₉T₁₀T₁₁T₁₂Device 1Device 2Device 3I₁I₂I₃I₄I₅I₆I₇I₈I₉I₁₀I₁₁I₁₂Device 1T₁+–––T₂–+––T₃––+–T₄–––+Device 2T₅+–––T₆–+––T₇––+–T₈–––+Device 3T₉+–––T₁₀–+––T₁₁––+–T₁₂–––+↓↓↓↓↓↓↓↓↓↓↓↓loss33%33%33%33%33%33%33%33%33%33%33%33%Device 1Device 2Device 3Device 1Device 2Device 3I₁I₂I₃I₄I₅I₆I₇I₈I₉I₁₀I₁₁I₁₂Device 3T₁✓✓✓✓––––T₂✓✓✓✓––––T₃✓✓✓✓––––T₄✓✓✓✓––––Device 1T₅––––✓✓✓✓T₆––––✓✓✓✓T₇––––✓✓✓✓T₈––––✓✓✓✓Device 2T₉––––✓✓✓✓T₁₀––––✓✓✓✓T₁₁––––✓✓✓✓T₁₂––––✓✓✓✓↓↓↓↓↓↓↓↓↓↓↓↓loss66%66%66%66%66%66%66%66%66%66%66%66%Device 1Device 2Device 3Device 1Device 2Device 3I₁I₂I₃I₄I₅I₆I₇I₈I₉I₁₀I₁₁I₁₂Device 2T₁✓✓✓✓––––✓✓✓✓T₂✓✓✓✓––––✓✓✓✓T₃✓✓✓✓––––✓✓✓✓T₄✓✓✓✓––––✓✓✓✓Device 3T₅✓✓✓✓✓✓✓✓––––T₆✓✓✓✓✓✓✓✓––––T₇✓✓✓✓✓✓✓✓––––T₈✓✓✓✓✓✓✓✓––––Device 1T₉––––✓✓✓✓✓✓✓✓T₁₀––––✓✓✓✓✓✓✓✓T₁₁––––✓✓✓✓✓✓✓✓T₁₂––––✓✓✓✓✓✓✓✓↓↓↓↓↓↓↓↓↓↓↓↓loss✓✓✓✓✓✓✓✓✓✓✓✓Device 1Device 2Device 3↘↓↙Cross Device Σ
Figure 2: The effect of pre-training batch size. Left: SigLiT results, trained for 18B seen examples. Sigmoid loss outper-
forms the softmax loss signiﬁcantly with small batch sizes, and performs similarly at larger batch sizes. We successfully
trained an SigLiT model with up to one million batch size. However, performance for both sigmoid and softmax saturate at
around 32 k batch size. Middle: SigLIP results, trained for 9B seen examples. Both sigmoid loss and softmax loss saturate
at a reasonable batch size, while the peak of the sigmoid loss comes earlier and slightly outperforms the peak of the softmax
loss. A very large batch size hurts both losses. Right: mSigLIP results, trained for 30B seen examples. With a multilingual
setup using over 100 languages, 32 k batch size is surprisingly sufﬁcient and scaling beyond that hurts performance on a
36-language cross-modal retrieval task.

method. In words, we ﬁrst compute the component of the
loss corresponding to the positive pairs, and b − 1 nega-
tive pairs. We then permute representations across devices,
so each device takes negatives from its neighbouring de-
vice (next iteration of sum B). The loss is then calculated
with respect to this chunk (sum C). This is done indepen-
dently in each device, such that each device computes the
loss with respect to its local batch b. Losses can then simply
be summed across all devices (sum A). Individual collec-
tive permutes (for sum B) are fast (and indeed D collective
permutes is typically faster than two all-gathers between D
devices), and the memory cost at any given moment is re-
duced from |B|2 to b2 (for sum C). Usually b is constant as
scaling |B| is achieved by increasing the number of accel-
erators. Due to being quadratic with respect to the batch
size, the vanilla loss computation rapidly bottlenecks scal-
ing up. This chunked approach enabled training with batch
sizes over 1 million on relatively few devices.

4. Results

In this section, we evaluate the proposed SigLiT and
SigLIP models across a wide range of batch sizes. We dis-
cuss what can be achieved with a small number of accel-
erator chips, using both SigLiT and SigLIP recipes. We
also brieﬂy discuss the impact of batch size on multilin-
gual language image pre-training. We ablate the importance
of our large-batch stabilization modiﬁcation and the intro-
duced learned bias term and present a study on the effect of
positive and negative pairs ratio in the sigmoid loss. Lastly,

we explore SigLIP’s data noise robustness.

To validate our models, we report zero-shot transfer re-
sults on the ImageNet dataset [14] and zero-shot retrieval
results across 36 languages on the XM3600 dataset [44].
We use the ScalingViT-Adafactor optimizer [58] by default
for all our experiments.

4.1. SigLiT: Scaling batch size to the limit

Following [59], we use the same precomputed embed-
dings for the images using a ViT-g vision model, and train
a base size text tower from scratch with the same hyperpa-
rameters using the LiT image-text dataset [59].

We perform a study over a wide range of batch sizes,
from 512 to 1 M , demonstrating the impact of batch size
for contrastive learning. Results are presented in Figure 2
(left). When the batch size is smaller than 16 k, sigmoid loss
outperforms softmax loss by a large margin. With growing
batch sizes, we observe that softmax loss quickly catches
up and potentially slightly underperforms sigmoid loss with
a large enough batch size. Overall, we recommend using
the SigLIP recipe for large batch sizes as well, due to the
simplicity, compute savings, and straightforward memory
efﬁcient implementation.

There is a consensus that contrastive learning beneﬁts
from large batch sizes, while most of the existing studies
stop at 64 k batch size [59, 35, 10]. We successfully trained
an SigLiT model at one million batch size, to explore the
limit of contrastive learning. To our surprise, the perfor-
mance saturates at 32 k batch size, further scaling up the
batch size only gives a minor boost, and the model peaks at


28322621024Batch Size (k)8182838485ImageNet 0-shotSigLiT48163298307Batch Size (k)6668707274SigLIPSigmoidSoftmax163265131245Batch Size (k)30313233343536XM T→I 36 lang. avg.mSigLIP
16 k

71.6

34.8

54.7
46.5
9.1
50.1
30.7

32 k

73.2

34.9

54.8
46.2
8.5
49.9
32.5

64 k

73.2

34.4

55.4
46.5
7.9
49.7
32.0

128 k

240 k

73.2

33.6

54.3
46.6
8.1
48.6
30.6

73.1

32.7

54.7
46.6
7.3
49.3
23.7

INet-0

XM avg

XM de
XM en
XM hi
XM ru
XM zh

Table 2: Multilingual SigLIP results with various batch
sizes, pre-trained for 30 billion seen examples. We report
zero-shot transfer results on ImageNet (INet-0) and aver-
aged text to image retrieval results across 36 languages on
the crossmodal 3600 dataset (XM). The full table on 36 lan-
guages can be found in Appendix.

As batch size increases, the gap between the sigmoid and
the softmax losses diminish. SigLIP performs best at batch
size 32 k, whereas the softmax loss required 98 k for optimal
performance and still didn’t outperform the sigmoid based
variant. Scaling further, a larger batch size like 307 k hurts
both losses.

4.3. mSigLIP: Multi-lingual pre-training

We further scale up the training data by keeping all the
100 languages from the WebLI dataset [13]. With multi-
lingual data, one usually needs to use a larger international
vocabulary. We ﬁrst verify the impact of two tokenizers: a
small multilingual vocabulary with 32 k tokens [37], and a
large multilingual vocabulary with 250 k tokens [54]. We
train B-sized ViT and text models for 900 M total exam-
ples seen, and observe slightly more than 1% improvement
when using a larger vocabulary.

However, the token embeddings become huge for very
large vocabulary sizes. Following the standard setup, we
would need to store a N ×W token embedding lookup table
to train the multilingual model, where N is the vocabulary
size mentioned above and W is the embedding dimension
of the text model. To save memory, we propose to use a
“bottlenecked” token embedding. We use N × K embed-
ding matrix and additional K × W projection, where the
bottleneck K is much smaller than W .

In our experiments, we observed that using a large mul-
tilingual vocabulary with a bottleneck can be scaled up as
efﬁciently as using a small multilingual vocabulary. Specif-
ically, by enabling the bottleneck of size K = 96 for Base
architecture with W = 768, we only see about a half per-
cent quality drop on ImageNet zero-shot transfer, compared
to using the full 250k vocabulary.

Figure 3: SigLiT ImageNet 0-shot transfer results with
different training durations. Large batch size results in a
big performance boost, but needs a sufﬁciently long sched-
ule to ramp up, as for short schedules, very large batch size
results in a small number of gradient update steps.

256 k batch size. Our best SigLiT with a B-sized text mode
achieves 84.7% zero-shot transfer accuracy on ImageNet,
while the original LiT paper reports a slightly better 85.2%
score with a 10 times larger g-sized text model. Figure 3
presents the impact of training duration for different batch
sizes. It demonstrates that large, 262 k batch size signiﬁ-
cantly outperforms smaller 8 k batch size when trained for
a sufﬁciently long time. Note, that for short training dura-
tions, large batch size leads to the fewer absolute number of
update steps and thus needs more time to ramp up.

4.2. SigLIP: Sigmoid loss is beneﬁcial for language-

image pre-training

We pre-train SigLIP models on the WebLI dataset [13],
using only English image and text pairs. We use CLIP (We-
bLI) to denote the CLIP baseline pre-trained on WebLI with
the standard softmax loss. We use moderately-sized mod-
els: B/16 ViT for image embeddings and B-sized trans-
former for text embeddings. The input images are resized to
224×224 resolution. The text is tokenized by a 32 k vocab-
ulary sentencepiece tokenizer [27] trained on the English
C4 dataset [37], and a maximum of 16 text tokens are kept.
Figure 2 middle plot shows SigLIP results, With less than
32 k batch size, SigLIP outperforms CLIP (WebLI) base-
lines. On the other end of the scale, the memory efﬁciency
of the sigmoid loss enabled much larger batch sizes. For ex-
ample, with four TPU-v4 chips, we could ﬁt a batch size of
4096 with a Base SigLIP but only 2048 with a correspond-
ing CLIP model. The two advantages together demonstrate
signiﬁcant beneﬁts of the sigmoid loss for language image
pre-training with ﬁxed resources, which will be discussed
in Section 4.5.


450900300018'000Examples Seen [M]7879808182838485ImageNet 0-shot    8k262kSigmoidSoftmax
Figure 4: Top: SigLIP with pre-trained encoders ramps up
quickly. However, only disabling weight decay on the pre-
trained encoder weights leads to stable behavior and good
ImageNet 0-shot transfer results. Bottom: ImageNet 10-
shot transfer results, where decaying the pre-trained weights
leads to deterioration of the pre-trained model visual repre-
sentation quality. Disabling weight decay ﬂattens the curve.

Figure 5: The effect of Adam and AdaFactor’s β2. As
we increase batch-size, we observe more frequent training
instability. This instability seen in the loss curves (top) is
caused by spikes in gradient norm (middle) leading to large
parameter updates (bottom). Decreasing the β2 momentum
stabilizes training. Occasional gradient spikes still happen
(see step at 2B), but do not destabilize the training process.

With the memory improvements, we train mSigLIP
models for various batch sizes, for a total of 30 billion ex-
amples seen. Table 2 and Figure 2 (right plot) show the
results. We were expecting a large batch size to improve
multilingual pre-training, where the model sees more ex-
amples from the same language as hard negatives in a sin-
gle mini-batch. However, we didn’t observe clear improve-
ments with a batch size larger than 32 k. A batch size of
32 k is sufﬁcient for a multilingual setup as well. On the
XM3600 cross-modal retrieval tasks, we found that going
beyond 32 k batch size leads to worse results on average
while on ImageNet zero-shot transfer it stays ﬂat. mSigLIP
sets the new state-of-the-art on XM3600 text to image re-
trieval task, with only a Base size model. Our best result is
34.9%, which is more than 6% higher than the previously
reported result 28.5% [13] with a standard LiT model [59]
using a much larger four billion ViT-e model. We further
scale up mSigLIP training in Section 4.6.

4.4. SigLiT with four TPU-v4 chips

For many practitioners, the important question usually is
“what can be trained with a limited amount of resources?”
We explore the usage of SigLiT models in this section with
only four TPU-v4 chips, as the memory efﬁcient sigmoid
loss is suitable for this application scenario.

We follow the same setup as in section 4.1. We use
the publicly available ViT-AugReg-B/8 [42] model as the
frozen (
) vision tower, and precompute embeddings to ac-
celerate the training [59]. The text model is a Large Trans-
former, but with a depth of only 12 layers (instead of 24).
It is trained using the LION [12] optimizer with decoupled
weight decay 1 × 10−7, linearly warm-up of learning rate
over 6.5k steps up to a peak of 1 × 10−4, followed by a co-
sine decay to 0. We train for a total of 65 000 steps with a
batch size of 32k – this leads to just under one day of train-
ing. Table 1 shows the results when training a model on four
chips for one day, achieving 79.7% 0-shot ImageNet classi-
ﬁcation accuracy; very competitive in this limited resource
regime. With a ViT-g/14 [58] model as the vision tower and
a Large text tower, we can train at 20 k batch size on four
chips for 107 k steps in under two days. This further pushes
the 0-shot ImageNet classiﬁcation accuracy up to 84.5%.

4.5. SigLIP with a small amount of TPU-v4 chips

It’s resource demanding to train a CLIP model from-
scratch in general, with SigLIP it’s possible to ﬁt a larger
train batch size with fewer amount of chips. In this section,
we explore ways to train SigLIP models efﬁciently with pre-
trained weights. We use pre-trained weights to initialize the
image model to accelerate the pre-training, which was orig-


12481624010203040506070INet 0-shot12481624Examples Seen [100M]010203040506070INet 10-shotfrom-scratchfine-tunefine-tune w/o enc.wd3456Loss Lβ2=0.999β2=0.95110||∇wL||1B2B3B4B5BExamples seen24||Δw||
Figure 6: The effect of batch composition. We simulate various batch compositions by masking out negatives, either
randomly, keeping only the hardest, or the easiest. With no masking, we have 16 k negatives for each positive in the batch
(1:16 k) and the strongest masking we apply (1:1.6) results in almost balanced minibatches. In one setting we match total
pairs seen by training for signiﬁcantly longer. We observe ImageNet 0-shot score, the ﬁnal value of the learned bias, and the
average logits of positive and negative pairs. Overall, the imbalance does not seem to be detrimental, but ﬁnding an efﬁcient
way of mining negatives might be beneﬁcial.

inally discussed in [59]. We use the public and unlocked
ViT-AugReg-B/16 [42] model to initialize our vision tower
and ﬁne-tune on the same WebLI English data as used for
SigLIP. In all the experiments, we apply a 0.1 learning rate
multiplier to the pre-trained image tower to make it suitable
for ﬁne-tuning.

Figure 4 presents unlocked

ﬁne-tuning results along-
side from-scratch randomly initialized baselines. We used
16 TPU-v4 chips and train at 16 k batch size for 2.4 B ex-
amples seen. We found that the ﬁne-tuning setup doesn’t
perform well out-of-the-box; this is consistent with prior
works [59] where ﬁnetuning image models degraded visual
representation quality. This is evidenced by ImageNet 10-
shot linear classiﬁcation, where in Figure 4 the ﬁne-tuned
setup is barely better than the from-scratch baseline.

We hypothesize that the default weight decay applied to
the pre-trained weights reduces their effectiveness. Moti-
vated by the ﬁne-tuning recipe from [17, 58, 25], that uses
no weight decay, we also propose disabling weight decay on
the pre-trained weights for SigLIP training. Weight decay
is therefore only applied to the randomly initialized weights
in the text model. This simple modiﬁcation signiﬁcantly
improved SigLIP results. Figure 4 shows that with our im-
proved recipe, SigLIP reaches 71% 0-shot accuracy on Im-
ageNet, using 16k batch size, trained on 16 chips for three
days. We also present from-scratch results in the bottom
rows of Table 1: with 32 TPUv4 chips for only two days,
SigLIP achieves 72.1% 0-shot accuracy. This presents a
signiﬁcant training cost reduction e.g. compared to CLIP
(approx. 2500 TPUv3-days for 72.6%) reported in [30].

4.6. Scaling up SigLIP and mSigLIP

In this section, we scale up SigLIP by “overtraining” the
model [45, 1]. We present results in Table 3 using ViT-B,
ViT-L or So-400m [1] as the vision encoder, with a text en-
coder of the same size (B, L and So-400m respectively).
Following the recipe described in Section 4.2, we train both
models for 40 billion examples seen at batch size 32 k, but
use (256/16)2 = 256 image patches and 64 text tokens (in-
stead of 16). To get SigLIP models for different resolutions,
we train for 5 billion more examples at the target resolution,
with a 100x smaller learning rate and no weight decay. In
Table 3, we report zero-shot classiﬁcation results on Im-
ageNet [14], ObjectNet [2], ImageNet-v2 [39], ImageNet
ReaL [3], and zero-shot image-to-text (I→T) retrieval, text-
to-image (I→T) retrieval results on MSCOCO [11].

We also scale up the multilingual mSigLIP ViT-B model
in the same way. We report image-text retrieval results
across 36 languages on the XM3600 benchmark [44]. The
scaled-up mSigLIP ViT-B model achieves the state-of-the-
art 42.6% image retrieval recall@1 and 54.1% text retrieval
recall@1 for a Base model. This is slightly outperformed
by the Large model in [48] getting 42.96% image retrieval
recall@1. Detailed results are provided in Appendix Table 9
and Figure 8, denoted as *32 k.

4.7. Stabilizing large-batch training

As we move to large batch sizes, the language image pre-
training using transformers becomes increasingly more un-
stable, even when using a modestly-sized model (e.g. Base
size). The reason for these instabilities is large spikes in the


1 : 16k1 : 1.6k1 : 1641 : 161 : 1.6607080ImageNet 0-shotRandomHardHard, matched pairsEasy1 : 16k1 : 1.6k1 : 1641 : 161 : 1.6−15−10−50Learned bias1 : 16k1 : 1.6k1 : 1641 : 161 : 1.6−20−15−10−505Average logit of pos and neg
Method

CLIP
OpenCLIP
EVA-CLIP
SigLIP

SigLIP
SigLIP
SigLIP

CLIP
OpenCLIP
CLIPA-v2
EVA-CLIP
SigLIP

CLIP
CLIPA-v2
EVA-CLIP
SigLIP

OpenCLIP
CLIPA-v2
EVA-CLIP
SigLIP

Image Encoder

ImageNet-1k

COCO R@1

ViT size

# Patches

Validation

B
B
B
B

B
B
B

L
L
L
L
L

L
L
L
L

G (2B)
H (630M)
E (5B)
SO (400M)


1024


68.3
70.2
74.7
76.2

76.7
78.6
79.2

75.5
74.0
79.7
79.8
80.5

76.6
80.3
80.4
82.1

80.1
81.8
82.0
83.2

v2

61.9
62.3
67.0
69.6

70.0
72.1
73.0

69.0
61.1
72.8
72.9
74.2

72.0
73.5
73.8
75.9

73.6
75.6
75.7
77.2

ReaL

ObjectNet

I → T

T → I

-
-
-
82.8

83.1
84.5
84.9

-
-
-
-
85.9

-
-
-
87.0

-
-
-
87.5

55.3
56.0
62.3
70.7

71.3
73.8
74.7

69.9
66.4
71.1
75.3
77.9

70.9
73.1
78.4
81.0

73.0
77.4
79.6
82.9

52.4
59.4
58.7
64.4

65.1
67.5
67.6

56.3
62.1
64.1
63.7
69.5

57.9
65.5
64.1
70.6

67.3
67.2
68.8
70.2

33.1
42.3
42.2
47.2

47.4
49.7
50.4

36.5
46.1
46.3
47.5
51.1

37.1
47.2
47.9
52.7

51.4
49.2
51.1
52.0

Table 3: Comparison with other publicly released models. Our SigLIP models outperform all prior models, e.g. Open-
CLIP [22] and CLIP [36], by a signiﬁcant margin on both zero-shot classiﬁcation and retrieval tasks. Compared to the concur-
rent EVA-CLIP [43] and CLIPA-v2 [29], our SigLIP-L performs better across the board, in both the low and high resolution
cases. Especially noteworthy is the Shape-Optimized 400M parameter ViT [1] architecture, which outperforms all signiﬁ-
cantly larger models. We publicly release our models: https://github.com/google-research/big_vision.

gradient norms, which translate to large-magnitude changes
in the weights that may destabilize the training process,
see Figure 5. We observe that reducing β2 in Adam and
AdaFactor from its default 0.999 to 0.95 (which was sug-
gested in [20, 9]) is enough to stabilize the training. Intu-
itively, this allows recovering from gradient spikes quicker.
We opt for setting β2 = 0.95 for all our experiments.

4.8. Negative ratio in sigmoid loss

One question which arises when shifting the perspective
from the softmax’s “pick the right class” view to the sig-
moid’s “rate this pair” view, is the imbalance in positive
versus negative pairs. For a batch size |B|, the batch con-
tains |B| positive pairs, but |B|2 − |B| negative examples.
In the modest batch-size of 16 k, there are actually 268 M
negative examples for only 16 k positive ones. At the same
time, because the sigmoid loss decomposes into a sum of
per-example losses, we can perform controlled experiments
to study the effect of the mini-batch composition and dis-

tribution of examples visited. We run experiments in the
SigLiT setup at batch-size 16 k for 900 M steps and vary
the composition of the batch by masking out (i.e. ignoring)
enough negative examples to reach a target “positive : neg-
ative” ratio, masking in the following ways:

• Random: Randomly choose negative pairs to mask.

• Hard: Keep hardest negative pairs (highest loss).

• Easy: Keep easiest negatives pairs (lowest loss).

• Hard + matching total pairs seen: Masking exam-
ples while training for a ﬁxed number of steps does
decrease the total number of pairs seen during train-
ing. Hence in the matched pairs setting, we increase
the number of training steps by the masking ratio in
order to keep the number of pairs seen constant.

Figure 6 shows the effect of the various masking strate-
gies. Randomly removing negatives to rebalance does dete-
riorate performance. Keeping the easiest examples does not
work at all, while keeping the hardest negatives does almost


Figure 7: Sigmoid-training increases robustness to data noise. Titles show the type of corruption applied, and x-axes show
the probability with which they are applied. With increasing corruption severity, M-scale models trained with sigmoid loss
for 3.6 billion examples retain superiority over corresponding softmax baseline.

maintain the quality, indicating that, as could be expected,
a lot of the learning on the negative side comes from the
harder examples. This is further conﬁrmed by the slightly
increased performance of training longer on the hardest ex-
amples in order to match the total pairs seen.

We also look at the value of the learned bias at the end of
training as well as the average logit value for positive and
negative examples across these settings, and ﬁnd the result
mostly follows what one would expect: as fewer negatives
are present, the bias and logits become more positive over-
Interestingly, when training with more hard negative
all.
pairs, the average logits of positive pairs stays mostly ﬂat.

This study conﬁrms that (1) the imbalance does not seem
to be a major reason for concern, while at the same time (2)
coming up with an efﬁcient way of including more negative
examples can be promising but is not trivial.

4.9. Bias term in sigmoid loss

We ablate the bias term in the loss function, using the
Base architecture with an 8 k batch size, trained for 900M
examples with the SigLIP setup. Zero-shot transfer results
are reported on ImageNet [14], Oxford-iiit pet [34] and Ci-
far100 [26]. Table 4 presents results with and without a bias
term in the sigmoid loss.

Table 4: Bias (b) and temperature (t(cid:48)) initialization. Re-
sults are reported using Base architecture, 8 k batch size,
trained for 900M examples. Enabling the bias term b with
−10 initialization improves results consistently.

Enabling the bias term with a −10 initialization consis-
tently improves performance across all tasks. This is be-
cause the bias term ensures that the training starts close to
the prior, preventing dramatic over-correction in early op-
timization. In contrast, a randomly chosen bias term ini-
tialization, such as the 0 initialization in Table 4, fails to
address the over-correction issue, leading to signiﬁcantly
worse results. This effect is particularly noticeable when
using a small temperature t(cid:48) initialization. We set the bias
and temperature initialization to b = −10 and t(cid:48) = log 10
(hence t = 10) as the default for all experiments.

4.10. Label noise robustness

Prior works demonstrated improved robustness against
label noise when using the sigmoid loss for classiﬁcation
models [3]. This property would be particularly useful here
in the face of the famously noisy nature of popular large-
scale image-text datasets. In order to study this for SigLIP,
we train M/16 image models alongside an M text model at
batch size 16384 for 3.6 billion seen examples. We corrupt
the training data using one of the following methods:

• Image: With probability p, replace the image with uni-

form random noise.

• Text: With probability p, replace tokenized text with a
new sequence of randomly sampled tokens, up to some
(sampled) sequence length.

• Batch alignment: Randomly shufﬂe the ordering of

p% of the batch.

b

n/a
-10
-10

t(cid:48)

log 10
log 10
log 1
log 10
log 1

INet-0

Pet-0

C100-0

• Image & text: Apply both with probability p each.

62.0
63.0
61.0
61.7
53.7

81.8
82.4
80.0
79.9
73.2

59.9
61.0
60.4
59.0
53.8

• Image, text & batch: Alongside (4), also shufﬂe frac-

tion p of alignments.

Results from varying the likelihood of the corruption are
shown in Figure 7. Models trained with sigmoid loss are
increasingly robust to all kinds of added noise.


0.00.20.40.500.520.540.56ImageNet 0shotImage0.00.20.4TextSigmoidSoftmax0.00.20.4p(corruption)Batch0.00.10.2Image & Text0.00.10.2Image, Text & Batch
5. Conclusion

We conducted a study on two language-image pre-
training instances that used the sigmoid loss: SigLiT and
SigLIP. Our results demonstrate that the sigmoid loss per-
forms better than the softmax baseline, particularly for
small train batch sizes. This loss function is also more mem-
ory efﬁcient, which allows larger train batch sizes without
requiring additional resources. We performed a thorough
investigation of the batch size in contrastive learning. Sur-
prisingly, we found that a relatively modest batch size of
32 k yielded nearly optimal performance. Further studies
have been performed to understand better the introduced
bias term in the sigmoid loss, robustness to data noises and
the impact of positive and negative pairs ratio in the sigmoid
loss. We hope this work will facilitate language-image pre-
training research with limited resources.

Acknowledgements. We thank Daniel Keysers, Ilya Tol-
stikhin, Olivier Bousquet and Michael Tschannen for their
valuable feedback and discussions on this paper. We thank
Joan Puigcerver, Josip Djolonga and Black Hechtman for
discussions on efﬁcient implementations of the chunked
contrastive loss. We thank Kaiming He and Xinlei Chen for
the discussion of β2 to stabilize the training. We also thank
Ross Wightman for spotting a mistake in the pseudocode in
the ﬁrst version of this paper, Boris Dayma and Krzysztof
Maziarz for spotting typos in the second and third versions
which made t vs t(cid:48) confusing. We thank the Google Deep-
mind team for providing a supportive research environment.
We use the big vision codebase [5, 4] for all experi-
ments in this project.


References

[1] Ibrahim Alabdulmohsin, Xiaohua Zhai, Alexander
in shape:
In

Kolesnikov, and Lucas Beyer.
Scaling laws for compute-optimal model design.
NeurIPS, 2023. 7, 8, 17

Getting vit

[2] Andrei Barbu, David Mayo, Julian Alverio, William Luo,
Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and
Boris Katz. ObjectNet: A large-scale bias-controlled dataset
for pushing the limits of object recognition models.
In
NeurIPS, 2019. 7, 17

[3] Lucas Beyer, Olivier J. H´enaff, Alexander Kolesnikov, Xi-
aohua Zhai, and A¨aron van den Oord. Are we done with
imagenet? CoRR, abs/2006.07159, 2020. 2, 7, 9, 17

[4] Lucas Beyer, Xiaohua Zhai, and Alexander Kolesnikov. Bet-

ter plain vit baselines for imagenet-1k, 2022. 10, 17

[5] Lucas Beyer, Xiaohua Zhai, and Alexander Kolesnikov. Big
vision. https://github.com/google-research/
big_vision, 2022. 10, 17

[6] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun
Lee, Woonhyuk Baek,
Coyo-
700m: Image-text pair dataset. https://github.com/
kakaobrain/coyo-dataset, 2022. 1

and Saehoon Kim.

[7] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu
Soricut. Conceptual 12M: Pushing web-scale image-text
pre-training to recognize long-tail visual concepts. In CVPR,
2021. 1

[8] Feilong Chen, Duzhen Zhang, Minglun Han, Xiu-Yi Chen,
Jing Shi, Shuang Xu, and Bo Xu. VLP: A survey on vision-
language pre-training. Int. J. Autom. Comput., 20(1):38–56,
2023. 2

[9] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Hee-
woo Jun, David Luan, and Ilya Sutskever. Generative pre-
In Proceedings of the 37th Interna-
training from pixels.
tional Conference on Machine Learning, ICML 2020, 13-18
July 2020, Virtual Event, volume 119 of Proceedings of Ma-
chine Learning Research, pages 1691–1703. PMLR, 2020.

[10] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Ge-
offrey E. Hinton. A simple framework for contrastive learn-
ing of visual representations. In ICML, 2020. 2, 4

[11] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedan-
tam, Saurabh Gupta, Piotr Doll´ar, and C. Lawrence Zitnick.
Microsoft COCO captions: Data collection and evaluation
server. CoRR, abs/1504.00325, 2015. 7, 17

[12] Xiangning Chen, Chen Liang, Da Huang, Esteban Real,
Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang
Luong, Cho-Jui Hsieh, Yifeng Lu, and Quoc V. Le. Symbolic
discovery of optimization algorithms, 2023. 2, 6

[13] Xi Chen, Xiao Wang, Soravit Changpinyo, A. J. Piergio-
vanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman,
Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander
Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Has-
san Akbari, Gaurav Mishra, Linting Xue, Ashish Thapliyal,
James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini,
Chao Jia, Burcu Karagol Ayan, Carlos Riquelme, Andreas
Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and


Radu Soricut. Pali: A jointly-scaled multilingual language-
image model. CoRR, abs/2209.06794, 2022. 1, 5, 6, 17
[14] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In CVPR, 2009. 4, 7, 9, 17

[15] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin John-
son. Redcaps: Web-curated image-text data created by the
people, for the people.
In Joaquin Vanschoren and Sai-
Kit Yeung, editors, Proceedings of the Neural Information
Processing Systems Track on Datasets and Benchmarks 1,
NeurIPS Datasets and Benchmarks 2021, December 2021,
virtual, 2021. 1

[16] Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen,
Shuyang Gu, Weiming Zhang, Lu Yuan, Dong Chen, Fang
Wen, and Nenghai Yu. Clip itself is a strong ﬁne-tuner:
Achieving 85.7% and 88.0% top-1 accuracy with vit-b and
vit-l on imagenet. CoRR, abs/2212.06138, 2022. 2

[17] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is
worth 16×16 words: Transformers for image recognition at
scale. In ICLR, 2021. 3, 7, 17

[18] Ian Goodfellow, Yoshua Bengio, and Aaron Courville.
http://www.

Deep Learning. MIT Press, 2016.
deeplearningbook.org. 1

[19] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimension-
ality reduction by learning an invariant mapping. In CVPR,
volume 2, 2006. 2

[20] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Pi-
otr Doll´ar, and Ross B. Girshick. Masked autoencoders
In IEEE/CVF Conference on
are scalable vision learners.
Computer Vision and Pattern Recognition, CVPR 2022, New
Orleans, LA, USA, June 18-24, 2022, pages 15979–15988.
IEEE, 2022. 8

[21] Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang,
Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up
vision-language pre-training for image captioning. CoRR,
abs/2111.12233, 2021. 1, 2

[22] Gabriel

Ilharco, Mitchell Wortsman, Nicholas Carlini,
Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok
Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi,
and Ludwig Schmidt. OpenCLIP. Zenodo, 2021. 8

[23] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh,
Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom
Duerig. Scaling up visual and vision-language representation
learning with noisy text supervision. In ICML, 2021. 1, 2

[24] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna,
Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and
In Hugo
Dilip Krishnan. Supervised contrastive learning.
Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-
Florina Balcan, and Hsuan-Tien Lin, editors, Advances in
Neural Information Processing Systems 33: Annual Con-
ference on Neural Information Processing Systems 2020,
NeurIPS 2020, December 6-12, 2020, virtual, 2020. 2
[25] Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan
Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby.


Big transfer (BiT): General visual representation learning. In
ECCV, 2020. 7

[26] Alex Krizhevsky. Learning multiple layers of features from
tiny images. Technical report, Univ. of Toronto, 2009. 9
[27] Taku Kudo and John Richardson. SentencePiece: A sim-
ple and language independent subword tokenizer and detok-
enizer for neural text processing. In EMNLP, 2018. 5, 14

[28] Junnan Li, Dongxu Li, Caiming Xiong, and Steven C. H.
Hoi. BLIP: bootstrapping language-image pre-training for
In
uniﬁed vision-language understanding and generation.
Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba
Szepesv´ari, Gang Niu, and Sivan Sabato, editors, Interna-
tional Conference on Machine Learning, ICML 2022, 17-
23 July 2022, Baltimore, Maryland, USA, volume 162 of
Proceedings of Machine Learning Research, pages 12888–
12900. PMLR, 2022. 2

[29] Xianhang Li, Zeyu Wang, and Cihang Xie. Clipa-v2: Scal-
ing CLIP training with 81.1% zero-shot imagenet accuracy
within a $10, 000 budget; an extra $4, 000 unlocks 81.8%
accuracy. CoRR, abs/2306.15658, 2023. 8

[30] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichten-
hofer, and Kaiming He. Scaling language-image pre-training
via masking. CoRR, abs/2212.00794, 2022. 2, 7

[31] Matthias Minderer, Alexey A. Gritsenko, Austin Stone,
Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy,
Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani,
Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf,
and Neil Houlsby. Simple open-vocabulary object detection.
In Shai Avidan, Gabriel J. Brostow, Moustapha Ciss´e, Gio-
vanni Maria Farinella, and Tal Hassner, editors, Computer
Vision - ECCV 2022 - 17th European Conference, Tel Aviv,
Israel, October 23-27, 2022, Proceedings, Part X, volume
13670 of Lecture Notes in Computer Science, pages 728–
755. Springer, 2022. 2

[32] Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker
Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer,
Inioluwa Deborah Raji, and Timnit Gebru. Model cards
for model reporting. In danah boyd and Jamie H. Morgen-
stern, editors, Proceedings of the Conference on Fairness,
Accountability, and Transparency, FAT* 2019, Atlanta, GA,
USA, January 29-31, 2019, pages 220–229. ACM, 2019. 17
[33] Jishnu Mukhoti, Tsung-Yu Lin, Omid Poursaeed, Rui Wang,
Ashish Shah, Philip H. S. Torr, and Ser-Nam Lim. Open
vocabulary semantic segmentation with patch aligned con-
trastive learning, 2022. 2

[34] Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and
C. V. Jawahar. Cats and dogs. In IEEE Conference on Com-
puter Vision and Pattern Recognition, 2012. 9

[35] Hieu Pham, Zihang Dai, Golnaz Ghiasi, Hanxiao Liu,
Adams Wei Yu, Minh-Thang Luong, Mingxing Tan, and
Quoc V. Le. Combined scaling for zero-shot transfer learn-
ing. CoRR, abs/2111.10050, 2021. 2, 4

[36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever. Learning transferable visual
models from natural language supervision. In ICML, 2021.
1, 2, 3, 8

[37] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee,
Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and
Peter J. Liu. Exploring the limits of transfer learning with a
uniﬁed text-to-text transformer. arXiv e-prints, 2019. 5, 14

[38] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee,
Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and
Peter J. Liu. Exploring the limits of transfer learning with
J. Mach. Learn. Res.,
a uniﬁed text-to-text transformer.
21:140:1–140:67, 2020. 17

[39] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and
Vaishaal Shankar. Do ImageNet classiﬁers generalize to Im-
ageNet? In ICML, 2019. 7, 17

[40] Christoph Schuhmann, Romain Beaumont, Richard Vencu,
Cade Gordon, Ross Wightman, Mehdi Cherti, Theo
Coombes, Aarush Katta, Clayton Mullis, Mitchell Worts-
man, Patrick Schramowski, Srivatsa Kundurthy, Katherine
Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia
Jitsev. LAION-5B: an open large-scale dataset for training
next generation image-text models. CoRR, abs/2210.08402,
2022. 1

[41] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael
Bendersky, and Marc Najork. WIT: wikipedia-based image
text dataset for multimodal multilingual machine learning.
CoRR, abs/2103.01913, 2021. 1

[42] Andreas Steiner, Alexander Kolesnikov, Xiaohua Zhai, Ross
Wightman, Jakob Uszkoreit, and Lucas Beyer. How to train
your ViT? Data, augmentation, and regularization in vision
transformers. CoRR, abs/2106.10270, 2021. 1, 6, 7

[43] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue
Cao. EVA-CLIP: improved training techniques for CLIP at
scale. CoRR, abs/2303.15389, 2023. 8

[44] Ashish V. Thapliyal, Jordi Pont-Tuset, Xi Chen, and Radu
Soricut. Crossmodal-3600: A massively multilingual mul-
timodal evaluation dataset.
In Yoav Goldberg, Zornitsa
Kozareva, and Yue Zhang, editors, Proceedings of the 2022
Conference on Empirical Methods in Natural Language Pro-
cessing, EMNLP 2022, Abu Dhabi, United Arab Emirates,
December 7-11, 2022, pages 715–729. Association for Com-
putational Linguistics, 2022. 4, 7, 17

[45] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste
Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aur´elien
Rodriguez, Armand Joulin, Edouard Grave, and Guillaume
Lample. Llama: Open and efﬁcient foundation language
models. CoRR, abs/2302.13971, 2023. 7

[46] A¨aron van den Oord, Yazhe Li, and Oriol Vinyals. Repre-
sentation learning with contrastive predictive coding. CoRR,
abs/1807.03748, 2018. 2

[47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia
Polosukhin. Attention is all you need. In NeurIPS, 2017. 3,

[48] Alexander Visheratin. Nllb-clip – train performant multilin-

gual image retrieval model on a budget, 2023. 7

[49] Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li,
Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang.
GIT: A generative image-to-text transformer for vision and
language. CoRR, abs/2205.14100, 2022. 1, 2


MLHC 2022, 5-6 August 2022, Durham, NC, USA, volume
182 of Proceedings of Machine Learning Research, pages
2–25. PMLR, 2022. 2

[50] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia
Tsvetkov, and Yuan Cao. Simvlm: Simple visual language
model pretraining with weak supervision. In The Tenth In-
ternational Conference on Learning Representations, ICLR
2022, Virtual Event, April 25-29, 2022. OpenReview.net,
2022. 2

[51] Ross Wightman, Hugo Touvron, and Herv´e J´egou. Resnet
strikes back: An improved training procedure in timm.
CoRR, abs/2110.00476, 2021. 2

[52] Mitchell Wortsman. Reaching 80% zero-shot accuracy with
https:

OpenCLIP: VIT-G/14 trained on LAION-2B.
//web.archive.org/web/20230127012732/
https://laion.ai/blog/giant-openclip/. 2

[53] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim,
Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gon-
tijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok
Namkoong, and Ludwig Schmidt. Robust ﬁne-tuning of
In IEEE/CVF Conference on Computer
zero-shot models.
Vision and Pattern Recognition, CVPR 2022, New Orleans,
LA, USA, June 18-24, 2022, pages 7949–7961. IEEE, 2022.

[54] Linting Xue, Noah Constant, Adam Roberts, Mihir Kale,
Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin
Raffel. mT5: A massively multilingual pre-trained text-to-
text transformer. In NAACL-HLT, 2021. 5, 17

[55] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Ce
Liu, Lu Yuan, and Jianfeng Gao. Uniﬁed contrastive learn-
ing in image-text-label space. In IEEE/CVF Conference on
Computer Vision and Pattern Recognition, CVPR 2022, New
Orleans, LA, USA, June 18-24, 2022, pages 19141–19151.
IEEE, 2022. 2

[56] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung,
Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Con-
trastive captioners are image-text foundation models. CoRR,
abs/2205.01917, 2022. 2

[57] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella,
Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang,
Boxin Li, Chunyuan Li, Ce Liu, Mengchen Liu, Zicheng Liu,
Yumao Lu, Yu Shi, Lijuan Wang, Jianfeng Wang, Bin Xiao,
Zhen Xiao, Jianwei Yang, Michael Zeng, Luowei Zhou, and
Pengchuan Zhang. Florence: A new foundation model for
computer vision. CoRR, abs/2111.11432, 2021. 2

[58] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lu-
cas Beyer. Scaling vision transformers. CVPR, 2022. 1, 4,
6, 7, 14

[59] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner,
Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer.
Lit: Zero-shot transfer with locked-image text tuning.
In
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, CVPR 2022, New Orleans, LA, USA, June 18-
24, 2022, pages 18102–18112. IEEE, 2022. 1, 2, 3, 4, 6, 7,

[60] Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D.
Manning, and Curtis P. Langlotz. Contrastive learning of
medical visual representations from paired images and text.
In Zachary C. Lipton, Rajesh Ranganath, Mark P. Sendak,
Michael W. Sjoding, and Serena Yeung, editors, Proceed-
ings of the Machine Learning for Healthcare Conference,


A. More results for SigLiT

In section 4.1, we use the same precomputed embed-
dings for the images using a ViT-g vision model from [59].
Only resize augmentation is applied, to a ﬁxed 288 × 288
resolution. We train a standard base size text tower, using
the ScalingViT-Adafactor optimizer [58] with β1 = 0.9 and
β2 = 0.95. We use 0.001 learning rate with a linear warmup
schedule for the ﬁrst 200 M examples seen, and then the
learning rate is decayed to zero with a cosine learning rate
schedule. Weight decay is set to 0.0001 for all the experi-
ments. The text is tokenized by a 32 k vocabulary sentence-
piece tokenizer [27] trained on the English C4 dataset [37],
and a maximum of 16 text tokens are kept. Table 8 shows
results with multiple train examples seen and batch sizes,
for both the sigmoid loss and the softmax loss baseline.

For training SigLiT in under a day with 4 chips (Sec-
tion 4.4), we used the LION optimizer with peak learning
rate 1 × 10−4 and weight decay 1 × 10−7. The learning rate
was warmed linearly to the peak in 6.5 k steps, then cosine
decayed to zero for the remaining 58.5 k steps.

B. More results for SigLIP

In Table 5, we present more results for SigLIP Base with
multiple train examples seen: 3 billion examples and 9 bil-
lion examples respectively.

Batch Size

3 B

9 B

sigmoid

softmax

sigmoid

softmax

1 k
2 k
4 k
8 k
16 k
32 k
98 k
307 k

51.5
57.3
62.1
65.3
68.6
-
69.9
69.5
-

47.7
53.2
59.3
63.8
66.6
-
69.9
69.7
-

-
-
-
68.4
70.6
72.3
73.4
73.0
71.6

-
-
-
66.6
69.4
71.7
72.9
73.2
72.6

Table 5: SigLIP zeor-shot accuracy (%) on the ImageNet
benchmark. Both the sigmoid loss and the softmax loss
baseline are presented. Experiments are performed on mul-
tiple train examples seen (3 B, 9 B) and train batch sizes
(from 512 to 307 k). When trained for 9 B examples, the
peak of the sigmoid loss comes earlier at 32 k than the peak
of the softmax loss at 98 k. Together with the memory ef-
ﬁcient advantage for the sigmoid loss, it allows one to train
the best language-image model with much fewer amount of
accelerators.

BS

8 k
16 k
32 k

Default

70.1
70.0
68.2

Best

70.1
70.0
69.0

Best LR

Best WD

0.001
0.001
0.0003

0.0001
0.0001
0.00003

Table 6: Default hyperparameters across different batch
sizes, perform either the best or close to the best hyperpa-
rameter from a sweep. Zero-shot accuracy on ImageNet
is reported. BS=batch size, LR=learning rate, WD=weight
decay.

C. Robustness of SigLIP results

Hyperparameters for different batch sizes. Sigmoid
loss doesn’t require tuning hyperparameters for different
batch sizes. For example, in both the SigLiP and SigLiT
setup, we only used default 0.001 learning rate and 0.0001
weight decay across a wide range of batch sizes (from 512
to 1024k). We further performed a sweep of 9 hyperparam-
eters across 3 batch sizes on the from-scratch SigLIP setup
for 3B seen examples: learning rate {0.0003, 0.001, 0.003}
× weight decay {0.00003, 0.0001, 0.0003} × batch size
{8 k, 16 k, 32 k}. We observed in Table 6 that the default
LR/WD is either the best or close to the best.

Standard deviation. We repeat SigLIP training ﬁve
times, using the recommended 32k batch size and 3B seen
examples. We report the average and std in Table 7. The std
of the ﬁve runs is very small for both sigmoid and softmax.

Alternative optimizers. We repeat the same experiment
with AdamW optimizer ﬁve times and got very similar re-
sults and std as reported in Table 7. We tested a linear learn-
ing rate scheduler instead of the default cosine learning rate
scheduler, it achieves 69.9% accuracy.

D. More results for mSigLIP

We present the mSigLIP Base crossmodal retrieval re-
sults on the Crossmodal-3600 dataset, across all the 36 lan-
gauges in Figure 8 and Table 9.

Loss

Softmax
Sigmoid
Sigmoid

Optimizer

Results (%)

ViT-Adafactor
ViT-Adafactor
AdamW

69.9 ± 0.1
70.1 ± 0.2
70.3 ± 0.1

Table 7: Mean and standard deviation of ﬁve repeated ex-
periments. Zero-shot accuracy on ImageNet is reported.


Batch Size

450 M

900 M

3 B

18 B

sigmoid

softmax

sigmoid

softmax

sigmoid

softmax

sigmoid

softmax

1 k
2 k
4 k
8 k
16 k
32 k
64 k
128 k
256 k
1024 k

72.5
75.5
77.1
79.2
80.8
81.2
81.9
81.6
80.5
72.8
-

69.5
73.6
76.3
78.3
79.7
81.2
81.4
81.6
80.0
72.2
-

75.0
77.2
79.3
80.8
82.0
82.7
83.1
83.0
83.1
82.1
-

72.8
76.0
78.1
79.8
81.0
82.1
82.7
82.8
83.2
81.7
-

77.2
79.6
81.3
82.4
83.1
83.8
84.2
84.3
84.2
84.3
-

74.6
77.9
80.1
81.2
82.6
83.5
84.0
84.1
84.4
84.2
-

-
-
82.2
83.0
83.6
84.2
84.6
84.7
84.7
84.7
84.7

-
-
81.2
82.0
83.1
84.1
84.4
84.4
84.6
84.6
-

Table 8: SigLiT zero-shot accuracy (%) on the ImageNet benchmark. Both the sigmoid loss and the softmax loss
baseline are presented. Extensive experiments are performed on multiple train examples seen (450 M, 900 M, 3 B, 18 B) and
train batch sizes (from 512 to 1 M).

Figure 8: Image-to-text and text-to-image zero-shot retrieval recall@1 results on all 36 languages of Crossmodal-3600.
Top: Image to text. Bottom: text to image. Colors are batch sizes. *32 k represents the scaled up results as described in
Section 4.6.

E. Label noise experiments

All models had an M/16 image tower and a M text tower.
They were trained from random initialisation for 3.6B ex-
amples seen, with a batch size of 16384. A cosine learning
rate schedule was used, with an initial linear warmup for
10% of steps up to a peak learning rate of 0.001.


arbncsdadeelenesfafifilfrhihrhuiditiwjakominlnoplptquzrorusvswtethtrukvizhavg0.00.10.20.30.40.50.60.70.816 k32 k64 k128 k240 k*32 karbncsdadeelenesfafifilfrhihrhuiditiwjakominlnoplptquzrorusvswtethtrukvizhavg0.00.10.20.30.40.50.616 k32 k64 k128 k240 k*32 k
Lang.

Image-to-text

Text-to-image

16 k

32 k

64 k

128 k

240 k

*32 k

16 k

32 k

64 k

128 k

240 k

*32 k

ar
bn
cs
da
de
el
en
es
fa
ﬁ
ﬁl
fr
hi
hr
hu
id
it
iw
ja
ko
mi
nl
no
pl
pt
quz
ro
ru
sv
sw
te
th
tr
uk
vi
zh

avg

52.4
11.4
54.1
62.7
70.3
36.9
50.1
64.7
57.0
54.9
23.2
65.7
19.9
52.7
57.0
64.8
65.9
48.4
46.4
50.8
0.4
59.6
61.4
62.2
63.1
6.8
52.1
62.2
62.3
14.8
1.2
36.1
53.1
51.4
59.6
44.1

51.3
10.8
53.7
62.4
71.4
35.8
50.5
64.9
57.8
54.1
22.8
66.9
18.8
53.0
57.1
67.1
66.4
47.9
45.9
49.5
0.4
60.4
62.4
62.0
63.6
6.4
51.4
63.6
63.5
14.4
1.2
35.8
54.5
51.5
59.8
45.7

51.5
10.4
53.7
62.0
71.2
35.1
50.2
67.2
56.1
53.8
22.9
67.0
19.9
53.0
56.3
66.6
67.1
47.7
42.9
49.4
0.6
58.9
62.0
62.0
64.9
6.4
51.0
63.1
63.5
14.3
1.2
35.6
53.7
51.2
59.5
44.1

47.2

47.4

47.1

51.5
10.3
52.8
60.4
71.1
34.5
49.9
65.3
55.3
51.7
21.4
66.1
19.5
49.9
54.8
65.4
65.2
46.1
43.7
50.2
0.6
58.3
60.9
61.1
64.3
6.6
50.6
62.7
63.1
14.2
1.7
35.6
52.9
49.9
58.5
41.9

46.3

51.1
9.9
51.8
59.3
70.2
33.8
50.7
65.6
54.6
51.7
21.2
66.5
17.4
49.6
53.0
64.7
66.1
45.2
30.2
46.8
0.4
57.9
59.9
60.5
63.2
6.7
49.3
63.1
61.2
13.8
1.1
28.3
51.2
49.2
58.8
36.1

45.0

59.7
30.1
58.9
68.4
79.7
47.4
52.5
66.3
66.2
59.1
29.2
71.2
32.2
62.6
62.9
73.7
72.3
62.2
55.1
61.4
0.3
63.6
65.3
67.1
65.4
6.8
61.0
68.4
67.7
17.4
8.4
39.0
62.0
61.2
68.4
53.9

54.1

37.6
5.5
41.8
47.0
54.7
22.4
46.5
54.8
39.6
37.7
12.8
55.9
9.1
38.2
41.4
48.5
55.5
31.8
31.0
34.4
0.2
48.9
45.3
48.8
52.4
2.7
37.2
50.1
47.9
7.8
0.4
21.6
37.3
34.5
41.4
30.7

37.4
6.2
41.6
47.0
54.8
22.8
46.2
55.0
40.2
37.1
12.9
57.1
8.5
37.1
40.2
49.4
56.4
31.8
31.3
34.7
0.2
49.5
46.2
47.4
52.3
2.6
35.6
49.9
48.2
7.2
0.3
23.1
37.4
33.2
41.9
32.5

37.1
4.9
41.5
45.6
55.4
22.0
46.5
55.5
38.4
36.4
12.4
55.5
7.9
36.4
40.2
49.5
55.8
31.9
29.2
33.2
0.2
48.9
45.0
48.7
52.3
2.7
34.3
49.7
47.6
7.1
0.3
22.2
37.8
33.8
41.9
32.0

34.8

34.9

34.4

36.3
5.1
39.9
43.0
54.3
21.3
46.6
54.5
38.4
34.0
12.2
54.4
8.1
35.2
38.6
47.8
54.8
30.1
28.9
33.1
0.2
48.4
43.5
46.8
51.9
2.7
34.5
48.6
46.2
6.9
0.5
21.6
37.0
32.5
40.6
30.6

33.6

36.0
4.4
39.4
43.5
54.7
20.8
46.6
55.2
38.3
34.5
11.3
54.3
7.3
34.3
38.2
47.3
54.1
30.1
18.5
31.5
0.2
47.9
43.7
46.7
52.4
2.8
32.5
49.3
46.2
6.3
0.3
16.8
36.1
32.4
40.3
23.7

32.7

44.9
20.0
47.0
52.9
65.3
32.2
47.6
57.0
50.0
44.0
20.4
61.8
17.3
47.2
51.2
60.5
62.3
48.0
42.3
45.9
0.3
53.6
50.0
56.7
57.3
2.9
49.3
59.9
52.0
10.7
4.3
24.6
48.1
48.3
52.3
46.8

42.6

Table 9: Image-to-text (text retrieval) and text-to-image (image retrieval) zero-shot recall@1 results on all 36 languages
of Crossmodal-3600, with mSigLIP models trained at different batch sizes for 30 B total examples seen. *32 k represents
the scaled up results as described in Section 4.6.


F. Model Card

We provide a description of our models following [32].

• Model Architecture: The model is trained using the
contrastive pre-training technique with sigmoid loss
as described in this paper. This contrastive model
contains two encoders, i.e.
vision transformer en-
coder [17] and language transformer encoder [47]. The
vision and language encoders always have the same
size, one of ViT-B, ViT-L and SoViT-400M [1].

• Inputs: The vision encoder takes an image (224 ×
224×3, 256×256×3, 384×384×3, 512×512×3) as
input. The text encoder takes a tokenized text [38, 54]
cropped to the ﬁrst 64 tokens as input.

• Outputs: The vision and text encoders both output a d
dimensional feature vector, where d is 768, 1024 and
1152 for ViT-B, ViT-L and SoViT-400M, respectively.

• Intended Use: The models are designed for multi-
modal research purposes. The models can be used
for zero-shot image classiﬁcation and zero-shot image-
text retrieval by comparing both feature vectors. We
provide both en-only and i18n-trained models to en-
courage research on the impact of this choice.

• Training Data: The contrastive model is pre-trained
from-scratch using the WebLI [13] dataset. SigLIP
models are pre-trained on a WebLI subset ﬁltered to
contain mostly English. mSigLIP models are pre-
trained on the WebLI dataset without language ﬁlters.

• Evaluation Data: Zero-shot classiﬁcation is per-
formed on ImageNet [14], ImageNet v2 [39], Ima-
geNet Real [3], and ObjectNet [2]. Zero-shot re-
trieval is performed on COCO [11] and the multilin-
gual XM3600 dataset [44].

• Hardware & Software: The models are developed
in the big vision codebase [5, 4] and trained on
Google Cloud TPUs.
