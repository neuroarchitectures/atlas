# Paper (Chen et al. 2021)

> Source: `https://arxiv.org/abs/2110.13900`

---

1
WavLM: Large-Scale Self-Supervised Pre-Training
for Full Stack Speech Processing
Sanyuan Chen*, Chengyi Wang*, Zhengyang Chen*, Yu Wu*, Shujie Liu, Zhuo Chen,
Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren,
Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, Furu Wei
Abstract—Self-supervised learning (SSL) achieves great suc-
cess in speech recognition, while limited exploration has been
attempted for other speech processing tasks. As speech signal
contains multi-faceted information including speaker identity,
paralinguistics, spoken content, etc., learning universal repre-
sentations for all speech tasks is challenging. To tackle the
problem, we propose a new pre-trained model, WavLM, to
solve full-stack downstream speech tasks. WavLM jointly learns
masked speech prediction and denoising in pre-training. By this
means, WavLM does not only keep the speech content modeling
capability by the masked speech prediction, but also improves the
potential to non-ASR tasks by the speech denoising. In addition,
WavLM employs gated relative position bias for the Transformer
structure to better capture the sequence ordering of input speech.
We also scale up the training dataset from 60k hours to 94k
hours. WavLM Large achieves state-of-the-art performance on
the SUPERB benchmark, and brings signiﬁcant improvements
for various speech processing tasks on their representative
benchmarks. The code and pre-trained models are available at
https://aka.ms/wavlm.
Index Terms—Self-Supervised Learning, Speech Pre-Training
I. INTRODUCTION
Over the past few years, self-supervised learning (SSL)
has achieved great success in the ﬁelds of natural language
processing (NLP) [1]–[3]. It leverages large amounts of text
data to learn universal text representations, which can beneﬁt
almost all NLP downstream tasks by ﬁne-tuning. Recently,
SSL has also shown prominent results for speech processing,
especially on phoneme classiﬁcation [4] and automatic speech
recognition (ASR) [5]–[7]. However, in other speech tasks, it
is still the standard practice to train models from scratch with
task-speciﬁc datasets.
Building a general pre-trained model for full stack speech
processing tasks is essential to the further development of
speech processing, because many tasks are short of supervised
data, especially for non-ASR tasks. A model pre-trained on
large-scale unlabeled data is able to boost the performance
of these tasks, reduce data labeling efforts, and lower entry
barriers for individual tasks. Furthermore, it is infeasible to
build different pre-trained models for different downstream
tasks, as the pre-training stage requires huge computational
resources. In the past, it has been infeasible to build such a
general model, as different tasks focus on different aspects
of speech signals. For instance, speaker veriﬁcation requires
* Equal contribution. The work was done at Microsoft during the internship
of the ﬁrst three authors.
Correspondence to Yu Wu: yuwu1@microsoft.com
the network to learn the speaker characteristic regardless of
the spoken content, while speech recognition demands the
network to discard speaker characteristics and focus only on
the content information. Meanwhile, unlike veriﬁcation and
recognition tasks, speaker diarization and speech separation
involve multiple speakers, which creates additional obstacles
to learning general speech representations. Recent advances
fueled by large-scale pre-trained models have changed the
situation. [8] proves the potential of pre-trained models on full-
stack speech tasks by using the weighted sum of embeddings
from different layers.1 They ﬁnd different layers containing
information useful for different tasks. For instance, the hidden
states of the top layers are useful for ASR, while the bottom
layers are more effective for speaker veriﬁcation.
While exciting as a proof of concept, there are still some
drawbacks in existing pre-trained models: 1) Current pre-trained
models are unsatisfactory for multi-speaker tasks, such as
speaker diarization and speech separation. Our experiments
show that speech separation models trained on top of HuBERT
[6], a top-performing speech pre-trained model, achieve only
marginal improvement compared with the models trained from
scratch. This is mainly because the pre-training methods do
not sufﬁciently enforce speaker discrimination, and the training
data contain only single-speaker audios. 2) Speech pre-training
crucially relies on high quality and large quantities of unlabeled
audios. The existing system utilizes Libri-Light [9] as the
main source, but the massive audiobook data mismatches
the data in a real scenario and using it exclusively hurts the
model performance when the acoustic characteristics of the
downstream tasks are different from those of the audiobook
[10]–[13]. [14] trains wav2vec 2.0 [5] on larger and more
diverse datasets, but there are still over 90% audio data derived
from audiobook. To eliminate the audiobook data bias, we try
to gather data from different sources as much as possible in
our experiments.
In this paper, we present WavLM, which learns universal
speech representations from massive unlabeled speech data and
adapts effectively across various speech processing tasks. We
propose a masked speech denoising and prediction framework
for WavLM, where some inputs are simulated noisy/overlapped
speech with masks and the target is to predict the pseudo-label
of the original speech on the masked region like HuBERT.
The framework combines the masked speech prediction and
1The paper does not explicitly mention it, but their presentation highlights
the contribution of weighted sum hidden states. Details can be found from
28:00 to 31:00 of https://www.youtube.com/watch?v=Fw2ujGzmfNA
arXiv:2110.13900v5  [cs.CL]  17 Jun 2022

2
denoising in pre-training. Therefore, the WavLM model learns
not only the ASR information by the masked speech prediction,
but also the knowledge of non-ASR tasks by the speech
denoising modeling. For instance, the process of pseudo-label
prediction on overlapped speech implicitly improves the model
capability on diariazation and separation tasks. The speaker
identity information and speech enhancement capability are
modeled by the pseudo-label prediction on simulated noisy
speech.
In addition, we optimize the model structure and training data
of HuBERT and wav2vec 2.0. We add gated relative position
bias (grep) [15] to the Transformer structure as the backbone,
which improves model performance for ASR and keeps almost
the same parameter number and training speed. Compared with
the convolutional relative position embedding used in wav2vec
2.0 and HuBERT, the gates allow the relative position bias to
be adjusted adaptively by conditioning on the current speech
content. To further improve the model robustness and alleviate
the data mismatch, we scale up unlabeled pre-training data
to 94k hours of public audios. The dataset consists of 60k
hours of Libri-Light, 10k hours of GigaSpeech [16], and 24k
hours of VoxPopuli [17]. The new dataset consists of training
instances from different scenarios, such as podcasts, YouTube,
and European Parliament (EP) event recordings.
We evaluate our models on nineteen subtasks, ﬁfteen of
which are from SUPERB, and the other four are classic speech
tasks on their representative testsets.
• WavLM achieves state-of-the-art (SOTA) performance on
SUPERB [8], [18]. WavLM Large outperforms HuBERT
Large on 14 subtasks, and achieves an absolute 2.4 point
improvement in the overall evaluation. Even WavLM
Base+, a 3 times smaller model, is better than HuBERT
Large owing to our three modiﬁcations.
• Speaker veriﬁcation is a task to verify the speaker’s
identity from the voice characteristics. We select this task
to evaluate the model’s capability of extracting speaker-
related features. WavLM Large exceeds the well-known
SOTA system, ECAPA-TDNN [19], by a large margin
and achieves 0.383%, 0.480% and 0.986% EER (Equal
Error Rate) on the three ofﬁcial trial lists of VoxCeleb1
[20].
• Speech separation is a classic multi-speaker task, which
is the key to solving the cocktail party problem. The
task can evaluate the model’s capability of extracting
multiple speech signals from a mixture of sounds. WavLM
achieves SOTA performance on the speech separation
LibriCSS benchmark [21], and signiﬁcantly outperforms
the previous Conformer model [22] by a 27.7% relative
word error rate (WER) reduction.
• Speaker diarization is a task to recognize “who spoke
when” from an input audio stream [23]. WavLM achieves
SOTA performance on the CALLHOME speaker diariza-
tion benchmark. Compared to the EEND-EDA clustering
method [24], our model achieves a 12.6% diarization error
rate reduction.
• Speech recognition requires the model to learn content
information, which is the main focus of the previous SSL
work. We evaluate our model in the LibriSpeech 960h
setting. WavLM shows comparable performance to the
wav2vec 2.0 and HuBERT, which achieves 1.8% and 3.2%
WER on the test-clean and test-other sets, respectively.
The contribution of the paper can be summarized as follows: 1)
WavLM sheds light on a general pre-trained model for full stack
speech processing tasks, in contrast to the previous SSL works
focusing on a group of similar tasks. 2) We propose simple but
effective modiﬁcations to the existing pre-trained models, which
show general and consistent improvements across downstream
tasks. 3) We scale-up self-supervised speech pre-training with
more unlabeled data and longer training steps. 4) We achieve
SOTA results on the SUPERB benchmark, and signiﬁcantly
boost the performance for various speech processing tasks on
their representative benchmarks, including speech separation,
speaker veriﬁcation, and speaker diarization. The models and
code are released 2 to facilitate future research.
II. RELATED WORK
SSL methods can be categorized into generative learning,
discriminative learning, and multitask learning, based on the
training objective. The research line of generative learning can
be traced back to the auto-encoding model, which reconstructs
the whole speech from latent variables, either continuous [25]–
[27] or discrete [28]. Recent works propose to predict future
frames from the history with an autoregressive model [29]–[32],
or recover the masked frames from the corrupted speech with
a non-autoregressive model [33]–[38]. Apart from generative
learning, discriminative learning has also gathered interests
recently. The well-known examples include CPC [4], wav2vec
[39], vq-wav2vec [40], wav2vec 2.0 [5], DiscreteBERT [41],
HuBERT [6] and w2v-BERT [42]. CPC and the wav2vec
series models use the contrastive InfoNCE loss to discriminate
the positive samples from negative samples. Motivated by
the masked language model loss in NLP, DiscreteBERT and
HuBERT predict discrete targets of masked regions. w2v-
BERT further combines the contrastive loss and the masked
prediction loss in an end-to-end fashion. Multi-task learning is
adopted in PASE [43] and PASE+ [44]. They employ lots of
pre-training objectives such as waveform generation, prosody
regression, and contrastive objectives. UniSpeech [7] and JUST
[45] combine SSL and supervised learning for ASR, and show
impressive results on multi-lingual test sets.
Unlike SSL in computer vision (CV) and NLP ﬁelds, where
one pre-trained model is adapted to various downstream tasks,
most speech SSL methods focus on phoneme classiﬁcation
and ASR. Recently, [8] proposed the SUPERB benchmark
to evaluate SSL models across different tasks. According to
the results, HuBERT enjoys the best generalization ability in
the overall evaluation. To better learn speaker characteristics,
[46] proposed UniSpeech-SAT, which extends the HuBERT
framework with speaker-aware pre-training. It signiﬁcantly
outperforms other pre-trained models on the speaker-related
tasks with a slight degradation on the ASR.
Compared with existing works, our model is the ﬁrst to
explore SSL for full stack tasks instead of focusing on ASR
or other speciﬁc tasks. It should be noted that a concurrent
2https://aka.ms/wavlm

3
x1
[M]
[M]
[M]
x5
x6
Z1
Z2
Z3
Z4
z5
z6
Utterance 
Mixing  audio
Transformer Encoder with 
Gated Relative Position Bias
Mask Prediction Loss
Target labels
CNN Encoders
Fig. 1. Model Architecture.
work BigSSL [47] also mentions large SSL model could
handle various speech tasks. The difference is that our work
demonstrates that the full stack tasks can be handled by
the careful pre-training and ﬁne-tuning strategy design, even
without scaling up the model size to 8 billion parameters.
III. BACKGROUND: HUBERT
HuBERT is an SSL method that beneﬁts from an ofﬂine
clustering step to provide target labels for a BERT-like
prediction loss [1]. The backbone is a Transformer encoder [48]
with L blocks. During pre-training, the Transformer consumes
masked acoustic features u and outputs hidden states hL. The
network is optimized to predict the discrete target sequence
z, where each zt ∈[C] is a C-class categorical variable. The
distribution over codewords is parameterized with
p(c|hL
t ) =
exp(sim(hL
t WP , ec)/τ)
PC
c′=1 exp(sim(hL
t WP , ec′)/τ)
(1)
where WP is a projection matrix, hL
t is the output hidden
state for step t, ec is the embedding for codeword c, sim(a, b)
computes the cosine similarity and τ = 0.1 scales the logit.
HuBERT proposes a masked speech prediction task, where
the prediction loss is only applied over the masked regions,
forcing the model to learn a combined acoustic and language
model over the continuous inputs.
HuBERT adopts an iterative re-clustering and re-training
process: For the ﬁrst iteration, the targets are assigned by
clustering the MFCC features of the training data; For the
second iteration, a new generation of training targets are created
by clustering the latent representations generated by the ﬁrst
iteration trained model.
IV. WAVLM
We propose a masked speech denoising and prediction
framework, where some inputs are simulated noisy/overlapped
with masks and the target is to predict pseudo-labels of the
original speech on the masked region. Unlike existing masked
speech modeling (HuBERT), which just focuses on the ASR
task, the masked speech denoising allows us to extend pre-
trained speech models to non-ASR tasks, since it implicitly
models information we need in the speaker identiﬁcation,
separation, and diarization tasks. We further optimize the
Transformer backbone and extend pre-training data to 94k
public English data.
A. Model Structure
Our model architecture uses the Transformer model as the
backbone. As shown in Figure 1, it contains a convolutional
feature encoder and a Transformer encoder. The convolutional
encoder is composed of seven blocks of temporal convolution
followed by layer normalization and a GELU activation layer.
The temporal convolutions have 512 channels with strides
(5,2,2,2,2,2,2) and kernel widths (10,3,3,3,3,2,2), resulting
in each output representing about 25ms of audio strode by
20ms. The convolutional output representation x is masked as
the Transformer input. The Transformer is equipped with a
convolution-based relative position embedding layer with 128
kernel size and 16 groups at the bottom.
To improve the model, we employ gated relative position
bias [15] which is encoded based on the offset between the
“key” and “query” in the Transformer self-attention mechanism.
Let {hi}T
i=1 denote the input hidden states for the self-attention
module, each hi is linearly projected to a triple of query, key
and value (qi, ki, vi) as:
qi, ki, vi = hiWQ, hiWK, hiWV
(2)
The self-attention outputs {˜hi}T
i=1 are computed via:
aij ∝exp{qi · kj
√dk
+ ri−j}
(3)
˜hi =
T
X
j=1
aijvj
(4)
where ri−j is the gated relative position bias added to the
attention logits. It is computed by:
g(update)
i
, g(reset)
i
= σ(qi · u), σ(qi · w)
˜ri−j = wg(reset)
i
di−j
ri−j = di−j + g(update)
i
di−j + (1 −g(update)
i
)˜ri−j
where di−j is a learnable scalar relative position bias, the
vectors u, w ∈Rdk are learnable parameters, σ is a sigmoid
function, and w is a learnable value.
In our work, di−j is a bucket relative position embedding [3]
and the embedding parameters are shared across all layers. We
use n = 320 embeddings and each corresponds to a range of
possible (i −j) offsets. The range increased logarithmically
up to a maximum offset of m = 800, beyond which we assign
all relative offsets to the same embedding, i.e.,
d|i−j| =





|i −j|,
|i −j| < n
4
⌊n
4 ( log(|i−j|)−log( n
4 )
log(m)−log( n
4 )
+ 1)⌋,
n
4 ≤|i −j| < m
n
2 −1,
|i −j| ≥m
(5)

4
Algorithm 1 Noisy/Overlapped Speech Simulation
1: given a batch of speech utterances U = {ui}B
i=1 with batch size B and length L, mixing probability p, a set of DNS noises N = {ni}M
i=1 with size
M, mixing noise probability pn
2: Choose S utterances US ⊂U by Bernoulli sampling with probability p
3: for each primary utterance upri ∈US do
4:
Sample a random value v from the continuous uniform distribution U(0, 1)
5:
if v > pn then
6:
Sample a secondary utterance usec from discrete uniform distribution with probability P(usec = x) = 1
B , x ∈U
7:
Sample the mixing energy ratio r from the continuous uniform distribution U(−5, 5)
8:
else
9:
Sample a noise usec from discrete uniform distribution with probability P(usec = x) =
1
M , x ∈N
10:
Sample the mixing energy ratio r from the continuous uniform distribution U(−5, 20)
11:
end if
12:
Sample the mix length l from discrete uniform distribution with probability P(l = x) = 2
L , x ∈{1, · · · , L
2 }
13:
Sample a start position spri of upri from discrete uniform distribution with probability P(spri = x) =
1
L−l, x ∈{1, · · · , L −l}
14:
Sample a start position ssec of usec from discrete uniform distribution with probability P(ssec = x) =
1
L−l, x ∈{1, · · · , L −l}
15:
Calculate the energy of the primary utterance Epri ←
P upri·upri
L
16:
Calculate the energy of the secondary utterance Esec ←
P usec·usec
L
17:
Calculate the mixing scale scl ←
r
Epri
10
r
10 Esec
18:
upri[spri : spri + l] ←upri[spri : spri + l] + scl · usec[ssec : ssec + l]
19: end for
20: return U
di−j = d|i−j| + n
2 · 1{i−j>0}
(6)
Compared with the convolutional relative position embedding
in wav2vec 2.0 and HuBERT, the gates take the content into
consideration, and adaptively adjust the relative position bias
by conditioning on the current speech content. Intuitively, the
same distance offset between two frames tends to play different
roles if one frame is the silence while the other belongs to a
speech segment.
B. Masked Speech Denoising and Prediction
We propose a masked speech denoising and prediction
framework to improve model robustness for complex acoustic
environments and the preservation of speaker identity. Speciﬁ-
cally, we manually simulated noisy/overlapped speech as inputs,
and predict the pseudo-labels of original speech on the masked
region.
We simulate the noisy speech with multiple speakers and
various background noise for self-supervised pre-training. We
randomly select some utterances from each training batch and
mix them with a randomly selected noise audio or secondary
utterance at a random region. The noise audio and secondary
utterance are randomly selected from the same batch, randomly
cropped, and scaled by a random source energy ratio. We ensure
that the overlap region is less than 50% and take the speaker
from the ﬁrst utterance as the main speaker. With the masked
speech denoising and prediction task, the model is trained to
identify the main speaker from the noisy/overlapped speech
and predict the content information corresponding to the main
speaker with the mask prediction loss.
1) Noisy/Overlapped Speech Simulation: The details of
our noisy/overlapped speech simulation method are shown in
Algorithm 1. Given a batch of speech utterances U = {ui}B
i=1
with batch size B and length L and a set of DNS (Deep Noise
Suppression) noises [49] N = {ni}M
i=1 with size M (line 1),
we ﬁrst randomly choose S utterances to mix US = {ui}S
i=1
from the batch by Bernoulli sampling with probability p (line
2). Then, for each utterance upri ∈US (line 3), we sample
a random value v the continuous uniform distribution U(0, 1)
to decide whether to mix noise or a secondary utterance (line
4). If the random value v is greater than the mixing noise
probability pn (line 5), we sample a secondary utterance usec
from a discrete uniform distribution over the batch U (line
6), and randomly sample the mixing energy ratio r from the
uniform distribution U(−5, 5) (line 7). Otherwise, we sample
a noise usec from a discrete uniform distribution over the set of
DNS noises N (line 9), and randomly sample the mixing energy
ratio r from the uniform distribution U(−5, 20) (line 10). The
sample range of the mixing energy ratio follows the typical
training utterance simulation process of speech separation task
[50]. Then, we randomly select the mixing regions for both
the utterances from the uniform distributions (line 12-14). The
mixing length l is uniformly sampled from {1, · · · , L
2 } (line
12), and the start positions spri and ssec for utterance upri
and usec are both uniformly sampled from {1, · · · , L −l}
(line 13 and 14). Note that as the mixing portion in each
utterance is constrained to be less than 50%, the primary
utterance would always be longer than the secondary utterance,
avoiding the problem of the indistinguishable main speaker in
the mixed speech signals. Next, given the mixing regions of the
primary utterance upri[spri : spri+l] and the secondary utterance
usec[ssec : ssec + l], we calculate the corresponding mixing
scale of the secondary utterance scl with the energy of the
primary utterance Epri, the secondary utterance Esec (line 15-
17). Finally, we mix the selected region of the primary utterance
upri[spri : spri+l] with the secondary utterance usec[ssec : ssec+l]
scaled by the mixing scale scl (line 18).
2) Mask Prediction Loss: Following HuBERT, we use the
mask prediction loss to optimize our network. Suppose that we
have an utterance u and its simulated version u′, we always
generate pseudo-labels z by feeding u to the last iteration
network. We follow HuBERT using the k-means clustering
center on MFCC or latent representations as the pseudo-labels.
The details will be introduced in Section V-A. Then, we obtain

5
the hidden state hL
t by feeding u′ to the current network, and
optimize the objective function:
L = −
X
l∈K
X
t∈M
logp(zt|hL
t )
(7)
where M denotes the set of masked indices in time domain
and hL
t is the L-layer Transformer output for step t. Compared
to previous methods, the framework is more beneﬁcial to
various non-ASR tasks, since it implicitly models the non-ASR
information in pre-training.
C. Pre-Training Data
We leverage large-scale unsupervised data from diverse
domains to improve the robustness of our model. Previous
works use LibriSpeech [51] or LibriLight [9] datasets for pre-
training, which limits the generalization capability of the pre-
trained model since the input data are all extracted from the
audiobook. The background acoustics of the speech obtained
from the audiobook is different from what is observed in
other conditions , since the real captured sounds are usually
accompanied by various types of noise.
Motivated by this, we extend the training data with two
datasets: (1) 10k hours of the GigaSpeech data [16]. It is
collected from audiobooks, podcasts and YouTube, covering
both read and spontaneous speaking styles, and a variety of
topics, such as arts, science, sports, etc. It should be noted that
the total data size of GigaSpeech is 40k, but 30k of them are not
well processed. For example, there is a large segment of silence
at the beginning or at the end of some utterances in the 30k data.
More seriously, some utterances just contain background noise
without any speech. Thus, we just use the subset of 10k hours
of GigaSpeech data, which is well processed and validated with
a segmentation pipeline proposed in [16]. (2) VoxPopuli data
[17]). It is a large-scale multi-lingual unlabeled audio dataset
consisting of over 400k hours of audio in 23 languages, which
is collected from 2009-2020 European Parliament (EP) event
recordings including plenary sessions, committee meetings,
and other events. Since our focus is English-only audio, we
use 24k hours of English data in VoxPopuli for pre-training.
In total, we collect 94k hours of data, including LibriLight,
VoxPopuli, and GigaSpeech. We believe the enriched dataset
can improve the model robustness as it contains diverse audio
backgrounds, more speakers, and different contents. We call
the dataset Mix 94k hr to make the description simple.
D. Stabilization of Training
Currently, it is a common practice to use 16-bit ﬂoat
precision (fp16) or mixed precision to pre-train large models
for faster computation and less GPU memory consumption.
Unfortunately, the training is unstable for large models due
to the overﬂow issue (characterized by NaN losses) [52]. A
major reason is the attention score qi·kj
√
d
is larger than the
upper bound value of the fp16, resulting in the overﬂow issue
in training.
We apply a simple trick to alleviate the overﬂow issue [52].
Given that softmax is invariant under translation by the same
value in each coordinate i.e. softmax(x + α)k = softmax(x)k,
where α denotes a constant number, the equation (3) can be
implemented as
αi,j ∝exp{qi · kj
√
d
+ ri−j}
= exp{( qi
c
√
d
· kj −max
j′≤T( qi
c
√
d
· kj′)) × c + ri−j}.
(8)
where c is a scale hyperparameter and set to 32 in our
work. In this way, the overﬂow issue could be solved, since
maxj′≤T ( qi
c
√
d · kj′) could guarantee the max value is smaller
than 216.
V. EXPERIMENT
A. Pre-Training Setup
The WavLM Base and WavLM Base+ have 12 Transformer
encoder layers, 768-dimensional hidden states, and 8 attention
heads, resulting in 94.70M parameters. The WavLM Large has
24 Transformer encoder layers, 1024-dimensional hidden states,
and 12 attention heads, resulting in 316.62M parameters. The
relative position embeddings are shared across all layers, which
avoids signiﬁcantly increasing the number of parameters. We
pre-train the WavLM Base model for 400k steps on LibriSpeech
960 hours audio [51] using the label generated by clustering
the 6-th transformer layer output of the 1st-iteration HuBERT
Base model. The WavLM Base+ and WavLM Large models
are pre-trained for 1M and 700k steps on 94k large-scale
diverse data using the labels generated by clustering the 9th
transformer layer output of the released 2nd-iteration HuBERT
Base model 3. Even if the 1st iteration WavLM is better
than HuBERT, we ﬁnd using the 1st iteration pseudo-label
of HuBERT results in a slightly better 2nd iteration WavLM
(0.2 WER reduction on LibriSpeech dev-other). The masked
speech denoising modeling is considered in 20% utterances,
where the mixing noise probability pn is set to 0 for WavLM
Base model, and 10% for WavLM Base+ and Large model.
For other training conﬁgurations, the same hyperparameters
are used following [6], which are shown in Appendix A.
B. Universal Representation Evaluation
1) Setup: We ﬁrst evaluate our models on SUPERB, which
is designed to provide a standard and comprehensive testbed
for pre-trained models on various speech tasks. It covers
ﬁfteen tasks, including Speaker Identiﬁcation (SID), Auto-
matic Speaker Veriﬁcation (ASV), Speaker Diarization (SD),
Phoneme Recognition (PR), Automatic Speech Recognition
(ASR), Out-Of-Domain Automatic Speech Recognition (OOD-
ASR), Keyword Spotting (KS), Query by Example Spoken
Term Detection (QbE), Speech Translation (ST), Intent Clas-
siﬁcation (IC), Slot Filling (SF), Emotion Recognition (ER),
Speech Enhancement (SE), Speech Separation (SS) and Voice
Conversion (VC). These tasks can be grouped into ﬁve aspects
of speech: content, speaker, semantics, paralinguistics, and
generation.
We follow the settings created by SUPERB. 1) We use the
same downstream models as the SUPERB implementations
3https://github.com/pytorch/fairseq/tree/main/examples/hubert

6
TABLE I
UNIVERSAL SPEECH REPRESENTATION EVALUATION ON SUPERB BENCHMARK. PARAL DENOTE PARALINGUISTICS ASPECT OF SPEECH.
Method
#Params
Corpus
Speaker
Content
Semantics
ParaL
Generation
Overall
SID
ASV
SD
PR
ASR
OOD-ASR
KS
QbE
ST
IC
SF
ER
SE
SS
VC
Acc ↑
EER ↓
DER ↓
PER ↓
WER ↓
WER ↓
Acc ↑
MTWV ↑
BLEU ↑
Acc ↑
F1 ↑
CER ↓
Acc ↑
PESQ ↑
STOI ↑
SI-SDRi ↑
MCD ↓
WER ↓
ASV ↑
Score ↑
FBANK
0
-
8.5E-4
9.56
10.05
82.01
23.18
63.58
8.63
0.0058
2.32
9.10
69.64
52.94
35.39
2.55
93.6
9.23
8.47
38.3
77.25
43.2
PASE+ [44]
7.83M
LS 50 hr
37.99
11.61
8.68
58.87
25.11
61.56
82.54
0.0072
3.16
29.82
62.14
60.17
57.86
2.56
93.9
9.87
8.66
30.6
63.20
51.5
APC [30]
4.11M
LS 360 hr
60.42
8.56
10.53
41.98
21.28
63.12
91.01
0.0310
5.95
74.69
70.46
50.89
59.33
2.56
93.4
8.92
8.05
27.2
87.25
59.2
VQ-APC [29]
4.63M
LS 360 hr
60.15
8.72
10.45
41.08
21.20
63.56
91.11
0.0251
4.23
74.48
68.53
52.91
59.66
2.56
93.4
8.44
7.84
22.4
94.25
59.5
NPC [33]
19.38M
LS 360 hr
55.92
9.40
9.34
43.81
20.20
61.66
88.96
0.0246
4.32
69.44
72.79
48.44
59.08
2.52
93.1
8.04
7.86
30.4
94.75
59.0
Mockingjay [35]
85.12M
LS 360 hr
32.29
11.66
10.54
70.19
22.82
65.27
83.67
6.6E-04
4.45
34.33
61.59
58.89
50.28
2.53
93.4
9.29
8.29
35.1
79.75
51.0
TERA [34]
21.33M
LS 960 hr
57.57
15.89
9.96
49.17
18.17
58.49
89.48
0.0013
5.66
58.42
67.50
54.17
56.27
2.54
93.6
10.19
8.21
25.1
83.75
57.2
DeCoAR 2.0 [37]
89.84M
LS 960 hr
74.42
7.16
6.59
14.93
13.02
53.62
94.48
0.0406
9.94
90.80
83.28
34.73
62.47
2.47
93.2
8.54
7.83
17.1
90.75
66.3
modiﬁed CPC [53]
1.84M
LL 60k hr
39.63
12.86
10.38
42.54
20.18
62.54
91.88
0.0326
4.82
64.09
71.19
49.91
60.96
2.57
93.7
10.40
8.41
26.2
71.00
56.9
wav2vec [39]
32.54M
LS 960 hr
56.56
7.99
9.9
31.58
15.86
55.86
95.59
0.0485
6.61
84.92
76.37
43.71
59.79
2.53
93.8
9.30
7.45
10.1
98.25
63.5
vq-wav2vec [40]
34.15M
LS 960 hr
38.80
10.38
9.93
33.48
17.71
60.66
93.38
0.0410
5.66
85.68
77.68
41.54
58.24
2.48
93.6
8.16
7.08
13.4
100.00
61.8
wav2vec 2.0 Base [5]
95.04M
LS 960 hr
75.18
6.02
6.08
5.74
6.43
46.95
96.23
0.0233
14.81
92.35
88.30
24.77
63.43
2.55
93.9
9.77
7.50
10.5
98.00
69.6
HuBERT Base [6]
94.68M
LS 960 hr
81.42
5.11
5.88
5.41
6.42
46.69
96.30
0.0736
15.53
98.34
88.53
25.20
64.92
2.58
93.9
9.36
7.47
8.0
98.50
70.9
WavLM Base
94.70M
LS 960 hr
84.51
4.69
4.55
4.84
6.21
42.81
96.79
0.0870
20.74
98.63
89.38
22.86
65.94
2.58
94.0
10.37
7.42
8
98.00
72.0
- w/o denoising task
94.70M
LS 960 hr
84.39
4.91
6.03
4.85
6.08
43.61
96.79
0.0799
21.03
98.42
88.69
23.43
65.55
2.56
93.9
9.91
7.43
7.5
97.75
71.7
- w/o structure modiﬁcation
94.68M
LS 960 hr
84.74
4.61
4.72
5.22
6.80
42.88
96.79
0.0956
20.03
98.31
88.56
24.00
65.60
2.58
94.0
10.29
7.45
8.4
99.00
71.9
WavLM Base+
94.70M
Mix 94k hr
89.42
4.07
3.50
3.92
5.59
38.32
97.37
0.0988
24.25
99.00
90.58
21.20
68.65
2.63
94.3
10.85
7.40
8.1
99.00
73.4
wav2vec 2.0 Large [5]
317.38M
LL 60k hr
86.14
5.65
5.62
4.75
3.75
44.69
96.66
0.0489
12.48
95.28
87.11
27.31
65.64
2.52
94.0
10.02
7.63
15.8
97.25
70.4
HuBERT Large [6]
316.61M
LL 60k hr
90.33
5.98
5.75
3.53
3.62
44.08
95.29
0.0353
20.01
98.76
89.81
21.76
67.62
2.64
94.2
10.45
7.22
9.0
99.25
72.2
WavLM Large
316.62M
Mix 94k hr
95.49
3.77
3.24
3.06
3.44
32.27
97.86
0.0886
26.57
99.31
92.21
18.36
70.62
2.70
94.5
11.19
7.30
9.0
99.00
74.6
for each downstream task. 2) Pre-trained models are frozen to
limit the space of the ﬁne-tuning hyperparameter search. 3)
The downstream models consume the weighted sum results
of the hidden states extracted from each layer of the pre-
trained model. The detailed hyperparameters for ﬁne-tuning
our WavLM models on SUPERB downstream tasks are shown
in Appendix A. The overall score is computed by ourselves:
we multiply the QbE score with 100, replace each error rate
score with (1 - error rate), and average the scores of all tasks.
2) Evaluation result: Table I shows the evaluation results.
We compare our WavLM with several SSL models which are
evaluated by [8], [18]. In general, WavLM is very powerful in
universal representation learning. Our WavLM Base+ model
has outperformed HuBERT large and wav2vec 2.0 large in the
overall score.
WavLM Base: From Table I, we can observe that WavLM
Base performs better than wav2vec 2.0 Base and HuBERT
Base on all downstream tasks. It is a fair comparison as the
three models use the same amount of pre-training data and the
number of parameters. The results indicate the effectiveness
of our structure and the masked speech denoising modeling
in universal speech representation learning. We ﬁnd the most
impressive result is speaker diarization, where the WavLM
Base outperforms HuBERT Base by 22.6% relatively. Our
explanation is that the additional overlapped speech forces the
model to deal with multi-speaker signals during pre-training.
To verify this assumption, we conduct an ablation study to
remove simulated noisy/overlapped speech in pre-training. The
performance of the “w/o denoising task” drops signiﬁcantly for
the speaker diarization task. We also evaluate the contribution
of the structure change. We can see that, in the “w/o structure
modiﬁcation” setting, performance degradation can be wit-
nessed especially for PR and ASR tasks. It indicates that the
gated relative position bias contributes to the performance
improvement of the content-related tasks. Meanwhile, we
can observe that WavLM performs very well on semantic,
paralinguistics, and generation tasks as well, demonstrating our
model is general for the full stack speech processing tasks.
WavLM Base+: WavLM Base+ shows the contribution
from larger and more diverse pre-training data. It consistently
improves WavLM Base and even outperforms the wav2vec 2.0
Large and HuBERT Large in the overall score. This indicates
that the 960h data are insufﬁcient to fulﬁll the capacity of
the Base model. The combined dataset especially boosts the
performance of the testsets which are not extracted from the
audiobook, such as ASV, OOD-ASR, IC, SF, and ER.
WavLM Large: Most tasks beneﬁt from the larger model
size, especially for the ASR. We obtain 38% word error rate
reduction on the ASR by model scaling-up. Furthermore, there
is 6.07% absolute improvement on the SID task, indicating
the large model size also impacts the speaker-related tasks.
Compared to the HuBERT Large model, WavLM Large is
consistently better across 14 downstream tasks, demonstrating
the modiﬁcations are effective for the large-scale models.
3) Analysis: Following the SUPERB policies, we weighted-
sum the hidden states of different layers and feed it to the
task-speciﬁc layers. Figure 2 shows the weights of different
layers of HuBERT and WavLM models on the different
downstream tasks of the SUPERB benchmark. The larger
weight indicates the greater contribution of the corresponding
layer. We normalize the weights from different layers based
on the hidden state values of their corresponding layers, which
eliminates the weight bias to layers with smaller hidden state
values.
As for the Base models, the contribution patterns of different
layers are similar between WavLM and HuBERT, as shown in
Figure 2a and 2b. We can observe that the bottom layers
contribute more to speaker-related tasks, such as speaker
identiﬁcation, automatic speaker veriﬁcation, and speaker
diarization. On the other hand, for automatic speech recognition,
phoneme recognition, intent classiﬁcation, and slot ﬁlling tasks,
the top layers are more important. It indicates the Base models
learn speaker information with the bottom layers while the
content and semantic information are encoded in the top layers.
The model behavior is similar to Large models. In Figures
2c and 2d, we can see that the top layers contribute most to
content and semantic tasks, while the middle layers have a
great impact on speaker tasks. The phenomenon indicates how
to leverage hidden states of middle layers is the key to the
success of speaker-related tasks.
Since SUPERB requires the pre-trained model frozen in
ﬁne-tuning, it cannot show the power of pre-trained models. To
explore the limit of our models, we further select typical speech
tasks to evaluate our pre-trained model performance. Four tasks
are used to evaluate our model from different perspectives, and
the training data amount is not on the same scale for the four
tasks. The details of the tasks can be found in Appendix A.

7
0
1
2
3
4
5
6
7
8
9
10
11
12
Layers
SF
IC
PR
ASR
KS
SD
ASV
SID
Tasks
0.0
0.1
0.2
0.3
0.4
(a) HuBERT Base
0
1
2
3
4
5
6
7
8
9
10
11
12
Layers
SF
IC
PR
ASR
KS
SD
ASV
SID
Tasks
0.0
0.1
0.2
0.3
0.4
(b) WavLM Base+
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
Layers
SF
IC
PR
ASR
KS
SD
ASV
SID
Tasks
0.00
0.05
0.10
0.15
0.20
0.25
0.30
(c) HuBERT Large
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
Layers
SF
IC
PR
ASR
KS
SD
ASV
SID
Tasks
0.00
0.05
0.10
0.15
0.20
0.25
0.30
(d) WavLM Large
Fig. 2. Weight analysis on the SUPERB Benchmark. Layer 0 corresponds to the input of the ﬁrst Transformer layer. The y-axis represents different tasks,
while the x-axis represents different layers.
C. Speaker Veriﬁcation
1) problem formulation: The training dataset for speaker
veriﬁcation contains audio and speaker id pairs as D = {xi, yi}.
Given audio clip x and a reference x′, the goal of speaker
veriﬁcation is to determine whether x′ is from the same speaker
as x.
2) Datasets: VoxCeleb1 [20] and VoxCeleb2 [54] datasets
are used in our experiments for speaker veriﬁcation. For
data pre-processing, we apply online data augmentation using
the MUSAN [55] noise, DNS noise [49] and the RIR 4
reverberation with probability 0.6. Voice activity detection
(VAD) processing is not adopted. We use all three ofﬁcial trial
lists Vox1-O, Vox1-E, and Vox1-H to evaluate the system.
3) Setup: We choose the ECAPA-TDNN (small) [19]
architecture as the downstream model and compare different
input speech representations, including handcrafted features and
the pre-training features. The model contains a frame encoder
to extract speaker information from the input sequence, a
statistic pooling layer to transform input to a ﬁxed-dimensional
representation, and a fully connected layer to extract speaker
embedding. For the handcrafted feature, we compare the
reported results in [19] with our re-implemented results, where
we extract the 40-dimensional Fbank feature with 25ms window
size and 10ms frameshift. For pre-trained representations, we
compare WavLM with HuBERT model. Following SUPERB
evaluation, we weighted-sum the representations from different
transformer layers with learnable weights as the input to the
downstream speaker veriﬁcation task.
In the training stage, all the recordings are chunked into 3s
segments to construct the training batches. We use the additive
angular margin (AAM) loss [56] for model optimization and
4https://www.openslr.org/28/
set the margin to 0.2. We also add an Inter-TopK penalty [57]
on the 5 easily misclassiﬁed centers with a penalty margin of
0.1. We train the ECAPA-TDNN system with Fbank feature
for 165 epochs. For systems using pre-trained representations,
we ﬁrst ﬁx the pre-trained model to train ECAPA-TDNN for
20 epochs and then ﬁnetune both the pre-trained and ECAPA-
TDNN models for another 5 epochs. When we add the large
margin ﬁne-tuning strategy [58], we train the systems for an
extra 2 epochs, during which we sample 6s training segments
and set the AAM margin to 0.4.
In the evaluation stage, the whole utterance is fed into the
system to extract speaker embedding. We use cosine similarity
to score the evaluation trial list. We also use the adaptive s-
norm [59], [60] to normalize the trial scores. The imposter
cohort is estimated from the VoxCeleb2 dev set by speaker-
wise averaging all the extracted speaker embeddings. We set
the imposter cohort size to 600 in our experiment. To further
push the performance, we also introduce the quality-aware
score calibration [58] for our best systems, where we randomly
generate 30k trials based on the VoxCeleb2 test set to train
the calibration model.
4) Results: Table II shows the results for the speaker
veriﬁcation task. From the results, we ﬁnd that all the systems
with pre-trained representations exceed the Fbank baseline
system on the Vox1-O and Vox1-E trials. The system with
HuBERT Base representations is slightly worse than the Fbank
feature on the Vox1-H trial. Interestingly, the representations
extracted from our proposed pre-trained models, WavLM Base+
and Large, both outperform the SOTA ECAPA-TDNN system.
Compared with the Fbank feature, the representations from
WavLM Large achieve over 35% relative EER improvement
on all three trials for the VoxCeleb1 evaluation set. To further

8
TABLE II
SPEAKER VERIFICATION RESULTS ON VOXCELEB1. FOR THE LINES WITH ∗
NOTATION, WE ADD THE LARGE MARGIN FINE-TUNING AND
QUALITY-AWARE SCORE CALIBRATION [58] TO PUSH THE LIMIT OF THE
PERFORMANCE.
Feature
EER (%)
Vox1-O
Vox1-E
Vox1-H
ECAPA-TDNN [19]
1.010
1.240
2.320
ECAPA-TDNN (Ours)
1.080
1.200
2.127
HuBERT Base
0.989
1.068
2.216
HuBERT Large
0.808
0.822
1.678
WavLM Base+
0.84
0.928
1.758
WavLM Large
0.617
0.662
1.318
HuBERT Large∗
0.585
0.654
1.342
WavLM Large∗
0.383
0.480
0.986
push the limit of the speaker veriﬁcation system, we introduce
the large margin ﬁne-tuning and quality-aware score calibration
strategies [58] into our best systems and the corresponding
results are listed at the bottom of Table II. With these two
strategies, our best system exceeds the winner system [57]
(Vox1-O: 0.461, Vox1-E: 0.634, Vox1-H: 0.993) in VoxSRC
challenge 20215 on all the three trials.
D. Speaker Diarization
1) problem formulation: Speaker diarization is the task to
answer “Who spoke when?”. Given a speech recording x =
(x1, ..., xT ), we should assign one or more labels to each xt
according to the speaker identity. When we assign more than
one label to xt, it indicates more than one person is speaking
at time t, i.e. speaker overlap. Normally, we cannot know the
number of speakers of a whole recording in advance. Thus,
the built diarization system should have the ability to predict
the speaker number for the whole recording and the speaker
labels for each frame at the same time.
2) Datasets: The dataset used in our experiments is split
into two parts. The ﬁrst part is the large-scale simulation
training data. The second part is the real data, which is used
for evaluation and adaptation. Following the data simulation
setup in [65], all the speech data from Switchboard-2 (Phase I &
II & III), Switchboard Cellular (Part 1 & 2), the NIST Speaker
Recognition Evaluation (2004 & 2005 & 2006 & 2008), the
noises from [55] and the simulated room impulse responses
used in [66] are leveraged for multi-talker speech simulation.
Based on the simulation pipeline introduced in [67], we
generate almost 7000 hours simulation data by setting Nspk = 3
and β = 10. We use the telephone conversation dataset
CALLHOME [68] for evaluation and adaptation. CALLHOME
dataset has 500 sessions of multilingual telephonic speech
where each session contains 2 to 6 speakers. Following the
data usage in [65], we split the CALLHOME dataset into two
parts. The ﬁrst part is used for adaptation and the second part
is used for evaluation.
3) Implementation Details: We leverage the system in [65]
as our downstream speaker diarization model. In the system, a
5https://www.robots.ox.ac.uk/∼vgg/data/voxceleb/interspeech2021.html
long-form recording is ﬁrst segmented into short blocks, where
each short block is assumed to contain at most SLocal speakers.
As with [65], we set SLocal = 3 in our experiment. Then, the
Mel-ﬁlterbank-based features extracted from each short block
are fed into a Transformer encoder to get the diarization results
and SLocal speaker embeddings. With the predicted diarization
results and estimated speaker embeddings, the whole system
is trained by a diarization loss and a speaker loss. During the
inference, the diarization results and speaker embeddings are
ﬁrst predicted for each block. A clustering method is then
applied to associate the embeddings from the same speaker
but in different blocks.
Following the implementation in [65], we set the block length
to 15s, 30s, 30s for the training, adaptation, and evaluation
stage respectively. The constrained AHC (Agglomerative Hier-
archical Clustering) method is used for embedding clustering
during the evaluation stage. When leveraging the pre-trained
representations, as with the implementation in section V-C3,
we just replace the handcrafted Fbank feature with the pre-
trained representation H. Unlike [65], when we feed the
diarization system with the pre-trained representations, we
do not concatenate the context features for each frame and do
not apply the 10 times down-sampling. We ﬁnd that updating
the parameters of the pre-trained model does not improve
performance on the CALLHOME dataset. Thus, we freeze
the pre-trained model in the ﬁne-tuning stage. One possible
explanation is that the test data are real recordings while the
training data are simulated recordings, and the model is over-
ﬁtted if the pre-trained model is not frozen.
4) Results: The speaker diarization results on CALLHOME
dataset are shown in Table III. In our experiment, we try to
reproduce the system in [65] but get slightly worse results.
When we replace the handcrafted feature with pre-trained
representations, all the systems exceed the performance of
our implemented EEND-vector clustering. Compared with the
HuBERT, the representations extracted from our proposed
WavLM are more useful in speaker diarization. Our proposed
WavLM Base+ even outperforms the HuBERT large model.
This is because the WavLM models have seen the multi-talker
and speaker-overlapped speeches during the training process,
and the corresponding training strategy is designed to help
WavLM better process this kind of input. Finally, we also
list the CALLHOME results from some recently published
works. Compared with these results, it is worth noting that
our best system has surpassed all the systems evaluated on the
CALLHOME dataset and achieved a new SOTA performance.
E. Speech Separation
1) problem formulation: The goal of speech separation is to
estimate individual speaker signals from their mixture, where
the source signals may be overlapped with each other entirely
or partially. Given S source signals {xs = (x1, ..., xT )}S
s=1,
the mixed signal is formulated as y = PS
s=1 xs. Xs and Y
denote the Short-Time Fourier Transform (STFT) of the source
signal and mixed signal, respectively. Following [70] and [71],
instead of directly predicting the source STFTs, we ﬁrstly
estimate a group of masks {Ms}S
s=1 with a deep learning

9
TABLE III
DIARIZATION ERROR RATE (DER %) RESULTS ON CALLHOME WITH ESTIMATED NUMBER OF SPEAKERS.
Method
# of speakers in a session
2
3
4
5
6
all
x-vector clustering [61]
15.45
18.01
22.68
31.40
34.27
19.43
SC-EEND [62] ‡
9.57
14.00
21.14
31.07
37.06
15.75
VBx [63] † ‡
9.44
13.89
16.05
13.87
24.73
13.28
EEND-EDA [64]
8.50
13.24
21.46
33.16
40.29
15.29
EEND-EDA clustering [24]
7.11
11.88
14.37
25.95
21.95
11.84
EEND-vector clustering [65]
7.96
11.93
16.38
21.21
23.10
12.49
EEND-vector clustering (Ours)
7.54
12.42
18.41
26.79
27.40
13.31
HuBERT Base & EEND-vector clustering
7.93
12.07
15.21
19.59
23.32
12.63
HuBERT Large & EEND-vector clustering
7.39
11.97
15.76
19.82
22.10
12.40
WavLM Base+ & EEND-vector clustering
6.99
11.12
15.20
21.61
21.70
11.78
WavLM Large & EEND-vector clustering
6.46
10.69
11.84
12.89
20.70
10.35
† Oracle speech segments were used.
‡ Results for these systems are provided in [65].
TABLE IV
SEPARATION RESULTS ON LIBRICSS DATASET. WE FREEZE THE
PRE-TRAINED PARAMETERS BY DEFAULT FOR THE SEPARATION TASK. THE
RESULTS DENOTE %WER SCORE EVALUATED WITH E2E TRANSFORMER
BASED ASR MODEL [69]. 0S AND 0L ARE UTTERANCES WITH
SHORT/LONG INTER-UTTERANCE SILENCE. THE AVG IS THE WEIGHTED
AVERAGED WER OF DIFFERENT OVERLAPPED TESTSETS.
System
Overlap ratio in %
0S
0L
10
20
30
40
avg
Conformer [22]
5.4 5.0 7.5 10.7 13.8 17.1 10.6
Conformer (rerun)
4.5 4.4 6.2
8.5
11.0 12.6
8.3
HuBERT Base
4.7 4.6 6.1
7.9
10.6 12.3
8.1
WavLM Base+
4.5 4.4 5.6
7.5
9.4
10.9
7.4
- unfreeze pre-trained parameters
4.5 4.3 5.9
8.3
11.1 12.5
8.2
WavLM Large
4.2 4.1 4.8
5.8
7.4
8.5
6.0
model, and then obtain each source STFT with Xs = Ms ⊙Y,
where ⊙is an elementwise product.
2) Datasets: Our training dataset for the separation task
consists of 219 hours of artiﬁcially reverberated and mixed
utterances that are sampled randomly from WSJ1 [72]. Four
different mixture types described in [73] are included in the
training set. To generate each training mixture, we randomly
pick one or two speakers from WSJ1 and convolve each with a
room impulse response (RIR) simulated with the image method
[74]. The reverberated signals are then rescaled and mixed with
a source energy ratio between -5 and 5 dB. In addition, we add
simulated isotropic noise [75] with a 0–10 dB signal to noise
ratio. The average overlap ratio of the training set is around
50%.
LibriCSS is used for evaluation [21]. The dataset has 10
hours of seven-channel recordings of mixed and concatenated
LibriSpeech test utterances. The recordings were made by
playing back the mixed audio in a meeting room. We apply the
single-channel utterance-wise evaluation schemes of LibriCSS,
where the long-form recordings are segmented into individual
utterances by using ground-truth time marks to evaluate the
pure separation performance.
3) Implementation details: For the separation task, we use
the previous SOTA work [22] as our baseline model, which
uses the Conformer-based model for separation, and consists
of 16 Conformer encoder layers with 4 attention heads, 256
attention dimensions, and 1024 FFN dimensions. A linear
projection layer and sigmoid activation function are attached
to the ﬁnal encoder for the mask prediction. Given the STFT
of mixed signal Y as the input, the separation model estimates
masks {Ms}S
s=1, then each source signal can be obtained as
{Xs = Ms ⊙Y}S
s=1 for each speaker.
To ﬁne-tune our pre-trained models on the separation, we
use WavLM models as feature extractors and the Conformer-
base architecture as the task-speciﬁc downstream model.
We begin by extracting the pre-trained representation H as
introduced in section V-C3. Secondly, we concatenate the
pre-trained representation and STFT representation in the
feature dimension. Since the window size and hop length
of STFT are typically set to 400 and 160, respectively, the
STFT representation Y = {Yt}T ′
t=1 has the half stride size
compared to the pre-trained representation H = {Ht}T ′/2
t=1 .
To match the size of the time dimension, we duplicate the
pre-trained representation with ˆH = { ˆHt = H⌈t
2 ⌉}T ′
t=1, then
we can concatenate the two representations [Yt, ˆHt] in the
feature dimension for each time step. Finally, we feed the
concatenated representations to the downstream model for the
mask estimation.
The separation models are trained with the AdamW optimizer
[76], where the weight decay is set to 1e-2. We set the learning
rate to 1e-4 and use a warm-up learning schedule with a linear
decay, in which the number of the warm-up steps is 10,000
and the total number of the training step is 260,000.
We follow the previous work [22], [50], [77], [78] to evaluate
our model with an end-to-end Transformer based ASR models
[69], which achieves 2.08% and 4.95% word error rates (WERs)
for LibriSpeech test-clean and test-other, respectively.
4) Results: Table IV shows the single-channel utterance-
wise separation results on LibriCSS dataset. Our WavLM
Base+ and Large models with the frozen pre-trained param-
eters achieve SOTA results on all the overlap ratio settings,
outperforming the baseline results by a large margin.

10
0
1
2
3
4
5
6
7
8
9
10
11
12
Layers
SV
SD
SS
Tasks
0.00
0.05
0.10
0.15
0.20
0.25
0.30
(a) WavLM Base+
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
Layers
SV
SD
SS
Tasks
0.0
0.1
0.2
0.3
(b) WavLM Large
Fig. 3. Weight analysis on the Speaker Veriﬁcation (SV), Speech Diarization
(SD) and Speech Separation (SS) tasks. Layer 0 corresponds to the input of
the ﬁrst Transformer layer. The y-axis represents different tasks, while the
x-axis represents different layers.
We rerun the previous SOTA work [22] with a modiﬁed
Conformer-base architecture [79] and a modiﬁed training loss
[50], which achieve much better baseline results. With the pre-
trained representation provided by the HuBERT Base model,
the performance is comparable with the baseline results for
all the overlap ratios. It is because the HuBERT model is
rarely optimized with speaker-overlapped speech and lacks
multi-speaker modeling during pre-training.
In contrast, our WavLM Base+ with a similar model size can
successfully reduce the WER scores, especially for the large
overlap ratio audios. We ﬁnd ﬁne-tuning the parameters of the
pre-trained model yields better training accuracy but worse
evaluation results than freezing the pre-trained parameters for
the separation task. An explanation is that the separation model
with pre-trained parameters adaptation would be over-ﬁtted with
the artiﬁcially mixed training data, and it is evaluated with a real
meeting recording dataset. With the pre-trained representation
provided by our WavLM Large model, the performance on all
the overlap ratio settings can be further improved. It can achieve
32.5% relative WER score reduction for the 40% overlap ratio
cases and 27.7% relative WER score reduction on average.
5) Weight Analysis: For the speaker veriﬁcation (Sec-
tion V-C), speech diarization (Section V-D) and speech separa-
tion (Section V-E) tasks, we weighted-sum the representations
from different layers of the pre-trained models as the input
to the task-speciﬁc downstream models. Figure 3 shows the
weights of different layers of WavLM Base+ and WavLM Large
models on these tasks. As with the weight analysis on the
SUPERB benchmark in Section V-B3, we can observe that the
contribution mostly comes from the bottom layers for all these
tasks. It indicates that the shallow layers of WavLM models
learn the speaker-related information during the SSL procedure.
It is essential to leverage hidden states of intermediate layers
for speaker-related tasks to make full use of the pre-trained
knowledge of WavLM models.
F. Speech Recognition
1) Problem Formulation: Given the input speech signal
x = (x1, ..., xT ), the goal of speech recognition is to generate
TABLE V
WER ON LIBRISPEECH TEST SETS WHEN TRAINED ON THE LIBRI-LIGHT
LOW-RESOURCE LABELED DATA SETUPS OF 1 HOUR, 10 HOURS AND THE
CLEAN 100H SUBSET OF LIBRISPEECH.
Model
Unlabled Data
LM
test-clean
test-other
1-hour labeled
wav2vec 2.0 Base
LS-960
None
24.5
29.7
WavLM Base
LS-960
None
24.5
29.2
WavLM Base+
MIX-94k
None
22.8
26.7
DeCoAR 2.0
LS-960
4-gram
13.8
29.1
DiscreteBERT
LS-960
4-gram
9.0
17.6
wav2vec 2.0 Base
LS-960
4-gram
5.5
11.3
HuBERT Base
LS-960
4-gram
6.1
11.3
WavLM Base
LS-960
4-gram
5.7
10.8
WavLM Base+
MIX-94k
4-gram
5.4
9.8
wav2vec 2.0 Large
LL-60k
4-gram
3.8
7.1
WavLM Large
MIX-94k
4-gram
3.8
6.6
wav2vec2.0 Large
LL-60k
Transformer
2.9
5.8
HuBERT Large
LL-60k
Transformer
2.9
5.4
WavLM Large
MIX-94k
Transformer
2.9
5.1
10-hour labeled
wav2vec 2.0
LS-960
None
11.1
17.6
WavLM Base
LS-960
None
9.8
16.0
WavLM Base+
MIX-94k
None
9.0
14.7
DeCoAR 2.0
LS-960
4-gram
5.4
13.3
DiscreteBERT
LS-960
4-gram
5.9
14.1
wav2vec 2.0
LS-960
4-gram
4.3
9.5
HuBERT Base
LS-960
4-gram
4.3
9.4
WavLM Base
LS-960
4-gram
4.3
9.2
WavLM Base+
MIX-94k
4-gram
4.2
8.8
wav2vec 2.0 Large
LL-60k
4-gram
3.0
5.8
WavLM Large
MIX-94k
4-gram
2.9
5.5
wav2vec 2.0 Large
LL-60k
Transformer
2.6
4.9
HuBERT Large
LL-60k
Transformer
2.4
4.6
WavLM Large
MIX-94k
Transformer
2.4
4.6
100-hour labeled
wav2vec 2.0 Base
LS-960
None
6.1
13.3
WavLM Base
LS-960
None
5.7
12.0
WavLM Base+
MIX-94k
None
4.6
10.1
DeCoAR 2.0
LS-960
4-gram
5.0
12.1
DiscreteBERT
LS-960
4-gram
4.5
12.1
wav2vec 2.0 Base
LS-960
4-gram
3.4
8.0
HuBERT Base
LS-960
4-gram
3.4
8.1
WavLM Base
LS-960
4-gram
3.4
7.7
WavLM Base+
MIX-94k
4-gram
2.9
6.8
wav2vec 2.0 Large
LL-60k
4-gram
2.3
4.6
WavLM Large
MIX-94k
4-gram
2.3
4.6
wav2vec 2.0 Large
LL-60k
Transformer
2.0
4.0
HuBERT Large
LL-60k
Transformer
2.1
3.9
WavLM Large
MIX-94k
Transformer
2.1
4.0
the corresponding transcription y = (y1, ..., yL), where T and
L are the lengths of the speech and transcription, respectively.
2) Datasets: We use LibriSpeech for our ASR experiments.
For the ﬁne-tuning, we consider four different partitions: 960
hours of transcribed LibriSpeech [51], the train-clean-100
subset (100 hours labeled data), as well as the Libri-Light
[9] limited resource training subsets originally extracted from
LibriSpeech, including train-10h (10 hours labeled data) and
train-1h (1 hour labeled data). We follow the evaluation protocol
of Libri-Light for these splits and evaluate on the standard
LibriSpeech test-clean/other sets.
3) Implementation Details: The pre-trained models are ﬁne-
tuned for speech recognition by adding a randomly initialized
linear projection layer on top of the Transformer encoder.
Models are optimized based on a CTC loss [84] where we
have 29 tokens for character targets plus a word boundary
token. We apply a modiﬁed version of SpecAugment [85]

11
TABLE VI
WER ON LIBRISPEECH WHEN USING ALL 960 HOURS OF LABELED DATA.
Model
Unlabled Data
LM
test-clean
test-other
Supervised
CTC Transf [80]
-
CLM+Transf.
2.5
5.5
S2S Transf. [80]
-
CLM+Transf.
2.3
5.2
Transf. Transducer [81]
-
Transf.
2.0
4.6
ContextNet [82]
-
LSTM
1.9
4.1
Conformer Transducer [83]
-
LSTM
1.9
3.9
Pre-training
wav2vec 2.0 Large
LL-60k
Transformer
1.8
3.3
HuBERT Large
LL-60k
Transformer
1.9
3.3
WavLM Large
MIX-94k
Transformer
1.8
3.2
by masking time-steps and channels: we randomly select the
starting positions with a predetermined probability and replace
a span of ten subsequent time-steps with a mask embedding;
different spans may overlap and we use the same masked time
step embedding as the one used for pre-training. We also mask
channels by choosing a number of channels as starting indices
and then expanding to the subsequent 64 channels. Spans may
overlap and the selected spans are set to zeros.
During ﬁne-tuning, the convolutional encoder is always ﬁxed
and we freeze the Transformer encoder for the ﬁrst 10k steps.
We optimize with Adam and a tri-stage rate schedule where
the learning rate is warmed up for the ﬁrst 10% of the updates,
held constant for the next 40%, and then linearly decayed for
the remainder. The Base and Base+ models are ﬁne-tuned on
8 GPUs with a batch size equivalent to 200 seconds of audio
for each GPU. The Large model is ﬁne-tuned on 24 GPUs
with a batch size equivalent to 80 seconds of audio for each
GPU. We also use LayerDrop [86], [87] at a rate of 0.05 for
Base/Base+ and 0.1 for LARGE. The summary of the ﬁne-
tuning hyperparameter settings used for different labeled data
setups can be found in Appendix A.
For evaluation, we use wav2letter++ [88] beam search
decoder with language model (LM) fused decoding as :
logpCT C(y|x) + w1logpLM(y) + w2|y|
(9)
where w1 is the language model weight and w2 is the
word insertion weight. We consider a 4-gram model and a
Transformer model, which are identical to [5]. The evaluation
hyperparameters are also based on [5].
4) Results: Table V presents the results for the low-resource
setup, where the pre-trained models are ﬁne-tuned on the 1
hour, 10 hours or 100 hours of labeled data. We compare our
method with several competitive self-supervised approaches in
the literature, including DeCoAR 2.0 [37], DiscreteBERT [41],
wav2vec 2.0 [5] and HuBERT [6]. Without LM fusion, the
WavLM Base model outperforms wav2vec 2.0 by a large margin
for all ﬁne-tuning splits, indicating the superiority of our model
architecture. Its performance matches or outperforms wav2vec
2.0 and HuBERT with LM. WavLM Base+ improves WavLM
Base, especially on the test-other set, indicating increasing the
out-of-domain unlabeled data also works for ASR. For the
Large model, the observation is consistent that our method
achieves comparable or better performance than the baselines.
Table VI reports results on the full 960 hours of LibriSpeech
data. Overall, the pre-training methods can outperform all
supervised models and our model is on par with the two best
pre-training results in this setting.
VI. CONCLUSION
We present WavLM, a large-scale pre-trained model with 94k
hour audio as inputs, to solve full stack speech processing tasks.
WavLM extends the HuBERT framework to masked speech
prediction and denoising modeling, enabling the pre-trained
models to perform well on both ASR and non-ASR tasks.
WavLM updates state-of-the-art results on the SUPERB, as
well as the representative testsets of speaker veriﬁcation, speech
separation, and speaker diarization. In contrast to previous SSL
models, WavLM is not only effective for the ASR task but
also has the potential to become the next-generation backbone
network for speaker-related tasks.
In the future, we would like to scale up the model size to
increase the model capability, as previous work has shown
the beneﬁts of more parameters [47]. Meanwhile, the model
compression technique is also worth trying due to the time
constraint and limited test time resources in real scenarios. It
is also a promising direction to jointly learn text and speech
representation in a self-supervised pre-training framework [89],
as the huge amount of text data might increase the capability
of speech content modeling.
APPENDIX A
HYPERPARAMTERS FOR PRE-TRAINING
Table VII shows the hyperparameters used for pre-training
our WavLM Base, Base+, and Large model, which are adapted
from the previous work [6].
TABLE VII
HYPERPARAMTERS FOR PRE-TRAINING WAVLM MODELS . THE UNIT IN
BATCH SIZE COMPUTING IS SECOND. WE USE 32 V100 GPUS FOR BASE
MODEL TRAINING, AND 64 V100 GPUS FOR LARGE MODEL.
Model
pre-train data
update steps
learning rate
warmup steps
batch size
WavLM Base
960h
400k
5e-4
32k
350s
WavLM Base+
94kh
1.2M
5e-4
96k
350s
WavLM Large
94kh
700k
1.5e-3
32k
720s
SETTINGS OF DOWNSTREAM TASKS
For the universal representation evaluation, we use the same
settings for all the SUPERB tasks in accordance with the
SUPERB policies [8].
As for the four additional downstream tasks, including
speaker veriﬁcation, speaker diarization, speech separation,
and speech recognition, the implementations are shown in
Table VIII, following the previous works [6], [19], [22], [65].
HYPERPARAMTERS FOR FINE-TUNING
As for the universal representation evaluation, Table IX
shows the hyperparameters of the learning rate and batch size
for ﬁne-tuning our WavLM models in the SUPERB downstream
tasks. For the QbE task, which is evaluated by dynamic time
warping without ﬁne-tuning, we ﬁnd that the best results for all
the three WavLM models are always from the representations of

12
TABLE VIII
DIFFERENT SETTINGS OF THE DOWNSTREAM TASKS. IN THE SPEAKER
DIARIZATION TASK, IT SHOULD BE NOTED THAT THE CALLHOME
DATASET IS USED FOR DOMAIN ADAPTATION.
Task
dataset
training data duration
downstream model
Speaker Veriﬁcation
VoxCeleb
2300h
ECAPA-TDNN
Speaker Diarization
CALLHOME
8.7h
Transformer
Speech Separation
LibriCSS
219h
Conformer
Speech Recognition
LibriSpeech
1h/10h/100h/960h
Linear
the last layer. All the other hyperparameters of each downstream
task are exactly the same as the ofﬁcial implementation of
SUPERB6.
As for the speech recognition task ﬁne-tuning, Table X
summarizes the hyperparameters used for different labeled data
setups.
TABLE IX
HYPERPARAMENTERS OF FINE-TUNING WAVLM MODELS IN SUPERB
DOWNSTREAM TASKS. THE BATCH SIZE OF SPEECH TRANSLATION TASK
DENOTES THE NUMBER OF TOKENS IN EACH TRAINING BATCH.
Task
WavLM Base
WavLM Base+
WavLM Large
learning rate
batch size
learning rate
batch size
learning rate
batch size
Speaker Identiﬁcation
2e-1
512
1e-1
512
5e-2
512
Automatic Speaker Veriﬁcation
5e-5
512
5e-5
512
5e-5
512
Speaker Diarization
2e-3
256
5e-4
256
5e-3
256
Phoneme Recognition
5e-4
128
5e-4
128
2e-4
128
Automatic Speech Recognition
5e-4
128
5e-4
128
1e-4
128
Out-of-domain Automatic Speech Recognition
1e-4
16
1e-4
16
1e-4
16
Keyword Spotting
1e-5
512
1e-5
512
1e-5
512
Speech Translation
1e-3
80k
1e-3
80k
1e-3
160k
Intent Classiﬁcation
5e-5
128
2e-5
128
5e-4
128
Slot Filling
2e-4
128
2e-4
128
1e-4
128
Emotion Recognition
1e-4
32
1e-4
32
1e-5
32
Speech Enhancement
5e-4
64
5e-4
64
5e-4
64
Speech Separation
5e-4
64
1e-3
64
5e-4
64
Voice Conversion
1e-4
6
1e-4
6
1e-4
6
TABLE X
HYPERPARAMENTERS OF FINE-TUNING WAVLM MODELS IN SPEECH
RECOGNITION TASK.
Setup
updates
learning rate
timestep mask prob.
channel mask prob.
1 hour (Base/Base+)
13k
5e-5
0.065
0.004
10 hour (Base/Base+)
25k
2e-5
0.075
0.008
100 hour (Base/Base+)
80k
3e-5
0.065
0.008
1 hour (Large)
13k
3e-4
0.075
0.004
10 hour (Large)
20k
1e-4
0.075
0.004
100 hour (Large)
80k
3e-5
0.005
0.008
960 hour (Large)
320k
3e-5
0.005
0.004
REFERENCES
[1] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training
of deep bidirectional transformers for language understanding,” in North
American Chapter of the Association for Computational Linguistics
(NAACL), 2019, pp. 4171–4186.
[2] Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. R. Salakhutdinov, and
Q. V. Le, “Xlnet: Generalized autoregressive pretraining for language
understanding,” in Advances in Neural Information Processing Systems
(NeurIPS), H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alch´e-Buc,
E. Fox, and R. Garnett, Eds., vol. 32.
Curran Associates, Inc., 2019.
[3] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou,
W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a
uniﬁed text-to-text transformer,” Journal of Machine Learning Research
(JMLR), vol. 21, no. 140, pp. 1–67, 2020.
[4] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with
contrastive predictive coding,” CoRR, vol. abs/1807.03748, 2018.
[5] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A
framework for self-supervised learning of speech representations,” in
Advances in Neural Information Processing Systems (NeurIPS), 2020.
6https://github.com/s3prl/s3prl
[6] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and
A. Mohamed, “Hubert: Self-supervised speech representation learning
by masked prediction of hidden units,” arXiv preprint arXiv:2106.07447,
2021.
[7] C. Wang, Y. Wu, Y. Qian, K. Kumatani, S. Liu, F. Wei, M. Zeng, and
X. Huang, “Unispeech: Uniﬁed speech representation learning with
labeled and unlabeled data,” in International Conference on Machine
Learning(ICML), ser. Proceedings of Machine Learning Research,
M. Meila and T. Zhang, Eds., vol. 139. PMLR, 2021, pp. 10 937–10 947.
[Online]. Available: http://proceedings.mlr.press/v139/wang21y.html
[8] S. Yang, P. Chi, Y. Chuang, C. J. Lai, K. Lakhotia, Y. Y. Lin, A. T. Liu,
J. Shi, X. Chang, G. Lin, T. Huang, W. Tseng, K. Lee, D. Liu, Z. Huang,
S. Dong, S. Li, S. Watanabe, A. Mohamed, and H. Lee, “SUPERB:
Speech Processing Universal PERformance Benchmark,” in Interspeech,
2021, pp. 1194–1198.
[9] J. Kahn, M. Rivi`ere, W. Zheng, E. Kharitonov, Q. Xu, P.-E. Mazar´e,
J. Karadayi, V. Liptchinsky, R. Collobert, C. Fuegen et al., “Libri-light:
A benchmark for asr with limited or no supervision,” in International
Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE,
2020, pp. 7669–7673.
[10] W. Chan, D. Park, C. Lee, Y. Zhang, Q. Le, and M. Norouzi, “Speechstew:
Simply mix all available speech recognition data to train one large neural
network,” arXiv preprint arXiv:2104.02133, 2021.
[11] T. Likhomanenko, Q. Xu, V. Pratap, P. Tomasello, J. Kahn, G. Avidov,
R. Collobert, and G. Synnaeve, “Rethinking evaluation in asr: Are our
models robust enough?” arXiv preprint arXiv:2010.11745, 2020.
[12] A. Narayanan, A. Misra, K. C. Sim, G. Pundak, A. Tripathi, M. Elfeky,
P. Haghani, T. Strohman, and M. Bacchiani, “Toward domain-invariant
speech recognition via large scale training,” in 2018 IEEE Spoken
Language Technology Workshop (SLT).
IEEE, 2018, pp. 441–447.
[13] A. Narayanan, R. Prabhavalkar, C.-C. Chiu, D. Rybach, T. N. Sainath, and
T. Strohman, “Recognizing long-form speech using streaming end-to-end
models,” in 2019 IEEE Automatic Speech Recognition and Understanding
Workshop (ASRU).
IEEE, 2019, pp. 920–927.
[14] W.-N. Hsu, A. Sriram, A. Baevski, T. Likhomanenko, Q. Xu, V. Pratap,
J. Kahn, A. Lee, R. Collobert, G. Synnaeve et al., “Robust wav2vec 2.0:
Analyzing domain shift in self-supervised pre-training,” arXiv preprint
arXiv:2104.01027, 2021.
[15] Z. Chi, S. Huang, L. Dong, S. Ma, S. Singhal, P. Bajaj, X. Song,
and F. Wei, “XLM-E: cross-lingual language model pre-training via
ELECTRA,” CoRR, vol. abs/2106.16138, 2021. [Online]. Available:
https://arxiv.org/abs/2106.16138
[16] G. Chen, S. Chai, G. Wang, J. Du, W. Zhang, C. Weng, D. Su, D. Povey,
J. Trmal, J. Zhang, M. Jin, S. Khudanpur, S. Watanabe, S. Zhao, W. Zou,
X. Li, X. Yao, Y. Wang, Y. Wang, Z. You, and Z. Yan, “Gigaspeech:
An evolving, multi-domain asr corpus with 10,000 hours of transcribed
audio,” in Interspeech, 2021.
[17] C. Wang, M. Rivi`ere, A. Lee, A. Wu, C. Talnikar, D. Haziza,
M. Williamson, J. Pino, and E. Dupoux, “Voxpopuli: A large-scale
multilingual speech corpus for representation learning, semi-supervised
learning and interpretation,” arXiv preprint arXiv:2101.00390, 2021.
[18] H.-S. Tsai, H.-J. Chang, W.-C. Huang, Z. Huang, K. Lakhotia, S.-w.
Yang, S. Dong, A. T. Liu, C.-I. J. Lai, J. Shi et al., “Superb-sg: Enhanced
speech processing universal performance benchmark for semantic and
generative capabilities,” arXiv preprint arXiv:2203.06849, 2022.
[19] B. Desplanques, J. Thienpondt, and K. Demuynck, “Ecapa-tdnn: Em-
phasized channel attention, propagation and aggregation in tdnn based
speaker veriﬁcation,” arXiv preprint arXiv:2005.07143, 2020.
[20] A. Nagrani, J. S. Chung, and A. Zisserman, “Voxceleb: a large-scale
speaker identiﬁcation dataset,” arXiv preprint arXiv:1706.08612, 2017.
[21] Z. Chen, T. Yoshioka, L. Lu, T. Zhou, Z. Meng, Y. Luo, J. Wu, X. Xiao,
and J. Li, “Continuous speech separation: Dataset and analysis,” in
International Conference on Acoustics, Speech and Signal Processing
(ICASSP).
IEEE, 2020, pp. 7284–7288.
[22] S. Chen, Y. Wu, Z. Chen, J. Wu, J. Li, T. Yoshioka, C. Wang, S. Liu,
and M. Zhou, “Continuous speech separation with conformer,” in
International Conference on Acoustics, Speech and Signal Processing
(ICASSP).
IEEE, 2021, pp. 5749–5753.
[23] T. J. Park, N. Kanda, D. Dimitriadis, K. J. Han, S. Watanabe, and
S. Narayanan, “A review of speaker diarization: Recent advances with
deep learning,” arXiv preprint arXiv:2101.09624, 2021.
[24] S. Horiguchi, S. Watanabe, P. Garcia, Y. Xue, Y. Takashima,
and Y. Kawaguchi, “Towards neural diarization for unlimited num-
bers of speakers using global and local attractors,” arXiv preprint
arXiv:2107.01545, 2021.
[25] Y.-C. Chen, S.-F. Huang, H.-y. Lee, Y.-H. Wang, and C.-H. Shen, “Audio
word2vec: Sequence-to-sequence autoencoding for unsupervised learning

13
of audio segmentation and representation,” IEEE/ACM Transactions on
Audio, Speech, and Language Processing (TASLP), vol. 27, no. 9, pp.
1481–1493, 2019.
[26] W.-N. Hsu, Y. Zhang, and J. Glass, “Learning latent representations
for speech generation and transformation,” in Interspeech, 2017, pp.
1273–1277.
[27] ——, “Unsupervised learning of disentangled and interpretable repre-
sentations from sequential data,” in Advances in Neural Information
Processing Systems (NeurIPS), 2017.
[28] J. Chorowski, R. J. Weiss, S. Bengio, and A. van den Oord, “Unsupervised
speech representation learning using wavenet autoencoders,” IEEE/ACM
Transactions on Audio, Speech, and Language Processing (TASLP),
vol. 27, no. 12, pp. 2041–2053, 2019.
[29] Y.-A. Chung, H. Tang, and J. Glass, “Vector-quantized autoregressive
predictive coding,” in Interspeech, 2020, pp. 3760–3764.
[30] Y.-A. Chung, W.-N. Hsu, H. Tang, and J. Glass, “An Unsupervised Au-
toregressive Model for Speech Representation Learning,” in Interspeech,
2019, pp. 146–150.
[31] Y. Chung and J. Glass, “Generative pre-training for speech with autore-
gressive predictive coding,” in International Conference on Acoustics,
Speech and Signal Processing (ICASSP).
IEEE, 2020, pp. 3497–3501.
[32] Y.-A. Chung and J. Glass, “Improved speech representations with multi-
target autoregressive predictive coding,” in Association for Computational
Linguistics (ACL).
Online: Association for Computational Linguistics,
Jul. 2020, pp. 2353–2358.
[33] A. H. Liu, Y.-A. Chung, and J. Glass, “Non-autoregressive predictive
coding for learning speech representations from local dependencies,”
arXiv preprint arXiv:2011.00406, 2020.
[34] A. T. Liu, S.-W. Li, and H.-y. Lee, “Tera: Self-supervised learning
of transformer encoder representation for speech,” arXiv preprint
arXiv:2007.06028, 2020.
[35] A. T. Liu, S.-w. Yang, P.-H. Chi, P.-c. Hsu, and H.-y. Lee, “Mockingjay:
Unsupervised speech representation learning with deep bidirectional
transformer encoders,” in International Conference on Acoustics, Speech
and Signal Processing (ICASSP).
IEEE, 2020.
[36] S. Ling, Y. Liu, J. Salazar, and K. Kirchhoff, “Deep contextualized
acoustic representations for semi-supervised speech recognition,” in
International Conference on Acoustics, Speech and Signal Processing
(ICASSP).
IEEE, 2020, pp. 6429–6433.
[37] S. Ling and Y. Liu, “DeCoAR 2.0: Deep contextualized acoustic repre-
sentations with vector quantization,” arXiv preprint arXiv:2012.06659,
2020.
[38] Y. Gong, C.-I. J. Lai, Y.-A. Chung, and J. Glass, “Ssast: Self-supervised
audio spectrogram transformer,” arXiv preprint arXiv:2110.09784, 2021.
[39] S. Schneider, A. Baevski, R. Collobert, and M. Auli, “wav2vec:
Unsupervised pre-training for speech recognition.” in Interspeech, 2019.
[40] A. Baevski, S. Schneider, and M. Auli, “vq-wav2vec: Self-supervised
learning of discrete speech representations,” in International Conference
on Learning Representations (ICLR), 2020.
[41] A. Baevski and A. Mohamed, “Effectiveness of self-supervised pre-
training for asr,” in International Conference on Acoustics, Speech and
Signal Processing (ICASSP).
IEEE, 2020, pp. 7694–7698.
[42] Y.-A. Chung, Y. Zhang, W. Han, C.-C. Chiu, J. Qin, R. Pang, and
Y. Wu, “W2v-bert: Combining contrastive learning and masked lan-
guage modeling for self-supervised speech pre-training,” arXiv preprint
arXiv:2108.06209, 2021.
[43] S. Pascual, M. Ravanelli, J. Serr`a, A. Bonafonte, and Y. Bengio, “Learning
problem-agnostic speech representations from multiple self-supervised
tasks,” in Interspeech, 2019, pp. 161–165.
[44] M. Ravanelli, J. Zhong, S. Pascual, P. Swietojanski, J. Monteiro, J. Trmal,
and Y. Bengio, “Multi-task self-supervised learning for robust speech
recognition,” in International Conference on Acoustics, Speech and Signal
Processing (ICASSP).
IEEE, 2020, pp. 6989–6993.
[45] J. Bai, B. Li, Y. Zhang, A. Bapna, N. Siddhartha, K. C. Sim, and T. N.
Sainath, “Joint unsupervised and supervised training for multilingual
asr,” in ICASSP 2022-2022 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP).
IEEE, 2022, pp. 6402–6406.
[46] S. Chen, Y. Wu, C. Wang, Z. Chen, Z. Chen, S. Liu, J. Wu, Y. Qian, F. Wei,
J. Li et al., “Unispeech-sat: Universal speech representation learning with
speaker aware pre-training,” arXiv preprint arXiv:2110.05752, 2021.
[47] Y. Zhang, D. S. Park, W. Han, J. Qin, A. Gulati, J. Shor, A. Jansen,
Y. Xu, Y. Huang, S. Wang et al., “Bigssl: Exploring the frontier of
large-scale semi-supervised learning for automatic speech recognition,”
arXiv preprint arXiv:2109.13226, 2021.
[48] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in
Neural Information Processing Systems (NeurIPS), 2017, pp. 5998–6008.
[49] C. K. Reddy, H. Dubey, K. Koishida, A. Nair, V. Gopal, R. Cutler,
S. Braun, H. Gamper, R. Aichner, and S. Srinivasan, “Interspeech 2021
deep noise suppression challenge,” arXiv preprint arXiv:2101.01902,
2021.
[50] J. Wu, Z. Chen, S. Chen, Y. Wu, T. Yoshioka, N. Kanda, S. Liu, and J. Li,
“Investigation of Practical Aspects of Single Channel Speech Separation
for ASR,” in Interspeech, 2021, pp. 3066–3070.
[51] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech:
An ASR corpus based on public domain audio books,” in International
Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE,
2015, pp. 5206–5210.
[52] M. Ding, Z. Yang, W. Hong, W. Zheng, C. Zhou, D. Yin, J. Lin, X. Zou,
Z. Shao, H. Yang et al., “Cogview: Mastering text-to-image generation
via transformers,” arXiv preprint arXiv:2105.13290, 2021.
[53] M. Rivi`ere, A. Joulin, P.-E. Mazar´e, and E. Dupoux, “Unsupervised
pretraining transfers well across languages,” in International Conference
on Acoustics, Speech and Signal Processing (ICASSP).
IEEE, 2020,
pp. 7414–7418.
[54] J. S. Chung, A. Nagrani, and A. Zisserman, “Voxceleb2: Deep speaker
recognition,” arXiv preprint arXiv:1806.05622, 2018.
[55] D. Snyder, G. Chen, and D. Povey, “Musan: A music, speech, and noise
corpus,” arXiv preprint arXiv:1510.08484, 2015.
[56] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “Arcface: Additive angular
margin loss for deep face recognition,” in Conference on Computer Vision
and Pattern Recognition (CVPR), 2019, pp. 4690–4699.
[57] M. Zhao, Y. Ma, M. Liu, and M. Xu, “The speakin system for voxceleb
speaker recognition challange 2021,” arXiv preprint arXiv:2109.01989,
2021.
[58] J. Thienpondt, B. Desplanques, and K. Demuynck, “The idlab voxsrc-20
submission: Large margin ﬁne-tuning and quality-aware score calibration
in dnn based speaker veriﬁcation,” in International Conference on
Acoustics, Speech and Signal Processing (ICASSP).
IEEE, 2021, pp.
5814–5818.
[59] Z. N. Karam, W. M. Campbell, and N. Dehak, “Towards reduced false-
alarms using cohorts,” in International Conference on Acoustics, Speech
and Signal Processing (ICASSP).
IEEE, 2011, pp. 4512–4515.
[60] S. Cumani, P. D. Batzu, D. Colibro, C. Vair, P. Laface, and V. Vasilakakis,
“Comparison of speaker recognition approaches for real applications.” in
Interspeech, 2011, pp. 2365–2368.
[61] S. Horiguchi, Y. Fujita, S. Watanabe, Y. Xue, and K. Nagamatsu, “End-to-
end speaker diarization for an unknown number of speakers with encoder-
decoder based attractors,” arXiv preprint arXiv:2005.09921, 2020.
[62] Y. Fujita, S. Watanabe, S. Horiguchi, Y. Xue, J. Shi, and K. Nagamatsu,
“Neural speaker diarization with speaker-wise chain rule,” arXiv preprint
arXiv:2006.01796, 2020.
[63] F. Landini, J. Profant, M. Diez, and L. Burget, “Bayesian hmm clustering
of x-vector sequences (vbx) in speaker diarization: theory, implementation
and analysis on standard tasks,” Computer Speech & Language, vol. 71,
p. 101254, 2022.
[64] S. Horiguchi, Y. Fujita, S. Watanabe, Y. Xue, and P. Garcia, “Encoder-
decoder based attractor calculation for end-to-end neural diarization,”
arXiv preprint arXiv:2106.10654, 2021.
[65] K. Kinoshita, M. Delcroix, and N. Tawara, “Advances in integration of
end-to-end neural and clustering-based diarization for real conversational
speech,” arXiv preprint arXiv:2105.09040, 2021.
[66] T. Ko, V. Peddinti, D. Povey, M. L. Seltzer, and S. Khudanpur, “A study
on data augmentation of reverberant speech for robust speech recognition,”
in International Conference on Acoustics, Speech and Signal Processing
(ICASSP).
IEEE, 2017, pp. 5220–5224.
[67] Y. Fujita, N. Kanda, S. Horiguchi, K. Nagamatsu, and S. Watanabe,
“End-to-end neural speaker diarization with permutation-free objectives,”
in Interspeech, 2019, pp. 4300–4304.
[68] M. Przybocki and A. Martin, “2000 nist speaker recognition evaluation
(ldc2001s97),” in Philadelphia, New Jersey: Linguistic Data Consortium,
2001.
[69] C. Wang, Y. Wu, Y. Du, J. Li, S. Liu, L. Lu, S. Ren, G. Ye, S. Zhao,
and M. Zhou, “Semantic mask for transformer based end-to-end speech
recognition,” Interspeech, 2020.
[70] Y. Wang, A. Narayanan, and D. Wang, “On training targets for supervised
speech separation,” IEEE/ACM Transactions on Audio, Speech, and
Language Processing (TASLP), vol. 22, no. 12, pp. 1849–1858, 2014.
[71] H. Erdogan, J. R. Hershey, S. Watanabe, and J. Le Roux, “Deep
recurrent networks for separation and recognition of single-channel
speech in nonstationary background audio,” in New Era for Robust
Speech Recognition.
Springer, 2017, pp. 165–186.
[72] L. D. C. Philadelphia, “CSR-II (WSJ1) Complete,” 1994, http://catalog.
ldc.upenn.edu/LDC94S13A.

14
[73] T. Yoshioka, H. Erdogan, Z. Chen, and F. Alleva, “Multi-microphone
neural speech separation for far-ﬁeld multi-talker speech recognition,” in
International Conference on Acoustics, Speech and Signal Processing
(ICASSP).
IEEE, 2018, pp. 5739–5743.
[74] J. Allen and D. Berkley, “Image method for efﬁciently simulating small-
room acoustics,” The Journal of the Acoustical Society of America (JASA),
vol. 65, pp. 943–950, 1979.
[75] E. A. Habets and S. Gannot, “Generating sensor signals in isotropic
noise ﬁelds,” The Journal of the Acoustical Society of America (JASA),
vol. 122, no. 6, pp. 3464–3470, 2007.
[76] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in
International Conference on Learning Representations (ICLR), 2018.
[77] S. Chen, Y. Wu, Z. Chen, T. Yoshioka, S. Liu, J. Li, and X. Yu, “Don’t
shoot butterﬂy with riﬂes: Multi-channel continuous speech separation
with early exit transformer,” in International Conference on Acoustics,
Speech and Signal Processing (ICASSP).
IEEE, 2021, pp. 6139–6143.
[78] S. Chen, Y. Wu, Z. Chen, J. Wu, T. Yoshioka, S. Liu, J. Li, and X. Yu,
“Ultra Fast Speech Separation Model with Teacher Student Learning,” in
Interspeech, 2021, pp. 3026–3030.
[79] G. Ye, V. Mazalov, J. Li, and Y. Gong, “Have best of both worlds:
two-pass hybrid and e2e cascading framework for speech recognition,”
arXiv preprint arXiv:2110.04891, 2021.
[80] G. Synnaeve, Q. Xu, J. Kahn, T. Likhomanenko, E. Grave, V. Pratap,
A. Sriram, V. Liptchinsky, and R. Collobert, “End-to-end asr: from
supervised to semi-supervised learning with modern architectures,” arXiv
preprint arXiv:1911.08460, 2019.
[81] Q. Zhang, H. Lu, H. Sak, A. Tripathi, E. McDermott, S. Koo, and
S. Kumar, “Transformer transducer: A streamable speech recognition
model with transformer encoders and rnn-t loss,” in International
Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE,
2020, pp. 7829–7833.
[82] W. Han, Z. Zhang, Y. Zhang, J. Yu, C.-C. Chiu, J. Qin, A. Gulati, R. Pang,
and Y. Wu, “Contextnet: Improving convolutional neural networks for
automatic speech recognition with global context,” Interspeech, pp. 3610–
3614, 2020.
[83] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han,
S. Wang, Z. Zhang, Y. Wu et al., “Conformer: Convolution-augmented
transformer for speech recognition,” Interspeech, pp. 5036–5040, 2020.
[84] A. Graves, S. Fern´andez, F. Gomez, and J. Schmidhuber, “Connectionist
temporal classiﬁcation: labelling unsegmented sequence data with
recurrent neural networks,” in International Conference on Machine
Learning (ICML), 2006, pp. 369–376.
[85] D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk,
and Q. V. Le, “Specaugment: A simple data augmentation method for
automatic speech recognition,” in Interspeech, 2019, pp. 2613–2617.
[86] A. Fan, E. Grave, and A. Joulin, “Reducing transformer depth on demand
with structured dropout,” arXiv preprint arXiv:1909.11556, 2019.
[87] G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger, “Deep
networks with stochastic depth,” in European Conference on Computer
Vision (ECCV).
Springer, 2016, pp. 646–661.
[88] V. Pratap et al., “Wav2letter++: A fast open-source speech recognition
system,” in International Conference on Acoustics, Speech and Signal
Processing (ICASSP).
IEEE, 2019, pp. 6460–6464.
[89] J. Ao, R. Wang, L. Zhou, S. Liu, S. Ren, Y. Wu, T. Ko, Q. Li, Y. Zhang,
Z. Wei et al., “Speecht5: Uniﬁed-modal encoder-decoder pre-training for
spoken language processing,” arXiv preprint arXiv:2110.07205, 2021.
