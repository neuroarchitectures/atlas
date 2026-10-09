# Paper (Jang et al. 2021)

> Source: `https://arxiv.org/abs/2106.07889`

---

UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram
Discriminators for High-Fidelity Waveform Generation
Won Jang1, Dan Lim2, Jaesam Yoon1, Bongwan Kim1, Juntae Kim1
1Kakao Enterprise Corporation, Seongnam, Republic of Korea
2Kakao Corporation, Seongnam, Republic of Korea
1{taylor.martin, jeff.rey, montae.k, jaytee.k}@kakaoenterprise.com
2satoshi.2018@kakaocorp.com
Abstract
Most neural vocoders employ band-limited mel-spectrograms
to generate waveforms. If full-band spectral features are used
as the input, the vocoder can be provided with as much acoustic
information as possible. However, in some models employing
full-band mel-spectrograms, an over-smoothing problem occurs
as part of which non-sharp spectrograms are generated. To ad-
dress this problem, we propose UnivNet, a neural vocoder that
synthesizes high-ﬁdelity waveforms in real time. Inspired by
works in the ﬁeld of voice activity detection, we added a multi-
resolution spectrogram discriminator that employs multiple lin-
ear spectrogram magnitudes computed using various parameter
sets. Using full-band mel-spectrograms as input, we expect to
generate high-resolution signals by adding a discriminator that
employs spectrograms of multiple resolutions as the input. In
an evaluation on a dataset containing information on hundreds
of speakers, UnivNet obtained the best objective and subjec-
tive results among competing models for both seen and unseen
speakers. These results, including the best subjective score for
text-to-speech, demonstrate the potential for fast adaptation to
new speakers without a need for training from scratch.
Index Terms: Neural vocoder, text-to-speech, multi-scale dis-
criminators, over-smoothing problem
1. Introduction
Vocoders have been employed in various ﬁelds such as text-to-
speech[1, 2, 3, 4], voice conversion[5], and speech-to-speech
translation[6]. Neural vocoders based on deep neural networks
can generate human-like voices, instead of using traditional
methods that contain audible artifacts[7, 8, 9].
Recently, a
few attempts have been made to apply generative adversarial
networks (GANs)[10] in vocoders to generate more realistic
waveforms[11, 12, 13, 14, 15]. This new methodology, involv-
ing a generator and discriminator, has allowed for waveforms to
be generated in the vocoders at a considerably higher real-time
speed than in models with autoregressive constraints[16, 17],
and also, waveforms can be generated with reasonable ﬁdelity
and fewer parameters than those required for models based on
bijective ﬂow[18, 19].
Most neural vocoder implementations employ band-limited
mel-spectrograms to generate waveforms[12, 13, 14, 15, 18].
In this situation, the corresponding acoustic information from
the high-frequency band is not provided to the model. If spec-
tral features of up to half of the sampling rate[20] are used as
the input, full-band acoustic information can be provided to the
model for the reconstruction of high-ﬁdelity waveforms. How-
ever, in some models employing full-band mel-spectrograms,
an over-smoothing problem occurred, where non-sharp spectro-
grams have been generated[12].
Herein, we attempted to solve this problem by using a dis-
criminator. In the ﬁeld of voice activity detection (VAD), the
use of multiple features computed at different spectral and tem-
poral resolutions is helpful for improving binary classiﬁcation
performance under various background noises[21]. Consider-
ing this background, if a discriminator operated as a binary clas-
siﬁer employs multiple resolution spectral features as well as
temporal features as the input, we expect that the spectrogram
resolution of the generated waveform would improve.
Therefore, we propose UnivNet1, a neural vocoder that syn-
thesizes high-ﬁdelity waveforms in real time. We added a multi-
resolution spectrogram discriminator (MRSD) that uses multi-
ple linear spectrogram magnitudes computed using various pa-
rameter sets. Using full-band mel-spectrograms as the input,
we expect to generate high-resolution signals over the full-band
using the MRSD. Our overall discriminators, which combine a
multi-period waveform discriminator (MPWD)[13] employing
multiple scales of the waveforms, are expected to support the
vocoder generating ﬁne-grained waveforms by modeling both
spectral and temporal domains.
We devised an evaluation scenario, in which the model was
trained from scratch using a large multi-speaker English dataset.
UnivNet exhibited the best scores among GAN-based vocoders
in all objective and subjective evaluations, regardless of whether
the speaker of the input feature is seen or unseen. For text-to-
speech evaluation, in which the model was ﬁne-tuned with the
predicted features from an acoustic model, UnivNet exhibited
the best subjective evaluation score, which demonstrates that it
can rapidly adapt to a new speaker without having to be trained
from scratch.
2. Related work
MelGAN[11] is one of the ﬁrst vocoders with which mel-
spectrogram inversion using a GAN setup consisting of a non-
autoregressive generator and a multi-scale waveform discrim-
inator has been successfully achieved. In the case of Parallel
WaveGAN[12], both fast model convergence and real-time in-
ference were achieved without a distillation process, via train-
ing of a non-autoregressive WaveNet[22] with the weighted
sum of the adversarial loss term and the multi-resolution short-
time Fourier transform (STFT) loss term. In the case of HiFi-
GAN[13], a multi-period waveform discriminator was pro-
posed that model various periodic patterns of input waveforms;
thereby, signals with a quality surpassing that of autoregressive
or ﬂow-based models have been generated.
1Audio demo samples can be found at the following URL:
https://kallavinka8045.github.io/is2021/
arXiv:2106.07889v1  [eess.AS]  15 Jun 2021

(a) Generator
(b) Discriminators
Figure 1: UnivNet architecture. ‘STFT #m’ denotes the process of computing a linear spectrogram magnitude using the m-th STFT
parameter set. ‘reshape2d(p)’ denotes the process of reshaping a 1-D signal of length T to a 2-D signal of height T/p and width p.
3. Description of the proposed model
3.1. Generator
Our generator G is inspired by MelGAN[11, 23]. A noise se-
quence z is used as the input, and a log-mel-spectrogram c is
used as the condition. The length of z is the same as that of c,
and the length of the output ˆx is the same as that of the target
waveform x owing to transposed convolutions.
To efﬁciently capture the local information of the condition,
location-variable convolution (LVC)[14] was added, which ob-
tained better sound quality and speed while maintaining the
model size. The kernels of the LVC layers are predicted using a
kernel predictor that takes the log-mel-spectrogram as the input.
As shown in Figure 1(a), the kernel predictor is connected to a
residual stack. One kernel predictor simultaneously predicts the
kernels of all LVC layers in one residual stack. Through inter-
nal experiments, we carefully decided on the optimal location
and number of the LVC layers as well as the kernel predictors
in terms of sound quality and generation speed. To improve
the generality of the multi-speaker dataset, a gated activation
unit (GAU)[24] was added to each residual connection. It was
effective in increasing the nonlinearity in WaveNet[16], and it
has also been used in various other studies[12, 14, 18, 19, 22].
3.2. Discriminators
Our discriminators D shown in Figure 1(b) employ multiple
spectrograms and reshaped waveforms computed from real or
generated signals. First, we propose a multi-resolution spec-
trogram discriminator (MRSD). To serve as input for each m-
th sub-discriminator, M real or generated linear spectrogram
magnitudes {sm=|FTm(x)|, ˆsm=|FTm(ˆx)|}M
m=1 were com-
puted from the same waveform using M STFT parameter sets
{FTm(·)}M
m=1, with each including the number of points in
the Fourier transform, frame shift interval, and window length.
There is a conceptual similar work[15] that discriminates one
spectral information by dividing it into several bands using a
ﬁlter bank. The difference is that MRSD employes multiple
spectrograms with various temporal and spectral resolutions, so
we expect to generate high-resolution signals over the full-band.
The model architecture is inspired by a multi-scale waveform
discriminator (MSWD)[11] and consists of strided 2-D convo-
lutions and leaky rectiﬁed linear unit (LReLU)[25].
To improve detailed adversarial modeling in the tempo-
ral domain, we added a multi-period waveform discriminator
(MPWD)[13]. The periodic components of the waveform are
extracted at intervals of a set of prime numbers and used as the
input to each sub-discriminator.
3.3. Training losses
Multi-resolution STFT loss[12] is used as an auxiliary loss to
train the model; it corresponds to the sum of multiple spectro-
gram losses computed using various STFT parameter sets. The
loss Laux, comprising the spectral convergence loss Lsc and log
STFT magnitude loss Lmag, is deﬁned as follows:
Lsc(s,ˆs) = ∥s −ˆs∥F
∥s∥F
, Lmag(s,ˆs) = 1
S ∥log s −log ˆs∥1
(1)
Laux(x, ˆx) = 1
M
M
X
m=1
Ex,ˆx
h
Lsc(sm,ˆsm) + Lmag(sm,ˆsm)
i
(2)
where ∥· ∥F and ∥· ∥1 denote the Frobenius and L1 norms,
respectively, and S denotes the number of elements in the spec-
trogram. Each m-th Lsc and Lmag reuse sm and ˆsm used in the
m-th MRSD sub-discriminator. The number of each loss is M,
which is the same as the number of MRSD sub-discriminators.
We used the objective functions of least-squares GAN[26]
for this study. Our overall objectives are deﬁned as follows:
LG = λLaux(x, G(z, c)) + 1
K
K
X
k=1
Ez,c[(Dk(G(z, c)) −1)2]
(3)
LD = 1
K
K
X
k=1
(Ex[(Dk(x)−1)2]+Ez,c[Dk(G(z, c))2]) (4)

where Dk denotes the k-th sub-discriminators of MRSD and
MPWD, and K denotes the number of all sub-discriminators.
λ represents a parameter that balances adversarial and auxiliary
loss. The sum of the outputs of all sub-discriminators is divided
by K to prevent the balance with auxiliary loss from changing
according to the number of sub-discriminators.
4. Experiments
4.1. Data conﬁgurations
We devised an evaluation scenario that employs a large multi-
speaker English dataset to train the model from scratch. The
LibriTTS[27] dataset is an English multi-speaker audiobook
dataset. The ‘train-clean-360’ subset consisting of 192 hours’
worth of data, 116k utterances, and 904 speakers was used to
train the model and evaluate the speakers used for training (seen
speakers). Here, 5% and 2% of the seen speakers’ utterances
were randomly sampled for validation and testing. The ‘test-
clean’ subset consisting of 9 hours’ worth of data, 5k utter-
ances, and 39 speakers was used to evaluate the speakers not
used for training (unseen speakers). For text-to-speech evalu-
ation, we ﬁne-tuned the vocoders with the predicted log-mel-
spectrograms to reduce feature mismatch.
We prepared the
LJSpeech[28] dataset containing 24 hours’ worth of data and
13k utterances of a single English female speaker.
All waveforms were resampled to a rate of 24 kHz. Using a
1024 point Fourier transform, 256 sample frame shift, and 1024
sample Hann window length, 100-band log-mel-spectrograms
(0–12 kHz) were computed; they were then globally normalized
using the mean and variance of the entire training set.
4.2. Evaluation metrics
We prepared two objective evaluations and one subjective eval-
uation to measure the sound quality of all the models for the
experiments. PESQ[29] (higher is better) is a metric used for
the objective evaluation of speech quality. For wideband PESQ
evaluation, each sample was downsampled to 16 kHz.
The
‘pypesq2’ library was used for the evaluation. In addition, we
deﬁned ‘RMSE’ which compares the root-mean square errors
of the linear spectrogram magnitude between the original and
synthesized sample (lower is better). To show the spectrogram
inversion performance of each model, we used the same val-
ues as those for the STFT parameters employed to compute the
log-mel-spectrograms of the training data.
We prepared a mean opinion score (MOS) subjective evalu-
ation in which raters scored the audio quality from 1 (negative)
to 5 (positive) for each sample. Here, 20 individual speakers’
utterances were prepared for each assessment and evaluated by
20 subjects in the United States located using a crowd-sourced
evaluation via Amazon Mechanical Turk. A total of 400 scores
were collected to compute the mean and 95% conﬁdence inter-
vals for each assessment. All MOS samples were normalized to
a loudness of -23 LUFS.
4.3. Model details
The details of UnivNet as used in the experiments are shown
in Figure 2. We set the dimension of the noise sequence z to
64. Each element of z was sampled from a normal Gaussian
distribution. All layers in the generator have the same chan-
nel size cG except for the last layer, LVC, and the layers in
2https://github.com/vBaiCai/python-pesq
(a) Generator
(b) Discriminators
Figure 2: Details of UnivNet as used in the experiments. The
ﬁrst two numbers in parentheses for each layer indicate the
channel size and kernel size, respectively.
the kernel predictors. We prepared two versions with a chan-
nel size cG of 16 and 32, respectively (e.g., UnivNet-c16 and
UnivNet-c32). The lengths of the kernel predictor outputs Lw
and Lb are dynamic based on the number of residual connec-
tions, input and output channel sizes, and kernel size of LVC.
LReLU with α = 0.2 was used for each activation, and weight
normalization[30] was applied to all layers. Three STFT pa-
rameter sets (number of points in the Fourier transform, sample
length of the frame shift, sample length of the Hann window)
for MRSD and auxiliary loss were selected through internal ex-
periments using the validation set, as follows: (1024, 120, 600),
(2048, 240, 1200), and (512, 50, 240). The balancing param-
eter, λ, was 2.5. The model was trained using the Adam[31]
optimizer with β1 = 0.5, β2 = 0.9, and under a 1e-4 learning
rate. We trained the generator with only auxiliary loss without
discriminators in the ﬁrst 200k steps.
We prepared the following GAN-based vocoders for com-
parison: MelGAN, Parallel WaveGAN, and HiFi-GAN. We
used the ofﬁcial implementations of MelGAN3 and HiFi-GAN
V14 for reproducibility, and we implemented Parallel Wave-
GAN as per the process followed by Yamamoto et al[12]. The
learning rate, optimizer, and all other parameters required for
training followed the reference conﬁgurations of each model.
As noted in section 4.1, all experiments were performed un-
der the full-band conditions (0–12 kHz log-mel-spectrograms at
24 kHz sampling rate), unlike the band-limited conditions (0–8
kHz log-mel-spectrograms at 22.05 kHz sampling rate) used in
the experimental setups in previous works[11, 12, 13].
All models in the experiments were trained to a mini-batch
size of 32 using four NVIDIA V100 GPUs. To ensure stable
sound quality, we trained up to 2M steps for MelGAN and up
to 1M steps for the other models.
For text-to-speech evaluation, we used the JDI-T[2] acous-
tic model with a pitch and energy predictor[3, 4]. We converted
text to phoneme sequences using open-sourced software5. Each
trained vocoder was ﬁne-tuned up to 100k steps using ground
truth waveforms and predicted log-mel-spectrograms.
3https://github.com/descriptinc/melgan-neurips
4https://github.com/jik876/hiﬁ-gan
5https://github.com/bootphon/phonemizer

Table 1: MOS of the ablation study.
Instance
G1
G2
D1
D2
D3
MOS
Recordings
-
-
-
-
-
4.16±0.09
UnivNet-c16
✓
✓
✓
✓
3.92±0.08
Without LVC6
✓
✓
✓
3.73±0.10
Without GAU
✓
✓
✓
3.82±0.09
Without MRSD
✓
✓
✓
3.38±0.11
Without MPWD
✓
✓
✓
3.15±0.11
MPWD→MSWD
✓
✓
✓
✓
3.58±0.10
With MSWD
✓
✓
✓
✓
✓
3.90±0.09
Figure 3: Spectrograms of the generated audio clips.
5. Results
5.1. Ablation study
To demonstrate the validity of the proposed model conﬁgura-
tion, we prepared instances in which each component of the
model was removed. We also prepared some combinations of
discriminators, including MSWD7. UnivNet-c16, a lightweight
version of the model, was used for comparison.
In Table 1, where the MOS of the ablation study are listed,
the components of the generator and discriminators are repre-
sented by the following index symbols for brevity: G1=LVC,
G2=GAU, D1=MRSD, D2=MPWD, and D3=MSWD. A com-
parison of the above four instances and UnivNet-c16 shows that
each component of the proposed model contributes to improv-
ing the performance. In particular, if MRSD is removed, an
6We used local conditioning[16] instead of LVC and changed the
channel sizes of the input layer and 3 residual stacks to 512, 256, 128,
and 64, respectively.
7We used the settings employed by Kumar et al[11].
over-smoothing problem occurs, as shown in Figure 3. This is
most notable in the high-frequency band of the generated wave-
form. The MOS of the instance is signiﬁcantly reduced because
of an audible metallic artifact. Further, the reduced score for the
instance of removing MPWD indicates that the model supports
the generator’s detailed prediction in the temporal domain.
The comparison of the two instances below and UnivNet-
c16 clariﬁes the reason why we chose MPWD instead of
MSWD; the replacement of MPWD with MSWD signiﬁcantly
reduced the MOS. Despite the lower training speed, the instance
of adding MSWD showed little difference from UnivNet-c16.
5.2. Comparison with existing models
We compared the sound quality and model complexity of the
two versions of UnivNet with various vocoders. Each model’s
generation speed was measured on an NVIDIA V100 GPU.
The results of the comparison are summarized in Table 2.
The ﬁrst and second columns represent the sound quality of
the seen and unseen speakers. MelGAN and Parallel Wave-
GAN achieved lower scores than the other models did, and their
scores decreased signiﬁcantly for the unseen speakers. Notably,
the over-smoothing problem occurred in Parallel WaveGAN,
because the experiments involving full-band features were per-
formed, rather than those with band-limited features[12]. HiFi-
GAN and UnivNet showed robustness by maintaining scores for
both seen and unseen speakers. They demonstrated the possibil-
ity of generating high-quality waveforms for arbitrary speakers.
This trend was also maintained in the text-to-speech scenario in
the Text-to-speech column, which demonstrates that the model
can rapidly adapt to a new speaker through ﬁne-tuning.
The overall results demonstrate that UnivNet-c16 has a rel-
atively small number of parameters and a speed that is 200
times higher than real time; further, UnivNet-c16 consistently
maintains high objective and subjective scores. UnivNet-c32
achieves the highest objective scores in all scenarios, with only
a slight decrease in speed. Compared with HiFi-GAN, the sub-
jective scores show little difference, but UnivNet-c32 has the
advantage that it can generate results approximately 1.5 times
faster for a similar number of parameters.
6. Conclusion
In this paper, we proposed UnivNet, a vocoder with a multi-
resolution spectrogram discriminator for high-ﬁdelity and real-
time waveform generation. The discriminator accepts multiple
spectrograms as inputs and improves the spectral resolutions of
the waveforms by alleviating the over-smoothing problem. In
the future, we will research a universal vocoder for use in a
multi-speaker text-to-speech pipeline without ﬁne-tuning.
Table 2: Results of comparison with existing models. ‘Speed’ indicates each model’s generation speed relative to real time.
Seen speakers
Unseen speakers
Text-to-speech
Model complexity
Model
MOS
PESQ
RMSE
MOS
PESQ
RMSE
MOS
Params
Speed
MelGAN
3.56±0.10
2.80
0.576
2.89±0.11
2.74
0.546
2.66±0.12
4.34M
×294.11
Parallel WaveGAN
3.07±0.10
3.60
0.357
2.61±0.11
3.61
0.332
2.85±0.11
1.44M
×66.23
HiFi-GAN
3.89±0.09
3.54
0.423
3.86±0.09
3.55
0.389
3.94±0.08
14.01M
×135.14
UnivNet-c16
3.92±0.08
3.59
0.337
3.79±0.09
3.60
0.313
3.87±0.09
4.00M
×227.27
UnivNet-c32
3.93±0.09
3.70
0.316
3.90±0.09
3.70
0.294
4.07±0.08
14.86M
×204.08
Recordings
4.16±0.09
4.50
0.000
4.30±0.08
4.50
0.000
-
-
-

7. References
[1] J. Shen, R. Pang, R. J. Weiss, M. Schuster, N. Jaitly, Z. Yang,
Z. Chen, Y. Zhang, Y. Wang, R. Skerry-Ryan et al., “Natural TTS
Synthesis by Conditioning WaveNet on Mel Spectrogram Predic-
tions,” in IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP), 2018.
[2] D. Lim, W. Jang, G. O, H. Park, B. Kim, and J. Yoon, “JDI-
T: Jointly trained Duration Informed Transformer for Text-To-
Speech without Explicit Alignment,” in Proceedings of the An-
nual Conference of the International Speech Communication As-
sociation (INTERSPEECH), 2020.
[3] Y. Ren, C. Hu, T. Qin, S. Zhao, Z. Zhao, and T.-Y. Liu, “Fast-
Speech 2: Fast and high-quality end-to-end text-to-speech,” in
International Conference on Learning Representations (ICLR),
2021.
[4] A. La´ncucki, “FastPitch: Parallel Text-to-Speech with Pitch Pre-
diction,” in IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP), 2021.
[5] L.-J. Liu, Z.-H. Ling, Y. Jiang, M. Zhou, and L.-R. Dai, “WaveNet
vocoder with limited training data for voice conversion.” in Inter-
speech, 2018, pp. 1983–1987.
[6] Y. Jia, R. J. Weiss, F. Biadsy, W. Macherey, M. Johnson, Z. Chen,
and Y. Wu, “Direct speech-to-speech translation with a sequence-
to-sequence model,” in Proceedings of the Annual Conference
of the International Speech Communication Association (INTER-
SPEECH), 2019.
[7] D. Grifﬁn and J. Lim, “Signal Estimation from Modiﬁed Short-
Time Fourier Transform,” IEEE Transactions on Acoustics,
Speech, and Signal Processing, vol. 32, no. 2, pp. 236–243, 1984.
[8] H. Kawahara, I. Masuda-Katsuse, and A. De Cheveigne, “Re-
structuring speech representations using a pitch-adaptive time-
frequency smoothing and an instantaneous-frequency-based f0
extraction:
Possible role of a repetitive structure in sounds,”
Speech communication, vol. 27, no. 3-4, pp. 187–207, 1999.
[9] M. Morise, F. Yokomori, and K. Ozawa, “WORLD: A Vocoder-
Based High-Quality Speech Synthesis System for Real-Time Ap-
plications,” IEICE TRANSACTIONS on Information and Systems,
vol. 99, no. 7, pp. 1877–1884, 2016.
[10] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-
Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative Adver-
sarial Nets,” in Advances in Neural Information Processing Sys-
tems (NeurIPS), 2014.
[11] K. Kumar, R. Kumar, T. de Boissiere, L. Gestin, W. Z. Teoh,
J. Sotelo, A. de Br´ebisson, Y. Bengio, and A. Courville, “Mel-
GAN: Generative Adversarial Networks for Conditional Wave-
form Synthesis,” in Advances in Neural Information Processing
Systems (NeurIPS), 2019.
[12] R. Yamamoto, E. Song, and J.-M. Kim, “Parallel WaveGAN: A
Fast Waveform Generation Model based on Generative Adversar-
ial Networks with Multi-Resolution Spectrogram,” in IEEE Inter-
national Conference on Acoustics, Speech and Signal Processing
(ICASSP), 2020.
[13] J. Kong, J. Kim, and J. Bae, “HiFi-GAN: Generative Adversarial
Networks for Efﬁcient and High Fidelity Speech Synthesis,” in
Advances in Neural Information Processing Systems (NeurIPS),
2020.
[14] Z. Zeng, J. Wang, N. Cheng, and J. Xiao, “LVCNet: Efﬁcient
Condition-Dependent Modeling Network for Waveform Genera-
tion,” in IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), 2021.
[15] A. Mustafa, N. Pia, and G. Fuchs, “StyleMelGAN: An Efﬁ-
cient High-Fidelity Adversarial Vocoder with Temporal Adaptive
Normalization,” in IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), 2021.
[16] A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals,
A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu,
“WaveNet: A Generative Model for Raw Audio,” in 9th ISCA
Speech Synthesis Workshop, 2016.
[17] N.
Kalchbrenner,
E.
Elsen,
K.
Simonyan,
S.
Noury,
N. Casagrande, E. Lockhart, F. Stimberg, A. van den Oord,
S. Dieleman, and K. Kavukcuoglu, “Efﬁcient Neural Audio
Synthesis,” in International Conference on Machine Learning
(ICML), 2018.
[18] R. Prenger, R. Valle, and B. Catanzaro, “WaveGlow: A Flow-
based Generative Network for Speech Synthesis,” in IEEE Inter-
national Conference on Acoustics, Speech and Signal Processing
(ICASSP), 2019.
[19] W. Ping, K. Peng, K. Zhao, and Z. Song, “WaveFlow: A Compact
Flow-based Model for Raw Audio,” in International Conference
on Machine Learning (ICML), 2020.
[20] C. E. Shannon, “Communication in the Presence of Noise,” Pro-
ceedings of the IRE, vol. 37, no. 1, pp. 10–21, 1949.
[21] X.-L. Zhang and D. Wang, “Boosting Contextual Information
for Deep Neural Network Based Voice Activity Detection,”
IEEE/ACM Transactions on Audio, Speech, and Language Pro-
cessing, vol. 24, no. 2, pp. 252–264, 2015.
[22] W. Ping, K. Peng, and J. Chen, “ClariNet: Parallel Wave Genera-
tion in End-to-End Text-to-Speech,” in International Conference
on Learning Representations (ICLR), 2018.
[23] G. Yang, S. Yang, K. Liu, P. Fang, W. Chen, and L. Xie, “Multi-
band MelGAN: Faster Waveform Generation for High-Quality
Text-to-Speech,” in IEEE Spoken Language Technology Work-
shop (SLT), 2021.
[24] A. Van den Oord, N. Kalchbrenner, L. Espeholt, O. Vinyals,
A. Graves et al., “Conditional Image Generation with PixelCNN
Decoders,” in Advances in Neural Information Processing Sys-
tems (NeurIPS), 2016.
[25] A. L. Maas, A. Y. Hannun, and A. Y. Ng, “Rectiﬁer Nonlineari-
ties Improve Neural Network Acoustic Models,” in International
Conference on Machine Learning (ICML), 2013.
[26] X. Mao, Q. Li, H. Xie, R. Y. Lau, Z. Wang, and S. Paul Smolley,
“Least Squares Generative Adversarial Networks,” in Proceedings
of the IEEE international conference on computer vision (ICCV),
2017.
[27] H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen,
and Y. Wu, “LibriTTS: A Corpus Derived from LibriSpeech for
Text-to-Speech,” in Proceedings of the Annual Conference of
the International Speech Communication Association (INTER-
SPEECH), 2019.
[28] K. Ito et al., “The LJ speech dataset,” https://keithito.com/LJ-
Speech-Dataset, 2017.
[29] A. W. Rix, J. G. Beerends, M. P. Hollier, and A. P. Hekstra, “Per-
ceptual Evaluation of Speech Quality (PESQ)-A New Method for
Speech Quality Assessment of Telephone Networks and Codecs,”
in IEEE International Conference on Acoustics, Speech and Sig-
nal Processing (ICASSP), 2001.
[30] T. Salimans and D. P. Kingma, “Weight Normalization: A Simple
Reparameterization to Accelerate Training of Deep Neural Net-
works,” in Advances in Neural Information Processing Systems
(NeurIPS), 2016.
[31] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Opti-
mization,” in International Conference on Learning Representa-
tions (ICLR), 2015.
