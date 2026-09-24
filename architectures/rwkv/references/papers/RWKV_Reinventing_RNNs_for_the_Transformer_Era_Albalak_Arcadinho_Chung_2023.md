# RWKV Reinventing RNNs for the Transformer Era Albalak Arcadinho Chung 2023

> Source: `RWKV_Reinventing_RNNs_for_the_Transformer_Era_Albalak_Arcadinho_Chung_2023.pdf`

---

                                                                   RWKV: Reinventing RNNs for the Transformer Era
                             Bo Peng1∗ Eric Alcaide2,3,4∗ Quentin Anthony2,5∗
                                                                       2,6
            Alon Albalak    Samuel Arcadinho2,7 Huanqi Cao8 Xin Cheng9 Michael Chung10
        Matteo Grella11 Kranthi Kiran GV12 Xuzheng He2 Haowen Hou13 Przemysław Kazienko14
       Jan Kocoń14 Jiaming Kong15 Bartłomiej Koptyra14 Hayden Lau2 Krishna Sri Ipsit Mantri16
Ferdinand Mom17,18 Atsushi Saito2,19 Xiangru Tang20 Bolun Wang27 Johan S. Wind21 Stanisław Woźniak14
     Ruichong Zhang8 Zhenyuan Zhang2 Qihang Zhao22,23 Peng Zhou27 Jian Zhu24 Rui-Jie Zhu25,26
                                         1
                                          RWKV Foundation 2 EleutherAI 3 University of Barcelona 4 Charm Therapeutics 5 Ohio State University
                                               6
                                                 University of California, Santa Barbara 7 Zendesk 8 Tsinghua University 9 Peking University
                                                   10
                                                      Storyteller.io 11 Crisis24 12 New York University 13 National University of Singapore
                                         14
                                            Wroclaw University of Science and Technology 15 Databaker Technology Co. Ltd 16 Purdue University
                                                   17
                                                      Criteo AI Lab 18 Epita 19 Nextremer Co. Ltd. 20 Yale University 21 University of Oslo
                                                     22
                                                        University of Science and Technology of China 23 Kuaishou Technology Co. Ltd




arXiv:2305.13048v1 [cs.CL] 22 May 2023
                                                           24
                                                              University of British Columbia 25 University of California, Santa Cruz
                                                         26
                                                            University of Electronic Science and Technology of China 27 RuoxinTech

                                                                         Abstract                              role in various scientific and industrial applica-
                                                                                                               tions. These applications often involve complex
                                                     Transformers have revolutionized almost all               sequential data processing tasks that include nat-
                                                     natural language processing (NLP) tasks but               ural language understanding, conversational AI,
                                                     suffer from memory and computational com-                 time-series analysis, and even indirect modalities
                                                     plexity that scales quadratically with sequence
                                                                                                               that can be reframed as sequences, such as im-
                                                     length. In contrast, recurrent neural networks
                                                     (RNNs) exhibit linear scaling in memory and               ages and graphs (Brown et al., 2020; Ismail Fawaz
                                                     computational requirements but struggle to                et al., 2019; Wu et al., 2020; Albalak et al., 2022).
                                                     match the same performance as Transform-                  Predominant among these techniques are RNNs,
                                                     ers due to limitations in parallelization and             convolutional neural networks (CNNs), and the
                                                     scalability. We propose a novel model ar-                 Transformer models (Vaswani et al., 2017).
                                                     chitecture, Receptance Weighted Key Value
                                                     (RWKV), that combines the efficient paral-                   Each of these has distinct drawbacks that restrict
                                                     lelizable training of Transformers with the effi-         their efficiency in certain scenarios. RNNs suf-
                                                     cient inference of RNNs. Our approach lever-              fer from the vanishing gradient problem, making
                                                     ages a linear attention mechanism and allows              them difficult to train for long sequences. Addition-
                                                     us to formulate the model as either a Trans-              ally, they cannot be parallelized in the time dimen-
                                                     former or an RNN, which parallelizes compu-               sion during training, which restricts their scalability
                                                     tations during training and maintains constant
                                                     computational and memory complexity during
                                                                                                               (Hochreiter, 1998; Le and Zuidema, 2016). CNNs,
                                                     inference, leading to the first non-transformer           on the other hand, are only adept at capturing local
                                                     architecture to be scaled to tens of billions             patterns, which limits their capacity to deal with
                                                     of parameters. Our experiments reveal that                long-range dependencies, crucial to many sequence
                                                     RWKV performs on par with similarly sized                 processing tasks (Bai et al., 2018).
                                                     Transformers, suggesting that future work can
                                                     leverage this architecture to create more effi-              Transformer models emerged as a powerful alter-
                                                     cient models. This work presents a signifi-               native due to their ability to handle both local and
                                                     cant step towards reconciling the trade-offs be-          long-range dependencies and their capability for
                                                     tween computational efficiency and model per-             parallelized training (Tay et al., 2022). Recent mod-
                                                     formance in sequence processing tasks.1                   els such as GPT-3 (Brown et al., 2020), ChatGPT
                                                                                                               (OpenAI, 2022; Kocoń et al., 2023), GPT-4 (Ope-
                                             1       Introduction                                              nAI, 2023), LLaMA (Touvron et al., 2023), and
                                                                                                               Chinchilla (Hoffmann et al., 2022) exemplify the
                                             Deep learning techniques have made significant                    capability of this architecture, pushing the frontiers
                                             strides in artificial intelligence, playing a pivotal             of what’s possible in NLP. Despite these signifi-
                                                     ∗
                                                       Equal first authorship. Others listed alphabetically.   cant advancements, the self-attention mechanism
                                                 1
                                                     Code at: https://github.com/BlinkDL/RWKV-LM               inherent to Transformers poses unique challenges,
Model                  Time               Space            work architectures. It offers a promising and viable
Transformer                2
                      O(T d)               2
                                     O(T + T d)            solution for handling tasks involving large-scale
Reformer            O(T log T d)   O(T log T + T d)        models with billions of parameters, exhibiting com-
Linear Transformers   O(T d2 )       O(T d + d2 )          petitive performance at a fraction of the computa-
Performer           O(T d log d) O(T d log d + d2 log d)
                         2

AFT-full              O(T 2 d)           O(T d)            tional cost. Our experimental results suggest that
MEGA                  O(cT d)           O(cT d)            RWKV could be a valuable tool for addressing the
RWKV (ours)           O(Td)               O(d)             ongoing challenges in scaling and deploying AI
                                                           models across various domains, particularly those
Table 1: Complexity comparison with different Trans-       involving sequential data processing. Thus, RWKV
formers: Reformer (Kitaev et al., 2020), Linear Trans-
                                                           paves the way for the next generation of more sus-
former (Katharopoulos et al., 2020), Performer (Choro-
manski et al., 2020), AFT (Zhai et al., 2021), MEGA        tainable and computationally efficient AI models
(Ma et al., 2023). Here T denotes the sequence length,     for sequence processing tasks.
d the feature dimension, and c is MEGA’s chunk size           Our contributions in this paper are as follows:
of quadratic attention.
                                                               • We introduce the RWKV network archi-
primarily due to its quadratic complexity. This com-             tecture, which combines the advantages of
plexity renders the architecture computationally ex-             RNNs and Transformers while mitigating
pensive and memory-intensive for tasks involving                 their known limitations.
long input sequences or in resource-constrained sit-           • We propose a new attention mechanism re-
uations. These limitations have spurred a wealth of              formulation that results in linear attention, es-
research aiming to improve the scaling properties                chewing the quadratic complexity associated
of Transformers, often at the expense of some of                 with standard Transformer models.
the properties that make it so effective (Wang et al.,         • We conduct a comprehensive series of experi-
2020; Zaheer et al., 2020; Dao et al., 2022a).                   ments on benchmark datasets to showcase the
   To tackle these challenges, we introduce the Re-              performance, efficiency and scaling of RWKV
ceptance Weighted Key Value (RWKV) model, a                      in managing tasks involving large-scale mod-
novel architecture that effectively combines the                 els and long-range dependencies.
strengths of RNNs and Transformers while cir-                  • We release pretrained model ranging in size
cumventing key drawbacks. RWKV is carefully                      from 169 million to 14 billion parameters
designed to alleviate the memory bottleneck and                  trained on the Pile (Gao et al., 2020).2
quadratic scaling associated with Transformers             2       Related Work
(Katharopoulos et al., 2020) with a more efficient
                                                           Recently, a number of techniques have been pro-
linear scaling, while still preserving the rich, ex-
                                                           posed to address the limitations of transformers.
pressive properties that make the Transformer a
dominant architecture in the field.                        Optimizing Attention Mechanism Many trans-
   One of the defining characteristics of RWKV             former variants (“x-formers”) have been introduced
is its ability to offer parallelized training and ro-      to reduce the complexity of transformers (Tay et al.,
bust scalability, similar to Transformers. More-           2022), including sparse attention (Beltagy et al.,
over, we have reformulated the attention mecha-            2020; Kitaev et al., 2020; Guo et al., 2022), ap-
nism in RWKV to introduce a variant of linear              proximating the full attention matrix (Wang et al.,
attention, eschewing the traditional dot-product to-       2020; Ma et al., 2021; Choromanski et al., 2020),
ken interaction in favor of more effective channel-        combining chunked attention with gating (Ma et al.,
directed attention. This approach contrasts signifi-       2023) and other efficient methods (Katharopoulos
cantly with the traditional Transformer architecture,      et al., 2020; Jaegle et al., 2021).
where specific token interactions predominantly               Some recent works like FlashAttention (Dao
drive attention. The implementation of linear atten-       et al., 2022a) and others (Rabe and Staats, 2022;
tion in RWKV is carried out without approxima-             Jang et al., 2019) share similarities with RWKV’s
tion, which offers a considerable improvement in           chunked computation scheme. Despite being
efficiency and enhances the scalability, see Table 1.      memory-efficient, their time complexity remains
   The overarching motivation behind developing            quadratic or contains chunk size as a hidden fac-
RWKV is to bridge the gap between computational            tor. In contrast, RWKV achieves better space and
                                                               2
efficiency and expressive capacity in neural net-                  https://huggingface.co/RWKV
time complexity during inference by formulating a         similarly):
linear attention as an RNN.
                                                                   ft = σg (Wf xt + Uf ht−1 + bf ),      (1)
                                                                   it = σg (Wi xt + Ui ht−1 + bi ),      (2)
Attention Free Models Another line of research                     ot = σg (Wo xt + Uo ht−1 + bo ),      (3)
replaces the attention mechanism with other mod-                   c̃t = σc (Wc xt + Uc ht−1 + bc ),     (4)
ules to scale to long sequences. MLP-Mixer and
                                                                   ct = ft   ct−1 + it    c̃t ,          (5)
others (Tolstikhin et al., 2021; Liu et al., 2021)
proposed the replacement of attention by Multi-                    ht = ot   σh (ct ).                   (6)
Layer Perceptrons (MLPs) in computer vision tasks.        The data flow of RNNs is shown in Fig. 1a. Al-
The Attention Free Transformer (AFT) (Zhai et al.,        though RNNs can be factored into two linear blocks
2021) replaces dot-product self-attention with a          (W and U ) and an RNN-specific block (1)–(6), as
computationally efficient alternative which can be        noted by Bradbury et al. (2017), the data depen-
seen as a multi-head attention where each feature         dency relying on previous time steps prohibits par-
dimension corresponds to a head. Inspired by AFT,         allelizing these typical RNNs.
RWKV takes a similar approach but modifies the
interaction weights for simplicity such that it can       3.2   Transformers and AFT
be transformed into an RNN. In parallel, RNN-             Introduced by Vaswani et al. (2017), Transformers
style (Hochreiter and Schmidhuber, 1997; Chung            are a class of neural networks that have become
et al., 2014) recursive components have also been         the dominant architecture for several NLP tasks.
modified to increase context length, such as the Re-      Instead of operating on sequences step-by-step like
current Memory Transformer (Bulatov et al., 2022,         RNNs, Transformers rely on attention mechanisms
2023) and Linear Recurrent Units (Orvieto et al.,         to capture relationships between all input and all
2023). State space models (SSM) like S4 (Gu et al.,       output tokens:
2022) and its variants (Dao et al., 2022b; Poli et al.,
2023) are also proposed.                                        Attn(Q, K, V ) = softmax(QK > )V,        (7)

   Notably, Quasi-Recurrent neural network                where the multi-headness and scaling factor √1d is
                                                                                                        k
(QRNN) (Bradbury et al., 2017) uses both con-             omitted for convenience. The core QK > multipli-
volutional layers and recurrent pooling functions         cation is an ensemble of pairwise attention scores
across timesteps and channels. While QRNN                 between each token in a sequence, which can be
utilizes convolutional filters with fixed sizes,          decomposed as vector operations:
RWKV employs a time-mixing module as an                                              PT q> ki
attention mechanism with time-decaying factors.                                             e t vi
                                                                  Attn(Q, K, V )t = Pi=1             .   (8)
                                                                                         T    qt> ki
Different from the element-wise pooling in QRNN,                                         i=1 e
RWKV includes a parametrized channel-mixing                  In AFT (Zhai et al., 2021), this is alternately
module (see the green blocks in Fig.1c) that is           formulated as
parallelizable.                                                                    Pt
                                                                   +                     ewt,i +ki vi
                                                               Attn (W, K, V )t = Pi=1t     wt,i +ki
                                                                                                      ,  (9)
                                                                                      i=1 e
3     Background                                          where {wt,i } ∈ RT ×T is the learned pair-wise po-
                                                          sition biases, and each wt,i is a scalar.
Here we briefly review the fundamentals of RNNs              Inspired by AFT, we let each wt,i in RWKV be
and Transformers.                                         a channel-wise time decay vector multiplied by the
                                                          relative position, traced backwards from current
                                                          time as it decays:
3.1    Recurrent Neural Networks (RNNs)
                                                                          wt,i = −(t − i)w,             (10)

Popular RNN architectures such as LSTM (Hochre-           where w ∈ (R≥0 )d , with d the number of chan-
iter and Schmidhuber, 1997) and GRU (Chung                nels. We require w to be non-negative to ensure
et al., 2014) are characterized by the following for-     that ewt,i ≤ 1 and the per-channel weights decay
mulation (shown for LSTM, others can be reasoned          backwards in time.
         Linear                         Convolution                                    Time-mixing
                                       Elementwise
RNN Cell/Linear                                                                   Channel-mixing
                                           Pooling
         Linear                         Convolution                                    Time-mixing
                                       Elementwise
RNN Cell/Linear                                                                  Channel-mixing
                                           Pooling

                  (a) RNN              (b) QuasiRNN (Bradbury et al., 2017)                          (c) RWKV

Figure 1: Computation structure of the RWKV in comparison to QRNN and RNN (Vanilla, LSTM, GRU, etc)
architectures. Color codes: orange indicates time-mixing, convolutions or matrix multiplications, and the contin-
uous block indicates that these computations can proceed simultaneously; blue signifies parameterless functions
that operate concurrently along the channel or feature dimension (element-wise). Green indicates channel-mixing.

4     The Receptance Weighted Key Value
      (RWKV) Model

The RWKV architecture derives its name from
the four primary model elements used in the time-
mixing and channel-mixing blocks:

    • R: Receptance vector acting as the accep-
      tance of past information.
    • W : Weight is the positional weight decay
      vector. A trainable model parameter.
    • K: Key is a vector analogous to K in tradi-
      tional attention.
    • V : Value is a vector analogous to V in tradi-
      tional attention.

   Interactions between the main elements for every
timestep are multiplicative, as illustrated in Fig. 2
                                                          Figure 2: RWKV block elements (left) and RWKV
4.1    High-Level Summary                                 residual block with a final head for language modeling
The RWKV architecture is comprised of a series            (right) architectures.
of stacked residual blocks, each formed by a time-                name                          is                    Bob
mixing and a channel-mixing sub-blocks with re-
current structures.                                             LM Head                      LM Head                LM Head

   The recurrence is formulated both as a linear in-
terpolation between the current input and the input
at the previous time step (a technique we refer to
as time-shift mixing or token shift, indicated by the          Channel Mix                 Channel Mix             Channel Mix
                                                                                  n                           n
diagonal lines in Fig. 3), which can be adjusted in-                         Toke                        Toke
                                                               Layer Norm      shift        Layer Norm     shift   Layer Norm
dependently for every linear projection of the input
embedding (e.g., R, K, V in time-mixing, and R,
K in channel-mixing), and as the time-dependent
                                                                             States                      States
update of the W KV which is formalized in equa-                 Time Mix                     Time Mix               Time Mix
tion 14. The W KV computation is similar to AFT                              Toke
                                                                                  n
                                                                                                         Toke
                                                                                                              n
                                                               Layer Norm      shift        Layer Norm     shift   Layer Norm
(Zhai et al., 2021), but W is now a channel-wise
vector multiplied by relative position rather than a           Layer Norm                   Layer Norm             Layer Norm
pairwise matrix in AFT. We also introduce a vector
                                                                   My                         name                     is
U for separately attending to the current token in
order to compensate for potential degeneration of         Figure 3: RWKV architecture for language modelling.
W (see Appendix G for more details).
  The time-mixing block is given by:                         Additionally, token shift is implemented as a sim-
                                                             ple offset in the temporal dimension at each block
       rt = Wr · (µr xt + (1 − µr )xt−1 ),          (11)     using PyTorch (Paszke et al., 2019) library as
       kt = Wk · (µk xt + (1 − µk )xt−1 ),          (12)     nn.ZeroPad2d((0,0,1,-1)).
   vt = Wv · (µv xt + (1 − µv )xt−1 ),      (13)
        Pt−1 −(t−1−i)w+ki                                    4.3      RNN-like Sequential Decoding
          i=1 e             vi + eu+kt vt
 wkvt = P   t−1 −(t−1−i)w+ki              , (14)             It is common in recurrent networks to use output
            i=1 e              + eu+kt                       at state t as input at state t + 1. This is especially
       ot = Wo · (σ(rt )    wkvt ),                 (15)     evident in the autoregressive decoding inference
                                                             of a language model, requiring each token to be
where the W KV computation, wkvt , plays the                 computed before fed into the next step, making it
role of Attn(Q, K, V ) in Transformers without in-           possible for RWKV to take advantage of its RNN-
curring a quadratic cost as interactions are between         like structure, referred to as time-sequential mode.
scalars. Intuitively, as time t increases, the vector        In such circumstances, RWKV can be conveniently
ot is dependent on a long history, represented by the        formulated recursively for decoding during infer-
summation of an increasing number of terms. For              ence, as shown in Appendix B, which leverages
the target position t, RWKV performs a weighted              the advantage that each output token is dependent
summation in the positional interval of [1, t], and          only on the latest state, which is of constant size,
then multiplies with the receptance σ(r). There-             irrespective of the sequence length.
fore, interactions are multiplicative inside a given            It then behaves as an RNN decoder, yielding
timestep and summed over different timesteps.                constant speed and memory footprint with respect
   Further, the channel-mixing block is given by:            to the sequence length, enabling the processing of
                                                             longer sequences more efficiently. In contrast, self-
        rt = Wr · (µr xt + (1 − µr )xt−1 ),         (16)
                                                             attention typically requires a KV cache growing
       kt = Wk · (µk xt + (1 − µk )xt−1 ),          (17)     linearly with respect to the sequence length, result-
                                          2
        ot = σ(rt )   (Wv · max(kt , 0) ),          (18)     ing in degraded efficiency and increasing memory
                                                             footprint and time as the sequence grows longer.
where we adopt squared ReLU activation (So et al.,
2021). Note that in both time-mixing and channel-            4.4      Software Implementation
mixing, by taking the sigmoid of the receptance,             RWKV is originally implemented using the Py-
we’re intuitively using it as a “forget gate” to elimi-      torch Deep Learning Library (Paszke et al., 2019)
nate unnecessary historical information.                     and a custom CUDA kernel for the W KV com-
4.2     Transformer-like Parallelization                     putation explained in 4.7. Although RWKV is a
                                                             general recurrent network, its current implemen-
RWKV can be efficiently parallelized in what we              tation focuses in the task of language modeling
call a time-parallel mode, reminiscent of Trans-             (RWKV-LM). The model architecture is comprised
formers. The time complexity of processing a                 of an embedding layer, for which we follow the
batch of sequences in a single layer is O(BT d2 ),           setup described in Section 4.7 and several identical
which mainly consists of matrix multiplications              residual blocks applied sequentially as seen in Fig.
W ,  ∈ {r, k, v, o} (assuming B sequences, T               2 and 3 following the principles outlined in Section
maximum tokens and d channels). Meanwhile, up-               4.6. After the last block, a simple output projec-
dating attention scores wkvt requires a serial scan          tion head composed by a LayerNorm (Ba et al.,
(see Appendix B for more detail) and has complex-            2016) and a linear projection is used to obtain the
ity O(BT d).                                                 logits to be used in the next-token prediction task
   The matrix multiplications can be parallelized            and calculate the cross entropy loss during training.
akin to W  ,  ∈ {Q, K, V, O} in typical Trans-             Both the embeddings generated after the last resid-
formers. The element-wise W KV computation                   ual block and the logits could also be used later
is time-dependent, but can be readily parallelized           for downstream NLP tasks. Training is performed
along the other two dimensions (Lei et al., 2018)3 .         in time-parallel mode (Section 4.2) while autore-
   3
    If the sequence is very long, more sophisticated meth-   gressive inference and a potential chat interface4
ods such as Martin and Cundy (2017) that parallelize over
                                                                4
sequence length could be used.                                      https://github.com/BlinkDL/ChatRWKV
leverage the time-sequential mode (Section 4.3).        over time, the model preserves a sense of temporal
                                                        locality and progression, which is essential for se-
4.5   Gradient Stability and Layer Stacking             quential processing. This treatment of positional
The RWKV architecture has been designed as a            information in sequential data exhibits similarities
fusion of both Transformers and RNNs, offering          to the Attention with Linear Biases (ALiBi) model
the advantage of stable gradients and deeper archi-     (Press et al., 2022), where the linear biases facili-
tectures of Transformers compared to traditional        tate input length extrapolation. In this context, the
RNNs while being efficient in inference.                RWKV architecture can be perceived as a trainable
   Previous work has sought to tackle the prob-         version of ALiBi, seamlessly incorporating posi-
lem of gradient stability in RNNs with a variety of     tional information without the necessity for explicit
techniques including using non-saturated activation     encoding. It can also be seen as an extension of the
functions (Chandar et al., 2019), gating mechanism      gated convolution introduced in Zhai et al. (2021)
(Gu et al., 2019), gradient clipping (Pascanu et al.,   to the full sequence length until a given step.
2012), and adding constraints (Kanai et al., 2017;         The token shift or time-shift mixing, or (diag-
Miller and Hardt, 2018). While these techniques         onal arrows in Figure 3), also contributes to the
have seen little success, RWKV avoids the problem       model’s adaptation to sequential data. By linearly
inherently by utilizing softmax in conjunction with     interpolating between the current input and the pre-
RNN-style updates.                                      vious time step input, the model naturally aggre-
   The RWKV model features a single-step pro-           gates and gates information in the input channels.
cess for updating attention-like scores, which in-      The overall structure of time-shift mixing bears
cludes a time-dependent softmax operation that          resemblance to the causal convolution with no dila-
helps numerical stability and guards against van-       tions in WaveNet (van den Oord et al., 2016), which
ishing gradients (for rigorous proof, see Appendix      is a classical architecture used for forecasting time
F). Intuitively, this operation ensures the gradient    series data.
is propagated along the most relevant path. Layer
normalization (Ba et al., 2016) is another key as-      4.7   Additional Optimizations
pect of the architecture which enhances the training    Custom Kernels To address inefficiencies in the
dynamics of deep neural networks by stabilizing         W KV computation due to the sequential nature of
gradients, addressing both vanishing and exploding      the task when using standard deep learning frame-
gradient issues.                                        works, we implement a custom CUDA kernel so
   These design elements not only contribute to the     as to launch a single compute kernel in training ac-
RWKV architecture’s stability and learning capa-        celerators. All other parts of the model are matrix
bilities but enable the stacking of multiple layers     multiplications and point-wise operations that can
in a manner that surpasses the capabilities of any      already be efficiently parallelized.
existing RNN. In doing so, the model is able to cap-
                                                        FFN with R gate Prior research (Tolstikhin et al.,
ture more complex patterns across various levels of
                                                        2021; Liu et al., 2021; Yu et al., 2022) sug-
abstraction (see also Appendix G).
                                                        gests that self-attention may not be as essential
4.6   Harnessing Temporal Structure for                 in Transformer-based vision tasks as previously
      Sequential Data Processing                        thought. Although it provided us with some in-
                                                        sights, replacing self-attention entirely in natural
RWKV captures and propagates sequential infor-
                                                        language tasks could be too drastic. In our study,
mation through the combination of three mecha-
                                                        we partially dismantle the attention mechanism by
nisms: recurrence, time decay and token shift.
                                                        replacing the fixed QKV formula with KV and in-
   The recurrence in the time-mixing block of
                                                        troducing a new time-decaying factor W . This ap-
RWKV is the basis for the model’s capacity to
                                                        proach enables us to incorporate token and channel-
capture intricate relationships between sequence
                                                        mixing components akin to MLP-mixer (Tolstikhin
elements and to propagate locality information
                                                        et al., 2021) and a gating unit R similar to gMLP
through time.
                                                        (Liu et al., 2021), which enhance the performance
   The time decay mechanism (e−w and eu in equa-
                                                        of our RWKV model.
tion 14), maintains sensitivity to the positional re-
lationship between sequence elements. By gradu-         Small Init Embedding During the initial stage
ally diminishing the influence of past information      of training a transformer model (Vaswani et al.,
2017), we observe that the embedding matrix un-        and BLOOM (Scao et al., 2022). RWKV even out-
dergoes slow changes, which pose a challenge for       performs Pythia and GPT-Neo (Black et al., 2022)
the model to deviate from its initial noisy embed-     in four tasks: PIQA, OBQA, ARC-E, and COPA
ding state. To mitigate this issue, we propose an      (See details in Appendix H). For RQ3, Fig. 5 shows
approach that involves initializing the embedding      that increasing context length leads to lower test
matrix with small values and subsequently apply-       loss on the Pile, an indication that RWKV can make
ing an additional LayerNorm operation. By imple-       effective use of long contextual information.
menting this technique, we accelerate and stabilize
the training process, enabling the training of deep    6    Inference Experiments
architectures with post-LN components. The effec-
                                                       We benchmark inference requirements according
tiveness of this approach is demonstrated in Figure
                                                       to size and family. Specifically, we evaluate text
8, where it is shown to facilitate improved conver-
                                                       generation speed and memory requirements on a
gence by allowing the model to quickly transition
                                                       typical compute platforms including CPU (x86)
away from the initially small embedding. This is
                                                       and GPU (NVIDIA A100 80GB). For all our ex-
achieved through small changes following a single
                                                       periments we use float32 precision. We include
step, which in turn lead to substantial alterations
                                                       all model parameters in parameter count, including
in directions and subsequently significant changes
                                                       both embedding and non-embedding layers. Per-
after the LayerNorm operation.
                                                       formance under different quantization setups is left
Custom Initialization Building on principles           to further work. See Appendix I for more results.
from previous works (He et al., 2016; Jumper et al.,
2021), we initialize parameters to values as similar
as possible to an identity mapping while break-
ing symmetry so there is a clean information path.
Most weights are initialized to zero. No biases are
used for linear layers. Specific formulas are given
in Appendix D. We find the choice of initialization
to be significant in convergence speed and quality
(see Appendix E).

5    Evaluations
In this section, we focus on evaluating to answer
the following questions:
                                                       Figure 6: Cumulative time during text generation for
    • RQ1:      Is RWKV competitive against            different LLMs.
      quadratic transformer architectures with equal
      number of parameters and training tokens?
    • RQ2: When increasing the number of param-           Additionally, we carried out comparative studies
      eters, does RWKV remain competitive against      on RWKV-4 and ChatGPT / GPT-4, see Appendix
      quadratic transformer architectures?             J. They revealed that RWKV-4 is very sensitive to
    • RQ3: Does increasing parameters of RWKV          prompt engineering. When the prompts were ad-
      yield better language modeling loss, when        justed from the ones used for GPT to more suitable
      RWKV models are trained for context lengths      for RWKV, the F1-measure performance increased
      that most open-sourced quadratic transform-      even from 44.2% to 74.8%.
      ers cannot efficiently process?
                                                       7    Future Work
  Addressing RQ1 and RQ2, from Fig. 4, we              There are several promising directions for future
can see that RWKV is very competitive on six           work on the RWKV architecture:
benchmarks (Winogrande, PIQA, ARC-C, ARC-E,
LAMBADA, and SciQ) against major open source               • Increasing model expressivity with enhanced
quadratic complexity transformer models: Pythia              time-decay formulations and exploring initial
(Biderman et al., 2023), OPT (Zhang et al., 2022)            model states while maintaining efficiency.
                           (a) Winogrande                          (b) PIQA                     (c) ARC-Challenge




                           (d) ARC-Easy                         (e) LAMBADA                         (f) SciQ

Figure 4: Zero-Shot Performance: The horizontal axis is a number of parameters and the vertical axis is accuracy.


                                                                                ing would be the performance under different
                                                          7B 8k
                                                          14B 8k                datasets and specific use cases.
                 22

Pile test loss
                                                                              • Adapting parameter-efficient fine-tuning
                                                                                methods such as LoRA (Hu et al., 2022)
                 21                                                             and characterizing behavior under different
                                                                                quantization schemes for the proposed
                      21      23     25       27     29   211                   architecture
                                    Context Length
                                                                         8     Conclusions
Figure 5: Increasing context length contributes to lower
test loss on the Pile (Gao et al., 2020).
                                                                        We introduced RWKV, a new approach to RNN
                                                                        models exploiting the potential of time-based mix-
                                                                        ing components. RWKV introduces several key
             • Further improving RWKV computational ef-                 strategies which allow it to capture locality and
               ficiency by applying parallel scan in the                long-range dependencies, while addressing limi-
               wkvt step to reduce the computational cost               tations of current architectures by: (1) replacing
               to O(B log(T )d).                                        the quadratic QK attention by a scalar formulation
             • Investigating the application of RWKV to                 with linear cost, (2) reformulating recurrence and
               encoder-decoder architectures and potential              sequential inductive biases to unlock efficient train-
               replacement of cross-attention mechanism.                ing parallelization and efficient inference, and (3)
               This could have applicability seq2seq or multi-          enhancing training dynamics using custom initial-
               modal settings, enhancing efficiency both in             izations.
               training and inference.                                     We benchmark the proposed architecture in a
             • Leveraging RWKV’s state (or context) for in-             wide variety of NLP tasks and show comparable
               terpretability, predictability in sequence data          performance to SoTA with reduced cost. Further
               and safety. Manipulating the hidden state                experiments on expressivity, interpretability, and
               could also guide behavior and allow greater              scaling showcase the model capabilities and draw
               customizability through prompt tuning.                   parallels in behavior between RWKV and other
             • Exploring fine-tuned models in specific set-             LLMs.
               tings for enhanced interaction with humans                  RWKV opens a new door to scalable and ef-
               (Ouyang et al., 2022). Particularly interest-            ficient architectures to model complex relation-
ships in sequential data. While many alternatives           2022 Conference on Empirical Methods in Natu-
to Transformers have been proposed with similar             ral Language Processing, pages 10936–10953, Abu
                                                            Dhabi, United Arab Emirates. Association for Com-
claims, ours is the first to back up those claims with
                                                            putational Linguistics.
pretrained models with tens of billions of parame-
ters.                                                     Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hin-
                                                             ton. 2016. Layer normalization.
9   Limitations
                                                          Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. 2018.
While our proposed RWKV model has demon-                    An empirical evaluation of generic convolutional
                                                            and recurrent networks for sequence modeling.
strated promising results regarding training and
memory efficiency during inference, some limita-          Francesco Barbieri, Jose Camacho-Collados, Luis Es-
tions should be acknowledged and addressed in               pinosa Anke, and Leonardo Neves. 2020. TweetE-
future work. First, the linear attention of RWKV            val: Unified benchmark and comparative evaluation
leads to significant efficiency gains but still, it may     for tweet classification. In Findings of the Associ-
                                                            ation for Computational Linguistics: EMNLP 2020,
also limit the model’s performance on tasks that            pages 1644–1650, Online. Association for Computa-
require recalling minutiae information over very            tional Linguistics.
long contexts. This is due to the funneling of in-
formation through a single vector representation          Iz Beltagy, Matthew E. Peters, and Arman Cohan.
                                                             2020. Longformer: The long-document transformer.
over many time steps, compared with the full in-
                                                             arXiv:2004.05150.
formation maintained by the quadratic attention of
standard Transformers. In other words, the model’s        Stella Biderman, Hailey Schoelkopf, Quentin An-
recurrent architecture inherently limits its ability to      thony, Herbie Bradley, Kyle O’Brien, Eric Halla-
“look back” at previous tokens, as opposed to tra-           han, Mohammad Aflah Khan, Shivanshu Purohit,
                                                             USVSN Sai Prashanth, Edward Raff, et al. 2023.
ditional self-attention mechanisms. While learned            Pythia: A suite for analyzing large language mod-
time decay helps prevent the loss of information,            els across training and scaling. arXiv preprint
it is mechanistically limited compared to full self-         arXiv:2304.01373.
attention.
                                                          Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jian-
    Another limitation of this work is the increased        feng Gao, and Yejin Choi. 2020. Piqa: Reasoning
importance of prompt engineering in comparison to           about physical commonsense in natural language. In
standard Transformer models. The linear attention           Thirty-Fourth AAAI Conference on Artificial Intelli-
mechanism used in RWKV limits the information               gence.
from the prompt that will be carried over to the
                                                          Sid Black, Leo Gao, Phil Wang, Connor Leahy, and
model’s continuation. As a result, carefully de-             Stella Biderman. 2022. Gpt-neo: Large scale autore-
signed prompts may be even more crucial for the              gressive language modeling with mesh-tensorflow,
model to perform well on tasks.                              2021.      URL: https://doi. org/10.5281/zenodo,
                                                             5297715.
Acknowledgements
                                                          James Bradbury, Stephen Merity, Caiming Xiong, and
                                                            Richard Socher. 2017. Quasi-recurrent neural net-
We acknowledge EleutherAI and StabilityAI for
                                                            works. In ICLR.
compute access and technical support in develop-
ment of RWKV. We also acknowledge the mem-                Tom Brown, Benjamin Mann, Nick Ryder, Melanie
bers of the RWKV Discord server for their help              Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
and work on further extending the applicability of          Neelakantan, Pranav Shyam, Girish Sastry, Amanda
                                                            Askell, et al. 2020. Language models are few-shot
RWKV to different domains. Finally, we thank                learners. Advances in neural information processing
Stella Biderman for feedback on the paper.                  systems, 33:1877–1901.

                                                          Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev.
References                                                  2023. Scaling transformer to 1m tokens and beyond
                                                            with rmt.
Alon Albalak, Yi-Lin Tuan, Pegah Jandaghi, Connor
  Pryor, Luke Yoffe, Deepak Ramachandran, Lise            Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev.
  Getoor, Jay Pujara, and William Yang Wang. 2022.          2022. Recurrent memory transformer. Advances in
  FETA: A benchmark for few-sample task transfer            Neural Information Processing Systems, 35:11079–
  in open-domain dialogue. In Proceedings of the            11091.
A. P. Sarath Chandar, Chinnadhurai Sankar, Eugene        Mandy Guo, Joshua Ainslie, David C Uthus, Santi-
  Vorontsov, Samira Ebrahimi Kahou, and Yoshua            ago Ontanon, Jianmo Ni, Yun-Hsuan Sung, and Yin-
  Bengio. 2019. Towards non-saturating recurrent          fei Yang. 2022. Longt5: Efficient text-to-text trans-
  units for modelling long-term dependencies. In          former for long sequences. In Findings of the Associ-
  AAAI Conference on Artificial Intelligence.             ation for Computational Linguistics: NAACL 2022,
                                                          pages 724–736.
Krzysztof Choromanski, Valerii Likhosherstov, David
  Dohan, Xingyou Song, Andreea Gane, Tamas Sar-          Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian
  los, Peter Hawkins, Jared Davis, Afroz Mohiuddin,        Sun. 2016. Identity mappings in deep residual net-
  Lukasz Kaiser, David Belanger, Lucy Colwell, and         works.
  Adrian Weller. 2020. Rethinking attention with per-
  formers.                                               Dan Hendrycks, Collin Burns, Steven Basart, Andy
                                                           Zou, Mantas Mazeika, Dawn Song, and Jacob Stein-
Junyoung Chung, Caglar Gulcehre, KyungHyun Cho,            hardt. 2021. Measuring massive multitask lan-
  and Yoshua Bengio. 2014. Empirical evaluation of         guage understanding. In International Conference
  gated recurrent neural networks on sequence model-       on Learning Representations.
  ing. In NIPS 2014 Deep Learning and Representa-
  tion Learning Workshop.                                Sepp Hochreiter. 1998. The vanishing gradient prob-
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot,      lem during learning recurrent neural nets and prob-
  Ashish Sabharwal, Carissa Schoenick, and Oyvind          lem solutions. International Journal of Uncer-
  Tafjord. 2018. Think you have solved question an-        tainty, Fuzziness and Knowledge-Based Systems,
  swering? try arc, the ai2 reasoning challenge. In        6(02):107–116.
  arXiv:1803.05457.
                                                         Sepp Hochreiter and Jürgen Schmidhuber. 1997.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Ja-        Long short-term memory. Neural Computation,
  cob Hilton, Reiichiro Nakano, Christopher Hesse,         9(8):1735–1780.
  and John Schulman. 2021. Training verifiers to
  solve math word problems. In arXiv, volume             Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch,
  abs/2110.14168.                                           Elena Buchatskaya, Trevor Cai, Eliza Rutherford,
                                                            Diego de Las Casas, Lisa Anne Hendricks, Johannes
Tri Dao, Daniel Y Fu, Stefano Ermon, Atri Rudra, and       Welbl, Aidan Clark, Tom Hennigan, Eric Noland,
   Christopher Re. 2022a. Flashattention: Fast and          Katie Millican, George van den Driessche, Bogdan
   memory-efficient exact attention with IO-awareness.      Damoc, Aurelia Guy, Simon Osindero, Karen Si-
   In Advances in Neural Information Processing Sys-        monyan, Erich Elsen, Jack W. Rae, Oriol Vinyals,
   tems.                                                    and Laurent Sifre. 2022. Training compute-optimal
                                                            large language models.
Tri Dao, Daniel Y Fu, Khaled K Saab, Armin W
   Thomas, Atri Rudra, and Christopher Ré. 2022b.        Edward J Hu, yelong shen, Phillip Wallis, Zeyuan
   Hungry hungry hippos: Towards language mod-             Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
   eling with state space models. arXiv preprint           Weizhu Chen. 2022. LoRA: Low-rank adaptation of
   arXiv:2212.14052.                                       large language models. In International Conference
                                                           on Learning Representations.
Dorottya Demszky, Dana Movshovitz-Attias, Jeong-
  woo Ko, Alan S. Cowen, Gaurav Nemade, and Su-          Hassan Ismail Fawaz, Germain Forestier, Jonathan We-
  jith Ravi. 2020. Goemotions: A dataset of fine-          ber, Lhassane Idoumghar, and Pierre-Alain Muller.
  grained emotions. In Proceedings of the 58th An-         2019. Deep learning for time series classification:
  nual Meeting of the Association for Computational        a review. Data mining and knowledge discovery,
  Linguistics, ACL 2020, Online, July 5-10, 2020,          33(4):917–963.
  pages 4040–4054. Association for Computational
  Linguistics.                                           Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol
Leo Gao, Stella Biderman, Sid Black, Laurence Gold-        Vinyals, Andrew Zisserman, and Joao Carreira.
  ing, Travis Hoppe, Charles Foster, Jason Phang, Ho-      2021. Perceiver: General perception with iterative
  race He, Anish Thite, Noa Nabeshima, et al. 2020.        attention. In International conference on machine
  The pile: An 800gb dataset of diverse text for lan-      learning, pages 4651–4664. PMLR.
  guage modeling. arXiv preprint arXiv:2101.00027.
                                                         Hanhwi Jang, Joonsung Kim, Jae-Eon Jo, Jaewon Lee,
Albert Gu, Karan Goel, and Christopher Ré. 2022. Ef-       and Jangwoo Kim. 2019. Mnnfast: A fast and
  ficiently modeling long sequences with structured        scalable system architecture for memory-augmented
  state spaces. In The International Conference on         neural networks. In Proceedings of the 46th Interna-
  Learning Representations (ICLR).                         tional Symposium on Computer Architecture, pages
                                                           250–263.
Albert Gu, Çaglar Gülçehre, Tom Le Paine,
  Matthew W. Hoffman, and Razvan Pascanu.                Matt Gardner Johannes Welbl Nelson F. Liu. 2017.
  2019. Improving the gating mechanism of recurrent       Crowdsourcing multiple choice science questions.
  neural networks. ArXiv, abs/1910.09890.                 In DOI:10.18653/v1/W17-4413.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke       Xuezhe Ma, Xiang Kong, Sinong Wang, Chunting
 Zettlemoyer. 2017. Triviaqa: A large scale distantly       Zhou, Jonathan May, Hao Ma, and Luke Zettle-
 supervised challenge dataset for reading comprehen-        moyer. 2021. Luna: Linear unified nested attention.
 sion. In ACL.                                              Advances in Neural Information Processing Systems,
                                                            34:2441–2453.
John Jumper, Richard Evans, Alexander Pritzel,
  Tim Green, Michael Figurnov, Olaf Ronneberger,          Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He,
  Kathryn Tunyasuvunakool, Russ Bates, Augustin             Liangke Gui, Graham Neubig, Jonathan May, and
  Žídek, Anna Potapenko, and et al. 2021. Highly            Luke Zettlemoyer. 2023. Mega: Moving average
  accurate protein structure prediction with alphafold.     equipped gated attention. In ICLR.
  Nature, 596(7873):583–589.
                                                          Eric Martin and Chris Cundy. 2017. Parallelizing
Sekitoshi Kanai, Yasuhiro Fujiwara, and Sotetsu Iwa-         linear recurrent neural nets over sequence length.
  mura. 2017. Preventing gradient explosions in gated       ArXiv, abs/1709.04057.
  recurrent units. In NIPS.                               Kevin Meng, David Bau, Alex Andonian, and Yonatan
                                                            Belinkov. 2022. Locating and editing factual asso-
Jared Kaplan, Sam McCandlish, Tom Henighan,                 ciations in GPT. Advances in Neural Information
   Tom B Brown, Benjamin Chess, Rewon Child, Scott          Processing Systems, 36.
   Gray, Alec Radford, Jeffrey Wu, and Dario Amodei.
   2020. Scaling laws for neural language models.         Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish
   arXiv preprint arXiv:2001.08361.                         Sabharwal. 2018. Can a suit of armor conduct elec-
                                                            tricity? a new dataset for open book question answer-
Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pap-           ing. In EMNLP.
  pas, and François Fleuret. 2020. Transformers are
  rnns: Fast autoregressive transformers with linear      John Miller and Moritz Hardt. 2018. Stable recurrent
  attention. In International Conference on Machine         models. arXiv: Learning.
  Learning, pages 5156–5165. PMLR.
                                                          Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong
Nikita Kitaev, L. Kaiser, and Anselm Levskaya.              He, Devi Parikh, Dhruv Batra, Lucy Vanderwende,
  2020. Reformer: The efficient transformer. ArXiv,         Pushmeet Kohli, and James Allen. 2016. A cor-
  abs/2001.04451.                                           pus and cloze evaluation for deeper understanding of
                                                            commonsense stories. In Proceedings of the 2016
Jan Kocoń, Igor Cichecki, Oliwier Kaszyca, Mateusz         Conference of the North American Chapter of the
   Kochanek, Dominika Szydło, Joanna Baran, Julita          Association for Computational Linguistics: Human
   Bielaniewicz, Marcin Gruza, Arkadiusz Janz, Kamil        Language Technologies, pages 839–849.
   Kanclerz, Anna Kocoń, Bartłomiej Koptyra, Wik-
   toria Mieleszczenko-Kowszewicz, Piotr Miłkowski,       OpenAI. 2022. Introducing chatgpt. https://openai.
   Marcin Oleksy, Maciej Piasecki, Łukasz Radliński,       com/blog/chatgpt.
   Konrad Wojtasik, Stanisław Woźniak, and Prze-
                                                          OpenAI. 2023. Gpt-4 technical report.
   mysław Kazienko. 2023. Chatgpt: Jack of all trades,
   master of none.                                        Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan
                                                            Fernando, Caglar Gulcehre, Razvan Pascanu, and
Jan Kocoń, Piotr Miłkowski, and Monika Zaśko-             Soham De. 2023.       Resurrecting recurrent neu-
   Zielińska. 2019. Multi-level sentiment analysis of      ral networks for long sequences. arXiv preprint
   polemo 2.0: Extended corpus of multi-domain con-         arXiv:2303.06349.
   sumer reviews. In Proceedings of the 23rd Confer-
   ence on Computational Natural Language Learning        Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida,
  (CoNLL), pages 980–991.                                   Carroll L. Wainwright, Pamela Mishkin, Chong
                                                            Zhang, Sandhini Agarwal, Katarina Slama, Alex
Phong Le and Willem Zuidema. 2016. Quantifying              Ray, John Schulman, Jacob Hilton, Fraser Kelton,
  the vanishing gradient and long distance dependency       Luke Miller, Maddie Simens, Amanda Askell, Pe-
  problem in recursive neural networks and recursive        ter Welinder, Paul Christiano, Jan Leike, and Ryan
  lstms. In Proceedings of the 1st Workshop on Repre-       Lowe. 2022. Training language models to follow in-
  sentation Learning for NLP, pages 87–93.                  structions with human feedback.

Tao Lei, Yu Zhang, Sida I. Wang, Hui Dai, and Yoav        Denis Paperno, Germán Kruszewski, Angeliki Lazari-
  Artzi. 2018. Simple recurrent units for highly par-       dou, Ngoc Quan Pham, Raffaella Bernardi, San-
  allelizable recurrence. In Proceedings of the 2018        dro Pezzelle, Marco Baroni, Gemma Boleda, and
  Conference on Empirical Methods in Natural Lan-           Raquel Fernandez. 2016. The LAMBADA dataset:
  guage Processing, pages 4470–4481, Brussels, Bel-         Word prediction requiring a broad discourse context.
  gium. Association for Computational Linguistics.          In Proceedings of the 54th Annual Meeting of the As-
                                                            sociation for Computational Linguistics (Volume 1:
Hanxiao Liu, Zihang Dai, David R. So, and Quoc V. Le.       Long Papers), pages 1525–1534, Berlin, Germany.
  2021. Pay attention to mlps.                              Association for Computational Linguistics.
Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio.            Ilya O. Tolstikhin, Neil Houlsby, Alexander
  2012. On the difficulty of training recurrent neural          Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas
  networks. In International Conference on Machine              Unterthiner, Jessica Yung, Andreas Steiner, Daniel
  Learning.                                                     Keysers, Jakob Uszkoreit, Mario Lucic, and
                                                                Alexey Dosovitskiy. 2021. Mlp-mixer: An all-mlp
Adam Paszke, Sam Gross, Francisco Massa, Adam                   architecture for vision. CoRR, abs/2105.01601.
  Lerer, James Bradbury, Gregory Chanan, Trevor
  Killeen, Zeming Lin, Natalia Gimelshein, Luca              Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
  Antiga, Alban Desmaison, Andreas Köpf, Edward                Martinet, Marie-Anne Lachaux, Timothée Lacroix,
  Yang, Zach DeVito, Martin Raison, Alykhan Tejani,            Baptiste Rozière, Naman Goyal, Eric Hambro,
  Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Jun-           Faisal Azhar, Aurelien Rodriguez, Armand Joulin,
  jie Bai, and Soumith Chintala. 2019. Pytorch: An             Edouard Grave, and Guillaume Lample. 2023.
  imperative style, high-performance deep learning li-         Llama: Open and efficient foundation language mod-
  brary.                                                       els.
Michael Poli, Stefano Massaroli, Eric Nguyen,
  Daniel Y Fu, Tri Dao, Stephen Baccus, Yoshua               Aäron van den Oord, Sander Dieleman, Heiga Zen,
  Bengio, Stefano Ermon, and Christopher Ré. 2023.             Karen Simonyan, Oriol Vinyals, Alex Graves,
  Hyena hierarchy: Towards larger convolutional lan-           Nal Kalchbrenner, Andrew W. Senior, and Koray
  guage models. arXiv preprint arXiv:2302.10866.               Kavukcuoglu. 2016. Wavenet: A generative model
                                                               for raw audio. ArXiv, abs/1609.03499.
Ofir Press, Noah A. Smith, and Mike Lewis. 2022.
  Train short, test long: Attention with linear biases en-   Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
  ables input length extrapolation. In The Tenth Inter-        Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
  national Conference on Learning Representations,             Kaiser, and Illia Polosukhin. 2017. Attention is all
  ICLR 2022, Virtual Event, April 25-29, 2022.                 you need. In Advances in Neural Information Pro-
                                                               cessing Systems, volume 30. Curran Associates, Inc.
Ilan Price, Jordan Gifford-Moore, Jory Flemming, Saul
   Musker, Maayan Roichman, Guillaume Sylvain,
                                                             David Vilares and Carlos Gómez-Rodríguez. 2019.
   Nithum Thain, Lucas Dixon, and Jeffrey Sorensen.
                                                               Head-qa: A healthcare dataset for complex reason-
   2020. Six attributes of unhealthy conversations.
                                                               ing. In ACL.
   In Proceedings of the Fourth Workshop on Online
   Abuse and Harms, pages 114–124, Online. Associa-
   tion for Computational Linguistics.                       Alex Wang, Yada Pruksachatkun, Nikita Nangia,
                                                               Amanpreet Singh, Julian Michael, Felix Hill, Omer
Markus N. Rabe and Charles Staats. 2022.            Self-      Levy, and Samuel Bowman. 2019. Superglue: A
 attention does not need o(n2 ) memory.                        stickier benchmark for general-purpose language un-
                                                               derstanding systems. In Advances in Neural Infor-
Melissa Roemmele, Cosmin Adrian Bejan, , and An-               mation Processing Systems, volume 32. Curran As-
 drew S. Gordon. 2018. Choice of plausible alterna-            sociates, Inc.
 tives: An evaluation of commonsense causal reason-
 ing. In AAAI.                                               Alex Wang, Amanpreet Singh, Julian Michael, Fe-
Teven Le Scao, Angela Fan, Christopher Akiki, El-              lix Hill, Omer Levy, and Samuel Bowman. 2018.
  lie Pavlick, Suzana Ilić, Daniel Hesslow, Ro-               GLUE: A multi-task benchmark and analysis plat-
  man Castagné, Alexandra Sasha Luccioni, François             form for natural language understanding. In Pro-
  Yvon, Matthias Gallé, et al. 2022. Bloom: A 176b-            ceedings of the 2018 EMNLP Workshop Black-
  parameter open-access multilingual language model.           boxNLP: Analyzing and Interpreting Neural Net-
  arXiv preprint arXiv:2211.05100.                             works for NLP, pages 353–355, Brussels, Belgium.
                                                               Association for Computational Linguistics.
Ramsha Siddiqui. 2019. SARCASMANIA: Sarcasm
  Exposed! http://www.kaggle.com/rmsharks4/                  Sinong Wang, Belinda Z. Li, Madian Khabsa, Han
  sarcasmania-dataset.    [Online; accessed 02-                 Fang, and Hao Ma. 2020. Linformer: Self-attention
  February-2023].                                              with linear complexity.
David R. So, Wojciech Manke, Hanxiao Liu, Zihang             Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong
  Dai, Noam Shazeer, and Quoc V. Le. 2021. Primer:             Long, Chengqi Zhang, and S Yu Philip. 2020. A
  Searching for efficient transformers for language            comprehensive survey on graph neural networks.
  modeling. CoRR, abs/2109.08668.                              IEEE transactions on neural networks and learning
Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan,             systems, 32(1):4–24.
  Zhe Zhao, and Che Zheng. 2020. Synthesizer: Re-
  thinking self-attention in transformer models.             Ellery Wulczyn, Nithum Thain, and Lucas Dixon. 2017.
                                                                Ex machina: Personal attacks seen at scale. In Pro-
Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald                ceedings of the 26th International Conference on
  Metzler. 2022. Efficient transformers: A survey.             World Wide Web, WWW 2017, Perth, Australia, April
  ACM Computing Surveys, 55(6):1–28.                            3-7, 2017, pages 1391–1399. ACM.
Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen          Xiangru Tang Manuscript (sections 2 and 3;
  Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng          contributions to abstract; revision and proofread-
 Yan. 2022. Metaformer is actually what you need
                                                          ing). Contributions to Appendix K.
  for vision.
                                                          Matteo Grella Manuscript (sections 4.5, 4.6, 8;
Manzil Zaheer, Guru Guruganesh, Kumar Avinava
 Dubey, Joshua Ainslie, Chris Alberti, Santiago On-       contributions to sections 1, 7 and 9; proofreading
 tanon, Philip Pham, Anirudh Ravula, Qifan Wang,          and revision). Contributions to Appendix B.
 Li Yang, et al. 2020. Big bird: Transformers for
 longer sequences. Advances in Neural Information         Ferdinand Mom Manuscript (contributions to
 Processing Systems, 33.                                  section 1, 2, 4.3, 4.6; proofreading and revision).
                                                          Contributions to Appendix B.
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali
  Farhadi, and Yejin Choi. 2019. Hellaswag: Can a         Atsushi Saito Manuscript (sections 3 and 5; con-
  machine really finish your sentence? In ACL.
                                                          tributions to section 2). Figures 1a , 1b, 1c. Contri-
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali            butions to Appendix H
  Farhadi, and Yejin Choi. 2020. Winogrande: An
  adversarial winograd schema challenge at scale. In      Krishna Sri Ipsit Mantri Figure 4
  ACL.
                                                          Rui-Jie Zhu     Tables 1 and 5. Experiments for
Shuangfei Zhai, Walter Talbott, Nitish Srivastava, Chen   table 5.
  Huang, Hanlin Goh, Ruixiang Zhang, and Josh
  Susskind. 2021. An attention free transformer.          Peng Zhou Contributions to Table 5.
Sheng Zhang, Xiaodong Liu, Jingjing Liu, Jianfeng         Qihang Zhao Manuscript (proofreading and re-
  Gao, Kevin Duh, and Benjamin Van Durme. 2018.           vision). Contributions to Table 5.
  Record: Bridging the gap between human and ma-
  chine commonsense reading comprehension. In             Xuzheng He Manuscript (contributions to sec-
  arXiv:1810.12885.
                                                          tion 3; proofreading and revision). Contributions
Susan Zhang, Stephen Roller, Naman Goyal, Mikel           to Figures 1, 7. Appendix G. Contributions to ap-
  Artetxe, Moya Chen, Shuohui Chen, Christopher De-       pendix F.
  wan, Mona Diab, Xian Li, Xi Victoria Lin, et al.
  2022. Opt: Open pre-trained transformer language        Hayden Lau Manuscript (contributions to sec-
  models. arXiv preprint arXiv:2205.01068.                tion 1; proofreading and revision). Contributions
                                                          to Appendix K.
A    Author Contributions
                                                          Michael Chung Manuscript (contributions to
Bo Peng Original RWKV idea, original code,                section 4.6; proofreading and revision).
performance optimizations, original experiments,
and trained RWKV models from 0.1B to 14B.                 Haowen Hou Figure 8. Appendix E

Eric Alcaide Manuscript (initial draft sections 1,        Jiaming Kong Manuscript (revision and proof-
2; sections 4, 7 and 8; revision and proofreading;        reading). Appendix F.
final version ). Figures (2, 3, 4, 7). Experiments
                                                          Johan S. Wind RWKV performance optimiza-
section 6. Appendices D, I. Contributions to Ap-
                                                          tions (CUDA), Contributions to Appendix C.
pendix K.
                                                          Jian Zhu Manuscript (section 2; proofreading
Quentin Anthony Led writing the paper.
                                                          and revision). Figures 3 and 5.
Manuscript (initial draft sections 1, 2, 3; revision
and proofreading; final version).                         Huanqi Cao Manuscript (contributions to 4.2
                                                          and 4.3; proofreading and revision). Experiments
Zhenyuan Zhang Manuscript (revision and
                                                          for Appendix G.
proofreading) Figure 3. Experiments Appendix
G. Contributions to Appendices B and K.                   Samuel Arcadinho Contributions to Figures 6,
                                                          10, and 11. Contributions to Appendix I.
Kranthi Kiran GV Manuscript (sections 2 and
5; contributions to section 3; revision and proof-        Xin Cheng Manuscript (proofreading and revi-
reading). Tables 3 and 4. Appendix C.                     sion). Contributions to Appendix K, H.
Alon Albalak Manuscript (abstract and sections             that
1, 9; proofreading and revision).
                                                                        a1 = e−w a0 + ek0 v0 = ek0 v0 ,                   (23)
Jan Kocon Manuscript (sections 1; proofreading                                    −w            k0      k0
                                                                        b1 = e         b0 + e        =e ,                 (24)
and revision). Contributions to Appendix J.
                                                           and we set a01 = v0 , b01 = 1, p0 = k0 , where pt−1
Przemysław Kazienko Manuscript (section 6;
                                                           stores the shared exponents of at and bt . Now the
proofreading and revision). Contributions Ap-
                                                           above recursion can be converted into a numerical
pendix J.
                                                           safe version, for each time step t > 1:
Ruichong Zhang Manuscript (proofreading and
revision); Contributions to Figure 5 and Appendix                       q := max(pt−1 , u + kt ),                         (25)
K.                                                                    a∗t = ept−1 −q a0t−1 + eu+kt −q vt ,                (26)
Stanisław Woźniak Appendix J.                                        b∗t = ept−1 −q b0t−1 + eu+kt −q ,                   (27)
                                                                            a∗
Bartłomiej Koptyra Contributions to Appendix                        wkvt = ∗t .                                           (28)
                                                                            bt
J.
                                                           The update to a0t , b0t and their shared exponent are
B    Time-Mixing Block as an RNN Cell                      also carried out in similar fashion:
As stated in 4.3, the RWKV time-mixing block can                        q := max(pt−1 − w, kt ),                          (29)
be formulated as an RNN, as the W KV computa-
tion can be written in such a recursive form:                          a0t = ept−1 −w−q a0t−1 + ekt −q vt ,               (30)
                                                                       b0t = ept−1 −w−q b0t−1 + ekt −q ,                  (31)
             a0 , b0 = 0,                         (19)                 pt = q.                                            (32)
                    at−1 + eu+kt vt
             wkvt =                 ,             (20)     C      Parameter and FLOP Count for the
                     bt−1 + eu+kt
                                                                  RWKV Models
                 at = e−w at−1 + ekt vt ,         (21)
                 bt = e   −w            kt
                                bt−1 + e .        (22)     The following section provides an overview of the
                                                           different RWKV model architectures along with
  The dataflow of the RNN-like time-mixing is              their respective parameter and FLOP counts in Ta-
shown in Fig. 7, where the hidden states h is the          ble 2.
numerator-denominator tuple (a, b).                         Name     Layers   Model Dimension     Parameters    FLOPs per token
                                                            169 M      12           768          1.693 × 108     2.613 × 108
                                                            430 M      24          1024          4.304 × 108     7.573 × 108
                                                            1.5 B      24          2048          1.515 × 109     2.823 × 109
                                                             3B        32          2560          2.985 × 109     5.710 × 109
                   1        2                                7B        32          4096          7.393 × 109     1.437 × 1010
                                                             14 B      40          5120          1.415 × 1010    2.778 × 1010
                                    3


                   e                                       Table 2: RWKV model architectures and associated
                                                           FLOP counts

                                                              The number of parameters for each model is
                                                           computed using the formula: #parameters =
                                                           2V D + 13D2 L + D(11L + 4) where V = 50277
Figure 7: RWKV time-mixing block formulated as an
                                                           is the vocabulary size, D represents the Model Di-
RNN cell. Color codes: yellow (µ) denotes the token
shift, red (1) denotes the denominator, blue (2) denotes   mension and L corresponds to the number of lay-
the numerator, pink (3) denotes the fraction computa-      ers.
tions in 14. h denotes the numerator-denominator tuple        FLOPs is for a forward pass for one token. It
(a, b).                                                    was calculated as 6(V D + 13D2 L), which is the
                                                           twice (add and multiply) the number of parameters
   To avoid overflow in calculating ekt , a numerical      in linear layers. The backwards pass FLOPs can be
trick is used in the official implementation. Note         approximated as twice that of the forward pass. So
the total is 6(V D + 13D2 L) per token for training                 • All LayerNorm weights start from 1 and bi-
(3x fw FLOPs). It is noteworthy that FLOPs are                        ases from 0.
independent of the context length, unlike regular
transformers. The FLOP approximations in this                 E          Small Init Embedding
paper are in line with the methodology used by                This section presents experimental validation of
Kaplan et al. (2020).                                         small initialization embedding. The experimental
   Alternative approximations for FLOPs include               setup is as follows. In the baseline configuration,
doubling the parameters which yields similar re-              the parameters are initialized using a normal distri-
sults within 2% for 14B and a 30% discrepancy for             bution with a mean of 0.0 and a standard deviation
169M variant. Another approximation is based on               of 0.02, which is a commonly used initialization
the number of non-embedding parameters multi-                 method in models like BERT and GPT. On the other
plied by 2. This gives 2(V D + 13D2 L + D(11L +               hand, in the small initialization of the embedding
4)) resulting in 1.6% more FLOPs for 14B model                (small init emb) experiment, the parameters are ini-
and 8% more FLOPs for 169M model.                             tialized using a uniform distribution with a range of
                                                              1e-4, which is slightly different from RWKV where
D     Parameter initializations                               a normal distribution with a standard deviation of
                                                              1e-4 is used. However, this difference is negligible
We describe the specific parameter initializations
                                                              and does not affect our conclusions. The experi-
below and motivate the design choices. Parame-
                                                              ments were conducted with a batch size of 400. As
ters belonging to residual blocks are often adjusted
                                                              depicted in the figure 8, the loss curve for the small
by layer depth and total number of layers. Let #
                                                              init emb exhibits a faster rate of decrease and con-
denote the vocabulary size, s denote the embed-
                                                              vergence compared to the traditional initialization
ding dimension, d denote the hidden size (we use
                                                              using a normal distribution.
d = 4s), L the number of layers, l the layer index
(from 0 to L-1), we use the following initializa-
tions:
                                                                         11                                       Baseline
                                                                                                                  Small Init Emb
    • Embeddings are initialized to U(±1e-4) as                          10
      explained in 4.7                                                   9
    • For the channel-mixing blocks (11), µki and
                                      l                                  8
      µri are initialized to ( si )1− L                           Loss
    • For the time-mixing blocks (16), initializa-                       7
                               l                  l
      tions are µki = ( si )1− L , µvi = ( si )1− L + L−1
                                                      0.3l               6
                          1− Ll
      and µri = 0.5( si )                                                5
    • wi (14), also known as “time decay”, is initial-                   4
                                  1.3l
                          i 0.7+ L−1
      ized to −5+8·( d−1     )         . Intuitively, it is                   0   10000   20000          30000   40000      50000
                                                                                                  Step
      the discount factor applied to previous tokens
      over time.                                                  Figure 8: Effect of small initialization embedding.
    • ui (14), also known as “bonus”, is set to
      0.5(((i + 1) mod 3) − 1) + log 0.3. It is
      the special weighting applied to the current            F          Gradient Stability in RWKV
      token in equation 14. The alternating zigzag            In this section, we present a mathematical descrip-
      pattern initially creates subtle variations in the      tion of the gradient stability property in RWKV,
      tensor elements, which are intended to help             focusing specifically on the time-mixing block. By
      the model treat different dimensions of the             gradient stability we mean that if the inputs xt
      embedding distinctively.                                are bounded and the model parameters are fixed,
    • Wo (15) (time-mixing) and W         qv (channel-        then the gradients with respect to Wk and Wv are
      mixing) are initialized to N (0, ds = 2)                uniformly bounded for all T (thus not exploding).
    • All Wr , Wk , Wv weights are initialized to 0           Consequently, we can control the amount each xt
      so the model can start learning from the be-            contributes to the gradient at T in a naturally de-
      ginning without noisy signals.                          caying fashion by the weight decay mechanism w
(thus not vanishing unless desired).                                                        Time decay (sorted along channel axis)
                                                                             1.0
   First, we make the simplification that there are
no token shifts, this will not affect the final conclu-
                                                                             0.8
sion. In this scenario, wkvT can be written as
                                                                                        Layer 1
                                                                             0.6        Layer 2

                                                                Time Decay
              PT                                                                        Layer 3
                      Kte vt                  S(vt )                                    Layer 4
   wkvT = Pt=1
            T
                               = E(vt ) =            ,   (33)                           Layer 5
                       e
                  t=1 Kt
                                              S(1)                           0.4        Layer 6
                                                                                        Layer 7
                                                                                        Layer 8
where                                                                        0.2        Layer 9
                                                                                        Layer 10
                                                                                        Layer 11
                            ∂(vt )i                                          0.0        Layer 12
         vt = Wv xt ,                = (xt )j ,                                     0     100       200   300      400     500   600   700   800
                           ∂(Wv )i,j                                                                            Channel
                                                                                                Information propagation path                 1
                                                                        The




                                                                                                                                             Log-probability of "Paris"
                               ∂(Kte )i                                     E                                                                2
 Kte = eWk xt +wT,t ,                     = (xt )j (Kte )i ,              iff
                            ∂(Wk )i,j                                     el                                                                 3
                                                                      Tower                                                                  4
                                                                           is
                                                                    located                                                                  5
and S(·) and E(·) are shorthand for denoting sums                         in
                                                                        the                                                                  6
and averages over weights Kte .                                         city                                                                 7
                                                                          of
  The loss function at position T can be written as                                 1           6         11          16         21
                                                                                                           Layer

               LT = l(f (wkvT ), yT ).                   (34)   Figure 9: Model behavior visualizations of the RWKV
                                                                model.
Because wkvT relates to (Wk )i,j and (Wv )i,j only
through the i-th channel (wkvT )i , we have                     G                  Model Behavior Visualization
        ∂LT         ∂LT ∂(wkvT )i                               In Figure 9, we present visualizations of some be-
                =                     .                  (35)   havior of the RWKV model.
      ∂(Wv )i,j   ∂(wkvT )i ∂(Wv )i,j
                                                                   The top plot illustrates the time decays (e−w ) in
The first part of above equation contains trivial               each layer of the RWKV-169M model, sorted along
operations like output layers, and other layers of              the channel axis. Notably, several decays in the last
time-mixing, which can be proven inductively. The               layers are very close or equal to one, implying that
second part of above equation can be bounded as                 certain information is preserved and propagated
                                                                throughout the model’s temporal context. Mean-
   ∂(wkvT )i   ∂Ei [(vt )i ]                                    while, many decays in the initial layer are close
             =                                                  to zero, which corresponds to local operations in
   ∂(Wv )i,j   ∂(Wv )i,j
                                                                wkv (14), likely to be associated with tasks such as
                 = |Ei [(xt )j ]| ≤ max |(xt )j |, (36)
                                          t                     text parsing or lexical analysis. (Note that the local
                                                                operations in wkv is due to the extra parameter u,
which is irrelevant to T . Similarly,                           when e−w is degenerated into 0.) These patterns of
                                                                time decays are partly learned, but also come from
 ∂(wkvT )i    Si [(vt )i ]                                      parameter initialization as it speeds up training.
           =∂              /∂(Wk )i,j
 ∂(Wk )i,j       Si (1)                                            The bottom plot shows the information retrieval
             Si [(xt )j (vt )i ] Si [(xt )j ]Si [(vt )i ]       and propagation path in the RWKV-430M model.
           =                     −
                  Si (1)                 Si (1)2                The experiment follows the causal trace method
           = Ei [(xt )j (vt )i ] − Ei [(xt )j ]Ei [(vt )i ]     introduced by Meng et al. (2022), where we
              = covi ((xt )j , (vt )i )                  (37)           1. Run the model once, and record all states and
                                                                           activation of each layer during the computa-
can also be bounded. Note that wkv’s softmax op-                           tion;
eration contains at least two non-zero terms (u and                     2. Corrupt the input embeddings of the subject
w), so the above “covariance” will not degenerate                          using noise (“The Eiffel Tower” in this exam-
into 0.                                                                    ple);
  3. Restore the states and activation of a certain              such as medicine, nursing, biology, chemistry,
     layer at a certain token during the compu-                  psychology, and pharmacology.
     tation, and record the log-probability of the             • OpenBookQA (Mihaylov et al., 2018) A QA
     model outputting the correct answer (“Paris”).              dataset to evaluate human comprehension of
                                                                 a subject by incorporating open book facts,
Unlike transformers, RWKV relies on recursive                    scientific knowledge, and perceptual common
propagation of information in the time dimension.                sense, drawing inspiration from open book
In this case, the fact that "the Eiffel Tower is located         exams.
in Paris" is retrieved in layer 4. It is then passed           • SciQ (Johannes Welbl Nelson F. Liu, 2017)
down to the subsequent layers. In layer 20, mostly,              A multiple-choice QA dataset which was cre-
the information is propagated through time until                 ated using an innovative approach to gather
reaching where it is needed. Finally, it is passed               well-crafted multiple-choice questions that are
down to the last layer for outputting the answer.                focused on a specific domain.
                                                               • TriviaQA (Joshi et al., 2017) A QA-IR dataset
H    Evaluation Details
                                                                 which is constituted of triples of questions,
The results for following tasks are in Table 3 and 4.            answers, supporting evidence, and indepen-
  Tasks:                                                         dently collected evidence documents, with an
                                                                 average of six documents per question for re-
    • LAMBADA (Paperno et al., 2016). A bench-                   liable sources.
      mark dataset that evaluates the model’s contex-          • ReCoRD (Zhang et al., 2018) A benchmark
      tual reasoning and language comprehension                  for evaluating commonsense reasoning in
      abilities by presenting context-target pairs,              reading comprehension by generating queries
      where the objective is to predict the most prob-           from CNN/Daily Mail news articles and re-
      able target token.                                         quiring text span answers from corresponding
    • PIQA (Bisk et al., 2020). A benchmark for                  summarizing passages.
      the task of physical common sense reasoning,             • COPA (Roemmele et al., 2018) A dataset to
      which consists of a binary choice task that                evaluate achievement in open-domain com-
      can be better understood as a set of two pairs,            monsense causal reasoning.
      namely (Goal, Solution).                                 • MMMLU (Hendrycks et al., 2021) A multi-
    • HellaSwag (Zellers et al., 2019) A novel                   task dataset for 57 tasks containing elementary
      benchmark for commonsense Natural Lan-                     mathematics, US history, computer science,
      guage Inference (NLI) which is build by ad-                law, etc.
      versarial filtering against transformer models.
    • Winogrande (Zellers et al., 2020) A dataset
                                                           I   Inference results
      designed to evaluate the acquisition of com-
      mon sense reasoning by neural language mod-          Figures 10 and 11 illustrate, respectively, the results
      els, aiming to determine whether we are ac-          on time (s) and memory (RAM, VRAM) require-
      curately assessing the true capabilities of ma-      ments for LLM inference in float32 precision. We
      chine common sense.                                  benchmark the following model families and sizes:
    • StoryCloze (Mostafazadeh et al., 2016) A
                                                               • RWKV: 169m, 430m, 1.4b, 3b, 7b, 14b
      benchmark to present a novel approach to as-
                                                               • Bloom (Scao et al., 2022): 560m, 1b, 3b
      sess comprehension of narratives, narrative
                                                               • OPT (Zhang et al., 2022): 125m, 350m, 1.3b,
      generation, and script acquisition, focusing on
                                                                 2.7b, 6.7b, 13b
      commonsense reasoning.
                                                               • GPT-Neo (Black et al., 2022): 125m, 1.3b,
    • ARC Challenge (Clark et al., 2018) A dataset
                                                                 2.7b
      designed for multiple-choice question answer-
                                                               • Pythia (Biderman et al., 2023): 160m, 410m,
      ing, encompassing science exam questions
                                                                 1.4b, 2.8b, 6.7b, 12b
      ranging from third grade to ninth grade.
    • ARC Easy An easy subset of ARC.                         Missing models in are due to Out Of Memory
    • HeadQA (Vilares and Gómez-Rodríguez,                 (OOM) errors. A comparison at 512 tokens is
      2019) A benchmark consisting of graduate-            shown in Figure 11 as some large transformer mod-
      level questions encompassing various fields          els produced an OOM when inferencing longer se-
 Model               Params    PIQA     StoryCloze      HellaSwag   WinoGrande      ARC-e      ARC-c      OBQA
                     B         acc      acc             acc_norm    acc             acc        acc_norm   acc_norm
 RWKV-4              0.17      65.07    58.79           32.26       50.83           47.47      24.15      29.60
 Pythia              0.16      62.68    58.47           31.63       52.01           45.12      23.81      29.20
 GPT-Neo             0.16      63.06    58.26           30.42       50.43           43.73      23.12      26.20
 RWKV-4              0.43      67.52    63.87           40.90       51.14           52.86      25.17      32.40
 Pythia              0.40      66.70    62.64           39.10       53.35           50.38      25.77      30.00
 GPT-Neo             0.40      65.07    61.04           37.64       51.14           48.91      25.34      30.60
 RWKV-4              1.5       72.36    68.73           52.48       54.62           60.48      29.44      34.00
 Pythia              1.4       71.11    67.66           50.82       56.51           57.74      28.58      30.80
 GPT-Neo             1.4       71.16    67.72           48.94       54.93           56.19      25.85      33.60
 RWKV-4              3.0       74.16    70.71           59.89       59.59           65.19      33.11      37.00
 Pythia              2.8       73.83    70.71           59.46       61.25           62.84      32.25      35.20
 GPT-Neo             2.8       72.14    69.54           55.82       57.62           61.07      30.20      33.20
 RWKV-4              7.4       76.06    73.44           65.51       61.01           67.80      37.46      40.20
 Pythia              6.9       74.54    72.96           63.92       61.01           66.79      35.07      38.00
 GPT-J               6.1       75.41    74.02           66.25       64.09           66.92      36.60      38.20
 RWKV-4              14.2      77.48    76.06           70.65       63.85           70.24      38.99      41.80
 GPT-level∗          14.2      76.49    74.97           68.72       65.14           70.77      37.99      39.27
 Pythia (c.f.)       11.8      75.90    74.40           67.38       64.72           69.82      36.77      38.80
 GPT-NeoX (c.f.)     20.6      77.69    76.11           71.42       65.98           72.69      40.44      40.20

Table 3: Zero-Shot Performance of the model on Common Sense Reasoning Tasks. ∗ Interpolation of Pythia and
GPT-Neo models




   Model              Params    LAMBADA         LAMBADA         headQA      sciq    triviaQA    ReCoRD     COPA
                      B         ppl             acc             acc_norm    acc     acc         em         acc
   RWKV-4             0.17      29.33           32.99           25.78       77.50   1.26        62.03      66.00
   Pythia             0.16      24.38           38.97           25.82       76.50   1.31        66.32      62.00
   GPT-Neo            0.16      30.27           37.36           25.16       76.60   1.18        64.92      64.00
   RWKV-4             0.43      13.04           45.16           27.32       80.30   2.35        70.48      65.00
   Pythia             0.40      11.58           50.44           25.09       81.50   2.03        75.05      67.00
   GPT-Neo            0.40      13.88           47.29           26.00       81.10   1.38        73.79      65.00
   RWKV-4             1.5       7.04            56.43           27.64       85.00   5.65        76.97      77.00
   Pythia             1.4       6.58            60.43           27.02       85.50   5.52        81.43      73.00
   GPT-Neo            1.4       7.5             57.25           27.86       86.00   5.24        80.62      69.00
   RWKV-4             3.0       5.25            63.96           28.45       86.50   11.68       80.87      82.00
   Pythia             2.8       4.93            65.36           28.96       87.70   9.63        85.10      77.00
   GPT-Neo            2.8       5.63            62.22           27.17       89.30   4.82        83.80      80.00
   RWKV-4             7.4       4.38            67.18           31.22       88.80   18.30       83.68      85.00
   Pythia             6.9       4.3             67.98           28.59       90.00   15.42       86.44      85.00
   GPT-J              6.1       4.1             68.31           28.67       91.50   16.74       87.71      83.00
   RWKV-4             14.2      3.86            70.83           32.64       90.40   24.58       85.67      85.00
   GPT-level∗         14.2      3.81            70.94           31.03       92.20   22.37       87.89      82.66
   Pythia (c.f.)      11.8      3.89            70.44           30.74       91.80   20.57       87.58      82.00
   GPT-NeoX (c.f.)    20.6      3.64            71.94           31.62       93.00   25.99       88.52      84.00

Table 4: Zero-Shot Performance of various models on different tasks. ∗ Interpolation of Pythia and GPT-Neo
models
     Method               L    d      T      Train bpc   Test bpc   Time Complexity   Space Complexity
                                                                        2
     Transformer          12   512    1024   0.977       1.137      O(T d)            O(T 2 + T d)
     Transformer          24   256    1024   1.039       1.130      O(T 2 d)          O(T 2 + T d)
     Reformer             12   512    1024   1.040       1.195      O(T log T d)      O(T log T + T d)
     Synthesizer          12   512    1024   0.994       1.298      O(T 2 d)          O(T 2 + T d)
     Linear Transformer   12   512    1024   0.981       1.207      O(T d2 )          O(T d + d2 )
     Performer            12   512    1024   1.002       1.199      O(T d2 log d)     O(T d log d + d2 log d)
     AFT-simple           12   512    1024   0.854       1.180      O(T d)            O(T d)
     RWKV-RNN             6    512    1024   0.720       -          O(Td)             O(d)

Table 5: Enwik8 results, measured in bits per character (bpc): the lower the better. Baseline comparisons are made
with Reformer (Kitaev et al., 2020), Synthesizer (Tay et al., 2020) (the best performing dense version), Linear
Transformer (Katharopoulos et al., 2020), Performer (Choromanski et al., 2020). L, d, and T denote the number
of blocks (network depth), dimension of features, and sequence length, respectively. Both Linear Transformer and
Performer are implemented with customized CUDA kernels (github.com/idiap/fast-transformers), and all other
models are implemented in native Pytorch.


quences. For GPU experiments, we use an NVIDIA
A100 with 80GB of VRAM. For CPU experiments,
we use an AMD EPYC processor with 30 CPU
cores and 200 GiB RAM.




                                                             Figure 11: Text generation inference time for LLMs.
Figure 10: Text generation inference memory (CPU
RAM, GPU VRAM) for LLMs. Model parameters are
not accounted.
Task Name    Measure    ChatGPT    GPT-4   RWKV-4       RWKV-4     SOTA     Task Name    Measure    ChatGPT      RWKV-4      SOTA
             type            [%]     [%]   GPT [%]   changed [%]     [%]
                                                                                         type            [%]   adapted [%]     [%]
RTE          F1 Macro       88.1    91.3      44.2          74.8    92.1
WNLI         Accuracy       81.7    91.6      47.9          49.3    97.9    Aggression   F1 Macro      69.10        56.66    74.45
GoEmotions   F1 Macro       25.6    23.1       7.9           7.9    52.8
PolEmo2      F1 Macro       44.1    41.0      38.2          40.9    76.4    MathQA       Accuracy      71.40        80.69    83.20
                                                                            Sarcasm      F1 Macro      49.88        50.96    53.57
Table 6: ChatGPT, GPT-4 and RWKV-4-Raven-14B                                TweetSent    F1 Macro      63.32        52.50    72.07
                                                                            Unhealthy    F1 Macro      45.21        43.30    50.96
reasoning performance comparison in RTE (Wang
et al., 2019), WNLI (Wang et al., 2018), GoEmotions
(Demszky et al., 2020), and PolEmo2 (Kocoń et al.,                        Table 7: ChatGPT and RWKV-4-Raven-14B perfor-
2019) benchmarks. SOTA is provided as a supplemen-                         mance comparison in Aggresion (Wulczyn et al., 2017),
tary reference.                                                            Sarcasm (Siddiqui, 2019), Unhealthy (Price et al.,
                                                                           2020), MathQA (Cobbe et al., 2021), and TweetSent
                                                                           (Barbieri et al., 2020) benchmarks. SOTA is provided
                                                                           as a supplementary reference.
J   Importance of prompt construction
    and comparison to GPT models
                                                                           premise: <here is a premise>
Inspired by article (Kocoń et al., 2023), we com-                         hypothesis: <here is a hypothesis>
pared the zero-shot performance of the RWKV-                                  While separating the instruction from the input
4-Raven-14B with ChatGPT (access in February                               is relatively easy to do, other aspects of prompt
2023) and GPT-4 using several known NLP tasks,                             engineering are harder to quantify. Testing the ap-
i.e., recognizing textual entailment (RTE), Wino-                          proach of stating the input after the question on
grad Natural Language Inference (WNLI), and rec-                           multiple other tasks, shown in tab. 7, suggests that
ognizing emotions elicited in readers (GoEmotions                          better prompts might reduce the disparity between
and PolEmo2). Each model got the same prompts                              models. Raven achieves comparable result to Chat-
manually chosen to receive proper responses from                           GPT on unhealthy conversation detection and even
the ChatGPT model. As shown in Tab. 6, RWKV                                surpasses it on the sarcasm detection dataset. While
performs significantly worse than ChatGPT and                              this approach in prompting looks necessary, it alone
GPT-4 in specific task performance. We suspect                             is not enough to replace the capability of having
that this disparity is likely caused by the choice                         free access to the whole context. Therefore, prompt
of prompts used to generate the answers. Given                             engineering seems to be of significantly more im-
that prompts are in natural language and do not                            portance to the RNN models compared to stan-
consider that RWKV is an RNN, so it can not look                           dard transformers. It is entirely possible that good
back inside an instruction.                                                prompts to RNN models do not mean additional
   When the instruction style was adapted to re-                           restrictions, but should simply be constructed using
spect that RNNs is not capable for retrospective                           completely different guidelines. While authors of a
processing, quality on some datasets increased sig-                        forementioned paper (Kocoń et al., 2023) perform
nificantly (ex. for RTE (Wang et al., 2019) F1                             chain-of-thought to improve results on MathQA
Macro increased from 44.2% to 74.8%). We hy-                               dataset, trying the same approach on the Raven
pothesize that RWKV models are more sensitive                              model achieved a very low accuracy of 48.11%.
to the position of the components in the context,                          Despite that, using a very similar prompt without
as RNN-based architectures cannot look back and                            the chain-of-thought outperformed ChatGPT on
readjust the weight of previous information. For                           this dataset. It would not be surprising if, thanks
better performance, desired information should be                          to the hidden state, RNN models do not need addi-
after the question. Example of ChatGPT prompt to                           tional steps during inference, and adding those only
RTE:                                                                       makes the model forget the information it analyzed
Having premise <here is a premise> judge if the                            and deemed important. This further emphasizes the
following hypothesis <here is a hypothesis> are                            importance of the order of information the model
logically connected with the premise? Answer "en-                          receives.
tailment" if yes, or "not_entailment" if no.                               The template used to prompt the Raven model in
RWKV prompt taking into account the characteris-                           MathQA with chain-of-thought prompting:
tics of the RNN:                                                           Write the reasoning and highlight the answer to
Can you tell me if the hypothesis is entailment or is                      the question at the end in format: ’Answer: ’. The
not entailment to the premise?                                             question is: <here is question>
 The template used to prompt the Raven model in
 MathQA without chain-of-thought prompting:
 Write the answer to the math question in the format:
’Answer: ’.
 The question is: <here is question>
K        Cases



In this part, we present a few instances of outputs produced by RWKV model using a Chat interface56 .


    RWKV-4-Raven-14B
    Tell me about ravens.

    Ravens are large, black birds with a distinctive white head and neck. They are found in most parts
    of the world, including North America, Europe, Asia, and Australia. Ravens are known for their
    intelligence and problem-solving abilities. They are also considered to be symbols of death and
    transformation in many cultures. Ravens are often associated with the afterlife or death because
    they have been known to eat carrion or even other birds. In some cultures, ravens are also believed
    to be messengers of the gods or guardians of treasure.




    RWKV-4-Raven-14B
    Explain the concept of machine learning to a 10-year-old.

    Machine learning is like a super-smart computer that can learn on its own. It can recognize patterns
    and make decisions based on what it has learned. Just like how you might remember something
    from the past, the computer can remember what it’s learned and use it to make predictions or
    decisions in the future. It’s like having a smart friend who always knows what to do!




    RWKV-4-Raven-14B
    Write a short poem describing nature’s beauty in springtime.

                                       Nature’s beauty in springtime is rare,
                                      A symphony of colors, a gentle surprise.
                                     The flowers bloom, the birds sing and sing,
                                     As the sun shines down, warming our skin.
                                        The trees sway in the gentle breeze,
                                        As we take in their beauty with ease.
                                       The sky is blue, the clouds are white,
                                         A sight to behold, it’s truly divine.




    5
        https://github.com/BlinkDL/ChatRWKV
    6
        https://huggingface.co/spaces/BlinkDL/ChatRWKV-gradio
RWKV-4-Raven-14B
Below is the numerical stable formula for updating the state variable in RWKV:

                                     a∗t = ep−q at−1 + eu+kt −q vt
                                    b∗t = ep−q bt−1 + eu+kt −q
                                          a∗
                                  wkvt = ∗t
                                          bt
Please write out the corresponding pytorch code to do so, where both at and bt are
torch.Tensor([n])

Here is the PyTorch code to update the state variable in RWKV using the given formula:

   import t o r c h
   def u p d a t e _ s t a t e ( a , b , u , v , p , q ) :
          a _ t = t o r c h . exp ( p − q ) * a + t o r c h . exp ( u + k − q ) * v
          b _ t = t o r c h . exp ( p − q ) * b + t o r c h . exp ( u + k − q )
          return a_t , b_t
   a , b = t o r c h . randn (2 , 3)
   u = t o r c h . randn ( 2 )
   v = t o r c h . randn ( 2 )
   wkv = t o r c h . r a n d n ( 2 )
   a _ t , b _ t = u p d a t e _ s t a t e ( a , b , u , v , wkv , t o r c h . r a n d n ( 2 ) )
   print ( " a_t : " , a_t )
   print ( " b_t : " , b_t )


This code defines a function update_state that takes in the state variables a and b, the control
variables u and v, and the parameters wkv and torch.randn. It then uses the PyTorch torch.exp
function to compute the exponential of the control variables and the log of the normalization factor.
Finally, it uses these values to update the state.




RWKV-4-Raven-14B
Explain the following metaphor: Life is like cats.

The metaphor "Life is like cats" suggests that life can be unpredictable and full of unexpected
twists and turns, just like a cat’s behavior. It implies that we cannot fully control or predict the
outcome of our lives, just as we cannot control the movements of a cat. The metaphor also implies
that life is complex and multifaceted, with many different variables and factors at play. Just as a
cat can have multiple personalities or moods, life can have many twists and turns that are difficult
to predict or understand. Overall, the metaphor suggests that life is complex and unpredictable,
much like a cat’s behavior.
RWKV-4-Raven-14B
                          P∞      −i
Test the convergence of      i=0 2 .

The given series is a geometric series with the first term a0 = 2 and common ratio r = 12 . We can
find the sum of this series using the formula for a geometric series:
                           ∞                  1
                           X             a0          1    2
                                 ai =       = 2 1 =    1 = =2
                                        1−r  1− 2   1− 2  1
                           i=0

So, the sum of the given series is 2 .



RWKV-4-Raven-14B
I would like to cook some Chinese food at home. Do you have any suggestions on the meal I
could choose?

Yes, of course! If you are not very experienced in cooking Chinese food, you could start with
making noodles or rice, then make a dish using one of those. I also think making stir-fry with
some meat is an excellent idea.

I want to make something more special. What about red braised pork belly?

Yes, red braised pork belly is also very popular and tasty.

Tell me the steps to cook.

You will need a few ingredients such as pork belly, star anise, sugar, ginger, scallions and wine.
Put all of these in a pot, cook on medium heat and simmer until the meat is tender. After that,
remove the meat from the pot and reserve the sauce. Slice the meat and add some ginger slices
into it, then boil until it’s almost done.

What do I need to be careful with the most during the cooking process?

To cook pork belly properly, it is very important to monitor the heat of the stove and be patient
with the cooking process. Overcooking it can cause the pork to be tough, dry and difficult to chew.
You can add a few ingredients like soy sauce or black beans to give the pork more flavor.

Tell me more about the proper heat and timing.

If the temperature of the water is too high, it can damage the collagen and connective tissues,
resulting in tough pork belly. It is essential to control the temperature carefully, especially
when braising the pork. Additionally, keep the water simmering but not boiling to maintain a
low simmer, as high heat will dry out the pork belly and result in tough and unpalatable meat.
Lastly, monitor the heat during cooking and make sure the heat is medium, as this is what is most
appropriate for slow cooking pork belly.

I see. Thank you for your advice.

You’re welcome! It was a pleasure chatting with you. Let me know if you need any further
assistance in the future.
RWKV-4-Raven-14B
If all apples are fruits and all fruits are sweet, are all apples sweet? Please explain your
answer in detail.

Yes, all apples are fruits, and all fruits are sweet. Therefore, it is safe to assume that all apples are
sweet.

If birds can fly. Penguins are birds. Can penguins fly? Please explain your answer in detail.

No, penguins cannot fly. Penguins are birds, but they are not capable of flight like other birds.
They have wings and can glide for short distances, but they cannot take off or fly long distances
like other birds.

The day before two days after the day before tomorrow is Saturday. What day is it today?
Please explain your answer in detail.

Today is Friday. Two days after tomorrow is Sunday. Tomorrow is Saturday.

If five cats can catch five mice in five minutes, how long will it take one cat to catch one
mouse? Please explain your answer in detail.

If five cats can catch five mice in five minutes, it means that each cat can catch one mouse in one
minute. To find out how long it will take one cat to catch one mouse, we can use the concept of
proportionality.
Let’s assume that it takes x minutes for a cat to catch a mouse. Then, the number of mice caught
by a cat in one minute is equal to the number of cats multiplied by the number of mice caught by
each cat: x = 5 * 1 = 5
So, it will take 5 minutes for one cat to catch one mouse.



RWKV-4-Raven-14B
Summarize the following paragraph.

<The Abstract of this paper goes here>

The paper presents a novel architecture, Receptance Weighted Key Value (RWKV), that combines
the efficient parallelizable training of Transformers with the efficient inference of RNNs. The
model is able to scale to tens of billions of parameters and exhibits linear computational complexity
during training and inference, making it a promising alternative to Transformers for sequence
processing tasks.

Please further summarize it in one sentence.

The researchers have developed a new architecture that combines the efficient parallelizable
training of Transformers with the efficient inference of RNNs, which allows for linear scaling in
memory and computational requirements.

