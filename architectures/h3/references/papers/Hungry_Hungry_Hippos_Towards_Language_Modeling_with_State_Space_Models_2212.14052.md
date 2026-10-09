# Paper (Fu et al. 2022)

> Source: `https://arxiv.org/abs/2212.14052`

---

Hungry Hungry Hippos: Towards Language Modeling with State
Space Models
Daniel Y. Fu∗†, Tri Dao∗†, Khaled K. Saab‡, Armin W. Thomas††,
Atri Rudra‡‡, and Christopher R´e†
†Department of Computer Science, Stanford University
‡Department of Electrical Engineering, Stanford University
††Department of Psychology, Stanford University
‡‡Department of Computer Science and Engineering, University at Buﬀalo, SUNY
{danfu,tridao}@cs.stanford.edu, {ksaab,athms}@stanford.edu, atri@buffalo.edu,
chrismre@cs.stanford.edu
December 28, 2022
Abstract
State space models (SSMs) have demonstrated state-of-the-art sequence modeling performance in
some modalities, but underperform attention in language modeling. Moreover, despite scaling nearly
linearly in sequence length instead of quadratically, SSMs are still slower than Transformers due to poor
hardware utilization. In this paper, we make progress on understanding the expressivity gap between
SSMs and attention in language modeling, and on reducing the hardware barrier between SSMs and
attention. First, we use synthetic language modeling tasks to understand the gap between SSMs and
attention. We ﬁnd that existing SSMs struggle with two capabilities: recalling earlier tokens in the
sequence and comparing tokens across the sequence. To understand the impact on language modeling, we
propose a new SSM layer, H3, that is explicitly designed for these abilities. H3 matches attention on
the synthetic languages and comes within 0.4 PPL of Transformers on OpenWebText. Furthermore, a
hybrid 125M-parameter H3-attention model that retains two attention layers surprisingly outperforms
Transformers on OpenWebText by 1.0 PPL. Next, to improve the eﬃciency of training SSMs on modern
hardware, we propose FlashConv. FlashConv uses a fused block FFT algorithm to improve eﬃciency on
sequences up to 8K, and introduces a novel state passing algorithm that exploits the recurrent properties
of SSMs to scale to longer sequences. FlashConv yields 2× speedup on the long-range arena benchmark
and allows hybrid language models to generate text 2.4× faster than Transformers. Using FlashConv,
we scale hybrid H3-attention language models up to 2.7B parameters on the Pile and ﬁnd promising
initial results, achieving lower perplexity than Transformers and outperforming Transformers in zero- and
few-shot learning on a majority of tasks in the SuperGLUE benchmark.
1
Introduction
State space models (SSMs) have achieved state-of-the-art sequence modeling performance in domains ranging
from time series analysis [25] to audio generation [22]. However, they have yet to match the performance of
Transformers on language modeling, often underperforming Transformers by multiple points in perplexity [25].
An natural question is whether this gap in performance is due to inherent inductive biases and capabilities
in attention [17, 49], or whether it is a function of the signiﬁcant organizational resources that have been
spent training and tuning large attention-based language models [10, 32, 66], as well as specialized hardware
support for attention, ranging from tensor cores [45] to transformer chips [34, 48].
We take ﬁrst steps towards answering these questions in this paper. First, we use synthetic language
modeling tasks to show that there is an expressivity gap between SSMs and attention. Using our insights,
∗Equal Contribution. Order determined by coin ﬂip.
1
arXiv:2212.14052v3  [cs.LG]  29 Apr 2023

Shift
SSM
Diag
SSM
X
K
Q
V
Y
H3 Layer
H3 for Associative Recall
null
null
Diag
I(xt-1=a)
I(xt+1=a)
I(xt+1=a)
I(xN-1=a)
null
I(xt=a)
a
Shift
I(xN=a)
a
null
yt
null
yt+1
Shift
Shift
a
I(xt=a)
3
xt
xt+1
Diag
3
I(xt=a)
3
3
a
xN
Diag
I(xN=a)
3
3
yN
Store key
Store val
Recall val
Input
Out
8K Chunk
Fused
Block
FFTConv
SSM
State
Fused
Block
FFTConv
SSM
State
FlashConv
Figure 1: Left: H3 stacks two discrete SSMs with shift and diagonal matrices and uses multiplicative interactions
between input projections and their outputs to model comparisons between points in a sequence. Middle: H3 can
perform associative recall—which is easy for attention, but not existing SSMs. Right: FlashConv uses a new
state-passing algorithm over fused block FFTConv to increase hardware eﬃciency of SSMs, allowing H3 to scale to
billion-parameter models.
we design a new SSM layer that nearly matches attention in language modeling. Second, we propose better
hardware-aware algorithms for SSMs that allow them to take advantage of modern accelerators—and run
faster than attention.
Understanding the Expressivity Gap. To understand the gap between SSMs and attention, we draw
on synthetic language modeling tasks that have been proposed as a mechanistic basis for in-context learning
in Transformers [49] These synthetic languages focus on the ability to manipulate text—recalling tokens
from earlier time steps, or comparing tokens from diﬀerent points in a sequence. We ﬁnd that existing SSMs
struggle to model these synthetic languages. To probe how important these skills are for language modeling,
we propose H3 (Hungry Hungry Hippo), a new SSM-based layer designed to solve these language modeling
tasks. H3 stacks two SSMs, with multiplicative interactions between their outputs and input projections.
The SSMs allow H3 to keep a log of tokens (to recall them later), while the multiplicative interactions allow
for comparisons across the sequence.
H3 matches attention on the synthetic languages and almost closes the gap with Transformers on language
modeling—coming within 0.4 perplexity of Transformers on OpenWebText (compared to 3.4 ppl for existing
SSMs—even those explicitly designed for language modeling [42]). Furthermore, a simple hybrid H3-attention
model that retains two attention layers surprisingly outperforms Transformers on OpenWebText by 1.0
perplexity. To further evaluate H3 on language modeling, we train 125M-, 355M-, 1.3B-, and 2.7B-parameter
hybrid H3-attention language models on the Pile [21], using hyperparameters from GPT-3 [7].
These
hybrid models outperform Transformer-based language models of the same size in perplexity, and match or
outperform them on a majority of tasks in the SuperGLUE benchmark in zero- and few-shot learning. Since
the SSM layers in these hybrid models admit a recurrent view, they can also perform 2.4× faster inference
than Transformers.
Scaling SSMs. Next, we improve the eﬃciency of SSMs on modern hardware, to reduce the hardware
barrier between attention and SSMs. SSMs scale nearly linearly in sequence length instead of quadratically
like attention, but still run slower on modern hardware due to poor hardware utilization. To close this gap, we
propose FlashConv, a hierarchical algorithm for computing SSMs, inspired by IO-Aware attention [15]. The
technical challenge is that SSMs require a FFT-based convolution over the input sequence, which requires an
FFT, pointwise multiply, and inverse FFT. When implemented in cuFFT [47], this operation incurs expensive
GPU memory reads/writes, and cannot utilize the specialized matrix multiply units available on modern
hardware1. To use specialized matrix multiply units, we appeal to classical techniques that split the FFT
into blocks and compute it using a series of matrix multiplications. Combined with kernel fusion, this “block”
FFT solution increases hardware eﬃciency, but only as long as the sequence length can ﬁt into GPU SRAM
1An A100 GPU has a maximum of 312 TFLOPs/s of FP16 with tensor cores, but only 20 TFLOPs/s of FP32 (and 40
TFLOPs/s of FP16) without tensor cores [46]. This trend started with the V100 GPUs [45] and has continued with the H100
GPUs [48].
2

(on-chip memory, analogous to L1 cache on the CPU)—up to sequence length 8K on modern A100.
To scale to sequences longer than 8K, we propose a state passing algorithm (Figure 1 right), specialized to
SSMs. The key insight is that we can use the recurrent properties of SSMs to process the input in chunks—as
long as we keep track of an additional state vector. The state passing algorithm splits the input into the
largest chunks that can ﬁt into GPU SRAM, eﬃciently computes the FFT-based convolution using block FFT,
and updates an intermediate state to start the next chunk. Using this state-passing algorithm, FlashConv
can scale SSMs to any sequence length—even longer than can ﬁt on GPU SRAM at once—while maintaining
a near linear compute complexity. FlashConv sets state-of-the-art speed on long range arena using S4 [25],
outperforming Transformers by 5.8× and previous S4 models by 2×. FlashConv trains H3 4-8× times
faster than attention for long sequences, and is a critical component for scaling to billion-parameter models2.
2
Background
We present some background on state space models and linear attention, which inspired our H3 layer.
2.1
State Space Models
A continuous-time state-space representation [6] deﬁnes a linear mapping from an input signal u(t) ∈R (as
a function of time t) to an output signal y(t) ∈R through a state-variable x(t) ∈Rm, with the following
diﬀerential equation, for some matrices A ∈Rm×m, B ∈Rm×1, C ∈R1×m, D ∈R1×1: ˙x(t) = Ax(t) + Bu(t),
y(t) = Cx(t) + Du(t).
Similarly, a discrete-time state-space representation deﬁnes a linear mapping from a discrete input signal
ui (for i = 1, 2, . . . ) to a discrete output signal yi though a state-variable xi ∈Rm:
xi = Axi−1 + Bui
yi = Cxi + Dui.
A state-space model (SSM) uses these representations as a layer in a deep learning pipeline, where the
matrices A, B, C, D are learned from data (e.g., with gradient-based optimization). One often has d of these
SSMs in parallel, each corresponding to one hidden dimension. To preserve the sequence history, HiPPO [24]
projects the history on a basis of orthogonal polynomials, which translates to having SSMs whose A, B
matrices are initialized to some special matrices.
This recurrent form of SSMs allows eﬃcient inference (i.e., generation): to generate the output of the
next time-step, one only needs the state of the current time-step, not the entire input history. Furthermore,
SSMs can freely extrapolate to sequences longer than seen during training.
SSMs as Convolution. For eﬃcient training, given the entire sequence of the input u1, . . . , uN, the
output sequence y1, . . . , yN can also be written as the convolution of the input with the ﬁlter [27]:
f = [CB, CAB, CA2B, . . . , CAN−1B].
That is, from an initial condition x0, we have yi = CAiBx0 + (f ∗u)i + Dui, where (f ∗u) denotes a
linear convolution between f and u. If we set the initial condition x0 to be zero, then y is exactly a linear
convolution of u, with a residual connection Du. More generally, any linear time-invariant system (of which
SSMs are a special case) can be written as a convolution.
Given a 1D input sequence u ∈RN of length N, we denote the 1D output sequence y ∈RN of an SSM
parameterized by matrices A, B, C, D as
y = SSMA,B,C,D(u).
To simplify notation, we omit the reference to A, B, C, D and write y = SSM(u) if they are clear from context.
When u is multidimensional of dimension d, we stack d of these SSMs together that deﬁnes a mapping from
u ∈RN×d to y ∈RN×d, using the same notation y = SSM(u).
To construct the ﬁlter f from A, B, C eﬃciently, A is often constrained to be diagonal [26, 29], or diagonal
plus low-rank [25].
SSM through FFTs. Computing the convolution naively through conventional matrix operations is
expensive for long kernels, scaling as O(N 2). Instead, we can use FFTs: take the FFT of f and u, multiply
them together pointwise, and then take the inverse FFT. This yields an O(N log N) algorithm.
2Code for H3 is available at https://github.com/HazyResearch/H3
3

2.2
Linear attention
We describe linear attention [35] and its connection to RNNs, which inspired our model design (Section 3).
In standard attention [62], we have N query/key/value tokens Qi, Ki, Vi ∈Rd for i = 1, . . . , N, where N
is the sequence length and d is the head dimension. For some similarity metric Sim: Rd × Rd →R, we want
to compute the output:
Oi =
Pi
j=1 Sim(Qi, Kj)Vj
Pi
j=1 Sim(Qi, Kj)
∈Rd.
For standard softmax attention, Sim(q, k) = eq⊤k (often the dot product is scaled by 1/
√
d). Linear attention
makes the assumption that Sim has the form Sim(q, k) = φ(q)⊤φ(k), for some (nonlinear) function φ.
The output is then Oi =
φ(Qi)⊤Pi
j=1 φ(Kj)V ⊤
j
φ(Qi)⊤Pi
j=1 φ(Kj) . Let Si = Pi
j=1 φ(Kj)V ⊤
j
∈Rd×d, zi = Pi
j=1 φ(Kj) ∈Rd,
di = φ(Qi)⊤zi ∈R. Then Oi = φ(Qi)⊤Si
di
. This connects linear attention to RNNs: the output Oi is a function
of Si and zi, both of which are incrementally updated (as cumulative sums).
3
Hungry Hungry Hippos Layer to Model Discrete Sequences
To understand the gap between SSMs and attention on language modeling, we examine two synthetic language
modeling tasks. These tasks motivate our H3 layer to add a discrete SSM (based on shift matrix) and
multiplicative interaction to eﬀectively model discrete sequences. We then show that the H3 layer is expressive
enough to solve these synthetic tasks, and that this understanding leads to better performance on a real
language modeling benchmark.
3.1
Motivation: Synthetic Language Modeling Tasks
We describe two closely-related synthetic tasks, summarized in Table 1. Olsson et al. [49] argue that the
ability to solve (variants of) these tasks accounts for the majority of the in-context learning capability of
Transformers, and more intuition is given in Appendix E.
Table 1: Synthetic language modeling tasks.
Task
Input
Output
Sequence Length
Vocab Size
Induction Head
a b c d e ⊢f g h i . . . x y z ⊢
f
30
20
Associative Recall
a 2 c 4 b 3 d 1 a
2
20
10
The Induction Head task tests how well a model can recall content after a special token (e.g., ⊢
in Table 1). At the end of the sequence, the model must recall the token that appeared immediately after
the special token earlier in the sequence. Associative Recall [1] is similar to the induction head task, but
requires the model to remember multiple key-value pairs. At the end of the sequence, the model must recall
a speciﬁc value belonging to a speciﬁc key.
Table 2: Evaluation of 2-layer models on synthetic language tasks.
Task
Random
S4D
Gated State Spaces
H3
Attention
Induction Head
5.0
35.6
6.8
100.0
100.0
Associative Recall
25.0
86.0
78.0
99.8
100.0
Table 2 (for two-layer models) shows that S4D [26] and Gated State Spaces [42] both fail to model these
synthetic languages, which suggests they may not have the expressivity for general language. We argue that
these failures suggest two missing capabilities: (i) to remember tokens that appear after a particular event
(e.g., the special token in the induction head task), and (ii) to compare tokens across the sequence (e.g.,
comparing keys to decide which value to recall). Attention has both these capabilities: it can compare tokens
by constructing the quadratic attention matrix QK⊤, and it can recall tokens by direct copying (multiplying
softmax(QK⊤) with V). In Section 3.2, we design our new layer H3 to enable these capabilities in SSMs,
narrowing the expressivity gap between SSMs and attention.
4

3.2
H3 Layer
H3 uses SSMs with shift and diagonal matrices, along with multiplicative operations against projections of
the input to capture the missing capabilities identiﬁed by the synthetics.
High-level Intuition. (i) To remember tokens from the past, we want the state xi to copy from the
input ui, and then pass that information to the next state xi+1. As xi+1 relates to xiby Axi, we use a
discrete SSM with a shift matrix A (described formally below) that shifts the elements of a state vector (e.g.,
mapping [a, b, c] →[0, a, b]). (ii) To compare tokens across the sequence, we use multiplicative interaction:
the output of an SSM, containing information from previous time steps, is multiplied with the input at the
current time steps, thus measuring similarity between tokens.
H3 is loosely inspired by linear attention (Section 2): we project the input u to get three signals Q, K, V.
Then we replace the non-linearity φ(K) with an SSM where A is a shift matrix (SSMshift), and we replace the
summation Si with a SSM with diagonal A (SSMdiag). The output, for the case of head dimension dh = 1, is:
Q ⊙SSMdiag(SSMshift(K) ⊙V),
where ⊙denotes pointwise multiplication. We can view this form as stacking two SSMs with multiplicative
interaction (each is a “hungry hippo”, hence the name of our layer). A more formal connection between
linear attention, time-varying systems, and H3 can be found in Appendix B.
Remembering Key Tokens: Shift and Diagonal SSMs. The shift and diagonal SSMs are designed
to address the capability to log tokens after particular events. In the shift SSM, we constrain A ∈Rm×m to
be a shift matrix Ai,j =
(
1
for i −1 = j
0
otherwise
. The action of this matrix on the hidden state xi is to shift each
coordinate down by one—thereby creating a “memory” of the previous states. For example, if B = e1, the
ﬁrst basis vector, then xi = [ui, ui−1, . . . , ui−m+1] contains the inputs from the previous m time steps. We
learn B and C (B can also be ﬁxed to e1 for simplicity, in which case the output is a 1D conv. with kernel
size m).
The diagonal SSM constrains A to be diagonal and initializes it from the diagonal version of HiPPO
(S4D [26]). This parameterization allows the model to remember state over the entire sequence. The shift
SSM can detect when a particular event occurs, and the diagonal SSM can remember a token afterwards for
the rest of the sequence.
Multiplicative Interaction for Comparison. We take the multiplicative interactions from linear
attention, but they provide another missing capability when combined with a shift matrix: comparing
tokens across the sequence. The multiplicative interactions between the output of the shift SSM and the V
projection mimics local multiplicative interactions in linear attention (depending on the size of the hidden
state). Similarly, multiplicative interactions with the Q projection and the output of the diagonal SSM allows
comparisons between tokens over the entire sequence.
H3 Layer. The overall layer is given in Algorithm 1 and shown schematically in Figure 1 (left). We
use the H3 layer to construct a model in the same style as Transformers by interleaving it with MLPs,
connected by residual connection and layer norm (i.e., pre-norm architecture [2]). We will also consider a
hybrid H3-attention model (two attention layers while the rest are H3, Sections 3.3 and 5).
Algorithm 1 H3 Layer
Require: Input sequence u ∈RN×d from the previous layer, weight matrices WQ, WK, WV , WO ∈Rd×d, a shift
SSM SSMshift, a diagonal SSM SSMdiag, head dimension dh.
1: Compute Q = uWQ, K = uWK, V = uWV ∈RN×d.
2: Pass K through the shift SSM: K = SSMshift(K) ∈RN×d.
3: Split Q, K, V into H “heads” (Q(h), K
(h), V(h) for h = 1, . . . , H), each a sequence of N vectors of size dh = d/H.
4: for 1 ≤h ≤H do
5:
Take the batched outer product K
(h)(V(h))⊤∈RN×dh×dh (batched in the N-dimension) and pass it through a
diagonal SSM: KV(h) = SSMdiag(K
(h)(V(h))⊤) ∈RN×dh×dh.
6:
Batch-multiply by Q: O(h) = [Q(h)
1 KV(h)
1 , . . . , Q(h)
N KV(h)
N ] ∈RN×dh (batched in the N-dimension).
7: end for
8: Concatenate the output O(h) of each head, and multiply by the output projection matrix WO ∈Rd×d.
5

Eﬃciency
We show that H3 scales as O(N log N) with sequence length N—asymptotically more eﬃcient
than attention, which typically requires O(N 2d) time and O(N 2) space3 (proof in Appendix D.3).
Proposition 1. Let N be the sequence length, d be the hidden dimension, and assume that the head dimension
dh is of order O(1). Then the H3 layer takes O(d2N + dN log N) time and O(dN) space to compute.
3.3
Expressivity
We show that H3 can model our synthetic languages, as well as natural language on OpenWebText [23]. We
also present a hybrid H3-attention extension that outperforms Transformers on OpenWebText.
Mechanism for Solving Associative Recall with H3. H3 is expressive enough to solve our synthetic
language modeling tasks, as shown in Table 2. Figure 1 (middle) shows a mechanism for a single H3
layer to solve the associative recall task for a particular key-value pair (a, 3). The shift SSM and following
multiplicative interaction act as a gate on whether to let a value through to the diagonal SSM, based on
whether the previous token was key a. The diagonal SSM stores the value 3 in memory, and continually
outputs it. The ﬁnal multiplicative interaction gates whether to let the diagonal SSM’s output through—based
on whether the current input token is the key a. We formally construct the weights of an H3 layer to solve
this task in Appendix D.1.
Table 3: Perplexity of SSM variants compared to Transformers on OpenWebText. All models have 12 layers, with
size around 125M, and are trained with the same hyperpameters, for 50B tokens.
H3
H3 Hybrid (2 Attn)
S4D
GSS
GSS Hybrid (2 Attn)
Transformer
21.0
19.6
24.9
24.0
19.8
20.6
Better Synthetic Language Modeling Translates to Better Natural Language Modeling. We
validate that when H3 can solve these synthetic tasks, it also improves the modeling capability on natural
language (e.g., on the OpenWebText dataset). As shown in Table 3, H3 comes within 0.4 perplexity points
of Transformers when trained for 50B tokens on OpenWebText, and performs much better than existing SSM
variants (S4D, GSS), by 3 −3.9 points.
Extension: H3-attention Hybrid Model. A simple hybrid H3-attention language model surprisingly
outperforms Transformers on OpenWebText by 1.0 point. Our hybrid model simply retains two self-attention
layers: one in the second layer, and one in the middle (layer 2 + N/2 for an N-layer model, N even). The
H3-attention hybrid also outperforms the GSS-attention hybrid [42].
4
FlashConv: Eﬃciently Training SSMs
To improve the eﬃciency of SSMs on modern hardware, we propose FlashConv. FlashConv fuses the
FFT, pointwise multiply, and inverse FFT to reduce memory reads/writes. It also uses a block FFT algorithm
to make use of specialized matrix multiply units (e.g., tensor cores on A100) for sequence lengths up to
8K. For sequences longer than 8K, the computation no longer ﬁts in GPU SRAM4, so we propose a novel
state-passing algorithm that splits the sequence into chunks to compute the FFT convolution one chunk at a
time. FlashConv can speed up any SSMs (not just H3).
4.1
Fused Block FFTConv
We deploy two techniques to speed up the FFT-based convolution for sequences shorter than 8K: kernel
fusion and block FFT. Kernel fusion addresses IO bottlenecks due to reading and writing of intermediate
results, while block FFT allows the FFT-based convolution to utilize specialized matrix multiplication units.
These techniques allow us to speed up FFTConv by 2× (Section 6) for sequences shorter than 8k.
Kernel Fusion. Naive implementations of FFTConv using standard libraries such as cuFFT are IO-
bound due to repeated reading and writing of intermediate results. The FFT convolution in an SSM with
input u and ﬁlter f has the form iFFT(FFT(u) ⊙FFT(f)) (where ⊙denotes pointwise multiplication).
3There are several memory-eﬃcient algorithms for attention [15, 52], though their time complexity is still quadratic in N,
which is a lower-bound for attention [36].
4SRAM, or on-chip memory, is much faster than oﬀ-chip GPU memory, but usually much smaller, on the order of around
100KB for each streaming processor.
6

It requires reading and writing intermediate results to GPU memory—which can dominate the runtime.
Following FlashAttention [15], we ﬁrst fuse the entire FFTConv into a single kernel and compute it in
SRAM to avoid this overhead.
Block FFT. To further speed up the computation of FFT-based convolution, we exploit specialized
matrix multiplication hardware on modern GPUs (e.g., Tensor Cores on Nvidia GPUs perform fast 16 × 16
matrix multiplication). We appeal to classical results that show that the FFT can be written as a series of
block-diagonal matrix multiplications interleaved with permutation. We note that such algorithms are not
new, but our setting (fused FFTConv on GPU) introduces new bottlenecks—by removing the IO bottlenecks,
compute becomes the bottleneck (note that a single FFT on GPU is usually IO bound).
Suppose that we want to perform an N-point FFT, which is equivalent to multiply by the DFT matrix FN.
Suppose that N = N1N2 for some integers N1, N2. By the Cooley-Tukey decomposition of the DFT [3, 11]
(also known as the four-step FFT algorithm), we can write FN = P(IN2 ⊗FN1)P⊤D(IN1 ⊗FN2)P, where P
denotes a ﬁxed permutation that reshapes the input as a N1 × N2 array and then transpose it, ⊗denotes
Kroneker product, D is a N × N diagonal matrix (called the twiddle factors) [14], and INi and FNi are the
identity and DFT matrix of size Ni × Ni. As IN2 ⊗FN1 and IN1 ⊗FN2 are just block-diagonal matrices,
we can make use of specialized matmul units to perform these multiplications. Similarly, if N = N1N2N3
then we can decompose the N-point FFT into a series of (block) FFT of size N1, N2, and N3, interleaved by
permutation.
The block FFT algorithm incurs O(Nr log N/ log r) FLOPs for a sequence length N, if N can be written
as rp for two integers r, p. This incurs more FLOPs than standard FFT (O(N log N)), but can run faster
when we using specialized matrix multiplication hardware.
4.2
State-Passing
However, the fused kernel cannot run if the sequence is too long to ﬁt into GPU SRAM (longer than 8K on
A100). We show how to exploit the particular form of the FFT in SSM to speed it up for long sequences.
The recurrent nature of SSMs allows us to split the FFTConv of a length-N sequence into chunks of size
N ′ each (N ′ is the longest FFT we can ﬁt into SRAM), assuming N is a multiple of N ′). We use FFTConv
to compute each chunk and use a recurrence to connect the chunks. In particular, we split the inputs u into
C = N/N ′ chunks u(c) ∈RN ′ for c = 1, . . . , C. Similarly, split the states x into x(c) ∈RN ′×m and the output
y into y(c) ∈RN′ for i = 1, . . . , C. We will only need the end-state x(c)
N′ of each chunk c.
Let f = [CB, CAB, CA2B, . . . , CAN′−1B] be the SSM ﬁlter. Recall from Section 2 that for each chunk c,
y(c)
i
= CAiBx(c−1)
N′
+ (f ∗u(c))i + Du(c)
i , since x(c−1)
N′
, the end-state of the previous chunk (c −1) is the initial
condition for the current chunk c. In vector notation, y(c) = Mxyx(c−1)
N′
+ f ∗u(c) + Du(c) for some matrix
Mxy ∈RN′×m. Additionally we need to update the end-state of each chunk with xc
N′ = AN′x(c−1)
N ′
+Muxu(c)
for some matrix Mm×N ′
ux
(derivation in Appendix C.2). In essence, we can compute the output for each chunk
with FFT-based convolution as long as we remember the end-state of the previous chunk, and the end-state
of each chunk can be updated recurrently. This yields a state-passing algorithm for long sequences, where we
only compute FFT of length N ′, and update some hidden state each iteration.
Let BlockFFTConv refer to our fused block FFTConv kernel. Then, the state-passing algorithm for 1D
input is given by Algorithm 2. For inputs of dimension d where we stack d SSMs, we simply batch Algorithm 2
along the d-dimension.
Algorithm 2 State Passing Algorithm
Require: Input u ∈RN, SSM parameterized by matrices A ∈Rm×m, B ∈Rm×1, C ∈R1×m, D ∈R1×1, chunk size
N ′ where N is a multiple of N ′.
1: Precompute AN′ ∈Rm×m, Mux = [AN′−1B, . . . , B] ∈Rm×N′, Mxy = [C, CA, . . . , CAN′−1] ∈RN′×m.
2: Split the inputs u1:N into C = N/N ′ chunks u(c)
1:N′ for c = 1, . . . , C.
3: Let the initial state be x(0)
N′ = 0 ∈Rm.
4: for 1 ≤c ≤C do
5:
Compute y(c) = Mxyx(c−1)
N′
+ BlockFFTConv(f, uj) +Du(c) ∈RN′.
6:
Update state: x(c)
N′ = AN′x(c−1)
N′
+ Muxu(c).
7: end for
8: Return y = [y(1) . . . y(C)].
7

Table 4:
Perplexity (lower is better) of models on the Pile, OpenWebText and WikiText-103. GPT-Neo and hybrid
H3 are trained on the Pile, while GPT2 is trained on WebText. All models use the same GPT2 tokenizer. We report
the perplexity of GPT-2 models on the Pile (∗) for context, though the performance is not directly comparable since
they were trained on diﬀerent data.
Model
Pile
OpenWebText
WikiText103
GPT-2 small (125M)
19.0*
22.6
29.9
GPT-Neo-125M
9.4
22.6
26.3
Hybrid H3-125M
8.8
20.9
23.7
GPT-2 medium (355M)
13.9*
17.0
21.8
Hybrid H3-355M
7.1
15.9
16.9
GPT-2 XL (1.5B)
12.4*
12.9
17.0
GPT-Neo-1.3B
6.2
13.1
13.3
Hybrid H3-1.3B
6.0
12.4
12.5
GPT-Neo-2.7B
5.7
11.7
11.5
Hybrid H3-2.7B
5.4
11.0
10.6
Table 5: Zero-shot acc. on SuperGLUE with logit scoring. Best results in bold, second best underline.
Model
WSC
WIC
RTE
CB
MultiRC
ReCoRD
BoolQ
COPA
Average
OPT-125M
39.4
52.0
48.7
37.4
58.9
44.9
59.6
60.0
50.1
GPT-Neo-125M
36.5
53.6
53.1
41.1
59.9
39.6
62.2
60.0
50.8
Hybrid H3-125M
39.4
51.4
59.2
48.2
51.4
55.0
59.6
67.0
53.9
GPT-2 medium (355M)
50.0
52.0
51.3
28.6
59.5
53.3
61.0
65.0
52.6
OPT-350M
53.5
50.8
53.4
35.7
58.9
51.4
60.9
60.0
53.1
Hybrid H3-355M
37.5
51.7
55.2
41.1
59.5
62.3
61.5
69.0
54.7
OPT-1.3B
36.5
49.5
53.4
39.3
58.3
61.8
55.0
69.0
52.9
GPT-Neo-1.3B
41.3
50.0
52.3
33.9
57.9
55.5
59.9
66.0
52.1
Hybrid H3-1.3B
52.9
50.3
53.4
33.9
58.2
67.8
61.7
74.0
56.5
OPT-2.7B
51.0
50.8
50.5
41.1
57.4
65.9
60.9
66.0
55.5
GPT-Neo-2.7B
37.5
50.0
52.3
50.0
59.1
60.0
61.1
67.0
54.6
Hybrid H3-2.7B
36.5
51.3
57.0
37.5
58.7
71.3
61.1
81.0
56.8
We prove that Algorithm 2 yields the same output as if one has computed the SSM using a large FFT of
size N (proof in Appendix D.4):
Proposition 2. For input u ∈RN and matrices A, B, C, D, the output y ∈RN returned by Algorithm 2 is
the same as the output deﬁned by the SSM parameterized by A, B, C, D.
5
H3 Evaluation
To understand how well capturing the synthetics in Section 3.1 translates to language modeling, we train
two hybrid hybrid H3-attention language models at sizes 125M, 355M, 1.3B, and 2.7B, and we evaluate
their performance against Transformers. The hybrid models match or exceed the quality of Transformers in
perplexity and zero/few-shot learning. We also validate that H3 models retain strong performance on non-text
sequence modeling. Appendix F contains additional experiments on more datasets, length extrapolation, and
scaling with data.
5.1
Language Modeling
We compare hybrid H3-attention language models against Transformer-based language models. We evaluate
language modeling performance using perplexity, zero-shot learning, and few-shot learning performance.
Hybrid H3 models outperform Transformers, which suggests that closing the gap between SSMs and attention
on the synthetic languages translates to real language modeling capabilities. We also report the generation
speed of hybrid H3 models compared to Transformers; since SSMs are recurrent models, they can generate
tokens 2.4× faster than Transformers. Appendix F shows performance of pure H3 language models on these
same evaluation metrics.
Setup
We train hybrid models at sizes 125M, 355M, 1.3B, and 2.7B on the Pile [21] for 400B tokens. We
compare against checkpoints of equivalent sizes from Open-AI [53] and GPT-Neo5 [4], from HuggingFace [65].
5There is no pretrained GPT-Neo at the 350M size.
8

Table 6: 3-shot acc. on SuperGLUE with logit scoring. Best results in bold, second best underline.
Model
WSC
WIC
RTE
CB
MultiRC
ReCoRD
BoolQ
COPA
Average
OPT-125M
36.5
50.2
47.3
44.6
57.9
44.9
41.9
60.0
47.9
GPT-Neo-125M
38.5
50.0
53.1
17.9
56.3
39.6
62.1
60.0
47.2
Hybrid H3-125M
43.3
49.1
58.1
51.8
48.9
55.0
56.1
67.0
53.7
GPT-2 medium (355M)
36.5
50.5
48.0
8.9
43.5
53.3
58.8
65.0
45.6
OPT-350M
37.5
50.0
45.8
44.6
49.8
51.4
61.7
60.0
50.1
Hybrid H3-355M
42.3
47.5
50.5
28.6
59.7
62.3
60.5
69.0
52.6
OPT-1.3B
44.2
51.1
53.4
16.1
59.9
62.1
38.3
70.0
49.4
GPT-Neo-1.3B
35.6
50.6
47.3
32.1
59.9
55.7
61.2
67.0
51.2
Hybrid H3-1.3B
36.5
49.2
55.2
23.2
59.3
67.6
56.9
76.0
53.0
OPT-2.7B
44.2
50.5
53.4
17.9
59.2
66.0
62.0
71.0
53.0
GPT-Neo-2.7B
49.0
51.9
51.6
21.4
57.0
60.0
56.0
68.0
51.9
Hybrid H3-2.7B
36.5
45.6
47.3
46.4
59.4
71.1
60.6
77.0
55.5
Table 7: Inference throughput on A100 80GB, 1.3B models. Batch size 64, prompt length 512, 1024, or 1536, and
generating 128 tokens per sequence in the batch (i.e., 64 × 128 tokens in a batch). Hybrid H3 is up to 2.4× faster
than a Transformer of similar size in inference. The diﬀerence is larger for longer sequences.
Tokens/s
Prompt length 512
Prompt length 1024
Prompt length 1536
Transformer-1.3B
1340
770
520
Hybrid H3-1.3B
1980
1580
1240
Perplexity
Table 4 shows perplexity on the Pile [21], OpenWebText [23], and WikiText-103 [43]. On
the Pile, our 125M hybrid model outperforms GPT-Neo, which was also trained on the Pile. Our hybrid
models also outperform GPT-Neo models and GPT-2 models on zero-shot transfer to OpenWebText and
WikiText103. We report the PPL of GPT-2 models for context, though the performance is not directly
comparable since they were trained on diﬀerent data.
Zero- and Few-shot Performance
We compare the zero- and few-shot performance of hybrid H3
language models against OPT [66], GPT-Neo, and GPT-2 models, where public checkpoints are available.
We report performance with rank classiﬁcation on the logits of the possible choices (see Appendix F.7 for raw
generation). Table 5 reports zero-shot performance on the SuperGLUE benchmark, and Table 6 reports the
3-shot performance. Following the perplexity results, the hybrid language models outperform or match the
best the Transformer baseline on more than half the tasks.
Language Modeling Inference
Finally, since SSMs are recurrent models, they admit faster text generation
than Transformers. Table 7 shows inference throughput of a 1.3B-parameter hybrid model compared to a
Transformer. The hybrid model has up to 2.4× higher throughput.
6
FlashConv Evaluation
We evaluate how well FlashConv speeds up SSMs. FlashConv sets state-of-the-art performance on
the long range arena benchmark [59] using S4 [25]. We report performance of training H3 module with
FlashConv compared to attention at various sequence lengths, from 256 to 32K and demonstrate nearly
linear scaling.
Long Range Arena
The Long Range Arena (LRA) benchmark [59] is a benchmark for long-range sequence
modeling. The state-of-the-art approach, S4 [28], is an SSM. Table 8 shows that FlashConv accelerates S4
by 2×, outperforming Transformers by 5.8×.
Benchmark H3 Against Attention
We benchmark the time to run the forward and backward pass of
H3 with FlashConv against attention. FlashConv maintains nearly linear scaling, even to very long
sequence lengths. Fig. 2 shows overall 2-3× speedup over FFTConv with cuFFT using our techniques (block
FFT, state-passing). Simple kernel fusion (even without block FFT) can yield speedup over cuFFT for short
sequences, since memory reads/writes are the bottleneck for short sequences. For long sequences, SSMs using
state passing can be dozens of times faster than even the fastest attention implementation.
9

Table 8: Speedup on the LRA benchmark.
Models
Speedup
Transformer
1×
FlashAttention [15]
2.4×
Block-sparse FlashAttention [15]
2.8×
S4 [28]
2.9×
S4 with FlashConv
5.8×
1K
4K
16K
Sequence Length
10
20
30
Time (ms)
Benchmarking FlashConv
Inset (256-4K)
2x
35x
FlashAttention
FlashConv
cuFFT Conv
Fused Conv
Fused Block FFT Conv
1K
2K
4K
Sequence Length
1
3
5
Time (ms)
Figure 2: We compare the speed of diﬀerent algorithms to perform FFT-based convolution, along with FlashAtten-
tion [15] (the fastest attention implementation we know of). We use batch size 8, hidden dimension 1024, and varying
sequence length from 256 to 32k, and measure on an A100-SMX4-40GB GPU. We see that kernel fusion gives up to
3.4× speedup over naive FFTConv for short sequences (up to 512), block FFT gives up to 2× speedup for medium
length sequences (1k - 8k), and state-passing allows 2.3× faster FFTConv for long sequences (16k and above).
7
Conclusion
Our main goal is to understand and narrow the gap between attention and SSMs in language modeling in terms
of modeling capabilities and hardware eﬃciency. Our exploration based on synthetic language tasks motivated
us to design the H3 layer, which is surprisingly competitive with attention. Our BlockFFTConv algorithm
exploits matrix multiplication units and the dual recurrent–convolution view of SSMs to substantially speed
up SSMs, reducing the hardware barrier between attention and SSMs. We are excited about several future
directions. Our H3 layer is a simple combination of two SSMs, and more sophisticated designs could be more
expressive. Our encouraging results on language models up to 1.3B parameters suggests that scaling SSMs to
larger sizes is a promising avenue. Since simply adding two attention layers to H3 models already outperforms
both the pure H3 model and Transformers, we are optimistic about combining the complementary strengths
of SSMs and attention in the future.
Acknowledgments
We thank Albert Gu for helpful discussion regarding the model architecture, and more importantly for sending
us daily hippo videos. We thank Together Computer for providing portions of the compute used to train
models in this paper. We gratefully acknowledge the support of NIH under No. U54EB020405 (Mobilize),
NSF under Nos. CCF1763315 (Beyond Sparsity), CCF1563078 (Volume to Velocity), and 1937301 (RTML);
ARL under No. W911NF-21-2-0251 (Interactive Human-AI Teaming); ONR under No. N000141712266
(Unifying Weak Supervision); ONR N00014-20-1-2480: Understanding and Applying Non-Euclidean Geometry
in Machine Learning; N000142012275 (NEPTUNE); NXP, Xilinx, LETI-CEA, Intel, IBM, Microsoft, NEC,
Toshiba, TSMC, ARM, Hitachi, BASF, Accenture, Ericsson, Qualcomm, Analog Devices, Google Cloud,
Salesforce, Total, the HAI-GCP Cloud Credits for Research program, the Stanford Data Science Initiative
(SDSI), Department of Defense (DoD) through the National Defense Science and Engineering Graduate
Fellowship (NDSEG) Program, Wu Tsai Neuroscience Stanford Interdisciplinary Graduate Fellowship, and
members of the Stanford DAWN project: Facebook, Google, and VMWare.
The U.S. Government is
authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright
10

notation thereon. Any opinions, ﬁndings, and conclusions or recommendations expressed in this material are
those of the authors and do not necessarily reﬂect the views, policies, or endorsements, either expressed or
implied, of DARPA, NIH, ONR, or the U.S. Government. Atri Rudra’s research is supported by NSF grant
CCF-1763481.
References
[1] Jimmy Ba, Geoﬀrey E Hinton, Volodymyr Mnih, Joel Z Leibo, and Catalin Ionescu. Using fast weights
to attend to the recent past. Advances in neural information processing systems, 29, 2016.
[2] Alexei Baevski and Michael Auli. Adaptive input representations for neural language modeling. In
International Conference on Learning Representations, 2018.
[3] David H Bailey. FFTs in external or hierarchical memory. The journal of Supercomputing, 4(1):23–35,
1990.
[4] Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large Scale Autoregres-
sive Language Modeling with Mesh-Tensorﬂow, March 2021. If you use this software, please cite it using
these metadata.
[5] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S
Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of
foundation models. arXiv preprint arXiv:2108.07258, 2021.
[6] Willian L Brogan. Modern control theory, 1974.
[7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners.
Advances in neural information processing systems, 33:1877–1901, 2020.
[8] Kyunghyun Cho, Bart Van Merri¨enboer, Dzmitry Bahdanau, and Yoshua Bengio. On the properties of
neural machine translation: Encoder-decoder approaches. arXiv preprint arXiv:1409.1259, 2014.
[9] Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane,
Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking
attention with performers. In International Conference on Learning Representations (ICLR), 2020.
[10] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts,
Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language
modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.
[11] James W. Cooley and John W. Tukey. An algorithm for the machine calculation of complex fourier
series. Mathematics of Computation, 19(90):297–301, 1965.
[12] Kamalaker Dadi, Ga¨el Varoquaux, Antonia Machlouzarides-Shalit, Krzysztof J Gorgolewski, Demian
Wassermann, Bertrand Thirion, and Arthur Mensch. Fine-grain atlases of functional modes for fmri
analysis. NeuroImage, 221:117126, 2020.
[13] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov.
Transformer-xl: Attentive language models beyond a ﬁxed-length context. In Proceedings of the 57th
Annual Meeting of the Association for Computational Linguistics, pages 2978–2988, 2019.
[14] Tri Dao, Beidi Chen, Nimit Sohoni, Arjun Desai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh
Rao, Atri Rudra, and Christopher R´e. Monarch: Expressive structured matrices for eﬃcient and accurate
training. In International Conference on Machine Learning (ICML), 2022.
[15] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and
memory-eﬃcient exact attention with io-awareness. In Advances in Neural Information Processing
Systems, 2022.
11

[16] Giannis Daras, Nikita Kitaev, Augustus Odena, and Alexandros G Dimakis. Smyrf-eﬃcient attention
using asymmetric clustering. Advances in Neural Information Processing Systems, 33:6476–6489, 2020.
[17] Benjamin L Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang. Inductive biases and variable creation
in self-attention mechanisms. In International Conference on Machine Learning, pages 5793–5831. PMLR,
2022.
[18] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda
Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac
Hatﬁeld-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario
Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathemat-
ical framework for transformer circuits.
Transformer Circuits Thread, 2021.
https://transformer-
circuits.pub/2021/framework/index.html.
[19] Bruce Fischl. Freesurfer. Neuroimage, 62(2):774–781, 2012.
[20] Robert S Fisher, Carlos Acevedo, Alexis Arzimanoglou, Alicia Bogacz, J Helen Cross, Christian E Elger,
Jerome Engel Jr, Lars Forsgren, Jacqueline A French, Mike Glynn, et al. Ilae oﬃcial report: a practical
clinical deﬁnition of epilepsy. Epilepsia, 55(4):475–482, 2014.
[21] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang,
Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language
modeling. arXiv preprint arXiv:2101.00027, 2020.
[22] Karan Goel, Albert Gu, Chris Donahue, and Christopher R´e. It’s raw! audio generation with state-space
models. arXiv preprint arXiv:2202.09729, 2022.
[23] Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. Openwebtext corpus, 2019.
[24] Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher R´e. Hippo: Recurrent memory with
optimal polynomial projections. Advances in Neural Information Processing Systems, 33:1474–1487,
2020.
[25] Albert Gu, Karan Goel, and Christopher R´e. Eﬃciently modeling long sequences with structured state
spaces. In The International Conference on Learning Representations (ICLR), 2022.
[26] Albert Gu, Ankit Gupta, Karan Goel, and Christopher R´e. On the parameterization and initialization
of diagonal state space models. In Advances in Neural Information Processing Systems, 2022.
[27] Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher R´e. Combining
recurrent, convolutional, and continuous-time models with linear state-space layers. Advances in neural
information processing systems, 34, 2021.
[28] Albert Gu, Isys Johnson, Aman Timalsina, Atri Rudra, and Christopher R´e. How to train your hippo:
State space models with generalized orthogonal basis projections. arXiv preprint arXiv:2206.12037, 2022.
[29] Ankit Gupta, Albert Gu, and Jonathan Berant. Diagonal state spaces are as eﬀective as structured state
spaces. In Advances in Neural Information Processing Systems, 2022.
[30] Curtis Hawthorne, Andrew Jaegle, C˘at˘alina Cangea, Sebastian Borgeaud, Charlie Nash, Mateusz
Malinowski, Sander Dieleman, Oriol Vinyals, Matthew Botvinick, Ian Simon, et al. General-purpose,
long-context autoregressive modeling with perceiver ar. arXiv preprint arXiv:2202.07765, 2022.
[31] Sepp Hochreiter and J¨urgen Schmidhuber. Lstm can solve hard long time lag problems. Advances in
neural information processing systems, 9, 1996.
[32] Jordan Hoﬀmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford,
Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal
large language models. arXiv preprint arXiv:2203.15556, 2022.
[33] Sara Hooker. The hardware lottery. Communications of the ACM, 64(12):58–65, 2021.
12

[34] Sheng-Chun Kao, Suvinay Subramanian, Gaurav Agrawal, and Tushar Krishna. An optimized dataﬂow
for mitigating attention performance bottlenecks. arXiv preprint arXiv:2107.06419, 2021.
[35] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Fran¸cois Fleuret. Transformers are RNNs:
Fast autoregressive transformers with linear attention. In International Conference on Machine Learning,
pages 5156–5165. PMLR, 2020.
[36] Feyza Duman Keles, Pruthuvi Mahesakya Wijewardena, and Chinmay Hegde. On the computational
complexity of self-attention. arXiv preprint arXiv:2209.04881, 2022.
[37] Michael Patrick Kerr. The impact of epilepsy on patients’ lives. Acta Neurologica Scandinavica, 126:1–9,
2012.
[38] Maedbh King, Carlos R Hernandez-Castillo, Russell A Poldrack, Richard B Ivry, and J¨orn Diedrichsen.
Functional boundaries in the human cerebellum revealed by a multi-domain task battery. Nature
neuroscience, 22(8):1371–1378, 2019.
[39] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The eﬃcient transformer. In The
International Conference on Machine Learning (ICML), 2020.
[40] Shengjie Luo, Shanda Li, Tianle Cai, Di He, Dinglan Peng, Shuxin Zheng, Guolin Ke, Liwei Wang, and
Tie-Yan Liu. Stable, fast and accurate: Kernelized attention with relative positional encoding. Advances
in Neural Information Processing Systems, 34:22795–22807, 2021.
[41] Christopher J Markiewicz, Krzysztof J Gorgolewski, Franklin Feingold, Ross Blair, Yaroslav O Halchenko,
Eric Miller, Nell Hardcastle, Joe Wexler, Oscar Esteban, Mathias Goncavles, et al. The openneuro
resource for sharing of neuroscience data. Elife, 10:e71774, 2021.
[42] Harsh Mehta, Ankit Gupta, Ashok Cutkosky, and Behnam Neyshabur. Long range language modeling
via gated state spaces. arXiv preprint arXiv:2206.13947, 2022.
[43] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models,
2016.
[44] Eric Nguyen, Karan Goel, Albert Gu, Gordon Downs, Preey Shah, Tri Dao, Stephen Baccus, and
Christopher R´e. S4nd: Modeling images and videos as multidimensional signals with state spaces. In
Advances in Neural Information Processing Systems, 2022.
[45] NVIDIA. Nvidia Tesla V100 GPU architecture, 2017.
[46] NVIDIA. Nvidia A100 tensor core GPU architecture, 2020.
[47] NVIDIA. cuﬀt v11.7.1 documentation, 2022. https://docs.nvidia.com/cuda/cuﬀt/index.html.
[48] NVIDIA. Nvidia H100 tensor core GPU architecture, 2022.
[49] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan,
Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac
Hatﬁeld-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal
Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah.
In-context learning and induction heads. Transformer Circuits Thread, 2022. https://transformer-
circuits.pub/2022/in-context-learning-and-induction-heads/index.html.
[50] Alan V Oppenheim. Applications of digital signal processing. Englewood Cliﬀs, 1978.
[51] Alan V Oppenheim, John R Buck, and Ronald W Schafer. Discrete-time signal processing. Vol. 2. Upper
Saddle River, NJ: Prentice Hall, 2001.
[52] Markus N Rabe and Charles Staats. Self-attention does not need O(n2) memory. arXiv preprint
arXiv:2112.05682, 2021.
13

[53] Alec Radford, Jeﬀrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language
models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
[54] Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, Chloe Hillier, and Timothy P Lillicrap. Com-
pressive transformers for long-range sequence modelling. In International Conference on Learning
Representations, 2019.
[55] Khaled Saab, Jared Dunnmon, Christopher R´e, Daniel Rubin, and Christopher Lee-Messer. Weak
supervision as an eﬃcient approach for automated seizure detection in electroencephalography. NPJ
digital medicine, 3(1):1–12, 2020.
[56] Vinit Shah, Eva Von Weltin, Silvia Lopez, James Riley McHugh, Lillian Veloso, Meysam Golmohammadi,
Iyad Obeid, and Joseph Picone. The temple university hospital seizure detection corpus. Frontiers in
neuroinformatics, 12:83, 2018.
[57] Mohammad Khubeb Siddiqui, Ruben Morales-Menendez, Xiaodi Huang, and Nasir Hussain. A review of
epileptic seizure detection using machine learning classiﬁers. Brain informatics, 7(1):1–18, 2020.
[58] Siyi Tang, Jared Dunnmon, Khaled Kamal Saab, Xuan Zhang, Qianying Huang, Florian Dubost, Daniel
Rubin, and Christopher Lee-Messer. Self-supervised graph neural networks for improved electroen-
cephalographic seizure analysis. In International Conference on Learning Representations, 2021.
[59] Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu
Yang, Sebastian Ruder, and Donald Metzler. Long range arena: A benchmark for eﬃcient transformers.
In International Conference on Learning Representations, 2020.
[60] Armin W Thomas, Christopher R´e, and Russell A Poldrack. Self-supervised learning of brain dynamics
from broad neuroimaging data. arXiv preprint arXiv:2206.11417, 2022.
[61] David C Van Essen, Stephen M Smith, Deanna M Barch, Timothy EJ Behrens, Essa Yacoub, Kamil
Ugurbil, Wu-Minn HCP Consortium, et al. The wu-minn human connectome project: an overview.
Neuroimage, 80:62–79, 2013.
[62] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz
Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing
systems, 30, 2017.
[63] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with
linear complexity. arXiv preprint arXiv:2006.04768, 2020.
[64] Pete Warden. Speech commands: A dataset for limited-vocabulary speech recognition. arXiv preprint
arXiv:1804.03209, 2018.
[65] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric
Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen,
Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame,
Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing.
In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System
Demonstrations, pages 38–45, Online, October 2020. Association for Computational Linguistics.
[66] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher
Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. OPT: Open pre-trained transformer language models.
arXiv preprint arXiv:2205.01068, 2022.
14

A
Related Work
State space models have shown promise in modeling sequential data, including time series data [25],
audio [22], and visual data [44]. Our model builds oﬀwork on simplifying and parameterizing diagonal
versions of S4 [26, 28, 29]. Gated state spaces [42] also aim to adapt SSMs to language modeling, but our
results suggest that the GSS model does not perform as well as H3 (or even as well as earlier SSMs like
S4D). The idea to combine SSMs with attention in hybrid models is not new; Mehta et al. [42] also showed
that interleaving attention with their GSS layer can improve performance, which we also validate on our
OpenWebText experiments. These positive results suggest that attention and SSMs are complementary, and
that hybrid models may be a promising direction for future work.
Large language foundation models [5] have demonstrated the power of scaling attention-based
networks to billions of parameters and training them on trillions of tokens [32]. Understanding the mechanistic
basis [18] behind these models may yield insights into better design choices for future models. These and
similar explorations have informed the design of H3 and our selection of synthetic languages. A number of
recent works have also explored how to address the shortcomings of attention by approximating the attention
computation [9, 16, 35, 39, 59, 63]. We believe these eﬀorts are complementary to SSMs, and we are excited
to see how they can be combined in future work.
Linear attention [35] and classical sequence models like RNNs serve as inspiration for H3. Appendix B
draws a direct connection between linear attention and LTI systems. Luo et al. [40] also propose a variant of
linear attention that can achieve O(n log n) scaling in sequence length. Appendix F evaluates linear attention
on language modeling, and ﬁnds that it underperforms exact attention, whereas H3 outperforms attention.
The multiplicative interactions in H3 are reminiscent of gating mechanisms in LSTMs [31] and GRUs [8],
which suggests that architectural lessons from these sequence models may be useful for adapting SSMs to
language modeling. A number of algorithms for scaling attention to longer sequences have also been proposed,
such as Transformer-XL [13], Reformer [39], Performer [9], and Perceiver AR [30]. Some of these approaches
underperform exact attention on language modeling, and may be slower in wall-clock speed [15]. A thorough
comparison of these alternatives to exact attention and how well they scale in model size and amount of
training data is fruitful future work.
FFT algorithms are used in a wide variety of applications, including signal processing [50], control
theory [6], and more. Various algorithms for computing the FFT have existed for decades [51]. We hope our
work on appealing to these classic algorithms to accelerate new applications such as learned SSMs will inspire
future algorithmic exploration, even if hardware is not designed for them [33].
B
Linear Attention and Time-Varying Systems
We draw some connections from linear attention to LTI systems and SSMs.
We ﬁrst present linear attention as a linear time-varying system, and show how converting it to a linear
time-invariant system matches H3.
Linear time-varying system and linear attention
In general, a layer in a sequence model takes in a
sequence and outputs a sequence. Many of these take the form of a linear time-varying system (thanks to the
Picard-Lindelof theorem, nonlinear systems can be approximated by a series of linear system):
xi = Aixi−1 + Biui,
yi = Cixi + Diui.
This has the same form as SSMs (Section 2), except that the matrices can depend on the timestep.
Recall the form of linear attention from Section 2. For the purpose of approximation, we ignore the
denominator in linear attention Section 2 (i.e., assuming di = 1). We see that Si is just a cumulative sum,
satisfying the recurrence Si+1 = Si+φ(Ki+1)V T
i+1. Similarly, Oi satisﬁes the recurrence Oi+1 = φ(Qi+1)T Si+1.
This is a linear time-varying system of the form xi+1 = Axi + Bui+1 and yi+1 = Ci+1xi+1 (with A = I,
B = I, ui = φ(Ki)V T
i , Ci = φ(Qi)T ). That is, A and B are constant, but C is time-variant.
To convert this into a linear time-invariant version, we treat the time-variant Ci as a post-processing step.
15

We instead of a ﬁxed C for the LTI. This yields an LTI:
xi+1 = Axi + Bφ(Ki)V T
i ,
yi+1 = Cxi,
for some matrices A, B, C that are learned. We then apply post-processing by multiply yi+1 with φ(Qi)T .
Replacing φ(Ki) with a shift SSM yields an analogue to H3.
C
Method details
Since we have described the forward pass in Section 3, we describe here the backward pass in details.
C.1
Backward Pass
We show how to compute the backward pass in a fused kernel.
Let y = f ∗u + Du. In our case, we have f and u have the same length, so they are symmetric as far as
the convolution is concerned.
Suppose we are given dy = ∂l
∂y (where l is some loss function). We wish to compute du, df, and dD (which
are
∂l
∂u, ∂l
∂f , and
∂l
∂D, respectively).
The most challenging part is computing the gradient through the convolution operator - but we’ll see
that we can re-use our FFT infrastructure for it. The rest of the operations are straightforward; we have
dD = dyuT .
Gradient of the Convolution
Here, we’ll discuss how to compute df by integrating w.r.t. the convolution
operator ∗. As an immediate consequence, we’ll be able to compute du as well.
Since f and u are the same length L, f ∗u and u ∗f have the same result. Thus, we’ll start from u ∗f
here.
For some notation, let O = u ∗f. Then, dO = dy. Recall that O[i] = Pi−1
j=0 u[i −j]f[j].
We’ll start by extending u and f with zeros, to give them length 2L. Let u′ = [u[0], u[1], . . . , u[L −
1], 0, . . . , 0], and f ′ extended similarly. Let O′ = u′ ∗f ′, and O = O′[: N]. Assume that we have all the values
of dO′ (we only have them for the ﬁrst half, but we’ll see that it doesn’t matter in the end).
Let’s construct a Toeplitz matrix Hu′ such that u′ ∗f ′ = Hu′f ′:
Hu′ =


u′[0]
0
. . .
0
u′[1]
u′[0]
. . .
0
...
...
...
...
u′[2L −1]
u′[2L −2]
. . .
u′[0]


Since, we have u′[i] = f ′[i] = 0 for i ≥L, we can actually ﬁll in the zeros of the above matrix as well:
Hu′ =


u′[0]
u′[2L −1]
. . .
u′[1]
u′[1]
u′[0]
. . .
u′[2]
...
...
...
...
u′[2L −1]
u′[2L −2]
. . .
u′[0]


Then, we can use the matrix multiplication chain rule to ﬁnd that:
df ′ = HT
u′dO′ =


u′[0]
u′[1]
. . .
u′[2L −1]
u′[2L −1]
u′[0]
. . .
u′[2L −2]
...
...
...
...
u′[1]
u′[2]
. . .
u′[0]


=


u′[0]
u′[−(2L −1)]
. . .
u′[−1]
u′[−1]
u′[0]
. . .
u′[−2]
...
...
...
...
u′[−(2L −1)]
u′[−(2L −2)]
. . .
u′[0]

,
16

where we use u′[−i] to mean u′[2L −i]. Notice that this matrix has the same format as Hu′! Let u′
∗=
[u′[0], u′[−1], . . . , u′[−(2N −1)]]. Then:
df ′ = (u′
∗∗dO′).
So how do we compute u′
∗eﬃciently? Naively, we might incur some nasty memory access issues. But a nice
property about the DFT saves us!
Let U[i] be the i-th element of the DFT of a signal u. Note that U[i] is complex. We have:
U ∗[i] = U[−i],
where here the ∗represents the complex conjugate. We can use this property to compute df ′ eﬃciently:
df ′ = u′
∗∗dO′ = iFFT(FFT ∗(u′)FFT(dO′)) ⇒df = df ′[: N] = iFFT(FFT ∗(u′)FFT(dy′))[: N],
where FFT ∗denotes taking the complex conjugate of the FFT, and dy′ denotes dy, padded with zeros.
Computing du
We can use this same trick to compute du, except we need to add in the contribution from
the original Du term. We end up with:
du = du′[: N] + Ddy = iFFT(FFT ∗(f ′)FFT(dy′))[: N] + Ddy.
C.2
State-Passing Matrices
We show how to derive Mux for the state update in our state-passing algorithm.
We wish to construct a matrix vMux ∈Rm×N′ such that Muxu = PN′
i=1 AN′−1Bui.
Note that
AiB ∈Rd×1 is a column vector. We can simply stack these column vectors to form a matrix: Mux =
[AN ′−1B, AN′−2B, . . . , B].
D
Proofs
We show parameterizations of H3 and attention that solves the associative recall task. We prove Proposition 1
and Proposition 2.
D.1
H3 Expressivity
This section formally describes a parameterization of H3 that solves the associative recall task.
D.1.1
Example Language Λ
Consider a simple language with 4 keys and 4 values. For concreteness, we will use the keys {k1, k2, k3, k4} =
LK and the values {v1, v2, v3, v4} = LV , i.e. our language L = LK ∪LV . Given a sequence of key-value pairs
with one key at the end, we want a model to generate the value associated with the key at the end. Assume
that the key at the end appeared in the sequence.
More formally, let N + 1 be the length of the sequence, N even. The language Λ consists of sequences
x ∈LN+1. Each sequence has an associated mapping fx : LK →LV . For each sequence, the odd indices are
randomly sampled from LK, for x1, x3, . . . , xN−1. The even indices are deﬁned by fx: x2∗i = fx(x2∗i−1), for
1 ≤i ≤N/2. The last item in the sequence xN+1, is randomly drawn from the keys that have appeared in x
already, i.e. xN+1 ∈∪{x1, x3, . . . , xN−1}. The goal of this language modeling task is to produce fx(xN+1) at
the end of the sequence.
D.1.2
H3 Model to Solve Λ
We describe a toy H3 model that can solve Λ.
Consider a model consisting of an embedding layer, an H3 model, and an output projection with softmax.
Recall that d is the dimension of the H3 model, m is the dimension of its hidden states, and H is the number
of heads. Let d = 8, m = 2, H = 4. Let the embedding layer map each key ki to the ei basis vector, and map
each value vi to the e4+i basis vector.
17

Let Bshift and Cshift denote the parameters of the shift SSM, and Adiag, Bdiag, and Cdiag denote the
parameters of the diagonal SSM (let D be zero for both). Let Bshift = Bdiag = Cdiag = e1. Let Cshift =
[01]. Let Adiag be a diagonal matrix with 1s along its diagonal for each H3.
Remark. The action of a diagonal SSM parameterized by Adiag, Bdiag, and Cdiag is to act as a cumulative
sum over all its input. The action of shift SSM parameterized by Bshift and Cshift is to shift its input by
one time step.
Recall that the H3 layer maps its input to Q, K, and V by applying uWQ, uWK, and uWV . Let WQ
and WK be the following:
WQ = WK =


1
1
0
0
0
0
0
0
0
0
1
1
0
0
0
0
0
0
0
0
1
1
0
0
0
0
0
0
0
0
1
1
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0


Recall that Q and K are split into H heads (Q(i), K(i) for i ∈{1, 2, 3, 4}), each of which is sent to an
independent H3.
Remark. The action of WQ and WK are to “assign” each key to a diﬀerent H3 head, i.e., Q(i)
t
is only
non-zero when xt = ki. Similarly, K
(i)
t
is only non-zero when xt−1 = ki (since Kt = Kt−1 due to the time
delay of the shift SSM).
Let WV be the following:
WV =


0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
1
0
1
0
1
0
1
1
0
1
0
1
0
1
0
1
1
1
1
1
1
1
1


Remark. The action of this matrix is to encode the input value (as “binary”), and send it to all H3 heads.
E.g., V(1)
t
= V(2)
t
= V(3)
t
= V(4)
t
for all i, and V(i)
t
= [0, 0] ⇔xt = v1, V(i)
t
= [0, 1] ⇔xt = v2, etc.
We claim that for xN+1 = ki, O(i)
N+1 will be a multiple of the binary encoding of fx(ki), and all the other
heads of the output O(j)
N+1, 1 ≤j ≤4, j̸ = i, will be zero. Let the output projection WO be such that, with
a non-linearity afterwards, it inverts the binary encoding to produce the embedding of the desired output
fx(ki). We will assume such a projection exists, proof left to the reader.
Proposition 3. The model described above solves the associative recall problem for the language Λ.
Proof. Proof sketch. WLOG, let xN+1 = ki. Then Q(i) = [1, 1], but Q(j) = [0, 0] for j̸ = i. Thus, O(j) = [0, 0]
for j̸ = i due to the multiplicative interaction.
Since Q(i) = [1, 1], O(i) is the output of the diag SSMs in the H3 head corresponding to ki (recall that
each head has two independent shift SSMs and two independent diag SSMs). The output of the diag SSMs
are the cumulative sum of all the inputs they have seen in the sequence.
For one of the diag SSMs to see a non-zero input, its preceding shift SSM must have a non-zero output.
The only times t this can happen in the sequence are when xt−1 = ki. But then xt = fx(ki). Thus, the input
to the diag SSMs are precisely the binary encoding of fx(ki). Then the output O(i) is a multiple of the binary
encoding of fx(ki), WO decodes this output to the embedding of fx(ki).
18

D.2
Attention Expressivity
We provide an informal sketch of a two-layer attention model that can solve the associative recall task,
inspired by the construction of [49]. The ﬁrst layer of the attention model outputs the embedding of the
previous token in the sequence, and concatenates it with the current token in the sequence. The second layer
compares the current token to the previous token embeddings, and outputs the paired embedding when there
is a match—which is exactly the key-value lookup.
The construction proceeds as follows:
• In the ﬁrst layer, let Qi be mapped to the positional embedding of token xi−1 (e.g., pi−1 if pi denotes
the positional embedding of token xi), and Ki be mapped to the positional embedding of token xi.
• The attention matrix A is computed as QKT , with a causal mask (i.e., Ai,j = 0 if j > i).
• Then, softmax(A) approximates the shift matrix (see Section 3).
• Let Vi be an encoding of token xi, constrained to the ﬁrst half of the hidden dimension.
• Then, for output O = softmax(QKT )V , the ﬁrst half of the vector Oi is the encoding of token xi−1.
• In the second layer, assume that you have a skip connection, that maps the encoding of the input token
xi to the second half of the vector Oi.
• Then, the input to the second layer encodes both xi−1 and xi.
• In the second layer, let Qi extract the encoding of xi, and let Ki extract the encoding of xi−1.
• Apply a causal mask on QKT . Then, the value of softmax(QKT )i,j is large if xi = xj−1, and i > j −1.
• Let Vi extract the encoding of xi.
• Then, output Oi is the sum of values xj such as xj−1 = xi. But then Oi is exactly a lookup of the token
that came after xi when it appeared previously in the sequence—which exactly solves associative recall.
We note that the above construction requires the ability for the positional encodings to select the previous
token based on the dot product and softmax, and for token comparisons through the dot product and softmax.
D.3
H3 Complexity
We prove Proposition 1, which states that the H3 layer takes O(d2N + dN log N) time and O(dN) space for
sequence length N and hidden dimension d.
Proof. We ﬁrst analyze the time complexity. Consider the matrix multiplies in H3, where the input u ∈RN×d
is multiplied by three weight matrices of size d × d. These take time O(d2N). The output O is also multiplied
with an output projection weight matrix of size d × d, also taking time O(d2N). Therefore the matrix
multiplies take time O(d2N).
Now consider the two SSMs in H3. The ﬁrst SSM involves a convolution of K ∈RN×d (in the N-
dimension) with a kernel of size N × d. This reduces to an FFT, a pointwise multiply, and an inverse FFT
(in the N-dimension). This takes time O(dN log N). The second SSM involves H convolutions, inputs of size
N × dh × dh, along the N-dimension. This takes time:
O(Hd2
hN log N) = O(ddhN log N) = O(dN log N),
where we use the fact that dh = d/H and that dh = O(1).
Therefore the two SSMs take total time
O(dN log N). As a result, the H3 layer takes time:
O(d2N + dN log N).
Now we analyze the space complexity. The matrix multiplies all take space O(dN). The FFTs, pointwise
multiplies, and inverse FFTs of the two SSMs takes O(dN) space and O(Hd2
hN) = O(ddhN) = O(dN) space.
Therefore the overall space complexity is O(dN).
19

D.4
State Passing Correctness
We prove Proposition 2.
We assume that the BlockFFTConv algorithm is correct, i.e., the output
y =BlockFFTConv(f, u) is equal to the output of an SSM with convolution kernel f and input u.
Proof. Proof by induction on C.
Base case:
C = 1. WTS y = [y(1)], Mxxx(0)
N′ + Muxu(1) = xN.
In this case, note that N = N ′. Then y(1) = Mxyx(0)
N′+BlockFFTConv(f, u1) =BlockFFTConv(f, u1).
But u = u1, so y = y(1) = [y(1)].
Additionally, by the recursive deﬁnition of a state space,
xN = AN−1x0 +
N
X
i=1
AN−iBui
= AN ′−1x0 +
N′
X
i=1
AN′−iBu(1)
i
= Mxyx(0)
N′ + [AN′−1B, AN ′−2B, . . . , B]u(1).
= Mxyx(0)
N′ + Muxu(1).
Inductive step:
C > 1. Assume that [y(1), . . . , y(C−1)] = y[: N ′(C −1)], and x(C−1)
N ′
= x(C−1)N ′. WTS
that y(C) = y[N ′(C −1) : N ′C], and Mxxx(C−1)
N′
+ Muxu(C) = xN. Let t denote N ′(C −1).
For i > (C −1)N ′, we have:
yi = CAi−tBxt + (f ∗[ut, ut+1, . . . , ut+N ′−1])i−t + Dui
= CAi−tBxt + (f ∗u(C))i−t + Dui
= CAi−tBxt + BlockFFTConv(f, u(C))i−N′
= (Mxyxt + BlockFFTConv(f, u(C)))i−N′
= (Mxyx(C−1)
N′
+ BlockFFTConv(f, u(C)))i−N ′
= y(C)
i−N ′.
Thus, y(C) = y[N ′(C −1) : N ′C].
Similarly,
xN = AN′−1x(C−1)N ′ +
N′
X
i=1
AN′−iBui+t
= AN′−1x(C−1)
N′
+
N′
X
i=1
AN′−iBu(C)
i
= Mxxx(C−1)
N ′
+ [AN′−1B, AN′−2B, . . . , B]u(C)
= Mxxx(C−1)
N′
+ Muxu(C).
E
Experimental Details
E.1
Synthetics
Our synthetic tasks, inspired by [49], are designed to mimic the in-context learning capability of large language
models—the ability to learn from examples in the input sequence, and use information from the input to
20

generate the right answer for the output. For example, the induction head task requires memorizing the
token that appears after the special ⊢token in the input sequence, and the associative recall task requires
learning the mapping from keys to tokens from the input sequence.
We evaluate synthetics by training two-layer versions of our GPT models, with diﬀerent modules replacing
attention. We train models with inner dimension 32, and MLP dimension 128. For all the synthetics, we use
a learning rate of 5e-4 and a weight decay of 0.1. We sample 5000 training examples and 500 test examples
from the same distribution, and we train for 200 epochs. Again, we use embedding dropout of 0.1 and residual
dropout of 0.0.
E.2
Model Architecture
For our 125M models, we use 12 layers, with hidden dimension 1024, and MLP dimension 4096. For our
355M models, we use 24 layers, with the same hidden dimension and MLP dimension. The 1.3B models have
24 layers, with hidden dimension 2048, and MLP dimension 8192. The 2.7B models have 32 layers, hidden
dimension 2560, and MLP dimension 10240. The hybrid models have 12, 16, 16, and 20 heads for the 125M,
355M, 1.3B, and 2.7B models, respectively. The 125M hybrid model has an attention layers at layers 1 and 7,
the 355M and 1.3B hybrid models have attention layers at layers 1 and 13, and the 2.7B hybrid model has
attention layers at layers 10 and 21. For both our hybrid models and our H3 models, we use SSM state size
64. Our hybrid model uses head dimension 1 for H3, while our pure H3 model uses head dimension 8. We
run models with mixed-precision training, with bf16 for the MLP’s and attention. When training language
models, we use fp32 for the FFTConv.
E.3
OpenWebText Training
For the 125M models trained on OpenWebText, we follow the training recipe of the Megatron-LM repo.
We use an eﬀective batch size of 512, and use gradient accumulation to ﬁt into available GPU memory.
We use the AdamW optimizer, with learning rate 6e-4 for GPT-2 small and 1.5e-4 for GPT-2 medium, and
weight decay of 0.1. All models are trained with the same hyperparameters for 100K steps. We run all
implementations with mixed-precision training (PyTorch AMP). We train models with sequence length 1024.
We use the Openwebtext dataset, with the GPT-2 BPE tokenizer. We randomly select 0.5% of the dataset
as the validation set, with the rest being used as training set. This random selection of validation set is done
once, and all models are evaluated on the same validation set.
E.4
The Pile Training
For the 125M and 355M models trained on the Pile, we follow the training recipe of GPT-3. We use batch
size 256, with sequence length 2048. We train our models for 800K steps. We use residual dropout 0.0 and
embedding dropout 0.1. We use the AdamW optimizer, with learning rate 6e-4 for the 125M model and 3e-4
for the 355M model, and a weight decay of 0.1. We use a cosine schedule with 8000 steps for linear warmup,
and decay the learning rate to 10% by 300B tokens, then continue training at 10% learning rate for another
100B tokens. We suspect that there exist better hyperparameters for H3 language models, but we did not
have the resources to tune them.
For the 1.3B models, we double the batch size to 512 (with sequence length 2048), again following the
training recipe of GPT-3. The number of training steps are halved so that we train on the same number of
tokens.
For the Pile dataset, we again use the GPT-2 BPE tokenizer, similar to GPT-3 and GPT-Neo.
E.5
SuperGLUE
We follow the prompts used in the GPT-3 paper [7]. For rank classiﬁcation on the binary classiﬁcation
tasks, we use yes/no for WSC, WIC, MultiRC, and BoolQ, and we use true/false for RTE. For CB, we use
true/false/neither as the three choices. For COPA and ReCoRD, we use the continuations provided by the
task.
E.6
Hardware
All models were trained on either a single 16xA100-40GB node or a cluster of 8xA100-80GB nodes.
21

F
Additional Experiments
F.1
LRA Accuracy
We evaluate the accuracy of H3 on LRA. We compare accuracy to S4D [26], since H3 uses an S4D kernel as
a component in its layer. We use the same hyperparameters as S4D, and make the layer bidirectional by
making two copies and running them in opposite directions.
Table 9: LRA performance of H3 compared to S4D.
Model
ListOps
Text
Retrieval
Image
Pathﬁnder
Path-X
Avg
S4D [26]
58.3
87.3
90.7
87.5
93.6
92.3
85.0
H3
57.5
88.2
91.0
87.3
93.0
91.8
84.8
Table 9 shows that H3 performs well on the LRA benchmark, even thought it was designed for autoregressive
language modeling. H3 outperforms S4D on two of the LRA tasks, and comes within 1 point on the others.
F.2
WikiText103
We train 125M-sized models on WikiText103 [43] and compare their test PPL to transformers, as well as
other variants of eﬃcient or long-range attention. We use the same hyperparameters and setup as training
on OpenWebText. We also provide results from Transformer-XL and Perceiver AR for context, though the
results may not be directly comparable due to diﬀerences in model size and tokenizer.
Table 10: Test PPL on WikiText103.
Models
PPL
Transformer (125M)
18.6
Hybrid H3 (125M)
18.5
Performer (125M) [9]
26.8
Reformer (125M) [39]
26.0
Linear Attention (125M) [35]
25.6
Perceiver AR (358M) [30]
18.4
Transformer-XL (285M) [13]
18.4
Table 10 shows that the Hybrid H3 model is competitive with Transformers of the same size, as well
as larger models such as the 358M Perceiver AR and 285M Transformer-XL. The hybrid H3 model also
signiﬁcantly outperforms transformers with performer, reformer, and linear attention.
We note that the Transformer-XL and Perceiver AR PPl numbers are from the original papers, and may
not be directly comparable to our results. In particular, they use a tokenizer with a diﬀerent vocab size,
which means that the PPLs are not directly comparable. In addition, the larger vocab size necessitates a
change in the model (adaptive softmax) that may aﬀect performance. The top ﬁve numbers in Table 10 are
trained with the same setup and are directly comparable to each other.
F.3
PG-19
We evaluate models trained on the PG-19 dataset [54], a natural language dataset comprised of texts from
books. We compare the performance of Hybrid H3 compared against Transformers and linear attention. We
use the same setup as evaluating on OpenWebText.
Table 11: Test PPL on PG-19.
Models
PPL
Transformer (125M)
17.0
Hybrid H3 (125M)
16.2
Linear Attention (125M) [35]
19.1
Table 11 shows that Hybrid H3 outperforms transformers and linear attention.
22

F.4
Length Extrapolation
One property of SSMs is that they can naturally extrapolate to sequence lengths longer than those seen
during training. We use the synthetic associative recall task to demonstrate that H3 maintains this capability.
To do so, we train a two-layer H3 model on sequences of length 20 drawn from the associative recall synthetic
language. Then, we evaluate accuracy of the last token prediction on sequences of length 20 and 40.
Table 12: Accuracy of an H3 model trained for associative recall on sequences of length 20, evaluated on sequences of
length 20 and 40.
Models
Acc, seqlen 20
Acc, seqlen 40
H3
99.8
98.4
Table 12 shows that H3 maintains accuracy on sequences of length 40, which is twice the length of the
training sequences.
F.5
Scaling in Number of Tokens
We evaluate how well a Hybrid H3 model scales with the number of tokens seen during training, compared to
a Transformer. For these experiments, we train a 125M Hybrid H3 model and a 125M Transformer on the
Pile for 5B, 10B, and 15B tokens. We use a learning rate of 6e-4, adjusting the warmup to be 1% of the total
training time, and adjusting the decay rate to decay the learning rate to 6e-5 by the end of training.
Table 13: Test PPL on the Pile for models trained with fewer tokens.
Train Tokens
Hybrid H3 (125M)
Transformer (125M)
5B
11.8
12.7
10B
10.7
11.3
15B
10.2
10.7
Table 13 shows the results. Both the Hybrid H3 model and Transformer model improve as the number of
training tokens increases.
F.6
H3 Language Model
Table 14: Zero-shot performance on SuperGLUE with rank classiﬁcation. Best results for each model size in bold.
Model
WSC
WIC
RTE
CB
MultiRC
ReCoRD
BoolQ
COPA
Average
OPT-125M
39.4
52.0
48.7
37.4
58.9
44.9
59.6
60.0
50.1
GPT-Neo-125M
36.5
53.6
53.1
41.1
59.9
39.6
62.2
60.0
50.8
H3-125M
61.5
50.0
53.1
41.1
4.6
15.8
46.4
51.0
40.4
Hybrid H3-125M
39.4
51.4
59.2
48.2
51.4
55.0
59.6
67.0
53.9
We report the results of a pure H3 language model on NLP evaluations. We train a 125M model on the
Pile for 400B tokens. Tables 14 and 15 show zero-shot and few-shot performance on SuperGLUE, respectively.
F.7
Generation Performance
We report results on SuperGLUE for generation. Instead of taking rank classiﬁcation, we instead let the
model generate a response, and we search for the gold label (i.e., “yes” or “no” for the yes/no questions)
in the output. Tables 16 and 17 report the results. The trends for few-shot learning match with the logit
results, but the hybrid and H3 models perform very poorly in zero-shot performance on some tasks. In these
cases, the models tend to generate long text responses that are not relevant to the answer. The few-shot
learning examples help the models generate answers in a parsable format.
F.8
Non-Text Sequence Modeling
We show that H3 outperforms Transformers on two non-text sequence modeling tasks: raw speech classiﬁcation
and seizure classiﬁcation over raw EEG signals. H3 sets state-of-the-art performance on seizure classiﬁcation
23

Table 15:
3-shot performance on SuperGLUE with rank classiﬁcation. Best results for each size in bold, second best
underline.
Model
WSC
WIC
RTE
CB
MultiRC
ReCoRD
BoolQ
COPA
Average
OPT-125M
36.5
50.2
47.3
44.6
57.9
44.9
41.9
60.0
47.9
GPT-Neo-125M
38.5
50.0
53.1
17.9
56.3
39.6
62.1
60.0
47.2
H3-125M
63.5
50.0
52.3
48.2
32.6
15.8
37.8
51.0
43.9
Hybrid H3-125M
43.3
49.1
58.1
51.8
48.9
55.0
56.1
67.0
53.7
Table 16: Zero-shot performance on SuperGLUE. Best results for each size in bold, second best underline.
Model
WSC
WIC
RTE
CB
MultiRC
ReCoRD
BoolQ
COPA
Average
OPT-125M
36.5
48.4
49.8
8.9
39.1
44.9
45.9
60.0
41.7
GPT-Neo-125M
27.9
11.3
45.8
8.9
19.1
39.6
56.4
60.0
33.6
Hybrid H3-125M
0.0
0.0
47.3
8.9
4.4
55.0
47.6
67.0
28.8
GPT-2 medium (355M)
50.0
50.2
16.2
21.4
10.5
53.3
38.4
65.0
38.1
OPT-350M
41.3
34.8
49.5
16.1
23.6
51.4
39.7
60.0
39.6
Hybrid H3-355M
22.1
21.5
47.3
8.9
17.1
62.3
44.4
69.0
36.6
and matches S4 on speech classiﬁcation—which suggests that H3, or one of its hybrids, may be a strong
candidate for a multimodal foundation model. Appendix E gives experimental details, and Appendix F gives
an additional experiment on brain fMRI data.
Seizure Classiﬁcation from EEG
Seizures, which are characterized by uncontrolled brain activity, are
some of the most common neurological disorders [20]. Chronic seizures, or epilepsy, cause a range of psychiatric
and psycho-social disorders and impact the lives of roughly one percent of the global population [37]. The
ﬁrst step to treating epilepsy is manual analysis of scalp EEG by board-certiﬁed neurologists. However, the
vast amount of EEG data produced by each patient (which can be up to days of data) makes manual EEG
analysis a costly and time-consuming process.
To mitigate the costs associated with EEG monitoring, recent deep learning techniques have began to show
promise in ﬂagging abnormal EEG segments for potential seizure events [57]. A challenge with classifying
EEG data is the trade-oﬀbetween increasing input sequence length, where more context has been shown to
improve seizure classiﬁcation performance [55], with the increased diﬃculty of training deep learning models
on long sequences (e.g., an EEG signal sampled at 200Hz produces 12,000 time steps per minute). As a
result, many techniques involve domain-specialized models and pre-processing steps, such as FFT transforms
and graphical representations [58].
We use the largest publicly available EEG corpus, TUSZ v1.5.2 [56], which includes 5,612 EEG signals
from 636 patients, with 3,050 annotated seizures. Signals are segmented into 60-second clips, and split into
train/val/test by patient. The train set contains 39765 clips, the val set contains 4351 clips, and the test set
contains 10001 clips.
We evaluate binary seizure classiﬁcation of 60-sec EEG clips, sampled at 200Hz, with 19 electrodes:
x ∈R12,000×19 and y ∈{0, 1} on the TUSZ v1.5.2 [56] corpus. Transformers cannot process the long sequence
length of EEG signals without running out of GPU memory, whereas H3 can—and sets state-of-the-art
performance.
Raw Speech Classiﬁcation
The SC10 speech commands task [64] contains raw audio signals one second
in length, sampled at 16kHz. Similarly to EEG signals, Transformers cannot process the long sequence length.
Table 19 shows that H3 comes within half a point of S4, the state-of-the-art method.
Functional Magnetic Resonance Imaging Data
Functional Magnetic Resonance Imaging (fMRI) data
are typically represented in four dimensions, indicating the measured blood-oxygen-level-dependent (BOLD)
signal in temporal sequences S = {V1, ..., Vt} of 3-dimensional volumes V ∈Rx×y×z, each indicating the
BOLD signal for all spatial locations of the brain (as deﬁned by three spatial dimensions x, y, and z). A key
challenge for the analysis of fMRI data is the high dimensionality and low sample size of its datasets, which
typically contain many hundred thousand dimensions (i.e., voxels) for each of several hundred volumes V in
each of tens to hundreds of sequences S. In this setting, where the number of features exceed the number of
samples, standard machine learning approaches are prone to overﬁtting.
In spite of the low sample size of individual datasets, neuroimaging research can be considered as recently
entering a big data era because researchers more frequently share their collected datasets publicly [41]. The
24

Table 17: 3-shot performance on SuperGLUE with generation. Best results for each size in bold, second best underline.
Model
WSC
WIC
RTE
CB
MultiRC
ReCoRD
BoolQ
COPA
Average
OPT-125M
36.5
49.1
47.3
33.9
35.5
44.8
38.5
60.0
43.2
GPT-Neo-125M
38.5
50.0
53.1
42.9
22.5
39.7
61.2
68.0
47.0
H3-125M
0.0
0.0
47.3
8.9
0.0
15.4
37.8
53.0
20.3
Hybrid H3-125M
43.3
49.1
58.1
41.1
40.3
55.2
49.5
67.0
50.5
GPT-2 medium (355M)
36.5
50.5
47.3
28.6
35.3
53.1
37.8
63.0
44.0
OPT-350M
37.5
50.0
46.2
41.1
40.6
51.3
39.4
59.0
45.6
Hybrid H3-355M
42.3
47.5
50.5
37.5
57.5
61.4
45.4
73.0
51.9
Table 18: Performance (AUROC) on 60s seizure classiﬁcation from raw EEG (sequence length 12000).
H3
Transformer
Dense-CNN
CNN-LSTM
LSTM
1D-CNN
83.2
x
78.0
68.6
69.3
69.7
availability of these data open up the opportunity for pre-training in neuroimaging at scale, as recently
demonstrated by [60], enabling models to utilize the knowledge that can be learned from public functional
neuroimaging data for the analysis of individual datasets. Speciﬁcally, [60] evaluate the performance of
several self-supervised learning frameworks for functional neuroimaging data by ﬁrst pre-training models on a
broad fMRI dataset spanning 11, 980 fMRI runs from 1, 726 individuals across 34 datasets and subsequently
adapting the pre-trained models to two downstream mental state decoding datasets (namely, the HCP [61] and
MDTB [38] datasets). In mental state decoding, predictive models are tasked with identifying (i.e., decoding)
some set of mental states (e.g., answering questions about a prose story or math problem) from measured brain
activity. The authors ﬁnd that a GPT-based model, pre-trained in a causal learning framework, performs
best in decoding the 20 (HCP) and 26 (MDTB) mental states of the two downstream datasets.
To evaluate the performance of H3 on fMRI data, we replicate this analysis, using the up- and downstream
fMRI datasets that were published by [60], treating H3 as a drop-in replacement for the GPT model. To
alleviate the high dimensionality challenge of fMRI data, and due to the generally high spatial correlation
of brain activity, the original authors have reduced the volumetric time series S to a set Θ ∈θ1, ..., θn of
n = 1, 024 functionally-independent brain networks θ (as deﬁned by the dictionaries of functional modes
(DiFuMo) brain atlas [12]), each describing the BOLD signal for some subset of voxels vx,y,z ∈V , such that
the resulting sequences X ∈Rt×n describe the activity pattern of each brain network n for time points t.
In line with [60], we ﬁrst pre-train models f(·) to predict the distribution of brain activity for the next time
point j of an input sequence X, using a mean absolute error (Lrec) training objective given the model’s output
ˆX ∈Rt×n: Lrec = 1
n
Pn
i=1 |Xj,i −ˆXj,i|; ˆXt,n = bn +P
n f(EX)t,ewe,n; EX
t,e = ET R +Epos +be +P
n Xt,nwn,e.
Here, ET R ∈Re and Epos ∈Re represent learnable embeddings for each possible time point and position of
an input sequence (for details, see [60]). As the sampling frequency of fMRI is variable, the same position
of an input sequence can correspond to diﬀerent time points. Note that f(·) processes the input in a
lower-dimensional embedding representation EX ∈Rt×e (with e = 768 dimensions).
We evaluate two model architectures for f(·), namely, the GPT variant used in [60], with 4 hidden layers
and 12 attention heads, and a corresponding H3 variant with 4 hidden layers (with H = 64 and m = 1;
see section 3). For both models, the sequence of hidden-states outputs of the last model layer are used to
determine ˆX.
Just as [60], we randomly divide the upstream data into distinct training and validation datasets by
randomly designating 5% of the fMRI runs of each of the 34 upstream datasets as validation data (at a
minimum of 2 runs per dataset) and using the rest of the runs for training. During upstream learning, we
then randomly sample sequences of 10 to 100 time points from the fMRI runs and train models with the
ADAM optimizer (with β1 = 0.9, β2 = 0.999, and ϵ = 1e−8 ) for 5, 000 steps at a mini-batch size of 512, and
a learning rate of 5e−4. We apply a linear learning rate decay schedule (with a warm-up phase of 1% of the
total number of training steps), gradient norm clipping at 1.0, and L2-regularisation (weighted by 0.1). We
also apply dropout at a rate of 0.1 for the GPT-based model (based on [60]) and evaluate three dropout rates
for H3: 0.1, 0.2, and 0.3.
We ﬁnd that the H3 variant trained with 0.2 dropout performs on par with the GPT model, in terms of
mean absolute error (Fig. 3), and therefore continue all further analyses with this model variant. We also ﬁnd
that both models exhibit almost identify Lrec error distributions throughout the brain, with relatively higher
errors in the posterior parietal, occipital, and cingulate cortices as well parts of the limbic system (Fig. 4).
To adapt the pre-trained models to mental state decoding, we add a learnable classiﬁcation embedding
Ecls ∈Rn to the end of input sequences X and forward the model’s prediction f(EX) to a decoding
25

Table 19: SC 10-class classiﬁcation on raw audio (sequence length 16000).
H3
S4
WaveGan-D
Transformer
Performer
CKConv
97.04
97.50
96.25
x
30.77
71.66
Figure 3:
Upstream mean absolute error (Lrec) in training and evaluation datasets over the course of model training.
head p(·), composed of a dense hidden layer with e model units (one for each embedding dimension, with
tanh activation) as well as a softmax output layer with one model unit i for each considered mental
state in the data. Accordingly, we adapt models by optimizing a standard cross entropy loss objective:
Lcls = −P
i yi log p(f(EX))i, where yi indicates a binary variable that is 1 if i is the correct mental state
and 0 otherwise.
During downstream adaptation, we begin training with the respective pre-trained model parameters and
then allow all parameters to change freely. Similar to [60], we randomly split each of the two downstream
datasets into distinct training and test datasets, each comprising 40 (HCP) or 10 (MDTB) distinct individuals.
We adapt models for 750 steps at a mini-batch size of 256 and a learning rate of 5e−5 (otherwise using the
same learning parameters as for upstream training). Importantly, we repeat each downstream training run 20
times using diﬀerent random seeds, leading to diﬀerent random splits of the data and variability in other
non-deterministic training factors (such as random initialization and data shuﬄing).
As for the upstream data, the H3 and GPT-based models generally perform on par in their mental state
decoding performances in the two left-out test datasets (Table 20).
Table 20:
Downstream adaptation performance of models pre-trained on fMRI data, averaged over 20 training runs
with varying random seeds. F1-scores are macro-averaged.
Dataset
Model
Acc. (±95%CI)
F1 (±95%CI)
HCP
GPT
88.44 (±0.39)
87.24 (±0.39)
H3
88.75 (±0.33)
87.16 (±0.37)
MDTB
GPT
89.47 (±0.44)
88.74 (±0.54)
H3
88.25 (±0.45)
85.76 (±0.61)
26

Figure 4: Mean absolute error (Lrec) of the ﬁnal pre-trained models for each voxel of the brain projected onto the
inﬂated cortical surface of the FsAverage template [19].
27
