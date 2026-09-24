# Mamba 3 Improved Sequenc Modeling Principles 2025

> Source: `Mamba_3_Improved_Sequenc_Modeling_Principles_2025.pdf`

---

      Under review as a conference paper at ICLR 2026




000
001
      MAMBA-3: IMPROVED SEQUENCE MODELING USING
002   STATE SPACE PRINCIPLES
003
004
005    Anonymous authors
006    Paper under double-blind review
007
008
009                                                    ABSTRACT
010
011             The recent scaling of test-time compute for LLMs has restricted the practical deployment
012             of models to those with strong capabilities that can generate high-quality outputs in an
013             inference-efficient manner. While current Transformer-based models are the standard,
014             their quadratic compute and linear memory bottlenecks have spurred the development
015             of sub-quadratic models with linear-scaling compute with constant memory requirements.
016
                However, many recent linear-style models lack certain capabilities or lag behind in quality,
                and even their linear-time inference is not hardware-efficient. Guided by an inference-first
017
                perspective, we introduce three core methodological improvements inspired by the state-
018
                space model viewpoint of linear models. We combine a: 1) more expressive recurrence, 2)
019             complex state update rule that enables richer state tracking, and 3) multi-input, multi-output
020             formulation together, resulting in a stronger model that better exploits hardware parallelism
021             during decoding. Together with architectural refinements, our Mamba-3 model achieves
022             significant gains across retrieval, state-tracking, and downstream language modeling tasks.
023             Our new architecture sets the Pareto-frontier for performance under a fixed inference
024             budget and outperforms strong baselines in a head-to-head comparison.
025
026   1    INTRODUCTION
027
      Test-time compute has emerged as a key driver of progress in AI, with techniques like chain-of-thought
028   reasoning and iterative refinement demonstrating that inference-time scaling can unlock new capabilities (Wu
029   et al., 2025; Snell et al., 2024). This paradigm shift makes inference efficiency (Kwon et al., 2023; Li
030   et al., 2024) paramount, as the practical impact of AI systems now depends critically on their ability to
031   perform large-scale inference during deployment. Model architecture design plays a fundamental role in
032   determining inference efficiency, as architectural choices directly dictate the computational and memory
033   requirements during generation. While Transformer-based models (Vaswani et al., 2017) are the current
034   industry standard, they are fundamentally bottlenecked by linearly increasing memory demands through the
035   KV cache and quadratically increasing compute requirements through the self-attention mechanism. These
036
      drawbacks have motivated recent lines of work on sub-quadratic models, e.g., state-space models (SSMs),
      which, despite utilizing only constant memory and linear compute, have comparable or better performance
037
      than their Transformer counterparts. Models that benefit the most from this new scaling paradigm perform
038
      well on the following three axes: (i) quality, (ii) capability, and (iii) inference efficiency.
039
040   Recent model architectures have tried to strike a balance between the three, but many fall short on at least
041   one of these three axes. In particular, Mamba-2 and Gated DeltaNet, which have gained significant traction
042   and adoption due to their inference efficiency, made architectural design choices that enable their linear
043   compute requirements but sacrifice quality and capabilities (Dao & Gu, 2024; Yang et al., 2025a). For
044   example, Mamba-2 was developed to improve training speed and simplicity over Mamba-1 (Gu & Dao,
045   2024), opting out of more expressive parameterizations of the underlying SSM and hindering the quality
046
      of the model (Dao & Gu, 2024). Linear attention-style models (Katharopoulos et al., 2020) have also been
      shown to lack certain capabilities, with poor state-tracking abilities, e.g., determining parity of bit sequences,
047
      being one of the most notable (Grazzi et al., 2025; Sarrof et al., 2024). In addition, despite these sub-quadratic
048
      models retaining linear inference, their inference itself is not hardware efficient. Because these algorithms
049   were developed from a training perspective, their decoding phase has low arithmetic intensity (the ratio
050   of FLOPs to memory traffic), resulting in large portions of hardware remaining idle.
051
052   To develop more performant models from an inference-first paradigm, in this paper we introduce three
053   core methodological changes, influenced by a SSM-centric viewpoint of sub-quadratic models, on top of
      Mamba-2. While many recent models fall into the linear attention framework (Dao & Gu, 2024; Yang et al.,


                                                              1
      Under review as a conference paper at ICLR 2026




054
      2025a; Sun et al., 2023), we find that the classical SSM toolbox (Kalman, 1960; Gopal, 1993) leads to natural
055   interpretations and improvements on modeling.
056
057   Trapezoidal Discretization. We discretize the underlying continuous-time dynamical system with a
058   trapezoidal methodology. The final recurrence is a more expressive superset of Mamba-2’s recurrence
059   and can be viewed as a convolution. We combine this new discretization with applied biases on the B,C,
060
      inspired by Yu & Erichson (2025), and find that their synergy is able to empirically replace the short causal
      convolution in language modeling.
061
062   Complexified State-Space Model. By viewing the underlying SSM of Mamba-3 as complex-valued, we
063   enable a more expressive state update compared to Mamba-2. This change in update rule, designed to be
064   lightweight for training and inference, overcomes the lack of state-tracking ability common for many current
065   linear models. We highlight that our complex-valued update rule is equivalent to a data-dependent rotary
066   embedding and thus can be calculated efficiently (Su et al., 2023).
067
      Multi-Input, Multi-Output SSM. To improve FLOP-efficiency during decoding, we shift from
068   outer-product-based state update to matrix-multiplication-based state update. In view of the signal processing
069   foundations of SSMs, such a transition exactly coincides with the generalization from a single-input
070   single-output (SISO) sequence dynamic to a multiple-input multiple-output (MIMO) one. Here, we found
071   that MIMO is particularly suitable for inference, as the extra expressivity allows for more compute during
072   state update, without increasing the state size and hence compromising speed.
073
      These three SSM-centric methodological changes are core to our Mamba-3 mixer primative. We also make
074
      adjustments to the overall architecture to ensure more similarity to the baseline Transformer architecture.
075
      Mamba-3 swaps the pre-output projection norm with the more common QK-normalization (Team et al.,
076   2025; OLMo et al., 2025) and makes the short convolution, a common component found in many other
077   sub-quadratic models (Gu & Dao, 2024; Yang et al., 2025a; von Oswald et al., 2025), optional.
078
079   We empirically validate our new model on a suite of synthetic and language-modeling tasks.
080
      • Better Quality. Mamba-3 matches or outperforms Mamba-2 and other open-source architectures on
081     standard downstream language modeling evaluations. For example, Mamba-3-1.5B’s average accuracy
082     on all downstream tasks is better than that of its Transformer, Mamba-2, and Gated DeltaNet counterparts.
083
084   • Better Capability. Mamba-3’s complexification of the SSM state enables the model to solve synthetic
085     state-tracking tasks that Mamba-2 cannot. We empirically demonstrate that the efficient RoPE-like
086
        calculation is able to near perfectly solve arithmetic tasks, while Mamba-3 without RoPE and Mamba-2
        perform not better than random guessing.
087
088   • Better Inference Efficiency. Mamba-3’s MIMO variant retains the same state size while enabling better
089     hardware utilization compared to standard Mamba-3 and other models. Its improved performance without
090     increasing memory requirements pushes the pareto-frontier of inference efficiency.
091
092
      2     PRELIMINARIES
093   2.1   NOTATION
094
      Scalars are denoted by plain-text letters (e.g., x,y). Tensors, including vectors and matrices, are denoted
095
      by bold letters (e.g., h,C). The shape of the tensor can be inferred from the context. We denote the input
096   sequence length as T , the model dimension as D, and the SSM state size as N. For time indices, we use
097   subscripts (e.g., xt for the input at time t). The Hadamard product between two tensors is denoted by ⊙.
098   For a vector of size v ∈ Rd, we denote Diag(v) ∈ Rd×d as the diagonal matrix with the vector v as the
                                                                                            ×    Qt
099   diagonal, and for products of scalars across time steps, we use the notation αt···s =αt:s = i=sαi.
100
101
      2.2   SSM PRELIMINARIES
102   State Space Models (SSMs) describe continuous-time linear dynamics via
103
                          ḣ(t)=A(t)h(t)+B(t)x(t),                          y(t)=C(t)⊤h(t),
104
105   where h(t)∈RN is the hidden state, x(t)∈R the input, and A(t)∈RN×N , B(t),C(t)∈RN . For discrete
106   sequences with step size ∆t, Euler’s discretization gives the recurrence
107                         ht =e∆tAt ht−1 +∆tBtxt,                             yt =C⊤
                                                                                     t ht .




                                                            2
      Under review as a conference paper at ICLR 2026

                                                                                            𝑡!



                                          𝛾'
                                                                                     ≈ !𝑒 !!(#!$%) 𝐵 𝜏 𝑥 𝜏 𝑑𝜏
                1
108                                                                                  𝑡!"#
                ×
               𝛼!:!    1                  𝛽!    𝛾!
109
110
      ℳ= 𝛼      ×      ×
                      𝛼%:%                      𝛽%     𝛾%
                %:!           1
111             ×
               𝛼&:!    ×
                      𝛼&:%    ×
                             𝛼&:&   1                  𝛽&    𝛾&
112
                                                                         𝑡!"#   𝑡!                      𝑡!"#   𝑡!
113
114   Figure 1: Left: The structured mask induced by the generalized trapezoid rule is a product of the decay and
115   convolutional mask. Right: Euler (hold endpoint) vs trapezoidal rule (average endpoints).
116
117
118
      Mamba-2’s parameterization. Mamba-2 (Dao & Gu, 2024) makes the SSM data-dependent and hardware-
      efficient by (i) projecting A=A∈R<0, and B,C∈RN from the current token and (ii) choosing transition ma-
119
      trix A=A as a data-dependent scalar. Writing αt :=e∆t At ∈(0,1) and γt :=∆t, the update becomes
120
121                                    ht =αtht−1 +γtBtxt,        yt =C⊤ t ht .
122   The scalar At < 0 is an input-dependent forget-gate (decay) αt, and the parameter selectivity ∆t jointly
123   controls the forget-gate (αt = exp(∆tAt)) and the input-gate (γt = ∆t): larger ∆t forgets faster and
124
      up-weights the current token more strongly, while smaller ∆t retains the hidden state with minimal
      contributions from the current token.
125
126   2.3   STRUCTURED MASKED REPRESENTATION AND STATE SPACE DUALITY
127   Dao & Gu (2024) show that a large class of SSMs admit a matrix form that vectorizes the time-step recurrence.
128   For instance, Mamba-2’s recurrence can be vectorized as a masked matrix multiplication,
129                                               
                                                        1
                                                                                    
130                                                α1       1
                          Y =(L⊙CB̄⊤)X=                                   ⊙CB⊤X,
                                                                                    
131
                                                     .
                                                   ..            ..                                           (1)
                                                                      .             
132
                                                        αT...1 ··· αT 1
133
134   where L∈RT ×T is the structured mask, B,C∈RT ×N , X∈RT ×D is the input to the SSM and Y ∈RT ×D
135
      is its output. Within this form, Mamba-2 can be viewed as a type of linear attention by setting Q=C, K=B,
      V =X and viewing L as a causal, data-dependent mask. When all α=1, the expression reduces to (causal)
136
      linear attention (Katharopoulos et al., 2020). A more detailed coverage of related linear-time sequence mixers
137
      can be found at Appendix A.
138
139   3     MODEL DESIGN FROM A STATE-SPACE VIEWPOINT
140   We introduce Mamba-3, with three new innovations rooted in classical state-space theory: trapezoidal
141   discretization for more expressive dynamics, complex-valued state spaces for state-tracking, and multi-input
142   multi-output (MIMO) to improve hardware utilization. These advances address the quality, capability, and
143   efficiency limitations of current sub-quadratic architectures.
144
      3.1   TRAPEZOIDAL DISCRETIZATION
145
146   Structured SSMs are naturally defined as continuous-time dynamical systems that map input functions, x(t)∈
147
      R, to output functions, y(t)∈R, for time t>0. In sequence modeling, however, the data is only observed at
      discrete time steps, which then requires applying a discretization step to the SSM to transform its continuous-
148
      time dynamics into a discrete recurrence. The preliminary step in deriving Mamba-3’s discretization is to apply
149
      the Variation of Constants formula (Proposition 5), which decomposes the hidden state into an exponentially
150   decay term and a state update term ‘information’ term dependent on the most recent inputs.
151
152   The first step in deriving the discretized recurrence is to approximate the “state-update” integral in equation 10.
153
      A straightforward choice, used in Mamba-2, is applying Euler’s rule (Süli & Mayers, 2003), which
      approximates the integral by holding the (right) endpoint constant throughout the interval (Fig. 1). This
154
      yields Mamba-2’s recurrence,
155
156
                                       ht = e∆t At ht−1 + (τt −τt−1)e(τt −τt )At Btxt
157                                      ≈ e∆t At ht−1 + ∆tBtxt.                                                     (2)
158   However, Euler’s rule provides only a first-order approximation to the “state-update” integral: local truncation
159   error is O(∆t2), which accumulates across steps to yield a global error of O(∆t) over the sequence. In
160   contrast, we adopt a generalized trapezoidal rule, which provides a second-order accurate approximation
161   of the integral, offering improved accuracy over the Euler’s rule. Specifically, it approximates the integral with
      a data-dependent, convex combination of both interval endpoints. This generalization extends the classical


                                                              3
      Under review as a conference paper at ICLR 2026




162
      trapezoidal rule (Süli & Mayers, 2003), which simply averages the interval endpoints, by allowing for a
163   data-dependent convex combination (Fig. 1).
164
165   Proposition 1 (Generalized Trapezoidal Discretization). Approximating the state-update integral
      in equation 10 by the general trapezoidal rule yields the recurrence,
166
167
                            ht = e∆t At ht−1 + (1−λt)∆te∆t At Bt−1xt−1 + λt∆tBtxt,                (3)
168                            := αtht−1 + βtBt−1xt−1 + γtBtxt,                                   (4)
169   where λt ∈[0,1] is a data-dependent scalar, αt :=e∆t At , βt :=(1−λt)∆te∆t At , γt :=λt∆t.
170   Remark 1 (Expressivity). Our scheme is a generalization of a) The classical trapezoid rule which is recovered
171   when λt = 12 . b) Mamba-2’s Euler’s rule, which is recovered when λt =1.
172
173   Remark 2 (Error Rate). This is a second-order discretization with local truncation error O(∆t3) and global
174
      error O(∆t2) over the sequence under standard stability assumptions.
175   3.1.1    TRAPEZOIDAL DISCRETIZATION IS A CONVOLUTIONAL MASK
176   We can view the generalized trapezoidal discretization as applying a data-dependent convolution of size
177   two on the projected input, Btxt, to the SSM. We now show that a similar vectorization to Equation (1)
178   holds with the generalized trapezoidal discretization. Unrolling the recurrence starting from h0 =γ0B0x0
179   results in hT =αT ···2(γ0α1 +β1)B0x0 +···+γT BT xT .
180   Unrolling these rows shows that the mask induced by the trapezoidal update is no longer a fixed averaging
181   of endpoints (as in the classical trapezoidal rule), but a data-dependent convex combination of the two interval
182   endpoints. In the SSD representation, this corresponds to a mask L:
183                    γ0                                        1                     γ0
                                                                                                       
184          (γ0α1 +β1)                                 α1          1            β1                    
             α (γ α +β )                γ2             α α                        0      γ2
185
                                                                                                            
             2 0 1 1                                  = 2 1                                            . (5)
186
                       .
                        ..                    ..               .
                                                                 .           ..       .
                                                                                        .           ..      
                                                .             .              .     .              .    
187
              αT ···2(γ0α1 +β1)               ··· γT          αT ···1        ··· 1      0           ··· γT
188
      Here, the first factor is precisely the lower-triangular decay mask from Mamba-2, while the second factor
189   encodes the size two convolution induced by the trapezoidal rule through the coefficients (βt,γt). We provide
190   a rigorous proof for this decomposition in Appendix B.2.
191
      3.2     COMPLEX-VALUED SSMS
192
193   Modern SSMs are designed with efficiency as the central goal, motivated by the need to scale to larger models
194   and longer sequences. For instance, successive architectures have progressively simplified the state transition
195
      matrix: S4 (Gu et al., 2022a) used complex-valued Normal plus Low Rank (NPLR) matrices, Mamba (Gu
      & Dao, 2024) reduced this to a diagonal of reals, and Mamba-2 (Dao & Gu, 2024) further simplified it
196
      to a single scalar. Although these simplifications largely maintain language modeling performance, recent
197
      works (Merrill et al., 2025; Sarrof et al., 2024; Grazzi et al., 2025) have shown that they degrade the
198   capabilities of the model on simple state-tracking tasks such as parity and modular arithmetic, which can
199   be solved by a one-layer LSTM.
200
201
      This limitation, formalized in Theorem-1 of (Grazzi et al., 2024), arises from restricting the eigenvalues
      of the transition matrix to real numbers, which cannot represent “rotational”
                                                                                P hidden state dynamics. For
202
      instance, consider the parity function on binary inputs {0,1}, defined as t xt mod 2. This task can be
203
      performed using update: ht =R(πxt)ht−1, where R(·) is a 2-D rotation matrix. Such rotational dynamics
204   cannot be expressed with real eigenvalues.
205
206
      To recover this capability, we begin with complex SSMs (6), which are capable of representing state-tracking
      dynamics. We show that, under discretization (Proposition 5), complex SSMs can be formulated as a real SSMs
207
      with a block-diagonal transition matrix composed of 2×2 rotation matrices (Proposition 2). We then show that
208
      this is equivalent to applying data-dependent rotary embeddings on both the input and output projections B,C
209   respectively. This result establishes a theoretical connection between complex SSMs and data-dependent RoPE
210   embeddings (Proposition 3). Finally, this allows for an efficient implementation of the SSM via “RoPE trick”,
211   enabling efficient complex-valued state transition matrix with minimal overhead over real SSMs.
212
      Proposition 2 (Complex-to-Real SSM Equivalence). Consider a complex-valued SSM
213                                                                    
214
                             ḣ(t)=Diag A(t)+iθ(t) h(t)+ B(t)+iB̂(t) x(t),                                        (6)
                                                   ⊤    
215                           y(t)=Re C(t)+iĈ(t) h(t) ,


                                                             4
      Under review as a conference paper at ICLR 2026




216
      where h(t) ∈ CN/2, θ(t),B(t),B̂(t),C(t),Ĉ(t) ∈ RN/2, and x(t),A(t) ∈ R. Under Euler discretization,
217   this system is equivalent to a real-valued SSM
218
219
                                              ht =e∆t At Rtht−1 +∆tBtxt,                                     (7)
220                                           yt =C⊤
                                                   t ht ,
221
      with state ht ∈RN , projections
222                                                                    
                                            Bt                       Ct
223                                   Bt =      ∈RN ,           Ct =        ∈RN ,
224                                         B̂t                      −Ĉt
225   and a transition matrix
                                                                                               
226                          
                                          N/2
                                              
                                                                                 cos(Θ) −sin(Θ)
                   Rt = Block {R(∆tθt[i])}i=1 ∈RN×N ,                   R(Θ)=                    .
227                                                                               sin(Θ) cos(Θ)
228
229   The proof is in Appendix C.1.
230
231   Proposition 2 shows that the discretized complex SSM has an equivalent real SSM with doubled state
232
      dimension (N), and a block-diagonal transition matrix multiplied with a scalar decay, where each 2 × 2
233
      block is a data-dependent rotation matrix (et∆t ARt). We now show that the rotations can equivalently be
      absorbed into the input and output projections Bt,Ct, yielding an equivalent view that complex SSMs are
234
      real SSMs equipped with data-dependent rotary embeddings (RoPE).
235
236   Proposition 3 (Complex SSM, Data-Dependent RoPE Equivalence). Under the notation established in
237   Proposition 2, consider the real SSM defined in Eq. 7 unrolled for T time-steps. The output of the above
238   SSM is equivalent to that of a vanilla scalar transition matrix-based SSM (Eq. 2) with a data-dependent
239   rotary embedding applied on the B,C components of the SSM defined as:
240                                      Y t                            Y t         ⊤
241                  ht =e∆t At ht−1 +( R⊤     i )B  x
                                                    t t ,         y t =  (   R ⊤
                                                                               i )Ct    ht                  (8)
242                                         i=0                            i=0
                                                                               Q1
243   where the matrix production represents right matrix multiplication, e.g., i=0 Ri = R0R1. We denote
244   employing the vanilla SSM to compute the Complex SSM as “RoPE trick”.
245
      The proof is in Appendix C.2.
246
247   To observe the connection of complex SSMs to RoPE embeddings, note that in the above proposition,
248   the data-dependent rotations Ri are aggregated across time-steps and applied to C,B, which, by the State
249   Space Duality of Dao & Gu (2024), correspond to the Query (Q) and Key (K) components of Attention.
250   Analogously, vanilla RoPE (Su et al., 2023) applies data-independent rotation matrices, where the rotation
251   angles follow a fixed frequency schedule θ[i]=10000−2i/N .
252
      Remark 3 (Generality). Proposition 3 extends to the fully general case where the transition is given by any
253
      complex matrix. By the complex diagonalization theorem, such a matrix is unitarily equivalent to a complex
254
                                         
      diagonal matrix, Diag A(t)+iθ(t) with A(t) ∈ RN . However, in practice, we restrict A(t) to a scalar,
255   mirroring the simplification from Mamba to Mamba-2, to enable faster implementation by avoiding GPU
256   memory bottlenecks.
257
258
      Proposition 4 (Rotary Embedding Equivalence with Trapezoidal Discretization). Discretizing a complex
      SSM with the trapezoidal rule (Proposition 1) yields the recurrence
259
260                                           t−1
                                               Y                       Yt    
                                                     ⊤
261
                            ht =αtht−1 +βt         Ri Bt−1xt−1 +γt          R⊤
                                                                             i Bt xt ,
                                                  i=0                      i=0
262
                                       t
263
                                      Y     ⊤
                                yt =     R⊤
                                          i Ct   ht.                                                         (9)
264
                                      i=0
265
      Here Rt is the block-diagonal rotation matrix defined in Proposition 3.
266
267   The proof is in Appendix C.5.
268
269   Remark 4 (RoPE Trick). Complex SSMs discretized with the general trapezoidal rule of a complex SSM
      naturally admit the RoPE trick we established for SSMs discretized with Euler’s rule.


                                                            5
      Under review as a conference paper at ICLR 2026




270
      3.3    MULTI-INPUT, MULTI-OUTPUT
271
272   During the decoding phase of autoregressive inference, outputs are generated one token at a time, and
273
      performance is typically measured using in Tokens generated Per Second (TPS). In this metric, sub-quadratic
      models, such as Mamba-2 (Dao & Gu, 2024), have a significant advantage over standard Transformer-style
274
      attention, since they feature a fixed-size hidden state (Equation (2)) rather than maintaining a key–value
275
      (KV) cache that grows linearly with the sequence length.
276
277   TPS, however, does not explicitly factor in hardware efficiency, where we aim to be in a compute-bound
278   regime (as opposed to memory-bound) in order to fully utilize on-chip accelerators. To better characterize
279   hardware efficiency, we would need to consider the arithmetic intensity of token generation. Recall that
280
      arithmetic intensity is defined as FLOPs divided by the number of input-output bytes, for a given op. In order
      to fully utilize both the accelerators and the bandwidth, we would like the arithmetic intensity to match the
281
      ops:byte ratio of the hardware, which in the case of NVIDIA H100-SXM5, is 295.2 bfloat16 ops per second
282
      with respect to the DRAM, and 31.9 bfloat16 ops per second with respect to the SRAM [Fleetwood].
283
284   Table 2(a) shows the arithmetic intensity for a single generation in the SSM component of Mamba (with
285   respect to 2-byte data). We see that it falls far short of a compute-bound regime, and moreover it is not clear
286
      how one can adjust the existing parameters in Mamba to mitigate the lack of hardware efficiency. We note
      that this observation applies generally to other sub-quadratic models, such as causal linear attention.
287
288
       Input        Output     FLOPs Arithmetic                  Input        Output      FLOPs Arithmetic
289                                                                                             Intensity
                                     Intensity
290
                                             5pn                                                   p(4nr+2n)
291    Ht :(n,p)    yt :(p)     5pn                              Ht :(n,p)    yt :(p,r)   4nrp+
                                        2(1+2n+p+np)                                       2np  2(1+2nr+pr+np)
292    xt :(p)                          ≈2.5=Θ(1)                xt :(p,r)                      ≈2r =Θ(r)
293    at :(1)                                                   at :(1)
294    bt :(n)                                                   bt :(n,r)
       ct :(n)                                                   ct :(n,r)
295
296                  (a) SISO (2-byte data).                                  (b) MIMO (2-byte data).
297
298         Figure 2: Arithmetic Intensity for (a) SISO, (b) MIMO. Batch and head dimensions cancel out.
299
300   In light of this, we made the following simple adjustment to our recurrent relation: instead of transforming
301   the input xt ∈ Rp to state Ht ∈ Rn×p via an outer product, i.e., Ht ← atHt−1 +bt ⊗xt, we made such
302
      a transformation via a matrix product, i.e., Ht ←atHt−1 +BtX⊤     t , where Bt ∈R
                                                                                          n×r
                                                                                               and Xt ∈Rp×r are
      now matrices with an addition rank r. The emission from state to output similarly acquire an extra rank r, i.e.,
303
      Yt ∈Rr×p ←C⊤       t Ht , where Ct ∈R
                                           n×r
                                                ,Ht ∈Rn×p. This simple change increases the arithmetic intensity
304
      of recurrence, which now scales with the rank r (Figure 2(b)). Hence, by increasing r, arithmetic intensity
305   improves and shifts decode generation towards a more compute-bound regime. This increase in FLOPs during
306   decode does not compromise runtime, as the operation is bounded by the I/O of state Ht ∈Rn×p.
307
308   Moreover, moving from outer-product-based state update to matrix-product-based coincides exactly with
309
      generalizing from SISO to MIMO SSM, with the rank r being the MIMO rank. Such a generalization
      recovers a key expressive feature of SSMs in classical literature; indeed, there has been previous work,
310
      namely Smith et al. (2023), that explored MIMO SSM as a drop-in replacement of attention, albeit not in
311
      the context of Mamba and not necessarily with inference in view.
312
313   Details of the MIMO formulation for Mamba-3 are provided in Appendix D.
314   3.4    MAMBA-3 ARCHITECTURE
315
      The Mamba-3 block retains the overall layout of its predecessor while introducing several key modifications.
316
      Most notably, the SSD layer is replaced with the more expressive trapezoidal SSM defined in Proposition 4.
317
      The extra normalization layer, first introduced between Mamba-1 and Mamba-2 for training stability,
318   is repositioned to follow the B, C projection, mirroring the QK-Norm commonly used in modern
319   Transformers (Henry et al., 2020; Wortsman et al., 2023). Building on the findings of Yu & Erichson
320   (2025), which prove adding channel-specific bias to B in a blockwise variant of Mamba-1 grants universal
321   approximation capabilities, Mamba-3 incorporates a head-specific, channel-wise bias into both the B and C
322   components after its normalization. Our trapezoidal discretization complements this bias, eliminating the need
323   for the original short causal convolution and its accompanying activation function (Section 4.3). Mamba-3
      employs the SISO SSM by default, though we view its MIMO variant as a flexible option that can be toggled


                                                             6
      Under review as a conference paper at ICLR 2026




324   Table 1: Downstream language modeling evaluations on models trained with 100B FineWeb-Edu tokens.
325   Best results for each size are bolded, and second best are underlined. All models are trained with the same
326   procedure. Mamba-3 outperforms Mamba-2 and others at every model scale.
327
328       Model                 FW-Edu   LAMB.    LAMB.    HellaS.   PIQA    Arc-E    Arc-C    WinoGr.   OBQA      Average
                                 ppl ↓    ppl ↓    acc ↑   acc n ↑   acc ↑   acc ↑   acc n ↑    acc ↑    acc n ↑    acc ↑
329
          Transformer-180M       16.89    45.0     32.5     39.0     67.1    59.8     27.9      51.2      21.8      42.8
330
          Gated DeltaNet-180M    16.61    35.9     33.7     40.2     66.8    59.6     28.5      51.2      21.6      43.1
331       Mamba-2-180M           16.76    41.8     30.9     40.1     66.8    60.1     27.3      52.0      23.2      42.9
332       Mamba-3-180M           16.59    37.7     32.5     40.8     66.1    61.5     27.9      52.0      22.8      43.4
333       Transformer-440M       13.03    21.2     41.7     50.5     69.9    67.6     34.6      56.7      26.0      49.6
          Gated DeltaNet-440M    13.12    19.0     40.4     50.5     70.5    67.5     34.0      55.3      25.8      49.1
334
          Mamba-2-440M           13.00    19.6     40.8     51.7     70.6    68.8     35.0      54.1      26.0      49.6
335       Mamba-3-440M           12.87    19.6     40.2     51.7     71.9    68.9     34.4      55.8      26.0      49.8
336       Transformer-820M       11.42    15.0     44.7     57.2     72.6    71.6     39.2      57.7      26.8      52.8
337       Gated DeltaNet-820M    11.39    12.7     47.1     57.5     72.6    72.5     38.8      57.9      30.6      53.9
          Mamba-2-820M           11.35    13.8     45.0     58.1     72.5    72.3     38.7      56.8      30.2      53.4
338       Mamba-3-820M           11.23    12.9     47.2     58.8     73.6    72.7     40.2      58.4      30.0      54.4
339
          Transformer-1.5B       10.51    11.1     50.3     60.6     73.8    74.0     40.4      58.7      29.6      55.4
340       Gated DeltaNet-1.5B    10.51    10.8     49.9     60.5     74.3    73.3     40.4      61.5      30.4      55.7
341       Mamba-2-1.5B           10.47    12.0     47.8     61.4     73.6    75.3     41.8      57.5      32.6      55.7
          Mamba-3-1.5B           10.35    10.9     49.4     61.9     73.6    75.9     42.7      59.4      32.0      56.4
342
343
344   depending on inference requirements. The overall architecture follows the Llama design (Grattafiori et al.,
345   2024), alternating Mamba-3 and SwiGLU blocks with pre-normalization.
346
347
      4      EMPIRICAL VALIDATION
348   We empirically validate our SSM-centric methodological changes through the overall Mamba-3 model on
349   a host of synthetic and real world tasks. Section 4.1 compares our SISO-variant of Mamba-3 on language
350   modeling and retrieval-based tasks, while Section 4.2 demonstrates inference efficiency of Mamba-3, and
351   MIMO Mamba-3’s benefits over SISO Mamba-3 under fixed inference compute. We ablate the impact
352
      of our new discretization and BC bias on performance and show that complexification of the SSM leads
      to previously out-of-reach capabilities in Section 4.3.
353
354   4.1     LANGUAGE MODELING
355   All models are pretrained with 100B tokens of the FineWeb-Edu dataset (Penedo et al., 2024) with the
356   Llama-3.1 tokenizer (Grattafiori et al., 2024) at a 2K context length with the same standard training protocol.
357   Training and evaluation details can be found in Appendix E.
358
      Across all four model scales, Mamba-3 outperforms popular baselines at various downstream tasks (Table 1).
359   We highlight that Mamba-3 does not utilize the short convolution that has been empirically identified as
360   an important component in many performant linear models (Allen-Zhu, 2025).
361
      4.1.1       RETRIEVAL CAPABILITIES
362
363   Beyond standard language modeling, an important measure for linear models is their retrieval ability — how
364   well they can recall information from earlier in the sequence (Arora et al., 2025a;b). Unlike attention models,
365
      which can freely revisit past context with the growing KV cache, linear models must compress context into a
      fixed-size state. This trade-off is reflected in the Transformer baseline’s substantially stronger retrieval scores.
366
      To evaluate Mamba-3 under this lens, Table 2 compares it against baselines on both real-world and synthetic
367
      needle-in-a-haystack (NIAH) tasks (Hsieh et al., 2024), using our pretrained 1.5B models from Section 4.1. We
368   restrict the task sequence length to 2K tokens to match the training setup and adopt the cloze-style format for
369   our real-world tasks to mirror the next-token-prediction objective, following Arora et al. (2025b; 2024).
370
371
      Mamba-3 is competitive on real-world associative recall and question-answering but struggles when extracting
      information from semi-structured or unstructured data. On synthetic NIAH tasks, however, Mamba-3
372
      surpasses or matches baselines on most cases and notably demonstrates markedly better out-of-distribution
373
      retrieval abilities than its Mamba-2 predecessor.
374
375
      4.2     INFERENCE EFFICIENCY
376   In this section, we investigate our methodological changes in the context of inference performance. We first
377   present our inference benchmark in Section 4.2.1; we then establish a framework for comparing the inference
      performance in Section 4.2.2. Finally, we focus on the effectiveness of MIMO in Section 4.2.3.


                                                               7
      Under review as a conference paper at ICLR 2026




378   Table 2: Retrieval capabilities measured by a mixture of real-world and synthetic retrieval tasks. Real-world retrieval tasks
379   utilize cloze variants of the original datasets and are truncated to 2K length. Mamba-3 demonstrates strong associative
380   recall and question-answering but suffers with information extraction of semi-structured and unstructured data. Mamba-3
381
      has strong needle-in-a-haystack (NIAH) accuracy and generalizes outside its trained context.
382
        Model (1.5B)     SWDE         SQUAD         FDA      TQA       NQ       Drop                            NIAH-Single-1            NIAH-Single-2              NIAH-Single-3
383
        Context Length                               2048                                    1024                      2048   4096    1024     2048   4096    1024      2048   4096
384     Transformer       48.9         46.6         58.4      67.5     31.7     26.4 100.0 100.0                               0.0    92.2 100.0       0.0    98.6      99.4    0
385     Gated DeltaNet    32.7         40.0         28.3    63.5      25.7      24.5 100.0 100.0                              99.8 100.0       93.8   49.8    83.8      68.4   34.2
386     Mamba-2           30.7         39.1         23.7    64.3      25.1      28.5 100.0 99.6                               62.0 100.0       53.8   11.8    95.8      87.4   13.4
        Mamba-3           28.5         40.1         23.4    64.5      26.5      27.4 100.0 100.0                              88.2 100.0       95.4   50.6    92.4      81.4   34.2
387
388
389
                                                                                                                        Relative Total State Size vs Pretraining Perplexity
390                                                                                                             15.0




                                                                                       Pretraining Perplexity
391
392                                                                                                             14.8
       Model                      FP32                        BF16
393                                                                                                                           Gated DeltaNet
                         dstate =64   dstate =128    dstate =64   dstate =128                                   14.6          Mamba-2
394    Mamba-2            0.295          0.409        0.127          0.203                                                    Mamba-3
       Gated DeltaNet     0.344          0.423        0.176          0.257                                      14.4          Mamba-3 MIMO
395    Mamba-3 (SISO)     0.261          0.356        0.106          0.152
396    Mamba-3 (MIMO)     0.285          0.392        0.136          0.185                                                              105
                                                                                                                                        Relative Total State Size
397
      Table 3: Latency (in milliseconds) comparison
398
      across models, precision, and dstate values. Both                                Figure 3: Exploration of state size (inference speed
399   Mamba-3 SISO and MIMO are faster than the                                        proxy) versus pretraining perplexity (performance
400   Mamba-2 and Gated DeltaNet at the commonly                                       proxy). Mamba-3 MIMO drives the-Pareto frontier
401   used bf16, dstate =128 setting.                                                  without increasing state size.
402
403
404   4.2.1       MAMBA-3 IS FAST AT INFERENCE
405
      We benchmarked the wallclock time for a single decoding step, and in a single sequence mixing layer, for
406
      each subquadratic model we consider in this paper. We adopted the standard reference code for Mamba-2 and
407   GDN, while writing our own custom kernels for the Mamba-3 step function. The result is recorded in Table
408   3. We observe that, despite having a more sophisticated SSM structure, Mamba-3 is in fact noticeably faster
409   when compared to Mamba-2, which in turn is faster than GDN, illustrating the viability of our inference-first
410   approach.
411
412   4.2.2       A PARETO FRONT FOR INFERENCE EFFICIENCY
413   For Mamba and many variants of sub-quadratic models, the generation of tokens during decoding is heavily
414   dominated by memory I/O due to the low arithmetic intensity of computing the recurrent update (c.f. Section
415   3.3). Furthermore, among the data being transferred, the latent state Ht dominates in terms of size. Indeed,
416   from Table 3, we see that the runtime scales with dstate, which configures the size of the hidden state.
417
418   As dstate dominates the decode runtime for the subquadratic models considered in this paper, we opt to use
419   it as a proxy for inference speed. By plotting the validation perplexity (itself a proxy for model performance)
420
      as a function of dstate, we aim to formulate a holistic picture about how the subquadratic models can trade
      off performance with inference speed.
421
422
      Figure 31 shows such a Pareto front for the subquadratic models considered in this paper. For each data
423
      point, we train a 440M parameter model to 17.8 billion tokens on the Fineweb-Edu dataset, where the model
424
      is configured with a dstate of {32,64,128}. As expected, we observe an inverse correlation between validation
425   loss and dstate; moreover, we noticed a general downward shift on the Pareto front moving from Mamba-2
426   to Mamba-3. A further downward shift is observed when moving from the SISO variant of Mamba-3 to
427   the MIMO variant of Mamba-3 (where we set the Mimo rank r =4 and decrease our MLP inner dimension
428   to match the parameter count). This highlights both the expressivity gain coming our methodology change
429   as well as the effectiveness of the MIMO mechanism in improving decoding efficiency.
430
431
          1
              The dstate =32 Mamba-3 MIMO did not finish training at time of submission.


                                                                                       8
      Under review as a conference paper at ICLR 2026




432   Table 4: Left: Ablations on core modeling components of Mamba-3, results on test split of dataset. A combination of
433   our BC bias and trapezoidal discretization makes the convolution optional. Right: Formal language evaluation (scaled
434   accuracy, %). Higher is better. Models are trained on short sequences and evaluated on longer lengths to test length
435
      generalization. For Gated DeltaNet we report the variant with eigenvalue range [−1,1].
436
                                                                                            Arith. w/o ↑ Arith. w/ ↑
437         Model Variant                ppl ↓         Model                     Parity ↑
                                                                                             brackets     brackets
438         Mamba-3 − bias − trap      16.68
                                                      Mamba-3                     100.00       98.51        87.75
439         Mamba-3 − bias             16.49
                                                      Mamba-3 (w/o RoPE)             2.27       1.49          0.72
440         Mamba-3                    15.72
                                                      Mamba-2                        0.90      47.81          0.88
441         Mamba-3 + conv             15.85
                                                      Gated DeltaNet [-1,1]       100.00       99.25        93.50
442           (a) Component ablation (350M).
                                                      (b) Performance comparison of various models on formal language
443                                                   tasks. Results show that unlike Mamba-2, Mamba-3 features state
444                                                   tracking ability stemming from data-dependent RoPE embeddings.
445
446
447   4.2.3     MIMO ENHANCES INFERENCE EFFICIENCY
448   With higher arithmetic intensity, MIMO increases the decoding FLOPs. Moreover, from Table 3, we observe
449   that the additional FLOPs does not have a drastic impact on the decode runtime.2 The implication is that
450   any performance gain from MIMO translates to efficiency gain in decoding, and this is in fact supported
451   by the downward shift of the MIMO Pareto curve we observed in Section 4.2.2.
452
      We aim to further verify the gain from MIMO by investigating its language-modeling capabilities. To that
453
      end, we train a 440M parameter MIMO model with MIMO rank r = 4 on 100B tokens on Fineweb-Edu
454   (i.e., same setting as the 440M parameter run in Section 4.1; we did not train 820M or 1.5B model due
455   to compute constraints). To ensure the total parameter count equals SISO, we decrease the inner dimension
456   of the MLP layers to compensate for the increase due to the MIMO projections.
457
458
      On both validation perplexity and our suite of language evaluation tasks (Table 5), we see significant gain when
      moving from SISO to MIMO. Namely, we attain a perplexity gain of 0.16 on the 100B tokens run, and Figure 3
459
      illustrates the downward shift in our validation loss. On the language evaluation front, we see significant gain on
460
      most tasks when compared to SISO, resulting in an overall gain of 1.2 point over SISO. This strongly supports
461   MIMO as a SSM-centric technique to improve model quality without compromising decoding speed.
462
463
      4.3     SSM-CENTRIC METHODOLOGICAL ABLATIONS
464   Table 4a ablates the changes made to the core SSM component, mainly the introduction of BC bias and
465   trapezoidal discretization. We report the pretraining test perplexity on models at the 440M scale, trained
466   for Chinchilla optimal tokens. We find that the bias and trapezoidal SSM synergize well and make the short
467   convolution utilized by many current linear models redundant.
468   We empirically demonstrate that data-dependent RoPE in Mamba-3 enables state tracking. Following
469   Grazzi et al. (2025), we evaluate on tasks from the Chomsky hierarchy—Parity, Modular Arithmetic
470   (without brackets), and Modular Arithmetic (with brackets)—and report scaled accuracies in Table 4b.
471   Mamba-3 solves Parity and Modular Arithmetic (without brackets), and nearly closes the accuracy gap
472   on Modular Arithmetic (with brackets). In contrast, Mamba-3 without RoPE and Mamba-2 fail to learn
473   these tasks. We use the state-tracking–enabled Gated DeltaNet variant of Grazzi et al. (2025) and observe
474   that Mamba-3 is competitive—matching parity and approaching its performance on both modular-arithmetic
475
      tasks. Experimental settings are covered in Appendix E.
476   5      CONCLUSION AND FUTURE WORK
477
      We introduce Mamba-3, an SSM model with three axes of improvement rooted in SSM principles: (i) improved
478
      quality, via trapezoidal discretization; (ii) new capabilities, through complex SSMs that recover state-tracking;
479
      and (iii) higher inference efficiency, with a MIMO formulation that raises arithmetic intensity. Mamba-3
480   delivers strong language modeling results and establishes a new Pareto frontier on the performance-efficiency
481   axes with respect to strong baseline models. A limitation remains in retrieval, where fixed-state architectures
482   lags attention-based models. We see hybrid Mamba-3 architectures that integrate retrieval mechanisms as a
483   promising path, alongside broader application of our design principles to linear-time sequence models.
484
         2
485        The kernel for MIMO Mamba-3 in fact fuses the MIMO projection, and so the reported wallclock time is actually an
      overestimate for the pure SSM update.


                                                               9
      Under review as a conference paper at ICLR 2026




486
      REFERENCES
487
488
      Zeyuan Allen-Zhu. Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon
        Layers. SSRN Electronic Journal, May 2025. https://ssrn.com/abstract=5240330.
489
490   Aryaman Arora, Neil Rathi, Nikil Roashan Selvam, Róbert Csordás, Dan Jurafsky, and Christo-
491     pher Potts. Mechanistic evaluation of transformers and state space models, 2025a. URL
492     https://arxiv.org/abs/2505.15105.
493
      Simran Arora, Aman Timalsina, Aaryan Singhal, Benjamin Spector, Sabri Eyuboglu, Xinyi Zhao, Ashish
494
        Rao, Atri Rudra, and Christopher Ré. Just read twice: closing the recall gap for recurrent language models,
495     2024. URL https://arxiv.org/abs/2407.05483.
496
497   Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, Dylan Zinsley, James Zou,
498     Atri Rudra, and Christopher Ré. Simple linear attention language models balance the recall-throughput
499     tradeoff, 2025b. URL https://arxiv.org/abs/2402.18668.
500   Aviv Bick, Kevin Y. Li, Eric P. Xing, J. Zico Kolter, and Albert Gu. Transformers to ssms: Distilling quadratic
501     knowledge to subquadratic models, 2025a. URL https://arxiv.org/abs/2408.10189.
502
503   Aviv Bick, Eric Xing, and Albert Gu. Understanding the skill gap in recurrent language models: The role
504
        of the gather-and-aggregate mechanism, 2025b. URL https://arxiv.org/abs/2504.18574.
505   Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical
506     commonsense in natural language, 2019. URL https://arxiv.org/abs/1911.11641.
507
508
      Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane,
        Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Be-
509
        langer, Lucy Colwell, and Adrian Weller. Rethinking attention with performers, 2022. URL
510
        https://arxiv.org/abs/2009.14794.
511
512   Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind
513     Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018. URL
514     https://arxiv.org/abs/1803.05457.
515   Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through
516      structured state space duality, 2024. URL https://arxiv.org/abs/2405.21060.
517
518   Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner.
519
        Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs, 2019. URL
        https://arxiv.org/abs/1903.00161.
520
521   Christopher Fleetwood. Domain specific architectures for ai inference.                        URL https:
522     //fleetwood.dev/posts/domain-specific-architectures.
523
      Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence
524
        Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa,
525
        Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish
526     Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL
527     https://zenodo.org/records/12608602.
528
529   Madan Gopal. Modern control system theory. New Age International, 1993.
530   Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
531     Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan,
532     Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev,
533     Arthur Hinsvark, Arun Rao, Aston Zhang, and et. al. The llama 3 herd of models, 2024. URL
534     https://arxiv.org/abs/2407.21783.
535
      Riccardo Grazzi, Julien Siems, Simon Schrodi, Thomas Brox, and Frank Hutter. Is mamba capable of
536
        in-context learning?, 2024. URL https://arxiv.org/abs/2402.03170.
537
538   Riccardo Grazzi, Julien Siems, Arber Zela, Jörg K. H. Franke, Frank Hutter, and Massimiliano
539     Pontil.  Unlocking state-tracking in linear rnns through negative eigenvalues, 2025. URL
        https://arxiv.org/abs/2411.12537.


                                                            10
      Under review as a conference paper at ICLR 2026




540
      Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2024. URL
541     https://arxiv.org/abs/2312.00752.
542
543   Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces,
544     2022a. URL https://arxiv.org/abs/2111.00396.
545   Albert Gu, Ankit Gupta, Karan Goel, and Christopher Ré. On the parameterization and ini-
546     tialization of diagonal state space models. arXiv preprint arXiv:2206.11893, 2022b. URL
547     https://arxiv.org/abs/2206.11893.
548
549
      Ankit Gupta, Albert Gu, and Jonathan Berant. Diagonal state spaces are as effective as structured state
        spaces, 2022. URL https://arxiv.org/abs/2203.14343.
550
551   Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and Yuxuan Chen. Query-key normalization for
552     transformers, 2020. URL https://arxiv.org/abs/2010.04245.
553
      Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang,
554
        and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models?, 2024.
555     URL https://arxiv.org/abs/2404.06654.
556
557   Samy Jelassi, David Brandfonbrener, Sham M. Kakade, and Eran Malach. Repeat after me: Transformers are
558     better than state space models at copying, 2024. URL https://arxiv.org/abs/2402.01032.
559   Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer.      Triviaqa: A large
560    scale distantly supervised challenge dataset for reading comprehension, 2017.    URL
561    https://arxiv.org/abs/1705.03551.
562
563
      Rudolph Emil Kalman. A new approach to linear filtering and prediction problems. 1960.
564   Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret.                           Trans-
565     formers are rnns: Fast autoregressive transformers with linear attention, 2020.                      URL
566     https://arxiv.org/abs/2006.16236.
567
      Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti,
568
        Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew
569
        Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A
570     benchmark for question answering research. Transactions of the Association for Computational Linguistics,
571     7:452–466, 2019. doi: 10.1162/tacl a 00276. URL https://aclanthology.org/Q19-1026/.
572
573   Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez,
574
       Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with
       pagedattention, 2023. URL https://arxiv.org/abs/2309.06180.
575
576   Baolin Li, Yankai Jiang, Vijay Gadepally, and Devesh Tiwari. Llm inference serving: Survey of recent
577     advances and opportunities, 2024. URL https://arxiv.org/abs/2407.12391.
578
      William Merrill, Jackson Petty, and Ashish Sabharwal. The illusion of state in state-space models, 2025.
579
        URL https://arxiv.org/abs/2404.08819.
580
581   Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal.      Can a suit of ar-
582     mor conduct electricity? a new dataset for open book question answering, 2018. URL
583     https://arxiv.org/abs/1809.02789.
584   Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling
585     Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson,
586     David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, Michal Guerquin,
587     and et. al. 2 olmo 2 furious, 2025. URL https://arxiv.org/abs/2501.00656.
588
      Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pas-
589
        canu, and Soham De. Resurrecting recurrent neural networks for long sequences, 2023. URL
590
        https://arxiv.org/abs/2303.06349.
591
592   Daniele Paliotta, Junxiong Wang, Matteo Pagliardini, Kevin Y. Li, Aviv Bick, J. Zico Kolter, Albert Gu,
593     François Fleuret, and Tri Dao. Thinking slow, fast: Scaling inference compute with distilled reasoners,
        2025. URL https://arxiv.org/abs/2502.20339.


                                                          11
      Under review as a conference paper at ICLR 2026




594
      Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro
595     Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction
596     requiring a broad discourse context, 2016. URL https://arxiv.org/abs/1606.06031.
597
598   Jongho Park, Jaeseung Park, Zheyang Xiong, Nayoung Lee, Jaewoong Cho, Samet Oymak, Kangwook
599
        Lee, and Dimitris Papailiopoulos. Can mamba learn how to learn? a comparative study on in-context
        learning tasks, 2024. URL https://arxiv.org/abs/2402.04248.
600
601   Guilherme Penedo, Hynek Kydlı́ček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel,
602     Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data
603     at scale, 2024. URL https://arxiv.org/abs/2406.17557.
604
      Bo Peng, Ruichong Zhang, Daniel Goldstein, Eric Alcaide, Xingjian Du, Haowen Hou, Jiaju Lin, Jiaxing
605
        Liu, Janna Lu, William Merrill, Guangyu Song, Kaifeng Tan, Saiteja Utpala, Nathan Wilce, Johan S.
606     Wind, Tianyi Wu, Daniel Wuttke, and Christian Zhou-Zheng. Rwkv-7 ”goose” with expressive dynamic
607     state evolution, 2025. URL https://arxiv.org/abs/2503.14456.
608
609   Pranav Rajpurkar, Jian Zhang, and Percy Liang. Know what you don’t know: Unanswerable questions
610
        for squad. In ACL 2018, 2018.
611   Yuval Ran-Milo, Eden Lumbroso, Edo Cohen-Karlik, Raja Giryes, Amir Globerson, and Nadav Cohen.
612     Provable benefits of complex parameterizations for structured state space models, 2024. URL
613     https://arxiv.org/abs/2410.14067.
614
      Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial
615
        winograd schema challenge at scale, 2019. URL https://arxiv.org/abs/1907.10641.
616
617   Yash Sarrof, Yana Veitsman, and Michael Hahn. The expressive capacity of state space models: A formal
618     language perspective, 2024. URL https://arxiv.org/abs/2405.17394.
619
      Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight
620     programmers, 2021. URL https://arxiv.org/abs/2102.11174.
621
622   Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, and Riccardo Grazzi.
623      Deltaproduct: Improving state-tracking in linear rnns via householder products, 2025. URL
624
         https://arxiv.org/abs/2502.10297.
625   Jimmy T. H. Smith, Andrew Warrington, and Scott W. Linderman. Simplified state space layers for sequence
626      modeling, 2023. URL https://arxiv.org/abs/2208.04933.
627
628
      Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more
        effective than scaling model parameters, 2024. URL https://arxiv.org/abs/2408.03314.
629
630   Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced trans-
631      former with rotary position embedding, 2023. URL https://arxiv.org/abs/2104.09864.
632
      Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and
633
        Furu Wei. Retentive network: A successor to transformer for large language models, 2023. URL
634
        https://arxiv.org/abs/2307.08621.
635
636   Endre Süli and David F. Mayers. An Introduction to Numerical Analysis. Cambridge University Press, 2003.
637
      Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej,
638     Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas
639     Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Cas-
640     bon, Etienne Pot, Ivo Penchev, Gaël Liu, and et. al. Gemma 3 technical report, 2025. URL
641     https://arxiv.org/abs/2503.19786.
642
      M. Tenenbaum and H. Pollard. Ordinary Differential Equations: An Elementary Textbook for Students
643
        of Mathematics, Engineering, and the Sciences. Dover Books on Mathematics. Dover Publications, 1985.
644
        ISBN 9780486649405. URL https://books.google.com/books?id=iU4zDAAAQBAJ.
645
646   Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser,
647     and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems,
        pp. 5998–6008, 2017. URL http://arxiv.org/abs/1706.03762.


                                                          12
      Under review as a conference paper at ICLR 2026




648
      Johannes von Oswald, Nino Scherrer, Seijin Kobayashi, Luca Versari, Songlin Yang, Maximilian Schlegel,
649     Kaitlin Maile, Yanick Schimpf, Oliver Sieberling, Alexander Meulemans, Rif A. Saurous, Guillaume Lajoie,
650     Charlotte Frenkel, Razvan Pascanu, Blaise Agüera y Arcas, and João Sacramento. Mesanet: Sequence mod-
651     eling by locally optimal test-time training, 2025. URL https://arxiv.org/abs/2506.05233.
652
653   Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. Co-Reyes,
654
        Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-dickstein, Kelvin Xu,
        Jaehoon Lee, Justin Gilmer, and Simon Kornblith. Small-scale proxies for large-scale transformer training
655
        instabilities, 2023. URL https://arxiv.org/abs/2309.14322.
656
657   Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. Inference scaling laws: An
658     empirical analysis of compute-optimal inference for problem-solving with language models, 2025. URL
659     https://arxiv.org/abs/2408.00724.
660
      Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule,
661
        2025a. URL https://arxiv.org/abs/2412.06464.
662
663   Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with
664     the delta rule over sequence length, 2025b. URL https://arxiv.org/abs/2406.06484.
665
      Annan Yu and N. Benjamin Erichson. Block-biased mamba for long-range sequence processing, 2025. URL
666
        https://arxiv.org/abs/2505.09022.
667
668   Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really
669     finish your sentence?, 2019. URL https://arxiv.org/abs/1905.07830.
670
671
672
673
674
675
676
677
678
679
680
681
682
683
684
685
686
687
688
689
690
691
692
693
694
695
696
697
698
699
700
701



                                                         13
      Under review as a conference paper at ICLR 2026




702
      LLM Usage. We utilized Large Language Models to polish the writing in our submission as well as generate
703   latex code for formatting tables and figures.
704
705   A     RELATED WORK
706   Linear-time sequence mixers. State-space models (SSMs) provide linear-time sequence mixing through
707   explicit dynamical states and efficient scan/convolution implementations, offering significant computational
708   advantages over quadratic-time attention mechanisms (Gu et al., 2022a; Smith et al., 2023; Gupta et al.,
709   2022). Mamba-1 (Gu & Dao, 2024) introduced input-dependent selectivity to SSMs, while Mamba-2 (Dao
710   & Gu, 2024) formalized the connection between SSMs and attention via structured state-space duality (SSD)
711   (Katharopoulos et al., 2020; Choromanski et al., 2022). Despite matching transformers on standard language
712   understanding benchmarks, these recurrent models exhibit limitations on tasks requiring precise algorithmic
713   reasoning. Recent evaluations identified gaps in capabilities such as associative retrieval (Bick et al., 2025b;
714
      Arora et al., 2025a), exact copying (Jelassi et al., 2024), and in-context learning (Park et al., 2024; Grazzi et al.,
      2024). To address these limitations, DeltaNet enhances linear attention by replacing additive updates with
715
      delta-rule recurrence (Schlag et al., 2021), with recent work developing hardware-efficient, sequence-parallel
716
      training algorithms for this architecture (Yang et al., 2025b). This has catalyzed a broader effort to improve the
717   algorithmic capabilities of linear-time models through architectural innovations including gating mechanisms,
718   improved state transition dynamics, and hybrid approaches (Peng et al., 2025; Siems et al., 2025; Yang et al.,
719   2025a; Paliotta et al., 2025; Bick et al., 2025a).
720
      Expressivity and state tracking in recurrent mixers. Recent work characterizes the types of state that recur-
721
      rent, constant-memory mixers can maintain, revealing algorithmic deficiencies in previous SSM-based models.
722
      Merrill et al. (2025) show that under finite precision, practical SSMs collapse to TC0, leading to failures on
723
      tasks like permutation composition over S5 unless the primitive is extended. Similarly, Yu & Erichson (2025)
724   prove that a single-layer Mamba is not a universal approximator. Several modifications have been proposed
725   to improve expressivity. For instance, the same work shows that a block-biased variant regains the universal
726   approximation property with only minor changes, either through block decomposition or a channel-specific
727   bias. Allowing negative eigenvalues or non-triangular transitions enables linear RNNs—including diagonal
728   and Householder/DeltaNet forms—to capture parity and, under mild assumptions, regular languages (Grazzi
729   et al., 2025). Complex-valued parameterizations provide another avenue for enhanced expressivity. Diagonal
730   LTI SSMs demonstrate effectiveness for language modeling (Gu et al., 2022b; Orvieto et al., 2023), with
731   complex variants achieving equivalent functions using smaller, well-conditioned parameters (Ran-Milo et al.,
732
      2024). However, the introduction of selectivity—the central innovation of modern SSMs (Gu & Dao, 2024)—
      narrowed the performance gap with Transformers by enabling input-dependent dynamics and achieving state-
733
      of-the-art results on language modeling benchmarks, leading practitioners to abandon complex states in favor of
734
      simpler real-valued architectures. We extend this line of work by reintroducing complex-valued state evolution
735   that yields a real SSM with doubled dimensionality and block-diagonal rotations applied to the update rule—
736   analogous through SSD (Dao & Gu, 2024) to how RoPE (Su et al., 2023) applies complex rotations to queries
737   and keys in attention. The resulting data-dependent rotational structure expands stable dynamics to include os-
738   cillatory modes, enabling richer states while maintaining constant memory and linear-time complexity.
739
740   B     TRAPEZOIDAL DISCRETIZATION PROOFS
741   B.1    PROOF OF PROPOSITION 5
742
      Proposition 5 (Variation of Constants (Tenenbaum & Pollard, 1985)). Consider the linear SSM,
743
744                                             ḣ(t) = A(t)h(t)+B(t)x(t),
745   where h(t) ∈ R is the hidden state, A(t) ∈ R is the scalar state transition matrix and B(t)x(t) ∈ RN is
                       N

746   the input projection. For a time grid τt =τt−1 +∆t, the hidden state satisfies,
                                                       Z τt
747
748
                                    ht ≈ e∆t At
                                                ht−1 +      e(τt −τ)At B(τ)x(τ)dτ ,                      (10)
                                                               τt−1
749                                                        |              {z              }
                                                                      state-update
750
751
752
753
      Proof. Let Φ(t, s) be the fundamental solution of the homogeneous system ḣ(t) = A(t)h(t), i.e.,
      ∂tΦ(t,s)=A(t)Φ(t,s) and Φ(s,s)=1 (since A is scalar). By variation of constants,
754                                                  Z t
755
                                  h(t)=Φ(t,s)h(s) + Φ(t,τ)B(τ)x(τ)dτ.
                                                                  s


                                                                14
      Under review as a conference paper at ICLR 2026




756
      Choosing (s,t)=(tk−1,tk ) gives the exact one–step relation
757                                                     Z tk
758                            hk =Φ(tk ,tk−1)hk−1 +          Φ(tk ,τ)B(τ)x(τ)dτ.
759                                                           tk−1

760   On the step [tk−1, tk ], apply zero–order hold to A(·): A(τ) ≈ Ak . Then Φ(tk , τ) ≈ e(tk −τ)Ak and
761   Φ(tk ,tk−1)≈e∆k Ak , yielding
                                                      Z tk
762
                                  hk ≈ e ∆k Ak
                                               hk−1 +      e(tk −τ)Ak B(τ)x(τ)dτ,
763                                                        tk−1
764   which is equation 10.
765
766   B.2   TRAPEZOID DISCRETIZATION’ MASK MATRIX
767
      Proof. When viewing the tensor contraction form, let us call C =(T,N),B =(S,N),L=(T,S),X =(S,P )
768   based on the Mamba-2 paper. With this decomposition of our mask, we can view L=contract(T Z,ZS →
769   T S)(L1,L2).
770
      The original contraction can be seen as
771
                                    contract(T N,SN,T S,SP →T P )(C,B,L,X)
772
      We can now view it as
773
                                contract(T N,SN,T J,JS,SP →T P )(C,B,L1,L2,X)
774
      This can be broken into the following:
775
                                       Z =contract(SN,SP →SNP )(B,X)
776
777
                                       Z ′ =contract(JS,SNP →JNP )(L2,Z)
778                                   H =contract(T J,JNP →T NP )(L1,Z ′)
779                                   Y =contract(T N,T NP →T P )(C,H)
780   Thus, we can view this step: contract(ZS,SNP →ZNP )(L2,Z) as a conv of size two applied on Bx with
781   the traditional SSD L=L1 matrix.
782
783   C     COMPLEX SSM PROOFS
784
      C.1   PROOF OF PROPOSITION 2
785
786   Proposition 2 (Complex-to-Real SSM Equivalence). Consider a complex-valued SSM
                                                                       
787                          ḣ(t)=Diag A(t)+iθ(t) h(t)+ B(t)+iB̂(t) x(t),                                       (6)
788
                                                   ⊤    
                              y(t)=Re C(t)+iĈ(t) h(t) ,
789
790   where h(t) ∈ CN/2, θ(t),B(t),B̂(t),C(t),Ĉ(t) ∈ RN/2, and x(t),A(t) ∈ R. Under Euler discretization,
791   this system is equivalent to a real-valued SSM
792                                          ht =e∆t At Rtht−1 +∆tBtxt,                                (7)
793                                          yt =C⊤
                                                  t ht ,
794                    N
      with state ht ∈R , projections
795                                                               
                                            Bt          Ct
796                                   Bt =      ∈RN ,
                                                   Ct =        ∈RN ,
                                            B̂t         −Ĉt
797
      and a transition matrix
798                                                                            
                                                                 cos(Θ) −sin(Θ)
                                              
                                           N/2
799                 Rt = Block {R(∆tθt[i])}i=1 ∈RN×N ,  R(Θ)=                    .
                                                                  sin(Θ) cos(Θ)
800
801
      Proof. We first present the derivation for N =2; the block-diagonal structure for general even N follows
802
      by grouping pairs of coordinates.
803
804   Let ht +iĥt denote the complexified hidden state, with parameters A(t)+iθ(t) and B(t)+iB̂(t) for the
805   transition and input, respectively. By the variation of constants formula (Proposition 5), applying zero–order
806   hold and Euler’s rule over a step [tk−1,tk ] gives
807                             hk +iĥk =e∆t (At +iθt )(hk−1 +iĥk−1)+∆t(Bt +iB̂t)xt.
808
      Expanding the exponential,
809                                                                          
                                   e∆t (At +iθt ) =e∆t At cos(∆tθt)+isin(∆tθt) ,


                                                            15
      Under review as a conference paper at ICLR 2026




810
                                   
                                   ht
811   so in real coordinates ht =      ∈R2 the recurrence becomes
                                   ĥt
812                                                                       
                                                                            Bt
                                     ∆t At cos(∆t θt ) −sin(∆t θt )
813                           ht =e                                  h +∆t     x.
                                            sin(∆tθt) cos(∆tθt) t−1         B̂t t
814                                       |           {z            }
815                                                        R(∆t θt )

816   Stacking across N/2 such pairs yields the block-diagonal transition  
817                                                          N/2          Bt
                             ht =e∆t At Block {R(∆tθt[i])}i=1 ht−1 +∆t        x.
818                                                                        B̂t t
819
      For the output,
820                                                                C ⊤
821                                                   ⊤                    t
                                    yt =Re (Ct +iĈt) (ht +iĥt) =              ht,
822                                                                     −Ĉt
823   which defines the real projection Ct ∈RN in the proposition. This proves the equivalence between complex
824   SSM and the real block-diagonal system with rotations.
825
      C.2    PROOF OF PROPOSITION 3
826
827   Proposition 3 (Complex SSM, Data-Dependent RoPE Equivalence). Under the notation established in
828   Proposition 2, consider the real SSM defined in Eq. 7 unrolled for T time-steps. The output of the above
      SSM is equivalent to that of a vanilla scalar transition matrix-based SSM (Eq. 2) with a data-dependent
829
      rotary embedding applied on the B,C components of the SSM defined as:
830                                        t                            t          ⊤
                                         Y                                Y
831                        ∆t At               ⊤                               ⊤
                     ht =e       ht−1 +( Ri )Btxt,                yt = ( Ri )Ct ht                          (8)
832                                    i=0                              i=0
                                                                               Q1
833   where the matrix production represents right matrix multiplication, e.g., i=0 Ri = R0R1. We denote
834   employing the vanilla SSM to compute the Complex SSM as “RoPE trick”.
835
836   Proof. Consider the SSM
837                                ht = e∆t At Rtht−1 + Btxt,           yt = C⊤ t ht ,                       (11)
                                                              ∆t At
838   where (as in Proposition 3) At ∈R is a scalar (so that e      is a scalar and commutes with rotations), and
839   Rt is block-diagonal orthogonal/unitary, hence R−1 t =Rt .
                                                                 ⊤

840   Unrolling the recurrence with the convention that an empty product is the identity,
841                                           X t  Y  t           
842                                      ht =             e∆s As Rs Bixi.                                   (12)
843                                                  i=0    s=i+1
844   Thus
                                                       t      t
                                                             Y          
845
                                                       X
                                      yt = C⊤
                                            t ht =       C⊤
                                                          t     e∆s As
                                                                       Rs Bi xi .                           (13)
846
                                                       i=0             s=i+1
847   Using unitarity property,
848                                t           t             i                    t          i
                                   Y           Y            Y          −1       Y         Y
                                                                                                 R⊤
                                                                                                    
849                                     Rs =         Rs          Rs           =       Rs          s .
850                               s=i+1        s=0           s=0                  s=0       s=0
851   Since e∆s As are scalars, they commute with rotations; hence
852                                Xt      t
                                          Y      Y   t           i
                                                                  Y    
853                          yt =     C⊤t    R s           e∆s As
                                                                      R⊤
                                                                       s Bi xi                              (14)
854                                   i=0      s=0            s=i+1                s=0

855                                     t
                                       Y              t  Y
                                                     ⊤X   t                         i
                                                                                   Y         
                                            R⊤                            e∆s As            R⊤
                                               
856                               =          s Ct                                            s Bi xi .      (15)
857                                     s=0            i=0 s=i+1             s=0
                                             Qt                           Qi
      Define the rotated parameters C̄t := s=0R⊤                                  ⊤
                                                                                    
858                                                    s  C t and B̄i :=    s=0 Rs Bi . Then
                                                       t  t            
859                                                ⊤
                                                     X      Y
                                                                  ∆s As
860                                       yt = C̄t               e        B̄ixi.                            (16)
                                                     i=0 s=i+1
861                                                        Qt
      Equivalently, introducing the rotated state h̃t := s=0R⊤
                                                                     
862                                                                s ht ,
863                                  h̃t = e∆t At h̃t−1 + B̄txt,        yt = C̄⊤
                                                                               t h̃t ,                      (17)



                                                                   16
      Under review as a conference paper at ICLR 2026




864
      C.3   PROOF OF PROPOSITION 4
865
866
      Proposition 4 (Rotary Embedding Equivalence with Trapezoidal Discretization). Discretizing a complex
      SSM with the trapezoidal rule (Proposition 1) yields the recurrence
867
                                              t−1
                                               Y                       Yt    
868                                                  ⊤
                            ht =αtht−1 +βt         Ri Bt−1xt−1 +γt          R⊤
                                                                             i Bt xt ,
869
                                                 i=0                        i=0
870                                  t
871
                                    Y     ⊤
                              yt =     R⊤
                                        i Ct   ht.                                                             (9)
872                                  i=0
873   Here Rt is the block-diagonal rotation matrix defined in Proposition 3.
874
      C.4   PROOF OF PROPOSITION 4
875
876   Proposition 4 (Rotary Embedding Equivalence with Trapezoidal Discretization). Discretizing a complex
877
      SSM with the trapezoidal rule (Proposition 1) yields the recurrence
                                              t−1
                                               Y                       Yt    
878                                                  ⊤
                            ht =αtht−1 +βt         Ri Bt−1xt−1 +γt          R⊤
                                                                             i Bt xt ,
879
                                                 i=0                        i=0
880                                  t
881
                                    Y     ⊤
                              yt =     R⊤
                                        i Ct   ht.                                                             (9)
882                                  i=0
883   Here Rt is the block-diagonal rotation matrix defined in Proposition 3.
884
      C.5   PROOF OF PROPOSITION 4
885
886   Proposition 4 (Rotary Embedding Equivalence with Trapezoidal Discretization). Discretizing a complex
887
      SSM with the trapezoidal rule (Proposition 1) yields the recurrence
                                              t−1                        t
888                                            Y                       Y     
                                                     ⊤
                            ht =αtht−1 +βt         Ri Bt−1xt−1 +γt          R⊤
                                                                             i Bt xt ,
889
                                                 i=0                        i=0
890
                                     t
891
                                    Y     ⊤
                              yt =     R⊤
                                        i Ct   ht.                                                             (9)
892
                                     i=0
893   Here Rt is the block-diagonal rotation matrix defined in Proposition 3.
894
895   Proof. We begin from the complex SSM (as in Prop. 2)
896                                                                 
                             ḣ(t)=Diag A(t)+iθ(t) h(t) + B(t)+iB̂(t) x(t),
897
                              y(t)=Re (C(t)+iĈ(t))⊤h(t) ,
                                                           
898
899
900
      where A(t)∈R is a scalar and θ(t),B(t),B̂(t),C(t),Ĉ(t)∈RN/2.
901
902
      Recall from Prop. 5,                        Z τt
                       ht ≈e∆t (At +iθt )ht−1 +           e(τt −τ)(At +iθt ) B(τ)+iB̂(τ) x(τ)dτ.
                                                                                        
903
904                                                τt−1
905   Applying Prop. 1 to the above integral, we get
                    ht =e∆t (At +iθt )ht−1 + βtei∆tθt Bt−1 +iB̂t−1 xt−1 + γt Bt +iB̂t xt,
                                                                                    
906                                                                                                           (18)
907   wherem
908                          αt :=e∆t At ,     βt :=(1−λt)∆te∆t At ,   γt :=λt∆t,
909
910   Since e∆t (At +iθt ) =αtei∆tθt and as shown in Prop. 2, multiplication by ei∆tθt is a block-diagonal rotation
911   in real coordinates, we get the real N-dimensional recurrence
912                                  ht =αtRtht−1 + βtRtBt−1xt−1 + γtBtxt,                                     (19)
913
914
                                  yt =C⊤ t ht ,                        
915                                   N/2                  cosΘ −sinΘ
      where Rt =Block {R(∆tθt[i])}i=1 where R(Θ)=                        , and projections
916                                                          sinΘ cosΘ
                         
917         Bt          Ct
      Bt =      ,Ct =        . Note that Rt is orthogonal, so R−1   ⊤
                                                                t =Rt .
            B̂t        −Ĉt

                                                             17
      Under review as a conference paper at ICLR 2026




918
919
920
921
922
                         N
923
924                  Y                                       Y

925                  SSM                                  Trapezoidal                                 Linear Projection

926                                                          SSM
                A    X       B   C                                                                    Sequence Transformation
                                                      A      X   B       C
927                                                                                                   MIMO Projection (optional)
928                      σ                 σ                             RoPE        σ                Nonlinearity (activation,
                                                                                                      normalization, multiplication, bias
929                                                                  + +                              addition)
                         Conv                                                    Θ
930                                                                  N       N
931
932
933
934
935
936
937
938
      Figure 4: Contrasting Mamba-2 and Mamba-3 Architectures: Key updates include trapezoidal discretization,
      data-dependent RoPE embeddings, MIMO projections, QK normalization, and learnable biases.
939
940
941   We define the following,
942                            t
                              Y                            t
                                                            Y                                t
                                                                                              Y      
943                    h̃t :=    R⊤
                                  s ht ,           B̄t :=            R⊤
                                                                      s Bt ,         C̄t :=        R⊤
                                                                                                    s Ct .
944                                  s=0                     s=0                              s=0
                                           Qt      ⊤             ⊤
945   Left-multiplying equation 19 by         s=0 Rs and using Rt Rt =I,
946                                        h̃t =αth̃t−1 + βtB̄t−1xt−1 + γtB̄txt,
947
                                         yt = C̄⊤
                                                t h̃t .
948   This is a vanilla scalar-transition SSM with data-dependent rotary embeddings absorbed into B,C via
949   cumulative products of R⊤  s.
950
951
      D    MIMO FOR MAMBA-3
952
953   With hindsight from Mamba and with inference in mind, we propose the following MIMO formulation:
954   Mamba with MIMO With a given batch, head, and sequence position t, consider the input Ut ∈ RD .
955   Also denote P,R∈N as the head dimension and MIMO rank, respectively. We first obtain SSM parameters
956   via a set of projections defined in terms of tensor contraction notation as follows:
957
958         Bt =contract(DNR,D →NR)(WB,Ut)                               Ct =contract(DNR,D →NR)(WC,Ut),
959
           X′t =contract(P D,D →P )(WX′ ,Ut)                             Xt =contract(P R,P →P R)(WX,X′t),
960
961
962   where WB,WC,WX′ ,WX are model parameters. Additionally, we obtain the residual term Zt in the same
963
      manner as Xt with weights WZ′ and WZ. The state update and the SSM output is then computed via the
      following MIMO SSM:
964
965
                           Ht = atHt−1 + BtX⊤    t ∈R
                                                      N×P
                                                            ,     Yt = H⊤  t Ct ∈R
                                                                                  P ×R
                                                                                       .
                               ′                                              ′
966
      The intermediate output Y t is obtained via some residual function ϕ, Y t ←ϕ(Yt,Zt). Finally, the layer
      output Ot ∈RD is computed via the following down projections:
967
              O′t =contract(P R,R→P )(WO′ ,Y′t)               Ot =contract(P,P D →D)(WO,O′t).
968
969   This formulation enhances the existing Mamba3 architecture by providing a lightweight parameterization
970   that transforms the set of independent SISO SSMs within each head into a set of MIMO SSMs. Here, we
971   note that the hardware-efficient chunking technique employed by Mamba2 for pretraining can be applied
      with little change, as the MIMO dimension r is orthogonal to the sequence dimension.


                                                                 18
       Under review as a conference paper at ICLR 2026




972
       E     EXPERIMENTAL DETAILS
973
974    Language Modeling Our pretraining procedures follow that of Dao & Gu (2024)’s section D.2. All models
975    at each scale follow the same procedure and were trained with bfloat16. The Mamba family of models were
976
       trained using the standard expand factor of 2 and a dstate of 128 and head dimension of 64. The Transformer
       baselines follows Dao & Gu (2024), and the Gated DeltaNet baselines follow (Yang et al., 2025a). We
977
       utilize the Llama-3.1 tokenizer (Grattafiori et al., 2024) for all models.
978
979    We utilize LM Evaluation Harness (Gao et al., 2024) to test the zero-shot languag modeling capabilities
980    of our pretrained model on LAMBADA (OpenAI version) (Paperno et al., 2016), HellaSwag (Zellers et al.,
981    2019), PIQA (Bisk et al., 2019), Arc-Easy/Arc-Challenge (Clark et al., 2018), WinoGrande (Sakaguchi
982
       et al., 2019), and OpenBookQA(Mihaylov et al., 2018).
983    Real-World and Synthetic Retrieval For our real-world retrieval tasks, we evaluate on the common suite
984    consisting of SWDE (Arora et al., 2025b), SQUAD (Rajpurkar et al., 2018), FDA (Arora et al., 2025b),
985    TriviaQA (Joshi et al., 2017), NQ (Kwiatkowski et al., 2019), and DROP (Dua et al., 2019). We utilize
986    the cloze-formatted version of the aforementioned tasks provided by Arora et al. (2025b; 2024), as the
987    original datasets are in a question-answering format, making it challenge for solely pretrained models. All
988
       tasks were truncated to match the training context length. The synthetic NIAH tasks (Hsieh et al., 2024)
       were also run with LM Evaluation Harness.
989
990    State-Tracking Synthetics Training follows a sequence length curriculum that progresses from 3 -40 to
991    160, evaluated at 256. Each curriculum runs for 104 steps with batch size 256. We use 1 layer models for
992    Parity and 3 layer models for Modular-arithmetic tasks. The state size is chosen to be 64, and we sweep
993    dmodel ∈ {32,64} and 8 learning rates logarithmically spaced between 10−4 and 10−2, reporting the best
994
       validation accuracy.
995    F     ADDITIONAL EXPERIMENTAL RESULTS
996
997
998
                                                      Context Length Extrapolation
999                                                                                             Train length = 2K
1000                              10.8                                                          Gated DeltaNet
1001
                                                                                                Mamba-2
                                                                                                Mamba-3
1002                              10.6

                     Perplexity
1003
1004
                                  10.4
1005
1006
1007                              10.2
1008
1009                              10.0
1010
                                         1K          2K             4K            8K            16K           32K
1011                                                                Context length
1012
1013
       Figure 5: Pretrained 1.5B models’ performance on the held-out FineWeb-Edu test set at varying context
1014   lengths. Mamba-3 exhibits strong length extrapolation while Mamba-2 falters at longer contexts.
1015
1016
1017   Table 5: Downstream language modeling evaluations on parameter-matched pretrained models, including
1018   Mamba-3 MIMO. Mamba-3 MIMO’s average accuracy on all tasks is more than 1 percentage point better
1019   than the next best (Mamba-3 SISO).
1020
1021       Model                         FW-Edu   LAMB.    LAMB.     HellaS.   PIQA    Arc-E    Arc-C    WinoGr.    OBQA      Average
                                          ppl ↓    ppl ↓    acc ↑    acc n ↑   acc ↑   acc ↑   acc n ↑    acc ↑     acc n ↑    acc ↑
1022
           Transformer-440M               13.03    21.2     41.7      50.5     69.9    67.6     34.6      56.7       26.0      49.6
1023       Gated DeltaNet-440M            13.12    19.0     40.4      50.5     70.5    67.5     34.0      55.3       25.8      49.1
1024       Mamba-2-440M                   13.00    19.6     40.8      51.7     70.6    68.8     35.0      54.1       26.0      49.6
           Mamba-3-440M                   12.87    19.6     40.2      51.7     71.9    68.9     34.4      55.8       26.0      49.8
1025       Mamba-3-MIMO-440M              12.72    17.1     43.4      52.8     70.8    69.6     35.6      56.3       28.4      51.0



                                                                      19
       Under review as a conference paper at ICLR 2026




1026
1027                            16.0
                                                   Mamba-3 Validation Perplexity
                                                                                           Mamba-3 MIMO
1028                                                                                       Mamba-3 SISO
1029                            15.5                                                       Llama
1030                                                                                       GatedDeltaNet
                                                                                           Mamba-2
1031
                                15.0
1032
1033
1034                            14.5


                   Perplexity
1035
1036                            14.0
1037
1038                            13.5
1039
1040
                                13.0
1041
1042
1043                            12.5
1044
1045                            12.0
                                       0   25000    50000   75000    100000   125000   150000   175000
1046                                                           Global Step
1047
1048   Figure 6: Mamba-3 demonstrates superior performance compared to strong baselines like Mamba-2, Llama,
1049   and Gated Deltanet. These are 440M models, evaluated on FineWeb-Edu and 100B tokens.
1050
1051
1052
1053
1054
1055
1056
1057
1058
1059
1060
1061
1062
1063
1064
1065
1066
1067
1068
1069
1070
1071
1072
1073
1074
1075
1076
1077
1078
1079



                                                                20

