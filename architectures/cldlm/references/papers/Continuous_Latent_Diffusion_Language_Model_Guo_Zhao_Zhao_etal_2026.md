# Continuous Latent Diffusion Language Model Guo Zhao Zhao etal 2026

> Source: `Continuous_Latent_Diffusion_Language_Model_Guo_Zhao_Zhao_etal_2026.pdf`

---

                                                   Cola
                                                          Continuous Latent Diffusion Language Model

                                        Hongcan Guo1,2 , Qinyu Zhao3,† , Yian Zhao1,4 , Shen Nie1,5 , Rui Zhu1 , Qiushan Guo1 ,
                                                                                                                          B
                                          Feng Wang1 , Tao Yang1 , Hengshuang Zhao2 , Guoqiang Wei1 , Yan Zeng1,
                                         1
                                             ByteDance Seed, 2 The University of Hong Kong, 3 The Australian National University, 4 Peking
                                                                      University, 5 Renmin University of China




arXiv:2605.06548v1 [cs.CL] 7 May 2026
                                                                                                                    B
                                                           †
                                                               Work done during an internship at Bytedance Seed,        Corresponding author



                                                                                             Abstract
                                                Large language models have achieved remarkable success under the autoregressive paradigm, yet
                                                high-quality text generation need not be tied to a fixed left-to-right order. Existing alternatives
                                                still struggle to jointly achieve generation efficiency, scalable representation learning, and effective
                                                global semantic modeling. We propose Cola DLM, a hierarchical latent diffusion language model
                                                that frames text generation through hierarchical information decomposition. Cola DLM first learns
                                                a stable text-to-latent mapping with a Text VAE, then models a global semantic prior in continuous
                                                latent space with a block-causal DiT, and finally generates text through conditional decoding. From
                                                a unified Markov-path perspective, its diffusion process performs latent prior transport rather than
                                                token-level observation recovery, thereby separating global semantic organization from local textual
                                                realization. This design yields a more flexible non-autoregressive inductive bias, supports semantic
                                                compression and prior fitting in continuous space, and naturally extends to other continuous
                                                modalities. Through experiments spanning 4 research questions, 8 benchmarks, strictly matched
                                                ∼2B-parameter autoregressive and LLaDA baselines, and scaling curves up to about 2000 EFLOPs,
                                                we identify an effective overall configuration of Cola DLM and verify its strong scaling behavior for
                                                text generation. Taken together, the results establish hierarchical continuous latent prior modeling
                                                as a principled alternative to strictly token-level language modeling, where generation quality
                                                and scaling behavior may better reflect model capability than likelihood, while also suggesting a
                                                concrete path toward unified modeling across discrete text and continuous modalities.

                                                Date: May 8, 2026
                                                Correspondence: Yan Zeng at zengyan.yanne@bytedance.com
                                                Project Page: https://hongcanguo.github.io/Cola-DLM/

                                                                                ★ Seedance Team Research Project ★




                                        1     Introduction
                                        Large language models have achieved remarkable success under the autoregressive paradigm [9, 30, 38, 55, 91].
                                        By factorizing the discrete text distribution through the chain rule [6, 22, 39, 45, 60, 102], autoregressive
                                        language models have driven major advances in large-scale pretraining, open-ended generation, and downstream
                                        transfer, and have become the dominant approach to modern language modeling [13, 65, 73, 96, 111]. However,
                                        this paradigm tightly couples generation to a fixed left-to-right order, making inference inherently sequential


                                                                                                   1
and restricting the model’s inductive bias to a single token ordering [1, 3, 23, 50, 67, 98, 104]. Recent progress
in both discrete and continuous diffusion-based text modeling suggests that high-quality language generation
need not rely on such a fixed order; instead, language models can also be defined through more general state
evolution and denoising paths [15, 59, 72, 87, 103, 106].
Despite extensive exploration along autoregressive, discrete diffusion, and continuous diffusion directions
[26, 41, 42, 47, 49, 85, 95, 99], existing methods still struggle to simultaneously achieve generation efficiency,
scalable representation, and global semantic modeling. Autoregressive models directly parameterize token-level
conditional probabilities, yielding a clear training objective, but their fixed generation order incurs inherent
sequential inference cost and introduces a strong hand-crafted inductive bias, which limits performance
on more general generation tasks [7, 17, 20, 53, 119]. Discrete diffusion language models remove explicit
left-to-right factorization [25, 35, 36, 110], yet they still typically perform observation recovery in discrete
token space, leading to costly multi-step sampling, while intermediate discrete states are not well suited to
stably represent global semantic structure [40, 62, 86, 90, 94, 115, 116]. Continuous diffusion methods further
introduce continuous representation spaces [28, 81, 89], but most existing approaches still use the diffusion
path to recover token-aligned representations rather than to explicitly model a latent prior [21, 29]. As a result,
current methods have not yet provided a unified framework that systematically combines non-autoregressive
generation, continuous representation, and probabilistic text modeling.
To address this gap, we propose Cola DLM, a hierarchical latent-space diffusion language model. Cola
DLM first learns a stable mapping between text and continuous latent variables through a Text VAE
[8, 46, 51, 83, 100, 112], then models the latent prior in continuous latent space with a block-causal DiT
[4, 11, 12, 57, 66, 75, 76, 108], and finally generates text through a conditional decoder. The key idea of Cola
DLM is to use diffusion not for token-level observation recovery, but for latent prior transport. From a unified
Markov-path perspective, this design explicitly decomposes text generation into two levels: global semantic
organization in continuous latent space and local textual realization through conditional decoding. This
decomposition weakens the inductive bias imposed by fixed token order, allows the geometry of continuous
space to directly support semantic compression and prior fitting, and enables a more flexible non-autoregressive
generation process. Moreover, block-causal prior modeling preserves cross-block causal structure while allowing
more efficient parallel computation within each block. Grounded in hierarchical latent-space modeling, Cola
DLM is also highly modular and naturally extensible to alternative latent modeling components and other
continuous modalities [19, 112].
Motivated by these observations, we systematically study diffusion language modeling in continuous latent
space from both theoretical and empirical perspectives. Our contributions are as follows.
   • We propose Cola DLM, a hierarchical latent-space language model that explicitly decomposes text
     generation into global semantic modeling and local textual realization within a unified probabilistic
     framework, while using diffusion-based prior modeling in continuous latent space to connect the two,
     thereby establishing a new paradigm for language generation from the perspective of hierarchical
     information decomposition.
   • We analyze the differences between Cola DLM and existing language modeling paradigms from a unified
     Markov-path perspective, clarifying its advantages in global semantic modeling, non-autoregressive
     inductive bias, and theoretical interpretability, which are further validated in the subsequent experiments.
   • Through extensive experiments spanning 4 research questions, 8 benchmarks, strictly matched ∼2B-
     parameter autoregressive and LLaDA baselines, and scaling curves up to about 2000 EFLOPs, we
     systematically validate the central claims of Cola DLM, identify an effective overall configuration, and
     verify its strong potential and favorable scaling behavior for text generation.
   • We further analyze several issues beyond the core framework, including the mismatch between likelihood
     estimation and generation quality, first-block conditioning, and latent compression. We also provide
     preliminary evidence that Cola DLM offers a natural bridge from discrete text to continuous modalities
     such as vision, pointing to a broader unified generative paradigm.




                                                        2
2     Related Work
2.1   Autoregressive Language Models
Autoregressive language models [56, 77, 92, 101] factorize the discrete text distribution by the chain rule
and are trained with token-level maximum likelihood, making them the most widely adopted paradigm for
text modeling. Their limitations are that generation is constrained by a fixed left-to-right order, inference
is inherently sequential, and they are less suitable for non-monotonic generation tasks such as infilling,
local editing, and global reorganization. In contrast, Cola DLM first models a global semantic prior in a
continuous latent space and then performs conditional decoding, thereby alleviating token-level ordering bias
and improving generation efficiency with a block-causal DiT.

2.2    Discrete Diffusion Language Models
Discrete diffusion language models mainly fall into two categories. The first category is based on discrete
transition kernels [2, 10, 88], which define forward perturbation and reverse recovery in discrete token space
and achieve non-autoregressive generation through multi-step denoising; however, sampling is usually slow and
these methods cannot easily exploit the smooth semantic structure of continuous spaces. The second category
is based on masking or absorbing states [69, 70, 80, 81, 84, 105, 113, 114, 117, 118], which construct training
objectives by progressively mapping tokens to masks or absorbing states and then recovering the original
text; however, information loss in intermediate states limits global semantic planning and fine-grained control.
In contrast, Cola DLM moves the diffusion process to a continuous latent space, where compressible latent
variables carry global semantics, thus combining the manipulability of continuous spaces with hierarchical
semantic modeling.

2.3    Continuous Diffusion Language Models
Continuous diffusion language models can be broadly divided into three categories. The first category consists
of high-dimensional vocabulary-aligned continuous methods [31, 43, 59, 79], which perform continuous diffusion
or flow modeling directly on one-hot vectors, logit simplexes, or probability simplexes to preserve alignment
with discrete vocabularies; however, their representation dimension scales with vocabulary size, which limits
scalability. The second category consists of token-embedding-based continuous methods [14, 21, 24, 27, 29, 52,
54, 87], which first map text into continuous embedding spaces and then apply diffusion or flow modeling to
improve generation flexibility; however, their generation process remains essentially the recovery of noisy target
representations, lacking an explicit hierarchical latent-variable interpretation and a unified marginal-likelihood
view of text distributions. The third category consists of latent-space continuous methods [44, 58, 63, 109],
which compress text into latent spaces with autoencoders or VAEs and then perform diffusion modeling. These
methods typically rely on latent-space design and autoregressive decoders, and usually treat the latent space
as a fixed representation rather than modeling it under a hierarchical latent-variable framework. In contrast,
Cola DLM explicitly separates global semantics from local realization through hierarchical latent-variable
modeling, and learns a semantic prior in a dynamic continuous latent space, thereby better balancing modeling
flexibility, inference efficiency, and theoretical interpretability.

3     Continuous Latent Diffusion Language Model
This section first presents Cola DLM as a hierarchical latent-variable language model with a rigorous
probabilistic definition. We also outline the overall workflow of Cola DLM. We then place Cola DLM in a
unified theoretical framework together with AR models, discrete denoising language models, and continuous
token-space methods. Detailed derivations and proofs are deferred to Appendices A, B, C and D.

3.1   Theoretical Foundations of Cola DLM
In this subsection, we present Cola DLM as a hierarchical latent-variable language model with a rigorous
probabilistic definition. We then introduce its unconditional and conditional probability estimators. Detailed
derivations and proofs are provided in Appendices A and B.



                                                        3
3.1.1   Theoretical Formulation of Cola DLM
Hierarchical latent-variable modeling.      Let x ∈ X denote a discrete text sequence, and let z0 ∈ Rd denote its
continuous latent variable. The generative model of Cola DLM consists of a conditional decoder pθ (x | z0 )
and a latent prior pψ (z0 ):
                                                                  Z
                        p(x, z0 ) = pθ (x | z0 ) pψ (z0 ), p(x) = pθ (x | z0 ) pψ (z0 ) dz0 .               (3.1)

Here, qϕ (z0 | x) is used only for variational inference during training, and is not part of the generative model
itself.
We model pψ (z0 ) with a continuous-flow prior. Let the base distribution be p1 (z1 ) = N (0, I), and let vψ (zt , t)
be the vector field. Then
                                                   dzt
                                   z1 ∼ p1 ,           = vψ (zt , t),     z0 = Φψ0←1 (z1 ),                   (3.2)
                                                   dt
which induces pψ = (Φψ    0←1 )♯ p1 . In the sequence implementation, the latent is further decomposed into blocks,
       (1)          (B)
z0 = (z0 , . . . , z0 ), with
                                                               B
                                                          (1)           (b)   (<b)
                                                              Y
                                          pψ (z0 ) = pψ (z0 )      pψ (z0 | z0 ).                             (3.3)
                                                              b=2
This factorization directly corresponds to the block-causal prior learning and block-wise inference used later.

ELBO and prior learning.       By Jensen’s inequality, the training lower bound of Cola DLM is
                   log p(x) ≥ Eqϕ (z0 |x) [log pθ (x | z0 ) + log pψ (z0 ) − log qϕ (z0 | x)] =: LELBO (x).            (3.4)
Training therefore maximizes LELBO (x), or equivalently minimizes −LELBO (x).
Let the aggregated posterior be q̄ϕ (z0 ) = qϕ (z0 | x) pdata (x) dx. The expected ELBO can then be written as
                                           R

                Epdata (x) [LELBO (x)] = Eq(x,z0 ) [log pθ (x | z0 )] − Iq (X; Z0 ) − KL(q̄ϕ (z0 ) ∥ pψ (z0 )),        (3.5)
where q(x, z0 ) = pdata (x)qϕ (z0 | x). This decomposition shows that Cola DLM separates text modeling into
conditional reconstruction, information compression, and prior matching.
When the encoder and decoder are fixed, prior learning reduces to
                            max Ez0 ∼q̄ϕ [log pψ (z0 )]    ⇐⇒       min KL(q̄ϕ (z0 ) ∥ pψ (z0 )).                      (3.6)
                              ψ                                      ψ

In practice, we do not optimize the density directly. Instead, we learn the corresponding vector field with
Flow Matching. For block b, the conditional FM objective is
                                  B           
                                                                                  2
                                                                                    
                                                    (b)    (<b)     (b)
                                 X
                          LFM =      Et,z0 ,z1 vψ zt , t; z0      − ut (z0 , z1 )     .                (3.7)
                                                                                            2
                                       b=1

Flow Matching is therefore a solver for the prior in Cola DLM, rather than the definition of the model itself.

    Summary. The generative distribution of Cola DLM is defined by the hierarchical factorization in Eq. (3.1):
    the latent prior pψ (z0 ) generates global continuous semantics, and the decoder pθ (x | z0 ) realizes discrete text.
    The encoder qϕ (z0 | x) is not part of the generative model, but an inference model that lifts the observed data
    distribution pdata (x) into a model-dependent latent joint distribution,
                                                                          Z
                           qϕ (x, z0 ) = pdata (x)qϕ (z0 | x), q̄ϕ (z0 ) = qϕ (z0 | x)pdata (x) dx.

    Thus, while pdata (x) is fixed, the induced latent distribution q̄ϕ is not. When the representation is fixed, prior
    learning fits pψ to q̄ϕ . Under joint training, however, q̄ϕ and pψ co-evolve: the encoder reshapes the latent data
    distribution, while the learned prior regularizes and organizes the latent space. Flow Matching is therefore
    only an implementation choice for learning this prior transport; the underlying model remains a hierarchical
    latent-variable language model.



                                                              4
Algorithm 1 Unconditional log-likelihood estimation                      Algorithm 2 Conditional log-likelihood estimation
Require: x, K                                                            Require: xpre , xres
 1: for k = 1, . . . , K do                                               1: estimate log pb(xpre , xres )
                 (k)                                                             using Algorithm 1
 2:     sample z0      ∼ qϕ (z0 | x)
                              (k)              (k)                        2: estimate log pb(xpre )
 3:     w(k) ← log pθ (x | z0 ) + log pψ (z0 )
                          (k)
                                                                                 using Algorithm 1
               − log qϕ (z0 | x)                                          3: return log pb(xres | xpre )
 4: end for                                                                      = log pb(xpre , xres ) − log pb(xpre )
                             1 PK      (k)
 5: output log pbELBO (x) = K    k=1 w                                           according to Eq. (3.14)
                               1 PK          (k) )
       or log pbIWAE (x) = log K   k=1 exp(w
       according to Eq. (3.12)



3.1.2   Probability Estimation for Cola DLM
                                                                                                                                  (k)
Unconditional probability estimation.                    At evaluation time, we approximate log p(x). For samples z0                    ∼
qϕ (z0 | x), define the importance weight
                                                              (k)               (k)                 (k)
                                    log w(k) = log pθ (x | z0 ) + log pψ (z0 ) − log qϕ (z0               | x).                    (3.8)

                                (k)
The prior term log pψ (z0 ) is evaluated by the CNF change-of-variables formula. Concretely, we solve the
augmented ODE                                               (k) 
                                 d zt        vψ (zt , t)      z0    z
                                        =                  ,     = 0 ,                               (3.9)
                                dt ℓt      ∇· vψ (zt , t)     ℓ0      0
                                         (k)    (k)
from t = 0 to t = 1, yielding (z1 , ℓ1 ). We then obtain
                                                             (k)              (k)       (k)
                                                     log pψ (z0 ) = log p1 (z1 ) + ℓ1 ,                                           (3.10)

where p1 is the terminal base distribution. In high dimensions, the divergence term is estimated with
Hutchinson’s trace estimator:
                                                      
                                          ∂vψ (zt , t)        ∂vψ (zt , t)
                     ∇· vψ (zt , t) = Tr                 ≈ ϵ⊤              ϵ, ϵ ∼ N (0, I),     (3.11)
                                            ∂zt                 ∂zt

where the same ϵ is fixed within one ODE solve.
This gives two standard estimators, namely the ELBO-style and IWAE-style estimators:
                                                     K                                                     K
                                                                                                                          !
                                       1 X                                                  1 X log w(k)
                    log pbELBO,K (x) =     log w(k) ,                log pbIWAE,K (x) = log    e                              .   (3.12)
                                       K                                                    K
                                                 k=1                                                      k=1

The IWAE-style estimator is typically tighter.

Conditional probability estimation.              For a prefix–response decomposition x = (xpre , xres ), the exact identity is

                                         log p(xres | xpre ) = log p(xpre , xres ) − log p(xpre ).                                (3.13)

We therefore obtain a plug-in estimator by scoring the joint sequence and the prefix with the same unconditional
estimator:
                                log pb(xres | xpre ) = log pb(xpre , xres ) − log pb(xpre ).              (3.14)

3.2     Workflow of Cola DLM
In this section, we describe the overall workflow of Cola DLM in detail. As illustrated in Figure 1, we explain
the method from three perspectives: the pretraining of the Text VAE, the pretraining of prior learning with
the Text DiT, and the inference process of Cola DLM.


                                                                     5
                              Pretrain Text VAE                                                                                                                                                                                                                                                                                                                                          Cola DLM Inference
                                                                                                                                     Cola
                                                                                                                                                                                                             Pretrain Text DiT
                                              Attention is all    you       need   !




                                                                                                                                                                                      clean latent
                                               [mask]    is all [mask] [mask] !                                                                                       Text VAE                                                                                                VAE loss                                               Text VAE                                           output Attention is all            you        need     !



                                                          Text VAE                                                                                                     Encoder                                                                                                                                                        Decoder                                                                     Text VAE
                                                                                                                                                                                                                                                                                                                                                                                                                   Decoder
                                                           Encoder




                                                                                                                                                                                                                                                                                              denoised latent
                                                                                                                                                                                                                                                                                                                                                                              cat




 (Cross Entropy loss)   reconstruction loss                                                        (Cross Entropy loss)                                                                                             noised latent
                                                                         kl loss                                                                                                                                                                  Diffusion                                                                                                                grad                                  predict latent


                                                                                       BERT loss
                                                                                                                                                                                                                                                                                                                                                                          detach
                                                          clean latent                                                                                                               reference                                                   Transformer



                                                                                                                          masked input token
                                                                                                                                                                   Initialize from                                                                                                                                                                                                                               Diffusion




                                                                                                                                               clean input token
                                                                                                                                                                                                                                                    Blocks                                                                                                invisible




                                                                                                                                                                                                                                                                                                                                                                                            kv cache
                                                        [mask] mix latent                                                                                                                                                                                                                                       Example:            ,
                                                                                                                                                                    Stage 1 VAE
                                                                                                                                                                                       kl loss                                                                                                                           DiT Training                                                                           Transformer
                                                                                                                                                                                                                                                      (block causal)                                                Attention mask
                                                                                                                                                                                                                                                                                                                                                     clean        noise       visible                              Blocks

                                                          Text VAE                                                                                                                                                                                                                                                           1
                                                                                                                                                                                                                                                diffusion loss                                                                                                                                                    noised latent
                                                           Decoder
                                                                                                                                                                                                                                                     ( mse loss )

                                                                                                                                                                                      reference       detached
                                                                                                                                                                                                                                                                                                                             2
                                                                                                                                                                    Reference                                                                                                                                                                                                                          clean latent              block noise

                                                                                                                                                                    Text VAE
                                                                                                                                                                                                                                                                                                                q block id
                                                                                                                                                                                                                                                                                                                             3
                                                                                                                                                                                                                                              : latent sequence length
                                              Attention is all    you       need   .

                                                                                                                                                                                     clean latent    clean latent
                                                                                                                                                                                                                                          : block num                                                                                                                                                             Text VAE
                                                                                                                                                                     Encoder
                                                                                                                                                                                                                                                                                                                             1
                                                Cola     is all   you       need   !                                                                                                                                                                                                                                                                                                                               Encoder
                                                                                                                                                                                                                                        : block size
                                                                                                                                                                                                                                                                                                                             2
                                                                                                                                                                                                                                      : batch size                        noise
                                                                                                                                                                                                                                                                                                                                                                                               prefix

                                                                                                                                                                                                                                                                                        noised latent
                                                                                                                                                                                                                                                                          part
                                                                                                                                                                                                                                                                                                                                                                                                                                 is
                                                                                                                                                                                                                                                                                                                             3
                                                                                                                                                                                                                                                                                                                                                                                                               Attention
                                                                                                                                                                                                                                                                                                                                     2       3        4                                        prompt

                                                                                                                                                                                                                                      clean latent
                                                                                                                                                                                                                                                                          (b, lsq)
                                                                                                                                                                                                                                                           clean latent                                                      4
                                                                                                                                                                                                                                                            (truncate)

                                              Training Stage 1                                                                                                     Training Stage 2                                                                                                                                                                                                          Inference Stage
                                                                                                                                                                                                                                                                                                                                 1       2       3        1       2       3       4
                                                                                                                                                                                                                                    (b, lsq)             (b, lsq-bs)                 (b, bn * bs)                                                    k block id       total




Figure 1 The Overall Workflow of Cola DLM. Detailed illustration of the training and inference pipeline of Cola DLM.
Training Stage 1 shows Text VAE pretraining with reconstruction, BERT, and KL losses. Training Stage 2 shows
joint pretraining of the Text VAE and Text DiT with gradient control for stable optimization, where a specialized
block-causal mechanism is adopted in the DiT. Inference Stage illustrates the decoding process with KV cache.


3.2.1                                             Text VAE Pretraining
In the first stage, we learn a stable latent–text correspondence. The encoder maps text into the latent space,
and the decoder reconstructs the original text conditioned on the latent:
                                                                                                                                                                                 z0 ∼ qϕ (z0 | x),                                                       x̂ ∼ pθ (x | z0 ).                                                                                                                                                           (3.15)
The goal of this stage is not to learn the final prior, but to establish a stable division of labor between
information stored in the latent and information recovered by the decoder.
The corresponding objective is
                                                                             LVAE = −Eqϕ (z0 |x) log pθ (x | z0 ) + β KL(qϕ (z0 | x) ∥ pbase (z0 )) + λmask Lmask .                                                                                                                                                                                                                                                                   (3.16)
Here, Lmask is the BERT-style masking loss shown in the figure. It prevents the VAE encoder from collapsing
semantically while the decoder merely memorizes surface text. In our experiments, the VAE does not compress
the sequence length. To prevent information leakage and facilitate subsequent streaming generation, both our
VAE encoder and decoder are strictly causal.

3.2.2                                              Prior Learning with Block-Causal DiT
In the second stage, we learn a conditional prior on the stabilized latent space. For block b, the visible set
consists of the historical clean latent blocks and the current noisy block:
                                                  n                o
                                                        (<b)   (b)
                                             Vb = sg(z0 ), zt        ,                                  (3.17)

where sg(·) denotes stop-gradient. This visibility constraint enforces bidirectional attention within each block
and causal dependence across blocks, consistent with Eq. (3.3).
Under this design, prior learning uses a joint objective that combines conditional Flow Matching with a
reference-encoder regularizer:
                                                                                                       
              Lstage2 = λVAE −Eqϕ (z0 |x) log pθ (x | z0 ) + βEqϕ (z0 |x) log qϕ (z0 | x) + λmask Lmask
                                                                                                        (3.18)
                        + λfm LFM + λref Epdata (x) KL qϕ (z0 | x) ∥ qϕref (z0 | x) .

The first group preserves the autoencoding structure with regularized latent learning, the second term learns
the block-level conditional prior, and the third term suppresses latent drift during joint training.


                                                                                                                                                                                                                                    6
3.2.3   Inference: Prefix Encoding, Block-wise Generation, and Conditional Decoding
At inference time, the model first encodes the prefix into clean latent conditions:

                                                    z pre ∼ qϕ (z pre | xpre ).                                        (3.19)

It then generates the response latent block by block. Each block is obtained by transporting a noise seed
under the historical condition:
                                                             
                                (b)                  pre (<b)
                              ẑ0 = Φψ
                                     0←1   ϵ (b)
                                                 ; z    , ẑ0   , ϵ(b) ∼ N (0, I).                  (3.20)

Finally, the decoder outputs the text response conditioned on the prefix and the generated latent blocks:
                                                                     (1:B) 
                                         x̂res ∼ pθ xres | z pre , ẑ0       .                        (3.21)


   Summary. The workflow of Cola DLM implements the above hierarchical probabilistic model through two
   training stages and one inference stage, rather than a mechanical cascade of VAE, DiT, and decoder. In Stage
   1, the base prior pbase regularizes the latent–text interface and stabilizes the autoencoding representation, but
   it is not the final generative prior. In Stage 2, the block-causal DiT learns the final latent prior pψ (z0 ) while
   the VAE remains trainable under reconstruction, masking, and reference regularization. This makes prior
   learning a controlled co-adaptation between the latent representation and the learned flow prior. At inference
   time, the model encodes the prefix, generates future latent blocks autoregressively in latent space, and realizes
   the response through the conditional decoder.


3.3     A Unified View of Cola DLM and Existing Methods
In this section, we compare Cola DLM with AR, LLaDA, and Plaid under a unified Markov-path perspective,
and theoretically characterize the specific advantages of Cola DLM. More detailed analysis and proofs are
provided in Appendices C and D.

3.3.1   Text Modeling under a Unified Stochastic-Path View
For a unified comparison, let τ = (St )t∈T be a stochastic process on state space S, with initial distribution µΘ ,
transition kernel KtΘ , and emission mechanism eΘ (x | τ ). A process-based generative model can be written as
                             Z                                              Y
                    pΘ (x) = eΘ (x | τ ) PΘ (dτ ),     PΘ (dτ ) = µΘ (ds0 )   KtΘ (dst | s<t ).              (3.22)
                                                                                     t>0

This common outer form does not determine the nature of the model. The essential distinction lies in the
state space of the path and its semantic role: a path over text or near-lossless text-aligned representations is
an observation path, whereas a path used only to generate a latent prior is a prior path.
For AR, the path is the prefix expansion itself, yielding an exact chain factorization but binding generation to
a left-to-right filtration:
                                                      YL
                                          pAR (x) =       pη (xi | x<i ).                                 (3.23)
                                                               i=1

For LLaDA, the path is a discrete corruption–recovery trajectory, whose objective is observation reconstruction
in a discrete state space:
                                             T
                                             Y                                                T
                                                                                              Y
                 q(s1:T | x) = q1 (s1 | x)         qt (st | st−1 ),     pθ (s0:T ) = p(sT )         pθ (st−1 | st ).   (3.24)
                                             t=2                                              t=1

Thus, LLaDA weakens the handcrafted left-to-right bias, but still modifies the observation-recovery process
rather than introducing an explicit hierarchical latent variable.


                                                                 7
   Method               State Space                Path Role                    Generative                   Where Continuity         Explicit
                                                                               Factorization                    Appears               Latent

                                                Direct Generation
                                                                                                                                        ✗
                                                                                Q
        AR              Prefix Tokens
                                                       Path                         i p(xi | x<i )                    None

                                                  Discrete
                       Discrete Masked                                                                               Discrete           ✗
                                                                                     Q
    LLaDA
                          Sequences
                                            Observation-Recovery           p(sT )        t pθ (st−1 | st )
                                                   Path                                                            Token Space

                         Continuous             Continuous                           Q                             Continuous           ✗
        Plaid          Token-Aligned        Observation-Recovery           p(hT )        t pθ (ht−1 | ht )
                       Representations             Path                                                            Token Space

                    Compressed Latent
                                                                                                                                        ✓
                                                                           R
  Cola DLM                                  Prior-Transport Path               pθ (x | z0 ) pψ (z0 ) dz0           Latent Space
                       Sequences

         Table 1 Unified Perspective. Key differences among text models under a unified Markov-path view.


Plaid further moves this recovery process to a continuous token-aligned representation h0 = E(x):
                                                T
                                                Y                                                            T
                                                                                                             Y
                q(h1:T | h0 ) = q1 (h1 | h0 )         qt (ht | ht−1 ),          pθ (h0:T ) = p(hT )                pθ (ht−1 | ht ).         (3.25)
                                                t=2                                                          t=1

Its core target is therefore continuous observation recovery, rather than a decomposition into a prior and a
conditional decoder.
In Cola DLM, by contrast, the stochastic path only transports the latent prior:

                                   z1 ∼ p1 ,          z0 = Φψ
                                                            0←1 (z1 ),               x ∼ pθ (x | z0 ),                                      (3.26)

with the marginal still given by Eq. (3.1). Hence, diffusion is used to learn a flexible continuous prior, not to
impose a left-to-right inductive bias on text.
The reason for using a continuous path is not that continuous modeling is inherently superior, but that it
naturally captures the geometry of the latent distribution. In Cola DLM, continuity appears in pψ (z0 ), rather
than in an observation-recovery trajectory:
                                          dzt
                                              = vψ (zt , t),             pψ = (Φψ
                                                                                0←1 )♯ p1 .                                                 (3.27)
                                          dt
Thus, the distinction between Cola DLM and LLaDA lies in both state space and modeling target.
Finally, the reason for using a latent variable is to explicitly separate semantic structure from token realization.
The information decomposition of the expected ELBO,

                 Epdata (x) [LELBO (x)] = Eq(x,z0 ) [log pθ (x | z0 )] − Iq (X; Z0 ) − KL(q̄ϕ (z0 ) ∥ pψ (z0 )),                            (3.28)

shows that z0 is not merely a continuous surrogate for discrete text, but an explicit marginalized intermediate
variable: global semantics are compressed into z0 , while local token realization is delegated to the decoder.

3.3.2     Theoretical Advantages of Cola DLM
A unified criterion.    Let the lower bound of the approximation error for a model family M be

                                           E(M) := inf KL(pdata (x) ∥ p(x)).                                                                (3.29)
                                                           p∈M

For AR, the population risk is determined solely by E(MAR ). In contrast, Cola DLM also incurs a variational
inference gap:
                              GColaDLM                                                                (3.30)
                                                                                    
                                infer   := Epdata (x) KL qϕ (z0 | x) ∥ pθ,ψ (z0 | x) .


                                                                   8
Its total statistical burden is therefore

                                  RColaDLM = E(MColaDLM ) + inf GColaDLM
                                                                 infer   .                                (3.31)
                                                                      ϕ,θ,ψ


Proposition 3.1 (Unified criterion). At the population level, Cola DLM outperforms a comparison model if
and only if its total statistical burden is smaller. Taking AR as an example,

                               ColaDLM ≻ AR            ⇐⇒         RColaDLM < E(MAR ).                     (3.32)

Rate-distortion and structured generation.   Whether a latent bottleneck is beneficial depends on whether
the data admits a low-rate but informative global representation. Define the representation rate-distortion
function as
                           D(R) :=      inf       inf Eq(x,z0 ) [− log pθ (x | z0 )].                (3.33)
                                           q(z0 |x): pθ (x|z0 )
                                        Iq (X;Z0 )≤R

If D(R) is already small at a low rate R, then the data admits a low-dimensional semantic variable, and a
latent bottleneck is more likely to reduce the overall mismatch. Conversely, if high-quality reconstruction
requires a high information rate, aggressive compression only makes conditional reconstruction harder.
This can be characterized further through a structured-generation assumption. Suppose there exists a global
variable G such that
                       Z
            pdata (x) = p⋆ (x | g) p⋆ (g) dg,   H(X | G) ≪ H(X),       dim(G) ≪ dim(E(X)),           (3.34)

then the factorization of Cola DLM is closer to the true generative mechanism: the prior models the distribution
of G, while the decoder handles conditional realization. In this case, the latent bottleneck helps rather than
hurts.

Three governing curves and the applicability boundary.        The applicability of Cola DLM is ultimately
determined by three curves: the representation rate-distortion curve D(R), the prior approximation curve,
and the inference-gap curve GColaDLM
                             infer   . More compactly,
                                                     h                                 i
                    Cola DLM is advantageous ⇐⇒ D(R) is already small at low R
                                                     h                          i
                                                   ∧ E(MColaDLM ) decreases                         (3.35)
                                                     h                           i
                                                   ∧ GColaDLM
                                                        infer     is controllable .

The benefit of Cola DLM is therefore not guaranteed by diffusion or continuity alone. It depends on whether the
data exhibits a structure with low-dimensional global semantics and high-dimensional local token realization.

    Summary. The central advantage of Cola DLM is not denoising itself, but the latent decomposition that
    separates text modeling into a global prior and a conditional realization process.



4   Experiments
In this section, we conduct experiments to address the following research questions:
    • RQ1: Does a global semantic structure exist within the latent space?
    • RQ2: What type of latent space is optimal for text generation?
    • RQ3: Which diffusion process is most effective for text generation?
    • RQ4: Why use a continuous latent diffusion model for language modeling?


                                                            9
4.1     Experimental Setup
Datasets. For training, we use external open-source pretraining data. For evaluation, the internal component
analysis of Cola DLM (Sections 4.2, 4.3 and 4.4) is conducted on randomly sampled subsets from the test
sets of LAMBADA [74], MMLU [33], and SIQA [82]. LAMBADA is a continuation benchmark, whereas
the remaining two are multiple-choice benchmarks. For external comparisons (Section 4.5), we additionally
evaluate on the test sets of SQuAD [78], Story Cloze [68], OBQA [64], RACE [48], and HellaSwag [107].
Additional dataset details are deferred to Appendix H.1.
Baselines. In the internal component comparison experiments (Sections 4.2, 4.3 and 4.4), we specify the
different configurations of Cola DLM. In Section 4.5, for the scaling comparison, we independently train the
autoregressive and LLaDA baselines under strictly matched settings. Specifically, the autoregressive and
discrete diffusion models are randomly initialized using the official modeling implementations of LLaMA [92]
and LLaDA [70], respectively. Details are provided in Appendix H.2.
Metrics.   As discussed in Section 5.1, the estimated perplexity exhibits a substantial mismatch with the
actual generation quality of Cola DLM. Moreover, prior work [32, 34, 61, 97] has noted that perplexity is
not strictly correlated with generation performance. To enable the most objective and fair comparison, we
therefore evaluate all models under a unified few-shot setting across both multiple-choice and generative tasks.
For multiple-choice benchmarks and continuation tasks such as Lambada and SQuAD, accuracy is computed
by strict string matching between the model output and the ground-truth answer under predefined rules.
Additional evaluation details are provided in Appendix H.3.
Setup.   Cola DLM uses the OLMo 2 [71] tokenizer and is trained with AdamW. The learning rate starts
at 1 × 10−6 , is linearly warmed up to 1.5 × 10−4 over the first 5,000 steps, and is then decayed with a
cosine schedule to 1 × 10−5 by 1,000,000 steps. All evaluations are conducted using the checkpoint at the
corresponding FLOPs budget, without EMA weights. The same tokenizer, optimization and evaluation
settings are used for Cola DLM and all external baselines. In Cola DLM, the VAE contains 500M parameters
and the DiT contains 1.8B parameters. For the autoregressive and discrete diffusion baselines, the embedding
layer has approximately 400M parameters and the non-embedding backbone has 1.8B parameters. We thus
keep the total model size of the two model families at a comparable scale of roughly 2B parameters. All
methods are trained with the same random seed so that the training data are matched across runs, with the
maximum sequence length set to 512. Additional details are provided in Appendix H.4 and H.5.

4.2      Evidence of Global Semantic Structures in Cola DLM (RQ1)
In this section, we first present an implication for the existence of global semantic structures, and then provide
strong empirical evidence for their existence by quantitatively examining the performance of latent spaces
with different dimensions under different timestep shifts. The full theoretical derivation, proof, and technical
details are provided in Appendix E.

      Implication 1. If the latent representation is purely local and fully separable, then the optimal timeshift does
      not exhibit a stable drift as the latent dimension changes. Therefore, if the optimal timeshift is observed to shift
      systematically with the latent dimension in experiments, this indicates the existence of cross-dimensional shared
      structures in the latent space; if this phenomenon is mainly reflected in semantic metrics, it further supports
      that these shared structures are related to high-level semantics.

Based on Implication 1, the focus of this section is not merely on the specific optimal loc values under different
latent dimensions, but on whether the peak position of the optimal timeshift exhibits a stable and regular
shift as the latent dimension varies. Figure 2 presents the corresponding experimental results.
Obs. ❶ The optimal timeshift exhibits a systematic drift with the latent dimension. As shown in the left
panel of Figure 2, the best loc for Task Avg shifts from approximately 1.0 at d = 16, to approximately 1.7 at
d = 64, and further to approximately 2.3 at d = 128. This trend is clear and approximately monotonic. This
phenomenon directly contradicts the separable null hypothesis. A more plausible explanation is that changing
the latent dimension alters the effective noise calibration position of some cross-dimensional shared structure.



                                                              10
                                                         Evidence of Global Semantic Structures
                                           Peak Shift of Optimal Timeshift                                                      Best loc by Metric


                                                                                                           LAMBADA
                                                                                                                                                               2.2
                                                                                                                      loc=1.1          loc=2.0       loc=2.0
                            0.0                                                                                         33.2             33.2          31.1




Normalized Task Average
                                                                                                                                                               2.0
                           −2.5




                                                                                                                                                                     Best loc (timeshift)
                                                                                                           MMLU
                                                                                                                      loc=1.1          loc=1.7       loc=2.3
                           −5.0                                                                                         5.4              8.7           10.6    1.8




 (score - max over loc)
                           −7.5                                                                                                                                1.6

                                                                                                           SIQA
                                                                                                                      loc=1.0          loc=2.0       loc=2.3
                                                                                                                        15.6             17.6          23.3
                          −10.0
                                                                                                                                                               1.4

                          −12.5

                                                                                                           Task Avg
                                                                                                                      loc=1.0          loc=1.7       loc=2.3   1.2
                                                                           d = 16          d = 128                     17.33            19.47         24.97
                          −15.0                                            d = 64          Predicted loc
                                                                                                                                                               1.0
                                  0.50   0.75   1.00   1.25   1.50      1.75        2.00      2.25                    d = 16           d = 64        d = 128

                                                        Timeshift loc


Figure 2 Evidence of global semantic structures in the latent space. Left: as latent dimension increases, optimal
timeshifts move toward larger locations, with empirical peaks matching predictions. Right: most of the evaluation
metrics consistently prefer larger best locations at higher dimensions. This stable cross-metric trend supports shared
global semantic structures within the latent space.


Obs. ❷ This trend is consistent across multiple semantic metrics. The right panel of Figure 2 further shows
that, although the best loc values for LAMBADA, MMLU, SIQA, and Task Avg are not exactly the same,
they all overall favor larger loc regions as the latent dimension increases. This indicates that the peak drift is
not an accidental fluctuation of any single task, but rather a stable phenomenon jointly supported by multiple
semantic evaluations. Therefore, what is modulated by timeshift is a representation structure shared across
different semantic tasks.
Obs. ❸ The empirical peaks are broadly consistent with the theoretical predictions. In the left panel of Figure 2,
the dashed lines indicate the theoretically predicted optimal loc values. It can be seen that, under all three
latent-dimension settings, the empirical peaks are close to the predicted positions, and the drift directions are
fully consistent. This suggests that the observed drift is not an arbitrary empirical hyperparameter effect, but
is instead consistent with the theoretical analysis in Appendix E, namely that shared latent structures lead to
dimension-dependent timeshift calibration.
Implication 1 provides a rigorous contrapositive statement for the existence of global semantic structures,
while the experimental results, by providing reverse evidence for this contrapositive, offer strong empirical
support for the existence of shared and semantically relevant global structures in the latent space of Cola
DLM. This also provides supporting evidence for the first condition in Eq. (3.35) of Section 3.3.2, thereby
further supporting the advantage of Cola DLM.

4.3                        Analysis of Different Latent Spaces in Cola DLM (RQ2)
In this section, we present a detailed empirical study of different latent spaces through both quantitative
evaluation and visualization. We analyze the design of the latent space from three perspectives: whether it
should be dynamic or static, what latent dimensionality is most appropriate, and how semantic smoothness
contributes to latent-space quality. Based on these analyses, we identify the most effective directions for
optimizing the latent space.

Fixed vs. Evolving Latent Space. As shown in Figure 3, this section studies whether the latent space should
evolve jointly with DiT during training. Under the same compute budget, we compare five strategies: fixing a
pretrained VAE (Fix VAE); initializing the VAE from pretrained weights and jointly training it with DiT
using a VAE learning rate equal to that of DiT or scaled to 0.01× (Joint DiT x1 / Joint DiT x0.01); jointly
training both VAE and DiT from random initialization with the same learning rate (All Scratch x1); and an


                                                                                               11
                        Fix VAE          Joint DiT x1         Joint DiT x0.01                                       Fix VAE          Joint DiT x1         Joint DiT x0.01
                                  All Scratch x1         Interval                                                             All Scratch x1         Interval

                                         Tasks Average                                                                                  LAMBADA
           25.4                                                                                        38.2



           21.8                                                                                        32.5




Accuracy                                                                                    Accuracy
            19                                                                                         29.6



           14.3                                                                                        18.8



           10.7                                                                                        14.6



             3                                                                                          3.9
                  117     156      195        234       272      311       350   389                          117     156      195        234       272      311       350   389
                                      Total FLOPs (EFLOPs)                                                                        Total FLOPs (EFLOPs)



                        Fix VAE          Joint DiT x1         Joint DiT x0.01                                       Fix VAE          Joint DiT x1         Joint DiT x0.01
                                  All Scratch x1         Interval                                                             All Scratch x1         Interval

                                               MMLU                                                                                         SIQA
           13.4                                                                                        25.3



           11.1                                                                                         22




Accuracy                                                                                    Accuracy
            8.5                                                                                        19.9



            7.1                                                                                        13.6



            4.7                                                                                        10.2



            1.9                                                                                         3.2
                  117     156      195        234       272      311       350   389                          117     156      195        234       272      311       350   389
                                      Total FLOPs (EFLOPs)                                                                        Total FLOPs (EFLOPs)




Figure 3 Comparison between fixed and evolving latent spaces. Across Task Average, LAMBADA, MMLU,
and SIQA, joint evolution with DiT achieves the best overall scaling and performance when initialized from a stable
pretrained VAE. Fixed spaces lead to earlier saturation, while training from scratch or interval updates remains less
effective. This suggests the space should evolve with DiT, but from a stable initialization rather than scratch.


interval-based strategy (Interval), where each 5k-step cycle consists of 2k steps of joint training followed by
3k steps with the VAE frozen. The overall results suggest that the latent space should neither remain fully
fixed nor be jointly optimized from scratch without constraint. Instead, the most effective strategy is to let it
evolve together with DiT on top of a stable initialization.
Obs. ❶ Joint DiT x1 shows the strongest scaling potential. At small compute budgets, Fix VAE and Joint
DiT x1 are close, and Fix VAE is sometimes slightly better. As FLOPs increase, however, Joint DiT x1
improves more steadily and achieves the best final results on Task Avg, LAMBADA, MMLU, and SIQA,
whereas Fix VAE gradually saturates. This indicates that a fixed latent space helps early stability but limits
the performance ceiling, while continuous co-adaptation with DiT is more beneficial for scaling.
Obs. ❷ The benefit of joint training depends on good initialization rather than trainability alone. All Scratch x1
performs consistently worse than the other methods across all metrics, and its gains remain limited throughout
training. This suggests that the advantage of Joint DiT x1 does not come from making the latent space
trainable by itself; it relies on starting from a meaningful pretrained latent space and then adapting it jointly
with DiT.
Obs. ❸ The latent-space visualization explains why All Scratch underperforms. Figure 4 shows that All


                                                                                       12
Figure 4 Visualization of latent spaces under different training strategies. Joint optimization on a stable
initialization yields a more structured, semantically organized latent space than training VAE and DiT from scratch.
Increasing the latent dimension (16 to 128) partially mitigates collapse but remains less structured than the stable-
initialization approach.


Scratch with d = 16 yields a more collapsed and less structured latent space, with trajectories dominated by
simple outward drift. Increasing the latent dimension to 128 partially alleviates this issue, but the geometry
still remains less organized than that of Joint DiT with stable initialization. In contrast, Joint DiT produces
more heterogeneous latent patterns and richer trajectories, suggesting a more structured and semantically
usable space.
Obs. ❹ Effective latent evolution requires both continuous participation and sufficient update strength. Joint
DiT x0.01 and Interval are both better than All Scratch x1, but still clearly worse than Joint DiT x1 in overall
trend and final performance. This shows that partial latent participation is not enough: overly weak updates
slow adaptation, while periodic freezing disrupts co-evolution with DiT. A better strategy is to update the
latent space continuously and strongly, while keeping the initialization stable.
Overall, Figure 3 and Figure 4 consistently show that the best latent-space strategy is neither to keep it fixed
nor to train it from scratch, but to let it evolve jointly with DiT on top of a good initialization. When the
VAE and DiT are trained jointly, the VAE is exposed to more data and can fit pdata (z|x) more accurately.
This further verifies the last condition in Eq. (3.35) of Section 3.3.2, and provides strong support for the
potential advantage of Cola DLM. Additional results are in Appendix H.6.

Dimensionality of the Latent Space. We next
                                                  Table 2 Dimensionality of the latent space under 117 EFLOPs.
study how the latent dimensionality affects       Larger latent dimensions improve the overall average under the
both performance and latent-space quality.        all-scratch setting with loc = 1.
Table 2 compares All Scratch models with
different latent dimensions under the same         Method                         Lambada MMLU SIQA Avg.
EFLOPs budget (117), Figure 4 provides the
corresponding latent-space visualization, and      All Scratch, d = 16, loc=1       14.3    6.9     4.9   8.7
Figure 2 shows how the optimal timeshift           All Scratch, d = 64, loc=1       20.9    5.4     7.6 11.3
changes with dimension. Taken together,            All Scratch, d = 128, loc=1      18.5    8.1     8.9   11.8
these results suggest that increasing the la-
tent dimension partially alleviates collapse and improves semantic capacity, but it also changes the effective
noise calibration of the latent space.
Obs. ❶ Increasing the latent dimension improves the overall semantic capacity under the same compute
budget. As shown in Table 2, the average score increases from 8.7 at d = 16 to 11.3 at d = 64, and further to
11.8 at d = 128. The improvement is most evident on MMLU and SIQA. Although LAMBADA peaks at
d = 64, the overall trend still suggests that a larger latent space carries stronger semantic capacity under the


                                                         13
            No BERT loss      BERT loss         No BERT loss    BERT loss                       No BERT loss      BERT loss         No BERT loss    BERT loss
            (lr=0.01)         (lr=0.01)         (lr=1)          (lr=1)                          (lr=0.01)         (lr=0.01)         (lr=1)          (lr=1)



                                     Tasks Average                                                                            LAMBADA
           26.9                                                                                40.1


           24.3                                                                                36.5




Accuracy                                                                            Accuracy
           22.7                                                                                 34


           20.4                                                                                31.5



           18.4                                                                                29.7



           14.4                                                                                25.6
                  117   156    195        234    272     311   350       389                          117   156    195        234    272     311   350       389
                                  Total FLOPs (EFLOPs)                                                                Total FLOPs (EFLOPs)



            No BERT loss      BERT loss         No BERT loss    BERT loss                       No BERT loss      BERT loss         No BERT loss    BERT loss
            (lr=0.01)         (lr=0.01)         (lr=1)          (lr=1)                          (lr=0.01)         (lr=0.01)         (lr=1)          (lr=1)



                                          MMLU                                                                                 SIQA
           13.6                                                                                 27



           12.3                                                                                25.5




Accuracy                                                                            Accuracy
           10.9                                                                                22.6



            9.9                                                                                19.9


            7.9                                                                                16.2



             6                                                                                  9.4
                  117   156    195        234    272     311   350       389                          117   156    195        234    272     311   350       389
                                  Total FLOPs (EFLOPs)                                                                Total FLOPs (EFLOPs)




Figure 5 Effect of semantic smoothness in the latent space under the Joint DiT setting. Adding a BERT-style
loss consistently improves performance, with larger gains during active latent updates (lr = 1). This suggests semantic
smoothness benefits latent-space quality, especially when evolving jointly with DiT.


same compute budget.
Obs. ❷ A larger latent dimension partially alleviates latent-space collapse, but does not fully solve it.
Figure 4 shows that increasing the dimension from 16 to 128 makes the latent space less collapsed and more
dispersed. However, the resulting geometry still remains clearly less structured than Joint DiT with stable
initialization. This indicates that increasing dimensionality is helpful, but cannot by itself replace proper
latent-space formation.
Obs. ❸ The effect of latent dimensionality is not only geometric, but also dynamical. Figure 2 shows that the
best timeshift systematically shifts toward larger loc values as the latent dimension increases. This means
that increasing the latent dimension does not merely enlarge the space; it also changes the denoising scale
at which semantic information is best recovered. Therefore, the benefit of a higher-dimensional latent space
depends not only on improved geometry, but also on proper noise calibration.
Overall, Table 2, Figure 4, and Figure 2 present a consistent picture: increasing the latent dimension improves
latent-space quality and downstream performance, but the gain is only partial, and its full benefit still depends
on proper training dynamics and timeshift calibration.




                                                                               14
Table 3 Performance under different VAE logSNR settings. VAE logSNR strongly affects downstream performance.
A learnable setting gives the best overall results, while fixed logSNR = 1.5 is the strongest fixed alternative.

Compute Budget                               EFLOPs = 77.86                         EFLOPs = 116.78
Method                           Lambada       MMLU     SIQA      Avg.    Lambada      MMLU     SIQA     Avg.
Fixed VAE logSNR = 1.0              27.1        5.7        11.3   14.70     30.4        7.7      18.4    18.83
Fixed VAE logSNR = 1.5              29.5        7.8        17.5   18.27     33.8        8.0      23.6    21.80
Fixed VAE logSNR = 2.0              30.9        5.1        14.3   16.77     32.7        9.7      19.5    20.63
Learnable VAE logSNR (≈ 4.5)        32.6        7.9        16.2   18.90     34.6        10.1     21.6     22.1



Semantic Importance of the Latent Space. As shown in Figure 5, all results in this subsection are obtained
under the Joint DiT setting, where the VAE is initialized from pretrained weights and jointly optimized with
DiT. We further compare whether to add a BERT-style loss in VAE training, which encourages the latent
space to preserve smoother local semantics. Here, the reported lr denotes the VAE learning-rate ratio relative
to DiT. The results show that such semantic smoothness is important for downstream performance, especially
when the latent space is allowed to evolve more actively.
Obs. ❶ Adding BERT loss consistently improves performance when the latent space is actively updated. When
the VAE learning-rate ratio is 1, BERT loss gives the best overall results across nearly the entire training
range. In Figure 5, the BERT-loss curve consistently outperforms its no-BERT counterpart on Task Average,
LAMBADA, MMLU, and SIQA, and also achieves the best final performance. This indicates that encouraging
masked-token recoverability makes the latent space more semantically useful for downstream prediction.
Obs. ❷ Strong latent evolution is effective only with semantic guidance. When the VAE learning-rate ratio is
0.01, adding BERT loss brings only limited gains, whereas its advantage becomes clear and stable when the
ratio is increased to 1. At the same time, simply increasing the VAE update strength without BERT loss does
not reliably improve performance and is even weaker at several later-stage points. This shows that trainability
alone is not sufficient: when the latent space evolves more actively, its updates must also be constrained
toward a semantically smoother organization.
Overall, Figure 5 shows that semantic smoothness is an important property of a useful latent space. It not
only improves final performance, but also makes joint latent evolution substantially more effective and stable.
These results suggest that the latent should be compact but semantically sufficient, consistent with Eq. (3.5)
and Eq. (3.35). The BERT-style loss helps retain useful semantics under the bottleneck.

Smoothness of the Latent Space. Table 3 compares different VAE logSNR settings at two compute budgets.
The results show that the VAE logSNR is an important factor for latent-space smoothness and downstream
performance. Under the current setup, learning the VAE logSNR gives the strongest overall results, while
fixing the VAE logSNR at 1.5 is the most competitive fixed alternative. The VAE logSNR formula is given in
Appendix H.7.
Obs. ❶ Learning the VAE logSNR gives the strongest overall performance under the current setup. At both
77.86 and 116.78 EFLOPs, the learnable VAE logSNR setting achieves the best Task Average in Table 3. It
also gives the best LAMBADA results at both checkpoints and the best MMLU result at the higher compute
budget. This suggests that keeping the VAE logSNR learnable is currently the strongest overall choice, likely
because it allows a more flexible smoothness profile during latent-space training.
Obs. ❷ Fixing the VAE logSNR at 1.5 is the strongest fixed alternative. Although the learnable VAE logSNR
setting ranks first on average, fixing the VAE logSNR at 1.5 remains very close at both compute budgets. It
also consistently achieves the best SIQA results and stays competitive on the other tasks. This indicates that
a properly chosen fixed VAE logSNR can already provide a strong balance between semantic preservation and
optimization stability.
Obs. ❸ The current results favor a learnable VAE logSNR, while still leaving room for further study of fixed
settings. The advantage of the learnable VAE logSNR over fixing the VAE logSNR at 1.5 is consistent but


                                                      15
                      block size=1    block size=64    Tasks Average                          block size=1    block size=64    Tasks Average
                      block size=16   block size=128                                          block size=16   block size=128


                        116.78 EFLOPs (30k steps)                                               155.71 EFLOPs (40k steps)
           35                                                                          35

           30                                                                          30

           25                                                                          25




Accuracy                                                                    Accuracy
           20                                                                          20

           15                                                                          15

           10                                                                          10

            5                                                                           5

            0                                                                           0
                    LAMBADA            MMLU                SIQA                             LAMBADA            MMLU                SIQA
                                        Task                                                                    Task




Figure 6 Impact of DiT block size. A moderate block size (especially 16) achieves the best overall performance.
Overly large blocks degrade results, while size 1 remains competitive but weaker than 16.


not large, suggesting that the current conclusion is clear but not yet definitive. Since Table 3 only reports two
compute budgets, the scaling behavior of different VAE logSNR settings remains open and deserves more
systematic study.
Overall, Table 3 shows that the VAE logSNR is an important factor in shaping latent-space smoothness and
downstream performance. Under the current setup, a learnable VAE logSNR is the strongest overall choice,
while fixing the VAE logSNR at 1.5 stands out as a highly competitive fixed alternative.

4.4             Ablation on the Diffusion Process in Cola DLM (RQ3)
In this section, we systematically study the training and inference design choices of the DiT module through
ablations. By combining quantitative results with visualizations, we further analyze the mechanisms behind
the observed optimization trends. On the training side, we investigate DiT models with different block sizes
and examine the effect of different noise training schedules on downstream performance. On the inference side,
we study the impact of the number of denoising steps and the choice of Classifier-Free Guidance (CFG) scales.

4.4.1           Training Stage
DiT Block Size. As shown in Figure 6, all results in this subsection are obtained under the Joint DiT setting:
the VAE is initialized from pretrained weights, the VAE and DiT are jointly optimized with the same learning
rate, and the training noise schedule uses loc= 1. We compare four DiT block sizes at two training checkpoints
to study how the local processing granularity affects downstream performance. The results show that block
size has a clear effect under the current setting, and that a moderate block size works best.
Obs. ❶ Block size 16 gives the best overall performance at both checkpoints. At both 30K and 40K checkpoints,
block size 16 achieves the highest Task Average in Figure 6. It also delivers the strongest or near-strongest
results on all three benchmarks, especially on LAMBADA and MMLU. This suggests that, under the current
setup, a moderate block size provides a favorable trade-off between local modeling capacity and semantic
aggregation.
Obs. ❷ Larger block sizes are generally less effective under the current setting. When the block size is
increased from 16 to 64 and 128, performance drops clearly on all three tasks at both checkpoints, with
especially visible degradation on SIQA and MMLU. This suggests that overly coarse block partitioning may
weaken useful semantic interactions inside the latent sequence. At the same time, since the training noise
schedule is fixed to loc= 1 here, we do not exclude the possibility that different block sizes may favor different
noise calibrations.
Obs. ❸ Block size 1 is competitive but still weaker than block size 16. Block size 1 remains a relatively



                                                                       16
                         Joint DiT         Joint DiT         Joint DiT                                             Joint DiT       Joint DiT         Joint DiT
                         (loc0)            (loc1)            (uniform)                                             (loc0)          (loc1)            (uniform)
                          Fix VAE          Fix VAE           Fix VAE                                                Fix VAE         Fix VAE          Fix VAE
                          (loc0)           (loc1)            (uniform)                                              (loc0)          (loc1)           (uniform)
                                         Tasks Average                                                                             LAMBADA
            25                                                                                       37.1



           21.2                                                                                      32.3




Accuracy                                                                                  Accuracy
           17.5                                                                                      29.5


            15                                                                                       28.2



           13.4                                                                                      26.4


           10.2                                                                                      23.6
                  117   156        195     234         272     311       350   389                          117   156        195    234        272     311       350   389
                                     Total FLOPs (EFLOPs)                                                                      Total FLOPs (EFLOPs)



                         Joint DiT         Joint DiT         Joint DiT                                             Joint DiT       Joint DiT         Joint DiT
                         (loc0)            (loc1)            (uniform)                                             (loc0)          (loc1)            (uniform)
                          Fix VAE          Fix VAE           Fix VAE                                                Fix VAE         Fix VAE          Fix VAE
                          (loc0)           (loc1)            (uniform)                                              (loc0)          (loc1)           (uniform)
                                            MMLU                                                                                      SIQA
           13.5                                                                                      25.1



           10.3                                                                                       21




Accuracy                                                                                  Accuracy
            7.1                                                                                      16.7



            5.3                                                                                      11.3



            3.5                                                                                       9.4



            0.5                                                                                       5.9
                  117   156        195     234         272     311       350   389                          117   156        195    234        272     311       350   389
                                     Total FLOPs (EFLOPs)                                                                      Total FLOPs (EFLOPs)




Figure 7 Noise-schedule ablation. Across all the tasks, loc = 1 gives the strongest overall performance, especially
under Joint DiT, while uniform schedules are generally weaker. This suggests that noise-schedule calibration is
important and becomes more beneficial when the latent space evolves jointly with DiT.


strong baseline and generally outperforms block sizes 64 and 128. However, it is still below block size 16 in
Task Average at both checkpoints, and is notably weaker on MMLU. This suggests that fully fine-grained,
completely causal processing is not necessarily the optimal way to model text in this setting, and that some
degree of local grouping can be beneficial.
Overall, Figure 6 shows a clear pattern: under the current setting with loc= 1, DiT block size should be
neither too small nor too large. A moderate block size, especially 16, provides the most effective balance and
leads to the best overall performance in the current experiments.

Noise Schedule. As shown in Figures 8 and 7, all results in this subsection are obtained with latent dimension
d = 16 under the Joint DiT setting: the VAE is initialized from pretrained weights, and the VAE and DiT are
jointly optimized with the same learning rate. We vary the schedule location parameter to study how noise
calibration affects downstream performance, and include Fix VAE curves in Figure 7 as references. From
the information-theoretic analysis in Appendix G, changing the schedule location is not merely changing a
training-time heuristic: it effectively shifts the logSNR trajectory of the denoising process, and therefore
changes how much semantic information remains available in the latent at different timesteps. The timestep
shift formula and visualizations are provided in Appendix H.9 and H.8.



                                                                                     17
                               loc=0.5   loc=0.9   loc=1.1                                         loc=0.5   loc=0.9   loc=1.1
                               loc=0.8   loc=1.0   Tasks Average                                   loc=0.8   loc=1.0   Tasks Average


                         116.78 EFLOPs (30k steps)                                           155.71 EFLOPs (40k steps)

           30                                                                       30

           25                                                                       25




Accuracy                                                                 Accuracy
           20
                                                                                    20

           15
                                                                                    15

           10
                                                                                    10

            5
                                                                                     5

                     LAMBADA             MMLU                SIQA                        LAMBADA             MMLU                SIQA
                                          Task                                                                Task




Figure 8 Noise-schedule comparison at different training checkpoints. At both checkpoints, loc = 1.0 achieves
the best Task Average and the most balanced overall performance across tasks. This indicates that the preferred
schedule location is stable across training.



            Implication 2. If the schedule location shifts the logSNR curve, then it also shifts the effective semantic-
            information regime seen by the DiT during denoising. Therefore, the best noise schedule is the one whose
            logSNR trajectory is best aligned with the latent space and the semantic scale to be recovered, rather than a
            universally fixed timestep parameterization.

Obs. ❶ A moderate schedule location around loc= 1.0 gives the best overall performance under the current
setting. Figure 8 shows that loc= 1.0 achieves the highest Task Average at both the 30K and 40K checkpoints.
It also gives the best or near-best results on the three tasks, with especially clear gains on MMLU and SIQA.
From the information-theoretic view developed in Appendix G, this suggests that loc= 1.0 places the denoising
trajectory in a more suitable effective logSNR range for semantic recovery, whereas both smaller and larger
shifts move the model away from that regime.
Obs. ❷ Proper noise calibration is especially important for Joint DiT. Figure 7 further shows that Joint DiT
with loc= 1 is the strongest trainable setting across Task Average, LAMBADA, MMLU, and SIQA, whereas
Joint DiT with loc= 0 or a uniform schedule remains clearly weaker throughout training. Moreover, Joint
DiT with loc= 1 eventually matches or surpasses the corresponding Fix VAE baselines, while the mismatched
schedules do not. This indicates that joint latent evolution becomes effective only when the denoising logSNR
trajectory is aligned with the semantic structure of the evolving latent space.
Obs. ❸ The effect of noise schedule should be understood through semantic-information calibration
rather than as an isolated hyperparameter effect. Appendix E further implies that schedule location,
latent dimension, and VAE logSNR all act on the same core object, namely the effective mutual-information
curve of the semantic variable along diffusion time. From this perspective, the sensitivity observed here is
not accidental: changing the noise schedule changes where the model spends its denoising capacity on the
semantic-information axis. This also helps explain why different latent dimensions, different VAE smoothness
settings, and potentially different DiT block sizes need not share the same optimal schedule.
Overall, Figures 8 and 7 show that the noise schedule is a key component of the training setup. Under the
current Joint DiT setting with d = 16, a properly calibrated schedule, especially loc= 1.0, is important not only
for stable optimization, but more fundamentally for aligning denoising with the effective semantic-information
regime of the latent space. As implied by Eq. (3.5), this will further improve the average ELBO and is
therefore theoretically well founded.




                                                                    18
                             LAMBADA     MMLU          SIQA                                                       LAMBADA        MMLU          SIQA
                                SQuAD     Tasks Average                                                              SQuAD        Tasks Average


                       Task-wise Accuracy Across Infer Steps                                              Task-wise Accuracy Across CFG Values
           54.5                                                                          54.2



           43.6                                                                          43.3



           32.7                                                                          32.5


Accuracy                                                                      Accuracy
           21.8                                                                          21.7



           10.9                                                                          10.8



             0                                                                             0
                  1      2   4    6     8       10      16    32   64   128                     0   0.5   1   3    4   5     6   6.5    7    7.5      8   9   10   15   20   60
                                        Infer Step                                                                                 CFG


                        (a) Impact of denoising steps                                                         (b) Impact of CFG scales

Figure 9 Impact of inference-time hyperparameters. Increasing denoising steps brings clear early gains but
quickly saturates, while a moderate CFG value achieves the best overall performance.


4.4.2                 Inference Stage
Denoising Steps. As shown in Figure 9a, all results in this subsection are obtained under the Joint DiT setting:
the VAE is initialized from pretrained weights, and the VAE and DiT are jointly optimized with the same
learning rate. We vary the number of denoising steps at inference time to study the efficiency–performance
trade-off. The results show that increasing the number of steps is highly beneficial in the low-step regime,
while the gain quickly saturates as the inference budget becomes larger.
Obs. ❶ Increasing denoising steps yields a clear early improvement. From 1–2 steps to 4–8 steps, all tasks
improve substantially. The gain is especially large on LAMBADA, while SQuAD, SIQA, and Task Average
also increase sharply. This indicates that very few denoising steps are insufficient for stable semantic recovery.
Obs. ❷ Performance saturates after a moderate number of steps. After roughly 16–32 steps, the Task
Average becomes nearly flat, and the marginal gain from additional steps is very limited. A similar saturation
pattern is also visible on SIQA and SQuAD. This suggests that most useful denoising progress is already
completed within a moderate inference budget.
Obs. ❸ Most of the practical gain is achieved with only 8--10 denoising steps. From an efficiency perspective,
8–10 steps already recover most of the final performance. Since our DiT uses a block size of 16, this means 16
tokens can be generated with only 8–10 sequential denoising iterations, corresponding to an idealized 1.6–2.0×
reduction in sequential generation depth compared with AR decoding.
Overall, Figure 9a shows that denoising steps are important, but more is not always better. Under the Joint
DiT setting, a moderate number of inference steps, around 10–32, already provides a strong trade-off between
accuracy and efficiency.

Classifier-Free Guidance (CFG) Scales. As shown in Figure 9b, all results in this subsection are obtained
under the Joint DiT setting: the VAE is initialized from pretrained weights, and the VAE and DiT are jointly
optimized with the same learning rate. We vary the Classifier-Free Guidance (CFG) scale at inference time
to study how guidance strength affects downstream performance. The results show a clear non-monotonic
pattern: increasing CFG is helpful at first, but overly large values significantly hurt performance.
Obs. ❶ A moderate CFG scale gives the best overall performance. The Task Average rises rapidly as CFG
increases from 0 to around 3–6, and then stays near its best region for a moderate range of values. This
indicates that an appropriate amount of guidance substantially improves conditional denoising and semantic
recovery.
Obs. ❷ Excessive guidance leads to clear degradation. After the moderate optimum region, all task curves



                                                                                  19
                                                                      AR                                  LLaDA                          Cola DLM (ours)
                                        MMLU                                                                            RACE                                                                        Story Cloze
                                                                                           16                                                                               35
           16
                                                                                           14
           14                                                                                                                                                               30
                                                                                           12
           12                                                                                                                                                               25



Accuracy                                                                        Accuracy                                                                         Accuracy
                                                                                           10
           10                                                                                                                                                               20
                                                                                            8
            8
                                                                                            6                                                                               15
            6
                                                                                            4                                                                               10
            4
                                                                                            2
                                                                                                                                                                             5
            2                                                                               0
                0   250   500     750   1000   1250   1500   1750   2000                        0   250   500     750    1000   1250   1500   1750   2000                        0   250   500     750   1000   1250   1500   1750   2000
                                Total FLOPs (EFLOPs)                                                            Total FLOPs (EFLOPs)                                                             Total FLOPs (EFLOPs)

                                   LAMBADA                                                                              OBQA                                                                        HellaSwag
                                                                                      22.5
           55                                                                                                                                                               14
                                                                                      20.0
           50                                                                                                                                                               12
                                                                                      17.5

           45                                                                         15.0                                                                                  10



Accuracy                                                                   Accuracy                                                                              Accuracy
           40                                                                         12.5                                                                                   8

                                                                                      10.0                                                                                   6
           35

                                                                                       7.5                                                                                   4
           30
                                                                                       5.0                                                                                   2
           25
                                                                                       2.5                                                                                   0
                0   250   500     750   1000   1250   1500   1750   2000                        0   250   500     750    1000   1250   1500   1750   2000                        0   250   500     750   1000   1250   1500   1750   2000
                                Total FLOPs (EFLOPs)                                                            Total FLOPs (EFLOPs)                                                             Total FLOPs (EFLOPs)

                                        SIQA                                                                            SQuAD                                                                     Tasks Average
           30

                                                                                           40                                                                          25.0
           25
                                                                                                                                                                       22.5

           20                                                                              30                                                                          20.0




Accuracy                                                                        Accuracy                                                                    Accuracy
                                                                                                                                                                       17.5
           15
                                                                                           20                                                                          15.0

                                                                                                                                                                       12.5
           10
                                                                                           10                                                                          10.0
            5
                                                                                                                                                                        7.5

                                                                                            0
                0   250   500     750   1000   1250   1500   1750   2000                        0   250   500     750    1000   1250   1500   1750   2000                        0   250   500     750   1000   1250   1500   1750   2000
                                Total FLOPs (EFLOPs)                                                            Total FLOPs (EFLOPs)                                                             Total FLOPs (EFLOPs)



Figure 10 Overall scaling performance under a unified few-shot generative evaluation protocol. Across eight
benchmarks and Task Average, Cola DLM exhibits strong scaling dynamics, ultimately reaching the best average
performance. It should be noted that the lower absolute accuracy observed on specific multiple-choice tasks is an
anticipated consequence of the rigorous generative evaluation paradigm; nevertheless, the underlying scaling trends are
robustly preserved. These findings imply that continuous latent prior modeling possesses significant scaling potential,
rendering the current performance a conservative measure of its true capacity.


begin to decline as CFG becomes larger. The drop is especially pronounced beyond CFG ≈ 10, and becomes
severe at very large values such as 20 and 60. This shows that overly strong guidance distorts the denoising
trajectory rather than improving it.
Overall, Figure 9b shows that CFG is an important inference-time hyperparameter. Under the Joint DiT
setting, a moderate CFG scale provides the best trade-off, while both weak guidance and excessive guidance
lead to inferior results.

4.5                 Comparison of Scaling Performance (RQ4)
In this section, we compare the scaling behavior of Cola DLM with strictly matched AR and LLaDA baselines
under the best configuration identified by the previous tuning experiments. Specifically, Cola DLM uses latent
dimension d = 16, block size 16, joint VAE–DiT training with a VAE/DiT learning-rate ratio of 1, BERT loss,
and a logit-normal training noise schedule with loc = 1; at inference time, we use 16 denoising steps and CFG
= 7. The AR and LLaDA baselines are matched in scale, with the non-embedding backbone controlled at
1.8B parameters, and LLaDA uses a denoising length equal to the generation length during inference.



                                                                                                                        20
It is also worth noting that the absolute scores in Figure 10 are relatively low mainly on the multiple-
choice benchmarks. This is because, for a fair comparison, all models are evaluated under a unified few-shot
generative protocol rather than standard likelihood-based classification: LAMBADA and SQuAD are evaluated
as generative tasks, while the remaining benchmarks are multiple-choice tasks but are also cast into few-shot
generation. As discussed in Section 5.1, likelihood estimation can be substantially misaligned with the actual
generation quality of Cola DLM. Therefore, although the absolute values on multiple-choice tasks are lower
than those in conventional discriminative evaluation, the relative scaling trends remain informative and fair
under this fully matched protocol.
As shown in Figure 10, Cola DLM exhibits strong overall scaling behavior, with increasingly encouraging
gains as the compute budget grows.
Obs. ❶ Cola DLM shows one of the strongest overall scaling trends. On Task Average, Cola DLM improves
steadily across the full compute range and reaches the best final performance. AR remains competitive at
smaller budgets, and LLaDA also shows clear early gains, but the curve of Cola DLM rises more persistently
toward the high-compute regime. This suggests that Cola DLM already exhibits highly competitive, and at
larger budgets stronger, scaling potential under the current matched setting.
Obs. ❷ The scaling advantage of Cola DLM is especially clear on reasoning-intensive and global-semantic
tasks. On MMLU, RACE, Story Cloze, and OBQA, Cola DLM maintains a strong upward trend and
achieves the best or near-best performance across a wide compute range. The gains are particularly visible at
medium-to-large budgets, indicating that continuous latent prior modeling is well suited to tasks that rely
more on global semantic organization and holistic answer formation.
Obs. ❸ On generative tasks, Cola DLM also shows encouraging scaling behavior. For LAMBADA and
SQuAD, the scaling trends remain clear under the unified generative evaluation protocol. On LAMBADA,
Cola DLM improves steadily with compute and remains close to AR at larger budgets, while SQuAD shows
a particularly clear gain with scale, where Cola DLM eventually surpasses AR and continues to approach
the strong performance region of LLaDA. These results suggest that, on generation-oriented evaluation, Cola
DLM already demonstrates scaling behavior comparable to strong baselines, with encouraging headroom as
compute increases.
Obs. ❹ The current result is a conservative estimate of the scaling potential of Cola DLM. The present
comparison is conducted under a relatively conservative configuration of Cola DLM. Earlier ablations already
show that increasing the latent dimension from 16 to 128 can improve semantic capacity, and the analysis of
logSNR also suggests that the current setting still leaves additional room for scaling. Therefore, Figure 10
should be viewed as evidence that Cola DLM already scales well under a restrained setting, rather than as
the upper bound of its capability.
Overall, Figure 10 supports a consistent conclusion: under a strictly matched comparison and a unified
generative evaluation protocol, Cola DLM exhibits scaling behavior that is fully competitive with strong AR
and diffusion-based baselines, and on several tasks already shows particularly encouraging late-stage gains.
Together with the remaining optimization headroom in latent-space design, these results provide supportive
evidence that continuous latent prior modeling is a promising scaling direction for language modeling.

5     Discussion
In this section, we further examine several additional properties and extensions of Cola DLM. We focus
on the structural gap between likelihood-oriented estimation and generation quality, analyze how different
conditioning and padding strategies affect block-causal Cola DLM in the first generation block, and present
a preliminary exploration of VAE-based text compression for faster generation. Finally, we highlight the
broader potential of Cola DLM for combining with other continuous modalities.

5.1   The Structural Gap Between Likelihood-Oriented Estimation and Generation Quality
This section studies a central phenomenon in continuous latent language models: generation quality can
already be reasonable while likelihood-oriented PPL remains poor. The key reason is that these two metrics


                                                     21
Figure 11 A local view of the mismatch between likelihood-oriented estimation and generation quality. Top:
local latent geometry around representative ground-truth tokens. Bottom: corresponding prior-density landscapes.
High decoder probe success and posterior hit contrast with sharply varying prior hit and density alignment. Thus, good
generation relies on covering decoder-valid regions, while likelihood estimation also demands precise local calibration
around the gold posterior.


target different properties. Generation only requires the prior mass to reach semantically decoder-valid regions,
whereas likelihood-oriented estimation additionally requires accurate local probability calibration around the
posterior neighborhood of the ground-truth target.
Let
                                                     x = (xpre , xres ),
where xpre is the prefix, xres is the response, and c denotes the conditional information induced by the prefix.
The exact conditional marginal is
                                                  Z
                                     p(x | c) = pθ (xres | z, c) pψ (z | c) dz,
                                         res
                                                                                                          (5.1)

while the practically accessible quantity is the local score
                                           h                                                         i
                    Sresp (x) = Eqϕ (z|x,c) log pθ (xres | z, c) + log pψ (z | c) − log qϕ (z | x, c) .          (5.2)

The mismatch between these two quantities is the starting point of our analysis.


                                                             22
Sample text. At dawn the research vessel Meridian slipped out of the harbor and followed a chain of islands that looked like
dark brushstrokes on the horizon. Mira stood at the bow with a notebook pressed against her jacket, listening as the engine
settled into a steady hum and the crew argued about . . .
Ground-truth Posterior     Prior        Decoder      Likelihood-derived Generated   Gen. PPL ↓    Gen. PPL ↓
   token     log p(z | x) log p(z)    log p(x | z)         PPL ↓          token   of generated of ground-truth
Direct training (unfixed VAE logSNR; measured effective VAE logSNR ≈ 4.5)
    at           18.70      4.74     −3.62 × 10−4       1.15 × 106        on            3.83              6.90
    her          22.61      2.26     −8.58 × 10−6       6.93 × 108        her           1.12              1.12
  settled        32.02      18.45    −8.54 × 10−3       7.88 × 105        of            5.05           2.58 × 103
Fixed VAE logSNR = 1.0
    at           -15.81     -22.28   −2.10 × 10−5         641.57           in            6.69             6.90
    her          -11.79     -15.68   −1.91 × 10−5         48.98           the           14.17             1.12
  settled        -12.56     -21.51   −1.40 × 10−3         7740             of           5.05           2.58 × 103
Fixed VAE logSNR = 1.5
    at           -8.04      -13.54   −7.06 × 10−5        245.36            ,           90.94              6.90
    her          -7.48      -15.91   −4.96 × 10−5       4.57 × 103        the          14.17              1.12
  settled        -6.12      -15.71   −8.77 × 10−5       1.46 × 104       went          460.16          2.58 × 103
Fixed VAE logSNR = 2.0
    at           11.50      -6.41    −5.91 × 10−5       5.99 × 107      behind         114.91             6.90
    her          13.74      0.21     −8.58 × 10−5       7.53 × 105        it         3.07 × 103           1.12
  settled        10.30      -6.57    −4.03 × 10−3       2.14 × 107      came           442.42          2.58 × 103

Table 4 Token-level comparison across direct training and fixed VAE logSNR settings. Across the same target
tokens, lower likelihood-derived PPL does not necessarily lead to better generation. This token-level evidence further
illustrates the structural gap between likelihood-oriented estimation and generation quality.



   Implication 3. In continuous latent language models, good generation and good likelihood-oriented estima-
   tion are not equivalent. Generation depends on whether the prior reaches semantically valid latent regions,
   whereas likelihood-oriented estimation additionally depends on local density calibration around the gold posterior
   neighborhood.

This distinction is directly supported by Figure 11 and Table 4. In Figure 11, decoder probe success and
posterior hit are consistently high, showing that the decoder can reliably recover the ground-truth token
inside the posterior neighborhood. However, the prior hit rates vary sharply, indicating that the main issue is
not decoder failure but prior misalignment around the gold latent region. Table 4 shows the same pattern
at the token level: for at, the likelihood-derived PPL improves dramatically from 1.15 × 106 to 641.57 and
then 245.36, while the generated token deteriorates from on to in and then to a comma. Similarly, for her,
smaller likelihood-derived PPL under fixed VAE logSNR does not recover the correct token. Thus, lower
likelihood-derived PPL does not necessarily imply better generation.

   Implication 4. In Cola DLM, generation quality is more related to semantic smoothness of the latent space,
   whereas likelihood-oriented PPL is more sensitive to probability-space smoothness shaped by the VAE logSNR.
   Since these two forms of smoothness are different, generation and PPL need not be aligned.

The fixed VAE logSNR settings in Table 4 should therefore be interpreted as changing the smoothness of the
latent probability space rather than merely tuning a hyperparameter. Lower VAE logSNR corresponds to a
flatter local density landscape, which tends to reduce pointwise density mismatch and improve likelihood-
derived PPL. However, such smoothing can also blur local semantic structure and bias the model toward
generic but semantically suboptimal continuations, such as in, the, or went. By contrast, direct training yields
much worse likelihood-derived PPL, but sometimes preserves more appropriate semantic behavior, such as



                                                             23
                                       Different Conditioning and Padding Strategies
                                                           Clean                                                                     Partial



 Non-Padding
                       Prefix Prompt                                                           Prefix Prompt
                                                         Condition                                                                   Repaint

                             ...             noise   noise noise noise                               ... partial noise
                                                                                                          on prompt
                                                                                                                       noise    noise noise noise



                                   1st Gen. Block    2nd Gen. Block                                        1st Gen. Block       2nd Gen. Block

                                                           Right                                                                      Left
                       Prefix Prompt                                                               Prefix Prompt
                                                          Padding                                                                    Padding

 Padding                     ...             pad     noise noise noise            pad                ...                        noise noise noise



                                           Attn Mask 1st Gen. Block             Attn Mask                                       1st Gen. Block


Figure 12 Different conditioning and padding strategies in the first generation block. The first generation
block is a mixed region that contains both known prompt latents and unknown latents to be generated. Clean
condition repaint keeps the known region fixed as a stable condition throughout denoising, partial repaint injects
timestep-matched noisy guidance only during part of the trajectory, and left/right padding instead modify the layout
of the known region without explicit repaint correction.

Table 5 Impact of first-block conditioning strategies. Clean condition repaint performs best, indicating strong,
persistent conditioning is optimal for the first block’s mixed denoising. Conversely, partial repaint is much weaker,
reducing m degrades performance, and increasing t yields no stable gains. Left and right padding outperform partial
repaint but remain inferior to clean conditioning.

                      Partial repaint (t = 1)            Partial repaint (t = 3)
Task                                                                                               Clean cond.        Left pad.       Right pad.
                   m = 1.0   m = 0.7     m = 0.3     m = 1.0       m = 0.7          m = 0.3
Lambada              8.5        8.5        6.6         7.0                7.3            5.6            37.1             24.6              24.7
MMLU                 7.9        7.9        7.8         7.6               6.7             7.0            11.9             8.4               11.5
SIQA                 8.8        8.7        8.2         13.3              13.0           12.0            24.8             14.9              13.8
Avg.                 8.4        8.4        7.5          9.3              9.0            8.2             24.6             16.0              16.7



correctly generating her.

       Summary. In Cola DLM, generation quality primarily reflects the semantic smoothness of the latent space,
       whereas likelihood-oriented PPL is more sensitive to the smoothness of the latent probability space shaped by
       the VAE logSNR. As a result, better generation does not necessarily imply better PPL, and vice versa.


5.2            Impact of Conditioning and Padding Strategies in the First Generation Block
In the first generation block, the input contains both known prompt latents and unknown latents to be
generated. Figure 12 illustrates four representative strategies for handling this mixed region. Partial repaint
injects timestep-matched noisy guidance on the known region, where t controls the number of repaint repetitions
and m controls the fraction of the denoising trajectory that receives such guidance. Clean condition repaint
instead keeps the known region fixed as clean guidance throughout denoising. By contrast, left and right
padding do not explicitly repaint the known region, but only change its positional layout relative to the
generated region. Notably, under the random-length setting, all aforementioned conditioning modes maintain
strict consistency between training and inference.
As shown in Table 5, clean condition repaint consistently achieves the best performance across all tasks. In


                                                                     24
Table 6 Performance under different sample labels and VAE patch sizes. Patch size 2 is overall weaker, but this
gap stems mainly from the Prompt Len Mod1 case (indivisible lengths). On Prompt Len Mod0, patch size 2 becomes
competitive and even outperforms size 1. This suggests the weakness arises from boundary misalignment rather than
latent compression itself.

Sample Label                      Overall                  Prompt Len Mod0                      Prompt Len Mod1
VAE Patch Size             p1                p2          p1                p2                 p1                p2
Lambada                   31.10             17.40       32.11             34.55              30.12             0.79
MMLU                      5.40              3.90         6.89              7.68              3.86              0.00
SIQA                      11.10             6.10        12.92             12.13              9.26              0.00
Avg.                      15.87             9.13        17.31             18.12              14.41             0.26



contrast, partial repaint is substantially weaker, and reducing m generally further degrades performance,
indicating that shortening the guided portion makes the known region harder to preserve. Increasing the
repaint repetitions from t = 1 to t = 3 also does not bring stable gains, suggesting that repeated early
corrections cannot compensate for weak conditioning. Left and right padding are often stronger than most
partial repaint settings because they avoid explicitly re-noising the known region, but still remain clearly below
clean condition repaint. This suggests that positional layout alone is insufficient: padding does not provide a
stable condition throughout denoising, and may further complicate the block-causal attention pattern.
Overall, these results show that the key challenge of the first generation block is to preserve the prompt-
conditioned region while generating the remaining unknown part. For this mixed denoising problem, strong
and persistent conditioning is more effective than partial noisy correction or positional layout alone. More
details are provided in Appendix I.1.

5.3    Compression of the Latent Space
This section discusses whether compressing the text sequence in the VAE is beneficial for Cola DLM. We
train two Text VAEs with the same latent dimensionality (d = 128) but different patch sizes: p1 maps each
token to one latent, while p2 compresses every two tokens into one latent. All other settings follow Section 4.5:
the DiT block size is 16, the training noise schedule uses logit-normal sampling with loc = 1 and scale = 0,
and inference uses 16 denoising steps with CFG = 7.0. In Table 6, Overall reports the full evaluation result,
while Prompt Len Mod0 and Prompt Len Mod1 group samples by whether the prompt length is divisible by 2.

   Implication 5. The weakness of patch size 2 does not mainly come from compression itself, but from the
   boundary case where the prompt length is not divisible by the patch size. Once the latent grouping is well aligned
   with the text sequence, compression can instead become beneficial.

At the overall level, p2 is much worse than p1. However, the parity split shows that this gap is almost entirely
caused by Prompt Len Mod1. For odd-length prompts, p2 nearly collapses on all tasks, whereas on Prompt
Len Mod0, namely the even-length case seen by the patching rule, p2 becomes competitive and even slightly
surpasses p1 on average. This suggests that the current failure is not evidence against latent compression
itself, but against a compression scheme that does not robustly handle non-divisible sequence boundaries.
The reason is likely that, under patch size 2, odd-length prompts necessarily involve padding or incomplete
token groups during compression. If this boundary pattern is not properly learned, the compressed prompt
latent becomes semantically shifted. In Cola DLM, this issue is particularly severe because the prompt latent
is the clean condition for subsequent block-wise prior generation rather than a weak auxiliary representation.
Once the prompt-side latent is biased, the error propagates through denoising and finally harms conditional
decoding, which naturally explains the near-zero performance on Mod1.
By contrast, the Mod0 result is encouraging. It shows that when the latent grouping is semantically valid,
compressing two tokens into one latent does not necessarily hurt generation and may even help it. This is



                                                          25
consistent with the core idea of Cola DLM: the latent space is not intended to preserve a token-aligned recovery
path, but to provide a lower-rate representation for global semantic organization, while the decoder handles
local realization. Under this view, moderate compression can be beneficial because each latent summarizes a
larger textual span and thus better matches the role of the prior.
This also makes latent compression attractive from the efficiency perspective. Under the same DiT block size,
one denoising block corresponds to patch size × block size text tokens after decoding. Therefore, with block
size 16, patch size 1 covers 16 text tokens per block, while patch size 2 covers 32. If the boundary issue can
be resolved, larger patch sizes may improve both semantic abstraction and generation efficiency.

      Summary. Table 6 suggests that latent compression is a promising direction for Cola DLM. Its current
      limitation mainly comes from unstable handling of non-divisible sequence boundaries, while the aligned even-
      length case already shows that compressed latents can support both stronger semantic abstraction and faster
      generation.


5.4     Robustness of VAE Latent Reconstruction
We further analyze the robustness of the VAE
                                                                                                                      Robustness of VAE Latent Reconstruction
latent space from the reconstruction perspective.                                                                                                  Noise ratio t/T
As shown in Figure 13, the VAE achieves nearly                                                         0%               10%               20%            30%           40%              50%             60%

                                                                                                                High-fidelity                       Graceful                      Heavy-noise
perfect reconstruction at t = 0, indicating that                                                               reconstruction                      degradation                      regime




                                                                 Text reconstruction accuracy
                                                                                                1.00                                                                 acc = 0.98

the learned latent–text mapping remains highly                                                                                                                        25% of full noise still
                                                                                                                                                                      yields acc ≈ 0.92
faithful and does not collapse. Moreover, the re-                                               0.80
                                                                                                               acc = 0.9998 @ t = 0
construction accuracy stays very high throughout                                                0.60
                                                                                                               no collapse of latent--text map


the low-noise regime, and still remains around                                                              acc = 0.50

                                                                                                                                                                                  ~50% accuracy still
0.92 at t = 250, before degrading more noticeably                                               0.40                                                                              retained at t ≈ 385


under heavier noise.                                                                            0.20          Measured accuracy
                                                                                                              Sweet spot (t = 250, 25% noise)

These results suggest that the latent space           0.00
                                                                                                              Clean latent (t = 0, acc = 0.9998)



learned by the VAE is not merely a fragile com-
                                                           0      100       200        300          400      500 600
                                                                       Diffusion timestep t (total T = 1000)
pressed code, but a stable and broadly usable in-  Figure 13 Robustness of VAE latent reconstruction. The
termediate representation for text. In particular, VAE preserves near-perfect reconstruction at low noise and
the graceful degradation pattern indicates that    degrades gracefully under stronger perturbations, indicating a
semantic information is not destroyed abruptly     stable latent–text mapping.
by small or moderate perturbations, which fur-
ther supports the view that the VAE latent space in Cola DLM is sufficiently robust to serve as the semantic
interface for subsequent prior modeling.

5.5     Towards a Unified Approach with Image Modalities
A broader implication of Cola DLM is that it provides a natural bridge from discrete text to continuous
multimodal modeling. The key idea of unified modeling is not merely to place text and image into one
backbone, but to map heterogeneous observations into a shared continuous latent interaction space, where
higher-level semantics can be organized under common dynamics.
A natural extension of Cola DLM follows the same probabilistic decomposition as in the text-only setting.
Let xtext and ximg denote the text and image observations, and let their modality-specific latent variables be

                                 z0text ∼ qϕtext (z | xtext ),                                     z0img ∼ qϕimg (z | ximg ).

We then define a joint latent state
                                                     z̃0 = z0text , z0img ,
                                                                         

and model the unified generative process as

                                   p(xtext , ximg , z̃0 ) = pθ xtext , ximg | z̃0 pψ (z̃0 ).
                                                                                 



                                                                 26
  Text to Text                                                                    Text to Image
 Prefix Prompt:
 The summer night was quiet and peaceful. Stars twinkled
 in the deep blue sky.                                                                                                     Text VAE                            Image VAE
 Generation:                                                                                                                Decoder                             Decoder
 The trees bloomed on the shore. The crowd stood in the distance...
                                                                                                                          Text Gen. Latent              Image Gen. Latent
 Prefix Prompt:
 I still remember the happiest days of my childhood. Every
 summer, my friends and                                                                                                                 MMDiT
 Generation:                                                                                                                             (block causal)

 I would go out on the beach and look out at the ocean, enjoying
 our time together towards the left of the garden...
                                                                                                                               Single-stream DiT Block                x2/3

                                                                                                                                Dual-stream DiT Block                 x1/3
  Image x Text to Text
                                                                                                                              Text & Image Condition                Noise




                                                                                                                          Block Segment             Text part is divided into blocks
                                                                                                                                       Image part is treated as a single large block

            (a)                    (b)                       (c)                                                            Text Block Latent          Image Block Latent
                          Prompt: 1. What is shown in this image?
                          2. Describe the content of this image in detail.
                                                                                                                          Text Clean Latent            Image Clean Latent
                         (a) City skyscrapers rising from the water, with many
                             buildings and roads on the sea.                                                               Text VAE                            Image VAE
                         (b) A person using a pen to write on a piece of paper.                                             Encoder                             Encoder
                         (c) Some people sitting around a long table in a
                             conference room.
            (d)          (d) The image shows a newlywed couple at a wedding...




Figure 14 Preliminary qualitative examples of unified text-image modeling. Left: text-only continuation and
image-conditioned text generation. Middle: text-to-image results with only pretraining. Right: a schematic extension
of Cola DLM, where text and image are mapped into modality-specific continuous latents and modeled by a shared
block-causal prior.


Under this view, modality-specific VAE encoders and decoders are responsible for surface-level representation
and realization, while the shared prior models the higher-level semantic structure and cross-modal dependency
in latent space.
This perspective is consistent with the central modeling principle of Cola DLM. In Cola DLM, diffusion is not
used for token-level observation recovery, but for latent prior transport:

                                                             z1 ∼ p1 ,             z0 = Φψ
                                                                                         0←1 (z1 ),   x ∼ pθ (x | z0 ).

In the unified setting, the same idea extends to the multimodal latent state z̃0 : the shared block-causal
MMDiT prior transports and organizes the joint latent semantics, while the modality-specific decoders handle
the final text or image realization. Therefore, continuity is introduced at the level of prior modeling rather
than direct token or pixel recovery.
From the ELBO viewpoint, the benefit of such a decomposition is also conceptually clear. A unified
latent-variable objective takes the form
                                                                                 
            E[LELBO ] = Eq log pθ (xtext , ximg | z̃0 ) − I (X text , X img ); Z̃0 − KL(q̄(z̃0 ) ∥ pψ (z̃0 )) ,
                                                      


which shows the same division of labor as in the text-only case: the latent variable carries compressed global
semantics, while the decoder is responsible for modality-specific realization. In this sense, unified modeling is
not simply parameter sharing across modalities, but a shared semantic prior over heterogeneous observations.
Figure 14 presents a preliminary prototype of this idea. In the current design, the text sequence is divided
into blocks, while the image latent is treated as a single large block. Specifically, the image representation is
obtained via an Image VAE trained on internal multi-resolution data (256 / 384 / 640 / 1024), with a spatial
downsampling factor of 16 and 64 latent channels, providing a compact yet expressive latent space for visual
content. The shared block-causal MMDiT prior operates over both text blocks and image latents, supporting
intra-modal processing as well as cross-modal interaction. Within a unified framework, this enables text-to-text
continuation, image-conditioned text generation, and text-to-image generation. We jointly optimize these


                                                                                             27
tasks on internal image–text pairs during training. For the text-to-image task, we first train on 256-resolution
data for 80k steps with a global batch size of approximately 3k, and then continue training on 640-resolution
data for 10k steps with a global batch size of approximately 1k. For image-conditioned text generation, we
adopt the same batch size configuration and train for approximately 50k steps. More result samples are
provided at I.2.
These results should be interpreted primarily as qualitative evidence of feasibility. As the current prototype
remains at an early stage of training, and our experiments are limited to moderate pretraining on in-house 256
and 640 resolution data, without extensive high-quality data curation or supervised fine-tuning. The goal of
this section is not to present a mature multimodal system. Rather, it is to demonstrate that the hierarchical
latent-prior formulation of Cola DLM naturally extends beyond text-only generation. In future work, we
plan to conduct more comprehensive unified multimodal training. More broadly, these findings suggest that
decoupling global latent organization from modality-specific realization may offer a structurally clean and
scalable path toward more native unified generative models.

    Summary. These preliminary results suggest that Cola DLM naturally extends to unified text–image modeling.
    A shared block-causal prior organizes global and cross-modal semantics, while modality-specific decoders handle
    final realization. Although still early-stage and qualitative, this prototype already shows a promising bridge
    from language generation to native multimodal generative modeling.



6   Limitations & Future Prospects
Although this paper has provided initial evidence for the feasibility, competitiveness, and promising scaling
potential of Cola DLM for text generation in continuous latent space, we view it as a starting point for further
exploration rather than a finished endpoint. First, at the scale and evaluation level, the current results reveal
encouraging trends, but the experiments are still conducted at a relatively controlled scale and mainly serve
to clarify the key properties of the framework. It is therefore natural and important to further examine its
upper bound under larger model sizes, longer training, and more substantial compute budgets. Second, at the
model-design level, our analyses show that the training strategy of the Text VAE, the text compression scheme,
the choice of latent dimensionality, the semantic smoothness of the latent space, and the joint calibration
of VAE logSNR, DiT block size, and noise schedule all affect the semantic organization of the latent space
and the final generation quality. In particular, the experiments suggest that stronger latent representations
usually require better-aligned noise calibration, indicating substantial room for further optimization. Finally,
at the framework level, the main value of Cola DLM lies not merely in the denoising process itself, but in
its decomposition of text generation into global semantic prior modeling and local textual realization. This
opens the door to exploring stronger latent modules, such as AE [5] and RAE [112], as well as more flexible
prior-learning approaches, such as drifting-model-based [19] distribution matching for continuous priors. More
broadly, following the idea of unified continuous latent-space modeling, the framework may also be extended
to continuous modalities such as images, further advancing unified generation.

7   Conclusion
In conclusion, this paper presents Cola DLM, a hierarchical continuous latent diffusion language model that
decomposes text generation into global semantic prior modeling in latent space and local textual realization
through conditional decoding, thereby providing a principled alternative to strictly token-level language
modeling. Across the full study, both the theoretical analysis and the experiments consistently suggest
that text generation can benefit from hierarchical information decomposition: we find evidence of shared
global semantic structure in latent space, identify effective design choices for latent-space formation and
diffusion modeling, and show that under strictly matched comparisons, Cola DLM exhibits strong generation
quality and encouraging scaling behavior. More broadly, our results indicate that for this class of models,
generation-oriented evaluation and scaling trends may be more informative than likelihood alone, while the
continuous latent formulation also offers a concrete path toward more native unified modeling across discrete
text and continuous modalities.


                                                          28
8     Afterword: Research Objectives and Significance
Viewed from a broader perspective, this study is not only concerned with proposing an alternative architecture
for text generation, but also with clarifying a more general picture of learning in which representation,
objective, and environment must be understood jointly. From this perspective, the three themes of this
work are closely connected rather than independent. The first concerns how text should be represented and
generated. The second concerns what kinds of objectives and evaluation criteria are genuinely aligned with
such representations. The third concerns the kind of environment in which a model should ultimately learn if
the goal is more general multimodal intelligence.
A useful starting point is to formalize learning itself as a model–environment interaction system. Let the
environment be
                                           E = (Ω, O, A, T , F, G),                                   (8.1)
where Ω is the environment state space, O is the observation space, A is the action or output space, T is the
state transition mechanism, F is the feedback generation mechanism, and G is the rule that converts feedback
into optimization signals. Importantly, the notion of environment is understood here in a broad sense: it
includes not only the external world, but also the data distribution presented to the model, task formats,
supervision protocols, and even the loss rules by which feedback is transformed into gradients.
Let the model be denoted by Mθ , with internal state space H, state update map Uθ , and policy or generation
map Πθ . At interaction step t, the closed-loop system can be written as

                                              ot ∼ PE (· | ωt ),                                          (8.2)
                                              ht = Uθ (ht−1 , ot ),                                       (8.3)
                                              at ∼ Πθ (· | ht ),                                          (8.4)
                                              ξt ∼ F(· | ωt , ot , at ),                                  (8.5)
                                          ωt+1 ∼ T (· | ωt , ot , at , ξt ),                              (8.6)
                                              ℓt = G(ωt , ot , at , ξt ).                                 (8.7)

The overall learning objective is therefore
                                                                 " T            #
                                                                  X
                                     J (θ; E) = Eτ ∼P (τ |θ,E)           γ t−1 ℓt ,                       (8.8)
                                                                   t=1

where τ denotes a complete interaction trajectory and γ is the discount factor.
This formalization shows directly that learning is never an isolated question of model structure alone. Rather,
it is jointly determined by three factors: first, the state space in which the model absorbs and organizes
information; second, the kind of feedback through which the environment defines improvement; and third,
the actual structure that generates observations, transitions, and feedback. In this work, these three aspects
correspond precisely to the three recurring themes of the paper: how text should be represented, which metrics
are aligned with the true learning objective, and what kind of environment unified models are ultimately
meant to enter.

8.1   Rethinking Text Modeling Paradigms: From State Space in the System to Hierarchi-
      cal Text Generation
From a system-level perspective, the central question of text modeling is not merely which generation order
to adopt, but rather in what kind of state text should be represented within the learning system. Mainstream
autoregressive language models bind the state tightly to the surface token prefix, and generation is therefore
written as
                                                    Yn
                                          pAR (x) =     pθ (xt | x<t ).                                  (8.9)
                                                       t=1




                                                         29
This factorization is highly effective, but it fundamentally corresponds to a strong modeling assumption: both
global semantics and local realization are propagated through the same token-level conditional chain. In other
words, it assumes that the surface string itself is the most natural and primary state space.
The route explored in this paper instead reconsiders text generation from the level of the state space itself. If
text indeed contains a low-dimensional yet sufficiently useful global semantic structure, then a more natural
approach is not to place the entire burden of generation on a token-level chain factorization, but to introduce
latent variables explicitly and model high-level semantic organization separately from local textual realization.
Correspondingly, the core factorization of Cola DLM is
                                                                  Z
                          p(x, z0 ) = pθ (x | z0 )pψ (z0 ), p(x) = pθ (x | z0 )pψ (z0 ) dz0 ,              (8.10)

where z0 is a continuous latent variable, pψ (z0 ) is the latent prior, and pθ (x | z0 ) is the conditional decoder.
The crucial change here is not merely the introduction of latent variables, but the redefinition of the role of
state in the system: the path no longer acts directly on observation recovery, but instead organizes global
semantics in latent space first, after which the decoder carries out local textual realization.
This point can be stated compactly through the information decomposition of the average ELBO. Let

                                             q(x, z0 ) := pdata (x)qϕ (z0 | x),                                  (8.11)

then
                                                                                                                 (8.12)
                                                                                                            
               Epdata (x) [LELBO (x)] = Eq(x,z0 ) [log pθ (x | z0 )] − Iq (X; Z0 ) − KL q̄ϕ (z0 ) ∥ pψ (z0 ) ,
where q̄ϕ (z0 ) is the aggregated posterior. This decomposition shows that hierarchical latent-space modeling
breaks the text problem into three coupled but analytically distinguishable components: conditional realization,
information compression, and prior matching. The latent variable is therefore not merely a continuous surrogate
for discrete tokens, but an explicit intermediate state through which global semantic organization can be
separated from local textual realization and modeled on its own terms.
From this perspective, compression must also be reconsidered. Prior work has emphasized the connection
between compression and intelligence [37], while recent explorations of generation closer to raw data forms in
images and videos, such as pixel-space modeling [19], further suggest that compression should not be equated
with harmful information deletion. The key question is not whether every local detail is preserved, but whether
the model can extract and organize structural information that is genuinely effective and generalizable. If text
indeed admits a hierarchical structure in which high-level semantics and low-level realization are relatively
separable, then reinterpreting text generation through informational hierarchy is not merely a change of
method, but a theoretical re-evaluation of text modeling itself.
Accordingly, the first theme of this paper is not to reject autoregression, but to point out that autoregression
occupies only one self-consistent, rather than unique, corner of the design space. If the data truly contains a
hierarchy between low-dimensional global semantics and high-dimensional local realization, then organizing
semantics first in a latent state and realizing text through conditional decoding may be closer to the true
generative mechanism. Text generation should therefore not be understood solely as next-token fitting over
discrete strings, but more generally as a systematic problem of how information is represented, compressed,
and organized hierarchically.

8.2    Understanding the Continuous Extension of Discrete Text: From Objective Mismatch
       to a Shift in Evaluation Emphasis
Once the state space of the system is changed, the issue at the objective level changes accordingly. For
conventional autoregressive language models, the training objective and evaluation quantities are naturally
well aligned: maximum likelihood training directly corresponds to probability fitting over text, and likelihood
and perplexity therefore have a clear and stable interpretation. In hierarchical continuous latent-space models,
however, the actual training path is no longer direct token-level maximum likelihood, but a hierarchical
objective jointly composed of reconstruction, latent prior learning, and representation regularization.


                                                             30
This can be seen from the relation between the ELBO and the true marginal likelihood:

                                                                                                            (8.13)
                                                                                    
                        −LELBO (x) = − log pθ,ψ (x) + KL qϕ (z0 | x) ∥ pθ,ψ (z0 | x) .

This shows that even at the level of the ELBO, the training objective is already separated from the true
log-likelihood by a variational inference gap. Furthermore, in the actual training of Cola DLM, the model must
jointly learn latent reconstruction, continuous prior fitting, and representation stabilization. The quantity
being optimized is therefore not a single token-level likelihood in the classical sense.
For this reason, the mismatch should not be interpreted as a failure to learn, but rather as evidence that
the model is learning something different. For autoregressive models and other paradigms that directly fit
discrete distributions, likelihood and perplexity remain highly informative because they are naturally aligned
with the training objective. For hierarchical continuous latent-space models, by contrast, the central issue
is no longer whether local discrete distributions are fitted as sharply as possible, but whether higher-level
semantic structures are effectively organized, whether the latent prior is well learned, and whether the final
generations satisfy the actual task requirements.
From the perspective of systematic modeling, this phenomenon is in fact expected: when the state space
expands from surface tokens to hierarchical latent variables, the optimization target correspondingly shifts
from precise fitting of local discrete distributions to the organization of higher-level semantic structure, stable
latent prior learning, and satisfaction of the true generative objective. For this route, generation-oriented
metrics are therefore often more closely aligned with what the model is actually trained to do than perplexity.
More importantly, model potential is often reflected more clearly in scaling behavior than in any single static
likelihood value: what matters is whether capability continues to improve steadily as model size, data, and
compute increase, rather than whether local fit under a particular pointwise metric is better.
This can also be connected to the perspective of the three governing curves developed in the theoretical analysis
of the paper. For Cola DLM, the applicability of this route is not determined by a single likelihood value, but
by whether three conditions hold simultaneously: the representation rate–distortion curve is already favorable
at relatively low rate, the approximation error of the latent prior continues to decrease, and the inference gap
remains controllable. In other words, the advantage of this route is not guaranteed automatically by latent
variables or flow-based modeling themselves; it depends on whether the data truly contains a compressible
global semantic structure, and whether the model can learn, fit, and realize that structure in a stable manner.
The second theme of this paper is therefore not merely that perplexity is inadequate, but that evaluation
language itself must change once representation and objective have changed. For this class of models,
generation quality and scaling behavior are often closer to the model’s true capability and long-term potential
than traditional perplexity.

8.3   Exploring Unified Models: Model-Environment Interaction and the Value of Multi-
      modal Unification
If we return again to the model–environment formalization in Eq. (8.1) and Eq. (8.8), the third theme becomes
more natural. The importance of unified models does not lie merely in placing multiple modalities within a
single parameterized network, but in changing the structure of the environment in which the model learns.
In the real world, observations, transitions, and feedback are usually not generated independently across
modalities; rather, they are often jointly determined by a shared latent state. A more general learning system
therefore requires not a set of isolated modality interfaces placed side by side, but unified representations that
can enter the same interaction state and share the same dynamical constraints.
This is closely related to two broader views of intelligence. One influential view understands intelligence
as a collection of skills across tasks [16]. Under this view, a system becomes more capable because it can
solve problems across more domains and under more diverse forms of supervision and interaction. The
recent development of large language models partly reflects this tendency. A representative example is the
progress of code agents that can operate within command-line environments. In such environments, the
observation space, action space, and feedback mechanism are unusually well aligned with discrete symbolic



                                                        31
representations. Interaction trajectories are easy to record, and correctness is often straightforward to verify,
so these environments provide dense and precise learning signals.
Another view, closer to the world-model perspective, holds that intelligence consists in acquiring an internal
model of the structure and dynamics of the world. Recent work on world models [93] moves in this direction
by seeking to learn richer environmental dynamics, thereby supporting stronger generalization and more
realistic interaction. From this perspective, the question is not only how many tasks a model can solve, but
whether it learns in an environment whose structure is rich enough to induce the right abstractions. The
environment therefore becomes central: a model can only internalize the regularities that are actually present
in the observations, transitions, and feedback it encounters.
This can also be written more formally. Let the observation at step t be multimodal,
                                       (1) (2)          (M )     (m)
                                ot = ot , ot , . . . , ot      , ot ∈ O(m) ,                                (8.14)

and suppose there exists a joint latent state
                                                                        
                                                       (1)          (M )
                                               zt = Φ ot , . . . , ot      ,                                (8.15)

such that feedback and transition depend primarily on this joint state rather than on marginal factorizations
over modalities:
                                                                                                       (8.16)
                                                                        
                                       ξt , ωt+1 ∼ p ξt , ωt+1 | zt , at .
If the true environmental dynamics satisfy
                                                          M
                                                                      (m)     (m)     (m)      (m) 
                                                          Y
                             p(ξt , ωt+1 | ot , at ) ̸=         pm ξt       , ωt+1 | ot     , at       ,    (8.17)
                                                          m=1

then the learning problem is structurally non-separable across modalities. In such a case, treating each
modality as an independent channel and only combining them superficially is generally insufficient. The
theoretical significance of unified models lies precisely in the fact that the environment itself is non-separable
in the sense of Eq. (8.17): the regularities that determine useful feedback are joint regularities rather than
regularities defined on the marginal distribution of each modality.
This clarifies why multimodal unification is not merely an engineering convenience. Its purpose is not simply
to process multiple data types with one backbone, but to allow the model to learn in an environment whose
observation, transition, and supervision structure more faithfully reflects the coupled regularities of the real
world. In such an environment, both inputs and outputs may be multimodal; useful feedback may depend
on how different modalities constrain each other jointly; and the learned internal state should ideally reflect
these joint constraints.
This also explains why text has long been the most difficult component in unified models. Images and videos
naturally operate in continuous spaces, whereas text is a prototypically discrete modality. If they are to enter
a common interaction state and share latent dynamics, a severe representational mismatch immediately arises.
This is precisely one of the central obstacles repeatedly identified in recent unified-model research [18]. In this
sense, the significance of Cola DLM lies not only in proposing another text generator, but in providing a
natural interface through which discrete text can enter a continuous latent space.
If discrete text is mapped into a continuous latent variable through

                                 z text ∼ qϕ (z | xtext ),           xtext ∼ pη (x | z text ),              (8.18)

then text acquires an interface compatible with other continuous modalities. One may then define a unified
interaction state                                                           
                                       z̃t = Ψ zttext , ztimg , ztvid , . . . ,                     (8.19)

and perform state evolution, decision making, and feedback modeling at this level. Equations (8.18)–(8.19)
formalize why Cola DLM may matter beyond text generation itself: its role is not only to generate text through


                                                                32
a different path, but to provide a bridge through which an intrinsically discrete modality can participate in a
continuous multimodal interaction state. In other words, it reduces the structural mismatch that otherwise
prevents text from naturally entering a shared continuous environment.
This is why the broader significance of Cola DLM is better understood through model–environment interaction
than through single-modality benchmarks alone. If learning is viewed as the optimization of Eq. (8.8) in
richer and more realistic environments, then unified models matter because they expand the environments in
which the model can learn. If text is to participate fully in such environments, then a bridge such as that in
Eq. (8.18) becomes especially desirable. In this sense, Cola DLM is not merely an alternative text generator;
it can also be understood as a candidate mechanism for aligning discrete text with continuous multimodal
learning systems.

8.4   The Three Themes Under a Unified Perspective
In summary, the three themes of this paper are not separate supplementary discussions, but three manifestations
of the same systematic problem. The first concerns the representation level: whether text should be modeled
entirely on the token surface, or whether higher-level semantics can be organized in an independent latent state.
The second concerns the objective level: once the model is trained through latent transport, reconstruction, and
regularization rather than direct token-level maximum likelihood, which metrics remain genuinely aligned with
the learning problem. The third concerns the environment level: if learning is ultimately model–environment
interaction, then what kind of environment future models should inhabit, and what representational interfaces
are needed for different modalities to become compatible within it.
From this perspective, autoregressive language modeling occupies a self-consistent corner of the design space:
representation is tightly bound to surface tokens, the training objective is direct likelihood maximization,
and the environment is largely symbolic and text-centered. The route explored in this work changes all
three assumptions simultaneously. It introduces a hierarchical latent-variable representation for text, thereby
changing the representational assumption; it moves optimization away from direct token-level likelihood,
thereby weakening the central interpretive role of perplexity; and it provides a continuous interface for discrete
text, thereby making text potentially more compatible with multimodal environments that are more naturally
expressed in continuous latent space.
We therefore hope that the contribution of this work is not only a viable alternative path for text generation,
but also a more systematic way of thinking that jointly considers representation, objective alignment, and
environment design. More broadly, we hope it encourages future research to rethink text, images, videos,
and other modalities not as isolated domains that must be solved separately, but as components of a larger
learning system in which unified representation, unified objectives, and unified environments may become
increasingly central to the development of more general multimodal intelligence.




                                                       33
References
 [1] PS Aishwarya, Pranav Ajit Nair, Yashas Samaga BL, Toby James Boyd, Sanjiv Kumar, Prateek Jain, and
     Praneeth Netrapalli. Tandem transformers for inference efficient llms. In Forty-first International Conference on
     Machine Learning, 2024.
 [2] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising
     diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993,
     2021.
 [3] Gregor Bachmann and Vaishnavh Nagarajan.             The pitfalls of next-token prediction.       arXiv preprint
     arXiv:2403.06963, 2024.
 [4] Hmrishav Bandyopadhyay, Nikhil Pinnaparaju, Rahim Entezari, Jim Scott, Yi-Zhe Song, and Varun Jampani.
     Block cascading: Training free acceleration of block-causal video models. arXiv preprint arXiv:2511.20426, 2025.
 [5] Dor Bank, Noam Koenigstein, and Raja Giryes. Autoencoders. Machine learning for data science handbook:
     data mining and knowledge discovery handbook, pages 353–374, 2023.
 [6] Loïc Barrault, Paul-Ambroise Duquenne, Maha Elbayad, Artyom Kozhevnikov, Belen Alastruey, Pierre Andrews,
     Mariano Coria, Guillaume Couairon, Marta R Costa-jussà, David Dale, et al. Large concept models: Language
     modeling in a sentence representation space. arXiv preprint arXiv:2412.08821, 2024.
 [7] Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain
     Evans. The reversal curse: Llms trained on" a is b" fail to learn" b is a". arXiv preprint arXiv:2309.12288, 2023.
 [8] Samuel Bowman, Luke Vilnis, Oriol Vinyals, Andrew Dai, Rafal Jozefowicz, and Samy Bengio. Generating
     sentences from a continuous space. In Proceedings of the 20th SIGNLL conference on computational natural
     language learning, pages 10–21, 2016.
 [9] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
     Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners.
     Advances in neural information processing systems, 33:1877–1901, 2020.
[10] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud
     Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing
     Systems, 35:28266–28279, 2022.
[11] Ricardo Cannizzaro, Jonathan Routley, and Lars Kunze. Towards a causal probabilistic framework for prediction,
     action-selection & explanations for robot block-stacking tasks. arXiv preprint arXiv:2308.06203, 2023.
[12] Keshigeyan Chandrasegaran, Michael Poli, Daniel Y Fu, Dongjun Kim, Lea M Hadzic, Manling Li, Agrim Gupta,
     Stefano Massaroli, Azalia Mirhoseini, Juan Carlos Niebles, et al. Exploring diffusion transformer designs via
     grafting. arXiv preprint arXiv:2506.05340, 2025.
[13] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang
     Wang, Yidong Wang, et al. A survey on evaluation of large language models. ACM transactions on intelligent
     systems and technology, 15(3):1–45, 2024.
[14] Jiaao Chen, Aston Zhang, Mu Li, Alex Smola, and Diyi Yang. A cheaper and better diffusion language model
     with soft-masked noise. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language
     Processing, pages 4765–4775, 2023.
[15] Tianqi Chen, Shujian Zhang, and Mingyuan Zhou. Dlm-one: Diffusion language models for one-step sequence
     generation. arXiv preprint arXiv:2506.00290, 2025.
[16] François Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019.
[17] Murtaza Dalal, Alexander C Li, and Rohan Taori. Autoregressive models: What are they good for? arXiv
     preprint arXiv:1910.07737, 2019.
[18] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan
     Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683,
     2025.
[19] Mingyang Deng, He Li, Tianhong Li, Yilun Du, and Kaiming He. Generative modeling via drifting. arXiv
     preprint arXiv:2602.04770, 2026.


                                                         34
[20] Justin Deschenaux and Caglar Gulcehre. Promises, outlooks and challenges of diffusion language modeling.
     arXiv preprint arXiv:2406.11473, 2024.
[21] Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H Richemond,
     Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, et al. Continuous diffusion for categorical data.
     arXiv preprint arXiv:2211.15089, 2022.
[22] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General
     language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of
     the Association for Computational Linguistics (Volume 1: Long Papers), pages 320–335, 2022.
[23] Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the sequential dependency of llm inference using
     lookahead decoding. arXiv preprint arXiv:2402.02057, 2024.
[24] Zhujin Gao, Junliang Guo, Xu Tan, Yongxin Zhu, Fang Zhang, Jiang Bian, and Linli Xu. Empowering diffusion
     models on the embedding space for text generation. In Proceedings of the 2024 Conference of the North
     American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume
     1: Long Papers), pages 4664–4683, 2024.
[25] Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky TQ Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman.
     Discrete flow matching. Advances in Neural Information Processing Systems, 37:133345–133385, 2024.
[26] Raghavv Goel, Risheek Garrepalli, Sudhanshu Agrawal, Chris Lott, Mingu Lee, and Fatih Porikli. Skip to the
     good part: Representation structure & inference-time layer skipping in diffusion vs. autoregressive llms. arXiv
     preprint arXiv:2603.07475, 2026.
[27] Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and LingPeng Kong. Diffuseq: Sequence to sequence text
     generation with diffusion models. arXiv preprint arXiv:2210.08933, 2022.
[28] Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao,
     Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. arXiv
     preprint arXiv:2410.17891, 2024.
[29] Ishaan Gulrajani and Tatsunori B Hashimoto. Likelihood-based diffusion language models. Advances in Neural
     Information Processing Systems, 36:16693–16715, 2023.
[30] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang,
     Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.
     arXiv preprint arXiv:2501.12948, 2025.
[31] Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplex-based diffusion
     language model for text generation and modular control. In Proceedings of the 61st Annual Meeting of the
     Association for Computational Linguistics (Volume 1: Long Papers), pages 11575–11596, 2023.
[32] Tatsunori B Hashimoto, Hugh Zhang, and Percy Liang. Unifying human and statistical evaluation for natural
     language generation. In Proceedings of the 2019 Conference of the North American Chapter of the Association
     for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages
     1689–1701, 2019.
[33] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
     Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.
[34] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration.
     arXiv preprint arXiv:1904.09751, 2019.
[35] Emiel Hoogeboom, Alexey A Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans.
     Autoregressive diffusion models. arXiv preprint arXiv:2110.02037, 2021.
[36] Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial
     diffusion: Learning categorical distributions. Advances in neural information processing systems, 34:12454–12465,
     2021.
[37] Yuzhen Huang, Jinghan Zhang, Zifei Shan, and Junxian He. Compression represents intelligence linearly. arXiv
     preprint arXiv:2404.09937, 2024.




                                                        35
[38] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila
     Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
[39] DeLesley Hutchins, Imanol Schlag, Yuhuai Wu, Ethan Dyer, and Behnam Neyshabur. Block-recurrent trans-
     formers. Advances in neural information processing systems, 35:33248–33261, 2022.
[40] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint
     arXiv:1611.01144, 2016.
[41] Daniel Jarrett, Miruna Pislar, Michiel A Bakker, Michael Henry Tessler, Raphael Köster, Jan Balaguer, Romuald
     Elie, Christopher Summerfield, and Andrea Tacchetti. Language agents as digital representatives in collective
     decision-making. arXiv preprint arXiv:2502.09369, 2025.
[42] Sullam Jeoung, Yubin Ge, Haohan Wang, and Jana Diesner. Examining alignment of large language models
     through representative heuristics: the case of political stereotypes. arXiv preprint arXiv:2501.14294, 2025.
[43] Jaehyeong Jo and Sung Ju Hwang.         Continuous diffusion model for language modeling.        arXiv preprint
     arXiv:2502.11564, 2025.
[44] Haoqiang Kang, Yizhe Zhang, Nikki Lijing Kuang, Nicklas Majamaki, Navdeep Jaitly, Yi-An Ma, and Lianhui
     Qin. Ladir: Latent diffusion enhances llms for text reasoning. arXiv preprint arXiv:2510.04573, 2025.
[45] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast
     autoregressive transformers with linear attention. In International conference on machine learning, pages
     5156–5165. PMLR, 2020.
[46] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.
[47] Preethi Lahoti, Nicholas Blumm, Xiao Ma, Raghavendra Kotikalapudi, Sahitya Potluri, Qijun Tan, Hansa
     Srinivasan, Ben Packer, Ahmad Beirami, Alex Beutel, et al. Improving diversity of demographic representation
     in large language models via collective-critiques and self-voting. In Proceedings of the 2023 Conference on
     Empirical Methods in Natural Language Processing, pages 10383–10405, 2023.
[48] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension
     dataset from examinations. In Proceedings of the 2017 conference on empirical methods in natural language
     processing, pages 785–794, 2017.
[49] Bocheng Li, Zhujin Gao, and Linli Xu. Unifying continuous and discrete text diffusion with non-simultaneous
     diffusion processes. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
     (Volume 1: Long Papers), pages 11530–11551, 2025.
[50] Chengze Li, Yitong Zhang, Jia Li, Liyi Cai, and Ge Li. Beyond autoregression: An empirical study of diffusion
     large language models for code generation. arXiv preprint arXiv:2509.11252, 2025.
[51] Chunyuan Li, Xiang Gao, Yuan Li, Baolin Peng, Xiujun Li, Yizhe Zhang, and Jianfeng Gao. Optimus: Organizing
     sentences via pre-trained modeling of a latent space. In Proceedings of the 2020 Conference on Empirical Methods
     in Natural Language Processing (EMNLP), pages 4678–4699, 2020.
[52] Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusion-lm improves
     controllable text generation. Advances in neural information processing systems, 35:4328–4343, 2022.
[53] Chu-Cheng Lin, Aaron Jaech, Xin Li, Matthew R Gormley, and Jason Eisner. Limitations of autoregressive
     models and their alternatives. In Proceedings of the 2021 conference of the North American chapter of the
     association for computational linguistics: Human language technologies, pages 5147–5173, 2021.
[54] Zhenghao Lin, Yeyun Gong, Yelong Shen, Tong Wu, Zhihao Fan, Chen Lin, Nan Duan, and Weizhu Chen. Text
     generation with diffusion language models: A pre-training approach with continuous paragraph denoise. In
     International Conference on Machine Learning, pages 21051–21064. PMLR, 2023.
[55] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai
     Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv
     preprint arXiv:2405.04434, 2024.
[56] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng,
     Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.




                                                        36
[57] Yuxuan Liu, Jingmin Sun, and Hayden Schaeffer. Bcat: A block causal transformer for pde foundation models
     for fluid dynamics. arXiv preprint arXiv:2501.18972, 2025.
[58] Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, and Kilian Q Weinberger. Latent diffusion for
     language generation. Advances in Neural Information Processing Systems, 36:56998–57025, 2023.
[59] Rabeeh Karimi Mahabadi, Hamish Ivison, Jaesung Tae, James Henderson, Iz Beltagy, Matthew E Peters, and
     Arman Cohan. Tess: Text-to-text self-conditioned simplex diffusion. In Proceedings of the 18th Conference
     of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages
     2347–2361, 2024.
[60] Eran Malach. Auto-regressive next-token predictors are universal learners. arXiv preprint arXiv:2309.06979,
     2023.
[61] Clara Meister and Ryan Cotterell. Language model evaluation beyond perplexity. In Proceedings of the 59th
     annual meeting of the association for computational linguistics and the 11th international joint conference on
     natural language processing (volume 1: long papers), pages 5328–5339, 2021.
[62] Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score
     matching for discrete data. Advances in Neural Information Processing Systems, 35:34532–34545, 2022.
[63] Viacheslav Meshchaninov, Egor Chimbulatov, Alexander Shabalin, Aleksandr Abramov, and Dmitry Vetrov.
     Cosmos: Compressed and smooth latent space for text diffusion modeling. arXiv preprint arXiv:2506.21170,
     2025.
[64] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a
     new dataset for open book question answering. In Proceedings of the 2018 conference on empirical methods in
     natural language processing, pages 2381–2391, 2018.
[65] Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and
     Jianfeng Gao. Large language models: A survey. arXiv preprint arXiv:2402.06196, 2024.
[66] Shentong Mo, Enze Xie, Ruihang Chu, Lanqing Hong, Matthias Niessner, and Zhenguo Li. Dit-3d: Exploring
     plain diffusion transformers for 3d shape generation. Advances in neural information processing systems, 36:
     67960–67971, 2023.
[67] Giovanni Monea, Armand Joulin, and Edouard Grave. Pass: Parallel speculative sampling. arXiv preprint
     arXiv:2311.13581, 2023.
[68] Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende,
     Pushmeet Kohli, and James Allen. A corpus and cloze evaluation for deeper understanding of commonsense stories.
     In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational
     Linguistics: Human Language Technologies, pages 839–849, 2016.
[69] Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan Li. Scaling
     up masked diffusion models on text. arXiv preprint arXiv:2410.18514, 2024.
[70] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen,
     and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.
[71] Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu,
     Shengyi Huang, Matt Jordan, et al. 2 olmo 2 furious. arXiv preprint arXiv:2501.00656, 2024.
[72] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing
     discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736,
     2024.
[73] James Pan and Guoliang Li. A survey of llm inference systems. arXiv preprint arXiv:2506.21901, 2025.
[74] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle,
     Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad
     discourse context. In Proceedings of the 54th annual meeting of the association for computational linguistics
     (volume 1: Long papers), pages 1525–1534, 2016.




                                                        37
[75] Byeongjun Park, Hyojun Go, Jin-Young Kim, Sangmin Woo, Seokil Ham, and Changick Kim. Switch diffusion
     transformer: Synergizing denoising tasks with sparse mixture-of-experts. In European Conference on Computer
     Vision, pages 461–477. Springer, 2024.
[76] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF
     international conference on computer vision, pages 4195–4205, 2023.
[77] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are
     unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
[78] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for ma-
     chine comprehension of text. In Proceedings of the 2016 conference on empirical methods in natural language
     processing, pages 2383–2392, 2016.
[79] Pierre H Richemond, Sander Dieleman, and Arnaud Doucet. Categorical sdes with simplex diffusion. arXiv
     preprint arXiv:2210.14784, 2022.
[80] Subham S Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander
     Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural
     Information Processing Systems, 37:130136–130184, 2024.
[81] Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, and Volodymyr
     Kuleshov. The diffusion duality. arXiv preprint arXiv:2506.10892, 2025.
[82] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social iqa: Commonsense reasoning
     about social interactions. In Proceedings of the 2019 conference on empirical methods in natural language
     processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages
     4463–4473, 2019.
[83] Stanislau Semeniuta, Aliaksei Severyn, and Erhardt Barth. A hybrid convolutional variational autoencoder
     for text generation. In Proceedings of the 2017 conference on empirical methods in natural language processing,
     pages 627–637, 2017.
[84] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked
     diffusion for discrete data. Advances in neural information processing systems, 37:103131–103167, 2024.
[85] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole.
     Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456,
     2020.
[86] Aadithya Srikanth, Mudit Gaur, and Vaneet Aggarwal. Discrete state diffusion models: A sample complexity
     perspective. arXiv preprint arXiv:2510.10854, 2025.
[87] Robin Strudel, Corentin Tallec, Florent Altché, Yilun Du, Yaroslav Ganin, Arthur Mensch, Will Grathwohl,
     Nikolay Savinov, Sander Dieleman, Laurent Sifre, et al. Self-conditioned embedding diffusion for text generation.
     arXiv preprint arXiv:2211.04236, 2022.
[88] Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuous-time discrete diffusion
     models. arXiv preprint arXiv:2211.16750, 2022.
[89] Jaesung Tae, Hamish Ivison, Sachin Kumar, and Arman Cohan. Tess 2: A large-scale generalist diffusion
     language model. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
     (Volume 1: Long Papers), pages 21171–21188, 2025.
[90] Yuhta Takida, Takashi Shibuya, WeiHsiang Liao, Chieh-Hsin Lai, Junki Ohmura, Toshimitsu Uesaka, Naoki
     Murata, Shusuke Takahashi, Toshiyuki Kumakura, and Yuki Mitsufuji. Sq-vae: Variational bayes on discrete
     representation with self-annealed stochastic quantization. arXiv preprint arXiv:2205.07547, 2022.
[91] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo
     Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.
[92] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix,
     Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language
     models. arXiv preprint arXiv:2302.13971, 2023.




                                                        38
 [93] Yuanpeng Tu, Hao Luo, Xi Chen, Xiang Bai, Fan Wang, and Hengshuang Zhao. Playerone: Egocentric world
      simulator. arXiv preprint arXiv:2506.09995, 2025.
 [94] Siddarth Venkatraman, Moksh Jain, Luca Scimeca, Minsu Kim, Marcin Sendera, Mohsin Hasan, Luke Rowe,
      Sarthak Mittal, Pablo Lemos, Emmanuel Bengio, et al. Amortizing intractable inference in diffusion models for
      vision, language, and control. Advances in neural information processing systems, 37:76080–76114, 2024.
 [95] Dimitri Von Rütte, Janis Fluri, Yuhui Ding, Antonio Orvieto, Bernhard Schölkopf, and Thomas Hofmann.
      Generalized interpolating discrete diffusion. arXiv preprint arXiv:2503.04482, 2025.
 [96] Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan, Yi Zhu,
      Quanlu Zhang, et al. Efficient large language models: A survey. arXiv preprint arXiv:2312.03863, 2023.
 [97] Shufan Wang, Yixiao Song, Andrew Drozdov, Aparna Garimella, Varun Manjunatha, and Mohit Iyyer. knn-lm
      does not improve open-ended text generation. In Proceedings of the 2023 Conference on Empirical Methods in
      Natural Language Processing, pages 15023–15037, 2023.
 [98] Shiguang Wu, Zhaochun Ren, Xin Xin, Jiyuan Yang, Mengqi Zhang, Zhumin Chen, Maarten de Rijke, and
      Pengjie Ren. Constrained auto-regressive decoding constrains generative retrieval. In Proceedings of the 48th
      International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2429–2440,
      2025.
 [99] Minkai Xu, Tomas Geffner, Karsten Kreis, Weili Nie, Yilun Xu, Jure Leskovec, Stefano Ermon, and Arash
      Vahdat. Energy-based diffusion language models for text generation. arXiv preprint arXiv:2410.21357, 2024.
[100] Peng Xu, Jackie Chi Kit Cheung, and Yanshuai Cao. On variational learning of controllable representations for
      text without supervision. In International Conference on Machine Learning, pages 10534–10543. PMLR, 2020.
[101] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen
      Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
[102] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. Xlnet:
      Generalized autoregressive pretraining for language understanding. Advances in neural information processing
      systems, 32, 2019.
[103] Jiacheng Ye, Jiahui Gao, Shansan Gong, Lin Zheng, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Beyond
      autoregression: Discrete diffusion for complex reasoning and planning. arXiv preprint arXiv:2410.14157, 2024.
[104] Haoran You, Yichao Fu, Zheng Wang, Amir Yazdanbakhsh, and Yingyan Celine Lin. When linear attention
      meets autoregressive decoding: Towards more effective and efficient linearized large language models. arXiv
      preprint arXiv:2406.07368, 2024.
[105] Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v:
      Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025.
[106] Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete diffusion multimodal large language model with
      parallel decoding. arXiv preprint arXiv:2505.16990, 2025.
[107] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really
      finish your sentence? In Proceedings of the 57th annual meeting of the association for computational linguistics,
      pages 4791–4800, 2019.
[108] Amy Zhang, Clare Lyle, Shagun Sodhani, Angelos Filos, Marta Kwiatkowska, Joelle Pineau, Yarin Gal, and
      Doina Precup. Invariant causal prediction for block mdps. In International Conference on Machine Learning,
      pages 11214–11224. PMLR, 2020.
[109] Yizhe Zhang, Jiatao Gu, Zhuofeng Wu, Shuangfei Zhai, Joshua Susskind, and Navdeep Jaitly. Planner: Generating
      diversified paragraph via latent language diffusion model. Advances in Neural Information Processing Systems,
      36:80178–80190, 2023.
[110] Siyan Zhao, Devaansh Gupta, Qinqing Zheng, and Aditya Grover. d1: Scaling reasoning in diffusion large
      language models via reinforcement learning. arXiv preprint arXiv:2504.12216, 2025.
[111] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang,
      Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 1(2):
      1–124, 2023.



                                                          39
[112] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation
      autoencoders. arXiv preprint arXiv:2510.11690, 2025.
[113] Huangjie Zheng, Shansan Gong, Ruixiang Zhang, Tianrong Chen, Jiatao Gu, Mingyuan Zhou, Navdeep Jaitly,
      and Yizhe Zhang. Continuously augmented discrete diffusion model for categorical generative modeling. arXiv
      preprint arXiv:2510.01329, 2025.
[114] Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion
      models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. arXiv preprint
      arXiv:2409.02908, 2024.
[115] Lin Zheng, Jianbo Yuan, Lei Yu, and Lingpeng Kong. A reparameterized discrete diffusion model for text
      generation. arXiv preprint arXiv:2302.05737, 2023.
[116] Kun Zhou, Yifan Li, Xin Zhao, and Ji-Rong Wen. Diffusion-nat: Self-prompting discrete diffusion for non-
      autoregressive text generation. In Proceedings of the 18th Conference of the European Chapter of the Association
      for Computational Linguistics (Volume 1: Long Papers), pages 1438–1451, 2024.
[117] Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai
      Lin, Ji-Rong Wen, et al. Llada 1.5: Variance-reduced preference optimization for large language diffusion models.
      arXiv preprint arXiv:2505.19223, 2025.
[118] Fengqi Zhu, Zebin You, Yipeng Xing, Zenan Huang, Lin Liu, Yihong Zhuang, Guoshan Lu, Kangyu Wang, Xudong
      Wang, Lanning Wei, et al. Llada-moe: A sparse moe diffusion language model. arXiv preprint arXiv:2509.24389,
      2025.
[119] Hanlin Zhu, Baihe Huang, Shaolun Zhang, Michael Jordan, Jiantao Jiao, Yuandong Tian, and Stuart Russell.
      Towards a theoretical understanding of the’reversal curse’via training dynamics. Advances in neural information
      processing systems, 37:90473–90513, 2024.




                                                          40
Appendix
A     Strict Probabilistic Formulation and Training Objective Details of Cola DLM
This appendix provides the strict probabilistic formulation of Cola DLM and clarifies the role of each training
component. We first define the latent-variable model and the CNF prior precisely, then derive the marginal
likelihood, the ELBO, and its population form, and finally explain the exact role of Flow Matching in the
overall theoretical framework.

A.1   Probabilistic Objects, CNF Prior, and Generative Distribution
Let x ∈ X denote a discrete text sequence, and let z0 ∈ Rd denote its continuous latent variable. Cola DLM
consists of three probabilistic components:

                           qϕ (z0 | x)    encoder (approximate posterior),                                 (A.1)
                           pθ (x | z0 )   decoder (conditional text generator),                            (A.2)
                              pψ (z0 )    latent prior (defined by a continuous flow).                     (A.3)

Here, qϕ (z0 | x) is used only for variational inference during training. The actual generative distribution is
defined by pθ (x | z0 ) and pψ (z0 ).
To ensure that the flow map and density-transport formula are well defined, we impose the standard regularity
assumption on the CNF vector field.
Assumption A.1 (Regularity of the CNF vector field). For each parameter ψ, the vector field vψ (z, t) is
locally Lipschitz in z, measurable in t, and satisfies a linear-growth condition ensuring the existence and
uniqueness of global ODE solutions. In addition, ∇· vψ (z, t) is integrable along the trajectories of interest.
Then, for any 0 ≤ s ≤ t ≤ 1, the flow map Φψt←s exists, is unique, and is invertible.

Let the base distribution be the standard Gaussian

                                               z1 ∼ p1 (z1 ) = N (0, I).                                   (A.4)

Given a time-dependent vector field vψ (zt , t), consider the ODE
                                           dzt
                                               = vψ (zt , t),     t ∈ [0, 1].                              (A.5)
                                           dt
By Assumption A.1, this ODE induces a flow map

                                                    zt = Φψ
                                                          t←s (zs ).                                       (A.6)

Therefore, a prior sample can be written as

                                           z0 = Φψ
                                                 0←1 (z1 ),       z1 ∼ p1 .                                (A.7)

Equivalently, the prior distribution is the pushforward of the base distribution under the reverse flow:

                                                  pψ = (Φψ
                                                         0←1 )♯ p1 .                                       (A.8)

This shows that pψ (z0 ) is not an abstract symbol, but a rigorously defined probability distribution induced
jointly by (p1 , vψ ).
Hence, the generative process of Cola DLM can be written as

                             z1 ∼ p1 (z1 ),       z0 = Φψ
                                                        0←1 (z1 ),         x ∼ pθ (x | z0 ).               (A.9)

The corresponding joint and marginal distributions are
                                                                       Z
                       p(x, z0 ) = pθ (x | z0 )pψ (z0 ),    p(x) =         pθ (x | z0 )pψ (z0 ) dz0 .   (A.10)


                                                           41
If the base variable z1 is introduced explicitly, the marginal can also be written as
                                            Z
                                     p(x) = p1 (z1 ) pθ x | Φψ                                                (A.11)
                                                                       
                                                              0←1 (z1 ) dz1 .


Therefore, Cola DLM is fundamentally a hierarchical latent-variable language model, rather than a model
that directly defines a reverse process in token space.

A.2    Explicit Density of the CNF Prior and the Outer ELBO
Let Zt denote the random variable induced by the ODE at time t, with density pt . Under Assumption A.1, pt
satisfies the continuity equation                                   
                                     ∂t pt (z) + ∇· pt (z) vψ (z, t) = 0.
Along a characteristic trajectory t 7→ zt , the instantaneous change-of-variables formula gives
                                           d
                                              log pt (zt ) = −∇· vψ (zt , t).                                 (A.12)
                                           dt
Integrating from 0 to 1, and noting that p0 = pψ , yields
                                                       Z 1
                       log pψ (z0 ) = log p1 (z1 ) +         ∇· vψ (zt , t) dt,            z1 = Φψ
                                                                                                 1←0 (z0 ).   (A.13)
                                                        0

Thus, the CNF prior admits an explicitly defined log-density rather than an implicit prior that exists only
through sampling.
The flow map can also be written as a deterministic Markov kernel. Define
                                          ψ
                                         Kt←s (dzt | zs ) := δΦψ                 (dzt ).                      (A.14)
                                                                     t←s (zs )


                                           ψ
By the semigroup property of ODE flows, Kt←s   satisfies the Chapman–Kolmogorov relation and therefore
forms a deterministic Markov semigroup. However, the Markov structure here comes from prior transport,
not from an observation-dependent corruption–recovery mechanism.
We now introduce the variational posterior qϕ (z0 | x) to handle the latent integral in the marginal likelihood.
For any qϕ (z0 | x) satisfying the required support condition,

                                                        pθ (x | z0 )pψ (z0 )
                                       Z
                         log p(x) = log qϕ (z0 | x)                          dz0                         (A.15)
                                                            qϕ (z0 | x)
                                               h                                                 i
                                  ≥ Eqϕ (z0 |x) log pθ (x | z0 ) + log pψ (z0 ) − log qϕ (z0 | x) .      (A.16)

Define the outer ELBO as
                                              h                                                 i
                      LELBO (x) := Eqϕ (z0 |x) log pθ (x | z0 ) + log pψ (z0 ) − log qϕ (z0 | x) .            (A.17)

Then
                                               log p(x) ≥ LELBO (x).                                          (A.18)
More importantly, the exact decomposition between the true log-likelihood and the ELBO is

                                                                                                              (A.19)
                                                                              
                             log p(x) = LELBO (x) + KL qϕ (z0 | x) ∥ p(z0 | x) ,

where the true posterior is
                                                             pθ (x | z0 )pψ (z0 )
                                           p(z0 | x) =                            .                           (A.20)
                                                                    p(x)
Equation (A.19) makes the distinction explicit: the true likelihood is always log p(x), whereas the training
objective directly optimized is its variational lower bound LELBO (x).


                                                               42
Substituting the explicit CNF density formula (A.13) into the outer ELBO gives
                                                h                                   Z 1
                      LELBO (x) = Eqϕ (z0 |x) log pθ (x | z0 ) + log p1 (z1 ) +   ∇· vψ (zt , t) dt              (A.21)
                                                                                0
                                                                i
                                              − log qϕ (z0 | x) ,        z1 = Φψ1←0 (z0 ).                       (A.22)

This is the strict training objective of Cola DLM in the setting where the decoder can compute log pθ (x | z0 ).

A.3     Average ELBO, Aggregated Posterior, and Prior Matching
To analyze the nature of prior learning, define the aggregated posterior
                                                  Z
                                      q̄ϕ (z0 ) := qϕ (z0 | x) pdata (x) dx,                                     (A.23)

and the joint distribution
                                              q(x, z0 ) := pdata (x)qϕ (z0 | x).                                 (A.24)
When the encoder qϕ and decoder pθ are fixed, the dependence of the data-averaged ELBO on ψ appears only
through the prior term:

                                                                                                           (A.25)
                                                                                                    
            Epdata (x) LELBO (x) = Eq(x,z0 ) log pθ (x | z0 ) − log qϕ (z0 | x) + Eq(x,z0 ) log pψ (z0 )
                                                                                                           (A.26)
                                                                   
                                  = C(ϕ, θ) + Ez0 ∼q̄ϕ log pψ (z0 ) ,

where C(ϕ, θ) is independent of ψ.
Using                                                                      
                         Ez0 ∼q̄ϕ [log pψ (z0 )] = −KL q̄ϕ (z0 ) ∥ pψ (z0 ) + Ez0 ∼q̄ϕ [log q̄ϕ (z0 )],
we obtain
                                                                                                                 (A.27)
                                                                                               
                          max Epdata (x) LELBO (x)           ⇐⇒       min KL q̄ϕ (z0 ) ∥ pψ (z0 ) .
                            ψ                                          ψ

Therefore, after fixing the encoder and decoder, the strict prior-learning subproblem is to fit the aggregated
posterior q̄ϕ with pψ .
The average ELBO can be further written in information-decomposition form. Adding and subtracting
log q̄ϕ (z0 ) in the prior term yields

                                                                                                                   (A.28)
                                                                                                                
                   Epdata (x) [LELBO (x)] = Eq(x,z0 ) [log pθ (x | z0 )] − Iq (X; Z0 ) − KL q̄ϕ (z0 ) ∥ pψ (z0 ) ,

where                                                       h                          i
                                   Iq (X; Z0 ) := Epdata (x) KL qϕ (z0 | x) ∥ q̄ϕ (z0 )                          (A.29)

is the amount of information retained in the latent variable.
Equation (A.28) reveals the three theoretical roles of the encoder. It determines the target distribution
q̄ϕ (z0 ) that the prior must fit, the compression strength through Iq (X; Z0 ), and therefore the division of labor
between global semantics in latent space and local realization in the decoder.

A.4     The Role of Flow Matching: A Prior Solver Rather Than the Model Definition
Direct optimization of log pψ (z0 ) usually requires repeated ODE solves and divergence estimation, which is
expensive in practice. For this reason, Flow Matching is often used to learn the vector field of the same prior
more efficiently. It should therefore be viewed as a prior solver, not as the likelihood objective itself.
To do so, one first specifies a bridge path from the aggregated posterior to the base distribution. Let

                                    z0 ∼ q̄ϕ (z0 ),      z1 ∼ p1 (z1 ),       t ∼ U [0, 1].                      (A.30)


                                                              43
Consider the conditional path
                                 zt = (1 − α(t))z0 + α(t)z1 ,          α(0) = 0, α(1) = 1.                 (A.31)
Here α can be any sufficiently smooth monotone function; linear interpolation corresponds to α(t) = t. The
target velocity field of this path is
                                                             dzt
                                          ut (z0 , z1 ) :=       = α̇(t)(z1 − z0 ).                        (A.32)
                                                             dt
The Flow Matching loss is then
                                                              h                               i
                                                                                            2
                            LFM (ψ; ϕ) := Et, z0 ∼q̄ϕ , z1 ∼p1 ∥vψ (zt , t) − ut (z0 , z1 )∥2 .            (A.33)

For a fixed path construction, the pointwise optimal solution is the conditional expectation
                                     vψ⋆ (z, t) = E ut (z0 , z1 ) | zt = z, t .                            (A.34)
                                                                            

This is simply a conditional-mean regression result, but it makes the role of Flow Matching precise: under
ideal conditions, the learned vector field corresponds to a continuous transport connecting q̄ϕ and p1 , which
in turn defines the prior pψ (z0 ).
Hence, two objectives must be distinguished. The first,
                                                max Ez0 ∼q̄ϕ [log pψ (z0 )],                               (A.35)
                                                  ψ

is the strict probabilistic objective for prior learning. The second,
                                                      min LFM (ψ; ϕ),                                      (A.36)
                                                       ψ

is a practical solver for learning the vector field of the same prior. They serve the same prior-modeling problem,
but they are not the same mathematical object. In particular, LFM cannot be identified term-by-term with
− log pψ (z0 ) in the ELBO.

A.5    Strict Form of the Training Objective
If the decoder can compute log pθ (x | z0 ) exactly, then the strict training objective is to maximize (A.17), or
equivalently to minimize the negative ELBO. Define
                                                                                                          (A.37)
                                                                              
                                     LREC (x) := −Eqϕ (z0 |x) log pθ (x | z0 ) ,
                                                                                                          (A.38)
                                                                          
                                   LPRIOR (x) := −Eqϕ (z0 |x) log pψ (z0 ) ,
                                                                                                          (A.39)
                                                                            
                                     LREG (x) := Eqϕ (z0 |x) log qϕ (z0 | x) ,
so that the total single-sample loss becomes
                        Lstrict
                         Total (x) := LREC (x) + LPRIOR (x) + LREG (x) = −LELBO (x).                       (A.40)
Taking expectation over the data distribution gives the strict training objective
                       min Epdata (x) Lstrict                                                              (A.41)
                                                                               
                                        Total (x)   ⇐⇒ max Epdata (x) LELBO (x) .
                         ϕ,θ,ψ                                       ϕ,θ,ψ

Therefore, the objective that truly corresponds to maximizing a lower bound of the marginal likelihood is
always the ELBO; Flow Matching is only an efficient implementation for learning the prior.

B     Sampling, Inference, and Probability Estimation Details
This appendix supplements the strict formulation of sampling, continuation, unconditional likelihood estimation,
and conditional probability estimation in Cola DLM. Since the main text only presents the overall framework,
we provide the details here.


                                                               44
B.1   Sampling and Inference Procedure
To generate a new sample from Cola DLM, one should follow the generative process defined by the model,
rather than the approximate posterior path used during training. First, sample a latent seed from the simple
base distribution:
                                          z1 ∼ p1 (z1 ) = N (0, I).                                     (B.1)
Second, integrate the vector field backward to obtain a latent sample:
                                         dzt
                                             = vψ (zt , t),    t : 1 → 0,                                 (B.2)
                                         dt
so that starting from z(1) = z1 yields z(0) = z0 . In practice, Euler, Heun, or higher-order ODE solvers can be
used. This step generates global latent semantics rather than tokens themselves.
Third, the decoder generates text conditioned on the latent:
                                                       x ∼ pθ (x | z0 ).                                             (B.3)
If the decoder is autoregressive, this step is conditional autoregressive sampling; if the decoder is non-
autoregressive or bidirectional, it becomes the corresponding conditional recovery procedure. Hence, sampling
in Cola DLM is inherently two-stage: one first samples a global latent, and then realizes the text conditioned
on it.
If a task provides a prefix xpre , inference first encodes the prefix into a conditional latent:
                                                  z pre ∼ qϕ (z pre | xpre ),                                        (B.4)
                                                                                                (<b)
and then generates the response latent block by block. Let the previously generated blocks be ẑ0 . The b-th
response block is generated by
                                    (b)                        (<b) 
                                   ẑ0 = Φψ    (b) pre
                                          0←1 ϵ ; z    , ẑ0            ,    ϵ(b) ∼ N (0, I).                        (B.5)
The response is then decoded as
                                                                             (1:B) 
                                           x̂res ∼ pθ xres | xpre , z pre , ẑ0        .                             (B.6)
This is exactly the prefix-encoding, blockwise-generation, and conditional-decoding procedure described in the
main text.

B.2   Unconditional Log-Likelihood and PPL Estimation
For a dataset D, if one directly uses the ELBO as a lower bound on log p(x), one may define an ELBO-based
perplexity as                                       P
                                                             LELBO (x)
                                                                       
                                   PPLELBO := exp − x∈D   P              ,                          (B.7)
                                                            x∈D |x|
where |x| denotes sequence length. Since LELBO (x) ≤ log p(x), PPLELBO should be interpreted as an
upper-bound-type estimate of the true perplexity.
If the decoder can compute log pθ (x | z0 ), one can also obtain tighter likelihood estimates through importance
sampling or IWAE. Let
                 (k)                                                 (k)                   (k)         (k)
               z0      ∼ qϕ (z0 | x),     log w(k) = log pθ (x | z0 ) + log pψ (z0 ) − log qϕ (z0            | x).   (B.8)
Then the ELBO-style and IWAE-style estimators are
                                          K                                                      K
                                                                                                             !
                                   1 X                                                1 X log w(k)
                log pbELBO,K (x) =     log w(k) ,              log pbIWAE,K (x) = log    e                       .   (B.9)
                                   K                                                  K
                                          k=1                                                    k=1
The IWAE estimator is generally tighter. When IWAE is used for scoring, the corresponding perplexity is
                                               P                     
                                                  x∈D log p
                                                          bIWAE,K (x)
                             PPLIWAE := exp −         P                  .                        (B.10)
                                                         x∈D |x|
If the decoder itself can provide only a lower bound, then evaluation can report only bound-based PPL rather
than exact perplexity.


                                                               45
B.3    Conditional Probability Estimation and Block-Level Conditional Scoring
In continuation, multiple-choice evaluation, and candidate-response ranking, conditional probability is more
important than the unconditional score. Let a full sample be decomposed as
                                                            x = (xpre , xres ),                                           (B.11)
where x     pre
                  is the given prefix and x   res
                                                    is the response to be evaluated. The exact conditional log-probability
satisfies
                                      log p(xres | xpre ) = log p(xpre , xres ) − log p(xpre ).                           (B.12)
Hence one can apply the same unconditional estimator separately to the joint sample and the prefix sample,
and then take the difference to obtain a plug-in conditional estimator:
                                      ELBO
                                log
                                 dp
                                    cond (x
                                           res
                                               | xpre ) := LELBO (xpre , xres ) − LELBO (xpre ),                          (B.13)
and
                               IWAE
                          log
                           dp
                              cond,K (x
                                       res
                                           | xpre ) := log pbIWAE,K (xpre , xres ) − log pbIWAE,K (xpre ).                (B.14)
It should be emphasized that both (B.13) and (B.14) are plug-in estimators. They are obtained by substituting
unconditional estimators into the conditional identity, and they do not automatically inherit a lower-bound
property from the unconditional estimators. Strictly speaking, they should therefore be interpreted as practical
estimators of conditional log-likelihood.
For a block-causal prior, the conditional estimator can also be written in a blockwise form aligned with the
architecture. Suppose the latent sequence is partitioned into prefix blocks and response blocks:
                                           z = (z (1) , . . . , z (B) , z (B+1) , . . . , z (B+R) ).                      (B.15)
If the prior admits a block-level causal factorization, then
                                                                     B+R
                                                                     Y
                                              pψ (z) = pψ (z (1) )          pψ (z (b) | z (<b) ).                         (B.16)
                                                                      b=2

Conditioned on the prefix latent z pre := z (1:B) , the conditional marginal of the response is
                                           Z
                    p(x | x , z ) = pθ (xres | xpre , z pre , z res ) pψ (z res | z pre ) dz res ,
                        res   pre pre
                                                                                                                          (B.17)

where z res := z (B+1:B+R) . For any approximate posterior qϕ (z res | xpre , xres , z pre ), one has the conditional
ELBO
                                                                        h
              log p(xres | xpre , z pre ) ≥Eqϕ (zres |xpre ,xres ,zpre ) log pθ (xres | xpre , z pre , z res )              (B.18)
                                                                                                                         i
                                                        + log pψ (z res | z pre ) − log qϕ (z res | xpre , xres , z pre ) . (B.19)

Marginalizing over the true posterior of the prefix latent then yields
                                           Z
                          p(xres | xpre ) = p(xres | xpre , z pre ) p(z pre | xpre ) dz pre .                             (B.20)

Thus, a strict conditional probability requires not only marginalization over the response latent, but also an
additional integration over the posterior uncertainty of the prefix latent. In practice, two approximations are
common: one uses the joint-minus-prefix estimator in (B.12), while the other approximates p(z pre | xpre ) with
the encoder mean or a single encoder sample and then evaluates (B.19).
If the response contains only one new block, the above expression reduces to a single-block response score:
                                                                       h
   Sresp (x(B+1) ; x(1:B) , z (1:B) ) :=Eqϕ (z(B+1) |x(1:B+1) ,z(1:B) ) log pθ (x(B+1) | x(1:B) , z (1:B+1) )               (B.21)
                                                                                                                         i
                                                   + log pψ (z (B+1) | z (1:B) ) − log qϕ (z (B+1) | x(1:B+1) , z (1:B) ) . (B.22)

This form is especially useful for scoring multiple-choice options, ranking candidate responses, and analyzing
whether the separation between candidates is driven mainly by the prior term or the decoder term.


                                                                     46
B.4   Numerical Evaluation of the CNF Prior and Estimation Algorithms
                     (k)
To evaluate log pψ (z0 ) in (B.8), one solves the augmented ODE
                                                               (k) 
                                 d zt         vψ (zt , t)        z0   z
                                         =                    ,     = 0 .                                                (B.23)
                                 dt tℓ       ∇· v  (z
                                                 ψ t    , t)     ℓ0     0
                                           (k)    (k)
Integrating from t = 0 to t = 1 gives (z1 , ℓ1 ), and therefore
                                                       (k)                (k)          (k)
                                          log pψ (z0 ) = log p1 (z1 ) + ℓ1 .                                             (B.24)

In high dimensions, the divergence can be approximated by the Hutchinson trace estimator:
                                                                                  
                                                ∂vψ (zt , t)           ∂vψ (zt , t)
                           ∇· vψ (zt , t) = Tr                 = Eϵ ϵ⊤              ϵ .                                  (B.25)
                                                  ∂zt                    ∂zt
In practice, one usually fixes the same random vector ϵ within a single ODE solve so that the induced dynamics
remain consistent along that trajectory.
We now present two estimators corresponding to the above formulas. To avoid extra package dependencies,
the algorithms are written in directly compilable pseudo-code style.
Algorithm A.1: Unconditional Log-Likelihood Estimation (ELBO / IWAE)

      Input: sequence x; encoder qϕ ; decoder pθ ; CNF prior pψ ; sample number K; estimation mode
      mode ∈ {ELBO, IWAE}.
      Procedure:
                                                 (k)
        1. For each k = 1, . . . , K, sample z0        ∼ qϕ (z0 | x).
                             (k)                                                         (k)         (k)
        2. Starting from (z0 , 0), solve (B.23) to t = 1 and obtain (z1 , ℓ1 ).
                              (k)                (k)         (k)
        3. Compute log pψ (z0 ) = log p1 (z1 ) + ℓ1 .
                                    (k)                  (k)
        4. Compute log pθ (x | z0 ) and log qϕ (z0             | x).
        5. Construct the importance weight
                                                             (k)                 (k)                       (k)
                              log w(k) ← log pθ (x | z0 ) + log pψ (z0 ) − log qϕ (z0                            | x).

        6. If mode = ELBO, return
                                                                        K
                                                   dp(x) = 1
                                                                        X
                                                  log                         log w(k) ;
                                                           K
                                                                        k=1

           otherwise return                                                                  !
                                                                         K
                                             dp(x) = log 1          (k)
                                                           X
                                            log              elog w                              .
                                                         K
                                                                         k=1


      Output: an unconditional log-likelihood estimate log
                                                        dp(x) for sequence x.

Algorithm A.2: Conditional Log-Likelihood Estimation (Joint Minus Prefix)

      Input: prefix xpre ; response xres ; sample number K; estimation mode mode ∈ {ELBO, IWAE}.
      Procedure:

        1. Construct the full sequence xfull = (xpre , xres ).
        2. Call Algorithm A.1 to estimate log
                                           dp(xfull ).

        3. Call Algorithm A.1 to estimate log
                                           dp(xpre ).


                                                                   47
        4. Compute
                                      dp(xres | xpre ) ← log
                                     log                  dp(xfull ) − log
                                                                        dp(xpre ).

      Output: a conditional log-likelihood estimate log
                                                     dp(xres | xpre ).

Algorithm A.1 applies to both ELBO-style and IWAE-style estimators. Algorithm A.2 treats it as a black-box
unconditional likelihood estimator and combines two calls through the conditional identity. If the task focuses
only on the local score of a single newly generated block, then (B.22) can be used directly to rank candidate
responses.

C     Comparison Under a Unified Markov Path View
This appendix further develops the mathematical comparison behind the main-text discussion of unified text
modeling. The central question is not which method is more diffusion-like at a superficial level, but rather:
into what state space and along what type of path does each method factorize the text distribution, and what
role does that path play in generation?

C.1   A Unified Framework for Process-Based Generative Models
To establish a common language, consider the abstract definition of a process-based generative model. Let

                                            Y = (Yt )t∈T ,        Yt ∈ S,                                    (C.1)

where S is the state space and T is either discrete time or continuous time. If there exists a family of transition
kernels
                                           Kt←s (dyt | ys ),     s ≤ t,                                      (C.2)
satisfying the Chapman–Kolmogorov relation, then Y forms a Markov process. If one additionally specifies an
emission kernel
                                               eθ (x | yt⋆ ),                                         (C.3)
then the text distribution can be written uniformly as
                                                Z
                                        pθ (x) = eθ (x | yt⋆ ) pθ (dY ).                                     (C.4)


Within this framework, the difference among AR, LLaDA, Plaid, and Cola DLM is not simply whether each
admits a stepwise sampling algorithm. The essential distinction is that they factorize the text distribution
over different state spaces and assign fundamentally different roles to the underlying paths. More specifically,
they differ at least in the following six aspects: state space, state trajectory object, transition mechanism, the
role of the path in generation, the point at which the observation is read out, and the tightness of the training
objective relative to the true log-likelihood.
This unified view shows that the key question is not which method uses diffusion, but rather what type of
distributional complexity is decomposed into what kind of path over what kind of state space, and whether
that path is an observation-dependent corruption path or an observation-independent latent prior path.

C.2    Difference from Autoregressive Models: Why Diffusion
AR models text directly via the chain rule:
                                                      L
                                                      Y
                                           pAR
                                            η (x) =          pη (xi | x<i ).                                 (C.5)
                                                      i=1

Its negative log-likelihood is
                                                      L
                                                      X
                                     − log pAR
                                            η (x) =          − log pη (xi | x<i ).                           (C.6)
                                                      i=1


                                                        48
The advantage of AR is therefore exact token-level likelihood and the most direct training and evaluation
protocol. Its limitation comes from the fixed left-to-right conditioning order.
Under the unified path framework, AR can be written using prefix states

                                                  Si := x1:i ,        S0 := ∅.                                   (C.7)

Then (Si )L
          i=0 forms a time-inhomogeneous Markov chain, whose one-step transition kernel is exactly the AR
conditional distribution:

                            P(Si = si | S0:i−1 ) = P(Si = si | Si−1 = si−1 ) = pη (xi | x<i ).                   (C.8)

Thus, the core inductive bias of AR is not merely Markovianity, but the fact that conditioning is restricted to
the unique prefix filtration
                                      σ(X1 ) ⊂ σ(X1:2 ) ⊂ · · · ⊂ σ(X1:L ).                               (C.9)

By contrast, the model distribution of Cola DLM is not defined through a fixed token order, but by latent
marginalization:                               Z
                                         pCola DLM (x) =         pθ (x | z0 ) pψ (z0 ) dz0 .                    (C.10)

If the decoder itself is not left-to-right autoregressive, the resulting model distribution is not tied to any
pre-specified token ordering. In this sense, the non-AR nature of Cola DLM comes from two levels: the prior
pψ (z0 ) is defined by continuous transport rather than token-wise chain factorization, and if the decoder is also
non-AR, then generation becomes a two-stage procedure of first sampling global semantic latent variables and
then realizing text conditionally.
From the optimization viewpoint, both families minimize

                                                                                                                (C.11)
                                                                           
                                                  KL pdata (x) ∥ pmodel (x) ,

but they parameterize p(x) differently. AR uses a chain-rule factorization, whereas Cola DLM uses latent
marginalization. Therefore, the strict meaning of why diffusion here is not to avoid maximum likelihood, but
to improve the fit of pψ (z0 ) to the aggregated posterior through a more expressive continuous prior family.
This can be quantified exactly. For any two candidate priors pa and pb ,

                                                                                                                (C.12)
                                                                                                          
               Eq̄ϕ (z0 ) [log pb (z0 ) − log pa (z0 )] = KL q̄ϕ (z0 ) ∥ pa (z0 ) − KL q̄ϕ (z0 ) ∥ pb (z0 ) .

Hence, whenever the flow/CNF prior is closer to q̄ϕ (z0 ) than a simple Gaussian prior, it improves the average
ELBO.

C.3    Difference from Discrete Denoising Language Models: Why Continuous
Discrete denoising language models such as LLaDA operate directly in token space. A typical training
objective can be written as
                                                     " L                                    #
                                                      1X       i                  i
                                                                                              (C.13)
                                                                                          
                     LMDM (θ) := Et∼U [0,1], x0 , xt        1[xt = M ] − log pθ (x0 | xt ) ,
                                                      t i=1

and satisfies
                                            −Epdata (x0 ) log pθ (x0 ) ≤ LMDM (θ).                              (C.14)
Thus, LLaDA learns a token-space distribution through reverse recovery under random mask ratios. Its
advantage is that it has no explicit encoder and no latent bottleneck; its limitation is that it always remains
in a discrete state space.
More precisely, the forward masking process of LLaDA can be embedded into a continuous-time Markov chain.
For a single token, let the state space be V ∪ {M } and define the generator

                            Gτ (a, M ) = λ(τ ),       Gτ (a, a) = −λ(τ ),             Gτ (M, M ) = 0,           (C.15)


                                                                 49
where a ∈ V. Then the transition probability from any non-mask state to the absorbing mask state is
                                                            Z τ          
                             P(Xτ = M | X0 = a) = 1 − exp −       λ(u) du .                         (C.16)
                                                                      0

If the right-hand side is identified with t ∈ [0, 1], one recovers the forward kernel in which each token is masked
with probability t, as used in (C.13). Therefore, LLaDA can be understood rigorously as a reverse recovery
problem over a CTMC with discrete states and absorption into the mask state.
By contrast, Cola DLM models pψ (z0 ) in continuous latent space rather than performing discrete diffusion in
token space. This continuous formulation provides three strict advantages. First, continuous-time change-
of-variables yields an explicit log-density, namely (A.13). Second, vector-field regression in Euclidean space
allows the use of Flow Matching to learn prior geometry directly. Third, continuous latent variables naturally
support interpolation, gradient guidance, and representation editing.
Accordingly, the meaning of why continuous is not that continuous representations are inherently superior
to discrete ones. Rather, compared with a discrete token-space corruption model, continuous latent space
permits a more direct description of probability geometry and reversible transport. Whether this yields actual
gains still depends on whether the continuous structure aligns with the low-dimensional semantic organization
of the data.

C.4   Difference from Plaid: Why Latent
Plaid is also a continuous method, but it makes continuous a token-aligned representation that stays close
to the original sequence, rather than an explicit semantic latent variable. A typical forward process can be
written as
                                      x, σ 2 (0)I),  q(ht | hs ) = N hs , σ 2 (t) − σ 2 (s) ,          (C.17)
                                                                                           
                       q(h0 | x) = N (e
where xe = Embed(x). Hence, as the noise level approaches zero, the state of Plaid still remains tightly
attached to the observation space. By contrast, Cola DLM explicitly separates global semantic representation
from textual realization through               Z
                                       p(x) = pθ (x | z0 )pψ (z0 ) dz0 .                              (C.18)

This distinction is not merely about where continuity is introduced; it is about how the model is factorized. If
one further lets
                                   qϕ (z0 | x) = N (E(x), σ02 I),  σ02 → 0,                              (C.19)
and E(x) is nearly invertible, then Cola DLM degenerates to the limiting case where z0 is almost just a
continuousized representation of the original sequence. This limit clarifies the structural relation: Plaid is
closer to the near-observation, non-compressive limit of Cola DLM, while the genuinely new ingredient of
Cola DLM is the latent decomposition itself.
The mathematical core of this difference is captured by the information decomposition of the average ELBO
in (A.28). That decomposition shows that representation learning in Cola DLM splits into three parts: the
conditional reconstruction term controls textual realization given the latent, the mutual information Iq (X; Z0 )
controls how much global information about the text is retained, and the KL between the aggregated posterior
and the prior controls the difficulty of prior matching. Precisely because this explicit latent variable is
introduced, the model can encode global planning, topic, and style constraints into z0 , while delegating local
token realization to the decoder.
Therefore, the strict answer to why latent is that a latent variable is not merely a continuous relaxation
of a discrete object. It is an intermediate random variable that explicitly participates in marginalization,
allowing the text generation problem to be decomposed into prior modeling plus conditional realization. This
decomposition yields statistical benefits only when the data indeed contain low-rate but high-value semantic
structure; this issue is analyzed further in Appendix D.




                                                        50
C.5    Why Cola DLM Is Not Merely a Latent Reverse-Diffusion LM
To avoid conceptual confusion, we now make the distinction more precise. Suppose a narrow latent diffusion
LM is defined as follows: one first specifies a data-conditioned forward noising process on a latent state,

                                                     q(z0:T | x),                                         (C.20)

and then learns its reverse process to gradually recover a latent state used for text generation. The core
structure of such a model is therefore the reversal of a data-conditioned forward corruption process.
Under this narrow definition, Cola DLM is not simply a latent reverse-diffusion LM. First, the continuous
path in Cola DLM lies in prior transport:

                               z1 ∼ p1 (z1 ),    z0 = Φψ
                                                       0←1 (z1 ),       x ∼ pθ (x | z0 ).                 (C.21)

This path first describes how to generate a semantic prior sample in latent space from a simple Gaussian.
During generation, it does not depend on any specific observation x. By contrast, the forward paths in Plaid
and LLaDA describe how a given sample x is perturbed into a high-noise state. Thus, although all three
can be described as path models, the path in Cola DLM is prior transport, whereas the paths in Plaid and
LLaDA are data corruption.
Second, in Cola DLM the encoder appears only in variational inference:
                                          h                                                 i
                    log p(x) ≥ Eqϕ (z0 |x) log pθ (x | z0 ) + log pψ (z0 ) − log qϕ (z0 | x) .            (C.22)

It does not participate in the generation procedure (C.21), and is therefore part of the variational posterior.
In Plaid and LLaDA, however, the forward process q is part of the model definition because the reverse model
is trained against that corruption law. In short, the former q belongs to inference, whereas the latter q belongs
to model construction.
Finally, the marginal distribution of Cola DLM is always defined by the hierarchical latent-variable identity
                                              Z
                                        p(x) = pθ (x | z0 )pψ (z0 ) dz0 .                              (C.23)

Strictly speaking, Cola DLM is therefore first and foremost a hierarchical latent-variable language model with
a CNF prior. Flow or diffusion is only a way to make the prior family more expressive, not the entirety of the
model itself.

D     Theoretical Advantages, Applicability, and Limitations
We finally compare Cola DLM with AR, LLaDA, and Plaid under a unified theoretical criterion. The goal is
not to argue that a more complex model should be stronger, but to ask under what conditions Cola DLM is
expected to be better, and under what conditions it is expected to be worse.

D.1   A Unified Population-Level Comparison Criterion
For any model class M, define its irreducible approximation error with respect to the true data distribution
pdata (x) as
                                                                                                       (D.1)
                                                                     
                                     E(M) := inf KL pdata (x) ∥ p(x) .
                                                   p∈M

This quantity measures the best unavoidable mismatch of the model class even after ignoring finite-sample
effects and optimization error.
For any model trained by exact negative log-likelihood, one has the standard identity

                                   inf Epdata (x) [− log p(x)] = H(pdata ) + E(M),                         (D.2)
                                  p∈M




                                                          51
where H(pdata ) := −Epdata (x) log pdata (x) is the data entropy. Therefore, for exact-NLL models, comparing
theoretical limits is equivalent to comparing E(M).
AR belongs to this class, and its population risk is

                             Epdata (x) [− log pAR                               AR
                                                                                                             (D.3)
                                                                                    
                                                η (x)] = H(pdata ) + KL pdata ∥ pη    .

By contrast, Cola DLM is trained through the ELBO, so for each sample

                                                                                                             (D.4)
                                                                                      
                          −LELBO (x) = − log pθ,ψ (x) + KL qϕ (z0 | x) ∥ pθ,ψ (z0 | x) .

Taking expectation over the data distribution gives
                                                                           infer
                                                                                                             (D.5)
                                                                       
                  Epdata (x) [−LELBO (x)] = H(pdata ) + KL pdata ∥ pθ,ψ + GCola  DLM (ϕ, θ, ψ),

where
                         infer
                                                                                                             (D.6)
                                                                                         
                        GCola  DLM (ϕ, θ, ψ) := Epdata (x) KL qϕ (z0 | x) ∥ pθ,ψ (z0 | x) ≥ 0

is the inference gap.
Similarly, the variational objective of Plaid can be written as true NLL plus a reverse/path-posterior gap, and
the mask-denoising objective of LLaDA can be written as true token-space NLL plus a surrogate gap. Hence
all four model classes can be unified as

  population risk = H(pdata ) + model mismatch + additional gap induced by the training objective.           (D.7)

Therefore, theoretical superiority does not follow from model complexity alone. It depends on whether the
total statistical burden is smaller.
Define the total statistical burden of Cola DLM as
                                                              infer
                            RCola DLM := E(MCola DLM ) + inf GCola  DLM (ϕ, θ, ψ),                           (D.8)
                                                                ϕ,θ,ψ

and define analogous quantities for AR, Plaid, and LLaDA. Then, at the population level, Cola DLM is better
than a comparison class if and only if its total statistical burden is smaller. For example,

               RCola DLM < RAR ⇐⇒ Cola DLM is better than AR at the population level.                        (D.9)

This gives a unified and rigorous comparison criterion.

D.2     Rate-Distortion and the Applicability Boundary of the Latent Bottleneck
The unified criterion still leaves a deeper question unanswered: why can latent decomposition reduce model
mismatch in some cases but hurt modeling in others? The answer comes from conditional entropy, mutual
information, and rate-distortion structure.
For any encoder qϕ (z0 | x) and decoder pθ (x | z0 ), under the joint distribution q(x, z0 ) = pdata (x)qϕ (z0 | x)
one always has
                                  Eq(x,z0 ) [− log pθ (x | z0 )] ≥ Hq (X | Z0 ).                            (D.10)
By the mutual-information identity

                                       Hq (X) = Hq (X | Z0 ) + Iq (X; Z0 ),                                 (D.11)

together with q(x) = pdata (x), it follows that

                                      Hq (X | Z0 ) = H(pdata ) − Iq (X; Z0 ).                               (D.12)

Therefore, the reconstruction cost satisfies

                               Eq(x,z0 ) [− log pθ (x | z0 )] ≥ H(pdata ) − Iq (X; Z0 ).                    (D.13)


                                                          52
This shows that if Iq (X; Z0 ) is small, then the latent retains little information about the text and the lower
bound on the best achievable reconstruction loss rises. If Iq (X; Z0 ) is large enough, then the decoder can
realize the text more easily given the latent.
To formalize this further, define the representation rate-distortion function

                          D(R) :=            inf              inf       Eq(x,z0 ) [− log pθ (x | z0 )].   (D.14)
                                    q(z0 |x): Iq (X;Z0 )≤R pθ (x|z0 )

It represents the minimum achievable average reconstruction cost when the latent channel is allowed to
transmit at most R nats of information about the text.
If, for a given data distribution, D(R) falls quickly to a low level already at small R, then the data admit a
low-rate but high-value latent representation, and a latent bottleneck is more likely to help. If substantial
reduction in reconstruction error occurs only when R approaches H(X), then the data are nearly incompressible
and a latent bottleneck is more likely to become an additional burden.
Therefore, whether Cola DLM benefits from the bottleneck is ultimately governed by the rate-distortion curve
of the data itself. If global topics, plans, style, and constraints occupy a relatively low information rate yet
strongly determine the final realization, then encoding them into z0 is efficient. If nearly every token-level
detail is irreducible, then latent compression does not bring genuine statistical benefit.

D.3   Structured Generation Regime: When Cola DLM Better Matches the True Mechanism
To formalize the above intuition one step further, consider the following structured generation assumption:
there exists a global variable G such that the true data distribution can be written as
                                                  Z
                                       pdata (x) = p⋆ (x | g) p⋆ (g) dg,                             (D.15)

where G determines topic, plan, discourse structure, style, or other global constraints, and p⋆ (x | g) realizes
these global factors into concrete text. If, in addition,

                               H(X | G) ≪ H(X),                 dim(G) ≪ dim(E(X)),                       (D.16)

then the data are in a favorable regime dominated by low-dimensional but high-value latent factors.
In this regime, the inductive bias of Cola DLM is better aligned with the true generation mechanism. The
encoder recovers posterior information about G from x, the prior learns a global semantic distribution close
to p⋆ (g), and the decoder learns textual realization conditioned on the global factor. In other words, the
potential advantage of Cola DLM does not come from being more complicated, but from splitting a hard
problem into two better-matched subproblems:

                            learning pdata (x)       ⇝      learning p⋆ (g) and p⋆ (x | g).               (D.17)

If the encoder can recover information about G with a small inference gap, pψ (z0 ) can approximate the
aggregated latent distribution corresponding to G, and pθ (x | z0 ) can approximate the conditional realization
well, then the statistical burden of Cola DLM is determined primarily by the modeling complexity of the
low-dimensional global factor rather than that of the full surface sequence. This is the fundamental source of
its possible advantage over AR, LLaDA, and Plaid.

D.4   Three Curves, Theoretical Advantages, and Limitations
In the opposite regime, where (D.16) does not hold, Cola DLM may systematically underperform more direct
modeling routes. Typical unfavorable cases include the absence of a low-rate sufficient representation, settings
where local surface form itself carries the main semantics, or regimes where a strong token-space model is




                                                             53
already adequate. In such cases, the additional costs of Cola DLM mainly come from three sources:
                          (i) inference gap GCola
                                             infer
                                                   DLM ,
                         (ii) the elevated reconstruction lower bound H(X | Z0 )
                             induced by the bottleneck,                                                    (D.18)
                         (iii) the optimization complexity of jointly training encoder,
                             prior, and decoder.
Hence, the success or failure of Cola DLM can be reduced to a competition among three curves: the
representation rate-distortion curve D(R), the prior-approximation curve, and the inference-gap curve
        DLM . Only when all three favor Cola DLM does latent decomposition become a true theoretical
  infer
GCola
advantage.
These conclusions can be summarized more directly as follows.
Theoretical Advantages. Relative to AR, if the decoder is non-autoregressive, then Cola DLM is not
constrained by a single left-to-right ordering and can determine global semantics in latent space before
realizing text, which gives it a potentially more natural inductive bias for infilling, constraint satisfaction,
and global planning. Relative to discrete DLMs, Cola DLM has the geometric structure of continuous latent
space and therefore supports explicit log-densities, vector-field regression, interpolation, and latent editing.
Relative to Plaid, Cola DLM explicitly introduces semantic compression, allowing the model to learn a
low-dimensional global representation. If the data indeed exhibit a hierarchical structure of low-dimensional
global semantics and high-dimensional local realization, this inductive bias is closer to the true mechanism.
Finally, the information decomposition of the average ELBO makes it natural to analyze the representation,
prior, and decoder as separate components.
Theoretical Limitations. Cola DLM inherently suffers from a variational gap:

                                                                                                           (D.19)
                                                                                
                               log p(x) − LELBO (x) = KL qϕ (z0 | x) ∥ p(z0 | x) .
If the encoder is poor, the model pays a systematic likelihood penalty. The latent bottleneck also introduces the
risk of posterior collapse: if the decoder is too strong, the latent dimension is too small, or the regularization
is too strong, then Iq (X; Z0 ) may become too small and the encoder may stop transmitting useful information.
In addition, accurate evaluation of log pψ (z0 ) often requires ODE integration and divergence estimation,
making evaluation more expensive than in AR models. Importantly, the potential advantage of Cola DLM
does not follow automatically from using latent variables and flows. It relies on the inductive assumption that
the data contain a favorable low-dimensional semantic latent structure. This requires experiments to verify.

E    Existence of Global Semantic Structures
The goal of this section is to establish a falsifiable null hypothesis and, based on it, show that if the latent
representation consists only of purely local and separable components, then the optimal timestep shift should
not exhibit a systematic drift as the latent dimension varies. We then compare the experimentally observed
shift drift against this null hypothesis, thereby obtaining a more robust conclusion regarding the existence of
shared structures. To avoid conflating conclusions that can be established rigorously with structural models
introduced for interpretation, we first formulate an abstract null hypothesis.
Assumption E.1 (Null hypothesis of purely separable representations). Let d denote the latent dimension,
let δ denote the timestep shift, and let Jd (δ) denote the overall performance objective under a fixed semantic
metric at latent dimension d. Assume that the following conditions hold:
    1. Local separability: there is no cross-dimensional joint structure in the latent representation that
       affects semantics, and the overall semantic performance is obtained by independently aggregating the
       contributions of local dimensions;
    2. Homogeneity: each local dimension shares the same optimal temporal scale with respect to shift;
       changing d only changes the number of independent dimensions and does not alter the functional form
       of each local subproblem;


                                                       54
   3. Additive aggregation: the overall objective can be written as
                                             d                                        d
                                             X                                     1X
                                  Jd (δ) =         ji (δ)      or       Jd (δ) =         ji (δ),            (E.1)
                                             i=1
                                                                                   d i=1

      where all ji attain their optimum at the same point. More strongly, if homogeneity holds exactly, then
      there exists a common function j and constants ad > 0, bd such that

                                                      Jd (δ) = ad j(δ) + bd .                               (E.2)

Assumption E.1 does not presuppose any specific network architecture. Rather, it describes a theoretical limit
in which semantics is carried entirely by mutually independent local dimensions. Under this limit, changing
the latent dimension can at most modify the scale of the overall objective, but should not change the location
of the optimal shift.
Proposition E.2 (Dimension-invariance of the optimal shift under the separable null). Under Assumption E.1,
let
                                           δd⋆ ∈ arg max Jd (δ)
                                                               δ

denote the optimal timestep shift at latent dimension d. Then for any two dimensions d1 , d2 , we have

                                       arg max Jd1 (δ) = arg max Jd2 (δ).                                   (E.3)
                                              δ                         δ

In particular, if the maximizer is unique, then

                                                        δd⋆1 = δd⋆2 .                                       (E.4)

Therefore, under purely local and separable representations, the optimal timestep shift should not drift
systematically with the latent dimension.

Proof. By (E.2), for any d,
                                       Jd (δ) = ad j(δ) + bd ,              ad > 0.
Since bd is independent of δ and ad only induces a positive rescaling, it follows that

                                         arg max Jd (δ) = arg max j(δ).
                                                  δ                     δ

Hence the optimal point is independent of d, which proves (E.3) and (E.4).

The key conclusion of Proposition E.2 is that what can be established rigorously is that if no shared or joint
structure affecting semantics exists, then the optimal shift should not drift systematically with the latent
dimension.
Corollary E.3 (Systematic drift refutes the purely separable null). If experiments show that, as the latent
dimension d varies, the optimal shift δ ⋆ (d) exhibits a stable, monotonic, and reproducible systematic drift, and
if this drift cannot be explained by changes in parameter count, insufficient training, optimization instability,
or evaluation noise, then Assumption E.1 does not hold.

Proof. By Proposition E.2, if Assumption E.1 holds, then the optimal shift must be identical across all latent
dimensions. If experiments repeatedly show that the optimal shift drifts systematically with d, then this
contradicts the necessary consequence of the null hypothesis. Therefore, the null hypothesis is rejected.

The boundary of the conclusion in Corollary E.3 is clear: it directly rules out the hypothesis that the
latent space is purely local and separable. Consequently, the strictly derivable conclusion is that some
cross-dimensional shared or joint structure must exist in the latent representation. If one further wishes
to interpret this structure as a global semantic factor, additional modeling assumptions and experimental
evidence are required.


                                                             55
An equivalent formulation from an information-theoretic perspective The above conclusion admits a consistent
restatement in information-theoretic terms. Let s denote a latent variable associated with high-level semantics,
and let z denote the VAE latent. Then the diffusion forward process can be written as

                                       zt = αt z + σt ϵ,              ϵ ∼ N (0, I),                       (E.5)

where αt and σt are determined by the noise schedule. If we further decompose the latent as

                                                     z = ϕ(s) + u,                                        (E.6)

where ϕ(s) denotes the effective signal induced by the semantic variable and u denotes encoding residuals or
posterior uncertainty, then the observation received by the DiT at timestep t becomes

                                            zt = αt ϕ(s) + αt u + σt ϵ .                                  (E.7)
                                                           | {z }
                                                                    total noise

Hence, what matters to the DiT is not the raw timestep itself, but rather how much effective information
about s is still preserved in the latent at that timestep.
Under Assumption E.1, if semantics is carried entirely by local independent dimensions, namely

                                    s = (s1 , . . . , sd ),         z = (z1 , . . . , zd ),

and if the channels are independent and identically distributed across dimensions, then the additivity of
mutual information yields
                                                    d
                                                    X
                                        I(s; zt ) =   I(si ; zt,i ).                                (E.8)
                                                              i=1

If each local channel has the same information curve with respect to shift, then changing d only rescales
the total mutual information linearly and does not change which shift places the effective information in
the range most suitable for semantic recovery. From this information-theoretic perspective, the essence of
Proposition E.2 is that, under purely separable representations, varying the dimension does not alter the shift
corresponding to the optimal semantic information regime.
Conversely, if the optimal shift drifts systematically with d, then the purely additive information structure
corresponding to (E.8) can no longer hold. This indicates that the effective semantic information in the latent
is not merely the sum of a set of completely homogeneous local subchannels, but is more likely to depend on
some shared structure that only emerges after cross-dimensional aggregation.

A shared-factor model and the interpretation of shift scaling laws To further explain why systematic drift
may exhibit a pattern similar to the resolution-dependent shift in Stable Diffusion, we introduce a stronger
but analytically convenient structural model. Suppose that each latent dimension can be written as

                                        zi = Ai g + ξi ,             i = 1, . . . , d,                    (E.9)

where g ∈ Rr denotes a low-dimensional shared factor, Ai is a linear observation matrix, and ξi is local
independent noise. Here, g may be understood as an abstract shared semantic state, without assuming that it
is a single scalar quantity.
In the simplest setting, assume that Ai ≡ A and that ξi ∼ N (0, σξ2 I) are independent across dimensions.
Standard results from linear Gaussian models imply that, as the latent dimension d increases, the posterior
precision for g grows linearly, while the posterior covariance shrinks as d−1 . Equivalently, the effective
signal-to-noise ratio for recovering the shared factor satisfies

                                                    SNReff (d) ∝ d.                                      (E.10)




                                                              56
From an information-theoretic viewpoint, this means that the recoverable information about the shared factor
g increases with d. More specifically, if the total effective noise variance at diffusion timestep t is denoted by
  2
σeff (t), then the mutual information about g can be written as
                                         1                                    −1        
                             I(g; zt ) = log det I + αt2 Σ1/2
                                                           g  A ⊤
                                                                    σ 2
                                                                      eff (t)I     AΣ 1/2
                                                                                      g     .               (E.11)
                                         2
Under a scalar or isotropic approximation, this further reduces to
                                                  r                            
                                       I(g; zt ) ≈ log 1 + c · d · SNReff (t) ,                             (E.12)
                                                  2
where the constant c > 0 is determined by the observation matrix and the prior covariance of the shared
factor.
If the effect of a shift δ is equivalent to translating the log-SNR of the denoising process, and if semantic
recovery performance is primarily determined by whether the effective log-SNR of the shared factor falls
within an appropriate range, then the shift required to maintain optimal semantic recovery must compensate
for the dimension dependence induced by (E.10) or (E.12). This leads to a logarithmic relation of the form
                                               δ ⋆ (d) = a log d + b,                                      (E.13)
where the sign of a depends on the specific time parameterization and the convention used to define the shift.
Remark E.4 (Structural homology with resolution-dependent shift). The significance of (E.13) is not that it
uniquely identifies a true global variable, but rather that once a shared factor is repeatedly observed across
multiple latent dimensions, the mechanism
    increasing dimension ⇒ stronger effective SNR ⇒ logarithmic compensation in the optimal shift
arises naturally. This is structurally homologous to the intuition underlying resolution-dependent timestep
shift in Stable Diffusion. The difference is that, in the present case, what is shared is a joint structure within
the latent representation, whereas in the image case the shared content comes from more globally distributed
information under higher resolution.
It should be emphasized that the main role of the shared-factor model is to explain the form of the systematic
drift, rather than to replace the rigorous contradiction argument above. The latter answers which hypotheses
must fail if drift is observed, whereas the former explains what kind of structure such drift is most consistent
with.

Relation to VAE logSNR The above analysis also explains another empirical phenomenon: even when the
latent dimension is fixed, the optimal timestep shift may still change when the logSNR of the VAE posterior
is altered. The reason is that the VAE is not a noiseless encoder. In (E.6), the term u precisely represents
posterior uncertainty. If the timestep t is fixed, then the total noise covariance with respect to the semantic
variable is
                                            Σnoise (t) = αt2 Σu + σt2 I.                                 (E.14)
When the VAE logSNR decreases, this typically implies that the posterior variance increases, that is, Σu
becomes larger. Consequently, at the same raw timestep, the effective semantic information preserved in the
latent,
                                           1                                    
                                I(s; zt ) = log det I + αt2 Σs (αt2 Σu + σt2 I)−1 ,                       (E.15)
                                           2
decreases overall. In other words, in a smoother latent space with larger posterior variance, the same diffusion
timestep corresponds to a lower effective semantic SNR. Therefore, in order for training and sampling to
remain within a similar optimal semantic information regime, the timestep shift must also be recalibrated.
This shows that changing the latent dimension and changing the VAE logSNR are, on the surface, two different
operations, but they act on the same core object from an information-theoretic perspective, namely the
effective mutual information curve I(s; zt ) of the semantic variable along diffusion time. The former mainly
changes how shared information is aggregated and amplified across dimensions, whereas the latter mainly
changes the effective noise floor at each timestep. Both can cause the optimal information regime to drift
along the raw timestep axis.


                                                        57
Why this further points to semantic structure Corollary E.3 alone only implies the existence of shared or
joint structure; it does not yet establish that this structure necessarily carries semantic information. To
strengthen the conclusion to shared structure that affects semantics, one needs task-level selective evidence. A
particularly informative situation is the following:
    1. reconstruction quality remains relatively stable across different shifts;
    2. semantic downstream metrics are highly sensitive to the shift;
    3. and the optimal shift exhibits systematic drift with the latent dimension.
If these three conditions hold simultaneously, then it is no longer plausible to explain the phenomenon merely
by saying that the shift changes the difficulty of local token realization. A more reasonable interpretation
is that the shift is adjusting the noise calibration of the latent prior with respect to some shared high-level
structure, and that this shared structure is precisely involved in semantic organization.
From an information-theoretic standpoint, this means that changing the shift does not substantially alter the
decoder’s ability to recover local details, but does substantially change the preservation regime of the shared
information required by high-level tasks during the diffusion process. Therefore, what is being recalibrated is
more likely not local reconstruction error, but the semantic component of mutual information at a higher level.
In other words, if reconstruction remains consistently strong while semantic metrics vary substantially, then
the main quantity being affected is not the decoder’s local realization capability, but rather the alignment of
the latent prior with high-level shared information. Combined with the contradiction result above, this yields
a stronger yet still robust conclusion: the latent space contains shared or joint structures that affect semantics.
Remark E.5 (Conclusions that are rigorously supported in this section). Under a broad class of purely local
and separable latent hypotheses, the optimal timestep shift should not drift systematically with the latent
dimension. The stable drift observed in experiments refutes this null hypothesis and therefore implies the
existence of cross-dimensional shared or joint structures in the latent space. Furthermore, if this drift is
manifested primarily in semantic metrics rather than reconstruction metrics, then one can further conclude
that such shared or joint structures indeed participate in high-level semantic organization.
These conclusions are sufficient to support the central hypothesis of this work: if such global or semi-global
semantic shared structures do exist in the latent space, then calibrating the noise schedule at the level of the
latent prior may offer a more substantial advantage than methods based purely on local modeling.

F    Theoretical Analysis of Likelihood Estimation Bias
This section provides a more complete theoretical analysis of the phenomenon discussed in Section 5.1 of
the main text. The central question is why Cola DLM can already generate reasonably high-quality text in
free-generation settings, while its PPL / conditional score estimated via the ELBO or a plug-in conditional
estimator remains substantially inflated. We argue that this is not an incidental engineering artifact, but
rather a structural consequence of the misalignment among the training objective, the underlying probabilistic
object, and the local geometry in continuous latent generative models.
To avoid conflating the conditional marginal probability of an entire response with the local prior density
evaluated at a particular latent point, we explicitly distinguish the following two objects throughout. Let

                                                 x = (xpre , xres ),

where xpre denotes the prefix / context and xres denotes the response, and let c denote the conditional
information induced by the prefix. Then the exact conditional marginal probability is
                                             Z
                                 p(x | c) = pθ (xres | z, c) pψ (z | c) dz,
                                    res
                                                                                                  (F.1)

whereas the local conditional score corresponding to the response-level score in the main text is
                                          h                                                         i
                   Sresp (x) = Eqϕ (z|x,c) log pθ (xres | z, c) + log pψ (z | c) − log qϕ (z | x, c) .       (F.2)


                                                         58
Equation (F.1) is the exact conditional marginal likelihood, whereas Equation (F.2) is a local estimator that
is more directly accessible in analysis. The gap between them is precisely the probabilistic foundation of the
discussion that follows.

F.1     Why the Flow Matching Training Objective Is Naturally Misaligned with Conditional
        PPL
F.1.1   Flow Matching optimizes vector-field regression rather than gold-specific local log-density
With the encoder and decoder fixed, the prior-learning subproblem can be understood as fitting pψ (z | c) to
the conditional latent distribution induced jointly by the data and the posterior. In practice, however, Flow
Matching does not directly maximize the conditional log-density at a particular gold latent point. Instead, it
learns a time-dependent vector field vψ (z, t, c). Its typical objective can be written as
                                                     h                    i
                                      LFM (ψ) = E ∥vψ (Zt , t, c) − U ⋆ ∥2 ,                             (F.3)

where U ⋆ is the target velocity field induced by a bridging path or a reference flow. Equation (F.3) shows
that Flow Matching directly regresses a dynamical system, rather than the local conditional log-density at a
specific gold latent point.
More importantly, the pointwise optimum of the squared loss is not the most likely velocity field in a
mode-seeking sense, but the conditional mean velocity field.
Proposition F.1 (The pointwise optimum of Flow Matching is the conditional mean velocity). For fixed
(z, t, c), define
                            Y := U ⋆ | (Zt = z, t, c), f := vψ (z, t, c).
Then the conditional risk
                                  R(f ; z, t, c) := E ∥f − Y ∥2 | Zt = z, t, c                           (F.4)
                                                                             

admits the unique minimizer
                                        f ⋆ (z, t, c) = E[Y | Zt = z, t, c].                             (F.5)

Proof. For any f , we have

                         f − Y = (f − E[Y | Zt = z, t, c]) − (Y − E[Y | Zt = z, t, c]).

Expanding the square, taking the conditional expectation, and using the fact that the cross term vanishes, we
obtain
           R(f ; z, t, c) = ∥f − E[Y | Zt = z, t, c]∥2 + E ∥Y − E[Y | Zt = z, t, c]∥2 | Zt = z, t, c .
                                                                                                   

The second term is independent of f , so the unique optimum is attained at

                                            f = E[Y | Zt = z, t, c].



The meaning of Proposition F.1 is direct: Flow Matching learns the conditional mean dynamics, rather than
a gold-specific local calibration specialized to the posterior tube of a particular response. As long as the
conditional response distribution under a given context is itself multimodal or broad-peaked, Flow Matching
is more naturally biased toward learning an average transport map that moves total probability mass into
a reasonable region, rather than automatically guaranteeing local density calibration around a particular
sample.




                                                        59
F.1.2   Why the prior mean may be far from the ground truth while generation remains good
This point can be formalized under multimodal conditional distributions. For a fixed context c, suppose
                                           M
                                           X                                    M
                                                                                X
                            pψ (z | c) =         πm (c) N (µm , Σm ),                 πm (c) = 1,          (F.6)
                                           m=1                                  m=1

and for a particular sample, its posterior is mainly concentrated on the m⋆ -th mode, namely

                                             qϕ (z | x, c) ≈ N (µm⋆ , Σq ).                                (F.7)

Then the global mean of the prior is
                                                             M
                                                             X
                                                 µ̄p (c) =         πm (c)µm .                              (F.8)
                                                             m=1


Corollary F.2 (Mean displacement under multimodal conditional distributions). Under the setting of
Equations (F.6)–(F.8),                         X
                            ∥µ̄p (c) − µm⋆ ∥ ≤   πm (c) ∥µm − µm⋆ ∥.                          (F.9)
                                                         m̸=m⋆

If the modes are well separated, then the prior mean may deviate substantially from the latent of a particular
ground-truth sample. Nevertheless, as long as the major mass of these modes lies within a decoder-good region,
free-generation quality can still remain good.

Proof. By
                                         M
                                         X                                 X
                       µ̄p (c) − µm⋆ =         πm (c)(µm − µm⋆ ) =                πm (c)(µm − µm⋆ )
                                         m=1                              m̸=m⋆

and the triangle inequality, the result follows immediately.

Therefore, a large distance between the prior mean and the ground-truth latent does not automatically imply
that the prior has failed to learn the distribution. A more plausible explanation is often that the context-level
conditional distribution is itself multimodal, while the posterior selects only one narrower gold-specific mode
among them.

F.2     Why This Further Leads to Good Generation but Poor PPL
F.2.1   Generation quality and conditional PPL correspond to two different geometric objects
Fix a context c, let r(xres ; c) denote some external generation-quality function, and let τ be a threshold.
Define the decoder-good region by

                                 Aτgood (c) := z : Exres ∼pθ (·|z,c) r(xres ; c) ≥ τ ,                (F.10)
                                                                              


and for a particular sample x and a constant ρ > 0, define its posterior typical set by

                                            Aρx := {z : qϕ (z | x, c) ≥ ρ} .                              (F.11)

If Aρx is narrow and sharp, we may refer to it as the gold tube of that sample. In general, Aρx is only a more
local and sharper subset inside Aτgood (c), namely

               Aρx ⊆ Aτgood (c)    or at least         Aρx is a more local subregion inside Aτgood (c).   (F.12)

This distinction shows that good generation and good conditional PPL are not the same requirement.




                                                              60
Proposition F.3 (Good generation and poor conditional score can hold simultaneously). Fix a context c.
Suppose there exist constants α, ∆ > 0 and thresholds τ, ρ such that
                                          pψ Aτgood (c) | c ≥ α,                               (F.13)
                                                           

                                                    qϕ (Aρx | x, c) ≈ 1,                                   (F.14)
                                 log qϕ (z | x, c) − log pψ (z | c) ≥ ∆,        ∀z ∈ Aρx ,                 (F.15)
and the decoder reconstruction term is uniformly bounded on Aρx :
                                        log pθ (xres | z, c) ≤ B,       ∀z ∈ Aρx .                         (F.16)
Then:
  1. under free sampling from the prior, at least an α fraction of the mass falls into the decoder-good region,
     so generation quality may already be good;
  2. but the conditional score satisfies
                                                  Sresp (x) ≤ B − ∆ + o(1),                                (F.17)
        and therefore the conditional ELBO or plug-in PPL may still be substantially biased whenever ∆ is
        sufficiently large.

Proof. The first claim follows directly from Equation (F.13). For the second claim, rewrite Equation (F.2) as
                                     h                                                        i
                     Sresp (x) = Eqϕ log pθ (xres | z, c) − log qϕ (z | x, c) − log pψ (z | c) .

By Equation (F.14), the mass of qϕ outside Aρx is o(1). Combining this with Equations (F.15) and (F.16)
yields
                                       Sresp (x) ≤ B − ∆ + o(1).



The essence of Proposition F.3 is that generation quality mainly requires the prior mass to enter Aτgood (c),
whereas conditional PPL additionally requires the prior to assign sufficiently high local density on the posterior
tube of a particular gold response. The former is a coverage requirement, whereas the latter is a calibration
requirement.

F.2.2    Why good reconstruction does not imply good PPL
Equation (F.2) can also be written as
                      Sresp (x) = Eqϕ (z|x,c) [log pθ (xres | z, c)] − KL qϕ (z | x, c) ∥ pψ (z | c) .     (F.18)
                                                                                                    
                                  |               {z               } |             {z               }
                                               R(x;c)                             G(x;c)


Proposition F.4 (Good reconstruction does not imply good conditional PPL). If there exists a sequence of
samples {xn } such that
                                         R(xn ; cn ) → Rmax ,
while at the same time
                             G(xn ; cn ) → +∞           or remains significantly positive,
then the corresponding conditional score may still be poor.

Proof. Equation (F.18) directly gives
                                         Sresp (xn ) = R(xn ; cn ) − G(xn ; cn ).
Therefore, as long as the posterior–prior gap remains large, the conditional score can still be low even when
reconstruction is already strong.


                                                            61
This shows that, in continuous latent language models, good reconstruction near the ground truth is not
sufficient to imply good PPL. What truly determines PPL also includes the degree of local matching between
the posterior and the prior.

F.3     Even If the Prior Mean Is Close to the Ground Truth, PPL May Still Be Poor
The previous subsection clarified the distinction between large-region coverage and gold-tube calibration.
However, even if the problem is further restricted to the case where the center of the prior is already
approximately aligned with the target, PPL may still remain poor. The reason is that local covariance,
orientation, curvature, and volume terms also enter the conditional score.

F.3.1   Under a local Gaussian approximation, center alignment is necessary but not sufficient
Suppose that, in a neighborhood of a gold sample,

                             qϕ (z | x, c) ≈ N (µq , Σq ),        pψ (z | c) ≈ N (µp , Σp ).             (F.19)

Then the corresponding KL divergence is
                                  1h                                                 det Σp i
                  KL(qϕ ∥pψ ) =      tr(Σ−1                 ⊤ −1
                                         p Σq ) + (µp − µq ) Σp (µp − µq ) − d + log         .           (F.20)
                                  2                                                  det Σq

Proposition F.5 (Center alignment alone is insufficient to guarantee good PPL). Under Equation (F.19),
even if µp ≈ µq , as long as Σp and Σq are mismatched in scale, orientation, or volume, KL(qϕ ∥pψ ) may still
be substantially large, and hence the conditional score may still be poor.

Proof. Equation (F.20) shows that, beyond the center-discrepancy term, there are also the explicit scale- and
volume-related terms
                                                             det Σp
                                          p Σq ) and
                                      tr(Σ−1             log        .
                                                             det Σq
Even if the center-discrepancy term is small, the KL divergence can remain significantly positive whenever
these latter terms are mismatched.

Therefore, in continuous density models, it is incorrect to claim that it suffices to move the prior mean close
to the ground-truth latent. PPL requires a stronger form of local probabilistic calibration: not only must the
centers be close, but the local covariance, local volume, and local orientation must also be matched.

F.3.2    An overly sharp posterior further amplifies this issue
If the posterior is highly concentrated, so that Σq is small, then the scale and volume terms in Equation (F.20)
become even more sensitive. Intuitively, the gold posterior is no longer a thick region, but rather resembles a
very thin tube. In this regime, even a slight deviation of the prior can induce a substantially amplified local
log-density gap. Therefore,

                 good generation =⇒ the prior hits a relatively large decoder-good region,               (F.21)

but
            good PPL =⇒ the prior must additionally hit a much narrower gold posterior tube.             (F.22)
This also explains why, in continuous latent language models, PPL behaves more like a local density calibration
metric rather than merely a generation quality metric.




                                                             62
F.4     Why PPL and Generation Are Usually More Strongly Correlated in Discrete Models
        Such as AR / LLaDA
F.4.1   AR: training, evaluation, and generation share the same discrete probabilistic object
For autoregressive models, the conditional distribution admits the exact chain decomposition
                                                         N
                                                         Xres

                               log pAR
                                    η (x
                                        res
                                            | xpre ) =          log pη (xres
                                                                         i |x
                                                                             pre
                                                                                 , xres
                                                                                    <i ),                (F.23)
                                                         i=1

and therefore the PPL is
                                                  Nres
                                                                             !
                                              1 X
                           PPLAR (x) = exp −                    res  pre res
                                                       log pη (xi | x , x<i ) .                          (F.24)
                                             Nres i=1
The next-token cross-entropy minimized during training is precisely the negative log of the terms in Equa-
tion (F.24), and generation also directly queries these same conditional probabilities. Hence, in AR models,
         training objective = the object evaluated by PPL = the object used during generation.           (F.25)
This is why PPL and generation quality are usually most strongly correlated in autoregressive models.

F.4.2   Discrete models optimize the probability mass of the correct event more directly
For a discrete classifier, if the gold token at a given position is y and the model outputs a probability vector
π ∈ ∆V −1 , then the cross-entropy is
                                                  LCE = − log πy .                                        (F.26)
What is directly optimized here is the probability mass assigned to the correct event itself. In contrast, Cola
DLM requires the computation of an integral over the continuous latent space:
                                       Z                                        
                          p(xres | c) = exp log pθ (xres | z, c) + log pψ (z | c) dz.                    (F.27)

Accordingly, its value depends not only on whether the mode center is correctly located, but also explicitly on
local volume and curvature. If one performs a Laplace approximation of
                                  Fx,c (z) := log pθ (xres | z, c) + log pψ (z | c)
at its maximizer, then
                                                                d          1
                           log p(xres | c) ≈ Fx,c (zx,c
                                                    ⋆
                                                        )+        log(2π) − log det Hx,c ,               (F.28)
                                                                2          2
where
                                            Hx,c := −∇2z Fx,c (zx,c
                                                                ⋆
                                                                    ).
This shows that conditional likelihood in continuous latent models depends not only on mode height, but also
on local geometry.

Therefore, the more accurate statement is not merely that discrete models are inherently simpler, but rather
that training, evaluation, and generation in discrete models all act more directly on the same discrete
probabilistic object, whereas continuous latent models are additionally mediated by latent-space integration,
posterior--prior matching, and decoder-emission compatibility.

F.4.3   LLaDA: the main discrepancy arises at the discrete observation level rather than the
        latent calibration level
LLaDA is not identical to AR, but compared with Cola DLM, its recovery object is still the discrete observed
state itself, rather than first encoding the sample-dependent posterior into a continuous latent and then
requiring an independent conditional prior to match it pointwise. Therefore, even though LLaDA may also
exhibit a surrogate gap, its main discrepancy is still concentrated within the discrete token space, rather than
in the posterior–prior density collision characteristic of Cola DLM. For this reason, the correlation between
PPL and generation is usually still stronger in LLaDA than in continuous latent models.


                                                          63
G     A Unified Theoretical View of Noise Schedule, logSNR, and Flow Matching
This section consolidates the preceding discussion of noise schedule, Flow Matching, latent-dimension-dependent
shift drift, and semantic information calibration into a single theoretical chain. The key question is not why a
particular empirical hyperparameter works, but rather: what mathematical object the noise schedule controls
in Cola DLM, and how this object enters the training objective and ultimately affects semantic modeling
quality. More specifically, we establish the following four points:
    1. the noise schedule and the logSNR curve are two equivalent parameterizations of the same forward
       noising process;
    2. in Flow Matching, the effect of logSNR on the loss arises both through the change of sampling measure
       inside the expectation and through the scale change of the supervised target velocity field;
    3. uniform-timestep training is generally not equivalent to uniform-logSNR training;
    4. the essential role of the optimal timestep shift is not to translate an abstract time label, but to recalibrate
       training and sampling toward a more appropriate effective semantic information regime.

G.1    Noise Schedule and logSNR as Equivalent Parameterizations
We begin with the standard forward process used throughout the main text and appendix:

                                          zt = αt z + σt ϵ,        ϵ ∼ N (0, I),                                (G.1)

where αt and σt are determined by the noise schedule. The signal-to-noise ratio is naturally defined by

                                             αt2                                   αt2
                                 SNR(t) =        ,      λ(t) := log SNR(t) = log       .                        (G.2)
                                             σt2                                   σt2

Under the standard normalization
                                                       αt2 + σt2 = 1,                                           (G.3)
one has
                                  αt2 = sigmoid(λ(t)),           σt2 = sigmoid(−λ(t)).                          (G.4)
Therefore, specifying λ(t) uniquely determines (αt , σt ), and conversely (αt , σt ) uniquely determines λ(t). In
this precise sense,
                                     noise schedule ⇐⇒ logSNR curve                                        (G.5)
so the two are not related through a secondary transformation of the loss; they are simply two equivalent
parameterizations of the same forward noising geometry.
Now consider a shifted logSNR curve
                                                     λδ (t) = λ(t) + δ.                                         (G.6)
The corresponding signal and noise coefficients become
                              2
                             αt,δ = sigmoid(λ(t) + δ),            2
                                                                 σt,δ = sigmoid(−λ(t) − δ).                     (G.7)

Hence the role of timestep shift is not merely to translate the raw timestep axis, but rather to remap the
same raw timestep to a different logSNR regime:

                      the same raw timestep t is reassigned to a different logSNR interval.                     (G.8)




                                                            64
G.2    Reparameterization of the Flow Matching Objective
As discussed above, the strict probabilistic objective of Cola DLM is always ELBO maximization together
with prior matching, while Flow Matching is the practical mechanism used to learn the same prior family.
To understand how the noise schedule enters training, it is therefore sufficient to analyze the mean-squared
vector-field regression objective.
To avoid a notation clash with the coefficients (αt , σt ) in Equation (G.1), let the interpolation coefficient of
the bridge path be denoted by β(t). Consider

                                                                                 dzt
                         zt = (1 − β(t))z0 + β(t)z1 ,          ut (z0 , z1 ) =       = β̇(t)(z1 − z0 ),     (G.9)
                                                                                 dt
and write the single-block Flow Matching objective as
                                         h                               i
                          LFM = Et,z0 ,z1 ∥vψ (zt , t) − ut (z0 , z1 )∥22 ,           t ∼ U [0, 1].        (G.10)

Equivalently,                     Z 1                                h                               i
                          LFM =         ℓ(t) dt,      ℓ(t) := Ez0 ,z1 ∥vψ (zt , t) − ut (z0 , z1 )∥22 .    (G.11)
                                    0

Now let
                                                   λ = Λ(t) := log SNR(t),                                 (G.12)
and assume that Λ is strictly monotone and differentiable, so that the inverse mapping t = T (λ) = Λ−1 (λ)
exists. By the change-of-variables formula,
                                                    dt
                                             dt =       dλ,                                        (G.13)
                                                    dλ
which yields                                             Z
                                                                         dt
                                               LFM =         ℓ(T (λ))       dλ.                            (G.14)
                                                                         dλ
Therefore, if training samples t uniformly, then the induced density on the logSNR axis is

                                                                   dt   dt
                                             pλ (λ) = pt (T (λ))      =    ,                               (G.15)
                                                                   dλ   dλ

where we used pt (t) ≡ 1.
Proposition G.1 (Uniform timestep is generally not equivalent to uniform logSNR). Let λ = Λ(t) be a
strictly monotone differentiable function, and suppose training uses t ∼ U [0, 1]. Then the sampling density on
the logSNR axis is
                                                           dt
                                                 pλ (λ) =     .                                         (G.16)
                                                          dλ
Therefore, unless Λ(t) is affine, uniform-timestep training is not equivalent to uniform-logSNR training.

Proof. By Equation (G.14), the pushforward of the uniform measure on t to the λ axis has density

                                                                         dt
                                                   pλ (λ) = pt (T (λ))      .
                                                                         dλ

Since pt (t) ≡ 1, Equation (G.16) follows immediately. The density pλ (λ) is constant if and only if dλ     dt is
constant, namely if and only if Λ(t) is affine. Hence the two training schemes are not equivalent in general.




                                                              65
The above proposition shows only that the sampling measure inside the expectation changes after reparame-
terization by logSNR. This is not yet the whole story. The supervised target velocity field itself also rescales
under the new time coordinate. Let
                                                                         dzλ
                                          zλ := zT (λ) ,         uλ :=       .                           (G.17)
                                                                         dλ
Then, by the chain rule,
                                     dzt    dzλ dλ                            dλ
                                ut =      =    ·   = λ̇(t) uλ ,     λ̇(t) :=     .                       (G.18)
                                      dt     dλ dt                            dt
Substituting this identity into the loss gives
                                   Z        h                               i dt
                           LFM = Ez0 ,z1 ∥vψ (zλ , T (λ)) − λ̇(T (λ))uλ ∥22        dλ.                   (G.19)
                                                                              dλ

Hence logSNR affects the Flow Matching loss in two distinct ways:
   1. it changes the sampling weight assigned to different noise regimes inside the expectation;
   2. it rescales the numerical magnitude of the supervised target velocity field.
If we further define the λ-time vector field by
                                                            1
                                        ṽψ (z, λ) :=             vψ (z, T (λ)),                         (G.20)
                                                        λ̇(T (λ))

then Equation (G.19) can be rewritten as
                                     Z             h                      i
                              LFM = weff (λ)Ez0 ,z1 ∥ṽψ (zλ , λ) − uλ ∥22 dλ,                           (G.21)

where the effective weight is
                                                                 dλ
                                                  weff (λ) =        .                                    (G.22)
                                                                 dt
This yields the following summary.
Remark G.2 (How logSNR enters the Flow Matching loss). In the Flow Matching training of Cola DLM, the
effect of logSNR is not merely that one samples different time indices. It changes the training geometry jointly
through
   1. the induced sampling measure on the logSNR axis, pλ (λ) =              dλ ;
                                                                             dt


   2. the time-scale rescaling of the supervised target, ut = λ̇(t)uλ .
Therefore, the noise schedule is not an external hyperparameter detached from the loss. It directly determines
which logSNR regimes are emphasized during training and how difficult the regression task is within those
regimes.

G.3   Equivalent Derivation Under the Signal--Noise Parameterization
For consistency with Equation (G.1), it is useful to derive the same scaling relation directly from the
signal–noise coefficients (αt , σt ). By Equation (G.4),

                                αt2 = sigmoid(λ(t)),            σt2 = sigmoid(−λ(t)).

Differentiating αt2 with respect to t yields

                                                d 2
                                                  α = αt2 σt2 λ̇(t),                                     (G.23)
                                                dt t



                                                           66
and therefore
                                                           1
                                                   α̇t =     αt σt2 λ̇(t).                             (G.24)
                                                           2
Similarly,
                                                         1
                                                  σ̇t = − σt αt2 λ̇(t).                                (G.25)
                                                         2
Differentiating Equation (G.1) then gives
                                      dzt                  1                        
                              ut :=       = α̇t z + σ̇t ϵ = λ̇(t) αt σt2 z − σt αt2 ϵ .                (G.26)
                                      dt                   2
If we define
                                                  dzt   1                    
                                        uλ :=         =    αt σt2 z − σt αt2 ϵ ,                       (G.27)
                                                  dλ    2
then we again obtain
                                                     ut = λ̇(t) uλ .                                   (G.28)
Thus, the same conclusion holds whether one starts from an abstract bridge-path parameterization or directly
from the signal–noise parameterization of diffusion.

G.4    The Optimal Schedule as Calibration of the Effective Semantic Information Regime
Showing that the schedule affects the loss is not yet sufficient. The more important question is: what
meaningful object in the latent space is ultimately being calibrated?
Following the notation in Appendix E, let s denote a high-level semantic variable, and decompose the latent
representation as
                                               z = ϕ(s) + u,                                        (G.29)
where ϕ(s) is the semantic signal and u represents posterior uncertainty or encoding residuals. Then the noisy
latent at diffusion time t is
                                           zt = αt ϕ(s) + αt u + σt ϵ.                                 (G.30)
From the viewpoint of the semantic variable s, the total noise covariance is therefore

                                             Σnoise (t) = αt2 Σu + σt2 I,                              (G.31)

where Σu is the covariance of the posterior uncertainty term. If Σs denotes the covariance of the semantic
signal ϕ(s), then the effective information about s preserved in the latent at time t can be written as
                                            1                                    
                                 I(s; zt ) = log det I + αt2 Σs (αt2 Σu + σt2 I)−1 .                    (G.32)
                                            2
This makes clear that what matters to the DiT is not the raw timestep itself, but how much effective semantic
information remains in the latent at that timestep. In this sense,

                           the schedule effectively controls the curve t 7→ I(s; zt ).                 (G.33)

Accordingly, the role of timestep shift should be interpreted not as the tuning of an isolated numerical
hyperparameter, but as the recalibration of training and sampling toward a different effective semantic-
information regime.
More abstractly, if the shifted logSNR curve is given by λδ (t) = λ(t) + δ, then the optimal shift can be
expressed as                                                                  
                                 δ ⋆ = arg max Perf Ieff,δ (t; d, Σu , G, B, ϑ) ,                   (G.34)
                                              δ

where d denotes the latent dimension, Σu the posterior uncertainty, G the latent geometry or semantic
smoothness, B the block size, and ϑ the collection of model parameters. Thus, the problem of selecting the
optimal noise schedule is best understood as an effective semantic information calibration problem rather
than an isolated numerical tuning problem.


                                                            67
G.5   Why Latent Dimension and VAE logSNR Shift the Optimum
The viewpoint developed above is consistent with Proposition E.2 and Corollary E.3. If the latent represen-
tation were purely local, separable, and homogeneous across dimensions, then the optimal shift would not
systematically drift with dimension. When a stable drift is observed empirically, the natural interpretation is
that latent dimension and VAE posterior logSNR both modify the effective semantic-information curve in
Equation (G.32).

Latent dimension.    Under the shared-factor model

                                        zi = Ai g + ξi ,        i = 1, . . . , d,                          (G.35)

if different latent dimensions repeatedly observe the same low-dimensional shared factor g, then standard
linear-Gaussian inference implies that the effective recovery signal-to-noise ratio scales as

                                                 SNReff (d) ∝ d.                                           (G.36)

Consequently, the recoverable information about the shared factor satisfies the approximation
                                              r                         
                                   I(g; zt ) ≈ log 1 + c · d · SNReff (t) .                                (G.37)
                                              2
This means that increasing the latent dimension can strengthen the recoverable information about shared
semantic structure. In order to keep training in a comparable semantic-recovery regime, the timestep shift
must compensate for this dimensional effect, which naturally leads to a logarithmic law of the form

                                               δ ⋆ (d) = a log d + b.                                      (G.38)

VAE posterior logSNR. On the other hand, lowering the posterior logSNR of the VAE typically corresponds
to increasing posterior variance, that is, enlarging Σu . By Equation (G.32), for fixed t, increasing

                                                   αt2 Σu + σt2 I

reduces the effective semantic mutual information. Therefore, the same raw timestep corresponds to a lower
effective semantic SNR in a smoother latent space with larger posterior variance. To align training and
sampling with a comparable semantic-information regime, the optimal shift must change accordingly.
This shows that changing the latent dimension and changing the VAE posterior logSNR are, on the surface,
two different operations, but from an information-theoretic perspective they act on the same core object: the
effective semantic-information curve I(s; zt ) along diffusion time. The former mainly changes how shared
information is aggregated and amplified across dimensions, whereas the latter mainly changes the effective
noise floor at each timestep.
Remark G.3 (Role of block size). Compared with latent dimension and posterior logSNR, the effect of block
size is not yet characterized by a closed-form law such as Equation (G.38). Nevertheless, block size changes
the granularity of within-block semantic aggregation, the strength of cross-block conditional dependence,
and the intrinsic modeling difficulty within the same noise regime. It therefore interacts with the schedule
indirectly through the shape of the effective semantic-information curve. The more precise statement is not
that block size directly determines the schedule, but that block size and schedule are coupled through the
effective geometry of the block-causal prior.
Remark G.4 (Summary). In Cola DLM, the noise schedule is not an empirical hyperparameter external to
the training objective. It first defines the forward noising geometry through (αt , σt ), or equivalently through
the logSNR curve; it then enters the Flow Matching objective jointly through the sampling measure and the
time-scale of the supervised target velocity field; and, at the semantic level, it calibrates not the raw timestep
itself but the effective mutual-information regime of high-level latent semantics. Consequently, changes in
latent dimension, posterior uncertainty, and latent geometry can all induce systematic drift in the optimal
timestep shift by altering the same effective semantic-information curve.


                                                           68
H     Experiments Details
H.1   Dataset Introduction
LAMBADA. LAMBADA [74] is a long-context word prediction benchmark designed to evaluate whether a
model can use broad discourse context for text continuation. The target word is usually difficult to infer from
local context alone and instead requires understanding information distributed across the preceding passage.
In our experiments, LAMBADA is used as a representative continuation-style benchmark to evaluate global
semantic modeling and long-range contextual coherence.

MMLU. MMLU [33] is a multitask multiple-choice benchmark covering a wide range of subjects, including
humanities, social sciences, STEM, and professional domains. It is widely used to assess broad factual
knowledge and reasoning ability. In our experiments, MMLU serves as a representative knowledge-intensive
benchmark for evaluating the quality of generated answers under multiple-choice settings.

SIQA. SIQA [82] (Social IQa) is a multiple-choice commonsense reasoning benchmark focused on social
situations, intentions, and reactions. Each example requires selecting the most plausible answer given a short
social context. We use SIQA to evaluate whether the model can capture socially grounded semantics and
perform coherent commonsense inference.

SQuAD. SQuAD [78] is a reading comprehension benchmark in which the model answers questions based on
a given passage. In our evaluation, it is used in a generative setting to assess whether the model can produce
accurate answers conditioned on contextual evidence. This benchmark complements multiple-choice tasks by
testing open-form text generation ability.

Story Cloze. Story Cloze [68] is a story understanding benchmark that asks the model to choose the most
plausible ending for a four-sentence story context. It mainly evaluates narrative coherence, causal reasoning,
and commonsense understanding. We include it as a benchmark for discourse-level semantic consistency.

OBQA. OBQA [64] (OpenBookQA) is a multiple-choice question answering benchmark designed to test the
combination of core science facts with broader commonsense knowledge. Many questions require multi-hop
reasoning beyond direct fact recall. In our experiments, it is used to assess knowledge integration and reasoning
under a constrained answer space.

RACE. RACE [48] is a large-scale multiple-choice reading comprehension benchmark collected from English
examinations for middle and high school students. The questions often require passage understanding,
inference, and reasoning rather than simple span extraction. We use RACE to evaluate more challenging
reading comprehension and answer selection ability.

HellaSwag. HellaSwag [107] is a multiple-choice benchmark for grounded commonsense reasoning and
sentence continuation. The task is to select the most plausible continuation among several adversarially
constructed candidates. It is widely regarded as a strong test of commonsense and semantic plausibility, and
is included here to evaluate the model’s ability to distinguish coherent continuations from distractors.

Usage in This Work. For the internal component analysis in Sections 4.2, 4.3, and 4.4, we conduct experiments
on randomly sampled subsets from the test sets of LAMBADA, MMLU, and SIQA for efficiency and controlled
comparison. For the broader external comparison in Section 4.5, we additionally evaluate on the test sets of
SQuAD, Story Cloze, OBQA, RACE, and HellaSwag to provide a more comprehensive assessment across
continuation, reading comprehension, knowledge-intensive reasoning, and commonsense reasoning tasks.




                                                       69
H.2   Baseline Details
For the internal component analysis in Sections 4.2, 4.3, and 4.4, the comparisons are conducted among
different configurations of Cola DLM itself, so no external baseline is introduced in these sections. External
baselines are mainly used in Section 4.5 for scaling comparisons.

Autoregressive Baseline. We use a standard autoregressive language model as the primary reference baseline.
The model is implemented based on the official LLaMA framework [92] and is trained from scratch with
random initialization. It adopts the conventional next-token prediction objective and serves as a representative
strong decoder-only autoregressive paradigm. To ensure a fair comparison, the autoregressive baseline is
trained under settings that are strictly matched to Cola DLM in terms of data source, tokenizer, training
budget, model scale, and optimization recipe wherever applicable.

Discrete Diffusion Baseline (LLaDA). As the discrete diffusion baseline, we use LLaDA [70], implemented
with its official codebase and also trained from scratch with random initialization. This baseline represents
the discrete diffusion language modeling paradigm, in which generation is performed by iterative denoising
in discrete token space. In the scaling experiments, LLaDA is trained under the same overall experimental
protocol as the autoregressive baseline and Cola DLM, including matched training data, tokenizer, compute
budget, and model size, so that the comparison focuses on the difference in modeling paradigm rather than
differences in training resources or implementation details.

Fairness of Comparison. For all external comparisons in Section 4.5, the autoregressive baseline, LLaDA,
and Cola DLM are trained independently under strictly matched settings. In particular, all models are trained
from scratch rather than initialized from pretrained checkpoints, and we keep the training corpus, parameter
scale, and compute budget aligned as closely as possible. Therefore, the reported differences mainly reflect the
properties of the underlying modeling frameworks, rather than advantages introduced by data, initialization,
or training recipes.

H.3   Metrics and Evaluation
To ensure the fairest possible comparison across model families, all models are evaluated under a unified
few-shot generative protocol. As discussed in Section 5.1, we do not use perplexity as the primary metric for
the main comparison, since likelihood-oriented estimates can exhibit a substantial mismatch with the actual
generation quality of Cola DLM. Instead, all models are prompted to directly generate task answers under
fixed templates, and performance is computed after unified post-processing and answer matching.
Across all tasks, we use manually written in-context demonstrations that are shared across model families and
do not overlap with the evaluation set. Unless otherwise noted, the templates adopt 2-shot prompting. For
multiple-choice tasks, the options are always presented with uppercase markers such as (A), (B), (C), and (D),
with the number of options adjusted dynamically for each benchmark. Importantly, the model is always asked
to generate the option text itself, rather than the option label, so that all tasks follow the same generative
evaluation protocol. Each prompt ends with a task-specific anchor such as Answer: or End:, without adding
an extra trailing newline, so that the model directly continues from the same completion point.
For all multiple-choice benchmarks, the generated output is truncated at the first newline and normalized
before scoring. We then compare the normalized prediction against all candidate option texts and map it to
the best-matching candidate. A prediction is counted as correct if and only if the selected candidate matches
the gold answer. For continuation-style generation such as LAMBADA, the model directly continues the
given prefix and the first generated word is used as the prediction. For SQuAD, we follow the same unified
generative protocol and evaluate the generated short answer after standard normalization. Unless otherwise
stated, the templates, demonstrations, truncation rules, and answer-matching rules are kept identical for AR,
LLaDA, and Cola DLM; the only difference is the native generation procedure used by each model.

Summary of prompt formats.      Table 7 summarizes the prompt format used for each benchmark.



                                                      70
         Task                       Format                   # Shots   # Options   Generation Anchor
         MMLU                     multiple choice               2          4             Answer:
         RACE             reading comprehension MCQ             2          4             Answer:
         Story Cloze          story ending selection            2          2               End:
         LAMBADA       continuation / last-word prediction      0          –       direct continuation
         OBQA                     multiple choice               2          4             Answer:
         HellaSwag       situational continuation MCQ           2          4             Answer:
         SIQA               social commonsense MCQ              2          3             Answer:
         SQuAD              extractive QA generation            1          –             Answer:

Table 7 Unified prompt formats used in evaluation. All models are evaluated with the same prompt templates and
answer-matching rules.


MMLU. MMLU is evaluated as a 4-way multiple-choice generation task. The prompt contains two in-context
examples, followed by the test question and its four candidate options. The model is required to generate the
text of the correct option.
  MMLU prompt template

  Question: Which gas do plants absorb from the air during photosynthesis?
  (A) Oxygen
  (B) Carbon dioxide
  (C) Nitrogen
  (D) Hydrogen
  Answer: Carbon dioxide

  Question: A triangle has angles 50 degrees and 60 degrees. What is the third angle?
  (A) 60 degrees
  (B) 70 degrees
  (C) 80 degrees
  (D) 90 degrees
  Answer: 70 degrees

  Question: {question}
  (A) {choices[0]}
  (B) {choices[1]}
  (C) {choices[2]}
  (D) {choices[3]}
  Answer:



RACE. RACE is also evaluated as a 4-way multiple-choice generation task. Here, the dataset field already
contains the article together with the question, so the full input is placed after Article: and the candidate
options are listed below it.
  RACE prompt template

  Read the following article and answer the question.

  Article: Mary went to the store to buy some fruits. She bought five apples and two oranges. She
       paid 5 dollars in total. What did Mary buy?
  Options:
  (A) Bananas
  (B) Apples and oranges



                                                       71
  (C) Grapes
  (D) Watermelon
  Answer: Apples and oranges

  Article: {question}
  Options:
  (A) {choices[0]}
  (B) {choices[1]}
  (C) {choices[2]}
  (D) {choices[3]}
  Answer:



Story Cloze. Story Cloze is formatted as a story-completion benchmark with two candidate endings. To
better match the semantics of narrative continuation, the prompt uses End: instead of Answer: as the
completion anchor.
  Story Cloze prompt template

  Story: I wanted to make an omelet. I cracked two eggs into a bowl and whisked them. Then I
      poured them into a hot pan.
  (A) I ate a delicious omelet for breakfast.
  (B) I decided to order a pizza instead.
  End: I ate a delicious omelet for breakfast.

  Story: The runner tied his shoes tight. He sprinted as fast as he could during the race. He
      crossed the finish line first.
  (A) He was sad that he lost the race.
  (B) He won the gold medal.
  End: He won the gold medal.

  Story: {question}
  (A) {choices[0]}
  (B) {choices[1]}
  End:



LAMBADA. LAMBADA is treated as open-ended continuation rather than multiple choice. Since the task
is essentially final-word prediction given a long context, we do not use in-context demonstrations. The model
directly continues the provided passage prefix, and the first generated word is taken as the prediction.
  LAMBADA prompt template

  {question}



OBQA. OBQA is evaluated with the same 2-shot generative multiple-choice protocol as MMLU. The model
is required to generate the text of the correct answer option rather than its label.
  OBQA prompt template

  Question: Which tool is best for tightening a screw?
  (A) spoon




                                                     72
  (B) hammer
  (C) screwdriver
  (D) paintbrush
  Answer: screwdriver

  Question: What do plants absorb from the air during photosynthesis?
  (A) carbon dioxide
  (B) oxygen
  (C) helium
  (D) salt
  Answer: carbon dioxide

  Question: {question}
  (A) {choices[0]}
  (B) {choices[1]}
  (C) {choices[2]}
  (D) {choices[3]}
  Answer:



HellaSwag. HellaSwag is evaluated as a 4-way situational continuation task. We use Context: as the input
field name to emphasize that the model must choose the most plausible continuation for the given context.
  HellaSwag prompt template

  Context: The girl puts the bread into the toaster and pushes the lever down. The bread
  (A) becomes a slice of pizza.
  (B) starts to toast and turn brown.
  (C) disappears immediately.
  (D) turns into a glass of water.
  Answer: starts to toast and turn brown.

  Context: The goalkeeper sees the ball coming towards the net. He dives and
  (A) catches the ball with his hands.
  (B) starts dancing in the field.
  (C) opens a laptop to check email.
  (D) runs away from the stadium.
  Answer: catches the ball with his hands.

  Context: {question}
  (A) {choices[0]}
  (B) {choices[1]}
  (C) {choices[2]}
  (D) {choices[3]}
  Answer:



SIQA. SIQA is a 3-way social commonsense multiple-choice benchmark. The template is identical in spirit
to the other multiple-choice prompts, except that only three options are presented.
  SIQA prompt template

  Question: Jordan wanted to tell a joke to his friends. What does Jordan need to do before this?
  (A) ignore his friends
  (B) think of a funny story



                                                   73
  (C) leave the room
  Answer: think of a funny story

  Question: Kai helped his neighbor carry heavy groceries inside. How would the neighbor feel?
  (A) angry
  (B) grateful
  (C) scared
  Answer: grateful

  Question: {question}
  (A) {choices[0]}
  (B) {choices[1]}
  (C) {choices[2]}
  Answer:



SQuAD. SQuAD is evaluated as short-answer generation under a standard extractive QA format. The
prompt provides a context paragraph and a question, and the model is asked to generate the answer phrase
directly. In our unified protocol, the generated answer is normalized before exact-match evaluation.
  SQuAD prompt template

  Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who
       in the 10th and 11th centuries gave their name to Normandy, a region in France. They were
      descended from Norse raiders and pirates from Denmark, Iceland and Norway.
  Question: In what country is Normandy located?
  Answer: France

  Context: {context}
  Question: {question}
  Answer:



Final remarks. These templates are kept identical across AR, LLaDA, and Cola DLM, so that the comparison
focuses on the model family itself rather than on prompt engineering differences. In all cases, evaluation
is performed after unified post-processing, including truncation at the first newline and text normalization
before matching against the candidate options or reference answers.

H.4   Setup Details
To ensure a fair comparison, we align AR, LLaDA, and Cola DLM on the same pretraining corpus, tokenizer
family, sequence length, batch size, optimizer, learning-rate schedule, hardware setup, and evaluation protocol.
As a result, all methods observe the same number of training tokens at matched optimization steps, making
step-wise comparison directly meaningful. We also keep the core generation backbone at a comparable scale:
AR and LLaDA use ∼1.8B non-embedding transformer parameters, while the DiT backbone in Cola DLM
is also kept at ∼1.8B; the remaining parameter difference comes from the additional VAE required by the
continuous latent formulation.
In addition to the matched data, optimization, and model-scale settings in Tables 8 and 9, we further verified
implementation-level numerical consistency across the three methods. As shown in Table 10, all models save
and load checkpoints in fp32, keep runtime weights in fp32, and perform the main forward computation under
bfloat16 autocast, while numerically sensitive operations such as normalization and softmax remain in fp32.
Therefore, the precision regime is broadly aligned across methods. The main implementation difference is that
Cola DLM contains an additional VAE encode/decode path, which is intrinsic to continuous latent modeling
rather than an extra numerical advantage.


                                                      74
Table 8 Controlled training and evaluation setup shared across AR, LLaDA, and Cola DLM. These settings are
matched unless method-specific generation mechanisms require otherwise.

Category                                                    Setting
Tokenizer                                                   OLMo 2 tokenizer
Supervised data mixing                                      None
Sequence length                                             512
Global batch size                                           1,408
Tokens per step                                             720,896
Precision                                                   bfloat16
Optimizer                                                   AdamW
Peak learning rate                                          1.5 × 10−4
Betas                                                       (0.9, 0.95)
Weight decay                                                0.01
Gradient clipping                                           1.0
Warmup                                                      5K steps, linear warmup from 1 × 10−6 to 1.5 × 10−4
LR decay                                                    Cosine decay to 1 × 10−5
Random seed / data seed                                     66 / 6198
Max evaluation new tokens                                   32



Overall, the comparison is controlled along the main axes that determine pretraining scale and optimization
difficulty, including data, tokenizer family, token budget, optimizer, learning-rate schedule, hardware, and
evaluation protocol. Therefore, the observed differences mainly reflect the modeling paradigm itself rather
than mismatched training resources or evaluation settings.

H.5   Why VAE Pretraining Does Not Compromise Fairness
A potential concern is that Cola DLM includes an additional VAE pretraining stage, whereas the AR baseline
is trained fully from scratch. We therefore clarify why this does not undermine the fairness of the comparison.
First, the pretrained VAE encoder is not itself the generative prior of Cola DLM. In our formulation, the
generative model is defined by a decoder pθ (x | z0 ) and a latent prior pψ (z0 ), while qϕ (z0 | x) is used only
for variational inference and latent construction during training. At inference time, new response content is
generated by the DiT prior block by block and then decoded into text. Therefore, the extra VAE stage should
be understood as learning a stable text–latent interface, rather than pretraining an additional text generator.
Second, the overall comparison is already controlled along the main axes that determine pretraining scale and
optimization difficulty. AR, LLaDA, and Cola DLM are aligned on the same training corpus, tokenizer family,
sequence length, batch size, optimizer, learning-rate schedule, hardware setup, and evaluation protocol. We
also keep the core generation backbone in a comparable regime: AR and LLaDA use ∼1.8B non-embedding
transformer parameters, while Cola DLM uses a ∼1.8B DiT prior; the remaining parameter difference comes
from the additional VAE required by the continuous latent formulation. As a result, the comparison remains
centered on the actual generation backbone rather than on mismatched training resources.
Third, Figure 15 provides an empirical perspective on representation-space stabilization. In the AR model,
both the contextual hidden states and the global input embeddings show relatively large drift at the beginning
of training, but quickly enter a stable regime. The inter-step drift becomes small after the early phase, and
the distance to the first checkpoint also gradually saturates. The pretrained VAE encoder exhibits an even
more explicit stabilization pattern: its inter-step drift drops rapidly and then remains nearly flat, while its
distance to the early checkpoint changes very little after the initial stage. This indicates that, in both model
families, the representation space becomes largely stable relatively early; once this happens, the dominant
learning burden is no longer representation construction itself, but the modeling of generation dynamics on
top of that space.



                                                       75
Table 9 Model capacity and architecture comparison. The core generation backbone is kept in a comparable
regime across methods. Cola DLM introduces an additional VAE module for continuous latent modeling.

Model                                     AR                   LLaDA                       Cola DLM
Model type                        Autoregressive LM     Masked diffusion LM     Continuous latent diffusion LM
Total parameters                        ∼2.2B                  ∼2.2B                        ∼2.3B
Core non-embedding parameters           ∼1.8B                  ∼1.8B                     DiT ∼1.8B
VAE parameters                            –                       –                        ∼500M
Embedding parameters                 410,738,688            410,738,688                  308,054,016
Vocabulary size                        100,278                100,278                      100,278
Main hidden dimension                   2,048                  2,048                     DiT: 2,048
FFN dimension                           8,192                  8,192                     DiT: 8,192
Number of backbone layers                 27                     27                        DiT: 24
Attention heads                           16                     16                        DiT: 16
Head dimension                           128                    128                          128
Positional encoding                     RoPE                   RoPE                         RoPE
Attention pattern                      Causal              Bidirectional        VAE causal + DiT block causal
Latent dimension                          –                       –                           16
VAE encoder / decoder blocks              –                      –                          4/4
VAE hidden dimension                      –                      –                          1,536
VAE FFN dimension                         –                       –                         6,144



This observation is important for interpreting fairness. AR models also spend a nontrivial portion of training
on forming and stabilizing their embedding and hidden-state geometry. Cola DLM makes this step explicit
through VAE pretraining, but this does not mean that it receives a hidden pretrained generative advantage.
Instead, it moves representation stabilization into a separate and modular stage, after which the main text-
modeling burden still lies in the DiT prior and the conditional decoder. In this sense, VAE pretraining mainly
amortizes representation learning, rather than replacing the need to learn the actual generative mechanism.
Moreover, the results in Figure 3 already suggest that a pretrained encoder alone cannot explain the final
gains. Although Fix VAE can be competitive at smaller compute budgets, it saturates earlier, whereas Joint
DiT x1 shows the strongest overall scaling trend. If the advantage mainly came from having a pretrained
encoder by itself, the fixed-space variant should have remained dominant. Instead, the empirical pattern
indicates that the key factor is the subsequent prior learning and co-adaptation with DiT, rather than a frozen
pretrained representation alone.
Overall, we view the additional VAE stage as a method-specific representation-construction cost required
by continuous latent modeling, analogous in spirit to the early embedding-space formation that AR models
must also undergo during training. Once the representation space has stabilized, the comparison remains
focused on the generative backbone under matched data, optimization, and evaluation settings. Therefore, the
observed performance differences are more appropriately attributed to the modeling paradigm itself, rather
than to an advantage from VAE pretraining.

H.6     Fixed vs. Evolving Latent Space Additional Results
As shown in Figure 16, we further compare continuous joint evolution and interval training under a stronger
setting where the Interval variant is also initialized from a pretrained VAE. This setting should therefore be
interpreted as isolating the effect of continuous versus intermittent latent updates after stable initialization,
rather than the effect of initialization quality itself. Even under this stronger setup, Joint DiT x1 remains
consistently better than Interval across Task Average, LAMBADA, MMLU, and SIQA. This suggests that the
advantage of continuous latent evolution does not merely come from a better starting point. Instead, once
a meaningful latent space has been established, the latent representation still benefits more from sustained
co-evolution with DiT, while periodic freezing tends to interrupt the mutual adaptation between the evolving


                                                       76
Table 10 Implementation-level numerical setup comparison across AR, LLaDA, and Cola DLM. We also verify
that the three methods use closely aligned checkpoint precision, runtime autocast policy, and numerically sensitive
operations.

Aspect                      Cola DLM                         AR                              LLaDA
Checkpoint save dtype       fp32                             fp32                            fp32
Checkpoint load dtype       fp32                             fp32                            fp32
Runtime weight dtype        fp32                             fp32                            fp32
Model input type            Latents kept in fp32 before      Token ids in integer format     Token ids in integer format
                            entering DiT; cast under au-
                            tocast when needed
Autocast policy             bfloat16 autocast for DiT/-      bfloat16 autocast for genera- bfloat16 autocast for diffu-
                            VAE forward                      tion forward                  sion generation forward
Main matmul / attention     bf16                             bf16                          bf16
compute
Numerically sensitive ops LayerNorm / softmax in fp32        LayerNorm / softmax in fp32     LayerNorm / softmax in fp32
Sampling-time accumula- Iterative latent update kept         Token decoding without it-      Sampling uses numerically
tion                      in fp32; CFG combination           erative floating-point latent   stable probability computa-
                          applied during denoising           accumulation                    tion for token selection



latent space and the DiT prior. And this results also further verify the last condition in Eq. (3.35) of
Section 3.3.2, and provides strong support for the potential advantage of Cola DLM.

H.7   VAE LogSNR Formula
For the VAE encoder posterior
                                       qϕ (z | x) = N µϕ (x), diag σϕ2 (x)                                       (H.1)
                                                                             
                                                                                  ,
we define the VAE logSNR as the log ratio between the signal power carried by the posterior mean and the
noise power carried by the posterior variance:
                                                                     
                                                      Ex,i µϕ,i (x)2
                                      logSNRvae = log                   ,                          (H.2)
                                                      Ex,i [σϕ,i (x)2 ]

where i indexes all latent coordinates, including token/spatial positions and channels, and the expectation is
taken over the data distribution and latent dimensions.
In practice, we estimate it empirically on a dataset or a mini-batch:
                                                    1
                                                      PN 1 PD 2
                                                              i=1 µn,i
                                    logSNRvae = log 1 Pn=1
                                     \              N
                                                       N
                                                           D
                                                           1
                                                             PD 2 ,                                              (H.3)
                                                    N  n=1 D  i=1 σn,i

where N is the number of samples and D is the number of latent coordinates per sample. If the encoder
predicts log-variance, we compute
                                         2             2
                                                                                                (H.4)
                                                           
                                       σn,i = exp log σn,i   .

A larger VAE logSNR indicates that the posterior mean dominates the posterior uncertainty, corresponding
to a cleaner and more deterministic latent representation. Throughout the paper, when we refer to VAE
logSNR, we specifically mean the quantity defined in Eq. (H.2).

H.8   Noise Schedule Timestep Shift Visualization
As shown in Figure 17, we visualize the timestep-sampling distributions induced by different timestep-shift
settings. Compared with the uniform schedule, LogitNormal schedules place non-uniform emphasis on different
parts of the denoising trajectory: increasing µ shifts the distribution toward later timesteps, while σ controls


                                                        77
                                                        Representation-space stability: AR model (from scratch) vs. VAE encoder (pre-trained, frozen for DiT)
                                                        (a) AR model    inter-step drift (trained from scratch)                                                                                                      (b) VAE encoder     inter-step drift (pre-trained)
                                                                                                                          0.06
                                                                                               Contextual hidden state                                                                                                                                      VAE encoder (mode)




                                                                                                                                       Input-embedding drift (cosine dist.)
                                   0.25                                                                                                                                                                       1.6
                                                                                               Global input embedding                                                                                                                                       stable regime




Contextual drift (cosine dist.)
                                                                                               stable regime              0.05                                                                                1.4




                                                                                                                                                                                    Cosine distance (×10 3)
                                   0.20
                                                                                                                                                                                                              1.2
                                                                                                                          0.04                                                                                1.0
                                   0.15
                                                                                                                                                                                                              0.8
                                   0.10                                                                                   0.03
                                                                                                                                                                                                              0.6
                                                                                                                          0.02                                                                                0.4
                                   0.05
                                                                                                                                                                                                              0.2
                                   0.00                                                                                   0.01                                                                                0.0
                                                            20k        40k        60k          80k          100k                                                                                                       20k        40k           60k           80k           100k
                                                                              Training step                                                                                                                                             Training step
                                                              (c) AR model      drift vs. first checkpoint                                                                                                             (d) VAE encoder      drift vs. first checkpoint
                                             0.8                                                                          0.6
                                                                                                                                                                                                        2.00




                                                                                                                            Input-embedding drift vs. first (cos.)
                                             0.7




         Contextual drift vs. first (cos.)
                                                                                                                          0.5                                                                           1.75




                                                                                                                                                                              Cosine distance (×10 3)
                                             0.6                                                                                                                                                        1.50
                                                                                                                          0.4
                                             0.5                                                                                                                                                        1.25
                                                                                                                                                                                                                                                        VAE encoder vs. step 5k
                                             0.4                                                                          0.3                                                                           1.00                                            flat / pre-trained regime
                                             0.3                                                                                                                                                        0.75
                                                                                                                          0.2
                                             0.2                                                                                                                                                        0.50
                                                                                    Contextual hidden state vs. step 5k
                                                                                    Global input embedding vs. step 5k    0.1
                                             0.1                                                                                                                                                        0.25
                                                                                    saturated regime
                                             0.0                                                                          0.0                                                                           0.00
                                                   0k         20k       40k         60k         80k         100k                                                                                                         20k       40k           60k           80k          100k
                                                                              Training step                                                                                                                                             Training step

Figure 15 Representation-space stability of AR training and pretrained VAE encoding. Although the AR
model learns its embedding and hidden-state geometry from scratch, and the VAE encoder starts from a pretrained
state, both representations stabilize rapidly in the early stage of training. The inter-step drift in AR drops quickly
and soon enters a stable regime, while the VAE encoder remains nearly flat throughout DiT training. This suggests
that representation-space formation is largely an early-stage effect in both cases, and its influence on later generation
performance is limited. The main generative burden therefore still lies in the subsequent backbone modeling, rather
than in continued changes of the embedding or latent space itself.


how concentrated or dispersed the sampling mass is. From the perspective of the main text, this means that
timestep shift effectively changes which semantic-information regime is more frequently emphasized during
training, rather than simply tuning an isolated numerical hyperparameter.

H.9                                                 Timestep Shift Formula
We implement timestep shift by replacing uniform timestep sampling with a LogitNormal distribution on the
normalized timestep. Specifically, let
                                              u ∼ N (µ, σ 2 ),                                      (H.5)
and map it to the unit interval by the logistic function:
                                                                                                                             1
                                                                                              s = sigmoid(u) =                    ,                                                                                 s ∈ (0, 1).                                            (H.6)
                                                                                                                          1 + e−u
The continuous timestep is then defined as
                                                                                                        t = T · s = T · sigmoid(u),                                                                                                                                        (H.7)
where T is the maximum diffusion timestep. In practice, a discrete training timestep is obtained by rounding
or flooring t to the valid timestep grid.
Equivalently, the normalized timestep s = t/T follows a LogitNormal distribution:
                                                                                                          s ∼ LogitNormal(µ, σ 2 ).                                                                                                                                        (H.8)


                                                                                                                            78
                        Joint DiT x1          Interval Joint DiT x1                                                Joint DiT x1          Interval Joint DiT x1
                        (BERT loss)           (BERT loss)                                                          (BERT loss)           (BERT loss)



                                      Tasks Average                                                                                    LAMBADA
           26.3                                                                                   41.7


           25.1                                                                                   40.2




Accuracy                                                                               Accuracy
           24.1                                                                                   39.2


           23.1                                                                                   37.1


           21.6                                                                                   34.8


           19.8                                                                                   33.1
                  117   156           195     234           273       312   351                          117   156               195     234           273       312     351
                                      Total FLOPs (EFLOPs)                                                                       Total FLOPs (EFLOPs)



                        Joint DiT x1          Interval Joint DiT x1                                                Joint DiT x1          Interval Joint DiT x1
                        (BERT loss)           (BERT loss)                                                          (BERT loss)           (BERT loss)



                                            MMLU                                                                                        SIQA
           12.7                                                                                   26.3


           11.6                                                                                   24.8




Accuracy                                                                               Accuracy
           10.5                                                                                   23.4



            9.8                                                                                   21.5



            8.4                                                                                   19.6



             6                                                                                    18.1
                  117   156           195     234           273       312   351                          117   156               195     234           273       312     351
                                      Total FLOPs (EFLOPs)                                                                       Total FLOPs (EFLOPs)




Figure 16 Continuous co-evolution outperforms interval updates with pretrained VAE initialization. Across
Task Average, LAMBADA, MMLU, and SIQA, Joint DiT x1 still shows a stronger overall trend and better final
performance than Interval, indicating that even with a stable pretrained VAE initialization, the latent space benefits
more from sustained co-evolution with DiT than from periodic freezing.


Its probability density function is
                                                                                                         s
                                                                                                                   !
                                                                                                               2
                                                    1      1          (log 1−s − µ)
                                      p(s; µ, σ) = √            exp −                                                  ,          s ∈ (0, 1).                           (H.9)
                                                  σ 2π s(1 − s)             2σ 2

After rescaling to the timestep domain t ∈ (0, T ), the corresponding density becomes
                                                                   
                                                       1    t
                                         p(t; µ, σ) = p       ; µ, σ .                                                                                                 (H.10)
                                                      T     T

Under this parameterization, µ mainly controls the location of the sampling mass: a larger µ shifts probability
toward later timesteps. In contrast, σ controls the spread of the distribution: a smaller σ produces a sharper
concentration, while a larger σ yields a more dispersed schedule. Therefore, timestep shift directly changes
the distribution of sampled logSNR regimes during training, rather than merely reweighting timestep indices
numerically.

I          Discussion


                                                                                  79
                                                             Timestep Sampling Distributions (0–1000)


                                    Uniform                                                     LogitNormal (μ=0, σ=1)                                              LogitNormal (μ=0.5, σ=1)
                                                                              0.00175                                                                0.0020
                                                                                                                                                                    Histogram
          0.0010                                                              0.00150                                                                               Theoretical PDF
                                                                                                                                                     0.0015
          0.0008                                                              0.00125



Density                                                             Density                                                                Density
                                                                              0.00100
          0.0006                                                                                                                                     0.0010
                                                                              0.00075
          0.0004
                                                                              0.00050                                                                0.0005
          0.0002                     Histogram                                                              Histogram
                                                                              0.00025
                                     Theoretical PDF                                                        Theoretical PDF
          0.0000                                                              0.00000                                                                0.0000
                     0    200      400       600       800   1000                           0    200      400       600       800   1000                        0    200      400     600   800   1000
                                     Timestep                                                               Timestep                                                            Timestep


                         LogitNormal (μ=1, σ=1)                                                 LogitNormal (μ=1.7, σ=1)                                            LogitNormal (μ=2, σ=1)
          0.0025
                         Histogram                                                  0.004       Histogram                                                           Histogram
                                                                                                                                                        0.005
                         Theoretical PDF                                                        Theoretical PDF                                                     Theoretical PDF
          0.0020
                                                                                    0.003                                                               0.004



Density                                                                   Density                                                             Density
          0.0015
                                                                                                                                                        0.003
                                                                                    0.002
          0.0010
                                                                                                                                                        0.002

                                                                                    0.001
          0.0005                                                                                                                                        0.001


          0.0000                                                                    0.000                                                               0.000
                     0    200      400       600       800   1000                           0    200      400       600       800   1000                        0    200      400     600   800   1000
                                     Timestep                                                               Timestep                                                            Timestep


                         LogitNormal (μ=2.3, σ=1)                                               LogitNormal (μ=1, σ=0.5)                                            LogitNormal (μ=1, σ=2)
             0.007       Histogram                                                              Histogram                                               0.004       Histogram
                                                                                    0.004
                         Theoretical PDF                                                        Theoretical PDF                                                     Theoretical PDF
             0.006

                                                                                    0.003                                                               0.003
             0.005


   Density                                                                Density                                                             Density
             0.004
                                                                                    0.002                                                               0.002
             0.003

             0.002
                                                                                    0.001                                                               0.001
             0.001

             0.000                                                                  0.000                                                               0.000
                     0    200      400       600       800   1000                           0    200      400       600       800   1000                        0    200      400     600   800   1000
                                     Timestep                                                               Timestep                                                            Timestep



Figure 17 Visualization of timestep-shifted sampling distributions. The figure compares the uniform schedule with
LogitNormal timestep-sampling distributions under different µ and σ. Increasing µ progressively shifts the sampling
mass toward later timesteps, while changing σ controls the sharpness of this concentration. This illustrates that
timestep shift does not merely change a numerical sampling preference, but effectively changes the logSNR regime
emphasized during training.


I.1            Understanding the Conditioning Strategies in the First Generation Block
This subsection explains the conditioning strategies compared in Table 5, and further analyzes them from the
perspective of conditional Flow Matching in Cola DLM. The key issue arises in the first generation block,
where the model simultaneously sees a known prompt region and an unknown region to be generated. In other
words, the input to this block is mixed: part of it should be preserved as condition, while the rest should be
completed by the model. The methods in Table 5 differ precisely in how they handle this mixed denoising
process.

What makes the first block special. Later generation blocks are purely generative: all latents in the block are
unknown and are produced by denoising. The first generation block is different because it sits immediately
after the prefix prompt. As a result, some positions in this block are already determined by the prompt, while
the remaining positions are not. A good conditioning strategy should therefore achieve two goals at the same
time: it should preserve the known region reliably, and it should allow the unknown region to grow coherently
from that preserved context.
From the viewpoint of Cola DLM, this distinction is especially important because the prior is learned block


                                                                                                           80
by block through a conditional flow:
                                                             B
                                                      (1)                  (b)   (<b)
                                                             Y
                                     pψ (z0 ) = pψ (z0 )           pψ (z0 | z0          ),
                                                             b=2

and the corresponding conditional Flow Matching objective learns a vector field of the form
                                                       (b)         (<b)
                                                 vψ (zt , t; z0           ).

That is, the current noisy block is predicted under clean historical latent conditions. At inference time,
generation follows the same factorization by transporting a noise seed under the clean prefix and historical
latent conditions. Therefore, the mathematically natural role of the known region in the first generation block
is not that of a denoising target, but that of a fixed condition.

A decomposition of the first generation block.    To make this point more explicit, let the first generation block
be decomposed as
                                                 z (1) = (zK , zU ),
where zK denotes the known region determined by the prefix, and zU denotes the unknown region to be
generated. In an ideal formulation, the task of the first generation block should be understood as

                               generate zU under the fixed condition (zpre , zK ),

rather than to jointly re-estimate both zK and zU . Equivalently, the desired conditional prior is closer to

                                                 pψ (zU | zpre , zK ),

where the known part serves as a stable boundary condition and only the unknown part is transported by the
flow.

Partial repaint. The first family of methods in Table 5 is partial repaint. These methods repeatedly revisit
the early denoising steps and re-inject guidance into the known region during only part of the trajectory.
The parameter t controls how many repaint cycles are used inside each early denoising step. For example,
t = 1 means that each step is executed once, while t = 3 means that the model denoises, re-noises, and
denoises again multiple times before moving on. The parameter m controls how long the known region remains
guided. When m = 1.0, guidance is applied throughout the whole denoising trajectory of the first block.
When m = 0.7 or m = 0.3, the model only receives such guidance in the earlier portion of the trajectory, and
then the remainder is left to evolve more freely.
The important point is that partial repaint does not keep the known region perfectly fixed. Instead, it replaces
the known region with a timestep-matched noisy surrogate during part of the trajectory and only nudges that
region back toward the target at selected denoising steps. This makes the strategy softer and more flexible,
but also fundamentally weaker as a conditioning mechanism.

A simple example for partial repaint. Consider a first generation block with four denoising transitions. If we
use partial repaint with t = 3 and m = 0.5, then only the first half of the trajectory receives explicit guidance
on the known region, while each of these early steps is revisited three times. Intuitively, the model repeatedly
tries to correct the early transition, but once it enters the later half of the trajectory, the known region is no
longer explicitly reinforced. The final result therefore depends on whether those earlier corrections are strong
enough to keep the conditional structure stable.




                                                             81
Clean condition repaint. The method labeled Clean cond. in Table 5 takes a much stronger approach. Instead
of providing only partial or noisy guidance, it keeps the known region directly aligned with the clean condition
throughout the denoising process of the first block. In effect, the model always sees an exact and stable
version of the prompt-conditioned part, while only the unknown region is allowed to change.
This strategy is much more conservative than partial repaint. It reduces ambiguity in the mixed region and
makes the role of the first block very clear: preserve the prompt-conditioned part exactly, and generate the
remaining part around it. This is especially useful when the boundary between known and unknown positions
is semantically important.

A simple example for clean conditioning. Suppose the first few positions in the block correspond to prompt
content and the remaining positions must be generated. Under clean condition repaint, the prompt side
is always kept fixed, no matter how many denoising steps are taken. The model therefore never needs to
re-discover or re-estimate that part. Its effort is focused entirely on making the unknown side compatible
with a stable context. This is a much easier problem than simultaneously preserving the known side and
generating the unknown side under drifting noise.

Left padding and right padding. The last two methods, Left pad. and Right pad., should be understood as
layout-based alternatives rather than repaint-based conditioning strategies. They change where the known
condition is placed relative to the region to be generated, but they do not explicitly enforce the condition
during denoising in the same way as repaint-based methods do.
Their role in the table is useful for interpretation. If padding alone were sufficient, then a simple positional
arrangement would already match the performance of explicit conditioning. If not, then the gain must come
from the conditioning mechanism itself rather than from layout alone.

A Flow Matching view of why clean conditioning works best. The advantage of clean condition repaint is
not merely that it provides “stronger guidance.” More fundamentally, it is the only strategy among those
compared that preserves the original conditional semantics of Cola DLM.
Under the standard Flow Matching formulation, let the target conditional velocity be denoted by ut , and let
the optimal vector field be the conditional expectation
                                              v ∗ (zt , t; c) = E[ut | zt , t, c],
where c denotes the conditioning information. For the first generation block, the ideal condition is
                                                      c = (zpre , zK ),
namely, the clean prefix latent together with the clean known region inside the first block. Under clean
condition repaint, the model effectively solves the transport problem under exactly this condition. In the
unknown region, this can be viewed as a conditional velocity regression of the form
                                 ∗
                                vclean (zU,t , t; zpre , zK ) = E[uU
                                                                   t | zU,t , t, zpre , zK ].

Although the implementation still predicts the vector field for the whole current block, the uncertainty that
truly matters is concentrated on the unknown region, while the known region acts as a fixed boundary
condition. This is precisely the role decomposition that the model is designed to exploit.

Why partial repaint is mathematically harder. Partial repaint changes this problem in a crucial way. Instead
of conditioning on the true clean known region zK , it supplies a degraded and time-varying surrogate z̃K,t
during only part of the trajectory. The effective condition becomes
                                                     c̃t = (zpre , z̃K,t ),
and the corresponding optimal regression target becomes
                               ∗
                              vpartial (zU,t , t; zpre , z̃K,t ) = E[uU
                                                                      t | zU,t , t, zpre , z̃K,t ].


                                                              82
This is generally a more difficult regression problem. Since z̃K,t is only a noisy surrogate of the true condition,
the same (zU,t , t, z̃K,t ) may correspond to a broader set of compatible clean targets. As a result, the conditional
variance of the target velocity is larger. In terms of Bayes risk, one can compare

                          R(c) = E Var(uU                     R(c̃t ) = E Var(uU
                                                                                               
                                          t | zU,t , t, c) ,                   t | zU,t , t, c̃t ) .

Since c̃t is a weaker and noisier version of c, the second quantity is generally larger. Therefore, partial repaint
increases the irreducible uncertainty of the conditional velocity regression itself. From the Flow Matching
perspective, its weakness is not simply that the guidance is softer, but that it turns a clean conditional
transport problem into a noisier mixed recovery problem.

Condition-role mismatch in partial repaint. There is also a structural mismatch. In Cola DLM, the flow
path is used for prior transport, not for observation recovery. Historical conditions are supposed to be stable
anchors, while the current noisy block is the object being transported. Partial repaint violates this separation
by converting the known region from a condition into a partially denoised state variable. In other words, it
changes the task from
                              transport the unknown part under fixed condition
to
                    jointly maintain a noisy known part and transport the unknown part.
This role confusion is especially harmful in the first generation block, because the known and unknown regions
are tightly coupled inside the same block. Once the known side drifts, the unknown side no longer grows from
a stable semantic anchor.

Why errors accumulate along the flow trajectory. The weakness of partial repaint is further amplified by the
continuous nature of flow-based generation. Let the ideal vector field under clean conditioning be v ∗ (z, t; c),
and let the field induced by partial repaint be

                                           ṽ(z, t) = v ∗ (z, t; c) + δ(z, t),

where δ denotes the condition-induced bias. Since inference transports the latent by integrating the learned
vector field along an ODE trajectory, this bias accumulates over time. Under standard stability assumptions,
the trajectory error satisfies a bound of the form
                                                          Z t
                                       ∥ẑt − zt∗ ∥ ≤ eLt     ∥δ(zs , s)∥ ds,
                                                             0

where L is a Lipschitz constant of the vector field. This expression makes the weakness of partial repaint
especially clear.
First, reducing m makes performance worse because the explicit guidance only covers an earlier portion of the
trajectory. Once guidance is removed, the later part evolves under a mismatched condition, so the induced
bias continues to accumulate during the unguided interval. Second, increasing t from 1 to 3 does not provide
stable gains because repeated corrections in the early stage cannot eliminate the structural mismatch in the
later stage. In other words, repaint can repeatedly repair local early transitions, but it cannot turn a transient
condition into a persistent one.

Why the first block is more sensitive than later blocks. Another reason for the strong gap is that the first
generation block is not a purely unknown block. The known and unknown regions coexist and interact inside
the same block. Therefore, any instability in the known region is immediately transmitted to the unknown
region through the within-block interactions. In latent space, this effect is even stronger because the known
region functions as a semantic anchor rather than a mere token-level hint. If that anchor is kept clean, the
model only needs to fit the unknown part to a stable context. If that anchor becomes noisy and drifts over
time, the model must simultaneously infer the boundary condition and generate the completion, which is a
fundamentally harder problem.


                                                           83
Why padding is better than partial repaint but still worse than clean conditioning. Left and right padding
do not explicitly re-noise the known region, so they avoid the most harmful failure mode of partial repaint.
This is why they are often stronger than most partial repaint settings. However, padding alone still does
not provide a fully stable conditioning mechanism throughout denoising. It changes the positional layout,
but it does not lock the known region as an exact condition in the same way as clean condition repaint.
Moreover, padding can also make the block-causal attention pattern more complicated: by reshaping the
relative arrangement between the known and unknown regions, it may introduce a less natural or less aligned
dependency structure for the model to follow during denoising. Therefore, padding can be helpful, but it
cannot match the benefit of preserving a stable condition explicitly.

A unified intuition across all methods. All methods in Table 5 are trying to solve the same problem: the
first generation block contains both a part that should remain faithful to the prompt and a part that should
be newly generated. The difference lies in how strongly the known region is protected.
Partial repaint protects it only temporarily and only approximately. Clean condition repaint protects it
strongly and continuously. Left and right padding do not explicitly protect it during denoising, but instead
hope that a favorable layout will make the interaction easier for the model.

How to read Table 5. The results show a very clear pattern. Clean condition repaint consistently performs
best across all three tasks. This suggests that, for the mixed denoising problem in the first block, strong and
persistent conditioning is much more effective than soft partial correction. By contrast, reducing m in partial
repaint generally hurts performance, which indicates that shortening the guided portion of the trajectory
makes the known region harder to preserve. Increasing the repaint cycles from t = 1 to t = 3 does not produce
stable gains either, suggesting that repeatedly revisiting early denoising steps cannot by itself compensate for
weak conditioning.
Another useful observation is that left and right padding are often stronger than partial repaint, but still
remain clearly below clean condition repaint. This indicates that positional arrangement does matter, but it is
not the main factor. In particular, although padding may alleviate some interference by changing the layout,
it can also make the block-causal attention structure less clean and more difficult to optimize. The dominant
advantage still comes from explicitly maintaining a stable condition in the known region throughout denoising.

Practical takeaway. The first generation block is not merely a smaller version of later generation blocks. It is
a special mixed region where conditioning and generation must coexist. From the perspective of conditional
Flow Matching, the key challenge is to preserve the known region as a stable condition while transporting
only the unknown region. The experiments suggest that this regime benefits most from a strong and explicit
preservation of the known condition. In practice, this is exactly what clean condition repaint provides, which
explains why it is consistently the strongest strategy in Table 5.

I.2     More Unified Approach Results with Image Modalities
In this section, we present additional samples that demonstrate cross-modal generation across different input
and output modalities. It is worth emphasizing that all text and image outputs are produced by a single
unified model. All tasks are pretrained from scratch, and the model is trained jointly on multiple task types,
with the relative data proportions adjusted across different training stages. Although the model is trained
with only a very limited amount of data, the resulting samples already exhibit promising quality, suggesting
that Cola DLM has substantial potential for multimodal understanding and generation.

I.2.1   Text to Text
In this subsection, we present examples of text-to-text generation. As shown in the samples, the generated
text is generally coherent and fluent, despite the fact that the model is trained on only approximately 1B text
tokens in total.




                                                      84
I.2.2   Text to Image
In this subsection, we further evaluate the model’s ability to perform text-to-image generation.

I.2.3   Image × Text to Text
In this subsection, we present examples of image-and-text-to-text generation. The outputs are generally
fluent and semantically aligned with the inputs. Notably, on top of a model that has received only limited
text-generation and image-generation training, we further train with merely approximately 5M image–text
pairs, after which the model already exhibits a preliminary captioning ability. This suggests that Cola
DLM can efficiently acquire cross-modal understanding and generation capabilities under limited multimodal
supervision. We also note that all images shown in this subsection are generated by external image generation
models from ground-truth captions, and therefore do not involve copyright concerns associated with real
images.




                                                     85
  Text to Text
  Prefix Prompt: You now have the opportunity to really              Prefix Prompt: Here is why the Indian Army, Navy, and
  learn a lot. Just ask when you do not know something; the          Air Force salute in a different way. Probably no one ever
  worst thing you can do is pretend that you understand.             noticed that the salute styles of the Indian Army, Navy, and
  Generation: This is great for you, so do not be intimidated.       Air Force are different from each other.
  Instead, show your openness and willingness to learn, and I        Generation: The gesture of respect comes with such modest
  am sure anyone on your team will be more than happy to             reasons that you will be amazed at the chivalry of the defence
  help.                                                              forces. When they raise their hands for the salute, they do
                                                                     not just greet, but honor a long-standing tradition.
  Prefix Prompt: As an adult, your hair is often one of the
  first indicators used to describe you. You are often known         Prefix Prompt: “Dark Knight Begins!” Walkthrough to the
  as the friend with the curls. If you then find a stylist who       Batman animated movies. DC Universe is a world filled with
  understands and respects curls, you will eventually relay your     iconic characters, and many fans spend hours watching its
  entire hair history to her.                                        animated series.
  Generation: But we all do this with a knowing look and a           Generation: Since every kid grows up knowing Batman
  slightly sarcastic grin. This is a common, shared history, one     and Superman, these stories remain a natural entry point
  that curly people seem to recognize in one another immedi-         into the darker side of the DC universe, especially for viewers
  ately.                                                             who want an accessible introduction to its animated films.

  Prefix Prompt: Comparative Advantage: Nobel laureate               Prefix Prompt: Eddie Huang has circled back to his roots
  Paul Samuelson was once challenged to name one proposition         to close out Fresh Off the Boat Season 2, and Part 1 of the
  in the social sciences that is both true and non-trivial. Sev-     New York episode travels to the Bronx, which, he learns from
  eral years later, he gave the correct response: comparative        some locals itching to get out, will probably never gentrify.
  advantage.                                                         Generation: Honestly, though, I am good with cake and
  Generation: That it is logically true need not be argued           aggressive pedicures, both of which the Bronx seems to have
  before a mathematician; that it is not trivial is attested by      in spades. Check out Fresh Off the Boat and see how the
  the thousands of intelligent men who have never been able          episode leans into the borough’s character and humor.
  to grasp the doctrine for themselves or to believe it after it
  was explained to them.                                             Prefix Prompt: Those who read Nietzsche in English trans-
                                                                     lation have been lied to, deceived, seduced, and hoodwinked
  Prefix Prompt: How do I draw a circle with arrows in               by dishonest translators and commentators.
  PowerPoint? Numbered Circular Arrows in PowerPoint using           Generation: My intention here will be twofold. First, to
  SmartArt.                                                          correct some of the more serious misinterpretations in the
  Generation: Step 1: Insert the relevant SmartArt graphic.          introduction to the Cambridge University Press translation
  Go to Insert, then SmartArt, then Cycle, and choose the            of Nietzsche’s Morgenröthe, and second, to clarify what is at
  segmented cycle graphic. Step 2: Convert it to shapes and          stake in reading the text more carefully.
  continue editing from there.
                                                                     Prefix Prompt: First off, thank Hashem that so few people
  Prefix Prompt: “It is N’Nance,” she said. That sent a small        in Israel have been hurt. It is not for lack of trying on the
  chill up my spine. She nodded at the horrified expression on       part of our foes.
  my face. “He has finally had his big accident.” I knew what        Generation: It is miraculous and wonderful. In reply to
  that meant. “I will be right there.”                               Dan, I am a physicist, and I also wondered when I first
  Generation: I turned to my own patient and said, “Athena,          saw the statistics. Although I would be cautious about
  can you lock this down by yourself?” The cybertank grumbled        overinterpreting them, the numbers are certainly striking.
  her engine for a second, then said, “I can manage. I will hold
  things here until you get back.”                                   Prefix Prompt: Europeans have a higher tolerance for
                                                                     government intrusiveness than do Americans. I do not think
  Prefix Prompt: Subject 3: Drilling and Completion of Coal          anything in that statement will surprise my readers.
  Bed Methane Wells. The drilling program for a CBM well is          Generation: The contrast becomes even more visible in
  generally a less complex process than that for other types of      television, where assumptions about privacy, regulation, and
  unconventional reservoirs.                                         public life often emerge in surprisingly casual ways and reveal
  Generation: This is true for several reasons: CBM reservoirs       broader cultural differences.
  are generally shallow, commonly ranging from roughly 500
  to 3000 feet in depth, and their operational requirements
  are often more straightforward than those of many other
  unconventional wells.


Figure 18 Additional text-to-text examples. We show more qualitative continuation results from the preliminary
unified prototype. The examples span conversational, narrative, expository, technical, and article-style text, illustrating
that the unified latent-prior formulation can support coherent continuation across heterogeneous textual styles.




                                                                86
Prompt: A delicate white dandelion seed      Prompt: Tall trees on both sides frame a       Prompt: Two large abstract paintings
head stands above a green meadow, with       dramatic sunset sky filled with glowing        dominate a polished interior hallway, their
other soft, blurred flowers nearby. A dark   orange, pale yellow, and blue cloud            vivid colors contrasting with the neutral
row of trees and a cloudy blue-gray sky      textures. The dark silhouettes below make      wall panels and glossy floor. Ceiling
give the open field a calm, breezy spring    the scene feel quiet, spacious, and gently     spotlights, patterned rugs, and nearby
atmosphere.                                  illuminated by evening light.                  doorways create the feeling of a private
                                                                                            gallery or elegant office space.




Prompt: A cozy outdoor dining area is lit    Prompt: A dense column of black and            Prompt: A vivid red rose fills the image
by a large yellow umbrella and several red   gray smoke rises from a rural property near    in extreme close-up, revealing many
hanging lanterns. Wooden tables, simple      small buildings and scattered trees. The       layered petals curling toward a tight spiral
chairs, and warm pools of light create an    surrounding dirt paths, fields, and pale sky   at the center. The blurred green
intimate evening setting with a festive,     make the sudden fire or burning event feel     background keeps attention on the flower’s
traditional atmosphere.                      stark and unsettling.                          saturated color and soft texture.

Figure 19 Text-to-image examples (Part I). Qualitative samples generated by the unified prototype from descriptive
natural-language prompts, covering open landscapes, interiors, dining scenes, and floral close-ups.




                                                                 87
Prompt: Several boats glide across             Prompt: White spring blossoms cluster        Prompt: Golden sunset light spreads
rippling water during sunset, appearing as     thickly along interwoven tree branches,      behind a line of dark mountain silhouettes
dark silhouettes beneath a long arched         mixed with small green leaves and            and reflects across the still lake below.
bridge. The orange sun and its reflection      unopened buds. The shallow depth of field    Grasses in the foreground add depth, while
stretch across the river, giving the scene a   makes the foreground flowers stand out       the mirrored peaks and warm sky create a
warm, tranquil waterfront mood.                while the background turns into a delicate   peaceful landscape composition.
                                               canopy of pale bloom.




Prompt: A bearded man with curly hair          Prompt: A broad blue lake stretches          Prompt: A sightseeing boat moves along
sits indoors wearing a dark blazer and         toward layered mountains beneath a           a wide river in front of grand historic
lavender shirt. Warm lamp light, stacks of     sweeping sky of textured white clouds.       architecture, including ornate towers, stone
documents, and framed artwork behind           The calm ripples, low dark shorelines, and   facades, and decorative rooflines.
him suggest an office or study, giving the     distant ridges create a cool, expansive      Reflections shimmer on the water,
portrait a serious conversational tone.        landscape with a quiet sense of depth.       emphasizing the scale and elegance of the
                                                                                            riverside cityscape.

Figure 20 Text-to-image examples (Part II). Additional samples spanning waterfront, floral, mountain, and
architectural prompts, together with a portrait composition.




                                                                   88
Prompt: A single leafless tree stands         Prompt: A simple stylized illustration       Prompt: Bright pink blossoms fill the
prominently in rolling green grassland, its   presents a red rose with green leaves,       foreground in dense clusters, their yellow
intricate branches spreading against a        centered inside a rounded purple             centers catching the light. More branches
bright sky. Distant mountains and soft        background with a thick black outline.       and flowers blur behind them, creating a
clouds frame the isolated tree, giving the    The bold colors, clean shapes, and           saturated spring scene that feels lively,
image a quiet, almost sculptural quality.     cartoon-like linework make the flower feel   sunny, and full of floral texture.
                                              decorative and playful.




Prompt: A loose navy blue sweater hangs       Prompt: A misty mountain rises beyond        Prompt: A calm lake reflects a bright
from a wooden hanger against a plain          a quiet lake, its slopes softened by gray    blue sky and the low green hills beyond
white background. The soft fabric, wide       haze and overcast light. Reeds in the        the opposite shore. Reeds and leafy plants
sleeves, and relaxed drape are emphasized     foreground frame the water, adding depth     in the foreground create a natural frame,
by the simple product-style composition       to the subdued scene and strengthening its   while distant buildings add a small human
and uncluttered studio-like setting.          calm, contemplative mood.                    presence to the peaceful view.

Figure 21 Text-to-image examples (Part III). Additional samples covering pastoral, illustrative, product-style, and
lakeside prompts, illustrating the breadth of compositions the model can synthesize.




                                                                  89
Prompt: A roadside viewpoint looks out       Prompt: A cup of foamy coffee sits beside     Prompt: Heavy gray storm clouds gather
across blue ocean water toward distant       fried pastries sprinkled with powdered        low over city buildings, power lines, and
islands and hazy mountains. Green shrubs,    sugar on brown paper. The colorful table      rooftops, filling most of the frame with
tall grasses, a utility pole, and a small    covering, golden snacks, and creamy drink     dark, textured sky. The urban skyline
signboard sit beside the path, giving the    create a cheerful cafe or street-food scene   beneath looks small and tense, suggesting
scenic overlook a casual travel feeling.     focused on sweet comfort food.                an approaching downpour or dramatic
                                                                                           weather change.




Prompt: A large steamed dumpling or          Prompt: A sculptural bust shows a             Prompt: A round celebration cake is
bun rests on a round wooden plate,           realistic human face emerging from a rough,   decorated with pink frosting roses, green
garnished with lettuce and small red         crown-like form and textured reddish base.    leaves, and a large bow across the top.
tomatoes. The bright tabletop, folded        The plain wall behind it keeps focus on the   Plates of food around it suggest a shared
cloth, and clean plating make the simple     unusual material contrast, facial detail,     meal, while the elaborate icing makes the
food presentation look fresh and carefully   and handmade artistic expression.             cake the table’s centerpiece.
arranged.

Figure 22 Text-to-image examples (Part IV). Additional samples on travel, food, urban weather, and sculptural
prompts, showing the model’s handling of varied lighting and textures.




                                                                 90
Prompt: Rows of grilled skewers are              Prompt: Thick steam rises from a large         Prompt: Small white flowers bloom in
packed tightly on a metal tray, coated in        pot filled with stew, meat, vegetables, and    dense clusters among glossy green leaves,
glossy red seasoning, herbs, and chili flakes.   rich brown broth. The close view captures      with many speckled petals and unopened
The close-up view highlights the rich sauce,     active cooking, with bubbling liquid and       buds visible. The close-up composition
charred edges, and spicy street-food appeal      softened ingredients creating a hot, hearty,   emphasizes delicate natural patterns, soft
of the dish.                                     and aromatic kitchen scene.                    lighting, and the fresh detail of the
                                                                                                flowering plant.




Prompt: A brilliant sun shines above a           Prompt: A vivid red boat rests on pale         Prompt: A monochrome mountain
winding mountain road bordered by lush           sand in front of calm turquoise water and a    landscape drawing shows steep ridges, deep
green plants. Distant ridges fade toward         clear horizon. The simple composition          valleys, winding water, and layered terrain
the blue horizon, while the intense light        emphasizes strong color contrast, seaside      fading into the distance. The sketch-like
and clear sky create a bright, open,             quietness, and the unusual stillness of a      shading gives the scene a rugged,
high-altitude travel scene.                      boat pulled ashore.                            hand-rendered quality, as if from a travel
                                                                                                notebook.

Figure 23 Text-to-image examples (Part V). Additional samples on culinary, floral, and travel prompts, together
with an illustrative monochrome landscape.




                                                                     91
Prompt: A young woman in a white coat          Prompt: A man in a gray polo shirt            Prompt: A close selfie shows a man
and patterned scarf stands at night in         stands on a balcony or rooftop, looking       wearing a dark bucket hat and light blue
front of glowing red and yellow lanterns.      thoughtfully to the side. A blurred city      shirt while seated inside a vehicle. Black
The warm lights and wooden structures          skyline behind him gives the portrait an      seats, side windows, and an orange curtain
behind her create a festive, atmospheric       urban setting and a calm, reflective mood.    provide context for the casual travel
portrait with soft contrast.                                                                 setting.




Prompt: An anime-style girl with long          Prompt: A woman with curly hair smiles        Prompt: Two men in gray shirts stand
dark hair wears a purple patterned outfit      gently in a very dark outdoor setting, lit    shoulder to shoulder in front of a green
with lace details, posed against a dreamy      softly from the front. The nearly black       screen. One smiles broadly while the other
sky. Large moonlike circles, stars, and soft   background isolates her face and shoulders,   gives a restrained expression, creating a
gradients create a romantic fantasy            creating an intimate portrait with            casual studio portrait with contrasting
illustration style.                            understated nighttime atmosphere.             personalities.

Figure 24 Text-to-image examples (Part VI). Portrait-focused samples covering nighttime scenes, urban backdrops,
stylized illustration, and studio-style group compositions.




                                                                   92
Prompt: Two adults sit at a restaurant     Prompt: A fluffy white cat sits indoors       Prompt: Two young women in
table behind a birthday cake topped with   with wide dark eyes, pink ears, and crossed   coordinated school-style outfits with red
lit candles. Warm indoor lighting,         front paws. The softly blurred background     bows take a close selfie outdoors. Trees,
surrounding dishes, cups, and desserts     and centered framing make the cat’s round     bicycles, and paved walkways in the
suggest a celebratory meal shared in a     face and clean white fur especially           background suggest a campus or park
relaxed dining setting.                    prominent.                                    setting on a bright day.

Figure 25 Text-to-image examples (Part VII). Final samples on celebratory dining, animal, and outdoor selfie
prompts, completing the qualitative tour of the unified model’s text-to-image generation capability.




                                                               93
Prompt: Describe the content of this           Prompt: Describe the content of this        Prompt: Describe the content of this
image in detail.                               image in detail.                            image in detail.
Caption: The image shows a woman               Caption: The image shows a person           Caption: The image shows a meeting
wearing an oversized long down jacket with     wearing a black jacket and sunglasses,      room with several people, around six to
a plain collar. The jacket features a letter   standing on a street.                       eight, sitting around a long conference
“A” design on it.                                                                          table and holding a meeting. A screen is
                                                                                           mounted on the wall, and bowls, cups, and
                                                                                           pens are placed on the table.




Prompt: Describe the content of this           Prompt: Describe the content of this        Prompt: Describe the content of this
image in detail.                               image in detail.                            image in detail.
Caption: The image shows a person              Caption: The image shows several people     Caption: The image shows a person
wearing a grey hoodie with a purple design     playing a ball game on a court. The         wearing a black suit with a white shirt
on the front. The background includes a        background includes a blue wall with some   underneath, seated.
white wall with a blue design.                 text, and the ground is also blue.

Figure 26 Image×text-to-text examples (Part I). Additional qualitative results under the prompt template Describe
the content of this image in detail.




                                                                   94
Prompt: Describe the content of this        Prompt: Describe the content of this         Prompt: Describe the content of this
image in detail.                            image in detail.                             image in detail.
Caption: The image shows a cluster of       Caption: The image is a cityscape shot       Caption: The image shows a large
ancient buildings with old-style            at dusk, showing an urban night scene        football (soccer) stadium with the field
architecture, featuring several old         with many tall high-rise buildings.          inside, resembling an outdoor stadium.
structures packed closely together.                                                      Another row of stands is visible on the
                                                                                         other side of the field.




Prompt: Describe the content of this        Prompt: What does this image depict?         Prompt: What does this image depict?
image in detail.                            Give a comprehensive description.            Give a comprehensive description.
Caption: The image shows a person           Caption: The image shows a woman             Caption: The image shows a portrait of
using a pen to write on a piece of paper.   with long, dark hair wearing a black top.    a young woman with blonde hair, wearing
                                            The background is a neutral color, with a    a white top and seated at a student desk
                                            window and a pink wall visible behind her.   with an open laptop. She appears to be
                                            She appears to be standing indoors.          speaking while using the laptop.

Figure 27 Image×text-to-text examples (Part II). Additional qualitative results with detailed descriptive prompts
across indoor scenes, cityscapes, and document-oriented imagery.




                                                               95
Prompt: What does this image depict?       Prompt: What is shown in this image?          Prompt: What is shown in this image?
Give a comprehensive description.          Please explain thoroughly.                    Please explain thoroughly.
Caption: The image shows a gravestone      Caption: The image shows a cluster of         Caption: The image is a medium shot of
with English text carved on it.            modern city buildings, including tall         a young East Asian woman with long,
                                           skyscrapers with distinctive and futuristic   straight dark hair, wearing a pink top and
                                           architectural designs.                        a necklace. She is posing with her hands
                                                                                         raised near her face, making a gesture with
                                                                                         her fingers.




Prompt: What is shown in this image?       Prompt: What is shown in this image?          Prompt: What is shown in this image?
Please explain thoroughly.                 Please explain thoroughly.                    Please explain thoroughly.
Caption: The image shows a blue card or    Caption: The image shows three people         Caption: The image shows a mountain
cover with a printed image and some text   standing in an open outdoor area. They        range with a massive peak. The surface of
on it.                                     are wearing colored clothing, and one of      the mountain reveals distinctive textures
                                           them appears to be carrying something on      and patterns.
                                           their back.

Figure 28 Image×text-to-text examples (Part III). Additional qualitative results under more open-ended explanatory
prompts, spanning objects, portraits, outdoor scenes, and landscapes.




                                                               96
Prompt: What is shown in this image?          Prompt: What is shown in this image?          Prompt: What is shown in this image?
Please explain thoroughly.                    Please explain thoroughly.                    Please explain thoroughly.
Caption: The image shows a man                Caption: The image shows a newlywed           Caption: The image is a cartoon-style
wearing a plaid shirt, working in a studio    couple at a wedding, dressed in wedding       drawing of a character dressed in a colored
or workshop. He is focused on an object on    attire.                                       outfit, with a body showing green and
a white surface in front of him.                                                            yellow stripes. The character is touching
                                                                                            its face with one hand.




Prompt: Describe what you see in this         Prompt: Describe what you see in this         Prompt: Describe what you see in this
picture.                                      picture.                                      picture.
Caption: The image shows musicians            Caption: The image shows a black              Caption: The image shows a person
performing with instruments at a ceremony.    sleeveless T-shirt with a printed design on   wearing a white blouse over a black and
The background includes a floor and a wall.   the front.                                    white striped skirt.

Figure 29 Image×text-to-text examples (Part IV). Additional qualitative results over people-centric, object-centric,
and illustration-style inputs.




                                                                  97
Prompt: Describe what you see in this       Prompt: Describe what you see in this        Prompt: Describe what you see in this
picture.                                    picture.                                     picture.
Caption: The image shows a white            Caption: The image shows a man               Caption: The image shows a necklace
Toyota car with a sleek design, featuring   wearing a black baseball cap and a grey      with a red, ball-shaped pendant.
prominent front and side headlights.        shirt. The focus is on the cap, and the
                                            background is a plain, neutral grey.




Prompt: Describe what you see in this       Prompt: Describe what you see in this        Prompt: Describe what you see in this
picture.                                    picture.                                     picture.
Caption: The image shows a person           Caption: The image shows an old              Caption: The image shows a decorative
standing on a paved area with a scenic      building with distinctive traditional        ornament in the shape of a flower.
green backdrop.                             Chinese-style architecture. It features
                                            traditional entrance details, and an iron
                                            structure with a roof is visible above the
                                            building.

Figure 30 Image×text-to-text examples (Part V). Additional qualitative results on apparel, accessories, vehicles,
architecture, and scenic human portraits.




                                                                98
Prompt: Describe what you see in this          Prompt: Generate a detailed caption for         Prompt: Generate a detailed caption for
picture.                                       this image.                                     this image.
Caption: The image shows several people        Caption: The image shows a soccer               Caption: The image shows a black
playing volleyball, with one of them           player in action during a match, wearing a      T-shirt with a design featuring a blue cat
jumping up. The scene takes place under a      numbered uniform. The player appears to         printed on the front.
clear afternoon sky.                           be running and kicking the ball on the field.




Prompt: Generate a detailed caption for        Prompt: Generate a detailed caption for         Prompt: Generate a detailed caption for
this image.                                    this image.                                     this image.
Caption: The image shows a young boy           Caption: The image shows a young girl           Caption: The image shows the stands of
wearing a grey vest over a white shirt, with   with straight hair, wearing a yellow crop       a large soccer stadium. The field and
a blue accent.                                 top and white shorts. She is making a fist      running track are visible, with green grass
                                               with one hand and jumping, set against a        surrounding the stands and a few
                                               blue background.                                spectators present.

Figure 31 Image×text-to-text examples (Part VI). Additional qualitative results for sports, apparel, portrait, and
stadium scenes, including caption-generation prompts.




                                                                    99

