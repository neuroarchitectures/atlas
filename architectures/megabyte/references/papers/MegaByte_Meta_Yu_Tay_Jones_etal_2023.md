# MegaByte Meta Yu Tay Jones etal 2023

> Source: `MegaByte_Meta_Yu_Tay_Jones_etal_2023.pdf`

---

                                                                                                                  Local         Local         Local          Local
                                                                                                                  Model         Model         Model          Model


                                                                                                                _meg           _ by t        _'' t r        _n sf              _



                                                                                                                                  Global Model


                                         M EGA B YTE: Predicting Million-byte Sequences with Multiscale   Transformers
                                                                                             Patch Embedder


                                                                                                                _ _ _ _        me ga         by t e         '' t r a
                                                                                                                                                                               _
                                              Lili Yu * 1 Dániel Simig * 1 Colin Flaherty * 2 Armen Aghajanyan 1 Luke Zettlemoyer 1 Mike Lewis 1


                                                                 Abstract                                            me ga          by t e       t r a n        s f o r
                                              Autoregressive transformers are spectacular mod-




arXiv:2305.07185v1 [cs.LG] 12 May 2023
                                              els for short sequences but scale poorly to long se-                     Local         Local         Local         Local
                                              quences such as high-resolution images, podcasts,                        Model         Model         Model         Model
                                              code, or books. We propose M EGA B YTE, a multi-
                                              scale decoder architecture that enables end-to-end                     _meg           _ by t        _ t r a       _ s f o
                                              differentiable modeling of sequences of over one
                                              million bytes. M EGA B YTE segments sequences
                                              into patches and uses a local submodel within                                             Global Model
                                              patches and a global model between patches. This
                                              enables sub-quadratic self-attention, much larger
                                              feedforward layers for the same compute, and im-                        Patch          Patch         Patch         Patch
                                              proved parallelism during decoding—unlocking                            Embed          Embed         Embed         Embed
                                              better performance at reduced cost for both train-
                                              ing and generation. Extensive experiments show                         _ _ _ _       me ga          by t e         t r an
                                              that M EGA B YTE allows byte-level models to per-
                                              form competitively with subword models on long
                                                                                                        Figure 1. Overview of M EGA B YTE with patch size P = 4. A
                                              context language modeling, achieve state-of-the-
                                                                                                        small local model autoregressively predicts each patch byte-by-
                                              art density estimation on ImageNet, and model             byte, using the output of a larger global model to condition on
                                              audio from raw files. Together, these results estab-      previous patches. Global and Local inputs are padded by P and 1
                                              lish the viability of tokenization-free autoregres-       token respectively to avoid leaking information about future tokens.
                                              sive sequence modeling at scale.


                                                                                                        which simply encodes a patch by losslessly concatenating
                                         1. Introduction                                                embeddings of each byte, (2) a global module, a large au-
                                         Sequences of millions of bytes are ubiquitous; for example,    toregressive transformer that inputs and outputs patch rep-
                                         music, image, or video files typically consist of multiple     resentations and (3) a local module, a small autoregressive
                                         megabytes. However, large transformer decoders (LLMs)          model that predicts bytes within a patch. Crucially, we
                                         typically only use several thousand tokens of context (Brown   observe that for many tasks, most byte predictions are rela-
                                         et al., 2020; Zhang et al., 2022a)—both because of the         tively easy (for example, completing a word given the first
                                         quadratic cost of self-attention but also, more importantly,   few characters), meaning that large networks per-byte are
                                         the cost of large feedforward networks per-position. This      unnecessary, and a much smaller model can be used for
                                         severely limits the set of tasks where LLMs can be applied.    intra-patch modelling.

                                         We introduce M EGA B YTE, a new approach to modeling           The M EGA B YTE architecture gives three major improve-
                                         long byte sequences. First, byte sequences are segmented       ments over Transformers for long sequence modelling:
                                         into fixed-sized patches, loosely analogous to tokens. Our
                                         model then consists of three parts: (1) a patch embedder,        1. Sub-quadratic self-attention Most work on long se-
                                          *
                                                                                                             quence models has focused on mitigating the quadratic
                                            Equal contribution                                               cost of self-attention. M EGA B YTE decomposes long
                                          1
                                            Meta AI.
                                          2
                                            Augment Computing. Work performed while at Meta AI.              sequences into two shorter sequences, and optimal
                                                                                                                                                                4
                                          Correspondence to: Lili Yu <liliyu@meta.com>, Mike Lewis           patch sizes reduces the self-attention cost to O(N 3 ),
                                         <mikelewis@meta.com>.                                               which remains tractable for even long sequences.
                                                                                                          2. Per-patch feedforward layers In GPT3-size mod-
                         M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers

     els, more than 98% of FLOPS are used in comput-               3 components: (1) a patch embedder that inputs a discrete
     ing position-wise feedforward layers. M EGA B YTE             sequence, embeds each element, and chunks it into patches
     uses large feedforward layers per-patch rather than per-      of length P (2) a large global Transformer that contextual-
     position, enabling much larger and more expressive            izes patch representations by performing self-attention over
     models for the same cost. With patch size P , where a         previous patches, and (3) a smaller local Transformer that
     baseline transformer would use the same feedforward           inputs a contextualized patch representation from the global
     layer with m parameters P times, M EGA B YTE can use          model, and autoregressively predict the next patch.
     a layer with mP parameters once for the same cost.
                                                                   2.2. Components
  3. Parallelism in Decoding Transformers must perform
     all computations serially during generation because the       Patch Embedder with patch size of P maps a byte se-
     input to each timestep is the output from the previous        quence x0..T to a sequence of patch embeddings of length
     timestep. By generating representations for patches in        K = PT and dimension P · DG .
     parallel, M EGA B YTE allows greater parallelism during
                                                                   First, each byte is embedded with a lookup table
     generation. For example, a M EGA B YTE model with
                                                                   E global-embed ∈ RV ×DG to an embedding of size DG and
     1.5B parameters can generate sequences 40% faster
                                                                   positional embeddings are added.
     than a standard 350M Transformer, whilst also improv-
     ing perplexity when trained with the same compute.

Together, these improvements allow us to train much larger                hembed
                                                                           t     = Exglobal-embed
                                                                                      t
                                                                                                  + Etpos          t ∈ [0..T ]   (1)
and better-performing models for the same compute budget,
scale to very long sequences, and improve generation speed         Then, byte embeddings are reshaped into a sequence of
during deployment.                                                 K patch embeddings with dimension P · DG . To allow
                                                                   autoregressive modelling, the patch sequence is padded
M EGA B YTE also provides a strong contrast to existing au-
                                                                   to start with a trainable patch-sized padding embedding
toregressive models that typically use some form of tok-
                                                                   (E global-pad ∈ RP ×DG ), and the last patch is removed from
enization, where sequences of bytes are mapped to larger
                                                                   the input. This sequence is the input to the global model,
discrete tokens (Sennrich et al., 2015; Ramesh et al., 2021;
                                                                   and is denoted hglobal-in ∈ RK×(P ·DG ) .
Hsu et al., 2021). Tokenization complicates pre-processing,
                                                                                         (
multi-modal modelling, and transfer to new domains, while                                 E global-pad ,       if k = 0,
                                                                             global-in
hiding useful structure from the model. It also means                     hk           =   embed
                                                                                                                               (2)
                                                                                          h((k−1)·P ):(k·P ) , k ∈ [1, .., K),
that most state-of-the-art models are not truly end to end.
The most widely used approaches to tokenization require
language-specific heuristics (Radford et al., 2019) or lose        Global Model is a decoder-only Transformer with dimen-
information (Ramesh et al., 2021). Replacing tokenization          sion P · DG that operates on a sequence of K patches. It in-
with efficient and performant byte models would therefore          corporates a self-attention mechanism and causal masking to
have many advantages.                                              capture dependencies between patches. It inputs a sequence
                                                                   of K patch representations hglobal-in
                                                                                                 0:K     , and outputs an updated
We conduct extensive experiments for both M EGA B YTE
                                                                   representation hglobal-out
                                                                                     0:K      by performing    self-attention over
and strong baselines. We use a fixed compute and data bud-
                                                                   previous patches.
get across all models to focus our comparisons solely on
the model architecture rather than training resources, which
are known to benefit all models. We find that M EGA B YTE
allows byte-level models to perform competitively with sub-                   hglobal-out
                                                                               0:K        = transformerglobal (hglobal-in
                                                                                                                0:K       )      (3)
word models on long context language modeling, achieve
state-of-the-art perplexities for density estimation on Im-        The output of the final global layer hglobal
                                                                                                         0:K contains K patch
ageNet, and allow audio modelling from raw audio files.            representations of dimension P · DG . For each of these, we
Together, these results establish the viability of tokenization-   reshape them into sequences of length P and dimension DG ,
free autoregressive sequence modeling at scale.                    where position p uses dimensions p · DG to (p + 1) · DG .
                                                                   Each position is then projected to the dimension of the local
2. M EGA B YTE Transformer                                         model with a matrix wGL ∈ RDG ×DL where DL is the
                                                                   local model dimension. We then combine these with byte
2.1. Overview                                                      embeddings of size DL for the tokens in the next patch
M EGA B YTE is an autoregressive model for efficiently mod-        Exlocal-embed
                                                                       (k·P +p−1)
                                                                                  . The local byte embeddings is offset by one
eling long input sequences. M EGA B YTE is comprised of            with a trainable local padding embedding (E local-pad ∈ RDL )
                               M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers


hembed
 t             = Exglobal-embed
                    t
                                + Etpos                                                           t ∈ [0..T ), E global-embed ∈ RV ×DG ,
                                                                                                   E pos ∈ RT ×DG , hembed ∈ RT ×DG
                   (
                       E global-pad ,       if k = 0,                                                                                T
hglobal-in
 k             =        embed
                                                                                                       E global-pad ∈ RP ×DG , K =
                       h((k−1)·P ):(k·P ) , k ∈ [1, .., K),                                                                          P
hglobal-out
 0:K        = transformerglobal (hglobal-in
                                  0:K       )                                           hglobal-out ∈ RK×P ·DG , hglobal-in ∈ RK×P ·DG
                                                   (
                                                       E local-pad ,  if p = 0                    E local-pad ∈ RDL , wGL ∈ RDG ×DL
hlocal-in      = wGL hglobal-out
                      k,(p·DG ):((p+1)·DG ) +
 k,p                                                     local-embed
                                                       Ex(k·P +p−1) , p ∈ [1, .., P )                            E local-embed ∈ RV ×DL
hlocal-out
 k,0:P         = transformerlocal (hlocal-in
                                    k,0:P )                                                    hlocal-in
                                                                                                k,p      ∈ RDL , hlocal-out ∈ RK×P ·DL
p(xt |x0:t ) = softmax(E local-embed hlocal-out
                                      k,p       )x                                                                      t=k·P +p
                                                     t


Figure 2. Summary of M EGA B YTE with vocabulary V , sequence length T , global and local dimensions DG and DL , and K patches of
size P . Transformer layers use masked self attention to not observe information from future timesteps.


to allow autoregressive modelling within a patch. This                      bytes before they are chunked into patches. We use a stack
results in a tensor hlocal-in ∈ RK×P ×DL .                                  of convolutional layers, with filter sizes of 3, 5 and 7.

   hlocal-in
    k,p      = wGL hglobal-out               local-embed
                    k,(p·DG ):((p+1)·DG ) + Ex(k·P +p−1)             (4)    2.3.2. C ROSS - PATCH ATTENTION
                                                                            The Local model uses short sequences for efficiency, and
Local Model is a smaller decoder-only Transformer of di-
                                                                            relies on the Global model for long-range information. How-
mension DL that operates on a single patch k containing
                                                                            ever, we can increase the context of the Local model with
P elements, each of which is the sum of an output from
                                                                            little overhead by allowing it to condition on r elements
the global model and an embedding of the previous byte
                                                                            from the previous patch. This approach allows the Global
in the sequence. K copies of the local models are run on
                                                                            model to focus on a longer-range context. Specifically, when
each patch independently (and in parallel during training),
                                                                            computing self-attention in each layer, we concatenate the
computing a representation hlocal-out ∈ RK×P ·DL .
                                                                            keys and values with the last r keys and queries from the pre-
                                                                            vious patch. We use rotary embeddings (Su et al., 2021) to
                                     local local-in
                                                                            model relative positions between elements in the sequence.
                 hlocal-out
                  k,0:P = transformer     (hk,0:P )                  (5)    This approach is reminiscent of TransformerXL (Dai et al.,
                                                                            2019) but differs by being fully differentiable.
Finally, we can compute the probability distribution over
the vocabulary at each position. The pth element of the kth                 2.3.3. S TRIDED I NFERENCE
patch corresponds to element t of the complete sequence,
where t = k · P + p:                                                        We observed empirically that the per-token loss within each
                                                                            patch would increase towards the end of the patch, as the
            p(xt |x0:t ) = softmax(E local-embed hlocal-out
                                                  k,p       )x       (6)    prediction relies more on the weaker Local model. To al-
                                                                 t
                                                                            leviate this issue, we propose strided inference, in which
                                                                            we predict the sequence with two forward passes of the full
2.3. Variations and Extensions
                                                                            model, whose inputs are offset by p/2 positions from each
We experiment with several extensions of M EGA B YTE.                       other. We then combine the first p/2 positions in each patch
                                                                            for our predictions to predict the complete sequence. Simi-
2.3.1. C ONVOLUTIONAL PATCH E NCODER                                        larly to sliding window techniques (Press et al., 2020), this
                                                                            approach doubles the cost of inference but improves results.
One limitation of chunking sequences into patches is that it
is not translation invariant, and byte sequences may receive
                                                                            2.4. Motivation
a different representation depending on their position in
the patch. This may mean, for example, that a model has                     Having described the model, we briefly discuss the motiva-
to relearn the meaning of a word at different offsets. To                   tion behind some of the architectural choices.
mitigate this issue, we experimented with augmenting the
Patch Embedder with causal convolutional layers, which                      Why is the local model needed? Many of the efficiency
allow translation-invariant contextual representations of the               advantages of the M EGA B YTE design could be realized
                         M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers

with the Global model alone, which would resemble a de-
coder version of ViT (Dosovitskiy et al., 2020). However,
the joint distribution over the patch p(xt+1 , .., xt+P |x0..t )
has an output space of size 256P so direct modeling is only
tractable for very small patches. We could instead factor
the joint distribution into conditionally independent distri-
butions p(xt+1 |x0..t )..p(xt+P |x0..t ), but this would greatly
limit the model’s expressive power. For example, it would
be unable to express a patch distribution such as 50% cat and
50% dog, and would instead have to assign probability mass
to strings such as cag and dot. Instead, our autoregressive
Local model conditions on previous characters within the
patch, allowing it to only assign probability to the desired
strings.
Increasing Parameters for Fixed Compute Transformer
models have shown consistent improvements with parameter
counts (Kaplan et al., 2020). However, the size of models is
limited by their increasing computational cost. M EGA B YTE        Figure 3. Computational cost (FLOPS/token) for different model
allows larger models for the same cost, both by making             architectures at different scales. M EGA B YTE architectures (here
self attention sub-quadratic, and by using large feedforward       with P = 8) use less FLOPS than equivalently sized Transformers
                                                                   and Linear Transformers (Katharopoulos et al., 2020) across a
layers across patches rather than individual tokens.
                                                                   wide range of model sizes and sequence lengths, allowing larger
Re-use of Established Components M EGA B YTE consists              models to be used for the same computational cost.
of two transformer models interleaved with shifting, re-
shaping and a linear projection. This re-use increases the
                                                                   Feedforward Layers However, attention is not the main
likelihood that the architecture will inherit the desirable
                                                                   cost in large transformers. Instead of increasing the se-
scaling properties of transformers.
                                                                   quence length, transformers are more commonly scaled by
                                                                   increasing the dimension of their latent state d, and the feed-
3. Efficiency Analysis                                             forward network cost dominates the model’s overall cost
                                                                   (Kaplan et al., 2020). For example, in the GPT3 architec-
3.1. Training Efficiency
                                                                   ture, the quadratic self-attention computation accounts for
We analyze the cost of different architectures when scaling        only 1.4% of FLOPS. Following the approximation of (Ka-
both the sequence length and size of the models.                   plan et al., 2020), a forward pass with a large transformer
                                                                   with m non-embedding parameters on a sequence of length
Attention The cost of the self attention in a transformer          T uses roughly 2mT FLOPS. M EGA B YTE contains two
architecture for a sequence of length T has O(T 2 ) com-           transformers: the Global model uses mg parameters on a se-
plexity. Much work has been explored reducing this; for            quence of length PT , and a Local model with ml parameters
example, Sparse Transformers (Child et al., 2019) and Rout-        that sees PT sequences of length P , giving an estimate of
                                                                        m
ing Transformers (Roy et al., 2020) show strong results with       2T ( Pg + ml ) FLOPS. When mg  ml , the FLOPS used
                     3
a complexity O(T 2 ). Numerous linear attention mecha-                                                 2T m
                                                                   by M EGA B YTE is approximately P g , allowing a model
nisms have also been proposed (Katharopoulos et al., 2020;         P times larger than a transformer with equivalent FLOPS.
Choromanski et al., 2020), although we are not aware of            This analysis holds irrespective of any efficient attention
competitive results on large scale language modeling tasks.        mechanisms used in the transformer.
As a function of sequence length T and patch size P , the
                                                             2
Global model has a sequence of length PT so uses O( PT 2 )         Combined Analysis To understand efficiency at differ-
operations, and the Local model uses PT sequences of length        ent sequence lengths and model sizes, we calculate the
                   2
P so uses O( TPP ) = O(P T ) operations. The overall cost          total FLOPS used by transformers, Linear Transformers
                                     2
of M EGA B YTE is therefore in O( PT 2 +T P ). P is a hyperpa-     and M EGA B YTE. For each operation, we use FLOP esti-
rameter that is chosen to create an architecture for sequences     mates from (Kaplan et al., 2020), except for attention in
                               1
of size T . By setting P = T 3 the complexity is in O(T 3 ).
                                                            4      Linear Transformers, which we estimate as 9D FLOPS/-
                                          1                        token1 , where D is the model embedding dimension. Fig-
Using much shorter patches of P = T 5 would give a com-
                 8
plexity of O(T 5 ). The cost is less than the transformer for         1
                                                                        This may underestimate the time taken by Linear Transformer
all non-trivial values of P such that 1 < P < T .                  decoders, which use a recurrence mechanism that is harder to
                          M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers

ure 3 shows that for models of size 660M to 173B and se-          loader, preprocessing step, and trainer to avoid any artifacts
quence lengths of up to 1M tokens, M EGA B YTE with P = 8         in our compute-controlled experiments.
uses less FLOPS than either transformers or Linear Trans-
formers. Baseline model architectures are based on GPT3,          4.3. Training Procedure
and Megabyte global/local model sizes are 452M/151M,
5.8B/604M, 170B/3.2B respectively.                                All models were trained using the Metaseq2 code
                                                                  base (Zhang et al., 2022b). The training used the PyTorch
                                                                  framework (Paszke et al., 2019), with fairscale to improve
3.2. Generation Efficiency
                                                                  memory efficiency through fully sharded model and opti-
Generating long sequences with transformers is slow, be-          mizer states (Baines et al., 2021). Mixed precision training
cause the input to each timestep is the output from the pre-      was used to improve training efficiency at scale (Micikevi-
vious timestep, meaning each layer must be computed for           cius et al., 2017). More training details and various model
each token serially. As running a layer on a single token typ-    parameters can be found in Section A.1 in the Appendix.
ically does not saturate the amount of parallelism available
                                                                  To validate our implementation of PerceiverAR, we repro-
within a GPU, for analysis, we model each layer as a con-
                                                                  duced their experiments on downsized ImageNet at 64 pix-
stant cost independently of size. Consider a M EGA B YTE
                                                                  els. By carefully matching hyperparameters, we achieved a
model with Lglobal layers in the Global model and Llocal lay-
                                                                  bits per byte (bpb) score of 3.53, compared to the reported
ers in the Local model and patch size P , compared with a
                                                                  3.54 in the original paper.
Transformer architecture with Llocal + Lglobal layers. Gener-
ating each patch with M EGA B YTE requires a sequence of
O(Lglobal + P · Llocal ) serial operations, whereas the Trans-    4.4. Inference Methods
former requires O(P · Lglobal + P · Llocal ) serial operations.   Several techniques have been proposed for trading off speed
When Lglobal  Llocal (i.e. the Global model has many             for performance during inference with language models, in-
more layers than the Local model), M EGA B YTE can reduce         cluding sliding windows (Press et al., 2020) and our strided
inference costs by a factor close to P .                          inference (Section 2.3.3). We only use these methods when
                                                                  comparing with prior published work (Tables 8 and 4).
4. Experimental setup
4.1. Controlling for Compute and Data
                                                                  5. Language Modeling

Models show consistent improvements when increasing               We evaluated the performance of M EGA B YTE on language
both data and compute (Kaplan et al., 2020; Hoffmann et al.,      modeling on a set of 5 diverse datasets emphasizing long-
2022), meaning that one model can outperform another be-          range dependencies: Project Gutenberg (PG-19), Books,
cause of an increased training budget instead of an improved      Stories, arXiv, and Code.
architecture. However, in practice, both compute and data         Datasets We experiment on a range of long form text
are typically limited. We conduct experiments using a fixed       datasets. The PG-19 dataset (Rae et al., 2019b) consists
compute and data budget across all models to focus compar-        of English-language books written before 1919 and is ex-
isons solely on the model architecture rather than training       tracted from the Project Gutenberg online library. The Sto-
resources. To achieve this, we adjust model hyperparame-          ries dataset (Trinh & Le, 2018) is a subset of CommonCrawl
ters (mainly, number of layers) within each architecture so       data meant to emulate Winograd schemas. Books (Gao et al.,
that the forward pass time taken per byte is matched, and         2020) is another collection of English-language books. The
then train all models for the same number of bytes.               arXiv dataset is a collection of technical publications written
                                                                  in LATEX from the arXiv online archive. Finally, the Code
4.2. Comparison Systems                                           dataset is a large publicly available dataset of open source
                                                                  code, under Apache, BSD or MIT licenses. More details on
We compare M EGA B YTE with both a standard decoder-
                                                                  dataset sizes and document lengths are shared in Table 6.
only Transformer and PerceiverAR (Hawthorne et al., 2022).
PerceiverAR extends the original transformer with a single        Controlled Experiments Table 7, lists bpb on each dataset.
cross-attention layer over a much longer context sequence,        Each model is trained for 80 billion bytes, and models
and is the best performing general purpose autoregressive         are scaled to use the same compute budget. We carefully
model we are aware of and achieves state-of-the-art results       tune hyperparameters for all architectures to best utilize the
across several modalities. We implemented both models             available compute budget. M EGA B YTE consistently outper-
in the same codebase, and all models share a similar data         forms both baseline transformers and PerceiverAR across all
                                                                     2
parallelize on current hardware.                                         https://github.com/facebookresearch/metaseq
                            M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers

       Dataset    Total Bytes     Mean document size (bytes)           three different resolutions of images, ranging from 64×64 to
       PG-19         10.1GB                          411,404           640×640 pixels – the latter requiring the effective modeling
       Stories       21.3GB                           35,265           of sequences with over 1.2M tokens. This generation task
       Books         79.7GB                          509,526           becomes increasingly challenging as the image’s resolution
       arXiv         91.5GB                           58,518           grows: doing well on this task requires the modeling of
       Code         353.7GB                            7,461           local patterns (textures, lines, etc.) and long-range context
                                                                       that provides information about the high level structure of
       Table 1. Text dataset sizes and mean document lengths.
                                                                       the image. Inspired by recent works in Vision Transform-
                                                                       ers (Dosovitskiy et al., 2020), we model image data patch
                    PG-19       Stories   Books   arXiv    Code        by patch (more details can be found in Appendix D.1).
  Transformer       1.057       1.064     1.097   0.816    0.575
  PerceiverAR       1.104       1.070     1.104   0.791    0.546       6.2. Comparison with State of the Art
  M EGA B YTE       1.000       0.978     1.007   0.678    0.411
                                                                       We train a large M EGA B YTE model on ImageNet 64x64
Table 2. Performance (bits-per-byte) of compute and data con-          with Global and Local models sized 2.7B and 350M parame-
trolled M EGA B YTE, PerceiverAR, and Transformer models on            ters, respectively, for 1.4T tokens. We estimate that training
various text modalities.                                               this model consumed less than half the GPU hours we would
                                                                       have needed to reproduce the best PerceiverAR model de-
                                                                       scribed by (Hawthorne et al., 2022). As shown in Table 8,
datasets. We use the same sets of parameters on all datasest.          M EGA B YTE matches the state-of-the-art performance of
In all experiments presented in Table 7, transformer has size          PerceiverAR whilst using only half the compute.
of 320M with context length of 1024, PerceiverAR has size
of 248M with context size of 8192 and latent size of 1024,             6.3. Scaling to higher resolutions
and M EGA B YTE global/local model sizes are 758M/262M
with context length of 8192 and patch size of 8.                       We compare three transformer variants (vanilla, Per-
                                                                       ceiverAR, M EGA B YTE) to test scalability to long sequences
Scaling Experiment We scale up our training data on PG-
                                                                       on increasingly large image resolutions. We use our own
19 (Table 8), and compare M EGA B YTE with byte baselines,
                                                                       implementations of these in the same framework and budget
as well as converting all results to word-level perplexities to
                                                                       the same amount of GPU hours and data to train each of
benchmark with state-of-art token based models.
                                                                       these model variants.
We train a byte-level Transformer, PerceiverAR and
                                                                       M EGA B YTE is able to handle all sequence lengths with a
M EGA B YTE models for 400B bytes and the same compute
                                                                       single forward pass of up to 1.2M tokens. We found nei-
budget using same model parameters as in the controlled
                                                                       ther the standard Transformer nor PerceiverAR could model
experiments. We find that M EGA B YTE outperforms other
                                                                       such long sequences at a reasonable model size, so instead
byte-level models by a wide margin at this scale.3
                                                                       we split images into segments of size 1024 and 12000 re-
We also compare with the best previously reported numbers              spectively. For Megabyte, we set patch size as 12 for Im-
for sub-word models. These results may be confounded by                age64 and patch size as 192 for Image256 and Image640
differing amounts of compute and tuning used, but show                 datasets. Model sizes are adjusted to match overall training
that M EGA B YTE gives results competitive with state-of-the-          speeds across models and we do not use any form of sliding
art models trained on subwords. These results suggest that             window evaluation in this experiment. As seen in Table 5,
M EGA B YTE may allow future large language models to be               M EGA B YTE outperforms baselines across all resolutions in
tokenization-free.                                                     this compute-controlled setting. The precise settings used
                                                                       for each of the baseline models such as context length and
                                                                       number of latents are summarized in Table 14.
6. Image Modeling
                                                                       Results show that M EGA B YTE outperforms the other sys-
6.1. Sequence Modeling on ImageNet
                                                                       tems at all resolutions, demonstrating an effective model of
We test M EGA B YTE on variants of the autoregressive image            sequences of over 1M bytes.
generation task on ImageNet (Oord et al., 2016), to mea-
sure its ability to efficiently use long context. We test on           7. Language Modeling
   3
     The only prior byte-level experiments we are aware of are         We evaluated the performance of M EGA B YTE on language
at a smaller scale in Hutchins et al. (2022), who report results
equivalent to test perplexities of 46.5 with a version of the Block-   modeling on a set of 5 diverse datasets emphasizing long-
Recurrent transformer, and 49.5 with Memorizing Transformers           range dependencies: Project Gutenberg (PG-19), Books,
(Wu et al., 2022), compared to 36.4 with our model.                    Stories, arXiv, and Code.
                          M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers

                                                       Tokenizer       Vocab Size            Context Length           Validation    Test
  TransformerXL (Rae et al., 2019a)                 SentencePiece         32k             512+1024 (subwords)            45.5       36.3
  CompressiveTransformer (Rae et al., 2019a)        SentencePiece         32k          512+512+2x512 (subwords)          43.4       33.6
  PerceiverAR (Hawthorne et al., 2022)              SentencePiece         32k               2048 (subwords)              45.9       28.9
  BlockRecurrent (Hutchins et al., 2022)            SentencePiece         32k          1024+recurrence (subwords)         -         26.5
  Transformer byte-level (ours)                         Bytes             256                  2048 (bytes)              81.6       69.4
  PerceiverAR byte-level (ours)                         Bytes             256                  8192 (bytes)             119.1       88.8
  M EGA B YTE                                           Bytes             256                  8192 (bytes)              42.8       36.4

Table 3. Larger scale experiments on PG19, converting bits-per-byte to word-level perplexities for comparison with prior work. Results
below the line are compute-matched. M EGA B YTE outperforms other byte models by a wide margin, and gives results competitive with
state-of-the-art models trained on subwords.


       ImageNet64                                       bpb                     Dataset    Total Bytes   Mean document size (bytes)

       Routing Transformer (Roy et al., 2020)          3.43                     PG-19         10.1GB                         411,404
                                                                                Stories       21.3GB                          35,265
       Combiner (Ren et al., 2021)                     3.42                     Books         79.7GB                         509,526
       Perceiver AR (Hawthorne et al., 2022)           3.40                     arXiv         91.5GB                          58,518
       M EGA B YTE                                     3.40                     Code         353.7GB                           7,461

Table 4. Bits per byte (bpb) on ImageNet 64×64. M EGA B YTE                     Table 6. Text dataset sizes and mean document lengths.
matches the current state-of-the-art while only using half the
amount of GPU hours to train.

                   Context    Image64      Image256      Image640
 Total len                       12288       196608       1228800
 Transformer         1024          3.62        3.801          2.847
 Perceiver AR       12000          3.55        3.373          2.345     In all experiments presented in Table 7, transformer has size
 M EGA B YTE           Full        3.52        3.158          2.282     of 320M with context length of 1024, PerceiverAR has size
                                                                        of 248M with context size of 8192 and latent size of 1024,
Table 5. Bits per byte (bpb) on ImageNet with different resolutions.    and M EGA B YTE global/local model sizes are 758M/262M
All models use the same compute and data. MEGABYTE scales               with context length of 8192 and patch size of 8.
well to sequences of over 1M tokens.
                                                                        Scaling Experiment We scale up our training data on PG-
                                                                        19 (Table 8), and compare M EGA B YTE with byte baselines,
Datasets We experiment on a range of long form text                     as well as converting all results to word-level perplexities to
datasets. The PG-19 dataset (Rae et al., 2019b) consists                benchmark with state-of-art token based models.
of English-language books written before 1919 and is ex-                We train a byte-level Transformer, PerceiverAR and
tracted from the Project Gutenberg online library. The Sto-             M EGA B YTE models for 400B bytes and the same compute
ries dataset (Trinh & Le, 2018) is a subset of CommonCrawl              budget using same model parameters as in the controlled
data meant to emulate Winograd schemas. Books (Gao et al.,              experiments. We find that M EGA B YTE outperforms other
2020) is another collection of English-language books. The              byte-level models by a wide margin at this scale.4
arXiv dataset is a collection of technical publications written
in LATEX from the arXiv online archive. Finally, the Code               We also compare with the best previously reported numbers
dataset is a large publicly available dataset of open source            for sub-word models. These results may be confounded by
code, under Apache, BSD or MIT licenses. More details on                differing amounts of compute and tuning used, but show
dataset sizes and document lengths are shared in Table 6.               that M EGA B YTE gives results competitive with state-of-the-
                                                                        art models trained on subwords. These results suggest that
Controlled Experiments Table 7, lists bpb on each dataset.              M EGA B YTE may allow future large language models to be
Each model is trained for 80 billion bytes, and models                  tokenization-free.
are scaled to use the same compute budget. We carefully                     4
tune hyperparameters for all architectures to best utilize the               The only prior byte-level experiments we are aware of are
                                                                        at a smaller scale in Hutchins et al. (2022), who report results
available compute budget. M EGA B YTE consistently outper-              equivalent to test perplexities of 46.5 with a version of the Block-
forms both baseline transformers and PerceiverAR across all             Recurrent transformer, and 49.5 with Memorizing Transformers
datasets. We use the same sets of parameters on all datasest.           (Wu et al., 2022), compared to 36.4 with our model.
                          M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers

                 PG-19      Stories   Books   arXiv   Code
  Transformer     1.057     1.064     1.097   0.816   0.575
  PerceiverAR     1.104     1.070     1.104   0.791   0.546
  M EGA B YTE     1.000     0.978     1.007   0.678   0.411

Table 7. Performance (bits-per-byte) of compute and data con-
trolled M EGA B YTE, PerceiverAR, and Transformer models on
various text modalities.


8. Audio Modeling
Audio has aspects of both the sequential structure of text       Figure 4. Average log probability assigned to the token at different
                                                                 positions within the context length by M EGA B YTE model with
and the continuous nature of images, so is an interesting
                                                                 8192 context size and by a vanilla transformer model trained using
application for M EGA B YTE.                                     the same compute (PG19 test set). M EGA B YTE likelihoods rise
Raw audio is typically stored as a sequence of 16-bit integer    throughout its context window, demonstrating that it can use tokens
values (one per timestep); a softmax layer would need to         from 8k bytes previously to improve its predictions.
output 65,536 probabilities per timestep to model all possi-
ble values. To address this issue, various techniques have
                                                                 perplexity as expected. However, M EGA B YTE also gener-
been developed to reduce the memory and computational re-
                                                                 ates a sequence of 8192 tokens 40% faster than transformer,
quirements of the softmax layer. For instance, van den Oord
                                                                 despite having over 4 times the parameters. This speed up is
et al. (2016) apply µ-law companding transformation and
                                                                 due to the bulk of the parameters being in the Global model,
quantizes the input into 256 possible values. Alternatively,
                                                                 which only needs to be computed once for every 8 tokens,
van den Oord et al. (2017) model the samples using the
                                                                 whereas all the parameters in the baseline model are used
discretized mixture of logistics distribution introduced by
                                                                 on every token.
Salimans et al. (2017). Finally, Kalchbrenner et al. (2018)
use a dual softmax technique to produce 8 coarse and 8 fine
bits. In our approach, we simplify the audio modeling pro-       9.2. Model Components
cess by directly reading the bytes (256 possible values) from    In Table 10, we analyze the significance of different com-
the audio file and conducting an autoregressive language         ponents in the M EGA B YTE architecture by studying arXiv,
model on top of that. This greatly streamlines the modeling      Librilight-L and ImageNet256 datasets. Removing Local
process, making it easier and more efficient.                    (w/o local model) or global (w/o global model) model, we
Our audio modeling approach focuses on 16 kHz, 16-bit            observe a substantial increase in bpb on all datasets, showing
audio, which equates to 32k bytes per one-second clip. We        that both parts are crucial. The performance of the model
use an extensive audio dataset consisting of 2 terabytes         without the cross-patch local model (w/o cross-patch local
(roughly 18,000 hours) of audio. We use a sequence length        model) is competitive, indicating that the architecture is ro-
of 524,288, a patch size of 32, and a batch size of 32 to        bust to this modification. We observe slight improvement on
facilitate model training. By utilizing these settings, we can   the Librilight-L and ImageNet256 datasets by augmenting
effectively train our model on large volumes of audio data,      the M EGA B YTE model with a CNN encoder (w/ CNN en-
helping to improve its accuracy and efficacy.                    coder). This suggests that the M EGA B YTE architecture can
                                                                 benefit from integrating alternative encoding mechanisms.
Our model obtains bpb of 3.477, much lower than the results
with perceiverAR (3.543) and vanilla transformer model           9.3. Effective Use of Context
(3.567). More ablation results are presented in Table 10.
                                                                 Long-context models often struggle to benefit from the full
                                                                 context (Sun et al., 2021). Figure 4 shows that later tokens
9. Analysis
                                                                 within each context window consistently have a higher like-
9.1. Generation speed                                            lihood, indicating that M EGA B YTE can effectively use at
                                                                 least 8k bytes of context on the PG19 dataset.
We also compare the text generation speed between
M EGA B YTE and a transformer. We compare a 350M pa-
                                                                 9.4. Strided Inference
rameter baseline transfomer and a M EGA B YTE model with
a 1.3B parameter Global model and a 218M parameter local         We find that within a single patch, on average, the
model, trained on PG19 with equal compute. As shown              M EGA B YTE performs worse on later tokens within a patch
in Table 9, the M EGA B YTE model achieves much lower            (see Figure 5). Section 2.3.3 proposes strided inference as
                          M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers

                                                      Tokenizer        Vocab Size           Context Length          Validation          Test
  TransformerXL (Rae et al., 2019a)                 SentencePiece         32k            512+1024 (subwords)              45.5          36.3
  CompressiveTransformer (Rae et al., 2019a)        SentencePiece         32k         512+512+2x512 (subwords)            43.4          33.6
  PerceiverAR (Hawthorne et al., 2022)              SentencePiece         32k              2048 (subwords)                45.9          28.9
  BlockRecurrent (Hutchins et al., 2022)            SentencePiece         32k         1024+recurrence (subwords)           -            26.5
  Transformer byte-level (ours)                         Bytes             256                2048 (bytes)               81.6            69.4
  PerceiverAR byte-level (ours)                         Bytes             256                8192 (bytes)              119.1            88.8
  M EGA B YTE                                           Bytes             256                8192 (bytes)               42.8            36.4

Table 8. Larger scale experiments on PG19, converting bits-per-byte to word-level perplexities for comparison with prior work. Results
below the line are compute-matched. M EGA B YTE outperforms other byte models by a wide margin, and gives results competitive with
state-of-the-art models trained on subwords.


                     Global    (Local)             Generation
                                           bpb
                      Size      Size                Time (s)
    Transformer         -       350M      1.064        132
    M EGA B YTE       1.3B      218M      0.991        93

Table 9. Comparison of bits per byte (bpb) and generation speed
of 8192 bytes of transformer model (with context length 1024) and
                                                                        Figure 5. An illustration of strided inference with patch size 8.
M EGA B YTE with context length 8192 and patch size 8.
                                                                        Lines below the text represent the patches used in the two rounds
                                                                        of inference, the plot above it represents the average probability
                                 Arxiv     Audio     ImageNet256        assigned to the token at a given position within a patch. By con-
                                                                        sidering only the first half of each patch from the two rounds of
 M EGA B YTE                    0.6871     3.477             3.158
                                                                        inference and combining them (bold lines on top), we achieve a
  w/o local model                1.263     5.955             4.768
  w/o global model               1.373     3.659             3.181      better overall bpb.
  w/o cross-patch attention     0.6781     3.481             3.259
  w/ CNN encoder                0.6871     3.475             3.155               Method                  Inference Cost          bpb
                                                                                 Basic Inference              1X            0.9079
Table 10. Ablation of M EGA B YTE model components, showing                       w/ Sliding Window           2X            0.8918
that both Local and Global models are critical to strong perfor-                  w/ Strided Inference        2X            0.8926
mance, but the architecture is robust to other modifications. We                  w/ Sliding & Strided        4X            0.8751
report bits-per-byte on text, audio, and image prediction tasks. All
models within a column are trained using the same compute and           Table 11. Performance of various inference techniques on the
data. The hyperparameters are listed in Table 14.                       PG19 test set using our best M EGA B YTE model.


a solution, where two forward passes are performed offset               can be different across modalities.
by P2 tokens, and results from the first half of each patch
                                                                           Patch Size     Global Size        Local Size                bpb
are combined. Table 11 shows performance improvements
from strided inference, which are additive with the standard                     48         125M         114M (D=768, L=11)            3.178
                                                                                192         125M         125M (D=768, L=12)            3.158
sliding window.                                                                 768         125M          83M (D=768, L=8)             3.186

9.5. Hyperparameters                                                    Table 12. Effects of patch size on performance on the Image256
                                                                        dataset. All versions use the same amount of GPU hours and data.
M EGA B YTE introduces several additional hyperparame-
ters. We tuned these parameters independently for different
                                                                        Local to Global model Size Ratio. We experimented with
modalities and reported performance based on the best set-
                                                                        different Local/Global model size ratios on PG19 dataset.
ting we found. All experiments in the same group use the
                                                                        By grouping bytes into patches, M EGA B YTE effectively
same compute.
                                                                        uses P times less tokens for the Global model as on the
Patch Size. We experimented with various patch sizes on                 Local model—enabling us to increase the size of the Global
Image256 dataset and found that there is a wide range of                model without reduced cost. We find that a given compute
values where M EGA B YTE performs similarly. We found                   budget is spent optimally when the Global model has more
similar robustness against the choice of this hyperparameter            parameters than the Local model. This trend was consistent
across all modalities, although the optimal patch size itself           across all modalities and various patch sizes.
                         M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers

         Global Size              Local Size           bpb         timestep.
   350M (D=1024,L=24)        290M (D=1024,L=20)       1.014        Tokenization The most common approach to shortening
   760M (D=1536,L=24)        262M (D=1024,L=18)       1.002        sequence lengths in Transformer decoders is to pre-process
   1.3B (D=2048,L=24)        218M (D=1024,L=15)       0.991
                                                                   the input with a form of tokenization, in which multiple
Table 13. Effects of Local / Global model size on performance      bytes are mapped to a single discrete token from a fixed
on the PG19 dataset. Increasing the capacity of global model       vocabulary. For text, this can be done losslessly using meth-
improves performance. Models are compute and data matched.         ods such as BPE (Sennrich et al., 2015) and SentencePiece
                                                                   (Kudo & Richardson, 2018), but these approaches can re-
                                                                   quire language-specific heuristics (Radford et al., 2019),
10. Related Work                                                   limit out-of-domain performance (Sharami et al., 2023),
                                                                   and can affect prompting and truncated sampling in unpre-
Prior research has explored the possibility of improving the       dictable ways.5 The amount of high-frequency information
efficiency of Transformers on long sequences, primarily            in images and audio means that tokenization cannot be per-
motivated by mitigating the quadratic cost of self-attention.      formed losslessly, and instead clustering (Hsu et al., 2021)
Efficient Encoder Models Several related techniques to             or discrete auto-encoders (Ramesh et al., 2021) are used to
ours have been developed for transformer encoder architec-         compress the inputs, which lose information and likely limit
tures but cannot be straightforwardly applied to decoders.         generative model performance. Our patches are analogous
In particular, patchifying operations have previously been         to traditional lossless tokens, and the Local model performs
used in image encoder models such as ViT (Dosovitskiy              the role of mapping a hidden state to a distribution over
et al., 2020), and down- and up-sampling operations have           possible patches.
been used for text encoders (Clark et al., 2022), but such
methods cannot be naively applied to decoder-only mod-             11. Conclusion
els without leaking information to future bytes in the same
patch. M EGA B YTE generalizes these approaches to an effi-        We introduced M EGA B YTE, a scaleable architecture for
cient decoder model by using a intra-patch transformer to          modeling long sequences. M EGA B YTE outperforms exist-
predict each sequence element’s likelihood, and offseting          ing byte-level models across a range of tasks and modalities,
the inputs to the two models to avoid leaking information.         allowing large models of sequences of over 1 million to-
Jaegle et al. (2021) use self-attention on a shorter latent        kens. It also gives competitive language modeling results
sequence also resembles patchification, but this technique         with subword models, which may allow byte-level models
cannot easily be applied to decoder architectures without          to replace tokenization. However, the scale of experiments
leaking information to future timesteps.                           here is far below those of state-of-the-art language models
                                                                   (Brown et al., 2020), and future work should explore scaling
Efficient Decoder models Improving the efficiency of de-           M EGA B YTE to much larger models and datasets.
coder models is more challenging because of the need to
make one prediction per timestep, and not leak information
to future timesteps. The most popular approaches can be cat-
                                                                   References
egorized as (1) chunking sequences into smaller blocks, and        Baines, M., Bhosale, S., Caggiano, V., Goyal, N., Goyal,
propagating information from previous blocks with either             S., Ott, M., Lefaudeux, B., Liptchinsky, V., Rabbat, M.,
recurrence (Dai et al., 2019; Hutchins et al., 2022) or cross-       Sheiffer, S., Sridhar, A., and Xu, M. FairScale: A gen-
attention (Hawthorne et al., 2022), (2) linear alternatives          eral purpose modular PyTorch library for high perfor-
to attention, which typically involve forms of token-level           mance and large scale training. https://github.
recurrence (Katharopoulos et al., 2020) or state space mod-          com/facebookresearch/fairscale, 2021.
els (Gu et al., 2021; Smith et al., 2022; Ma et al., 2022), or
(3) sparse approximations of attention (Kitaev et al., 2020;       Beltagy, I., Peters, M. E., and Cohan, A.        Long-
Beltagy et al., 2020; Child et al., 2019; Wu et al., 2022).          former: The long-document transformer. arXiv preprint
However, the performance of dense attention means it is              arXiv:2004.05150, 2020.
typically still chosen for large scale decoders (Touvron et al.,
2023; Chowdhery et al., 2022). M EGA B YTE takes the alter-        Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D.,
native approach of decomposing the complete sequence into            Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
two shorter sequences, giving sub-quadratic attention. We            Askell, A., et al. Language models are few-shot learners.
also note that feedforward networks are the dominant cost            Advances in neural information processing systems, 33:
in large decoders, not self-attention. Our approach to com-          1877–1901, 2020.
pressing sequences allows much larger models than would               5
                                                                        For example, whether or not a prompt should end in whites-
be possible when using large feedforward networks at every         pace depends on details of the underlying subwod algorithm used.
                        M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers

Child, R., Gray, S., Radford, A., and Sutskever, I. Gen-         Hutchins, D., Schlag, I., Wu, Y., Dyer, E., and Neyshabur,
  erating long sequences with sparse transformers. arXiv           B. Block-recurrent transformers. arXiv preprint
  preprint arXiv:1904.10509, 2019.                                 arXiv:2203.07852, 2022.
Choromanski, K., Likhosherstov, V., Dohan, D., Song, X.,         Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman,
  Gane, A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin,         A., and Carreira, J. Perceiver: General perception with it-
  A., Kaiser, L., et al. Rethinking attention with performers.     erative attention. In International conference on machine
  arXiv preprint arXiv:2009.14794, 2020.                           learning, pp. 4651–4664. PMLR, 2021.
Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra,        Kalchbrenner, N., Elsen, E., Simonyan, K., Noury, S.,
  G., Roberts, A., Barham, P., Chung, H. W., Sutton, C.,           Casagrande, N., Lockhart, E., Stimberg, F., van den Oord,
  Gehrmann, S., et al. Palm: Scaling language modeling             A., Dieleman, S., and Kavukcuoglu, K. Efficient neural
  with pathways. arXiv preprint arXiv:2204.02311, 2022.            audio synthesis. CoRR, abs/1802.08435, 2018. URL
                                                                   http://arxiv.org/abs/1802.08435.
Clark, J. H., Garrette, D., Turc, I., and Wieting, J. Canine:
  Pre-training an efficient tokenization-free encoder for        Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B.,
  language representation. Transactions of the Association         Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and
  for Computational Linguistics, 10:73–91, 2022.                   Amodei, D. Scaling laws for neural language models.
                                                                   arXiv preprint arXiv:2001.08361, 2020.
Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q. V.,
  and Salakhutdinov, R. Transformer-xl: Attentive lan-           Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F.
  guage models beyond a fixed-length context, 2019. URL            Transformers are rnns: Fast autoregressive transformers
  https://arxiv.org/abs/1901.02860.                                with linear attention. In International Conference on
                                                                  Machine Learning, pp. 5156–5165. PMLR, 2020.
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn,
  D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M.,     Kingma, D. P. and Ba, J. Adam: A method for stochastic
  Heigold, G., Gelly, S., et al. An image is worth 16x16           optimization. In ICLR, 2015.
 words: Transformers for image recognition at scale. arXiv
                                                                 Kitaev, N., Kaiser, Ł., and Levskaya, A. Reformer: The
  preprint arXiv:2010.11929, 2020.
                                                                   efficient transformer. arXiv preprint arXiv:2001.04451,
Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T.,          2020.
  Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N.,
                                                                 Kudo, T. and Richardson, J. Sentencepiece: A sim-
  Presser, S., and Leahy, C. The pile: An 800gb dataset of
                                                                   ple and language independent subword tokenizer and
  diverse text for language modeling, 2020.
                                                                   detokenizer for neural text processing. arXiv preprint
Gu, A., Goel, K., and Ré, C. Efficiently modeling long            arXiv:1808.06226, 2018.
  sequences with structured state spaces. arXiv preprint
                                                                 Ma, X., Zhou, C., Kong, X., He, J., Gui, L., Neubig, G., May,
  arXiv:2111.00396, 2021.
                                                                  J., and Zettlemoyer, L. Mega: moving average equipped
Hawthorne, C., Jaegle, A., Cangea, C., Borgeaud, S., Nash,        gated attention. arXiv preprint arXiv:2209.10655, 2022.
  C., Malinowski, M., Dieleman, S., Vinyals, O., Botvinick,
                                                                 Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen,
  M., Simon, I., et al. General-purpose, long-context autore-
                                                                  E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O.,
  gressive modeling with perceiver ar. In International Con-
                                                                  Venkatesh, G., et al. Mixed precision training. arXiv
  ference on Machine Learning, pp. 8535–8558. PMLR,
                                                                  preprint arXiv:1710.03740, 2017.
  2022.
                                                                 Oord, A. v. d., Kalchbrenner, N., and Kavukcuoglu, K.
Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E.,           Pixel Recurrent Neural Networks. ICML, 4:2611–2620,
  Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A.,     1 2016. doi: 10.48550/arxiv.1601.06759. URL https:
 Welbl, J., Clark, A., et al. Training compute-optimal            //arxiv.org/abs/1601.06759v3.
  large language models. arXiv preprint arXiv:2203.15556,
  2022.                                                          Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J.,
                                                                   Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga,
Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K.,               L., et al. PyTorch: An imperative style, high-performance
  Salakhutdinov, R., and Mohamed, A. Hubert: Self-                 deep learning library. In NeurIPS, 2019.
  supervised speech representation learning by masked
  prediction of hidden units. IEEE/ACM Transactions on           Press, O., Smith, N. A., and Lewis, M. Shortformer: Better
  Audio, Speech, and Language Processing, 29:3451–3460,            language modeling using shorter inputs. arXiv preprint
  2021.                                                            arXiv:2012.15832, 2020.
                        M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and       Trinh, T. H. and Le, Q. V. A simple method for common-
  Sutskever, I. Language models are unsupervised multitask        sense reasoning. arXiv preprint arXiv:1806.02847, 2018.
  learners. 2019.
                                                                van den Oord, A., Dieleman, S., Zen, H., Simonyan, K.,
Rae, J. W., Potapenko, A., Jayakumar, S. M., and Lillicrap,       Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A. W.,
  T. P. Compressive transformers for long-range sequence          and Kavukcuoglu, K. Wavenet: A generative model for
  modelling. arXiv preprint arXiv:1911.05507, 2019a.              raw audio. CoRR, abs/1609.03499, 2016. URL http:
                                                                  //arxiv.org/abs/1609.03499.
Rae, J. W., Potapenko, A., Jayakumar, S. M., and Lillicrap,
  T. P. Compressive transformers for long-range sequence        van den Oord, A., Li, Y., Babuschkin, I., Simonyan, K.,
  modelling. arXiv preprint arXiv:1911.05507, 2019b.              Vinyals, O., Kavukcuoglu, K., van den Driessche, G.,
                                                                  Lockhart, E., Cobo, L. C., Stimberg, F., Casagrande, N.,
Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Rad-         Grewe, D., Noury, S., Dieleman, S., Elsen, E., Kalchbren-
  ford, A., Chen, M., and Sutskever, I. Zero-shot text-           ner, N., Zen, H., Graves, A., King, H., Walters, T., Belov,
  to-image generation. In International Conference on             D., and Hassabis, D. Parallel wavenet: Fast high-fidelity
  Machine Learning, pp. 8821–8831. PMLR, 2021.                    speech synthesis. CoRR, abs/1711.10433, 2017. URL
                                                                  http://arxiv.org/abs/1711.10433.
Ren, H., Dai, H., Dai, Z., Yang, M., Leskovec, J., Schu-
  urmans, D., and Dai, B. Combiner: Full attention              Wu, Y., Rabe, M. N., Hutchins, D., and Szegedy, C. Mem-
  transformer with sparse computation cost, 2021. URL            orizing transformers. arXiv preprint arXiv:2203.08913,
  https://arxiv.org/abs/2107.05768.                              2022.
Roy, A., Saffar, M., Vaswani, A., and Grangier, D. Efficient    Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M.,
  content-based sparse attention with routing transform-          Chen, S., Dewan, C., Diab, M., Li, X., Lin, V., Mihaylov,
  ers, 2020. URL https://arxiv.org/abs/2003.                      T., Ott, M., Shleifer, S., Shuster, K., Simig, D., Koura,
  05997.                                                          S., Sridhar, A., Wang, T., Zettlemoyer, L., and Ai, M.
                                                                  OPT: Open Pre-trained Transformer Language Models.
Salimans, T., Karpathy, A., Chen, X., and Kingma, D. P.
                                                                  5 2022a. doi: 10.48550/arxiv.2205.01068. URL https:
  Pixelcnn++: Improving the pixelcnn with discretized lo-
                                                                  //arxiv.org/abs/2205.01068v4.
  gistic mixture likelihood and other modifications. CoRR,
  abs/1701.05517, 2017. URL http://arxiv.org/                   Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M.,
  abs/1701.05517.                                                 Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V.,
                                                                  et al. Opt: Open pre-trained transformer language models.
Sennrich, R., Haddow, B., and Birch, A. Neural machine
                                                                  arXiv preprint arXiv:2205.01068, 2022b.
  translation of rare words with subword units. arXiv
  preprint arXiv:1508.07909, 2015.

Sharami, J., Shterionov, D., and Spronck, P. A systematic
  analysis of vocabulary and bpe settings for optimal fine-
  tuning of nmt: A case study of in-domain translation.
  arXiv preprint arXiv:2303.00722, 2023.

Smith, J. T., Warrington, A., and Linderman, S. W. Sim-
  plified state space layers for sequence modeling. arXiv
  preprint arXiv:2208.04933, 2022.

Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu,
  Y. Roformer: Enhanced transformer with rotary position
  embedding. arXiv preprint arXiv:2104.09864, 2021.

Sun, S., Krishna, K., Mattarella-Micke, A., and Iyyer, M.
  Do long-range language models actually use long-range
  context? arXiv preprint arXiv:2109.09115, 2021.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,
  M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E.,
  Azhar, F., et al. Llama: Open and efficient foundation lan-
  guage models. arXiv preprint arXiv:2302.13971, 2023.
                        M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers

A. Appendices
A.1. Training Details
To ensure stable training, we applied gradient clipping with a maximum norm of 1.0 and used the Adam optimizer with
β1 = 0.9, β2 = 0.98 (Kingma & Ba, 2015). We used the built-in polynomial decay learning rate scheduler in MetaSeq with
500 warmup updates and the end learning rate set to 0. All models are trained with pre-norm and using ReLU activation.
We apply a dropout of 0.1 throughout, but we do not apply any dropout to embeddings. We also use weight decay of 0.1. To
initialize the weights, we use a variant based on Megatron-LM codebase, which involves using a normal distribution with a
mean of zero and a standard deviation of 0.006. We truncate this normal distribution within two standard deviations and
observed substantial gain in both training stability and performance.

A.2. Model Details
As discussed in Section 4.1, we conduct experiments using a fixed compute and data budget across all models to focus our
comparisons solely on the model architecture rather than training resources. To achieve this, we adjust model hyperparameters
within each architecture so that the time taken for a single update is matched and then train all models for the same number
of updates. We list all of model details in Table 14 and Table 15.

                                                Model     #L    dmodel   #H    dhead
                                          S1    125M      12      768     12     64
                                          S2    350M      24     1024     16     64
                                          S3    760M      24     1536     16     96
                                          S4     1.3B     24     2048     32     64
                                          S5     2.7B     32     2560     32     80
                                          S6     6.7B     32     4096     32    128

Table 14. Common Model architecture details by size. For each model size, we show the number of layers (#L), the embedding size
(dmodel ), the number of attention heads (#H), the dimension of each attention head (dhead ).
                           M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers

 Model                             (Global) Size             Local Size                BS      LR           Context Length (in bytes)
 arXiv
 Transformer                       320M (D=1024, L=22)       N/A                       72      2.00E-04     1,024
 Perceiver AR                      248M (D=1024, L=17)       N/A                       72      2.00E-04     8,192 (1024 latents)
 M EGA B YTE                       758M (D=2048, L=14)       262M (D=1024, L=18)       48      2.00E-04     8,192 (patch size 8)
   w/o Local model                 2.3B (D=2560, L=20)       N/A                       48      1.50E-04     8,192 (patch size 4)
   w/o global model                N/A                       350M (D=1024, L=24)       192     2.00E-04     8,192 (patch size 8)
   w/o cross-patch Local model     921M (D=2048, L=17)       350M (D=1024, L=24)       48      2.00E-04     8,192 (patch size 8)
   w/ CNN encoder                  704M (D=2048, L=13)       262M (D=1024, L=18)       48      2.00E-04     8,192 (patch size 8)
 Image task 64 (Table 8)
 M EGA B YTE                       2.7B (D=2560, L=32)       350M (D=1024, L=24)       2       2.00E-04     12,288 (patch size 12)
 Image task 64 (Table 5)
 Transformer                       760M (D=1536, L=24)       N/A                       512     3.00E-04     2,048
 Perceiver AR                      227M (D=1024, L=16)       N/A                       512     3.00E-04     12,288 (1024 latents)
 M EGA B YTE                       1.3B (D=2048, L=24)       1.3B (D=2048, L=24)       256     3.00E-04     12,288 (patch size 12)
 Image task 256
 Transformer                       62M (D=768, L=6)          N/A                       1536    2.00E-04     1,024
 Perceiver AR                      62M (D=768, L=6)          N/A                       256     2.00E-04     8,192 (768 latents)
 M EGA B YTE                       125M (D=768, L=12)        125M (D=768, L=12)        16      2.00E-04     196,608 (patch size 192)
   w/o local model                 2.7B (D=4096, L=32)       N/A                       16      2.00E-04     196,608 (patch size 48)
   w/o global model                125M (D=768, L=12)        125M (D=768, L=12)        16      2.00E-04     196,608 (patch size 192)
   w/o cross-patch Local model     250M                      156M (D=768, L=15)        16      2.00E-04     196,608 (patch size 192)
   w/ CNN encoder                  125M (D=768, L=12)        125M (D=768, L=12)        16      2.00E-04     196,608 (patch size 192)
 Image task 640
 Transformer                       83M (D=768, L=8)          N/A                       4800    3.00E-04     1,024
 Perceiver AR                      62M (D=768, L=6)          N/A                       2048    3.00E-04     4,096 (1024 latents)
 M EGA B YTE                       125M (D=768, L=12)        83M (D=768, L=8)          32      3.00E-04     1,228,800 (192 patch size)
 audio
 Transformer                       135M (D=768, L=13)        N/A                       2048    2.00E-04     1024
 Perceiver AR                      62M (D=768, L=6)          N/A                       384     2.00E-04     8,192 (1024 latents)
 M EGA B YTE                       350M (D=1024, L=24)       125M (D=768, L=12)        256     2.00E-04     524,288 (32 patch size)
   w/o local model                 2.7B (D=4096, L=32)       125M (D=768, L=12)        256     2.00E-04     524,288 (32 patch size)
   w/o global model                350M (D=1024, L=24)       125M (D=768, L=12)        256     2.00E-04     524,288 (32 patch size)
   w/o cross-patch Local model     350M (D=1024, L=24)       146M (D=768, L=14)        256     2.00E-04     524,288 (32 patch size)
   w/ CNN encoder                  350M (D=1024, L=24)       125M (D=768, L=12)        256     2.00E-04     524,288 (32 patch size)

Table 15. Model architecture details. We report the model size, the embedding size (D), number of layaers(L), total batch size (BS),
learning rate(LR), and context length. When we vary the number of model layers from the standard amount for the given size (Table 14),
we note this accordingly. For PerceiverAR models, we note the number of latents used, and for M EGA B YTE models we note the patch
sizes.


B. Pseudocode

                                             Listing 1. Pseudocode of Megabyte model

class MegaByteDecoder:
    def __init__(
        self,
        global_args,
        local_args,
        patch_size,
    ):
        self.pad = 0
        self.patch_size = patch_size
        self.globalmodel = TransformerDecoder(global_args)
        self.localmodel = TransformerDecoder(local_args)
                        M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers


     def forward(
         self,
         bytes,
     ):
         bytes_global, bytes_local = self.prepare_input(bytes)

           global_bytes_embedded = self.globalmodel.embed(bytes_global)
           global_in = rearrange(
               global_bytes_embedded,
               "b (t p) e -> b t (p e)",
               p=self.patch_size,
           )
           global_output = self.globalmodel(global_in)

           global_output_reshaped = rearrange(
               global_output,
               "b t (p e) -> (b t) p e",
               p=self.patch_size,
           )
           local_bytes_embedded = self.localmodel.embed(bytes_local)
           local_in = local_bytes_embedded + global_output_reshaped
           local_output = self.localmodel(local_in)

           batch_size = bytes_global.shape[0]
           x = rearrange(local_output, "(b t) l v                 -> b (t l) v", b=batch_size)
           return x

     def prepare_input(self, bytes):
         padding_global = bytes.new(bytes.shape[0], self.patch_size).fill_(self.pad)
         bytes_global = torch.cat((padding_global, bytes[:, : -self.patch_size]), -1)

           bytes_input = rearrange(bytes, "b (t p) -> (b t) p", p=self.patch_size)
           padding_local = bytes_input.new(bytes_input.shape[0], 1).fill_(self.pad)
           bytes_local = torch.cat((padding_local, bytes_input[:, :-1]), -1)

           return bytes_global, bytes_local



C. PerceiverAR Implementation
To reproduce PerceiverAR in a compute-controlled setting we extended the standard transformer implementation in metaseq
with an additonal cross attention layer to compute the latents and match the architecture of PerceiverAR. We trained the
model by sampling random spans from each text, matching the procedure used in the PerceiverAR codebase. To be consistent
with the original work, we use sliding window evaluation with a stride of num latents/2 unless otherwise noted. In several
cases we used the standard metaseq implementation as opposed to specific techniques reported in the original paper: 1)
we used standard attention dropout instead of cross-attention dropout 2) We did not implement chunked attention. We
verified our implementation by reproducing the ”Standard Ordering” experiments in Table 5 of the Perceiver AR paper.
After carefully matching context size, number of latents, the amount of data and training steps used and learning rate, we
achieved 3.53 bpb vs 3.54 reported in the original paper.

D. More results
D.1. Patch scan Implementation
Images have a natural structure, containing a grid of n × n pixels each composed of 3 bytes (corresponding to color channels).
We explore two ways of converting images to sequences for modeling (see Figure 6). Firstly, raster scan where the pixels
are linearized intoq3 bytes and concatenated row-by-row. Secondly, patch scan where we create patches of shape p × p × 3
bytes where p = P3 , and then use a raster scan both within and between patches. Unless otherwise specified, M EGA B YTE
models use patch scan for image data.
                          M EGA B YTE: Predicting Million-byte Sequences with Multiscale Transformers




Figure 6. Two ways to model 2D data sequentially. Left, raster scan, by taking bytes row by row and left to right; right, patch scan, where
we first split an image into patches, and do raster scan across patches and within a patch. (T=36, K=9, P=4).


D.2. Patch scan vs Raster scan
The patch scan method is inspired by recent works in Vision Transformers (Dosovitskiy et al., 2020), and it is more effective
than raster scan for modeling image sequencing. We found it improves both M EGA B YTE and Perceiver AR.

                                              (Global) Size              Local Size                   context              bpb
           M EGA B YTE (patch scan)       62M (D=768, L=6)                N/A                  8,192 (768 latents)        3.158
           M EGA B YTE (raster scan)      62M (D=768, L=6)                N/A                  8,192 (768 latents)        3.428
           Perceiver AR (patch scan)     125M (D=768, L=12)        125M (D=768, L=12)        196,608 (patch size 192)     3.373
           Perceiver AR (raster scan)    125M (D=768, L=12)        125M (D=768, L=12)        196,608 (patch size 192)     3.552

                Table 16. ImageNet256 performance with patch scan vs raster scan for M EGA B YTE and Perceiver AR.



D.3. Longer sequence modeling
For our pg19 scaling experiment, we also use longer context length for M EGA B YTE. The results are shown in Table 17.
With longer sequence, we didn’t observer further improvement, consistent with findings in Hawthorne et al. (2022). We
think we will benefit more from longer sequence when we futher scale up the model size and data.

                                                                    context               bpb
                                           M EGA B YTE         8,192 (patch size 8)     0.8751
                                           M EGA B YTE        16,384 (patch size 8)     0.8787

Table 17. Longer sequence for PG19 dataset. For both experiments, we set global model as 1.3b, local model as 350m, and M EGA B YTE
patch size as 8.

