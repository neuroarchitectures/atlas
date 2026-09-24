# ModernBERT Unknown Clavie Taghadouini 2025

> Source: `ModernBERT_Unknown_Clavie_Taghadouini_2025.pdf`

---

                                         Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast,
                                             Memory Efficient, and Long Context Finetuning and Inference
                                                            Benjamin Warner1† Antoine Chaffin2† Benjamin Clavié1†
                                                              Orion Weller3 Oskar Hallström2 Said Taghadouini2
                                                         Alexis Gallagher1 Raja Biswas1 Faisal Ladhak4* Tom Aarsen5
                                                         Nathan Cooper1 Griffin Adams1 Jeremy Howard1 Iacopo Poli2
                                                  1
                                                      Answer.AI 2 LightOn 3 Johns Hopkins University 4 NVIDIA 5 HuggingFace
                                                                           †: core authors, *: work done while at Answer.AI
                                                                   Correspondence: {bw,bc}@answer.ai, antoine.chaffin@lighton.ai

                                                              Abstract                                option against encoder-decoder and decoder-only
                                                                                                      language models when dealing with substantial
                                             Encoder-only transformer models such as                  amounts of data (Penedo et al., 2024).
                                             BERT offer a great performance-size tradeoff




arXiv:2412.13663v2 [cs.CL] 19 Dec 2024
                                                                                                         Encoder models are particularly popular in In-
                                             for retrieval and classification tasks with re-
                                             spect to larger decoder-only models. Despite             formation Retrieval (IR) applications, e.g., seman-
                                             being the workhorse of numerous production               tic search, with notable progress on leveraging en-
                                             pipelines, there have been limited Pareto im-            coders for this task (Karpukhin et al., 2020; Khat-
                                             provements to BERT since its release. In this            tab and Zaharia, 2020). While LLMs have taken
                                             paper, we introduce ModernBERT, bringing                 the spotlight in recent years, they have also moti-
                                             modern model optimizations to encoder-only               vated a renewed interest in encoder-only models
                                             models and representing a major Pareto im-
                                                                                                      for IR. Indeed, encoder-based semantic search is
                                             provement over older encoders. Trained on
                                             2 trillion tokens with a native 8192 sequence            a core component of Retrieval-Augmented Gener-
                                             length, ModernBERT models exhibit state-of-              ation (RAG) pipelines (Lewis et al., 2020), where
                                             the-art results on a large pool of evaluations           encoder models are used to retrieve and feed LLMs
                                             encompassing diverse classification tasks and            with context relevant to user queries.
                                             both single and multi-vector retrieval on dif-              Encoder-only models are also still frequently
                                             ferent domains (including code). In addition             used for a variety of discriminative tasks such as
                                             to strong downstream performance, Modern-                classification (Tunstall et al., 2022) or Natural En-
                                             BERT is also the most speed and memory effi-
                                                                                                      tity Recognition (NER) (Zaratiana et al., 2024),
                                             cient encoder and is designed for inference on
                                             common GPUs.                                             where they often match the performance of special-
                                                                                                      ized LLMs. Here again, they can be used in con-
                                         1   Introduction                                             junction with LLMs, for example detecting toxic
                                                                                                      prompts (Ji et al., 2023; Jiang et al., 2024b) and pre-
                                         After the release of BERT (Devlin et al., 2019),             venting responses, or routing queries in an agentic
                                         encoder-only transformer-based (Vaswani et al.,              framework (Yao et al., 2023; Schick et al., 2023).
                                         2017) language models dominated most appli-                     Surprisingly, these pipelines currently rely on
                                         cations of modern Natural Language Processing                older models, and quite often on the original BERT
                                         (NLP). Despite the rising popularity of Large Lan-           itself as their backbone (Wang et al., 2022; Xiao
                                         guage Models (LLMs) such as GPT (Radford et al.,             et al., 2023), without leveraging improvements de-
                                         2018, 2019; Brown et al., 2020), Llama (Touvron              veloped in recent years. Practitioners face many
                                         et al., 2023; Dubey et al., 2024), and Qwen (Bai             drawbacks: sequence lengths limited to 512 tokens,
                                         et al., 2023; Yang et al., 2024), encoder-only               suboptimal model design (Anthony et al., 2024)
                                         models remain widely used in a variety of non-               and vocabulary sizes (Karpathy, 2023), and gen-
                                         generative downstream applications.                          erally inefficient architectures, whether in terms
                                            The encoder’s popularity is largely due to their          of downstream performance or computational ef-
                                         modest inference requirements, enabling them to              ficiency. Finally, training data is limited in vol-
                                         efficiently process corpora of documents at scale            ume and restricted to narrow domains (especially
                                         for retrieval and quickly perform discriminative             lacking code data) or lacking knowledge of recent
                                         tasks. Encoder models offer a compelling trade-              events.
                                         off in quality versus size, making them a popular               Recent modernization efforts have only partially
                                             https://github.com/AnswerDotAI/ModernBERT                addressed the shortcomings of encoder-only mod-

                                                                                                  1
els due to limited breadth. MosaicBERT (Portes               final decoder linear layer2 . We also disable all bias
et al., 2023), CrammingBERT (Geiping and Gold-               terms in Layer Norms (Xu et al., 2019). These two
stein, 2023), and AcademicBERT (Izsak et al.,                changes allow us to spend more of our parameter
2021) focused on matching BERT performance                   budget in linear layers.
with better training efficiency. NomicBERT (Nuss-               Positional Embeddings We use rotary posi-
baum et al., 2024) and GTE-en-MLM (Zhang et al.,             tional embeddings (RoPE) (Su et al., 2024) instead
2024) (developed concurrently to this work) intro-           of absolute positional embeddings. This choice is
duced longer-context encoder models focused on               motivated by the proven performance of RoPE in
retrieval applications, but did not optimize for effi-       short- and long-context language models (Black
ciency or classification performance, and re-used            et al., 2022; Dubey et al., 2024; Gemma et al.,
older training data mixtures which is especially             2024), efficient implementations in most frame-
apparent in programming-related tasks.                       works, and ease of context extension.
   Contributions We present ModernBERT, a mod-                  Normalization We use a pre-normalization
ernized encoder-only transformer model, with an              block (Xiong et al., 2020) with the standard layer
improved architecture designed to increase down-             normalization (Lei Ba et al., 2016), which is known
stream performance and efficiency, especially over           to help stabilize training (Xiong et al., 2020). Sim-
longer sequence lengths. We also bring encoder-              ilar to CrammingBERT (Geiping and Goldstein,
only models to modern, larger data scales, by                2023) which also uses pre-normalization, we add
training on 2 trillion tokens, with a data mix-              a LayerNorm after the embedding layer. To avoid
ture including code data. We release two mod-                repetition, we remove the first LayerNorm in the
els, ModernBERT-base and ModernBERT-large,                   first attention layer.
which reach state-of-the-art overall performance                Activation We adopt GeGLU (Shazeer, 2020),
against all existing encoder models on a wide vari-          a Gated-Linear Units (GLU)-based (Dauphin et al.,
ety of downstream tasks. These results are achieved          2017) activation function built on top of the origi-
with considerably higher inference efficiency, pro-          nal BERT’s GeLU (Hendrycks and Gimpel, 2016)
cessing sequences of 8192 tokens almost two times            activation function. This is in line with recent work
faster than previous models.                                 showing consistent empirical improvements when
   To support future research on encoder-only mod-           using GLU variants (Shazeer, 2020; Geiping and
els, we release FlexBERT1 , our modular architec-            Goldstein, 2023).
ture framework allowing easy experimentation, and
inspired by Pythia (Biderman et al., 2023), all in-          2.1.2    Efficiency Improvements
termediate training checkpoints (further detailed in         Alternating Attention Following recent work on
Section 2.2.2).                                              efficient long context models (Gemma et al., 2024),
                                                             attention layers in ModernBERT alternate between
2       Methods                                              global attention, where every token within a se-
2.1      Architectural Improvements                          quence attends to every other token, and local atten-
                                                             tion, where tokens only attend to each other within
Our model architecture extends the standard trans-           a small sliding window (Beltagy et al., 2020). In
former architecture (Vaswani et al., 2017) by incor-         ModernBERT, every third layer employs global
porating extensively tested recent advances (Sec-            attention with a RoPE theta of 160,000 and the
tion 2.1.1). We introduce additional efficiency-             remaining layers use a 128 token, local sliding win-
oriented modifications, through both architectural           dow attention with a RoPE theta of 10,000.
and implementation improvements (Section 2.1.2)                 Unpadding ModernBERT follows Mo-
and a GPU optimized model design (Section 2.1.3).            saicBERT (Portes et al., 2023) and GTE (Zhang
All of our architectural decisions were informed by          et al., 2024) in employing unpadding (Zeng et al.,
ablations, which we detail in Appendix D.                    2022) for both training and inference. Encoder-
2.1.1     Modern Transformer                                 only language models typically use padding tokens
                                                             to ensure a uniform sequence length in a batch,
Bias Terms Following (Dayma et al., 2021), we
disable bias terms in all linear layers except for the          2
                                                                  While many efficient BERT training recipes disable the
                                                             bias term in the decoder, e.g. Geiping and Goldstein (2023),
    1
    FlexBERT is built on top of a revised Mo-                we hypothesized a decoder bias might help alleviate weight
saicBERT (Portes et al., 2023) codebase.                     tying’s negative effects (Gao et al., 2019; Welch et al., 2020).


                                                         2
wasting compute on semantically empty tokens.                aiming to be as Deep & Narrow as possible without
Unpadding avoids this inefficiency by removing               a significant inference slowdown.
padding tokens, concatenating all sequences                     ModernBERT has 22 and 28 layers for the base
from a minibatch into a single sequence, and                 and large models, for a total parameter count of 149
processing it as a batch of one. Prior unpadding             and 395 million, respectively, striking the balance
implementations unpad and repad sequences                    between downstream performance and hardware
internally for different model layers, wasting               efficiency. ModernBERT base has a hidden size of
compute and memory bandwidth. We use Flash                   768 with a GLU expansion of 2,304, while large
Attention’s variable length attention and RoPE               has a hidden size of 1,024 and GLU expansion
implementations, allowing jagged attention masks             of 5,248. These ratios allow optimal tiling across
and RoPE applications on one unpadded sequence.              tensor cores and the most efficient tiling across the
ModernBERT unpads inputs before the token                    differing number of streaming multiprocessors on
embedding layer and optionally repads model                  our target basket of GPUs. More details on model
outputs leading to a 10-to-20 percent performance            design are provided in Appendix B.
improvement over other unpadding methods.
   Flash Attention Flash Attention (Dao et al.,              2.2   Training
2022) is a core component of modern transformer-             2.2.1 Data
based models, providing memory and compute ef-               Mixture Both ModernBERT models are trained on
ficient attention kernels. At the start of this work,        2 trillion tokens of primarily English data from a
Flash Attention 3 (Shah et al., 2024), the most              variety of data sources, including web documents,
recent iteration for Nvidia H100 GPUs, did not               code, and scientific literature, following common
include support for sliding window attention. Mod-           modern data mixtures. We choose the final data
ernBERT uses a mixture of Flash Attention 3 for              mixture based on a series of ablations.
global attention layers and Flash Attention 2 (Dao,             Tokenizer Unlike the majority of recent en-
2023) for local attention layers.                            coders which reuse the original BERT tok-
   torch.compile We leverage PyTorch’s built-in              enizer (Nussbaum et al., 2024; Portes et al., 2023;
compiling (Ansel et al., 2024) to improve the train-         Zhang et al., 2024), we opt to use a modern BPE
ing efficiency by compiling all compatible modules.          tokenizer. We use a modified version of the OLMo
This yields a 10 percent improvement in throughput           tokenizer (Groeneveld et al., 2024) which provides
with negligible compilation overhead.                        better token efficiency and performance on code-
                                                             related tasks. The ModernBERT tokenizer uses the
2.1.3   Model Design                                         same special tokens (e.g., [CLS] and [SEP]) and
At the same parameter count, models with more                templating as the original BERT model (Devlin
narrow layers (Deep & Narrow) have different                 et al., 2019), facilitating backwards compatibility.
learning patterns than models with fewer wide lay-           To ensure optimal GPU utilization (Anthony et al.,
ers (Shallow & Wide) (Nguyen et al., 2021). Tay              2024; Karpathy, 2023), the vocabulary is set to
et al. (2022) and (Liu et al., 2024) have shown              50,368, a multiple of 64 and includes 83 unused
that Deep & Narrow language models have bet-                 tokens to support downstream applications.
ter downstream performance than their shallower                 Sequence Packing In order to avoid high
counterparts, at the expense of slower inference.            minibatch-size variance within our training batches
     Anthony et al. (2024) highlighted that large            as a result of unpadding, we adopt sequence pack-
runtime gains can be unlocked by designing mod-              ing (Raffel et al., 2020; Krell et al., 2022) with
els in a hardware-aware way, which had previ-                a greedy algorithm, which resulted in a sequence
ously been anecdotally observed by many prac-                packing efficiency of over 99 percent, ensuring
titioners (Shoeybi et al., 2019; Karpathy, 2023;             batch size uniformity.
Black et al., 2022). ModernBERT was designed
                                                             2.2.2 Training Settings
through many small-scale ablations to maximize
the utilization of a basket of common GPUs3 , while          MLM We follow the Masked Language Modeling
                                                             (MLM) setup used by MosaicBERT (Portes et al.,
   3
     Which, at the time of this work, are server GPUs:       2023). We remove the Next-Sentence Prediction
NVIDIA T4, A10, L4, A100, and H100 and consumer GPUs:
NVIDIA RTX 3090 and 4090. Prioritization was given to        objective which introduces noticeable overhead for
inference GPUs (excluding A100 & H100).                      no performance improvement (Liu et al., 2019a;

                                                         3
Izsak et al., 2021), and use a masking rate of 30            initialize -large’s weights from ModernBERT-base.
percent, as the original rate of 15 percent has since        In ablation runs, this consistently matched Phi’s
been shown to be sub-optimal (Wettig et al., 2023).          improved training results and greatly speed up the
   Optimizer We use the StableAdamW opti-                    initial loss decrease of our model training5 . Details
mizer (Wortsman et al., 2023), which improves                are provided in Appendix A.2.
upon AdamW (Loshchilov and Hutter, 2019) by                     Context Length Extension After training on 1.7
adding Adafactor-style (Shazeer and Stern, 2018)             trillion tokens at a 1024 sequence length and RoPE
update clipping as a per-parameter learning rate             theta of 10,000, we extend the native context length
adjustment. StableAdamW’s learning rate clipping             of ModernBERT to 8192 tokens by increasing the
outperformed standard gradient clipping on down-             global attention layer’s RoPE theta to 160,000 and
stream tasks and led to more stable training. Hy-            train for an additional 300 billion tokens. We first
perparameters details are given in Appendix A.               train at a constant lower learning rate6 of 3e-4 for
   Learning Rate Schedule During pretraining,                250 billion tokens on an 8192 token mixture of the
we use a modified trapezoidal Learning Rate                  original pretraining dataset sampled following Fu
(LR) schedule (Xing et al., 2018), also known as             et al. (2024). Next, we upsample higher-quality
Warmup-Stable-Decay (WSD) (Zhai et al., 2022;                sources following Gao et al. (2024) and conduct
Hu et al., 2024). After a short LR warmup, the               the decay phase with a 1 − sqrt LR schedule over
trapezoidal schedule holds the LR constant for the           50 billion tokens. This context extension process
majority of training, followed by a short LR de-             yielded the most balanced model on downstream
cay. This schedule has been shown to match the               tasks, as most of our ablations using only one of
performance of cosine scheduling (Hägele et al.,             these strategies resulted in a performance loss on
2024; Hallström et al., 2024) with the benefit of            either retrieval or classification tasks.
enabling continual training on any checkpoint with-
out cold restart issues (Ash and Adams, 2019). Un-           3       Downstream Evaluation
like most trapezoidal schedules, we use a 1 − sqrt           We performed an extensive set of evaluations,
LR decay (Hägele et al., 2024), as we found it to            across a large range of tasks, aiming to demon-
outperform linear and cosine decay.                          strate the versatility of ModernBERT in common
   We trained ModernBERT-base at a constant LR               scenarios.
of 8e-4 for 1.7 trillion tokens following a 3 billion           For all tasks, ModernBERT is evaluated against
token warmup. After a 2 billion token warmup,                existing encoders of similar size. The BASE size,
we trained ModernBERT-large at a LR of 5e-4 for              conventionally defined as under 150 million pa-
900 billion tokens. We rolled back and restarted             rameters, includes BERT-base (Devlin et al., 2019),
training at 5e-5 for the remaining 800 billion tokens        DeBERTa-v3-base (He et al., 2023), RoBERTa-
after large’s loss plateaued for a few hundred billion       base (Liu et al., 2019a), as well as the more re-
tokens at 5e-4.                                              cent 8192 context NomicBERT (Nussbaum et al.,
   Batch Size Schedule Batch size scheduling                 2024) and GTE-en-MLM-base (Zhang et al., 2024).
starts with smaller gradient accumulated batches,            The LARGE size, conventionally defined as above
increasing over time to the full batch size. In abla-        300 million and under 500 million parameters, in-
tions, this schedule accelerated training progress.          cludes BERT-large-uncased (Devlin et al., 2019),
We warmup the batch size from 768 to 4,608 over              DeBERTa-v3-large (He et al., 2023) and RoBERTa-
50 billion tokens and from 448 to 4,928 over 10              large (Liu et al., 2019a) and GTE-en-MLM-
billion tokens, for ModernBERT-base and -large,              large (Zhang et al., 2024).
respectively, with an uneven token schedule so each
batch size has the same number of update steps.              3.1      Evaluation Setting
Details are provided in Appendix A.1.                        3.1.1 Natural Language Understanding
   Weight Initialization and Tiling We initialize            The General Language Understanding Evaluation
ModernBERT-base with random weights following                (GLUE) benchmark (Wang et al., 2018) is the
the Megatron initialization (Shoeybi et al., 2019).          standard Natural Language Understanding (NLU)
For ModernBERT-large, we follow the Phi model                    5
                                                                   This initialization reduced the amount of batch size and
family (Li et al., 2023; Javaheripi et al., 2023)4 and       LR warmup needed for ModernBERT-large
                                                                 6
                                                                   We only lowered the LR for ModernBERT-base, as large
   4
       As detailed in their 2023 NeurIPS presentation.       already decreased LR during the 1024 token training phase.


                                                         4
benchmark for encoder models, aiming to measure                        similarity between a query and a document can then
how well a model performs across a range of sen-                       be computed through distance operations, such as
tence or sentence-pair understanding tasks, such as                    cosine similarity. Models are finetuned using con-
sentiment detection (Liu et al., 2019b) or language                    trastive learning to create representations which
entailment, through tasks such as MNLI (Williams                       are close if a document is relevant to a query, and
et al., 2018). Although GLUE is often regarded                         distant if not (van den Oord et al., 2018).
as saturated by the best-performing models, such                          We train every base model using the MS-
as large language models (Zhao et al., 2023), it                       MARCO (Bajaj et al., 2016) dataset with mined
remains one of the most commonly used evaluation                       hard negatives (Xuan et al., 2020) on 1.25M sam-
suites for smaller encoder-based models, and pro-                      ples with a batch size of 16 and learning rate
vides a good impression of a model’s performance                       warmup for 5% of the training using sentence-
on common classification tasks (Portes et al., 2023;                   transformers (Reimers and Gurevych, 2019).
Zhang et al., 2024; He et al., 2023).                                     Multi vector retrieval Multi-vector retrieval,
   We follow the practice of previous studies (De-                     championed by ColBERT (Khattab and Zaharia,
vlin et al., 2019; Liu et al., 2019a; He et al.,                       2020), seeks to mitigate lost information from com-
2023) and conduct a hyperparameter search on each                      pressing an entire sequence into a single vector.
GLUE subset (detailed in Appendix E.1) in order                        In multi-vector retrieval, each document is repre-
to provide values comparable to other models.7                         sented by all of its individual token vectors, and
                                                                       the similarity between a query and a document is
3.1.2    Text Retrieval                                                computed using the MaxSim9 operator.
Information Retrieval (IR) is one of the most com-                        We adopt the training setup of JaCol-
mon applications of encoder-only models,8 where                        BERTv2.5 (Clavié, 2024), an update on the
they are used to represent documents and queries                       ColBERTv2 (Santhanam et al., 2022) training
in semantic search (Karpukhin et al., 2020). This                      procedure, with a batch size of 16 and a 5%
domain has recently seen considerable growth and                       learning rate warmup. We train all models by
interest following the spread of LLMs where se-                        distilling the knowledge of a teacher model by
mantic search powered by lightweight models is                         using the KL-Divergence between the normalized
used to provide relevant context to LLMs as part of                    teacher and student scores. Models are trained
Retrieval-Augmented Generation pipelines.                              on 810k samples from MS-Marco (Bajaj et al.,
   We evaluate models in both the single-vector                        2016) and teacher scores from BGE-M3 (Chen
Dense Passage Retrieval (DPR) (Karpukhin et al.,                       et al., 2024), using the PyLate library (Chaffin and
2020) setting and the multi-vector ColBERT (Khat-                      Sourty, 2024).
tab and Zaharia, 2020) setting.
                                                                       3.1.3 Long-Context Text Retrieval
   We report retrieval results on the popular BEIR
evaluation suite (Thakur et al., 2021), the com-                       With a native 8192 context length, ModernBERT
mon standard for evaluating retrieval performance                      improves long-context performance over most ex-
across a variety of tasks and domains, using the                       isting encoders. However, there are relatively
nDCG@10 metric. For each setting detailed below,                       few standardized long-context benchmarks for
we conduct a learning rate sweep based on results                      encoder-only models, and most benchmarks, such
over a subset of the BEIR benchmarks to select the                     as Needle-in-a-haystack (Kamradt, 2023) and
final model, detailed in Appendix E.2.                                 RULER (Hsieh et al., 2024) are geared towards gen-
   Single vector retrieval One of the most com-                        erative tasks. Given this limitation, we demonstrate
mon approaches to neural retrieval using encoders                      improved long-context performance on the English
is DPR (Karpukhin et al., 2020), where a single-                       subset of MLDR (Chen et al., 2024), a long-context
vector is used to represent an entire document. The                    retrieval benchmark comprised of over 200,000
                                                                       long documents. We evaluate three settings:
   7
      As (Zhang et al., 2024) do not explicitly mention a param-          Single Vector – Out-Of-Domain Models are
eter sweep, we initially ran the same hyperparameter sweep             trained on short-context MS-MARCO as described
as we did for ModernBERT, but observed inconsistencies in
the results. To avoid under-representing GTE-en-MLM’s ca-              above, and is evaluated on long context MLDR
pabilities, we choose to use their reported GLUE results.              without any further fine-tuning.
    8
      At the time of this paper’s writing, over half of the 100
                                                                          9
most downloaded models on the HuggingFace Model Hub                       The sum for every query token of its similarity with the
were encoder-based retrieval models.                                   most similar document token


                                                                   5
                                           IR (DPR)                          IR (ColBERT)        NLU          Code
          Model               BEIR      MLDROOD          MLDRID             BEIR   MLDROOD       GLUE     CSN     SQA
          BERT                38.9          23.9            32.2            49.0      28.1        84.7    41.2    59.5
          RoBERTa             37.7          22.9            32.8            48.7      28.2        86.4    44.3    59.6
          DeBERTaV3           20.2           5.4            13.4            47.1      21.9        88.1    17.5    18.6
  Base    NomicBERT           41.0          26.7            30.3            49.9      61.3        84.0    41.6    61.4
          GTE-en-MLM          41.4          34.3            44.4            48.2      69.3        85.6    44.9    71.4
          ModernBERT          41.6          27.4            44.0            51.3      80.2        88.4    56.4    73.6
          BERT                38.9          23.3            31.7            49.5      28.5        85.2    41.6    60.8
          RoBERTa             41.4          22.6            36.1            49.8      28.8        88.9    47.3    68.1
  Large   DeBERTaV3           25.6           7.1            19.2            46.7      23.0        91.4    21.2    19.7
          GTE-en-MLM          42.5          36.4            48.9            50.7      71.3        87.6    40.5    66.9
          ModernBERT          44.0          34.3            48.6            52.4      80.4        90.4    59.5    83.9

Table 1: Results for all models across an overview of all tasks. CSN refers to CodeSearchNet and SQA to StackQA.
MLDRID refers to in-domain (fine-tuned on the training set) evaluation, and MLDROOD to out-of-domain.


   Single Vector – In Domain Models trained                           et al., 2024), where the model must identify rel-
on MS-MARCO are further fine-tuned on long-                           evant responses to StackOverflow questions, in a
context MLDR training set before being evaluated.                     "hybrid" setting where documents contain both text
   Multi-Vector – Out-Of-Domain Due to its                            and code. The latter benchmark also leverages long-
token-level MaxSim mechanism, ColBERT mod-                            context capabilities, as its queries and documents
els are able to generalize to long-context without                    respectively contain 1,400 and 1,200 words on aver-
any specific training (Bergum, 2024). We directly                     age, leading to average token counts of over 2000.
evaluate the best checkpoints from Section 3.1.2                         We evaluate these benchmarks using the CoIR
without any further fine-tuning on MLDR.                              (CodeIR) framework (Li et al., 2024), as single-
                                                                      vector retrieval tasks. All models are trained by
3.1.4     Code Retrieval                                              re-using the best hyper-parameters identified in
Fueled by increasingly good code completion mod-                      Section 3.1.2.
els (Jiang et al., 2024a), downstream applications
have quickly grown in popularity following the                        3.2    Downstream Results and Discussion
emergence of code assistants.10 Encoder-only mod-                     Aggregated results for all evaluations are presented
els are used to process and retrieve large quantities                 in Table 1. For BEIR and GLUE, the two common
of code-related information under resource con-                       evaluation suites, we follow existing practice in
straints, increasing the importance of measuring                      reporting the average results. Detailed results are
and improving code capabilities of encoder models                     provided in Appendix E.
(Li et al., 2024). Unlike most previous encoders                         In terms of downstream performance, Modern-
which were largely trained only on textual data (De-                  BERT is the strongest overall model at both the
vlin et al., 2019; Liu et al., 2019a; Portes et al.,                  BASE and LARGE model sizes. ModernBERT rep-
2023; Zhang et al., 2024; Nussbaum et al., 2024),                     resents a Pareto improvement on all tasks over the
ModernBERT is pre-trained on code and uses a                          original BERT and RoBERTA models, with better
code-aware tokenizer11 .                                              performance on every evaluation category.
   To measure programming-related performance,                           Short-Context Retrieval On BEIR, both vari-
we evaluate all models on CodeSearchNet (Hu-                          ants of ModernBERT outperform existing encoders
sain et al., 2019), a code-to-text benchmark where                    in both the DPR and ColBERT settings, including
the model must identify relevant docstring or com-                    the recent GTE-en-MLM and NomicBERT mod-
ments for code blocks, and StackOverflow-QA (Li                       els designed to serve as better backbones for re-
  10
                                                                      trieval (Zhang et al., 2024; Nussbaum et al., 2024).
    Spearheaded by GitHub Copilot in 2021
  11
    Avoiding issues such as the ones seen in T5 (Raffel et al.,          While ModernBERT-base only narrowly edges
2020), whose vocabulary did not include curly braces.                 out GTE-en-MLM-base on DPR evaluations,

                                                                  6
                                                               Short                      Long
                Model                  Params      BS        Fixed     Variable   BS   Fixed     Variable
                BERT                    110M     1096        180.4        90.2     –       –           –
                RoBERTa                 125M      664        179.9        89.9     –       –           –
                DeBERTaV3               183M      236         70.2        35.1     –       –           –
        Base    NomicBERT               137M      588        117.1        58.5    36    46.1        23.1
                GTE-en-MLM              137M      640        123.7        61.8    38    46.8        23.4
                GTE-en-MLMxformers      137M      640        122.5       128.6    38    47.5        67.3
                ModernBERT              149M     1604        148.1       147.3    98   123.7       133.8
                BERT                    330M       792        54.4        27.2     –       –           –
                RoBERTa                 355M       460        42.0        21.0     –       –           –
        Large   DeBERTaV3               434M       134        24.6        12.3     –       –           –
                GTE-en-MLM              435M       472        38.7        19.3    28    16.2         8.1
                GTE-en-MLMxformers      435M       472        38.5        40.4    28    16.5        22.8
                ModernBERT              395M       770        52.3        52.9    48    46.8        49.8

Table 2: Memory (max batch size, BS) and Inference (in thousands of tokens per second) efficiency results on an
NVIDIA RTX 4090, averaged over 10 runs. Dashes indicate unsupported configurations.


ModernBERT-large increases its lead despite hav-             at least a 9 NDCG@10 point lead on both model
ing comparatively fewer parameters at 395M to                sizes. We theorize that these sizable gains could
GTE-en-MLM-large’s 435M.                                     be explained by our long pretraining ensuring few,
   Long-Context Retrieval - Single Vector In the             if any, tokens are under-trained, as well as a po-
DPR setting, ModernBERT achieves impressive                  tentially synergistic effect of local attention with
performance on MLDR, a long-context text re-                 ColBERT-style retrieval, but leave further explo-
trieval task. However, these results also highlight          ration of this phenomenon to future work.
an interesting phenomenon: without long-context                 Natural Language Understanding Both Mod-
finetuning ModernBERT outperforms both shorter-              ernBERT models demonstrate exceptional NLU
context models and the long-context NomicBERT                results, as measured by GLUE. ModernBERT-
but performs noticeably worse than GTE-en-MLM.               base surpasses all existing base models, includ-
The performance gap narrows considerably when                ing DeBERTaV3-base, becoming the first MLM-
evaluated in-domain, with both models performing             trained model to do so. This is surprising,
similarly. This suggests that ModernBERT can ef-             as DeBERTaV3 was trained with the Replaced-
fectively process long context sequences as a dense          Token-Detection objective, which was previously
encoder but may require more adapted tuning. We              thought to yield stronger downstream NLU per-
plan to explore multiple potential explanations for          formance (Clark et al., 2020; He et al., 2023).
this phenomenon in future work, including the im-            ModernBERT-large is the second-best large en-
pact of local attention or GTE-en-MLM having                 coder on GLUE, almost matching DeBERTaV3-
spent a larger part of its pretraining compute bud-          large with one-tenth fewer parameters while pro-
get on longer sequence lengths (Zhang et al., 2024).         cessing tokens in half the time (see Section 4).
   Long-Context Retrieval - Multi-Vector In                     Code On programming tasks, in both code-to-
the ColBERT setting, long-context models (GTE-               text (CodeSearchNet) and longer-context hybrid
en-MLM, NomicBERT, and ModernBERT) all                       settings (StackQA), ModernBERT outperforms all
outperform short-context models by at least 40               other models. This result was expected, as it is the
NDCG@10 points without requiring any specific                only evaluated encoder to be trained on a data mix-
finetuning. These results confirm the findings of            ture including programming data. These results,
Bergum (2024), who showed that ColBERT models                combined with ModernBERT’s strong showings
are particularly well-suited to long-context retrieval       on other tasks, indicates that ModernBERT has im-
tasks. Among the long-context models, Modern-                proved understanding of code at no detriment to its
BERT outperforms other long-context models, with             ability to process natural text.

                                                         7
4        Efficiency                                                     percent more tokens per second at low context
                                                                        lengths and 98.8-118.8 percent more at longer con-
4.1       Evaluation Setting
                                                                        text lengths, thanks to its use of local attention.
To measure inference efficiency across multiple                            ModernBERT is the overall most memory effi-
sequence lengths, we create 4 synthetic sets of                         cient model on both model sizes. ModernBERT-
8192 documents12 . The first two document sets                          base is able to process batch sizes twice as
are fixed-length: in fixed short-context, all docu-                     large as every other model on both input lengths.
ments contain 512 tokens and in fixed long-context                      ModernBERT-large is slightly less memory effi-
all documents contain 8192 tokens13 . To account                        cient than the original BERT-large on short-context
for the impact of unpadding, we also create two                         inputs, but can process batches at least 60 percent
varying-length document sets, where the number                          bigger than every other large model.
of tokens in each set are defined by a normal dis-
tribution centered on half the maximum sequence
length, 256 and 4096 tokens, respectively. Full data                    5   Conclusion
statistics are provided in Appendix F.
   We then evaluate all models based on the number                      We present ModernBERT, an open family of
of tokens they can process per second, averaged                         encoder-only models which set a new state of the
over ten runs. All efficiency evaluations are ran                       art over existing encoder models on a wide range
on a single NVIDIA RTX 4090, one of the target                          of classification and retrieval tasks. We show that
GPUs of ModernBERT outlined in Section 2.1.3                            encoders benefit from both recent pretraining data
We evaluate the GTE-en-MLM models under two                             scales and architecture improvements from autore-
settings: out-of-the box, and with the use of the                       gressive LLMs.
xformers (Lefaudeux et al., 2022) library, which en-                       ModernBERT has a native sequence length of
ables efficiency enhancements such as unpadding.                        8,192 tokens and incorporates recent architecture
4.2       Results                                                       improvements, such as GeGLU layers, RoPE po-
                                                                        sitional embeddings, and alternating local-global
All tokens-per-second efficiency results are pre-                       attention. ModernBERT is the first open model
sented in Table 2, with absolute run-times provided                     to feature entire model unpadding and is the first
in Appendix F. ModernBERT stands out as the                             encoder designed in a hardware-aware way to max-
most efficient model overall. On short context, it                      imize inference efficiency.
processes fixed-length 512 token inputs faster than
                                                                           ModernBERT pushes the encoder state of the art
all other recent encoders, although slower than the
                                                                        forward across a wide range of benchmarks. On
original BERT and RoBERTa models14 . On long-
                                                                        GLUE, ModernBERT-base is the first encoder to
context, ModernBERT is faster than all competing
                                                                        beat DeBERTaV3-base since its release in 2021.
encoders, processing documents 2.65 and 3 times
                                                                        ModernBERT is in a class of its own in code and
faster than the next-fastest encoder at the BASE and
                                                                        ColBERT-style long-context retrieval benchmarks,
LARGE sizes, respectively. ModernBERT-large’s
                                                                        scoring at least 6.85 and 9.1 percentage points
processing speed at length 8192 (46,801 tokens
                                                                        higher than the closest model, respectively, while
per second) is closer to that of GTE-en-MLM base
                                                                        remaining state-of-the-art on short-context retrieval
(47,507 tokens per second) than it is to GTE-en-
                                                                        in both single and multi-vector settings.
MLM-large (16,532 tokens per second).
   On variable-length inputs, both GTE-en-MLM                              At the same time, ModernBERT processes short
and ModernBERT models are considerably faster                           context inputs twice as fast as DeBERTaV3 and
than all other models, largely due to unpadding.                        long-context inputs two times faster than the next
However, ModernBERT remains noticeably more                             fastest model with best-in-class memory efficiency.
efficient than GTE-en-MLM, processing 14.5-30.9                            ModernBERT is a generational leap over the
    12                                                                  original encoder models, with notable performance
      Many common benchmarks are biased towards low and
uniform sequence lengths, which is unrepresentative of many             improvements over BERT and RoBERTa on both
real-world situations.                                                  classification and retrieval tasks. ModernBERT is
   13
      512 being the maximum length of most existing encoders,           one of the few encoders to support long-context and
while 8192 is the maximum length of all long-context ones.
   14
      This is partially due to the relatively low parameter count       programming applications, while simultaneously
of BERT and RoBERTa compared to more recent encoders.                   setting a new record in encoder inference efficiency.

                                                                    8
6   Limitations                                              Vallez, and Pedro Cuenca for assisting with day-
                                                             one HuggingFace support.
Language This study focuses exclusively on the                  Finally, we acknowledge Orange Business Cloud
English language, and trains on a very large num-            Avenue as compute provider and their hardware
ber of tokens. As such, a major limitation of our            support throughout the project and thank LightOn
work is that it is not directly applicable to other          for sponsoring the compute.
languages, and potentially even less-so to lower
resources languages.
                                                             8   Contribution Statement
   Biases Our model is trained largely on web data,
as a result, all of its representations are subject to       BW, AC, and BC jointly led the project and con-
the biases present in such data.                             tributed to all parts of it.
   Harmful Content Generation The MLM objec-                 BW worked on all aspects of the project and con-
tive gives the model some ability to generate text           tributed to all major decisions. He led model de-
by suggesting a given token to replace the [MASK]            sign, model training, implemented the majority of
token (Samuel, 2024), which could result in the              the model architecture, and assisted with data se-
generation of harmful content. However, Modern-              lection, elevations, and paper writing.
BERT is not, primarily, a generative model, and as           AC co-initiated the project and worked on all as-
such, has not been trained to and therefore cannot           pects of it, including project coordination. Notably,
generate longer sequences of text. As a result, it is        he contributed to monitoring training runs and co-
considerably less likely to be at risk of generating         led ablations, final evaluations and paper writing.
harmful content of any kind.                                 BC initiated the project and worked on all aspects
   MLM-only objective Given the strong results of            of it. He contributed to model design and co-led
DeBERTav3 on classification tasks but weak ones              final evaluations, led paper writing, and contributed
on retrieval, it seems that a training leveraging both       to the context extension data processing.
MLM and RTD might be better suited to achieve                OW led and conducted the majority of the data se-
best results on classification. Extending our work           lection, processing, and discussion, for all stages
to RTD is thus a promising line of research.                 of training. He also contributed valuable inputs
   Scaling Besides the architectural modifications,          throughout all stages of the project.
a key aspect of our studies is data scaling. How-            OH and ST contributed to a majority of the stages
ever, other scaling axes, notably in terms of model          of the project, in particular model architecture and
parameters are left unexplored.                              training, with both discussions, implementations
                                                             and paper writing. Other contributions include pre-
7   Acknowledgements                                         training monitoring, final traditional evaluations,
                                                             and ablations. ST specifically worked on adapting
The authors would like to acknowledge & thank the            the RoPE kernel for unpadded sequences and run-
many people who assisted, supported, or offered              ning the final GLUE benchmarks. OH additionally
insights useful for the completion of this project.          conducted a thorough investigation into complex
   We are particularly thankful for the one-off im-          issues that arose during training.
plementation or evaluation work conducted by                 RB contributed greatly to the initial evaluation
Jack Cook, Mark Tenenholtz, Johno Whitaker, and              work, focusing on ablations and in-training evals.
Wayde Gilliam. We also extend similar thanks to              AG and FL contributed to training efficiency, espe-
Zach Nussbaum for assisting in resolving issues we           cially in implementing sequence packing.
encountered with NomicBERT during evaluation.                AG and GA contributed to model evaluations, es-
   We would like to acknowledge Enrico Shippole,             pecially in long context evaluations.
Daniel Han, Colin Raffel, Pierre-Carl Langlais,              TA contributed to discussions throughout the
Omar Khattab, Urchade Zaratiana, Aurélien Lac,               project and assisted in integrating the original re-
Amélie Chatelain, and Raphaël Sourty, for their              search implementation with open source software.
helpful contributions to discussions.                        NC contributed to context extension data mixtures,
   We also thank Weights&Biases for providing                and provided insight into model training and on
free access to their platform, in particular Morgan          improving the quality of code data.
McGuire and Thomas Capelle for their support.                IP and JH provided guidance and support through-
   We thank HuggingFace’s Arthur Zucker, Cyril               out the project, especially on key decisions.

                                                         9
References                                                       Jack Clark, Christopher Berner, Sam McCandlish,
                                                                 Alec Radford, Ilya Sutskever, and Dario Amodei.
Jason Ansel, Edward Yang, Horace He, Natalia                     2020. Language models are few-shot learners. In Ad-
   Gimelshein, Animesh Jain, Michael Voznesensky,                vances in Neural Information Processing Systems 33:
   Bin Bao, Peter Bell, David Berard, Evgeni Burovski,           Annual Conference on Neural Information Process-
   et al. 2024. Pytorch 2: Faster machine learning               ing Systems 2020, NeurIPS 2020, December 6-12,
   through dynamic python bytecode transformation and            2020, virtual.
   graph compilation. In Proceedings of the 29th ACM
   International Conference on Architectural Support           Antoine Chaffin and Raphaël Sourty. 2024. Pylate:
   for Programming Languages and Operating Systems,              Flexible training and retrieval for late interaction
   volume 2, pages 929–947.                                      models.
Quentin Anthony, Jacob Hatef, Deepak Narayanan,                Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
  Stella Biderman, Stas Bekman, Junqi Yin, Aamir                  Luo, Defu Lian, and Zheng Liu. 2024. M3-
  Shafi, Hari Subramoni, and Dhabaleswar Panda.                   embedding: Multi-linguality, multi-functionality,
  2024. The case for co-designing model architectures             multi-granularity text embeddings through self-
  with hardware. Preprint, arXiv:2401.14489.                      knowledge distillation. In Findings of the Asso-
                                                                  ciation for Computational Linguistics, ACL 2024,
Jordan T. Ash and Ryan P. Adams. 2019. On the                     Bangkok, Thailand and virtual meeting, August 11-
   difficulty of warm-starting neural network training.           16, 2024, pages 2318–2335. Association for Compu-
  CoRR, abs/1910.08475.                                           tational Linguistics.
Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang,          Aakanksha Chowdhery, Sharan Narang, Jacob Devlin,
   Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei               Maarten Bosma, Gaurav Mishra, Adam Roberts,
   Huang, et al. 2023. Qwen technical report. arXiv              Paul Barham, Hyung Won Chung, Charles Sutton,
   preprint arXiv:2309.16609.                                    Sebastian Gehrmann, Parker Schuh, Kensen Shi,
                                                                 Sasha Tsvyashchenko, Joshua Maynez, Abhishek
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,              Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vin-
  Jianfeng Gao, Xiaodong Liu, Rangan Majumder,                   odkumar Prabhakaran, Emily Reif, Nan Du, Ben
  Andrew McNamara, Bhaskar Mitra, Tri Nguyen,                    Hutchinson, Reiner Pope, James Bradbury, Jacob
  et al. 2016. Ms marco: A human generated ma-                   Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin,
  chine reading comprehension dataset. arXiv preprint            Toju Duke, Anselm Levskaya, Sanjay Ghemawat,
  arXiv:1611.09268.                                              Sunipa Dev, Henryk Michalewski, Xavier Garcia,
                                                                 Vedant Misra, Kevin Robinson, Liam Fedus, Denny
Iz Beltagy, Matthew E. Peters, and Arman Cohan.                  Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim,
   2020. Longformer: The long-document transformer.              Barret Zoph, Alexander Spiridonov, Ryan Sepassi,
   Preprint, arXiv:2004.05150.                                   David Dohan, Shivani Agrawal, Mark Omernick, An-
Jo Kristian Bergum. 2024. Announcing vespa long-                 drew M. Dai, Thanumalayan Sankaranarayana Pil-
   context ColBERT. Vespa Blog.                                  lai, Marie Pellat, Aitor Lewkowycz, Erica Moreira,
                                                                 Rewon Child, Oleksandr Polozov, Katherine Lee,
Stella Biderman, Hailey Schoelkopf, Quentin Gregory              Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark
  Anthony, Herbie Bradley, Kyle O’Brien, Eric Hal-               Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy
   lahan, Mohammad Aflah Khan, Shivanshu Purohit,                Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov,
   USVSN Sai Prashanth, Edward Raff, et al. 2023.                and Noah Fiedel. 2023. Palm: Scaling language mod-
   Pythia: A suite for analyzing large language mod-             eling with pathways. J. Mach. Learn. Res., 24:240:1–
   els across training and scaling. In International             240:113.
  Conference on Machine Learning, pages 2397–2430.
                                                               Kevin Clark, Minh-Thang Luong, Quoc V. Le, and
   PMLR.
                                                                 Christopher D. Manning. 2020. ELECTRA: pre-
Sidney Black, Stella Biderman, Eric Hallahan, Quentin            training text encoders as discriminators rather than
  Anthony, Leo Gao, Laurence Golding, Horace He,                 generators. In 8th International Conference on
   Connor Leahy, Kyle McDonell, Jason Phang, et al.              Learning Representations, ICLR 2020, Addis Ababa,
   2022. Gpt-neox-20b: An open-source autoregres-                Ethiopia, April 26-30, 2020. OpenReview.net.
   sive language model. In Proceedings of BigScience           Benjamin Clavié. 2024. Jacolbertv2.5: Optimis-
  Episode# 5–Workshop on Challenges & Perspectives               ing multi-vector retrievers to create state-of-the-
   in Creating Large Language Models, pages 95–136.              art japanese retrievers with constrained resources.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie                 Preprint, arXiv:2407.20750.
  Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind             Tri Dao. 2023. Flashattention-2: Faster attention with
  Neelakantan, Pranav Shyam, Girish Sastry, Amanda                better parallelism and work partitioning. In The
  Askell, Sandhini Agarwal, Ariel Herbert-Voss,                  Twelfth International Conference on Learning Repre-
  Gretchen Krueger, Tom Henighan, Rewon Child,                    sentations.
  Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
  Clemens Winter, Christopher Hesse, Mark Chen, Eric           Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and
  Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess,             Christopher Ré. 2022. Flashattention: Fast and


                                                          10
  memory-efficient exact attention with io-awareness.         Alexander Hägele, Elie Bakouch, Atli Kosson,
  Advances in Neural Information Processing Systems,            Loubna Ben Allal, Leandro von Werra, and Mar-
  35:16344–16359.                                               tin Jaggi. 2024. Scaling laws and compute-optimal
                                                                training beyond fixed training durations. CoRR,
Yann N. Dauphin, Angela Fan, Michael Auli, and David            abs/2405.18392.
  Grangier. 2017. Language modeling with gated con-
  volutional networks. In Proceedings of the 34th In-         Oskar Hallström, Said Taghadouini, Clément Thiriet,
  ternational Conference on Machine Learning, vol-              and Antoine Chaffin. 2024. Passing the torch: Train-
  ume 70 of Proceedings of Machine Learning Re-                 ing a mamba model for smooth handover.
  search, pages 933–941. PMLR.
                                                              Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2023.
Boris Dayma, Suraj Patil, Pedro Cuenca, Khalid Saiful-          Debertav3: Improving deberta using electra-style
  lah, Tanishq Abraham, Phúc Lê Khăc, Luke Melas,              pre-training with gradient-disentangled embedding
  and Ritobrata Ghosh. 2021. Dall·e mini.                       sharing. In The Eleventh International Conference
                                                                on Learning Representations, ICLR 2023, Kigali,
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and                   Rwanda, May 1-5, 2023. OpenReview.net.
   Kristina Toutanova. 2019. BERT: pre-training of            Dan Hendrycks and Kevin Gimpel. 2016. Gaus-
   deep bidirectional transformers for language under-          sian error linear units (gelus). arXiv preprint
   standing. In Proceedings of the 2019 Conference of           arXiv:1606.08415.
   the North American Chapter of the Association for
  Computational Linguistics: Human Language Tech-             Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shan-
   nologies, NAACL-HLT 2019, Minneapolis, MN, USA,              tanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang,
  June 2-7, 2019, Volume 1 (Long and Short Papers),             and Boris Ginsburg. 2024. Ruler: What’s the real
   pages 4171–4186. Association for Computational               context size of your long-context language models?
   Linguistics.                                                 arXiv preprint arXiv:2404.06654.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,              Shengding Hu, Yuge Tu, Xu Han, Chaoqun He,
  Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,               Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang,
  Akhil Mathur, Alan Schelten, Amy Yang, Angela                 Yuxiang Huang, Weilin Zhao, Xinrong Zhang,
  Fan, et al. 2024. The llama 3 herd of models. arXiv           Zhen Leng Thai, Kai Zhang, Chongyi Wang, Yuan
  preprint arXiv:2407.21783.                                    Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu
                                                                Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li,
Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Han-             Zhiyuan Liu, and Maosong Sun. 2024. Minicpm: Un-
  naneh Hajishirzi, Yoon Kim, and Hao Peng. 2024.               veiling the potential of small language models with
  Data engineering for scaling language models to 128k          scalable training strategies. CoRR, abs/2404.06395.
  context. Preprint, arXiv:2402.10171.
                                                              Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis
Jun Gao, Di He, Xu Tan, Tao Qin, Liwei Wang, and Tie-           Allamanis, and Marc Brockschmidt. 2019. Code-
  Yan Liu. 2019. Representation degeneration prob-              searchnet challenge: Evaluating the state of semantic
  lem in training natural language generation models.           code search. arXiv preprint arXiv:1909.09436.
  ArXiv, abs/1907.12009.
                                                              Alexander Hägele, Elie Bakouch, Atli Kosson,
Tianyu Gao, Alexander Wettig, Howard Yen, and Danqi             Loubna Ben Allal, Leandro Von Werra, and Mar-
   Chen. 2024. How to train long-context language               tin Jaggi. 2024. Scaling laws and compute-optimal
   models (effectively). Preprint, arXiv:2410.02660.            training beyond fixed training durations. Preprint,
                                                                arXiv:2405.18392.
Jonas Geiping and Tom Goldstein. 2023. Cramming:              Peter Izsak, Moshe Berchansky, and Omer Levy. 2021.
  Training a language model on a single GPU in one              How to train BERT with an academic budget. In Pro-
  day. In International Conference on Machine Learn-            ceedings of the 2021 Conference on Empirical Meth-
  ing, ICML 2023, 23-29 July 2023, Honolulu, Hawaii,            ods in Natural Language Processing, pages 10644–
  USA, volume 202 of Proceedings of Machine Learn-              10652, Online and Punta Cana, Dominican Republic.
  ing Research, pages 11117–11143. PMLR.                        Association for Computational Linguistics.
Team Gemma, Morgane Riviere, Shreya Pathak,                   Mojan Javaheripi, Sébastien Bubeck, Marah Abdin, Jy-
  Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupati-          oti Aneja, Sebastien Bubeck, Caio César Teodoro
  raju, Léonard Hussenot, Thomas Mesnard, Bobak                Mendes, Weizhu Chen, Allie Del Giorno, Ronen
  Shahriari, Alexandre Ramé, et al. 2024. Gemma 2:             Eldan, Sivakanth Gopi, et al. 2023. Phi-2: The sur-
  Improving open language models at a practical size.          prising power of small language models. Microsoft
  arXiv preprint arXiv:2408.00118.                             Research Blog, 1(3):3.
Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bha-         Jiaming Ji, Mickel Liu, Juntao Dai, Xuehai Pan, Chi
  gia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh               Zhang, Ce Bian, Chi Zhang, Ruiyang Sun, Yizhou
  Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang,               Wang, and Yaodong Yang. 2023. Beavertails: To-
  et al. 2024. Olmo: Accelerating the science of lan-            wards improved safety alignment of llm via a human-
  guage models. arXiv preprint arXiv:2402.00838.                 preference dataset. arXiv preprint arXiv:2307.04657.


                                                         11
Juyong Jiang, Fan Wang, Jiasi Shen, Sungju Kim,                   and Ruiming Tang. 2024. Coir: A comprehensive
  and Sunghun Kim. 2024a. A survey on large lan-                  benchmark for code information retrieval models.
  guage models for code generation. arXiv preprint                arXiv preprint arXiv:2407.02883.
  arXiv:2406.00515.
                                                                Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del
Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger,            Giorno, Suriya Gunasekar, and Yin Tat Lee. 2023.
  Faeze Brahman, Sachin Kumar, Niloofar Mireshghal-               Textbooks are all you need ii: phi-1.5 technical report.
  lah, Ximing Lu, Maarten Sap, Yejin Choi, and Nouha              Preprint, arXiv:2309.05463.
  Dziri. 2024b. Wildteaming at scale: From in-the-
  wild jailbreaks to (adversarially) safer language mod-        Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
  els. Preprint, arXiv:2406.18510.                                dar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
                                                                  Luke Zettlemoyer, and Veselin Stoyanov. 2019a.
Gregory Kamradt. 2023. Needle In A Haystack - pres-               Roberta: A robustly optimized BERT pretraining
  sure testing LLMs. Github.                                      approach. CoRR, abs/1907.11692.
Andrej Karpathy. 2023. The most dramatic optimization           Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
  to nanogpt so far ( 25% speedup) is to simply increase          dar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
  vocab size from 50257 to 50304 (nearest multiple of             Luke Zettlemoyer, and Veselin Stoyanov. 2019b.
  64).                                                            Roberta: A robustly optimized BERT pretraining
                                                                  approach. CoRR, abs/1907.11692.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
  S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen,            Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen
  and Wen-tau Yih. 2020. Dense passage retrieval for              Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong,
  open-domain question answering. In Proceedings of               Ernie Chang, Yangyang Shi, Raghuraman Krish-
  the 2020 Conference on Empirical Methods in Nat-                namoorthi, Liangzhen Lai, and Vikas Chandra. 2024.
  ural Language Processing, EMNLP 2020, Online,                   Mobilellm: Optimizing sub-billion parameter lan-
  November 16-20, 2020, pages 6769–6781. Associa-                 guage models for on-device use cases. Preprint,
  tion for Computational Linguistics.                             arXiv:2402.14905.
Omar Khattab and Matei Zaharia. 2020. Colbert: Ef-              Ilya Loshchilov and Frank Hutter. 2019. Decoupled
 ficient and effective passage search via contextual-              weight decay regularization. In International Confer-
 ized late interaction over BERT. In Proceedings of                ence on Learning Representations.
 the 43rd International ACM SIGIR conference on
 research and development in Information Retrieval,             The Mosaic ML Team. 2021. composer. https://
 SIGIR 2020, Virtual Event, China, July 25-30, 2020,              github.com/mosaicml/composer/.
 pages 39–48. ACM.
                                                                Thao Nguyen, Maithra Raghu, and Simon Kornblith.
Mario Michael Krell, Matej Kosec, Sergio P. Perez, and            2021. Do wide and deep networks learn the same
 Andrew Fitzgibbon. 2022. Efficient sequence pack-                things? uncovering how neural network representa-
  ing without cross-contamination: Accelerating large             tions vary with width and depth. In International
  language models without impacting performance.                  Conference on Learning Representations.
 Preprint, arXiv:2107.02027.
                                                                Zach Nussbaum, John X. Morris, Brandon Duderstadt,
Benjamin Lefaudeux, Francisco Massa, Diana                        and Andriy Mulyar. 2024. Nomic embed: Training
  Liskovich, Wenhan Xiong, Vittorio Caggiano,                     a reproducible long context text embedder. CoRR,
  Sean Naren, Min Xu, Jieru Hu, Marta Tintore,                    abs/2402.01613.
  Susan Zhang, Patrick Labatut, Daniel Haziza,
  Luca Wehrstedt, Jeremy Reizenstein, and Grigory               Guilherme Penedo, Hynek Kydlíček, Loubna Ben al-
  Sizov. 2022. xformers: A modular and hackable                   lal, Anton Lozhkov, Margaret Mitchell, Colin Raffel,
  transformer modelling library. https://github.                  Leandro Von Werra, and Thomas Wolf. 2024. The
  com/facebookresearch/xformers.                                  fineweb datasets: Decanting the web for the finest
                                                                  text data at scale. Preprint, arXiv:2406.17557.
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hin-
   ton. 2016. Layer normalization. ArXiv e-prints,              Jacob Portes, Alexander Trott, Sam Havens, Daniel
   pages arXiv–1607.                                               King, Abhinav Venigalla, Moin Nadeem, Nikhil Sar-
                                                                   dana, Daya Khudia, and Jonathan Frankle. 2023. Mo-
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio               saicbert: A bidirectional encoder optimized for fast
  Petroni, Vladimir Karpukhin, Naman Goyal, Hein-                  pretraining. In Advances in Neural Information Pro-
  rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-                 cessing Systems 36: Annual Conference on Neural
  täschel, et al. 2020. Retrieval-augmented genera-                Information Processing Systems 2023, NeurIPS 2023,
  tion for knowledge-intensive nlp tasks. Advances in             New Orleans, LA, USA, December 10 - 16, 2023.
  Neural Information Processing Systems (NeurIPS),
  33:9459–9474.                                                 Rushi Qiang, Ruiyi Zhang, and Pengtao Xie. 2024.
                                                                  Bilora: A bi-level optimization framework for
Xiangyang Li, Kuicai Dong, Yi Quan Lee, Wei Xia,                  overfitting-resilient low-rank adaptation of large pre-
  Yichun Yin, Hao Zhang, Yong Liu, Yasheng Wang,                  trained models. CoRR, abs/2403.13037.


                                                           12
Alec Radford, Karthik Narasimhan, Tim Salimans, and             Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta
  Ilya Sutskeve. 2018. Improving language understand-             Raileanu, Maria Lomeli, Eric Hambro, Luke Zettle-
  ing by generative pre-training. In OpenAI Tech Re-              moyer, Nicola Cancedda, and Thomas Scialom. 2023.
  port.                                                           Toolformer: Language models can teach themselves
                                                                  to use tools. In Advances in Neural Information Pro-
Alec Radford, Jeff Wu, Rewon Child, David Luan,                   cessing Systems 36: Annual Conference on Neural
  Dario Amodei, and Ilya Sutskever. 2019. Language                Information Processing Systems 2023, NeurIPS 2023,
  models are unsupervised multitask learners. OpenAI              New Orleans, LA, USA, December 10 - 16, 2023.
  Blog.
                                                                Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay
Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie                Thakkar, Pradeep Ramani, and Tri Dao. 2024.
   Millican, Jordan Hoffmann, Francis Song, John                  Flashattention-3: Fast and accurate attention with
  Aslanides, Sarah Henderson, Roman Ring, Susan-                  asynchrony and low-precision. arXiv preprint
   nah Young, Eliza Rutherford, Tom Hennigan, Ja-                 arXiv:2407.08608.
   cob Menick, Albin Cassirer, Richard Powell, George
  van den Driessche, Lisa Anne Hendricks, Mari-                 Noam Shazeer. 2020. Glu variants improve transformer.
   beth Rauh, Po-Sen Huang, Amelia Glaese, Jo-                    arXiv preprint arXiv:2002.05202.
   hannes Welbl, Sumanth Dathathri, Saffron Huang,
  Jonathan Uesato, John Mellor, Irina Higgins, Anto-            Noam Shazeer and Mitchell Stern. 2018. Adafactor:
   nia Creswell, Nat McAleese, Amy Wu, Erich Elsen,               Adaptive learning rates with sublinear memory cost.
   Siddhant Jayakumar, Elena Buchatskaya, David Bud-              In Proceedings of the 35th International Conference
   den, Esme Sutherland, Karen Simonyan, Michela Pa-              on Machine Learning, volume 80 of Proceedings
   ganini, Laurent Sifre, Lena Martens, Xiang Lorraine            of Machine Learning Research, pages 4596–4604.
   Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena                   PMLR.
   Gribovskaya, Domenic Donato, Angeliki Lazaridou,
                                                                Mohammad Shoeybi, Mostofa Patwary, Raul Puri,
  Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsim-
                                                                 Patrick LeGresley, Jared Casper, and Bryan Catan-
   poukelli, Nikolai Grigorev, Doug Fritz, Thibault Sot-
                                                                 zaro. 2019. Megatron-lm: Training multi-billion
   tiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong,
                                                                 parameter language models using model parallelism.
   Daniel Toyama, Cyprien de Masson d’Autume, Yujia
                                                                 arXiv preprint arXiv:1909.08053.
   Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin,
  Aidan Clark, Diego de Las Casas, Aurelia Guy,                 Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng
   Chris Jones, James Bradbury, Matthew Johnson,                   Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: En-
   Blake Hechtman, Laura Weidinger, Iason Gabriel,                 hanced transformer with rotary position embedding.
  William Isaac, Ed Lockhart, Simon Osindero, Laura                Neurocomputing, 568:127063.
   Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub,
  Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Ko-           Yi Tay, Mostafa Dehghani, Jinfeng Rao, William Fedus,
   ray Kavukcuoglu, and Geoffrey Irving. 2022. Scaling             Samira Abnar, Hyung Won Chung, Sharan Narang,
   language models: Methods, analysis & insights from              Dani Yogatama, Ashish Vaswani, and Donald Met-
   training gopher. Preprint, arXiv:2112.11446.                    zler. 2022. Scale efficiently: Insights from pretrain-
                                                                   ing and finetuning transformers. In International
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine                Conference on Learning Representations (ICLR) 22.
  Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
  Wei Li, and Peter J Liu. 2020. Exploring the lim-             Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
  its of transfer learning with a unified text-to-text            hishek Srivastava, and Iryna Gurevych. 2021. BEIR:
  transformer. Journal of machine learning research,              A heterogeneous benchmark for zero-shot evaluation
  21(140):1–67.                                                   of information retrieval models. In Proceedings of
                                                                  the Neural Information Processing Systems Track on
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:             Datasets and Benchmarks 1, NeurIPS Datasets and
  Sentence embeddings using siamese bert-networks.                Benchmarks 2021, December 2021, virtual.
  In Proceedings of the 2019 Conference on Empirical
  Methods in Natural Language Processing. Associa-              Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
  tion for Computational Linguistics.                             bert, Amjad Almahairi, Yasmine Babaei, Nikolay
                                                                  Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
David Samuel. 2024. Berts are generative in-context               Bhosale, et al. 2023. Llama 2: Open founda-
  learners. CoRR, abs/2406.04823.                                 tion and fine-tuned chat models. arXiv preprint
                                                                  arXiv:2307.09288.
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon,
  Christopher Potts, and Matei Zaharia. 2022. Col-              Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke
  bertv2: Effective and efficient retrieval via                   Bates, Daniel Korat, Moshe Wasserblat, and Oren
  lightweight late interaction. In Proceedings of the             Pereg. 2022. Efficient few-shot learning without
  2022 Conference of the North American Chapter of                prompts. arXiv preprint.
  the Association for Computational Linguistics: Hu-
  man Language Technologies, NAACL 2022, Seattle,               Aäron van den Oord, Yazhe Li, and Oriol Vinyals. 2018.
  WA, United States, July 10-15, 2022, pages 3715–                Representation learning with contrastive predictive
  3734. Association for Computational Linguistics.                coding. CoRR, abs/1807.03748.


                                                           13
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob               Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng,
  Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz                 Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan
  Kaiser, and Illia Polosukhin. 2017. Attention is all           Lan, Liwei Wang, and Tie-Yan Liu. 2020. On layer
  you need. In Advances in Neural Information Pro-               normalization in the transformer architecture. In Pro-
  cessing Systems 30: Annual Conference on Neural                ceedings of the 37th International Conference on
  Information Processing Systems 2017, December 4-9,             Machine Learning, ICML 2020, 13-18 July 2020, Vir-
  2017, Long Beach, CA, USA, pages 5998–6008.                    tual Event, volume 119 of Proceedings of Machine
                                                                 Learning Research, pages 10524–10533. PMLR.
Ellen Voorhees, Tasmeer Alam, Steven Bedrick, Dina
   Demner-Fushman, William R Hersh, Kyle Lo, Kirk              Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao,
   Roberts, Ian Soboroff, and Lucy Lu Wang. 2021.                 and Junyang Lin. 2019. Understanding and improv-
   Trec-covid: constructing a pandemic information re-            ing layer normalization. Advances in neural informa-
   trieval test collection. In ACM SIGIR Forum, vol-              tion processing systems, 32.
   ume 54, pages 1–12. ACM New York, NY, USA.
                                                               Hong Xuan, Abby Stylianou, Xiaotong Liu, and Robert
Alex Wang, Amanpreet Singh, Julian Michael, Felix                Pless. 2020. Hard negative examples are hard, but
  Hill, Omer Levy, and Samuel Bowman. 2018. GLUE:                useful. In Computer Vision–ECCV 2020: 16th Euro-
  A multi-task benchmark and analysis platform for nat-          pean Conference, Glasgow, UK, August 23–28, 2020,
  ural language understanding. In Proceedings of the             Proceedings, Part XIV 16, pages 126–142. Springer.
  2018 EMNLP Workshop BlackboxNLP: Analyzing                   An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
  and Interpreting Neural Networks for NLP, pages                Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
  353–355, Brussels, Belgium. Association for Com-               Li, Dayiheng Liu, Fei Huang, et al. 2024. Qwen2
  putational Linguistics.                                        technical report. arXiv preprint arXiv:2407.10671.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing                  Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
  Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,               Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023.
  and Furu Wei. 2022. Text embeddings by weakly-                 React: Synergizing reasoning and acting in language
  supervised contrastive pre-training. arXiv preprint            models. In The Eleventh International Conference
  arXiv:2212.03533.                                              on Learning Representations, ICLR 2023, Kigali,
                                                                 Rwanda, May 1-5, 2023. OpenReview.net.
Benjamin Warner. 2023. optimı̄: Fast, modern, memory
  efficient, and low precision pytorch optimizers.             Urchade Zaratiana, Nadi Tomeh, Pierre Holat, and
                                                                 Thierry Charnois. 2024. Gliner: Generalist model for
Charles Welch, Rada Mihalcea, and Jonathan K. Kum-               named entity recognition using bidirectional trans-
  merfeld. 2020. Improving low compute language                  former. In Proceedings of the 2024 Conference of
  modeling with in-domain embedding initialisation.              the North American Chapter of the Association for
  Preprint, arXiv:2009.14109.                                    Computational Linguistics: Human Language Tech-
                                                                 nologies (Volume 1: Long Papers), pages 5364–5376.
Alexander Wettig, Tianyu Gao, Zexuan Zhong, and
  Danqi Chen. 2023. Should you mask 15% in masked              Jinle Zeng, Min Li, Zhihua Wu, Jiaqi Liu, Yuang Liu,
  language modeling? Preprint, arXiv:2202.08005.                  Dianhai Yu, and Yanjun Ma. 2022. Boosting dis-
                                                                  tributed training performance of the unpadded bert
Adina Williams, Nikita Nangia, and Samuel Bowman.                 model. arXiv preprint arXiv:2208.08124.
  2018. A broad-coverage challenge corpus for sen-             Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby,
  tence understanding through inference. In Proceed-             and Lucas Beyer. 2022. Scaling vision transformers.
  ings of the 2018 Conference of the North American              In IEEE/CVF Conference on Computer Vision and
  Chapter of the Association for Computational Lin-              Pattern Recognition, CVPR 2022, New Orleans, LA,
  guistics: Human Language Technologies, Volume 1                USA, June 18-24, 2022, pages 1204–1213. IEEE.
  (Long Papers), pages 1112–1122.
                                                               Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie,
Mitchell Wortsman, Tim Dettmers, Luke Zettle-                    Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang,
  moyer, Ari Morcos, Ali Farhadi, and Ludwig                     Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie
  Schmidt. 2023. Stable and low-precision training               Li, and Min Zhang. 2024. mgte: Generalized long-
  for large-scale vision-language models. Preprint,              context text representation and reranking models for
  arXiv:2304.13013.                                              multilingual text retrieval. In Proceedings of the 2024
                                                                 Conference on Empirical Methods in Natural Lan-
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas                guage Processing: EMNLP 2024 - Industry Track,
  Muennighoff. 2023. C-pack: Packaged resources                  Miami, Florida, USA, November 12-16, 2024, pages
  to advance general chinese embedding. Preprint,                1393–1412. Association for Computational Linguis-
  arXiv:2309.07597.                                              tics.
Chen Xing, Devansh Arpit, Christos Tsirigotis, and             Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang,
  Yoshua Bengio. 2018. A walk with sgd. Preprint,               Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
  arXiv:1802.08770.                                             Zhang, Junjie Zhang, Zican Dong, et al. 2023. A


                                                          14
    survey of large language models. arXiv preprint           A.4    Final Checkpoints
    arXiv:2303.18223.
                                                              Inspired by recent work showing that checkpoint
A     Training Settings                                       averaging yields stronger final models (Dubey
                                                              et al., 2024; Clavié, 2024), we selected our final
Detailed training settings can be found in Table 3.           checkpoints by experimenting with various aver-
   During training we used MNLI as a live evalu-              aging methods and evaluating them on a subset
ation, along with validation loss and token accu-             of evaluation tasks. In no cases did Exponen-
racy metrics on a 500 million randomly sampled                tial Moving Average during annealing, as used by
sequences from the source datasets.                           Dubey et al. (2024), result in stronger performance.
   We use Composer (Mosaic ML Team, 2021) as                  ModernBERT-base is the result of averaging the 3
our training framework and optimı̄ (Warner, 2023)             best performing annealing checkpoints with the fi-
for our optimizer implementations.                            nal one. Averaging did not yield successful results
                                                              on the large size, ModernBERT-Large model is the
A.1    Batch Size Schedule
                                                              best performing annealing checkpoint.
Batch size warmup is a common-knowledge trick
to speed up model training when working with                  B     Model Design
medium to large batch sizes. Instead of "wasting" a
                                                              From Anthony et al. (2024), in addition to setting
full batch on updating the suboptimal initial weight
                                                              attention heads as multiples of 64 and setting the
distribution, we update the model weights on a
                                                              embedding matrix as a power of 2 or multiple of
gradually increasing batch size. Batch size warmup
                                                              64, there are three model design choices to max-
is usually longer than learning rate warmup, and
                                                              imize performance (assuming float16 or bfloat16
can be thought of as providing a higher initial learn-
                                                              computation):
ing rate with a mini-learning rate decay to the de-
fined learning rate schedule. We warmup Mod-                      • Tensor Core Requirement: Weight matrix
ernBERT’s batch size from 768 to 4,608 over 50                      dimensions should be divisible by 64
billion tokens and from 448 to 4,928 over 10 billion
tokens, for -base and -large, respectively, with an               • Tile Quantization: Weight matrix is divisible
uneven token schedule so each batch size has the                    into 128 × 256 blocks.
same number of update steps.
                                                                  • Wave Quantization: Number of blocks is
A.2    Weight Tiling                                                divisible by the number of streaming multi-
Following the Phi family of models (Li et al.,                      processors (SM).
2023; Javaheripi et al., 2023), we initialized                   Given that we wanted to target good performance
ModernBERT-large directly from ModernBERT-                    across multiple GPUs with a wide variety of SM
base’s pretraining weights using center tiling and            counts, wave quantization is an impossible ask. So
Gopher layer scaling (Rae et al., 2022). Since                we selected a basket of GPUs (NVIDIA T4, A10,
Base’s weight matrices are smaller than Large’s,              L4, RTX 3090, RTX 4090, A100, and H100) and
we centered Base’ weights, accounting for each                calculated the approximate SM utilization for each
token embedding and attention head, then filled               by dividing the modulus blocks by the number of
rest the of the weights using wraparound. Like Phi,           SMs. This appeared to be a decent performance
we tested center initialization with random edge              heuristic in our spot checking. We then designed
values and tiling from an edge, but both of these un-         our models to maximize performance on the basket
derperformed center tiling with wraparound. This              of GPUs, putting more weight on inference GPUs.
weight initialization strategy greatly accelerates
ModernBERT-large’s initial training.                          C     Training Log
A.3    Weight Decay                                           C.1    Sampling Issue
We did not apply weight decay to the bias terms               Our first pretraining run of ModernBERT-base
or normalization layers. Instead of PyTorch-style             ended in disaster as the loss exhibited a slow see-
decoupled weight decay, we applied fully decou-               saw pattern before slowly diverging. Despite us-
pled weight decay following Loshchilov and Hutter             ing PyTorch’s distributed random sampler, train-
(2019).                                                       ing metrics suggested that the model was training

                                                         15
                                       Pretraining Phase         Context Extension: Phase One       Context Extension: Phase Two
                                   Base             Large            Base           Large                  Base          Large
 Training Tokens                         1.719 trillion                      250 billion                          50 billion
 Max Sequence Length                        1,024                              8,192                                8,192
 Batch Size                        4,608            4,928            72               77                   72              78
   Warmup (tokens)               50 billion       10 billion          -                -                    -               -
 Microbatch Size                     96              56              12               7                    12              6
 Learning Rate                     8e-4        5e-4, 5e-5            3e-4            5e-5                  3e-4           5e-5
   Schedule                             Trapezoidal                    -               -                            1-sqrt
   Warmup (tokens)               3 billion      2 billion              -               -                    -               -
   Decay (tokens)                    -              -                  -               -                          50 billion
 Weight Decay                      1e-5        1e-5, 1e-6            1e-5            1e-6                  1e-5           1e-6
 Total Time (hours)                194.2            425.3            39.9            80.7                  11.5           21.7
 Training Time (hours)             191.1            420.4            36.3            75.1                   7.5           15.3
 Model Initialization            Megatron         From Base              -             -                    -              -
 Dropout (attn out)              0.1
 Dropout (all other layers)      0.0
 Optimizer                       StableAdamW
  Betas                          (0.90, 0.98)
  Epsilon                        1e-06
 Training Hardware               8x H100
 Training Strategy               Distributed DataParallel
 Software Libraries              PyTorch 2.4.0, Cuda 12.4.0, Composer 0.24.1, Flash Attention 2.6.3, FA3 commit 32792d3

              Table 3: ModernBERT training settings. Dropout and below are shared across all phases.

                                                                     Base                   Large
                               Vocabulary                           50,368                 50,368
                               Unused Tokens                           83                     83
                               Layers                                  22                    28
                               Hidden Size                            768                   1024
                               Transformer Block                  Pre-Norm               Pre-Norm
                               Activation Function                  GeLU                   GeLU
                               Linear Bias                           False                  False
                               Attention                          Multi-head             Multi-head
                               Attention Heads                         12                    16
                               Global Attention                Every three layers     Every three layers
                               Local Attention Window                 128                    128
                               Intermediate Size                     1,152                  2,624
                               GLU Expansion                         2,304                  5,248
                               Normalization                     LayerNorm              LayerNorm
                               Norm Epsilon                          1e-5                   1e-5
                               Norm Bias                             False                  False
                               RoPE theta                          160,000                160,000
                               Local Attn RoPE theta                10,000                 10,000

                                              Table 4: ModernBERT model design


on the dataset in a non-random order. Like the                           pler with NumPy’s PCG64DXSM random sampler.
Olmo authors15 , we determined that the PyTorch
random sampler returns sequentially biased sam-                          C.2    Large Rollback
ples when the number of samples is somewhere                             We rolled back and restarted ModernBERT-large
between 500 million and 1 billion samples16 . We                         training at a lower learning rate of 5e-5 and lower
resolved this issue by replacing the PyTorch sam-                        weight decay of 1e-6 for the last 800 billion to-
                                                                         kens. Prior to restarting training, large’s training
  15
     We found a comment and GitHub issue about this in the               loss, validation metrics, and live evaluations on
Olmo codebase after resolving the issue ourselves.
  16
     We did not conduct a rigorous statistical analysis to deter-        MNLI had plateaued for a few hundred billion to-
mine exactly when this happens.                                          kens at the higher 5e-4 learning rate. In contrast,

                                                                    16
ModernBERT-base showed a continuous, but di-                        however possible that larger encoders and/or
minishing, improvement on training loss, valida-                    larger sequence lengths might see a different
tion metrics, and live evaluations through the entire               trade-off.
1.719 trillion token training phase. This highlights
one of the risks of training with a constant learning             • We explored the use of alternating global/local
rate, other learning rate schedules can mitigate se-                attention, with global attention every 3 layers
lecting a too high learning rate (or too small batch                and local attention over a 128 token sliding
size) by lowering the learning rate throughout train-               window otherwise. This setup yielded identi-
ing.                                                                cal downstream performance when compared
                                                                    to the use of global attention in every layer,
D    Architecture ablations                                         even at 100 billion tokens, while resulting in
                                                                    major speedups.
To select the updates to add in the ModernBERT
architecture, we performed different ablations, ex-               • We experimented with multiple tokenizers, be-
cept where stated, most ablations where ran at the                  fore selecting our final one, based on a mod-
8-20 billion token scale:                                           ified OLMo (Groeneveld et al., 2024) tok-
                                                                    enizer, which performed the best out of the
    • We compared two GLU layers, GeGLU and                         recent tokenizers evaluated. Tokenizers from
      SwiGLU. We find close to no difference be-                    the BERT and RoBERTa generation of en-
      tween the two and choose to use GeGLU lay-                    coder models had competitive downstream
      ers.                                                          performance on MNLI, but we theorized that
    • Using different percentage of the head dimen-                 their lack of recent training data and lack of
      sion for the RoPE dimension (50, 75, 100).                    code support would hinder downstream appli-
      Lower percentages gave slightly better results.               cations. Interestingly, we observed significant
      However, the observed difference was min-                     downstream performance degradation when
      imal. As the ablations were conducted at a                    using the Llama 2 (Touvron et al., 2023) tok-
      considerably smaller scale than the final train-              enizer.
      ing, we choose to err on the side of caution
                                                              E     Extended results
      and opt to keep the dimension at 100 % to
      avoid potentially hindering the capabilities of         E.1    Full GLUE results
      the fully trained models.                               The results for all the models each GLUE subsets
    • Both LayerNorm and RMSNorm yielded very                 are presented in Table 5. The values for prior mod-
      similar results. While RMSNorm is theo-                 els are extracted from the literature. As mentioned
      retically faster, at the time this work was             in Section 3.1.1, we follow standard practice (Liu
      conducted, PyTorch did not have a native                et al., 2019a; Portes et al., 2023; He et al., 2023)
      RMSNorm implementation, leading to eager-               and conduct an hyperparameter search on each
      mode RMSNorm being the default implemen-                subset. More specifically, we perform a sweep
      tation used for many users. To ensure Modern-           over learning rates in [1e−5, 3e−5, 5e−5, 8e−5],
      BERT has the highest possible out-of-the-box            weight decay in [1e−6, 5e−6, 8e−6, 1e−5], and
      efficiency, we choose to use LayerNorm in the           number of epochs in [1, 2, 3] for tasks in SST-2,
      final models.                                           MNLI, and RTE, and [2, 5, 10] for tasks in QNLI,
                                                              QQP, CoLA, MRPC, and STS-B. The final values
    • We investigated using parallel attention to             are detailed in Table 6. Early stopping is used for
      compute the MLP and attention matrices at               all the fine-tuning runs which reduces the overall
      the same time, which has been shown to in-              fine-tuning time considerably. RTE MRPC and
      crease processing speeds for larger model               STS-B checkpoints are trained starting from the
      sizes (Chowdhery et al., 2023). However, for            MNLI checkpoint.
      models within our targe sizes and pre-training
      sequence length, the speed-up we observed               E.2    Full BEIR results
      was minimal while we encountered signifi-               In the main body, we only report the average
      cant degradation in downstream performance.             score over the 15 very diverse datasets of BEIR.
      As such, we do not use parallel attention. It is        We report the results on every subsets for both

                                                         17
                                             Single Sentence      Paraphrase and Similarity   Natural Language Inference
         Model             Params    Seq.    CoLA     SST-2       MRPC     STS-B     QQP      MNLI     QNLI      RTE
             β
         BERT               110M      512     59.0    93.1         89.5      89.4     91.4    85.4     91.6      78.2
         RoBERTaα           125M      512     63.6    94.8         90.2      91.2     91.9    87.6     92.8      78.7
         DeBERTav3ϵ         183M      512     69.2    95.6         89.5      91.6     92.4    90.0     94.0      83.8
 Base    MosaicBERT-128β    137M      128     58.2    93.5         89.0      90.3     92.0    85.6     91.4      83.0
         NomicBERT-2048γ    137M     2048     50.0    93.0         88.0      90.0     92.0    86.0     92.0      82.0
         GTE-en-MLMδ        137M     8192     57.0    93.4         92.1      90.2     88.8    86.7     91.9      84.8
         ModernBERT         149M     8192     65.1    96.0         92.2      91.8     92.1    89.1     93.9      87.4
         BERTβ              330M      512     56.2    93.3         87.8      90.6     90.9    86.3     92.8      83.8
         RoBERTaα           355M      512     68.0    96.4         90.9      92.4     92.2    90.2     94.7      86.6
 Large   DeBERTav3ζ         434M      512     75.3    96.9         92.2      93.0     93.3    91.8     96.0      92.7
         GTE-en-MLMδ        434M     8192     60.4    95.1         93.5      91.4     89.2    89.2     93.9      88.1
         ModernBERT         395M     8192     71.4    97.1         91.7      92.8     92.7    90.8     95.2      92.1

Table 5: GLUE (Wang et al., 2018) dev set scores. α taken from Table 8 of (Liu et al., 2019a), β taken from Table
S3 of (Portes et al., 2023), γ from Table 2 of (Nussbaum et al., 2024), δ from Table 21 of (Zhang et al., 2024), ϵ
from Table 2 of (Qiang et al., 2024) and ζ from Table 3 of (He et al., 2023)

                                               Base                        Large
                            Task        LR       WD       Ep         LR       WD       Ep
                            CoLA       8e−5      1e−6      5       3e−5      8e−6      5
                            MNLI       5e−5      5e−6     1        3e−5      1e−5      1
                            MRPC       5e−5      5e−6     10       8e−5      5e−6      2
                            QNLI       8e−5      5e−6     2        3e−5      5e−6      2
                            QQP        5e−5      5e−6     10       5e−5      8e−6      2
                            RTE        5e−5      1e−5     3        5e−5      8e−6      3
                            SST-2      8e−5      1e−5     2        1e−5      1e−6      3
                            STSB       8e−5      5e−6     10       8e−5      1e−5      10

Table 6: Fine-tuning hyperparameters for ModernBERT on GLUE tasks. LR: Learning Rate, WD: Weight Decay,
Ep: Epochs.


single and multi-vector retrieval in Table 7 and              F    Efficiency
Table 8 respectively. For both settings and for
every model, we perform a sweep for learning                 Full statistics of the synthetic datasets used to eval-
rates in [1e−5, 2e−5, 3e−5, 5e−5, 8e−5, 1e−4]                uate the efficiency of the models in Section 4 are
and choose the model obtaining the best average              given in Table 10. The detailed runtimes, alongside
result over a subset of datasets composed of NFCor-          with the maximum batch size for every model is
pus, SciFact, TREC-Covid and FiQA as the final               detailed in Table 11.
model. Best learning rates for every setting are                The high maximum batch-size achieved by Mod-
reported in Table 9. Although ModernBERT show-               ernBERT models, considerably higher than any
case strong results across the board, it should be           other models, highlight the strong memory effi-
noted that an important factor in its performance is         ciency of the model at both sizes. Inversely, it
TREC-COVID (Voorhees et al., 2021), potentially              is worth noting that while DeBERTaV3 has com-
showcasing the benefits of ModernBERT being                  petitive GLUE performance, it stands out as par-
trained with a more recent knowledge cutoff than             ticularly inefficient, both in its memory use and
most existing encoders. However, NomicBERT                   processing speed. Indeed, on both model sizes, De-
and GTE have also been trained on updated data,              BERTaV3’s memory use is 5-to-7 times higher than
so the cutoff cannot be the only factor affecting the        ModernBERT’s, and it processes inputs, two times
performance.                                                 slower even in the most favorable scenario where
                                                             all sequences are at the maximum possible length,
                                                             thus negating any advantage from unpadding.

                                                        18
         Model        NFCorpus     SciFact   TREC-Covid   FiQA   ArguAna   Climate-FEVER    DBPedia   FEVER   HotpotQA    MSMARCO    NQ     Quora   SciDocs   Touche2020   CQADupstack   Avg.
         BERT           24.3        51.3        49.5      22.8    31.6         21.9          28.2      64.1     47.9        58.5     37.9   83.1     12.9        20.4         28.5       38.9
         RoBERTa        20.4        45.6        52.2      26.1    35.2         22.3          23.1      60.2     45.0        56.0     34.7   84.0     11.4        21.1         28.8       37.7
         DeBERTaV3       8.0        22.6        48.4      11.5    26.1          9.7           5.3      17.3      8.0        25.2     12.5   74.7      5.4        14.2         14.2       20.2
 Base    NomicBERT      25.7        52.0        63.0      23.5    35.5         22.9          30.3      65.0     48.0        60.6     42.6   84.5     12.6        19.0         29.2       41.0
         GTE-en-MLM     26.3        54.1        49.7      30.1    35.7         24.5          28.9      66.5     49.9        63.1     41.7   85.2     14.1        19.1         32.5       41.4
         ModernBERT     23.7        57.0        72.1      28.8    35.7         23.6          23.8      59.9     46.1        61.6     39.5   85.9     12.5        20.8         33.1       41.6
         BERT           23.3        50.7        48.9      24.0    35.2         22.1          27.2      61.7     45.9        59.8     39.5   83.6     13.0        19.5         28.9       38.9
         RoBERTa        23.9        53.4        55.0      33.4    37.6         23.5          25.4      65.2     47.1        60.4     43.3   85.8     13.7        21.1         33.0       41.4
 Large   DeBERTaV3       9.6        31.2        56.6      15.8    26.3         14.4           6.8      29.4     15.3        32.4     21.5   79.1      7.0        18.8         19.9       25.6
         GTE-en-MLM     27.7        57.6        48.4      34.0    35.3         24.0          27.0      65.4     50.8        64.1     44.9   85.3     15.6        21.4         35.5       42.5
         ModernBERT     26.2        60.4        74.1      33.1    38.2         20.5          25.1      62.7     49.2        64.9     45.5   86.5     13.8        23.1         36.5       44.0



                      Table 7: BEIR (Thakur et al., 2021) nDCG@10 scores for single-vector retrieval models.

         Model        NFCorpus     SciFact   TREC-Covid   FiQA   ArguAna   Climate-FEVER    DBPedia   FEVER   HotpotQA    MSMARCO    NQ     Quora   SciDocs   Touche2020   CQADupstack   Avg.
         BERT           34.2        71.5        69.9      35.0    49.9         19.2          42.4      83.1     69.8        45.4     55.4   84.1     14.7        27.0         34.2       49.0
         RoBERTa        33.7        70.8        69.8      37.4    48.9         18.9          39.3      81.2     66.1        43.7     56.3   83.6     14.8        31.7         34.4       48.7
         DeBERTaV3      31.9        68.5        75.5      35.5    46.5         18.3          35.6      78.1     65.3        39.5     50.4   83.7     14.6        31.1         32.3       47.1
 Base    NomicBERT      35.5        72.2        73.5      35.9    44.8         19.0          43.6      83.9     71.1        46.3     58.5   84.0     15.1        31.3         33.9       49.9
         GTE-en-MLM     35.1        71.5        69.4      36.0    48.5         17.4          41.2      79.9     67.0        44.4     52.8   85.2     15.0        25.4         34.6       48.2
         ModernBERT     35.2        73.0        80.5      38.0    49.1         22.2          42.0      85.8     70.4        45.4     57.1   86.3     16.0        33.9         35.1       51.3
         BERT           34.6        72.9        68.8      35.5    48.3         19.7          42.4      83.6     70.7        45.9     57.2   84.8     15.2        28.9         34.9       49.5
         RoBERTa        35.0        72.3        74.4      38.7    50.0         19.6          41.0      82.0     66.2        44.7     57.5   85.9     15.3        27.9         36.0       49.8
 Large   DeBERTaV3      31.7        70.2        73.3      35.0    46.2         18.0          36.5      79.0     63.2        39.4     51.6   81.1     14.1        28.6         33.1       46.7
         GTE-en-MLM     35.2        72.4        67.2      39.6    50.3         20.8          44.4      82.5     72.0        47.0     60.1   86.4     15.9        30.9         35.4       50.7
         ModernBERT     36.0        73.2        81.3      40.3    50.3         22.3          44.1      85.8     72.5        46.0     59.9   86.1     16.9        34.6         35.9       52.4



                      Table 8: BEIR (Thakur et al., 2021) nDCG@10 scores for multi-vector retrieval models.

                                             Model                         Single-vector (DPR)                     Multi-vector (ColBERT)
                                             BERT                                     5 × 10−5                                     8 × 10−5
                                             RoBERTa                                  3 × 10−5                                     8 × 10−5
                                             DeBERTaV3                                8 × 10−5                                     5 × 10−5
                                 Base        NomicBERT                                5 × 10−5                                     1 × 10−4
                                             GTE-en-MLM                               5 × 10−5                                     8 × 10−5
                                             ModernBERT                               8 × 10−5                                     1 × 10−4
                                             BERT                                     3 × 10−5                                     1 × 10−4
                                             RoBERTa                                  3 × 10−5                                     1 × 10−5
                                 Large       DeBERTaV3                                8 × 10−5                                     1 × 10−5
                                             GTE-en-MLM                               3 × 10−5                                     3 × 10−5
                                             ModernBERT                               1 × 10−4                                     3 × 10−5

Table 9: Learning rate used for reported results on BEIR (Thakur et al., 2021) for both single and multi vector
retrieval

                                                                                           Short                                        Long
                                                                                 Fixed                Variable                     Fixed             Variable
                           Total Token Count                               4,194,304            2,096,510                67,108,864            33,604,913
                           Standard deviation                                      0                   64                         0                 1,024
                           Average Length                                        512                  256                     8,192                 4,102
                           Longest sequence                                      512                  476                     8,192                 7,624
                           Shortest sequence                                     512                   32                     8,192                   171
                           Number of sequences                                 8,192                8,192                     8,192                 8,192

                           Table 10: Token statistics for the synthetic datasets used in efficiency evaluations.


G          Licensing
We release the ModernBERT model architecture,
model weights, and training codebase under the
Apache 2.0 license.


                                                                                               19
                                                      Short                                Long
        Model                 Params      BS          Fixed      Variable   BS            Fixed        Variable
        BERT                   110M     1096    23.3 ± 0.02             –    –                –               –
        RoBERTa                125M      664    23.3 ± 0.19             –    –                –               –
        DeBERTaV3              183M      236    59.7 ± 0.11             –    –                –               –
Base    NomicBERT              137M      588    35.8 ± 0.01             –   36    1455.5 ± 0.31               –
        GTE-en-MLM             137M      640    33.9 ± 1.21             –   38    1434.7 ± 3.69               –
        GTE-en-MLMxformers     137M      640    34.2 ± 0.10   16.3 ± 0.04   38    1412.6 ± 3.19    499.2 ± 0.11
        ModernBERT             149M     1604    28.3 ± 0.55   14.2 ± 0.01   98     542.4 ± 0.20    251.2 ± 0.32
        BERT                   330M      792    77.1 ± 1.50             –    –                –               –
        RoBERTa                355M      460    99.8 ± 1.79             –    –                –               –

Large
        DeBERTaV3              434M      134   170.8 ± 0.06             –    –                –               –
        GTE-en-MLM             435M      472   108.4 ± 0.07             –   28    4144.7 ± 0.05               –
        GTE-en-MLMxformers     435M      472   109.0 ± 0.14   51.9 ± 0.02   28    4059.1 ± 4.55   1476.3 ± 0.94
        ModernBERT             395M      770    80.1 ± 1.65   39.6 ± 0.02   48    1433.9 ± 0.99    674.9 ± 0.15

        Table 11: Inference runtime for all models. Bold indicates the best for the column within two SDs.




                                                       20

