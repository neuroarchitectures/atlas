# Large Concept Models Language Modeling in a Sentence Represe Alastruey Andrews Coria 2025

> Source: `Large_Concept_Models_Language_Modeling_in_a_Sentence_Represe_Alastruey_Andrews_Coria_2025.pdf`

---

                                             Large Concept Models:
                                             Language Modeling in a Sentence Representation Space
                                             LCM team , Loïc Barrault∗ , Paul-Ambroise Duquenne∗ , Maha Elbayad∗ , Artyom Kozhevnikov∗ ,
                                             Belen Alastruey† , Pierre Andrews† , Mariano Coria† , Guillaume Couairon+† , Marta R.
                                             Costa-jussà† , David Dale† , Hady Elsahar† , Kevin Heffernan† , João Maria Janeiro† , Tuan Tran† ,
                                             Christophe Ropers† , Eduardo Sánchez† , Robin San Roman† , Alexandre Mourachko‡ , Safiyyah
                                             Saleem‡ , Holger Schwenk‡

                                             FAIR at Meta




arXiv:2412.08821v2 [cs.CL] 15 Dec 2024
                                             ∗
                                              Core contributors, alphabetical order, † Contributors to data preparation, LCM extensions and
                                             evaluation, alphabetical order, ‡ Research and project management, alphabetical order, + Initial work
                                             while at FAIR at Meta, new affiliation: INRIA, France

                                             LLMs have revolutionized the field of artificial intelligence and have emerged as the de-facto tool for
                                             many tasks. The current established technology of LLMs is to process input and generate output at
                                             the token level. This is in sharp contrast to humans who operate at multiple levels of abstraction, well
                                             beyond single words, to analyze information and to generate creative content. In this paper, we present
                                             an attempt at an architecture which operates on an explicit higher-level semantic representation,
                                             which we name a “concept”. Concepts are language- and modality-agnostic and represent a higher
                                             level idea or action in a flow. Hence, we build a “Large Concept Model”. In this study, as
                                             proof of feasibility, we assume that a concept corresponds to a sentence, and use an existing sentence
                                             embedding space, SONAR, which supports up to 200 languages in both text and speech modalities.

                                             The Large Concept Model is trained to perform autoregressive sentence prediction in an embedding
                                             space. We explore multiple approaches, namely MSE regression, variants of diffusion-based generation,
                                             and models operating in a quantized SONAR space. These explorations are performed using 1.6B
                                             parameter models and training data in the order of 1.3T tokens. We then scale one architecture to a
                                             model size of 7B parameters and training data of about 2.7T tokens. We perform an experimental
                                             evaluation on several generative tasks, namely summarization and a new task of summary expansion.
                                             Finally, we show that our model exhibits impressive zero-shot generalization performance to many
                                             languages, outperforming existing LLMs of the same size. The training code of our models is freely
                                             available.a

                                             Date: December 12, 2024
                                             Correspondence: Holger Schwenk at schwenk@meta.com
                                                 a
                                                     https://github.com/facebookresearch/large_concept_model




                                         1   Introduction

                                         Large Language models (LLMs) are dominating current research in natural language processing,
                                         and with their recent extension to more modalities, namely images, video and speech, they seem to
                                         be considered as the de-facto technique to follow to approach human intelligence. LLMs achieve
                                         indeed impressive performance on a large variety of tasks, such as providing detailed answers for
                                         general knowledge questions, helping in performing long document analysis, or drafting different
                                         types of messages, and writing or debugging code. Building an LLM from scratch requires access to

                                                                                                 1
enormous computational resources to process ever larger amounts of data and train models, the size
of which now exceeds four hundred billion parameters. Knowledge acquisition in LLMs is heavily
data-driven and extending them to more languages or modalities usually requires injecting additional
(synthetic) data to cover them.

The landscape of available LLMs can be structured into open models such as Llama (The Llama3
team, 2024), Mistral (Jiang et al., 2024), Bloom (BigScience Workshop, 2023) or Falcon
(Almazrouei et al., 2023), on the one hand, and closed models such as Gemini (Gemini Team Google,
2024), GPT (OpenAI, 2024) or Claude (Anthropic, 2024), on the other. It is striking that all
these models are based on the same underlying architecture: a transformer-based, decoder-only
language model, which is pretrained to predict the next token, given a long context of preceding
tokens. Despite the undeniable success of LLMs and continued progress, all current LLMs miss
a crucial characteristic of human intelligence: explicit reasoning and planning at multiple levels of
abstraction. The human brain does not operate at the word level only. We usually have a top-down
process to solve a complex task or compose a long document: we first plan at a higher level the
overall structure, and then step-by-step, add details at lower levels of abstraction. One may argue
that LLMs are implicitly learning a hierarchical representation, but we stipulate that models with
an explicit hierarchical architecture are better suited to create coherent long-form output.

Imagine a researcher giving a fifteen-minute talk. In such a situation, researchers do not usually
prepare detailed speeches by writing out every single word they will pronounce. Instead, they outline
a flow of higher-level ideas they want to communicate. Should they give the same talk multiple
times, the actual words being spoken may differ, the talk could even be given in different languages,
but the flow of higher-level abstract ideas will remain the same. Similarly, when writing a research
paper or essay on a specific topic, humans usually start by preparing an outline that structures the
whole document into sections, which they then refine iteratively. Humans also detect and remember
dependencies between the different parts of a longer document at an abstract level. If we expand
on our previous research writing example, keeping track of dependencies means that we need to
provide results for each of the experiment mentioned in the introduction. Finally, when processing
and analyzing information, humans rarely consider every single word in a large document. Instead,
we use a hierarchical approach: we remember which part of a long document we should search to
find a specific piece of information.

To the best of our knowledge, this explicit hierarchical structure of information processing and
generation, at an abstract level, independent of any instantiation in a particular language or modality,
cannot be found in any of the current LLMs. In this work, we present a new approach which
moves away from processing at the token level and closer to (hierarchical) reasoning in an abstract
embedding space. This abstract embedding space is designed to be independent of the language or
modality in which the content is expressed; in other words, we aim to model the underlying reasoning
process at a purely semantic level, not its instantiation in a specific language. In order to verify our
approach, we limit our study to two levels of abstraction: subword tokens and concepts. We define a
concept as an abstract atomic idea. In practice, a concept would often correspond to a sentence in a
text document, or an equivalent speech utterance. We posit that a sentence is an appropriate unit to
achieve language independence, in opposition to single words. This is in sharp contrast to current
LLMs techniques which are heavily English centric and token based.

Our fundamental idea could be based on any fixed-size sentence embedding space for which an
encoder and decoder are available. In particular, we could aim to train a new embedding space
specifically optimized to our reasoning architecture. In this work, we chose an existing and freely
available sentence embedding, named SONAR (Duquenne et al., 2023b). SONAR supports text

                                                   2
   Figure 1 - Left: visualization of reasoning in an embedding space of concepts (task of summarization).
                 Right: fundamental architecture of an Large Concept Model (LCM).
                              ⋆: concept encoder and decoder are frozen.


input and output in 200 languages, speech input in 76 languages, and speech output in English. We
discuss the constraints and impact of this choice in Section 2.1, and share some ideas on alternative
embedding spaces in Section 6.
Figure 1-left visualizes reasoning in an embedding space with the example of a summarization task,
which is materialized by a function on the embedding space, mapping five concept representations
into two. Figure 1-right summarizes the overall architecture and processing flow. The input is first
segmented into sentences, and each one is encoded with SONAR to achieve a sequence of concepts,
i.e., sentence embeddings. This sequence of concepts is then processed by a Large Concept Model
(LCM) to generate at the output a new sequence of concepts. Finally, the generated concepts are
decoded by SONAR into a sequence of subwords. The encoder and decoder are fixed and are not
trained. It is important to highlight that the unchanged sequence of concepts at the output of
the LCM can be decoded into other languages or modalities without performing again the whole
reasoning process. In the same spirit, a particular reasoning operation such as summarization can
be performed in a zero-shot setting on input in any language or modality, since it solely operates
on concepts. To summarize, the LCM neither has information on the input language or modality
nor generates output in a particular language or modality. We explore multiple architectures to
train the LCM, in particular several variants of diffusion. Finally, we envision an additional level of
abstraction beyond concepts which could correspond to a short description of a paragraph or small
section. In Section 4.3 we report initial ideas on how conditioning and predicting such higher-level
representations can improve consistency of output generated by an LCM.
To some extent, the LCM architecture resembles the Jepa approach (LeCun, 2022) that also aims to
predict the representation of the next observation in an embedding space. However, unlike Jepa that
places more emphasis on learning a representation space in a self-supervised way, the LCM focuses
on accurate prediction in the existing embedding space.




                                                     3
The mains characteristics of our generic Large Concept Model approach are as follows:
• Reasoning at an abstract language- and modality-agnostic level beyond tokens:
  – We model the underlying reasoning process, not its instantiation in a particular language.
  – The LCM can be trained, i.e. acquire knowledge, on all languages and modalities at once,
    promising scalability in an unbiased way.
• Explicit hierarchical structure:
  – Better readability of long-form output by a human.
  – Facilitates local interactive edits by a user.
• Handling of long context and long-form output:
  – The complexity of a vanilla transformer model increases quadratically with the sequence length.
    This makes handling of large context windows challenging and several techniques have been
    developed to alleviate this problem, e.g., sparse attention (Child et al., 2019) or LSH attention
    (Kitaev et al., 2020). Our LCM operates on sequences which are at least an order of magnitude
    shorter.1
• Unparalleled zero-shot generalization:
  – Independently of the language or modality the LCM is pre-trained and fine-tuned on, it can be
    applied to any language and modality supported by the SONAR encoders, without the need of
    additional data or fine-tuning. We report results for multiple languages in the text modality.
• Modularity and extensibility:
  – Unlike multimodal LLMs that can suffer from modality competition (Aghajanyan et al., 2023;
    Chameleon team, 2024), concept encoders and decoders can be independently developed and
    optimized without any competition or interference.
  – New languages or modalities can be easily added for an existing system.
The goal of this paper is to provide a proof of concept of this high-level vision of an alternative
architecture to current best practice in language modeling. In the next section we present the main
design principles of our models and discuss several variants to build and train a Large Concept
Model. We discuss several designs to implement diffusion approaches with concept embeddings and
carefully study noise scheduling. This section is completed by a compute complexity comparison
with token-based LLMs. Section 3 is dedicated to the analysis of a larger 7B parameter model. We
discuss challenges when instruction fine-tuning this model on multiple generative tasks, and provide
a comparison with existing LLMs of comparable size. The paper concludes with a discussion of
related work, the current limitations and perspectives of our approach.
To foster research in this area, we make our LCM training code2 as well as SONAR encoders and
decoders3 for up to 200 languages and multiple modalities freely available.




  1
    We assume an average sentence length of 10–20 tokens.
  2
    https://github.com/facebookresearch/large_concept_model
  3
    https://github.com/facebookresearch/SONAR


                                                     4
2       Main Design Principles
In this section, we outline the main design principles of the LCM. We first describe the SONAR
embedding space with its encoders and decoders. Then, we discuss details of data preparation,
namely sentence segmentation i.e., how we split long documents into sentences. And finally, we
describe in details the different versions of LCMs introduced in this work.

2.1       The SONAR embedding space
The motivation of this work is to perform reasoning at a higher conceptual level than tokens. This
requires an embedding space which is highly semantic. We chose SONAR (Duquenne et al., 2023b)
since it achieves best performance on several semantic similarity metrics like xsim or xsim++ (Chen
et al., 2023b), and it was successfully used in large-scale bitext mining for translation (Seamless
Communication et al., 2023b).
The SONAR text embedding space was trained as an encoder/decoder architecture, with a fixed-size
bottleneck instead of cross-attention (see Figure 2). The criterion combines a machine translation
objective for 200 languages into and out of English, denoising auto-encoding and an explicit MSE loss
at the embedding bottleneck layer. Once the text embedding space was trained, a teacher-student
approach was applied to extend the SONAR space to the speech modality. More details on the
architecture and training procedure can be found in Duquenne et al. (2023b), and detailed speech
recognition and translation results in the appendix of Seamless Communication et al. (2023a).
Our LCM operates directly on SONAR concepts embeddings, hence, it can perform reasoning on
all supported languages and modalities. Table 1 compares the language coverage of several other
LLMs. The LCM supports substantially more languages than other models, in particular many
low-resource languages. In addition to the text modality, SONAR supports 76 languages for speech
input and speech output in English. We have also developed an experimental encoder for American
Sign language (ASL). All these encoders and decoders are freely available.4 Exact listings of the
supported languages can be found in the SONAR GitHub repository.


    4
        https://github.com/facebookresearch/SONAR




 Figure 2 - Encoder/decoder bottleneck architecture to train the SONAR text embeddings (right part of
             figure). Teacher-student approach to extend SONAR to the speech modality (left part).


                                                      5
                             Text              Speech             Image              Video
               Model    Input Output       Input Output       Input Output       Input Output
           Gemini         47       47        62        ✓        ✓        ✓         ✓         ✗
              GPT         85       85        ✓         ✓        ✓        ✓         ?         ✗
          Claude          37       37        ✓         ✓        ✓        ✓         ✗         ✗
           Bloom          46       46        ✗         ✗        ✓        ✓         ✗         ✗
      Llama 3-400B        8        8         34        ✗        ✓        ✓         ✗         ✗
      LCM-SONAR          200       200      76         1        ✗        ✗      (ASL)        ✗

Table 1 - Comparison of language and modality coverage for several LLMs and our LCM operating on the
SONAR embedding space. SONAR has an experimental support for American Sign Language (ASL) which
                                      is not used in this paper.


2.2   Data preparation
To train and evaluate the LCM, we need to convert raw text datasets into a sequence of SONAR
embeddings, each one corresponding to a sentence. Dealing with large text corpora presents several
practical limitations. First, the precise segmentation of a text into sentences can be challenging due
to the presence of errors, specific formatting issues or any other sources of noise. This requires us to
apply robust automatic text segmentation techniques. Second, some sentences (even well formed)
can be very long and complex, which might negatively impact the quality of the encoded SONAR
embeddings. This is particularly prevalent for texts in the scientific domain. In the following, we
discuss strategies for sentence segmentation and how they affect the SONAR encoding.

Sentence segmentation analysis We have identified two potential sentence segmentation tech-
niques; as we are exploring multilingual data, we focus on sentence segmenters with a large language
coverage:
   1. SpaCy segmenter (SpaCy) (Honnibal et al., 2020) is a well established multilingual NLP toolkit
      that provides a rule-based approach to sentence segmentation. SpaCy is thoroughly tested for
      high-resource languages.
   2. Segment any Text (SaT) (Minixhofer et al., 2023; Frohmann et al., 2024) offers a suite of
      models and adapters that predict sentence boundaries at the token level. SaT is designed
      to be resilient to perturbations, particularly avoiding the over-reliance on punctuation and
      capitalization. This is valuable in domains where these conventional markers are often missing.
      The quality of SaT’s segmentation is however dependent on the choice of an “appropriate” split
      probability threshold.
We additionally customize both methods by incorporating a maximum sentence length cap in
characters. We refer to these extensions by SpaCy Capped and SaT Capped. Long sentences
are broken down into smaller, logically coherent fragments using a rule-based approach based on
punctuation marks for SpaCy. For SaT, we leverage the provided splitting probability estimates to
identify the next best potential split.
To measure the efficacy of a given segmenter, we evaluate the quality of the reconstructed sentences
with AutoBLEU. It is defined as a BLEU score (Papineni et al., 2002) comparing the decoded text
from a SONAR vector after encoding a segment, to the the reference segment. A good segmentation
will yield segments that can be encoded and then decoded without loss of signal, and thus score a
higher AutoBLEU.

                                                   6
For this analysis, we sample 10k documents from our pretraining datasets, representing approximately
500k sentences. The documents are processed with each segmenter, the sentences are encoded then
decoded and the AutoBLEU score is calculated. We stratified the results based on the lengths of
the original sentences.


                                  0.95                                                0.95




        Average Auto-BLEU Score
                                  0.90                                                0.90

                                  0.85                                                0.85

                                  0.80                                                0.80

                                  0.75                                                0.75

                                  0.70                                                0.70

                                  0.65       SAT                                      0.65        SAT Capped
                                             SpaCy                                                SpaCy Capped
                                  0.60                                                0.60
                                         0    100    200     300     400    500              0   25     50   75   100   125   150   175   200
                                              Sentence Size in Characters                             Sentence Size in Characters

  Figure 3 - Segmenters quality. Average Auto-BLEU scores for different sentence segmentation methods
      depending on sentence length, for both out of the box (left) and capped implementations (right).


As illustrated in Figure 3 and with a capping at 200 characters, the SaT Capped method demon-
strates a slight but consistent advantage over SpaCy Capped. Both out-of-the-box segmenters,
however, exhibit significant under-performance across all sentence lengths. This lower performance is
especially pronounced for sentences exceeding 250 characters, underscoring the limitations of using
the segmenters without capping.

Accordingly, we prepare the LCM training data with SaT Capped. We discuss in Appendix A
technical and engineering challenges faced when handling large amounts of SONAR embeddings.


2.3    Large Concept Model variants
The design of the LCM is driven by the need to conditionally generate a continuous sentence
embedding. This obviously contrasts with how current LLMs work, i.e., estimating a probability
distribution over a vocabulary of discrete tokens. A straightforward way of solving the task is to
train a transformer model to generate an embedding with the objective of minimizing the MSE loss
(see Section 2.3.1). However, a given context may have many plausible, yet semantically different,
continuations. The model should thus be able to learn a conditional probability distribution over the
continuous embedding of the next sentence.

There is a large body of work in computer vision aiming to learn such conditional probability
distributions over continuous data (Dhariwal and Nichol, 2021; Rombach et al., 2021). Models like
Dall-E 3 (Betker et al., 2023) or Imagen Video (Ho et al., 2022) use a diffusion process to generate an
image or video from a text prompt. Many different real images may satisfy the same input prompt,
hence the model has to learn a probability distribution over continuous pixel data. This motivates
the exploration of diffusion models for sentence embedding generation. Two variants are presented in
Sections 2.3.3 and 2.3.4. Another prevalent take on continuous data generation consists of quantizing
said data to ultimately model with discrete units; we explore LCM modeling with quantization in
Section 2.3.5.

                                                                                  7
                        x̂n



                 PostNet                                Linear                 Feature normalizer
             Transformer                          RdSONAR → Rdmodel
                                                                                    Linear
               Decoder                            Feature normalizer          RdSONAR → Rdmodel
                  PreNet
                                                         PreNet                           PostNet
                        xn       x̂n


Figure 4 - TheBase-LCM. Illustration of the Base-LCM. At its core is a standard decoder-only Transformer
                                       surrounded with a PreNet and a PostNet.


2.3.1   Base-LCM
Our baseline architecture for next-concept prediction is a standard decoder-only Transformer that
transduces a sequence of preceding concepts (read sentence embeddings) into a sequence of future
ones. As illustrated in Figure 4, the Base-LCM is equipped with a “PostNet” and a “PreNet”. The
PreNet normalizes the input SONAR embeddings and maps them to the model’s hidden dimension
dmodel .
                                                         t
                                PreNet(x) = normalize(x)Wpre + bpre ,                                (1)
                                                                   t
                                                                                                     (2)
                                                                                  
                                PostNet(x) = denormalize         xWpost + bpost       ,
                                                                                                     (3)

where Wpost ∈ RdSONAR ×dmodel , bpost ∈ RdSONAR , Wpre ∈ Rdmodel ×dSONAR and bpre ∈ Rdmodel .
In order to learn the maps “normalize” and its inverse “denormalize” we fit a robust scaler to a set
of randomly sampled SONAR vectors from different corpora and domains of text data. This scaler
removes the median statistics and scales the data according to the interquartile range (IQR).
                                                x−µ
                         normalize(x) =             ,     denormalize(x) = µ + σx.                   (4)
                                                 σ

The Base-LCM is trained on the semi-supervised task of next concept prediction, that is, the model
predicts the next concept x̂n and its parameters θ are optimized to regress the ground truth next
concept (xn ).

                               x̂n = f (x<n ; θ),    MSE(x̂n , xn ) = ∥x̂n − xn ∥2 .                 (5)

Given a data distribution q of documents (sequences of concepts), the training loss is evaluated as:
                                                         |x|
                                                        hX                           i
                              LBase-LCM (θ) = Ex∼q             MSE (f (x<n ; θ), xn ) .              (6)
                                                         n=1


In order to enable the generation of variable length documents at inference time, we suffix training
documents with the sentence “End of text.”. Similar to any sentence in the document, this special

                                                         8
                                                                 −→
suffix will be encoded with SONAR. This means that x|x| = eot := encode("End of text."). During
inference, we implement two main early stopping mechanisms: the first one measures the similarity
                                     −→
of the generated embedding x̂n to eot and stops if the cosine similarity exceeds a threshold seot . The
second mechanism compares the newly generated embedding x̂n to the previous generation x̂n−1 and
stops if their cosine similarity is higher than a threshold sprev . We set both seot and sprev to 0.9.

2.3.2   Diffusion-based LCM
Diffusion-based LCMs are generative latent variable models that learn a model distribution pθ
approximating a data distribution q. Similar to the Base-LCM, we model the diffusion LCMs as
auto-regressive models that generate concepts in a document, one at a time. The model distribution
is thus expressed at each position n of the sequence as pθ (xn |x<n ) i.e., the generation of the next
concept is conditioned on the preceding context.
In what follows we use a superscript for the denoising/diffusion step (t ∈ [0, 1]) and a subscript (n)
for indexing the sequence of concepts. We simplify for a given n the conditional model distribution
pθ (x0n |x0<n ) as pθ (x0 ), and the conditional data distribution q(x0n |x0<n ) as q(x0 ).
Diffusion models involve two processes: a forward noising process and a reverse denoising process (Ho
et al., 2020; Song et al., 2020):

Forward process and noise schedule The forward process is a Gaussian diffusion process
characterized by the marginal distribution q(xt |x0 ), given for every timestep t ∈ [0, 1] as:

                                                  q(xt |x0 ) := N (αt x0 , σt2 I).                                      (7)

With the reparameterization trick, we can sample from this marginal distribution via:

                                         xt = αt x0 + σt ϵ where ϵ ∼ N (0, I)                                           (8)

We use a variance-preserving forward process (Karras et al., 2022) for which we have:

        αt2 = sigmoid(λt ),             σt2 = sigmoid(−λt ) = 1 − sigmoid(λt ),                  λt = log αt2 /σt2 ,    (9)
                                                                                                                  


where λt is the log signal-to-noise ratio (log-SNR) for timestep t.
The noise schedule is a strictly monotonically decreasing function fλ that maps from the timestep
t ∈ [0, 1] to a log-SNR level: λt = fλ (t).
It is common in previous work to also define the noise schedule based on a discrete variance schedule
(β0 , . . . , βT ). This stems from the formulation of the forward process as a discrete-time Markov chain
that gradually adds Gaussian noise to the data according to said variance schedule:
                                       T
                                       Y
               q(x                           q(xt |xt−1 ),     q(xt |xt−1 ) := N (xt ;                                 (10)
                     1...T     0
                                                                                         p
                             |x ) :=                                                      1 − βt xt−1 , βt I),
                                       t=1

where to simplify the notation, xt is short for xt/T as the timesteps are now discretized.
From the variance schedule (βt )t , the noise schedule can be expressed as:
                                                               t
                                                               Y
                                                       αt2 =       (1 − βs ).                                          (11)
                                                               s=1


                                                                  9
Following Kingma and Gao (2024), for any given noise schedule, we visualize the distribution over
noise levels p(λ) = −dt/dλ in order to characterize how much time we are spending at every noise
level during training.
In this work, we consider three types of noise schedules:
Cosine. The schedule formulated in Nichol and Dhariwal (2021) as:
                                                           
                                                      t+s π
               αt = f (t)/f (0), where f (t) = cos
                2                                 2
                                                         .    , where s = 0.008.                                 (12)
                                                      1+s 2

Quadratic. The schedule introduced in Ho et al. (2020) where the variances (βt )t are set to constants
increasing quadratically from β0 to β1 .
                                                                             2
                                                        t p    p 
                                                                                                                 (13)
                                                   p
                                      βt/T =        β0 + .  β1 − β0                .
                                                        T

Sigmoid. We introduce in this work, the sigmoid schedule as a means to study the impact of the SNR
distribution on the training of our models. The schedule is parametrized by two hyper-parameters
(γ, δ) and is defined as:

                        αt2 = f (t)/f (0), where f (t) = sigmoid (δ − γ logit(t)) ,                              (14)

where “sigmoid” is the sigmoid function sigmoid x 7→ ex /(ex + 1) and “logit” its inverse function
logit : x 7→ log(x/(1 − x)). The hyper-parameter γ controls the scale of the log-SNR distribution
p(λ) and δ its center (see Figure 5).
In all our experiments, we follow Lin et al. (2024) and rescale the variance schedule (β1 , . . . βT ) to
enforce zero terminal SNR i.e., βT = 1.

Reverse process and objective function The joint distribution of the diffusion model pθ (x0...1 )
is called the reverse process and is defined as a Markov chain with learned Gaussian transitions
starting at p(x1 ) = N (0, I). In its discretized form:
                                T
                                Y
         pθ (x0:T ) := p(xT )         pθ (xt−1 |xt ),   pθ (xt−1 |xt ) := N (xt−1 ; µθ (xt , t), Σθ (xt , t)),   (15)
                                t=1

where µθ and Σ are predicted statistics. Σ is set to to the constant σt2 I (matching the transitions
of the forward process). µθ can be decomposed into a linear combination of xt−1 and a noise
approximation model ϵθ . This prediction method is dubbed ϵ-prediction (Ho et al., 2020; Nichol
and Dhariwal, 2021; Nichol et al., 2022). In this work we adopt x0 -prediction i.e., we predict the
noiseless state and optimize the simple reconstruction loss:
                                                              h                            i
                                                                                         2
            L(θ) := Et∼U (0,1) ω(t)L(t, θ) , L(t, θ) := Ex0 ,ϵ x0 − µθ (αt x0 + σt ϵ, t) 2 .    (16)
                                         


Different weighting strategies for the reconstruction loss were proposed in the literature (Ho et al.,
2020; Salimans and Ho, 2022; Hang et al., 2023). In this work, we default to the simple reconstruction
loss (ω(t) = 1, ∀t) and we experiment with a clamped-SNR weighting strategy:

                       ω(t) = max(min(exp(λt ), λmax ), λmin ), λt = log(αt2 /σt2 ),                             (17)


                                                         10
              1.0                                               7.5                                     0.175
                                                                                                        0.150                                      Cosine
                                                                5.0                                                                                Sigmoid(1.5, -2)
              0.8



                                               λt = log SNR t
                                                                2.5                                     0.125                                      Sigmoid(1.5, -1)
              0.6                                               0.0                                     0.100
         αt
              0.4                                               2.5
                                                                                                 p(λ)   0.075
                                                                5.0                                     0.050
              0.2
                                                                7.5                                     0.025
              0.0                                                                                       0.000
                    0.00 0.25 0.50 0.75 1.00                          0.00 0.25 0.50 0.75 1.00                  10        5       0       5
                               t                                                 t                                   λt = log SNR t
              1.0                                               15
                                                                10
                                                                                                        0.30                                       Cosine
              0.8                                                                                       0.25                                       Sigmoid(0.8, -1)


                                               λt = log SNR t
                                                                 5                                                                                 Sigmoid(3.5, 0)
              0.6                                                                                       0.20
                                                                 0
         αt
              0.4
                                                                                                 p(λ)   0.15
                                                                 5
                                                                                                        0.10
              0.2                                               10
                                                                                                        0.05
              0.0                                               15
                                                                                                        0.00
                    0.00 0.25 0.50 0.75 1.00                         0.00 0.25 0.50 0.75 1.00                         10          0       10
                               t                                                 t                                   λt = log SNR t
              1.0                                                7.5                                       0.20
                                                                                                                                                     Cosine
                                                                 5.0                                                                                 Quadaratic-1
              0.8                                                                                          0.15


                                               λt = log SNR t
                                                                 2.5                                                                                 Quadratic-2
              0.6                                                0.0
         αt
              0.4                                                2.5                                p(λ)   0.10
                                                                 5.0
              0.2                                                                                          0.05
                                                                 7.5
              0.0                                               10.0
                                                                                                           0.00
                    0.00 0.25 0.50 0.75 1.00                           0.00 0.25 0.50 0.75 1.00                      10       5       0        5
                               t                                                     t                                    λt = log SNR t

 Figure 5 - Noise schedules. Illustrations of the different noise schedules explored in this work. Our default
    schedule being cosine. Quadratic-1 is characterized by (β0 = 0.001, βT = 0.0012) and Quadratic-2 by
(β0 = 0.02, βT = 0.022) For each schedule we visualize the curve of (αt )t (see Equation (8)), the curve of the
         log-SNR and the associated distribution over noises levels p(λ) (Kingma and Gao, 2024).


which is a generalization of Salimans and Ho (2022)’s truncated-SNR weighting and Hang et al.
(2023)’s min-SNR strategy where the SNR is clamped between a min- and max-value λmin and λmax .
Additionally, we consider a weighting strategy that factors in the quality of the sample x0 . We use
as sample weight a scalar ω(x0 ) ∈ [0, 1] correlated with the sample’s fragility score i.e., how easy
it is to reconstruct a noised sample (see Section 2.5.2). Fragile samples will be assigned a smaller
weight and thus contribute less to the objective function.
                                                      h                                   i
                                                                                        2
                    Lfragility (θ) := Et∼U (0,1),x0 ,ϵ ω(x0 ) x0 − µθ (αt x0 + σt ϵ, t) 2 ,      (18)
                                    ω(x0 ) = sigmoid(a fragility(x0 ) + b),                                                                                           (19)
where a < 0 and b are hyper-parameters to tune.

Classifier-free diffusion guidance for the LCM Classifier-free diffusion guidance (Ho and
Salimans, 2022) consists of jointly training a conditional and an unconditional diffusion model. The
resulting conditional and unconditional score estimates are combined at inference time to achieve a
trade-off between sample quality and diversity. This combined score is defined as follows:
                                   ∇x logγ p(x|y) = (1 − γ)∇x log p(x) + γ∇x log p(x|y),                                                                              (20)


                                                                                         11
where y is the conditioning variable, in our case the sequence of preceding embeddings (x1 , . . . xn−1 )
when denoising xn .
The hyper-parameter γ controls the contribution of the conditional score; For γ = 0, this is equivalent
to an unconditional model, and for γ = 1, it is a fully conditional model. In practice for vision
models, γ is set to a value greater than 1, thus amplifying the signal from the conditioning model.

Inference At inference time, the reverse process is applied. xT is obtained by sampling a random
noise from p(xT ) = N (0, I), and is then iteratively denoised by taking steps in the direction of the
score function (i.e., the direction in which the log-likelihood increases fastest). Additional noise is
added during the process in order to avoid falling down into modes of the distribution.
Practically, we start from xT ∼ N (0, σinit  2 I) and find that the quality of the sampled output is

sensitive to the initial noise scale σinit .
Although we train the model on a large number of discretized timesteps, e.g., T=100, we only generate
with a smaller number of steps, e.g., S=40, at inference via accelerated generation processes (Song
et al., 2020). We select the sample steps following the trailing method of Lu et al. (2022) as it is
found to be more efficient for smaller steps S (Lin et al., 2024). That is we generate along the sampled
steps (τ1 , . . . τS ) = round(flip(arange(T, 0, −T/S))). During inference, we employ the classifier-free
guidance rescaling technique of Lin et al. (2024) proven to alleviate the image over-exposure problem
encountered in image synthesis diffusion models as the terminal SNR approaches zero. We denote
with gscale and grescale the guidance scale and guidance rescale factors used at inference.
Following Ning et al. (2023), we perform Epsilon-scaling at inference time as it is shown to alleviate
the exposure bias problem in diffusion models. In its simplified version, this is a training-free method
that consists of scaling down the over-predicted magnitude of error by a scalar λeps .
We describe in Section 2.3.3 and Section 2.3.4 two variants of diffusion LCM: One-Tower and
Two-Tower.

                      x̂ 0
                                                                      x̂ 0


            PostNet                                                 PostNet
                               ×(s)
        Transformer                                             Feed forward


                                                                                  modulator
                             denoising                                                              ×(s)
                                               PostNet
          Decoder              steps                           Cross-attention                t   denoising
            PreNet
                                           Transformer                                              steps
                                                               Self-attention⋆
                                             Decoder          ×Ld

        0    0   0     t                       PreNet               PreNet

                      xnt                                             xnt



Figure 6 - Inference with diffusion-based LCMs. In the left-hand side, an illustration of the One-Tower LCM
                     and on the right-hand side an illustration of the Two-Tower LCM.


2.3.3   One-Tower Diffusion LCM
This model, depicted in the left panel of Figure 6, consists of a single transformer backbone whose
task is to predict the clean next sentence embedding x0n given a noisy input xtn , conditioned on
previous clean sentence embeddings x0<n . During training, self-attention can be dropped with a

                                                    12
                                          x̂00          x̂10            x̂20



                                                       PostNet
                                                 Transformer
                                                   Decoder
                                                       PreNet


                                          t0     0      t1        0     t2

                                           t
                                          x00    x00     t
                                                        x11       x10    t
                                                                        x22



 Figure 7 - Training of One-Tower diffusion LCM. Interleaving the clean and noisy embeddings and sampling
                         different diffusion timesteps allows for efficient training.


certain probability for unconditional training. This enables classifier-free guidance at inference time
(see Section 2.3.2 for details).
Each input embedding is concatenated with the corresponding diffusion timestep embedding. The
learned position embeddings are added to the input vectors prior to being fed to LCM. The backbone
utilizes a causal multi-head self-attention.
For efficient training, the model is trained to predict each and every sentence in a document at once.
As depicted in Figure 7, during the diffusion process, the model attends to the clean sentences in
the context using causal multi-head attention layers. The input is specially prepared by interleaving
the noisy (blue) and clean (light blue) sentence embeddings, and the attention mask is prepared
accordingly to only attend to the clean sentence embeddings (gray arrows).

2.3.4   Two-Tower Diffusion LCM
This model, depicted in the right panel of Figure 6, separates the encoding of the preceding context
from the diffusion of the next embedding. A first model, labeled contextualizer, takes as input the
context vectors x<n and encodes them causally i.e., we apply a decoder-only Transformer with causal
self-attention. The outputs of the contextualizer are then fed to a second model dubbed denoiser,
which predicts the clean next sentence embedding x0n by iteratively denoising the latent x1n ∼ N (0, I).
The denoiser consists of a stack of Transformer blocks with cross-attention block to attend over
the encoded context. Both the denoiser and the contextualizer share the same Transformer hidden
dimension dmodel . Each block of each Transformer layer in the denoiser (including the cross-attention
layer) is modulated with adaptive layer norm (AdaLN, Perez et al. (2018); Peebles and Xie (2023)).
The AdaLN modulator of Two-Tower regresses channel-wise scale (γ), shift (β) and residual gates
(α) from the embedding of the current diffusion timestep t.



                               [β, γ, α] = SiLU(embed(t))Wt + b,                                      (21)
                                       y = x + α Block((1 + γ) x + β),                                (22)


Following Peebles and Xie (2023) and Goyal (2017) we initialize each residual block in a Transformer
layer (“Block”) with the identity function via initializing W and b in Equation (21) to zero. The

                                                             13
                                         x̂n0     x̂n1   x̂n2


                s0     s1     s2                PostNet                                      h4   1   1    1    1    1    0
                                     Feed forward                                                 1   0    0    0    0    0


                                                                modulator
                                                                                             h3
                 PostNet
                                    Cross-attention                         t0 , t 1 , t 2   h2   1   1    1    0    0    0
              Transformer                                                                    h1   1   1    0    0    0    0
                                    Self-attention⋆
                Decoder            ×Ld
                                                                                             h0   1   0    0    0    0    0
                     PreNet                     PreNet
                                                                                                  0   s0   s1   s2   s3   s4

                x0     x1     x2          t       t       t
                                         x00     x11     x22



Figure 8 - Training Two-Tower diffusion LCM. On the left panel, a Two-Tower forward pass in training time
in order to denoise multiple embeddings in parallel. On the right side panel a visualization of the denoiser’s
    cross-attention masks with the red highlighted row signaling a sample dropped to train the denoiser
unconditionally. (h1 , . . . , h4 ) denotes the sequence of intermediate representations in the denoiser right before
                                                 the cross-attention layer.


diffusion timestep t is embedded using a 256-dimensional frequency embedding (Dhariwal and Nichol,
2021; Peebles and Xie, 2023) followed by a two-layer MLP with SiLU as activation function. “embed”
maps to the denoiser’s hidden dimension dmodel . The self-attention layers in the denoiser do only
attend to the current position i.e., we do not attend to the preceding noised context. The self-
attention layers were kept for consistency with a standard Transformer block and for the possible
extension of denoising multiple vectors at once.

Two-Tower training. At training time, Two-Tower’s parameters are optimized for the
next-sentence prediction task on unsupervised sequences of embeddings. The causal embeddings
from the contextualizer are shifted by one position in the denoiser and a causal mask is used in its
cross-attention layers. A zero vector is prepended to the context vectors to enable the prediction
of the first position in the sequence (see Figure 8). To train the model both conditionally and
unconditionally in preparation for inference with classifier-free guidance scaling, we drop random
rows from the cross-attention mask with a rate of pcfg and denoise the corresponding positions with
only the zero vector as context.

2.3.5   Quantized LCM
Two major approaches currently stand to deal with continuous data generation in the image or speech
generation fields: one is diffusion modeling, the other is learning quantization of the data before
modeling on top of these discrete units.
In addition, the text modality remains discrete, and despite dealing with continuous representations
in the SONAR space, all possible text sentences (of less than a given number of characters) are a
cloud of points rather than a real continuous distribution in the SONAR space. These considerations
motivate the exploration of quantization of SONAR representations and then modeling on these
discrete units to address the next sentence prediction task. Finally, following such an approach
enables the natural use of temperature, top-p or top-k sampling, to control the level of randomness
and diversity in the sampling of the next sentence representation.
In this section, we learn residual quantizers for the SONAR space, and then build a Quantized
Large Concept Model based on these discrete units. We tried to come up with an architecture
as close as the diffusion LCM models, to be able to compare approaches.

                                                                   14
Quantization of SONAR space. We use Residual Vector Quantization (RVQ; Zeghidour et al.
(2021)) as a coarse-to-fine quantization technique to discretize SONAR representations. Vector
quantization maps continuous input embeddings to the nearest entry in a learnt codebook. RVQ
iteratively quantize residual errors from previous quantizations using additional codebook for each
iteration. We use FAISS implementation (Douze et al., 2024) which performs iterative k-means
clustering of residuals. We use the Improved Residual Vector Quantization (IRVQ) method from Liu
et al. (2015), with a beam size of 1 for memory efficiency. We trained the RVQ codebooks on 15
million English sentences extracted from Common Crawl using ncodebooks = 64 number of quantizers
with nunits-per-codebook = 8192 units per codebook.
One property of RVQ is that the cumulative sum of centroid embeddings of the first codebooks are
an intermediate coarse approximation of input SONAR vectors. In that way, we can report the
evolution of auto-encoding BLEU scores with the increasing number of codebooks used to quantize
SONAR embeddings, before using the SONAR text decoder to decode quantized embeddings. We
notice in Figure 9 that auto-encoding BLEU consistently improves as the number of codebooks
increases , reaching around 70% of the auto-encoding BLEU score achieved with continuous SONAR
embeddings, when using all 64 codebooks.



                                                        80




                             Auto-Encoding BLEU score
                                                        60


                                                        40


                                                        20                                   Base decoder
                                                                                             Finetuned decoder
                                                                                             Unquantized topline
                                                         0
                                                             10   20      30         40        50        60
                                                                       Number of codebooks

Figure 9 - Auto-encoding BLEU scores on FLORES devtest set, encoding sentences with SONAR encoder,
    quantizing with a varying number of codebooks, dequantizing and decoding with SONAR decoder.



Finetuning the SONAR decoder on quantized representations. We fine-tuned SONAR
decoder on quantized representations to adjust it for the space created by the quantizers on 1.2M
English sentences. To make the decoder more robust against
                                                        residual representations
                                                                                 from intermediate
codebooks, we randomly select a codebook number k ∈ 23 · ncodebooks , ncodebooks during fine-tuning,
with probability p = 0.3, and use the quantized representation with codebooks up to k. Figure 9
shows the improvement in auto-encoding performance when the decoder is adapted to quantized
representations.

Quant-LCM architecture. In the same spirit of diffusion LCM, we aim at coarse-to-fine
generation of SONAR embeddings conditioned on left-context sentences. However, we do not follow
a denoising task as in diffusion modeling, but an iterative generation of SONAR embeddings based on
intermediate quantized representations instead. In order to generate a SONAR embedding conditioned
on left-context sentences, the Quant-LCM model starts with the intermediate representation as
a vector filled with zeros. We iteratively add to this intermediate representation the predicted
residual centroid embeddings. In that way, the predicted SONAR embeddings are iteratively refined
based on the growing cumulative sum of centroid embeddings of first codebooks, until all codebooks

                                                                            15
have been seen. We used the One-Tower architecture for Quant-LCM experiments even though
it could be trained with Two-Tower architecture too. Compared to the diffusion LCM, noisy
input representations are replaced with intermediate quantized representations and diffusion timestep
embeddings as input are replaced by codebook index embeddings.


Discrete targets. Following previous work on modeling discrete units from residual quantizers
(Wang et al., 2023; Rubenstein et al., 2023; Lee et al., 2022), a Quant-LCM can be trained to
predict the unit from the next codebook, parameterized with a softmax output layer. For parameter
efficiency, we do not use ncodebooks · nunits-per-codebook unique indices as discrete targets which would
imply ncodebooks · nunits-per-codebook output dimensions, but only nunits-per-codebook output dimensions
while inputting the information of the codebook index to the model. At training time, similarly
to diffusion LCM training, we randomly sample codebook index k between 1 and ncodebooks , and
compute the cumulative sum of centroid embeddings of the first k−1 codebooks as input. We use
the unit from codebook k of the target embedding as target index for cross entropy loss computation.
At inference time, we iteratively predict the unit from the next codebook, get the corresponding
centroid embedding and add it to the current intermediate representation as additional predicted
residual embedding. Finally, we also enable classifier-free guidance on logits at inference time (Gafni
et al., 2022) by randomly dropping left-context conditioning during training as previously described
in Section 2.3.3. This modeling approach with discrete targets is dubbed Quant-LCM-d in the
following sections. The improved SONAR decoder for quantized representations is used to bridge
the compression gap coming from SONAR quantization in following ablation studies when using
Quant-LCM-d.


Continuous targets. We also explored a modeling approach that predicts continuous target
SONAR vectors based on left-context sentences and intermediate quantized representation of the
target vector, minimizing the Mean Squared Error between prediction and target embeddings. At
inference time, we can either iteratively add the closest centroid embedding based on the predicted
residual r̂ or sample a centroid ci from the following distribution:

                                                   e−β·∥ci −r̂∥2
                                      p(ci |r̂) = P −β·∥c −r̂∥ ,                                    (23)
                                                   ke
                                                            k    2



where β is a temperature hyper-parameter. This modeling approach with continuous targets is
denoted with Quant-LCM-c in the following sections.


2.4     Ablations
In this section, we delineate the ablations experiments conducted to evaluate the aforementioned LCM
designs. We compare all the variants of LCMs introduced above, namely, Base-LCM, One-Tower,
Two-Tower and Quant-LCM.


2.4.1   Experimental setup

For our ablation study and for the sake of reproducibility, we pre-train our models on the Fineweb-
edu dataset (Lozhkov et al., 2024). All models are configured to have approximately 1.6B trainable
parameters and are pre-trained on Meta’s Research Super Cluster (RSC, Lee and Sengupta (2022))
for 250k optimization steps spanning 32 A100 GPUs with a total batch size of 229k concepts.

                                                   16
Models architectures. The Base-LCM has 32 layers and a model dimension dmodel = 2048
with 16 attention heads. It uses rotary position embeddings (RoPE, Su et al. (2024)), applies
pre-normalization using RMSNorm (Zhang and Sennrich, 2019), uses the SwiGLU activation func-
tion (Shazeer, 2020) and is trained with a dropout rate of p=0.1.
The One-Tower diffusion LCM is made of 32 transformer blocks, each made of a self-attention
layer with 32 attention heads and followed by a feed-forward neural network with inner size 8192.
It has a dimension dmodel of 2048 and uses learned position embeddings. The noise scheduler is set
with T=100 diffusion timesteps. During training, self-attention is dropped with a probability of 0.15
for unconditional training, enabling classifier-free guidance at inference time.
The Two-Tower diffusion LCM has 5 layers in its contextualizer and 13 layers in its denoiser.
Similar to the Base-LCM, it has 16 attention heads, a model dimension dmodel = 2048, and uses
SwiGLU activations and RMSNorm in both contextualizer and denoiser. The contextualizer uses
RoPE for embedding positions whereas the denoiser is without positional embeddings. We use by
default the cosine noise schedule with T=100 and train with a dropout rate of p=0.1. For training
the model unconditionally we use a cross-attention mask dropout of rate 0.15 (see Section 2.3.4).
The pre-training documents are wrapped at 128 sentences. Unless otherwise mentioned we decode
with S=40 sample steps with a guidance scale gscale = 3, a guidance rescaling factor of grescale = 0.7,
an initial noise scale σinit = 0.6 and epsilon-scaling with λeps = 1.00045.
The Quant-LCM follows exactly the same architecture as the One-Tower diffusion LCM, except
for Quant-LCM-d which differs only by its output dimension which is set to nunits-per-codebook = 8192
for softmax computation. For single sentence prediction tasks, we use topk = 1 and gscale = 2 for
Quant-LCM-d and topk = 1 with gscale = 3 for Quant-LCM-c, while for multi-sentence generation
tasks we used temperature of 1, topk = 3, gscale = 1, for Quant-LCM-d and temperature of 0.005,
topk = 5, gscale = 1.5 for Quant-LCM-c, as higher guidance or lower temperature setups led to
repeated generated sentences.

                                               #Llama2 Tokens              #Sentences          Total
        Dataset                   #Docs         Q1         Q2      Q3    Q1     Q2     Q3    sentences

        ROC-stories (dev)            2000       48       57        64      5     5      5         10K
        ROC-stories (test)           1871       50       56        62      5     5      5        9.4K
        C4 (dev)                     1000      136      288       577      6    12     24       20.6K
        C4 (test)                    1000      133      282       599      6    11     25       21.9K
        Wikipedia-en (dev)           1000      146      332       736      5    10     23       21.1K
        Wikipedia-en (test)          1000      147      312       673      5     9     21       19.1K
        Gutenberg (dev)                55    10297    15934     22259    328   530    687      216.2K
        Gutenberg (test)               61    10752    15204     23414    325   457    735      562.2K

Table 2 - Statistics of the pre-training evaluation datasets. For each subset we report the number of documents,
   the total number of sentences and document lengths quartiles in sentences and in Llama2 tokens for
                                               reference.


Pre-training evaluation. Pre-trained token-level language models are typically evaluated with
perplexity: a measure of how well each next token is predicted given a teacher-forced (i.e., ground
truth) prefix of the document. In a similar spirit, we evaluate pre-trained LCMs in a teacher-forced
mode. But as they cannot produce the probability explicitly, we resort to a custom set of metrics of
the quality of next sentence prediction.

                                                      17
Each pre-trained model is initially evaluated on the quality of its predicted next sentence x̂n given
a ground truth context x<n . Practically, for a given document x1:N , we run the LCM inference in
teacher-forcing mode and evaluate the following metrics:
• L2 distance (ℓ2 ). Euclidean distance in the SONAR space between the predicted embedding x̂n
  and the ground truth continuation xn : ℓ2 := ∥x̂n − xn ∥2 .
• Round-trip L2 distance (ℓ2-r ). Euclidean distance in the SONAR space between the re-encoded
  sentence generated from the predicted embedding and the ground truth continuation xn ,

                                   ℓ2-r := ∥ encode(decode(x̂n )) − xn ∥2 .

  Since an LCM can predict an embedding outside of the distribution of real embeddings (obtained
  by encoding natural sentences), the SONAR decoder might shift these embeddings to the nearest
  plausible embeddings subspace. The ℓ2-r metric is introduced to capture the shift in embeddings
  after decoding them into text then re-embedding them again in the SONAR space. The more the
  generated embeddings are out-of-distribution, the higher the delta between ℓ2-r and ℓ2 would be.
• Contrastive accuracy (CA). The ratio of embeddings in a batch that are further away (in terms of
  ℓ2 ) from the predicted embedding x̂n than the ground truth xn (for each n, we exclude xn and its
  two neighboring ground truth embeddings from the comparison). This metric naturally assigns
  higher penalty for large ℓ2 values in the regions with high density of sentence embeddings.
• Paraphrasing (PAR). The maximum cosine similarity (CS) between the generated embedding x̂n
  and the context embeddings x<n , normalized by the score of the ground truth sentence. Thus,
  PAR = max CS(x̂n , xm )/ max CS(xn , xm ). The goal of this metric is to capture if the model is
           m<n               m<n
  simply copying or paraphrasing a sentence from the context more (> 1) or less (< 1) than the
  reference sentence.
• Mutual information (MI). This metric of text coherence evaluates the mutual information between
  the next predicted sentence ŝn = decode(x̂n ) and the previous k = 10 ground truth sentences by
  computing the difference between the unconditional perplexity of ŝn and its perplexity conditioned
  on the prompt:
                                    1
                           MI =          (log pLM (ŝn ) − log pLM (ŝn |sn−k:n−1 )) .
                                  |ŝn |
  We estimate the perplexity with a small language model, GPT-2 (Radford et al., 2019). We
  prepend a newline symbol to ŝn , so that a probability could be assigned to its first token, and we
  compute the average mutual information per-token by normalizing it with |ŝn |, the length of ŝn in
  tokens. When averaging MI over a dataset, |ŝn | are used as weights.

Pre-training evaluation data. The pre-training evaluation is performed on sampled subsets from
four corpora covering different domains: ROC-stories (Mostafazadeh et al., 2016), C4 (Raffel et al.,
2019), Wikipedia-en (English Wikipedia dump) and Gutenberg. We sample two distinct subsets
(dev and test) from each corpus, we use the dev split for tuning inference hyper-parameters and
report the results on the test splits. The statistics of the evaluation corpora are presented in Table 2.
The results of the pre-training evaluation are presented in Table 3.
First, diffusion-based LCM and Quant-LCM variants have similar ℓ2 and ℓ2-r scores despite an
important difference in their learning objectives. The only model that shows substantially lower ℓ2
score is the Base-LCM. This is expected since Base-LCM effectively optimizes ℓ2 score during
training. Yet, ℓ2-r score is not improved compared to other models. This could be explained by the

                                                    18
                                      ROC-stories                                     C4
 Model
                      ℓ2       ℓ2-r      PAR     CA        MI      ℓ2      ℓ2-r     PAR     CA        MI
 Base-LCM             0.177    0.237     1.847   72.4%     0.062   0.204   0.261    1.964   69.1%     -0.105
 One-Tower            0.236    0.236     1.939   80.2%     0.977   0.279   0.273    2.239   77.1%     1.110
 Two-Tower            0.233    0.231     2.088   80.6%     1.137   0.265   0.261    2.265   75.4%     1.134
 Quant-LCM-c          0.236    0.237     1.683   76.0%     0.610   0.279   0.283    2.013   77.2%     0.715
 Quant-LCM-d          0.240    0.246     1.871   81.1%     0.682   0.270   0.282    1.808   75.0%     0.359
                                      Wikipedia-en                                Gutenberg
 Model
                      ℓ2       ℓ2-r      PAR     CA        MI      ℓ2      ℓ2-r     PAR     CA        MI
 Base-LCM             0.229    0.283     1.770   69.6%     0.071   0.207   0.264    1.780   67.8%     -0.184
 One-Tower            0.324    0.311     2.087   80.9%     1.202   0.284   0.281    2.051   75.1%     0.725
 Two-Tower            0.307    0.297     2.079   78.8%     1.307   0.267   0.267    2.077   73.0%     0.684
 Quant-LCM-c          0.306    0.317     1.842   79.5%     0.744   0.269   0.281    1.774   72.1%     0.419
 Quant-LCM-d          0.295    0.311     1.592   76.0%     0.323   0.276   0.290    1.599   72.0%     0.153

Table 3 - Comparing architectures. Pre-training evaluation results on the four select corpora. For each subset,
we report ℓ2 (L2 distance in SONAR space), ℓ2-r (round-trip L2 distance after decoding and re-encoding the
     generated embeddings), PAR (similarity to preceding embeddings) and CA (contrastive accuracy)


fact that when many plausible next sentence continuations are possible, Base-LCM generates their
average in SONAR space (instead of sampling one of plausible modes) which may not correspond to
any relevant point in SONAR space. This hypothesis is also highlighted by the poor Base-LCM
performance in term of CA and MI scores.
We do not notice any significant difference in CA scores between diffusion LCMs and Quant-LCM
variants. MI scores, on the contrary, are consistently higher for diffusion-based models compared
to Quant-LCM. At the same time, diffusion LCMs tend to paraphrase more the context in the
generated embeddings, which also correlates with an increased MI score. Still, Quant-LCM variants
significantly outperform Base-LCM on MI metric. Now comparing the different variants, Quant-
LCM-c outperforms Quant-LCM-d modeling variant: one hypothesis is that predicting codebook
indices with cross-entropy loss is harder than MSE objective where Quant-LCM-c can more easily
learn combination of left-context vectors for next sentence embedding.
For diffusion LCMs, we don’t observe any consistent difference between One-Tower and Two-
Tower when looking across all metrics and datasets. Note that overall, to tackle the next sentence
prediction task in the SONAR space, diffusion-based methods give clearly better results compared to
all other models.

Instruction-tuning evaluation. Subsequently, the pre-trained models are instruction-tuned on
the stories subset of Cosmopedia (Ben Allal et al., 2024) and are evaluated on a held-out subset
of Cosmopedia itself. We aim with this finetuning to evaluate the ability of the models to follow
instructions and generate consistent stories.
For the sake of comparison, we trained a small Llama (Touvron et al., 2023) on the same training
data (Fineweb-edu) and finetuned it on Cosmopedia. This model has 24 transformer layers, each
with 16 attention heads and a model dimension of 2048 for a total of 1.4B parameters. This model
will be referred to as smaLlama.

                                                      19
We evaluate the following metrics:

• ROUGE-L (R-L). ROUGE-L (F-measure) (Lin, 2004) between the generated and reference stories.

• Coherence (Coherence). This reference-free metric is computed with a bidirectional transformer
  model fine-tuned by Jwalapuram et al. (2022) to assign higher scores to positive “natural” documents
  than to negative examples with permuted sentences. For reporting, we normalize it with a sigmoid
  (with a temperature 3.0, empirically set to make the scores of “certainly incoherent” documents
  close to 0 and those of “certainly coherent” documents close to 1).


                                Model               R-L ↑    Coherence ↑
                                Base-LCM             23.69           0.482
                                One-Tower            33.40           0.968
                                Two-Tower            33.64           0.938
                                Quant-LCM-c          30.87           0.847
                                Quant-LCM-d          28.01           0.704
                                smaLlama             34.88           0.984

   Table 4 - Comparing architectures. Instruction-tuning evaluation results. For each model we score the
            generated stories on the held-out test prompts and report R-L (ROUGE-L) scores.


The scores in terms of R-L and Coherence of the finetuning evaluation are presented in Table 4.
Those quantitative results are in line with the pretraining evaluation ones. Both R-L and Coherence
scores correlate with the model ordering based from MI scores, mainly Quant-LCM is outperformed
by diffusion-based models, and both outperform Base-LCM by a large margin.

We also note that smaLlama outperforms the LCMs on this downstream task on both metrics.
It is well known that LLMs produce very fluent outputs, that explains the higher Rouge-L score.
We also note that the One-Tower and Two-Tower produce coherent outputs, on par with the
smaLlama outputs.


2.4.2   Importance of the diffusion inference hyper-parameters

In this section we study the effect of different inference hyper-parameters on the quality of the
generated text. To this end, we generate outputs for the C4 test split with the Two-Tower LCM
model above, while varying the following hyper-parameters: the guidance scale gscale , the initial
noise scale σinit , and the number of inference sample steps S. We score the generations following the
same protocol above and report the results in Figure 10. We note that as we increase the guidance
scale, the mutual information between the prefix and the generated suffix increases, and so does
paraphrasing as we are paying more attention to the conditioning context variables. In the opposite
direction of the mutual information, the ℓ2 distance from the ground truth continuation increases as
the model prefers to stay close to the prefix. Regarding the initial noise scale, we observe that values
between 0.5 and 0.7 achieve the best MI score. In particular, the generated individual sentences
are usually longer with a higher σinit . The ℓ2 distance on the other hand does not reflect this trend.
Lastly, we increase the number of inference steps and measure the mutual information (MI) of the
generated texts. With more inference steps, we can improve the prefix-suffix mutual information, but
there is diminishing returns to increasing the inference cost with little qualitative improvement.

                                                    20
Mutual information (MI)
                                                             L2-distance (`2 )
                                                                                                                                  L2-distance (`2 )
                          1.4         MI                                                            MI                                                   1.2
                                                                                    1.5
                                      `2               0.30                                         `2                      0.30
                          1.2                                                                                                                            1.1
                          1.0                                                       1.0
                                                       0.28                                                                 0.28                         1.0
                          0.8


                                                          Mutual information (MI)
                                                                                    0.5                                                                  0.9


                                                                                                                               Mutual information (MI)
                          0.6                                                                                               0.26
                                                       0.26                                                                                                              λeps =1.0025
                          0.4                                                       0.0                                                                  0.8
                                                                                                                            0.24                                         λeps =1.00045
                          0.2                          0.24                                                                                              0.7             λeps =1
                                                                                    0.5
                                  2        4    6                                             0.4         0.6   0.8   1.0                                      20   40     60    80
                                Guidance scale (gs )                                      Initial noise scale (σinit )                                   Inference sample steps (S)
  Figure 10 - Importance of inference hyper-parameters. The first panel shows the quality of the generated
output measured with MI and ℓ2 as we vary the guidance scale gscale with fixed σinit = 0.6 and S = 40. The
 second panel varies the initial noise scale σinit with fixed guidance gscale = 3 and S = 40. The third panel
varies the inference steps S while holding the guidance scale gscale = 1.5 and σinit = 0.6 fixed. We consider 3
          values for λeps to see the impact of epsilon-scaling in the regime of large inference steps.



2.4.3                           Studying the noise schedules

In this section, we compare different Two-Tower diffusion LCMs trained with different noise
schedules, namely:

Cosine. Our default cosine noise schedule.

Quadratic. The first quadratic schedule (Quadratic-1) has β0 = 0.001 and βT = 0.0012, whereas
the second quadratic schedule (Quadratic-2) has β0 = 0.02 and βT = 0.022.

Sigmoid. Four different sigmoid schedules with with (α, β) ∈ {(1.5, −1), (1.5, −2), (0.8, −1), (3.5, 0)}.

All schedules are set with the default T=100. For the exact description of each noise schedule refer
to Section 2.3.2 and Figure 5. We selected the quadratic schedule as a commonly used schedule for
reference with two configurations, Quadratic-1 closer to the cosine schedule and Quadratic-2 with
more weight given to lower log-SNR. The selected sigmoid schedules with δ = 1.5 are configured
with γ = −1 and γ = −2, γ = −2 being slightly shifted to the left on the log-SNR distribution i.e.,
more weight to lower log-SNR regimes. We then change the δ parameter of the sigmoid schedule to
choose (δ = 0.8, γ = −1) for a peaked distribution of log-SNR around -1 and (δ = 3.5, γ = 0) for a
flat distribution over noise levels.

We follow the experimental setup described in Section 2.4.1 and report the results of the pre-training
evaluation in Table 5. Both quadratic schedules achieve a better MI score while the wide sigmoid
schedule (δ, γ) = (3.5, 0) achieves the highest accuracy CA on both C4 and Wikipedia-en. To
further understand the differences between those schedules we re-evaluate on C4 while varying the
guidance scale gscale . The results in Figure 11 confirm that the wide sigmoid schedule (δ = 3.5, γ = 0)
has a much higher contrastive accuracy across the board (rightmost panel) while closely matching
the cosine schedule in terms of mutual information (MI). This schedule being trained on a wider
spectrum of log-SNR learns to contrast more than to regress. Contrast this behavior to the peaked
sigmoid schedule (δ = 0.8, γ = −1) where the model focuses on regressing to the target (lower ℓ2 ),
akin to a Base-LCM, resulting in a lower contrastive accuracy.

                                                                                                         21
                                                                                     C4                                                                          Wikipedia-en
 Model
                                           ℓ2          ℓ2-r              PAR                CA             MI          ℓ2                                 ℓ2-r        PAR     CA        MI
 Cosine                                    0.265       0.261             2.265              75.4%          1.134       0.307                              0.297       2.079   78.8%     1.307
 Quadratic-1                               0.268       0.264             2.341              75.7%          1.252       0.309                              0.300       2.202   79.1%     1.409
 Quadratic-2                               0.270       0.265             2.320              76.2%          1.252       0.312                              0.303       2.185   79.7%     1.399
 Sigmoid(1.5, -1)                          0.257       0.259             2.226              74%            1.083       0.298                              0.292       2.110   77%       1.271
 Sigmoid(1.5, -2)                          0.277       0.267             2.291              77.2%          1.173       0.321                              0.303       2.179   80.3%     1.308
 Sigmoid(0.8, -1)                          0.252       0.255             2.053              70.6%          0.936       0.285                              0.283       1.883   71.7%     1.127
 Sigmoid(3.5, 0)                           0.307       0.265             2.347              80.3%          1.154       0.347                              0.303       2.187   83.7%     1.288

    Table 5 - Comparing noise schedules. Results of the pre-training evaluation on two corpora, C4 and
                                                                                            Wikipedia-en.


                               1.6




                                                                                                                            Contrastive accuracy (AC)
                                                                                     0.34




     Mutual information (MI)
                                                                                                                                                        0.80
                               1.4
                                                                                     0.32



                                                                 L2 distance (`2 )
                               1.2
                                                                                                                                                        0.75
                               1.0                                                   0.30
                               0.8                                                   0.28                                                               0.70
                                                  Cosine
                               0.6                Quadratic-1
                               0.4                Quadratic-2                        0.26                                                               0.65
                                                  Sigmoid(3.5)
                               0.2                Sigmoid(0.8)                       0.24
                                                                                                                                                        0.60
                                      2         4        6                                    2        4           6                                              2       4        6
                                     Guidance scale (gs )                                    Guidance scale (gs )                                                Guidance scale (gs )

   Figure 11 - Comparing noise schedules. The prefix-suffix mutual information (MI), the ℓ2 distance and
contrastive accuracy (CA) scores of evaluated C4 documents while varying the guidance scale (gscale ) under
                                            different schedules.


2.4.4                    Studying the loss weighting strategies
In this section we compare the baseline Two-Tower diffusion LCM trained with the simplified
objective (i.e., ω(t) = 1, ∀t)) to models trained with the clamped-SNR weighting strategy. We
consider two sets of (λmin , λmax ): (λmin , λmax ) = (0, 10) and (λmin , λmax ) = (0.001, 5). All models in
this section are trained with the cosine noise schedule.

Fragility as a sample weighing strategy As introduced in Equation (19), we train in this
section a Two-Tower LCM model with loss terms weighted by the sample’s fragility.

                                                              ω(x0 ) = sigmoid(a fragility(x0 ) + b)                                                                                     (24)

Given that estimating the fragility score of each sample as defined in Section 2.5.2 can be very
costly, we resort to training a simple MLP (3-layers) on 50M sampled sentences to approximate these
fragility scores. This model is referred to as F. Henceforth, the sample weight is:

                                                                         ω(x0 ) = sigmoid(a F(x) + b)                                                                                    (25)

The two hyper-parameters (a, b) are chosen so that extremely fragile sentences contribute less to
the loss (ω(x0 ) ≈ 0), and so that sample weights should increase smoothly as sample robustness

                                                                                                  22
improves. Figure 12 plots the sample weights distribution evaluated on a pre-training dataset with
a = −4 and b = 3.5.

                                2.5


                                2.0


                                1.5

                      Density
                                1.0


                                0.5


                                0.00.0           0.2             0.4             0.6           0.8              1.0
                                                                 Sample Weight

 Figure 12 - Resulting distribution of fragility sample weights ω(x0 ) with ω(x0 ) = sigmoid(−4 F(x) + 3.5).


We follow the experimental setup described in Section 2.4.1 and report the results of the pre-training
evaluation in Table 6. We observe that weighting with clamped SNRs does not improve the quality
of generated texts as measured with our pre-training evaluation metrics. When it comes to the
fragility-aware weighting strategy, we observe an improvement in the contrastive accuracy of the
model. In the remainder of this work we default to the simplified training objective (ω(t) = 1, ∀t).

                                                        C4                                                  Wikipedia-en
  Model
                         ℓ2              ℓ2-r          PAR      CA          MI         ℓ2            ℓ2-r     PAR     CA      MI
  Baseline ω(t) = 1      0.265           0.261         2.265    75.4%       1.134      0.307         0.297    2.079   78.8%   1.307
  SNR (0,10)             0.280           0.264         2.334    74.8%       1.107      0.320         0.296    2.102   77.9%   1.212
  SNR (0.001,5)          0.266           0.261         2.269    73.4%       1.094      0.304         0.291    2.007   76.6%   1.295
  Fragility              0.2815          0.273         2.306    76.5%       1.103      0.321         0.308    2.118   80.7%   1.193

  Table 6 - Comparing weighting strategies. Results of the pre-training evaluation on two corpora, C4 and
                                                               Wikipedia-en.



2.5     Analysis
2.5.1    Inference efficiency of LCMs
We compare in this section the inference computational cost of the Two-Tower LCM to that of a
vanilla LLM as a function of the total length in tokens of the prompt and output combined. We chose
the theoretical number of FLOPs, independent of any specific optimization. These optimizations are
generic to the transformer architecture and also apply to our LCM. We include in this comparison
two configurations of LCMs; the 1.6B used in the previous ablation studies and a 7B model we
scale up to in the following sections. For both LCMs, we estimate the inference cost with inference
sample steps S = 40. Given the quadratic complexity of the attention mechanism in transformers, the
complexity sharply increases with the context size (see upper right corner of Figure 13’s left panel).
The complexity of the LCM depends on how the context is sentencized: a context length of 200
tokens split into 10 sentences (20 tokens each) will incur a higher cost than the same 200 tokens split

                                                                       23
                         ×1015                                                            ×1014

                                 LLama2-7b                                          1.0           LLama2-7b
                  1.50           Two-tower LCM (7B)                                               Two-tower LCM (7B)
                                 Two-tower LCM (1.6B)                               0.8           Two-tower LCM (1.6B)
                  1.25




Inference Flops
                  1.00                                                              0.6

                  0.75
                                                                                    0.4
                  0.50
                                                                                    0.2
                  0.25

                  0.00                                                              0.0
                           0     10000     20000   30000   40000   50000   60000           0       500       1000    1500    2000       2500   3000
                                         Total context size in tokens                                    Total context size in tokens

  Figure 13 - Theoretical inference Flops of LCMs and LLLms. We evaluate the inference flops for different text
  lengths (in Llama2 tokens) with a variable average sentence length. Only extremely short sentences (≤ 10
                                               tokens) favor LLMs.


into 5 sentences (40 tokens each). We account for this by computing the cost on a range of sentence
lengths but report the total context size on the x-axis (context size = sentence length × number
of sentences). The LCM shows substantially better scalability with respect to increasing context
size. The inference computational cost of the LCM includes the three steps of (1) encoding into
SONAR, (2) LCM prediction in the sentence space then (3) decoding with a SONAR decoder. The
inference cost of LCMs varies significantly depending on the average length in tokens per sentence.
For extremely short sentences (less than 10 tokens), an LLM is more computationally efficient (see
lower left corner of Figure 13’s right panel).

2.5.2                     Fragility of SONAR space
When we perform modeling in a latent space, we primarily rely on the induced geometry (L2-distance).
However, the homogeneous Euclidean geometry of any latent representation will not perfectly match
the underlying text semantics. This is evidenced by the fact that a small perturbation in the
embedding space may result in a drastic loss of semantic information after decoding. We dub such
embeddings “fragile”. For this reason, we aim to quantify the fragility of semantic embeddings
(namely SONAR codes) to understand the quality of the LCM training data and how this fragility
can hinder the LCM training dynamics.
Given a text fragment w and its SONAR code x = encode(w), we define the fragility of w as:

                                            fragility(w) := −Eα∼U ([0,1]), ϵ∼N (0,I) [score(w, wα,ϵ )] ,                                       (26)
                                                                           √                            √ 
                                                     xα,ϵ = denormalize 1 − α normalize(x) + α ϵ ,                                             (27)
                                                     wα,ϵ = decode(xα,ϵ ),                                                                     (28)

where normalize and denormalize are the normalization and denormalization operators introduced
in Equation (4) with the goal of making x’s coordinates scale-independent. The “encode“ operation
maps text fragments into SONAR space, and the “decode“ operation produces a text fragment from
a given vector in the SONAR space. For each α in [0, 1], xα,ϵ is the perturbed version of x where a
noise vector of variance α is linearly combined with x. The perturbed vector is then decoded into a

                                                                               24
text fragment wα,ϵ . This perturbation is similar to the variance-preserving noising used in diffusion
LCMs (see Section 2.3.2).
The “score” operator in Equation (26) is set to be a semantic similarity metric comparing the
perturbed text wα,ϵ to the original w. We considered the following options:
• Auto-Encoding BLEU. score(w, wα,ϵ ) = BLEU(w, wα,ϵ ).
• External cosine similarity. Provided an external text encoder (read unrelated to SONAR) that
  encodes a text fragment w into encodeext (w) score(w, wα,ϵ ) = CS(encodeext (w), encodeext (wα,ϵ )),
  where CS is the cosine similarity measure. Compared to Auto-Encoding BLEU, this method is
  typically more robust to paraphrasing.

Finetuned robust decoder. To serve as a testbed for our fragility analysis, a new SONAR
decoder for English text is finetuned on a sample of our pre-training data. In order to improve the
decoder’s robustness to imperfectly generated embeddings from the LCM, we follow Equation (27)
and add random noise vectors to SONAR embeddings during training. As reported in Table 7, the
finetuned SONAR decoder exhibits stronger performance across a range of public corpora.

                                        Model                                                                     Flores              CNN DailyMail                                      Gutenberg        C4
                                        Base SONAR decoder                                                              79.5                          75.9                                  70.5          75.7
                                        Finetuned SONAR decoder                                                         88.0                          87.6                                  85.6          87.5

Table 7 - Comparing SONAR decoders. Raw reconstruction performance of our base SONAR decoder vs. the
 new decoder trained with noised embeddings. Scores are Auto-Encoding BLEU on random subsets of 10k
           sentences from each dataset, except for Flores where we use the entire dev split.


Fragility study. We sample 50M random text fragments, and for each sample we generate 9
perturbations corresponding to different noise levels α ∈ [0.1, 0.2, . . . , 0.9]. For the external cosine
similarity metric we use mGTE as external encoder (Zhang et al., 2024).

                             0.55                                              0.86
                                                                                                                                                                    1.0
                                                                                                                                                                                           BLEU: Base decoder
                                                                                                            0.8                                                                            BLEU: Finetuned decoder
                             0.50
                                                                               0.84                                                                                 0.9                    CosSim: Base decoder
                                                                                                                                                                                           CosSim: Finetuned decoder




  Auto-Encoding BLEU Score
                                                                                      Cosine Similarity
                             0.45
                                                                               0.82                         0.6




                                                                                                                                                                     Cosine Similarity
                                                                                                                                                                    0.8
                             0.40
                                                                               0.80




                                                                                 Auto-Encoding BLEU Score
                             0.35                                                                           0.4                                                     0.7
                                                                               0.78

                             0.30
                                                                               0.76                                                                                 0.6
                                                                                                            0.2
                             0.25

                                                                               0.74
                             0.20                                                                                                                                   0.5

                                                                               0.72                         0.0
                                    0      50     100     150    200     250                                      0.0      0.2        0.4       0.6     0.8   1.0
                                         Sentence Length in Characters                                                                      α

Figure 14 - Fragility scores. Auto-Encoding BLEU and external cosine similarity. In the left-hand panel as a
function of the text length (α-averaged) and in the right-hand panel as a function of the noise variance α.

We depict in the right panel of Figure 14 the curves of both score functions with respect to the noise
level α. We observe that BLEU scores decrease faster than the cosine similarity. Most importantly,
fragility scores are sensitive to the choice of the decoder. In particular, both Auto-Encoding BLEU

                                                                                                                                 25
                                                    Score
                               2.5
                                           BLEU: Base decoder
                                           BLEU: Finetuned decoder
                               2.0
                                           CosSim: Base decoder
                                           CosSim: Finetuned decoder


                     Density
                               1.5



                               1.0



                               0.5



                               0.0
                                     0.0          0.2         0.4             0.6   0.8   1.0
                                                                Score value

        Figure 15 - Auto-Encoding BLEU scores and cosine similarity (α-averaged) distributions.



and cosine similarity scores decrease at a markedly slower rate for the Finetuned decoder than for
the Base one as the amount of noise increases. We note also that the overall score distribution (after
averaging over all α), shown in Figure 15, exhibits a large spread of fragility scores across SONAR
samples.
One factor that can explain such a discrepancy is the text length. Compared to the Auto-Encoding
BLEU metric (which drops only by 1–2% for long sentences), fragility is more sensitive to the length
of sentences and drops faster for both similarity metrics. This shows that using a max sentence
length over 250 can be extremely challenging for SONAR and the LCM model. On the other hand,
even if short sentences are on average more robust, splitting a long sentence in the wrong place may
result in shorter but more fragile sub-sentences.
Taking a closer look at the 5% most fragile embeddings, we notice that they are very noisy. Typically,
they correspond to hyperlinks, references, unique ids, code-switched or numerical entries. These are
likely artifacts that the SONAR models were not exposed to during training or where the SONAR
tokenizer fails. Fragility can thus be used to filter out hard samples from the training data. We also
observe that short but complex technical phrases can be more fragile than common language phrases
of similar length.


3   Scaling the model to 7B
This section describes our effort to scale our model to 7B parameters and compare the performance
against other approaches such as token-based LLMs on more challenging tasks such as summarization
and summarization expansion (detailed in Section 3.1).
Based on the results in Section 2.4 where the two diffusion-based LCMs (One-Tower and Two-
Tower) outperform the other variants, we decided to scale a diffusion model to 7B parameters. We
chose to scale Two-Tower given its smaller memory footprint, particularly when processing long
contexts with a shallower contextualizer tower
The large 7B Two-Tower diffusion LCM has 5 layers in its contextualizer and 14 layers in its
denoiser. Its dimension has been extended to dmodel =4096. Each self-attention layer has 32 attention
heads. All other parameters are kept the same as for the 1.6B Two-Tower model. The model is

                                                                    26
pre-trained on a dataset of 2.3B documents, representing 2.7T tokens and 142.4B concepts/sentences.
We pre-trained this model on Meta’s RSC for 124k optimization steps spanning 256 A100 GPUs
with a total batch size of 1M concepts. We further extend the context length of this model to cover
2048 concepts instead of the 128 concepts in the ablation experiments. We trained using the AdamW
optimizer with (β1 , β2 ) = (0.9, 0.95), ϵ = 1e-5 and a weight decay of 0.1. We use a cosine learning
rate schedule, with warm-up of 10,000 steps up to LR = 3e-4. To improve training stability we clip
gradients at a maximum norm of g = 10. We subsequently finetune the 7B Two-Tower LCM on
publicly available instruction tuning datasets following Chung et al. (2024). Each sample consists of
a prompt and an answer and we back-propagate on answer sentences only. Each answer sequence
is suffixed with the phrase “End of response.” to teach the model when to stop generating. The
finetuning data totals 389M sentences, of which 53M are answers (i.e., targets). For supervised
finetuning, we use a cosine learning rate schedule with an initial rate of LR = 3e-5 and finetune the
model for 7 epochs with a batch size of 262K sentences (prompts and answers combined). We will
refer to the pre-trained model as Two-Tower-7B and the finetuned model as Two-Tower-7B-IT.

3.1     Evaluation Tasks and Data
This section describes the tasks on which we are evaluating and benchmarking our proposed model.
We detail datasets, baselines and metrics. For each task, the dataset was processed with the same
sentence splitter and SONAR encoder as used in the LCM training.

3.1.1    Metrics

As longform text generation is the main challenge for LCM, our benchmarking is mainly focused
on generative tasks, which are notoriously more difficult to evaluate automatically. Therefore,
we evaluate them with multiple automatic metrics, chosen to focus on complementary aspects on
generation quality. All metrics used in this section are summarized in Table 8.
For summarization and summary expansion (defined below), we report the traditional reference-based
Rouge-L metric (Lin, 2004). As summarization models have a tendency to copy content from the
source or from its own generated prefix, we report two additional word-based metrics. To evaluate
how much content is directly copied from the source, we report the proportion of word 3-grams of
the source that are present in the output (OVL-3). To evaluate repetitiveness of the generated texts,
we report the portion of duplicated word 4-grams in the output (REP-4).
To complement word-based metrics with summarization-focused neural evaluation, we use two metrics
introduced by Clark et al. (2023): average probabilities of the SEAHORSE classifiers for Q4 (whether
all the information in the summary is fully attributable to the source), denoted as SH-4 in the
following and Q5 (whether the summary captures the main ideas of the source), denoted as SH-5.
As a metric of the overall fluency of the generated sentences, we report an average probability that
the sentence is linguistically acceptable, as predicted by a classifier trained by Krishna et al. (2020)
on the CoLA dataset (Warstadt et al., 2019), further referred to as CoLA. To evaluate the local
coherence of the generated text, we report the average cosine similarity between each n’th and
n + 2’th sentence (Parola et al., 2023).

3.1.2    Summarization

Task and datasets. When considering a relatively long document, a summarization task can be
described as the act of generating a much shorter corresponding document that includes the essential

                                                  27
  Task            Area                Metric     Description                               Reference
  Summarization Target similarity     R-L        ROUGE-L                                   Lin (2004)
                Source similarity     OVL-3      N-grams overlap (N=3)
                Grammaticality        REP-4      Portion of duplicated N-grams (N=4)       Welleck et al. (2019)
                Fluency               CoLA       Sentence fluency classifier score         Krishna et al. (2020)
                Attribution           SH-4       Seahorse-Large-Q4 score                   Clark et al. (2023)
                Semantic coverage     SH-5       Seahorse-Large-Q5 coverage score          Clark et al. (2023)
  Summary         Grammaticality      REP-4      (see above)                               Welleck et al. (2019)
  Expansion       Fluency             CoLA       (see above)                               Krishna et al. (2020)


Table 8 - Summary of automatic metrics used in different tasks in Section 3.1. Order mostly follows paper’s
                                                   narrative.


information contained in the long document and the same logical structure linking the various pieces
of essential information.
Summarization techniques can range from more extractive to more abstractive. Extractive techniques
attempt to preserve in the summary the same vocabulary as that found in the long document,
thereby shortening the long by removing details and superfluous wording. Abstractive techniques, on
the other hand, attempt to produce the summary by rephrasing the essential pieces of information
found in the long document. Our work focuses more on abstractive summarization, as such type of
summarization cannot be performed without some form of understanding and reasoning.
We use the CNN DailyMail (Hermann et al., 2015) and XSum (Narayan et al., 2018) datasets. We
also report results on the challenging LCFO corpus which takes long documents as input, approx.
5k words (Costa-jussà et al., 2024). The task is to provide abstractive summaries with lengths
representing 20%, 10%, and 5% of the input document. Detailed statistics are provided in Table 9.

                                               #Llama2 Tokens                           #Sentences
    Dataset              #Docs            Q1                Q2          Q3         Q1        Q2         Q3
    CNN DailyMail          11.5k       605/61        892/78        1266/97      10/3        14/4       21/4
    XSum                   11.3k       273/25        445/30         735/35      25/1        30/1       35/1
    LCFO.5%                  249     6559/341      7214/378       7916/418    209/12      295/15     527/18
    LCFO.10%                 249     6559/654      7214/718       7916/796    209/22      295/27     527/32
    LCFO.20%                 249    6559/1276     7214/1403      7916/1524    209/41      295/48     527/59

  Table 9 - Statistics of the test split of evaluation benchmarks. For each subset we report the number of
documents and statistics of document and summary length in terms of sentences and Llama2 tokens. Each
                            table cell shows “document/summary” length quartiles.


Baselines. For CNN DailyMail and XSum, we compare against several baselines of different
architectures (encoder-decoder transformer, decoder-only LLMs) that are known to perform well
on summarization tasks. For encoder-decoder transformer models, we use T5 (Raffel et al., 2020).
For decoder-only LLMs, we choose Gemma-7B, Llama-3.1-8B and Mistral-7B-v0.3. We chose
the published instruction-tuned models to compare with the LCM with the same training regime,
and have a similar size (7B). Note that while T5 has much smaller sizes than the LCM, this is
compensated by using models that are fine-tuned explicitly on the target evaluation dataset.

Summarization results. Table 10 contains the results of different baselines and our LCM model for
summarization (CNN DailyMail and XSum). We can notice that the LCM produces competitive
Rouge-L scores when compared to a specifically tuned LLM (T5-3B) and even surpasses the

                                                       28
Model                 Paradigm                         CNN DailyMail
                                 R-L(↑)   OVL-3 (↑)   REP-4 (↓)   CoLA (↑)     SH-4 (↑)   SH-5 (↑)
Ground truth          —          100.00     0.170       0.684          0.850    0.683      0.586
T5-3B                 SFT         37.56     0.174       0.854          0.946    0.773      0.503
Gemma-7B-IT           IFT         31.14     0.245       1.032          0.963    0.740      0.560
Mistral-7B-v0.3-IT    IFT         36.06     0.200       0.780          0.972    0.780      0.676
Llama-3.1-8B-IT       IFT         34.97     0.248       0.928          0.973    0.763      0.692

Two-Tower-7B-IT       IFT         36.47     0.177       0.757          0.767    0.723      0.459

Model                 Paradigm                                  XSum
                                 R-L(↑)   OVL-3 (↑)   REP-4 (↓)   CoLA (↑)     SH-4 (↑)   SH-5 (↑)
Ground truth          —          100.00     0.108       0.399          0.987    0.352      0.418
T5-3B                 —           17.11     0.221       0.671          0.939    0.680      0.450
Gemma-7B-IT           IFT         18.20     0.177       0.620          0.769    0.546      0.446
Mistral-7B-v0.3-IT    IFT         21.22     0.162       0.480          0.922    0.633      0.621
Llama-3.1-8B-IT       IFT         20.35     0.186       0.501          0.941    0.687      0.658
Two-Tower-7B-IT       IFT         23.71     0.106       0.464          0.683    0.358      0.284

           Table 10 - Performance on the CNN DailyMail and XSum summarization tasks.


instruct-finetuned LLMs. Our model tends to generate more abstractive summaries rather than
extractive ones, as shown by the lower OVL-3 scores. The LCM produces fewer repetitions compared
to LLMs, and more importantly, the repetition rate is closer to the ground truth one. The LCM
generates globally less fluent summaries according to CoLA classifier. However, we can remark that
even the human generated ground truth gets a lower score compared to the LLM. A similar behavior
is observed for the source attribution (SH-4) and semantic coverage (SH-5). This may be explained
by model-based metrics that are more biased towards LLM generated content.

Long-context summarization results. Table 11 presents the results for long-context summariza-
tion (LCFO.5%, LCFO.10% and LCFO.20%). This is a challenging task for most of the models. For
example, Mistral-7B-v0.3-IT seems to be unable to follow the length instruction of the summary–it
always generates summaries which length is about 50% of the source. Mistral-7B-v0.3-IT also
has the highest SH-4 score, i.e., source attribution. The summaries generated by Gemma-7B-IT
tend to be longer than requested, while Llama-3.1-8B-IT generates summaries which length is the
closest to the requested size.
The LCM has only seen a limited amount of long documents in the pretraining and fine-tuning data.
Nevertheless, it performs well for this task. It outperforms Mistral-7B-v0.3-IT and Gemma-7B-IT
in the metric Rouge-L for the 5 and 10% conditions, and is close to Gemma-7B-IT for the 20%
condition. We also observe that the LCM yields high SH-5 scores for all conditions, i.e., the
summaries can be attributed to the source.
Finally, we observe that Llama-3.1-8B-IT performs substantially better than the other LLMs,
according to Rouge-L, while all LLMs have similar performance on the CNN DailyMail and XSum
summarization tasks. This could be explained by training data contamination for Llama-3.1-8B-IT,
or by the fact that the other two LLMs struggle to handle the long input context.



                                               29
Method                 WR                                    LCFO.5%
                                 R-L(↑)    OVL-3 (↑)   REP-4 (↓)    CoLA (↑)    SH-4 (↑)    SH-5 (↑)
Gemma-7B-IT            0.107       25.21     0.151        4.711        0.688      0.357      0.174
Mistral-7B-v0.3-IT     0.512       21.36     0.532        5.997        0.854      0.656      0.296
Llama-3.1-8B-IT        0.076      37.67      0.190        2.767       0.931       0.488      0.314
Two-Tower-7B-IT        0.060       26.88     0.162        2.473    0.796          0.628      0.196
                                                           LCFO.10%
                                 R-L(↑)    OVL-3 (↑)   REP-4 (↓) CoLA (↑)       SH-4 (↑)    SH-5 (↑)
Gemma-7B-IT            0.150       29.25     0.164        6.427        0.667      0.377      0.194
Mistral-7B-v0.3-IT     0.549       25.00     0.537        6.289        0.848      0.660      0.306
Llama-3.1-8B-IT        0.128      42.85      0.243        3.804       0.907       0.486      0.310
Two-Tower-7B-IT        0.089       29.38     0.202        3.00     0.791          0.623      0.183
                                                           LCFO.20%
                                 R-L(↑)    OVL-3 (↑)   REP-4 (↓) CoLA (↑)       SH-4 (↑)    SH-5 (↑)
Gemma-7B-IT            0.257       33.32     0.201        9.188        0.603      0.425      0.239
Mistral-7B-v0.3-IT     0.493       28.82     0.527        5.806        0.858      0.658      0.293
Llama-3.1-8B-IT        0.179      46.92      0.272        4.783       0.888       0.485      0.315
Two-Tower-7B-IT        0.140       31.74     0.253        3.664        0.779      0.613      0.187

  Table 11 - Performance on the long-context summarization task of LCFO. WR is the word count ratio
                         between the generated text and the source document.


4 Large Concept Model Extensions
In this section, we explore several extension of the Large Concept Model. First, we evaluate
the LCM on the new task of summary expansion, i.e., given a summary, create a longer text. We
then showcase the good zero-shot generalization performance of the LCM. Finally, we explore an
approach to add higher level information beyond sentences.

4.1 Summary Expansion
Task and datasets. When considering a short and concise document that has similar properties
to those of a summary (i.e., mainly a stand-alone document that abstracts from details), a summary
expansion task can be described as the act of generating a much longer document that preserves
the essential elements found in the corresponding short document, as well as the logical structure
that connects such elements. As this is a more freely generative task, an additional requirement
to be taken into consideration is that of coherence (for example, the detailed information included
in one generated sentence should not contradict that included in another sentence). The summary
expansion task presented here consists in taking summaries as inputs, from CNN DailyMail and
XSum, and generating a long document. Note that the goal is not to recreate the factual information
of the initial document rather than evaluating the capability of the model to extend the input text in
a meaningful and fluent way. We use similar baselines and metrics as in the previous section 3.1.2.

Results. Table 12 shows the results of the summary expansion for CNN DailyMail and XSum.
First of all, regarding the word count ratio, we can see different behaviours for the two corpora.
For CNN DailyMail, the models tend to generate texts that are 6 times larger than the input.
Llama-3.1-8B-IT produces even longer outputs (factor 8 instead of 6). But for XSum, while

                                                 30
                                               CNN DailyMail
               Method                   WR     R-L(↑)    OVL-3 (↑)   REP-4 (↓)      CoLA (↑)
               Gemma-7B-IT               6.8    35.54        0.801      2.104        0.951
               Mistral-7B-v0.3-IT        6.4    34.24        0.817      2.063        0.959
               Llama-3.1-8B-IT           8.5    37.76        0.822      2.582        0.844
               Two-Tower-7B-IT           6.3    30.85        0.726      2.911        0.474

                                                    XSum
               Method                   WR     R-L(↑)    OVL-3 (↑)   REP-4 (↓)      CoLA (↑)
               Gemma-7B-IT              19.5    17.89        0.963      10.238       0.116
               Mistral-7B-v0.3-IT        1.6    29.31        0.893      2.268        0.939
               Llama-3.1-8B-IT          19.8    28.84        0.915       2.543       0.898
               Two-Tower-7B-IT           7.1    23.82        0.561      1.542        0.603

Table 12 - Performance on the summary expansion tasks of CNN DailyMail and XSum, evaluated with the
metrics described in Table 8. WR is the word count ratio between the hypothesis and the source summary.


Gemma-7B-IT and Llama-3.1-8B-IT generate very long texts (almost 20 times longer than the
prompt), the LCM generates output of approximately the same length ratio as for CNN DailyMail.
Only Mistral-7B-v0.3-IT fails to generate long outputs for this corpus.

Then, we clearly see a different trend compared to the summarization task. The LLMs get higher
Rouge-L scores compared to the LCM. As mentioned above, the goal of this task is not to recreate
the original document. However, the R-L score tells us how much of the content of the full document
can be recreated. Contrary to the LLMs, our model tends to generate different sentences compared
to the original document from which the summary has been created (with an exception for Gemma-
7B-IT on XSum). This is expected since our model generates embeddings that are then processed
by a decoder trained on a translation task that tends to paraphrase the initial content. However, the
CoLA results show that this comes along with lower fluency, especially for CNN DailyMail.

4.2     Zero-shot generalization performance
SONAR is a semantic space that can represent 200 languages. In this paper, all experiments
presented so far have been done on English text. In this section, we explore the capability of our
proposed LCM approach to process other languages in a zero-shot fashion by leveraging SONAR’s
ability to represent multilingual data.

We use the XLSum (Hasan et al., 2021) corpus, a large scale multilingual abstractive news summa-
rization benchmark covering 45 languages. We score model outputs using the multilingual rouge
scoring scripts released with the benchmark.5 Note that Rouge-L scores are heavily dependent on
language-specific text tokenization and stemming. Unless provided in the aforementioned scripts, we
tokenize the model outputs and references with the default tokenization from Lin (2004). Languages
like Korean, Telugu and Tamil can benefit from a more appropriate stemming and tokenization.

We compare the LCM performance with Llama-3.1-8B-IT which officially supports eight languages:
English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai. According to The Llama3
team (2024), the model has seen many additional languages during pretraining, but was instruction
  5
      https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring


                                                        31
                30                      Two-Tower-7B-IT           Llama-3.1-8B-IT   Officially supported Llama3 language



XLSUM Rouge-L
                20

                10

                    0       Vietnamese
                                 English
                                  Pashto
                                 Swahili
                               Burmese
                                 French
                                   Hausa
                                    Urdu
                                   Hindi
                            Indonesian
                            Portuguese
                                  Arabic
                                Spanish
                                   Welsh
                                  Somali
                                 Turkish
                                Russian
                                 Persian
                               Japanese
                                     Igbo
                     Chinese simplified
                                 Kirundi
                                     Thai
                                Tigrinya
                        Serbian Cyrillic
                         Scottish Gaelic
                     Chinese traditional
                                  Yoruba
                              Ukrainian
                            Azerbaijani
                                Amharic
                                Marathi
                                    Tamil
                                 Sinhala
                                  Oromo
                                  Telugu
                                 Korean
                                 Punjabi
                                 Bengali
                                Gujarati
                                  Kyrgyz
                                  Nepali
                         Figure 16 - Rouge-L scores on XLSum for Llama-3.1-8B-IT and Two-Tower-7B-IT.



 finetuned on those eight languages only. The LCM, on the other hand, has never seen any language
 other than English, and we do not use the English XLSum training data either.
We report Rouge-L scores for 42 languages in Figure 16. Three languages were excluded since they
are currently not supported by SONAR: Pidgin, Serbian in Latin script and Uzbek in Cyrillic script.
The LCM substantially outperforms Llama-3.1-8B-IT on English (23.5 compared to 20.7 Rouge-L)
and on the average over all six languages officially supported by both models and included in XLSum
(20.2 versus 19.7 Rouge-L).6 We also observe that the LCM generalizes very well to many other
languages, in particular low-resource languages like Southern Pashto, Burmese, Hausa or Welsch
which all have Rouge-L scores greater than 20. Other well performing low-resource languages are
Somali, Igbo or Kirundi. Finally, the LCM obtains a Rouge-L score of 30.4 on Vietnamese. Overall,
these results highlight the impressive zero-shot generalization performance of the LCM to languages
it has never seen.

 4.3                    Exploring explicit planning
 When writing long-form text, it is often practical to first think about how to structure our narrative.
 Indeed Dijk (1977) states that all texts have an innate macrostructure i.e., a global discourse structure
 spanning the entirety of the text. In general scientific discourse, one such macrostructure is that of
 problem-solving (Heffernan and Teufel, 2022), where an author must first motivate and describe the
 problem, before proposing an appropriate solution.
 Composing such a macrostructure is no easy task. Yet in the realm of LLMs, there have been many
 recent efforts to help guide model generation using similar structures. One such popular approach
 is that of creating outlines, which provide a high-level overview of the desired narrative (Li et al.,
 2024). An alternative approach to outlines is creating summaries of future targets, which the model
 can then expand upon (Sun et al., 2022).
 Given the nature of the LCM operating at the concept level, it naturally creates long-form output.
 Therefore, it is important to ensure that the model is capable of creating coherent generations given
 the multitudes of possibilities for next concept prediction. In order to address this, we envision an
                6
                    English, French, Hindi, Portuguese, Spanish and Thai.


                                                                       32
explicit capability for planning. Similar to creating summaries, we propose a complementary planning
model which creates a high-level overview of what should be generated next, given the prior context.
The proposed plan could span multiple concepts, such as a paragraph. The LCM is conditioned on
this plan, before it then generates the subsequent output sequence.
Operationally the model predicts auto-regressively a sequence of concepts followed by a break concept,
which represents a natural topic cadence such as a paragraph break. Once the break concept is
predicted, the large planning model (LPM) generates a plan in order to condition the LCM for
prediction of the subsequent sequence. The model then continues generation as usual conditioned on
both the prior concepts and the proposed plan. An overview of the approach is shown in Figure 17.




      Figure 17 - LCM conditioned on both the prior context and a high-level plan for the next sequence.


Although we envision a separate (but complementary) model for planning, we present here an initial
experiment using a simplified single-model approach where the LCM is trained in a multitask setting
to also predict both the break concepts and plans. Additionally, instead of the idealized plan spanning
multiple concepts (such as a paragraph), we use a single concept in order to capture what should
come next (i.e. a plan concept). We call this simplified multitask approach a Large Planning Concept
Model (LPCM).

Methodology
In order to evaluate this single-model approach, we perform an ablation study. As a baseline method,
we train a One-Tower LCM (cf. Section 2.3.3) without any visibility to break or plan concepts.
We then subsequently train a LPCM with the same number of parameters as the baseline. Both
models were trained on the same data7 for the same number of steps.

Data preprocessing. In order to represent break concepts, we begin by first segmenting the data
into paragraphs. Given that most real world datasets are absent of paragraph structure (or it is
not easy to recover), we apply the Segment Any Text (Frohmann et al., 2024) paragraph splitting
API8 . We additionally force each paragraph to be less than 10 sentences, and merge small (e.g. one
sentence) consecutive paragraphs together. In order to represent plan concepts, we generate synthetic
high-level topic description for each preceding segmented paragraph using an existing open-sourced
LLM, namely Llama-3.1-8B-IT, which offers a good trade-off between the generated topic quality
and the generation speed. The system prompt used to generate these topic descriptions is listed in
Appendix C. In total we process approximately 320M paragraphs with topic descriptions, spanning
1.5B segmented concepts (i.e. approximately 30B tokens).

Metrics. We focus on coherence as our main measure of evaluation. Previous ablations (cf.
Section 2.4) used the coherence metric introduced by Jwalapuram et al. (2022). However, we explore
  7
      Compared to previous experiments, we use a different data mixture more favorable for long-form generation.
  8
      https://github.com/segment-any-text/wtpsplit


                                                         33
here LLM-as-a-judge as an alternative. Specifically, we use Llama-3.1-8B-IT in order to evaluate the
coherence of the generated model outputs, which is prompted to return an overall coherence score
between [0, 5]. The prompt used is listed in Appendix D. In order to validate this prompt, we evaluate
it against a dataset of human judgements introduced by Jwalapuram et al. (2022), and observed it
reported an agreement9 with human annotators which improves upon their coherence model. We
therefore choose this metric for our evaluation. To be consistent across both model results, we do
not include the special break or plan concepts generated by the LPCM when calculating coherence
scores.

                                                        Llama-3.1-8B-IT (↑)
                                             LPCM          2.82 ± 0.62
                                             Baseline      2.74 ± 0.70

                                    Table 13 - LPCM ablation coherence score results.



Results
We provide the results of our ablation experiment in Table 13. Results are reported over a held-out
subset of Cosmopedia (Ben Allal et al., 2024) following instruction fine-tuning, similar to previous
ablations (cf. Section 2.4). We observe that the LPCM achieves significantly10 higher coherence
scores (significance was measured using a paired t-test) than the baseline One-Tower LCM. This
finding suggests that the LPCM is capable of producing significantly more coherent outputs than
the LCM as a result of the additional structure coming from predicted plan concepts, helping the
LPCM produce a more coherent narrative, which is essential for the objective of generating long-form
output.


5         Related work

5.1        Sentence representations
Multilingual sentence representations Learning effective sentence embeddings has been a well
studied subject in recent years. Significant progress has been made in this field, largely due to the
capabilities of transformer-based language models that by learning contextual representations for
individual tokens (Devlin et al., 2018; Conneau et al., 2020), are able to effectively learn the semantics
of language. However, such models are not optimal to create sentence representations.
Following approaches built upon these initial works, and aimed at learning general sentence repre-
sentations, leveraging dual encoder architectures (Guo et al., 2018; Reimers and Gurevych, 2019;
Ni et al., 2021). These architectures encode the source and target into a common embedding space,
and use a distance-based metric to create an alignment loss that approximate semantically identical
sentences. Such architectures have been extended to leverage multilingual data to create general,
aligned embedding spaces across languages (Feng et al., 2020; Janeiro et al., 2024; Sturua et al.,
2024). Initial approaches leveraged the contrastive loss to align translations across languages (Feng
et al., 2020; Yang et al., 2019), using only translation data to train. Other architectural changes,
namely using token-level objectives combined with the sentence level objectives, have proven useful
to improve the quality of multilingual sentence representations based on translation data only (Li
    9
         Krippendorff’s α = 0.48
    10
         Significance observed at the 99.9% level.


                                                           34
et al., 2023; Janeiro et al., 2024). Recent approaches explore using data from other tasks, besides
translation data, to increase the generality of the sentence representations (Wang et al., 2024b; Mohr
et al., 2024). Other approaches change their embeddings per task, either with task-specific prompts
(Wang et al., 2024b; Su et al., 2022; Lee et al., 2024b) or with task-specific parameters (Sturua et al.,
2024).
Another successful line of work to create general purpose, multilingual, sentence representations is to
leverage the translation objective. LASER (Artetxe and Schwenk, 2019), and SONAR (Duquenne
et al., 2023b) leverage an encoder-decoder architecture, with a fixed-size sentence representation
between the encoder and the decoder, trained with a translation objective. SONAR is initialized
from the NLLB-200 model (NLLB Team et al., 2022), and covers 200 languages, making it one of the
open-source models with the widest language coverage. SONAR also provides open-source speech
encoders aligned to their sentence encoders for 73 languages (Seamless Communication et al., 2023a),
aligned through a teacher-student approach. SONAR has been used as the basis for several works
(Seamless Communication et al., 2023a,b; Chen et al., 2023a), and its speech decoders have been
extended to keep the expressiveness of the original speech (Duquenne et al., 2023a).

Joint speech/text sentence representations There has been a large body of research on
unsupervised representation learning for monolingual (Baevski et al., 2020) and multilingual speech
(Babu et al., 2022), with recently w2v-bert (Chung et al., 2021) that combines contrastive learning and
masked language modeling to learn self-supervised representations from speech. Other works explored
multilingual and multimodal (speech/text) pre-training methods, including mSLAM (Bapna et al.,
2022). Finally, Duquenne et al. (2021), followed by Khurana et al. (2022), introduced multilingual
and multimodal sentence embeddings, extending a pre-existing multilingual text sentence embedding
space to the speech modality with a distillation approach. Duquenne et al. (2022, 2023c) also showed
that it is possible to efficiently decode multilingual speech sentence embeddings with decoders trained
on text sentence embeddings into different languages, to perform zero-shot speech translation.

LLM based sentence representations Several text representation methods have been proposed
which are based on existing LLMs. Wang et al. (2024a) proposed extracting text embeddings from
the last token of LLMs fine-tuned with instructions on contrastive data. Lee et al. (2024a) improved
text embedding capabilities of fine-tuned LLMs by removing the causal attention mask and applying
extra nonlinear layers before pooling the token embeddings. Embeddings as a service are supported
by some commercial LLM providers, for example, Mistral-embed.11 Such embeddings proved
competitive on retrieval benchmarks; however, to the best of our knowledge, their applicability to
reconstructing the texts back from the embedding space has not been demonstrated.

5.2      Multilingual LLMs
Most of the leading LLMs have been trained on texts in several languages. Table 1 summarizes
the coverage of several of them. Nevertheless, the pretraining data of these LLMs seems to be
mainly English texts. For example, The Llama3 team (2024) mentions that pretraining data contains
significantly more English texts, requiring continued pre-training with multilingual data, out of which
34.6% is translated reasoning data.
There are also several efforts to train LLMs optimized on specific languages, e.g. LeoLM for
German,12 Fuano for Italian (Bacciu et al., 2024), ALLaM for Arabic (Bari et al., 2024), and several
  11
       https://docs.mistral.ai/capabilities/embeddings/
  12
       https://laion.ai/blog/leo-lm/


                                                          35
models for Chinese: ErniBot,13 Tongyi Qianwen,14 or ChatGLM (Team GLM et al., 2024).
Some adaptations of LLMs to a massive number of languages also exist. LOLA (Srivastava et al., 2024)
is a recent mixture-of-experts LLM supporting 160 languages, MALA-500 (Ji et al., 2024) adapts
LLaMA2 to 546 languages. However, such models typically face a trade-off between language coverage
and other capabilities. For example, the Aya model (Üstün et al., 2024), following instructions in
101 languages, was superseded by Aya-23 (Aryabumi et al., 2024) that exchanged some breadth for
depth, focusing on 23 languages only. The LCM architecture, combining a language-agnostic model
for knowledge and reasoning with potentially language-specialized encoders and decoders, is expected
to exhibit this trade-off to a lesser extent.

5.3      Alternative LLM architectures
Predicting the next state in the embedding space is a core idea of the Joint Embedding Predictive
Architecture (Jepa) proposed by LeCun (2022). This idea has been implemented for images (I-JEPA
by Assran et al. (2024)) and video (V-JEPA by Bardes et al. (2024)) as a self-supervised approach
to learning representations. For language, equivalent models have not yet been explored.


Sentence embeddings for language modeling. For text completion, Ippolito et al. (2020)
proposed a sentence-level language model operating by choosing the next sentence from a finite set
of candidates. Their model demonstrated success in selecting appropriate continuations for short
stories, but it has not been scaled to longer inputs or to fully generative outputs. Golestani et al.
(2021) studied a similar problem in the even more restrictive sentence ordered setting, but with a
more thorough study of architectural choices. The INSET architecture (Huang et al., 2020) solves the
sentence infilling task by combining a denoising autoencoder that encodes sentences into fixed-size
vectors and decodes them back and a bidirectional transformer that predicts the embedding of a
missing sentence.

Marfurt and Henderson (2021) and Cornille et al. (2024) used predicted next sentence embeddings in
a fully generative setting, for summarization and generic language modeling, respectively. However,
their architectures considered sentence-level connections only as an addition to the token-level
connections across sentences, not as their replacement.

In a recent work of An et al. (2024), the SentenceVAE architecture performs language modeling on
the sentence level using a sentence encoder to prepare the inputs and a sentence decoder to produce
the outputs. However, its input and output embedding spaces are not tied, so the inference is only
possible by decoding each predicted sentence into text and then re-encoding it for adding it to the
context.


Language modeling with diffusion. A series of more recent works tried adapting diffusion
modeling, originally developed for continuous data, to the discrete text domain. The PLANNER
architecture (Zhang et al., 2023) consists of a variational autoencoder for paragraphs and a diffusion
model trained to predict latent autoencoder representations conditional on the textual context or on
the class label. Lovelace et al. (2024) augmented a decoder-only language model with an encoded
semantic proposal of the continuation text, with an easily guidable diffusion model predicting the
embedding of the next proposal. A TEncDM model (Shabalin et al., 2024) performs diffusion in the
space of contextual token embeddings which are then decoded non-autoregressively.
  13
       http://research.baidu.com/Blog/index-view?id=183
  14
       https://www.alibabacloud.com/en/solutions/generative-ai


                                                       36
Some applications of diffusion to sequence modeling have targeted the planning capabilities of the
sequence models. Semformer (Yin et al., 2024) proposed training transformers language models
to plan several steps ahead by including special planning tokens, the representations of which are
trained to be informative about the future tokens. Ye et al. (2024) applied discrete diffusion to
language models as an alternative to autoregressive generation, more suitable for tasks that require
multi-step planning. Ubukata et al. (2024) give an overview of applications of diffusion for planning
tasks, but most of them are not concerned with the language domain.
Overall, while many of the previous works used hidden representations for language modeling or
related tasks, all of them either relied on token-level inputs or outputs, or were not intented for
generating texts of arbitrary length. The LCM seems to be the first fully generative language model
implemented fully in a highly semantic, reconstructable sentence representation space.


6   Limitations
In this section we discuss the possible limitations of the presented Large Concept Modeling approach.

Choice of the embedding space. The choice and design of the embedding space plays a crucial
role in the LCM modeling approach.
    • The SONAR embedding space was chosen for its good multilingual and multimodal represen-
      tations, as well as the availability of a massively multilingual decoder, which achieves excellent
      results in both translation and auto-encoding. However, the SONAR model was trained on
      very specific training data, namely bitext machine translation data containing rather short
      sentences. This has several consequences:
        1. SONAR is trained to sustain a local geometry (sentences with very similar meanings are
           geometrically close) with no special guarantees for sentences that are only loosely related.
           Yet, predicting next sentences distribution requires the space to operate well globally.
        2. SONAR auto-encodes surprisingly well texts containing links, references, or merely
           numbers or code data. Yet, such texts tend to be fragile, highlighting a distribution
           mismatch between the SONAR training data and commonly used LLM pre-training text
           corpora. Therefore, the accurate prediction of the sentences containing such a content
           (non-negligible in LCM pre-training data) will be hard for any LCM SONAR based model.
           For instance, the factuality of fragile generated sentences may easily be compromised.
    • Using a frozen encoder represents some interesting trade-offs. Any frozen encoder which is
      learned in a different data context, and with no a-priori strong connection to LCM modeling,
      may be suboptimal compared to encoders that are learned in an end-to-end fashion (with
      the loss coming from the decoder). At the same time, learning an encoder within end-to-end
      training can be challenging and the resulting space is not guaranteed to result in good semantic
      representations shared across languages and modalities.
      Training the concept representation and the LCM end-to-end would also be less data and
      compute efficient since all modeling data should be multilingual and -modal, bearing the risk
      of modality competition.




                                                  37
Concept granularity
  • In this work, the definition of concepts is interpreted at sentence level. However, the manifold of
    possible next sentences is very wide, attributing a proper probability to each of such sentences
    is much harder (even with a modeling within the latent space) that to the discrete set of tokens.
  • In NLP, we encounter sentences of variable length. Combinatorial complexity of possible next
    sentences grows exponentially with the maximum character length. The choice of granularity
    for LCM is not trivial as long sentences (>120 characters) could reasonably be considered as
    several concepts. However, any finer splitting of such sentences does not necessary separate
    well these concepts. This shows the limitation of a fixed size embedding representation for one
    sentence. Text splitting (such as sentence splitting) or one-to-many mapping of a sentence into
    several embeddings is a major future direction of research.
  • Each document in a training corpus typically contains a sequence of unique sentences or a little
    number of repetitions. This data sparsity effect manifests as well at large corpora level: the
    large majority of sentences are merely unique. In principle, this issue can be addressed with
    higher-level semantic embedding representations. These higher-level representations come with
    trade-off between requirement of lossless data encoding (think of named-entities or numbers,
    critical in many language modeling tasks) and good level of abstraction to enable reasoning
    capabilities. Compared to a monolingual auto-encoder which would simply compress input
    sentences, SONAR offers semantic representations with good auto-encoding quality but still
    certainly sacrificing generalization capacities.
  • This generalization issue can be partially mitigated by splitting or encoding input text as new
    conceptual units which are more commonly shared across source documents. This is in the spirit
    of stemming or lemmatization techniques studied in NLP for words. That being said, building
    such conceptual units that are also language and modality agnostic is a challenging task. Such
    shared multilingual and multimodal conceptual units are also key for generalization across
    languages and across modalities. To maximize cross-lingual and cross-modal transfers, Large
    Concept Models should be exposed to a richer variety of multilingual and multi-modal data.
Continuous versus discrete
  • Diffusion modeling has proven to be very efficient in generative modeling of continuous data
    like images or speech. As previously stated, sentences in the SONAR space, despite being
    represented as continuous vectors, remain discrete combinatorial objects. This makes diffusion
    modeling struggle on the text modality (either at word or sentence embedding level).
  • The contrastive nature of cross-entropy loss based on softmax outputs which is used for next
    token prediction plays a critical role for many downstream task where higher accuracy is
    required (e.g. MCQ tasks, code or math generation). On the opposite, continuous diffusion
    modeling does not allow to integrate such a contrastive objective.
  • The Quant-LCM could be a way to address the discrete nature of text while modeling on
    coarse-to-fine semantic units shared across languages and modalities. The limited performance
    of the Quant-LCM approaches presented in this paper may be explained by the fact that
    SONAR space was not trained to be efficiently quantizable, yielding a significant number
    of codebooks and a large amount of units per codebook. Therefore, the current SONAR
    quantization suffers from the exponentially increasing number of RVQ units combinations which
    does not solve the data sparsity/uniqueness issue discussed earlier. This indicates once again
    the importance of developing a new representation space, either continuous or discrete, for the
    Large Concept Model.

                                                 38
7   Acknowledgments
We would like to thank Robbie Adkins, Can Balioglu, Joy Chen, Pascale Fung, Jason Holland, Amita
Kamath, Justine Kao, Sagar Miglani, Alice Rakotoarison, Abhilasha Sancheti, Arjang Talattof, Ellen
Tan, Carleigh Wood, Shireen Yates, Bokai Yu and Luke Zettlemoyer for comments and suggestions
on this work, as well helping us to improve this paper.


8   Conclusion and Future Work
Current best practice for large scale language modeling is to operate at the token level, i.e. to
learn to predict the next tokens given a sequence of preceding tokens. There is a large body of
research on improvements of LLMs, but most works concentrate on incremental changes and do
not question the main underlying architecture. In this paper, we have proposed a new architecture,
named a Large Concept Model (LCM), which substantially differs from current LLMs in two
aspects: 1) all modeling is performed in a high-dimensional embedding space instead of on a discrete
token representation; and 2) modeling is not instantiated in a particular language or modality, but
at a higher semantic and abstract level. We have named the general form of this representation
a “concept”.
In this paper, to verify the feasibility of the high-level idea, we have assumed that a concept corresponds
to a sentence in the text domain, or an equivalent speech segment, and that the embeddings are
obtained by the freely available SONAR sentence encoder (Duquenne et al., 2023b). With respect to
the specific architecture of the LCM, we have first shown that directly minimizing the MSE loss in
the embedding space does not yield good results. We then explored several architectures based on a
diffusion process: the One-Tower and Two-Tower LCM, as well as a Quant-LCM which uses
quantization of SONAR representations and then modeling on these discrete units. These ablation
experiments were performed with models with 1.6B parameters and focused on the generative task of
continuing a sequence of sentences. We have then scaled our models to a size of 7B parameters and
instruction-finetuned them on several summarization and summary expansion tasks. We provide a
detailed comparison to other public models of the same size, namely Gemma, Mistral and Llama.
By design, a LCM exhibits strong zero-shot generalization performance. In this paper, we trained
models on English texts only, and applied them to text in other languages, without any additional
training data, neither aligned nor unlabeled. The LCM outperforms Llama-3.1-8B-IT on English
and on the average over foreign languages officially supported by the LLM. The LCM itself could
also be trained on multilingual- and model data to acquire knowledge from these sources. We will
explore this in future versions of the LCM. In short, all languages and modalities are first class
citizens and handled equally at all stages of a LCM.
We have observed that next sentence prediction is substantially more challenging than next token
prediction. First, given that we operate in an embedding space and at a higher semantic level, the
number of possible sentences is virtually unlimited, while token vocabularies are usually in the range
of 100k. Second, even given a long context, there is unavoidably more ambiguity in choosing the
next sentence than the next token. And third, the usual softmax output layer over the fixed size
token vocabulary provides a normalized probability distribution over all possible token continuations.
Theoretically, a diffusion process should be able to learn a probability distribution over an output
embedding space, but our current experimental evidence indicates that more research is needed to
take full advantage of the properties of Large Concept Models. As an example, the ability
to sample multiple embeddings and associate a score would enable beam search to find the best
sequence of sentences. Finally, small modeling errors could yield predictions in the embedding space

                                                    39
which do not correspond to valid sentences, i.e. that cannot be decoded into a syntactically and
semantically correct sentence. We will work on alternative concept embeddings to SONAR which
would be better suited to the next sentence prediction task, and would improve modeling approaches
in that concept embedding space.
We see the models and results discussed in this paper as a step towards increasing scientific diversity
and a move away from current best practice in large scale language modeling. We acknowledge that
there is still a long path to reach the performance of current flagship LLMs. This will require of
course further improving the core architecture, but also careful data selection and curation, extensive
ablations, optimized and diverse instruction fine-tuning, and finally, scaling to models with more
than 70B parameters.
We open-source the full training code of all our LCM variants, together with a set of supporting
scripts,15 to make it easy for other teams to train LCM models. By these means, we hope to foster
research on alternative LLMs and contribute to advance the field of machine intelligence.


References
A. Aghajanyan, L. Yu, A. Conneau, W.-N. Hsu, K. Hambardzumyan, S. Zhang, S. Roller, N. Goyal, O. Levy,
  and L. Zettlemoyer. Scaling laws for generative mixed-modal language models. In International Conference
  on Machine Learning, pages 265–279. PMLR, 2023.
E. Almazrouei, H. Alobeidli, A. Alshamsi, A. Cappelli, R. Cojocaru, M. Debbah, Étienne Goffinet, D. Hesslow,
  J. Launay, Q. Malartic, D. Mazzotta, B. Noune, B. Pannier, and G. Penedo. The Falcon series of open
   language models. ArXiv, abs/2311.16867, 2023. URL https://arxiv.org/pdf/2311.16867.
H. An, Y. Chen, Z. Sun, and X. Li. SentenceVAE: Enable next-sentence prediction for large language models
  with faster speed, higher accuracy and longer context. arXiv preprint arXiv:2408.00655, 2024.
Anthropic. The Claude 3 model family: Opus, sonnet, haiku, 2024. URL https://www-cdn.anthropic.com/
  de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf.
M. Artetxe and H. Schwenk. Massively multilingual sentence embeddings for zero-shot cross-lingual transfer
  and beyond. TACL, pages 597–610, 2019.
V. Aryabumi, J. Dang, D. Talupuru, S. Dash, D. Cairuz, H. Lin, B. Venkitesh, M. Smith, K. Marchisio, S. Ruder,
  et al. Aya 23: Open weight releases to further multilingual progress. arXiv preprint arXiv:2405.15032, 2024.
M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas. Self-supervised
  learning from images with a joint-embedding predictive architecture. ArXiv, abs/2301.08243, 2024. URL
  https://arxiv.org/pdf/2301.08243.
A. Babu, C. Wang, A. Tjandra, K. Lakhotia, Q. Xu, N. Goyal, K. Singh, P. von Platen, Y. Saraf, J. Pino,
  A. Baevski, A. Conneau, and M. Auli. XLS-R: Self-supervised Cross-lingual Speech Representation Learning
  at Scale. In Proc. Interspeech 2022, pages 2278–2282, 2022. doi: 10.21437/Interspeech.2022-143.
A. Bacciu, G. Trappolini, A. Santilli, E. Rodolà, and F. Silvestri. Fauno: The italian large language model
  that will leave you senza parole! ArXiv, abs/2306.14457, 2024. URL https://arxiv.org/pdf/2306.14457.
A. Baevski, Y. Zhou, A. Mohamed, and M. Auli. wav2vec 2.0: A framework for self-supervised learning of
  speech representations. NeurIPS, 33:12449–12460, 2020.
C. Balioglu. fairseq2, 2023. URL http://github.com/facebookresearch/fairseq2.
A. Bapna, C. Cherry, Y. Zhang, Y. Jia, M. Johnson, Y. Cheng, S. Khanuja, J. Riesa, and A. Conneau. mslam:
  Massively multilingual joint pre-training for speech and text. arXiv preprint arXiv:2202.01374, 2022.
  15
       https://github.com/facebookresearch/large_concept_model


                                                      40
A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas. Revisiting
  feature prediction for learning visual representations from video. ArXiv, abs/2404.08471, 2024. URL
  https://arxiv.org/pdf/2404.08471.
M. S. Bari, Y. Alnumay, N. A. Alzahrani, N. M. Alotaibi, H. A. Alyahya, S. AlRashed, F. A. Mirza, S. Z.
 Alsubaie, H. A. Alahmed, G. Alabduljabbar, R. Alkhathran, Y. Almushayqih, R. Alnajim, S. Alsubaihi,
  M. A. Mansour, M. Alrubaian, A. Alammari, Z. Alawami, A. Al-Thubaity, A. Abdelali, J. Kuriakose,
 A. Abujabal, N. Al-Twairesh, A. Alowisheq, and H. Khan. ALLaM: Large language models for arabic and
  english. ArXiv, abs/2407.15390, 2024. URL https://arxiv.org/pdf/2407.15390.
L. Ben Allal, A. Lozhkov, G. Penedo, T. Wolf, and L. von Werra. Cosmopedia, 2024. URL https://
  huggingface.co/datasets/HuggingFaceTB/cosmopedia.
J. Betker, G. Goh, L. Jing, TimBrooks, J. Wang, L. Li, LongOuyang, JuntangZhuang, JoyceLee, YufeiGuo,
   WesamManassra, PrafullaDhariwal, CaseyChu, YunxinJiao, and A. Ramesh. Improving image generation
   with better captions, 2023. URL https://api.semanticscholar.org/CorpusID:264403242.
BigScience Workshop. BLOOM: a 176b-parameter open-access multilingual language model.                ArXiv,
  abs/2211.05100, 2023. URL https://arxiv.org/pdf/2211.05100.
Chameleon team. Chameleon: Mixed-modal early-fusion foundation models. ArXiv, abs/2405.09818, 2024.
  URL https://arxiv.org/pdf/2405.09818.
M. Chen, P.-A. Duquenne, P. Andrews, J. Kao, A. Mourachko, H. Schwenk, and M. R. Costa-jussà. BLASER:
  A text-free speech-to-speech translation evaluation metric. In ACL, pages 9064–9079, 2023a. URL
  https://aclanthology.org/2023.acl-long.504.
M. Chen, K. Heffernan, O. Çelebi, A. Mourachko, and H. Schwenk. xSIM++: An improved proxy to
  bitext mining performance for low-resource languages. In ACL, pages 101–109, 2023b. URL https:
 //aclanthology.org/2023.acl-short.10.
R. Child, S. Gray, A. Radford, and I. Sutskever. Generating long sequences with sparse transformers. arXiv
  preprint arXiv:1904.10509, 2019.
H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma,
  et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53,
  2024.
Y.-A. Chung, Y. Zhang, W. Han, C.-C. Chiu, J. Qin, R. Pang, and Y. Wu. W2v-bert: Combining contrastive
  learning and masked language modeling for self-supervised speech pre-training. In 2021 IEEE Automatic
  Speech Recognition and Understanding Workshop (ASRU), pages 244–250. IEEE, 2021.
E. Clark, S. Rijhwani, S. Gehrmann, J. Maynez, R. Aharoni, V. Nikolaev, T. Sellam, A. Siddhant, D. Das,
  and A. Parikh. Seahorse: A multilingual, multifaceted dataset for summarization evaluation. In Proceedings
  of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9397–9413, 2023.
A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán, E. Grave, M. Ott, L. Zettle-
  moyer, and V. Stoyanov. Unsupervised cross-lingual representation learning at scale. In ACL, 2020.
N. Cornille, M.-F. Moens, and F. Mai. Learning to plan for language modeling from unlabeled data. arXiv
  preprint arXiv:2404.00614, 2024.
M. R. Costa-jussà, P. Andrews, M. C. Megliogli, J. Chen, J. Chuang, D. Dale, C. Ropers, A. Mourachko,
  E. Sánchez, H. Schwenk, T. Tran, A. Turkatenko, and C. Wood. LCFO: Long context and long form output
  dataset and benchmarking. ArXiv, 2024.
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of deep bidirectional transformers
   for language understanding. arXiv preprint arXiv:1810.04805, 2018.
P. Dhariwal and A. Nichol. Diffusion models beat gans on image synthesis. Advances in neural information
   processing systems, 34:8780–8794, 2021.


                                                    41
V. Dijk. Text and Context: Explorations in the Semantics and Pragmatics of Discourse. Longman, 1977.
M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré, M. Lomeli, L. Hosseini, and H. Jégou.
  The Faiss library. ArXiv, abs/2401.08281, 2024. URL https://arxiv.org/pdf/2401.08281.
P.-A. Duquenne, H. Gong, and H. Schwenk. Multimodal and multilingual embeddings for large-scale speech
   mining. In NeurIPS, volume 34, pages 15748–15761, 2021. URL https://proceedings.neurips.cc/paper/
   2021/file/8466f9ace6a9acbe71f75762ffc890f1-Paper.pdf.
P.-A. Duquenne, H. Gong, B. Sagot, and H. Schwenk. T-modules: Translation modules for zero-shot cross-
   modal machine translation. In EMNLP, pages 5794–5806, 2022. URL https://aclanthology.org/2022.
   emnlp-main.391.pdf.
P.-A. Duquenne, K. Heffernan, A. Mourachko, B. Sagot, and H. Schwenk. Sonar expressive: Zero-shot
   expressive speech-to-speech translation, 2023a.
P.-A. Duquenne, H. Schwenk, and B. Sagot. SONAR: sentence-level multimodal and language-agnostic
   representations, 2023b. URL https://arxiv.org/abs/2308.11466.
P.-A. Duquenne, H. Schwenk, and B. Sagot. Modular speech-to-text translation for zero-shot cross-modal
   transfer. In Interspeech, 2023c.
F. Feng, Y. Yang, D. Cer, N. Arivazhagan, and W. Wang. Language-agnostic BERT sentence embedding.
  arXiv preprint arXiv:2007.01852, 2020.
M. Frohmann, I. Sterner, I. Vulić, B. Minixhofer, and M. Schedl. Segment Any Text: A universal approach
  for robust, efficient and adaptable sentence segmentation. In EMNLP, pages 11908–11941, 2024. URL
  https://aclanthology.org/2024.emnlp-main.665.
O. Gafni, A. Polyak, O. Ashual, S. Sheynin, D. Parikh, and Y. Taigman. Make-a-scene: Scene-based text-to-
  image generation with human priors. In European Conference on Computer Vision, pages 89–106. Springer,
  2022.
Gemini Team Google. Gemini 1.5 unlocking multimodal understanding across millions of tokens of conte.
 ArXiv, abs/2403.05530, 2024. URL https://arxiv.org/pdf/2403.05530.
M. Golestani, S. Z. Razavi, Z. Borhanifard, F. Tahmasebian, and H. Faili. Using BERT encoding and
  sentence-level language model for sentence ordering. In International Conference on Text, Speech, and
 Dialogue, pages 318–330. Springer, 2021.
P. Goyal. Accurate, large minibatch sg d: training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.
M. Guo, Q. Shen, Y. Yang, H. Ge, D. Cer, G. H. Abrego, K. Stevens, N. Constant, Y.-H. Sung, B. Strope, et al.
  Effective parallel corpus mining using bilingual sentence embeddings. arXiv preprint arXiv:1807.11906,
  2018.
T. Hang, S. Gu, C. Li, J. Bao, D. Chen, H. Hu, X. Geng, and B. Guo. Efficient diffusion training via min-snr
  weighting strategy. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages
  7441–7451, 2023.
T. Hasan, A. Bhattacharjee, M. S. Islam, K. Mubasshir, Y.-F. Li, Y.-B. Kang, M. S. Rahman, and R. Shahriyar.
  XL-sum: Large-scale multilingual abstractive summarization for 44 languages. In Findings of the Association
  for Computational Linguistics: ACL-IJCNLP 2021, pages 4693–4703, Online, Aug. 2021. Association for
  Computational Linguistics. URL https://aclanthology.org/2021.findings-acl.413.
K. Heffernan and S. Teufel. Problem-solving recognition in scientific text. In Proceedings of the Thirteenth
  Language Resources and Evaluation Conference, pages 6045–6058, Marseille, France, June 2022. European
  Language Resources Association. URL https://aclanthology.org/2022.lrec-1.650.
K. M. Hermann, T. Kocisky, E. Grefenstette, L. Espeholt, W. Kay, M. Suleyman, and P. Blunsom. Teaching
  machines to read and comprehend. Advances in neural information processing systems, 28, 2015.


                                                     42
J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. CoRR, abs/2006.11239, 2020. URL
   https://arxiv.org/abs/2006.11239.
J. Ho, W. Chan, C. Saharia, J. Whang, R. Gao, A. Gritsenko, D. P. Kingma, B. Poole, M. Norouzi, D. J.
   Fleet, and T. Salimans. Imagen video: High definition video generation with diffusion models. ArXiv,
   abs/2210.02303, 2022. URL https://arxiv.org/abs/2210.02303.
M. Honnibal, I. Montani, S. Van Landeghem, and A. Boyd. spaCy: Industrial-strength Natural Language
  Processing in Python, 2020.
Y. Huang, Y. Zhang, O. Elachqar, and Y. Cheng. INSET: Sentence infilling with inter-sentential transformer.
  In ACL, pages 2502–2515, 2020. URL https://aclanthology.org/2020.acl-main.226.pdf.
D. Ippolito, D. Grangier, D. Eck, and C. Callison-Burch. Toward better storylines with sentence-level language
  models. ArXiv, abs/2005.05255, 2020. URL https://arxiv.org/pdf/2005.05255.
J. M. Janeiro, B. Piwowarski, P. Gallinari, and L. Barrault. Mexma: Token-level objectives improve sentence
   representations, 2024. URL https://arxiv.org/abs/2409.12737.
S. Ji, Z. Li, I. Paul, J. Paavola, P. Lin, P. Chen, D. O’Brien, H. Luo, H. Schütze, J. Tiedemann, et al. Emma-500:
   Enhancing massively multilingual adaptation of large language models. arXiv preprint arXiv:2409.17892,
   2024.
A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. de las Casas,
  E. B. Hanna, F. Bressand, G. Lengyel, G. Bour, G. Lample, L. R. Lavaud, L. Saulnier, M.-A. Lachaux,
  P. Stock, S. Subramanian, S. Yang, S. Antoniak, T. L. Scao, T. Gervet, T. Lavril, T. Wang, T. Lacroix,
  and W. E. Sayed. Mixtral. ArXiv, abs/2401.04088, 2024. URL https://arxiv.org/pdf/2401.04088.
P. Jwalapuram, S. Joty, and X. Lin. Rethinking self-supervision objectives for generalizable coherence modeling.
   In ACL, pages 6044–6059, 2022. URL https://aclanthology.org/2022.acl-long.418.
T. Karras, M. Aittala, T. Aila, and S. Laine. Elucidating the design space of diffusion-based generative
  models. Advances in neural information processing systems, 35:26565–26577, 2022.
S. Khurana, A. Laurent, and J. Glass. Samu-xlsr: Semantically-aligned multimodal utterance-level cross-lingual
   speech representation. arXiv preprint arXiv:2205.08180, 2022.
D. Kingma and R. Gao. Understanding diffusion objectives as the elbo with simple data augmentation.
  Advances in Neural Information Processing Systems, 36, 2024.
N. Kitaev, Ł. Kaiser, and A. Levskaya. Reformer: The efficient transformer. arXiv preprint arXiv:2001.04451,
  2020.
K. Krishna, J. Wieting, and M. Iyyer. Reformulating unsupervised style transfer as paraphrase generation. In
  Empirical Methods in Natural Language Processing, 2020.
Y. LeCun. A path towards autonomous machine intelligence, 2022. URL https://openreview.net/pdf?id=
  BZ5a1r-kVsf.
C. Lee, R. Roy, M. Xu, J. Raiman, M. Shoeybi, B. Catanzaro, and W. Ping. NV-Embed: Improved techniques
  for training LLMs as generalist embedding models. arXiv preprint arXiv:2405.17428, 2024a.
D. Lee, C. Kim, S. Kim, M. Cho, and W.-S. Han. Autoregressive image generation using residual quantization.
  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523–
  11532, 2022.
J. Lee, Z. Dai, X. Ren, B. Chen, D. Cer, J. R. Cole, K. Hui, M. Boratko, R. Kapadia, W. Ding, Y. Luan,
   S. M. K. Duddu, G. H. Abrego, W. Shi, N. Gupta, A. Kusupati, P. Jain, S. R. Jonnalagadda, M.-W.
   Chang, and I. Naim. Gecko: Versatile text embeddings distilled from large language models, 2024b. URL
   https://arxiv.org/abs/2403.20327.


                                                       43
K. Lee and S. Sengupta. Introducing the ai research supercluster — meta’s cutting-edge ai supercomputer for
  ai research, 2022. URL https://ai.facebook.com/blog/ai-rsc/.
Y. Li, Q. Chen, W. Yan, W. Wang, Q. Zhang, and H. Sundaram. Advancing precise outline-conditioned text
  generation with task duality and explicit outline control, 2024. URL https://arxiv.org/abs/2305.14459.
Z. Li, S. Huang, Z. Zhang, Z.-H. Deng, Q. Lou, H. Huang, J. Jiao, F. Wei, W. Deng, and Q. Zhang. Dual-
   alignment pre-training for cross-lingual sentence embedding. In A. Rogers, J. Boyd-Graber, and N. Okazaki,
   editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume
  1: Long Papers), pages 3466–3478, Toronto, Canada, July 2023. Association for Computational Linguistics.
   doi: 10.18653/v1/2023.acl-long.191. URL https://aclanthology.org/2023.acl-long.191.
C.-Y. Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out,
  pages 74–81, 2004.
S. Lin, B. Liu, J. Li, and X. Yang. Common diffusion noise schedules and sample steps are flawed. In
  Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 5404–5411, 2024.
S. Liu, H. Lu, and J. Shao. Improved residual vector quantization for high-dimensional approximate nearest
   neighbor search. arXiv preprint arXiv:1509.05195, 2015.
J. Lovelace, V. Kishore, Y. Chen, and K. Q. Weinberger. Diffusion guided language modeling. ArXiv,
   abs/2408.04220, 2024. URL https://arxiv.org/pdf/2408.04220.
A. Lozhkov, L. Ben Allal, L. von Werra, and T. Wolf. Fineweb-edu, May 2024. URL https://huggingface.
  co/datasets/HuggingFaceFW/fineweb-edu.
C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic
  model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787,
  2022.
A. Marfurt and J. Henderson. Sentence-level planning for especially abstractive summarization. In ACL,
  pages 1–14, 2021. URL https://aclanthology.org/2021.newsum-1.1.pdf.
B. Minixhofer, J. Pfeiffer, and I. Vulić. Where’s the point? self-supervised multilingual punctuation-agnostic
  sentence segmentation. In Proceedings of the 61st Annual Meeting of the Association for Computational
  Linguistics (Volume 1: Long Papers), pages 7215–7235, Toronto, Canada, July 2023. Association for
  Computational Linguistics. URL https://aclanthology.org/2023.acl-long.398.
I. Mohr, M. Krimmel, S. Sturua, M. K. Akram, A. Koukounas, M. Günther, G. Mastrapas, V. Ravishankar,
   J. F. Martínez, F. Wang, Q. Liu, Z. Yu, J. Fu, S. Ognawala, S. Guzman, B. Wang, M. Werk, N. Wang,
   and H. Xiao. Multi-task contrastive learning for 8192-token bilingual text embeddings, 2024. URL
   https://arxiv.org/abs/2402.17016.
N. Mostafazadeh, N. Chambers, X. He, D. Parikh, D. Batra, L. Vanderwende, P. Kohli, and J. Allen. A
  corpus and cloze evaluation for deeper understanding of commonsense stories. In NAACL, pages 839–849,
  June 2016. URL https://aclanthology.org/N16-1098.
S. Narayan, S. B. Cohen, and M. Lapata. Don’t give me the details, just the summary! topic-aware
   convolutional neural networks for extreme summarization. arXiv preprint arXiv:1808.08745, 2018.
J. Ni, G. H. Abrego, N. Constant, J. Ma, K. B. Hall, D. Cer, and Y. Yang. Sentence-t5: Scalable sentence
   encoders from pre-trained text-to-text models. arXiv preprint arXiv:2108.08877, 2021.
A. Q. Nichol and P. Dhariwal. Improved denoising diffusion probabilistic models. In International conference
  on machine learning, pages 8162–8171. PMLR, 2021.
A. Q. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. Mcgrew, I. Sutskever, and M. Chen. GLIDE:
  Towards photorealistic image generation and editing with text-guided diffusion models. In K. Chaudhuri,
  S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, editors, Proceedings of the 39th International
  Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 16784–
  16804. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/nichol22a.html.


                                                     44
M. Ning, M. Li, J. Su, A. A. Salah, and I. O. Ertugrul. Elucidating the exposure bias in diffusion models.
 arXiv preprint arXiv:2308.15321, 2023.
NLLB Team, M. R. Costa-jussà, J. Cross, O. Çelebi, M. Elbayad, K. Heafield, K. Heffernan, E. Kalbassi,
 J. Lam, D. Licht, J. Maillard, A. Sun, S. Wang, G. Wenzek, A. Youngblood, B. Akula, L. Barrault, G. Mejia-
  Gonzalez, P. Hansanti, J. Hoffman, S. Jarrett, K. R. Sadagopan, D. Rowe, S. Spruit, C. Tran, P. Andrews,
  N. F. Ayan, S. Bhosale, S. Edunov, A. Fan, C. Gao, V. Goswami, F. Guzmán, P. Koehn, A. Mourachko,
  C. Ropers, S. Saleem, H. Schwenk, and J. Wang. No language left behind: Scaling human-centered machine
  translation, 2022. URL https://arxiv.org/abs/2207.04672.
OpenAI. GPT-4 technical report. ArXiv, abs/2303.08774, 2024. URL https://arxiv.org/pdf/2303.08774.
K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu. Bleu: a method for automatic evaluation of machine
  translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics,
  pages 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics.
  doi: 10.3115/1073083.1073135. URL https://aclanthology.org/P02-1040.
A. Parola, J. M. Lin, A. Simonsen, V. Bliksted, Y. Zhou, H. Wang, L. Inoue, K. Koelkebeck, and R. Fusaroli.
  Speech disturbances in schizophrenia: Assessing cross-linguistic generalizability of nlp automated measures
  of coherence. Schizophrenia Research, 259:59–70, 2023.
W. Peebles and S. Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF
 International Conference on Computer Vision, pages 4195–4205, 2023.
E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville. Film: Visual reasoning with a general
  conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, 2018.
A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language models are unsupervised
  multitask learners. OpenAI blog, 1(8):9, 2019.
C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring
  the limits of transfer learning with a unified text-to-text transformer. arXiv e-prints, 2019.
C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring the
  limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research,
  21(140):1–67, 2020. URL http://jmlr.org/papers/v21/20-074.html.
N. Reimers and I. Gurevych. Sentence-bert: Sentence embeddings using siamese Bert-networks. arXiv preprint
  arXiv:1908.10084, 2019.
R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent
  diffusion models. CoRR, abs/2112.10752, 2021. URL https://arxiv.org/abs/2112.10752.
P. K. Rubenstein, C. Asawaroengchai, D. D. Nguyen, A. Bapna, Z. Borsos, F. de Chaumont Quitry,
  P. Chen, D. E. Badawy, W. Han, E. Kharitonov, H. Muckenhirn, D. Padfield, J. Qin, D. Rozenberg,
  T. N. Sainath, J. Schalkwyk, M. Sharifi, M. T. Ramanovich, M. Tagliasacchi, A. Tudor, M. Velimirovic,
  D. Vincent, J. Yu, Y. Wang, V. Zayats, N. Zeghidour, Y. Zhang, Z. Zhang, L. Zilka, and C. H. Frank.
  Audiopalm: A large language model that can speak and listen. CoRR, abs/2306.12925, 2023. URL
  https://doi.org/10.48550/arXiv.2306.12925.
T. Salimans and J. Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint
  arXiv:2202.00512, 2022.
Seamless Communication, L. Barrault, Y.-A. Chung, M. C. Meglioli, D. Dale, N. Dong, M. Duppenthaler,
  P.-A. Duquenne, B. Ellis, H. Elsahar, J. Haaheim, J. Hoffman, M.-J. Hwang, H. Inaguma, C. Klaiber,
  I. Kulikov, P. Li, D. Licht, J. Maillard, R. Mavlyutov, A. Rakotoarison, K. R. Sadagopan, A. Ramakrishnan,
  T. Tran, G. Wenzek, Y. Yang, E. Ye, I. Evtimov, P. Fernandez, C. Gao, P. Hansanti, E. Kalbassi, A. Kallet,
  A. Kozhevnikov, G. M. Gonzalez, R. S. Roman, C. Touret, C. Wong, C. Wood, B. Yu, P. Andrews,
  C. Balioglu, P.-J. Chen, M. R. Costa-jussà, M. Elbayad, H. Gong, F. Guzmán, K. Heffernan, S. Jain, J. Kao,
  A. Lee, X. Ma, A. Mourachko, B. Peloquin, J. Pino, S. Popuri, C. Ropers, S. Saleem, H. Schwenk, A. Sun,


                                                     45
  P. Tomasello, C. Wang, J. Wang, S. Wang, and M. Williamson. Seamless: Multilingual expressive and
  streaming speech translation. ArXiv, abs/2312.05187, 2023a. URL https://arxiv.org/abs/2312.05187.

Seamless Communication, L. Barrault, Y.-A. Chung, M. C. Meglioli, D. Dale, N. Dong, P.-A. Duquenne,
  H. Elsahar, H. Gong, K. Heffernan, J. Hoffman, C. Klaiber, P. Li, D. Licht, J. Maillard, A. Rakotoarison,
  K. R. Sadagopan, G. Wenzek, E. Ye, B. Akula, P.-J. Chen, N. E. Hachem, B. Ellis, G. M. Gonzalez,
  J. Haaheim, P. Hansanti, R. Howes, B. Huang, M.-J. Hwang, H. Inaguma, S. Jain, E. Kalbassi, A. Kallet,
  I. Kulikov, J. Lam, D. Li, X. Ma, R. Mavlyutov, B. Peloquin, M. Ramadan, A. Ramakrishnan, A. Sun,
  K. Tran, T. Tran, I. Tufanov, V. Vogeti, C. Wood, Y. Yang, B. Yu, P. Andrews, C. Balioglu, M. R.
  Costa-jussà, O. Celebi, M. Elbayad, C. Gao, F. Guzmán, J. Kao, A. Lee, A. Mourachko, J. Pino, S. Popuri,
  C. Ropers, S. Saleem, H. Schwenk, P. Tomasello, C. Wang, J. Wang, and S. Wang. SeamlessM4T - massively
  multilingual & multimodal machine translation, 2023b. URL https://arxiv.org/abs/2308.11596.

A. Shabalin, V. Meshchaninov, E. Chimbulatov, V. Lapikov, R. Kim, G. Bartosh, D. Molchanov, S. Markov,
  and D. Vetrov. Tencdm: Understanding the properties of diffusion model in the space of language model
  encodings. ArXiv, abs/2402.19097, 2024. URL https://arxiv.org/pdf/2402.19097.

N. Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. CoRR, abs/2010.02502, 2020. URL
   https://arxiv.org/abs/2010.02502.

N. Srivastava, D. Kuchelev, T. M. Ngoli, K. Shetty, M. Röder, D. Moussallem, H. Zahera, and A.-C. N. Ngomo.
  Lola–an open-source massively multilingual large language model. arXiv preprint arXiv:2409.11272, 2024.

S. Sturua, I. Mohr, M. K. Akram, M. Günther, B. Wang, M. Krimmel, F. Wang, G. Mastrapas, A. Koukounas,
   N. Wang, and H. Xiao. jina-embeddings-v3: Multilingual embeddings with task lora, 2024. URL https:
   //arxiv.org/abs/2409.10173.

H. Su, W. Shi, J. Kasai, Y. Wang, Y. Hu, M. Ostendorf, W.-t. Yih, N. A. Smith, L. Zettlemoyer, and T. Yu.
  One embedder, any task: Instruction-finetuned text embeddings. arXiv preprint arXiv:2212.09741, 2022.

J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu. Roformer: Enhanced transformer with rotary position
   embedding. Neurocomputing, 568:127063, 2024.

X. Sun, Z. Sun, Y. Meng, J. Li, and C. Fan. Summarize, outline, and elaborate: Long-text generation via
  hierarchical supervision from extractive summaries. In Proceedings of the 29th International Conference
  on Computational Linguistics, pages 6392–6402, Gyeongju, Republic of Korea, Oct. 2022. International
  Committee on Computational Linguistics. URL https://aclanthology.org/2022.coling-1.556.

Team GLM, A. Zeng, B. Xu, B. Wang, C. Zhang, D. Yin, D. Zhang, D. Rojas, G. Feng, H. Zhao, H. Lai,
  H. Yu, H. Wang, J. Sun, J. Zhang, J. Cheng, J. Gui, J. Tang, J. Zhang, J. Sun, J. Li, L. Zhao, L. Wu,
  L. Zhong, M. Liu, M. Huang, P. Zhang, Q. Zheng, R. Lu, S. Duan, S. Zhang, S. Cao, S. Yang, W. L. Tam,
  W. Zhao, X. Liu, X. Xia, X. Zhang, X. Gu, X. Lv, X. Liu, X. Liu, X. Yang, X. Song, X. Zhang, Y. An,
  Y. Xu, Y. Niu, Y. Yang, Y. Li, Y. Bai, Y. Dong, Z. Qi, Z. Wang, Z. Yang, Z. Du, Z. Hou, and Z. Wang.
  ChatGLM: A family of large language models from glm-130b to glm-4 all tools. ArXiv, abs/2406.12793,
  2024. URL https://arxiv.org/pdf/2406.12793.

The Llama3 team. The Llama 3 herd of models. ArXiv, abs/2407.21783, 2024. URL https://arxiv.org/pdf/
  2407.21783.

H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava,
  S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu,
  B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez,
  M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu,
  Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta,
  K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams,
  J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic,
  S. Edunov, and T. Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.


                                                     46
T. Ubukata, J. Li, and K. Tei. Diffusion model for planning: A systematic literature review. ArXiv,
  abs/2408.10266, 2024. URL https://arxiv.org/pdf/2408.10266.
A. Üstün, V. Aryabumi, Z.-X. Yong, W.-Y. Ko, D. D’souza, G. Onilude, N. Bhandari, S. Singh, H.-L. Ooi,
  A. Kayid, et al. Aya model: An instruction finetuned open-access multilingual language model. arXiv
  preprint arXiv:2402.07827, 2024.
C. Wang, S. Chen, Y. Wu, Z. Zhang, L. Zhou, S. Liu, Z. Chen, Y. Liu, H. Wang, J. Li, L. He, S. Zhao, and
  F. Wei. Neural codec language models are zero-shot text to speech synthesizers. CoRR, abs/2301.02111,
  2023. URL https://doi.org/10.48550/arXiv.2301.02111.
L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, and F. Wei. Improving text embeddings with large
   language models. In ACL, pages 11897–11916, 2024a. URL https://aclanthology.org/2024.acl-long.642.
L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, and F. Wei. Multilingual e5 text embeddings: A
  technical report, 2024b. URL https://arxiv.org/abs/2402.05672.
A. Warstadt, A. Singh, and S. R. Bowman. Neural network acceptability judgments. Transactions of
  the Association for Computational Linguistics, 7:625–641, 2019. doi: 10.1162/tacl_a_00290. URL
  https://aclanthology.org/Q19-1040.
S. Welleck, I. Kulikov, S. Roller, E. Dinan, K. Cho, and J. Weston. Neural text generation with unlikelihood
   training. arXiv preprint arXiv:1908.04319, 2019.
Y. Yang, G. H. Abrego, S. Yuan, M. Guo, Q. Shen, D. Cer, Y.-H. Sung, B. Strope, and R. Kurzweil. Improving
  multilingual sentence embedding using bi-directional dual encoder with additive margin softmax. arXiv
  preprint arXiv:1902.08564, 2019.
J. Ye, J. Gao, S. Gong, L. Zheng, X. Jiang, Z. Li, and L. Kong. Beyond autoregression: Discrete diffusion for
   complex reasoning and planning. ArXiv, abs/2410.14157, 2024. URL https://arxiv.org/pdf/2410.14157.
Y. Yin, J. Ding, K. Song, and Y. Zhang. Semformer: Transformer language models with semantic planning.
  ArXiv, abs/2409.11143, 2024. URL https://arxiv.org/pdf/2409.11143.
N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi. Soundstream: An end-to-end neural
  audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30:495–507, 2021.
B. Zhang and R. Sennrich. Root mean square layer normalization. Advances in Neural Information Processing
  Systems, 32, 2019.
X. Zhang, Y. Zhang, D. Long, W. Xie, Z. Dai, J. Tang, H. Lin, B. Yang, P. Xie, F. Huang, M. Zhang, W. Li,
  and M. Zhang. mgte: Generalized long-context text representation and reranking models for multilingual
  text retrieval, 2024. URL https://arxiv.org/abs/2407.19669.
Y. Zhang, J. Gu, Z. Wu, S. Zhai, J. Susskind, and N. Jaitly. Planner: Generating diversified paragraph via
  latent language diffusion model. In NeurIPS, 2023.




                                                     47
A       Technical consideration for data preparation
Since our modeling approach uses a fixed encoder and a fixed document segmentation method, we
decided to use pre-computed SONAR embeddings instead of producing them on-the-fly for each
training run. This allows for faster iteration on the same data mix, trading expensive GPU compute
against storage capacity.
As we are storing sequences of SONAR embedding, which are fixed size tensors of 1024 floats, the
storage requirements become more demanding than storing the raw text. For one terra bytes of raw
text data we need to store between fifteen and twenty terra bytes of encoded data. Overall, this
trade-off in space vs compute reduces the GPU memory occupation and the compute load and lets
use iterate faster. Typically, with on single GPU we can produce around 300-400 SONAR sentence
embeddings per second whereas by loading precomputed data (potentially from remote storage) we
can load over 20 thousand embeddings per second per GPU (with around 15 CPU per GPU).
We store sequences of SONAR embeddings with 16 bits precision (FP16) in parquet datasets.
Embeddings remain aligned with the segmented texts and the parquet binary format and library
ecosystem is well suited for storing and loading efficiently such complex data structures. Parquet
also lets us store extra data (such as quality metrics for each sentences) and enables non-trivial last
mile data filtering and transformation.
For training the LCM, we processed around four billion documents, generating 310 billion sentences
with an average of 27 tokens per sentences for 88 characters length on average; totaling a bit more
than 889 terra-bytes of raw text.


B       Open Sourced Code
In the spirit of reproducibility, we release under an open source license the training, evaluation
and data processing code for the LCM. This is available at https://github.com/facebookresearch/
large_concept_model.
The training code is based on the Fairseq2 framework (Balioglu, 2023) that allowed us to build and
iterate over the different model architectures discussed above. While Fairseq2 shares the same name
as the popular fairseq toolchain, its API architecture is different. It is not a monolithic toolchain but
a set of modules that can be composed, this allows us to build different architectures side by side
and easily share training components.
We also release our evaluation framework so the evaluation tasks reported in 3.1 and comparison
between the LCM and other models can easily be reproduced. The evaluation framework provides a
clear abstraction between predictors, tasks and data loading, which again, makes it modular and lets
us describe a set of tasks to be evaluated. The evaluation framework can be run locally or distributed
over a SLURM cluster to run evaluations at scale.
Finally, we release an updated version of stopes16 to simplify large scale data pre-processing on a
SLURM cluster. This was used to run the sentence segmentation and SONAR encoding described
in section 2.2. The stopes data processing framework deals with scheduling and monitoring large
number of jobs on SLURM or to run everything locally for small scale jobs. It provides an API
compatible with ray.data17 that makes it easy to process large datasets in blocks and apply transform
  16
       https://github.com/facebookresearch/stopes
  17
       https://docs.ray.io/


                                                    48
function over it. This makes our code reusable outside of a SLURM cluster as it can also be used
with a ray.io cluster.


C   System prompt: Generation of Topic Descriptions
You are a topic description generator. Your job is to read an extract of text and then generate a
topic description. The extract may be well formed or not. The topic description you will write will
be at most one sentence in length, and use as few words as possible. However, it can not be generic
and it can not contain any profanity.
Here is an example of an extract, an ideal topic description, and some examples of bad topic
descriptions:
Example extract: “One day, one neighborhood of the city was completely devasted. Glass windows
were shattered, shops turned upside down, and many civilian killed. Superman instantly recognized
the signature of one of his old enemies, Voltar, who he had barely beaten in the past. This was a
message to him: "I challenge you! Come find me!""
An example of a good topic description: An old enemy of Superman’s, Voltar, appeared and challenged
him.
An example of a bad topic description: Superman
An example of a bad topic description: Voltar
An example of a bad topic description:


D   User prompt: LLM As a Judge - Coherence
Below is a text extract. Your task is to analyze the extract and assign a coherence score between 0
and 5 inclusive, where:
0: The text is completely incoherent and lacks any logical connection.
1: The text has some minor connections, but overall it is disjointed and hard to follow.
2: The text has some coherence, but it is still difficult to understand due to unclear relationships
between ideas.
3: The text is moderately coherent, with some clear connections between ideas, but may lack depth
or clarity.
4: The text is highly coherent, with clear and logical connections between ideas, making it easy to
follow.
5: The text is extremely coherent, with a clear and concise structure, making it effortless to
understand.
You will provide a score ONLY. Do NOT also provide an explanation.
The extract: <extract>
After examining the extract, the coherence score between 0 and 5 inclusive is:




                                                49

