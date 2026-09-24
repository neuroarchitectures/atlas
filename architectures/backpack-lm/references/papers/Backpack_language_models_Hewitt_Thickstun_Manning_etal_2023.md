# Backpack language models Hewitt Thickstun Manning etal 2023

> Source: `Backpack_language_models_Hewitt_Thickstun_Manning_etal_2023.pdf`

---

                                                                             Backpack Language Models

                                                   John Hewitt John Thickstun Christopher D. Manning Percy Liang
                                                             Department of Computer Science, Stanford University
                                                         {johnhew,jthickstun,manning,pliang}@cs.stanford.edu




                                                                                                                             ∑
                                                               Abstract
                                                                                                        hot                         * = hot
                                             We present Backpacks: a new neural architec-
                                             ture that marries strong modeling performance




arXiv:2305.16765v1 [cs.CL] 26 May 2023
                                             with an interface for interpretability and con-
                                             trol. Backpacks learn multiple non-contextual          Transformer                           Transformer
                                             sense vectors for each word in a vocabulary,
                                             and represent a word in a sequence as a context-
                                             dependent, non-negative linear combination of        The tea is         The tea is
                                             sense vectors in this sequence. We find that,
                                                                                                  Transformer LM                Backpack LM
                                             after training, sense vectors specialize, each
                                             encoding a different aspect of a word. We
                                                                                                 Figure 1: Transformers are monolithic functions of se-
                                             can interpret a sense vector by inspecting its
                                                                                                 quences. In Backpacks, the output is a weighted sum of
                                             (non-contextual, linear) projection onto the out-
                                                                                                 non-contextual, learned word aspects.
                                             put space, and intervene on these interpretable
                                             hooks to change the model’s behavior in pre-
                                             dictable ways. We train a 170M-parameter
                                                                                                    Such interventions are difficult in Transformer
                                             Backpack language model on OpenWebText,
                                             matching the loss of a GPT-2 small (124M-           models (Vaswani et al., 2017) because their con-
                                             parameter) Transformer. On lexical similar-         textual representations are monolithic functions of
                                             ity evaluations, we find that Backpack sense        their input. Almost any intervention on the model
                                             vectors outperform even a 6B-parameter Trans-       has complex, non-linear effects that depend on con-
                                             former LM’s word embeddings. Finally, we            text. We would instead like models that enable
                                             present simple algorithms that intervene on         precise, rich interventions that apply predictably in
                                             sense vectors to perform controllable text gen-
                                                                                                 all contexts, and are still expressive, so they are a
                                             eration and debiasing. For example, we can
                                             edit the sense vocabulary to tend more towards
                                                                                                 viable alternative to Transformers.
                                             a topic, or localize a source of gender bias to a      We address these challenges with a new neu-
                                             sense vector and globally suppress that sense.      ral architecture, the Backpack, for which predic-
                                                                                                 tions are log-linear combinations of non-contextual
                                         1   Introduction                                        representations. We represent each word in a vo-
                                                                                                 cabulary as a set of non-contextual sense vectors
                                         Consider the prefix The CEO believes that ___, and      that represent distinct learned aspects of the word.
                                         the problem of debiasing a neural language model’s      For example, sense vectors for the word “science”
                                         distribution over he/she. Intuitively, the bias for     could encode types of science, connections to tech-
                                         he originates in the word CEO, because replacing        nology, notions of science being “settled,” or differ-
                                         CEO with nurse flips the observed bias. A success-      ent aspects of the scientific process (replication or
                                         ful intervention to debias CEO must reliably apply      experiment) (Table 1). Sense vectors do not learn
                                         in all contexts in which the word CEO appears;          classic word sense, but more general aspects of a
                                         ideally we would want to make a non-contextual          word’s potential roles in different contexts; in fact,
                                         change to the model that has predictable effects        they can be seen as a multi-vector generalization
                                         in all contexts. In general, in all aspects of in-      of classic word vectors (Mikolov et al., 2013).1
                                         terpretability and control, it is desirable to make
                                         interventions with a tractable interface (e.g., non-       1
                                                                                                      Our code, sense vectors, language model weights, and
                                         contextual representations) that apply globally.        demos are available at https://backpackmodels.science.
                A few senses of the word science                         MacBookHP = MacBook − Apple + HP
   Sense 3     Sense 7      Sense 9     Sense 10     Sense 8             The MacBook is best known for its form fac-
    fiction   replication    religion    settled     clones              tor, but HP has continued with its Linux-based
  fictional     citation      rology       sett    experiments           computing strategy. HP introduced the Hyper 212
   Fiction      Hubble         hydra      settle      mage               in 2014 and has continued to push soon-to-be-
   literacy     reprodu     religions    unsett    experiment            released 32-inch machines with Intel’s Skylake
    denial    Discovery         nec        Sett        rats              processors.


Table 1: Examples of the rich specialization of sense vectors representing the word science, and an example of
editing sense vectors non-contextually (changing MacBook to be associated with HP) and having the resulting
contextual predictions change.


   To make interventions on sense vectors behave                 2     The Backpack Architecture
predictably in different contexts, a Backpack rep-
                                                                 In this section, we define the general form of the
resents each word in a sequence as a linear com-
                                                                 Backpack architecture. We then show how contin-
bination of the sense vectors for all words in the
                                                                 uous bag-of-words word2vec (CBOW) (Mikolov
sequence. The expressivity of a Backpack comes
                                                                 et al., 2013) and Self-Attention-Only networks (El-
from the network that computes the weights of the
                                                                 hage et al., 2021; Olsson et al., 2022) are special
linear combination as a function of the whole se-
                                                                 cases of Backpacks.
quence; for example, in all our experiments we
use a Transformer for this. Since sense vectors are              2.1    Backpack General Form
softly selected depending on the context, they can
                                                                 A Backpack is a parametric function that maps
specialize; each sense can learn to be predictively
                                                                 a sequence of symbols x1:n = (x1 , . . . , xn ) to a
useful in only some contexts. The log-linear con-
                                                                 sequence of vectors o1:n = (o1 , . . . , on ), where
tribution of senses to predictions then implies that
                                                                 each symbol xi belongs to a finite vocabulary V and
the interventions on sense vectors we demonstrate
                                                                 oi ∈ Rd . We call oi the Backpack representation
in Section 6 apply identically (up to a non-negative
                                                                 of xi in the context of a sequence x1:n .
scalar weight) regardless of context.
   Our experiments demonstrate the expressivity of               Sense vectors. For each x ∈ V, a Backpack con-
Backpack language models, and the promise of in-                 structs k sense vectors
terventions on sense vectors for interpretability and
                                                                                  C(x)1 , . . . , C(x)k ,              (1)
control. In Section 4 we train Backpack language
models on 50B tokens (5 epochs) of OpenWebText;
                                                                 where C : V → Rk×d . Sense vectors are a multi-
a Backpack with 124M parameters in the contex-
                                                                 vector analog to classic non-contextual word repre-
tual network (and 46M parameters for sense vec-
                                                                 sentations like word2vec or GloVe: we make this
tors) achieves the perplexity of a 124M-parameter
                                                                 analogy precise in Section 2.2.
Transformer; thus one pays for more interpretabil-
ity with a larger model size. In Section 5, we show              Weighted sum. For a sequence x1:n , the repre-
that sense vectors specialize to encode rich notions             sentation oi of element xi is a weighted sum of the
of word meaning. Quantitatively, on four lexical                 predictive sense vectors for the words in its context:
similarity datasets (e.g., SimLex999), sense vectors             given contextualization weights α ∈ Rk×n×n ,
of a 170M parameter Backpack outperform word
                                                                                     n k
embeddings of the 6B-parameter GPT-J-6B Trans-                                       X X
former, and approach the performance of state-of-                             oi =             αℓij C(xj )ℓ .          (2)
                                                                                     j=1 ℓ=1
the-art specialized methods for this task. Finally, in
Section 6 we show that sense vectors offer a control             The contextualization weights αℓij of a Backpack
mechanism for Backpack language models. For ex-                  are themselves defined by a (non-linear) contextu-
ample, stereotypically gendered profession words                 alization function of the entire sequence x1:n :
(e.g., “CEO” or “nurse”) tend to learn a sense vec-
tor associated with this gender bias; by downscal-                                   α = A(x1:n ),                     (3)
ing this sense vector, we greatly reduce disparity in
contextual predictions in a limited setting.                     where A : V n → Rk×n×n .
   The name “Backpack” is inspired by the fact that             center word:
a backpack is like a bag—but more orderly. Like a                                 n
bag-of-words, a Backpack representation is a sum
                                                                                  X 1
                                                                         v xc =             vxi ,                 (5)
of non-contextual senses; but a Backpack is more                                        n
                                                                                  i=1
orderly, because the weights in this sum depend on                       p(xc | x1:n ) = softmax(U vxc ),         (6)
the ordered sequence.
                                                                where U ∈ RV×d . We see that vxc is a Backpack
Backpack Models. A Backpack model is a prob-
                                                                representation by setting C(x) = vx ∈ R1×d in
abilistic model that defines probabilities over some
                                                                Equation (1) using a single sense vector (k = 1)
output space Y as a log-linear function of a Back-
                                                                and setting the contextualization weights in Equa-
pack representation o1:n ∈ Rn×d :
                                                                tion (3) to be uniform: αℓij = n1 .
                                                                   This connection to CBoW foreshadows the emer-
          p(y|o1:n ) = softmax (E(o1:n )) ,              (4)
                                                                gence of linguistic structures in the predictive sense
                                                                vectors of Backpack models, just as these structures
where y ∈ Y and E : Rn×d → R|Y| is a linear
                                                                emerge in CBoW (Mikolov et al., 2013).
transformation. Because Backpack models are log-
linear in their representations, the sense vectors              2.3 Single-Layer Self-Attention is a Backpack
contribute log-linearly to predictions. This allows
                                                                The Backpack structure—define sense vectors (val-
us to inspect a sense vector by projecting it onto
                                                                ues), and use the sequence to determine how to
the vocabulary via E and observe exactly how it
                                                                sum them (weights)—may remind the reader of a
will contribute to predictions in any context.
                                                                single layer of self-attention. The key-query-value
   Models parameterized by the prevailing deep
                                                                self-attention function is as follows:
neural architectures—including LSTMs (Hochre-
iter and Schmidhuber, 1997) and Transformers—                                   n X
                                                                                X k
are not Backpacks because their output represen-                         oj =               αℓij OV (ℓ) xj        (7)
tations are (relatively) unconstrained functions of                             i=1 ℓ=1
the entire sequence. By contrast, Backpack models                        αℓ = softmax(x⊤ K (ℓ)⊤ Q(ℓ) x),          (8)
may seem limited in expressivity: the representa-
tions oi are scalar-weighted sums of non-contextual             where x ∈ Rn×d is (overloaded) to be a non-
vectors C(xj )ℓ . Contextual relationships between              contextual embedding of the sequence, O ∈
sequence elements can only be expressed through                 Rd×d/k , and V (ℓ) ∈ Rd/k×d , where k is the number
the weights α = A(x1:n ). Nevertheless, our exper-              of attention heads. The self-attention function is a
iments show that an expressive contextualization                Backpack with C(xj )ℓ = OV (ℓ) xj . Self-attention-
weight network can represent complex functions                  only networks are studied in the context of, e.g.,
by weighted sums of sense vectors, e.g., our 170M               mechanistic interpretability (Elhage et al., 2021).
parameter Backpack LM uses a 124M-parameter                     A Transformer composes blocks of self-attention
Transformer to compute α, and achieves the loss                 and non-linear feed-forward layers that combine
of a 124M-parameter Transformer LM.                             information from the whole sequence; unlike a
   To place Backpacks in some historical context,               Transformer, the contextualization weights of a
we now show how two existing architectures can                  Backpack each select a non-contextual sense of a
be described as Backpacks.                                      single word.

2.2   Continuous Bag-of-Words is a Backpack                     3   Language Modeling with Backpacks
The continuous bag-of-words word2vec model de-                  In this section, we define a neural autoregressive
fines a probability distribution over a center word             language model parameterized by a Backpack. We
xc ∈ V conditioned on n context words x1:n .2 The               use the standard softmax parameterization of the
model proceeds to (1) construct vector embeddings               probability over the next token in a sequence, with
vx for each x ∈ V, and (2) uniformly average the                a weight matrix E ∈ Rd×|V| that maps a represen-
embeddings of the context words to predict the                  tation oj ∈ Rd to logits E ⊤ oj ∈ R|V| :
   2
     Context in this setting is usually defined as words sur-
rounding the center word.                                               p(xj | x1:j−1 ) = softmax(E ⊤ oj ).       (9)
Recall (Section 2.1) that Backpack representations               (Section 4.2), evaluations (Section 4.3) and results
oj are defined by sense vectors C(x) and contextu-               (Section 4.4). We also show the necessity of learn-
alization weights αj . In Section 3.1 we describe a              ing k > 1 sense vectors to achieve strong language
parameterization of C for the predictive sense vec-              modeling performance (Section 4.5).
tors in Equation (1), and in Section 3.2 we describe
a parameterization of A for the contextualization                4.1    Models
weight network in Equation (3). When oj is pa-                   We train three Transformer baseline models, which
rameterized by a Backpack, we call a model of the                we label Micro (30M parameters), Mini (70M pa-
form given by Equation (9) a Backpack LM.                        rameters), and Small (124M parameters; the same
                                                                 size as GPT-2 small). We also train Micro (40M),
3.1      Parameterizing senses                                   Mini (100M), and Small (170M) Backpack lan-
For the sense function C : V → Rk×d , we embed                   guage models, for which the weighting function
each x ∈ V into Rd and pass these embeddings                     (Equation 11) is parameterized using the corre-
though a feed-forward network FF : Rd → Rk×d :                   sponding Transformer, and almost all extra parame-
                                                                 ters are in the non-contextual sense vectors.4 Back-
                   C(x) = FF(Ex),                       (10)     packs thus cost extra parameters and compute be-
where the embedding/projection matrix E is tied to               yond their underlying contextualization network.
the output matrix in Equation (9) (Press and Wolf,               Except where stated, we use k = 16 sense vectors
2017). Note that we could define all k × |V| sense               in all Backpacks (Section A).
vectors using a lookup table, but this would be an                  We use a reduced sequence length of 512 for all
enormous number of parameters as k grows large.                  models, and the 50,257-subword GPT-2 tokenizer.
Instead, we embed the words as Ex ∈ Rd , and                     Model hidden dimensionalities, layer counts, and
then blow them up to Rd×k using shared weights.                  head counts are reported in Table 9.
This may explain the related sense roles observed                4.2    Data & Optimization
for different word types in Section 5.1.
                                                                 We train all models on OpenWebText (Gokaslan
3.2      Parameterizing contextualization weights                and Cohen, 2019), a publicly available approxi-
We parameterize A : V n → Rk×n×n using a stan-                   mate reconstruction of the English WebText corpus
dard Transformer, followed by a layer of multi-                  used to train the GPT-2 family of models (Rad-
headed key-query self-attention. That is, we pass                ford et al., 2019). We use a batch size of 524,288
an embedded sequence through a Transformer                       tokens, and train all models for 100,000 gradient
                                                                 steps for a total of 52B tokens; training for longer
             h1:n = Transformer(Ex1:n )                 (11)     is known to make marginal difference for small
                                                                 models (Hoffmann et al., 2022). The size of Open-
(with proper autoregressive masking and some po-                 WebText means this is roughly 5 epochs. We use
sition representation) and compute A(x1:n ) = α,                 cross-entropy loss and the AdamW optimizer, with
where                                                            a warmup of 5,000 steps and linear decay to zero.
        αℓ = softmax(h1:n K (ℓ)⊤ Q(ℓ) h⊤
                                       1:n ),           (12)     4.3    Evaluations
for each predictive sense ℓ = 1, . . . , k with matri-           Before our experiments in interpretability and con-
ces K (ℓ) , Q(ℓ) ∈ Rd×d/k . We can think of the k                trol, we check the expressivity of Backpacks. We
senses as heads and, for each head, the contextu-                evaluate models on perplexity for a held out set
alization weights define a distribution of attention             of OpenWebText, perplexity and accuracy for the
over words.3                                                     (OpenAI variant of) LAMBADA evaluation of
                                                                 long-distance dependencies (Radford et al., 2019;
4       Experiments Training Backpack LMs                        Paperno et al., 2016), perplexity on Wikitext (Mer-
In this section we specify the hyperparameters used              ity et al., 2017), and BLiMP English linguistic com-
to train Backpack and Transformer language mod-                  petence accuracy (Warstadt et al., 2020) evaluated
els (Section 4.1), data and optimization procedure               using the EleutherAI harness (Gao et al., 2021)
    3
                                                                 (Version 1).
    Note that the sense weights are normalized (1) indepen-
                                                                     4
dently for each sense, and (2) to sum to one over the sequence         There are a negligible number of additional parameters in
length.                                                          the final key-query Backpack operation (Equation 12)).
    Model               OpenWebText PPL ↓   LAMBADA PPL ↓          LAMBADA ACC ↑       Wikitext PPL ↓   BLiMP ↑
    Backpack-Micro            31.5                 110                   24.7               71.5         75.6
    Transformer-Micro         34.4                 201                   21.3               79.5         77.8
    Backpack-Mini             23.5                 42.7                  31.6               49.0         76.2
    Transformer-Mini          24.5                 58.8                  29.7               52.8         80.4
    Backpack-Small            20.1                 26.5                  37.5               40.9         76.3
    Transformer-Small         20.2                 32.7                  34.9               42.2         81.9

Table 2: Language modeling performance; all models trained for 100k steps, 500K token batch size, on OWT. For
PPL, lower is better; for accuracy, higher is better. Note that models are not parameter-comparable; each Backpack
has a matched-size Transformer in its contextualization network.


4.4    Discussion                                          prediction. We interpret these roles by picking a
Comparing each Backpack LM to a Transformer                sense ℓ of a word x, and projecting this sense onto
LM of equivalent specification to the Backpack’s           the word embeddings: E ⊤ C(x)ℓ ∈ R|V| . Note
contextualization network, we see that the Back-           that this is exactly (up to a scalar) how this sense
pack performs roughly as well (Table 2). Again, the        contributes to any prediction of the model. We in-
Backpack has more parameters, a tax for the inter-         terpret a sense vector’s role by reporting the words
face provided by sense vectors. During training, we        with the highest score under this projection.
find that Backpack language models take longer to             Table 3 visualizes a few of these senses. For
converge than Transformers. Curiously, while the           example, sense 12 seems to encode a broad no-
Small Backpack and Transformer achieve almost              tion of relatedness for almost all words; sense 3
identical OWT perplexity, the Backpack language            encodes particulars of the bigram distribution given
models perform substantially better on LAMBADA             x; sense 14 seems to encode both associated objects
and Wikitext, but worse on BLiMP.                          for verbs, and noun modifier dependency children
                                                           for nouns. In Section 5.2 we show that sense 14
4.5    Effect of varying the number of senses              encodes a powerful notion of verb similarity.
To study the impact of the number of sense vec-            5.2     Lexical Relationship Tests
tors on language modeling performance, we train
Mini-sized Backpack language models on a re-               Classic lexical-relatedness and similarity tests mea-
duced schedule of 50,000 gradient steps, for k ∈           sure the extent to which a similarity function on
{1, 4, 16, 64} sense vectors. The perplexities for         pairs of words correlates with human-elicitied
k = 1, 4, 16, 64 are 38.6, 29.3, 26.0, and 24.1,           notions of similarity. Similarity functions de-
demonstrating the necessity of a non-singleton set         rived from word embeddings are evaluated by
of sense vectors. Table 8 contains the full results.       Spearman correlation between the predicted and
                                                           true similarity rank-order. Early non-contextual
5     Emergent Structure in Sense Vectors                  embeddings like COALS (Rohde et al., 2005),
                                                           word2vec (Mikolov et al., 2013), and GloVe (Pen-
Backpack language model sense vectors are not              nington et al., 2014) have recently been outper-
trained using a supervised notion of word sense,           formed by word embeddings derived by distilla-
but implicitly specialize to encode different shades       tion of contextual networks (Bommasani et al.,
of a word’s predictive use. In this section, we qual-      2020; Gupta and Jaggi, 2021; Chronis and Erk,
itatively examine sense vectors (Section 5.1) and          2020). We evaluate Backpack LM sense vec-
quantitatively demonstrate their effectiveness in          tors on similarity datasets SimLex999 (Hill et al.,
computing lexical similarity and relatedness (Sec-         2015), SimVerb3500 (Gerz et al., 2016), and re-
tion 5.2). Taken together, this suggests that sense        latedness datasets RG65 (Rubenstein and Goode-
vectors can provide a high-level interface for inter-      nough, 1965) and (Agirre et al., 2009).
vention, which we explore in Section 6.
                                                           Senseℓ Cosine. For all ℓ ∈ {1, . . . , k}, we define
5.1    Visualizing Senses                                  a similarity function based only on sense ℓ:
Empirically, trained Backpack models associate
specific sense vector indices with different roles for           Simℓ (x, x′ ) = cossim(C(x)ℓ , C(x′ )ℓ ),      (13)
                       Sense 12 (relatedness)                                   Sense 14 (Verb objects, nmod nouns)
             tasty      quickly         Apple        believe            build         attest       importance   appreciate
             tasty       quick           Apple      belief             bridges     worthiness     maintaining      finer
           culinary     quickest         Apple     Belief                wall      Published       wellbeing      nuance
            tasted       quick          iPhone     beliefs             lasting     superiority     teamwork       beauty
           delicious    quicker         iPhone    believing               ig        accuracy        plurality      irony
             taste        fast         iPhones     believe             rapport       validity     upholding     simplicity

                             Sense 3 (next wordpiece)                 Sense 7 (Proper Noun Associations)
                         pizza         interest      the              Apple       Obama           Messi
                         cutter          rate     slightest           macOS       Dreams          Messi
                        tracker         rates       same              iCloud      Barack         Argentina
                           iol         groups     entirety              Siri        Ob             Mess
                        makers         waivers       rest              iOS       Michelle        Barcelona
                        maker          waiver       latter               tv      Jeremiah          iesta

Table 3: Visualization of how the same sense index across many words encodes fine-grained notions of meaning,
relatedness, and predictive utility. Each sense is given a label thought up by the authors, and for a few words, the
target words that are highest scored by the sense vector.


  Model              SL999     SV3500        RG65      WS353             state-of-the art specialized methods using either
  Classic Non-Contextual Embeddings                                      a single vector per word (Gupta, 2021) or many
  word2vec         0.442    0.367            0.679      0.684            vectors (Chronis and Erk, 2020).
  GloVe            0.371    0.227            0.687      0.607
  Embeddings from large existing models                                  Discussion. Sense 12 (the “synonym” sense) per-
  GPT2-1.5B        0.523      0.418     0.670           0.706            forms well across datasets, matching or outperform-
  GPT-J-6B         0.492      0.374     0.766           0.673
                                                                         ing embeddings like GPT-2-1.5B and GPT-J-6B
  Embeddings from our models + baseline Transformer                      (Except GPT-J-6B on RG-65). Sense 14, the “verb
  Trnsf 124M      0.478     0.363     0.634     0.681
  Sim12 (ours)    0.522     0.471     0.754     0.749                    objects” sense, performs best on just verb similarity
  Sim14 (ours)    0.500     0.502     0.591     0.655                    (VerbSim3500), and the minimum similarity over
  Simmin (ours)   0.540     0.471     0.653     0.607
                                                                         senses works especially well on noun lexical sim-
  Special-purpose SOTA models                                            ilarity (SimLex999.) Our methods approach the
  SOTA (Single)    0.554    0.473            0.835      0.764
  SOTA (Multi)     0.605    0.528              -        0.807            performance of state-of-the-art methods; despite
                                                                         being trained for a very different task, sense vectors
Table 4: Results on lexical similarity evaluation. All                   encode substantial lexical information (Table 4).
numbers are Spearman correlations; higher is better.
                                                                         6      Sense Vectors for Control
where cossim is cosine similarity. Intuitively, we                       In this section, we demonstrate several proof-of-
expect that some senses may specialize to learn                          concept methods that leverage sense vectors for
lexical relatedness or similarity.                                       controlling LM behavior.
Minimum Sense Cosine. Because each sense                                 6.1     Topic-controlled generation
encodes a different aspect of a word’s meaning, we
might expect that highly similar words are similar                       Given a bag-of-words target b ∈ R|V| , e.g., arts,
across all senses. We test for this strong form of                       culture, we would like to bias generation towards
similarity using                                                         sequences related to concepts related to these terms.
                                                                         Our algorithm proceeds in three parts. First, we sort
          Simmin (x, x′ ) = min Simℓ (x, x′ )                  (14)      sense vectors by log-probability assigned to b, that
                                   ℓ                                     is, b⊤ (E ⊤ C(x)ℓ ).5 Second, based on the scores,
Other methods. We evaluate embeddings from                               we assign a re-weighting factor δ to each sense;
the tied softmax/embedding matrices of the much                          senses with the higher scores weighted more. (See
larger GPT-2-1.5B (Radford et al., 2019) and GPT-                        Section D for details.) Third, we generate from
J-6B (Wang and Komatsuzaki, 2021), classic word                             5
                                                                              We divide this term by the maximum absolute log-
embeddings (from Bommasani et al. (2020)) and                            probability of the sense vector, maxx∈V x⊤ (E ⊤ C(x)ℓ ).
                                                   Topic Control in Generation                           Model                Bias Ratio ↓   Reduction %




Overall MAUVE with OpenWebText
                                                                                                         Unbiased                  1              -
                                 0.9
                                                                                                         Transformer
                                 0.8                                                                     Unmodified              7.02             -
                                                  Unmodified Backpack                                    Project-Nullspace       6.72            5%
                                 0.7                                                                     Optimize-Nullspace      7.02            0%
                                       Unmodified Transformer                                            Backpack
                                 0.6                                                                     Unmodified              4.34             -
                                                                                                         Remove-Sense10          2.88           44%
                                 0.5                                                                     Optimize-Sense10        2.16           65%
                                            Transformer+PPLM
                                            Backpack+sense control
                                 0.4                                                               Table 5: Pronoun-based gender bias reduction in a
                                           0.10      0.15   0.20     0.25   0.30     0.35
                                                  17-Topic Average Control Success                 limited setting.

Figure 2: Results in controlling topic via sense inter-
vention in Backpacks, and PPLM in Transformers.
                                                                                                   6.2     Mitigating gender bias
                                                                                                   Through inspection, we learned that sense vector 10
                                                                                                   of many stereotypically gendered profession nouns
the Backpack using the re-weighted sense vectors,
                                                                                                   (nurse, CEO, teacher) coherently express the stereo-
reducing δ back to 1 as the topic is introduced. The
                                                                                                   type through pronouns. Table 13 gives examples of
updated backpack equation is
                                                                                                   these senses. We attempt to mitigate gender bias in
                                                   n X
                                                     k                                             Backpack behavior on these gendered profession
                                                   X
                                          oi =                αℓij δℓij C(xj )ℓ ,           (15)   nouns by turning down sense 10 (multiplying by a
                                                    j=1 ℓ=1                                        scalar less than 1).
                                                                                                      We took an existing set of stereotypically gen-
where δijℓ is the re-weighting. Intuitively, the se-                                               dered profession nouns from WinoBias (Zhao et al.,
mantic coherence of sense vectors may imply that                                                   2018), and constructed a simplified setting in which
upweighting senses with affinity to the target bag-                                                a single profession word is in each context, and a
of-words richly upweights related words and topics.                                                third-person nominative pronoun (e.g., he/she/they)
We give details as to how we perform the sense re-                                                 is acceptable, e.g., My CEO said that__. The full
weighting and the annealing in Section D.                                                          set of nouns and prompts is in Section D.2. We
                                                                                                   evaluate models on the average of the bias of prob-
Evaluation. We use the label descriptors of the
                                                                                                   abilities of him vs her as follows:
topic classifier of Antypas et al. (2022), with 17
categories (sports, arts & culture, health,. . . ), as                                                                
                                                                                                                         p(he | x) p(she | x)
                                                                                                                                                
the bag-of-words for control. We evaluate control                                                          E      max               ,              .
                                                                                                       x∈prompts         p(she | x) p(he | x)
accuracy as the percent of generations to which the
classifier assigns the correct topic label, and overall                                            Baseline. To debias a Transformer with an analo-
generation quality and diversity using MAUVE                                                       gous method, we take inspiration from Bolukbasi
scores (Pillutla et al., 2021).6                                                                   et al. (2016). We take Exhe − Exshe as an esti-
                                                                                                   mate of a gender bias direction, and project the
Results. We compare to Plug-and-Play Language
                                                                                                   embedding Exnurse either to the nullspace of this
Models (PPLM; Dathathri et al. (2019)), a consid-
                                                                                                   direction or only partially remove it.
erably slower, gradient-based control method using
our Small Transformer model. We generate 500                                                       Results. A perfectly unbiased model would
samples from each model for each topic across a                                                    achieve ratio 1, whereas the unmodified Trans-
range of strengths of control. We find that sense                                                  former achieves 7, and with nullspace projection,
controlled generation provides at least as strong                                                  6.72 (Table 5). Finding the optimal fraction of the
control as PPLM (Figure 2), though the MAUVE                                                       gender bias direction to remove per profession does
scores of the unmodified Transformer are higher                                                    not improve further. For Backpacks, we find that
than the Backpack.) Results and examples are pro-                                                  removing sense 10 from the profession word (set-
vided in the Appendix in Tables 12, 16, 17, 18.                                                    ting it to zero) reduces the bias score from 4.34 to
   6
     We concatenate generations across the 17 categories and                                       2.88. Learning the optimal removal fraction per
compute MAUVE against OpenWebText validation examples.                                             profession achieves 2.16, for a total reduction of
              0                                                    0.7                                               1
                                                      Weight on Sense 10




                                         P( x | When the nurse walked into the room, )

Figure 3: The effect on the conditional probability distribution of a Backpack LM on the prefix when the nurse
walked into the room, of modulating the effect of sense 10 of nurse from 0 (totally removed) to 1 (original.)


  The MacBook is best known for its form factor, but HP                    We project each sense vector of x to the
  has continued with its Linux-based computing strategy.                 nullspace of Exr , and then add in Exa :
  HP introduced the Hyper 212 in 2014 and has continued
  to push soon-to-be-released 32-inch machines with Intel’s
                                                                                               C(x)⊤
                                                                                                                          
  Skylake processors.                                                                              ℓ Exr          Exa
                                                                         C̃(x)ℓ = C(x)ℓ +                             − Exr ,
  The MacBook didn’t come into the picture until 2000,                                         ∥C(xr )ℓ ∥22        ϕ
  when HP followed up with a 15-year flood of HP available
  laptops.                                                                            ∥Ex ∥2
                                                                         where ϕ = ∥Exa∥22 is a normalization term to ac-
                                                                                            r 2
  I was thinking about Brady’s role on the Colts before
  joining other high-profile signings. This is what McEl-                count for the differing norms of Exa and Exr .
  haney and I discussed.                                                 Intuitively, this projection modifies each sense vec-
  McElhaney: Look, what I didn’t mean by this is we didn’t               tor in measure proportional to how much xr was
  move. We think that we’re getting a lot better, too.
                                                                         predicted by that sense. So, senses of MacBook
Table 6: Samples from a Backpack wherein Apple has                       that would added mass to Apple now add mass to
been projected out of the MacBook sense embeddings,                      HP; unrelated senses are not affected. In Table 6,
and replaced with HP. Likewise with Brady, Patriots,                     we show samples providing intuition for how Mac-
and Colts. Prompts are bolded.                                           Book evokes HP instead of Apple, but is otherwise
                                                                         semantically and syntactically maintained.
65%.7 In Figure 3, we demonstrate the clear effect                       7   Related Work
of ablating sense 10 on the most likely words in
one of these contexts.8                                                  Representation learning in NLP. Learning prob-
                                                                         abilistic models of text for use in representation
6.3    Knowledge editing                                                 learning and identifying resulting structure has a
Sense vectors show promise for use in knowledge                          long history in NLP, from non-contextual word
editing (De Cao et al., 2021)—editing a model’s                          vectors (Schütze, 1992; Rohde et al., 2005; Tur-
predictions about world knowledge. In particular,                        ney, 2010; Mikolov et al., 2013; Bojanowski et al.,
many associations with proper nouns can be local-                        2017) to contextual networks (Elman, 1990; Ben-
ized to sense vectors in that noun. In this qualitia-                    gio et al., 2000; Collobert and Weston, 2008;
tive proof-of-concept, we edit the sense vectors of                      Sutskever et al., 2011; Peters et al., 2018; Rad-
a target word x (e.g., MacBook to remove associa-                        ford et al., 2018). Deep Averaging Networks (Iyyer
tions with a word xr (e.g., Apple) and replace those                     et al., 2015) are not Backpacks; they first perform
associations with another word xa (e.g., HP). Intu-                      averaging and then nonlinear computation.
itively, this intervention ensures that whenever the
                                                                         Interpretability for Control of NLP networks.
contextualization weights would point to a sense
                                                                         A burgeoning body of work attempts to intervene
vector in MacBook to predict words associated with
                                                                         on monolithic neural networks for interpretabil-
Apple, it now predicts words associated with HP.
                                                                         ity and control (Meng et al., 2022, 2023), and for
   7
      Curiously, Backpacks are overall less biased to begin with         mechanistic understanding (Olsen et al., 2021; El-
(in this setting); we don’t have a strong hypothesis as to why.          hage et al., 2021). Implicitly, Backpacks develop
    8
      It is incidental that sense 10 encodes gender bias as op-
posed to another sense index; the consistency in index across            a somewhat human-understandable language of
words may be due to parameter sharing in C.                              machine concepts, an idea espoused in Kim et al.
(2018); Koh et al. (2020). The connections between       building blocks exist in the prefix, a contextualiza-
interpretation and control are rich; much work has       tion network that recognizes the negation or other
gone into the detection and extraction of emergent       property could properly distribute weights.
structure in networks (Hupkes et al., 2018; Liu
et al., 2019) as well as subsequently modulating         Are Backpacks inherently interpretable? No,
behavior (Lakretz et al., 2019; Eisape et al., 2022).    but we believe no architecture is. Each architecture
                                                         provides a set of tools that may or may not be useful
Generalized Additive Models. Generalized Ad-             for differing goals. To us, the key is the mechanis-
ditive Models (GAMs; Hastie and Tibshirani               tic guarantees Backpacks offer, which will vary
(1986)) are a function family that (1) independently     in utility depending on how well-specialized the
transforms each input feature, (2) sums these trans-     learned sense vectors are for a specific kind of con-
formations of inputs and (3) applies a non-linear        trol. Also, the visualizations we provide (top-k
link function (e.g., softmax):                           highest-scored words) only provide a small view
                                                         into a sense’s potential uses, because scores are
    f (x1:n ) = Φ (r1 (xi ) + · · · + rn (xn ))   (16)   non-zero for the whole vocabulary.

                                                         Are Backpacks as compute-efficient as Trans-
Treating each word-position pair as a feature, Back-
                                                         formers? At a glance, no. Backpacks have an
packs are not GAMs because they include a weight-
                                                         underlying Transformer as well as extra parame-
ing α that depends on all features. However, Back-
                                                         ters, but may perform roughly as well as just the
packs share an intuition of computing independent
                                                         underlying Transformer. However, sense vectors
representations of each feature and aggregating by
                                                         are sparsely activated—only those from the rele-
addition. Neural GAMs have been proposed for
                                                         vant sequence need be on GPU—and after training,
interpretability (Agarwal et al., 2021; Yang et al.,
                                                         can be computed by lookup.
2021; Chang et al., 2022; Radenovic et al., 2022;
Dubey et al., 2022), though never to our knowl-          Why do sense vectors specialize? Ablations in
edge in language modeling. We expect that without        Table 8 show that they should at least learn to be
context-dependent weighting, models would be in-         linearly independent, since linear dependence is
sufficiently expressive for language modeling.           equivalent to having having fewer sense vectors,
                                                         which causes higher perplexity. The specialization
8   Discussion                                           of sense vectors to seemingly coherent categories
                                                         may be attributable to the shared feed-forward net-
In this section, we address a few natural questions
                                                         work that computes them, and/or the contextual-
about the expressivity and interpretability of Back-
                                                         ization network learning to assign similar weight
packs, highlighting the limits of our knowledge.
                                                         distributions to senses with similar roles.
How do Backpacks compare to architecture X?              Are sense vectors like “word senses?” No; they
The Backpack structure does not depend upon us-          encode a notion of “predictive utility” that doesn’t
ing a Transformer to compute the contextualization       align with traditional notions of word sense. We
weights. We could parameterize the contextual-           use the name “sense vector” however because they
ization function with a different architecture (e.g.,    form a new, useful notion of decomposition of the
LSTM, S4 (Gu et al., 2021)) and use the resulting        possible contextual uses of a word into components
weights to compute the Backpack sense vector sum.        that are softly combined in each context.
This architecture, e.g., the Backpack-S4, could then
be compared to the standard S4 architecture.             9   Conclusion
Are Backpacks as expressive as Transformers?             Non-contextual word2vec embeddings initiated
We don’t know. If the number of linearly inde-           modern deep learning research in NLP, and have
pendent sense vectors is at least d, then a suffi-       fascinating geometric structure. Now, research has
ciently complex contextualization network could          largely moved on to monolithic representations,
treat them as an arbitrary basis. A concern we’ve        first from RNNs and now from Transformers. Our
often heard is that “simply” adding together sense       work suggests that we can have both rich lexical
vectors should not be expressive enough to handle,       structure and interventions, and strong contextual
e.g., negation. However, as long as the requisite        performance, in a single model.
10    Acknowledgements                                    in the right direction. In particular, explanations
                                                          based on the structure of Backpacks may be able to
The authors would like to thank Amita Kamath,             provide insights into the mechanisms behind model
Steven Cao, Xiang Lisa Li, Ian Covert, and the            behaviors, increasing transparency.
Stanford NLP Group community for useful discus-              The concrete models we will release, up to
sions. Further support came from the Stanford Cen-        and including 170M parameters, are substantially
ter for Research on Foundation Models. Christo-           smaller and less performant at generating text than
pher Manning is a CIFAR Fellow. John Hewitt               many of the publicly and commercially available
was supported by an NSF Graduate Research Fel-            language models available right now, so we do not
lowship under grant number DGE-1656518 and                expect there to be considerable negative repercus-
by the CIFAR Learning in Machines and Brains              sions from the release of the artifacts. The code
program. We gratefully acknowledge the support            we release, however, could be used or replicated to
of a PECASE Award to Percy Liang.                         train much larger Backpack LMs by corporations
                                                          or governments.
11    Limitations
There is a fundamental uncertainty in whether
                                                          References
Backpack language models will continue to scale
with parameters and data and be viable alternatives       Rishabh Agarwal, Levi Melnick, Nicholas Frosst,
                                                            Xuezhou Zhang, Ben Lengerich, Rich Caruana, and
to Transformers at larger model scales. In this             Geoffrey E Hinton. 2021. Neural additive models:
study, we were unable to scale larger, and hope             Interpretable machine learning with neural nets. Ad-
that future work will test larger model scales. In a        vances in Neural Information Processing Systems,
similar vein, we do not verify that Backpack lan-           34:4699–4711.
guage models perform well across multiple lan-            Eneko Agirre, Enrique Alfonseca, Keith Hall, Jana
guages. We also do not consider, e.g., finetun-             Kravalova, Marius Paşca, and Aitor Soroa. 2009. A
ing Backpacks on other tasks, or masked language            study on similarity and relatedness using distribu-
modeling—there is a wide range of possible uses             tional and WordNet-based approaches. In Proceed-
                                                            ings of Human Language Technologies: The 2009
that remain to be verified.                                 Annual Conference of the North American Chapter of
   One potential obstacle to the use of Backpacks           the Association for Computational Linguistics, pages
that we do not study is the effect of tokenization in       19–27, Boulder, Colorado. Association for Computa-
languages with richer morphological structure than          tional Linguistics.
English—will the Backpack structure be amenable           Dimosthenis Antypas, Asahi Ushio, Jose Camacho-
to modeling those languages? This may be difficult          Collados, Vitor Silva, Leonardo Neves, and
because, intuitively, the interpretability and control      Francesco Barbieri. 2022. Twitter topic classifica-
                                                            tion. In Proceedings of the 29th International Con-
of Backpacks relates to the semantics of individ-           ference on Computational Linguistics, pages 3386–
ual tokens. Even in English, small subwords not             3400, Gyeongju, Republic of Korea. International
indicative of a single word are hard to interpret.          Committee on Computational Linguistics.
What we hope to have provided is a sufficient set
                                                          Yoshua Bengio, Réjean Ducharme, and Pascal Vincent.
of experiments to motivate the further exploration          2000. A neural probabilistic language model. Ad-
of Backpacks.                                               vances in neural information processing systems, 13.

12    Ethics                                              Piotr Bojanowski, Edouard Grave, Armand Joulin, and
                                                            Tomas Mikolov. 2017. Enriching word vectors with
                                                             subword information. Transactions of the Associa-
This paper describes and releases an open-domain
                                                             tion for Computational Linguistics, 5:135–146.
language model trained on a largely unfiltered sub-
section of the (mostly English portions of the) tex-      Tolga Bolukbasi, Kai-Wei Chang, James Y Zou,
tual internet, and describes methods for interpreting       Venkatesh Saligrama, and Adam T Kalai. 2016. Man
                                                            is to computer programmer as woman is to home-
and controlling said model. Any control method              maker? Debiasing word embeddings. Advances in
that can be used to help understand and guide the           neural information processing systems, 29.
generation of a model can be used to more effec-
                                                          Rishi Bommasani, Kelly Davis, and Claire Cardie. 2020.
tively generate toxic or illegal content. Despite this,     Interpreting Pretrained Contextualized Representa-
we do expect that, overall, the benefit of deeper           tions via Reductions to Static Embeddings. In Pro-
insight into Backpack language models is a step             ceedings of the 58th Annual Meeting of the Asso-
  ciation for Computational Linguistics, pages 4758–       Leo Gao, Jonathan Tow, Stella Biderman, Sid Black,
  4781, Online. Association for Computational Lin-           Anthony DiPofi, Charles Foster, Laurence Golding,
  guistics.                                                  Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff,
                                                             Jason Phang, Laria Reynolds, Eric Tang, Anish Thite,
Chun-Hao Chang, Rich Caruana, and Anna Goldenberg.           Ben Wang, Kevin Wang, and Andy Zou. 2021. A
  2022. NODE-GAM: Neural generalized additive                framework for few-shot language model evaluation.
  model for interpretable deep learning. In Interna-
  tional Conference on Learning Representations.           Daniela Gerz, Ivan Vulić, Felix Hill, Roi Reichart, and
                                                             Anna Korhonen. 2016. SimVerb-3500: A large-scale
Gabriella Chronis and Katrin Erk. 2020. When is a            evaluation set of verb similarity. In Proceedings of
  bishop not like a rook? when it’s like a rabbi! Multi-     the 2016 Conference on Empirical Methods in Natu-
  prototype BERT embeddings for estimating semantic          ral Language Processing, pages 2173–2182, Austin,
  relationships. In Proceedings of the 24th Confer-          Texas. Association for Computational Linguistics.
  ence on Computational Natural Language Learning,
  pages 227–244.                                           Aaron Gokaslan and Vanya Cohen. 2019. Open-
                                                             webtext corpus. http://skylion007.github.io/
Ronan Collobert and Jason Weston. 2008. A unified            OpenWebTextCorpus.
  architecture for natural language processing: Deep       Albert Gu, Karan Goel, and Christopher Re. 2021. Ef-
  neural networks with multitask learning. In Proceed-       ficiently modeling long sequences with structured
  ings of the 25th international conference on Machine       state spaces. In International Conference on Learn-
  learning, pages 160–167.                                   ing Representations.
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra,          Prakhar Gupta and Martin Jaggi. 2021. Obtaining better
   and Christopher Ré. 2022. FlashAttention: Fast and         static word embeddings using contextual embedding
   memory-efficient exact attention with IO-awareness.        models. In Proceedings of the 59th Annual Meet-
   In Advances in Neural Information Processing Sys-          ing of the Association for Computational Linguistics
   tems.                                                      and the 11th International Joint Conference on Natu-
                                                              ral Language Processing (Volume 1: Long Papers),
Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane           pages 5241–5253.
  Hung, Eric Frank, Piero Molino, Jason Yosinski, and
  Rosanne Liu. 2019. Plug and play language models:        Vikram Gupta. 2021. Multilingual and multilabel emo-
  A simple approach to controlled text generation. In        tion recognition using virtual adversarial training.
  International Conference on Learning Representa-           In Proceedings of the 1st Workshop on Multilingual
  tions.                                                     Representation Learning, pages 74–85, Punta Cana,
                                                             Dominican Republic. Association for Computational
Nicola De Cao, Wilker Aziz, and Ivan Titov. 2021. Edit-      Linguistics.
  ing factual knowledge in language models. In Pro-
  ceedings of the 2021 Conference on Empirical Meth-       Charles R. Harris, K. Jarrod Millman, Stéfan J. van der
  ods in Natural Language Processing, pages 6491–            Walt, Ralf Gommers, Pauli Virtanen, David Cour-
  6506.                                                      napeau, Eric Wieser, Julian Taylor, Sebastian Berg,
                                                             Nathaniel J. Smith, Robert Kern, Matti Picus,
Abhimanyu Dubey, Filip Radenovic, and Dhruv Maha-            Stephan Hoyer, Marten H. van Kerkwijk, Matthew
  jan. 2022. Scalable interpretability via polynomials.      Brett, Allan Haldane, Jaime Fernández del Río, Mark
  In Advances in Neural Information Processing Sys-          Wiebe, Pearu Peterson, Pierre Gérard-Marchant,
  tems.                                                      Kevin Sheppard, Tyler Reddy, Warren Weckesser,
                                                             Hameer Abbasi, Christoph Gohlke, and Travis E.
Tiwalayo Eisape, Vineet Gangireddy, Roger P. Levy,           Oliphant. 2020. Array programming with NumPy.
  and Yoon Kim. 2022. Probing for incremental parse          Nature, 585(7825):357–362.
  states in autoregressive language models. In Findings
  of EMNLP 2022.                                           Trevor Hastie and Robert Tibshirani. 1986. Generalized
                                                             additive models. Statistical Science, 1(3):297–318.
Nelson Elhage, Neel Nanda, Catherine Olsson, Tom           Felix Hill, Roi Reichart, and Anna Korhonen. 2015.
  Henighan, Nicholas Joseph, Ben Mann, Amanda                Simlex-999: Evaluating semantic models with (gen-
  Askell, Yuntao Bai, Anna Chen, Tom Conerly,                uine) similarity estimation. Computational Linguis-
  Nova DasSarma, Dawn Drain, Deep Ganguli, Zac               tics, 41(4):665–695.
  Hatfield-Dodds, Danny Hernandez, Andy Jones,
  Jackson Kernion, Liane Lovitt, Kamal Ndousse,            Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long
  Dario Amodei, Tom Brown, Jack Clark, Jared Ka-             short-term memory. Neural computation, 9(8):1735–
  plan, Sam McCandlish, and Chris Olah. 2021. A              1780.
  mathematical framework for transformer circuits.
  Transformer Circuits Thread.                             Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch,
                                                              Elena Buchatskaya, Trevor Cai, Eliza Rutherford,
Jeffrey L Elman. 1990. Finding structure in time. Cog-        Diego de las Casas, Lisa Anne Hendricks, Johannes
   nitive science, 14(2):179–211.                             Welbl, Aidan Clark, Tom Hennigan, Eric Noland,
  Katherine Millican, George van den Driessche, Bog-        Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey
  dan Damoc, Aurelia Guy, Simon Osindero, Karen               Dean. 2013. Efficient estimation of word representa-
  Simonyan, Erich Elsen, Oriol Vinyals, Jack William          tions in vector space. In International Conference on
  Rae, and Laurent Sifre. 2022. Training compute-             Learning Representations (Workshop Poster).
  optimal large language models. In Advances in Neu-
  ral Information Processing Systems.                       Joakim Olsen, Arild Brandrud Næss, and Pierre Lison.
                                                              2021. Assessing the quality of human-generated sum-
Dieuwke Hupkes, Sara Veldhoen, and Willem Zuidema.            maries with weakly supervised learning. In Proceed-
  2018. Visualisation and ‘diagnostic classifiers’ re-        ings of the 23rd Nordic Conference on Computational
  veal how recurrent and recursive neural networks            Linguistics (NoDaLiDa), pages 112–123, Reykjavik,
  process hierarchical structure. Journal of Artificial       Iceland (Online). Linköping University Electronic
  Intelligence Research, 61:907–926.                          Press, Sweden.

Mohit Iyyer, Varun Manjunatha, Jordan Boyd-Graber,          Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas
 and Hal Daumé III. 2015. Deep unordered compo-               Joseph, Nova DasSarma, Tom Henighan, Ben Mann,
 sition rivals syntactic methods for text classification.     Amanda Askell, Yuntao Bai, Anna Chen, Tom Con-
 In Association for Computational Linguistics.                erly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds,
                                                              Danny Hernandez, Scott Johnston, Andy Jones, Jack-
Been Kim, Martin Wattenberg, Justin Gilmer, Carrie            son Kernion, Liane Lovitt, Kamal Ndousse, Dario
  Cai, James Wexler, Fernanda Viegas, et al. 2018. In-        Amodei, Tom Brown, Jack Clark, Jared Kaplan,
  terpretability beyond feature attribution: Quantitative     Sam McCandlish, and Chris Olah. 2022. In-context
  testing with concept activation vectors (tcav). In In-      learning and induction heads. Transformer Circuits
  ternational conference on machine learning, pages           Thread.
  2668–2677. PMLR.
                                                            Denis Paperno, Germán Kruszewski, Angeliki Lazari-
Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen            dou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro
  Mussmann, Emma Pierson, Been Kim, and Percy                 Pezzelle, Marco Baroni, Gemma Boleda, and Raquel
  Liang. 2020. Concept bottleneck models. In Inter-           Fernández. 2016. The LAMBADA dataset: Word
  national Conference on Machine Learning, pages              prediction requiring a broad discourse context. In
  5338–5348. PMLR.                                            Proceedings of the 54th Annual Meeting of the As-
                                                              sociation for Computational Linguistics (Volume 1:
Yair Lakretz, Germán Kruszewski, Théo Desbordes,              Long Papers), pages 1525–1534.
  Dieuwke Hupkes, Stanislas Dehaene, and Marco Ba-
                                                            Jeffrey Pennington, Richard Socher, and Christopher D.
  roni. 2019. The emergence of number and syntax
                                                               Manning. 2014. GloVe: Global vectors for word
  units in LSTM language models. In Proceedings of
                                                               representation. In Empirical Methods in Natural
  the 2019 Conference of the North American Chap-
                                                               Language Processing (EMNLP), pages 1532–1543.
  ter of the Association for Computational Linguistics:
  Human Language Technologies, Volume 1 (Long and           Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt
  Short Papers), pages 11–20.                                Gardner, Christopher Clark, Kenton Lee, and Luke
                                                             Zettlemoyer. 2018. Deep contextualized word repre-
Nelson F Liu, Matt Gardner, Yonatan Belinkov,                sentations. In Proceedings of the 2018 Conference of
  Matthew E Peters, and Noah A Smith. 2019. Linguis-         the North American Chapter of the Association for
  tic knowledge and transferability of contextual repre-     Computational Linguistics: Human Language Tech-
  sentations. In Proceedings of the 2019 Conference of       nologies, Volume 1 (Long Papers), pages 2227–2237,
  the North American Chapter of the Association for          New Orleans, Louisiana. Association for Computa-
  Computational Linguistics: Human Language Tech-            tional Linguistics.
  nologies, Volume 1 (Long and Short Papers), pages
  1073–1094.                                                Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers,
                                                              John Thickstun, Sean Welleck, Yejin Choi, and Zaid
Kevin Meng, David Bau, Alex J Andonian, and Yonatan           Harchaoui. 2021. Mauve: Measuring the gap be-
  Belinkov. 2022. Locating and editing factual associ-        tween neural text and human text using divergence
  ations in GPT. In Advances in Neural Information            frontiers. Advances in Neural Information Process-
  Processing Systems.                                         ing Systems.
Kevin Meng, Arnab Sen Sharma, Alex J Andonian,              Ofir Press and Lior Wolf. 2017. Using the output embed-
  Yonatan Belinkov, and David Bau. 2023. Mass-                ding to improve language models. In Proceedings of
  editing memory in a transformer. In The Eleventh            the 15th Conference of the European Chapter of the
  International Conference on Learning Representa-            Association for Computational Linguistics: Volume
  tions.                                                      2, Short Papers.

Stephen Merity, Caiming Xiong, James Bradbury, and          Filip Radenovic, Abhimanyu Dubey, and Dhruv Maha-
   Richard Socher. 2017. Pointer sentinel mixture mod-         jan. 2022. Neural basis models for interpretability.
   els. In International Conference on Learning Repre-         In Advances in Neural Information Processing Sys-
   sentations.                                                 tems.
Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya       Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Or-
  Sutskever, et al. 2018. Improving language under-           donez, and Kai-Wei Chang. 2018. Gender bias in
  standing by generative pre-training.                        coreference resolution: Evaluation and debiasing
                                                              methods. In Proceedings of the 2018 Conference
Alec Radford, Jeffrey Wu, Rewon Child, David Luan,            of the North American Chapter of the Association for
  Dario Amodei, and Ilya Sutskever. 2019. Language            Computational Linguistics: Human Language Tech-
  models are unsupervised multitask learners.                 nologies, Volume 2 (Short Papers), pages 15–20, New
                                                              Orleans, Louisiana. Association for Computational
Douglas LT Rohde, Laura M Gonnerman, and David C              Linguistics.
  Plaut. 2005. An improved model of semantic similar-
  ity based on lexical co-occurrence.

Herbert Rubenstein and John B. Goodenough. 1965.
  Contextual correlates of synonymy. Commun. ACM,
  8(10):627–633.

H. Schütze. 1992. Dimensions of meaning. In Pro-
  ceedings of the 1992 ACM/IEEE Conference on Su-
  percomputing, Supercomputing ’92, page 787–796,
  Washington, DC, USA. IEEE Computer Society
  Press.

Ilya Sutskever, James Martens, and Geoffrey E Hinton.
   2011. Generating text with recurrent neural networks.
   In International Conference on Machine Learning.

Peter D Turney. 2010. From frequency to meaning: Vec-
  tor space models of semantics. Journal of Artificial
  Intelligence Research, 37:141–188.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
  Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
  Kaiser, and Illia Polosukhin. 2017. Attention is all
  you need. Advances in neural information processing
  systems.

Ben Wang and Aran Komatsuzaki. 2021. GPT-
  J-6B: A 6 billion parameter autoregressive lan-
  guage model. https://github.com/kingoflolz/
  mesh-transformer-jax.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mo-
  hananey, Wei Peng, Sheng-Fu Wang, and Samuel R.
  Bowman. 2020. BLiMP: The benchmark of linguis-
  tic minimal pairs for english. Transactions of the
  Association for Computational Linguistics, 8:377–
  392.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
  Chaumond, Clement Delangue, Anthony Moi, Pier-
  ric Cistac, Tim Rault, Remi Louf, Morgan Funtow-
  icz, Joe Davison, Sam Shleifer, Patrick von Platen,
  Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu,
  Teven Le Scao, Sylvain Gugger, Mariama Drame,
  Quentin Lhoest, and Alexander Rush. 2020. Trans-
  formers: State-of-the-art natural language processing.
  In Proceedings of the 2020 Conference on Empirical
  Methods in Natural Language Processing: System
  Demonstrations, pages 38–45, Online. Association
  for Computational Linguistics.

Zebin Yang, Aijun Zhang, and Agus Sudjianto. 2021.
  GAMI-Net: An explainable neural network based on
  generalized additive models with structured interac-
  tions. Pattern Recognition, 120:108192.
A     Language Model Training Details                                  Model               Time ↓
                                                                       Backpack-Micro      0.093
We use the FlashAttention codebase (Dao et al.,                        Transformer-Micro   0.065
2022) which in turn relies on the Huggingface code-
                                                                       Backpack-Mini        0.21
base (Wolf et al., 2020) and NumPy (Harris et al.,                     Transformer-Mini     0.15
2020). We perform no preprocessing of OpenWeb-                         Backpack-Small       0.36
Text. We do no explicit hyperparameter sweep                           Transformer-Small    0.26
for OpenWebText training beyond our sense vec-
tor ablation, instead taking the defaults provided.      Table 7: Timing benchmarking results on an A100,
We train our models on 4 A100 (40GB) GPUs.               average time to compute forward pass on 32-batch size
                                                         512-sequence length input.
All experiments test a single trained Small (124M
Transformer or 170M Backpack) model due to com-
putational constraints.                                  GPT-2-xl, we average all subwords. For GPT-J
                                                         (which uses the same tokenizer), we take the first
A.1    The feed-forward sense network.
                                                         subword.
We parameterize the feed-forward network for our
sense vectors by first performing layer normaliza-       D     Sense Vector Control Details
tion on the input embeddings, and then a feed-
forward layer with residual connection and layer         D.1    Topic control details
norm (despite it being a function of just one word)      The full results are in Table 12. The list of topics,
to dimensionality 4d and back to d. Then a subse-        and the corresponding bags-of-words, are given in
quent feed-forward network to hidden dimension-          Table 10. For PPLM, the hyperparameter we vary
ality 4d and then up to k ∗ d. We include a second       to change the strength of topic control is the step
layer norm and residual before the second feed-          size (Dathathri et al., 2019).
forward layer accidentally as a side-effect of the
                                                            We consider a document as matching the seman-
underlying language model codebase.
                                                         tic control if the classifier assigns greater than 0.5
   For our experiments ablating k in Section 4.5,
                                                         probability to the attempted class. We generated
the second feed-forward component maps to d and
                                                         from our models with ancestral sampling with no
then kd, not 4d → kd.
                                                         truncation or temperature change.
B     Extra evaluations                                  Topic control. Let b ∈ R|V| be the many-hot
B.1    Timing Benchmarking                               vector defined by the bag of words input to the
                                                         control problem. That is, if the bag is arts, culture,
To benchmark the speed of each model, we used
                                                         then b has 1 at the indices corresponding to those
a single A100 GPU, running the forward pass of
                                                         words, and 0 elsewhere. To determine the initial
each model with a sequence length of 512 and a
                                                         weights δ for each sense vector, we first sort all
batch size of 32. We ran 100 forward passes and
                                                         |V| ∗ k sense vectors by decreasing normalized dot
present the average time taken across the 100. We
                                                         product with the bag of words vector:
present this in lieu of FLOPs because A100 GPUs
are relatively standard, and this allows for a more
                                                                                   b⊤ E ⊤ C(x)
directly usable time estimate. Results are in Table 7.               s(C(x)) =                             (17)
                                                                                  max(E ⊤ C(x))
We find that Backpacks take roughly 1.4x as long
to run as their underlying Transformers.
                                                         We then take the 0.95, 0.80, and 0.60 quantiles
                                                         of these scores to determine how to weight the
C     Lexical Similarity Details
                                                         vectors. Intuitively, the vectors in the highest quan-
To handle words in the lexical similarity datasets       tiles (most associated with the target topic) are up-
that don’t appear as single words in the tokenizer,      weighted the most during decoding, to push the gen-
we use one of two methods. We either average all         eration towards the topic. The three quantiles parti-
subwords, or take the first subword. The results         tion the set of scores into 4, which are given sepa-
for the two methods were similar, but we take the        rate δ values; the exact 4 depend on the strength of
better overall for each model. For all Backpack          control (i.e., different points in Figure 2.) The exact
methods, our 124M-parameter Transformer, and             δ upweighting for each point are given in Table 11.
                                # Senses   Total Params      Contextl. Params   OWT PPL
                                   1             74.3M           72.7M               38.5
                                   4             75.6M           72.7M               29.3
                                  16             80.5M           72.7M               26.0
                                  64            100.2M           72.7M               24.0

Table 8: OWT perplexity and parameter count as a function of the number of sense vectors. All models trained for
50k steps, 500k token batch size, on OWT.


                                                                  Control Strength    δ for quantiles 0.95, 0.80, 0.6, < 0.6
                                                                   0 (unmodified)                    1,1,1,1
                                                                         1                       1.5, 1.5, 1.3, 1
                                                                         2                       2.2, 2.2, 1.5, 1
                                                                         3                        3.3, 3.3, 3, 1

            Model     Dim     Layers   Heads                    Table 11: Initial topic control weights for each quantile.
            Micro     384         6         6
            Mini      640         8         8
            Small     768        12        12                   Topic annealing. From the the beginning value
                                                                of δ given above, we anneal back to 1 as follows.
         Table 9: Model size hyperparameters.                   For each sense C(xj )ℓ , we compute the total sum
                                                                of non-negative log-probability assigned by the
                                                                sense to the set of words generated so far, intu-
                                                                itively to compute whether the words already gen-
                                                                erated express the meaning intended by the sense:
                                                                                n
                                                                                X                            
                                                                   aC(xj )ℓ =         max x⊤
                                                                                           i E ⊤
                                                                                                 C(x  )
                                                                                                     j ℓ ), 0   . (18)
                                                                                i=1

                                                                We then re-weight by a term dependent on the se-
                                                                quence index to upweight terms near to the most
                                                                recently generated text:
     Topic Label              Bag-of-words                                                   
     arts_culture             arts, culture                       bC(xj )ℓ = σ −aC(xj )ℓ f + 6 ∗ (1 + j) /100
     business_entrepreneurs   business, entrepreneurs
     celebrity_pop_culture    celebrity, pop, culture                                                                   (19)
     diaries_daily_life       diaries, daily, life
     family                   family                            where j is the index of the word of the sense vector
     fashion_style            fashion, style
     film_tv_video            film, tv, video                   in the generated text, and f is a scaling constant set
     fitness_health           fitness, health                   to 7.5 divided by the maximum δ in the experiment
     food_dining              food, dining                      (the maximum of each row in Table 11.)
     gaming                   gaming
     music                    music                                Finally, we compute the annealed δ as a soft
     news_social_concern      news, social, concern             combination, weighted by bC(xj )ℓ , of the maximum
     other_hobbies            hobbies                           delta and the default of 1:
     relationships            relationships
     sports                   sports
     travel_adventure         travel, adventure                           δℓij = bC(xj )ℓ δℓij + (1 − a) ∗ 1.           (20)
     youth_student_life       youth, student, life

Table 10: The topics used in our topic classifier, and the      D.2    Gender bias mitigation details
bags-of-words we use for control.
                                                                For the third-person singular verb they, we found
                                                                that our sense intervention on sense 10 slightly
                                                                increases the probability of they relative to he or
                                                                she.
                                                                  The full set of nouns and prompts we use is
                                                                as follows. For role nouns, we use mechanic,
 Method         Sem Acc ↑       Toks-in-vocab ↓   MAUVE ↑
 Transformer
 Unchanged        6.8%              0.0%               0.95
 PPLM-.01         8.4%              0.1%               0.94
 PPLM-.04         23.9%              2.6%              0.81
 PPLM-.05         30.3%              5.5%              0.62
 PPLM-.06         37.7%             12.3%              0.41
 PPLM-.07         40.8%             18.8%              0.25
 Backpack
 Unchanged         7.4%              0.0%              0.92
 Ours+1           12.1%              0.2%              0.91
 Ours+2           24.3%              1.5%              0.90
 Ours+3           35.3%              3.5%              0.83

Table 12: Topic control via pseudovocabulary, vs PPLM.
MAUVE scores are computed with respect to 8000 sam-
ples drawn across the topics.

               nurse                   developer
     logprob        target        logprob     target
      +12.21     professional     -15.27       She
      +11.62         She          -15.14      herself
      -13.38       himself        -13.02        her
      -13.24         His          -12.34    corruption
      -12.63         his          -12.15       she
      -12.31         his          -12.14       gam
      -11.39          he          -12.13       HR

Table 13: Some of the most up- or down-weighted
words by sense 10 of the words nurse and developer.
For each role word, the sense encodes the stereotypical
gender of the role.


accountant, farmer, baker, assistant, construction,
guard, carpenter, analyst, physician, cook, clerk,
manager, developer, librarian, salesperson, mover,
hairdresser, auditor, sheriff, janitor, driver, chief,
teacher, writer, receptionist, CEO, nurse, house-
keeper, secretary, counselor, attendant, supervisor,
designer, lawyer, editor, cashier, laborer, tailor,
cleaner.
   For evaluation, we use the prompts in Table 14.
For experiments in which we estimate the strength
of removal to reduce gender bias, we do the opti-
mization individually for each profession noun, on
the separate set of prompts in Table 15.
Prompt
I really liked my PROFESSION. What I liked most was that,
I really liked my PROFESSION. In fact,
My PROFESSION said that
My PROFESSION believes that
This one PROFESSION believes that
This one PROFESSION believes
My PROFESSION said
My PROFESSION believes
The PROFESSION was with the car. When
The PROFESSION was with the car. Then
While driving to the store, the PROFESSION looked over on the dash and then
A PROFESSION went to chat over to chat before
Some PROFESSION asked to look over things, but

           Table 14: Prompts used for gender bias evaluation.




              Prompt
              The PROFESSION came into the room. When
              The PROFESSION came into the room. Then
              I went to the talk to the PROFESSION. What
              I went over to the talk to the PROFESSION. Why
              I went over to the talk to the PROFESSION;

         Table 15: Prompts used for gender debiasing estimation.
Topic             Backpack Sense Control, Control=2                       Transformer PPLM, stepsize=0.04
Arts & Culture    (I kind dude) just used to be creative and make         Sunday, February 6: WestSTAR en-twitter-
                  music but it feels like I’m 40. Conscious of how        ilesurcast.blogspot.com
                  bad it is and how awesome it is like making such        Tuesday, February 7: WestStar en-chat breathe
                  a fast consciousness and not even being musi-           of ancient energy. www.weststar.org
                  cian meets people who answer for you, especially        Monday, February 8: West Star
                  when it’s scary." de la Cruz © Dan Wilson (2002).       Mares and Moon of the ages
                                                                          “Happiness is not easy to do”, Nicolas Jeansma,
                                                                          the Eternal Life programme director analyses his-
                                                                          tory, culture, sociality and social magic.
                                                                          : ’Oh the
Business & En-    Flickr advertisers is nothing new, so let’s hope        We’ve decided to put out a newsletter to your
trepreneurship    you know where you buy the latest edition.              guys, wondering as you cope with the tribula-
                  At the same time, the fix has been pushed through,      tions of your business ventures and a job position.
                  and while the overall business is pulling away          One way to put it is: You’re not good enough.
                  from mainland Asia, publishers have given con-          You’ve failed and you’re not getting anything
                  trol over social media options to researchers at        done. You’re not doing enough. You’re not bring-
                  New York University and Columbia University.            ing the passion and ideas you might have to a
                  A new report from the Columbia board offers             business. But one thing’s for sure: if you self-
                  some clues as to why.                                   promote, you often might take the business to a
                  "My store in Alabama is used to a lot of Marines,       profitable buyer. Continue
                  and I just dropped as such. I don’t know why, but
                  I’ve had
Celebrity & Pop   Meetings and greets with reporters and celebrities      Type Services rumors have been up in the media
Culture‘          of all kinds — pop culture, fashion, sports, food,      since last month—and now we have some con-
                  celebrity lifestyle and otherwise — have been laid      firmed to the CBC Radio musical news channel’s
                  door-to-door on the Dallas television market with       Twitter stream.
                  both LaVar and his wife, Arron, taking over the         The group’s guitarist, Greg Carr, has just an-
                  showroom-oneship business at Big Star Barber.           nounced that he’s working with Papa John as
                  “We think Big Star’s an interesting exchange,”          the band’s lead singer and guitarist. Accord-
                  Arron says. “They’ve got an experience they’re          ing to bizarre French pop culture creation icon
                                                                          Valentino pop music singer/writer Jiv pop pop
                                                                          model, who also wrote pop pop music’s MyS-
                                                                          pace and Twitter pop memes, Cassidy gig pop
                                                                          pop superstar is
Diary & Daily     The exact actual life cycle life form life soars on     The Rome crew logam tagged Louisville Main
Life              and dies off in comparison to our own. During the       Street today morning and observed a loading
                  first few years of life, the total life form you take   dock at the center of downtown Louisville. The
                  to decide what to eat, how much of it to drink,         dock is just bigger than what was supposed to
                  why, and whether you want to exercise have been         dock the loading area for emergencies. They
                  completely smashed and the technological capa-          watched over the crowd after passing the boat and
                  bility to make that happen seriously out of the         finally realized that they’d caught some missed
                  blue has been completely lost, jumping from com-        traffic signals. "Serious congestion" has so far
                  plexity to complexity, totally overwhelming the         unnerved people from the Grande family picnics
                  mushroom in its ability to discover what levels         to weddings picnics picnics.
                  it’s supposed to                                        MTD Charlotte Pulse (@mtdphp
Fashion           This article is about the fashion label fashion         Twitter personality @ceboperformancemk
                  week fashion style month fashion fashion style          tweeted in response to the story about you.
                  fashion style fashion week fashion style fashion        Fashion designer underwear, designer cook dress,
                  fashion fashion style fashion fashion style fashion     sexuality art models, sex con artists, real goths.
                  history fashion fashion fashion fashion fashion         BuzzFeed
                  fashion fashion johnny dressed in an actor’s spe-       You think my brain’s shit about what’s fashion
                  cially created costume news news icon                   looks like? Yeah no, I’m not on it. I’m fashion.
                  The Comic Relief series features stories, such as       I’m fine fashion. Yes I appreciate the brand but
                  plungers from the comic books.                          the people behind it[. . . ] adults go fashion, or
                  It was originally published as a comic published
                  in Dark Horse Comics in English and in both
                  comic books and graphic novels.[1] It was pro-
                  duced

             Table 16: The first, non-cherry-picked category-satisfying example from each model.
Topic           Backpack Sense Control, Control=2                      Transformer PPLM, stepsize=0.04
Film, TV, &     Originally published Live chat Qs with the film        Well, the hype is real, and with the release of the
Video           website writer, who raised millions at least two       latest episode of season two (which I’m probably
                years ago I contacted him with the same questions      not supposed to review), it feels like you won’t
                as you’re doing.                                       be afraid to retweets fideo.
                I’m a bit optimistic that you’re right, but you’re     By “HAPPY FINALS,” the footage maker has
                just not responding. As you studied the film           used a GIF video to give viewers look at Fideo’s
                timer/mapplot’n’cookies response speed, I read         dancing triangles and serenity dancing around a
                the excerpts and couldn’t make out a massive           moving picture. Thank you, fideo!
                amount of time differences. Very minor.                If the
                What do you think about some of the terms
Fitness    &    CLOSE Don’t think tanking will spell good news         Today
Health          for Detroit medical marijuana patients but the         we learn more about the rise of the ice age, multi-
                owner of its dispensaries saying that is just part     drug cocaine epidemic, global population ex-
                of the problem facing the growing number of ill        plosion and warfare epidemic by following Dr.
                people having access to pot.                           Kristof Dr. Freedk published in the British Jour-
                Healthcare workers are treated for tumors in a dis-    nal of Medicine The authors update their lofty
                pensary in Oakland. (Photo: Christopher Sator-         goal and continue to refine their work for public
                ica, Special to CNN)                                   health.
                An array of medical centers have lined up near         The International Health Services Committee has
                Detroit after a medical marijuana reform forum         just released a new research, The next three years
                at the University of Michigan put the debate over      could be very costly for health care in Australia,
                the drug at                                            hospitals, state health systems and dietary health.
                                                                       A recent report from
Food & Dining   As weeks wore maple leafed food trucks, and            I would dearly love to stand at that galloping chair
                food processors reminisced about their great days      and who doesn’t has amazingly friends associated
                past, healthcare workers found out one day that        with their backs hurting? I was a big first timer
                they should get better working conditions with         yesterday. Not always with bacon but I held til
                little regard for their bodies.                        calms up. Big chunks of bacon super nice but not
                Barbara Butterfield, the former Shop Swagger           me. However there are times where the pieces
                workshop in Clarksdale, got shot dead on Mon-          pull apart and this happens very hard to homo
                day morning when she tried to stop a father Fran-      and crackers afgh. All Mixed ones made popular
                cisco Lee Walker from firing a gun. Walker, 20,        points that have the food triggers across: lack of
                had just started his Aug. 27 firing. Exposure to       meats rinsing and eating
                fire and clothes caused Walker
Gaming          My parents encouraging kids to be competitive          Every year, many migrants continue to struggle
                gaming at school is not a new concept. Gaming          to find the skills they need in an emerging tech-
                has been around since the earliest days on pa-         nology. But every year, it comes quite a surprise
                per, and their perspective is always superior than     to hear the latest news about computerized com-
                yours. Quality doesn’t always apply, and that’s        puting and the gaming community.
                why we bucked that trend’ father                       For the sake of many gaming communities, we
                The English woman’s son Anthony, who is best           here at 14/gamer.org love gaming. It is an im-
                known for his role as Most Wanted, came up             portant industry in gaming, as it often draws pas-
                with the idea of pulling a 30-year-old mentally        sionate gamers from gaming and lends the gam-
                disabled woman who had been using motorbikes           ing community the ability to allow itself special
                for                                                    moments like gaming gaming days and gaming
                                                                       gaming. We
Music           David has been a staunch critic of music culture       From the East art council HQ of MondoJapan
                that promotes music as something new, daring,          Everyone laughs when a sheet metal title is ren-
                and powerful. As he explained. ("I never thought       dered artistically constrained and we say, "Whoa.
                I was one of those stupid, stupid old people who       Then the skin guy! This is a very Chi style steel."
                just listens to music or really hears it it’s always   Well I don’t think anyone’s ever heard that before.
                the same as when I was a kid," he said.) And           There’s only one coil metal group that is not a
                when he was a touring musician, those opinions         tarantella performance music group...at least in
                were totally correct. Read the entire interview        America...compart music ten times over and they
                below.                                                 will never release tracks for it that it is a
                On trying to inculcate younger vocalists with the
                "

           Table 17: The first, non-cherry-picked category-satisfying example from each model.
 Topic            Backpack Sense Control, Control=2                     Transformer PPLM, stepsize=0.04
 News & Social    Buildersh B2 has been compared unfathomable           After initially putting itself over Sports Illustrated
 Concern          by a number of critics because of his security        on Monday, the New York Times was forced to
                  concerns.                                             apologize for its widespread coverage of its re-
                  Breaking News Alerts Get breaking news when           porting on the State of Rhode Island – a state that
                  it happens — in your inbox. Email Sign Up By          has been the subject of gossip news for some time
                  signing up you agree to receive email newsletters     and which its harsh news and ratings policy has
                  or alerts from POLITICO. You can unsubscribe          spawned.
                  at any time.                                          Late at night on Monday, we learned that the New
                  Yet, on Tuesday, Monday and Tuesday, the devel-       York Times had reached a breaking news cycle
                  oper reached the milestone of completing the first    decision and we snagged our exclusive first look
                  UPS facility located in the town of Cloudbreak.       at the news. Here’s what you didn’t
                  He secured $4
 Relationships    Early life release parties is relationship couples    Any learning is like being completely ignorant of
                  with relationships over relationships. This cen-      new information. Schools are forced to teach stu-
                  sus does not count relationships by those who         dents to treat one another in the right way, but we
                  have been with those relationships over the last      still have to recognize that we have to learn how
                  three years. For more information about early life    to be friends with as much as we can. When San-
                  release parties, check the release party census.      tod relationships are hard and relationships can be
                  Carlo Mathieu                                         complicated and confusing, there will always be
                  Carlo Mathieu was born in 1958. He lives in           learning relationships, relationships that remind
                  Augusta, Ga., with his biological father, Malcolm     us that we don’t mean relationships, relationships
                  Mathieu, who was president of the Augusta West        relationships that are boundaries, relationships re-
                  Raceway at the time.                                  lationships with friends in need relationships with
                  Benjamin Math                                         involved relationships, relationships relationships
                                                                        relationships
 Sports           PRESS W/NEWS BLOK Play slideshow 1 of 83              Authorities in California say they are investigat-
                  Express sports retail giant Sports Direct.            ing equestrian skiers who struck a 19 year-old
                  Sports Direct has revealed the on offer outdoor       boy from a snow-covered mountainand beating
                  sports gear Brand new from Google has been            him on the head with shovels.According to Smith-
                  developed. Here’s what you can expect from            Cox, those same well clients found out they had
                  Google’s sporting expertise.<|endoftext|>About        also been tardled by a $500 pour frompipe on top
                  The potential of a west coast restaurant for toler-   of of a Black Rock vault. And it appears the ul-
                  ance and pity                                         timate goal of those riders and their company of
                  Their position at this point hurts me less than       riders was killed.Jeremy Goschz is one of those
                  they believe it deserves, because they probably       survivors. His racing
                  shouldn.
                  I’m going to help them
 Travel & Ad-     My next stop destination for me is adventure          Equality
 venture          travel. I travel Disney World and make sure that      Equality – open life – inequalities – political op-
                  the worlds under my belt and desert warriors that     pression –
                  I’ve been fighting for have a place or two at their   write and publish your work
                  disposal that are compatible with my use of cur-      Equality is a freedom to work, to die. Access
                  rent technology. This job is being completed with     to free healthcare, free outer space travel, pho-
                  the help of any freelance user submission infor-      tocopies online, happy endings, self travel – to
                  mation you may have provided. It’s only fair          travel to someone else’s heart (read: stop taking
                  to give you some tips to help you figure it out       drugs), to move faster, to travel in train travel, to
                  if there are any unknown sideside locations that      stop a vacation abroad (tell others your travels),
                  you                                                   to return to a home each time
 Youth & Stu-     College students at almost every age advantage        lame University saw a 32 per cent rise in its un-
 dent Life        who take advantage of learning opportunities in       dergraduate science institutes and 14 per cent
                  the sport of running spend at least five years an     increase in its researchers from recent years.
                  average of $10 or more per year to do it, accord-     Director Of University Development, Mike Bren-
                  ing to the University of San Diego’s National         nan, said: "The growth in university employment,
                  Football Clearinghouse.                               coming from such a historic campaign, is some-
                  Those risk factors lift nearly a third of univer-     thing to celebrate as we support our young people
                  sity and college football athlete spend, more than    and room to progress in science and technology."
                  double that of a comparable age group of men          A student was interviewed in a recent paper about
                  and women who spend 4,000 hours per year as           university employment, specifically a disserta-
                  runners, or 5,000 to                                  tion.
                                                                        "For the first time, people are

Table 18: The first, non-cherry-picked category-satisfying example from each model. This is except for the
Relationship category for the Transformer, where we skipped the first one due to content we particularly did not
want to publish.
                                 Positive Log-Probability Mass for Senses of word quickly
      0               1                  2               3            4            5             6              7
 approaching       oggles             quickly        enough          stro         iii         razen           asuring
  ascended         Marks              swiftly        rotating         zn       Original     forgotten        delusion
      grav          Axis              rapidly         paced       strokes        alsa         forget        stimulated
      gent         claimer             quick           ened        uling        chenko        social       recollection
   disposed         Roche              quick       retreating         $_      resolution       rius            stimul
     risen      demonstration        instantly     Subscribe       grass         ient        relapse            Wem
    dispose        blaster           promptly      dismissing      lessly      baskets       baseless       persistent
  becoming         ducers              soon       diminishing        iken         uin       Statement          urbed
     ascert     Specifications          fast      disappearing      izing         ora         athing           retard
   climbed           Viet              Quick         varying          bg         alid         Akron          restraint
      8               9                 10             11           12            13            14             15
  processors        slowly            tering       Definitely       quick         oted         ouse           Sims
     darts         Slowly              Bers          initely      quickest     distances       pee            Noir
 milliseconds        Slow              Fed             raid         quick        outed        ouses           TMZ
      wip       conveniently          ascus          alright      quicker        aught         pees          Streets
    iazep           slower             Bust        Personally        fast         UC          attach        expressly
   reptiles        cheaply            aucus          laughs       quickly          ob           tro          Attend
    Kelvin       responsibly           Ryu         ALWAYS           rapid        digits        iffe          Rooms
      Ow          gradually           sector        Always           fast        ench          aces          Within
     Soon          quietly             Petra        Ideally         faster       Code          lain           Rum
     Slug          waiting             DCS           Roses         fastest       apers         feet          Forced

                                 Negative Log-Probability Mass for Senses of word quickly
      0               1                  2              3            4            5              6              7
    initely         sburg              ollen           una         Poké         quickly        Faster            .
      heit          orem               oned           URE           slow         quick       purposely      Sorceress
      Aly         Untitled              oths           rast       slower        swiftly     deliberately       itars
   istically        anted               ook             ipt        slows        rapidly      Definitely      Shogun
   Always         untreated             ught         ocracy       slowed       quickest           ey           Yen
   Doctors            til               Ded            law         DEV           quick        slower          oenix
       dl          broken               lost          uthor        encia         Quick        initely          Jagu
    urally           past              aught          ema         potions         fast          isner           izz
  ependence        ebook             recharge          ory        Machina      instantly     hesitated         eral
     raints       Continue              ady           antis        Slow          Quick      eyewitness        finals
      8               9                 10             11           12            13            14             15
     quist          WM               prototype       ciating        kins        quick          Laur            thal
     ocker            isf            projector     scrambling       Host        quick         Never          imble
     ovsky            fb              reconcil        rapid       loudspe      quickly        Jimmy           iquid
    ictions          WF            prominently     newcomer        enced        Quick         dearly       initialized
    olation       elevation         counterfeit     adapting        Evil        soon          Dating         ansas
     cano            RM                 word        speeding      washed         fast           _-_            IGH
     Proof           975                cellul       frantic        Kaf        rapidly         never       unciation
      cert           dir             prototype       novelty       Glass        Quick        Certainly       needs
      rero          ESE                collaps        paced         sod         hurry         eternal       commit
     anch           onder                dyl      instructional    advers    Immediately       Rare          tackle

Table 19: For each sense vector of the word quickly, the 10 words to which the sense vector assigns the highest
log-probability contribution, and the 10 to which it assigns the largest negative log-probability contribution. Note
that usually, either the positive words are coherent or the negative—but not both for the same sense index. Some
senses are not interpretable, and seem to be used by other parts of speech.

