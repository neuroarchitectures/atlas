# Knowledge hypergraphs prediction beyond binary relations Fatemi Taslakian 2019

> Source: `Knowledge_hypergraphs_prediction_beyond_binary_relations_Fatemi_Taslakian_2019.pdf`

---

                                                        Knowledge Hypergraphs: Prediction Beyond Binary Relations∗

                                                         Bahare Fatemi1,2† , Perouz Taslakian2 , David Vazquez2 and David Poole1
                                                                               1
                                                                                 University of British Columbia
                                                                                         2
                                                                                           Element AI
                                                              {bfatemi, poole}@cs.ubc.ca, {perouz,dvazquez}@elementai.com,


                                                                   Abstract




arXiv:1906.00137v3 [cs.LG] 15 Jul 2020
                                                                                                            In these studies, knowledge graphs are defined as directed
                                                                                                            graphs having nodes as entities and labeled edges as relations;
                                              Knowledge graphs store facts using relations be-              edges are directed from the head entity to the tail entity. The
                                              tween two entities. In this work, we address the              common data structure for representing knowledge graphs is
                                              question of link prediction in knowledge hyper-               a set of triples relation(head, tail) that represents informa-
                                              graphs where relations are defined on any number              tion as a collection of binary relations. There exist a large
                                              of entities. While techniques exist (such as reifica-         number of knowledge graphs that are publicly available, such
                                              tion) that convert non-binary relations into binary           as F REEBASE [Bollacker et al., 2008]. Wen et al. (2016)
                                              ones, we show that current embedding-based meth-              observe that in the original F REEBASE more than 1/3rd of
                                              ods for knowledge graph completion do not work                the entities participate in non-binary relations (i.e., defined
                                              well out of the box for knowledge graphs obtained             on more than two entities). We observe, in addition, that 61%
                                              through these techniques. To overcome this, we in-            of the relations in the original Freebase are non-binary.
                                              troduce HSimplE and HypE, two embedding-based                    Embedding-based models [Nguyen, 2017; Kazemi et al.,
                                              methods that work directly with knowledge hyper-              2020] have proved to be effective for knowledge graph com-
                                              graphs. In both models, the prediction is a func-             pletion. These approaches learn embeddings for entities and
                                              tion of the relation embedding, the entity embed-             relations. To find out if a triple relation(head, tail) is
                                              dings and their corresponding positions in the rela-          true, such models define a function that uses embeddings
                                              tion. We also develop public datasets, benchmarks             of relation, head, and tail and output the probability of
                                              and baselines for hypergraph prediction and show              the triple. While successful, such embedding-based methods
                                              experimentally that the proposed models are more              make the strong assumption that all relations are binary.
                                              effective than the baselines.
                                                                                                               Knowledge hypergraph completion is a relatively under-
                                                                                                            explored area. We motivate our work by outlining that con-
                                         1   Introduction                                                   verting non-binary relations into binary ones using methods
                                                                                                            such as reification or star-to-clique [Wen et al., 2016], and
                                         Knowledge hypergraphs are graph structured knowledge               then applying known link prediction methods does not yield
                                         bases that store facts about the world in the form of rela-        satisfactory results. Reification is one common approach of
                                         tions among any number of entities. They can be seen as            converting higher-arity relations into binary ones. In order to
                                         one generalization of knowledge graphs in which relations          reify a tuple having a relation defined on k entities e1 , . . . , ek ,
                                         are defined on at most two entities. Since accessing and stor-     we form k new binary relations, one for each position in this
                                         ing all the facts in the world is difficult, knowledge bases       relation, and a new entity e for this tuple and connect e to
                                         are incomplete; the goal of link prediction in knowledge (hy-      each of the k entities that are part of the given tuple using the
                                         per)graphs (or knowledge (hyper)graph completion) is to pre-       k binary relations. Another conversion approach is Star-to-
                                         dict unknown links or relationships between entities based on
                                                                                                            clique, which converts a tuple defined on k entities into k2
                                         existing ones. In this work, we are interested in the problem
                                                                                                            tuples with distinct relations between all pairwise entities in
                                         of link prediction in knowledge hypergraphs. Our motiva-
                                                                                                            the tuple.
                                         tion for studying link prediction in these more sophisticated
                                         knowledge structures is based on the fact that most knowl-            Both conversion approaches have their caveats when cur-
                                         edge in the world has inherently complex compositions.             rent link prediction models are applied to the resulting graphs.
                                            Link prediction in knowledge graphs is a problem that           The example in Figure 1a shows three facts that pertain to the
                                         is studied extensively, and has applications in several tasks      relation flies between. When we reify the hypergraph in this
                                         such as automatic question answering [Ferrucci et al., 2010].      example (Figure 1b), we add three reified entities. In terms
                                                                                                            of representation, the binary relations created are equivalent
                                             ∗                                                              to the original representation and reification does not lose
                                               This paper with the supplementary material can be found at
                                         https://arxiv.org/abs/1906.00137.                                  information during conversion. A problem with reification,
                                             †
                                               Contact Author                                               however, arises at test time: because we introduce new enti-
ties that the model never encounters during training, we do                                                                       e1             e2
                                                                                     Air
not have a learned embedding for these entities; and current                        Canada
                                                                                                                                    Air
embedding-based methods require an embedding for each en-                                                                          Canada
                                                                        Montreal                    Vancouver         Vancouver
tity in order to be able to make a prediction.
                                                                                                                                                 Montreal
   Applying the star-to-clique method to a hypergraph does                     Los                   New
not yield better results, as star-to-clique conversion loses in-              Angeles                York                  New          Los
                                                                                                                           York        Angeles
formation. Figure 1c shows the result of applying star-to-                               United
                                                                                                                        United
clique to the original hypergraph (Figure 1a), in which the tu-                          Airline
                                                                                                                        Airline                  e3
ple flies between(Air Canada, New York, Los Angeles) might
be interpreted as being true (since the corresponding entities                          (a)                                        (b)
are connected by edges), whereas looking at the original hy-                                                  Air
pergraph, it is clear that Air Canada does not fly from New                                                  Canada
York to Los Angeles.                                                                                                       Vancouver
                                                                                          Montreal
   In this work, we introduce two embedding-based models
that perform link prediction directly on knowledge hyper-
graphs without converting them to graphs. Both proposed                                             Los                   New
models are based on the idea that predicting the existence of                                      Angeles                York
a relation between a set of entities depends on the position
of the entities in the relation; otherwise, the relation is sym-                                             United
                                                                                                             Airline
metric. On the other hand, learning entity embeddings for
each position independently does not work well either, as this                                                  (c)
does not let the information flow between the embeddings for
the different positions of the same entity. The first model we     Figure 1: (a) Relation flies between with arity 3 defined on three
propose is HSimplE. For a given entity, HSimplE shifts the         tuples. (b) Reifying non-binary relations by creating three addi-
entity embedding by a value that depends on the position of        tional entities e1 , e2 , and e3 . (c) Converting non-binary relations
the entity in the given relation. Our second model is HypE,        into cliques using star-to-clique.
which in addition to learning entity embeddings, learns posi-
tional (convolutional) embeddings; these positional embed-         Knowledge hypergraph completion. Soft-rule mod-
dings are disentangled from entity representations and are         els [De Raedt et al., 2007; Kazemi et al., 2014] can easily
used to transform the representation of an entity based on         handle variable arity relations and have the advantage of
its position in a relation. This makes HypE more robust to         being interpretable. However, they can only learn a subset
changes in the position of an entity within a tuple. We show       of patterns [Nickel et al., 2016]. Guan et al. (2019) propose
that both HSimplE and HypE are fully expressive. To evalu-         an embedding-based method based on the star-to-clique
ate our models, we introduce two new datasets from subsets         approach. The caveats of this approach are discussed earlier.
of F REEBASE, and develop baselines by extending existing          m-TransH [Wen et al., 2016] extends TransH [Wang et
models on knowledge graphs to work with hypergraphs.               al., 2014] to knowledge hypergraph completion. Kazemi
   The contributions of this paper are: (1) showing that cur-      and Poole (2018) prove that TransH, and consequently
rent techniques to convert a knowledge hypergraph to knowl-        m-TransH, are not fully expressive and have restrictions in
edge graph do not yield satisfactory results for the link pre-     modeling relations. In contrast, we prove that our proposed
diction task, (2) introducing HypE and HSimplE, two mod-           models are fully expressive.
els for knowledge hypergraph completion, (3) a set of base-
lines for knowledge hypergraph completion, and (4) two new         Learning on hypergraphs. Hypergraph learning has been
datasets containing multi-arity relations obtained from sub-       employed to model high-order correlations among data in
sets of F REEBASE, which can serve as new evaluation bench-        many tasks, such as in video object segmentation [Huang et
marks for knowledge hypergraph completion methods. We              al., 2009] and in modeling image relationships and image
also show that our proposed methods outperform baselines.          ranking [Huang et al., 2010]. There is also a line of work
                                                                   extending graph neural networks to hypergraph neural net-
                                                                   works [Feng et al., 2019] and hypergraph convolution net-
2   Related Work                                                   works [Yadati et al., 2018]. These models are designed for
Existing methods that relate to our work in this paper can be      undirected hypergraphs, with edges that are not labeled (no
grouped into the following three main categories.                  relations), while knowledge hypergraphs are directed and la-
                                                                   beled graphs. As there is no clear or easy way of extending
Knowledge graph completion. Embedding-based models
                                                                   these models to our knowledge hypergraph setting, we do not
for knowledge graph completion such as translational [Bor-
                                                                   consider them as baselines for our experiments.
des et al., 2013; Wang et al., 2014], bilinear [Yang et al.,
2015; Kazemi and Poole, 2018], and deep models [Socher et
al., 2013] have proved to be effective for knowledge graphs        3    Definition and Notation
where all relations are binary. In Section 6 we extend some        A world consists of a finite set of entities E, a finite set of
of the models in this category to knowledge hypergraphs, and       relations R, and a set of tuples τ where each tuple in τ is
compare their performance with the proposed methods.               of the form r(e1 , e2 , . . . , ek ) where r ∈ R is a relation and
each ei ∈ E is an entity, for all i = 1, 2, . . . , k. The arity         knowledge completion as a multi-arity problem, without first
|r| of a relation r is the number of arguments that the relation         setting it up within the frame of binary relation prediction.
takes and is fixed for each relation. A world specifies what is          HSimplE. HSimplE is an embedding-based method for
true: all the tuples in τ are true, and the tuples that are not in       link prediction in knowledge hypergraphs that is inspired by
τ are false. A knowledge hypergraph consists of a subset of              SimplE [Kazemi and Poole, 2018]. SimplE learns two em-
the tuples τ 0 ⊆ τ . Link prediction in knowledge hypergraphs            bedding vectors e(1) and e(2) for an entity e (one for each
is the problem of predicting the missing tuples in τ 0 , that is,        possible position of the entity), and two embedding vectors
finding the tuples τ \ τ 0 .
                                                                         r(1) and r(2) for a relation r (with one relation embedding as
    An embedding is a function that converts an entity or a re-          the inverse of the other). It then computes the score of a triple
lation into a vector (or sometimes a higher order tensor) over                                          (1) (2)             (1) (2)
a field (typically the real numbers). We use bold lower-case             as φ(r(e1 , e2 )) = (r(1) , e1 , e2 ) + (r(2) , e2 , e1 ).
for vectors, that is, e ∈ Rk is an embedding of entity e, and               In HSimplE, we adopt the idea of having different repre-
r ∈ Rl is an embedding of a relation r.                                  sentations for an entity based on its position in a relation and
    Let v1 , v2 , . . . , vk be a set of vectors. The variadic func-     updating all these representations from a single training tuple.
tion concat(v1 , . . . , vk ) outputs the concatenation of its in-       We do this by representing each entity e as a single vector e
put vectors. We define the variadic function () to be the                (instead of multiple vectors as in SimplE), and each relation
sum of the element-wise product of its input vectors, namely             r as a single vector r. Conceptually, each e can be seen as
                             P`                                          the concatenation of the different representations of e based
   (v1 , v2 , . . . , vk ) = i=1 v1 (i) v2 (i) . . . vk (i) where each
vector vi has the same length, and vj (i) is the i-th element of         on every possible position. For example, in a knowledge hy-
vector vj . The 1D convolution operator ∗ takes as input a vec-          pergraph where the relation with maximum arity is α, an en-
tor v, a convolution weight filter ω, and a stride s and outputs         tity can appear in α different positions; hence e will be the
the standard 1D convolution as defined in torch.nn.Conv1D                concatenation of α vectors, one for each possible position.
function in PyTorch [Paszke et al., 2019].                               HSimplE scores a tuple using the following function.
    For the task of knowledge graph completion, an                           φ(r(ei , ej , . . . , ek )) = (r, ei , shift(ej , len(ej )/α), . . . ,
embedding-based model defines a function φ that takes a tu-                                              shift(ek , len(ek ) · (α − 1)/α))) (1)
ple x as input, and generates a prediction, e.g., a probability
(or score) of the tuple being true. A model is fully expressive          Here, shift(v, x) shifts vector v to the left by x steps, len(e)
if given any complete world (full assignment of truth values             returns length of vector e, and α = maxr∈R (|r|). We ob-
to all tuples), there exists an assignment of values to the em-          serve that for knowledge graphs (α = 2), SimplE is a spe-
beddings of the entities and relations that accurately separates         cial instance of HSimplE, with e = concat(e(1) , e(2) ) and
the tuples that are true in the world from those that are false.         r = concat(r(1) , r(2) ). The architecture of HSimplE is sum-
                                                                         marized in Figure 2a.
4    Knowledge Hypergraph Completion:                                    HypE. HypE learns a single representation for each entity,
                                                                         a single representation for each relation, and positional con-
     Proposed Methods                                                    volutional weight filters for each possible position. When an
The idea at the core of our methods is that the way an entity            entity appears in a specific position, the appropriate positional
representation is used to make predictions is affected by the            filters are first used to transform the embedding of each en-
role (or position) that the entity plays in a given relation. In         tity in the given fact; these transformed entity embeddings are
the example in Figure 1a, Montreal is the departure city; but            then combined with the embedding of the relation to produce
it may appear in a different position (e.g., arrival city) in an-        a score, i.e., the probability of the input tuple to be true. The
other tuple. This means that the way we use Montreal’s em-               architecture of HypE is summarized in Figures 2b and 2c.
bedding for computing predictions may need to vary based                     Let n denote the number of filters per position, l the filter-
on the position it appears in within the tuple. In general,              length, d the embedding dimension, and s the stride of a con-
when the embedding of an entity does not depend on its po-               volution. Let ωi ∈ Rn×l be the convolutional filters asso-
sition in the tuple during prediction, then the relation has to          ciated with position i, and let ωij ∈ Rl be the jth row of
be symmetric (which is not the case for most relations). On              ωi . We denote by P ∈ Rnq×d the projection matrix, where
the other hand, when entity embeddings are based on position             q = b(d − l)/sc + 1 is the feature map size. For a given
but are learned independently, information about one position            tuple, define f (e, i) = concat(e ∗ ωi1 , . . . , e ∗ ωin )P to be
will not interact with that of others. It should be noted that           a function that returns a vector of size d based on the entity
in several embedding-based methods for knowledge graph                   embedding e and its position i in the tuple. Thus, each en-
completion, such as canonical polyadic [Hitchcock, 1927;                 tity embedding e appearing at position i in a given tuple is
Lacroix et al., 2018], ComplEx [Trouillon et al., 2016], and             convolved with the set of position-specific filters ωi to give n
SimplE [Kazemi and Poole, 2018], the prediction depends on               feature maps of size q. All n feature maps corresponding to
the position of each entity in the tuple.                                an entity are concatenated into a vector of size nq and pro-
   In what follows, we propose two embedding-based meth-                 jected to the embedding space through multiplication by P .
ods for link prediction in knowledge hypergraphs. The first              The projected vectors of entities and the embedding of the
model is inspired by SimplE and has its roots in knowledge               relation are combined by inner-product to define φ:
graph completion; the second model takes a fresh look at                     φ(r(e1 , . . . , e|r| )) = (r, f (e1 , 1), . . . , f (e|r| , |r|)). (2)
                     (a)                                              (b)                                             (c)

Figure 2: Visualization of HSimplE and HypE architectures. (a) φ for HSimplE transforms entity embeddings by shifting them based on their
position and combining them with the relation embedding. (b) f (e, i) for HypE takes an entity embedding and the position the entity appears
in the given tuple, and returns a vector. (c) φ takes as input a tuple and outputs the score of HypE for the tuple.


    The advantage of learning positional filters disentangled               spectively, so that τ 0 = τtrain
                                                                                                       0        0
                                                                                                             ∪ τtest     0
                                                                                                                      ∪ τvalid . For any tuple
                                                                                  0
from entity embeddings is two-folds: On one hand, learning                  x in τ , we let Tneg (x) be a function that generates a set of
a single vector per entity keeps entity representations simple              negative samples through the process described above. Let r
and disentangled from its position in a given fact. On the                  and e represent relation and entity embeddings respectively.
other hand, unlike HSimplE, HypE learns positional filters                  We define the following cross entropy loss:
from all entities that appear in the given position; overall, this                                                          0
                                                                                                                       eφ(x )
                                                                                                                                        
separation of representations for entities, relations, and posi-                              X
                                                                                L(r, e) =            −log
tions facilitates the representation of knowledge bases having
                                                                                                                           X
                                                                                           x0 ∈τ 0           eφ(x0 ) +             eφ(x)
facts of arbitrary number of entities. It also gives HypE an ad-                               train
                                                                                                                      x∈Tneg (x0 )
ditional robustness, such as in the case when we test a trained
HypE model on a tuple that contains an entity in a position
never seen before at train time. We discuss this in Section 6.1.            5     Experimental Setup
    Both HSimplE and HypE are fully expressive — an im-
                                                                            5.1    Datasets
portant property that has been the focus of several stud-
ies [Fatemi et al., 2019]. A model that is not fully expressive             The experiments on knowledge hypergraph completion are
can embed assumptions that may not be reflected in reality.                 conducted on three datasets. The first is JF17K proposed
Theorem 1 (Expressivity). For any ground truth over en-                     by Wen et al. (2016); as no validation set is proposed for
tities E and relations R containing |τ | true tuples and                    JF17K, we randomly select 20% of the train set as validation.
α = maxr∈R (|r|) , there exists a HypE and a HSimplE model                  We also create two datasets FB- AUTO and M -FB15K from
with embedding vectors of size max(α|τ |, α) that represents                F REEBASE. For the experiments on datasets with binary rela-
that ground truth.                                                          tions, we use two standard benchmarks for knowledge graph
                                                                            completion: WN18 [Bordes et al., 2014] and FB15k [Bordes
Proof Sketch. To prove the theorem, we show an assignment                   et al., 2013]. See Table 2 for statistics of the datasets.
of embedding values for each of the entities and relations in τ
such that the scoring function of HypE and HSimplE gives 1                  5.2    Baselines
for t ∈ τ and 0 otherwise.
                                                                            To compare our results to that of existing work, we first de-
4.1   Objective Function and Training                                       sign simple baselines that extend current models to work with
                                                                            knowledge hypergraphs. We only consider models that admit
Both HSimplE and HypE are trained using stochastic gradient                 a simple extension to higher-binary relations for the link pre-
descent with mini-batches. In each learning iteration, we take              diction task. The baselines for this task are grouped into the
a batch of positive tuples from the knowledge hypergraph.                   following categories: (1) methods that work with binary re-
As we only have positive instances available, we need to also               lations and that are easily extendable to higher-arity, namely
train our model on negative instances; thus, for each positive              r-SimplE, m-DistMult, and m-CP; (2) existing methods that
instance, we produce a set of negative instances. For nega-                 can handle higher-arity relations, namely m-TransH. Below
tive sample generation, we follow the contrastive approach of               we give some details about methods in category (1).
[Bordes et al., 2013] for knowledge graphs and extend it to
knowledge hypergraphs: for each tuple, we produce a set of                  r-SimplE. To test the performance of a model trained on
negative samples of size N |r| by replacing each of the enti-               reified data, we convert higher-arity relations in the train set
ties with N random entities in the tuple, one at a time. Here,              to binary relations through reification. We then use SimplE
N is the ratio of negative samples in our training set and is a             (that we call r-SimplE) on this reified data. In this setting, at
hyperparameter.                                                             test time higher-arity relations are first reified to a set of bi-
    Given a knowledge hypergraph defined on τ 0 , we let τtrain
                                                             0
                                                                  ,         nary relations; this process creates new auxiliary entities for
  0          0
τtest , and τvalid denote the train, test, and validation sets, re-         which the model has no learned embeddings. To embed the
                                                    JF17K                             FB- AUTO                          M -FB15K

      Model                       MRR Hit@1 Hit@3 Hit@10 MRR Hit@1 Hit@3 Hit@10 MRR Hit@1 Hit@3 Hit@10
      r-SimplE                    0.102 0.069 0.112 0.168 0.106 0.082 0.115 0.147 0.051 0.042 0.054 0.070
      m-DistMult                  0.463 0.372 0.510 0.634 0.784 0.745 0.815 0.845 0.705 0.633 0.740 0.844
      m-CP                        0.391 0.298 0.443 0.563 0.752 0.704 0.785 0.837 0.680 0.605 0.715 0.828
      m-TransH [Wen et al., 2016] 0.444 0.370 0.475 0.581 0.728 0.727 0.728 0.728 0.623 0.531 0.669 0.809
      HSimplE (Ours)              0.472 0.378 0.520 0.645 0.798 0.766 0.821 0.855 0.730 0.664 0.763 0.859
      HypE (Ours)                 0.494 0.408 0.538 0.656 0.804 0.774 0.823 0.856 0.777 0.725 0.800 0.881

Table 1: Knowledge hypergraph completion results on JF17K, FB- AUTO and M -FB15K for baselines and the proposed method. The prefixes
‘r’ and ‘m’ in the model names stand for reification and multi-arity respectively. Both our methods outperform the baselines on all datasets.


         Dataset   |E|   |R|   #train #valid #test
       WN18      40,943 18 141,442 5,000 5,000
       FB15k     14,951 1,345 483,142 50,000 59,071
       JF17K     29,177 327 77,733      –    24,915
       FB- AUTO 3,410      8   6,778 2,255 2,180
       M -FB15K 10,314    71 415,375 39,348 38,797

                     Table 2: Dataset Statistics.


auxiliary entities for the prediction step, we use the observa-
tion we have about them at test time. For example, a higher-                                                  (a)
arity relation r(e1 , e2 , e3 ) is reified at test time by adding a
new entity e0 and converting the higher-arity tuple to three
binary facts: r1 (e0 , e1 ), r2 (e0 , e2 ), and r3 (e0 , e3 ). When pre-
dicting the tail entity of r1 (e0 , ?), we use the other two reified
facts to learn an embedding for entity e0 . Because e0 is added
only to help represent the higher-arity relations as a set of bi-
nary relations, we only need to do tail prediction for reified
relations. Note that we do not reify binary relations.
m-DistMult. DistMult [Yang et al., 2015] defines a
score function φ(r(ei , ej )) =                 (r, ei , ej ). To accom-
                                                                                                              (b)
modate non-binary relations, we redefine this function as
φ(r(ei , . . . , ej )) = (r, ei , . . . , ej ).                            Figure 3: These experiments show that HypE outperforms HSimplE
m-CP. Canonical Polyadic decomposition [Hitchcock,                         when trained with fewer parameters, and when tested on samples
1927] is a tensor decomposition approach. We refer to the                  that contain at least one entity in a position never encountered dur-
version that only handles binary relations as CP. CP em-                   ing training. (a) MRR of HypE and HSimplE for different embed-
                                                                           ding dimensions. (b) Results of m-CP, HSimplE, and HypE on the
beds each entity e as two vectors e(1) and e(2) , and each                 missing positions test set (containing 1,806 test samples).
relation r as a single vector r; it defines the score func-
                                (1) (2)
tion φ(r(ei , ej )) = (r, ei , ej ). We extend CP to m-
CP, which accommodates relations of any arity. m-CP em-                    the set of corrupted tuples, plus r(e1 , . . . , ek ), be denoted
beds each entity e as α different vectors e(1) , .., e(α) , where          by θi (r(e1 , . . . , ek )). Let ranki (r(e1 , . . . , ek )) be the rank-
α = maxr∈R (|r|); it computes the score of a tuple as                      ing of r(e1 , . . . , ek ) within θi (r(e1 , . . . , ek )) based on the
                               (1)           (|r|)                         score φ(x) for each x ∈ θi (r(e1 , . . . , ek )). We compute
φ(r(ei , . . . , ej )) = (r, ei , ..., ej ).                                               1
                                                                                               P                       Pk             1
                                                                           the MRR as K                           0
                                                                                                  r(e1 ,...,ek )∈τtest  i=1 ranki r(e1 ,...,ek ) where
                                                                                  P
5.3    Evaluation Metrics                                                  K = r(e1 ,...ek )∈τ 0 |r| is the number of prediction tasks.
                                                                                                     test
                                                                                                                                        0
Given a knowledge hypergraph on τ 0 , we evaluate various                  Hit@t measures the proportion of tuples in τtest                  that rank
                                                        0          0       among the top t in their corresponding corrupted sets. We
completion methods using a train and test set τtrain         and τtest .
We use two evaluation metrics: Hit@t and Mean Recipro-                     follow [Bordes et al., 2013] and remove all corrupted tuples
cal Rank (MRR). Both these measures rely on the ranking                    that are in τ 0 from our computation of MRR and Hit@t.
                    0
of a tuple x ∈ τtest       within a set of corrupted tuples. For
                                        0
each tuple r(e1 , . . . , ek ) in τtest      and each entity position i    6       Experiments
in the tuple, we generate |E| − 1 corrupted tuples by re-
placing the entity ei with each of the entities in E \ {ei }.              This section summarizes our experiments 1 .
For example, by corrupting entity ei , we would obtain a
new tuple r(e1 , . . . , eci , . . . , ek ) where eci ∈ E \ {ei }. Let         1
                                                                                   Code and data available at https://github.com/ElementAI/HypE.
                                                                            WN18                            FB15k
        Model                                              MRR Hit@1 Hit@3 Hit@10 MRR Hit@1 Hit@3 Hit@10
        CP [Hitchcock, 1927]                               0.074 0.049 0.080 0.125 0.326 0.219 0.376 0.532
        TransH [Wang et al., 2014]                           -     -     -   0.867   -     -     -   0.585
        m-TransH [Wen et al., 2016]                        0.671 0.495 0.839 0.923 0.351 0.228 0.427 0.559
        DistMult [Yang et al., 2015]                       0.822 0.728 0.914 0.936 0.654 0.546 0.733 0.824
        HSimplE (Ours) and SimplE [Kazemi and Poole, 2018] 0.942 0.939 0.944 0.947 0.727 0.660 0.773 0.838
        HypE (Ours)                                        0.934 0.927 0.940 0.944 0.725 0.648 0.777 0.856

Table 3: Knowledge graph completion results on WN18 and FB15K for baselines and HypE. Note that in knowledge graphs (binary relations),
HSimplE and SimplE are equivalent, both theoretically and experimentally. The results show that our methods outperform the baselines.


                                         Arity                        at least one entity in a position it never appears in within the
  Model                          2     3    4-5-6 All                 training dataset. The results on these experiments (Figure 3b)
  r-SimplE                     0.478 0.025 0.017 0.168                show that (1) both HSimplE and HypE outperform m-CP
  m-DistMult                   0.495 0.648 0.809 0.634                (which learns different embeddings for each entity-position
  m-CP                         0.409 0.563 0.765 0.560                pair), and more importantly, (2) HypE significantly outper-
  m-TransH [Wen et al., 2016] 0.411 0.617 0.826 0.596                 forms HSimplE for this challenging test set, leading us to
  HSimplE (Ours)               0.497 0.699 0.745 0.645                believe that disentangling entity and position representations
  HypE (Ours)                  0.466 0.693 0.858 0.656                may be a better strategy for this scenario.
  # train tuples              36,293 18,846 6,772 61,911
  # test tuples               10,758 10,736 3,421 24,915              6.2    Knowledge Graph Completion Results
                                                                      To show that HSimplE and HypE work well also on the more
Table 4: Breakdown performance of Hit@10 across relations with        common knowledge graphs, we evaluate them on WN18 and
different arities on JF17K dataset along with their statistics.       FB15K. Table 3 shows link prediction results on WN18 and
                                                                      FB15K. Baseline results are taken from the original papers
6.1   Knowledge Hypergraph Completion Results                         except that of m-TransH, which we implement ourselves. In-
                                                                      stead of tuning the parameters of HypE to get potentially bet-
The results of our experiments, summarized in Table 1,                ter results, we follow the Kazemi and Poole (2018) setup with
show that both HSimplE and HypE outperform the proposed               the same grid search approach by setting n = 2, l = 2, and
baselines across the three datasets JF17K, FB- AUTO, and              s = 2. This results in all models in Table 3 having the same
M -FB15K. They further demonstrate that reification for
                                                                      number of parameters, and thus makes them directly compa-
the r-SimplE model does not work well; this is because the            rable to each other. Note that for knowledge graph comple-
reification process introduces auxiliary entities for which the       tion (all binary relations) HSimplE is equivalent to SimplE,
model does not learn appropriate embeddings because these             both theoretically and experimentally (as shown in Section 4).
auxiliary entities appear in very few facts. Comparing the            The results show that on WN18 and FB15K, HSimplE and
results of r-SimplE against HSimplE, we can also see that ex-         HypE outperform all baselines.
tending a model to work with hypergraphs works better than
reification when high-arity relations are present.                    6.3    Ablation Study on Different Arities
   The ability of knowledge sharing through the learned
                                                                      We break down the performance of models across different
position-dependent convolution filters suggests that HypE
                                                                      arities. As the number of test tuples in higher arities (4-5-6)
would need a lower number of parameters than HSimplE to
                                                                      is much less than in smaller arities (2-3), we used equiva-
obtain good results. To test this, we train both models with
                                                                      lent size bins to show the decomposed results for a reason-
different embedding dimensions. Figure 3a shows the MRR
                                                                      able number of test tuples. Table 4 shows the Hit@10 results
on the test set for each model with different embedding sizes.
                                                                      of the models for bins of arity 2, 3, and 4-5-6 in JF17K. The
Based on the MRR result, we can see that HypE outperforms
                                                                      proposed models outperform the state-of-the-art and the base-
HSimplE by 24% for embedding dimension 50, implying that
                                                                      lines in all arities. We highlight that r-SimplE and HSimplE
HypE works better under a constrained budget.
                                                                      are quite different models for relations having arity > 2.
   Disentangling the representations of entity embeddings
and positional filters enables HypE to better learn the role of
position within a relation because the learning process con-          7     Conclusions
siders the behavior of all entities that appear in a given po-        Knowledge hypergraph completion is an important problem
sition at the time of training. This becomes especially im-           that has received little attention. Having introduced two
portant in the case when some entities never appear in certain        new knowledge hypergraph datasets, baselines, and two new
positions in the train set, but you still want to be able to reason   methods for link prediction in knowledge hypergraphs, we
about them no matter what position they appear in at test time.       hope to kindle interest in the problem. Unlike graphs, hyper-
In order to test the effectiveness of our models in this more         graphs have a more complex structure that opens the door to
challenging scenario, we created a missing positions test set         more challenging questions such as: how do we effectively
by selecting the tuples from our original test set that contain       predict the missing entities in a given (partial) tuple?
References                                                       [Lacroix et al., 2018] Timothée Lacroix, Nicolas Usunier,
[Bollacker et al., 2008] Kurt Bollacker, Colin Evans,               and Guillaume Obozinski. Canonical tensor decomposi-
   Praveen Paritosh, Tim Sturge, and Jamie Taylor. Freebase:        tion for knowledge base completion. ICML, 2018.
   a collaboratively created graph database for structuring      [Nguyen, 2017] Dat Quoc Nguyen. An overview of embed-
   human knowledge. In ACM ICMD, 2008.                              ding models of entities and relationships for knowledge
[Bordes et al., 2013] Antoine Bordes, Nicolas Usunier,              base completion. arXiv preprint arXiv:1703.08098, 2017.
   Alberto Garcia-Duran, Jason Weston, and Oksana                [Nickel et al., 2011] Maximilian Nickel, Volker Tresp, and
   Yakhnenko.        Translating embeddings for modeling            Hans-Peter Kriegel. A three-way model for collective
   multi-relational data. In NIPS, 2013.                            learning on multi-relational data. In ICML, 2011.
[Bordes et al., 2014] Antoine Bordes, Xavier Glorot, Jason       [Nickel et al., 2016] Maximilian Nickel, Kevin Murphy,
   Weston, and Yoshua Bengio. A semantic matching energy            Volker Tresp, and Evgeniy Gabrilovich. A review of re-
   function for learning with multi-relational data. Machine        lational machine learning for knowledge graphs. IEEE,
   Learning, 2014.                                                  2016.
[Carlson et al., 2010] Andrew Carlson, Justin Betteridge,        [Paszke et al., 2019] Adam Paszke, Sam Gross, Francisco
   Bryan Kisiel, Burr Settles, Estevam R Hruschka Jr, and           Massa, Adam Lerer, James Bradbury, Gregory Chanan,
   Tom M Mitchell. Toward an architecture for never-ending          Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca
   language learning. In AAAI, 2010.                                Antiga, Alban Desmaison, Andreas Kopf, Edward Yang,
[De Raedt et al., 2007] Luc De Raedt, Angelika Kimmig,              Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank
                                                                    Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and
   and Hannu Toivonen. Problog: A probabilistic prolog and
                                                                    Soumith Chintala. Pytorch: An imperative style, high-
   its application in link discovery. IJCAI, 2007.
                                                                    performance deep learning library. In NeurIPS. 2019.
[Fatemi et al., 2019] Bahare Fatemi, Siamak Ravanbakhsh,
                                                                 [Singhal, 2012] Amit Singhal. Introducing the knowledge
   and David Poole. Improved knowledge graph embedding
                                                                    graph: things, not strings, 2012.
   using background taxonomic information. In AAAI, 2019.
                                                                 [Socher et al., 2013] Richard Socher, Danqi Chen, Christo-
[Feng et al., 2019] Yifan Feng, Haoxuan You, Zizhao
                                                                    pher D Manning, and Andrew Ng. Reasoning with neural
   Zhang, Rongrong Ji, and Yue Gao. Hypergraph neural net-          tensor networks for knowledge base completion. In NIPS,
   works. In AAAI, 2019.                                            2013.
[Ferrucci et al., 2010] David Ferrucci, Eric Brown, Jennifer     [Trouillon et al., 2016] Théo Trouillon, Johannes Welbl, Se-
   Chu-Carroll, James Fan, David Gondek, Aditya A Kalyan-           bastian Riedel, Éric Gaussier, and Guillaume Bouchard.
   pur, Adam Lally, J William Murdock, Eric Nyberg, John            Complex embeddings for simple link prediction. In ICML,
   Prager, et al. Building watson: An overview of the deepqa        2016.
   project. AI magazine, 2010.
                                                                 [Trouillon et al., 2017] Théo Trouillon, Christopher R
[Guan et al., 2019] Saiping Guan, Xiaolong Jin, Yuanzhuo
                                                                    Dance, Éric Gaussier, Johannes Welbl, Sebastian Riedel,
   Wang, and Xueqi Cheng. Link prediction on n-ary rela-
                                                                    and Guillaume Bouchard. Knowledge graph completion
   tional data. In WWW, 2019.
                                                                    via complex tensor factorization. JMLR, 2017.
[Hitchcock, 1927] Frank L Hitchcock. The expression of a
                                                                 [Wang et al., 2014] Zhen Wang, Jianwen Zhang, Jianlin
   tensor or a polyadic as a sum of products. Journal of Math-
                                                                    Feng, and Zheng Chen. Knowledge graph embedding by
   ematics and Physics, 6(1-4):164–189, 1927.
                                                                    translating on hyperplanes. In AAAI, 2014.
[Huang et al., 2009] Yuchi Huang, Qingshan Liu, and Dim-
                                                                 [Wen et al., 2016] Jianfeng Wen, Jianxin Li, Yongyi Mao,
   itris Metaxas. Video object segmentation by hypergraph
                                                                    Shini Chen, and Richong Zhang. On the representation
   cut. In CVPR, 2009.
                                                                    and embedding of knowledge bases beyond binary rela-
[Huang et al., 2010] Yuchi Huang, Qingshan Liu, Shaoting            tions. In IJCAI, 2016.
   Zhang, and Dimitris N Metaxas. Image retrieval via prob-      [Xu et al., 2018] Keyulu Xu, Weihua Hu, Jure Leskovec, and
   abilistic hypergraph ranking. In CVPR, 2010.                     Stefanie Jegelka. How powerful are graph neural net-
[Kazemi and Poole, 2018] Seyed Mehran Kazemi and David              works? arXiv preprint arXiv:1810.00826, 2018.
   Poole. Simple embedding for link prediction in knowledge      [Yadati et al., 2018] Naganand         Yadati,       Madhav
   graphs. In NIPS, 2018.                                           Nimishakavi, Prateek Yadav, Anand Louis, and Partha
[Kazemi et al., 2014] Seyed Mehran Kazemi, David Buch-              Talukdar. Hypergcn: Hypergraph convolutional net-
   man, Kristian Kersting, Sriraam Natarajan, and David             works for semi-supervised classification. arXiv preprint
   Poole. Relational logistic regression. In KR, 2014.              arXiv:1809.02589, 2018.
[Kazemi et al., 2020] Seyed Mehran Kazemi, Rishab Goel,          [Yang et al., 2015] Bishan Yang, Wen tau Yih, Xiaodong He,
   Kshitij Jain, Ivan Kobyzev, Akshay Sethi, Peter Forsyth,         Jianfeng Gao, and Li Deng. Embedding entities and re-
   and Pascal Poupart. Representation Learning for Dynamic          lations for learning and inference in knowledge bases.
   Graphs: A Survey. In JMLR, 2020.                                 CoRR, abs/1412.6575, 2015.
Supplementary Material                                             Database Statistics
Dataset Creation                                                   Table 5 summarizes the statistics of the datasets.
We downloaded F REEBASE [Bollacker et al., 2008] from              Theorems
https://developers.google.com/freebase. Note that F REE -
                                                                   Theorem 2 (Expressivity of HypE). Let τ be a set of true
BASE is a reified dataset; that is, it is created from a knowl-
                                                                   tuples defined over entities E and relations R, and let α =
edge base having facts with relations defined on two or more
                                                                   maxr∈R (|r|) be the maximum arity of the relations in R.
entities. Relation names in F REEBASE have three parts sep-
                                                                   There exists a HypE model with embedding vectors of size
arated with a ‘.’ e.g., film.performance.actor. The
                                                                   max(α|τ |, α) that assigns 1 to the tuples in τ and 0 to others.
first part in the relation shows the subject of this triple, the
second part shows the actual relation, and the third part shows
the role in the relation.                                          Proof. To prove the theorem, we show an assignment of em-
   Here are three examples of binary relations in F REEBASE        bedding values for each entity and relation in τ such that the
with the subject of ”film”.                                        scoring function of HypE is as follows:
                                                                                             
   • film.performance.actor(/m/04q4bd8,                                                        1 if x ∈ τ
                                                                                     φ(x) =
     /m/07yyhx)                                                                                0 otherwise
   • film.performance.character(/m/04q4bd8,                            We begin the proof by first describing the embeddings of
     /m/0qzpnz5)                                                   each of the entities and relations in HypE; we then proceed to
   • film.performance.film(/m/04q4bd8,                             show that with such an embedding, HypE can represent any
     /m/04mxv5p)                                                   world accurately.
From the Knowledge Graph Search API Link,                              Let us first assume that |τ | > 0 and let fp be the pth fact
(https://developers.google.com/knowledge-graph/reference/          in τ . We let each entity e ∈ E be represented with a vector
rest/v1/?apix=true&apix params=%7B%22ids%22%3A%                    of length α|τ | in which the pth block of α-bits is the one-
5B%22%2Fm%2F04mxv5p%22%5D%7D) the entities                         hot representation of e in fact fp : if e appears in fact fp at
/m/07yyhx, /m/0qzpnz5, and /m/04mxv5p refer, re-                   position i, then the ith bit of the pth block is set to 1, and
spectively, to actress “Kathlyn Williams”, character “Queen        to 0 otherwise. Each relation r ∈ R is then represented as
Isabel of Bourbon”, and movie “The Spanish Dancer”. The            a vector of length τ whose pth bit is equal to 1 if fact fp is
entity /m/04q4bd8 is a reified entity.                             defined on relation r, and 0 otherwise.
   To obtain a knowledge hypergraph H from F REEBASE, we               HypE defines convolutional weight filters for each entity
perform an inverse reification process by following the steps      position within a tuple. As we have at most α possible po-
below.                                                             sitions, we define each convolutional filter ωi as a vector of
                                                                   length α where the ith bit is set to 1 and all others to 0, for
 1. From F REEBASE, remove the facts that have relations           each i = 1, 2, . . . , α. When the scoring function φ is applied
    defined on a single entity, or that contain numbers or         to some tuple x, for each entity position i in x, convolution
    enumeration as entities.                                       filter ωi is applied to the entity at position i in the tuple as
 2. Join the triples in F REEBASE that have the same subject       a first step; the () function is then applied to the resulting
    and actual relation, and the same entity. The role (the 3rd    vector and the relation embedding to obtain a score.
    part of the relation) provides the position is the resulting       Given any tuple x, we want to show that φ(x) = 1 if x ∈ τ
    relation.                                                      and 0 otherwise.
                                                                       First assume that x = fp is the pth fact in τ that is defined
    For example, we convert the three tuples above to a tuple      on relation r and entities where ei is the entity at position
    of relation of arity 3 as:                                     i. Convolving each ei with ωi results in a vector of length |τ |
             film.performance(/m/07yyhx,                           where the pth bit is equal to 1 (since both ωi and the pth block
               /m/0qzpnz5, /m/04mxv5p)                             of ei have a 1 at the ith position) (See Figure 4. Then, as a first
                                                                   step, function () computes the element-wise multiplication
     with the first, second, and third positions representing      between the embedding of relation r (that has 1 at position
     the actor, character, and movie of the performance re-        p) and all of the convolved entity vectors (each having 1 at
     spectively. in H.                                             position p); this results in a vector of length |τ | where the pth
 3. Create the FB- AUTO dataset by selecting the facts from        bit is set to 1 and all other bits set to 0. Finally, (()) sums
    H whose subject is ‘automotive’ (first part of relation        the outcome of the resulting products to give us a score of 1.
    names is ‘automative’) .                                           To show that φ(x) = 0 when x ∈     / τ , we prove the contra-
                                                                   positive, namely that if φ(x) = 1, then x must be a fact in
 4. Create the M -FB15K dataset by following a strategy            τ . We proceed by contradiction. Assume that there exists a
    similar to that proposed by [Bordes et al., 2013]: se-         tuple x ∈ / τ such that φ(x) = 1. This means that at the time
    lect the facts in H that pertain to entities present in the    of computing the element-wise product in the () function,
    Wikilinks database [Singh et al., 2012].                       there was a position j at which all input vectors to () had
 5. Split the facts in each of FB- AUTO and M -FB15K ran-          a value of 1. This can happen only when (1) applying the
    domly into train, test, and validation sets.                   convolution filter wj to each of the entities in x produces a
                                                                       number of tuples                                                               number of tuples with respective arity
                 Dataset    |E|   |R|   #train #valid #test #arity=2 #arity=3 #arity=4 #arity=5 #arity=6
                 WN18     40,943 18 141,442 5,000 5,000 151,442          0        0        0        0
                 FB15k    14,951 1,345 483,142 50,000 59,071 592,213     0        0        0        0
                 JF17K    29,177 327 77,733       –   24,915 56,322 34,550     9,509     2,230     37
                 FB- AUTO 3,410     8   6,778 2,255 2,180 3,786          0      215      7,212      0
                 M -FB15K 10,314   71 415,375 39,348 38,797 82,247 400,027       26     11,220      0

                                                                                   Table 5: Dataset Statistics.

                                                           τ                      𝛼

                            ej    ...   ...   ...    ...             ...    ...    ...          ...             0         1           0           0         ...     ...     ...         ...         ...     ...     ...     ...


                            ⍵2     0    1     0      0                                  ...         ...         1         ...         ...                   e j ∗ ⍵2              ...         ...     1       ...     ...



                        Figure 4: An example of an embedding where |τ | = 5, α = 4 and f3 is the third fact in τ


vector having 1 at position j, and (2) the embedding of rela-                                                                         We begin the proof by first describing the embeddings of
tion r ∈ x has 1 at position j.                                                                                                   each of the entities and relations in HSimplE; we then pro-
   The first case can happen only if all entities of x appear in                                                                  ceed to show that with such an embedding, HSimplE can rep-
the jth fact fj ∈ τ ; the second case happens only if relation                                                                    resent any world accurately.
r ∈ x appears in fj . But if all entities of x as well as its                                                                         Let us first assume that |τ | > 0. We let each entity e ∈ E
relation appear in fact fj , then x ∈ τ , contradicting our as-                                                                   be represented with a vector of length α|τ | in which the pth
sumption. Therefore, if φ(x) = 1, then x must be a fact in                                                                        block of |τ |-bits indicates entities that appeared in position
τ.                                                                                                                                p in the facts. In an entity embedding, the ith bit in the jth
   To complete the proof, we consider the case when |τ | = 0.                                                                     block is set to 1 if the entity appears in position j in the ith
In this case, since there are no facts, all entities and relations                                                                fact. Each relation r ∈ R is then represented as a vector of
are represented by zero-vectors of length α. Then, for any                                                                        length α ∗ τ whose pth bit is equal to 1 if fact fp is defined on
tuple x, φ(x) = 0. This completes the proof.                                                                                      relation r, and 0 otherwise.
                                                                                                                                      Given any tuple x, we want to show that φ(x) = 1 if x ∈ τ
Theorem 3 (Expressivity of HSimplE). Let τ be a set of                                                                            and 0 otherwise.
true tuples defined over entities E and relations R, and let                                                                          First assume that x = fi is the ith fact in τ that is defined
α = maxr∈R (|r|) be the maximum arity of the relations in                                                                         on relation r and entities e1 , e2 , . . . , where ep is the entity at
R. There exists an HSimplE model with embedding vectors                                                                           position p. Shifting each ep results in a vector where the ith
of size max(α|τ |, α) that assigns 1 to the tuples in τ and 0 to                                                                  bit is equal to 1 (See Figure 5). Then, as a first step, function
others.                                                                                                                              () computes the element-wise multiplication between the
Proof. To prove the theorem, we follow the same strategy as                                                                       embedding of relation r (that has 1 at position i) and all of
Theorem 1 by showing an assignment of embedding values                                                                            the shifted entity vectors (each having 1 at position i). Finally,
for each entity and relation in τ such that the scoring function                                                                     (()) sums the outcome of the resulting products to give us a
of HSimplE is as follows:                                                                                                         score of 1.
                                                                                                                                     To show that φ(x) = 0 when x ∈        / τ , we prove the contra-
                               1 if x ∈ τ                                                                                         positive, namely that if φ(x) = 1, then x must be a fact in
                   φ(x) =
                               0 otherwise                                                                                        τ . We proceed by contradiction. Assume that there exists a

                            f3= r(ei , ej , ek)       |τ| = 5                     𝛼=4

                            ei    ...   ...   1     ...        ...         ...    ...         0           ...       ...         ...         ...       0       ...     ...         ...         ...     ...    ...     ...


                            ej    ...   ...   0     ...        ...         ...    ...         1           ...       ...         ...         ...       0       ...     ...         ...         ...     ...    ...     ...


                            ek    ...   ...   0     ...        ...         ...    ...         0           ...       ...         ...         ...       1       ...     ...         ...         ...     ...    ...     ...



                            r     ...   ...   1     ...        ...         ...    ...         ...         ...       ...         ...         ...       ...     ...     ...         ...         ...     ...    ...     ...




                        Figure 5: An example of an embedding where |τ | = 5, α = 4 and f3 is the third fact in τ
tuple x ∈/ τ such that φ(x) = 1. This means that at the time
of computing the element-wise product in the () function,
there was a position i at which all input vectors to () had a
value of 1. This can happen only when (1) applying the shift
operation to each of the entities in x produces a vector having
1 at position i, and (2) the embedding of relation r ∈ x has 1
at position i.
   The first case can happen only if all entities of x appear in
the ith fact fi ∈ τ ; the second case happens only if relation
r ∈ x appears in fi . But if all entities and relations of x
appear in fact fi , then x ∈ τ , contradicting our assumption.
Therefore, if φ(x) = 1, then x must be a fact in τ .
   To complete the proof, we consider the case when |τ | = 0.
In this case, since there are no facts, all entities and relations
are represented by zero-vectors of length α. Then, for any
tuple x, φ(x) = 0. This completes the proof.

References
[Singh et al., 2012] Sameer Singh, Amarnag Subramanya,
   Fernando Pereira, and Andrew McCallum. Wikilinks: A
   large-scale cross-document coreference corpus labeled via
   links to wikipedia. University of Massachusetts, Amherst,
   Tech. Rep. UM-CS-2012, 15, 2012.

