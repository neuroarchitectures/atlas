# CANE Context aware network embedding for relation modeling Tu Liu Sun 2017

> Source: `CANE_Context_aware_network_embedding_for_relation_modeling_Tu_Liu_Sun_2017.pdf`

---

         CANE: Context-Aware Network Embedding for Relation Modeling

                       Cunchao Tu1,2∗ , Han Liu3∗, Zhiyuan Liu1,2†, Maosong Sun1,2
1 Department of Computer Science and Technology, State Key Lab on Intelligent Technology and Systems,

                National Lab for Information Science and Technology, Tsinghua University, China
     2 Jiangsu Collaborative Innovation Center for Language Ability, Jiangsu Normal University, China
                                             3 Northeastern University, China




                          Abstract                              and manage large-scale networks, alleviating the
                                                                computation and sparsity issues of conventional
     Network embedding (NE) is playing a                        symbol-based representations. Hence, NE is at-
     critical role in network analysis, due to                  tracting many research interests in recent years
     its ability to represent vertices with ef-                 (Perozzi et al., 2014; Tang et al., 2015; Grover and
     ficient low-dimensional embedding vec-                     Leskovec, 2016), and achieves promising perfor-
     tors. However, existing NE models aim to                   mance on many network analysis tasks including
     learn a fixed context-free embedding for                   link prediction, vertex classification, and commu-
     each vertex and neglect the diverse roles                  nity detection.
     when interacting with other vertices. In
     this paper, we assume that one vertex usu-                                        I am studying NLP problems,
     ally shows different aspects when interact-                                        including syntactic parsing,
                                                                                       machine translation and so on.
     ing with different neighbor vertices, and
     should own different embeddings respec-
     tively. Therefore, we present Context-
     Aware Network Embedding (CANE), a
     novel NE model to address this issue.
                                                                 My research focuses on typical                I am an NLP researcher in
     CANE learns context-aware embeddings                         NLP tasks, including word                  machine translation, especially
                                                                  segmentation, tagging and                  using deep learning models to
     for vertices with mutual attention mecha-                         syntactic parsing.                    improve machine translation.
     nism and is expected to model the seman-
     tic relationships between vertices more                    Figure 1: Example of a text-based information
     precisely. In experiments, we compare                      network. (Red, blue and green fonts represent con-
     our model with existing NE models on                       cerns of the left user, right user and both respec-
     three real-world datasets. Experimen-                      tively.)
     tal results show that CANE achieves sig-
     nificant improvement than state-of-the-art                    In real-world social networks, it is intuitive that
     methods on link prediction and compara-                    one vertex may demonstrate various aspects when
     ble performance on vertex classification.                  interacting with different neighbor vertices. For
     The source code and datasets can be ob-                    example, a researcher usually collaborates with
     tained from https://github.com/                            various partners on diverse research topics (as il-
     thunlp/CANE.                                               lustrated in Fig. 1), a social-media user contacts
 1   Introduction                                               with various friends sharing distinct interests, and
                                                                a web page links to multiple pages for different
 Network embedding (NE), i.e., network represen-                purposes. However, most existing NE methods
 tation learning (NRL), aims to map vertices of a               only arrange one single embedding vector to each
 network into a low-dimensional space according                 vertex, and give rise to the following two invertible
 to their structural roles in the network. NE pro-              issues: (1) These methods cannot flexibly cope
 vides an efficient and effective way to represent              with the aspect transition of a vertex when inter-
     ∗
         Indicates equal contribution                           acting with different neighbors. (2) In these mod-
     †
         Corresponding Author: Z. Liu (liuzy@tsinghua.edu.cn)   els, a vertex tends to force the embeddings of its
neighbors close to each other, which may be not         node2vec (Grover and Leskovec, 2016).
the case all the time. For example, the left user          We conduct experiments on three real-world
and right user in Fig. 1, share less common inter-      datasets of different areas. Experimental results
ests, but are learned to be close to each other since   on link prediction reveal the effectiveness of our
they both link to the middle person. This will ac-      framework as compared to other state-of-the-art
cordingly make vertex embeddings indiscrimina-          methods. The results suggest that context-aware
tive.                                                   embeddings are critical for network analysis, in
   To address these issues, we aim to propose           particular for those tasks concerning about com-
a Context-Aware Network Embedding (CANE)                plicated interactions between vertices such as link
framework for modeling relationships between            prediction. We also explore the performance of
vertices precisely. More specifically, we present       our framework via vertex classification and case
CANE on information networks, where each ver-           studies, which again confirms the flexibility and
tex also contains rich external information such as     superiority of our models.
text, labels or other meta-data, and the significance
of context is more critical for NE in this scenario.    2   Related Work
Without loss of generality, we implement CANE
                                                        With the rapid growth of large-scale social net-
on text-based information networks in this paper,
                                                        works, network embedding, i.e. network repre-
which can easily extend to other types of informa-
                                                        sentation learning has been proposed as a critical
tion networks.
                                                        technique for network analysis tasks.
   In conventional NE models, each vertex is rep-          In recent years, there have been a large num-
resented as a static embedding vector, denoted as       ber of NE models proposed to learn efficient ver-
context-free embedding. On the contrary, CANE           tex embeddings (Tang and Liu, 2009; Cao et al.,
assigns dynamic embeddings to a vertex according        2015; Wang et al., 2016; Tu et al., 2016a). For ex-
to different neighbors it interacts with, named as      ample, DeepWalk (Perozzi et al., 2014) performs
context-aware embedding. Take a vertex u and            random walks over networks and introduces an ef-
its neighbor vertex v for example. The context-         ficient word representation learning model, Skip-
free embedding of u remains unchanged when in-          Gram (Mikolov et al., 2013a), to learn network
teracting with different neighbors. On the con-         embeddings. LINE (Tang et al., 2015) optimizes
trary, the context-aware embedding of u is dy-          the joint and conditional probabilities of edges
namic when confronting different neighbors.             in large-scale networks to learn vertex represen-
   When u interacting with v, their context em-         tations. Node2vec (Grover and Leskovec, 2016)
beddings concerning each other are derived from         modifies the random walk strategy in DeepWalk
their text information, Su and Sv respectively. For     into biased random walks to explore the network
each vertex, we can easily use neural models, such      structure more efficiently. Nevertheless, most of
as convolutional neural networks (Blunsom et al.,       these NE models only encode the structural infor-
2014; Johnson and Zhang, 2014; Kim, 2014) and           mation into vertex embeddings, without consider-
recurrent neural networks (Kiros et al., 2015; Tai      ing heterogeneous information accompanied with
et al., 2015), to build context-free text-based em-     vertices in real-world social networks.
bedding. In order to realize context-aware text-           To address this issue, researchers make great
based embeddings, we introduce the selective at-        efforts to incorporate heterogeneous information
tention scheme and build mutual attention be-           into conventional NE models. For instance,
tween u and v into these neural models. The mu-         Yang et al. (2015) present text-associated Deep-
tual attention is expected to guide neural models       Walk (TADW) to improve matrix factorization
to emphasize those words that are focused by its        based DeepWalk with text information.           Tu
neighbor vertices and eventually obtain context-        et al. (2016b) propose max-margin DeepWalk
aware embeddings.                                       (MMDW) to learn discriminative network rep-
   Both context-free embeddings and context-            resentations by utilizing labeling information of
aware embeddings of each vertex can be effi-            vertices. Chen et al. (2016) introduce group-
ciently learned together via concatenation using        enhanced network embedding (GENE) to inte-
existing NE methods such as DeepWalk (Per-              grate existing group information in NE. Sun et
ozzi et al., 2014), LINE (Tang et al., 2015) and        al. (2016) regard text content as a special kind
of vertices, and propose context-enhanced net-          these embeddings, we can simply concatenate
work embedding (CENE) through leveraging both           them and obtain the vertex embeddings as v =
structural and textural information to learn net-       vs ⊕ vt , where ⊕ indicates the concatenation op-
work embeddings.                                        eration. Note that, the text-based embedding vt
   To the best of our knowledge, all existing NE        can be either context-free or context-aware, which
models focus on learning context-free embed-            will be introduced detailedly in section 4.4 and 4.5
dings, but ignore the diverse roles when a vertex       respectively. When vt is context-aware, the over-
interacts with others. In contrast, we assume that      all vertex embeddings v will be context-aware as
a vertex has different embeddings according to          well.
which vertex it interacts with, and propose CANE           With above definitions, CANE aims to maxi-
to learn context-aware vertex embeddings.               mize the overall objective of edges as follows:
                                                                                X
3     Problem Formulation                                                 L=        L(e).                (1)
                                                                                e∈E
We first give basic notations and definitions in this
work. Suppose there is an information network           Here, the objective of each edge L(e) consists of
G = (V, E, T ), where V is the set of vertices,         two parts as follows:
E ⊆ V × V are edges between vertices, and T de-
notes the text information of vertices. Each edge                     L(e) = Ls (e) + Lt (e),             (2)
eu,v ∈ E represents the relationship between two
vertices (u, v), with an associated weight wu,v .       where Ls (e) denotes the structure-based objective
Here, the text information of a specific vertex         and Lt (e) represents the text-based objective.
v ∈ V is represented as a word sequence Sv =              In the following part, we give the detailed intro-
(w1 , w2 , . . . , wnv ), where nv = |Sv |. NRL aims    duction to the two objectives respectively.
to learn a low-dimensional embedding v ∈ Rd for         4.2   Structure-based Objective
each vertex v ∈ V according to its network struc-
                                                        Without loss of generality, we assume the network
ture and associated information, e.g. text and la-
                                                        is directed, as an undirected edge can be consid-
bels. Note that, d  |V | is the dimension of rep-
                                                        ered as two directed edges with opposite directions
resentation space.
                                                        and equal weights.
   Definition 1. Context-free Embeddings: Con-
                                                           Thus, the structure-based objective aims to
ventional NRL models learn context-free embed-
                                                        measure the log-likelihood of a directed edge us-
ding for each vertex. It means the embedding of
                                                        ing the structure-based embeddings as
a vertex is fixed and won’t change with respect to
its context information (i.e., another vertex it in-               Ls (e) = wu,v log p(vs |us ).          (3)
teracts with).
   Definition 2. Context-aware Embeddings:                Following LINE (Tang et al., 2015), we define
Different from existing NRL models that learn           the conditional probability of v generated by u in
context-free embeddings, CANE learns various            Eq. (3) as
embeddings for a vertex according to its differ-
ent contexts. Specifically, for an edge eu,v , CANE                               exp(us · vs )
                                                                p(vs |us ) = P              s   s
                                                                                                  .       (4)
learns context-aware embeddings v(u) and u(v) .                                  z∈V exp(u · z )

4     The Method                                        4.3   Text-based Objective
                                                        Vertices in real-world social networks usually ac-
4.1    Overall Framework
                                                        company with associated text information. There-
To take full use of both network structure and as-      fore, we propose the text-based objective to take
sociated text information, we propose two types         advantage of these text information, as well as
of embeddings for a vertex v, i.e., structure-          learn text-based embeddings for vertices.
based embedding vs and text-based embedding                The text-based objective Lt (e) can be defined
vt . Structure-based embedding can capture the          with various measurements. To be compatible
information in the network structure, while text-       with Ls (e), we define Lt (e) as follows:
based embedding can capture the textual mean-
ings lying in the associated text information. With      Lt (e) = α · Ltt (e) + β · Lts (e) + γ · Lst (e), (5)
where α, β and γ control the weights of various             where Si:i+l−1 denotes the concatenation of word
parts, and                                                  embeddings within the i-th window and b is the
                                                            bias vector. Note that, we add zero padding vec-
            Ltt (e) = wu,v log p(vt |ut ),                  tors (Hu et al., 2014) at the edge of the sentence.
            Lts (e) = wu,v log p(vt |us ),           (6)       Max-pooling. To obtain the text embedding vt ,
                                     s   t                  we operate max-pooling and non-linear transfor-
            Lst (e) = wu,v log p(v |u ).
                                                            mation over {xi0 , . . . , xin } as follows:
The conditional probabilities in Eq. (6) map the
two types of vertex embeddings into the same rep-                         ri = tanh(max(xi0 , . . . , xin )),                (8)
resentation space, but do not enforce them to be
identical for the consideration of their own char-             At last, we encode the text information of a ver-
acteristics. Similarly, we employ softmax function          tex with CNN and obtain its text-based embedding
for calculating the probabilities, as in Eq. (4).           vt = [r1 , . . . , rd ]T . As vt is irrelevant to the other
   The structure-based embeddings are regarded as           vertices it interacts with, we name it as context-
parameters, the same as in conventional NE mod-             free text embedding.
els. But for text-based embeddings, we intend to
                                                            4.5    Context-Aware Text Embedding
obtain them from associated text information of
vertices. Besides, the text-based embeddings can
                                                                Text
be obtained either in context-free ways or context-           Embedding          ut(v)=P·ap                   vt(u)=Q·aq

aware ones. In the following sections, we will give
                                                                          ap                       F                         aq
detailed introduction respectively.
                                                                                                             Column-pooling +
                                                                                                                 softmax
4.4   Context-Free Text Embedding                                              Row-pooling +
                                                                                 softmax
There has been a variety of neural models to obtain
text embeddings from a word sequence, such as                                                   tanh(PTAQ)
convolutional neural networks (CNN) (Blunsom
et al., 2014; Johnson and Zhang, 2014; Kim, 2014)
and recurrent neural networks (RNN) (Kiros et al.,
                                                                          P                                                  Q
2015; Tai et al., 2015).
   In this work, we investigate different neural net-                                               A
works for text modeling, including CNN, Bidi-                                   Convolutional                Convolutional
                                                                                    Unit                         Unit
rectional RNN (Schuster and Paliwal, 1997) and
GRU (Cho et al., 2014), and employ the best per-                                    Text                         Text
formed CNN, which can capture the local seman-                                   Description                  Description

tic dependency among words.
   Taking the word sequence of a vertex as input,                 Edge               u                            v
CNN obtains the text-based embedding through
three layers, i.e. looking-up, convolution and              Figure 2: An illustration of context-aware text em-
pooling.                                                    bedding.
   Looking-up. Given a word sequence S =
(w1 , w2 , . . . , wn ), the looking-up layer transforms       As stated before, we assume that a specific ver-
each word wi ∈ S into its corresponding word                tex plays different roles when interacting with oth-
                            0
embedding wi ∈ Rd and obtains embedding se-                 ers vertices. In other words, each vertex should
quence as S = (w1 , w2 , . . . , wn ). Here, d0 indi-       have its own points of focus about a specific ver-
cates the dimension of word embeddings.                     tex, which leads to its context-aware text embed-
   Convolution. After looking-up, the convolu-              dings.
tion layer extracts local features of input embed-             To achieve this, we employ mutual attention
ding sequence S. To be specific, it performs con-           to obtain context-aware text embedding. The sim-
volution operation over a sliding window of length          ilar technique was originally applied in question
                                                 0
l using a convolution matrix C ∈ Rd×(l×d ) as fol-          answering as attentive pooling (dos Santos et al.,
lows:                                                       2016). It enables the pooling layer in CNN to be
                   xi = C · Si:i+l−1 + b,             (7)   aware of the vertex pair in an edge, in a way that
text information from a vertex can directly affect    4.6      Optimization of CANE
the text embedding of the other vertex, and vice      According to Eq. (3) and Eq. (6), CANE aims
versa.                                                to maximize several conditional probabilities be-
   In Fig. 2, we give an illustration of the gen-     tween u ∈ {us , ut(v) } and v ∈ {vs , v(u)t }. It
erating process of context-aware text embedding.      is intuitive that optimizing the conditional prob-
Given an edge eu,v with two corresponding text        ability using softmax function is computation-
sequences Su and Sv , we can get the matrices         ally expensive. Thus, we employ negative sam-
P ∈ Rd×m and Q ∈ Rd×n through convolution             pling (Mikolov et al., 2013b) and transform the
layer. Here, m and n represent the lengths of Su      objective into the following form:
and Sv respectively. By introducing an attentive
                                                                            k
matrix A ∈ Rd×d , we compute the correlation ma-                            X
                                                       log σ(uT ·v)+               Ez∼P (v) [log σ(−uT ·z)], (13)
trix F ∈ Rm×n as follows:
                                                                             i=1

               F = tanh(PT AQ).                 (9)   where k is the number of negative samples and σ
                                                      represents the sigmoid function. P (v) ∝ dv 3/4
Note that, each element Fi,j in F represents the      denotes the distribution of vertices, where dv is the
pair-wise correlation score between two hidden        out-degree of v.
vectors, i.e., Pi and Qj .                               Afterward, we employ Adam (Kingma and Ba,
                                                      2015) to optimize the transformed objective. Note
  After that, we conduct pooling operations along
                                                      that, CANE is exactly capable of zero-shot scenar-
rows and columns of F to generate the importance
                                                      ios, by generating text embeddings of new vertices
vectors, named as row-pooling and column pool-
                                                      with well-trained CNN.
ing respectively. According to our experiments,
mean-pooling performs better than max-pooling.        5       Experiments
Thus, we employ mean-pooling operation as fol-
lows:                                                 To investigate the effectiveness of CANE on mod-
                                                      eling relationships between vertices, we conduct
          gip = mean(Fi,1 , . . . , Fi,n ),           experiments of link prediction on several real-
                                              (10)    world datasets. Besides, we also employ vertex
          giq = mean(F1,i , . . . , Fm,i ).
                                                      classification to verify whether context-aware em-
                                                      beddings of a vertex can compose a high-quality
   The importance vectors of P and Q are ob-
                                                      context-free embedding in return.
tained as gp = [g1p , . . . , gm p T
                                   ] and gq =
  q            q T
[g1 , . . . , gn ] .                                  5.1      Datasets
   Next, we employ softmax function to transform
importance vectors gp and gq to attention vectors                Datasets          Cora   HepTh    Zhihu
ap and aq . For instance, the i-th element of ap is              #Vertices     2, 277     1, 038   10, 000
formalized as follows:                                           #Edges        5, 214     1, 990   43, 894
                                                                 #Labels            7         −         −

             p      exp(gip )
            ai = P             p .            (11)                Table 1: Statistics of Datasets.
                  j∈[1,m] exp(gj )
                                                         We select three real-world network datasets as
  At last, the context-aware text embeddings of u     follows:
and v are computed as                                    Cora1 is a typical paper citation network con-
                                                      structed by (McCallum et al., 2000). After filter-
                   ut(v) = Pap ,                      ing out papers without text information, there are
                    t
                                              (12)    2, 277 machine learning papers in this network,
                   v(u) = Qaq .
                                                      which are divided into 7 categories.
                                                         HepTh2 (High Energy Physics Theory) is
   Now, given an edge (u, v), we can obtain the       another citation network from arXiv3 released
context-aware embeddings of vertices with their           1
                                                            https://people.cs.umass.edu/∼mccallum/data.html
structure embeddings and context-aware text em-           2
                                                            https://snap.stanford.edu/data/cit-HepTh.html
beddings as u(v) = us ⊕ut(v) and v(u) = vs ⊕v(u)
                                             t .          3
                                                            https://arxiv.org/
                 %Training edges     15%    25%    35%    45%    55%    65%    75%    85%    95%
                     MMB             54.7   57.1   59.5   61.9   64.9   67.8   71.1   72.6   75.9
                    DeepWalk         56.0   63.0   70.2   75.5   80.1   85.2   85.3   87.8   90.3
                      LINE           55.0   58.6   66.4   73.0   77.6   82.8   85.6   88.4   89.3
                    node2vec         55.9   62.4   66.1   75.0   78.7   81.6   85.9   87.3   88.2
               Naive Combination     72.7   82.0   84.9   87.0   88.7   91.9   92.4   93.9   94.0
                    TADW             86.6   88.2   90.2   90.8   90.0   93.0   91.0   93.4   92.7
                     CENE            72.1   86.5   84.6   88.1   89.4   89.2   93.9   95.0   95.9
               CANE (text only)      78.0   80.5   83.9   86.3   89.3   91.4   91.8   91.4   93.3
              CANE (w/o attention)   85.8   90.5   91.6   93.2   93.9   94.6   95.4   95.1   95.5
                   CANE              86.8   91.5   92.2   93.9   94.6   94.9   95.6   96.6   97.7

                          Table 2: AUC values on Cora. (α = 1.0, β = 0.3, γ = 0.3)


by (Leskovec et al., 2005). We filter out papers             CENE (Sun et al., 2016) leverages both struc-
without abstract information and retain 1, 038 pa-        ture and textural information by regarding text
pers at last.                                             content as a special kind of vertices, and optimizes
   Zhihu4 is the largest online Q&A website in            the probabilities of heterogeneous links.
China. Users follow each other and answer ques-
tions on this site. We randomly crawl 10, 000 ac-         5.3    Evaluation Metrics and Experiment
tive users from Zhihu, and take the descriptions of              Settings
their concerned topics as text information.               For link prediction, we adopt a standard evaluation
   The detailed statistics are listed in Table 1.         metric AUC (Hanley and McNeil, 1982), which
                                                          represents the probability that vertices in a random
5.2     Baselines                                         unobserved link are more similar than those in a
                                                          random nonexistent link.
We employ the following methods as baselines:
                                                             For vertex classification, we employ L2-
   Structure-only:                                        regularized logistic regression (L2R-LR) (Fan
   MMB (Airoldi et al., 2008) (Mixed Membership           et al., 2008) to train classifiers, and evaluate the
Stochastic Blockmodel) is a conventional graphi-          classification accuracies of various methods.
cal model of relational data. It allows each vertex          To be fair, we set the embedding dimension to
to randomly select a different ”topic” when form-         200 for all methods. In LINE, we set the number
ing an edge.                                              of negative samples to 5; we learn the 100 dimen-
   DeepWalk (Perozzi et al., 2014) performs ran-          sional first-order and second-order embeddings re-
dom walks over networks and employ Skip-Gram              spectively, and concatenate them to form the 200
model (Mikolov et al., 2013a) to learn vertex em-         dimensional embeddings. In node2vec, we em-
beddings.                                                 ploy grid search and select the best-performed
   LINE (Tang et al., 2015) learns vertex embed-          hyper-parameters for training. We also apply grid
dings in large-scale networks using first-order and       search to set the hyper-parameters α, β and γ in
second-order proximities.                                 CANE. Besides, we set the number of negative
   Node2vec (Grover and Leskovec, 2016) pro-              samples k to 1 in CANE to speed up the train-
poses a biased random walk algorithm based on             ing process. To demonstrate the effectiveness of
DeepWalk to explore neighborhood architecture             considering attention mechanism and two types of
more efficiently.                                         objectives in Eqs. (3) and (6), we design three
   Structure and Text:                                    versions of CANE for evaluation, i.e., CANE with
   Naive Combination: We simply concatenate the           text only, CANE without attention and CANE.
best-performed structure-based embeddings with
                                                          5.4    Link Prediction
CNN based embeddings to represent the vertices.
   TADW (Yang et al., 2015) employs matrix fac-           As shown in Table 2, Table 3 and Table 4, we eval-
torization to incorporate text features of vertices       uate the AUC values while removing different ra-
into network embeddings.                                  tios of edges on Cora, HepTh and Zhihu respec-
                                                          tively. Note that, when we only keep 5% edges
  4
      https://www.zhihu.com/                              for training, most vertices are isolated, which re-
              %Training edges      15%    25%    35%    45%    55%     65%    75%      85%    95%
                  MMB              54.6   57.9   57.3   61.6    66.2   68.4   73.6     76.0   80.3
                 DeepWalk          55.2   66.0   70.0   75.7    81.3   83.3   87.6     88.9   88.0
                   LINE            53.7   60.4   66.5   73.9    78.5   83.8   87.5     87.7   87.6
                 node2vec          57.1   63.6   69.9   76.2    84.3   87.3   88.4     89.2   89.2
             Naive Combination     78.7   82.1   84.7   88.7    88.7   91.8   92.1     92.0   92.7
                  TADW             87.0   89.5   91.8   90.8    91.1   92.6   93.5     91.9   91.7
                   CENE            86.2   84.6   89.8   91.2    92.3   91.8   93.2     92.9   93.2
             CANE (text only)      83.8   85.2   87.3   88.9   91.1    91.2   91.8   93.1     93.5
            CANE (w/o attention)   84.5   89.3   89.2   91.6   91.1    91.8   92.3   92.5     93.6
                 CANE              90.0   91.2   92.0   93.0   94.2    94.6   95.4   95.7     96.3

                      Table 3: AUC values on HepTh. (α = 0.7, β = 0.2, γ = 0.2)

              %Training edges      15%    25%    35%    45%    55%     65%    75%      85%    95%
                  MMB              51.0   51.5   53.7   58.6    61.6   66.1   68.8     68.9   72.4
                 DeepWalk          56.6   58.1   60.1   60.0    61.8   61.9   63.3     63.7   67.8
                   LINE            52.3   55.9   59.9   60.9    64.3   66.0   67.7     69.3   71.1
                 node2vec          54.2   57.1   57.3   58.3    58.7   62.5   66.2     67.6   68.5
             Naive Combination     55.1   56.7   58.9   62.6    64.4   68.7   68.9     69.0   71.5
                  TADW             52.3   54.2   55.6   57.3    60.8   62.4   65.2     63.8   69.0
                   CENE            56.2   57.4   60.3   63.0    66.3   66.0   70.2     69.8   73.8
             CANE (text only)      55.6   56.9   57.3   61.6   63.6    67.0   68.5   70.4     73.5
            CANE (w/o attention)   56.7   59.1   60.9   64.0   66.1    68.9   69.8   71.0     74.3
                 CANE              56.8   59.3   62.9   64.5   68.9    70.4   71.4   73.6     75.4

                       Table 4: AUC values on Zhihu. (α = 1.0, β = 0.3, γ = 0.3)


sults in the poor and meaningless performance of        tions. It demonstrates the flexibility and robust-
all the methods. Thus, we omit the results under        ness of CANE.
this training ratio. From these tables, we have the        (3) By introducing attention mechanism, the
following observations:                                 learnt context-aware embeddings obtain consider-
   (1) Our proposed CANE consistently achieves          able improvements than the ones without atten-
significant improvement comparing to all the base-      tion. It verifies our assumption that a specific ver-
lines on all different datasets and different train-    tex should play different roles when interacting
ing ratios. It indicates the effectiveness of CANE      with other vertices, and thus benefits the relevant
when applied to link prediction task, and verifies      link prediction task.
that CANE has the capability of modeling rela-             To summarize, all the above observations
tionships between vertices precisely.                   demonstrate that CANE can learn high-quality
                                                        context-aware embeddings, which are conducive
   (2) What calls for special attention is that, both   to estimating the relationship between vertices
CENE and TADW exhibit unstable performance              precisely. Moreover, the experimental results on
under various training ratios. Specifically, CENE       link prediction task state the effectiveness and ro-
performs poorly under small training ratios, be-        bustness of CANE.
cause it reserves much more parameters (e.g.,
convolution kernels and word embeddings) than
                                                        5.5    Vertex Classification
TADW, which need more data for training. Differ-
ent from CENE, TADW performs much better un-            In CANE, we obtain various embeddings of a ver-
der small training ratios, because DeepWalk based       tex according to the vertex it connects to. It’s intu-
methods can explore the sparse network struc-           itive that the obtained context-aware embeddings
ture well through random walks even with lim-           are naturally applicable to link prediction task.
ited edges. However, it achieves poor performance       However, network analysis tasks, such as vertex
under large ones, as its simplicity and the limita-     classification and clustering, require a global em-
tion of bag-of-words assumption. On the contrary,       bedding, rather than several context-aware embed-
CANE has a stable performance in various situa-         dings for each vertex.
   To demonstrate the capability of CANE to solve                                                              Eq. (11). To obtain the weights of words, we as-
these issues, we generate the global embedding                                                                 sign the attention weight to each word in this win-
of a vertex u by simply averaging all the context-                                                             dow, and add the attention weights of a word to-
aware embeddings as follows:                                                                                   gether as its final weight.
                                                1            X
                                          u=                                 u(v) ,
                                                N
                                                         (u,v)|(v,u)∈E

where N indicates the number of context-aware
embeddings of u.
                   100




Accuracy (× 100)
                    90

                    80

                    70

                    60

                    50   B            k    E         ec     C        W       E            )       n)       E
                         M        al       N        2v      N    D       EN            ly     nt           N
                     M           pW       LI    de              TA                   on         io     A
                                                                         C        xt       at          C
                             ee                no                            E(              te
                             D                                                  te     /0
                                                                             N       E(
                                                                         A             w
                                                                         C        N
                                                                                 A
                                                                              C



             Figure 3: Vertex classification results on Cora.

   With the generated global embeddings, we con-
duct 2-fold cross-validation and report the aver-
age accuracy of vertex classification on Cora. As
shown in Fig. 3, we observe that:
   (1) CANE achieves comparable performance
with state-of-the-art model CENE. It states that the
learnt context-aware embeddings can transform
into high-quality context-free embeddings through
simple average operation, which can be further                                                                   Figure 4: Visualizations of mutual attention.
employed to other network analysis tasks.
   (2) With the introduction of mutual attention                                                                  The proposed attention mechanism makes the
mechanism, CANE has an encouraging improve-                                                                    relations between vertices explicit and inter-
ment than the one without attention, which is in                                                               pretable. We select three connected vertices in
accordance with the results of link prediction. It                                                             Cora for example, denoted as A, B and C. From
denotes that CANE is flexible to various network                                                               Fig. 4, we observe that, though there exists cita-
analysis tasks.                                                                                                tion relations with identical paper A, paper B and
                                                                                                               C concern about different parts of A. The atten-
5.6                      Case Study                                                                            tion weights over A in edge #1 are assigned to
To demonstrate the significance of mutual atten-                                                               “reinforcement learning”. On the contrary, the
tion on selecting meaningful features from text in-                                                            weights in edge #2 are assigned to “machine learn-
formation, we visualize the heat maps of two ver-                                                              ing’”, “supervised learning algorithms” and “com-
tex pairs in Fig. 4. Note that, every word in this                                                             plex stochastic models”. Moreover, all these key
figure accompanies with various background col-                                                                elements in A can find corresponding words in B
ors. The stronger the background color is, the                                                                 and C. It’s intuitive that these key elements give
larger the weight of this word is. The weight of                                                               an exact explanation of the citation relations. The
each word is calculated according to the attention                                                             discovered significant correlations between vertex
weights as follows.                                                                                            pairs reflect the effectiveness of mutual attention
   For each vertex pair, we can get the attention                                                              mechanism, as well as the capability of CANE for
weight of each convolution window according to                                                                 modeling relations precisely.
6   Conclusion and Future Work                         Jifan Chen, Qi Zhang, and Xuanjing Huang. 2016.
                                                          Incorporate group information to enhance network
In this paper, we propose the concept of Context-         embedding. In Proceedings of CIKM.
Aware Network Embedding (CANE) for the first
                                                       Kyunghyun Cho, Bart Van Merriënboer, Caglar Gul-
time, which aims to learn various context-aware          cehre, Dzmitry Bahdanau, Fethi Bougares, Holger
embeddings for a vertex according to the neigh-          Schwenk, and Yoshua Bengio. 2014. Learning
bors it interacts with. Specifically, we implement       phrase representations using rnn encoder-decoder
CANE on text-based information networks with             for statistical machine translation. In Proceedings
                                                         of EMNLP.
proposed mutual attention mechanism, and con-
duct experiments on several real-world informa-        Cıcero Nogueira dos Santos, Ming Tan, Bing Xiang,
tion networks. Experimental results on link pre-         and Bowen Zhou. 2016. Attentive pooling net-
diction demonstrate that CANE is effective for           works. CoRR, abs/1602.03609 .
modeling the relationship between vertices. Be-
                                                       Rong-En Fan, Kai-Wei Chang, Cho-Jui Hsieh, Xiang-
sides, the learnt context-aware embeddings can           Rui Wang, and Chih-Jen Lin. 2008. Liblinear: A
compose high-quality context-free embeddings.            library for large linear classification. JMLR 9:1871–
   We will explore the following directions in fu-       1874.
ture:
                                                       Aditya Grover and Jure Leskovec. 2016. Node2vec:
   (1) We have investigated the effectiveness of         Scalable feature learning for networks. In Proceed-
CANE on text-based information networks. In fu-          ings of KDD.
ture, we will strive to implement CANE on a wider
                                                       James A Hanley and Barbara J McNeil. 1982. The
variety of information networks with multi-modal         meaning and use of the area under a receiver operat-
data, such as labels, images and so on.                  ing characteristic (roc) curve. Radiology 143(1):29–
   (2) CANE encodes latent relations between ver-        36.
tices into their context-aware embeddings. Fur-
                                                       Baotian Hu, Zhengdong Lu, Hang Li, and Qingcai
thermore, there usually exist explicit relations in      Chen. 2014. Convolutional neural network architec-
social networks (e.g., families, friends and col-        tures for matching natural language sentences. In
leagues relations between social network users),         Proceedings of NIPS. pages 2042–2050.
which are expected to be critical to NE. Thus, we
                                                       Rie Johnson and Tong Zhang. 2014.       Effective
want to explore how to incorporate and predict
                                                         use of word order for text categorization with
these explicit relations between vertices in NE.         convolutional neural networks.  arXiv preprint
                                                         arXiv:1412.1058 .
Acknowledgements
                                                       Yoon Kim. 2014. Convolutional neural networks for
This work is supported by the 973 Program (No.           sentence classification.
2014CB340501), the National Natural Science
                                                       Diederik Kingma and Jimmy Ba. 2015. Adam: A
Foundation of China (NSFC No. 61572273,                  method for stochastic optimization. In Proceedings
61532010, 61661146007), and Tsinghua Uni-                of ICLR.
versity Initiative Scientific Research Program
(20151080406).                                         Ryan Kiros, Yukun Zhu, Ruslan R Salakhutdinov,
                                                         Richard Zemel, Raquel Urtasun, Antonio Torralba,
                                                         and Sanja Fidler. 2015. Skip-thought vectors. In
                                                         Proceedings of NIPS. pages 3294–3302.
References
                                                       Jure Leskovec, Jon Kleinberg, and Christos Faloutsos.
Edoardo M Airoldi, David M Blei, Stephen E Fienberg,      2005. Graphs over time: densification laws, shrink-
  and Eric P Xing. 2008. Mixed membership stochas-        ing diameters and possible explanations. In Pro-
  tic blockmodels. JMLR 9(Sep):1981–2014.                 ceedings of KDD. pages 177–187.

Phil Blunsom, Edward Grefenstette, and Nal Kalch-      Andrew McCallum, Kamal Nigam, Jason Rennie, and
  brenner. 2014. A convolutional neural network for      Kristie Seymore. 2000. Automating the construc-
  modelling sentences. In Proceedings of ACL.            tion of internet portals with machine learning. In-
                                                         formation Retrieval Journal 3:127–163.
Shaosheng Cao, Wei Lu, and Qiongkai Xu. 2015.
  Grarep: Learning graph representations with global   Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey
  structural information. In Proceedings of CIKM.        Dean. 2013a. Efficient estimation of word represen-
  pages 891–900.                                         tations in vector space. In Proceedings of ICIR.
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Cor-
  rado, and Jeff Dean. 2013b. Distributed representa-
  tions of words and phrases and their compositional-
  ity. In Proceedings of NIPS. pages 3111–3119.
Bryan Perozzi, Rami Al-Rfou, and Steven Skiena.
  2014. Deepwalk: Online learning of social repre-
  sentations. In Proceedings of KDD. pages 701–710.
Mike Schuster and Kuldip K Paliwal. 1997. Bidirec-
  tional recurrent neural networks. IEEE Transactions
  on Signal Processing 45(11):2673–2681.
Xiaofei Sun, Jiang Guo, Xiao Ding, and Ting Liu.
  2016. A general framework for content-enhanced
  network representation learning. arXiv preprint
  arXiv:1610.02906 .
Kai Sheng Tai, Richard Socher, and Christopher D
  Manning. 2015. Improved semantic representations
  from tree-structured long short-term memory net-
  works. In Proceedings of ACL.
Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun
   Yan, and Qiaozhu Mei. 2015. Line: Large-scale in-
   formation network embedding. In Proceedings of
   WWW. pages 1067–1077.
Lei Tang and Huan Liu. 2009. Relational learning
  via latent social dimensions. In Proceedings of
  SIGKDD. pages 817–826.
Cunchao Tu, Hao Wang, Xiangkai Zeng, Zhiyuan Liu,
  and Maosong Sun. 2016a. Community-enhanced
  network representation learning for network analy-
  sis. arXiv preprint arXiv:1611.06645 .
Cunchao Tu, Weicheng Zhang, Zhiyuan Liu, and
  Maosong Sun. 2016b. Max-margin deepwalk: Dis-
  criminative learning of network representation. In
  Proceedings of IJCAI.
Daixin Wang, Peng Cui, and Wenwu Zhu. 2016. Struc-
  tural deep network embedding. In Proceedings of
  KDD.
Cheng Yang, Zhiyuan Liu, Deli Zhao, Maosong Sun,
  and Edward Y Chang. 2015. Network representa-
  tion learning with rich text information. In Proceed-
  ings of IJCAI.

