# metapath2vec Scalable Representation Learning for Heterogene Heterogeneous 2017

> Source: `metapath2vec_Scalable_Representation_Learning_for_Heterogene_Heterogeneous_2017.pdf`

---

                  metapath2vec: Scalable Representation Learning for
                             Heterogeneous Networks
                    Yuxiao Dong∗                                               Nitesh V. Chawla                                                                   Ananthram Swami
                 Microsoft Research                                        University of Notre Dame                                                     Army Research Laboratory
                Redmond, WA 98052                                           Notre Dame, IN 46556                                                           Adelphi, MD 20783
              yuxdong@microsoft.com                                            nchawla@nd.edu                                                         ananthram.swami.civ@mail.mil

ABSTRACT                                                                                                  D. Song
                                                                                                                                               ISCA
                                                                                                                                                        SIGCOMM                    W. B. Croft
                                                                                                                                                                                                                                    SIGIR
                                                                                                       R. N. Taylor                                                                   J. Han              J. Malik              IJCAI
                                                                                                       S. Shenker                         FOCS                                     M. I.Jordan                                   ACL
We study the problem of representation learning in heterogeneous                                               O. Mutlu                               S&P
                                                                                                                                                              ICSE             C. D. Manning                                     WWW
                                                                                                                                                                                                                                   KDD
                                                                                                                     R. E. Tarjan                                                         T. Kanade                                 CVPR
                                                                                                                 J. Han                                     ACL            A. Tomkins                                              NIPS
networks. Its unique challenges come from the existence of mul-                                     R. Agrawal
                                                                                               A. Tomkins
                                                                                                                                                                                               H. Ishii
                                                                                                                                                        WWW                                                             SIGGRAPH
                                                                                                                                                                            R. Agrawal
tiple types of nodes and links, which limit the feasibility of the                                        J. Dean                                        IJCAI                          H. Jensen                         CHI
                                                                                                                       W. B. Croft               OSDI                               O. Mutlu                                      FOCS
                                                                                                                                                   CVPR                          R. N. Taylor
conventional network embedding techniques. We develop two                                         M. I.Jordan          H. Ishii                     CHI                            R. E. Tarjan
                                                                                                                                     SIGGRAPH KDD                                                                                   SIGMOD
                                                                                                                                                                                  J. Dean
                                                                                                                                                                               S. Shenker                                    ICSE
                                                                                                C. D. Manning                               SIGMOD SIGIR                                                             ISCA
scalable representation learning models, namely metapath2vec and                                             T. Kanade
                                                                                                                                                                                        D. Song
                                                                                                                H. Jensen                         NIPS
metapath2vec++. The metapath2vec model formalizes meta-path-                                                                                                                                                                 SIGCOMM
                                                                                                                                                                                                                            S&P
                                                                                                          J. Malik                                                                                                           OSDI
based random walks to construct the heterogeneous neighborhood
of a node and then leverages a heterogeneous skip-gram model                                               (a) DeepWalk / node2vec                                                                   (b) PTE
to perform node embeddings. The metapath2vec++ model further                                                                                                             S. Shenker                                           OSDI
                                                                                                              J. Han         A. Tomkins
                                                                                                                                                                             J. Dean                                           SIGCOMM
enables the simultaneous modeling of structural and semantic cor-                                          KDD          WWW
                                                                                                                                  R. Agrawal
                                                                                                                                                                             D. Song                                           S&P
                                                                                                                                     SIGMOD                                 O. Mutlu                                         ISCA
relations in heterogeneous networks. Extensive experiments show                                             SIGIR      W. B. Croft                                                                                            ICSE
                                                                                                                                                                             R. N. Taylor
that metapath2vec and metapath2vec++ are able to not only outper-                                        ACL                                  J. Dean
                                                                                                                                      FOCS D. Song       S. Shenker
                                                                                                                                                                        R. E. Tarjan                                        FOCS
                                                                                                                   M. I.Jordan           R. N. Taylor                                                                             SIGMOD
                                                                                               C. D. Manning                                               OSDI          R. Agrawal
form state-of-the-art embedding models in various heterogeneous                                             NIPS           R. E. Tarjan        S&P      SIGCOMM                  H. Ishii                                   CHI
                                                                                                                                        ICSE                                                                                        WWW
                                                                                                       IJCAI            CHI                                            A. Tomkins
network mining tasks, such as node classification, clustering, and                                                                                   O. Mutlu                J. Han                                          SIGGRAPH
                                                                                                                                         ISCA
                                                                                                                             H. Ishii                                    H. Jensen
                                                                                                                                                                         W. B. Croft                                              KDD
similarity search, but also discern the structural and semantic cor-                                    CVPR                                                                                                                     SIGIR
                                                                                                                 T. Kanade                                              M. I.Jordan                                             IJCAI
                                                                                                   J. Malik                                                                T. Kanade                                            NIPS
                                                                                                                                 SIGGRAPH                             C. D. Manning                                             ACL
relations between diverse network objects.                                                                       H. Jensen
                                                                                                                                                                             J. Malik                                       CVPR


                                                                                                                    (c) metapath2vec                                                        (d) metapath2vec++
CCS CONCEPTS
•Information systems →Social networks; •Computing method-                                     Figure 1: 2D PCA projections of the 128D embeddings of 16
ologies →Unsupervised learning; Learning latent represen-                                     top CS conferences and corresponding high-profile authors.
tations; Knowledge representation and reasoning;
                                                                                              and information networks are similarly rich and complex data that
KEYWORDS                                                                                      encode the dynamics and types of human interactions, and are sim-
Network Embedding; Heterogeneous Representation Learning; La-                                 ilarly amenable to representation learning using neural networks.
tent Representations; Feature Learning; Heterogeneous Information                             In particular, by mapping the way that people choose friends and
Networks                                                                                      maintain connections as a “social language,” recent advances in
ACM Reference format:                                                                         natural language processing (NLP) [3] can be naturally applied to
Yuxiao Dong, Nitesh V. Chawla, and Ananthram Swami. 2017. metap-                              network representation learning, most notably the group of NLP
ath2vec: Scalable Representation Learning for Heterogeneous Networks. In                      models known as word2vec [17, 18]. A number of recent research
Proceedings of KDD ’17, August 13-17, 2017, Halifax, NS, Canada, , 10 pages.                  publications have proposed word2vec-based network representa-
DOI: http://dx.doi.org/10.1145/3097983.3098036                                                tion learning frameworks, such as DeepWalk [22], LINE [30], and
                                                                                              node2vec [8]. Instead of handcrafted network feature design, these
1     INTRODUCTION                                                                            representation learning methods enable the automatic discovery of
                                                                                              useful and meaningful (latent) features from the “raw networks.”
Neural network-based learning models can represent latent embed-
                                                                                                 However, these work has thus far focused on representation
dings that capture the internal relations of rich, complex data across
                                                                                              learning for homogeneous networks—representative of singular
various modalities, such as image, audio, and language [15]. Social
                                                                                              type of nodes and relationships. Yet a large number of social and
∗ This work was done when Yuxiao was a Ph.D. student at University of Notre Dame.
                                                                                              information networks are heterogeneous in nature, involving diver-
Permission to make digital or hard copies of all or part of this work for personal or         sity of node types and/or relationships between nodes [25]. These
classroom use is granted without fee provided that copies are not made or distributed         heterogeneous networks present unique challenges that cannot
for profit or commercial advantage and that copies bear this notice and the full citation     be handled by representation learning models that are specifically
on the first page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,       designed for homogeneous networks. Take, for example, a het-
to post on servers or to redistribute to lists, requires prior specific permission and/or a   erogeneous academic network: How do we effectively preserve
fee. Request permissions from permissions@acm.org.
                                                                                              the concept of “word-context” among multiple types of nodes, e.g.,
KDD ’17, August 13-17, 2017, Halifax, NS, Canada
© 2017 ACM. 978-1-4503-4887-4/17/08. . . $15.00                                               authors, papers, venues, organizations, etc.? Can random walks,
DOI: http://dx.doi.org/10.1145/3097983.3098036                                                such those used in DeepWalk and node2vec, be applied to networks
                         Table 1: Case study of similarity search in the heterogeneous DBIS data used in [26].
      Method       PathSim [26]          DeepWalk / node2vec [8, 22]      LINE (1st+2nd) [30]           PTE [29]               metapath2vec              metapath2vec++

       Input       meta-paths          heterogeneous random walk paths    heterogeneous edges     heterogeneous edges     probabilistic meta-paths   probabilistic meta-paths

       Query   PKDD     C. Faloutsos   PKDD          C. Faloutsos        PKDD     C. Faloutsos   PKDD      C. Faloutsos   PKDD       C. Faloutsos    PKDD       C. Faloutsos

         1      ICDM       J. Han      R. S.            J. Pan           W. K.   C. Aggarwal       KDD     C. Aggarwal    A. S.     C. Aggarwal        KDD      R. Agrawal
         2       SDM    R. Agrawal     M. N.           H. Tong           S. A.       P. Yu        ICDM         P. Yu      M. B.         J. Pei       PAKDD         J. Han
         3     PAKDD        J. Pei     R. P.           H. Yang           A. B.   D. Gunopulos      SDM        Y. Tao      P. B.         P. Yu         ICDM          J. Pei
         4       KDD    C. Aggarwal    G. G.           R. Filho          M. S.    N. Koudas      DMKD       N. Koudas     M. S.      H. Cheng        DMKD       C. Aggarwal
         5     DMKD     H. Jagadish    F. J.           R. Chan           S. A.    M. Vlachos     PAKDD      R. Rastogi    M. K.       V. Ganti         SDM          P. Yu



of multiple types of nodes? Can we directly apply homogeneous                              used in [26] (see Section 4 for details). By modeling the hetero-
network-oriented embedding architectures (e.g., skip-gram) to het-                         geneous neighborhood and further leveraging the heterogeneous
erogeneous networks?                                                                       negative sampling technique, metapath2vec++ is able to achieve the
    By solving these challenges, the latent heterogeneous network                          best top-five similar results for both types of queries. Figure 1 shows
embeddings can be further applied to various network mining tasks,                         the visualization of the 2D projections of the learned embeddings
such as node classification [13], clustering [27, 28], and similarity                      for 16 CS conferences and corresponding high-profile researchers
search [26, 35]. In contrast to conventional meta-path-based meth-                         in each field. Remarkably, we find that metapath2vec++ is capable
ods [25], the advantage of latent-space representation learning lies                       of automatically organizing these two types of nodes and implicitly
in its ability to model similarities between nodes without connected                       learning the internal relationships between them, suggested by the
meta-paths. For example, if authors have never published papers in                         similar directions and distances of the arrows connecting each pair.
the same venue—imagine one publishes 10 papers all in NIPS and                             For example, it learns J. Dean → OSDI and C. D. Manning → ACL.
the other has 10 publications all in ICML; their “APCPA”-based Path-                       metapath2vec is also able to group each author-conference pair
Sim similarity [26] would be zero—this will be naturally overcome                          closely, such as R. E. Tarjan and FOCS. All of these properties are
by network representation learning.                                                        not discoverable from conventional network embedding models.
Contributions. We formalize the heterogeneous network repre-                                  To summarize, our work makes the following contributions:
sentation learning problem, where the objective is to simultane-                           (1) Formalizes the problem of heterogeneous network represen-
ously learn the low-dimensional and latent embeddings for multiple                             tation learning and identifies its unique challenges resulting
types of nodes. We present the metapath2vec and its extension meta-                            from network heterogeneity.
path2vec++ frameworks. The goal of metapath2vec is to maximize                             (2) Develops effective and efficient network embedding frame-
the likelihood of preserving both the structures and semantics of a                            works, metapath2vec & metapath2vec++, for preserving both
given heterogeneous network. In metapath2vec, we first propose                                 structural and semantic correlations of heterogeneous networks.
meta-path [25] based random walks in heterogeneous networks                                (3) Through extensive experiments, demonstrates the efficacy and
to generate heterogeneous neighborhoods with network seman-                                    scalability of the presented methods in various heterogeneous
tics for various types of nodes. Second, we extend the skip-gram                               network mining tasks, such as node classification (achieving
model [18] to facilitate the modeling of geographically and seman-                             relative improvements of 35–319% over benchmarks) and node
tically close nodes. Finally, we develop a heterogeneous negative                              clustering (achieving relative gains of 13–16% over baselines).
sampling-based method, referred to as metapath2vec++, that en-                             (4) Demonstrates the automatic discovery of internal semantic
ables the accurate and efficient prediction of a node’s heterogeneous                          relationships between different types of nodes in heterogeneous
neighborhood.                                                                                  networks by metapath2vec & metapath2vec++, not discoverable
    The proposed metapath2vec and metapath2vec++ models are dif-                               by existing work.
ferent from conventional network embedding models, which focus
on homogeneous networks [8, 22, 30]. Specifically, conventional
models suffer from the identical treatment of different types of
                                                                                           2     PROBLEM DEFINITION
nodes and relations, leading to the production of indistinguishable                        We formalize the representation learning problem in heterogeneous
representations for heterogeneous nodes—as evident through our                             networks, which was first briefly introduced in [21]. In specific, we
evaluation. Further, the metapath2vec and metapath2vec++ models                            leverage the definition of heterogeneous networks in [25, 27] and
also differ from the Predictive Text Embedding (PTE) model [29]                            present the learning problem with its inputs and outputs.
in several ways. First, PTE is a semi-supervised learning model
that incorporates label information for text data. Second, the het-                           Definition 2.1. A Heterogeneous Network is defined as a graph
erogeneity in PTE comes from the text network wherein a link                               G = (V , E,T ) in which each node v and each link e are associated
connects two words, a word and its document, and a word and its                            with their mapping functions ϕ (v) : V → TV and φ(e) : E → TE ,
label. Essentially, the raw input of PTE is words and its output is                        respectively. TV and TE denote the sets of object and relation types,
the embedding of each word, rather than multiple types of objects.                         where |TV | + |TE | > 2.
    We summarize the differences of these methods in Table 1, which
                                                                                              For example, one can represent the academic network in Figure
lists their input to learning algorithms, as well as the top-five simi-
                                                                                           2(a) with authors (A), papers (P), venues (V), organizations (O) as
larity search results in the DBIS network for the same two queries
                                                                                           nodes, wherein edges indicate the coauthor (A–A), publish (A–P,
P–V), affiliation (O–A) relationships. By considering a heteroge-         the heterogeneous network structures into skip-gram, we propose
neous network as input, we formalize the problem of heterogeneous         meta-path-based random walks in heterogeneous networks.
network representation learning as follows.                               Heterogeneous Skip-Gram. In metapath2vec, we enable skip-
   Problem 1. Heterogeneous Network Representation Learn-                 gram to learn effective node representations for a heterogeneous
ing: Given a heterogeneous network G, the task is to learn the d-         network G = (V , E,T ) with |TV | > 1 by maximizing the probability
                                                                          of having the heterogeneous context Nt (v), t ∈ TV given a node v:
dimensional latent representations X ∈ R |V |×d , d  |V | that are
able to capture the structural and semantic relations among them.                               X X          X
                                                                                       arg max                     log p(c t |v; θ )      (2)
                                                                                              θ
   The output of the problem is the low-dimensional matrix X, with                                v ∈V t ∈TV c t ∈N t (v )
the v t h row—a d-dimensional vector Xv —corresponding to the
representation of node v. Notice that, although there are different       where Nt (v) denotes v’s neighborhood with the t th type of nodes
types of nodes in V , their representations are mapped into the           and p(c t |v; θ ) is commonly defined as a softmax function [3, 7, 18,
                                                                                                               X c ·Xv
same latent space. The learned node representations can benefit           24], that is: p(c t |v; θ ) = P e et Xu ·Xv , where Xv is the v th row of
                                                                                                            u ∈V
various heterogeneous network mining tasks. For example, the              X, representing the embedding vector for node v. For illustration,
embedding vector of each node can be used as the feature input of         consider the academic network in Figure 2(a), the neighborhood
node classification, clustering, and similarity search tasks.             of one author node a 4 can be structurally close to other authors
   The main challenge of this problem comes from the network              (e.g., a 2 , a 3 & a 5 ), venues (e.g., ACL & KDD), organizations (CMU
heterogeneity, wherein it is difficult to directly apply homogeneous      & MIT), as well as papers (e.g., p2 & p3 ).
language and network embedding methods. The premise of network                To achieve efficient optimization, Mikolov et al. introduced neg-
embedding models is to preserve the proximity between a node              ative sampling [18], in which a relatively small set of words (nodes)
and its neighborhood (context) [8, 22, 30]. In a heterogeneous envi-      are sampled from the corpus (network) for the construction of soft-
ronment, how do we define and model this ‘node–neighborhood’              max. We leverage the same technique for metapath2vec. Given a
concept? Furthermore, how do we optimize the embedding models             negative sample size M, Eq. 2 is updated as follows: log σ (X c t ·Xv )+
                                                                          PM
                                                                            m=1 Eu m ∼P (u ) [log σ (−X u ·Xv )], where σ (x ) = 1+e −x and P (u)
that effectively maintain the structures and semantics of multiple                                            m
                                                                                                                                      1
types of nodes and relations?                                             is the pre-defined distribution from which a negative node um is
                                                                          drew from for M times. metapath2vec builds the the node frequency
3     THE METAPATH2VEC FRAMEWORK                                          distribution by viewing different types of nodes homogeneously
We present a general framework, metapath2vec, which is capable            and draw (negative) nodes regardless of their types.
of learning desirable node representations in heterogeneous net-          Meta-Path-Based Random Walks. How to effectively trans-
works. The objective of metapath2vec is to maximize the network           form the structure of a network into skip-gram? In DeepWalk [22]
probability in consideration of multiple types of nodes and edges.        and node2vec [8], this is achieved by incorporating the node paths
                                                                          traversed by random walkers over a network into the neighborhood
3.1    Homogeneous Network Embedding                                      function.
We, first, briefly introduce the word2vec model and its application           Naturally, we can put random walkers in a heterogeneous network
to homogeneous network embedding tasks. Given a text corpus,              to generate paths of multiple types of nodes. At step i, the transition
Mikolov et al. proposed word2vec to learn the distributed represen-       probability p(v i+1 |v i ) is denoted as the normalized probability
tations of words in a corpus [17, 18]. Inspired by it, DeepWalk [22]      distributed over the neighbors of v i by ignoring their node types.
and node2vec [8] aim to map the word-context concept in a text            The generated paths can be then used as the input of node2vec
corpus into a network. Both methods leverage random walks to              and DeepWalk. However, Sun et al. demonstrated that heterogeneous
achieve this and utilize the skip-gram model to learn the repre-          random walks are biased to highly visible types of nodes—those with
sentation of a node that facilitates the prediction of its structural     a dominant number of paths—and concentrated nodes—those with a
context—local neighborhoods—in a homogeneous network. Usu-                governing percentage of paths pointing to a small set of nodes [26].
ally, given a network G = (V , E), the objective is to maximize the           In light of these issues, we design meta-path-based random walks
network probability in terms of local structures [8, 18, 22], that is:    to generate paths that are able to capture both the semantic and
                              Y Y                                         structural correlations between different types of nodes, facilitat-
                     arg max             p(c |v; θ )              (1)
                         θ                                                ing the transformation of heterogeneous network structures into
                             v ∈V c ∈N (v )
                                                                          metapath2vec’s skip-gram.
where N (v) is the neighborhood of node v in the network G, which             Formally, a meta-path scheme P is defined as a path that is
can be defined in different ways such as v’s one-hop neighbors,                                          R1        R2        Rt          R l −1
                                                                          denoted in the form of V1 −−→ V2 −−→ · · · Vt −−→ Vt +1 · · · −−−−→ Vl ,
and p(c |v; θ ) defines the conditional probability of having a context
                                                                          wherein R = R 1 ◦ R 2 ◦ · · · ◦ Rl −1 defines the composite relations
node c given a node v.
                                                                          between node types V1 and Vl [25]. Take Figure 2(a) as an example,
                                                                          a meta-path “APA” represents the coauthor relationships on a paper
3.2    Heterogeneous Network Embedding:                                   (P) between two authors (A), and “APVPA” represents two authors
       metapath2vec                                                       (A) publish papers (P) in the same venue (V). Previous work has
To model the heterogeneous neighborhood of a node, metapath2vec           shown that many data mining tasks in heterogeneous information
introduces the heterogeneous skip-gram model. To incorporate              networks can benefit from the modeling of meta-paths [6, 25, 27].
                                                                                                                                                                   output layer
                                                                                                                                                                   prob. that
                                                                                                                                                                   KDD appears
                                                                                                                                                                                  prob. that
                                                             input layer     hidden            output layer                input layer        hidden                              ACL appears
                                                                              layer                                                            layer              |VV| x kV
                                                                                               prob. that
                                                            KDD 0                              KDD apears                  KDD 0
                                                            ACL 0                                                          ACL 0
Org     Author Paper Venue                                  a 1 0                                                          a 1 0
                                                                                                                                                                   prob. that
                                                                                                                                                                   a3 appears
                                                            a                                                              a
                                       meta paths
                                                                0                                                              0
         a                                                    2                                                              2

                                                            a                                                              a                                                      prob. that
             1
                                                                0                                                              0
                                                                                                 ... ...
                     p1                                       3                                                              3                                                    a5 appears
 MIT
         a   2                            APA               a 4 1                                                          a 4 1                                  |VA| x kA
                                                            a                                                              a
                           ACL

                                                              5 0                                                            5 0
         a           p2                                     MIT 0                                                          MIT 0
                                                                                                                                                                   prob. that
          3                                                                                                                                                        CMU appears
                                         APVPA
                                                            CMU 0                                                          CMU 0
         a   4
                     p3                                      p1 0                                                          p 1 0                                  |Vo| x ko
                                                             p                                                             p
 CMU                       KDD

         a                              OAPVPAO               2 0                                                            2 0
                                                             p                                                prob. that
                                                                                                                           p
             5
                                                              3 0                                             p3 appears     3 0                                   prob. that
                                                                                                                                                                   p2 appears
                                                                                                                                                                                  prob. that
                                                             |V|-dim                           |V| x k                      |V|-dim                                               p3 appears
                                                                                                                                                                  |Vp| x kP
                 (a) An academic network                    (b) Skip-gram in metapath2vec, node2vec, & DeepWalk                          (c) Skip-gram in metapath2vec++


Figure 2: An illustrative example of a heterogeneous academic network and skip-gram architectures of metapath2vec and
metapath2vec++ for embedding this network. (a). Yellow dotted lines denote coauthor relationships and red dotted lines denote citation
relationships. (b) The skip-gram architecture used in metapath2vec when predicting for a 4 , which is the same with the one in node2vec if
node types are ignored. |V |=12 denotes the number of nodes in the heterogeneous academic network in (a) and a 4 ’s neighborhood is set to
include CMU, a 2 , a 3 , a 5 , p2 , p3 , ACL, & KDD, making k = 8. (c) The heterogeneous skip-gram used in metapath2vec++. Instead of one set of
multinomial distributions for all types of neighborhood nodes in the output layer, it specifies one set of multinomial distributions for each
type of nodes in a 4 ’s neighborhood. Vt denotes one specific t-type nodes and V = VV ∪ VA ∪ VO ∪ VP . kt specifies the size of a particular
type of one’s neighborhood and k = kV + k A + kO + k P .


   Here we show how to use meta-paths to guide heterogeneous                                      In other words, in order to infer the specific type of context c t in
random walkers. Given a heterogeneous network G = (V , E,T ) and                                  Nt (v) given a node v, metapath2vec actually encourages all types
                                        R1          R2           Rt             R l −1            of negative samples, including nodes of the same type t as well as
a meta-path scheme P: V1 −−→ V2 −−→ · · · Vt −−→ Vt +1 · · · −−−−→ Vl ,
the transition probability at step i is defined as follows:                                       the other types in the heterogeneous network.
                                                                                                  Heterogeneous negative sampling. We further propose the
                                   1
                             |Nt +1 (vti ) |
                            
                                                 (v i+1 , vti ) ∈ E, ϕ (v i+1 ) = t+1            metapath2vec++ framework, in which the softmax function is nor-
        i+1
  p(v            |vti , P) =                     (v i+1 , vti ) ∈ E, ϕ (v i+1 ) , t+1            malized with respect to the node type of the context c t . Specifically,
                             
                                   0
                                                  (v i+1 , vti ) < E
                             
                                                                                                  p(c t |v; θ ) is adjusted to the specific node type t, that is,
                            
                            
                                  0
                                                                                         (3)
                                                                                                                                                 e X ct ·Xv
where vti ∈ Vt and Nt +1 (vti ) denote the Vt +1 type of neighborhood                                                      p(c t |v; θ ) = P               X ut ·X v
                                                                                                                                                                                               (5)
of node vti . In other words, v i+1 ∈ Vt +1 , that is, the flow of the                                                                         u t ∈Vt e
walker is conditioned on the pre-defined meta-path P. In addition,
                                                                                                  where Vt is the node set of type t in the network. In doing so,
meta-paths are commonly used in a symmetric way, that is, its first
                                                                                                  metapath2vec++ specifies one set of multinomial distributions for
node type V1 is the same with the last one Vl [25, 26, 28], facilitating
                                                                                                  each type of neighborhood in the output layer of the skip-gram
its recursive guidance for random walkers, i.e.,
                                                                                                  model. Recall that in metapath2vec and node2vec / DeepWalk, the
                          p(v i+1 |vti ) = p(v i+1 |v 1i ), if t = l                     (4)      dimension of the output multinomial distributions is equal to the
                                                                                                  number of nodes in the network. However, in metapath2vec++’s
  The meta-path-based random walk strategy ensures that the                                       skip-gram, the multinomial distribution dimension for type t nodes
semantic relationships between different types of nodes can be                                    is determined by the number of t-type nodes. A clear illustration
properly incorporated into skip-gram. For example, in a traditional                               can be seen in Figure 2(c). For example, given the target node a 4 in
random walk procedure, in Figure 2(a), the next step of a walker                                  the input layer, metapath2vec++ outputs four sets of multinomial
on node a 4 transitioned from node CMU can be all types of nodes                                  distributions, each corresponding to one type of neighbors—venues
surrounding it—a 2 , a 3 , a 5 , p2 , p3 , and CMU. However, under the                            V , authors A, organizations O, and papers P.
meta-path scheme ‘OAPVPAO’, for example, the walker is biased                                         Inspired by PTE [29], the sampling distribution is also specified
towards paper nodes (P) given its previous step on an organization                                by the node type of the neighbor c t that is targeted to predict, i.e.,
node CMU (O), following the semantics of this path.                                               Pt (·). Therefore, we have the following objective:

3.3      metapath2vec++                                                                                                                   M
                                                                                                                                          X
metapath2vec distinguishes the context nodes of node v conditioned                                   O(X) = log σ (X c t · Xv ) +              Eutm ∼Pt (ut ) [log σ (−Xutm · Xv )]
on their types when constructing its neighborhood function Nt (v)                                                                        m=1
in Eq. 2. However, it ignores the node type information in softmax.                                                                                                                            (6)
    Input: The heterogeneous information network G = (V , E,T ),                     computer scientists and 3,194,405 papers from 3,883 computer sci-
           a meta-path scheme P, #walks per node w, walk                             ence venues—both conferences and journals—held until 2016. We
           length l, embedding dimension d, neighborhood size k                      construct a heterogeneous collaboration network, in which there
    Output: The latent node embeddings X ∈ R |V |×d                                  are three types of nodes: authors, papers, and venues. The links
                                                                                     represent different types of relationships among three sets of nodes—
    initialize X ;
                                                                                     such as collaboration relationships on a paper.
    for i = 1 → w do                                                                    The DBIS dataset was constructed and used by Sun et al. [26]. It
         for v ∈ V do                                                                covers 464 venues, their top-5000 authors, and corresponding 72,902
             MP = MetaPathRandomWalk(G, P, v, l) ;                                   publications. We also construct the heterogeneous collaboration
             X = HeterogeneousSkipGram(X, k, MP) ;                                   networks from DBIS wherein a link may connect two authors, one
         end                                                                         author and one paper, as well as one paper and one venue.
    end
    return X ;                                                                       4.1    Experimental Setup
    MetaPathRandomWalk(G, P, v, l)                                                   We compare metapath2vec and metapath2vec++ with several recent
    MP[1] = v ;                                                                      network representation learning methods:
    for i = 1 → l−1 do                                                               (1) DeepWalk [22] / node2vec [8]: With the same random walk
        draw u according to Eq. 3 ;                                                      path input (p=1 & q=1 in node2vec), we find that the choice be-
        MP[i+1] = u ;                                                                    tween hierarchical softmax (DeepWalk) and negative sampling
    end                                                                                  (node2vec) techniques does not yield significant differences.
    return MP ;                                                                          Therefore we use p=1 and q=1 [8] in node2vec for comparison.
                                                                                     (2) LINE [30]: We use the advanced version of LINE by considering
    HeterogeneousSkipGram(X, k, MP)                                                      both the 1st- and 2nd-order of node proximity;
    for i = 1 → l do                                                                 (3) PTE [29]: We construct three bipartite heterogeneous networks
       v = MP[i] ;                                                                       (author–author, author–venue, venue–venue) and restrain it as
        for j = max(0, i-k) → min(i+k, l) & j , i do                                     an unsupervised embedding method;
            c t = MP[j] ;                                                            (4) Spectral Clustering [33] / Graph Factorization [2]: With the
                                     ∂ O(X)
          X new = X old − η ·          ∂X (Eq. 7) ;
                                                                                         same treatment to these methods in node2vec [8], we exclude
                                                                                         them from our comparison, as previous studies have demon-
       end
                                                                                         strated that they are outperformed by DeepWalk and LINE.
    end
       ALGORITHM 1: The metapath2vec++ Algorithm.                                       For all embedding methods, we use the same parameters listed
                                                                                     below. In addition, we also vary each of them and fix the others for
                                                                                     examining the parameter sensitivity of the proposed methods.
whose gradients are derived as follows:                                              (1) The number of walks per node w: 1000;
                                                                                     (2) The walk length l: 100;
              ∂O(X)
                    = (σ (Xutm · Xv − Ic t [utm ]))Xv                         (7)    (3) The vector dimension d: 128 (LINE: 128 for each order);
              ∂Xutm                                                                  (4) The neighborhood size k: 7;
                       M
              ∂O(X)                                                                  (5) The size of negative samples: 5.
                          (σ (Xutm · Xv − Ic t [utm ]))Xutm
                      X
                    =
               ∂Xv    m=0
                                                                                        For metapath2vec and metapath2vec++, we also need to specify
                                                                                     the meta-path scheme to guide random walks. We surveyed most
where Ic t [utm ] is an indicator function to indicate whether utm is                of the meta-path-based work and found that the most commonly
the neighborhood context node c t and when m = 0, ut0 = c t . The                    and effectively used meta-path schemes in heterogeneous academic
model is optimized by using stochastic gradient descent algorithm.                   networks are “APA” and “APVPA” [12, 25–27]. Notice that “APA”
The pseudo code of metapath2vec++ is listed in Algorithm 1.                          denotes the coauthor semantic, that is, the traditional (homoge-
                                                                                     neous) collaboration links / relationships. “APVPA” represents the
4    EXPERIMENTS                                                                     heterogeneous semantic of authors publishing papers at the same
In this section, we demonstrate the efficacy and efficiency of the                   venues. Our empirical results also show that this simple meta-path
presented metapath2vec and metapath2vec++ frameworks for het-                        scheme “APVPA” can lead to node embeddings that can be general-
erogeneous network representation learning.                                          ized to diverse heterogeneous academic mining tasks, suggesting its
Data. We use two heterogeneous networks, including the AMiner                        applicability to potential applications for academic search services.
Computer Science (CS) dataset [31] and the Database and Infor-                          We evaluate the quality of the latent representations learned
mation Systems (DBIS) dataset [26]. Both datasets and code are                       by different methods over three classical heterogeneous network
publicly available1 . This AMiner CS dataset consists of 9,323,739                   mining tasks, including multi-class node classification [13], node
                                                                                     clustering [27], and similarity search [26]. In addition, we also use
1 The network data, learned latent representations, labeled ground truth data, and   the embedding projector in TensorFlow [1] to visualize the node
source code can be found at https://ericdongyx.github.io/metapath2vec/m2v.html       embeddings learned from the heterogeneous academic networks.
                                      Table 2: Multi-class venue node classification results in AMiner data.

                 Metric              Method              5%        10%       20%      30%      40%      50%      60%      70%      80%      90%

                              DeepWalk/node2vec        0.0723     0.1396    0.1905   0.2795   0.3427   0.3911   0.4424   0.4774   0.4955   0.4457
                                LINE (1st+2nd)         0.2245     0.4629    0.7011   0.8473   0.8953   0.9203   0.9308   0.9466   0.9410   0.9466
                Macro-F1
                                     PTE               0.1702     0.3388    0.6535   0.8304   0.8936   0.9210   0.9352   0.9505   0.9525   0.9489
                                 metapath2vec          0.3033     0.5247    0.8033   0.8971   0.9406   0.9532   0.9529   0.9701   0.9683   0.9670
                                metapath2vec++         0.3090     0.5444    0.8049   0.8995   0.9468   0.9580   0.9561   0.9675   0.9533   0.9503

                              DeepWalk/node2vec        0.1701     0.2142    0.2486   0.3266   0.3788   0.4090   0.4630   0.4975   0.5259   0.5286
                                LINE (1st+2nd)         0.3000     0.5167    0.7159   0.8457   0.8950   0.9209   0.9333   0.9500   0.9556   0.9571
                Micro-F1
                                     PTE               0.2512     0.4267    0.6879   0.8372   0.8950   0.9239   0.9352   0.9550   0.9667   0.9571
                                 metapath2vec          0.4173     0.5975    0.8327   0.9011   0.9400   0.9522   0.9537   0.9725   0.9815   0.9857
                               metapath2vec++          0.4331     0.6192    0.8336   0.9032   0.9463   0.9582   0.9574   0.9700   0.9741   0.9786


                                     Table 3: Multi-class author node classification results in AMiner data.

                 Metric              Method              5%        10%       20%      30%      40%      50%      60%      70%      80%      90%

                              DeepWalk/node2vec        0.7153     0.7222    0.7256   0.7270   0.7273   0.7274   0.7273   0.7271   0.7275   0.7275
                                LINE (1st+2nd)         0.8849     0.8886    0.8911   0.8921   0.8926   0.8929   0.8934   0.8936   0.8938   0.8934
                Macro-F1
                                     PTE               0.8898     0.8940     0.897   0.8982   0.8987   0.8990   0.8997   0.8999   0.9002   0.9005
                                 metapath2vec          0.9216     0.9262    0.9292   0.9303   0.9309   0.9314   0.9315   0.9316   0.9319   0.9320
                                metapath2vec++         0.9107     0.9156    0.9186   0.9199   0.9204   0.9207   0.9207   0.9208   0.9211   0.9212

                              DeepWalk/node2vec        0.7312     0.7372    0.7402   0.7414   0.7418   0.7420   0.7419   0.7420   0.7425   0.7425
                                LINE (1st+2nd)         0.8936     0.8969    0.8993   0.9002   0.9007   0.9010   0.9015   0.9016   0.9018   0.9017
                Micro-F1
                                     PTE               0.8986     0.9023    0.9051   0.9061   0.9066   0.9068   0.9075   0.9077   0.9079   0.9082
                                 metapath2vec          0.9279     0.9319    0.9346   0.9356   0.9361   0.9365   0.9365   0.9365   0.9367   0.9369
                                metapath2vec++         0.9173     0.9217    0.9243   0.9254   0.9259   0.9261   0.9261   0.9262   0.9264   0.9266



4.2     Multi-Class Classification                                                     Results. Tables 2 and 3 list the eight-class classification results.
For the classification task, we use third-party labels to determine                    Overall, the proposed metapath2vec and metapath2vec++ models
the class of each node. First, we match the eight categories2 of                       consistently and significantly outperform all baselines in terms
venues in Google Scholar3 with those in AMiner data. Among all of                      of both metrics. When predicting for the venue category, the ad-
the 160 venues (20 per category × 8 categories), 133 of them are suc-                  vantage of both metapath2vec and metapath2vec++ are particular
cessfully matched and labeled correspondingly (Most of unmatched                       strong given a small size of training data. Given 5% of nodes as train-
venues are pre-print venues, such as arXiv). Second, for each au-                      ing data, for example, metapath2vec and metapath2vec++ achieve
thor who published in these 133 venues, his / her label is assigned                    0.08–0.23 (relatively 35–319%) improvements in terms of Macro-
to the category with the majority of his / her publications, and a                     F1 and 0.13–0.26 (relatively 39–145%) gains in terms of Micro-F1
tie is resolved by random selection among the possible categories;                     over DeepWalk / node2vec, LINE, and PTE. When predicting for
246,678 authors are labeled with research category.                                    authors’ categories, the performance of each method is relatively
   Note that the node representations are learned from the full                        stable when varying the train-test split. The constant gain achieved
dataset. The embeddings of above labeled nodes are then used as                        by the proposed methods is around 2-3% over LINE and PTE, and
the input to a logistic regression classifier. In the classification                   ∼20% over DeepWalk / node2vec.
experiments, we vary the size of the training set from 5% to 90%                          In summary, metapath2vec and metapath2vec++ learn signifi-
and the remaining nodes for testing. We repeat each prediction                         cantly better heterogeneous node embeddings than current state-
experiment ten times and report the average performance in terms                       of-the-art methods, as measured by multi-class classification perfor-
of both Macro-F1 and Micro-F1 scores.                                                  mance. The advantage of the proposed methods lies in their proper
                                                                                       consideration and accommodation of the network heterogeneity
                                                                                       challenge—the existence of multiple types of nodes and relations.
2 1. Computational Linguistics, 2. Computer Graphics, 3. Computer Networks &
                                                                                       Parameter sensitivity. In skip-gram-based representation learn-
Wireless Communication, 4. Computer Vision & Pattern Recognition, 5. Computing
Systems, 6. Databases & Information Systems, 7. Human Computer Interaction, and 8.     ing models, there exist several common parameters (see Section
Theoretical Computer Science.                                                          4.1). We conduct a sensitivity analysis of metapath2vec++ to these
3 https://scholar.google.com/citations?view op=top venues&hl=en&vq=eng. Accessed
                                                                                       parameters. Figure 3 shows the classification results as a function
on February, 2017.
             1                                               1                                              1                                                   1


           0.95                                            0.95                                           0.95                                                0.95


      F1    0.9                                                                                      F1    0.9
                                                      F1    0.9                                                                                          F1    0.9
                               venue Macro-F1                                                                                          venue Macro-F1
           0.85                venue Micro-F1                                     venue Macro-F1          0.85                         venue Micro-F1                                venue Macro-F1
                               author Macro-F1             0.85                   venue Micro-F1                                       author Macro-F1        0.85                   venue Micro-F1
                               author Micro-F1                                    author Macro-F1                                      author Micro-F1                               author Macro-F1
            0.8                                                                   author Micro-F1          0.8                                                                       author Micro-F1
                   10
                   200              15           0          0.8                                                   6        6    51     64       89             0.8
                                                                                                                 12 4
                      0                                                                                                                            6
                   40
                   60 0
                      0               00    20                                                                     8    25
                                                                                                                         38        2     0     10
                   80
                  10  0
                     00                        0                  40 60 80 100        150      200                          4                     24                 3   5      7    9    11 13   15
                       #walks per node w                                   walk length l                                     #dimensions d                                   neighborhod size k

                  (a) #walks per node w                               (b) walk length l                             (c) #dimensions d                                (d) neighborhood size k


       Figure 3: Parameter sensitivity in multi-class node classification. 50% as training data and the remaining as test data.


   Table 4: Node clustering results (NMI) in AMiner data.                                                 by each method is input to a clustering model. Here we leverage
                                                                                                          the k-means algorithm to cluster the data and evaluate the cluster-
                           methods                   venue           author                               ing results in terms of normalized mutual information (NMI) [26].
                   DeepWalk/node2vec                 0.1952          0.2941                               In addition, we also report metapath2vec++’s sensitivity with re-
                                                                                                          spect to different parameter choices. All clustering experiments are
                     LINE (1st+2nd)                  0.8967          0.6423
                                                                                                          conducted 10 times and the average performance is reported.
                          PTE                        0.9060          0.6483
                                                                                                          Results. Table 4 shows the node clustering results as measured
                      metapath2vec                   0.9274          0.7470                               by NMI in the AMiner CS data. Overall, the table demonstrates
                    metapath2vec++                   0.9261          0.7354                               that metapath2vec and metapath2vec++ outperform all the compar-
                                                                                                          ative methods. When clustering for venues, the task is trivial as
                                                                                                          evident from the high NMI scores produced by most of the methods:
                                                                                                          metapath2vec, metapath2vec++, LINE, and PTE. Nevertheless, the
of one chosen parameter when the others are controlled for. In                                            proposed two methods outperform LINE and PTE by 2–3%. The
general, we find that in Figures 3(a) and 3(b) the number of walks w                                      author clustering task is more challenging than the venue case, and
rooting from each node and the length l of each walk are positive to                                      the gain obtained by metapath2vec and metapath2vec++ over the
the author classification performance, while they are surprisingly                                        best baselines (LINE and PTE) is more significant—around 13–16%.
inconsequential for inferring venue nodes’ categories as measured                                            In summary, metapath2vec and metapath2vec++ generate more
by Macro-F1 and Micro-F1 scores. The increase of author clas-                                             appropriate embeddings for different types of nodes in the network
sification performance converges as w and l reach around 1000                                             than comparative baselines, suggesting their ability to capture and
and 100, respectively. Similarly, Figures 3(c) and 3(d) suggest that                                      incorporate the underlying structural and semantic relationships
the number of embedding dimensions d and neighborhood size                                                between various types of nodes in heterogeneous networks.
k are again of relatively little relevance to the predictive task for
                                                                                                          Parameter sensitivity. Following the same experimental proce-
venues, and k on the other hand is positively crucial to determine
                                                                                                          dure in classification, we study the parameter sensitivity of meta-
the class of a venue. However, the descending lines as the increase
                                                                                                          path2vec++ as measured by the clustering performance. Figure 4
of k for author classifications imply that a smaller neighborhood
                                                                                                          shows the clustering performance as a function of each of the four
size actually produces the best embeddings for separating authors.
                                                                                                          parameters when fixing the other three. From Figures 4(a) and 4(b),
This finding differs from those in a homogeneous environment [8],
                                                                                                          we can observe that the balance between computational cost (a
wherein the neighborhood size generally shows a positive effect
                                                                                                          small w and l in x-axis) and efficacy (a high NMI in y-axis) can be
on node classification.
                                                                                                          achieved at around w = 800∼1000 and l = 100 for the clustering of
    According to the analysis, metapath2vec++ is not strictly sen-
                                                                                                          both authors and venues. Further, different from the positive effect
sitive to these parameters and is able to reach high performance
                                                                                                          of increasing w and l on author clustering, d and k are negatively
under a cost-effective parameter choice (the smaller, the more ef-
                                                                                                          correlated with the author clustering performance, as observed from
ficient). In addition, our results also indicate that those common
                                                                                                          Figures 4(c) and 4(d). Similarly, the venue clustering performance
parameters show different functions for heterogeneous network
                                                                                                          also shows an descending trend with an increasing d, while on the
embedding with those in homogeneous network cases, demonstrat-
                                                                                                          other hand, we observe a first-increasing and then-decreasing NMI
ing the request of different ideas and solutions for heterogeneous
                                                                                                          line when k is increased. Both figures together imply that d = 128
network representation learning.
                                                                                                          and k = 7 are capable of embedding heterogeneous nodes into latent
                                                                                                          space for promising clustering outcome.
4.3    Node Clustering
We illustrate how the latent representations learned by embed-
ding methods can help the node clustering task in heterogeneous                                           4.4           Case Study: Similarity Search
networks. We employ the same eight-category author and venue                                              We conduct two case studies to demonstrate the efficacy of our
nodes used in the classification task above. The learned embeddings                                       methods. We select 16 top CS conferences from the corresponding
              1                                                1                                               1                                                                 1
                                                                                                                                                 venue clustering
                                                                                                                                                 author clustering
             0.9                                              0.9                                             0.9                                                               0.9


       NMI   0.8                                                                                        NMI   0.8
                                                        NMI   0.8                                                                                                         NMI   0.8

             0.7                                                                                              0.7
                                venue clustering              0.7                                                                                                               0.7
                                author clustering                                   venue clustering                                                                                                   venue clustering
             0.6                                                                    author clustering         0.6                                                                                      author clustering
                    10
                    2000             15             0         0.6                                                    6
                                                                                                                    12 4
                                                                                                                             6            51     64        89 6                 0.6
                    40
                    60 0
                       0               00      20                                                                     8    25
                                                                                                                           38 4              2     0      10
                    80
                   10  0
                      00                          0                 40 60 80 100        150       200                                                        24                       3   5      7    9    11 13     15
                        #walks per node w                                    walk length l                                      #dimensions d                                                 neighborhod size k

                   (a) #walks per node w                                (b) walk length l                              (c) #dimensions d                                              (d) neighborhood size k


                                                                    Figure 4: Parameter sensitivity in clustering.

                           TREC     SIGIR
                      EMNLP    ECIR       WWW                                                                                             40
                      ACL NAACL CIKM WSDM                                                                                                                  metapath2vec
                                                                                                                                                           metapath2vec++
             IJCAI                ICDM       ICDE
                    AAAI             SDM        SIGMOD
                                                                                                                                          32
              ECAI        ICML   KDD       VLDB
                      NIPS AISTATS                                                                                                        24

                                                                                                                                speedup
                   CVPR              SODA
               ECCV ICCV                 FOCS
                                   STOC          CCS                                                                                      16
                         VIS                 S&P USEC
                    SI3D SIGGRAPH MICRO HotNets

                                                                                   SIGCOMM
                                                                                                                                           8
                   UIST CHI       HPCA ISCA OSDI
                                          SOSP NSDI                                                                                        4
                 CSCW             ASE         HotOS
                                                                                                                                           2
                                                                                                                                           1

                               ICSE FSE                                                                                                           12 4       8       16          24
                                                                                                                                                                          #threads
                                                                                                                                                                                                  32           40



Figure 5: 2D t-SNE projections of the 128D embeddings of                                                      Figure 6: Scalability of metapath2vec and metapath2vec++.
48 CS venues, three each from 16 sub-fields.

                                                                                                              models. First, we project multiple types of nodes—16 top CS confer-
sub-fields in the AMiner CS data and another 5 from the DBIS data.                                            ences and corresponding top-profile authors—into the same space in
This results in a total of 21 query nodes. We use cosine similarity                                           Figure 1. From Figure 1(d), we can clearly see that metapath2vec++
to determine the distance (similarity) between the query node and                                             is able to automatically organize these two types of nodes and im-
the remaining others.                                                                                         plicitly learn the internal relationships between them, indicated by
    Table 5 lists the top ten similar results for querying the 16 leading                                     the similar directions and distances of the arrows connecting each
conferences in corresponding computer science sub-fields. One can                                             pair of them, such as J. Dean → OSDI, C. D. Manning → ACL, R. E.
observe that for the query “ACL”, for example, metapath2vec++                                                 Tarjan → FOCS, M. I. Jordan → NIPS, and so on. In addition, these
returns venues with the same focus—natural language processing,                                               two types of nodes are clearly located in two separate and straight
such as EMNLP (1st ), NAACL (2nd ), Computational Linguistics                                                 columns. Neither of these two results can be made by the recent
(3r d ), CoNLL (4t h ), COLING (5t h ), and so on. Similar performance                                        network embedding models in Figures 1(a) and 1(b).
can be also achieved when querying the other conferences from                                                    As to metapath2vec, instead of separating the two types of nodes
various fields. More surprisingly, we find that in most cases, the                                            into two columns, it is capable of grouping each pair of one venue
top three results cover venues with similar prestige to the query                                             and its corresponding author closely, such as R. E. Tarjan and FOCS,
one, such as STOC to FOCS in theory, OSDI to SOSP in system,                                                  H. Jensen and SIGGRAPH, H. Ishli and CHI, R. Agrawal and SIG-
HPCA to ISCA in architecture, CCS to S&P in security, CSCW to                                                 MOD, etc. Together, both models arrange nodes from similar fields
CHI in human-computer interaction, EMNLP to ACL in NLP, ICML                                                  close to each other and dissimilar ones distant from each other, such
to NIPS in machine learning, WSDM to WWW in Web, AAAI to                                                      as the “Core CS” cluster of systems (OSDI), networking (SIGCOMM),
IJCAI in artificial intelligence, PVLDB to SIGMOD in database, etc.                                           security (S&P), and architecture (ISCA), as well as the “Big AI” clus-
Similar results can also be observed in Tables 6 and 1, which show                                            ter of data mining (KDD), information retrieval (SIGIR), artificial
the similarity search results for the DBIS network.                                                           intelligence (AI), machine learning (NIPS), NLP (ACL), and vision
                                                                                                              (CVPR). These groupings are also reflected by their corresponding
4.5    Case Study: Visualization                                                                              author nodes.
We employ the TensorFlow embedding projector to further visualize                                                Second, Figure 5 visualizes the latent vectors—learned by meta-
the low-dimensional node representations learned by embedding                                                 path2vec++—of 48 venues used in similarity search of Section 4.4,
                                                         Table 5: Case study of similarity search in AMiner Data
    Rank     ACL          NIPS     IJCAI       CVPR        FOCS      SOSP      ISCA      S&P      ICSE    SIGGRAPH   SIGCOMM     CHI       KDD     SIGMOD     SIGIR     WWW

      0      ACL          NIPS     IJCAI       CVPR        FOCS      SOSP      ISCA      S&P      ICSE    SIGGRAPH   SIGCOMM     CHI       KDD     SIGMOD     SIGIR     WWW
      1     EMNLP         ICML     AAAI        ECCV        STOC      TOCS     HPCA       CCS      TOSEM     TOG        CCR      CSCW       SDM      PVLDB      ECIR     WSDM
      2     NAACL        AISTATS     AI        ICCV       SICOMP     OSDI     MICRO     NDSS       FSE      SI3D     HotNets    TOCHI      TKDD      ICDE     CIKM      CIKM
      3       CL          JMLR      JAIR        IJCV       SODA     HotOS     ASPLOS   USENIX S    ASE       RT       NSDI       UIST      ICDM     DE Bull    IR J     TWEB
      4      CoNLL         NC      ECAI        ACCV         A-R    SIGOPS E   PACT     ACSAC      ISSTA     CGF      CoNEXT       DIS      DMKD     VLDBJ     TREC      ICWSM
      5     COLING        MLJ       KR         CVIU        TALG      ATC       ICS       JCS       E SE     NPAR       IMC       HCI       KDD E    EDBT      SIGIR F    HT
      6     IJCNLP        COLT     AI Mag      BMVC        ICALP     NSDI     HiPEAC   ESORICS    MSR        Vis      TON      MobileHCI   WSDM     TODS      ICTIR     SIGIR
      7      NLE           UAI     ICAPS        ICPR       ECCC      OSR      PPOPP      TISS     ESEM       JGT     INFOCOM   INTERACT    CIKM      CIDR     WSDM      KDD
      8      ANLP         KDD        CI       EMMCVPR      TOC      ASPLOS     ICCD    ASIACCS    A SE     VisComp    PAM       GROUP      PKDD    SIGMOD R    TOIS      TIT
      9      LREC         CVPR      AIPS       T on IP     JAlG     EuroSys    CGO      RAID      ICPC       GI      MobiCom   NordiCHI    ICML     WebDB      IPM      WISE
      10     EACL         ECML      UAI        WACV        ITCS    SIGCOMM    ISLPED    CSFW      WICSA      CG       IPTPS    UbiComp     PAKDD    PODS       AIRS     WebSci




    Table 6: Case study of similarity search in DBIS Data                                         11–12× speedup with 16 cores and 24–32× speedup with 40 cores
                                                                                                  used. By using 40 cores, metapath2vec++’s learning process costs
                   Rank     KDD     SIGMOD      SIGIR     WWW      WSDM
                                                                                                  only 9 minutes for embedding the full AMiner CS network, which is
                     0      KDD     SIGMOD      SIGIR     WWW      WSDM
                                                                                                  composed of over 9 million authors with 3 million papers published
                     1      SDM     PVLDB       TREC       CIKM    WWW
                     2     ICDM      ICDE       CIKM       SIGIR   SIGIR
                                                                                                  in more than 3800 venues. Overall, the proposed metapath2vec and
                     3     DMKD      TODS        IPM       KDD      KDD                           metapath2vec++ models are efficient and scalable for large-scale
                     4     KDD E     VLDBJ        IRJ      ICDE    AIRWeb                         heterogeneous networks with millions of nodes.
                     5     PKDD      PODS        ECIR     TKDE     CIKM
                     6     PAKDD     EDBT        TOIS      VLDB    WebDB
                     7     TKDE      CIDR       WWW       TOTT     ICDM
                     8     CIKM      TKDE       JASIST    SIGMOD   VLDB
                                                                                                  5       RELATED WORK
                     9      ICDE     ICDT       JASIS     WebDB    VLDBJ                          Network representation learning can be traced back to the usage of
                    10     TKDD     DE Bull    SIGIR F     WISE     SDM                           latent factor models for network analysis and graph mining tasks
                                                                                                  [10, 34], such as the application of factorization models for rec-
                                                                                                  ommendation systems [14, 16], node classification [32], relational
three each from 16 sub-fields. We can see that conferences from                                   mining [19], and role discovery [9]. This rich line of research focuses
the same domain are geographically grouped to each other and                                      on factorizing the matrix/tensor format (e.g., the adjacency matrix)
each group is well separated from others, further demonstrating                                   of a network, generating latent-dimension features for nodes or
the embedding ability of metapath2vec++. In addition, similar to the                              edges in this network. However, the computational cost of decom-
observation in Figure 1, we can also notice that the heterogeneous                                posing a large-scale matrix/tensor is usually very expensive, and
embeddings are able to unveil the similarities across different do-                               also suffers from its statistical performance drawback [8], making it
mains, including the “Core CS” sub-field cluster at the bottom right                              neither practical nor effective for addressing tasks in big networks.
and the “Big AI” sub-field cluster at the top right.                                                 With the advent of deep learning techniques, significant effort
   Thus, Figures 1 and 5 intuitively demonstrate metapath2vec++’s                                 has been devoted to designing neural network-based representa-
novel capability to discover, model, and capture the underlying                                   tion learning models. For example, Mikolov et al. proposed the
structural and semantic relationships between multiple types of                                   word2vec framework—a two-layer neural network—to learn the
nodes in heterogeneous networks.                                                                  distributed representations of words in natural language [17, 18].
                                                                                                  Building on word2vec, Perozzi et al. suggested that the “context”
4.6        Scalability                                                                            of a node can be denoted by their co-occurrence in a random walk
In the era of big (network) data, it is necessary to demonstrate                                  path [22]. Formally, they put random walkers over networks to
the scalability of the proposed network embedding models. The                                     record their walking paths, each of which is composed of a chain
metapath2vec and metapath2vec++ methods can be parallelized by                                    of nodes that could be considered as a “sentence” of words in a text
using the same mechanism as word2vec and node2vec [8, 18]. All                                    corpus. More recently, in order to diversify the neighborhood of
codes are implemented in C and C++ and our experiments are                                        a node, Grover & Leskovec presented biased random walkers—a
conducted in a computing server with Quad 12 (48) core 2.3 GHz                                    mixture of breadth-first and width-first search procedures—over
Intel Xeon CPUs E7-4850. We run experiments on the AMiner CS                                      networks to produce paths of nodes [8]. With node paths gener-
data with the default parameters with different number of threads,                                ated, both works leveraged the skip-gram architecture in word2vec
i.e., 1, 2, 4, 8, 16, 24, 32, 40, each of them utilizing one CPU core.                            to model the structural correlations between nodes in a path. In
    Figure 6 shows the speedup of metapath2vec & metapath2vec++                                   addition, several other methods have been proposed for learning
over the single-threaded case. Optimal speedup performance is                                     representations in networks [4, 5, 11, 20, 23]. In particular, to learn
denoted by the dashed y = x line, which represents perfect distribu-                              network embeddings, Tang et al. decomposed a node’s context
tion and execution of computation across all CPU cores. In general,                               into first-order (friends) and second-order (friends’ friends) prox-
we find that both methods achieve acceptable sublinear speedups                                   imity [30], which was further developed into a semi-supervised
as both lines are close to the optimal line. In specific, they can reach                          model PTE for embedding text data [29].
   Our work furthers this direction of investigation by designing                     [6] Yuxiao Dong, Jing Zhang, Jie Tang, Nitesh V. Chawla, and Bai Wang. 2015.
the metapath2vec and metapath2vec++ models to capture hetero-                             CoupledLP: Link Prediction in Coupled Networks. In KDD ’15. ACM, 199–208.
                                                                                      [7] Yoav Goldberg and Omer Levy. 2014. word2vec Explained: deriving Mikolov et
geneous structural and semantic correlations exhibited from large-                        al.’s negative-sampling word-embedding method. CoRR abs/1402.3722 (2014).
scale networks with multiple types of nodes, which can not be                         [8] Aditya Grover and Jure Leskovec. 2016. Node2Vec: Scalable Feature Learning
                                                                                          for Networks. In KDD ’16. ACM, 855–864.
handled by previous models, and applying these models to a variety                    [9] Keith Henderson, Brian Gallagher, Tina Eliassi-Rad, Hanghang Tong, Sugato
of network mining tasks.                                                                  Basu, Leman Akoglu, Danai Koutra, Christos Faloutsos, and Lei Li. 2012. Rolx:
                                                                                          structural role extraction & mining in large graphs. In KDD ’12. ACM, 1231–1239.
                                                                                     [10] Peter D Hoff, Adrian E Raftery, and Mark S Handcock. 2002. Latent space ap-
6   CONCLUSION                                                                            proaches to social network analysis. Journal of the American Statistical association
                                                                                          97, 460 (2002), 1090–1098.
In this work, we formally define the representation learning prob-                   [11] Xiao Huang, Jundong Li, and Xia Hu. 2017. Label Informed Attributed Network
lem in heterogeneous networks in which there exist diverse types                          Embedding. In WSDM ’17. na.
of nodes and links. To address the network heterogeneity chal-                       [12] Zhipeng Huang, Yudian Zheng, Reynold Cheng, Yizhou Sun, Nikos Mamoulis,
                                                                                          and Xiang Li. 2016. Meta structure: Computing relevance in large heterogeneous
lenge, we propose the metapath2vec and metapath2vec++ meth-                               information networks. In KDD ’16. ACM, 1595–1604.
ods. We develop the meta-path-guided random walk strategy in                         [13] Ming Ji, Jiawei Han, and Marina Danilevsky. 2011. Ranking-based classification
                                                                                          of heterogeneous information networks. In KDD ’11. ACM, 1298–1306.
a heterogeneous network, which is capable of capturing both the                      [14] Yehuda Koren. 2008. Factorization meets the neighborhood: a multifaceted
structural and semantic correlations of differently typed nodes and                       collaborative filtering model. In KDD ’08. ACM, 426–434.
relations. To leverage this method, we formalize the heterogeneous                   [15] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. 2015. Deep learning. Nature
                                                                                          521, 7553 (2015), 436–444.
neighborhood function of a node, enabling the skip-gram-based                        [16] Hao Ma, Dengyong Zhou, Chao Liu, Michael R Lyu, and Irwin King. 2011.
maximization of the network probability in the context of multiple                        Recommender systems with social regularization. In WSDM ’11. 287–296.
types of nodes. Finally, we achieve effective and efficient optimiza-                [17] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient
                                                                                          Estimation of Word Representations in Vector Space. CoRR abs/1301.3781 (2013).
tion by presenting a heterogeneous negative sampling technique.                           http://arxiv.org/abs/1301.3781
Extensive experiments demonstrate that the latent feature repre-                     [18] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013.
                                                                                          Distributed representations of words and phrases and their compositionality. In
sentations learned by metapath2vec and metapath2vec++ are able to                         NIPS ’13. 3111–3119.
improve various heterogeneous network mining tasks, such as sim-                     [19] Jennifer Neville and David Jensen. 2005. Leveraging relational autocorrelation
ilarity search, node classification, and clustering. Our results can                      with latent group models. In Proceedings of the 4th international workshop on
                                                                                          Multi-relational mining. ACM, 49–55.
be naturally applied to real-world applications in heterogeneous                     [20] Mingdong Ou, Peng Cui, Jian Pei, Ziwei Zhang, and Wenwu Zhu. 2016. Asym-
academic networks, such as author, venue, and paper search in                             metric Transitivity Preserving Graph Embedding. In KDD ’16. ACM, 1105–1114.
academic search services.                                                            [21] Siddharth Pal, Yuxiao Dong, Bishal Thapa, Nitesh V Chawla, Ananthram Swami,
                                                                                          and Ram Ramanathan. 2016. Deep learning for network analysis: Problems,
   Future work includes various optimizations and improvements.                           approaches and challenges. In Military Communications Conference, MILCOM
For example, 1) the metapath2vec and metapath2vec++ models, as                            2016-2016. IEEE, 588–593.
                                                                                     [22] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. DeepWalk: Online
is also the case with DeepWalk and node2vec, face the challenge                           Learning of Social Representations. In KDD ’14. ACM, 701–710.
of large intermediate output data when sampling a network into a                     [23] Xiang Ren, Wenqi He, Meng Qu, Clare R Voss, Heng Ji, and Jiawei Han. 2016.
huge pile of paths, and thus identifying and optimizing the sampling                      Label noise reduction in entity typing by heterogeneous partial-label embedding.
                                                                                          In KDD ’16. ACM.
space is an important direction; 2) as is also the case with all meta-               [24] Xin Rong. 2014. word2vec Parameter Learning Explained. CoRR abs/1411.2738
path-based heterogeneous network mining methods, metapath2vec                             (2014). http://arxiv.org/abs/1411.2738
and metapath2vec++ can be further improved by the automatic                          [25] Yizhou Sun and Jiawei Han. 2012. Mining Heterogeneous Information Networks:
                                                                                          Principles and Methodologies. Morgan & Claypool Publishers.
learning of meaningful meta-paths; 3) extending the models to                        [26] Yizhou Sun, Jiawei Han, Xifeng Yan, Philip S. Yu, and Tianyi Wu. 2011. Pathsim:
incorporate the dynamics of evolving heterogeneous networks; and                          Meta path-based top-k similarity search in heterogeneous information networks.
                                                                                          In VLDB ’11. 992–1003.
4) generalizing the models for different genres of heterogeneous                     [27] Yizhou Sun, Brandon Norick, Jiawei Han, Xifeng Yan, Philip S. Yu, and Xiao Yu.
networks.                                                                                 2012. Integrating Meta-path Selection with User-guided Object Clustering in
                                                                                          Heterogeneous Information Networks. In KDD ’12. ACM, 1348–1356.
Acknowledgments. We would like to thank Reid Johnson for dis-                        [28] Yizhou Sun, Yintao Yu, and Jiawei Han. 2009. Ranking-based Clustering of
                                                                                          Heterogeneous Information Networks with Star Network Schema. In KDD ’09.
cussions and suggestions. This work is supported by the Army Re-                          ACM, 797–806.
search Laboratory under Cooperative Agreement Number W911NF-                         [29] Jian Tang, Meng Qu, and Qiaozhu Mei. 2015. PTE: Predictive Text Embedding
09-2-0053 and the National Science Foundation (NSF) grants CNS-                           Through Large-scale Heterogeneous Text Networks. In KDD ’15. ACM, 1165–
                                                                                          1174.
1629914 and IIS-1447795.                                                             [30] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei.
                                                                                          2015. LINE: Large-scale Information Network Embedding.. In WWW ’15. ACM.
                                                                                     [31] Jie Tang, Jing Zhang, Limin Yao, Juanzi Li, Li Zhang, and Zhong Su. 2008. Ar-
REFERENCES                                                                                netMiner: Extraction and Mining of Academic Social Networks. In KDD ’08.
[1] Martı́n Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey           990–998.
    Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, and others. 2016.        [32] Lei Tang and Huan Liu. 2009. Relational learning via latent social dimensions.
    TensorFlow: A system for large-scale machine learning. In OSDI ’16.                   In KDD ’09. 817–826.
[2] Amr Ahmed, Nino Shervashidze, Shravan Narayanamurthy, Vanja Josifovski, and      [33] Lei Tang and Huan Liu. 2011. Leveraging social media networks for classification.
    Alexander J. Smola. 2013. Distributed Large-scale Natural Graph Factorization.        DMKD 23, 3 (2011), 447–478.
    In WWW ’13. ACM, 37–48.                                                          [34] Shuicheng Yan, Dong Xu, Benyu Zhang, Hong-Jiang Zhang, Qiang Yang, and
[3] Yoshua Bengio, Aaron Courville, and Pierre Vincent. 2013. Representation              Stephen Lin. 2007. Graph embedding and extensions: A general framework for
    learning: A review and new perspectives. IEEE TPAMI 35, 8 (2013), 1798–1828.          dimensionality reduction. IEEE TPAMI 29, 1 (2007).
[4] Shiyu Chang, Wei Han, Jiliang Tang, Guo-Jun Qi, Charu C. Aggarwal, and           [35] Jing Zhang, Jie Tang, Cong Ma, Hanghang Tong, Yu Jing, and Juanzi Li. 2015.
    Thomas S. Huang. 2015. Heterogeneous Network Embedding via Deep Architec-             Panther: Fast top-k similarity search on large networks. In KDD ’15. ACM,
    tures. In KDD ’15. ACM, 119–128.                                                      1445–1454.
[5] Ting Chen and Yizhou Sun. 2017. Task-Guided and Path-Augmented Heteroge-
    neous Network Embedding for Author Identification. In WSDM ’17. ACM.

