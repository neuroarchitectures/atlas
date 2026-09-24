# TriCL Lee Shin 2023

> Source: `TriCL_Lee_Shin_2023.pdf`

---

                                                                               I’m Me, We’re Us, and I’m Us:
                                                                   Tri-directional Contrastive Learning on Hypergraphs
                                                                                        Dongjin Lee1 and Kijung Shin1,2
                                                         1
                                                             School of Electrical Engineering, 2 Kim Jaechul Graduate School of AI, KAIST, South Korea
                                                                                         {dongjin.lee, kijungs}@kaist.ac.kr



                                                                    Abstract                                 ich, and Leskovec 2016), ranking (Yu et al. 2021), and out-




arXiv:2206.04739v4 [cs.LG] 5 Jan 2023
                                                                                                             lier detection (Lee, Choe, and Shin 2022).
                                          Although machine learning on hypergraphs has attracted con-
                                          siderable attention, most of the works have focused on (semi-         Previous studies have largely focused on developing en-
                                          )supervised learning, which may cause heavy labeling costs         coder architectures so-called hypergraph neural networks
                                          and poor generalization. Recently, contrastive learning has        for hypergraph-structured data (Feng et al. 2019; Yadati
                                          emerged as a successful unsupervised representation learn-         et al. 2019; Dong, Sawin, and Bengio 2020; Bai, Zhang, and
                                          ing method. Despite the prosperous development of con-             Torr 2021; Arya et al. 2020), and in most cases, such hy-
                                          trastive learning in other domains, contrastive learning on        pergraph neural networks are trained in a (semi-)supervised
                                          hypergraphs remains little explored. In this paper, we pro-        way. However, data labeling is often time, resource, and
                                          pose TriCL (Tri-directional Contrastive Learning), a gen-          labor-intensive, and neural networks trained only in a super-
                                          eral framework for contrastive learning on hypergraphs. Its
                                                                                                             vised way can easily overfit and may fail to generalize (Rong
                                          main idea is tri-directional contrast, and specifically, it aims
                                          to maximize in two augmented views the agreement (a) be-           et al. 2020), making it difficult to be applied to other tasks.
                                          tween the same node, (b) between the same group of nodes,             Thus, self-supervised learning (Liu et al. 2022; Jaiswal
                                          and (c) between each group and its members. Together with          et al. 2020; Liu et al. 2021), which does not require labels,
                                          simple but surprisingly effective data augmentation and neg-       has become popular, and especially contrastive learning has
                                          ative sampling schemes, these three forms of contrast enable       achieved great success in computer vision (Chen et al. 2020;
                                          TriCL to capture both node- and group-level structural infor-      Hjelm et al. 2019) and natural language processing (Gao,
                                          mation in node embeddings. Our extensive experiments using         Yao, and Chen 2021). Contrastive learning has proved effec-
                                          14 baseline approaches, 10 datasets, and two tasks demon-          tive also for learning on (ordinary) graphs (Veličković et al.
                                          strate the effectiveness of TriCL, and most noticeably, TriCL
                                                                                                             2018b; Peng et al. 2020; Hassani and Khasahmadi 2020;
                                          almost consistently outperforms not just unsupervised com-
                                          petitors but also (semi-)supervised competitors mostly by sig-     Zhu et al. 2020, 2021b; You et al. 2020), and a common ap-
                                          nificant margins for node classification. The code and datasets    proach is to (a) create two augmented views from the input
                                          are available at https://github.com/wooner49/TriCL.                graph and (b) learn machine learning models to maximize
                                                                                                             the agreement between the two views.
                                                                                                                However, contrastive learning on hypergraphs remains
                                                               1    Introduction                             largely underexplored with only a handful of previous stud-
                                        Many real-world interactions are group-wise. Examples in-            ies (Xia et al. 2021; Zhang et al. 2021; Yu et al. 2021) (see
                                        clude collaborations of researchers, discussions on online           Section 2 for details). Especially, the following questions re-
                                        Q&A sites, group conversations on messaging apps, co-                main open: (Q1) what to contrast?, (Q2) how to augment a
                                        citations of documents, and co-purchases of items. A hy-             hypergraph?, and (Q3) how to select negative samples?
                                        pergraph, which is a generalized graph, allows an edge to               For Q1, which is our main focus, we propose tri-
                                        join an arbitrary number of nodes, and thus each such edge,          directional contrast. In addition to node-level contrast,
                                        which is called a hyperedge, naturally represents such group-        which is the only form of contrast employed in the previous
                                        wise interactions (Benson et al. 2018; Do et al. 2020; Lee,          studies, we propose the use of group-level and membership-
                                        Ko, and Shin 2020).                                                  level contrast. That is, in two augmented views, we aim to
                                           Recently, machine learning on hypergraphs has drawn               maximize agreements (a) between the same node, (b) be-
                                        a lot of attention from a broad range of fields, includ-             tween the same group of nodes, and (c) between each group
                                        ing social network analysis (Yang et al. 2019), recom-               and its members. These three forms of contrast are comple-
                                        mender systems (Xia et al. 2021), and bioinformatics (Zheng          mentary, leading to representations that capture both node-
                                        et al. 2019). Hypergraph-based approaches often outperform           and group-level (i.e., higher-order) relations in hypergraphs.
                                        graph-based ones on various machine learning tasks, includ-             In addition, for Q2, we demonstrate that combining two
                                        ing classification (Feng et al. 2019), clustering (Benson, Gle-      simple augmentation strategies (spec., membership corrup-
                                        Copyright © 2023, Association for the Advancement of Artificial      tion and feature corruption) is effective. For Q3, we reveal
                                        Intelligence (www.aaai.org). All rights reserved.                    that uniform random sampling is surprisingly successful,
and in our experiments, even an extremely small sample size       munity, thus information on subgroups (i.e., a smaller group
leads to marginal performance degradation.                        in a large community) cannot be used. On the other hand,
   Our proposed method TriCL, which is based on the afore-        TriCL can preserve and fully utilize such group information.
mentioned observations, is evaluated extensively using 14
                                                                  Hypergraph contrastive learning Contrastive learning
baseline approaches, 10 datasets, and two tasks. The most
                                                                  on hypergraphs is still in its infancy. Recently, several stud-
notable result is that, for node classification, TriCL outper-
                                                                  ies explore contrastive learning on hypergraphs (Zhang et al.
forms not just unsupervised competitors but also all (semi-
                                                                  2021; Xia et al. 2021; Yu et al. 2021). For example, Zhang
)supervised competitors on almost all considered datasets,
                                                                  et al. (2021) proposes S2 -HHGR for group recommenda-
mostly by considerable margins. Moreover, we demonstrate
                                                                  tion, which applies contrastive learning to remedy a data
the consistent effectiveness of tri-directional contrast, which
                                                                  sparsity issue. In particular, they propose a hypergraph aug-
is our main contribution.
                                                                  mentation scheme that uses a coarse- and fine-grained node
                                                                  dropout for each view. However, they do not consider group-
                   2    Related Work                              wise contrast. Although Xia et al. (2021) employ group-
Hypergraph learning Due to its enough expressiveness              wise contrast for session recommendation, they do not ac-
to capture higher-order structural information, learning on       count for node-wise and node-group pair-wise relationships
hypergraphs has received a lot of attention. Many recent          when constructing their contrastive loss. Moreover, these ap-
studies have focused on generalizing graph neural networks        proaches have been considered only in the context of group-
(GNNs) to hypergraphs (Feng et al. 2019; Bai, Zhang, and          based recommendation but not in the context of general rep-
Torr 2021; Yadati et al. 2019). Most of them redefine hy-         resentation learning.
pergraph message aggregation schemes based on clique
expansion (i.e., replacing hyperedges with cliques to ob-                      3    Proposed Method: TriCL
tain a graph) or its variants. While its simplicity is ap-        In this section, we describe TriCL, our proposed frame-
pealing, clique expansion causes structural distortion and        work for hypergraph contrastive learning. First, we intro-
leads to undesired information loss (Hein et al. 2013; Li         duce some preliminaries on hypergraphs and hypergraph
and Milenkovic 2018). On the other hand, HNHN (Dong,              neural networks, and then we elucidate the problem setting
Sawin, and Bengio 2020) prevents information loss by ex-          and details of the proposed method.
tending star expansion using two distinct weight matrices for
node- and hyperedge-side message aggregations. Arya et al.        3.1    Preliminaries
(2020) propose HyperSAGE for inductive learning on hy-            Hypergraphs and notation. A hypergraph, a set of hy-
pergraphs based on two-stage message aggregation. Several         peredges, is a natural extension of a graph, allowing the
studies attempt to unify hypergraphs and GNNs (Huang and          hyperedge to contain any number of nodes. Formally, let
Yang 2021; Zhang et al. 2022); and Chien et al. (2022) gen-       H = (V, E) be a hypergraph, where V = {v1 , v2 , . . . , v|V | }
eralize message aggregation methods as multiset functions
                                                                  is a set of nodes and E = {e1 , e2 , . . . , e|E| } is a set of hyper-
learned by Deep Sets (Zaheer et al. 2017) and Set Trans-
                                                                  edges, with each hyperedge is a non-empty subset of V . The
former (Lee et al. 2019). Most approaches above use (semi-
)supervised learning.                                             node feature matrix is represented by X ∈ R|V |×F , where
                                                                  xi = X[i, :]T ∈ RF is the feature of node vi . In general, a
Contrastive learning In the image domain, the latest con-         hypergraph can alternatively be represented by its incidence
trastive learning frameworks (e.g., SimCLR (Chen et al.           matrix H ∈ {0, 1}|V |×|E| , with entries defined as hij = 1
2020) and MoCo (He et al. 2020)) leverage the unchanging          if vi ∈ ej , and hij = 0 otherwise. In other words, hij = 1
semantics under various image transformations, such as ran-       when node vi and hyperedge ej form a membership. Each
dom flip, rotation, color distortion, etc, to learn visual fea-   hyperedge ej ∈ E is assigned a positive weight wj , and all
tures. They aim to learn distinguishable representations by       the weights formulate a diagonal matrix W ∈ R|E|×|E| . We
contrasting positive and negative pairs.                          use the diagonal matrix DV toP    represent the degree of ver-
   In the graph domain, DGI (Veličković et al. 2018b) com-      tices, where its entries di =        j wj hij . Also we use the
bines the power of GNNs and contrastive learning, which           diagonal matrix DE to denote        the    degree of hyperedges,
seeks to maximize the mutual information between node             where its element δj =
                                                                                             P
                                                                                                  hij   represents      the number of
                                                                                               i
embeddings and graph embeddings. Recently, a number of            nodes connected by the hyperedge ej .
graph contrastive learning approaches (You et al. 2020; Zhu
et al. 2020, 2021b; Hassani and Khasahmadi 2020) that fol-        Hypergraph neural networks. Modern hypergraph neu-
low a common framework (Chen et al. 2020) have been pro-          ral networks (Feng et al. 2019; Yadati et al. 2019; Bai,
posed. Although these methods have achieved state-of-the-         Zhang, and Torr 2021; Dong, Sawin, and Bengio 2020;
art performance on their task of interest, they cannot natu-      Arya et al. 2020; Chien et al. 2022) follow a two-stage
rally exploit group-wise interactions, which we focus on in       neighborhood aggregation strategy: node-to-hyperedge and
this paper. More recently, gCooL (Li, Jing, and Tong 2022)        hyperedge-to-node aggregation. They iteratively update the
utilizes community contrast, which is a similar concept to        representation of a hyperedge by aggregating representa-
membership-level contrast in TriCL, to maximize commu-            tions of its incident nodes and the representation of a node
nity consistency between two augmented views. However,            by aggregating representations of its incident hyperedges.
                                                                                         0                      00
gCooL has an information loss when constructing a com-            Let P(k) ∈ R|V |×Fk and Q(k) ∈ R|E|×Fk be the node
                                                  Node-level contrast       Group-level contrast          Membership-level contrast

                      Node feature masking                                                          𝐏!
                                                                                   Hypergraph                   𝑔# (⋅)                                       𝒁!
                      Membership masking
                                             𝒯"                                     Encoder
                                                                                                    𝐐!
                                                                                      𝑓# (⋅)                                          𝑔" (⋅)                 𝒀!
                                                                                                              Node
                                                         ℋ! = (𝐗 ! , 𝐇!)                                    Projection
                                                                                   Shared weights             Head                Hyperedge
                                                                                                                                  Projection
                                                                                                    𝐏"                              Head
                        ℋ = (𝐗, 𝐇)                                                 Hypergraph                   𝑔# (⋅)                                       𝒁$
                                                                                    Encoder
                                             𝒯!                                                     𝐐"
                                                                                      𝑓# (⋅)                                          𝑔" (⋅)                 𝒀$
                                                         ℋ$ = (𝐗 $ , 𝐇$ )


Figure 1: Overview of our proposed TriCL method. First, two different semantically similar views are generated by aug-
mentations T1 and T2 from the original hypergraph. From these, we use a shared hypergraph encoder fθ (·) to form node and
hyperedge representations. After passing node and hyperedge representations to their respective projection heads (i.e., gφ (·)
and gψ (·)), we maximize the agreement between two views via our proposed tri-directional contrast, which is a combination of
node-, group-, and membership-level contrast.

and hyperedge representations at the k-th layer, respectively.                             ships. Figure 1 visually summarizes TriCL’s architecture.
Formally, the k-th layer of a hypergraph neural network is                                 TriCL is composed of the following four major components:
                                                                                           (1) Hypergraph augmentation. We consider a hyper-
                                                   
        (k)      (k)     (k−1)  (k−1)
       qj = fV →E qj           , pi     : vi ∈ e j ,
                                                                                           graph H = (X, H). TriCL first generates two alternate
                                                           (1)
                                                                                           views of the hypergraph H: H1 = (X1 , H1 ) and H2 =
                                                
        (k)      (k)     (k−1)  (k)
       pi = fE→V pi            , q j : vi ∈ e j ,
                                                                                           (X2 , H2 ), by applying stochastic hypergraph augmentation
            (0)                                                                            function T1 and T2 , respectively. We use a combination of
where pi = xi . The choice of aggregation rules, fV →E (·)                                 random node feature masking (You et al. 2020; Zhu et al.
and fE→V (·), is critical, and a number of models have been                                2020) and membership masking to augment a hypergraph in
proposed. In HGNN (Feng et al. 2019), for example, they                                    terms of attributes and structure. Following previous stud-
choose fV →E and fE→V to be the weighted sum over inputs                                   ies (You et al. 2020; Thakoor et al. 2022), node feature
with normalization as:
                                                                                           masking is not applied to each node independently, and in-
 (k)
            X p(k−1) (k)     
                               1    X wj qj(k) Θ(k)      
                                                                                           stead, we generate a single random binary mask of size
qj     =        i
                √    , pi = σ √                     +b(k) ,                                F where each entry is sampled from a Bernoulli distri-
           v ∈e
                  di           di e :v ∈e   δj
           i      j                                  j    i   j                            bution B(1 − pf ), and we use it to mask features of all
                                                                             (2)           nodes.Similarly, we use a binary mask of size K = nnz(H)
where Θ(k) is a learnable weight matrix, b(k) is a bias, and                               where each element is sampled from a Bernoulli distribution
σ denotes a non-linear activation function. Many other hy-                                 B(1 − pm ) to mask node-hyperedge memberships. The de-
pergraph neural networks can be represented by (1).                                        gree of augmentation is controlled by pf and pm , and we can
                                                                                           adopt different hyperparameters for each augmented view.
3.2    Problem Setting: Hypergraph-based                                                   More details on hypergraph augmentation are provided in
       Contrastive Learning                                                                Appendix D.
Our objective is to train a hypergraph encoder, fθ :
                                    0           00                                         (2) Hypergraph encoder. A hypergraph encoder fθ (·)
R|V |×F × R|V |×|E| → R|V |×F × R|E|×F , such that                                         produces node and hyperedge representations, P and Q,
fθ (X, H) = (P, Q) produces low-dimensional representa-                                    respectively, for two augmented views: (P1 , Q1 ) :=
tions of nodes and hyperedges in a fully unsupervised man-                                 fθ (X1 , H1 ) and (P2 , Q2 ) := fθ (X2 , H2 ). TriCL does not
ner, specifically a contrastive manner. These representations                              constrain the choice of hypergraph encoder architectures if
may then be utilized for downstream tasks, such as node                                    they can be formulated by (1). In our proposed method, we
classification and clustering.                                                             use the element-wise mean pooling layer as a special in-
                                                                                           stance of (1) (see Appendix E.2 for comparison with an al-
3.3    TriCL: Tri-directional Contrastive Learning                                         ternative). That is, fV →E and fE→V are as:
                                                                                                                          X            (k−1)         (k)               
Basically, TriCL follows the conventional multi-view graph                                                (k)                         pi            ΘE            (k)
                                                                                                     qj         =σ                                          + bE            ,
contrastive learning paradigm, where a model aims to max-                                                                    vi ∈ej
                                                                                                                                               δj
imize the agreement of representations between different                                                                                            (k)     (k)
                                                                                                                                                                                        (3)
                                                                                                                                           wj qj ΘV
                                                                                                                                                                               
views (You et al. 2020; Hassani and Khasahmadi 2020;                                                  (k)
                                                                                                     pi = σ
                                                                                                                               X                                     (k)
                                                                                                                                                                  + bV              ,
Zhu et al. 2020). While most existing approaches only use                                                                    ej :vi ∈ej
                                                                                                                                                di
node-level contrast, TriCL applies three forms of contrast
                                                                                                    (k)              (k)                                                        (k)     (k)
for each of the three essential elements constituting hyper-                               where ΘE and ΘV are trainable weights and bE and bV
graphs: nodes, hyperedges, and node-hyperedge member-                                      are trainable biases. We use wj = 1 for simplicity, and (3)
can be represented as (4) in matrix form.                            sample, and the other representations from the second view,
                                          (k)     (k)               z2,k , where k 6= i, are regarded as negative samples. Let
          Q(k) = σ D−1 T (k−1)
                    E H P      ΘE + bE ,                             s(·, ·) denote the score function (a.k.a. critic function) that
                                                              (4)
          P(k) = σ D−1    (k) (k)   (k)                             assigns high values to the positive pair, and low values to
                    V HWQ    ΘV + bV ,
                                                                     negative pairs (Tschannen et al. 2019). We use the cosine
where P(0) = X and W is the identity matrix.                         similarity as the score (i.e. s(u, v) = uT v/kukkvk). Then
                                                                     the loss function for each positive node pair is defined as:
(3) Projection head. Chen et al. (2020) empirically
demonstrate that including a non-linear transformation                                                   es(z1,i ,z2,i )/τn
                                                                             `n (z1,i , z2,i ) = − log P|V |                    ,
called projection head which maps representations to an-                                                      s(z1,i ,z2,k )/τn
                                                                                                        k=1 e
other latent space where contrastive loss is applied helps to
improve the quality of representations. We also adopt two            where τn is a temperature parameter. In practice, we sym-
projection heads denoted by gφ (·) and gψ (·) for projecting         metrize this loss by setting the node representation of the
node and hyperedge representations, respectively. Both pro-          second view as the anchor. The objective function for node-
jection heads in our method are implemented with a two-              level contrast is the average over all positive pairs as:
layer MLP and ELU activation (Clevert, Unterthiner, and                               |V |
Hochreiter 2016). Formally, Zk := gφ (Pk ) and Yk :=                              1 X
                                                                         Ln =              `n (z1,i , z2,i ) + `n (z2,i , z1,i ) .            (5)
gψ (Qk ), where k = 1, 2 for two augmented views.                               2|V | i=1
(4) Tri-directional contrastive loss. In TriCL framework,            Group-level contrast. For any hyperedge (i.e., a group of
we employ three contrastive objectives: (a) node-level con-          nodes) ej , its representation from the first view, y1,j , is set
trast aims to discriminate the representations of the same           to the anchor, the representation of it from the other view,
node in the two augmented views from other node rep-                 y2,j , is treated as the positive sample, and the other rep-
resentations, (b) group-level contrast tries to distinguish          resentations from the view where the positive samples lie,
the representations of the same hyperedge in the two aug-            y2,k , where k 6= j, are regarded as negative samples. We
mented views from other hyperedge representations, and (c)           also use the cosine similarity as the critic, and then the loss
membership-level contrast seeks to differentiate a “real”            function for each positive hyperedge pair is defined as:
node-hyperedge membership from a “fake” one across the                                                  es(y1,j ,y2,j )/τg
two augmented views. We utilize the InfoNCE loss (Oord,                     `g (y1,j , y2,j ) = − log P|E|                     ,
                                                                                                             s(y1,j ,y2,k )/τg
Li, and Vinyals 2018), one of the popular contrastive losses,                                          k=1 e
as in (Zhu et al. 2020, 2021b; Qiu et al. 2020).                     where τg is a temperature parameter. The objective function
   In the rest of this subsection, we first provide a motivat-       for group-level contrast is defined as:
ing example for the tri-directional contrastive loss. Then, we
                                                                                      |E|
describe each of its three components in detail.                                1 X
                                                                         Lg =          `g (y1,j , y2,j ) + `g (y2,j , y1,j ) .                (6)
Motivating example. How can the three forms of contrast                       2|E| j=1
be helpful for node representation learning? In node clas-
sification tasks, for example, information about a group of          Membership-level contrast. For any node vi and hyper-
nodes could help improve performance. Specifically, in co-           edge ej that form membership (i.e., vi ∈ ej ) in the origi-
                                                                     nal hypergraph, the node representation from the first view,
authorship networks such as Cora-A and DBLP, nodes and               z1,i , is set to the anchor, the hyperedge representation from
hyperedges represent papers and authors, respectively, and           the other view, y2,j , is treated as the positive sample. The
papers written by the same author are more likely to belong          negative samples are drawn from the representations of the
to the same field and cover similar topics (i.e. homophily           other hyperedges that are not associated with node vi , de-
exists in hypergraphs (Veldt, Benson, and Kleinberg 2021)).          noted by y2,k , where k : i ∈  / k. Symmetrically, y2,j can also
Thus, high-quality author information could be useful in in-         be the anchor, in which case the negative samples are z1,k ,
ferring the field of the papers he wrote, especially when in-        where k : k ∈     / j. To differentiate a “real” node-hyperedge
formation about the paper is insufficient.                           membership from a “fake” one, we employ a discrimina-
                                                                                     0       00
   Furthermore, leveraging node-hyperedge membership                 tor, D : RF × RF → R as the scoring function so that
helps enrich the information of each node and hyperedge.             D(z, y) represents the probability scores assigned to this
For example, the fact that a meteorology paper is written by         node-hyperedge representation pair (should be higher for
an author who studies mainly machine learning is a useful            “real” pairs) (Hjelm et al. 2019; Veličković et al. 2018b).
clue to suspect that (a) the paper is about application of ma-       For simplicity, we omit the augmented view number in the
chine learning techniques to meteorological problems and             equation. Then we use the following objective:
(b) the author is interested not only in machine learning but                                                eD(zi ,yj )/τm
                                                                       `m (zi , yj ) = − log
also in meteorology. In order to utilize such benefits explic-                                 eD(zi ,yj )/τm + k:i /
                                                                                                                P
                                                                                                                        ∈k e
                                                                                                                             D(zi ,yk )/τm
itly, we proposed the tri-directional contrastive loss, which                            |                      {z                        }
is described below.                                                                                     when zi is the anchor

                                                                                                             eD(zi ,yj )/τm
Node-level contrast. For any node vi , its representation                             − log     D(zi ,yj )/τm
                                                                                                                P            D(zk ,yj )/τm
                                                                                                                                           ,
from the first view, z1,i , is set to the anchor, the representa-                              e              + k:k /   ∈j e
                                                                                         |                      {z                        }
tion of it from the second view, z2,i , is treated as the positive                                      when yj is the anchor
where τm is a temperature parameter. From a practical point                For the clustering task, we assess the quality of repre-
of view, considering a large number of negatives poses                  sentations using the k-means clustering by operating it on
a prohibitive cost, especially for large graphs (Zhu et al.             the frozen node representations produced by each model.
2020; Thakoor et al. 2022). We, therefore, decide to ran-               We employ the local Lloyd algorithm (Lloyd 1982) with
domly select a single negative sample per positive sample               the k-means++ seeding (Arthur and Vassilvitskii 2006) ap-
for `m (zi , yj ). Since two views are symmetric, we get two            proach. For a fair comparison, we train each model with 5
node-hyperedge pairs for a single membership. The objec-
tive function for membership-level contrast is defined as:              random weight initializations, perform k-means 5 times on
                                                                        each trained encoder, and report the averaged results.
             |V | |E|
         1 XX               
                                                                        Baselines. We compare TriCL with various representative
 Lm =              1[hij =1] `m (z1,i , y2,j ) + `m (z2,i , y1,j ) .
        2K i=1 j=1                                                      baseline approaches including 10 (semi-)supervised models
                                                                  (7)   and 4 unsupervised models. A detailed description of these
  Finally, by integrating Eq. (5), (6), and (7), our proposed           baselines is provided in Appendix B. Note that, since the
contrastive loss is formulated as:                                      methods working on graphs can not be directly applied to
                                                                        hypergraphs, we use them after transforming hypergraphs
                  L = Ln + ωg Lg + ωm Lm ,                       (8)    to graphs via clique expansion. For all baselines, we report
where ωg and ωm are the weights of Lg and Lm , respectively.            their performance based on their official implementations.
   To sum up, TriCL jointly optimizes three contrastive ob-             Implementation details. We employ a one-layer
jectives (i.e., node-, group-, and membership-level contrast),          mean pooling hypergraph encoder described in (4) and
which enable the learned embeddings of nodes and hyper-                 PReLU (He et al. 2015) activation for non-linearlity. Fol-
edges to preserve both the node- and group-level structural             lowing Tschannen et al. (2019), which has experimentally
information at the same time.                                           shown a bilinear critic yields better downstream perfor-
                                                                        mance than higher-capacity MLP critics, we use a bilinear
                        4   Experiments                                 function as a discriminator to score node-hyperedge repre-
In this section, we empirically evaluate the quality of node            sentation pairs, formulated as D(z, y) = σ(z T Sy). Here,
representations learnt by TriCL on two hypergraph learning              S denotes a trainable scoring matrix and σ is the sigmoid
tasks: node classification and clustering, which have been              function to transform scores into probabilities of (z, y)
commonly used to benchmark hypergraph learning algo-                    being a positive sample. A description of the optimizer and
rithms (Zhou, Huang, and Schölkopf 2006).                              model hyperparameters are provided in Appendix C.
4.1   Dataset                                                           4.3   Performance on Node Classification
We assess the performance of TriCL on 10 commonly used                  Table 1 summarizes the empirical performance of all meth-
benchmark datasets; these datasets are categorized into (1)             ods. Overall, our proposed method achieves the strongest
co-citation datasets (Cora, Citeseer, and Pubmed) (Sen et al.           performance across all datasets. In most cases, TriCL out-
2008), (2) co-authorship datasets (Cora and DBLP (Rossi                 performs its unsupervised baselines by significant margins,
and Ahmed 2015)), (3) computer vision and graphics                      and also outperforms the models trained with label supervi-
datasets (NTU2012 (Chen et al. 2003) and ModelNet40 (Wu                 sion. Below, we make three notable observations.
et al. 2015)), and (4) datasets from the UCI Categorical                   First, applying graph contrastive learning methods, such
Machine Learning Repository (Dua and Graff 2017) (Zoo,                  as Node2vec, DGI, and GRACE, to hypergraph datasets is
20Newsgroups, and Mushroom). Further descriptions and                   less effective. They show significantly lower accuracy com-
the statistics of datasets are provided in Appendix A.                  pared to TriCL. This is because converting hypergraphs to
                                                                        graphs via clique expansion involves a loss of structural in-
4.2   Experimental Setup                                                formation (Dong, Sawin, and Bengio 2020). Especially, the
Evaluation protocol. For the node classification task, we               Zoo dataset has large maximum and average hyperedge sizes
follow the standard linear-evaluation protocol as introduced            (see Appendix A). When clique expansion is performed, a
in Veličković et al. (2018b). The encoder is firstly trained in       nearly complete graph, where most of the nodes are pair-
a fully unsupervised manner and computes node represen-                 wisely connected to each other, is obtained, and thus most
tations; then, a simple linear classifier is trained on top of          of the structural information is lost, resulting in significant
these frozen representations through a `2 -regularized logis-           performance degradation.
tic regression loss, without flowing any gradients back to the             Second, rather than just using node-level contrast, con-
encoder. For all the datasets, we randomly split them, where            sidering the different types of contrast (i.e., group- and
10%, 10%, and 80% of nodes are chosen for the training,                 membership-level contrast) together can help improve per-
validation, and test set, respectively, as has been followed            formance. We propose and evaluate two model variants, de-
in Zhu et al. (2020); Thakoor et al. (2022). We evaluate the            noted as TriCL-N and TriCL-NG, which use only node-level
model with 20 dataset splits over 5 random weight initial-              contrast and node- and group-level contrast, respectively, to
izations for unsupervised setting, and report the averaged              validate the effect of each type of contrast. From Table 1,
accuracy on each dataset. In a supervised setting, we use               we note that the more types of contrast we use, the better
20 dataset splits and a different model initialization for each         the performance tends to be. To be more specific, we ana-
split and report the averaged accuracy.                                 lyze the effectiveness of each type of contrast (i.e., Ln , Lg ,
Table 1: Node classification accuracy and standard deviations. Graph methods, marked as ?, are applied after converting hyper-
graphs to graphs via clique expansion. For each dataset, the best and the second-best performances are highlighted in boldface
and underlined, respectively. A.R. denotes average rank, OOT denotes cases where results are not obtained within 24 hours, and
OOM indicates out of memory on a 24GB GPU. In most cases, TriCL outperforms all others, including the supervised ones.

                   Method           Cora-C         Citeseer      Pubmed        Cora-A         DBLP             Zoo         20News       Mushroom      NTU2012      ModelNet40     A.R.↓
                   MLP            60.32 ± 1.5    62.06 ± 2.3   76.27 ± 1.1   64.05 ± 1.4   81.18 ± 0.2    75.62 ± 9.5    79.19 ± 0.5   99.58 ± 0.3   65.17 ± 2.3    93.75 ± 0.6    12.5
                   GCN ?          77.11 ± 1.8    66.07 ± 2.4   82.63 ± 0.6   73.66 ± 1.3   87.58 ± 0.2     36.79 ± 9.6      OOM        92.47 ± 0.9   71.17 ± 2.4    91.67 ± 0.2    11.7
                   GAT ?          77.75 ± 2.1    67.62 ± 2.5   81.96 ± 0.7   74.52 ± 1.3   88.59 ± 0.1    36.48 ± 10.0      OOM           OOM        70.94 ± 2.6    91.43 ± 0.3     11


    Supervised
                   HGNN           77.50 ± 1.8    66.16 ± 2.3   83.52 ± 0.7   74.38 ± 1.2   88.32 ± 0.3    78.58 ± 11.1   80.15 ± 0.3   98.59 ± 0.5   72.03 ± 2.4    92.23 ± 0.2    8.1
                   HyperConv      76.19 ± 2.1    64.12 ± 2.6   83.42 ± 0.6   73.52 ± 1.0   88.83 ± 0.2    62.53 ± 14.5   79.83 ± 0.4   97.56 ± 0.6   72.62 ± 2.6    91.84 ± 0.1    9.8
                   HNHN           76.21 ± 1.7    67.28 ± 2.2   80.97 ± 0.9   74.88 ± 1.6   86.71 ± 1.2    78.89 ± 10.2   79.51 ± 0.4   99.78 ± 0.1   71.45 ± 3.2    92.96 ± 0.2    8.9
                   HyperGCN       64.11 ± 7.4    59.92 ± 9.6   78.40 ± 9.2   60.65 ± 9.2   76.59 ± 7.6     40.86 ± 2.1   77.31 ± 6.0   48.26 ± 0.3   46.05 ± 3.9    69.23 ± 2.8    15.1
                   HyperSAGE      64.98 ± 5.3    52.43 ± 9.4   79.49 ± 8.7   64.59 ± 4.3   79.63 ± 8.6     40.86 ± 2.1      OOT           OOT           OOT            OOT         14.7
                   UniGCN         77.91 ± 1.9    66.40 ± 1.9   84.08 ± 0.7   77.30 ± 1.4   90.31 ± 0.2    72.10 ± 12.1   80.24 ± 0.4   98.84 ± 0.5   73.27 ± 2.7    94.62 ± 0.2     5.9
                   AllSet         76.21 ± 1.7    67.83 ± 1.8   82.85 ± 0.9   76.94 ± 1.3   90.07 ± 0.3    72.72 ± 11.8   79.90 ± 0.4   99.78 ± 0.1   75.09 ± 2.5    96.85 ± 0.2     6.2
                   Node2vec ?     70.99 ± 1.4    53.85 ± 1.9   78.75 ± 0.9   58.50 ± 2.1   72.09 ± 0.3    17.02 ± 4.1    63.35 ± 1.7   88.16 ± 0.8   67.72 ± 2.1    84.94 ± 0.4    15.6
                   DGI ?          78.17 ± 1.4    68.81 ± 1.8   80.83 ± 0.6   76.94 ± 1.1   88.00 ± 0.2    36.54 ± 9.7       OOM           OOM        72.01 ± 2.5    92.18 ± 0.2     9.3


    Unsupervised
                   GRACE ?        79.11 ± 1.7    68.65 ± 1.7   80.08 ± 0.7   76.59 ± 1.0      OOM          37.07 ± 9.3      OOM           OOM        70.51 ± 2.4    90.68 ± 0.3    10.4
                   S2 -HHGR       78.08 ± 1.7    68.21 ± 1.8   82.13 ± 0.6   78.15 ± 1.1   88.69 ± 0.2    80.06 ± 11.1   79.75 ± 0.3   97.15 ± 0.5   73.95 ± 2.4    93.26 ± 0.2     6.8
                   Random-Init    63.62 ± 3.1    60.44 ± 2.5   67.49 ± 2.2   66.27 ± 2.2   76.57 ± 0.6    78.43 ± 11.0   77.14 ± 0.6   97.40 ± 0.6   74.39 ± 2.6    96.29 ± 0.3    11.9
                   TriCL-N        80.23 ± 1.2    70.28 ± 1.5   83.44 ± 0.6   81.94 ± 1.1   90.88 ± 0.1    79.94 ± 11.1   80.18 ± 0.2   99.76 ± 0.2   75.20 ± 2.6    97.01 ± 0.2     3.4
                   TriCL-NG       81.45 ± 1.2    71.38 ± 1.2   83.68 ± 0.7   82.00 ± 1.0   90.94 ± 0.1    80.19 ± 11.1   80.18 ± 0.2   99.81 ± 0.1   75.25 ± 2.5    97.02 ± 0.1      2
                   TriCL          81.57 ± 1.1    72.02 ± 1.2   84.26 ± 0.6   82.15 ± 0.9   91.12 ± 0.1    80.25 ± 11.2   80.14 ± 0.2   99.83 ± 0.1   75.23 ± 2.4    97.08 ± 0.1    1.5


Table 2: Comparison of node classification accuracy according to whether or not to use each type of contrast (i.e., Ln , Lg , and
Lm ). Using all types of contrasts (i.e., node-, group-, and membership-level contrast) achieves the best performance in most
cases as they are complementarily reinforcing each other.

        Ln          Lg   Lm        Cora-C         Citeseer      Pubmed         Cora-A        DBLP              Zoo        20News       Mushroom      NTU2012       ModelNet40     A.R.↓
            3       -       -    80.23 ± 1.2    70.28 ± 1.5*   83.44 ± 0.6   81.94 ± 1.1   90.88 ± 0.1   79.94 ± 11.1    80.18 ± 0.2   99.76 ± 0.2   75.20 ± 2.6   97.01 ± 0.2     3.8
            -       3       -    79.69 ± 1.6    71.02 ± 1.3*   80.20 ± 1.3   78.98 ± 1.4   88.60 ± 0.2   79.31 ± 10.7    79.35 ± 0.4   99.13 ± 0.3   74.41 ± 2.6   96.66 ± 0.2     5.7
            -       -       3    76.76 ± 1.8     63.98 ± 2.0   79.86 ± 0.9   76.77 ± 1.1   63.95 ± 7.2   79.80 ± 11.0    79.27 ± 0.3   94.87 ± 0.7   73.11 ± 2.8   96.57 ± 0.2     6.9
            3       3       -    81.45 ± 1.2    71.38 ± 1.4    83.68 ± 0.7   82.00 ± 1.0   90.94 ± 0.1   80.19 ± 11.1    80.18 ± 0.2   99.81 ± 0.1   75.25 ± 2.5   97.02 ± 0.1     2.3
            3       -       3    80.49 ± 1.3     70.46 ± 1.5   83.98 ± 0.7   81.62 ± 1.0   90.75 ± 0.1   80.19 ± 11.1    80.15 ± 0.2   99.74 ± 0.2   75.12 ± 2.5   97.03 ± 0.1     3.6
            -       3       3    80.80 ± 1.1     71.73 ± 1.4   82.81 ± 0.7   80.24 ± 1.0   90.17 ± 0.1   80.20 ± 11.1    79.29 ± 0.2   99.82 ± 0.1   73.76 ± 2.5   96.74 ± 0.1     4.1
            3       3       3    81.57 ± 1.1     72.02 ± 1.4   84.26 ± 0.6   82.15 ± 0.9   91.12 ± 0.1   80.25 ± 11.2    80.14 ± 0.2   99.83 ± 0.1   75.23 ± 2.4   97.08 ± 0.1     1.4



and Lm ) on the node classification task in Table 2. We con-                                             Robustness to the number of negatives. To analyze how
duct experiments on all combinations of all types of contrast.                                           the number of negative samples influences the node clas-
The results show that using all types of contrast achieves the                                           sification performance, we propose an approximation of
best performance in most cases as they are complementarily                                               TriCL’s objective called TriCL-Subsampling. Here, instead
reinforcing each other (see Section 3.3 for motivating ex-                                               of constructing the contrastive loss with all negatives, we
amples of how different types of contrast can be helpful for                                             randomly subsample k negatives across the hypergraph for
node representation learning). In most cases, using a com-                                               node- and group-level contrast, respectively, at every gra-
bination of any two types of contrast is more powerful than                                              dient step. Our results in Table 3 show that TriCL is very
using only one. It is noteworthy that while membership-level                                             robust to the number of negatives; even if only two neg-
contrast causes model collapse1 (especially for the Citeseer,                                            ative samples are used for node- and group-level contrast,
DBLP, and Mushroom datasets) when used alone, it boosts                                                  the performance degradation is less than 1%, still outper-
performance when used with node- or group-level contrast.                                                forming the best performing unsupervised baseline method,
   Lastly, in Table 2, we note that group-level contrast is                                              S2 -HHGR, by great margins. Additionally, the results indi-
more crucial than node-level contrast for the Citeseer dataset                                           cate that the random negative sampling is sufficiently effec-
(marked with asterisk), even though the downstream task is                                               tive for TriCL, and there is no need to select hard negatives,
node-level. This result empirically supports our motivations                                             which incur additional computational costs.
mentioned in Section 1.
   To sum up, the superior performance of TriCL demon-                                                   4.4     Performance on Clustering
strates that it produces highly generalized representations.                                             To show how well node representations trained with TriCL
More ablation studies and sensitivity analysis on hyperpa-                                               generalize across various downstream tasks, we evaluate the
rameters used in TriCL are provided in Appendix E.                                                       representations on the clustering task by k-means as de-
                                                                                                         scribed in Section 4.2. We use the node labels as ground
1
    Model collapse (Zhu et al. 2021a) indicates that the model can-                                      truth for the clusters. To evaluate the clusters generated by
    not significantly outperform or even underperform Random-Init.                                       k-means, we measure the agreement between the true labels
    The qualitative analysis of the collapsed models is provided in                                      and the cluster assignments by two metrics: Normalized Mu-
    Appendix F.2.                                                                                        tual Information (NMI) and pairwise F1 score. Table 4 sum-
Table 3: TriCL is very robust to the number of negative samples (i.e., k). A.P.D. stands for average performance degradation.
Even if only two negative samples are used at every gradient step, the performance degrades by less than 1%.

  Method                            Cora-C       Citeseer          Pubmed        Cora-A        DBLP               Zoo          20News       Mushroom       NTU2012      ModelNet40     A.P.D.
  S2 -HHGR all negatives          78.08 ± 1.7   68.21 ± 1.8    82.13 ± 0.6     78.15 ± 1.1   88.69 ± 0.2    80.06 ± 11.1   79.75 ± 0.3      97.15 ± 0.5   73.95 ± 2.4   93.26 ± 0.2      -
  TriCL-Subsampling (k = 2)       80.62 ± 1.3 71.95 ± 1.3 83.22 ± 0.7          81.25 ± 1.0 90.66 ± 0.2      80.10 ± 11.1 80.03 ± 0.2 99.82 ± 0.1 74.95 ± 2.6            97.02 ± 0.1    0.49%
  TriCL-Subsampling (k = 4)       81.15 ± 1.2 72.24 ± 1.2 83.91 ± 0.7          81.85 ± 0.9 90.83 ± 0.1      80.16 ± 11.3 80.08 ± 0.2 99.84 ± 0.1 75.02 ± 2.6            97.05 ± 0.1    0.18%
  TriCL-Subsampling (k = 8)       81.32 ± 1.2 72.04 ± 1.3 83.88 ± 0.7          82.05 ± 0.9 90.93 ± 0.1      80.14 ± 11.2 80.12 ± 0.2 99.84 ± 0.1 75.09 ± 2.5            97.05 ± 0.1    0.14%
  TriCL-Subsampling (k = 16)      81.49 ± 1.1 72.02 ± 1.2 84.23 ± 0.7          82.10 ± 0.9 90.97 ± 0.1      80.10 ± 11.1 80.13 ± 0.2 99.84 ± 0.1 75.16 ± 2.5            97.07 ± 0.1    0.06%
  TriCL all negatives             81.57 ± 1.1 72.02 ± 1.2 84.26 ± 0.6          82.15 ± 0.9 91.12 ± 0.1      80.25 ± 11.2 80.14 ± 0.2 99.83 ± 0.1 75.23 ± 2.4            97.08 ± 0.1       -


Table 4: Evaluation of the embeddings learned by unsupervised methods using k-means clustering. As a naı̈ve baseline method,
expressed by ‘features’, we only use the node features as an input of k-means. All metrics are normalized by multiplying 100.
Larger NMI and F1 indicate better performance, and A.R. denotes average ranking. TriCL ranks first in clustering performance.

                  Cora-C           Citeseer        Pubmed             Cora-A           DBLP                Zoo           20News           Mushroom        NTU2012       ModelNet40
 Method                                                                                                                                                                                 A.R.↓
               NMI↑        F1↑    NMI↑   F1↑    NMI↑        F1↑    NMI↑      F1↑    NMI↑     F1↑    NMI↑         F1↑    NMI↑    F1↑     NMI↑     F1↑      NMI↑   F1↑    NMI↑     F1↑
 features       20.0       28.8   21.5   36.1    19.5       53.4    17.2     29.2   37.0     47.3   78.3         77.3   15.7 41.1         36.6 72.4       81.7   69.0    90.6   86.5     3.8
 Node2vec ?     39.1       44.5   24.5   38.5    23.1       40.1    16.0     34.1   32.4     37.8   11.5         41.6   8.7  26.6         1.6   44.0      78.3   57.7    72.9   53.1     5.0
 DGI ?          54.8       60.1   40.1   51.7    30.4       53.0    45.2     52.5   58.0     57.7   13.0         13.8     OOM                OOM          79.6   61.7    85.0   73.7     3.1
 GRACE ?        44.4       45.6   33.3   45.7    16.7       41.9    37.9     43.3   16.7     41.9    7.3         29.4     OOM                OOM          74.6   47.5    79.4   59.9     4.9
 S2 -HHGR       51.0       56.8   41.1   53.1    27.7       53.2    45.4     52.3   60.3     62.7   90.9         91.1   39.0 58.7         18.6 60.6       82.7   71.2    91.0   90.6     2.1
 TriCL          54.5       60.6   44.1   57.4    30.0       51.7    49.8     56.7   63.1     63.0   91.2         89.3   35.6 54.2          3.8  65.1      83.2   71.5    95.7   94.7     1.6



Table 5: t-SNE plots of the node representations produced by                                        marizes the empirical performance. Our results show that
TriCL and its two variants. The node embeddings of TriCL                                            TriCL achieves strong clustering performance in terms of
exhibits the most distinct clusters with the help of group and                                      all metrics across all datasets (1st place in terms of the aver-
membership contrast, as measured numerically by the Sil-                                            age rank). This is because the node embeddings learned by
houette score (the higher, the better).                                                             TriCL simultaneously preserve local and community struc-
                                                                                                    tural information by fully utilizing group-level contrast.
                       Cora-C                                 Citeseer
                                                                                                    4.5          Qualitative Analysis
                                                                                                    To represent and compare the quality of embeddings intu-
                                                                                                    itively, Table 5 shows t-SNE (Van der Maaten and Hinton
                                                                                                    2008) plots of the node embeddings produced by TriCL
  TriCL-N
                                                                                                    and its two variants (i.e., TriCL-N and TriCL-NG) on the
                                                                                                    Citeseer and Cora Co-citation dataset. As expected from
                                                                                                    the quantitative results, the 2-D projection of embeddings
                                                                                                    learned by TriCL shows visually and numerically (based on
                                                                                                    the Silhouette score (Rousseeuw 1987)) more distinguish-
                                                                                                    able clusters than those obtained by its two variants. In Ap-
                                                                                                    pendix F, we give additional qualitative analysis.


  TriCL-NG
                                                                                                                                      5     Conclusion
                                                                                                    In this paper, we proposed TriCL, a novel hypergraph con-
                                                                                                    trastive representation learning approach. We summarize our
                                                                                                    contributions as follows:
                                                                                                     • We proposed the use of tri-directional contrast, which is
                                                                                                       a combination of node-, group-, and membership-level
                                                                                                       contrast, that consistently and substantially improves the
                                                                                                       quality of the learned embeddings.

  TriCL
                                                                                                     • We achieved state-of-the-art results in node classification
                                                                                                       on hypergraphs by using tri-directional contrast together
                                                                                                       with our data augmentation schemes. Moreover, we veri-
                                                                                                       fied the surprising effectiveness of uniform negative sam-
                                                                                                       pling for our use cases.
                                                                                                     • We demonstrated the superiority of TriCL by conducting
                                                                                                       extensive experiments using 14 baseline approaches, 10
                                                                                                       datasets, and two tasks.
Acknowledgements. This work was supported by Sam-                Grover, A.; and Leskovec, J. 2016. node2vec: Scalable fea-
sung Electronics Co., Ltd. and Institute of Information &        ture learning for networks. In KDD, 855–864.
Communications Technology Planning & Evaluation (IITP)           Hassani, K.; and Khasahmadi, A. H. 2020. Contrastive
grant funded by the Korea government (MSIT) (No. 2022-           multi-view representation learning on graphs. In ICML,
0-00157, Robust, Fair, Extensible Data-Centric Continual         4116–4126.
Learning) (No. 2019- 0-00075, Artificial Intelligence Grad-
uate School Program (KAIST)).                                    He, K.; Fan, H.; Wu, Y.; Xie, S.; and Girshick, R. 2020.
                                                                 Momentum contrast for unsupervised visual representation
                                                                 learning. In CVPR, 9729–9738.
                       References
                                                                 He, K.; Zhang, X.; Ren, S.; and Sun, J. 2015. Delving deep
Arthur, D.; and Vassilvitskii, S. 2006. k-means++: The ad-       into rectifiers: Surpassing human-level performance on ima-
vantages of careful seeding. Technical report, Stanford.         genet classification. In ICCV, 1026–1034.
Arya, D.; Gupta, D. K.; Rudinac, S.; and Worring, M. 2020.
                                                                 Hein, M.; Setzer, S.; Jost, L.; and Rangapuram, S. S. 2013.
Hypersage: Generalizing inductive representation learning
                                                                 The total variation on hypergraphs-learning on hypergraphs
on hypergraphs. arXiv:2010.04558.
                                                                 revisited. In NIPS.
Bai, S.; Zhang, F.; and Torr, P. H. 2021. Hypergraph convo-
                                                                 Hjelm, R. D.; Fedorov, A.; Lavoie-Marchildon, S.; Grewal,
lution and hypergraph attention. Pattern Recognition, 110:
                                                                 K.; Bachman, P.; Trischler, A.; and Bengio, Y. 2019. Learn-
107637.
                                                                 ing deep representations by mutual information estimation
Benson, A. R.; Abebe, R.; Schaub, M. T.; Jadbabaie, A.; and      and maximization. In ICLR.
Kleinberg, J. 2018. Simplicial closure and higher-order link
                                                                 Huang, J.; and Yang, J. 2021. Unignn: a unified framework
prediction. PNAS, 115(48).
                                                                 for graph and hypergraph neural networks. In IJCAI.
Benson, A. R.; Gleich, D. F.; and Leskovec, J. 2016.
Higher-order organization of complex networks. Science,          Jaiswal, A.; Babu, A. R.; Zadeh, M. Z.; Banerjee, D.; and
353(6295): 163–166.                                              Makedon, F. 2020. A survey on contrastive self-supervised
                                                                 learning. Technologies, 9(1): 2.
Chen, D.-Y.; Tian, X.-P.; Shen, Y.-T.; and Ouhyoung, M.
2003. On visual similarity based 3D model retrieval. Com-        Kingma, D. P.; and Ba, J. 2015. Adam: A method for
puter graphics forum, 22(3): 223–232.                            stochastic optimization. In ICLR.
Chen, T.; Kornblith, S.; Norouzi, M.; and Hinton, G. 2020.       Kipf, T. N.; and Welling, M. 2017. Semi-supervised classi-
A simple framework for contrastive learning of visual repre-     fication with graph convolutional networks. In ICLR.
sentations. In ICML, 1597–1607.                                  Lee, G.; Choe, M.; and Shin, K. 2022. HashNWalk: Hash
Chien, E.; Pan, C.; Peng, J.; and Milenkovic, O. 2022. You       and Random Walk Based Anomaly Detection in Hyperedge
are AllSet: A Multiset Function Framework for Hypergraph         Streams. In IJCAI.
Neural Networks. In ICLR.                                        Lee, G.; Ko, J.; and Shin, K. 2020. Hypergraph motifs: con-
Clevert, D.-A.; Unterthiner, T.; and Hochreiter, S. 2016. Fast   cepts, algorithms, and discoveries. In Proceedings of the
and accurate deep network learning by exponential linear         VLDB Endowment.
units (elus). In ICLR.                                           Lee, J.; Lee, Y.; Kim, J.; Kosiorek, A.; Choi, S.; and Teh,
Do, M. T.; Yoon, S.-e.; Hooi, B.; and Shin, K. 2020. Struc-      Y. W. 2019. Set transformer: A framework for attention-
tural patterns and generative models of real-world hyper-        based permutation-invariant neural networks. In ICML,
graphs. In KDD, 176–186.                                         3744–3753.
Dong, Y.; Sawin, W.; and Bengio, Y. 2020. HNHN: Hyper-           Li, B.; Jing, B.; and Tong, H. 2022. Graph Communal Con-
graph networks with hyperedge neurons. In ICML Workshop          trastive Learning. In WWW, 1203–1213.
on Graph Representation Learning and Beyond.                     Li, P.; and Milenkovic, O. 2018. Submodular hypergraphs:
Dua, D.; and Graff, C. 2017. UCI Machine Learning Repos-         p-laplacians, cheeger inequalities and spectral clustering. In
itory.                                                           ICML, 3014–3023.
Feng, Y.; You, H.; Zhang, Z.; Ji, R.; and Gao, Y. 2019. Hy-      Liu, X.; Zhang, F.; Hou, Z.; Mian, L.; Wang, Z.; Zhang, J.;
pergraph neural networks. In AAAI, volume 33, 3558–3565.         and Tang, J. 2021. Self-supervised learning: Generative or
Feng, Y.; Zhang, Z.; Zhao, X.; Ji, R.; and Gao, Y. 2018.         contrastive. TKDE.
Gvcnn: Group-view convolutional neural networks for 3d           Liu, Y.; Jin, M.; Pan, S.; Zhou, C.; Zheng, Y.; Xia, F.; and Yu,
shape recognition. In CVPR, 264–272.                             P. 2022. Graph self-supervised learning: A survey. TKDE.
Fey, M.; and Lenssen, J. E. 2019. Fast graph representation      Lloyd, S. 1982. Least squares quantization in PCM. IEEE
learning with PyTorch Geometric. arXiv:1903.02428.               Transactions on Information Theory, 28(2): 129–137.
Gao, T.; Yao, X.; and Chen, D. 2021. Simcse: Simple con-         Loshchilov, I.; and Hutter, F. 2019. Decoupled weight decay
trastive learning of sentence embeddings. In EMNLP.              regularization. In ICLR.
Glorot, X.; and Bengio, Y. 2010. Understanding the diffi-        Oord, A. v. d.; Li, Y.; and Vinyals, O. 2018. Rep-
culty of training deep feedforward neural networks. In AIS-      resentation learning with contrastive predictive coding.
TATS, 249–256.                                                   arXiv:1807.03748.
Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.;        Yang, D.; Qu, B.; Yang, J.; and Cudre-Mauroux, P. 2019.
Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.;     Revisiting user mobility and social relationships in lbsns: a
et al. 2019. Pytorch: An imperative style, high-performance       hypergraph embedding approach. In WWW, 2147–2157.
deep learning library. In NeurIPS.                                You, Y.; Chen, T.; Sui, Y.; Chen, T.; Wang, Z.; and Shen, Y.
Peng, Z.; Huang, W.; Luo, M.; Zheng, Q.; Rong, Y.; Xu,            2020. Graph contrastive learning with augmentations. In
T.; and Huang, J. 2020. Graph representation learning via         NeurIPS.
graphical mutual information maximization. In WWW, 259–           Yu, J.; Yin, H.; Li, J.; Wang, Q.; Hung, N. Q. V.; and Zhang,
270.                                                              X. 2021. Self-supervised multi-channel hypergraph convo-
Qiu, J.; Chen, Q.; Dong, Y.; Zhang, J.; Yang, H.; Ding, M.;       lutional network for social recommendation. In WWW, 413–
Wang, K.; and Tang, J. 2020. Gcc: Graph contrastive coding        424.
for graph neural network pre-training. In KDD, 1150–1160.         Zaheer, M.; Kottur, S.; Ravanbakhsh, S.; Poczos, B.;
Rong, Y.; Huang, W.; Xu, T.; and Huang, J. 2020. Drope-           Salakhutdinov, R. R.; and Smola, A. J. 2017. Deep sets.
dge: Towards deep graph convolutional networks on node            In NIPS.
classification. In ICLR.                                          Zhang, J.; Gao, M.; Yu, J.; Guo, L.; Li, J.; and Yin, H.
Rossi, R.; and Ahmed, N. 2015. The network data repository        2021. Double-Scale Self-Supervised Hypergraph Learning
with interactive graph analytics and visualization. In AAAI.      for Group Recommendation. In CIKM, 2557–2567.
Rousseeuw, P. J. 1987. Silhouettes: a graphical aid to the        Zhang, J.; Li, F.; Xiao, X.; Xu, T.; Rong, Y.; Huang, J.;
interpretation and validation of cluster analysis. Journal of     and Bian, Y. 2022. Hypergraph Convolutional Networks via
Computational and Applied Mathematics, 20: 53–65.                 Equivalency between Hypergraphs and Undirected Graphs.
                                                                  arXiv:2203.16939.
Sen, P.; Namata, G.; Bilgic, M.; Getoor, L.; Galligher, B.;
                                                                  Zheng, X.; Zhu, W.; Tang, C.; and Wang, M. 2019. Gene se-
and Eliassi-Rad, T. 2008. Collective classification in net-
                                                                  lection for microarray data classification via adaptive hyper-
work data. AI magazine, 29(3): 93–93.
                                                                  graph embedded dictionary learning. Gene, 706: 188–200.
Su, H.; Maji, S.; Kalogerakis, E.; and Learned-Miller, E.         Zhou, D.; Huang, J.; and Schölkopf, B. 2006. Learning with
2015. Multi-view convolutional neural networks for 3d             hypergraphs: Clustering, classification, and embedding. In
shape recognition. In ICCV, 945–953.                              NIPS.
Thakoor, S.; Tallec, C.; Azar, M. G.; Azabou, M.; Dyer,           Zhu, R.; Zhao, B.; Liu, J.; Sun, Z.; and Chen, C. W. 2021a.
E. L.; Munos, R.; Veličković, P.; and Valko, M. 2022. Large-    Improving contrastive learning by visualizing feature trans-
scale representation learning on graphs via bootstrapping. In     formation. In ICCV, 10306–10315.
ICLR.
                                                                  Zhu, Y.; Xu, Y.; Yu, F.; Liu, Q.; Wu, S.; and Wang, L. 2020.
Tschannen, M.; Djolonga, J.; Rubenstein, P. K.; Gelly, S.;        Deep graph contrastive representation learning. In ICML
and Lucic, M. 2019. On Mutual Information Maximization            Workshop on Graph Representation Learning and Beyond.
for Representation Learning. In ICLR.
                                                                  Zhu, Y.; Xu, Y.; Yu, F.; Liu, Q.; Wu, S.; and Wang, L. 2021b.
Van der Maaten, L.; and Hinton, G. 2008. Visualizing data         Graph contrastive learning with adaptive augmentation. In
using t-SNE. Journal of Machine Learning Research, 9(11).         WWW, 2069–2080.
Veldt, N.; Benson, A. R.; and Kleinberg, J. 2021.
Higher-order homophily is combinatorially impossible.
arXiv:2103.11818.
Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Liò,
P.; and Bengio, Y. 2018a. Graph Attention Networks. In
ICLR.
Veličković, P.; Fedus, W.; Hamilton, W. L.; Liò, P.; Bengio,
Y.; and Hjelm, R. D. 2018b. Deep Graph Infomax. In ICLR.
Wang, F.; and Liu, H. 2021. Understanding the behaviour of
contrastive loss. In CVPR, 2495–2504.
Wu, Z.; Song, S.; Khosla, A.; Yu, F.; Zhang, L.; Tang, X.;
and Xiao, J. 2015. 3d shapenets: A deep representation for
volumetric shapes. In CVPR, 1912–1920.
Xia, X.; Yin, H.; Yu, J.; Wang, Q.; Cui, L.; and Zhang, X.
2021. Self-supervised hypergraph convolutional networks
for session-based recommendation. In AAAI, volume 35,
4503–4511.
Yadati, N.; Nimishakavi, M.; Yadav, P.; Nitin, V.; Louis, A.;
and Talukdar, P. 2019. Hypergcn: A new method for training
graph convolutional networks on hypergraphs. In NeurIPS.
                    A     Dataset Details                              Torr 2021), HNHN (Dong, Sawin, and Bengio 2020), Hy-
We use 10 benchmark datasets from the existing hypergraph              perGCN (Yadati et al. 2019), HyperSAGE (Arya et al.
neural networks literature; these datasets are categorized             2020), UniGCN (Huang and Yang 2021), and AllSetTrans-
into (1) co-citation datasets (Cora, Citeseer, and Pubmed)             former (Chien et al. 2022) applied directly to hypergraphs),
2
   (Sen et al. 2008), (2) co-authorship datasets (Cora 3 and           and (2) unsupervised learning methods (Node2vec (Grover
DBLP 4 (Rossi and Ahmed 2015)), (3) computer vision and                and Leskovec 2016), DGI (Veličković et al. 2018b), and
graphics datasets (NTU2012 (Chen et al. 2003) and Model-               GRACE (Zhu et al. 2020), which are representative graph
Net40 (Wu et al. 2015)), and (4) datasets from the UCI Cate-           contrastive learning methods and S2 -HHGR (Zhang et al.
gorical Machine Learning Repository (Dua and Graff 2017)               2021), which is a hypergraphs contrastive learning method).
(Zoo, 20Newsgroups, and Mushroom). Some basic statistics               To measure the quality of the inductive biases inherent in the
of the datasets are provided in Table 6.                               encoder model, we also consider Random-Init (Veličković
    The co-citation datasets are composed of a set of pa-              et al. 2018b; Thakoor et al. 2022), an encoder with the same
pers and their citation links. To represent a co-citation re-          architecture as TriCL but with randomly initialized param-
lationship as a hypergraph, papers become nodes and ci-                eters, as a baseline. Since the methods working on graphs
tation links become hyperedges. To be specific, the nodes              can not be directly applied to hypergraphs, we use them af-
v1 , . . . , vk compose a hyperedge e when the papers corre-           ter transforming hypergraphs to graphs via clique expansion.
sponding to v1 , . . . , vk are referred by the document e. The        In the case of S2 -HHGR, it is originally designed for group
co-authorship datasets are composed of a set of papers with            recommendations with supervisory signals, and therefore it
their authors. In hypergraphs that model the co-authorship             is not directly applicable to node classification tasks. Thus
datasets, nodes and hyperedges represent papers and au-                we slightly modified the algorithm so that it uses only its
thors, respectively. Precisely, the nodes v1 , . . . , vk compose      self-supervised loss. For all the baseline approaches, we re-
a hyperedge e when the papers corresponding to v1 , . . . , vk         port their performance using their official implementations.
are written by the author e. Features of each node are rep-
resented by bag-of-words features from its abstract. Nodes                         C     Implementation Details
are labeled with their categories. The hypergraphs prepro-             C.1    Infrastructures and Implementations
cessed from all the co-citation and co-authorship datasets are
publicly available with the official implementation of Hyper-          All experiments are performed on a server with NVIDIA
GCN 5 (Yadati et al. 2019).                                            RTX 3090 Ti GPUs (24GB memory), 256GB of RAM, and
    For visual datasets, the hypergraph construction follows           two Intel Xeon Silver 4210R Processors. Our models are im-
the setting described in Feng et al. (2019), and the node              plemented using PyTorch 1.11.0 (Paszke et al. 2019) and
features are extracted by Group-View Convolutional Neural              PyTorch Geometric 2.0.4 (Fey and Lenssen 2019).
Network (GVCNN) (Feng et al. 2018) and Multi-View Con-
volutional Neural Network (MVCNN) (Su et al. 2015).                    C.2    Hyperparameters
    In the 20Newsgroups dataset, the TF-IDF representations            As described in Section 4.2, we use a one-layer mean pool-
of news messages are used as the node features. In the Mush-           ing hypergraph encoder as in Eq. (4) and PReLU (He et al.
room dataset, the node features indicate categorical descrip-          2015) activation in all the experiments. Note that, to each
tions of 23 species of mushrooms. In the Zoo dataset, the              node, we add a self-loop which is a hyperedge which con-
node features are a mix of categorical and numerical mea-              tains exactly one node, before the hypergraph is fed into the
surements describing different animals.                                encoder. In Appendix E.1, we show that adding self-loops
    We remove nodes that are not included in any hyperedge             helps to improve the quality of representations. When con-
(i.e. isolated nodes) from the hypergraphs, because such               structing the proposed tri-directional contrastive loss, self-
nodes cause trivial structures in hypergraphs and their pre-           loops and empty-hyperedges (i.e., hyperedges with degree
dictions would only depend on the features of that node. For           zero) are ignored.
all the datasets, we randomly select 10%, 10%, and 80% of                 In all our experiments, all models are initialized with Glo-
nodes disjointly for the training, validation, and test sets, re-      rot initialization (Glorot and Bengio 2010) and trained using
spectively. The datasets and train-valid-test splits used in our       the AdamW optimizer (Kingma and Ba 2015; Loshchilov
experiments are provided as supplementary materials.                   and Hutter 2019) with weight decay set to 10−5 . We train
                                                                       the model for a fixed number of epochs at which the perfor-
                   B     Baseline Details                              mance of node classification sufficiently converges.
We compare our proposed method with various representa-                   The augmentation hyperparameters pf and pm , which
tive baseline approaches that can be categorized into (1) su-          control the sampling process for node feature and mem-
pervised learning methods (GCN (Kipf and Welling 2017)                 bership masking, respectively, are chosen between 0.0 and
and GAT (Veličković et al. 2018a) applied to graphs and              0.4 so that the original hypergraph is not overly corrupted.
HGNN (Feng et al. 2019), HyperConv (Bai, Zhang, and                    Some prior works (Zhu et al. 2020, 2021b) have demon-
                                                                       strated that using a different degree of augmentation for
2
  https://linqs.soe.ucsc.edu/data                                      each view shows better results, and we can also adopt dif-
3
  https://people.cs.umass.edu/ mccallum/data.html                      ferent hyperparameters for each augmented view (as men-
4
  https://aminer.org/lab-datasets/citation/DBLP-citation-Jan8.tar.bz   tioned in Section 3.3). However, our contributions are or-
5
  https://github.com/malllabiisc/HyperGCN                              thogonal to this problem, thus we choose the same hyperpa-
                                           Table 6: Statistics of datasets used in our experiments.

                                Cora-C     Citeseer      Pubmed        Cora-A   DBLP       Zoo     20News        Mushroom       NTU2012       ModelNet40
         # Nodes                 1,434      1,458        3,840          2,388   41,302      101       16,242          8,124         2,012          12,311
         # Hyperedges            1,579      1,079        7,963          1,072   22,363      43          100            298          2,012          12,311
         # Memberships           4,786      3,453        34,629         4,585   99,561     1,717      65,451         40,620         10,060         61,555
         Avg. hyperedge size     3.03       3.20          4.35          4.28     4.45      39.93      654.51         136.31           5              5
         Avg. node degree        3.34       2.37          9.02          1.92     2.41      17.00       4.03           5.00            5              5
         Max. hyperedge size       5         26           171            43       202       93         2241           1808            5              5
         Max. node degree         145        88            99            23        18       17          44              5             19             30
         # Features              1,433      3,703         500           1,433    1,425       16         100             22           100            100
         # Classes                 7          6             3             7         6         7          4               2            67             40


                                             Table 7: Hyperparameter settings on each dataset.

                                                                                Learning    Training      Node           Hyperedge       Projection
               Dataset         pf    pm    τn       τg    τm      ωg      ωm
                                                                                  Rate      Epochs       Emb. Size       Emb. Size      Hidden Size
               Cora-C          0.4   0.4   0.5   0.5      1.0      22      1      5e-4         300             512            512            512
               Citeseer        0.4   0.4   1.0   1.0      0.8      22      21     5e-5         500             512            512            512
               Pubmed          0.1   0.4   0.3   0.2      0.6      22      21     5e-4        1,000            512            512            512
               Cora-A          0.3   0.2   0.6   0.5      0.6     2−1     2−1     1e-4         800             512            512            512
               DBLP            0.2   0.2   0.8   0.2      1.0     2−4     2−2     5e-3         500             256            256            256
               Zoo             0.4   0.2   0.9   0.9      1.0      21      21     1e-3         100             128            128            128
               20News          0.1   0.4   0.7   0.1      1.0     2−4     2−4     1e-3         500             256            256            256
               Mushroom        0.0   0.4   1.0   0.9      0.1      22      1      1e-3         500             512            512            512
               NTU2012         0.0   0.4   1.0   0.7      0.5     2−1     2−4     1e-3         200             512            512            512
               ModelNet40      0.0   0.4   0.9   0.3      0.9     2−2     2−3     1e-3         200             256            256            256



rameters for two augmented views (i.e., pf,1 = pf,2 = pf                              size |V | where each element is sampled from a Bernoulli
and pm,1 = pm,2 = pm ) for simplicity. In Appendix D, we                              distribution B(1 − pn ) to mask nodes.
demonstrate that using node feature masking and member-                             • Hyperedge masking: randomly mask a portion of hyper-
ship masking together is a reasonable choice.                                         edges in the original hypergraph. Precisely, we use a bi-
   The three temperature hyperparameters τn , τg , and τm ,                           nary mask of size |E| where each element is sampled from
which control the uniformity of the embedding distribu-                               a Bernoulli distribution B(1 − pe ) to mask hyperedges.
tion (Wang and Liu 2021), are selected from 0.1 to 1.0,
respectively. The weights ωg and ωm are chosen from                                 • Membership masking: randomly mask a portion of node-
[2−4 , 2−3 , . . . , 24 ], respectively. The size of node embed-                      hyperedge memberships in the original hypergraph. In par-
dings, hyperedge embeddings, and a hidden layer of pro-                               ticular, we use a binary mask of size K = nnz(H)
jection heads are set to the same values for simplicity. In                           where each element is sampled from a Bernoulli distribu-
Table 7, we provide hyperparameters we found through a                                tion B(1 − pm ) to mask node-hyperedge memberships.
small grid search based on the validation accuracy, as many                         • Node feature masking: randomly mask a portion of di-
self-supervised learning methods do (Chen et al. 2020; Zhu                            mensions with zeros in node features. Specifically, we gen-
et al. 2020, 2021b; Thakoor et al. 2022).                                             erate a single random binary mask of size F where each
                                                                                      entry is sampled from a Bernoulli distribution B(1 − pf ),
          D    Hypergraph Augmentations                                               and use it to mask features of all nodes in the hypergraph.
 Generating augmented views is crucial for contrastive learn-                        The degree of augmentation can be controlled by pn , pe ,
 ing methods. Different views provide different contexts or                          pm , and pf . These masking methods corrupt the hypergraph
 semantics for datasets. While creating semantically mean-                           structure, except for node feature masking, which impairs
 ingful augmentations is critical for contrastive learning, in                       the hypergraph attributes.
 the hypergraph domain, it is an underexplored problem than                             To show which types of augmentation are advantageous,
 in other domains such as vision. In the graph domain, simple                        we first examine the node classification performance for dif-
 and effective graph augmentation methods have been pro-                             ferent augmentation pairs with a masking rate of 0.2. We
 posed, and these are commonly used in graph contrastive                             summarize the results in Figure 2. Note that, when using
 learning (You et al. 2020; Zhu et al. 2020). Borrowing these                        only one augmentation for each view, the effect of node fea-
 approaches, in this section, we analyze four types of aug-                          ture masking is consistently good, but in particular, hyper-
 mentation (i.e., node masking, hyperedge masking, mem-                              edge masking performs poorly. Next, using the structural
 bership masking, and node feature masking), which are nat-                          and attribute augmentations together always yields better
 urally applicable to hypergraphs, along with TriCL.                                 performance than using just one. Among them, the pair of
• Node masking: randomly mask a portion of nodes in the                              membership masking and node feature masking shows the
   original hypergraph. Formally, we use a binary mask of                            best performance, demonstrating that using it in TriCL is a
                         Cora Co-authorship              High                                                              Cora Co-authorship                      High
       Mem + Feat 81.3 81.3 81.3 81.7 81.8 81.8 81.8                                                       0.9 79.0 80.9 81.5 81.7 81.6 81.3 81.1 80.6 80.3 79.9




                                                                            Membership Masking Rate (pm)
       Node + Feat 81.3 81.3 81.3 81.7 81.8 81.8 81.8                                                      0.8 79.0 81.2 81.7 82.0 82.0 81.8 81.4 81.0 80.7 80.3
         HE + Feat 81.1 81.3 81.3 81.5 81.7 81.8 81.8                                                      0.7 79.0 81.2 81.8 82.1 82.2 81.9 81.5 81.2 80.8 80.6
                                                                                                           0.6 79.0 81.3 81.9 82.1 82.2 82.0 81.6 81.2 80.9 80.7
         Feat Mask 81.0 81.2 81.2 81.3 81.5 81.7 81.7                                                      0.5 79.1 81.3 81.9 82.2 82.3 82.0 81.6 81.2 80.9 80.7
        Mem Mask 78.9 79.2 79.1 81.1 81.3 81.3 81.3                                                        0.4 79.2 81.3 81.9 82.2 82.3 82.0 81.7 81.3 80.9 80.6
        Node Mask 78.9 79.2 79.2 81.2 81.3 81.3 81.3                                                       0.3 79.2 81.4 81.9 82.2 82.2 82.0 81.6 81.3 80.8 80.4
          HE. Mask 77.5 78.9 78.8 81.0 81.2 81.4 81.3                                                      0.2 79.1 81.3 81.9 82.2 82.2 82.0 81.6 81.2 80.7 80.1
                                                         Low                                               0.1 78.6 81.1 81.7 82.0 82.1 81.9 81.6 81.1 80.4 79.5
                                             t     t t
                        ask ask ask ask ea ea ea                                                           0.0 75.8 80.6 81.3 81.7 81.8 81.7 81.4 80.8 79.9 79.0
                   E . M de M m M at M E + F e + F + F                                                         0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
                                                                                                                                                                   Low
                  H No Me Fe H od em                                                                                   Node Feature Masking Rate (pf)
                                           N M
                              Citeseer                   High                                                                    Citeseer                          High
       Mem + Feat 71.2 71.3 71.2 71.4 71.6 71.6 71.6                                                    0.9 70.5 71.0 71.4 71.6 71.6 71.5 71.4 71.0 70.4 68.8




                                                                         Membership Masking Rate (pm)
       Node + Feat 71.3 71.2 71.2 71.5 71.6 71.6 71.6                                                   0.8 70.4 71.0 71.4 71.7 71.8 71.8 71.7 71.5 70.9 69.5
         HE + Feat 71.1 71.3 71.2 71.3 71.4 71.6 71.5                                                   0.7 70.4 71.0 71.5 71.7 71.8 71.8 71.8 71.6 71.3 69.9
                                                                                                        0.6 70.5 71.2 71.6 71.8 71.9 72.0 71.8 71.7 71.3 70.1
         Feat Mask 70.9 71.0 70.9 71.0 71.3 71.5 71.4                                                   0.5 70.5 71.1 71.6 71.9 72.0 71.9 71.9 71.8 71.4 70.2
        Mem Mask 70.9 70.6 70.6 71.0 71.2 71.2 71.2                                                     0.4 70.5 71.2 71.7 71.9 72.0 72.0 72.0 71.7 71.4 70.3
        Node Mask 70.9 70.7 70.7 71.0 71.2 71.3 71.2                                                    0.3 70.7 71.2 71.7 71.8 72.0 72.0 71.9 71.7 71.3 70.3
          HE. Mask 70.3 70.9 70.9 70.8 71.0 71.2 71.2                                                   0.2 70.8 71.1 71.6 71.8 71.9 71.9 71.8 71.6 71.3 70.3
                                                         Low                                            0.1 70.8 71.1 71.4 71.7 71.8 71.8 71.7 71.5 71.2 70.2
                                             t     t t
                        ask ask ask ask ea ea ea                                                        0.0 69.8 70.6 71.0 71.3 71.5 71.6 71.6 71.5 71.1 70.1
                   E . M de M m M at M E + F e + F + F                                                      0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
                                                                                                                                                                   Low
                  H No Me Fe H od em                                                                                Node Feature Masking Rate (pf)
                                           N M
Figure 2: Node classification accuracy (%) when employ-            Figure 3: Node classification accuracy (%) according to the
ing different augmentation pairs. Using the structural and         masking rates of membership and node features. A moderate
attribute augmentations together always yields better perfor-      extent of augmentation (i.e., masking rate between 0.3 and
mance than using just one.                                         0.7) benefits the downstream performance most.
                                                                   Table 8: Node classification accuracy (%) of TriCL when the
reasonable choice. The combination of node masking and             hypergraphs, which the encoder receives as input, are with
node feature masking is also a good choice.                        and without self-loops. Adding self-loops helps to improve
   Figure 3 shows the node classification accuracy accord-         the quality of the node representations.
ing to the membership and the node feature masking rate. It
demonstrates that a moderate extent of augmentation (i.e.,                                                                                      TriCL
masking rate between 0.3 and 0.7) benefits the downstream           Dataset
                                                                                                                    Without Self-loops          With Self-loops (Proposed)
performance most. If the masking rate is too small, two sim-
ilar views are generated, which are insufficient to learn the       Cora-C                                              80.95 ± 1.29                     81.57 ± 1.12
discriminant ability of the encoder, and if it is too large, the    Citeseer                                            71.10 ± 1.09                     72.02 ± 1.16
underlying semantic of the original hypergraph is broken.           Pubmed                                              84.02 ± 0.56                     84.26 ± 0.62
                                                                    Cora-A                                              79.77 ± 0.95                     82.15 ± 0.89
                                                                    DBLP                                                90.19 ± 0.12                     91.12 ± 0.11
             E     Additional Experiments                           Zoo                                                 80.43 ± 11.3                     80.25 ± 11.2
E.1    Ablation Study: Effects of Self-loops                        20News                                              79.56 ± 0.24                     80.14 ± 0.19
                                                                    Mushroom                                            99.80 ± 0.14                     99.83 ± 0.13
In TriCL, we add self-loops to the hypergraph after hyper-          NTU2012                                             74.01 ± 2.54                     75.23 ± 2.45
graph augmentation and before it is passed through the en-          ModelNet40                                          93.48 ± 0.16                     97.08 ± 0.13
coder. We conduct an ablation study to demonstrate the ef-
fects of self-loops. The results are summarized in Table 8,
and it empirically verifies that adding self-loops is advanta-     pooling layer consistently and slightly outperforms the one
geous. The reason for the better performance we speculate          with HGNN as an encoder. This result justifies our choice of
is that a self-loop helps each node make a better use of its       using the mean pooling layer as our backbone encoder.
initial features. Specifically, a hyperedge corresponding to a
self-loop receives a message only from the node it contains        E.3         Sensitivity Analysis
and sends a message back to the node without aggregating           We investigate the impact of hyperparameters used in TriCL,
the features of any other nodes. This allows each node to          especially, τg and τm in Eq. (6) and (7) as well as ωg and ωm
make a better use of its features.                                 in Eq. (8), with the Citeseer and Cora Co-citation datasets.
                                                                   We only change these hyperparameters in this analysis, and
E.2    Ablation Study: Backbone Encoder                            the others are fixed as provided in Appendix C.2.
The superiority of the encoder used in TriCL over HGNN is             We conduct node classification while varying the values
verified in Table 9. We compare the accuracy of two TriCL          of τg and τm from 0.1 to 1.0 and report the accuracy gain
models that use (1) HGNN and (2) the mean pooling layer            over TriCL-N, which only considers node-level contrast, in
(proposed), respectively, as an encoder. TriCL with the mean       Figure 4. From the figure, it can be observed that TriCL
                                        Cora Co-authorship                 High                                    Cora Co-authorship                  High
                           1.0 0.7 1.2 0.8 0.6 0.5 0.6 0.7 0.7 0.8 0.9                                 24 1.1 1.1 1.2 1.2 1.3 1.3 1.2 0.7 -0.3
                           0.9 0.8 1.3 0.9 0.8 0.7 0.8 0.8 0.9 0.9 1.0                                 23 1.2 1.2 1.2 1.3 1.4 1.3 1.1 0.5 0.0
                           0.8 0.8 1.3 1.1 1.0 0.9 1.0 1.0 1.0 1.1 1.1                                 22 1.1 1.2 1.2 1.3 1.4 1.2 0.9 -0.0 0.1


        Temperature ( g)
                           0.7 0.7 1.3 1.3 1.2 1.1 1.1 1.1 1.2 1.2 1.3                                 21 1.1 1.2 1.2 1.2 1.2 1.0 0.5 -0.5 -0.3
                                                                                        Weight ( g)
                           0.6 0.7 1.2 1.5 1.3 1.3 1.2 1.2 1.3 1.3 1.3                                 20 0.9 0.9 1.0 1.0 0.9 0.6 -0.1 -0.2 -1.2
                           0.5 0.7 1.2 1.4 1.3 1.4 1.3 1.3 1.3 1.3 1.3
                           0.4 0.6 1.1 1.3 1.3 1.3 1.3 1.3 1.3 1.3 1.3                                2 1 0.7 0.7 0.8 0.8 0.5 0.1 -0.2 -0.4 -1.3
                           0.3 0.4 0.8 1.1 1.2 1.3 1.3 1.2 1.2 1.2 1.2                                2 2 0.4 0.4 0.5 0.6 0.3 -0.2 -0.0 -0.5 -1.3
                           0.2 -0.2 0.4 0.7 0.9 0.9 0.9 0.9 0.9 0.8 0.8                               2 3 0.2 0.3 0.4 0.4 0.1 -0.2 -0.1 -0.5 -1.6
                           0.1 -1.0 -0.2 0.1 0.3 0.3 0.3 0.3 0.2 0.2 0.2                              2 4 0.1 0.2 0.3 0.3 0.0 0.2 -0.1 -0.7 -1.5
                                                                           Low                                                                         Low
                               0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0                                    2 4 2 3 2 2 2 1 20 21 22 23 24
                                             Temperature ( m)                                                          weight ( m)
                                              Citeseer                     High                                          Citeseer                      High
                           1.0 0.3 1.0 1.7 2.2 2.4 2.5 2.5 2.5 2.5 2.4                                 24 1.6 1.7 1.7 1.7 1.9 2.1    2.5    2.4 2.2
                           0.9 0.3 1.0 1.8 2.2 2.4 2.5 2.5 2.5 2.4 2.4                                 23 1.6 1.6 1.7 1.8 1.9 2.3    2.6    2.3 1.6
                           0.8 0.3 1.0 1.7 2.2 2.4 2.5 2.5 2.4 2.4 2.3                                 22 1.6 1.6 1.6 1.8 2.1 2.5    2.5    2.1 1.3


        Temperature ( g)
                           0.7 0.3 1.0 1.7 2.1 2.4 2.4 2.4 2.4 2.3 2.3                                 21 1.4 1.5 1.6 1.7 2.1 2.6
                                                                                        Weight ( g)
                                                                                                                                     2.2    1.4 0.6
                           0.6 0.2 1.0 1.7 2.1 2.4 2.4 2.4 2.3 2.3 2.3                                 20 1.2 1.2 1.4 1.6 2.0 2.2    1.6    0.9 -0.7
                           0.5 0.2 1.1 1.7 2.0 2.2 2.3 2.3 2.3 2.2 2.2
                           0.4 0.2 1.0 1.6 2.0 2.1 2.2 2.2 2.2 2.1 2.1                                2 1 0.9 1.0 1.1 1.5 1.8 1.8    1.1    0.3 -2.3
                           0.3 0.2 0.9 1.5 1.8 2.0 2.0 2.0 2.0 1.9 1.9                                2 2 0.7 0.9 1.0 1.4 1.7 1.2    1.0   -0.0 -2.7
                           0.2 0.3 0.9 1.4 1.7 1.8 1.8 1.8 1.8 1.8 1.7                                2 3 0.6 0.8 1.0 1.4 1.2 0.9    0.5   -0.4 -2.7
                           0.1 0.4 1.1 1.5 1.6 1.6 1.6 1.5 1.5 1.5 1.4                                2 4 0.5 0.7 1.0 1.1 0.9 0.7    0.3   -0.3 -2.5
                                                                           Low                                                                         Low
                               0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0                                    2 4 2 3 2 2 2 1 20 21      22    23    24
                                           Temperature ( m)                                                            weight ( m)
Figure 4: Node classification accuracy gain (%) of TriCL                          Figure 5: Node classification accuracy gain (%) of TriCL
over TriCL-N, when different temperature parameter pairs                          over TriCL-N, when different weight pairs are used. The
are used. The baseline accuracies are 70.28% and 80.49%                           baseline accuracies are 70.28% and 80.49% for the Citeseer
for the Citeseer and Cora Co-citation datasets, respectively.                     and Cora Co-citation datasets, respectively.

Table 9: The accuracy of two TriCL methods that use (1)                           time per epoch (ms). Note that subsampling is not used and
HGNN and (2) the mean pooling layer (proposed), respec-                           TriCL computes the membership contrastive loss with mini-
tively, as an encoder.                                                            batches of size 4096 when training for the Pubmed, DBLP,
                                                                                  20News, Mushroom, and ModelNet40 datasets due to the
      Dataset                        TriCL (with HGNN)         TriCL (Proposed)   memory limits. Generally, TriCL-NG shows similar execu-
      Cora-C                             81.03 ± 1.31             81.57 ± 1.12    tion times to baseline approaches, and TriCL is slower due
      Citeseer                           71.97 ± 1.26             72.02 ± 1.16    to membership contrast.
      Pubmed                             83.80 ± 0.63             84.26 ± 0.62
      Cora-A                             82.22 ± 1.09             82.15 ± 0.89                              F    Qualitative Analysis
      DBLP                               90.93 ± 0.16             91.12 ± 0.11
      Zoo                                79.47 ± 11.0             80.25 ± 11.2    F.1   Analysis on the Pubmed, Cora
      20News                             79.93 ± 0.24             80.14 ± 0.19          Co-authorship, and DBLP datasets
      Mushroom                           98.93 ± 0.33             99.83 ± 0.13    As additional experiments, in Table 11, We provide t-
      NTU2012                            74.63 ± 2.53             75.23 ± 2.45    SNE (Van der Maaten and Hinton 2008) plots of the
      ModelNet40                         97.33 ± 0.14             97.08 ± 0.13
                                                                                  node representations produced by TriCL and its two vari-
                                                                                  ants, TriCL-N and TriCL-NG, on the Pubmed, Cora Co-
achieves an accuracy gain in most cases when both the tem-                        authorship, and DBLP datasets. As expected from the quan-
perature parameters are not too small (i.e., 0.1), as shown                       titative results, the 2-D projection of embeddings learned
in the blue area in the figure. It indicates that pursuing ex-                    by TriCL shows more numerically (based on Silhouette
cessive uniformity in the embedding space rather degrades                         score (Rousseeuw 1987)) distinguishable clusters than its
the node classification performance (Wang and Liu 2021).                          two variants.
We also conduct the same task while varying the values of
ωg and ωm from 2−4 to 24 and report the accuracy gain                             F.2   Analysis on the Collapsed Models
over TriCL-N, in Figure 5. Using a large ωg and a small ωm                        Using membership contrast alone sometimes causes model
together degrades the performance. This causes model col-                         collapse. t-SNE plots of the collapsed models are shown in
lapse by making the proportion of membership contrastive                          Figure 6. There is no clear distinction between the represen-
loss relatively larger than node and group contrastive losses.                    tations of nodes of different classes, and they overlap. It even
                                                                                  looks randomly scattered around two clusters in the Citeseer
E.4       Training Time Comparison                                                dataset. One potential reason the model fails to produce sep-
We compare the training time of baseline models and                               arable embeddings is that there is no guidance between node
TriCL by the elapsed time of a single epoch. We run each                          representations or between edge representations. Using node
method for 50 epochs and measured the average elapsed                             or group contrast together, this problem could be solved.
Table 10: Single epoch running time (in milliseconds) averaged over 50 training epochs. The hyphen(-) indicates that the
running time cannot be measured due to out of memory.

                              Cora-C   Citeseer   Pubmed    Cora-A      DBLP    Zoo    20News    Mushroom   NTU2012   ModelNet40
             DGI                4         4         16            4       76      4         -          -        4         13
             GRACE             15        15         32           16        -     16         -          -       18         93
             S2 -HHGR          25        25         54           36      524     9         185        94       33        142
             TriCL-N            13       12         18        11         488     10         78         37      11         50
             TriCL-NG           19       17         38        17         625     15         83         40      16         85
             TriCL             102       79        652       121        3,156    32        396        702     194        636



Table 11: t-SNE plots of the node representations from TriCL and its two variants. The TriCL’s embeddings exhibits the most
distinct clusters with the help of group and membership contrast, as measured by the Silhouette score (the higher, the better).

                                       Pubmed                         Cora Co-authorship                    DBLP




                   TriCL-N




                   TriCL-NG




                   TriCL




                                                  (a) Citeseer                             (b) DBLP

Figure 6: t-SNE plots of the node representations from TriCL when model collapse occurred. There is no clear distinction
between the representations of nodes of different classes.

