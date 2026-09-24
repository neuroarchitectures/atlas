# Multiplex Heterogeneous Graph Neural Network with Behavior P Qingdao China 2023

> Source: `Multiplex_Heterogeneous_Graph_Neural_Network_with_Behavior_P_Qingdao_China_2023.pdf`

---

 Multiplex Heterogeneous Graph Neural Network with Behavior
                      Pattern Modeling
                       Chaofan Fu                                                  Guanjie Zheng                                                             Chao Huang
             Ocean University of China                                   Shanghai Jiao Tong University                                         The University of Hong Kong
                  Qingdao, China                                               Shanghai, China                                                      Hong Kong, China
             fuchaofan@stu.ouc.edu.cn                                        gjzheng@sjtu.edu.cn                                                chaohuang75@gmail.com

                                                       Yanwei Yu∗                                                      Junyu Dong
                                            Ocean University of China                                        Ocean University of China
                                                Qingdao, China                                                   Qingdao, China
                                             yuyanwei@ouc.edu.cn                                              dongjunyu@ouc.edu.cn

ABSTRACT                                                                                              ACM Reference Format:
Heterogeneous graph neural networks have gained great popularity                                      Chaofan Fu, Guanjie Zheng, Chao Huang, Yanwei Yu, and Junyu Dong. 2023.
                                                                                                      Multiplex Heterogeneous Graph Neural Network with Behavior Pattern
in tackling various network analysis tasks on heterogeneous net-
                                                                                                      Modeling. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge
work data. However, most existing works mainly focus on general                                       Discovery and Data Mining (KDD ’23), August 6–10, 2023, Long Beach, CA,
heterogeneous networks, and assume that there is only one type of                                     USA. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3580305.
edge between two nodes, while ignoring the multiplex character-                                       3599441
istics between multi-typed nodes in multiplex heterogeneous net-
works and the different importance of multiplex structures among
                                                                                                                                                                       Meta-path based model MAGNN
nodes for node embedding. In addition, the over-smoothing issue of                                        E-commerce       (a) Meta-path
                                                                                                                             sampling
                                                                                                                                           ……       ……       ……                0.525
                                                                                                                                                                                               Node Classification
graph neural networks limits existing models to only capturing local                                      Network
                                                                                                                                                                                                           0.473
structure signals but hardly learning the global relevant information                                                                                                          0.475
                                                                                                                                           U-I              U-I-U
of the network. To tackle these challenges, this work proposes a
                                                                                                                                                                     Relation 0.425              0.403
model called Behavior Pattern based Heterogeneous Graph Neural                                                                                                       aware
                                                                                                                             (b) Matrix
                                                                                                                                           ……                ……      meta-path
                                                                                                                                                                               0.375
Network (BPHGNN) for multiplex heterogeneous network embed-                                                                 aggregation             ……               model
                                                                                                                                                                                       0.344
                                                                                                                                                                     MHGCN
ding. Specifically, BPHGNN can collaboratively learn node repre-                                                                                                               0.325
                                                                                                                                                                                       MAGNN MHGCN BPHGNN
sentations across different multiplex structures among nodes with                                                                          ����            �������
adaptive importance learning from local and global perspectives in
multiplex heterogeneous networks through depth behavior pattern                                                  click       (c) Basic
                                                                                                                                                                                        Our BPHGNN with
                                                                                                                 cart        Behavior
                                                                                                                                                                                        behavior pattern
aggregation and breadth behavior pattern aggregation. Extensive                                                  buy          Pattern
                                                                                                                                                                                        modeling
                                                                                                                             generator
experiments on six real-world networks with various network ana-                                                 collect
                                                                                                                                           Basic Behavior Patterns
lytical tasks demonstrate the significant superiority of BPHGNN
against state-of-the-art approaches in terms of various evaluation                                    Figure 1: The difference between multiplex heterogeneous
metrics.                                                                                              network embedding methods and performance improvement
                                                                                                      (measured by Macro-F1) of the proposed model over MHGCN
CCS CONCEPTS                                                                                          and MAGNN on Alibaba dataset: (a) Learning node embed-
• Mathematics of computing → Graph algorithms; • Comput-                                              ding through meta-path sampling; (b) Learning node em-
ing methodologies → Learning latent representations.                                                  bedding through relation-aware meta-path aggregation; (c)
                                                                                                      Learning node embedding through our basic behavior pat-
KEYWORDS                                                                                              tern modeling. As can be seen, BPHGNN achieves a +17.37%
Graph Representation Learning; Multiplex Heterogeneous Net-                                           performance lift compared to MHGCN.
works; Graph Neural Networks; Basic Behavior Pattern
∗ Yanwei Yu is the corresponding author.                                                              1      INTRODUCTION
                                                                                                      In recent years, network representation learning has emerged as
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed                 a new learning paradigm to embed complex networks into a low-
for profit or commercial advantage and that copies bear this notice and the full citation             dimensional vector space while preserving the proximities of nodes
on the first page. Copyrights for components of this work owned by others than the                    in both network topological structures and intrinsic properties,
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission         which advances various network analysis tasks, ranging from node
and/or a fee. Request permissions from permissions@acm.org.                                           classification [20, 22, 33, 44, 48], link prediction [6, 14, 37, 57], to
KDD ’23, August 6–10, 2023, Long Beach, CA, USA                                                       recommendation [19, 34, 39]. In particular, a class of graph neu-
© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-0103-0/23/08. . . $15.00                                                          ral networks, e.g., Graph Convolutional Network (GCN) [22], is
https://doi.org/10.1145/3580305.3599441                                                               proposed to learn node representation for complex networks with




                                                                                                482
KDD ’23, August 6–10, 2023, Long Beach, CA, USA                                                     Chaofan Fu, Guanjie Zheng, Chao Huang, Yanwei Yu, and Junyu Dong


attribute features, which has been successfully applied in many                 information usually represents low-level semantic information in
domains such as recommendation systems [19, 50, 64], natural                    the network, such as whether the user nodes are friends in a so-
language processing [25, 41, 53, 54], and spatio-temporal predic-               cial network, or whether a user clicks or purchases a product in
tion [8, 9, 36, 56].                                                            an E-commerce network. This kind of information is indeed very
    Early methods have made many efforts on representation learn-               important in network embedding learning, but high-level seman-
ing for homogeneous networks with a single type of nodes [14, 22,               tic information, e.g., community structures, or similar behavior
35, 42]. To capture the network heterogeneity properties, many sub-             interactions, is also necessary. However, existing multiplex hetero-
sequent studies are designed to model heterogeneous graph struc-                geneous methods do not exploit this type of information in network
tures based on predefined meta-paths, such as metapath2vec [10],                representation learning.
HIN2Vec [12], and HERec [39]. Inspired by the advancement of
graph neural networks (GNNs) in achieving unprecedented suc-                               0.35                            0.36                              0.35
                                                                                                                                             0.325                        0.31 0.292
cess for graph representation tasks, various GNN models have                                           0.279 0.284                   0.319
                                                                                            0.3                      0.263 0.33                      0.309    0.3 0.275                0.267
been proposed to capture the rich neighborhood contextual signals                                                               0.302
                                                                                Macro-F1
                                                                                                  0.252
for heterogeneous graph learning, such as relational graph convo-                          0.25                             0.3                              0.25
lutional networks (R-GCN) [37], heterogeneous graph attention
networks (HAN) [45], meta-path aggregated graph neural network                              0.2                            0.27                               0.2
                                                                                                   1      2   3       4           1    2      3       4             1      2    3       4
(MAGNN) [13], and heterogeneous graph structure Learning net-
work (HGSL) [66].                                                                                         GTN                          HGTN                               MHGCN

    In reality, many networks are much more complex, comprised
of not only multi-typed nodes and diverse edges even between                    Figure 2: The node classification performance with different
the same pair-wise nodes but also a rich set of attributes [3]. Such            layers (measured by Macro-F1) on Taobao dataset
networks are ubiquitous in a variety of contexts. For example, as
shown in Figure 1, users in an E-commerce platform may have                        (3) Over-smoothing issue of GNNs limits the model repre-
different kinds of interactions (e.g., click, purchase, add-to-cart, or         sentation performance. The graph neural network architecture
add-to-collect) with items, which form a multiplex heterogeneous                with deeper convolution layers may result in indistinguishable
network. While the existing heterogeneous network embedding                     node vectors, which limits the representation quality of high-order
methods [10, 12, 18, 19, 29, 38, 47] can be applied on such networks,           heterogeneous relations. Figure 2 illustrates the over-smoothing
it has been shown that they yield suboptimal representations be-                issues of GNNs (e.g., GTN, HGTN, MHGCN) on node classification,
cause these methods do not model heterogeneous interactions or                  and the same phenomenon for recommendation performance on
multiple edge types in multiplex heterogeneous networks. As a re-               LightGCN [15], GCCF [7], PinSAGE [55] and ST-GCN [61] is also
sult, several representation learning methods have been developed               proved in [51]. The over-smoothing issue restricts GNN models
specifically for multiplex heterogeneous networks, e.g., GATNE [3],             from stacking more aggregation layers, that is, GNN models can
FAME [28], DualHGCN [52] and MHGCN [57]. Although the above                     only aggregate low-order local information and cannot capture
models have offered state-of-the-art performance for multiplex                  higher-order global information. This also makes it difficult for
heterogeneous network embedding, three key issues remain less                   GNNs relying on neighbor aggregation to learn the global rele-
explored:                                                                       vant information of the network. It should be emphasized that the
    (1) None of the existing methods can effectively model mul-                 purpose of this work is not to solve the over-smoothing issue of
tiplex structures in multiplex heterogeneous networks. The                      GNNs, but to provide a way to mitigate the problem of difficulty in
significant difference between a multiplex heterogeneous network                aggregating global information due to the over-smoothing issue.
and a general heterogeneous network is the multiplex structure be-              That is, even when the number of layers is set too small, we can also
tween nodes, which yields networks with multiple different views.               learn global relevant information to improve model performance.
Therefore, considering such multiplex structures is crucial for mul-               Presented Work. To address the aforementioned challenges, we
tiplex heterogeneous network embedding. Nevertheless, none of                   propose a novel heterogeneous graph neural network with behavior
the existing methods explicitly model the multiplex heterogeneous               pattern modeling, named BPHGNN, for multiplex heterogeneous
structures, but treat them as a linear superposition of relations. For          network embedding. Specifically, to incorporate the characteristics
example, MHGCN first decouples the network by different relations               of multiple layers into node embedding, we first define the concept
and then uses the weighted superposition of different relations to              of Basic Behavior Pattern (BBP) to model the multiplex structures in
represent multiplex relational structures. But in fact, the multiplex           multiplex heterogeneous networks. To automatically capture local
structures contain richer semantic information, such as mutual pro-             information across different multiplex structures, we design a depth
motion or mutual inhibition, and should not be simply regarded as               behavior pattern aggregation network, which can aggregate the local
a linear superposition of individual relations. However, none of the            information through depth behavior patterns with adaptive impor-
existing methods can effectively model such multiplex structures                tance learning of all basic behavior patterns. Additionally, to learn
for multiplex heterogeneous network embedding.                                  global relevant information, we propose a breadth behavior pattern
    (2) Existing approaches pay attention to low-level seman-                   aggregation network to implement feature aggregation among nodes
tic information and ignore high-level semantic information.                     according to the similarity of breadth behavior patterns between
At present, almost all methods learn network representations by                 nodes. Furthermore, contrastive learning is used to collaboratively
aggregating neighbor feature information. This kind of neighbor                 learn node representations from local and global perspectives, and




                                                                          483
Multiplex Heterogeneous Graph Neural Network with Behavior Pattern Modeling                                       KDD ’23, August 6–10, 2023, Long Beach, CA, USA


fuse them to obtain the final representation. Experiments on six real-              introduces two hypergraphs into the representation learning of
world datasets show the significant superiority of our proposed                     multiplex heterogeneous networks, and learns node embedding
model over state-of-the-art baselines. As illustrated in Figure 1,                  using spectral hypergraph convolutional networks. FAME [28] and
BPHGNN achieves a +17.37% performance lift in terms of Macro-F1                     MHGCN [57] realize message passing in multiplex heterogeneous
compared to MHGCN for node classification on Alibaba dataset.                       networks by automatically capturing useful relation-aware meta-
   We summarize the key contributions of this work as follows:                      paths. However, the existing multiplex heterogeneous network
                                                                                    representation learning methods do not pay enough attention to
     • We propose an effective multiplex heterogeneous graph neu-
                                                                                    the multiplex structure modeling in the multiplex heterogeneous
       ral network, called BPHGNN. BPHGNN is the first to con-
                                                                                    network, only regard it as a simple superposition of relationships,
       sider modeling the multiplex structures for multiplex het-
                                                                                    and also ignore the high-level semantic information learning.
       erogeneous network embedding, which can collaboratively
       learn node representation from both local and global per-
       spectives leveraging contrastive learning.                                   3    PROBLEM DEFINITION
     • Based on the defined basic behavior pattern, we further pro-                 Generally, a network is denoted as G = {V, E}, where V is the
       pose the depth behavior pattern aggregation and the breadth                  collection of nodes, and E is the collection of edges between the
       behavior pattern aggregation, so that our model can not                      nodes, each edge representing a relationship between two nodes.
       only aggregate neighbor information adaptively with depth
       behavior patterns, but also learn high-level semantic infor-                    Definition 1 (Attributed Multiplex Heterogeneous Net-
       mation such as breadth behavior pattern similarity.                          work, or AMHEN). Given the defined network G, we further as-
     • We conduct extensive experiments on six real-world datasets                  sociate all nodes in V with the attribute feature matrix X ∈ R𝑛×𝑚 .
       to verify the superiority of our BPHGNN in both node clas-                   Here, 𝑛 and 𝑚 represent the size of node set V and attribute features,
       sification and link prediction when competing with state-                    respectively. With the consideration of node and edge heterogeneity, we
       of-the-art baselines. We also prove the effectiveness of each                define the node type and edge type mapping functions as 𝜙 : V → O
       sub-module in our model through ablation study.                              and 𝜓 : E → R, where O and R denote the set of all node types and
                                                                                    the set of all edge types, respectively. Each node 𝑣 ∈ V belongs to
                                                                                    a particular node type, and each edge 𝑒 ∈ E is categorized into a
2    RELATED WORK
                                                                                    specific edge type. If |O| + |R| > 2, the network is heterogeneous.
Homogeneous Network Embedding. In literature, a large body                          With the consideration of edge multiplexity, if multi-typed edges exist
of work extensively studies representation learning for homoge-                     between the same node pairs, the network is attributed multiplex
neous networks, ranging from random walk-based methods [14,                         heterogeneous.
33], random projection methods [5, 65], matrix factorization-based
methods [2, 35], to graph neural networks [22, 43, 48, 49]. However,                   Although meta-path is a commonly used tool in heterogeneous
these methods do not take the heterogeneity of the network into                     network embedding, it is difficult to describe the multiplex structure
account. Hence, it is difficult to directly apply these methods to                  in multiplex heterogeneous networks. We next propose the concept
heterogeneous network representation learning.                                      of basic behavior pattern, which is a tool used to describe the
Heterogeneous Network Embedding. Heterogeneous graph                                complex multiplex interaction between node pairs in AMHENs.
learning has already received extensive attention [16, 18, 24, 39]
due to the ubiquity of heterogeneous graph modeling. Among                            Definition 2 (Basic Behavior Pattern, or BBP). A basic be-
heterogeneous network embedding methods, meta-path is a gen-                        havior pattern between two node types 𝑂𝑖 and 𝑂 𝑗 in multiplex hetero-
eral way to model network heterogeneity. For example, metap-                                                               [𝑟 1 ]&[𝑟 2 ]&···&[𝑟 |R| ]
                                                                                    geneous networks is defined as 𝑂𝑖 −−−−−−−−−−−−−−−−−→ 𝑂 𝑗 which de-
ath2vec [10], HAN [45], MAGNN [13], HERec [39], HIN2rec [12]
                                                                                    scribes the multiplex interaction behaviors between two nodes, where
and HeCo [46] show good results in heterogeneous network repre-
                                                                                    [·] denotes optional, and at least one relation 𝑟𝑖 exists.
sentation learning by using specified meta-paths, while GTN [58],
SR-RSC [62] can automatically capture various meta-paths in the                                                         𝑐𝑙𝑖𝑐𝑘
heterogeneous graph. In addition to meta-path, there are also some                    For example, in Figure 1, 𝑈 −−−−→ 𝐼 is a basic behavior pattern
methods that show excellent performance in heterogeneous net-                       between user and item nodes, which represents that there is only
                                                                                                                                              𝑐𝑙𝑖𝑐𝑘&𝑏𝑢𝑦
work embedding by means of heterogeneous graph neural network,                      click interaction between users and items. 𝑈 −−−−−−−−−→ 𝐼 is also
hypergraph, VAE [21], generative adversarial network (GAN), and                     a basic behavior pattern, where there are both click and purchase
other tools, such as HetGNN [59], MWNN [40], Hyper-SAGNN [63],                      behaviors between users and items. As shown in Figure 1, the
HeGAN [17], HeterHG-VAE [11]. However, none of the above het-                       interactions between 𝑢 1 and 𝑖 1 , 𝑢 1 and 𝑖 2 , and 𝑢 3 and 𝑖 3 all belong
erogeneous models can effectively learn multiplex relational signals                               𝑐𝑙𝑖𝑐𝑘&𝑏𝑢𝑦
                                                                                    to pattern 𝑈 −−−−−−−−−→ 𝐼 , and no interaction between users and
among multi-typed nodes in multiplex heterogeneous networks.
                                                                                                          𝑐𝑙𝑖𝑐𝑘
Multiplex Heterogeneous Network Embedding. Recently, many                           items belongs to 𝑈 −−−−→ 𝐼 .
multiplex network embedding approaches [4, 20, 27, 66] are de-                         Notice that for a multiplex heterogeneous network with |R| rela-
signed to project diverse node edges into latent representations.                   tions, it can generate up to 2 | R | − 1 basic behavior patterns. Taking
MNE [60], GATNE [3], and DMGI [32] learn the representation of                      the e-commerce network in Figure 1 as an example, it has four
nodes in each specific relationship, respectively, and then aggre-                  types of edges, so it can generate up to 15 basic behavior patterns.
gate them to obtain the final node representation. DualHGNN [52]                    However, based on the instance network in Figure 1, because some




                                                                              484
KDD ’23, August 6–10, 2023, Long Beach, CA, USA                                                                    Chaofan Fu, Guanjie Zheng, Chao Huang, Yanwei Yu, and Junyu Dong



               click(cl)
                                         00001100                           00000000
                                                                            00001000   Rows 1
                                                                                              0
                                                                                                                    Depth Behavior Pattern Aggregation
                                         00000100
                                                                            00000000        0
                                         00000010                                      Sum 0      ��
              cart(ca)                                                      00000000                                                  ������
                                         00000010
                                              �
                                         1 0 0 0 0�0 0 0
                                                                              ���
                                                                            01000000
                                                                            00000000
                                                                                              0
                                                                                              1             Weighted
              buy(bu)                    11000000                           00000000          0              Sum                           ···
                                                                            00000000    ��    0                                                                          MeanPooling   ···
             collect(co)                 00110000
                                                                                                                                           ������
                                         00000000                           00000000          0
                                                                            00000000   Rows   0

                                         00000000
                                                                            00000000
                                                                            00000001   Sum
                                                                                              0
                                                                                              1
                                                                                                  ��                                                                                   ···
                                         00001100
                                         00000000
                                                                              ���
                                                                            00000000
                                                                            00000000
                                                                                              0
                                                                                              0
                                                                                                               ������           (�)               (�−�)    (�)
                                                                                                                               ������ = ������ ∙ ������ ∙ ������

       u�            ��                  00000000                           00000000          0
                                              �
                                         0 0 1 0 0�0 0 0
                                                                            00010000    ��    1

                                         00100000                           00001100          2
                                         00000000                           00000000   Rows 0
      ��             ��                  00000000          Basic Behavior   00000010
                                                                                       Sum 0
                                                                                            1     ��                                                                                   ···
                                                           Pattern Matrix
                                                             Generator
                                                                            ���&��
                                                                            00000000
                                                                            10000000          1
                                                                                                                                                                         �������
                                         00001100
                                                                                                        ���������(�·�� )
                                                                            10000000          1
                                         00000000                           00100000          1
      ��             ��                                                     00000000    ��    0                                                                                        ···
                                         00000010
                                         00000000
                                              �
                                         1 0 0 0 0�0 0 0
                                                                            00000000
                                                                            00000100
                                                                            00000000
                                                                                       Rows 1
                                                                                              0
                                                                                                                                               ������
                                         10000000                                           0     ��
                                                                            00000000   Sum 0
      ��             ��                  00100000
                                                                            ���&��
                                                                            00000000          0                                                    ···
                                         00000000                           01000000          1            �         �������
            user                                                            00000000
                                                                                        ��
                                                                                              0                                                    ������
                                         00000000                           00000000          0                                                                                        ···
                                         00000000
                                                                            00000000          0                                          (�)                     (�−�)   (�)
                                         00000000                                                                                      ������� = ������� ∙ ������� ∙ �������
                                         00000011
                                                                            00000000   Rows   0

            item                              �
                                         0 0 0 0 0�0 0 0
                                                                            00000000
                                                                            00000010   Sum
                                                                                              0
                                                                                              1
                                                                                                  ��                                                                                   ···
                                                                                                        Weighted
                                         00000000                           ���&��
                                                                            00000000
                                                                            00000000
                                                                                              0
                                                                                              0          Concat
                                         00010000
                                         00010000
                                                                            00010000
                                                                            00000000    ��
                                                                                              1
                                                                                              0
                                                                                                                    Breadth Behavior Pattern Aggregation


                                                     Figure 3: The overview of the proposed BPHGNN.

                                                                             𝑐𝑙𝑖𝑐𝑘
basic behavior patterns do not have instances, such as 𝑈 −−−−→ 𝐼 ,                                among nodes, and fuse them to obtain the final representation for
only five basic behavior patterns are finally produced.                                           downstream tasks.
  Based on the above definitions, we formally defined our studied
problem for representation learning over the multiplex heteroge-
neous network.                                                                                    4.1      Basic Behavior Pattern Generator
                                                                                                  To make full use of the multiplex interaction structures between
   Problem (AMHEN Representation Learning). Given an AMHEN
                                                                                                  nodes in multiplex heterogeneous networks, we first design a basic
G = {V, E, X}, the goal of our representation learning task is to learn
                                                                                                  behavior pattern generator that can directly extract all the basic
a 𝑑-dimensional latent embedding (𝑑 ≪ |V |) for each node 𝑣 ∈ V,
                                                                                                  behavior pattern matrices from a multiplex heterogeneous network.
with the preservation of network heterogeneity and multiplexity and
                                                                                                     We first decouple the multiplex heterogeneous network accord-
high-level semantic information.
                                                                                                  ing to the type of edges. Let {A𝑟 ∈ R𝑛×𝑛 |𝑟 = 1, 2, . . . , |R|} de-
    The key notations are summarized in Table 5 in the supplement.                                note the basic adjacency matrices of the generated sub-graphs,
                                                                                                  where 𝑛 is the number of all nodes in the network. Then, each
4    METHODOLOGY                                                                                  adjacency matrix and a corresponding logical variable (i.e., 1 or
                                                                                                  0) are operated with the 𝑋 𝑁𝑂𝑅 to generate |R| intermediate ma-
In this section, we present the details of our BPHGNN with the                                            ® 𝑟 ∈ R𝑛×𝑛 |𝑟 = 1, 2, . . . , |R|}. Here, a logical variable of
                                                                                                  trices {A
overall architecture shown in Figure 3. Particularly, our BPHGNN
                                                                                                  1 is taken if the relation represented by the adjacency matrix is
consists of four key components: (i) Basic behavior pattern generator,
                                                                                                  preserved in the basic behavior pattern, and 0 otherwise. Finally,
(ii) Depth behavior pattern aggregation, (iii) Breadth behavior pattern
                                                                                                  the intermediate matrices are bitwise 𝐴𝑁 𝐷 operated to obtain the
aggregation and (vi) Contrastive learning. Basic behavior pattern
                                                                                                  basic behavior pattern matrix. That is, these relations correspond-
generator is used to extract all basic behavior patterns from the
                                                                                                  ing to a logical variable value of 1 are retained in the final basic
multiplex heterogeneous network, so as to make full use of the
                                                                                                  behavior pattern. Notice that if a zero matrix is obtained, then this
multiple interaction structures in the multiplex heterogeneous net-
                                                                                                  basic behavior pattern does not exist in the network. By adjust-
work. Depth behavior pattern aggregation can realize information
                                                                                                  ing the logical variables, all the basic behavior pattern matrices
aggregation through depth behavior patterns in multiplex hetero-
                                                                                                  {Ā𝑖 ∈ R𝑛×𝑛 |𝑖 = 1, 2, . . . , |2 | R | − 1|} can be obtained.
geneous networks from a local perspective, and can adaptively
                                                                                                     For example, if our goal is to obtain the basic behavior pattern
learn the importance of various basic behavior patterns, so as to                                      𝑐𝑙𝑖𝑐𝑘&𝑏𝑢𝑦
achieve a better information aggregation effect. Breadth behavior                                 𝑈 −−−−−−−−−→ 𝐼 in the E-commerce network, therefore, the logical
pattern aggregation aims to implement information aggregation                                     variables {𝑐𝑙 : 1, 𝑐𝑎 : 0, 𝑏𝑢 : 1, 𝑐𝑜 : 0} are used to 𝑋 𝑁𝑂𝑅 with the
between nodes from a global perspective according to the similar-                                 corresponding adjacency matrices, respectively. Namely, only click
ity of breadth behavior patterns between nodes. The more similar                                  and buy relation corresponding to a value of 1 is kept in the basic
the breadth behavior patterns between two nodes are, the stronger                                 behavior pattern.
their connection will be in the breadth behavior pattern aggregation.                                To generate basic behavior patterns efficiently, we can gener-
Contrastive learning is to collaboratively learn node representations                             ate all basic behavior pattern matrices in parallel through logical
from local and global perspectives to capture complex relationships                               operations between matrices with different logical variables.




                                                                                          485
Multiplex Heterogeneous Graph Neural Network with Behavior Pattern Modeling                                             KDD ’23, August 6–10, 2023, Long Beach, CA, USA


4.2    Depth Behavior Pattern Aggregation                                                For example, in E-commerce networks, cautious consumers are
A depth behavior pattern is a longer behavior pattern, which is a                        more likely to click to browse a large number of products, then
composite pattern of multiple basic behavior patterns. It describes                      add some products to the shopping cart, and then buy fewer prod-
the basic behavior pattern from the perspective of depth, so we call                     ucts; buyers or impulsive users are more inclined to click directly
it the depth behavior pattern.                                                           and then buy a large number of products. Therefore, node repre-
    The overall process of depth behavior pattern aggregation is                         sentations that consider behavior similarities among nodes would
shown in the upper right part (brown) of Figure 3. After obtaining                       benefit many downstream network analysis tasks, such as node
basic behavior patterns, the depth behavior pattern aggregation                          classification.
module aims to aggregate features from the multiplex structures                              The purpose of breadth behavior pattern aggregation is to aggre-
among nodes from the local perspective by differentiating each                           gate the features between nodes from the global perspective based
basic behavior pattern with different importance.                                        on the similarity of breadth behavior patterns between nodes. The
    To capture multiplex structures between nodes, the depth behav-                      overall process of breadth behavior pattern aggregation is shown
ior pattern aggregation module first uses a set of learnable weight                      in the lower right part (blue) of Figure 3.
parameters 𝛼𝑖 to aggregate basic behavior patterns as:                                       Specifically, we first generate a matrix to represent the breadth
                                                                                         behavior pattern of nodes based on the obtained basic behavior
                                           N
                                                                                         patterns. In particular, we first add the rows to get a column vector
                                          ∑︁
                              Ã𝑙𝑜𝑐𝑎𝑙 =          𝛼𝑖 Ā𝑖 ,                    (1)
                                           𝑖=1
                                                                                         for each basic behavior pattern matrix. Here, each column vector
                                                                                         describes the number of corresponding basic behavior patterns of
where N is the number of obtained basic behavior patterns.
                                                                                         all nodes relative to all other nodes. Because different basic behavior
   Then we feed the aggregated matrix Ã𝑙𝑜𝑐𝑎𝑙 ∈ R𝑛×𝑛 into the
                                                                                         patterns may have different contributions to the similarity when
graph convolution network for convolution operation. The depth
                                                                                         calculating the similarity of breadth behavior patterns, we use a set
of the basic behavior pattern captured by our aggregation module
                                                                                         of learnable weights 𝛽𝑖 to concatenate pattern-type-specific column
can increase with the increase of the number of convolution layers.
                                                                                         vectors to obtain the breadth behavior pattern matrix B ∈ R𝑛×N as:
Following MHGCN [57], our convolution also adopts the idea of
simplifying GCN, that is, no nonlinear activation function is used:                                                              ∑︁|
                                                                                                                                 |V

                         (1)                    (1)                                                                  B𝑝 (𝑖 ) =         Ā𝑝 (𝑖,𝑗 ) ,                 (5)
                        H𝑙𝑜𝑐𝑎𝑙 = Ã𝑙𝑜𝑐𝑎𝑙 · X · W𝑙𝑜𝑐𝑎𝑙 ,                      (2)                                                 𝑗=1
                                                                       (1)
where X ∈ R𝑛×𝑚 is the node attribute matrix and W𝑙𝑜𝑐𝑎𝑙 ∈ R𝑚×𝑑                                            B = 𝛽 1 · B1 ∥ 𝛽 2 · B2 ∥ · · · ∥ 𝛽 N · BN
is the learnable weights. Therefore, the single-layer GCN can ef-                                                                                                   (6)
                                                                                                              = (B1 ∥ · · · ∥ BN ) · Λ𝛽 ,
fectively learn the node representation that contains interaction
information in all basic behavior patterns.                                              where B𝑝 ∈ R𝑛×1 is the column vector corresponding to the 𝑝-
    To capture deeper behavior patterns, we can extend it to 𝑙-layer:                    th basic behavior pattern, ∥ denotes concatenation operation, and
                                                                                         Λ𝛽 = 𝑑𝑖𝑎𝑔(𝛽 1, 𝛽 2, · · · , 𝛽 N ) is the learnable diagonal matrix.
                 (𝑙 )                     (𝑙 −1)        (𝑙 )
               H𝑙𝑜𝑐𝑎𝑙 = Ã𝑙𝑜𝑐𝑎𝑙 · H𝑙𝑜𝑐𝑎𝑙 · W𝑙𝑜𝑐𝑎𝑙                                           Then we multiply the breadth behavior pattern matrix by its
                                                 (1)            (𝑙 )                     transpose and normalize it to obtain the breadth behavior pattern
                          = Ã𝑙𝑙𝑜𝑐𝑎𝑙 · X · W𝑙𝑜𝑐𝑎𝑙 · · · W𝑙𝑜𝑐𝑎𝑙 .             (3)
                                           |      {z        }                            similarity matrix:
                                                            𝑙                                           Ã𝑔𝑙𝑜𝑏𝑎𝑙 = 𝑛𝑜𝑟𝑚𝑎𝑙𝑖𝑧𝑒 (B · BT ) ∈ R𝑛×𝑛 .                     (7)
   As shown in Eq. (3), the depth of behavior patterns can be deter-
                                                                                         Intuitively, the more similar the breadth behavior pattern of two
mined by the number of convolution layers. Since depth behavior
                                                                                         nodes is, the greater their weight in the breadth behavior pattern
pattern aggregation aims to capture the local information of nodes
                                                                                         similarity matrix is.
in our model, the number of convolution layers is generally set to
                                                                                            We next input the breadth behavior pattern similarity matrix
be relatively small, e.g., two or three layers.
                                                                                         into the graph convolution network for information aggregation,
   To capture all multiplex interaction information in behavior
                                                                                         so as to obtain the node representation from the breadth behavior
patterns at different depths, we finally fuse outputs of all layers to
                                                                                         pattern similarities:
obtain the local node representation as:
                                                                                                       (𝑙 )                       (𝑙 −1)              (𝑙 )
                                     𝑙                                                               H𝑔𝑙𝑜𝑏𝑎𝑙 = Ã𝑔𝑙𝑜𝑏𝑎𝑙 · H𝑔𝑙𝑜𝑏𝑎𝑙 · W𝑔𝑙𝑜𝑏𝑎𝑙
                                 1 ∑︁ (𝑖 )
                        H𝑙𝑜𝑐𝑎𝑙 =      H      ∈ R𝑛×𝑑 .                        (4)                                    𝑙                    (1)                 (𝑙 )
                                 𝑙 𝑖=1 𝑙𝑜𝑐𝑎𝑙                                                                    = Ã𝑔𝑙𝑜𝑏𝑎𝑙 · X · W𝑔𝑙𝑜𝑏𝑎𝑙 · · · W𝑔𝑙𝑜𝑏𝑎𝑙 ,            (8)
                                                                                                                                 |       {z         }
4.3    Breadth Behavior Pattern Aggregation                                                                                                             𝑙
The breadth behavior pattern is used to represent the type and                                     (1)                (2)                      (𝑙 )
                                                                                         where W𝑔𝑙𝑜𝑏𝑎𝑙 ∈ R𝑚×𝑑 , W𝑔𝑙𝑜𝑏𝑎𝑙 ∈ R𝑑 ×𝑑 , · · · , and W𝑔𝑙𝑜𝑏𝑎𝑙                ∈
quantity of the basic behavior patterns of each node. It describes
the basic behavior patterns of nodes from the perspective of breadth,                    R𝑑 ×𝑑 are the learnable weight matrix.
                                                                                                                                               (𝑙 )
so we call it the breadth behavior pattern.                                                 We take the output of the last layer H𝑔𝑙𝑜𝑏𝑎𝑙 ∈ R𝑛×𝑑 as the global
   In fact, the breadth behavior pattern represents the interaction                      node representation H𝑔𝑙𝑜𝑏𝑎𝑙 .
behavior information of each node with all other nodes. Intuitively,                        Notice that, in the breadth behavior pattern aggregation, the
nodes with the same community structure behave more similarly.                           strength of the connection between the two nodes is determined




                                                                                   486
KDD ’23, August 6–10, 2023, Long Beach, CA, USA                                                     Chaofan Fu, Guanjie Zheng, Chao Huang, Yanwei Yu, and Junyu Dong


by the similarity of their breadth behavior patterns. Even if the two                   Table 1: Statistics of Datasets (n-type: node type, e-type: edge
nodes are far away in the network topology, if their breadth behav-                     type, feat.: features, and Mult.: Multiplex network)
ior pattern similarity is large, their learned global representation
vectors would be relatively correlated. Therefore, breadth behavior                      Dataset #nodes #edges #n-type #e-type #feat. #label Mult.
pattern aggregation is a process of information aggregation from a                        IMDB        11,616    17,106          3          2        19       3   ×
global perspective.                                                                     Alibaba-s     15,218    27,036          2          3       −−        5   ✓
                                                                                         Alibaba      22,649    45,734          2          3        18       5   ✓
4.4     Contrastive Learning                                                              DBLP        26,128    119,783         4          3      4,635      4   ×
Contrastive learning has demonstrated its superiority in various                         Taobao       21,318    41,676          2          4        19       3   ✓
graph learning tasks. Inspired by that, we propose to use contrastive                   Douban        20,000    304,829         2          6       −−       −−   ✓
learning between depth and breadth behavior pattern aggregations
to empower the representation learning ability of the model that
maximizes the agreement of node representations learned across                          where H𝑣 is the representation of node 𝑣, T denotes matrix transpo-
different views while capturing different relationships.                                sition, 𝜎 (·) is the sigmoid function, 𝑠 can be any vector similarity
   In the contrastive learning module, we regard the same nodes as                      measure function, Ω is the set of positive node pairs, Ω − is the set
positive samples and different nodes as negative samples in both                        of negative node pairs sampled from all unconnected node pairs.
depth and breadth behavior pattern aggregations. With the local                            Finally, we integrate unsupervised learning loss or semi-supervised
and global node representations H𝑙𝑜𝑐𝑎𝑙 and H𝑔𝑙𝑜𝑏𝑎𝑙 , we have the                        learning loss with our contrastive learning loss to optimize our
following cross-view contrastive loss with InfoNCE [31] as:                             model jointly:
                                                                                                                      L = L∗ + 𝛾 · L𝑐𝑙                           (13)
                  ∑︁                exp(𝑠 (H𝑙𝑜𝑐𝑎𝑙,𝑖 , H𝑔𝑙𝑜𝑏𝑎𝑙,𝑖 )/𝜏)
        L𝑐𝑙 = −         log Í                                               (9)         where L∗ is the unsupervised or semi-supervised learning loss
                  𝑖∈V             𝑗 ∈ V exp(𝑠 (H𝑙𝑜𝑐𝑎𝑙,𝑖 , H𝑔𝑙𝑜𝑏𝑎𝑙,𝑗 )/𝜏)                function L𝑢𝑠𝑙 or L𝑠𝑠𝑙 , and 𝛾 is the hyperparameter for tuning the
                                                                                        importance of contrastive learning.
where H∗,𝑖 is the local/global node representation of the 𝑖-th node,
                                                                                           The pseudo-code of our BPHGNN is shown in Algorithm 1 in
𝑠 (·, ·) denotes the cosine similarity function, and 𝜏 is the tunable
                                                                                        the supplement.
temperature hyperparameter to adjust the scale for softmax. This
contrastive learning allows the depth behavior pattern and breadth
behavior pattern views to collaboratively supervise each other,                         5 EXPERIMENT
which enhances the node representation learning.                                        5.1 Datasets
    Eventually, we use the outputs of both the depth behavior pat-                      Six publicly available real-world datasets are used in our experi-
tern aggregation and the breadth behavior pattern aggregation to                        mental evaluation, i.e., IMDB [62] dataset1 , Alibaba-s [52] dataset2 ,
obtain the final node representation H ∈ R𝑛×𝑑 through the average                       Alibaba [52] dataset3 , DBLP [57] dataset4 , Taobao [57] dataset5 ,
pooling operation for downstream tasks as:                                              Douban [26] dataset6 . The statistics of datasets are summarized in
                            1                                                           Table 1 and the detailed dataset description can be found in the
                       H = (H𝑙𝑜𝑐𝑎𝑙 + H𝑔𝑙𝑜𝑏𝑎𝑙 ).                  (10)
                            2                                                           supplement.
4.5     Model Learning                                                                  5.2     Baselines
In this section, we present the objective function to train our model
                                                                                        We compare our BPHGNN with the following fourteen baseline
to learn the final node representation. According to different down-
                                                                                        methods, which are divided into three categories: Homogeneous
stream tasks, we can train our model in two manners: unsupervised
                                                                                        network embedding methods include node2vec [14], SGC [49],
learning and semi-supervised learning.
                                                                                        and AM-GCN [48]; Heterogeneous network embedding methods
   For semi-supervised learning, we use back-propagation and gra-
                                                                                        contain MAGNN [13], GTN [58], Simple-HGN [30], HGTN [23],
dient descent to optimize the model parameters by minimizing the
                                                                                        and SR-RSC [62]; Multiplex Heterogeneous network embedding
following cross-entropy loss:
                               ∑︁                                                       methods include MNE [60], GATNE [3], DMGI [32], FAME [28],
                    L𝑠𝑠𝑙 = −        Y𝑖 ln(C · H𝑖 )               (11)                   DualHGNN [52], and MHGCN [57].
                                     𝑖 ∈ V𝑖𝑑𝑠                                              Since DualHGNN is designed only for multiplex bipartite net-
where V𝑖𝑑𝑠 is the set of node indices that have labels, Y𝑖 is the label                 works, it can only work on Alibaba-s, Alibaba, Taobao, and Douban
of the 𝑖-th node, and C is the node classifier parameter.                               networks. The detailed description of baselines can be found in A.4
   For unsupervised learning, we optimize the model parameters                          in the supplement. The source code of our paper is available at
by minimizing the following binary cross-entropy loss function                          https://github.com/FuChF/BPHGNN.
through negative sampling:
                           ∑︁                                                           1 https://github.com/RuixZh/SR-RSC
               L𝑢𝑠𝑙 = −         log 𝜎 (𝑠 (H𝑢T , H𝑣 ))                                   2 https://github.com/xuehansheng/DualHGCN
                                                                                        3 https://github.com/xuehansheng/DualHGCN
                              (𝑢,𝑣) ∈Ω
                                ∑︁                                         (12)         4 https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=0
                       −                      log 𝜎 (−𝑠 (H𝑢T′ , H𝑣 ′ )),                5 https://tianchi.aliyun.com/competition/entrance/231719/information/
                                                                                        6 https://github.com/7thsword/MFPR-Datasets
                           (𝑢 ′ ,𝑣 ′ ) ∈Ω −




                                                                                  487
Multiplex Heterogeneous Graph Neural Network with Behavior Pattern Modeling                                    KDD ’23, August 6–10, 2023, Long Beach, CA, USA


Table 2: Node classification performance comparison of different methods on five datasets. Marker * indicates the result is
statistically significant (t-test with p-value < 0.01)

                           IMDB                       Alibaba-s                     Alibaba                  DBLP                       Taobao
      Method
                     Macro-F1 Micro-F1           Macro-F1 Micro-F1            Macro-F1 Micro-F1        Macro-F1 Micro-F1          Macro-F1 Micro-F1
    node2vec           0.4935        0.5056        0.2357        0.2432        0.3098       0.3377       0.3535       0.3513        0.2547        0.3376
      MNE              0.5327        0.5472        0.4124        0.4484        0.3435       0.4298       0.5416       0.5271        0.2923        0.3957
     GATNE             0.5587        0.5625        0.2986        0.3125        0.3496       0.3859        OOT          OOT          0.2895        0.3984
     DMGI              0.5165        0.5142        0.2355        0.2567        0.3159       0.3232       0.7524       0.7633        0.3091        0.3963
     FAME              0.5564        0.5646        0.2412        0.2536        0.3597       0.4515       0.8364       0.8722        0.3128        0.4035
   DualHGNN               /             /          0.3400        0.3432        0.4338       0.4623          /            /          0.3155        0.4016
      SGC             0.6044        0.6072         0.2528        0.2674        0.3422       0.3830       0.6228      0.6234         0.2746        0.3743
    AM-GCN            0.5133        0.5121         0.4278        0.4755        0.3213       0.3362       0.8584      0.8626         0.2935        0.3443
    MAGNN             0.5942        0.5953         0.4056        0.4459        0.3448       0.3846       0.8813      0.9024         0.3048        0.3558
      GTN             0.5976        0.5865         0.4226        0.4577        0.3692       0.4573       0.8824      0.8945         0.2796        0.3861
  Simple-HGN          0.6131        0.6174         0.4247        0.4633        0.4076       0.4728       0.9083      0.9048         0.3015        0.3812
     SR-RSC           0.6119        0.6178         0.4134        0.4328        0.3623       0.4532       0.9027      0.9066         0.2948        0.3965
     HGTN             0.6193        0.6185         0.3916        0.4553        0.3810       0.4814       0.8245      0.8324         0.3198        0.4026
    MHGCN             0.6136        0.6197         0.4303        0.4774        0.4034       0.4967       0.9018      0.9145         0.3156        0.4136
   BPHGNN            0.6447*       0.6468*        0.4972*       0.5455*       0.4732*      0.5474*      0.9391*     0.9458*        0.3393*       0.4304*
  Improvement         4.10%         4.37%          15.55%        14.26%        9.08%        10.20%       4.03%       3.42%          6.19%         4.06%

5.3    Experimental Setting                                                         of Macro-F1 and Micro-F1 on DBLP dataset. On these datasets,
For all baselines, we use their released source code and the param-                 our proposed behavior pattern falls back to the decoupled model
eters recommended by their papers to ensure that their methods                      of MHGCN, however, our model is still significantly better than
achieve the desired effect. More detailed experimental settings can                 state-of-the-art models such as MHGCN. This is because our model
be found in A.5 in the supplement.                                                  adequately learns global relevant information on the similarity of
                                                                                    behavior patterns into node representations, which is very impor-
5.4    Node Classification                                                          tant for node classification but ignored by previous heterogeneous
                                                                                    network embedding approaches.
We first evaluate the effectiveness of our model compared with
state-of-the-art baselines on the semi-supervised node classification
task. The experimental results are shown in Table 2, where the best
                                                                                    5.5    Link Prediction
is shown in bold, and the second best is underlined. The first six                  We next evaluate the model performance by comparing our BPHGNN
baselines are unsupervised embedding methods, and the rest are                      with state-of-the-art baselines on unsupervised link prediction. The
semi-supervised embedding methods.                                                  results are reported in Table 3, where the best is shown in bold
    From the experimental results, we can observe that our BPHGNN                   and the second best is underlined. The first two baselines are ho-
significantly outperforms all baselines in terms of both Macro-F1                   mogeneous network methods, the second three are heterogeneous
and Micro-F1 on all evaluated heterogeneous networks in node                        network methods, and the last six are multiplex network methods.
classification. In particular, BPHGNN achieves average 7.80% and                       As we see, BPHGNN also achieves state-of-the-art performance
7.26% improvement over the best-performed baselines in terms of                     in terms of all metrics on all tested networks for link prediction. The
Macro-F1 and Micro-F1 across five datasets, respectively. Compared                  experimental results show that our BPHGNN achieves an average
with state-of-the-art baselines, our BPHGNN exhibits great supe-                    improvement of 5.92% and 5.54% in terms of ROC-AUC and PR-AUC
riority for node classification on three multiplex heterogeneous                    over state-of-the-art multiplex GNN MHGCN across six datasets, re-
networks, i.e., Alibaba-s, Alibaba, and Taobao. Especially on Alibaba-              spectively. Especially, our BPHGNN, FAME, and MHGCN realize a
s dataset, our model achieves average gains of 15.55% and 14.26%                    high accuracy of more than 99% prediction performance on Taobao
in terms of Macro-F1 and Micro-F1 in comparison to state-of-the-                    dataset. More specifically, on multiplex networks, BPHGNN out-
art GNN baseline MHGCN, respectively. The main reasons may                          performs state-of-the-art multiplex bipartite network embedding
include: first, our model effectively learns the effect of multiplex                method DualHGNN by average 4.56% and 4.07% improvement, re-
structure on node representation in AMHENs through behavior                         spectively. We believe that our BPHGNN leverages the defined basic
pattern modeling; second, our model can capture high-level seman-                   behavior pattern to more effectively learn the multiplex structural
tic information such as structural similarity in the network through                relationships between multi-typed nodes for multiplex heteroge-
breadth behavioral pattern aggregation. Additionally, we also see                   neous network embedding, compared to models based on relation-
that BPHGNN significantly outperforms state-of-the-art methods                      aware meta-path learning (e.g., FAME, MHGCN) and models based
on general heterogeneous networks with multi-typed nodes (i.e.,                     on hypergraphs (e.g., DualHGNN). On general heterogeneous net-
IMDB and DBLP), achieving 4.03% and 3.42% improvement in terms                      works (i.e., IMDB and DBLP), our BPHGNN model shows even more




                                                                              488
KDD ’23, August 6–10, 2023, Long Beach, CA, USA                                           Chaofan Fu, Guanjie Zheng, Chao Huang, Yanwei Yu, and Junyu Dong


Table 3: Link prediction performance comparison of different methods on six datasets. Marker * indicates the result is
statistically significant (t-test with p-value < 0.01)

                        IMDB                   Alibaba-s             Alibaba              DBLP         Taobao                           Douban
      Method
                    R-AUC PR-AUC            R-AUC PR-AUC         R-AUC PR-AUC         R-AUC PR-AUC R-AUC PR-AUC                     R-AUC PR-AUC
   node2vec         0.3982       0.3486      0.5034    0.5143    0.5741    0.5918      0.4486     0.4524     0.6142     0.5803      0.5027      0.6061
     SGC            0.6733       0.6408      0.6311    0.7023    0.8319    0.8272      0.5894     0.5916     0.6858     0.7078      0.7814      0.8123
   MAGNN            0.8529       0.8473      0.8592    0.8547    0.8732    0.8421      0.6955     0.7047     0.9654     0.9637      0.8052      0.8171
 Simple-HGN         0.8752       0.8521      0.8854    0.8467    0.8682    0.8418      0.7459     0.7561     0.9761     0.9748      0.8259      0.8319
   SR-RSC           0.8722       0.8504      0.8631    0.8583    0.8618    0.8469      0.7521     0.7634     0.9816     0.9855      0.8152      0.8313
    MNE             0.7521       0.7063      0.7774    0.7265    0.7698    0.7157      0.6572     0.6605     0.9446     0.9462      0.6335      0.6458
   GATNE            0.7364       0.8215      0.7092    0.8298    0.7442    0.8422       OOT        OOT       0.9815     0.9867       OOT         OOT
    DMGI            0.8325       0.8274      0.7594    0.7463    0.7328    0.7419      0.6105     0.6152     0.8574     0.7816      0.7688      0.7741
    FAME            0.8635       0.8593      0.6255    0.6194    0.6036    0.6538      0.6421     0.6505     0.9942     0.9913      0.7793      0.7612
  DualHGNN             /            /        0.8718    0.8327    0.8746    0.8548         /          /       0.9742     0.9771      0.8486      0.8517
   MHGCN            0.8924       0.8567      0.9024    0.8592    0.8786    0.8523      0.7182     0.7216     0.9942     0.9953      0.8272      0.8493
  BPHGNN 0.9223* 0.8982* 0.9364* 0.8941* 0.9053* 0.8650* 0.8427* 0.8418* 0.9955                                         0.9934     0.8920* 0.9017*
 Improvement 3.35% 4.53%  3.77%   4.06%   3.09%   1.19%   12.05% 10.27%   0.13%                                         -0.19%      5.11%   5.87%
    OOT: Out Of Time (36 hours). R-AUC: ROC-AUC.
Table 4: Ablation study on node classification (Ma: Macro-F1,                       Based on the results, we have the following three observations:
Mi: Micro-F1)                                                                       (1) After removing the basic behavior patterns, the performance
                                                                                of the model drops significantly on the three multiplex networks
             IMDB Alibaba-s Alibaba                    DBLP      Taobao         (i.e., w/o BBP vs. Full). This also demonstrates the contribution of
            Ma Mi Ma Mi Ma Mi                         Ma Mi     Ma Mi           the proposed BBP to the model performance improvement.
w/o BBP /       / 0.457 0.521 0.436 0.489 /         / 0.316 0.401                   (2) w/o Local performs the worst and is significantly worse than
w/o Glo 0.613 0.619 0.488 0.512 0.397 0.473 0.896 0.914 0.289 0.406             w/o Global variant. Depth behavior pattern aggregation is to aggre-
w/o Loc 0.177 0.363 0.113 0.257 0.127 0.270 0.136 0.142 0.232 0.406             gate local feature information, which is the intrinsic nature of node
w/o CL 0.633 0.634 0.491 0.508 0.431 0.492 0.920 0.926 0.325 0.404              representation and thus plays a decisive role in node classification.
  Full 0.645 0.646 0.497 0.545 0.473 0.547 0.939 0.946 0.339 0.430              Breadth behavior pattern aggregation can pass feature information
                                                                                between nodes based on the similarity of behavior patterns, effec-
significant superiority. To be specific, BPHGNN achieves gains of               tively supplementing global relevant information to facilitate node
12.05% ROC-AUC and 10.27% PR-AUC in comparison to state-of-the-                 classification. Both variants are worse than the Full method on all
art heterogeneous network embedding baseline SR-RSC on DBLP                     tested datasets, which verifies that both play an important role in
network. The underlying reason is that our model effectively learns             node representation learning of heterogeneous networks for node
global relevant information by exploiting breadth behavior pattern              classification.
similarity aggregation in the network, which greatly facilitates the                (3) w/o CL performs worse than the full model in all metrics
link prediction task.                                                           on all datasets. This demonstrates that contrastive learning can
                                                                                effectively adjust the depth behavior pattern aggregation and the
5.6     Ablation Study                                                          breadth behavior pattern aggregation to learn more effective node
                                                                                representation.
To evaluate the effectiveness of each component of our model, we
further conduct the ablation study on different BPHGNN variations.
                                                                                5.7    Effectiveness Study on Multiplex Networks
We report the results of the ablation study on five datasets for node
classification in Table 4. Specifically, we generate four variants:             To further verify the effectiveness of BPHGNN on multiplex net-
                                                                                works, we further conduct experiments for node classification and
   • w/o BBP - In this variant, we replace our basic behavior
                                                                                link prediction on two multiplex networks (i.e., Taobao and Douban)
      patterns with the decoupled adjacency matrices of sub-graphs
                                                                                compared to several strong multiplex network baselines varying
      used in MHGCN.
                                                                                the number of edge types. Specifically, we first choose to keep one
   • w/o Local - This variant removes the depth behavior pattern
                                                                                relation on Taobao (i.e., buy) and two relations on Douban (i.e.,
      aggregation module, that is, only breadth behavior pattern
                                                                                comment and rating), and perform node classification and link pre-
      aggregation is kept.
                                                                                diction tasks respectively. Since it is an important task to predict
   • w/o Global - This variant only keeps the depth behavior
                                                                                whether users will rate items on the Douban dataset, we only per-
      pattern aggregation module.
   • w/o CL - For this variant, we remove the contrastive learning,             form link prediction for rating-type edges on Douban dataset using
      and directly perform mean pooling for H𝑙𝑜𝑐𝑎𝑙 and H𝑔𝑙𝑜𝑏𝑎𝑙 to
      obtain the final node representation.




                                                                          489
Multiplex Heterogeneous Graph Neural Network with Behavior Pattern Modeling                                                   KDD ’23, August 6–10, 2023, Long Beach, CA, USA




                (a) Macro-F1 score #layers                                   (b) Macro-F1 score hyperparameter 𝛾                    (c) Macro-F1 score #rounds

                               Figure 4: Parameter sensitivity of proposed BPHGNN w.r.t. #layers, 𝛾, and #rounds.

  0.4                                        0.95                                                 datasets refers to the right-ordinate axis, and the performance on
               1 edge    2 edges                         2 edges   3 edges   4 edges
 0.35          3 edges   4 edges              0.9        5 edges   6 edges                        IMDB w.r.t. 𝛾 also refers to the right-ordinate axis.
  0.3                                        0.85                                                    Effect of Layers. As shown in Figure 4(a), firstly, the perfor-
 0.25                                         0.8                                                 mance of BPHGNN increases with the increasing number of layers.
  0.2                                        0.75                                                 When the number of layers reaches 2 to 3, the model performance
 0.15                                         0.7
        FAME   DualHGNN MHGCN BPHGNN                  FAME   DualHGNN MHGCN       BPHGNN
                                                                                                  begins to decline slowly. The model has a significant performance
                                                                                                  drop on DBLP. It should be emphasized that the purpose of this
    (a) Node classification on Taobao               (b) Link prediction on Douban
                                                                                                  work is not to solve the over-smoothing issue of GNNs, that is, even
Figure 5: Experiment results of multiplex effectiveness study                                     when the number of layers is set too small, e.g., 2, our model can
                                                                                                  also learn global relevant information to improve model perfor-
ROU-AUC as the evaluation metric. Then, we gradually increase
                                                                                                  mance. This solves the shortcoming of existing GNNs that cannot
the types of edges in networks and conduct the same experiment.
                                                                                                  achieve message passing between distant nodes.
The experimental results are shown in Figure 5. Notice that we
                                                                                                     Effect of hyperparameter 𝛾. From Figure 4(b), we can find that
gradually add other types of edges in the order of add-to-collect,
                                                                                                  our BPHGNN achieves the best performance on almost all datasets
add-to-cart, and click on Taobao dataset, and in order of wish, tag,
                                                                                                  when 𝛾 falls near 0.01. The model performance is improved on all
reading, and read on Douban dataset.
                                                                                                  datasets when 𝛾 increases from 0 to 0.01, indicating that contrastive
    As shown in Figure 5(a), BPHGNN performs much better than
                                                                                                  learning can effectively coordinate the node representation learning
the strong competitors in terms of Macro-F1 for node classification
                                                                                                  of the two aggregation modules in BPHGNN for improving node
in retaining the same type of edges on Taobao dataset. To be specific,
                                                                                                  classification performance. When 𝛾 is set too large, it will affect the
our model improves by 22.75% over state-of-the-art MHGCN when
                                                                                                  importance of the loss function of the specific task in the model
both buy and add-to-collect relations are preserved. As the types of
                                                                                                  learning, and thus damaging the model performance.
edges increase, the classification performance of our model also in-
                                                                                                     Effect of #rounds. As shown in Figure 4(c), for the node classi-
creases significantly. This also demonstrates that our BPHGNN can
                                                                                                  fication task, BPHGNN can achieve fast convergence. Especially on
effectively capture the multi-relational structures in the network
                                                                                                  Alibaba, Alibaba-s, IMDB, and Taobao datasets, after training for
to help improve classification performance.
                                                                                                  only 60 rounds, stable performance can be achieved.
    From Figure 5(b), it can be seen that our BPHGNN even outper-
forms other baseline methods keeping all relations when only two                                  6     CONCLUSION
relations (i.e., comment and rating) are retained in our model, and
the prediction performance of BPHGNN for rating edge continues                                    In this paper, we propose an embedding model BPHGNN for mul-
to increase as the number of edge types increase. The main reason                                 tiplex heterogeneous networks. BPHGNN mainly consists of four
is that our proposed basic behavior pattern effectively models the                                components: basic behavior pattern generator, depth behavior pat-
multiplex interaction structures in the multiplex Douban network,                                 tern aggregation, breadth behavior pattern aggregation, and con-
and learns the interaction effect between different relations on                                  trastive learning. BPHGNN can automatically capture both local
node representation through depth behavior pattern aggregation.                                   and global relevant information among nodes across multiplex
Additionally, compared with other multiple network embedding                                      structures through depth and breadth behavior pattern aggrega-
methods, the global relevant information learned by the breadth                                   tions, and can adaptively learn the importance of various behavior
behavior pattern aggregation also further improves the prediction                                 patterns for multiplex network embedding. Experiment results on
performance of our model.                                                                         six real-world heterogeneous networks show the superiority of the
                                                                                                  proposed BPHGNN in both node classification and link prediction.
5.8     Parameter Sensitivity
                                                                                                  ACKNOWLEDGMENTS
We finally evaluate the sensitivity of our BPHGNN with respect to
three main hyperparameters (i.e., the number of aggregation layers,                               This work is partially supported by the National Natural Science
parameter 𝛾, and the number of training rounds). The Macro-F1                                     Foundation of China under grant Nos. 62176243, 61773331 and
score on node classification with different settings on five datasets is                          41927805, and the National Key Research and Development Program
depicted in Figure 4. Notice that the performance on three multiplex                              of China under grant Nos. 2018AAA0100602 and 2019YFC1509100.




                                                                                            490
KDD ’23, August 6–10, 2023, Long Beach, CA, USA                                                                Chaofan Fu, Guanjie Zheng, Chao Huang, Yanwei Yu, and Junyu Dong


REFERENCES                                                                                             Discovery from Data (TKDD) (2022).
 [1] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori                      [24] Wen-Zhi Li, Ling Huang, Chang-Dong Wang, and Yu-Xin Ye. 2021. StarGAT: Star-
     Koyama. 2019. Optuna: A next-generation hyperparameter optimization frame-                        Shaped Hierarchical Graph Attentional Network for Heterogeneous Network
     work. In Proceedings of the 25th ACM SIGKDD international conference on knowl-                    Representation Learning. In IEEE International Conference on Data Mining (ICDM).
     edge discovery & data mining. 2623–2631.                                                          IEEE, 1198–1203.
 [2] Shaosheng Cao, Wei Lu, and Qiongkai Xu. 2015. Grarep: Learning graph rep-                    [25] Hu Linmei, Tianchi Yang, Chuan Shi, Houye Ji, and Xiaoli Li. 2019. Heteroge-
     resentations with global structural information. In Proceedings of the 24th ACM                   neous graph attention networks for semi-supervised short text classification. In
     international on conference on information and knowledge management. 891–900.                     Proceedings of the 2019 Conference on Empirical Methods in Natural Language Pro-
 [3] Yukuo Cen, Xu Zou, Jianwei Zhang, Hongxia Yang, Jingren Zhou, and Jie Tang.                       cessing and the 9th International Joint Conference on Natural Language Processing
     2019. Representation Learning for Attributed Multiplex Heterogeneous Network.                     (EMNLP-IJCNLP). 4821–4830.
     In Proceedings of the 25th ACM SIGKDD international conference on knowledge                  [26] Jian Liu, Chuan Shi, Binbin Hu, Shenghua Liu, and Philip S Yu. 2017. Personal-
     discovery & data mining. 1358–1368.                                                               ized ranking recommendation via integrating multiple feedbacks. In Pacific-Asia
 [4] Yukuo Cen, Xu Zou, Jianwei Zhang, Hongxia Yang, Jingren Zhou, and Jie Tang.                       Conference on Knowledge Discovery and Data Mining. Springer, 131–143.
     2019. Representation learning for attributed multiplex heterogeneous network.                [27] Weiyi Liu, Pin-Yu Chen, Sailung Yeung, Toyotaro Suzumura, and Lingli Chen.
     In Proceedings of the 25th ACM SIGKDD international conference on knowledge                       2017. Principled multilayer network embedding. In IEEE International Conference
     discovery & data mining. 1358–1368.                                                               on Data Mining Workshops (ICDMW). IEEE, 134–141.
 [5] Haochen Chen, Syed Fahad Sultan, Yingtao Tian, Muhao Chen, and Steven                        [28] Zhijun Liu, Chao Huang, Yanwei Yu, Baode Fan, and Junyu Dong. 2020. Fast
     Skiena. 2019. Fast and Accurate Network Embeddings via Very Sparse Random                         Attributed Multiplex Heterogeneous Network Embedding. In Proceedings of the
     Projection. In Proceedings of the 28th ACM international conference on information                29th ACM International Conference on Information & Knowledge Management.
     and knowledge management. 399–408.                                                                995–1004.
 [6] Hongxu Chen, Hongzhi Yin, Weiqing Wang, Hao Wang, Quoc Viet Hung Nguyen,                     [29] Yuanfu Lu, Chuan Shi, Linmei Hu, and Zhiyuan Liu. 2019. Relation structure-
     and Xue Li. 2018. PME: projected metric embedding on heterogeneous networks                       aware heterogeneous information network embedding. In Proceedings of the
     for link prediction. In Proceedings of the 24th ACM SIGKDD international confer-                  AAAI Conference on Artificial Intelligence. 4456–4463.
     ence on knowledge discovery & data mining. 1177–1186.                                        [30] Qingsong Lv, Ming Ding, Qiang Liu, Yuxiang Chen, Wenzheng Feng, Siming
 [7] Lei Chen, Le Wu, Richang Hong, Kun Zhang, and Meng Wang. 2020. Revisiting                         He, Chang Zhou, Jianguo Jiang, Yuxiao Dong, and Jie Tang. 2021. Are we really
     graph based collaborative filtering: A linear residual graph convolutional network                making much progress? revisiting, benchmarking and refining heterogeneous
     approach. In Proceedings of the AAAI conference on artificial intelligence, Vol. 34.              graph neural networks. In Proceedings of the 27th ACM SIGKDD international
     27–34.                                                                                            conference on knowledge discovery & data mining. 1150–1160.
 [8] Shaojie Dai, Jinshuai Wang, Chao Huang, Yanwei Yu, and Junyu Dong. 2021.                     [31] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning
     Temporal multi-view graph convolutional networks for citywide traffic volume                      with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018).
     inference. In IEEE International Conference on Data Mining (ICDM). IEEE, 1042–               [32] Chanyoung Park, Donghyun Kim, Jiawei Han, and Hwanjo Yu. 2020. Unsu-
     1047.                                                                                             pervised Attributed Multiplex Network Embedding. In Proceedings of the AAAI
 [9] Shaojie Dai, Jinshuai Wang, Chao Huang, Yanwei Yu, and Junyu Dong. 2023.                          Conference on Artificial Intelligence, Vol. 34. 5371–5378.
     Dynamic Multi-View Graph Neural Networks for Citywide Traffic Inference.                     [33] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: Online learning
     ACM Transactions on Knowledge Discovery from Data (TKDD) 17, 4 (2023), 1–22.                      of social representations. In Proceedings of the 20th ACM SIGKDD international
[10] Yuxiao Dong, Nitesh V Chawla, and Ananthram Swami. 2017. metapath2vec:                            conference on Knowledge discovery & data mining. 701–710.
     Scalable representation learning for heterogeneous networks. In Proceedings of               [34] Guangming Qin, Lexue Song, Yanwei Yu, Chao Huang, Wenzhe Jia, Yuan Cao,
     the 23rd ACM SIGKDD international conference on knowledge discovery & data                        and Junyu Dong. 2023. Graph Structure Learning on User Mobility Data for
     mining. 135–144.                                                                                  Social Relationship Inference. In Proceedings of the AAAI Conference on Artificial
[11] Haoyi Fan, Fengbin Zhang, Yuxuan Wei, Zuoyong Li, Changqing Zou, Yue Gao,                         Intelligence.
     and Qionghai Dai. 2021. Heterogeneous hypergraph variational autoencoder for                 [35] Jiezhong Qiu, Yuxiao Dong, Hao Ma, Jian Li, Chi Wang, Kuansan Wang, and Jie
     link prediction. IEEE Transactions on Pattern Analysis and Machine Intelligence                   Tang. 2019. Netsmf: Large-scale network embedding as sparse matrix factoriza-
     (2021).                                                                                           tion. In In Proceedings of the Web Conference. 1509–1520.
[12] Tao-yang Fu, Wang-Chien Lee, and Zhen Lei. 2017. Hin2vec: Explore meta-paths                 [36] Xuan Rao, Lisi Chen, Yong Liu, Shuo Shang, Bin Yao, and Peng Han. 2022. Graph-
     in heterogeneous information networks for representation learning. In Proceed-                    flashback network for next location recommendation. In Proceedings of the 28th
     ings of the 2017 ACM on Conference on Information and Knowledge Management.                       ACM SIGKDD international conference on knowledge discovery & data mining.
     1797–1806.                                                                                        1463–1471.
[13] Xinyu Fu, Jiani Zhang, Ziqiao Meng, and Irwin King. 2020. Magnn: Metapath                    [37] Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg, Ivan
     aggregated graph neural network for heterogeneous graph embedding. In In                          Titov, and Max Welling. 2018. Modeling relational data with graph convolu-
     Proceedings of the Web Conference. 2331–2341.                                                     tional networks. In The Semantic Web: 15th International Conference, ESWC 2018,
[14] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for                    Heraklion, Crete, Greece, June 3–7, 2018, Proceedings 15. Springer, 593–607.
     networks. In Proceedings of the 22nd ACM SIGKDD international conference on                  [38] Jingbo Shang, Meng Qu, Jialu Liu, Lance M Kaplan, Jiawei Han, and Jian Peng.
     Knowledge discovery & data mining. 855–864.                                                       2016. Meta-path guided embedding for similarity search in large-scale heteroge-
[15] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng                              neous information networks. arXiv preprint arXiv:1610.09769 (2016).
     Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for                 [39] Chuan Shi, Binbin Hu, Wayne Xin Zhao, and S Yu Philip. 2018. Heterogeneous
     recommendation. In Proceedings of the 43rd International ACM SIGIR conference                     information network embedding for recommendation. IEEE Transactions on
     on research and development in Information Retrieval. 639–648.                                    Knowledge and Data Engineering 31, 2 (2018), 357–370.
[16] Huiting Hong, Hantao Guo, Yucheng Lin, Xiaoqing Yang, Zang Li, and Jieping                   [40] Xiangguo Sun, Hongzhi Yin, Bo Liu, Hongxu Chen, Jiuxin Cao, Yingxia Shao,
     Ye. 2020. An attention-based graph neural network for heterogeneous structural                    and Nguyen Quoc Viet Hung. 2021. Heterogeneous hypergraph embedding for
     learning. In Proceedings of the AAAI conference on artificial intelligence, Vol. 34.              graph classification. In Proceedings of the 14th acm international conference on
     4132–4139.                                                                                        web search and data mining. 725–733.
[17] Binbin Hu, Yuan Fang, and Chuan Shi. 2019. Adversarial learning on heteroge-                 [41] Jian Tang, Meng Qu, and Qiaozhu Mei. 2015. Pte: Predictive text embedding
     neous information networks. In Proceedings of the 25th ACM SIGKDD international                   through large-scale heterogeneous text networks. In Proceedings of the 21th
     conference on knowledge discovery & data mining. 120–129.                                         ACM SIGKDD international conference on knowledge discovery & data mining.
[18] Ziniu Hu, Yuxiao Dong, Kuansan Wang, and Yizhou Sun. 2020. Heterogeneous                          1165–1174.
     graph transformer. In In Proceedings of the Web Conference. 2704–2710.                       [42] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei.
[19] Houye Ji, Xiao Wang, Chuan Shi, Bai Wang, and S Yu Philip. 2021. Heteroge-                        2015. Line: Large-scale information network embedding. In In Proceedings of the
     neous graph propagation network. IEEE Transactions on Knowledge and Data                          Web Conference. 1067–1077.
     Engineering 35, 1 (2021), 521–532.                                                           [43] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro
[20] Baoyu Jing, Chanyoung Park, and Hanghang Tong. 2021. Hdmi: High-order deep                        Liò, and Yoshua Bengio. 2018. Graph Attention Networks. In International Con-
     multiplex infomax. In In Proceedings of the Web Conference. 2414–2424.                            ference on Learning Representations.
[21] Diederik P Kingma and Max Welling. 2013. Auto-encoding variational bayes.                    [44] Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Yanfang Ye, Peng Cui, and Philip S
     arXiv preprint arXiv:1312.6114 (2013).                                                            Yu. 2019. Heterogeneous graph attention network. In In Proceedings of the Web
[22] Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with                         Conference. 2022–2032.
     Graph Convolutional Networks. In International Conference on Learning Repre-                 [45] Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Yanfang Ye, Peng Cui, and Philip S
     sentations.                                                                                       Yu. 2019. Heterogeneous Graph Attention Network. In In Proceedings of the Web
[23] Mengran Li, Yong Zhang, Xiaoyong Li, Yuchen Zhang, and Baocai Yin. 2022.                          Conference. ACM, 2022–2032.
     Hypergraph Transformer Neural Networks. ACM Transactions on Knowledge




                                                                                            491
Multiplex Heterogeneous Graph Neural Network with Behavior Pattern Modeling                                                            KDD ’23, August 6–10, 2023, Long Beach, CA, USA


[46] Xiao Wang, Nian Liu, Hui Han, and Chuan Shi. 2021. Self-supervised hetero-                     [56] Bing Yu, Haoteng Yin, and Zhanxing Zhu. 2018. Spatio-temporal graph convolu-
     geneous graph neural network with co-contrastive learning. In Proceedings of                        tional networks: a deep learning framework for traffic forecasting. In International
     the 27th ACM SIGKDD international conference on knowledge discovery & data                          Joint Conference on Artificial Intelligence. 3634–3640.
     mining. 1726–1736.                                                                             [57] Pengyang Yu, Chaofan Fu, Yanwei Yu, Chao Huang, Zhongying Zhao, and Junyu
[47] Xiao Wang, Yuanfu Lu, Chuan Shi, Ruijia Wang, Peng Cui, and Shuai Mou. 2020.                        Dong. 2022. Multiplex heterogeneous graph convolutional network. In Proceed-
     Dynamic heterogeneous information network embedding with meta-path based                            ings of the 28th ACM SIGKDD international conference on knowledge discovery &
     proximity. Transactions on Knowledge and Data Engineering (2020).                                   data mining. 2377–2387.
[48] Xiao Wang, Meiqi Zhu, Deyu Bo, Peng Cui, Chuan Shi, and Jian Pei. 2020. Am-                    [58] Seongjun Yun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, and Hyunwoo J Kim.
     gcn: Adaptive multi-channel graph convolutional networks. In Proceedings of                         2019. Graph transformer networks. Advances in neural information processing
     the 26th ACM SIGKDD International conference on knowledge discovery & data                          systems 32.
     mining. 1243–1253.                                                                             [59] Chuxu Zhang, Dongjin Song, Chao Huang, Ananthram Swami, and Nitesh V
[49] Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, et al. 2019. Sim-                  Chawla. 2019. Heterogeneous graph neural network. In Proceedings of the 25th
     plifying Graph Convolutional Networks. In International conference on machine                       ACM SIGKDD international conference on knowledge discovery & data mining.
     learning. 6861–6871.                                                                                793–803.
[50] Lianghao Xia, Chao Huang, Yong Xu, Peng Dai, Bo Zhang, and Liefeng Bo.                         [60] Hongming Zhang, Liwei Qiu, Lingling Yi, and Yangqiu Song. 2018. Scalable
     2020. Multiplex behavioral relation learning for recommendation via memory                          multiplex network embedding. In International Joint Conference on Artificial
     augmented transformer network. In Proceedings of the 43rd International ACM                         Intelligence, Vol. 18. 3082–3088.
     SIGIR Conference on Research and Development in Information Retrieval. 2397–                   [61] Jiani Zhang, Xingjian Shi, Shenglin Zhao, and Irwin King. 2019. Star-gcn: Stacked
     2406.                                                                                               and reconstructed graph convolutional networks for recommender systems. arXiv
[51] Lianghao Xia, Chao Huang, Yong Xu, Jiashu Zhao, Dawei Yin, and Jimmy Huang.                         preprint arXiv:1905.13129 (2019).
     2022. Hypergraph contrastive collaborative filtering. In Proceedings of the 45th               [62] Rui Zhang, Arthur Zimek, and Peter Schneider-Kamp. 2022. A simple meta-
     International ACM SIGIR conference on research and development in information                       path-free framework for heterogeneous network embedding. In Proceedings of
     retrieval. 70–79.                                                                                   the 31st ACM International Conference on Information & Knowledge Management.
[52] Hansheng Xue, Luwei Yang, Vaibhav Rajan, Wen Jiang, Yi Wei, and Yu Lin. 2021.                       2600–2609.
     Multiplex bipartite network embedding using dual hypergraph convolutional                      [63] Ruochi Zhang, Yuesong Zou, and Jian Ma. 2019. Hyper-SAGNN: a self-attention
     networks. In In Proceedings of the Web Conference. 1649–1660.                                       based graph neural network for hypergraphs. arXiv preprint arXiv:1911.02613
[53] Liang Yao, Chengsheng Mao, and Yuan Luo. 2019. Graph convolutional net-                             (2019).
     works for text classification. In Proceedings of the AAAI conference on artificial             [64] Weifeng Zhang, Jingwen Mao, Yi Cao, and Congfu Xu. 2020. Multiplex graph
     intelligence, Vol. 33. 7370–7377.                                                                   neural networks for multi-behavior recommendation. In Proceedings of the 29th
[54] Zhihao Ye, Gongyao Jiang, Ye Liu, Zhiyong Li, and Jin Yuan. 2020. Document                          ACM International Conference on Information & Knowledge Management. 2313–
     and word representations generated by graph convolutional network and bert                          2316.
     for short text classification. In European Conference on Artificial Intelligence 2020.         [65] Ziwei Zhang, Peng Cui, Haoyang Li, Xiao Wang, and Wenwu Zhu. 2018. Billion-
     IOS Press, 2275–2281.                                                                               Scale Network Embedding with Iterative Random Projection. In IEEE International
[55] Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L Hamilton,                          Conference on Data Mining (ICDM). 787–796.
     and Jure Leskovec. 2018. Graph convolutional neural networks for web-scale                     [66] Jianan Zhao, Xiao Wang, Chuan Shi, Binbin Hu, Guojie Song, and Yanfang Ye.
     recommender systems. In Proceedings of the 24th ACM SIGKDD international                            2021. Heterogeneous graph structure learning for graph neural networks. In
     conference on knowledge discovery & data mining. 974–983.                                           Proceedings of the AAAI conference on artificial intelligence, Vol. 35. 4697–4705.




                                                                                              492
KDD ’23, August 6–10, 2023, Long Beach, CA, USA                                           Chaofan Fu, Guanjie Zheng, Chao Huang, Yanwei Yu, and Junyu Dong


A SUPPLEMENT                                                                   A.3     Detailed Dataset Description
A.1 Notations                                                                  IMDB dataset contains three types of nodes, movie, actor, and
                                                                               director. We use the label of movies as the class label in node classi-
Key notations used in the paper and their definitions are summa-
                                                                               fication. Alibaba-s dataset includes three types of edges between
rized in Table 5.
                                                                               user and item nodes, i.e., click, inquiry, and contact. We use the
         Table 5: Main notations and their definitions.                        category labels of items in node classification. Alibaba dataset also
                                                                               includes three types of edges between user and item nodes, which
   Notation                                 Definition                         is a larger dataset than Alibaba-s, including more nodes and edges.
          G                             the input network                      DBLP dataset contains four types of nodes, i.e., author, paper, term,
        V, E                         the node/edge set of G                    and venue. We use the authors’ label to complete the node classifi-
        O, R                      the node/edge type set of G                  cation task. Taobao dataset includes four types of edges between
          X                     the node attribute matrix of G                 user and item nodes (i.e., buy, click, add-to-collect, and add-to-cart).
         G𝑟                       the sub-network edge type 𝑟                  We use the category labels of items in node classification. Douban
         A𝑟                        the adjacency matrix of G𝑟                  dataset includes six types of edges between user and item nodes.
         Ā𝑝                the adjacency matrix of BBP type 𝑝                 Since this dataset does not contain label information, we only use
       𝛼 𝑝 , 𝛽𝑝             the learnable weight for BBP type 𝑝                it for link prediction task.
       Ã𝑙𝑜𝑐𝑎𝑙                the aggregated adjacency matrix
          B                 the breadth behavior pattern matrix                A.4     Baselines
     Ã𝑔𝑙𝑜𝑏𝑎𝑙           breadth behavior pattern similarity matrix             The public source codes of baselines can be available at the follow-
  (𝑙 )         (𝑙 )
 W𝑙𝑜𝑐𝑎𝑙 , W𝑔𝑙𝑜𝑏𝑎𝑙      the learnable weight matrix for the 𝑙-th layer          ing URLs:
                                                                                   • node2vec – https://github.com/aditya-grover/node2vec
 H𝑙𝑜𝑐𝑎𝑙 , H𝑔𝑙𝑜𝑏𝑎𝑙           the hidden representation for nodes
                                                                                   • SGC – https://github.com/Tiiiger/SGC
          H                           the node embeddings
                                                                                   • AM-GCN – https://github.com/zhumeiqiBUPT/AM-GCN
          𝑑                      the dimension of embeddings
                                                                                   • MAGNN – https://github.com/cynricfu/MAGNN
        𝑛, 𝑚                   the number of nodes/attributes
                                                                                   • Simple-HGN – https://github.com/THUDM/HGB
          N                            the number of BBPs
                                                                                   • GTN – https://github.com/seongjunyun/Graph_Transformer_
                                                                                     Networks
                                                                                   • HGTN – https://github.com/limengran98/HGTN
A.2     Algorithm Pseudo-Code                                                      • SR-RSC – https://github.com/RuixZh/SR-RSC
Algorithm 1 shows the pseudo-code of our proposed BPHGNN.                          • MNE – https://github.com/HKUST-KnowComp/MNE
                                                                                   • GATNE – https://github.com/THUDM/GATNE
Algorithm 1 The Learning Process of BPHGNN                                         • DMGI – https://github.com/pcy1302/DMGI
Input: Input G, node feature matrix X, embedding dimension 𝑑,                      • FAME – https://github.com/ZhijunLiu95/FAME
     the number of convolution layers 𝑙                                            • DualHGNN – https://github.com/xuehansheng/DualHGCN
Output: Node embeddings H                                                          • MHGCN – https://github.com/NSSSJSS/MHGCN
  1: Decouple the attributed multiplex heterogeneous network into
     homogeneous networks and bipartite networks to obtain the                 A.5     Detailed Experimental Settings
     adjacency matrices {A𝑟 |𝑟 = 1, 2, . . . , |R|}                            For the link prediction task, we treat the connected nodes in net-
  2: Generate basic behavior pattern matrix Ā𝑝                                works as positive node pairs, and consider all unlinked nodes as
                                                                               negative node pairs. For each edge type, we divide the positive
                            ÍN
  3: Calculate Ã𝑙𝑜𝑐𝑎𝑙 = 𝑝=1         𝛼𝑝 Ā𝑝
  4: for 𝑖 = 1 to 𝑙 do                                                         node pairs into the training set, verification set, and test set ac-
                       (𝑖 )                   (𝑖 −1)    (𝑖 )                   cording to the proportion of 50%, 25%, and 25%. At the same time,
  5:   Calculate H𝑙𝑜𝑐𝑎𝑙 ← Ã𝑙𝑜𝑐𝑎𝑙 · H𝑙𝑜𝑐𝑎𝑙 · W𝑙𝑜𝑐𝑎𝑙
  6: end for                                                                   we randomly select the same number of negative node pairs to
               1     (1)                (𝑙 )                                   add to the training set, validation set, and test set. Notice that we
  7: H𝑙𝑜𝑐𝑎𝑙 = (H
               𝑙    𝑙𝑜𝑐𝑎𝑙
                            + · · · + H𝑙𝑜𝑐𝑎𝑙 )
                                                                               predict each type of edge using all types of edges in datasets, and
  8: Calculate breadth behavior pattern matrix Ã𝑔𝑙𝑜𝑏𝑎𝑙 using Eq. (5),
                                                                               finally take the average of all edges as the final result. For the node
     Eq. (6) and Eq. (7)
                                                                               classification task, we take 20% of the nodes as the training set,
  9: for 𝑖 = 1 to 𝑙 do
                       (𝑖 )                      (𝑖 −1)      (𝑖 )              40% as the validation set, and 40% as the test set. Notice that we
 10:   Calculate H𝑔𝑙𝑜𝑏𝑎𝑙 ← Ã𝑔𝑙𝑜𝑏𝑎𝑙 · H𝑔𝑙𝑜𝑏𝑎𝑙 · W𝑔𝑙𝑜𝑏𝑎𝑙
                                                                               repeat each experiment 10 times to report average results. For a
 11: end for
                  (𝑙 )                                                         fair comparison, we uniformly set the number of training rounds
 12: H𝑔𝑙𝑜𝑏𝑎𝑙 = H
                 𝑔𝑙𝑜𝑏𝑎𝑙                                                        to 500 for link prediction and the number of training rounds to 200
        1
13: H = 2 (H𝑙𝑜𝑐𝑎𝑙 + H𝑔𝑙𝑜𝑏𝑎𝑙 )                                                  for node classification. Following [28, 57], we set the embedding
14: Calculate L using Eq. (13);                                                dimension 𝑑 of all methods to 200.
15: Back propagation and update parameters in BPHGNN                              For our BPHGNN, we set the number of aggregation layers to 2,
16: Return H                                                                   and 𝜏 to 0.1. For node classification task, we set 𝛾 to 0.01, and we set




                                                                         493
Multiplex Heterogeneous Graph Neural Network with Behavior Pattern Modeling                                                     KDD ’23, August 6–10, 2023, Long Beach, CA, USA


𝛾 to 0.0001 for link prediction task. Following [57], we set 𝑝 = 2 and                              Alibaba-s dataset, which demonstrates that contrastive learning
𝑞 = 0.5 for node2vec and set 𝛼𝑟 and 𝛽𝑟 to 1 for every edge type 𝑟                                   can also effectively align two aggregations to obtain better node
on GATNE. For FAME, we perform Optuna [1] to tune the weights                                       representation for link prediction.
𝛼 1, . . . , 𝛼𝐾 , 𝛽 1, . . . , 𝛽 | R | as described in the original paper. We tune
the learning rate in {0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005} for all                             A.7        Time and Memory Complexity Analysis
deep learning methods(e.g., GATNE, GTN). For GTN, we use the                                        BPHGNN mainly consists of three computing modules: depth be-
sparse version of their released source code and set GT layers to 3                                 havior pattern aggregation, breadth behavior pattern aggregation,
for all datasets. For DMGI, we set the self-connection weight 𝑤 = 3                                 and contrastive learning.
and tune 𝛼, 𝛽, 𝛾 in {0.0001, 0.001, 0.01, 0.1}. For AM-GCN, we tune                                    The time complexity of aggregating all basic behavior patterns is
loss aggregation parameters 𝛽, 𝛾 in {0.0001, 0.001, 0.01, 0.1}. For                                 𝑂 (N𝑛 2 ), and the time complexity of graph convolution is 𝑂 (𝑛 2𝑑𝑙 +
MAGNN, we set the number of independent attention mechanisms                                        𝑛𝑚𝑑 + 𝑛𝑑 2 (𝑙 − 1)), so the total time complexity of depth behavior
𝑘 = 4. For DualHGNN, we use the asymmetric operator and set 𝜆 as                                    pattern aggregation is 𝑂 (𝑛 2 (N+𝑑𝑙) +𝑛𝑚𝑑 +𝑛𝑑 2 (𝑙 −1)). The breadth
0.5. For SR-RSC, we use its semi-supervised version to perform the                                  behavior aggregation module first calculates the breadth behavior
node classification task. For HGTN, we set the number of channels                                   pattern similarity matrix, and the time complexity is 𝑂 (𝑛 2 N). The
to 2 and the number of attention heads to 8, other parameters use                                   time complexity of the graph convolution is also 𝑂 (𝑛 2𝑑𝑙 + 𝑛𝑚𝑑 +
the default parameters of the paper. For MHGCN, we set the number                                   𝑛𝑑 2 (𝑙 − 1)), and thus the overall time complexity is 𝑂 (𝑛 2 (N + 𝑑𝑙) +
of convolution layers 𝑙 to 2.                                                                       𝑛𝑚𝑑 +𝑛𝑑 2 (𝑙 −1)). Lastly, the time complexity of contrastive learning
                                                                                                    is 𝑂 (𝑛 2𝑑). Therefore, the total time complexity of our BPHGNN is
A.6       Ablation Study on Link Prediction                                                         𝑂 ((N + 𝑑𝑙)𝑛 2 + (𝑚 + 𝑑 2 (𝑙 − 1))𝑛).
                                                                                                       The input of depth behavior pattern aggregation includes matri-
                                                                                                    ces composed of various BBPs and attribute matrix 𝑋 ∈ R𝑛×𝑚 , and
    1        w/o BBP      w/o Local      1       1                                      0.8
                                                            w/o BBP      w/o Local                                                                      (1)
 0.95
             w/o Global
             Full
                          w/o CL
                                               0.95
                                                            w/o Global   w/o CL
                                                                                        0.7
                                                                                                    the weight vector and matrices contain 𝛼 1:N , 𝑊𝑙𝑜𝑐𝑎𝑙 ∈ R𝑚×𝑑 and
                                         0.8                Full
                                                                                                      (𝑙 )
  0.9                                           0.9                                     0.6         𝑊𝑙𝑜𝑐𝑎𝑙 ∈ R𝑑 ×𝑑 , and thus the space complexity of depth behavior
                                         0.6
 0.85                                          0.85                                     0.5         pattern aggregation is 𝑂 (𝑛 2 N+𝑛𝑚+𝑚𝑑 +𝑑 2 (𝑙 −1)+N). Similarly, the
  0.8                                    0.4    0.8                                     0.4         input of the breadth behavior pattern aggregation includes matrix
        Alibaba-s   Alibaba     Douban                 Alibaba-s   Alibaba     Douban
                                                                                                    B ∈ R𝑛×N and adjacency matrix 𝐴˜𝑔𝑙𝑜𝑏𝑎𝑙 ∈ R𝑛×𝑛 , and the weight
    (a) ROC-AUC on link prediction                    (b) PR-AUC on link prediction                                                         (1)                       (𝑙 )
                                                                                                    vector and matrices contain 𝛽 1:N , 𝑊𝑔𝑙𝑜𝑏𝑎𝑙 ∈ R𝑚×𝑑 and 𝑊𝑔𝑙𝑜𝑏𝑎𝑙 ∈
             Figure 6: Ablation study on link prediction                                            R𝑑 ×𝑑 , and thus the space complexity of breath behavior pattern ag-
                                                                                                    gregation is 𝑂 (𝑛N+𝑛 2 +𝑚𝑑 +𝑑 2 (𝑙 −1)+N). Therefore, the total space
   We also conduct an ablation study for the link prediction task on                                complexity of BPHGNN is 𝑂 (N𝑛 2 + (𝑚 + N)𝑛 + 𝑚𝑑 + 𝑑 2 (𝑙 − 1) + N).
three multiplex heterogeneous networks (i.e., Alibaba-s, Alibaba,                                      Furthermore, we conduct the efficiency evaluation of our model
and Douban). The results are shown in Figure 6, where the perfor-                                   compared with existing multiplex heterogeneous network embed-
mance of w/o Local refers to the right-ordinate axis, while others                                  ding baselines. The time consumption (seconds) of each epoch
refer to the left-ordinate axis.                                                                    for each method is reported in Table 6. From the experimental re-
   It can be seen from Figure 6 that after replacing the basic be-                                  sults, we can see that the time consumption of our BPHGNN is
havior pattern, w/o BBP variant is significantly worse than the                                     much lower than those of DualHGNN and GATNE. In particular,
full model on the three multiplex datasets. Specifically, w/o BBP                                   on Taobao dataset, compared with DualHGNN, the efficiency of
reduces performance by 2.22%, 3.08%, and 5.31% in terms of ROC-                                     our model has increased by 215%. BPHGNN achieves an efficiency
AUC on the three datasets, respectively. This also verifies that the                                close to that of the state-of-the-art baseline MHGCN, about 0.76
proposed basic behavior pattern modeling effectively improves the                                   times lower than MHGCN. This is consistent with the expected
performance of the model on the link prediction task.                                               time complexity, since our BPHGNN model integrates aggregate
   Similarly, w/o Local variant is far worse than the other variants.                               learning of both local and global information, while MHGCN only
Compared with the full model, w/o Local drops the performance by                                    includes the aggregation of local information.
more than 30% in both ROC-AUC and PR-AUC across three datasets.
This also demonstrates that depth behavior pattern aggregation,                                     Table 6: Runtime of multiplex heterogeneous network em-
that is, low-order local information, is the fundamental basis for                                  bedding methods (Second)
the link prediction task. w/o Global variant is also significantly
worse than the full method on the three datasets, which indicates                                            Method   IMDB Alibaba-s Alibaba DBLP Taobao
that the breath behavior pattern aggregation, that is, multi-relation
structural similarity, also plays a positive role in promoting link                                      GATNE         640        840          1320      1800       1440
prediction performance.                                                                                DualHGNN         /         13.5         20.3        /         56.4
   The comparison of w/o CL with the full model reflects the contri-                                    MHGCN          3.4         5.4          8.8      16.5       10.24
bution of the contrastive learning module to the model. In particular,                                  BPHGNN         5.2         8.4         14.4      38.9        17.9
w/o CL lowers the performance by 3.48% in terms of PR-AUC on




                                                                                              494

