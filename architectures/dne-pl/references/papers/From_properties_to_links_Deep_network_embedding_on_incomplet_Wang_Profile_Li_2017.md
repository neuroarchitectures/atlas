# From properties to links Deep network embedding on incomplet Wang Profile Li 2017

> Source: `From_properties_to_links_Deep_network_embedding_on_incomplet_Wang_Profile_Li_2017.pdf`

---

See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/320885185



From Properties to Links: Deep Network Embedding on Incomplete Graphs

Conference Paper · November 2017
DOI: 10.1145/3132847.3132975




CITATIONS                                                                                                 READS
12                                                                                                        1,052


5 authors, including:

            Senzhang Wang                                                                                            Chaozhuo Li
            Nanjing University of Aeronautics & Astronautics                                                         Beihang University (BUAA)
            73 PUBLICATIONS 435 CITATIONS                                                                            16 PUBLICATIONS 40 CITATIONS

               SEE PROFILE                                                                                                SEE PROFILE



            Xiaoming Zhang
            Beihang University (BUAA)
            62 PUBLICATIONS 300 CITATIONS

               SEE PROFILE




 All content following this page was uploaded by Senzhang Wang on 14 November 2017.

 The user has requested enhancement of the downloaded file.
Session 2D: Network Embedding 2                                                                                             CIKM’17, November 6-10, 2017, Singapore




             From Properties to Links: Deep Network Embedding on
                               Incomplete Graphs
                      Dejian Yang                                               Senzhang Wang                                         Chaozhuo Li∗
                 Beihang University                                  Nanjing University of Aeronautics                              Beihang University
              dejianyang@buaa.edu.cn                                         and Astronautics                                    lichaozhuo@buaa.edu.cn
                                                                           szwang@nuaa.edu.cn

                                                Xiaoming Zhang                                                  Zhoujun Li
                                                Beihang University                                          Beihang University
                                                yolixs@buaa.edu.cn                                           lizj@buaa.edu.cn

ABSTRACT                                                                                            CCS CONCEPTS
As an effective way of learning node representations in networks,                                   • Information systems → Data mining; • Computing method-
network embedding has attracted increasing research interests re-                                   ologies → Dimensionality reduction and manifold learning;
cently. Most existing approaches use shallow models and only work
on static networks by extracting local or global topology infor-                                    KEYWORDS
mation of each node as the algorithm input. It is challenging for                                   Network Embedding, Incomplete Graph, Deep Learning
such approaches to learn a desirable node representation on incom-
plete graphs with a large number of missing links or on dynamic
graphs with new nodes joining in. It is even challenging for them                                   1   INTRODUCTION
to deeply fuse other types of data such as node properties into the                                 Networks are ubiquitous nowadays, such as social networks (Twit-
learning process to help better represent the nodes with insuffi-                                   ter, Flickr), paper citation networks (DBLP, Google Scholar), knowl-
cient links. In this paper, we for the first time study the problem of                              edge graphs (Freebase, Wikipedia) and communication networks
network embedding on incomplete networks. We propose a Multi-                                       (Email). Mining valuable knowledge from networks can facilitate
View Correlation-learning based Deep Network Embedding method                                       many real applications in practice. For example, clustering the users
named MVC-DNE to incorporate both the network structure and                                         in social networks into different communities can help advertisers
the node properties for more effectively and efficiently perform net-                               better perform online advertisement targeting [3]. A fundamental
work embedding on incomplete networks. Specifically, we consider                                    problem in network mining is how to learn a desirable representa-
the topology structure of the network and the node properties as                                    tion vector for each node [5]. As an effective way to embed each
two correlated views. The insight is that the learned representation                                node into a low-dimensional continuous feature vector, network
vector of a node should reflect its characteristics in both views.                                  embedding has been extensively studied recently. The learned high-
Under a multi-view correlation learning based deep autoencoder                                      quality node representation vectors are fundamental to perform
framework, the structure view and property view embeddings are                                      many data mining and machine learning tasks, such as classification
integrated and mutually reinforced through both self-view and                                       [22], social recommendation [8] and link prediction [16].
cross-view learning. As MVC-DNE can learn a representation map-                                        Most existing network embedding models such as DeepWalk
ping function, it can directly generate the representation vectors for                              [19], LINE [23], GraRep [5] and Node2Vec [10] are shallow mod-
the new nodes without retraining the model. Thus it is especially                                   els. They adopt a well-designed objective function to capture the
more efficient than previous methods. Empirically, we evaluate                                      topology information of the input network, while the node rep-
MVC-DNE over three real network datasets on two data mining ap-                                     resentation vectors are considered as parameters that need to be
plications, and the results demonstrate that MVC-DNE significantly                                  learned in the objective function. Such network structure based
outperforms state-of-the-art methods.                                                               shallow models purely utilize the network topology information
                                                                                                    and encode the local and global structures of the nodes into their
                                                                                                    representation vectors. To achieve better embedding performance
∗ Corresponding author: Chaozhuo Li(lichaozhuo@buaa.edu.cn)
                                                                                                    on the networks with sparse links, some recent works [13, 14, 28, 29]
                                                                                                    try to incorporate node properties and community structures as
                                                                                                    auxiliary information into the learning of the network embedding.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed                  Although remarkable efforts have been made on node represen-
for profit or commercial advantage and that copies bear this notice and the full citation           tation learning on static complete networks, how to effectively and
on the first page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
                                                                                                    efficiently perform network embedding on incomplete networks
to post on servers or to redistribute to lists, requires prior specific permission and/or a         is still not well studied. Typically, parts of the links among the
fee. Request permissions from permissions@acm.org.                                                  nodes in an incomplete graph can be missing, leading to the chal-
CIKM’17, , November 6–10, 2017, Singapore.
                                                                                                    lenge for existing approaches which purely or largely rely on the
© 2017 Association for Computing Machinery.
ACM ISBN 978-1-4503-4918-5/17/11. . . $15.00                                                        topology information of the networks to learn a promising network
https://doi.org/10.1145/3132847.3132975                                                             embedding result. Meanwhile, a more challenging case is when




                                                                                              367
Session 2D: Network Embedding 2                                                                            CIKM’17, November 6-10, 2017, Singapore




                                                                                     To address above mentioned problem, in this paper we propose
                                                                                  a deep network embedding framework on incomplete graphs with
                                                                                  only partial links available, aiming to more efficiently and effec-
                                                                                  tively learn representation vectors by considering both the network
                                                                                  topology and node properties. We view the topology structure and
                                                                                  the node properties as two correlated modalities of the network,
                                                                                  and propose a Multi-View Correlation learning based Deep Net-
                                                                                  work Embedding method, named MVC-DNE. The insight is that
                                                                                  the learned representation vector of a node should reflect its char-
                                                                                  acteristics in both views (as shown in the lower part of Figure 1).
                                                                                  Specifically, we first utilize the deep autoencoder to obtain the latent
                                                                                  representations in each single view. Note that the representations
                                                                                  are calculated by deep autoencoders, and thus it can be considered
                                                                                  as a map function. Then we design two frameworks to learn the
                                                                                  correlations in multiple views in the encoding stage and decoding
                                                                                  stage, respectively. Besides self-view learning implemented by the
                                                                                  deep autoencoder in each single view, our method considers the
                                                                                  features of one view as labels to guide the encoding process of
                                                                                  another view. The self-view and cross-view learnings are able to
                                                                                  capture the high level associations between the two views.
                                                                                     We summarize the main contributions of this paper as follows:
Figure 1: An illustration of multi-view data in the initial                            • We for the first time study the novel problem of network
spaces in social networks and their corresponding represen-                               embedding on incomplete graphs. We introduce node prop-
tations in the multi-view embedding space.                                                erties and propose a deep network embedding framework
                                                                                          MVC-DNE to address the challenge of sparsity in the net-
                                                                                          work structure view.
the graph is dynamic with new nodes joining in. Previous models                        • We design two frameworks to learn the correlations of node
face with the challenges of efficiency and effectiveness in learning                      representations in the two views. The consistency captured
representations for such new nodes with very limited or even no                           by cross-view learning from one view is able to complete or
links. To learn the representation vectors for new nodes, tradition                       refine the features of another view. By learning a represen-
models need to retrain the model which is very time consuming. For                        tation vector mapping function, our models are much more
example, when dealing with the new nodes, the popular DeepWalk                            efficient to generate representations for new nodes without
algorithm needs to regenerate the entire nodes sequences as the                           needing to retrain the model.
new input, and then retrains the Skip-Gram model [19].                                 • Extensively, we evaluate MVC-DNE on three real network
    In this paper, we for the first time study the problem of network                     datasets through multi-class classification and link predic-
embedding on incomplete networks. Given a node in a network,                              tion. The results show the superior performance of MVC-
besides the links between the node to others, it can be also asso-                        DNE by comparison with state-of-the-art baseline methods.
ciated with a set of node properties such as the user profiles of a               The rest of this paper is organized as follows. Section 2 summarizes
user in social networks (as shown in the upper part of Figure 1) or               the related works. Then we formally define the studied problem
the metadata of a paper in citation networks. Such node properties                in Section 3. Section 4 introduces the proposed multi-view deep
are highly correlated to links in terms of node similarity, and thus              network embedding framework in details. Section 5 presents the
they are important complementary information to the structural                    experimental results. Finally we conclude this work in section 6.
topology in network embedding. For example, as shown in Figure
1, two connected users are more likely to have similar user profiles              2 RELATED WORK
than two users that are far away from each other topologically due
to the characteristic of homophily [1, 7, 17]. Motivated by this idea,            2.1 Network Embedding
we propose to introduce node properties into network embedding                    Network Embedding aims to project each node in a network into
on incomplete graphs. The node properties potentially encode dif-                 a distributed representation. Most earlier works on this problem
ferent types of but highly correlated information to the network                  [2, 4, 11, 15, 20, 25] represent the network as an affinity matrix
topology, and integrate them into a unified learning framework                    and then extract the leading eigenvectors as the representations
is expected to achieve a better performance. Although previous                    of nodes by using matrix factorization techniques. For example,
work [29] tried to utilize texts associated with each node to im-                 IsoMAP [25] uses the feature vectors of the nodes to construct an
prove the performance of network embedding, that method used                      affinity graph, and then represent the nodes of the network with
the two types of data in a shallow way. It is difficult for [29] to learn         the solved leading eigenvectors.
the highly nonlinear correlations between the texts and links by                     Recently, DeepWalk [19], a online learning method, utilizes ran-
considering the two types of data separately and combining them                   dom walks to transform the structural information into a set of
shallowly.                                                                        node sequences. Then skip-gram is adopted to learn the network




                                                                            368
Session 2D: Network Embedding 2                                                                                   CIKM’17, November 6-10, 2017, Singapore




representations like word embedding. It has been proven to be                     vi ∈ V in both cases, our task is to learn such a distributed represen-
effective for preserving the second-order proximity of the nodes                  tation yi ∈ R2d = f (xti , xpi ) where d ≪ |V | is the output dimension
in the network. LINE [23] is the first to use first-order proximity               of each view, xti and xpi represent the input vector of two views
and second-order proximity, which preserves both the local and                    respectively.
global structure information of network. Correspondingly, [23]
takes the direct linking relations which represent the first order                   In our problem, T and P represent the information of a network
proximity into account and proposes a faster algorithm to perform                 from different views. Most existing methods adopt a well-designed
network embedding. To overcome the limitation of prior works in                   objective function to perform network embedding only with the
failing to offer flexibility in sampling nodes from a network [19, 23],           topology structure preserved. In an incomplete network, the links
Node2Vec [10] designs a much more flexible objective that is not                  or properties of nodes may be missing. It is challenging for previous
tied to a particular sampling strategy and provides parameters to                 methods to learn a satisfactory representations in such a case. To
tune the explored search space. In addition, [26] adopt max-margin                address this problem, one solution is to integrate the two views and
principle based on DeepWalk and incorporate labeling information                  deeply mine their latent correlations in the shared embedding space.
into vertex representations.                                                      To this aim, next we will introduce how to integrate the topology
   However, the above discussed methods can all be regarded as                    structure view and the node property view under a multi-view
shallow models, and they are not effective to capture the highly                  deep correlation learning framework to better performa network
nonlinear structural information of the networks. To address this                 embedding on incomplete networks.
problem, SDNE [27] is the first deep model which utilizes the modi-
fied deep atuoencoder. SDNE can preserve both the first and second
                                                                                  4     DEEP NETWORK EMBEDDING VIA
order proximity in the network with nonlinear structure relations                       MULTI-VIEW CORRELATION LEARNING
captured by the sigmoid functions and the multiple layers. How-                   In this section, we will first introduce how to adopt deep autoen-
ever, SDNE can not effectively handle multi-view data such as node                coders to perform network embedding in the network structure
properties. TADW [29] for the first time incorporates the citation                view. Based on it, we will next propose two learning frameworks,
relations and the rich text information of paper to learn a text-                 called MVC-DNEDB and MVC-DNEEB to perform network embed-
associated representations for each node in the citation network.                 ding on incomplete graphs by incorporating node properties and
As TADW is based on the matrix factorization technique, it is very                learning the correlations from the two views.
time consuming to process large networks. In addition, it is difficult
for TADW to achieve a satisfactory performance when the network                   4.1    AutoEncoder-based Deep Network
structure is sparse or when the text information is insufficient due                     Embedding Model in a Single View
the interdependence between topology structure and the texts.
                                                                                  The traditional deep autoencoder has been proven to be effective
                                                                                  in dimension reduction [21]. SDNE is the first work that utilizes
3    PROBLEM DEFINITION                                                           autoencoder based deep network embedding model and achieves
In this section, we will give a formal definition of the studied prob-            promising performance on static and complete graphs [27]. Thus
lem. We first give some notations as follows. Network G with each                 here we propose to utilize the deep autoencoder to perform network
node associated with a set of properties is defined as G =(V ,T , P),             embedding in a single view of network. The model, named SV-DNE,
where V = {v 1 , v 2 , . . . , v |V | } denotes the nodes in the network.         maps the input data of one view to distributed representations
T ∈ R |V |×|V | represents the adjacency matrix and P ∈ R |V |× |P |              through the encoder part, and then use the decoder to reconstruct
is the property matrix of nodes. Here |V | and |P | represent the                 the input data. Formally, given the input data x, the representations
dimensions of the adjacency matrix and the node properties, re-                   in each hidden layer are defined as follows:
spectively. Note that we define the dimension of the node properties                           y (1) = σ (W (1) · x + b (1) )                                            (1)
as m i=1 ni , where m is the size of the property categories and ni
    Í
                                                                                                   (k)            (k )        (k −1)        (k )
is the number of possible discrete values of the ith property. With                            y         = σ (W          ·y            +b          ), k = 2, . . . , K   (2)
such a definition, the element value of the node property vector is               where K denotes the number of layers, W (k ) and b (k ) denote the k-
in {0, 1} such that it can be processed by our proposed deep model.               th layer weight matrix and biases respectively. We adopt the sigmoid
With the above defined notations, we formally define the studied                  activation function to capture the complex relations of features, i.e.
problem as follows.                                                               the highly nonlinear structure in view T . When y (K ) is obtained,
                                                                                  we can reverse the above process to calculate the reconstructed
   Definition 3.1. Network Embedding on Incomplete Graphs                         input data x̂. By minimizing the reconstruction error between x
with Multi-View Data. Given a network G = (V ,T , P), with T                      and x̂, the node can be represented by a d-dimensional vector y (K )
denoting the network structure view data, and P denoting the node                 with the information in the corresponding view preserved. The loss
property view data as shown in Figure 1, network embedding aims to                function of network embedding in a single view is as follows.
learn a function f , which maps the node of both view features into
a continuous low-dimensional vector in the following two cases: 1)                                                              Õ|
                                                                                                                                |V

the network G is incomplete with only partial links in the T view                                            L(x; θ ) =                ∥xˆi − x i ∥22                    (3)
available; 2) the network G is dynamic with new nodes joining in,                                                               i=1
and the links of the new nodes are very sparse. That is for each node             where θ = {W (k ) , b (k ) , W ˆ(k ) , b (k
                                                                                                                           ˆ ) }K denotes all the parameters.
                                                                                                                                k =1




                                                                            369
Session 2D: Network Embedding 2                                                                             CIKM’17, November 6-10, 2017, Singapore




   However, a major problem of this method is that all features are
treated equally during the reconstruction process. Study in [27]
has shown that paying more attention to the non-zero elements
of the feature vector can learn a better representation as the input
feature vector x has far more zero elements. By taking this idea
into consideration, the loss function is redefined as follows.


                                Õ|
                                |V
                   L(x; θ ) =         ∥(xˆi − x i ) ⊙ w i ∥22         (4)
                                i=1
                                                                                                           (a) Decoding-based MVC-DNE

where w i is a weight vector with the same dimension as x, and each
element w i, j = β > 1 if x i, j > 0, otherwise w i, j = 1. Note that β
varies for different views. Therefore, this model is able to represent
the nodes with data information denoted by the non-zero elements
fully preserved, i.e. the topology structure or the nodes properties.
   As a typical unsupervised learning method, the above SV-DNE
utilizes the non-zero elements to supervise the learning process. We
call this feature encoding self-view learning. The hidden vectors in
the Kth layer are the output node representation in a single view.


4.2    AutoEncoder-based Deep Network
       Embedding in Multiple Views
Based on the above proposed network embedding with deep au-
toencoder framework in a single view, we next will introduce how
to construct a deep network embedding model with multi-view
data by incorporating node properties and network structure. A                                           (b) Encoding-based MVC-DNE
straightforward method is to build two subnetworks SV-DNET on
view T and SV-DNEP on view P respectively, and then concatenate                    Figure 2: Two implementations of MVC-DNE: MVC-DNEDB
yt (K ) (the output representation in view T ) and y (K ) (the output rep-         and MVC-DNEEB . The purple arrows indicate self-view
resentation in view P) as the final 2d-dimensional representations.                learning, and the red arrows indicate the cross-view learn-
However, this direct feature concatenation of SV-DNET and SV-                      ing, respectively.
DNEP is not capable of capturing the correlations between two
views. This combination does not work well when one view data
is missing. Besides, the correlations is of the essence to handle the
sparse and incomplete networks, such as new nodes embedding.
To address this problem, we propose the following two models.                                 L DB (xt, xp; θ ) =Lt (xt, xp; θ ) + Lp (xt, xp; θ )       (5)
    MVC-DNEDB . As the features in the network structure view T                                                     Õ|
                                                                                                                    |V
and the node property view P can be highly correlated, the input                                Lt (xt, xp; θ ) =         ((1 − α)∥xti − xtˆit ∥22
features of one view can encode some shared latent information                                                      i=1
reflected by the input of the other view. Inspired by this idea, we                                                             ˆp
                                                                                                                    + α ∥xti − xti ∥22 )                 (6)
propose a framework named MVC-DNEDB which learns the cross-
view correlations by the decoding operation. This framework aims                                                    Õ|
                                                                                                                    |V
                                                                                                                                          ˆp
to mine the latent information of the other view and make the                                   Lp (xt, xp; θ ) =         ((1 − α)∥xpi − xpi ∥22
                                                                                                                    i=1
learned representations in each view unified in both views. As
shown in Figure 2(a), the Kth-layer representation yt (K ) (yp (K ) ) not                                           + α ∥xpi − xpˆ it ∥22 )              (7)
only reconstructs the input of view T (view P) denoted as self-view                                  ˆp
decoding, but also reconstructs the input of another view denoted                    Here xtˆit and xti denote the reconstruction outputs of xti de-
                                                                                                                                   (K )           (K )
as cross-view decoding. Thus the hidden vectors(yt (K ) and yp (K ) )              coded by the hidden representations yti and ypi respectively,
is able to represent the information of both views. The red arrows                                                                 ˆp
                                                                                   and one can also get the meanings of xpˆ t and xp similarly. α is a
                                                                                                                                    i         i
in Figure 2(a) denote the cross-view learning. Specifically, MVC-                  parameter to balance the importance of the self-view and cross-view
DNEDB utilizes xt (xp) to guide and refine the encoding process                    reconstruct errors.
in view P (T ). Therefore, we sum the reconstruction errors in both                   By minimizing the above four reconstruction errors, the inputs
views as the final objective function:                                             of both views are projected into a consistent latent semantic space.




                                                                             370
Session 2D: Network Embedding 2                                                                                      CIKM’17, November 6-10, 2017, Singapore




Thus the representation in the hidden space of each view not only                           we can represent the hidden vectors yt (K ) or yp (K ) as the learned
preserves the features of its own view but also encodes the infor-                          features. For example, in link prediction, yt (K ) may achieve bet-
mation from the other view.                                                                 ter performance than the concatenated vectors because structural
    MVC-DNEEB . Another possible solution is to refine or modify                            information may play a more important role.
the node characteristics reflected in one view data by bring the in-
formation from the other view data. Following this idea, we propose
a new model called MVC-DNEEB . Different from MVC-DNEDB that                                4.3    Analysis and Discussions
learns the correlations of the two views in the decoding operation,                         Some analysis and discussions on MVC-DNEDB and MVC-DNEEB
MVC-DNEEB tries to learn the correlations in the encoding opera-                            are presented as follows:
tion. Specifically, as illustrated in Figure 2(b), MVC-DNEEB adopts                            Embedding New Nodes. Many real-world networks are dy-
four types of encoders to represent the inputs of the two views.                            namic and it is practically important to quickly obtain reasonable
These four encoders are classified into the self-view encoding (pur-                        representations for the new nodes with very limited or even no
ple arrow in Figure 2) and cross-view encoding (red arrow in Figure                         links. Compared to SNDE [27] that cannot effectively handle such
2). The self-view encoding aims to preserve the features of its own                         new nodes, our models utilize the properties of nodes and learn the
view while the cross-view encoding aims to integrate the shared                             correlations between node properties and links, and thus obtain
latent information of the other view. When obtaining the outputs                            satisfactory representations for the new nodes.
   (K )  (K )    (K )     (K )                                                                 A significant advantage of our modes is that our models adopt
ytt , ytp , ypp , ypt of the four encoders, the final represen-
                                                                                            deep autoencoders to implement the map function f from the input
tations yt (K ) in view T and yp (K ) in view P can be calculated as
                                                                                            to the representations in both views, denoted as the concatenation
follows.
                                                                                            of yt (K ) and yp (K ) specifically. Thus, given the adjacency vector
                                                                                            xtnew and property vector xpnew of a new node, the representation
                          ˜(K )     (K )  ˜(K )   (K )
              yt (K ) = ytt + zt (ytp , ytt ) ⊙ ytp                             (8)         vector can be directly generated by the representation mapping
                         ˜(K )           ˜(K )                                              function ynew = f (xtnew , xpnew ) in time O(1).
                                   (K )          (K )
             yp (K ) = ypp + zp (ypt , ypp ) ⊙ ypt                              (9)            MVC-DNEDB vs MV-DNEEB . Although MVC-DNEDB and MV-
           ˜ )     ˜(K )                                                                    DNEEB are based on the similar idea, they conduct the cross-view
           (K
   where ytt and ypp denote the outputs of the drop-out oper-                               correlation learning in different learning stages. MVC-DNEDB per-
           (K )        (K )
ation of ytt and ypp , which only works in the training step. To                            forms the cross-view learning in the decoding stage, and utilizes the
guide the cross-view encoding and avoid overfitting, we randomly                            input features of one view to guide the encoding of another view.
drop out 50% of the latent representations. zt (zp ) is a weight vector,                    The representation learning of each view relies on decoding the
                                                              (K )     (K )                 inputs of both views. MV-DNEEB performs the cross-view learning
which determines how much information of ytp                         (ypt ) flowing
                                                                                            in the encoding stage, and implements the cross-view represen-
into the output representation yt (K ) (yp (K ) ).                                          tation learning of one view through selecting the features of the
                              (X )             (C)
                                                                                            other view that are highly correlated to the dropped out features of
           z j (x, C) = σ (Wj        · x + Wj        · C + b j ), j = t, p     (10)         the view by the filter gate. The final output of MVC-DNEDB is a
   Note that the combination of the two representations are cal-                            unified representation vector which fuses both view information;
culated by a filter gate, which has been proven its significant per-                        while the output of MV-DNEEB are two separate representation
formance in information fusion [9]. In the decoding operation, the                          vectors with each one associated with one view.
latent representations yt (K ) and yp (K ) are used to reconstruct the
initial input data xt and xp. We define the loss function of MVC-
                                                                                            5     EXPERIMENTS
DNEEB as follows.
                                                                                            In this section, we will evaluate the performance of our proposed
                                                                                            models on three real network datasets through two tasks, multi-
            L EB (xt, xp; θ ) = Lt (xt, xp; θ ) + Lp (xt, xp; θ )              (11)         class classification and link prediction. We will first introduce the
                                  |V |                                                      used datasets and the baseline methods, and then show and analyze
                                         (∥xti − xtˆit ∥22
                                  Õ
              Lt (xt, xp; θ ) =                                                (12)         the results under different experiment settings. Finally, we will
                                  i=1                                                       conduct parameter sensitivity analysis.
                                  |V |                                                         We focus on evaluating the performance of the proposed models
                                                  ˆp
                                  Õ
              Lp (xt, xp; θ ) =          (∥xpi − xpi ∥22                       (13)         from the following three aspects:
                                  i=1
   To minimize the reconstruction errors, the dropped representa-                               (1) whether the proposed models can effectively integrate multi-
tion of one view can only be learned from the other view, and thus                                  view data and outperforms previous shallow network embed-
the information of one view is enhanced and complemented by the                                     ding models on complete networks;
information of the other view. This is especially helpful when one                              (2) whether the proposed models can learn a promising network
view information like the network structure is sparse while the                                     embedding result on incomplete networks;
other view information like node properties is sufficient. Finally, we                          (3) whether the proposed models can efficiently and effectively
concatenate yt (K ) and yp (K ) as the final multi-view representations                             generate representations for news nodes without retraining the
in MVC-DNEDB and MVC-DNEEB . Note that for a particular task,                                       model.




                                                                                      371
Session 2D: Network Embedding 2                                                                                         CIKM’17, November 6-10, 2017, Singapore




Table 1: Statistics of the three datasets. (avдt denotes the                                        Table 2: Layer structures of the model on
average number of links per node and avдp denotes the av-                                           the three dataset
erage number of properties per node.)
                                                                                                       Dataset       layers in view |V |   layers in view |P |
  Dataset         |V |         |E |          |P |   avдt     avдp    Categories                       Citeseer           3,312-100             3,703-100
  Citeseer       3,312       4,732       3,703       2.85    31.75       6                            Google+         15,819-300-100           4,211-100
  Google+       15,819     4,369,682     4,211      140.52   2.01        5                             DBLP           35,750-500-100         6,295-200-100
   DBLP         35,750      143,848      6,295       8.05     6.38       8

                                                                                              • SDNE [27] is the first deep network embedding method
5.1      Datasets                                                                               proposed recently. It uses the deep autoencoder to capture
Our datasets include two academic paper citation networks Cite-                                 the no-linear relations and also designs an explicit objective
seer1 [22] and DBLP2 [24], and one social network dataset Google+3                              function to preserve the local and global structure. SDNE
[12]. Table 1 shows the statistics of the three datasets after data                             can only handle the network structure information.
preprocessing. We briefly introduce the datasets as follows.                                  • SV-DNEP is a variation of our proposed method, which
                                                                                                takes the property information of nodes as the input of SV-
      • Citeseer is a paper citation network with around 3 thousand
                                                                                                DNE to obtain the node representations. It only utilizes the
        nodes. Each node represents a paper and each link indicates
                                                                                                property information of the nodes.
        a citation relationship between two papers. The key words,
                                                                                              • SDNE+SV-DNEP is a naive combination of SDNE and SV-
        abstract, and title are the property data of the paper. We use
                                                                                                DNEP . It concatenates the vectors learned by the two meth-
        a binary adjacent vector of 3,312 dimensions to represent the
                                                                                                ods as the output of node representations.
        structure features. The property features of each node are
        described by a binary vector of 3, 703 dimensions indicating
        the presence of the corresponding word. These papers are
                                                                                        5.3      Parameter Settings
        divided into 6 classes based on their topics or research fields.                In SDNE and our models, the deep autoencoders are used and the
      • Google+ is a social network, in which nodes refer to users                      number of layers varies with the scale of networks. The details of
        and the edges denote the friend relationships among users.                      neural network structure are shown in Table 2. The output dimen-
        The user profiles, i.e. gender, institution, university and work-               sion d is set as 100 in each view. The parameter βT denoting the
        place, are the properties of the users. The job title is extracted              β in view T and βT denoting the β in view P are tuned from 1 to
        as the job label of the user and we remove the nodes with no                    9, and we find 7 for βT and 5 for β P are reasonable settings after
        job title. Then we select the top 10 popular labels and cluster                 validation. We set the pre-training learning rate in SGD as 0.1 and
        them into 5 classes as the final job category.                                  fine-tuning learning rate in Adam as 0.001. The mini-batch size of
      • DBLP contains 35,750 papers from 8 research domains. We                         optimization is fixed to 32.
        choose the tile as the properties of each node. The nodes with                     In models LINE, DeepWalk and TADW, we follow the parameter
        no links are removed. Similar to Citeseer, we use a binary                      settings in their papers, and set the negative samples as 5, the total
        vector of 6,295 dimensions indicating the presence of the                       training samples as 10 billion and the initial learning rate as 0.025,
        words in the property view to represent each node.                              window size as 10, walk length as 40, walks per node as 40. The
                                                                                        hyper-parameter parameter α in SDNE is fixed to 0.5.
5.2      Baseline Methods
We compare our models with the following baselines. The first four
                                                                                        5.4      Experiment Results on Complete Networks
baselines are state-of-the-art network embedding methods, and the                       In this section, we evaluate the performance of various models on
other two are the variants of our proposed models.                                      complete graphs by conducting experiments on multi-class classifi-
                                                                                        cation and link prediction tasks.
      • DeepWalk [19] is a shallow model. It combines random
                                                                                            Performance evaluation on multi-class classification. We
        walks and skip-gram to learn network representations, which
                                                                                        use the learned node representations as the input features of clas-
        ensures the nodes with similar topology context are closer
                                                                                        sification model. The LinearSVM package implemented by scikit-
        to each other in the projected latent space.
                                                                                        learn4 [18] is adopted to train the classifier. We randomly sample
      • LINE [23] is a also shallow network embedding model with
                                                                                        a portion of labeled nodes as the training data. Then we use the
        the first order and second order proximity preserved. In this
                                                                                        remainder nodes as the testing data. The proportion of training
        paper, we use LINE1st and LINE2nd as two baseline methods
                                                                                        samples, denoted as Tr , varies from 10% to 50% for Citeseer and
        respectively.
                                                                                        Google+ datasets, and from 1% to 5% for DBLP dataset. We run the
      • TADW [29] is the first shallow model incorporates rich text
                                                                                        experiment 10 times and average the results. The average accuracy
        information of the network and adopts matrix factorization
                                                                                        is reported in Table 3.
        technique to obtain the node representations.
                                                                                            From Table 3 one can see that the performances of MVC-DNEDB
1 http://linqs.cs.umd.edu/projects/projects/lbc/index.html                              and MVC-DNEEB achieve similar performance, and both methods
2 https://cn.aminer.org/billboard/citation
3 https://snap.stanford.edu/data/index.html                                             4 http://scikit-learn.org/




                                                                                  372
Session 2D: Network Embedding 2                                                                                 CIKM’17, November 6-10, 2017, Singapore




                              Table 3: Multi-class classification results (Accuracy) on the three datasets.

        Model                            Citeseer                                       Google+                                       DBLP
         Tr             10%     20%        30%      40%       50%       10%     20%       30%     40%     50%       1%        2%        3%        4%         5%
     DeepWalk        0.309     0.339      0.354     0.363    0.367     0.546   0.551     0.554    0.552   0.554    0.619    0.646     0.660     0.667       0.674
      LINE1st        0.460     0.507      0.525     0.538    0.547     0.476   0.488     0.492    0.493   0.497    0.588    0.612     0.624     0.630       0.635
      LINE2nd        0.459     0.494      0.512     0.523    0.526     0.479   0.481     0.482    0.482   0.484    0.593    0.601     0.606     0.608       0.611
       TADW          0.661     0.691      0.703     0.711    0.719     0.405   0.426     0.438    0.444   0.450    0.682    0.704     0.715     0.722       0.728
       SDNE          0.515     0.544      0.556     0.566    0.574     0.569   0.575     0.576    0.577   0.579    0.659    0.678     0.691     0.699       0.705
     SV-DNEP         0.668     0.684      0.695     0.697    0.703     0.367   0.373     0.376    0.378   0.379    0.704    0.719     0.728     0.732       0.735
   SDNE+SV-DNEP      0.685     0.708      0.718     0.724    0.730     0.580   0.587     0.593    0.595   0.597    0.738    0.768     0.777     0.785       0.789
    MVC-DNED B       0.722     0.733      0.740     0.743    0.744     0.598   0.601     0.604    0.605   0.607    0.788    0.808     0.812     0.817       0.821
    MVC-DNEE B       0.726     0.739      0.746     0.752    0.755     0.587   0.594     0.600    0.602   0.603    0.770    0.788     0.797     0.802       0.805




                              (a) Sparsity Sensitivity of Network Structure                                  (b) Sparsity Sensitivity of Nodes Properties


      Figure 3: Embedding performance for original nodes w.r.t the proportion of removed links and nodes properties.

Table 4: Link prediction results (AUC) on the three datasets                          especially when the structure information is sparse. TADW does
                                                                                      not achieve satisfactory performance comparing to deep methods.
                Model          Citeseer     Google+       DBLP                        This indicates its performance largely depends on sufficient nodes
                                                                                      properties. Due to the sparsity of network structure, TADW and
             DeepWalk            0.606        0.768       0.889
              LINE1s t           0.663        0.804       0.901                       SV-NDEP are comparable on Citeseer and DBLP datasets. The im-
              LINE2nd            0.655        0.733       0.715                       provement on Google+ is less significant. This is mainly because
               TADW              0.923        0.522       0.843                       Google+ is a much denser network with sufficient connections
               SDNE              0.693        0.852       0.923                       among nodes.
             SV-DNEP             0.884        0.463       0.825                          Performance evaluation on link prediction. To conduct link
           SDNE+SV-DNEP          0.905        0.731       0.945                       prediction, we randomly removed 50% edges from the original net-
            MVC-DNED B           0.934        0.884       0.960                       work and consider them as the links for prediction. Thus, the re-
            MVC-DNEE B           0.941        0.917       0.938                       moved edges in the above mentioned operation are considered as
                                                                                      positive samples. We then randomly select the same number of un-
                                                                                      connected node pairs as negative links. The positive and negative
outperform the baselines in all the cases. The proposed two meth-                     node pairs form a balanced dataset of this task. We first compute
ods significantly outperform DeepWalk, LINE and SDNE. One can                         the representation vectors of the original nodes with various em-
also see that MVC-DNEDB and MVC-DNEEB are both consistently                           bedding methods. Then we utilize the representation vectors to
better than SDNE+SV-DNEP , demonstrating that compared to a                           calculate the cosine similarity of each node pair. Area Under Curve
naive combination, our models can much more effectively integrate                     (AUC) [6] is used as the performance evaluation metric, which can
the complementary information of different views. On dataset Cite-                    find the optimal threshold to predict positive and negative links
seer and DBLP, SV-DNEP outperforms DeepWalk and Line, which
shows that the properties of nodes are essential to mine the network




                                                                               373
Session 2D: Network Embedding 2                                                                        CIKM’17, November 6-10, 2017, Singapore




automatically. Thus AUC well reflects the correspondence of link               incomplete. Thus we randomly remove a proportion of proper-
labels and similarity scores.                                                  ties of each node to build an incomplete network with missing
    Table 4 shows the experiment results. From this table, one can             nodes properties. After obtaining the representation vector of each
observe that our proposed two models are consistently better than              node, we conduct multi-class classification to evaluate the perfor-
the baseline methods. On Google+ dataset, LINE1st and SNDE out-                mance. In this experiment, we do not compare with those baselines
perform TADW and SDNE+SV-DNEP as the node properties in                        that can only handle topology structure information. The result is
Google+ is rather sparse and noisy. Bringing in the noisy and sparse           reported in Figure 3(b). One can see that the accuracies of all meth-
nodes properties may not be that helpful to network embedding                  ods decrease quickly with the increase of proportion of the node
learning. MVC-DNEDB and MVC-DNEEB still work well for ef-                      properties. MVC-DNEDB and MV-DNEEB consistently outperform
fectively utilizing the view of the topology structure as the major            baseline methods SDNE and SV-DNEP as their accuracy curves are
information, and considering the node property view as comple-                 always above the curves of the other two methods.
mentary information.
    MVC-DNEEB beats the best baseline SDNE+SV-DNEP on AUC                      5.6    Experiment Results for New Nodes
in link prediction by around 10% on average on the all three datasets,
demonstrating its power in deeply fusing data of the two views. The
                                                                                      Embedding
less desirable performances of MVC-DNEDB on Google+ demon-                     New nodes can be considered as an extreme case for incomplete net-
strate that the node properties do not help much in learning a                 works. Previous models cannot effectively deal with the new nodes
better node representation as the node properties in Google+ is                except for retraining the model, which is very time-consuming. In
rather sparse and noisy. MVC-DNEEB performs much better than                   this section, we will first introduce how to prepare dataset, and then
MVC-DNEDB , because it learns a better node representation by                  show the experiment results. Finally, we will explore and analyze
more effectively utilizing the view of the topology structure as               the sensitivity of the models to the sparsity of the new nodes con-
the major information, and only taking the node property view as               nections. Note that the baseline methods for new nodes embedding
complementary information.                                                     are SDNE, SV-DNEP and SDNE+SV-DNEP .
                                                                                  Data partition for new nodes. To construct news nodes for
                                                                               a network, we split each dataset into two parts by 9:1. 90% of the
5.5    Experiment Results on Incomplete                                        nodes are used as the original nodes and the remainder 10% nodes
                                                                               are considered as new nodes. Therefore, the adjacent matrix of the
       Network
                                                                               90% original nodes are used as the structural input of models. By
In this subsection, we conduct experiments to evaluate the perfor-             removing the 10% nodes, all the models are trained on the remaining
mance of our proposed models on embedding incomplete networks.                 90% nodes. We assume that when a new node comes, there is very
We will first describe how to build incomplete graphs from the views           few connections between it and other nodes, and note that the new
of network structure and nodes properties, respectively. Then we               nodes also have sparse node property information. To let the two
report and analyze the results.                                                view data of the new nodes sparse, we randomly drop out 50% of
   Experiment on Incomplete Networks with Missing Links.                       the edges and properties for the new nodes.
To construct an incomplete network with missing links, we remove                  Performance evaluation on multi-class classification. Sim-
a proportion of links from the original networks. Thus the topol-              ilar to our pervious experiment, we use a proportion of labeled
ogy structure can be considered as incomplete. Specifically, the               original nodes to train the LinearSVM. Note that Tr in this exper-
proportion of removed links increases from 0% to100% to make                   iment denotes the portion of original nodes. The new nodes are
the structure information sparser and sparser. We then adopt our               considered as the test data. We report the classification accuracy
proposed models and 6 baselines to learn the corresponding em-                 of new nodes in Table 5. One can see that the performances of
beddings on such incomplete networks.                                          MVC-DNEDB and MVC-DNEEB are similar and both methods out-
   We also evaluate the performance of various methods on the                  perform all the baselines. The significant improvement over SDNE
incomplete networks by conducting multi-class classification and               shows the effectiveness of our models in incorporating information
link-prediction tasks. Figure 3(a) shows the details of results. One           from the nodes properties by multi-view learning.
can see that with the increase of removed link proportion, the                    Performance evaluation on link prediction. Differing from
curves of classification accuracy and link prediction AUC score                the experiment for original nodes embedding, we want to predict
decrease. However, the decline trends of TADW, SDNE+SV-DNEP                    the edges connecting new nodes to the original nodes. To this aim,
and our proposed MVC-DNEDB , MV-DNEEB methods are signif-                      the removed edges in above mentioned drop-out operation are
icantly smaller than other methods, and they tend to be stable                 considered as positive samples. We then randomly select the same
finally. It demonstrates the effectiveness of incorporating the node           number of unconnected node pairs as negative links. Based on the
properties for network embedding on incomplete network with                    map function f learned in the training step ,we firstly calculate the
sparse links among nodes. Overall, MVC-DNEDB and MV-DNEEB                      representation vectors of the new nodes with various embedding
are less sensitive with the increase of missing link proportions due           methods. After obtaining the cosine similarity of each node pair, we
to their power in multi-view correlation learning to alleviate the             can adopt AUC to evaluate the performance of new nodes embed-
data sparsity issue in the network structure view.                             ding on link prediction. Table 6 shows the experiment result. One
   Experiment on Incomplete Networks with Missing Node                         can see that MVC-DNEEB and MVC-DNEDB outperform all the
Properties. The features in the nodes property view can also be                baselines. Specially, MVC-DNEEB outperforms the best baseline




                                                                         374
Session 2D: Network Embedding 2                                                                                CIKM’17, November 6-10, 2017, Singapore




                        Table 5: Multi-class classification results (Accuracy) on three datasets for new nodes.

      Model                           Citeseer                                        Google+                                     DBLP
       Tr             10%     20%         30%       40%       50%    10%     20%       30%       40%     50%       1%      2%      3%          4%      5%
      SDNE           0.385    0.418     0.435      0.440     0.444   0.473   0.474     0.475    0.476    0.477    0.580   0.575    0.596      0.600   0.592
    SV-DNEP          0.612    0.623     0.628      0.632     0.640   0.371   0.378     0.379    0.381    0.380    0.641   0.652   0.660       0.666   0.665
  SDNE+SV-DNEP       0.634    0.653     0.663      0.663     0.670   0.481   0.484     0.485    0.489    0.490    0.663   0.687   0.7000      0.702   0.707
   MVC-DNED B        0.650    0.665     0.667      0.674     0.675   0.505   0.510     0.512    0.514    0.515    0.710   0.725   0.729       0.734   0.736
   MVC-DNEE B        0.670    0.680     0.681      0.682     0.683   0.509   0.511     0.512    0.513    0.517    0.696   0.715    0.720      0.724   0.725




            (a) dimension d                                (b) α                                (c) βT                              (d) β P


       Figure 4: Embedding Performance for new nodes w.r.t parameter dimension d, hyper-parameter α, βT and β P .

Table 6: Link prediction results (AUC) on three datasets for                       MVC-DNEEB slightly outperforms MVC-DNEDB , because MVC-
new nodes                                                                          DNEEB is able to learn a characteristic representations in both
                                                                                   views. The characteristics of network structure view is more useful
                Model          Citeseer     Google+        DBLP                    for link prediction for new nodes when two views of data are both
                                                                                   sparse.
                SDNE            0.721           0.697      0.831
              SV-DNEP           0.789           0.457      0.734                      Sparsity sensitivity analysis for new nodes embedding. We
            SDNE+SV-DNEP        0.792           0.538      0.854                   tune the proportion of removed links connecting the new nodes
             MVC-DNED B         0.806           0.730      0.895                   to the regular nodes from 20% to 100%. Then we evaluate the per-
             MVC-DNEE B         0.884           0.750      0.896                   formance of the models on new nodes embeddings over Citeseer
                                                                                   dataset. Figure 5 shows the results. One can see that 1) MVC-DNEEB ,
                                                                                   MVC-DNEDB , and SDNE+SV-DNEP perform significantly and con-
                                                                                   sistently better than SDNE; and 2) the performances of all the meth-
                                                                                   ods drop with the increase of the removed link proportions. The
                                                                                   dropping trend of SDNE is much more significant than the other
                                                                                   three methods that consider both view information. It implies that
                                                                                   the proposed methods are more robust to learn representations for
                                                                                   new nodes whose links are very sparse due to their powerful ability
                                                                                   to encode the correlated knowledge from the node properties.


                                                                                     5.7   Parameter Sensitivity Study
                                                                                   Finally, we study the sensitivity of the proposed models on the
                                                                                   parameter represent dimension d, and hyper-parameters α, βT
                                                                                   and β P . We use the classification accuracy and link prediction AUC
                                                                                   score on dataset Citeseer for new nodes to evaluate the performance.
Figure 5: Embedding performance for new nodes w.r.t differ-                        The results are shown in Figure 4. Generally, it is important to set the
ent proportions of removed links on Citeseer dataset.                              number of dimensions for the latent embedding space for previous
                                                                                   methods, but our method is not very sensitive to this parameter. α
                                                                                   is only used in MVC-DNEDB to balance the weight between self-
SDNE+SV-DNEP by around 5% on average. One can also see that                        view reconstruction errors and cross-view reconstruction errors. As




                                                                             375
  Session 2D: Network Embedding 2                                                                                               CIKM’17, November 6-10, 2017, Singapore




 shown in Figure 4(b), we find 0.5 is a reasonable choice. Figure 4(c)                           [13] Chaozhuo Li, Senzhang Wang, Dejian Yang, Zhoujun Li, Yang Yang, and Xiaoming
 and 4(d) show that we should pay more attention on the non-zero                                      Zhang. 2017. PPNE: Property Preserving Network Embedding. In DASFAA.
                                                                                                      Springer, 163–179.
 elements in both views. βT and β P determine the reconstruction                                 [14] Chaozhuo Li, Senzhang Wang, Dejian Yang, Zhoujun Li, Yang Yang, and Xiaoming
 weight of non-zero elements in view T and P. The figures show                                        Zhang. 2017. Semi-Supervised Network Embedding. In DASFAA. Springer, 131–
                                                                                                      147.
 that overall the performance of our models are not very sensitively                             [15] Ping Li, Trevor J Hastie, and Kenneth W Church. 2006. Very sparse random
 to these parameters, demonstrating the robustness of the models.                                     projections. In Proceedings of the 12th ACM SIGKDD international conference on
                                                                                                      Knowledge discovery and data mining. ACM, 287–296.
                                                                                                 [16] David Liben-Nowell and Jon Kleinberg. 2007. The link-prediction problem for
  6        CONCLUSION                                                                                 social networks. journal of the Association for Information Science and Technology
                                                                                                      58, 7 (2007), 1019–1031.
 In this paper, we propose a deep learning-based method MVC-DNE                                  [17] Jiquan Ngiam, Aditya Khosla, Mingyu Kim, Juhan Nam, Honglak Lee, and An-
 to perform network embedding with multi-view data. Specifically,                                     drew Y Ng. 2011. Multimodal deep learning. In Proceedings of the 28th international
                                                                                                      conference on machine learning (ICML-11). 689–696.
 we consider the topology structure and the node properties as                                   [18] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M.
 two correlated views of a node in a network. Then we implement                                       Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cour-
 two models to learn the correlations between two views with the                                      napeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine
                                                                                                      Learning in Python. Journal of Machine Learning Research 12 (2011), 2825–2830.
 self-view and cross-view learning. Benefiting from the consistent                               [19] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: Online learning
 information in multiple views, the learned node representations                                      of social representations. In Proceedings of the 20th ACM SIGKDD international
                                                                                                      conference on Knowledge discovery and data mining. ACM, 701–710.
 are much more robust to the incomplete networks. Experimental                                   [20] Sam T Roweis and Lawrence K Saul. 2000. Nonlinear dimensionality reduction
 results on three real network datasets demonstrate the significant                                   by locally linear embedding. science 290, 5500 (2000), 2323–2326.
 effectiveness of MVC-DNE, especially for efficiently generating                                 [21] Ruslan Salakhutdinov and Geoffrey Hinton. 2009. Semantic hashing. International
                                                                                                      Journal of Approximate Reasoning 50, 7 (2009), 969–978.
 representations for the new nodes.                                                              [22] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and
                                                                                                      Tina Eliassi-Rad. 2008. Collective classification in network data. AI magazine 29,
                                                                                                      3 (2008), 93.
  7        ACKNOWLEDGMENT                                                                        [23] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei.
                                                                                                      2015. Line: Large-scale information network embedding. In Proceedings of the
  This work was supported by the National Natural Science Founda-                                     24th International Conference on World Wide Web. ACM, 1067–1077.
  tion of China (Grand Nos. U1636211, 61672081, 61602237, 61370126),                             [24] Jie Tang, Jing Zhang, Limin Yao, Juanzi Li, Li Zhang, and Zhong Su. 2008. Ar-
  National High Technology Research and Development Program                                           netminer: extraction and mining of academic social networks. In Proceedings of
                                                                                                      the 14th ACM SIGKDD international conference on Knowledge discovery and data
  of China (No.2015AA016004), fund of the State Key Laboratory                                        mining. ACM, 990–998.
  of Software Development Environment (No. SKLSDE-2017ZX-19),                                    [25] Joshua B Tenenbaum, Vin De Silva, and John C Langford. 2000. A global geometric
  and the Directors Project Fund of Key Laboratory of Trustworthy                                     framework for nonlinear dimensionality reduction. science 290, 5500 (2000), 2319–
                                                                                                      2323.
  Distributed Computing and Service (BUPT), Ministry of Education                                [26] Cunchao Tu, Weicheng Zhang, Zhiyuan Liu, and Maosong Sun. 2016. Max-margin
  (Grant No. 2017KF03).                                                                               DeepWalk: discriminative learning of network representation. In Proceedings of
                                                                                                      the Twenty-Fifth International Joint Conference on Artificial Intelligence (IJCAI
                                                                                                      2016). 3889–3895.
  REFERENCES                                                                                     [27] Daixin Wang, Peng Cui, and Wenwu Zhu. 2016. Structural deep network em-
                                                                                                      bedding. In Proceedings of the 22nd ACM SIGKDD International Conference on
   [1] Galen Andrew, Raman Arora, Jeff Bilmes, and Karen Livescu. 2013. Deep canonical                Knowledge Discovery and Data Mining. ACM, 1225–1234.
       correlation analysis. In International Conference on Machine Learning. 1247–1255.         [28] Xiao Wang, Peng Cui, Jing Wang, Jian Pei, Wenwu Zhu, and Shiqiang Yang.
   [2] Mikhail Belkin and Partha Niyogi. 2003. Laplacian eigenmaps for dimensionality                 2017. Community Preserving Network Embedding. In Proceedings of the Thirty-
       reduction and data representation. Neural computation 15, 6 (2003), 1373–1396.                 First AAAI Conference on Artificial Intelligence, February 4-9, 2017, San Francisco,
   [3] Daniel Benyamin, Michael C McGinley, Michael Aaron Hall, and Nicholas J Bina.                  California, USA. 203–209. http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/
       2009. Social advertisement network. (May 18 2009). US Patent App. 12/467,981.                  view/14589
   [4] Jianping Cao, Senzhang Wang, Fengcai Qiao, Hui Wang, Feiyue Wang, and S Yu                [29] Cheng Yang, Zhiyuan Liu, Deli Zhao, Maosong Sun, and Edward Y Chang. 2015.
       Philip. 2016. User-guided large attributed graph clustering with multiple sparse               Network Representation Learning with Rich Text Information.. In Proceedings of
       annotations. In Pacific-Asia Conference on Knowledge Discovery and Data Mining.                the 24th International Joint Conference on Artificial Intelligence. 2111–2117.
       Springer, 127–138.
   [5] Shaosheng Cao, Wei Lu, and Qiongkai Xu. 2015. Grarep: Learning graph rep-
       resentations with global structural information. In Proceedings of the 24th ACM
       International on Conference on Information and Knowledge Management. ACM,
       891–900.
   [6] Tom Fawcett. 2006. An introduction to ROC analysis. Pattern Recognition Letters
       27, 8 (2006), 861–874.
   [7] Fangxiang Feng, Xiaojie Wang, and Ruifan Li. 2014. Cross-modal retrieval with
       correspondence autoencoder. In Proceedings of the 22nd ACM international con-
       ference on Multimedia. ACM, 7–16.
   [8] Francois Fouss, Alain Pirotte, Jean-Michel Renders, and Marco Saerens. 2007.
       Random-walk computation of similarities between nodes of a graph with appli-
       cation to collaborative recommendation. IEEE Transactions on knowledge and
       data engineering 19, 3 (2007).
   [9] Felix A Gers, Jürgen Schmidhuber, and Fred Cummins. 2000. Learning to forget:
       Continual prediction with LSTM. Neural computation 12, 10 (2000), 2451–2471.
  [10] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for
       networks. In Proceedings of the 22nd ACM SIGKDD International Conference on
       Knowledge Discovery and Data Mining. ACM, 855–864.
  [11] Qingbo Hu, Sihong Xie, Shuyang Lin, Senzhang Wang, and S Yu Philip. 2016.
       Clustering Embedded Approaches for Efficient Information Network Inference.
       Data Science and Engineering 1, 1 (2016), 29–40.
  [12] Jure Leskovec and Julian J Mcauley. 2012. Learning to discover social circles in
       ego networks. In Advances in neural information processing systems. 539–547.




                                                                                           376

View publication stats

