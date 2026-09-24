# An attention based collaboration framework for multi view ne Tang Shang Ren etal 2017

> Source: `An_attention_based_collaboration_framework_for_multi_view_ne_Tang_Shang_Ren_etal_2017.pdf`

---

                                             An Attention-based Collaboration Framework for Multi-View
                                                          Network Representation Learning
                                                                Meng Qu1 , Jian Tang2,3 , Jingbo Shang1 , Xiang Ren1 , Ming Zhang4 , Jiawei Han1
                                                                                                1 University of Illinois at Urbana-Champaign, IL, USA
                                                                                                                       2 HEC Montreal, Canada
                                                                                                 3 Montreal Institute of Learning Algorithms, Canada
                                                                                                               4 Peking University, Beijing, China
                                                       1 {mengqu2, shang7, xren7, hanj}@illinois.edu                                     2,3 tangjianpku@gmail.com         4 mzhang cs@pku.edu.cn


                                         ABSTRACT




arXiv:1709.06636v1 [cs.SI] 19 Sep 2017
                                                                                                                                                     View 1                   View 2                   View 3
                                         Learning distributed node representations in networks has been
                                         attracting increasing attention recently due to its effectiveness in a
                                         variety of applications. Existing approaches usually study networks
                                         with a single type of proximity between nodes, which defines a
                                         single view of a network. However, in reality there usually exists
                                         multiple types of proximities between nodes, yielding networks
                                         with multiple views. This paper studies learning node represen-                                      Figure 1: An example multi-view network with three views.
                                         tations for networks with multiple views, which aims to infer ro-                                    Each view corresponds to a type of proximity between nodes,
                                         bust node representations across different views. We propose a                                       which is characterized by a set of edges. Different views are
                                         multi-view representation learning approach, which promotes the                                      complementary to each other.
                                         collaboration of different views and lets them vote for the robust
                                                                                                                                                  Though empirically effective and efficient in many networks, all
                                         representations. During the voting process, an attention mecha-
                                                                                                                                              these work assumes there only exists a single type of proximity
                                         nism is introduced, which enables each node to focus on the most
                                                                                                                                              between nodes in a network, whereas in reality multiple types
                                         informative views. Experimental results on real-world networks
                                                                                                                                              of proximities exist. Take the network between authors in the
                                         show that the proposed approach outperforms existing state-of-the-
                                                                                                                                              scientific literature as an example, the proximity can be induced
                                         art approaches for network representation learning with a single
                                                                                                                                              by co-authorship, meaning whether two authors have once coau-
                                         view and other competitive approaches with multiple views.
                                                                                                                                              thored a paper, or citing relationship, meaning whether one author
                                                                                                                                              cited the papers written by the other one. Another example is the
                                         1     INTRODUCTION                                                                                   network between users in social media sites (e.g, Twitter), where
                                                                                                                                              multiple types of proximities also exist such as the ones induced by
                                         Mining and analyzing large-scale information networks (e.g., social                                  the following-followee, reply, retweet, and mention relationships.
                                         networks [31], citation networks [24] and airline networks [11]) has                                 Each proximity defines a view of a network, and multiple proxim-
                                         attracted a lot of attention recently due to their wide applications                                 ities yield a network with multiple views. Each individual view
                                         in the real world. To effectively and efficiently mine such networks,                                is usually sparse and biased, and thus the node representations
                                         a prerequisite is to find meaningful representations of networks.                                    learned by existing approaches may not be so comprehensive. To
                                         Traditionally, networks are represented as their adjacency matrices,                                 learn more robust node representations, a natural solution could
                                         which are both high-dimensional and sparse. Recently, there is a                                     be leveraging the information from multiple views.
                                         growing interest in representing networks into low-dimensional                                           This motivated us to study a new problem: learning node repre-
                                         spaces (a.k.a, network embedding) [10, 20, 26], where each node is                                   sentations for networks with multiple views, aiming to learn robust
                                         represented with a low-dimensional vector. Such vector represen-                                     node representations by considering multiple views of a network.
                                         tations are able to preserve the proximities between nodes, which                                    In literature, various methods have been proposed for learning data
                                         can be treated as features and benefit a variety of downstream ap-                                   representations from multiple views, such as multi-view clustering
                                         plications, such as node classification [20, 26], link prediction [10]                               methods [3, 12, 12, 33, 37] and multi-view matrix factorization meth-
                                         and node visualization [26].                                                                         ods [9, 16, 22]. These methods perform well on many applications
                                                                                                                                              such as clustering [12, 16] and recommendation [22]. However,
                                         Permission to make digital or hard copies of all or part of this work for personal or
                                         classroom use is granted without fee provided that copies are not made or distributed                when applied to our problem, they have the following limitations:
                                         for profit or commercial advantage and that copies bear this notice and the full citation            (1) Insufficient collaboration of views. As each individual view of
                                         on the first page. Copyrights for components of this work owned by others than the                   a network is usually biased, learning robust node representations
                                         author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
                                         republish, to post on servers or to redistribute to lists, requires prior specific permission        requires the collaboration of multiple views. However, most of
                                         and/or a fee. Request permissions from permissions@acm.org.                                          existing approaches for multi-view learning aim to find compatible
                                         CIKM’17, November 6–10, 2017, Singapore.                                                             representations across different views rather than promote the col-
                                         © 2017 Copyright held by the owner/author(s). Publication rights licensed to ACM.
                                         ISBN 978-1-4503-4918-5/17/11. . . $15.00                                                             laboration of different views for finding robust node representations.
                                         DOI: https://doi.org/10.1145/3132847.3133021                                                         (2) Lack of weight learning. To learn robust node representations,
the information from multiple views needs to be integrated. Dur-              Traditionally, networks are represented as their adjacency ma-
ing integration, as the importance of different views can be quite         trices, which are sparse and high-dimensional. Recently, learning
different, their weights need to be carefully decided. Existing ap-        low-dimensional vector representations of networks (a.k.a. network
proaches usually assign equal weights to all views. In other words,        embedding) attracts increasing attention, which is defined below:
different views are equally treated, which is not reasonable for most
multi-view networks. To overcome the limitations, we are seeking              Definition 2.2. (Network Embedding) Given an information net-
an approach that is able to promote the collaboration of different         work G = (V , E), the problem of network embedding aims to learn
views, and also automatically infer the weights of views during            a low-dimensional vector representation xv ∈ Rd for each node v
integration.                                                               with d  |V |, which preserves the proximities between the nodes.
   In this paper, we propose such an approach. We first introduce a
                                                                              Various network embedding approaches [10, 20, 26] have been
set of view-specific node representations to preserve the proximities
                                                                           proposed recently. Although they have been proved to be effective
of nodes in different views. The view-specific node representations
                                                                           and efficient in various scenarios, they all focus on networks with a
are then combined for voting the robust node representations. Since
                                                                           single type of proximity/relationship. However, in reality we often
the quality of the information in different views may be different,
                                                                           observe multiple types of proximities/relationships between nodes.
it would be ideal to treat the views differently during the voting
                                                                           For example, for users in social media sites such as Twitter, besides
process. In other words, each view should be weighted differently.
                                                                           the following relationships, other relationships also exist such as
Inspired by the recent progress of the attention mechanism [1]
                                                                           retweet, meaning one user forwarded the tweets written by another
for neural machine translation, in this paper we propose an at-
                                                                           user, and mention, meaning one user mentioned another user in his
tention based method to infer the weights of views for different
                                                                           tweets. Each type of proximity/relationship defines a view of net-
nodes, which will leverage a few labeled data. The whole model can
                                                                           works between nodes, and multiple types of proximity/relationship
be efficiently trained through the backpropagation algorithm [21],
                                                                           yield networks with multiple views. Different views of a network
alternating between optimizing the view-specific node representa-
                                                                           are usually complementary to each other, and thus considering
tions and voting for the robust node representations by learning
                                                                           multiple views may help learn more robust node representations.
the weights of different views.
                                                                           This motivated us to study the problem of learning node represen-
   We conduct extensive experiments on various real world multi-
                                                                           tations for networks with multiple views, and we formally define
view networks. Experimental results on both the multi-label node
                                                                           the problem as follows:
classification task and link prediction task show that our proposed
approach outperforms state-of-the-art approaches for learning node            Definition 2.3. (Multi-view Network Embedding) Given an
representation with individual views and other competitive ap-             information network with K views, denoted as G = (V, E 1 , E 2 , . . . , E K ),
proaches with multiple views.                                              the problem of Multi-view Network Embedding aims to learn the
   In summary, in this paper we make the following contributions:          robust node representations {xv }v ∈V ⊆ Rd , which are consistent
• We propose to study multi-view network representation learning,          across different views. Rd is a low-dimensional space with d  |V |.
   which aims to learn node representations by leveraging informa-
   tion from multiple views.                                                  To learn robust node representations across different views, it
• We propose a novel collaboration framework, which promotes               would be desirable to design an approach to promote the collabora-
   the collaboration of different views to vote for robust node rep-       tion of different views and vote for the robust node representations.
   resentations. An attention mechanism is introduced for learning         Since the quality of views are different, the approach should also
   the weights of different views during voting.                           be able to weight the views differently during voting. In the next
• We conduct experiments on several multi-view networks. Experi-           section, we introduce such an approach.
   mental results on two tasks prove the effectiveness and efficiency
   of our proposed approach over many competitive baselines.               3    MULTI-VIEW NETWORK EMBEDDING
                                                                           In this section, we introduce our proposed approach for embedding
2    PROBLEM DEFINITION                                                    networks with multiple views. When applied to the problem, most
                                                                           existing approaches, e.g., multi-view clustering and multi-view
In this section, we introduce some background knowledge and
                                                                           matrix factorization algorithms, fail to achieve satisfactory results.
formally define the problem of multi-view network embedding. We
                                                                           This is because they cannot effectively promote the collaboration of
first define information networks and their views as follows:
                                                                           different views during training. Moreover, they also cannot assign
                                                                           proper weights to different views when combining the information
   Definition 2.1. (Information Network, View) An Informa-                 from them.
tion Network, denoted as G = (V, E), encodes the relationships                To solve these challenges, our approach first mines the node
between different objects, where V is a set of objects and E is a set of   proximities encoded in single views, during which, a collaboration
edges between the objects. Each edge e = (u, v) is associated with a       framework (Sec. 3.1) is proposed to promote the collaboration of
weight wuv > 0, indicating the strength of the relationship between        views. After that, we further integrate different views to vote for
u and v. A view of a network is derived from a single type of prox-        more robust node representations. During voting, we automati-
imity or relationship between the nodes, which can be characterized        cally learn the voting weights of views through an attention based
by a set of edges E.                                                       approach (Sec. 3.2).
 Multi-view         View-specific Node        Voting Weights                               Directly optimizing the above objective is computationally ex-
  Network            Representations             of Views             Labeled Data
                                                                                        pensive because it involves traversing all nodes when computing
                                                                                        the conditional probability. Therefore we adopt the negative sam-
                                                                                        pling techniques [17, 18], which modify the conditional probability
                                                                       Robust Node      pk (v j |vi ) in Eqn. 3 as follows:
                                                                      Representations
                                                           Voting                                                      N
                                                                                                 log σ (cj · xki ) +         Evn ∼P k (v) [log σ (−cn · xki )],
                                                                                                                       Õ
                                          Regularization                                                                                                          (4)
                                                                                                                                   neд
                                                                                                                       n=1
                                                                                        where σ (x) = 1/(1 + exp(−x)) is the sigmoid function. The first
Figure 2: Overview of the proposed approach. The collabo-
                                                                                        term maximizes the probability of some observed edges, and the
ration framework (yellow parts) preserves the node proxim-
                                                                                        second term minimizes the probability of N noisy node pairs, with
ities of different views with a set of view-specific node rep-
                                                                                                                               k (v) ∝ d (k )3/4 and d (k) is
                                                                                        vn sampled from a noise distribution Pneд
resentations, which further vote for the robust representa-                                                                              v            v
tions. During voting, we learn the weights of views through                             the degree of node v in view k.
an attention based method (blue parts), which enables nodes                                By minimizing the objective (3), the view-specific representa-
to focus on the most informative views.                                                 tions {xki }k=1
                                                                                                    K are able to preserve the structure information en-
                                                                                        coded in different views. Next, we promote the collaboration of
                                                                                        different views for voting the robust node representations. In this
   The overall objective of our approach is summarized below:                           process, as the importance of views can be quite different, we try
                        O = Ocoll ab + O at t n .                                 (1)   to assign different weights to them. With all these in mind, we
                                                                                        introduce the following regularization term.
Ocoll ab is the objective function of the collaboration framework,
                                                                                                                            K
                                                                                                                       |V | Õ
in which we aim to learn the node proximities in individual views
                                                                                                                                  λki ||xki − xi ||22 ,
                                                                                                                       Õ
                                                                                                              R=                                                  (5)
and meanwhile vote for the robust node representations. O at t n is                                                    i=1 k =1
the objective function for weight learning. Next, we introduce the
details of each part.                                                                   where || · ||2 is the Euclidean norm of a vector, λki is the weight
                                                                                        of view k assigned by node vi . Intuitively, by learning proper
3.1      Collaboration Framework                                                        weights {λki }kK=1 for each node vi , our approach can let each node
The goal of the collaboration framework is to capture the node                          focus on the most informative views. We will introduce how we
proximities encoded in individual views and meanwhile integrate                         automatically learn such weights in the next section. By minimizing
them to vote for the robust node representations. Therefore, for each                   this objective function, different view-specific representations will
node vi , we introduce a set of view-specific representations {xki }k=1
                                                                     K                  vote for the robust representations based on the following equation:
to preserve the structure information encoded in individual views.                                                             K
                                                                                                                                      λ ik xki .
                                                                                                                               Õ
We also introduce a robust representation xi , which integrates the                                                     xi =                                      (6)
                                                                                                                               k =1
information from all different views.
   To preserve the structure information of individual views with                       Naturally, the robust representations are calculated as the weighted
the view-specific node representations, for a directed edge (vi , v j )                 combinations of the view-specific representations with the coeffi-
in view k, we first define the probability pk (v j |vi ) as follows:                    cients as the voting weights of views, which is quite intuitive.
                                                                                           By integrating both objectives, the final objective of the collabo-
                       pk (v j |vi ) ∝ exp(xkj · ci ),                            (2)   ration framework can be summarized below:
                                                                                                                                  K
where ci is a context representations of node vi . In our approach,                                                               Õ
                                                                                                                Ocoll ab =               O k + ηR,                (7)
the context representations are shared across different views, so
                                                                                                                                  k =1
that different view-specific node representations will locate in the
same semantic space. We also tried using different context represen-                    where η is a parameter used to control the weight of the regulariza-
tations for different views, and we will compare with this variant                      tion term.
in the experiments.
   Following existing studies [25, 26], for each view k, we try to                      3.2    Learning the Weights of Views through
minimize the KL-divergence between the estimated neighbor distri-                              Attention
bution pk (·|vi ) and the empirical neighbor distribution p̂k (·|vi ). The              The above framework proposes a flexible way to let different views
                                                        (k ) (k )
empirical distribution is defined as p̂k (v j |vi ) = w i j /di , where                 collaborate with each other. In this part, we introduce an attention
  (k )                                                              (k )
w i j is the weight of the edge (vi , v j ) in view k and di is the out-                based approach for learning the weights of views during voting. Our
                                                                                        proposed approach is very general, which can automatically learn
degree of node vi in view k. After some simplification, we obtain
                                                                                        the weights of views for different nodes by providing a few labeled
the following objective function for each view k:
                                                                                        data for specific tasks. For example, for the node classification task,
                                     (k )                                               only a few labeled nodes are required; for the link prediction task,
                            Õ
                 Ok = −           w i j log pk (v j |vi ).           (3)
                           (i, j)∈E k                                                   a limited number of links are provided.
   Following the recent attention based models for neural machine              pairwise loss is used:
translation [1], we define the weight of view k for node vi using a                                  link
                                                                                                                     Õ
softmax unit as follows:                                                                           O at tn = −                   cos(xi , xj ).       (12)
                                                                                                                 (v i ,v j )∈S
                                    exp(zk · xC
                                              i )
                      λki = ÍK                                     ,    (8)    In the objective function, (vi , v j ) is a linked node pair and cos(·, ·)
                               k 0 =1
                                        exp(zk 0 · xC
                                                    i )                        is the cosine similarity between vectors.
where xC i is the concatenation of all the view-specific representa-
tions of node vi , and zk is a feature vector of view k, describing
                                                                               3.3    Model Optimization
what kinds of nodes will consider view k as informative. If xC                 The objective function of our approach can be efficiently optimized
                                                                   i
and zk have a large dot product, meaning node vi believes that                 with the coordinate gradient descent algorithm [32] and the back-
view k is an informative view, then the weight of view k for node              propagation algorithm [21]. Specifically, in each iteration, we first
vi will be large based on the definition. Besides, we see that the             follow existing studies [25, 26] to sample a set of edges from the net-
weights of views for each node are determined by the concatenation             work, and optimize the view-specific node representations. Then
of its view-specific representations. Therefore, nodes with large              we infer the parameter vectors of views with the labeled data, and
proximities are likely to have similar view-specific representations,          update the voting weights of views for different nodes. Finally,
and thus focus on similar views. Such property is quite reasonable,            different view-specific node representations will be integrated to
which allows us to better infer the attentions of different nodes by           vote for the robust representations based on the learned weights.
leveraging their proximities preserved in the learned view-specific            The overall optimization algorithm is summarized in Alg. 1.
representations.
   With the above weights as coefficients, different view-specific             Algorithm 1 Optimization Algorithm of MVE.
node representations can be weighted combined to obtain the robust
                                                                               Input: G = (V , E 1, E 2, . . . , E K ), a set of labeled data S , number of
representations according to Eqn. 6. Then we may apply the robust                  samples T , number of negative samples N .
node representations to different predictive tasks, and the voting             Output: Robust node representation.
weights could be automatically learned with the backpropagation                 1: while not converge do
algorithm [21] based on the predictive error. Specifically, taking              2:      Updating the view-specific node representations.
the node classification task as an example, we try to minimize the              3:    while smp ≤ T do
following objective function with respect to the feature vectors of             4:       Randomly pick up a view, denoted as k.
views {zk }kK=1 :                                                               5:       Sample an edge from E k and also N negative edges.
                                                                                6:       Update view-specific representations w.r.t. Eqn. (7) (9).
                                      Õ                                         7:       Update the context representations w.r.t. Eqn. (7).
                        O at t n =              L(xi , yi ),            (9)     8:    end while
                                     v i ∈S                                     9:      Updating the voting weights of views for different nodes.
where S is the set of labeled nodes, xi is the robust representation           10:    Optimize the parameters of the softmax unit w.r.t. Eqn. (9).
                                                                               11:    Update the weights of views for each node according to Eqn. (8).
of node vi , yi is the label of node vi , and L is a specific loss function.
                                                                        K      12:      Updating the robust node representations.
Then the gradient of the objective function with respect to {zk }k=1
                                                                               13:    Vote for the robust representations according to Eqn. (6).
can be calculated as follows:                                                  14: end while
                       Õ K
                            "                                 #
         ∂O at t n            Õ ∂O at t n ∂O at t n
                                                         k l
                    =            (          −          )λi λi xi .     (10)
           ∂zk                     ∂λk           ∂λl
                     v i ∈S l =1            i                  i
                                                                               3.4    Time Complexity
   After optimizing the parameter vectors {zk }kK=1 , the weights              The time complexity of the proposed algorithm is determined by
of views for both the labeled nodes and unlabeled nodes can be                 three processes: learning the view-specific representations, learn-
directly calculated with the definition Eqn. (8). In the experiments,          ing the robust representations and learning the voting weights of
we will show that our weight learning method only requires a small             views. According to the previous study [26], learning view-specific
number of labeled data to converge (Sec. 4.5.2), and we will also              representations takes O(|E|dN ) time, where |E| is the total number
show that our learning method is very efficient (Sec. 4.6).                    of edges in different views, d is the dimension of the node repre-
   In this paper, we investigate two predictive tasks: node classi-            sentations, and N is the number of samples in negative sampling.
fication and link prediction. For the node classification task, the            Learning robust representations takes O(|V |dK) time, where K is
objective function Eqn. (9) is defined as the square loss:                     the number of views in a network. Updating voting weights takes
                      cl ass                                                   O(|S |dK) time, where |S | is the number of labeled data. In prac-
                                   ||wxi − yi ||22 .
                               Õ
                    O at tn =                                    (11)
                                                                               tice, we only have a very small number of labeled data, and thus
                                   v i ∈S
                                                                               |S |  |V |. Besides, we also have |V |  |E| for most networks.
In the objective function, yi is the label vector of node vi , in which        Therefore, the total time complexity of our algorithm can be simpli-
the dimension j is set as 1 if vi belongs to category j and set as             fied as O(|E|dN ), which is proportional to the total number of edges
0 otherwise. w is the parameter set of the classifier. For the link            in the given network. For most real-world networks, as the number
prediction task, the labeled data are a collection of links and the            of edges is usually small, our approach will be very efficient in most
cases. We will study the efficiency performance of the proposed              retweet, mention and friendship. Similarly, the friendship view
approach in Sec. 4.6.                                                        is used for link prediction and the other three views are used
                                                                             for training, which are reconstructed in the same way as the PPI
4     EXPERIMENT                                                             dataset.
We evaluate our proposed approach on two tasks including node                The detailed statistics of these networks are summarized in Ta-
classification and link prediction. We first introduce our setup.         ble 1.

4.1     Experiment Setup                                                     4.1.2 Compared Algorithms. We compare two types of approaches:
   4.1.1 Datasets. We select the following five networks, in which        single-view based and multi-view based.
the first three are used for the node classification task and the last    • LINE [26]: A scalable network embedding model for single views.
two for the link prediction task.                                            We report the best results on single views.
• DBLP: An author network from the DBLP dataset [27]1 . Three             • node2vec [10]: Another scalable network embedding model for
  views are identified including the co-authorship, author-citation          single views. The best results on single views are reported.
  and text-similarity views. The weights of the edges in the co-          • node2vec-merge: A variant of the node2vec model. To ex-
  authorship view are defined as the number of papers coauthored             ploit multiple views of a network, we merge the edges of differ-
  by each pair of authors; the weights in the author-citation view           ent views into a unified view and embed the unified view with
  are defined as the number of papers written by one author and              node2vec.
  cited by the other; the text-similarity view is a 5-nearest neighbor    • node2vec-concat: A variant of the node2vec model. To exploit
  graph and the similarity is calculated based on the titles and             multiple views of a network, we first apply node2vec to learn
  abstracts of each author using TF-IDF. For node classification, we         node representations on each single view, and then concatenate
  select eight diverse research fields as labels including “machine          all learned representations.
  learning”, “computational linguistics”, “programming language”,         • CMSC: A co-regularized multi-view spectral clustering model [12],
  “data mining”, “database”, “system technology”, “hardware” and             which can apply to our problem but cannot scale up to very large
  “theory”. For each field, several representative conferences are           networks. The centroid based variant is used due to its better
  selected, and only papers published in these conferences are kept          efficiency, and the centroid eigenvectors are treated as the node
  to construct the three views.                                              representations.
• Flickr: A user network constructed from Flickr dataset [30]2 ,          • MultiNMF: A multi-view non-negative matrix factorization
  including the friendship view and the tag-similarity view. The             model [16], which can also apply to our problem but cannot
  tag-similarity view is a 100-nearest neighbor graph between                scale up to very large networks.
  users and the user similarity is calculated according to their tags.    • MultiSPPMI: SPPMI [13] is a word embedding model, which
  The community membership are used as classification labels.                learns word embeddings by factorizing the word co-occurrence
• PPI: A protein-protein interaction network constructed from                matrices. We leverage the model to learn node representations
  the STRING database v9.1 [8]. Only the human genes are kept                by jointly factorizing the adjacency matrices of different views
  as nodes. Six views are constructed based on the coexpression,             and sharing the node representations across different views.
  cooccurrance, database, experiments, fusion and neighborhood            • MVE: Our proposed approach for multi-view network embed-
  information. As the original network is very sparse, we follow             ding, which deploys both the collaboration framework and the
  the same way in [26] to reconstruct the six views to make them             attention mechanism.
  denser. More specifically, for each view, we expand the neigh-          • MVE-NoCollab: A variant of MVE. We introduce different con-
  borhood set of the nodes whose degree are less than 1,000 by               text node representations for different views, so that the view-
  adding their neighbors of neighbors until the size of the extended         specific representations will locate in different semantic spaces,
  neighborhood set reaches 1,000. The gene groups provided in                and thus they cannot collaborate with each other during training.
  the Hallmark gene set [15] are treated as the categories of nodes.      • MVE-NoAttn: A variant of MVE. We assign equal weights to dif-
• Youtube: A user network constructed from [35]3 . Five views are            ferent views during voting, without learning the voting weights
  identified including the number of common friends, the number              of views through the attention based approach.
  of common subscriptions, the number of common subscribers,              Note that the DeepWalk model [20] can be viewed as a variant of
  the number of common favorite videos, and the friendship. We            the node2vec model [10] with the parameters p and q as 1, and thus
  believe the friendship view can better reflect the proximity be-        we will not compare with DeepWalk in our experiments.
  tween the users. Therefore, we select the other four views for
  training and predict the links in the friendship view.                     4.1.3 Parameter Settings. For all approaches except node2vec-
• Twitter: A user network constructed from Higgs Twitter Dataset [5]4 .   concat, the dimension of the node representations is set as 100. For
  Due to the sparsity of the original network, we treat it as an undi-    node2vec-concat, the dimension is set as 100K, and K is the number
  rected network here. Four views are identified including reply,         of views in a network. For LINE and MVE, the number of negative
                                                                          samples N is set as 5, and the initial learning rate is set as 0.025, as
1 https://aminer.org/AMinerNetwork
2 http://dmml.asu.edu/users/xufei/datasets.html                           suggested in [17, 26]. For node2vec, we set the window size as 10,
3 http://socialcomputing.asu.edu/datasets/YouTube2                        the walk length as 40, as suggested in [10, 20]. The parameters p
4 https://snap.stanford.edu/data/higgs-twitter.html                       and q are selected based on the labeled data. For MVE, the number
                                                                    Table 1: Statistics of the datasets
                      Category          Dataset   # Node                                    # Edges in Each View                                         # Labeled Data
                                                                  430,117         763,029          691,090
                                         DBLP     69,110                                                                                                      200
                                                               (Co-author)       (Citation)      (Text-sim)
                                                                 3,017,530       3,531,300
                 Node Classification     Flickr   35,314                                                                                                      100
                                                              (Friendship)       (Tag-sim)
                                                                  659,781          14,167           137,930         246,274        1,339      39,220
                                          PPI     16,545                                                                                                      200
                                                             (Coexpression)   (Cooccurrance)     (Database)     (Experimental)   (Fusion)   (Neighbor)
                                                                 1,940,806       5,574,249         2,239,440       3,797,635
                                        Youtube   14,901                                                                                                      500
                                                                 (Friends)    (Subscriptions)   (Subscribers)      (Videos)
                   Link Prediction
                                                                35,254,194       2,845,120        93,051,769
                                        Twitter   304,692                                                                                                     500
                                                                (Mention)         (Reply)         (Retweet)


Table 2: Quantitative results on the node classification task. Without learning the weights of views (MVE-NoAttn), our ap-
proach has already outperformed all baseline approaches. By learning the weights of views through the attention based ap-
proach (MVE), the results are further improved. Removing the collaboration of views (MVE-NoCollab) decreases the results.
                                                                          DBLP                                Flickr                                PPI
                  Category                  Algorithm
                                                                    Macro-F1 Micro-F1                   Macro-F1 Micro-F1                    Macro-F1 Micro-F1
                                             LINE                      70.29             70.77                34.49          54.99              20.69           24.70
                 Single View
                                           node2vec                    71.52             72.22                34.43          54.82              21.20           25.04
                                        node2vec-merge                 72.05             72.62                29.15          52.08              21.00           24.60
                                        node2vec-concat                70.98             71.34                32.21          53.67              21.12           25.28
                                            CMSC                         -                 -                    -              -                 8.97           13.10
                                          MultiNMF                     51.26             59.97                18.16          51.18               5.19            9.84
                 Multi View
                                          MultiSPPMI                   54.34             55.65                32.56          53.80              20.21           23.34
                                         MVE-NoCollab                  71.85             72.40                28.03          54.62              18.23           22.40
                                         MVE-NoAttn                    73.36             73.77                32.41          54.18              22.24           25.41
                                             MVE                       74.51             74.85                34.74          58.95              23.39           26.96

Table 3: Quantitative results on the link prediction task.
                                                                                                100 nodes are randomly sampled for training and 10000 nodes for
MVE achieves the best results through the collaboration
                                                                                                testing. To learn the weights of views, we select a small number
framework and the attention mechanism.
                                                                                                of nodes as labeled data, and the concrete numbers are reported in
      Category                   Algorithm             Youtube          Twitter                 Table 1.
                                  LINE                      85.31        64.18                      We present the results of different approaches on the node clas-
      Single View                                                                               sification task in Table 2. As CMSC cannot scale up to very large
                                node2vec                    88.71        78.75
                                                                                                networks, only the results on the PPI network are reported. For
                             node2vec-merge                 90.31        81.80
                                                                                                the single-view based approaches, both LINE and node2vec do not
                             node2vec-concat                92.12        75.00
                                                                                                perform well. To leverage the information from multiple views,
                                 CMSC                       74.25          -
                                                                                                node2vec-merge combines the edges of different views. However,
                               MultiNMF                     68.30          -
      Multi View                                                                                the proximities of different views are usually not comparable, and
                               MultiSPPMI                   86.35        53.95
                                                                                                simply combining them will destroy the network structures of indi-
                              MVE-NoCollab                  89.47        73.26
                                                                                                vidual views. On the Flickr dataset, the performance of node2vec-
                              MVE-NoAttn                    93.10        82.62
                                                                                                merge is even inferior to the performance of single-view based
                                  MVE                       94.01        84.98
                                                                                                approaches. On the other hand, node2vec-concat will concatenate
                                                                                                all node representations learned on individual views. However, the
                                                                                                representations from some sparse views can be very biased, which
of samples T used in each iteration is set as 10 millions, and the
                                                                                                may destroy the final representations, and thus node2vec-concat
parameter η is set as 0.05 by default.
                                                                                                does not significantly outperform other approaches, even with
                                                                                                much higher dimensions. Both the multi-view clustering method
4.2     Quantitative Results                                                                    (CMSC) and multi-view matrix factorization methods (MultiNMF
   4.2.1 Node Classification. We start by introducing the results                               and MultiSPPMI) fail to perform well, as they cannot effectively
on the node classification task. We treat the node representations                              achieve the collaboration of different views and also learn their
learned by different algorithms as features, and train one-vs-rest                              weights.
linear classifiers using the LibLinear package [6] 5 . For both the                                 For our proposed framework MVE, without leveraging the label-
DBLP and PPI datasets, 10% nodes are randomly sampled as the                                    ing information to learn the voting weights of views (MVE-NoAttn),
training examples and the rest 90% for testing. For the Flickr dataset,                         it already outperforms all baseline approaches, including node2vec-
5 https://www.csie.ntu.edu.tw/∼cjlin/liblinear/
                                                                                                concat, which learns representations with much higher dimensions.
By learning the weights of views using the attention mechanism
(MVE), the results are further improved. Besides, if we remove                       80
                                                                                                               ●
                                                                                                                     ●

                                                                                                                          ●   ●     ●
                                                                                                                                                        96
                                                                                                                                                                 ●   MVE
                                                                                                                                                                                              ●




the collaboration of different views (MVE-NoCollab), we observe
                                                                                                                                                                     MVE−NoCollab
                                                                                                       ●
                                                                                                                                                        95           node2vec−merge
                                                                                                                                        ●




inferior results, which shows that our collaboration framework can
                                                                                                                                                                                          ●



                                                                                                   ●                                                    94
                                                                                     75

                                                                          Micro−F1                                                           Micro−F1
indeed improve the performances by promoting the collaboration                                                                                          93
                                                                                                                                                                                  ●




of views.
                                                                                               ●                                                                        ●




                                                                                     70                                                                 92   ●


                                                                                                           ●       MVE
                                                                                                                   MVE−NoCollab                         91
    4.2.2 Link Prediction. Next we introduce our results on the link                      ●
                                                                                                                   node2vec−merge


prediction task, which aims to predict the links that are most likely
                                                                                     65                                                                 90
                                                                                               2       4             6        8         10                   1          2         3       4   5
to form given existing networks. As the node sets are very large,                                              Group                                                            Group


predicting links on the whole node sets is unrealistic. Therefore,                                     (a) DBLP                                                             (b) Youtube
we follow the experimental setting in [14] to construct a core set       Figure 3: Performances of the robust node representations
of nodes for each dataset, and we only predict the links between         w.r.t. sparsity. The left groups consist of high-degree nodes
the nodes in the core sets. For the Youtube dataset, the core set        (dense) while the right ones consist of low-degree nodes
contains all the nodes appearing in the four views, which has 7,654      (sparse). MVE outperforms MVE-NoCollab and node2vec-
nodes in total. For the Twitter dataset, as there are too many nodes,    merge, especially on the right groups (more sparsity).
we randomly sample 5,000 nodes appearing in all the three views.
For each pair of nodes in the core set, the probability of forming
a link between them is measured as the cosine similarity between         groups in the DBLP dataset and 5 different groups in the Youtube
their robust node representations. To learn the voting weights of        dataset according to their degrees. We compare the robust node rep-
views in our MVE model, we randomly sample 500 edges from                resentations learned by MVE, node2vec-merge and MVE-NoCollab
each dataset as the labeled data, which are then excluded during         (the variant of MVE without promoting the collaboration of views),
evaluation. The performance is measured with the commonly used           and we report the performances on different node groups. The re-
AUC metric [7]. The results of link prediction with different models     sults are presented in Figure 3. The left groups contain nodes with
are presented in Table 3. For CMSC and MultiNMF, as they cannot          larger degrees, in which the data are quite dense; while the right
scale up to very large networks, only the results on the Youtube         groups contain nodes with smaller degrees, and the data are very
dataset are reported.                                                    sparse. For the DBLP dataset, all the three models do not perform
    We see that for the single view based approaches, both node2vec      well on the left node groups since many high-degree nodes belong
and LINE fail to perform well. By merging the edges of different         to multiple research domains, which are more difficult to classify.
views, the results of node2vec-merge are significantly improved, as      On the right groups, node2vec-merge and MVE-NoCollab still have
different views are comparable and complementary on these two            quite poor performances, while our proposed MVE significantly
datasets. Concatenating the representations learned on each view         outperforms them. For the Youtube dataset, similar results are ob-
(node2vec-concat) leads to inferior results on the Twitter dataset,      served. On the left groups, the performance of the three models
as some sparse views, e.g., the view constructed with the replying       are pretty close. On the right groups with low-degree nodes, MVE
relationship, may destroy the concatenated representations. The          outperforms both node2vec-merge and MVE-NoCollab. Overall,
multi-view clustering method (CMSC) and multi-view matrix fac-           compared with MVE-NoCollab and node2vec-merge, we see that
torization methods (MultiNMF and MultiSPPMI) still fail to perform       MVE achieves better results especially on the right node groups
well as they cannot effectively achieve the collaboration of different   (more sparsity), which demonstrates that our approach can effec-
views.                                                                   tively address the data sparsity problem and help learn more robust
    For our proposed framework MVE, it outperforms all the base-         node representations.
line approaches. If we remove the collaboration of views (MVE-
NoCollab) or remove the weight learning method (MVE-NoAttn),             4.4                  Analysis of the Learned Attentions
the results will drop, which demonstrates the effectiveness of our                            (Weights) Over Views
collaboration framework and the importance of the attention mech-        In our proposed MVE model, we adopt an attention based approach
anism.                                                                   to learn the weights of views during voting, so that different nodes
                                                                         can focus most of their attentions on the most informative views.
4.3    Performances w.r.t. Data Sparsity                                 The quantitative results have shown that MVE achieves better re-
                                                                         sults by learning attentions 6 over views. In this part, we will exam-
Based on the above results, we have already seen that our proposed
                                                                         ine the learned attentions to understand why it can help improve
approach MVE can effectively leverage the information from multi-
                                                                         the performances.
ple views to improve the overall performances. In this part, we take
                                                                            We first study what kinds of views turn to attract more attentions
a further step and examine whether MVE is robust to data sparsity
                                                                         from nodes. We take the DBLP and Youtube datasets as examples.
by integrating information from multiple views.
                                                                         For each view, we report the results of the view-specific represen-
   Specifically, we study the performances of MVE on nodes with
                                                                         tations corresponded to this view, and also the average attentions
different degrees, which correspond to different levels of data spar-
                                                                         assigned by different nodes on this view. The results are presented
sity. The degree of each node is calculated as the sum of the degrees
in different views. All the nodes are assigned into 10 different         6 The term attention and the term weight have the same meaning here.
                                                                         other hand, several areas in our dataset are related to artificial in-
                                                                         telligence, such as data mining and machine learning. For authors
                                                                         in those areas, they may use similar terms as well as cite similar
                                                                         papers, so the text-similarity view and the author citation view
                                                                         cannot discriminate these areas from each other. Therefore, authors
                                                                         in these areas pay less attentions to the text-similarity view and
                                                                         the author citation view, and they focus more on the co-authorship
                                                                         view.
                                                                            Overall, the attentions (weights) over views learned by our at-
             (a) DBLP                         (b) Youtube                tention mechanism are very intuitive, which enable different nodes
Figure 4: Comparison of performances on individual views                 to focus on those most informative views.
and the average weights of views. Views with better perfor-
mances usually attract more attentions from nodes.                       4.5                  Parameter Sensitivity
                                                                         Next, we investigate the sensitivity of different parameters in our
in Figure 4. Overall, the performances of views and the average          framework, including η and the number of labeled data.
attentions they receive are positively correlated. In other words,
our approach will let different nodes focus on the views with the
best performances, which is quite reasonable.                                        76
                                                                                                      ●            ●              ●
                                                                                                                                                         94                                ●
                                                                                                                                                                             ●
                                                                                                                                                                                                              ●

                                                                                     74                                                       ●
                                                                                                                                                                                                                       ●
                                                                                                                                                                ●
                                                                                          ●
                                                                                                                                                         92
                                                                                     72

                                                                          Micro−F1
                                                                                                                                                   AUC   90
                                                                                                                        ●   MVE                                                                 ●       MVE
                                                                                     70
                                                                                                                                                         88
                                                                                     68
                                                                                                                                                         86
                                                                                     66
                                                                                          0         0.025        0.05            0.1         0.2                0       0.025             0.05            0.1         0.2
                                                                                                                  eta                                                                     eta


                                                                                                          (a) DBLP                                                               (b) Youtube
                                                                         Figure 6: Performances w.r.t. η. Within a large range
                                                                         (0.025, 0.1), the performance is not sensitive to η. The per-
Figure 5: Case study of the learned attentions on the DBLP               formance remains very stable with the default value 0.05.
dataset. We compare the attentions of authors in four re-
search areas. HW stands for hardware, PL for programming
language, DM for data mining and ML for machine learning.                            59
                                                                                                                                                         85.0


   We further study the learned attentions on a more fine-grained                    58
                                                                                                                                                         84.5

level by comparing the attentions of nodes in different semantic                     57
                                                                                                                                                         84.0
                                                                          Micro−F1
                                                                                                                                                   AUC
groups. We take the DBLP dataset as an example, and study the                        56                                                                  83.5
authors in four different research areas including hardware (HW),                                                       ●   MVE−NoAttn
                                                                                                                            MVE
                                                                                                                                                                                                ●       MVE−NoAttn
                                                                                                                                                                                                        MVE
programming language (PL), data mining (DM) and machine learn-                       55                                                                  83.0

ing (ML). For each research area, we calculate the average view                      54
                                                                                          ●     ●           ●      ●        ●           ●     ●

                                                                                                                                                         82.5
                                                                                                                                                                ●   ●    ●           ●              ●             ●    ●




weights assigned by authors within this area, and we report the                           0    20         40      60
                                                                                                            # Labeled Nodes
                                                                                                                            80         100   120                0       100         200          300
                                                                                                                                                                                 # Linked Node Pairs
                                                                                                                                                                                                              400     500


ratio to the average view weights assigned by other authors, in
                                                                                                          (a) DBLP                                                               (b) Youtube
order to know which views are the most informative for each re-
search area. The results are presented in Figure 5. For authors in the   Figure 7: Performances w.r.t. #labeled data. By learning
areas of hardware and programming language, they have relatively         the voting weights of views with the labeled data, MVE
more attentions on the author citation view; while for authors in        consistently outperforms MVE-NoAttn, which assigns equal
the areas of data mining and machine learning, they focus more           weights. MVE requires only a few labeled data to converge.
on the co-authorship view. This is because we are studying the
node classification task, aiming at predicting the research areas of        4.5.1 Performances w.r.t. η. In our collaboration framework, the
different authors. To increase the prediction accuracy, our attention    parameter η controls the weight of the regularization term (Eqn. 7),
mechanism needs to let the authors focus on the views that can           which trades off between preserving the proximities encoded in
best discriminate them from authors in other areas. For authors          single views and reaching agreements among different views. We
in the hardware and programming language areas, they may cite            compare the performances of our proposed approach w.r.t. η on
very different papers compared with authors in other areas, and          both the node classification task and the link prediction task.
hence the author citation view is the most discriminative for them,         Figure 6 presents the results on the DBLP and Youtube datasets.
so they have more attentions on the author citation view. On the         When η is set as 0, all the three approaches will not perform so
                                                                        Table 4: Efficiency study. Our approach has close running
well as different views are not able to communicate with each
                                                                        time to LINE and node2vec. Learning weights of views takes
other through the regularization term. As we increase η from
                                                                        less than 15% of the running time on both datasets.
0, the performances are improved, which remain stable with a
large range (0.025, 0.1) on both datasets. If we further increase               Algorithm               DBLP               Twitter
η, the performances will begin to drop. This is because a large η                  LINE                91.45 s             589.29 s
forces different views to fully agree with each other, ignoring the              node2vec              144.77 s            981.96 s
differences between the views.                                                  MVE-NoAttn             105.05 s            732.26 s
                                                                                   MVE                 120.38 s            847.65 s
   4.5.2 Performances w.r.t. the number of labeled nodes. To learn
the voting weights of views for different nodes, our framework
requires some labeled nodes. In this part, we investigate the perfor-
                                                                        combination of the two strategies. However, all these approaches
mance of our framework w.r.t. the number of labeled nodes. We
                                                                        focus on learning node representations for networks with a single
take the Flickr and Twitter datasets as examples, and report the
                                                                        view while we study networks with multiple views.
performances of both MVE and MVE-NoAttn.
                                                                            The other line of the related work is multi-view learning, which
   We present the results in Figure 7. We see that by leveraging the
                                                                        aims to exploit information from multiple views and has shown
labeled nodes to learn the voting weights of views, MVE consis-
                                                                        effectiveness in various tasks such as classification [2, 12, 29], clus-
tently outperforms its variant MVE-NoAttn, which assigns equal
                                                                        tering [3, 12, 12, 33, 37], ranking [34], topic modeling [28] and
weights to different views. On both datasets, MVE requires only a
                                                                        activity recovery [36]. The work which is the most similar to ours
very small number of labeled nodes to converge, which shows the
                                                                        is the multi-view clustering [3, 12, 33, 37] and multi-view matrix
effectiveness of our attention based method for weight learning.
                                                                        factorization [9, 16, 22] methods. For example, Kumar et al. [12]
                                                                        proposed a spectral clustering framework to regularize the cluster-
4.6    Efficiency Study                                                 ing hypotheses across different views. Liu et al. [33] proposed a
In this part, we study the efficiency of our proposed framework.        multi-view nonnegative matrix factorization model, which aims to
We select the DBLP and Twitter datasets as examples, and compare        minimize the distance between the coefficient matrix of each view
the running time of MVE with node2vec, LINE and MVE-NoAtten             and the consensus matrix. Our multi-view network representation
(the variant of MVE without learning the voting weights of views).      approach shares similar intuition with these pieces of work, aiming
   Table 4 presents the results. We see that MVE has close running      to find robust data representations across multiple views. However,
time with LINE and node2vec on both datasets. On the Twitter            a major difference is that existing approaches assign equal weights
dataset with more than 300 thousands nodes and 100 millions edges,      to all views, while our approach adopts an attention based method,
the training process of MVE takes less than 15 minutes, which is        which learns different voting weights of views for different nodes.
quite efficient. Besides, comparing the running time of MVE and             Besides, our work is also related to the attention based models,
MVE-NoAttn, we observe that the weight learning process in MVE          which aim to infer the importance of different parts of the training
takes less than 15% of the total running time on both datasets,         data, and let the learning algorithms focus on the most informative
which shows the good efficiency of our attention based approach         parts. Attention based models have been applied to various tasks,
for weight learning.                                                    including image classification [19], machine translation [1] and
                                                                        speech recognition [4]. To the best of our knowledge, this is the
4.7    Case Study                                                       first effort to adopt the attention-based approach in the problem of
Our collaboration framework can effectively preserve the node           multi-view network representation learning.
proximities encoded in different views through the view-specific
representations, which are further used to vote for the robust node     6    CONCLUSIONS
representations. In this part, we give some illustrative examples to    In this paper, we studied learning node representations for net-
show the differences between the view-specific and the robust node      works with multiple views. We proposed an effective framework
representations. We take the author network in DBLP as an example.      to let different views collaborate with each other and vote for the
To compare these node representations, we list the most similar         robust node representations across different views. During voting,
authors given a query author according to the cosine similarity         we proposed an attention based approach to automatically learn
calculated with different node representations. Table 5 presents        the voting weights of views, which requires only a small number
the results. From the nearest neighbors, we can see that the view-      of labeled data. We evaluated the performance of our proposed
specific node representations can well preserve the proximities         approach on five real-world networks with multiple views. Ex-
encoded in the individual views, whereas the robust representations     perimental results on both the node classification task and link
combine the information from all different views.                       prediction task demonstrated the effectiveness and efficiency of our
                                                                        proposed framework. In the future, we plan to apply our frame-
5     RELATED WORK                                                      work to more applications. One promising direction is learning
Our work is related to the existing scalable approaches for learning    node representations for heterogeneous information networks, i.e.,
network representations including DeepWalk [20], LINE [26] and          networks with multiple types of nodes and edges. In such networks,
node2vec [10], which use different search strategies to exploit the     each meta-path [23] characterizes a type of proximity between the
network structures: depth-first search, breadth-first search, and a     nodes, and various meta-paths yield networks with multiple views.
Table 5: Examples of nearest neighbors according to similarity calculated by view-specific node representations and robust
node representations on the DBLP dataset.
                Query              View1: Co-authorship                View2: Author-citation         View3: Text-similarity                     Robust
                                        Hong Cheng                             Jian Pei                      Philip S. Yu                        Jian Pei
                                         Xifeng Yan                        Rakesh Agrawal                   Qiang Yang                          Xifeng Yan
             Jiawei Han                   Jing Gao                      Ramakrishnan Srikant             Christos Faloutsos                    Philip S. Yu
                                         Xiaolei Li                          Philip S. Yu                   Zheng Chen                          Dong Xin
                                         Feida Zhu                            Ke Wang                    S. Muthukrishnan                      Hong Cheng
                                      Erik B. Sudderth                      David M. Blei                   David Barber                    Erik B. Sudderth
                                      Francis R. Bach                       Andrew Y. Ng               Ryan Prescott Adams                Christopher M. Bishop
         Michael I. Jordan             David M. Blei                      Andrew McCallum                   Emily B. Fox                      David M. Blei
                                        Ling Huang                        Thomas Hofmann                 David B. Dunson                      Yee Whye Teh
                                      Tommi Jaakkola                         Eric P. Xing                   Frank Wood                       Alan S. Willsky


ACKNOWLEDGMENTS                                                                             [16] J. Liu, C. Wang, J. Gao, and J. Han. Multi-view clustering via joint nonnegative
                                                                                                 matrix factorization. In Proc. of SDM, volume 13, pages 252–260. SIAM, 2013.
Research was sponsored in part by the U.S. Army Research Lab. un-                           [17] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed rep-
der Cooperative Agreement No. W911NF-09-2-0053 (NSCTA), Na-                                      resentations of words and phrases and their compositionality. In Advances in
                                                                                                 neural information processing systems, pages 3111–3119, 2013.
tional Science Foundation IIS-1320617 and IIS 16-18481, and grant                           [18] A. Mnih and Y. W. Teh. A fast and simple algorithm for training neural proba-
1U54GM114838 awarded by NIGMS through funds provided by the                                      bilistic language models. arXiv preprint arXiv:1206.6426, 2012.
trans-NIH Big Data to Knowledge (BD2K) initiative (www.bd2k.nih.gov).                       [19] V. Mnih, N. Heess, A. Graves, et al. Recurrent models of visual attention. In
                                                                                                 Advances in neural information processing systems, pages 2204–2212, 2014.
The views and conclusions contained in this document are those                              [20] B. Perozzi, R. Al-Rfou, and S. Skiena. Deepwalk: Online learning of social
of the author(s) and should not be interpreted as representing the                               representations. arXiv preprint arXiv:1403.6652, 2014.
official policies of the U.S. Army Research Laboratory or the U.S.                          [21] D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by
                                                                                                 back-propagating errors. Cognitive modeling, 5(3):1.
Government. The U.S. Government is authorized to reproduce and                              [22] A. P. Singh and G. J. Gordon. Relational learning via collective matrix factor-
distribute reprints for Government purposes notwithstanding any                                  ization. In Proceedings of the 14th ACM SIGKDD international conference on
                                                                                                 Knowledge discovery and data mining, pages 650–658. ACM, 2008.
copyright notation hereon. Research was partially supported by                              [23] Y. Sun, J. Han, X. Yan, P. S. Yu, and T. Wu. Pathsim: Meta path-based top-k
the National Natural Science Foundation of China (NSFC Grant                                     similarity search in heterogeneous information networks. Proceedings of the
Nos. 61472006, 61772039 and 91646202).                                                           VLDB Endowment, 4(11):992–1003, 2011.
                                                                                            [24] Y. Sun, Y. Yu, and J. Han. Ranking-based clustering of heterogeneous information
                                                                                                 networks with star network schema. In Proceedings of the 15th ACM SIGKDD
REFERENCES                                                                                       international conference on Knowledge discovery and data mining, pages 797–806.
                                                                                                 ACM, 2009.
 [1] D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly
                                                                                            [25] J. Tang, M. Qu, and Q. Mei. Pte: Predictive text embedding through large-
     learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.
                                                                                                 scale heterogeneous text networks. In Proceedings of the 21th ACM SIGKDD
 [2] A. Blum and T. Mitchell. Combining labeled and unlabeled data with co-training.
                                                                                                 International Conference on Knowledge Discovery and Data Mining, pages 1165–
     In Proceedings of the eleventh annual conference on Computational learning theory,
                                                                                                 1174. ACM, 2015.
     pages 92–100. ACM, 1998.
                                                                                            [26] J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, and Q. Mei. Line: Large-scale infor-
 [3] K. Chaudhuri, S. M. Kakade, K. Livescu, and K. Sridharan. Multi-view clustering
                                                                                                 mation network embedding. In Proceedings of the 24th International Conference
     via canonical correlation analysis. In Proceedings of the 26th annual international
                                                                                                 on World Wide Web, pages 1067–1077. ACM, 2015.
     conference on machine learning, pages 129–136. ACM, 2009.
                                                                                            [27] J. Tang, J. Zhang, L. Yao, J. Li, L. Zhang, and Z. Su. Arnetminer: extraction and
 [4] J. Chorowski, D. Bahdanau, K. Cho, and Y. Bengio. End-to-end continuous
                                                                                                 mining of academic social networks. In Proceedings of the 14th ACM SIGKDD
     speech recognition using attention-based recurrent nn: First results. arXiv
                                                                                                 international conference on Knowledge discovery and data mining, pages 990–998.
     preprint arXiv:1412.1602, 2014.
                                                                                                 ACM, 2008.
 [5] M. De Domenico, A. Lima, P. Mougel, and M. Musolesi. The anatomy of a
                                                                                            [28] J. Tang, M. Zhang, and Q. Mei. One theme in all views: modeling consensus
     scientific rumor. Scientific reports, 3, 2013.
                                                                                                 topics in multiple contexts. In Proceedings of the 19th ACM SIGKDD international
 [6] R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin. Liblinear: A
                                                                                                 conference on Knowledge discovery and data mining, pages 5–13. ACM, 2013.
     library for large linear classification. The Journal of Machine Learning Research,
                                                                                            [29] W. Wang and Z.-H. Zhou. A new analysis of co-training. In Proceedings of the
     9:1871–1874, 2008.
                                                                                                 27th international conference on machine learning (ICML-10), pages 1135–1142,
 [7] T. Fawcett. An introduction to roc analysis. Pattern recognition letters, 27(8):861–
                                                                                                 2010.
     874, 2006.
                                                                                            [30] X. Wang, L. Tang, H. Liu, and L. Wang. Learning with multi-resolution overlap-
 [8] A. Franceschini, D. Szklarczyk, S. Frankild, M. Kuhn, M. Simonovic, A. Roth,
                                                                                                 ping communities. Knowledge and Information Systems (KAIS), 2012.
     J. Lin, P. Minguez, P. Bork, C. Von Mering, et al. String v9. 1: protein-protein
                                                                                            [31] S. Wasserman and K. Faust. Social network analysis: Methods and applications,
     interaction networks, with increased coverage and integration. Nucleic acids
                                                                                                 volume 8. Cambridge university press, 1994.
     research, 41(D1):D808–D815, 2013.
                                                                                            [32] S. J. Wright. Coordinate descent algorithms. Mathematical Programming,
 [9] D. Greene and P. Cunningham. A matrix factorization approach for integrating
                                                                                                 151(1):3–34, 2015.
     multiple data views. In Joint European Conference on Machine Learning and
                                                                                            [33] T. Xia, D. Tao, T. Mei, and Y. Zhang. Multiview spectral embedding. Systems,
     Knowledge Discovery in Databases, pages 423–438. Springer, 2009.
                                                                                                 Man, and Cybernetics, Part B: Cybernetics, IEEE Transactions on, 40(6):1438–1446,
[10] A. Grover and J. Leskovec. node2vec: Scalable feature learning for networks.
                                                                                                 2010.
[11] P. Jaillet, G. Song, and G. Yu. Airline network design and hub location problems.
                                                                                            [34] J. Yu, Y. Rui, and B. Chen. Exploiting click constraints and multi-view features
     Location science, 4(3):195–212, 1996.
                                                                                                 for image re-ranking. IEEE Transactions on Multimedia, 16(1):159–168, 2014.
[12] A. Kumar, P. Rai, and H. Daume. Co-regularized multi-view spectral clustering.
                                                                                            [35] R. Zafarani and H. Liu. Social computing data repository at ASU, 2009.
     In Advances in Neural Information Processing Systems, pages 1413–1421, 2011.
                                                                                            [36] C. Zhang, K. Zhang, Q. Yuan, H. Peng, Y. Zheng, T. Hanratty, S. Wang, and
[13] O. Levy and Y. Goldberg. Neural word embedding as implicit matrix factorization.
                                                                                                 J. Han. Regions, periods, activities: Uncovering urban dynamics via cross-modal
     In Advances in neural information processing systems, pages 2177–2185, 2014.
                                                                                                 representation learning. In Proceedings of the 26th International Conference on
[14] D. Liben-Nowell and J. Kleinberg. The link-prediction problem for social net-
                                                                                                 World Wide Web, pages 361–370. International World Wide Web Conferences
     works. Journal of the American society for information science and technology,
                                                                                                 Steering Committee, 2017.
     58(7):1019–1031, 2007.
                                                                                            [37] D. Zhou and C. J. Burges. Spectral clustering and transductive learning with
[15] A. Liberzon, A. Subramanian, R. Pinchback, H. Thorvaldsdóttir, P. Tamayo,
                                                                                                 multiple views. In Proceedings of the 24th international conference on Machine
     and J. P. Mesirov. Molecular signatures database (msigdb) 3.0. Bioinformat-
                                                                                                 learning, pages 1159–1166. ACM, 2007.
     ics, 27(12):1739–1740, 2011.

