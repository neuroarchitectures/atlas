# Attributed network embedding for learning in a dynamic envir Li Dani Hu etal 2017

> Source: `Attributed_network_embedding_for_learning_in_a_dynamic_envir_Li_Dani_Hu_etal_2017.pdf`

---

                                                                     Attributed Network Embedding for Learning
                                                                              in a Dynamic Environment
                                                                Jundong Li                                                  Harsh Dani                                      Xia Hu
                                                       Arizona State University                                     Arizona State University                        Texas A&M University
                                                          jundongl@asu.edu                                              hdani@asu.edu                                 hu@cse.tamu.edu

                                                              Jiliang Tang                                                    Yi Chang                                    Huan Liu
                                                      Michigan State University                                   Huawei Research America                          Arizona State University




arXiv:1706.01860v2 [cs.SI] 26 Aug 2018
                                                         tangjili@msu.edu                                            yichang@acm.org                                  huan.liu@asu.edu

                                         ABSTRACT                                                                                     1 INTRODUCTION
                                         Network embedding leverages the node proximity manifested to                                 Attributed networks are ubiquitous in myriad of high impact do-
                                         learn a low-dimensional node vector representation for each node                             mains, ranging from social media networks, academic networks, to
                                         in the network. The learned embeddings could advance various                                 protein-protein interaction networks. In contrast to conventional
                                         learning tasks such as node classiﬁcation, network clustering, and                           plain networks where only pairwise node dependencies are ob-
                                         link prediction. Most, if not all, of the existing works, are over-                          served, nodes in attributed networks are often aﬃliated with a rich
                                         whelmingly performed in the context of plain and static networks.                            set of attributes. For example, in scientiﬁc collaboration networks,
                                         Nonetheless, in reality, network structure often evolves over time                           researchers collaborate and are distinct from others by their unique
                                         with addition/deletion of links and nodes. Also, a vast majority                             research interests; in social networks, users interact and commu-
                                         of real-world networks are associated with a rich set of node at-                            nicate with others and also post personalized content. It has been
                                         tributes, and their attribute values are also naturally changing, with                       widely studied and received that there exhibits a strong correlation
                                         the emerging of new content patterns and the fading of old content                           among the attributes of linked nodes [35, 38]. The root cause of the
                                         patterns. These changing characteristics motivate us to seek an ef-                          correlations can be attributed to social inﬂuence and homophily
                                         fective embedding representation to capture network and attribute                            eﬀect in social science theories [30, 31]. Also, many real-world ap-
                                         evolving patterns, which is of fundamental importance for learn-                             plications, such as node classiﬁcation, community detection, topic
                                         ing in a dynamic environment. To our best knowledge, we are                                  modeling and anomaly detection [18, 22, 24, 28, 51], have shown
                                         the ﬁrst to tackle this problem with the following two challenges:                           signiﬁcant improvements by modeling such correlations.
                                         (1) the inherently correlated network and node attributes could be                              Network embedding [7, 8, 11, 16, 19, 20, 34, 36, 41] has attracted
                                         noisy and incomplete, it necessitates a robust consensus represen-                           a surge of research attention in recent years. The basic idea is
                                         tation to capture their individual properties and correlations; (2)                          to preserve the node proximity in the embedded Euclidean space,
                                         the embedding learning needs to be performed in an online fash-                              based on which the performance of various network mining tasks
                                         ion to adapt to the changes accordingly. In this paper, we tackle                            such as node classiﬁcation [3, 6], community detection [42, 51],
                                         this problem by proposing a novel dynamic attributed network em-                             and link prediction [4, 29, 47] can be enhanced. However, a vast
                                         bedding framework - DANE. In particular, DANE ﬁrst provides an                               majority of existing work are predominately designed for plain
                                         oﬄine method for a consensus embedding and then leverages ma-                                networks. They inevitably ignore the node attributes that could
                                         trix perturbation theory to maintain the freshness of the end em-                            be potentially complementary in learning better embedding rep-
                                         bedding results in an online manner. We perform extensive exper-                             resentations, especially when the network suﬀers from high spar-
                                         iments on both synthetic and real attributed networks to corrobo-                            sity. In addition, a fundamental assumption behind existing net-
                                         rate the eﬀectiveness and eﬃciency of the proposed framework.                                work embedding methods is that networks are static and given a
                                                                                                                                      prior. Nonetheless, most real-world networks are intrinsically dy-
                                         CCS CONCEPTS                                                                                 namic with addition/deletion of edges and nodes; examples include
                                         •Information systems → Data mining;                                                          co-author relations between scholars in an academic network and
                                                                                                                                      friendships among users in a social network. Meanwhile, similar as
                                         KEYWORDS                                                                                     network structure, node attributes also change naturally such that
                                                                                                                                      new content patterns may emerge and outdated content patterns
                                         Dynamic Networks; Attributed Networks; Network Embedding
                                                                                                                                      will fade. For example, humanitarian and disaster relief related
                                                                                                                                      topics become popular on social media sites after the earthquakes
                                         Permission to make digital or hard copies of all or part of this work for personal or
                                         classroom use is granted without fee provided that copies are not made or distributed        as users continuously post related content. Consequently, other
                                         for proﬁt or commercial advantage and that copies bear this notice and the full cita-        topics may receive less public interests. In this paper, we refer
                                         tion on the ﬁrst page. Copyrights for components of this work owned by others than
                                         ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or re-
                                                                                                                                      this kind of networks with both network and node attribute value
                                         publish, to post on servers or to redistribute to lists, requires prior speciﬁc permission   changes as dynamic attributed networks.
                                         and/or a fee. Request permissions from permissions@acm.org.
                                         CIKM’17, November 6–10, 2017, Singapore.
                                         © 2016 ACM. ISBN 978-1-4503-4918-5/17/11. . . $15.00
                                         DOI: https://doi.org/10.1145/3132847.3132919
    Despite the widespread of dynamic attributed networks in real-      2 PROBLEM DEFINITION
world applications, the study in analyzing and mining these net-        We ﬁrst summarize some notations used in this paper. Following
works are rather limited. One natural question to ask is when at-       the commonly used notations, we use bold uppercase characters
tributed networks evolve, how to correct and adjust the staleness       for matrices (e.g., A), bold lowercase characters for vectors (e.g., a),
of the end embedding results for network analysis, which will shed      normal lowercase characters for scalars (e.g., a). Also we represent
light on the understanding of their evolving nature. However, dy-       the i-th row of matrix A as A(i, :), the j-th column as A(:, j), the (i, j)-
namic attributed network embedding remains as a daunting task,          th entry as A(i, j), transpose of A as A′ , trace of A as tr (A) if it is
mainly because of the following reasons: (1) Even though network        a square matrix. 1 denotes a vector whose elements are all 1 and
topology and node attributes are two distinct data representations,     I denotes the identity matrix. The main symbols used throughout
they are inherently correlated. In addition, the raw data represen-     this paper are listed in Table 1.
tations could be noisy and even incomplete, individually. Hence,
it is of paramount importance to seek a noise-resilient consensus        Notations                     Deﬁnitions or Descriptions
embedding to capture their individual properties and correlations;          G (t )                 attributed network at time step t
(2) Applying oﬄine embedding methods from scratch at each time             G (t +1)              attributed network at time step t + 1
step is time-consuming and cannot seize the emerging patterns               A(t )         adjacency matrix for the network structure in G (t )
timely. It necessitates the design of an eﬃcient online algorithm           X(t )                     attribute information in G (t )
that can give embedding representations promptly.                          A(t +1)       adjacency matrix for the network structure in G (t +1)
    To tackle the aforementioned challenges, we propose a novel            X(t +1)                  attribute information in G (t +1)
embedding framework for dynamic attributed networks. The main               ∆A        change of adjacency matrix between time steps t and t + 1
contributions can be summarized as follows:                                 ∆X         change of attribute values between time steps t and t + 1
                                                                              n                  number of instances (nodes) in G (t )
                                                                              d                       number of attributes in G (t )
      • Problem Formulations: we formally deﬁne the problem                   k        embedding dimension for network structure or attributes
        of dynamic attributed network embedding. The key idea is              l                 ﬁnal consensus embedding dimension
        to initiate an oﬄine model at the very beginning, based on                                Table 1: Symbols.
        which an online model is presented to maintain the fresh-
        ness of the end attributed network embedding results.               Let U (t ) = {u 1, u 2, ..., un } denote a set of n nodes in the at-
      • Algorithms and Analysis: we propose a novel frame-              tributed network G (t ) at time step t. We use the adjacency ma-
        work - DANE for dynamic attributed network embedding.           trix A(t ) ∈ Rn×n to represent the network structure of U (t ) . In
        Speciﬁcally, we introduce an oﬄine embedding method as          addition, we assume that nodes are aﬃliated with d-dimensional
        a base model to preserve node proximity in terms of both        attributes F = { f 1 , f 2 , ..., fd } and X(t ) ∈ Rn×d denotes the node at-
        network structure and node attributes for a consensus em-       tributes. At the following time step, the attributed network is char-
        bedding representation in a robust way. Then to timely ob-      acterized with both topology and content drift such that new/old
        tain an updated embedding representation when both net-         edges and nodes may be included/deleted, and node attribute val-
        work structure and attributes drift, we present an online       ues could also change. We use ∆A and ∆X to denote the network
        model to update the consensus embedding with matrix             and attribute value changes between two consecutive time step t
        perturbation theory. We also theoretically analyze its time     and time step t + 1, respectively. Following the settings of [44], and
        complexity and show its superiority over oﬄine methods.         for the ease of presentation, we consider the number of nodes is
      • Evaluations: we perform extensive experiments on both           constant over time, but our method can be naturally extended to
        synthetic and real-world attributed networks to corrobo-        deal with node addition/deletion scenarios. As mentioned earlier,
        rate the eﬃcacy in terms of two network mining tasks            node attributes are complementary in mitigating the network spar-
        (both unsupervised and supervised). Also, we show its           sity for better embedding representations. Nonetheless, employing
        eﬃciency by comparing it with other baseline methods            oﬄine embedding methods repeatedly in a dynamic environment
        and its oﬄine counterpart. In particular, our experimental      is time-consuming and cannot seize the emerging/fading patterns
        results show that the proposed method outperforms the           promptly, especially when the networks are of large-scale. There-
        best competitors in terms of both clustering and classiﬁ-       fore, developing an eﬃcient online embedding algorithm upon an
        cation performance. Most importantly, it is much faster         oﬄine model is fundamentally important for dynamic network anal-
        than competitive oﬄine embedding methods.                       ysis, and also could beneﬁt many real-wold applications. Formally,
                                                                        we deﬁne the dynamic attributed embedding problem as two sub-
                                                                        problems as follows. The work ﬂow of the proposed framework
   The rest of this paper is organized as follows. The problem state-   DANE is shown in Figure 1.
ment of dynamic attributed network embedding is introduced in
                                                                          Problem 1. The oﬄine model of DANE at time step t: given net-
Section 2. Section 3 presents the proposed framework DANE with
                                                                        work topology A(t ) and node attributes X(t ) ; output attributed net-
analysis. Experiments on synthetic and real datasets are presented
                                                                        work embedding Y(t ) for all nodes.
in Section 4 with discussions. Section 5 brieﬂy reviews related
work. Finally, Section 6 concludes the paper and visions the future       Problem 2. The online model of DANE at time step t +1: given net-
work.                                                                   work topology A(t +1) and node attributes X(t +1) , and intermediate
            Time
                                             ܎૚ ܎૛ ܎૜ ܎૝ ܎૞                                                                              Problem 1

                                                               ܎૚ ܎૛ ܎૜ ܎૝ ܎૞                    ܎ ܎ ܎ ܎ ܎
                                                                                                ૚ ૛ ૜ ૝ ૞
                                                                                attributes ‫ܝܝ‬૛૚           embedding                              consensus
                     ܎૚ ܎૛ ܎૜ ܎૝ ܎૞                                                         ‫ܝ‬૜
                                                ‫ܝ‬૚                                          ‫ܝ‬૝                                                  embedding
                                      ‫ܝ‬૛                      ‫ܝ‬૜                                      X                        YX
                                                                                               ‫ܝ‬૚‫ܝ‬૛‫ܝ‬૜‫ܝ‬૝
        t                                                                       network      ‫ܝ‬૚                   embedding                           Y
                                                ‫ܝ‬૝                                           ‫ܝ‬૛
                                                                                             ‫ܝ‬૜
                                                                                             ‫ܝ‬૝
                                                                                                      A                        YA
                                           ܎૚ ܎૛ ܎૜ ܎૝ ܎૞


                                            ܎૚ ܎૛ ܎૜ ܎૝ ܎૞
                                                                                                                                         Problem 2
                                                              ܎૚ ܎૛ ܎૜ ܎૝ ܎૞                     ܎૚ ܎૛ ܎૜ ܎૝ ܎૞
                                                                                            ‫ܝ‬૚
                                                                                attribute   ‫ܝ‬૛
                    ܎૚ ܎૛ ܎૜ ܎૝ ܎૞                                                          ‫ܝ‬૜                     online
                                               ‫ܝ‬૚                               changes     ‫ܝ‬૝                     update
       t+1                            ‫ܝ‬૛                     ‫ܝ‬૜                             ‫ܝ‬૞                                                   consensus
                                                                                                     ΔX                       new YX            embedding
                                                                                network        ‫ܝ‬૚‫ܝ‬૛‫ܝ‬૜‫ܝ‬૝‫ܝ‬૞
                                               ‫ܝ‬૝                      ‫ܝ‬૞                   ‫ܝ‬૚
                                                                                changes     ‫ܝ‬૛                    online
                                                                                            ‫ܝ‬૜                                                     new Y
                                                              ܎૚ ܎૛ ܎૜ ܎૝ ܎૞                ‫ܝ‬૝                    update
                                                                                            ‫ܝ‬૞
                                           ܎૚ ܎૛ ܎૜ ܎૝ ܎૞                                            ΔA                       new YA



Figure 1: An illustration of the proposed dynamic attributed network embedding framework - DANE. At time step t, DANE
performs spectral embedding on network structure A and node attributes X, and obtain two embeddings YA and YX . After-
wards, DANE maximizes their correlation for a consensus embedding representation Y. At the following time step t + 1, the
network is characterized by both topology structure and attribute value changes ∆A and ∆X (the changes are highlighted in
orange). DANE leverages matrix perturbation theory to update YA and YX , and give the new consensus embedding Y.


embedding results at time step t; output attributed network embed-                          jeopardized as links are inadequate to provide enough node prox-
ding Y(t +1) for all nodes.                                                                 imity information. Fortunately, rich node attributes are readily
                                                                                            available and could be potentially helpful to mitigate the network
                                                                                            sparsity in ﬁnding better embeddings. Hence, it is more desired
3     THE PROPOSED FRAMEWORK - DANE
                                                                                            to make these two representations compensate each other for a
In this section, we ﬁrst present an oﬄine model that works in a                             consensus embedding. However, as mentioned earlier, both repre-
static setting to tackle the Problem 1 in ﬁnding a consensus embed-                         sentations could be noisy and the existence of noise could degen-
ding representation. Then to tackle the Problem 2, we introduce an                          erate the learning of consensus embedding. Hence, it motivates us
online model that provides a fast solution to update the consensus                          to reduce the noise of these two raw data representations before
embedding on the ﬂy. At the end, we analyze the computational                               learning consensus embedding.
complexity of the online model and show its superiority over the                               Let A(t ) ∈ Rn×n be the adjacency matrix of the attributed net-
oﬄine model.                                                                                                                                                  (t )
                                                                                            work at time step t and DA be the diagonal matrix with DA (i, i) =
                                                                                            Ín      (t )           (t )     (t )     (t ) is a Laplacian matrix. Ac-
3.1    DANE: Oﬀline Model                                                                     j=1 A (i, j), then LA = DA − A
                                                                                            cording to spectral theory [5, 45], by mapping each node in the
Network topology and node attributes in attributed networks are                             network to a k-dimensional embedded space, i.e., yi ∈ Rk (k ≪ n),
presented in diﬀerent representations. Typically, either of these                           the noise in the network can be substantially reduced. A ratio-
two representations could be incomplete and noisy, presenting great                                                            (t )
                                                                                            nal choice of the embedding YA = [y1 , y2 , ..., yn ]′ ∈ Rn×k is to
challenges to embedding representation learning. For example, so-
                                                                                            minimize the loss 21 i, j A(t ) (i, j)||yi − yj ||22 . It ensures that con-
                                                                                                                  Í
cial networks are very sparse as a large amount of users only have
a limited number of links [1]. Thus, network embedding could be                             nected nodes are close to each other in the embedded space. In this
case, the problem boils down to solving the following generalized                    important to build an eﬃcient online embedding algorithm which
                    (t )           (t )                                              gives an informative embedding representation on the ﬂy.
eigen-problem LA a = λDA a. Let a1 , a2 , ..., an be the eigenvec-
tors of the corresponding eigenvalues 0 = λ 1 ≤ λ 2 ≤ ... ≤ λn .                        The proposed online embedding model is motivated by the ob-
It is easy to verify that 1 is the only eigenvector for the eigen-                   servation that most of real-world networks, with no exception for
value λ 1 = 0. Then the k-dimensional embedding YA ∈ Rn×k
                                                            (t )                     attributed networks, often evolve smoothly in the temporal dimen-
of the network structure is given by the top-k eigenvectors start-                   sion between two consecutive time steps [2, 13, 25, 48]. Hence, we
                        (t )                                                         use ∆A and ∆X to denote the perturbation of network structure
ing from a2 , i.e., YA = [a2 , ..., ak , ak+1 ]. For the ease of presen-
                                                                                     and node attributes between two consecutive time steps t and t + 1,
tation, in the following parts of the paper, we refer these k eigen-
                                                                                     respectively. With these, the diagonal matrix and Laplacian matrix
vectors and their eigenvalues as the top-k eigenvectors and eigen-
                                                                                     of A and X also evolve smoothly such that:
values, respectively. Akin to the network structure, noise in the
node attributes can be reduced in a similar fashion. Speciﬁcally, we                                    (t +1)            (t )             (t +1)               (t )
                                                                                                       DA              = DA + ∆DA,         LA          = LA + ∆LA ,
ﬁrst normalize attributes of each node and obtain the cosine simi-                                                                                                                     (3)
                                                                                                        (t +1)            (t )             (t +1)               (t )
larity matrix W(t ) . Afterwards, we obtain the top-k eigenvectors                                     DX              = DX + ∆DX,         LX          = LX + ∆LX .
  (t )
YX = [b2 , ..., bk+1 ] of the generalized eigen-problem correspond-                     As discussed in the previous subsection, the problem of attrib-
ing to W(t ) .                                                                       uted network embedding in an oﬄine setting boils down to solving
    The noisy data problem is resolved by ﬁnding two intermedi-                      generalized eigen-problems. In particular, oﬄine model focuses on
                      (t )        (t )
ate embeddings YA and YX . We now take advantage of them to                          ﬁnding the top eigenvectors corresponding to the smallest eigen-
seek a consensus embedding. However, since they are obtained                         values of the generalized eigen-problems. Therefore, the core idea
individually, these two embeddings may not be compatible and in                      to enable online update of the embeddings is to develop an eﬃ-
the worst case, they may be independent of each other. To capture                    cient way to update the top eigenvectors and eigenvalues. Other-
their interdependency and to make them compensate each other,                        wise, we have to perform generalized eigen-decomposition each
we propose to maximize their correlations (or equivalently mini-                     time step, which is not practical due to its high time complexity.
mize their disagreements) [17]. In particular, we seek two projec-                      Without loss of generality, we use the network topology as an
               (t )          (t )                            (t )    (t )            example to illustrate the proposed algorithm for online embedding.
tion vectors pA and pX such that the correlation of YA and YX
                                                                                     By the matrix perturbation theory [39], we have the following
is maximized after projection. It is equivalent to solving the fol-
                                                                                     equation in embedding the network structure at the new time step:
lowing optimization problem:
                                                                                                (t )                                               (t )
                          (t )′ (t )′ (t ) (t )   (t )′ (t )′ (t ) (t )                     (LA + ∆LA )(a + ∆a) = (λ + ∆λ)(DA + ∆DA )(a + ∆a).                                         (4)
              max pA YA YA pA + pA YA YX pX
              (t ) (t )
             pA , pX                                                                 For a speciﬁc eigen-pair (λi , ai ), we have the following equation:
                  (t )′
                      (t )′ (t ) (t )   (t )′ (t )′ (t ) (t )                 (1)        (t )                                                     (t )
             + pX YX YA pA + pX YX YX pX .                                            (LA + ∆LA )(ai + ∆ai ) = (λi + ∆λi )(DA + ∆DA )(ai + ∆ai ). (5)
                  (t )′ (t )′ (t ) (t )      (t )′ (t )′ (t ) (t )
            s.t. pA YA YA pA + pX YX YX pX = 1.                                      The problem now is how to compute the change of the i-th eigen-
   Let γ be the Lagrange multiplier for the constraint, by setting                   pair (∆ai , ∆λi ) by taking advantage of the small perturbation ma-
                                                (t )    (t )                         trices ∆D and ∆L.
the derivative of the Lagrange function w.r.t. pA and pX to zero,
                                                                                        A - Computing the change of eigenvalue ∆λ i .
                                                  (t )   (t )
we obtain the optimal solution for [pA ; pX ], which corresponds                     By expanding the above equation, we have:
to the eigenvector of the following generalized eigen-problem:
                                                                                                        (t )                        (t )
                                                                                                       LA ai + ∆LA ai + LA ∆ai + ∆LA ∆ai
     (t )′ (t )  (t )′ (t )                   (t )′ (t )
  "                         #"       #     "                        #"       #
                                (t )                                    (t )
    YA YA       YA YX          pA            YA YA            0        pA                                      (t )                             (t )
         ′
     (t ) (t )       ′
                 (t ) (t )      (t )   = γ                    ′
                                                          (t ) (t )     (t ) .                    =λi DA ai + λi ∆DA ai + ∆λi DA ai + ∆λi ∆DA ai                                       (6)
    YX YA       YX YX          pX                 0      YX YX         pX
                                                                                                        (t )              (t )
                                                                               (2)                +(λi DA + λi ∆DA + ∆λi DA + ∆λi ∆DA )∆ai .
As a result, to obtain a consensus embedding representation from                     The higher order terms, i.e., ∆λi ∆DA ai , λi ∆DA ∆ai , ∆λi DA ∆ai
                                                                                                                                                                                (t )
YA and YX , we could take the top-l eigenvectors of the above gener-                 and ∆λi ∆DA ∆ai can be removed as they have limited eﬀects on
alized eigen-problem and stack these top-l eigenvectors together.                    the accuracy of the generalized eigen-systems [15]. By using the
Suppose the projection matrix P(t ) ∈ R2k×l is the concatenated                                 (t )       (t )
                                                                                     fact that LA ai = λi DA ai , we have the following formulation:
top-l eigenvectors, the ﬁnal consensus embedding representation
                               (t ) (t )                                                                         (t )                                    (t )          (t )
can be computed as Y(t ) = [YA , YX ] × P(t ) .                                             ∆LA ai + LA ∆ai = λi ∆DA ai + ∆λi DA ai + λi DA ∆ai .                                      (7)

3.2     Online Model of DANE                                                         Multiplying both sides with ai′ , we now have:
                                                                                                                (t )                                            (t )          (t )
More often than not, attributed networks often exhibit high dy-                      ai′ ∆LA ai + ai′ LA ∆ai = λi ai′ ∆DA ai + ∆λi ai′ DA ai + λi ai′ DA ∆ai .
namics. For example, in social media sites, social relations are con-                                                                                     (8)
tinuously evolving, and user posting behaviors may also evolve                                                             (t )
                                                                                     Since both the Laplacian matrix LA and the diagonal matrix DA
                                                                                                                                                           (t )
accordingly. It raises challenges to the existing oﬄine embedding                    are symmetric, we have:
methods as they have to rerun at each time step which is time-
                                                                                                                             (t )               (t )
consuming and is not scalable to large networks. Therefore, it is                                                        ai′ LA ∆ai = λi ai′ DA ∆ai .                                  (9)
Therefore, Eq. (8) can be reformulated as follows:                                                          Then the solution of αii is as follows:
                                                        (t )                                                                                 1
                  ai′ ∆LA ai = λi ai′ ∆DA ai + ∆λi ai′ DA ai .                                      (10)                            αii = − ai′ ∆DA ai .                  (17)
                                                                                                                                             2
Through this, the variation of eigenvalue, i.e., ∆λi , is:                                                     With the solutions of αip (p , i) and αii , the perturbation of
                              a ′ ∆LA ai − λi ai′ ∆DA ai                                                    eigenvector ai is given as follows:
                         ∆λi = i                                                                    (11)
                                                                                                                                        Õ a ′j ∆LA ai − λi a ′j ∆DA ai
                                                         .
                                           (t )                                                                                         k+1
                                      ai′ DA ai                                                                      1
                                                                                                              ∆ai = − ai′ ∆DA ai ai +          (                       )a j . (18)
  Theorem 3.1. In the generalized eigen-problem Av = λBv, if A                                                       2                j=2, j,i
                                                                                                                                                    λi − λ j
and B are both Hermitian matrices and B is a positive-semideﬁnite                                               Overall, the i-th eigen-pair (∆λi , ∆ai ) can be updated on the ﬂy
matrix, the eigenvalue λ are real; and eigenvectors vj (i , j) are                                          by Eq. (12) and Eq. (18), the pseudocode of the updating process is
B-orthogonal such that vi′ Bvj = 0 and vi′ Bvi = 1 [33].                                                    illustrated in Algorithm 1. The ﬁrst input is the top-k eigen-pairs of
                                 (t )                              (t )                                     the generalized eigen-problem, they can be computed by standard
   Corollary 3.2. ai′ DA ai = 1 and ai′ DA a j = 0 (i , j).                                                 methods like power iteration and Lanczos method [15]. Another
                                                                                                            input is the variation of the diagonal matrix and the Laplacian ma-
                          (t )            (t )                                                              trix. For the top-k eigen-pairs, we update eigenvalues in line 2 and
   Proof. Both DA and LA are symmetric and are also Hermit-
                                                                               (t )                         update eigenvectors in line 3.
ian matrices. Meanwhile, the Laplacian matrix LA is a positive-
                                                                                                                Likewise, the embedding of node attributes can also be updated
deﬁnite matrix, which completes the proof.                                                                                                                            (t )     (t )
                                                                                                            in an online manner by Algorithm 1. Speciﬁcally, let YA and YX
   Therefore, the variation of the eigenvalue λi is as follows:                                             denote the embedding of network structure and node attributes at
                   ∆λi = ai′ ∆LA ai − λi ai′ ∆DA ai .                   (12)                                time step t, then at the following time step t +1, we ﬁrst employ the
   B - Computing the change of eigenvector ∆ai .                                                            proposed online model to update their embedding representations,
As network structure often evolves smoothly between two contin-                                             then a ﬁnal consensus embedding representation Y(t +1) is derived
uous time steps, we assume that the perturbation of the eigenvec-                                           by the correlation maximization method mentioned previously.
tors ∆ai lies in the column space that is composed by the top-k
eigenvectors at time step t such that ∆ai = k+1
                                                 Í                                                          Algorithm 1 Updating of embedding results for the network
                                                   j=2 α i j a j , where α i j
is a weight indicating the contribution of the j-th eigenvector a j                                         Input: Top-k eigen-pairs of the generalized eigen-problem
in approximating the new i-th eigenvector. Next, we show how                                                    {(λ 2, a2 ),(λ 3 , a3 ),…,(λk+1 , ak+1 )} at time t, variation of the di-
to determine these weights such that the perturbation ∆ai can be                                                agonal matrix ∆LA and Laplacian matrix ∆DA .
estimated.                                                                                                                                           (t +1) (t +1)     (t +1) (t +1)
                                                                                                            Output: Top-k eigen-pairs {(λ 2 , a2 ),…,(λk+1 , ak+1 )} at
   By plugging ∆ai = k+1
                        Í
                          j=2 α i j a j into Eq. (7) and using the fact                                         time step t + 1.
      (t ) Ík+1          (t ) Ík+1
that LA j=2 αi j a j = DA j=2 αi j λ j a j , we obtain the following:                                        1: for i = 2 to k + 1 do
                                                                                                             2:      Calculate the variation of ∆λi by Eq. (12);
                  k +1                                                                  k +1
           (t )
                  Õ                                                                     Õ                    3:      Calculate the variation of ∆ai by Eq. (18);
 ∆LA ai + DA             α i j λ j a j = λi ∆DA ai + ∆λi D(t )        (t )
                                                          A ai + λ i DA                        α i j aj .              (t +1)                 (t +1)
                  j=2                                                                    j=2                 4:     λi        = λi + ∆λi ; ai        = ai + ∆ai ;
                                                                                                     (13)    5: end for
By multiplying eigenvector ap′ (2 ≤ p ≤ k + 1, p , i) on both sides
of Eq. (13) and taking advantage of the orthonormal property from
Corollary 3.2, we obtain the following:                                                                       C - Computational Complexity Analysis. We theoretically
                                                                                                            analyze the computational complexity of the proposed online al-
                                               k+1
                                        (t )
                                               Õ                                                            gorithm and show its superiority over the oﬄine embedding meth-
           ap′ ∆LA ai + ap′ DA                       αi j λ j aj                                            ods.
                                               j=2
                                                                          k+1                                  Lemma 3.3. The time complexity of the proposed online embed-
                                                                                                    (14)
                                    (t )           (t )                                                     ding algorithm over T time steps is O(Tk 2 (n + l + l a + l x + d x + d x )),
                                                                          Õ
         = λi ap′ ∆DA ai + ∆λi ap′ DA ai + λi ap′ DA                                αi j aj
                                                                              j=2                           where k is the intermediate embedding dimension for network (or at-
                                                                                                            tributes), l is the ﬁnal consensus embedding dimension, n is the num-
     ⇒ ap′ ∆LA ai + αip λp = λi ap′ ∆DA ai + αip λi .
                                                                                                            ber of nodes, and l a , l x , d a , d x are the number of non-zero entries in
Hence, the weight αip can be determined by:                                                                 the sparse matrices ∆LA , ∆LX , ∆DA , and ∆DX , respectively.
                                  ap′ ∆LA ai − λi ap′ ∆DA ai                Proof. In each time step, to update the top-k eigenvalues of
                         αip =                                            .                         (15)
                                                     λ i − λp            the network and node attributes in an online fashion, it requires
After eigenvector perturbation, we still need to make the orthonor-      O(k(d a + l a )) and O(k(d x + l x )), respectively. Also, the online
mal condition holds for new eigenvectors, thus we have (ai +∆ai )′ (DA + updating   of the top-k eigenvectors for the network and attributes
                                                                                 2 (d +l +n)) and O(k 2 (d +l +n)), respectively. After that,
∆DA )(ai + ∆ai ) = 1. By expanding it and removing the second-           are O(k     a a                    x x
order and third-order terms, we obtain the following equation:           the complexity    for the consensus  embedding is O(k 2l). Therefore,
                                                                         the computational complexity of the proposed online model over
                         (t )          (t )
                   2ai′ DA ∆ai + ai′ ∆DA ai = 0.               (16)      T time steps are O(Tk 2 (n + l + l a + l x + d x + d x )).          
                   BlogCatalog Flickr Epinions DBLP                       4.2 Experimental Settings
      # Nodes         5,196      7,575     14,180    23,393               One commonly adopted way to evaluate the quality of the embed-
    # Attributes      8,189      12,047     9,936     8,945               ding representation [8, 21, 34, 41] is by the following two unsu-
       # Edges       173,468    242,146 227,642 289,478                   pervised and supervised tasks: network clustering and node clas-
      # Classes         6          9         20         7                 siﬁcation. First, we validate the eﬀectiveness of the embedding
    # Time Steps        10         10        16        16                 representations by DANE on the network clustering task. Two
         Table 2: Detailed information of the datasets.                   standard clustering performance metrics, i.e., clustering accuracy
                                                                          (ACC) and normalized mutual information (NMI) are used. In par-
                                                                          ticular, after obtaining the embedding representation of each node
                                                                          in the attributed network, we perform K-means clustering based
   Lemma 3.4. The time complexity of the proposed oﬄine embed-            on the embedding representations. The K-means algorithm is re-
ding algorithm over T time steps is O(Tn 2 (k + l)), where k is the       peated 10 times and the average results are reported since K-means
intermediate embedding dimension for network (or attributes), l is        may converge to the local minima due to diﬀerent initializations.
the ﬁnal consensus embedding dimension.                                   Another way to assess the embedding is by the node classiﬁcation
                                                                          task. Speciﬁcally, we split the the embedding representations of all
    Proof. Omitted for brevity.                                          nodes via a 10-fold cross-validation, using 90% of nodes to train a
                                                                          classiﬁcation model by logistic regression and the rest 10% nodes
  As can be shown, since ∆LA , ∆LX , ∆DA , and ∆DX are often very         for the testing. The whole process is repeated 10 times and the
sparse, thus l a , l x , d a , d x are usually very small, meanwhile we   average performance are reported. Three evaluation metrics, clas-
have k ≪ n and l ≪ n. Based on the above analysis, the proposed           siﬁcation accuracy, F1-Macro and F1-Micro are used. How to de-
online embedding algorithm for dynamic attributed networks is             termine the optimal number of embedding dimensions is still an
much more eﬃcient than rerunning the oﬄine method repeatedly.             open research problem, thus we vary the embedding dimension as
                                                                          {10, 20, ..., 100} and the best results are reported.
4     EXPERIMENTS                                                            4.2.1 Baseline Methods. DANE is measured against the follow-
In this section, we conduct experiments to evaluate the eﬀective-         ing baseline methods on the two aforementioned tasks:
ness and eﬃciency of the proposed DANE framework for dynamic                     • Deepwalk: learns network embeddings by word2vec and
attributed network embedding. In particular, we attempt to answer                  truncated random walk techniques [34].
the following two questions: (1) Eﬀectiveness: how eﬀective are the              • LINE: learns embeddings by preserving the ﬁrst-order and
embeddings obtained by DANE on diﬀerent learning tasks? (2) Eﬃ-                    second-order proximity structures of the network [41].
ciency: how fast is the proposed framework DANE compared with                    • DANE-N: is a variation of the proposed DANE with only
other oﬄine embedding methods? We ﬁrst introduce the datasets                      network information.
and experimental settings before presenting details of the experi-               • DANE-A: is a variation of the proposed DANE with only
mental results.                                                                    attribute information.
                                                                                 • CCA: directly uses the original network structure and at-
4.1    Datasets                                                                    tributes for a joint low-dimensional representation [17].
We use four datasets BlogCatalog, Flickr, Epinions and DBLP for                  • LCMF: maps network and attributes to a shared latent
experimental evaluation. Among them, BlogCatalog and Flickr are                    space by collective matrix factorization [56].
synthetic data from static attributed networks, and they have been               • LANE: is a label informed attributed network embedding
used in previous research [26, 27]. We randomly add 0.1% new                       method, we use one of its variant LANE w/o Label [20].
edges and change 0.1% attribute values at each time step to sim-                 • DANE-O: is a variation of DANE that reruns the oﬄine
ulate its evolving nature. The other two datasets, Epinions and                    model at each time step.
DBLP are real-world dynamic attributed networks. Epinions is a               It is important to note that Deepwalk, LINE, CCA, LCMF, LANE,
product review site in which users share their reviews and opin-          and DANE-O can only handle static networks. To have a fair com-
ions about products. Users themselves can also build trust net-           parison with the proposed DANE framework, we rerun these base-
works to seek advice from others. Node attributes are formed by           line methods at each time step and report the average performance
the bag-of-words model on the reviews, while the major categories         over all time steps1 . We follow the suggestions of the original pa-
of reviews by users are taken as the ground truth of class labels.        pers to set the parameters of all these baselines.
The data has 16 diﬀerent time steps. In the last dataset DBLP, we
extracted a DBLP co-author network for the authors that publish           4.3 Unsupervised Task - Network Clustering
at least two papers between the years of 2001 and 2016 from seven
                                                                          To evaluate the eﬀectiveness of embedding representations, we
diﬀerent areas. Bag-of-words model is applied on the paper title
                                                                          ﬁrst compare DANE with baseline methods on network clustering
to obtain the attribute information, and the major area the authors
                                                                          which is naturally an unsupervised learning task. As per the fact
publish is considered as ground truth. It should be noted that in all
                                                                          that the attributed networks are constantly evolving, we compare
these four datasets, the evolution of network structure and node
attributes are very smooth. The detailed statistics of these datasets     1 For baseline methods that cannot ﬁnish in 24hrs, we only run it once. As networks
are listed in Table 2.                                                    evolve smoothly, there is not much diﬀerence in terms of average performance.
the average clustering performance over all time steps. The aver-       4.5 Eﬃciency of Online Embedding
age clustering performance comparison w.r.t. ACC and NMI are            To evaluate the eﬃciency of the proposed DANE framework, we
presented in Table 3. We make the following observations:               compare DANE with several baseline methods CCA, LCMF, LANE
                                                                        which also use two data representations. Also, we include the of-
      • DANE and its oﬄine version DANE-O consistently out-             ﬂine version of DANE, i.e., DANE-O. As all these methods are not
        perform all baseline methods on four dynamic attributed         designed to handle network dynamics, we compare their cumu-
        networks by achieving better clustering performance. We         lative running time over all time steps and plot it in a log scale.
        also perform pairwise Wilcoxon signed-rank test [14] be-        As can be observed from Figure 2, the proposed DANE is much
        tween DANE, DANE-O and these baseline methods and               faster than all these comparison methods. In all these datasets, it
        the test results show that DANE and DANE-O are signiﬁ-          terminates within one hour while some oﬄine methods need sev-
        cantly better (with both 0.01 and 0.05 signiﬁcance levels).     eral hours or even days to run. It can also be shown that both
      • DANE, DANE-O and LANE achieve better clustering per-            DANE and DANE-O are much faster than all other oﬄine meth-
        formance than network embedding methods such as Deep-           ods. To be more speciﬁc, for example, DANE is 84×, 21× and 14×
        walk, LINE and DANE-N and attribute embedding method            faster than LCMF, CCA and LANE respectively on Flickr dataset.
        DANE-A. The improvements indicate that attribute infor-         To further investigate the superiority of DANE against its oﬄine
        mation is complementary to pure network topology and            version DANE-O, we compare the speedup rate of DANE against
        can help learn more informative embedding representa-           DANE-O w.r.t. diﬀerent embedding dimensions in Figure 3. As can
        tions. Meanwhile, DANE also outperforms the CCA and             be observed, when the embedding dimension is small (around 10),
        LCMF which also leverage node attributes. The reason is         DANE achieves around 8×, 10×, 8×, 12× speedup on BlogCatalog,
        that although these methods learn a low-dimensional rep-        Flickr, Epinions, and DBLP, respectively. When the embedding di-
        resentation by using both sources, they are not explicitly      mensionality gradually increases, the speedup of DANE decreases,
        designed to preserve the node proximity. Also, their per-       but it is still signiﬁcantly faster than DANE-O. With all the above
        formance degenerates when the data is very noisy.               observations, we can draw a conclusion that the proposed DANE
      • Even though DANE leverages matrix perturbation theory           framework is able to learn informative embeddings for attributed
        to update the embedding representations, its performance        networks eﬃciently without jeopardizing the classiﬁcation and the
        is very close to DANE-O which reruns at each time step.         clustering performance.
        It implies that the online embedding model does not sacri-
        ﬁce too much informative information in terms of embed-
        ding.                                                           5 RELATED WORK
                                                                        We brieﬂy review related work from (1) network embedding; (2)
                                                                        attributed network mining; and (3) dynamic network analysis.
4.4   Supervised Task - Node Classiﬁcation                                 The pioneer of network embedding can be dated back to the
Next, we assess the eﬀectiveness of embedding representations on        2000s when many graph embedding algorithms [5, 37, 43] were
a supervised learning task - node classiﬁcation. Similar to the set-    proposed. These methods target to build an aﬃnity matrix that
tings of network clustering, we report the average classiﬁcation        preserves the local geometry structure of the data manifold and
performance over all time steps. The classiﬁcation results in terms     then embed the data to a low-dimensional representation. Moti-
of three diﬀerent measures are shown in Table 4. The following          vated by the graph embedding techniques, Chen et al. [11] pro-
ﬁndings can be inferred from the table:                                 posed one of the ﬁrst network embedding algorithms for directed
                                                                        networks. They used random walk to measure the proximity struc-
      • Generally, we have the similar observations as the cluster-     ture of the directed network. Recently, network embedding tech-
        ing task. The methods which only use link information or        niques have received a surge of research interests in network sci-
        node attributes (e.g., Deepwalk, LINE, DANE-N, DANE-A)          ence. Among them, Deepwalk [34] generalizes the word embed-
        and methods which do not explicitly model node proxim-          ding and employs a truncated random walk to learn latent rep-
        ity (e.g., CCA, LCMF) give poor classiﬁcation results.          resentations of a network. Node2vec [16] further extends Deep-
      • The embeddings learned by DANE and DANE-O help train            walk by adding the ﬂexibility in exploring node neighborhoods.
        a more discriminative classiﬁcation model by obtaining          LINE [41] carefully designs an optimized objective function that
        higher classiﬁcation performance. In addition, pairwise         preserves ﬁrst-order and second-order proximities to learn network
        Wilcoxon signed-rank test [14] shows that DANE and DANE-        representations. GraRep [7] can be regarded as an extension of
        O are signiﬁcantly better.                                      LINE which considers high-order information. Most recently, some
      • For the node classiﬁcation task, the attribute embedding        deep learning based approaches are proposed to enhance the learned
        method DANE-A works better than the network embed-              embeddings [46, 52].
        ding method in the BlogCatalog, Flickr and Epinions datasets.      All the above mentioned approaches, however, are limited to
        The reason is that in these datasets, the class labels are      deal with plain networks. In many cases, we are often faced with
        more closely related to the attribute information than the      attributed networks. Many eﬀorts have been devoted to gain in-
        network structure. However, it is a diﬀerent case for the       sights from attributed networks. For example, Zhu et al. [56] pro-
        DBLP dataset in which the labels of authors are more closely    posed a collective matrix factorization model that learns a low-
        related to the coauthor relationships.                          dimensional latent space by both the links and node attributes.
                                                                      Table 3: Clustering results (%) comparison of diﬀerent embedding methods.

                                                                             Datasets                               BlogCatalog          Flickr                                                    Epinions                 DBLP
                                                                             Methods                                ACC NMI          ACC NMI                                                     ACC NMI                ACC NMI
                                                                                                Deepwalk            49.85 30.51      40.70 24.29                                                 13.31 12.72            53.61 32.54
                                                                    Network                       LINE              50.20 29.53      42.93 26.01                                                 14.34 12.65            51.61 30.74
                                                                                                DANE-N              37.05 21.84      31.89 18.91                                                 12.01 11.95            56.61 31.54
                                                                    Attributes                  DANE-A              62.32 45.95      63.80 48.29                                                 16.12 11.62            47.37 20.64
                                                                                                  CCA               33.42 11.86      24.39 10.89                                                 10.85 8.61             26.42 18.60
                                                                                                 LCMF               55.72 40.38      27.03 13.06                                                 12.86 10.73            42.27 26.48
                                                           Network+Attributes                    LANE               65.06 48.89      65.45 52.58                                                 32.18 22.09            55.80 31.84
                                                                                                DANE-O              80.31 59.46      67.33 53.04                                                 34.11 23.07            59.14 35.31
                                                                                                 DANE               79.69 59.32      67.24 52.19                                                 34.52 22.36            57.68 34.87

                                                                    Table 4: Classiﬁcation results (%) comparison of diﬀerent embedding methods.

                                                  Datasets                                      BlogCatalog                          Flickr                                                                  Epinions                               DBLP
                                                  Methods                              AC         Micro Macro                 AC     Micro Macro                                                      AC      Micro Macro               AC          Micro Macro
                                                                    Deepwalk          68.05        67.15   68.18             60.08   58.93  59.08                                                    22.12    17.43   20.10            74.38        69.65 72.37
               Network                                                LINE            70.20        69.88   70.91             61.03   60.90  60.01                                                    23.54    17.17   21.05            72.97        67.56 70.97
                                                                    DANE-N            66.97        66.06   67.78             49.37   47.82  49.34                                                    21.25    20.57   21.88            71.99        65.33 71.94
    Attributes                                                      DANE-A            80.23        79.86   80.23             76.66   75.59  76.60                                                    23.76    21.57   22.00            63.92        54.80 62.97
                                                                      CCA             48.63        49.96   49.63             27.09   26.54  26.09                                                    11.53    9.43    10.56            45.67        42.08 43.83
                                                                     LCMF             84.41        89.01   89.26             66.27   66.75  65.71                                                    19.14    9.22    10.14            69.71        68.01 68.42
Network+Attributes                                                   LANE             87.52        87.52   87.93             77.54   77.81  77.26                                                    27.74    28.45   28.87            72.15        71.09 73.48
                                                                    DANE-O            89.34        89.15   89.23             79.68   79.52  79.95                                                    31.23    31.28   31.35            77.21        74.96 75.48
                                                                     DANE             89.09        88.78   88.94             79.56   78.94  79.56                                                    30.87    30.93   30.81            76.64        74.53 75.69

                                                          Running Time Comparison on BlogCatalog                                                                                             Running Time Comparison on Flickr




        Cumulative Running Time(s)                                                                                                        Cumulative Running Time(s)
                                                  CCA                                                                                                                               CCA
                                                  LCMF                                                                                                                              LCMF
                                                  LANE                                                                                                                              LANE
                                          4
                                                  DANE-O                                                                                                                            DANE-O
                                     10           DANE                                                                                                                      4       DANE
                                                                                                                                                                       10



                                          3
                                     10
                                                                                                                                                                            3
                                                                                                                                                                       10


                                          2
                                     10

                                              1       2         3     4       5        6          7        8        9   10                                                      1       2        3       4      5        6        7        8        9    10
                                                                            Time Steps                                                                                                                        Time Steps
                                                                          (a) BlogCatalog                                                                                                                     (b) Flickr

                                                           Running Time Comparison on Epinions                                                                                               Running Time Comparison on DBLP
                                                                                                                                                                                                                                                CCA




    Cumulative Running Time(s)                                                                                                        Cumulative Running Time(s)
                                                  CCA
                                                  LCMF                                                                                                                                                                                          LCMF
                                                  LANE                                                                                                                                                                                          LANE
                                                  DANE-O                                                                                                                                                                                        DANE-O
                                                  DANE                                                                                                                                                                                          DANE

                                      4
                                 10                                                                                                                                     4
                                                                                                                                                                   10




                                      3
                                 10
                                                  2         4         6           8        10         12       14       16                                                          2        4           6          8        10       12       14        16
                                                                            Time Steps                                                                                                                        Time Steps
                                                                           (c) Epinions                                                                                                                       (d) DBLP

                                                                                                Figure 2: Cumulative running time comparison.
                                 Speedup of DANE on BlogCatalog                                              Speedup of DANE on Flickr
                                                                     DANE                    10                                               DANE
                   8
                                                                     DANE-O                                                                   DANE-O
                                                                                              8
                   6


         Speedup                                                                   Speedup
                                                                                              6
                   4
                                                                                              4
                   2
                                                                                              2

                   0                                                                          0
                       10   20    30     40   50   60    70     80    90   100                    10   20   30     40   50   60    70    80    90   100
                                       Embedding Dimension                                                       Embedding Dimension
                                          (a) BlogCatalog                                                             (b) Flickr

                                  Speedup of DANE on Epinions                                                Speedup of DANE on DBLP
                                                                                            14
                                                                     DANE                                                                     DANE
                  8                                                                         12
                                                                     DANE-O                                                                   DANE-O
                                                                                            10
                  6


        Speedup                                                                   Speedup
                                                                                             8
                  4                                                                          6

                                                                                             4
                  2
                                                                                             2
                  0                                                                          0
                       10   20    30     40   50   60    70     80    90   100                    10   20   30     40   50   60    70    80    90   100
                                       Embedding Dimension                                                       Embedding Dimension
                                           (c) Epinions                                                              (d) DBLP

                                       Figure 3: Running time speedup of DANE against its oﬀline version DANE-O.


Similar matrix factorization based methods are proposed in [50, 54].             to perform unsupervised feature selection in a dynamic and con-
Chang et al. [8] used deep learning techniques to learn a joint fea-             nected environment. Zhou et al. [55] investigated the rare category
ture representation for heterogeneous networks. Huang et al. [20]                detection problem on time-evolving graphs. A more detailed re-
studied whether label information can help learn better feature                  view of dynamic network analysis can be referred to [2]. However,
representation in attributed networks. Instead of directly learning              all these methods are distinct from our proposed framework as we
embeddings, another way is to perform unsupervised feature selec-                are the ﬁrst to tackle the problem of attributed network embedding
tion [12, 27, 40]. Nevertheless, all these methods can only handle               in a dynamic environment.
static networks; it is still not clear how to learn embedding rep-
resentations eﬃciently when attributed networks are constantly                   6 CONCLUSIONS AND FUTURE WORK
evolving over time. The problem of attributed network embedding                  The prevalence of attributed networks in many real-world applica-
is also related to but distinct from multi-modality or multi-view                tions presents new challenges for many learning problems because
embedding [23, 49, 53]. In attributed networks, the network struc-               of its natural heterogeneity. In such networks, interactions among
ture is more than a single view of data as it encodes other types of             networked instances tend to evolve gradually, and the associated
rich information, such as connectivity, transitivity, and reciprocity.           attributes also change accordingly. In this paper, we study a novel
   As mentioned above, many real-world networks, especially so-                  problem: how to learn embedding representations for nodes in dy-
cial networks, are not static but are continuously evolving. Hence,              namic attributed networks to enable further learning tasks. In par-
the results of many network mining tasks will become stale and                   ticular, we ﬁrst build an oﬄine model for a consensus embedding
need to be updated to keep freshness. For example, Tong et al. [44]              presentation which could capture node proximity in terms of both
proposed an eﬃcient way to sample columns and/or rows from                       network topology and node attributes. Then in order to capture
the network adjacency matrix to achieve low-rank approximation.                  the evolving nature of attributed network, we present an eﬃcient
In [42], the authors employed the temporal information to analyze                online method to update the embeddings on the ﬂy. Experimental
the multi-mode network when multiple interactions are evolving.                  results on synthetic and real dynamic attributed networks demon-
Ning et al. [32] proposed an incremental approach to perform spec-               strate the eﬃcacy and eﬃciency of the proposed framework.
tral clustering on networks dynamically. Aggarwal and Li [3] pro-                   There are many future research directions. First, in this paper,
posed a random-walk based method to perform dynamic classiﬁ-                     we employ ﬁrst-order matrix perturbation theory to update the
cation in content-based networks. In [9, 10], a fast eigen-tracking              embedding representations in an online fashion. We would like
algorithm is proposed which is essential for many graph mining                   to investigate how the high-order approximations can be applied
algorithms involving adjacency matrix. Li et al. [25] studied how                to the online embedding learning problem. Second, this paper fo-
                                                                                 cuses on online embedding for two diﬀerent data representations;
we also plan to extend the current framework to multi-mode and                         [28] Jundong Li, Liang Wu, Osmar R Zaı̈ane, and Huan Liu. 2017. Toward Personal-
multi-dimensional dynamic networks.                                                         ized Relational Learning. In SDM. SIAM, 444–452.
                                                                                       [29] David Liben-Nowell and Jon Kleinberg. 2007. The Link-Prediction Problem for
                                                                                            Social Networks. JASIST 58, 7 (2007), 1019–1031.
ACKNOWLEDGEMENTS                                                                       [30] Peter V Marsden and Noah E Friedkin. 1993. Network Studies of Social Inﬂuence.
                                                                                            Sociological Methods & Research 22, 1 (1993), 127–151.
This material is based upon work supported by, or in part by, the                      [31] Miller McPherson, Lynn Smith-Lovin, and James M Cook. 2001. Birds of A
National Science Foundation (NSF) grant 1614576, and the Oﬃce                               Feather: Homophily in Social Networks. Annual Review of Sociology (2001),
                                                                                            415–444.
of Naval Research (ONR) grant N00014-16-1-2257.                                        [32] Huazhong Ning, Wei Xu, Yun Chi, Yihong Gong, and Thomas S Huang. 2007.
                                                                                            Incremental Spectral Clustering With Application to Monitoring of Evolving
REFERENCES                                                                                  Blog Communities. In SDM. SIAM, 261–272.
                                                                                       [33] Beresford N Parlett. 1980. The Symmetric Eigenvalue Problem. Vol. 7.
 [1] Lada A Adamic and Bernardo A Huberman. 2000. Power-Law Distribution of            [34] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: Online learn-
     the World Wide Web. Science 287, 5461 (2000), 2115–2115.                               ing of Social Representations. In KDD. ACM, 701–710.
 [2] Charu Aggarwal and Karthik Subbian. 2014. Evolutionary Network Analysis: A        [35] Joseph J Pfeiﬀer III, Sebastian Moreno, Timothy La Fond, Jennifer Neville, and
     Survey. Comput. Surveys 47, 1 (2014), 10.                                              Brian Gallagher. 2014. Attributed Graph Models: Modeling Network Structure
 [3] Charu C Aggarwal and Nan Li. 2011. On Node Classiﬁcation in Dynamic                    with Correlated Attributes. In WWW. ACM, 831–842.
     Content-based Networks. In SDM. SIAM, 355–366.                                    [36] Meng Qu, Jian Tang, Jingbo Shang, Xiang Ren, Min Zhang, and Jiawei Han.
 [4] Nicola Barbieri, Francesco Bonchi, and Giuseppe Manco. 2014. Who to Follow             2017. An Attention-based Collaboration Framework for Multi-View Network
     and Why: Link Prediction with Explanations. In KDD. ACM, 1266–1275.                    Representation Learning. In CIKM. ACM.
 [5] Mikhail Belkin and Partha Niyogi. 2001. Laplacian Eigenmaps and Spectral Tech-    [37] Sam T Roweis and Lawrence K Saul. 2000. Nonlinear Dimensionality Reduction
     niques for Embedding and Clustering. In NIPS. 585–591.                                 by Locally Linear Embedding. Science (2000).
 [6] Smriti Bhagat, Graham Cormode, and S Muthukrishnan. 2011. Node Classiﬁca-         [38] Cosma Rohilla Shalizi and Andrew C Thomas. 2011. Homophily and Contagion
     tion in Social Networks. In Social Network Data Analytics. 115–148.                    are Generically Confounded in Observational Social Network Studies. Sociolog-
 [7] Shaosheng Cao, Wei Lu, and Qiongkai Xu. 2015. Grarep: Learning Graph Rep-              ical Methods & Research 40, 2 (2011), 211–239.
     resentations with Global Structural Information. In CIKM. ACM, 891–900.           [39] Gilbert W Stewart. 1990. Matrix Perturbation Theory. (1990).
 [8] Shiyu Chang, Wei Han, Jiliang Tang, Guo-Jun Qi, Charu C Aggarwal, and             [40] Jiliang Tang and Huan Liu. 2012. Unsupervised Feature Selection for Linked
     Thomas S Huang. 2015. Heterogeneous Network Embedding via Deep Archi-                  Social Media Data. In KDD. ACM, 904–912.
     tectures. In KDD. ACM, 119–128.                                                   [41] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei.
 [9] Chen Chen and Hanghang Tong. 2015. Fast Eigen-Functions Tracking on Dy-                2015. LINE: Large-Scale Information Network Embedding. In WWW. ACM,
     namic Graphs. In SDM. SIAM, 559–567.                                                   1067–1077.
[10] Chen Chen and Hanghang Tong. 2017. On the Eigen-Functions of Dynamic              [42] Lei Tang, Huan Liu, Jianping Zhang, and Zohreh Nazeri. 2008. Community
     Graphs: Fast Tracking and Attribution Algorithms. Statistical Analysis and Data        Evolution in Dynamic Multi-Mode Networks. In KDD. ACM, 677–685.
     Mining: The ASA Data Science Journal 10, 2 (2017), 121–135.                       [43] Joshua B Tenenbaum, Vin De Silva, and John C Langford. 2000. A Global Geo-
[11] Mo Chen, Qiong Yang, and Xiaoou Tang. 2007. Directed Graph Embedding. In               metric Framework for Nonlinear Dimensionality Reduction. Science 290, 5500
     IJCAI. 2707–2712.                                                                      (2000), 2319–2323.
[12] Kewei Cheng, Jundong Li, and Huan Liu. 2017. Unsupervised Feature Selection       [44] Hanghang Tong, Spiros Papadimitriou, Jimeng Sun, Philip S Yu, and Christos
     in Signed Social Networks. In KDD. ACM, 777–786.                                       Faloutsos. 2008. Colibri: Fast Mining of Large Static and Dynamic Graphs. In
[13] Yun Chi, Xiaodan Song, Dengyong Zhou, Koji Hino, and Belle L Tseng. 2007.              KDD. ACM, 686–694.
     Evolutionary Spectral Clustering by Incorporating Temporal Smoothness. In         [45] Ulrike Von Luxburg. 2007. A Tutorial on Spectral Clustering. Statistics and
     KDD. ACM, 153–162.                                                                     Computing 17, 4 (2007), 395–416.
[14] Janez Demšar. 2006. Statistical Comparisons of Classiﬁers over Multiple Data     [46] Daixin Wang, Peng Cui, and Wenwu Zhu. 2016. Structural Deep Network Em-
     Sets. JMLR 7 (2006), 1–30.                                                             bedding. In KDD. ACM, 1225–1234.
[15] Gene H Golub and Charles F Van Loan. 2012. Matrix Computations. Vol. 3.           [47] Dashun Wang, Dino Pedreschi, Chaoming Song, Fosca Giannotti, and Albert-
[16] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable Feature Learning             Laszlo Barabasi. 2011. Human Mobility, Social Ties, and Link Prediction. In
     for Networks. In KDD. ACM, 855–864.                                                    KDD. ACM, 1100–1108.
[17] David R Hardoon, Sandor Szedmak, and John Shawe-Taylor. 2004. Canonical           [48] Xin Wang, Roger Donaldson, Christopher Nell, Peter Gorniak, Martin Ester, and
     Correlation Analysis: An Overview with Application to Learning Methods. Neu-           Jiajun Bu. 2016. Recommending Groups to Users Using User-Group Engagement
     ral Computation 16, 12 (2004), 2639–2664.                                              and Time-Dependent Matrix Factorization.. In AAAI. 1331–1337.
[18] Yuan He, Cheng Wang, and Changjun Jiang. 2017. Modeling Document Net-             [49] Chang Xu, Dacheng Tao, and Chao Xu. 2013. A Survey on Multi-View Learning.
     works with Tree-Averaged Copula Regularization. In WSDM. ACM, 691–699.                 arXiv preprint arXiv:1304.5634 (2013).
[19] Xiao Huang, Jundong Li, and Xia Hu. 2017. Accelerated Attributed Network          [50] Cheng Yang, Zhiyuan Liu, Deli Zhao, Maosong Sun, and Edward Y Chang. 2015.
     Embedding. In SDM. SIAM, 633–641.                                                      Network Representation Learning with Rich Text Information. In IJCAI. 2111–
[20] Xiao Huang, Jundong Li, and Xia Hu. 2017. Label Informed Attributed Network            2117.
     Embedding. In WSDM. ACM, 731–739.                                                 [51] Tianbao Yang, Rong Jin, Yun Chi, and Shenghuo Zhu. 2009. Combining Link
[21] Yann Jacob, Ludovic Denoyer, and Patrick Gallinari. 2014. Learning Latent Rep-         and Content for Community Detection: A Discriminative Approach. In KDD.
     resentations of Nodes for Classifying in Heterogeneous Social Networks. In             ACM, 927–936.
     WSDM. ACM, 373–382.                                                               [52] Zhilin Yang, William Cohen, and Ruslan Salakhudinov. 2016. Revisiting Semi-
[22] Ling Jian, Jundong Li, and Huan Liu. 2017. Toward Online Node Classiﬁcation            Supervised Learning with Graph Embeddings. In ICML. 40–48.
     on Streaming Networks. DMKD (2017), 1–27.                                         [53] Chao Zhang, Keyang Zhang, Quan Yuan, Fangbo Tao, Luming Zhang, Tim Han-
[23] Abhishek Kumar, Piyush Rai, and Hal Daume. 2011. Co-Regularized Multi-view             ratty, and Jiawei Han. 2017. ReAct: Online Multimodal Embedding for Recency-
     Spectral Clustering. In NIPS. 1413–1421.                                               Aware Spatiotemporal Activity Modeling. In SIGIR. ACM, 245–254.
[24] Jundong Li, Harsh Dani, Xia Hu, and Huan Liu. 2017. Radar: Residual Analysis      [54] Daokun Zhang, Jie Yin, Xingquan Zhu, and Chengqi Zhang. 2016. Collective
     for Anomaly Detection in Attributed Networks. In IJCAI. 2152–2158.                     Classiﬁcation via Discriminative Matrix Factorization on Sparsely Labeled Net-
[25] Jundong Li, Xia Hu, Ling Jian, and Huan Liu. 2016. Toward Time-Evolving                works. In CIKM. ACM, 1563–1572.
     Feature Selection on Dynamic Networks. In ICDM. IEEE, 1003–1008.                  [55] Dawei Zhou, Kangyang Wang, Nan Cao, and Jingrui He. 2015. Rare Category
[26] Jundong Li, Xia Hu, Jiliang Tang, and Huan Liu. 2015. Unsupervised Streaming           Detection on Time-Evolving Graphs. In ICDM. IEEE, 1135–1140.
     Feature Selection in Social Media. In CIKM. ACM, 1041–1050.                       [56] Shenghuo Zhu, Kai Yu, Yun Chi, and Yihong Gong. 2007. Combining Content
[27] Jundong Li, Xia Hu, Liang Wu, and Huan Liu. 2016. Robust Unsupervised Fea-             and Link for Classiﬁcation Using Matrix Factorization. In SIGIR. ACM, 487–494.
     ture Selection on Networked Data. In SDM. SIAM, 387–395.

