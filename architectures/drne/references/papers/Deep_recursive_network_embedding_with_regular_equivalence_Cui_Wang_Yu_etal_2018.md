# Deep recursive network embedding with regular equivalence Cui Wang Yu etal 2018

> Source: `Deep_recursive_network_embedding_with_regular_equivalence_Cui_Wang_Yu_etal_2018.pdf`

---

 Deep Recursive Network Embedding with Regular Equivalence
                           Ke Tu∗                                                      Peng Cui                                   Xiao Wang
              Tsinghua University                                              Tsinghua University                         Tsinghua University
          tuk15@mails.tsinghua.edu.cn                                         cuip@tsinghua.edu.cn                   wangxiao007@mail.tsinghua.edu.cn

                                                     Philip S. Yu                                        Wenwu Zhu
                                      University of Illinois at Chicago &                           Tsinghua University
                                          Institute for Data Science,                              wwzhu@tsinghua.edu.cn
                                             Tsinghua University
                                                psyu@uic.edu

ABSTRACT                                                                                      1   INTRODUCTION
Network embedding aims to preserve vertex similarity in an em-                                Network embedding [12, 33–35] has aroused considerable research
bedding space. Existing approaches usually define the similarity by                           interests in recent years. The fundamental problem is how to pre-
direct links or common neighborhoods between nodes, i.e. struc-                               serve the vertex similarity in an embedding space, i.e. if two vertexes
tural equivalence. However, vertexes which reside in different parts                          are structurally similar in the original network, they should have
of the network may have similar roles or positions, i.e. regular                              the similar embedding vectors. There are multiple ways to quantify
equivalence, which is largely ignored by the literature of network                            the similarity of vertexes in a network. The most common one is
embedding. Regular equivalence is defined in a recursive way that                             structural equivalence [18]. Two vertexes are structurally equiva-
two regularly equivalent vertexes have network neighbors which                                lent if they share many of the same network neighbors. Most of
are also regularly equivalent. Accordingly, we propose a new ap-                              previous works on network embedding aim to preserve structural
proach named Deep Recursive Network Embedding (DRNE) to learn                                 equivalence through high-order proximities [33, 35], where net-
network embeddings with regular equivalence. More specifically,                               work neighbors are extended into high-order neighbors, e.g. direct
we propose a layer normalized LSTM to represent each node by                                  neighbors, neighbors-of-neighbors, etc.
aggregating the representations of their neighborhoods in a recur-                                There are, however, many cases in which vertexes have similar
sive way. We theoretically prove that some popular and typical                                roles or occupy similar positions without any common neighbor.
centrality measures which are consistent with regular equivalence                             For example, two mothers have the same pattern of connections
are optimal solutions of our model. This is also demonstrated by                              with a husband and several children. Although the two mothers
empirical results that the learned node representations can well                              are not structurally equivalent if they do not have the same rel-
predict the indexes of regular equivalence and related centrality                             atives, they do share similar roles or positions. These cases lead
scores. Furthermore, the learned node representations can be di-                              us to an extended definition of vertex similarity known as regular
rectly used for end applications like structural role classification                          equivalence. Two vertexes are defined to be regularly equivalent
in networks, and the experimental results show that our method                                if they have network neighbors which are themselves similar (i.e.
can consistently outperform centrality-based methods and other                                regularly equivalent) [30]. It is apparent that regular equivalence
state-of-the-art network embedding methods.                                                   is a relaxation of structural equivalence. Structural equivalence
                                                                                              promises regular equivalence, but the reverse direction does not
KEYWORDS                                                                                      hold. Comparatively, regular equivalence is more flexible and ca-
network embedding, regular equivalence, recurrent neural network                              pable of covering a broad range of network applications related to
                                                                                              structural roles or node importance, but is largely ignored by the
ACM Reference Format:
                                                                                              literature of network embedding.
Ke Tu, Peng Cui, Xiao Wang, Philip S. Yu, and Wenwu Zhu. 2018. Deep
Recursive Network Embedding with Regular Equivalence. In KDD 2018:
                                                                                                  In order to preserve regular equivalence in network embedding,
24th ACM SIGKDD International Conference on Knowledge Discovery & Data                        i.e. two regularly equivalent nodes should have similar embed-
Mining, August 19–23, 2018, London, United Kingdom. ACM, New York, NY,                        dings, a straightforward method is to explicitly calculate the regular
USA, 10 pages. https://doi.org/10.1145/3219819.3220068                                        equivalence of all vertex pairs and require the similarities of node
                                                                                              embeddings to approximate their corresponding regular equiva-
∗ Tsinghua National Laboratory for Information Science and Technology(TNList)
                                                                                              lence. But this is infeasible for large-scale networks due to the high
Permission to make digital or hard copies of all or part of this work for personal or         complexity of calculating regular equivalence. An alternative is to
classroom use is granted without fee provided that copies are not made or distributed         replace regular equivalence into simpler graph theoretic metrics,
for profit or commercial advantage and that copies bear this notice and the full citation     such as centrality measures. Although many centrality measures
on the first page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,       have been designed to characterize the role and importance of a
to post on servers or to redistribute to lists, requires prior specific permission and/or a   vertex, one centrality can only capture a specific aspect of network
fee. Request permissions from permissions@acm.org.
                                                                                              role, making it difficult to learn general and task-independent node
KDD 2018, August 19–23, 2018, London, United Kingdom
© 2018 Association for Computing Machinery.                                                   embeddings. Not to mention that some centrality measures, like
ACM ISBN 978-1-4503-5552-0/18/08. . . $15.00                                                  betweeness centrality, also bear high computational complexity.
https://doi.org/10.1145/3219819.3220068
                                                                           The rest of the paper is structured as follows. In the next section,
                                                                        we briefly survey recent related work. Section 3 presents the pro-
                                                                        posed model in details. Experimental results are presented in section
                                                                        4. Finally, Section 5 concludes the paper with a brief discussion.


                                                                        2   RELATED WORK
Figure 1: A simple graph to illustrate the rationality of why
recursive embedding can preserve regular equivalence. The               Recently, network representation learning [12, 23, 33, 39, 40] arouses
nodes with the same color are regularly equivalent.                     considerable research interests and an elaborate survey can be
                                                                        found in [8]. Most of the existing network embedding methods are
                                                                        developed along the line of preserving observed pair-wise similar-
                                                                        ity and structural equivalence. For example, DeepWalk [28] uses
                                                                        random walk to generate sequences of nodes from a network and
                                                                        exploits a language model to learn node representations by treating
How to effectively and efficiently preserve regular equivalence in      the sequences as sentences. node2vec [12] extends this idea and pro-
network embedding is still an open problem.                             pose a biased second order random walk model. LINE [33] optimizes
    As mentioned, the definition of regular equivalence is recursive.   an objective function which aims to preserve both the pairwise
This enlightens us to learn network embedding in a recursive way,       similarity and structural equivalence of nodes. More macroscopic
i.e. the embedding of one node is aggregated by its neighbors’ em-      structure, the community structure, is incorporated by M-NMF [36]
beddings. In one recursive step (as shown in Figure 1), if nodes        into embedding methods. Furthermore, [35] claims that the under-
3 and 5, 4 and 6, 7 and 8 are regularly equivalent and thus have        lying structure of the network is highly non-linear and propose a
similar embeddings already, then nodes 1 and 2 would have similar       deep auto-encoder model to preserve the first-order and second-
embeddings, leading to their regular equivalence as true. It is based   order proximities of network structure. Besides, some recent works,
upon this idea that we propose a novel Deep Recursive Network           such as [15] and [19], affiliate node attribute into the networks and
Embedding (DRNE) method. More specifically, we transform the            smoothly embed both attribute information and topology structure
neighbors of a node into an ordered sequence, and propose a layer       into a low-dimensional representation. Recently there is a paucity
normalized LSTM (Long Short Term Memory networks) [14] to               of works related to our targeting problem. For example, RolX [13]
aggregate the embeddings of neighbors into the embedding of a           enumerates various hand-crafted structural features for nodes and
targeting node in a non-linear way. We theoretically prove that         finds the more suited basis vector for this joint feature space. Sim-
some popular and typical centrality measures are optimal solutions      ilarly, struc2vec [29] measures structural similarity by defining a
of our model. This is also demonstrated by empirical results that the   certain form of centrality heuristically. The explicit calculation of
learned node representations can well preserve pair-wise regular        pair-wise centrality similarities makes it unscalable. None of these
equivalence and predict the values of multiple centrality measures      methods preserve regular equivalence while learning representa-
for each node. The learned node representations can be directly         tions.
used for end applications like structural role classification in net-       Regular equivalence, as a relaxed notion of structural equiva-
works, and the experimental results show that our method can            lence, can better capture the structural information. REGE [7] and
consistently outperform each single centrality measure, combina-        CATREGE [7] are proposed by searching for an optimal match-
tion of multiple centrality measures and other node representation      ing between the neighbors of the two vertices iteratively. Ver-
learning methods.                                                       texSim [18] uses the recursive method of linear algebra to construct
    It is worthwhile to highlight the following contributions of this   the measures of similarity. But this is infeasible for large-scale
paper:                                                                  networks due to the high complexity of calculating regular equiv-
                                                                        alence. Besides, centrality measures are another way to measure
    • We investigate a novel problem of learning node representa-       the structure information of nodes in networks. A set of central-
      tions with regular equivalence, which is critical for network     ities [3, 20, 21] have been proposed to study how to capture the
      analysis and largely ignored in the literature of network         structural information better. Since each of them only captures
      representation learning.                                          one aspect of structural information, a certain centrality cannot
    • We find an effective and efficient way to incorporate global      well support different networks and applications. In addition, the
      regular equivalence related information into node represen-       hand-crafted manner of designing centrality measures makes them
      tations, and propose a novel deep model DRNE to learn node        less comprehensive to incorporate regular equivalence related in-
      representations by aggregating neighbors’ representations         formation.
      recursively in a non-linear way.                                      In summary, there is still no sound solution to learning node
    • We theoretically prove that the learned node representations      representations with regular equivalence.
      can well preserve pair-wise regular equivalence and reflect
      several popular and typical node centralities. The empirical
      results also show the significant advantages of our method        3   DEEP RECURSIVE NETWORK EMBEDDING
      over centrality measures as well as other network embedding       In this section, we introduce the proposed method Deep Recursive
      methods in structural role classification.                        Network Embedding (DRNE). The framework is shown in Figure 2.
Figure 2: Framework of Deep Recursive Network Embedding (DRNE). (a): Sampling neighborhoods. (b): Sorting neighborhoods
by their degrees. (c): Layer-normalized LSTM to aggregate embeddings of neighboring nodes into the embedding of the target
node. X i is the embedding of node i and LN means layer normalization. (d): A Weakly guided regularizer.


3.1    Notations and Definitions                                         the neighbors of a node have no natural ordering in networks. Here
Given a network G = (V , E), where V is the set of nodes and             we use the degree of nodes as the criterion to sort neighbors into
E ∈ V × V is the set of edges. For a node v ∈ V , N (v) = {u|(v, u) ∈    an ordered sequence, mainly because degree is the most efficient
E} is the set of its neighborhoods. The learned embeddings are           measure for neighbor ordering and degree often plays an important
defined as X ∈ R |V |×k where k is the dimension and Xv ∈ Rk             role in many graph-theoretic measures, especially those related
represents the embedding of node v. We define the degree of node         with structural roles such as PageRank [27] and Katz [25].
v as dv = |N (v)| and function I (x) = 1 if x ≥ 0 otherwise 0. We also        Suppose the embeddings of the ordered neighbors are {X 1 , X 2 ,
give the strict mathematical definition of structural equivalence        ..., X t , ..., XT }. At each time step t, the hidden state ht is a function
and regular equivalence.                                                 of input embedding X t at time t and its previous hidden state ht −1 ,
                                                                         i.e. ht = LST MCell(ht −1 , X t ). When the embedding sequence is
  Definition 3.1 (Structural Equivalence). We denote s(u) = s(v) if      processed by the LSTM Cell recursively from 1 to T , the information
nodes u and v are structurally equivalent. Then s(u) = s(v) if and       of hidden representation ht will be more and more abundant. hT can
only if N (u) = N (v).                                                   be regarded as the aggregating representation of the neighbors. To
  Definition 3.2 (Regular Equivalence). We denote r (u) = r (v) if       learn long-distance correlations in long sequence, The LSTM utilizes
nodes u and v are regularly equivalent. Then r (u) = r (v) if and        gating mechanisms. The forget gate decides what information we
only if {r (i)|i ∈ N (u)} = {r (j)|j ∈ N (u)}.                           are going to throw away from the memory, the input gate along
                                                                         with old memory decides what new information we are going to
3.2    Recursive Embedding                                               store in the memory and output gate decides what we are going
                                                                         to output based on the memory. Specifically, The LSTM transition
According to Definition 3.2, we learn node embeddings in a recur-
                                                                         equation LST MCell is the following:
sive way that the embedding of a target node can be approximated
by the aggregation of its neighbors’ embeddings. Based on this                              ft = σ (Wf · [ht −1 , Xt ] + bf ),                   (2)
notion, we design the following loss function:
                   Õ                                                                        it = σ (Wi · [ht −1 , Xt ] + bi ),                   (3)
            L1 =      ||Xv − Aдд({Xu |u ∈ N (v)})||F2 ,       (1)                          ot = σ (Wo · [ht −1 , Xt ] + bo ),                    (4)
                   v ∈V
                                                                                           C˜t = tanh(WC · [ht −1 , Xt ] + bC ),                 (5)
where Aдд is the aggregating function. In one recursive step, the
learned embedding of a node can preserve the local structure of                            Ct = ft ∗ Ct −1 + it ∗ C˜t ,                          (6)
its neighbors. By updating the learned representations iteratively,                        ht = ot ∗ tanh(Ct ),                                  (7)
the learned node embeddings can incorporate the their structural
information in a global sense, which is consistent with the definition   where σ is the sigmoid function, · and ∗ represent matrix product
of regular equivalence.                                                  and element-wise product respectively, ft , it and ot are forget gate,
    As the underlying structures of real networks are often highly       input gate and output gate respectively and Ct is the cell state. W∗
nonlinear [22], we design a deep model, the layer normalized Long        and b∗ are learned parameters.
Short-Term Memory (ln-LSTM) [2] as the aggregating function.                Besides, in order to avoid the problems of exploding or vanishing
LSTM is known to be effective for modeling sequences. However,           gradients [14] with the long sequences as inputs, we also introduced
Layer Normalization [2]. The layer normalized LSTM makes it               Through Time (BPTT) algorithm [37]. The learning rate α for Adam
invariant to re-scaling all of the summed inputs. It results in much      is initially set to 0.0025 at the beginning of the training.
more stable dynamics. Particularly, it recenters and re-scales the
cell state Ct after Equation 6 using the extra normalization like         3.5     Theoretical Analysis
follows:
                                д                                         Our method is designed according to the recursive definition of reg-
                        Ct′ =      ∗ (Ct − µ t ),                 (8)
                               Σt                                         ular equivalence. It is intuitive that the learned embedding should
                                       q                                  be able to preserve regular equivalence and the empirical results
where µ t = 1/k ki=1 Ct i and Σt = 1/k ki=1 (Ct i − µ t )2 are the
                 Í                             Í
                                                                          in section 4.3 also demonstrate it. As an additional evidence, here
mean and variance of Ct , and д is gain parameter scaling the             we further theoretically prove that the resulted embeddings of our
normalized activation.                                                    method can well reflect several typical and common centrality mea-
                                                                          sures which are closely related with regular equivalence. Without
3.3    Regularization                                                     loss of generality, we ignore the regularizer term in Equation 9
L1 defined in Equation 1 expresses the recursive embedding pro-           which is used for avoiding trivial solution.
cess according to Definition 3.2 without any other constraints. It
                                                                             Theorem 3.3. Degree centrality, eigenvector centrality [5], PageR-
has such a strong expressive power that multiple solutions can be
                                                                          ank centrality [27] are three optimal solutions of our model respec-
derived as long as they satisfy the given recursive process. It is
                                                                          tively.
risky for this model to degenerate to the trivial solution with all
the embeddings being 0. To avoid the trivial solution, we use node            To prove Theorem 3.3, we first prove following lemmas:
degree as the weakly guided information and impose a constraint
that the learned embedding of a node should be able to approxi-              Lemma 3.4. For any computable function, there exists a finite
mate the degree of the node. Accordingly, we design the following         recurrent neural network (RNN) [24] that can compute it.
regularizer:
           Õ                                                                  Proof. This is a direct consequence of Theorem 1 in [32].                  □
   Lr eд =      ∥log(dv + 1) − MLP(Aдд({Xu |u ∈ N (v)}))∥ 2F , (9)
           v ∈V
                                                                              Theorem 3.5. If the centrality C(v) of node v satisfies that C(v) =
where dv is the degree of node v, MLP is single-layer multilayer
                                                                            u ∈N(v) F (u)C(u) and F (v) = f ({F (u), u ∈ N (v)}) where f is any
                                                                          Í
perceptron with rectified linear unit (ReLU) [11] activation which is     computable function, then C(v) is one of the optimal solutions of our
defined as ReLU (x) = max(0, x). In total, we minimize the objective      model.
function by combining reconstruction loss of Equation 1 and the
regularizer of Equation 9:                                                   Proof. For simplicity, we suppose that all the activation function
                         L = L1 + λLr eд ,                        (10)    of LSTM are linear activation. We prove this lemma by proving that
                                                                          there exists a parameter setting {Wf , Wi , Wo , WC , bf , bi , bo , bC }
where the λ is the weight of regularizer. Note that here the degree
                                                                          in Equation 2-5 such that the node embedding Xu = [F (u), C(u)] is
information is not used as supervise information of network em-
                                                                          a fixed point. We directly construct this parameter settings. Sup-
bedding. Instead, it eventually plays an auxiliary role to induce the
                                                                          pose Wa,i donates the i-th row of Wa . With the input of sequence
solution to be far from the trivial solution. Thus, the λ here is quite
                                                                          {[F (u), C(u)], u ∈ N (v)}, set Wf ,2 and Wo,2 as [0, 0], Wi,2 as [1, 0],
a small value.
                                                                          WC,2 as [0, 1], bf ,2 and bo,2 as 1, bi,2 and bC,2 as 0, then we can
   Neighborhood Sampling. In real networks, the degree of nodes
often follow heavy-tailed distribution [9], i.e. a small number of        easily obtain ht,2 = of ,2 ∗ Ct,2 = Ct,2 = ft,2 ∗ Ct −1,2 + it,2 ∗ C̃t,2 =
                                                                          Ct −1,2 + F (t) ∗ C(t). Thus hT ,2 = u ∈N(v) F (u)C(u) = C(v) where
                                                                                                               Í
nodes have very high degree while the majority of nodes have very
small degree. In order to improve the efficiency, we downsample           T is the length of the input sequence. Besides, by Lemma 3.4, there
the neighbors of nodes with large degree before inputing them into        exists a parameter setting {Wf′ , Wi′ , Wo′ , WC′ , bf′ , bi′ , bo′ , bC′ } to ap-
the ln-LSTM. Specifically, we set an upper bound of the number            proximate f . By set Wf ,1 as [Wf′ , 0], Wo,1 as [Wo′ , 0] and so on,
of the neighbors S. If the number of neighbors exceeds the upper          we can obtain that hT ,1 = f ({F (u), u ∈ N (v)}) = F (v). Thus
bound S, we downsample them into S different nodes. Figure 2 (a)          hT = [F (v), C(v)] and the node embedding Xv = [F (v), C(v)] is a
and (b) shows an example on how to sample from neighborhoods.             fixed point. This completes the proof.                                          □
In power-law networks, the nodes with large degree carry more
unique structural information than the common nodes with small               By the definitions of centralities in Table 1 with (F (v), f ({x i })),
degree. Thus we design a biased sampling strategy to keep the             we can easily obtain that degree centrality, eigenvector centrality,
nodes with large degree by setting the sampling probability P(v)          PageRank centrality satisfy the condition of Theorem 3.5, which
of node v being proportional to its degree dv , P(v) ∝ dv .               completes the proof of Theorem 3.3.
                                                                             According to Theorem 3.3, for any graph, there exists such a
3.4    Optimization                                                       parameter setting of our proposed method that the learned em-
To optimize the aforementioned model, the goal is to minimize the         beddings can be one of the three centralities. This demonstrates
overall loss L as a function of the neural network parameters set         the expressive power of our method in capturing different aspects
θ and the embeddings X . Adam [16] is used to optimize these pa-          of network structural information that are related with regular
rameters. The derivatives are estimated using the BackPropagation         equivalence.
                 Table 1: Definition of centralities.                            • Centralities: Four popular centralities, i.e. Closeness cen-
                                                                                   trality [26], Betweenness centrality [4], Eigenvector central-
    Centrality        Definition C(v)           F (v)      f ({x i })              ity [6] and K-core [17], are used to measure node importance.
                                                                                   Besides, We concatenate all of the four previous centralities
                    dv = u ∈N(v) I (du )                 1/( I (x i ))
                         Í                                   Í
      Degree                                    1/dv                               into one vector as another baseline (Combined).
                    1/λ ∗ u ∈N(v) C(u)                                        For our model, we generally set the length of embeddings k as 16,
                         Í
    Eigenvector                                 1/λ         mean
                                                                           the weight of the regularizer λ is 1, and the limited neighborhood
                     u ∈N(v) 1/du ∗ C(u)                 1/( I (x i ))
                    Í                                       Í
    PageRank                                    1/dv                       number S is 300. We will show that our model is not very sensitive
                                                                           to these parameters in Section 4.5.

3.6     Analysis and Discussions                                           4.2     Network Visualization
In this section, we present the out-of-sample extension and the            In this section, we visualize the learned embeddings to intuitively
complexity analysis.                                                       show that our model preserves regular equivalence.
   3.6.1 Out-of-sample Extension. For a newly arrived node v, if its           Visualization on Barbell Network. The graph shown in Fig-
connections to the existing nodes are known, we can directly feed          ure 3(a) consists of two complete graphs K 1 and K 2 connected by a
the embeddings of its neighbors into the aggregating function and          path graph P of length 10. Each complete graph has 10 nodes. {p1 ,
obtain the aggregated representation as the embedding of the node          p2 ,...,p10 } denotes the nodes of path graph P and pi connects pi+1
with Equation 1. The complexity for such a procedure is O(dv k),           for i = 1, 2, ..., 9. p1 connects node b1 from K 1 and p2 connects b2
where k is the length of embeddings and dv is the degree of node v.        from K 2 . As a result of the symmetry of barbell graph, there are
                                                                           many node pairs with identical structural roles. All nodes from
   3.6.2 Complexity Analysis. During the training procedure, for a         {K 1 /{b1 }} ∪ {K 2 /{b2 }} should be regularly equivalent. Besides, all
single node v in each iteration, the time complexity of calculating        node pairs (pi , p11−i ), i = 1, ...10 and (b1 , b2 ) should also be regu-
gradients and updating parameters is O(dv k 2 ), where dv is the           larly equivalent. The barbell graph is illustrated in Figure 3(a) and
degree of node v, and k is the length of embeddings. Due to the            the nodes are regularly equivalent.
sampling process, the degree dv will not exceed the limited number             Figure 3 shows the learned latent representations by different
S. Thus the overall training complexity is O(|V |Sk 2 I ) where I is the   baselines. From the results, we have following observations:
number of iterations. The length of embeddings k is usually set as               • We can see that the nodes from the two complete graph K 1
a small number such as 32, 64, 128. The limited number S is set as                 and K 2 have a large margin in the embedding spaces gen-
300 in this work. And the number of iterations I is normally a small               erated by DeepWalk, LINE and node2vec. DeepWalk and
number but independent with the number of nodes |V |. Therefore,                   LINE fail to preserve the structural importance, which is
the complexity of training procedure is linear to the number of                    natural since they only consider the pair-wise relationships.
nodes |V |.                                                                        Although node2vec could incorporate some local neighbor-
                                                                                   hood information, the algorithm still tend to focus on pair-
4     EXPERIMENTS                                                                  wise relationships.
In this section, We evaluate our method on different benchmarks                  • struc2vec and our proposed model DRNE achieve similar
to prove its efficacy.                                                             results in this task. Both of them place nodes with similar
                                                                                   structural roles close in latent space. However, struc2vec only
4.1     Baselines and Parameter Settings                                           preserves the similarities of local K-neighborhoods which
      • DeepWalk [28]: This algorithm learns node representations                  cannot reflect global structural information. Since the local
        by modeling a stream of short random walks. We set window                  and global structural informations are fuzzy in such a small
        size as 10, walk length as 40 and walks per node as 40.                    symmetric graph, the performance of DRNE and struc2vec
      • LINE [33]: This algorithm learns feature representations                   cannot be differentiated. We will prove it in the next section.
        in two separate phases which preserve the first-order and             Visualization on Karate Network. Karate network network
        second-order proximities separately. The number of nega-           shown in Figure 4(a) is a well-known social network of a university
        tive samples K = 5 and the mini-batch size of the stochastic       karate club. It captures 34 members and 78 pair-wise links between
        gradient descent is set to 1.                                      members who interact outside the club. The learned latent repre-
      • node2vec [12]: Node2vec extends DeepWalk by proposing              sentations are shown in Figure 4. The color of nodes represents the
        a biased second order random walk model, making it more            value of k-core, which measures the spreading efficiency of a node
        flexible when generating the context of a node. The basic          and shows the global position of a node in the networks. The colors
        parameter settings are the same as DeepWalk. We set p as           red, light blue, light yellow, draw blue represent the k-core value
        1 and q as 2 in node2vec to discover more neighborhood             from 1 to 4 respectively.
        information.                                                          From Figure 4, we can see that the results of DeepWalk, LINE
      • struc2vec [29]: Struc2vec learns latent representations for the    and node2vec are similar to those on the barbell graph. They can-
        structural identity of nodes. Due to its high computational        not distinguish the difference of global structures and mix them
        complexity, we use the combination of all optimizations            together. The struc2vec confuses nodes with different k-core val-
        proposed in the paper for large networks.                          ues in the right part of Figure 4(e). It is reasonable since struc2vec
                         (a) barbell graph                      (b) DeepWalk                             (c) LINE




                         (d) node2vec                           (e) struc2vec                           (f) DRNE


Figure 3: Visualization on Barbell graph. (a) Barbell graph. Latent representations in R2 learned by (b) DeepWalk, (c) LINE,
(d)node2vec, (e) struc2vec and (f) our proposed method DRNE.




                         (a) karate graph                       (b) DeepWalk                             (c) LINE




                         (d) node2vec                           (e) struc2vec                           (f) DRNE


Figure 4: Visualization on karate graph. (a) Karate graph. Latent representations in R2 learned by (b) DeepWalk, (c) LINE,
(d)node2vec, (e) struc2vec and (f) DRNE.


preserves only the similarities of local neighborhoods and cannot          gradually. This result demonstrates that our model can preserve
capture the global structural roles. Our proposed model separates          both the local neighborhood information and the global structural
the nodes with equivalent k-core values approximatively. From              roles information of the target networks.
left-top to right-bottom in Figure 4(f), the k-core value increases
Table 2: The MSE value of predicting centralities on Jazz                 Here we set up two tasks, including regular equivalence predic-
dataset (∗10−2 )                                                       tion and centrality score prediction.
                                                                          For the first task, we use a commonly used regular equivalence-
  centrality    closeness    betweenness     eignvector    k-core      based similarity measure method Vertex Similarity in Network
                                                                       (VS) [18] as ground truth. The pairwise similarity of two nodes is
  DeepWalk        0.6016        3.7188        2.1543      13.2755      measured by the inner product of their embeddings. We rank node
     LINE         0.5153        4.3919        1.5072      15.8179      pairs by their similarities and compare the result rank with the rank
  node2vec        1.0489        3.4065        3.9436      39.2156      by VS. Kendall rank correlation coefficient [1] is used to measure
  struc2vec       0.2365       0.25371        1.0544       9.0858      the correlation of these two ranks. It is defined as follows:
    DRNE          0.1909       0.1261         0.5267      5.5683
                                                                                    | {(i, j) |(x i −x j )(yi −y j )>0} |−| {(i, j) |(x i −x j )(yi −y j )<0} |
                                                                       τ (x, y) =                                    n(n−1)/2                                   .
Table 3: The MSE value of predicting centralities on BlogCat-                                                                         (11)
alog dataset (∗10−2 )                                                  The results are shown in Figure 5. We can see that our method out-
                                                                       performs all the baselines. It demonstrates our method can preserve
                                                                       regular equivalence while learning representations.
  centrality    closeness    betweenness     eignvector    k-core          For the second task, we calculate the four mentioned central-
  DeepWalk        0.2982        1.7836        1.1194      19.7016      ities in Section 4.1 as ground truth. Then we randomly hide 20
     LINE         0.3979        1.8425        1.5167      34.9079      percentage of nodes and use the remaining nodes to train a linear
  node2vec        0.3573        1.6958        1.1432      24.1704      regression model to predict the centrality scores based on node
  struc2vec       0.2947        1.6018        1.0445      25.3047      embeddings. After training, we use the regression model to predict
    DRNE          0.1101        0.6676        0.3108      7.7210       the centralities of the held-out nodes. Mean Square Error (MSE) is
                                                                       used to evaluate the performance. The MSE is defined as follows:
                                                                                                                      n
                                                                                                                  1Õ
                                                                                              MSE(Y , Ŷ ) =            (Y − Ŷ )2 ,                        (12)
                                                                                                                  n i=1

                                                                       where Y is the observed value and Ŷ is the predicted value. Since
                                                                       the scales of different centralities are significantly different, we
                                                                       rescale the MSE value by dividing it by the corresponding averaged
                                                                       centrality values of all nodes.
                                                                          The results are shown in Table 2 and Table 3. The observations
                                                                       are illustrated as follows:
                                                                             • Our method achieves significant improvements over the
                                                                               baselines in predicting all the four centralities. It demon-
                                                                               strates that our method has powerful representation ability
                                                                               to preserve structural role related information.
                                                                             • Compared with other baselines, struc2vec achieves higher
                                                                               improvements on small and dense Jazz dataset than on large
                                                                               and sparse BlogCatalog dataset. Because struc2vec can only
Figure 5: Kendall rank correlation coefficient by fitting reg-                 capture the structural information of local neighbors, which
ular equivalence on Jazz and BlogCatalog dataset.                              is not sufficient to describe the regular equivalence of two
                                                                               nodes in a large and sparse network. Comparatively, our
                                                                               DRNE is able to capture global structural information due to
4.3     Regular Equivalence Prediction                                         the recursive embedding procedure. The significant differ-
In this section, we evaluate whether the learned embeddings pre-               ence of DRNE and struc2vec in predicting k-core can fully
serve regular equivalence on two real world datasets, i.e. Jazz [10]           demonstrate this point, considering that k-core is designed
and BlogCatalog [38]:                                                          to reflect the global position of a node in networks.
      • Jazz [10]: This dataset is the collaboration network between
        Jazz musicians in 2003. Each edge denotes that two musicians   4.4     Structural Role Classification
        have played together in a band. It contains 198 nodes and      Classification [31] is a common and important network application.
        2742 edges.                                                    In this structural role based classification task, the labels for nodes
      • BlogCatalog [38]: This is the data set crawled from BlogCat-   are more related to their structural information than to the labels
        alog. BlogCatalog is the social blog directory which manages   of their adjacent nodes. In this section, we evaluate the ability of
        the bloggers and their blogs. The edges denote two blog-       learned embeddings in predicting the structural roles of nodes in a
        gers are friends in BlogCatalog. It contains 10312 nodes and   given network. We compare our model with not only the state-of-
        333983 edges.                                                  the-art network embedding methods but also four popular centrality
                     Deepwalk         node2vec           DRNE          betweenness         kcore            4.5.1 Effect of parameter k. We show how the dimension of em-
                     LINE             struc2vec          closeness     eigenvector         combined
           0.60
                                                                                                         bedding space k affects the performance in Figure 7(a). Since higher
                                                             0.60                                        embedding dimensions can embody more information, the perfor-
           0.55
                                                             0.55                                        mance raises firstly when the number of embedding dimension
           0.50
                                                             0.50                                        increases. Then after the dimension k exceeds 16, the performance

Accuracy
           0.45                                              0.45                                        becomes stable. This demonstrates that our algorithm is insensitive
           0.40                                              0.40                                        to embedding dimensions.
           0.35                                              0.35
           0.30                                              0.30                                            4.5.2 Effect of parameter λ. Figure 7(b) shows how the weight
           0.25                                             0.25                                          of regularizer λ affects the performance. We select the λ from {0.1,
               0.0    0.2       0.4   0.6         0.8   1.0     0.0   0.2    0.4     0.6     0.8      1.0
                                Percentage                                  Percentage
                                                                                                          0.5, 1.0, 1.5, 2.0}. The performance curve is relatively stable, demon-
                                                                                                          strating that our model is not sensitive to this parameter. This also
                                                                                                          support our notion that the degree information is used to avoid
Figure 6: Average accuracy for multi-class node classifica-
                                                                                                          trivial solution, rather than supervise information for learning em-
tion in air-traffic network. Left: Europe air-traffic. Right:
                                                                                                          beddings.
American air-traffic
                                                                                                           4.5.3 Effect of parameter S. The parameter S controls the maxi-
measures, including closeness centrality, betweenness centrality,                                       mum number of neighbors when sampling. As shown in Figure 7(c),
eigenvector centrality and k-core.                                                                      when this parameter increases in early range, the performance be-
   We evaluate these methods in two air-traffic networks, i.e. Eu-                                      comes better since more neighborhoods contains more structural
ropean air-traffic network and American air-traffic network [29].                                       informations. However, the performance drops when the parame-
The nodes represent airports and the edges indicate the existence                                       ter S is too large. The main reason is that the gradients explode or
of direct flights. The total number of landings plus takeoffs or the                                    vanish when the sequences are too long in an LSTM model. The
total number of people that pass the airports can be used to mea-                                       setting of parameter S needs to trade-off between the obtained in-
sure their activities that reflects their structural roles in the flight                                formation and the capacity of optimizer. In practice, we choose the
networks. One of four possible labels is assigned for each airport                                      parameter as 300 so that we can obtain enough information within
by splitting their activity distribution uniformly.                                                     the capacity of the optimizer.
   After obtaining the embeddings of the airports, we train a logis-
tic regression to predict the labels based on the embeddings. We                                         4.6    Scalability
randomly sample 10% to 90% of the nodes as the training samples
                                                                                                        To testify the scalability, we test training time per epoch. The result
and use the left nodes to test the performance. Averaged accuracy is
                                                                                                        is shown in Figure 8. The parameters of struc2vec are set as the
used to evaluate the performance. The results are shown in Figure
                                                                                                        default value with all proposed optimizations reported in [29]. We
6. The curves with solid line indicate network embedding methods
                                                                                                        can observe that the training time of our model scales linearly
and the other curves with dashed line indicate centrality methods.
                                                                                                        with the number of nodes, and the slope of the curve is close to 1
   From the results, we have following observations:
                                                                                                        (the bottom dashed line). struc2vec scales super-linearly (closer to
            • Our model DRNE outperforms all the other baselines on                                     n1.5 ), and the slope of the curve is close to 1.5 (the top dashed line).
              these two datasets. It demonstrates the effectiveness of our                              Our model DRNE is much faster than struc2vec by several order
              proposed method. Moreover, Our model outperforms the                                      of magnitude. This result conforms to the complexity analysis in
              combined centralities, it indicates that DRNE can preserve                                Section 3.6.2 and proves that our proposed methods is linear to the
              more comprehensive regular equivalence related informa-                                   number of nodes and can scale to large graphs.
              tion than manually defined centrality measures.
            • The relative improvement of our method over the best base-
              line struc2vec in American air-traffic network is more ob-                                 5     CONCLUSION
              vious than that in European air-traffic network. Note that                                 In this paper, we propose a novel deep model to learn node repre-
              struc2vec can only preserve the similarity of local K-neighbors,                           sentations with regular equivalence in networks. Assuming that
              and cannot capture the global structural information in the                                the regular equivalence information of a node has been encoded by
              larger American air-traffic network. This demonstrates the                                 the representations of its neighboring nodes, we propose a layer
              importance of preserving global structural information as in                               normalized LSTM model to learn node embeddings recursively. For
              DRNE.                                                                                      a given node, the structural importance information of far-distance
                                                                                                         nodes can be recursively propagated to its neighbor nodes and
4.5               Parameter Sensitivity                                                                  thus can be embedded into its embeddings. Therefore, the learned
In this section, we evaluate the scalability and how parameters                                          node embeddings can reflect their structural information in a global
influence the performance. Especially, we evaluate the effect of the                                     sense, which is consistent with regular equivalence and central-
embedding dimension k, the weight of regularizer λ and the upper                                         ity measures. The empirical results demonstrate that our method
bound of neighborhood size S. For brevity, we report the results in                                      can significantly and consistently outperform the state-of-the-art
classification task with flight datasets.                                                                algorithms.
               0.62                                                                             0.62                                                          0.62
                                                                       europe                          europe                                                                                          europe
                                                                       usa                             usa                                                    0.61                                     usa
               0.60                                                                             0.60                                                          0.60
               0.58                                                                                                                                           0.59


    Accuracy                                                                         Accuracy                                                      Accuracy
                                                                                                0.58
                                                                                                                                                              0.58
               0.56
                                                                                                                                                              0.57
                                                                                                0.56
               0.54                                                                                                                                           0.56
                                                                                                0.54                                                          0.55
               0.52
                                                                                                                                                              0.54
               0.50                                                                             0.52                                                          0.53
                      2            11     20     29      38     47     56       65                       0.5        1.0         1.5        2.0                       0   100    200        300   400      500
                                               Dimension                                                                λ                                                              S
                                               (a) k                                                            (b) λ                                                          (c) S


Figure 7: Parameter Sensitivity w.r.t. the dimension of embeddings k, the regularizer weights λ and the upper bound of the
number of neighbors S.


                                                                                                                         [4] Marc Barthelemy. 2004. Betweenness centrality in large complex networks. The
                                   4.0           DRNE
                                                                                                                             European Physical Journal B-Condensed Matter and Complex Systems 38, 2 (2004),
                                                 struc2vec
                                   3.5                                                                                       163–168.
                                                                                                                         [5] Phillip Bonacich. 2007. Some unique properties of eigenvector centrality. Social
                                   3.0                                                                                       networks 29, 4 (2007), 555–564.
                                                              n1.5                                                       [6] Phillip Bonacich and Paulette Lloyd. 2015. Eigenvector centrality and structural
                                   2.5

                      log10 time
                                                                                                                             zeroes and ones: When is a neighbor not a neighbor? Social Networks 43 (2015),
                                                                                                                             86–90.
                                   2.0                                                                                   [7] Stephen P Borgatti and Martin G Everett. 1993. Two algorithms for computing
                                                                                                                             regular equivalence. Social networks 15, 4 (1993), 361–376.
                                   1.5                                                                                   [8] Peng Cui, Xiao Wang, Jian Pei, and Wenwu Zhu. 2017. A Survey on Network
                                   1.0                                                                                       Embedding. arXiv preprint arXiv:1711.08752 (2017).
                                                                                                                         [9] Young-Ho Eom and Hang-Hyun Jo. 2015. Tail-scope: Using friends to estimate
                                   0.5                                      n1.0                                             heavy tails of degree distributions in large-scale complex networks. Scientific
                                                                                                                             reports 5 (2015).
                                   0.0                                                                                  [10] Pablo M Gleiser and Leon Danon. 2003. Community structure in jazz. Advances
                                                                                                                             in complex systems 6, 04 (2003), 565–573.
                                                                                                                        [11] Xavier Glorot, Antoine Bordes, and Yoshua Bengio. 2011. Deep sparse rectifier
                                        2.0        2.5           3.0          3.5               4.0                          neural networks. In Proceedings of the Fourteenth International Conference on
                                                             log10 nodes                                                     Artificial Intelligence and Statistics. 315–323.
                                                                                                                        [12] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for
                                                                                                                             networks. In Proceedings of the 22nd ACM SIGKDD international conference on
    Figure 8: Average training time of struc2vec and DRNE.                                                                   Knowledge discovery and data mining. ACM, 855–864.
                                                                                                                        [13] Keith Henderson, Brian Gallagher, Tina Eliassi-Rad, Hanghang Tong, Sugato
                                                                                                                             Basu, Leman Akoglu, Danai Koutra, Christos Faloutsos, and Lei Li. 2012. Rolx:
                                                                                                                             structural role extraction & mining in large graphs. In KDD. ACM, 1231–1239.
6          ACKNOWLEDGMENTS                                                                                              [14] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural
This work was supported in part by National Program on Key                                                                   computation 9, 8 (1997), 1735–1780.
                                                                                                                        [15] Xiao Huang, Jundong Li, and Xia Hu. 2017. Label informed attributed network
Basic Research Project No. 2015CB352300, National Natural Science                                                            embedding. In Proceedings of the Tenth ACM International Conference on Web
Foundation of China Major Project No. U1611461; National Natural                                                             Search and Data Mining. ACM, 731–739.
                                                                                                                        [16] Diederik Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimiza-
Science Foundation of China No. 61772304, 61521002, 61531006,                                                                tion. arXiv preprint arXiv:1412.6980 (2014).
61702296; NSF IIS-1526499, IIS-1763325, and CNS-1626432, and                                                            [17] Maksim Kitsak, Lazaros K Gallos, Shlomo Havlin, Fredrik Liljeros, Lev Muchnik,
NSFC 61672313. Thanks for the research fund of Tsinghua-Tencent                                                              H Eugene Stanley, and Hernán A Makse. 2010. Identification of influential
                                                                                                                             spreaders in complex networks. arXiv preprint arXiv:1001.5285 (2010).
Joint Laboratory for Internet Innovation Technology, and the Young                                                      [18] Elizabeth A Leicht, Petter Holme, and Mark EJ Newman. 2006. Vertex similarity
Elite Scientist Sponsorship Program by CAST. All opinions, findings,                                                         in networks. Physical Review E 73, 2 (2006), 026120.
conclusions and recommendations in this paper are those of the                                                          [19] Jundong Li, Harsh Dani, Xia Hu, Jiliang Tang, Yi Chang, and Huan Liu. 2017.
                                                                                                                             Attributed network embedding for learning in a dynamic environment. In Proceed-
authors and do not necessarily reflect the views of the funding                                                              ings of the 2017 ACM on Conference on Information and Knowledge Management.
agencies.                                                                                                                    ACM, 387–396.
                                                                                                                        [20] Jian-Hong Lin, Qiang Guo, Wen-Zhao Dong, Li-Ying Tang, and Jian-Guo Liu.
                                                                                                                             2014. Identifying the node spreading influence with largest k-core values. Physics
REFERENCES                                                                                                                   Letters A 378, 45 (2014), 3279–3284.
[1] Hervé Abdi. 2007. The Kendall rank correlation coefficient. Encyclopedia of                                         [21] Linyuan Lü, Tao Zhou, Qian-Ming Zhang, and H Eugene Stanley. 2016. The
    Measurement and Statistics. Sage, Thousand Oaks, CA (2007), 508–510.                                                     H-index of a network node and its relation to degree and coreness. Nature
[2] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016. Layer normaliza-                                            communications 7 (2016), 10168.
    tion. arXiv preprint arXiv:1607.06450 (2016).                                                                       [22] Dijun Luo, Feiping Nie, Heng Huang, and Chris H Ding. 2011. Cauchy graph em-
[3] Joonhyun Bae and Sangwook Kim. 2014. Identifying and ranking influential                                                 bedding. In Proceedings of the 28th International Conference on Machine Learning
    spreaders in complex networks by neighborhood coreness. Physica A: Statistical                                           (ICML-11). 553–560.
    Mechanics and its Applications 395 (2014), 549–559.                                                                 [23] Jianxin Ma, Peng Cui, and Wenwu Zhu. 2018. DepthLGP: Learning Embeddings
                                                                                                                             of Out-of-Sample Nodes in Dynamic Networks. (2018).
[24] Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Černockỳ, and Sanjeev Khu-        [32] Hava T Siegelmann and Eduardo D Sontag. 1995. On the computational power
     danpur. 2010. Recurrent neural network based language model. In Eleventh                  of neural nets. Journal of computer and system sciences 50, 1 (1995), 132–150.
     Annual Conference of the International Speech Communication Association.             [33] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei.
[25] Eisha Nathan and David A Bader. 2017. A Dynamic Algorithm for Updating Katz               2015. Line: Large-scale information network embedding. In Proceedings of the
     Centrality in Graphs. In Proceedings of the 2017 IEEE/ACM International Conference        24th International Conference on World Wide Web. International World Wide Web
     on Advances in Social Networks Analysis and Mining 2017. ACM, 149–154.                    Conferences Steering Committee, 1067–1077.
[26] Kazuya Okamoto, Wei Chen, and Xiang-Yang Li. 2008. Ranking of closeness              [34] Ke Tu, Peng Cui, Xiao Wang, Fei Wang, and Wenwu Zhu. 2017. Structural Deep
     centrality for large-scale social networks. Lecture Notes in Computer Science 5059        Embedding for Hyper-Networks. arXiv preprint arXiv:1711.10146 (2017).
     (2008), 186–195.                                                                     [35] Daixin Wang, Peng Cui, and Wenwu Zhu. 2016. Structural deep network em-
[27] Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. 1999. The                 bedding. In Proceedings of the 22nd ACM SIGKDD international conference on
     PageRank citation ranking: Bringing order to the web. Technical Report. Stanford          Knowledge discovery and data mining. ACM, 1225–1234.
     InfoLab.                                                                             [36] Xiao Wang, Peng Cui, Jing Wang, Jian Pei, Wenwu Zhu, and Shiqiang Yang. 2017.
[28] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: Online learning           Community Preserving Network Embedding.. In AAAI. 203–209.
     of social representations. In Proceedings of the 20th ACM SIGKDD international       [37] Paul J Werbos. 1990. Backpropagation through time: what it does and how to do
     conference on Knowledge discovery and data mining. ACM, 701–710.                          it. Proc. IEEE 78, 10 (1990), 1550–1560.
[29] Leonardo FR Ribeiro, Pedro HP Saverese, and Daniel R Figueiredo. 2017. struc2vec:    [38] Reza Zafarani and Huan Liu. 2009. Social computing data repository at ASU.
     Learning node representations from structural identity. In Proceedings of the 23rd        (2009).
     ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.          [39] Ziwei Zhang, Peng Cui, Jian Pei, Xiao Wang, and Wenwu Zhu. 2017.
     ACM, 385–394.                                                                             TIMERS: Error-Bounded SVD Restart on Dynamic Networks. arXiv preprint
[30] Ryan A Rossi and Nesreen K Ahmed. 2015. Role discovery in networks. IEEE                  arXiv:1711.09541 (2017).
     Transactions on Knowledge and Data Engineering 27, 4 (2015), 1112–1131.              [40] Dingyuan Zhu, Peng Cui, Ziwei Zhang, Jian Pei, and Wenwu Zhu. 2018. High-
[31] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and         order Proximity Preserved Embedding For Dynamic Networks. IEEE Transactions
     Tina Eliassi-Rad. 2008. Collective classification in network data. AI magazine 29,        on Knowledge and Data Engineering (2018).
     3 (2008), 93.

