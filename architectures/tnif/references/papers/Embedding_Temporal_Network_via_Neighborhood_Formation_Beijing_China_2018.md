# Embedding Temporal Network via Neighborhood Formation Beijing China 2018

> Source: `Embedding_Temporal_Network_via_Neighborhood_Formation_Beijing_China_2018.pdf`

---

Research Track Paper                                                                                            KDD 2018, August 19‒23, 2018, London, United Kingdom




     Embedding Temporal Network via Neighborhood Formation
                         Yuan Zuo                                                   Guannan Liu∗                                              Hao Lin
              School of Economics and                                         School of Economics and                                School of Economics and
                    Management                                                      Management                                              Management
                 Beihang University                                              Beihang University                                     Beihang University
               Beijing 100191, China                                           Beijing 100191, China                                   Beijing 100191, China
               zuoyuan@buaa.edu.cn                                               liugn@buaa.edu.cn                                   linhao2014@buaa.edu.cn

                           Jia Guo                                                   Xiaoqian Hu                                            Junjie Wu†
              School of Economics and                                        School of Economics and                                School of Economics and
                    Management                                                     Management                                              Management
                 Beihang University                                             Beihang University                            Beijing Advanced Innovation Center
               Beijing 100191, China                                          Beijing 100191, China                            for Big Data and Brain Computing
              guojia1608@buaa.edu.cn                                         huxiaoqian@buaa.edu.cn                                    Beihang University
                                                                                                                                      Beijing 100191, China
                                                                                                                                       wujj@buaa.edu.cn
ABSTRACT                                                                                               rate inferred from node embeddings shows excellent predictive
Given the rich real-life applications of network mining as well                                        power of the proposed model.
as the surge of representation learning in recent years, network
embedding has become the focal point of increasing research in-                                        CCS CONCEPTS
terests in both academic and industrial domains. Nevertheless, the                                     • Information systems → Data mining; Network data mod-
complete temporal formation process of networks characterized by                                       els; • Computing methodologies → Dimensionality reduc-
sequential interactive events between nodes has yet seldom been                                        tion and manifold learning;
modeled in the existing studies, which calls for further research on
the so-called temporal network embedding problem. In light of this,                                    KEYWORDS
in this paper, we introduce the concept of neighborhood formation                                      Temporal Network; Network Embedding; Learning Representation;
sequence to describe the evolution of a node, where temporal exci-                                     Hawkes Process
tation effects exist between neighbors in the sequence, and thus we
propose a Hawkes process based Temporal Network Embedding                                              ACM Reference Format:
(HTNE) method. HTNE well integrates the Hawkes process into                                            Yuan Zuo, Guannan Liu, Hao Lin, Jia Guo, Xiaoqian Hu, and Junjie Wu.
network embedding so as to capture the influence of historical                                         2018. Embedding Temporal Network via Neighborhood Formation. In KDD
                                                                                                       ’18: The 24th ACM SIGKDD International Conference on Knowledge Discovery
neighbors on the current neighbors. In particular, the interactions
                                                                                                       & Data Mining, August 19–23, 2018, London, United Kingdom. ACM, New
of low-dimensional vectors are fed into the Hawkes process as base                                     York, NY, USA, 10 pages. https://doi.org/10.1145/3219819.3220054
rate and temporal influence, respectively. In addition, attention
mechanism is also integrated into HTNE to better determine the
influence of historical neighbors on current neighbors of a node. Ex-                                  1    INTRODUCTION
periments on three large-scale real-life networks demonstrate that                                     Network embedding has become a focal point of study in recent
the embeddings learned from the proposed HTNE model achieve                                            years, aiming at representing large-scale networks by mapping
better performance than state-of-the-art methods in various tasks                                      nodes to low-dimensional space [6, 9, 10, 20]. It provides an efficient
including node classification, link prediction, and embedding visu-                                    way to uncover the network structure and perform various network
alization. In particular, temporal recommendation based on arrival                                     mining tasks such as node classification [20], link prediction [9],
∗ Corresponding author
                                                                                                       community detection [5], etc. Recent work on network embedding
† Also with , Beijing Key Laboratory of Emergency Support Simulation Technologies                      methods [9, 20, 22] generally focuses on static network structure by
for City Operations, Beihang University.                                                               considering various contextual information, e.g., the neighbors of a
                                                                                                       node. One non-trivial but often-overlooked assumption underlying
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
                                                                                                       these methods is that the neighbors of a node are unordered; in
for profit or commercial advantage and that copies bear this notice and the full citation              other words, the link formation history is omitted.
on the first page. Copyrights for components of this work owned by others than the                        In reality, however, a network is formed by adding nodes and
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission          edges sequentially, which indeed should be regarded as a dynamic
and/or a fee. Request permissions from permissions@acm.org.                                            process driven by interactive events between a node and its neigh-
KDD ’18, August 19–23, 2018, London, United Kingdom                                                    bors. As a result, the neighborhood of a node is not formed si-
© 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-5552-0/18/08. . . $15.00
                                                                                                       multaneously and the observed snapshot network structure is the
https://doi.org/10.1145/3219819.3220054                                                                accumulation of neighborhood in certain time periods. For example,




                                                                                                2857
Research Track Paper                                                                                  KDD 2018, August 19‒23, 2018, London, United Kingdom




                                     paper A: 2009                                            other neighbor arrival events with node 3. Assume node 1 becomes
                                     paper B: 2010
                                     paper C: 2011                                            a professor after graduation, then he/she can develop some new
                                2    paper D: 2013                                            co-authorships, and the influence from the advisor might vanish
                                                                                              and the newly connected co-authors (e.g., node 5) might further
                      3                                7      paper F: 2016
                                                                                              excite other co-authors (e.g., node 6), as shown in Figure 1c.
      paper B: 2010                      1                                                        Therefore, how nodes connect to their neighbors sequentially
      paper D: 2013
                                                       6      paper E: 2015                   can reveal the dynamic changes, and should be exploited to better
                                                              paper F: 2016
                          4                                                                   represent the network. Though several recent dynamic network
                                                                                              embedding methods [29, 30] have attempted to model the dynamics
            paper C: 2011            5       paper D: 2013
                                             paper E: 2014                                    by segmenting timelines into fixed time windows, the learned em-
                                                                                              beddings are still representations in particular time periods without
               (a) The ego co-author temporal network
                                                                                              taking the dynamic process into account. To directly model neigh-
                                                                                              borhood formation sequences, therefore, remains a great challenge.
        1   : 2       2     3       2 4        2 3 5             5     6      6   7               To tackle the above challenge, in this paper, we propose a Hawkes
                                                                                              process based Temporal Network Embedding (HTNE) method. Specif-
               (b) The neighborhood forma9on sequence                                         ically, we firstly induce the neighborhood formation sequence from
                                                                                              the network structure driven by sequential events. Since Hawkes
                                                     node 3                                   process [11] well captures the exciting effects between sequential
                                                                     node 5
                                      node 2
                                                                                              events, particularly the influence of history on the current events,
                                                                              node 6          we adapt it for modeling the neighborhood formation process. Then,
                                                                                              in order to derive the node embeddings from the Hawkes process,
                                                                                              low-dimensional vectors are fed into the Hawkes process by map-
      (c) The arrival rate of several target neighbours in the sequence
                                                                                              ping the pairwise vectors to the base rate and the influence from the
                                                                                              history, respectively. Moreover, the influence of historical neighbors
Figure 1: Toy example for temporal network and neighbor-                                      on the current neighbor formation can vary with different nodes,
hood formation sequence.                                                                      and thus we further adopt attention mechanism to enhance the
                                                                                              expressiveness of the influence from the neighborhood formation
                                                                                              history on the current neighbor formation event.
Figure 1a shows the ego network of one author: node 1, and his/her                                In order to deal with large-scale networks, our HTNE model
neighbors i.e., nodes 2 to 6. Taking a snapshot perspective on the                            is solved by optimizing the likelihood of neighborhood formation
network structure, we only observe the up-to-date co-authorship,                              sequences rather than the conditional intensity function. We con-
whereas how and when the nodes are connected remains unknown.                                 duct extensive experiments on three large-scale real-life temporal
As a matter of fact, in most real networks, edges between nodes                               networks to train the node embeddings and apply them for several
are generally established by sequential events, which constitute                              interesting tasks including node classification, link prediction and
the so-called temporal network [12]. For example, the co-author                               visualization. In particular, we design a temporal recommendation
network is driven by co-authored papers with clear timestamps.                                experiment by utilizing the conditional intensity function inferred
As shown in Figure 1a, we see each edge is annotated with several                             from neighborhood formation sequence. The experimental results
papers co-authored between node 1 and its neighbors in chronolog-                             all show significant improvements over some state-of-the-art base-
ical order. The ego temporal network can thus be unfolded into a                              line methods.
node-specific neighbor sequence according to the timing of events,
which is defined as Neighborhood Formation Sequence and shown                                 2 PRELIMINARIES
in Figure 1b.
   Neighborhood formation sequences indeed contain much richer                                2.1 Neighborhood Formation Sequence
information than the static network snapshot in representing nodes.                           Network formation can be viewed as a dynamic process of adding
We can see from Figure 1b that neighbors might appear repeatedly                              nodes and edges, which encodes the underlying mechanisms of
in the sequence due to the repeated co-authorship between the                                 how nodes connect with each other and evolve in the network.
authors, which could provide more semantic meanings than one                                  The static network snapshot is indeed accumulation of historical
single edge. We can also observe the dynamic changes of neighbors                             formation process and only represents the structure at one par-
more explicitly in the sequence that node 1 is more likely to co-                             ticular time period. Therefore, it is more desirable to consider the
author with nodes 2 and 3 in the earlier years while shifts to co-                            detail historical network formation process in order to recover the
author with 5, 6, and 7 recently. Moreover, the events of the target                          network structure and represent the network. However, most prior
neighbors in the sequence are correlated with each other, or in other                         studies in network embedding often focus on network snapshot
words, historical events can influence the current neighborhood                               without resorting to how the network is formed, and most network
formation. For example, we assume node 1 is a Ph.D. student in the                            embedding approaches are based solely on the static neighborhood
earlier years, and hence most of his/her papers are co-authored with                          in representing a node [9, 20, 22].
his/her advisor, e.g., node 2. Node 3 might be an academic friend of                             Traditionally, dynamic network attempts to capture the evolving
node 2, and therefore the co-authorship with node 2 can excite some                           network structures based on predefined time windows [29, 30],




                                                                                       2858
Research Track Paper                                                                            KDD 2018, August 19‒23, 2018, London, United Kingdom




which however, can only represent the snapshots in different time                      co-authors may have more AI people. Therefore in this paper, we
periods and cannot reveal the complete temporal process of net-                        aim to take the dynamic neighborhood formation sequence into
work formation. Therefore in this paper, we trace back the network                     consideration, in order to learn the representations of the nodes.
formation process by tracking the neighborhood formation of each
node. As a matter of fact, edges are generally formed by sequen-
                                                                                       2.2    Problem Definition
tial interactive events between pairwise nodes. For example, in
co-author network, the relationship between authors are formed                         Given a large-scale temporal network G =< V, E; A >, the neigh-
due to their co-authorship on a paper at certain time. In one net-                     bors and the corresponding chronological events of each node
work snapshot, the relationship between any two authors can only                       x ∈ V can be induced into a neighborhood formation sequence Hx
be represented by one single edge, with the times of papers ever                       by tracking all the timestamped events in which x interacts with
co-authored as weight. Driven by the sequential interactive events                     its neighbors. Then, temporal network embedding aims to learning
on each edge, we can formally define the temporal network.                             a D-dimensional vector to represent each node, which is indeed
                                                                                       learning a mapping function ϕ : V → RD , where D ≪ |V |.
    Definition 2.1. (Temporal Network.) Temporal network is a                              Different from recent network embedding methods where only
network with edges annotated by chronological interactive events                       the static set of neighbors are considered, we tackle the embedding
between nodes, which can be denoted as G =< V, E; A >, where                           problem by firstly modeling the neighborhood formation sequence
V denotes the set of nodes, E denotes the set of edges and A                           and the excitation effects between the neighbors.
denotes the set of events. Each edge (x, y) ∈ E between nodes x
and y is annotated by chronological events , i.e., ax,y = {a 1 →
a 2 → · · · } ⊂ A, where ai denotes an event with timestamp ti .                       3 METHODOLOGY
   Given the temporal network, the evolution of co-authorship
                                                                                       3.1 Hawkes Process
can be more explicitly depicted, which can also provide clues for                      Point process models the discrete sequential events by assuming
predicting future co-authors of a node. Therefore, the adjacent                        that historical events before time t can influence the occurrence
neighbors of a node in the network can be organized as a sequence                      of the current event. Conditional intensity function characterizes
according to the ascending time of the interactive events with the                     the arrival rate of sequential events, which can be defined as the
neighbors, representing the neighborhood formation process. Then,                      number of events occurring in a small time window [t, t + ∆t ) given
we can formally define the neighborhood formation sequence in                          all the historical events H (t ).
brief as follows.
                                                                                                                             E[N (t + ∆t )|Ht ]
   Definition 2.2. (Neighborhood Formation Sequence.) Given                                           λ(t |H (t )) = lim                        .          (1)
a source node in temporal network x ∈ V, the neighborhood of                                                         ∆t →0          ∆t
the node is N (x ) = {yi |i = 1, 2, · · · }, and the edge between the
node and each neighbor is annotated with chronological inter-                            Hawkes process is a typical temporal point process, with the
active events ax,yi . Mathematically, the neighborhood formation                       conditional intensity function defined as follows,
sequence can be represented as a series of target neighbor arrival                                                         Z t
events, i.e., {x : (y1 , t 1 ) → (y2 , t 2 ) → · · · → (yn , tn )}, with each
                                                                                                        λ(t ) = µ (t ) +         κ (t − s)dN (s),          (2)
tuple representing an event that node yi is formed as a neighbor of                                                         −∞
node x at time ti .
   It is worth mentioning that the neighborhood of a node usually                      where µ (t ) is the base intensity of a particular event, showing the
indicates a non-repeated node set. While according to the definition                   spontaneous event arrival rate at time t; κ (·) is a kernel function
of Neighborhood Formation Sequence, each neighbor can appear                           that models the time decay effect of past history on the current
repeatedly in the sequence to represent multiple interactions with                     event, which is usually in the form of an exponential function.
the source node. With the defined sequence, the changes of node                           The conditional intensity function of Hawkes process shows
connections over time can be manifested explicitly, such that the                      that the occurrence of current event does not only depend on the
hidden structure of nodes in network can be inferred from the                          event of last time step, but is also influenced by the historical events
sequence. Take the co-author network as an example again, the                          with time decay effect. Such property is desirable for modeling the
neighborhood formation sequence of a node shows its changes of                         neighborhood formation sequences, because the current neighbor
co-authors, and we can help infer the authors’ research interests.                     formation can be influenced with higher intensity by the more
   Moreover, the events in neighborhood formation sequence are                         recent events, while the events occurring in longer history would
not independent because the historical neighbor formation events                       contribute less to the current occurrence of target neighbors.
can influence the current neighbor formation. For instance, a re-                         In order to handle different types of arrival events, Hawkes pro-
searcher may focus on one particular field such as data mining, and                    cess can be extended to multivariate case where the conditional
thus the co-authors are mainly data mining researchers. But when                       intensity function is designed for each event type as one dimen-
deep learning has become a focal point of study in recent years, we                    sion [14]. The excitation effects are indeed a sum over all the histor-
may gradually observe some AI researchers in his/her co-authors                        ical events with different types, captured by an excitation rate αd,d ′
sequences, and can predict from recent neighborhood sequence                           between dimension d and d ′ . Next, we introduce how to model the
that the author may shift the research interests to AI and the future                  neighborhood formation with multivariate Hawkes process.




                                                                                2859
Research Track Paper                                                                         KDD 2018, August 19‒23, 2018, London, United Kingdom




3.2    Modeling Neighborhood Formation                                              fixed co-authors through time, such that the neighborhood for-
       Sequence via Multivariate Hawkes Process                                     mation sequence remains stable and is more predictable, and the
                                                                                    historical events have larger impacts on the current target node
As discussed previously, when we regard each node as a source,
                                                                                    in this scenario. While some other researchers may change their
a sequence of target neighbors driven by interactive events can
                                                                                    co-authors from time to time, as a result they have varied inten-
be entailed. The neighborhood formation sequence of a node is
                                                                                    sity of affinity with different historical co-authors. Therefore, it is
indeed a counting process, with the current target node influenced
                                                                                    necessary to incorporate such characters of the source nodes in
by the historical events. Thus, it is naturally appealing to apply
                                                                                    modeling the distinct excitation effects α, which is not addressed
Hawkes process to model the neighborhood formation sequence of
                                                                                    in previously proposed conditional intensity function.
the source node x, and the conditional intensity function for the
                                                                                        Following the recent attention based models for neural machine
arrival event of target y in the sequence of x can be formulated as,
                                                                                    translation [2], we define the weights between the source node and
                                      X                                             its historical nodes using a Softmax unit as follows:
                λ̃y |x (t ) = µ x,y +   αh,y κ (t − th ),        (3)
                                    th <t
                                                                                                                 exp(−||ex − eh || 2 )
where µ x,y represents the base rate of the event to form an edge                                      wh,x = P                           .           (6)
                                                                                                                h ′ exp(−||ex − eh ′ || )
                                                                                                                                       2
between x and y, while h is the historical target node in the neigh-
borhood formation sequence of node x prior to time t. αh,y repre-                      For consistence, we choose negative Euclidean distance function
sents the degree to which a historical neighbor h excites the current               to score the affinity between the source and history node. Therefore,
neighbor y, and the kernel function κ (·) denotes the time decay                    the influence from the historical neighbors on the current target
effect which can be written in the form of an exponential function,                 can be re-formulated as,

                    κ (t − th ) = exp(−δs (t − th )).                 (4)                                         αh,y = wh,x f (eh , ey )                           (7)

To note that the discount rate δ is a source dependent parameter,                   3.4    Model Optimization
which illustrates the fact that for each source node, the histori-
                                                                                    By modeling the neighborhood formation sequences with multivari-
cal neighbor can influence the current neighbor formation with
                                                                                    ate Hawkes process, we can infer the current neighbor formation
different intensity.
                                                                                    events from the conditional intensity. Then, given the neighbor-
   By following the intensity function in Equation (3), the neigh-
                                                                                    hood formation sequence of node x before time t, denoted by Hs (t ),
borhood formation sequence of each source node can be modeled.
                                                                                    the probability of forming connection between x and the target
Next, in order to learn the D-dimensional representations for the
                                                                                    neighbor y at t can be inferred through the conditional intensity as,
nodes in the network, each node is assumed to be represented by a
D-dimensional vector and fed into the intensity function. Specif-                                                                      λy |x (t )
                                                                                                         p(y|x, Hx (t )) = P                             .           (8)
ically, assume that the node embedding of node i is ei , then the                                                                     y ′ λy ′ |x (t )
base rate for connecting source x to target y can be mapped from a                  Then, the log likelihood of neighborhood formation sequences for
function f (·) : RD × RD → R.                                                       all the nodes in the network can be written as,
   Intuitively, the base rate reveals the natural affinity of source                                        X X
node x with target node y. Thus, we use negative squared Euclidean                                 log L =            log p(y|x, Hx (t )).       (9)
distance as a similarity measure to capture the affinity between the                                               x ∈V y ∈Hx
embeddings of node x and y for brevity, i.e., µ x,y = f (ex , ey ) =                   Due to the exp(·) transfer function introduced in Equation (5),
−||ex − ey || 2 . Similarly, in computing the historical influence on               p(y|x, Hx (t )) is actually a Softmax unit applied to λ̃y |x (t ), which
the current node αh,y , we use the same similarity measure αh,y =                   can be optimized approximately via negative sampling [18]. Nega-
f (eh , ey ) = −||eh − ey || 2 .                                                    tive sampling helps us to avoid the summation over the entire set
   As the similarity measure we introduced takes negative value, we                 of nodes in calculating Equation (8), which costs huge computa-
apply an exponential function to transfer the conditional intensity                 tions. According to the degree distribution Pn (v) ∝ dv 3/4 , where
rate to a positive real number, i.e., д : R → R+ , since λy |x (t ) should          dv is the degree for node v, we sample negative nodes which have
take positive value when regarded as a rate per unit time. Then, we                 not occurred in the neighborhood formation sequence. Then the
can define the conditional intensity function for the neighbor as:                  objective function of the edge between a source x and a historical
                                                                                    target node y at time t can be computed as follows,
                    λy |x (t ) = exp(λ̃y |x (t )).             (5)                                                K
                                                                                                                  X
As will be described later, using the exp(·) as transfer function                         log σ (λ̃y |x (t )) +          Ev k ∼Pn (v ) [− log σ (λ̃v k |x (t ))],   (10)
brings us convenience to define and optimize the likelihood.                                                      k =1
                                                                                    where K is the number of negative nodes sampled according to
3.3    Attention for Sequence Formation                                             Pn (v), σ (x ) = 1/(1 + exp(−x )) is the sigmoid function.
Considering the conditional intensity function, the influence from                     In addition, the length of the neighborhood formulation sequence
historical events is decomposed as the affinity between the his-                    influences the computation complexity of λy |x (t ), where nodes
torical nodes with the current target node. Intuitively, the affinity               have long historical lengths. Thus, in the model optimization, we
between the history and the target node should depend on the                        fix the maximum length of history h and only retain the target
source node. For example, some researchers may have relatively                      nodes in the recent sequence.




                                                                             2860
Research Track Paper                                                                          KDD 2018, August 19‒23, 2018, London, United Kingdom




                             Table 1: Data statistics.                                     Table 2: The ten research areas selected from DBLP.

                 # nodes      # static edges   # temporal edges   # classes             Research Area                     Conference
      DBLP        28,085         162,451            236,894          10                 Database                          ICDE, VLDB, SIGMOD
      Yelp       424,450        2,610,143          2,610,143          5                 Data Mining                       KDD, ICDM, SDM, CIKM
      Tmall      577,314        2,992,964          4,807,545          5                 Information Retrieval             SIGIR
                                                                                        Artificial Intelligence           IJCAI, AAAI, ICML, NIPS
                                                                                        Computer Vision                   CVPR, ICCV
   We adopt Stochastic Gradient Descent (SGD) to optimize the                           Theory                            STOC, SODA, COLT
                                                                                        Computational Linguistics         ACL, EMNLP, COLING
objective function in Equation (10). In each iteration, we sample a
                                                                                        Computer Networks                 SIGCOMM, INFOCOM
mini-batch of edges with timestamps and fixed length of recently                        Operating Systems                 SOSP, OSDI
formed neighbors of the source node to update the parameters.                           Programming Languages             POPL

4     EXPERIMENTAL SETUP
We validate the effectiveness of the proposed methods on three                         ComE [5]: This method models community embedding, which
large scale real-world networks. Hereinafter, we use “HTNE-a” to                     can be utilized to optimize the node embeddings by introducing a
denote the HTNE with attention. Four state-of-the-art baseline                       community-aware high-order proximity.
methods are included for a thorough comparative study.
                                                                                     4.3     Parameter Settings
4.1      Data Sets                                                                   For our method, we set the mini-batch size, the learning rate of
We first briefly introduce the three real-world networks used in our                 the SGD, and the number of negative samples to be 1000, 0.01, 5
experiments, with data statistics listed in Table 1.                                 respectively. We set the history length as 5, 2 and 2 for DBLP, Yelp
   DBLP: We derive a co-author network from DBLP1 of ten re-                         and Tmall respectively. For LINE, we set the number of total edge
search areas (see Table 2). We treat the research areas as labels, and               samples to be 10 billion, and other parameters are set by default.
assume that a researcher belongs to a particular area if over half of                For other baseline methods, we apply default parameters except for
his or her most recent ten papers were published in corresponding                    the embedding size, which is fixed to be 128 for all the methods.
conferences.
   Yelp: This dataset is extracted from the Yelp2 Challenge Dataset.                 4.4     Tasks and Evaluation Measures
Users and businesses are regarded as nodes, and commenting be-                       We first validate the quality of the learned node embeddings from
haviors are taken as edges. Each business is assigned with one or                    each model by treating them as features for tasks such as node
more categories. We only retain the top five categories during the                   classification and link prediction. Then, by performing a customized
experiments, and the businesses with more than one categories are                    temporal recommendation task, we evaluate the conditional in-
labeled by the top one category.                                                     tensity function λy |x (t ) (see Equation (5)) inferred from the node
   Tmall: This dataset is extracted from the sales data of the “Dou-                 embeddings in our method. We also visualize the node embeddings
ble 11” shopping event in 2014 at Tmall.com 3 . We take users and                    by arranging the network layout on a two-dimensional space. Fi-
items as nodes, purchases as edges. Each item is assigned with                       nally, we perform a parameter sensitivity study. The evaluation
one category. We only retain the five most frequently purchased                      tasks and corresponding measures are described as follows.
categories during the experiments.                                                      Node classification: Given the inferred node embeddings as node
                                                                                     features, we train a classifier and predict the node labels. We use
4.2      Baseline Methods                                                            both Macro-F1 and Micro-F1 as measures.
Following are four network embedding methods applied as base-                           Link prediction: We aim to determine if there is an edge between
lines in our experiments.                                                            two given nodes based on the absolute difference in positions be-
   LINE [22]: This method optimizes node representations by pre-                     tween their corresponding embedding vectors. We apply Macro-F1
serving first-order or second-order proximities for a network. In                    as the measure.
the comparative study, we employ the second-order proximity to                          Temporal Recommendation: Given a test time point t, we first
learn representations.                                                               train the node embeddings on the data in time interval [t ′, t ), then
   DeepWalk [20]: This method first applies random walks to gen-                     recommend possible new connections of a source node x at time t.
erate sequences of nodes from the network, and then uses it as                       We apply Precision@k and Recall@k as measures.
input to the Skip-gram model to learn representations.
   node2vec [9]: This method extends DeepWalk by developing a
                                                                                     5 EXPERIMENTAL RESULTS
biased random walk procedure to explore neighborhood of a node,                      5.1 Evaluation of Node Embeddings
which can strike a balance between local and global properties of a                  As described above, we evaluate the quality of learned representa-
network.                                                                             tions by feeding the representations into the tasks including node
1 http://dblp.uni-trier.de                                                           classification and link prediction.
2 https://www.yelp.com                                                                  Node classification results. We first apply all the methods on
3 https://tianchi.aliyun.com/datalab/dataSet.htm?id=5
                                                                                     each network to learn its node embeddings, and then train a Logistic




                                                                              2861
Research Track Paper                                                                      KDD 2018, August 19‒23, 2018, London, United Kingdom




                Table 3: Link prediction results.                                5.2    Evaluation of the Conditional Intensity
                                                                                        Function
                          DBLP           Yelp        Tmall                       The conditional intensity function λy |x (t ) (see Equation(5)) indi-
        DeepWalk          0.8126        0.7678       0.7745
                                                                                 cates the arrival rate of target neighbor y given its source node
        LINE              0.6350        0.8529       0.8265
                                                                                 x, time t and history Hx (t ). Loosely speaking, under the tempo-
        node2vec          0.8049        0.7712       0.5901
        ComE              0.7921        0.8120       0.6917
                                                                                 ral network scenario, λy |x (t ) can be viewed as the possibility of y
        HTNE              0.8521        0.8944       0.7834                      being connected to x at t, which can be exploited to recommend
        HTNE-a            0.8608        0.8861       0.7928                      the future neighbors of the node. Therefore, we design a temporal
                                                                                 recommendation task on DBLP co-author network.
                                                                                     We first extract a co-author network from DBLP data in the time
                                                                                 interval [t ′, t ), and fit each model to that network. Then, given an
Regression classifier with node embeddings as features. We vary the              author, we can apply the fitted model to predict his or her top-k
size of the training set from 10% to 90% and the remaining nodes                 possible co-authors at time t. Since baseline methods purely learn
as testing. We repeat each classification experiment for ten times               node embeddings, we take the inner product of two researchers’
and report the average performance in terms of both Macro-F1 and                 embeddings as the ranking score. While in our method, we directly
Micro-F1 scores. Results on DBLP, Yelp and Tmall are presented in                apply the λy |x (t ) as the ranking score for researcher x and y. Specif-
Table 4, 5 and 6 respectively.                                                   ically, we set t to be the year of 2017, and set t ′ to vary from the year
    As the classification results show, our methods perform the best             of 2012 to 2016, i.e., the time span of the training set varies from 1 to
on all the three datasets. Specifically, HTNE-a performs the best on             5. We apply the Precision@k and Recall@k as evaluation measures.
DBLP and Yelp consistently with all varying sizes of training data, as           Besides, to make the results under different time spans more compa-
measured by both Macro-F1 and Micro-F1. HTNE performs the best                   rable, we only recommend co-authors to the researchers that occur
on Tmall with all varying sizes of training data according to Micro-             in every single year from 2012 to 2017, and the recommendation
F1, and performs the best on Tmall when the training size is larger              results are displayed in Figure 2, where k = 5, 10.
than 20% as measured by Macro-F1. The stable performances of                         From the results we can see HTNE-a consistently outperforms
our methods against different training sizes indicate the robustness             all the baseline methods. As shown in Figure 2, the performance of
of our learned node embeddings when served as features for node                  DeepWalk decreases rapidly with the increasing of time span, and
classification.                                                                  becomes the worst when the time span is larger than 3. Though
    HTNE-a performs better than HTNE on DBLP and Yelp as mea-                    the performance of node2vec decreases slowly, the performances
sured by Macro-F1 and Micro-F1, which indicates that attention                   remain at a worse level. In most cases, LINE performs the second
to history nodes based on the source node could help to learn bet-               best. However, the Precision@10 and Recall@10 of LINE decrease
ter node embeddings. It is notable that HTNE-a performs slightly                 more rapidly than HTNE-a, which indicates that the modeling of
worse than HTNE on Tmall, which we believe is due to the fact                    neighborhood formation helps HTNE-a less influenced by out-of-
that purchase behaviors of a user in short term may have less sig-               date temporal patterns. The trend of ComE is very similar to that of
nificant temporal patterns as compared to that in long term. More                HTNE-a, which indicates that the higher order network structure,
results in Figure 4c can serve as evidence for the above discussions,            i.e., community structure, can prevent ComE from being severely
since when history length is larger than 2, HTNE-a can outperform                influenced by out-of-date co-author patterns. Nevertheless, the
HTNE on Tmall.                                                                   performance of ComE is less competitive against our proposed
    Link prediction results. Given an edge and its two ends x and                HTNE-a. As witnessed in Figure 2, we can see that when the time
y, we define the edge’s representation as |ex − ey |, where ex and ey            span increases, the performances of all the methods decrease, which
are embeddings of x and y respectively. The above definition works               is indeed counterfactual at first glance. Because in most cases, a
for any pair of nodes, no matter an edge exists or not between the               larger time span with more training data should generally provide
nodes, which can be utilized as features for link prediction. On each            a better performance; while the experimental results prove to be
dataset, we randomly hold out 10,000 edges as positive ones, and                 on the contrary, but this result exactly confirms the toy example
also choose 10,000 false ones (i.e., two nodes share no link). We                described in Section 1 that the co-authorship of a researcher may
train a Logistic Regression classifier on the constructed datasets, and          evolve with time, such that recent data is more meaningful for
list the Macro-F1 results in Table 3.                                            recommendation.
    From the results, we can find our methods perform the best on
DBLP and Yelp, and HTNE-a performs the second best on Tmall.
                                                                                 5.3    Network Visualization
The above promising results suggest that the node embeddings
learned by our methods can also serve as favorable features for                  Network visualization is an effective approach to qualitatively eval-
link prediction. We also notice that LINE performs the best on                   uate node embeddings learned by different methods. Here, we em-
Tmall, which might be due to the characteristics of the dataset.                 ploy the t-SNE method [24] to project embeddings of researchers
Moreover, LINE performs the best among the baseline methods on                   to a 2-dimensional space on the DBLP data. Those researchers are
Yelp and Tmall but performs the worst on DBLP, while in contrast,                sampled from three different research areas namely data mining,
our methods achieve satisfactory results in all the dataset, showing             computer vision and computer networks. For each area, we ran-
that our methods are more robust.                                                domly choose 500 researchers.




                                                                          2862
Research Track Paper                                                  KDD 2018, August 19‒23, 2018, London, United Kingdom




                                    Table 4: Node classification results on DBLP.


   Metric       Method      10%       20%        30%        40%        50%           60%    70%        80%       90%
                DeepWalk   0.6345   0.6553     0.6635     0.6681     0.6698      0.6721    0.6734    0.6725     0.6745
                node2vec   0.6332   0.6511     0.6589     0.6631     0.6655      0.6667    0.6670    0.6639     0.6660
                LINE       0.6163   0.6350     0.6415     0.6455     0.6474      0.6489    0.6498    0.6466     0.6490
   Macro-F1
                ComE       0.6508   0.6632     0.6680     0.6718     0.6753      0.6764    0.6794    0.6769     0.6791
                HTNE       0.6235   0.6409     0.6490     0.6526     0.6564      0.6592    0.6596    0.6570     0.6608
                HTNE-a     0.6528   0.6656     0.6729     0.6768     0.6799      0.6824    0.6854    0.6836     0.6844
                DeepWalk   0.6435   0.6604     0.6662     0.6690     0.6700      0.6711    0.6711    0.6709     0.6719
                node2vec   0.6492   0.6626     0.6688     0.6717     0.6736      0.6742    0.6742    0.6730     0.6735
                LINE       0.6229   0.6371     0.6433     0.6455     0.6476      0.6484    0.6487    0.6470     0.6463
   Micro-F1
                ComE       0.6608   0.6723     0.6758     0.6782     0.6806      0.6810    0.6816    0.6801     0.6816
                HTNE       0.6620   0.6693     0.6737     0.6752     0.6778      0.6797    0.6793    0.6777     0.6784
                HTNE-a     0.6706   0.6789     0.6834     0.6853     0.6869      0.6881    0.6883    0.6879     0.6866

                                    Table 5: Node classification results on Yelp.


   Metric       Method      10%       20%        30%        40%        50%           60%    70%        80%       90%
                DeepWalk   0.3739   0.3776     0.3786     0.3800     0.3791      0.3809    0.3811    0.3807     0.3796
                node2vec   0.4086   0.4193     0.4248     0.4271     0.4269      0.4287    0.4282    0.4278     0.4299
                LINE       0.4097   0.4161     0.4191     0.4188     0.4188      0.4185    0.4188    0.4188     0.4186
   Macro-F1
                ComE       0.4187   0.4284     0.4358     0.4372     0.4373      0.4375    0.4387    0.4384     0.4401
                HTNE       0.3627   0.3803     0.3892     0.3943     0.3967      0.3997    0.4006    0.3990     0.3993
                HTNE-a     0.4211   0.4348     0.4421     0.4460     0.4485      0.4508    0.4511    0.4507     0.4487
                DeepWalk   0.5361   0.5461     0.5488     0.5505     0.5510      0.5513    0.5516    0.5520     0.5515
                node2vec   0.5488   0.5600     0.5641     0.5662     0.5663      0.5670    0.5672    0.5677     0.5704
                LINE       0.5456   0.5573     0.5611     0.5628     0.5628      0.5632    0.5641    0.5643     0.5635
   Micro-F1
                ComE       0.5589   0.5672     0.5717     0.5727     0.5730      0.5733    0.5745    0.5749     0.5762
                HTNE       0.5569   0.5644     0.5684     0.5708     0.5716      0.5733    0.5741    0.5734     0.5728
                HTNE-a     0.5834   0.5901     0.5941     0.5961     0.5974      0.5988    0.5989    0.5983     0.5971

                                    Table 6: Node classification results on Tmall.


   Metric       Method      10%       20%        30%        40%        50%           60%    70%        80%       90%
                DeepWalk   0.4862   0.4892     0.4913     0.4922     0.4923      0.4927    0.4939    0.4941     0.4940
                node2vec   0.5298   0.5348     0.5363     0.5377     0.5368      0.5376    0.5386    0.5391     0.5391
                LINE       0.4311   0.4350     0.4364     0.4370     0.4370      0.4369    0.4382    0.4387     0.4384
   Macro-F1
                ComE       0.5373   0.5416     0.5435     0.5442     0.5442      0.5451    0.5465    0.5455     0.5428
                HTNE       0.5292   0.5413     0.5476     0.5511     0.5524      0.5539    0.5559    0.5563     0.5563
                HTNE-a     0.5373   0.5433     0.5468     0.5479     0.5485      0.5491    0.5496    0.5493     0.5507
                DeepWalk   0.5652   0.5704     0.5721     0.5732     0.5736      0.5742    0.5749    0.5759     0.5758
                node2vec   0.5971   0.6025     0.6037     0.6049     0.6046      0.6052    0.6059    0.6068     0.6067
                LINE       0.5285   0.5339     0.5358     0.5365     0.5369      0.5370    0.5377    0.5388     0.5388
   Micro-F1
                ComE       0.6059   0.6100     0.6110     0.6117     0.6119      0.6126    0.6136    0.6131     0.6113
                HTNE       0.6219   0.6286     0.6314     0.6328     0.6332      0.6339    0.6345    0.6352     0.6343
                HTNE-a     0.6194   0.6231     0.6248     0.6251     0.6253      0.6259    0.6259    0.6262     0.6266




                                                        2863
Research Track Paper                                                                                  KDD 2018, August 19‒23, 2018, London, United Kingdom




                            0.35                                                                    0.35


                            0.30                                                                    0.30



            P recision@5                                                                Recall@5
                            0.25                                                                    0.25


                            0.20    DeepWalk        ComE                                            0.20     DeepWalk         ComE
                                    node2vec        DNRL2                                                    node2vec         DNRL2
                                    LINE                                                                     LINE
                            0.15                                                                    0.15
                                1      2         3          4           5                               1        2         3          4        5
                                     Time span of training data                                                Time span of training data

                                        (a) Precision@5                                                            (b) Recall@5
                            0.24


                            0.22                                                                    0.40




            P recision@10
                                                                                        Recall@10
                            0.20
                                                                                                    0.35

                            0.18

                                                                                                    0.30
                                    DeepWalk        ComE                                                     DeepWalk         ComE
                            0.16
                                    node2vec        DNRL2                                                    node2vec         DNRL2
                                    LINE                                                                     LINE
                            0.14                                                                    0.25
                                1      2         3          4           5                               1        2         3          4        5
                                     Time span of training data                                                Time span of training data

                                       (c) Precision@10                                                           (d) Recall@10

                                                          Figure 2: Temporal recommendation results.


   We illustrate the scatter plots of the 1500 researchers in Figure 3,             reducing computation costs. Specifically, as illustrated in Figure 4,
using the color and shape of a node to indicate its research area.                  we report the Macro-F1 of HTNE and HTNE-a on DBLP, Yelp and
Specifically, we use purple triangle to represent “data Mining”, blue               Tmall, with h varying from 1 to 5.
dot to represent “computer network” and green star to represent                        From the results of HTNE, we can see h affects the Macro-F1
“computer vision”. It’s not hard to find that both LINE, DeepWalk                   differently on three datasets. For example, in DBLP and Tmall, the
and node2vec failed to separate all the three areas apart clearly. For              Macro-F1 of HTNE first increases along with h, and then begins
example, as shown in Figure 3a, LINE mixtures the data mining and                   to drop when h > 2. In contrast, the Macro-F1 of HTNE starts to
computer networks areas. Besides, three areas are mixed together                    drop at the beginning. Moreover, we can also find that the attention
in the middle of Figure 3a. By modeling the community embedding,                    mechanism introduced into HTNE helps our method to be more
ComE achieves satisfactory visualizing result among baseline meth-                  robust against different settings of h, as the Macro-F1 of HTNE-a is
ods, as the three areas are roughly separated apart from each other.                stable on Yelp and Tmall, even increases along with h on DBLP.
However, there is no clear margin between the areas. Both of our
methods can clearly separate three areas apart, while the one with
attention achieves a larger margin. Above results indicate that by                  6       RELATED WORK
modeling nodes’ neighborhood formation sequence, our method                         Network embedding, also known as graph embedding or graph
has potential to be applied to community-level applications such                    representation learning, aims to find a low-dimensional vector
as community detection with good performances.                                      space that can maximumly preserve the original network structural
                                                                                    information and network properties [4]. Conventional network
5.4    Parameter Sensitivity                                                        embedding works have been developed with general dimension re-
In this subsection, we study an important parameter named history                   duction techniques, e.g., by constructing and embedding the affinity
length h, which is designed to truncate the whole history of a source               graph into a low dimensional space [3, 13, 21, 23], or by applying
node at a specific time into a recent sequence with fixed length, for               matrix factorization to find the low-dimensional embedding [1].




                                                                             2864
Research Track Paper                                                                                                    KDD 2018, August 19‒23, 2018, London, United Kingdom




                                  (a) LINE                                                  (b) DeepWalk                                                (c) node2vec




                                  (d) ComE                                                       (e) HTNE                                               (f) HTNE-a

Figure 3: Network visualizations. Color of a node indicates the community of the author. Purple: “data mining”, blue: “com-
puter network”, green: “computer vision”.

                   0.655                                                          0.50                                                        0.60

                                                                                                                                              0.58
                   0.650
                                                                                  0.45
                                                                                                                                              0.56
                   0.645


        Macro-F1                                                       Macro-F1                                                    Macro-F1
                                                                                  0.40                                                        0.54
                   0.640
                                                                                                                                              0.52
                                                                                  0.35
                   0.635                                                                                                                      0.50

                                                                                  0.30                                                        0.48
                   0.630    HTNE                                                          HTNE                                                       HTNE
                            HTNE-a                                                        HTNE-a                                              0.46   HTNE-a
                   0.625                                                          0.25
                        1     2            3          4         5                     1      2           3          4       5                    1      2           3          4   5
                                     History length                                                History length                                             History length

                                  (a) DBLP                                                       (b) Yelp                                                   (c) Tmall

                                                          Figure 4: Impacts of history length on DBLP, Yelp and Tmall.


However, works along this line usually suffer from heavy compu-                                            extended the random walk procedure to a biased version, which can
tational cost or statistical performance drawbacks, making them                                            strike a balance between local and global properties of a network.
neither practical nor effective in large-scale networks.                                                   Tang et al. preserved network structures by approximating first-
   With the advent of deep learning methods, significant efforts                                           order and second-order proximities in the embedding space [22].
have been devoted to designing neural network-based representa-                                            Moreover, high-order proximities of nodes as well as community
tion learning models. Especially, Mikolov et al. proposed an efficient                                     structures have also been taken into consideration in network em-
neural network framework to learn the distributed representations                                          bedding models [25, 28].
of words in natural language [16, 17]. Motivated by this work, Per-                                           More recently, network embedding have been continuously stud-
ozzi et al.[20] utilized random walks to generate sequences of nodes                                       ied and received arising attentions from different perspectives. For
in large-scale network, and then considered the walking path, i.e.,                                        instance, rich auxiliary information has been leveraged to facili-
sequence of nodes, as a sentence of words. Grover and Leskovec [9]                                         tate the embedding. Pan et al. proposed a tri-party deep neural




                                                                                                    2865
Research Track Paper                                                                                  KDD 2018, August 19‒23, 2018, London, United Kingdom




network model, which jointly models node structures, contents                                [3] Mikhail Belkin and Partha Niyogi. 2001. Laplacian Eigenmaps and Spectral
and labels [19]. To address the problem that previous transductive                               Techniques for Embedding and Clustering. In NIPS. MIT Press, Cambridge, MA,
                                                                                                 USA, 585–591.
approaches do not naturally generalize to unseen nodes, Hamilton                             [4] HongYun Cai, Vincent W. Zheng, and Kevin Chen-Chuan Chang. 2017. A Com-
et al. developed an inductive framework that incorporates node                                   prehensive Survey of Graph Embedding: Problems, Techniques and Applications.
                                                                                                 CoRR abs/1709.07604 (2017).
feature information for generating node embeddings [10]. Besides,                            [5] Sandro Cavallari, Vincent W. Zheng, Hongyun Cai, Kevin Chen-Chuan Chang,
the strengths of generative adversarial networks [8] have also been                              and Erik Cambria. 2017. Learning Community Embedding with Community
exploited with network embedding [6, 26]. Nevertheless, most ex-                                 Detection and Node Embedding on Graphs. In CIKM. 377–386.
                                                                                             [6] Quanyu Dai, Qiang Li, Jian Tang, and Dan Wang. 2017. Adversarial Network
isting network embedding techniques mainly focused on the set-                                   Embedding. CoRR abs/1711.07838 (2017).
ting of static networks. To address this issue, Zhu et al. developed                         [7] Nan Du, Yichen Wang, Niao He, and Le Song. 2015. Time-sensitive Recommen-
a dynamic network embedding algorithm based on matrix fac-                                       dation from Recurrent User Activities. In NIPS. 3492–3500.
                                                                                             [8] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-
torization [30]. Yang et al. presented a model with exploring the                                Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative
evolution patterns of triads, which can preserve structural infor-                               Adversarial Nets. In NIPS. MIT Press, Cambridge, MA, USA, 2672–2680.
                                                                                             [9] Aditya Grover and Jure Leskovec. 2016. Node2Vec: Scalable Feature Learning for
mation and get the latent representation vectors for vertices at                                 Networks. In SIGKDD. ACM, New York, NY, USA, 855–864.
different timesteps [29]. The dynamics of these embedding meth-                             [10] William L. Hamilton, Rex Ying, and Jure Leskovec. 2017. Inductive Representation
ods [27, 29, 30] only focus on segmenting the timelines into fixed                               Learning on Large Graphs. CoRR abs/1706.02216 (2017).
                                                                                            [11] Alan G Hawkes. 1971. Spectra of some self-exciting and mutually exciting point
time windows, such that the learned node embeddings are only                                     processes. Biometrika 58, 1 (1971), 83–90.
a representation of the snapshot network. In contrast, our model                            [12] Petter Holme and Jari SaramÃďki. 2012. Temporal networks. Physics Reports 519,
takes the full historical neighborhood formation process into ac-                                3 (2012), 97 – 125.
                                                                                            [13] Joseph B Kruskal and Myron Wish. 1978. Multidimensional Scaling. CRC press.
count, providing a more comprehensive representation in view of                                  875–878 pages.
the history.                                                                                [14] RÃľmi Lemonnier, Kevin Scaman, and Argyris Kalogeratos. 2017. Multivariate
   Moreover, our proposed network embedding methods are based                                    Hawkes Processes for Large-Scale Inference. In AAAI.
                                                                                            [15] Remi Lemonnier and Nicolas Vayatis. 2014. Nonparametric Markovian Learning
on Hawkes process, which is a traditionally powerful temporal                                    of Triggering Kernels for Mutually Exciting and Mutually Inhibiting Multivariate
point process in modeling sequences [11] and has also been studied                               Hawkes Processes. In Machine Learning and Knowledge Discovery in Databases.
                                                                                                 161–176.
extensively to adapt for different scenarios. Particularly, recent stud-                    [16] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient
ies on Hawkes process mainly focus on tackling the challenges of                                 Estimation of Word Representations in Vector Space. CoRR abs/1301.3781 (2013).
scalability by employing the memoryless property [15] or imposing                           [17] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013.
                                                                                                 Distributed Representations of Words and Phrases and Their Compositionality.
the low-rank structure on the infectivity matrix [7, 14], etc.                                   In NIPS. Curran Associates Inc., USA, 3111–3119.
                                                                                            [18] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013.
                                                                                                 Distributed Representations of Words and Phrases and their Compositionality.
7   CONCLUSIONS                                                                                  In NIPS. 3111–3119.
                                                                                            [19] Shirui Pan, Jia Wu, Xingquan Zhu, Chengqi Zhang, and Yang Wang. 2016. Tri-
In this paper, we propose a Hawkes process based Temporal Net-                                   party Deep Network Representation. In IJCAI. AAAI Press, 1895–1901.
work embedding (HTNE) method. By formulating the neighbor-                                  [20] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. DeepWalk: Online Learn-
hood formation sequence of a temporal network as Hawkes process,                                 ing of Social Representations. In SIGKDD. ACM, New York, NY, USA, 701–710.
                                                                                            [21] Sam T. Roweis and Lawrence K. Saul. 2000. Nonlinear Dimensionality Reduction
HTNE achieves the learning of node embedding, and capturing the                                  by Locally Linear Embedding. Science 290, 5500 (2000), 2323–2326.
influence of the historical neighbors on the current neighbor for-                          [22] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei. 2015.
mation simultaneously. By plugging an attention mechanism in                                     LINE: Large-scale Information Network Embedding. In WWW. International
                                                                                                 World Wide Web Conferences Steering Committee, Republic and Canton of
the influence rate of Hawkes process, HTNE gains ability to de-                                  Geneva, Switzerland, 1067–1077.
cide which parts of the historical neighbor are more influential.                           [23] Joshua B. Tenenbaum, Vin de Silva, and John C. Langford. 2000. A Global
                                                                                                 Geometric Framework for Nonlinear Dimensionality Reduction. Science 290,
Extensive experiments on three large scale real-world networks                                   5500 (2000), 2319–2323.
demonstrate the superiority of our methods to leading network em-                           [24] Laurens van der Maaten and Geoffrey E. Hinton. 2008. Visualizing High-
bedding methods. Future work includes integrating the attributes                                 Dimensional Data Using t-SNE. JMLR 9 (2008), 2579–2605.
                                                                                            [25] Daixin Wang, Peng Cui, and Wenwu Zhu. 2016. Structural Deep Network Em-
of the temporal edges into our model, and seeking the potential of                               bedding. In SIGKDD. ACM, New York, NY, USA, 1225–1234.
our formulation in solving the optimization problem of Hawkes                               [26] Hongwei Wang, Jia Wang, Jialin Wang, Miao Zhao, Weinan Zhang, Fuzheng
process with large scale event types.                                                            Zhang, Xing Xie, and Minyi Guo. 2017. GraphGAN: Graph Representation
                                                                                                 Learning with Generative Adversarial Nets. CoRR abs/1711.08267 (2017).
                                                                                            [27] Jingyuan Wang, Fei Gao, Peng Cui, Chao Li, and Zhang Xiong. 2014. Discovering
8   ACKNOWLEDGMENTS                                                                              urban spatio-temporal structure from time-evolving traffic networks. In Proceed-
                                                                                                 ings of the 16th Asia-Pacific Web Conference. Springer International Publishing,
Dr. Junjie Wu’s work was partially supported by the National Nat-                                93–104.
                                                                                            [28] Xiao Wang, Peng Cui, Jing Wang, Jian Pei, Wenwu Zhu, and Shiqiang Yang. 2017.
ural Science Foundation of China (NSFC) (71531001, U1636210,                                     Community Preserving Network Embedding.
71725002). Dr. Guannan Liu’s work was supported in part by NSFC                             [29] Lekui Zhou, Yang Yang, Xiang Ren, Fei Wu, and Yueting Zhuang. 2018. Dy-
(71701007).                                                                                      namic Network Embedding by Modeling Triadic Closure Process. In The AAAI
                                                                                                 Conference on Artificial Intelligence.
                                                                                            [30] L. Zhu, D. Guo, J. Yin, G. V. Steeg, and A. Galstyan. 2016. Scalable Temporal
REFERENCES                                                                                       Latent Space Inference for Link Prediction in Dynamic Social Networks. IEEE
                                                                                                 Transactions on Knowledge and Data Engineering 28, 10 (Oct 2016), 2765–2777.
[1] Amr Ahmed, Nino Shervashidze, Shravan Narayanamurthy, Vanja Josifovski, and
    Alexander J. Smola. 2013. Distributed Large-scale Natural Graph Factorization.
    In WWW. ACM, New York, NY, USA, 37–48.
[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural ma-
    chine translation by jointly learning to align and translate. arXiv preprint
    arXiv:1409.0473.




                                                                                     2866

