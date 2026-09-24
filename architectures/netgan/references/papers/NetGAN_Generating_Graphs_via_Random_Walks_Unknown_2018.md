# NetGAN Generating Graphs via Random Walks Unknown 2018

> Source: `NetGAN_Generating_Graphs_via_Random_Walks_Unknown_2018.pdf`

---

                                                                   NetGAN: Generating Graphs via Random Walks


                                                       Aleksandar Bojchevski * 1 Oleksandr Shchur * 1 Daniel Zügner * 1 Stephan Günnemann 1


                                                                    Abstract                                  An increasingly popular way to address this issue in other
                                                                                                              fields is by switching from explicit (prescribed) models to
                                               We propose NetGAN – the first implicit genera-                 implicit ones. This transition is especially notable in com-
                                               tive model for graphs able to mimic real-world                 puter vision, where generative adversarial networks (GANs)




arXiv:1803.00816v2 [stat.ML] 1 Jun 2018
                                               networks. We pose the problem of graph genera-                 (Goodfellow et al., 2014) significantly advanced the state
                                               tion as learning the distribution of biased random             of the art over the classic prescribed approaches like mix-
                                               walks over the input graph. The proposed model                 tures of Gaussians (Blanken et al., 2007). GANs achieve
                                               is based on a stochastic neural network that gener-            unparalleled results in scenarios such as image and 3D ob-
                                               ates discrete output samples and is trained using              jects generation (e.g., Karras et al., 2017; Berthelot et al.,
                                               the Wasserstein GAN objective. NetGAN is able                  2017; Wu et al., 2016). However, despite their massive suc-
                                               to produce graphs that exhibit well-known net-                 cess when dealing with real-valued data, adapting GANs
                                               work patterns without explicitly specifying them               to handle discrete objects like graphs or text remains an
                                               in the model definition. At the same time, our                 open research problem (Goodfellow, 2016). In fact, dis-
                                               model exhibits strong generalization properties,               creteness is only one of the obstacles when applying GANs
                                               as highlighted by its competitive link prediction              to network data. Large repositories of graphs that all come
                                               performance, despite not being trained specifi-                from the same distribution are not available. This means
                                               cally for this task. Being the first approach to               that in a typical setting one has to learn from a single graph.
                                               combine both of these desirable properties, Net-               Additionally, any model operating on a graph necessarily
                                               GAN opens exciting avenues for further research.               has to be permutation invariant, as graphs are isomorphic
                                                                                                              under node reordering.
                                                                                                              In this work we introduce NetGAN – the first implicit gener-
                                          1. Introduction
                                                                                                              ative model for graphs and networks that tackles all of the
                                          Generative models for graphs have a longstanding history,           above challenges. We formulate the problem of learning the
                                          with applications including data augmentation, anomaly              graph topology as learning the distribution of biased random
                                          detection and recommendation (Chakrabarti & Faloutsos,              walks over the graph. Like in the typical GAN setting, the
                                          2006). Explicit probabilistic models such as Barabási-Albert       generator G – in our case defined as a stochastic neural
                                          or stochastic blockmodels are the de-facto standard in this         network with discrete output samples – learns to generate
                                          field (Goldenberg et al., 2010). However, it has also been          random walks that are plausible in the real graph, while the
                                          shown on multiple occasions that our intuitions about struc-        discriminator D then has to distinguish them from the true
                                          ture and behavior of graphs may be misleading. For in-              ones that are sampled from the original graph.
                                          stance, heavy-tailed degree distributions in real graphs were
                                                                                                              The main requirement for a graph generative model is the
                                          in strong disagreement with the models existing at the time
                                                                                                              ability to generate realistic graphs. In the experimental
                                          of their discovery (Barabási & Albert, 1999). More recent
                                                                                                              section we compare NetGAN to other established prescribed
                                          works like Dong et al. (2017) and Broido & Clauset (2018)
                                                                                                              models on this task. We observe that our proposed method
                                          keep bringing up other surprising characteristics of real-
                                                                                                              consistently reproduces most known patterns inherent to
                                          world networks that question the validity of the established
                                                                                                              real-world networks without explicitly specifying any of
                                          models. This leads us to the question: “How do we de-
                                                                                                              them in the model definition (e.g., degree distribution, as
                                          fine a model that captures all the essential (potentially still
                                                                                                              seen in Fig. 1). However, a model that simply replicates the
                                          unknown) properties of real graphs?”
                                                                                                              original graph would also trivially fulfill this requirement,
                                            *
                                              Equal contribution 1 Technical University of Munich, Germany.   which clearly isn’t our goal. In order to prove that this
                                          Correspondence to: Daniel Zügner <zuegnerd@in.tum.de>.             is not the case we examine the generalization properties
                                                                                                              of NetGAN by evaluating its link prediction performance.
                                          Proceedings of the 35 th International Conference on Machine        As our experiments show, our model exhibits competitive
                                          Learning, Stockholm, Sweden, PMLR 80, 2018. Copyright 2018
                                          by the author(s).                                                   performance in this task and even achieves state-of-the-art
                                          NetGAN: Generating Graphs via Random Walks


                                                                                                                                  Citeseer
                                                                                                                                  NetGAN




                                                                                        Number of nodes
                                                                                                          102



                                                                                                          101



                                                                                                          100
                                                                                                                100         101              102
                                                                           44% edge
                                                                            overlap                                      Degree

               (a) Original graph                   (b) Graph generated by NetGAN                         (c) Degree distribution comparison

Figure 1: (a) Subgraph of the C ITESEER network and (b) the respective subset of the graph generated by NetGAN. Both
have similar structure but are not identical. (c) shows that the degree distributions of the two graphs are very close.


results on some datasets. This result is especially impressive,     Deep learning methods for graph data have mostly been
since NetGAN is not trained explicitly for performing link          studied in the context of node embeddings (Perozzi et al.,
prediction. To summarize, our main contributions are:               2014; Grover & Leskovec, 2016; Kipf & Welling, 2016).
                                                                    The main idea behind these approaches is that of model-
  • We introduce NetGAN1 – the first of its kind GAN                ing the probabilities of each individual edge’s existence,
    architecture that generates graphs via random walks.            p(Auv ), as some function of the respective node embed-
    Our model tackles the associated challenges of staying          dings, f (hu , hv ), where f is represented by a neural net-
    permutation invariant, learning from a single graph and         work. The recently proposed GraphGAN (Wang et al., 2017)
    generating discrete output.                                     is another instance of such prescribed edge-level probabilis-
                                                                    tic models, where f is optimized using the GAN objective
  • We show that our method preserves important topolog-            instead of the traditional cross-entropy. Deep embedding
    ical properties, without having to explicitly specifying        based methods achieve state-of-the-art scores in tasks like
    them in the model definition. Moreover, we demon-               link prediction and node classification. Nevertheless, as
    strate how latent space interpolation leads to producing        we show in Sec. 3.2, using such approaches for generating
    graphs with smoothly changing characteristics.                  entire graphs produces samples that don’t preserve any of
  • We highlight the generalization properties of NetGAN            the patterns inherent to real-world networks.
    by its link prediction performance that is competitive          Prescribed generative models for graphs have a long his-
    with the state of the art on real-word datasets, despite        tory and are well-studied. For a survey we refer the reader
    the model not being trained explicitly for this task.           to Chakrabarti & Faloutsos (2006) and Goldenberg et al.
                                                                    (2010). Typically, prescribed generative approaches are de-
2. Related Work                                                     signed to capture and reproduce some predefined subset
                                                                    of graph properties (e.g., degree distribution, community
So far, no GAN architectures applicable to real-world net-          structure, clustering coefficient). Notable examples include
works have been proposed. Liu et al. (2017) propose a GAN           the configuration model (Bender & Canfield, 1978; Molloy
architecture for learning topological features of subgraphs.        & Reed, 1995), variants of the degree-corrected stochas-
Tavakoli et al. (2017) apply GANs to graph data by trying to        tic blockmodel (Karrer & Newman, 2011; Bojchevski &
directly generate adjacency matrices. Because their model           Günnemann), Exponential Random Graph Models (Holland
produces the entire adjacency matrix – including the zero           & Leinhardt, 1981), Multiplicative Attribute Graph model
entries – it requires computations and memory quadratic in          (Kim & Leskovec, 2011), and the block two-level Erdős-
the number of nodes. Such quadratic complexity is infeasi-          Réniy random graph model (Seshadhri et al., 2012). In
ble in practice, allowing to process only small graphs, with        Sec. 4 we compare with some of these prescribed models
reported runtime of over 60 hours for a graph with only 154         on the tasks of graph generation and link prediction.
nodes. In contrast, NetGAN operates on random walks – it
considers only the non-zero entries of the adjacency matrix         Due to the challenging nature of the problem, only few ap-
efficiently exploiting the sparsity of real-world graphs – and      proaches able to generate discrete data using GANs exist.
is readily applicable to graphs with thousands of nodes.            Most approaches focus on generating discrete sequences
                                                                    such as text, with some of them using reinforcement learn-
   1
       Code available at: https://www.kdd.in.tum.de/netgan
                                       NetGAN: Generating Graphs via Random Walks

ing techniques to enable backpropagation through sam-             distribution is passed through a parametric function gθ0 to
pling discrete random variables (Yu et al., 2017; Kusner          initialize m0 . The generative process of G is summarized
& Hernández-Lobato, 2016; Li et al., 2017; Liang et al.,         in the box below.
2017). Other approaches modify the GAN objective to
tackle the same challenge (Che et al., 2017; Hjelm et al.,
2017). Focusing on non-sequential discrete data, Choi et al.                                                  z ∼ N (0, I d )
(2017) generate high-dimensional discrete features (e.g. bi-                                                   m0 = gθ0 (z)
nary indicators, counts) in patient records. None of these          v 1 ∼ Cat(σ(p1 )),              (p1 , m1 ) = fθ (m0 , 0)
methods have considered graph structured data.                      v 2 ∼ Cat(σ(p2 )),             (p2 , m2 ) = fθ (m1 , v 1 )
                                                                         ..                                      ..
3. Model                                                                  .                                       .
                                                                    v T ∼ Cat(σ(pT )), (pT , mT ) = fθ (mT −1 , v T −1 )
In this section we introduce NetGAN - a Generative Ad-
versarial Network model for graph / network data. Its core
idea lies in capturing the topology of a graph by learn-          In this work we focus our attention on the Long short-term
ing a distribution over the random walks. Given an input          memory (LSTM) architecture for fθ , introduced by Hochre-
graph of N nodes, defined by a binary adjacency matrix            iter & Schmidhuber (1997). The memory state mt of an
A ∈ {0, 1}N ×N , we first sample a set of random walks of         LSTM is represented by the cell state C t , and the hid-
length T from A. This collection of random walks serves           den state ht . The latent code z goes through two separate
as a training set for our model. We use the biased second-        streams, each consisting of two fully connected layers with
order random walk sampling strategy described in Grover &         tanh activation, and then used to initialize (C 0 , h0 ).
Leskovec (2016), as it better captures both local and global      A natural question might arise: ”Why use a model with
graph structure. An important advantage of using random           memory and temporal dependencies, when the random
walks is their invariance under node reordering. Addition-        walks are Markov processes?” (2nd order Markov for biased
ally, random walks only include the nonzero entries of A,         RWs). Or put differently, what’s the benefit of using random
thus efficiently exploiting the sparsity of real-world graphs.    walks of length greater than 2? In theory, a model with large
Like any typical GAN architecture, NetGAN consists of two         enough capacity could simply memorize all existing edges
main components - a generator G and a discriminator D.            in the graph and recreate them. However, for large graphs
The goal of the generator is to generate synthetic random         achieving this in practice is not feasible. More importantly,
walks that are plausible in the input graph. At the same time,    pure memorization is not the goal of NetGAN, rather we
the discriminator learns to distinguish the synthetic random      want to have generalization and to generate graphs with
walks from the real ones that come from the training set.         similar properties, not exact replicas. Having longer ran-
Both G and D are trained end-to-end using backpropagation.        dom walks combined with memory helps the model to learn
At any point of the training process it is possible to use G to   the topology and general patterns in the data (e.g., commu-
generate a set of random walks, which can then be used to         nity structure). Our experiments in Sec. 4.2 confirm this,
produce an adjacency matrix of a new generated graph. In          showing that longer random walks are indeed beneficial.
the rest of this section we describe each stage of this process   After each time step, to generate the next node in the random
and our design choices in more detail. An overview of our         walk, the network fθ should output the logits pt of length N .
model’s complete architecture can be seen in Fig. 2.              However, operating in such high dimensional space leads to
                                                                  an unnecessary computational overhead. To tackle this issue,
3.1. Architecture                                                 the LSTM outputs ot ∈ RH instead, with H  N , which
Generator. The generator G defines an implicit probabilis-        is then up-projected to RN using the matrix W up ∈ RH×N .
tic model for generating random walks: (v 1 , ..., v T ) ∼ G.     This enables us to efficiently handle large-scale graphs.
We model G as a sequential process based on a neural net-         Sampling the next node in the random walk v t presents
work fθ parametrized by θ. At each step t, fθ produces two        another challenge. Since sampling from a categorical dis-
values: the probability distribution over the next node to        tribution is a non-differentiable operation it blocks the
be sampled, parametrized by the logits pt , and the current       flow of gradients and precludes backpropagation. We
memory state of the model, denoted as mt . The next node          solve this problem by using the Straight-Through Gum-
v t , represented as a one-hot vector, is sampled from a cate-    bel estimator by Jang et al. (2016). More specifically,
gorical distribution v t ∼ Cat(σ(pt )), where σ(·) denotes        we perform the following transformation: First, we let
the softmax function, and together with mt is passed into fθ      v ∗t = σ ((pt + g)/τ )), where τ is a temperature param-
at the next step t + 1. Similarly to the classic GAN setting,     eter, and gi ’s are i.i.d. samples from a Gumbel distribution
a latent code z drawn from a multivariate standard normal         with zero mean and unit scale. Then, the next sample is
                                            NetGAN: Generating Graphs via Random Walks

    Generator                                                                                                                NetGAN
    architecture                v1                   v1             vT                                      G(z)             architecture
                          sample                          v2            sample
                           p1          pN                      p1          pN            Generator
                                W up                                W up

      z                                                                                                                       Discrimi-
                                o  1                                o T

                                                                                                                               nator
                C0                          C1 ⋯
                                                                                         Graph

                 h0                         h1 ⋯
                                                                                                                              D real D fake
                                            W down    ⋯                                                   Random
      z ∼ N (0, Id )
                                                                                                           walk
                       (a) Generator architecture                                                  (b) NetGAN architecture
             Figure 2: The NetGAN architecture proposed in this work (b) and the generator architecture (a).


chosen as v t = onehot(arg max v ∗t ). While the one-hot                         first strategy, named VAL -C RITERION is concerned with
sample v t is passed as input to the next time step, during the                  the generalization properties of NetGAN. During training,
backward pass the gradients will flow through the differen-                      we keep a sliding window of the random walks generated in
tiable v ∗t . The choice of τ allows to trade-off between better                 the last 1,000 iterations and use them to construct a matrix
flow of gradients (large τ , more uniform v ∗t ) and more exact                  of transition counts. This matrix is then used to evaluate the
calculations (small τ , v ∗t ≈ v t ).                                            link prediction performance on a validation set (i.e. ROC
                                                                                 and AP scores, for more details see Sec. 4.2). We stop with
Now that a new node v t is sampled, it needs to be projected
                                                                                 training when the validation performance stops improving.
back to a lower-dimensional representation before feeding
into the LSTM. This is done by means of down-projection                          The second strategy, named EO-C RITERION makes Net-
matrix W down ∈ RN ×H .                                                          GAN very flexible and gives the user control over the graph
                                                                                 generation. We stop training when we achieve a user spec-
Discriminator. The discriminator D is based on the stan-
                                                                                 ified edge overlap between the generated graphs (see next
dard LSTM architecture. At every time step t, a one-hot
                                                                                 section) and the original one at a given iteration. Based on
vector v t , denoting the node at the current position, is fed
                                                                                 her end task the user can choose to generate graphs with
as input. After processing the entire sequence of T nodes,
                                                                                 either small or large edge overlap with the original, while
the discriminator outputs a single score that represents the
                                                                                 maintaining structural similarity. This will lead to generated
probability of the random walk being real.
                                                                                 graphs that either generalize better or are closer replicas
                                                                                 respectively, yet still capture the properties of the original.
3.2. Training
Wasserstein GAN. We train our model based on the                                 3.3. Assembling the Adjacency Matrix
Wasserstein GAN (WGAN) framework (Arjovsky et al.,
                                                                                 After finishing the training, we use the generator G to con-
2017), as it prevents mode collapse and leads to more stable
                                                                                 struct a score matrix S of transition counts, i.e. we count
training overall. To enforce the Lipschitz constraint of the
                                                                                 how often an edge appears in the set of generated random
discriminator, we use the gradient penalty as in Gulrajani
                                                                                 walks (typically, using a much larger number of random
et al. (2017). The model parameters {θ, θ0 } are trained us-
                                                                                 walks than for early stopping, e.g., 500K). While the raw
ing stochastic gradient descent with Adam (Kingma & Ba,
                                                                                 counts matrix S is sufficient for link prediction purposes,
2014). Weights are regularized with an L2 penalty.
                                                                                 we need to convert it to a binary adjacency matrix Â if we
Early stopping. Because we are interested in generalizing                        wish to reason about the synthetic graph. First, S is sym-
the input graph, the “trivial” solution where the generator                      metrized by setting sij = sji = max{sij , sji }. Because
has memorized all existing edges is of no interest to us.                        we cannot explicitly control the starting node of the random
This means that we need to control how closely the gen-                          walks generated by G, some high-degree nodes will likely
erated graphs resemble the original one. To achieve this,                        be overrepresented. Thus, a simple binarization strategy like
we propose two possible early stopping strategies, either                        thresholding or choosing top-k entries might lead to leaving
of which can be used depending on the task at hand. The                          out the low-degree nodes and producing singletons.
                                       NetGAN: Generating Graphs via Random Walks

Table 1: Statistics of C ORA -ML and the graphs generated by NetGAN and the baselines, averaged over 5 trials. NetGAN
closely matches the input networks in most properties, while other methods either deviate significantly in at least one statistic
or overfit. * indicates values for the conf. model that by definition exactly match the original.

                               Max. Assort-    Triangle Power    Inter-comm. Intra-comm.          Cluster- Charac.      Average
  Graph
                              degree ativity    count   law exp. unity density unity density     ing coeff. path len.    rank
  C ORA -ML                    240   -0.075     2,814     1.860     4.3e-4        1.7e-3          2.73e-3     5.61
  Conf. model       (1% EO)      *   -0.030      322        *       1.6e-3        2.8e-4          3.00e-4     4.38       7.50
  Conf. model      (52% EO)      *   -0.051      626        *       9.8e-4        9.9e-4          6.10e-4     4.46       5.83
  DC-SBM           (11% EO)    165   -0.052     1,403     1.814     6.7e-4        1.2e-3          3.30e-3     5.12       3.36
  ERGM             (56% EO)    243   -0.077     2,293     1.786     6.9e-4        1.2e-3          2.17e-3     4.59       2.88
  BTER            (2.2% EO)    199    0.033     3,060     1.787     1.0e-3        7.5e-4          4.62e-3     4.59       4.75
  VGAE            (0.3% EO)     13   -0.009       14      1.674     1.4e-3        3.2e-4          1.17e-3     5.28       5.88
  NetGAN VAL       (39% EO)    199   -0.060     1,410     1.773     6.5e-4        1.3e-3          2.33e-3     5.17       3.00
  NetGAN EO        (52% EO)    233   -0.066     1,588     1.793     6.0e-4        1.4e-3          2.44e-3     5.20       1.75


To address this issue, we use the following approach: (i) We      4.1. Graph Generation
ensure that every node i has at least one edge by sampling
                                           s                      Setup. In this task, we fit NetGAN to the C ORA -ML
a neighbor j with probability pij = P ijsiv . If an edge
                                           v                      and C ITESEER citation networks in order to evaluate qual-
was already sampled before, we repeat the procedure; (ii)
                                                                  ity of the generated graphs. We compare to the following
We continue sampling edges without replacement using for
                                              s                   baselines: configuration model (Molloy & Reed, 1995),
each edge (i, j) the probability pij = P ijsuv , until we
                                             u,v                  degree-corrected stochastic blockmodel (DC-SBM) (Kar-
reach the desired amount of edges (e.g., as many edges as in      rer & Newman, 2011), exponential random graph model
the original graph). To obtain an undirected graph for every      (ERGM) (Holland & Leinhardt, 1981) and the block two-
edge (i, j) we also include (j, i). Note that this procedure      level Erdős-Réniy random graph model (BTER) (Seshadhri
is not guaranteed to produce a fully connected graph.             et al., 2012). Additionally, we use the variational graph
                                                                  autoencoder (VGAE) (Kipf & Welling, 2016) as a represen-
4. Experiments                                                    tative of network embedding approaches. We randomly hide
                                                                  15% of the edges (which are used for the stopping criterion;
In this section we evaluate the quality of the graphs gener-      see Sec. 3.2) and fit all the models on the remaining graph.
ated by NetGAN by computing various graph statistics. We          We sample 5 graphs from each of the trained models and
quantify the generalization power of the proposed model           report their average statistics in Table 1. Definitions of the
by evaluating its link prediction performance. Furthermore,       statistics, additional metrics, standard deviations and details
we demonstrate how we can generate graphs with smoothly           about the baselines are given in the supplementary material.
changing properties via latent space interpolation. Addi-
tional experiments are provided in the supp. mat.                 Evaluation. The general trend that becomes apparent
                                                                  from the results in Table 1 (and Table 2 in supplementary
Datasets. For the experiments we use five well-known              material) is that prescribed models excel at recovering the
citation datasets and the Political Blogs dataset. For the        statistics that they directly model (e.g., degree sequence for
large C ORA dataset and its commonly used subset of ma-           DC-SBM). At the same time, these models struggle when
chine learning papers denoted with C ORA -ML we use the           dealing with graph properties that they don’t account for
same preprocessing as in Bojchevski & Günnemann (2018).          (e.g., assortativity for BTER). On the other hand, NetGAN
For all the experiments we treat the graphs as undirected         is able to capture all the graph properties well, although
and only consider the largest connected component (LCC).          none of them are explicitly specified in its model definition.
Information about the datasets is listed in Table 2.              We also see that VGAE is not able to produce realistic
Table 2: Dataset statistics. NLCC , ELCC - number of nodes        graphs. This is expected, since the main purpose of VGAE is
and edges respectively in the largest connected component.        learning node embeddings, and not generating entire graphs.
                                                                  The final column shows the average rank of each method
  Name            NLCC     ELCC     Reference
                                                                  for all statistics, with NetGAN performing the best. ERGM
  C ORA -ML        2,810    7,981   (McCallum et al., 2000)
  C ORA           18,800   64,529   (McCallum et al., 2000)
                                                                  seems to be performing surprisingly well, however it suffers
  C ITE S EER      2,110    3,757   (Sen et al., 2008)            from severe overfitting – using the same fitted ERGM for
  P UBMED         19,717   44,324   (Sen et al., 2008)            the link prediction task we get both AUC and AP scores
  DBLP            16,191   51,913   (Pan et al., 2016)            close to 0.5 (worst possible value). In contrast, NetGAN
  P OL . B LOGS    1,222   16,714   (Adamic & Glance, 2005)       does a good job both at preserving properties in generated
                                                                  graphs, as well as generalizing, as we see in Sec. 4.2.
                                                               NetGAN: Generating Graphs via Random Walks


                                                        NetGAN                    Input Graph             Val-Criterion                       EO-Criterion

                                                                                  0.02                                                        1.00
                                                     Cora-ML




                                                                                                                               Edge overlap
Number of nodes
                                                                  Assortativity
                                                     NetGAN
                  102                                                             0.04                                                        0.75

                                                                                                                                              0.50
                  101                                                             0.06
                                                                                                                                              0.25
                                                                                  0.08
                  100                                                                                                                         0.00
                        100             101             102                          0k     20k   40k     60k     80k   100k                      0k                         20k   40k   60k   80k   100k
                                        Degree                                               Training iteration                                                              Training iteration

                              (a) Degree distribution                                    (b) Assortativity over                                      (c) Edge overlap (EO) over
                                                                                           training iterations                                            training iterations

                                              Figure 3: Properties of graphs generated by NetGAN trained on C ORA -ML.


Is the good performance of NetGAN in this experiment                                                    best results on different datasets. NetGAN shows competi-
only due to the overlapping edges (existing in the input                                                tive performance for all datasets, even achieving state-of-the-
graph)? To rule out this possibility we perform the following                                           art results for some of them (C ITESEER and P OL B LOGS),
experiment: We take the graph generated by NetGAN, fix                                                  despite not being explicitly trained for this task.
the overlapping edges and rewire the rest according to the
                                                                                                        Interestingly, the NetGAN performance increases when in-
configuration model. The properties of the resulting graph
                                                                                                        creasing the number of random walks sampled from the
(row #3 in Table 1) deviate strongly from the input graph.
                                                                                                        generator. This is especially true for the larger networks
This confirms that NetGAN does not simply memorize some
                                                                                                        (C ORA, DBLP, P UBMED), since given their size we need
edges and generates the rest at random, but rather captures
                                                                                                        more random walks to cover the entire graph. This suggests
the underlying structure of the network.
                                                                                                        that for an additional computational cost one can get sig-
In line with our intuition, we can see that higher EO leads                                             nificant gains in link prediction performance. Note, that
to generated graphs with statistics closer to the original.                                             while 100M may seem like a large number, the sampling
Figs. 3b and 3c show how the graph statistics evolve during                                             procedure can be trivially parallelized.
the training process. Fig. 3c shows that the edge overlap
                                                                                                        Sensitivity analysis. Although NetGAN has many hy-
smoothly increasing with the number of epochs. We provide
                                                                                                        perparameters – typical for a GAN model – in practice
plots for other statistics and for C ITESEER in the supp. mat.
                                                                                                        most of them are not critical for performance, as long
                                                                                                        as they are within a reasonable range (e.g. H ≥ 30).
4.2. Link Prediction                                                                                    One important ex-
Setup. Link prediction is a common graph mining task                                                    ception is the the



                                                                                                                                                     Link prediction score
where the goal is to predict the existence of unobserved links                                          random walk length           0.90

in a given graph. We use it to evaluate the generalization                                              T . To choose the            0.85
properties of NetGAN. We hold out 10% of edges from the                                                 optimal value, we                            ROC AUC
                                                                                                                                     0.80
graph for validation and 5% as the test set, along with the                                             evaluate the change                          Avg. precision

same amount of randomly selected non-edges, while ensur-                                                in link prediction                 2 4    8         16    20
ing that the training network remains connected. We mea-                                                performance as we                     Random walk length

sure the performance with two commonly used metrics: area                                               vary T on C ORA -
under the ROC curve (AUC) and average precision (AP). To                                                ML. We train multi-       Figure 6: Effect of the random
evaluate NetGAN’s performance, we sample a given number                                                 ple models with dif-      walk length T on the performance.
of random walks (500K/100M) from the trained generator                                                  ferent random walk
and we use the observed transition counts between any two                                               lengths, and evaluate the scores ensuring each one observes
nodes as a measure of how likely there is an edge between                                               equal number of transitions. Results averaged over 5 runs
them. We compare with DC-SBM, node2vec and VGAE, as                                                     are given in Fig. 6. We empirically confirm that the model
well as Adamic/Adar(Adamic & Adar, 2003).                                                               benefits from using longer random walks as opposed to just
                                                                                                        edges (i.e. T =2). The performance gain for T = 20 over
Evaluation. The results are listed in Table 3. There is no                                              T = 16 is marginal and does not outweigh the additional
overall dominant method, with different methods achieving                                               computational cost, thus we set T = 16 for all experiments.
                                        NetGAN: Generating Graphs via Random Walks

                                         Table 3: Link prediction performance (in %).

                        C ORA -ML            C ORA          C ITESEER          DBLP             P UBMED         P OL B LOGS
 Method
                       AUC     AP        AUC      AP       AUC     AP       AUC    AP         AUC     AP       AUC       AP
 Adamic/Adar           92.16 85.43       93.00 86.18       88.69 77.82      91.13 82.48       84.98 70.14      85.43 92.16
 DC-SBM                96.03 95.15       98.01 97.45       94.77 93.13      97.05 96.57       96.76 95.64      95.46 94.93
 node2vec              92.19 91.76       98.52 98.36       95.29 94.58      96.41 96.36       96.49 95.97      85.10 83.54
 VGAE                  95.79 96.30       97.59 97.93       95.11 96.31      96.38 96.93       94.50 96.00      93.73 94.12
 NetGAN (500K)         94.00 92.32       82.31 68.47       95.18 91.93      82.45 70.28       87.39 76.55      95.06 94.61
 NetGAN (100M)         95.19 95.24       84.82 88.04       96.30 96.89      86.61 89.21       93.41 94.59      95.51 94.83


4.3. Latent Variable Interpolation                                 space respectively. In Fig. 5b and 5c, we see the evolution
                                                                   of selected community shares when following a trajectory
Setup. Latent space interpolation is a good way to gain
                                                                   from top to bottom, and left to right, respectively. The com-
insight into what kind of structure the generator was able
                                                                   munity histograms resulting from sampling random walks
to capture. To be able to visualize the properties of the
                                                                   from opposing regions of the latent space are very different;
generated graphs we train our model using a 2-dimensional
                                                                   again the transitions between these histograms are smooth,
noise vector z drawn as before from a bivariate standard
                                                                   as can be seen in the trajectories in Fig. 5b and 5c.
normal distribution. This corresponds to a 2-dimensional
latent space Ω = R2 . Then, instead of sampling z from the
entire latent space Ω, we now sample from subregions of            5. Discussion and Future Work
Ω and visualize the results. Specifically, we divide Ω into
                                                                   When evaluating different graph generative models in Sec.
20 × 20 subregions (bins) of equal probability mass using
                                                                   3.2, we observed a major limitation of explicit models.
the standard normal cumulative distribution function Φ. For
                                                                   While the prescribed approaches excel at recovering the
each bin we generate 62.5K random walks. We evaluate
                                                                   properties directly included in their definition, they perform
properties of both the generated random walks themselves,
                                                                   significantly worse with respect to the rest. This clearly
as well as properties of the resulting graphs obtained by
                                                                   indicates the need for implicit graph generators such as
sampling a binary adjacency matrix for each bin.
                                                                   NetGAN. Indeed, we notice that our model is able to con-
Evaluation. In Fig. 4a and 4b we see properties of the             sistently capture all the important graph characteristics (see
generated random walks; in Fig. 4c and 4d, we visualize            Table 1). Moreover, NetGAN generalizes beyond the input
properties of graphs sampled from the random walks in              graph, as can be seen by its strong link prediction perfor-
the respective bins. In all four heatmaps, we see distinct         mance in Sec. 4.2. Still, being the first model of its kind,
patterns, e.g. higher average degree of starting nodes for the     NetGAN possesses certain limitations, and a number of
bottom right region of Fig. 4a, or higher degree distribution      related questions could be addressed in follow-up works:
inequality in the top-right area of Fig. 4c. While Fig. 4c and
                                                                   Scalability. We have observed in Sec. 4.2 that it takes a
4d show that certain regions of z correspond to generated
                                                                   large number of generated random walks to get represen-
graphs with very different degree distributions, recall that
                                                                   tative transition counts for large graphs. While sampling
sampling from the entire latent space (Ω) yields graphs with
                                                                   random walks from NetGAN is trivially parallelizable, a
degree distribution similar to the original graph (see Fig. 1c).
                                                                   possible extension of our model is to use a conditional gen-
The model was trained on C ORA -ML. More heatmaps for
                                                                   erator, i.e. the generator can be provided a desired starting
other metrics (16 in total) and visualizations for C ITESEER
                                                                   node, thus ensuring a more even coverage. On the other
can be found in the supplementary material.
                                                                   hand, the sampling procedure itself can be sped up by in-
This experiment clearly demonstrates that by interpolating         corporating a hierarchical softmax output layer - a method
in the latent space we can obtain graphs with smoothly             commonly used in natural language processing.
changing properties. The smooth transitions in the heatmaps
                                                                   Evaluation. It is nearly impossible to judge whether a
provide evidence that our model learns to map specific parts
                                                                   graph is realistic by visually inspecting it (unlike images,
of the latent space to specific properties of the graph.
                                                                   for example). In this work we already quantitatively evaluate
We can also see this mapping from latent space to the gen-         the performance of NetGAN on a large number of standard
erated graph properties in the community distribution his-         graph statistics. However, developing new measures ap-
tograms on a 10 × 10 grid in Fig. 5. Marked by (*) and             plicable to (implicit) graph generative models will deepen
(Ω) we see the community distributions for the input graph         our understanding of their behavior, and is an important
and the graph obtained by sampling on the complete latent          direction for future work.
                                                                       NetGAN: Generating Graphs via Random Walks

    1                                                      1                                                    1                                                             1
                                                                                                 0.42                                                                                                          450
                                              22                                                                                                                   0.64
                                              20                                                 0.4                                                               0.62                                        400
                                              18                                                 0.38                                                              0.6                                         350

  Φ(z2)                                                  Φ(z2)                                                Φ(z2)                                                         Φ(z2)
                                              16                                                 0.35                                                              0.58                                        300
                                              14                                                                                                                   0.56
                                                                                                 0.32                                                                                                          250
                                              12                                                                                                                   0.54
                                                                                                 0.3                                                                                                           200
                                              10                                                                                                                   0.52
                                                                                                 0.28                                                                                                          150
                                              8                                                                                                                    0.5
    0                                                      0                                                    0                                                             0
          0             Φ(z1)            1                       0           Φ(z1)         1                          0         Φ(z1)                       1                       0       Φ(z1)          1

                 (a) Avg. degree                                     (b) Avg. share of nodes                               (c) Gini coefficient                                          (d) Max. degree
                   of start node                                       in start community                                  (input graph: 0.48)                                          (input graph: 240)

 Figure 4: Properties of the random walks (4a and 4b) as well as the graphs (4c and 4d) sampled from the 20 × 20 bins.




                                                                                                                                          Community share
                      0.9                                                                                                                                   0.3

                      0.8                                                                                                                                                                           Community 2
                                                                                                                                                            0.2                                     Community 5
                      0.7

                      0.6                                                                                                 (*)                               0.1
                                                                                                                                                                   0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
                                                                                                                                                                                          Φ(z2)
              Φ(z2)
                      0.5

                      0.4                                                                                                                                                 (b) Top to bottom trajectory
                                                                                                                          (Ω)




                                                                                                                                          Community share
                      0.3                                                                                                                                   0.20

                      0.2                                                                                                                                                      Community 4
                                                                                                                                                            0.15               Community 7
                      0.1

                      0.0                                                                                                                                   0.10
                            0.0   0.1   0.2        0.3    0.4          0.5    0.6    0.7   0.8          0.9      1.0                                                0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
                                                                     Φ(z1)                                                                                                                Φ(z1)

                                              (a) Community histograms                                                                                                     (c) Left to right trajectory

Figure 5: Community histograms of graphs sampled from subsets of the latent space. (a) shows complete community
histograms on a 10 × 10 grid. (b) and (c) show how shares of specific communities change along trajectories. (Ω) is the
community distribution when sampling from the entire latent space, and (*) is the community histogram of C ORA -ML.
Available as an animation at https://goo.gl/bkNcVa.


Experimental scope. In the current work we focus on the                                                         6. Conclusion
setting of a single large graph. Adaptation to other scenarios,
such as a collection of smaller i.i.d. graphs, that frequently                                                  In this work we introduce NetGAN - an implicit generative
occur in other fields (e.g., chemistry, biology), would be an                                                   model for network data. NetGAN is able to generate graphs
important extension of our model. Studying the influence of                                                     that capture important topological properties of complex
the graph topology (e.g., sparsity, diameter) on NetGAN’s                                                       networks, such as community structure and degree distri-
performance will shed more light on the model’s properties.                                                     bution, without having to manually specify any of them.
                                                                                                                Moreover, our proposed model shows strong generalization
Other types of graphs. While plain graphs are ubiqui-                                                           properties, as highlighted by its competitive link prediction
tous, many of important applications deal with attributed,                                                      performance on a number of datasets. NetGAN can also
k-partite or heterogeneous networks. Adapting the Net-                                                          be used for generating graphs with continuously varying
GAN model to handle these other modalities of the data is                                                       characteristics using latent space interpolation. Combined
a promising direction for future research. Especially im-                                                       our results provide strong evidence that implicit generative
portant would be an adaptation to the dynamic / inductive                                                       models for graphs are well-suited for capturing the complex
setting, where new nodes are added over time.                                                                   nature of real-world networks.
                                      NetGAN: Generating Graphs via Random Walks

Acknowledgments                                                 Choi, E., Biswal, S., Malin, B., Duke, J., Stewart, W. F., and
                                                                  Sun, J. Generating multi-label discrete electronic health
This research was supported by the German Research Foun-          records using generative adversarial networks. arXiv
dation, Emmy Noether grant GU 1409/2-1, and by the Tech-          preprint arXiv:1703.06490, 2017.
nical University of Munich - Institute for Advanced Study,
funded by the German Excellence Initiative and the Euro-        Dong, Y., Johnson, R. A., Xu, J., and Chawla, N. V. Struc-
pean Union Seventh Framework Programme under grant                tural diversity and homophily: A study across more than
agreement no 291763, co-funded by the European Union.             one hundred big networks. In Proceedings of the SIGKDD
                                                                  International Conference on Knowledge Discovery and
References                                                       Data Mining, pp. 807–816, 2017.

Adamic, L. A. and Adar, E. Friends and neighbors on the         Goldenberg, A., Zheng, A. X., Fienberg, S. E., Airoldi,
 web. Social networks, 25(3):211–230, 2003.                       E. M., et al. A survey of statistical network models.
                                                                 Foundations and Trends in Machine Learning, 2(2):129–
Adamic, L. A. and Glance, N. The political blogosphere and        233, 2010.
  the 2004 US election: divided they blog. In Proceedings
  of the international workshop on Link discovery, pp. 36–      Goodfellow, I. NIPS 2016 tutorial: Generative adversarial
  43, 2005.                                                       networks. arXiv preprint arXiv:1701.00160, 2016.

Arjovsky, M., Chintala, S., and Bottou, L. Wasserstein          Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B.,
  GAN. arXiv preprint arXiv:1701.07875, 2017.                    Warde-Farley, D., Ozair, S., Courville, A., and Bengio,
                                                                 Y. Generative adversarial nets. In Advances in neural
Barabási, A.-L. and Albert, R. Emergence of scaling in           information processing systems, pp. 2672–2680, 2014.
  random networks. Science, 286(5439):509–512, 1999.
                                                                Grover, A. and Leskovec, J. node2vec: Scalable feature
Bender, E. A. and Canfield, E. R. The asymptotic number           learning for networks. In Proceedings of the SIGKDD
  of labeled graphs with given degree sequences. Journal          international conference on Knowledge discovery and
  of Combinatorial Theory, Series A, 24(3):296–307, 1978.         data mining, pp. 855–864, 2016.
Berthelot, D., Schumm, T., and Metz, L. Began: Bound-           Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., and
  ary equilibrium generative adversarial networks. arXiv          Courville, A. Improved training of Wasserstein GANs.
  preprint arXiv:1703.10717, 2017.                                arXiv preprint arXiv:1704.00028, 2017.
Blanken, H. M., de Vries, A. P., Blok, H. E., and Feng, L.      Handcock, M. S., Hunter, D. R., Butts, C. T., Goodreau,
  Multimedia retrieval. 2007.                                     S. M., Krivitsky, P. N., and Morris, M. ERGM: Fit, Sim-
                                                                  ulate and Diagnose Exponential-Family Models for Net-
Bojchevski, A. and Günnemann, S. Bayesian robust at-             works. The Statnet Project, 2017. R package version
  tributed graph clustering: Joint learning of partial anoma-     3.8.0.
  lies and group structure. In Proceedings of the AAAI
  Conference on Artificial Intelligence, 2018, pp. 2738–        Hjelm, R. D., Jacob, A. P., Che, T., Cho, K., and Bengio, Y.
  2745.                                                           Boundary-seeking generative adversarial networks. arXiv
                                                                  preprint arXiv:1702.08431, 2017.
Bojchevski, A. and Günnemann, S. Deep gaussian em-
  bedding of graphs: Unsupervised inductive learning via        Hochreiter, S. and Schmidhuber, J. Long short-term memory.
  ranking. In International Conference on Learning Repre-         Neural Computation, 9(8):1735–1780, 1997.
  sentations, 2018.
                                                                Holland, P. W. and Leinhardt, S. An exponential family
Broido, A. D. and Clauset, A. Scale-free networks are rare.       of probability distributions for directed graphs. Journal
  arXiv preprint arXiv:1801.03400, 2018.                          of the american Statistical association, 76(373):33–50,
                                                                 1981.
Chakrabarti, D. and Faloutsos, C. Graph mining: Laws,
  generators, and algorithms. Computing Surveys (CSUR),         Jang, E., Gu, S., and Poole, B. Categorical repa-
  38(1):2, 2006.                                                  rameterization with Gumbel-softmax. arXiv preprint
                                                                  arXiv:1611.01144, 2016.
Che, T., Li, Y., Zhang, R., Hjelm, R. D., Li, W., Song,
  Y., and Bengio, Y. Maximum-likelihood augmented dis-          Karras, T., Aila, T., Laine, S., and Lehtinen, J. Progres-
  crete generative adversarial networks. arXiv preprint           sive growing of gans for improved quality, stability, and
  arXiv:1702.07983, 2017.                                         variation. arXiv preprint arXiv:1710.10196, 2017.
                                       NetGAN: Generating Graphs via Random Walks

Karrer, B. and Newman, M. E. Stochastic blockmodels and           Tavakoli, S., Hajibagheri, A., and Sukthankar, G. Learn-
  community structure in networks. Physical Review E, 83            ing social graph topologies using generative adversarial
  (1):016107, 2011.                                                 neural networks. 2017.

Kim, M. and Leskovec, J. Modeling social networks with            Wang, H., Wang, J., Wang, J., Zhao, M., Zhang, W., Zhang,
  node attributes using the multiplicative attribute graph         F., Xie, X., and Guo, M. GraphGAN: Graph represen-
  model. arXiv preprint arXiv:1106.5053, 2011.                     tation learning with generative adversarial nets. arXiv
                                                                   preprint arXiv:1711.08267, 2017.
Kingma, D. and Ba, J. Adam: A method for stochastic
  optimization. arXiv preprint arXiv:1412.6980, 2014.             Wu, J., Zhang, C., Xue, T., Freeman, B., and Tenenbaum,
                                                                   J. Learning a probabilistic latent space of object shapes
Kipf, T. N. and Welling, M. Variational graph auto-encoders.       via 3d generative-adversarial modeling. In Advances in
  arXiv preprint arXiv:1611.07308, 2016.                           Neural Information Processing Systems, pp. 82–90, 2016.
Kusner, M. J. and Hernández-Lobato, J. M. GANs for               Yu, L., Zhang, W., Wang, J., and Yu, Y. SeqGAN: Sequence
  sequences of discrete elements with the Gumbel-softmax            generative adversarial nets with policy gradient. In AAAI,
  distribution. arXiv preprint arXiv:1611.04051, 2016.              pp. 2852–2858, 2017.

Li, J., Monroe, W., Shi, T., Ritter, A., and Jurafsky, D.
  Adversarial learning for neural dialogue generation. arXiv
  preprint arXiv:1701.06547, 2017.

Liang, X., Hu, Z., Zhang, H., Gan, C., and Xing, E. P. Recur-
  rent topic-transition GAN for visual paragraph generation.
  arXiv preprint arXiv:1703.07022, 2017.

Liu, W., Chen, P.-Y., Cooper, H., Oh, M. H., Yeung, S., and
  Suzumura, T. Can GAN learn topological features of a
  graph? arXiv preprint arXiv:1707.06197, 2017.

McCallum, A. K., Nigam, K., Rennie, J., and Seymore,
 K. Automating the construction of internet portals with
 machine learning. Information Retrieval, 3(2):127–163,
 2000.

Molloy, M. and Reed, B. A critical point for random graphs
 with a given degree sequence. Random structures &
 algorithms, 6(2-3):161–180, 1995.

Pan, S., Wu, J., Zhu, X., Zhang, C., and Wang, Y. Tri-party
  deep network representation. Network, 11(9):12, 2016.

Peixoto, T. P. The graph-tool python library. URL
  http://figshare.com/articles/graph_
  tool/1164194.

Perozzi, B., Al-Rfou, R., and Skiena, S. Deepwalk: On-
  line learning of social representations. In Proceedings
  of the SIGKDD international conference on Knowledge
  discovery and data mining, pp. 701–710, 2014.

Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B.,
  and Eliassi-Rad, T. Collective classification in network
  data. AI magazine, 29(3):93, 2008.

Seshadhri, C., Kolda, T. G., and Pinar, A. Community
  structure and scale-free collections of Erdős-Rényi graphs.
  Physical Review E, 85(5):056109, 2012.
                                                NetGAN: Generating Graphs via Random Walks




A. Graph statistics
                            Table 4: Graph statistics used to measure graph properties in this work.
 Metric name                Computation                                   Description
 Maximum degree             max d(v)                                      Maximum degree of all nodes in a graph.
                            v∈V
                                                                          Pearson correlation of degrees of connected nodes, where the (xi , yi )
 Assortativity              ρ = cov(X,Y
                                  σX σY
                                        )
                                                                          pairs are the degrees of connected nodes.
                            |{{u,v,w}|{(u,v),(v,w),(u,w)}⊆E}|             Number of triangles in the graph, where u ∼ v denotes that u and v
 Triangle count                              6                            are connected.
                                                   −1
                                                                          Exponent of the power law distribution, where dmin denotes the mini-
                                         log dd(u)
                                    P
 Power law exponent         1+n
                                   u∈V
                                               min                        mum degree in a network.
                            1
                              PK PK             1
                                                       P      P
 Inter-community density    K   j=1     k=1 |Ck |        u∈Cj  v∈Ck Auv   Fraction of possible inter-community edges present in graph.
                                        k6=j ( |Cj | )
                            1
                              P K      1
                                            P
 Intra-community density    K   j=1 (|Cj |)     u,v∈Cj Auv                Fraction of possible intra-community edges present in graph.
                                       2
                                   d(v)
                            P           
 Wedge count                    v∈V        2
                                                                          Number of wedges (2-stars), i.e. two-hop paths in an undirected graph.
                               1
                                      P        d(v)    d(v)               Entropy of degree distribution, 1 means uniform, 0 means a single
 Rel. edge distr. entropy   ln |V |       v∈V − |E| ln |E|                node is connected to all others.
                                                                          Size of largest connected component, where F are all connected com-
 LCC                        Nmax = max |f |
                                   f ⊆F                                   ponents of the graph.
                                  d(v)
                            P         
 Claw count                     v∈V     3
                                                                          Number of claws (3-stars)
                                                                          Common measure for inequality in a distribution, where dˆ is the sorted
                                P|V | ˆ
                            2   i=1 idi    |V |+1
 Gini coefficient                |V | ˆ −    |V |                         list of degrees in the graph.
                                P
                            |V | i=1 di
                                                                          Share of in- and outgoing edges of community Ci , normalized by the
                                  P
                                    v∈Ci d(v)
 Community distribution     ci = P
                                     v∈V d(v)                             number of edges in the graph.




B. Baselines
 • Configuration model. In addition to randomly rewiring all edges in the input graph, we also generate random graphs
   with similar overlap as graphs generated by NetGAN using the configuration model. For this, we randomly select a
   share of edges (e.g. 39%) and keep them fixed, and shuffle the remaining edges. This leads to a graph with the specified
   edge overlap; in Table 2 we show that with the same edge overlap, NetGAN’s generated graphs in general match the
   input graph better w.r.t the statistics we measure.
 • Exponential random graph model. We use the R implementation of ERGM from the ergm package (Handcock
   et al., 2017). We used the following parameter settings: edge count, density, degree correlation,
   deg1.5, and gwesp. Here, deg1.5 is the sum of all degrees to the power of 1.5, and gwesp refers to the
   geometrically weighted edgewise shared partner distribution.

 • Degree-corrected stochastic blockmodel. We use the Python implementation from the graph-tool package
   (Peixoto) using the recommended hyperparameter settings.
 • Variational graph autoencoder. We use the implementation provided by the authors (https://github.com/
   tkipf/gae). We construct the graph from the predicted edge probabilities using the same protocol as in Sec. 3.3 of
   our paper. To ensure a fair comparison we perform early stopping, i.e. select the weights that achieve the best validation
   set performance.
                                          NetGAN: Generating Graphs via Random Walks




C. Properties of generated graphs
Table 5: Comparison of graph statistics between the C ITESEER /C ORA -ML graph and graphs generated by GraphGAN and
DC-SBM, averaged after 5 trials.
                                Max.                             Triangle     Power law     Avg. Inter-com-   Avg. Intra-com-         Clustering
 Graph                                      Assortativity
                               degree                             count       exponent      munity density    munity density          coefficient
                           Avg. Std.      Avg.     Std.        Avg. Std.    Avg.    Std.    Avg.      Std.    Avg.      Std.        Avg.       Std.
 C ITESEER                 77             -0.022               451          2.239           4.9e-4            9.3e-4               1.08e-2
 Conf. model                 *       *    -0.017 ± 0.006       20 ± 6.50      *       *     1.1e-3 ± 1e-5     2.3e-4 ± 2e-5        5.80e-4 ± 1.29e-4
 Conf. model    (42% EO)     *       *    -0.020 ± 0.009       54 ± 8.8       *       *     8.4e-4 ± 1e-5     5.1e-4 ± 1e-5        1.33e-3 ± 6.15e-5
 Conf. model    (76% EO)     *       *    -0.024 ± 0.006       207 ± 11.8     *       *     6.3e-4 ± 1e-5     7.6e-4 ± 1e-5        5.00e-3 ± 2.57e-4
 DC-SBM        (6.6% EO)   53 ± 5.6       0.022 ± 0.018        257 ± 30.9   2.066 ± 0.014   7.6e-4 ± 2e-5     5.3e-4 ± 3e-5        1.00e-2 ± 2.63e-3
 ERGM           (27% EO)   66 ± 1         0.052 ± 0.005        415.6 ± 8    2.0 ± 0.01      9.3e-4 ± 2e-5     4.8e-4 ± 6e-6        1.49e-2 ± 5.68e-4
 BTER            (2% EO)   70 ± 7.2       0.065 ± 0.014        449 ± 33     2.049 ± 0.01    1.1e-3 ± 2e-5     2.8e-4 ± 6e-6        1.22e-2 ± 2.31e-3
 VGAe          (0.2% EO)   9.2 ± 0.7      -0.057 ± 0.016       2     ±1     2.039 ± 0.00    1.2e-3 ± 1e-5     2.5e-4 ± 2e-5        1.35e-3 ± 9.96e-4
 NetGAN VAL     (42% EO)   54 ± 4.2       -0.082 ± 0.009       316 ± 11.2   2.154 ± 0.003   6.5e-4 ± 2e-5     8.0e-4 ± 2e-5        1.99e-2 ± 3.48e-3
 NetGAN EO      (76% EO)   63 ± 4.3       -0.054 ± 0.006       227 ± 13.3   2.204 ± 0.003   5.9e-4 ± 2e-5     8.6e-4 ± 1e-5        7.71e-3 ± 2.43e-4
 C ORA -ML                 240            -0.075               2,814        1.86            4.3e-4            1.7e-3               2.73e-3
 Conf. model                 *       *    -0.030 ± 0.003       322 ± 31       *       *     1.6e-3 ± 1e-5     2.8e-4 ± 1e-5        3.00e-4 ± 2.88e-5
 Conf. model    (39% EO)     *       *    -0.050 ± 0.005       420 ± 14       *       *     1.1e-3 ± 1e-5     8.0e-4 ± 1e-5        4.10e-4 ± 1.40e-5
 Conf. model    (52% EO)     *       *    -0.051 ± 0.002       626 ± 19       *       *     9.8e-4 ± 1e-5     9.9e-4 ± 2e-5        6.10e-4 ± 1.85e-5
 DC-SBM         (11% EO)   165 ± 9.0      -0.052 ± 0.004       1,403 ± 67   1.814 ± 0.008   6.7e-4 ± 2e-5     1.2e-3 ± 4e-5        3.30e-3 ± 2.71e-4
 ERGM           (56% EO)   243 ± 1.94     -0.077 ± 0.000       2,293 ± 23   1.786 ± 0.003   6.9e-4 ± 2e-5     1.2e-3 ± 1e-5        2.17e-3 ± 5.44e-5
 BTER            (2% EO)   199 ± 13       0.033 ± 0.008        3060 ± 114   1.787 ± 0.004   1.1e-3 ± 1e-5     7.5e-4 ± 1e-5        4.62e-3 ± 5.92e-4
 VGAe          (0.3% EO)   13.1 ± 1       -0.010 ± 0.014       14 ± 3       1.674 ± 0.001   1.4e-3 ± 2e-5     3.2e-4 ± 1e-5        1.17e-3 ± 2.02e-4
 NetGAN VAL     (39% EO)   199 ± 6.7      -0.060 ± 0.004       1,410 ± 30   1.773 ± 0.002   6.5e-4 ± 1e-5     1.3e-3 ± 2e-5        2.33e-3 ± 1.75e-4
 NetGAN EO      (52% EO)   233 ± 3.6      -0.066 ± 0.003       1,588 ± 59   1.793 ± 0.003   6.0e-4 ± 1e-5     1.4e-3 ± 1e-5        2.44e-3 ± 1.91e-4

                                                Rel. edge           Largest                                                          Characteristic
 Graph                      Wedge count                                           Claw count        Gini coeff.    Edge overlap
                                                distr. entr.     conn. comp                                                           path length
                            Avg.    Std.      Avg.      Std.     Avg. Std.       Avg.      Std.    Avg.    Std.    Avg.    Std.      Avg.    Std.
 C ITESEER                 16,824             0.959              2,110          125,701            0.404           1                 10.33
 Conf. model                  *       *       0.955 ± 0.001      2,011 ± 6.8       *         *       *       *     0.008 ± 0.001     5.95 ± 0.03
 Conf. model    (42% EO)      *       *       0.956 ± 0.001      2,045 ± 12.5      *         *       *       *     0.42 ± 0.002      6.14 ± 0.03
 Conf. model    (76% EO)      *       *       0.957 ± 0.001      2,065 ± 10.2      *         *       *       *     0.76 ± 0.0        6.85 ± 0.04
 DC-SBM        (6.6% EO)   15,531 ± 592       0.938 ± 0.001      1,697 ± 27     69,818 ± 11,969    0.502 ± 0.005   0.066 ± 0.011     7.75 ± 0.26
 ERGM           (27% EO)   16,346 ± 101       0.945 ± 0.001      1,753 ± 15     80,510 ± 1,337     0.474 ± 0.003   0.27 ± 0.01       5.92 ± 0.01
 BTER            (2% EO)   18,193 ± 661       0.940 ± 0.001      1,708 ± 14     113,425 ± 19,737   0.491 ± 0.007   0.02 ± 0.002      5.66 ± 0.07
 VGAe          (0.2% EO)   8,141 ± 47         0.986 ± 0.000      2,110 ± 0      6,611 ± 144        0.256 ± 0.003   0.002 ± 0.001     7.75 ± 0.04
 NetGAN VAL     (42% EO)   12,998 ± 84.6      0.969 ± 0.000      2,079 ± 12.6   57,654 ± 4,226     0.354 ± 0.001   0.42 ± 0.006      8.28 ± 0.11
 NetGAN EO      (76% EO)   15,202 ± 378       0.963 ± 0.000      2,053 ± 23     94,149 ± 11,926    0.385 ± 0.002   0.76 ± 0.01       7.68 ± 0.13
 C ORA -ML                 101,872            0.941              2,810          3.1e6              0.482           1                 5.61
 Conf. model                  *       *       0.928 ± 0.002      2,785 ± 4.9       *         *       *       *     0.013 ± 0.001     4.38 ± 0.01
 Conf. model    (39% EO)      *       *       0.931 ± 0.002      2,793 ± 2.0       *         *       *       *     0.39 ± 0.0        4.41 ± 0.02
 Conf. model    (52% EO)      *       *       0.933 ± 0.001      2,793 ± 6.0       *         *       *       *     0.52 ± 0.0        4.46 ± 0.02
 DC-SBM         (11% EO)   73,921 ± 3,436     0.934 ± 0.001      2,474 ± 18.9   1.2e6 ± 170,045    0.523 ± 0.003   0.11 ± 0.003      5.12 ± 0.04
 ERGM           (56% EO)   98,615 ± 385       0.932 ± 0.001      2,489 ± 11     3,1e6 ± 57,092     0.517 ± 0.002   0.56 ± 0.014      4.59 ± 0.02
 BTER            (2% EO)   91,813 ± 3,546     0.935 ± 0.000      2,439 ± 19     2.0e6 ± 280,945    0.515 ± 0.003   0.02 ± 0.001      4.59 ± 0.03
 VGAe          (0.3% EO)   31,290 ± 178       0.990 ± 0.000      2,810 ± 0      46,586 ± 937       0.223 ± 0.003   0.003 ± 0.001     5.28 ± 0.01
 NetGAN VAL     (39% EO)   75,724 ± 1,401     0.959 ± 0.000      2,809 ± 1.6    1.8e6 ± 141,795    0.398 ± 0.002   0.39 ± 0.004      5.17 ± 0.04
 NetGAN EO      (52% EO)   86,763 ± 1,096     0.954 ± 0.001      2,807 ± 1.6    2.6e6 ± 103,667    0.42 ± 0.003    0.52 ± 0.001      5.20 ± 0.02
                                                              NetGAN: Generating Graphs via Random Walks


D. Graph statistics during the training process

                                                NetGAN                   Input Graph                             Val-Criterion                                  EO-Criterion

                       250                                                            0.02




      Max. degree                                                    Assortativity                                                     Triangle count
                       200                                                            0.04                                                              2000


                                                                                      0.06
                       150                                                                                                                              1000

                                                                                      0.08
                       100
                                                                                                                                                           0
                            0k    20k    40k    60k    80k   100k                           0k    20k    40k    60k   80k   100k                            0k    20k   40k    60k   80k   100k
                                   Training iteration                                              Training iteration                                              Training iteration

                                          (a)                                                            (b)                                                             (c)

                                                                                     2800                                                         1.00




 Power law exp.
                  1.850




                                                                                                                                   Edge overlap
                                                                                                                                                  0.75
                  1.825                                                              2700

                                                                    LCC                                                                           0.50
                  1.800                                                              2600
                                                                                                                                                  0.25
                  1.775                                                              2500
                                                                                                                                                  0.00
                             0k    20k   40k     60k   80k   100k                       0k        20k    40k    60k   80k   100k                      0k         20k    40k    60k   80k   100k
                                   Training iteration                                             Training iteration                                              Training iteration

                                          (d)                                                            (e)                                                             (f)

                                                  Figure 7: Evolution of graph statistics during training on C ORA -ML




                                                NetGAN                   Input Graph                             Val-Criterion                                  EO-Criterion




                                                                                                                                       Triangle count
                                                                                                                                                        400




      Max. degree                                                   Assortativity
                       70                                                             0.000

                                                                                      0.025                                                             300
                       60

                                                                                      0.050                                                             200
                       50

                                                                                      0.075                                                             100
                       40
                                                                                                                                                          0
                         0k       20k    40k    60k    80k   100k                            0k    20k    40k   60k   80k 100k                             0k     20k   40k    60k   80k   100k
                                  Training iteration                                               Training iteration                                             Training iteration

                                          (a)                                                            (b)                                                             (c)
                                                                                     2100                                                         1.00




      Power law exp.                                                                                                               Edge overlap
                       2.20                                                                                                                       0.75
                                                                                     2000

                       2.15                                         LCC              1900                                                         0.50

                                                                                     1800                                                         0.25
                       2.10

                                                                                     1700                                                         0.00
                             0k    20k   40k     60k   80k   100k                        0k       20k    40k    60k   80k   100k                      0k         20k    40k    60k   80k   100k
                                   Training iteration                                             Training iteration                                              Training iteration

                                          (d)                                                            (e)                                                             (f)

                                                  Figure 8: Evolution of graph statistics during training on C ITESEER
                                                        NetGAN: Generating Graphs via Random Walks




E. Latent space interpolation heatmaps
   1                                        1                                        1                                     1
                                                                           0.42                                                                            450
                                  22                                                                              0.64
                                  20                                       0.4                                    0.62                                     400
                                  18                                       0.38                                   0.6                                      350

 Φ(z2)                                    Φ(z2)                                    Φ(z2)                                 Φ(z2)
                                  16                                       0.35                                   0.58                                     300
                                  14                                                                              0.56
                                                                           0.32                                                                            250
                                  12                                                                              0.54
                                                                           0.3                                                                             200
                                  10                                                                              0.52
                                                                           0.28                                                                            150
                                  8                                                                               0.5
   0                                        0                                        0                                     0
         0       Φ(z1)        1                   0         Φ(z1)      1                   0      Φ(z1)       1                  0        Φ(z1)      1


               (a) Avg. degree             (b) Avg. share of nodes in the                      (c) Gini coefficient                     (d) Max. degree
                 of start node            same comm. as the starting node                      (input graph: 0.48)                     (input graph: 240)
   1                                        1                              16.75     1                                     1
                                                                                                                                                           2800
                                  0.0                                      16.5                                   0.93
                                                                           16.25                                                                           2780
                                  −0.02                                                                           0.92
                                                                           16.0

 Φ(z2)                                    Φ(z2)                                    Φ(z2)                                 Φ(z2)
                                  −0.04                                    15.75                                  0.91                                     2760

                                                                           15.5
                                  −0.06                                                                           0.9                                      2740
                                                                           15.25
                                  −0.08                                    15.0                                   0.89                                     2720
                                  −0.1                                     14.75
   0                                        0                                        0                            0.88     0
         0       Φ(z1)        1                   0         Φ(z1)      1                   0      Φ(z1)       1                  0        Φ(z1)      1


                (e) Assortativity                          (f) Claw count                  (g) Rel. edge distr. entro-               (h) Largest conn. comp.
             (input graph: -0.075)                    (input graph: 3.1 × 106 )              py (input graph: 0.94)                    (input graph: 2,810)
   1                              0.08      1                              4.25      1                                     1
                                                                                                                  0.84
                                  0.08                                     4.0                                                                             0.76
                                                                           3.75
                                  0.07                                                                            0.82                                     0.74
                                                                           3.5

 Φ(z2)                                    Φ(z2)                                    Φ(z2)                                 Φ(z2)
                                  0.06                                     3.25                                                                            0.72
                                                                                                                  0.8
                                  0.06                                     3.0
                                                                                                                                                           0.7
                                                                           2.75                                   0.78
                                  0.06
                                                                           2.5                                                                             0.68
                                  0.05                                                                            0.76
                                                                           2.25
                                                                                                                                                           0.66
   0                                        0                                        0                                     0
         0       Φ(z1)        1                   0         Φ(z1)      1                   0      Φ(z1)       1                  0        Φ(z1)      1


                   (i) Edge                           (j) Power law exponent                   (k) Avg. precision                         (l) ROC AUC
                   overlap                               (input graph: 1.86)                     link prediction                         link prediction
   1                                                                                 1                                     1                               200000
                                            1
                                                                                                                  5000
                                  0.04
                                                                           0.96                                                                            180000
                                                                                                                  4000

 Φ(z2)                                                                             Φ(z2)                                 Φ(z2)
                                  0.03                                     0.94
                                                                                                                                                           160000

                                          Φ(z2)
                                                                                                                  3000
                                                                           0.92
                                  0.02                                                                                                                     140000
                                                                                                                  2000
                                                                           0.9
                                  0.01                                                                                                                     120000
                                                                                                                  1000
                                                                           0.88
   0                                                                                 0                                     0
         0       Φ(z1)        1             0                                              0      Φ(z1)       1                  0        Φ(z1)      1
                                                  0         Φ(z1)      1

              (m) Share of walks                                                                (o) Triangle count                      (p) Wedge count
                                             (n) tAvg. start node entropy
             in single community                                                               (input graph: 2,814)                  (input graph: 101,872)

Figure 9: Properties of the random walks as well as the graphs sampled from the 20 × 20 latent space bins, trained on
C ORA -ML.
                                                       NetGAN: Generating Graphs via Random Walks




   1                                        1                                     1                            0.48     1                               140
                                                                         0.5
                                  9                                                                                                                     130
                                                                                                               0.46
                                                                         0.48                                                                           120
                                  8
                                                                                                               0.44                                     110

 Φ(z2)                                    Φ(z2)                                 Φ(z2)                                 Φ(z2)
                                                                         0.46
                                  7
                                                                                                                                                        100
                                                                                                               0.42
                                                                         0.44
                                  6                                                                                                                     90
                                                                         0.42                                  0.4                                      80
                                  5
                                                                                                               0.38                                     70
                                  4                                      0.4
                                                                                                                                                        60
   0                                        0                                     0                                     0
         0       Φ(z1)        1                   0        Φ(z1)     1                  0       Φ(z1)      1                  0        Φ(z1)      1


               (a) Avg. degree            (b) Avg. share of nodes in start                   (c) Gini coefficient                     (d) Max. degree
                 of start node                      community                               (input graph: 0.404)                     (input graph: 77)
   1                                        1                                     1                            0.96     1                               2110
                                  0.0                                    13.5
                                                                                                               0.96                                     2100

                                  −0.02                                  13.0                                  0.96                                     2090


 Φ(z2)                                    Φ(z2)                                 Φ(z2)                                 Φ(z2)
                                                                                                               0.95                                     2080
                                  −0.04                                  12.5
                                                                                                               0.94
                                                                                                                                                        2070
                                  −0.06                                  12.0
                                                                                                               0.94
                                                                                                                                                        2060
                                                                         11.5                                  0.94
                                  −0.08
                                                                                                                                                        2050
   0                                        0                                     0                                     0
         0       Φ(z1)        1                   0        Φ(z1)     1                  0       Φ(z1)      1                  0        Φ(z1)      1


                (e) Assortativity                         (f) Claw count                (g) Rel. edge distr. entro-               (h) Largest conn. comp.
             (input graph: -0.022)                    (input graph: 125,701)              py (input graph: 0.96)                    (input graph: 2,110)
   1                                        1                                     1                            0.96     1
                                  0.16
                                  0.15                                   2.4                                                                            0.92
                                                                                                               0.95
                                  0.14
                                                                         2.35                                  0.94                                     0.9

 Φ(z2)                                    Φ(z2)                                 Φ(z2)                                 Φ(z2)
                                  0.14
                                  0.14                                   2.3                                   0.93
                                                                                                                                                        0.88
                                  0.13
                                                                                                               0.92
                                                                         2.25                                                                           0.86
                                  0.12
                                                                                                               0.91
                                  0.12
                                                                         2.2                                                                            0.84
                                  0.12                                                                         0.9
   0                                        0                                     0                                     0
         0       Φ(z1)        1                   0        Φ(z1)     1                  0       Φ(z1)      1                  0        Φ(z1)      1


                   (i) Edge                           (j) Power law exponent                 (k) Avg. precision                        (l) ROC AUC
                   overlap                              (input graph: 2.239)                   link prediction                        link prediction
   1                                                                              1                                     1
                                  0.08      1                                                                  400
                                                                                                                                                        35000
                                                                         0.96
                                                                                                               350
                                  0.07
                                                                         0.94                                                                           30000
                                                                                                               300

 Φ(z2)                                                                          Φ(z2)                                 Φ(z2)
                                  0.06                                   0.92

                                          Φ(z2)
                                                                                                               250                                      25000
                                                                         0.9
                                  0.05
                                                                                                               200
                                                                         0.88                                                                           20000
                                  0.04                                                                         150
                                                                         0.86
   0                                                                              0                                     0                               15000
                                                                         0.84
         0       Φ(z1)        1             0                                           0       Φ(z1)      1                  0        Φ(z1)      1
                                                  0        Φ(z1)     1

              (m) Share of walks                                                            (o) Triangle count                        (p) Wedge count
                                                (n) Avg. start node entropy
             in single community                                                            (input graph: 451)                     (input graph: 16,824)

Figure 10: Properties of the random walks as well as the graphs sampled from the 20 × 20 latent space bins, trained on
C ITESEER.
                                               NetGAN: Generating Graphs via Random Walks


F. Latent space interpolation community histrograms – C ITESEER

                                  0.9


                                  0.8


                                  0.7


                                  0.6


                                  0.5

                          Φ(z2)
                                  0.4


                                  0.3


                                  0.2


                                  0.1


                                  0.0
                                        0.0   0.1   0.2   0.3   0.4    0.5    0.6   0.7   0.8   0.9   1.0
                                                                      Φ(z1)

Figure 11: Community distributions of graphs generated by NetGAN on subregions of the latent space z, trained on the
C ITESEER network.


G. Recovering ground-truth edge probabilities
To further investigate the ability of NetGAN to capture the graph structure we perform an additional experiment with the
goal of analyzing how well we can recover the ground-truth edge probabilities given a graph generated from a prescribed
generative model. Towards that end, first, we generate a graph from DC-SBM (N = 300 nodes and 3 communities), then
we fit NetGAN on this graph, and finally we compare the ground truth edge probabilities to the edge scores inferred by
NetGAN – specifically we compute their ranking correlation. We find a correlation of 0.998 (with EO = 0.42), which shows
that NetGAN uncovered the underlying generative process, without overfitting to the input graph.


H. Hyperparameter configuration
As discussed in Sec. 4.2 NetGAN is not sensitive to the choice of most hyperparameters. For completeness, we report
here sensible defaults that we used in used in our experiments. The generator and discriminator each have a single hidden
layer with 40 and 30 hidden units respectively. The down-projection matrix for the generator is W down,g ∈ RN ×Hg with
Hg = 64, and for the discriminator is W down,d ∈ RN ×Hd with Hd = 32. The latent code z is drawn from a d = 16
dimensional multivariate standard normal distribution. We anneal the temperature from τ = 1.0 down to τ = 0.5 every 500
iterations with a multiplicative decay of 0.995. We tune the parameters p and q (used to bias the generated random walks)
for each dataset separately using the procedure in Grover & Leskovec (2016).
We use Adam (Kingma & Ba, 2014) to optimize all the parameters with a learning rate of 1e−3 and we set the regularization
strength for the L2 penalty to 1e−6. We perform five update steps for the parameters of the discriminator for each single
update step of the parameters of the generator, and we set the Wasserstein gradient penalty applied to the discriminator to 10
as suggested by Gulrajani et al. (2017). For early stopping, we evaluate the score every 500 iterations, and set the patience to
5 evaluation steps. To calculate the validation score we generate 15M transitions, e.g. for a random walk of length 16 (i.e.
15 transitions per random walk) this equals 1M random walks.
For more details we refer the reader to the provided reference implementation at https://www.kdd.in.tum.de/netgan.

