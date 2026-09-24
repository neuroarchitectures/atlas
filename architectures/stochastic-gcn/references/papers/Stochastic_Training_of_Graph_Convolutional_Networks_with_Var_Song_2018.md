# Stochastic Training of Graph Convolutional Networks with Var Song 2018

> Source: `Stochastic_Training_of_Graph_Convolutional_Networks_with_Var_Song_2018.pdf`

---

                                          Stochastic Training of Graph Convolutional Networks with Variance Reduction


                                                                                       Jianfei Chen 1 Jun Zhu 1 Le Song 2 3


                                                                    Abstract                                  outperforming multi-layer perceptron (MLP) models that
                                               Graph convolutional networks (GCNs) are power-                 do not use the graph structure, and graph embedding ap-
                                               ful deep neural networks for graph-structured data.            proaches (Perozzi et al., 2014; Tang et al., 2015; Grover &




arXiv:1710.10568v3 [stat.ML] 1 Mar 2018
                                               However, GCN computes the representation of a                  Leskovec, 2016) that do not use node features.
                                               node recursively from its neighbors, making the                However, the graph convolution operation makes GCNs dif-
                                               receptive field size grow exponentially with the               ficult to be trained efficiently. The representation of a node
                                               number of layers. Previous attempts on reducing                at layer L is computed recursively by the representations
                                               the receptive field size by subsampling neighbors              of all its neighbors at layer L − 1. Therefore, the receptive
                                               do not have a convergence guarantee, and their                 field of a single node grows exponentially with respect to
                                               receptive field size per node is still in the order            the number of layers, as illustrated in Fig. 1(a). Due to the
                                               of hundreds. In this paper, we develop control                 large receptive field size, Kipf & Welling (2017) propose
                                               variate based algorithms which allow sampling                  to train GCN by a batch algorithm, which computes the
                                               an arbitrarily small neighbor size. Furthermore,               representations of all the nodes altogether. However, batch
                                               we prove new theoretical guarantee for our algo-               algorithms cannot handle large-scale datasets because of
                                               rithms to converge to a local optimum of GCN.                  their slow convergence and the requirement to fit the entire
                                               Empirical results show that our algorithms enjoy               dataset in GPU memory.
                                               a similar convergence with the exact algorithm
                                               using only two neighbors per node. The runtime                 Hamilton et al. (2017a) make an initial attempt to develop
                                               of our algorithms on a large Reddit dataset is only            stochastic training algorithms for GCNs via a scheme of
                                               one seventh of previous neighbor sampling algo-                neighbor sampling (NS). Instead of considering all the neigh-
                                               rithms.                                                        bors, they randomly subsample D(l) neighbors at the l-th
                                                                                                              layer.
                                                                                                              Q (l)Therefore, they reduce the receptive field size to
                                                                                                                 l D , as shown in Fig. 1(b). They find that for two-layer
                                          1. Introduction                                                     GCNs, keeping D(1) = 10 and D(2) = 25 neighbors can
                                                                                                              achieve comparable performance with the original model.
                                          Graph convolution networks (GCNs) (Kipf & Welling,                  However, there is no theoretical guarantee on the conver-
                                          2017) generalize convolutional neural networks (CNNs) (Le-          gence of the stochastic training algorithm with NS. More-
                                          Cun et al., 1995) to graph structured data. The “graph              over, the time complexity of NS is still D(1) D(2) = 250
                                          convolution” operation applies same linear transformation           times larger than training an MLP, which is unsatisfactory.
                                          to all the neighbors of a node, followed by mean pooling
                                          and nonlinearity. By stacking multiple graph convolution            In this paper, we develop novel control variate-based
                                          layers, GCNs can learn node representations by utilizing            stochastic approximation algorithms for GCN. We utilize
                                          information from distant neighbors. GCNs and their vari-            the historical activations of nodes as a control variate. We
                                          ants (Hamilton et al., 2017a; Veličković et al., 2017) have       show that while the variance of the NS estimator depends
                                          been applied to semi-supervised node classification (Kipf &         on the magnitude of the activation, the variance of our algo-
                                          Welling, 2017), inductive node embedding (Hamilton et al.,          rithms only depends on the difference between the activation
                                          2017a), link prediction (Kipf & Welling, 2016; Berg et al.,         and its historical value. Furthermore, our algorithms bring
                                          2017) and knowledge graphs (Schlichtkrull et al., 2017),            new theoretical guarantees. At testing time, our algorithms
                                                                                                              give exact and zero-variance predictions, and at training
                                              1
                                                Dept. of Comp. Sci. & Tech., TNList Lab, State Key Lab for    time, our algorithms converge to a local optimum of GCN
                                          Intell. Tech. & Sys., Tsinghua University, Beijing, 100084, China
                                          2
                                                                                                              regardless of the neighbor sampling size D(l) . The the-
                                            Georgia Institute of Technology 3 Ant Financial. Correspondence   oretical results allow us to significantly reduce the time
                                          to: Jun Zhu <dcszj@mail.tsinghua.edu.cn>.
                                                                                                              complexity by sampling only two neighbors per node, yet
                                          Proceedings of the 35 th International Conference on Machine        still retain the quality of the model.
                                          Learning, Stockholm, Sweden, PMLR 80, 2018. Copyright 2018
                                          by the author(s).                                                   We empirically test our algorithms on six graph datasets, and
                            Stochastic Training of Graph Convolutional Networks with Variance Reduction

                                                                                                                                          Latest activation
Layer 2                          Layer 2                             Layer 2                                         H (2)         (2)   Historical activation
                                                                                                           GraphConv         GraphConv

                                                                                                            Dropout
Layer 1                          Layer 1                             Layer 1                                         H (1)         (1)
                                                                                                           GraphConv         GraphConv

                                                                                                            Dropout
Input                            Input                               Input

                                                                                                             Input


            (a) Exact                    (b) Neighbour sampling                (c) Control variate           (d) CVD network

                        Figure 1. Two-layer graph convolutional networks, and the receptive field of a single vertex.


     Dataset        V            E             Degree     Degree 2             label yv . We observe the labels for some vertices VL . The
     Citeseer     3,327        12,431            4           15                goal is to predict the labels for the rest vertices VU := V\VL .
       Cora       2,708        13,264            5           37
     PubMed      19,717       108,365            6           60                The edges are represented as a symmetric V × V adjacency
      NELL       65,755       318,135            5         1,597               matrix A, where Auv is the weight of the edge between u
       PPI       14,755       458,973           31          970                and v, and the propagation matrix
                                                                                                             P       P is a normalized version
                                                                                                                                       1     1
      Reddit     232,965     23,446,803         101        10,858              of A: Ã = A + I, D̃uv = v Ãuv , and P = D̃− 2 ÃD̃− 2 .
                                                                               A graph convolution layer is defined as
Table 1. Number of vertexes, edges, and average number of 1-hop
and 2-hop neighbors per node for each dataset. Undirected edges                        Z (l+1) = P H (l) W (l) ,         H (l+1) = σ(Z l+1 ),             (1)
are counted twice and self-loops are counted once.                                        (l)
                                                                               where H is the activation matrix in the l-th layer, whose
show that our techniques significantly reduce the bias and                     each row is the activation of a graph node. H (0) = X is the
variance of the gradient from NS with the same receptive                       input feature matrix, W (l) is a trainable weight matrix, and
field size. Despite sampling only D(l) = 2 neighbors, our                      σ(·) is an activation function. Denote |·| as the cardinality
algorithms achieve the same predictive performance with the                    of a set. The training loss is defined as
exact algorithm in a comparable number of epochs on all the                                            1 X
datasets, i.e., we reduce the time complexity while having                                      L=               f (yv , zv(L) ),         (2)
                                                                                                     |VL |
                                                                                                            v∈VL
almost no loss on the speed of convergence, which is the best
we can expect. On the largest Reddit dataset, the training                     where f (·, ·) is a loss function. A graph convolution layer
time of our algorithm is 7 times shorter than that of the best-                propagates information to nodes from their neighbors by
performing competitor among the exact algorithm (Kipf &                        computing the neighbor averaging P H (l) . Let n(u) be
Welling, 2017), neighbor sampling (Hamilton et al., 2017a)                     the set of neighbors of node u, and n(u) be its cardi-
and importance sampling (Chen et al., 2018) algorithms.                        nality, the neighbor averaging of node u, (P H (l) )u =
                                                                               PV           (l)     P              (l)
                                                                                  v=1 Puv hv =         v∈n(u) Puv hv , is a weighted sum of
                                                                               neighbors’ activations. Then, a fully-connected layer is ap-
2. Backgrounds
                                                                               plied on all the nodes, with a shared weight matrix W (l)
We now briefly review graph convolutional networks                             across all the nodes.
(GCNs), stochastic training, and the neighbor sampling (NS)                    We denote the receptive field of node u at layer l as all
and importance sampling (IS) algorithms.                                                         (l)                                   (L)
                                                                               the activations hv on layer l needed for computing zu .
                                                                               If the layer is not explicitly mentioned, it means layer 0.
2.1. Graph Convolutional Networks
                                                                               The receptive field of node u is all its L-hop neighbors,
We present our algorithm with a GCN for semi-supervised                        i.e., nodes that are reachable from u within L hops, as
node classification (Kipf & Welling, 2017). However, the                       illustrated in Fig. 1(a). When P = I, GCN reduces to a
algorithm is neither limited to the task nor the model. Our                    multi-layer perceptron (MLP) model which does not use the
algorithm is applicable to other models (Hamilton et al.,                      graph structure. For MLP, the receptive field of a node u is
2017a) and tasks (Kipf & Welling, 2016; Berg et al., 2017;                     just the node itself.
Schlichtkrull et al., 2017; Hamilton et al., 2017b) that in-
volve computing the average activation of neighbors.                           2.2. Stochastic Training
In the node classification task, we have an undirected graph                   It is generally expensive to compute the batch gradient
                                                                                             P                  (L)
G = (V, E) with V = |V| vertices and E = |E| edges,                            ∇L = |V1L | v∈VL ∇f (yv , zv ), which involves iterat-
where each vertex v consists of a feature vector xv and a                      ing over the entire labeled set of nodes. A possible solution
                          Stochastic Training of Graph Convolutional Networks with Variance Reduction

is to approximate the batch gradient by a stochastic gradient      receptive field size D(1) × D(2) = 250 is much larger than
                   1 X                                             that of MLP, which is 1, so the training is still expensive.
                             ∇f (yv , zv(L) ),            (3)
                 |VB |
                          v∈VB
                                                                   2.4. Importance Sampling
where VB ⊂ VL is a minibatch of labeled nodes. However,
this gradient is still expensive to compute, due to the large      FastGCN (Chen et al., 2018) is another sampling-based
receptive field size. For instance, as shown in Table 1, the       algorithm similar as NS. Instead of sampling neighbors
number of 2-hop neighbors on the NELL dataset is averagely         for each node, FastGCN directly subsample the receptive
1,597, which means computing the gradient for a single node        field for each layer altogether. Formally, it approximates
in a 2-layer GCN involves touching 1, 597/65, 755 ≈ 2.4%           (P H (l) )u with S samples v1 , . . . , vS ∈ V as
nodes of the entire graph.                                                           V
                                                                                     X                       X
                                                                                         1          V
                                                                   (P H (l) )u = V         Puv h(l)
                                                                                                v ≈                    Puv h(l)
                                                                                                                            vs /q(vs ),
In subsequent sections, two other stochasticity will be intro-                           V          S
                                                                                     v=1                   vs ∼q(v)
duced besides the random selection of the minibatch: the                                                          PV      2
random sampling of neighbors (Sec. 2.3) and the random             where P the importance distribution q(v) ∝        u=1 Puv =
                                                                     1                1
dropout of features (Sec. 5).                                      n(v)     (u,v)∈E n(u) , according the definition of P in
                                                                   Sec. 2.1. We refer to this estimator as importance sam-
2.3. Neighbor Sampling                                             pling (IS). Chen et al. (2018) show that IS performs better
                                                                   than using a uniform sample distribution q(v) ∝ 1. NS
To reduce the receptive field size, Hamilton et al. (2017a)        can be viewed asP    an IS estimator with the importance dis-
propose a neighbor sampling (NS) algorithm. NS randomly            tribution q(v) ∝ (u,v)∈E n(u)   1
                                                                                                      , because each node u has
chooses D(l) neighbors for each node at layer l and devel-                        1
ops an estimator NS(l)          (l)                                probability n(u) to choose the neighbor v. Though IS may
                    u of (P H )u based on Monte-Carlo
approximation:                                                     have a smaller variance than NS, it still only guarantees
                                                                   the convergence as the sample size S goes to infinity. Em-
                              n(u) X
      (P H (l) )u ≈ NS(l)
                       u :=                Puv hv(l) ,             pirically, we find IS to work even worse than NS because
                              D(l)    (l)                          sometimes it can select many neighbors for one node, and
                                        v∈n̂   (u)
         (l)                                                       no neighbor for another, in which case the activation of the
where n̂ (u) ⊂ n(u) is a subset of D(l) random neigh-
                                                                   latter node is just meaningless zero.
bors. Therefore, NS reduces the receptive field size from all
QLL-hop
the         neighbors to the number of sampled neighbors,
        (l)              (l)
  l=1 D . We refer NSu as the NS estimator of (P H )u ,
                                                        (l)        3. Control Variate Based Algorithm
          (l)
and (P H )u itself as the exact estimator.
                                                                   We present a novel control variate based algorithm that uti-
Neighbor sampling can also be written in a matrix form as          lizes historical activations to reduce the estimator variance.
   Z (l+1) = P̂ (l) H (l) W (l) ,   H (l+1) = σ(Z (l+1) ),   (4)
where the propagation matrix P is replaced by a sparser
                                                        (l)        3.1. Control Variate Based Estimator
unbiased estimator P̂ (l) , i.e., EP̂ (l) = P , where P̂uv =                                                       P
n(u)              (l)             (l)                                                                                             (l)
      P if v ∈ n̂ (u), and P̂uv = 0 otherwise. Hamilton
 D (l) uv
                                                                   While computing the neighbor average               v∈n(u) Puv hv ,
et al. (2017a) propose to perform the approximate forward                                                  (l)
                                                                   we cannot afford to evaluate all the hv terms because they
propagation as Eq. (4), and do stochastic gradient descent         need to be computed recursively, i.e., we again need the
(SGD) with the auto-differentiation gradient. The approxi-                      (l−1)
                                                                   activations hw     of all of v’s neighbors w.
mated gradient has two sources of randomness: the random
selection of minibatch VB ⊂ VL , and the random selection                                                   (l)               (l)
                                                                   Our idea is to maintain the history h̄v for each hv as an
of neighbors.                                                                                                     (l)
                                                                   affordable approximation. Each time when hv is computed,
                                                                                (l)       (l)             (l)        (l)
Though P̂ (l) is an unbiased estimator of P ,                      we update h̄v with hv . We expect h̄v and hv to be simi-
σ(P̂ (l) H (l) W (l) ) is not an unbiased estimator of             lar if the model weights do not change too fast during the
                                                                                              (l)    (l)      (l)
σ(P H (l) W (l) ), due to the non-linearity of σ(·). In the        training. Formally, let ∆hv = hv − h̄v , we approximate
                                                         (L)                        X                    X
sequel, both the prediction Z (L) and gradient ∇f (yv , zv )        (P H (l) )u =        Puv ∆h(l)              Puv h̄(l)   (l)
                                                                                                  v +                 v ≈ CVu
obtained by NS are biased, and the convergence of SGD                             v∈n(u)                 v∈n(u)
is not guaranteed, unless the sample size D(l) goes to                           X                       X
                                                                       n(u)
infinity. Because of the biased gradient, the sample                 := (l)                 Puv ∆h(l)
                                                                                                  v +             Puv h̄(l)
                                                                                                                        v ,          (5)
size D(l) needs to be large for NS, to keep comparable                 D
                                                                              v∈n̂(l) (u)               v∈n(u)
predictive performance with the exact algorithm. Hamilton                                    (l)                     (l)       (l)
et al. (2017a) choose D(1) = 10 and D(2) = 25, and the             where we represent hv as the sum of ∆hv and h̄v , and
                         Stochastic Training of Graph Convolutional Networks with Variance Reduction

                                                                (l)
we only apply Monte-Carlo approximation on the ∆hv                    3.3. Implementation Details and Time Complexity
                                (l)
term. Averaging over all the h̄v ’s is still affordable because       Training with the CV estimator is similar as with the NS
they do not need to be computed recursively. Since we ex-             estimator (Hamilton et al., 2017a). Particularly, each
       (l)      (l)
pect hv and h̄v to be close, ∆hv will be small and CV(l)        u     iteration of the algorithm involves the following steps:
should have a smaller variance than NS(l)      u   . Particularly,      Stochastic GCN with Variance Reduction
                                          (l)
if the model weight is kept fixed, h̄v should eventually                1. Randomly select a minibatch VB ∈ VL of nodes;
             (l)                           P                (l)
equal with hv , so that CV(l) u = 0+          v∈n(u) Puv h̄v =          2. Build a computation graph that only contains the acti-
P               (l)        (l)
                                                                                      (l)        (l)
                                                                            vations hv and h̄v needed for the current minibatch;
   v∈n(u) Puv hv = (P H )u , i.e., the estimator has zero
variance. This estimator is referred as CV. We will com-                3. Get the predictions by forward propagation as Eq. (6);
pare the variance of NS and CV estimators in Sec. 3.2 and               4. Get the gradients by backward propagation, and up-
show that the variance of CV will be eventually zero dur-                   date the parameters by SGD;
ing the training in Sec. 4. The term CV(l)        u − NSu =
                                                           (l)          5. Update the historical activations.
P               (l)  n(u) P                      (l)                  Step 3 and 4 are handled automatically by frameworks such
   v∈n(u) Puv h̄u − D (l)      v∈n̂(l) (u) Puv h̄u is a control
variate (Ripley, 2009, Chapter 5) added to the neighbor               de TensorFlow (Abadi et al., 2016). The computational
sampling estimator NS(l)                                              graph at Step 2 is defined by the receptive field r(l) and the
                       u , to reduce its variance.
                                                                      propagation matrices P̂ (l) at each layer. The receptive field
In matrix form, let H̄ (l) be the matrix formed by stacking                                                (l)
                                                                      r(l) specifies the activations hv of which nodes should be
  (l)
h̄v , then CV can be written as                                       computed for the current minibatch, according to Eq. (6).
                                                  
      Z (l+1) = P̂ (l) (H (l) − H̄ (l) ) + P H̄ (l) W (l) . (6)       We can construct r(l) and P̂ (l) from top to bottom, by
                                                                      randomly adding D(l) neighbors for each node in r(l+1) ,
                                                                                                                   (l)
3.2. Variance Analysis                                                starting with r(L) = VB . We assume hv is always needed
                                                                                     (l+1)
                                                                      to compute hv        , i.e., v is always selected as a neighbor of
We analyze the variance of the estimators assuming all the            itself. The receptive fields are illustrated in Fig. 1(c), where
features are 1-dimensional. The analysis can be extended to           red nodes are in receptive fields, whose activations hv are
                                                                                                                                  (l)
multiple dimensions by treating each dimension separately.                                             (l)
                                                                      needed, and the histories h̄v of blue nodes are also needed.
We further assume that n̂(l) (u) is created by sampling D(l)                                               (l)      (l)
                                                                      Finally, in Step 5, we update h̄v with hv for each v ∈ r(l) .
neighbors without replacement from n(u). The following                We have the pseudocode for the training in Appendix D.
proposition is proven in Appendix A:
                                                                      GCN has two main types of computation, namely, the sparse-
Proposition    1.        If     n̂(l) (u)      contains  D(l)         dense matrix multiplication (SPMM) such as P H (l) , and the
samples     from        h n(u)P without i replacement,                dense-dense matrix multiplication (GEMM) such as U W (l) .
           Varn̂(l) (u) n(u)                                          We assume that the node feature is K-dimensional and the
then                      D (l)   v∈n̂(l) (u) xv           =
  (l) P         P                                                     first hidden layer is A-dimensional.
 Cu                                               2
2D (l) v1 ∈n(u)     v2 ∈n(u) (xv1      − x v2 ) ,       where
  (l)
                                                                      For batch GCN, the time complexity is O(EK) for SPMM
Cu = 1 − (D(l) − 1)/(n(u) − 1).                                       and O(V KA) for GEMM. For our stochastic training al-
                                                 h          i         gorithm with control variates, the dominant SPMM com-
By Proposition 1, we have Varn̂(l) (u) N Su
                                                        (l)
                                                                  =   putation is the average of neighbor Qhistory P H̄ (0) for the
                                                                                                             L
   (l) P          P                                                   nodes in r , whose size is O(|VB | l=2 D(l) ). For exam-
                                                                                 (1)
 Cu                                  (l)         (l) 2                                                             (l)
2D (l)   v1 ∈n(u)   v2 ∈n(u) (Puv1 hv1 − Puv2 hv2 ) , which is        ple, in a 2-layer GCN QLwhere(l)we sample D = 2 neigh-
the total distance of the weighted activations of all pairs of        bors for each node, l=2 D = 2. Therefore, the time
                                                                                                           QL
neighbors, and is zero iff Puv hv is identical for all neighbors,     complexity of SPMM is O(V DK l=2 D(l) ) per epoch,
in which case any neighbor contains all the information of            where D is the average degree of nodes in r(1) .1 The
the entire neighborhood.                                              dominant GEMM computation is the first fully-connected
                                                     h          i     layer on all            in r(1) , whose time complexity is
The variance of the CV estimator is Varn̂(l) (u) CVu =
                                                            (l)                 QLthe nodes
                                                                                        (l)
                                                                      O(V KA l=2 D ) per epoch.
   (l) P          P
 Cu                                     (l)                   (l)
2D (l)   v1 ∈n(u)   v2 ∈n(u) (Puv1 ∆hv1      − Puv2 ∆hv2 )2 ,             1
                                                                            V D 6= E because the probability of each node to present in
                   (l)        (l)            (l)
which replaces hv by ∆hv . Since ∆hv is usually much                  r(1) is different. Nodes of higher degree have larger probability to
               (l)                                                    present. We can also subsample neighbors’ history if D is large.
smaller than hv , the CV estimator enjoys much smaller
variance than the NS estimator. Furthermore, as we will
                      (l)
show in Sec. 4.2, ∆hv converges to zero during training,
so we achieve not only variance reduction but variance
elimination, as the variance vanishes eventually.
                         Stochastic Training of Graph Convolutional Networks with Variance Reduction

4. Theoretical Results                                               4.2. Convergence Guarantee
Besides smaller variance, CV also has stronger theoretical           The following theorem shows that SGD training with the
guarantees than NS. In this section, we present two theo-            approximated gradients gCV,i (Wi ) still converges to a local
rems. One states that if the model parameters are fixed,             optimum, regardless of the neighbor sampling size D(l) .
e.g., during testing, CV produces exact predictions after L          Therefore, we can choose arbitrarily small D(l) without
epochs; and the other establishes the convergence towards a          worrying about the convergence.
local optimum regardless of the neighbor sampling size.
In this section, we assume that the algorithm is run by
epochs. In each epoch, we randomly partition the vertex set          Theorem 2. Assume that (1) the activation σ(·) is ρ-
V as I minibatches V1 , . . . , VI , and in the i-th iteration, we   Lipschitz, (2) the gradient of the cost function ∇z f (y, z)
run a forward pass to compute the predictions for nodes in           is ρ-Lipschitz and bounded, (3) kgCV,V (W )k∞ , kg(W )k∞ ,
Vi , an optional back propagation to compute the gradients,          and k∇L(W )k∞ are all bounded by G > 0 for all P̂ , V and
and update the history. Note that in each epoch we scan all          W . (4) The loss L(W ) is ρ-smooth, i.e., |L(W2 )−L(W1 )−
                                                                                                                2
the nodes instead of just training nodes, to ensure that the         h∇L(W1 ), W2 −W1 i| ≤ ρ2 kW2 − W1 kF ∀W1 , W2 , where
history of each node is updated at least once per epoch.                             >
                                                                     hA, Bi = tr(A B) is the inner product of matrix A and ma-
We denote the model parameters in the i-th iteration as              trix B. Then, there exists K > 0, s.t., ∀N > LI, if we run
Wi . At training time, Wi is updated by SGD over time; at            SGD for R ≤ N iterations, where R is chosen uniformly
testing time, Wi is kept fixed. To distinguish, the activa-          from [N ]+ , we have
                                                      (l)
tions produced by CV at iteration i are denoted as ZCV,i                                 2     L(W1 ) − L(W ∗ ) + K + ρK
                                                                        ER k∇L(WR )kF ≤ 2                   √                ,
        (l)                                                                                                     N
and HCV,i , and the activations produced by the exact algo-
                                (l)       (l)                        for the updates Wi+1 = Wi − γgCV,i (Wi ) and the step size
rithm (Eq. 1) are denoted as Zi and Hi . At iteration i, the         γ = min{ ρ1 , √1N }.
network computes the predictions and gradients for the mini-
                                      P                   (L)
batch Vi , where gCV,i (Wi ) := |V1i | v∈Vi ∇f (yv , zCV,i,v )
                      P                  (L)
and gi (Wi ) := |V1i | v∈Vi ∇f (yv , zi,v ) are the stochas-
                                                                                                               2
tic gradients computed by CV and the exact algorithm.                Particularly, limN →∞ ER k∇L(WR )k = 0. Therefore,
                   P                  (L)
∇L(Wi ) = |V1L | v∈VL ∇f (yv , zv ) is the determinis-               our algorithm converges to a local optimum as the max
tic batch gradient computed by the exact algorithm. The              number of iterations N goes to infinity. The full proof is in
subscript i may be omitted for the exact algorithm if Wi             Appendix C. For short, we show that gCV,i (Wi ) is unbiased
is a constant sequence. We let [L] = {0, . . . , L} and              as i → ∞, and then show that SGD with such asymptoti-
[L]+ = {1, . . . , L}. The gradient gCV,i (Wi ) has two              cally unbiased gradients converges to a local optimum.
sources of randomness: the random selection of the mini-
batch Vi and randomness of the neighbors P̂ , so we may              5. Handling Dropout of Features
take expectation of gCV,i (Wi ) w.r.t. either Vi or P̂ , or both.
                                                                     In this section, we consider introducing a third source of ran-
4.1. Exact Testing                                                   domness, the random dropout of features (Srivastava et al.,
                                                                     2014). Let Dropoutp (X) = M ◦ X be the dropout oper-
The following theorem reveals the connection of the exact            ation, where Mij ∼ Bern(p) are i.i.d. Bernoulli random
and approximate predictions by CV.                                   variables, and ◦ is the element-wise product. Let EM be the
Theorem 1. For a constant sequence of Wi = W and any                 expectation over dropout masks.
i > LI (i.e., after L epochs), the activations computed                                                      (l)
                                                                     With dropout, all the activations hv are random vari-
                        (l)
by CV are exact, i.e., ZCV,i = Z (l) for each l ∈ [L] and            ables whose randomness comes from dropout, even in
  (l)
HCV,i = H (l) for each l ∈ [L − 1].                                  the exact algorithm Eq. (1). We want to design a
                                                                     cheap estimator for the random variable (P H (l) )u =
Theorem 1 shows that at testing time, we can run forward             P              (l)
                                                                        v∈n(u) Puv hv , based on a stochastic neighborhood
propagation with CV for L epoches and get exact predic-
tion. This outperforms NS, which cannot recover the exact            n̂(l) (u). An ideal estimator should have the same dis-
prediction unless the neighbor sample size goes to infinity.         tribution with (P H (l) )u . However, such an estimator
Comparing with directly making exact predictions by an               is difficult to design. Instead, we develop an estimator
exact batch algorithm, CV is more scalable because it does           CVD(l) u that eventually has the same mean and variance
not need to load the entire graph into memory. The proof             with (P H (l) )u , i.e., En̂(l) (u) EM CVD(l)
                                                                                                               u = EM (P H )u
                                                                                                                          (l)

can be found in Appendix B.                                          and Varn̂(l) (u) VarM CVD(l)          (l)
                                                                                              u = VarM (P H )u .
                            Stochastic Training of Graph Convolutional Networks with Variance Reduction

  Esti.                    VNS                      VD             set n(u) without replacement, x1 , . . . , xV are random
                                                     (l)
                                                                   variables, ∀v, E [xv ] = 0 andh ∀v1P6= v2 , Cov [xiv1 , xv2 ] =
 Exact                      0                      Su
                    (l)        (l)               n(u) (l)          0,     then     VarX,n̂(l) (u) n(u)
                                                                                                  D (l)  v∈n̂(l) (u) xv          =
  NS         (Puv1 µv1 − Puv2 µv2 )2           D(l) Su           n(u)  P
  CV
                    (l)           (l)
           (Puv1 ∆µv1 − Puv2 ∆µv2 )2           3 + n(u)
                                                          (l)
                                                         Su        D (l)   v∈n(u) Var [xv ] .
                                                   D (l)
                      (l)               (l)          (l)           Proposition 3. X and Y are two random variables, and
  CVD      (Puv1 ∆µv1 − Puv2 ∆µv2 )2               Su
                                                                   f (X, Y ) and g(Y ) are two functions. If EX f (X, Y ) =
Table 2. Variance of different estimators. To save space we omit   0, then VarX,Y [f (X, Y ) + g(Y )] = VarX,Y f (X, Y ) +
  (l) P
 Cu                                                                VarY g(Y ).
2D (l)   v1 ,v2 ∈n(u) before all the VNS terms.

5.1. Control Variate for Dropout                                   By Proposition 3, Varn̂ VarM         CVD(l)   can be i writ-
                                                                                                     h√   Pu
                     (l)        (l)    (l)                                                                             (l)
With dropout, ∆hv = hv − h̄v is not necessarily small              ten as the sum of Varn̂ VarM          R v∈n̂ Puv h̊v       and
          (l)     (l)                                                   h P                    P                     i
even if h̄v and hv have the same distribution. We develop                                 (l)                    (l)
                                                                   Varn̂ R v∈n̂ Puv ∆µv + v∈n(u) Puv µ̄v . We refer
another stochastic approximation algorithm, control variate
                                                                   the first term as the variance from dropout (VD) and
for dropout (CVD), that works well with dropout.
                                                                   the second term as the variance from neighbor sam-
Our method is based on the weight scaling procedure (Sri-          pling (VNS). Ideally, VD should equal to the variance
vastava et al.,
            h 2014) i to approximately compute the mean            of (P H (l) )u and VNS should be zero. VNS can be de-
 (l)            (l)                                                rived by replicating the analysis in Sec. 3.2, replacing h
µv := EM hv . That is, along with the dropout model,
                                                                                   (l)          (l)             (l)        (l)
we can run a copy of the model without dropout to obtain the       with µ. Let sv = VarM hv = VarM h̊v , and Su =
                                                                                       P              (l)
       (l)                                                         VarM (P H (l) )u = v∈n(u) Puv  2
                                                                                                     s , By Proposition 2, VD
mean µv , as illustrated in Fig. 1(d). We obtain a stochastic
                                                                                P               h vi
                                                                           (l)            2         (l)     (l)
approximation by separating the mean and variance                  of CVDu , v∈n(u) Puv Var h̊v = Su , equals with the
                                                                   VD of the exact estimator as desired.
                  X
                                                      (l)
(P H (l) )u =             Puv (h̊(l)   (l)   (l)
                                 v + ∆µv + µ̄v ) ≈ CVDu            We summarize the estimators and their variances in Table 2,
                 v∈n(u)                                            where the derivations are in Appendix A. As in Sec. 3.2,
      √ X            X           X                                 VNS of CV and CVD depends on ∆µv , which converges to
 :=    R  Puv h̊(l)
                v +R   Puv ∆µ(l)
                             v +   Puv µ̄(l)
                                         v ,
                                                                   zero as the training progresses, while VNS of NS depends
          v∈n̂                  v∈n̂           v∈n(u)
                                                                   on the non-zero µv . On the other hand, CVD is the only
                                                         (l)
where n = n̂(l) (u), R = n(u)/D(l) for short, h̊v =                estimator except the exact one that gives correct VD.
  (l)   (l)   (l)
hv −µv , µ̄v is the historical mean activation, obtained by
          (l)            (l)        (l)      (l)     (l)
storing µv instead of hv , and ∆µv = µv − µ̄v . We sep-            5.3. Preprocessing Strategy
       (l)                                              (l)
arate hv as three terms, the latter two terms on µv do not         There are two possible models adopting dropout,
                                            (l)
have the randomness from dropout, and µv are treated as if         Z (l+1)     =     P Dropoutp (H (l) )W (l) or Z (l+1)       =
  (l)
hv for the CV estimator. The first term has zero mean w.r.t.       Dropoutp (P H (l) )W (l) .     The difference is whether
                    (l)
dropout, i.e., EM h̊v = 0. We have En̂(l) (u) EM CVD(l)     u =    the dropout layer is before or after neighbor averaging. Kipf
      P                 (l)    (l)
0 + v∈n(u) Puv (∆µv + µ̄v ) = EM (P H )u , i.e., the
                                                 (l)               & Welling (2017) adopt the former one, and we adopt the
estimator is unbiased, and we shall see that the estimator         latter one, while the two models performs similarly in prac-
                                        (l)
eventually has the correct variance if hv ’s are uncorrelated      tice, as we shall see in Sec. 6.1. The advantage of the latter
in Sec. 5.2.                                                       model is that we can preprocess U (0) = P H (0) = P X and
                                                                   takes U (0) as the new input. In this way, the actual number
5.2. Variance Analysis                                             of graph convolution layers is reduced by one — the first
                                                                   layer is merely a fully-connected layer instead of a graph
We analyze the variance under the assumption    h      that ithe   convolution one. Since most GCNs only have two graph
                                                   (l)   (l)
node activations are uncorrelated, i.e., CovM hv1 , hv2 =          convolution layers (Kipf & Welling, 2017; Hamilton et al.,
0, ∀v1 6= v2 . We report the correlation between nodes em-         2017a), this gives a significant reduction of the receptive
pirically in Appendix G. To facilitate the analysis of the vari-   field size and speeds up the computation. We refer this
ance, we introduce two propositions proven in Appendix A .         optimization as the preprocessing strategy.
The first helps the derivation of the dropout variance; and
the second implies that we can treat the variance introduced       6. Experiments
by neighbor sampling and by dropout separately.
                                                                   We examine the variance and convergence of our algo-
Proposition 2. If n̂(l) (u) contains D(l) samples from the         rithms empirically on six datasets, including Citeseer, Cora,
                                Stochastic Training of Graph Convolutional Networks with Variance Reduction

                                                                                                             citeseer                                cora
       Dataset             M0                  M1             M1+PP                      0.72                                       0.80
       Citeseer         70.8 ± .1           70.9 ± .2        70.9 ± .2                   0.71                                       0.79
         Cora           81.7 ± .5           82.0 ± .8        81.9 ± .7
       PubMed           79.0 ± .4           78.7 ± .3        78.9 ± .5                   0.70                                       0.78
        NELL                -              64.9 ± 1.7        64.2 ± 4.6                  0.690          50     100        150
                                                                                                                                    0.77
                                                                                                                                 200 0        50      100      150    200
         PPI           97.9 ± .04          97.8 ± .05        97.6 ± .09                                      pubmed                                   nell
                                                                                                                                    0.675
        Reddit         96.2 ± .04          96.3 ± .07        96.3 ± .04                  0.80
Table 3. Testing accuracy of different algorithms and models after                                                                  0.650
fixed number of epochs. Our implementation does not support M0                           0.78                                       0.625
on NELL so the result is not reported.                                                                                              0.6000
                                                                                                0       50     100        150    200          100     200      300    400
                 citeseer                                    cora                                             reddit                                  ppi
1.0                                     1.0
                                                                                         0.965                                      0.96
                                        0.8
0.8                                                                                      0.960                                      0.94
                                        0.6
0.6                                     0.4                                              0.955                                      0.92
0.40                                    0.2                                              0.9500         10   20      30     40   500.900     20     40    60     80 100
           50      100        150    200 0           50         100        150     200
                 pubmed                                      nell
0.8                                      1.5                                                    M1+PP         NS           NS+PP       IS+PP         CV+PP         CVD+PP
0.6                                                                                      Figure 3. Comparison of validation accuracy with respect to num-
                                           1.0
0.4                                                                                      ber of epochs. NS converges to 0.94 on the Reddit dataset and 0.6
                                           0.5                                           on the PPI dataset.
0.2
   0       50      100        150    200         0   100        200        300     400   three settings performs similarly, i.e., switching the order
                   reddit                                       ppi
0.100                                                                                    does not affect the predictive performance. Therefore, we
0.075                                   0.04                                             use the fastest M1+PP as the exact baseline in following
0.050                                                                                    convergence experiments.
                                        0.02
0.025
0.0000     10     20     30     40    500.000                                            6.2. Convergence Results
                                                     20    40         60     80    100
         M1+PP           NS           NS+PP                IS+PP                 CV+PP
                                                                                         Having the M1+PP algorithm as an exact baseline, the next
Figure 2. Comparison of training loss with respect to number of                          goal is reducing the time complexity per epoch to make it
epochs without dropout. The CV+PP curve overlaps with the Exact                          comparable with the time complexity of MLP, by setting
curve in the first four datasets. The training loss of NS and IS+PP                      D(l) = 2. We cannot set D(l) = 1 because GraphSAGE
are not shown on some datasets because they are too high.                                explicitly need the activation of a node itself besides the
PubMed and NELL from Kipf & Welling (2017) and Red-                                      average of its neighbors. Four approximate algorithms are
dit, PPI from Hamilton et al. (2017a), as summarized in                                  included for comparison: (1) NS, which adopts the NS es-
Table 1, with the same train / validation / test splits. To mea-                         timator with no preprocessing. (2) NS+PP, which is same
sure the predictive performance, we report Micro-F1 for the                              with NS but uses preprocessing. (3) CV+PP, which adopts
multi-label PPI dataset, and accuracy for all the other multi-                           the CV estimator and preprocessing. (4) CVD+PP, which
class datasets. The model is GCN for the former 4 datasets                               uses the CVD estimator. All the four algorithms have sim-
and GraphSAGE (Hamilton et al., 2017a) for the latter 2                                  ilar low time complexity per epoch with D(l) = 2, while
datasets, see Appendix E for the details on the architectures.                           M1+PP takes D(l) = 20. We study how much convergence
We repeat the convergence experiments 10 times on Citeseer,                              speed per epoch and model quality do these approximate
Cora, PubMed and NELL, and 5 times on Reddit and PPI.                                    algorithms sacrifice comparing with the M1+PP baseline.
The experiments are done on a Titan X (Maxwell) GPU.
                                                                                         We set the dropout rate as zero and plot the training loss
                                                                                         with respect to number of epochs as Fig. 2. We can see that
6.1. Impact of Preprocessing
                                                                                         CV+PP can always reach the same training loss with M1+PP,
We first examine the impact of switching the order of                                    while NS, NS+PP and IS+PP have higher training losses
dropout and computing neighbor averaging in Sec. 5.3.                                    because of their biased gradients. CVD+PP is not included
Let M0 be the Z (l+1) = P Dropoutp (H (l) )W (l) model                                   because it is the same with CV+PP when the dropout rate
by (Kipf & Welling, 2017), and M1 be our Z (l+1) =                                       is zero. The results matches the conclusion of Theorem 2,
Dropoutp (P H (l) )W (l) model, we compare three settings:                               which states that training with the CV estimator converges
M0 and M1 are exact algorithms without any neighbor sam-                                 to a local optimum of Exact, regardless of D(l) .
pling, and M1+PP samples a large number of D(l) = 20                                     Next, we turn dropout on and compare the validating accu-
neighbors and preprocesses P H (0) so that the first neigh-                              racy obtained by the model trained with different algorithms
bor averaging is exact. In Table 3 we can see that all the
                                        Stochastic Training of Graph Convolutional Networks with Variance Reduction
                                                                                                                      Bias (without dropout)
                            Alg.   Valid. acc.    Epochs         Time (s)                               7.5
                                                                                                                                                         Algorithm

                                                                                    Gradient Bias
                        Exact          96.0          4.8           252                                  5.0                                                  NS
                                                                                                                                                             NS+PP
                         NS            94.4        102.0           445                                  2.5                                                  CV+PP
                       NS+PP           96.0         39.8           161                                                                                       Exact
                       IS+PP           95.8         52.0           251                                  0.0 cora pubmed nell citeseer ppi reddit
                                                                                                                          Dataset
                       CV+PP           96.0          7.6            39                                     Standard deviation (without dropout)
                      CVD+PP           96.0          6.8            37                                  40



                                                                                   Gradient Std. Dev.
Table 4. Time complexity comparison of different algorithms on
                                                                                                                                                         Algorithm
                                                                                                                                                             NS
the Reddit dataset.                                                                                     20                                                   NS+PP
                                                                                                                                                             CV+PP
                                                                                                                                                             Exact
                      1.0                                                                                   0 cora pubmed nell citeseer ppi     reddit


   Testing accuracy
                                                                   Algorithm                                               Dataset
                                                                                                                      Bias (with dropout)
                      0.5                                              NS
                                                                       NS+PP                            3                                                Algorithm

                                                                                    Gradient Bias
                                                                       CV                                                                                   NS
                                                                       Exact                            2                                                   NS+PP
                      0.0 cora pubmed nell citeseer ppi reddit                                                                                              CV+PP
                                       Dataset                                                          1                                                   CVD+PP
Figure 4. Comparison of the accuracy of different testing algo-                                         0 cora pubmed nell citeseer ppi                     Exact
rithms. The y-axis is Micro-F1 for PPI and accuracy otherwise.
                                                                                                                                               reddit
                                                                                                                        Dataset
at each epoch. Regardless of the training algorithm, the                                                       Standard deviation (with dropout)



                                                                                  Gradient Std. Dev.
exact algorithm is used for computing predictions on the                                                                                                 Algorithm
                                                                                                       20                                                   NS
validating set. The result is shown in Fig. 3. We find that                                                                                                 NS+PP
when dropout is present, CVD+PP is the only algorithm                                                  10                                                   CV+PP
that can reach comparable validation accuracy with the ex-                                                                                                  CVD+PP
                                                                                                        0 cora pubmed nell citeseer ppi reddit              Exact
act algorithm on all datasets. Furthermore, its convergence
speed with respect to the number of epochs is comparable                                                               Dataset
                                                                               Figure 5. Bias and standard deviation of the gradient for different
with M1+PP, implying almost no loss of the convergence
                                                                               algorithms during training.
speed despite its D(l) is 10 times smaller. This is already the
best we can expect - comparable time complexity with MLP,                      ing accuracy as the exact algorithm, while NS and NS+PP
yet similar model quality with GCN. CVD+PP performs                            perform much worse.
much better than M1+PP on the PubMed dataset, we sus-                          Finally, we compare the average bias and variance of the
pect it finds a better local optimum. Meanwhile, the simpler                   gradients per dimension for first layer weights relative to
CV+PP also reaches a comparable accuracy with M1+PP                            the magnitude of the weights in Fig. 5. For models without
for all datasets except PPI. IS+PP works worse than NS+PP                      dropout, the gradient of CV+PP is almost unbiased. For
on the Reddit and PPI datasets, perhaps because sometimes                      models with dropout, the bias and variance of CV+PP and
nodes can have no neighbor selected, as we mentioned in                        CVD+PP are usually smaller than NS and NS+PP.
Sec. 2.4. Our accuracy result for IS+PP can match the re-
sult reported by Chen et al. (2018), while their NS baseline,
GraphSAGE (Hamilton et al., 2017a), does not implement                         7. Conclusions
the preprocessing technique in Sec. 5.3.                                       The large receptive field size of GCN hinders its fast stochas-
                                                                               tic training. In this paper, we present control variate based
6.3. Further Analysis on Time Complexity, Testing                              algorithms to reduce the receptive field size. Our algorithms
     Accuracy and Variance                                                     can achieve comparable convergence speed with the exact
Table 4 reports the average number of epochs and time                          algorithm even the neighbor sampling size D(l) = 2, so that
to reach a given 96% validation accuracy on the largest                        the per-epoch cost of training GCN is comparable with train-
Reddit dataset. Sparse and dense computations are defined                      ing MLPs. We also present strong theoretical guarantees,
in Sec. 3.3. We found that CVD+PP is about 7 times faster                      including exact prediction and the convergence to a local
than M1+PP due to the significantly reduced receptive field                    optimum.
size. Meanwhile, NS and IS+PP does not converge to the
given accuracy.
                                                                               References
We compare the quality of the predictions made by different
                                                                               Abadi, Martı́n, Barham, Paul, Chen, Jianmin, Chen, Zhifeng,
algorithms, using the same model trained with M1+PP in
                                                                                 Davis, Andy, Dean, Jeffrey, Devin, Matthieu, Ghemawat,
Fig. 4. As Theorem 1 states, CV reaches the same test-
                       Stochastic Training of Graph Convolutional Networks with Variance Reduction

  Sanjay, Irving, Geoffrey, Isard, Michael, et al. Tensorflow:   Tang, Jian, Qu, Meng, Wang, Mingzhe, Zhang, Ming, Yan,
  A system for large-scale machine learning. In OSDI,              Jun, and Mei, Qiaozhu. Line: Large-scale information net-
  volume 16, pp. 265–283, 2016.                                    work embedding. In Proceedings of the 24th International
                                                                   Conference on World Wide Web, pp. 1067–1077. Interna-
Berg, Rianne van den, Kipf, Thomas N, and Welling, Max.            tional World Wide Web Conferences Steering Committee,
  Graph convolutional matrix completion. arXiv preprint            2015.
  arXiv:1706.02263, 2017.
                                                                 Veličković, Petar, Cucurull, Guillem, Casanova, Aran-
Chen, Jie, Ma, Tengfei, and Xiao, Cao. Fastgcn: Fast learn-        txa, Romero, Adriana, Liò, Pietro, and Bengio,
  ing with graph convolutional networks via importance             Yoshua. Graph attention networks. arXiv preprint
  sampling. arXiv preprint arXiv:1801.10247, 2018.                 arXiv:1710.10903, 2017.

Grover, Aditya and Leskovec, Jure. node2vec: Scalable
  feature learning for networks. In Proceedings of the 22nd
  ACM SIGKDD international conference on Knowledge
  discovery and data mining, pp. 855–864. ACM, 2016.

Hamilton, William L, Ying, Rex, and Leskovec, Jure. In-
  ductive representation learning on large graphs. arXiv
  preprint arXiv:1706.02216, 2017a.

Hamilton, William L, Ying, Rex, and Leskovec, Jure. Rep-
  resentation learning on graphs: Methods and applications.
  arXiv preprint arXiv:1709.05584, 2017b.

Kipf, Thomas N and Welling, Max. Variational graph auto-
  encoders. arXiv preprint arXiv:1611.07308, 2016.

Kipf, Thomas N and Welling, Max. Semi-supervised clas-
  sification with graph convolutional networks. In ICLR,
  2017.

LeCun, Yann, Bengio, Yoshua, et al. Convolutional net-
  works for images, speech, and time series. The hand-
  book of brain theory and neural networks, 3361(10):1995,
  1995.

Perozzi, Bryan, Al-Rfou, Rami, and Skiena, Steven. Deep-
  walk: Online learning of social representations. In Pro-
  ceedings of the 20th ACM SIGKDD international con-
  ference on Knowledge discovery and data mining, pp.
  701–710. ACM, 2014.

Ripley, Brian D. Stochastic simulation, volume 316. John
  Wiley & Sons, 2009.

Schlichtkrull, Michael, Kipf, Thomas N, Bloem, Peter, Berg,
  Rianne van den, Titov, Ivan, and Welling, Max. Modeling
  relational data with graph convolutional networks. arXiv
  preprint arXiv:1703.06103, 2017.

Srivastava, Nitish, Hinton, Geoffrey E, Krizhevsky, Alex,
  Sutskever, Ilya, and Salakhutdinov, Ruslan. Dropout: a
  simple way to prevent neural networks from overfitting.
  Journal of machine learning research, 15(1):1929–1958,
  2014.
     Stochastic Training of Graph Convolutional
         Networks with Variance Reduction:
               Supplementary Material



A        Derivation of the variance
The following proposition is widely used in this section.

Proposition A. Let X1 , . . . , XN are random variables, then
                      "N          #    N
                         X            X NX
                  Var           Xi =        Cov [Xi , Xj ] .
                                 i=1               i=1 j=1

Proof.
                     "N          #                                                 !2
                      X                  XN X
                                            N                             N
                                                                          X
               Var          Xi       = E     Xi Xj  −               E         Xi
                      i=1                   i=1 j=1                       i=1
                                        
                                           N X N            N
                                                                 !
                                         X               1  X
                                     = E         Xi Xj − E    Xi 
                                          i=1 j=1
                                                         N i=1
                                           N
                                         N X
                                         X
                                     =             Cov [Xi , Xj ] .
                                         i=1 j=1




    We begin with the proof for the three propositions in the main text.
                        (l)                (l)
Proposition 1. If n̂   h (u)Pcontains Di samples      from n(u) without replace-
                                                (l) P       P
                        n(u)                   Cu
ment, then Varn̂(l) (u) D(l) v∈n̂(l) (u) xv = 2D(l) v1 ∈n(u) v2 ∈n(u) (xv1 −xv2 )2 ,
         (l)
where Cu = 1 − (D(l) − 1)/(n(u) − 1).
                                                                              1
                                                                                          P
Proof. We denote the D(l) samples in the set as v1 , . . . , vD(l) .Let x̄ = n(u)             v∈n(u) xv ,




                                                    1
then
                                    
                n(u)    X
 Varn̂(l) (u)  (l)               xv 
                D
                      v∈n̂(l) (u)
                           (l)
                                    
                          D
                          X
                    n(u)
=Varv1 ,...,vD(l)  (l)         xv 
                    D i=1 i
              2 D
                  XD
                    (l) (l)
                        X
        n(u)                                            
=                             Covv1 ,...,vD(l) xvi , xvj
        D(l)      i=1 j=1
                    (l)                                   
               2 D
        n(u)        X    2 X                           
=                 Varvi xvi +         Covvi ,vj xvi , xvj
        D(l)                                              
              i=1                i6=j
                                                                                                            
        2
    n(u)  D(l) X                   2   D (l)
                                              (D (l)
                                                     − 1)  
                                                               X                             X
                                                                                                             2
                                                                                                               
=                         (x v − x̄)  +                              (x i − x̄)(x j − x̄) −        (x i − x̄)
    D(l)     n(u)                      n(u)(n(u) − 1)                                                         
                   v∈n(u)                                   i,j∈n(u)                        i∈n(u)
                                               
                     
  n(u)       D(l) − 1  X 2
= (l) 1 −                         xv − n(u)x̄2 
  D          n(u) − 1
                           v∈n(u)
                                                                   
                                   X              X
    1         D(l) − 1 
= (l) 1 −                   2n(u)        x2v −             2xv1 xv2 
  2D          n(u) − 1
                                              v∈n(u)         v1 ,v2 ∈n(u)
   (l)
  Cu             X
= (l)                   (xv1 − xv2 )2 .
 2D
           v1 ,v2 ∈n(u)




Proposition 2. If n̂(l) (u) contains D(l) samples from the set n(u) without re-
placement, x1 , . . . , xhV are random variables,
                                            i      ∀v, E [xv ] = 0 and ∀v1 6= v2 , Cov [xv1 , xv2 ] =
                           n(u) P                 n(u) P
0, then VarX,n̂(l) (u) D(l) v∈n̂(l) (u) xv = D(l) v∈n(u) Var [xv ] .




                                                       2
Proof.
                                            
                        n(u)        X
             VarX,n̂(l) (u)             xv 
                        D(l)     (l)
                             v∈n̂ (u)
                         (l)                                            
                 2    DX                      X                       
             n(u)                          2
         =           EX       E n̂(l) (u) xv +        En̂(l) (u) xvi xvj
             D(l)                           i
                                                                         
                          i=1                    i6=j
                                                                                            
                 2     D(l) X                                          X                  
             n(u)                           2    D(l) (D(l) − 1)
         =           EX                   x i  +                                     xi xj
             D(l)        n(u)                   n(u)(n(u) − 1)                              
                                    i∈n(u)                           i,j∈n(u),i6=j

             n(u) X
         =          Var [xi ] .
             D(l)
                   i∈n(u)




Proposition 3. X and Y are two random variables, and f (X, Y ) and g(Y ) are
two functions. If EX f (X, Y ) = 0, then VarX,Y [f (X, Y ) + g(Y )] = VarX,Y f (X, Y )+
VarY g(Y ).
Proof.

VarX,Y [f (X, Y ) + g(Y )] = VarX,Y f (X, Y ) + VarY g(Y ) + 2CovX,Y [f (X, Y ), g(Y )],

where

 CovX,Y [f (X, Y ), g(Y )] = EY EX [(f (X, Y ) − EX,Y f (X, Y ))(g(Y ) − EY g(Y ))]
                                = EY [(EX f (X, Y ) − 0)(g(Y ) − EY g(Y ))]
                                = EY [0(g(Y ) − EY g(Y ))] = 0.



    Then, we derive the variance of the estimators with dropout is present.

A.1         Variance of the exact estimator
                                                     
              X                            X                     X               h      i
VarM               Puv h(l)
                         v
                              = VarM           Puv h̊(l)
                                                       v
                                                           =             2
                                                                         Puv VarM h̊(l)
                                                                                    v     = Su(l) .
          v∈n(u)                        v∈n(u)                  v∈n(u)




                                                 3
A.2     Variance of the NS estimator
                  h     i
   Varn̂(l) (u),M NS(l)
                      u
                                              
                    n(u)   X
  =Varn̂(l) (u),M  (l)              Puv h(l)
                                           v
                                               
                    D
                         v∈n̂(l) (u)
                                                      
                    n(u)   X
                                                   (l) 
  =Varn̂(l) (u),M  (l)              Puv (h̊(l)
                                            v + µv )
                    D
                         v∈n̂(l) (u)
                                                                                            
                    n(u)   X                                                X
  =Varn̂(l) (u),M  (l)              Puv h̊(l)
                                           v
                                                + Varn̂(l) (u)  n(u)                 Puv µ(l)
                                                                                            v
                                                                                                ,
                    D        (l)
                                                                  D(l)
                             v∈n̂   (u)                                  v∈n̂(l) (u)

where the last equality is by Proposition 3. By Proposition 2, VD is
                                                                
                                      n(u)   X
                     Varn̂(l) (u),M  (l)              Puv h̊(l)
                                                             v
                                                                 
                                      D
                                           v∈n̂(l) (u)
                                                                   
                                         n(u)      X
                   =VarM Varn̂(l) (u)  (l)               Puv h̊(l)
                                                                  v
                                                                    
                                         D          (l)
                                                   v∈n̂   (u)

                       n(u)
                      = (l) Su(l) ,
                       D
                P              h          i
         (l)                          (l)
where Su = v∈n(u) VarM Puv hv is defined in Sec. 5.2. By Proposition 1,
VNS is
                                     
                       X                      (l) X
                 n(u)
  Varn̂(l) (u)  (l)         Puv µ(l)
                                    v
                                       = Cu        (Puv1 µ(l)        (l) 2
                                                           v1 − Puv2 µv2 ) .
                 D                          2D(l)
                        (l)
                      v∈n̂    (u)                     v1 ,v2 ∈n(u)


A.3     Variance of the CVD estimator
                h       i
 Varn̂(l) (u),M CVD(l)
                     u
                r                                                                          
                   n(u)    X                   n(u)    X                    X
=Varn̂(l) (u),M                  Puv h̊(l)
                                        v +                     Puv ∆µ(l)
                                                                      v +         Puv µ̄(l)
                                                                                        v
                                                                                            
                   D(l)                        D (l)
                             (l)
                         v∈n̂ (u)                         (l)
                                                     v∈n̂ (u)              v∈n(u)
                r                                                                                    
                   n(u)    X                                          X                     X
=Varn̂(l) (u),M                  Puv h̊(l)
                                        v
                                             + Varn̂(l) (u)  n(u)         Puv ∆µ(l)
                                                                                   v +        Puv µ̄(l)
                                                                                                    v
                                                                                                        ,
                   D(l)      (l)
                                                                D (l)
                                                                       (l)
                             v∈n̂   (u)                                  v∈n̂   (u)                  v∈n(u)




                                              4
where the last equality is by Proposition 3. By Proposition 2, VD is
                                 r                           
                                    n(u)     X
                  Varn̂(l) (u),M                   Puv h̊(l)
                                                          v
                                                              
                                    D(l)       (l)
                                           v∈n̂ (u)
                                                                    
                   D(l)                     n(u)    X
                 =      VarM Varn̂(l) (u)  (l)            Puv h̊(l)
                                                                 v
                                                                     
                   n(u)                     D        (l)       v∈n̂   (u)

                    =Su(l) .

By Proposition 1, VNS is
                                                                             
                                n(u)   X                      X
                 Varn̂(l) (u)  (l)              Puv ∆µ(l)
                                                       v +          Puv µ̄(l)
                                                                          v
                                                                              
                                D
                                     v∈n̂(l) (u)             v∈n(u)
                                                          
                                n(u) X
                =Varn̂(l) (u)  (l)              Puv ∆µ(l)
                                                       v
                                                           
                                D        (l)
                                          v∈n̂   (u)
                   (l)
                  Cu           X
                = (l)                   (Puv1 ∆µ(l)         (l) 2
                                                v1 − Puv2 ∆µv2 ) .
                 2D
                          v1 ,v2 ∈n(u)


A.4     Variance of the CV estimator
                h      i
 Varn̂(l) (u),M CV(l)
                    u
                                                               
                  n(u)   X                      X
=Varn̂(l) (u),M  (l)              Puv ∆h(l)
                                          v +         Puv h̄(l)
                                                            v
                                                                
                  D        (l)
                       v∈n̂ (u)                v∈n(u)
                                                                                                           
                  n(u)   X                      X         ¯        n(u)   X                 X
=Varn̂(l) (u),M  (l)              Puv ∆h̊(l)
                                          v +         Puv h̊(l)
                                                            v +                  Puv ∆µ(l)
                                                                                       v +        Puv µ̄(l)
                                                                                                        v
                                                                                                            
                  D                                                D(l)
                           (l)
                       v∈n̂ (u)                v∈n(u)                       (l)
                                                                        v∈n̂ (u)           v∈n(u)
                                                               
                  n(u) X                        X         ¯ 
=Varn̂(l) (u),M  (l)              Puv ∆h̊(l)
                                          v +         Puv h̊(l)
                                                            v
                  D
                       v∈n̂(l) (u)             v∈n(u)
                                                               
                  n(u)    X                      X
 + Varn̂(l) (u)  (l)              Puv ∆µ(l)
                                           v +        Puv µ̄(l)
                                                             v
                                                                ,
                  D         (l)
                          v∈n̂    (u)                      v∈n(u)

          (l)       (l)     (l)          (l)     (l)
where ∆h̊v = (hv − µv ) − (h̄v − µ̄v ), and the last equality is by Proposition 3.
The VNS term is the same with CVD’s VNS term.
                                         ¯ (l)     (l)                    (l)
   To analyze the VD, we further assume h̊v and h̊v are i.i.d., so EM ∆h̊v = 0,




                                                       5
           (l)                      (l)                  (l)          (l)                 (l)
EM (∆h̊v )2 = 2EM (h̊v )2 , and EM h̊v ∆h̊v = EM (h̊v )2 .
                                                       
                  n(u) X                  X       ¯
 Varn̂(l) (u),M  (l)         Puv ∆h̊(l)
                                     v +      Puv h̊(l)
                                                    v
                                                        
                  D    (l)         v∈n̂    (u)                      v∈n(u)
                 n  n(u) 2              X                                                 X
                                                                        (l)         (l)                      ¯ (l) ¯ (l)
=En̂(l) (u),M                                         Pui Puj ∆h̊i ∆h̊j +                            Pui Puj h̊i h̊j
                         D(l)
                                      i,j∈n̂(l) (u)                                       i,j∈n(u)

         n(u)               X                                       o
                                                     (l) ¯ (l)
    +2                                    Pui Puj ∆h̊i h̊j
         D(l)
                 i∈n̂(l) (u),j∈n(u)
              2      X                                                      X                      X
        n(u)                                              (l)                                ¯ (l)                (l) ¯ (l)
=                                 En̂(l) (u),M (Pui ∆h̊i )2 +                          2
                                                                                      Pui EM h̊i + 2   Pui Puj ∆h̊i h̊j
        D(l)
                    i∈n̂(l) (u)                                          i∈n(u)                           ij∈n(u)

 n(u) X                 (l)
                                X            ¯ (l)    X            ¯ (l)
= (l)        Pui2 EM (h̊i )2 +         2
                                      Pui EM h̊i + 2         2
                                                            Pui EM h̊i
 D
      i∈n(u)                   i∈n(u)                i∈n(u)
            
       n(u)
= 3 + (l) Su(l) .
       D

B        Proof of Theorem 1
Theorem 1. For a constant sequence of Wi = W and any i > LI (i.e., after L
                                                          (l)
epochs), the activations computed by CV are exact, i.e., ZCV,i = Z (l) for each
                      (l)
l ∈ [L] and HCV,i = H (l) for each l ∈ [L − 1].
                                                                                                                    (0)
Proof. We prove by induction. After the first epoch the activation hi,v is at
                                                                              (0)           (0)
least computed once for each node v. So H̄CV,i = HCV,i = H (0) = X for all
                                                 (l)            (l)
i > I. Assume that we have H̄CV,i = HCV,i = H (l) for all i > (l + 1)I. Then for
all i > (l + 1)I
                                       
  (l+1)       (l) (l)    (l)        (l)               (l)
ZCV,i = P̂i (HCV,i − H̄CV,i ) + P H̄CV,i W (l) = P H̄CV,i W (l) = P H (l) W (l) = Z (l+1) .
                                                                                                                           (1)
 (l+1)     (l+1)
HCV,i = σ(ZCV,i ) = H (l+1)

                                                                        (l+1)
After one more epoch, all the activations hCV,i,v are computed at least once for
                    (l+1)           (l+1)
each v, so H̄CV,i = HCV,i = H (l+1) for all i > (l + 2)I. By induction, we know
                                                 (L−1)              (L−1)
that after LI steps, we have H̄CV,i = HCV,i = H (L−1) . By Eq. 1 we also have
  (L)
Z̄CV,i = Z (L) .




                                                                6
C        Proof of Theorem 2
We proof Theorem 2 in 3 steps:
    1. Lemma 1: For a sequence of weights W (1) , . . . , W (N ) which are close to
       each other, CV’s approximate activations are close to the exact activations.
    2. Lemma 2: For a sequence of weights W (1) , . . . , W (N ) which are close to
       each other, CV’s gradients are close to be unbiased.

    3. Theorem 2: An SGD algorithm generates the weights that changes slow
       enough for the gradient bias goes to zero, so the algorithm converges.
    The following proposition is needed in our proof
Proposition B. Let kAk∞ = maxij |Aij |, then

    • kABk∞ ≤ col(A) kAk∞ kBk∞ , where col(A) is the number of columns of
      the matrix A.
    • kA ◦ Bk∞ ≤ kAk∞ kBk∞ , where ◦ is the element wise product.
    • kA + Bk∞ ≤ kAk∞ + kBk∞ .

Proof.

                     X                           X
    kABk∞ = max           Aik Bik ≤ max                  kAk∞ kBk∞ = col(A) kAk∞ kBk∞ .
                ij                          ij
                      k                              k
kA ◦ Bk∞ = max |Aij Bij | ≤ max kAk∞ kBk∞ = kAk∞ kBk∞ .
                ij                ij

kA + Bk∞ = max |Aij + Bij | ≤ max {|Aij | + |Bij |} ≤ max |Aij | + max |Bij | = kAk∞ + kBk∞ .
                ij                     ij                          ij        ij




   We define C := max{col(P ), col(H (0) ), . . . , col(H (L) )} to be the maximum
number of columns we can possibly encounter in the proof.

C.1      Single layer GCN
The following proposition states that if the inputs and the weights of an one-layer
GCN with CV estimator does not change too much, then its output does not
change too much, and is close to the output of an exact one-layer GCN.
Proposition C. If the activation σ(·) is ρ-Lipschitz, for any series of T inputs,
weights, and stochastic propagation matrices (Xi , XCV,i , Wi , P̂i )Ti=1 , s.t.,

    1. all the matrices are bound by B, i.e., kXCV,i k∞ ≤ B, kXi k∞ ≤ B,
       kWi k∞ ≤ B and P̂i         ≤ B,
                              ∞




                                                 7
   2. the differences are bound by , i.e., kXCV,i − XCV,j k∞ < , kXCV,i − Xi k∞ <
       and kWi − Wj k∞ < ,

let P = EP̂i . If at time i we feed (XCV,i , Wi , P̂i ) to an one-layer GCN with CV
estimator to evaluate the prediction for nodes in the minibatch Vi , 1
                                                  
       ZCV,i = P̂i (XCV,i − X̄CV,i ) + P X̄CV,i Wi , HCV,i = σ(ZCV,i ).

where X̄CV,i is the maintained history at time i, and (Xi , Wi , P ) to an one-layer
GCN with exact estimator

                              Zi = P Xi Wi ,       Hi = σ(Zi ),

then there exists K that depends on C, B and ρ, s.t. for all I < i, j ≤ T , where
I is the number of iterations per epoch:

   1. The outputs does not change too fast: kZCV,i − ZCV,j k∞ < K and
      kHCV,i − HCV,j k∞ < K,
   2. The outputs are close to the exact output: kZCV,i − Zi k∞ < K and
      kHCV,i − Hi k∞ < K.

Proof. Because for all i > I (i.e., after one epoch), the elements of X̄CV,i are all
taken from previous iterations, i.e., XCV,1 , . . . , XCV,i−1 , we know that

            X̄CV,i − XCV,i ∞ ≤ max kXCV,j − XCV,i k∞ ≤                 (∀i > I).         (2)
                                     j≤i

By triangular inequality, we also know

                          X̄CV,i − X̄CV,j ∞ < 3 (∀i, j > I).                             (3)
                              X̄CV,i − Xi ∞ < 2 (∀i > I).                                (4)

Since kXCV,1 k∞ , . . . , kXCV,T k∞ are bounded by B, X̄CV,i ∞ is also bounded
   1 Conceptually we feed the data for all the nodes in V, but since we only require the

predictions for the nodes in Vi , the algorithm will only fetch the input of a subset of nodes
⊂ V, and update history for those nodes.




                                               8
by B for i > I. Then,
 kZCV,i − ZCV,j k∞
                                                                     
= P̂i (XCV,i − X̄CV,i ) + P X̄CV,i Wi − P̂j (XCV,j − X̄CV,j ) + P X̄CV,j Wj
                                                                                         ∞

≤ P̂i (XCV,i − X̄CV,i )Wi − P̂j (XCV,j − X̄CV,j )Wj               + ρ P X̄CV,i Wi − P X̄CV,j Wj ∞
                                                              ∞

≤C 2 [ P̂i − P̂j        XCV,i − X̄CV,i ∞ kWi k∞
                    ∞

  + P̂j        XCV,i − X̄CV,i − XCV,j + X̄CV,j ∞ kWi k∞
           ∞

  + P̂j        XCV,j − X̄CV,j ∞ kWi − Wj k∞
           ∞
  + kP k∞ X̄CV,i − X̄CV,j ∞ kWi k∞
  + kP k∞ X̄CV,j ∞ kWi − Wj k∞ ]
≤C 2 [ P̂i − P̂j       kWi k∞ + 2 P̂j         kWi k∞ + P̂j        kWi − Wj k∞
                    ∞                      ∞                  ∞

+3 P̂j    kWi k∞ + P̂j      X̄CV,j ∞ ]
       ∞                 ∞
                                     
≤C 2 2B 2 + 2B 2 + 2B 2 + 3B 2 + B 2
=K1 ,
where K1 = 10C 2 B 2 , and
                                                            
    kZCV,i − Zi k∞ ≤ P̂i (XCV,i − X̄CV,i ) + P (X̄CV,i − Xi )                   kWi k∞
                                                                            ∞

                         ≤ C( P̂i         + 2 kP k∞ ) kWi k∞
                                     ∞
                         ≤ 3CB 2 
                         = K2 ,
where K2 = 3CB 2 . By Lipschitz continuity
                              kHCV,i − HCV,j k∞ ≤ ρK1 ,
                                   kHCV,i − Hi k∞ ≤ ρK2 .


We just let K = max{ρK1 , ρK2 , K1 , K2 }.

C.2      Lemma 1: Activation of Multi-layer GCN
The following lemma bounds the approximation error of activations in a multi-
layer GCN with CV. Intuitively, there is a sequence of slow-changing model
parameters (Wi ), where Wi is the model at the i-th iteration. At each iteration i
we use GCN with CV and GCN with Exact estimator to compute the activations
for the minibatch Vi , and update the corresponding history. Then after L epochs,
the error of the predictions by the CV estimator is bounded by the rate of change
of (Wi ), regardless of the stochastic propagation matrix P̂i .


                                                9
Lemma 1. Assume all the activations are ρ-Lipschitz, given a fixed dataset
X and a sequence of T model weights and stochastic propagation matrices
(Wi , P̂i )Ti=1 , s.t.,

  1. kWi k∞ ≤ B and P̂i                    ≤ B,
                                       ∞

  2. kWi − Wj k∞ < , ∀i, j,

let P = EP̂i . If at time i we feed (X, Wi , P̂i ) to a GCN with CV estimator to
evaluate the prediction for nodes in the minibatch Vi ,
                                                 
      (l+1)       (l)   (l)     (l)         (l)       (l)   (l+1)     (l+1)
     ZCV,i = P̂i (HCV,i − H̄CV,i ) + P H̄CV,i Wi , HCV,i = σ(ZCV,i ).

        (l)
where H̄CV,i is the maintained history at time i, and (X, Wi , P ) to a GCN with
exact estimator
                            (l+1)          (l)    (l)          (l+1)        (l+1)
                       Zi           = P Hi Wi ,              Hi        = σ(Zi       ),

then there exists K that depends on C, B and ρ s.t.,
         (L)          (L)
   •   Hi      − HCV,i              < K, ∀i > LI, l = 1, . . . , L − 1,
                              ∞

         (L)      (L)
   •   Zi      − ZCV,i            < K, ∀i > LI, l = 1, . . . , L.
                             ∞

                                                                                         (1)         (1)
Proof. By Proposition C.1, we know there exists K (1) , s.t., Hi                               − HCV,i <
                (1)            (1)
K (1)  and HCV,i − HCV,j < K (1) , ∀i > I.
                                                                                                     (L)      (L)
   Repeat this for L−1 times, we know there exist K (1) , . . . , K (L) , s.t., Hi                         − HCV,i <
        (L)           (K)                   (L)          (L)                      (L)          (K)
K, HCV,i − HCV,j < K, Zi                        − ZCV,i < K and ZCV,i − ZCV,j < K,
                  QL
∀i > LI, where K = l=1 K (l) .


C.3     Lemma 2: Gradient of Multi-layer GCN
We reiterate some notations defined in Sec. 4 of the main text. Vi is the minibatch
of nodes at iteration i that we would like to evaluate the predictions and gradients
                             (L)
on. gCV,v (Wi ) := ∇f (yv , zCV,i,v ) is the stochastic gradient propagated through
                                                               P              (L)
the node v by the CV estimator, and gCV,i (Wi ) := |V1i | v∈Vi ∇f (yv , zCV,i,v )
                                                                            (L)
is the minibatch gradient by CV. gv (Wi ) := ∇f (yv , zv ) is stochastic gradi-
ent propagated through the node v by the Exact estimator, and gi (Wi ) :=
  1
      P                (L)
|Vi |   v∈Vi ∇f (yv , zv ) is the minibatch gradient by the Exact estimator. Fi-
                         P              (L)
nally, ∇L(Wi ) = |V1L | v∈VL ∇f (yv , zi,v ) is the exact full-batch gradient.
      The following lemma bounds the bias of the gradients by the CV estimator.
Intuitively, there is a sequence of slow-changing model parameters (Wi ), where


                                                        10
Wi is the model at the i-th iteration. At each iteration i we use GCN with CV
and GCN with Exact estimator to compute the activations for the minibatch Vi ,
and update the corresponding history. After L epochs, we compute the gradient
by backpropagating through CV’s predictions on the minibatch of nodes Vi .
The gradient gCV,i (Wi ) is a random variable of both the stochastic propagation
matrix P̂i and the minibatch Vi . But the expectation of the gradient w.r.t. P̂i
and Vi , EP̂i ,Vi gCV,i (Wi ), is close to the full-batch gradient by the Exact estimator
∇L(Wi ), i.e., the gradient is close to be unbiased.
    To study the gradient, we need the backpropagation rules of the networks. Let
               (L)                           (L)
fv = f (yv , zv ) and fCV,i,v = f (yz , zCV,i,v ), we first derive the backpropagation
rule for the exact algorithm. Differentiating both sides of Eq. (1), we have:

               ∇H (l) fv = P > ∇Z (l+1) fv W (l)>                       l = 1, . . . , L − 1
               ∇Z (l) fv = σ 0 (Z (l) ) ◦ ∇H (l) fv                     l = 1, . . . , L − 1
                                   (l) >
               ∇W (l) fv = (P H      ) ∇Z (l+1) fv                     l = 0, . . . , L − 1.         (5)

Similarly, differentiating both sides of Eq. (5), we have

        ∇H (l) fCV,v = P̂ (l) ∇Z (l+1) fCV,v W (l)>                           l = 1, . . . , L − 1
               CV                        CV
                                   (l)
         ∇Z (l) fCV,v = σ 0 (ZCV ) ◦ ∇H (l) fCV,v                             l = 1, . . . , L − 1
               CV                                CV
                                         (l)
         ∇W (l) fCV,v = (P̂ (l) HCV )> ∇Z (l+1) fCV,v                      l = 0, . . . , L − 1.     (6)
                                                   CV


Lemma 2. Assume σ(·) and ∇z f (y, z) are ρ-Lipschitz, k∇z f (y, z)k∞ ≤ B.
Given a fixed dataset X and a sequence of T weights and stochastic propagation
matrices (Wi , P̂i )Ti=1 , s.t.,

   1. kWi k∞ ≤ B, P̂i              ≤ B, and kσ 0 (ZCV,i )k∞ ≤ B,
                               ∞

   2. kWi − Wj k∞ < , ∀i, j,

let P = EP̂i . If at time i we feed (X, Wi , P̂i ) to a GCN with CV estimator to
evaluate the prediction for nodes in the minibatch Vi ,
                                                 
      (l+1)       (l)   (l)     (l)         (l)       (l)   (l+1)     (l+1)
     ZCV,i = P̂i (HCV,i − H̄CV,i ) + P H̄CV,i Wi , HCV,i = σ(ZCV,i ).

         (l)
where H̄CV,i is the maintained history at time i, and (X, Wi , P ) to a GCN with
exact estimator
                         (l+1)             (l)   (l)         (l+1)         (l+1)
                       Zi        = P Hi Wi ,                Hi       = σ(Zi        ),

then there exists K that depends on C, B and ρ s.t.,

                         EP̂i ,Vi (Wi ) − ∇L(Wi )                ≤ K, ∀i > LI.
                                                             ∞




                                                       11
Proof. By Lipschitz continuity of ∇z f (y, z) and Lemma 1, there exists K̇, for
             (0)           (L−1)
all P̂i = (P̂i , . . . , P̂i     )

                                                                     ∇z(L) fCV,v − ∇z(L) fv
                                                                                                     v
                                                                          CV,v                                 ∞
                                                                          (L)
                                                                ≤ρ       zCV,v − zv(L)
                                                                                                ∞
                                                                ≤ρK̇,
                            (l)
                      σ 0 (ZCV ) − σ 0 (Z (l) )                 ≤ρK̇                                                         (7)
                                                       ∞

We prove by induction that there exists Kl , s.t., ∀l ∈ [L],,
                                    (l)
             EP̂ (≥l) ∇zCV,v fCV,v − ∇zv(l) fv                            ≤ Kl ,          ∀P̂ (0) , . . . , P̂ (l−1) ,       (8)
                                                                     ∞

where P̂ (≥l) = (P̂ (l) , . . . , P̂ (L−1) ). By Eq. (7) the statement holds for l = L,
where KL = ρK̇. If the statement holds for l + 1, i.e.,
                                     (l+1)
           EP̂ (≥l+1) ∇zCV,v fCV,v − ∇zv(l+1) fv                                  ≤ Kl ,        ∀P̂ (0) , . . . , P̂ (l) ,
                                                                           ∞

then by Eq. (5, 6),
                  (l)
     EP̂ (≥l) ∇zCV,v fCV,v − ∇zv(l) fv
                                       ∞
                  (l)
=    EP̂ (≥l) σ (ZCV ) ◦ P̂ ∇Z (l+1) fCV,v − σ 0 (Z (l) ) ◦ P > ∇Z (l+1) fv
              0            (l)
                               CV                                                                             ∞
             nh                                                      i                                             o
                      (l)
                      0                                                       0                  >
= EP̂ (≥l)        σ (ZCV ) ◦ P̂ (l) ∇Z (l+1) fCV,v                       − σ (Z     (l)
                                                                                          ) ◦ P ∇Z (l+1) fv
                                      CV                                                                                ∞
             nh                                                                            io
                    (l)
≤ EP̂ (≥l)    σ 0 (ZCV ) − σ 0 (Z (l) ) ◦ P̂ (l) ∇Z (l+1) fCV,v
                                                      CV            ∞
                      h                                                  i
                          0   (l)      (l)
 + EP̂ (l) EP̂ (≥l+1) σ (Z ) ◦ P̂            ∇Z (l+1) fCV,v − ∇Z (l+1) fv
                                                 CV                          ∞
             nh                                           io
                  0   (l)         (l)      >
 + EP̂ (≥l) σ (Z ) ◦ P̂ − P                     ∇Z (l+1) fv
                                                               ∞
         h                                                       i
            0   (l)         0   (l)          (l)
≤EP̂ (≥l) σ (ZCV ) − σ (Z )               P̂ ∇Z (l+1) fCV,v
                                                   ∞                     CV                 ∞
                  0       (l)                             (l)
    + EP̂ (l) σ (Z              )         EP̂ (≥l+1) P̂              EP̂ (≥l+1) ∇Z (l+1) fCV,v − ∇Z (l+1) fv
                                     ∞                           ∞                          CV                                ∞
    +0
≤ρK̇B 2 C 2 + B 2 C 2 Kl+1 
=Kl ,

where Kl = B 2 C 2 (ρK̇ + Kl+1 ). By induction, Eq. (8) holds. Similarly, we can
show that there exists K, s.t.,

                       EP̂ ∇W (l) fCV,v − ∇W (l) fv ∞ < K,                                 ∀l ∈ [L − 1].


                                                                 12
Therefore,

                       EVi ,P̂i gCV,i (Wi ) − ∇L(Wi )
                                                        ∞

                    = Ev∈V,P̂i gCV,v (Wi ) − Ev∈V gv (Wi )
                                                             ∞

                    ≤EVi EP̂i gCV,v (Wi ) − gv (Wi )
                                                        ∞
                    ≤EVi max EP̂ ∇W (l) fCV,v − ∇W (l) fv ∞
                           l
                    ≤K.




C.4     Proof of Theorem 2
Theorem 2. Assume that (1) the activation σ(·) is ρ-Lipschitz, (2) the gradient
of the cost function ∇z f (y, z) is ρ-Lipschitz and bounded, (3) kgCV,V (W )k∞ ,
kg(W )k∞ , and k∇L(W )k∞ are all bounded by G > 0 for all P̂ , V and W . (4)
The loss L(W ) is ρ-smooth, i.e., |L(W2 ) − L(W1 ) − h∇L(W1 ), W2 − W1 i| ≤
ρ             2                                >
2 kW2 − W1 kF ∀W1 , W2 , where hA, Bi = tr(A B) is the inner product of matrix
A and matrix B. Then, there exists K > 0, s.t., ∀N > LI, if we run SGD for
R ≤ N iterations, where R is chosen uniformly from [N ]+ , we have

                               2     L(W1 ) − L(W ∗ ) + K + ρK
               ER k∇L(WR )kF ≤ 2                √              ,
                                                  N

for the updates Wi+1 = Wi − γgCV,i (Wi ) and the step size γ = min{ ρ1 , √1N }.

Proof. This proof is a modification of [2], but using biased stochastic gradients
instead. We assume the algorithm is already warmed-up for LI steps with
the initial weights W0 , so that Lemma 2 holds for step i > 0. Denote δi =
gCV,i (Wi ) − ∇L(Wi ). By smoothness we have
                                                ρ                2
L(Wi+1 ) ≤ L(Wi ) + h∇L(Wi ), Wi+1 − Wi i + γ 2 kgCV,i (Wi )kF
                                                2
                                                ρ                2
         = L(Wi ) − γh∇L(Wi ), gCV,i (Wi )i + γ 2 kgCV,i (Wi )kF
                                                2
                                                    2    ρ h       2             2
                                                                                             i
         = L(Wi ) − γh∇L(Wi ), δi i − γ k∇L(Wi )k + γ 2 kδi k + k∇L(Wi )kF + 2hδi , ∇L(Wi )i
                                                         2
                                                         2
                                                      ργ              2  ρ         2
         = L(Wi ) − (γ − ργ 2 )h∇L(Wi ), δi i − (γ −       ) k∇L(Wi )kF + γ 2 kδi kF .
                                                       2                 2
                                                                               (9)




                                        13
   For each i, consider the sequence of LI + 1 weights Wi−LI , . . . , Wi .
                                                        i−1
                                                        X
                         max      kWj − Wk k∞ ≤                kWj − Wj+1 k∞
                    i−LI≤j,k≤i
                                                      j=i−LI
                     i−1
                     X                                i−1
                                                      X
               =               γ kgCV (Wj )k∞ ≤               γG = LIGγ.
                    j=i−LI                           j=i−LI

By Lemma 2, there exists K̇ > 0, s.t.

       EP̂ ,VB δi        = EP̂ ,VB gCV (Wi ) − ∇L(Wi )             ≤ K̇LIGγ,      ∀i > 0.
                    ∞                                          ∞

Assume that W is D-dimensional,

    EP̂ ,VB h∇L(Wi ), δi i ≤ D k∇L(Wi )k∞ EP̂ ,VB δi                   ≤ K̇LIDG2 γ ≤ Kγ,
                                                                   ∞
                           2
            EP̂ ,VB kδi kF ≤ D kgCV,i (Wi )k∞ + D k∇L(Wi )k∞ ≤ 2DG2 ≤ K,

where K = max{K̇LIDG2 , 2DG2 }. Taking EP̂ ,VB to both sides of Eq. 9 we
have
                                            ργ 2            2
    L(Wi+1 ) ≤ L(Wi ) + (γ − ργ 2 )Kγ − (γ −     ) k∇L(Wi )kF + ρKγ 2 /2.
                                             2
Summing up the above inequalities and re-arranging the terms, we obtain,
                    ργ 2 X              2
               (γ −     )    k∇L(Wi )kF
                     2    i
                                                                       ρK
                    ≤L(W1 ) − L(W ∗ ) + KN (γ − ργ 2 )γ +                 N γ2.
                                                                        2
                                        2
Dividing both sides by N (γ − ργ2 ), and take γ = min{ ρ1 , √1N }
                                            2
                     ER∼PR k∇L(WR )kF
                         L(W1 ) − L(W ∗ ) + KN (γ − ργ 2 )γ + ρK
                                                               2 Nγ
                                                                    2
                    ≤2
                                         N γ(2 − ργ)
                       L(W1 ) − L(W ∗ ) + KN (γ − ργ 2 )γ + ρK
                                                             2 Nγ
                                                                  2
                    ≤2
                                           Nγ
                       L(W1 ) − L(W ∗ )
                    ≤2                  + Kγ(1 − ργ) + ρKγ
                             Nγ
                       L(W1 ) − L(W ∗ )            √
                    ≤2       √          + Kγ + ρK/ N
                               N
                       L(W1 ) − L(W ∗ ) + K + ρK
                    ≤2            √              .
                                    N
                                                                         2
Particularly, when N → ∞, we have ER∼PR k∇L(WR )kF = 0, which implies
that the gradient is asymptotically unbiased.



                                                14
Algorithm 1 Constructing the receptive fields and random propagation matrices.

  r(L) ← VB
  for layer l ← L − 1 to 0 do
      r(l) ← ∅
      P̂ (l) ← 0
      for each node u ∈ r(l+1) do
           r(l) ← r(l) ∪ {u}
             (l)      (l)
           P̂uu ← P̂uu + Puu n(u)/D(l)
                  (l)
           for D − 1 random neighbors v ∈ n(u) do
               r(l) ← r(l) ∪ {v}
                 (l)      (l)
               P̂uv ← P̂uv + Puv n(u)/D(l)
           end for
      end for
  end for


D      Pseudocode
As mentioned in Sec. 3.3, an iteration of our algorithm consists the following
operations:
   1. Randomly select a minibatch VB ∈ VL of nodes;
                                                                        (l)      (l)
   2. Build a computation graph that only contains the activations hv and h̄v
      needed for the current minibatch;

   3. Get the predictions by forward propagation as Eq. (6) in the main text;
   4. Get the gradients by backward propagation, and update the parameters
      by SGD;
   5. Update the historical activations.

For step 2, we construct the receptive fields r(l) and stochastic propagation
matrices P̂ (l) as Alg. 1.

D.1     Training with the CV estimator
Alg. 2 depicts the training algorithm using the CV estimator. We perform
forward propagation according to Eq. (6), compute the stochastic gradient,
and then update the historical activations H̄ (l) for all the nodes in r(l) . Let
W = (W (0) , . . . , W (L−1) ) be all the trainable parameters, the gradient ∇W L is
computed automatically by frameworks such as TensorFlow.




                                        15
Algorithm 2 Training with the CV algorithm
  for each minibatch VB ⊂ V do
     Compute the receptive fields r(l) and stochastic propagation matrices P̂ (l)
  as Alg. 1.
     (Forward propgation)
     for each layer l ← 0 to L − 1 do              
         Z (l+1) ← P̂ (l) (H (l) − H̄ (l) + P H̄ (l) W (l)
        H (l+1) ← σ(Z (l+1) )
     end for                     P             (L)
     Compute the loss L = |V1B | v∈VB f (yv , Zv )
     (Backward propagation)
     W ← W − γi ∇ W L
     (Update historical activations)
     for each layer l ← 0 to L − 1 do
        for each node v ∈ r(l) do
              (l)    (l)
            h̄v ← hv
        end for
     end for
  end for


D.2     Training with the CVD estimator
Training with the CVD estimator is similar with the CV estimator, except it
runs two versions of the network, with and without dropout, to compute   the
                                                           (l)    (l) p
samples H and their mean µ of the activation. The matrix P̄uv = P̂uv / n(v) ,
where n(v) is the degree of node v.


E     Experiment setup
In this sections we describe the details of our model architectures. We use the
Adam optimizer [4] with learning rate 0.01.
    • Citeseer, Cora, PubMed and NELL: We use the same architecture as [5]:
      two graph convolution layers with one linear layer per graph convolution
      layer. We use 32 hidden units, 50% dropout rate and 5 × 10−4 L2 weight
      decay for Citeseer, Cora and PubMed and 64 hidden units, 10% dropout
      rate and 10−5 L2 weight decay for NELL.
    • PPI and Reddit: We use the mean pooling architecture GraphSAGE-
      mean proposed by [3]. We use two linear layers per graph convolution
      layer. We set weight decay as zero, dropout rate as 20%, and adopt layer
      normalization [1] after each linear layer. We use 512 hidden units for PPI
      and 128 hidden units for Reddit. We find that our architecture can reach
      97.8% testing micro-F1 on the PPI dataset, which is significantly higher


                                       16
Algorithm 3 Training with the CVD algorithm
  for each minibatch VB ⊂ V do
     Compute the receptive fields r(l) and stochastic propagation matrices P̂ (l)
  as Alg. 1.
     (Forward propgation)
     for each layer l ← 0 to L − 1 do                                  
         U ← P̄ (l) (H (l) − µ(l) ) + P̂ (l) (µ(l) − µ̄(l) ) + P H̄ (l)
        H (l+1) ← σ(Dropoutp (U )W (l) )
        µ(l+1) ← σ(U W (l) )
     end for                     P             (L)
     Compute the loss L = |V1B | v∈VB f (yv , Hv )
     (Backward propagation)
     W ← W − γi ∇ W L
     (Update historical activations)
     for each layer l ← 0 to L − 1 do
        for each node v ∈ r(l) do
              (l)    (l)
            h̄v ← hv
        end for
     end for
  end for

                Table 1: Time to reach 0.95 testing accuracy.

                       Valid.                 Time   Sparse    Dense
             Alg.               Epochs
                        acc.                   (s)   GFLOP    TFLOP
            Exact      0.940      3.0          199    306       11.7
             NS        0.940      24.0         148    33.6      9.79
           NS+PP       0.940      12.0          68    2.53      4.89
           CV+PP       0.940      5.0          32     8.06     2.04
          CVD+PP       0.940      5.0           36    16.1      4.08


     than 59.8% reported by [3]. We find the improvement is from wider hidden
     layer, dropout and layer normalization.


F     Experiment for 3-layer GCNs
We test 3-layer GCNs on the Reddit dataset. The settings are the same with
2-layer GCNs in Sec. 6.2. To ensure M1+PP can run in a reasonable amount of
time, we subsample the graph so that the maximum degree is 10. The convergence
result is shown as Fig. 1, where the conclusion is similar with the two-layer
models: CVD+PP is the best-performing approximate algorithm, followed by
CV+PP, and then NS+PP and NS. The time consumption to reach 0.94 testing
accuracy is shown in Table 1.


                                         17
                                      reddit3
                  0.96
                  0.94
                  0.92
                  0.900      10       20      30       40      50
    M1+PP           NS        NS+PP          IS+PP          CV+PP               CVD+PP
Figure 1: Comparison of validation accuracy with respect to number of epochs
for 3-layer GCNs.


G      Correlation between node activations
In our analysis of the variance for the CVD estimator in Sec. 5.2, we
                                                                   h assumei that
                                                                     (l) (l)
the activations for different nodes are uncorrelated, i.e., CovM hu , hv = 0,
for all u 6= v, where M is the dropout mask. We show the rationale behind
this assumption in this section. For 2-layer GCNs, the activations are indeed
independent, and the correlation is still weak for deeper GCNs due to the sparsity
of our sampled graph.

G.1     Results for 2-layer GCNs
For a 2-layer GCN with the first layer pre-processed, the activations of nodes
are independent. Suppose we want to compute the prediction for a node on
the second layer. Without loss of generality, assume that we want to com-
       (2)                                                               (1)
pute
   z1 , and the neighbors of node 1 are 1, . . . , D. The activation hv =
            (0)                (0)
σ (Mv ◦ uv )W (0) , where uv = (P H (0) )v is a random variable with respect
                                                                    (1)         (1)
to Mv , Mv ∼ Bernoulli(p) is the dropout mask. We show that hv            and hv0 are
independent, for v 6= v 0 by the following lemma.
Lemma 3. If a and b are independent random variables, then their transforma-
tions f1 (a) and f2 (b) are independent.
    Because for any event A and B, P (f1 (a) ∈ f1 (A), f2 (b) ∈ f2 (B)) = P (a ∈
A, b ∈ B) = P (a ∈ A)P (b ∈ B) = P (f1 (a) ∈ f1 (A))P (f2 (B) ∈ f2 (B)), where
f1 (A) = {f1 (a)|a ∈ A} andf2 (B) = {f2 (b)|b∈ B}.                                
         (1)                      (0)           (1)                         (0)
    Let hv = f1 (Mv ) := σ (Mv ◦ uv )W (0) and hv0 = f1 (Mv0 ) := σ (Mv0 ◦ uv0 )W (0) ,
                                                                          (1)         (1)
because Mv and Mv0 are independent Bernoulli random variables, hv and hv0
are independent.
    The result can be further generalized to deeper models. If the receptive fields
of two nodes does not overlap, they should be independent.


                                        18
                    citeseer                                           cora
                                            0.06
   0.075
   0.050                                    0.04
   0.025               Feature correlation 0.02                       Feature correlation
                       Neighbor correlation                           Neighbor correlation
   0.000                                    0.00
             2      4        6          8                    2     4        6          8
                      Topics                                         Topics
                    pubmed                                              ppi
   0.04                                                                 Feature correlation
                                                0.10                    Neighbor correlation
   0.02               Feature correlation 0.05
                      Neighbor correlation
             2     4        6          8                     2     4            6       8
                     Topics                                            Topics
   Figure 2: Average feature and neighbor correlations in a 10-layer GCN.


G.2        Empirical results for deeper GCNs
Because we only sample two neighbors per node, the sampled subgraph is very
close to a graph with all its nodes isolated, which reduces to the MLP case
that [6] discuss.
    We empirically study the correlation between feature dimensions and neigh-
bors. The definition of the correlation between feature dimensions is the same
with [6]. For each node v on layer l, we compute the correlation between each
                      (l)
feature dimension of hv
                               (l,v)           (l)   (l)
                         Covij         := C[hvi , hvj ]
                                                           (l,v)
                               (l,v)             Covij
                        Corrij         := q            q          ,
                                                 (l,v)      (l,v)
                                              Covii      Covjj

where i and j are the indices for different hidden dimensions, and C[X, Y ] =
E[(X − EX)(Y − EY )] is the covariance between two random variables X and
                         (l,v)                                          (l)    (l)
Y . We approximate Covij with 1,000 samples of the activations hvi and hvj ,
by running the forward propagation 1,000 times with different dropout masks.
                                                                  (l,v)
We define the average feature correlation on layer l to be Covij averaged by
the nodes v and dimension pairs i 6= j.
     To compute the correlation between neighbors, we treat each feature dimen-
sion separately. For each layer l + 1, node v, and dimension d, we compute the
                                               (l)
correlation matrix of all the activations {hid |i ∈ n̄(l) (v)} that are needed by
  (l+1)                      (l)
hvd , where n̄(l) (v) = {i|P̂vi 6= 0} is the set of subsampled neighbors for node




                                              19
v:
                          (l,v,d)               (l)   (l)
                      Covij         := C[hid , hjd ]
                                                            (l,v,d)
                          (l,v,d)           Covij
                     Corrij         := q           q           ,
                                           (l,v,d)     (l,v,d)
                                        Covii       Covjj

where the indices i, j ∈ n̄(l) (v). Then, we compute the average correlation of all
pairs of neighbors i 6= j.
                                                   1          X        (l,v,d)
            AvgCorr(l,v,d) :=                       (l)
                                                                   Corrij      ,
                                    n̄(l) (v)   ( n̄ (v) − 1) i6=j

and define the average neighbor correlation on layer l as AvgCorr(l,v,d) averaged
over all the nodes v and dimensions d.
    We report the average feature correlation and the average neighbor correlation
per layer, on the Citeseer, Cora, PubMed and PPI datasets. These quantities
are too expensive to compute for NELL and Reddit. On each dataset, we train a
GCN with 10 graph convoluation layers until early stopping criteria is met, and
compute the average feature correlation and the average neighbor correlation for
layer 1 to 9. We are not interested in the correlation on layer 10 because there
are no more graph convolutional layers after it. The result is shown as Fig. 2.
As analyzed in Sec. G.1, the average neighbor correlation is close to zero on the
first layer, but it is not exactly zero due to the finite sample size for computing
the empirical covariance. There is no strong tendency of increased correlation
as the number of layers increases, after the third layer. The average neighbor
correlation and the average feature correlation remain on the same order of
magnitude, so bringing correlated neighbors does not make the activations much
more correlated than the MLP case [6]. Finally, both correlations are much
smaller than one.


References
[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization.
    arXiv preprint arXiv:1607.06450, 2016.
[2] Saeed Ghadimi and Guanghui Lan. Stochastic first-and zeroth-order methods
    for nonconvex stochastic programming. SIAM Journal on Optimization,
    23(4):2341–2368, 2013.
[3] William L Hamilton, Rex Ying, and Jure Leskovec. Inductive representation
    learning on large graphs. arXiv preprint arXiv:1706.02216, 2017.
[4] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimiza-
    tion. arXiv preprint arXiv:1412.6980, 2014.



                                                20
[5] Thomas N Kipf and Max Welling. Semi-supervised classification with graph
    convolutional networks. In ICLR, 2017.
[6] Sida Wang and Christopher Manning. Fast dropout training. In Proceedings
    of the 30th International Conference on Machine Learning (ICML-13), pages
    118–126, 2013.




                                     21

