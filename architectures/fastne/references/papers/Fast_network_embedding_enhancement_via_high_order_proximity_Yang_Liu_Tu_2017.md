# Fast network embedding enhancement via high order proximity Yang Liu Tu 2017

> Source: `Fast_network_embedding_enhancement_via_high_order_proximity_Yang_Liu_Tu_2017.pdf`

---

Fast Network Embedding Enhancement via High Order Proximity Approximation

                        Cheng Yang1 , Maosong Sun2 ∗ , Zhiyuan Liu2 , Cunchao Tu1 ,
         Department of Computer Science and Technology, Tsinghua University, Beijing 100084, China
                1
                  {cheng-ya14,tcc13}@mails.tsinghua.edu.cn, 2 {sms,liuzy}@tsinghua.edu.cn



                             Abstract                            improve the flexibility of features. In terms of network anal-
                                                                 ysis, Network Representation Learning (NRL) which aim-
        Many Network Representation Learning (NRL)               s to learn distributed real-valued embeddings for vertices of
        methods have been proposed to learn vector repre-        a network has attracted much attention in recent years [Per-
        sentations for vertices in a network recently. In this   ozzi et al., 2014; Tang et al., 2015b; Cao et al., 2015;
        paper, we summarize most existing NRL methods            Grover and Leskovec, 2016].
        into a unified two-step framework, including prox-          In this paper, we summarize most existing NRL meth-
        imity matrix construction and dimension reduction.       ods into a unified two-step framework, including proximity
        We focus on the analysis of proximity matrix con-        matrix construction and dimension reduction. The first step
        struction step and conclude that an NRL method           builds a proximity matrix M where each entry Mij encodes
        can be improved by exploring higher order proxim-        the proximity information between vertex i and j. The sec-
        ities when building the proximity matrix. We pro-        ond step reduces the dimension of the proximity matrix to
        pose Network Embedding Update (NEU) algorithm            obtain network embeddings. Different NRL methods employ
        which implicitly approximates higher order prox-         various dimension reduction algorithms such as eigenvector
        imities with theoretical approximation bound and         computation and SVD decomposition. Our analysis of the
        can be applied on any NRL methods to enhance             first step, i.e. proximity matrix construction, shows that the
        their performances. We conduct experiments on            quality of network embeddings can be improved when higher
        multi-label classification and link prediction tasks.    order proximities are encoded into the proximity matrix.
        Experimental results show that NEU can make a               However, an accurate computation of high-order proxim-
        consistent and significant improvement over a num-       ities is time-consuming and thus not scalable for large-scale
        ber of NRL methods with almost negligible run-           networks. Thus we can only approximate higher order prox-
        ning time on all three publicly available datasets.      imity matrix for learning better network embeddings. In order
        The source code of this paper can be obtained from       to be more efficient, we also seek to use the network repre-
        https://github.com/thunlp/NEU.                           sentations which encode the information of lower order prox-
                                                                 imities as our basis to avoid repeated computations. There-
                                                                 fore, we propose Network Embedding Update (NEU) algo-
1       Introduction                                             rithm which could be applied to any NRL methods to enhance
   A network is an essential data type widely used in our        their performances. The intuition behind is that the embed-
daily lives and academic researches, such as friendship net-     dings processed by NEU algorithm can implicitly approxi-
works in Facebook and citation networks in DBLP [Ley,            mate higher order proximities with a theoretical approxima-
2002]. Researchers have made great effort on developing          tion bound and thus achieve better performances.
machine learning algorithms for various network application-        We conduct experiments on multi-label classification and
s, e.g. vertex classification [Sen et al., 2008], tag recom-     link prediction tasks over three publicly available datasets to
mendation [Tu et al., 2014], anomaly detection [Akoglu et        evaluate the quality of network embeddings. Experimental
al., 2015] and link prediction [Liben-Nowell and Kleinberg,      results show that the network embeddings learned by existing
2007]. Most supervised machine learning algorithms applied       NRL methods can be improved consistently and significantly
on these applications require a set of informative features as   on both evaluation tasks after enhanced by NEU. Moreover,
input [Grover and Leskovec, 2016]. Hand-crafted features         the running time of NEU takes less than 1% training time of
may suit the need but require much human effort and ex-          popular NRL methods such as DeepWalk and LINE, which
pert knowledge. Therefore, representation learning [Bengio       could be negligible.
et al., 2013] which learns feature embeddings via optimiza-         Our main contributions are two-fold:
tion learning was proposed to avoid feature engineering and         1) We summarize a number of existing NRL methods into
                                                                 a unified framework, i.e. proximity matrix construction and
    ∗
        Corresponding author: sms@tsinghua.edu.cn                dimension reduction, and conclude that the quality of network
embeddings can be enhanced if higher order proximities are       d is the embedding dimension. We assume that networks are
encoded into the proximity matrix.                               unweighted and undirected in this paper without loss of gen-
   2) We propose NEU algorithm to improve the perfor-            erality and define adjacency matrix A e ∈ R|V |×|V | as A
                                                                                                                         eij = 1
mances of any network embeddings learned by existing N-          if (vi , vj ) ∈ E and Aij = 0 otherwise. Diagonal matrix
                                                                                        e
RL algorithms. The embeddings processed by NEU can im-
                                                                 D ∈ R|V |×|V | denotes the degree matrix where Dii = di
plicitly approximate higher order proximities with theoretical
bound. Experimental results on multi-label classification and    represents the degree of vertex vi . A = D−1 A e is normalized
link prediction demonstrate the efficiency and effectiveness     adjacency matrix where the summation of each row equals to
of our algorithm.                                                1. Similarly, we also have Laplacian matrix L e = D−A     e and
                                                                                                         1      1
   Related Works Here we give a brief introduction to ex-        normalized Laplacian matrix L = D− 2 e    LD− 2 .
isting NRL methods and some of which will be thoroughly
analyzed in next section. Spectral Clustering [Tang and Li-      2.1   K-order Proximity
u, 2011] computes top-d eigenvectors of normalized Lapla-           The (normalized) adjacency matrix and Laplacian matrix
cian matrix as d-dimensional network embeddings. Deep-           characterize the first-order proximity which models the lo-
Walk [Perozzi et al., 2014] employs Skip-gram [Mikolov et        cal pairwise proximity between vertices. Note that each off-
al., 2013] model, which is originally used in word repre-        diagonal nonzero entry of the first-order proximity matrix
sentation learning, on random walks for NRL. LINE [Tang          corresponds to an edge in the network. However, real-world
et al., 2015b] models first-order and second-order proximi-      networks are always sparse which indicates that O(E) =
ties between vertices for learning large-scale network embed-    O(V ). Therefore the first-order proximity matrix is usu-
dings. GraRep [Cao et al., 2015] factorizes different k-order    ally very sparse and insufficient to fully model the pair-
proximity matrices and concatenates the embeddings learned       wise proximity between vertices. As a result, people al-
from each proximity matrix. Though GraRep achieves bet-          so explore higher order proximity to model the strength be-
ter performance than DeepWalk and LINE, GraRep suffer-           tween vertices [Perozzi et al., 2014; Tang et al., 2015b;
s heavily from inefficiency problem. Besides the above N-        Cao et al., 2015]. For example, the second-order proximity
RL methods that focus on network topology, researcher-           can be characterized by the number of common neighbors be-
s also explore algorithms to incorporate meta information,       tween vertices. As an alternative view, the second-order prox-
e.g. text and label information, into NRL. TADW [Yang            imity between vi and vj can also be modeled by the probabil-
et al., 2015] takes text information into consideration un-      ity that a 2-step random walk from vi reaches vj . Intuitively,
der matrix factorization framework and MMDW [Tu et al.,          the probability will be large if vi and vj share many common
2016] learns semi-supervised network embeddings with max-        neighbors. In the probabilistic setting based on random walk,
margin constraints between vertices from different labels. As    we can easily generalize it to k-order proximity [Cao et al.,
another semi-supervised NRL method, node2vec [Grover and         2015]: the probability that a random walk starts from vi and
Leskovec, 2016] further generalizes DeepWalk with Breadth-       walks to vj with exactly k steps. Note that the normalized
First Search (BFS) and Depth-First Search (DFS) on random        adjacency matrix A is the transition probability matrix of a
walks. GCN [Kipf and Welling, 2017], DDRW [Li et al.,            single step random walk. Then we can compute k-step tran-
2016] and Planetoid [Yang et al., 2016] are also proposed        sition probability matrix as the k-order proximity matrix
for semi-supervised graph embeddings. SDNE [Wang et al.,
2016] employs deep neural model for NRL. Other extensions                              Ak = A
                                                                                            | · A{z. . . A},                (1)
include asymmetric transitivity [Ou et al., 2016], communi-                                         k
ty preserving [Wang et al., 2017] and heterogenous [Tang et
al., 2015a; Chang et al., 2015; Huang and Mamoulis, 2017;        where the entry Akij is the k-order proximity between vertex
Xu et al., 2017; Huang et al., 2017] network embeddings. We      vi and vj .
focus on the most general case where NRL methods use only
network topology in this paper.                                  2.2   NRL Framework
                                                                    Now we have introduced the concept of k-order proxim-
                                                                 ity matrix. In this subsection, we first summarize the NRL
2   Framework of Existing NRL Algorithms                         framework based on dimension reduction of a proximity ma-
   In this section, we present a unified framework which can     trix and then conduct a theoretical analysis to show that most
cover several representative NRL algorithms include Spectral     existing NRL algorithms can be formalized into this frame-
Clustering [Tang and Liu, 2011], DeepWalk [Perozzi et al.,       work.
2014], TADW [Yang et al., 2015], LINE [Tang et al., 2015b]          We summarize NRL methods as a two-step framework:
and GraRep [Cao et al., 2015]. First, we clarify the notations      Step 1: Proximity Matrix Construction. Compute a
and formalize the problem of NRL. Then we introduce the          proximity matrix M ∈ R|V |×|V | which encodes the informa-
concept of k-order proximity. Finally, we summarize an NRL       tion of k-order proximity matrix where k = 1, 2 . . . , K. For
                                                                                  1     1 2         1 K
framework based on proximity matrix factorization and show       example, M = K     A+ K  A · · ·+ K  A stands for an average
that the aforementioned NRL methods fall into the category.      combination of k-order proximity matrix for k = 1, 2 . . . , K.
   Let G = (V, E) be a given network where V is vertex           The proximity matrix M is usually represented by a poly-
set and E is edge set. The task of NRL is to learn a real-       nomial of normalized adjacency matrix A of degree K and
valued representation rv ∈ Rd for each vertex v ∈ V where        we denote the polynomial as f (A) ∈ R|V |×|V | . Here the
                                                                                1
degree K of polynomial f (A) corresponds to the maximum             and S T Σ 2 , respectively. The computation of k-order rep-
order of proximities encoded in proximity matrix. Note that         resentation R{k} naturally follows our framework. However,
the storage and computation of proximity matrix M does not          GraRep cannot efficiently scale to large networks [Grover and
necessarily take O(|V |2 ) time because we only need to save        Leskovec, 2016]: though the first-order proximity matrix A
and compute the nonzero entries.                                    is sparse, a direct computation of Ak (k ≥ 2) takes O(|V |2 )
    Step 2: Dimension Reduction. Find network embedding             time which is unacceptable for large-scale networks.
matrix R ∈ R|V |×d and context embedding C ∈ R|V |×d                   It is straightforward that TADW [Yang et al., 2015] and
so that the product R · C T approximates proximity matrix           LINE [Tang et al., 2015b] can also be formalized into our
M . Here different algorithms may employ different distance         framework and the proofs are omitted due to space limitation.
functions to minimize the distance between M and R · C T .
For example, we can naturally use the norm of matrix M −
R · C T to measure the distance and minimize it.                    3      Observation and Problem Formalization
    Spectral Clustering [Tang and Liu, 2011] computes the              By far, we have shown five representative NRL algorithms
first d eigenvectors of normalized Laplacian matrix L as d-         can be formulated into our two-step framework, i.e. proxim-
dimensional network representations. The information em-            ity matrix construction and dimension reduction. In this sec-
bedded in the eigenvectors comes from the first-order prox-         tion, we focus on the first step and study how to define a good
imity matrix L. Note that the real-valued symmetric matrix          proximity matrix for NRL. The study of different dimension
L can be factorized as L = QΛQ−1 via Eigendecomposi-                reduction methods, e.g. SVD decomposition, will be left as
tion where Λ ∈ R|V |×|V | is a diagonal matrix, Λ11 ≥ Λ22 ≥         future work.
. . . Λ|V ||V | are eigenvalues and Q ∈ R|V |×|V | is the eigen-
vector matrix.                                                                Table 1: Comparisons among three NRL methods.
    We can equivalently reduce Spectral Clustering to our N-
RL framework by setting proximity matrix M as the first-                                     SC        DeepWalk          GraRep
                                                                                                       PK Ak
order proximity matrix L, network embedding R as the first              Proximity Matrix      L          k=1 K
                                                                                                                     k
                                                                                                                    A ,k = 1...K
d columns of eigenvector matrix Q and context embedding                   Computation      Accurate   Approximate      Accurate
C T as the first d rows of ΛQ−1 .                                          Scalability       Yes          Yes            No
    DeepWalk [Perozzi et al., 2014] generates random walk-                Performance       Low         Middle          High
s and employs Skip-gram model for representation learn-
ing. DeepWalk learns two representations for each vertex
                                                                       We summarize the comparisons among Spectral Cluster-
and we denote the network embedding and context embed-
                                                                    ing (SC), DeepWalk and GraRep in Table 1 and conclude the
ding as matrix R ∈ R|V |×d and C ∈ R|V |×d . As proved by           following observations.
[Yang et al., 2015], DeepWalk implicitly factorizes a matrix
                                                                       Observation 1 Modeling higher order and accurate prox-
M ∈ R|V |×|V | into the product of R · C T , where                  imity matrix can improve the quality of network represen-
                            A + A2 + · · · + Aw                     tation. In other words, NRL could benefit if we explore a
               M = log                           ,            (2)   polynomial proximity matrix f (A) of a higher degree.
                                     w
                                                                       From the development of NRL methods, we can see that
and w is the window size used in Skip-gram model. Matrix
                                                                    DeepWalk outperforms Spectral Clustering because Deep-
M characterizes the average of the first-order, second-order,
                                                                    Walk considers higher order proximity matrices and the high-
. . . , w-order proximities. DeepWalk algorithm approximates
                                                                    er order proximity matrices can provide complementary in-
high-order proximity by Monte Carlo sampling based on ran-
                                                                    formation for lower order proximity matrices. GraRep out-
dom walk generation without calculating the k-order proxim-
                                                                    performs DeepWalk because GraRep accurately calculates
ity matrix directly.
                                                                    the k-order proximity matrix rather than approximating it by
    To adopt DeepWalk algorithm to our two-step frame-
                                                                    Monte Carlo simulation as DeepWalk did.
work, we can simply set proximity matrix M = f (A) =
A+A2 +···+Aw
                                                                       Observation 2 Accurate computation of high order prox-
         w       . Note that we omit the log operation in Eq. (2)   imity matrix is not feasible for large-scale networks.
because they yield competitive performance as reported by              The major drawback of GraRep is the computation com-
[Yang et al., 2015].
                                                                    plexity of calculating the accurate k-order proximity matrix.
    GraRep [Cao et al., 2015] accurately calculates k-order         In fact, the computation of high order proximity matrix takes
proximity matrix Ak for k = 1, 2 . . . , K, computes a specif-      O(|V |2 ) time and the time complexity of SVD decomposition
ic representation for each k, and concatenates these embed-         also increases as k-order proximity matrix gets dense when k
dings. Specifically, GraRep reduces the dimension of k-order        grows. In summary, a time complexity of O(|V |2 ) is too ex-
proximity matrix Ak for k-order representation via SVD de-          pensive to handle large-scale networks.
composition. In detail, we assume that k-order proximity ma-
                                                                       The first observation motivates us to explore higher order
trix Ak is factorized into the product of U ΣS where Σ ∈
                                                                    proximity matrix in NRL algorithm but the second obser-
R|V |×|V | is a diagonal matrix, Σ11 ≥ Σ22 ≥ . . . Σ|V ||V | ≥ 0    vation prevents us from an accurate inference of higher or-
are singular values and U, S ∈ R|V |×|V | are unitary matrices.     der proximity matrices. Therefore we turn out to study the
GraRep defines k-order network embedding and context em-            problem that how to learn network embeddings from approx-
                                                                1
bedding R{k} , C{k} ∈ R|V |×d as the first d columns of U Σ 2       imate higher order proximity matrices efficiently. In order
to be more efficient, we aim to use the network representa-           In our experimental settings, we assume that the weight
tions which encode the information of lower order proximity        of lower order proximities should be larger than higher order
matrices as our basis to avoid repeated computations. We for-      proximities because they are more directly related to the orig-
malize our problem below.                                          inal network. Therefore, given g(A) = f (A) + 2λAf (A) +
   Problem Formalization Assume that we have normalized            λ2 A2 f (A), we have 1 ≥ 2λ ≥ λ2 > 0 which indicates
adjacency matrix A as the first-order proximity matrix, net-       that λ ∈ (0, 21 ]. The proof indicates that the updated em-
work embedding R and context embedding C where R, C ∈              bedding can implicitly approximate a polynomial g(A) of 2
R|V |×d . Suppose that the embeddings R and C are learned by       more degrees within 49 times matrix infinite norm of previous
the above NRL framework which indicates that the product           embeddings. QED.
R · C T approximates a polynomial proximity matrix f (A) of           Algorithm The update Eq. (3) can be further generalized
degree K. Our goal is to learn a better representation R0 and      in two directions. First we can update embeddings R and C
C 0 which approximates a polynomial proximity matrix g(A)          according to Eq. (5):
with higher degree than f (A). Also, the algorithm should be
efficient in the linear time of |V |. Note that the lower bound                R0 = R + λ1 A · R + λ2 A · (A · R),
                                                                                                                              (5)
of time complexity is O(|V |d) which is the size of embedding                  C 0 = C + λ1 AT · C + λ2 AT · (AT · C).
matrix R.
                                                                   The time complexity is still O(|V |d) but Eq. (5) can obtain
                                                                   higher proximity matrix approximation than Eq. (3) in one
4   Approximation Algorithm                                        iteration. More complex update formulas which explores fur-
   In this section, we present a simple, efficient and effective   ther higher proximities than Eq. (5) can also be applied but
iterative updating algorithm to solve the above problem.           we use Eq. (5) in our experiments as a cost-effective choice.
   Method Given hyperparameter λ ∈ (0, 21 ], normalized ad-           Another direction is that the update equation can be pro-
jacency matrix A, we update network embedding R and con-           cessed for T rounds to obtain higher proximity approxima-
text embedding C as follows:                                       tion. However, the approximation bound would grow ex-
                                                                   ponentially as the number of rounds T grows and thus the
                     R0 = R + λA · R,                              update cannot be done infinitely. Note that the update op-
                                                            (3)
                     C 0 = C + λAT · C.                            eration of R and C are completely independent. Therefore
                                                                   we only need to update network embedding R for NRL. We
The time complexity of computing A · R and AT · C is               name our algorithm as Network Embedding Update (NEU).
O(|V |d) because matrix A is sparse and has O(|V |) nonzero        NEU avoids an accurate computation of high-order proximi-
entries. Thus the overall time complexity of one iteration of      ty matrix but can yield network embeddings that actually ap-
operation (3) is O(|V |d).                                         proximate high-order proximities. Hence our algorithm can
   Recall that product of previous embedding R and C ap-           improve the quality of network embeddings efficiently. Intu-
proximates polynomial proximity matrix f (A) of degree K.          itively, Eq. (3) and (5) allow the learned embeddings to fur-
Now we prove that the algorithm can learn better embeddings        ther propagate to their neighbors. Thus proximities of longer
R0 and C 0 where the product R0 · C 0T approximates a poly-        distance between vertices will be embedded.
nomial proximity matrix g(A) of degree K + 2 bounded by
matrix infinite norm.                                              5       Experiments
   Theorem Given network and context embedding R and C,
we suppose that the approximation between R · C T and prox-           We evaluate the qualities of network embeddings on two
imity matrix M = f (A) is bounded by r = ||f (A) − R ·             tasks: multi-label classification and link prediction. We per-
C T ||∞ and f (·) is a polynomial of degree K. Then the prod-      form our Network Embedding Update algorithm (NEU) over
uct of updated embeddings R0 and C 0 from Eq. (3) approxi-         the embeddings learned by baseline methods and report both
mates a polynomial g(A) = f (A) + 2λAf (A) + λ2 A2 f (A)           evaluation performance and running time.
of degree K + 2 with approximation bound r0 = (1 + 2λ +
λ2 )r ≤ 94 r.                                                      5.1      Datasets
   Proof Assume that S = f (A)−RC T and thus r = ||S||∞ .             We conduct experiments on three publicly available
                                                                   datasets: Cora1 [Sen et al., 2008], BlogCatalog and Flick-
||g(A) − R0 C 0T ||∞ = ||g(A) − (R + λAR)(C T + λC T A)||∞         r2 [Tang and Liu, 2011]. We assume that all three datasets are
 = ||g(A) − RC T − λARC T − λRC T A − λ2 ARC T A||∞
                                                                   undirected and unweighted networks.
                                                                      Cora contains 2, 708 machine learning papers drawn from
 = ||S + λAS + λSA + λ2 ASA||∞                                     7 classes and 5, 429 citation links between them. Each paper
 ≤ ||S||∞ + λ||A||∞ ||S||∞ + λ||S||∞ ||A||∞ + λ2 ||S||∞ ||A||2∞    has exactly one class label. Each paper in Cora dataset also
                                                                   has text information denoted by a 1, 433 dimensional binary
 = r + 2λr + λ2 r.
                                                                   vector indicating the presence of the corresponding words.
                                                             (4)
where the second last equality replaces g(A) and f (A) −               1
                                                                       http://linqs.cs.umd.edu/projects/
RC T by the definitions of g(A) andP S and the last equali-        /projects/lbc/index.html.
ty uses the fact that ||A||∞ = maxi j |Aij | = 1 because             2
                                                                       http://socialcomputing.asu.edu/pages/
the summation of each row of A equals to 1.                        datasets.
   BlogCatalog contains 10, 312 bloggers and 333, 983 social     5.3    Multi-label Classification
relationships between them. The labels represent the topic          For multi-label classification task, we randomly select a
interests provided by the bloggers. The network has 39 labels    portion of vertices as training set and leave the rest as test
and a blogger may have multiple labels.                          set. We treat network embeddings as vertex features and feed
   Flickr contains 80, 513 users from a photo sharing website    them into a one-vs-rest SVM classifier implemented by Li-
and 5, 899, 882 friendships between them. The labels repre-      bLinear [Fan et al., 2008] as previous works did [Tang and
sent the group membership of users. The network has 195          Liu, 2009; 2011]. We repeat the process for 10 times and
labels and a user may have multiple labels.                      report the average Macro-F1 and Micro-F1 score. Since a
                                                                 vertex of Cora dataset has exactly one label, we only re-
5.2   Baselines and Experimental Settings
                                                                 port classification accuracy for this dataset. We normalize
   We consider a number of baselines to demonstrate the ef-      each dimension of network embeddings so that the L2-norm
fectiveness and robustness of NEU algorithm. For all meth-       of each dimension equals to 1 before we feed the embed-
ods and datasets, we set the embedding dimension d = 128.        dings into the classifier as suggested by [Ben-Hur and We-
   Graph Factorization (GF) simply factorizes the normal-        ston, 2010]. We also perform the same normalization be-
ized adjacency matrix A via SVD decomposition to reduce          fore and after NEU. The experimental results are listed in Ta-
dimensions for network embeddings.                               ble 2, 3 and 4. The numbers in the brackets represent the per-
   Spectral Clustering (SC) [Tang and Liu, 2011] computes        formances of corresponding methods after processing NEU.
the first d eigenvectors of normalized Laplacian matrix as d-    “+0.1”,“+0.3”,“+1” and “+8” in the time column indicate
dimensional embeddings.                                          the additional running time of NEU on Cora, BlogCatalog
   DeepWalk [Perozzi et al., 2014] has three hyperparame-        and Flickr dataset, respectively. As an illustrative example on
ters besides embedding dimension d: window size w, random        Cora dataset, NEU takes 0.1 second and improves the classi-
walk length t and walks per vertex γ. As these hyperparam-       fication accuracy from 78.1 to 84.4 for network embeddings
eters increase, the number of training samples and running       learned by TADW when labeled ratio is 10%. We exclude n-
time will increase. We evaluate three groups of hyperparam-      ode2vec on Flickr as node2vec failed to terminate in 24 hours
eters of DeepWalk, i.e. a default setting of the authors’ im-    on this dataset. We bold the results when NEU achieves more
plementation DeepWalklow where w = 5, t = 40, γ = 10,            than 10% relative improvement. We conduct 0.05 level paired
the setting used in node2vec [Grover and Leskovec, 2016]         t-test and mark all entries that fail to reject the null hypothesis
DeepWalkmid where w = 10, t = 80, γ = 10 and the                 with ∗ .
setting used in the original paper [Perozzi et al., 2014]
DeepWalkhigh where w = 10, t = 40, γ = 80.                                  Table 2: Classification results on Cora dataset.
   LINE [Tang et al., 2015b] learns two separate network rep-
resentations LINE1st and LINE2nd respectively. We use de-                                            % Accuracy
                                                                                                                                  Time (s)
                                                                    % Labeled Nodes       10%           50%           90%
fault settings for all hyperparameters except the number of               GF           50.8 (68.0)   61.8 (77.0)   64.8 (77.2)    4 (+0.1)
total training samples s = 104 |V | so that LINE has compara-             SC           55.9 (68.7)   70.8 (79.2)   72.7 (80.0)    1 (+0.1)
ble running time against DeepWalkmid .                               DeepWalklow       71.3 (76.2)   76.9 (81.6)   78.7 (81.9)   31 (+0.1)
                                                                     DeepWalkmid       68.9 (76.7)   76.3 (82.0)   78.8 (84.3)   69 (+0.1)
   TADW [Yang et al., 2015] incorporates text information           DeepWalkhigh       68.4 (76.1)   74.7 (80.5)   75.4 (81.6)   223 (+0.1)
into DeepWalk and we add this baseline for Cora dataset.               LINE1st         64.8 (70.1)   76.1 (80.9)   78.9 (82.2)   62 (+0.1)
                                                                       LINE2nd         63.3 (73.3)   73.4 (80.1)   75.6 (80.3)   67 (+0.1)
   node2vec [Grover and Leskovec, 2016] is a semi-                     node2vec        76.9 (77.5)   81.0 (81.6)   81.4 (81.9)   56 (+0.1)
supervised NRL method and we use the same hyperparameter                TADW           78.1 (84.4)   83.1 (86.6)   82.4 (87.7)    2 (+0.1)
setting used in their paper: w = 10, t = 80, γ = 10. We em-             GraRep         70.8 (76.9)   78.9 (82.8)   81.8 (84.0)   67 (+0.3)
ploy a grid search over return parameter and in-out parameter
p, q ∈ {0.25, 0.5, 1, 2, 4} for semi-supervised training.
   GraRep [Cao et al., 2015] is only used for the smallest       5.4    Link Prediction
dataset Cora due to its inefficiency [Grover and Leskovec,
2016]. We set K = 5 and thus GraRep has 128 × 5 = 640               For the purpose of link prediction, we need to score each
dimensions.                                                      pair of vertices given their embeddings. For each pair of net-
   Experimental Settings For Spectral Clustering, Deep-          work embedding ri and rj , we try three scoring functions, i.e.
                                                                                        r ·r
Walk, LINE, node2vec, we directly use the implementation-        cosine similarity ||ri ||i2 ||rj j ||2 , inner product ri · rj and inverse
s provided by their authors. We set the hyperparameters of       L2-distance 1/||ri − rj ||2 . We use AUC value [Hanley and
NEU as follows: λ1 = 0.5, λ2 = 0.25 for all three datasets,      McNeil, 1982] which indicates the probability that the score
T = 3 for Cora and BlogCatalog and T = 1 for Flickr. Here        of an unobserved link is higher than that of a nonexistent link
λ1 , λ2 are set empirically following the intuition that lower   as our evaluation metric and select the scoring function with
proximity matrix should have a higher weight and T is set as     best performance for each baseline. We remove 20% edges
the maximum iteration before the performance on 10% ran-         of Cora, 50% of BlogCatalog and Flickr as test set and use
dom validation set begins to drop. In fact, we can simply set    the remaining links to train network embeddings. We also
T = 1 if we have no prior knowledge of downstream tasks.         add three commonly used link prediction baselines for com-
The experiments are executed on a single CPU for the ease        parison: Common Neighbors (CN), Jaccard Index and Salton
of running time comparison and the CPU type is Intel Xeon        Index [Salton and McGill, 1986]. We only report the best
E5-2620 @ 2.0GHz.                                                performance for DeepWalk∈{DeepWalklow , DeepWalkmid ,
                                                                 Table 3: Classification results on BlogCatalog dataset.

                                                                      % Macro-F1                                       % Micro-F1
                                                                                                                                                   Time (s)
                               % Labeled Nodes           1%               5%              9%                 1%            5%            9%
                                     GF               6.6 (7.9)        9.8 (11.3)     10.3 (12.2)        17.0 (19.6)   22.2 (25.0)   23.7 (26.7)    19 (+1)
                                     SC               8.4 (9.3)       13.1 (14.8)     14.5 (17.0)        19.4 (20.3)   26.9 (28.1)   29.0 (31.0)    10 (+1)
                                DeepWalklow          11.3 (12.4)      15.9 (17.4)     17.1 (18.6)        24.5 (26.4)   31.0 (33.4)   32.8 (35.1)   100 (+1)
                                DeepWalkmid          11.2 (13.3)      16.9 (19.2)     18.4 (20.8)        24.0 (27.1)   31.0 (33.8)   32.8 (35.7)   225 (+1)
                                DeepWalkhigh         12.4 (13.6)      18.3 (20.1)     20.4 (22.0)        24.9 (26.4)   31.5 (33.7)   33.7 (35.9)   935 (+1)
                                  LINE1st            11.1 (12.2)      16.6 (18.3)     18.6 (20.1)        23.1 (24.7)   29.3 (31.6)   31.8 (33.5)   241 (+1)
                                  LINE2nd            10.3 (11.2)      15.0 (16.8)     16.5 (18.3)        21.5 (25.0)   27.9 (31.6)   30.0 (33.6)   244 (+1)
                                  node2vec           12.5 (13.0)      19.2 (19.8)     21.9 (22.5)        25.0 (27.0)   31.9 (34.5)   35.1 (37.2)   454 (+1)

                                                                      Table 4: Classification results on Flickr dataset.

                                                                   % Macro-F1                                          % Micro-F1
                                                                                                                                                    Time (s)
                              % Labeled Nodes          1%              5%               9%                  1%             5%          9%
                                    GF              4.3 (5.2)       4.9 (5.4)        5.0 (5.4)          21.1 (21.8)    22.0 (23.1) 21.7 (23.4)      241 (+8)
                                    SC             8.6 (10.9)      11.6 (14.3)      12.3 (15.0)         24.1 (29.2)    27.5 (34.1) 28.3 (34.7)      102 (+8)
                               DeepWalklow          7.8 (8.6)      10.1 (11.6)      10.4 (12.1)         28.5 (31.4)    30.9 (33.5) 31.3 (33.8)     1,449 (+8)
                               DeepWalkmid          8.8 (9.9)      12.3 (14.3)      13.2 (15.1)         29.5 (31.9)    32.4 (35.1) 33.0 (35.4)     2,282 (+8)
                               DeepWalkhigh        10.5 (11.6)     17.1 (17.8)      19.1 (19.8)         31.8 (33.1)    36.3 (36.7) 37.3 (37.6)     9,292 (+8)
                                 LINE1st           10.3 (10.7)     16.0 (16.6)      17.6 (18.2)         32.0 (32.7)    35.9 (36.4) 36.8 (37.2)     2,664 (+8)
                                 LINE2nd            7.8 (8.5)      13.1 (13.5)      14.7 (15.2)         30.0 (31.0)    34.2 (34.4) 35.1 (35.2)∗    2,740 (+8)


DeepWalkhigh } and LINE∈{LINE1st , LINE2nd } and omit                                               dataset where the average degree is 4, NEU has very signifi-
the results of node2vec as it only yields comparable and even                                       cant improvement as high-order proximity plays an important
worse performance than the best performed DeepWalk. The                                             role for sparse networks.
experimental results are shown in Figure 1. For each dataset,                                          (2) NEU facilitates NRL method to converge fast and sta-
the three leftmost columns represent the traditional link pre-                                      bly. We can see that the performances of DeepWalklow +NEU
diction baselines. Then each pair of columns stands for an                                          and DeepWalkmid +NEU are comparable and even better than
NRL method and its performance after NEU.                                                           that of DeepWalkmid and DeepWalkhigh respectively and the
                                                                                                    former ones use much less time. Also, DeepWalk encounters
                      100
                                                                                                    the overfitting problem on Cora dataset as the classification
                                                                              CN
                                                                                                    accuracy drops when hyperparameters increase. However,
                       90
                                                                              Jaccard               the performances of DeepWalk+NEU are stable and robust.
                                                                              Salton
                                                                                                       (3) NEU also works for NRL algorithms which don’t fol-


 AUC values (× 100)
                                                                              GF

                       80
                                                                              GF+NEU
                                                                              SC
                                                                                                    low our two-step NRL framework, i.e. node2vec. This obser-
                                                                              SC+NEU                vation demonstrates the effectiveness and robustness of NEU.
                                                                              DeepWalk
                       70                                                     DeepWalk+NEU
                                                                              LINE
                                                                              LINE+NEU              6     Conclusion
                                                                              TADW
                       60
                                                                              TADW+NEU                 In this paper, we propose a unified NRL framework which
                                                                              GraRep
                                                                              GraRep+NEU            covers a number of existing NRL methods. We analyze the
                       50
                                    Cora      BlogCatalog        Flickr                             first step of the framework, i.e. proximity matrix construc-
                                                                                                    tion, compare the proximity matrices used in different NR-
                             Figure 1: Experimental results on link prediction.                     L algorithms and conclude that we can learn better network
                                                                                                    embeddings if higher order proximities are encoded into the
                                                                                                    proximity matrix. Then we present Network Embedding Up-
5.5                         Experimental Results Analysis                                           date (NEU) algorithm to improve the performance of any giv-
   We have four main observations over the experimental re-                                         en network embeddings by implicitly approximating higher
sults of two evaluation tasks:                                                                      order proximity matrix. The running time of NEU is almost
   (1) NEU consistently and significantly improves the per-                                         negligible and the improvement over baseline methods are
formance of various network embeddings using almost neg-                                            consistent and significant.
ligible running time on both evaluation tasks. The absolute
improvements on Flickr are not as significant as that on Cora                                       Acknowledgements
and BlogCatalog because Flickr dataset has an average degree
of 147 which is much denser than Cora and BlogCatalog and                                             This work is supported by the National 973 Program
thus the impact of higher order proximity information is di-                                        (No.2014CB340501) and the Major Project of the National
luted by rich first-order proximity information. But for Cora                                       Social Science Foundation of China (No.13&ZD190).
References                                                     [Ou et al., 2016] Mingdong Ou, Peng Cui, Jian Pei, Ziwei
[Akoglu et al., 2015] Leman Akoglu, Hanghang Tong, and            Zhang, and Wenwu Zhu. Asymmetric transitivity preserv-
                                                                  ing graph embedding. In Proceedings of KDD, 2016.
  Danai Koutra. Graph based anomaly detection and de-
  scription: a survey. Data Mining and Knowledge Discov-       [Perozzi et al., 2014] Bryan Perozzi, Rami Al-Rfou, and
  ery, 2015.                                                      Steven Skiena. Deepwalk: Online learning of social rep-
                                                                  resentations. In Proceedings of KDD, 2014.
[Ben-Hur and Weston, 2010] Asa Ben-Hur and Jason West-
  on. A users guide to support vector machines. Data mining    [Salton and McGill, 1986] Gerard Salton and Michael J M-
  techniques for the life sciences, 2010.                         cGill. Introduction to modern information retrieval. 1986.
[Bengio et al., 2013] Yoshua Bengio, Aaron Courville, and      [Sen et al., 2008] Prithviraj Sen, Galileo Mark Namata,
  Pascal Vincent. Representation learning: A review and           Mustafa Bilgic, Lise Getoor, Brian Gallagher, and Tina
  new perspectives. IEEE transactions on PAMI, 2013.              Eliassi-Rad. Collective classification in network data. AI
                                                                  Magazine, 2008.
[Cao et al., 2015] Shaosheng Cao, Wei Lu, and Qiongkai X-
                                                               [Tang and Liu, 2009] Lei Tang and Huan Liu. Relational
  u. Grarep: Learning graph representations with global
                                                                  learning via latent social dimensions. In Proceedings of
  structural information. In Proceedings of CIKM, 2015.
                                                                  KDD, 2009.
[Chang et al., 2015] Shiyu Chang, Wei Han, Jiliang Tang,       [Tang and Liu, 2011] Lei Tang and Huan Liu. Leveraging
  Guo-Jun Qi, Charu C Aggarwal, and Thomas S Huang.               social media networks for classification. Proceedings of
  Heterogeneous network embedding via deep architectures.         KDD, 2011.
  In Proceedings of KDD, 2015.
                                                               [Tang et al., 2015a] Jian Tang, Meng Qu, and Qiaozhu Mei.
[Fan et al., 2008] Rong-En Fan, Kai-Wei Chang, Cho-Jui H-         Pte: Predictive text embedding through large-scale hetero-
   sieh, Xiang-Rui Wang, and Chih-Jen Lin. Liblinear: A           geneous text networks. In Proceedings of KDD, 2015.
   library for large linear classification. JMLR, 2008.
                                                               [Tang et al., 2015b] Jian Tang, Meng Qu, Mingzhe Wang,
[Grover and Leskovec, 2016] Aditya Grover and Jure                Ming Zhang, Jun Yan, and Qiaozhu Mei. Line: Large-
  Leskovec.    node2vec: Scalable feature learning for            scale information network embedding. In Proceedings of
  networks. 2016.                                                 WWW, 2015.
[Hanley and McNeil, 1982] James A Hanley and Barbara J         [Tu et al., 2014] Cunchao Tu, Zhiyuan Liu, and Maosong
  McNeil. The meaning and use of the area under a receiver        Sun. Inferring correspondences from multiple sources for
  operating characteristic (roc) curve. Radiology, 1982.          microblog user tags. In Proceedings of SMP, 2014.
[Huang and Mamoulis, 2017] Zhipeng Huang and Nikos             [Tu et al., 2016] Cunchao Tu, Weicheng Zhang, Zhiyuan Li-
  Mamoulis. Heterogeneous information network embed-              u, and Maosong Sun. Max-margin deepwalk: discrimina-
  ding for meta path based proximity. arXiv preprint arX-         tive learning of network representation. In Proceedings of
  iv:1701.05291, 2017.                                            IJCAI, 2016.
[Huang et al., 2017] Xiao Huang, Jundong Li, and Xia Hu.       [Wang et al., 2016] Daixin Wang, Peng Cui, and Wenwu
  Label informed attributed network embedding. In Pro-            Zhu. Structural deep network embedding. In Proceedings
  ceedings of WSDM, 2017.                                         of KDD, 2016.
[Kipf and Welling, 2017] Thomas N Kipf and Max Welling.        [Wang et al., 2017] Xiao Wang, Peng Cui, Jing Wang, Jian
  Semi-supervised classification with graph convolutional         Pei, Wenwu Zhu, and Shiqiang Yang. Community pre-
  networks. In Proceedings of ICLR, 2017.                         serving network embedding. Proceedings of AAAI, 2017.
[Ley, 2002] Michael Ley. The dblp computer science bibli-      [Xu et al., 2017] Linchuan Xu, Xiaokai Wei, Jiannong Cao,
   ography: Evolution, research issues, perspectives. In In-      and Philip S Yu. Embedding of embedding (eoe): Joint
   ternational symposium on string processing and informa-        embedding for coupled heterogeneous networks. In Pro-
   tion retrieval, 2002.                                          ceedings of WSDM, 2017.
                                                               [Yang et al., 2015] Cheng Yang, Zhiyuan Liu, Deli Zhao,
[Li et al., 2016] Juzheng Li, Jun Zhu, and Bo Zhang. Dis-
                                                                  Maosong Sun, and Edward Y Chang. Network represen-
   criminative deep random walk for network classification.
                                                                  tation learning with rich text information. In Proceedings
   In Proceedings of ACL, 2016.
                                                                  of IJCAI, 2015.
[Liben-Nowell and Kleinberg, 2007] David Liben-Nowell          [Yang et al., 2016] Zhilin Yang, William Cohen, and Ruslan
   and Jon Kleinberg. The link-prediction problem for social      Salakhutdinov. Revisiting semi-supervised learning with
   networks. journal of the Association for Information           graph embeddings. In Proceedings of ICML, 2016.
   Science and Technology, 2007.
[Mikolov et al., 2013] Tomas Mikolov, Ilya Sutskever, Kai
  Chen, Greg S Corrado, and Jeff Dean. Distributed rep-
  resentations of words and phrases and their composition-
  ality. In Proceedings of NIPS, 2013.

