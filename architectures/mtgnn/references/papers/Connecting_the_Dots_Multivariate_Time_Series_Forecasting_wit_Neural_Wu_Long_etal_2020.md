# Connecting the Dots Multivariate Time Series Forecasting wit Neural Wu Long etal 2020

> Source: `Connecting_the_Dots_Multivariate_Time_Series_Forecasting_wit_Neural_Wu_Long_etal_2020.pdf`

---

                                          Connecting the Dots: Multivariate Time Series Forecasting with
                                                            Graph Neural Networks
                                                              Zonghan Wu                                                      Shirui Pan∗                                Guodong Long
                                                University of Technology Sydney                                         Monash University                       University of Technology Sydney
                                               zonghan.wu-3@student.uts.edu.au                                       shirui.pan@monash.edu                         guodong.long@uts.edu.au

                                                                 Jing Jiang                                               Xiaojun Chang                                  Chengqi Zhang
                                                University of Technology Sydney                                        Monash University                        University of Technology Sydney
                                                     jing.jiang@uts.edu.au                                        xiaojun.chang@monash.edu                        chengqi.zhang@uts.edu.au




arXiv:2005.11650v1 [cs.LG] 24 May 2020
                                         ABSTRACT                                                                                      ACM Reference Format:
                                         Modeling multivariate time series has long been a subject that has                            Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, and Chengqi
                                                                                                                                       Zhang. 2020. Connecting the Dots: Multivariate Time Series Forecasting
                                         attracted researchers from a diverse range of fields including eco-
                                                                                                                                       with Graph Neural Networks. In 26th ACM SIGKDD Conference on Knowl-
                                         nomics, finance, and traffic. A basic assumption behind multivariate                          edge Discovery and Data Mining (KDD ’20), August 23–27, 2020, Virtual Event,
                                         time series forecasting is that its variables depend on one another                           USA. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/XXXXXX.
                                         but, upon looking closely, it’s fair to say that existing methods fail to                     XXXXXX
                                         fully exploit latent spatial dependencies between pairs of variables.
                                         In recent years, meanwhile, graph neural networks (GNNs) have
                                         shown high capability in handling relational dependencies. GNNs                               1    INTRODUCTION
                                         require well-defined graph structures for information propagation                             Modern societies have benefited from a wide range of sensors
                                         which means they cannot be applied directly for multivariate time                             to record changes in temperature, price, traffic speed, electricity
                                         series where the dependencies are not known in advance. In this                               usage, and many other forms of data. Recorded time series from
                                         paper, we propose a general graph neural network framework de-                                different sensors can form multivariate time series data and can
                                         signed specifically for multivariate time series data. Our approach                           be interlinked. For example, the rise in daily temperature may
                                         automatically extracts the uni-directed relations among variables                             cause an increase in electricity usage. To capture systematic trends
                                         through a graph learning module, into which external knowledge                                over a group of dynamically changing variables, the problem of
                                         like variable attributes can be easily integrated. A novel mix-hop                            multivariate time series forecasting has been studied for at least
                                         propagation layer and a dilated inception layer are further proposed                          sixty years. It has seen tremendous applications in the domains of
                                         to capture the spatial and temporal dependencies within the time                              economics, finance, bioinformatics, and traffic.
                                         series. The graph learning, graph convolution, and temporal con-                                 Multivariate time series forecasting methods inherently assume
                                         volution modules are jointly learned in an end-to-end framework.                              interdependencies among variables. In other words, each variable
                                         Experimental results show that our proposed model outperforms                                 depends not only on its historical values but also on other variables.
                                         the state-of-the-art baseline methods on 3 of 4 benchmark datasets                            However, existing methods do not exploit latent interdependencies
                                         and achieves on-par performance with other approaches on two                                  among variables efficiently and effectively. Statistical methods, such
                                         traffic datasets which provide extra structural information.                                  as vector auto-regressive model (VAR) and Gaussian process model
                                                                                                                                       (GP), assume a linear dependency among variables. The model
                                         CCS CONCEPTS                                                                                  complexity of statistical methods grows quadratically with the
                                         • Computing methodologies → Neural networks; Artificial                                       number of variables. They face the problem of overfitting with a
                                         intelligence.                                                                                 large number of variables. Recently developed deep-learning-based
                                                                                                                                       methods, including LSTNet [12] and TPA-LSTM [19], are powerful
                                         KEYWORDS                                                                                      to capture non-linear patterns. LSTNet encodes short-term local
                                         Graph neural networks, graph structure learning, multivariate time                            information into low dimensional vectors using 1D convolutional
                                         series forecasting, spatial-temporal graphs                                                   neural networks and decodes the vectors through a recurrent neural
                                                                                                                                       network. TPA-LSTM processes the inputs by a recurrent neural
                                         ∗ Corresponding Author.
                                                                                                                                       network and employs a convolutional neural network to calculate
                                                                                                                                       the attention score across multiple steps. LSTNet and TPA-LSTM do
                                         Permission to make digital or hard copies of all or part of this work for personal or
                                         classroom use is granted without fee provided that copies are not made or distributed         not model the pair-wise dependencies among variables explicitly,
                                         for profit or commercial advantage and that copies bear this notice and the full citation     which weakens model interpretability.
                                         on the first page. Copyrights for components of this work owned by others than ACM               Graphs are a special form of data which describes the relation-
                                         must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
                                         to post on servers or to redistribute to lists, requires prior specific permission and/or a   ships between different entities. Recently, graph neural networks
                                         fee. Request permissions from permissions@acm.org.                                            have achieved great success in handling graph data due to their
                                         KDD ’20, August 23–27, 2020, Virtual Event, USA                                               permutation-invariance, local connectivity, and compositionality.
                                         © 2020 Association for Computing Machinery.
                                         ACM ISBN 978-1-4503-7998-4/20/08. . . $15.00                                                  By propagating information through structures, graph neural net-
                                         https://doi.org/10.1145/XXXXXX.XXXXXX                                                         works allow each node in a graph to be aware of its neighborhood
context. Multivariate time series forecasting can be viewed natu-
rally from a graph perspective. Variables from multivariate time             Graph Structure           Graph
series can be considered as nodes in a graph, and they are inter-               Learning             Convolution
linked through their hidden dependency relationships. It follows
that modeling multivariate time series data using graph neural net-
                                                                                Time series           Temporal              Forecasting
works can be a promising way to preserve their temporal trajectory             (data points)         Convolution              Results
while exploiting the interdependency among time series.
   The most suitable type of graph neural networks for multivari-
ate time series is spatial-temporal graph neural networks. Spatial-
temporal graph neural networks take multivariate time series and           Figure 1: A concept map of our proposed framework.
an external graph structure as inputs, and they aim to predict fu-
ture values or labels of multivariate time series. Spatial-temporal
graph neural networks have achieved significant improvements                • To the best of our knowledge, this is the first study on mul-
compared to methods that do not utilize structural information.               tivariate time series data generally from a graph-based per-
However, these approaches still fall short for modeling multivariate          spective with graph neural networks.
time series due to the following challenges:                                • We propose a novel graph learning module to learn hidden
                                                                              spatial dependencies among variables. Our method opens a
    • Challenge 1: Unknown Graph Structure. Existing GNN ap-                  new door for GNN models to handle data without explicit
      proaches rely heavily on a pre-defined graph structure in               graph structure.
      order to perform time series forecasting. In most cases, multi-       • We present a joint framework for modeling multivariate time
      variate time series does not have an explicit graph structure.          series data and learning graph structures. Our framework
      The relationships among variables has to be discovered from             is more generic than any existing spatial-temporal graph
      data rather than being provided as ground truth knowledge.              neural network as it can handle multivariate time series
    • Challenge 2: Graph Learning & GNN Learning. Even though                 with or without a pre-defined graph structure.
      a graph structure is available, most GNN approaches focus             • Experimental results show that our method outperforms the
      only on message passing (GNN Learning) and overlook the                 state-of-the-art methods on 3 of 4 benchmark datasets and
      fact that the graph structure is not optimal and should be              achieves on-par performance with other GNNs on two traffic
      updated during training. The question then is how to simul-             datasets which provide extra structural information.
      taneously learn the graph structure and the GNN for time
      series in an end-to-end framework.                                2 BACKGROUNDS
   In this paper, we propose a novel approach to overcome these         2.1 Multivariate Time Series Forecasting
challenges. As demonstrated by Figure 1, our framework consists         Time series forecasting has been studied for a long time. The ma-
of three core components - the graph learning layer, the graph          jority of existing methods follow a statistical approach. The auto-
convolution module, and the temporal convolution module. For            regressive integrated moving average (ARIMA) [1] generalizes a
Challenge 1, we propose a novel graph learning layer, which ex-         family of a linear model, including auto-regressive (AR), moving
tracts a sparse graph adjacency matrix adaptively based on data.        average (MA), and auto-regressive moving average (ARMA). The
Furthermore, we develop a graph convolution module to address the       vector auto-regressive model (VAR) extends the AR model to cap-
spatial dependencies among variables, given the adjacency matrix        ture the linear interdependencies among multiple time series. Simi-
computed by the graph learning layer. This is designed specifically     larly, the vector auto-regressive moving average model (VARMA) is
for directed graphs and avoids the over-smoothing problem that          proposed as a multivariate version of the ARMA model. Gaussian
frequently occurs in graph convolutional networks. Finally, we pro-     process (GP), as a Bayesian approach, models the distribution of a
pose a temporal convolution module to capture temporal patterns         multivariate variable over functions. GP can be applied naturally to
by modified 1D convolutions. It can both discover temporal patterns     model multivariate time series data [5]. Although statistical models
with multiple frequencies and process very long sequences.              are widely used in time series forecasting due to their simplicity
   As all parameters are learnable through gradient descent, the        and interpretability, they make strong assumptions with respect
proposed framework is able to model multivariate time series data       to a stationary process and they do not scale well to multivariate
and learn the internal graph structure simultaneously in an end-to-     time series data. Deep-learning-based approaches are free from
end manner (for Challenge 2). To reduce the difficulty of solving a     stationary assumptions and they are effective methods to capture
highly non-convex optimization problem and to reduce memory             non-linearity. Lai et al. [12] and Shih et al. [19] are the first two
occupation in processing large graphs, we propose a learning algo-      deep-learning-based models designed for multivariate time series
rithm that uses a curriculum learning strategy to find a better local   forecasting. They employ convolutional neural networks to capture
optimum and splits multivariate time series into subgroups during       local dependencies among variables and recurrent neural networks
training. The advantages here are that our proposed framework is        to preserve long-term temporal dependencies. Convolutional neu-
generally applicable to both small and large graphs, short and          ral networks encapsulate interactions among variables into a global
long time series, with and without externally defined graph             hidden state. Therefore, they cannot fully exploit latent dependen-
structures. In summary, our main contributions are as follows:          cies between pairs of variables.
2.2    Graph Neural Networks                                                   among nodes using the graph adjacency matrix. The graph adja-
Graph neural networks have enjoyed great success in handling                   cency matrix is not given by the multivariate time series data in
spatial dependencies among entities in a network. Graph neural                 most cases and will be learned by our model.
networks assume that the state of a node depends on the states of its
neighbors. To capture this type of spatial dependency, various kinds           4 FRAMEWORK OF MTGNN
of graph neural networks have been developed through message                   4.1 Model Architecture
passing [7], information propagation [11], and graph convolution
                                                                               We first elaborate on the general framework of our model. As illus-
[10]. Sharing similar roles, they essentially capture a node’s high-
                                                                               trated in Figure 2, MTGNN on the highest level consists of a graph
level representation by passing information from a node’s neighbors
                                                                               learning layer, m graph convolution modules, m temporal convolution
to the node itself. Most recently, we have seen the emergence of a
                                                                               modules, and an output module. To discover hidden associations
type of graph neural networks known as spatial-temporal graph
                                                                               among nodes, a graph learning layer computes a graph adjacency
neural networks. This form of neural networks is proposed initially
                                                                               matrix, which is later used as an input to all graph convolution
to solve the problem of traffic prediction [3, 13, 21, 23, 26] and
                                                                               modules. Graph convolution modules are interleaved with temporal
skeleton-based action recognition [18, 22]. The inputs to spatial-
                                                                               convolution modules to capture spatial and temporal dependencies
temporal graph neural networks are multivariate time series with an
                                                                               respectively. Figure 3 gives a demonstration of how a temporal con-
external graph structure which describes the relationships among
                                                                               volution module and a graph convolution module collaborate with
variables in multivariate time series. For spatial-temporal graph
                                                                               each other. To avoid the problem of gradient vanishing, residual
neural networks, spatial dependencies among nodes are captured by
                                                                               connections are added from the inputs of a temporal convolution
graph convolutions, while temporal dependencies among historical
                                                                               module to the outputs of a graph convolution module. Skip connec-
states are preserved by recurrent neural networks [13, 17] or 1D
                                                                               tions are added after each temporal convolution module. To get the
convolutions [22, 23]. Although existing spatial-temporal graph
                                                                               final outputs, the output module projects the hidden features to the
neural networks have achieved significant improvements compared
                                                                               desired output dimension. In more detail, the core components of
to methods without using a graph structure, they are incapable of
                                                                               our model are illustrated in the following:
handling pure multivariate time series data effectively due to the
absence of a pre-defined graph and lack of a general framework.
                                                                               4.2    Graph Learning Layer
3     PROBLEM FORMULATION                                                      The graph learning layer learns a graph adjacency matrix adap-
In this paper, we focus on the task of multivariate time series fore-          tively to capture the hidden relationships among time series data.
casting. Let zt ∈ R N denote the value of a multivariate variable of           To construct a graph, existing studies measure the similarity be-
dimension N at time step t, where zt [i] ∈ R denote the value of the           tween pairs of nodes by a distance metric, such as dot product and
i th variable at time step t. Given a sequence of historical P time steps      Euclidean distance [13]. This leads inevitably to the problem of high
of observations on a multivariate variable, X = {zt1 , zt2 , · · · , zt P },   time and space complexity with O(N 2 ). It means the computation
our goal is to predict the Q-step-away value of Y = {zt P +Q }, or             and memory cost grows quadratically with the increase of graph
a sequence of future values Y = {zt P +1 , zt P +2 , · · · , zt P +Q }. More   size. This restricts the model’s capability of handling larger graphs.
generally, the input signals can be coupled with other auxiliary               To address this limitation, we adopt a sampling approach, which
features such as time of the day, day of the week, and day of the              only calculates pair-wise relationships among a subset of nodes.
season. Concatenating the input signals with auxiliary features,               This cuts off the bottleneck of computation and memory in each
we assume the inputs instead are X = {St1 , St2 , · · · , St P } where         minibatch. More details will be provided in Section 4.6.
                                                                                  Another problem is that existing distance metrics are often sym-
Sti ∈ R N ×D , D is the feature dimension, the first column of Sti
                                                                               metric or bi-directional. In multivariate time series forecasting, we
equals to zti , and the rest are auxiliary features. We aim to build a
                                                                               expect that the change of a node’s condition causes the change of
mapping f (·) from X to Y by minimizing the absolute loss with l2
                                                                               another node’s condition such as traffic flow. Therefore the learned
regularization.
                                                                               relation is supposed to be uni-directional. Our proposed graph
    Graphs describe the relationships among entities in a network.
                                                                               learning layer is specifically designed to extract uni-directional
We give a formal definition of graph-related concepts below.
                                                                               relationships, illustrated as follows:
   Definition 3.1 (Graph). A graph is formulated as G = (V , E) where
V is the set of nodes, and E is the set of edges. We use N to denote                         M1 = tanh(αE1 Θ1 )                                  (1)
the number of nodes in a graph.                                                              M2 = tanh(αE2 Θ2 )                                  (2)
  Definition 3.2 (Node Neighborhood). Let v ∈ V to denote a node                               A = ReLU (tanh(α(M1 MT2 − M2 MT1 )))              (3)
and e = (v, u) ∈ E to denote an edge pointing from u to v. The
                                                                                             f or i = 1, 2, · · · , N                            (4)
neighborhood of a node v is defined as N (v) = {u ∈ V |(v, u) ∈ E}.
                                                                                                 idx = arдtopk(A[i, :])                          (5)
  Definition 3.3 (Adjacency Matrix). The adjacency matrix is a
mathematical representation of a graph, denoted as A ∈ R N ×N                                    A[i, −idx] = 0,                                 (6)
with Ai j = c > 0 if (vi , v j ) ∈ E and Ai j = 0 if (vi , v j ) < E.          where E1 , E2 represents randomly initialized node embeddings,
   From a graph-based perspective, we consider variables in multi-             which are learnable during training, Θ1 , Θ2 are model parameters,
variate time series as nodes in graphs. We describe the relationships          α is a hyper-parameter for controlling the saturation rate of the
                   Node embeddings
                                                               Graph Learning Layer
                   /static features

                             Residual Connections         Residual Connections                         Residual Connections

                                                    A                                 A                                          A
                   1×1         TC Module         GC         TC Module           GC              …        TC Module         GC
                   Conv          (d=𝑞 " )       Module        (d=𝑞# )          Module                     (d=𝑞 $ )        Module



                  Inputs:
                    𝜒 ∈ 𝑅 *+,×-×.      Skip Connections                      Skip Connections                          Skip Connections
                                                                Output Module                           0 ∈ 𝑅 *123×-
                                                                                                Outputs:Υ



Figure 2: The framework of MTGNN. A 1 × 1 standard convolution first projects the inputs into a latent space. Afterward, tem-
poral convolution modules and graph convolution modules are interleaved with each other to capture temporal and spatial
dependencies respectively. The hyper-parameter, dilation factor d, which controls the receptive field size of a temporal con-
volution module, is increased at an exponential rate of q. The graph learning layer learns the hidden graph adjacency matrix,
which is used by graph convolution modules. Residual connections and skip connections are added to the model to avoid the
problem of gradient vanishing. The output module projects hidden features to the desired dimension to get the final results.


                                                                                                                                                  𝐻*+,
                                                                                                +                                             +

       N                                    N                                 A     Mix-hop           Mix-hop     AT
                                                                                  Propagation       Propagation          MLP0         MLP1                MLPK
                                                                                     Layer             Layer
                                                                                                                                  A               A … A
                              D                           D’                                                             𝐻 (()        𝐻 (#)               𝐻 (')
                  T                               T’                                                                        𝐻%&           𝐻%&                𝐻%&

                                                                                          (a) GC module                 (b) Mix-hop propagation layer
Figure 3: A demonstration of how a temporal convolution module
and a graph convolution module collaborate with each other. A tem-
                                                                               Figure 4: Graph convolution and mix-hop propagation layer.
poral convolution module filters the inputs by sliding a 1D window
over the time and node axes, as denoted by the red. A graph convo-
lution module filters the inputs at each step, denoted by the blue.
                                                                            advantage of our approach is that we can learn stable and inter-
                                                                            pretable node relationships over the period of the training dataset.
                                                                            Once the model is trained in an on-line learning version, our graph
activation function, and arдtopk(·) returns the index of the top-k          adjacency matrix is also adaptable to change as new training data
largest values of a vector. The asymmetric property of our proposed         updates the model parameters.
graph adjacency matrix is achieved by Equation 3. The subtraction
term and the ReLU activation function regularize the adjacency              4.3       Graph Convolution Module
matrix so that if Avu is positive, its diagonal counterpart Auv will        The graph convolution module aims to fuse a node’s information
be zero. Equation 5-6 is a strategy to make the adjacency matrix            with its neighbors’ information to handle spatial dependencies
sparse while reducing the computation cost of the following graph           in a graph. The graph convolution module consists of two mix-
convolution. For each node, we select its top-k closest nodes as its        hop propagation layers to process inflow and outflow information
neighbors. While retaining the weights for connected nodes, we              passed through each node separately. The net inflow information
set the weights of non-connected nodes as zero.                             is obtained by adding the outputs of the two mix-hop propagation
   Incorporate External Data. The inputs to the graph learning layer        layers. Figure 4 shows the architecture of the graph convolution
are not limited to node embeddings. In case that external knowl-            module and the mix-hop propagation layer.
edge about the attributes of each node is given, we can also set
E1 = E2 = Z, where Z is a static node feature matrix. Some works               Mix-hop Propagation Layer. Given a graph adjacency matrix, we
have considered capturing dynamic spatial dependencies [8, 18]. In          propose the mix-hop propagation layer to handle information flow
other words, they dynamically adjust the weight of two connected            over spatially dependent nodes. The proposed mix-hop propaga-
nodes based on temporal inputs. However, assuming dynamic spa-              tion layer consists of two steps - the information propagation step
tial dependencies makes the model extremely hard to converge                and the information selection step. We first give the mathematical
when we need to learn the graph structure at the same time. The             form of these two steps and then illustrate our motivations. The
information propagation step is defined as follows:
                                                                                    tanh        sigmoid
                                                                                            ×
                      (k )                             (k −1)                                                              Concatenate
                  H          = βHin + (1 − β)ÃH                ,   (7)          Dilated         Dilated
                                                                                Inception       Inception
                                                                                                                    1×2    1×3    1×6    1×7
where β is a hyper parameter, which controls the ratio of retaining               Layer           Layer

the root node’s original states. The information selection step is
defined as follows

                                      K                                            (a) TC module                   (b) Dilated inception layer
                                      Õ
                             Hout =         H(k ) W(k ) ,           (8)
                                      i=0                                  Figure 5: The temporal convolution and dilated inception layer.

where K is the depth of propagation, Hin represents the input
hidden states outputted by the previous layer, Hout represents            4.4    Temporal Convolution Module
the output hidden states of the current layer, H(0) = Hin , Ã =          The temporal convolution module applies a set of standard dilated
D̃−1 (A + I), and D̃ii = 1 + j Ai j . In Figure 4b, we demonstrate the
                             Í
                                                                          1D convolution filters to extract high-level temporal features. This
information propagation step and information selection step in the        module consists of two dilated inception layers. One dilated incep-
proposed mix-hop propagation layer. It first propagates information       tion layer is followed by a tangent hyperbolic activation function
horizontally and selects information vertically.                          and works as a filter. The other layer is followed by a sigmoid acti-
   The information propagation step propagates node information           vation function and functions as a gate to control the amount of
along with the given graph structure recursively. A severe limitation     information that the filter can pass to the next module. Figure 5
of graph convolutional networks is that node hidden states converge       shows the architecture of the temporal convolution module and
to a single point as the number of graph convolution layers goes to       the dilated inception layer.
infinity. This is because the graph convolutional network with many
                                                                              Dilated Inception Layer. The temporal convolution module cap-
layers reaches the random walk’s limit distribution regardless of the
                                                                          tures sequential patterns of time series data through 1D convo-
initial node states. To address this problem, motivated by Klicpera
                                                                          lutional filters. To come up with a temporal convolution module
et al. [11], we retain a proportion of nodes’ original states during
                                                                          that is able to both discover temporal patterns with various ranges
the propagation process so that the propagated node states can
                                                                          and handle very long sequences, we propose the dilated inception
both preserve locality and explore a deep neighborhood. However,
                                                                          layer which combines two widely applied strategies from convo-
if we only apply Equation 7, some node information will be lost.
                                                                          lutional neural networks, i.e., using filters with multiple sizes [20]
Under the extreme circumstance that no spatial dependencies exist,
                                                                          and applying dilated convolution [24].
aggregating neighborhood information simply adds useless noises
                                                                              First, choosing the right kernel size is a challenging problem for
to each node. Therefore, the information selection step is introduced
                                                                          convolutional networks. The filter size can be too large to represent
to filter out important information produced at each hop. According
                                                                          short-term signal patterns subtly, or too small to discover long-term
to Equation 8, the parameter matrix W(k ) functions as a feature
                                                                          signal patterns sufficiently. In image processing, a widely employed
selector. When the given graph structure does not entail spatial
                                                                          strategy is called inception, which concatenates the outputs of 2D
dependencies, Equation 8 is still able to preserve the original node-
                                                                          convolution filters with three different kernel sizes, 1 × 1, 3 × 3, and
self information by adjusting W(k ) to 0 for all k > 0.                   5 × 5. Moving from 2D images to 1D time series, the set of 1 × 1,
   Connection to existing works. The idea of mix-hop has been ex-         1 × 3, and 1 × 5 filter sizes do not suit the nature of temporal signals.
plored by [9] and [2]. Kapoor et al. [9] concatenate information          As temporal signals tend to have several inherent periods such as
from different hops. Chen et al. [2] propose an attention mecha-          7, 12, 24, 28, and 60, a stack of inception layers with filter size 1 × 1,
nism to weight information among different hops. They both apply          1 × 3, and 1 × 5 cannot well encompass those periods. Alternatively,
GCN for information propagation. However, as GCN faces the over-          we propose a temporal inception layer consisting of four filter sizes,
smoothing problem, information from higher hops may not or                viz. 1 × 2, 1 × 3, 1 × 6, and 1 × 7. The aforementioned periods can
negatively contribute to the overall performance. To avoid this, our      all be covered by the combination of these filter sizes. For example,
approach keeps a balance between local and neighborhood infor-            to represent the period 12, a model can pass the inputs through a
mation. Furthermore, Kapoor et al. [9] show that their proposed           1 × 7 filter from the first temporal inception layer followed by a
model with two mix-hop layers has the capability to represent the         1 × 6 filter from the second temporal inception layer.
delta difference between two consecutive hops. Our approach can               Second, the receptive field size of a convolutional network grows
achieve the same effect with only one mix-hop propagation layer.          in a linear progression with the depth of the network and the kernel
Suppose K = 2, W(0) = 0, W(1) = −1, and W(2) = 1, then                    size of the filter. Consider a convolutional network with m 1D
                                                                          convolution layers of kernel size c, the receptive field size of the
                 Hout = ∆(H(2) , H(1) ) = H2 − H1 .                 (9)   convolutional network is,
                                                                                                       R = m(c − 1) + 1.                       (10)
From this perspective, using summation is more efficient to rep-
resent all linear interactions of different hops compared with the        To process very long sequences, it requires either a very deep net-
concatenation method.                                                     work or very large filters. We adopt dilated convolution to reduce
model complexity. Dilated convolution operates a standard convo-           Algorithm 1 The learning algorithm of MTGNN.
lution filter on down-sampled inputs with a certain frequency. For             1: Input: The dataset O , node set V , the initialized MTGNN model f (·)
example, where the dilation factor is 2, it applies standard convo-              with Θ, learning rate γ , batch size b, step size s, split size m (default=1).
lution on inputs sampled every two steps. Following [14], we let               2: set it er = 1, r = 1
the dilation factor for each layer increase exponentially at a rate of         3: repeat
                                                                                                                                       ′
q (q > 1). Suppose the initial dilation factor is 1, the receptive field    4:     sample a batch (X ∈ R b×T ×N ×D , Y ∈ R b×T ×N ) from O .
size of a m layer dilated convolutional network with kernel size c is       5:     random split the node set V into m groups, ∪m      i =1Vi = V .
                                                                            6:     if it er %s == 0 and r <= T ′ then
                  R = 1 + (c − 1)(qm − 1)/(q − 1).                  (11)    7:         r =r +1
This indicates that the receptive field size of the network also grows      8:     end if
                                                                            9:     for i in 1:m do
exponentially with an increase in the number of hidden layers at
                                                                           10:         compute Ŷ = f (X[:, :, id(Vi ), :]; Θ)
the rate of q. Therefore, using this dilation strategy can capture
                                                                           11:         compute L = loss( Ŷ[:, : r, :], Y[:, : r, id (Vi )])
much longer sequences than proceeding without it.                          12:         compute the stochastic gradient of Θ according to L.
   Formally, combining inception and dilation, we propose the di-          13:         update model parameters Θ according to their gradients and
lated inception layer, demonstrated by Figure 5b. Given a 1D se-               the learning rate γ .
quence input z ∈ RT and filters consisting of f1×2 ∈ R2 , f1×3 ∈ R3 ,      14:     end for
f1×6 ∈ R6 , and f1×7 ∈ R7 , our dilated inception layer takes the form,    15:     it er = it er + 1.
                                                                           16: until convergence
          z = concat(z ⋆ f1×2 , z ⋆ f1×3 , z ⋆ f1×6 , z ⋆ f1×7 ),   (12)
where the outputs of the four filters are truncated to the same
length according to the largest filter and concatenated across the         into s groups, we can reduce the time and space complexity of our
channel dimension, and the dilated convolution denoted by z⋆f1×k           graph learning layer from O(N 2 ) to (N /s)2 in each iteration. After
is defined as                                                              training, as all node embeddings are well-trained, a global graph
                                 kÕ
                                  −1                                       can be constructed to fully utilize spatial relationships. Although
                z ⋆ f1×k (t) =         f1×k (s)z(t − d × s),        (13)   it is computationally expensive, the adjacency matrix can be pre-
                                 s=0                                       computed in parallel before making predictions.
where d is the dilation factor.                                                The second consideration of our proposed algorithm is to fa-
                                                                           cilitate our model stabilize in a better local optimum. In the task
4.5    Skip Connection Layer & Output Module                               of multi-step forecasting, we observe that long-term predictions
Skip connection layers are essentially 1 × Li standard convolutions        often achieve greater improvements than those in the short-term
where Li is the sequence length of the inputs to the i t h skip con-       in terms of model performance. We believe the reason is that our
nection layer. It standardizes information that jumps to the output        model predicts multi-steps altogether, and long-term predictions
module to have the same sequence length 1. The output module               produce a much higher loss than short-term predictions. As a result,
consists of two 1 × 1 standard convolution layers, transforming the        to minimize the overall loss, the model focuses more on improving
channel dimension of the inputs to the desired output dimension.           the accuracy of long-term predictions. To address this issue we
In case we want to predict a certain future step only, the desired         propose a curriculum learning strategy for the multi-step forecast-
output dimension is 1. When we want to predict Q consecutive               ing task. The algorithm starts with solving the easiest problem,
steps, the desired output dimension is Q.                                  predicting the next one-step only. It is very advantageous for the
                                                                           model to find a good starting point. With the increase in iteration
4.6    Proposed Learning Algorithm                                         numbers, we increase the prediction length of the model gradually
                                                                           so that the model can learn the hard task step by step. Covering
We propose a learning algorithm to enhance our model’s capability
                                                                           all this, our algorithm is given in Algorithm 1. Further complexity
of handling large graphs and stabilizing in a better local optimum.
                                                                           analysis of our model can be found in Appendix A.1.
Training on a graph often requires storing all node intermediate
states into memory. If a graph is large, it will face the problem of
memory overflow. Most relevant to us, Chiang et al. [4] propose a          5      EXPERIMENTAL STUDIES
sub-graph training algorithm to tackle the memory bottleneck. They         We validate MTGNN on two tasks - both single-step and multi-
apply a graph clustering algorithm to partition a graph into sub-          step forecasting. First, we compare the performance of MTGNN
graphs and train a graph convolutional network on the partitioned          with other multivariate time series models on four benchmark
sub-graphs. In our problem, it is not practical to cluster nodes based     datasets for multivariate time series forecasting, where the aim
on their topological information because our model learns the latent       is to predict a single future step. Furthermore, to show how well
graph structure at the same time. Alternatively, in each iteration, we     MTGNN performs, compared with other spatial-temporal graph
randomly split the nodes into several groups and let the algorithm         neural networks which, in contrast, use pre-defined graph structural
learn a sub-graph structure based on the sampled nodes. This gives         information, we evaluate MTGNN on two benchmark datasets for
each node the full possibilities of being assigned with another node       spatial-temporal graph neural networks, where the aim is to predict
in one group so that the similarity score between these two nodes          multiple future steps. Further results on parameter study can be
can be computed and updated. As a side benefit, if we split the nodes      found in Appendix A.4.
                        Table 1: Dataset statistics.                                of MTGNN only degrades marginal when it samples sub-graphs
                                                                                    for training. In the following, we discuss experimental results of
 Datasets        # Samples   # Nodes   Sample Rate   Input Length   Output Length   single-step and multi-step forecasting respectively.
 traffic            17,544       862        1 hour           168               1
 solar-energy       52,560       137    10 minutes           168               1    5.3.1 Single-step forecasting. In this experiment, we compare MT-
 electricity        26,304       321        1 hour           168               1    GNN with other multivariate time series models. Table 2 shows
 exchange-rate       7,588         8         1 day           168               1    the experimental results for the single-step forecasting task. In gen-
 metr-la            34272        207     5 minutes            12              12    eral, our MTGNN achieves state-of-the-art results over almost all
 pems-bay           52116        325     5 minutes            12              12
                                                                                    horizons on Solar-Energy, Traffic, and Electricity data. In particular,
                                                                                    on Traffic data, the improvement of MTGNN in terms of RSE is
5.1       Experimental Setting                                                      significant. MTGNN lowers down RSE by 7.24%, 3.88%, 4.83% over
                                                                                    the horizons of 3, 12, 24 on the traffic data. The main reason why
In Table 1, we summarize statistics of benchmark datasets. More                     MTGNN improves the results of traffic data evidently is that the
details about the datasets is given in Appendix A.2. We use five eval-              nature of traffic data is better suited for our model assumption
uation metrics, including Mean Absolute Error (MAE), Root Mean                      about the spatial-temporal dependencies. Obviously, the future traf-
Squared Error (RMSE), Mean Absolute Percentage Error (MAPE),                        fic occupancy rate of a road not only depends on its past but also
Root Relative Squared Error (RRSE), and Empirical Correlation Co-                   on its connected roads’ occupancy rates. MTGNN fails to make im-
efficient (CORR). For RMSE, MAE, MAPE, and RRSE, lower values                       provements on the exchange-rate data, possibly due to the smaller
are better. For CORR, higher values are better. Other experimental                  graph size and fewer training examples of exchange-rate data.
setups are given in Appendix A.3.
                                                                                    5.3.2 Multi-step forecasting. In this experiment, we compare MT-
5.2       Baseline Methods for Comparision                                          GNN with other spatial-temporal graph neural network models.
MTGNN and MTGNN+sampling are our models to be evaluated.                            Table 3 shows the experimental results for the task of multi-step
MTGNN is our proposed model. MTGNN+sampling is our proposed                         forecasting. The significance of MTGNN lies in that it achieves
model trained on a sampled subset of a graph in each iteration.                     on-par performance with state-of-the-art spatial-temporal graph
Baseline methods are summarized in the following:                                   neural networks without using a pre-defined graph, while DCRNN,
                                                                                    STGCN, and MRA-BGCN fully rely on pre-defined graphs. Graph
5.2.1     Single-step forecasting.                                                  Wavenet proposes a self-adaptive adjacency matrix, but it needs to
        • AR: An auto-regressive model.                                             combine with a pre-defined graph in order to achieve optimal per-
        • VAR-MLP: A hybrid model of the multilayer perception                      formance. ST-MetaNet employs attention mechanisms to adjust the
          (MLP) and auto-regressive model (VAR) [25].                               edge weights of a pre-defined graph. GMAN leverages node2vec al-
        • GP: A Gaussian Process time series model [6, 16].                         gorithm to preserve node structural information while performing
        • RNN-GRU: A recurrent neural network with fully connected                  attention mechanisms. When a graph is not defined, these methods
          GRU hidden units.                                                         cannot model multivariate times series data efficiently.
        • LSTNet: A deep neural network, which combines convolu-
          tional neural networks and recurrent neural networks [12].                5.4     Ablation Study
        • TPA-LSTM: An attention-recurrent neural network [19].                     We conduct an ablation study on the METR-LA data to validate the
5.2.2     Multi-step forecasting.                                                   effectiveness of key components that contribute to the improved
        • DCRNN: A diffusion convolutional recurrent neural network,                outcomes of our proposed model. We name MTGNN without dif-
          which combines diffusion graph convolutions with recurrent                ferent components as follows:
          neural networks [13].                                                           • w/o GC: MTGNN without the graph convolution module.
        • STGCN: A spatial-temporal graph convolutional network,                            We replace the graph convolution module with a linear layer.
          which incorporates graph convolutions with 1D convolu-                          • w/o Mix-hop: MTGNN without the information selection
          tions [23].                                                                       step in the mix-hop propagation layer. We pass the outputs of
        • Graph WaveNet: A spatial-temporal graph convolutional                             the information propagation step to the next module directly.
          network, which integrates diffusion graph convolutions with                     • w/o Inception: MTGNN without inception in the dilated
          1D dilated convolutions [21].                                                     inception layer. While keeping the same number of output
        • ST-MetaNet: A sequence-to-sequence architecture, which                            channels, we use a single 1 × 7 filter only.
          employs meta networks to generate parameters [15].                              • w/o CL: MTGNN without curriculum learning. We train
        • GMAN: A graph multi-attention network with spatial and                            MTGNN without gradually increasing the prediction length.
          temporal attentions [26].                                                    We repeat each experiment 10 times with 50 epochs per repeti-
        • MRA-BGCN: A multi-range attentive bicomponent GCN [3].                    tion and report the average of MAE, RMSE, MAPE with a standard
                                                                                    deviation over 10 runs on the validation set in Table 4. The in-
5.3       Main Results                                                              troduction of graph convolution modules significantly improves
Table 2 and Table 3 provide the main experimental results of MT-                    the results as it enables information flow among isolated but in-
GNN and MTGNN+sampling. We observe that MTGNN achieves                              terdependent nodes. The effect of mix-hop is evident as well: it
state-of-the-art results on most of the tasks, and the performance                  validates that the use of mix-hop is helpful for selecting useful
                     Table 2: Baseline comparison under single-step forecasting for multivariate time series methods.


           Dataset                                Solar-Energy                                            Traffic                                Electricity                                 Exchange-Rate


                                                       Horizon                                            Horizon                                    Horizon                                       Horizon

 Methods               Metrics           3         6          12             24            3          6             12         24       3        6           12           24        3          6            12        24

 AR                    RSE            0.2435     0.3790     0.5911      0.8699           0.5991   0.6218       0.6252       0.63      0.0995   0.1035      0.1050       0.1054   0.0228      0.0279       0.0353    0.0445
                       CORR           0.9710     0.9263     0.8107      0.5314           0.7752   0.7568       0.7544      0.7519     0.8845   0.8632      0.8591       0.8595   0.9734      0.9656       0.9526    0.9357

 VARMLP                RSE            0.1922     0.2679     0.4244      0.6841           0.5582   0.6579       0.6023      0.6146     0.1393   0.1620      0.1557       0.1274   0.0265      0.0394       0.0407    0.0578
                       CORR           0.9829     0.9655     0.9058      0.7149           0.8245   0.7695       0.7929      0.7891     0.8708   0.8389      0.8192       0.8679   0.8609      0.8725       0.8280    0.7675

 GP                    RSE            0.2259     0.3286     0.5200      0.7973           0.6082   0.6772       0.6406      0.5995     0.1500   0.1907      0.1621       0.1273   0.0239      0.0272       0.0394    0.0580
                       CORR           0.9751     0.9448     0.8518      0.5971           0.7831   0.7406       0.7671      0.7909     0.8670   0.8334      0.8394       0.8818   0.8713      0.8193       0.8484    0.8278

 RNN-GRU               RSE            0.1932     0.2628     0.4163      0.4852           0.5358   0.5522       0.5562      0.5633     0.1102   0.1144      0.1183       0.1295   0.0192     0.0264        0.0408    0.0626
                       CORR           0.9823     0.9675     0.9150      0.8823           0.8511   0.8405       0.8345      0.8300     0.8597   0.8623      0.8472       0.8651   0.9786     0.9712        0.9531    0.9223

 LSTNet-skip           RSE            0.1843     0.2559     0.3254      0.4643           0.4777   0.4893       0.4950      0.4973     0.0864   0.0931      0.1007       0.1007   0.0226      0.0280       0.0356    0.0449
                       CORR           0.9843     0.9690     0.9467      0.8870           0.8721   0.8690       0.8614      0.8588     0.9283   0.9135      0.9077       0.9119   0.9735      0.9658       0.9511    0.9354

 TPA-LSTM              RSE            0.1803     0.2347     0.3234      0.4389           0.4487   0.4658       0.4641      0.4765     0.0823   0.0916      0.0964       0.1006   0.0174     0.0241        0.0341    0.0444
                       CORR           0.9850     0.9742     0.9487      0.9081           0.8812   0.8717       0.8717      0.8629     0.9439   0.9337      0.9250       0.9133   0.9790     0.9709        0.9564    0.9381

 MTGNN                 RSE         0.1778        0.2348    0.3109       0.4270           0.4162   0.4754      0.4461       0.4535     0.0745   0.0878     0.0916        0.0953   0.0194      0.0259       0.0349    0.0456
                       CORR        0.9852        0.9726    0.9509       0.9031           0.8963   0.8667      0.8794       0.8810     0.9474   0.9316     0.9278        0.9234   0.9786      0.9708       0.9551    0.9372

 MTGNN+sampling        RSE            0.1875     0.2521     0.3347      0.4386           0.4170   0.4435       0.4469      0.4537     0.0762   0.0862      0.0938       0.0976   0.0212      0.0271       0.0350    0.0454
                       CORR           0.9834     0.9687     0.9440      0.8990           0.8960   0.8815       0.8793      0.8758     0.9467   0.9354      0.9261       0.9219   0.9788      0.9704       0.9574    0.9382



Table 3: Baseline comparison under multi-step forecasting                                                                                             Table 4: Ablation study.
for spatial-temporal graph neural networks.
                                                                                                                            Methods   MTGNN             w/o GC           w/o Mix-hop      w/o Inception    w/o CL

                             Horizon 3                   Horizon 6                       Horizon 12                         MAE       2.7715±0.0119     2.8953±0.0054    2.7975±0.0089    2.7772±0.0100    2.7828±0.0105
                     MAE      RMSE      MAPE      MAE     RMSE     MAPE       MAE         RMSE    MAPE                      RMSE      5.8070±0.0512     6.1276±0.0339    5.8549±0.0474    5.8251±0.0429    5.8248±0.0366
 METR-LA                                                                                                                    MAPE      0.0778±0.0009     0.0831±0.0009    0.0779±0.0009    0.0778±0.0010    0.0784±0.0009
 DCRNN                2.77     5.38      7.30%    3.15     6.45      8.80%        3.60     7.60   10.50%
 STGCN                2.88     5.74      7.62%    3.47     7.24      9.57%        4.59     9.40   12.70%
 Graph WaveNet        2.69     5.15      6.90%    3.07     6.22      8.37%        3.53     7.37   10.01%
 ST-MetaNet           2.69     5.17      6.91%    3.10     6.28      8.57%        3.59     7.52   10.63%                 the easiest task, and fine-tune parameters step by step as the level
 MRA-BGCN             2.67     5.12      6.80%    3.06     6.17      8.30%        3.49     7.30   10.00%                 of learning difficulty increases.
 GMAN                 2.77     5.48      7.25%    3.07     6.34      8.35%        3.40     7.21    9.72%
 MTGNN                2.69     5.18     6.86%     3.05     6.17    8.19%          3.49     7.23    9.87%
 MTGNN+sampling       2.76     5.34     5.18%     3.11     6.32    8.47%          3.54     7.38   10.05%                 5.5        Study of the Graph Learning Layer
 PEMS-BAY                                                                                                                To validate the effectiveness of our proposed graph learning layer,
 DCRNN                1.38     2.95     2.90%     1.74     3.97    3.90%          2.07     4.74   4.90%
 STGCN                1.36     2.96     2.90%     1.81     4.27    4.17%          2.49     5.69   5.79%                  we conduct a study which experiments with different ways of con-
 Graph WaveNet        1.30     2.74     2.73%     1.63     3.70    3.67%          1.95     4.52   4.63%                  structing a graph adjacency matrix. Table 5 shows different forms
 ST-MetaNet           1.36     2.90     2.82%     1.76     4.02    4.00%          2.20     5.06   5.45%
 MRA-BGCN             1.29     2.72     2.90%     1.61     3.67    3.80%          1.91     4.46   4.60%
                                                                                                                         of A with experimental results tested on the validation set of the
 GMAN                 1.34     2.82     2.81%     1.62     3.72    3.63%          1.86     4.32   4.31%                  METR-LA data averaged on 10 runs. Predefined-A is constructed by
 MTGNN                1.32     2.79      2.77%    1.65     3.74      3.69%        1.94     4.49    4.53%                 road network distance [13]. Global-A assumes the adjacency matrix
 MTGNN+sampling       1.34     2.83      2.83%    1.67     3.79      3.78%        1.95     4.49    4.62%
                                                                                                                         is a parameter matrix, which contains N 2 parameters. Motivated
                                                                                                                         by [21], Undirected-A and Directed-A are computed by the similar-
                                                                                                                         ity scores of node embeddings. Motivated by [8, 18], Dynamic-A
                                                                                                                         assumes the spatial dependency at each time step is dependent on
information at each information propagation step in the mix-hop                                                          its node inputs. Uni-directed-A is our proposed method. According
propagation layer. The effect of inception is significant in terms of                                                    to Table 5, our proposed uni-directed-A achieves the lowest mean
RMSE, but marginal in terms of MAE. This is because using a single                                                       MAE, RMSE, and MAPE. It improves over predefined-A, undirected-
1 × 7 filter has half more parameters than using a combination of                                                        A, and dynamic-A significantly. Our uni-directed-A improves over
1 × 2, 1 × 3, 1 × 5, 1 × 7 filters under the condition that the number                                                   undirected-A and directed-A marginally in terms of MAE and MAPE
of output channels for the dilated inception layer remains the same.                                                     but proves to be more robust due to a lower RMSE.
Lastly, our curriculum learning strategy proves to be effective. It                                                          We further investigate the learned graph adjacency matrix via a
enables our model to converge quickly to an optimum that fits for                                                        case study. In Figure 6a, we plot the raw time series of node 55 and
                70                                                                                           70
                                                                                                                                                                                             ACKNOWLEDGMENTS
                65                                                                                           65
                                                                                                                                                                                             This work was supported in part by the Australian Research Council
traffic speed                                                                                traffic speed
                                                                               variable                                                                                         variable
                                                                                  node55
                                                                                  node38
                                                                                  node112
                                                                                                                                                                                   node55
                                                                                                                                                                                   node146
                                                                                                                                                                                   node25
                                                                                                                                                                                             (ARC) under Grant LP160100630, LP180100654 and DE190100626.
                                                                                  node43                                                                                           node37
                60                                                                                           60




                55
                      12:00           13:00
                                              2012−03−04
                                                           14:00       15:00
                                                                                                             55
                                                                                                                   12:00          13:00
                                                                                                                                          2012−03−04
                                                                                                                                                       14:00            15:00
                                                                                                                                                                                             REFERENCES
                                                                                                                                                                                              [1] George EP Box, Gwilym M Jenkins, Gregory C Reinsel, and Greta M Ljung. 2015.
(a) Time series of node 55 and                                                              (b) Time series of node 55 and                                                                        Time series analysis: forecasting and control. John Wiley & Sons.
its top-3 neighbors given by the                                                            its top-3 neighbors given by the                                                                  [2] Fengwen Chen, Shirui Pan, Jing Jiang, Huan Huo, and Guodong Long. 2019.
                                                                                                                                                                                                  DAGCN: Dual Attention Graph Convolutional Networks. In Proc. of IJCNN.
pre-defined A.                                                                              learned A.
                                                                                                                                                                                              [3] Weiqi Chen, Ling Chen, Yu Xie, Wei Cao, Yusong Gao, and Xiaojie Feng. 2019.
                                                                                                                                                                                                  Multi-Range Attentive Bicomponent Graph Convolutional Network for Traffic
                                                                                                                                                                                                  Forecasting. In Proc. of AAAI.
                                                                                                                    38                                           25                           [4] Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh.
                                                                                                                                   112                                                            2019. Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph
                                                                                                                                                                                                  Convolutional Networks. In Proc. of KDD.
                                                                                      146                                  55                                                                 [5] Roger Frigola. 2015. Bayesian time series learning with Gaussian processes. Ph.D.
                               37                                                                                                              43                                                 Dissertation. University of Cambridge.
                                                                                                                                                                                              [6] Roger Frigola-Alcalde. 2016. Bayesian time series learning with Gaussian processes.
                                                                                                                                                                                                  Ph.D. Dissertation. University of Cambridge.
                                                                                                                                                                                              [7] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E
                                                                                                                                                                                                  Dahl. 2017. Neural message passing for quantum chemistry. In Proc. of ICML.
                                                                                                                                                                                                  1263–1272.
                     (c) Node locations of node 55 and its neighbors marked on                                                                                                                [8] Shengnan Guo, Youfang Lin, Ning Feng, Chao Song, and Huaiyu Wan. 2019.
                     Google Maps. Yellow nodes represent node 55’s top-3 neighbors                                                                                                                Attention based spatial-temporal graph convolutional networks for traffic flow
                     given by the pre-defined A. Green nodes represent node 55’s top-                                                                                                             forecasting. In Proc. of AAAI, Vol. 33. 922–929.
                     3 neighbors given by the learned A.                                                                                                                                      [9] Amol Kapoor, Aram Galstyan, Bryan Perozzi, Greg Ver Steeg, Hrayr Harutyunyan,
                                                                                                                                                                                                  Kristina Lerman, Nazanin Alipourfard, and Sami Abu-El-Haija. 2019. MixHop:
                                                                                                                                                                                                  Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood
                                                                   Figure 6: Case study                                                                                                           Mixing. In Proc. of ICML.
                                                                                                                                                                                             [10] Thomas N Kipf and Max Welling. 2017. Semi-supervised classification with graph
                                                                                                                                                                                                  convolutional networks. In Proc. of ICLR.
          Table 5: Comparison of different graph learning methods.                                                                                                                           [11] Johannes Klicpera, Aleksandar Bojchevski, and Stephan Günnemann. 2019. Pre-
                                                                                                                                                                                                  dict then propagate: Graph neural networks meet personalized pagerank. In Proc.
                                                                                                                                                                                                  of ICLR.
                      Methods                       Equation                                                      MAE               RMSE                       MAPE
                                                                                                                                                                                             [12] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. 2018. Modeling
                      Pre-defined-A                 -                                                             2.9017±0.0078     6.1288±0.0345              0.0836±0.0009                      long-and short-term temporal patterns with deep neural networks. In Proc. of
                      Global-A                      A = ReLU (W)                                                  2.8457±0.0107     5.9900±0.0390              0.0805±0.0009                      SIGIR. ACM, 95–104.
                      Undirected-A                  A = ReLU (tanh(α(M1 MT1 )))                                   2.7736±0.0185     5.8411±0.0523              0.0783±0.0012                 [13] Yaguang Li, Rose Yu, Cyrus Shahabi, and Yan Liu. 2018. Diffusion convolutional
                      Directed-A                    A = ReLU (tanh(α(M1 MT2 )))                                   2.7758±0.0088     5.8217±0.0451              0.0783±0.0006                      recurrent neural network: Data-driven traffic forecasting. In Proc. of ICLR.
                      Dynamic-A                     At = So f tMax(tanh(Xt W1 )tanh(WT2 XTt ))                    2.8124±0.0102     5.9189±0.0281              0.0794±0.0008                 [14] Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol
                      Uni-directed-A (ours)         A = ReLU (tanh(α(M1 MT2 − M2 MT1 )))                          2.7715±0.0119     5.8070±0.0512              0.0778±0.0009                      Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu.
                                                                                                                                                                                                  2016. Wavenet: A generative model for raw audio. arXiv:1609.03499 (2016).
                                                                                                                                                                                             [15] Zheyi Pan, Yuxuan Liang, Weifeng Wang, Yong Yu, Yu Zheng, and Junbo Zhang.
its pre-defined top-3 neighbors. In Figure 6b, we chart the raw time                                                                                                                              2019. Urban Traffic Prediction from Spatio-Temporal Data Using Deep Meta
                                                                                                                                                                                                  Learning. In KDD. ACM, 1720–1730.
series of node 55 and its learned top-3 neighbors. Figure 6c shows                                                                                                                           [16] Stephen Roberts, Michael Osborne, Mark Ebden, Steven Reece, Neale Gibson,
the geo-location of these nodes, with green nodes representing the                                                                                                                                and Suzanne Aigrain. 2013. Gaussian processes for time-series modelling. Philos.
                                                                                                                                                                                                  Trans. R. Soc. A 371, 1984 (2013), 20110550.
central node’s learned top-3 neighbors and yellow nodes represent-                                                                                                                           [17] Youngjoo Seo, Michaël Defferrard, Pierre Vandergheynst, and Xavier Bresson.
ing the central node’s pre-defined top-3 neighbors. We observe that                                                                                                                               2018. Structured sequence modeling with graph convolutional recurrent net-
the central node’s pre-defined top-3 neighbors are much closer to                                                                                                                                 works. In Proc. of NIPS. Springer, 362–373.
                                                                                                                                                                                             [18] Lei Shi, Yifan Zhang, Jian Cheng, and Hanqing Lu. 2019. Two-Stream Adaptive
the node itself on the map. As a result, their time series are more                                                                                                                               Graph Convolutional Networks for Skeleton-Based Action Recognition. In Proc.
correlated simultaneously, as shown by the red circles in Figure 6a.                                                                                                                              of CVPR. 12026–12035.
On the contrary, the central node’s learned top-3 neighbors distrib-                                                                                                                         [19] Shun-Yao Shih, Fan-Keng Sun, and Hung-yi Lee. 2019. Temporal pattern attention
                                                                                                                                                                                                  for multivariate time series forecasting. Machine Learning 108, 8-9 (2019), 1421–
ute further away from it but still lie on the same road it follows.                                                                                                                               1441.
According to Figure 6b, time series of the learned top-3 neighbors                                                                                                                           [20] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir
                                                                                                                                                                                                  Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. 2015.
are more capable of indicating extreme traffic conditions of the                                                                                                                                  Going deeper with convolutions. In Proc. of CVPR. 1–9.
central node in advance.                                                                                                                                                                     [21] Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, and Chengqi Zhang. 2019.
                                                                                                                                                                                                  Graph WaveNet for Deep Spatial-Temporal Graph Modeling. In Proc. of IJCAI.
                                                                                                                                                                                             [22] Sijie Yan, Yuanjun Xiong, and Dahua Lin. 2018. Spatial temporal graph con-
6                             CONCLUSIONS                                                                                                                                                         volutional networks for skeleton-based action recognition. In Proc. of AAAI.
In this paper, we introduce a novel framework for multivariate                                                                                                                                    3482–3489.
                                                                                                                                                                                             [23] Bing Yu, Haoteng Yin, and Zhanxing Zhu. 2018. Spatio-Temporal Graph Convo-
time series forecasting. To the best of our knowledge, we are the                                                                                                                                 lutional Networks: A Deep Learning Framework for Traffic Forecasting. In Proc.
first to address the multivariate time series forecasting problem                                                                                                                                 of IJCAI. 3634–3640.
via a graph-based deep learning approach. We propose an effective                                                                                                                            [24] Fisher Yu and Vladlen Koltun. 2016. Multi-scale context aggregation by dilated
                                                                                                                                                                                                  convolutions. In ICLR.
method to exploit the inherent dependency relationships among                                                                                                                                [25] G Peter Zhang. 2003. Time series forecasting using a hybrid ARIMA and neural
multiple time series. Our method demonstrates superb performance                                                                                                                                  network model. Neurocomputing 50 (2003), 159–175.
                                                                                                                                                                                             [26] Chuanpan Zheng, Xiaoliang Fan, Cheng Wang, and Jianzhong Qi. 2020. GMAN:
in a variety of multivariate time series forecasting tasks and opens                                                                                                                              A Graph Multi-Attention Network for Traffic Prediction. In Proc. of AAAI.
a new door to use GNNs to handle diverse non-structural data.
A       APPENDIX: REPRODUCIBILITY                                           • Exchange-Rate: the exchange-rate dataset contains the daily
In this section, we provide the details of our implementation for             exchange rates of eight foreign countries including Australia,
reproducibility. Our source codes 1 are publicly available.                   British, Canada, Switzerland, China, Japan, New Zealand,
                                                                              and Singapore ranging from 1990 to 2016.
A.1      Complexity Analysis                                            Following [12], we split these four datasets into a training set (60%),
We analyze the time complexity of the main components of the            validation set (20%), and test set (20%) in chronological order. The
proposed model MTGNN, which is summarized in Table 6. The               input sequence length is 168 and the output sequence length is 1.
time complexity of the graph learning layer is (O(Ns 1s 2 + N 2s 2 ))   Models are trained independently to predict the target future step
where N denotes the number of nodes, s 1 represents the dimension       (horizon) 3, 6, 12, and 24.
of node input feature vectors, and s 2 represents the dimension of
node hidden feature vectors. Treating s 1 and s 2 as constants, the     A.2.2   Multi-step forecasting.
time complexity of the graph learning layer becomes O(N 2 ). It is          • METR-LA: the METR-LA dataset from the Los Angeles Met-
attributed to the pairwise computation of node hidden feature vec-            ropolitan Transportation Authority contains average traffic
tors. The graph convolution module incurs O(K(Md 1 + Nd 1d 2 ) time           speed measured by 207 loop detectors on the highways of
complexity, where K is the propagation depth, N is the number of              Los Angeles County ranging from Mar 2012 to Jun 2012.
nodes, d 1 denotes the input dimension of node states, d 2 denotes          • PEMS-BAY: the PEMS-BAY dataset from California Trans-
the output dimension of node states. Regarding K, d 1 and d 2 as              portation Agencies (CalTrans) contains average traffic speed
constants, the time complexity of the graph convolution module                measured by 325 sensors in the Bay Area ranging from Jan
turns to O(M). This result comes from the fact that in the infor-             2017 to May 2017.
mation propagation step, each node receives information from its
                                                                        Following [13], we split these two datasets into a training set (70%),
neighbors and the sum of the number of neighbors of each node
                                                                        validation set (20%), and test set (10%) in chronological order. The
exactly equals the number of edges. The time complexity of the
                                                                        input sequence length is 12, and the target sequence contains the
temporal convolution module equals to O(Nlc i co /d), where l is the
                                                                        next 12 future steps. The time of the day is used as an auxiliary fea-
input sequence length, c i is the number of input channels, co is the
                                                                        ture for the inputs. For the selected baseline methods, the pairwise
number of output channels, and d is the dilation factor. The time
                                                                        road network distances are used as the pre-defined graph structure.
complexity of the temporal convolution module mainly depends
on N × l, which is the size of the input feature map.
                                                                        A.3     Experimental Setup
                                                                        We repeat the experiment 10 times and report the average value
    Components                            Time Complexity               of evaluation metrics. The model is trained by the Adam optimizer
                                                                        with gradient clip 5. The learning rate is 0.001. The l2 regularization
    Graph Learning Layer                  O(Ns 1s 2 + N 2s 2 )          penalty is 0.0001. Dropout with 0.3 is applied after each temporal
                                                                        convolution module. Layernorm is applied after each graph convo-
    Graph Convolution Module              O(K(Md 1 + Nd 1d 2 )
                                                                        lution module. The depth of the mix-hop propagation layer is set
    Temporal Convolution Module           O(Nlc i co /d)                to 2. The retain ratio from the mix-hop propagation layer is set to
               Table 6: Time Complexity Analysis                        0.05. The saturation rate of the activation function from the graph
                                                                        learning layer is set to 3. The dimension of node embeddings is 40.
                                                                        Other hyper-parameters are reported according to different tasks.

                                                                        A.3.1 Single-step forecasting. We use 5 graph convolution modules
A.2      Data                                                           and 5 temporal convolution modules with the dilation exponential
In Table 1, we summarize statistics of benchmark datasets. Details      factor 2. The starting 1 × 1 convolution has 1 input channel and
of these datasets are introduced below.                                 16 output channels. The graph convolution module and the tempo-
                                                                        ral convolution modules both have 16 output channels. The skip
A.2.1    Single-step forecasting.
                                                                        connection layers all have 32 output channels. The first layer of
     • Traffic: the traffic dataset from the California Department of   the output module has 64 output channels and the second layer of
       Transportation contains road occupancy rates measured by         the output module has 1 output channel. The number of training
       862 sensors in San Francisco Bay area freeways during 2015       epochs is 30. For Traffic, Solar-Energy, and Electricity, the number
       and 2016.                                                        of neighbors for each node is 20. For Exchange-Rate, the number
     • Solar-Energy: the solar-energy dataset from the National Re-     of neighbors for each node is 8. The batch size is set to 4. For the
       newable Energy Laboratory contains the solar power output        MTGNN+sampling model, we split the nodes of a graph into three
       collected from 137 PV plants in Alabama State in 2007.           partitions randomly with a batch size of 16. Following [12], we use
     • Electricity: the electricity dataset from the UCI Machine        RSE and CORR as evaluation metrics.
       Learning Repository contains electricity consumption for
       321 clients from 2012 to 2014.                                   A.3.2 Multi-step forecasting. We use 3 graph convolution modules
                                                                        and 3 temporal convolution modules with the dilation exponential
1 https://github.com/nnzhan/MTGNN                                       factor 1. The starting 1 × 1 convolution has 2 input channels and
      2.82
             ●
                                                                                                 2.95


                                                                                                            ●
                                                                                                                                                                                             A.4    Parameter Study
      2.80
                                                                                                 2.90
                                                                                                                                                                                             We conduct a parameter study on eight core hyper-parameters
MAE
                                                                             Group
                                                                             ●
                                                                                 METR−LA   MAE
                                                                                                 2.85                                                                          Group
                                                                                                                                                                               ●
                                                                                                                                                                                   METR−LA
                                                                                                                                                                                             which influence the model complexity of MTGNN. We list these
                                                                                                                                                                                             hyper-parameters as follows: Number of layers, the number of
                               ●                                                                                ●




      2.78
                                                                                                 2.80
                                                                                                                    ●
                                          ●




                                                                                                                                                                                             temporal convolution modules, ranges from 1 to 6. Number of
                                                                         ●


                                                    ●
                                                                                                                                 ●
                                                               ●


                                                                                                                                                  ●
                                                                                                                                                                           ●

      2.76                                                                                       2.75

                               2
                                              x
                                                    4                   6                               0                                50
                                                                                                                                                      x
                                                                                                                                                                  100                        filters, the number of output channels for temporal convolution
                                                                                                                                                                                             modules and graph convolution modules, ranges from 4 to 128.
                           (a) Number of layers                                                                     (b) Number of filters
                                                                                                                                                                                             Number of neighbors, the parameter k in Equation 5, ranges from
      2.80                                                                                       2.79                                                                                        10 to 60. Saturation rate, the parameter α in Equation 1, 2, and 3,
      2.79
                                                                         ●




                                                                                                 2.78       ●
                                                                                                                                                                           ●



                                                                                                                                                                                             ranges from 0.5 to 5. Retain ratio of mix-hop propagation layer, the
                                                                                                                                                                                             parameter β in Equation 7, ranges from 0 to 0.8. Depth of mix-hop
                                                                                                                                                                   ●

                                                                                                                                                          ●
                                                                                                                        ●


                                                               ●




MAE                                                                                        MAE
                                                    ●
                                                                             Group                                                                                             Group
      2.78                                                                   ●
                                                                                 METR−LA         2.77                                                                          ●
                                                                                                                                                                                   METR−LA


             ●

                               ●
                                                                                                                                     ●
                                                                                                                                                                                             propagation layer, the parameter K in Equation 8, ranges from 1 to
                                          ●
      2.77                                                                                       2.76

                                                                                                                                                                                             6.
      2.76

                               20                   40                  60
                                                                                                 2.75

                                                                                                                        1            2                    3        4       5
                                                                                                                                                                                                 We repeat each experiment 10 times with 50 epochs each time
                                              x                                                                                                       x

                                                                                                                                                                                             and report the average of MAE with a standard deviation over
                 (c) Number of neighbors                                                                                    (d) Saturation rate                                              10 runs on the validation set. We change the parameter under
                                                                                                 2.79
                                                                                                                                                                                             investigation and fix other parameters in each experiment. Figure 7
                                                                    ●




      2.80                                                                                       2.78
                                                                                                            ●
                                                                                                                                                                                             shows the experimental results of our parameter study. As shown
MAE
                                                                             Group
                                                                                           MAE
                                                                                                                             ●

                                                                                                                                                                       ●
                                                                                                                                                                               Group
                                                                                                                                                                                             in Figure 7a and Figure 7b, increasing the number of layers and
                                               ●                             ●
                                                                                 METR−LA                                                                                       ●
                                                                                                                                                                                   METR−LA
                                                                                                 2.77
      2.78        ●
                           ●




                                     ●
                                                                                                                                              ●               ●



                                                                                                                                                                           ●
                                                                                                                                                                                             filters enhances our model’s expressive capacity, while reducing the
                       ●
                                                                                                 2.76                                                                                        MAE loss. Figure 7c shows that a small number of neighbors gives
      2.76

                 0.0                0.2       0.4        0.6       0.8                                                       2                                4            6
                                                                                                                                                                                             better results. It is possibly because a node may only depend on a
                                              x                                                                                                       x
                                                                                                                                                                                             limited number of other nodes, and increasing its neighborhood
                                    (e) Retain ratio                                                            (f) Depth of propagation                                                     merely introduces noises to the model. The model performance is
                                                                                                                                                                                             not sensitive to the saturation rate, as shown in Figure 7d. However,
                                                         Figure 7: Parameter Study                                                                                                           a large saturation rate can impose values of the adjacency matrix
                                                                                                                                                                                             produced by the graph learning layer approach to 0 or 1. As shown
32 output channels. The graph convolution module and the tem-                                                                                                                                in Figure 7e, a high retain ratio degrades the model performance
poral convolution module both have 32 output channels. The skip                                                                                                                              significantly. We think it is because by default the propagation
connection layers all have 64 output channels. The first layer of                                                                                                                            depth of the mix-hop propagation layer is set to 2, and as a result,
the output module has 128 output channels and its second layer                                                                                                                               keeping a high proportion of root information constrains a node
has 12 output channels. The number of neighbors for each node is                                                                                                                             from exploring its neighborhood. Figure 7f shows that it is enough
20. The number of training epochs is 100. The batch size is set to                                                                                                                           to propagate node information with 2 or 3 steps. With the increase
64. Following [13], we use MAE, RMSE , and MAPE as evaluation                                                                                                                                of the depth of propagation, the proposed mix-hop propagation
metrics.                                                                                                                                                                                     layer does not suffer from the over-smoothing problem incurred
                                                                                                                                                                                             by information aggregation. With the depth of propagation equal
                                                                                                                                                                                             to 6, it has the lowest mean MAE with a larger variation.

