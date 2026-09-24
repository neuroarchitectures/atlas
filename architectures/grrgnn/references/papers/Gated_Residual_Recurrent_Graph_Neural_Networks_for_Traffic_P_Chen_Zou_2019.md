# Gated Residual Recurrent Graph Neural Networks for Traffic P Chen Zou 2019

> Source: `Gated_Residual_Recurrent_Graph_Neural_Networks_for_Traffic_P_Chen_Zou_2019.pdf`

---

                                      The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19)




       Gated Residual Recurrent Graph Neural Networks for Traffic Prediction

                        Cen Chen,? Kenli Li,? * Sin G. Teo,† Xiaofeng Zou,? Kang Wang?
                                          Jie Wang,† Zeng Zeng† *
                            ?
                           College of Information Science and Engineering, Hunan University, China
                                          †
                                            Institute for Infocomm Research, Singapore
                   {chencen, lkl, zouxiaofeng, zztwk}@hnu.edu.cn, {teosg, wangjie, zengz}@i2r.a-star.edu.sg


                            Abstract                                                                r1

  Traffic prediction is of great importance to traffic manage-
  ment and public safety, and very challenging as it is af-
  fected by many complex factors, such as spatial dependency
  of complicated road networks and temporal dynamics, and                                           r2                  r3
  many more. The factors make traffic prediction a challenging
  task due to the uncertainty and complexity of traffic states.                   Figure 1: Three roads in the directed road network
  In the literature, many research works have applied deep
  learning methods on traffic prediction problems combining
  convolutional neural networks (CNNs) with recurrent neural
                                                                              roads r2 and r1 are not. Even r2 is closer in the distance
  networks (RNNs), which CNNs are utilized for spatial de-
  pendency and RNNs for temporal dynamics. However, such                      to r1 than r3 , traffic conditions of r2 have greater effect
  combinations cannot capture the connectivity and globality                  on r3 , rather than on r1 .
  of traffic networks. In this paper, we first propose to adopt             • Factor 2: Multiple temporal dependencies. Traffic con-
  residual recurrent graph neural networks (Res-RGNN) that                    ditions usually involve a mixture of the repeating time pat-
  can capture graph-based spatial dependencies and temporal                   terns. For example, the traffic jam of a road occurring at 6
  dynamics jointly. Due to gradient vanishing, RNNs are hard
                                                                              pm will affect the same road’s traffic condition in the fol-
  to capture periodic temporal correlations. Hence, we further
  propose a novel hop scheme into Res-RGNN to utilize the pe-                 lowing hour, i.e., 7 pm. We also utilize the patterns such
  riodic temporal dependencies. Based on Res-RGNN and hop                     as on daily and weekly basis to predict traffic conditions
  Res-RGNN, we finally propose a novel end-to-end multiple                    of the road, e.g., using peak hour patterns for traffic pre-
  Res-RGNNs framework, referred to as “MRes-RGNN”, for                        diction.
  traffic prediction. Experimental results on two traffic datasets          • Factor 3: External factor. Traffic conditions can be sig-
  have demonstrated that the proposed MRes-RGNN outper-
                                                                              nificantly affected by external factors such as vehicle ac-
  forms state-of-the-art methods significantly.
                                                                              cidents, road maintenance, weather conditions, holidays
                                                                              and other special events, and so on.
                        Introduction                                           Many statistical approaches have been widely used in
Traffic prediction is one of the most challenging tasks in                  predicting traffic conditions. In recent years, deep learn-
Intelligent Transportation Systems (ITS) (Jabbarpour et al.                 ing based methods outperform many traditional statistical
2018). This task is important and critical for many trans-                  approaches (e.g., k-nearest neighbours and support vector
portation services, such as vehicle flow control, road trip                 machines) in various prediction tasks, including traffic pre-
planning and navigation and so on. The goal of traffic predic-              diction. Convolutional neural networks (CNN) (Zeng et al.
tion is to predict future traffic states of road networks using             2018) have been proven be good spatial feature extractors.
sequential traffic historical states. Three key complex factors             Many methods (Zhang, Zheng, and Qi 2017; Du et al. 2018;
that can affect traffic conditions are investigated as follows.             Yu et al. 2017; Yao et al. 2018b; Chen et al. 2018) that have
                                                                            combined CNNs with long short-term memory (LSTM) net-
• Factor 1: Spatial dependencies on a directed road net-                    works to capture spatial-temporal features from traffic data
  work. Spatial correlation such as geographic distance and                 have outperformed many traditional machine learning meth-
  connectivity in directed road network can affect traffic                  ods (Hong 2011; Xia et al. 2016).
  prediction performance. For example, given three roads                       However, one of the main limitations of the above com-
  r1 , r2 and r3 in the directed road network as depicted in                bination is that normal convolutional operations can capture
  Figure 1, Roads r2 and r3 are obviously correlated, while                 the spatial features of regular grid structures existing in im-
    * They are corresponding author of this paper.                          ages or videos, rather than the features of general graph
Copyright © 2019, Association for the Advancement of Artificial             forms. The traffic network is a non-Euclidean and direc-
Intelligence (www.aaai.org). All rights reserved.                           tional topology that makes the convolution operation less


                                                                      485
effective. (Li et al. 2018; Yu, Yin, and Zhu 2017) have pro-               a multi-view spatio-temporal network that combines local
posed different networks to fully utilize spatial information              CNN, LSTM and semantic network to predict short-term
on the traffic networks. To the best of our knowledge, (Li                 traffic conditions. Lastly, (Du et al. 2018) proposed a hy-
et al. 2018; Yu, Yin, and Zhu 2017) have combined GCNs                     brid multi-modal deep learning framework based on multi-
and some other neural networks to achieve the state-of-the-                ple CNN-GRU algorithms, which can effectively extract lo-
art results on traffic prediction. However, some limitations               cal spatial features and long dependency features together
are still in these works, as described in the following: (i) It is         with spatio-temporal correlations from the multi-modal traf-
hard to train deep conventional RNNs well, due to the van-                 fic data. However, the discussed approaches cannot capture
ishing gradient and exploding gradient problems (He et al.                 the spatial features of traffic networks.
2016a; 2016b), which can cause serious training issues when                   To overcome the limitations above, (Defferrard, Bresson,
graph convolutions become complicated, (ii) The models are                 and Vandergheynst 2016; Yu, Yin, and Zhu 2017; Li et
less sensitive to large linear scale changes of inputs, due to             al. 2018) proposed different networks in traffic forecasting.
the non-linear nature of the convolutional and recurrent op-               (Defferrard, Bresson, and Vandergheynst 2016) proposed
erations and (iii) They only consider near past traffic condi-             the graph convolutional neural networks (GCN) to capture
tions, without capturing the existing periods and repeating                the non-Euclidean spatial features of traffic data. (Yu, Yin,
patterns.                                                                  and Zhu 2017) proposed a traffic forecasting framework that
   In this paper, we propose a novel framework for traffic                 uses GCN to learn spatio-temporal features of traffic data
prediction, referred to as “MRes-RGNN”, based on multi-                    applicable only to undirected graph. (Li et al. 2018) also
ple residual recurrent graph neural networks. Through rig-                 proposed a traffic forecasting framework to combine both of
orous experiments, we have demonstrated the performance                    the diffusion convolutional and recurrent neural network to-
of MRes-RGNN and proven the advantages of the proposed                     gether in forecasting traffic conditions. These above the two
MRes-RGNN over other state-of-the-art methods in traffic                   frameworks even can capture the spatial dependency of the
prediction. Our contributions can be summarized as follows.                traffic data using bidirectional random walks on the graph,
 • We propose to utilize residual recurrent graph neural net-              and the temporal dependency of the traffic data using the
   works (Res-RGNN) to capture the graph-based spatial de-                 encoder-decoder network. However, they do not use more
   pendencies and temporal dynamics jointly. Res-RGNN                      complex traffic features such as the multiple temporal de-
   can properly extract spatio-temporal features for traffic               pendencies and external factors.
   networks, and make the models more sensitive to unex-
   pected changes.                                                                                Preliminaries
 • We design a novel hop Res-RGNN to improve the perfor-
                                                                           Traffic Prediction on Graphs
   mance of traffic prediction that can discover the periodic
   patterns among the time series signals.                                 Traffic prediction is to perform prediction on future traffic
 • By combining the outputs of Res-RGNN and hop Res-                       states (i.e., traffic speed and traffic flow), as given historical
   RGNN branches with dynamic weights, we propose a                        traffic states from a series of road segments or observation
   novel end-to-end framework, named as MRes-RGNN,                         sensors. In this section, we first define the traffic prediction
   that considers spatial and multiple temporal dependencies               problems in Definition 1.
   with external factors for traffic prediction.                              Definition 1 (Traffic Prediction): Formally, given a se-
                                                                           ries of fully observed time series signals X = x1 , x2 , ..., xt
                       Related Work                                        of all the road segments or sensors, the predicted future sig-
                                                                           nals can be defined as Y = yt+1 . Since roads contain di-
In this section, we discuss some deep learning methods ap-
                                                                           rectional information, traffic networks can be modelled as
plied in traffic forecasting that outperform many traditional
                                                                           directed graphs with structured time-series data.
statistical methods (Hong 2011; Xia et al. 2016). CNN or its
related convolutional-based residual network can extract the                  In the following, we shall present the key definitions that
spatial dependencies of the traffic networks by converting                 are used for directed graph based traffic prediction.
the dynamic traffic data into images (Ma et al. 2017). This                   Definition 2 (Directed Graph for Traffic Network): A
neural network architecture only capture spatial-temporal                  traffic network can be modeled as a weighted directed graph,
information or correlation of the traffic flow data, sepa-                 G = (V, E, W ), where V is a set of nodes, e.g., sensors,
rately. (Zhang et al. 2018; Zhang, Zheng, and Qi 2017;                     that can monitor road segments in the traffic network, E is
Yao et al. 2018b; 2018a) proposed to combine of both RNN                   the connectivity among the nodes, e.g., ε(i,j) = 1 if υi and
and CNN networks that can learn spatio-temporal infor-                     υj are connected, and ε(i,j) = 0 if not, and W ∈ RN ×N is
mation simultaneously to overcome the limitation as dis-                   the weighted adjacency matrix of G representing the nodes’
cussed previously. To further improve accuracy performance                 proximities, e.g., the distance between any pair of nodes, N
in traffic forecasting, (Yu et al. 2017; Yao et al. 2018b;                 is the number of nodes. Specifically, w( i, j) can be defined
Du et al. 2018) proposed different networks as discussed in                as the edge weight from υi to υj .
the following.                                                                Definition 3 (Traffic Prediction on Graphs): Given a
   (Yu et al. 2017) proposed a deep LSTM model using the                   G, let xt ∈ RN ×P be a graph signal observed at time t,
normal traffic hours and a mixture of deep LSTM model us-                  where P is the number of features observed by the node,
ing the incident traffic period. (Yao et al. 2018b) proposed               e.g., vehicle speed, flow, etc. The prediction of yt+1 can be


                                                                     486
formulated as follows:                                                                                                          Time
                                                                              Graph signals
              yt+1 = f [xt−h̃+1 , ..., xt ], G .              (1)                   and
                                                                              external factors                                           
                                                                                                                                                           t   t+1
Convolutions on Graphs
Conventional 2D CNNs that are used well for regular                                                                       
grids, such as images, cannot be applied directly to learn                                   Hop
                                                                                          Res-RGNN                           s
                                                                                                                                              
features from general graphs, as indicated in (Niepert,                        Multiple                    RGNN

                                                                                                                  
                                                                              branches

Ahmed, and Kutzkov 2016; Defferrard, Bresson, and Van-
dergheynst 2016; Bruna et al. 2013). In the literature, two                                          Res-RGNN                       

main approaches have been proposed to generalize CNNs
                                                                                                                  Weighted feature fusion
for graphs. The first one is to transfer the data into spec-
tral domain using graph Fourier transforms (Bruna et al.                                                          Fully connected layer       Loss

2013). Thus, the performance of graph convolution has been
improved significantly, i.e., the computational complexity
is reduced from O(n2 ) to linear (Defferrard, Bresson, and                     Figure 2: Network Architecture of MRes-RGNN.
Vandergheynst 2016; Kipf and Welling 2016). However,
this approach can only extract spatial features of undirected             through time (BPTT) (Werbos 1990; Sutskever, Vinyals, and
graphs. The other one is to extend CNNs to support general                Le 2014).
directed graph-structured data by using diffusion convolu-
tion operation (Atwood and Towsley 2016; Li et al. 2018;                  Graph CNNs for Extracting Spatial Features
Teng 2016), as defined in the following:                                  A traffic network can be modelled as a graph mathemat-
   Definition 4 (Diffusion Convolution): Diffusion convo-                                                               

                                                                          ically. However, in the previous studies (Yu et al. 2017;
                                                                                                                                                     




lution operation over a graph signal x ∈ RN ×P and a filter               Yao et al. 2018b), which utilized CNNs to extract the spa-
Θ is defined as:                                                          tial features without considering spatial attributes of traf-
           K−1
           X                  k                   k
                                                                         fic networks; as a result, the connectivity and globality of
                        −1
  x?Θ=           θk,1 (DO  W ) + θk,2 (DI−1 W T )          x, (2)         the networks were overlooked as roads were split into mul-
           k=0                                                            tiple segments or grids. In the paper, the traffic network
                                                                          is firstly modeled as a directed graph, and then the diffu-
where ? denotes the diffusion convolution, K is the number
                                                                          sion convolution (Atwood and Towsley 2016; Li et al. 2018;
of diffusion steps, θ ∈ RK×2 are the learned parameters
                                                                          Teng 2016) is applied directly on graph-structured data to
of Θ for two directions of the graph; DO = diag(W 1) are
                                                                          extract patterns and features in space domain. Based on the
the out-degree diagonal matrix, and 1 ∈ RN denotes the
                                                                          definition of the convolution operation, we have a new neu-
all one vector; DI = diag(W T ) is the in-degree diagnose;
        −1                                                                ral network layer, diffusion convolutional layer, that is de-
and DO     W , DI−1 W T represent the transition matrices of              fined in Definition 4 as follows.
the diffusion process and the reverse one, respectively.                     Definition 5 (Diffusion Convolutional Layer): Based on
   In the case of the undirected graphs, (Li et al. 2018) has             the diffusion convolution defined in Definition 1, a diffu-
shown that many existing graph structured convolutional op-               sion convolutional layer that maps P -dimensional features
erations, including the spectral graph convolution (Bruna et              to Q-dimensional outputs can be defined as follows. Let
al. 2013), are special cases of diffusion convolution.                    the parameter tensor be Θ ∈ RQ×P ×K×2 = [θ]q,p , where
                                                                          Θq,p,:,: ∈ RK×2 parameterizes the convolutional filter for
       Proposed MRes-RGNN Framework                                       the p-th input and the q-th output. Hence, the diffusion con-
The network architecture of the proposed MRes-RGNN is                     volutional layer is:
shown in Figure 2, where the inputs of MRes-RGNN are                                           P
                                                                                                                 !
the historical traffic states and external features, while the
                                                                                               X
                                                                                    H:,q = σ      x:,p ? Θq,p,:,: , q ∈ 1, ..., Q,  (3)
outputs are the predictions on the future traffic states. Mul-                                       p=1
tiple temporal dependencies contain near and periodic, e.g.,
daily, weekly, and monthly, dependencies. Our framework                   where x ∈ RN ×P is the input, H ∈ RN ×Q is the output,
consists of several MRES-RGNN modules with different                      Θq,p,:,: are the filters and σ is an activation function (e.g.,
hops. Obviously, Res-RGNN is a special case of MRES-                      ReLU and Sigmoid functions).
RGNN with hop 1. The Res-RGNN branch is utilized to                          This diffusion convolutional layer can be used to learn
learn spatial information on graphs with the near depen-                  the representations of the graph structured data, which can
dency together, and multiple hop Res-RGNN branches are                    be trained by using stochastic gradient descent (SGD).
for spatial information with multiple periodic dependencies
respectively. The extracted spatial and multiple temporal                 Residual Recurrent Graph Neural Networks
features are further fused with a weighted scheme and then,               (Res-RGNN) for Extracting Spatio-Temporal
integrated by an output layer to achieve a final prediction.              Features
The entire end-to-end framework is trained by maximiz-                    In MRes-RGNN, we utilize graph convolutional, recurrent,
ing the likelihood of a graph signal using back-propagation               and residual neural networks together to extract the spatial-


                                                                    487
                          s_i                                 s_t
              RGNN                 RGNN     ...       RGNN          FC   y_t+1
                                                                                          RGNN              RGNN                RGNN      J            RGNN

     concat               concat             concat                              concat            concat              concat                 concat
              e_i   x_i         e_j   x_j         e_t   x_t                            e_i   x_i        e_j   x_j           e_i   x_i              e_j   x_j
                                      (a)                                                      (b)                                      (c)

Figure 3: (a) Recurrent graph neural networks (RGNN) which combine diffusion convolution operation (a type of graph convo-
lution operation); (b) RGNN with residual shortcuts (Res-RGNN); (c) RGNN with gated residual learning (Gated Res-RGNN).


temporal features with the external features under consider-                                 With the benefits of (He et al. 2016a; 2016b), we intro-
ation.                                                                                    duce the residual error into RGNN. A RGNN cell is consid-
                                                                                          ered as a computation block where the residual information
RGNN + External Features Gated Recurrent Unit (GRU)
                                                                                          is passed by shortcut of the blocks so as to speed up a con-
is a simple, yet powerful variant of RNNs for time series
                                                                                          vergence rate, as illustrated in Figure 3(b). Hence, the hid-
prediction due to the gating mechanism (Chung et al. 2014).
                                                                                          den state of t after adding the residual shortcut path can be
GRUs are carefully designed to memorize historical infor-
                                                                                          formulated as follows:
mation, e.g., long-term dependencies. Inspired by (Chung
et al. 2014), we combine graph convolution and GRU to-                                       s̃t = φ(s̃t−1 , xt , et , Θr , Θu , Θc ) + Wlinear s̃t−1 ,        (5)
gether, denoted as RGNN, to discover the spatial-temporal
dynamics jointly as depicted in Figure 3(a). The matrix mul-                              where φ denotes the function of GRU cell, s̃t denotes the
tiplications in GRU can be replaced with the graph convolu-                               state of t after the residual shortcut path added and Wlinear
tions that is similar to the methods used in (Seo et al. 2016;                            is the linear projection weight.
Li et al. 2018). This method combines the traditional fea-                                   Equation 5 is composed of a shortcut connection and an
ture engineering methods with deep learning based meth-                                   element-wise addition. Importantly, the shortcut connection
ods. The external factor attributes are first embedded and                                has not added extra parameters and increased computation
concatenated with graph signals and then, used as the input                               complexity. This reconstruction can make the loss function
of RGNN model as shown in Figure 3. Thus, the process of                                  approximate to an identity mapping. Thus, the training er-
a RGNN unit at time t can be formulated as:                                               rors are not increasing since the recurrent connections are
                     rt = σ(Θr ? [xt , et , st−1 ] + br ),                                formulated as the identity mapping.
                                                                                             It is obvious that the previous hidden states of RGNN
                    ut = σ(Θu ? [xt , et , st−1 ] + bu ),                                 units are added to the hidden states of the next RGNN units
          ct = tanh(Θc ? [xt , et , (rt st−1 )] + bc ),      (4)                          without non-linear activation in Equation 5. This linear ad-
                     st = ut st−1 + (1 − ut ) ct ,                                        ditive nature makes RGNN more sensitive and robust to sud-
                                                                                          den changes in traffic historical states.
                                        yt+1 = Wo st ,
where xt , et and st are the graph signal, external feature and                           Gated Res-RGNN Motivated by the gate function in GRU
the hidden state output at time t, rt and ut denote the reset                             (Chung et al. 2014), we also add a gate function to control
gate and update gate at the time t, respectively, ? denotes                               the flow of residual information as depicted in Figure 3(c) in
the graph convolution, Θr , Θu and Θc are the learned pa-                                 the proposed MRes-RGNN. This gating mechanism helps to
rameters of filters, yt+1 is the output at t + 1 and lastly Wo                            make decision on the threshold that how much the previous
denotes the learned parameters of the output layer.                                       residual information can affect next time segments. The hid-
   RGNN unit is similar to the diffusion convolutional layer.                             den state at time t with the gated residual shortcut path can
The RGNN unit can be used to build recurrent neural net-                                  be formulated as follows:
work layers and also trained by using BPTT. Like the deep                                    s̃t = gφ(s̃t−1 , xt , et , Θr , Θu , Θc ) + Wlinear s̃t−1 ,
RNN methods in (Pascanu et al. 2013; Du, Wang, and Wang                                                                                                        (6)
                                                                                                                         g = σWg [xt , et ] + Ug st−1 ,
2015), hierarchical RGNN layers can be stacked to form the
deep RGNN model.                                                                          where Wg is the linear projection weight, Ug is the state-to-
                                                                                          state weight matrix and σ is the activation function.
Res-RGNN Residual neural network is used with a refer-
ence to the direct hidden state, instead of an unreferenced                               Hop Res-RGNN for Very Long-term Dependencies
function. In (He et al. 2016a; 2016b), the authors introduced                             One of the common ways to predict near future traffic con-
the residual-shortcut structures to build deep networks with                              dition is to use latest time segments. Besides, the periodic
a depth of 152 layers. The networks have achieved good per-                               patterns can also be utilized to improve the traffic prediction
formance on COCO and ImageNet object detection datasets                                   performance. In Figure 4(a), we describe the speed value
(He et al. 2016a). The residual learning and linear shortcut                              during each time interval in 7 days plotting by a sensor
connections can help to solve the exploding and vanishing                                 in METR-LA dataset. The plotting curves show a certain
gradient problems in the long-term back-propagation (He                                   repeatability pattern. Therefore, we can easily identify the
et al. 2016a; 2016b), especially in the networks with deep                                daily patterns in the given dataset. Moreover, we plot Figure
structures.                                                                               4(b) to illustrate the weekly periodic patterns of the traffic


                                                                                 488
              (a)                                (b)                                          (a)                            (b)

Figure 4: Periodic temporal dependencies for METR-LA.                           Figure 5: (a) Sensor networks of PEMS-BAY, each dot de-
(a) daily; (b) weekly.                                                          notes a sensor station. (b) Heat map of weighted adjacency
                                                                                matrix.

data. The two figures have clearly shown that the temporal
dependencies of periods give significant impacts on the traf-                   formulated as follows:
fic state; e.g., near time, daily, and weekly periodic patterns                                                  γ
regardless of the degrees of influences which are not com-                                     sf = Wc ⊗ sc +
                                                                                                                 X
                                                                                                                       Wi ⊗ si ,          (8)
pletely the same.
                                                                                                                 i=1
   As the recurrent layers with GRU and LSTM units are
carefully designed to memorize the historical information,                      where sf denotes the fused hidden state features, ⊗ is
relatively long-term dependencies is aware by the layers.                       Hadamard product (i.e., element-wise multiplication for ten-
Both GRU and LSTM cannot capture very long-term (daily,                         sors), sc are the hidden state features of the Res-RGNN
weekly) correlation in practice due to the notorious gradi-                     branch, γ denotes the number of hop Res-RGNN branches,
ent vanishing problem (He et al. 2016a). To solve this issue,                   si means the hidden state features of the hop Res-RGNN
we propose a novel hop scheme in our MRes-RGNN frame-                           branch with the index i, and lastly Wc , W1 , ..., Wγ are the
work which leverages the periodic patterns in traffic data.                     learnable parameters that adjust the degrees affected by dif-
Specifically, in our model, hop-links are added to connect                      ferent branches.
the current RGNN cell and the RGNN cells in a same phase                           Once the features of the two branches are fused by the
of adjacent periods. Suppose we want to predict the traffic                     proposed weighted strategy, the merged features sf can be
state at t+1, the updating process formula of the last RGNN                     used as the input of a dense layer to obtain the output of the
cell for a specific hop Res-RGNN branch without any gate                        MRes-RGNN framework.
can be formulated as follows:
  rt+1−β = σ(Θr ? [x(t+1)−β , e(t+1)−β , s̃(t+1)−β×2 ] + br ),                                        Experiments
  ut+1−β = σ(Θu ? [x(t+1)−β , e(t+1)−β , s̃(t+1)−β×2 ] + bu ),                  In this section, we evaluate the performance of our proposed
  ct+1−β = tanh(Θc ? [x(t+1)−β , e(t+1)−β , (rt        s̃(t+1)−β×2 )]           MRes-RGNN compared with other existing methods in traf-
           + bc ),                                                              fic prediction.
  s̃t+1−β = ut s(t+1)−β×2 + (1 − ut )       ct
                                     + Wlinear s̃(t+1)−β×2 ,                    Experiment Settings
                                                                    (7)         We configure a Linux server and the other configurations
where β is the number of hidden cells skipped through and                       to run the experiment as follows: 8 Intel(R) Xeon(R) CPU
   is the element-wise multiplication for tensors. β can be                     E5-2680 v4 @ 2.40GHZ; 256GB RAM; 4 NVIDIA P100
calculated by periods (daily or weekly).                                        GPUs. Two large real-world datasets, i.e., METR-LA and
                                                                                PEMS-BAY datasets, are used in the experiment:
Weighted Fusion for Multiple Res-RGNN Branches
                                                                                • METR-LA: Traffic data are collected from observation
As discussed previously, traffic states are affected by mul-                      sensors in the highway of Los Angeles County. We use
tiple temporal dependencies (e.g., near, daily and weekly).                       207 sensors and 4 months of data dated from 1st Mar 2012
The degrees of influence on the traffic states may be dif-                        until 30th Jun 2012 in the experiment.
ferent. In our MRes-RGNN framework, a branch of Res-
                                                                                • PEMS-BAY: Traffic data are collected by California
RGNN targets at near dependency, while multiple hop Res-
                                                                                  Transportation Agencies Performance Measurement Sys-
RGNNs target at multiple periodic (daily and weekly) depen-
                                                                                  tem (PeMS). We use 325 sensors in the Bay Area and 6
dencies. Inspired by the observations, we propose a novel
                                                                                  months of data dated from 1st Jan 2017 until 31th May
parametric-tensor-based fusion method that can fuse multi-
                                                                                  2017 in the experiment. The distribution of the sensors
ple branches in the framework as well.
                                                                                  and the road network are shown in Figure 5(a).
   The fusion method of the hidden state at time t can be


                                                                          489
Table 1: Baseline comparison on METR-LA and PEMS-                           In our implementation, two Res-RGNN layers are uti-
BAY.                                                                     lized, and the graph convolution kernel size is set to 64.
                                                                         To have a fair comparison, these parameters are set same in
                          METR-LA                PEMS-BAY
      Methods
                   MAE     MAPE   RMSE    MAE     MAPE    RMSE
                                                                         DCRNN. The proposed framework utilizes one Res-RGNN
         HA        3.65    11.2%    6.4   2.88     6.8%    4.76
                                                                         branch for near time, and one hop1 Res-RGNN branch for
      ARIMA        3.47    8.0%    7.81   2.33     5.4%    4.76          daily period. In the above Res-RGNN branch, 6 observed
        VAR        3.93    8.2%    6.59   2.32     5.0%    4.25          data points are used to forecast traffic conditions. Another
        SVR        3.45    8.0%    7.05   2.48     5.5%    5.18          branch, the hop Res-RGNN for daily period, 4 historical
        FNN        3.31    7.8%    7.43   2.30     5.4%    4.63
                                                                         data points are utilized in the experiments. Due to the lim-
     FC-LSTM       2.94    7.5%    5.92   2.20     5.2%    4.55
      DCRNN        2.49    5.5%    3.95   1.72     3.9%    3.97
                                                                         ited size of the datasets, a branch for a week hop is not used
     M-RGNN        2.34    5.3%    3.94   1.67     3.8%    3.73          in the experiments. To overcome this issue, we have utilized
   MRes-RGNN-NG    2.19    5.1%    3.86   1.61     3.7%    3.54          the “metadata” as external factors to explore the weekly pat-
   MRes-RGNN-G     2.07    4.9%    3.57   1.52     3.2%    3.23          terns.

                                                                         Notations of Variants of Our Framework
Compared Methods and Evaluation Metrics
                                                                         In these experiments, some variants of our proposed frame-
We compare traffic prediction performance of our proposed                work are (i) M-RGNN denotes a RGNN branch and a hop
MRes-RGNN with widely used time series regression mod-                   RGNN branch that both of them are utilized, (ii) MRes-
els that include (i) HA: Historical Average, which can model             RGNN-G denotes our proposed framework with the gated
traffic flow as a seasonal process, and then use weighted av-            residual scheme and multiple RGNN branches, (iii) MRes-
erage of previous seasons as traffic prediction; (ii) ARIMA:             RGNN-NG presents the residual shortcut connections are
Auto-regressive integrated moving average is a well-known                adopted, while the gated residual scheme is not utilized,
model that can understand and predict future values in a                 (iv) RGNN denotes the proposed recurrent graph neural net-
time series; (iii) SVR: Support vector regression uses linear            works with only a branch for the near temporal dependency
support vector machine for regression tasks. Subsequently,               without gated residual shortcuts, (v) Res-RGNN-G stands
the traffic performance of MRes-RGNN is also compared                    for one branch of RGNN for the near with gated residual
with the deep neural network based methods such as (iv)                  shortcuts, and lastly, (vi) Res-RGNN-NG means one branch
FNN: Feed forward neural network with two hidden lay-                    of RGNN for the near with residual shortcuts.
ers; (v) FC-LSTM: Fully connected LSTM neural networks;                     Please note that, in the above variants, the external factors
(vi) DCRNN: MRes-RGNN (Li et al. 2018) is compared                       are utilized.
with DCRNN uses both of the diffusion convolutions and
RNN together in traffic prediction. In summary, we compare               Overview of Performance Evaluation
the traffic prediction performance of our proposed MRes-
RGNN with 6 existing methods as discussed previously.                    The traffic prediction performance comparison of different
   Three evaluation metrics can be used to measure the traf-             approaches on both datasets is shown in Table 1. 3 out 7
fic prediction performance of above the methods as follows,              variants of our proposed MRes-RGNN framework are uti-
(i) Mean Absolute Error (MAE), (ii) Mean Absolute Per-                   lized in the experiments.
centage Error (MAPE), and lastly, (iii) Root Mean Squared                   Some phenomena can be clearly observed from Table
Error (RMSE).                                                            1. (1) Our proposed framework MRes-RGNN-G together
                                                                         with its variants M-RGNN and MRes-RGNN-NG have out-
Implementation Details                                                   performed other baselines using measurement from above
                                                                         three evaluation metrics. (2) Even M-RGNN gets better re-
We first aggregate traffic speed readings into 5-minutes win-
                                                                         sults than other baselines. One of the reasons is the hop
dows for METR-LA, 60-minutes windows for PEMS-BAY,
                                                                         scheme that can effectively model temporal dependencies.
and then apply Z-Score normalization. In each dataset, 70%,
                                                                         (3) MRes-RGNN-G and MRes-RGNN-NG achieve better
20% and 10% of its dataset are split into training, validation
                                                                         traffic prediction performance than M-RGNN using mea-
and testing datasets, respectively.
                                                                         surement from above three evaluation metrics. One of the
   To construct a sensor graph, we compute the pairwise road
                                                                         reasons is to introduce residual shortcut connections into M-
network distances among sensors and also build the adja-
                                                                         RGNN, while the performance is further improved by adopt-
cency matrix using the threshold Gaussian kernel:
             (                                                           ing the gated residual learning scheme.
                          dij 2                    d2i,j                    The traditional time-series prediction methods such as
                 exp(−       2 ), i 6= j and exp(− ϕ2 ) ≥ ,
   w(i, j) =               ϕ                                             HA and ARIMA cannot get good traffic prediction results
                 0, otherwise,                                           as they rely on historical records to predict the future val-
                                                             (9)         ues without considering spatial and other related external
where w(i, j) represents the edge weight from sensor i to                features. Even the regression-based methods such as VAR
sensor j, di,j denotes the road network distance from sensor             and SVR can take spatial correlations as their features. As
i to sensor j, ϕ is the standard deviation of the distances and          a result, the regression-based methods can achieve better
 is the threshold to control distribution and sparsity of the
matrix. At the end, the adjacency matrix of PEMS-BAY is                     1
                                                                              The hop size of the daily branch is set to (all the minutes in a
generated as shown in Figure 5(b).                                       day)/q, where q is the length of the time segmentation in minute.


                                                                   490
                (a)                            (b)                               (c)                             (d)

Figure 6: Effectiveness of residual and gated mechanisms. (a) Speed prediction in the morning peak; (b) Speed prediction in
the evening peak;(c) Speed prediction for sudden changes in the morning; (d) MAE versus the number of training epochs.


performance results than other conventional time-series ap-           Table 2: Results of MRes-RGNN and its variants on the
proaches. However, they are still hard to capture the complex         METR-LA dataset
non-linear temporal dependencies and the dynamic spatial
                                                                                    Methods       MAE    MAPE      RMSE
relationships. Deep neural networks methods (e.g., FNN and
                                                                                  Res-RGNN-N      2.32   5.18%      3.88
FC-LSTM) can overcome above the limitation. In Table 1,                          Res-RGNN-ND      2.16   4.98%      3.72
FNN and FC-LSTM have outperformed VAR and SVR.                                    MRes-RGNN       2.07   4.91%      3.57
   Again, In Table 1, our proposed MRes-RGNN framework
and DCRNN have outperformed FNN and FC-LSTM. One                      Table 3: Results of MRes-RGNN and its variants on the
of the reasons is that both our framework and DCRNN can               PEMS-BAY dataset
effectively model spatio-temporal dependencies using diffu-
sion convolutions and RNN together.                                                 Methods       MAE    MAPE      RMSE
                                                                                  Res-RGNN-N      2.32   5.18%      3.88
                                                                                 Res-RGNN-ND      2.16   4.98%      3.72
Effectiveness of Gated Residual Mechanism                                         MRes-RGNN       2.07   4.91%      3.57
In this section, we evaluate the effects of gated residual
mechanism. In the experiment, we only adopt a branch
to extract spatial and near temporal features. Figures 6(a)           means only the near branch is adopted without external fac-
and 6(b) show the forecasting visualization in the morning            tors; Res-RGNN-ND denotes near and daily branches are
peak hours and evening rush hours using METR-LA dataset,              adopted without external factors.
while Figure 6(c) shows the forecasting results on situations            From these figures, we can see that the Res-RGNN-N that
with sudden changes in the morning and evening, respec-               only uses the near branch performs better than the other
tively. In above these figures, GT denotes the ground truth.          baselines as shown in Tables 2 and 3 . This demonstrates
   In this experiment, Res-RGNN-NG has been proven that               the effectiveness of applying gated residual RGNN to learn
can capture the trend of rush hours more accurately than              spatio-temporal features in traffic prediction. The perfor-
RGNN. Importantly, it can detect the ending of the rush               mance is improved by adding the daily branch. Besides,
hours earlier than others. Therefore, the traffic performance         considering the external factors can also improve the perfor-
can be further improved by adopting the gated mechanism.              mance. Lastly, in the Tables 2 and 3, the proposed method
Figure 6c has shown that Res-RGNN is more sensitive to                that combines the near and daily with external features to-
sudden changes than RGNN.                                             gether can achieve the best results.
   To investigate the running performance of above the com-
pared deep learning models, the MAE of the testing data                                     Conclusions
of METR-LA is plotted on the training phase, as shown in
Figure 6(d). This figure has shown that the gated residual            Traffic prediction is a vital part and challenging task in the
mechanism can achieve much faster training procedure.                 domain of intelligent transportation system as it is affected
                                                                      by many complex factors, such as spatio-temporal depen-
Effectiveness of Multiple Res-RGNN and External                       dencies with external influences. In this paper, we propose to
                                                                      adopt gated residual recurrent graph neural networks to cap-
Facotrs
                                                                      ture graph-based spatial dependencies and temporal dynam-
In this section, we also investigate the traffic prediction           ics jointly. Experimental results have demonstrated that the
performance of multiple Res-RGNN branches and exter-                  proposed MRes-RGNN framework outperforms the state-
nal factors. Table 2 shows that the results of our proposed           of-the-art baselines. In our future work, we plan to investi-
MRes-RGNN and its variants on the METR-LA dataset.                    gate the spatio-temporal features learned by MRes-RGNN
In these figures, MRes-RGNN means our proposed frame-                 for better interpretability. We will also investigate to ap-
work, which combines gated residual learning, the near                ply our framework in network anomaly detection (Xie et al.
and the daily branches with external factors; Res-RGNN-N              2018a; 2018b; 2017)


                                                                491
                   Acknowledgements                                         for large-scale transportation network speed prediction. In Sensors,
The research was partially funded by the National Key                       818.
R&D Program of China (Grant No.2018YFB1003401),                             Niepert, M.; Ahmed, M.; and Kutzkov, K. 2016. Learning convo-
the National Outstanding Youth Science Program of Na-                       lutional neural networks for graphs. In International conference on
tional Natural Science Foundation of China (Grant No.                       machine learning, 2014–2023.
61625202), the International (Regional) Cooperation and                     Pascanu, R.; Gulcehre, C.; Cho, K.; and Bengio, Y. 2013. How
Exchange Program of National Natural Science Founda-                        to construct deep recurrent neural networks. In arXiv preprint
tion of China (Grant No. 61661146006, 61860206011),                         arXiv:1312.6026.
the National Natural Science Foundation of China (Grant                     Seo, Y.; Defferrard, M.; Vandergheynst, P.; and Bresson, X. 2016.
No.61602350), the Singapore-China NRF-NSFC Grant                            Structured sequence modeling with graph convolutional recurrent
(Grant No. NRF2016NRF-NSFC001-111).                                         networks. In arXiv preprint arXiv:1612.07659.
                                                                            Sutskever, I.; Vinyals, O.; and Le, Q. V. 2014. Sequence to se-
                         References                                         quence learning with neural networks. In Advances in neural in-
                                                                            formation processing systems, 3104–3112.
Atwood, J., and Towsley, D. 2016. Diffusion-convolutional neural
networks. In Advances in Neural Information Processing Systems,             Teng, S.-H. 2016. Scalable algorithms for data and network analy-
1993–2001.                                                                  sis. In Foundations and Trends® in Theoretical Computer Science,
                                                                            1–274.
Bruna, J.; Zaremba, W.; Szlam, A.; and LeCun, Y. 2013. Spec-
tral networks and locally connected networks on graphs. In arXiv            Werbos, P. J. 1990. Backpropagation through time: what it does
preprint arXiv:1312.6203.                                                   and how to do it. In Proceedings of the IEEE, 1550–1560.
Chen, C.; Li, K.; Teo, S. G.; Chen, G.; Zou, X.; Yang, X.; Vijay,           Xia, D.; Wang, B.; Li, H.; Li, Y.; and Zhang, Z. 2016. A dis-
R. C.; Feng, J.; and Zeng, Z. 2018. Exploiting spatio-temporal cor-         tributed spatial–temporal weighted model on mapreduce for short-
relations with multiple 3d convolutional neural networks for city-          term traffic flow forecasting. In Neurocomputing, 246–263.
wide vehicle flow prediction. In IEEE International Conference on           Xie, K.; Li, X.; Wang, X.; Xie, G.; Wen, J.; Cao, J.; and Zhang,
Data Mining, 893–898.                                                       D. 2017. Fast tensor factorization for accurate internet anomaly
Chung, J.; Gulcehre, C.; Cho, K.; and Bengio, Y. 2014. Empirical            detection. In IEEE/ACM transactions on networking, 3794–3807.
evaluation of gated recurrent neural networks on sequence model-            Xie, K.; Li, X.; Wang, X.; Cao, J.; Xie, G.; Wen, J.; Zhang, D.; and
ing. In arXiv preprint arXiv:1412.3555.                                     Qin, Z. 2018a. On-line anomaly detection with high accuracy. In
Defferrard, M.; Bresson, X.; and Vandergheynst, P. 2016. Con-               IEEE/ACM Transactions on Networking, 1222–1235.
volutional neural networks on graphs with fast localized spectral           Xie, K.; Peng, C.; Wang, X.; Xie, G.; Wen, J.; Cao, J.; Zhang, D.;
filtering. In Advances in Neural Information Processing Systems,            and Qin, Z. 2018b. Accurate recovery of internet traffic data under
3844–3852.                                                                  variable rate measurements. In IEEE/ACM Transactions on Net-
Du, S.; Li, T.; Gong, X.; Yu, Z.; and Horng, S.-J. 2018. A hybrid           working, 1137–1150.
method for traffic flow forecasting using multimodal deep learning.         Yao, H.; Tang, X.; Wei, H.; Zheng, G.; Yu, Y.; and Li, Z. 2018a.
In arXiv preprint arXiv:1803.02099.                                         Modeling spatial-temporal dynamics for traffic prediction. In arXiv
Du, Y.; Wang, W.; and Wang, L. 2015. Hierarchical recurrent                 preprint arXiv:1803.01254.
neural network for skeleton based action recognition. In Inter-             Yao, H.; Wu, F.; Ke, J.; Tang, X.; Jia, Y.; Lu, S.; Gong, P.; and
national conference on computer vision and pattern recognition,             Ye, J. 2018b. Deep multi-view spatial-temporal network for taxi
1110–1118.                                                                  demand prediction. In Association for the advancement of artificial
He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016a. Deep residual learn-         intelligence (to be appeared).
ing for image recognition. In International conference on computer          Yu, R.; Li, Y.; Shahabi, C.; Demiryurek, U.; and Liu, Y. 2017.
vision and pattern recognition, 770–778.                                    Deep learning: A generic approach for extreme condition traffic
He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016b. Identity mappings            forecasting. In SIAM International Conference on Data Mining,
in deep residual networks. In European conference on computer               777–785.
vision, 630–645.                                                            Yu, B.; Yin, H.; and Zhu, Z. 2017. Spatio-temporal graph con-
Hong, W.-C. 2011. Traffic flow forecasting by seasonal svr with             volutional neural network: A deep learning framework for traffic
chaotic simulated annealing algorithm. In Neurocomputing, 2096–             forecasting. In arXiv preprint arXiv:1709.04875.
2107.                                                                       Zeng, Z.; Liang, N.; Yang, X.; and Hoi, S. C. H. 2018. Multi-target
Jabbarpour, M. R.; Zarrabi, H.; Khokhar, R. H.; Shamshirband, S.;           deep neural networks: Theoretical analysis and implementation. In
and Choo, K. R. 2018. Applications of computational intelligence            Neurocomputing, 634–642.
in vehicle traffic congestion problem: a survey. In Soft Computing,         Zhang, J.; Zheng, Y.; Qi, D.; Li, R.; Yi, X.; and Li, T. 2018. Pre-
2299–2320.                                                                  dicting citywide crowd flows using deep spatio-temporal residual
Kipf, T. N., and Welling, M. 2016. Semi-supervised classi-                  networks. In Artificial Intelligence, 147–166.
fication with graph convolutional networks. In arXiv preprint               Zhang, J.; Zheng, Y.; and Qi, D. 2017. Deep spatio-temporal resid-
arXiv:1609.02907.                                                           ual networks for citywide crowd flows prediction. In Association
Li, Y.; Yu, R.; Shahabi, C.; and Liu, Y. 2018. Diffusion convo-             for the advancement of artificial intelligence, 1655–1661.
lutional recurrent neural network: Data-driven traffic forecasting.
In International conference on learning representations (to be ap-
peared).
Ma, X.; Dai, Z.; He, Z.; Ma, J.; Wang, Y.; and Wang, Y. 2017.
Learning traffic as images: a deep convolutional neural network


                                                                      492

