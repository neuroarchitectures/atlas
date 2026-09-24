# ST MGCN Geng Zhang Yang etal 2019

> Source: `ST_MGCN_Geng_Zhang_Yang_etal_2019.pdf`

---

                                              The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19)




                                         Spatiotemporal Multi-Graph
                           Convolution Network for Ride-Hailing Demand Forecasting
           Xu Geng,∗1 Yaguang Li,∗2 Leye Wang,1,3 Lingyu Zhang,4 Qiang Yang,1 Jieping Ye,4 Yan Liu2,4
1
    Hong Kong University of Science and Technology, 2 University of Southern California, 3 Peking University, 4 Didi AI Labs, Didi Chuxing
                       xgeng@connect.ust.hk, yaguang@usc.edu, wly@cse.ust.hk, zhanglingyu@didichuxing.com,
                                   qyang@cse.ust.hk, yejieping@didichuxing.com, yanliu.cs@usc.edu


                                    Abstract                                        demand of a region is usually affected by its spatially ad-
                                                                                    jacent neighbors and at the same time correlated with dis-
          Region-level demand forecasting is an essential task in ride-
          hailing services. Accurate ride-hailing demand forecasting
                                                                                    tant regions with the similar contextual environment. On the
          can guide vehicle dispatching, improve vehicle utilization, re-           other hand, non-linear dependencies also exist among differ-
          duce the wait-time, and mitigate traffic congestion. This task            ent temporal observations. The prediction of a certain time is
          is challenging due to the complicated spatiotemporal depen-               usually correlated with various historical observations, e.g.,
          dencies among regions. Existing approaches mainly focus on                an hour ago, a day ago or even a week ago.
          modeling the Euclidean correlations among spatially adjacent
          regions while we observe that non-Euclidean pair-wise cor-
          relations among possibly distant regions are also critical for
          accurate forecasting. In this paper, we propose the spatiotem-
          poral multi-graph convolution network (ST-MGCN), a novel
                                                                                                                                  4           R14
          deep learning model for ride-hailing demand forecasting. We
                                                                                                                                                               Highway 1
          first encode the non-Euclidean pair-wise correlations among                          Park


          regions into multiple graphs and then explicitly model these                                     3                                               1            2
          correlations using multi-graph convolution. To utilize the                                                        R46

          global contextual information in modeling the temporal cor-                         Hospital             School                     Hospital
                                                                                                                                                               School


          relation, we further propose contextual gated recurrent neu-
          ral network which augments recurrent neural network with                                                          6
                                                                                                                                              Amusement Park                Park
                                                                                          Amusement Park
          a contextual-aware gating mechanism to re-weights different
          historical observations. We evaluate the proposed model on                                           Highway 2                  5
          two real-world large scale ride-hailing demand datasets and
                                                                                                                                Factory   Lake
          observe consistent improvement of more than 10% over state-
          of-the-art baselines.
                                                                                    Figure 1: An example of different correlations among re-
                                                                                    gions. To predict the demand in region 1, spatially adjacent
                                Introduction                                        region 2, functionality similar region 3 and transportation
        Spatiotemporal forecasting is a crucial task in urban com-                  connected region 4 are considered more important, while
        puting. It has a wide range of applications from autonomous                 distant and irrelevant regions 5 are less relevant.
        vehicles operations, to energy and smart grid optimization,
        to logistics and supply chain management. In this paper, we                    Recent advances in deep learning enable promising re-
        study one important task: region-level ride-hailing demand                  sults in modeling the complex spatiotemporal relationship
        forecasting, which is one of the essential components of the                in region-based spatiotemporal forecasting. With convolu-
        intelligent transportation systems. The goal of region-level                tional neural network and recurrent neural network, state-
        ride-hailing demand forecasting is to predict the future de-                of-the-art results are achieved in (Shi et al. 2015; Yu et
        mand of regions in a city given historical observations. Ac-                al. 2017; Shi et al. 2017; Zhang, Zheng, and Qi 2017;
        curate ride-hailing demand forecasting can help organize ve-                Zhang et al. 2018a; Ma et al. 2017; Yao et al. 2018b;
        hicle fleet, improve vehicle utilization, reduce the wait-time,             2018a). Despite promising results, we argue that two im-
        and mitigate traffic congestion (Yao et al. 2018b). This task               portant aspects are largely overlooked in modeling the spa-
        is challenging mainly due to the complex spatial and tem-                   tiotemporal correlations. First, these methods mainly focus
        poral correlations. On the one hand, complicated dependen-                  on modeling the Euclidean correlations among different re-
        cies are observed among different regions. For example, the                 gions, however, we observe that non-Euclidean pair-wise
            ∗
              Equal contribution. Work done primarily while authors were            correlations are also critical for accurate forecasting. Fig-
        interns at Didi AI Labs, Didi Chuxing.                                      ure 1 shows an example. For region 1, in addition to neigh-
        Copyright c 2019, Association for the Advancement of Artificial             borhood region 2, it may also correlate to a distant region
        Intelligence (www.aaai.org). All rights reserved.                           3 that shares similar functionality, i.e., they are both near


                                                                             3656
schools and hospitals. Besides, region 1 may also be affected              Non-Euclidean structured data also exists in urban com-
by region 4, which is directly connected to region 1 via a              puting. Usually, station or point based prediction tasks, like
highway. Second, in these methods, when modeling tempo-                 traffic prediction (Li et al. 2018c; Yu, Yin, and Zhu 2018;
ral correlation with RNN, each region is processed indepen-             Yao et al. 2018a), point-based taxi demand prediction (Tong
dently or only based on local information. However, we ar-              et al. 2017) and station-based bike flow prediction (Chai,
gue that global and contextual information are also impor-              Wang, and Yang 2018) are naturally non-Euclidean as the
tant. For example, a global increase/decrease in ride-hailing           data format is no longer a matrix and convolution neural net-
demand usually indicates the occurrence of some events that             works becomes less helpful. Manual feature engineering or
will affect future demand.                                              graph convolution networks are state-of-the-art techniques
   To address these challenges, we propose a novel deep                 for handling non-Euclidean structure data. Different from
learning model called spatiotemporal multi-graph convolu-               previous works, ST-MGCN encodes pair-wise relationships
tion network (ST-MGCN). In ST-MGCN, we propose to                       among regions into semantic graphs. Though ST-MGCN
encode the non-Euclidean correlations among regions into                is designed for region based prediction, the irregularity of
multiple graphs. Different from (Yao et al. 2018b), which               region-wise relationship makes it a prediction problem for
uses the graph embedding as extra constant features for                 non-Euclidean data.
each region, we leverage the graph convolution to explic-                  In (Yao et al. 2018b), the authors propose DMVST-Net
itly model the pair-wise relationship among regions. Graph              which encodes the region-wise relationship as graph for taxi
convolution is able to aggregate neighborhood information               demand prediction. DMVST-Net mainly uses graph embed-
when performing the prediction which is hard to achieve                 ding as an external features for spatiotemporal prediction,
through traditional graph embedding. Furthermore, to in-                and consequently fails to use the demand values from re-
corporate global contextual information when modeling the               lated regions. In (Yao et al. 2018a), the authors further im-
temporal correlation, we propose contextual gated recurrent             proves (Yao et al. 2018b) by modeling the periodically shift
neural network (CGRNN). It augments RNN by learning a                   problem with the attention mechanism. However, none of
gating mechanism, which is calculated based on the summa-               these approaches explicitly models the non-Euclidean pair-
rized global information, to re-weight observations in dif-             wise relationships among regions. In this work, ST-MGCN
ferent timestamps. When evaluated on two real-world large               uses the proposed multi-graph convolution to incorporate
scale ride-hailing demand datasets, ST-MGCN consistently                features from related regions, which is able to make pre-
outperforms state-of-the-art baselines by a large margin. In            dictions from demand values of regions that are related in
summary, this paper makes the following contributions:                  different perspective.
• We identify non-Euclidean correlations among regions in                  Recent research in neuroimage analysis for Parkinson’s
   ride-hailing demand forecasting and propose to encode                disease (Zhang et al. 2018b) shows the effectiveness of
   them using multiple graphs. Then we further leverage                 graph convolution network in spatial feature extraction. It
   the proposed multi-graph convolution to explicitly model             uses GCN to learn features from most similar regions and
   these correlations.                                                  proposed a multi-view structure to fuse different MRI ac-
                                                                        quisitions. However, temporal dependency is not considered
• We propose the Contextual Gated RNN (CGRNN) to in-
                                                                        in above work. ST-GCN is used in spatiotemporal predic-
   corporate the global contextual information when model-
                                                                        tion for skeleton based action recognition (Li et al. 2018a;
   ing the temporal dependencies.
                                                                        Yan, Xiong, and Lin 2018).The transformation of ST-GCN
• We conduct extensive experiments on two large-scale                   is a combination of spatial dependency and local temporal
   real-world datasets, and the proposed approach achieves              recurrence. However, we argue in these models, the contex-
   more than 10% relative error reduction over state-of-the-            tual information or the global information is largely over-
   art baseline methods for ride-hailing demand forecasting.            looked in the temporal dependency modeling.
                      Related work                                      Graph convolution network
Spatiotemporal prediction in urban computing                            Graph convolution network (GCN) is defined over a graph
Spatiotemporal prediction is a fundamental problem for                  G = (V, A), where V is the set of all vertices and A ∈
data-driven urban management. There are rich amount of                  R|V |×|V | is the adjacency matrix whose entries represent
works on this topic, including predicting bike flows (Zhang,            the connections between vertices. GCN is able to extract lo-
Zheng, and Qi 2017), the taxi demand (Ke et al. 2017b;                  cal features with different reception fields from translation
Yao et al. 2018b), the arrival time (Li et al. 2018b), and the          variant non-Euclidean structures (Hammond et al. 2011).
precipitation (Shi et al. 2015; 2017), where the prediction             Let L = I − D −1/2 AD −1/2 denotes the graph Laplacian
is aggregated in rectangular regions, and region-wise rela-             matrix, where D is the degree matrix, a graph convolution
tionship is modeled by geographical distance. More specif-              operation (Defferrard, Bresson, and Vandergheynst 2016) is
ically, the spatial structure of urban data is formulated as a          defined as
                                                                                                    K−1
matrix whose entries represent rectangular regions. In pre-                                          X
vious works, regions and their pair-wise relationships nat-                              Xl+1 = σ(       αk Lk Xl ) 1             (1)
urally formulate an Euclidean structure, and consequently                                            k=0
convolution neural networks are leveraged for effective pre-               1
                                                                             In a graph convolution layer with P inputs and Q outputs,
diction.                                                                there will be P Q convolution operations. Here, we only show one


                                                                 3657
where Xl denotes the features in the l-th layer, αk is the                                 Encode pair-wise
                                                                                     correlations between regions
                                                                                                                       Aggregate different
                                                                                                                        observations with
                                                                                                                                             Capture spatial dependency
                                                                                                                                             with graph convolution on    Generate prediction

trainable coefficient, Lk is the k-th power of the graph                                 using multiple graphs        Contextual Gated RNN        multiple graphs


Laplacian matrix, σ is the activation function.                                                                                       RNN                          …
                                                                                                                                                     GCN
                                                                                                                        Contextual
                                                                                                                         Gating

                                                                                                   Neighborhood       Contextual Gated RNN          GCN
Channel-wise attention
Channel-wise attention (Hu, Shen, and Sun 2018; Chen et                                                                 Contextual
                                                                                                                                      RNN            GCN           …
al. 2017) is proposed in the computer vision literature. The                                                             Gating

                                                                                                                      Contextual Gated RNN
                                                                                                   Func. similarity                                 GCN
intuition behind channel-wise attention is to learn a weight
for each channel, in order to find the most important frames                                                                          RNN
                                                                                                                                                      GCN          …
and emphasize them by giving higher weights. Let X ∈                                                                     Contextual
                                                                                                                          Gating



RW ×H×C denotes the input, where W and H are the di-                                                Connectivity      Contextual Gated RNN           GCN


mensions of the input image, and C denotes the number of
channels, then the pipeline of channel-wise attention is de-                   Figure 2: System architecture of the proposed spatiotempo-
fined as follows:                                                              ral multi-graph convolution network (ST-MGCN). We en-
                                                                               code different aspects of relationships among regions, in-
                               W      H
                          1 XX                                                 cluding neighborhood, functional similarity and transporta-
zc = Fpool (X:,:,c ) =               Xi,j,c for c = 1, 2, · · · C              tion connectivity, using multiple graphs. First, the proposed
                         W H i=0 j=0
                                                                               contextual gated recurrent neural network (CGRNN) is used
                   s = σ(W2 δ(W1 z))                              (2)          to aggregate observations in different times considering the
                                                                               global contextual information. After that, multi-graph con-
              X̃:,:,c = X:,:,c ◦ sc       for c = 1, 2, · · · C
                                                                               volution is used to model the non-Euclidean correlations
Fpool is a global average pooling operation, which summa-                      among regions.
rizes each channel into a scalar zc where c is the channel in-
dex. Then an attention operation is applied to generate adap-
tive channel weights s by applying non-linear transforma-                      different aspects of correlations between regions as multi-
tions on the summarized vector z, where W1 and W2 is the                       ple graphs, whose vertices represent regions and edges en-
corresponding weights, δ and σ is the ReLU and sigmoid                         code the pair-wise relationship among regions. First, we use
function respectively. After that, s is applied to the input                   the proposed Contextual Gated Recurrent Neural Network
via channel-wise dot product. Finally, the input channels are                  (CGRNN) to aggregate observations in different times con-
scaled based learned weights. In this work, we adopt the idea                  sidering the global contextual information. After that, multi-
of channel-wise attention, and generalize it for temporal de-                  graph convolution is applied to capture different types of
pendency modeling among a sequence of graphs.                                  correlations between regions. Finally, a fully connected neu-
                                                                               ral network is used to transform features into the prediction.
                         Methodology                                           Spatial dependency modeling
We formalize the learning problem of spatiotemporal ride-                      In this section, we show how to encode different types of
hailing demand forecasting and describe how to model the                       correlations among regions using multiple graphs and how
spatial and temporal dependencies using the proposed spa-                      to model these relationships using the proposed multi-graph
tiotemporal multi-graph convolution network (ST-MGCN).                         convolution.
                                                                                  We model three types of correlations among regions
Region-level ride-hailing demand forecasting                                   with graphs, including (1) the neighborhood graph GN =
We divide a city into equal-size grids, and each grid is de-                   (V, AN ), which encode the spatial proximity, (2) functional
fined as a region v ∈ V , where V denotes the set of all                       similarity graph GF = (V, AF ), which encodes the similar-
disjoint regions in the city. Let X (t) represent the num-                     ity of surrounding Point of Interests (POIs) of regions, and
ber of orders in all regions at the t-th interval. Then the                    (3) the transportation connectivity graph GT = (V, AT ),
region-level ride-hailing demand forecasting problem is for-                   which encodes the connectivity between distant regions.
mulated as a single step spatiotemporal prediction given in-                   Note that, our approach can be easily extended to model new
put with a fixed temporal length, i.e., learning a function                    types of correlations by constructing related graphs.
f : R|V |×T → R|V | that maps historical demands of all
regions to the demand in the next timestep.                                    Neighborhood Neighborhood of a region is defined based
                                                                               on the spatial proximity. We construct the graph by connect-
                                           f (·)                               ing a region to its 8 adjacent regions in a 3 × 3 grid.
             [X (t−T +1) , · · · , X (t) ] −−→ X (t+1)
                                                                                                     
                                                                                                       1, vi and vj are adjacent
                                                                                           AN,ij =                                      (3)
                                                                                                       0, otherwise
Framework overview The system architecture of the pro-
posed model ST-MGCN is shown in Figure 2. We represent
                                                                               Functional similarity When making prediction for a re-
operation for simplicity.                                                      gion, it is intuitive to refer to other regions that are similar to


                                                                        3658
this one in terms of functionality. Region functionality could
be characterized using its surrounding POIs for each cate-
gory, and the edge between two vertices (regions) is defined
                                                                                                   𝛼" 𝐿" 𝑋%
as the POI similarity:
               AS,i,j = sim(Pvi , Pvj ) ∈ [0, 1]           (4)                                                   (

where Pvi , Pvj are the POI vectors of regions vi and vj re-
                                                                                                   𝛼& 𝐿& 𝑋%
spectively, whose dimension equals to the number of POI                         𝑋%                                             𝑋%*"
categories and each entry represents the number of a spe-
                                                                                                       '
cific POI category in the region.                                                                  𝛼' 𝐿 𝑋%



Transportation connectivity The transportation system
is also an important factor when performing spatiotemporal               Figure 3: An example of the ChebNet graph convolution
predictions. Intuitively, those geographically distant but con-          centralized at the black vertex. Left: The centralized region
veniently reachable regions can be correlated. These kinds               is marked black. The one-hop neighbors are marked yellow,
of connectivity are induced by roads like motorway, high-                while the two-hop neighbors are marked red. Middle: with
way or public transportation like subway. Here, we define                the increase of degree of the graph Laplacian, the reception
regions that are directly connected by these roads as “con-              field grows (marks green). Right: The output of this layer is
nected” and the corresponding edge is defined as:                        a sum among graph transformations with degree value from
                                                                         1 to K.
   AC,i,j = max(0, conn(vi , vj ) − AN,i,j ) ∈ {0, 1}      (5)
where conn(u, v) is the indicator function of the connectiv-
ity between vi and vj . Note that, the neighborhood edges are            and the corresponding entries of the 1-degree graph Lapla-
removed from connectivity graph to avoid redundant corre-                cian are:
lations and also results in a sparser graph.
                                                                                     L1C,1,4 6= 0; L1C,1,6 = 0; L1C,4,6 6= 0
Multi-graph convolution for spatial dependency model-                    If the maximum degree of graph Laplacian K is set to 1,
ing With these graphs constructed, we propose the multi-                 the transformed feature vector of region 1, i.e., Xl+1,1,:
graph convolution to model the spatial dependency as de-                 will not contain the feature vector of region 6: Xl,6,: , since
fined in Equation 6.                                                     L1C,1,6 = 0. When increasing K to 2, the corresponding en-
                                            !
                       G                                                 try L2C,1,6 becomes non-zero, and consequently Xl+1,1,: can
            Xl+1 = σ        f (A; θi )Xl Wl          (6)                 utilize information from Xl,6,: .
                           A∈A                                              The multi-graph convolution based spatial dependency
where Xl ∈ R     |V |×Pl
                         , Xl+1 ∈ R|V |×Pl+1 are the feature             modeling is not restricted to these three types of region-
vectors of |V | regions in layer l and                                   wise relationships mentioned above, and it can be easily ex-
                                    F l + 1 respectively. σ de-          tended to model other region-wise relationships as well as
notes the activation function, and denotes the aggregation
function, e.g., sum, max, average etc. A denotes the set of              other spatiotemporal forecasting problems. It models spa-
graphs, and f (A; θi ) ∈ R|V |×|V | represents the aggregation           tial dependencies by feature extraction through region-wise
matrix of different samples based on graph A ∈ A param-                  relationship. With small reception field, the feature extrac-
eterized by θi , while Wl ∈ RPl ×Pl+1 denotes the feature                tion will focus on close regions, i.e., neighbors that can be
transformation matrix, For example, if f (A; θi ) is the poly-           reaches with small number of hops. Increasing the max de-
nomial function of the Laplacian matrix L, then this will                gree of graph Laplacian or stacking multiple convolution
become ChebNet (Defferrard, Bresson, and Vandergheynst                   layers will increase the reception field and consequently en-
2016) on multiple graphs. If f (A; θi ) = I, i.e., the identity          courage the model to capture more global dependencies.
matrix, then this will fall back to the fully connected net-                Graph embedding is an alternative technique for mod-
work.                                                                    eling the region-wise correlation. In DMVST-Net (Yao et
   In the implementation, f (A; θi ) is chosen to be the K or-           al. 2018b), the authors use graph embedding2 to represent
der polynomial function of the graph Laplacian L, and Fig-               region-wise relationship, and then add these embeddings
ure 3 shows an example of the value transformation for a                 as extra features to each region. We argue that spatial de-
centralized region through the graph convolution layer. Sup-             pendency modeling approach in ST-MGCN is preferred for
pose all the entries in the adjacency matrix are 0 or 1, entry           the following reasons: ST-MGCN encodes region-wise re-
Lkij 6= 0 means vi is able to reach vj in k-hop. In terms of             lationships into graphs and aggregate demand values from
convolution operation, k defines the size of reception field             related regions by graph convolution. While in DMVST-
during spatial feature extraction. Using road connectivity               Net, the region-wise relationship was embedded to a tem-
graph GC = (V, AC ) in Figure 1 to illustrate. In the ad-                poral invariant region-based feature as input to the model.
jacency matrix AC , we have:                                                2
                                                                              The graph embedding is pre-computed. It produces a
            AC,1,4 = 1; AC,1,6 = 0; AC,4,6 = 1,                          temporal-invariant feature vector for each region.


                                                                  3659
  [𝑿 𝒕 , 𝑿(𝒕+𝟏) , … ] ∈ ℝ𝑇× 𝑉 ×𝑃                                                                                                   Then an attention operation (Equation 9) is applied to the
                                       𝑧 ∈ ℝ𝑇×1×𝑃
                            𝐹𝑝𝑜𝑜𝑙                                                                                                  summarized vector z, where W1 and W2 is the correspond-
                                                                                                                                   ing weights, δ and σ is the ReLU and sigmoid function re-
 𝑇 observations
                       𝐹𝑝𝑜𝑜𝑙 (𝐹 𝐾
                               𝐺𝐶 )
                                   ′
                                                                                                                                   spectively.
                                                              Share-weight RNN
                                        𝐹𝑎𝑡𝑡𝑛
                                                𝑠 ∈ ℝ𝑇            RNN            RNN      ...        RNN
                                                                                                                                                 X̃ (t) = X (t) ◦ s(t)       for t = 1, 2, · · · T   (10)
                     Contextual
                      Gating                                                                                                       Finally, s is applied to the scale each temporal observation
                      ෩ 𝒕 ,𝑿
                      𝑿    ෩ 𝒕+𝟏 , … ∈ ℝ𝑇× 𝑉 ×𝑃                                                                                    (Equation 10).
                                                                                                                                                          (1)            (T )
                                                                                                           T                         Hi,: = RNN(X̃i,: , · · · , X̃i,: ; W3 ) for i = 1, · · · , |V |
                                                              T
                                                                                                                                                                                                 (11)
                𝑇 reweighted observations                         𝑇 reweighted observations are aggregated with RNN
                                                                                                                                   After the contextual gating, a shared RNN layer with weight
                                                                                                                                   W3 across all regions is applied to aggregate the gated input
Figure 4: Temporal correlation modeling with contextual                                                                            sequence of a region into a single vector Hi,: (Equation 11).
gated recurrent neural network (CGRNN). It first produces                                                                          The intuition of sharing RNN among regions is to find a
region descriptions using the global average pooling over the                                                                      universal aggregation rule for all regions, which encourages
input and its graph convolution output for each observation.                                                                       model generalization and reduces model complexity.
Then it transfer the summarized vector z into weights which
are used to scale each observation. Finally, a shared RNN
layer across all regions is applied to aggregate the gated in-
                                                                                                                                                                Experiments
put sequence of each region into a single vector.                                                                                  In this section, we compare the proposed model ST-MGCN
                                                                                                                                   with other state-of-the-art baselines for region-level ride-
                                                                                                                                   hailing demand forecasting.
Though DMVST-Net also captures the topological informa-
tion, it is hard to aggregate demand values from related re-                                                                       Dataset We conduct experiments on two real-world large
gions through the region-wise relationship. Also, invariant                                                                        scale ride-hailing datasets collected in cities: Beijing and
features have limited contribution to the model training.                                                                          Shanghai. Both of these datasets are collected in the main
                                                                                                                                   city zone of ride-hailing orders within the time period from
Temporal correlation modeling                                                                                                      Mar 1st, 2017 to Dec 31st, 2017. For data split, we use the
We propose the Contextual Gated Recurrent Neural Net-                                                                              data from Mar 1st 2017 to Jul 31st 2017 for training, data
work (CGRNN) to model the correlations between observa-                                                                            from Aug 1st 2017 to Sep 30th 2017 as validation, and the
tions in different timestamps. CGRNN incorporates contex-                                                                          data from Oct 1st 2017 to Dec 31st 2017 is used for testing.
tual information into the temporal modeling by augmenting                                                                          The POI data is collected in 2017, and contains 13 primary
RNN with a context aware gating mechanism whose archi-                                                                             POI categories. Each region is associated with a POI vec-
tecture is shown in Figure 4. Suppose, we have T temporal                                                                          tor, whose entry is the number of instances of a certain POI
observations and X (t) ∈ R|V |×P denotes the t-th observa-                                                                         category. The road network data used for transportation con-
tion, where P is the feature dimensions, P will be 1 if the                                                                        nectivity evaluation is provided by OpenStreetMap (Haklay
feature only contains the number of orders. Then the work-                                                                         and Weber 2008).
flow of contextual gating mechanism is as follows.
                                                          0
                                                                                                                                   Experimental Settings
             X̂ (t) = [X (t) , FGK (X (t) )] for t = 1, 2, · · · T                                                    (7)          Recall that the learning task is formulated as learning a func-
First, the contextual gating mechanism produces region de-                                                                         tion f : R|V |×T → R|V | . In the experiment, we generate the
scriptions by concatenating the historical data of a certain                                                                       region set V by partitioning city map into grids with size
region with information from related regions. The informa-                                                                         equals to 1km × 1km 3 . There are totally 1296 regions in
tion from related regions is regarded as contextual informa-                                                                       Beijing, and 896 regions in Shanghai. Following the prac-
                                                            0
tion, and is extracted by a graph convolution operation FGK                                                                        tice in (Zhang, Zheng, and Qi 2017), the input of the net-
                     0
with max degree K (Equation 7) using the corresponding                                                                             work consists of 5 historical observations, including 3 lat-
graph Laplacian matrix. The contextual gating mechanism                                                                            est closeness components, 1 period component and 1 latest
is designed to involve information from related regions by                                                                         trend component. In building the transportation connectiv-
performing graph convolution operation before the pooling                                                                          ity graph, we consider the following high-speed roads, in-
step.                                                                                                                              cluding motorway, highway and subway. Two regions are
                                                                                                                                   regarded as “connected” as long as there is a high-speed road
                                                              |V |
                                                          1 X (t)                                                                  directly connecting them.
 z (t) = Fpool (X̂ (t) ) =                                         X̂ for t = 1, 2, · · · T (8)                                       In the experiment, f (A; θi ) in Equation 6 is chosen to be
                                                         |V | i=1 i,:
                                                                                                                                   the Chebyshev polynomial functionl (Defferrard, Bresson,
Secondly, we use the global average pooling Fpool over all                                                                         and Vandergheynst 2016) ofFthe graph Laplacian with the
regions to produce the summary of each temporal observa-                                                                           degree K equals to 2, and         is chosen to be the sum ag-
tion (Equation 8).                                                                                                                 gregation function. The number of hidden layers is 3, with
                                                                                                                                      3
                                        s = σ(W2 δ(W1 z))                                                             (9)                 Referred to industrial practice.


                                                                                                                            3660
Table 1: Performance comparison of different approaches                 model uses CNN with residual connections to capture the
for ride-hailing demand forecasting. ST-MGCN achieves the               trend, the periodicity, and the closeness information.
best performance with all metrics on both datasets.
                                                                      • DMVST-Net (Yao et al. 2018b): DMVST-Net is a multi-
                                                                        view based deep learning approach for taxi demand pre-
                  Beijing                 Shanghai                      diction. It consists of three different views: the temporal
 Method
               RMSE MAPE(%)            RMSE MAPE(%)                     view, the spatial view, and the semantic view modeled
 HA           16.14     23.9      17.15     34.8                        with LSTM, CNN and graph embedding respectively.
 LASSO     14.24±0.14 23.8±0.8 10.62±0.06 22.9±0.8                    • DCRNN, ST-GCN: Both DCRNN (Li et al. 2018c) and
 Ridge     14.24±0.11 23.8±0.9 10.61±0.04 23.1±0.8                      ST-GCN (Yu, Yin, and Zhu 2018) are graph convolu-
 VAR       13.32±0.17 22.4±1.6 10.54±0.18 23.7±1.4                      tion based models for traffic forecasting. Both models
 STAR      13.16±0.22 22.2±1.9 10.52±0.21 23.2±1.4                      use road network for building non-euclidean region-wise
 GBM       13.66±0.16 23.1±1.5 10.25±0.11 23.4±1.2
                                                                        relationship. DCRNN models the spatiotemporal depen-
                                                                        dency by integrating graph convolution into the gated re-
 STResNet  11.77±0.95 14.8±6.0 9.87±0.94 14.9±6.0                       current unit, while ST-GCN models the both the spatial
 DMVST-Net 11.62±0.48 12.3±5.5 9.61±0.44 13.8±1.2                       and temporal dependencies with convolution structures
 ST-GCN    11.62±0.36 10.1±5.1 9.29±0.31 11.2±1.3                       and achieves better efficiency.
 ST-MGCN 10.78±0.25 8.8±3.5          8.30±0.16    9.3±0.9
                                                                      Performance comparison
                                                                      For all approaches, we tune the model parameters using grid
64 hidden units each and an L2 regularization with a weight           search based on the performance on the validation dataset,
decay equal to 1e-4 is applied to each layer. Specially, the          and report the performance on the testing dataset over multi-
graph convolution degree K 0 in CGRNN equals to 1.                    ple runs. We evaluate the performance of based on two popu-
   We use ReLU as the activation in the graph convolution             lar metrics, i.e., Root Mean Square Error (RMSE) and Mean
network. The learning rate of ST-MGCN is set to 2e-3, and             Absolute Percentage Error (MAPE) 4 . Table 1 shows the test
early stopping on the validation dataset is used. All neural          error comparison of different approaches for ride-hailing de-
network based approaches are implemented using Tensor-                mand forecasting over of ten runs.
flow (Abadi and others 2016), and trained using the Adam                 We observe the following phenomena in both datasets:
optimizer (Kingma and Ba 2015) for minimizing RMSE.                   (1) deep learning based methods, including ST-ResNet,
The training of ST-MGCN takes 10GB RAM and 9GB GPU                    DMVST-Net, ST-GCN and the proposed ST-MGCN, which
memory. The training process takes about 1.5 hour on a sin-           are able to model the non-linear spatiotemporal dependen-
gle Tesla P40.                                                        cies, generally outperform other baselines; (2) ST-MGCN
                                                                      achieves the best performance regarding all the metrics on
Methods for evaluation We compare the proposed model                  both datasets, outperforming the second best baseline by
(ST-MGCN) with the following methods for ride-hailing de-             at least 10% in terms of relative error reduction, which
mand forecasting:                                                     suggests the effectiveness of proposed approaches for spa-
                                                                      tiotemporal correlations modeling; (3) compared with other
• Historical Average (HA): which models the ride-hailing              deep learning models, ST-MGCN also shows lower vari-
  demand as a seasonal process, and uses the average of               ance.
  previous seasons as the prediction. The period used is 1
  week, and the prediction is based on aggregated data from           Effect of spatial dependency modeling
  the same time in previous weeks.
                                                                      To investigate the effect of spatial and temporal dependency
• LASSO, Ridge: which takes historical data from different            modeling, we evaluate the following variants of ST-MGCN
  timestamps as input for linear regression with L1 and L2            by removing different components from the model, includ-
  regularization respectively.                                        ing: (1) the neighborhood graph, (2) the functional similarity
• Auto-regressive model (VAR,STAR): VAR is the multi-                 graph, (3) the transportation connectivity graph. The result
  variate extension of auto-regressive model which is able            is shown in Table 2. Removing any graph component causes
  to model the correlation between regions. STAR (Pace et             a significant error increase which justifies the importance of
  al. 1998) is a an AR extension specifically for spatiotem-          each type of relationship. These graphs encode the impor-
  poral modeling problems. In the experiment, the number              tant prior knowledge, i.e., region-wise correlation, which is
  of lags used is 5.                                                  leveraged for more accurate forecasting.
• Gradient boosted machine (GBM): gradient boosting                      To evaluate the effect of incorporating multiple region-
  decision tree based regression implemented using Light-             wise relationships, we extend existing single graph-based
  GBM (Ke et al. 2017a). The following setting is used in             models, including DCRNN (Li et al. 2018c) and ST-
  the experiment: the number of trees is 50, the maximum              GCN (Yu, Yin, and Zhu 2018) with the multi-graph con-
  depth is 4 and the learning rate is 2e-3.                           volution framework and the resulted models are DCRNN+
• ST-ResNet (Zhang, Zheng, and Qi 2017): ST-ResNet is                    4
                                                                           Following the practice in (Yao et al. 2018b), we filter the sam-
  a CNN-based framework for traffic flow prediction. The              ples with demand values less than 10 when computing MAPE.


                                                               3661
and ST-GCN+. As shown in table 3, both DCRNN+ and ST-                    Table 4: Effect of temporal correlation modeling on the Bei-
GCN+ achieve improved performance which shows the ef-                    jing dataset
fectiveness of incorporating multiple region-wise relation-
ships.                                                                             Temporal modeling approach                                RMSE
                                                                                                      Average pooling                        12.74
Table 2: Effect of spatial correlation modeling on the Bei-
                                                                                                           RNN                               11.05
jing dataset. Removing any component will result in a sta-
                                                                                                            CG                               11.82
tistically significant error increase.
                                                                                                          GRNN                               10.91
               Removed component        RMSE                                                                   CGRNN                         10.78
                  Neighborhood           11.47
                    Functional           11.42
                  Transportation         11.69                                                                              .
                    ST-MGCN              10.78                                                                                  .   
                                                                                                                            .   
                                                                                                                                .   
                                                                                     5 0 6 (                                .   
Table 3: Effect of adding multi-graph design to existing                                                                        .   
methodologies on the Beijing dataset. Adding extra graph to                                        
original model will result in a statistically significant error
decrease.                                                                                          
                                                                                                                                            
                                                                                                                        / D \ H U
                      Model        RMSE
                     ST-GCN        11.62                                 Figure 5: Effect of number of layers and the polynomial or-
                    ST-GCN+        11.20                                 der K of the graph convolution on the Beijing dataset.
                     DCRNN         12.02
                    DCRNN+         11.55
                    ST-MGCN        10.78                                 K and number of layers in the graph convolution. Figure 5
                                                                         shows the performance on test set. We observe that with
                                                                         the increase of number of layers, the error first decreases
Effect of temporal dependency modeling                                   and then increases. While the error first decreases and then
To further investigate the effect of temporal dependency                 plateaus with the increase of K. Larger K or the number of
modeling, we evaluate the following variants of ST-MGCN                  layers will enable the model capture more global correlation
using different methods for temporal modeling, including                 at the cost of increased model complexity and more prune to
(1) Average pooling: which aggregates different temporal                 overfitting.
observations using the average pooling, (2) RNN: which
aggregates temporal observations using the recurrent neu-                             Conclusion and Future work
ral network (RNN) (3) CG: which uses contextual gating to                In this paper, we investigated the region-level ride-hailing
re-weight different temporal observations but without RNN                demand forecasting problem and identified its unique spa-
(4) GRNN: CGRNN without the graph convolution (Equa-                     tiotemporal correlations. We proposed a novel deep learning
tion 7). The results are shown in Table 4. We observe the                based model which encoded the non-Euclidean correlations
following phenomena:                                                     among regions using multiple graphs and explicitly captured
• Average pooling which blindly averages different obser-                them using multi-graph convolution. We further augmented
   vations has the worst performance, while RNN which is                 the recurrent neural network with contextual gating mech-
   able to do content dependent non-linear temporal aggre-               anism to incorporate global contextual information in the
   gation achieves clearly improved results.                             temporal modeling procedure. When evaluated on two large
• CGRNN which augments RNN with contextual gating                        scale real-world ride-hailing demand datasets, the proposed
   mechanism achieves further improved result than RNN.                  approach achieved significantly better results than state-of-
   Besides, removing either the RNN (CG) or the graph con-               the-art baselines. For future work, we plan to investigate the
   volution operation (GRNN) results in clear worse perfor-              following aspects (1) evaluate the proposed model on other
   mance which justify the effectiveness of each component.              spatiotemporal forecasting tasks; (2) extend the proposed
                                                                         approach for multiple step sequence forecasting.
Effect of model parameters
To study the effects of different hyperparameters of the pro-
                                                                                                          Acknowledgement
posed model, we evaluate models on the Beijing by varying                This research has been funded in part by NSF grants IIS-
two of the most important hyperparameters, i.e., the degree              1254206, IIS-1539608, ITSP project No. ITS/391/15FX and


                                                                  3662
Hong Kong CERG grants 16209715, 16244616. The re-                      Ma, X.; Dai, Z.; He, Z.; Ma, J.; Wang, Y.; and Wang, Y.
search is supported by Didi Chuxing. Any opinions, find-               2017. Learning traffic as images: a deep convolutional neu-
ings, and conclusions or recommendations expressed in this             ral network for large-scale transportation network speed pre-
material are those of the authors and do not necessarily re-           diction. Sensors 17(4):818.
flect the views of any of the sponsors such as NSF.                    Pace, R. K.; Barry, R.; Clapp, J. M.; and Rodriquez, M.
                                                                       1998. Spatiotemporal autoregressive models of neighbor-
                       References                                      hood effects. The Journal of Real Estate Finance and Eco-
Abadi, M., et al. 2016. Tensorflow: Large-scale machine                nomics 17(1):15–33.
learning on heterogeneous distributed systems. In 12th                 Shi, X.; Chen, Z.; Wang, H.; Yeung, D.-Y.; Wong, W.-K.;
USENIX Symposium on Operating Systems Design and Im-                   and Woo, W.-c. 2015. Convolutional lstm network: A ma-
plementation (OSDI ’16).                                               chine learning approach for precipitation nowcasting. In Ad-
Chai, D.; Wang, L.; and Yang, Q. 2018. Bike flow prediction            vances in neural information processing systems, 802–810.
with multi-graph convolutional networks. SIGSPATIAL.                   Shi, X.; Gao, Z.; Lausen, L.; Wang, H.; Yeung, D.-Y.; Wong,
Chen, L.; Zhang, H.; Xiao, J.; Nie, L.; Shao, J.; Liu, W.;             W.-k.; and Woo, W.-c. 2017. Deep learning for precipitation
and Chua, T.-S. 2017. SCA-CNN: Spatial and channel-                    nowcasting: A benchmark and a new model. In Advances in
wise attention in convolutional networks for image caption-            Neural Information Processing Systems, 5617–5627.
ing. In Computer Vision and Pattern Recognition (CVPR),                Tong, Y.; Chen, Y.; Zhou, Z.; Chen, L.; Wang, J.; Yang, Q.;
2017 IEEE Conference on, 6298–6306. IEEE.                              Ye, J.; and Lv, W. 2017. The simpler the better: a unified
Defferrard, M.; Bresson, X.; and Vandergheynst, P. 2016.               approach to predicting original taxi demands based on large-
Convolutional neural networks on graphs with fast localized            scale online platforms. In Proceedings of the 23rd ACM
spectral filtering. In Advances in Neural Information Pro-             SIGKDD International Conference on Knowledge Discov-
cessing Systems, 3844–3852.                                            ery and Data Mining, 1653–1662. ACM.
Haklay, M., and Weber, P. 2008. Openstreetmap: User-                   Yan, S.; Xiong, Y.; and Lin, D. 2018. Spatial temporal graph
generated street maps. IEEE Pervasive Computing 7(4):12–               convolutional networks for skeleton-based action recogni-
18.                                                                    tion. In 2018 AAAI Conference on Artificial Intelligence
Hammond, D. K.; Vandergheynst, P.; Gribonval, R.; Ham-                 (AAAI’18).
mond, D. K.; Vandergheynst, P.; and Gribonval, R. 2011.                Yao, H.; Tang, X.; Wei, H.; Zheng, G.; Yu, Y.; and Li, Z.
Wavelets on graphs via spectral graph theory. 30(2):129–               2018a. Modeling spatial-temporal dynamics for traffic pre-
150.                                                                   diction. arXiv preprint arXiv:1803.01254.
Hu, J.; Shen, L.; and Sun, G. 2018. Squeeze-and-excitation             Yao, H.; Wu, F.; Ke, J.; Tang, X.; Jia, Y.; Lu, S.; Gong, P.;
networks. In Computer Vision and Pattern Recognition                   Ye, J.; and Li, Z. 2018b. Deep multi-view spatial-temporal
(CVPR), 2018 IEEE Conference on. IEEE.                                 network for taxi demand prediction. In 2018 AAAI Confer-
Ke, G.; Meng, Q.; Finley, T.; Wang, T.; Chen, W.; Ma, W.;              ence on Artificial Intelligence (AAAI’18).
Ye, Q.; and Liu, T.-Y. 2017a. Lightgbm: A highly efficient             Yu, R.; Li, Y.; Shahabi, C.; Demiryurek, U.; and Liu, Y.
gradient boosting decision tree. In Advances in Neural In-             2017. Deep learning: A generic approach for extreme con-
formation Processing Systems, 3149–3157.                               dition traffic forecasting. In SIAM International Conference
Ke, J.; Zheng, H.; Yang, H.; and Chen, X. M. 2017b. Short-             on Data Mining (SDM).
term forecasting of passenger demand under on-demand ride              Yu, B.; Yin, H.; and Zhu, Z. 2018. Spatio-temporal graph
services: A spatio-temporal deep learning approach. Trans-             convolutional networks: A deep learning framework for traf-
portation Research Part C: Emerging Technologies 85:591–               fic forecasting. In IJCAI’18).
608.                                                                   Zhang, J.; Zheng, Y.; Qi, D.; Li, R.; Yi, X.; and Li, T. 2018a.
Kingma, D. P., and Ba, J. 2015. Adam: A method for                     Predicting citywide crowd flows using deep spatio-temporal
stochastic optimization. In International Conference on                residual networks. Artificial Intelligence 259:147–166.
Learning Representations (ICLR ’14).                                   Zhang, X.; He, L.; Chen, K.; Luo, Y.; Zhou, J.; and Wang,
Li, C.; Cui, Z.; Zheng, W.; Xu, C.; and Yang, J. 2018a.                F. 2018b. Multi-view graph convolutional network and its
Spatio-temporal graph convolution for skeleton based action            applications on neuroimage analysis for parkinson’s disease.
recognition. In 2018 AAAI Conference on Artificial Intelli-            arXiv preprint arXiv:1805.08801.
gence (AAAI’18).                                                       Zhang, J.; Zheng, Y.; and Qi, D. 2017. Deep spatio-temporal
Li, Y.; Fu, K.; Wang, Z.; Shahabi, C.; Ye, J.; and Liu, Y.             residual networks for citywide crowd flows prediction. In
2018b. Multi-task representation learning for travel time es-          AAAI, 1655–1661.
timation. In International Conference on Knowledge Dis-
covery and Data Mining (KDD ’18).
Li, Y.; Yu, R.; Shahabi, C.; and Liu, Y. 2018c. Diffusion
convolutional recurrent neural network: Data-driven traffic
forecasting. In International Conference on Learning Rep-
resentations (ICLR ’18).


                                                                3663

