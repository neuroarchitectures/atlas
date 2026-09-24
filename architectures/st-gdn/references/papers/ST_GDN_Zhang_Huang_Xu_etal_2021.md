# ST GDN Zhang Huang Xu etal 2021

> Source: `ST_GDN_Zhang_Huang_Xu_etal_2021.pdf`

---

                                              Traffic Flow Forecasting with Spatial-Temporal Graph Diffusion Network
                                                             Xiyue Zhang1 , Chao Huang2 * , Yong Xu1,3,4 , Lianghao Xia1 , Peng Dai2
                                                                         Liefeng Bo2 , Junbo Zhang5,6 , Yu Zheng5,6
                                                                                     1
                                                                                   South China University of Technology, China
                                                                                     2
                                                                                       JD Finance America Corporation, USA
                                                                   3
                                                                     Communication and Computer Network Laboratory of Guangdong, China
                                                                                          4
                                                                                            Peng Cheng Laboratory, China
                                                         5
                                                           JD Intelligent Cities Research, China, 6 JD Intelligent Cities Business Unit, JD Digits, China
                                                         {zhang.xiyue,cslianghao.xia}@mail.scut.edu.cn, yxu@scut.edu.cn, chaohuang75@gmail.com




arXiv:2110.04038v1 [cs.LG] 8 Oct 2021
                                                                  {peng.dai,liefeng.bo}@jd.com, {msjunbozhang,msyuzheng}@outlook.com


                                                                    Abstract                                    With the advancement of deep learning techniques, many
                                                                                                             efforts have been devoted to developing traffic predic-
                                          Accurate forecasting of citywide traffic flow has been play-
                                          ing critical role in a variety of spatial-temporal mining appli-
                                                                                                             tion methods with various neural network architecture for
                                          cations, such as intelligent traffic control and public risk as-   spatial-temporal pattern modeling. Inspired by the sequence
                                          sessment. While previous work has made significant efforts         learning paradigm, recent neural networks have been uti-
                                          to learn traffic temporal dynamics and spatial dependencies,       lized to model temporal effects of traffic variations (Liu
                                          two key limitations exist in current models. First, only the       et al. 2016; Yu et al. 2017). To make use of spatial fea-
                                          neighboring spatial correlations among adjacent regions are        tures, some research work propose to adopt convolutional
                                          considered in most existing methods, and the global inter-         neural network to model correlations between adjacent re-
                                          region dependency is ignored. Additionally, these methods          gions (Zhang, Zheng, and Qi 2017), along with using re-
                                          fail to encode the complex traffic transition regularities ex-     current neural layers on the temporal dimension (Yao et al.
                                          hibited with time-dependent and multi-resolution in nature.        2018). Although both spatial and temporal correlations have
                                          To tackle these challenges, we develop a new traffic predic-
                                          tion framework–Spatial-Temporal Graph Diffusion Network
                                                                                                             been considered in existing methods, several key challenges
                                          (ST-GDN). In particular, ST-GDN is a hierarchically struc-         have not been well addressed.
                                          tured graph neural architecture which learns not only the local       In real-life scenarios, traffic flow pattern is often com-
                                          region-wise geographical dependencies, but also the spatial        plex and multi-periodic (Zhang, Zheng, and Qi 2017; Deng
                                          semantics from a global perspective. Furthermore, a multi-         et al. 2016), as different views with respect to time resolu-
                                          scale attention network is developed to empower ST-GDN             tions (e.g., hourly, daily, weekly) reflect the traffic dynamics
                                          with the capability of capturing multi-level temporal dynam-       from different temporal dimensions. The captured tempo-
                                          ics. Experiments on several real-life traffic datasets demon-
                                          strate that ST-GDN outperforms different types of state-of-
                                                                                                             ral patterns are often complementary with each other (Wu
                                          the-art baselines. Source codes of implementations are avail-      et al. 2018). Hence, learning accurate representations of traf-
                                          able at https://github.com/jill001/ST-GDN.                         fic variation patterns requires the collaboration of multi-
                                                                                                             ple views with different time resolutions. While recurrent
                                                                                                             neural network-based approaches have achieved good per-
                                                                Introduction                                 formance on various spatial-temporal sequence prediction
                                        Accurate forecasting of traffic flow across different geo-           tasks, they can only be effective for short-term, smooth dy-
                                        graphical regions in a city, have played a critical role in          namics and can hardly make predictions over the high-order
                                        smart transformation systems, such as intelligent transporta-        multi-dimensional time horizons.
                                        tion (Wei et al. 2018; Huang et al. 2020) and public risk as-           Most current forecasting approaches merely focus on
                                        sessment (Gao et al. 2019; Huang et al. 2018). For example,          modeling nearby geographical correlations (Yao et al. 2018;
                                        in disaster control, by predicting future traffic volume, local      Zhang, Zheng, and Qi 2017), while ignoring the cross-region
                                        governments and communities is able to design better trans-          inter-dependencies under a global context. For example, two
                                        portation scheduling and mobility management strategies, to          geographical areas with similar urban functions (e.g., shop-
                                        mitigate the tragedies caused by the crowd flow (Zhao et al.         ping zone or transportation hub) can be correlated in terms
                                        2017). In general, the objective of traffic prediction is to         of their traffic distribution, although they are not spatially
                                        forecast the traffic volume (e.g., inflow and outflow of each        adjacent or even far away from each other (Shen et al. 2018;
                                        region), from past traffic observations (Diao et al. 2019).          Wang and Li 2017). Hence, the learned region-wise rela-
                                            * Corresponding author: Chao Huang                               tional structures without the global-level traffic transition in-
                                        Copyright © 2021, Association for the Advancement of Artificial      formation, are insufficient to distill not only local geograph-
                                        Intelligence (www.aaai.org). All rights reserved.                    ical dependencies, but also global relations across regions,
which leads to suboptimal predictions results.                                              Methodology
    To tackle the above challenges, we propose a new predic-        This section presents the our ST-GDN with the descriptions
tive framework Spatial-Temporal Graph Diffusion Network             of different components (as shown in Figure 1).
(ST-GDN), for region-specific traffic flow. In ST-GDN, we
develop a multi-scale self-attention network to investigate
                                                                    Temporal Hierarchy Modeling
multi-grained temporal dynamics across various time reso-           We first propose a multi-scale self-attention network to
lutions, in order to encode temporal hierarchy of traffic tran-     jointly map multi-level temporal signals into common latent
sitional regularities. To promote the collaboration of differ-      representations, for capturing the complex traffic patterns.
ent granularity-aware temporal representations, an aggrega-         Definition 3 Temporal Resolution p. We define p to indi-
tion layer is proposed to model the underlying dependencies         cate how often we sample traffic volume measurement xti,j
across multi-level temporal dynamics. In addition, the devel-       from the overall traffic flow tensor X, i.e., the time difference
oped hierarchical graph neural network via attentive graph                                                               0
                                                                    between two consecutive data points xti,j and xti,j measured
diffusion paradigm, endows the ST-GDN with the capability
of incorporating spatial semantics from local-level spatially       from region rm,n . For example, (t0 − t) can be a hour, a day
adjacent relations to global-level traffic pattern representa-      or a week, given the resolution p is set as hourly, daily and
tions across the city in a joint way.                               weekly, respectively, i.e., p ∈ {hour, day, week}.
    We highlight the key contributions of this work as below:          Given each temporal resolution p, we could generate
                                                                                                      T
• We highlight the critical importance of explicitly explor-        resolution-aware traffic series xi,jp , where Tp is the corre-
   ing the multi-resolution traffic transitional information        sponding traffic series length with the resolution of p. Then,
   and local-global cross-region dependencies, in studying          we propose a self-attentive network to encode the traffic
   the traffic prediction problem.                                  variation patterns from the temporal dimension. In partic-
• We propose a new traffic prediction framework (ST-GDN)            ular, our encoder is built upon the scaled dot-product atten-
   which explicitly embeds multi-level temporal contextual          tion architecture with three transformation matrices: query
   signals into granularity-aware latent representations, with      (Q ∈ RTp ×d ), key (K ∈ RTp ×d ) and value (V ∈ RTp ×d )
   the cooperation of the designed multi-scale self-attention       matrices. The resolution-aware attentive aggregation mech-
   network and temporal hierarchy aggregation layer.                anism can be formally presented with the matrix calculation:
• ST-GDN preserves both local and global region-wise de-                   Q = Ep · WQ , K = Ep · WK , V = Ep · WV .                     (1)
   pendencies, via a hierarchically structured graph neural ar-
                                                                                                                                  T
   chitecture which is consisted of a graph attention network       We further perform the operation of Yp = σ( QK √ )V, where
                                                                                                                     d
   and convolution-based graph diffusion mechanism.                   p        p      p         d
                                                                    yi,j ∈ Y and yi,j ∈ R denotes the learned resolution-
• Our extensive experiments on three real-world datasets
                                                                    aware hidden representation of region ri,j . Ep ∈ R|R|×d is
   demonstrate that ST-GDN outperforms baselines of dif-
                                                                    the initialized embeddings of all regions ri,j ∈ R. Addi-
   ferent types in yielding better forecasting performance.
                                                                    tionally, σ(·) denotes the softmax function. Here, WQ , WK ,
   Furthermore, model efficiency study is conducted for ST-
                                                                    WV are projection matrices.
   GDN in the traffic prediction process.
                                                                    Traffic Dependency Learning with Global Context
                   Problem Definition
                                                                    The goal of this step is to exploit the global-level dependen-
In this section, we begin with some key definitions and pre-
                                                                    cies across different regions in terms of their dynamic traffic
liminary terms which are relevant to the solution.
                                                                    transition patterns. Towards this end, we first define a region
Definition 1 Spatial Region. We partition a city into I × J         graph G = (R, E), in which R is the region set and E de-
disjoint grids (given the geographical coordinates), in which       notes the pairwise relationships between two spatial regions.
each grid is regarded as a spatial region ri,j (i ∈ [1, ..., I],    Motivated by the attention-based neural network in encod-
j ∈ [1, ..., J]). ri,j is our target unit for traffic prediction.   ing the dependencies among regions (Huang et al. 2019), we
Definition 2 Traffic Flow Tensor. After the grid-based par-         develop an attentive aggregation mechanism to capture both
tition, we represent the citywide traffic volume distributions      local and global traffic dependency between regions. Specif-
across regions during past T time slots as a three-way ten-         ically, we perform the message aggregation over G with the
sor: X ∈ RI×J×T , where each entry xti,j denotes the traffic        following attentive operations.
volume measurement at region ri,j in the t-th time slot (e.g.,                                      H
hour or day). To study the prediction on both the incoming
and outgoing traffic follow, we generate two traffic flow ten-               mp(i,j)←(i0 ,j 0 ) =          h
                                                                                                          ω(i,j);(i             p
                                                                                                                    0 ,j 0 ) · Y · W
                                                                                                                                     p
                                                                                                                                         (2)
sors: Xα (incoming) and Xβ (outgoing), respectively.                                                h=1

Task Formulation. Based on the aforementioned defini-               where mp(i,j)←(i0 ,j 0 ) is the feature message propagated from
tions, the traffic prediction problem is formulated as: In-         region ri0 ,j 0 to ri,j . Here, we endow the cross-region rel-
put: the observed traffic volume information during past            evance encoding with multi-head (h ∈ [1, ..., H]), to cap-
T time slots across the entire city Xα ∈ RI×J×T and                 ture the region-wise relation semantic from different learn-
Xβ ∈ RI×J×T . Output: a predictive function which effec-            ing subspaces. Furthermore, Wp ∈ Rd×d is the parameter-
tively infers the unknown future traffic volume of regions.         ized projection matrix. The underlying attentive relevance
                                                                                                                                                                          Weekend/Weekday
                                                                    Trim Graph Diffusion convolution                                       k = K-1
                                                     𝑟𝑟2 𝑟𝑟3
                                                                                                                                       k = 0 𝚲𝚲 ∈ ℝ𝒑𝒑        |𝑅𝑅|×𝑄𝑄
                                                𝛼𝛼13 𝛼𝛼12                                                  Adjacent Matrix: A 𝑓𝑓                                          Holidays/Weather
                                                                                                                                          Θ
      𝑇𝑇𝑝𝑝 𝐄𝐄𝑝𝑝 ∈ ℝ|𝑅𝑅|×𝑑𝑑              𝑟𝑟 𝛼𝛼14 𝑟𝑟1 𝛼𝛼11 𝑟𝑟1′                                                                       −1 ⊺ 𝑘𝑘 �                                  Binary vectors




                                                                                                                                                                  Λ 𝑖𝑖,𝑗𝑗 = 𝑊𝑊 ℎ ∘ Λℎ𝑖𝑖,𝑗𝑗 + 𝑊𝑊 𝑑𝑑 ∘ Λ𝑑𝑑𝑖𝑖,𝑗𝑗 + 𝑊𝑊 𝑤𝑤 ∘ Λ𝑤𝑤
                                                                                                             −1      𝑘𝑘
                                                                                                  𝜃𝜃𝑘𝑘,1 (𝐷𝐷𝑜𝑜 𝐴𝐴) + 𝜃𝜃𝑘𝑘,2 (𝐷𝐷𝑖𝑖 𝐴𝐴 )
   𝑥𝑥𝑖𝑖,𝑗𝑗
                    𝐖𝐖𝐾𝐾   K                    𝛼𝛼15 𝛼𝛼1𝐼𝐼×𝐽𝐽                          𝒑𝒑
                                                                                                                                                                           Temperature and
                    𝐖𝐖𝑄𝑄   Q                𝑟𝑟5 … 𝑟𝑟𝐼𝐼×𝐽𝐽                            𝒁𝒁𝒊𝒊,𝒋𝒋  Input embedding: 𝒁𝒁𝒊𝒊,𝒋𝒋
                                                                                                                         𝒑𝒑
                                                                                                                                                                                  Wind speed
                                          Mutiple-heads𝐺𝐺 = (𝑅𝑅 , 𝐸𝐸 , 𝐴𝐴)                                                       … …

                                                                                                                                                                                                                         𝑖𝑖,𝑗𝑗
                    𝐖𝐖𝑉𝑉   V                                  𝑠𝑠     𝑠𝑠 𝑠𝑠
                                                                                                                                                                                   Range[0,1]
                                        Graph Attention                                                                                      k = K-1
                                                                    Trim Graph Diffusion convolution                                                                         External       factors
                                            𝑟𝑟3      𝑟𝑟2                                                                                 k = 0 𝚲𝚲𝒑𝒑 ∈ ℝ|𝑅𝑅|×𝑄𝑄
                                                                                                           Adjacent Matrix: A 𝑓𝑓                                                   embedding
                                                𝛼𝛼13 𝛼𝛼12                                                                                  Θ
                                                                                                                                                                                                     𝑔𝑔𝑡𝑡
      𝑇𝑇𝑝𝑝 𝐄𝐄𝑝𝑝 ∈ ℝ|𝑅𝑅|×𝑑𝑑             𝑟𝑟4 𝛼𝛼14 𝑟𝑟1 𝛼𝛼11 𝑟𝑟1′                                      𝜃𝜃𝑘𝑘,1 (𝐷𝐷𝑜𝑜−1 𝐴𝐴)𝑘𝑘 + 𝜃𝜃𝑘𝑘,2 (𝐷𝐷𝑖𝑖−1 𝐴𝐴⊺ )𝑘𝑘 �
   𝑥𝑥𝑖𝑖,𝑗𝑗
                    𝐖𝐖𝐾𝐾   K                    𝛼𝛼15 𝛼𝛼1𝐼𝐼×𝐽𝐽                            𝒑𝒑
                                                                                      𝒁𝒁𝒊𝒊,𝒋𝒋                             𝒑𝒑
                    𝐖𝐖𝑄𝑄   Q                𝑟𝑟5 … 𝑟𝑟𝐼𝐼×𝐽𝐽                                      Input embedding: 𝒁𝒁𝒊𝒊,𝒋𝒋                                                                              𝑔𝑔�𝑡𝑡
                                                                                                                                  … …
                    𝐖𝐖𝑉𝑉   V             Mutiple-heads
                                                           𝐺𝐺𝑠𝑠 = (𝑅𝑅𝑠𝑠 , 𝐸𝐸𝑠𝑠 , 𝐴𝐴)
                                       Graph Attention
                                                                                                                                            k = K-1                        Concatenation
                                                                    Trim Graph Diffusion convolution
                                            𝑟𝑟3      𝑟𝑟2
                                                                                                                                        k = 0 𝚲𝚲 ∈ ℝ𝒑𝒑        |𝑅𝑅|×𝑄𝑄
                                                𝛼𝛼13 𝛼𝛼12                                                  Adjacent Matrix: A 𝑓𝑓
                                       𝑟𝑟4 𝛼𝛼14 𝑟𝑟1 𝛼𝛼11 𝑟𝑟1′
                                                                                                                                           Θ
      𝑇𝑇𝑝𝑝 𝐄𝐄𝑝𝑝 ∈ ℝ|𝑅𝑅|×𝑑𝑑                                                                         𝜃𝜃𝑘𝑘,1 (𝐷𝐷𝑜𝑜−1 𝐴𝐴)𝑘𝑘 + 𝜃𝜃𝑘𝑘,2 (𝐷𝐷𝑖𝑖 𝐴𝐴⊺ )𝑘𝑘 �
                                                                                                                                     −1
   𝑥𝑥𝑖𝑖,𝑗𝑗
                    𝐖𝐖𝐾𝐾   K                    𝛼𝛼15 𝛼𝛼1𝐼𝐼×𝐽𝐽
                                                                                        𝒑𝒑
                                                                                                                                                                             𝑂𝑂𝑂𝑂𝑂𝑂𝑂𝑂𝑂𝑂𝑂𝑂𝑂𝑂𝑂𝑂𝑂𝑂
                    𝐖𝐖𝑄𝑄   Q                𝑟𝑟5 … 𝑟𝑟𝐼𝐼×𝐽𝐽                             𝒁𝒁𝒊𝒊,𝒋𝒋  Input embedding: 𝒁𝒁𝒊𝒊,𝒋𝒋
                                                                                                                          𝒑𝒑                           𝐼𝐼−1 𝐽𝐽−1              𝛼𝛼
                                                                                                                                                                      𝜆𝜆[ 𝑥𝑥̅𝑖𝑖,𝑗𝑗,𝑡𝑡         𝛼𝛼
                                                                                                                                                                                       − 𝑥𝑥𝑖𝑖,𝑗𝑗,𝑡𝑡 ]2
                                          Mutiple-heads                                                                           … …
                    𝐖𝐖𝑉𝑉   V                               𝐺𝐺𝑠𝑠 = (𝑅𝑅𝑠𝑠 , 𝐸𝐸𝑠𝑠 , 𝐴𝐴)                                                            𝓛𝓛 = � �                              𝛽𝛽            𝛽𝛽
                                        Graph Attention                                                                                                𝑖𝑖=0 𝑗𝑗=0
                                                                                                                                                                  +(1 − 𝜆𝜆)[ 𝑥𝑥̅𝑖𝑖,𝑗𝑗,𝑡𝑡 − 𝑥𝑥𝑖𝑖,𝑗𝑗,𝑡𝑡 ]2
          Temporal Hierarchy Modeling Traffic Dependency Learning with               Region-wise Spatial Relation Encoding Cross-resolution Pattern Aggregation
                                                   Global Context
                               Figure 1: The framework of our developed spatial-temporal graph diffusion networks.
 h
ω(i,j);(i 0 ,j 0 ) is formally estimated as follows:                                                        jointly
                                                                                                            √       preserves
                                                                                                                    √         the geographical adjacent relations (ri,j ’s
                                                                                                              K × K = K neighboring regions) and high traffic de-
     h
                                                 ypi,j ||e
                                      exp(LR(αT [e       ypi0 ,j 0 ]))                                      pendencies (larger ω(i,j);(i0 ,j 0 ) value). A denotes the adja-
    ω(i,j);(i0 ,j 0 ) = P
                                                         T yp ||ep
                              (i0 ,j 0 )∈N (i,j) exp(LR(α [ei,j yi0 ,j 0 ]))                                cent matrix weighted by a vertex distance function. Here, we
                                                                         p                p                 define Do = A · I to denote the out-degree diagonal matrix,
 where we perform concatenation between e             yi0 ,j 0 and y
                                                                   ei0 ,j 0                                 where I is the identify matrix of Gs . The designed diffusion
  p
 yi0 ,j 0 = ypi0 ,j 0 · Wp ). Then, the attentive coefficient vec-
(e                                                                                                          convolution operation performs the diffusion process across
tor α is incorporated with the production. LR(·) denotes the                                                nodes in Gs to generate new feature representations:
LeakyReLU function. Based on the constructed message and
                                                                                                                                  K−1
learned quantitative region-wise relevance score ω(i,j);(i0 ,j 0 ) ,                                                              X
we perform the information aggregation as:                                                                     f (zpi,j )Θ =              (θk,1 (Do−1 A)k + θk,2 (Di−1 A| )k )zpi,j (5)
                               X                                                                                                   k=0
                 zpi,j = f (          mp(i,j)←(i0 ,j 0 ) )           (3)
                                  ri0 ,j 0 ∈Ni,j
                                                                                                            where θk,1 , θk,2 ∈ RK×2 . Do−1 A (in-degree) and Di−1 A| )
                                                                                                            (out-degree) denote the bi-directional transition matrices of
where zpi,j is the aggregated feature embedding of ri,j .                                                   the diffusion process, which corresponds to the inflow and
                                                                                                            outflow in our prediction scenario. The parameter tensor de-
High-order Information Propagation. The information
                                                                                                            noted as Θ ∈ RQ×d×K×2 , in which the Q-dimensional out-
aggregation from the (l)-th layer to the (l + 1)-th layer with
the high-order relation modeling is represented as:                                                         put Λp ∈ R|R|×Q of diffusion convolutional layer is given:
                                                                                                                                                               d
    p,(l+1)                                    p,(l)
   zi,j     ← Aggregate           Propagate(zi,j , G)      (4)
                                                                                                                                                                 X
                                                                                                                               Λpq = LeakyReLU                                                f (Zpd0 )Θq,d0
                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                     (6)
                    i∈Nu (j);j 0 ∈Nv (j)
                                                                                                                                                                 d0 =1

Propagate(·) and Aggregate(·) denotes the message con-
                                                                                                             where q ∈ {1, ..., Q}. The obtained region representation
struction and information fusion, respectively. We finally
                                                                                                            Λpi,j jointly preserves the temporal (traffic time-varying pat-
generate the global-level representation of region ri,j as:
         p,(l)        p,(L)
                                                                                                            terns) and spatial (geographical relations) contextual signals
zpi,j = zi,j ⊕ ... ⊕ zi,j . ⊕ is the element-wise addition.                                                 under a global perspective.
                                                                                                               We next aggregate the resolution-aware traffic representa-
Region-wise Relation Learning with Graph                                                                    tion Λpi,j by introducing a gating mechanism. To be specific,
Diffusion Paradigm                                                                                          our gated aggregation mechanism conducts the parametric
In addition to the global dependencies across different re-                                                 matrix-based sum operation over the multi-resolution traffic
gions in terms of their traffic evolving patterns, we further                                               pattern representations, i.e., hourly (Λph ), daily (Λpd ) and
incorporate spatial relationships between regions into our                                                  weekly (Λpw ) as: Λi,j = Wh ◦Λhi,j +Wd ◦Λdi,j +Ww ◦Λw       i,j .
prediction framework. Particularly, motivated by (Li et al.                                                 Here, the trainable transformation matrices are denoted as
2018), we develop a graph-structured diffusion network to                                                   Wh , Wd and Ww corresponding to hourly, daily and weekly
refine the learned resolution-aware region representations                                                  patterns. We finally generate the conclusive multi-resolution
zpi,j from the above graph attention module. We generate an-                                                traffic representation Λi,j which preserves multi-grained
other region-wise relation graph Gs = (Rs , Es , A) which                                                   temporal hierarchy of traffic regularities.
Traffic Prediction Phase                                           NYC-Taxi (Yao et al. 2019). This data contains 22,000,000+
In urban sensing, there exist external factors (e.g., me-          taxi trajectories collected from 01/01/2015 to 03/01/2015 in
teorological conditions) which impact traffic transitional         New York City with a 10 × 20 grid map. The traffic data
regularities. Thus, we further augment our ST-GDN with the         sample period is also half an hour.
capability of fusing external factors. In particular, we con-      NYC-Bike (Zhang, Zheng, and Qi 2017). It includes the tra-
sider several types of external factors: Weather conditions,       jectories of the bike system from New York with a 16 × 8
Temperature/◦ C, Wind speed/mph. We map these features             grid map. Traffic volume is estimated on a hourly basis.
into vectors gt . After that, we utilize a multi-layer per-        Evaluation Protocols. We leverage two representative
ceptron framework to perform projection over ĝt . Finally,        metrics for evaluation: Root Mean Squared Error (RMSE)
we feed the concatenated embedding (Λi,j and ĝt ) into            and Mean Absolute Percentage Error (MAPE).
the prediction layer to infer the traffic volume of each region.
                                                                   Methods for Comparison. In the comparison, we con-
Optimized Loss Function. We define our loss function with          sider the following baselines with various model structures.
the joint consideration of inflow and outflow traffic volume       • ARIMA (Pan, Demiryurek et al. 2012). it is a representa-
of each region in a city as below:                                   tive method for forecasting time series data.
                   I−1 J−1
                   X   X                                           • Support Vector Regression (SVR) (Chang and Lin
             L=              λ[(x̄α           α
                                  i,j,t ) − (xi,j,t )]
                                                      2
                                                                     2011): another traditional time series analysis model via
                   i=0 j=0                                  (7)      learning feature mapping functions.
                   +(1 − λ)[(x̄βi,j,t ) − (xβi,j,t )]2             • Fuzzy+NN (Srinivasan, Chan, and Balaji 2009): it inte-
                                                                     grates the feed-forward neural layers with the fuzzy input
                     β
where x̄αi,j,t and x̄i,j,t denote the estimated incoming and
                                                                     filter to model the traffic patterns.
outgoing traffic volume of region ri,j at the t-th time slot,      • RNN (Liu et al. 2016): it leverages the recurrent neural
respectively. Their influences are decided by λ. Ground              networks for capturing both the spatial and temporal ef-
                                                 β
truth information are represented xα  i,j,t and xi,j,t .             fects for making sequential data prediction.
                                                                   • LSTM (Yu et al. 2017): it jointly models the normal and
Model Complexity Analysis. In this part, we analyze the              abnormal traffic variations based on stacked long short-
time complexity of our ST-GDN framework. Particularly,               term memory networks.
the multi-scale self-attentive network takes O(3 × T × I ×
J × d) for learning query, key and value matrices, and             • DeepST (Zhang et al. 2016): it utilizes the convolution
O(3×T 2 ×d) for weighted summation. The next attentional             neural network to encode the spatial correlations between
graph module takes O(3 × I 2 × J 2 × d0 ) to estimate the rel-       regions over a citywide grid map.
evance scores and perform feature aggregation, which dom-          • ST-ResNet (Zhang, Zheng, and Qi 2017): the residual
inates the computational cost of our ST-GDN. Additionally,           connection technique is employed to alleviate overfitting
the graph diffusion-based spatial relation modeling compo-           issue for spatial-temporal prediction.
nent takes O(K × |Es |) complexity.                                • DMVST-Net (Yao et al. 2018): it integrates the graph em-
                                                                     bedding method with the joint convolutional recurrent net-
                         Evaluation                                  works to capture spatial-temporal signals
In this section, we evaluate the performance of ST-GDN on          • DCRNN (Li et al. 2018): it is a data-driven forecasting
a series of experiments on several real-world datasets, which        framework with diffusion recurrent neural network to cap-
are summarized to answer the following research questions:           ture the spatial-temporal dependencies.
• RQ1: How is the overall traffic prediction performance of        • STDN (Yao et al. 2019): it designs a periodically shifted
  ST-GDN as compared to various baselines?                           attention for learning transition regularities of traffic.
• RQ2: How do designed different sub-modules contribute            • ST-GCN (Yu, Yin, and Zhu 2018): it is an integrative
  to the model performance?                                          framework of graph convolution network and convolu-
• RQ3: How does ST-GDN perform w.r.t different time                  tional sequence modeling layer for modeling spatial and
  granularity configurations for temporal context modeling?          temporal dependencies.
• RQ4: What is the influence of hyperparameter settings?           • ST-MGCN (Geng et al. 2019): it develops a multi-modal
                                                                     graph convolutional network to capture region-wise non-
• RQ5: How is the model efficiency of ST-GDN?                        Euclidean pair-wise correlations.
Experimental Settings                                              • GMAN (Zheng et al. 2020): it is a encoder-decoder traffic
                                                                     prediction method based on the graph multi-attention.
Data Description. Experiments are performed on three
real-world traffic datasets, which are summarized in Table 1:      • UrbanFM (Liang et al. 2019): it is a deep fusion network
BJ-Taxi (Zhang, Zheng, and Qi 2017). There are 34,000+               to model traffic flow distributions.
processed taxi trajectories included in this data. Each trajec-    • ST-MetaNet (Pan et al. 2019): it is a meta-learning ap-
tory is mapped into one of 32 × 32 grid-based geographical           proach to perform knowledge transfer across series with a
regions. The traffic volume is measured every half an hour.          recurrent graph attentive network.
         Dataset           BJ-Taxi            NYC-Taxi              NYC-Bike        External Factors                        Beijing                   New York City
   Data type            Taxi GPS              Taxi GPS             Bike Rental         Weather                     sunny, rainy et al.        sunny, rainy et al.
 Time interval         30 minutes            30 minutes             one hour        Temperature/◦ C                  [−24.6, 41.0]             [−10.3, 31.40]
 Gird map size           32×32                 10×20                  16×8          Wind speed/mph                     [0, 48.60]                 [0, 63.75]
  # of records          34,000+             22,000,000+              6,800+            holidays                   weekends, holidays         weekends, holidays
                                               Table 1: Statistical information of experimented datasets.

                                                                                                                       5
   25                                                     20                            20
                                15                                                                                     4                         20
   20

                            MAPE/%                                                  MAPE/%                                                   MAPE/%
                                                          15                            15
                                                        RMSE
                                                                                                                                                 15
 RMSE                                                                                                               RMSE
                                10                                                                                     3
   15           ST-GDN-d                     ST-GDN-d                    ST-GDN-d                      ST-GDN-d                   ST-GDN-d                     ST-GDN-d
                ST-GDN-g                     ST-GDN-g     10             ST-GDN-g       10             ST-GDN-g                   ST-GDN-g       10            ST-GDN-g
   10           ST-GDN-s                     ST-GDN-s                    ST-GDN-s                      ST-GDN-s        2          ST-GDN-s                     ST-GDN-s
                ST-GDN-n             5       ST-GDN-n                    ST-GDN-n                      ST-GDN-n                   ST-GDN-n                     ST-GDN-n
        5       ST-GDN-e                     ST-GDN-e          5         ST-GDN-e            5         ST-GDN-e        1          ST-GDN-e            5        ST-GDN-e
                ST-GDN                       ST-GDN                      ST-GDN                        ST-GDN                     ST-GDN                       ST-GDN
        0 Inflow Outflow             0 Inflow Outflow          0Inflow    Outflow            0Inflow    Outflow        0 Inflow Outflow               0 Inflow Outflow
        (a) NYC-Taxi                 (b) NYC-Taxi                  (c) BJ-Taxi                   (d) BJ-Taxi               (e) NYC-Bike               (f) NYC-Bike
                           Figure 2: Model ablation study of ST-GDN framework in terms of RMSE and MAPE.


Parameter Settings. The ST-GDN is implemented with
Tensorflow. The training phase is performed using the Adam
optimizer with the learning rate of 1e−3 and batch size
of 32. The embedding dimension size d and the depth re-
cursive graph neural layers L are set as 64 and 3, respec-
tively. We select the input sequence length from the range
of {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6}, which re-                               (a) ST-GCN               (b) ST-MGCN               (c) ST-MetaNet
spectively corresponds to three different time resolutions
(hour–Th , day–Td and week–Tw ). We stack three feed-
forward layers in the final prediction phase. The experiments
of most baselines are performed with their released code.

Performance Comparison (RQ1)
                                                                                                    (d) DCRNN                 (e) GMAN                    (f) ST-GDN
Performance Superiority of ST-GDN. The comparison re-
sults of all methods are presented in Table 2. We can observe                                      Figure 3: Visualization for Traffic Prediction Errors.
that ST-GDN consistently yields the best performance in all
cases, which demonstrates the effectiveness of our ST-GDN
in jointly modeling of multi-level temporal dynamics and                                     Comparison with Variants (RQ2)
global-level region-wise dependencies. Figure 3 visualize
the prediction error ([(x̄i,j,t ) − (xi,j,t )]2 ) of our ST-GDN                              We perform ablation experiments to analyze the effects of
and five best performed baselines on BJ-taxi data, where                                     sub-modules in our ST-GDN framework with five variants:
a brighter pixel means a larger error. The superiority of                                    • ST-GDN-s: ST-GDN without the multi-scale self-attention
ST-GDN can still be observed, which is consistent with the                                     network to capture multi-level traffic dynamics.
quantitative results in Table 2.
                                                                                             • ST-GDN-g: ST-GDN without the graph attention module
                                                                                               to model the global region-wise traffic dependencies.
Performance Comparison between Baselines. Compared
with conventional time series approaches, neural network-                                    • ST-GDN-d: ST-GDN without the graph diffusion network
based models perform better in most evaluation cases. The                                      to integrate spatial context with cross-region traffic pattern
subsequent attention-based and recurrent-convolutional net-                                    correlations for representation recalibration.
work methods (e.g., STDN, DMVST-Net) obtain better per-                                      • ST-GDN-n: ST-GDN without the incorporation of neigh-
formance than recurrent neural models (e.g., D-LSTM),                                          borhood spatial context into the graph diffusion.
which justifies the necessity to simultaneously capture both
spatial and temporal relations in traffic prediction. Among                                  • ST-GDN-e: ST-GDN without the external factor fusion.
various baselines, GNN-based methods have better perfor-                                        The evaluation results are shown in Figure 2. We can ob-
mance than other types of competitors, which ascertains the                                  serve that the joint version of ST-GDN outperforms other
rationality of designing graph-structured information aggre-                                 variants consistently. Hence, each designed sub-modules has
gation mechanism to fuse spatial and temporal signals.                                       positive effects for prediction performance improvement. It
       Datasets                BJ-Taxi                        NYC-Bike                          NYC-Taxi
       Metrics          RMSE         MAPE (%)           RMSE          MAPE (%)             RMSE       MAPE (%)
       Methods        In      Out    In      Out      In     Out      In       Out      In     Out    In    Out
       ARIMA        22.10 24.01 30.89 32.24 9.30 11.81 35.82 36.47 27.21 36.54 20.90 22.18
         SVR        21.44 22.12 22.64 22.32 8.65 9.07 23.58 24.10 26.16 34.71 18.25 21.01
      Fuzzy+NN      22.35 23.06 22.67 22.73 8.56 9.17 24.03 24.48 25.98 34.50 18.92 21.54
        RNN         27.16 27.90 24.17 24.72 8.99 9.24 28.22 28.58 29.88 37.23 25.97 26.55
        LSTM        26.99 27.56 23.64 24.17 8.64 9.10 27.51 28.07 29.52 37.04 25.81 26.11
       DeepST       19.30 21.06 22.45 22.52 7.66 8.16 22.81 23.21 23.56 26.79 22.34 22.39
      ST-ResNet     17.00 22.31 23.51 23.74 6.28 6.61 23.92 24.79 21.72 26.30 21.12 21.24
     DMVST-Net 16.61 17.14 22.52 23.06 5.82 6.09 22.45 23.67 20.63 25.80 17.19 17.44
        STDN        15.19 18.63 21.04 22.13 4.50 5.92 21.71 22.61 19.31 24.19 16.43 16.59
      UrbanFM       15.18 18.42 20.54 20.88 3.99 4.64 21.59 22.47 19.11 24.14 16.34 16.46
     ST-MetaNet 15.06 18.29 19.91 20.74 3.85 4.64 21.26 22.18 18.30 23.88 16.19 16.27
       DCRNN        15.13 18.37 20.14 20.88 3.86 4.65 21.14 21.05 18.19 23.74 16.11 16.16
       ST-GCN       15.11 18.30 19.92 20.77 3.76 4.70 21.12 21.94 18.02 23.08 15.94 15.92
      ST-MGCN       15.08 18.25 19.96 20.70 3.75 4.63 21.04 21.95 17.97 23.00 15.87 15.91
       GMAN         15.07 18.23 19.97 20.68 3.73 4.64 21.02 21.93 17.95 22.96 15.84 15.89
       ST-GDN       14.57 17.56 19.03 20.27 3.00 3.97 20.48 21.31 17.10 22.09 15.17 15.29
              Table 2: Performance comparison of all methods on three datasets in terms of RMSE and MAPE.
is necessary to build a joint framework to collectively in-      may introduce noise which mislead the traffic modeling.
tegrate the multi-resolution traffic temporal patterns, global   Kernel Size K. We vary the kernel size to investigate the
region-wise traffic dependencies, and regions’ geographical      convolution operations in our graph diffusion process. We
relations, into the spatial-temporal traffic pattern modeling.   can observe that K = 3 achieves the best performance.
Multi-Resolution Temporal Effects (RQ3)                          Channel Dimensionality. The results suggest that larger
                                                                 channel dimension size does not always bring the stronger
In this subsection, we study the effects of different temporal   representation ability, due to the overfitting issue.
resolution settings in our integrative architecture of multi-
scale self-attention network and cross-resolution pattern ag-
gregation layer, with the following contrast models:
                                                                 Model Efficiency Study (RQ5)
• ST-GDN h : P ∈ {hour/30mins}                                   We finally investigate the model efficiency (measured by
                                                                 running time) of our ST-GDN. All experiments are con-
• ST-GDN h,d : P ∈ {hour/30mins, day}                            ducted with the default parameter configurations on a single
• ST-GDN h,w : P ∈ {hour/30mins, week}                           NVIDIA GeForce GTX 1080 Ti GPU. We observe that in
                                                                 several best performed baselines, ST-GCN has good predic-
• ST-GDN h,d,w : P ∈ {hour/30mins, day, week}
                                                                 tion accuracy and running speed. Our ST-GDN outperforms
   We present the study results in Figure 4. As we can seen,     most of compared approaches and could achieve competi-
the best prediction accuracy is achieved by ST-GDN h,d,w         tive efficiency as compared to ST-GCN, i.e., the attention-
which is configured with more resolutions. Leaning the tem-      based graph embedding propagation layer has higher com-
poral hierarchy with hourly and daily/weekly traffic patterns    putational cost than the adjacent matrix-based graph convo-
(ST-GDN h,d , ST-GDN h,w ) provide better results as com-        lution. Considering the prediction accuracy comparison be-
pared to the variant with singular-dimensional time granu-       tween ST-GDN and ST-GCN, the additional computational
larity (ST-GDN h ). Overall, decomposing the temporal ef-        cost could bring positive effect via learning global region
fects into more multiple resolution-specific feature represen-   inter-dependencies in an explicit manner.
tations is helpful for more accurate modeling of traffic tem-       We finally investigate the efficiency (measured by running
poral regularity and resolution-aware region relations.          time) of our ST-GDN. Table 3 presents the computational
                                                                 cost of training (with 300 epochs) and inference phase for
Parameter Sensitivity (RQ4)                                      ST-GDN and five best performed baselines on three different
Depth of Graph Attention Network L. We can notice that           datasets. All experiments are conducted with the default pa-
increasing the depth of our graph attention module by stack-     rameter configurations on a single NVIDIA GeForce GTX
ing multiple embedding propagation layers could boost the        1080 Ti GPU. We can observe that ST-GDN outperforms
performance. The results also indicate that exploring third-     most of compared approaches and could achieve competi-
order relations among region entities is sufficient to capture   tive efficiency as compared to ST-GCN, i.e., the attention-
the global traffic dependencies.                                 based graph embedding propagation layer has higher com-
Length of Encoded Input Sequence T . The performance             putational cost than the adjacent matrix-based graph convo-
is initially improved with the increase of Th and Td , since     lution. Considering the prediction accuracy comparison be-
longer traffic series can provide more useful temporal infor-    tween ST-GDN and ST-GCN, the additional computational
mation. However, the further increasing of sequence length       cost could bring positive effect via learning global region
                                                                                                                                                 6
                                         20
                                                                                                           20                                    5                              20
       20


                                  MAPE/%                                                               MAPE/%                                                               MAPE/%
                                         15                                                                                                      4
RMSE                                                                                                                                       RMSE
                                                                                                           15                                                                   15
       15
                                         10                                                                                                      3
       10        ST-GDNh                            ST-GDNh                                                10             ST-GDNh                           ST-GDNh             10            ST-GDNh
                 ST-GDNh, w                         ST-GDNh, w                                                            ST-GDNh, w
                                                                                                                                                 2          ST-GDNh, w                        ST-GDNh, w
       5         ST-GDNh, d                5        ST-GDNh, d                                                  5         ST-GDNh, d                        ST-GDNh, d               5        ST-GDNh, d
                 ST-GDNh, d, w                      ST-GDNh, d, w                                                         ST-GDNh, d, w
                                                                                                                                                 1          ST-GDNh, d, w                     ST-GDNh, d, w
       0                                   0                                                                    0                                0                                   0
            Inflow Outflow                     Inflow Outflow                                                       Inflow Outflow                   Inflow Outflow                      Inflow Outflow
        (a) NYC-Taxi                       (b) NYC-Taxi                        (c) BJ-Taxi                          (d) BJ-Taxi                   (e) NYC-Bike                       (f) NYC-Bike
                                                             Figure 4: Multi-resolution temporal effect studies.
  24                                                                  24
                                    22                                                                    22                                22                                 22
  22

                                  RMSE                              RMSE                               RMSE                               RMSE                              RMSE
                                                                      22

RMSE
                                    20                                                                    20                                20                                 20
  20                                                                  20
                                    18                                                                    18                                18                                 18
  18                                                                  18
             Inflow    Outflow      16          Inflow   Outflow      16        Inflow       Outflow      16         Inflow    Outflow      16        Inflow    Outflow        16         Inflow   Outflow
  16
       2 2.5 3 3.5 4 4.5 5                 50 100 150 200 250              1    2   3    4    5   6             1     2    3    4   5            1    1.5   2   2.5   3              1    2    3   4    5
             Filter Size          Hidden Dimensionality               Sequence Length Th                   Sequence Length Td               Sequence Length Tw Number of GAT Layers


                                                Figure 5: Hyper-parameter study on NYC-Taxi data in terms of RMSE.

                                                       Training                                                 work, ST-GDN endows the spatial-temporal pattern rep-
            Methods
                              BJ-Taxi                NYC-Taxi          NYC-Bike                                 resentation process with the preservation of hierarchical
        ST-MetaNet           16121.01                 1298.55           1020.14                                 temporal dynamics and global-enhanced region-wise de-
         DCRNN                7996.24                  981.36            705.64                                 pendencies. While there exist research work that considers
         ST-GCN               4088.90                  744.65            500.40                                 the global dependency among regions (Zhang et al. 2020),
        ST-MGCN               8263.27                 1023.29            789.71                                 it is limited in its separately modeling of traffic dependency
          GMAN                7368.31                  854.66            547.12                                 and nearby region relations based on convolution neural
         ST-GDN               7625.19                  891.63            569.26                                 network. In this work, ST-GDN incorporates the global
                                                      Inference                                                 context enhanced region-wise explicit relevance into a graph
            Methods
                                 BJ-Taxi             NYC-Taxi          NYC-Bike                                 diffusion paradigm to capture comprehensive high-order
        ST-MetaNet                0.42                  0.32             0.29                                   region dependencies in a joint learning manner.
         DCRNN                    0.26                  0.24             0.21
         ST-GCN                   0.25                  0.22             0.19                                   Graph-based Spatial-Temporal Prediction. It is worth
        ST-MGCN                   0.31                  0.27             0.25                                   mentioning that several recent efforts have investigated
          GMAN                    0.25                  0.23             0.19                                   Graph Neural Networks (GNNs) for spatial-temporal data
         ST-GDN                   0.26                  0.23             0.20                                   forecasting (Guo et al. 2019; Song et al. 2020). For exam-
                                                                                                                ple, ST-GCN (Yu, Yin, and Zhu 2018) and ST-MGCN (Geng
                      Table 3: Model Efficiency Study.                                                          et al. 2019) proposes to leverage graph convolution net-
                                                                                                                work to model correlations between regions. Furthermore,
                                                                                                                attention mechanism has been introduced for information
                                                                                                                aggregation from adjacent roads (Zheng et al. 2020). Mo-
inter-dependencies in an explicit manner.
                                                                                                                tivated by these work, we develop a hierarchical graph
                                                                                                                neural architectures to promote the cooperation between
                                 Related Work                                                                   the multi-resolution temporal context with the cross-region
Traffic Prediction with Deep Learning. Recently, many                                                           inter-correlations, which have not been well explored in ex-
efforts have been devoted to developing traffic prediction                                                      isting solutions.
techniques based on various neural network architectures.
One straightforward solution is to apply the recurrent
neural networks (e.g., LSTM) to encode the temporal
                                                                                                                                                     Conclusion
features of traffic series (Yu et al. 2017). The subsequent                                                     This work investigates the traffic prediction problem by
extensions propose to integrate the recurrent neural layers                                                     proposing a new architecture (ST-GDN) based graph neu-
with the convolutional network (Zhang, Zheng, and Qi                                                            ral networks. Specifically, it first designs a resolution-aware
2017; Yao et al. 2018) or attention mechanism (Yao et al.                                                       self-attention network to encode the multi-level temporal
2019), so as to joint model the spatial-temporal signals.                                                       signals. Then, the local spatial contextual information and
In addition, some hybrid methods have been proposed for                                                         global traffic dependencies across different regions, are sub-
traffic prediction with the exploration of heterogeneous                                                        sequently integrated to enhance the spatial-temporal pat-
data fusion (Liang et al. 2019) and meta-learning-based                                                         tern representations. Comprehensive experiments demon-
knowledge transfer (Pan et al. 2019). Different from these                                                      strate that the proposed ST-GDN significantly outperforms
many baselines over several datasets consistently. Our future    Pan, B.; Demiryurek, U.; et al. 2012. Utilizing real-world
work lies in the deployment of our developed prototype in a      transportation data for accurate traffic prediction. In ICDM,
cloud-based working system for real-time traffic prediction.     595–604. IEEE.
                                                                 Pan, Z.; Liang, Y.; Wang, W.; Yu, Y.; Zheng, Y.; and Zhang,
                  Acknowledgments                                J. 2019. Urban Traffic Prediction from Spatio-Temporal
The authors would like to thank the anonymous refer-             Data Using Deep Meta Learning. In KDD. ACM.
ees for their valuable comments and helpful suggestions.         Shen, B.; Liang, X.; Ouyang, Y.; Liu, M.; Zheng, W.; and
This work is supported by National Nature Science Foun-          Carley, K. M. 2018. Stepdeep: a novel spatial-temporal mo-
dation of China (62072188, 61672241), Natural Science            bility event prediction framework based on deep neural net-
Foundation of Guangdong Province (2016A030308013),               work. In KDD, 724–733.
Science and Technology Program of Guangdong Province             Song, C.; Lin, Y.; Guo, S.; and Wan, H. 2020. Spatial-
(2019A050510010). This work is also partially supported by       Temporal Synchronous Graph Convolutional Networks: A
National Key R&D Program of China (2019YFB2101801)               New Framework for Spatial-Temporal Network Data Fore-
and the Beijing Nova Program (Z201100006820053).                 casting. In AAAI, volume 34, 914–921.
                       References                                Srinivasan, D.; Chan, C. W.; and Balaji, P. 2009. Compu-
Chang, C.-C.; and Lin, C.-J. 2011. LIBSVM: a library for         tational intelligence-based congestion prediction for a dy-
support vector machines. TIST 2(3): 27.                          namic urban street network. Neurocomputing 72(10-12):
                                                                 2710–2716.
Deng, D.; Shahabi, C.; Demiryurek, U.; Zhu, L.; Yu, R.; and
                                                                 Wang, H.; and Li, Z. 2017. Region representation learning
Liu, Y. 2016. Latent space model for road networks to pre-
                                                                 via mobility flow. In CIKM, 237–246.
dict time-varying traffic. In KDD, 1525–1534.
                                                                 Wei, H.; Zheng, G.; Yao, H.; and Li, Z. 2018. Intellilight: A
Diao, Z.; Wang, X.; Zhang, D.; Liu, Y.; Xie, K.; et al. 2019.    reinforcement learning approach for intelligent traffic light
Dynamic spatial-temporal graph convolutional neural net-         control. In KDD, 2496–2505.
works for traffic forecasting. In AAAI, volume 33, 890–897.
                                                                 Wu, X.; Shi, B.; Dong, Y.; Huang, C.; Faust, L.; and Chawla,
Gao, Y.; Zhao, L.; Wu, L.; Ye, Y.; Xiong, H.; and Yang,          N. V. 2018. Restful: Resolution-aware forecasting of behav-
C. 2019. Incomplete Label Multi-Task Deep Learning for           ioral time series data. In CIKM, 1073–1082.
Spatio-Temporal Event Subtype Forecasting. In AAAI, vol-
ume 33, 3638–3646.                                               Yao, H.; Tang, X.; Wei, H.; Zheng, G.; and Li, Z. 2019.
                                                                 Revisiting Spatial-Temporal Similarity: A Deep Learning
Geng, X.; Li, Y.; Wang, L.; Zhang, L.; Yang, Q.; Ye, J.; and     Framework for Traffic Prediction. In AAAI.
Liu, Y. 2019. Spatiotemporal Multi-Graph Convolution Net-
                                                                 Yao, H.; Wu, F.; Ke, J.; Tang, X.; Jia, Y.; Lu, S.; Gong, P.;
work for Ride-hailing Demand Forecasting. In AAAI.
                                                                 Ye, J.; and Li, Z. 2018. Deep multi-view spatial-temporal
Guo, S.; Lin, Y.; Feng, N.; Song, C.; and Wan, H. 2019. At-      network for taxi demand prediction. In AAAI, 2588–2595.
tention based spatial-temporal graph convolutional networks      Yu, B.; Yin, H.; and Zhu, Z. 2018. Spatio-temporal graph
for traffic flow forecasting. In AAAI, volume 33, 922–929.       convolutional networks: A deep learning framework for traf-
Huang, C.; Zhang, C.; Dai, P.; and Bo, L. 2020. Cross-           fic forecasting. In IJCAI.
Interaction Hierarchical Attention Networks for Urban            Yu, R.; Li, Y.; Shahabi, C.; Demiryurek, U.; and Liu, Y.
Anomaly Prediction. In IJCAI.                                    2017. Deep learning: A generic approach for extreme con-
Huang, C.; Zhang, C.; Zhao, J.; Wu, X.; Yin, D.; and             dition traffic forecasting. In SDM, 777–785. SIAM.
Chawla, N. 2019. Mist: A multiview and multimodal spatial-       Zhang, J.; Zheng, Y.; and Qi, D. 2017. Deep spatio-temporal
temporal learning framework for citywide abnormal event          residual networks for citywide crowd flows prediction. In
forecasting. In WWW, 717–728.                                    AAAI.
Huang, C.; Zhang, J.; Zheng, Y.; and Chawla, N. V. 2018.         Zhang, J.; Zheng, Y.; Qi, D.; et al. 2016. DNN-based predic-
DeepCrime: attentive hierarchical recurrent networks for         tion model for spatio-temporal data. In SIGSPATIAL, 1–4.
crime prediction. In CIKM, 1423–1432.                            Zhang, X.; Huang, C.; Xu, Y.; et al. 2020. Spatial-Temporal
Li, Y.; Yu, R.; Shahabi, C.; and Liu, Y. 2018. Diffusion con-    Convolutional Graph Attention Networks for Citywide Traf-
volutional recurrent neural network: Data-driven traffic fore-   fic Flow Forecasting. In CIKM, 1853–1862.
casting. In ICLR.                                                Zhao, L.; Sun, Q.; Ye, J.; Chen, F.; et al. 2017. Feature con-
Liang, Y.; Ouyang, K.; Jing, L.; Ruan, S.; Liu, Y.; Zhang, J.;   strained multi-task learning models for spatiotemporal event
Rosenblum, D. S.; and Zheng, Y. 2019. UrbanFM: Inferring         forecasting. TKDE 29(5): 1059–1072.
Fine-Grained Urban Flows. In KDD, 3132–3142. ACM.                Zheng, C.; Fan, X.; Wang, C.; and Qi, J. 2020. Gman:
Liu, Q.; Wu, S.; Wang, L.; and Tan, T. 2016. Predicting the      A graph multi-attention network for traffic prediction. In
Next Location: A Recurrent Model with Spatial and Tempo-         AAAI, 1234–1241.
ral Contexts. In AAAI, 194–200.

