# Financial time series forecasting with multi modality graph Recognition 2022

> Source: `Financial_time_series_forecasting_with_multi_modality_graph_Recognition_2022.pdf`

---

                                                                       Pattern Recognition 121 (2022) 108218



                                                                  Contents lists available at ScienceDirect


                                                                      Pattern Recognition
                                                           journal homepage: www.elsevier.com/locate/patcog




Financial time series forecasting with multi-modality graph neural
network
Dawei Cheng a,∗, Fangzhou Yang b, Sheng Xiang c, Jin Liu b
a
  Department of Computer Science and Technology, Tongji University, Shanghai, China
b
  AI Lab, Emoney Inc, Shanghai, China
c
  Center for Artiﬁcial Intelligence, University of Technology Sydney, Sydney, Australia




a r t i c l e            i n f o                           a b s t r a c t

Article history:                                           Financial time series analysis plays a central role in hedging market risks and optimizing investment de-
Received 15 December 2020                                  cisions. This is a challenging task as the problems are always accompanied by multi-modality streams
Revised 22 July 2021
                                                           and lead-lag effects. For example, the price movements of stock are reﬂections of complicated market
Accepted 31 July 2021
                                                           states in different diffusion speeds, including historical price series, media news, associated events, etc.
Available online 2 August 2021
                                                           Furthermore, the ﬁnancial industry requires forecasting models to be interpretable and compliant. There-
Keywords:                                                  fore, in this paper, we propose a multi-modality graph neural network (MAGNN) to learn from these
Graph neural network                                       multimodal inputs for ﬁnancial time series prediction. The heterogeneous graph network is constructed
Graph attention                                            by the sources as nodes and relations in our ﬁnancial knowledge graph as edges. To ensure the model
Deep learning                                              interpretability, we leverage a two-phase attention mechanism for joint optimization, allowing end-users
Quantitative investment                                    to investigate the importance of inner-modality and inter-modality sources. Extensive experiments on
                                                           real-world datasets demonstrate the superior performance of MAGNN in ﬁnancial market prediction. Our
                                                           method provides investors with a proﬁtable as well as interpretable option and enables them to make
                                                           informed investment decisions.
                                                                                                                              © 2021 Elsevier Ltd. All rights reserved.



1. Introduction                                                                             terns by extracting meaningful technical indicators [7] and/or la-
                                                                                            tent features [8]. Lately, with the development of social media and
    The ﬁnancial market capitalization of US domestic listed com-                           natural language processing technologies, unstructured news has
panies reaches 30 trillion dollars in 2019, account over 1.5 times                          been leveraged to improve the prediction model capability [9]. But
the Gross Domestic Product (GDP) in the United States [1]. In this                          these technologies do not capture internal relations among equi-
massive yet volatile market, forecasting the price movement of                              ties, which limits their potentials for the forecasting model. For ex-
equities is very important for both ﬁnancial institutions and in-                           ample, the term-level feature of an event “Qualcomm ﬁles lawsuit
vestors. According to the eﬃcient market hypothesis (EMH) [2],                              against Apple” cannot differentiate the appellor “Qualcomm” and
ideally, the stock’s prices reﬂect all available information in an ef-                      appellee “Apple”, so it is diﬃcult to infer the corresponding price
ﬁcient market, which includes historical prices, news, events, etc.                         movements of the related equities, Qualcomm and Apple Inc.
However, in a real-world situation, different equities responding to                            Recently, researchers [10] tend to improve the representation of
different events are non-intuitive and non-synchronized. Thus, it is                        market information by extracting structural event tuples and indi-
challenging to model this intricate phenomenon, named the lead-                             cators (i.e., sentiment indicator) [11] from media news. The main
lag effect [3], in a time series forecasting framework.                                     idea is to learn distributed representations that similar events or
    The ﬁnancial industry has researched price prediction models                            similar sentiment news could have similar features. These features
since the beginning of the twentieth century [4] and has perfected                          are then linked to listed companies and integrated with historical
these technologies ever since, investing millions of dollars in this                        time series for price prediction [12]. But two similar events may
process. Traditional quantitative methods rely on historical time-                          be quiet unrelated, such as “Steve Jobs quits Apple” and “David
series price data for stock price movement prediction [5,6]. These                          Peter leaves Starbucks”. To overcome this, studies [13,14] employ
models aim to reduce the stochasticity and capture consistent pat-                          external information from knowledge graphs (KG) in the feature
                                                                                            learning process [15]. Then, the above two events can have differ-
                                                                                            ent representations according to the semantic differences in KG,
    ∗
        Corresponding author.                                                               because Steve Jobs is the founder of Apple while David Peter is
        E-mail address: dcheng@tongji.edu.cn (D. Cheng).                                    more like to be a customer in Starbucks.


https://doi.org/10.1016/j.patcog.2021.108218
0031-3203/© 2021 Elsevier Ltd. All rights reserved.
D. Cheng, F. Yang, S. Xiang et al.                                                                                    Pattern Recognition 121 (2022) 108218


    However, the stock’s price movements in the ﬁnancial market               2.1. Lead-lag effect
not only rely on individual events of itself but also related to
the connections of other equities [16]. These multi-modality in-                  In an informational eﬃcient market, price movements of stocks
puts, including numerical time series, unstructured texts and rela-           can be deemed as the reaction of ﬁnancial events or news [17].
tional graphs, contribute differently as a synergy effect on the price        However, when a new event hit the stock market, prices of some
movement. For instance, an event “Qualcomm suits against Apple”               stocks response faster than others. This phenomenon of correlated
will also inﬂuence other players (i.e., competitors, upstream and             yet asynchronous price movement is referred to as lead-lag effect
downstream ﬁrms) of the smartphone market in different diffu-                 [3]. For example, in Fig. 1, when a new event (“Qualcomm suites
sion speeds, such as Samsung, Foxconn, and Google, etc. Effectively           against Apple”) hit the market, it will not only bring price ﬂuctua-
forecasting the prices of related equities from the lead-lag effects is       tion of “Qualcomm” and “Apple”, but will also inﬂuence upstream
challenging, due to the incompleteness of ﬁnancial domain knowl-              and downstream companies, such as Samsung (supplier and major
edge and intricate sequential patterns.                                       competitor of Apple in smart phone market) and Foxconn (man-
    Therefore, in this paper, we propose a multi-modality graph               ufacturer of Apple). But their price movements are asynchronous
neural network model for forecasting the price movements by in-               because the event diffusion speed is different over different enti-
corporating sources of lead-lag relationships, including historical           ties. Therefore, it is a challenging task to learn from this lead-lag
prices, media events, and corresponding knowledge from KG. In                 relationship in ﬁnancial market.
particular, we ﬁrst extract relations of linked entities from raw
news by and then store them in our ﬁnancial knowledge graphs                  2.2. Heterogeneous graph construction
(FinKG). Then, we propose a heterogeneous graph attention net-
work to learn the uniﬁed representation of target time series, in                In MAGNN, multi-modality heterogeneous graph extends
which multi-modality sources are deﬁned as source nodes and the               the conventional heterogeneous graph [18] with multi-modality
predicted equity as target node. We leverage a two-phase attention            sources. Graph nodes are divided into six types (source, news,
mechanism (inner-modality and inter-modality attention) to in-                events, market, bridge and target nodes) with three modality in-
fer the internal sequential patterns and inter-source lead-lag rela-          puts (numeral time series, media texts, and relations). We give the
tions. Inner-modality attention mechanism is designed to automat-             deﬁnition as follows:
ically learn different contributions of graph-structured sources to
the target node within each modality inputs. While inter-modality             Deﬁnition 1Heterogeneous graph. . A heterogeneous graph is de-
attention is proposed to learn weights among different modali-                noted as
ties dynamically for a decent price movement prediction of tar-
get nodes, as different modality contributes differently in differ-           G = ( VT , VS , E ),
ent time period. Afterwards, the learned informative features are             where VT represents the set of target nodes, VS denotes the set of
fed into prediction layer for price movement forecasting. Exten-              source nodes and E is the set of links connecting between nodes.
sive experiments on real market data show the effectiveness of our
method and interpretability of the proposed two-phase attention               Deﬁnition 2Source nodes. VS are associated with different modal-
mechanism.                                                                    ity by a mapping function  : VS → , where  denotes the set of
    In a nutshell, the main contribution of this paper includes:              modalities, including numeral market data, media texts, and rela-
                                                                              tions.
  • We formalize the problem of lead-lag effects in ﬁnancial time
    series forecasting and identify their unique challenges arising           Deﬁnition 3Target nodes. VT are our predicted equities in the
    from real ﬁnancial industry applications.                                 graph, which is designed to receive and aggregate messages from
  • We propose a novel multi-modality graph neural network                    other nodes via directed links.
    (MAGNN) to learn the lead-lag effects for ﬁnancial time series            Deﬁnition 4Bridge node. denotes the connected nodes between
    forecasting, which preserves informative market information as            multi-modality sources and target nodes. They are extracted from
    inputs, including historical prices, raw news text and relations          the domain knowledge graph FinKG.
    in KG. To our best knowledge, this is the ﬁrst study to explore
    the lead-lag effects by embedding informative sources in a uni-           Deﬁnition 5Attributed nodes. include news, event and market
    ﬁed graph neural framework for price movements prediction.                nodes, which only connect to their subject companies.
  • In order to follow highly regulated processes in the ﬁnancial in-
    dustry, we design and implement a two-phase attention mech-                   Multi-modality inputs are seemed as nodes in a heterogeneous
    anism to infer the interpretability from both the inner-modality          graph, in which they can pass messages to other nodes via links.
    and inter-modality sources. We also validate the effectiveness            A company might be one of source, target or bridge node, while
    of designed attention technologies in learning the internal se-           attributed (news, event, market) nodes only connect to its subject
    quential patterns and inter-source lead-lag relations through             company. For example, the market node (M) of Apple only connect
    empirical studies.                                                        target node Apple, as shown in Fig. 1.
  • Extensive experimental results on 3714 stocks demonstrate the             Deﬁnition 6. Edges (E) are a set of links connecting between
    superior performance of our proposed method. Furthermore,                 nodes, which include directed and undirected edges. The relation-
    Our model has been deployed in a major ﬁnancial service                   ship among companies (source, target or bridge nodes) are di-
    provider of China and we validate its performance of prof-                rected, which arrow from the subject to the object. The connection
    itability and interpretability in real-world scenarios. The source        between company and its attributed nodes are undirected.
    codes will be released in near future.
                                                                                  Fig. 1 shows a running example of heterogeneous graph and
                                                                              multi-modality inputs. When an event (or news) “Qualcomm suits
2. Preliminaries                                                              against Apple” hit the market, we extract the relation between
                                                                              its subject (Qualcomm) and object (Apple), and establish an edge
   In this section, we introduce the background of lead-lag effect            (suit against) directed from the subject (Qualcomm) to the ob-
and the construction process of heterogeneous graphs.                         ject (Apple). Then, if we want to forecast Apple’s price movements

                                                                          2
D. Cheng, F. Yang, S. Xiang et al.                                                                                              Pattern Recognition 121 (2022) 108218




                                         Fig. 1. An illustration of multi-modality inputs and heterogeneous graph.



in the following day, we set it as target node and extract multi-                   In the implementation, we employ a pretrained BERT1 [19] as
modality inputs accordingly, which includes event (news) seman-                 our news embedding model, and ﬁnetune BERT model from our
tic embeddings, linked source node (Qualcomm) and its historical                large-scale ﬁnancial news corpus. For event tuple extraction, we
prices, target node’s (Apple) market data, and the relations (includ-           leverage the widely-used OpenIE [20] and utilize the embedding
ing edges and bridge nodes, such as Samsung, Motorola, Foxconn,                 of structured tuples learned by tensor neural network [21] as event
etc.) of source and target nodes. For example, in smart phone mar-              feature. In the FinKG construction, we employ OpenNRE2 to extract
ket, Samsung is competitor of Apple and a downstream customer                   relations from massive news text and store them in our knowledge
of Google. Thus, it is a bridge node of Google and Apple in this                graph FinKG. If the entity of an event (or news) is a listed company,
scenario. As shown in Fig. 1, normally, each event (news) is ac-                we mark them as the source node. The rest entities are denoted as
complished with a source node and a target node. We construct                   bridge nodes in the knowledge graph. When a set of events hit
heterogeneous graphs by multi-modality inputs of the linked nodes               the FinKG, we extract the adjacent nodes and corresponding re-
and corresponding relations in FinKG. The detailed methods of re-               lations of the mentioned entities as base graph. Then, we mark
lation and graph construction are presented in Section 3.1. These               the predicted stocks as target nodes. Afterwards, the news, events
informative inputs are then fed into MAGNN for joint and inter-                 and market data are linked to each entity and ﬁnally form the het-
pretable learning.                                                              erogeneous graph, as shown Fig. 2a. We update the heterogeneous
                                                                                graph by every trading day.


3. Methodology
                                                                                3.2. Inner-modality graph attention

   In this section, we ﬁrst introduce the general framework and
                                                                                   Given each modality input feature and the constructed hetero-
multi-modality inputs of our proposed approach, and then present
                                                                                geneous graph, inner-modality graph attention is designed to prop-
inner-modality graph attention and inter-modality source atten-
                                                                                agate and aggregate information from source nodes to the target
tion, respectively. Lastly, we introduce the target forecasting net-
                                                                                node. As shown in Fig. 2b, the inputs of InnGAT include the pre-
work and model optimization.                                                                                                         S
                                                                                trained embeddings of the source node ei φ and the target node ets ,
                                                                                where φ ∈ {n, e, p} denotes the modality type and i ∈ Ns indicates
                                                                                the ith neighbors of node S. Ns is the set of neighbors.
3.1. Model framework and inputs                                                     We design two-phase projections for mapping the multi-
                                                                                modality inputs into latent representations, named source projec-
    Fig. 2 shows the general framework of the proposed multi-                   tion and target projection. They are parameterized by weight ma-
                                                                                       φ                 φ
modality graph neural network for ﬁnancial time series forecasting.             trix WS ∈ Rdh ×dφ and WT ∈ Rdh ×dt , respectively. dφ , dt and dh de-
We construct the heterogeneous graph ﬁrst by the events, news,                  note the dimension of source node embedding, target node em-
relations in KG and the market data, as shown Fig. 2a. Then, multi-             bedding and projected hidden features. Then, a shared attention
modality inputs are fed into inner-modality graph attention layer               mechanism is introduced to compute node-level attention coeﬃ-
(InnGAT) in parallel, in which each modality input is learned by                cients, which is parameterized by a weight vector aφ ∈ R2dh . Fi-
InnGAT independently over the heterogeneous graph. The inter-                   nally, The inner-modality attention coeﬃcient for source type φ
modality source attention (IntSAT) takes the output of InnGAT and
learn high-order representations from all modalities. Finally, the
learned features are fed into a feed-forward and classiﬁcation net-               1
                                                                                      https://github.com/google-research/bert
work for target forecasting.                                                      2
                                                                                      https://github.com/thunlp/OpenNRE


                                                                            3
D. Cheng, F. Yang, S. Xiang et al.                                                                                                                                Pattern Recognition 121 (2022) 108218




Fig. 2. The general framework of the proposed multi-modality graph neural network. It includes multi-modality inputs, inner-modality graph attention layer, inter-modality
source attention layer and the target forecasting network. In the heterogeneous graph, the symbol of S, B, E, N, M, T denotes the source, bridge, events, news, market and
target nodes respectively.



between source node i and target node s is formulated as:                                                        Finally, we construct the representation of the target node reps
                                    φ          φ  S                                                           by the concatenation of the attention-weighted projected features
  Sφ       exp LeakyReLU aφ  [WT · ets WS · ei φ ]
                          (                      (              ||                ))
α    =                                                                                     ,       (1)       from all three modalities, formulated as:
                                φ T [WTφ · ets WSφ · ekφ ]
  si                                                   S
             Sφ exp LeakyReLU a   (                         (        ||                ))                     reps = [αsSn Wz zsSn ||αsSe Wz zsSe ||αs p Wz zs p ],
                                                                                                                                                       S      S
                                                                                                                                                                                                   (4)
         k∈N     s
                                                                                                                                         S
where aφ  represents transposition of aφ and || is the concatena-                                          where the αsSn , αsSe and αs p are the attention coeﬃcient of IntSAT.
tion operation.                                                                                               Wz denotes the learned weights and reps means the output repre-
   Afterward, we compute the output features of the target node                                               sentation of inter-modality source attention network.
S for modality type φ as weighted average of the source hidden
features with sigmoid function, which is formulated as:                                                       3.4. Target forecasting network and optimization
          ⎛                                     ⎞
                                                                                                                 Given the learned representation of target node from InnGAT
zs φ = σ ⎝                αsiφ WSφ ei φ ⎠,
 S                            S             S
                                                                                                    (2)       and IntSAT, we then employ a shallow neural network for the tar-
              i∈Ns
                     Sφ
                                                                                                              get price forecasting, as shown in Fig. 2d. In particular, we formu-
                                                                                                              late the forecasting task as a classiﬁcation problem, which means
            φ
where WS denotes the learned weights, and σ is the sigmoid func-                                              we divide the trend of price movements into three categories
tion. Ns indicates the neighbors set of node S. zs φ denotes the out-
                                                                          S
                                                                                                              {up, neural, down}. We will detailed describe the settings in exper-
put feature of InnGAT for node S in modality φ . In the implemen-                                             iment section. The forecasting network consists of two full connect
tation, we extend the InnGAT with multi-head attention in order                                               layers and one softmax layer. They are deﬁned as:
                                                                                                                               
to stabilize the learning process.                                                                            Yˆs = softmax NN f (Wn reps + bn )                                                   (5)
                                                                                                              where NN f denotes a shallow neural network with two-layers of
3.3. Inter-modality source attention
                                                                                                              full connection. Wn ∈ Rds ×dl and bn ∈ Rdl are the weight matrix and
    Inter-modality source attention (IntSAT) is proposed to selec-                                            bias respectively. dl is the number of target categories. In this pa-
tively aggregate the information from multi-modality sources for                                              per, we set the dl = 3.
target node representation. As illustrated in Fig. 2c, the inputs                                                 Finally, we deﬁne the loss function of the proposed model by
                                                                                                S             the cross-entropy of the likelihood in output layer as below:
of IntSAT for target node include the output features zs φ of In-
nGAT from all modalities, where φ ∈ {n, e, p}. In the inter-modality                                                         
                                                                                                                              dl

source attention network, a shared linear transformation param-                                               Ltarget = −               Ysc ln Yˆsc                                                (6)
eterized by a weight matrix Wz ∈ Rdr ×dz and a multi-source atten-                                                           s∈VT c=1

tion mechanism parameterized by a weight vector ar ∈ Rdr are em-                                             where Ysc is the ground-truth label of cth movement category for
ployed to compute source attention coeﬃcients, respectively. dz in-                                           stock s, which is marked as 1 for the “up” price movements, 0 for
                                                 S
dicates the dimension of zs φ and dr is the dimension of the trans-                                           the “neural” and −1 for the “down” movement, respectively. VT
formed hidden features. Mathematically, the attention coeﬃcient                                               denotes the set of target nodes.
of modality type φ for target node can be formulated by:                                                          Our proposed multi-modality graph neural network can be
                                                                                                              trained in an end-to-end manner by minimizing the classiﬁcation
                                            S
    S        exp(ar  · Wz zs φ )                                                                            cross-entropy loss. Theoretically, we can optimize the model by the
αs φ =                               
                                                        ,                                           (3)
            k∈ exp a
                    r        (           · Wz zs k )
                                                 S                                                            standard stochastic gradient descent process. In practice, we em-
                                                                                                              ploy Adam algorithm [22] as the optimizer of our model. We set
                                                                              S
where ar and Wz are the learned weights, and αs φ denotes the at-                                            the initial learning rate to 0.001, and the batch size to 64 by de-
tention coeﬃcient of modality type φ .                                                                        fault.

                                                                                                          4
D. Cheng, F. Yang, S. Xiang et al.                                                                                              Pattern Recognition 121 (2022) 108218


4. Experiments                                                                             Table 1
                                                                                           The comparison of forecasting performance.

   In this section, we conduct extensive experiments to validate                                                  Micro-F1   Macro-F1    Weighted-F1
the effectiveness of our proposed technologies. We introduce the                                Stock-LSTM        0.4540     0.4233      0.4489
data acquisition and experimental settings ﬁrst, and then report                                News-ATT          0.4551     0.4551      0.4502
the result of each experiment in turn.                                                          Stock-GAT         0.4656     0.4396      0.4654
                                                                                                Event-NTN         0.4720     0.4478      0.4718
                                                                                                MAGNN-G           0.4815     0.4607      0.4798
4.1. Datasets and experimental settings
                                                                                                MAGNN-S           0.4813     0.4604      0.4793
                                                                                                MAGNN-all         0.4838∗∗   0.4627∗∗    0.4825∗∗
    Data acquisition Generating informative datasets from massive
                                                                                           ∗∗
                                                                                             indicates that the improvements are statistically signiﬁ-
multi-modality sources is challenging in our experiment. To en-
                                                                                           cant for p < 0.01 judged by paired t-test.
sure fairness, we collect ﬁnancial events, news, market prices and
the knowledge graph for all 3714 public companies listed in China
A-shares market,3 from Jan 01 2018 to Dec 31 2019. In particu-              Stock-GAT [23], and Event-NTN [21]. All parameters are set based
lar, we crawl the public announcements and leverage event extrac-           on their default suggestions in the paper. For instance, Stock-LSTM
tion methods to construct events for list companies. There are total        is set as two layers with a hidden size of 100 and 50. Our method
143,884 structured events across 41 categories in our dataset, such         has two variations: MAGNN-G and MAGNN-S, which only employ
as seasonal/annual reports, asset restructuring, increase/decrease of       the inner-modality graph attention or the inter-modality source at-
credit ratings, change of the chairman or board members, produc-            tention alone. MAGNN-all denotes the full version of our proposed
tion accident, etc. For ﬁnancial news, we crawl information from            techniques.
87 major websites that cover most important reports in the mar-                 For the evaluation, we apply the Micro-F1, Macro-F1 and
ket. There are 5.13 million news during the time interval. We lever-        Weighted-F1 score to measure the performance of forecasting ac-
age named entity recognition (and linking) and neural relation ex-          curacy. For the constructed portfolio, we employ asset accumulate
traction technologies to extract entities and relations from raw            return (A Return), average daily return (D Return) and widely-used
texts. These linked listed companies are stored as nodes in the             Sharpe Ratio [24] as evaluation metrics. A Return is formulated as
knowledge graph, in which each relation is stored as an edge be-
tween nodes. Finally, there are total 5.26 million entities and 6.93                               pt − pt−1
                                                                                          1
million relations in the FinKG.                                             ARt =                          i      i
                                                                                                                      ,                                          (8)
    We gather the stock price data of China A-shares listed compa-                   |   St−1   | i∈St−1       pt−1
                                                                                                                i
nies from the Shanghai and Shenzhen Stock Exchange sources from
2018 to 2020, including 500 trading days. The daily market data in-         where St−1 denotes the set of stocks in portfolio at time t − 1. pti
cludes stock prices (open, close, high, and low) and the trading in-        is the price for stock i at time t and | · | denotes the number of
formation (trading volume and turnover rate) of that day for each           set items. Sharpe ratio (SR) is the average return earned in excess
stock. In the experiment, we remove the trading suspension stock            of the risk-free rate per unit of volatility, which is expressed as:
and untradable prices (such as limit-up, limit-down stocks) from            SR = (R p − R f )/σ p where R p is the return of the portfolio, R f is the
the dataset. In the China stock market, investors need to follow a          risk-free rate, σ p is the standard deviation of the portfolio’s excess
10% limit up-limit down mechanism strictly.                                 return. We use 1-year China Government Bond Yield4 as the risk
    Experimental settings We forecast the price movement into three         free rate.
categories {up, neural, down}. For the stock i in day t, the return
rate can be computed by Rr,i = pti / pt−1  − 1. We set the ground-
                                        i                                   4.2. Financial forecasting
truth label of the price movements as:
              up          Rr ≥ rup ,                                            In this section, we evaluate the forecasting accuracy of ﬁnancial
f ( Rr ) =    neural      rdown < Rr < rup ,                     (7)        time series, which is the main task of this paper. Table 1 reports
              down        Rr ≤ rdown                                        the Micro-F1, Macro-F1 and Weighted-F1 score of each approach.
                                                                            ∗∗ denotes that the improvements are statistically signiﬁcant for
where we set rup = 0.01 and rdown = −0.01. In our dataset, there
                                                                            p < 0.01 judged by paired t-test.
are 226,585 samples in “up” category, 327,851 “neural” and
                                                                                The ﬁrst four lines of Table 1 shows the classiﬁcation result
238,630 samples in “down” category.
                                                                            of compared baselines. It is clear that, Stock-LSTM and News-ATT
    In the experiment, we employ the data of the year 2018 as the
                                                                            are not satisfactory, demonstrating neither stock nor news alone
training set and evaluate the performance in the year 2019. Partic-
                                                                            could achieve optimal performance. Stock-GAT is slightly better
ularly, we construct features from multi-modality data in the re-
                                                                            than Stock-LSTM, proving the effectiveness of preserving graph
cent 60 trading days and apply the next day’s price movement as
                                                                            structure in a time series forecasting model. In all baseline, Event-
the label. Then, we apply a sliding window of each trading day and
                                                                            NTN is most competitive, which considerably outperforms New-
report the average result of 2019 in the experiment. In the trad-
                                                                            ATT. The process of extracting structured events from raw news
ing strategy settings, we simply buy the forecasted “up”, sell the
                                                                            shows useful in learning representative embeddings. Line 5 and 6
“down” equities, and keep no action on “neural” stocks. The trad-
                                                                            display the performance of the variations of our proposed method.
ing percentage is allocated by the linear weights of predicted prob-
                                                                            As we can see, MAGNN-S is similar to MAGNN-G. Both are bet-
ability. Please note that there are many techniques for developing
                                                                            ter than the most competitive baselines. The validity of integrating
a trading strategy, which is beyond the scope of this paper. We ig-
                                                                            multi-modality inputs in our task is strongly proved. It is essen-
nore the transaction costs of all compared methods for simplicity
                                                                            tial to design an innovative model to learn from the above sources,
and fairness in the experiment.
                                                                            which is the primary motivation of this paper. MAGNN-all outper-
    Compared methods and evaluation metrics We utilize the fol-
                                                                            forms all baselines, demonstrating its superiority in learning from
lowing widely used approaches as baselines to validate the effec-
                                                                            multi-modality inputs for ﬁnancial forecasting.
tiveness of our proposed method: Stock-LSTM [13], News-ATT [9],

  3                                                                          4
      https://www.investopedia.com/terms/a/a-shares.asp                          http://yield.chinabond.com.cn/


                                                                        5
D. Cheng, F. Yang, S. Xiang et al.                                                                                                      Pattern Recognition 121 (2022) 108218




Fig. 3. The accumulated returns gained in the test set (2019) by the proposed method and compared baselines. For better illustration, we divide it into four quarters view.



              Table 2                                                                   sition is simply set linear to the probability of forecasting model
              The return of portfolios with different method.
                                                                                        outputs.
                Methods          A Return   D Return     Sharpe ratio                       Table 2 reports the performance of investment portfolios con-
                Stock-LSTM       0.3002     0.0012       2.7919                         structed by our proposed method and other baselines. We can ob-
                News-ATT         0.3960     0.0015       3.5097                         serve that, in all evaluation metrics, our proposed technique out-
                Stock-GAT        0.5035     0.0018       3.4506                         performs the baseline signiﬁcantly. Particularly, Stock-LSTM and
                Event-NTN        0.5507     0.0019       3.4467                         News-ATT achieve lower performance in “A Return” and “Sharpe
                MAGNN-G          0.6521     0.0021       3.5082
                                                                                        Ratio”, which indicates the poor proﬁtable returns. By incorpo-
                MAGNN-S          0.6775     0.0021       3.5561
                MAGNN-all        0.8571∗∗   0.0027∗∗     3.7619∗∗                       rating the knowledge graph and structured events, the return of
                                                                                        Stock-GAT and Event-NTN is higher than classic baselines. The
                                                                                        same phenomenon is observed in the sharpe ratio metrics. The
                                                                                        last three rows display the result of our proposed method and its
   In this experiment, We would like to stress that the market
                                                                                        sub-models. MAGNN constantly performs better than all compared
trend prediction is very challenging and a small fraction of im-
                                                                                        methods in three widely-used evaluation metrics. The effective-
provement can already bring a large amount of revenue in the ﬁ-
                                                                                        ness of our proposed method on constructing proﬁtable portfolios
nancial industry. According to the practice from Marcos et al.,[25]
                                                                                        is strongly proved.
even 0.005 improvements in the prediction accuracy is very diﬃ-
                                                                                            To further evaluate the return of our proposed method through-
cult for new researchers, which could normally lead to over 12%
                                                                                        out the test time interval, we examined the accumulated return
excess proﬁts. Our method improves the best baselines over 1% in
                                                                                        in each trading day and report the compared results in Fig. 3. We
Table 1 and consequently leads to near 30% proﬁt improvements in
                                                                                        can observe that Stock-LSTM is very close to the market CSI 300
the accumulated returns, as reported in Table 2. Therefore, we can
                                                                                        index5 throughout the year 2019. News-ATT are fared better than
safely claim that our proposed methods signiﬁcantly outperform
                                                                                        Stock-LSTM. Since the end of 2019Q1, our method leads the returns
state-of-the-art baselines in the forecasting task.
                                                                                        and enlarge the gap in 2019Q2 with compared baselines, which is
                                                                                        noteworthy. We then conduct empirical studies with ﬁnancial do-
4.3. Performance of the portfolio
                                                                                        main experts on the showcase of return curves. The reason appears
                                                                                        to be that when the market goes down (in 2019Q2), our method
   In the constructed portfolio performance evaluation, we report
                                                                                        could forecast the “down” signal in advance, which is learned from
the asset return (A Return), averaged daily return (D Return) and
                                                                                        the implicitly lead-lag effects in multi-modality sources. As a re-
Sharpe Ratio ﬁrst. Then, we present the accumulated return curve
across the time interval of the test period. As described above, we
buy the predicted “up” stocks and sell the “down” ones. The po-                           5
                                                                                              http://www.csindex.com.cn/en/indices/index-detail/0 0 030 0


                                                                                    6
D. Cheng, F. Yang, S. Xiang et al.                                                                                                  Pattern Recognition 121 (2022) 108218




                           Fig. 4. The accumulated returns gained in the validation window (2020) by the proposed method and compared baselines.



sult, MAGNN performs the best performance ever since the end of                         which is also related to its own historical performance. Thus, we
2019Q1 and leads the superiority until the end of 2019Q4, with                          need to include multi-modality (news, event, market) sources as
over 80% of investment proﬁts.                                                          the input of our model. However, different sources contribute dif-
                                                                                        ferently. Inter-modality attention mechanism could automatically
4.4. Performance generalization                                                         learn their weights in price prediction and so that achieve the
                                                                                        state-of-the-art performance. Moreover, within each modality, the
    In order to observe the performance generalization of our pro-                      inner relation of different companies is also very important. For
posed method, we chose a longer duration to evaluate the port-                          example, an event “Microsoft buys LinkedIn” was reported on Jun
folio’s return compared with baselines. In particular, we train our                     13, 2016. Immediately, the price of LinkedIn raised 46.81% and Mi-
model by the historical data during 2019 and then predict the                           crosoft declined 3.2% on that day. Interestingly, the prices of Sales-
price movements in 2020. The trading strategy is set as same as                         force, which is the main competitor of LinkedIn, decreased over
previous experiment and the trading percentage is also allocated                        6% in the following two weeks. Salesforce is not the direct subject
by the linear weights of predicted probability. Fig. 4 report the                       of this event but it is also deep inﬂuenced. Inner-modality atten-
performance of accumulated returns by the proposed method and                           tion model could learn this graph-structured relations between the
compared baselines. As we can see, the Stock-LSTM is the lowest,                        input sources and target prediction. Therefore, our method could
which is very close to the market return at the end of 2020. While                      help to predict stock price movement more accurately and the ex-
the News-ATT and Stock-GAT perform better that Stock-LSTM and                           periment results strongly demonstrate its superior performance.
Market, proving the effectiveness of including news and graph re-                          Then, in order to explore the interpretability of our proposed
lations in stock price prediction task. In all baselines, Event-NTN is                  method, we visualize the attention weights of both inner-modality
most competitive. Our method achieve the best result in the ob-                         graph attention and inter-modality source attention in Fig. 5. We
servation window and steadily leading the performance since the                         locate each equity according to the predicted return (x-axis) and
beginning of the second quarter of 2020. The result of model gen-                       their situation in the heterogeneous graph or modality source (y-
eralization experiment by training data in 2019 is consistent with                      axis) in the heat map. Then, we color it by the averaged attention
the result of training the model with 2018, which demonstrates the                      weights in the forecasting model.
effectiveness and generalization of our proposed method.                                   Fig. 5 a displays the learned weights of InnGAT. As we can see,
                                                                                        equities with approximately three to six neighborhood nodes gen-
                                                                                        erally contribute more important in the model. Besides, there is no
4.5. Interpretability of attention model
                                                                                        noteworthy difference for node structures in different market sit-
                                                                                        uations (e.g., across the x-axis). The result proves that the return
   As described in the preliminary section, price movements of
                                                                                        of a node with about four neighbors in the heterogeneous graph is
stocks can be seemed as the reaction of ﬁnancial events or news,

                                                                                    7
D. Cheng, F. Yang, S. Xiang et al.                                                                                                      Pattern Recognition 121 (2022) 108218




Fig. 5. Visualization of the attentional coeﬃcients. X-axis denotes the daily return of equities. Y-axis denotes the number of neighbors in subgraph (a) and the modality type
in (b), in which N,E,M denotes the news, events and market prices.




Fig. 6. The interface of our proposed MAGNN in web-based portfolio management system, which is deployed in a major ﬁnancial service provider of China. We translate the
key information by handwritten orange and underlined words.



more likely to be inﬂuenced by adjacent nodes, which means the                            structed portfolio, and the leading events. The left navigator pro-
lead-lag effects are more prominent in this situation.                                    vides the effects of each modality sources, including news, events
    A more interpretable output is observed in Fig. 5b. By visualiz-                      and market prices, and the lead-lag status of stocks on selected
ing each modality’s attention weights, we ﬁnd that all sources (N,                        events. Fig. 6b reports the forecasting view on a typical equity
E, and M) contribute importantly to the forecasting model, which                          Spring Airlines (Code: 601021) Ltd., which is China’s ﬁrst and North
strongly proves the declaration of this paper. In addition, news                          Asia’s largest low fare airline. The upper part shows the ground-
performs similarly in different market situations, with a slightly                        truth stock’s price, and the lower part displays the predicted price
prominent small positive return. The same phenomenon is ob-                               movements ratio since Jan 01, 2020.
served in market modality. On the contrary, events’ contribution                              As we can see, our method successfully forecasted four signif-
is signiﬁcantly higher in large positive and negative returns than                        icant ﬂuctuations in advance of Spring Airlines. At last, we report
the small ones. The reason might be that the sharp change of eq-                          the portfolio’s performance details in Fig. 6c. The result shows that
uity is mostly driven by events, instead of routine news or price                         our method signiﬁcantly improves the returns with over 60% of ex-
data. The result demonstrates the effects of multi-modality input                         cess proﬁt. In addition, by learning the lead-lag effects from multi-
and the proposed attentional model.                                                       modality sources, our method could avoid large losses in the mar-
                                                                                          ket, which decreases the maximum drawdown from −16.08% to
4.6. System implementation and deployment                                                 −12.48%.
                                                                                              In the implementation, we employ distributed Scrapy as the
   We then deploy our method in real-world scenarios and eval-                            web crawler, Redis as the in-memory database. The proposed
uate its performance in the market tracking experiment. Fig. 6                            model is written in Tensorﬂow on Python and requires two hours
shows the interface of the portfolio management system of our                             for training on two pieces of Telsa P100 GPU. The integrated port-
proposed method. In the homepage navigation view (Fig. 6a), we                            folio management system is implemented by Spring Cloud micro-
can investigate the forecasted “up” and “down” stocks, the con-                           services and written in Java.

                                                                                      8
D. Cheng, F. Yang, S. Xiang et al.                                                                                           Pattern Recognition 121 (2022) 108218


5. Related works                                                           (6210070838), the Shanghai “Scientiﬁc and Technological Innova-
                                                                           tion Action Plan” High and New Technology Projects (19511101300).
    In this section, we introduce some works that are related to
our research, including ﬁnancial time series analysis and multi-           References
modality graph neural network.
    Financial time series analysis In recent decades, numerous works        [1] Worldbank, Market capitalization of listed domestic companies, 2020, Ac-
                                                                                cessed 16-March-2021, https://data.worldbank.org/indicator/CM.MKT.LCAP.CD.
have proposed to forecast the ﬁnancial time series [26]. Conven-
                                                                            [2] E.F. Fama, The behavior of stock-market prices, J. Bus. 38.1 (1965) 34–105 APA.
tional approaches includes autoregressive model, moving average             [3] M.L. O’Connor, The cross-sectional relationship between trading costs and
method, factor analysis [16], indicator optimization [27], etc. Af-             lead/lag effects in stock & option markets, Financ. Rev. 34 (4) (1999) 95–117.
                                                                            [4] A. Cowles 3rd, Can stock market forecasters forecast? Econometrica 1 (3)
terwards, the machine learning techniques have been employed
                                                                                (1933) 309–324.
for stock price prediction, such as support vector machine [28],            [5] S.J. Taylor, Modelling Financial Time Series[M], World Scientiﬁc, 2008.
boosting trees [29], and neural network, especially for the RNN             [6] C.Q. Cao, R.S. Tsay, Nonlinear time-series analysis of stock volatilities[J], J. Appl.
and LSTM [30]. Recently, some researchers demonstrated the effec-               Econom. 7 (S1) (1992) S165–S185.
                                                                            [7] R.D. Edwards, J. Magee, W.H. Bassetti, Technical Analysis of Stock Trends, 2012.
tiveness of leveraging unstructured text news and events to learn           [8] D. Cheng, Y. Tu, Z. Niu, L. Zhang, Learning temporal relationships between ﬁ-
representative embeddings for stock price prediction [31,32]. Ad-               nancial signals, in: 2018 IEEE International Conference on Acoustics, Speech
vanced transfer learning [33] and unsupervised learning [34] tech-              and Signal Processing (ICASSP), IEEE, 2018, pp. 2641–2645.
                                                                            [9] Z. Hu, W. Liu, J. Bian, X. Liu, T.-Y. Liu, Listening to chaotic whispers: a deep
niques are also introduced to learn the meaningful embeddings                   learning framework for news-oriented stock trend prediction, in: Proceedings
for time series analysis. However, existing approaches fail to learn            of the Eleventh ACM International Conference on Web Search and Data Mining,
the internal relations of stock movements on the lead-lag effects               WSDM ’18, Association for Computing Machinery, New York, NY, USA, 2018,
                                                                                pp. 261–269.
of events (or news). The majority of them only employ a single             [10] D. Tashiro, H. Matsushima, K. Izumi, H. Sakaji, Encoding of high-frequency or-
modality source for forecasting, which may dismiss much useful                  der information and prediction of short-term stock price by deep learning,
information.                                                                    Quant. Finance 19 (9) (2019) 1499–1506.
                                                                           [11] S. Bharathi, A. Geetha, Sentiment analysis for effective stock market prediction,
    Multi-modality and graph attention network Graph neural net-
                                                                                Int. J. Intell. Eng. Syst. 10 (3) (2017) 146–154.
work (GNN) has shown its superior performance for representing             [12] Y. Xu, S.B. Cohen, Stock movement prediction from tweets and historical prices,
graph-structured data [35,36]. The graph attention model improves               in: Proceedings of the 56th Annual Meeting of the Association for Computa-
                                                                                tional Linguistics (Volume 1: Long Papers), 2018, pp. 1970–1979.
the node representation by adjusting meaningful weights in the
                                                                           [13] T. Fischer, C. Krauss, Deep learning with long short-term memory networks for
aggregation process with adjacent nodes [37], indicating the im-                ﬁnancial market predictions, Eur. J. Oper. Res. 270 (2) (2018) 654–669.
portance of corresponding nodes [38] and the attributed relations          [14] S. Deng, N. Zhang, W. Zhang, J. Chen, J.Z. Pan, H. Chen, Knowledge-driven
[39]. GNN with attention mechanism has shown its effectiveness in               stock trend prediction and explanation via temporal convolutional network,
                                                                                in: Companion Proceedings of The 2019 World Wide Web Conference, 2019,
a wide range of ﬁelds, including ﬁnance [40], healthcare [41], com-             pp. 678–685.
puter vision [42], e-commerce [43,44] etc. Recently, some works            [15] D. Cheng, F. Yang, X. Wang, Y. Zhang, L. Zhang, Knowledge graph-based event
explore to apply GNN to learning from multi-modality inputs, such               embedding framework for ﬁnancial quantitative investments, in: Proceedings
                                                                                of the 43rd International ACM SIGIR Conference on Research and Development
as disease diagnosis by multi-modality images [45]. However, there              in Information Retrieval, 2020, pp. 2221–2230.
are few studies on ﬁnancial forecasting by multi-modality graph            [16] G. Ganeshapillai, J. Guttag, A. Lo, Learning connections in ﬁnancial time series,
neural network.                                                                 in: International Conference on Machine Learning, 2013, pp. 109–117.
                                                                           [17] W.S. Chan, Stock price reaction to news and no-news: drift and reversal after
                                                                                headlines, J. Financ. Econ. 70 (2) (2003) 223–260.
6. Conclusion                                                              [18] R. Hussein, D. Yang, P. Cudré-Mauroux, Are meta-paths necessary? Revisit-
                                                                                ing heterogeneous graph embeddings, in: Proceedings of the 27th ACM In-
                                                                                ternational Conference on Information and Knowledge Management, 2018,
    In this paper, we propose a novel multi-modality graph neu-                 pp. 437–446.
ral network for ﬁnancial time series forecasting. Our method ad-           [19] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, Bert: pre-training of deep bidirec-
dresses the key problem of price prediction in the ﬁnancial in-                 tional transformers for language understanding, 2018arXiv:1810.04805
                                                                           [20] S. Saha, et al., Open information extraction from conjunctive sentences, in:
dustry, interpretable learning the lead-lag effects with informa-
                                                                                Proceedings of the 27th International Conference on Computational Linguis-
tive source, by inner-modality graph attention and inter-modality               tics, 2018, pp. 2288–2299.
source attention mechanism. We thoroughly evaluate the proposed            [21] X. Ding, K. Liao, T. Liu, Z. Li, J. Duan, Event representation learning enhanced
method’s effectiveness by comparing it with the state-of-the-art                with external commonsense knowledge, in: Proceedings of the 2019 Confer-
                                                                                ence on Empirical Methods in Natural Language Processing and the 9th In-
baselines on the massive historical datasets. In addition, we de-               ternational Joint Conference on Natural Language Processing (EMNLP-IJCNLP),
ploy the model in real-world applications, and the result proves                2019, pp. 4896–4905.
that our work could avoid signiﬁcant ﬁnancial investment losses.           [22] D.P. Kingma, J. Ba, Adam: a method for stochastic optimization,
                                                                                arXiv preprint arXiv:1412.6980(2014).
    In conclusion, this is the ﬁrst work to study the ﬁnancial time        [23] P. Velickovic, A. Casanova, P. Lio, G. Cucurull, A. Romero, Y. Bengio, Graph at-
series forecasting problem by advanced GNN techniques with in-                  tention networks, in: 6th International Conference on Learning Representa-
formative sources, which may innovate more studies on both the                  tions, ICLR 2018 - Conference Track Proceedings, 2018.
                                                                           [24] O. Ledoit, M. Wolf, Robust performance hypothesis testing with the sharpe ra-
computer science and ﬁnance communities. On the one hand, we                    tio, J. Empir. Finance 15 (5) (2008) 850–859.
extend the graph attention model to multi-modality scenarios; on           [25] M.L. De Prado, Advances in Financial Machine Learning, John Wiley & Sons,
the other hand, we improve ﬁnancial forecasting with learning on                2018.
                                                                           [26] T.G. Andersen, R.A. Davis, J.-P. Kreiß, T.V. Mikosch, Handbook of Financial Time
alternative data.                                                               Series, Springer Science & Business Media, 2009.
                                                                           [27] Z. Li, D. Yang, L. Zhao, J. Bian, T. Qin, T.-Y. Liu, Individualized indicator for all:
Declaration of Competing Interest                                               stock-wise technical indicator optimization with stock embedding, in: Proceed-
                                                                                ings of the 25th ACM SIGKDD International Conference on Knowledge Discov-
                                                                                ery & Data Mining, 2019, pp. 894–902.
    The authors declare that they have no known competing ﬁnan-            [28] L.-J. Cao, F.E.H. Tay, Support vector machine with adaptive parameters in ﬁnan-
cial interests or personal relationships that could have appeared to            cial time series forecasting, IEEE Trans. Neural Netw. 14 (6) (2003) 1506–1518.
                                                                           [29] A. Agapitos, A. Brabazon, M. O’Neill, Regularised gradient boosting for ﬁnancial
inﬂuence the work reported in this paper.
                                                                                time-series modelling, Comput. Manag. Sci. 14 (3) (2017) 367–391.
                                                                           [30] A. Sagheer, M. Kotb, Unsupervised pre-training of a deep LSTM-based stacked
Acknowledgments                                                                 autoencoder for multivariate time series forecasting problems, Sci. Rep. 9 (1)
                                                                                (2019) 1–16.
                                                                           [31] D. Zhou, L. Zheng, Y. Zhu, J. Li, J. He, Domain adaptive multi-modality neural
   The work is supported by the National Key R&D Program of                     attention network for ﬁnancial forecasting, in: Proceedings of The Web Con-
China (2018YFB2100801), the National Science Foundation of China                ference 2020, 2020, pp. 2230–2240.


                                                                       9
D. Cheng, F. Yang, S. Xiang et al.                                                                                                          Pattern Recognition 121 (2022) 108218


[32] O.B. Sezer, M.U. Gudelek, A.M. Ozbayoglu, Financial time series forecasting             [44] D. Cheng, X. Wang, Y. Zhang, et al., Graph neural network for fraud detection
     with deep learning: a systematic literature review: 2005–2019, Appl. Soft                    via spatial-temporal attention[J], IEEE Trans. Knowl. Data Eng. (2020) https:
     Comput. 90 (2020) 106181.                                                                    //ieeexplore.ieee.org/abstract/document/9204584/.
[33] R. Ye, Q. Dai, Implementing transfer learning across different datasets for time        [45] X. Fang, Z. Liu, M. Xu, Ensemble of deep convolutional neural networks based
     series forecasting, Pattern Recognit. 109 (2021) 107617.                                     multi-modality images for Alzheimer’s disease diagnosis, IET Image Proc. 14
[34] H. Wang, Q. Zhang, J. Wu, S. Pan, Y. Chen, Time series feature learning with                 (2) (2019) 318–326.
     labeled and unlabeled data, Pattern Recognit. 89 (2019) 55–66.
[35] N. Dehmamy, A.-L. Barabási, R. Yu, Understanding the representation power               Dawei Cheng is an assistant professor with the department of computer science
     of graph neural networks in learning graph topology, in: Advances in Neural             and technology, Tongji University, Shanghai, China. Before that, Dawei was a post-
     Information Processing Systems, 2019, pp. 15413–15423.                                  doctoral associate at MoE key lab of artiﬁcial intelligence, department of computer
[36] F. Manessi, A. Rozza, M. Manzo, Dynamic graph convolutional networks, Pat-              science and engineering, Shanghai Jiao Tong University. He received the Ph.D. De-
     tern Recognit. 97 (2020) 1070 0 0.                                                      gree in computer science from Shanghai Jiao Tong University, Shanghai, China, in
[37] B. Knyazev, G.W. Taylor, M. Amer, Understanding attention and generalization            2018. His research interests include graph learning, data mining and machine learn-
     in graph neural networks, in: Advances in Neural Information Processing Sys-            ing.
     tems, 2019, pp. 4202–4212.
[38] N. Park, A. Kan, X.L. Dong, T. Zhao, C. Faloutsos, Estimating node importance
     in knowledge graphs using graph neural networks, in: Proceedings of the 25th            Fangzhou Yang is a senior research scientist with the laboratory of artiﬁcial intelli-
     ACM SIGKDD International Conference on Knowledge Discovery & Data Min-                  gence (Seek-Data team), Emoney Inc., Shanghai, China. Fangzhou received his mas-
     ing, 2019, pp. 596–606.                                                                 ter degree from Technical University of Berlin and Shanghai Jiao Tong University.
[39] S. Feng, C. Xu, Y. Zuo, G. Chen, F. Lin, J. XiaHou, Relation-aware dynamic at-          Before that, he received his B.Sc. degree in computer science from Shanghai Jiao
     tributed graph attention network for stocks recommendation[J], Pattern Recog-           Tong University. His research interests include pattern recognition and data mining
     nit. (2021) 108–119, doi:10.1016/j.patcog.2021.108119.                                  in ﬁnance, portfolio management and quantitative investment.
[40] D. Cheng, Z. Niu, Y. Zhang, Contagious chain risk rating for networked-guaran-
     tee loans, in: Proceedings of the 26th ACM SIGKDD International Conference              Sheng Xiang is a Ph.D. candidate in the Center for Artiﬁcial Intelligence, major in
     on Knowledge Discovery & Data Mining, 2020, pp. 2715–2723.                              Computer Science, University of Technology Sydney, Australia. He received his B.Sc.
[41] E. Choi, M.T. Bahadori, L. Song, W.F. Stewart, J. Sun, Gram: graph-based atten-         degrees in Bioinformatics Engineering from Shanghai Jiao Tong University. His re-
     tion model for healthcare representation learning, in: Proceedings of the 23rd          search interests include big graph processing and analysis, query processing on data
     ACM SIGKDD International Conference on Knowledge Discovery and Data Min-                stream, uncertain data and graphs.
     ing, 2017, pp. 787–795.
[42] J. Wu, S.-h. Zhong, Y. Liu, Dynamic graph convolutional network for multi-              Jin Liu is a senior research scientist with the laboratory of artiﬁcial intelligence
     -video summarization, Pattern Recognit. 107 (2020) 107382.                              (Seek-Data team), Emoney Inc., Shanghai, China. He received his BSc and Master
[43] D. Cheng, S. Xiang, C. Shang, Y. Zhang, F. Yang, L. Zhang, Spatio-temporal at-          degree in computer science and engineering from Wuhan University, Hubei, China.
     tention-based neural network for credit card fraud detection, in: Proceedings           His research interests include ﬁnancial data mining, factor analysis and machine
     of the AAAI Conference on Artiﬁcial Intelligence, 34, 2020, pp. 362–369.                learning.




                                                                                        10

