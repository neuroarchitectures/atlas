# Graph Neural Network Based Anomaly Detection in Multivariate Deng Hooi 2021

> Source: `Graph_Neural_Network_Based_Anomaly_Detection_in_Multivariate_Deng_Hooi_2021.pdf`

---

                                    The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21)




   Graph Neural Network-Based Anomaly Detection in Multivariate Time Series
                                                      Ailin Deng, Bryan Hooi
                                                 National University of Singapore
                                          ailin@comp.nus.edu.sg, bhooi@comp.nus.edu.sg




                           Abstract                                       allow them to diagnose and respond to the anomaly as
                                                                          quickly as possible.
  Given high-dimensional time series data (e.g., sensor data),
  how can we detect anomalous events, such as system faults                  Due to the inherent lack of labeled anomalies in his-
  and attacks? More challengingly, how can we do this in a                torical data, and the unpredictable and highly varied na-
  way that captures complex inter-sensor relationships, and de-           ture of anomalies, the anomaly detection problem is typi-
  tects and explains anomalies which deviate from these rela-             cally treated as an unsupervised learning problem. In past
  tionships? Recently, deep learning approaches have enabled              years, many classical unsupervised approaches have been
  improvements in anomaly detection in high-dimensional                   developed, including linear model-based approaches (Shyu
  datasets; however, existing methods do not explicitly learn             et al. 2003), distance-based methods (Angiulli and Pizzuti
  the structure of existing relationships between variables, or           2002), and one-class methods based on support vector ma-
  use them to predict the expected behavior of time series. Our           chines (Schölkopf et al. 2001). However, such approaches
  approach combines a structure learning approach with graph
  neural networks, additionally using attention weights to pro-
                                                                          generally model inter-relationships between sensors in rela-
  vide explainability for the detected anomalies. Experiments             tively simple ways: for example, capturing only linear rela-
  on two real-world sensor datasets with ground truth anoma-              tionships, which is insufficient for complex, highly nonlin-
  lies show that our method detects anomalies more accurately             ear relationships in many real-world settings.
  than baseline approaches, accurately captures correlations be-             Recently, deep learning-based techniques have enabled
  tween sensors, and allows users to deduce the root cause of a           improvements in anomaly detection in high-dimensional
  detected anomaly.                                                       datasets. For instance, Autoencoders (AE) (Aggarwal 2015)
                                                                          are a popular approach for anomaly detection which uses
                     1    Introduction                                    reconstruction error as an outlier score. More recently, Gen-
                                                                          erative Adversarial Networks (GANs) (Li et al. 2019) and
With the rapid growth in interconnected devices and sensors               LSTM-based approaches (Qin et al. 2017) have also re-
in Cyber-Physical Systems (CPS) such as vehicles, indus-                  ported promising performance for multivariate anomaly de-
trial systems and data centres, there is an increasing need to            tection. However, most methods do not explicitly learn
monitor these devices to secure them against attacks. This is             which sensors are related to one another, thus facing diffi-
particularly the case for critical infrastructures such as power          culties in modelling sensor data with many potential inter-
grids, water treatment plants, transportation, and communi-               relationships. This limits their ability to detect and explain
cation networks.                                                          deviations from such relationships when anomalous events
   Many such real-world systems involve large numbers of                  occur.
interconnected sensors which generate substantial amounts
                                                                             How do we take full advantage of the complex rela-
of time series data. For instance, in a water treatment plant,
                                                                          tionships between sensors in multivariate time series? Re-
there can be numerous sensors measuring water level, flow
                                                                          cently, graph neural networks (GNNs) (Defferrard, Bresson,
rates, water quality, valve status, and so on, in each of their
                                                                          and Vandergheynst 2016) have shown success in modelling
many components. Data from these sensors can be related in
                                                                          graph-structured data. These include graph convolution net-
complex, nonlinear ways: for example, opening a valve re-
                                                                          works (GCNs) (Kipf and Welling 2016), graph attention net-
sults in changes in pressure and flow rate, leading to further
                                                                          works (GATs) (Veličković et al. 2017) and multi-relational
changes as automated mechanisms respond to the change.
                                                                          approaches (Schlichtkrull et al. 2018). However, applying
   As the complexity and dimensionality of such sensor data
                                                                          them to time series anomaly detection requires overcom-
grow, humans are increasingly less able to manually mon-
                                                                          ing two main challenges. Firstly, different sensors have very
itor this data. This necessitates automated anomaly detec-
                                                                          different behaviors: e.g. one may measure water pressure,
tion approaches which can rapidly detect anomalies in high-
                                                                          while another measures flow rate. However, typical GNNs
dimensional data, and explain them to human operators to
                                                                          use the same model parameters to model the behavior of
Copyright © 2021, Association for the Advancement of Artificial           each node. Secondly, in our setting, the graph edges (i.e. re-
Intelligence (www.aaai.org). All rights reserved.                         lationships between sensors) are initially unknown, and have


                                                                   4027
to be learned along with our model, while GNNs typically                Multivariate Time Series Modelling These approaches
treat the graph as an input.                                            generally model the behavior of a multivariate time series
   Hence, in this work, we propose our novel Graph Devi-                based on its past behavior. A comprehensive summary is
ation Network (GDN) approach, which learns a graph of                   given in (Blázquez-Garcı́a et al. 2020).
relationships between sensors, and detects deviations from                 Classical methods include auto-regressive models (Hauta-
these patterns1 . Our method involves four main components:             maki, Karkkainen, and Franti 2004) and the auto-regressive
1) Sensor Embedding, which uses embedding vectors to                    integrated moving average (ARIMA) models (Zhang et al.
flexibly capture the unique characteristics of each sensor;             2012; Zhou et al. 2018), based on a linear model given
2) Graph Structure Learning learns the relationships be-                the past values of the series. However, their linearity makes
tween pairs of sensors, and encodes them as edges in a                  them unable to model complex nonlinear characteristics in
graph; 3) Graph Attention-Based Forecasting learns to                   time series, which we are interested in.
predict the future behavior of a sensor based on an atten-                 To learn representations for nonlinear high-dimensional
tion function over its neighboring sensors in the graph; 4)             time series and predict time series data, deep learning-
Graph Deviation Scoring identifies and explains deviations              based time series methods have attracted interest. These
from the learned sensor relationships in the graph.                     techniques, such as Convolutional Neural Network (CNN)
   To summarize, the main contributions of our work are:                based models (Munir et al. 2018), Long Short Term Memory
• We propose GDN, a novel attention-based graph neural                  (LSTM) (Filonov, Lavrentyev, and Vorontsov 2016; Hund-
  network approach which learns a graph of the dependence               man et al. 2018a; Park, Hoshi, and Kemp 2018) and Gen-
  relationships between sensors, and identifies and explains            erative Adversarial Networks (GAN) models (Zhou et al.
  deviations from these relationships.                                  2019; Li et al. 2019), have found success in practical time
                                                                        series tasks. However, they do not explicitly learn the rela-
• We conduct experiments on two water treatment plant                   tionships between different time series, which are meaning-
  datasets with ground truth anomalies. Our results demon-              ful for anomaly detection: for example, they can be used to
  strate that GDN detects anomalies more accurately than                diagnose anomalies by identifying deviations from these re-
  baseline approaches.                                                  lationships.
• We show using case studies that GDN provides an ex-                      Graph-based methods provide a way to model the re-
  plainable model through its embeddings and its learned                lationships between sensors by representing the inter-
  graph. We show that it helps to explain an anomaly, based             dependencies with edges. Such methods include probabilis-
  on the subgraph over which a deviation is detected, atten-            tic graphical models, which encode joint probability distri-
  tion weights, and by comparing the predicted and actual               butions, as described in (Bach and Jordan 2004; Tank, Foti,
  behavior on these sensors.                                            and Fox 2015). However, most existing methods are de-
                                                                        signed to handle stationary time series, and have difficulty
                       2    Related Work                                modelling more complex and highly non-stationary time se-
                                                                        ries arising from sensor settings.
We first review methods for anomaly detection, and meth-
ods for multivariate time series data, including graph-based
approaches. Since our approach relies on graph neural net-              Graph Neural Networks In recent years, graph neural
works, we summarize related work in this topic as well.                 networks (GNNs) have emerged as successful approaches
                                                                        for modelling complex patterns in graph-structured data. In
Anomaly Detection Anomaly detection aims to detect un-                  general, GNNs assume that the state of a node is influenced
usual samples which deviate from the majority of the data.              by the states of its neighbors. Graph Convolution Networks
Classical methods include density-based approaches (Bre-                (GCNs) (Kipf and Welling 2016) model a node’s feature rep-
unig et al. 2000), linear-model based approaches (Shyu et al.           resentation by aggregating the representations of its one-step
2003), distance-based methods (Angiulli and Pizzuti 2002),              neighbors. Building on this approach, graph attention net-
classification models (Schölkopf et al. 2001), detector en-            works (GATs) (Veličković et al. 2017) use an attention func-
sembles (Lazarevic and Kumar 2005) and many others.                     tion to compute different weights for different neighbors
   More recently, deep learning methods have achieved                   during this aggregation. Related variants have shown suc-
improvements in anomaly detection in high-dimensional                   cess in time-dependent problems: for example, GNN-based
datasets. These include approaches such as autoencoders                 models can perform well in traffic prediction tasks (Yu, Yin,
(AE) (Aggarwal 2015), which use reconstruction error as an              and Zhu 2017; Chen et al. 2019). Applications in recommen-
anomaly score, and related variants such as variational au-             dation systems (Lim et al. 2020; Schlichtkrull et al. 2018)
toencoders (VAEs) (Kingma and Welling 2013), which de-                  and relative applications (Wang et al. 2020) verify the effec-
velop a probabilistic approach, and autoencoders combining              tiveness of GNN to model large-scale multi-relational data.
with Gaussian mixture modelling (Zong et al. 2018).                        However, these approaches use the same model param-
   However, our goal is to develop specific approaches for              eters to model the behavior of each node, and hence face
multivariate time series data, explicitly capturing the graph           limitations in representing very different behaviors of dif-
of relationships between sensors.                                       ferent sensors. Moreover, GNNs typically require the graph
                                                                        structure as an input, whereas the graph structure is initially
   1
       The code is available at https://github.com/d-ailin/GDN          unknown in our setting, and needs to be learned from data.


                                                                 4028
                                           1. Sensor Embedding                                             3.3   Sensor Embedding
                          Input:                                                                           In many sensor data settings, different sensors can have very
                                                            vi
                      N sensors              …
                                                                      ..
                                                                       .         N sensors                 different characteristics, and these characteristics can be re-
                                                                                                           lated in complex ways. For example, imagine we have two
                                          Time                                                             water tanks, each containing a sensor measuring the water
                                                                                                           level in the tank, and a sensor measuring the water quality
       2. Graph Structure Learning
                                                                                                           in the tank. Then, it is plausible that the two water level sen-
                               Learned Relations                                                           sors would behave similarly, and the two water quality sen-
                      …
            X1
                                                    3. Graph Attention-Based Forecasting                   sors would behave similarly. However, it is equally plausible
       X3
                 X2                                Attention-Based Features                  Forecast
                                                                                                           that sensors within the same tank would exhibit strong cor-
                                                            Z1
                                                                                                           relations. Hence, ideally, we would want to represent each
                                                                           ...
                                                                                                           sensor in a flexible way that captures the different ‘factors’
                                                                 Z2
                                                                                                           underlying its behavior in a multidimensional way.
       4. Graph Deviation Scoring                      Z3
                                                                                                              Hence, we do this by introducing an embedding vector
      Observation                   Prediction                                                             for each sensor, representing its characteristics:
                                                                                                                          vi ∈ Rd , for i ∈ {1, 2, · · · , N }
                                                                                                           These embeddings are initialized randomly and then trained
        Figure 1: Overview of our proposed framework.                                                      along with the rest of the model.
                                                                                                              Similarity between these embeddings vi indicates simi-
                                                                                                           larity of behaviors: hence, sensors with similar embedding
                          3        Proposed Framework                                                      values should have a high tendency to be related to one an-
                                                                                                           other. In our model, these embeddings will be used in two
3.1      Problem Statement                                                                                 ways: 1) for structure learning, to determine which sensors
 In this paper, our training data consists of sensor (i.e. mul-                                            are related to one another, and 2) in our attention mecha-
 tivariate time series) data from N sensorsh over Ttrain time           i                                  nism, to perform attention over neighbors in a way that al-
                                             (1)              (T )                                         lows heterogeneous effects for different types of sensors.
 ticks: the sensor data is denoted strain = strain , · · · , straintrain ,
 which is used to train our approach. In each time tick t, the                                             3.4   Graph Structure Learning
                          (t)
 sensor values strain ∈ RN form an N dimensional vector rep-                                               A major goal of our framework is to learn the relationships
 resenting the values of our N sensors. Following the usual                                                between sensors in the form of a graph structure. To do this,
 unsupervised anomaly detection formulation, the training                                                  we will use a directed graph, whose nodes represent sen-
 data is assumed to consist of only normal data.                                                           sors, and whose edges represent dependency relationships
    Our goal is to detect anomalies in testing data, which                                                 between them. An edge from one sensor to another indicates
 comes from the same N sensors but over a separate                                                         that the first sensor is used for modelling the behavior of the
hset of Ttest timei ticks: the test data is denoted stest =                                                second sensor. We use a directed graph because the depen-
    (1)             (T )
   stest , · · · , stesttest .                                                                             dency patterns between sensors need not be symmetric. We
    The output of our algorithm is a set of Ttest binary labels                                            use an adjacency matrix A to represent this directed graph,
 indicating whether each test time tick is an anomaly or not,                                              where Aij represents the presence of a directed edge from
 i.e. a(t) ∈ {0, 1}, where a(t) = 1 indicates that time t is                                               node i to node j.
 anomalous.                                                                                                   We design a flexible framework which can be applied ei-
                                                                                                           ther to 1) the usual case where we have no prior information
3.2      Overview                                                                                          about the graph structure, or 2) the case where we have some
                                                                                                           prior information about which edges are plausible (e.g. the
Our GDN method aims to learn relationships between sen-                                                    sensor system may be divided into parts, where sensors in
sors as a graph, and then identifies and explains deviations                                               different parts have minimal interaction).
from the learned patterns. It involves four main components:                                                  This prior information can be flexibly represented as a set
1. Sensor Embedding: uses embedding vectors to capture                                                     of candidate relations Ci for each sensor i, i.e. the sensors
   the unique characteristics of each sensor;                                                              it could be dependent on:
2. Graph Structure Learning: learns a graph structure rep-                                                                    Ci ⊆ {1, 2, · · · , N } \ {i}            (1)
   resenting dependence relationships between sensors;                                                     In the case without prior information, the candidate relations
3. Graph Attention-Based Forecasting: forecasts future                                                     of sensor i is simply all sensors, other than itself.
   values of each sensor based on a graph attention function                                                  To select the dependencies of sensor i among these can-
   over its neighbors;                                                                                     didates, we compute the similarity between node i’s embed-
                                                                                                           ding vector, and the embeddings of its candidates j ∈ Ci :
4. Graph Deviation Scoring: identifies deviations from the
                                                                                                                                   vi > vj
   learned relationships, and localizes and explains these de-                                                           eji =                for j ∈ Ci               (2)
   viations.                                                                                                                    kvi k · kvj k
Figure 1 provides an overview of our framework.                                                                         Aji = 1{j ∈ TopK({eki : k ∈ Ci })}             (3)


                                                                                                    4029
                                                                                          (t)
That is, we first compute eji , the normalized dot product be-              feature Wxi , and a is a vector of learned coefficients for
tween the embedding vectors of sensor i, and the candidate                  the attention mechanism. We use LeakyReLU as the non-
relation j ∈ Ci . Then, we select the top k such normalized                 linear activation to compute the attention coefficient, and
dot products: here TopK denotes the indices of top-k val-                   normalize the attention coefficents using the softmax func-
ues among its input (i.e. the normalized dot products). The                 tion in Eq. (8).
value of k can be chosen by the user according to the desired
sparsity level. Next, we will define our graph attention-based              Output Layer From the above feature extractor, we obtain
model which makes use of this learned adjacency matrix A.                                                                    (t)      (t)
                                                                            representations for all N nodes, namely {z1 , · · · , zN }.
                                                                                        (t)
3.5   Graph Attention-Based Forecasting                                     For each zi , we element-wise multiply (denoted ◦) it with
In order to provide useful explanations for anomalies, we                   the corresponding time series embedding vi , and use the re-
would like our model to tell us:                                            sults across all nodes as the input of stacked fully-connected
                                                                            layers with output dimensionality N , to predict the vector of
• Which sensors are deviating from normal behavior?
                                                                            sensor values at time step t, i.e. s(t) :
• In what ways are they deviating from normal behavior?                                          h                            i
                                                                                                           (t)             (t)
   To achieve these goals, we use a forecasting-based ap-                              ŝ(t) = fθ v1 ◦ z1 , · · · , vN ◦ zN             (9)
proach, where we forecast the expected behavior of each
sensor at each time based on the past. This allows the user to                 The model’s predicted output is denoted as ŝ(t) . We use
easily identify the sensors which deviate greatly from their                the Mean Squared Error between the predicted output ŝ(t)
expected behavior. Moreover, the user can compare the ex-                   and the observed data, s(t) , as the loss function for mini-
pected and observed behavior of each sensor, to understand                  mization:
why the model regards a sensor as anomalous.                                                           1
                                                                                                              Ttrain
                                                                                                              X                   2
   Thus, at time t, we define our model input x(t) ∈ RN ×w                             LMSE =                        ŝ(t) − s(t)             (10)
based on a sliding window of size w over the historical time                                      Ttrain − w t=w+1                2

series data (whether training or testing data):
                   h                                 i                      3.6    Graph Deviation Scoring
          x(t) := s(t−w) , s(t−w+1) , · · · , s(t−1)      (4)               Given the learned relationships, we want to detect and ex-
                                                                            plain anomalies which deviate from these relationships.
The target output that our model needs to predict is the sen-               To do this, our model computes individual anomalousness
sor data at the current time tick, i.e. s(t) .                              scores for each sensor, and also combines them into a sin-
                                                                            gle anomalousness score for each time tick, thus allowing
Feature Extractor To capture the relationships between                      the user to localize which sensors are anomalous, as we will
sensors, we introduce a graph attention-based feature extrac-               show in our experiments.
tor to fuse a node’s information with its neighbors based on                   The anomalousness score compares the expected behavior
the learned graph structure. Unlike existing graph attention                at time t to the observed behavior, computing an error value
mechanisms, our feature extractor incorporates the sensor                   Err at time t and sensor i:
embedding vectors vi , which characterize the different be-                                                      (t)
                                                                                                  Erri (t) = |si − ŝi |
                                                                                                                         (t)
                                                                                                                                              (11)
haviors of different types of sensors. To do this, we compute
node i’s aggregated representation zi as follows:                           As different sensors can have very different characteristics,
                                                                          their deviation values may also have very different scales.
     (t)                    (t)
                                   X               (t)
                                                                            To prevent the deviations arising from any one sensor from
    zi = ReLU αi,i Wxi +                αi,j Wxj  , (5)                   being overly dominant over the other sensors, we perform a
                                   j∈N (i)                                  robust normalization of the error values of each sensor:
         (t)                                                                                             Erri (t) − µ
                                                                                                                    ei
where xi     ∈ Rw is node i’s input feature, N (i) =                                           ai (t) =                ,             (12)
{j | Aji > 0} is the set of neighbors of node i obtained from                                                  σ
                                                                                                               ei
                                           d×w                              where µ ei and σei are the median and inter-quartile range
the learned adjacency matrix A, W ∈ R          is a trainable
weight matrix which applies a shared linear transformation                  (IQR2 ) across time ticks of the Erri (t) values respectively.
to every node, and the attention coefficients αi,j are com-                 We use median and IQR instead of mean and standard devi-
puted as:                                                                   ation as they are more robust against anomalies.
             (t)             (t)
                                                                               Then, to compute the overall anomalousness at time tick
           gi = vi ⊕ Wxi                                       (6)          t, we aggregate over sensors using the max function (we use
                                                                            max as it is plausible for anomalies to affect only a small
                                                        
                                             (t)    (t)
        π (i, j) = LeakyReLU a> gi ⊕ gj                        (7)          subset of sensors, or even a single sensor) :
                          exp (π (i, j))                                                            A (t) = max ai (t)                        (13)
           αi,j = P                              ,             (8)                                               i
                      k∈N (i)∪{i} exp (π (i, k))
                                                                               2
                                              (t)
                                                                                  IQR is defined as the difference between the 1st and 3rd quar-
where ⊕ denotes concatenation; thus gi concatenates the                     tiles of a distribution or set of values, and is a robust measure of the
sensor embedding vi and the corresponding transformed                       distribution’s spread.


                                                                     4030
   To dampen abrupt changes in values are often not per-                   Datasets     #Features     #Train     #Test    Anomalies
fectly predicted and result in sharp spikes in error values
even when this behavior is normal, similar with (Hundman                     SWaT            51        47515     44986      11.97%
et al. 2018b), we use a simple moving average(SMA) to gen-                   WADI           127       118795     17275       5.99%
erate the smoothed scores As (t).
   Finally, a time tick t is labelled as an anomaly if As (t)              Table 1: Statistics of the two datasets used in experiments
exceeds a fixed threshold. While different approaches could
be employed to set the threshold such as extreme value the-
ory (Siffer et al. 2017), to avoid introducing additional hy-             4.2   Baselines
perparameters, we use in our experiments a simple approach                We compare the performance of our proposed method with
of setting the threshold as the max of As (t) over the valida-            five popular anomaly detection methods, including:
tion data.                                                                • PCA: Principal Component Analysis (Shyu et al. 2003)
                                                                            finds a low-dimensional projection that captures most of
                    4    Experiments                                        the variance in the data. The anomaly score is the recon-
In this section, we conduct experiments to answer the fol-                  struction error of this projection.
lowing research questions:
                                                                          • KNN: K Nearest Neighbors uses each point’s distance
 • RQ1 (Accuracy): Does our method outperform baseline                      to its kth nearest neighbor as an anomaly score (Angiulli
   methods in accuracy of anomaly detection in multivariate                 and Pizzuti 2002).
   time series, based on ground truth labelled anomalies?
                                                                          • FB: A Feature Bagging detector is a meta-estimator that
 • RQ2 (Ablation): How do the various components of the
                                                                            fits a number of detectors on various sub-samples of the
   method contribute to its performance?
                                                                            dataset, then aggregates their scores (Lazarevic and Ku-
 • RQ3 (Interpretability of Model): How can we under-                       mar 2005).
   stand our model based on its embeddings and its learned
   graph structure?                                                       • AE: Autoencoders consist of an encoder and decoder
                                                                            which reconstruct data samples (Aggarwal 2015). It uses
 • RQ4 (Localizing Anomalies): Can our method localize                      the reconstruction error as the anomaly score.
   anomalies and help users to identify the affected sensors,
   as well as to understand how the anomaly deviates from                 • DAGMM: Deep Autoencoding Gaussian Model joints
   the expected behavior?                                                   deep Autoencoders and Gaussian Mixture Model to gen-
                                                                            erate a low-dimensional representation and reconstruction
4.1   Datasets                                                              error for each observation (Zong et al. 2018).
As real-world datasets with labeled ground-truth anomalies                • LSTM-VAE: LSTM-VAE (Park, Hoshi, and Kemp
are scarce, especially for large-scale plants and factories,                2018) replaces the feed-forward network in a VAE with
we use two sensor datasets based on water treatment phys-                   LSTM to combine LSTM and VAE. It can measure re-
ical test-bed systems: SWaT and WADI, where operators                       construction error with the anomaly score.
have simulated attack scenarios of real-world water treat-                • MAD-GAN: A GAN model is trained on normal
ment plants, recording these as the ground truth anomalies.                 data, and the LSTM-RNN discriminator along with
   The Secure Water Treatment (SWaT) dataset comes from                     a reconstruction-based approach is used to compute
a water treatment test-bed coordinated by Singapore’s Public                anomaly scores for each sample (Li et al. 2019).
Utility Board (Mathur and Tippenhauer 2016). It represents
a small-scale version of a realistic modern Cyber-Physical                4.3   Evaluation Metrics
system, integrating digital and physical elements to control
and monitor system behaviors. As an extension of SWaT,                    We use precision (Prec), recall (Rec) and F1-Score (F1)
Water Distribution (WADI) is a distribution system compris-               over the test dataset and its ground truth values to evalu-
ing a larger number of water distribution pipelines (Ahmed,               ate the performance of our method and baseline models:
Palleti, and Mathur 2017). Thus WADI forms a more com-                    F1 = 2×Prec×Rec                      TP
                                                                                  Prec+Rec , where Prec = TP+FP and Rec = TP+FN ,
                                                                                                                                  TP

plete and realistic water treatment, storage and distribution             and TP, TN, FP, FN are the numbers of true positives, true
network. The datasets contain two weeks of data from nor-                 negatives, false positives, and false negatives. Note that our
mal operations, which are used as training data for the re-               datasets are unbalanced, which justifies the choice of these
spective models. A number of controlled, physical attacks                 metrics, which are suitable for unbalanced data. To detect
are conducted at different intervals in the following days,               anomalies, we use the maximum anomaly score over the val-
which correspond to the anomalies in the test set.                        idation dataset to set the threshold. At test time, any time
   Table 1 summarises the statistics of the two datasets. In or-          step with an anomaly score over the threshold will be re-
der to speed up training, the original data samples are down-             garded as an anomaly.
sampled to one measurement every 10 seconds by taking the
median values. The resulting label is the most common label               4.4   Experimental Setup
during the 10 seconds. Since the systems took 5-6 hours to                We implement our method and its variants in Py-
reach stabilization when first turned on (Goh et al. 2016),               Torch (Paszke et al. 2017) version 1.5.1 with CUDA 10.2
we eliminate the first 2160 samples for both datasets.                    and PyTorch Geometric Library (Fey and Lenssen 2019)


                                                                   4031
                         SWaT                   WADI                                              SWaT                       WADI
      Method     Prec     Rec      F1    Prec     Rec      F1                 Method      Prec     Rec     F1    Prec         Rec        F1

      PCA        24.92   21.63   0.23   39.53     5.63   0.10                 GDN         99.35   68.12   0.81   97.50       40.19      0.57
     KNN          7.83    7.83   0.08    7.76     7.75   0.08                 - T OP K    97.41   64.70   0.78   92.21       35.12      0.51
       FB        10.17   10.17   0.10    8.60     8.60   0.09                   - E MB    92.31   61.25   0.76   91.86       33.49      0.49
       AE        72.63   52.63   0.61   34.35    34.35   0.34                     - ATT   71.05   65.06   0.68   61.33       38.85      0.48
   DAGMM         27.46   69.52   0.39   54.44    26.99   0.36
 LSTM-VAE        96.24   59.91   0.74   87.79    14.45   0.25           Table 3: Anomaly detection accuracy in term of perci-
 MAD-GAN         98.97   63.74   0.77   41.44    33.92   0.37           sion(%), recall(%), and F1-score of GDN and its variants.
        GDN      99.35   68.12   0.81   97.50    40.19   0.57

Table 2: Anomaly detection accuracy in terms of preci-
sion(%), recall(%), and F1-score, on two datasets with
ground-truth labelled anomalies. Part of the results are from
(Li et al. 2019).

                                                                                                                         2_FIC_101_C
                                                                                                                         2_FIC_101_CO
                                                                                                                         O
version 1.5.0, and train them on a server with Intel(R)                                                                  2_FIC_201_CO
                                                                                                                         2_FIC_201_C
                                                                                                                         2_FIC_301_CO
Xeon(R) CPU E5-2690 v4 @ 2.60GHz and 4 NVIDIA RTX                                                                        O    ...
                                                                                                                                ...
2080Ti graphics cards. The models are trained using the                                                                  2_FIC_301_C
                                                                                                                         O
Adam optimizer with learning rate 1 × 10−3 and (β1 , β2 ) =
(0.9, 0.99). We train models for up to 50 epochs and use
early stopping with patience of 10. We use embedding vec-
tors with length of 128(64), k with 30(15) and hidden lay-              Figure 2: A t-SNE plot of the sensor embeddings of our
ers of 128(64) neurons for the WADI (SWaT) dataset, corre-              trained model on the WADI dataset. Node colors denote
sponding to their difference in input dimensionality. We set            classes. Specifically, the dashed circled region shows local-
the sliding window size w as 5 for both datasets.                       ized clustering of 2 FIC x01 CO sensors. These sensors are
                                                                        measuring similar indicators in WADI.
4.5   RQ1. Accuracy
In Table 2, we show the anomaly detection accuracy in terms
of precision, recall and F1-score, of our GDN method and                  the graph structure learner enhances performance, espe-
the baselines, on the SWaT and WADI datasets. The results                 cially for large-scale datasets.
show that GDN outperforms the baselines in both datasets,
with high precision in both datasets of 0.99 on SWaT and                • The variant which removes the sensor embedding from
0.98 on WADI. In terms of F-measure, GDN outperforms the                  the attention mechanism underperforms the original
baselines on SWaT; on WADI, it has 54% higher F-measure                   model in both datasets. This implies that the embedding
than the next best baseline. WADI is more unbalanced than                 feature improves the learning of weight coefficients in the
SWaT and has higher dimensionality than SWaT as shown in                  graph attention mechanism.
Table 1. Thus, our method shows effectiveness even in un-               • Removing the attention mechanism degrades the model’s
balanced and high-dimensional attack scenarios, which are                 performance most in our experiments. Since sensors have
of high importance in real-world applications.                            very different behaviors, treating all neighbors equally in-
                                                                          troduces noise and misleads the model. This verifies the
4.6   RQ2. Ablation                                                       importance of the graph attention mechanism.
To study the necessity of each component of our method, we
gradually exclude the components to observe how the model               These findings suggest that GDN’s use of a learned graph
performance degrades. First, we study the importance of the             structure, sensor embedding, and attention mechanisms all
learned graph by substituting it with a static complete graph,          contribute to its accuracy, which provides an explanation for
where each node is linked to all the other nodes. Second,               its better performance over the baseline methods.
to study the importance of the sensor embeddings, we use
an attention mechanism without sensor embeddings: that is,              4.7     RQ3. Interpretability of Model
gi = Wxi in Eq. (6). Finally, we disable the attention mech-            Interpretability via Sensor Embeddings To explain the
anism, instead aggregating using equal weights assigned to              learned model, we can visualize its sensor embedding vec-
all neighbors. The results are summarized in Table 3 and                tors, e.g. using t-SNE(Maaten and Hinton 2008), shown on
provide the following findings:                                         the WADI dataset in Figure 2. Similarity in this embedding
• Replacing the learned graph structure with a complete                 space indicate similarity between the sensors’ behaviors, so
  graph degrades performance in both datasets. The effect               inspecting this plot allows the user to deduce groups of sen-
  on the WADI dataset is more obvious. This indicates that              sors which behave in similar ways.


                                                                 4032
                                              1_AIT_005_PV
                                                                            1_FIT_001_PV
                             1_MV_001_STATUS

                                                 1_LT_001_PV



                                         1_FIT_001_PV
                                                                            1_MV_001_STATUS




Figure 3: Left: Force-directed graph layout with attention weights as edge weights, showing an attack in WADI. The red triangle
denotes the central sensor identified by our approach, with highest anomaly score. Red circles indicate nodes with edge weights
larger than 0.1 to the central node. Right: Comparing expected and observed data helps to explain the anomaly. The attack
period is shaded in red.


   To validate this, we color the nodes using 7 colors cor-              Figure 3 (left). The large deviation at this sensor indicates
responding to 7 classes of sensors in WADI systems. The                  that 1 MV 001 STATUS could be the attacked sensor, or
representation exhibits localized clustering in the projected            closely related to the attacked sensor.
2D space, which verifies the effectiveness of the learned fea-              GDN indicates (in red circles) the sensors with highest at-
ture representations to reflect the localized sensors’ behavior          tention weights to the deviating sensor. Indeed, these neigh-
similarity. Moreover, we observe a group of sensors forming              bors are closely related sensors: the 1 FIT 001 PV neigh-
a localized cluster, shown in the dashed circled region. In-             bor is normally highly correlated with 1 MV 001 STATUS,
specting the data, we find that these sensors measure similar            as the latter shows the valve status for a valve which con-
indicators in water tanks that perform similar functions in              trols the flow measured by the former. However, the at-
the WADI water distribution network, explaining the simi-                tack caused a deviation from this relationship, as the at-
larity between these sensors.                                            tack gave false readings only to 1 FIT 001 PV. GDN fur-
                                                                         ther allows understanding of this anomaly by comparing the
Interpretability via Graph Edges and Attention Weights                   predicted and observed sensor values in Figure 3 (right):
Edges in our learned graph provide interpretability by in-               for 1 MV 001 STATUS, our model predicted an increase (as
dicating which sensors are related to one another. More-                 1 FIT 001 PV increased, and our model has learned that the
over, the attention weights further indicate the importance of           sensors increase together). Due to the attack, however, no
each of a node’s neighbors in modelling the node’s behav-                change was observed in 1 MV 001 STATUS, leading to a
ior. Figure 3 (left) shows an example of this learned graph on           large error which was detected as an anomaly by GDN.
the WADI dataset. The following subsection further shows a                  In summary: 1) our model’s individual anomaly scores
case study of using this graph to localize and understand an             help to localize anomalies; 2) its attention weights help to
anomaly.                                                                 find closely related sensors; 3) its predictions of expected
                                                                         behavior of each sensor allows us to understand how anoma-
4.8   RQ4. Localizing Anomalies                                          lies deviate from expectations.
How well can our model help users to localize and under-
stand an anomaly? Figure 3 (left) shows the learned graph                                     5   Conclusion
of sensors, with edges weighted by their attention weights,
and plotted using a force-directed layout(Kobourov 2012).                In this work, we proposed our Graph Deviation Network
   We conduct a case study involving an anomaly with                     (GDN) approach, which learns a graph of relationships be-
a known cause: as recorded in the documentation of the                   tween sensors, and detects deviations from these patterns,
WADI dataset, this anomaly arises from a flow sensor,                    while incorporating sensor embeddings. Experiments on two
1 FIT 001 PV, being attacked via false readings. These false             real-world sensor datasets showed that GDN outperformed
readings are within the normal range of this sensor, so de-              baselines in accuracy, provides an interpretable model, and
tecting this anomaly is nontrivial.                                      helps users to localize and understand anomalies. Future
   During this attack period,             GDN     identifies             work can consider additional architectures and online train-
1 MV 001 STATUS as the deviating sensor with the                         ing methods, to further improve the practicality of the ap-
highest anomaly score, as indicated by the red triangle in               proach.


                                                                  4033
                  Acknowledgments                                       lstms and nonparametric dynamic thresholding. In Proceed-
This work was supported in part by NUS ODPRT Grant                      ings of the 24th ACM SIGKDD international conference on
R252-000-A81-133. The datasets are provided by iTrust,                  knowledge discovery & data mining, 387–395.
Centre for Research in Cyber Security, Singapore Univer-                Hundman, K.; Constantinou, V.; Laporte, C.; Colwell, I.; and
sity of Technology and Design.                                          Soderstrom, T. 2018b. Detecting spacecraft anomalies using
                                                                        lstms and nonparametric dynamic thresholding. In Proceed-
                       References                                       ings of the 24th ACM SIGKDD international conference on
                                                                        knowledge discovery & data mining, 387–395.
Aggarwal, C. C. 2015. Outlier analysis. In Data mining,
237–263. Springer.                                                      Kingma, D. P.; and Welling, M. 2013. Auto-encoding varia-
Ahmed, C. M.; Palleti, V. R.; and Mathur, A. P. 2017. WADI:             tional bayes. arXiv preprint arXiv:1312.6114 .
a water distribution testbed for research in the design of se-          Kipf, T. N.; and Welling, M. 2016. Semi-supervised classi-
cure cyber physical systems. In Proceedings of the 3rd In-              fication with graph convolutional networks. arXiv preprint
ternational Workshop on Cyber-Physical Systems for Smart                arXiv:1609.02907 .
Water Networks, 25–28.
                                                                        Kobourov, S. G. 2012. Spring embedders and force directed
Angiulli, F.; and Pizzuti, C. 2002. Fast outlier detection              graph drawing algorithms. arXiv preprint arXiv:1201.3011
in high dimensional spaces. In European conference on                   .
principles of data mining and knowledge discovery, 15–27.
Springer.                                                               Lazarevic, A.; and Kumar, V. 2005. Feature bagging for out-
                                                                        lier detection. In Proceedings of the eleventh ACM SIGKDD
Bach, F. R.; and Jordan, M. I. 2004. Learning graphical                 international conference on Knowledge discovery in data
models for stationary time series. IEEE transactions on sig-            mining, 157–166.
nal processing 52(8): 2189–2199.
                                                                        Li, D.; Chen, D.; Jin, B.; Shi, L.; Goh, J.; and Ng, S.-K.
Blázquez-Garcı́a, A.; Conde, A.; Mori, U.; and Lozano, J. A.           2019. MAD-GAN: Multivariate anomaly detection for time
2020. A review on outlier/anomaly detection in time series              series data with generative adversarial networks. In Interna-
data. arXiv preprint arXiv:2002.04236 .                                 tional Conference on Artificial Neural Networks, 703–716.
Breunig, M. M.; Kriegel, H.-P.; Ng, R. T.; and Sander, J.               Springer.
2000. LOF: identifying density-based local outliers. In Pro-            Lim, N.; Hooi, B.; Ng, S.-K.; Wang, X.; Goh, Y. L.;
ceedings of the 2000 ACM SIGMOD international confer-                   Weng, R.; and Varadarajan, J. 2020. STP-UDGAT: Spatial-
ence on Management of data, 93–104.                                     Temporal-Preference User Dimensional Graph Attention
Chen, W.; Chen, L.; Xie, Y.; Cao, W.; Gao, Y.; and Feng,                Network for Next POI Recommendation. In Proceedings
X. 2019. Multi-range attentive bicomponent graph con-                   of the 29th ACM International Conference on Information
volutional network for traffic forecasting. arXiv preprint              & Knowledge Management, 845–854.
arXiv:1911.12093 .                                                      Maaten, L. v. d.; and Hinton, G. 2008. Visualizing data using
Defferrard, M.; Bresson, X.; and Vandergheynst, P. 2016.                t-SNE. Journal of machine learning research 9(Nov): 2579–
Convolutional neural networks on graphs with fast localized             2605.
spectral filtering. In Advances in neural information pro-              Mathur, A. P.; and Tippenhauer, N. O. 2016. SWaT: a water
cessing systems, 3844–3852.                                             treatment testbed for research and training on ICS security.
Fey, M.; and Lenssen, J. E. 2019. Fast Graph Representation             In 2016 International Workshop on Cyber-physical Systems
Learning with PyTorch Geometric. In ICLR Workshop on                    for Smart Water Networks (CySWater), 31–36. IEEE.
Representation Learning on Graphs and Manifolds.                        Munir, M.; Siddiqui, S. A.; Dengel, A.; and Ahmed, S.
Filonov, P.; Lavrentyev, A.; and Vorontsov, A. 2016. Mul-               2018. DeepAnT: A deep learning approach for unsupervised
tivariate industrial time series with cyber-attack simulation:          anomaly detection in time series. IEEE Access 7: 1991–
Fault detection using an lstm-based predictive data model.              2005.
arXiv preprint arXiv:1612.06676 .                                       Park, D.; Hoshi, Y.; and Kemp, C. C. 2018. A multimodal
Goh, J.; Adepu, S.; Junejo, K. N.; and Mathur, A. 2016. A               anomaly detector for robot-assisted feeding using an lstm-
dataset to support research in the design of secure water               based variational autoencoder. IEEE Robotics and Automa-
treatment systems. In International conference on critical              tion Letters 3(3): 1544–1551.
information infrastructures security, 88–99. Springer.                  Paszke, A.; Gross, S.; Chintala, S.; Chanan, G.; Yang, E.;
Hautamaki, V.; Karkkainen, I.; and Franti, P. 2004. Outlier             DeVito, Z.; Lin, Z.; Desmaison, A.; Antiga, L.; and Lerer,
detection using k-nearest neighbour graph. In Proceedings               A. 2017. Automatic differentiation in PyTorch. In NIPS-W.
of the 17th International Conference on Pattern Recogni-                Qin, Y.; Song, D.; Chen, H.; Cheng, W.; Jiang, G.; and
tion, 2004. ICPR 2004., volume 3, 430–433. IEEE.                        Cottrell, G. 2017. A dual-stage attention-based recurrent
Hundman, K.; Constantinou, V.; Laporte, C.; Colwell, I.; and            neural network for time series prediction. arXiv preprint
Soderstrom, T. 2018a. Detecting spacecraft anomalies using              arXiv:1704.02971 .


                                                                 4034
Schlichtkrull, M.; Kipf, T. N.; Bloem, P.; Van Den Berg, R.;
Titov, I.; and Welling, M. 2018. Modeling relational data
with graph convolutional networks. In European Semantic
Web Conference, 593–607. Springer.
Schölkopf, B.; Platt, J. C.; Shawe-Taylor, J.; Smola, A. J.;
and Williamson, R. C. 2001. Estimating the support of a
high-dimensional distribution. Neural computation 13(7):
1443–1471.
Shyu, M.-L.; Chen, S.-C.; Sarinnapakorn, K.; and Chang,
L. 2003. A novel anomaly detection scheme based on
principal component classifier. Technical report, MIAMI
UNIV CORAL GABLES FL DEPT OF ELECTRICAL
AND COMPUTER ENGINEERING.
Siffer, A.; Fouque, P.-A.; Termier, A.; and Largouet, C.
2017. Anomaly detection in streams with extreme value
theory. In Proceedings of the 23rd ACM SIGKDD Interna-
tional Conference on Knowledge Discovery and Data Min-
ing, 1067–1075.
Tank, A.; Foti, N.; and Fox, E. 2015. Bayesian struc-
ture learning for stationary time series. arXiv preprint
arXiv:1505.03131 .
Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Lio,
P.; and Bengio, Y. 2017. Graph attention networks. arXiv
preprint arXiv:1710.10903 .
Wang, Y.; Wang, W.; Ca, Y.; Hooi, B.; and Ooi, B. C.
2020. Detecting Implementation Bugs in Graph Convolu-
tional Network based Node Classifiers. In 2020 IEEE 31st
International Symposium on Software Reliability Engineer-
ing (ISSRE), 313–324. IEEE.
Yu, B.; Yin, H.; and Zhu, Z. 2017. Spatio-temporal graph
convolutional networks: A deep learning framework for traf-
fic forecasting. arXiv preprint arXiv:1709.04875 .
Zhang, Y.; Hamm, N. A.; Meratnia, N.; Stein, A.; Van
De Voort, M.; and Havinga, P. J. 2012. Statistics-based out-
lier detection for wireless sensor networks. International
Journal of Geographical Information Science 26(8): 1373–
1392.
Zhou, B.; Liu, S.; Hooi, B.; Cheng, X.; and Ye, J. 2019.
BeatGAN: Anomalous Rhythm Detection using Adversar-
ially Generated Time Series. In IJCAI, 4433–4439.
Zhou, Y.; Qin, R.; Xu, H.; Sadiq, S.; and Yu, Y. 2018. A
data quality control method for seafloor observatories: the
application of observed time series data in the East China
Sea. Sensors 18(8): 2628.
Zong, B.; Song, Q.; Min, M. R.; Cheng, W.; Lumezanu, C.;
Cho, D.; and Chen, H. 2018. Deep autoencoding gaussian
mixture model for unsupervised anomaly detection. In In-
ternational Conference on Learning Representations.




                                                                 4035

