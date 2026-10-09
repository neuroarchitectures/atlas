# Paper (Triebe et al. 2021)

> Source: `https://arxiv.org/abs/2111.15397`

---

NeuralProphet: Explainable Forecasting at Scale
Oskar Triebe a,1,∗, Hansika Hewamalagec, Polina Pilyuginad,
Nikolay Laptevb, Christoph Bergmeirc, Ram Rajagopala
aStanford University
bFacebook, Inc.
cMonash University
dSkoltech University
Abstract
We introduce NeuralProphet, a successor to Facebook Prophet, which set an industry standard
for explainable, scalable, and user-friendly forecasting frameworks. With the proliferation of time
series data, explainable forecasting remains a challenging task for business and operational decision
making. Hybrid solutions are needed to bridge the gap between interpretable classical methods
and scalable deep learning models. We view Prophet as a precursor to such a solution. However,
Prophet lacks local context, which is essential for forecasting the near-term future and is challenging
to extend due to its Stan backend.
NeuralProphet is a hybrid forecasting framework based on PyTorch and trained with standard
deep learning methods, making it easy for developers to extend the framework. Local context
is introduced with auto-regression and covariate modules, which can be conﬁgured as classical
linear regression or as Neural Networks. Otherwise, NeuralProphet retains the design philosophy
of Prophet and provides the same basic model components.
Our results demonstrate that NeuralProphet produces interpretable forecast components of
equivalent or superior quality to Prophet on a set of generated time series. NeuralProphet outper-
forms Prophet on a diverse collection of real-world datasets. For short to medium-term forecasts,
NeuralProphet improves forecast accuracy by 55 to 92 percent.
Keywords:
Explainable, Forecasting, Neural networks, Time series, Deep learning
∗Corresponding author. Email address: triebe@stanford.edu
1Maintainer of Github repository.
arXiv:2111.15397v1  [cs.LG]  29 Nov 2021

1. Introduction
1.1. Background
Time series data is prominent in most industrial sectors. Though extensively studied in theory
and applications such as economics, practical forecasting in industrial applications has not received
widespread attention until recently.
Since many companies have signiﬁcantly improved their data collection to the extent where data
availability now exceeds their data analytic capabilities, how to process and forecast industrial time
series is quickly becoming an important topic. Often the person tasked with solving a particular
issue is not a time series expert but rather a practitioner forecaster.
Accurate time series forecasting is essential for decision-making processes, especially in infras-
tructure planning, budget allocation and supply chain management.
Over-forecasts as well as
under-forecasts can be both be costly. In applications, where forecasts inform business or oper-
ational decisions with potentially fatal consequences, it is often a requirement for the model to
be explainable. A common approach is to break the forecast down into components which are
individually interpretable. This allows for the forecast to be reviewed by a domain expert who
may make adjustments when appropriate, a procedure known as human-in-the-loop.
Classical Time Series Methods. The ﬁeld of forecasting was traditionally dominated by statistical
techniques. This was most notable on many of the early forecasting competitions such as NN3,
NN5 and M3 (Makridakis & Hibon, 2000; Crone, 2008). Classical models such as Auto-Regressive
Integrated Moving Average (ARIMA) and Exponential Smoothing (ETS) have been well studied
and provide interpretable components.
However, their restrictive assumptions and parametric
nature limit their performance in real world applications. A skillful forecasting expert can transform
data and combine algorithms to satisfy speciﬁc conditions for better performance. This requires
deep domain knowledge in the application itself and in classical time series modelling.
Machine Learning Methods. Machine Learning (ML) based models were constantly performing
poorly at the early forecasting competitions. Neural Networks (NN) were once even labeled as
not being competitive for forecasting (Hyndman, 2018). They were further criticized in literature
for their black-box nature, as in Makridakis, Spiliotis & Assimakopoulos (2018). However, with
the explosion in availability of large scale time series, NN-based data-driven techniques regained
2

their popularity in forecasting. The amount of data available is no longer insuﬃcient for ML and
Deep Learning (DL) techniques to train well. Nevertheless, explainability of these models remains
mostly an open research problem in the ﬁeld of forecasting. Further, they often require substantial
engineering eﬀorts to preprocess data and ﬁne-tune hyperparameters.
Consequently, most non-expert forecasting practitioners involved in diﬀerent industries do not
use the most accurate state-of-the-art models for their particular task. Rather, they are more
interested in ﬁnding a reasonably accurate model, subject to explainability, scalability and minimal
tuning. These are commonly seen as requirements for forecasting applications in industry.
Practitioners tend to opt for traditional statistical techniques which are user-friendly and in-
terpretable in functionality, despite their poor forecasting performance. Hence, new methods need
to be developed which can bridge this gap between classical time series modelling and ML-based
methods.
Hybrid Methods. A precursor to hybrid methods, Facebook Prophet (Taylor & Letham, 2017),
provides an interpretable model which scales to many forecasting applications. The forecasting
framework provides full automation to novices and ﬁne-tuning capabilities to domain experts.
Prophet remains one of the ﬁrst forecasting packages which a data scientist will use, and also one
of the few which they often continue to use as their skills grow. It has made classic time series
forecasting approachable and useful to a wide demographic.
Yet, its limitations around key features, such as the lack of local context, and extensibility have
presented challenges for users. The lack of local context, which is essential for forecasting the near-
term future, has restricted the usefulness of Prophet in industrial applications. Because Prophet
was built on top of Stan (Carpenter, Gelman, Hoﬀman, Lee, Goodrich, Betancourt, Brubaker,
Guo, Li & Riddell, 2017), a probabilistic programming language, it is diﬃcult to extend the original
forecasting library.
Our Contribution. Inspired by Prophet’s impact on the forecasting community, it is our objective
to continue the democratization of forecasting tooling, by making hybrid models as accessible as
Prophet made classic models. As a step towards truly hybrid methods, we introduce NeuralProphet,
a user-friendly and interpretable forecasting tool built on PyTorch (Paszke, Gross, Massa, Lerer,
Bradbury, Chanan, Killeen, Lin, Gimelshein, Antiga, Desmaison, Kopf, Yang, DeVito, Raison,
Tejani, Chilamkurthy, Steiner, Fang, Bai & Chintala, 2019).
NeuralProphet fuses the classic
3

time series components introduced by the Prophet package with NN modules into a hybrid model
enabling it to ﬁt to non-linear dynamics. Two such NN components are the auto-regression and
covariate modules. By using PyTorch as the backend, NeuralProphet can be updated with the
latest innovation in deep learning. We provide a convenient tool for non-experts in forecasting by
pre-selecting strong defaults and automating many of the modelling decisions. Advanced users may
also ﬁnd the package useful as they can incorporate domain knowledge with many customizaiton
options. Overall, NeuralProphet abstracts a lot of forecasting domain knowledge and incorporates
ML best practices so users can focus on the task at hand.
in this paper. We introduce the NeuralProphet model and its components in detail and compare it
to Prophet. The methods section describes the experimental setup and evaluation. The synthetic
data results quantify the accuracy of the decomposed interpretable forecast components. The real-
world benchmarks demonstrate the realistic out-of-the-box forecasting performance on a diverse
set of univariate time series. Finally, we summarize the ﬁndings and provide suggestions on which
model to use depending on the task at hand.
2. The NeuralProphet Model
2.1. Model Components
A core concept of the NeuralProphet model is its modular composability. The model is com-
posed of modules which each contribute an additive component to the forecast. Most components
can also be conﬁgured to be scaled by the trend for a multiplicative eﬀect. Each module has its
individual inputs and modelling processes. However, all modules must produce h outputs, where
h deﬁnes the number of steps to be forecasted into the future at once. These are added up as the
predicted values ˆyt, ..., ˆyt+h−1 for the time series future values yt, ..., yt+h−1. If the model is only
time-dependent, an arbitrary number of forecasts can be produced. In the following descriptions,
that special case will be treated mathematically equivalent to a one-step ahead forecast with h = 1.
ˆyt = T(t) + S(t) + E(t) + F(t) + A(t) + L(t)
(1)
4

where,
T(t) = Trend at time t
S(t) = Seasonal eﬀects at time t
E(t) = Event and holiday eﬀects at time t
F(t) = Regression eﬀects at time t for future-known exogenous variables
A(t) = Auto-regression eﬀects at time t based on past observations
L(t) = Regression eﬀects at time t for lagged observations of exogenous variables
All model component modules can be individually conﬁgured and combined to compose the
model. If all modules are switched oﬀ, only a static oﬀset parameter is ﬁtted as the trend com-
ponent.
By default, only the trend and seasonality modules are activated.
The full model is
summarized in equation 1
In the following subsections we discuss each of the components in more detail.
2.1.1. Trend
A classic approach to modelling trend is to model it as the combination of an oﬀset m and
a growth rate k. The trend eﬀect at a time t1 is given by multiplying the growth rate by the
diﬀerence in time since the starting point t0 on top of the oﬀset m.
T(t1) = T(t0) + k · ∆t = m + k · (t1 −t0)
(2)
NeuralProphet models the trend component with this classic approach, but allows the growth
rate to change at a number of locations. Thus, the trend is modelled as a continuous piece-wise
linear series. This results in an interpretable, yet non-linear form of trend modelling. It is simple
to interpret, as in a segment between two points, the trend eﬀect is given by the steady growth
rate multiplied by the diﬀerence in time. We can generalize the trend by deﬁning a time-dependent
growth rate δ(t) and a time-dependent oﬀset ρ(t).
T(t) = δ(t) · t + ρ(t)
5

The piece-wise linear trend only varies the growth rate at a ﬁnite number of changepoints. A set
C of nc changepoints are deﬁned at diﬀerent times as C = (c1, c2, ..., cnc). Between changepoints,
the trend growth rate is kept constant. The ﬁrst segment’s growth rate and oﬀset are given as δ0
and ρ0 respectively. Rate adjustments at each changepoint can be deﬁned as a vector δ ∈RnC,
where δj is the rate change at the jth change-point. The growth rate at time t is determined by
adding the initial growth rate δ0 with the summation of the rate adjustments at all the change-
points up to time step t. Each growth rate change δj is a parameter to be ﬁtted on the data. A
corresponding vector of oﬀset adjustments can be deﬁned as ρ ∈Rnc. Similarly, the oﬀset at time
t is given by the initial oﬀset ρ0 and the sum of the oﬀset adjustments at each change-point up to
time t. However, the oﬀset for a changepoint cj is not an independent parameter but deﬁned as
ρj = −cjδj. This particular oﬀset deﬁnition makes the piece-wise linear series continuous. Further,
we introduce a binary vector Γ(t) ∈Rnc representing whether the time t is past each changepoint.
Thus, we can deﬁne a vectorized equation for the trend T(t) at time t, which is given in equation 3:
T(t) = (δ0 + Γ(t)T δ) · t + (ρ0 + Γ(t)T ρ)
(3)
where,
δ = (δ1, δ2, ..., δnC)
ρ = (ρ1, ρ2, ..., ρnC)
Γ(t) = (Γ1(t), Γ2(t), ..., Γnc(t))
Γj(t) =





1,
if t ≥cj
0,
otherwise
NeuralProphet provides a simple, semi-automatic mechanism for the selection of relevant change-
points. Given the number nC of desired change-points, nC equidistant points along the series are
selected as initial changepoints. Optionally, their growth change rate parameters can be regularized
during model training. This is is similar to a fully automatic changepoint selection, as only the
most relevant changepoints will be selected, if any. The user can also opt to manually deﬁne the
speciﬁc times of a custom number of changepoints. To avoid overﬁtting on a small number of ﬁnal
points, the ﬁnal trend segment (after the last changepoint) is set to a larger set of observations
6

(default: 15 % of training data). When making predictions into the unobserved future, the ﬁnal
growth rate is used to linearly extrapolate the trend.
2.1.2. Seasonality
Seasonality in NeuralProphet is handled by using Fourier terms (Harvey & Shephard, 1993) as
was originally done in Prophet (Taylor & Letham, 2017). In this technique, a number of Fourier
terms are deﬁned for each seasonality as in Equation 4, where k refers to the number of Fourier
terms deﬁned for the seasonality with periodicity p. Fourier terms are deﬁned as sine, cosine pairs
and allow to model multiple seasonalities as well as seasonalities having non-integer periodicities
such as yearly seasonality with daily data (p = 365.25) or with weekly data (p = 52.18). In a
multiple seasonality scenario, diﬀerent values for n can be deﬁned for each periodicity.
Sp(t) =
k
X
j=1

aj · cos
2πjt
p

+ bj · sin
2πjt
p

(4)
Fourier terms are a great tool for modelling seasonality as they produce smooth functions which
are simple to interpret and stable to ﬁt to data. However, Fourier terms only model deterministic
seasonal shapes which are assumed to be ﬁxed through time. A higher number of Fourier terms
allows the model to ﬁt a more complex seasonal pattern. Too much ﬂexibility may lead to overﬁtting
or to random patterns between observations. Thus, each Fourier term corresponds to a frequency
proportional to j
p, modelled by by a weighted combination of a sine and cosine transform. Every
seasonality is associated with 2k number of coeﬃcients. For time step t, the eﬀect from all the
seasonalities considered in the model can be indicated by S(t) in below Equation 5, where P refers
to the set of all the periodicities.
S(t) =
X
p∈P
S⋆
p(t)
(5)
Both additive and multiplicative seasonal patterns are supported. Each seasonal periodicity S⋆
p
can individually be conﬁgured to be multiplicative, in which case the component is multiplied by
the trend.
S⋆
p(t) =





S†
p(t) = T(t) · Sp(t),
if Sp is multiplicative
Sp(t),
otherwise
The framework automatically activates daily, weekly and or yearly seasonality depending on
data frequency and length. Each of these three types of seasonal periodicities is activated if the
7

data frequency is higher resolution than the respective periodicity, and if at least two full periods
of data are available. As an example, if the data is of daily frequency, the model will enable yearly
seasonality if the data spans two years or more. Weekly seasonality will also be added if two or
more weeks of data are available. Daily seasonality will not be activated, as the daily frequency is
not of higher resolution to allow for intra-day seasonality. Unless otherwise speciﬁed, the default
number of Fourier terms per seasonality are: k = 6 for p = 365.25 yearly, k = 3 for p = 7 weekly,
and k = 6 for p = 1 daily seasonality.
2.1.3. Auto-Regression
Auto-regression (AR) refers to the process of regressing a variable’s future value against its
past values, a critical part of many forecasting applications as can be seen in Hyndman & Athana-
sopoulos (2014). The number of past values included is usually referred to as the order p of the
AR(p) model. Hereby, a coeﬃcient θi is ﬁtted for each past value. Each coeﬃcient θi controls
the direction and magnitude of eﬀect of a particular past value on the forecast. In a classic AR
process, an intercept c and a white noise term ϵt are included.
yt = c +
i=p
X
i=1
θi · yt−i + et
In many applications, we are interested to predict multiple steps into the future. We refer to
the number of forecasts made at once as the forecast horizon h. A traditional AR model will only
produce one prediction for a one-step-ahead forecast with h = 1. If a multi-step ahead forecast of
length h > 1 is demanded, h models will have to be ﬁtted, one for each forecast step. The AR
module in NP is based on a modiﬁed version of AR-Net in Triebe, Laptev & Rajagopal (2019).
AR-Net is able to produce all h forecasts with one model, which can be of linear or non-linear
nature. In any conﬁguration, the p last observations of the target variable yt−1, yt−2, ..., yt−p, also
referred to as lags, are the inputs to the module. The outputs are h values corresponding to the
AR-eﬀect for each forecast step At(t), At(t + 1), ..., At(t + h −1). Hereby, At(t + 2) denotes the
predicted AR-eﬀect for forecast ˆyt+2 at t + 2, three steps in the future, predicted at forecast origin
t with observed data up to and including t −1.
At(t), At(t + 1), ..., At(t + h −1) = AR-Net(yt−1, yt−2, ..., yt−p)
(6)
8

It is important to note, that each time we forecast at a speciﬁc origin, we obtain h predictions.
Thus, for a given point in time, we have up to h diﬀerent predictions, each originating from a
diﬀerent forecast made in the past. They diﬀer from each other based on the data available to the
model at the time of the forecast. As an example, when we use an AR(5) model to forecast h = 3
steps into the future, at a given time t = 3, we will have 3 predicted values of the AR-eﬀect for
ˆyt=3. These three eﬀects At=1(t = 3), At=2(t = 3), ..., At=3(t = 3) are each an estimation of the
true AR-eﬀect A(t = 3), with diﬀerent ’age’. The oldest estimate At=1(t = 3) is 3 steps old, while
the most recent estimate At=3(t = 3) is one step old. With Auto-Regression, any estimate will be
at least one step old, as we assume to never know the true value of the series at the current time t
but only the observed value at the last step t −1 and older.
AR order. The most important parameter for this module is the number of past values to be
regressed over, also known as the order p of the AR(p) model. This parameter should be chosen
based on the approximate length of relevant context in past observations. In practice, it is hard to
determine accurately and is commonly set to twice the innermost periodicity or twice the forecast
horizon. Alternatively, a conservatively large order can be chosen when used in combination with
regularization to obtain a sparse AR-model.
Linear AR. The default AR-Net conﬁguration does not contain hidden layers, and is functionally
identical to a classic AR model. In practice, it is a single layer NN with p inputs, h outputs, no
biases, and no activation function. The single layer weights each regress a particular lag onto a
particular forecast step. Thus, each weight can be matched to a corresponding coeﬃcient of a
collection of h classic AR(p) models, making it simple to interpret the model.
y = Wx
where,
x = (yt−1, yt−2, ..., yt−p)
y = (At(t), At(t + 1), ..., At(t + h −1))
We can denote the model as a vector-matrix multiplication for the predicted AR eﬀects y ∈Rh,
with lagged observations as input x ∈Rp, and with weight matrix W ∈Rh×p, where Wi,j denotes
the coeﬃcent deﬁning the linear impact of lag yj on AR-eﬀect At(t = i)
9

Deep AR. AR-Net based AR module can model non-linear dynamics when hidden layers are con-
ﬁgured. In this case, the module trains a fully connected Neural Network (NN) with the speciﬁed
number of hidden layers and dimensions.
The addition of hidden layers may lead to a better
forecasting accuracy, however it a partial trade-oﬀin interpretability. Instead of being able to
directly quantify the contribution of a particular past observation to a particular prediction, we
can only observe the relative importance of a given past observation on all predictions. This can
be approximated by comparing the sums of the absolute weights of the ﬁrst layer for each input
position.
The p last observations of the time series are the inputs to the ﬁrst layer. After each hidden
layer, the logits are passed through an activation function, in our case, a rectiﬁed linear unit
(ReLU). The ﬁnal layer outputs h logits, are not transformed by an activation function and have
no bias. For l hidden layers with a hidden layer dimension of size d, we have:
a1 = fa(W1x + b1)
ai = fa(Wiai−1 + bi) for i ∈[2, ..., l]
y = Wl+1al
where,
fa(x) = ReLU(x) =





x,
x ≥0
0,
x < 0
Hereby, the layer biases all have the same dimensions b ∈Rd, while the layer weights are
W ∈Rd×d, except for the ﬁrst W1 ∈Rd×p and last Wl+1 ∈Rh×d layer weights.
Sparse AR. AR-Net demonstrated that the correct order can be approximated by setting the order
to a slightly larger than expected value when regularization is used to sparsify the model weights.
This allows for a more convenient model conﬁguration while retaining interpretability.
The regularization function proposed by the authors of AR-Net is given in equation 8. Though
we make this regularization function available, by default, our implementation uses a diﬀerent
regularization function, which we found to work better for a wider range of data. We use the
10

regularization function introduced in equation 14 with parameters ϵ = 3 and α = 1, where θ is the
vector of weights corresponding to the p AR model coeﬃcients:
ΛA(θ) = Λ(θ, ϵ = 3, α = 1) = 1
p
p
X
i=1
log( 1
3 · e + |θi|) + log(3) + 1
(7)
where,
θ =





W ∈Rp·h,
no hidden layers (l = 0)
W1 ∈Rp·d,
with hidden layers (l ≥1)
As an example, when forecasting the next 24 steps of an hourly time series, the forecaster could
choose to set the AR order to 100 with some regularization. After ﬁtting, the AR module may then
exhibit sparse coeﬃcients with the most signiﬁcant weights located on a few positions only. In this
example, only positions 1, 2, 3, 24, 48, and 72 could have non-zero weights. These positions can be
interpreted as the locations with an auto-correlation, and further, their weights can be examined
to study how diﬀerent combinations of past observation values aﬀect the forecast.
For reference, the original regularization function proposed in AR-Net is given by:
ΛAR−Net(θ, c1, c2) = 1
p
p
X
i=1
2 · (1 + exp(−c1 · |θi|
1
c2 )−1 −1
(8)
where the authors suggest setting c1 ≈3 and c2 ≈3.
2.1.4. Lagged Regressors
Lagged regressors are used to correlate other observed variables to our target time series. They
are often referred to as covariates.
Unlike future regressors, the future of lagged regressors is
unknown to us. At the time t of forecasting, we only have access to their observed, past values up
to and including t −1.
L(t) =
X
x∈X
Lx(xt−1, xt−2, ..., xt−p)
(9)
Given a set of covariates X ∈RT×nl, we create a separate lagged regressor module for each of
the m covariates x of lenght T. This allows to individually attribute the eﬀect of each covariate
on the predictions. Each lagged regressor module is functionally identical to the AR module, with
the only diﬀerence being the inputs. Here, the p last observations of the covariate x are the inputs
to the module (instead of the series y itself as in AR). The outputs are of identical form, each
11

module producing h additive components Lt
x(t), Lt
x(t + 1), ..., Lt
x(t + h −1) to the overall forecasts
ˆyt, ˆyt+1, ..., ˆyt+h−1.
Lt
x(t), Lt
x(t + 1), ..., Lt
x(t + h −1) = AR-Net(xt−1, xt−2, ..., xt−p)
(10)
All considerations regarding the order, depth, sparsiﬁcation, and interpretability of the AR-
Net modules for each covariate are identical to the AR module, as discussed in section 2.1.3, if we
substitute:
x = (xt−1, xt−2, ..., xt−p)
y = (Lt
x(t), Lt
x(t + 1), ..., Lt
x(t + h −1))
2.1.5. Future Regressors
To model future regressors, both past and future values of these regressors have to be known.
Given the set of future regressors as F ∈RT×nf , where nf is the number of regressors, the eﬀect
from all future regressors at time step t can be denoted by F(t) as in Equation 11, where df stands
for the coeﬃcient of the model for future regressor f ∈V. By default, future regressors have an
additive eﬀect, which can be conﬁgured to be multiplicative instead.
F(t) =
X
f∈F
F ⋆
f (t)
(11)
where,
Ff(t) = dff(t)
F ⋆
f (t) =





F †
f (t) = T(t) · Ff(t),
if f is multiplicative
Ff(t),
otherwise
2.1.6. Events and Holidays
Eﬀects from special events or holidays may occur sporadically. Such events are modelled anal-
ogous to future regressors, with each event e as a binary variable e ∈[0, 1], signaling whether the
event occurs on the particular day or not. For a set of events E ∈RT×ne with ne number of events
and the length of the series T, the eﬀect from all events at time step t can be denoted by E(t) in
Equation 12, where ze denotes the coeﬃcient of the model corresponding to the event e ∈E.
12

E(t) =
X
e∈E
E⋆
e(t)
(12)
where,
Ee(t) = zee(t)
E⋆
e(t) =





E†
e(t) = T(t) · Ee(t),
if e is multiplicative
Ee(t),
otherwise
NeuralProphet allows the modelling of two types of events; 1) user deﬁned 2) country speciﬁc
holidays. Given a country name, its national holidays are automatically retrieved and added to
the set of events E. Similar to seasonal eﬀects, events can also be speciﬁed as either additive or
multiplicative.
Additionally, for a given event at time te, a window [te −i, te +j] of i+j days can be conﬁgured
to be considered as special events of their own. For example, by setting a window of [−1, 0] for
Christmas day, will allow the day before Christmas to have its own eﬀect on the forecast. Hereby,
a new variable is created for each day within the window around event and added to the set of
events E.
2.2. Preprocessing
2.2.1. Missing Data
Missing data is less of an issue when working with non-lagged input variables, as corresponding
timesteps can simply be dropped. In doing so, one data sample will be lost per missing entry. With
Auto-regression or lagged regressor modules in use, a missing data point would lead to h + p data
samples being dropped due to h missing forecast targets and p missing lag values. For example, a
single missing point leads to the loss of 13 samples for a model forecasting h = 3 steps ahead with
AR(p = 10). Thus, we implemented a data imputation mechanism to avoid excessive data loss
when working with incomplete data. The imputation mechanism follows the following heuristics:
Data Imputation. If not speciﬁed or missing, events are assumed to not be happening. Missing
Events are ﬁlled in with zeros, indicating their absence. All other real-valued regressor variables,
including the time series itself, if autoregression is enabled, are imputed in a three step procedure.
13

First. The missing values are approximated by a bi-directional linear interpolation. Hereby, the
last and the ﬁrst known value before and after the missing values are used as anchor points for the
interpolation. This is done for up to 5 missing values in each direction. If there are more than 10
missing values, they will remain NAN after this step. The amount of missing values to interpolate
is user-settable.
Second. The remaining missing values are imputed with a centred rolling average. The rolling
average is computed over a window of 30, and ﬁlls at most 20 consecutive missing values. The
amount of missing values to ﬁll with a rolling average is user-settable.
Third. If there are more than 30 consecutive missing values, the imputation algorithm aborts and
instead drops all the missing datapoints.
2.2.2. Data Normalization
The type of normalization to apply to the time series can be set by the user. The available
options are described in table 1. If not speciﬁed, or set to ’auto’, the default option ’soft’ is used,
unless the time series values are binary, in which case ’minmax’ is applied.
Name
Normalization Procedure
’auto’
’minmax’ if binary, else ’soft’
’oﬀ’
bypasses data normalization
’minmax’
scales the minimum value to 0.0 and the maximum value to 1.0
’standardize’
zero-centers and divides by the standard deviation
’soft’
scales the minimum value to 0.0 and the 95th quantile to 1.0
’soft1’
scales the minimum value to 0.1 and the 90th quantile to 0.9
Table 1: The available data normalization options.
2.2.3. Tabularization
We tabularize the time series data to create a pseudo independent and identically distributed
dataset, as is required for SGD based training. Hereby, a data sample is created for each available
time-stamp of the target time series ’y’. A sample includes a normalized time-stamp, and all inputs
required by each conﬁgured module.
14

The modules Auto-regression and lagged covariates hereby extract the p values before a given
time-stamp from their respective time series and store them in a vector of size p serving as input
to the model. If forecasting multiple steps ahead with h > 1, the forecast targets also are stored
in a h-sized vector for each time-stamp.
This is not a memory-eﬃcient approach to prepare the dataset to training. Hoever, we chose
this procedure due to its simplicity and for its compute-eﬃciency when training.
Analogous, the Fourier terms are prepared by computing the 2k diﬀerent sine and cosine trans-
forms of the standardized time component, for each of the conﬁgured seasonal periodicities.
2.3. Training
One of the fundamental diﬀerences between Prophet and NeuralProphet is the ﬁtting procedure.
Prophet utilizes L-BFGS (Nocedal & Wright, 2006), implemented in Stan (Carpenter et al., 2017)
for ﬁtting model parameters to the data. NeuralProphet retools Prophet, from the bottom up,
replacing Stan with PyTorch, which is both ﬂexible and easy to use.
NeuralProphet relies on a modern version of mini-batch stochastic gradient descent (SGD).
This ﬁtting procedure is compatible with the vast majority of deep learning methods, can be
scaled reliably to large datasets, and can accommodate more complex model components. Any
model component which is trainable by SGD can be included as a module. This makes it easy for
developers to extend the framework with new features, and to adopt new research.
2.3.1. Loss Function
The default loss function is the Huber loss, also known as smooth L1-loss, which can be seen in
equation 13. Below a given threshhold β, it is equivalent to mean squared error (MSE). For values
larger than β it is equivalent to the mean absolute error (MAE). Compared to a pure MSE loss,
it is less sensitive to outliers and may help prevent exploding gradients Girshick (2015). We chose
β = 1 as threshold. Users can however choose to alternatively opt for MSE or MAE loss, or any
other available or self-implemented function matching the PyTorch loss function format as their
loss function.
Lhuber(y, ˆy) =





1
2β (y −ˆy)2,
for |y −ˆy| < β
|y −ˆy| −β
2 ,
otherwise
(13)
15

2.3.2. Regularization
We use a scaled and shifted log-transform of the absolute weight values as general regularization
function. For a module with model weights θ it is parametrized as follows:
Λ(θ, ϵ, α) = 1
n
n
X
i=1
log( 1
ϵ · e + α · |θi|) + log(ϵ) + 1
(14)
where ϵ ∈(0, ∞) sets the inverse of the oﬀset and α ∈(0, ∞) sets the scaling of the log-
transform. Higher values of ϵ lead to a steeper curve for weights close to zero. The parameter ϵ
could be described as a control of the eagerness to sparsify weights. Higher values of α lead to a
ﬂatter curve for weights of larger magnitude. The parameter α can be interpreted as a control of the
acceptance of large weights. As default values, we suggest ϵ = 1 and α = 1 for most applications.
Λ(θ, ϵ = 1, α = 1) = 1
n
n
X
i=1
log(1
e + |θi|) + 1
The regularization is applied in the per-module conﬁgured strength and added to the loss
function, to be back-propagated. The regularization only commences after a speciﬁed percentage
of training, by default, after 50%, and is thereafter linearly increased from zero to its full conﬁgured
strength at the end of training.
2.3.3. Optimizer
Analogous to the loss function, any optimizer matching the signature of a PyTorch optimizer
can be conﬁgured. As a reliable default, the AdamW optimizer (Loshchilov & Hutter, 2019) is
used, unless otherwise speciﬁed. In our testing we observed AdamW to ﬁt most reliably, with
a slight tendency to overﬁt.
We initialize AdamW with the conﬁgured learning rate, and set
β = (0.9, 0.999), ϵ = 1e −08, and a weight decay of 1e −04.
As a fallback option, we also provide a classic SGD optimizer with momentum set to 0.9, and
weight decay set to 1e −4. In our testing, we found SGD to lead to better validation performance,
but at the expense of occasional divergence.
If a user is willing to ﬁne-tune training related
hyperparameters, SGD poses an attractive alternative to AdamW.
2.3.4. Learning Rate
As we do not want to require users to be experts in machine learning, we make the learning
rate, an essential hyperparameter for any NN, optional. We achieve this by relying on a simple
16

but reasonably eﬀective approach to estimate a learning rate, introduced as a learning rate range
test in Smith (2017).
A learning rate range test is executed for 100 + log10(10 + T) ∗50) iterations, starting at
η = 1e −7, ending at η = 1e + 2 with the conﬁgured batch-size. After each iteration, the learning
rate is increased exponentially until the ﬁnal learning rate is reached in the last iteration. Excluding
the ﬁrst 10 and last 5 iterations, the steepest learning rate is deﬁned as the resulting learning rate.
The steepest learning rate is found by selecting the learning rate at the position which maximizes
the negative gradient of the losses.
In order to increase reliability, we perform the the test three times and take the log10-mean of
the three runs η1, η2, η3 as the selected learning rate η:
η⋆= 1
3(log10(η1) + log10(η2) + log10(η3))
η = 10η⋆
2.3.5. Batch Size
The batch size B is an optional parameter. If not user speciﬁed, the following heuristic will
determine the batch size based on the length T of the dataset:
B⋆= 22+⌊log10(T)⌋
B = min(T, max(16, min(256, B⋆)))
2.3.6. Training Epochs
The number of training epochs Nepoch is an optional parameter.
If not user speciﬁed, the
following heuristic will determine the number of epochs :
N⋆
epoch = 1000 · 2
5
2 ·log10(T)
T
Nepoch = min(500, max(50, ⌊N⋆
epoch⌋))
2.3.7. Scheduler
As all training related hyperparameters are automatically approximated, none of them are
optimal, and may thus potentially lead to training issues. Nonetheless, a user expects the model
17

to be reasonably ﬁtted to the data without issues. In order to meet these expectations despite
suboptimal hyperparameters, we rely on the training schedule called ’1cycle’ policy, which allows
for ’superconvergence’ of NN training, according to Smith & Topin (2018). Hereby, the initial
learning rate
η
100 is gradually increased up to the peak learning rate η, reached at 30% of training.
Thereafter, the learning rate is annealed along a cosine curve down to
η
5000 at the end of training.
2.4. Postprocessing
At the end of training or predicting, the normalized values on which the model operates are
transformed back into the initial distribution.
This, alongside with the many heuristically or
procedurally set optional hyperparameters abstract a lot of the machine learning speciﬁc and time
series speciﬁc decision making, making NeuralProphet approachable to forecasting practitioners.
2.4.1. Metrics
A few metrics are recorded by default: The conﬁgured loss function, RMSE, and MAE. Further,
any user deﬁned metrics will also be recorded over the training and validation runs.
RMSE =
v
u
u
t 1
T
T
X
i=1
(yi −ˆyi)2
(15)
MAE = 1
T
T
X
i=1
|yi −ˆyi|
(16)
where
T = size of data
2.4.2. Forecast Presentation
When used to predict, the model returns a dataframe with a column for each forecast com-
ponent. These are the additive (or multiplicative) contributions of each component. They are
individually displayed for each forecasted step ahead. Each of the components is referring to its
row’s datestamp-located target and ordered by the age of the foregast. e.g. ’yhat3’ refers to the
predicted y-value for the current location, based on data available three steps ago. Similarly, ’ar2’
refers to the AR-eﬀect predicted for the current location, based on data available two steps ago.
18

There are diﬀerent plotting utilities available for visualizing the forecast itself, the forecast
decomposition, the model parameters, and for speciﬁc forecast horizons of interest.
3. Experimental Setup
The experiments in this work serve to contrast the capabilities of NeuralProphet with Prophet.
First, we compare the accuracy of the interpretable forecast components of Prophet and Neural-
Prophet on a set of synthetic datasets. Second, we compare the performance of the two frameworks
on a number of real-world datasets. In the following, these experiments are described in detail.
3.1. Decomposition Demonstration on Synthetic Data
These experiments contrast the ability of Prophet and NeuralProphet to decompose a time
series into individual components. Since we are interested in the decomposition accuracy of the
diﬀerent modelling components, we must know their ground truth. Due to a lack of suitable real-
world datasets with known underlying components, we create synthetic datasets as the sum of
a selection of generated time series with added noise. This allows us to compare the estimated
decomposed components of both models to the original components.
3.1.1. Synthetic Dataset Composition
We generate the individual component time series to be identical to the type of components
that Prophet and NeuralProphet claim to model. Thus, ideally, both models should be able to
perfectly decompose the time series into its components, with some error due to additive noise
included in the synthetic datasets.
In Table 2 we display which components are included in the synthetic dataset of each exper-
imental scenario. The comparison between the two models focuses on the accuracy of individual
decomposed components. We also measure the overall accuracy and the compute time. Hereby,
the emphasis is on the accuracy of the individual components as this quantiﬁes the decomposition
accuracy of the models.
As many real-world datasets exhibit auto-regressive properties or depend on lagged regressors,
we also include three experiments with lagged components. However, only NeuralProphet claims
to be able to model Auto-Regression and Lagged regressor components. In experiments including
such lagged components, Prophet’s prediction for these components will be marked as zero.
19

Experiment
Trend
Seasonality
Events
Future reg.
AR
Lagged reg.
S-TS


S-EF


S-TSEF




S-mTSEF

†
†
†
S-AL


S-TSAL




S-TSEFAL






Table 2: Synthetic components included in the experimental scenarios. † indicates that the respective component
is multiplied by the trend.
Due to the freedom to choose oﬀsets in their additive components, the models may ﬁnd inter-
pretable components which match the dynamics of the underlying components, but shifted by an
oﬀset. Thus, before computing any component-wise metrics, we zero-center all components.
3.1.2. Generated Component Time Series
Each experiment’s dataset consists of 5 independently generated time series of daily resolution
with a length of 6000 samples each. Each time series is composed as the aggregate of multiple
generated time series, each representing a particular component type. A component time series
is scaled to range [0, 1] before being aggregated with the other components. The aggregate time
series is ﬁnally re-scaled to range [0, 1]. To denote the building blocks of the synthetic experiments
we use the following notation where each component is represented by a letter:
T: Trend. A piece-wise linear trend with one random change-point. Trend increases linearly before
the change-point, and decreases linearly after the change-point.
S: Seasonality. The generated seasonal component time series consists of Fourier terms combined
with random weights. The components are generated based on equation 4, with aj and bj drawn
from a random uniform distribution such that aj ∈[0, 1) and bj ∈[0, 1) and k = 5.
For the
experiments we create two independent seasonal components: monthly S30(t) and yearly S365(t).
Both are added to the aggregate series and both are individually evaluated in their decomposition
accuracy.
20

E: Events. The event component is a binary series with 25 occurrences of an event at random
locations. Analogous to equation 12, we have E = (e1) ∈RT×1, with e1 = (e1
1, ..., e1
T ) ∈RT where
e1
i = 1 if an event of type e1 occurs at timestamp ti and 0 otherwise. The initial eﬀect strength
ze = 1 will eﬀectively become ze ≥
1
Ncomponents after scaling the main series.
F: Future Regressor. The future regressor component is generated as an AR(3) process with co-
eﬃcients φ1 = 0.2, φ2 = 0.3, φ3 = −0.5 and additive white noise. After scaling the series to a
range of [0, 1], the regressor is added to the aggregate series with a weight of df = 1, which will be
df ≥
1
Ncomponents after scaling the main series.
A: Auto-Regression. The Auto-regressive component is drawn from an AR(2) process with coeﬃ-
cients φ1 = 0.3, φ2 = 0.3, and additive white noise. This series is generated independent of the the
other components. This is not identical to how NeuralProphet models the auto-regressive behavior.
NeuralProphet does not decompose the inputs to auto-regression module, but rather uses the raw
time series values as inputs.
L: Lagged Regressor. First, an independent time series x is generated with an AR(2) process,
analogous to future regressors. Next, we create the lagged regressor eﬀect series L(t) ∈RT , where
each eﬀect depends on a weighted combination of the last three observations from series x :
L(t) = c1 · xt−1 + c2 · xt−2 + c3 · xt−3
where c1, c2, ..., clag are weights from a random uniform distribution on (0, 1]. Finally, we utilize
the series L(t) as the lagged regressor component.
m: multiplicative mode. In multiplicative mode, we multiply each component element-wise by the
trend. For example, in S-mTSEF:
yt = T(t) + E†(t) + S†(t) + F †(t))
= T(t) · (1 + E(t) + S(t) + F(t))
3.2. Main Benchmark on Real-World Datasets
We evaluate the forecasting accuracy of Prophet and NeuralProphet on a diverse collection
of datasets covering many univariate forecasting tasks. The datasets span data lengths from one
21

hundred to half a million samples, while data record intervals span minutely to monthly frequencies.
This wide spectrum of univariate time series is in stark contrast to most benchmarking eﬀorts
focused on NN-based methods. The results represent realistic out-of-the-box performance across
applications such as energy demand, tourism, environmental factors, and retail sales among others.
In this section, we present the selected datasets and the evaluation procedure in more detail.
(a) Histogram of lengths.
(b) Counts of sampling frequencies.
(c) Counts of data types.
Figure 1: Histograms of diﬀerent characteristics of the datasets.
3.2.1. Selected Datasets
The datasets are selected to represent a wide range of univariate datasets. Figure 1a shows
that the lengths of datasets range from one hundred to one million samples, with most datasets
having around 10,000 samples. The data observation frequencies are distributed from monthly to
minutely sampling frequencies. Figure 1b bins minutely to half-hourly frequencies together and
displays the count of each type of frequency, with hourly frequencies being the most common.
Our collection of testing datasets covers a wide variety of subjects, including environmental,
energy-related, social, and economic data, with environmental and energy-related datasets being
the most common (Figure 1c).
Quantitative criteria for inclusion in our collection of testing datasets were: Univariate time
series of minutely to daily sampling frequency, length of 1000 samples or more, and permissive
license. We also include the four time series which Prophet uses as demonstration time series. Two
of those datasets do not meet the inclusion criteria, as they are of monthly frequency and much less
than 1000 samples long. They are included nonetheless, to make sure the datasets are not biased
towards the proposed model NeuralProphet, but rather towards the comparison model Prophet.
Further details on each dataset can be found in Table 8 (Appendix A).
22

(a) Boxplot of length for data frequencies.
(b) Boxplot of length for data types.
Figure 2: Distribution of data length for diﬀerent data frequencies and data types.
Sources. Datasets are taken from various sources including the UCI Machine Learning reposi-
tory (Dua & Graﬀ, 2017), Monash time series repository (Godahewa, Bergmeir, Webb, Hyndman
& Montero-Manso, 2021), and Prophet Github repository (Taylor & Letham, 2017). Additionally,
we collected and processed data from public energy and government webpages, including DOE
(2015), ERCOT (2021), RTE (2018), and OpenEI (2015).
Preprocessing. Most of the selected datasets do not contain missing values, but for those that do, we
impute the values using linear interpolation unless otherwise described. However, for datasets which
are mostly zero or near-zero-valued, such as data covering solar, parking, wind, and air quality
subjects, we ﬁll missing values with zeros. Missing time-stamps are also regarded as missing data.
Further, we manually preprocess each dataset and ﬁx minor misalignments, such as time-stamps
of 01:59 instead of 02:00 on data of hourly sampling frequency.
3.2.2. Benchmark Models
In this work, we compare the performance of time series forecasting packages which support
full model automation, oﬀer interpretability and scale to large datasets, without requiring domain
knowledge in machine learning or time series.
Prophet was the ﬁrst such package, which we
benchmark against its only fully featured successor, NeuralProphet.
Hyperparameters. Given that we aim to demonstrate the real world performance a non-expert
forecasting practitioner can reasonably expect, we do not manually tune any hyperparameters. We
23

only manually set hyperparameters in the synthetic benchmarking section if no automatic setting
is available, or if needed to make models comparable.
We mention any manually set parameters explicitly when discussing results. Occurrences of
manual model conﬁguration, but not hyperparameter tuning, include: When studying the eﬀect of
adding auto-regression to the model, we set diﬀerent numbers of lags p, which are all shown in the
results. For multi-step ahead forecasts, we further need to set the forecast horizon h. In a reverse
ablation study we further examine the impact of adding hidden layers to the model, hereby we also
set the learning rate to the same value of 0.001 for all of the model conﬁgurations and all datasets.
These ablation studies of number of lags, number of forecasts and neural network conﬁgurations
will also serve to quantify the sensitivity of NeuralProphet to parameter choices.
3.3. Forecast Evaluation
In the following, we describe the forecasting task and how the forecast performance is evaluated.
3.3.1. Forecast Horizon
In this study, we are interested in univariate time series forecasting covering typical forecasting
tasks. Given a desired number of steps to forecast into the future, we predict the next few consec-
utive values of the series, starting from the present timestamp. The number of next consecutive
values to be forecasted from the last observed value is also called forecast horizon. If auto-regression
is conﬁgured, the model will regress over the last few observations of the series itself. We treat
each given forecasting horizon as a separate forecasting task.
A forecasting task with a horizon of ten targets involves predicting the next ten targets starting
at the current timestamp t, i.e. obtaining a series {ˆyt, ˆyt+1, ..., ˆyt+9} containing the predictions over
the forecast horizon. In our empirical evaluation over real-world datasets, we evaluate the forecast
horizons of [∞, 1, 3, 15, 60]. Hereby, a horizon of ∞refers to forecasting the entire test set at once,
as is possible with model based on time-features only, such as Prophet and default NeuralProphet.
3.3.2. Expanding Origin Backtest
Evaluation procedures in time series forecasting are diverse in nature. Naming conventions have
only been establish for evaluations of classical time series models (Tashman (2000)). Conventions
for evaluating machine learning based time series forecasting models are currently being established.
We utilize a method based on a fusion of cross-validation and backtesting, which originate from
24

machine learning and trading algorithm evaluation. This method, called expanding origin backtest,
has recently been adopting by researchers and companies trying to evaluate machine learning based
models on time series data (Uber (2018, 2020)).
The procedure consists of an outer and an inner loop. The term expanding origin backtest
mostly refers to the outer loop, but is often used to describe the whole procedure. The inner
loop is usually based on a variation of the classical evaluation procedures. In our case, the inner
loop is a rolling origin forecast evaluation with non-constant holdout size ((Svetunkov, 2021)) with
updating but without reﬁtting of the model. In classical time series literature, reﬁtting is referred
to as reparametrization, while updating the model refers to updating the inputs to the model to
produce the next forecast without reﬁtting of model parameters. This means that a model will be
re-ﬁtted only k times per evaluation procedure.
A more verbose name for our forecast evaluation procedure could be: K-fold expanding origin
backtest with an inner rolling origin multi-step forecast evaluation with updating but without
reﬁtting of the model.
Train
Test
1 fold
Train
Test
2 fold
Train
Test
3 fold
Train
Test
4 fold
Train
Test
5 fold
time
Figure 3: Expanding Origin Backtest with 5 folds. Test sets contain 10% of the time series each.
Outer Loop: K-Fold Expanding Origin Backtest With Constant Holdout Size. Analogous to k-fold
cross-validation, the overall dataset is replicated into k folds, each containing training and testing
data, covering diﬀerent parts of the overall dataset. The model to be evaluated is ﬁtted separately
on each fold’s training data and evaluated on the same folds testing data. This signiﬁes that the
model
Unlike k-fold cross-validation, the folds are not randomized. Each fold has a training set with
values strictly occurring before the same fold’s test set, a visualization can be seen in Fig. 3.
25

We perform a 5-fold expanding origin backtest with test sets of 10 percent and a forward rolling
of the origin on 5 percent per fold. This means that each test fold will share 50 percent of their
data with the previous fold, and that each train fold will grow by 5 percent, while the size of the
test fold remains constant. Hereby, the ﬁrst fold will use the ﬁrst 70 percent of data for training
and the next 10 percent for test. The ﬁfth fold will utilize 90 percent of data for training and the
last 10 percent of data for testing. Thus, a model is tested ﬁve times on 10 percent of the data,
covering the last 30 percent of data over all tests.
Inner Loop: Rolling Origin With Updating But Without Reﬁtting. The inner loop is repeated for
each train-test fold. First, the model is ﬁtted over the train set. The test set evaluation is identical
to a rolling origin forecast evaluation with a non-constant holdout size ((Svetunkov, 2021)) with
updating but without reﬁtting of the model.
Hereby, the forecast origin rolls forward by 1 step at a time, without re-training the model
with the new data. This repeats through a single test fold shown in Figure 3 until the end of the
test fold is reached. Auto-regressive models relying on lagged observations as inputs are updated
with the new data point to produce the next forecast. If a model produces multiple forecasts per
forecast origin, the average accuracy of all forecast steps is recorded. For models with unlimited
forecast steps which do not depend on being updated with new observations, we forecast over the
entire test set at once.
The single-forecast version of this procedure is also known as time series cross-validation in
Hyndman & Athanasopoulos (2014), which we consider unfortunate naming, as it is not to be
confused with the outer loop (Expanding Origin Backtest) which is more closely related to k-fold
cross-validation.
3.3.3. Metrics
We calculate metrics on each i-step-ahead prediction versus corresponding actual values, and
then average this error over all steps. For evaluation we use two metrics: Mean Absolute Scaled
Error (MASE) and Root Mean Squared Scaled Error (RMSSE), as they are deﬁned in Hyndman
& Koehler (2006). MSSE is given as the RMSE of the evalutated method divided by RMSE of
the Na¨ıve prediction. This is the squared analogue of mean absolute scaled error (MASE). Hereby,
Na¨ıve refers to predicting the next observation to be identical to the previous value observed.
26

MASE and RMSSE metrics allow to evaluate the performance of a method compared to Na¨ıve
forecasting directly, as a value smaller one signiﬁes an improvement over Na¨ıve. Further, as the
metrics are scaled quantities, diﬀerent models can be compared across diﬀerent datasets.
MASE =
1
J
PT+J
j=T+1 |yi −ˆyi|
1
T−1
PT
i=2 |yi −yi−1|
(17)
RMSSE =
q
1
J
PT+J
j=T+1(yi −ˆyi)2
q
1
T−1
PT
i=2(yi −yi−1)2
(18)
where
T = size of train data
J = size of test data
3.4. Computational Resources
Unless otherwise mentioned, all models were trained and evaluated on a regular laptop with an
8-core CPU (2020 Macbook Pro with M1 chip).
4. Results
4.1. Demonstration of Interpretable Component Decomposition
The results are displayed separately for experiments S-TS, S-EF, S-TSEF, and S-mTSEF with-
out lagged components and for experiments S-AL, S-TSAL, and S-TSEFAL including lagged com-
ponents. In the former, we expect both models to perform identically, and in the latter, we expect
NeuralProphet to perform better.
On data without lagged components, NeuralProphet performs identical or slightly better than
Prophet, as can be seen in ﬁgure 4 and ﬁgure 5a. The overall goodness of ﬁt on the ’y’ component
(the sum of the components) is comparable or slightly better. The biggest diﬀerence is seen on
S-mTSEF with multiplicative components, where NeuralProphet has a signiﬁcantly lower error
and standard deviation of error.
On experiments involving lagged components, NeuralProphet performs signiﬁcantly better than
Prophet, as seen in ﬁgure 4 and ﬁgure 5a. Most of the diﬀerence comes from a better modelling of
lagged regressors.
27

(a) Experiments without lagged components.
(b) Experiments including lagged components.
Figure 4: RMSSE error of the model ﬁt on the synthetic time-series y itself. The plotted bar displays mean value
and the line displays standard deviation over 5 runs.
The ﬁt on the series itself is not as relevant as the component-wise decomposition accuracy, as
the overall goodness of ﬁt does not protect from overﬁtting. The component-wise accuracy, is an
adequate measure, as a model that can capture the true underlying dynamics will generalize well.
4.1.1. Component-Wise Decomposition Accuracy
Without Lagged Components. When compared on the component-wise accuracy NeuralProphet
performs identical or better than Prophet, as can be seen in ﬁgure 5a. Both models ﬁt the monthly
seasonality, events and future regressors near perfectly. Both models mostly struggle to ﬁnd a good
ﬁt for the yearly seasonality. This is likely explained by the limited number of full annual periods
in the data to ﬁt the model on. A signiﬁcant diﬀerence is observed in the goodness of ﬁt for the
trend component, where Neuralprophet exhibits a fraction of the error of Prophet.
Including Lagged Components. When modelling lagged components, NeuralProphet oﬀers a signif-
icant improvement decomposing lagged regressors. However, the AR component is not signiﬁcantly
more accurate than Prophet’s zero-prediction. This is due to the fact, the NeuralProphet does not
decompose the inputs to the AR module, but rather uses the raw time series values, while in this
experiment, we evaluate the performance based on an independently added AR process component.
On real applications, such a distinction largely does not matter.
Zooming in on the performance for individual experimental setups in Fig. 7 (Appendix B), we
28

(a) Experiments without lagged components.
(b) Experiments including lagged components.⋆
Figure 5: Component-wise RMSE of predicted forecast component to underlying true generated component value.
The plotted bar displays mean value and the line displays standard deviation over all synthetic datasets and over
5 runs.
⋆Note: Prophet does not support Auto-Regression and Lagged Regressor components. Prophet’s approxi-
mation of said components is implicitly assumed to be zero. Thus, the shown error for Prophet is equivalent to the
component’s standard deviation.
ﬁnd no notable new observations diﬀering from what we already discussed regarding ﬁgure 4 and
ﬁgure 5a.
4.1.2. Computational Time
Table 3 shows that training for NeuralProphet is on average 4.0 times slower compared to
Prophet.
However, both exhibit a large standard deviation, with Prophet having a long tail
upwards and NeuralProphet downwards. Prediction time exhibits the inverse relationship, with
NeuralProphet computing 13.5 times faster compared to Prophet. NeuralProphet has a tight spread
of the prediction times with no upward outliers, while Prophet has a large standard deviation with
a long tail upwards, as seen in ﬁgure 6.
Training Time
Prediction Time
Model
(seconds)
(seconds)
NeuralProphet
20.50 (±4.70)
0.16 (±0.05)
Prophet
5.07 (±2.30)
2.16 (±1.08)
Table 3: Computational time in seconds to train a model and time predict over entire dataset. Showing mean and
standard deviation over all folds for experiments without lagged components.
29

These observations mean that NeuralProphet may take some extra resources to be ﬁt, but will
need signiﬁcantly less resources in deployment. Further, as NeuralProphet is faster in computing
predictions by a magnitude, it makes it possible to deploy it in time-sensitive applications where
the next prediction needs to be reliably computed in a fraction of a second.
Figure 6: Training and prediction times across over all folds for experiments without lagged components.
Thanks to being fully implemented in PyTorch, the NeuralProphet model may be parallelized
in the future, allowing it to be deployed on GPUs. We would expect this to close the gap in training
time, and the gap in prediction time to widened.
4.2. Main Benchmark Results
The overall results are displayed in table 4. We would like to highlight three observations: First,
Prophet and NeuralProphet perform near identical in their default mode. However, both perform
signiﬁcantly worse than Naive one step ahead. Second, NeuralProphet forecasts improve substan-
tially when conﬁguring any amount of lags, consistently performing better than Naive forecasting
one step ahead. More lags generally lead to a better performance. Third, when forecasting multiple
steps ahead, even when we increase the horizon to 60 steps, NeuralProphet with any number of
lags still performs better than Prophet or NeuralProphet in default conﬁguration.
We also observe that the speciﬁc number of lags chosen has a small impact on the forecast
accuracy, suggesting that the model is not sensitive to optimal parameter choices.
All model parameters which were set for any of the models are explicitly displayed in Table 4:
number of forecast steps and number of lags. No hyperparameters were manually set or tuned.
30

MASE for diﬀerent forecast horizons
Model
1 step
3 steps
15 steps
60 steps
Prophet⋆1
8.54 (±2.17)
NeuralProphet⋆1
8.49 (±2.03)
NeuralProphet (1 lags)
0.83 (±0.09)
N/A
N/A
N/A
NeuralProphet (3 lags)
0.72 (±0.07)
N/A
N/A
N/A
NeuralProphet (12 lags)
0.69 (±0.07)
1.16 (±0.16)
N/A
N/A
NeuralProphet (30 lags)
0.62 (±0.07)
0.99 (±0.12)
2.07 (±0.30)
N/A
NeuralProphet (120 lags)⋆2
0.62 (±0.09)
0.94 (±0.12)
1.97 (±0.29)
3.77 (±0.81)
Table 4: Forecast error (MASE) for each forecast horizon across all datasets. The standard deviation is shown in
paranthesis. The corresponding RMSSE values are given in Table 9 (Appendix C). Note⋆1: The models without lags
can produce forecasts for an arbitrary number of forecasts. Note⋆2: Models with 120 lags were not evaluated over
monthly datasets due to their short length.
The choice of MASE or RMSSE as the error measure is perfectly adequate for the one step
ahead forecasting task, but not for multiple steps, as it only measures the naive model’s one step
ahead forecast error. When forecasting multiple steps ahead, an MASE value of 1 is no longer
equal to a naive models performance, as a naive model would likely have a higher error for multiple
steps ahead.
Nonetheless, the MASE error remains a useful quantity to compare the forecast
performance of diﬀerent models across diﬀerent datasets, as it scales the error into a somewhat
standardized space.
4.2.1. Ablation Study: Depth of Neural Network
We further examine the impact of diﬀerent Neural Network conﬁguration options on the forecast
accuracy for diﬀerent forecast horizons. Overall, most conﬁgurations perform within range of their
linear counterparts.
Except for the longest forecast horizon of 60 steps, all NN conﬁgurations
signiﬁcantly outperform their linear counterpart. The speciﬁc choices of NN parameters have no
discernible impact on the forecast accuracy. Thus, the model is insensitive to non-optimal NN
parameter choices, and for short term forecasts, also to lack of any hidden layers.
For this ablation study and any other mentions of a conﬁguration involving a NN, the learning
31

MASE for diﬀerent forecast horizons
Model
1 step
3 steps
15 steps
60 steps
(30 lags)
0.62 (±0.07)
0.99 (±0.12)
2.07 (±0.30)
N/A
(30 lags, 1x32 NN)
0.60 (±0.08)
0.93 (±0.12)
1.88 (±0.23)
N/A
(30 lags, 2x24 NN)
0.59 (±0.07)
0.91 (±0.13)
1.89 (±0.29)
N/A
(30 lags, 4x16 NN)
0.61 (±0.10)
0.93 (±0.12)
1.91 (±0.26)
N/A
(120 lags)
0.62 (±0.09)
0.94 (±0.12)
1.97 (±0.29)
3.77 (±0.81)
(120 lags, 1x32 NN)
0.63 (±0.11)
0.92 (±0.13)
1.81 (±0.24)
3.06 (±0.52)
(120 lags, 2x24 NN)
0.69 (±0.14)
0.90 (±0.12)
1.81 (±0.22)
3.07 (±0.57)
(120 lags, 4x16 NN)
0.66 (±0.11)
0.96 (±0.15)
1.86 (±0.31)
3.03 (±0.47)
Table 5: NeuralProphet (with NN) forecast error (MASE) for each forecast horizon across all datasets. The corre-
sponding RMSSE values are given in Table 10 (Appendix C). Monthly datasets were excluded due to their insuﬃcient
sample sizes to ﬁt the parameters of a NN. The standard deviation is shown in paranthesis. The model deﬁnition
shows the number of input lags and the number of layers, including their hidden sizes. e.g. ’4x16 NN’ signiﬁes that
the model was set to use 4 hidden layers of dimension 16. Note: For all models with a NN, the learning rate was
manually set, but not tuned, to a guesstimate of 0.001.
rate was manually set to a value of 0.001 for all model variations and datasets. This value is
a guesstimate which was not tuned. It was set to avoid potential sources of variation from the
learning rate range test. All other model parameters which were set for any of the models are
explicitly displayed in Table 5: number of forecast steps, number of lags, number of hidden layers,
and and hidden layer dimensions.
4.2.2. Results Based on Data Subject Type
In general, models with more lags tend to perform better. This is particularly evident in the case
of datasets of energy subject type, where the 120-lags NeuralProphet model reduces the forecast
error by 96% on 1-step ahead and 66% on 60-step ahead horizons. Further, we observe that the
model conﬁgured with a 4 layer, 16 dimension NN is outperformed by linear models on 1 step and
partially on 3 step ahead forecasts, while performing best on 15 step and 60 step forecasts for all
data types.
Overall, conﬁguring any amount of lags signiﬁcantly improves the forecasting performance
of NeuralProphet for any forecast horizon.
Except for 60-step horizons on datasets of human
32

human
economic
environment
energy
∞-step MASE
Prophet
1.11 (±0.26)
4.25 (±3.09)
9.27 (±3.89)
11.13 (±1.22)
NeuralProphet
1.13 (±0.21)
2.39 (±0.85)
9.48 (±3.75)
11.13 (±1.40)
1-step MASE
NeuralProphet (1 lags)
0.76 (±0.08)
1.01 (±0.21)
0.81 (±0.10)
0.83 (±0.05)
NeuralProphet (3 lags)
0.75 (±0.08)
0.98 (±0.14)
0.76 (±0.09)
0.63 (±0.05)
NeuralProphet (12 lags)
0.66 (±0.07)
0.97 (±0.17)
0.75 (±0.08)
0.60 (±0.04)
NeuralProphet (30 lags)
0.62 (±0.07)
0.82 (±0.18)
0.74 (±0.10)
0.48 (±0.03)
NeuralProphet (120 lags)⋆
0.67 (±0.15)
0.75 (±0.12)
0.75 (±0.11)
0.46 (±0.04)
NeuralProphet (120 lags, 4x16 NN)⋆
0.75 (±0.13)
0.77 (±0.06)
0.83 (±0.16)
0.48 (±0.07)
3-step MASE
NeuralProphet (12 lags)
0.77 (±0.09)
1.81 (±0.60)
1.25 (±0.17)
1.10 (±0.09)
NeuralProphet (30 lags)
0.73 (±0.09)
1.08 (±0.20)
1.21 (±0.18)
0.88 (±0.06)
NeuralProphet (120 lags)⋆
0.73 (±0.11)
0.98 (±0.13)
1.19 (±0.18)
0.79 (±0.06)
NeuralProphet (120 lags, 4x16 NN)⋆
0.87 (±0.13)
0.83 (±0.09)
1.27 (±0.25)
0.76 (±0.08)
15-step MASE
NeuralProphet (30 lags)
0.89 (±0.16)
1.58 (±0.33)
2.52 (±0.48)
2.17 (±0.18)
NeuralProphet (120 lags)⋆
0.93 (±0.21)
1.30 (±0.30)
2.45 (±0.46)
2.04 (±0.18)
NeuralProphet (120 lags, 4x16 NN)⋆
0.88 (±0.16)
1.18 (±0.10)
2.40 (±0.57)
1.85 (±0.19)
60-step MASE
NeuralProphet (120 lags)⋆
1.38 (±0.38)
1.70 (±0.43)
5.14 (±1.50)
3.78 (±0.43)
NeuralProphet (120 lags, 4x16 NN)⋆
1.05 (±0.34)
1.52 (±0.12)
3.75 (±0.81)
3.34 (±0.28)
Table 6: Forecast error (MASE) for each dataset subject type for diﬀerent forecast horizons. The standard deviation
is shown in paranthesis. Note⋆: Models with 120 lags were not evaluated over monthly datasets due to their short
length.
33

subject type, where only a NN-based model outperforms Prophet and default NeuralProphet.
Measurements of human activity tend to have strong seasonality and have a quickly fading impact
of local context. When forecasting into the distant future, statistical patterns tend to dominate
human activity data.
4.3. Task-dependent Strength of Prophet and NeuralProphet
In Table 7 we present our subjective advice on which model to use depending on the task at
hand. These are qualitative suggestions based on the empirical evidence we found when evaluating
the two models against each other.
Task
Prophet
NeuralProphet
Small dataset (T ≪100)

Medium to large dataset (T ≫100)

Long range forecast (h ≫100)


Short to medium range forecast (1 ≤h ≪1000)

Speciﬁc forecast horizon (e.g. h = 24)

Auto-correlation (dependence on previous observations)

Lagged regressors (dependence on observed covariates)

Non-linear dynamics

Global modelling of panel dataset

Frequent retraining on small datasets with

constrained computational resources
Fast prediction (computational inference time)

Table 7: Comparison of Prophet’s and NeuralProphet’s task-dependent strength.  marks which of the two models
we suggest to use for the given task.
5. Conclusion
NeuralProphet is the ﬁrst hybrid forecasting framework that meets the industry standards
for explainability and simplicity-of-use set by Facebook Prophet. As NeuralProphet is based on
PyTorch and trained with standard deep learning methods, any algorithm trainable by mini-batch
Stochastic Gradient Descent can be included as a module, which makes it easy for developers to
extend the framework with new features, and to adopt new research.
34

NeuralProphet can model local context with auto-regression and covariate regression, making
it a suitable tool for time-series where the near-term future depends on the current state of the sys-
tem. This is evidenced by our empirical results where NeuralProphet with auto-regression enabled
signiﬁcantly outperforms Prophet on all forecasting horizons. Both models perform comparably
when conﬁgured identically. On short to medium-term forecast horizons, most conﬁgurations of
NeuralProphet reduce the forecast error by 50% to 90%. Fusing the model with a Neural Network
further improves forecast accuracy for medium to long range horizons.
In our ablation studies, we observe that NeuralProphet’s performance is not sensitive to the
speciﬁc number of lags or the Neural Network conﬁguration.
NeuralProphet is accessible to forecasting beginners thanks to solid default and automatic
hyperparameters, while advanced users can incorporate domain knowledge with optional model
customization. Upgrading Prophet with NeuralProphet empowers forecasting practitioners with
an explainable model in a convenient and scalable framework, covering a wide range of forecasting
applications.
Acknowledgements
We are thankful for the invaluable advice by our academic and industrial advisers at Meta, Face-
book AI, Netﬂix, Uber, Monash and Skoltech; in particular, Italo Lima, Caner Komurlu, Alessandro
Panella, Evgeny Burnaev, Sean Taylor, and Benjamin Letham. We gratefully thank Total Energies
for their continued support; in particular, Lluvia Ochoa, Gonzague Henri, and Michel Lutz. We
would also like to acknowledge the hard work of our lovely open-source contributors; in particular
Mateus De Castro Ribeiro, Riley De Haan, and Bernhard Hausleitner.
The work presented herein was funded in part by a research agreement between Stanford Uni-
versity and Total Energies. The views and opinions of authors expressed herein do not necessarily
state or reﬂect those of the funding source.
References
Carpenter, B., Gelman, A., Hoﬀman, M., Lee, D., Goodrich, B., Betancourt, M., Brubaker, M., Guo, J., Li, P., &
Riddell, A. (2017). Stan : A probabilistic programming language. Journal of Statistical Software, 76. doi:10.
18637/jss.v076.i01.
Crone, S. F. (2008). Nn5 competition. URL: neural-forecasting-competition.com/NN5.
35

DOE, N. (2015).
Sanfrancisco hourly solar ground illumination data in w per m2 from national solar radiation
database. URL: nsrdb.nrel.gov/about/tmy.
Dua, D., & Graﬀ, C. (2017). Uci machine learning repository. URL: archive.ics.uci.edu/ml.
ERCOT (2021). Ercot control area hourly load from january 2004 to september 2021, hourly load data archives.
URL: ercot.com/gridinfo/load/load_hist.
Girshick, R. (2015). Fast r-cnn. arXiv:1504.08083.
Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P. (2021).
Monash time series
forecasting archive. arXiv:2105.06643.
Harvey, A. C., & Shephard, N. (1993). Structural time series models. In Handbook of Statistics (pp. 261–302).
Amsterdam: North Holland volume Vol. 11:Econometrics. ((edited by g.s. maddala, c.r. rao and h.d. vinod) ed.).
Hyndman, R. (2018). A brief history of time series forecasting competitions. URL: robjhyndman.com/hyndsight/
forecasting-competitions.
Hyndman, R., & Athanasopoulos, G. (2014). Forecasting: principles and practice. OTexts. URL: books.google.
com/books?id=gDuRBAAAQBAJ.
Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal
of Forecasting, 22, 679–688. URL: https://www.sciencedirect.com/science/article/pii/S0169207006000239.
doi:https://doi.org/10.1016/j.ijforecast.2006.03.001.
Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. arXiv:1711.05101.
Makridakis, S., & Hibon, M. (2000). The m3-competition: results, conclusions and implications. Int. J. Forecast.,
16, 451–476.
Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and machine learning forecasting methods:
Concerns and ways forward. PLOS ONE, 13, e0194889. doi:10.1371/journal.pone.0194889.
Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer Science and Business Media.
OpenEI (2015). Sanfrancisco hospital hourly power consumption data of facility in kw. URL: data.openei.org/
submissions/153.
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga,
L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang,
L., Bai, J., & Chintala, S. (2019). Pytorch: An imperative style, high-performance deep learning library. In
H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alche Buc, E. Fox, & R. Garnett (Eds.), Advances in Neural
Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc.
RTE (2018). Record of hourly power consumption from french utility rte from january 2017 to december 2018. URL:
services-rte.com/en/download-data-published-by-rte.
Smith, L. N. (2017). Cyclical learning rates for training neural networks. arXiv:1506.01186.
Smith, L. N., & Topin, N. (2018). Super-convergence: Very fast training of neural networks using large learning
rates. arXiv:1708.07120.
Svetunkov, I. (2021). Forecasting and analytics with adam. OpenForecast. URL: https://openforecast.org/adam/.
Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. International Journal
of Forecasting, 16, 437–450. URL: https://www.sciencedirect.com/science/article/pii/S0169207000000650.
36

doi:https://doi.org/10.1016/S0169-2070(00)00065-0. The M3- Competition.
Taylor, S. J., & Letham, B. (2017). Forecasting at scale. PeerJ, . URL: doi.org/10.7287/peerj.preprints.3190v2.
doi:10.7287/peerj.preprints.3190v2.
Triebe, O., Laptev, N., & Rajagopal, R. (2019). Ar-net: A simple auto-regressive neural network for time-series.
arXiv:1911.12436.
Uber (2018). Omphalos, uber’s parallel and language-extensible time series backtesting tool. URL: eng.uber.com/
omphalos/.
Uber (2020). Building a backtesting service to measure model performance at uber-scale. URL: eng.uber.com/
backtesting-at-scale/.
37

6. Appendices
6.1. Appendix A: Details on datasets
Name
Type
Subtype
Freq
T
log10(T)
air passengers
human
tourism
M
144
2.2
retail sales
economic
sales
M
293
2.5
peyton manning
human
clicks
D
2905
3.5
birth
human
population
D
7305
3.9
load hospital
energy
load
H
8760
3.9
solar sf
environment
sun
H
8,760
3.9
load rte
energy
load
H
17518
4.2
load victoria
energy
load
30min
17520
4.2
yosemite temps
environment
weather
5min
18721
4.3
river
environment
water
D
23741
4.4
price ercot
economic
energy price
H
25537
4.4
air beijing
environment
air quality
H
43800
4.6
sunspot
environment
sun
D
73924
4.9
load ercot
energy
load
H
154872
5.2
power wind
energy
wind
1min
493144
5.7
power solar
energy
solar
1min
493149
5.7
Table 8: Summary of dataset characteristics. T refers to the dataset length.
38

6.2. Appendix B: Decomposition RMSE per component per experiment
(a) Experiments without lagged components.
(b) Experiments including lagged components.⋆
Figure 7: Here we display the decomposition performance for each experiment and component. Metric is given as
RMSE of predicted forecast component to underlying true generated component value. The plotted bar displays
mean value and the line displays standard deviation over 5 runs. ⋆Note: Prophet does not support Auto-Regression
and Lagged Regressor components. Prophet’s approximation of said components is implicitly assumed to be zero.
Thus, the shown error for Prophet is equivalent to the component’s standard deviation.
39

6.3. Appendix C: RMSSE per forecast horizon
RMSSE for diﬀerent forecast horizons
Model
1 step
3 steps
15 steps
60 steps
Prophet⋆1
4.37 (±0.99)
NeuralProphet⋆1
4.41 (±0.97)
NeuralProphet (1 lags)
0.71 (±0.09)
N/A
N/A
N/A
NeuralProphet (3 lags)
0.61 (±0.08)
N/A
N/A
N/A
NeuralProphet (12 lags)
0.59 (±0.08)
0.92 (±0.17)
N/A
N/A
NeuralProphet (30 lags)
0.57 (±0.11)
0.82 (±0.16)
1.51 (±0.21)
N/A
NeuralProphet (120 lags)⋆2
0.51 (±0.08)
0.76 (±0.11)
1.45 (±0.22)
2.46 (±0.49)
Table 9: Forecast error (RMSSE) for each forecast horizon across all datasets. The standard deviation is shown in
paranthesis. Note⋆1: The models without lags can produce forecasts for an arbitrary number of forecasts. Note⋆2:
The model with 120 lags was not evaluated over monthly datasets due to their short length.
RMSSE for diﬀerent forecast horizons
Model
1 step
3 steps
15 steps
60 steps
(30 lags, 1x32 NN)
0.51 (±0.07)
0.76 (±0.11)
1.42 (±0.20)
N/A
(30 lags, 2x24 NN)
0.50 (±0.07)
0.75 (±0.11)
1.43 (±0.22)
N/A
(30 lags, 4x16 NN)
0.51 (±0.08)
0.76 (±0.11)
1.44 (±0.21)
N/A
(120 lags, 1x32 NN)
0.52 (±0.09)
0.75 (±0.11)
1.36 (±0.20)
2.07 (±0.35)
(120 lags, 2x24 NN)
0.55 (±0.11)
0.74 (±0.11)
1.37 (±0.19)
2.09 (±0.39)
(120 lags, 4x16 NN)
0.55 (±0.10)
0.78 (±0.12)
1.41 (±0.25)
2.09 (±0.35)
Table 10: NeuralProphet (with NN) forecast error (RMSSE) for each forecast horizon across all datasets, except
monthly, due to their insuﬃcient length to ﬁt a NN. The standard deviation is shown in paranthesis. The model
deﬁnition shows the number of input lags and the number of layers an their hidden sizes. e.g. 4x16 signiﬁes 4 hidden
layers of dimension 16.
40
