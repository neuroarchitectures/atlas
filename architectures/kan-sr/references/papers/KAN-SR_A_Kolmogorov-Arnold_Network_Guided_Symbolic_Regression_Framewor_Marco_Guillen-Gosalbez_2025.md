# KAN-SR A Kolmogorov-Arnold Network Guided Symbolic Regression Framewor Marco Guillen-Gosalbez 2025

> Source: `KAN-SR_A_Kolmogorov-Arnold_Network_Guided_Symbolic_Regression_Framewor_Marco_Guillen-Gosalbez_2025.pdf`

---

                                                  KAN-SR: A KOLMOGOROV-A RNOLD N ETWORK G UIDED
                                                        S YMBOLIC R EGRESSION F RAMEWORK

                                                                                                 A P REPRINT


                                                                 Marco A. Bühler                             Gonzalo Guillén-Gosálbez∗
                                                             ETH Zürich, Switzerland                         ETH Zürich, Switzerland
                                                          marco.buehler@chem.ethz.ch                gonzalo.guillen.gosalbez@chem.ethz.ch




arXiv:2509.10089v1 [cs.LG] 12 Sep 2025
                                                                                                A BSTRACT
                                                     We introduce a novel symbolic regression framework, namely KAN-SR, built on Kolmogorov Arnold
                                                     Networks (KANs) which follows a divide-and-conquer approach. Symbolic regression searches for
                                                     mathematical equations that best fit a given dataset and is commonly solved with genetic programming
                                                     approaches. We show that by using deep learning techniques, more specific KANs, and combining
                                                     them with simplification strategies such as translational symmetries and separabilities, we are able to
                                                     recover ground-truth equations of the Feynman Symbolic Regression for Scientific Discovery (SRSD)
                                                     dataset. Additionally, we show that by combining the proposed framework with neural controlled
                                                     differential equations, we are able to model the dynamics of an in-silico bioprocess system precisely,
                                                     opening the door for the dynamic modeling of other engineering systems.

                                         Keywords: Symbolic Regression · Kolmogorov-Arnold Networks · Feynman-SRSD Dataset

                                         1       Introduction
                                         Symbolic regression (SR) aims to recover closed-form mathematical expressions that accurately model a given dataset.
                                         Unlike traditional regression approaches, which assume a fixed model structure, SR jointly infers both the functional
                                         form (equation) and its parameters (constants) from the data, producing models that are not only predictive, but also (to
                                         some extent) interpretable. This flexibility makes SR especially appealing in scientific domains, where understanding
                                         the underlying mechanisms can be more important than raw predictive performance [1, 2].
                                         Because of this, SR emphasizes simplicity and interpretability alongside predictive performance. The goal is to find
                                         concise mathematical relationships, ideally those that offer insight into the system being modeled [3]. This contrasts
                                         with black-box machine learning models, which may perform well but often lack transparency and might extrapolate
                                         poorly.
                                         Classical SR methods, such as genetic programming (GP) [4, 5, 6, 7], explore the space of symbolic expressions
                                         using evolutionary algorithms applied to expression trees [8]. Although flexible, GP-based methods tend to be sample-
                                         inefficient, computationally expensive, and often face difficulties with scalability. More recent approaches have tried
                                         to overcome these limitations using pre-trained language models [9, 10, 11], reinforcement-learning [12], hybrid
                                         frameworks that integrate deep learning with GP-style symbolic search [13, 14] and physics-inspired heuristics [15].
                                         Despite this progress, challenges remain. Many neural-symbolic systems still rely on discrete symbolic representations
                                         that are difficult to train end-to-end, limiting their ability to learn efficiently or produce more interpretable results.
                                         Furthermore, there is no universally adopted benchmark, which makes comparing SR methods inconsistent. A step
                                         toward covering this gap was the Symbolic Regression for Scientific Discovery (SRSD) benchmark by Matsubara et
                                         al. [16], which introduced 240 datasets inspired by the Feynman lectures, with sampling which corresponds to the real
                                         measured phenomena and the inclusion of irrelevant variables to test model robustness.
                                         In engineering sciences such as chemical engineering and bioengineering, the insights created by the SR algorithm
                                         can lead to faster and more robust process development or a deeper understanding of phenomena. Applications range
                                             ∗
                                                 Corresponding Author
from finding kinetic models to reactions [2, 17, 18, 19], designing new catalysts [20] and describing different transport
phenomena [21]. Additionally, dynamic systems are at the core of many of these applications, and identifying these
governing equations in a dynamic environment is an open problem. An algorithm that is tailored to identify differential
equations is SINDy [1] which leverages sparse regression to find the best fitting combination of hand-made equation
blocks.
In this paper, we introduce KAN-SR, a symbolic regression framework based on Kolmogorov-Arnold Networks
(KANs) [22, 23]. KANs are a recent neural network architecture inspired by the Kolmogorov-Arnold representation
theorem. Instead of traditional neurons with fixed activation functions and linear weights, KANs use learnable univariate
functions, allowing more expressive and often more interpretable function approximations.
To adapt KANs for symbolic regression, we develop a hybrid extraction pipeline that combines differentiable KAN train-
ing with symbolic simplification strategies inspired by AI Feynman [15]. By applying these symbolic simplifications,
e.g. symmetries and separabilities, the initial complex problem can be decomposed into simpler subproblems, which
can often be directly fitted by a single-layer KAN. After fitting a KAN to the data, we decompose the network into
modular algebraic components which are then matched to a symbolic library via non-linear least squares. This enables
the extraction of compact, interpretable expressions without relying on brute-force search or discrete enumeration.
We evaluate KAN-SR on the SRSD dataset, comparing it against state-of-the-art symbolic regression models evaluated
in Matsubara et al. [16]. Our results show that KAN-SR achieves competitive or superior performance in symbolic
recovery, particularly in scenarios involving compositional structure, extrapolation, and noisy inputs with irrelevant
variables.
Furthermore, we extend KAN-SR to an in-silico dynamic biological system, demonstrating its application to a bioprocess
modeling case study, where it recovers interpretable kinetic rate equations [2]. Overall, our work opens up new avenues
for the application of KANs to SR problems in a range of engineering problems, including the modeling of dyamic
systems.

Contributions
       • We propose a symbolic regression framework based on Kolmogorov-Arnold Networks and symbolic modeling,
         unifying compositional deep learning and symbolic extraction.
       • We introduce a symbolic extraction and simplification pipeline inspired by AI Feynman [15], using a compre-
         hensive and extendable univariate function library in combination with KANs.
       • We evaluate our approach on the SRSD benchmark [16], showing improvements in symbolic recovery.
       • We demonstrate the extension of our method to dynamic systems through a bioprocess modeling case study,
         leveraging neural-controlled differential equations.


2     Methods

SR searches for an equation y(x) = Θ(x, θ) in the space of expressions Θ and constants θ, to find the best fit
combination that minimizes an objective function, which often is a combination of the fitting error and a complexity
measure of the equation, such as the Bayesian Information Criterion (BIC). SR assumes that there is an analytical
ground truth of the form y(x) = Θ∗ (x, θ∗ ) + ϵ that created the training data x ∈ Rd with the target y ∈ R in the
presence of Gaussian noise ϵ.

Implementation details: KAN-SR was implemented in Python using Jax [24] as the base for all calculations,
Equinox [25] for the KANs, Optimistix [26] for the non-linear least squares regression and optimizers, Optax [27] for
first-order optimizers and Diffrax [28] for the neural differential equations [29].

2.1   Workflow

The proposed symbolic regression framework is highly flexible and involves several hyperparameters and algorithmic
components. This flexible design is motivated by the fact that there is no universal KAN architecture that can
accommodate all target functions, even when strong regularization is applied. Consequently, the workflow consists of
a sequence of optional and configurable steps, some of which can be enabled or skipped, depending on the problem
characteristics and performance criteria.
The general workflow is as follows, see Figures 1 for a more detailed illustration:


                                                           2
         1. Preprocessing: Normalize or scale the input variables to improve the numerical stability and convergence.
         2. Brute-force search: Optionally perform an exhaustive search over simple expressions.
         3. Single-layer KANs with a single unit: Attempt a symbolic approximation using minimal architectures with a
            summation or multiplication unit.
         4. Single-layer KANs with multiple units: Expand the search space by allowing multiple units within a
            single-layer architecture.
         5. Simplification and subproblem decomposition: Optionally identify structural simplifications or separable
            components in the learned expression, and recursively restart the procedure on the subproblems.
         6. Output transformation: Optionally apply transformations to the output (e.g., logarithmic, exponential) to
            simplify the functional form and rerun the symbolic search on the transformed target.
         7. Deep KAN fitting: As a final step, if chosen, employ deeper KAN architectures to capture more complex or
            hierarchical functional relationships if simpler models fail, see Figure 2 for an illustration.

                                         Input Data               Optional Components

                                       Preprocessing

                           Brute-force Symbolic Matching


                                                            Yes
                                     error ≤ threshold             Return Symbolic Model


                                          No                                                                           Start Deep KAN Loop
                                     Single-Unit KAN
                                                                                                                              Set d = 1
                                    Symbolic Extraction
                                                                                                                                             Yes
                                                                                                                               d > dmax              Return Best So Far
                                                            Yes
                                     error ≤ threshold             Return Symbolic Model
                                                                                                                               No
                                                                                                                             Set w = wmin
                                          No
                                      Multi-Unit KAN
                                                                                                                       Fit KAN with (d, w)
                                    Symbolic Extraction

                                                                                                                       Symbolic Extraction
                                                            Yes
                                     error ≤ threshold             Return Symbolic Model
                                                                                                                                                   Yes
                                                                                                                         error ≤ threshold               Return Symbolic Model
                                          No
                               Simplification / Transformation                                                                 No
                                                                                                                       Yes
                                                                                                         Increment w          w < wmax
                         Yes                                 No
      Simplified Input              Simplification Found           Deep KAN Loop, Figure 2                                          No
                                                                                                                             Increment d

Figure 1: Main symbolic regression workflow. Optional                                            Figure 2: Nested deep KAN fitting loop. For each depth d,
modules include brute-force matching, simplification, and                                        the system explores increasing widths w up to a maximum.
deep KAN fitting. The pipeline returns early if a symbolic                                       If no symbolic match is found, the depth is incremented
match is found, otherwise, it proceeds through increas-                                          until the limit is reached.
ingly expressive modeling stages. If the threshold is not
reached after a full completion of the algorithm, the best
equation found during the search is returned.


2.2      Brute-Force Symbolic Matching

To rapidly identify simple and low-complexity symbolic relationships, we implement a brute-force symbolic regression
module as a first step in the workflow. This stage exhaustively fits a set of simple multivariate multiplicative expressions
to the data.
The expression library includes forms such as:
                                                                                                        θ 1 x0
                                                             f (x) = θ1 x0 x1 + θ2 ,         f (x) =            + θ3 ,
                                                                                                       x1 + θ 2

                                                                                             3
as well as higher-order expressions such as:
                                                                                θ1
                               f (x) = θ1 x0 x1 x2 x3 + θ2 ,   f (x) =                     + θ3 .
                                                                         x0 x1 x2 x3 + θ 2

Each individual expression is defined as a parameterized function f (x; θ) and is fitted using a BFGS (Broyden–Fletcher–
Goldfarb–Shanno) algorithm. If the resulting error of the best fitting equation is below a threshold, the symbolic form is
accepted without further modeling.
This brute-force module is especially effective when the target function has a simple algebraic form and allows for a
much faster symbolic identification before applying more expensive models.

2.3   Kolmogorov–Arnold Networks

In the original KAN formulation [22, 23], univariate activation functions are represented using B-splines. Although
expressive, this implementation has practical limitations: the grid of spline knots can be manually extended during
training allowing for better accuracy but this stops training and an additional fit has to be performed. Additionally the
evaluation of the splines is slow and can thus become a computational bottleneck. To address these issues, we adopt the
improved Fast-KAN approach [30], which replaces B-splines with radial basis functions for greater efficiency.
In our work, the learnable univariate activation is implemented using a Reflectional Switch Activation Function
(RSWAF) [31], defined as:
                                                                     
                                                            2   x − ci
                                               wi si − tanh               ,
                                                                  hi
where:
         • wi is a learnable weight that modulates the contribution of the i-th basis function,
         • si is the reference activation level,
         • ci denotes the center of the activation,
         • hi is a scale parameter that controls the sharpness of the activation around ci .
This formulation yields smooth, localized basis functions centered on ci , and defined in the complete domain. The
compositional flexibility of these units allows the network to approximate a wide range of non-linear univariate
functions.
The final activation function is then of the form:
                                                X                 
                                                                     x − ci
                                                                            
                                                                 2
                                        ϕ(x) =      wi si − tanh                                                         (1)
                                                  i
                                                                       hi
This reformulation from B-Splines to radial basis functions allows for faster training, decreasing the time needed for
the forward and backward pass. Empirically, we find that most target functions can be approximated with just five
such basis functions per univariate activation, resulting in 20 learnable parameters per activation, the only limitation
being high-frequency periodic signals. By adapting these weights via gradient descent, the KAN can learn a variety of
univariate functions (see Figure 3 for an example).
However, a limitation of the original KAN is its inherent inability to directly represent multiplicative interactions of the
form
                                                f (x1 , x2 ) = g(x1 ) · h(x2 )
using a single-layer architecture. Such factorizable functions require multiple layers under a purely additive composition
due to the lack of interaction terms.
To overcome this, we incorporate ideas from the multiplicative KAN extension [23] and explicitly allow additive and
multiplicative combinations of learnable activations. Specifically, instead of computing only additive outputs, we also
construct terms of the form:
                                                                    n
                                                                    Y
                                             f (x1 , . . . , xn ) =   ϕi (xi ),
                                                                   i=1
where each ϕi (xi ) is the learnable univariate function parameterized as above. This allows the model to directly
represent separable multiplicative functions common in scientific domains.


                                                               4
(a) Random initialization of five basis func-       (b) Basis functions after training.          (c) Element-wise sum over all five basis
tions with a uniform grid.                                                                       functions of (b); Equation 1.
Figure 3: Example of the training of a single univariate activation function, where the grid is directly adapted during
training, to best fit the target exponential function. The activation function is only evaluated over the input range of the
variable to extract the best fitting equation.


Our final model architecture of a single-layer KAN consists of a sum over K such sub-networks (or units), where each
unit computes a composition over the full input space via either summation or multiplication:

                                                          K
                                                          X
                                 f (x1 , . . . , xn ) =         Ck (ϕk1 (x1 ), ϕk2 (x2 ), . . . , ϕkn (xn )) ,
                                                          k=1
                                                         Pn
                                                               z , if unit k is additive,
                                  Ck (z1 , . . . , zn ) = Qni=1 i
                                                           i=1 i ,
                                                               z   if unit k is multiplicative.

This hybrid composition framework allows the network to represent a rich class of functions by flexibly combining both
additive and multiplicative structures over the full input domain.
To apply the method to deeper or multi-output KANs, we extend the above equations to an additional dimension j that
represents either the number of input features of the next layer or multiple output features.
                                                         Kj
                                                         X
                               fj (x1 , . . . , xn ) =         Ck (ϕjk1 (x1 ), ϕjk2 (x2 ), . . . , ϕjkn (xn )) ,
                                                         k=1


To guide learning, a base linear layer is included, as in the original implementation [22] with a GELU [32] activation
function. This basis function is similar to residual connections, such that the final output is the sum of the basis function
and the learnable activation functions.

Regularization
Our final KAN architecture consists of multiple subunits, each operating on the full input space. Due to this, the model
is often overspecified, whereas most symbolic expressions, particularly in scientific domains, often only depend on a
small set of operators, each with a subset of input variables. To promote sparsity and thus interpretability, we introduce
a regularization scheme designed to penalize unnecessary input usage and enforce compact functional representations,
similar to the one proposed in the original implementation [22], as described below.
For each subunit k ∈ K: We construct a matrix A ∈ Rm×n , where Aji represents the mean absolute activation of
ϕji (xi ) for input xi i ∈ n and the output dimension j ∈ m. That is,
                                                                Aji = |ϕji (xi )|
We interpret A as capturing the degree to which each input dimension contributes to the output between samples.
By summing over the matrix we get the first penalty (magnitude):
                                                                     m X
                                                                     X n
                                                   Lmagnitude =                 ∥ϕji (xi )∥1
                                                                      j=1 i=1


                                                                         5
To encourage sparsity on the whole structure, we again take matrix Aij and use it to compute the row-wise and
column-wise entropies (Hrow , Hcol ), which are a measure of the diffuseness of activations across inputs and samples,
respectively. Let:

                                                      Aji                             Aji
                                       pcol
                                        ji = Pm                 ,       prow
                                                                         ji = Pn
                                                    j=1 Aji + ϵ                     i=1 Aji + ϵ

Then the entropy loss is defined as follows:
                                                Lentropy = Hcol (A) + Hrow (A),
                                                               n    m
                                                           1 X X col
                                          Hcol (A) = −              p log(pcol
                                                                           ji + ϵ)
                                                           n i=1 j=1 ji
                                                              m     n
                                                          1 X X row
                                        Hrow (A) = −               p log(prow
                                                                          ji + ϵ),
                                                          m j=1 i=1 ji

These entropy terms encourage each subunit to focus on a small, confident set of inputs, reducing ambiguous or diffuse
activation patterns. This is important to align the learned representations with the sparse structure of symbolic equations.
Finally, we include, when applicable, the ℓ1 penalty on the weights (Wbase ) of the base linear layer. The total
regularization objective is the following, with subunits k ∈ K:

                                 K
                                 X              K
                                                X
                        Lreg =       Lreg,k =        λ0 (λ1 Lmagnitude,k + λ2 Lentropy,k + λ3 ∥Wbase,k ∥1 ),
                                 k              k
where λi are hyperparameters.

2.4   Symbolic Simplification via Separability and Symmetry Detection

Before training single-layer KANs, we apply the recursive decomposition strategies proposed by Udrescu et al. [15].
For this, a lightweight interpolating network is used to analyze the target function and detect structural properties, such
as variable separability and symmetry, that can reduce the complexity of the problem. All the methods in this section
come from Udrescu et al. [15] and only a short summary of the methods used is given here.

Separability Detection. We test whether the target function is approximately additively or multiplicatively separable
across disjoint subsets of input variables. For each candidate split (S, S̄), we define:

       • fS (x): with S̄ fixed to its mean,
       • fS̄ (x): with S fixed to its mean,
       • f0 (x): with both subsets fixed to their mean.

We then compute normalized approximation errors:
                                                                                                                    
                         1                                                           1            fS (x) · fS̄ (x)
               ϵadd =        ∥y − (fS (x) + fS̄ (x) − f0 (x))∥1 ,         ϵmult =        y−                                  ,
                        ϵval                                                        ϵval              f0 (x)             1

where ϵval is the validation error of the interpolating network. If either error falls below a fixed threshold (ϵ < 10), the
function is considered separable. The best performing subset is used further if more than one subset was found during
the search. Then two new data sets are created that only depend on the individual subsets found, using the trained
interpolating network to create the new targets y’ and y”. This allows KANs to model lower-dimensional subproblems.

Symmetry Detection. We also test for input symmetries by applying structured transformations to input pairs (xi , xj )
and measuring the invariance of the output. The following transformations are considered, with a a constant:

       • Negative translation: f (xi + a, xj + a) ≈ f (xi − xj )
       • Positive translation: f (xi + a, xj − a) ≈ f (xi + xj )
       • Multiplicative: f (axi , xj /a) ≈ f (xi · xj )


                                                                    6
       • Divisive: f (axi , axj ) ≈ f (xi /xj )

Each transformation is applied and the transformed output is compared with the original (f (xi , xj ) ) normalized by the
validation error. If the relative error falls below a threshold, the symmetry is deemed valid.
When multiple symmetries or separabilities are detected, we apply the one with the lowest relative error. Symmetries are
evaluated first because they reduce dimensionality without relying on model-generated targets. Separabilities depend on
the network output and may introduce an additional approximation error.
These symbolic simplification steps reduce complexity and dimensionality, allowing KANs to focus on simpler
subproblems, ultimately improving interpretability and performance on scientific regression tasks.

2.5   Symbolic Expression Extraction

To convert trained KANs into closed-form symbolic models, we implement a post-hoc symbolic extraction pipeline that
approximates each univariate activation with interpretable expressions from a predefined, extendable function library.
Each univariate activation ϕki (xi ) within the trained KAN is evaluated over its effective input range, and a set of
candidate symbolic functions are fitted minimizing the mean squared error (MSE). The candidate functions are drawn
from a predefined library comprising polynomials, trigonometric forms, exponential and logarithmic functions, and
domain-specific functions such as Michaelis-Menten kinetics (often used to model biological systems).
The best-fitting function is selected according to one of two criteria:

       • Best fit: The candidate with the lowest MSE.
       • Score-based: A simple trade-off metric between MSE and expression complexity, penalizing longer and more
         complex expressions. The complexity is penalized according to the following formula: With Θ being the
         complexity manually chosen of the fitted function.

                                                              Score = MSE · 5Θ
         The parameters Θ for each equation are chosen arbitrarily and tuned during experimentation, with lower
         complexity equations (such as linear or constants) having the lowest values and complex equations (such as
         x · sin x) higher values. This scoring approach offers a simple, yet effective, trade-off between accuracy and
         interpretability. While it can be easily extended to incorporate more complex model selection criteria, such as
         the Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC), which balance likelihood
         with model complexity, we find that this lightweight heuristic performs well enough in practice. Moreover,
         multi-objective optimization could also be used to handle the trade-off between complexity and accuracy, yet
         this would also lead to more complex formulations.

Once all symbolic approximations are determined, the network structure is symbolically propagated layer by layer. In
the case of additive or multiplicative compositions (as specified by each sub-unit), symbolic expressions are either
summed or multiplied accordingly. Intermediate expressions are simplified using SymPy [33] and constants are
re-optimized using a BFGS algorithm to minimize final output error.
This procedure results in a fully symbolic approximation of the trained KAN model:
                                                            K
                                                            X                                     
                                   f (x1 , . . . , xn ) ≈         Ck ϕ̂k1 (x1 ), . . . , ϕ̂kn (xn ) ,
                                                            k=1

where ϕ̂ki are the symbolic approximations extracted for each activation and Ck ∈ {sum, product} denotes the unit-level
composition type.

2.6   Output Transformation for Simplification

If after simplifying and fitting various single-layer KANs, a good fitting function has not been found, we start by
transforming the target variable y. Target functions may include non-linear or nested structures that are difficult to
simplify directly or require deeper KAN architectures. To address this, we apply transformations to the output space to
make the function simpler for symbolic approximation.
Let f (x) denote the original target function. We apply a transformation T , such that:
                                                                  T (f (x))


                                                                     7
has a simplified structure.
For example, if f (x1 , x2 ) = exp(sin(x1 ) + sin(x2 )), then applying the logarithm yields ln(f (x1 , x2 )) = sin(x1 ) +
sin(x2 ), which is significantly simpler. After identifying a symbolic approximation fˆ of the transformed function, we
recover the original via the inverse transformation: f (x1 , x2 ) = exp(fˆ(x1 , x2 )).

2.7   Neural Controlled Differential Equations for Noise-Robust Derivative Estimation

Most SR algorithms are built and applied to static data. However, many engineering systems are inherently dynamic,
making the direct application of these standard SR algorithms to them challenging. SINDy [1] and ARGOS [34] are a
purpose-built SR algorithm to discover the equations governing differential equations, but often require hand-made
equation building blocks and sparse regression to guide the search. A different approach is ODEFormer [35] which is a
transformer-based architecture which directly predicts the governing equations from data profiles. Here, we discuss the
use of our algorithm to model dynamic data, with emphasis on kinetic modeling of a bioprocess. In the modeling of
dynamic systems, accurate derivative estimation is a crucial prerequisite to rediscover the underlying equations that
govern them, and classical numerical differentiation methods (e.g., finite differences) are highly sensitive to noise,
often producing unreliable gradients when applied to real-world, noisy measurements. To address this limitation, we
use Neural Controlled Differential Equations (Neural CDEs) [36, 28] as a data-driven framework to extract smooth
derivative estimates from noisy time series that will be used as input to KAN-SR.
Neural CDEs generalize recurrent neural networks (RNNs) by modeling the evolution of a hidden state in continuous
time, where the input is treated as a control signal driving a differential equation. This idea extends the modeling
paradigm introduced in Neural ODEs [29], whilst offering a more flexible framework for handling continuously arriving
and irregularly sampled time series data.
A neural CDE models the evolution of a latent hidden state Z(t) ∈ Rd , driven by a known continuous control path
U (t) ∈ Rm . The control path U (t) is assumed to be a smooth function of time, contrary to our observed state variables
X(t) ∈ Rn , which can be noisy or discretely observed. It is typically constructed via interpolation from observational
data or control inputs.
The dynamics follow:
                                              dZ(t)               dU (t)
                                                    = fθ (Z(t)) ·        ,
                                               dt                  dt

where fθ : Rd → Rd×m is a neural network that parameterizes the vector field and dUdt(t) is the derivative of the control
path that we can compute from the interpolation. Using the control path, the model can effectively steer the evolution of
the hidden state to capture complex dependencies that are difficult to extract solely from noisy measurements.
To ensure smoothness and differentiability of U (t), we can construct it using cubic Hermite spline interpolation over
discrete control points or other interpolation techniques. The Neural CDE model comprises three main components:

       • Initial Encoder: A neural network maps the initial observation X(t0 ) (and optionally other context) into the
         initial latent state Z(t0 ).
       • Vector Field: The neural function fθ governs the evolution of Z(t) with respect to changes in U (t).

       • Decoder: A linear projection maps each latent state Z(ti ) to an estimate of the time derivative X̂ ′ (ti ) of the
         observed signal.

The system is numerically integrated via the Diffrax python library [28] using a differentiable explicit ODE solver [37]
(for example, Dormand-Prince [38, 39] or Tsitouras 5/4 Runge-Kutta method [40]) using adaptive time-stepping [41, 42],
enabling end-to-end training via discretise-then-optimise automatic differentiation [43, 44, 45]. The learned derivatives
serve as a smooth and robust approximation to the system’s underlying dynamics, particularly beneficial for symbolic
regression, which relies on clean gradient information.

State Trajectory Reconstruction via Integration

While Neural CDEs are often used to directly predict state trajectories, we instead first estimate the time derivatives,

                                               X̂ ′ (ti ) = Decoder(Z(ti )).


                                                            8
Afterward, reconstruct the state X̂(t) by numerical integration. For instance, applying a standard integrator or simple
methods, such as the trapezoidal rule, we compute:
                                                i−1
                                                X    1 ′                      
                          X̂(ti ) ≈ X(t0 ) +            X̂ (tj ) + X̂ ′ (tj+1 ) · (tj+1 − tj ) .
                                                j=0
                                                     2

This two-stage approach, derivative estimation followed by integration, provides a noise-robust estimate of the full
state-trajectory. More importantly, the estimated derivatives X̂ ′ (t) can then be passed to symbolic regression algorithms
to discover interpretable models.

3     Experiments
We demonstrate the effectiveness of our proposed KAN-SR framework on two different tasks integral to scientific
discovery. First, we evaluate KAN-SR’s performance on the SRSD-Feynmann [16] dataset, which is an adaptation of
the widely adopted Feynmann equation dataset. In a second part, we extend the algorithm to the dynamic modelling of
a bioprocess using neural-differential equations to extract the rate laws governing the process in a two-step approach.
It is important to mention here that all experiments were run without the additional Deeper-KAN loop (Figure 2),
to showcase the additional value of searching for simplifications. This allows us to circumvent using more complex
models which may find solutions that are less interpretable.

3.1   SRSD-Dataset

The SRSD-Dataset [16] compared to the popular Feynman equation dataset, has more realistic sampling ranges of the
independent variables and additional dummy variables. The improved sampling strategy makes the individual problems
much more complex, as variables, depending on the set, can span many orders of magnitude, making the search much
more difficult.
Contrary to standard machine learning methods, we are interested in finding not only low fitting error equations but also
rediscovering the true underlying function. Our main objective is therefore the solution rate mentioned in La Cava et
al. [3] and not solely the accuracy metrics such as the mean squared error or the coefficient of determination.
Since in a scientific environment measurements are noisy, we also added different levels of white Gaussian noise to the
dependent variable y as a fraction of the signal root mean squared value. [3]

                                                                   v         
                                                                    u
                                                                    u1 X N
                                     ynoise = y + ϵ,    ϵ ∼ N 0, γ t      y2                                          (2)
                                                                      N i=1 i

We compare our framework to other SR tools benchmarked in the SRSD paper [16], namely gplearn, AFP, AFP-FE,
AIF, DSR, E2E, uDSR and PySR [46, 47, 7, 15, 12, 11, 14, 5]. For details of the baseline models, we refer the reader to
the corresponding papers. For the implementation and hyperparameters of the models used in the benchmark, we refer
to the original SRSD paper [16].

3.2   Solution Rate on the SRSD Dataset

All methods benchmarked in this study, with the exception of KAN-SR, were evaluated in Matsubara et al. [16].

Baseline results

    Group        gplearn      AFP       AFP-FE         AIF           DSR     E2E       uDSR       PySR       KAN-SR

    Easy          6.67%      20.0%       23.3%         30.0%     46.7%      0.00%      50.0%      60.0%        93.3%
    Medium        0.00%      2.50%       2.50%         2.50%     10.0%      0.00%      17.5%      30.0%        60.0%
    Hard          0.00%      0.00%       0.00%         2.00%     2.00%      0.00%      4.00%      4.00%        12.0%
            Table 1: Results of the solution rate [3] of different methods on the SRSD-Feynman [16] dataset.



                                                             9
Results Including Dummy Variables


  Group        gplearn       AFP       AFP-FE        AIF        DSR        E2E      uDSR       PySR       KAN-SR

  Easy          0.00%       16.7%       16.7%        0.00%     10.0%      0.00%     10.0%      20.0%       86.7%
  Medium        0.00%       0.00%       0.00%        0.00%     0.00%      0.00%     7.50%      5.00%       55.0%
  Hard          0.00%       0.00%       0.00%        0.00%     2.00%      0.00%     0.00%      0.00%       10.0%
      Table 2: Solution rate [3] of different methods on the SRSD-Feynman dataset with dummy variables [16].


Impact of Noise


                         Noise-Level     Easy-Dataset     Medium-Dataset      Hard-Dataset

                         None                93.3%             60.0%              12.0%
                         0.001               80.0%             45.0%              8.00%
                         0.01                76.7%             45.0%              8.00%
                         0.1                 53.3%             25.0%              4.00%
Table 3: Solution rate [3] of KAN-SR on the SRSD-Feynman datasets [16] with varying noise levels according to
Equation 2.


KAN-SR achieved the highest solution rates for all levels of difficulty in the SRSD-Feynman dataset, significantly
outperforming previous methods (Table 1). In Easy problems, it solved 93.3% of the tasks, compared to 60.0% by the
next-best method. Even on the Medium and Hard problems, where most other models struggled, KAN-SR reached
60.0% and 12.0% solution rates, respectively. When dummy variables were added to the equations, performance
dropped for all methods, but KAN-SR remained far ahead, solving 86.7% of Easy, 55.0% of Medium, and 10.0% of
Hard problems (Table 2).
We also tested how well KAN-SR handled noise in the data. As expected, performance decreased as noise increased,
but the model remained rather robust, solving over half of the Easy problems even with 0.1 noise (Table 3). Overall,
these results show that KAN-SR is not only accurate, but also robust with respect to irrelevant features and noisy data.

3.3   In-Silico Bioprocess Model

To evaluate our framework on dynamic systems, we benchmark it using an in-silico model of batch fermentation [48].
The system is described by the following set of ordinary differential equations (ODEs):


                                 dX
                                     =µ·X
                                  dt
                                 dS       1
                                     =−      ·µ·X
                                  dt    YX/S
                                 dP    YP/S
                                     =      ·µ·X
                                  dt   YX/S
                                                                             
                                                 S      k1 (T )          X
                                  µ = µmax ·         ·            · 1−
                                               KS + S 1 + k2 (T )      Kµ + X

X, S, and P represent the concentrations of biomass, substrate, and product, respectively. The specific growth rate µ
is described using a Monod type model that accounts for substrate saturation, temperature effects on activation and
inactivation, and inhibitory impact at high biomass levels.
The temperature dependence is modeled using Arrhenius-type equations:


                                                          10
                                                                      
                                                                  EA,1
                                             k1 (T ) = A1 · exp −
                                                                  R·T
                                                                      
                                                                  EA,2
                                             k2 (T ) = A2 · exp −
                                                                  R·T


To recover the governing expressions that generated the state variables X, S, and P , we follow a two-step procedure
similar to Forster et al. [2]. First, a neural controlled differential equation (NCDE) model (see Section 2.7) [29, 28] is
trained on observed time series data to obtain smooth derivatives. These estimated derivatives are then used as targets in
a second symbolic regression step that aims to rediscover the underlying kinetic model.
Due to the high cost, typically associated with bioprocess experiments, the number of in-silico simulations was limited
to 12. Each simulation spans 80 hours with uniform sampling every 2 hours, resulting in 40 samples per run. The initial
conditions for biomass (X), substrate (S), and product (P ) were sampled using a Latin hypercube design. The lower
bounds were set to [0.1, 50, 0] and the upper bounds to [0.4, 90, 0.4]. Since the simplified process does not contain
any controlled variable, time is the only input to the control path in this case. Although a standard neural ODE would
suffice without any controlled variables, we use the NCDE framework to support the direct extension to possibly more
complex scenarios involving multiple controlled variables and showcase it here as a possible modeling framework for
dynamical systems in the scope of symbolic regression.


                    Symbol     Value         Unit                           Description
                    µmax       0.25          h−1                  Maximum specific growth rate
                    KS         105.4         kg/m3                   Monod saturation constant
                    YX/S       0.07          dimensionless          Biomass yield on substrate
                    YP/S       0.167         dimensionless           Product yield on substrate
                    Kµ         121.87        g/L                  Inhibition constant for biomass
                    T          308.15        K                     Operating temperature (35°C)
                    R          0.0083145     kJ/(K·mol)                Universal gas constant
                    A1         130.03        dimensionless       Pre-exponential factor (activation)
                    EA,1       12.43         kJ/mol                      Activation energy
                    A2         3.83·104      dimensionless      Pre-exponential factor (inactivation)
                    EA,2       298.55        kJ/mol                     Inactivation energy
                     Table 4: Model parameters for the in-silico batch fermentation ODE system.



With these parameters, the system reduces to:



                                          dX         30.87 · X · S
                                              =                                                                        (3)
                                           dt   (X + 121.87)(S + 105.4)
                                          dS          440.99 · X · S
                                              =−                                                                       (4)
                                           dt    (X + 121.87)(S + 105.4)
                                          dP         73.65 · X · S
                                              =                                                                        (5)
                                           dt   (X + 121.87)(S + 105.4)


To test whether the NCDE model can extract smooth derivatives in the presence of observation noise, synthetic noise
was added to the state variables using Equation 2 with γ = 0.02. Symbolic regression was then applied to the noisy
derivative estimates.
The trained NCDE model is able to model the process well, looking at Figure 4 where a sample run is shown.


                                                           11
Figure 4: Sample concentration profiles of the NCDE model. The blue lines represent the predicted values by the NCDE
model, the orange lines are the noisy, observed measurements and the green lines are the noise-free measurements.



Looking at the parity plot of the extracted derivatives (Figure 5), using the NCDE model, we are able to obtain smooth
derivatives which are in good agreement with the real derivatives.




                                 Figure 5: Extracted derivatives of the NCDE model.


These derivatives are then symbolically regressed using KAN-SR. The resulting system of ODEs, rounded to two
decimal places, is:



                                                dX    0.25 · X · S
                                                    =
                                                 dt   S + 103.51
                                                dS      0.34 · X · S
                                                    =−
                                                 dt     S + 118.76
                                                dP    0.70 · X · S
                                                    =
                                                 dt   S + 129.51


Compared to Equations 3, 4, and 5, these expressions are structurally similar. However, the symbolic regression model
does not recover the biomass-dependent denominator, suggesting that it did not identify the self-inhibition effect.
Despite this, the parity plot in Figure 6 shows that the model-generated state trajectories agree closely with observed
test data, sampled within the same initial bounds. This indicates that the symbolic model can reproduce the system
dynamics with only minor deviations.


                                                          12
    Figure 6: Parity plot of the different species concentrations after integrating the obtained symbolic derivatives.


4   Conclusion
Here, we presented KAN-SR as a symbolic regression framework which is capable of recovering mathematical
expressions from noisy, high-dimensional data, as shown for numerical examples. Across the SRSD-Feynman
benchmark, KAN-SR outperformed the investigated baseline methods in terms of solution rate, with notable gains if
irrelevant variables were present in the data. This improvement can be attributed to the divide-and-conquer approach of
combining sparse compositional univariate function learning and the simplification pipeline that assists in discovering
simpler subproblems.
Combining additive and multiplicative KAN layers allows us to represent a wide range of expressions. Furthermore,
the symbolic extraction stage, using a domain-informed library and nonlinear optimization, supports in recovering
closed-form expressions that align with ground truth equations.
The extension to dynamic systems, as demonstrated in the bioprocess modeling case study, suggests that KAN-SR is
capable of identifying governing equations in the context of neural differential equations. However, broader validation
across real-world time-series systems is required to assess the generalizability and robustness of the framework under
varying experimental conditions. Neural CDEs showed a promising way to integrate these dynamical systems into
the KAN-SR framework, but it remains a two-step approach. A cleaner way would be to directly integrate symbolic
regression algorithms into the Neural CDE, but this is out of scope for the proposed framework, due to the iterative
manner of KAN-SR.
Although KAN-SR showed good performance across the SRSD-Feynman dataset, there are several limitations. The
observed superior performance is only obtained when single-layer representations combined with the simplifications
and the univariate library are enough to fit the equation. Our approach breaks down the symbolic regression problem,
which is a large non-convex mixed integer non-linear programming problem in two levels. We restrict our search
space to an upper bound of the number of operators using a fixed number of units in a single KAN layer and then
further restrict the space by only fitting our limited univariate function library. This library by design may not contain
all possible univariate functions that could describe our trainable activation function, and thus represents an inherent
limitation.
Other algorithms such as GP explore the binary operator search space by evolving populations and performing non-linear
constant fitting, while KAN-SR narrows the search space and may find local solutions. The drawback is that, if the
optimal solution lies outside this constrained space, KAN-SR often fails to produce a good fitting equation, unlike other
methods. Larger and deeper KANs can mitigate this, but excessive size makes them diffuse and hinders the extraction
of symbolic expressions, whilst greatly increasing the number of tunable hyperparameters. In the studied physics case,
all data points came from a closed-form equation, meaning the ground truth was recoverable from the training variables,
unlike black-box or real-world problems where key variables might be missing. Although KAN-SR handles Gaussian
noise well, this lack of complete information may prevent the discovery of good fitting equations.




                                                           13
References
 [1] Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz. Discovering governing equations from data by sparse
     identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences, 113(15):3932–
     3937, 2016.
 [2] Tim Forster, Daniel Vázquez, Claudio Müller, and Gonzalo Guillén-Gosálbez. Machine learning uncovers
     analytical kinetic models of bioprocesses. Chemical Engineering Science, 300, 12 2024.
 [3] William La Cava, Patryk Orzechowski, Bogdan Burlacu, Fabrício Olivetti de França, Marco Virgolin, Ying
     Jin, Michael Kommenda, and Jason H. Moore. Contemporary symbolic regression methods and their relative
     performance, 2021.
 [4] John R Koza. Genetic programming as a means for programming computers by natural selection. Technical
     report, 1994.
 [5] Miles Cranmer. Interpretable machine learning for science with pysr and symbolicregression.jl, 2023.
 [6] Roger Guimerà, Ignasi Reichardt, Antoni Aguilar-Mogas, Francesco A. Massucci, Manuel Miranda, Jordi Pallarès,
     and Marta Sales-Pardo. A bayesian machine scientist to aid in the solution of challenging scientific problems.
     Science Advances, 6(5):eaav6971, 2020.
 [7] Michael Schmidt and Hod Lipson. Distilling free-form natural laws from experimental data. Science, 324(5923):81–
     85, 2009.
 [8] Ying Jin, Weilin Fu, Jian Kang, Jiadong Guo, and Jian Guo. Bayesian symbolic regression, 2020.
 [9] Luca Biggio, Tommaso Bendinelli, Alexander Neitz, Aurelien Lucchi, and Giambattista Parascandolo. Neural
     symbolic regression that scales, 2021.
[10] Arya Grayeli, Atharva Sehgal, Omar Costilla-Reyes, Miles Cranmer, and Swarat Chaudhuri. Symbolic regression
     with a learned concept library, 2024.
[11] Pierre-Alexandre Kamienny, Stéphane d’Ascoli, Guillaume Lample, and François Charton. End-to-end symbolic
     regression with transformers, 2022.
[12] Brenden K. Petersen, Mikel Landajuela, T. Nathan Mundhenk, Claudio P. Santiago, Soo K. Kim, and Joanne T.
     Kim. Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients,
     2021.
[13] T. Nathan Mundhenk, Mikel Landajuela, Ruben Glatt, Claudio P. Santiago, Daniel M. Faissol, and Brenden K.
     Petersen. Symbolic regression via neural-guided genetic programming population seeding, 2021.
[14] Mikel Landajuela, Chak Shing Lee, Jiachen Yang, Ruben Glatt, Claudio Santiago prata, llnlgov T Nathan Mund-
     henk, Ignacio Aravena, Garrett Mulcahy, and Brenden Petersen. A unified framework for deep symbolic regression.
     In Proceedings of the 36th International Conference on Neural Information Processing Systems. Curran Associates
     Inc., 11 2022.
[15] Silviu-Marian Udrescu and Max Tegmark. AI Feynman: A physics-inspired method for symbolic regression.
     Science Advances, 6(16):eaay2631, 2020.
[16] Yoshitomo Matsubara, Naoya Chiba, Ryo Igarashi, and Yoshitaka Ushiku. Rethinking symbolic regression
     datasets and benchmarks for scientific discovery, 2024.
[17] Ben Cohen, Burcu Beykal, and George M. Bollas. Data-driven Discovery of Reaction Kinetic Models in Dynamic
     Plug Flow Reactors using Symbolic Regression, volume 53, pages 2947–2952. Elsevier B.V., 1 2024.
[18] Zachary T. Wilson and Nikolaos V. Sahinidis. The alamo approach to machine learning. Computers and Chemical
     Engineering, 106:785–795, 11 2017.
[19] Miguel Ángel de Carvalho Servia, Ilya Orson Sandoval, Klaus Hellgardt, King Kuok, Hii, Dongda Zhang, and
     Ehecatl Antonio del Rio Chanona. The automated discovery of kinetic rate models – methodological frameworks,
     2023.
[20] Baicheng Weng, Zhilong Song, Rilong Zhu, Qingyu Yan, Qingde Sun, Corey G. Grice, Yanfa Yan, and Wan Jian
     Yin. Simple descriptor derived from symbolic regression accelerating the discovery of new perovskite catalysts.
     Nature Communications, 11, 12 2020.
[21] Mehrad Ansari, Heta A. Gandhi, David G. Foster, and Andrew D. White. Iterative symbolic regression for learning
     transport equations. AIChE Journal, 68(6):e17695, 2022.
[22] Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, and
     Max Tegmark. Kan: Kolmogorov-arnold networks, 2025.


                                                        14
[23] Ziming Liu, Pingchuan Ma, Yixuan Wang, Wojciech Matusik, and Max Tegmark. Kan 2.0: Kolmogorov-arnold
     networks meet science, 2024.
[24] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George
     Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transforma-
     tions of Python+NumPy programs, 2018.
[25] Patrick Kidger and Cristian Garcia. Equinox: neural networks in JAX via callable PyTrees and filtered transforma-
     tions. Differentiable Programming workshop at Neural Information Processing Systems 2021, 2021.
[26] Jason Rader, Terry Lyons, and Patrick Kidger. Optimistix: modular optimisation in jax and equinox, 2024.
[27] DeepMind, Igor Babuschkin, Kate Baumli, Alison Bell, Surya Bhupatiraju, Jake Bruce, Peter Buchlovsky, David
     Budden, Trevor Cai, Aidan Clark, Ivo Danihelka, Antoine Dedieu, Claudio Fantacci, Jonathan Godwin, Chris
     Jones, Ross Hemsley, Tom Hennigan, Matteo Hessel, Shaobo Hou, Steven Kapturowski, Thomas Keck, Iurii
     Kemaev, Michael King, Markus Kunesch, Lena Martens, Hamza Merzic, Vladimir Mikulik, Tamara Norman,
     George Papamakarios, John Quan, Roman Ring, Francisco Ruiz, Alvaro Sanchez, Laurent Sartran, Rosalia
     Schneider, Eren Sezener, Stephen Spencer, Srivatsan Srinivasan, Miloš Stanojević, Wojciech Stokowiec, Luyu
     Wang, Guangyao Zhou, and Fabio Viola. The DeepMind JAX Ecosystem, 2020.
[28] Patrick Kidger. On Neural Differential Equations. PhD thesis, University of Oxford, 2021.
[29] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations,
     2019.
[30] Ziyao Li. Kolmogorov-arnold networks are radial basis function networks, 2024.
[31] Athanasios Delis. Fasterkan. https://github.com/AthanasiosDelis/faster-kan/, 2024.
[32] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus), 2023.
[33] Aaron Meurer, Christopher P. Smith, Mateusz Paprocki, Ondřej Čertík, Sergey B. Kirpichev, Matthew Rocklin,
     Amit Kumar, Sergiu Ivanov, Jason K. Moore, Sartaj Singh, Thilina Rathnayake, Sean Vig, Brian E. Granger,
     Richard P. Muller, Francesco Bonazzi, Harsh Gupta, Shivam Vats, Fredrik Johansson, Fabian Pedregosa, Matthew J.
     Curry, Andy R. Terrel, Štěpán Roučka, Ashutosh Saboo, Isuru Fernando, Sumith Kulal, Robert Cimrman, and
     Anthony Scopatz. Sympy: symbolic computing in python. PeerJ Computer Science, 3:e103, January 2017.
[34] Kevin Egan, Weizhen Li, and Rui Carvalho. Automatically discovering ordinary differential equations from data
     with sparse regression. Communications Physics, 7, 12 2024.
[35] Stéphane d’Ascoli, Sören Becker, Alexander Mathis, Philippe Schwaller, and Niki Kilbertus. Odeformer: Symbolic
     regression of dynamical systems with transformers, 2023.
[36] Patrick Kidger, James Morrill, James Foster, and Terry Lyons. Neural controlled differential equations for irregular
     time series, 2020.
[37] E. Hairer, S.P. Nørsett, and G. Wanner. Solving Ordinary Differential Equations I Nonstiff Problems. Springer,
     Berlin, second revised edition edition, 2008.
[38] J. R. Dormand and P. J. Prince. A family of embedded Runge–Kutta formulae. J. Comp. Appl. Math, 6:19–26,
     1980.
[39] Lawrence F. Shampine. Some practical Runge-Kutta formulas. Mathematics of Computation, 46(173):135–150,
     1986.
[40] Ch Tsitouras. Runge–kutta pairs of order 5 (4) satisfying only the first column simplifying assumption. Computers
     & Mathematics with Applications, 62(2):770–775, 2011.
[41] E. Hairer and G. Wanner. Solving Ordinary Differential Equations II Stiff and Differential-Algebraic Problems.
     Springer, Berlin, second revised edition edition, 2002.
[42] Gustaf Söderlind. Automatic control and adaptive time-stepping. Numerical Algorithms, 31:281–310, 2002.
[43] Yingbo Ma, Vaibhav Dixit, Michael J Innes, Xingjian Guo, and Chris Rackauckas. A comparison of automatic
     differentiation and continuous sensitivity analysis for derivatives of differential equation solutions. In 2021 IEEE
     High Performance Extreme Computing Conference (HPEC), pages 1–9, 2021.
[44] Philipp Stumm and Andrea Walther. New algorithms for optimal online checkpointing. SIAM Journal on Scientific
     Computing, 32(2):836–854, 2010.
[45] Qiqi Wang, Parviz Moin, and Gianluca Iaccarino. Minimal repetition dynamic checkpointing algorithm for
     unsteady adjoint calculation. SIAM Journal on Scientific Computing, 31(4):2549–2567, 2009.
[46] John R. Koza and Riccardo Poli. Genetic Programming, pages 127–164. Springer US, Boston, MA, 2005.


                                                          15
[47] Michael Schmidt and Hod Lipson. Age-Fitness Pareto Optimization, pages 129–146. Springer New York, New
     York, NY, 2011.
[48] Richard Turton, Richard C Bailie, Wallace B Whiting, and Joseph A Shaeiwitz. Analysis, synthesis and design of
     chemical processes. Pearson Education, 2008.




                                                        16

