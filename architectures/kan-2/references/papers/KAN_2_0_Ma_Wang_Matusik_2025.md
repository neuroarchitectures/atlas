# KAN 2 0 Ma Wang Matusik 2025

> Source: `KAN_2_0_Ma_Wang_Matusik_2025.pdf`

---

                                                                  KAN 2.0:
                                                    Kolmogorov-Arnold Networks Meet Science


                                          Ziming Liu1,4∗ Pingchuan Ma1,3 Yixuan Wang2 Wojciech Matusik1,3 Max Tegmark1,4
                                                                       1
                                                                         Massachusetts Institute of Technology
                                                                         2
                                                                           California Institute of Technology




arXiv:2408.10205v1 [cs.LG] 19 Aug 2024
                                                       3
                                                         Computer Science and Artificial Intelligence Laboratory (CSAIL), MIT
                                                     4
                                                        The NSF Institute for Artificial Intelligence and Fundamental Interactions



                                                                                       Abstract

                                                    A major challenge of AI + Science lies in their inherent incompatibility: to-
                                                    day’s AI is primarily based on connectionism, while science depends on sym-
                                                    bolism. To bridge the two worlds, we propose a framework to seamlessly syner-
                                                    gize Kolmogorov-Arnold Networks (KANs) and science. The framework high-
                                                    lights KANs’ usage for three aspects of scientific discovery: identifying relevant
                                                    features, revealing modular structures, and discovering symbolic formulas. The
                                                    synergy is bidirectional: science to KAN (incorporating scientific knowledge into
                                                    KANs), and KAN to science (extracting scientific insights from KANs). We high-
                                                    light major new functionalities in pykan: (1) MultKAN: KANs with multiplica-
                                                    tion nodes. (2) kanpiler: a KAN compiler that compiles symbolic formulas into
                                                    KANs. (3) tree converter: convert KANs (or any neural networks) to tree graphs.
                                                    Based on these tools, we demonstrate KANs’ capability to discover various types
                                                    of physical laws, including conserved quantities, Lagrangians, symmetries, and
                                                    constitutive laws.




                                                          Figure 1: Synergizing science and the Kolmogorov-Arnold Network (KAN).



                                            ∗
                                                zmliu@mit.edu



                                         Preprint. Under review.
1   Introduction
In recent years, AI + Science has emerged as a promising new field, leading to significant scien-
tific advancements including protein folding prediction [37], automated theorem proving [95, 83],
weather forecast [41], among others. A common thread among these tasks is that they can all be
well formulated into problems with clear objectives, optimizable by black-box AI systems. While
this paradigm works exceptionally well for application-driven science, a different kind of science
exists: curiosity-driven science. In curiosity-driven research, the procedure is more exploratory,
often lacking clear goals beyond “gaining more understanding”. To clarify, curiosity-driven sci-
ence is far from useless; quite the opposite. The scientific knowledge and understanding gained
through curiosity often lay a solid foundation for tomorrow’s technology and foster a wide range of
applications.
Although both application-driven and curiosity-driven science are invaluable and irreplaceable, they
ask different questions. When astronomers observe the motion of celestial bodies, application-driven
researchers focus on predicting their future states, while curiosity-driven researchers explore the
physics behind the motion. Another example is AlphaFold, which, despite its tremendous success
in predicting protein structures, remains in the realm of application-driven science because it does
not provide new knowledge at a more fundamental level (e.g., atomic forces). Hypothetically, Al-
phaFold must have uncovered important unknown physics to achieve its highly accurate predictions.
However, this information remains hidden from us, leaving AlphaFold largely a black box. There-
fore, we advocate for new AI paradigms to support curiosity-driven science. This new paradigm of
AI + Science demands a higher degree of interpretability and interactivity in AI tools so that they
can be seamlessly integrated into scientific research.
Recently, a new type of neural network called Kolmogorov-Arnold Network (KAN) [57], has shown
promise for science-related tasks. Unlike multi-layer perceptrons (MLPs), which have fixed activa-
tion functions on nodes, KANs feature learnable activation functions on edges. Because KANs
can decompose high-dimensional functions into one-dimensional functions, interpretability can be
gained by symbolically regressing these 1D functions. However, their definition of interpretability is
somewhat narrow, equating it almost exclusively with the ability to extract symbolic formulas. This
limited definition restricts their scope, as symbolic formulas are not always necessary or feasible in
science. For example, while symbolic equations are powerful and prevalent and physics, systems
in chemistry and biology the systems are often too complex to be represented by such equations.
In these fields, modular structures and key features may be sufficient to characterize interesting
aspects of these systems. Another overlooked aspect is the reverse task of embedding knowledge
into KANs: How can we incorporate prior knowledge into KANs, in the spirit of physics-informed
learning?
We enhance and extend KANs to make them easily used for curiosity-driven science. The goal of
this paper can be summarized as follows:

              Goal: Synergize Kolmogorov-Arnold Networks ⇔ Science.
              ⇐: Build in scientific knowledge to KANs (Section 3).
              ⇒: Extract out scientific knowledge from KANs (Section 4).

To be more concrete, scientific explanations may have different levels, ranging from the coars-
est/easiest/correlational to the finest/hardest/causal:
       • Important features: For example, “y is fully determined by x1 and x2 , while other factors
         do no matter.” In other words, there exists a function f such that y = f (x1 , x2 ).
       • Modular structures: For instance, “x1 and x2 contributes to y independently in an additive
         way.“ This means there exists functions g and h such that y = g(x1 ) + h(x2 ).
       • Symbolic formulas: For example, “y depends on x1 as a sine function and on x2 as an
         exponential function”. In other words, y = sin(x1 ) + exp(x2 ).
The paper reports on how to incorporate and extract these properties from KANs. The structure of
the paper is as follows (illustrated in Figure 1): In Section 2, we augment the original KAN with
multiplication nodes, introducing a new model called MultKAN. In Section 3, we explore ways to
embed scientific inductive biases into KANs, focusing on important features (Section 3.1), modular


                                                  2
Figure 2: Top: comparing KAN and MultKAN diagrams. MultKAN has extra multiplication layers M. Bot-
tom: After training on f (x, y) = xy, KAN learns an algorithm requiring two addition nodes, while MultKAN
requires only one multiplication node.



structures (Section 3.2), and symbolic formulas (Section 3.3). In Section 4, we propose methods to
extract scientific knowledge from KANs, again covering important features (Section 4.1), modular
structures (Section 4.2), and symbolic formulas (Section 4.3). In Section 5, we apply KANs to
various scientific discovery tasks using the tools developed in the previous sections. These tasks
include discovering conserved quantities, symmetries, Lagrangians, and constitutive laws. Codes
are available at https://github.com/KindXiaoming/pykan and can also be installed via pip
install pykan. Although the title of the paper is “KAN 2.0”, the release version of pykan is
0.2.x.

2   MultKAN: Augmenting KANs with multiplications
The Kolmogorov-Arnold representation theorem (KART) states that any continuous high-
dimensional function can be decomposed into a finite composition of univariate continuous func-
tions and additions:
                                                     2n+1    n
                                                                         !
                                                      X      X
                      f (x) = f (x1 , · · · , xn ) =      Φq   ϕq,p (xp ) .                 (1)
                                                         q=1     p=1

This implies that addition is the only true multivariate operation, while other multivariate operations
(including multiplication) can be expressed as additions combined with univariate functions. For
example, to multiply two positive numbers x and y, we can express this as xy = exp(logx + logy) 2
whose right-hand side only consists of addition and univariate functions (log and exp).

    2
      If x and y can be negative, one may choose a large c > 0 and express xy = exp(log(x + c) + log(y +
c)) − c(x + y) − c2 . Other constructions include quadratic functions, such as xy = ((x + y)2 − (x − y)2 )/4
or xy = ((x + y)2 − x2 − y 2 )/2.


                                                     3
However, given the prevalence of multiplications in both science and everyday life, it is desirable
to explicitly include multiplications in KANs, which could potentially enhance both interpretability
and capacity.
Kolmogorov-Arnold Network (KAN) While the KART Eq. (1) corresponds to a two-layer net-
work, Liu et al. [57] managed to extend it to arbitrary depths by recognizing that seemingly different
outer functions Φq and inner functions ϕq,p can be unified through their proposed KAN layers. A
depth-L KAN can be constructed simply by stacking L KAN layers. The shape of a depth-L KAN is
represented by an integer array [n0 , n1 , · · · , nL ] where nl denotes the number of neurons in the lth
neuron layers. The lth KAN layer, with nl input dimensions and nl+1 output dimensions, transforms
an input vector xl ∈ Rnl to xl+1 ∈ Rnl+1
                               ϕl,1,1 (·)          ϕl,2,1 (·)  ···    ϕl,nl ,1 (·)
                                                                                  
                             ϕl,1,2 (·)           ϕl,2,2 (·)  ···    ϕl,nl ,2 (·) 
                    xl+1 =        .
                                   ..                  .
                                                       ..                 ..        xl ,             (2)
                                                                          .       
                                  ϕl,1,nl+1 (·) ϕl,2,nl+1 (·) · · ·   ϕl,nl ,nl+1 (·)
                              |                         {z                              }
                                                           Φl

and the whole network is a composition of L KAN layers, i.e.,
                                  KAN(x) = (ΦL−1 ◦ · · · ◦ Φ1 ◦ Φ0 )x.                                     (3)
In diagrams, KANs can be intuitively visualized as a network consisting of nodes (summation) and
edges (learnable activations), as shown in Figure 2 top left. When trained on the dataset generated
from f (x, y) = xy, the KAN (Figure 2 bottom left) uses two addition nodes, making it unclear
what the network is doing. However, after some consideration, we realize it leverages the equality
xy = ((x + y)2 − (x − y)2 )/4 but this is far from obvious.
Multiplicative Kolmogorov-Arnold Networks (MultKAN) To explicitly introduce multiplication
operations, we propose the MultKAN, which can reveal multiplicative structures in data more
clearly. A MultKAN (shown in Figure 2 top right) is similar to a KAN, with both having stan-
dard KAN layers. We refer to the input nodes of a KAN layer as nodes, and the output nodes of a
KAN layer subnodes. The difference between KAN and MultKAN lies in the transformations from
the current layer’s subnodes to the next layer’s nodes. In KANs, nodes are directly copied from the
previous layer’s subnodes. In MultKANs, some nodes (addition nodes) are copied from correspond-
ing subnodes, while other nodes (multiplication nodes) perform multiplication on k subnodes from
the previous layer. For simplicity, we set k = 2 below 3 .
Based on the MultKAN diagram (Figure 2 top right), it can be intuitively understood that a
MultKAN is a normal KAN with optional multiplications inserted in. To be mathematically pre-
cise, we define the following notations: The number of addition (multiplication) operations in
layer l are denoted as nal (nm       l ), respectively. These are collected into arrays: addition width
na ≡ [na0 , na1 , · · · , naL ] and multiplication width nm ≡ [nm     m            m          m    m
                                                                 0 , n1 , · · · , nL ]. When n0 = n1 =
         m
· · · = nL = 0, the MultKAN reduces to a KAN. For example, Figure 2 (top right) shows a
MultKAN with na = [2, 2, 1] and nm = [0, 2, 0].
A MultKAN layer consists of a standard KANLayer Φl and a multiplication layer Ml . Φl takes in
                          a   m                                 a     m
an input vector xl ∈ Rnl +nl and outputs zl = Φl (x) ∈ Rnl+1 +2nl+1 . The multiplication layer
consists of two parts: the multiplication part performs multiplications on subnode pairs, while the
other part performs identity transformation. Written in Python, Ml transforms zl as follows:
                                                                                            a     m
        Ml (zl ) = concatenate(zl [: nal+1 ], zl [nal+1 :: 2] ⊙ zl [nal+1 + 1 :: 2]) ∈ Rnl+1 +nl+1 ,       (4)
where ⊙ is element-wise multiplication. The MultKANLayer can be succinctly represented as Ψl ≡
Ml ◦ Φl . The whole MultKAN is thus:
                          MultKAN(x) = (ΨL ◦ ΨL−1 ◦ · · · ◦ Ψ1 ◦ Ψ0 )x.                                    (5)
Since there are no trainable parameters in multiplication layers, all sparse regularization techniques
(e.g., ℓ1 and entropy regularization) for KANs [57] can be directly applied to MultKANs. For
    3
      We set k = 2 for simplicity, but the pykan package allows k to be any integer k ≥ 2. Users can even set
different k values for different multiplication nodes. However, if different ks values are used within the same
layer, it can be challenging to parallelize these multiplications.


                                                       4
Figure 3: Adding
            p        auxiliary variables to inputs enhances interpretability. For the relativistic mass equation,
m = m0 / 1 − v 2 /cp     2 , (a) a two-layer KAN is needed if only (m , v, c) are used as inputs. (b) If we add
                                                                       0
β ≡ v/c and γ ≡ 1/ 1 − β 2 as auxiliary variables to KANs, a one-layer KAN suffices (seed 0). (c) seed 1
finds a different solution, which is sub-optimal and can be avoided through hypothesis testing (Section 4.3).


the multiplication task f (x, y) = xy, the MultKAN indeed learns to use one multiplication node,
making it perform simple multiplication, as all the learned activation functions are linear (Figure 2
bottom right).
Although KANs have previously been seen as a special case of MultKANs, we extend the definition
and treat “KAN” and “MultKAN” as synonyms. By default, when we refer to KANs, multiplication
is allowed. If we specifically refer to a KAN without multiplication, we will explicitly state so.

3    Science to KANs
In science, domain knowledge is crucial, allowing us to work effectively even with small or zero
data. Therefore, it is beneficial to adopt a physics-informed approach for KANs: we should in-
corporate available inductive biases into KANs while preserving their flexibility to discover new
physics from data.
We explore three types of inductive biases that can be integrated into KANs. From the coars-
est/easiest/correlational to the finest/hardest/causal, they are important features (Section 3.1), mod-
ular structures (Section 3.2) and symbolic formulas (Section 3.3).
3.1 Adding important features to KANs
In a regression problem, the goal is to find a function f such that y = f (x1 , x2 , · · · , xn ). Suppose
we want to introduce an auxiliary input variable a = a(x1 , x2 , . . . , xn ), transforming the function
to y = f (x1 , · · · , xn , a). Although the auxiliary variable a does not add new information, it can
increase the expressive power of the neural network. This is because the network does not need to
expend resources to calculate the auxiliary variable. Additionally, the computations may become
simpler, leading to improved interpretability. Users can add auxiliary features to inputs using the
augment_input method:
    model.augment_input(original_variables, auxiliary_variables, dataset)                                    (6)
                                                                                  p
As an example, consider the formula for relativistic mass m(m0 , v, c) = m0 / 1 − (v/c)2 where
m0 is the rest mass, v is the velocity of the point mass, and c is p
                                                                   the speed of light.
                                                                                   p Since physicists
often work with dimensionless numbers β ≡ v/c and γ ≡ 1/ 1 − β 2 ≡ 1/ 1 − (v/c)2 , they
might introduce β and γ alongside v and c as inputs. Figure 3, shows KANs with and without these
auxiliary variables: (a) illustrates the KAN compiled from the symbolic formula (see Section 3.3 for
the KAN compiler), which requires 5 edges; (b)(c) shows KANs with auxiliary variables, requiring
only 2 or 3 edges and achieving loses of 10−6 and 10−4 , respectively. Note that (b) and (c) differ
only in random seeds. Seed 1 represents a sub-optimal solution because it also identifies
                                                                                       p    β = v/c
as a key feature. This is not surprising, as in the classical limit v ≪ c, γ ≡ 1/ 1 − (v/c)2 ≈
1 + (v/c)2 /2 = 1 + β 2 /2. The variation due to different seeds can be seen either as a feature or
a bug: As a feature, this diversity can help find sub-optimal solutions which may nevertheless offer


                                                       5
       Figure 4: Building modular structures to KANs: (a) multiplicative separability;(b) symmetries.


interesting insights; as a bug, it can be eliminated using the hypothesis testing method proposed in
Section 4.3.
3.2 Building modular structures to KANs
Modularity is prevalent in nature: for example, the human cerebral cortex is divided into several
functionally distinct modules, each of these modules responsible for specific tasks such as percep-
tion or decision making. This modularity simplifies the understanding of neural networks, as it
allows us to interpret clusters of neurons collectively rather than analyzing each neuron individually.
Structural modularity is characterized by clusters of connections where intra-cluster connections are
much stronger than inter-cluster ones. To enforce modularity, we introduce the module method,
which preserves intra-cluster connections while removing inter-cluster connections. The modules
are specified by users. The syntax is
   model.module(start_layer_id, ‘[nodes_id]->[subnodes_id]->[nodes_id]...’)
                                                                                             (7)
For example, if a user wants to assign specific nodes/subnodes to a module – say, the 0th node
in layer 1, the 1st and 3rd subnode in layer 1, the 1st and 3rd node in layer 2 – they might use
module(1,‘[0]->[1,3]->[1,3]’). To be concrete, there are two types of modularity: separabil-
ity and symmetry.
Separability We say a function is considered separable if it can be expressed as a
sum or product of functions of non-overlapping variable groups. For example, a four-
variable function f (x1 , x2 , x3 , x4 ) is maximally multiplicatively separable if it has the form
f1 (x1 )f2 (x2 )f3 (x3 )f4 (x4 ), creating four distinct groups (1), (2), (3), (4). Users can create these
modules by calling the module method four times: module(0,‘[i]->[i]’), i = 0, 1, 2, 3,
shown in Figure 4 (a). The final call may be skipped since the first three are sufficient to de-
fine the groups. Weaker forms of multiplicative separability might be f1 (x1 , x2 )f2 (x3 , x4 ) (calling
module(0,‘[0,1]->[0,1]’)) or f1 (x1 )f2 (x2 , x3 , x4 ) (calling module(0,‘[0]->[0]’)).
Generalized Symmetry We say a function is symmetric in variables (x1 , x2 ) if f (x1 , x2 , x3 , · · · ) =
g(h(x1 , x2 ), x3 , · · · ). This property is termed symmetry because the value of f remains un-
changed as long as h(x1 , x2 ) is constant, even if x1 and x2 vary. p For example, a function f
is rotational invariant in 2D if f (x1 , x2 ) = g(r), where r ≡         x21 + x22 . When symmetry in-
volves only a subset of variables, it can be considered hierarchical since x1 and x2 interact first
through h (2-Layer KAN), and then h interacts with other variables via g (2-Layer KAN). Sup-
pose a four-variable function has a hierarchical form f (x1 , x2 , x3 , x4 ) = h(f (x1 , x2 ), g(x3 , x4 )),


                                                     6
Figure 5: KAN compiler (kanpiler) converts symbolic expressions to KANs. (a) how kanpiler works: the
symbolic formula is first parsed to an expression tree, which is then converted to a KAN. (b) Applying KANs
to 10 equations (selected from the Feynman dataset). (c) Expand a compiled KAN to increase its expressive
power.


as illustrated in Figure 4 (b). We can use the module method to create this structure by call-
ing module(0,‘[0,1]->[0,1]->[0,1]->[0]’), ensuring that the variable groups (x1 , x2 ) and
(x3 , x4 ) do not interact in the first two layers.

3.3   Compiling symbolic formulas to KANs
Scientists often find satisfaction in representing complex phenomena through symbolic equations.
However, while these equations are concise, they may lack the expressive power needed to capture


                                                    7
all nuances due to their specific functional forms. In contrast, neural networks are highly expressive
but may inefficiently spend training time and data to learn domain knowledge already known to
scientists. To leverage the strengths of both approaches, we propose a two-step procedure: (1)
compile symbolic equations into KANs and (2) fine-tune these KANs using data. The first step aims
to embed known domain knowledge into KANs, while the second step focuses on learning new
“physics” from data.
kanpiler (KAN compiler) The goal of the kanpiler is to convert a symbolic formula to a KAN. The
process, illustrated in Figure 5 (a), involves three main steps: (1) The symbolic formula is parsed
into a tree structure, where nodes represent expressions, and edges denote operations/functions. (2)
This tree is then modified to align with the structure of a KAN graph. Modifications include moving
all leaf nodes to the input layer via dummy edges, and adding dummy subnodes/nodes to match
KAN architecture. These dummy edges/nodes/subnodes only perform identity transformation. (3)
The variables are combined in the first layer, effectively converting the tree into a graph. For visual
clarity, 1D curves are placed on edges to represent functions. We have benchmarked the kanpiler on
the Feynman dataset and it successfully handles all 120 equations. Examples are shown in Figure 5
(b). The kanpiler takes input variables (as sympy symbols) and output expression (as a sympy
expression), and returns a KAN model
                  model = kanpiler(input_variables, output_expression)                                  (8)
Note that the returned KAN model is in the symbolic mode, i.e., the symbolic functions are exactly
encoded. If we instead use cubic splines to approximate these symbolic functions, we get MSE
losses ℓ ∝ N −8 [57], where N is the number of grid intervals (proportional to the number of model
parameters).
Width/depth expansion for increased expressive power The KAN network generated by the kan-
piler is compact, without no redundant edges, which might limit its expressive power and hinder
further fine-tuning. To address this, we propose expand_width and expand_depth methods to
expand the network to become wider and deeper, as shown in Figure 5 (c). The expansion methods
initially add zero activation functions, which suffer from zero gradients during training. Therefore,
the perturb method should be used to perturb these zero functions into non-zero values, making
them trainable with non-zero gradients.

4         KANs to Science
Today’s black box deep neural networks are powerful, but interpreting these models remains chal-
lenging. Scientists seek not only high-performing models but also the ability to extract meaningful
knowledge from the models. In this section, we focus on enhancing the interpretability of KANs
scientific purposes. We will explore three levels of knowledge extraction from KANs, from the most
basic to the most complex: important features (Section 4.1), modular structures (Section 4.2), and
symbolic formulas (Section 4.3).

4.1       Identifying important features from KANs
Identifying important variables is crucial for many tasks. Given a regression model f where y ≈
f (x1 , x2 , . . . , xn ), we aim to assign scores to the input variables to gauge their importance. Liu et
al. [57], used the function L1 norm to indicate the importance of edges, but this metric could be
problematic as it only considers local information.
To address this, we introduce a more effective attribution score which better reflects the importance
of variables than the L1 norm. For simplicity, let us assume there are multiplication nodes, so we
do not need to differentiate between nodes and subnodes 4 . Suppose we have an L-layer KAN
with width [n0 , n1 , · · · , nL ]. We define El,i,j as the standard deviation of the activations on the
(l, i, j) edge, and Nl,i as the standard deviation of the activations on the (l, i) node. We then define
the node (attribution) score Al,i and the edge (attribution) score Bl,i,j . In [57], we simply defined
Bl,i,j = El,i,j and Al,i = Nl,i . However, this definition fails to account for the later parts of the
network; even if a node or an edge has a large norm itself, it may not contribute to the output if the
rest of the network is effectively a zero function. Therefore, we now compute node and edge scores
      4
    For subnodes belonging to multiplication node, the subnodes inherit their scores from the multiplication
node.


                                                     8
Figure 6: Identifying important features in KANs. (a) comparing the attribution score to the L1 norm used
in Liu et al. [57]. On two synthetic tasks, the attribution score brings more insights than the L1 norm. (b)
Attribution scores can be computed for inputs and used for input pruning.



iteratively from the output layer to the input layer. We set all output dimensions to have unit scores,
i.e., AL,i = 1, i = 0, 1, · · · , nL − 1 5 , and compute scores as follows:
                                                      nl
                                 El,j                 X
              Bl−1,i,j = Al,j          ,   Al−1,i =         Bl−1,i,j ,   l = L, L − 1, · · · , 1.       (9)
                                Nl+1,j                j=0



Comparing El,i,j and Bl,i,j We find that Bl,i,j provides a more accurate reflection of edge im-
portance. In Figure 6, we compare KANs trained on two equations y = exp(sin(πx1 ) + x22 ) and
y = (x21 + x22 )2 + (x23 + x24 )2 and visualize KANs with importance scores being E (L1 norm) or B
(attribution score). For the first equation, attributions scores reveal a cleaner graph than L1 norms,
as many active edges in the first layer do not contribute to the final output due to inactive subsequent
edges. The attribution score accounts for this, resulting in a more meaningful graph. For the second
equation y = (x21 + x22 )2 + (x23 + x24 )2 , we can tell from the symbolic equation that all four vari-
ables are equally important. The attribution scores correctly reflect the equal importance of all four
variables, whereas the L1 norm incorrectly suggests that x3 and x4 are more important than x1 and
x2 .

Pruning inputs based on attribution scores In real datasets, input dimensionality can be large,
but only a few variables may be relevant. To address this, we propose pruning away irrelevant
features based on attribution scores so that we can focus on the most relevant ones. Users can apply
the prune_input to retain only the most relevant variables. For instance, if there are 100 input
                                                                 P99
features ordered by decreasing relevance in the function y = i=0 x2i /2i , xi ∈ [−1, 1], and after
training, only the first five features show significantly higher attribution scores, the prune_input
method will retain only these five features. The pruned network becomes compact and interpretable,
whereas the original KAN with 100 inputs is too dense for straightforward interpretation.

4.2   Identifying modular structures from KANs
Although the attribution score provides valuable insights into which edges or nodes are important,
it does not reveal modular structures, i.e., how the important edges and nodes are connected. In this
part, we aim to uncover modular structures from trained KANs and MLPs by examining two types
of modularity: anatomical modularity and functional modularity.


                                                      9
Figure 7: Inducing anatomical modularity in neural networks through neuron swapping. The approach involves
assigning spatial coordinates to neurons and permuting them to minimize the overall connection cost. For two
tasks (left: multitask parity, right: hierarchical majority voting), neuron swapping works for KANs (top) in
both cases and works for MLPs (bottom) for multitask parity.


4.2.1 Anatomical modularity
Anatomical modularity refers to the tendency for neurons placed close to each other spatially to
have stronger connections than those further apart. Although artificial neural networks lack physical
spatial coordinates, introducing the concept of physical space has been shown to enhance inter-
pretability [51, 52]. We adopt the neuron swapping method from [51, 52], which shortens connec-
tions while preserving the network’s functionality. We call the method auto_swap. The anatomi-
cal modular structure revealed through neuron swapping facilitates easy identification of modules,
even visually, for two tasks shown Figure 7: (1) multitask sparse parity; and (2) hierarchical ma-
jority voting. For multitask sparse parity, we have 10 input bits xi ∈ {0, 1}, i = 1, 2, · · · , 10,
and output yj = x2j−1 ⊕ x2j , j = 1, · · · , 5, where ⊕ denotes modulo 2 addition. The task
exhibits modularity because each output depends only on a subset of inputs. auto_swap suc-
cessfully identifies modules for both KANs and MLPs, with the KAN discovering simpler mod-
ules. For hierarchical majority voting, with 9 input bits xi ∈ {0, 1}, i = 1, · · · , 9, and the output
y = maj(maj(x1 , x2 , x3 ), maj(x4 , x5 , x6 ), maj(x7 , x8 , x9 )), where maj stands for majority voting
(output 1 if two or three inputs are 1, otherwise 0). The KAN reveals the modular structure even
before auto_swap, and the diagram becomes more organized after auto_swap. The MLP shows
some modular structure from the pattern of the first layer weights, indicating interactions among
variables, but the global modular structure remains unclear regardless of auto_swap.
4.2.2 Functional modularity
Functional modularity pertains to the overall function represented by the neural network. Given an
Oracle network where internal details such as weights and hidden layer activations are inaccessible
(too complicated to analyze), we can still gather information about functional modularity through
forward and backward passes at the inputs and outputs. We define three types of functional modu-
larity (see Figure 8 (a)), based largely on [84].
Separability: A function f is additively separable if
                          f (x1 , x2 , · · · xn ) = g(x1 , . . . , xk ) + h(xk+1 , . . . , xn ).       (10)
              2
Note that ∂x∂i ∂x
                f
                  j
                      = 0 when 1 ≤ i ≤ k, k + 1 ≤ j ≤ n. To detect the separability, we can compute
                                                        2
the Hessian matrix H ≡ ∇T ∇f (Hij = ∂x∂i ∂x
                                         f
                                            j
                                              ) and check for block structure. If Hij = 0 for
   5
     Other choices can be made based on the perceived importance of each output dimension, though this is
less critical when outputs are typically one-dimensional.


                                                            10
Figure 8: Detecting functional modularity in KANs. (a) We study three types of functional modularity: sep-
arability (additive or multiplicative), general separability, and symmetry. (b) Applying these tests recursively
converts a function into a tree. Here the function can be symbolic functions (top), KANs (middle) or MLPs
(bottom). Both KANs and MLPs produce correct tree graphs at the end of training but show different training
dynamics.




                                                      11
all 1 ≤ i ≤ k and k + 1 ≤ j ≤ n, then we know f is additively separable. For multiplicative
separability, we can convert it to additive separability by taking the logarithm:
               f (x1 , x2 , · · · xn ) = g(x1 , . . . , xk ) × h(xk+1 , . . . , xn )
                                                                                                            (11)
               log |f (x1 , x2 , · · · , xn )| = log |g(x1 , . . . , xk )| + log |h(xk+1 , . . . , xn )|
                                                                   2
                                                                |
To detect multiplicative separability, we define Hij ≡ ∂∂xlog|f
                                                           i ∂xj
                                                                  , and check for block structure. Users
can call test_separability to test general separability.
Generalized separability: A function f has generalized separability if
                      f (x1 , x2 , · · · xn ) = F (g(x1 , . . . , xk ) + h(xk+1 , . . . , xn )).            (12)
To detect generalized separability, we compute
     ∂f     ∂F ∂g                 ∂f    ∂F ∂h
         =          (1 ≤ i ≤ k),      =         (k + 1 ≤ j ≤ n)
     ∂xi     ∂g ∂xi               ∂xj   ∂h ∂xi
                                                                                                            (13)
     ∂f /∂xi    ∂F/∂g ∂g/∂xi       ∂g/∂xi                                       1
              =                  =        = gxi (x1 , x2 , · · · xk ) ×                          .
     ∂f /∂xj    ∂F/∂h ∂h/∂xj       ∂h/∂xj                               hxj (xk+1 , · · · , xn )
                             ∂F           ∂f /∂xi
where we have used ∂F ∂g = ∂h . Note that ∂f /∂xj is multiplicatively separable, it can be detected by
the separability test proposed above. Users can call test_general_separability to check for
additive or multiplicative separability.
Generalized Symmetry: A function has generalized symmetry (in the first k variables) if
                        f (x1 , x2 , · · · , xn ) = g(h(x1 , · · · , xk ), xk+1 , · · · , xn ).             (14)
We denote y = (x1 , · · · , xk ) and z = (xk+1 , · · · , xn ). This property is called generalized symme-
try because f retains the same value as long as h is held constant, regardless of individual values
                                                                                    ∂g              ∂g
of x1 , · · · , xk . We compute the gradient of f with respect to y: ∇y f = ∂h         ∇y h. Since ∂h   is a
                                                                                                    ∇y f
scalar function, it does not change the direction of ∇y h. Thus, the direction of ∇y f ≡ |∇y f | is
                                                                                          d
independent of z, i.e.,
                                            ∇z (∇ d  y f ) = 0,                                        (15)
which is the condition for symmetry. Users can call the test_symmetry method to check for
symmetries.
Tree converter The three types of functional modularity form a hierarchy: symmetry is the most
general, general separability is intermediate, and separability is the most specific. Mathematically,
               Separability ⊂ Generalized Separability ⊂ Generalized Symmetry                               (16)
To obtain the maximal hierarchy of modular structures, we apply generalized symmetry detection
recursively, forming groups as small as k = 2 variables and extending to all k = n variables. For
example, let us consider an 8-variable function
           f (x1 , · · · , x8 ) = ((x21 + x22 )2 + (x23 + x24 )2 )2 + ((x25 + x26 )2 + (x27 + x28 )2 )2 ,   (17)
which has four k = 2 generalized symmetries, involving groups (x1 , x2 ), (x3 , x4 ), (x5 , x6 ),
(x7 , x8 ); two k = 2 generalized symmetries, involving groups (x1 , x2 , x3 , x4 ) and (x5 , x6 , x7 , x8 ).
As such, each k = 4 group contains two k = 2 groups, demonstrating a hierarchy. For each gen-
eralized symmetry, we can also test if the generalized symmetry is further generalized separable or
separable. Users can use the method plot_tree to obtain the tree graph for a function (the function
could be any Python expressions, neural networks, etc.). For a neural network model, users can
simply call model.tree(). The tree plot can have the style ‘tree’ (by default) or ‘box’.
Examples Figure 8 (b) provides two examples. When the exact symbolic functions are input to
plot_tree, the ground truth tree graphs are obtained. We are particularly interested in whether the
tree converter works for neural networks. For these simple cases, both KANs and MLPs can find
the correct graph if sufficiently trained. Figure 8 (b) (bottom) shows the evolution of the tree graphs
during KAN and MLP training. It is particularly interesting to see how neural networks gradually
learn the correct modular structure. In the first case f (x1 , x2 , x3 , x4 ) = (x21 + x22 )2 + (x23 + x24 )2 ,
both KAN and MLP gradually pick up more inductive biases (their intermediate states are different)


                                                         12
Figure 9: Three tricks to facilitate symbolic regression. Trick A (top row): detecting and leveraging modular
structures. Trick B (middle row): sparse connection initialization. Trick C (bottom row): Hypothesis testing.

                                                                                         p
until they reach the correct structure. In the second case, f (x1 , x2 , x3 ) = sin(x1 )/ x22 + x23 , both
the models initially detect multiplicative separability for all three variables, showing even higher
symmetry than the correct structure. After training progresses, both models “realize” that: in order
to better fit data (loss becomes lower), such high symmetry structure can no longer be met and should
be relaxed to a less stringent structure. An additional observation is that KAN has an intermediate
structure not found in the MLP. There are two caveats we would like to mention: (1) results can
be seed and/or threshold-dependent. (2) all tests rely on second-order derivatives, which may not
be robust due to the model being trained only on zero-order information. Adversarial constructions
such as fϵ (x) = f (x)+ϵsin( xϵ ) could lead to issues, because although |fϵ (x)−f (x)| → 0 as ϵ → 0,
|fϵ′′ (x) − f ′′ (x)| → ∞ as ϵ → 0. Although such extreme cases are unlikely in practice, smoothness
is necessary to ensure the success of our methods.

4.3   Identifying symbolic formulas from KANs
Symbolic formulas are the most informative, as they clearly reveal both important features and
modular structures once they are known. In Liu et al. [57], the authors showed a bunch of examples
from which they can extract symbolic formulas, with some prior knowledge when needed. With
the new tools proposed above (feature importance, modular structures, and symbolic formulas),
users can leverage these new tools to easily interact and collaborate with KANs, making symbolic
regression easier. We present three tricks below, illustrated in Figure 9.
Trick A: discover and leverage modular structures We can first train a general network and
probe its modularity. Once the modular structure is identified, we initialize a new model with this
modular structure as inductive biases. For instance, consider the function f (q, v, B, m) = qvB/m.


                                                     13
We first initialize a large KAN (presumably expressive enough) to fit the dataset to a reasonable
accuracy. After training, the tree graph is extracted (ref Sec 4.2) from the trained KAN, which
shows multiplicative separability. Then we can build the modular structure into a second KAN (ref
Sec 3.2), train it, and then symbolify all 1D functions to derive the formula.
Trick B: Sparse initialization Symbolic formulas typically correspond to KANs with sparse con-
nections (see Figure 5 (b)), so initializing KANs sparsely aligns them better with the inductive biases
of symbolic formulas. Otherwise, densely initialized KANs require careful regularization to promote
sparsity. Sparse initialization can be achieved by passing the argument “sparse_init=True” to the
KAN initializer. For example, for the function f (q, E, v, B, θ) = q(E + vBsinθ), a sparsely initial-
ized KAN closely resembles the final trained KAN, requiring only minor adjustments in training. In
contrast, a dense initialization would involve extensive training to remove unnecessary edges.
Trick C: Hypothesis Testing When faced with multiple reasonable hypotheses, we can try all of
them (branching into “parallel universes”) to test which hypothesis is the most accurate and/or sim-
plest. To facilitate hypothesis testing, we build a checkpoint system that automatically saves model
versions whenever p   changes (e.g., training, pruning) are made. For example, consider the function
f (m0 , v, c) = m0 / 1 − (v/c)2 . We start from a randomly initialized KAN, which has   pversion 0.0.
After training, it evolves to version 0.1, where it activates on both β = v/c and γ = 1/ 1 − (v/c)2 .
Hypothesize that only β or γ might be needed. We first set the edge on γ to zero, and train the model,
obtaining a 6.5 × 10−4 test RMSE (version 0.2). To test the alternative hypothesis, we want to re-
vert back to the branching point (version 0.1) – we call model.rewind(‘0.1’) which rewinds the
model back to version 0.1. To indicate that rewind is called, version 0.1 is renamed to version 1.1.
Now we set the edge on β to be zero, train the model, obtaining a 2.0 × 10−6 test RMSE (the version
becomes 1.2). Comparing versions 0.2 and 1.2 indicates that the second hypothesis is better due to
the lower loss given the same complexity (both hypotheses have two non-zero edges).

5     Applications
The previous sections primarily focused on regression problems for pedagogical purposes. In this
section, we apply KANs to discover physical concepts, such as conserved quantities, Lagrangians,
hidden symmetries, and constitutive laws. These examples illustrate how the tools proposed in this
paper can be effectively integrated into real-life scientific research to tackle these complex tasks.

5.1   Discovering conserved quantities




          Figure 10: Using KANs to discover conserved quantities for the 2D harmonic oscillator.

Conserved quantities are physical quantities that remain constant over time. For example, a free-
falling ball converts its gravitational potential energy into kinetic energy, while the total energy (the
sum of both forms of energy) remains constant (assuming negligible air resistance). Conserved
quantities are crucial because they often correspond to symmetries in physical systems and can sim-
plify calculations by reducing the dimensionality of the system. Traditionally, deriving conserved
quantities with paper and pencil can be time-consuming and demands extensive domain knowl-
edge. Recently, machine learning techniques have been explored to discover conserved quanti-
ties [55, 53, 54, 58, 32, 89].


                                                   14
We follow the approach Liu et al. [53], which derived a differential equation that conserved quan-
tities must satisfy, thus transforming the problem of finding conserved quantities into differential
equation solving. They used multi-layer perceptrons (MLPs) to parameterize conserved quantities.
We basically follow their procedure but replace MLPs with KANs. To be specific, they consider a
dynamical system with the state variable z ∈ Rd governed by the equation dz
                                                                          dt = f (z). The necessary
and sufficient condition for a function H(z) to be a conserved quantity is that f (z) · ∇H(z) = 0
for all z. For example, in a 1D harmonic oscillator, the phase space is characterized by position
and momentum, z = (x, p), and the evolution equation is d(x, p)/dt = (p, −x). The energy
H = 21 (x2 + p2 ) is a conserved quantity because f (z) · ∇H(z) = (p, −x) · (x, p) = 0. We param-
                                                                  PN                         2
eterize H using a KAN, and train it with the loss function ℓ = i=1 f (z(i) ) · ∇H(z
                                                                                 b      (i)
                                                                                            ) where
∇b is the normalized gradient, and z(i) are the ith data point uniformly drawn from the hypercube
[−1, 1]d .
We choose the 2D harmonic oscillator to test KANs, characterized by (x, y, px , py ). It has three
conserved quantities: (1) energy along x direction: H1 = 21 (x2 + p2x ); (2) energy along y direction:
H2 = 21 (y 2 + p2y ); (3) angular momentum H3 = xpy − ypx . We train [4, [0, 2], 1] KANs with three
different random seeds, as shown in Figure 10, which correspond to H1 , H2 and H3 respectively.

5.2   Discovering Lagrangians




Figure 11: Use KANs to learn Lagrangians for the single pendulum (top) and a relativistic mass in a uniform
field (bottom).

In physics, Lagrangian mechanics is a formulation of classical mechanics based on the principle
of stationary action. It describes a mechanical system using phase space and a smooth function L
known as the Lagrangian. For many systems, L = T − V , where T and V represent the kinetic and
potential energy of the system, respectively. The phase space is typically described by (q, q̇), where
q and q̇ denotes coordinates and velocities, respectively. The equation of motion can be derived
                                                         d ∂L
from the Lagrangian via the Euler-Lagrange equation: dt   ( ∂ q̇ ) = ∂L
                                                                     ∂q , or equivalently

                                                                      ˙
                                q̈ = (∇q̇ ∇Tq̇ L)−1 [∇q L − (∇q ∇Tq̇ q)]                              (18)
Given the fundamental role of the Lagrangian, an interesting question is whether we can infer the La-
grangian from data. Following [19], we train a Lagrangian neural network to predict q̈ from (q, q̇).
An LNN uses an MLP to parameterize L(q, q̇), and computes the Eq. (18) to predict instant accel-
erations q̈. However, LNNs face two main challenges: (1) The training of LNNs can be unstable


                                                    15
due to the second-order derivatives and matrix inversion in Eq. (18). (2) LNNs lack interpretability
because MLPs themselves are not easily interpretable. We address these issues using KANs.
To tackle the first challenge, we note that the matrix inversion of the Hessian (∇q̇ ∇Tq̇ L)−1 be-
comes problematic when the Hessian has eigenvalues close to zero. To mitigate this, we initialize
(∇q̇ ∇Tq̇ L) as a positive definite matrix (or a positive number in 1D). Since (∇q̇ ∇Tq̇ L) is the mass
m in classical mechanics and kinetic energy is usually T = 21 mq̇2 , encoding this prior knowledge
into KANs is more straightforward than into MLPs (using the kanpiler introduced in Section 3.3).
The kanpiler can convert the symbolic formula T into a KAN (as shown in Figure 11). We use this
converted KAN for initialization and continue training, resulting in much greater stability compared
to random initialization. After training, symbolic regression can be applied to each edge to extract
out symbolic formulas, addressing the second challenge.
We show two 1D examples in Figure 11, a single pendulum and a relativistic mass in a uniform field.
The compiled KANs are displayed on the left, with edges on q̇ displaying quadratic functions and
edges on q as zero functions.
Single pendulum The q̇ part remains a quadratic function T (q̇) = 21 q̇ 2 while the q part learns to be
a cosine function, as V (q) = 1 − cos(q). In Figure 11 top, the results from suggest_symbolic
display the top five functions that best match the splines, considering both fitness and simplicity. As
expected, the cosine and the quadratic function appear at the top of the lists.
Relativistic mass in a uniform field After training, the kinetic energy part deviates from T =
1 2                                                      2 −1/2
2 q̇ because, for a relativistic particle, Tr = (1 − q̇ )       − 1. In Figure 11 (bottom), symbolic
regression successfully finds V (q) = q, but fails to identify Tr due to its compositional nature, as our
symbolic regression only searches for simple functions. By assuming the first function composition
is quadratic, we create another [1, 1, 1] KAN to fit Tr and set the first function to be the quadratic
function using fix_symbolic, and train only the second learnable function. After training, we see
that the ground truth x−1/2 appears among the top five candidates. However, x1/2 fits the spline
slightly better, as indicated by a higher R-squared value. This suggests that symbolic regression is
sensitive to noise (due to imperfect learning) and prior knowledge is crucial for correct judgment.
For instance, knowing that kinetic energy should diverge as velocity approaches the speed of light
helps confirm x−1/2 as the correct term, since x1/2 does not exhibit the expected divergence.
5.3   Discovering hidden symmetry




Figure 12: Rediscovering the hidden symmetry of the Schwarzschild black hole with MLPs and KANs. (a)
∆t(r) learned by the MLP is a globally smooth solution; (b) ∆t(r) learned by the KAN is a domain-wall
solution; (c) The KAN shows a loss spike at the domain wall; (d) A KAN can be used to fine-tune the MLP
solution close to machine precision.

Philip Anderson famously argued that “it is only slightly overstating that case to say that physics is
the study of symmetry”, emphasizing how the discovery of symmetries has been invaluable for both
deepening our understanding and solving problems more efficiently.
However, symmetries are sometimes not manifest but hidden, only revealed by applying some co-
ordinate transformation. For example, after Schwarzschild discovered his eponymous black hole
metric, it took 17 years for Painlevé, Gullstrand and Lemaître to uncover its hidden translational
symmetry. They demonstrated that the spatial sections could be made translationally invariant with
a clever coordinate transformation, thereby deepening our understanding of black holes [65]. Liu &
Tegmark [56] showed that the Gullstrand-Painlevé transformation can be discovered by training an
MLP in minutes. However, they did not get extremely high precision (i.e., machine precision) for
the solution. We attempt to revisit this problem using KANs.


                                                   16
Suppose there is a Schwarzschild black hole in spacetime (t, x, y, z) with mass 2M = 1, centered
at x = y = z = 0 with a radius rs = 2M = 1. The Schwarzschild metric describes how space and
time distorts around it:
                       1 − 2M              0                   0                  0
                                                                                              
                              r                  2
                                            2M x              2M xy             2M xz
                      0           −1 − (r−2M    )r 2     − (r−2M  )r 2     − (r−2M   )r 2
                                                                                               
              gµν =                                                                                    (19)
                                                                     2
                                                                                               
                                         2M xy                  2M y            2M yz
                      0             − (r−2M )r2        −1 − (r−2M )r2      − (r−2M )r2 
                                                                                               
                                         2M xz                2M yz                2M z 2
                           0         − (r−2M  )r 2        − (r−2M  )r 2   −1 − (r−2M     )r 2 .

Applying the Gullstrand-Painlevé transformation t′ = t + 2M (2u + ln( u−1                             ′
                                                                                             p r
                                                                              u+1 )), u ≡       2M , x = x,
y ′ = y, z ′ = z, the metric in the new coordinates becomes:
                                              q              q           q      
                                                                2M y
                                 1 − 2M      −      2M x
                                                           −            −   2M z
                               q r                   r r         r r         r r
                                     2M x
                              −                  −1            0           0
                                                                                
                       ′              r r
                                                                                 
                      gµν =     q                                              ,                     (20)
                                     2M  y
                              −                   0           −1           0
                                                                                 
                                      r  r
                                                                                 
                               q                                                
                                 − 2M r r
                                         z
                                                   0            0          −1

which exhibits translation invariance in the spatial section (the lower right 3 × 3 block is the
Euclidean metric). Liu & Tegmark [56] used an MLP to learn the mapping from (t, x, y, z) to
                                                            ′ ′ ′ ′
(t′ , x′ , y ′ , z ′ ). Defining the Jacobian matrix J ≡ ∂(t ,x ,y ,z )                     ′
                                                          ∂(t,x,y,z) , g is tranformed to g = J
                                                                                                  −T
                                                                                                      gJ−1 .
We take the bottom right 3 × 3 block of g′ and take its difference to the Euclidean metric to obtain
the MSE loss. The loss is minimized by doing gradient descents on the MLP. To make things simple,
they assume knowing x′ = x, y ′ = y, z ′ = z, and only use an MLP (1         pinput and 1 output) to predict
the temporal difference ∆t(r) = t′ − t = 2M (2u + ln( u−1       u+1  )), u ≡    r
                                                                               2M  from the radius r.
MLP and KAN find different solutions We trained both an MLP and a KAN to minimize this
loss function, with results shown in Figure 12. Since the task has 1 input dimension and 1 output
dimension, the KAN effectively reduces to a spline. We originally expected KANs to outperform
MLPs, because splines are known to be superior in low-dimensional settings [63]. However, while
MLP can achieve 10−8 loss, the KAN gets stuck at 10−3 loss despite grid refinements. It turned
out that KAN and MLP learned two different solutions: while the MLP found a globally smooth
solution (Figure 12 (a)), the KAN learned a domain-wall solution (Figure 12 (b)). The domain wall
solution has a singular point that separates the whole curve into two segments. The left segment
learns ∆t(r) correctly, while the right segment learns −∆t(r), which is also a valid solution but
differs from the left segment by a minus sign. There is a loss spike appearing at the singular point
(Figure 12 (c)). One might consider this as a feature of KANs because domain wall solutions are
prevalent in nature. However, if one considers this a flaw, KANs can still obtain globally smooth
solutions by adding regularizations (to reduce spline oscillations) or experimenting with different
random seeds (roughly 1 out of 3 random seeds finds a global smooth solution).
KANs can achieve extreme precision Although the MLP finds the globally smooth solution and
achieves 10−8 loss, the loss is still far from machine precision. We found that neither longer training
nor increasing the MLP’s size significantly reduced the loss. Therefore, we turned to KANs, which,
as splines in 1D, can achieve arbitrary accuracy by refining the grid (given infinite data). We first
used the MLP as a teacher, generating supervised pairs (x, y) to train the KAN to fit the supervised
data. This way, the KAN is initialized to a globally smooth solution. We then iteratively refined the
KAN by increasing the number of grid intervals to 1000. In the end, the fine-tuned KANs achieve a
loss of 10−15 , close to machine precision (Figure 12 (d)).
5.4 Learning constitutive laws
A constitutive law defines the behavior and properties of a material by modeling how it responds to
external forces or deformations. One of the simplest forms of constitutive law is Hooke’s Law [34],
which relates the strain and stress of elastic materials linearly. Constitutive laws encompass a wide
range of materials, including elastic materials [80, 68], plastic materials [64], and fluids [8]. Tra-
ditionally, these laws were derived from first principles based on theoretical and experimental stud-
ies [79, 81, 6, 29]. Recent advancements, however, have introduced data-driven approaches that
leverage machine learning to discover and refine these laws from dedicated datasets [73, 91, 59, 60].


                                                    17
Figure 13: Discovering constitutive laws (relations between the pressure tensor P and the strain tensor F ) with
KANs by interacting with them. Top: predicting the diagonal element P11 ; bottom: predicting the off-diagonal
element P12 .




                                                      18
We follow the standard notations and experimental setups in the elasticity part of NCLaw [59] and
define the constitutive law as a parameterized function Eθ (F) → P, where F denotes the defor-
mation tensor, P the first Piola–Kirchhoff stress tensor, and θ the parameters in the constitutive
law.
Many isotropic materials have linear constitutive laws when deformation is small:
                              Pl = µ(F + FT − 2I) + λ(Tr(F) − 3)I.                                  (21)
However, when deformation gets larger, nonlinear effects start to kick in. For example, a Neo-
Hookean material has the following constitutive law:
                                 P = µ(FFT − I) + λlog(det(F))I,                                    (22)
where µ and λ are the so-called Lamé parameters determined by the so-called Young’s modulus Y
                              Y             Yν
and Poisson ratio ν as µ = 2(1+ν) , λ = (1+ν)(1−2ν) . For simplicity, we choose Y = 1 and ν = 0.2,
           5                    5
hence µ = 12 ≈ 0.42 and λ = 18 ≈ 0.28.
Assuming we are working with Neo-Hookean materials, and our goal is to use KANs to predict the
P tensor from the F tensor. Suppose we do not know they are neo-Hookean materials, but we have
the prior knowledge that the linear constitutive law is approximately valid for small deformation.
Due to symmetries, it suffices to demonstrate that we can accurately predict P11 and P12 from
the 9 matrix elements of F. We want to compile linear constitutive laws into KANs, which are
P11 = 2µ(F11 − 1) + λ(F11 + F22 + F33 − 3), and P12 = µ(F12 + F21 ). We want to extract
                                                                 2     2      2
Neo-Hookean laws from trained KANs, which are P11 = µ(F11           + F12 + F13 − 1) + λlog(det(F)),
and P12 = µ(F11 F21 + F12 F22 + F13 F23 ). We generate a synthetic dataset by sampling Fij
independently from U [δij − w, δij + w](w = 0.2) and using the Neo-Hookean constitutive law to
compute P. Our interaction with KANs is illustrated in Figure 13. In both cases, we successfully
figured out the true symbolic formulas in the end, with the aid of some inductive biases. However, the
key takeaway is not that we can rediscover the exact symbolic formulas – given that prior knowledge
skews the process – but rather in real-world scenarios, where the answers are unknown and users
can make guesses based on prior knowledge, the pykan package makes it easy to test or incorporate
prior knowledge.
Predicting P11 In step 1, we compile the linear constitutive law P11 = 2µ(F11 − 1) + λ(F11 +
F22 + F33 − 3) to a KAN using kanpiler, resulting in a 10−2 loss. In step 2, we perturb the KAN
so that it becomes trainable (indicated by the color change from red to purple; red denotes a purely
symbolic part, while purple indicates that both symbolic and spline parts are active). In step 3, we
train the perturbed model until convergence, giving a 6 × 10−3 loss. In step 4, assuming that the
determinant is a key auxiliary variable, we use expand_width (for the KAN) and augment_input
(for the dataset) to include the determinant |F |. In step 5, we train the KAN until convergence,
giving a 2 × 10−4 loss. In step 6, we symbolify the KAN to obtain a symbolic formula P11 =
0.42(F112      2
           + F12 + F132
                        − 1) + 0.28log(|F |), which achieves a 3 × 10−11 loss.
Predicting P12 We experimented with and without encoding the linear constitutive law as prior
knowledge. With prior knowledge: in step 1, we compile the linear constitutive law to a KAN,
resulting in a loss of 10−2 . We then perform a series of operations, including expand (step 2), perturb
(step 3), train (step 4), prune (step 5) and finally symbolic (step 6). The influence of prior knowledge
is evident, as the final KAN only identifies minor correction terms to the linear constitutive law. The
                                                                                2         2
final KAN is symbolified as P12 = 0.42(F12 +F21 )+0.44F13 F23 −0.03F21            +0.02F12   which yields
a 7 × 10−3 loss, only slightly better than the linear constitutive law. Without prior knowledge: in
step 1, we randomly initialize the KAN model. In step 2, we train the KAN with regularization. In
step 3, we prune the KAN to be a more compact model. In step 4, we symbolify the KAN, yielding
P12 = 0.42(F11 F21 + F12 F22 + F13 F23 ), which closely matches the exact formula, achieving a
6 × 10−9 loss. Comparing the two scenarios – one with and one without prior knowledge – reveals a
surprising outcome: in this example, prior knowledge appears harmful, possibly because the linear
constitutive law is probably near a (bad) local minimum which is hard for the model to escape.
However, we should probably not randomly extrapolate this conclusion to more complicated tasks
and larger networks. For more complicated tasks, finding a local minimum via gradient descent
might be challenging enough, making an approximate initial solution desirable. Additionally, larger
networks might be sufficiently over-parameterized to eliminate bad local minima, ensuring that all
local minima are global and interconnected.


                                                   19
Figure 14: KAN interpolates between software 1.0 and 2.0. (a) KANs strike a balance between interpretability
(software 1.0) and learnability (software 2.0). (b) KANs’ Pareto frontier on the interpretability-scale plane. The
amount of interpretation we can get from KANs depends on problem scales and interpretability methods.


6    Related works
Kolmogorov-Arnold Networks (KANs), inspired by the Kolmogorov-Arnold representation the-
orem (KART), were recently proposed by Liu et al. [57]. Although the connection between
KART and networks has long been deemed irrelevant [30], Liu et al. generalized the origi-
nal two-layer network to arbitrary depths and demonstrated their promise for science-oriented
tasks given their accuracy and interpretability. Subsequent research has explored the application
of KANs across various domains, including graphs [12, 22, 38, 99], partial differential equa-
tions [87, 78] and operator learning [1, 78, 67], tabular data [70], time series [85, 28, 93, 27],
human activity recognition [49, 50],neuroscience [96, 33], quantum science [40, 46, 4], computer
vision [17, 7, 44, 16, 76, 10], kernel learning [101], nuclear physics [48], electrical engineering [69],
biology [71]. Liu et al. used B-splines to parameterize 1D functions, and other research have ex-
plored various activation functions, including wavelet [11, 76], radial basis function [47], Fourier
series [92]), finite basis [35, 82], Jacobi basis functions [2], polynomial basis functions [75], ratio-
nal functions [3]. Other techniques for KANs have also been proposed including regularization [5],
Kansformer (combining transformer and KAN) [15], adaptive grid update [72], federated learn-
ing [98] , Convolutional KANs [10]. There have been ongoing debates regarding whether KANs
really outperform other neural networks (especially MLPs) on various domains [7, 16, 42, 77, 97],
which suggests that while KANs show promise for machine learning tasks, further development is
needed to surpass state-of-the-art models.
Machine Learning for Physical Laws A major goal for KANs is to aid in the discovery of
new physical laws from data. Previous research has shown that machine learning can be used to
learn various types of physical laws, including equations of motion [90, 13, 43, 20], conservation
laws [55, 53, 54, 58, 32, 89], symmetries [39, 56, 94], phase transitions [88, 14], Lagrangian and
Hamiltonian [19, 31], and symbolic regression [18, 61, 23, 74], etc. However, making neural net-
works interpretable often requires domain-specific knowledge, limiting their generality. We hope
that KANs will evolve into universal foundation models for physical discoveries.
Mechanistic Interpretability seeks to understand how neural networks operate in a fundamental
level [21, 62, 86, 25, 66, 100, 51, 24, 45, 26]. Some research in this area focuses on designing
models that are inherently interpretable [24] or proposing training methods that explicitly promote
interpretability [51]. KANs fall into this category since the Kolmogorov-Arnold theorem decom-
poses a high-dimensional function into a collection of 1D functions, which are significantly easier
to interpret than high-dimensional functions.

7    Discussion
KAN interpolates between software 1.0 and 2.0 The key difference between Kolmogorov-Arnold
Networks (KANs) and other neural networks (software 2.0, a term coined by Andrej Karpathy) lies
in their greater interpretability, which allows for manipulation by users, similar to traditional soft-
ware (software 1.0). However, KANs are not entirely traditional software, as they (1) learnability
(good), enabling them to learn new things from data, and (2) reduced interpretability (bad) as they


                                                       20
become less interpretable and controllable as the network scales increase. In Figure 14 (a) visualizes
the position of software 1.0, software 2.0, and KANs on the interpretability-learnability plane, illus-
trating how KANs can balance the trade-offs between these two paradigms. The goal of this paper
is to propose various tools that make KANs more like software 1.0, while leveraging the learnability
of software 2.0.
Efficiency improvement The original pykan package [57] was poor in efficiency. We have incor-
porated a few techniques to improve its efficiency.
        1. Efficient splines evaluations. Inspired by Efficient KAN [9], we have optimized spline eval-
           uations by avoiding unnecessary input expansions. For a KAN with L layers, N neurons
           per layer, and grid size G, memory usage has been reduced from O(LN 2 G) to O(LN G).
        2. Enabling the symbolic branch only when needed. A KAN layer contains both a spline
           branch and a symbolic branch. The symbolic branch is much more time-consuming than the
           spline branch since it cannot be parallelized (disastrous double loops are needed). However,
           in many applications, the symbolic branch is unnecessary, so we can skip it when possible,
           significantly reducing runtime, especially when the network is large.
        3. Saving intermediate activations only when needed. To plot KAN diagrams, intermediate
           activations must be saved. Initially, activations were saved by default, leading to slower
           runtime and excessive memory usage. We now save intermediate activations only when
           needed (e.g., for plotting or applying regularizations in training). Users can enable these
           efficiency improvements with a single line: model.speed().
        4. GPU acceleration. Initially, all models were run on CPUs due to the small-scale nature of
           the problems. We have now made the model GPU-compatible 6 . For example, training a
           [4,100,100,100,1] with Adam for 100 steps used to take an entire day on a CPU (before
           implementing 1, 2, 3), but now takes 20 seconds on a CPU and less than one second on a
           GPU. However, KANs still lag behind MLPs in efficiency, especially at large scales. The
           community has been working towards benchmarking and improving KAN’s efficiency and
           the efficiency gap has been significantly reduced [36].
Since the objective of this paper is to make KANs more like software 1.0, when facing trade-offs
between 1.0 (being interactive and versatile) and 2.0 (being efficient and specific), we prioritize
interactivity and versatility over efficiency. For example, we store cached data within models (which
consumes additional memory), so users can simply call model.plot() to generate a KAN diagram
without manually doing a forward pass to collect data.
Interpretability Although the learnable univariate functions in KANs are more interpretable than
weight matrices in MLPs, scalability remains a challenge. As KAN models scale up, even if all
spline functions are interpretable individually, it becomes increasingly difficult to manage the com-
bined output of these 1D functions. Consequently, a KAN may only remain interpretable when
the network scale is relatively small (Figure 14 (b), thick red line). It is important to note that
interpretability depends on both intrinsic factors (related to the model itself) and extrinsic factors
(related to interpretability methods). Advanced interpretability methods should be able to handle
interpretability at various levels. For example, by interpreting KANs with symbolic regression,
modularity discovery and feature attribution (Figure 14 (b), thin red lines), the Pareto Frontier of
interpretability versus scale extends beyond what a KAN alone can achieve. A promising direction
for future research is to develop more advanced interpretability methods that can further push the
current Pareto Frontiers.
Future work This paper introduces a framework that integrates KANs with scientific knowledge,
focusing primarily on small-scale, physics-related examples. Moving forward, two promising direc-
tions include applying this framework to larger-scale problems and extending it to other scientific
disciplines beyond physics.

Acknowledgement
We would like to thank Yizhou Liu, Di Luo, Akash Kundu and many GitHub users for fruitful
discussion and constructive suggestions. We extend special thanks to GitHub user Blealtan for
   6
       Models can be trained on GPUs, but not all functionalities already supported GPU.


                                                       21
making public their awesome work on making KANs efficient. Z.L. and M.T. are supported by
IAIFI through NSF grant PHY-2019786.

References
  [1] D. W. Abueidda, P. Pantidis, and M. E. Mobasher. Deepokan: Deep operator network based
      on kolmogorov arnold networks for mechanics problems. arXiv preprint arXiv:2405.19143,
      2024.
  [2] A. A. Aghaei. fkan: Fractional kolmogorov-arnold networks with trainable jacobi basis func-
      tions. arXiv preprint arXiv:2406.07456, 2024.
  [3] A. A. Aghaei. rkan: Rational kolmogorov-arnold networks. arXiv preprint arXiv:2406.14495,
      2024.
  [4] T. Ahmed and M. H. R. Sifat. Graphkan: Graph kolmogorov arnold network for small
      molecule-protein interaction predictions. In ICML’24 Workshop ML for Life and Material
      Science: From Theory to Industry Applications, 2024.
  [5] M. G. Altarabichi. Dropkan: Regularizing kans by masking post-activations. arXiv preprint
      arXiv:2407.13044, 2024.
  [6] E. M. Arruda and M. C. Boyce. A three-dimensional constitutive model for the large
      stretch behavior of rubber elastic materials. Journal of the Mechanics and Physics of Solids,
      41(2):389–412, 1993.
  [7] B. Azam and N. Akhtar. Suitability of kans for computer vision: A preliminary investigation.
      arXiv preprint arXiv:2406.09087, 2024.
  [8] G. K. Batchelor. An introduction to fluid dynamics. Cambridge university press, 2000.
  [9] Blealtan. Blealtan/efficient-kan: An efficient pure-pytorch implementation of kolmogorov-
      arnold network (kan).
 [10] A. D. Bodner, A. S. Tepsich, J. N. Spolski, and S. Pourteau. Convolutional kolmogorov-
      arnold networks. arXiv preprint arXiv:2406.13155, 2024.
 [11] Z. Bozorgasl and H. Chen. Wav-kan: Wavelet kolmogorov-arnold networks. arXiv preprint
      arXiv:2405.12832, 2024.
 [12] R. Bresson, G. Nikolentzos, G. Panagopoulos, M. Chatzianastasis, J. Pang, and M. Vazir-
      giannis. Kagnns: Kolmogorov-arnold networks meet graph learning. arXiv preprint
      arXiv:2406.18380, 2024.
 [13] S. L. Brunton, J. L. Proctor, and J. N. Kutz. Discovering governing equations from data by
      sparse identification of nonlinear dynamical systems. Proceedings of the national academy
      of sciences, 113(15):3932–3937, 2016.
 [14] J. Carrasquilla and R. G. Melko. Machine learning phases of matter. Nature Physics,
      13(5):431–434, 2017.
 [15] Y. Chen, Z. Zhu, S. Zhu, L. Qiu, B. Zou, F. Jia, Y. Zhu, C. Zhang, Z. Fang, F. Qin, et al.
      Sckansformer: Fine-grained classification of bone marrow cells via kansformer backbone
      and hierarchical attention mechanisms. arXiv preprint arXiv:2406.09931, 2024.
 [16] M. Cheon. Demonstrating the efficacy of kolmogorov-arnold networks in vision tasks. arXiv
      preprint arXiv:2406.14916, 2024.
 [17] M. Cheon. Kolmogorov-arnold network for satellite image classification in remote sensing.
      arXiv preprint arXiv:2406.00600, 2024.
 [18] M. Cranmer. Interpretable machine learning for science with pysr and symbolicregression. jl.
      arXiv preprint arXiv:2305.01582, 2023.
 [19] M. Cranmer, S. Greydanus, S. Hoyer, P. Battaglia, D. Spergel, and S. Ho. Lagrangian neural
      networks. arXiv preprint arXiv:2003.04630, 2020.


                                               22
[20] M. Cranmer, A. Sanchez Gonzalez, P. Battaglia, R. Xu, K. Cranmer, D. Spergel, and S. Ho.
     Discovering symbolic models from deep learning with inductive biases. Advances in neural
     information processing systems, 33:17429–17442, 2020.
[21] H. Cunningham, A. Ewart, L. Riggs, R. Huben, and L. Sharkey. Sparse autoencoders find
     highly interpretable features in language models. arXiv preprint arXiv:2309.08600, 2023.
[22] G. De Carlo, A. Mastropietro, and A. Anagnostopoulos. Kolmogorov-arnold graph neural
     networks. arXiv preprint arXiv:2406.18354, 2024.
[23] O. Dugan, R. Dangovski, A. Costa, S. Kim, P. Goyal, J. Jacobson, and M. Soljačić. Occamnet:
     A fast neural model for symbolic regression at scale. arXiv preprint arXiv:2007.10784, 2020.
[24] N. Elhage, T. Hume, C. Olsson, N. Nanda, T. Henighan, S. Johnston, S. ElShowk, N. Joseph,
     N. DasSarma, B. Mann, D. Hernandez, A. Askell, K. Ndousse, A. Jones, D. Drain, A. Chen,
     Y. Bai, D. Ganguli, L. Lovitt, Z. Hatfield-Dodds, J. Kernion, T. Conerly, S. Kravec, S. Fort,
     S. Kadavath, J. Jacobson, E. Tran-Johnson, J. Kaplan, J. Clark, T. Brown, S. McCan-
     dlish, D. Amodei, and C. Olah. Softmax linear units. Transformer Circuits Thread, 2022.
     https://transformer-circuits.pub/2022/solu/index.html.
[25] N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, Z. Hatfield-Dodds,
     R. Lasenby, D. Drain, C. Chen, et al. Toy models of superposition. arXiv preprint
     arXiv:2209.10652, 2022.
[26] J. Engels, I. Liao, E. J. Michaud, W. Gurnee, and M. Tegmark. Not all language model
     features are linear. arXiv preprint arXiv:2405.14860, 2024.
[27] R. Genet and H. Inzirillo. A temporal kolmogorov-arnold transformer for time series fore-
     casting. arXiv preprint arXiv:2406.02486, 2024.
[28] R. Genet and H. Inzirillo. Tkan: Temporal kolmogorov-arnold networks. arXiv preprint
     arXiv:2405.07344, 2024.
[29] A. N. Gent. A new constitutive relation for rubber. Rubber chemistry and technology,
     69(1):59–61, 1996.
[30] F. Girosi and T. Poggio. Representation properties of networks: Kolmogorov’s theorem is
     irrelevant. Neural Computation, 1(4):465–469, 1989.
[31] S. Greydanus, M. Dzamba, and J. Yosinski. Hamiltonian neural networks. Advances in neural
     information processing systems, 32, 2019.
[32] S. Ha and H. Jeong. Discovering conservation laws from trajectories via machine learning.
     arXiv preprint arXiv:2102.04008, 2021.
[33] L. F. Herbozo Contreras, J. Cui, L. Yu, Z. Huang, A. Nikpour, and O. Kavehei. Kan-eeg:
     Towards replacing backbone-mlp for an effective seizure detection system. medRxiv, pages
     2024–06, 2024.
[34] R. Hooke. Lectures de potentia restitutiva, or of spring explaining the power of springing
     bodies. Number 6. John Martyn, 2016.
[35] A. A. Howard, B. Jacob, S. H. Murphy, A. Heinlein, and P. Stinis. Finite basis kolmogorov-
     arnold networks: domain decomposition for data-driven and physics-informed problems.
     arXiv preprint arXiv:2406.19662, 2024.
[36] Jerry-Master. Jerry-master/kan-benchmarking.
[37] J. Jumper, R. Evans, A. Pritzel, T. Green, M. Figurnov, O. Ronneberger, K. Tunyasuvunakool,
     R. Bates, A. Žídek, A. Potapenko, et al. Highly accurate protein structure prediction with
     alphafold. nature, 596(7873):583–589, 2021.
[38] M. Kiamari, M. Kiamari, and B. Krishnamachari. Gkan: Graph kolmogorov-arnold networks.
     arXiv preprint arXiv:2406.06470, 2024.
[39] S. Krippendorf and M. Syvaeri. Detecting symmetries with neural networks. Machine Learn-
     ing: Science and Technology, 2(1):015010, 2020.


                                               23
[40] A. Kundu, A. Sarkar, and A. Sadhu. Kanqas: Kolmogorov arnold network for quantum
     architecture search. arXiv preprint arXiv:2406.17630, 2024.
[41] R. Lam, A. Sanchez-Gonzalez, M. Willson, P. Wirnsberger, M. Fortunato, F. Alet, S. Ravuri,
     T. Ewalds, Z. Eaton-Rosen, W. Hu, et al. Learning skillful medium-range global weather
     forecasting. Science, 382(6677):1416–1421, 2023.
[42] T. X. H. Le, T. D. Tran, H. L. Pham, V. T. D. Le, T. H. Vu, V. T. Nguyen, Y. Nakashima,
     et al. Exploring the limitations of kolmogorov-arnold networks in classification: Insights to
     software training and hardware implementation. arXiv preprint arXiv:2407.17790, 2024.
[43] P. Lemos, N. Jeffrey, M. Cranmer, S. Ho, and P. Battaglia. Rediscovering orbital mechanics
     with machine learning. Machine Learning: Science and Technology, 4(4):045002, 2023.
[44] C. Li, X. Liu, W. Li, C. Wang, H. Liu, and Y. Yuan. U-kan makes strong backbone for medical
     image segmentation and generation. arXiv preprint arXiv:2406.02918, 2024.
[45] K. Li, A. K. Hopkins, D. Bau, F. Viégas, H. Pfister, and M. Wattenberg. Emergent world
     representations: Exploring a sequence model trained on a synthetic task. In The Eleventh
     International Conference on Learning Representations, 2023.
[46] X. Li, Z. Feng, Y. Chen, W. Dai, Z. He, Y. Zhou, and S. Jiao. Coeff-kans: A paradigm to
     address the electrolyte field with kans. arXiv preprint arXiv:2407.20265, 2024.
[47] Z. Li. Kolmogorov-arnold networks are radial basis function networks. arXiv preprint
     arXiv:2405.06721, 2024.
[48] H. Liu, J. Lei, and Z. Ren. From complexity to clarity: Kolmogorov-arnold networks in
     nuclear binding energy prediction, 2024.
[49] M. Liu, S. Bian, B. Zhou, and P. Lukowicz. ikan: Global incremental learning with kan for
     human activity recognition across heterogeneous datasets. arXiv preprint arXiv:2406.01646,
     2024.
[50] M. Liu, D. Geißler, D. Nshimyimana, S. Bian, B. Zhou, and P. Lukowicz. Initial investigation
     of kolmogorov-arnold networks (kans) as feature extractors for imu based human activity
     recognition. arXiv preprint arXiv:2406.11914, 2024.
[51] Z. Liu, E. Gan, and M. Tegmark. Seeing is believing: Brain-inspired modular training for
     mechanistic interpretability. Entropy, 26(1):41, 2023.
[52] Z. Liu, M. Khona, I. R. Fiete, and M. Tegmark. Growing brains: Co-emergence of anatomical
     and functional modularity in recurrent neural networks. arXiv preprint arXiv:2310.07711,
     2023.
[53] Z. Liu, V. Madhavan, and M. Tegmark. Machine learning conservation laws from differential
     equations. Physical Review E, 106(4):045307, 2022.
[54] Z. Liu, P. O. Sturm, S. Bharadwaj, S. J. Silva, and M. Tegmark. Interpretable conservation
     laws as sparse invariants. Phys. Rev. E, 109:L023301, Feb 2024.
[55] Z. Liu and M. Tegmark. Machine learning conservation laws from trajectories. Phys. Rev.
     Lett., 126:180604, May 2021.
[56] Z. Liu and M. Tegmark. Machine learning hidden symmetries. Physical Review Letters,
     128(18):180201, 2022.
[57] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljačić, T. Y. Hou, and M. Tegmark.
     Kan: Kolmogorov-arnold networks. arXiv preprint arXiv:2404.19756, 2024.
[58] P. Y. Lu, R. Dangovski, and M. Soljačić. Discovering conservation laws using optimal trans-
     port and manifold learning. Nature Communications, 14(1):4744, 2023.
[59] P. Ma, P. Y. Chen, B. Deng, J. B. Tenenbaum, T. Du, C. Gan, and W. Matusik. Learning neural
     constitutive laws from motion observations for generalizable pde dynamics. In International
     Conference on Machine Learning, pages 23279–23300. PMLR, 2023.


                                               24
[60] P. Ma, T.-H. Wang, M. Guo, Z. Sun, J. B. Tenenbaum, D. Rus, C. Gan, and W. Matusik.
     Llm and simulation as bilevel optimizers: A new paradigm to advance physical scientific
     discovery. In Forty-first International Conference on Machine Learning, 2024.
[61] G. Martius and C. H. Lampert.       Extrapolation and learning equations.    arXiv preprint
     arXiv:1610.02995, 2016.
[62] K. Meng, D. Bau, A. J. Andonian, and Y. Belinkov. Locating and editing factual associations
     in GPT. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, Advances in Neural
     Information Processing Systems, 2022.
[63] E. J. Michaud, Z. Liu, and M. Tegmark. Precision machine learning. Entropy, 25(1):175,
     2023.
[64] R. v. Mises. Mechanik der festen körper im plastisch-deformablen zustand. Nachrichten
     von der Gesellschaft der Wissenschaften zu Göttingen, Mathematisch-Physikalische Klasse,
     1913:582–592, 1913.
[65] C. Misner, K. Thorne, and J. Wheeler. Gravitation. Princeton University Press, 2017.
[66] N. Nanda, L. Chan, T. Lieberum, J. Smith, and J. Steinhardt. Progress measures for grokking
     via mechanistic interpretability. In The Eleventh International Conference on Learning Rep-
     resentations, 2023.
[67] G. Nehma and M. Tiwari. Leveraging kans for enhanced deep koopman operator discovery.
     arXiv preprint arXiv:2406.02875, 2024.
[68] R. W. Ogden. Non-linear elastic deformations. Courier Corporation, 1997.
[69] Y. Peng, M. He, F. Hu, Z. Mao, X. Huang, and J. Ding. Predictive modeling of flexible ehd
     pumps using kolmogorov-arnold networks. arXiv preprint arXiv:2405.07488, 2024.
[70] E. Poeta, F. Giobergia, E. Pastor, T. Cerquitelli, and E. Baralis. A benchmarking study of
     kolmogorov-arnold networks on tabular data. arXiv preprint arXiv:2406.14529, 2024.
[71] P. Pratyush, C. Carrier, S. Pokharel, H. D. Ismail, M. Chaudhari, and D. B. KC. Calmphoskan:
     Prediction of general phosphorylation sites in proteins via fusion of codon aware embeddings
     with amino acid aware embeddings and wavelet-based kolmogorov arnold network. bioRxiv,
     pages 2024–07, 2024.
[72] S. Rigas, M. Papachristou, T. Papadopoulos, F. Anagnostopoulos, and G. Alexandridis.
     Adaptive training of grid-dependent physics-informed kolmogorov-arnold networks. arXiv
     preprint arXiv:2407.17611, 2024.
[73] A. Sanchez-Gonzalez, J. Godwin, T. Pfaff, R. Ying, J. Leskovec, and P. Battaglia. Learning
     to simulate complex physics with graph networks. In International conference on machine
     learning, pages 8459–8468. PMLR, 2020.
[74] M. Schmidt and H. Lipson. Distilling free-form natural laws from experimental data. science,
     324(5923):81–85, 2009.
[75] S. T. Seydi. Exploring the potential of polynomial basis functions in kolmogorov-arnold
     networks: A comparative study of different groups of polynomials. arXiv preprint
     arXiv:2406.02583, 2024.
[76] S. T. Seydi. Unveiling the power of wavelets: A wavelet-based kolmogorov-arnold network
     for hyperspectral image classification. arXiv preprint arXiv:2406.07869, 2024.
[77] H. Shen, C. Zeng, J. Wang, and Q. Wang. Reduced effectiveness of kolmogorov-arnold
     networks on functions with noise. arXiv preprint arXiv:2407.14882, 2024.
[78] K. Shukla, J. D. Toscano, Z. Wang, Z. Zou, and G. E. Karniadakis. A comprehensive and
     fair comparison between mlp and kan representations for differential equations and operator
     networks. arXiv preprint arXiv:2406.02917, 2024.
[79] E. Sifakis and J. Barbic. Fem simulation of 3d deformable solids: a practitioner’s guide to
     theory, discretization and model reduction. In Acm siggraph 2012 courses, pages 1–50. 2012.


                                              25
[80] W. S. Slaughter. The linearized theory of elasticity. Springer Science & Business Media,
     2012.
[81] B. Smith, F. D. Goes, and T. Kim. Stable neo-hookean flesh simulation. ACM Transactions
     on Graphics (TOG), 37(2):1–15, 2018.
[82] H.-T. Ta. Bsrbf-kan: A combination of b-splines and radial basic functions in kolmogorov-
     arnold networks. arXiv preprint arXiv:2406.11173, 2024.
[83] T. H. Trinh, Y. Wu, Q. V. Le, H. He, and T. Luong. Solving olympiad geometry without
     human demonstrations. Nature, 625(7995):476–482, 2024.
[84] S.-M. Udrescu, A. Tan, J. Feng, O. Neto, T. Wu, and M. Tegmark. Ai feynman 2.0: Pareto-
     optimal symbolic regression exploiting graph modularity. Advances in Neural Information
     Processing Systems, 33:4860–4871, 2020.
[85] C. J. Vaca-Rubio, L. Blanco, R. Pereira, and M. Caus. Kolmogorov-arnold networks (kans)
     for time series analysis. arXiv preprint arXiv:2405.08790, 2024.
[86] K. R. Wang, A. Variengien, A. Conmy, B. Shlegeris, and J. Steinhardt. Interpretability in the
     wild: a circuit for indirect object identification in GPT-2 small. In The Eleventh International
     Conference on Learning Representations, 2023.
[87] Y. Wang, J. Sun, J. Bai, C. Anitescu, M. S. Eshaghi, X. Zhuang, T. Rabczuk, and Y. Liu.
     Kolmogorov arnold informed neural network: A physics-informed deep learning framework
     for solving pdes based on kolmogorov arnold networks. arXiv preprint arXiv:2406.11045,
     2024.
[88] S. J. Wetzel. Unsupervised learning of phase transitions: From principal component analysis
     to variational autoencoders. Physical Review E, 96(2):022140, 2017.
[89] S. J. Wetzel, R. G. Melko, J. Scott, M. Panju, and V. Ganesh. Discovering symmetry invariants
     and conserved quantities by interpreting siamese neural networks. Phys. Rev. Res., 2:033499,
     Sep 2020.
[90] T. Wu and M. Tegmark. Toward an artificial intelligence physicist for unsupervised learning.
     Physical Review E, 100(3):033311, 2019.
[91] H. Xu, F. Sin, Y. Zhu, and J. Barbič. Nonlinear material design using principal stretches.
     ACM Transactions on Graphics (TOG), 34(4):1–11, 2015.
[92] J. Xu, Z. Chen, J. Li, S. Yang, W. Wang, X. Hu, and E. C.-H. Ngai. Fourierkan-gcf: Fourier
     kolmogorov-arnold network–an effective and efficient feature transformation for graph col-
     laborative filtering. arXiv preprint arXiv:2406.01034, 2024.
[93] K. Xu, L. Chen, and S. Wang. Kolmogorov-arnold networks for time series: Bridging pre-
     dictive power and interpretability. arXiv preprint arXiv:2406.02496, 2024.
[94] J. Yang, R. Walters, N. Dehmamy, and R. Yu. Generative adversarial symmetry discovery. In
     International Conference on Machine Learning, pages 39488–39508. PMLR, 2023.
[95] K. Yang, A. Swope, A. Gu, R. Chalamala, P. Song, S. Yu, S. Godil, R. J. Prenger, and
     A. Anandkumar. Leandojo: Theorem proving with retrieval-augmented language models.
     Advances in Neural Information Processing Systems, 36, 2024.
[96] S. Yang, L. Qin, and X. Yu. Endowing interpretability for neural cognitive diagnosis by
     efficient kolmogorov-arnold networks. arXiv preprint arXiv:2405.14399, 2024.
[97] R. Yu, W. Yu, and X. Wang.           Kan or mlp: A fairer comparison.           arXiv preprint
     arXiv:2407.16674, 2024.
[98] E. Zeydan, C. J. Vaca-Rubio, L. Blanco, R. Pereira, M. Caus, and A. Aydeger. F-kans:
     Federated kolmogorov-arnold networks. arXiv preprint arXiv:2407.20100, 2024.
[99] F. Zhang and X. Zhang. Graphkan: Enhancing feature extraction with graph kolmogorov
     arnold networks. arXiv preprint arXiv:2406.13597, 2024.


                                                26
[100] Z. Zhong, Z. Liu, M. Tegmark, and J. Andreas. The clock and the pizza: Two stories in mech-
      anistic explanation of neural networks. In Thirty-seventh Conference on Neural Information
      Processing Systems, 2023.
[101] S. Zinage, S. Mondal, and S. Sarkar. Dkl-kan: Scalable deep kernel learning using
      kolmogorov-arnold networks. arXiv preprint arXiv:2407.21176, 2024.




                                               27

