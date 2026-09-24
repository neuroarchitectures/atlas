# A physics-informed graph neural network conserving linear Sharma Fink 2025

> Source: `A_physics-informed_graph_neural_network_conserving_linear_Sharma_Fink_2025.pdf`

---

                                    Article                                                                                             https://doi.org/10.1038/s41467-025-67802-5


                                    A physics-informed graph neural network
                                    conserving linear and angular momentum
                                    for dynamical systems
                                    Received: 23 January 2025                         Vinay Sharma         & Olga Fink

                                    Accepted: 9 December 2025

                                                                                      Accurate, interpretable, and real-time modeling of multi-body dynamical sys-
                                                                                      tems is essential for predicting behaviors and inferring physical properties in
                                       Check for updates
                                                                                      natural and engineered environments. Traditional physics-based models face

1234567890():,;   1234567890():,;
                                                                                      scalability challenges and are computationally demanding, while data-driven
                                                                                      approaches like Graph neural networks (GNNs) often lack physical con-
                                                                                      sistency, interpretability, and generalization. In this paper, we propose
                                                                                      DYNAMI-CAL GRAPHNET, a Physics-Informed Graph Neural Network that
                                                                                      integrates the learning capabilities of GNNs with physics-based inductive bia-
                                                                                      ses to address these limitations. DYNAMI-CAL GRAPHNET enforces pairwise
                                                                                      conservation of linear and angular momentum for interacting nodes using
                                                                                      edge-local reference frames that are equivariant to rotational symmetries,
                                                                                      invariant to translations, and equivariant to node permutations. This design
                                                                                      ensures physically consistent predictions of node dynamics while offering
                                                                                      interpretable, edge-wise linear and angular impulses resulting from pairwise
                                                                                      interactions. Evaluated on a 3D granular system with inelastic collisions,
                                                                                      DYNAMI-CAL GRAPHNET demonstrates stable error accumulation over
                                                                                      extended rollouts, effective extrapolation to unseen conﬁgurations, and
                                                                                      robust handling of heterogeneous interactions and external forces. DYNAMI-
                                                                                      CAL GRAPHNET offers signiﬁcant advantages in ﬁelds requiring accurate,
                                                                                      interpretable, and real-time modeling of complex multi-body dynamical sys-
                                                                                      tems, such as robotics, aerospace engineering, and materials science. By
                                                                                      providing physically consistent and scalable predictions that adhere to fun-
                                                                                      damental conservation laws, it enables the inference of forces and moments
                                                                                      while efﬁciently handling heterogeneous interactions and external forces. This
                                                                                      makes it invaluable for designing control systems, optimizing mechanical
                                                                                      processes, and analyzing dynamic behaviors in both natural and engineered
                                                                                      systems.


                                    Dynamical systems are fundamental to both natural and engineered           systems. Accurately modeling these systems is essential for predicting
                                    environments, encompassing phenomena such as granular ﬂows,                behaviors and informing design, optimization, and operational man-
                                    molecular dynamics, and planetary motions in nature, as well as            agement decisions. In operational settings, reliable models are espe-
                                    engineered components like bearings, gearboxes, and suspension             cially valuable for learning from observational data, tracking state


                                    Intelligent Maintenance and Operations Systems, EPFL, Lausanne, Switzerland.   e-mail: olga.ﬁnk@epﬂ.ch


                                    Nature Communications | (2026)17:1045                                                                                                           1
Article                                                                                               https://doi.org/10.1038/s41467-025-67802-5


evolution in real time, and inferring key physical quantities that inﬂu-   node features to produce richer scalar embeddings that generate
ence system dynamics1. However, developing physics-grounded                learned coefﬁcients to modulate local basis vectors-enabling direc-
models with explicit parametric differential equations requires a          tional encoding beyond what relative vectors offer. Equivariant Graph
deep understanding of underlying mechanics-posing challenges for           Hierarchical Networks (EGHN)21 build on GMN by incorporating hier-
complex systems with unknown interaction laws or unmeasurable              archical message passing to capture multi-scale dynamics. Other
parameters. Moreover, for many systems, the computational cost of          methods in this paradigm include Radial Field Networks (RF)22 and
numerical simulation hinders real-time deployment2, motivating the         SchNet16.
need for alternative approaches that learn interpretable dynamical               The second class of high-degree steerable models includes
models directly from observed data and enable fast inference in            methods that encode steerable features using spherical harmonics,
operational settings.                                                      transform them under rotations via Wigner-D matrices, and fuse them
     To achieve these advantages, data-driven models that learn            through Clebsch-Gordan tensor products-achieving full SE(3) equiv-
dynamics from trajectory data have become popular, including sur-          ariance (i.e., equivariance to the Special Euclidean group in 3D,
rogate models for predictive maintenance3, control4, and efﬁcient          encompassing both rotations and translations) at a higher computa-
multi-body simulations5. However, these models often lack physical         tional cost18. Representative examples include Tensor Field Networks
consistency and generalize poorly beyond training conditions, learn-       (TFNs)23, SE(3)-Transformers24, Neural Equivariant Interatomic Poten-
ing spurious patterns tied to the training distribution6. Additionally,    tials (NequIP)25, and Steerable E(3) Equivariant Graph Neural Networks
they suffer from error accumulation during rollout, leading to poor        (SEGNNs)26.
long-term predictions. Moreover, they typically require large training           Symmetry-based inductive biases-such as translation and rotation
datasets-feasible for small systems with simple dynamics but prohibi-      equivariance—have substantially advanced the capability of GNNs to
tive in complex multiphysics simulations (e.g., Direct numerical           model the dynamics of physical systems. However, these symmetries
simulation of ﬂuid ﬂow) or real-world scenarios with limited, costly       alone do not always guarantee that the learned models will respect
measurements, such as full-body motion capture or internal loads in        fundamental physical laws. Several approaches have sought to embed
rotating machinery.                                                        hard physical priors by enforcing energy conservation, such as in
     Physics-Informed Neural Networks7 address data scarcity and           Hamiltonian and Lagrangian GNNs27,28. While effective for conservative
promote physical consistency by enforcing a learning bias during           systems, such energy-based formulations tend to underperform in
training. This involves using governing equations as additional con-       realistic, non-conservative settings involving dissipation, external for-
straints alongside data. However, they are sensitive to hyperpara-         cing, or constraints. Hamiltonian and Lagrangian formulations
meters, expensive to train8, and face challenges in complex systems        describe dynamics via a scalar energy or action whose gradients yield
like multi-body dynamics, where enforcing constraints at every step        forces and torques-an elegant construction under conservation, but
becomes difﬁcult (e.g., modeling a 9-segment human walker requires         less adaptable when energy is not preserved. For example, incorpor-
17 nonlinear constraints9). Moreover, governing equations often rely       ating Hamiltonian structure into EGNN has been shown to degrade
on simpliﬁed assumptions to model effects like nonlinear friction,         performance on such tasks15. In these cases, energy-based models
resulting in suboptimal performance compared to data-driven                require explicit augmentation to represent dissipative or externally
alternatives10.                                                            driven effects29,30.
     Graph Neural Networks (GNNs) offer a ﬂexible alternative for                In contrast, Newtonian mechanics provides a more direct and
learning the dynamics of physical systems by embedding inductive           general formulation of dynamics by expressing interactions through
biases into their architecture. Spatial inductive bias—representing        three-dimensional vector quantities-forces and torques. According to
components as nodes and interactions as edges—enables GNNs to              Newton’s third law, internal pairwise interactions inherently conserve
learn the dynamics of physical systems via message passing, as             linear and angular momentum, even in the presence of dissipation or
demonstrated by the Graph Neural Simulator (GNS)11. GNNs have since        external forcing. Although learning in this vector space is more com-
been applied across domains, including molecular dynamics12, gran-         plicated than regressing a scalar potential, it remains computationally
ular ﬂows13, and engineered systems such as bearings14. However,           efﬁcient and naturally aligned with equivariant GNNs, where interac-
models relying solely on spatial inductive bias often struggle to          tions are exchanged as vectorial messages between nodes.
maintain physical consistency, leading to error accumulation in long             Building on this Newtonian formulation, we propose a principled
rollouts and poor generalization to unseen conditions15.                   method that integrates the conservation of linear and angular
     Another important inductive bias in physical modeling is sym-         momentum directly into our equivariant GNN architecture by enfor-
metry—speciﬁcally, equivariance to 3D translations and rotations—          cing Newton’s third law at the level of internal pairwise interactions. By
reﬂecting the principle that physical laws are independent of the          embedding these universal conservation laws as structural biases
observer’s coordinate frame16,17. Equivariant-GNNs incorporate this        within the network, we ensure that the model’s predictions con-
inductive bias, enabling improved modeling of physical systems18.          sistently respect these key physical principles, even under non-
Broadly, Equivariant-GNNs fall into two classes: (i) scalarization-        conservative conditions.
vectorization approaches that operate directly in 3D space, and (ii)             While some existing models (e.g., RF22, EGNN17, GMN19, and
high-degree steerable models that lift features to higher-order repre-     ClofNet20) can conserve linear momentum under certain architectural
sentations using spherical harmonics.                                      constraints, this property is often lost in practice. For instance, RF,
     In the scalarization-vectorization paradigm, directional messages     EGNN, and GMN form edge embeddings as mij = ϕ(ZTZ, hi, hj), where Z
are generated by ﬁrst computing scalar edge embeddings from node           encodes relative geometric features (such as x    ~ij or ~
                                                                                                                                    vij ). Incorporating
and edge features (scalarization), and then using them to scale geo-       node features hi, hj for expressivity often results in non-symmetric
metric vectors such as relative positions (vectorization). For example,    embeddings (mij ≠ mji), which, when used to scale antisymmetric
E(n)-Equivariant Graph Neural Networks (EGNNs)17 follow this               relative vectors (e.g., ψðmij Þ  ~
                                                                                                             xij ), lead to non-antisymmetric forces
approach by modulating relative position vectors with learned               ~      ~
                                                                           (f ≠  f ) and violate the net force cancellation required for linear
weights. Building on this, Graph Mechanics Networks (GMN)19 extend           ij    ji

single-channel message passing by incorporating multiple geometric         momentum conservation. ClofNet introduces more expressive, non-
channels—e.g., relative position and velocity vectors—each scaled                                                         xi × ~
                                                                           relative edge embeddings (due to features like ~    xj in addition to
independently to capture richer interactions. Further, ClofNet20           relative ~
                                                                                    xij ) by projecting geometric features onto an edge-local
introduces equivariant edge-local reference frames, using projected        reference frame. However, it inherits the same node dependence


Nature Communications | (2026)17:1045                                                                                                                 2
Article                                                                                                             https://doi.org/10.1038/s41467-025-67802-5


(mij = ϕ(ZTZ, hi, hj)) and thus fails to ensure mij = mji. Moreover, its             interaction vector (governing total angular momentum exchange),
orthonormal basis ð~    a, ~
                           b, ~
                              cÞ, deﬁned as ~       bij , ~
                                              aij = x           bi × x
                                                          bij = x    bj , and        and (iii) a predicted reference point. The spin torque—responsible for
            ~ , is not fully antisymmetric under node interchange:                   angular velocity updates—is computed by isolating the orbital con-
~     ~ij × b
cij = a       ij                                                                     tribution from the angular interaction vector, treating each edge as a
~ = a
aij
          ~ ,~
         ji    b = ~
              ij    b , but ~
                      ji    c =~
                               c , which again breaks the antisymmetry
                               ij   ji                                               self-contained dynamical system. (3) Spatiotemporal message passing:
needed for force conservation. Other approaches, such as Flux-GNN31                  Our message-passing scheme incorporates sub-time stepping,
and Conservation-informed GNN32, preserve ﬂux symmetry in scalar                     enabling edge embeddings to accumulate information across both
conservation laws using permutation-invariant constructions (e.g.,                   spatial neighbors and previous iterations. This design facilitates ﬁne-
DeepSets33), ensuring mij = mji. However, both are designed for scalar               grained dynamic modeling and robust generalization across diverse
partial differential equations (PDEs): FluxGNN relies on radial vectors              physical systems.
(normals to cells) and cannot capture non-central forces—a drawback                       Overall, DYNAMI-CAL GRAPHNET advances the ﬁeld by directly
shared with EGNN-while CiGNN lacks rotational equivariance. Fur-                     embedding the core conservation principles of classical mechanics
thermore, these methods construct edge embeddings solely from                        into the model architecture, enabling accurate and generalizable pre-
relative features, which can limit their expressivity when modeling                  dictions for the dynamics of complex, real-world systems. It is a gen-
complex directional interactions.                                                    eral framework for modeling six-degree-of-freedom (6-DoF) dynamics
      Conserving angular momentum poses an even greater challenge.                   in complex physical systems using a structured scalarization-
For two bodies with moments of inertia Ii, masses mi, angular velocities             vectorization pipeline. The model is highly versatile, accommodating
ωi , and linear velocities ~
~                             vi -interacting via equal and opposite but non-        a wide range of systems-granular assemblies, biomolecules, and
central forces (i.e., not aligned with the vector connecting their cen-              articulated human motion-by representing them as graphs. In this
ters), the total angular momentum about a reference point ~            r0 is given   formulation, nodes encode position, linear velocity v                 ~i , and angular
               P                        P
by both spin        ~ i and orbital ð~
                  Iiω                       ri  ~         ~i contributions. Non-
                                                 r0 Þ × m i v                        velocity ~ωi , while bi-directional edges represent pairwise interactions
central forces alter the orbital component of angular momentum by                    between system components.
changing linear momentum, necessitating compensatory rotational                           The overall architecture and data ﬂow are illustrated in Fig. 1,
torques to preserve total angular momentum. These torques, arising                   highlighting how this graph-based approach enables ﬂexible and
from force-moment arm interactions or pure couples, are neither                      accurate modeling of complex dynamical behaviors across diverse
symmetric nor antisymmetric and are not explicitly modeled in prior                  domains. Each edge is assigned a local orthonormal reference frame
GNN architectures. While forces govern changes in translational                      that is equivariant to 3D rotations (SO(3)), invariant to translations
degrees of freedom, it is these torques that drive the evolution of the              (T(3)), and antisymmetric under node interchange (Fig. 1b.i). In
rotational state.                                                                    practice, this means that if the direction of an edge is reversed, all
      To address these limitations, we propose DYNAMI-CAL GRAPH-                     three basis vectors change signs-ensuring antisymmetry in all sub-
NET-a DYNAMIcs-predictor GRAPH neural NETwork that explicitly                        sequent projections and derived interactions. In the scalarization
Conserves Angular and Linear momentum by embedding these con-                        step (Fig. 1b.ii), node vector features-such as velocity and angular
servation laws directly into the model architecture. This approach                   velocity-are projected onto these edge-local frames, yielding scalar
enables physically consistent predictions even under complex, non-                   components. These projected scalars are then combined with other
central, and dissipative interactions, while remaining applicable across             scalar node features to create edge embeddings that are invariant to
diverse systems. As a six-degree-of-freedom model, DYNAMI-CAL                        node ordering. This approach encodes both the directional and
GRAPHNET predicts both internal forces and rotational torques                        scalar information about local interactions, all while preserving the
through three key innovations: (1) Edge-local reference frames: We                   system’s underlying symmetries. During vectorization (Fig. 1b.iii), the
introduce a edge-aligned orthonormal basis that is equivariant to 3D                 edge embeddings are decoded into physically meaningful interac-
rotations (SO(3)), invariant to translations (T(3)), and antisymmetric               tion terms.
under node exchange-ensuring equal and opposite internal forces in                        First, internal forces are predicted as antisymmetric vectors
accordance with Newton’s third law. Node vector features (e.g., velo-                ~ = F
                                                                                     F       ~ , representing changes in linear momentum per node and
                                                                                       ij      ji
city, angular velocity) are projected onto this basis and combined with              ensuring local conservation (Fig. 1b.iii.i). To achieve this, three scalar
node and edge scalar features to form expressive invariant edge                      coefﬁcients are extracted from each edge embedding using learned
embeddings during scalarization. While several recent approaches                     functions. These coefﬁcients modulate the basis vectors of the edge-
have explored the use of reference frames to achieve equivariance,                   local reference frame, reconstructing the 3D force vector. Because the
they differ fundamentally from the dynamic edge-centric formulation                  edge embeddings are invariant under node interchange, the decoded
adopted here. The method in ref. 34 deﬁnes node-centric coordinate                   scalar coefﬁcients also satisfy f1,ij = f1,ji, f2,ij = f2,ji, and f3,ij = f3,ji. Together
systems and enforces global rotational and translational invariance in               with the antisymmetric ﬂipping of the local basis vectors, this ensures
node dynamics, but does not model pairwise interactions within local                 that the reconstructed force vectors for edges i → j and j → i are equal in
frames or enforce antisymmetry between interacting nodes.                            magnitude and opposite in direction, ensuring ~                       Fij =  F  ~ . This
                                                                                                                                                                       ji
Similarly35, learns a single global canonical frame shared across all                decoding mechanism directly embeds conservation of linear
nodes, providing a common orientation for message passing but lim-                   momentum into the architecture. Moreover, because the force vectors
iting the model’s ability to represent distinct relative conﬁgurations               are constructed from a reference frame that is SO(3)-equivariant and
between node pairs. ClofNet20 introduces edge-local reference frames;                T(3)-invariant, they inherit these symmetry properties.
however, its basis construction is only partially antisymmetric under                     Second, angular momentum changes are decoded as antisym-
node exchange (~   cij = ~
                         cji ), preventing strict enforcement of equal-and-          metric vectors ~ Aij =  ~ Aji , following the same approach: scalar coefﬁ-
opposite internal forces and, consequently, exact momentum con-                      cients from the edge embedding are used to scale the local basis
servation. In contrast, DYNAMI-CAL GRAPHNET employs dynamic,                         vectors (Fig. 1b.iii.ii). The resulting vectors represent the total angular
antisymmetric edge-aligned frames that evolve jointly with the inter-                momentum exchange between nodes i and j, combining both spin and
action state, enabling explicit conservation of both linear and angular              orbital components. Only the spin component, however, directly
momentum through physically consistent pairwise exchanges. (2)                       affects angular velocity. To isolate it, the orbital contribution, com-
Physically grounded vectorization: Edge embeddings are decoded into                  puted as the cross product of the predicted internal force and the
three vector channels: (i) an antisymmetric internal force vector                    relative position vector of the body from the reference point, is sub-
(enforcing linear momentum exchange), (ii) a pairwise angular                        tracted from the total angular momentum change. This step relies on


Nature Communications | (2026)17:1045                                                                                                                                       3
Article                                                                                                             https://doi.org/10.1038/s41467-025-67802-5




Fig. 1 | DYNAMI-CAL GRAPHNET. a Input the model inputs graph representation            subtraction of the orbital component, yielding non-symmetric torques that update
of the dynamical system state at time “t”. b Scalarization-vectorization scheme with   angular velocity. c Spatiotemporal message passing c.i decoded edge interaction
conservation of linear and angular momentum b.i each edge is equipped with an          vectors are aggregated at nodes. c.ii The node aggregated vectors are scaled using
SO(3)-equivariant, T(3)-invariant, and antisymmetric local reference frame, enfor-     learned coefﬁcients to decode linear and angular velocity updates. These updates
cing symmetry under node interchange. b.ii Scalarization: node vector features         are then integrated using implicit Euler stepping to compute the new node posi-
(e.g., ~   ~ ) are projected into these frames and combined with scalar features to
        v, ω                                                                           tions. c.iii The edge embeddings are retained as latent memory and used as skip
form invariant scalar edge embeddings. b.iii Vectorization: decoding edge inter-       connections to inform the edge embeddings in the subsequent message-passing
actions. b.iii.i Embeddings are decoded into antisymmetric internal forces             step. c.iv The graph with updated node states is used for the next message passing
(~
 Fij =  F ~ ), conserving linear momentum. b.iii.ii Decoded antisymmetric angular     step. This mechanism enables the model to perform spatiotemporal reasoning with
            ji
momentum changes (~       Aij =  ~
                                  Aji ) ensure total angular momentum conservation.    multiple message passing steps within a single prediction step.
b.iii.iii The predicted reference point enables isolation of spin torque via


decoding a consistent reference point, x        ~0ij = x~0ji , which is shared         internal forces and the consistent, shared structure of the reference
between both directions of an edge between two interacting bodies                      points used for each edge.
(Fig. 1b.iii.iii). This reference point serves as the effective location                    The spatiotemporal message-passing scheme in DYNAMI-CAL
where angular momentum is conserved, enabling spin component to                        GRAPHNET is applied after computing physically consistent internal
be computed from impulse-momentum relation for unit time as                            forces and torques at each edge. As illustrated in Fig. 1c, the decoded
τ ij = ~
~      Aij  ð~    x0ij Þ × ~
              ri  ~        Fij where ~
                                      Fij Δt = mi Δ~
                                                   vi :       While    angular         edge-wise internal forces and rotational torques are ﬁrst aggregated
momentum is typically conserved globally about a ﬁxed reference                        on the connected nodes to obtain the net force and torque acting on
point, DYNAMI-CAL GRAPHNET instead enforces conservation locally                       each node. These vectors are then scaled by coefﬁcients derived from
at each edge by anchoring interactions to the predicted reference                      the scalar node embeddings, resulting in updates to each node’s linear
point. This localized formulation enables modular, ﬁne-grained mod-                    and angular velocities. The updated positions (and, optionally, orien-
eling of complex systems, scales efﬁciently to large graphs, and natu-                 tations) are then computed using implicit Euler integration. This pro-
rally incorporates non-central and dissipative effects. As demonstrated                cess constitutes a single message-passing layer of DYNAMI-CAL
in Supplementary Section §5.1, this edge-level formulation provably                    GRAPHNET. Crucially, this message-passing step is iteratively repeated
ensures global conservation of angular momentum under symmetry-                        to emulate sub-time stepping within a single prediction interval. The
preserving aggregation. This follows directly from the antisymmetry of                 architecture uses only two blocks-an initialization block for the ﬁrst


Nature Communications | (2026)17:1045                                                                                                                                  4
Article                                                                                               https://doi.org/10.1038/s41467-025-67802-5


step and a shared block reused for all subsequent steps. At each             physically consistent, enabling efﬁcient modeling of complex bound-
iteration, the most recent node states and the previously computed           ary interactions in a wide range of dynamical systems. Further details
edge embeddings are used to inform the next round of edge encoding.          can be found in Section “Scalar edge embedding from projections onto
This evolving representation is maintained as a latent memory on each        edge-reference frames”.
edge-referred to as Edge Memory in Fig. 1c.iii. As a result, the model
achieves spatiotemporal reasoning by continually enriching edge              Results
embeddings with both spatial context (from neighboring nodes) and            Overview of experiments
temporal coherence (through accumulated interaction history across           We evaluate DYNAMI CAL GRAPHNET across four benchmarks span-
message passing steps that mimic explicit time stepping). This design        ning simulated and real-world physical systems. These include two
allows DYNAMI-CAL GRAPHNET to capture dynamic behavior over                  simulated domains–(i) granular six-degree-of-freedom (6-DoF) colli-
multiple time scales while preserving physically grounded inductive          sions and (ii) charged particles connected by sticks or hinges-and two
biases at each step.                                                         real-world domains characterized by complex spatiotemporal dyna-
     We further propose a mesh-free and particle-free modeling of            mics–(iii) human walking kinematics from Carnegie Mellon University
boundaries. To achieve a comprehensive representation of dynamic             (CMU) motion-capture data39, and (iv) protein molecular dynamics in
systems, it is essential to accurately model interactions with bound-        water and ion solution under isothermal-isobaric (NPT: constant
aries such as walls, ﬂoors, and rigid enclosures. These boundaries are       number of particles, pressure, and temperature) conditions (300 K,
integral to system behavior–constraining motion in robotic systems,          1 bar)40. These tasks capture key challenges in modeling dynamical
supporting the body, generating ground reaction forces, and conﬁning         systems, such as rotational dynamics, holonomic constraints, spatio-
particles in granular simulations. Existing approaches typically repre-      temporal coherence, and ﬁne-scale conformational changes. We
sent boundaries with dense meshes or collections of particles36–38.          compare DYNAMI CAL GRAPHNET with several state-of-the-art base-
While effective, these approaches introduce signiﬁcant computational         lines for each benchmark. All models are trained using single-step
overhead and require special treatment of wall-body interactions,            supervision, with multi-step rollouts used where applicable to assess
distinct from body-body interactions. An alternative is to represent         long-horizon prediction accuracy and stability. Overall, these evalua-
boundaries implicitly by embedding additional features—such as dis-          tions demonstrate the ability of DYNAMI CAL GRAPHNET to model
tance to the wall—for each component11. Although this method is more         diverse dynamical systems with varying physical constraints, interac-
efﬁcient, it struggles in scenarios with multiple or moving boundaries,      tion types, and temporal scales.
since ﬁxed distance features cannot distinguish overlapping con-                   The granular 6-DoF collision benchmark serves as a primary
straints or adapt to dynamic, time-varying surfaces. We address these        testbed for evaluating the ability of DYNAMI CAL GRAPHNET to model
limitations by proposing a mesh-free boundary treatment for GNNs             coupled translational and rotational dynamics under contact-rich,
that uniﬁes body-wall and body-body interactions within a single fra-        dissipative conditions. The dataset consists of 6-DoF trajectories of
mework. Our approach reﬂects all nodes in the system across the              granular spheres undergoing inelastic collisions—both inter-sphere
outward normal of each boundary to create ghost nodes. These ghost           and with enclosure walls—simulated using the MFiX Discrete Element
nodes inherit the scalar properties of the boundary (e.g., degrees of        Method (DEM)41,42. The underlying physics includes nonlinear normal
freedom) and vector features (e.g., velocity and angular velocity); for      and tangential contact forces, damping, Coulomb friction, and exter-
stationary walls, the vector features are set to zero. Edges are then        nally applied forces. All simulation details, parameters, and imple-
established between original nodes and their corresponding ghost             mentation setup are provided in Supplementary Information Section
nodes according to a distance threshold—ensuring that only nodes             §1.1. On this benchmark, we evaluate model performance using three
sufﬁciently close to the boundary are connected to their reﬂections.         physically grounded experiments that assess generalization, con-
These edges naturally encode body–wall interactions, capturing nor-          servation behavior, and robustness to external forcing. In the ﬁrst
mal forces, tangential reactions, and frictional effects. Additionally,      experiment described in Section “Conﬁned 6-DoF granular collisions”,
ghost nodes can inherit the motion of moving or rotating walls,              we introduce a conﬁned granular collisions experiment comprising
allowing for the accurate modeling of dynamic boundary effects.              ﬁve simulated trajectories of 60 identical spheres conﬁned within a
During message passing, their states are overwritten at each step with       stationary cuboidal enclosure and initialized with random velocities.
the wall’s prescribed values, ensuring that boundary dynamics are            This setting evaluates the model’s ability to generate stable long-
correctly enforced and learned by the network. The reﬂective                 horizon rollouts and capture the physically consistent evolution of
mechanism treats the boundary as an intermediate point between a             system-level kinetic energy, linear momentum, and angular momen-
node and its ghost, mirroring the modeling of body–body interactions         tum in an open, dissipative system—where energy and momentum are
through center-to-center vectors. Because reﬂections are performed           absorbed at the stationary walls. Performance is assessed under both
along the wall normals, the resulting edge directions naturally align        within-distribution (interpolation) and out-of-distribution (extrapola-
with the boundary surface normals, ensuring accurate modeling of             tion) initial velocities. Dataset conﬁguration, including training, vali-
interactions occurring in those directions. This approach requires only      dation, and test splits, learning objectives, and evaluation metrics, is
basic geometric information—such as the normal vector and a point on         detailed in Supplementary Information Section §1.1. In the second
the plane for ﬂat walls, or the axis and radius for cylindrical enclosures   experiment described in Section “Evaluation of physical consistency”,
—making it both simple to implement and broadly applicable. While            we assess the physical consistency of the learned dynamics. The model
the method introduces (W − 1) × Nn additional nodes for W boundaries         trained on the homogeneous conﬁned collision task (Section “Con-
and Nn physical nodes, this overhead is ﬁxed and remains signiﬁcantly        ﬁned 6-DoF granular collisions”) is evaluated in a controlled two-
lower than that of particle- or mesh-based boundary representations.         sphere setup, undergoing an oblique collision in a closed system. This
For example, modeling large ﬂoors with explicit particles or meshes          experiment assesses the model’s ability to conserve total linear and
can quickly become infeasible due to memory constraints37, whereas           angular momentum in the absence of external forces and provides an
our method maintains a constant number of ghost nodes (i.e., Nn for a        interpretable measure of physical consistency. In the third experiment
single ﬂoor), regardless of the boundary’s size. Moreover, in GNNs, the      described in Section “Extrapolation to moving boundaries (rotating
primary computational cost arises from edge operations rather than           cylindrical hopper)”, we assess the model’s ability to generalize and
the number of nodes. Since edges are created only for nodes close to         extrapolate to previously unseen boundary conditions and large-scale
the boundary, the additional computational overhead remains mini-            system conﬁgurations. The model is trained on ﬁve trajectories of
mal. These features make the proposed approach both scalable and             60 spheres within a stationary cuboidal enclosure, where the spheres


Nature Communications | (2026)17:1045                                                                                                               5
Article                                                                                                https://doi.org/10.1038/s41467-025-67802-5


are inﬂuenced by gravity and interact via heterogeneous sphere-              energy decay as well as in the evolution of linear and angular
sphere and sphere-wall contact parameters (e.g., coefﬁcients of resti-       momentum over time (see Supplementary Information Section §1.1.4
tution, friction angles, and stiffness values). At test time, the model is   for detailed results). Their equivariant architectures struggle to model
evaluated in a signiﬁcantly more complex, real-world-inspired sce-           the non-linear, event-driven nature of inelastic collisions, while GNS,
nario: a rotating cylindrical hopper mixer with curved walls, containing     despite lacking equivariance, proves more expressive in capturing
2073 spheres and subjected to nonuniform rotational acceleration.            impulse-driven dynamics. As a result, GNS is retained as the primary
Dataset details and the extrapolation test design are provided in            baseline in the main paper.
Supplementary Information Section §1.2.1.                                         Figure 2 compares DYNAMI-CAL GRAPHNET and GNS in both
     Across the remaining benchmarks–constrained N-body dynamics             interpolation and extrapolation regimes. DYNAMI-CAL GRAPHNET
(Section “Constrained N-body dynamics”), articulated human motion            reliably retains all particles, accurately tracks kinetic energy decay, and
(Section “Human motion prediction”), and protein biomolecular con-           preserves momentum evolution over 500 steps, exhibiting low var-
formational dynamics (Section “Protein dynamics in solvent”)—we              iance across random seeds. In contrast, GNS diverges early in the
demonstrate the broad applicability of DYNAMI-CAL GRAPHNET to                extrapolation setting, leading to particle escape due to its inability to
diverse physical regimes and its competitive performance against             generalize learned interactions under high-momentum conditions,
state-of-the-art methods as system complexity, interaction structure,        where increased collision speeds require accurate resolution of
and temporal scales increase.                                                impulsive contact forces to maintain conﬁnement. Even for retained
                                                                             spheres, it predicts increasing deviations from expected physical
Conﬁned 6-DoF granular collisions                                            behavior.
We evaluate the long-horizon rollout performance of DYNAMI-CAL                    These results highlight the robustness and superior general-
GRAPHNET on a granular system of 60 identical spheres conﬁned in a           ization ability of DYNAMI-CAL GRAPHNET for modeling dissipative,
stationary cuboidal box. The training set comprises ﬁve DEM-                 contact-rich 6-DoF dynamics.
simulated trajectories under zero gravity (see Supplementary Infor-
mation Section §1.1.1 for dataset details). The model is evaluated in        Evaluation of physical consistency
both within-distribution (interpolation) and out-of-distribution             To assess physical consistency, we evaluate models trained in the
(extrapolation) regimes, with the latter initialized at approximately        homogeneous conﬁned setting (Section “Conﬁned 6-DoF granular
three times the kinetic energy observed during training. In this open,       collisions”) using a controlled two-sphere system undergoing an obli-
dissipative system, energy is lost through inelastic collisions and          que, inelastic collision. Both spheres are initialized with zero angular
momentum is absorbed by the walls, causing the system to gradually           velocity and assigned velocities to induce an angled impact. In this
settle over time.                                                            closed system, total linear and angular momentum should be con-
      We monitor the evolution of kinetic energy, linear momentum,           served, while kinetic energy dissipates due to inelasticity.
and angular momentum of the retained spheres over time, using                     Figure 3 presents results across three random seeds. DYNAMI-CAL
metric formulations detailed in Supplementary Information Section            GRAPHNET accurately preserves all components of linear and angular
§1.1.2. Unlike existing dynamical systems benchmarks that primarily          momentum and closely tracks the expected decay in kinetic energy. In
focus on qualitative behavior or position accuracy, our evaluation           contrast, the GNS baseline violates conservation laws and exhibits
focuses on physically consistent rollouts and accurate learning of           unphysical kinetic energy gain under one of the seeds. Supplementary
contact interactions—both of which are critical for stable long-horizon      Information Section §1.1.5, Fig. 2, extends this comparison to addi-
prediction.                                                                  tional baselines (EGNN, GMN, and ClofNet), which display over-
      At each time step, the system is represented as a graph, where         damped behavior and fail to conserve linear and angular momentum.
nodes correspond to spheres and encode positional and dynamical                   These ﬁndings demonstrate DYNAMI-CAL GRAPHNET’s ability to
features, including linear and angular velocities at times t and t − 1.      faithfully model contact-driven, multi-body dynamics with physically
Wall interactions are modeled via ghost nodes, which are reﬂections          consistent impulse responses. Supplementary Fig. 3 further quantiﬁes
about the enclosure walls. The ghost nodes inherit the same prop-            post-collision errors, conﬁrming DYNAMI-CAL GRAPHNET’s superior
erties as the boundaries, speciﬁcally in this case, zero velocity and        accuracy in capturing both translational and rotational dynamics.
boundary identiﬁers. Edges are established between all sphere and
ghost nodes based on a distance threshold. All features are normal-          Extrapolation to moving boundaries (rotating cylindrical
ized by their maximum values observed during training to preserve            hopper)
directionality. The model is trained to predict per-sphere updates in        To assess generalization and extrapolation under complex external
position, linear velocity, and angular velocity, using the ground-truth      forcing and boundary conditions, we evaluate DYNAMI-CAL GRAPH-
differences between consecutive time steps as targets. These pre-            NET on a real-world-inspired granular mixing task: 2073 spheres in a
dicted updates are then integrated autoregressively during infer-            rotating cylindrical hopper—a scenario relevant to industrial mixing
ence. Full implementation details are provided in Supplementary              and particulate ﬂow applications. The model is trained exclusively on
Section §1.1.2.                                                              ﬁve trajectories, each consisting of 1500 time steps, involving
      We compare DYNAMI-CAL GRAPHNET with GNS11 and our reim-                60 spheres conﬁned within a stationary cuboidal box and subject to
plemented 6-DoF variants of EGNN17, GMN19, and ClofNet20, where we           gravity, featuring heterogeneous sphere-sphere and sphere-wall con-
extend the original architectures to predict angular velocity updates in     tact parameters. During testing, particle reﬂections are computed
addition to linear velocity and position updates at each time step. All      dynamically, and the corresponding interaction graph is created at
models receive identical inputs, share the same training objectives,         each rollout step based on the curved hopper walls and their instan-
and utilize a common reﬂection-based wall modeling approach (see             taneous motion (see Supplementary Information Section §1.2.1 for
Supplementary Information Section §1.1.3). The effectiveness of our          dataset details, and Sections §1.2.2 and §1.2.3 for implementation of
proposed boundary modeling strategy is further demonstrated in               DYNAMI-CAL GRAPHNET and GNS with boundary adaptation). Fig-
Supplementary Section §1.1.6, where GNS achieves improved perfor-            ure 4a illustrates the training and test conﬁgurations.
mance over its original distance-feature formulation. Among the                   At test time, the hopper rotates about the Y-axis with a time-
reimplemented baselines, EGNN, GMN, and ClofNet consistently                 varying angular velocity, generating tangential impulses that
underperform. Evaluated on 500-step rollouts in both interpolation           induce a dynamic surface slope in the X-Z plane. Remarkably,
and extrapolation settings, they exhibit clear deviations in kinetic         despite being trained only on ﬂat, stationary boundaries,


Nature Communications | (2026)17:1045                                                                                                                 6
Article                                                                                                               https://doi.org/10.1038/s41467-025-67802-5




Fig. 2 | Long-horizon rollouts for conﬁned granular collisions. 6-DoF rollouts of        momentum evolution, while GNS diverges early in all metrics. Extrapolation
60 spheres inside a cuboidal box under two regimes: interpolation (1a–1g) and            (2a–2d): DYNAMI-CAL GRAPHNET remains stable and physically consistent under
extrapolation (2a–2g), with the latter tripling the initial kinetic energy relative to   unseen initial velocities, whereas GNS fails to conﬁne particles and exhibits large
training. Interpolation (1a–1d): system-level metrics—a number of spheres retained,      errors (metrics are calculated only for retained spheres). Rollout snapshots
b kinetic energy per unit mass, c largest component of linear momentum, and              (1e–1g and 2e–2g): Selected time steps visualize spatial dynamics across Ground
d largest component of angular momentum-tracked over 500 time steps. DYNAMI-             Truth, GNS, and DYNAMI-CAL GRAPHNET, highlighting the close match between
CAL GRAPHNET retains all particles and accurately captures energy dissipation and        predicted DYNAMI-CAL GRAPHNET trajectories and ground truth.


DYNAMI-CAL GRAPHNET delivers highly accurate predictions over                            Constrained N-body dynamics
16,000 rollout steps. The model not only tracks the detailed spa-                        To evaluate applicability to systems with mixed interaction types and
tial trajectories of thousands of particles (Fig. 4b), but also pre-                     structural constraints, we use the Constrained N-Body dataset intro-
cisely captures the evolving macroscopic surface slope                                   duced in ref. 19, which extends the 3D charged particle simulation of
throughout the entire simulation (Fig. 4c).                                              Kipf et al.43 by incorporating holonomic constraints in the form of rigid
      In contrast, the GNS baseline (see implementation details for this                 sticks and hinges (see Supplementary Information Section §2.1 for
case in Supplementary Section §1.2.3) destabilizes early and fails to                    dataset details). This benchmark presents a challenging testbed for
generalize to the new boundary conditions. These results highlight the                   dynamics prediction, as it combines long-range Coulomb interactions
strong extrapolation capability of DYNAMI-CAL GRAPHNET, demon-                           with constraint-induced couplings that govern collective motion.
strating generalization across conﬁgurations, initial conditions, and                    Given the state of the system at time t, the target is to predict the future
boundary regimes. This experiment further underscores DYNAMI-CAL                         state at time t + 10, which corresponds to 1000 simulation time steps.
GRAPHNET’s ﬂexibility as a deployable, general-purpose simulator—                              We benchmark DYNAMI-CAL GRAPHNET against several strong
capable of handling diverse geometries and evolving environments                         baselines: GMN19 (constraint enforcement via generalized coordinates
without retraining to each new conﬁguration, architectural changes or                    and handcrafted forward kinematics), EGNN17 (lightweight E(n)-
ad hoc interventions such as remeshing.                                                  equivariant message passing), EGNNReg19 (explicit constraint penal-
      Additional evaluations presented in the Supplementary Informa-                     ties), Radial Field Networks (RF)22 (E(n)-equivariant updates based on
tion further demonstrate the model’s robustness and versatility: it deli-                edge distances), Tensor Field Networks (TFN)23 (SE(3)-equivariant
vers accurate angular responses across a range of impact angles (Section                 feature propagation with spherical harmonics), SE(3)-Transformer24
§1.1.7), maintains stability under sparse temporal sampling and stiff                    (attention-based extension of TFN), ClofNet20 (edge-wise local refer-
interactions (Section §1.1.8), and provides interpretable force decom-                   ence frames), a message-passing GNN44, and a linear kinematic pre-
position into tangential and normal components (Section §1.1.9).                         dictor p(t) = p(0) + v(0)t.



Nature Communications | (2026)17:1045                                                                                                                                     7
Article                                                                                                         https://doi.org/10.1038/s41467-025-67802-5




Fig. 3 | Oblique collision of two granular spheres. a Collision trajectory. b Total linear momentum per unit mass. c Total angular momentum. d Total kinetic energy.
DYNAMI-CAL GRAPHNET accurately conserves momentum and predicts energy dissipation, while GNS violates conservation laws.



      Additionally, we also report a data-augmented baseline for the               rollout evaluation, we report results for GMN only, which is the
message-passing GNN (GNN (aug.)), in which all training samples are                strongest-performing one-step baseline. For rollout, both GMN and
subjected to uniform random 3D rotations and small random transla-                 DYNAMI-CAL GRAPHNET were trained to predict the single-step posi-
tions (Gaussian noise with standard deviation 0.2 in normalized units)             tion and velocity targets. The original paper19 did not evaluate GMN
at each training epoch. This procedure effectively doubles the dataset             under multi-step rollout; we include this to assess long-term stability and
size per epoch and exposes the non-equivariant model to continuous                 physical consistency in predicted dynamics.
geometric symmetries, thereby providing a fairer reference for com-                     Figure 5 summarizes the results on the Constrained N-Body
parison against equivariant architectures.                                         benchmark. DYNAMI-CAL GRAPHNET consistently outperforms all
      For fair comparison, we adopt the training and evaluation settings           baseline models in both single- and multi-step prediction tasks. In
presented in ref. 19 for conﬁguring DYNAMI-CAL GRAPHNET (see Sup-                  Fig. 5a, our model achieves the lowest single-step prediction error on
plementary Information Section §2.2 for implementation details). For               both seen (3, 2, 1) and unseen (2, 4, 0) and (1, 0, 3) conﬁgurations,
EGNN, EGNNReg, RF, TFN, SE(3)-Transformer, and the standard                        surpassing GMN, EGNN, and ClofNet.
message-passing GNN, we report results directly from ref. 19, all                       Introducing random rotation-translation augmentation notably
obtained under the same experimental setup. In our experiments, we                 improves the performance of the non-equivariant GNN, narrowing its
additionally evaluate ClofNet on this benchmark using its publicly                 gap to equivariant architectures. However, despite the increased data
released implementation, aligning its conﬁguration with that of the                volume and explicit symmetry exposure, GNN (aug.) remains less
other baselines. We also conduct additional rollout evaluations for GMN            accurate than the simplest equivariant baseline (EGNN) and DYNAMI-
using its publicly available code and default settings. Further details on         CAL GRAPHNET across all evaluated conﬁgurations, underscoring that
all baseline models are provided in Supplementary Information Section              architectural inductive biases remain essential for generalizing con-
§2.3. All models are trained using single-step supervision; for multi-step         strained physical dynamics.


Nature Communications | (2026)17:1045                                                                                                                             8
Article                                                                                                         https://doi.org/10.1038/s41467-025-67802-5




Fig. 4 | Extrapolation to rotating curved boundaries: performance in cylind-       accurately predicts particle motion and surface evolution, matching the DEM
rical hopper. a DYNAMI-CAL GRAPHNET, trained on 60-sphere trajectories within      ground truth (red). In contrast, GNS (green) destabilizes early, failing to capture
stationary box walls, is evaluated on a cylindrical hopper with 2073 spheres and   stable sphere-wall interactions. c Evolution of the average X-Z surface slope over
rotating walls. The wall rotation proﬁle (rev/s vs. time) is shown at top right.   time. The slope responds to changing rotation direction and speed; DYNAMI-CAL
b Rollout snapshots over 16,000 steps show that DYNAMI-CAL GRAPHNET (blue)         GRAPHNET accurately reproduces the ground truth surface evolution.



     For multi-step rollout prediction, Fig. 5b compares DYNAMI-CAL                physically consistent internal forces and moments, without relying on
GRAPHNET with GMN, showing DYNAMI-CAL GRAPHNET maintains                           explicit constraint formulations, as in GMN. GMN enforces structural
stable long-horizon accuracy over multi-step rollout up to four steps              constraints through generalized coordinates and a handcrafted for-
(one step = 10 frames = 1000 simulation steps), whereas GMN accu-                  ward kinematics module, making it susceptible to integration errors
mulates signiﬁcant error over time. Figure 5c presents qualitative                 and constraint drift over long rollouts. In contrast, DYNAMI-CAL
rollouts on the unseen (1, 0, 3) conﬁguration, demonstrating that our              GRAPHNET employs evolving edge embeddings that serve as latent
model accurately captures constrained dynamics, despite being                      memory units—conditioned on edge type and iteratively updated
trained only with single-step supervision on a different topology.                 through message passing across spatial neighbors and multiple sub-
     While all models receive edge-type labels (e.g., stick or hinge),             time steps—to effectively capture temporal dynamics. Coupled with an
DYNAMI-CAL GRAPHNET uniquely exploits this information to infer                    architecture grounded in physical laws, such as conservation of linear


Nature Communications | (2026)17:1045                                                                                                                               9
Article                                                                                                           https://doi.org/10.1038/s41467-025-67802-5




Fig. 5 | Performance on the constrained N-body benchmark. Models are trained          constrained dynamics on unseen (1, 0, 3) system. d Prediction error vs. number of
on the (3, 2, 1) conﬁguration and evaluated on both seen and unseen systems: (3, 2,   training samples on constrained N-body system (3, 2, 1). DYNAMI-CAL GRAPHNET
1), (2, 4, 0), and (1, 0, 3). a Single-step error: DYNAMI-CAL GRAPHNET achieves the   (blue) achieves the best performance across all training sizes, demonstrating
lowest error across all conﬁgurations. b Multi-step rollout: compared to GMN, our     strong data efﬁciency.
model maintains stable accuracy over long horizons. c Qualitative rollout: accurate



and angular momentum, this approach enables the model to learn                        Section §2.4), which show that removing either conservation laws or
robust and generalizable dynamics across a wide range of constrained                  spatiotemporal message passing substantially impairs performance.
systems. The importance of these components is further supported by                        We further assess data efﬁciency on the constrained N-body (3, 2,
ablation results on the (3, 2, 1) setup (Supplementary Information                    1) setup by varying the number of training samples. This dataset is


Nature Communications | (2026)17:1045                                                                                                                               10
Article                                                                                                https://doi.org/10.1038/s41467-025-67802-5


chosen to enable direct benchmarking against the reported data efﬁ-          ground-truth motion over 90 future frames. In this experiment, both
ciency results in ref. 19, which include GNN, EGNN, and GMN. In              GMN and DYNAMI-CAL GRAPHNET were trained for 1000 epochs to
addition, we evaluate ClofNet using its publicly available imple-            predict the target position and velocity.
mentation under the same settings.                                                 Together with the results on the constrained N-body system
     Figure 5d presents single-step prediction errors for DYNAMI-CAL         (Section “Constrained N-body dynamics”), these ﬁndings provide
GRAPHNET and all considered baselines across different training set          strong empirical evidence that DYNAMI-CAL GRAPHNET performs
sizes. The dashed curve (GNN (aug.)) indicates that per-epoch random         effective spatiotemporal reasoning by evolving edge embeddings over
SO(3) rotations and small translations improve the non-equivariant           time. These embeddings are enriched with spatial context through
GNN. However, its accuracy remains below EGNN and DYNAMI-CAL                 message passing between neighboring nodes and are temporally
GRAPHNET at all sizes-even with effectively doubled data–consistent          propagated via iterative updates that mimic sub-time stepping. Cru-
with prior studies showing that data augmentation cannot fully sub-          cially, this mechanism is grounded in universal physical inductive
stitute for built-in equivariance45,46.                                      biases—namely, conservation of linear and angular momentum gov-
     DYNAMI-CAL GRAPHNET demonstrates strong performance even                erning internal forces and moments—which apply across diverse phy-
with as few as 500 training samples, exhibiting only marginal                sical systems. As a result, the model is able to learn localized physical
improvement as more data is added. This highlights the model’s ability       interactions over space and time, enabling robust and generalizable
to learn robust dynamics from limited data. Extended large-scale             dynamics across a wide range of structured physical systems.
experiments (50 k–100 k samples) reported in Supplementary Infor-
mation Section §2.4 conﬁrm that this advantage persists in the high-         Protein dynamics in solvent
data regime, with DYNAMI-CAL GRAPHNET retaining the lowest error             We choose this benchmark to assess DYNAMI-CAL GRAPHNET’s ability
across all dataset scales.                                                   to capture ﬁne-grained conformational changes across multiple spatial
                                                                             and temporal scales, and to demonstrate its applicability to a real-
Human motion prediction                                                      world scientiﬁc domain. In this benchmark, we evaluate its capacity to
We evaluate DYNAMI-CAL GRAPHNET on a real-world benchmark                    model complex, thermally driven protein dynamics that give rise to
using the CMU Motion Capture dataset39, chosen to assess the model’s         both global and local structural rearrangements. Speciﬁcally, the task
ability to capture articulated, constrained dynamics from real-world         involves predicting protein motion within a thermally ﬂuctuating,
motion data. The dataset records articulated 3D joint trajectories           high-dimensional molecular environment by forecasting the future
during various human activities. Following the data split presented in       positions of heavy atoms from their current conﬁguration.
ref. 19, we use walking sequences from subject #35. Dataset splits and            We use the apo adenylate kinase (AdK) equilibrium trajectory
preprocessing are detailed in Supplementary Information Section §3.1.        dataset40, accessed via the MDAnalysis toolkit47, which tracks the ato-
The task involves predicting the positions and velocities of all joints at   mistic motions of the protein solvated in explicit water and ions. This
a future time step (t + 30), given their current state at time t. This       setup closely mirrors a realistic aqueous cellular environment under
requires the model to reason over both spatial articulation and tem-         near-physiological conditions (300 K, 1 bar), where the protein exhi-
poral dynamics using partial observations (i.e., joint motion only,          bits both global conformational transitions and local side-chain rear-
without external ground reaction forces).                                    rangements, driven by thermal ﬂuctuations and solvent interactions.
       Implementation details of DYNAMI-CAL GRAPHNET are described           The forecasting task—predicting the future positions of heavy atoms—
in Supplementary Information Section §3.2. We further assess a var-          presents a signiﬁcant challenge due to the stochastic nature and
iant, DYNAMI-CAL GRAPHNET (ﬂoor reﬂ.), which augments the skele-             structural ﬂexibility of biomolecules. We choose this dataset for its
ton with ghost foot nodes by reﬂecting the original foot nodes across        biological relevance and physical complexity, and we follow the
the ground plane—deﬁned as the minimum z-coordinate at each frame            experimental setup and data-split protocol introduced21. The task is to
—using the same reﬂection scheme as in the 6-DoF benchmark (Section          predict the protein’s conformation at time t + 15 given the state at time
“Conﬁned 6-DoF granular collisions”). These ghost nodes are con-             t. Further dataset details are provided in Section §2.1 of the Supple-
nected via 1-hop edges to the foot nodes and inherit ground features-        mentary Information.
i.e., zero velocity and a distinct ground label (2), distinguishing them          DYNAMI-CAL GRAPHNET represents the protein as a graph, where
from the foot nodes (labeled 1) and the rest of the joints (labeled 0)-      nodes correspond to backbone heavy atoms and are assigned their 3D
enabling the model to better capture ground contact behavior.                positions, current velocities at time t, and velocities from the previous
       We compare our approach against a comprehensive set of base-          step t − 1, computed via ﬁnite differencing of recorded trajectory
lines evaluated in prior work19 (see Supplementary Information Section       positions during preprocessing. Interactions are modeled with edges
§3.3), including models with handcrafted kinematics (GMN19) and its          that represent covalent bonds between atoms. Importantly, we restrict
learned variant (GMN-L), as well as a range of equivariant graph neural      the edge structure to backbone covalent bonds only, as augmenting it
networks: EGNN17, EGNNReg19, TFN23, SE(3)-Transformer24, Radial Field        based on geometric proximity (e.g., using a 10 Å cutoff) resulted in
Networks (RF)22, ClofNet20, and a message-passing GNN44.                     reduced performance. Implementation details are provided in Sup-
       Figure 6a reports single-step prediction accuracy on the CMU          plementary Information Section §2.2.
human walk benchmark. DYNAMI-CAL GRAPHNET achieves the lowest                     We compare our approach with EGHN21, a U-Net-style architecture
error among all compared methods, outperforming GMN, which                   that captures both local and global molecular interactions while pre-
models the human skeleton using 19 joints and 6 manually speciﬁed            serving geometric equivariance. EGHN integrates message passing
rigid links (e.g., (0,11), (2,3), (7,8)) enforced by a handcrafted forward   with hierarchical pooling and unpooling operations to model ﬁne-
kinematics (FK) module. Notably, the DYNAMI-CAL GRAPHNET (ﬂoor               grained atomic details as well as broader structural patterns. Local
reﬂ.) variant, which augments the skeleton with ghost foot nodes             edges correspond to covalent bonds, while global edges connect
reﬂected across the ground plane, achieves even lower errors across          atoms within a 10 Å distance threshold, enabling the modeling of long-
three random seeds.                                                          range interactions. In addition to EGHN, we report results for several
       Figure 6b demonstrates that, despite training with only single-       baseline models from ref. 21: linear, EGNN17, radial ﬁeld networks (RF)21,
step supervision, DYNAMI-CAL GRAPHNET maintains stable accuracy              and a message passing neural network originally proposed for mole-
during multi-step rollouts, whereas GMN quickly diverges. Qualitative        cular dynamics (MPNN)44.
results in Fig. 6c further conﬁrm that predicted joint trajectories               Figure 7a presents single-step prediction errors (mean squared
remain coherent and physically plausible, closely tracking                   error, MSE). DYNAMI-CAL GRAPHNET achieves the second-best


Nature Communications | (2026)17:1045                                                                                                               11
Article                                                                                                            https://doi.org/10.1038/s41467-025-67802-5




Fig. 6 | Performance on the CMU motion capture benchmark (subject                    sustains low error over time. c Qualitative rollout: despite being trained with single-
#35, walk). a Single-step prediction error: DYNAMI-CAL GRAPHNET achieves the         step supervision, the model produces stable predictions that accurately track the
lowest error among all baselines, with the ﬂoor-reﬂection variant performing best.   ground truth over three rollout steps (90 frames), demonstrating its ability to learn
b Multi-step rollout error: while GMN diverges rapidly, DYNAMI-CAL GRAPHNET          spatiotemporal dynamics effectively.



performance, closely following EGHN, and surpassing EGNN, RF, and                    backbones show that our model’s predictions remain con-
the linear baseline. Although MPNN shows competitive MSE, it lacks                   formationally faithful over time, capturing both large-scale structure
rotational equivariance and is highly sensitive to test-time transfor-               and ﬁne details. Although multi-step rollouts were not part of the
mations: as demonstrated in ref. 21, applying a random rotation during               original EGHN evaluation in ref. 21, we apply them here to both EGHN
evaluation increases its MSE dramatically to 605.7.                                  and DYNAMI-CAL GRAPHNET as a stringent test of physical ﬁdelity-
     Figure 7b evaluates multi-step rollouts—again, for models                       where sustained stability under autoregressive prediction indicates
trained only with single-step supervision. While EGHN diverges                       ﬁdelity of the learned dynamics.
quickly beyond the second step, DYNAMI-CAL GRAPHNET maintains                             These results highlight the exceptional capability of DYNAMI-
accurate predictions for up to 3 steps corresponding to 45 frames of                 CAL GRAPHNET to model physically consistent dynamics in com-
simulated trajectory. This temporal stability is further illustrated in              plex, ﬁne-grained systems such as proteins. This accuracy is rooted
Fig. 7c, where overlays of predicted and ground-truth protein                        in two core principles: (i) modeling internal forces and moments by


Nature Communications | (2026)17:1045                                                                                                                                    12
Article                                                                                                        https://doi.org/10.1038/s41467-025-67802-5




Fig. 7 | Protein molecular dynamics (AdK equilibrium). a Single-step prediction   stable and accurate multi-step rollout predictions. c Qualitative rollout: predicted
error: DYNAMI-CAL GRAPHNET achieves the second-lowest mean squared error          backbone structures from DYNAMI-CAL GRAPHNET are shown starting from an
(MSE), closely trailing EGHN and outperforming MPNN, EGNN, RF, and a linear       initial protein conﬁguration and compared to ground-truth conformations at each
baseline. One prediction step corresponds to 15 trajectory frames. b Multi-step   rollout step (up to 3 steps or 45 simulation frames). The model closely tracks
rollout error: while EGHN diverges rapidly, DYNAMI-CAL GRAPHNET maintains         conformational evolution despite training with only single-step supervision.



enforcing conservation of linear and angular momentum, and (ii)                   Computational complexity
capturing the spatiotemporal evolution of edge embeddings—                        We analyze DYNAMI-CAL GRAPHNET, a standard message-passing
enriched with spatial context from neighboring nodes and propa-                   GNN (GNS) adapted for 6-DoF dynamics, and an equivariant
gated through iterative message passing that emulates sub-time                    architecture (EGNN) similarly extended for 6-DoF motion
stepping.                                                                         prediction. For N nodes, E edges, latent width L, and M


Nature Communications | (2026)17:1045                                                                                                                              13
Article                                                                                                      https://doi.org/10.1038/s41467-025-67802-5




Fig. 8 | Computational complexity analysis. a Training-time comparison per-      (0.735M vs. 0.880M and 1.572M). b Inference-time comparison rollout wall-clock
epoch wall time (single CPU core, single thread) for DYNAMI-CAL GRAPHNET, GNS,   time vs. rollout steps for the 6-DoF DEM and one-step inference on human motion.
and EGNN on 6-DoF DEM and human motion. Despite its physically grounded          DYNAMI-CAL GRAPHNET achieves inference times close to GNS and consistently
6-DoF decoding, DYNAMI-CAL GRAPHNET maintains training times within 2× of        faster than EGNN, conﬁrming that its physically enriched computations do not
GNS and comparable to EGNN, consistent with its compact parameter count          incur runtime penalties.



message-passing steps, all three exhibit identical leading-order               consistently faster than EGNN across rollout steps. Together, the
scaling-time Θ MðE + NÞL2 and memory Θ ððE + NÞLÞ-but differ in                  MAC analysis and runtime measurements indicate that DYNAMI-
constant factors and parameter growth. As detailed in Supple-                    CAL offsets its higher per-step arithmetic through parameter
mentary Information Section §6, the per-step multiply-accumu-                    sharing during inference, achieving efﬁcient runtime and a strong
late counts (MACs) are approximately (15E + 6N)L2 for DYNAMI-                    parameter economy.
CAL GRAPHNET, (5E + 4N)L2 for GNS, and (10E + 7N)L2 for EGNN.
Although it performs denser physical computations, DYNAMI-CAL                    Discussion
GRAPHNET employs an initialization block for the ﬁrst step and                   In this work, we propose DYNAMI-CAL GRAPHNET, a physics-informed
reuses a shared process block for all subsequent steps. At each                  graph neural network to model six-degree-of-freedom dynamics in
step, only a LayerNorm is applied to the latent edge memory,                     diverse multi-body systems. The architecture embeds conservation of
where the previous embedding is added to the current one before                  linear and angular momentum as an inductive bias, enabling the model
normalization. This design yields parameter scaling of O(L2)                     to learn physically consistent interactions, including those involving
(independent of M), in contrast to O(ML2) for GNS and EGNN.                      external forces and dissipation, directly from data. By employing edge-
     Empirically, Fig. 8a shows higher training times for DYNAMI-                local reference frames that are equivariant to rotations, invariant to
CAL GRAPHNET, consistent with its larger MAC count, whereas                      translations, and antisymmetric under node interchange, the model
Fig. 8b reports inference times that are close to GNS and                        captures both geometric symmetries and fundamental conservation


Nature Communications | (2026)17:1045                                                                                                                         14
Article                                                                                                https://doi.org/10.1038/s41467-025-67802-5


laws. Interactions are aggregated and integrated through a multi-step     Methods
message-passing scheme that mimics sub-time stepping, allowing for        Graph representation of multi-body dynamical systems
stable long-horizon predictions and interpretable dynamics. Although      We represent a multi-body dynamical system as a graph G = (V, E),
its denser per-edge physical computations increase training cost          where V = {vi∣i = 1, 2, …, n} denotes the set of bodies, and
compared to simpler GNN baselines, the shared process block ensures       E = {(eij, eji)∣i ≠ j, (i, j) ∈ V × V} represents bidirectional edges that encode
that inference remains efﬁcient and suitable for large-scale rollout      interactions between distinct bodies, excluding self-loops. Edge con-
prediction. As a result, DYNAMI-CAL GRAPHNET provides a general,          nectivity is determined either from the system’s known geometry or
scalable, and physically principled framework for learning dynamics in    dynamically computed using metrics such as Euclidean distance. For
systems ranging from granular materials and molecular assemblies to       example, in the granular collision dynamics examined in Section
human motion.                                                             “Conﬁned 6-DoF granular collisions”, edges are formed between
     We demonstrated the versatility of DYNAMI-CAL GRAPHNET               bodies i and j if k r               ~j k ≤ d c , where dc is the threshold distance
                                                                                                         ~i  r
across four diverse benchmarks, encompassing both simulated and           criterion (taken as 1.25 × sphere diameter in the granular system).
real-world systems. On the 6-DoF granular benchmark introduced in               Each node vi in the graph is assigned two types of node features in
this work, characterized by dissipation and external forcing, the         addition              to           the        position       vectors:     (1)Vector
model learned physically consistent dynamics from just ﬁve training       features:Vi = ½v   ~ti , ω
                                                                                                   ~ ti , ~t1
                                                                                                          vi , ~
                                                                                                                 t1
                                                                                                               ωi , which include the linear velocity ~
                                                                                                                                                         t
                                                                                                                                                       vi and
trajectories involving 60 spheres and successfully extrapolated to a                             t
                                                                                             ~ i at the current time step t, as well as their values
                                                                          angular velocities ω
rotating hopper with over 2000 particles, maintaining stable rollouts      t1     t1
over 16,000 time steps. In the constrained N-body benchmark,              vi and ~
                                                                          ~      ωi at the previous time step t − 1. (2)scalar features αi, which
DYNAMI-CAL GRAPHNET outperformed baselines by learning holo-              represent categorical or continuous attributes that encode node-
nomic constraints and enabling stable multi-step rollouts, even on        speciﬁc properties. For example, in the 6-DoF granular system 2.2,
unseen conﬁgurations. Applied to real-world human motion capture,         scalar labels distinguish original nodes (αi = 0) from ghost boundary-
the model inferred joint dynamics directly from data and produced         reﬂection nodes (αi = 1). In contrast, for the constrained N-body dataset
stable multi-step predictions, surpassing constraint-aware baselines.     2.5, the scalar feature corresponds to each particle’s electric
In the protein dynamics benchmark, it captured both ﬁne-scale             charge (αi = Zi).
ﬂuctuations and large-scale conformational changes over extended               Each edge ij is characterized by the edge distance vector between
horizons, outperforming a hierarchical baseline speciﬁcally designed      the connected nodes: dx   ~ = ð~rj  ~
                                                                                                               ri Þ, where ~
                                                                                                                           rj and ~ri represent the
                                                                                                       ij
for this task. Collectively, these results highlight the robustness,      position vectors of the receiver node j and the sender node i, respec-
generalization ability, and physical ﬁdelity of DYNAMI-CAL GRAPH-         tively. Additionally, edge features can include scalar labels that encode
NET across a broad range of dynamical systems. These strengths            different types of interactions, such as collision forces, joint con-
also position DYNAMI-CAL GRAPHNET as an efﬁcient surrogate in             straints, or electromagnetic inﬂuences, allowing the model to distin-
scenarios where traditional simulation pipelines are challenged by        guish and appropriately process the diverse interaction mechanisms
frequent reconﬁguration or limited knowledge of underlying                present within the multi-body system.
dynamics.                                                                      The graph representation, constructed from the system’s physical
     At its core, DYNAMI-CAL GRAPHNET internalizes fundamental            properties, serves as the input data for DYNAMI-CAL GRAPHNET. At
conservation laws by treating each interaction as an instantaneous        each time step t, the model processes the graph and predicts changes
closed system, thereby guaranteeing the preservation of linear and        in the state of each node, speciﬁcally the changes in velocity δv,          ~
angular momentum as well as translational and rotational symmetries-                         ~                         ~
                                                                          angular velocity δω, and position vector δr. The training data consists
directly reﬂecting Noether’s theorem and Newton’s third law.              of input-output pairs derived from observed trajectories. Each pair
     A natural extension of this framework is to continuum mechanics,     comprises the graph representation at time t and the corresponding
where conservation laws are equally essential. For example, in ﬁnite      changes in state from t to t + 1. When the positions and angular velo-
element analysis (FEA), the Cauchy stress relation enforces linear        cities of the system’s components are observed, velocities and angular
momentum balance, and the symmetry of the stress tensor ensures           velocities are computed using ﬁnite differences. Alternatively, directly
angular momentum conservation; in ﬂuid dynamics, the Navier-Stokes        measured linear velocity and angular velocity vectors can be used if
                                                                                                                                    t
equations encode momentum conservation. Adapting DYNAMI-CAL               available. The vector features of each node, Vi = ½~         ~ ti , ~
                                                                                                                                  vi , ω
                                                                                                                                               t1
                                                                                                                                                   ~ t1
                                                                                                                                              vi , ω i ,
GRAPHNET to such domains introduces new challenges, especially in                                                    ~
                                                                          along with the edge distance vector dxij , are normalized by scaling
enforcing local conservation laws across ﬁelds with effectively inﬁnite   them by their respective maximum magnitudes.
degrees of freedom.
     An additional challenge arises in partially observed or unclosed     DYNAMI-CAL GRAPHNET architecture
systems, such as when external agents, like unmodeled magnetic            The DYNAMI-CAL GraphNet model leverages observed trajectories as
ﬁelds, act on otherwise closed dynamics, where strict momentum            training data to learn the system’s implicit edge-wise interaction
conservation is no longer observed. Addressing these cases requires       dynamics. By integrating inductive biases that enforce the conserva-
explicit modeling of external forces at the node level, conditioned on    tion of linear and angular momentum, the model ensures that the
latent state embeddings. In our experiments, this was straightfor-        learned dynamics are physically consistent. The model follows the
ward for gravitational forces, which depend only on scalar node           scalarization-vectorization paradigm, as illustrated in Fig. 1.
embeddings (representing masses) and can be decoded indepen-                   The scalarization step computes edge embeddings from node and
dently at each node. However, modeling more complex, context-             edge features, and the vectorization step decodes these embeddings
dependent external forces remains an open direction for future            into edge-wise forces and moments. These are then aggregated at each
research.                                                                 node to update its state, and the updated graph undergoes repeated
     By anchoring learned dynamics in fundamental physical princi-        scalarization-vectorization passes through multiple message-passing
ples while retaining the ﬂexibility to accommodate real-world com-        iterations, effectively emulating sub-time-step state updates within a
plexities, DYNAMI-CAL GRAPHNET provides a strong foundation for           single time step. The model predicts node-wise changes in linear
advancing data-driven modeling of physical systems. Extending this        velocity (Δ~                         ~ i ), and displacement (Δ~
                                                                                      vi ), angular velocity (Δω                         xi ). Training
approach to continuum domains and partially observed environments         is performed by minimizing the mean squared error between pre-
opens exciting avenues for future research at the intersection of         dicted and ground-truth values, with gradients backpropagated to
machine learning, physics, and engineering.                               learn the system dynamics.


Nature Communications | (2026)17:1045                                                                                                                     15
Article                                                                                                            https://doi.org/10.1038/s41467-025-67802-5




Fig. 9 | Edge local reference frame calculation for bi-directional edges. The reference frames are antisymmetric under node interchange.




Scalarization. In this step, we transform the vector and scalar                                    ~ij is antisymmetric under node interchange, meaning
                                                                                       This vector a
properties of nodes and edges into high-dimensional scalar                        that swapping nodes i and j reverses its direction. Additionally, a   ~ij is both
embeddings. These embeddings serve as a comprehensive repre-                      rotation equivariant and translation invariant. The challenge lies in
sentation of the interactions for each edge, capturing the essential              constructing the remaining basis vectors, which must form an ortho-
features necessary for the model to understand the system’s                       gonal set with a~ij while preserving antisymmetry under node inter-
dynamics.                                                                         change. A straightforward approach, such as computing ~          xi × ~
                                                                                                                                                        xj , initially
                                                                                  produces an antisymmetric vector due to the anti-commutative nature
Edge local reference frame calculation. Constructing an edge local                of the cross product. However, deriving a third basis vector through
reference frame—antisymmetric under the interchange of nodes—is                   another cross product ð~       xj Þ × ~
                                                                                                            xi × ~      aij , results in an unsuitable symmetric
crucial for enforcing conservation laws within our model. This process            vector. To address this, we introduce an intermediate vector based on
is illustrated in Fig. 9. For each edge eij, we begin by deﬁning the ﬁrst         the state vectors of the nodes connected by the edge ij:
basis vector ~ aij as the unit vector along the distance vector between
nodes i and j:                                                                                0       ~    ~i
                                                                                                      vj + v      ~j + ~
                                                                                                                  ω    ωi     ð~            ~j  ~
                                                                                                                                    ~i Þ × ðr
                                                                                                                               vj  v            ri Þ
                                                                                             ~
                                                                                             bij =             +           +
                                                                                                     k~    ~i k k ω
                                                                                                      vj + v      ~j + ~
                                                                                                                       ωi k k ð~    vi Þ × ð~
                                                                                                                               vj  ~       rj  ~
                                                                                                                                                 ri Þ k
                                        rj  ~
                                        ~    ri                                                                  ~ i Þ × ð~
                              ~ij =
                              a                   ,                                                         ~j  ω
                                                                                                           ðω             rj  ~
                                                                                                                               ri Þ
                                      k rj  ~
                                        ~    ri k                                                    +                                :
                                                                                                                 ωi Þ × ð~
                                                                                                            ~j  ~
                                                                                                         k ðω             rj  ~
                                                                                                                               ri Þ k

 where ~rj and ~
               ri are unscaled position vectors of the receiver and               This intermediate vector is both antisymmetric to node interchange,
                                                                                                                                                     0
sender nodes.                                                                     translation invariant and rotation equivariant. We then decompose ~
                                                                                                                                                    bij



Nature Communications | (2026)17:1045                                                                                                                              16
Article                                                                                                               https://doi.org/10.1038/s41467-025-67802-5

                                              ~ij :
into components parallel and perpendicular to a                                               This ensures that node i’s vectors are always aligned with the
                                                                                        reference frame of edge ij, and node j’s vectors are aligned with edge ji,
                                                   0           1
                          0                   0     ~      ~0
                                                     aij  b                            regardless of the edge direction. By maintaining this structure, we
                         ~                   ~
                         bijk~aij = proj~aij bij = @
                                                            ij Aa
                                                                ~ij ,                   achieve node interchange invariant scalar features for constructing
                                                     ka~ij k2
                                                                                        interaction embeddings for edges that remain invariant to node
                                                                                        interchange.
                                   0          0     0                                         Furthermore, the projected scalars—and thus the resulting inter-
                                  bij?~aij = ~
                                  ~          bij  ~
                                                   bijk~aij
                                                                                        action embeddings—inherit additional symmetries based on the
                                                                                        properties of the edge reference frame. Speciﬁcally, if the reference
 Both components are antisymmetric to node interchange. Using the                       frame is rotation equivariant, the projected scalars remain rotation
perpendicular component, we deﬁne the second basis vector:                              invariant. This is because the relative alignment between vectors and
                                                                                        the basis vectors stays consistent under rotation. Similarly, since the
                                             0
                                            ~
                                            bij?~aij × ~
                                                       aij                              state vectors (e.g., velocity and angular velocity) are translation
                                  ~
                                  bij =                                                 invariant, the projected scalars inherit translation invariance provided
                                              0
                                          k~
                                           bij?~aij × ~
                                                      aij k                             the reference frame is translation invariant.
                                                                                              These projected scalars for both sender and receiver nodes are
 This cross product yields an antisymmetric vector by combining a                       transformed into higher-dimensional embeddings, denoted as ϵijsender
symmetric vector with an antisymmetric one. Finally, the third basis                    and ϵijreceiver , using the function ϕe1 , which is implemented as a multi-
vector is computed as:                                                                  layer                                                              perceptron
                                                                                        (MLP): ϵijsender = ϕe1 projframe Vi , ϵijreceiver = ϕe1 projframe Vj .
                                             0
                                            bijk~aij × ~
                                            ~          bij                                    Additionally, we create another invariant embedding from the
                                  ~
                                  cij =     0                                           magnitude of the edge distance vector Δ~           xij = ~
                                                                                                                                                 rj  ~
                                                                                                                                                      ri using another MLP
                                          k~
                                           bijk~aij × ~
                                                      bij k                             ϕe2 : ϵedge  = ϕ     ðjjΔ~x    jjÞ.
                                                                                               ij         e2        ij
                                                                                              The node scalar features αi for each node vi ∈ G(V, E) are encoded
           0
       ~                       ~                          ~                             using the MLP function ϕn: hi = ϕn(αi). For an edge ij, the node
 Since b    aij is parallel to aij , the resulting vector cij is orthogonal to
         ijk~
                                                                                        embeddings hi and hj correspond to nodes i and j, respectively.
both ~         ~ , while also maintaining antisymmetry under node
       aij and b ij                                                                           The edge embeddings—ϵedge       ij   , ϵsender
                                                                                                                                       ij    , ϵreceiver
                                                                                                                                                 ij      —along with node
interchange.                                                                            embeddings hj, hj, are then processed in the subsequent step to create
      By constructing this orthogonal basis set ~   aij , ~
                                                          bij , ~
                                                                cij —antisym-           the ﬁnal comprehensive and expressive edge interaction embedding.
metric under node interchange, translation invariant, and rotation
equivariant—the model ensures symmetrical interactions, which are                       Intuition for invariance of edge embedding under node inter-
vital for enforcing the conservation of linear and angular momentum.                    change. Invariance of edge embedding under node interchange is
                                                                                        crucial for physically consistent modeling, especially in systems where
Degeneracy of the Local Reference Frame. The local reference frame                      interactions depend on the relative positions or states of nodes, such
becomes degenerate under two speciﬁc conditions: (1) When the                           as forces in a spring or other pairwise interactions. For instance, con-
                          0
intermediate vector ~    bij = 0: In this scenario, the edge system is sta-             sider a spring connecting two nodes i and j with positions r~i and ~
                                                                                                                                                           rj . The
tionary, exhibiting no linear or angular velocities. The interaction can                stretch or compression of the spring depends solely on the relative
be fully captured using only the ﬁrst basis vector a    ~ij along the edge. (2)         distance k ~rj  ~
                                                                                                         ri k, which remains unchanged regardless of the order
         0
When ~                    ~ij : This indicates that the velocities and angular
       bij is parallel to a                                                             in which the nodes are considered. Therefore, to accurately model
velocities are aligned with the edge vector, implying that the interac-                 such physical interactions, the embeddings derived from the node
tion is constrained along the edge direction. Consequently, a         ~ij sufﬁ-         features must be invariant. In our context, this means that the inter-
ciently represents the interaction. In both cases, the system remains                   action embeddings for edges ij and ji are identical, ensuring con-
effectively non-degenerate for representing the relevant interactions,                  sistency and physical accuracy in the model’s predictions.
ensuring robust and accurate modeling of the multi-body system’s
dynamics.                                                                               Final edge interaction embedding. In this step, the edge embeddings
                                                                                        are ﬁrst combined and then transformed using a function θe to pro-
Scalar edge embedding from projections onto edge-                                       duce an invariant edge embedding:
reference frames. After establishing the edge-wise local reference
frames, the vector features of the connected nodes are projected onto                                                                                        
these reference frames. Speciﬁcally, for an edge ij, the sender node’s                                   ϵij = θe ϵijedge + ϵijsender + ϵijreceiver + hi + hj
                                               t
vector features are deﬁned as: Vi = ½~            ~ ti , ~
                                             vi , ω
                                                          t1
                                                              ~ t1
                                                         vi , ω i . These features
are projected onto the basis vectors of the edge’s reference frame                      Incorporating interaction history in edge embedding via skip con-
aij , ~
~     bij , ~
            cij . Conversely, for the receiver node j, its vector features Vj are       nection. The interaction embedding at the message-passing step L,
                                                                                        denoted ϵLij , is combined with the previous layer’s edge embedding ϵL1
                                                                                                                                                             ij
projected onto the antisymmetric reference frame, speciﬁcally
                                                                                        through a skip connection. The resulting sum is passed through a layer
 ~ij ,  ~
a       bij ,  ~
                 cij . This projection strategy ensures that the scalar pro-            normalization operation (θLN) to yield the updated interaction embed-
jections for the sender (i) and receiver (j) nodes remain invariant when                ding ϵ0ij . This recursive dependence on prior edge embeddings enables
the nodes are swapped. To illustrate, consider the reverse direction                    the model to capture temporal dynamics unfolding across multiple
edge ji:                                                                                steps. At each message-passing step, the raw interaction embedding ϵLij is
   • The sender node j’s features a   ~ji , ~    ~ji are projected onto the
                                            bji , c                                     enriched with spatial context through distance-based features derived
     reference frame of edge ji, which corresponds to ~        aij ,  ~
                                                                        bij ,  ~
                                                                                cij .   from neighboring node conﬁgurations, while the skip connection inte-
   • The receiver node i’s features are then projected onto the anti-                   grates temporal memory from previous interactions. Together, these
     symmetric reference frame ~   aji , b~ , ~  cji , which is equivalent to        mechanisms allow the model to learn localized physical interactions
                                              ji
     ~ij , ~
     a           ~ij .
           bij , c
                                                                                        across both space and time, leading to robust and generalizable pre-
                                                                                        dictions in a wide range of structured dynamical systems.


Nature Communications | (2026)17:1045                                                                                                                                  17
Article                                                                                                               https://doi.org/10.1038/s41467-025-67802-5


Vectorization. In the vectorization step, the invariant edge interaction                relation to substitute internal force for the change in linear momen-
embedding ϵ0ij is decoded using multiple MLP functions to extract the                   tum in unit time ð~Fij Δt ¼ mi ð~
                                                                                                                         tþΔt
                                                                                                                        vi    ~
                                                                                                                                t
                                                                                                                               vi ÞÞ, is:
internal forces ~                           ~ij vectors, while ensuring the
                 Fij and rotational torques τ
conservation of both linear and angular momentum. These vectors are                                                        
                                                                                                                        ωi = ~                    ~ ,
                                                                                                                t + Δt
then aggregated for each node to account for the cumulative effects of                                     Ii ~ωi      ~
                                                                                                                          t
                                                                                                                                     ~it  ~
                                                                                                                              Aij  ðr     r0 Þ × F ij                   ð5Þ
all interactions.
      Additionally, this step involves estimating the inverse mass m1 and
                                                                      i                  and a similar expression holds for the rotational torque acting on body
inverse moment of inertia I1 for each node from their scalar embed-
                               i                                                        j. For the full derivation, refer to Supplementary Information Sec-
dings. These values are subsequently used to compute the change in
                                                                                        tion §5.4.
linear velocity Δ~ vi and angular velocity Δ ~
                                            ωi , enabling accurate updates
                                                                                             We now show how conservation of angular momentum in
to the system’s dynamics.
                                                                                        DYNAMI-CAL GraphNet is incorporated. For any edge ij connecting
      If external forces are present, the changes in velocity and angular
                                                                                        nodes i (sender) and j (receiver), the internal angular interaction vector
velocity are decoded directly from the node scalar embeddings hi for                    ~
                                                                                        Aij is decoded from the edge interaction embedding ϵ0ij using the
each node vi. This allows the model to incorporate both internal
                                                                                        function ψea . The invariant edge interaction embedding ϵ0ij is trans-
interactions and external inﬂuences in a physically consistent manner.
                                                                                        formed into scalar coefﬁcients, which are then used to modulate the
                                                                                        basis vectors of the localreference frame a  ~ij , ~
                                                                                                                                           bij , ~
                                                                                                                                                 cij :
Internal force vectors. The invariant edge interaction embedding ϵ0ij is
decoded into invariant coefﬁcients which modulate the reference
                    aij , ~
frame basis vectors ~           cij . This results in the internal forces ~
                          bij , ~                                         Fij as                  ~
                                                                                                  Aij = ψea ðϵ0ij Þ ½0  a                      ~ + ψ ðϵ0 Þ ½2  ~
                                                                                                                          ~ij + ψe ðϵ0ij Þ ½1  b                 cij
                                                                                                                                  a                ij ea ij
follows:
                                                                                        Since the basis vectors are antisymmetric under node interchange, the
                             ~ij + ψe ðϵ0ij Þ ½1  ~
           ~ = ψ ðϵ0 Þ ½0  a
           F                                        bij + ψef ðϵ0ij Þ ½2  ~
                                                                            cij         decoded interaction vector ~ Aij also preserves this antisymmetry:
             ij ef ij                f



 Here, ψef ðϵ0ij Þ provides the scalar coefﬁcients for the basis vectors. By                                                Aij =  ~
                                                                                                                            ~       Aji ,                                ð6Þ
construction, the forces are anti-symmetric, ensuring the conservation
of linear momentum.
                                                                                          thereby ensuring angular momentum conservation as stated in
                                                                                        Equation (3).
                                      ~ = ~
                                      Fij  Fji                                               To isolate the spin component from ~    Aij , we compute the edge
                                                                                        reference point. This point, denoted ~
                                                                                                                             r0ij , is computed as a weighted
 This anti-symmetry guarantees that the internal forces between any                     sum of the position vectors of nodes i and j. The weights are derived
two connected nodes i and j are equal in magnitude and opposite in                      from the node scalar embeddings hi and hj using the function ψn1.
direction, maintaining the physical principle of linear momentum
conservation within the system.
                                                                                                                         ψn1 ðhi Þ  ~
                                                                                                                                     ri + ψn1 ðhj Þ  ~
                                                                                                                                                      rj
                                                                                                                ~
                                                                                                                r0ij =
Rotational torque vectors: isolated edge dynamical system. Rota-                                                            ψn1 ðhi Þ + ψn1 ðhj Þ
tional torque is decoded by enforcing angular momentum conserva-
tion at each edge, modeled as an isolated dynamical system. We begin                    This formulation ensures that the reference point remains consistent
by demonstrating conservation about a reference point ~      r0 for two                 under node interchange for bi-directional edges:
interacting bodies i and j, each with velocity ~                     ~i ,
                                               vi , angular velocity ω
mass mi, and moment of inertia Ii.                                                                                            ~0 = ~
                                                                                                                              r    r0ji :
                                                                                                                                ij
     The angular momentum of body i about a reference point ~         r0
comprises both spin and orbital components:                                                    ~ ,~          ~
                                                                                         After F ij Aij , and r0ij are decoded for edge ij, the rotational torque on
                                                                                      the receiver node j is computed as:
                          Lti = I i ~
                                      t
                                         ~it  ~
                                    ωi + r     r0 × mi~
                                                        t
                                                      vi :                        ð1Þ
                                                                                                                      ~j = ~
                                                                                                               I j  Δω    Aij  ð~    r0ij Þ × ~
                                                                                                                                  rj  ~        Fij λij :
    In a closed two-body system, total angular momentum is then
given by:
                                                                                         Here, λij = ψel ðϵ0ij Þ represents a scalar decoded from the edge interac-
                                      Lt = Lti + Ltj                              ð2Þ   tion embedding, introduced to enhance stability by mitigating the
                                                                                        inﬂuence of negligible noisy edge forces on the calculation of rota-
                                                                                        tional torque. This approach ensures that the predicted rotational
Conservation of angular momentum implies:
                                                                                        torques between nodes are physically consistent (Physics derivation
                                                                                        shown in Equation (5)), thereby upholding the conservation of angular
                                      Lt + δt = Lt ,                              ð3Þ
                                                                                        momentum throughout the system.
which in turn implies that the total angular momentum transfer from
                                                                                        Aggregation of edge forces and moments on the nodes. The
body i to j, denoted ~Aij , is equal and opposite to that from j to i,
                                                                                        decoded forces and moments from each edge are aggregated on the
        Aji . The expression for ~
denoted ~                         Aij is:
                                                                                        receiver nodes. These aggregated internal forces and moments are
                                                                                    then used to determine the changes in linear and angular velocities for
           ~
           Aij = I i ~
                       t + Δt
                      ωi
                                 t
                               ωi + ð~
                              ~
                                       t
                                     ri  ~          ~ti + Δt  ~
                                          r0 Þ × m i v
                                                                  t
                                                                vi ,              ð4Þ   each node.

    The rotational torque that contributes to updating the spin
                                                                                        Decoding change in velocity and angular velocity for each node.
angular velocity of body i is computed by decoupling the spin com-
                                                                                        Using the functions ψn2 and ψn3, the inverse mass m1 and inverse
ponent from ~ A . The resulting expression for the change in spin
                   ij
                                                                                                                                                i
                                                                                        moment of inertia I1 are decoded from each node’s scalar embeddings
                                                                                                            i
angular momentum of body i, considering impulse-momentum                                hi. These decoded values are utilized to compute the changes in linear


Nature Communications | (2026)17:1045                                                                                                                                    18
Article                                                                                                           https://doi.org/10.1038/s41467-025-67802-5




Fig. 10 | Mesh and particle-free modeling of wall boundaries. a Illustrates the       form an edge with the reﬂected sphere. b Demonstrates the contact modeling for a
normal contact and collision modeling of a granular sphere with the walls of a box    cylindrical boundary, which is parameterized by its diameter, height, and axis
boundary. The sphere is reﬂected along the outward normal vector of each wall. Of     vector. The spheres are reﬂected off both the curved surface and the planar end
the six possible reﬂections, only those that satisfy a predeﬁned threshold distance   caps, effectively handling interactions with the cylindrical geometry.



          ~i and angular velocity Δ ~
velocity Δv                        ωi for each node:                                  resulting in the updated angular velocity:
                                    X                           X
                Δ~
                 vi = ψn2 ðhi Þ        ~ , Δ~
                                        F    ωi = ψn3 ðhi Þ        ~
                                                                    Mij
                                                                                                                      new
                                                                                                                    ωi = ~
                                                                                                                    ~     ωi + Δ ~
                                                                                                                                ωi
                                                                                                                                   net
                                         ij

      P~
where F    ij represents the total internal force acting on node i from all            Subsequently, the position of each node is updated using the com-
                      P~
connected edges and       Mij represents the total internal torque acting             puted velocities through single-step Euler integration, utilizing the
on node i. These updates ensure accurate changes in the system’s                      same time step Δt as used during forward differencing:
dynamics based on both internal interactions and node-speciﬁc
                                                                                                                                   new
properties.                                                                                                                 ~i + ~
                                                                                                                           ðv    vi Þ
                                                                                                                   Δ~
                                                                                                                    xi =               Δt
                                                                                                                                2
Decoding external forces. Additionally, when external forces are
present, the change in velocity due to external inﬂuences is decoded                  Thus, the new position of node i is:
using the function ψn4:
                                                                                                                       new
                                                                                                                     xi = ~
                                                                                                                     ~     xi + Δ~
                                                                                                                                 xi
                                       ext
                                    Δ~
                                     vi = ψn4 ðhi Þ
                                                                                       The updated system state is then fed back into the model pipeline,
Updating the graph state. Finally, the net change in both linear and                  starting from the encode step, allowing for iterative updates. This
angular velocities, resulting from internal and external forces, is                   iterative process ensures that both velocities and positions are adjus-
computed and applied to update the node states. This step is crucial                  ted based on the cumulative effects of internal and external forces. By
for advancing the system dynamics forward in time. The net change in                  incorporating forward integration bias, the method achieves physi-
linear velocity is computed as:                                                       cally consistent multi-step updates, enabling precise and interpretable
                                                                                      modeling of the evolving system dynamics over time.
                                ~inet = Δ~
                               Δv        vi + Δ~
                                               vi
                                                 ext


                                                                                      Mesh-particle free modeling of boundaries through reﬂections.
Thus, the updated velocity of node i is:                                              Many dynamical systems involve interactions between their compo-
                                                                                      nents and boundaries. Such systems are prevalent in various domains,
                                  new         net
                                vi = ~
                                ~     vi + Δ~
                                            vi                                        including granular systems (as demonstrated in Section “Results”),
                                                                                      rigid body dynamics (e.g., spheres rolling on a surface), and muscu-
Similarly, the net change in angular velocity is given by:                            loskeletal systems. In this work, we introduce a mesh-free and particle-
                                                                                      free approach for modeling boundary interactions in multi-body
                                         net
                                     Δ~
                                      ωi       = Δ~
                                                  ωi                                  dynamical systems. Unlike prior methods that rely on explicit meshes
                                                                                      or densely sampled particles to represent walls and enclosures17,36–38,


Nature Communications | (2026)17:1045                                                                                                                              19
Article                                                                                                  https://doi.org/10.1038/s41467-025-67802-5


DYNAMI-CAL GRAPHNET models each boundary as a reﬂective surface                information; thus, no human-subject approval was required. We
deﬁned by its outward normal. Physical components are mirrored                 advocate for the responsible use of this technology, especially in real-
across this normal to generate ghost nodes that encode boundary                world or safety-critical applications, where rigorous validation and
effects (Fig. 10). Crucially, these boundary interactions are integrated       domain-speciﬁc safeguards are essential.
into the message-passing framework in the same manner as body-body
interactions, leveraging consistent geometric structures without               Data availability
requiring task-speciﬁc modules or special heuristics.                          The dataset for the 6-DoF granular collision benchmark introduced in
     The reﬂection process leverages the outward normal to the                 this study is publicly available at https://zenodo.org/records/17589419.
boundary. For a body at position ~    r, the reﬂected position ~
                                                               rreflected is   All simulations were performed using the MFIX-DEM framework,
computed using the outward normal vector n      ~ as follows:                  available at https://mﬁx.netl.doe.gov/products/mﬁx/. Detailed simu-
                                                                               lation parameters are provided in the Supplementary Information and
                         rreflected = ~
                         ~                  ~ ~
                                      r  2ðr    ~:
                                               nÞn                             included in the Zenodo repository. The constrained N-body and
                                                                               human motion benchmarks were obtained from the GMN repository19
where n  ~ is the outward normal vector to the boundary. For planar            (https://github.com/hanjq17/GMN), and the protein dynamics bench-
boundaries such as ﬂoors or walls, this results in one reﬂected node           mark was obtained from the EGHN repository21 (https://github.com/
per physical node per wall. For example, a single ﬂoor yields Nn ghost         hanjq17/EGHN).
nodes, while a cuboidal enclosure with six walls produces 6Nn ghost
nodes. In the case of a curved cylindrical boundary, the normal n   ~ is       Code availability
computed by normalizing the vector from the cylinder’s axis to the             The source code used in this study is hosted in a license-protected
particle’s position, ensuring radially outward reﬂection. If the cylin-        repository on the Open Science Framework (OSF). Access is granted
der is capped, two additional planar reﬂections are applied, resulting         for non-commercial academic use upon completion of a license
in a total of 3Nn ghost nodes. As previously discussed, this introduces        agreement form, available at https://osf.io/ywdu5/overview?view_
at most (W − 1) × Nn additional nodes for W boundaries, a ﬁxed and             only=3589826a94e040bd95ed36636c43fa1b. Researchers who sub-
tractable overhead that remains signiﬁcantly lower than particle- or           mit the form will be automatically granted access to the repository
mesh-based representations37, and does not scale with bound-                   containing the implementation and training scripts.
ary size.
      The reﬂected body inherits wall-speciﬁc features, including velo-        References
city and angular velocity vectors, as well as one-hot encoded labels           1.  Jimenez, J. J. M., Schwartz, S., Vingerhoeds, R., Grabot, B. & Salaün,
indicating blocked degrees of freedom. For stationary walls, both                  M. Towards multi-model approaches to predictive maintenance: a
velocity and angular velocity vectors are set to zero, ensuring an                 systematic literature survey on diagnostics and prognostics. J.
accurate physical representation of boundary constraints. For moving               Manuf. Syst. 56, 539–557 (2020).
boundaries, the reﬂected (ghost) nodes inherit motion characteristics          2. An, D., Kim, N. H. & Choi, J.-H. Practical options for selecting data-
from the wall, including translational velocity, tangential velocity               driven or physics-based prognostics algorithms with reviews.
induced by rotation, and angular velocity-depending on the wall’s                  Reliab. Eng. Syst. Saf. 133, 223–236 (2015).
dynamic state. These quantities are computed directly from the wall’s          3. Zhang, W., Yang, D. & Wang, H. Data-driven methods for predictive
geometry and motion. For instance, in the case of a rotating cylindrical           maintenance of industrial equipment: a survey. IEEE Syst. J. 13,
boundary (Fig. 10), the tangential velocity of each reﬂected node is               2213–2227 (2019).
calculated using the wall’s angular velocity vector ~ω and the reﬂected        4. Brunton, S. L. & Kutz, J. N. Data-driven Science and Engineering:
node’s position relative to the cylinder’s axis of rotation, denoted               Machine Learning, Dynamical Systems, and Control (Cambridge
~
dreflected, i . The tangential velocity is given by:                               University Press, 2022).
                                                                               5. Choi, H.-S. et al. Data-driven simulation for general-purpose multi-
                        vtangential = ~
                        ~                             ~:
                                      dreflected, i × ω                            body dynamics using deep neural networks. Multibody Syst. Dyn.
                                                                                   51, 419–454 (2021).
     This formulation ensures that ghost nodes inherit the correct             6. Karniadakis, G. E. et al. Physics-informed machine learning. Nat.
dynamic boundary conditions, allowing the model to capture the                     Rev. Phys. 3, 422–440 (2021).
inﬂuence of both translational and rotational wall motion in a physi-          7. Raissi, M., Perdikaris, P. & Karniadakis, G. E. Physics-informed neural
cally consistent and uniﬁed manner.                                                networks: a deep learning framework for solving forward and
     The system of physical spheres and their reﬂections is repre-                 inverse problems involving nonlinear partial differential equations.
sented as a graph, where nodes correspond to both real bodies and                  J. Comput. Phys. 378, 686–707 (2019).
their ghost counterparts. Edges are dynamically constructed at each            8. Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R. & Mahoney, M. W.
time step between a physical node and its reﬂected ghost node if their             Characterizing possible failure modes in physics-informed neural
separation distance falls below a threshold proportional to the parti-             networks. Adv. Neural Inf. Process. Syst. 34, 26548–26560 (2021).
cle’s diameter. This formulation ensures that only bodies in close             9. Hu, T., Lin, Z., Abel, M. F. & Allaire, P. E. Human gait modeling:
proximity to a boundary form interactions with their reﬂections,                   dealing with holonomic constraints. In Proc. 2004 American Control
enabling accurate modeling of boundary contact effects while keeping               Conference, Vol. 3, 2296–2301 (IEEE, 2004).
computational overhead minimal.                                                10. Peng, H., Song, N., Li, F. & Tang, S. A mechanistic-based data-driven
     During rollout, this reﬂection process is recalculated at every time          approach for general friction modeling in complex mechanical
step, ensuring that interactions remain strictly aligned along the nor-            system. J. Appl. Mech. 89, 071005 (2022).
mal direction.                                                                 11. Sanchez-Gonzalez, A. et al. Learning to simulate complex physics
                                                                                   with graph networks. In Proc. International Conference on Machine
Ethics statement                                                                   Learning, 8459–8468 (PMLR, 2020).
This paper presents DYNAMI-CAL GRAPHNET, a physics-informed,                   12. Atz, K., Grisoni, F. & Schneider, G. Geometric deep learning on
learning-based method for modeling discrete dynamical systems. All                 molecular representations. Nat. Mach. Intell. 3, 1023–1032 (2021).
experiments use either synthetic simulations or publicly available             13. Choi, Y. & Kumar, K. Graph neural network-based surrogate model
benchmarks that do not include any personally identiﬁable or sensitive             for granular ﬂows. Comput. Geotechnics 166, 106015 (2024).


Nature Communications | (2026)17:1045                                                                                                                20
Article                                                                                                  https://doi.org/10.1038/s41467-025-67802-5


14. Sharma, V., Ravesloot, J., Taal, C. & Fink, O. Graph neural networks       33. Zaheer, M. et al. Deep sets. in (eds Guyon, I. et al.) Advances in
    for dynamic modeling of roller bearings. In Proc. Annual Conference            Neural Information Processing Systems, Vol. 30, https://
    of the PHM Society, Vol. 15 (PHM Society, 2023).                               proceedings.neurips.cc/paper_ﬁles/paper/2017/ﬁle/
15. Han, J. et al. Learning physical dynamics with subequivariant graph            f22e4747da1aa27e363d86d40ff442fe-Paper.pdf (Curran Associ-
    neural networks. Adv. Neural Inf. Process. Syst. 35,                           ates, Inc., 2017).
    26256–26268 (2022).                                                        34. Koﬁnas, M., Nagaraja, N. S. & Gavves, E. Roto-translated local
16. Schütt, K. T., Sauceda, H. E., Kindermans, P.-J., Tkatchenko, A. &             coordinate frames for interacting dynamical systems. in (eds Bey-
    Müller, K.-R. Schnet: a deep learning architecture for molecules and           gelzimer, A., Dauphin, Y., Liang, P. & Vaughan, J. W.) Advances in
    materials. J. Chem. Phys. 148, 241722 (2018).                                  Neural Information Processing Systems, https://openreview.net/
17. Satorras, V. G., Hoogeboom, E. & Welling, M. E (n). Equivariant                forum?id=c3RKZas9am (2021).
    graph neural networks. In Proc. International Conference on                35. Kaba, S.-O., Mondal, A. K., Zhang, Y., Bengio, Y. & Ravanbakhsh, S.
    Machine Learning, 9323–9332 (PMLR, 2021).                                      Equivariance with learned canonicalization functions. In Proc.
18. Han, J. et al. A survey of geometric graph neural networks: data               International Conference on Machine Learning, 15546–15566
    structures, models and applications. Front. Comput. Sci. 19,                   (PMLR, 2023).
    1911375 (2025).                                                            36. Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A. & Battaglia, P. Learn-
19. Huang, W. et al. Equivariant graph mechanics networks with con-                ing mesh-based simulation with graph networks. In Proc. Interna-
    straints. In Proc. International Conference on Learning Representa-            tional Conference on Learning Representations, https://openreview.
    tions, https://openreview.net/forum?id=SHbhHHfePhP (2022).                     net/forum?id=roNqYL0_XP (2021).
20. Du, W. et al. Se (3) equivariant graph neural networks with complete       37. Allen, K. R. et al. Learning rigid dynamics with face interaction graph
    local frames. In Proc. International Conference on Machine Learning,           networks. In Proc. The Eleventh International Conference on Learn-
    5583–5608 (PMLR, 2022).                                                        ing Representations, https://openreview.net/forum?id=
21. Han, J., Huang, W., Xu, T. & Rong, Y. Equivariant graph hierarchy-             J7Uh781A05p (2023).
    based neural networks. in (eds Oh, A. H., Agarwal, A., Belgrave, D. &      38. Allen, K. R. et al. Graph network simulators can learn discontinuous,
    Cho, K.) Advances in Neural Information Processing Systems,                    rigid contact dynamics. In Proc. Conference on Robot Learning,
    https://openreview.net/forum?id=ywxtmG1nU_6 (2022).                            1157–1167 (PMLR, 2023).
22. Köhler, J., Klein, L. & Noé, F. Equivariant ﬂows: sampling conﬁg-          39. Carnegie Mellon University. CMU motion capture database. http://
    urations for multi-body systems with symmetric energies. Preprint              mocap.cs.cmu.edu (2003).
    at https://doi.org/10.48550/arXiv.1910.00753 (2019).                       40. Seyler, S. & Beckstein, O. Molecular dynamics trajectory for
23. Thomas, N. et al. Tensor ﬁeld networks: rotation-and translation-              benchmarking mdanalysis. Figshare https://ﬁgshare.com/articles/
    equivariant neural networks for 3d point clouds. Preprint at https://          5108170 (2017).
    doi.org/10.48550/arXiv.1802.08219 (2018).                                  41. Lu, L. Gpu accelerated mﬁx-dem simulations of granular and mul-
24. Fuchs, F., Worrall, D., Fischer, V. & Welling, M. Se(3)-transformers:          tiphase ﬂows. Particuology 62, 14–24 (2022).
    3d roto-translation equivariant attention networks. Adv. Neural Inf.       42. Garg, R., Galvin, J., Li, T. & Pannala, S. Open-source mﬁx-dem
    Process. Syst. 33, 1970–1981 (2020).                                           software for gas–solids ﬂows: part i-veriﬁcation studies. Powder
25. Batzner, S. et al. E (3)-equivariant graph neural networks for data-           Technol. 220, 122–137 (2012).
    efﬁcient and accurate interatomic potentials. Nat. Commun. 13,             43. Kipf, T., Fetaya, E., Wang, K.-C., Welling, M. & Zemel, R. Neural
    2453 (2022).                                                                   relational inference for interacting systems. In Proc. International
26. Brandstetter, J., Hesselink, R., van der Pol, E., Bekkers, E. J. & Well-       Conference on Machine Learning, 2688–2697 (PMLR, 2018).
    ing, M. Geometric and physical quantities improve e(3) equivariant         44. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O. & Dahl, G.
    message passing. In Proc. International Conference on Learning                 E. Neural message passing for quantum chemistry. In Proc.
    Representations, https://openreview.net/forum?id=_                             International Conference on Machine Learning, 1263–1272
    xwr8gOBeV1 (2022).                                                             (PMLR, 2017).
27. Sanchez-Gonzalez, A., Bapst, V., Cranmer, K. & Battaglia, P. W.            45. Wang, R., Walters, R. & Yu, R. Data augmentation vs. equivariant
    Hamiltonian graph networks with ode integrators. Preprint at                   networks: a theoretical study of generalizability on dynamics fore-
    https://doi.org/10.48550/arXiv.1909.12790 (2019).                              casting. In Proc. ICML 2022 Workshop on Principles of Distribution
28. Bhattoo, R., Ranu, S. & Krishnan, N. M. A. Learning articulated rigid          Shift (PODS). https://icml.cc/virtual/2022/20551.
    body dynamics with Lagrangian graph neural network. in (eds Oh,                2206.09450 (2022).
    A. H., Agarwal, A., Belgrave, D. & Cho, K.) Advances in Neural             46. Gerken, J. et al. Equivariance versus augmentation for spherical
    Information Processing Systems, https://openreview.net/forum?id=               images. In Proc. International Conference on Machine Learning,
    nOdfIbo3A-F (2022).                                                            7404–7421 (PMLR, 2022).
29. Gruver, N., Finzi, M. A., Stanton, S. D. & Wilson, A. G. Deconstructing    47. Gowers, R. J. et al. Mdanalysis: a python package for the rapid
    the inductive biases of Hamiltonian neural networks. In Proc.                  analysis of molecular dynamics simulations (Los Alamos National
    International Conference on Learning Representations, https://                 Laboratory (LANL), 2019).
    openreview.net/forum?id=EDeVYpT42oS (2022).
30. Sosanya, A. & Greydanus, S. Dissipative hamiltonian neural net-            Acknowledgements
    works: learning dissipative and conservative dynamics separately.          This research was funded by the Swiss National Science Foundation
    Preprint at https://doi.org/10.48550/arXiv.2201.10085 (2022).              (SNSF) Grant Number 200021_200461.
31. Horie, M. & MITSUME, N. Graph neural PDE solvers with conserva-
    tion and similarity-equivariance. In Proc. Forty-ﬁrst International        Author contributions
    Conference on Machine Learning, https://openreview.net/forum?              V.S. led the conceptualization, methodology development, and
    id=WajJf47TUi (2024).                                                      numerical experiments. O.F. supervised the research and contributed to
32. Mi, Y. et al. Conservation-informed graph learning for spatio-             the experimental design and interpretation. V.S. drafted the manuscript
    temporal dynamics prediction. Preprint at https://doi.org/10.              with input and revisions from O.F. Both authors approved the ﬁnal ver-
    48550/arXiv.2412.20962 (2024).                                             sion of the paper.



Nature Communications | (2026)17:1045                                                                                                                  21
Article                                                                                               https://doi.org/10.1038/s41467-025-67802-5


Competing interests                                                         Publisher’s note Springer Nature remains neutral with regard to jur-
V.S. and O.F. are the inventors on patent application EP 24220892.4,        isdictional claims in published maps and institutional afﬁliations.
ﬁled by École Polytechnique Fédérale de Lausanne (EPFL) at the Eur-
opean Patent Ofﬁce in Munich, Germany. The application covers aspects       Open Access This article is licensed under a Creative Commons
of the physics-informed graph neural network framework described in         Attribution-NonCommercial-NoDerivatives 4.0 International License,
this manuscript.                                                            which permits any non-commercial use, sharing, distribution and
                                                                            reproduction in any medium or format, as long as you give appropriate
Additional information                                                      credit to the original author(s) and the source, provide a link to the
Supplementary information The online version contains                       Creative Commons licence, and indicate if you modiﬁed the licensed
supplementary material available at                                         material. You do not have permission under this licence to share adapted
https://doi.org/10.1038/s41467-025-67802-5.                                 material derived from this article or parts of it. The images or other third
                                                                            party material in this article are included in the article’s Creative
Correspondence and requests for materials should be addressed to            Commons licence, unless indicated otherwise in a credit line to the
Olga Fink.                                                                  material. If material is not included in the article’s Creative Commons
                                                                            licence and your intended use is not permitted by statutory regulation or
Peer review information Nature Communications thanks the anon-              exceeds the permitted use, you will need to obtain permission directly
ymous reviewers for their contribution to the peer review of this work. A   from the copyright holder. To view a copy of this licence, visit http://
peer review ﬁle is available.                                               creativecommons.org/licenses/by-nc-nd/4.0/.

Reprints and permissions information is available at                        © The Author(s) 2026
http://www.nature.com/reprints




Nature Communications | (2026)17:1045                                                                                                                22

