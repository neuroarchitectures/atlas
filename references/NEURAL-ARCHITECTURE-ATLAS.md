# Neural Architecture Atlas

> **A Taxonomy and Evolution Map of Neural Network Architectures, 1943–2026**

**Status:** Living Research Document  
**Scope:** Neural network architecture history, taxonomy, structural primitives, architecture search, and autonomous architecture discovery  
**Related Research:** Autonomous Neural Architecture Design (ANAD)  
**Project:** Veya Architect

---

## 1. Purpose

The **Neural Architecture Atlas** is a structured map of neural network architectures and their evolution.

It is not intended to be a simple list of models or papers. The Atlas organizes neural architectures according to the underlying mechanisms by which they organize computation:

- operators
- connectivity
- routing
- state
- memory
- representation
- geometry
- recurrence
- conditional computation

The central thesis is:

> **The history of neural networks is a history of discovering new ways to organize computation.**

This perspective is particularly useful for **Autonomous Neural Architecture Design (ANAD)**, where the objective is not merely to optimize known architectures, but to explore and potentially expand the architecture space itself.

---

# 2. Architecture as a Computational Object

A neural architecture can be viewed as:

```text
Architecture
    =
    Operators
  + Connectivity
  + Routing
  + State
  + Memory
  + Representation
  + Computation Schedule
```

A conventional model name such as ResNet, ViT, GPT, or Mamba identifies a particular point or family within this larger space.

At a finer level, an architecture can be represented as:

```text
Neural Architecture
│
├── Operator Set
│   ├── Linear
│   ├── Convolution
│   ├── Attention
│   ├── MLP
│   ├── State Update
│   └── Message Passing
│
├── Connectivity
│   ├── Sequential
│   ├── Residual
│   ├── Dense
│   ├── Skip
│   ├── Graph
│   └── Hierarchical
│
├── Routing
│   ├── Static
│   ├── Input-dependent
│   ├── Sparse
│   └── Conditional
│
├── State
│   ├── Stateless
│   ├── Recurrent
│   ├── Latent
│   └── External
│
└── Representation
    ├── Vector
    ├── Grid
    ├── Token
    ├── Graph
    ├── Field
    └── World
```

---

# 3. Historical Timeline

## 3.1 1940s–1960s — Artificial Neurons

### Key architectures

| Architecture | Period | Structural idea |
|---|---:|---|
| McCulloch–Pitts neuron | 1943 | Threshold computation |
| Perceptron | 1957 | Trainable linear decision unit |
| ADALINE | 1960 | Adaptive linear neuron |
| MADALINE | 1962 | Multi-unit adaptive network |

### Primitive

```text
Weighted Aggregation
        +
Activation / Decision
```

This established the fundamental computational abstraction of an artificial neuron.

---

# 4. 1960s–1980s — Layered Feedforward Networks

## 4.1 Multilayer Perceptron

The multilayer network introduced hierarchical function composition:

```text
Input
  │
Dense
  │
Activation
  │
Dense
  │
Activation
  │
Output
```

Core principle:

```text
f(x) = fₙ(...f₂(f₁(x))...)
```

### Architectural contribution

**Deep composition**

The central architectural question became:

> How should multiple transformations be composed?

---

# 5. 1980s–1990s — Structured Connectivity

## 5.1 Convolutional Neural Networks

CNNs introduced architecture explicitly matched to spatial structure.

```text
Image
  │
Convolution
  │
Pooling
  │
Convolution
  │
Pooling
  │
Classifier
```

### Core primitives

- local connectivity
- parameter sharing
- translation equivariance
- hierarchical receptive fields

### Architectural principle

> **Data geometry can determine connectivity.**

---

## 5.2 Recurrent Neural Networks

RNNs introduced explicit temporal state:

```text
xₜ ──► RNN ──► hₜ
         ▲
         │
       hₜ₋₁
```

Formal abstraction:

```text
hₜ = f(xₜ, hₜ₋₁)
```

### Architectural principle

> **A neural architecture can contain state.**

---

# 6. 1980s–2000s — Memory and Attractor Architectures

## 6.1 Hopfield Networks

Hopfield networks introduced attractor-based computation:

```text
Input
  │
Energy Landscape
  │
Attractor
  │
Stable State
```

Core concept:

> computation as convergence toward an energy minimum.

---

## 6.2 Boltzmann Machines

```text
Visible Units
     │
Hidden Units
     │
Energy Model
```

Important descendants:

- Boltzmann Machine
- Restricted Boltzmann Machine
- Deep Belief Network

These architectures established energy-based and probabilistic neural computation as a distinct branch.

---

# 7. 1990s–2010 — Latent Representation Architectures

## 7.1 Autoencoders

```text
Input
  │
Encoder
  │
Latent Representation
  │
Decoder
  │
Reconstruction
```

Major variants:

- Autoencoder
- Sparse Autoencoder
- Denoising Autoencoder
- Contractive Autoencoder
- Variational Autoencoder
- β-VAE

### Architectural principle

> **Representation can be organized around a latent bottleneck.**

---

# 8. 1997–2015 — Gated Recurrent Architectures

## 8.1 LSTM

LSTM introduced explicit memory cells and gates:

```text
                 ┌───────────────┐
Input ──────────►│   Input Gate  │
                 │  Forget Gate  │
State ──────────►│  Output Gate  │
                 │  Memory Cell  │
                 └───────┬───────┘
                         │
                       State
```

## 8.2 GRU

A simplified gated recurrent architecture.

### Architecture family

```text
RNN
│
├── Vanilla RNN
├── Bidirectional RNN
├── LSTM
├── GRU
└── Echo State Network
```

### Architectural principle

> **State can be selectively retained, updated, or exposed.**

---

# 9. 2012–2017 — Deep Convolutional Architecture Explosion

The modern deep-learning era produced a large CNN architecture family.

```text
CNN
│
├── AlexNet
├── VGG
├── Inception
│   ├── Inception v1
│   ├── Inception v2
│   ├── Inception v3
│   └── Inception v4
│
├── ResNet
├── DenseNet
├── Xception
├── MobileNet
├── ShuffleNet
└── EfficientNet
```

---

# 10. ResNet — Connectivity as a First-Class Primitive

Residual networks changed the role of topology.

Instead of:

```text
x → F(x) → y
```

use:

```text
       ┌─────────────┐
x ─────┼──► F(x) ────┼──► +
       │             │
       └─────────────┘
```

```text
y = F(x) + x
```

### Architectural principle

> **Connectivity itself is an inductive bias.**

This is one of the most important ideas for architecture search.

---

# 11. 2014–2018 — Attention and External Memory

## 11.1 Neural Turing Machine

Neural computation was coupled to external differentiable memory:

```text
             ┌─────────────┐
             │   Memory    │
             └──────┬──────┘
                    ▲
                 read/write
                    │
             Neural Controller
```

This introduced:

- external state
- differentiable memory access
- content-based addressing

---

## 11.2 Attention

Attention changed connectivity from static to input-dependent:

```text
Query
  │
  ├── similarity ──► Keys
  │
  ▼
weighted Values
```

### Architectural principle

> **Connectivity can depend on the current input.**

---

# 12. 2017 — Transformer

The Transformer made attention the central computational primitive.

```text
Transformer Block
│
├── Self-Attention
│
├── Residual
│
├── Normalization
│
├── MLP
│
├── Residual
│
└── Normalization
```

Major branches:

```text
Transformer
│
├── Encoder-only
│   └── BERT
│
├── Decoder-only
│   └── GPT
│
├── Encoder-decoder
│   └── T5
│
├── Vision Transformer
│
├── Multimodal Transformer
│
└── Mixture-of-Experts Transformer
```

### Architectural principle

> **Global, content-dependent routing.**

---

# 13. 2018–Present — Graph Neural Networks

Graph architectures make relational topology explicit.

```text
Graph
│
├── Node
├── Edge
└── Neighborhood
       │
       ▼
Message Passing
       │
       ▼
Aggregation
       │
       ▼
Updated Representation
```

Major families:

```text
GNN
│
├── GCN
├── GraphSAGE
├── GAT
├── GIN
├── MPNN
└── Graph Transformer
```

### Architectural principle

> **Connectivity follows relational structure.**

---

# 14. 2018–Present — Neural Fields

Neural fields represent continuous functions using neural networks.

```text
Coordinate
   │
   ▼
Neural Network
   │
   ▼
Continuous Field
```

Examples:

- NeRF
- Occupancy Networks
- Neural SDF
- Neural Radiance Fields
- Neural implicit representations

Typical abstraction:

```text
f(x, y, z) → density / color / occupancy / feature
```

### Architectural principle

> **The network can represent a continuous world function rather than a finite tensor alone.**

This family is particularly relevant to spatial intelligence and physical-world representation.

---

# 15. 2020–Present — Generative Architectures

## 15.1 Generative Adversarial Networks

```text
Noise ──► Generator ──► Sample
                         ▲
                         │
                    Discriminator
```

Architecture principle:

> **Two networks form a coupled optimization system.**

---

## 15.2 Diffusion Models

```text
Noise
 │
 ▼
Neural Denoiser
 │
 ▼
Cleaner State
 │
 ▼
...
 │
 ▼
Sample
```

Major branches:

```text
Diffusion
│
├── DDPM
├── DDIM
├── Score-based Models
├── Latent Diffusion
└── Diffusion Transformer
```

### Architectural principle

> **Computation can be an iterative trajectory rather than a single forward pass.**

---

# 16. 2020–Present — State Space Architectures

State Space Models provide another approach to sequence computation.

```text
Input
 │
 ▼
State Update
 │
 ▼
State
 │
 ▼
Output
```

Major developments:

```text
SSM
│
├── S4
├── S5
├── Hyena
├── H3
└── Mamba
```

### Architectural principle

> **Long-range sequence computation can be organized around selective state evolution rather than attention.**

---

# 17. Mixture-of-Experts and Conditional Computation

MoE introduces conditional execution:

```text
                  Router
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Expert A  Expert B  Expert C
          │         │         │
          └─────────┼─────────┘
                    ▼
                 Output
```

### Architectural principle

> **The active computational graph can depend on the input.**

This is a major transition:

```text
Static computation
       ↓
Conditional computation
       ↓
Input-dependent computation
```

---

# 18. Multimodal Architectures

Multimodal models combine different representation spaces.

## 18.1 Dual Encoder

CLIP-like architecture:

```text
Image ──► Vision Encoder ──► Image Embedding
                                  │
                                  │ Similarity
                                  │
Text ───► Text Encoder ─────► Text Embedding
```

### Principle

> **Different modalities can be aligned in a shared representation space.**

---

## 18.2 Cross-Attention Architectures

```text
Modality A ──► Encoder ──┐
                         ├──► Cross Attention ──► Joint Representation
Modality B ──► Encoder ──┘
```

---

# 19. Foundation Model Architectures

Modern foundation models increasingly combine multiple architecture primitives:

```text
Foundation Model
│
├── Transformer / SSM
├── Multimodal Encoders
├── MoE
├── Retrieval
├── Memory
├── Tool Interfaces
└── External State
```

The unit of design is therefore increasingly moving from:

```text
Neural Network
```

toward:

```text
Neural Computational System
```

---

# 20. Architecture Primitive Atlas

A useful abstraction is to classify major architecture innovations by primitive.

| Primitive | Representative families | Main idea |
|---|---|---|
| Dense | MLP | Global parameterized transform |
| Convolution | CNN | Local spatial computation |
| Recurrence | RNN | Temporal state |
| Gating | LSTM / GRU | Selective state update |
| Residual | ResNet | Shortcut connectivity |
| Dense connectivity | DenseNet | Feature reuse |
| Attention | Transformer | Dynamic routing |
| Message passing | GNN | Relational computation |
| Memory | NTM / Memory Networks | External state |
| Mixture | MoE | Conditional computation |
| Field | NeRF / implicit networks | Continuous representation |
| Diffusion | DDPM / DiT | Iterative generation |
| State space | S4 / Mamba | Efficient state evolution |
| Retrieval | RAG-like systems | External information access |

---

# 21. Architecture Evolution Map

```text
Artificial Neuron
       │
       ▼
Layer
       │
       ▼
Deep Composition
       │
       ├──────────────► CNN
       │                 │
       │                 └──► ResNet / DenseNet
       │
       ├──────────────► RNN
       │                 │
       │                 └──► LSTM / GRU
       │
       ├──────────────► Autoencoder
       │                 │
       │                 └──► VAE
       │
       ├──────────────► Energy / Attractor
       │
       └──────────────► Attention
                           │
                           ▼
                       Transformer
                           │
             ┌─────────────┼─────────────┐
             │             │             │
            NLP          Vision      Multimodal
             │             │             │
             └─────────────┼─────────────┘
                           │
               ┌───────────┼────────────┐
               │           │            │
              MoE         Memory       Retrieval
               │
               ▼
         Foundation Models
               │
       ┌───────┼────────┐
       │       │        │
      GNN     SSM     Neural Field
       │       │        │
       └───────┼────────┘
               │
               ▼
          World Models
               │
               ▼
      Physical Intelligence
```

---

# 22. Nine Major Architecture Primitives

Across the history of neural networks, a large fraction of architectures can be described using nine recurring primitives:

1. **Dense Transformation**
2. **Convolution**
3. **Recurrence**
4. **Attention**
5. **Message Passing**
6. **Memory**
7. **Conditional Routing**
8. **Continuous Field Representation**
9. **State-Space Evolution**

These primitives can be composed into larger architectures.

---

# 23. Architecture Space

The number of possible architectures is not equivalent to the number of named architectures in the literature.

For example:

```text
10 operators
× 10 connection patterns
× 20 depths
× 10 widths
× 10 activation choices
```

already yields:

```text
2,000,000
```

possible configurations.

Real search spaces can be vastly larger.

Therefore:

> **There is no meaningful finite count of all possible neural architectures.**

A more useful distinction is:

| Level | Approximate scale |
|---|---:|
| Fundamental primitives | Tens |
| Major architecture families | Hundreds |
| Named research architectures | Thousands–tens of thousands |
| Concrete architecture variants | Potentially millions+ |
| Full combinatorial architecture space | Astronomical / effectively open-ended |

These values are conceptual ranges rather than a census.

---

# 24. Human-Designed vs Machine-Discovered Space

```text
                 Architecture Space
                        │
          ┌─────────────┴─────────────┐
          │                           │
   Human-explored                 Unexplored
          │                           │
   ResNet / ViT / GPT                 ?
   Mamba / GNN / NeRF                 ?
          │                           │
          └─────────────┬─────────────┘
                        │
                    ANAD / NAS
                        │
                        ▼
              Architecture Discovery
```

Traditional NAS generally searches within a predefined search space.

ANAD aims to move toward:

> **systems capable of modifying, extending, or discovering the search space itself.**

---

# 25. Neural Architecture Search

NAS can be decomposed into three major components:

```text
NAS
│
├── Search Space
│
├── Search Strategy
│
└── Architecture Evaluation
```

## 25.1 Search Spaces

Common representations:

- chain
- cell
- graph
- block hierarchy
- operator graph
- macro architecture
- micro architecture

## 25.2 Search Strategies

Major approaches:

```text
NAS
│
├── Reinforcement Learning
├── Evolutionary Search
├── Random Search
├── Bayesian Optimization
├── Differentiable NAS
├── Weight Sharing
├── Zero-Cost Search
└── Generative / LLM-guided Search
```

---

# 26. ANAD — Autonomous Neural Architecture Design

ANAD extends the problem from:

> **Search within a known architecture space**

to:

> **Discover and evolve architecture spaces and architectures.**

Conceptually:

```text
             Architecture Genome
                     │
        ┌────────────┼────────────┐
        │            │            │
     Operators   Topology       Routing
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
             Neural Program
                     │
             ┌───────┴───────┐
             │               │
        Train / Adapt     Evaluate
             │               │
             └───────┬───────┘
                     ▼
              Architecture
                     │
                  Mutate
                     │
                     ▼
             New Architecture
```

---

# 27. From NAS to ANAD

| NAS | ANAD |
|---|---|
| Fixed search space | Evolving search space |
| Human-defined primitives | Discoverable primitives |
| Optimize architecture | Discover architecture |
| Task-specific objective | Multi-objective discovery |
| Usually model-centric | Computation-centric |
| Search known structures | Explore unknown structures |
| Architecture optimization | Architecture invention |

This distinction should remain central to Veya Architect.

---

# 28. Open Architecture Space

The most interesting unexplored regions may not be another variation of CNN or Transformer.

Potential directions include:

```text
Unknown Architecture Space
│
├── New connectivity primitives
├── New state mechanisms
├── New routing mechanisms
├── New memory architectures
├── New spatial computation primitives
├── New temporal computation primitives
├── New continuous/discrete hybrids
├── Neural-symbolic architectures
├── Dynamic computational graphs
└── Architectures coupling perception, state, and action
```

---

# 29. Relevance to Spatial Intelligence

For spatial intelligence, the relevant architecture space is broader than image recognition.

A physical-world architecture may need to represent:

```text
Observation
    │
    ▼
Perception
    │
    ▼
Geometry
    │
    ▼
Objects / Relations
    │
    ▼
State
    │
    ▼
Dynamics
    │
    ▼
Memory
    │
    ▼
Prediction / Reasoning
    │
    ▼
Action
```

This introduces a new architectural question:

> **How should neural computation be organized around a persistent physical-world state?**

This is not naturally equivalent to CNN, Transformer, or RNN.

---

# 30. Architecture Taxonomy for Veya Architect

The Veya Architect research space can therefore be organized as:

```text
Veya Architect
│
├── Operator Discovery
│   ├── Vision operators
│   ├── Spatial operators
│   ├── Temporal operators
│   └── Reasoning operators
│
├── Topology Discovery
│   ├── Sequential
│   ├── Hierarchical
│   ├── Graph
│   ├── Recurrent
│   └── Dynamic
│
├── State Discovery
│   ├── Stateless
│   ├── Recurrent
│   ├── Latent state
│   └── World state
│
├── Routing Discovery
│   ├── Static
│   ├── Attention
│   ├── Sparse
│   └── Conditional
│
└── Architecture Evolution
    ├── Mutation
    ├── Composition
    ├── Decomposition
    ├── Crossover
    └── Primitive invention
```

---

# 31. Atlas Research Questions

The Atlas should track the following questions over time.

### Q1. What are the fundamental architecture primitives?

Can the hundreds of named architectures be reduced to a much smaller set of computational primitives?

### Q2. Which architecture innovations changed topology?

Examples:

- convolution
- recurrence
- residual connections
- attention
- message passing
- conditional routing

### Q3. Which innovations changed computation itself?

Examples:

- external memory
- diffusion
- state-space evolution
- neural fields
- conditional execution

### Q4. What regions of architecture space remain poorly explored?

This is the core ANAD research question.

### Q5. Can architecture discovery become self-expanding?

Instead of:

```text
Human → Search Space → NAS
```

investigate:

```text
Human
  │
  ▼
Initial primitives
  │
  ▼
Architecture Discovery
  │
  ▼
New primitives
  │
  ▼
Expanded Search Space
  │
  ▼
Further Discovery
```

---

# 32. Working Definition

For this Atlas:

> **A neural architecture is a structured computational organization composed of operators, connectivity, state, routing, memory, representation, and computation schedule.**

An **architecture family** is a set of architectures sharing a defining computational organization.

An **architecture primitive** is a reusable mechanism that defines a fundamental mode of computation or connectivity.

An **architecture space** is the combinatorial space generated by possible primitives, compositions, topologies, and configurations.

---

# 33. Core Thesis

The historical progression can be summarized as:

```text
Neuron
  ↓
Layer
  ↓
Deep Composition
  ↓
Structured Connectivity
  ↓
State
  ↓
Memory
  ↓
Dynamic Routing
  ↓
Global Attention
  ↓
Conditional Computation
  ↓
Continuous World Representation
  ↓
Structured World Representation
  ↓
Persistent World State
  ↓
???
```

The final stage remains an open research problem.

That open space is where **Autonomous Neural Architecture Design** becomes relevant.

---

# 34. Selected References

The Atlas should maintain primary references for each major architectural transition.

- McCulloch & Pitts — *A Logical Calculus of the Ideas Immanent in Nervous Activity* (1943)
- Rosenblatt — *The Perceptron* (1958)
- Rumelhart, Hinton & Williams — *Learning Representations by Back-Propagating Errors* (1986)
- LeCun et al. — *Gradient-Based Learning Applied to Document Recognition* (1998)
- Hochreiter & Schmidhuber — *Long Short-Term Memory* (1997)
- Kingma & Welling — *Auto-Encoding Variational Bayes* (2013)
- He et al. — *Deep Residual Learning for Image Recognition* (2015)
- Vaswani et al. — *Attention Is All You Need* (2017)
- Devlin et al. — *BERT* (2018)
- Dosovitskiy et al. — *An Image Is Worth 16x16 Words* (2020)
- Radford et al. — *Learning Transferable Visual Models From Natural Language Supervision* (2021)
- Kipf & Welling — *Semi-Supervised Classification with Graph Convolutional Networks* (2017)
- Mildenhall et al. — *NeRF* (2020)
- Ho et al. — *Denoising Diffusion Probabilistic Models* (2020)
- Gu & Dao — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023)

---

# 35. Maintenance

This document is intended to be a **living atlas**.

Future additions should be organized by:

1. **Architecture family**
2. **Structural primitive**
3. **Connectivity pattern**
4. **State / memory mechanism**
5. **Routing mechanism**
6. **Representation type**
7. **Historical significance**
8. **Relation to existing families**
9. **Potential unexplored architecture space**
10. **Relevance to ANAD**

New architectures should not merely be appended to a chronological list. Each addition should identify **what computational organization is genuinely new**.

---

## Appendix A — Compact Architecture Map

```text
NEURAL ARCHITECTURE
│
├── Feedforward
│   └── MLP
│
├── Spatial
│   ├── CNN
│   ├── ResNet
│   ├── DenseNet
│   └── ViT
│
├── Temporal
│   ├── RNN
│   ├── LSTM
│   ├── GRU
│   └── SSM / Mamba
│
├── Relational
│   ├── GNN
│   ├── GAT
│   └── Graph Transformer
│
├── Attention
│   ├── Transformer
│   ├── Cross Attention
│   └── Multimodal Transformer
│
├── Memory
│   ├── Hopfield
│   ├── NTM
│   └── Retrieval / External Memory
│
├── Generative
│   ├── AE
│   ├── VAE
│   ├── GAN
│   ├── Diffusion
│   └── Flow
│
├── Conditional
│   ├── MoE
│   ├── Dynamic Routing
│   └── Sparse Computation
│
├── Continuous Representation
│   ├── Neural Fields
│   ├── NeRF
│   └── Neural SDF
│
└── Emerging
    ├── World Models
    ├── Physical Intelligence
    ├── Neuro-symbolic
    └── Autonomous Architecture Discovery
```

---

## Appendix B — ANAD Position

```text
                  HUMAN-DESIGNED
                  ARCHITECTURES
                         │
                         ▼
                 Known Architecture
                      Families
                         │
                         ▼
                 Architecture Space
                         │
            ┌────────────┴────────────┐
            │                         │
       Known Region             Unknown Region
            │                         │
            ▼                         ▼
      Traditional NAS              ANAD
            │                         │
            │                  Discover / Extend
            │                  Architecture Space
            │                         │
            └────────────┬────────────┘
                         ▼
               New Computational
                   Organizations
```

**Veya Architect** is positioned on the right-hand side of this map.
