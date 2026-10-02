# Basic NN Building Blocks — Index

A catalog of **distinct, reusable neural network building blocks** extracted from the real architectures in [`../architectures/`](../architectures/). Each block has its own folder with a `README.md` documenting its design philosophy, functionality, which models use it, key features, and evolution.

**Total blocks: 295** — no duplication. Each block is a distinct structural or functional unit that appears in one or more real architectures.

---

## Block Catalog

| # | Block | Folder |
|---|-------|--------|
| 1 | 3D Gaussian Primitives | [`3d-gaussian-primitives/`](./3d-gaussian-primitives/) |
| 2 | Action-Conditioned Latent Dynamics | [`action-conditioned-latent-dynamics/`](./action-conditioned-latent-dynamics/) |
| 3 | ActNorm (Activation Normalization) | [`actnorm/`](./actnorm/) |
| 4 | AdamW (Adam with Decoupled Weight Decay) | [`adamw-optimizer/`](./adamw-optimizer/) |
| 5 | Adaptive Adjacency Learning | [`adaptive-adjacency-learning/`](./adaptive-adjacency-learning/) |
| 6 | Adaptive Computation Time (ACT) | [`adaptive-computation-time/`](./adaptive-computation-time/) |
| 7 | Adaptive LayerNorm (adaLN / adaLN-Zero) | [`adaptive-layernorm/`](./adaptive-layernorm/) |
| 8 | ALiBi (Attention with Linear Biases) | [`alibi/`](./alibi/) |
| 9 | All-MLP Segmentation Decoder | [`all-mlp-decoder/`](./all-mlp-decoder/) |
| 10 | All-Pairs Correlation Pyramid | [`all-pairs-correlation-pyramid/`](./all-pairs-correlation-pyramid/) |
| 11 | Antisymmetric Edge-Local Frame GNN | [`antisymmetric-edge-frame-gnn/`](./antisymmetric-edge-frame-gnn/) |
| 12 | ArcFace Loss | [`arcface-loss/`](./arcface-loss/) |
| 13 | Area Attention | [`area-attention/`](./area-attention/) |
| 14 | Atrous Spatial Pyramid Pooling (ASPP) | [`aspp-decoder/`](./aspp-decoder/) |
| 15 | Associative Memory Transformer (ARMT) | [`associative-memory-transformer/`](./associative-memory-transformer/) |
| 16 | Learned Attention Sink | [`attention-sink/`](./attention-sink/) |
| 17 | Auxiliary-Loss-Free Load Balancing | [`auxiliary-loss-free-balancing/`](./auxiliary-loss-free-balancing/) |
| 18 | Axial-Planar 3D Convolution | [`axial-planar-3d-conv/`](./axial-planar-3d-conv/) |
| 19 | Backpack Sense Vectors | [`backpack-sense-vectors/`](./backpack-sense-vectors/) |
| 20 | Barlow Twins Loss | [`barlow-twins-loss/`](./barlow-twins-loss/) |
| 21 | BatchNorm (Batch Normalization) | [`batch-norm/`](./batch-norm/) |
| 22 | BEV Query Grid | [`bev-query-grid/`](./bev-query-grid/) |
| 23 | Bidirectional Selective SSM (Vim) | [`bidirectional-ssm/`](./bidirectional-ssm/) |
| 24 | Block-Causal Diffusion Generation | [`block-causal-diffusion/`](./block-causal-diffusion/) |
| 25 | Bottleneck Residual Block | [`bottleneck-residual/`](./bottleneck-residual/) |
| 26 | Camera-Aware Depth Conditioning | [`camera-aware-depth-conditioning/`](./camera-aware-depth-conditioning/) |
| 27 | Candidate-in-Sequence | [`candidate-in-sequence/`](./candidate-in-sequence/) |
| 28 | Canonical Camera Space Transform | [`canonical-camera-transform/`](./canonical-camera-transform/) |
| 29 | Anchor-Free Center Heatmap Head | [`center-heatmap-head/`](./center-heatmap-head/) |
| 30 | Center Loss | [`center-loss/`](./center-loss/) |
| 31 | Customized Gate Control (CGC / PLE) | [`cgc-expert-separation/`](./cgc-expert-separation/) |
| 32 | Channel-Split Shuffle Block | [`channel-split-shuffle-block/`](./channel-split-shuffle-block/) |
| 33 | Classifier-Free Guidance | [`classifier-free-guidance/`](./classifier-free-guidance/) |
| 34 | Codebook Quantization of Latents | [`codebook-quantization/`](./codebook-quantization/) |
| 35 | Complex MIMO State-Space Mixer (Mamba-3) | [`complex-mimo-ssm/`](./complex-mimo-ssm/) |
| 36 | Confidence-Weighted Regression Loss | [`confidence-weighted-loss/`](./confidence-weighted-loss/) |
| 37 | Conformer Block (Conv + Attention) | [`conformer-block/`](./conformer-block/) |
| 38 | Continuous Next-Embedding Prediction | [`continuous-embedding-prediction/`](./continuous-embedding-prediction/) |
| 39 | Contrastive Denoising Queries (CDN) | [`contrastive-denoising-queries/`](./contrastive-denoising-queries/) |
| 40 | Contrastive Dual-Encoder (CLIP) | [`contrastive-dual-encoder/`](./contrastive-dual-encoder/) |
| 41 | Conv-BN-ReLU Block | [`conv-bn-relu/`](./conv-bn-relu/) |
| 42 | Conv1D Feature Extractor (Audio Stem) | [`conv1d-audio-stem/`](./conv1d-audio-stem/) |
| 43 | Convex Upsampling | [`convex-upsampling/`](./convex-upsampling/) |
| 44 | ConvGRU Recurrent Update Operator | [`convgru-update-operator/`](./convgru-update-operator/) |
| 45 | ConvNeXt Block | [`convnext-block/`](./convnext-block/) |
| 46 | Convolutional Positional Embedding | [`convolutional-positional-embedding/`](./convolutional-positional-embedding/) |
| 47 | Cooperative Action-Environment GNN | [`cooperative-action-network/`](./cooperative-action-network/) |
| 48 | Copy Mechanism | [`copy-mechanism/`](./copy-mechanism/) |
| 49 | Cosine Learning Rate Schedule | [`cosine-lr-schedule/`](./cosine-lr-schedule/) |
| 50 | Coupling Layer (RealNVP) | [`coupling-layer/`](./coupling-layer/) |
| 51 | Cross-Attention Block | [`cross-attention/`](./cross-attention/) |
| 52 | Cross-Graph Attention Matching | [`cross-graph-attention/`](./cross-graph-attention/) |
| 53 | Cross-Stage-Partial (CSP) Stage | [`csp-block/`](./csp-block/) |
| 54 | Cross-Shaped Window Attention | [`cswin-attention/`](./cswin-attention/) |
| 55 | DARTS Cell (Differentiable NAS) | [`darts-cell/`](./darts-cell/) |
| 56 | DDPM Noise Schedule | [`ddpm-noise-schedule/`](./ddpm-noise-schedule/) |
| 57 | Decoder-Only Block (GPT/LLaMA style) | [`decoder-only-block/`](./decoder-only-block/) |
| 58 | Decoupled Anchor-Free Detection Head | [`decoupled-detection-head/`](./decoupled-detection-head/) |
| 59 | Deep & Cross Network (DCN / DCN-v2) | [`deep-cross-network/`](./deep-cross-network/) |
| 60 | Deep Graph Infomax (DGI) | [`deep-graph-infomax/`](./deep-graph-infomax/) |
| 61 | DeepFM Block | [`deepfm/`](./deepfm/) |
| 62 | Deformable Convolution (DCN-v2) | [`deformable-convolution/`](./deformable-convolution/) |
| 63 | Dense Connectivity Block (DenseNet) | [`dense-connectivity/`](./dense-connectivity/) |
| 64 | Depthwise-Separable Convolution | [`depthwise-separable-conv/`](./depthwise-separable-conv/) |
| 65 | Dice Loss (and Soft Dice) | [`dice-loss/`](./dice-loss/) |
| 66 | Differential Attention | [`differential-attention/`](./differential-attention/) |
| 67 | Diffusion U-Net (Latent Diffusion Denoiser) | [`diffusion-unet/`](./diffusion-unet/) |
| 68 | DiT Block (Diffusion Transformer) | [`dit-block/`](./dit-block/) |
| 69 | Divided Space-Time Attention | [`divided-space-time-attention/`](./divided-space-time-attention/) |
| 70 | DPT Reassembly + Fusion Decoder | [`dpt-decoder/`](./dpt-decoder/) |
| 71 | Dynamic Routing (Capsule Network) | [`dynamic-routing/`](./dynamic-routing/) |
| 72 | E(3)-Equivariant MLP | [`e3-equivariant-mlp/`](./e3-equivariant-mlp/) |
| 73 | Edge-Feature Attention Injection | [`edge-feature-attention/`](./edge-feature-attention/) |
| 74 | Edge Memory Message Passing | [`edge-memory-message-passing/`](./edge-memory-message-passing/) |
| 75 | EEG Conv Tokenizer Stem | [`eeg-conv-stem/`](./eeg-conv-stem/) |
| 76 | Exponential Moving Average of Weights (EMA / SWA) | [`ema-weights/`](./ema-weights/) |
| 77 | Embedding Distillation | [`embedding-distillation/`](./embedding-distillation/) |
| 78 | Entire-Space Multi-Task Towers | [`entire-space-multitask/`](./entire-space-multitask/) |
| 79 | Equalized Learning Rate (GAN) | [`equalized-lr/`](./equalized-lr/) |
| 80 | Expert Choice Routing | [`expert-choice-routing/`](./expert-choice-routing/) |
| 81 | External Memory Bank | [`external-memory-bank/`](./external-memory-bank/) |
| 82 | Factorized Space-Time Encoder | [`factorised-space-time-encoder/`](./factorised-space-time-encoder/) |
| 83 | Fast Feedforward Network (FFF) | [`fast-feedforward-network/`](./fast-feedforward-network/) |
| 84 | FCN Mask Branch | [`fcn-mask-head/`](./fcn-mask-head/) |
| 85 | Feature Pyramid Network (FPN) | [`feature-pyramid-network/`](./feature-pyramid-network/) |
| 86 | Fine-Grained Distribution Refinement (FDR) | [`fine-grained-distribution-refinement/`](./fine-grained-distribution-refinement/) |
| 87 | Flow Matching Objective | [`flow-matching/`](./flow-matching/) |
| 89 | FNet Block (Fourier Transform) | [`fnet-block/`](./fnet-block/) |
| 90 | Focal Loss | [`focal-loss/`](./focal-loss/) |
| 91 | Fourier Positional Encoding (Coordinate Mapping) | [`fourier-positional-encoding/`](./fourier-positional-encoding/) |
| 92 | 3D Frustum Position Embedding | [`frustum-3d-position-embedding/`](./frustum-3d-position-embedding/) |
| 93 | Frustum-to-BEV Voxel Pooling (Lift-Splat) | [`frustum-voxel-pooling/`](./frustum-voxel-pooling/) |
| 94 | GAN Minimax Block (Generator + Discriminator) | [`gan-minimax/`](./gan-minimax/) |
| 95 | Gated Cross-Attention (Multimodal Fusion) | [`gated-cross-attention/`](./gated-cross-attention/) |
| 96 | Gaussian Node Embedding | [`gaussian-node-embedding/`](./gaussian-node-embedding/) |
| 97 | GeGLU Feed-Forward Network | [`geglu-ffn/`](./geglu-ffn/) |
| 98 | GELU (Gaussian Error Linear Unit) | [`gelu/`](./gelu/) |
| 99 | Geometry Encoding Volume (GEV) | [`geometry-encoding-volume/`](./geometry-encoding-volume/) |
| 100 | Ghost Module | [`ghost-module/`](./ghost-module/) |
| 101 | GIoU / CIoU / DIoU Loss | [`giou-loss/`](./giou-loss/) |
| 102 | Global-Local Feature Fusion (HQ-Features) | [`global-local-feature-fusion/`](./global-local-feature-fusion/) |
| 103 | Global/Local Patch Decoder (MegaByte) | [`global-local-patch-decoder/`](./global-local-patch-decoder/) |
| 104 | GMF + MLP Fusion | [`gmf-mlp-fusion/`](./gmf-mlp-fusion/) |
| 105 | gMLP Block (Spatial Gating) | [`gmlp-block/`](./gmlp-block/) |
| 106 | Global Optimal Localization Self-Distillation (GoLSD) | [`go-lsd-self-distillation/`](./go-lsd-self-distillation/) |
| 107 | Gradient Checkpointing (Activation Recomputation) | [`gradient-checkpointing/`](./gradient-checkpointing/) |
| 108 | Gram Anchoring Loss | [`gram-anchoring-loss/`](./gram-anchoring-loss/) |
| 109 | Graph Attention Layer (GAT) | [`graph-attention-layer/`](./graph-attention-layer/) |
| 110 | Graph Coarsening Hierarchy | [`graph-coarsening-hierarchy/`](./graph-coarsening-hierarchy/) |
| 111 | Graph Convolution Layer (GCN) | [`graph-convolution-layer/`](./graph-convolution-layer/) |
| 112 | Convolution-Based Graph Diffusion | [`graph-diffusion-convolution/`](./graph-diffusion-convolution/) |
| 113 | Graph Prompt Tokens | [`graph-prompting/`](./graph-prompting/) |
| 114 | Graph Propagation Attention (GPA) | [`graph-propagation-attention/`](./graph-propagation-attention/) |
| 115 | Graph Softmax | [`graph-softmax/`](./graph-softmax/) |
| 116 | Graphlet Degree Vector (GDV) | [`graphlet-degree-vector/`](./graphlet-degree-vector/) |
| 117 | GraphSAGE Neighborhood Aggregator | [`graphsage-aggregator/`](./graphsage-aggregator/) |
| 118 | Grouped-Query Attention (GQA) / Multi-Query Attention (MQA) | [`grouped-query-attention/`](./grouped-query-attention/) |
| 119 | GroupNorm | [`groupnorm/`](./groupnorm/) |
| 120 | Group Relative Policy Optimization (GRPO) | [`grpo/`](./grpo/) |
| 121 | Gated Recurrent Unit (GRU) | [`gru-cell/`](./gru-cell/) |
| 122 | Straight-Through Gumbel-Softmax | [`gumbel-softmax-straight-through/`](./gumbel-softmax-straight-through/) |
| 123 | HiFi-GAN Generator | [`hifi-gan-generator/`](./hifi-gan-generator/) |
| 124 | High-Order Proximity Update | [`high-order-proximity-update/`](./high-order-proximity-update/) |
| 125 | Hungarian (Bipartite) Matching Loss | [`hungarian-matching-loss/`](./hungarian-matching-loss/) |
| 126 | Position-Dependent Hyperedge Embedding | [`hyperedge-position-embedding/`](./hyperedge-position-embedding/) |
| 127 | Tri-Level Hypergraph Contrast | [`hypergraph-contrastive-learning/`](./hypergraph-contrastive-learning/) |
| 128 | Inception Module | [`inception-module/`](./inception-module/) |
| 129 | InfoNCE (Contrastive) Loss | [`infonce-contrastive-loss/`](./infonce-contrastive-loss/) |
| 130 | Instance Appearance Adapter | [`instance-appearance-embedding/`](./instance-appearance-embedding/) |
| 131 | Interleaved NoPE (iRoPE) | [`interleaved-nope/`](./interleaved-nope/) |
| 132 | Intermediate-Layer Feature Tap | [`intermediate-layer-tap/`](./intermediate-layer-tap/) |
| 133 | Inverted Residual Block (MobileNetV2) | [`inverted-residual/`](./inverted-residual/) |
| 134 | Invertible 1x1 Convolution (Glow) | [`invertible-1x1-conv/`](./invertible-1x1-conv/) |
| 135 | Joint Detection-Tracking Decoder Queries | [`joint-detection-tracking-queries/`](./joint-detection-tracking-queries/) |
| 136 | Joint Space-Time Track Attention | [`joint-spacetime-attention/`](./joint-spacetime-attention/) |
| 137 | Kaiming (He) Initialization | [`kaiming-init/`](./kaiming-init/) |
| 138 | KAN Layer (Kolmogorov-Arnold Network) | [`kan-layer/`](./kan-layer/) |
| 139 | Keypoint Heatmap Head | [`keypoint-heatmap-head/`](./keypoint-heatmap-head/) |
| 140 | Label Smoothing | [`label-smoothing/`](./label-smoothing/) |
| 141 | Language-Guided Query Selection | [`language-guided-query-selection/`](./language-guided-query-selection/) |
| 142 | Laplacian Positional Encoding | [`laplacian-positional-encoding/`](./laplacian-positional-encoding/) |
| 143 | LayerNorm (Layer Normalization) | [`layer-norm/`](./layer-norm/) |
| 144 | Learned Absolute Position Embedding | [`learned-position-embedding/`](./learned-position-embedding/) |
| 145 | LightGCN Propagation | [`lightgcn-propagation/`](./lightgcn-propagation/) |
| 146 | Linear Attention | [`linear-attention/`](./linear-attention/) |
| 147 | Linear LR Warmup | [`linear-warmup/`](./linear-warmup/) |
| 148 | Lion Optimizer | [`lion-optimizer/`](./lion-optimizer/) |
| 149 | Local/Global Interleaved Attention | [`local-global-interleaved-attention/`](./local-global-interleaved-attention/) |
| 150 | Look-Forward-Twice Box Supervision | [`look-forward-twice/`](./look-forward-twice/) |
| 151 | LSTM Cell | [`lstm-cell/`](./lstm-cell/) |
| 152 | MambaVision Mixer | [`mambavision-mixer/`](./mambavision-mixer/) |
| 153 | Masked Attention | [`masked-attention/`](./masked-attention/) |
| 154 | Masked Autoencoder (MAE) | [`masked-autoencoder/`](./masked-autoencoder/) |
| 155 | Masked Language Modeling (Cloze) | [`masked-language-modeling/`](./masked-language-modeling/) |
| 156 | Masked Latent Prediction (JEPA) | [`masked-latent-prediction/`](./masked-latent-prediction/) |
| 157 | MBConv Block | [`mbconv/`](./mbconv/) |
| 158 | Memory Token Segment Recurrence | [`memory-tokens/`](./memory-tokens/) |
| 159 | Meta-Path Guided Random Walk | [`metapath-random-walk/`](./metapath-random-walk/) |
| 160 | Minibatch Standard Deviation | [`minibatch-stddev/`](./minibatch-stddev/) |
| 161 | Mixed Precision (AMP / BF16 / FP16) | [`mixed-precision-amp/`](./mixed-precision-amp/) |
| 162 | Mixture Density Network | [`mixture-density-network/`](./mixture-density-network/) |
| 163 | Mixture-of-Experts (MoE) Layer | [`mixture-of-experts/`](./mixture-of-experts/) |
| 164 | Mixture-of-Laplace Loss | [`mixture-of-laplace-loss/`](./mixture-of-laplace-loss/) |
| 165 | MLP-Mixer Block | [`mlp-mixer-block/`](./mlp-mixer-block/) |
| 166 | MLP Projector Bridge (LLaVA-style) | [`mlp-projector-bridge/`](./mlp-projector-bridge/) |
| 167 | MobileViT Block | [`mobilevit-block/`](./mobilevit-block/) |
| 168 | Multi-Arity Expand-Reduce Hyperedge Layer | [`multi-arity-hyperedge-layer/`](./multi-arity-hyperedge-layer/) |
| 169 | Multi-Axis Attention | [`multi-axis-attention/`](./multi-axis-attention/) |
| 170 | Multi-Gate Mixture-of-Experts (MMoE) | [`multi-gate-moe/`](./multi-gate-moe/) |
| 171 | Multi-Head Attention (Full / Dense) | [`multi-head-attention/`](./multi-head-attention/) |
| 172 | Multi-Head Latent Attention (MLA) | [`multi-head-latent-attention/`](./multi-head-latent-attention/) |
| 173 | Ambiguity-Aware Multi-Mask Output | [`multi-mask-iou-head/`](./multi-mask-iou-head/) |
| 175 | Multi-Resolution Parallel Streams (HRNet) | [`multi-resolution-parallel-streams/`](./multi-resolution-parallel-streams/) |
| 176 | Multi-Scale Deformable Attention (MSDeformAttn) | [`multi-scale-deformable-attention/`](./multi-scale-deformable-attention/) |
| 177 | Multi-Token Prediction (MTP) | [`multi-token-prediction/`](./multi-token-prediction/) |
| 178 | Multi-View Graph Fusion | [`multi-view-graph-fusion/`](./multi-view-graph-fusion/) |
| 179 | MultKAN Multiplication Node | [`multiplication-node/`](./multiplication-node/) |
| 180 | Mutual Selective Attention | [`mutual-attention/`](./mutual-attention/) |
| 181 | N-BEATS Block | [`n-beats-block/`](./n-beats-block/) |
| 182 | Native Sparse Attention (NSA) | [`native-sparse-attention/`](./native-sparse-attention/) |
| 183 | Neural ODE Block | [`neural-ode-block/`](./neural-ode-block/) |
| 184 | Neural Synchronization Representation | [`neural-synchronization/`](./neural-synchronization/) |
| 185 | Neuron-Level Model (NLM) | [`neuron-level-model/`](./neuron-level-model/) |
| 186 | Nyströmformer Attention | [`nystromformer-attention/`](./nystromformer-attention/) |
| 187 | One-Shot Probabilistic Graph Decoder | [`one-shot-graph-decoder/`](./one-shot-graph-decoder/) |
| 188 | OneCycle LR Schedule | [`onecycle-lr-schedule/`](./onecycle-lr-schedule/) |
| 189 | Orthogonal Regularization | [`orthogonal-regularization/`](./orthogonal-regularization/) |
| 190 | Pairwise Dot-Interaction | [`pairwise-dot-interaction/`](./pairwise-dot-interaction/) |
| 191 | Pairwise Ranking Loss (BPR / Margin / Hop-Distance) | [`pairwise-ranking-loss/`](./pairwise-ranking-loss/) |
| 192 | Pairwise Sigmoid Contrastive Loss | [`pairwise-sigmoid-contrastive-loss/`](./pairwise-sigmoid-contrastive-loss/) |
| 193 | Parallel Attention+MLP Block (GPT-NeoX) | [`parallel-residual-block/`](./parallel-residual-block/) |
| 194 | Partial Convolution (PConv) | [`partial-convolution/`](./partial-convolution/) |
| 195 | Patch Embedding (ViT) | [`patch-embedding/`](./patch-embedding/) |
| 196 | PatchTST Block (Channel-Independent Patch Transformer) | [`patchtst-block/`](./patchtst-block/) |
| 197 | Path-Based Message Passing | [`path-message-passing/`](./path-message-passing/) |
| 198 | Per-Region Optimal Supervision | [`per-region-optimal-supervision/`](./per-region-optimal-supervision/) |
| 199 | Perceiver Resampler | [`perceiver-resampler/`](./perceiver-resampler/) |
| 200 | Pixel Normalization (GAN) | [`pixelnorm/`](./pixelnorm/) |
| 201 | Poincaré Ball Embedding | [`poincare-embedding/`](./poincare-embedding/) |
| 202 | Point Transformer Attention | [`point-transformer-attention/`](./point-transformer-attention/) |
| 203 | Pointmap Regression | [`pointmap-regression/`](./pointmap-regression/) |
| 204 | PointNet Shared MLP | [`pointnet-mlp/`](./pointnet-mlp/) |
| 205 | k-Step PPMI-SVD Embedding | [`ppmi-svd-embedding/`](./ppmi-svd-embedding/) |
| 206 | Presence / Occlusion Head | [`presence-occlusion-head/`](./presence-occlusion-head/) |
| 207 | Sparse/Dense Prompt Encoder | [`prompt-encoder/`](./prompt-encoder/) |
| 208 | Prototype Mask Head (YOLACT-style) | [`prototype-mask-head/`](./prototype-mask-head/) |
| 209 | Q-Former (Learned Query Bridge) | [`q-former/`](./q-former/) |
| 210 | QK-Norm | [`qk-norm/`](./qk-norm/) |
| 211 | Uncertainty-Minimal Query Selection | [`quality-aware-query-selection/`](./quality-aware-query-selection/) |
| 212 | Object-Query Memory Queue | [`query-memory-queue/`](./query-memory-queue/) |
| 213 | Reconciled Polynomial Layer (RPN) | [`reconciled-polynomial-layer/`](./reconciled-polynomial-layer/) |
| 214 | Recurrent World Model + Imagination | [`recurrent-world-model/`](./recurrent-world-model/) |
| 215 | Reformer Attention (LSH) | [`reformer-attention/`](./reformer-attention/) |
| 216 | RPN (Region Proposal Network) | [`region-proposal-network/`](./region-proposal-network/) |
| 217 | Region-Text Contrastive Head | [`region-text-contrastive-head/`](./region-text-contrastive-head/) |
| 218 | Relative Position Bias | [`relative-position-bias/`](./relative-position-bias/) |
| 219 | Residual Block (Identity Skip) | [`residual-block/`](./residual-block/) |
| 220 | Residual-Depth-Scaled Init / muP Scaled Residuals | [`residual-scaling-init/`](./residual-scaling-init/) |
| 221 | Residual Vector Quantization (RVQ) | [`residual-vector-quantization/`](./residual-vector-quantization/) |
| 222 | RL Attention-Guided Graph Walk | [`rl-attention-guided-graph-walk/`](./rl-attention-guided-graph-walk/) |
| 223 | RLHF PPO Pipeline | [`rlhf-ppo-pipeline/`](./rlhf-ppo-pipeline/) |
| 224 | RMSNorm | [`rmsnorm/`](./rmsnorm/) |
| 225 | RoIAlign | [`roialign/`](./roialign/) |
| 226 | Rotary Position Embedding (RoPE) | [`rope/`](./rope/) |
| 227 | RSSM (Recurrent State-Space Model) | [`rssm/`](./rssm/) |
| 228 | RWKV Channel Mixing | [`rwkv-channel-mixing/`](./rwkv-channel-mixing/) |
| 229 | RWKV Time-Mixing Block | [`rwkv-time-mixing/`](./rwkv-time-mixing/) |
| 230 | Scale-and-Shift-Invariant Depth Loss | [`scale-shift-invariant-loss/`](./scale-shift-invariant-loss/) |
| 231 | SConv (Stateful Convolution) Block | [`sconv-block/`](./sconv-block/) |
| 232 | Score Matching | [`score-matching/`](./score-matching/) |
| 233 | SE(3)-Equivariant Attention | [`se3-equivariant-attention/`](./se3-equivariant-attention/) |
| 234 | Seasonal-Trend Decomposition | [`seasonal-trend-decomposition/`](./seasonal-trend-decomposition/) |
| 235 | Selective State Space Model (Mamba Block) | [`selective-ssm/`](./selective-ssm/) |
| 236 | Set Abstraction (PointNet++) | [`set-abstraction/`](./set-abstraction/) |
| 237 | SGD with Momentum (Heavy-Ball / Nesterov) | [`sgd-momentum/`](./sgd-momentum/) |
| 238 | Shared-Expert MoE (DeepSeekMoE) | [`shared-expert-moe/`](./shared-expert-moe/) |
| 239 | SiLU / Swish | [`silu-swish/`](./silu-swish/) |
| 240 | SimCC Coordinate Classification Head | [`simcc-head/`](./simcc-head/) |
| 241 | Sinusoidal Positional Encoding | [`sinusoidal-positional-encoding/`](./sinusoidal-positional-encoding/) |
| 242 | Hypersphere SLERP Residual Update (nGPT) | [`slerp-residual-update/`](./slerp-residual-update/) |
| 243 | Sliding-Window Attention | [`sliding-window-attention/`](./sliding-window-attention/) |
| 244 | Slimmable Supernet (Width/Structure Switching) | [`slimmable-supernet/`](./slimmable-supernet/) |
| 245 | SlowFast Two-Pathway Factorization | [`slowfast-two-pathway/`](./slowfast-two-pathway/) |
| 246 | Sparse Attention (Sparse Transformer) | [`sparse-attention/`](./sparse-attention/) |
| 247 | Spatial-Reduction Attention | [`spatial-reduction-attention/`](./spatial-reduction-attention/) |
| 248 | Spatial Transformer Module | [`spatial-transformer/`](./spatial-transformer/) |
| 249 | Spectral Normalization | [`spectral-norm/`](./spectral-norm/) |
| 250 | Spatial Pyramid Pooling — Fast (SPPF) | [`sppf/`](./sppf/) |
| 251 | Squash Function (Capsule) | [`squash-function/`](./squash-function/) |
| 252 | Squeeze-and-Excite (SE) Block | [`squeeze-and-excite/`](./squeeze-and-excite/) |
| 253 | 2D Selective Scan (SS2D / Cross-Scan) | [`ss2d-selective-scan/`](./ss2d-selective-scan/) |
| 254 | SSM-Attention Hybrid Stack | [`ssm-attention-hybrid/`](./ssm-attention-hybrid/) |
| 255 | Star Operation Block | [`star-operation-block/`](./star-operation-block/) |
| 256 | Stochastic Depth (DropPath) | [`stochastic-depth/`](./stochastic-depth/) |
| 257 | Streaming Memory Bank | [`streaming-memory-bank/`](./streaming-memory-bank/) |
| 258 | Structural Re-Parameterization | [`structural-reparameterization/`](./structural-reparameterization/) |
| 259 | Multilayer Structural Similarity Graph | [`structural-similarity-multigraph/`](./structural-similarity-multigraph/) |
| 260 | Structured State Space Model (S4) | [`structured-ssm-s4/`](./structured-ssm-s4/) |
| 261 | StyleGAN Generator Block | [`stylegan-generator/`](./stylegan-generator/) |
| 262 | SwiGLU Feed-Forward Network | [`swiglu-ffn/`](./swiglu-ffn/) |
| 263 | Swin Shifted-Window Attention Block | [`swin-shifted-window-attention/`](./swin-shifted-window-attention/) |
| 264 | SyncBatchNorm | [`syncbatchnorm/`](./syncbatchnorm/) |
| 265 | Target Attention (DIN Activation Unit) | [`target-attention-din/`](./target-attention-din/) |
| 266 | Task Token | [`task-token/`](./task-token/) |
| 267 | Recurrent Temporal BEV Attention | [`temporal-bev-attention/`](./temporal-bev-attention/) |
| 268 | Temporal Consistency Loss | [`temporal-consistency-loss/`](./temporal-consistency-loss/) |
| 269 | Temporal Random Walk Sampler | [`temporal-random-walk/`](./temporal-random-walk/) |
| 270 | Tensorized Hypergraph Convolution | [`tensorized-hypergraph-conv/`](./tensorized-hypergraph-conv/) |
| 271 | Tile-Based Splatting Rasterizer | [`tile-based-splatting/`](./tile-based-splatting/) |
| 272 | Time-Aware LSTM | [`time-aware-lstm/`](./time-aware-lstm/) |
| 273 | TimesBlock (1D-to-2D Time-Series Block) | [`timesblock/`](./timesblock/) |
| 274 | Token Shift | [`token-shift/`](./token-shift/) |
| 275 | Top-K Routing (MoE) | [`top-k-routing/`](./top-k-routing/) |
| 276 | Topological Message Passing (Simplicial/Cell Complexes) | [`topological-message-passing/`](./topological-message-passing/) |
| 277 | Transformer Decoder Block | [`transformer-decoder-block/`](./transformer-decoder-block/) |
| 278 | Transformer Encoder Block | [`transformer-encoder-block/`](./transformer-encoder-block/) |
| 279 | Triplet Loss | [`triplet-loss/`](./triplet-loss/) |
| 280 | Tube Masking | [`tube-masking/`](./tube-masking/) |
| 281 | Non-Linear Tuplewise Similarity | [`tuplewise-hyperedge-similarity/`](./tuplewise-hyperedge-similarity/) |
| 282 | Two-Stage Point Tracking | [`two-stage-point-tracking/`](./two-stage-point-tracking/) |
| 283 | Two-Way Transformer | [`two-way-transformer/`](./two-way-transformer/) |
| 284 | U-Net Encoder-Decoder with Skip Connections | [`unet-encoder-decoder/`](./unet-encoder-decoder/) |
| 285 | Universal Inverted Bottleneck (UIB) | [`universal-inverted-bottleneck/`](./universal-inverted-bottleneck/) |
| 286 | Variance-Reduced GCN Sampling | [`variance-reduced-gcn-sampling/`](./variance-reduced-gcn-sampling/) |
| 287 | VICReg Loss | [`vicreg-loss/`](./vicreg-loss/) |
| 288 | ViT Dense Prediction Adapter | [`vit-dense-adapter/`](./vit-dense-adapter/) |
| 289 | Differentiable Volume Rendering | [`volume-rendering/`](./volume-rendering/) |
| 290 | WaveNet Dilated Causal Convolution | [`wavenet-dilated-conv/`](./wavenet-dilated-conv/) |
| 291 | Weight Tying | [`weight-tying/`](./weight-tying/) |
| 292 | Wide & Deep Block | [`wide-and-deep/`](./wide-and-deep/) |
| 293 | Windowed Local Scan | [`windowed-local-scan/`](./windowed-local-scan/) |
| 294 | WSD Schedule (Warmup–Stable–Decay) | [`wsd-lr-schedule/`](./wsd-lr-schedule/) |
| 295 | Xavier (Glorot) Initialization | [`xavier-init/`](./xavier-init/) |
| 296 | xLSTM Block (sLSTM / mLSTM) | [`xlstm-block/`](./xlstm-block/) |
| 297 | YaRN RoPE Scaling | [`yarn-rope-scaling/`](./yarn-rope-scaling/) |
