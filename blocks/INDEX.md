# Basic NN Building Blocks — Index

A catalog of **distinct, reusable neural network building blocks** extracted from the real architectures in [`../architectures/`](../architectures/). Each block has its own folder with a `README.md` documenting its design philosophy, functionality, which models use it, key features, and evolution.

**Total blocks: 334** — no duplication. Each block is a distinct structural or functional unit that appears in one or more real architectures.

---

## Block Catalog

| # | Block | Folder |
|---|-------|--------|
| 1 | 2D Selective Scan (SS2D / Cross-Scan) | [`ss2d-selective-scan/`](./ss2d-selective-scan/) |
| 2 | 3D Frustum Position Embedding | [`frustum-3d-position-embedding/`](./frustum-3d-position-embedding/) |
| 3 | 3D Gaussian Primitives | [`3d-gaussian-primitives/`](./3d-gaussian-primitives/) |
| 4 | Action-Conditioned Latent Dynamics | [`action-conditioned-latent-dynamics/`](./action-conditioned-latent-dynamics/) |
| 5 | ActNorm (Activation Normalization) | [`actnorm/`](./actnorm/) |
| 6 | AdamW (Adam with Decoupled Weight Decay) | [`adamw-optimizer/`](./adamw-optimizer/) |
| 7 | Adaptive Adjacency Learning | [`adaptive-adjacency-learning/`](./adaptive-adjacency-learning/) |
| 8 | Adaptive Computation Time (ACT) | [`adaptive-computation-time/`](./adaptive-computation-time/) |
| 9 | Adaptive LayerNorm (adaLN / adaLN-Zero) | [`adaptive-layernorm/`](./adaptive-layernorm/) |
| 10 | ALiBi (Attention with Linear Biases) | [`alibi/`](./alibi/) |
| 11 | All-MLP Segmentation Decoder | [`all-mlp-decoder/`](./all-mlp-decoder/) |
| 12 | All-Pairs Correlation Pyramid | [`all-pairs-correlation-pyramid/`](./all-pairs-correlation-pyramid/) |
| 13 | Ambiguity-Aware Multi-Mask Output | [`multi-mask-iou-head/`](./multi-mask-iou-head/) |
| 14 | Anchor-Free Center Heatmap Head | [`center-heatmap-head/`](./center-heatmap-head/) |
| 15 | Anti-Aliased Activation (BlurPool) | [`anti-aliased-activation/`](./anti-aliased-activation/) |
| 16 | Antisymmetric Edge-Local Frame GNN | [`antisymmetric-edge-frame-gnn/`](./antisymmetric-edge-frame-gnn/) |
| 17 | AnyRes Tiling (LLaVA-NeXT) | [`anyres-tiling/`](./anyres-tiling/) |
| 18 | ArcFace Loss | [`arcface-loss/`](./arcface-loss/) |
| 19 | Area Attention | [`area-attention/`](./area-attention/) |
| 20 | Associative Memory Transformer (ARMT) | [`associative-memory-transformer/`](./associative-memory-transformer/) |
| 21 | Atrous Spatial Pyramid Pooling (ASPP) | [`aspp-decoder/`](./aspp-decoder/) |
| 22 | Auxiliary-Loss-Free Load Balancing | [`auxiliary-loss-free-balancing/`](./auxiliary-loss-free-balancing/) |
| 23 | Axial-Planar 3D Convolution | [`axial-planar-3d-conv/`](./axial-planar-3d-conv/) |
| 24 | Backpack Sense Vectors | [`backpack-sense-vectors/`](./backpack-sense-vectors/) |
| 25 | Barlow Twins Loss | [`barlow-twins-loss/`](./barlow-twins-loss/) |
| 26 | BatchNorm (Batch Normalization) | [`batch-norm/`](./batch-norm/) |
| 27 | BEV Query Grid | [`bev-query-grid/`](./bev-query-grid/) |
| 28 | Bi-Interaction Pooling (NFM) | [`bi-interaction-pooling/`](./bi-interaction-pooling/) |
| 29 | Bidirectional Selective SSM (Vim) | [`bidirectional-ssm/`](./bidirectional-ssm/) |
| 30 | BiFPN (Weighted Bidirectional FPN) | [`bifpn/`](./bifpn/) |
| 31 | Block-Causal Diffusion Generation | [`block-causal-diffusion/`](./block-causal-diffusion/) |
| 32 | Bottleneck Residual Block | [`bottleneck-residual/`](./bottleneck-residual/) |
| 33 | Camera-Aware Depth Conditioning | [`camera-aware-depth-conditioning/`](./camera-aware-depth-conditioning/) |
| 34 | Candidate-in-Sequence | [`candidate-in-sequence/`](./candidate-in-sequence/) |
| 35 | Canonical Camera Space Transform | [`canonical-camera-transform/`](./canonical-camera-transform/) |
| 36 | Center Loss | [`center-loss/`](./center-loss/) |
| 37 | Center-Ness Head (FCOS) | [`center-ness-head/`](./center-ness-head/) |
| 38 | Channel-Split Shuffle Block | [`channel-split-shuffle-block/`](./channel-split-shuffle-block/) |
| 39 | Chebyshev Graph Convolution | [`chebyshev-graph-conv/`](./chebyshev-graph-conv/) |
| 40 | Classifier-Free Guidance | [`classifier-free-guidance/`](./classifier-free-guidance/) |
| 41 | Codebook Quantization of Latents | [`codebook-quantization/`](./codebook-quantization/) |
| 42 | Complex MIMO State-Space Mixer (Mamba-3) | [`complex-mimo-ssm/`](./complex-mimo-ssm/) |
| 43 | Compressed Interaction Network (CIN) | [`cin/`](./cin/) |
| 44 | Confidence-Weighted Regression Loss | [`confidence-weighted-loss/`](./confidence-weighted-loss/) |
| 45 | Conformer Block (Conv + Attention) | [`conformer-block/`](./conformer-block/) |
| 46 | Continuous Next-Embedding Prediction | [`continuous-embedding-prediction/`](./continuous-embedding-prediction/) |
| 47 | Continuous-Filter Convolution (SchNet) | [`continuous-filter-convolution/`](./continuous-filter-convolution/) |
| 48 | Contrastive Denoising Queries (CDN) | [`contrastive-denoising-queries/`](./contrastive-denoising-queries/) |
| 49 | Contrastive Dual-Encoder (CLIP) | [`contrastive-dual-encoder/`](./contrastive-dual-encoder/) |
| 50 | Conv-BN-ReLU Block | [`conv-bn-relu/`](./conv-bn-relu/) |
| 51 | Conv1D Feature Extractor (Audio Stem) | [`conv1d-audio-stem/`](./conv1d-audio-stem/) |
| 52 | Convex Upsampling | [`convex-upsampling/`](./convex-upsampling/) |
| 53 | ConvGRU Recurrent Update Operator | [`convgru-update-operator/`](./convgru-update-operator/) |
| 54 | ConvNeXt Block | [`convnext-block/`](./convnext-block/) |
| 55 | Convolution-Based Graph Diffusion | [`graph-diffusion-convolution/`](./graph-diffusion-convolution/) |
| 56 | Convolutional Positional Embedding | [`convolutional-positional-embedding/`](./convolutional-positional-embedding/) |
| 57 | Cooperative Action-Environment GNN | [`cooperative-action-network/`](./cooperative-action-network/) |
| 58 | Copy Mechanism | [`copy-mechanism/`](./copy-mechanism/) |
| 59 | Cosine Learning Rate Schedule | [`cosine-lr-schedule/`](./cosine-lr-schedule/) |
| 60 | Coupling Layer (RealNVP) | [`coupling-layer/`](./coupling-layer/) |
| 61 | Cross-Attention Block | [`cross-attention/`](./cross-attention/) |
| 62 | Cross-Dimension Attention (Crossformer) | [`cross-dimension-attention/`](./cross-dimension-attention/) |
| 63 | Cross-Graph Attention Matching | [`cross-graph-attention/`](./cross-graph-attention/) |
| 64 | Cross-Shaped Window Attention | [`cswin-attention/`](./cswin-attention/) |
| 65 | Cross-Stage-Partial (CSP) Stage | [`csp-block/`](./csp-block/) |
| 66 | Customized Gate Control (CGC / PLE) | [`cgc-expert-separation/`](./cgc-expert-separation/) |
| 67 | DARTS Cell (Differentiable NAS) | [`darts-cell/`](./darts-cell/) |
| 68 | DDPM Noise Schedule | [`ddpm-noise-schedule/`](./ddpm-noise-schedule/) |
| 69 | Decoder-Only Block (GPT/LLaMA style) | [`decoder-only-block/`](./decoder-only-block/) |
| 70 | Decoupled Anchor-Free Detection Head | [`decoupled-detection-head/`](./decoupled-detection-head/) |
| 71 | Decoupled Cross-Attention (IP-Adapter) | [`decoupled-cross-attention/`](./decoupled-cross-attention/) |
| 72 | Deep & Cross Network (DCN / DCN-v2) | [`deep-cross-network/`](./deep-cross-network/) |
| 73 | Deep Graph Infomax (DGI) | [`deep-graph-infomax/`](./deep-graph-infomax/) |
| 74 | DeepFM Block | [`deepfm/`](./deepfm/) |
| 75 | Deformable Convolution (DCN-v2) | [`deformable-convolution/`](./deformable-convolution/) |
| 76 | Dense Connectivity Block (DenseNet) | [`dense-connectivity/`](./dense-connectivity/) |
| 77 | Depthwise-Separable Convolution | [`depthwise-separable-conv/`](./depthwise-separable-conv/) |
| 78 | Dice Loss (and Soft Dice) | [`dice-loss/`](./dice-loss/) |
| 79 | Differentiable Volume Rendering | [`volume-rendering/`](./volume-rendering/) |
| 80 | Differential Attention | [`differential-attention/`](./differential-attention/) |
| 81 | Diffusion U-Net (Latent Diffusion Denoiser) | [`diffusion-unet/`](./diffusion-unet/) |
| 82 | Directional Message Passing (DimeNet) | [`directional-message-passing/`](./directional-message-passing/) |
| 83 | Disentangled Attention (DeBERTa) | [`disentangled-attention/`](./disentangled-attention/) |
| 84 | DiT Block (Diffusion Transformer) | [`dit-block/`](./dit-block/) |
| 85 | Divided Space-Time Attention | [`divided-space-time-attention/`](./divided-space-time-attention/) |
| 86 | DPT Reassembly + Fusion Decoder | [`dpt-decoder/`](./dpt-decoder/) |
| 87 | Dynamic Routing (Capsule Network) | [`dynamic-routing/`](./dynamic-routing/) |
| 88 | E(3)-Equivariant MLP | [`e3-equivariant-mlp/`](./e3-equivariant-mlp/) |
| 89 | Edge Memory Message Passing | [`edge-memory-message-passing/`](./edge-memory-message-passing/) |
| 90 | Edge-Feature Attention Injection | [`edge-feature-attention/`](./edge-feature-attention/) |
| 91 | EEG Conv Tokenizer Stem | [`eeg-conv-stem/`](./eeg-conv-stem/) |
| 92 | Embedding Distillation | [`embedding-distillation/`](./embedding-distillation/) |
| 93 | Entire-Space Multi-Task Towers | [`entire-space-multitask/`](./entire-space-multitask/) |
| 94 | Equalized Learning Rate (GAN) | [`equalized-lr/`](./equalized-lr/) |
| 95 | Evoformer Block | [`evoformer-block/`](./evoformer-block/) |
| 96 | Expert Choice Routing | [`expert-choice-routing/`](./expert-choice-routing/) |
| 97 | Exponential Moving Average of Weights (EMA / SWA) | [`ema-weights/`](./ema-weights/) |
| 98 | External Memory Bank | [`external-memory-bank/`](./external-memory-bank/) |
| 99 | Factorized Embedding Parameterization (ALBERT) | [`factorized-embedding/`](./factorized-embedding/) |
| 100 | Factorized Space-Time Encoder | [`factorised-space-time-encoder/`](./factorised-space-time-encoder/) |
| 101 | Fast Feedforward Network (FFF) | [`fast-feedforward-network/`](./fast-feedforward-network/) |
| 102 | FCN Mask Branch | [`fcn-mask-head/`](./fcn-mask-head/) |
| 103 | Feature Pyramid Network (FPN) | [`feature-pyramid-network/`](./feature-pyramid-network/) |
| 104 | Fine-Grained Distribution Refinement (FDR) | [`fine-grained-distribution-refinement/`](./fine-grained-distribution-refinement/) |
| 105 | Flow Matching Objective | [`flow-matching/`](./flow-matching/) |
| 106 | FNet Block (Fourier Transform) | [`fnet-block/`](./fnet-block/) |
| 107 | Focal Loss | [`focal-loss/`](./focal-loss/) |
| 108 | Fourier Attention (FEDformer) | [`fourier-attention/`](./fourier-attention/) |
| 109 | Fourier Positional Encoding (Coordinate Mapping) | [`fourier-positional-encoding/`](./fourier-positional-encoding/) |
| 110 | Frustum-to-BEV Voxel Pooling (Lift-Splat) | [`frustum-voxel-pooling/`](./frustum-voxel-pooling/) |
| 111 | GAN Minimax Block (Generator + Discriminator) | [`gan-minimax/`](./gan-minimax/) |
| 112 | Gated Cross-Attention (Multimodal Fusion) | [`gated-cross-attention/`](./gated-cross-attention/) |
| 113 | Gated Recurrent Unit (GRU) | [`gru-cell/`](./gru-cell/) |
| 114 | Gaussian Node Embedding | [`gaussian-node-embedding/`](./gaussian-node-embedding/) |
| 115 | GeGLU Feed-Forward Network | [`geglu-ffn/`](./geglu-ffn/) |
| 116 | GELU (Gaussian Error Linear Unit) | [`gelu/`](./gelu/) |
| 117 | Geometry Encoding Volume (GEV) | [`geometry-encoding-volume/`](./geometry-encoding-volume/) |
| 118 | Ghost Module | [`ghost-module/`](./ghost-module/) |
| 119 | GIoU / CIoU / DIoU Loss | [`giou-loss/`](./giou-loss/) |
| 120 | Global Optimal Localization Self-Distillation (GoLSD) | [`go-lsd-self-distillation/`](./go-lsd-self-distillation/) |
| 121 | Global Response Normalization (GRN) | [`global-response-normalization/`](./global-response-normalization/) |
| 122 | Global-Local Feature Fusion (HQ-Features) | [`global-local-feature-fusion/`](./global-local-feature-fusion/) |
| 123 | Global/Local Patch Decoder (MegaByte) | [`global-local-patch-decoder/`](./global-local-patch-decoder/) |
| 124 | GMF + MLP Fusion | [`gmf-mlp-fusion/`](./gmf-mlp-fusion/) |
| 125 | gMLP Block (Spatial Gating) | [`gmlp-block/`](./gmlp-block/) |
| 126 | Gradient Checkpointing (Activation Recomputation) | [`gradient-checkpointing/`](./gradient-checkpointing/) |
| 127 | Gram Anchoring Loss | [`gram-anchoring-loss/`](./gram-anchoring-loss/) |
| 128 | Graph Attention Layer (GAT) | [`graph-attention-layer/`](./graph-attention-layer/) |
| 129 | Graph Coarsening Hierarchy | [`graph-coarsening-hierarchy/`](./graph-coarsening-hierarchy/) |
| 130 | Graph Convolution Layer (GCN) | [`graph-convolution-layer/`](./graph-convolution-layer/) |
| 131 | Graph Pooling / Unpooling (gPool/gUnpool) | [`gpool-unpool/`](./gpool-unpool/) |
| 132 | Graph Prompt Tokens | [`graph-prompting/`](./graph-prompting/) |
| 133 | Graph Propagation Attention (GPA) | [`graph-propagation-attention/`](./graph-propagation-attention/) |
| 134 | Graph Softmax | [`graph-softmax/`](./graph-softmax/) |
| 135 | Graphlet Degree Vector (GDV) | [`graphlet-degree-vector/`](./graphlet-degree-vector/) |
| 136 | GraphSAGE Neighborhood Aggregator | [`graphsage-aggregator/`](./graphsage-aggregator/) |
| 137 | Group Relative Policy Optimization (GRPO) | [`grpo/`](./grpo/) |
| 138 | Grouped-Query Attention (GQA) / Multi-Query Attention (MQA) | [`grouped-query-attention/`](./grouped-query-attention/) |
| 139 | GroupNorm | [`groupnorm/`](./groupnorm/) |
| 140 | HiFi-GAN Generator | [`hifi-gan-generator/`](./hifi-gan-generator/) |
| 141 | High-Order Proximity Update | [`high-order-proximity-update/`](./high-order-proximity-update/) |
| 142 | Highway Network | [`highway-network/`](./highway-network/) |
| 143 | Hungarian (Bipartite) Matching Loss | [`hungarian-matching-loss/`](./hungarian-matching-loss/) |
| 144 | Hypersphere SLERP Residual Update (nGPT) | [`slerp-residual-update/`](./slerp-residual-update/) |
| 145 | Inception Module | [`inception-module/`](./inception-module/) |
| 146 | InfoNCE (Contrastive) Loss | [`infonce-contrastive-loss/`](./infonce-contrastive-loss/) |
| 147 | Instance Appearance Adapter | [`instance-appearance-embedding/`](./instance-appearance-embedding/) |
| 148 | Interleaved NoPE (iRoPE) | [`interleaved-nope/`](./interleaved-nope/) |
| 149 | Intermediate-Layer Feature Tap | [`intermediate-layer-tap/`](./intermediate-layer-tap/) |
| 150 | Invariant Point Attention (IPA) | [`invariant-point-attention/`](./invariant-point-attention/) |
| 151 | Inverted Residual Block (MobileNetV2) | [`inverted-residual/`](./inverted-residual/) |
| 152 | Invertible 1x1 Convolution (Glow) | [`invertible-1x1-conv/`](./invertible-1x1-conv/) |
| 153 | Joint Detection-Tracking Decoder Queries | [`joint-detection-tracking-queries/`](./joint-detection-tracking-queries/) |
| 154 | Joint Space-Time Track Attention | [`joint-spacetime-attention/`](./joint-spacetime-attention/) |
| 155 | k-Step PPMI-SVD Embedding | [`ppmi-svd-embedding/`](./ppmi-svd-embedding/) |
| 156 | Kaiming (He) Initialization | [`kaiming-init/`](./kaiming-init/) |
| 157 | KAN Layer (Kolmogorov-Arnold Network) | [`kan-layer/`](./kan-layer/) |
| 158 | Keypoint Heatmap Head | [`keypoint-heatmap-head/`](./keypoint-heatmap-head/) |
| 159 | Label Smoothing | [`label-smoothing/`](./label-smoothing/) |
| 160 | Language-Guided Query Selection | [`language-guided-query-selection/`](./language-guided-query-selection/) |
| 161 | Laplacian Positional Encoding | [`laplacian-positional-encoding/`](./laplacian-positional-encoding/) |
| 162 | Large-Kernel Convolution (RepLKNet) | [`large-kernel-conv/`](./large-kernel-conv/) |
| 163 | LayerNorm (Layer Normalization) | [`layer-norm/`](./layer-norm/) |
| 164 | Learned Absolute Position Embedding | [`learned-position-embedding/`](./learned-position-embedding/) |
| 165 | Learned Attention Sink | [`attention-sink/`](./attention-sink/) |
| 166 | Length Regulator | [`length-regulator/`](./length-regulator/) |
| 167 | LightGCN Propagation | [`lightgcn-propagation/`](./lightgcn-propagation/) |
| 168 | Linear Attention | [`linear-attention/`](./linear-attention/) |
| 169 | Linear LR Warmup | [`linear-warmup/`](./linear-warmup/) |
| 170 | Lion Optimizer | [`lion-optimizer/`](./lion-optimizer/) |
| 171 | Local/Global Interleaved Attention | [`local-global-interleaved-attention/`](./local-global-interleaved-attention/) |
| 172 | Location-Sensitive Attention | [`location-sensitive-attention/`](./location-sensitive-attention/) |
| 173 | Look-Forward-Twice Box Supervision | [`look-forward-twice/`](./look-forward-twice/) |
| 174 | LSTM Cell | [`lstm-cell/`](./lstm-cell/) |
| 175 | MambaVision Mixer | [`mambavision-mixer/`](./mambavision-mixer/) |
| 176 | Masked Attention | [`masked-attention/`](./masked-attention/) |
| 177 | Masked Autoencoder (MAE) | [`masked-autoencoder/`](./masked-autoencoder/) |
| 178 | Masked Convolution (PixelCNN) | [`masked-convolution/`](./masked-convolution/) |
| 179 | Masked Language Modeling (Cloze) | [`masked-language-modeling/`](./masked-language-modeling/) |
| 180 | Masked Latent Prediction (JEPA) | [`masked-latent-prediction/`](./masked-latent-prediction/) |
| 181 | MBConv Block | [`mbconv/`](./mbconv/) |
| 182 | Memory Token Segment Recurrence | [`memory-tokens/`](./memory-tokens/) |
| 183 | Meta-Path Guided Random Walk | [`metapath-random-walk/`](./metapath-random-walk/) |
| 184 | MetaFormer Block | [`metaformer-block/`](./metaformer-block/) |
| 185 | Minibatch Standard Deviation | [`minibatch-stddev/`](./minibatch-stddev/) |
| 186 | Mixed Precision (AMP / BF16 / FP16) | [`mixed-precision-amp/`](./mixed-precision-amp/) |
| 187 | Mixture Density Network | [`mixture-density-network/`](./mixture-density-network/) |
| 188 | Mixture-of-Experts (MoE) Layer | [`mixture-of-experts/`](./mixture-of-experts/) |
| 189 | Mixture-of-Laplace Loss | [`mixture-of-laplace-loss/`](./mixture-of-laplace-loss/) |
| 190 | MLP Projector Bridge (LLaVA-style) | [`mlp-projector-bridge/`](./mlp-projector-bridge/) |
| 191 | MLP-Mixer Block | [`mlp-mixer-block/`](./mlp-mixer-block/) |
| 192 | MobileViT Block | [`mobilevit-block/`](./mobilevit-block/) |
| 193 | Multi-Arity Expand-Reduce Hyperedge Layer | [`multi-arity-hyperedge-layer/`](./multi-arity-hyperedge-layer/) |
| 194 | Multi-Axis Attention | [`multi-axis-attention/`](./multi-axis-attention/) |
| 195 | Multi-Gate Mixture-of-Experts (MMoE) | [`multi-gate-moe/`](./multi-gate-moe/) |
| 196 | Multi-Head Attention (Full / Dense) | [`multi-head-attention/`](./multi-head-attention/) |
| 197 | Multi-Head Latent Attention (MLA) | [`multi-head-latent-attention/`](./multi-head-latent-attention/) |
| 198 | Multi-Resolution Parallel Streams (HRNet) | [`multi-resolution-parallel-streams/`](./multi-resolution-parallel-streams/) |
| 199 | Multi-Scale Deformable Attention (MSDeformAttn) | [`multi-scale-deformable-attention/`](./multi-scale-deformable-attention/) |
| 200 | Multi-Token Prediction (MTP) | [`multi-token-prediction/`](./multi-token-prediction/) |
| 201 | Multi-View Graph Fusion | [`multi-view-graph-fusion/`](./multi-view-graph-fusion/) |
| 202 | Multilayer Structural Similarity Graph | [`structural-similarity-multigraph/`](./structural-similarity-multigraph/) |
| 203 | Multimodal RoPE (M-RoPE) | [`m-rope/`](./m-rope/) |
| 204 | MultKAN Multiplication Node | [`multiplication-node/`](./multiplication-node/) |
| 205 | Mutual Selective Attention | [`mutual-attention/`](./mutual-attention/) |
| 206 | N-BEATS Block | [`n-beats-block/`](./n-beats-block/) |
| 207 | Native Sparse Attention (NSA) | [`native-sparse-attention/`](./native-sparse-attention/) |
| 208 | Nested Skip Decoder (UNet++) | [`nested-skip-decoder/`](./nested-skip-decoder/) |
| 209 | Neural Filter Block (UnivNet) | [`neural-filter-block/`](./neural-filter-block/) |
| 210 | Neural ODE Block | [`neural-ode-block/`](./neural-ode-block/) |
| 211 | Neural Synchronization Representation | [`neural-synchronization/`](./neural-synchronization/) |
| 212 | Neuron-Level Model (NLM) | [`neuron-level-model/`](./neuron-level-model/) |
| 213 | Non-Linear Tuplewise Similarity | [`tuplewise-hyperedge-similarity/`](./tuplewise-hyperedge-similarity/) |
| 214 | Nyströmformer Attention | [`nystromformer-attention/`](./nystromformer-attention/) |
| 215 | Object-Query Memory Queue | [`query-memory-queue/`](./query-memory-queue/) |
| 216 | One-Shot Probabilistic Graph Decoder | [`one-shot-graph-decoder/`](./one-shot-graph-decoder/) |
| 217 | OneCycle LR Schedule | [`onecycle-lr-schedule/`](./onecycle-lr-schedule/) |
| 218 | Orthogonal Regularization | [`orthogonal-regularization/`](./orthogonal-regularization/) |
| 219 | Pairwise Dot-Interaction | [`pairwise-dot-interaction/`](./pairwise-dot-interaction/) |
| 220 | Pairwise Ranking Loss (BPR / Margin / Hop-Distance) | [`pairwise-ranking-loss/`](./pairwise-ranking-loss/) |
| 221 | Pairwise Sigmoid Contrastive Loss | [`pairwise-sigmoid-contrastive-loss/`](./pairwise-sigmoid-contrastive-loss/) |
| 222 | Parallel Attention+MLP Block (GPT-NeoX) | [`parallel-residual-block/`](./parallel-residual-block/) |
| 223 | Partial Convolution (PConv) | [`partial-convolution/`](./partial-convolution/) |
| 224 | Patch Embedding (ViT) | [`patch-embedding/`](./patch-embedding/) |
| 225 | PatchGAN Discriminator | [`patchgan-discriminator/`](./patchgan-discriminator/) |
| 226 | PatchTST Block (Channel-Independent Patch Transformer) | [`patchtst-block/`](./patchtst-block/) |
| 227 | Path-Based Message Passing | [`path-message-passing/`](./path-message-passing/) |
| 228 | Per-Region Optimal Supervision | [`per-region-optimal-supervision/`](./per-region-optimal-supervision/) |
| 229 | Perceiver Resampler | [`perceiver-resampler/`](./perceiver-resampler/) |
| 230 | Personalized PageRank Propagation (APPNP) | [`ppr-propagation/`](./ppr-propagation/) |
| 231 | Pixel Normalization (GAN) | [`pixelnorm/`](./pixelnorm/) |
| 232 | Poincaré Ball Embedding | [`poincare-embedding/`](./poincare-embedding/) |
| 233 | Point Transformer Attention | [`point-transformer-attention/`](./point-transformer-attention/) |
| 234 | Pointmap Regression | [`pointmap-regression/`](./pointmap-regression/) |
| 235 | PointNet Shared MLP | [`pointnet-mlp/`](./pointnet-mlp/) |
| 236 | Position-Dependent Hyperedge Embedding | [`hyperedge-position-embedding/`](./hyperedge-position-embedding/) |
| 237 | Presence / Occlusion Head | [`presence-occlusion-head/`](./presence-occlusion-head/) |
| 238 | Prototype Mask Head (YOLACT-style) | [`prototype-mask-head/`](./prototype-mask-head/) |
| 239 | Q-Former (Learned Query Bridge) | [`q-former/`](./q-former/) |
| 240 | QK-Norm | [`qk-norm/`](./qk-norm/) |
| 241 | Quantile Embedding (IQN) | [`quantile-embedding/`](./quantile-embedding/) |
| 242 | Rational-Quadratic Spline Flow | [`rational-quadratic-spline/`](./rational-quadratic-spline/) |
| 243 | Reconciled Polynomial Layer (RPN) | [`reconciled-polynomial-layer/`](./reconciled-polynomial-layer/) |
| 244 | Recurrent Temporal BEV Attention | [`temporal-bev-attention/`](./temporal-bev-attention/) |
| 245 | Recurrent World Model + Imagination | [`recurrent-world-model/`](./recurrent-world-model/) |
| 246 | Reformer Attention (LSH) | [`reformer-attention/`](./reformer-attention/) |
| 247 | Region-Text Contrastive Head | [`region-text-contrastive-head/`](./region-text-contrastive-head/) |
| 248 | Relative Position Bias | [`relative-position-bias/`](./relative-position-bias/) |
| 249 | Replaced Token Detection (ELECTRA) | [`replaced-token-detection/`](./replaced-token-detection/) |
| 250 | Residual Block (Identity Skip) | [`residual-block/`](./residual-block/) |
| 251 | Residual Vector Quantization (RVQ) | [`residual-vector-quantization/`](./residual-vector-quantization/) |
| 252 | Residual-Depth-Scaled Init / muP Scaled Residuals | [`residual-scaling-init/`](./residual-scaling-init/) |
| 253 | RL Attention-Guided Graph Walk | [`rl-attention-guided-graph-walk/`](./rl-attention-guided-graph-walk/) |
| 254 | RLHF PPO Pipeline | [`rlhf-ppo-pipeline/`](./rlhf-ppo-pipeline/) |
| 255 | RMSNorm | [`rmsnorm/`](./rmsnorm/) |
| 256 | RoIAlign | [`roialign/`](./roialign/) |
| 257 | Rotary Position Embedding (RoPE) | [`rope/`](./rope/) |
| 258 | RPN (Region Proposal Network) | [`region-proposal-network/`](./region-proposal-network/) |
| 259 | RSSM (Recurrent State-Space Model) | [`rssm/`](./rssm/) |
| 260 | RWKV Channel Mixing | [`rwkv-channel-mixing/`](./rwkv-channel-mixing/) |
| 261 | RWKV Time-Mixing Block | [`rwkv-time-mixing/`](./rwkv-time-mixing/) |
| 262 | Scale-and-Shift-Invariant Depth Loss | [`scale-shift-invariant-loss/`](./scale-shift-invariant-loss/) |
| 263 | SConv (Stateful Convolution) Block | [`sconv-block/`](./sconv-block/) |
| 264 | Score Matching | [`score-matching/`](./score-matching/) |
| 265 | SE(3)-Equivariant Attention | [`se3-equivariant-attention/`](./se3-equivariant-attention/) |
| 266 | Seasonal-Trend Decomposition | [`seasonal-trend-decomposition/`](./seasonal-trend-decomposition/) |
| 267 | Selective State Space Model (Mamba Block) | [`selective-ssm/`](./selective-ssm/) |
| 268 | Set Abstraction (PointNet++) | [`set-abstraction/`](./set-abstraction/) |
| 269 | SGD with Momentum (Heavy-Ball / Nesterov) | [`sgd-momentum/`](./sgd-momentum/) |
| 270 | Shared-Expert MoE (DeepSeekMoE) | [`shared-expert-moe/`](./shared-expert-moe/) |
| 271 | SiLU / Swish | [`silu-swish/`](./silu-swish/) |
| 272 | SimCC Coordinate Classification Head | [`simcc-head/`](./simcc-head/) |
| 273 | Sinusoidal Positional Encoding | [`sinusoidal-positional-encoding/`](./sinusoidal-positional-encoding/) |
| 274 | Sliding-Window Attention | [`sliding-window-attention/`](./sliding-window-attention/) |
| 275 | Slimmable Supernet (Width/Structure Switching) | [`slimmable-supernet/`](./slimmable-supernet/) |
| 276 | SlowFast Two-Pathway Factorization | [`slowfast-two-pathway/`](./slowfast-two-pathway/) |
| 277 | Snake Activation (Periodic) | [`snake-activation/`](./snake-activation/) |
| 278 | Sparse Attention (Sparse Transformer) | [`sparse-attention/`](./sparse-attention/) |
| 279 | Sparse/Dense Prompt Encoder | [`prompt-encoder/`](./prompt-encoder/) |
| 280 | Spatial Pyramid Pooling — Fast (SPPF) | [`sppf/`](./sppf/) |
| 281 | Spatial Transformer Module | [`spatial-transformer/`](./spatial-transformer/) |
| 282 | Spatial-Reduction Attention | [`spatial-reduction-attention/`](./spatial-reduction-attention/) |
| 283 | Spectral Normalization | [`spectral-norm/`](./spectral-norm/) |
| 284 | Squash Function (Capsule) | [`squash-function/`](./squash-function/) |
| 285 | Squeeze-and-Excite (SE) Block | [`squeeze-and-excite/`](./squeeze-and-excite/) |
| 286 | SSM-Attention Hybrid Stack | [`ssm-attention-hybrid/`](./ssm-attention-hybrid/) |
| 287 | Star Operation Block | [`star-operation-block/`](./star-operation-block/) |
| 288 | Stochastic Depth (DropPath) | [`stochastic-depth/`](./stochastic-depth/) |
| 289 | Straight-Through Gumbel-Softmax | [`gumbel-softmax-straight-through/`](./gumbel-softmax-straight-through/) |
| 290 | Streaming Memory Bank | [`streaming-memory-bank/`](./streaming-memory-bank/) |
| 291 | Structural Re-Parameterization | [`structural-reparameterization/`](./structural-reparameterization/) |
| 292 | Structured State Space Model (S4) | [`structured-ssm-s4/`](./structured-ssm-s4/) |
| 293 | StyleGAN Generator Block | [`stylegan-generator/`](./stylegan-generator/) |
| 294 | SwiGLU Feed-Forward Network | [`swiglu-ffn/`](./swiglu-ffn/) |
| 295 | Swin Shifted-Window Attention Block | [`swin-shifted-window-attention/`](./swin-shifted-window-attention/) |
| 296 | SyncBatchNorm | [`syncbatchnorm/`](./syncbatchnorm/) |
| 297 | Target Attention (DIN Activation Unit) | [`target-attention-din/`](./target-attention-din/) |
| 298 | Task Token | [`task-token/`](./task-token/) |
| 299 | Temporal Consistency Loss | [`temporal-consistency-loss/`](./temporal-consistency-loss/) |
| 300 | Temporal Random Walk Sampler | [`temporal-random-walk/`](./temporal-random-walk/) |
| 301 | Tensor Product Convolution (TFN) | [`tensor-product-convolution/`](./tensor-product-convolution/) |
| 302 | Tensorized Hypergraph Convolution | [`tensorized-hypergraph-conv/`](./tensorized-hypergraph-conv/) |
| 303 | Tile-Based Splatting Rasterizer | [`tile-based-splatting/`](./tile-based-splatting/) |
| 304 | Time-Aware LSTM | [`time-aware-lstm/`](./time-aware-lstm/) |
| 305 | TimesBlock (1D-to-2D Time-Series Block) | [`timesblock/`](./timesblock/) |
| 306 | Token Shift | [`token-shift/`](./token-shift/) |
| 307 | Top-K Routing (MoE) | [`top-k-routing/`](./top-k-routing/) |
| 308 | Topological Message Passing (Simplicial/Cell Complexes) | [`topological-message-passing/`](./topological-message-passing/) |
| 309 | Transformer Decoder Block | [`transformer-decoder-block/`](./transformer-decoder-block/) |
| 310 | Transformer Encoder Block | [`transformer-encoder-block/`](./transformer-encoder-block/) |
| 311 | Tri-Level Hypergraph Contrast | [`hypergraph-contrastive-learning/`](./hypergraph-contrastive-learning/) |
| 312 | Triplet Loss | [`triplet-loss/`](./triplet-loss/) |
| 313 | Tube Masking | [`tube-masking/`](./tube-masking/) |
| 314 | Two-Stage Point Tracking | [`two-stage-point-tracking/`](./two-stage-point-tracking/) |
| 315 | Two-Stream Attention (XLNet) | [`two-stream-attention/`](./two-stream-attention/) |
| 316 | Two-Way Transformer | [`two-way-transformer/`](./two-way-transformer/) |
| 317 | U-Net Encoder-Decoder with Skip Connections | [`unet-encoder-decoder/`](./unet-encoder-decoder/) |
| 318 | Uncertainty-Minimal Query Selection | [`quality-aware-query-selection/`](./quality-aware-query-selection/) |
| 319 | Universal Inverted Bottleneck (UIB) | [`universal-inverted-bottleneck/`](./universal-inverted-bottleneck/) |
| 320 | V-Trace (IMPALA) | [`v-trace/`](./v-trace/) |
| 321 | Variance Adaptor (FastSpeech 2) | [`variance-adaptor/`](./variance-adaptor/) |
| 322 | Variance-Reduced GCN Sampling | [`variance-reduced-gcn-sampling/`](./variance-reduced-gcn-sampling/) |
| 323 | VICReg Loss | [`vicreg-loss/`](./vicreg-loss/) |
| 324 | Visual Expert (CogVLM) | [`visual-expert/`](./visual-expert/) |
| 325 | ViT Dense Prediction Adapter | [`vit-dense-adapter/`](./vit-dense-adapter/) |
| 326 | WaveNet Dilated Causal Convolution | [`wavenet-dilated-conv/`](./wavenet-dilated-conv/) |
| 327 | Weight Tying | [`weight-tying/`](./weight-tying/) |
| 328 | Wide & Deep Block | [`wide-and-deep/`](./wide-and-deep/) |
| 329 | Windowed Local Scan | [`windowed-local-scan/`](./windowed-local-scan/) |
| 330 | WSD Schedule (Warmup–Stable–Decay) | [`wsd-lr-schedule/`](./wsd-lr-schedule/) |
| 331 | Xavier (Glorot) Initialization | [`xavier-init/`](./xavier-init/) |
| 332 | xLSTM Block (sLSTM / mLSTM) | [`xlstm-block/`](./xlstm-block/) |
| 333 | YaRN RoPE Scaling | [`yarn-rope-scaling/`](./yarn-rope-scaling/) |
| 334 | Zero Convolution (ControlNet) | [`zero-convolution/`](./zero-convolution/) |
