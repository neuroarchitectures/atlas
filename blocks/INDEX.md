# Basic NN Building Blocks — Index

A catalog of **distinct, reusable neural network building blocks** extracted from the real architectures in [`../architectures/`](../architectures/). Each block has its own folder with a `README.md` documenting its design philosophy, functionality, which models use it, key features, and evolution.

**Total blocks: 251** — no duplication. Each block is a distinct structural or functional unit that appears in one or more real architectures. Sections 1–12 are **structural** blocks (layers/modules); section 13 is **training primitives** (objectives, distillation, init, RL post-training); sections 14–16 are specialized families (hypergraph/topology, graph embeddings, 3D vision/flow/tracking).

Blocks 1–87 were the initial curated catalog; blocks 88–251 were extracted (2026-09-22) from all 288 architecture packages via the `block-extractor` skill, deduplicated against the existing library.

---

## How to Read This Index

Blocks are grouped by family, but many blocks span families (e.g., cross-attention is used in both seq2seq and diffusion). The "Used By" column in each README points to the real architectures in `../architectures/` that instantiate the block.

---

## 1. Convolutional Blocks (Vision)

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 1 | Conv-BN-ReLU Block | [`conv-bn-relu/`](./conv-bn-relu/) | AlexNet, VGG-16 |
| 2 | Residual Block (Identity Skip) | [`residual-block/`](./residual-block/) | ResNet |
| 3 | Bottleneck Residual Block | [`bottleneck-residual/`](./bottleneck-residual/) | ResNet-50/101/152 |
| 4 | Inception Module | [`inception-module/`](./inception-module/) | GoogLeNet / Inception |
| 5 | Depthwise-Separable Convolution | [`depthwise-separable-conv/`](./depthwise-separable-conv/) | MobileNetV2, Xception |
| 6 | Inverted Residual Block | [`inverted-residual/`](./inverted-residual/) | MobileNetV2 |
| 7 | MBConv Block | [`mbconv/`](./mbconv/) | EfficientNet-B0 |
| 8 | Squeeze-and-Excite (SE) Block | [`squeeze-and-excite/`](./squeeze-and-excite/) | SE-Net, EfficientNet |
| 9 | Dense Connectivity Block | [`dense-connectivity/`](./dense-connectivity/) | DenseNet-121 |
| 10 | ConvNeXt Block | [`convnext-block/`](./convnext-block/) | ConvNeXt-Tiny |
| 11 | Deformable Convolution (DCN-v2) | [`deformable-convolution/`](./deformable-convolution/) | DCN-v2 |
| 12 | Ghost Module | [`ghost-module/`](./ghost-module/) | GhostNet |
| 13 | Partial Convolution (PConv) | [`partial-convolution/`](./partial-convolution/) | FasterNet |
| 14 | Channel-Split Shuffle Block | [`channel-split-shuffle-block/`](./channel-split-shuffle-block/) | ShuffleNet V2 |
| 15 | Universal Inverted Bottleneck (UIB) | [`universal-inverted-bottleneck/`](./universal-inverted-bottleneck/) | MobileNetV4 |
| 16 | Structural Re-Parameterization | [`structural-reparameterization/`](./structural-reparameterization/) | RepViT, RepVGG |
| 17 | MobileViT Block | [`mobilevit-block/`](./mobilevit-block/) | MobileViT |
| 18 | Multi-Resolution Parallel Streams | [`multi-resolution-parallel-streams/`](./multi-resolution-parallel-streams/) | HRNet |
| 19 | Cross-Stage-Partial (CSP) Stage | [`csp-block/`](./csp-block/) | CSPNeXt (RTMPose), YOLO C3K2/R-ELAN |
| 20 | Star Operation Block | [`star-operation-block/`](./star-operation-block/) | StarNet |
| 21 | SlowFast Two-Pathway Factorization | [`slowfast-two-pathway/`](./slowfast-two-pathway/) | SlowFast Networks |

## 2. Encoder-Decoder & Detection Blocks

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 22 | U-Net Encoder-Decoder with Skips | [`unet-encoder-decoder/`](./unet-encoder-decoder/) | U-Net |
| 23 | Feature Pyramid Network (FPN) | [`feature-pyramid-network/`](./feature-pyramid-network/) | Mask R-CNN, RetinaNet |
| 24 | RoIAlign | [`roialign/`](./roialign/) | Mask R-CNN |
| 25 | Region Proposal Network (RPN) | [`region-proposal-network/`](./region-proposal-network/) | Faster R-CNN |
| 26 | All-MLP Segmentation Decoder | [`all-mlp-decoder/`](./all-mlp-decoder/) | SegFormer |
| 27 | Atrous Spatial Pyramid Pooling (ASPP) | [`aspp-decoder/`](./aspp-decoder/) | MobileNetV3 (LR-ASPP), DeepLab |
| 28 | DPT Reassembly + Fusion Decoder | [`dpt-decoder/`](./dpt-decoder/) | Depth Anything V2, MoGe |
| 29 | FCN Mask Branch | [`fcn-mask-head/`](./fcn-mask-head/) | Mask R-CNN |
| 30 | Prototype Mask Head (YOLACT-style) | [`prototype-mask-head/`](./prototype-mask-head/) | FastSAM, YOLACT |
| 31 | Anchor-Free Center Heatmap Head | [`center-heatmap-head/`](./center-heatmap-head/) | CenterPoint, CenterNet |
| 32 | Decoupled Anchor-Free Detection Head | [`decoupled-detection-head/`](./decoupled-detection-head/) | YOLOv11, YOLO26 |
| 33 | Spatial Pyramid Pooling — Fast (SPPF) | [`sppf/`](./sppf/) | YOLOv11 |
| 34 | Area Attention | [`area-attention/`](./area-attention/) | YOLOv12 |
| 35 | SimCC Coordinate Classification Head | [`simcc-head/`](./simcc-head/) | RTMPose |
| 36 | Keypoint Heatmap Head | [`keypoint-heatmap-head/`](./keypoint-heatmap-head/) | ViTPose |
| 37 | Masked Attention | [`masked-attention/`](./masked-attention/) | Mask2Former |
| 38 | Multi-Scale Deformable Attention | [`multi-scale-deformable-attention/`](./multi-scale-deformable-attention/) | Deformable DETR, DINO, BEVFormer, Mask2Former |
| 39 | Uncertainty-Minimal Query Selection | [`quality-aware-query-selection/`](./quality-aware-query-selection/) | RT-DETR, RT-DETRv3 |
| 40 | Language-Guided Query Selection | [`language-guided-query-selection/`](./language-guided-query-selection/) | Grounding DINO |
| 41 | Region-Text Contrastive Head | [`region-text-contrastive-head/`](./region-text-contrastive-head/) | Grounding DINO |
| 42 | Contrastive Denoising Queries (CDN) | [`contrastive-denoising-queries/`](./contrastive-denoising-queries/) | DINO (DN-DETR) |
| 43 | Look-Forward-Twice Box Supervision | [`look-forward-twice/`](./look-forward-twice/) | DINO |
| 44 | Fine-Grained Distribution Refinement (FDR) | [`fine-grained-distribution-refinement/`](./fine-grained-distribution-refinement/) | D-FINE |
| 45 | Joint Detection-Tracking Queries | [`joint-detection-tracking-queries/`](./joint-detection-tracking-queries/) | SAM 3 |
| 46 | Instance Appearance Adapter | [`instance-appearance-embedding/`](./instance-appearance-embedding/) | MASA |
| 47 | Sparse/Dense Prompt Encoder | [`prompt-encoder/`](./prompt-encoder/) | SAM, SAM 2, SAM 3 |
| 48 | Two-Way Transformer | [`two-way-transformer/`](./two-way-transformer/) | SAM, SAM 2 |
| 49 | Ambiguity-Aware Multi-Mask Output | [`multi-mask-iou-head/`](./multi-mask-iou-head/) | SAM |
| 50 | Global-Local Feature Fusion (HQ-Features) | [`global-local-feature-fusion/`](./global-local-feature-fusion/) | SAM-HQ |
| 51 | Streaming Memory Bank | [`streaming-memory-bank/`](./streaming-memory-bank/) | SAM 2 |
| 52 | Presence / Occlusion Head | [`presence-occlusion-head/`](./presence-occlusion-head/) | SAM 2, SAM 3 |
| 53 | Intermediate-Layer Feature Tap | [`intermediate-layer-tap/`](./intermediate-layer-tap/) | Perception Encoder |
| 54 | ViT Dense Prediction Adapter | [`vit-dense-adapter/`](./vit-dense-adapter/) | ViT-Adapter |
| 55 | Task Token | [`task-token/`](./task-token/) | OneFormer |
| 56 | BEV Query Grid | [`bev-query-grid/`](./bev-query-grid/) | BEVFormer |
| 57 | Recurrent Temporal BEV Attention | [`temporal-bev-attention/`](./temporal-bev-attention/) | BEVFormer |
| 58 | Frustum-to-BEV Voxel Pooling | [`frustum-voxel-pooling/`](./frustum-voxel-pooling/) | BEVDepth (Lift-Splat) |
| 59 | 3D Frustum Position Embedding | [`frustum-3d-position-embedding/`](./frustum-3d-position-embedding/) | PETR |
| 60 | Object-Query Memory Queue | [`query-memory-queue/`](./query-memory-queue/) | StreamPETR |

## 3. Transformer / Attention Blocks

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 61 | Patch Embedding (ViT) | [`patch-embedding/`](./patch-embedding/) | ViT-B/16 |
| 62 | Transformer Encoder Block | [`transformer-encoder-block/`](./transformer-encoder-block/) | BERT, ViT |
| 63 | Transformer Decoder Block | [`transformer-decoder-block/`](./transformer-decoder-block/) | Transformer, T5, Whisper |
| 64 | Decoder-Only Block (GPT/LLaMA) | [`decoder-only-block/`](./decoder-only-block/) | GPT-2/3, LLaMA-3 |
| 65 | Multi-Head Attention (Full) | [`multi-head-attention/`](./multi-head-attention/) | Transformer |
| 66 | Grouped-Query Attention (GQA) | [`grouped-query-attention/`](./grouped-query-attention/) | LLaMA-3, Mistral |
| 67 | Multi-Head Latent Attention (MLA) | [`multi-head-latent-attention/`](./multi-head-latent-attention/) | DeepSeek-V2/V3 |
| 68 | Sliding-Window Attention | [`sliding-window-attention/`](./sliding-window-attention/) | Longformer, Mistral |
| 69 | Native Sparse Attention (NSA) | [`native-sparse-attention/`](./native-sparse-attention/) | NSA (DeepSeek) |
| 70 | Differential Attention | [`differential-attention/`](./differential-attention/) | Differential Transformer |
| 71 | Cross-Attention Block | [`cross-attention/`](./cross-attention/) | Transformer, Whisper, SD |
| 72 | Swin Shifted-Window Attention | [`swin-shifted-window-attention/`](./swin-shifted-window-attention/) | Swin-Tiny |
| 73 | SwiGLU Feed-Forward Network | [`swiglu-ffn/`](./swiglu-ffn/) | LLaMA, Mistral |
| 74 | Fast Feedforward Network (FFF) | [`fast-feedforward-network/`](./fast-feedforward-network/) | FFF |
| 75 | Multi-Query Attention (MQA) | [`multi-query-attention/`](./multi-query-attention/) | Falcon-7B, Fast Transformer Decoder, MobileNetV4 |
| 76 | Parallel Attention+MLP Block (GPT-NeoX) | [`parallel-residual-block/`](./parallel-residual-block/) | Falcon-7B, Phi-2, Pythia |
| 77 | GeGLU Feed-Forward Network | [`geglu-ffn/`](./geglu-ffn/) | Gemma 4, ModernBERT |
| 78 | Local/Global Interleaved Attention | [`local-global-interleaved-attention/`](./local-global-interleaved-attention/) | Gemma 4, gpt-oss, ModernBERT, VGGT |
| 79 | Interleaved NoPE (iRoPE) | [`interleaved-nope/`](./interleaved-nope/) | Llama-4 Scout |
| 80 | Learned Attention Sink | [`attention-sink/`](./attention-sink/) | gpt-oss-120b/20b |
| 81 | YaRN RoPE Scaling | [`yarn-rope-scaling/`](./yarn-rope-scaling/) | gpt-oss-120b/20b |
| 82 | QK-Norm | [`qk-norm/`](./qk-norm/) | Qwen3-8B, ViT-22B |
| 83 | Spatial-Reduction Attention (SRA) | [`spatial-reduction-attention/`](./spatial-reduction-attention/) | PVT, MViTv2 |
| 84 | Multi-Axis Attention | [`multi-axis-attention/`](./multi-axis-attention/) | MaxViT |
| 85 | Cross-Shaped Window Attention | [`cswin-attention/`](./cswin-attention/) | CSWin |
| 86 | Divided Space-Time Attention | [`divided-space-time-attention/`](./divided-space-time-attention/) | TimeSformer |
| 87 | Factorised Space-Time Encoder | [`factorised-space-time-encoder/`](./factorised-space-time-encoder/) | ViViT |
| 88 | Joint Space-Time Track Attention | [`joint-spacetime-attention/`](./joint-spacetime-attention/) | CoTracker3 |
| 89 | Weight Tying | [`weight-tying/`](./weight-tying/) | GPT-1, MiniCPM-2B |

## 4. Normalization & Positional Encoding

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 90 | RMSNorm | [`rmsnorm/`](./rmsnorm/) | LLaMA, Mistral |
| 91 | Adaptive LayerNorm (adaLN-Zero) | [`adaptive-layernorm/`](./adaptive-layernorm/) | DiT-XL/2 |
| 92 | Rotary Position Embedding (RoPE) | [`rope/`](./rope/) | LLaMA, Qwen, Mistral |
| 93 | ALiBi (Linear Bias Positions) | [`alibi/`](./alibi/) | BLOOM, MPT |
| 94 | Learned Absolute Position Embedding | [`learned-position-embedding/`](./learned-position-embedding/) | GPT-2, BERT, ViT |
| 95 | Sinusoidal Positional Encoding | [`sinusoidal-positional-encoding/`](./sinusoidal-positional-encoding/) | Transformer |
| 96 | Relative Position Bias | [`relative-position-bias/`](./relative-position-bias/) | T5, Swin-V2 |
| 97 | Convolutional Positional Embedding | [`convolutional-positional-embedding/`](./convolutional-positional-embedding/) | Wav2Vec2 |
| 98 | Fourier Positional Encoding | [`fourier-positional-encoding/`](./fourier-positional-encoding/) | NeRF |

## 5. Sparse / Mixture-of-Experts

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 99 | Mixture-of-Experts (MoE) Layer | [`mixture-of-experts/`](./mixture-of-experts/) | Mixtral, DeepSeek-V2 |
| 100 | Multi-Gate Mixture-of-Experts (MMoE) | [`multi-gate-moe/`](./multi-gate-moe/) | MMoE (multi-task ranking) |
| 101 | Shared-Expert MoE (DeepSeekMoE) | [`shared-expert-moe/`](./shared-expert-moe/) | DeepSeek-V2/V3, GLM-4.5-Air |
| 102 | Auxiliary-Loss-Free Load Balancing | [`auxiliary-loss-free-balancing/`](./auxiliary-loss-free-balancing/) | DeepSeek-V3 |
| 103 | Customized Gate Control (CGC / PLE) | [`cgc-expert-separation/`](./cgc-expert-separation/) | PLE |

## 6. Recurrent / State-Space Blocks

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 104 | LSTM Cell | [`lstm-cell/`](./lstm-cell/) | LSTM |
| 105 | xLSTM Block (sLSTM / mLSTM) | [`xlstm-block/`](./xlstm-block/) | xLSTM |
| 106 | Selective State Space Model (Mamba) | [`selective-ssm/`](./selective-ssm/) | Mamba |
| 107 | Structured State Space Model (S4) | [`structured-ssm-s4/`](./structured-ssm-s4/) | S4 |
| 108 | RWKV Time-Mixing Block | [`rwkv-time-mixing/`](./rwkv-time-mixing/) | RWKV |
| 109 | Associative Memory Transformer | [`associative-memory-transformer/`](./associative-memory-transformer/) | ARMT |
| 110 | Memory Token Segment Recurrence | [`memory-tokens/`](./memory-tokens/) | RMT, ARMT |
| 111 | Gated Recurrent Unit (GRU) | [`gru-cell/`](./gru-cell/) | DIEN, SEA-RAFT |
| 112 | Token Shift | [`token-shift/`](./token-shift/) | RWKV |
| 113 | RWKV Channel Mixing | [`rwkv-channel-mixing/`](./rwkv-channel-mixing/) | RWKV |
| 114 | Complex MIMO State-Space Mixer | [`complex-mimo-ssm/`](./complex-mimo-ssm/) | Mamba-3 |
| 115 | MambaVision Mixer | [`mambavision-mixer/`](./mambavision-mixer/) | MambaVision |
| 116 | Bidirectional Selective SSM (Vim) | [`bidirectional-ssm/`](./bidirectional-ssm/) | Vision Mamba |
| 117 | 2D Selective Scan (SS2D) | [`ss2d-selective-scan/`](./ss2d-selective-scan/) | VMamba |
| 118 | Windowed Local Scan | [`windowed-local-scan/`](./windowed-local-scan/) | LocalMamba |
| 119 | Time-Aware LSTM | [`time-aware-lstm/`](./time-aware-lstm/) | SLi-Rec |

## 7. Graph Neural Network Blocks

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 120 | Graph Convolution Layer (GCN) | [`graph-convolution-layer/`](./graph-convolution-layer/) | GCN |
| 121 | Graph Attention Layer (GAT) | [`graph-attention-layer/`](./graph-attention-layer/) | GAT |
| 122 | GraphSAGE Neighborhood Aggregator | [`graphsage-aggregator/`](./graphsage-aggregator/) | GraphSAGE, PinSage |
| 123 | LightGCN Propagation | [`lightgcn-propagation/`](./lightgcn-propagation/) | LightGCN |
| 124 | Adaptive Adjacency Learning | [`adaptive-adjacency-learning/`](./adaptive-adjacency-learning/) | MTGNN, GAD, StemGNN, UAGSL |
| 125 | Convolution-Based Graph Diffusion | [`graph-diffusion-convolution/`](./graph-diffusion-convolution/) | ST-GDN |
| 126 | Multi-View Graph Fusion | [`multi-view-graph-fusion/`](./multi-view-graph-fusion/) | MHGCN, ST-MGCN, MVNE, DNE-PL, FMTSF |
| 127 | Mutual Selective Attention | [`mutual-attention/`](./mutual-attention/) | CANE |
| 128 | Cross-Graph Attention Matching | [`cross-graph-attention/`](./cross-graph-attention/) | GMN |
| 129 | Graph Propagation Attention (GPA) | [`graph-propagation-attention/`](./graph-propagation-attention/) | GraphProp |
| 130 | Path-Based Message Passing | [`path-message-passing/`](./path-message-passing/) | RouteNet |
| 131 | Edge Memory Message Passing | [`edge-memory-message-passing/`](./edge-memory-message-passing/) | PI-GNN |
| 132 | Antisymmetric Edge-Local Frame GNN | [`antisymmetric-edge-frame-gnn/`](./antisymmetric-edge-frame-gnn/) | PI-GNN |
| 133 | Cooperative Action-Environment GNN | [`cooperative-action-network/`](./cooperative-action-network/) | CoGNN |
| 134 | Deep Graph Infomax (DGI) | [`deep-graph-infomax/`](./deep-graph-infomax/) | DGI |
| 135 | Copy Mechanism | [`copy-mechanism/`](./copy-mechanism/) | Graph2Seq |
| 136 | Laplacian Positional Encoding | [`laplacian-positional-encoding/`](./laplacian-positional-encoding/) | Graph Transformer |
| 137 | Edge-Feature Attention Injection | [`edge-feature-attention/`](./edge-feature-attention/) | Graph Transformer |
| 138 | Graph Prompt Tokens | [`graph-prompting/`](./graph-prompting/) | SAMGPT |
| 139 | Graph Softmax | [`graph-softmax/`](./graph-softmax/) | GraphGAN |
| 140 | One-Shot Probabilistic Graph Decoder | [`one-shot-graph-decoder/`](./one-shot-graph-decoder/) | GraphVAE |
| 141 | Variance-Reduced GCN Sampling | [`variance-reduced-gcn-sampling/`](./variance-reduced-gcn-sampling/) | FastGCN, Stochastic-GCN |

## 8. Recommendation System Blocks

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 142 | Wide & Deep Block | [`wide-and-deep/`](./wide-and-deep/) | Wide & Deep |
| 143 | DeepFM Block | [`deepfm/`](./deepfm/) | DeepFM |
| 144 | Deep & Cross Network (DCN) | [`deep-cross-network/`](./deep-cross-network/) | DCN / DCN-v2 |
| 145 | Target Attention (DIN Activation Unit) | [`target-attention-din/`](./target-attention-din/) | DIN |
| 146 | Candidate-in-Sequence | [`candidate-in-sequence/`](./candidate-in-sequence/) | BST |
| 147 | GMF + MLP Fusion | [`gmf-mlp-fusion/`](./gmf-mlp-fusion/) | NeuMF |
| 148 | Pairwise Dot-Interaction | [`pairwise-dot-interaction/`](./pairwise-dot-interaction/) | DLRM |
| 149 | Entire-Space Multi-Task Towers | [`entire-space-multitask/`](./entire-space-multitask/) | ESMM |

## 9. Generative & Diffusion Blocks

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 150 | GAN Minimax Block | [`gan-minimax/`](./gan-minimax/) | GAN, DCGAN, StyleGAN |
| 151 | Diffusion U-Net (Latent Diffusion) | [`diffusion-unet/`](./diffusion-unet/) | Stable Diffusion |
| 152 | DiT Block (Diffusion Transformer) | [`dit-block/`](./dit-block/) | DiT-XL/2, SD3, Sora |
| 153 | Block-Causal Diffusion Generation | [`block-causal-diffusion/`](./block-causal-diffusion/) | Block Diffusion, CLDLM |
| 154 | Flow Matching Objective | [`flow-matching/`](./flow-matching/) | CLDLM, VGGT-World |
| 155 | Continuous Next-Embedding Prediction | [`continuous-embedding-prediction/`](./continuous-embedding-prediction/) | LCM |
| 156 | Global/Local Patch Decoder (MegaByte) | [`global-local-patch-decoder/`](./global-local-patch-decoder/) | MegaByte |

## 10. Audio Blocks

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 157 | Conv1D Audio Stem | [`conv1d-audio-stem/`](./conv1d-audio-stem/) | Wav2Vec2, Whisper |
| 158 | Residual Vector Quantization (RVQ) | [`residual-vector-quantization/`](./residual-vector-quantization/) | EnCodec |
| 159 | Codebook Quantization of Latents | [`codebook-quantization/`](./codebook-quantization/) | Wav2Vec2 |

## 11. Multimodal Blocks

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 160 | Contrastive Dual-Encoder (CLIP) | [`contrastive-dual-encoder/`](./contrastive-dual-encoder/) | CLIP ViT-B/32 |
| 161 | Q-Former (Learned Query Bridge) | [`q-former/`](./q-former/) | BLIP-2 |
| 162 | Perceiver Resampler | [`perceiver-resampler/`](./perceiver-resampler/) | Flamingo |
| 163 | Gated Cross-Attention (Multimodal) | [`gated-cross-attention/`](./gated-cross-attention/) | Flamingo |
| 164 | MLP Projector Bridge (LLaVA) | [`mlp-projector-bridge/`](./mlp-projector-bridge/) | LLaVA-1.5-7B |

## 12. Alternative / Research Blocks

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 165 | KAN Layer | [`kan-layer/`](./kan-layer/) | KAN |
| 166 | TimesBlock (1D-to-2D) | [`timesblock/`](./timesblock/) | TimesNet |
| 167 | PatchTST Block | [`patchtst-block/`](./patchtst-block/) | PatchTST |
| 168 | Backpack Sense Vectors | [`backpack-sense-vectors/`](./backpack-sense-vectors/) | Backpack-LM |
| 169 | Neuron-Level Model (NLM) | [`neuron-level-model/`](./neuron-level-model/) | Continuous Thought Machines |
| 170 | Neural Synchronization Representation | [`neural-synchronization/`](./neural-synchronization/) | Continuous Thought Machines |
| 171 | EEG Conv Tokenizer Stem | [`eeg-conv-stem/`](./eeg-conv-stem/) | EEGNet, EEG Conformer |
| 172 | Slimmable Supernet | [`slimmable-supernet/`](./slimmable-supernet/) | Slimmable Networks, Once-for-All |
| 173 | MultKAN Multiplication Node | [`multiplication-node/`](./multiplication-node/) | KAN 2.0 |
| 174 | Reconciled Polynomial Layer (RPN) | [`reconciled-polynomial-layer/`](./reconciled-polynomial-layer/) | RPN, RPN-v2 |
| 175 | SConv (Stateful Convolution) Block | [`sconv-block/`](./sconv-block/) | Nemotron |
| 176 | SSM-Attention Hybrid Stack | [`ssm-attention-hybrid/`](./ssm-attention-hybrid/) | Jamba, Nemotron |
| 177 | Recurrent World Model + Imagination | [`recurrent-world-model/`](./recurrent-world-model/) | DreamerV3 |
| 178 | Action-Conditioned Latent Dynamics | [`action-conditioned-latent-dynamics/`](./action-conditioned-latent-dynamics/) | V-JEPA 2-AC |

## 13. Training Primitives

Non-structural components that every architecture above depends on: objectives, distillation, normalization-adjacent init, RL post-training, and task-specific losses.

| # | Block | Folder | Origin |
|---|-------|--------|--------|
| 179 | LayerNorm | [`layer-norm/`](./layer-norm/) | Ba et al., 2016 |
| 180 | BatchNorm | [`batch-norm/`](./batch-norm/) | Ioffe & Szegedy, 2015 |
| 181 | GELU | [`gelu/`](./gelu/) | Hendrycks & Gimpel, 2016 |
| 182 | SiLU / Swish | [`silu-swish/`](./silu-swish/) | Ramachandran et al., 2017 |
| 183 | Kaiming (He) Init | [`kaiming-init/`](./kaiming-init/) | He et al., 2015 |
| 184 | Xavier (Glorot) Init | [`xavier-init/`](./xavier-init/) | Glorot & Bengio, 2010 |
| 185 | AdamW Optimizer | [`adamw-optimizer/`](./adamw-optimizer/) | Loshchilov & Hutter, 2019 |
| 186 | Cosine LR Schedule | [`cosine-lr-schedule/`](./cosine-lr-schedule/) | Loshchilov & Hutter, SGDR 2017 |
| 187 | Linear LR Warmup | [`linear-warmup/`](./linear-warmup/) | Goyal et al., 2017 |
| 188 | Focal Loss | [`focal-loss/`](./focal-loss/) | Lin et al., RetinaNet 2017 |
| 189 | Hungarian Matching Loss | [`hungarian-matching-loss/`](./hungarian-matching-loss/) | Carion et al., DETR 2020 |
| 190 | InfoNCE Contrastive Loss | [`infonce-contrastive-loss/`](./infonce-contrastive-loss/) | van den Oord et al., CPC 2018 |
| 191 | GIoU / CIoU / DIoU Loss | [`giou-loss/`](./giou-loss/) | Rezatofighi 2019 / Zheng 2020 |
| 192 | EMA of Weights / SWA | [`ema-weights/`](./ema-weights/) | Izmailov et al., SWA 2018 |
| 193 | SGD with Momentum | [`sgd-momentum/`](./sgd-momentum/) | Polyak 1964 / Nesterov 1983 |
| 194 | Lion Optimizer | [`lion-optimizer/`](./lion-optimizer/) | Chen et al., 2023 |
| 195 | WSD LR Schedule (Warmup-Stable-Decay) | [`wsd-lr-schedule/`](./wsd-lr-schedule/) | MiniCPM / Hu et al., 2024 |
| 196 | OneCycle LR Schedule | [`onecycle-lr-schedule/`](./onecycle-lr-schedule/) | Smith & Topin, 2018 |
| 197 | Dice Loss | [`dice-loss/`](./dice-loss/) | Milletari et al., V-Net 2016 |
| 198 | Label Smoothing | [`label-smoothing/`](./label-smoothing/) | Szegedy et al., Inception-v3 2016 |
| 199 | Stochastic Depth / DropPath | [`stochastic-depth/`](./stochastic-depth/) | Huang et al., 2016 |
| 200 | Gradient Checkpointing | [`gradient-checkpointing/`](./gradient-checkpointing/) | Chen et al., 2016 |
| 201 | Mixed Precision (AMP / BF16) | [`mixed-precision-amp/`](./mixed-precision-amp/) | Micikevicius et al., 2018 |
| 202 | SyncBatchNorm | [`syncbatchnorm/`](./syncbatchnorm/) | Peng et al., MegDet 2018 |
| 203 | GroupNorm | [`groupnorm/`](./groupnorm/) | Wu & He, 2018 |
| 204 | Masked Language Modeling (Cloze) | [`masked-language-modeling/`](./masked-language-modeling/) | BERT, BERT4Rec |
| 205 | Masked Autoencoder (MAE) | [`masked-autoencoder/`](./masked-autoencoder/) | VideoMAE, EfficientSAM (SAMI), SimMIM, HuBERT |
| 206 | Masked Latent Prediction (JEPA) | [`masked-latent-prediction/`](./masked-latent-prediction/) | I-JEPA, V-JEPA / V-JEPA-2 |
| 207 | Tube Masking | [`tube-masking/`](./tube-masking/) | V-JEPA, VideoMAE |
| 208 | RLHF PPO Pipeline | [`rlhf-ppo-pipeline/`](./rlhf-ppo-pipeline/) | InstructGPT, GPT-4, Gemini |
| 209 | Group Relative Policy Optimization (GRPO) | [`grpo/`](./grpo/) | DeepSeekMath |
| 210 | Multi-Token Prediction (MTP) | [`multi-token-prediction/`](./multi-token-prediction/) | DeepSeek-V3 |
| 211 | Residual-Depth-Scaled Init / muP Residuals | [`residual-scaling-init/`](./residual-scaling-init/) | GPT-2, MiniCPM-2B, LWM |
| 212 | Hypersphere SLERP Residual Update | [`slerp-residual-update/`](./slerp-residual-update/) | nGPT |
| 213 | Pairwise Ranking Loss (BPR / Margin) | [`pairwise-ranking-loss/`](./pairwise-ranking-loss/) | GRU4Rec, PinSage, SDGE, SNE |
| 214 | Pairwise Sigmoid Contrastive Loss | [`pairwise-sigmoid-contrastive-loss/`](./pairwise-sigmoid-contrastive-loss/) | SigLIP |
| 215 | Scale-and-Shift-Invariant Depth Loss | [`scale-shift-invariant-loss/`](./scale-shift-invariant-loss/) | Depth Anything V2, Metric3D |
| 216 | Per-Region Optimal Supervision | [`per-region-optimal-supervision/`](./per-region-optimal-supervision/) | MoGe |
| 217 | Confidence-Weighted Regression Loss | [`confidence-weighted-loss/`](./confidence-weighted-loss/) | DUSt3R, MASt3R |
| 218 | Mixture-of-Laplace Loss | [`mixture-of-laplace-loss/`](./mixture-of-laplace-loss/) | SEA-RAFT |
| 219 | Temporal Consistency Loss | [`temporal-consistency-loss/`](./temporal-consistency-loss/) | Video Depth Anything |
| 220 | Gram Anchoring Loss | [`gram-anchoring-loss/`](./gram-anchoring-loss/) | DINOv3 |
| 221 | Embedding Distillation | [`embedding-distillation/`](./embedding-distillation/) | MobileSAM, MiniLM, EdgeSAM, VideoPrism |
| 222 | Global Optimal Localization Self-Distillation | [`go-lsd-self-distillation/`](./go-lsd-self-distillation/) | D-FINE |
| 223 | Straight-Through Gumbel-Softmax | [`gumbel-softmax-straight-through/`](./gumbel-softmax-straight-through/) | CoGNN, MolGAN, NetGAN |

## 14. Hypergraph & Topological Blocks

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 224 | Tensorized Hypergraph Convolution | [`tensorized-hypergraph-conv/`](./tensorized-hypergraph-conv/) | THNN |
| 225 | Position-Dependent Hyperedge Embedding | [`hyperedge-position-embedding/`](./hyperedge-position-embedding/) | KHG |
| 226 | Non-Linear Tuplewise Similarity | [`tuplewise-hyperedge-similarity/`](./tuplewise-hyperedge-similarity/) | DHNE |
| 227 | Multi-Arity Expand-Reduce Hyperedge Layer | [`multi-arity-hyperedge-layer/`](./multi-arity-hyperedge-layer/) | SAL-HR |
| 228 | Topological Message Passing | [`topological-message-passing/`](./topological-message-passing/) | TDL |
| 229 | Tri-Level Hypergraph Contrast | [`hypergraph-contrastive-learning/`](./hypergraph-contrastive-learning/) | TriCL |

## 15. Graph Embedding Methods

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 230 | k-Step PPMI-SVD Embedding | [`ppmi-svd-embedding/`](./ppmi-svd-embedding/) | GraRep |
| 231 | High-Order Proximity Update | [`high-order-proximity-update/`](./high-order-proximity-update/) | FastNE, DANE |
| 232 | Poincaré Ball Embedding | [`poincare-embedding/`](./poincare-embedding/) | HGEmb |
| 233 | Gaussian Node Embedding | [`gaussian-node-embedding/`](./gaussian-node-embedding/) | DVNE, SDGE |
| 234 | Graphlet Degree Vector (GDV) | [`graphlet-degree-vector/`](./graphlet-degree-vector/) | SNE-Enhanced |
| 235 | Temporal Random Walk Sampler | [`temporal-random-walk/`](./temporal-random-walk/) | CTNE |
| 236 | Meta-Path Guided Random Walk | [`metapath-random-walk/`](./metapath-random-walk/) | metapath2vec(++) |
| 237 | Multilayer Structural Similarity Graph | [`structural-similarity-multigraph/`](./structural-similarity-multigraph/) | struc2vec |
| 238 | Graph Coarsening Hierarchy | [`graph-coarsening-hierarchy/`](./graph-coarsening-hierarchy/) | HARP, MILE |
| 239 | RL Attention-Guided Graph Walk | [`rl-attention-guided-graph-walk/`](./rl-attention-guided-graph-walk/) | Structural Attention GNN |

## 16. 3D Vision, Flow & Tracking Blocks

| # | Block | Folder | Origin Architecture |
|---|-------|--------|----------------------|
| 240 | 3D Gaussian Primitives | [`3d-gaussian-primitives/`](./3d-gaussian-primitives/) | 3D Gaussian Splatting, MonoSplat |
| 241 | Tile-Based Splatting Rasterizer | [`tile-based-splatting/`](./tile-based-splatting/) | 3D Gaussian Splatting |
| 242 | Differentiable Volume Rendering | [`volume-rendering/`](./volume-rendering/) | NeRF |
| 243 | Pointmap Regression | [`pointmap-regression/`](./pointmap-regression/) | DUSt3R, MASt3R, MoGe, VGGT |
| 244 | Canonical Camera Space Transform | [`canonical-camera-transform/`](./canonical-camera-transform/) | Metric3D |
| 245 | Camera-Aware Depth Conditioning | [`camera-aware-depth-conditioning/`](./camera-aware-depth-conditioning/) | BEVDepth, UniDepth |
| 246 | All-Pairs Correlation Pyramid | [`all-pairs-correlation-pyramid/`](./all-pairs-correlation-pyramid/) | RAFT, RAFT-Stereo, SEA-RAFT, CoTracker3, GMFlow |
| 247 | ConvGRU Recurrent Update Operator | [`convgru-update-operator/`](./convgru-update-operator/) | RAFT, RAFT-Stereo, GMFlow, FlowFormer, IGEV |
| 248 | Convex Upsampling | [`convex-upsampling/`](./convex-upsampling/) | RAFT, RAFT-Stereo, SEA-RAFT, GMFlow |
| 249 | Geometry Encoding Volume (GEV) | [`geometry-encoding-volume/`](./geometry-encoding-volume/) | IGEV-Stereo |
| 250 | Two-Stage Point Tracking | [`two-stage-point-tracking/`](./two-stage-point-tracking/) | TAPIR |
| 251 | Axial-Planar 3D Convolution | [`axial-planar-3d-conv/`](./axial-planar-3d-conv/) | FoundationStereo |

---

## Deduplication Notes

Each block is **distinct** — no two blocks serve the same structural role. Original catalog (blocks 1–87):

- **Residual vs. Bottleneck Residual vs. Inverted Residual**: All three are skip-connection blocks, but they differ in *topology* (basic 3×3×2, 1×1-3×3-1×1, narrow-wide-narrow). Treated as three blocks.
- **MHA vs. GQA vs. MLA vs. Sliding-Window vs. NSA vs. Differential**: All are attention, but each is a *distinct mechanism* (dense, shared-KV, latent-compressed, windowed, block-sparse, differential). Six blocks.
- **SE vs. Cross-Attention vs. Target Attention**: All are attention/gating, but SE is channel-attention, cross-attention is seq-to-seq, target attention is query-to-sequence. Three blocks.
- **MoE vs. MMoE**: MoE is a single-gate sparse FFN layer; MMoE is multi-gate multi-task. Two blocks.
- **U-Net vs. Diffusion U-Net vs. DiT**: U-Net is the topology; Diffusion U-Net adds cross-attention + timestep; DiT replaces it with a Transformer. Three blocks.
- **LSTM vs. xLSTM vs. S4 vs. Mamba vs. RWKV**: All are recurrent/SSM, but each is a *distinct cell/block*. Five blocks.
- **GCN vs. GAT vs. GraphSAGE**: All are graph message passing, but GCN = fixed spectral, GAT = attention, GraphSAGE = sampled aggregation. Three blocks.
- **Wide & Deep vs. DeepFM vs. DCN**: All are hybrid CTR, but the *interaction mechanism* differs (manual crosses, FM pairwise, learned crosses). Three blocks.
- **LayerNorm vs. BatchNorm vs. RMSNorm**: All normalize activations, but the *reduction axes differ*. Three blocks.
- **GELU vs. SiLU/Swish**: Nearly the same curve, but different origin and different dominant use. Two blocks.
- **Kaiming vs. Xavier**: Same goal with different activation assumptions. Two blocks.
- **Focal vs. Hungarian vs. InfoNCE vs. GIoU**: Four different *problems* — class imbalance, one-to-one assignment, representation alignment, metric-aligned box regression. Four blocks.

Extraction pass (blocks 88–251) — key split/merge decisions:

- **GQA vs. MQA vs. MLA**: kept three blocks — KV *grouping* (GQA), single shared KV head (MQA), latent-compressed cache (MLA) are distinct cache strategies.
- **Shared-Expert MoE merged** across DeepSeek-V2/V2-Lite/V3 and GLM-4.5-Air (one mechanism, four users); PLE's CGC kept separate (task-specific expert groups, not shared+fine-grained routing).
- **Attention interleaving merged**: Gemma (5:1), gpt-oss, ModernBERT (1:2), and VGGT (frame-local/global) all alternate local and global attention — one block; Jamba's SSM/attention hybrid kept separate (SSM mixer, not local attention).
- **RoPE-family**: partial RoPE (ChatGLM3, Phi-2), XPOS (LWM), and NTK/YaRN kept distinct only where the mechanism differs — YaRN scaling and attention sinks are blocks; partial RoPE is a hyperparameter of `rope` (noted there).
- **Deformable attention merged** across Deformable-DETR, DINO, D-FINE, Mask2Former (pixel decoder), BEVFormer (spatial cross-attention) — one mechanism; PVT's SRA and MViTv2's pooling attention merged as `spatial-reduction-attention` (KV reduction, same role).
- **RAFT family merged**: all-pairs correlation pyramid, ConvGRU update operator, and convex upsampling are one trio shared verbatim by RAFT, RAFT-Stereo, SEA-RAFT, GMFlow, FlowFormer; IGEV's GEV kept separate (geometry-fused volume); CoTracker3's local correlation folded into the pyramid block.
- **Memory mechanisms**: RMT/ARMT memory tokens merged; SAM 2's memory bank (frame tokens) and StreamPETR's query queue (object queries) kept separate — different memory units; BEVFormer's temporal BEV attention separate (dense-grid recurrence).
- **Masked pretraining trio**: MLM (discrete tokens), MAE (reconstructed inputs — pixels/features/IDs), JEPA (predicted latent targets) kept as three blocks; tube masking separate (the video masking pattern); HuBERT's masked cluster prediction folded into MAE's target variants; wav2vec2's codebook quantization separate (target construction mechanism).
- **Depth supervision**: scale-shift-invariant loss (global affine), per-region optimal supervision (region affine + rejection), confidence-weighted loss (learned weighting) kept as three distinct reliability mechanisms; Metric3D's random-proposal normalization folded into the scale-shift block.
- **Distillation**: GoLSD (self, cross-layer), Gram anchoring (similarity geometry), and embedding distillation (teacher-encoder embeddings, incl. MiniLM attention distill, EdgeSAM disagreement prompts, VideoPrism global→local) kept as three; DINO's look-forward-twice kept separate (box-coordinate supervision alignment, not distillation).
- **Graph-structure learning merged**: MTGNN embedding-based adjacency, GAD top-k, StemGNN latent correlation, UAGSL uncertainty thresholds → one `adaptive-adjacency-learning` block; FastGCN importance sampling and Stochastic-GCN control variates merged as `variance-reduced-gcn-sampling`.
- **Walk/coarsening samplers**: temporal (CTNE) and metapath (metapath2vec) walks kept separate (time vs type constraints); struc2vec's structural multigraph separate (auxiliary graph); HARP and MILE coarsening merged (same hierarchy mechanism, different refinement).
- **YOLO minutiae trimmed**: C3K2 and R-ELAN folded into `csp-block`; C2PSA folded into `area-attention`; SPPF kept (distinct pooling-chain mechanism).
- **Folds into existing blocks**: hard-swish → `silu-swish`; DCN-v2 low-rank cross → `deep-cross-network`; dilated inception (MTGNN) → `inception-module`; norm-head (Baichuan), non-parametric LayerNorm (OLMo) → `layer-norm`/`weight-tying` notes; partial RoPE → `rope`.
- **Deliberately not block-ified**: ByteTrack/OC-SORT Kalman + association stages (classical CV algorithms, not NN blocks); hiera encoder (backbone architecture, not a reusable mechanism); side-tuning (FoundationStereo), cascade masking & token shuffling (VideoPrism), data-driven noise schedule (Block Diffusion), Hoyer sparsity & information-sufficiency sampling (SAL-HR) — pretraining/recipe-level details rather than architecture choices; mean-pool / CLS-token pooling (generic); GeGLU-vs-SwiGLU, GQA head ratios (hyperparameters).

---

## Relationship to `../architectures/`

Each block's README has a **Used By** table pointing to the real architecture packages that instantiate it. The architectures are the *models*; the blocks are the *reusable components* those models are built from. A single architecture typically composes several blocks (e.g., LLaMA-3 = decoder-only block + GQA + RoPE + RMSNorm + SwiGLU).

Note: Used By tables of blocks 1–87 reflect their original catalogs; newly extracted packages that reuse pre-existing blocks (e.g., Qwen3 using GQA, internlm2 using SwiGLU) are catalogued in the new blocks' Used By tables and the architecture packages themselves. Back-filling the pre-existing blocks' Used By tables for the full 288-package corpus is a tracked follow-up.
