# Block Catalog Redundancy Findings (YRR-02)

**Scope:** All 297 entries under `/Users/xiaming/Workspace/atlas/blocks/` (verified: 297 folders on disk == 297 rows in `INDEX.md`; zero exact-duplicate folder names; zero exact-duplicate display names).

**Method:** Parsed `INDEX.md` for the canonical (display name, folder) list, then read the `README.md` of every block flagged as a suspected redundancy cluster to compare design philosophy, functionality, and "Used By" tables. The catalog explicitly states "no duplication" and most blocks are genuinely distinct; the groups below are the cases where two or more entries describe the *same underlying mechanism* (alias / near-duplicate / one-is-a-special-case-of-the-other) rather than genuinely independent blocks.

---

## Suspected Redundant Block Groups

### Group 1 — Grouped-Query Attention vs. Multi-Query Attention (alias / special-case)
- `grouped-query-attention` — "Grouped-Query Attention (GQA) / Multi-Query Attention (MQA)"
- `multi-query-attention` — "Multi-Query Attention (MQA)"

**Rationale:** The `grouped-query-attention` README's own title already includes "Multi-Query Attention (MQA)", and its functionality section states MQA is the `G=1` extreme of GQA. The separate `multi-query-attention` block restates the same mechanism (H query heads, 1 shared KV head) as a standalone entry. The two READMEs cross-reference each other ("GQA (existing block) is the interpolating middle ground"). This is the strongest redundancy in the catalog: MQA is a strict special case of GQA, and the GQA entry already documents it. **Recommendation:** merge `multi-query-attention` into `grouped-query-attention` (or vice versa) and keep a single "GQA/MQA" entry.

---

### Group 2 — Flow Matching Objective vs. Flow Matching ODE (near-duplicate)
- `flow-matching` — "Flow Matching Objective"
- `flow-matching-ode` — "Flow Matching ODE"

**Rationale:** Both blocks describe learning a velocity field `v(x,t)` transporting noise to data via the flow-matching loss `‖v − u_t‖²`, then integrating the ODE at inference. The functionality sections are nearly identical (same loss form, same Euler/RK4 inference, same "Used By" mentions of Rectified Flow / Stable Diffusion 3 / Voicebox). The only nominal distinction is "objective" (training loss) vs. "ODE" (inference integrator), but each README actually covers both halves. Their Evolution sections list the same predecessors (DDPM) and successors (rectified flow, stochastic interpolants). **Recommendation:** merge into a single `flow-matching` block; the ODE is the inference side of the same objective, not a separate block.

---

### Group 3 — Inverted Residual vs. MBConv (near-duplicate / one-adds-SE)
- `inverted-residual` — "Inverted Residual Block (MobileNetV2)"
- `mbconv` — "MBConv Block"

**Rationale:** MBConv *is* the inverted-residual block plus a squeeze-and-excite gate; the `mbconv` README explicitly states "MobileNetV2's inverted-residual block with a squeeze-and-excite (SE) channel-attention gate inserted after the depthwise conv" and lists "Predecessor: Inverted residual (MobileNetV2) + standalone SE-Net." The two blocks share the identical structure (`1×1 expand → depthwise → 1×1 project` with linear bottleneck) and the same "Used By" models (EfficientNet, MobileNetV2/V3). The only functional delta is the SE module, which is already documented separately as `squeeze-and-excite`. **Recommendation:** either merge MBConv into `inverted-residual` (noting the SE variant) or keep both but make the READMEs cross-reference the SE addition as the *only* difference rather than restating the whole block.

---

### Group 4 — BatchNorm vs. SyncBatchNorm (special-case / multi-GPU variant)
- `batch-norm` — "BatchNorm (Batch Normalization)"
- `syncbatchnorm` — "SyncBatchNorm"

**Rationale:** SyncBatchNorm is BatchNorm with an all-reduce of the per-replica statistics; the `syncbatchnorm` README itself frames it as "BatchNorm's statistics are computed over the batch, so with data parallelism each replica normalizes with its own local batch… SyncBatchNorm computes mean/variance across all replicas." The math, affine params, running stats, and inference behavior are identical to BatchNorm — the only addition is the cross-reduce communication step. The `batch-norm` README already lists SyncBatchNorm as a successor/variant. **Recommendation:** acceptable to keep separate for discoverability (it is a distinct operational pattern), but they should be explicitly cross-linked as "same op, multi-GPU sync variant" rather than presented as independent blocks.

---

### Group 5 — LayerNorm vs. RMSNorm (near-duplicate / drop-one-term variant)
- `layer-norm` — "LayerNorm (Layer Normalization)"
- `rmsnorm` — "RMSNorm"

**Rationale:** RMSNorm is LayerNorm with the mean-subtraction term removed (`x / RMS(x) · γ`); the `rmsnorm` README states "LayerNorm without the mean subtraction" and "Empirically interchangeable with LayerNorm for LLM training." The `layer-norm` README lists RMSNorm as a direct variant/successor. Functionality is otherwise identical (per-sample, per-token, learnable per-channel scale, pre-norm placement). **Recommendation:** keep both (they are distinct enough operationally and RMSNorm is the modern LLM default), but ensure both READMEs frame RMSNorm explicitly as "LayerNorm minus mean" rather than as an independent normalization scheme.

---

### Group 6 — Residual Block vs. Bottleneck Residual (special-case / wider-narrower variant)
- `residual-block` — "Residual Block (Identity Skip)"
- `bottleneck-residual` — "Bottleneck Residual Block"

**Rationale:** The bottleneck residual is the 1×1→3×3→1×1 variant of the basic two-3×3 residual block; `bottleneck-residual` lists "Predecessor: The basic residual block (two 3×3 convs)" and `residual-block` lists "Successor: Bottleneck residual (1×1-3×3-1×1)." Both share the identical identity-skip + post-add ReLU philosophy and the same "Used By" family (ResNet-50/101/152). The only functional difference is the channel-bottleneck sandwich. **Recommendation:** keep both (they are distinct enough in FLOPs/parameter profile), but they should be presented as a variant pair, not as independent inventions.

---

### Group 7 — Transformer Encoder / Decoder / Decoder-Only (overlapping family)
- `transformer-encoder-block` — "Transformer Encoder Block"
- `transformer-decoder-block` — "Transformer Decoder Block"
- `decoder-only-block` — "Decoder-Only Block (GPT/LLaMA style)"

**Rationale:** These three form a strict inheritance chain: decoder = encoder + masked self-attention + cross-attention; decoder-only = decoder − cross-attention. Each README explicitly states its relationship to the others ("Successor: Decoder block…", "Predecessor: Encoder block + seq2seq RNN decoders", "Drop the encoder-decoder cross-attention; keep only masked self-attention + FFN"). The `transformer-decoder-block` and `decoder-only-block` share the masked-self-attention + FFN core; the decoder-only is a strict subset of the decoder. **Recommendation:** the encoder and decoder blocks are legitimately distinct (encoder-decoder vs. decoder-only are different architectures), but `transformer-decoder-block` and `decoder-only-block` overlap heavily — the decoder-only is the decoder minus cross-attention. Consider merging the decoder-only into the decoder block as a "GPT/LLaMA variant" section, or keep all three but make the subset/superset relationship explicit in each README.

---

### Group 8 — Diffusion U-Net vs. U-Net Encoder-Decoder (special-case / conditioned variant)
- `unet-encoder-decoder` — "U-Net Encoder-Decoder with Skip Connections"
- `diffusion-unet` — "Diffusion U-Net (Latent Diffusion Denoiser)"

**Rationale:** The diffusion U-Net is the segmentation U-Net operating on VAE latents with cross-attention + timestep conditioning added; `diffusion-unet` lists "Predecessor: U-Net (segmentation); DDPM (pixel-space diffusion)" and `unet-encoder-decoder` lists "Successor: Diffusion U-Net (adds cross-attention + timestep conditioning)." The encoder-decoder-skip topology is identical; the only additions are the latent-space input and the conditioning injection. **Recommendation:** keep both (the conditioning makes them operationally distinct), but cross-link them as "same topology, diffusion-conditioned variant."

---

### Group 9 — DiT Block vs. Adaptive LayerNorm (component-vs-block overlap)
- `dit-block` — "DiT Block (Diffusion Transformer)"
- `adaptive-layernorm` — "Adaptive LayerNorm (adaLN / adaLN-Zero)"

**Rationale:** The DiT block's defining feature *is* adaLN-Zero conditioning; the `dit-block` README restates the full adaLN-Zero mechanism (`cond = MLP(timestep + class) → scale, shift, gate; adaLN(x) = scale ⊙ LayerNorm(x) + shift; zero-init`), which is verbatim the `adaptive-layernorm` functionality section. The `adaptive-layernorm` README's "Used By" table is identical to DiT's (DiT-XL/2, SD3, PixArt, Sora). **Recommendation:** `adaptive-layernorm` is a legitimate standalone component (reusable beyond DiT, e.g., in other conditional generators), so keep both, but the DiT README should reference `adaptive-layernorm` rather than re-documenting the mechanism inline.

---

### Group 10 — Cross-Attention vs. Gated Cross-Attention (special-case / gated variant)
- `cross-attention` — "Cross-Attention Block"
- `gated-cross-attention` — "Gated Cross-Attention (Multimodal Fusion)"

**Rationale:** Gated cross-attention is cross-attention with a tanh gate initialized to zero; the `gated-cross-attention` README states "Predecessor: Cross-attention (Transformer decoder); adapter layers (zero-init residual additions)" and `cross-attention` lists "Successor: Gated cross-attention (Flamingo, tanh-gated)." The Q-from-one-stream / K-V-from-another core is identical; the only addition is the zero-init scalar gate. **Recommendation:** keep both (the gating is a distinct training-stability technique), but frame gated as a variant of cross-attention, not an independent block.

---

### Group 11 — Sparse Attention / Sliding-Window / Native Sparse Attention (efficiency-variant family)
- `sparse-attention` — "Sparse Attention (Sparse Transformer)"
- `sliding-window-attention` — "Sliding-Window Attention"
- `native-sparse-attention` — "Native Sparse Attention (NSA)"

**Rationale:** All three are subquadratic attention variants that differ only in the sparsity pattern (strided+fixed vs. local window vs. learned top-k blocks). `sliding-window-attention` lists "Successor: Sparse attention (block-sparse, top-k selection)" and `native-sparse-attention` lists "Predecessor: Sliding window (local only)." `sparse-attention` already documents the strided + fixed pattern that subsumes the windowed case. **Recommendation:** keep all three (the patterns are genuinely different and each has distinct "Used By" models), but they should be presented as a family of "sparse attention patterns" with explicit cross-references rather than as independent mechanisms.

---

### Group 12 — Positional Encoding Family (six overlapping entries)
- `sinusoidal-positional-encoding`
- `learned-position-embedding`
- `relative-position-bias`
- `rope` — "Rotary Position Embedding"
- `yarn-rope-scaling` — "YaRN RoPE Scaling"
- `interleaved-nope` — "Interleaved NoPE (iRoPE)"
- `convolutional-positional-embedding`
- `fourier-positional-encoding`

**Rationale:** This is a large family where several entries are strict variants of others:
- `yarn-rope-scaling` is a post-training scaling method *for* RoPE; the `rope` README already mentions "NTK / YaRN scaling" as part of RoPE's functionality. YaRN is a RoPE extension, not an independent encoding.
- `interleaved-nope` is a layer-interleaving strategy that uses RoPE in some layers and *no* positional encoding in others; it is a usage pattern of RoPE/NoPE, not a new encoding.
- `sinusoidal`, `learned`, `relative-bias`, `rope`, `convolutional`, `fourier` are genuinely distinct encoding mechanisms.

**Recommendation:** the four core mechanisms (sinusoidal, learned, relative-bias, RoPE, convolutional, Fourier) are legitimately distinct. But `yarn-rope-scaling` and `interleaved-nope` are RoPE *extensions/usage patterns* and could be folded into the `rope` block as sections, or kept separate only if the catalog's policy is to treat "scaling method" as a distinct block type.

---

### Group 13 — SSM / Mamba Vision Family (overlapping scan variants)
- `selective-ssm` — "Selective State Space Model (Mamba Block)"
- `structured-ssm-s4` — "Structured State Space Model (S4)"
- `bidirectional-ssm` — "Bidirectional Selective SSM (Vim)"
- `ss2d-selective-scan` — "2D Selective Scan (SS2D / Cross-Scan)"
- `windowed-local-scan` — "Windowed Local Scan"
- `mambavision-mixer` — "MambaVision Mixer"
- `complex-mimo-ssm` — "Complex MIMO State-Space Mixer (Mamba-3)"

**Rationale:** `bidirectional-ssm`, `ss2d-selective-scan`, and `windowed-local-scan` form a strict inheritance chain: bidirectional (2 flat directions) → SS2D (4 directional routes) → windowed-local-scan (SS2D restricted to windows). Each README lists the previous as its direct predecessor. `mambavision-mixer` and `complex-mimo-ssm` are both "redesigned selective SSM block" variants that build on `selective-ssm`. **Recommendation:** the core S4 / selective-SSM / Mamba-3 entries are distinct generations. But `bidirectional-ssm`, `ss2d-selective-scan`, and `windowed-local-scan` are scan-direction variants of the same selective-SSM-over-images idea and could be consolidated into a single "2D selective scan variants" entry with sub-sections.

---

### Group 14 — Memory Bank Family (overlapping external-memory entries)
- `external-memory-bank` — "External Memory Bank"
- `streaming-memory-bank` — "Streaming Memory Bank"
- `memory-tokens` — "Memory Token Segment Recurrence"
- `query-memory-queue` — "Object-Query Memory Queue"

**Rationale:** `external-memory-bank` is the generic NTM/Memory-Network mechanism (content-addressable read/write matrix). The other three are domain-specific instantiations: `streaming-memory-bank` (SAM 2 video segmentation), `memory-tokens` (segment-recurrence for long-context transformers), `query-memory-queue` (StreamPETR 3D detection). Each README cross-references the others as "Related." They are operationally distinct (different memory contents, different read/write mechanisms), but `external-memory-bank` is the abstract parent and the others are special cases. **Recommendation:** keep all four (the domains and mechanisms differ enough), but ensure `external-memory-bank` is framed as the generic parent and the others reference it as such.

---

### Group 15 — Graph Convolution Family (overlapping propagation entries)
- `graph-convolution-layer` — "Graph Convolution Layer (GCN)"
- `lightgcn-propagation` — "LightGCN Propagation"
- `graph-diffusion-convolution` — "Convolution-Based Graph Diffusion"
- `graphsage-aggregator` — "GraphSAGE Neighborhood Aggregator"
- `variance-reduced-gcn-sampling` — "Variance-Reduced GCN Sampling"

**Rationale:** `lightgcn-propagation` is explicitly "GCN stripped to pure propagation (no W, no σ)" — a strict simplification of `graph-convolution-layer`. `graph-diffusion-convolution` is "GCN fixed k-hop aggregation" extended to multi-step diffusion. `variance-reduced-gcn-sampling` is a *sampling* strategy for GCN, not a new convolution. `graphsage-aggregator` is the inductive (sampling + aggregate) variant. Each lists `graph-convolution-layer` as predecessor. **Recommendation:** keep all (they serve different settings — CF, long-range, scalable, inductive), but `lightgcn-propagation` and `variance-reduced-gcn-sampling` are most clearly special-cases of GCN and should be framed as such.

---

### Group 16 — Contrastive Loss Family (overlapping loss entries)
- `infonce-contrastive-loss` — "InfoNCE (Contrastive) Loss"
- `triplet-loss` — "Triplet Loss"
- `pairwise-sigmoid-contrastive-loss` — "Pairwise Sigmoid Contrastive Loss"
- `barlow-twins-loss` — "Barlow Twins Loss"
- `vicreg-loss` — "VICReg Loss"
- `deep-graph-infomax` — "Deep Graph Infomax (DGI)"
- `hypergraph-contrastive-learning` — "Tri-Level Hypergraph Contrast"

**Rationale:** `infonce-contrastive-loss` is the parent loss; its README states "Superset of several losses: with a single negative it reduces to logistic BCE." `triplet-loss` is the predecessor (anchor-positive-negative margin). `pairwise-sigmoid-contrastive-loss` (SigLIP) is "InfoNCE with sigmoid instead of softmax" — a strict variant. `barlow-twins-loss` and `vicreg-loss` are both *negative-free* collapse-prevention losses (VICReg's README lists Barlow Twins as predecessor). `deep-graph-infomax` is "InfoNCE transferred to graphs with a global summary." `hypergraph-contrastive-learning` is "InfoNCE on hypergraph three levels." **Recommendation:** keep all (the variants matter for different training regimes), but `pairwise-sigmoid-contrastive-loss` and `vicreg-loss` are most clearly strict variants of `infonce-contrastive-loss` / `barlow-twins-loss` respectively and should be framed as such.

---

### Group 17 — CTR / Feature-Interaction Family (overlapping recommendation blocks)
- `wide-and-deep` — "Wide & Deep Block"
- `deepfm` — "DeepFM Block"
- `deep-cross-network` — "Deep & Cross Network (DCN / DCN-v2)"
- `gmf-mlp-fusion` — "GMF + MLP Fusion"
- `pairwise-dot-interaction` — "Pairwise Dot-Interaction"

**Rationale:** All five are "linear/memorization path + deep MLP path, fused at the logit" for CTR prediction. `deepfm` lists `wide-and-deep` as predecessor (replaces manual crosses with FM). `deep-cross-network` lists `wide-and-deep` and `deepfm` as predecessors. `gmf-mlp-fusion` and `pairwise-dot-interaction` are the CF-side analogues (element-wise product / dot products feeding an MLP). The family is a strict inheritance chain where each entry adds one mechanism to the previous. **Recommendation:** keep all (each is a distinct deployed architecture), but they should be presented as a family tree rather than as independent blocks.

---

### Group 18 — Initialization Family (overlapping init entries)
- `xavier-init` — "Xavier (Glorot) Initialization"
- `kaiming-init` — "Kaiming (He) Initialization"
- `equalized-lr` — "Equalized Learning Rate (GAN)"
- `residual-scaling-init` — "Residual-Depth-Scaled Init / muP Scaled Residuals"

**Rationale:** `kaiming-init` is the ReLU-aware correction of `xavier-init` (factor-of-2 difference, same variance-preservation goal). `equalized-lr` is "equivalent to He init but applied dynamically, not at initialization" per its own README — a runtime variant of `kaiming-init`. `residual-scaling-init` is a depth-dependent scaling of residual branches (GPT-2's 1/√N). **Recommendation:** `xavier` and `kaiming` are legitimately distinct (different activations). But `equalized-lr` is explicitly "equivalent to He init" and is the strongest candidate for merging into `kaiming-init` as a "runtime-scaling variant" section.

---

### Group 19 — Optimizer Family (overlapping optimizer entries)
- `sgd-momentum` — "SGD with Momentum"
- `adamw-optimizer` — "AdamW"
- `lion-optimizer` — "Lion Optimizer"

**Rationale:** These are genuinely distinct optimizers (SGD vs. adaptive AdamW vs. sign-based Lion), and each README cross-references the others in "See Also." `lion-optimizer` lists "SignSGD/Signum" as ancestor and "AdamW" as the comparison point. **Recommendation:** no merge needed — these are correctly distinct. Listed here only because they form a family; no redundancy action required.

---

### Group 20 — Gated FFN Family (overlapping FFN entries)
- `geglu-ffn` — "GeGLU Feed-Forward Network"
- `swiglu-ffn` — "SwiGLU Feed-Forward Network"

**Rationale:** The two blocks are structurally identical gated FFNs (`(gate ⊙ value) W_out`) differing only in the gate activation: GELU vs. Swish/SiLU. The `geglu-ffn` README states "Same FLOP family as SwiGLU — drop-in activation swap" and lists `swiglu-ffn` as "Sibling." **Recommendation:** keep both (the activation choice is empirical and model-family-specific), but frame them as siblings of the same gated-FFN template rather than as independent mechanisms.

---

### Group 21 — Perceiver Resampler vs. Q-Former (near-duplicate multimodal bridges)
- `perceiver-resampler` — "Perceiver Resampler"
- `q-former` — "Q-Former (Learned Query Bridge)"

**Rationale:** Both are "a fixed set of learned query tokens that cross-attend a frozen vision encoder's output to produce a constant-length visual summary for a downstream LLM." The `perceiver-resampler` README states "Predecessor: Perceiver (learned latents); BLIP-2's Q-Former (similar idea)" and the `q-former` README states "Predecessor: Perceiver (learned latents cross-attend inputs); Flamingo's Perceiver Resampler." The only real difference is insertion style (Perceiver Resampler is spliced via gated cross-attention into a frozen LLM; Q-Former outputs are linear-projected to LLM token space). Functionally they are near-duplicates. **Recommendation:** strong candidate for merging into a single "learned-query visual bridge" block with two insertion variants, or keep both but make the near-duplicate relationship explicit.

---

### Group 22 — Self-Distillation / Embedding-Distillation overlap
- `go-lsd-self-distillation` — "Global Optimal Localization Self-Distillation (GoLSD)"
- `embedding-distillation` — "Embedding Distillation"

**Rationale:** Both distill from a "teacher" that is the model's own final layer / frozen large encoder. `go-lsd-self-distillation` is cross-layer self-distillation within one model (D-FINE decoder layers); `embedding-distillation` is encoder-embedding distillation from a frozen teacher. They are operationally distinct (cross-layer vs. encoder-to-student), but both are "distill embeddings/distributions from a teacher that is part of the same pipeline." **Recommendation:** keep both (the distillation directions differ), but cross-link them as members of a "distillation" family.

---

## Summary

| # | Group | Blocks | Redundancy Type | Merge Priority |
|---|-------|--------|-----------------|----------------|
| 1 | GQA / MQA | 2 | alias / special-case | **High** |
| 2 | Flow Matching Objective / ODE | 2 | near-duplicate | **High** |
| 3 | Inverted Residual / MBConv | 2 | near-duplicate (SE delta) | Medium |
| 4 | BatchNorm / SyncBatchNorm | 2 | special-case (multi-GPU) | Low |
| 5 | LayerNorm / RMSNorm | 2 | drop-one-term variant | Low |
| 6 | Residual / Bottleneck Residual | 2 | special-case | Low |
| 7 | Transformer Enc/Dec/Decoder-Only | 3 | inheritance chain | Medium |
| 8 | U-Net / Diffusion U-Net | 2 | conditioned variant | Low |
| 9 | DiT Block / Adaptive LayerNorm | 2 | component-vs-block | Low |
| 10 | Cross-Attention / Gated Cross-Attention | 2 | gated variant | Low |
| 11 | Sparse / Sliding-Window / NSA | 3 | sparsity-pattern family | Low |
| 12 | Positional Encoding family | 8 | family + RoPE extensions | Medium |
| 13 | SSM / Mamba Vision scans | 7 | scan-direction chain | Medium |
| 14 | Memory Bank family | 4 | generic + special cases | Low |
| 15 | Graph Convolution family | 5 | inheritance chain | Low |
| 16 | Contrastive Loss family | 7 | parent + strict variants | Low |
| 17 | CTR Feature-Interaction family | 5 | inheritance chain | Low |
| 18 | Initialization family | 4 | family + runtime variant | Medium (equalized-lr) |
| 19 | Optimizer family | 3 | family (no redundancy) | None |
| 20 | GeGLU / SwiGLU FFN | 2 | activation-swap siblings | Low |
| 21 | Perceiver Resampler / Q-Former | 2 | near-duplicate bridges | **High** |
| 22 | GoLSD / Embedding Distillation | 2 | distillation family | Low |

**Highest-priority merges:** Groups 1 (GQA/MQA), 2 (Flow Matching Objective/ODE), and 21 (Perceiver Resampler/Q-Former) — these are the clearest cases where two separate entries describe the same mechanism, and the catalog's own READMEs already cross-reference each other as "the same idea."

**Note on catalog policy:** The `INDEX.md` header states "Total blocks: 297 — no duplication. Each block is a distinct structural or functional unit." Most groups above are *intentional* family relationships (predecessor/successor) rather than accidental duplicates. The catalog appears to treat "a named variant with its own README" as a distinct block even when it is a strict special case of another. The merge candidates above are the cases where this policy most clearly produces redundancy.
