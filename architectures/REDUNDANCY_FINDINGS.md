# Architecture Catalog Redundancy Findings

**Scope:** All 503 entries under `architectures/` (verified: 503 folders on disk == 503 rows in `INDEX.md` == 503 `model.json` files; zero exact-duplicate display names; zero exact-duplicate `model.json` `id`/`name` values).

**Companion:** the block catalog was audited separately in [`../blocks/REDUNDANCY_FINDINGS.md`](../blocks/REDUNDANCY_FINDINGS.md). The two documents overlap in Group B below.

**Method:**
1. Structural fingerprinting — for every package, a signature of its component-type multiset and its `(type~name)` set, to catch packages that are byte-for-byte the same graph.
2. Cross-catalog comparison — folder-slug intersection and display-name intersection between `architectures/` and `blocks/`.
3. Fuzzy display-name matching within `architectures/`.

**Key negative result:** many architectures share an *identical* component-type multiset — every LLaMA-family 7B decoder (`llama3-8b`, `mistral-7b`, `internlm2-7b`, `olmo-7b`, `phi-2`, `yi-6b`), the BERT-family encoders (`bert-base`, `bge-base-en`, `chinese-roberta-wwm-ext`, `ernie-3.0-base-zh`), and the GPT-OSS pair. This is **intentional NAXS template reuse** (a block template + `repeat` with different parameters), not duplication. It is listed here only so it is not re-flagged. The only packages with a truly identical `(type~name)` signature are the ones named in that list, and all differ in parameters (layers, dims, vocab), so none are duplicates.

The genuine redundancy falls into three groups.

---

## Group A — Same slug in BOTH catalogs (4)

These are the strongest duplications: an architecture package and a block package with the **identical folder name and the same underlying concept**.

| slug | `architectures/` | `blocks/` |
|------|------------------|-----------|
| `deepfm` | DeepFM (Recommendation) | DeepFM Block |
| `diffusion-unet` | Diffusion UNet (Stable-Diffusion-style) (Generative) | Diffusion U-Net (Latent Diffusion Denoiser) |
| `flow-matching` | Flow Matching (Generative) | Flow Matching Objective |
| `wide-and-deep` | Wide & Deep (Recommendation) | Wide & Deep Block |

**Rationale:** The block catalog is defined as "reusable units extracted *from* the architectures", so a concept legitimately appears in both — but only for these 4 does the slug collide exactly, meaning the architecture entry is really block-level (a component graph) rather than a full model, or the block should carry a distinct slug.

**Recommendation:** decide a policy for the `Recommendation`/`Generative` small-graph entries. Either (a) keep the architecture entry and rename the block slug (e.g. `deepfm-block`), or (b) treat these 4 as block-only and remove the architecture package. Do **not** leave two identically-named folders across catalogs.

---

## Group B — `architectures/` "Architecture Block" category restates `blocks/` (12)

The `Architecture Block` category in `architectures/INDEX.md` (18 entries) documents reusable blocks that already have a dedicated entry in `blocks/`. Each pair below describes the same mechanism.

| `architectures/` | mirrors `blocks/` |
|------------------|-------------------|
| `attn-sliding-window` — "Sliding-Window Attention Block" | `sliding-window-attention` |
| `attn-sparse` — "Native Sparse Attention Block" | `native-sparse-attention` |
| `posenc-alibi` — "Linear Bias Positions (ALiBi)" | `alibi` |
| `posenc-learned` — "Learned Absolute Positions" | `learned-position-embedding` |
| `posenc-rope` — "Rotary Positions (RoPE)" | `rope` |
| `resnet-block` — "ResNet Block" | `residual-block` |
| `s4` — "S4" | `structured-ssm-s4` |
| `fff` — "FFF" | `fast-feedforward-network` |
| `darts` — "DARTS" | `darts-cell` |
| `transformer` — "Transformer" | `transformer-encoder-block` + `transformer-decoder-block` |
| `llama3-block` — "LLaMA-3 Decoder Block" | `decoder-only-block` + `grouped-query-attention` + `swiglu-ffn` + `rope` + `layer-norm` |
| `mixtral-block` — "Mixtral MoE Block" | `decoder-only-block` + `multi-gate-moe` + `grouped-query-attention` + `rope` |

**Rationale:** `posenc-*` are prefixed clones of the `alibi` / `learned-position-embedding` / `rope` blocks; `resnet-block` is a hand-expanded `residual-block`; `s4` is `structured-ssm-s4`; `fff`/`darts`/`transformer` mirror their block counterparts almost verbatim. `llama3-block` and `mixtral-block` are self-described "companions" to `llama3-8b`/`mixtral-8x7b` that decompose the decoder into exactly the blocks listed. Only `attn-full` has no block twin (it is the dense baseline) and is legitimately unique.

Note: this category also appears in the taxonomy — `README.md` lists it as an intentional category ("Full Attention, Sliding-Window, Native Sparse"). So this is a **policy** redundancy (two homes for block-level knowledge), not an accidental one.

**Recommendation:** pick one home for block-level entries. Cleanest: keep blocks canonical in `blocks/` and reduce the `architectures/` `Architecture Block` entries to thin pointers (or drop them), keeping only those that are *not* in `blocks/` (e.g. `attn-full`, and the graph/temporal entries `mile`, `mile-block`, `thnn`, `tnif`, `masa`).

---

## Group C — Within-catalog candidates (2)

### C1 — `ncf` vs `neumf` (identical display name)
- `ncf` — "Neural Collaborative Filtering"
- `neumf` — "Neural Collaborative Filtering (NeuMF)"

**Rationale:** Both are titled "Neural Collaborative Filtering". NeuMF is the *fused GMF+MLP* variant proposed in the same paper as the NCF framework. The prior cleanup (commit `589f0f1`) inspected them and kept both as "structurally distinct graphs".

**Recommendation:** keep both, but make the titles unambiguous — e.g. `ncf` → "Neural Collaborative Filtering (GMF path)" and `neumf` → "Neural Collaborative Filtering (NeuMF / GMF+MLP fusion)" — so the identical display name does not read as a duplicate.

### C2 — `mile` vs `mile-block` (companion pair)
- `mile` — "MILE" (the full multi-level graph-embedding architecture)
- `mile-block` — "MILE Coarsening-Refinement Block" ("the companion to the full mile entry")

**Rationale:** `mile-block` expands MILE's three phases (coarsen / base-embed / refine) into individual ops. It is the same pattern as `llama3-block`, `mixtral-block`, and `resnet-block` (companion decompositions of a full model). Not a strict duplicate — but `mile-block` was **orphaned**: it existed on disk but was absent from `INDEX.md` and referenced by no sibling README.

**Recommendation:** keep as a companion. Resolved as part of the index-integrity fix below (it is now indexed).

---

## Index integrity (fixed alongside this review)

These were structural defects (not semantic duplication) found while comparing the catalog against disk; all are now corrected:

- **10 broken `INDEX.md` links** — the first-column ID/link used the `model.json` `id` while the folder had been renamed. Repointed to the real folders: `ernie-3.0-base-zh`, `glm-4.5-air`, `gpt-2`, `graphsage`, `kimi-k2.6`, `llava-1.5-7b`, `mamba`, `modernbert`, `pythia-1.4b`, `qwen2.5-7b`.
- **1 orphan folder** — `mile-block` added to `INDEX.md`.
- **Count drift** — `INDEX.md` header and `README.md` corrected to the true 503 packages (was 501), CV 155→154, NLP 105→103, Architecture Block 17→18, and the conformance "288 model.json" figure → 503.

**Residual note (not fixed):** in the 10 repointed packages, the `model.json` `id` still differs from the folder slug (e.g. folder `gpt-2` vs `id: gpt2-small`; folder `graphsage` vs `id: graph-sage-rec`). Aligning the two would require touching `model.json` and is out of scope for the index fix.

---

## Summary

| # | Group | Entries | Type | Action |
|---|-------|---------|------|--------|
| A | Shared slug in both catalogs | 4 | exact cross-catalog duplicate | **High** — decide policy, remove or rename one side |
| B | Architecture Block ↔ blocks | 12 | block-level knowledge in two homes | Medium — pick one canonical home |
| C1 | `ncf` / `neumf` | 2 | identical display name | Low — disambiguate titles |
| C2 | `mile` / `mile-block` | 2 | companion (was orphaned) | Done — indexed |

**Highest-priority:** Group A. Four folder names exist verbatim in both `architectures/` and `blocks/`, which is the clearest case of the same concept catalogued twice.
