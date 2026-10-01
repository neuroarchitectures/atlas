# Neural Architecture Atlas

> A curated catalog of neural network architectures and reusable building blocks, exchanged in [NAXS](https://github.com/neuroarchitectures/naxs) format.

**Project:** Veya Architect — Autonomous Neural Architecture Design (ANAD)

---

## Overview

The **Neural Architecture Atlas** is a structured knowledge base of neural network architectures and the reusable blocks they are built from. Every architecture is stored as a machine-readable directed graph in **NAXS (Neural Architecture Exchange Specification)** — a vendor-independent, framework-agnostic JSON format that describes *what* an architecture is (components, parameters, connectivity) without prescribing *how* it runs.

NAXS is not an execution format (ONNX), a training config, a compiler IR, or a weight checkpoint — it captures topology only.

**Specification:** <https://github.com/neuroarchitectures/naxs>

---

## Contents

- **374 architecture packages** in [`architectures/`](./architectures/) — full directed graphs with documentation and diagrams. See the [Architecture Index](./architectures/INDEX.md).
- **297 reusable building blocks** in [`blocks/`](./blocks/) — deduplicated structural units extracted from the real architectures. See the [Block Index](./blocks/INDEX.md).
- **Reference material** in [`references/`](./references/) — taxonomy and evolution maps, including the [Neural Architecture Atlas taxonomy (1943–2026)](./references/NEURAL-ARCHITECTURE-ATLAS.md).

---

## Directory Layout

```
atlas/
├── architectures/            ← 374 architecture packages
│   ├── INDEX.md             ← master index
│   └── <name>/
│       ├── model.json       ← NAXS document (the graph)
│       ├── architecture.md  ← human-readable documentation
│       ├── README.md        ← package summary
│       ├── assets/          ← diagram.svg, diagram.png
│       └── references/      ← source papers
├── blocks/                  ← 297 reusable building blocks
│   ├── INDEX.md             ← master index
│   └── <name>/README.md     ← design, usage, evolution
└── references/              ← taxonomy & surveys
```

---

## Categories

All 374 architectures are classified into 11 domain categories:

| Category | Count | Examples |
|----------|-------|----------|
| Computer Vision | 128 | ResNet-50, ViT-B/16, YOLO-v11, SAM, DINO |
| NLP | 87 | BERT-Base, GPT-2, Llama3-8B, T5-Small |
| Graph | 53 | GCN, GraphSAGE, GAT, LightGCN |
| Recommendation | 22 | DeepFM, NCF, BERT4Rec, Wide-and-Deep |
| Generative | 22 | DiT-XL/2, Diffusion U-Net, DDPM, VAE, StyleGAN |
| Architecture Block | 17 | Full Attention, Sliding-Window, Native Sparse |
| Scientific | 10 | 3DGS, NeRF, MonoSplat, Instant-NGP |
| Multimodal | 9 | BLIP-2, CLIP, Flamingo, LLaVA-1.5, ImageBind |
| Time Series | 8 | Patch-TST, N-BEATS, Informer, Autoformer |
| Reinforcement Learning | 8 | Dreamer-V3, PPO, SAC, Decision Transformer |
| Audio | 8 | Whisper-Small, Wav2Vec2-Base, EnCodec, WaveNet |

---

## The NAXS Format

Each `model.json` is a NAXS document: a directed graph of typed, parameterized components connected by edges. A minimal example:

```json
{
  "spec_version": "0.1",
  "id": "bert-base",
  "name": "BERT-Base",
  "category": "NLP",
  "real_param_count": 110100000,
  "components": [
    {
      "id": "n1", "type": "input", "name": "Input",
      "params": { "shape": [1, 512, 768] },
      "inputs": [], "outputs": ["n2"]
    },
    {
      "id": "n2", "type": "embedding", "name": "Embedding",
      "params": { "vocabSize": 30522, "embeddingDim": 768, "maxSeqLen": 512 },
      "inputs": ["n1"], "outputs": ["n3", "n4"],
      "scope": "embeddings"
    }
  ]
}
```

Edges are doubly represented: each component lists its `inputs` (source IDs) and `outputs` (destination IDs), giving O(1) lookup in both directions. The full field reference, operator catalog (79 types across 9 categories), and JSON schema live in the [NAXS specification](https://github.com/neuroarchitectures/naxs).

---

## Package Contents

**Architecture packages** (`architectures/<name>/`) each contain the NAXS graph (`model.json`), human-readable documentation (`architecture.md`, `README.md`), vector and raster diagrams (`assets/`), and source papers (`references/`).

**Block packages** (`blocks/<name>/`) each document a reusable unit's design philosophy, functionality, which architectures instantiate it, key features, and its design lineage. Blocks span 16 families: convolutional, transformer, GNN, recurrent/SSM, recommendation, generative, audio, multimodal, training primitives, hypergraph/topology, graph embeddings, and 3D vision/flow/tracking.

---

## Conformance

All 288 `model.json` files declare `spec_version`, carry a `category`, use valid kebab-case component IDs (no dots — `.` is reserved for scope paths), include doubly-represented connectivity, and pass JSON Schema validation against the NAXS schema.

---

## Acknowledgments

This atlas draws on the curated architecture knowledge and taxonomy work published at [**neurarch.com**](https://neurarch.com), which provides the foundational surveys and evolution maps that informed the classification and structure of this catalog.
