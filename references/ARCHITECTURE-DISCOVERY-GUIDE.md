# Architecture Discovery Guide

> **How to find new architectures to add to the Neural Architecture Atlas**

**Status:** Living methodology document
**Scope:** Concrete strategies for discovering architectures not yet in the Atlas (310 packages as of 2026-09-30)

---

## 1. Taxonomy Gap Analysis

The [Neural Architecture Atlas taxonomy](./NEURAL-ARCHITECTURE-ATLAS.md) names many architecture families and individual models in its evolution maps. The most reliable discovery method is to systematically cross-reference every named architecture in the taxonomy against the existing packages.

### How to do it

1. Extract every architecture name mentioned in the taxonomy's family trees (Sections 5–19).
2. Compare against `architectures/INDEX.md`.
3. Any named architecture without a package is a candidate.

### Known gaps (as of 2026-09-30)

The taxonomy explicitly names these families/variants that may still lack packages:

| Family | Named in taxonomy | Status |
|--------|-------------------|--------|
| RNN | Echo State Network | Missing |
| Autoencoder | Sparse AE, Denoising AE, Contractive AE, β-VAE | Missing (only Autoencoder + VAE present) |
| Boltzmann | Deep Belief Network | Missing |
| CNN | Inception v1/v2/v3/v4 (only generic `inception/` exists) | Partial |
| Attention | Memory Networks variants | Partial |
| Diffusion | Score-based Models | Missing |
| Neural Field | Occupancy Networks, Neural SDF | Missing |
| Foundation Model | RAG-like systems | Missing |
| ANAD | Architecture Genome, Neural Program | Conceptual — not yet instantiated |

### Action

```bash
# Quick gap check: extract named architectures from taxonomy, diff against packages
grep -oE '\b[A-Z][A-Za-z0-9-]+\b' references/NEURAL-ARCHITECTURE-ATLAS.md | sort -u > /tmp/taxonomy_names.txt
ls architectures/ | sort -u > /tmp/package_names.txt
comm -23 /tmp/taxonomy_names.txt /tmp/package_names.txt
```

---

## 2. Category Gap Analysis

The Atlas has 11 categories with uneven coverage. Thin categories are prime targets for expansion.

| Category | Current count | Priority |
|----------|--------------|----------|
| Time Series | 1 (Patch-TST) | **High** — needs TimesNet (exists), N-BEATS, TFT, Informer, Autoformer, etc. |
| Reinforcement Learning | 3 | **High** — needs PPO networks, SAC, Rainbow, IMPALA |
| Audio | 4 | **Medium** — needs SoundStream, AudioLM, MusicGen, EnCodec variants |
| Scientific | 6 | **Medium** — needs Instant-NGP, Plenoxels, Neural SDF |
| Multimodal | 7 | **Medium** — needs Perceiver, ImageBind, 3LLaVA-1.6 |
| Generative | 12 | **Low** — needs StyleGAN, BigGAN, Glow, Normalizing Flow |

### Action

For each thin category, search arXiv/PapersWithCode for the top-cited architectures in that domain and create packages for the ones not yet present.

---

## 3. Paper-Driven Discovery (arXiv + OpenAlex)

Query academic APIs for recent high-impact architecture papers.

### arXiv API (rate-limited, 3s between requests)

```python
import urllib.request, urllib.parse, xml.etree.ElementTree as ET

def search_arxiv(query, max_results=10):
    url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&max_results={max_results}"
    resp = urllib.request.urlopen(url)
    root = ET.fromstring(resp.read())
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip()
        arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
        print(f"{arxiv_id}: {title}")
```

### OpenAlex API (no aggressive rate limiting — preferred)

```python
import urllib.request, json

def search_openalex(query, per_page=25):
    url = f"https://api.openalex.org/works?search={urllib.parse.quote(query)}&per_page={per_page}&sort=cited_by_count:desc"
    resp = urllib.request.urlopen(url)
    data = json.loads(resp.read())
    for work in data['results']:
        print(f"{work['cited_by_count']:>6} cites | {work['title']}")
```

### Recommended queries by domain

| Domain | Search query |
|--------|-------------|
| Vision | `"neural network architecture" vision transformer 2024 2025` |
| NLP | `"language model architecture" efficient 2024 2025` |
| Graph | `"graph neural network" architecture novel 2024 2025` |
| Time Series | `"time series forecasting" architecture transformer 2024` |
| Diffusion | `"diffusion model" architecture novel 2024 2025` |
| SSM | `"state space model" architecture mamba 2024 2025` |

### PapersWithCode leaderboards

Browse <https://paperswithcode.com/sota> for each task. The top-performing models on each leaderboard are strong candidates — they represent architectures that the community has validated as effective.

---

## 4. Model Registry Scanning

Popular model hubs reveal architectures that have real-world traction.

### HuggingFace model hub

- Browse <https://huggingface.co/models> sorted by downloads
- Filter by task (text-generation, image-classification, etc.)
- Any model with >100K downloads that isn't in the Atlas is a candidate

### Key model families to check

| Hub | What to look for |
|-----|-----------------|
| HuggingFace | Popular models not yet packaged (e.g., Cohere Command-R, AI21 Jamba, Reka) |
| TorchVision | Classification/detection/segmentation models in `torchvision.models` |
| TIMM | All models in `timm` library (`timm.list_models()`) |
| Ultralytics | YOLO variants and other detection models |
| MMSegmentation | Segmentation architectures |

### Action

```python
# List all timm models not yet in the atlas
import timm
models = timm.list_models()
# Compare against architectures/ directory names
```

---

## 5. Block-Driven Discovery

The Atlas has **251 reusable building blocks** in `blocks/`. New architectures can be discovered by looking for novel combinations of existing blocks, or for blocks that only appear in one architecture (suggesting the family is under-explored).

### Strategy

1. For each block in `blocks/INDEX.md`, check the "Used By" column.
2. Blocks used by only **1 architecture** indicate a narrow family — look for other architectures that use the same block.
3. Block **combinations** not yet seen (e.g., "diffusion + graph neural network") may point to unexplored architectures.

### Action

```bash
# Find blocks used by only one architecture (narrow families)
grep -r "Used By" blocks/*/README.md | grep -v "," | head -30
```

---

## 6. Evolution Map Traversal

The taxonomy's evolution map (Section 21) shows how architecture families branch. Each branch point is a discovery opportunity:

```
Transformer
    │
    ├── NLP ──────► [encoder-only, decoder-only, encoder-decoder]
    ├── Vision ────► [ViT, Swin, CSWin, DeiT, ...]
    ├── Multimodal ► [CLIP, BLIP, Flamingo, LLaVA, ...]
    ├── MoE ───────► [Mixtral, DeepSeek-MoE, ...]
    ├── Memory ────► [NTM, Memory Network, RMT, ...]
    └── Retrieval ─► [RAG architectures — MISSING]
```

### Strategy

Walk each branch to its leaves. If a leaf has no package, it's a candidate. Then look for **new branches** — architectures that combine two branches (e.g., SSM + GNN, Diffusion + Neural Field).

---

## 7. Conference & Venue Scanning

Scan recent proceedings of top ML/AI venues for novel architecture papers.

### Priority venues

| Venue | Focus | URL |
|-------|-------|-----|
| NeurIPS | Broad ML | papers.nips.cc |
| ICML | Broad ML | proceedings.mlr.press |
| ICLR | Broad ML | openreview.net |
| CVPR / ICCV / ECCV | Vision | openaccess.thecvf.com |
| ACL / EMNLP | NLP | aclanthology.org |
| KDD / WWW | Graph/Recsys | dl.acm.org |
| SIGGRAPH | Neural fields/3D | diglib.eg.org |

### Action

For each venue, scan the most recent year's accepted papers list. Filter for "architecture" or "model" in the title. Cross-reference against existing packages.

---

## 8. Primitive Combination Exploration

The taxonomy defines **9 fundamental primitives** (Section 22). The Atlas covers known combinations, but many cross-primitive combinations are unexplored.

### Primitive combination matrix

| | Convolution | Recurrence | Attention | Message Passing | Memory | Routing | Field | Diffusion | SSM |
|---|---|---|---|---|---|---|---|---|---|
| **Convolution** | ✅ CNN | ✅ CRNN | ✅ ConvTransformer | ❓ | ❓ | ✅ CondConv | ✅ NeRF | ❓ | ❓ |
| **Recurrence** | | ✅ RNN | ✅ AttentiveRNN | ❓ | ✅ LSTM | ❓ | ❓ | ❓ | ✅ Mamba |
| **Attention** | | | ✅ Transformer | ✅ GraphTransformer | ✅ RMT | ✅ MoE | ❓ | ✅ DiT | ✅ Hybrid |
| **Message Passing** | | | | ✅ GNN | ❓ | ❓ | ❓ | ❓ | ❓ |
| **Memory** | | | | | ✅ NTM | ❓ | ❓ | ❓ | ❓ |
| **Routing** | | | | | | ✅ MoE | ❓ | ❓ | ❓ |
| **Field** | | | | | | | ✅ NeRF | ❓ | ❓ |
| **Diffusion** | | | | | | | | ✅ DDPM | ❓ |
| **SSM** | | | | | | | | | ✅ S4 |

**❓ = unexplored combination** — a research opportunity and a potential new architecture family.

### Strategy

For each ❓ cell, search arXiv for papers combining those primitives. If papers exist, create packages. If not, this is an ANAD research opportunity.

---

## 9. Automated Discovery Pipeline

For systematic, repeatable discovery, build a script that combines the above methods.

### Recommended pipeline

```
1. Taxonomy gap scan        → candidates.txt
2. Category gap fill        → append to candidates.txt
3. arXiv/OpenAlex query     → append recent papers
4. HuggingFace scan         → append popular models
5. TIMM/TorchVision scan    → append vision models
6. Deduplicate              → final_candidates.txt
7. For each candidate:
   a. Verify it's a real architecture (not a training trick)
   b. Check it's not already in the Atlas (fuzzy match)
   c. Find its primary paper (arXiv ID)
   d. Create package: model.json + architecture.md + README.md
   e. Fetch reference papers
   f. Update INDEX.md
```

### Script template

```python
#!/usr/bin/env python3
"""Discover new architecture candidates for the Atlas."""

import json, os, urllib.request, urllib.parse

EXISTING = set(os.listdir("architectures")) - {"INDEX.md"}

def openalex_search(query, limit=10):
    url = f"https://api.openalex.org/works?search={urllib.parse.quote(query)}&per_page={limit}&sort=cited_by_count:desc&filter=type:article"
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        data = json.loads(resp.read())
        return [(w['title'], w.get('cited_by_count', 0)) for w in data.get('results', [])]
    except Exception:
        return []

QUERIES = [
    "novel neural network architecture 2025",
    "new vision transformer architecture 2025",
    "state space model architecture 2025",
    "graph neural network architecture 2025",
    "diffusion model architecture 2025",
    "time series forecasting architecture 2025",
    "efficient language model architecture 2025",
    "multimodal foundation model architecture 2025",
]

for q in QUERIES:
    results = openalex_search(q)
    for title, cites in results:
        # Fuzzy match against existing packages
        title_lower = title.lower()
        if not any(pkg in title_lower for pkg in EXISTING):
            print(f"[{cites:>6} cites] {title}")
```

---

## 10. Quality Criteria for New Packages

Not every model deserves a package. Use these criteria to prioritize:

| Criterion | Why it matters |
|-----------|---------------|
| **Novel topology** | The architecture introduces a new connectivity pattern, not just hyperparameter tuning |
| **Citation count > 100** | Community-validated significance |
| **Adoption** | Used in production or by >1 research group |
| **Structural distinctness** | Not a minor variant of an existing package (e.g., ResNet-18 vs ResNet-50 don't both need packages) |
| **Primitive innovation** | Introduces or combines primitives in a new way |
| **Historical significance** | Even old/cited architectures (ADALINE, MADALINE) belong if they represent a structural milestone |

### Anti-criteria (do NOT add)

- Training tricks without architectural change (e.g., new loss function)
- Hyperparameter variants of existing architectures
- Model weights/checkpoints without architectural novelty
- Framework-specific implementations

---

## 11. Summary: Discovery Priority Queue

Ranked by ease and impact:

1. **Taxonomy gaps** (Section 1) — known missing architectures, easy to verify
2. **Thin categories** (Section 2) — Time Series, RL, Audio need filling
3. **Model registries** (Section 4) — HuggingFace/TIMM models with high adoption
4. **arXiv/OpenAlex search** (Section 3) — recent high-citation papers
5. **Block combinations** (Section 5) — unexplored block pairings
6. **Primitive matrix** (Section 8) — cross-primitive combinations
7. **Conference scanning** (Section 7) — labor-intensive but catches bleeding-edge work
8. **ANAD exploration** (Section 28 of taxonomy) — machine-discovered architectures

---

## Appendix: Current Atlas Statistics

- **310 architecture packages** (288 with visual diagrams, 22 with model.json pending diagrams)
- **251 reusable building blocks**
- **11 categories**: CV (114), NLP (72), Graph (53), Recsys (22), Arch Block (16), Generative (12), Multimodal (7), Scientific (6), Audio (4), RL (3), Time Series (1)
- **1,114+ reference papers** across all packages
