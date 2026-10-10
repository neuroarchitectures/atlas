# Paper (Chumachenko et al. 2025)

> Source: `https://arxiv.org/abs/2511.20478`

---

**NVIDIA Nemotron Parse 1.1**

Kateryna Chumachenko, Amala Sanjay Deshmukh, Jarno Seppanen, Ilia Karmanov, Chia-Chih Chen, Lukas Voegtle, Philipp Fischer, Marek Wawrzos, Saeid Motiian, Roman Ageev et al.

**Abstract**

We introduce Nemotron-Parse-1.1, a lightweight document parsing and OCR model that advances the capabilities of its predecessor, Nemoretriever-Parse-1.0. Nemotron-Parse-1.1 delivers improved capabilities across general OCR, markdown formatting, structured table parsing, and text extraction from pictures, charts, and diagrams. It also supports a longer output sequence length for visually dense documents. As with its predecessor, it extracts bounding boxes of text segments, as well as corresponding semantic classes. Nemotron-Parse-1.1 follows an encoder-decoder architecture with 885M parameters, including a compact 256M-parameter language decoder. It achieves competitive accuracy on public benchmarks making it a strong lightweight OCR solution. We release the model weights publicly on Huggingface, as well as an optimized NIM container, along with a subset of the training data as part of the broader Nemotron-VLM-v2 dataset. Additionally, we release Nemotron-Parse-1.1-TC which operates on a reduced vision token length, offering a 20% speed improvement with minimal quality degradation.

---

- **arXiv ID**: `2511.20478`
- **Published**: 2025-11-25
- **Categories**: cs.LG
- **PDF**: https://arxiv.org/pdf/2511.20478
