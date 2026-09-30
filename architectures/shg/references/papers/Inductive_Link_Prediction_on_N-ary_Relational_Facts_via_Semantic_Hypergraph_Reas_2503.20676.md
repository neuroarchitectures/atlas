# Paper (Yin et al. 2025)

> Source: `https://arxiv.org/abs/2503.20676`

---

**Inductive Link Prediction on N-ary Relational Facts via Semantic Hypergraph Reasoning**

Gongzhu Yin, Hongli Zhang, Yuchen Yang, Yi Luo

**Abstract**

N-ary relational facts represent semantic correlations among more than two entities. While recent studies have developed link prediction (LP) methods to infer missing relations for knowledge graphs (KGs) containing n-ary relational facts, they are generally limited to transductive settings. Fully inductive settings, where predictions are made on previously unseen entities, remain a significant challenge. As existing methods are mainly entity embedding-based, they struggle to capture entity-independent logical rules. To fill in this gap, we propose an n-ary subgraph reasoning framework for fully inductive link prediction (ILP) on n-ary relational facts. This framework reasons over local subgraphs and has a strong inductive inference ability to capture n-ary patterns. Specifically, we introduce a novel graph structure, the n-ary semantic hypergraph, to facilitate subgraph extraction. Moreover, we develop a subgraph aggregating network, NS-HART, to effectively mine complex semantic correlations within subgraphs. Theoretically, we provide a thorough analysis from the score function optimization perspective to shed light on NS-HART's effectiveness for n-ary ILP tasks. Empirically, we conduct extensive experiments on a series of inductive benchmarks, including transfer reasoning (with and without entity features) and pairwise subgraph reasoning. The results highlight the superiority of the n-ary subgraph reasoning framework and the exceptional inductive ability of NS-HART. The source code of this paper has been made publicly available at https://github.com/yin-gz/Nary-Inductive-SubGraph.

---

- **arXiv ID**: `2503.20676`
- **Published**: 2025-03-26
- **Categories**: cs.AI, cs.LG
- **PDF**: https://arxiv.org/pdf/2503.20676
