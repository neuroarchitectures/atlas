# References: DNE-PL

## Papers

- [`From_properties_to_links_Deep_network_embedding_on_incomplet_Wang_Profile_Li_2017.md`](papers/From_properties_to_links_Deep_network_embedding_on_incomplet_Wang_Profile_Li_2017.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **MVC-DNE** — No official public repository found; the method is described in full detail in the source paper
- **Paper (ACM DL)** — https://doi.org/10.1145/3132847.3132975 — Official CIKM '17 paper page

## Related

**Predecessors:**
- DeepWalk (Perozzi et al., 2014) — Skip-gram on random walks; topology-only, transductive
- LINE (Tang et al., 2015) — First/second-order proximity; topology-only
- GraRep (Cao et al., 2015) — k-order proximity factorization; topology-only
- node2vec (Grover & Leskovec, 2016) — Biased random walks; topology-only, transductive
- TADW (Yang et al., 2015) — Text-attributed DeepWalk; shallow text-topology fusion
- SDNE (Wang et al., 2016) — Deep autoencoder for network embedding; topology-only

**Successors:**
- GraphSAGE (Hamilton et al., 2017) — Inductive GNN via sampling and aggregation; also learns a mapping function
- GCN (Kipf & Welling, 2017) — Spectral graph convolution; end-to-end trainable
- DANE (Li et al., 2017) — Dynamic attributed network embedding; similar goals for dynamic graphs

**Alternatives:**
- MVNE (Qu et al., 2017) — Attention-based multi-view network embedding
- SNEA (Wang et al., 2017) — Attributed signed network embedding
- CANE (Tu et al., 2017) — Context-aware network embedding for relation modeling
