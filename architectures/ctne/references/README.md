# References: Continuous-Time Dynamic Network Embeddings

## Papers

- [`Continuous_Time_Dynamic_Network_Embeddings_18_Embeddings_2018.md`](papers/Continuous_Time_Dynamic_Network_Embeddings_18_Embeddings_2018.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **CTNE — WWW '18 paper resources** — The paper (Nguyen, Lee, Rossi, Ahmed, Koh, Kim, WWW '18) provides the full method description; check the authors' institutional pages (Adobe Research, Georgia Tech) for code releases.
- **N. K. Ahmed — research page** — https://www.nkohail.com/ or https://github.com/nkohall (co-author; may host related temporal graph code and datasets)

## Related

- **DeepWalk** (Perozzi et al., 2014) — https://github.com/phanein/deepwalk — Direct predecessor; CTNE extends DeepWalk's random-walk + Skip-Gram to temporal walks.
- **LINE** (Tang et al., 2015) — Static first/second-order proximity embedding predecessor (baseline).
- **node2vec** (Grover & Leskovec, 2016) — https://github.com/aditya-grover/node2vec — Static biased random walk embedding predecessor (baseline).
- **TGNet** (Song et al., CIKM 2018) — Snapshot-based dynamic embedding; the discrete-snapshot approach whose limitations CTNE addresses.
- **Temporal network analysis / temporal reachability** — Prior work defining temporal walks and temporal connectivity that CTNE operationalizes.
- **Temporal/evolving GNNs** (TGAT, TGN, EvolveGCN) — Successor neural methods for dynamic graphs extending temporal representation learning with attention/recurrent mechanisms.
- **Point-process-based dynamic network models** — Methods modeling edge formation as temporal point processes, a continuous-time framing related to CTNE.
- **DANE** (Li et al., 2017) — Dynamic attributed network embedding (offline + online update); sibling dynamic-embedding approach (snapshot/matrix-perturbation based).
