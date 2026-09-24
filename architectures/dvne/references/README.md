# References: DVNE

## Papers

- [`Deep_Variational_Network_Embedding_in_Wasserstein_Space_Cui_Wang_Zhu_2018.md`](papers/Deep_Variational_Network_Embedding_in_Wasserstein_Space_Cui_Wang_Zhu_2018.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **DVNE reference resources** — The paper (Zhu, Cui, Wang, Zhu, KDD '18) provides the full method description; check Tsinghua NEL (Network Embedding Lab) and Peng Cui's group pages for code releases: http://pel.cuc.edu.cn/ and https://cuip.thu.edu.cn/
- **Tsinghua NEL group** — https://github.com/thunlp (THUNLP organization hosts related network embedding code; DVNE may be available via the group's repositories)

## Related

- **Graph2Gauss (G2G)** (Bojchevski & Günnemann, 2017) — https://github.com/abojchevski/graph2gauss — Direct predecessor; first to embed nodes as Gaussian distributions, but uses KL divergence (asymmetric, no triangle inequality) and all-pairs shortest paths — the limitations DVNE addresses.
- **DeepWalk** (Perozzi et al., 2014) — https://github.com/phanein/deepwalk — Random-walk + Skip-Gram point embedding predecessor.
- **LINE** (Tang et al., 2015) — First/second-order proximity point embedding predecessor.
- **node2vec** (Grover & Leskovec, 2016) — https://github.com/aditya-grover/node2vec — Biased random walk embedding predecessor.
- **SDNE** (Wang et al., 2016) — Deep autoencoder embedding predecessor.
- **GAE / Graph VAE** (Kipf & Welling, 2016) — https://github.com/tkipf/gae — Variational graph embedding foundation.
- **Wasserstein GAN / Optimal Transport** — The 2-Wasserstein distance theory that DVNE adopts as its distribution metric.
- **Gaussian word embeddings** (Vilnis & McCallum, 2014) — Origin of distributional embeddings that DVNE extends to networks.
