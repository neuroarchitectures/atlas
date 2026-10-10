# Paper (Oliveira et al. 2026)

> Source: `https://arxiv.org/abs/2609.11932`

---

**Throughput per Megabyte: A Pilot Benchmark of Language-Stack Efficiency for Self-Hosted HTTP Services on a Raspberry Pi 5**

William Oliveira

**Abstract**

Cloud-centric web benchmarks miss constraints that matter for self-hosted services on ARM64 single-board computers, especially idle RAM footprint and energy per request. We ran a pilot benchmark on one Raspberry Pi 5, measuring equivalent SQLite-backed CRUD APIs implemented in Go 1.26/net/http, Rust 1.95/Axum, Python 3.13/FastAPI+Granian, Node.js 24/Fastify, and .NET 10 Native AOT across N=50 randomized runs per stack and endpoint, with a separate concurrency sweep and memory time-series collection. Rust achieved the highest weighted throughput-to-RAM ratio (310.25 req/s/MB, 2.8x Go) because its weighted peak RSS was 7.36 MB, while .NET, Node.js, Go, and Rust formed a raw-throughput cluster with overlapping bootstrap CIs (2,125--2,461 req/s) and Python remained below that cluster at 969 req/s. The concurrency sweep on GET /items/:id showed divergent scaling: .NET reached 26,209 req/s at c=240, Go and Rust plateaued near c=120, Node.js saturated near 9,700 req/s, and Python stayed below 2,300 req/s; memory snapshots showed Rust holding a flat 7.8 MB RSS while Node.js grew by 117 MB without visible GC drops. For RAM-constrained self-hosted CRUD-over-SQLite deployments on this Pi 5 unit, the data support Rust as the most resource-efficient measured stack, .NET as the strongest high-concurrency stack, and Go as a practical option where ecosystem breadth matters; replication is needed before extending these findings to other devices or storage backends.

---

- **arXiv ID**: `2609.11932`
- **Published**: 2026-05-23
- **Categories**: cs.PF, cs.DC, cs.PL
- **PDF**: https://arxiv.org/pdf/2609.11932
