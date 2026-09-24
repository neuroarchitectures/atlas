# 2D Selective Scan (SS2D / Cross-Scan)

## Design Philosophy

1D scans don't match 2D images: a single traversal breaks spatial adjacency. VMamba's SS2D traverses the image along *four routes* (horizontal L→R/R→L, vertical T→B/B→T) and merges the four outputs — bridging ordered 1D selective scans and non-sequential 2D structure.

## Functionality

- Unfold the image into 4 scan-order sequences → selective SSM per sequence → merge (sum) the four results.
- Inside VSS blocks: SS2D + gating + residual, linear complexity with global (per-route) receptive field.

## Used By

| Model | Role |
|-------|------|
| VMamba | VSS blocks across hierarchical stages |

## Features

- **2D-aware recurrence** — each spatial position is reachable from every direction.
- **Dynamic weights** — selectivity is input-dependent, unlike fixed-SSM 2D variants.

## Evolution

- **Predecessor**: Vim's bidirectional-ssm (two flat directions).
- **Successor**: localmamba's windowed-local-scan restricts routes to windows; MambaVision's mixer hybridizes with conv/attention.
