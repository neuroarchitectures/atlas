# Slimmable Supernet (Width/Structure Switching)

## Design Philosophy

Train once, deploy many: make channel width (slimmable) — or depth, kernel size, and resolution (Once-for-All progressive shrinking) — *runtime-switchable*, training all configurations jointly so any sub-network works without retraining. OFA's elastic space spans >10¹⁹ sub-networks with latency-guided selection.

## Functionality

- Slimmable: shared weights, width selected per forward; Switchable BN keeps separate per-width statistics so normalization behaves at every width.
- OFA: progressive shrinking trains large-first, then fine-tunes to include smaller depths/widths/kernels/resolutions; deployment picks sub-networks by measured latency.

## Used By

| Model | Role |
|-------|------|
| Slimmable Networks | Width-switchable training across all widths |
| Once-for-All Network | Elastic depth/width/kernel/resolution supernet |

## Features

- **One training run → deployment portfolio** — the core value for edge fleets.
- **Switchable normalization** — the subtle mechanism making shared weights width-agnostic.

## Evolution

- **Predecessor**: per-width model zoo training.
- **Successor**: NAS-derived per-device models (MobileNetV4), dynamic networks (once-for-all's descendant line).
