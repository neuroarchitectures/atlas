# Path-Based Message Passing

## Design Philosophy

Network problems (routing, logistics) are about *paths*, not nodes. RouteNet represents each source-destination path as an ordered sequence of link states, propagates information along the sequence, and reads out per-path predictions — graph structure supplies the medium, paths supply the computation.

## Functionality

- Node + link hidden states on the physical graph.
- Per path: ordered link-state sequence → sequential/attention propagation → per-path readout (delay/loss via GLM-inspired probabilistic output).

## Used By

| Model | Role |
|-------|------|
| RouteNet | Computer-network performance modeling: per-path delay/loss from topology + routing |

## Features

- **Path as the prediction unit** — matches the actual quantity of interest.
- **Topology + configuration injection** — link states carry capacities/queues.

## Evolution

- **Predecessor**: node-level GNN regression averaged over paths.
- **Related**: path-sequence models in combinatorial optimization; graph2seq for generation over graphs.
