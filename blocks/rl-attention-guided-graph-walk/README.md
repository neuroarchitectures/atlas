# RL Attention-Guided Graph Walk

## Design Philosophy

Graph classification usually pools over all nodes — diluting signal in large graphs. Instead, *walk* the graph: an attention module selects which neighbor to move to next, trained as an RL policy (reward = classification accuracy) under a visit budget T; the RNN's hidden state doubles as the memory, and the final state classifies the graph.

## Functionality

- RNN hidden state = memory; attention over current node's neighborhood → policy π(next node | state).
- Policy gradient training under visit budget T nodes; final hidden state → classifier.

## Used By

| Model | Role |
|-------|------|
| Structural Attention GNN | Attention-guided walking for graph classification |

## Features

- **Learned node relevance** — the walk concentrates on informative subgraphs.
- **Budget = compute knob** — T bounds worst-case cost.

## Evolution

- **Predecessor**: fixed random-walk pooling; attention-free GNN readouts.
- **Related**: cooperative-action-network — learned *behavior* on graphs (communication vs traversal).
