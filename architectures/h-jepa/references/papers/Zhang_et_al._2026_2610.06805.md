# H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning

**Authors:** Chenlin Zhang, Mike Rabbat, Yann LeCun, Randall Balestriero
**arXiv:** 2610.06805 (2026)

## Abstract

Single-timescale JEPA world models suffer from error accumulation in long-range rollout and a single-latent-space bottleneck where fast and slow dynamics compete for representational capacity. We introduce H-JEPA, a hierarchical JEPA world model where each layer operates in its own latent space at a progressively coarser temporal stride. Planning is top-down: the top layer predicts a distant-future subgoal, which cascades down as subgoals for lower layers, culminating in actions. On Visual AntMaze, a three-layer hierarchy improves success rate from 18% to 73% while using less planning compute than the flat baseline. Upper layers discard fast unpredictable details and retain slow task-relevant state, making long-range prediction tractable. We validate on DROID real robot video, demonstrating applicability beyond simulation.
