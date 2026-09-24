# Customized Gate Control (CGC / PLE)

## Design Philosophy

Multi-task MoE suffers negative transfer: shared experts are pulled by conflicting tasks. PLE separates experts into *task-specific groups plus one shared group*; each task's gate attends only over its own experts and the shared group — and stacked extraction networks let the separation compose.

## Functionality

- Per task t: `y_t = Gate_t( [Experts_specific(t), Experts_shared] )`; gates see only their allowed expert set.
- Extraction networks stack CGC units (task A experts, shared experts, task B experts per level).

## Used By

| Model | Role |
|-------|------|
| PLE (Progressive Layered Extraction) | Multi-task CTR/CVR ranking; reference topology with task-specific + shared expert groups |

## Features

- **No forced sharing** — conflicting gradients never reach task-specific experts.
- **Composable depth** — stacked extraction layers refine separations progressively.

## Evolution

- **Predecessor**: Shared-Bottom, MMoE (multi-gate-moe) — MMoE gates over one common expert pool.
- **Successor**: PLEv2; task-specific routing ideas carried into LLM MoE via shared-expert-moe.
