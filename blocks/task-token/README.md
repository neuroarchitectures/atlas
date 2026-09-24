# Task Token

## Design Philosophy

One model, many segmentation tasks (semantic/instance/panoptic) — but the model must know *which* task it is performing. A learned task token, injected at every conditioning point, switches the shared model's behavior without task-specific heads or weights.

## Functionality

- One learnable token per task, fed to the task-conditioned transformer decoder (added to queries / concatenated at each layer).
- Shared query decoder + shared pixel decoder; only the token differs per task.

## Used By

| Model | Role |
|-------|------|
| OneFormer | Single decoder serving semantic, instance, and panoptic segmentation; trained with a contrastive task-assignment objective |

## Features

- **True multi-task universality** — one set of weights, three task behaviors.
- **Token as control flow** — cheap, differentiable task switching.

## Evolution

- **Predecessor**: Mask2Former (one model per task).
- **Related**: adaptive-layernorm (DiT) — conditioning via modulation rather than a token; both are "condition the shared network" mechanisms.
