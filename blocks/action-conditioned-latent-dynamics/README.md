# Action-Conditioned Latent Dynamics

## Design Philosophy

Dreamer-style world models learn their own representations; V-JEPA 2-AC shows a *frozen* perception model plus a small action-conditioned dynamics transformer suffices: predict next latents from (current latents, action, proprioception), roll out recurrently, score plans with a latent reward classifier — MPC in latent space with online replanning.

## Functionality

- Dynamics: small transformer `f(z_t, a_t, p_t) → z_{t+1}`; trained on <62h of robot video.
- Planning: rollout candidate action sequences in latent space, latent reward classifier scores, online replanning executes.

## Used By

| Model | Role |
|-------|------|
| V-JEPA 2-AC | Robot manipulation control over frozen V-JEPA 2 latents |

## Features

- **Frozen perception** — planning learns to *use* representations, not make them.
- **Tiny data budget** — hours of robot video, not thousands.

## Evolution

- **Predecessor**: recurrent-world-model (Dreamer) — learned representation + imagination.
- **Related**: masked-latent-prediction — the pretraining that produces the frozen latents.
