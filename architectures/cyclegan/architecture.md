# Architecture: CycleGAN

## Motivation

Paired image translation requires aligned image pairs, which are expensive to obtain. CycleGAN enables translation between domains using unpaired images by enforcing cycle consistency.

## Core Idea

Train two generators G: A->B and F: B->A, and two discriminators D_A and D_B. The cycle consistency loss ||F(G(x)) - x|| ensures that translating to B and back to A recovers the original. The adversarial loss ensures the translated images look like the target domain.

## Architecture

### Overview

![cyclegan architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image A | `input` | shape: [3, 256, 256] |
| 1 | Image B | `input` | shape: [3, 256, 256] |
| 2 | Generator G: A->B | `conv` | ResNet generator, 9 residual blocks |
| 2 | Generator F: B->A | `conv` | ResNet generator, 9 residual blocks |
| 3 | Discriminator D_B | `conv` | PatchGAN, 70x70 receptive field |
| 3 | Discriminator D_A | `conv` | PatchGAN, 70x70 receptive field |
| 4 | Cycle F(G(A)) | `conv` | cycle consistency |
| 4 | Cycle G(F(B)) | `conv` | cycle consistency |
| 5 | Outputs | `output` | adversarial + cycle + identity losses |

</details>

### Components

1. **Two generators** — G: A->B and F: B->A, each a ResNet with 9 residual blocks. 2. **Two discriminators** — D_A and D_B, each a PatchGAN (70x70 receptive field) that classifies local patches. 3. **Cycle consistency loss** — ||F(G(x_A)) - x_A||_1 + ||G(F(x_B)) - x_B||_1 ensures round-trip translation recovers the original. 4. **Adversarial loss** — LSGAN loss for stable training. 5. **Identity loss** — ||G(x_B) - x_B||_1 preserves color composition.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Train two generators G: A->B and F: B->A, and two discriminators D_A and D_B. The cycle consistency loss ||F(G(x)) - x|| ensures that translating to B and back to A recovers the original. The adversar
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Pix2Pix (paired). Successor: CycleGAN-VC (voice), MUNIT, DRIT.

## References

- Zhu et al. 2017
