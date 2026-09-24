# Channel-Split Shuffle Block

## Design Philosophy

ShuffleNet V1's group convolutions + shuffles reduced FLOPs but *increased* memory access cost — the true latency driver on mobile. The V2 block replaces group conv with a simple channel split: one half is convolved, the other passes through, and the halves are concatenated (not added) then channel-shuffled.

## Functionality

- Split channels equally: branch A → 1×1 conv → 3×3 depthwise conv → 1×1 conv; branch B → identity.
- Concat A ∥ B (keeps information flow explicit), then channel shuffle to mix branches.
- Downsample variant: no split, both branches convolved, concat after spatial stride.

## Used By

| Model | Role |
|-------|------|
| ShuffleNet V2 | Basic and downsample units in all stages |

## Features

- **Half the channels computed** per block at equal width.
- **No element-wise add** — concat preserves both branch identities; shuffle restores cross-branch mixing.

## Evolution

- **Predecessor**: ShuffleNet V1 (group conv + shuffle), depthwise-separable-conv.
- **Successor**: MobileNet V3/V4 blocks; split-then-merge recurs in CSP-block (YOLO family) and StarNet.
