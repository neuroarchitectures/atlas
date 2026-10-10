# Multimodal RoPE (M-RoPE)

## Design Philosophy

Text RoPE encodes a single position axis. Images and video have several: a token belongs to a time index, a height, and a width. Splitting the rotary dimensions across these axes lets a multimodal decoder reason about spatial and temporal structure natively, while keeping the text case as a special case.

## Functionality

Partition the rotary dimension `d` into three sub-vectors `(d_t, d_h, d_w)`. For each token with coordinates `(t, h, w)`:

```
RoPE_mrope(z) : rotate the d_t part by position t,
                the d_h part by position h,
                the d_w part by position w.
```

For text-only tokens, `t = h = w =` the running token index, which reduces exactly to standard 1-D RoPE.

## Used By

Qwen2-VL

## Features

- **Factorised position**: separates temporal, height, and width axes.
- **Text-compatible**: degenerates to standard RoPE for text.
- **Resolution-agnostic**: absolute coordinate ranges scale with image size.
- **No extra parameters**: pure positional transformation.

## Evolution

Predecessor: 1-D RoPE (text only). Sibling: 2-D RoPE for images. Successor: multimodal position schemes in later VLMs.
