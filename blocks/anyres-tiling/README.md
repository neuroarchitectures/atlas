# AnyRes Tiling (LLaVA-NeXT)

## Design Philosophy

A fixed-resolution vision encoder forces every image into one aspect ratio and token count, wasting compute on simple images and losing detail on high-resolution ones. AnyRes keeps the native aspect ratio by splitting an image into tiling grids at several resolutions and encoding each tile separately.

## Functionality

Given an image and a set of candidate grids (e.g. `(2x2), (1x4), ...`):

1. Pick the grid whose aspect ratio best matches the image.
2. Split the image into that grid of tiles plus one downscaled global thumbnail.
3. Encode every tile (and the thumbnail) with the same ViT.
4. Concatenate all tile tokens as the visual input to the LLM.

## Used By

LLaVA-NeXT

## Features

- **Aspect-ratio preserving**: no forced resizing or padding.
- **Resolution-adaptive**: token count grows with input size.
- **Detail retention**: tiles keep high-resolution local information.
- **Global context**: the thumbnail supplies the overall view.

## Evolution

Predecessor: fixed-resolution ViT projection (LLaVA-1.5). Sibling: dynamic-resolution patching (Qwen2-VL). Successor: native-resolution / variable-resolution vision encoders.
