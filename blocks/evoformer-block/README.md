# Evoformer Block

## Design Philosophy

A protein's MSA (evolutionary history across species) and its pair representation (residue-residue relationships) constrain each other. The Evoformer block lets these two tracks exchange information every layer, alternating attention *along the MSA* and *across the pair matrix* so co-evolution and contact structure are refined jointly.

## Functionality

Two interleaved tracks, each with residual + gated updates:

- **MSA track**: row-wise gated self-attention (across residues in each sequence), then column-wise attention (across sequences at a fixed residue) — the latter weighted by the pair representation bias.
- **Pair track**: triangular multiplicative updates (`outgoing` and `incoming` triangle attention) plus triangle self-attention, injecting the geometric triangle inequality over residue triples.

Between tracks, a "MSA transition" and "outer-product mean" pass information across.

## Used By

AlphaFold 2

## Features

- **Two-track co-design**: MSA and pair features mutually refine.
- **Column attention**: aggregates co-evolution across the alignment.
- **Triangle operations**: enforce consistency over triplets of residues.
- **Stacked 48x**: depth builds long-range structure in one forwarding pass.

## Evolution

Predecessor: MSA transformers (trRosetta, AlphaFold 1). Successor: single-sequence substitutes (ESMFold) that drop the MSA track. Sibling: IPA in the structure module downstream.
