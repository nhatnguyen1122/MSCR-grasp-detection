"""MSCR: inference-time decoding, candidate generation and the re-ranking scorer.

  mscr.common     dataset / checkpoint loading, network forward -> post-processed maps
  mscr.decode     mask gating, top-1 decoding, candidate strategies
  mscr.scorer     rotated crops, frozen-CLIP features, scorer MLP and losses
"""
