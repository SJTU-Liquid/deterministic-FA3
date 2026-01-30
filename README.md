# deterministic-FA3 (DASH / ICLR 2026)

This repository is a fork of FlashAttention with additional commits implementing and experimenting with deterministic FlashAttention-3 backward scheduling strategies from the paper:

**DASH: Deterministic Attention Scheduling for High-throughput Reproducible LLM Training (ICLR 2026)**

## Status
This codebase is primarily for research / development. It currently does not provide a stable user-facing switch or turnkey script to enable/compare schedulers.

## What’s in this fork
- Deterministic FA3 backward scheduling experiments (Hopper / sm90 focus)
- Implementations of multiple schedulers explored in the paper (e.g., shift / symmetric variants), kept here for completeness and comparison

## Where to look
- Scheduler logic: `hopper/tile_m_scheduler.hpp` (and related headers)
- Backward mainloop integration: `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp` (sm90 bwd path)

## Install
Installation steps are the same as upstream FlashAttention. Please refer to:
https://github.com/Dao-AILab/flash-attention

## Citation
```bibtex
@misc{qiang2026dashdeterministicattentionscheduling,
      title={DASH: Deterministic Attention Scheduling for High-throughput Reproducible LLM Training},
      author={Xinwei Qiang and Hongmin Chen and Shixuan Sun and Jingwen Leng and Xin Liu and Minyi Guo},
      year={2026},
      eprint={2601.21824},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.21824},
}
```
