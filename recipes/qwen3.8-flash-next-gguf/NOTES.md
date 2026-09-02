# Qwen3.8-Flash-Next GGUF lane on 4x RTX 5090: field notes

Companion to README.md. The vLLM recipe (`../qwen3.8-flash-next/`) is the
fast path on the same hardware; this lane exists for people who want a GGUF.

## Traps and knobs (measured 2026-09-02)

- Pass `-md` with the explicit `MTP/...` path. The heads live in a
  subfolder that sidecar auto-discovery does not search; `--spec-type
  draft-mtp` alone finds nothing and you get spec-off speed with no error.
- The `shared-` head cannot be loaded on its own, so the automatic memory
  fit logs `borrow_shared_tensor: this model is a draft head without its
  own 'token_embd.weight'` and `failed to measure the memory of the extra
  model`, then continues. Consequence: the fit does not count the head, so
  pass `-c` yourself. 131,072 fits 4x 32 GB with the head; 262,144 is
  untested with it.
- Draft length is workload-shaped. Acceptance on prose: 0.56 at n-max 2,
  0.43-0.51 at 3, 0.31 at 5. At 5 the misses cost more than the hits and
  prose lands below spec-off (82 vs 88-89 tok/s), while a copy-heavy
  refactor jumps to 218. Keep 2 unless the traffic is edits of existing
  code, then 3.
- Spec-off decode on PR #28243 is 89-99 tok/s; the earlier mainline PR
  #27742 build measured 76-84 on the same GPUs. The branch carries kernel
  work beyond the MTP graph, so do not attribute the whole gain to MTP.
- Greedy 19*23 exact on every variant. Every number above is single
  stream; speculative gains shrink with temperature and concurrency.
- Compared with the vLLM lane on the same four cards: decode 203-211 vs
  130-156 on code, prefill 2,350-2,690 vs 1,456-1,712. Same model, same
  GPUs; the GGUF lane costs about a third of the speed.
