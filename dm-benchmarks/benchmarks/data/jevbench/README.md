# JevBench public decisions (vendored copy)

- **Source**: https://github.com/fstandhartinger/jevbench — `datasets/public/{original,easy,hard}.jsonl` at `HEAD`, fetched `2026-11-26`.
- **License**: MIT (see `LICENSE`); every item's `provenance.license` is `MIT`.
- **Counts**: original 72 (36 paraphrase pairs), easy 48, hard 111 = 231 items. Held-out/private tiers are not included.
- **Integrity**: `SHA256SUMS`.
- **Format**: each line has `state`, `question` (a `/v1/systemone` question: `type`, `instructions`, `criteria`), `labels`, `expected`, `family`, `group`, `tier` (added at load time), and `provenance` (hard items include `rationale`, `why_hard`, `surface_answer`).
- **Used by**: `pai-poc/decision_bench.py`.
- **Caveat**: results here are not comparable to the official JevBench board (different scorer version, single local run, M4 hardware, quantized weights).
