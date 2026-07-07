# Commit hash map (history filter, 2026-07-07)

This repository's git history was rewritten once, on 2026-07-07, with
`git filter-repo`, for a single reason: the original workspace repository
contained **unrelated, unpublished research projects** alongside Paper A, and
publishing them was never intended. The filter kept only
`paper_a_frontier/`, `README.md`, `manifest.json`, `AGENTS.md`, and
`.gitignore`.

What the rewrite does and does not change:

- **File contents are untouched.** Every file blob (git object) under
  `paper_a_frontier/` is byte-identical before and after the filter, and the
  data-file SHA-256 hashes in `manifest.json` and the audit appendix are
  unaffected. Only commit IDs change, because a commit ID hashes the whole
  repository tree.
- **Commit messages, authorship, dates, and order are preserved.**
- The audit table (`paper_a_frontier/review/appendix_M8_audit.tex`) and the
  run log (`paper_a_frontier/RUN_LOG.md`) now cite the post-filter hashes.
  Manuscript or referee-report copies that predate 2026-07-07 cite the
  pre-filter hashes; the table below maps them.
- The pre-filter repository is archived privately in full and can be made
  available to the editor on request.

| pre-filter (old) | post-filter (new) | anchor |
|---|---|---|
| `85a7806` | `b811573` | frontier.py original commit |
| `2d7834c` | `9d35e1b` | OSAP v2.00 real-data frontier run (Table 2, 4 pts) |
| `01b5ded` | `1126183` | Stage 4–6 (envelope, null gate, grid ext., margin) |
| `9d091ea` | `a8a4129` | sim engine + regression test (E1–E5 first pass) |
| `b2656c1` | `a6efa22` | E1/E5 baseline-reveal rerun |
| `5b9cbac` | `f059eb9` | R1 mechanism check, R2 DSR comparator |
| `d7b2e7c` | `b59acb8` | R1b e-BH tautness |
| `d5d468f` | `d809d82` | review round 2 (M4–M11, R3) |
| `f2719eb` | `ad0ba0d` | review round 4 (envelope-consistent, t5 long, A_j sets) |
| `f1d7c5b` | `528b29d` | review round 5 (R8/Min7 gates, DSR sensitivity) |
| `4be0a19` | `d9ac163` | AGENTS.md; execution base of round-4 runs |

Verification: for any row, `git show <new>:paper_a_frontier/<file>` in this
repository reproduces the exact file content the log attributes to the old
hash (blob IDs are identical).
