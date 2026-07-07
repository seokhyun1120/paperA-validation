# paperA-validation

Companion repository for **Paper A** (day-1 detectability frontier). The
paper's code, data derivatives, and run history live under
[`paper_a_frontier/`](paper_a_frontier/); other directories are unrelated
research projects sharing this workspace repository.

Key entry points:

- `paper_a_frontier/RUN_LOG.md` — full chronological run log (stages,
  simulation experiments E1–E5, review-response rounds), with seeds and the
  commit hash each result was produced at.
- `paper_a_frontier/review/appendix_M8_audit.tex` — audit table mapping every
  reported number to its commit, seed, and data hash.
- `manifest.json` — SHA-256 (full 64-hex) and byte size of every file in
  `paper_a_frontier/data/`, including the two upstream inputs that are **not**
  committed for license reasons (`osap_LS_v200.csv.gz`, `SignalDoc.csv` —
  Chen–Zimmermann Open Source Asset Pricing release 2025.10 v2.00). Download
  them with `paper_a_frontier/stage1_download_osap.py` and verify integrity
  against the manifest.

## Data file naming convention

Files in `paper_a_frontier/data/` carry `rev3_` / `rev4_` / `rev5_` prefixes.
These mark the **review-response round (analysis revision) in which the file
was produced**; they are provenance labels, not version numbers of the data
itself. The files are deliberately **not renamed** after the fact, because
`manifest.json` and the audit appendix pin their SHA-256 hashes — renaming
would break the integrity chain. Where rounds overlap, canonical status is:

- `rev5_m1_full_adj.csv` — **canonical** envelope-consistent certification
  table (full run: all 199 matured factors adjusted with
  m_env_j = max(1.3, sqrt(pre-pub mean Y²))).
- `rev4_m1_envelope_adj.csv` — historical **partial** run kept for provenance
  (adjustment applied to the six certified factors only, per the round-4
  reviewer request); superseded by `rev5_m1_full_adj.csv` and retained because
  round-4 conclusions were derived from it.
- All other `rev*` files are the canonical (and only) source for their
  respective analyses; `stage*`/`m*`-prefixed files are pre-review pipeline
  outputs referenced by the run log.
