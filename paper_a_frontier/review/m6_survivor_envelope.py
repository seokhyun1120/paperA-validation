"""M6 — survivor envelope-consistent frontier 재계산.

생존 팩터의 자기 envelope(m_env)로 frontier를 직접 MC 재계산해 pass/fail 확정:
  AnnouncementReturn: n=336, m_env=1.46, 실현 SR_ann = 1.352
  AnalystRevision:    n=480, m_env=1.39, 실현 SR_ann = 1.055
방법: sim.eprocess.log_e_path(=frontier 커널)로 delta grid(0..1.2, 31점) crossing
prob 계산 후 50% 보간 — frontier_for_n과 동일 절차, m_env만 오버라이드.
N_SIM=10,000, SEED=0, gaussian, B_solo = 212/0.05 = 4240. 참조로 m_env=1.3도 병기.
실행: MPLBACKEND=Agg python3 review/m6_survivor_envelope.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim import eprocess as ep

fr = ep.frontier
N_SIM, SEED = 10_000, 0

CASES = [  # (팩터, n, m_env, 실현 SR_ann — data/osap_postpub_sharpe.csv)
    ("AnnouncementReturn", 336, 1.46, 1.352001572987291),
    ("AnalystRevision",    480, 1.39, 1.054904205097313),
]


def frontier_mc(n, m_env):
    rng = np.random.default_rng(SEED)
    eps = rng.standard_normal((N_SIM, n))
    target = 1 - fr.BETA
    probs = np.empty(len(fr.DELTA_GRID))
    for i, d in enumerate(fr.DELTA_GRID):
        logE = ep.log_e_path(d + eps, m_env=m_env)
        probs[i] = (logE.max(axis=1) >= fr.LOG_B).mean()
    if probs[-1] < target:
        return np.nan
    idx = int(np.argmax(probs >= target))
    if idx == 0:
        return float(fr.DELTA_GRID[0])
    d0, d1 = fr.DELTA_GRID[idx - 1], fr.DELTA_GRID[idx]
    p0, p1 = probs[idx - 1], probs[idx]
    return float(d0 + (target - p0) * (d1 - d0) / (p1 - p0))


print(f"{'팩터':>20} | {'n':>4} | {'m_env':>5} | {'frontier_ann':>12} | "
      f"{'실현 SR_ann':>11} | 판정")
print("-" * 78)
for name, n, m_env, sr in CASES:
    for m in (1.3, m_env):
        d = frontier_mc(n, m)
        ann = d * np.sqrt(12)
        verdict = "PASS (frontier 상회)" if sr > ann else "FAIL (frontier 하회)"
        ref = " (참조)" if m == 1.3 else ""
        print(f"{name:>20} | {n:>4} | {m:>5.2f} | {ann:>12.4f} | "
              f"{sr:>11.4f} | {verdict}{ref}")
