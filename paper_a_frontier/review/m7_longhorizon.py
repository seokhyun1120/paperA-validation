"""M7 — 장기 frontier 직접 MC (Table 2 확장).

n ∈ {480, 540, 600}, gaussian, N_SIM=10,000, SEED=0. frontier.py 무수정 —
모듈 rng를 SEED로 재설정하고 N_GRID가 [60,...,600]으로 확장됐을 때와 동일한
추첨 순서로 frontier_for_n을 호출. 기존 4점(60/120/240/360)이 import 시
계산된 frontier.catoni와 정확히 일치하는지 확인(불변 게이트), 신규 3점 보고.
실행: MPLBACKEND=Agg python3 review/m7_longhorizon.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim import eprocess as ep   # frontier import (savefig 가드 포함)

fr = ep.frontier
N_ALL = [60, 120, 240, 360, 480, 540, 600]

fr.rng = np.random.default_rng(fr.SEED)   # N_GRID 확장 시나리오의 추첨 순서 재현
print(f"{'n (mo)':>7} | {'oracle d*':>9} {'ann':>6} | {'Catoni d*':>9} {'ann':>6} | 비고")
print("-" * 62)
for i, n in enumerate(N_ALL):
    d_star, _ = fr.frontier_for_n(n)
    o = fr.oracle_frontier(n)
    note = ""
    if i < 4:
        assert d_star == fr.catoni[i], \
            f"STOP: 기존 점 변동 n={n}: {d_star} vs {fr.catoni[i]}"
        note = "기존 점 불변 확인"
    else:
        note = "신규"
    print(f"{n:>7} | {o:>9.4f} {o*np.sqrt(12):>6.2f} | "
          f"{d_star:>9.4f} {d_star*np.sqrt(12):>6.2f} | {note}")
print("\n기존 4점 == frontier.catoni (bit-exact) 확인 완료")
