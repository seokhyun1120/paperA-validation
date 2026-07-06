"""Stage 7 — OSAP haircut sensitivity (stage6 확장).

사전 등록 grid a ∈ {0.005, 0.01} (연율)에 대해 월별 hurdle
  h = (1+a)^(1/12) - 1  (returns가 percent 단위이므로 ×100)
을 빼고 post-pub Sharpe·margin 재계산. A_j 관례·frontier는 stage2/6과 동일 —
hurdle이 상수라서 sd_m 불변이므로 기존 mean_m/sd_m/frontier_at_nj에서 직접 계산:
  sharpe_ann(a) = sqrt(12) * (mean_m - h_pct) / sd_m,
  margin(a) = sharpe_ann(a) - frontier_at_nj.
below-frontier %는 212개 전체 기준 (Stage 3의 98.6% 관례), margin 중앙값은
matured 199개 (censored 제외) 기준.
실행: python3 stage7_haircut_sensitivity.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
A_GRID = [0.0, 0.005, 0.01]     # 연율 hurdle (0 = primary)

df = pd.read_csv(DATA / "osap_postpub_sharpe.csv")
matured = ~df["censored"]

print(f"{'a (연율)':>9} | {'h (%/월)':>9} | {'below % (212)':>13} | "
      f"{'margin 중앙값 (matured 199)':>26}")
print("-" * 68)
rows = []
for a in A_GRID:
    h_pct = ((1 + a) ** (1 / 12) - 1) * 100
    sharpe_ann = np.sqrt(12) * (df["mean_m"] - h_pct) / df["sd_m"]
    margin = sharpe_ann - df["frontier_at_nj"]
    below = float((sharpe_ann <= df["frontier_at_nj"]).mean())
    med = float(margin[matured].median())
    rows.append({"a_ann": a, "h_pct_m": h_pct, "below_frontier_212": below,
                 "margin_median_matured": med})
    print(f"{a:>9.3f} | {h_pct:>9.5f} | {below:>12.1%} | {med:>26.3f}")

out = pd.DataFrame(rows)
out.to_csv(DATA / "stage7_haircut_sensitivity.csv", index=False)
print("\nsaved -> data/stage7_haircut_sensitivity.csv")
