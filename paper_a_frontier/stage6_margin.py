"""Stage 6 — 팩터별 frontier margin 산출.

data/osap_postpub_sharpe.csv에 세 컬럼 추가:
  frontier_at_nj : n_j 위치의 Catoni frontier (연율). 60<=n_j<=360은 선형 보간,
                   n_j > 360은 sqrt(n) 스케일 외삽 (frontier(360)*sqrt(360/n_j)),
                   n_j < 60도 동일 방식으로 위쪽 외삽 (해당 팩터는 전부 censored).
  margin         : sharpe_ann - frontier_at_nj
  censored       : n_j < 120 (아직 10년 horizon이 안 찬 right-censored 케이스,
                   below/fail 집계 금지)

frontier 값 출처: frontier.py (커밋 기준 DELTA_GRID 상한 1.2, M_ENV=1.3,
N_SIM=10,000, SEED=0, NOISE=gaussian) 실행 출력 run_gaussian.log의 Catoni d*.
t5는 gaussian과 delta* 차이 <= 0.001이라 gaussian 값을 사용.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"

N_GRID = np.array([60, 120, 240, 360])
CATONI_M = np.array([0.826, 0.531, 0.370, 0.301])   # monthly delta* (gaussian)
CATONI_ANN = CATONI_M * np.sqrt(12)
CENSOR_N = 120

# 불변 확인: 기존 frontier 값 (연율, 소수 둘째 자리) 1.84 / 1.28 / 1.04
assert np.allclose(np.round(CATONI_ANN[1:], 2), [1.84, 1.28, 1.04]), CATONI_ANN

df = pd.read_csv(DATA / "osap_postpub_sharpe.csv")
n_j = df["n_j"].to_numpy(dtype=float)

frontier = np.interp(n_j, N_GRID, CATONI_ANN)
hi = n_j > N_GRID[-1]
frontier[hi] = CATONI_ANN[-1] * np.sqrt(N_GRID[-1] / n_j[hi])
lo = n_j < N_GRID[0]
frontier[lo] = CATONI_ANN[0] * np.sqrt(N_GRID[0] / n_j[lo])

df["frontier_at_nj"] = frontier
df["margin"] = df["sharpe_ann"] - df["frontier_at_nj"]
df["censored"] = df["n_j"] < CENSOR_N

df.to_csv(DATA / "osap_postpub_sharpe.csv", index=False)

cens = df[df["censored"]]
mat = df[~df["censored"]]
print(f"censored (n_j < {CENSOR_N}): {len(cens)}개 -> {sorted(cens['signalname'])}")
print(f"matured: {len(mat)}개")

qs = mat["margin"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
print("\nmatured margin 분위수 (연율 Sharpe 단위):")
for q, v in qs.items():
    print(f"  {q:>4.0%}: {v:+.3f}")

above = mat[mat["margin"] > 0]
print(f"\nfrontier 위 (margin > 0): {len(above)}개 -> {sorted(above['signalname'])}")
print("saved -> data/osap_postpub_sharpe.csv (frontier_at_nj, margin, censored 추가)")
