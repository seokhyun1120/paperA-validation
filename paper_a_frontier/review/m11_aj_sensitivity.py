"""M11(7번) — A_j 등록시점 관례 sensitivity.

세 관례: (a) 출판연도 1월초 (post >= Year-01-01), (b) 7월초 (post >= Year-07-01),
(c) 12월말 (기존 primary, post >= (Year+1)-01-01).
각각 n_j·sharpe_ann 재계산 후 stage6과 동일한 frontier 보간/외삽 규칙
(Catoni 월간 delta* [0.826, 0.531, 0.370, 0.301] — stage6 표기와 동일)으로
below-frontier % (212)와 matured(n_j>=120) median margin 보고.
실행: python3 review/m11_aj_sensitivity.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data"
N_GRID = np.array([60, 120, 240, 360])
CATONI_ANN = np.array([0.826, 0.531, 0.370, 0.301]) * np.sqrt(12)
CENSOR_N = 120

ls = pd.read_csv(DATA / "osap_LS_v200.csv.gz", parse_dates=["date"])
doc = pd.read_csv(DATA / "SignalDoc.csv")
pred = doc[doc["Cat.Signal"] == "Predictor"][["Acronym", "Year"]]
m = ls.merge(pred, left_on="signalname", right_on="Acronym", how="inner")
m = m.dropna(subset=["ret"])

CONVENTIONS = {
    "Year 1월초": lambda y: pd.to_datetime(y.astype(str) + "-01-01"),
    "Year 7월초": lambda y: pd.to_datetime(y.astype(str) + "-07-01"),
    "Year 12월말 (primary)": lambda y: pd.to_datetime((y + 1).astype(str) + "-01-01"),
}


def frontier_at(n_j):
    f = np.interp(n_j, N_GRID, CATONI_ANN)
    hi = n_j > N_GRID[-1]
    f[hi] = CATONI_ANN[-1] * np.sqrt(N_GRID[-1] / n_j[hi])
    lo = n_j < N_GRID[0]
    f[lo] = CATONI_ANN[0] * np.sqrt(N_GRID[0] / n_j[lo])
    return f


print(f"{'관례':>22} | {'below % (212)':>13} | {'matured 수':>9} | {'median margin':>13}")
print("-" * 70)
for name, cutoff in CONVENTIONS.items():
    sub = m[m["date"] >= cutoff(m["Year"])]
    g = sub.groupby("signalname")["ret"]
    res = pd.DataFrame({"n_j": g.size(), "mean_m": g.mean(),
                        "sd_m": g.std(ddof=1)})
    res["sharpe_ann"] = np.sqrt(12) * res["mean_m"] / res["sd_m"]
    f = frontier_at(res["n_j"].to_numpy(dtype=float))
    below = float((res["sharpe_ann"] <= f).mean())
    matured = res["n_j"] >= CENSOR_N
    med_margin = float((res["sharpe_ann"] - f)[matured].median())
    print(f"{name:>22} | {below:>12.1%} | {int(matured.sum()):>9} | {med_margin:>+13.3f}")
