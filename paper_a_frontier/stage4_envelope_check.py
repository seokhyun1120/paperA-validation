"""Stage 4 — envelope(M_ENV) 진단.

각 팩터의 pre-publication 데이터만 사용 (A_j = Year 12월 말 이전, lookahead 금지):
  - v_{t-1} = 직전 36개월 trailing sd (ddof=1, 최소 24개월 미만 시점은 skip)
  - v_min   = 자기 pre-publication trailing-sd 시계열의 하위 5% 분위수 (팩터별 floor)
  - Y_t     = ret_t / max(v_{t-1}, v_min),  팩터별 block mean(Y^2) 계산
coverage 기준 (사전 확정): block mean(Y^2) <= m_env^2 인 팩터 비율 >= 90%.
m_env = min{ m in {1.3, 1.5, 1.7} : 기준 충족 }.  1.7도 미달이면 채택하지 않고 STOP.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"

WINDOW = 36
MIN_OBS = 24
VMIN_Q = 0.05
M_CANDIDATES = [1.3, 1.5, 1.7]
COVERAGE_TARGET = 0.90

ls = pd.read_csv(DATA / "osap_LS_v200.csv.gz", parse_dates=["date"])
doc = pd.read_csv(DATA / "SignalDoc.csv")
pred = doc[doc["Cat.Signal"] == "Predictor"][["Acronym", "Year"]]

m = ls.merge(pred, left_on="signalname", right_on="Acronym", how="inner")
# pre-publication: 등록시점 A_j = Year 12월 말 이전의 데이터만
m = m[m["date"] <= pd.to_datetime(m["Year"].astype(str) + "-12-31")]
m = m.dropna(subset=["ret"]).sort_values(["signalname", "date"])

rows = []
for sig, g in m.groupby("signalname"):
    r = g["ret"].to_numpy()
    v = (
        pd.Series(r)
        .rolling(WINDOW, min_periods=MIN_OBS)
        .std(ddof=1)
        .shift(1)  # v_{t-1}: t 시점 수익률에는 직전 정보만 사용
        .to_numpy()
    )
    valid = ~np.isnan(v)
    if valid.sum() == 0:
        rows.append({"signalname": sig, "n_pre": len(r), "n_Y": 0,
                     "v_min": np.nan, "meanY2": np.nan})
        continue
    v_min = np.nanquantile(v, VMIN_Q)
    y = r[valid] / np.maximum(v[valid], v_min)
    rows.append({"signalname": sig, "n_pre": len(r), "n_Y": int(valid.sum()),
                 "v_min": v_min, "meanY2": float(np.mean(y**2))})

res = pd.DataFrame(rows)
usable = res.dropna(subset=["meanY2"])
print(f"팩터 수: {len(res)}  (meanY2 산출 가능: {len(usable)}, "
      f"pre-pub 표본 부족 제외: {len(res) - len(usable)})")

qs = usable["meanY2"].quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
print("\nblock mean(Y^2) 분위수:")
for q, val in qs.items():
    print(f"  {q:>5.0%}: {val:.4f}")

print(f"\ncoverage (기준: >= {COVERAGE_TARGET:.0%}):")
adopted = None
for mc in M_CANDIDATES:
    cov = (usable["meanY2"] <= mc**2).mean()
    mark = ""
    if adopted is None and cov >= COVERAGE_TARGET:
        adopted = mc
        mark = "  <- 채택"
    print(f"  m_env = {mc}:  m^2 = {mc**2:.2f},  coverage = {cov:.1%} "
          f"({(usable['meanY2'] <= mc**2).sum()}/{len(usable)}){mark}")

if adopted is None:
    print("\n*** STOP: 1.7도 coverage 기준 미달 — 채택하지 않고 보고 ***")
else:
    print(f"\n채택 m_env = {adopted}" + ("  (기존 1.3과 동일 — frontier 재실행 불필요)"
                                        if adopted == 1.3 else
                                        "  (기존 1.3에서 변경 — frontier 재실행 필요, 실행은 대기)"))

res.to_csv(DATA / "stage4_envelope_check.csv", index=False)
print("saved -> data/stage4_envelope_check.csv")
