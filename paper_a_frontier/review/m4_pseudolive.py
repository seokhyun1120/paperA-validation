"""M4 — fixed-H=120 pseudo-live 인증 분석.

n_j >= 120인 199개 팩터를 post-pub 첫 120개월에서 freeze — 첫 120개월 raw
Sharpe(연율)를 frontier(120) = sqrt(12)*0.5309384 = 1.8392 (frontier.catoni[1],
SEED=0, 본문 표기 1.84)와 비교, 통과(>= frontier) 수 보고.
실행: python3 review/m4_pseudolive.py  (repo 루트에서)
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data"
H = 120
FRONTIER_120_ANN = np.sqrt(12) * 0.5309384   # frontier.catoni[1] (SEED=0)

ls = pd.read_csv(DATA / "osap_LS_v200.csv.gz", parse_dates=["date"])
doc = pd.read_csv(DATA / "SignalDoc.csv")
pred = doc[doc["Cat.Signal"] == "Predictor"][["Acronym", "Year"]]
m = ls.merge(pred, left_on="signalname", right_on="Acronym", how="inner")
m = m[m["date"] >= pd.to_datetime((m["Year"] + 1).astype(str) + "-01-01")]
m = m.dropna(subset=["ret"]).sort_values(["signalname", "date"])

rows = []
for sig, g in m.groupby("signalname"):
    r = g["ret"].to_numpy()
    if len(r) < H:
        continue                       # n_j < 120 — censored, 제외
    r = r[:H]                          # 첫 120개월에서 freeze
    rows.append({"signalname": sig,
                 "sr120_ann": np.sqrt(12) * r.mean() / r.std(ddof=1)})
df = pd.DataFrame(rows)
print(f"대상 팩터 (n_j >= {H}): {len(df)}개 (기대 199)")
passed = df[df["sr120_ann"] >= FRONTIER_120_ANN].sort_values("sr120_ann",
                                                             ascending=False)
print(f"frontier(120) = {FRONTIER_120_ANN:.4f} (본문 1.84)")
print(f"통과 (첫 120개월 SR_ann >= frontier): {len(passed)} / {len(df)} "
      f"= {len(passed)/len(df):.1%}")
for _, r in passed.iterrows():
    print(f"  {r['signalname']}: {r['sr120_ann']:.3f}")
q = df["sr120_ann"].quantile([0.25, 0.5, 0.75])
print(f"첫 120개월 SR_ann 분위수: Q1={q[0.25]:.3f} med={q[0.5]:.3f} Q3={q[0.75]:.3f}")
df.to_csv(DATA / "m4_pseudolive_sr120.csv", index=False)
print("saved -> data/m4_pseudolive_sr120.csv")
