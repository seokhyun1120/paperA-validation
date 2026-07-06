"""M10 — standardized-score Sharpe sensitivity.

SR_std = sqrt(12) * mean(Y_t), Y_t = ret_t / max(v_{t-1}, v_min):
v_{t-1} = 직전 36개월 trailing sd (ddof=1, 최소 24개월), v_min = 자기
pre-publication trailing-sd의 하위 5% 분위수 (stage4와 동일 규칙). post-pub
윈도우는 stage2와 동일 (Year+1년 1월부터). raw SR(sharpe_ann)과의 상관·중앙값
차이 보고. 실행: python3 review/m10_std_sharpe.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data"
W, MIN_OBS, VMIN_Q = 36, 24, 0.05

ls = pd.read_csv(DATA / "osap_LS_v200.csv.gz", parse_dates=["date"])
doc = pd.read_csv(DATA / "SignalDoc.csv")
pred = doc[doc["Cat.Signal"] == "Predictor"][["Acronym", "Year"]]
m = ls.merge(pred, left_on="signalname", right_on="Acronym", how="inner")
m = m.dropna(subset=["ret"]).sort_values(["signalname", "date"])

rows = []
for sig, g in m.groupby("signalname"):
    year = int(g["Year"].iloc[0])
    r = g["ret"].to_numpy()
    dates = g["date"].to_numpy()
    v = (pd.Series(r).rolling(W, min_periods=MIN_OBS).std(ddof=1)
         .shift(1).to_numpy())
    pre = dates <= np.datetime64(f"{year}-12-31")
    post = dates >= np.datetime64(f"{year + 1}-01-01")
    v_pre = v[pre]
    if np.isnan(v_pre).all():
        continue
    v_min = np.nanquantile(v_pre, VMIN_Q)
    ok = post & ~np.isnan(v)
    if ok.sum() == 0:
        continue
    y = r[ok] / np.maximum(v[ok], v_min)
    rows.append({"signalname": sig, "n_std": int(ok.sum()),
                 "sr_std_ann": np.sqrt(12) * y.mean()})

std = pd.DataFrame(rows)
raw = pd.read_csv(DATA / "osap_postpub_sharpe.csv")[["signalname", "sharpe_ann", "n_j"]]
df = std.merge(raw, on="signalname")
df["diff"] = df["sr_std_ann"] - df["sharpe_ann"]
print(f"팩터 수: {len(df)} (기대 212)")
print(f"Pearson corr(raw, std)  = {df['sharpe_ann'].corr(df['sr_std_ann']):.4f}")
print(f"Spearman corr           = {df['sharpe_ann'].corr(df['sr_std_ann'], method='spearman'):.4f}")
print(f"median raw = {df['sharpe_ann'].median():.4f}, median std = {df['sr_std_ann'].median():.4f}")
print(f"diff (std - raw): median = {df['diff'].median():+.4f}, "
      f"Q1 = {df['diff'].quantile(0.25):+.4f}, Q3 = {df['diff'].quantile(0.75):+.4f}")
df.to_csv(DATA / "m10_std_sharpe.csv", index=False)
print("saved -> data/m10_std_sharpe.csv")
