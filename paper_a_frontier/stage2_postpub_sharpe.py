"""Stage 2 — predictor별 post-registration 월별 Sharpe 계산.

등록시점 관례 (변경 금지): A_j = 출판연도(Year)의 12월 말.
post-registration 윈도우 = Year+1년 1월부터의 월별 LS 수익률만 사용 (lookahead 금지).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"

ls = pd.read_csv(DATA / "osap_LS_v200.csv.gz", parse_dates=["date"])
doc = pd.read_csv(DATA / "SignalDoc.csv")

pred = doc[doc["Cat.Signal"] == "Predictor"][["Acronym", "Year"]]
print(f"SignalDoc predictors: {len(pred)}")

m = ls.merge(pred, left_on="signalname", right_on="Acronym", how="inner")
print(f"LS signals matched to predictors: {m['signalname'].nunique()}")

# post-registration: Year+1년 1월 이후만
m = m[m["date"] >= pd.to_datetime((m["Year"] + 1).astype(str) + "-01-01")]
m = m.dropna(subset=["ret"])

g = m.groupby("signalname")
res = pd.DataFrame({
    "pub_year": g["Year"].first(),
    "n_j": g["ret"].size(),
    "mean_m": g["ret"].mean(),
    "sd_m": g["ret"].std(ddof=1),
})
res["sharpe_m"] = res["mean_m"] / res["sd_m"]
res["sharpe_ann"] = np.sqrt(12) * res["sharpe_m"]
res = res.reset_index().rename(columns={"signalname": "signalname"})
res = res[["signalname", "pub_year", "n_j", "mean_m", "sd_m", "sharpe_m", "sharpe_ann"]]
res = res.sort_values("signalname")
res.to_csv(DATA / "osap_postpub_sharpe.csv", index=False)

dropped = pred[~pred["Acronym"].isin(res["signalname"])]

print(f"\npredictors with post-pub data: {len(res)}  (기대 212; 탈락 {len(dropped)})")
if len(dropped):
    print("dropped:", dropped.to_dict("records"))
q = res["sharpe_ann"].quantile([0.25, 0.5, 0.75])
print(f"sharpe_ann quartiles: Q1={q[0.25]:.3f}  median={q[0.5]:.3f}  Q3={q[0.75]:.3f}")
print(f"n_j: min={res['n_j'].min()}  median={res['n_j'].median():.0f}  max={res['n_j'].max()}")
print(f"mean_m (percent units): median={res['mean_m'].median():.3f}  "
      f"Q1={res['mean_m'].quantile(0.25):.3f}  Q3={res['mean_m'].quantile(0.75):.3f}")
print(f"sd_m median={res['sd_m'].median():.3f}")
print("saved -> data/osap_postpub_sharpe.csv")
