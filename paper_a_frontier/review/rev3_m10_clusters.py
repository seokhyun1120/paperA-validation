"""Rev3 M10 — Cat.Economic 클러스터 재집계.

Predictor 212개 전체의 post-publication margin을 Cat.Economic별로 묶고,
클러스터 margin을 소속 팩터 margin의 최댓값으로 정의한다.
실행: ../../.venv/bin/python3 review/rev3_m10_clusters.py
산출: data/rev3_m10_clusters.csv
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"


def load_inputs() -> pd.DataFrame:
    """raw margin과 Predictor Cat.Economic를 결합한다."""
    raw = pd.read_csv(DATA / "osap_postpub_sharpe.csv")
    doc = pd.read_csv(DATA / "SignalDoc.csv")
    pred = doc.loc[doc["Cat.Signal"] == "Predictor", ["Acronym", "Cat.Economic"]]
    assert pred["Acronym"].is_unique, "Predictor Acronym 중복"
    assert pred["Cat.Economic"].isna().sum() == 0, "Cat.Economic 결측 존재"
    merged = raw.merge(pred, left_on="signalname", right_on="Acronym", how="left")
    assert len(merged) == 212, f"팩터 수 불일치: {len(merged)}"
    assert merged["Cat.Economic"].isna().sum() == 0, "merge 후 Cat.Economic 결측 존재"
    return merged


def primary_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Cat.Economic별 max margin과 best factor를 계산한다."""
    idx = df.groupby("Cat.Economic")["margin"].idxmax()
    best = (
        df.loc[idx, ["Cat.Economic", "signalname", "margin"]]
        .rename(
            columns={
                "Cat.Economic": "cluster",
                "signalname": "best_factor",
                "margin": "max_margin",
            }
        )
        .reset_index(drop=True)
    )
    counts = (
        df.groupby("Cat.Economic")
        .size()
        .rename("n_members")
        .reset_index()
        .rename(columns={"Cat.Economic": "cluster"})
    )
    out = counts.merge(best, on="cluster", how="inner")
    out["below"] = out["max_margin"] < 0
    out = out[["cluster", "n_members", "best_factor", "max_margin", "below"]]
    out = out.sort_values("cluster", kind="mergesort").reset_index(drop=True)
    assert len(out) == 35, f"primary 클러스터 수 불일치: {len(out)}"
    return out


def other_singleton_sensitivity(df: pd.DataFrame, primary: pd.DataFrame) -> tuple[int, int]:
    """other 클러스터만 팩터별 싱글턴으로 분해한 민감도 below 수를 계산한다."""
    other = df.loc[df["Cat.Economic"] == "other"].copy()
    assert len(other) == 27, f"other 팩터 수 불일치: {len(other)}"
    non_other = primary.loc[primary["cluster"] != "other", ["cluster", "max_margin"]].copy()
    assert len(non_other) == 34, f"non-other 클러스터 수 불일치: {len(non_other)}"
    singleton = pd.DataFrame(
        {
            "cluster": other["signalname"].to_numpy(),
            "max_margin": other["margin"].to_numpy(),
        }
    )
    sens = pd.concat([non_other, singleton], ignore_index=True)
    assert len(sens) == 61, f"민감도 클러스터 수 불일치: {len(sens)}"
    return int((sens["max_margin"] < 0).sum()), int(len(sens))


def main() -> None:
    """M10 클러스터 산출물을 생성한다."""
    df = load_inputs()
    primary = primary_clusters(df)
    out_csv = DATA / "rev3_m10_clusters.csv"
    primary.to_csv(out_csv, index=False)

    below = int(primary["below"].sum())
    total = int(len(primary))
    sens_below, sens_total = other_singleton_sensitivity(df, primary)
    above = primary.loc[~primary["below"]].sort_values("max_margin", ascending=False)

    print("클러스터 margin은 censored 포함 212개 전 팩터 margin의 최댓값으로 계산.")
    print(f"primary Cat.Economic 클러스터: {total}개")
    print(f"primary below 클러스터: {below}/{total} = {below / total:.1%}")
    print("primary above 클러스터:")
    for _, row in above.iterrows():
        print(f"  {row['cluster']}: {row['best_factor']} (max margin {row['max_margin']:+.3f})")
    print(f"other 싱글턴 민감도: below {sens_below}/{sens_total} = {sens_below / sens_total:.1%}")
    print("헤드라인 209/212 = 98.6% 대비, 클러스터 집계는 한 클러스터 안의 최선 margin을 기준으로 한다.")
    print(f"saved -> {out_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
