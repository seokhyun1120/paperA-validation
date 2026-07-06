"""Rev3 E2 — parquet에서 exact 검출률 표를 추출한다.

기존 E2 시뮬레이션 결과(`sim_results/E2.parquet`)를 재계산 없이 읽어
셀별 검출률, MC 표준오차, 검출된 run의 TTD 분위수를 appendix 표로 고정한다.
실행: ../../.venv/bin/python3 review/rev3_e2_rates.py
산출: data/rev3_e2_exact_rates.csv, review/appendix_rev3_e2.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
REVIEW = REPO / "review"
E2_PARQUET = REPO / "sim_results" / "E2.parquet"

EXPECTED_N = 10_000
RUN_LOG_ROUNDED = {
    ("gaussian", 60): {0.5: "0.10%", 1.0: "49.2%", 1.5: "99.9%"},
    ("gaussian", 120): {0.5: "0.20%", 1.0: "50.1%", 1.5: "99.9%"},
    ("gaussian", 240): {0.5: "0.21%", 1.0: "51.0%", 1.5: "99.8%"},
    ("t5", 120): {0.5: "0.15%", 1.0: "50.5%", 1.5: "99.9%"},
}


def tex_escape(value: object) -> str:
    """LaTeX 표 셀의 특수문자를 보호한다."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(value))


def rounded_rate(rate: float, mult: float) -> str:
    """RUN_LOG 표의 자리수와 같은 백분율 문자열을 만든다."""
    if mult == 0.5:
        return f"{rate:.2%}"
    return f"{rate:.1%}"


def load_e2() -> pd.DataFrame:
    """E2 parquet를 읽고 스키마 불변식을 확인한다."""
    df = pd.read_parquet(E2_PARQUET)
    expected_cols = {"noise", "D", "mult", "run_id", "delta", "crossed", "ttd"}
    missing = expected_cols - set(df.columns)
    assert not missing, f"E2 parquet 필수 컬럼 누락: {sorted(missing)}"
    assert len(df) == 120_000, f"E2 row 수 불일치: {len(df)}"
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """셀별 exact rate와 TTD 분위수를 계산한다."""
    key = ["noise", "D", "mult"]
    grouped = df.groupby(key, sort=True)
    summary = grouped.agg(
        det_rate=("crossed", "mean"),
        n=("crossed", "size"),
        delta=("delta", "first"),
        delta_nunique=("delta", "nunique"),
    ).reset_index()
    assert (summary["delta_nunique"] == 1).all(), "셀 내부 delta 값이 고정되어 있지 않음"
    summary = summary.drop(columns=["delta_nunique"])

    ttd = (
        df.loc[df["crossed"].astype(bool)]
        .groupby(key, sort=True)["ttd"]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .rename(columns={0.25: "ttd_q25", 0.5: "ttd_q50", 0.75: "ttd_q75"})
        .reset_index()
    )
    summary = summary.merge(ttd, on=key, how="left")
    summary["mc_se"] = np.sqrt(summary["det_rate"] * (1.0 - summary["det_rate"]) / summary["n"])
    summary = summary[
        ["noise", "D", "mult", "delta", "det_rate", "n", "mc_se", "ttd_q25", "ttd_q50", "ttd_q75"]
    ]
    noise_order = {"gaussian": 0, "t5": 1}
    summary["_noise_order"] = summary["noise"].map(noise_order)
    summary = (
        summary.sort_values(["_noise_order", "D", "mult"], kind="mergesort")
        .drop(columns=["_noise_order"])
        .reset_index(drop=True)
    )
    assert len(summary) == 12, f"E2 셀 수 불일치: {len(summary)}"
    assert (summary["n"] == EXPECTED_N).all(), "N=10,000/cell 불변식 실패"
    return summary


def write_csv(summary: pd.DataFrame) -> Path:
    """float 풀 정밀도 CSV를 저장한다."""
    out = DATA / "rev3_e2_exact_rates.csv"
    summary.to_csv(out, index=False, float_format="%.17g")
    return out


def fmt_ttd(value: float) -> str:
    """TTD 분위수의 결측과 소수 한 자리를 표시한다."""
    if pd.isna(value):
        return "--"
    return f"{value:.1f}"


def write_tex(summary: pd.DataFrame) -> Path:
    """appendix용 booktabs LaTeX 표를 저장한다."""
    lines = [
        r"% Rev3 E2 -- exact rates from sim_results/E2.parquet",
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{E2 exact rates from \texttt{sim\_results/E2.parquet}, "
        r"N=10,000/cell, seed [20260706, 2, noise, D, batch].}",
        r"\label{tab:rev3-e2-exact-rates}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Noise & $D$ & mult. & $\delta$ & det. rate (MC SE) & $n$ & TTD q25/q50/q75 \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        rate = f"{row['det_rate']:.4f} ({row['mc_se']:.4f})"
        qtxt = f"{fmt_ttd(row['ttd_q25'])}/{fmt_ttd(row['ttd_q50'])}/{fmt_ttd(row['ttd_q75'])}"
        lines.append(
            f"{tex_escape(row['noise'])} & {int(row['D'])} & {row['mult']:.1f} & "
            f"{row['delta']:.4f} & {rate} & {int(row['n']):,} & {qtxt} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = REVIEW / "appendix_rev3_e2.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def print_sanity(summary: pd.DataFrame) -> None:
    """RUN_LOG 반올림 표와 1.0x 검출률 sanity를 출력한다."""
    print("RUN_LOG rounded-rate sanity:")
    for (noise, d_months), expected in RUN_LOG_ROUNDED.items():
        cells = []
        for mult in (0.5, 1.0, 1.5):
            row = summary.loc[
                (summary["noise"] == noise) & (summary["D"] == d_months) & (summary["mult"] == mult)
            ].iloc[0]
            actual = rounded_rate(float(row["det_rate"]), mult)
            assert actual == expected[mult], (
                f"RUN_LOG 반올림 불일치: {noise} D={d_months} {mult}x "
                f"{actual} vs {expected[mult]}"
            )
            cells.append(f"{mult:.1f}x {actual}")
        at1 = summary.loc[
            (summary["noise"] == noise) & (summary["D"] == d_months) & (summary["mult"] == 1.0)
        ].iloc[0]
        near = abs(float(at1["det_rate"]) - 0.5) <= 0.10
        assert near, f"1.0x 검출률 50%±10%p 이탈: {noise} D={d_months}"
        print(
            f"  {noise:>8} D={d_months:>3}: "
            f"{' | '.join(cells)}; 1.0x det_rate={at1['det_rate']:.4f} ~ 0.50 PASS"
        )


def main() -> None:
    """E2 exact 산출물을 생성하고 검산 결과를 출력한다."""
    df = load_e2()
    summary = summarize(df)
    out_csv = write_csv(summary)
    out_tex = write_tex(summary)

    print(summary.to_string(index=False))
    print()
    print_sanity(summary)
    print(f"\nsaved -> {out_csv.relative_to(REPO)}")
    print(f"saved -> {out_tex.relative_to(REPO)}")


if __name__ == "__main__":
    main()
