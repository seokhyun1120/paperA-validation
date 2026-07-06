"""Rev3 M4 — raw vs standardized 분류 안정성 표.

raw post-publication Sharpe와 stage4 표준화 점수
SR_std = sqrt(12) * mean(Y_t)의 frontier 분류를 요약한다.
실행: ../../.venv/bin/python3 review/rev3_m4_raw_std.py
산출: data/rev3_m4_raw_std.csv, review/appendix_rev3_m4.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
REVIEW = REPO / "review"

SURVIVORS = ["AnalystRevision", "AnnouncementReturn", "DivYieldST"]


def tex_escape(value: object) -> str:
    """LaTeX 표 셀에 들어갈 문자열의 특수문자를 보호한다."""
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


def pct(num: int, den: int) -> str:
    """분자/분모와 백분율을 함께 표시한다."""
    return f"{num}/{den} ({num / den:.1%})"


def signed(value: float) -> str:
    """부호가 보이는 소수점 세 자리 문자열."""
    return f"{value:+.3f}"


def status(margin: float) -> str:
    """frontier 대비 above/below 상태를 margin 부호로 판정한다."""
    label = "above" if margin >= 0 else "below"
    return f"{label} ({signed(margin)})"


def top3_text(df: pd.DataFrame, value_col: str) -> str:
    """상위 3개 팩터와 값을 한 셀에 요약한다."""
    parts = []
    for _, row in df.iterrows():
        parts.append(f"{row['signalname']} ({row[value_col]:.3f})")
    return "; ".join(parts)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """필요 CSV 네 개를 읽고 기본 불변식을 확인한다."""
    raw = pd.read_csv(DATA / "osap_postpub_sharpe.csv")
    std = pd.read_csv(DATA / "m10_std_sharpe.csv")
    m4 = pd.read_csv(DATA / "m4_pseudolive_sr120.csv")
    m1 = pd.read_csv(DATA / "rev3_m1_ebh_cert.csv")
    assert len(raw) == 212, f"raw 팩터 수 불일치: {len(raw)}"
    assert (std["n_std"].to_numpy() == std["n_j"].to_numpy()).all(), "m10 n_std != n_j"
    return raw, std, m4, m1


def build_summary() -> tuple[pd.DataFrame, list[tuple[str, str, str]], dict[str, object]]:
    """기계가독 CSV와 LaTeX 2열 표에 들어갈 값을 계산한다."""
    raw, std, m4, m1 = load_inputs()
    merged = raw.merge(
        std[["signalname", "n_std", "sr_std_ann", "sharpe_ann", "n_j", "diff"]],
        on="signalname",
        how="inner",
        suffixes=("_raw", "_std_input"),
    )
    assert len(merged) == len(raw), "raw/std merge 후 팩터 수 불일치"
    merged["margin_std"] = merged["sr_std_ann"] - merged["frontier_at_nj"]

    total = len(raw)
    raw_below = int((raw["margin"] < 0).sum())
    assert raw_below == 209, f"raw below 수 기대값 불일치: {raw_below}"
    std_below = int((merged["margin_std"] < 0).sum())

    raw_top3 = m4.sort_values("sr120_ann", ascending=False).head(3)
    std_top3 = m1.sort_values("sr120_std_ann", ascending=False).head(3)
    uncensored = ~merged["censored"].astype(bool)
    assert int(uncensored.sum()) == 199, f"uncensored matured 수 불일치: {uncensored.sum()}"
    raw_med = float(merged.loc[uncensored, "margin"].median())
    std_med = float(merged.loc[uncensored, "margin_std"].median())

    csv_rows: list[dict[str, object]] = [
        {
            "metric": "below_frontier_count",
            "row": "all_212",
            "raw_factor": "",
            "raw_value": raw_below,
            "raw_denominator": total,
            "raw_margin": np.nan,
            "raw_status": pct(raw_below, total),
            "standardized_factor": "",
            "standardized_value": std_below,
            "standardized_denominator": total,
            "standardized_margin": np.nan,
            "standardized_status": pct(std_below, total),
        }
    ]
    for rank, ((_, r_raw), (_, r_std)) in enumerate(
        zip(raw_top3.iterrows(), std_top3.iterrows()), start=1
    ):
        csv_rows.append(
            {
                "metric": "fixed_h120_top3",
                "row": f"rank_{rank}",
                "raw_factor": r_raw["signalname"],
                "raw_value": float(r_raw["sr120_ann"]),
                "raw_denominator": np.nan,
                "raw_margin": np.nan,
                "raw_status": "",
                "standardized_factor": r_std["signalname"],
                "standardized_value": float(r_std["sr120_std_ann"]),
                "standardized_denominator": np.nan,
                "standardized_margin": np.nan,
                "standardized_status": "",
            }
        )
    for factor in SURVIVORS:
        row = merged.loc[merged["signalname"] == factor]
        assert len(row) == 1, f"생존 팩터 누락: {factor}"
        r = row.iloc[0]
        csv_rows.append(
            {
                "metric": "survivor_status",
                "row": factor,
                "raw_factor": factor,
                "raw_value": np.nan,
                "raw_denominator": np.nan,
                "raw_margin": float(r["margin"]),
                "raw_status": status(float(r["margin"])),
                "standardized_factor": factor,
                "standardized_value": np.nan,
                "standardized_denominator": np.nan,
                "standardized_margin": float(r["margin_std"]),
                "standardized_status": status(float(r["margin_std"])),
            }
        )
    csv_rows.append(
        {
            "metric": "median_margin_uncensored",
            "row": "censored_false_199",
            "raw_factor": "",
            "raw_value": raw_med,
            "raw_denominator": int(uncensored.sum()),
            "raw_margin": raw_med,
            "raw_status": signed(raw_med),
            "standardized_factor": "",
            "standardized_value": std_med,
            "standardized_denominator": int(uncensored.sum()),
            "standardized_margin": std_med,
            "standardized_status": signed(std_med),
        }
    )
    machine = pd.DataFrame(csv_rows)

    survivor_raw = "; ".join(
        f"{factor}: {machine.loc[(machine['metric'] == 'survivor_status') & (machine['row'] == factor), 'raw_status'].iloc[0]}"
        for factor in SURVIVORS
    )
    survivor_std = "; ".join(
        f"{factor}: {machine.loc[(machine['metric'] == 'survivor_status') & (machine['row'] == factor), 'standardized_status'].iloc[0]}"
        for factor in SURVIVORS
    )
    table_rows = [
        ("Below frontier count", pct(raw_below, total), pct(std_below, total)),
        ("Fixed-$H=120$ top 3", top3_text(raw_top3, "sr120_ann"), top3_text(std_top3, "sr120_std_ann")),
        ("Three surviving factors", survivor_raw, survivor_std),
        ("Median margin, censored excluded", signed(raw_med), signed(std_med)),
    ]
    stats: dict[str, object] = {
        "raw_below": raw_below,
        "std_below": std_below,
        "total": total,
        "raw_top3": top3_text(raw_top3, "sr120_ann"),
        "std_top3": top3_text(std_top3, "sr120_std_ann"),
        "survivor_raw": survivor_raw,
        "survivor_std": survivor_std,
        "raw_med": raw_med,
        "std_med": std_med,
    }
    return machine, table_rows, stats


def write_appendix(rows: list[tuple[str, str, str]]) -> None:
    """raw/std 안정성 비교 booktabs 표를 저장한다."""
    lines = [
        r"% Rev3 M4 -- raw vs standardized classification stability",
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{Raw vs. standardized classification stability. "
        r"The standardized column uses $SR_{\rm std}=\sqrt{12}\,\mathrm{mean}(Y)$, "
        r"with the stage4 predictable volatility rule and floor.}",
        r"\label{tab:rev3-m4-raw-std}",
        r"\begin{tabular}{p{0.26\linewidth}p{0.33\linewidth}p{0.33\linewidth}}",
        r"\toprule",
        r"Comparison & raw & standardized \\",
        r"\midrule",
    ]
    for label, raw_cell, std_cell in rows:
        lines.append(f"{label} & {tex_escape(raw_cell)} & {tex_escape(std_cell)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = REVIEW / "appendix_rev3_m4.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"saved -> {out.relative_to(REPO)}")


def main() -> None:
    """M4 raw/std 비교 산출물을 생성한다."""
    machine, table_rows, stats = build_summary()
    out_csv = DATA / "rev3_m4_raw_std.csv"
    machine.to_csv(out_csv, index=False)
    write_appendix(table_rows)
    print(f"saved -> {out_csv.relative_to(REPO)}")
    print("raw vs standardized 요약:")
    print(f"  below-frontier: raw {stats['raw_below']}/{stats['total']}, "
          f"std {stats['std_below']}/{stats['total']}")
    print(f"  fixed-H=120 top3 raw: {stats['raw_top3']}")
    print(f"  fixed-H=120 top3 std: {stats['std_top3']}")
    print(f"  survivor raw: {stats['survivor_raw']}")
    print(f"  survivor std: {stats['survivor_std']}")
    print(f"  median margin uncensored: raw {stats['raw_med']:+.3f}, "
          f"std {stats['std_med']:+.3f}")


if __name__ == "__main__":
    main()
