"""R5b — DSR comparator sensitivity (Rev5 M4).

E5와 동일한 CRN 세계를 재생성한 뒤, terminal batch DSR comparator의
trial count 정의와 skew/kurtosis 처리 방식에 대한 민감도를 계산한다.

실행: MPLBACKEND=Agg python3 -m sim.run_R5b
"""
import time

import numpy as np
import pandas as pd
from scipy import stats

from . import common
from .run_E5 import ALPHA, J_BUDGET, LAM_ARR, N_RUN, T
from .run_R5 import EULER_GAMMA, regenerate_e5_world

TRIAL_DEFS = ("n_reg", "J=212", "n_arrived")
MOMENT_DEFS = ("empirical", "normal")


def trial_count(world_e5: dict[str, np.ndarray | int], trials_def: str) -> int:
    """SR0에 들어갈 trial count 정의를 선택한다."""
    if trials_def == "n_reg":
        return int(world_e5["n_reg"])
    if trials_def == "J=212":
        return J_BUDGET
    if trials_def == "n_arrived":
        return int(world_e5["n_arrived"])
    raise ValueError(f"알 수 없는 trials_def: {trials_def}")


def dsr_discoveries(
    Ymat: np.ndarray,
    lens: np.ndarray,
    n_reg: int,
    trials_n: int,
    moments: str,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, int, int, int]:
    """BLdP(2014) DSR batch 발견 플래그를 민감도 인자로 일반화한다."""
    if moments not in MOMENT_DEFS:
        raise ValueError(f"알 수 없는 moments 정의: {moments}")

    disc = np.zeros(n_reg, bool)
    family = lens >= 2
    excluded_short = int((~family).sum())
    idx = np.nonzero(family)[0]
    trials_n_raw = int(trials_n)
    guard_count = int(trials_n_raw < 2)
    trials_n_eff = max(trials_n_raw, 2)
    if len(idx) < 2:
        return disc, excluded_short, int(len(idx)), guard_count

    sr = np.empty(len(idx))
    if moments == "empirical":
        g3 = np.empty(len(idx))
        g4 = np.empty(len(idx))
    else:
        g3 = g4 = None

    for out_i, j in enumerate(idx):
        y = Ymat[j, : lens[j]]
        n_i = len(y)
        m1 = float(y.mean())
        centered = y - m1
        m2 = float((centered ** 2).mean())
        assert m2 > 0.0, "DSR sd 정의 불가 경로 발생"
        sr[out_i] = m1 / np.sqrt(m2 * n_i / (n_i - 1))
        if moments == "empirical":
            m3 = float((centered ** 3).mean())
            m4 = float((centered ** 4).mean())
            g3[out_i] = m3 / (m2 ** 1.5)
            g4[out_i] = m4 / (m2 ** 2)

    V = float(np.var(sr, ddof=1))
    sr0 = np.sqrt(V) * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / trials_n_eff)
        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (trials_n_eff * np.e))
    )
    if moments == "empirical":
        denom = np.clip(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12, None)
    else:
        denom = 1 + 0.5 * sr ** 2
    z = (sr - sr0) * np.sqrt(lens[idx] - 1) / np.sqrt(denom)
    disc[idx] = stats.norm.cdf(z) > 1 - alpha
    return disc, excluded_short, int(len(idx)), guard_count


def format_pct3(x: float) -> str:
    """민감도 표용 고정 소수점 3자리 퍼센트 포맷."""
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.3f}%"


def format_pct3_tex(x: float) -> str:
    """LaTeX 표용 고정 소수점 3자리 퍼센트 포맷."""
    return format_pct3(x).replace("%", "\\%")


def update_cell(
    cells: dict[tuple[float, str, str], dict[str, float]],
    lam: float,
    trials_def: str,
    moments: str,
    n_alt: int,
    disc: np.ndarray,
    is_alt: np.ndarray,
    guard_count: int,
) -> None:
    """런 단위 DSR 결과를 셀 집계에 누적한다."""
    key = (lam, trials_def, moments)
    cell = cells.setdefault(
        key,
        {
            "n_alt": 0,
            "alt_disc": 0,
            "n_disc": 0,
            "n_false": 0,
            "fdp_sum": 0.0,
            "n_run": 0,
            "guard_count": 0,
        },
    )
    false_disc = disc & ~is_alt
    n_disc = int(disc.sum())
    n_false = int(false_disc.sum())
    cell["n_alt"] += n_alt
    cell["alt_disc"] += int((disc & is_alt).sum())
    cell["n_disc"] += n_disc
    cell["n_false"] += n_false
    cell["fdp_sum"] += n_false / max(n_disc, 1)
    cell["n_run"] += 1
    cell["guard_count"] += guard_count


def assert_r5_gate(cells: dict[tuple[float, str, str], dict[str, float]]) -> None:
    """baseline 조합의 DSR power가 저장된 R5 집계와 정확히 같은지 확인."""
    r5 = pd.read_parquet(common.RESULTS / "R5.parquet")
    ref = r5.groupby("lam_arr").agg(
        n_alt=("n_alt", "sum"),
        alt_disc=("dsr_alt_disc", "sum"),
    )

    messages = []
    for lam in LAM_ARR:
        cell = cells[(lam, "n_reg", "empirical")]
        got_alt = int(cell["n_alt"])
        got_disc = int(cell["alt_disc"])
        ref_alt = int(ref.loc[lam, "n_alt"])
        ref_disc = int(ref.loc[lam, "alt_disc"])
        got_power = got_disc / got_alt
        ref_power = ref_disc / ref_alt
        if got_alt != ref_alt or got_disc != ref_disc or got_power != ref_power:
            raise RuntimeError(
                "R5 gate 실패: "
                f"lam={lam}, got={got_disc}/{got_alt} ({got_power:.17g}), "
                f"ref={ref_disc}/{ref_alt} ({ref_power:.17g})"
            )
        messages.append(f"λ={lam:.1f}: {got_disc}/{got_alt}={got_power:.17g}")
    print("Gate (c) PASS: (N=n_reg, empirical) DSR alt power == sim_results/R5.parquet")
    for msg in messages:
        print(f"  {msg}")


def build_output(cells: dict[tuple[float, str, str], dict[str, float]]) -> pd.DataFrame:
    """셀 집계를 요청된 CSV 스키마로 변환한다."""
    rows = []
    for lam in LAM_ARR:
        for trials_def in TRIAL_DEFS:
            for moments in MOMENT_DEFS:
                cell = cells[(lam, trials_def, moments)]
                rows.append(
                    {
                        "lam_arr": lam,
                        "trials_def": trials_def,
                        "moments": moments,
                        "power": cell["alt_disc"] / max(cell["n_alt"], 1),
                        "n_disc": int(cell["n_disc"]),
                        "n_false": int(cell["n_false"]),
                        "fdp_mean": cell["fdp_sum"] / max(cell["n_run"], 1),
                        "guard_count": int(cell["guard_count"]),
                    }
                )
    return pd.DataFrame(rows)


def write_tex(out: pd.DataFrame) -> None:
    """Rev5 M4 DSR 민감도 LaTeX 표를 생성한다."""
    trial_labels = {
        "n_reg": "$n_{\\rm reg}$",
        "J=212": "$J=212$",
        "n_arrived": "$n_{\\rm arrived}$",
    }
    lines = [
        "% Rev5 M4 -- DSR comparator sensitivity (auto-generated by sim/run_R5b.py)",
        "\\begin{table}[htbp]",
        "\\centering\\small",
        "\\caption{DSR comparator sensitivity to trial-count definition and moment treatment. "
        "DSR per BLdP(2014): $SR_i$ is the monthly Sharpe of the strategy's "
        "post-registration score window $[A_j+1,\\min(A_j+120,T-1)]$ "
        "(mean/sd, ddof=1, $n_i$ = window length); skew/kurtosis = population "
        "moments of the same window (or fixed at Gaussian 0/3 in the "
        "``normal'' variant); $SR_0$ uses cross-sectional $V=\\operatorname{Var}(SR, "
        "\\mathrm{ddof}=1)$ and the stated trial count; terminal batch test at "
        "$T=360$; 50\\% of arrivals are alternatives with $\\delta=\\delta^*(120)$; "
        "strategies with $n_i<2$ are excluded from the family (counted); discovery "
        "iff $\\Phi(z)>0.95$.}",
        "\\label{tab:rev5-m4-dsr-sensitivity}",
        "\\begin{tabular}{cccc}",
        "\\toprule",
        "$\\lambda_{\\rm arr}$ & Trial count in $SR_0$ & empirical power & normal power \\\\",
        "\\midrule",
    ]
    for lam in LAM_ARR:
        for trials_def in TRIAL_DEFS:
            sub = out[(out["lam_arr"] == lam) & (out["trials_def"] == trials_def)]
            emp = float(sub[sub["moments"] == "empirical"]["power"].iloc[0])
            norm = float(sub[sub["moments"] == "normal"]["power"].iloc[0])
            lines.append(
                f"{lam:.1f} & {trial_labels[trials_def]} & "
                f"{format_pct3_tex(emp)} & {format_pct3_tex(norm)} \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{minipage}{0.94\\linewidth}",
            "\\footnotesize",
            "Notes: The R5 table values 0.2\\%/0.006\\%/0.004\\% match the "
            "$(n_{\\rm reg},\\ \\mathrm{empirical})$ column above; gate (c) "
            "asserts exact equality against \\texttt{sim\\_results/R5.parquet}.",
            "\\end{minipage}",
            "\\end{table}",
            "",
        ]
    )
    path = common.REPO / "review" / "appendix_rev5_m4_dsr.tex"
    path.write_text("\n".join(lines), encoding="utf-8")


def print_table(out: pd.DataFrame) -> None:
    """stdout용 18셀 요약표를 출력한다."""
    shown = out.copy()
    shown["power"] = shown["power"].map(format_pct3)
    shown["fdp_mean"] = shown["fdp_mean"].map(lambda x: f"{x:.6f}")
    print("\n18-cell DSR sensitivity table:")
    print(shown.to_string(index=False))


def main() -> pd.DataFrame:
    t0 = time.time()
    print("R5b DSR comparator sensitivity")
    print("Seed policy: E5 stream reused via sim.run_R5.regenerate_e5_world; no new RNG stream.")
    print(
        f"Grid: {len(LAM_ARR)} lambda_arr x {len(TRIAL_DEFS)} trial definitions "
        f"x {len(MOMENT_DEFS)} moment treatments, N_RUN={N_RUN}/lambda"
    )

    cells: dict[tuple[float, str, str], dict[str, float]] = {}
    for lam in LAM_ARR:
        tc = time.time()
        for run_idx in range(N_RUN):
            world_e5 = regenerate_e5_world(lam, run_idx)
            is_alt = world_e5["is_alt"]
            lens = world_e5["lens"]
            Ymat = world_e5["Ymat"]
            n_reg = int(world_e5["n_reg"])
            n_alt = int(is_alt.sum())
            for trials_def in TRIAL_DEFS:
                trials_n = trial_count(world_e5, trials_def)
                for moments in MOMENT_DEFS:
                    disc, _, _, guard_count = dsr_discoveries(
                        Ymat=Ymat,
                        lens=lens,
                        n_reg=n_reg,
                        trials_n=trials_n,
                        moments=moments,
                    )
                    update_cell(
                        cells=cells,
                        lam=lam,
                        trials_def=trials_def,
                        moments=moments,
                        n_alt=n_alt,
                        disc=disc,
                        is_alt=is_alt,
                        guard_count=guard_count,
                    )
        print(f"  lam_arr={lam}: 6 DSR variants complete ({time.time() - tc:.0f}s)", flush=True)

    assert_r5_gate(cells)
    out = build_output(cells)
    out_path = common.REPO / "data" / "rev5_m4_dsr_sensitivity.csv"
    out.to_csv(out_path, index=False)
    write_tex(out)
    print_table(out)
    print(f"\nGuard total: {int(out['guard_count'].sum())}")
    print(
        f"R5b 완료 (총 {time.time() - t0:.0f}s, commit {common.commit_hash()})\n"
        "saved -> data/rev5_m4_dsr_sensitivity.csv, review/appendix_rev5_m4_dsr.tex"
    )
    return out


if __name__ == "__main__":
    main()
