"""R5c — DSR alt-share sensitivity (Paper A micro 2).

E5형 mixed queue/timing world에서 alternative share만 바꾸고, terminal batch
DSR comparator의 power와 SR0 메커니즘 진단을 측정한다.

실행: MPLBACKEND=Agg python3 -m sim.run_R5c
"""
import time

import numpy as np
import pandas as pd
from scipy import stats

from . import common, eprocess as ep, world
from .run_E5 import ALPHA, DL, J_BUDGET, N_RUN, T
from .run_R5 import EULER_GAMMA, dsr_discoveries

ALT_SHARES = (0.1, 0.25, 0.5)
LAM_ARR = 1.0
R5_LAM1_REFERENCE_POWER = 5.6809574306923196e-05
SANITY_POWER_MAX = 0.001


def regenerate_e5_altshare_world(share: float, run_idx: int) -> dict[str, np.ndarray | int]:
    """E5형 등록 및 score 생성부를 새 seed 계열로 재생성한다."""
    rng = common.rng_for(common.EXP_ID["R5c"], int(share * 100), run_idx)
    d_star = float(ep.frontier.catoni[1])              # run_E5의 delta*(120) 정의

    # run_E5.run_one과 동일 추첨 순서, share만 파라미터: counts -> arrivals -> A -> is_alt -> eps -> Ymat
    counts = rng.poisson(LAM_ARR, T)
    arrivals = np.repeat(np.arange(T), counts)
    n_arrived = len(arrivals)
    A = arrivals[:J_BUDGET]
    budget_exh_month = int(arrivals[J_BUDGET - 1]) if n_arrived >= J_BUDGET else -1
    J = len(A)
    is_alt = rng.random(J) < share

    lens = np.minimum(DL, T - 1 - A)
    eps = world.draw_eps(rng, J, DL, noise="gaussian")
    Ymat = np.zeros((J, DL))
    for j in range(J):
        Ymat[j, : lens[j]] = eps[j, : lens[j]] + (d_star if is_alt[j] else 0.0)
    return {
        "A": A,
        "is_alt": is_alt,
        "lens": lens,
        "Ymat": Ymat,
        "n_arrived": n_arrived,
        "n_reg": J,
        "budget_exh_month": budget_exh_month,
    }


def dsr_sr0_diagnostics(Ymat: np.ndarray, lens: np.ndarray, n_reg: int) -> tuple[float, float]:
    """R5 baseline 창에서 sqrt(V)와 SR0만 재계산한다."""
    family = lens >= 2
    idx = np.nonzero(family)[0]
    if n_reg < 2 or len(idx) < 2:
        return np.nan, np.nan

    sr = np.empty(len(idx))
    for out_i, j in enumerate(idx):
        y = Ymat[j, : lens[j]]
        n_i = len(y)
        m1 = float(y.mean())
        centered = y - m1
        m2 = float((centered ** 2).mean())
        assert m2 > 0.0, "DSR sd 정의 불가 경로 발생"
        sr[out_i] = m1 / np.sqrt(m2 * n_i / (n_i - 1))

    sqrt_v = float(np.sqrt(np.var(sr, ddof=1)))
    sr0 = sqrt_v * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_reg)
        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_reg * np.e))
    )
    return sqrt_v, float(sr0)


def format_pct3(x: float) -> str:
    """stdout 표용 고정 소수점 3자리 퍼센트 포맷."""
    if pd.isna(x):
        return "NA"
    return f"{100 * x:.3f}%"


def format_float(x: float, digits: int = 6) -> str:
    """stdout 표용 실수 포맷."""
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def build_row(share: float, run_rows: list[dict[str, float]]) -> dict[str, float]:
    """런 단위 결과를 요청된 CSV 스키마로 집계한다."""
    total_alt = sum(int(row["n_alt"]) for row in run_rows)
    total_alt_disc = sum(int(row["n_alt_disc"]) for row in run_rows)
    total_disc = sum(int(row["n_disc"]) for row in run_rows)
    total_false = sum(int(row["n_false"]) for row in run_rows)
    total_delay_count = sum(int(row["delay_count"]) for row in run_rows)
    total_delay_sum = sum(float(row["delay_sum"]) for row in run_rows)
    return {
        "share": share,
        "n_run": len(run_rows),
        "alt_power": total_alt_disc / max(total_alt, 1),
        "n_disc": total_disc,
        "n_false": total_false,
        "fdp_mean": float(np.mean([row["fdp"] for row in run_rows])),
        "null_share": total_false / max(total_disc, 1),
        "delay_mean": total_delay_sum / total_delay_count if total_delay_count else np.nan,
        "sqrtV_mean": float(np.nanmean([row["sqrtV"] for row in run_rows])),
        "sr0_mean": float(np.nanmean([row["sr0"] for row in run_rows])),
    }


def assert_sanity_gate(out: pd.DataFrame) -> str:
    """share=0.5의 DSR power가 R5 lambda=1.0과 같은 자릿수인지 확인한다."""
    power = float(out.loc[out["share"] == 0.5, "alt_power"].iloc[0])
    if power > SANITY_POWER_MAX:
        raise RuntimeError(
            "Sanity gate 실패: "
            f"share=0.5 power={power:.8g} > {SANITY_POWER_MAX:.8g}; "
            f"R5 lambda=1.0 기준 power={R5_LAM1_REFERENCE_POWER:.8g}"
        )
    return (
        "Sanity gate PASS: "
        f"share=0.5 power={format_pct3(power)} <= {format_pct3(SANITY_POWER_MAX)} "
        f"(R5 lambda=1.0 reference={format_pct3(R5_LAM1_REFERENCE_POWER)})"
    )


def print_table(out: pd.DataFrame) -> None:
    """stdout용 share별 요약표를 출력한다."""
    shown = out.copy()
    shown["alt_power"] = shown["alt_power"].map(format_pct3)
    shown["fdp_mean"] = shown["fdp_mean"].map(lambda x: format_float(x, 6))
    shown["null_share"] = shown["null_share"].map(lambda x: format_float(x, 6))
    shown["delay_mean"] = shown["delay_mean"].map(lambda x: format_float(x, 1))
    shown["sqrtV_mean"] = shown["sqrtV_mean"].map(lambda x: format_float(x, 6))
    shown["sr0_mean"] = shown["sr0_mean"].map(lambda x: format_float(x, 6))
    print("\nR5c DSR alt-share sensitivity:")
    print(shown.to_string(index=False))


def main() -> pd.DataFrame:
    """R5c alt-share grid를 실행하고 CSV 산출물을 기록한다."""
    t0 = time.time()
    print("R5c DSR alt-share sensitivity")
    print("Seed policy: rng = common.rng_for(common.EXP_ID['R5c'], int(share*100), run_idx)")
    print(f"Grid: alt_share={ALT_SHARES}, lambda_arr={LAM_ARR}, N_RUN={N_RUN}/share")
    print("DSR definition: run_R5.dsr_discoveries 그대로 사용 (N=n_reg, empirical moments)")

    rows = []
    for share in ALT_SHARES:
        tc = time.time()
        run_rows = []
        for run_idx in range(N_RUN):
            world_e5 = regenerate_e5_altshare_world(share, run_idx)
            A = world_e5["A"]
            is_alt = world_e5["is_alt"]
            lens = world_e5["lens"]
            Ymat = world_e5["Ymat"]
            n_reg = int(world_e5["n_reg"])

            disc, _, _ = dsr_discoveries(Ymat, lens, n_reg, alpha=ALPHA)
            alt_disc = disc & is_alt
            false_disc = disc & ~is_alt
            delays = (T - A[alt_disc]).astype(int)
            n_disc = int(disc.sum())
            n_false = int(false_disc.sum())
            sqrt_v, sr0 = dsr_sr0_diagnostics(Ymat, lens, n_reg)
            run_rows.append(
                {
                    "n_alt": int(is_alt.sum()),
                    "n_alt_disc": int(alt_disc.sum()),
                    "n_disc": n_disc,
                    "n_false": n_false,
                    "fdp": n_false / max(n_disc, 1),
                    "delay_sum": float(delays.sum()) if len(delays) else 0.0,
                    "delay_count": int(len(delays)),
                    "sqrtV": sqrt_v,
                    "sr0": sr0,
                }
            )
        rows.append(build_row(share, run_rows))
        print(f"  share={share:.2f}: DSR 완료 ({time.time() - tc:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    gate_message = assert_sanity_gate(out)
    out_path = common.REPO / "data" / "rev5_m4_dsr_altshare.csv"
    out.to_csv(out_path, index=False)
    print(gate_message)
    print_table(out)
    print(f"\nR5c 완료 (총 {time.time() - t0:.0f}s, commit {common.commit_hash()})")
    print("saved -> data/rev5_m4_dsr_altshare.csv")
    return out


if __name__ == "__main__":
    main()
