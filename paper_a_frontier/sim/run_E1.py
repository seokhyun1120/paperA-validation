"""E1 — Validity (최우선 게이트).

전부-null 세계 + 적응 등록 + peeking/조기중단. 측정: 실현 sup_m FDP의 평균(SupFDR).
셀: {noise: gaussian, t5} × {rho: 0, 0.3} × {s_reg: 약 0.15, 강 0.30} ×
    {behavior: honest, adversarial}  — 16셀, N_RUN=2000/셀.
합격 기준: empirical SupFDR <= alpha + 3*MC SE. 초과 시 STOP (파라미터 조정 금지).

설계 상수 (RUN_LOG에 기재):
  T=480, L=100, W=36, J_budget=212 (b_solo = 212/0.05 = 4240 = frontier B_SOLO),
  D_j = A_j + 120 (primary).
CRN: 기본 라이브러리 잡음 seed는 (noise, rho, run)만 사용 — s_reg/behavior 축과
무관하게 동일 세계를 공유. 클론/뮤턴트 잡음은 별도 스트림 (behavior 포함).
실행: MPLBACKEND=Agg python3 -m sim.run_E1
"""
import itertools
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import common, ebh, eprocess as ep, searcher
from .registry import Registry

ALPHA = 0.05
J_BUDGET = 212
T = 480
L = 100
W = 36
DEADLINE_LEN = 120
N_RUN = 2000
M_ENV = 1.3

CELLS = list(itertools.product(["gaussian", "t5"], [0.0, 0.3],
                               [0.15, 0.30], ["honest", "adversarial"]))


def run_one(noise, rho, s_reg, behavior, run_idx):
    nc, bc = common.NOISE_CODE[noise], common.BEHAVIOR_CODE[behavior]
    # CRN: 세계 잡음은 s_reg/behavior와 무관
    rng_world = common.rng_for(common.EXP_ID["E1"], 0, nc, int(rho * 10), run_idx)
    rng_mut = common.rng_for(common.EXP_ID["E1"], 1, nc, int(rho * 10),
                             int(s_reg * 100), bc, run_idx)
    reg = Registry(J_BUDGET, ALPHA, M_ENV, DEADLINE_LEN)
    searcher.run_search(rng_world, rng_mut, reg, T=T, L=L, W=W, s_reg=s_reg,
                        behavior=behavior, noise=noise, rho=rho)
    regs = reg.regs
    E_cal, is_null, tau_cal, _ = ebh.build_calendar_logE(regs, T, reg.log_b)
    res = ebh.online_ebh(E_cal, is_null, ALPHA, J_BUDGET)
    n_ab = sum(1 for r in regs if r.abandon_t is not None)
    max_logE = float(E_cal.max()) if len(regs) else 0.0
    return {"run_id": run_idx, "noise": noise, "rho": rho, "s_reg": s_reg,
            "behavior": behavior, "n_reg": len(regs),
            "n_base": sum(1 for r in regs if r.kind == "base"),
            "n_derived": sum(1 for r in regs if r.kind != "base"),
            "n_abandoned": n_ab, "budget_exhausted": reg.exhausted,
            "sup_fdp": res["sup_fdp"], "n_disc": res["n_disc"],
            "n_false": res["n_false"], "n_solo_cross": int((tau_cal >= 0).sum()),
            "max_logE": max_logE}


def main(n_run=N_RUN):
    t0 = time.time()
    rows = []
    for ci, (noise, rho, s_reg, behavior) in enumerate(CELLS):
        tc = time.time()
        for r in range(n_run):
            rows.append(run_one(noise, rho, s_reg, behavior, r))
        sub = pd.DataFrame(rows[-n_run:])
        print(f"[{ci+1:2d}/16] {noise:>8} rho={rho} s_reg={s_reg} {behavior:>11}: "
              f"SupFDR={sub['sup_fdp'].mean():.5f}  n_reg(평균)={sub['n_reg'].mean():.1f} "
              f"disc합={sub['n_disc'].sum()}  ({time.time()-tc:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    path = common.write_parquet(df, "E1.parquet")

    # 셀별 요약 + 합격 판정
    g = df.groupby(["noise", "rho", "s_reg", "behavior"], sort=False)
    summ = g.agg(supfdr=("sup_fdp", "mean"), n=("sup_fdp", "size"),
                 n_reg=("n_reg", "mean"), n_derived=("n_derived", "mean"),
                 n_abandoned=("n_abandoned", "mean"),
                 budget_exh=("budget_exhausted", "mean"),
                 disc_total=("n_disc", "sum"),
                 max_logE=("max_logE", "max")).reset_index()
    summ["mc_se"] = np.sqrt(summ["supfdr"] * (1 - summ["supfdr"]) / summ["n"])
    # SupFDR=0인 셀은 SE=0 -> 기준선 SE는 p=alpha 가정으로 보수적으로 산정
    se_ref = np.sqrt(ALPHA * (1 - ALPHA) / n_run)
    summ["threshold"] = ALPHA + 3 * se_ref
    summ["pass"] = summ["supfdr"] <= summ["threshold"]
    print(f"\n기준: SupFDR <= alpha + 3*SE(p=alpha 기준) = {ALPHA + 3*se_ref:.4f}")
    print(summ.to_string(index=False))
    all_pass = bool(summ["pass"].all())
    print(f"\nE1 {'PASS — 전체 셀 합격' if all_pass else '*** FAIL: STOP ***'}"
          f"   (총 {time.time()-t0:.0f}s, commit {common.commit_hash()})")

    # ---- 그림: 셀별 SupFDR (전부-null 세계) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{n[0]}|ρ{r}|s{s}|{b[:3]}" for n, r, s, b in
              zip(summ["noise"], summ["rho"], summ["s_reg"], summ["behavior"])]
    ax.bar(range(len(summ)), summ["supfdr"], color="#0072B2", width=0.6,
           label="empirical SupFDR")
    ax.axhline(ALPHA, color="#D55E00", lw=2, label=f"α = {ALPHA}")
    ax.axhline(ALPHA + 3 * se_ref, color="#D55E00", lw=1.2, ls="--",
               label="α + 3×MC SE (pass line)")
    ax.set_xticks(range(len(summ)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("empirical SupFDR")
    ax.set_ylim(0, max(0.08, summ["supfdr"].max() * 1.3 + 1e-3))
    ax.set_title(f"E1 — all-null world, adaptive registration "
                 f"(N_RUN={n_run}/cell, J_budget={J_BUDGET})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(common.REPO / "sim_E1_supfdr.png", dpi=140)
    print(f"saved -> {path.name}, sim_E1_supfdr.png")
    return all_pass


if __name__ == "__main__":
    main()
