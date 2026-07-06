"""E1 — Validity (최우선 게이트). baseline reveal 판 + 비교군 2종.

전부-null 세계 + 적응 등록 + peeking/조기중단. 세계·seed는 1차 실행과 동일 (CRN
재사용). 세 절차를 같은 등록 집합에 적용:

  (a) full protocol — 동결 e-value의 등록순 reveal + e-BH (ebh.baseline_reveal).
      합격 기준: empirical SupFDR <= alpha + 3*SE(p=alpha). 초과 시 STOP.
  (b) 비교군 1 naive full-history — 각 등록 전략의 전체 이력(in-sample 포함,
      [0, min(D_j, T-1)])에 1표본 t-검정 (H1: mean>0, alpha=0.05). 파생 전략은
      잠재 시계열 전체 = "변형 전략의 풀히스토리 백테스트"에 해당. 포기와 무관하게
      데드라인까지 사용 (naive 연구자는 abandon하지 않음).
  (c) 비교군 2 fresh-data 반복 검정 — post-A_j 데이터에 매월 누적 t-검정
      (최소 6개월부터), 한 번이라도 p < alpha면 발견 (peeking 미교정).
      역시 데드라인까지 사용.

전부-null이므로 비교군 FDP는 발견 1건 이상이면 1.
live 변형(online_ebh)은 ebh.py에 secondary로 보존 (1차 결과: 전 셀 SupFDR=0).
셀: {noise} × {rho} × {s_reg} × {behavior} = 16, N_RUN=2000.
설계 상수: T=480, L=100, W=36, J_budget=212, D_j=A_j+120.
실행: MPLBACKEND=Agg python3 -m sim.run_E1
"""
import itertools
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from . import common, ebh, searcher
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

# 1표본 t-검정 (one-sided, alpha=0.05) 기각 임계값, index = df-1
T_CRIT = stats.t.ppf(1 - ALPHA, np.arange(1, T + 1))


def comparators(regs):
    """비교군 두 종의 전략별 '발견' 여부 (전부-null 세계 전제)."""
    J = len(regs)
    if J == 0:
        return np.zeros(0, bool), np.zeros(0, bool)
    A = np.array([r.A for r in regs])
    S = np.stack([r.series for r in regs])            # (J, T) 잠재 시계열
    # (b) naive full-history: [0, min(D_j, T-1)]
    n = np.minimum(A + DEADLINE_LEN, T - 1) + 1
    cs = np.cumsum(S, axis=1)
    cs2 = np.cumsum(S * S, axis=1)
    jj = np.arange(J)
    m1 = cs[jj, n - 1] / n
    v = (cs2[jj, n - 1] - n * m1 ** 2) / (n - 1)
    t_full = m1 / np.sqrt(v / n)
    naive_disc = t_full >= T_CRIT[n - 2]              # df = n-1
    # (c) fresh-data 반복 t-검정: post-A_j, s = 6..min(120, T-1-A)
    lens = np.minimum(DEADLINE_LEN, T - 1 - A)
    P = np.zeros((J, DEADLINE_LEN))
    for j in range(J):
        if lens[j] > 0:
            P[j, :lens[j]] = regs[j].series[A[j] + 1:A[j] + 1 + lens[j]]
    s = np.arange(1, DEADLINE_LEN + 1)[None, :]
    cp = np.cumsum(P, axis=1)
    cp2 = np.cumsum(P * P, axis=1)
    mu = cp / s
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.clip((cp2 - s * mu ** 2) / np.maximum(s - 1, 1), 0, None)
        t_s = mu / np.sqrt(var / s)
    valid = (s >= 6) & (s <= lens[:, None])
    fresh_disc = ((t_s >= T_CRIT[np.maximum(s - 2, 0)]) & valid).any(axis=1)
    return naive_disc, fresh_disc


def run_one(noise, rho, s_reg, behavior, run_idx):
    nc, bc = common.NOISE_CODE[noise], common.BEHAVIOR_CODE[behavior]
    # CRN: 1차 실행과 동일한 seed 스트림 (세계 잡음은 s_reg/behavior와 무관)
    rng_world = common.rng_for(common.EXP_ID["E1"], 0, nc, int(rho * 10), run_idx)
    rng_mut = common.rng_for(common.EXP_ID["E1"], 1, nc, int(rho * 10),
                             int(s_reg * 100), bc, run_idx)
    reg = Registry(J_BUDGET, ALPHA, M_ENV, DEADLINE_LEN)
    searcher.run_search(rng_world, rng_mut, reg, T=T, L=L, W=W, s_reg=s_reg,
                        behavior=behavior, noise=noise, rho=rho)
    regs = reg.regs
    e_log, tau_cal, solo, _, is_null = ebh.freeze_summary(regs, T, reg.log_b)
    base = ebh.baseline_reveal(e_log, tau_cal, is_null, ALPHA, J_BUDGET)
    naive_disc, fresh_disc = comparators(regs)
    return {"run_id": run_idx, "noise": noise, "rho": rho, "s_reg": s_reg,
            "behavior": behavior, "n_reg": len(regs),
            "n_abandoned": sum(1 for r in regs if r.abandon_t is not None),
            "budget_exhausted": reg.exhausted,
            "sup_fdp": base["sup_fdp"], "n_disc": base["n_disc"],
            "n_solo": int(solo.sum()), "max_log_e": float(e_log.max(initial=0.0)),
            "naive_fdp": float(naive_disc.any()), "naive_n_disc": int(naive_disc.sum()),
            "fresh_fdp": float(fresh_disc.any()), "fresh_n_disc": int(fresh_disc.sum())}


def main(n_run=N_RUN):
    t0 = time.time()
    rows = []
    for ci, (noise, rho, s_reg, behavior) in enumerate(CELLS):
        tc = time.time()
        for r in range(n_run):
            rows.append(run_one(noise, rho, s_reg, behavior, r))
        sub = pd.DataFrame(rows[-n_run:])
        print(f"[{ci+1:2d}/16] {noise:>8} rho={rho} s_reg={s_reg} {behavior:>11}: "
              f"SupFDR base={sub['sup_fdp'].mean():.5f} "
              f"naive={sub['naive_fdp'].mean():.3f} fresh={sub['fresh_fdp'].mean():.3f} "
              f"({time.time()-tc:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    path = common.write_parquet(df, "E1.parquet")

    g = df.groupby(["noise", "rho", "s_reg", "behavior"], sort=False)
    summ = g.agg(supfdr=("sup_fdp", "mean"), naive=("naive_fdp", "mean"),
                 fresh=("fresh_fdp", "mean"), n=("sup_fdp", "size"),
                 n_reg=("n_reg", "mean"),
                 naive_ndisc=("naive_n_disc", "mean"),
                 fresh_ndisc=("fresh_n_disc", "mean"),
                 disc_total=("n_disc", "sum"),
                 max_log_e=("max_log_e", "max")).reset_index()
    se_ref = np.sqrt(ALPHA * (1 - ALPHA) / n_run)
    thr = ALPHA + 3 * se_ref
    summ["pass"] = summ["supfdr"] <= thr
    print(f"\n합격선 (full protocol만): alpha + 3*SE(p=alpha) = {thr:.4f}")
    print(summ.to_string(index=False))
    all_pass = bool(summ["pass"].all())
    print(f"\nE1(baseline reveal) {'PASS — 전체 셀 합격' if all_pass else '*** FAIL: STOP ***'}"
          f"   (총 {time.time()-t0:.0f}s, commit {common.commit_hash()})")

    # ---- 그림: 3절차 SupFDR 비교 ----
    okabe = ["#0072B2", "#E69F00", "#009E73"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(summ))
    ax.bar(x - 0.27, summ["supfdr"], 0.25, color=okabe[0],
           label="full protocol (frozen e-values, baseline reveal)")
    ax.bar(x, summ["naive"], 0.25, color=okabe[1],
           label="naive full-history t-test (incl. in-sample)")
    ax.bar(x + 0.27, summ["fresh"], 0.25, color=okabe[2],
           label="fresh-data repeated t-test (uncorrected peeking)")
    ax.axhline(ALPHA, color="#D55E00", lw=1.5, label=f"α = {ALPHA}")
    ax.axhline(thr, color="#D55E00", lw=1, ls="--", label="α + 3×MC SE (pass line)")
    labels = [f"{n[0]}|ρ{r}|s{s}|{b[:3]}" for n, r, s, b in
              zip(summ["noise"], summ["rho"], summ["s_reg"], summ["behavior"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("empirical SupFDR (all-null: P(any false discovery))")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"E1 — protocol vs naive procedures, all-null adaptive search "
                 f"(N_RUN={n_run}/cell)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(common.REPO / "sim_E1_supfdr.png", dpi=140)
    print(f"saved -> {path.name}, sim_E1_supfdr.png")
    return all_pass


if __name__ == "__main__":
    main()
