"""R2 — deflated Sharpe ratio 비교군 (심사 M3 대응).

기존 E1 세계·등록 집합(16셀, 동일 seed 스트림으로 재생성) 재사용. 각 런 종료
시점에 등록 전략 전체를 batch family로 취급, full-history Sharpe
([0, min(D_j, T-1)], naive 비교군과 동일 창)에 Bailey & López de Prado (2014,
JPM 40(5)) deflated Sharpe ratio 적용:

  trials N = n_reg,  V = Var({SR_i}, ddof=1)  (family 내 SR 추정치의 횡단면 분산)
  SR0 = sqrt(V) * [ (1-γ_EM)·Φ⁻¹(1-1/N) + γ_EM·Φ⁻¹(1-1/(N·e)) ]   (γ_EM = 0.5772…)
  DSR_i = Φ( (SR_i - SR0)·sqrt(n_i-1) / sqrt(1 - γ3_i·SR_i + ((γ4_i-1)/4)·SR_i²) )
  (γ3 = 수익률 왜도, γ4 = 첨도(non-excess), 모두 population 모멘트; SR은 월간,
   sd는 ddof=1; n_i = 전략별 관측 개월 수 — BLdP의 공통 n을 전략별로 일반화)

발견: DSR p = 1 - DSR < 0.05 (즉 DSR > 0.95). 전부-null이므로 FDP = 1{발견>=1}.
full protocol / naive / fresh-peek도 같은 런에서 재계산해 4절차 비교 그림으로
sim_E1_supfdr.png 교체. 실행: MPLBACKEND=Agg python3 -m sim.run_R2
"""
import itertools
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from . import common, ebh, searcher
from .registry import Registry
from .run_E1 import comparators, ALPHA, J_BUDGET, T, L, W, DEADLINE_LEN, M_ENV

N_RUN = 2000
EULER_GAMMA = 0.5772156649015329

CELLS = list(itertools.product(["gaussian", "t5"], [0.0, 0.3],
                               [0.15, 0.30], ["honest", "adversarial"]))


def dsr_discoveries(regs, alpha=ALPHA):
    """BLdP(2014) DSR: 전략별 발견 여부 (batch family = 등록 전체)."""
    J = len(regs)
    if J < 2:
        return np.zeros(J, bool)
    A = np.array([r.A for r in regs])
    S = np.stack([r.series for r in regs])
    n = np.minimum(A + DEADLINE_LEN, T - 1) + 1
    jj = np.arange(J)
    c1 = np.cumsum(S, axis=1)[jj, n - 1]
    c2 = np.cumsum(S ** 2, axis=1)[jj, n - 1]
    c3 = np.cumsum(S ** 3, axis=1)[jj, n - 1]
    c4 = np.cumsum(S ** 4, axis=1)[jj, n - 1]
    m1 = c1 / n
    m2c = c2 / n - m1 ** 2                       # population 분산
    m3c = c3 / n - 3 * m1 * c2 / n + 2 * m1 ** 3
    m4c = c4 / n - 4 * m1 * c3 / n + 6 * m1 ** 2 * c2 / n - 3 * m1 ** 4
    sd_s = np.sqrt(m2c * n / (n - 1))            # 표본 sd (ddof=1) — SR용
    sr = m1 / sd_s                               # 월간 SR
    g3 = m3c / m2c ** 1.5                        # 왜도
    g4 = m4c / m2c ** 2                          # 첨도 (non-excess)
    V = np.var(sr, ddof=1)
    z1 = stats.norm.ppf(1 - 1 / J)
    z2 = stats.norm.ppf(1 - 1 / (J * np.e))
    sr0 = np.sqrt(V) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    denom = np.clip(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12, None)
    z = (sr - sr0) * np.sqrt(n - 1) / np.sqrt(denom)
    dsr = stats.norm.cdf(z)
    return dsr > 1 - alpha                       # p = 1-DSR < alpha


def run_one(noise, rho, s_reg, behavior, run_idx):
    nc, bc = common.NOISE_CODE[noise], common.BEHAVIOR_CODE[behavior]
    # E1과 동일한 seed 스트림 → 동일 세계·등록 집합 재생성
    rng_world = common.rng_for(common.EXP_ID["E1"], 0, nc, int(rho * 10), run_idx)
    rng_mut = common.rng_for(common.EXP_ID["E1"], 1, nc, int(rho * 10),
                             int(s_reg * 100), bc, run_idx)
    reg = Registry(J_BUDGET, ALPHA, M_ENV, DEADLINE_LEN)
    searcher.run_search(rng_world, rng_mut, reg, T=T, L=L, W=W, s_reg=s_reg,
                        behavior=behavior, noise=noise, rho=rho)
    regs = reg.regs
    e_log, tau_cal, _, _, is_null = ebh.freeze_summary(regs, T, reg.log_b)
    base = ebh.baseline_reveal(e_log, tau_cal, is_null, ALPHA, J_BUDGET)
    naive_disc, fresh_disc = comparators(regs)
    dsr_disc = dsr_discoveries(regs)
    return {"run_id": run_idx, "noise": noise, "rho": rho, "s_reg": s_reg,
            "behavior": behavior, "n_reg": len(regs),
            "sup_fdp": base["sup_fdp"],
            "naive_fdp": float(naive_disc.any()), "fresh_fdp": float(fresh_disc.any()),
            "dsr_fdp": float(dsr_disc.any()), "dsr_n_disc": int(dsr_disc.sum())}


def main(n_run=N_RUN):
    t0 = time.time()
    rows = []
    for ci, (noise, rho, s_reg, behavior) in enumerate(CELLS):
        tc = time.time()
        for r in range(n_run):
            rows.append(run_one(noise, rho, s_reg, behavior, r))
        sub = pd.DataFrame(rows[-n_run:])
        print(f"[{ci+1:2d}/16] {noise:>8} rho={rho} s_reg={s_reg} {behavior:>11}: "
              f"proto={sub['sup_fdp'].mean():.4f} dsr={sub['dsr_fdp'].mean():.3f} "
              f"naive={sub['naive_fdp'].mean():.3f} ({time.time()-tc:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    path = common.write_parquet(df, "R2.parquet")

    g = df.groupby(["noise", "rho", "s_reg", "behavior"], sort=False)
    summ = g.agg(supfdr=("sup_fdp", "mean"), naive=("naive_fdp", "mean"),
                 fresh=("fresh_fdp", "mean"), dsr=("dsr_fdp", "mean"),
                 dsr_ndisc=("dsr_n_disc", "mean"), n_reg=("n_reg", "mean"),
                 n=("sup_fdp", "size")).reset_index()
    print("\n" + summ.to_string(index=False))
    print(f"\nR2 완료 (총 {time.time()-t0:.0f}s, commit {common.commit_hash()})")

    # ---- 그림: E1 4절차 비교로 교체 ----
    okabe = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    fig, ax = plt.subplots(figsize=(11.5, 5))
    x = np.arange(len(summ))
    ax.bar(x - 0.30, summ["supfdr"], 0.19, color=okabe[0],
           label="full protocol (frozen e-values, baseline reveal)")
    ax.bar(x - 0.10, summ["naive"], 0.19, color=okabe[1],
           label="naive full-history t-test (incl. in-sample)")
    ax.bar(x + 0.10, summ["fresh"], 0.19, color=okabe[2],
           label="fresh-data repeated t-test (uncorrected peeking)")
    ax.bar(x + 0.30, summ["dsr"], 0.19, color=okabe[3],
           label="deflated SR, batch family (Bailey & LdP 2014)")
    se_ref = np.sqrt(ALPHA * (1 - ALPHA) / n_run)
    ax.axhline(ALPHA, color="#D55E00", lw=1.5, label=f"α = {ALPHA}")
    ax.axhline(ALPHA + 3 * se_ref, color="#D55E00", lw=1, ls="--",
               label="α + 3×MC SE (pass line, protocol only)")
    labels = [f"{n[0]}|ρ{r}|s{s}|{b[:3]}" for n, r, s, b in
              zip(summ["noise"], summ["rho"], summ["s_reg"], summ["behavior"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("empirical SupFDR (all-null: P(any false discovery))")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"E1 — protocol vs naive/DSR procedures, all-null adaptive search "
                 f"(N_RUN={n_run}/cell)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(common.REPO / "sim_E1_supfdr.png", dpi=140)
    print(f"saved -> {path.name}, sim_E1_supfdr.png")
    return summ


if __name__ == "__main__":
    main()
