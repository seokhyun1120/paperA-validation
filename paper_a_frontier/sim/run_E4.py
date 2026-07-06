"""E4 — envelope 미스펙 스트레스 (경험적 degradation curve).

전부-null 세계에서 진짜 score 분산을 등록 envelope의 xi배로 부풀림:
  Var(Y) = xi * m_env_reg^2  (Y = m_env_reg * sqrt(xi) * eps, eps 단위분산)
e-process는 등록값 m_env_reg로 베팅/페널티 계산 (envelope 위반 상황).
측정: 실현 SupFDR의 경험적 degradation curve. α·ξ_var는 참조선일 뿐
Corollary 2의 상계가 아님 (해당 정리는 지속적 분산 위반을 커버하지 않음) —
PASS/FAIL 게이트 없음.

셀: m_env_reg ∈ {1.2, 1.3, 1.5} × xi ∈ {1.2, 1.5, 2.0} + 기준셀 (1.3, 1.0).
J=212 전략 동시 등록(t=0), D=120, gaussian, N_RUN=2,000/셀.
실행: MPLBACKEND=Agg python3 -m sim.run_E4
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import common, ebh, eprocess as ep, world

ALPHA = 0.05
J_BUDGET = 212
J_REG = 212          # 예산 전액 등록 (union 최악 케이스)
D = 120
N_RUN = 2_000
BATCH = 40

CELLS = [(1.3, 1.0)] + [(m, x) for m in (1.2, 1.3, 1.5) for x in (1.2, 1.5, 2.0)]


def run_cell(m_env, xi):
    log_b = ep.log_b_solo(1.0 / J_BUDGET)
    scale = m_env * np.sqrt(xi)
    sup_fdps, n_discs = [], []
    ci = CELLS.index((m_env, xi))
    for b in range(N_RUN // BATCH):
        rng = common.rng_for(common.EXP_ID["E4"], ci, b)
        eps = world.draw_eps(rng, BATCH * J_REG, D, noise="gaussian")
        logE = ep.log_e_path(scale * eps, m_env=m_env)
        logE_frozen, _ = ep.freeze_at_crossing(logE, log_b)
        logE_frozen = logE_frozen.reshape(BATCH, J_REG, D)
        is_null = np.ones(J_REG, bool)
        for r in range(BATCH):
            res = ebh.online_ebh(logE_frozen[r].T, is_null, ALPHA, J_BUDGET)
            sup_fdps.append(res["sup_fdp"])
            n_discs.append(res["n_disc"])
    return np.array(sup_fdps), np.array(n_discs)


def main():
    t0 = time.time()
    rows = []
    for m_env, xi in CELLS:
        sup_fdps, n_discs = run_cell(m_env, xi)
        supfdr = sup_fdps.mean()
        se = sup_fdps.std(ddof=1) / np.sqrt(len(sup_fdps))
        rows.append({"m_env_reg": m_env, "xi": xi, "n_run": N_RUN,
                     "supfdr": supfdr, "mc_se": se,
                     "ref_alpha_xi_var": ALPHA * xi,
                     "disc_total": int(n_discs.sum()),
                     "any_disc_rate": float((n_discs > 0).mean())})
        print(f"  m_env={m_env} xi={xi}: SupFDR={supfdr:.5f} (SE {se:.5f}) "
              f"vs ref α·ξ_var={ALPHA * xi:.3f} ({time.time()-t0:.0f}s)", flush=True)
    summ = pd.DataFrame(rows)
    path = common.write_parquet(summ, "E4.parquet")
    print("\n" + summ.to_string(index=False))
    print(f"\nE4 empirical degradation curve 완료 "
          f"(총 {time.time()-t0:.0f}s, commit {common.commit_hash()})")

    # ---- 그림: SupFDR vs xi (경험적 degradation curve) ----
    okabe = ["#0072B2", "#E69F00", "#009E73"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    xs = np.array([1.0, 1.2, 1.5, 2.0])
    ax.plot(xs, ALPHA * xs, "--", color="#D55E00", lw=2,
            label="reference line α·ξ_var (not the Corollary-2 bound)")
    ax.axhline(ALPHA, color="gray", lw=1, ls=":", label="α")
    for mi, m_env in enumerate((1.2, 1.3, 1.5)):
        s = summ[summ["m_env_reg"] == m_env].sort_values("xi")
        ax.plot(s["xi"], s["supfdr"], "o-", lw=2, color=okabe[mi],
                label=f"m_env_reg={m_env}")
    ax.set_xlabel("true variance inflation ξ_var (Var = ξ_var·m_env²)")
    ax.set_ylabel("empirical SupFDR")
    ax.set_title(f"E4 — envelope misspecification stress, empirical degradation "
                 f"curve\n(all-null, J={J_REG}, D={D}, N_RUN={N_RUN}/cell)",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(common.REPO / "sim_E4_stress.png", dpi=140)
    print(f"saved -> {path.name}, sim_E4_stress.png")
    return summ


if __name__ == "__main__":
    main()
