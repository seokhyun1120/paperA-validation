"""E5 — Queue/timing (v4 §5.8 3항 분해, baseline reveal 판).

등록 도착 과정: 포아송 강도 lam_arr/월, 도착 즉시 등록 (A_j = arrival; 수용능력
제약 없음), 예산 J_budget=212 소진 시 거절. 도착의 50%는 alternative
(delta = delta*(120) 상수), 50%는 null. D_j = A_j + 120, T = 360, gaussian.

발견 시점은 baseline reveal (동결 e-value의 등록순 reveal)로 계산.
발견된 alternative별 분해 (전 항 >= 0):
  T_freeze = tau_j - A_j          (동결까지: solo 도달 또는 deadline)
  W_fifo   = B_j - tau_j          (등록순 큐: 앞 전략들의 동결 대기)
  T_unlock = B_{M_disc} - B_j     (reveal 후 문턱 완화 대기)
총 지연 (A_j 기준) = T_freeze + W_fifo + T_unlock.
셀: lam_arr ∈ {0.5, 1.0, 2.0}, N_RUN = 500/셀.
실행: MPLBACKEND=Agg python3 -m sim.run_E5
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import common, ebh, eprocess as ep, world

ALPHA = 0.05
J_BUDGET = 212
M_ENV = 1.3
T = 360
DL = 120
N_RUN = 500
P_ALT = 0.5
LAM_ARR = [0.5, 1.0, 2.0]


def run_one(lam, run_idx):
    rng = common.rng_for(common.EXP_ID["E5"], int(lam * 10), run_idx)
    log_b = ep.log_b_solo(1.0 / J_BUDGET)
    d_star = float(ep.frontier.catoni[1])              # delta*(120)

    # 도착 즉시 등록 (예산 소진 시 거절)
    counts = rng.poisson(lam, T)
    arrivals = np.repeat(np.arange(T), counts)
    n_arrived = len(arrivals)
    A = arrivals[:J_BUDGET]
    budget_exh_month = int(arrivals[J_BUDGET - 1]) if n_arrived >= J_BUDGET else -1
    J = len(A)
    is_alt = rng.random(J) < P_ALT

    # post-A score, delta 주입 → e-process 동결
    lens = np.minimum(DL, T - 1 - A)
    eps = world.draw_eps(rng, J, DL, noise="gaussian")
    Ymat = np.zeros((J, DL))
    for j in range(J):
        Ymat[j, :lens[j]] = eps[j, :lens[j]] + (d_star if is_alt[j] else 0.0)
    logE = ep.log_e_path(Ymat, m_env=M_ENV)
    logE_frozen, tau = ep.freeze_at_crossing(logE, log_b)
    tau = np.where((tau >= 0) & (tau < lens), tau, -1)
    solo = tau >= 0
    e_log = np.zeros(J)
    for j in range(J):
        if lens[j] > 0:
            e_log[j] = logE_frozen[j, lens[j] - 1]
    tau_cal = np.where(solo, A + 1 + tau, A + lens)

    res = ebh.baseline_reveal(e_log, tau_cal, ~is_alt, ALPHA, J_BUDGET)
    B = res["B"]

    recs = []
    for j in range(J):
        recs.append({"lam_arr": lam, "run_id": run_idx, "A": int(A[j]),
                     "is_alt": bool(is_alt[j]), "solo": bool(solo[j]),
                     "tau_cal": int(tau_cal[j]), "B_j": int(B[j]),
                     "discovered": bool(res["disc"][j]),
                     "t_disc": int(res["disc_time"][j])})
    summary = {"lam_arr": lam, "run_id": run_idx, "n_arrived": n_arrived,
               "n_reg": J, "budget_exh_month": budget_exh_month,
               "n_disc": res["n_disc"], "n_false": res["n_false"],
               "sup_fdp": res["sup_fdp"]}
    return recs, summary


def main():
    t0 = time.time()
    all_recs, all_summ = [], []
    for lam in LAM_ARR:
        for r in range(N_RUN):
            recs, summ = run_one(lam, r)
            all_recs.extend(recs)
            all_summ.append(summ)
        print(f"  lam_arr={lam}: done ({time.time()-t0:.0f}s)", flush=True)

    det = pd.DataFrame(all_recs)
    summ = pd.DataFrame(all_summ)
    common.write_parquet(det, "E5_strategies.parquet")
    common.write_parquet(summ, "E5.parquet")

    # 3항 분해 (발견된 alternative만, 전 항 >= 0)
    d = det[det["is_alt"] & det["discovered"]].copy()
    d["t_freeze"] = d["tau_cal"] - d["A"]
    d["w_fifo"] = d["B_j"] - d["tau_cal"]
    d["t_unlock"] = d["t_disc"] - d["B_j"]
    assert (d[["t_freeze", "w_fifo", "t_unlock"]].to_numpy() >= 0).all(), \
        "분해 항에 음수 발생"
    print("\n발견 지연 3항 분해 (발견된 alt, 평균 [중앙값], 단위 월):")
    decomp = []
    for lam in LAM_ARR:
        s = d[d["lam_arr"] == lam]
        srun = summ[summ["lam_arr"] == lam]
        alt_all = det[(det["lam_arr"] == lam) & det["is_alt"]]
        row = {"lam_arr": lam, "n_disc_alt": len(s),
               "t_freeze_mean": s["t_freeze"].mean(), "t_freeze_med": s["t_freeze"].median(),
               "w_fifo_mean": s["w_fifo"].mean(), "w_fifo_med": s["w_fifo"].median(),
               "t_unlock_mean": s["t_unlock"].mean(), "t_unlock_med": s["t_unlock"].median(),
               "total_mean": (s["t_freeze"] + s["w_fifo"] + s["t_unlock"]).mean(),
               "alt_det_rate": float(alt_all["discovered"].mean()),
               "k2_only_share": float((~s["solo"]).mean()),
               "budget_exh_month_med": srun["budget_exh_month"].replace(-1, np.nan).median(),
               "sup_fdp_mean": srun["sup_fdp"].mean()}
        decomp.append(row)
        print(f"  λ={lam}: T_freeze={row['t_freeze_mean']:.1f} [{row['t_freeze_med']:.0f}]"
              f" + W_fifo={row['w_fifo_mean']:.1f} [{row['w_fifo_med']:.0f}]"
              f" + T_unlock={row['t_unlock_mean']:.1f} [{row['t_unlock_med']:.0f}]"
              f" = {row['total_mean']:.1f} | alt검출률={row['alt_det_rate']:.1%}"
              f" k≥2전용={row['k2_only_share']:.1%} supFDP={row['sup_fdp_mean']:.5f}")
    dec = pd.DataFrame(decomp)
    common.write_parquet(dec, "E5_decomp.parquet")
    print("\n" + dec.to_string(index=False))
    print(f"\nE5 완료 (총 {time.time()-t0:.0f}s, commit {common.commit_hash()})")

    # ---- 그림: 3항 분해 스택 바 (전 항 >= 0) ----
    okabe = ["#0072B2", "#E69F00", "#009E73"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(LAM_ARR))
    tf = dec["t_freeze_mean"].to_numpy()
    wf = dec["w_fifo_mean"].to_numpy()
    tu = dec["t_unlock_mean"].to_numpy()
    ax.bar(x, tf, 0.55, color=okabe[0], label="T_freeze (to solo/deadline freeze)")
    ax.bar(x, wf, 0.55, bottom=tf, color=okabe[1], label="W_fifo (reveal queue wait)")
    ax.bar(x, tu, 0.55, bottom=tf + wf, color=okabe[2],
           label="T_unlock (threshold relaxation wait)")
    for xi_, total in zip(x, tf + wf + tu):
        ax.annotate(f"{total:.0f}mo", (xi_, total + 2), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ_arr={l}/mo" for l in LAM_ARR])
    ax.set_ylabel("months since A_j (mean over discovered alternatives)")
    ax.set_ylim(0, (tf + wf + tu).max() * 1.4)   # 범례·주석 겹침 방지 여백
    ax.set_title(f"E5 — discovery delay decomposition, baseline reveal\n"
                 f"(immediate registration, J_budget={J_BUDGET}, N_RUN={N_RUN}/cell)",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(common.REPO / "sim_E5_timing.png", dpi=140)
    print("saved -> E5.parquet, E5_strategies.parquet, E5_decomp.parquet, sim_E5_timing.png")


if __name__ == "__main__":
    main()
