"""E5 — Queue/timing (§4.1 3항 분해 재현).

등록 도착 과정: 포아송 강도 lam_arr/월, 원장 수용 능력 1건/월 (FIFO 큐 — 대기 발생),
도착 전략의 50%는 alternative (delta = delta*(120) 상수), 50%는 null.
J_budget=212, D_j = A_j + 120, T=360, gaussian, N_RUN=500/셀.

발견 지연 분해 (발견된 alternative별):
  W_queue = A_j - arrival           (등록 대기)
  T_grow  = tau_solo - A_j          (e-process 성장; solo 도달 전 k>=2로 발견되면
                                     T_grow = t_disc - A_j, order 항은 결측 처리)
  T_order = t_disc - tau_solo <= 0  (online e-BH의 다중성 이득 — reveal 순서 효과)
측정 부가: 동시 활성 전략 수 경로(평균/최대), gamma 예산 소진 시점.
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
R_MAX = 1              # 월당 등록 수용 능력
LAM_ARR = [0.5, 1.0, 2.0]


def run_one(lam, run_idx):
    rng = common.rng_for(common.EXP_ID["E5"], int(lam * 10), run_idx)
    log_b = ep.log_b_solo(1.0 / J_BUDGET)
    d_star = float(ep.frontier.catoni[1])              # delta*(120)

    # 도착·큐·등록 (FIFO, 월당 R_MAX건, 예산 J_BUDGET)
    counts = rng.poisson(lam, T)
    arrivals = np.repeat(np.arange(T), counts)         # 도착 월 (시간순)
    queue, A_list, arr_list = [], [], []
    ai = 0
    budget_exh_month = -1
    for t in range(T):
        while ai < len(arrivals) and arrivals[ai] == t:
            queue.append(t)
            ai += 1
        for _ in range(R_MAX):
            if not queue or len(A_list) >= J_BUDGET:
                break
            arr_list.append(queue.pop(0))
            A_list.append(t)
        if len(A_list) >= J_BUDGET and budget_exh_month < 0:
            budget_exh_month = t
    J = len(A_list)
    A = np.array(A_list)
    arr = np.array(arr_list)
    is_alt = rng.random(J) < P_ALT

    # post-A score (관측 가능 길이만), delta 주입
    lens = np.minimum(DL, T - 1 - A)
    Ymat = np.zeros((J, DL))
    eps = world.draw_eps(rng, J, DL, noise="gaussian")
    for j in range(J):
        Ymat[j, :lens[j]] = eps[j, :lens[j]] + (d_star if is_alt[j] else 0.0)
    logE = ep.log_e_path(Ymat, m_env=M_ENV)
    logE_frozen, tau = ep.freeze_at_crossing(logE, log_b)
    tau = np.where((tau >= 0) & (tau < lens), tau, -1)

    E_cal = np.zeros((T, J))
    for j in range(J):
        ell = lens[j]
        if ell > 0:
            E_cal[A[j] + 1:A[j] + 1 + ell, j] = logE_frozen[j, :ell]
            E_cal[A[j] + 1 + ell:, j] = logE_frozen[j, ell - 1]
    res = ebh.online_ebh(E_cal, ~is_alt, ALPHA, J_BUDGET)

    tau_cal = np.where(tau >= 0, A + 1 + tau, -1)
    t_disc = res["disc_time"]
    # 동시 활성: 등록됨 & 데드라인 전 & 아직 발견 안 됨
    tgrid = np.arange(T)[:, None]
    end_active = np.where(t_disc >= 0, np.minimum(t_disc, A + lens), A + lens)
    active = ((tgrid > A[None, :]) & (tgrid <= end_active[None, :])).sum(axis=1)

    recs = []
    for j in range(J):
        recs.append({"lam_arr": lam, "run_id": run_idx, "arrival": int(arr[j]),
                     "A": int(A[j]), "is_alt": bool(is_alt[j]),
                     "w_queue": int(A[j] - arr[j]),
                     "discovered": t_disc[j] >= 0,
                     "t_disc": int(t_disc[j]), "tau_solo": int(tau_cal[j])})
    summary = {"lam_arr": lam, "run_id": run_idx, "n_arrived": len(arrivals),
               "n_reg": J, "budget_exh_month": budget_exh_month,
               "n_disc": res["n_disc"], "n_false": res["n_false"],
               "sup_fdp": res["sup_fdp"], "active_mean": float(active.mean()),
               "active_max": int(active.max())}
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
    path = common.write_parquet(summ, "E5.parquet")

    # 3항 분해 (발견된 alternative만)
    d = det[det["is_alt"] & det["discovered"]].copy()
    d["t_grow"] = np.where(d["tau_solo"] >= 0, d["tau_solo"] - d["A"],
                           d["t_disc"] - d["A"])
    d["t_order"] = np.where(d["tau_solo"] >= 0, d["t_disc"] - d["tau_solo"], np.nan)
    print("\n발견 지연 3항 분해 (발견된 alt, 평균 [중앙값]):")
    decomp = []
    for lam in LAM_ARR:
        s = d[d["lam_arr"] == lam]
        srun = summ[summ["lam_arr"] == lam]
        row = {"lam_arr": lam, "n_disc_alt": len(s),
               "w_queue_mean": s["w_queue"].mean(), "w_queue_med": s["w_queue"].median(),
               "t_grow_mean": s["t_grow"].mean(), "t_grow_med": s["t_grow"].median(),
               "t_order_mean": s["t_order"].mean(),
               "ebh_early_share": float((s["t_order"] < 0).mean()),
               "k2_only_share": float((s["tau_solo"] < 0).mean()),
               "budget_exh_month_med": srun["budget_exh_month"].replace(-1, np.nan).median(),
               "active_mean": srun["active_mean"].mean(),
               "sup_fdp_mean": srun["sup_fdp"].mean(),
               "alt_det_rate": float(det[(det["lam_arr"] == lam) & det["is_alt"]]
                                     .groupby(lambda _: 0)["discovered"].mean().iloc[0])}
        decomp.append(row)
        print(f"  λ={lam}: W_queue={row['w_queue_mean']:.1f} [{row['w_queue_med']:.0f}] "
              f"+ T_grow={row['t_grow_mean']:.1f} [{row['t_grow_med']:.0f}] "
              f"+ T_order={row['t_order_mean']:.2f} "
              f"(e-BH 조기발견 비율={row['ebh_early_share']:.1%}, "
              f"k≥2 전용={row['k2_only_share']:.1%})")
    dec = pd.DataFrame(decomp)
    common.write_parquet(dec, "E5_decomp.parquet")
    print("\n" + dec.to_string(index=False))
    print(f"\nE5 완료 (총 {time.time()-t0:.0f}s, commit {common.commit_hash()})")

    # ---- 그림: 지연 분해 스택 바 ----
    okabe = ["#0072B2", "#E69F00", "#009E73"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(LAM_ARR))
    wq = dec["w_queue_mean"].to_numpy()
    tg = dec["t_grow_mean"].to_numpy()
    to = dec["t_order_mean"].to_numpy()
    ax.bar(x, wq, 0.55, color=okabe[0], label="W_queue (registration wait)")
    ax.bar(x, tg, 0.55, bottom=wq, color=okabe[1], label="T_grow (e-process growth)")
    ax.bar(x, to, 0.55, bottom=wq + tg, color=okabe[2],
           label="T_order (e-BH multiplicity gain, ≤0)")
    for xi_, total in zip(x, wq + tg + to):
        ax.annotate(f"net {total:.0f}mo", (xi_, total + 2), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ_arr={l}/mo" for l in LAM_ARR])
    ax.set_ylabel("months (mean over discovered alternatives)")
    ax.set_ylim(0, (wq + tg).max() * 1.12)
    ax.set_title(f"E5 — discovery delay decomposition\n"
                 f"(capacity {R_MAX}/mo, J_budget={J_BUDGET}, N_RUN={N_RUN}/cell)",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(common.REPO / "sim_E5_timing.png", dpi=140)
    print("saved -> E5.parquet, E5_strategies.parquet, E5_decomp.parquet, sim_E5_timing.png")


if __name__ == "__main__":
    main()
