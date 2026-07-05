"""E3 — Alpha decay (§5.7 hard-wall 경계 실측).

delta_{A+s} = delta0 * rho_d^(s-1), 누적 edge = delta0/(1-rho_d).
누적 edge를 기준벽 W0 = delta*(120) * 120 (frontier 정확값)의 c배로 설정:
  c ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0},  rho_d ∈ {0.99, 0.98, 0.95}.
측정: D=120 내 solo 도달률 — "탐지 가능 ⟺ 누적 edge > 문턱" 경계의 위치와
날카로움(sharpness)을 rho_d별로 실측, 50% 지점(경험적 벽)을 보간으로 보고.

CRN: eps 시드는 batch 인덱스만 사용 — 모든 (rho_d, c) 셀이 동일 잡음 공유.
N_TRIAL = 10,000/셀, gaussian. 실행: MPLBACKEND=Agg python3 -m sim.run_E3
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import common, eprocess as ep, world

ALPHA = 0.05
J_BUDGET = 212
M_ENV = 1.3
D = 120
N_TRIAL = 10_000
BATCH = 2_000
RHO_D = [0.99, 0.98, 0.95]
C_MULT = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def main():
    t0 = time.time()
    d_star_120 = float(ep.frontier.catoni[1])          # delta*(120) 정확값
    W0 = d_star_120 * 120                              # 기준벽 (상수 delta 총 edge)
    log_b = ep.log_b_solo(1.0 / J_BUDGET)
    print(f"기준벽 W0 = delta*(120)*120 = {W0:.3f}")

    rows = []
    for b in range(N_TRIAL // BATCH):
        rng = common.rng_for(common.EXP_ID["E3"], b)   # CRN: 전 셀 공유
        eps = world.draw_eps(rng, BATCH, D, noise="gaussian")
        for rho_d in RHO_D:
            for c in C_MULT:
                cumedge = c * W0
                delta0 = cumedge * (1 - rho_d)
                dpath = world.decay_delta(delta0, rho_d, D)
                logE = ep.log_e_path(dpath[None, :] + eps, m_env=M_ENV)
                _, tau = ep.freeze_at_crossing(logE, log_b)
                for i in range(BATCH):
                    rows.append({"rho_d": rho_d, "c": c, "cumedge": cumedge,
                                 "delta0": delta0, "run_id": b * BATCH + i,
                                 "crossed": tau[i] >= 0,
                                 "ttd": int(tau[i]) + 1 if tau[i] >= 0 else -1})
        print(f"  batch {b+1}/{N_TRIAL//BATCH} done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    path = common.write_parquet(df, "E3.parquet")

    g = df.groupby(["rho_d", "c"], sort=False)
    summ = g.agg(det_rate=("crossed", "mean"), cumedge=("cumedge", "first"),
                 delta0=("delta0", "first"), n=("crossed", "size")).reset_index()
    med_ttd = df[df["crossed"]].groupby(["rho_d", "c"], sort=False)["ttd"].median() \
        .rename("ttd_med").reset_index()
    summ = summ.merge(med_ttd, on=["rho_d", "c"], how="left")
    print("\n" + summ.to_string(index=False))

    # 경험적 벽: 검출률 50% 지점을 cumedge에서 선형 보간
    print("\n경험적 벽 (검출률 50% 지점, cumedge 단위):")
    walls = {}
    for rho_d in RHO_D:
        s = summ[summ["rho_d"] == rho_d].sort_values("cumedge")
        r, x = s["det_rate"].to_numpy(), s["cumedge"].to_numpy()
        if (r >= 0.5).any() and (r < 0.5).any():
            i = int(np.argmax(r >= 0.5))
            wall = x[i - 1] + (0.5 - r[i - 1]) * (x[i] - x[i - 1]) / (r[i] - r[i - 1])
        else:
            wall = np.nan
        walls[rho_d] = wall
        print(f"  rho_d={rho_d}: wall ≈ {wall:.1f} (= {wall/W0:.2f} × W0)")
    print(f"\nE3 완료 (총 {time.time()-t0:.0f}s, commit {common.commit_hash()})")

    # ---- 그림: 검출률 vs 누적 edge ----
    okabe = ["#0072B2", "#E69F00", "#009E73"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for ri, rho_d in enumerate(RHO_D):
        s = summ[summ["rho_d"] == rho_d].sort_values("cumedge")
        ax.plot(s["cumedge"], s["det_rate"], "o-", lw=2, color=okabe[ri],
                label=f"ρ_d={rho_d} (n_eff={1/(1-rho_d):.0f}mo)")
    ax.axhline(0.5, color="gray", lw=1.2, ls="--")
    ax.axvline(W0, color="#D55E00", lw=1.2, ls=":",
               label=f"W0 = δ*(120)·120 = {W0:.1f}")
    ax.set_xlabel("cumulative edge δ₀/(1−ρ_d)  (monthly-Sharpe months)")
    ax.set_ylabel(f"P(solo crossing within D={D})")
    ax.set_title(f"E3 — alpha decay hard wall (N={N_TRIAL}/cell, gaussian)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(common.REPO / "sim_E3_wall.png", dpi=140)
    print(f"saved -> {path.name}, sim_E3_wall.png")
    return walls


if __name__ == "__main__":
    main()
