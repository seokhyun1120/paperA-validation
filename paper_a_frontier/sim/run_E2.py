"""E2 — Power vs frontier (frontier의 설계도구 검증).

alternative delta를 frontier Catoni delta*의 {0.5x, 1.0x, 1.5x} 배수로 주입,
등록 직후부터 D개월 내 solo boundary(b=4240) 도달률과 time-to-detection 측정.
frontier의 beta=0.5 정의와 정합: 1.0x에서 검출률 ~= 50% 기대, |편차| > 10%p면
원인 조사 (STOP·보고).

셀: gaussian × D ∈ {60, 120, 240} × mult ∈ {0.5, 1.0, 1.5}
    + t5 × D=120 × mult   (t5 delta*는 gaussian과 <=0.001 차이 — Stage 3 —
    이므로 gaussian delta* 값을 공용; frontier.catoni의 정확값 사용)
CRN: eps는 (noise, D, batch)로만 시드 — mult 축은 동일 잡음 공유.
N_TRIAL = 10,000/셀. 실행: MPLBACKEND=Agg python3 -m sim.run_E2
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import common, eprocess as ep, world

ALPHA = 0.05
J_BUDGET = 212
M_ENV = 1.3
MULTS = [0.5, 1.0, 1.5]
N_TRIAL = 10_000
BATCH = 2_000
N_GRID = [60, 120, 240, 360]

CELLS = [("gaussian", D) for D in (60, 120, 240)] + [("t5", 120)]


def delta_star(D):
    """frontier가 계산한 Catoni 월간 delta* (gaussian, SEED=0) 정확값."""
    return float(ep.frontier.catoni[N_GRID.index(D)])


def main():
    t0 = time.time()
    log_b = ep.log_b_solo(1.0 / J_BUDGET)
    assert log_b == ep.frontier.LOG_B
    rows = []
    for noise, D in CELLS:
        nc = common.NOISE_CODE[noise]
        d_star = delta_star(D)
        for b in range(N_TRIAL // BATCH):
            rng = common.rng_for(common.EXP_ID["E2"], nc, D, b)   # CRN: mult 제외
            eps = world.draw_eps(rng, BATCH, D, noise=noise)
            for mult in MULTS:
                logE = ep.log_e_path(mult * d_star + eps, m_env=M_ENV)
                _, tau = ep.freeze_at_crossing(logE, log_b)
                for i in range(BATCH):
                    rows.append({"noise": noise, "D": D, "mult": mult,
                                 "run_id": b * BATCH + i,
                                 "delta": mult * d_star,
                                 "crossed": tau[i] >= 0,
                                 "ttd": int(tau[i]) + 1 if tau[i] >= 0 else -1})
        print(f"  {noise:>8} D={D:>3}: done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    path = common.write_parquet(df, "E2.parquet")

    g = df.groupby(["noise", "D", "mult"], sort=False)
    summ = g.agg(det_rate=("crossed", "mean"), n=("crossed", "size")).reset_index()
    ttd = df[df["crossed"]].groupby(["noise", "D", "mult"], sort=False)["ttd"] \
        .quantile([0.25, 0.5, 0.75]).unstack()
    summ = summ.merge(ttd, on=["noise", "D", "mult"], how="left")
    summ["mc_se"] = np.sqrt(summ["det_rate"] * (1 - summ["det_rate"]) / summ["n"])
    # 게이트: mult=1.0 검출률이 50% ± 10%p 이내
    at1 = summ[summ["mult"] == 1.0]
    summ["gate"] = ""
    summ.loc[summ["mult"] == 1.0, "gate"] = np.where(
        (at1["det_rate"] - 0.5).abs() <= 0.10, "PASS", "FAIL")
    print("\n" + summ.to_string(index=False))
    all_pass = (summ.loc[summ["mult"] == 1.0, "gate"] == "PASS").all()
    print(f"\nE2 {'PASS — 1.0x 셀 전부 50%±10%p 이내' if all_pass else '*** FAIL: 원인 조사 필요 ***'}"
          f"   (총 {time.time()-t0:.0f}s, commit {common.commit_hash()})")

    # ---- 그림: 검출률 vs 배수 ----
    okabe = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]   # 고정 순서
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for ci, (noise, D) in enumerate(CELLS):
        s = summ[(summ["noise"] == noise) & (summ["D"] == D)]
        ax.plot(s["mult"], s["det_rate"], "o-", lw=2, color=okabe[ci],
                label=f"{noise}, D={D}")
    ax.axhline(0.5, color="gray", lw=1.2, ls="--")
    ax.annotate("target 50% at 1.0x (β=0.5)", (0.52, 0.515), fontsize=8, color="gray")
    ax.axvline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("delta multiple of frontier δ*(D)")
    ax.set_ylabel("P(solo crossing within D)")
    ax.set_title(f"E2 — power vs frontier prediction (N={N_TRIAL}/cell)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(common.REPO / "sim_E2_power.png", dpi=140)
    print(f"saved -> {path.name}, sim_E2_power.png")
    return bool(all_pass)


if __name__ == "__main__":
    main()
