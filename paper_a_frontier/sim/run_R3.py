"""R3 — boundary-near null stress (심사 M11 대응).

null 수익률에 세 가지 위반 결합:
  - GARCH(1,1) 확률변동성: sigma²_t = w + a·r²_{t-1} + b·sigma²_{t-1},
    a=0.10, b=0.85, w=0.05 (무조건부 분산 1)
  - AR(1) 자기상관: x_t = phi·x_{t-1} + sqrt(1-phi²)·z_t, phi=0.2 (단위 주변분산)
  - 공통요인: z = sqrt(rho)·F + sqrt(1-rho)·eta, rho=0.3
  r_t = sigma_t · x_t  →  E[r_t | F_{t-1}] = 0 (null 유지, 분포 가정만 위반).

envelope는 등록 규칙대로 추정: 등록 시점까지의 trailing 36개월 sd(v)와
v_min(등록 전 v의 하위 5% 분위수)으로 Y_t = r_t/max(v_{t-1}, v_min) 표준화,
m_env = 1.3 등록. 등록: in-sample 36개월 raw SR > 0.15 첫 돌파 (E1 honest의
기본 등록; 뮤테이션·포기 없음 — 분포 스트레스에 집중, RUN_LOG에 명기).
full protocol = 표준화 score의 Catoni e-process + baseline reveal.
J_budget=212, D=A+120, T=480, L=100, N_RUN=2,000. gaussian 혁신.
산출: empirical SupFDR + 95% CI (Wilson; 0건이면 Clopper-Pearson 상한 병기).
실행: MPLBACKEND=Agg python3 -m sim.run_R3
"""
import time

import numpy as np
import pandas as pd

from . import common, ebh, eprocess as ep, searcher

ALPHA = 0.05
J_BUDGET = 212
M_ENV = 1.3
T = 480
L = 100
W = 36
S_REG = 0.15
DL = 120
N_RUN = 2000
PHI_AR = 0.2
RHO = 0.3
GARCH_A, GARCH_B = 0.10, 0.85
GARCH_W = 1.0 - GARCH_A - GARCH_B          # 무조건부 분산 1
VMIN_Q = 0.05
MIN_OBS = 24


def gen_world(rng):
    """GARCH(1,1) + AR(1) + 공통요인 null 수익률 (L, T)."""
    eta = rng.standard_normal((L, T))
    F = rng.standard_normal((1, T))
    z = np.sqrt(RHO) * F + np.sqrt(1 - RHO) * eta
    x = np.empty((L, T))
    x[:, 0] = z[:, 0]
    for t in range(1, T):
        x[:, t] = PHI_AR * x[:, t - 1] + np.sqrt(1 - PHI_AR ** 2) * z[:, t]
    r = np.empty((L, T))
    sig2 = np.ones(L)
    for t in range(T):
        r[:, t] = np.sqrt(sig2) * x[:, t]
        sig2 = GARCH_W + GARCH_A * r[:, t] ** 2 + GARCH_B * sig2
    return r


def trailing_sd(r):
    """(L, T) → v[j, t] = r[j, t-36:t]의 sd (ddof=1, 최소 24) — t 시점 예측가능."""
    from numpy.lib.stride_tricks import sliding_window_view
    v = np.full(r.shape, np.nan)
    win = sliding_window_view(r, W, axis=1)          # (L, T-W+1, W)
    v[:, W:] = win.std(axis=2, ddof=1)[:, :-1]       # shift(1)
    # 24~35개월 구간: 부분 윈도우
    for t in range(MIN_OBS, W):
        v[:, t] = r[:, :t].std(axis=1, ddof=1)
    return v


def run_one(run_idx):
    rng = common.rng_for(common.EXP_ID.get("R3", 8), run_idx)
    log_b = ep.log_b_solo(1.0 / J_BUDGET)
    r = gen_world(rng)
    sr = searcher.rolling_sr(r, W)
    above = sr > S_REG
    has = above.any(axis=1)
    first = np.where(has, above.argmax(axis=1), -1)
    order = np.argsort(np.where(first < 0, T + 1, first))
    v = trailing_sd(r)

    A_list, Y_list, lens = [], [], []
    for j in order:
        if first[j] < 0 or len(A_list) >= J_BUDGET:
            continue
        A = int(first[j])
        v_pre = v[j, :A + 1]
        if np.isnan(v_pre).all():
            continue
        v_min = np.nanquantile(v_pre, VMIN_Q)          # 등록 전 정보만 (규칙 동일)
        ell = min(DL, T - 1 - A)
        if ell <= 0:
            continue
        vv = v[j, A + 1:A + 1 + ell]
        y = r[j, A + 1:A + 1 + ell] / np.maximum(vv, v_min)
        A_list.append(A)
        Y_list.append(y)
        lens.append(ell)

    J = len(A_list)
    if J == 0:
        return {"run_id": run_idx, "n_reg": 0, "sup_fdp": 0.0, "n_disc": 0,
                "max_log_e": 0.0}
    Ymat = np.zeros((J, DL))
    for i, y in enumerate(Y_list):
        Ymat[i, :lens[i]] = y
    lens = np.array(lens)
    A = np.array(A_list)
    logE = ep.log_e_path(Ymat, m_env=M_ENV)
    logE_frozen, tau = ep.freeze_at_crossing(logE, log_b)
    tau = np.where((tau >= 0) & (tau < lens), tau, -1)
    e_log = logE_frozen[np.arange(J), lens - 1]
    tau_cal = np.where(tau >= 0, A + 1 + tau, A + lens)
    # baseline reveal은 등록순 정렬 필요
    o = np.argsort(A, kind="stable")
    res = ebh.baseline_reveal(e_log[o], tau_cal[o], np.ones(J, bool), ALPHA, J_BUDGET)
    return {"run_id": run_idx, "n_reg": J, "sup_fdp": res["sup_fdp"],
            "n_disc": res["n_disc"], "max_log_e": float(e_log.max())}


def main(n_run=N_RUN):
    t0 = time.time()
    rows = [run_one(i) for i in range(n_run)]
    df = pd.DataFrame(rows)
    path = common.write_parquet(df, "R3.parquet")
    p = df["sup_fdp"].mean()
    n = len(df)
    # Wilson 95% CI
    z = 1.959963984540054
    den = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    print(f"R3 (GARCH+AR(1) phi={PHI_AR}+rho={RHO}, all-null, N_RUN={n_run}):")
    print(f"  n_reg 평균 = {df['n_reg'].mean():.1f}, 발견 총 {df['n_disc'].sum()}건, "
          f"max log e = {df['max_log_e'].max():.3f} (log b = {np.log(J_BUDGET/ALPHA):.3f})")
    print(f"  empirical SupFDR = {p:.5f},  Wilson 95% CI = "
          f"[{max(ctr-hw,0):.5f}, {ctr+hw:.5f}]")
    if df["n_disc"].sum() == 0:
        cp_hi = 1 - 0.025 ** (1 / n)
        print(f"  (발견 0건 — Clopper-Pearson 95% 상한 = {cp_hi:.5f})")
    print(f"  vs alpha + 3*SE(p=alpha) = {ALPHA + 3*np.sqrt(ALPHA*(1-ALPHA)/n):.4f}")
    print(f"완료 ({time.time()-t0:.0f}s, commit {common.commit_hash()})")
    print(f"saved -> {path.name}")
    return df


if __name__ == "__main__":
    main()
