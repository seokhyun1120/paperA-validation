"""R4 — boundary-near valid-null stress (3차 M6 대응).

전 전략 동시 등록(A=0), all-null, gamma=1/J. 점수는
Y_t = sqrt(0.95) * eps_t, eps_t ~ N(0, 1)로 생성한다. 따라서 조건부 평균은 0이고
mean(Y^2)=0.95 <= m_env^2=1.05^2=1.1025이므로 envelope가 유효한 null이다.

J=5를 primary로 N_RUN=20,000회 실행한다. 발견 런이 0이면 지정된 fallback으로
J=2를 한 번 더 실행한다. e-process 경로는 BATCH=500 런 단위로 벡터화하여 계산한다.
실행: MPLBACKEND=Agg python3 -m sim.run_R4
"""
import time

import numpy as np
import pandas as pd

from . import common, ebh, eprocess as ep, world

ALPHA = 0.5
J_PRIMARY = 5
J_FALLBACK = 2
T = 600
M_ENV = 1.05
NOISE_VAR = 0.95
NOISE_SCALE = float(np.sqrt(NOISE_VAR))
N_RUN = 20000
BATCH = 500


def wilson_ci(p: float, n: int) -> tuple[float, float]:
    """R3와 동일한 Wilson 95% 신뢰구간."""
    z = 1.959963984540054
    den = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return max(ctr - hw, 0.0), ctr + hw


def run_cell(J: int, n_run: int = N_RUN) -> tuple[pd.DataFrame, dict[str, float]]:
    """한 J 셀을 배치별로 실행하고 런별 결과와 집계를 반환."""
    t0 = time.time()
    log_b = ep.log_b_solo(1.0 / J, alpha=ALPHA)
    rows = []
    is_null = np.ones(J, bool)
    n_batch = int(np.ceil(n_run / BATCH))

    print(f"\nR4 J={J}: gamma={1.0/J:.3f}, b_solo={np.exp(log_b):.3f}, "
          f"log_b={log_b:.6f}, N_RUN={n_run}", flush=True)
    for batch_idx in range(n_batch):
        start = batch_idx * BATCH
        size = min(BATCH, n_run - start)
        rng = common.rng_for(common.EXP_ID["R4"], J, batch_idx)
        eps = world.std_noise(rng, (size * J, T), noise="gaussian")
        Y = NOISE_SCALE * eps
        mean_y2 = (Y ** 2).mean(axis=1).reshape(size, J).mean(axis=1)

        logE = ep.log_e_path(Y, m_env=M_ENV)
        max_logE = logE.max(axis=1).reshape(size, J).max(axis=1)
        logE_frozen, tau = ep.freeze_at_crossing(logE, log_b)
        tau = tau.reshape(size, J)
        solo_any = (tau >= 0).any(axis=1)
        e_log = logE_frozen[:, T - 1].reshape(size, J)
        tau_cal = np.where(tau >= 0, tau + 1, T)

        for b in range(size):
            res = ebh.baseline_reveal(e_log[b], tau_cal[b], is_null, ALPHA, J_budget=J)
            n_disc = int(res["n_disc"])
            rows.append({"J": J, "batch": batch_idx, "run_id": start + b,
                         "n_disc": n_disc, "fdp": float(n_disc > 0),
                         "max_logE": float(max_logE[b]),
                         "solo_any": bool(solo_any[b]),
                         "near_boundary": bool(max_logE[b] >= log_b - 1.0),
                         "mean_y2": float(mean_y2[b])})
        if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == n_batch:
            print(f"  batch {batch_idx + 1:2d}/{n_batch} 완료 "
                  f"({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    summary = summarize(df, J, log_b)
    return df, summary


def summarize(df: pd.DataFrame, J: int, log_b: float) -> dict[str, float]:
    """stdout 보고용 집계."""
    n = len(df)
    p = float(df["fdp"].mean())
    ci_lo, ci_hi = wilson_ci(p, n)
    q50, q90, q99 = df["max_logE"].quantile([0.50, 0.90, 0.99]).to_numpy()
    out = {"J": float(J), "n": float(n), "supfdr": p, "ci_lo": ci_lo,
           "ci_hi": ci_hi, "disc_runs": float(df["fdp"].sum()),
           "n_disc_sum": float(df["n_disc"].sum()),
           "solo_any_rate": float(df["solo_any"].mean()),
           "max_logE_mean": float(df["max_logE"].mean()),
           "max_logE_q50": float(q50), "max_logE_q90": float(q90),
           "max_logE_q99": float(q99), "max_logE_max": float(df["max_logE"].max()),
           "near_boundary_rate": float(df["near_boundary"].mean()),
           "mean_y2": float(df["mean_y2"].mean())}
    print(f"R4 J={J} 집계:")
    print(f"  유효 null 확인: E[Y|과거]=0, mean(Y^2)={out['mean_y2']:.6f} "
          f"<= m_env^2={M_ENV ** 2:.6f}")
    print(f"  empirical SupFDR = {p:.6f}, Wilson 95% CI = "
          f"[{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"  발견 런 수 = {int(out['disc_runs'])}/{n}, n_disc 합 = "
          f"{int(out['n_disc_sum'])}")
    print(f"  solo crossing 런 비율 = {out['solo_any_rate']:.6f}")
    print(f"  max_t logE 런별 최댓값: mean={out['max_logE_mean']:.3f}, "
          f"q50={q50:.3f}, q90={q90:.3f}, q99={q99:.3f}, "
          f"max={out['max_logE_max']:.3f}")
    print(f"  경계 근접 P(max logE >= log_b - 1) = "
          f"{out['near_boundary_rate']:.6f} (log_b={log_b:.3f})")
    return out


def main(n_run: int = N_RUN) -> pd.DataFrame:
    t0 = time.time()
    print("R4 boundary-near valid-null stress")
    print(f"  Y=sqrt({NOISE_VAR})*eps, eps~N(0,1), mean(Y^2)={NOISE_VAR} "
          f"<= m_env^2={M_ENV ** 2:.4f}; 조건부 평균 0으로 유효 null 유지")
    frames = []
    df5, summ5 = run_cell(J_PRIMARY, n_run=n_run)
    frames.append(df5)
    if int(summ5["disc_runs"]) == 0:
        print("\nJ=5 발견 런이 0이므로 지정 fallback J=2를 실행합니다.", flush=True)
        df2, _ = run_cell(J_FALLBACK, n_run=n_run)
        frames.append(df2)
    else:
        print("\nJ=5에서 발견 런이 있어 J=2 fallback은 실행하지 않습니다.", flush=True)

    df = pd.concat(frames, ignore_index=True)
    path = common.write_parquet(df, "R4.parquet")
    print(f"\nR4 완료 (총 {time.time() - t0:.0f}s, commit {common.commit_hash()})")
    print(f"saved -> {path.name}")
    return df


if __name__ == "__main__":
    main()
