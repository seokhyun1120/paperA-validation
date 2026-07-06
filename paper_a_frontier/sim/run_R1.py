"""R1 — validity mechanism check (심사 M2 대응).

all-null, 기존 E1 searcher 재사용. J_budget=20, gamma=1/20,
alpha ∈ {0.2, 0.5} → b_solo = 1/(alpha*gamma) = 100, 40. D=120, gaussian,
rho ∈ {0, 0.3}, behavior ∈ {honest, adversarial} — 8셀, N_RUN=2000/셀.
목적: 발견이 실제로 발생하는 영역에서 empirical SupFDR <= alpha + 3*SE 확인.
발견 0인 셀이 있어도 alpha를 더 올려 재실행하지 않고 그대로 보고.

CRN: 세계 잡음 seed는 (rho, run)만 사용 — alpha/behavior 축과 무관하게 동일
세계 공유 (등록 집합은 alpha와 무관, behavior에만 의존).
s_reg = 0.15 (약한 selection, E1의 weak 셀) 고정.
실행: MPLBACKEND=Agg python3 -m sim.run_R1
"""
import itertools
import time

import numpy as np
import pandas as pd

from . import common, ebh, searcher
from .registry import Registry

J_BUDGET = 20
T = 480
L = 100
W = 36
S_REG = 0.15
DEADLINE_LEN = 120
N_RUN = 2000
M_ENV = 1.3

CELLS = list(itertools.product([0.2, 0.5], [0.0, 0.3], ["honest", "adversarial"]))


def run_one(alpha, rho, behavior, run_idx):
    bc = common.BEHAVIOR_CODE[behavior]
    rng_world = common.rng_for(common.EXP_ID["R1"], 0, int(rho * 10), run_idx)
    rng_mut = common.rng_for(common.EXP_ID["R1"], 1, int(rho * 10), bc, run_idx)
    reg = Registry(J_BUDGET, alpha, M_ENV, DEADLINE_LEN)
    searcher.run_search(rng_world, rng_mut, reg, T=T, L=L, W=W, s_reg=S_REG,
                        behavior=behavior, noise="gaussian", rho=rho)
    regs = reg.regs
    e_log, tau_cal, solo, _, is_null = ebh.freeze_summary(regs, T, reg.log_b)
    res = ebh.baseline_reveal(e_log, tau_cal, is_null, alpha, J_BUDGET)
    return {"run_id": run_idx, "alpha": alpha, "rho": rho, "behavior": behavior,
            "n_reg": len(regs), "sup_fdp": res["sup_fdp"],
            "n_disc": res["n_disc"], "n_false": res["n_false"],
            "n_solo": int(solo.sum())}


def main(n_run=N_RUN):
    t0 = time.time()
    rows = []
    for ci, (alpha, rho, behavior) in enumerate(CELLS):
        tc = time.time()
        for r in range(n_run):
            rows.append(run_one(alpha, rho, behavior, r))
        sub = pd.DataFrame(rows[-n_run:])
        print(f"[{ci+1}/8] alpha={alpha} rho={rho} {behavior:>11}: "
              f"SupFDR={sub['sup_fdp'].mean():.4f} 거짓발견합={sub['n_false'].sum()} "
              f"({time.time()-tc:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    path = common.write_parquet(df, "R1.parquet")

    g = df.groupby(["alpha", "rho", "behavior"], sort=False)
    summ = g.agg(supfdr=("sup_fdp", "mean"), n=("sup_fdp", "size"),
                 disc_rate_run=("n_disc", lambda x: (x > 0).mean()),
                 n_false_total=("n_false", "sum"),
                 n_false_mean=("n_false", "mean"),
                 n_reg=("n_reg", "mean"), n_solo_mean=("n_solo", "mean")).reset_index()
    summ["se_ref"] = np.sqrt(summ["alpha"] * (1 - summ["alpha"]) / summ["n"])
    summ["threshold"] = summ["alpha"] + 3 * summ["se_ref"]
    summ["pass"] = summ["supfdr"] <= summ["threshold"]
    print("\n" + summ.to_string(index=False))
    all_pass = bool(summ["pass"].all())
    zero_cells = int((summ["n_false_total"] == 0).sum())
    print(f"\nR1 {'PASS — 전체 셀 SupFDR <= alpha + 3*SE' if all_pass else '*** FAIL: STOP·보고 ***'}"
          f"  (발견 0 셀 {zero_cells}개는 그대로 보고)"
          f"  (총 {time.time()-t0:.0f}s, commit {common.commit_hash()})")
    print(f"saved -> {path.name}")
    return summ


if __name__ == "__main__":
    main()
