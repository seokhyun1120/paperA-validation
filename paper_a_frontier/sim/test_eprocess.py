"""회귀 테스트 — sim/eprocess가 frontier.py와 동일 입력에서 동일 출력 (허용오차 0).

1) frontier.frontier_for_n(n)의 crossing prob 벡터를, 동일 rng 상태에서 뽑은
   동일 잡음으로 sim.eprocess.log_e_path로 재현 → np.array_equal (bit-exact).
2) delta* 보간값 동일.
3) log_b_solo(1/212) == frontier.LOG_B.
4) freeze_at_crossing 불변식: 도달 후 값 상수, 미도달 경로는 원본과 동일.

실행: python3 -m sim.test_eprocess  (repo 루트에서)
"""
import numpy as np

from . import eprocess as ep

fr = ep.frontier


def _reproduce_probs(n):
    fr.rng = np.random.default_rng(fr.SEED)          # 모듈 rng 상태 재설정 (파일 무수정)
    d_ref, probs_ref = fr.frontier_for_n(n)
    fr.rng = np.random.default_rng(fr.SEED)
    eps = fr.draw_noise((fr.N_SIM, n))
    probs = np.empty(len(fr.DELTA_GRID))
    for i, d in enumerate(fr.DELTA_GRID):
        logE = ep.log_e_path(d + eps, m_env=fr.M_ENV)
        probs[i] = (logE.max(axis=1) >= fr.LOG_B).mean()
    # frontier_for_n과 동일한 delta* 보간 재현
    target = 1 - fr.BETA
    if probs[-1] < target:
        d_star = np.nan
    else:
        idx = int(np.argmax(probs >= target))
        if idx == 0:
            d_star = fr.DELTA_GRID[0]
        else:
            d0, d1 = fr.DELTA_GRID[idx - 1], fr.DELTA_GRID[idx]
            p0, p1 = probs[idx - 1], probs[idx]
            d_star = d0 + (target - p0) * (d1 - d0) / (p1 - p0)
    return d_ref, probs_ref, d_star, probs


def main():
    for n in (120, 360):
        d_ref, probs_ref, d_star, probs = _reproduce_probs(n)
        assert np.array_equal(probs, probs_ref), f"n={n}: crossing prob 불일치"
        assert (np.isnan(d_ref) and np.isnan(d_star)) or d_star == d_ref, \
            f"n={n}: delta* 불일치 {d_star} vs {d_ref}"
        print(f"  n={n}: probs bit-exact 일치 ({len(probs)}개 grid점), "
              f"delta*={d_star:.6f} == frontier ✓")

    assert ep.log_b_solo(1.0 / 212) == fr.LOG_B, "b_solo 불일치"
    print(f"  log_b_solo(1/212) = {ep.log_b_solo(1.0/212):.6f} == frontier.LOG_B ✓")

    # freeze 불변식
    rng = np.random.default_rng(123)
    logE = np.cumsum(rng.standard_normal((500, 200)), axis=1)
    log_b = 5.0
    frz, tau = ep.freeze_at_crossing(logE, log_b)
    for r in range(500):
        if tau[r] >= 0:
            assert (frz[r, tau[r]:] == logE[r, tau[r]]).all()
            assert (logE[r, :tau[r]] < log_b).all()
            assert logE[r, tau[r]] >= log_b
            assert (frz[r, :tau[r]] == logE[r, :tau[r]]).all()
        else:
            assert (frz[r] == logE[r]).all() and (logE[r] < log_b).all()
    print("  freeze_at_crossing 불변식 ✓")
    print("\n회귀 테스트 전체 통과 (허용오차 0)")


if __name__ == "__main__":
    main()
