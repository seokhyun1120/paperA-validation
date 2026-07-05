"""sim/ebh.py — online e-BH.

매월(등록순 reveal) 현재 e-값 집합에 e-BH 적용:
  k_t = max{ k : e_(k) >= 1/(alpha * gamma * k) } = max{ k : e_(k) >= J_budget/(alpha*k) }
  (gamma = 1/J_budget 균등).  k_t = 0 -> R_t = 공집합.
발견 집합은 누적 (R̄_t = ∪_{s<=t} R_s) — 한번 발견되면 유지. e-값은 solo boundary
도달/데드라인/포기 시점에 동결 (freeze rule, sim/eprocess.freeze_at_crossing).
주의: proofs final §4의 k_m 정의 원문이 없어 표준 e-BH + 누적 발견으로 구현
(누적 union은 FDP를 크게 잡는 보수적 방향 — validity 측정에 안전).
기록: FDP 경로, sup FDP, TDR, time-to-detection.
"""
import numpy as np

from . import eprocess as ep


def build_calendar_logE(regs, T, log_b):
    """등록 목록 -> (T, J) calendar log e-값 행렬 + 메타.

    각 등록 j: post-A_j 유효 구간 [A+1, min(D_j, T-1, abandon_t)]에서 e-process,
    solo 도달 시 동결, 유효 구간 종료 후 마지막 값 유지. 등록 전은 e=1 (log 0).
    """
    J = len(regs)
    if J == 0:
        return np.zeros((T, 0)), np.zeros(0, bool), np.full(0, -1), np.zeros(0, int)
    DL = max(r.deadline - r.A for r in regs)
    lens = np.zeros(J, int)
    Ymat = np.zeros((J, DL))
    for j, r in enumerate(regs):
        end = min(r.deadline, T - 1)
        if r.abandon_t is not None:
            end = min(end, r.abandon_t)
        ell = max(0, end - r.A)
        lens[j] = ell
        if ell > 0:
            Ymat[j, :ell] = r.series[r.A + 1:r.A + 1 + ell]
    logE = ep.log_e_path(Ymat, m_env=regs[0].m_env)
    logE_frozen, tau = ep.freeze_at_crossing(logE, log_b)
    # tau가 유효 구간 밖이면 미도달 처리 (패딩 구간은 감소 경로라 실제로는 발생 안 함)
    tau = np.where((tau >= 0) & (tau < lens), tau, -1)

    E_cal = np.zeros((T, J))
    is_null = np.zeros(J, bool)
    tau_cal = np.full(J, -1)          # solo 도달 calendar 월
    for j, r in enumerate(regs):
        ell = lens[j]
        is_null[j] = r.is_null
        if ell > 0:
            E_cal[r.A + 1:r.A + 1 + ell, j] = logE_frozen[j, :ell]
            E_cal[r.A + 1 + ell:, j] = logE_frozen[j, ell - 1]   # 종료 후 동결
        if tau[j] >= 0:
            tau_cal[j] = r.A + 1 + tau[j]
    A_arr = np.array([r.A for r in regs])
    return E_cal, is_null, tau_cal, A_arr


def online_ebh(E_cal, is_null, alpha, J_budget):
    """E_cal: (T, J) log e-값. 매월 e-BH -> 누적 발견, FDP 경로, sup FDP 등."""
    T, J = E_cal.shape
    out = {"sup_fdp": 0.0, "n_disc": 0, "n_false": 0,
           "disc_time": np.full(J, -1), "fdp_path": np.zeros(T)}
    if J == 0:
        return out
    log_thr = np.log(J_budget / (alpha * np.arange(1, J + 1)))
    order = np.argsort(-E_cal, axis=1)
    S = np.take_along_axis(E_cal, order, axis=1)
    cond = S >= log_thr[None, :]
    k_t = np.where(cond, np.arange(1, J + 1)[None, :], 0).max(axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(J), (T, J)).copy(), axis=1)
    member = ranks < k_t[:, None]
    disc = np.maximum.accumulate(member, axis=0)       # 누적 발견 (monotone)
    n_disc_t = disc.sum(axis=1)
    n_false_t = (disc & is_null[None, :]).sum(axis=1)
    fdp = n_false_t / np.maximum(n_disc_t, 1)
    first = np.where(disc.any(axis=0), disc.argmax(axis=0), -1)
    out.update(sup_fdp=float(fdp.max()), n_disc=int(n_disc_t[-1]),
               n_false=int(n_false_t[-1]), disc_time=first, fdp_path=fdp)
    return out
