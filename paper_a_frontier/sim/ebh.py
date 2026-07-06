"""sim/ebh.py — online e-BH.

두 가지 reveal 방식:

1) baseline_reveal (논문 baseline, primary) — "동결 e-value의 등록순 reveal".
   각 전략 j의 동결값 e_j = E_{j,tau_j}, tau_j = solo 도달/데드라인/포기 중 최선착.
   reveal 시점 B_m = max_{i<=m} tau_i_cal (등록순 prefix가 전부 동결된 calendar 월;
   늦게 등록됐지만 먼저 동결된 전략은 FIFO 대기). 각 B_m에서 동결값 e_1..e_m에
   e-BH 적용 (proofs final §4):
     k_m = max{ k in {1..m} : #{ j<=m : gamma_j*e_j >= 1/(alpha*k) } >= k },
     k 없으면 k_m = 0 -> R_m = 공집합;
     R_m = { j<=m : gamma_j*e_j >= 1/(alpha*k_m) }.
   gamma_j = 1/J_budget 균등이므로 조건은 e_j >= J_budget/(alpha*k)와 동치.
   R_m은 m에 대해 nested (발견 철회 없음 — 코드에서 assert).
   FDP_m = (R_m 내 null 수)/(|R_m| ∨ 1), sup FDP = max_m FDP_m.

2) online_ebh (live 변형, secondary 보존) — 진행 중 e-process에 매월 e-BH.
   발견 집합 누적 union (FDP를 크게 잡는 보수적 방향).
"""
import numpy as np

from . import eprocess as ep


def _freeze_paths(regs, T, log_b):
    """등록 목록 -> e-process 동결 경로 내부 표현 (두 reveal 방식 공용).

    각 등록 j: post-A_j 유효 구간 [A+1, min(D_j, T-1, abandon_t)]에서 e-process,
    solo boundary 도달 시 freeze rule로 동결.
    """
    J = len(regs)
    DL = max((r.deadline - r.A for r in regs), default=1)
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
    if J == 0:
        return dict(logE_frozen=np.zeros((0, DL)), lens=lens, tau=np.zeros(0, int),
                    A=np.zeros(0, int), is_null=np.zeros(0, bool))
    logE = ep.log_e_path(Ymat, m_env=regs[0].m_env)
    logE_frozen, tau = ep.freeze_at_crossing(logE, log_b)
    tau = np.where((tau >= 0) & (tau < lens), tau, -1)   # 유효 구간 밖 도달은 미도달
    return dict(logE_frozen=logE_frozen, lens=lens, tau=tau,
                A=np.array([r.A for r in regs]),
                is_null=np.array([r.is_null for r in regs]))


def freeze_summary(regs, T, log_b):
    """baseline reveal 입력: 동결 log e-값, 동결 calendar 월, solo 도달 여부.

    tau_cal_j = A+1+tau (solo 도달) 또는 A+lens (데드라인/포기/표본 끝 — 관측
    마지막 월에 동결). lens=0이면 e=1로 A에 즉시 동결.
    """
    fp = _freeze_paths(regs, T, log_b)
    J = len(regs)
    e_log = np.zeros(J)
    for j in range(J):
        if fp["lens"][j] > 0:
            e_log[j] = fp["logE_frozen"][j, fp["lens"][j] - 1]
    solo = fp["tau"] >= 0
    tau_cal = np.where(solo, fp["A"] + 1 + fp["tau"], fp["A"] + fp["lens"])
    return e_log, tau_cal, solo, fp["A"], fp["is_null"]


def baseline_reveal(e_log, tau_cal, is_null, alpha, J_budget):
    """동결 e-값의 등록순 reveal + e-BH (proofs final §4의 k_m).

    입력은 등록순으로 정렬돼 있어야 한다. 반환: sup_fdp, 발견 플래그/시점(B_{M_disc}),
    reveal 시점 B (등록순), FDP 경로 (reveal step m 인덱스).
    """
    J = len(e_log)
    out = {"sup_fdp": 0.0, "n_disc": 0, "n_false": 0,
           "disc": np.zeros(J, bool), "disc_time": np.full(J, -1),
           "disc_step": np.full(J, -1), "B": np.zeros(J, int),
           "fdp_path": np.zeros(J)}
    if J == 0:
        return out
    B = np.maximum.accumulate(tau_cal)
    log_thr = np.log(J_budget / (alpha * np.arange(1, J + 1)))
    disc = np.zeros(J, bool)
    disc_time = np.full(J, -1)
    disc_step = np.full(J, -1)
    sup_fdp = 0.0
    fdp_path = np.zeros(J)
    for m in range(1, J + 1):
        prefix = e_log[:m]
        s = np.sort(prefix)[::-1]
        ok = s >= log_thr[:m]
        if ok.any():
            k_m = int(np.arange(1, m + 1)[ok].max())
            in_R = prefix >= log_thr[k_m - 1]
            # nested 확인: 발견 철회가 없어야 함
            assert not (disc[:m] & ~in_R).any(), "R_m nested 위반"
            new = in_R & ~disc[:m]
            idx = np.nonzero(new)[0]
            disc[idx] = True
            disc_time[idx] = B[m - 1]
            disc_step[idx] = m
            fdp = float((in_R & is_null[:m]).sum() / max(in_R.sum(), 1))
        else:
            fdp = 0.0
        fdp_path[m - 1] = fdp
        sup_fdp = max(sup_fdp, fdp)
    out.update(sup_fdp=sup_fdp, n_disc=int(disc.sum()),
               n_false=int((disc & is_null).sum()), disc=disc,
               disc_time=disc_time, disc_step=disc_step, B=B, fdp_path=fdp_path)
    return out


def build_calendar_logE(regs, T, log_b):
    """(live 변형용) 등록 목록 -> (T, J) calendar log e-값 행렬 + 메타.

    등록 전은 e=1 (log 0), 유효 구간 종료 후 마지막 값 유지.
    """
    J = len(regs)
    if J == 0:
        return np.zeros((T, 0)), np.zeros(0, bool), np.full(0, -1), np.zeros(0, int)
    fp = _freeze_paths(regs, T, log_b)
    E_cal = np.zeros((T, J))
    tau_cal = np.full(J, -1)
    for j, r in enumerate(regs):
        ell = fp["lens"][j]
        if ell > 0:
            E_cal[r.A + 1:r.A + 1 + ell, j] = fp["logE_frozen"][j, :ell]
            E_cal[r.A + 1 + ell:, j] = fp["logE_frozen"][j, ell - 1]
        if fp["tau"][j] >= 0:
            tau_cal[j] = r.A + 1 + fp["tau"][j]
    return E_cal, fp["is_null"], tau_cal, fp["A"]


def online_ebh(E_cal, is_null, alpha, J_budget):
    """(live 변형, secondary) E_cal: (T, J) log e-값. 매월 e-BH -> 누적 발견."""
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
