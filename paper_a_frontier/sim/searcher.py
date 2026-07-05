"""sim/searcher.py — 적응 탐색자.

라이브러리 L개 후보를 in-sample 윈도우(직전 W개월) Sharpe로 스캔, SR > s_reg면
등록 (A_j = 데이터 의존 stopping time). 핵심: null 후보도 in-sample 운으로
등록됨 — selection bias 내장.

행동 프로파일:
  honest      — 후보별 최초 임계 돌파 시 등록. 등록 +24개월 인터림 1회:
                post 평균 <= 0이면 포기, post SR > 0.1이면 winner mutation 1개.
  adversarial — 최초 돌파 등록 + 등록 즉시 고상관 클론(kappa=0.9) 추가 등록
                ("몰아넣기"), 부모 in-sample SR 신고점마다 뮤턴트(kappa=0.5,
                부모당 최대 2) 추가, +6개월부터 매월 peeking해 누적 post 평균 < 0이면
                즉시 포기 (공격적 optional stopping).

두 행동 모두 인과적(과거 정보만 사용) — E1의 validity 주장 대상.
"""
from collections import defaultdict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import world

KAPPA_CLONE = 0.9
KAPPA_MUTANT = 0.5
KAPPA_CHILD = 0.5
MUTANT_CAP = 2          # adversarial 부모당 추가 뮤턴트 상한
HONEST_INTERIM = 24     # honest 인터림 시점 (등록 후 개월)
ADV_PEEK_MIN = 6        # adversarial peeking 시작 (등록 후 개월)
HONEST_MUT_SR = 0.1     # honest winner mutation 문턱 (post 월간 SR)


def rolling_sr(Y, W):
    """(L,T) → (L,T) in-sample 월간 Sharpe. t < W-1은 -inf (등록 불가)."""
    win = sliding_window_view(Y, W, axis=1)
    mu = win.mean(axis=2)
    sd = win.std(axis=2, ddof=1)
    sr = np.full(Y.shape, -np.inf)
    sr[:, W - 1:] = mu / sd
    return sr


def _abandon_honest(series, A, T):
    """인터림 1회 (+24). 반환: (abandon_t | None, spawn_child: bool)."""
    t_i = A + HONEST_INTERIM
    if t_i >= T:
        return None, False
    post = series[A + 1:t_i + 1]
    if post.mean() <= 0:
        return t_i, False
    sr_post = post.mean() / post.std(ddof=1)
    return None, bool(sr_post > HONEST_MUT_SR)


def _abandon_adversarial(series, A, T):
    """매월 peeking: 누적 post 평균이 처음 음수가 되는 월 (>= A+6)에 포기."""
    post = series[A + 1:T]
    if len(post) < ADV_PEEK_MIN:
        return None
    cm = np.cumsum(post) / np.arange(1, len(post) + 1)
    bad = np.nonzero(cm[ADV_PEEK_MIN - 1:] < 0)[0]
    if len(bad) == 0:
        return None
    return A + ADV_PEEK_MIN + int(bad[0])


def run_search(rng_world, rng_mut, registry, *, T, L, W, s_reg, behavior,
               noise="gaussian", rho=0.0):
    """전부-null 세계에서 적응 탐색 → registry에 등록 채움 (E1용).

    rng_world: 기본 라이브러리 잡음 (CRN: behavior/s_reg와 무관한 seed 사용 권장)
    rng_mut:   클론/뮤턴트/자식의 신규 잡음
    """
    Y = world.draw_eps(rng_world, L, T, noise=noise, rho=rho)   # null: delta=0
    sr = rolling_sr(Y, W)
    above = sr > s_reg
    has_cross = above.any(axis=1)
    first_cross = np.where(has_cross, above.argmax(axis=1), -1)

    # adversarial 뮤턴트 트리거: in-sample SR가 s_reg 위 신고점을 찍는 월
    if behavior == "adversarial":
        runmax = np.maximum.accumulate(sr, axis=1)
        newhigh = above & (sr >= runmax)

    events = defaultdict(list)   # 월 -> [(kind, payload)]
    for j in range(L):
        if first_cross[j] >= 0:
            events[int(first_cross[j])].append(("base", j))

    reg_of_base = {}
    mutant_count = defaultdict(int)

    def _register(A, series, kind, parent):
        r = registry.register(A, series, is_null=True, kind=kind, parent=parent)
        if r is None:
            return None
        if behavior == "honest":
            ab, spawn = _abandon_honest(series, A, T)
            if ab is not None:
                r.stop_abandon(ab)
            if spawn and kind == "base":            # 자식은 재변형 없음 (1세대)
                events[A + HONEST_INTERIM].append(("child", r.rid))
        else:
            ab = _abandon_adversarial(series, A, T)
            if ab is not None:
                r.stop_abandon(ab)
        return r

    for t in range(W - 1, T):
        if registry.exhausted:
            break
        for kind, payload in events.get(t, []):
            if registry.exhausted:
                break
            if kind == "base":
                j = payload
                r = _register(t, Y[j], "base", None)
                if r is not None:
                    reg_of_base[j] = r
                    if behavior == "adversarial":   # 등록 즉시 클론 몰아넣기
                        cs = world.mutate_series(rng_mut, Y[j], KAPPA_CLONE, noise)
                        _register(t, cs, "clone", r.rid)
            elif kind == "child":                    # honest winner mutation
                parent = registry.regs[payload]
                if parent.abandon_t is not None and parent.abandon_t <= t:
                    continue
                cs = world.mutate_series(rng_mut, parent.series, KAPPA_CHILD, noise)
                _register(t, cs, "child", parent.rid)
        # adversarial: 등록된 부모의 in-sample 신고점 → 뮤턴트 추가
        if behavior == "adversarial":
            for j, r in list(reg_of_base.items()):
                if registry.exhausted:
                    break
                if t > r.A and newhigh[j, t] and mutant_count[j] < MUTANT_CAP \
                        and (r.abandon_t is None or t < r.abandon_t):
                    ms = world.mutate_series(rng_mut, Y[j], KAPPA_MUTANT, noise)
                    if _register(t, ms, "mutant", r.rid) is not None:
                        mutant_count[j] += 1
    return Y
