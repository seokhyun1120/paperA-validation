"""sim/eprocess.py — frontier.py의 Catoni-mixture 커널 import (복사 금지).

frontier.py는 모듈 레벨에서 전체 frontier 계산을 수행하므로 import에 ~40초가
들고 그림 저장을 시도한다. 파일 무수정 원칙을 지키면서 부작용만 막기 위해
import 동안 matplotlib Figure.savefig를 일시 무력화한다 (frontier.png 보호).

log_e_path()는 frontier.frontier_for_n 내부의 e-process 구성 연산을
순서까지 그대로 사용한다 — sim/test_eprocess.py 회귀 테스트(허용오차 0) 대상.
"""
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.figure as _mfig

_orig_savefig = _mfig.Figure.savefig
_mfig.Figure.savefig = lambda *a, **k: None
try:
    import frontier  # noqa: E402  모듈 레벨 계산 실행 — 커널 단일 출처
finally:
    _mfig.Figure.savefig = _orig_savefig

PHI = frontier.phi          # Catoni influence (커널)
C_GRID = frontier.C_GRID    # 등록된 베팅 그리드
W_MIX = frontier.W          # 혼합 가중치
ALPHA = frontier.ALPHA
M_ENV_DEFAULT = frontier.M_ENV


def log_b_solo(gamma, alpha=ALPHA):
    """solo first-discovery boundary: b = 1/(alpha*gamma)의 log."""
    return float(np.log(1.0 / (alpha * gamma)))


def log_e_path(Y, m_env=M_ENV_DEFAULT, c_grid=C_GRID, w=W_MIX):
    """Y: (paths, T) 표준화 score → (paths, T) log e-process 경로.
    frontier.frontier_for_n의 증분/혼합 구성과 연산 순서 동일."""
    lam = c_grid / m_env
    logw = np.log(w)
    arg = Y[:, :, None] * lam[None, None, :]
    inc = PHI(arg) - 0.5 * (lam ** 2) * (m_env ** 2)
    logprod = np.cumsum(inc, axis=1)
    a = logprod + logw[None, None, :]
    m = a.max(axis=2, keepdims=True)
    return m[:, :, 0] + np.log(np.exp(a - m).sum(axis=2))


def freeze_at_crossing(logE, log_b):
    """freeze rule: solo boundary 최초 도달 이후 값을 도달 시점 값으로 동결.
    반환: (동결된 경로, tau) — tau는 최초 도달 인덱스 (미도달 -1)."""
    crossed = logE >= log_b
    ever = crossed.any(axis=1)
    idx = crossed.argmax(axis=1)
    n, T = logE.shape
    frozen_val = logE[np.arange(n), idx]
    tgrid = np.arange(T)[None, :]
    out = np.where(ever[:, None] & (tgrid >= idx[:, None]),
                   frozen_val[:, None], logE)
    tau = np.where(ever, idx, -1)
    return out, tau
