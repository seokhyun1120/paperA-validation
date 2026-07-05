"""sim/registry.py — 등록 원장 (append-only).

등록 시점에 Theta_j(여기서는 score 시계열 참조와 종류), A_j, gamma_j, envelope,
deadline이 동결된다. 조기 포기(abandon)는 등록 취소가 아니라 관측 중단 기록
(set-once)이며 gamma 예산은 반환되지 않는다.
gamma_j = 1/J_budget 균등 (§13), b_j_solo = 1/(alpha*gamma_j).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Registration:
    rid: int
    A: int                      # 등록 월 (데이터 의존 stopping time)
    gamma: float
    m_env: float
    deadline: int               # D_j = A_j + deadline_len (calendar 월)
    is_null: bool
    series: np.ndarray          # 잠재 score 전체 시계열 (post-A_j만 e-process에 사용)
    kind: str = "base"          # base / clone / mutant / child
    parent: Optional[int] = None
    abandon_t: Optional[int] = field(default=None)  # set-once 관측 중단 월

    def stop_abandon(self, t):
        assert self.abandon_t is None, "abandon은 set-once"
        self.abandon_t = t


class Registry:
    """append-only 원장. 예산 소진 시 등록 거절 (None 반환)."""

    def __init__(self, J_budget, alpha, m_env, deadline_len):
        self.J_budget = J_budget
        self.alpha = alpha
        self.gamma = 1.0 / J_budget
        self.m_env = m_env
        self.deadline_len = deadline_len
        self.log_b = float(np.log(1.0 / (alpha * self.gamma)))
        self._regs = []

    def register(self, A, series, is_null=True, kind="base", parent=None):
        if len(self._regs) >= self.J_budget:
            return None
        r = Registration(rid=len(self._regs), A=int(A), gamma=self.gamma,
                         m_env=self.m_env, deadline=int(A) + self.deadline_len,
                         is_null=is_null, series=series, kind=kind, parent=parent)
        self._regs.append(r)
        return r

    @property
    def regs(self):
        return list(self._regs)   # 사본 — 원장 자체는 append-only

    def __len__(self):
        return len(self._regs)

    @property
    def exhausted(self):
        return len(self._regs) >= self.J_budget
