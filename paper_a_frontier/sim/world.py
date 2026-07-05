"""sim/world.py — score 생성기.

전략 j의 잠재 시계열 Y_{j,t} = delta_{j,t} + eps_{j,t}.
eps는 표준화 잡음 (gaussian / t5, 단위분산; t5 표준화는 frontier.draw_noise와
동일한 sqrt(df/(df-2)) 스케일). 선택적 공통요인:
  eps = sqrt(rho) * F_t + sqrt(1-rho) * eta_{j,t}   (cross-sectional 의존 실험용)
null: delta ≡ 0. alternative: delta 상수 또는 decay (E3).
"""
import numpy as np


def std_noise(rng, shape, noise="gaussian"):
    """단위분산 표준화 잡음."""
    if noise == "gaussian":
        return rng.standard_normal(shape)
    if noise == "t5":
        df = 5
        return rng.standard_t(df, size=shape) / np.sqrt(df / (df - 2))
    raise ValueError(noise)


def draw_eps(rng, n_strat, T, noise="gaussian", rho=0.0):
    """(n_strat, T) 단위분산 잡음. rho>0이면 공통요인 구조 (marginal 분산 1 유지)."""
    eta = std_noise(rng, (n_strat, T), noise)
    if rho > 0:
        F = std_noise(rng, (1, T), noise)
        eta = np.sqrt(rho) * F + np.sqrt(1.0 - rho) * eta
    return eta


def mutate_series(rng, parent_series, kappa, noise="gaussian"):
    """winner mutation: 부모와 상관 kappa^0.5인 파생 시계열 (단위분산 유지).
    null 세계에서 부모가 null이면 자식도 null (조건부 평균 0)."""
    fresh = std_noise(rng, parent_series.shape, noise)
    return np.sqrt(kappa) * parent_series + np.sqrt(1.0 - kappa) * fresh


def decay_delta(delta0, rho_d, horizon):
    """E3: delta_{A+s} = delta0 * rho_d^(s-1), s=1..horizon.
    무한합 = delta0/(1-rho_d) (브리프의 누적 edge 정의와 일치)."""
    s = np.arange(1, horizon + 1)
    return delta0 * rho_d ** (s - 1)
