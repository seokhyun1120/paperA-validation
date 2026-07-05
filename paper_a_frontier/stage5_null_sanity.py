"""Stage 5 — null sanity gate (pass/fail).

N_SIM=100,000, n=360, delta=0에서 Catoni-mixture e-process의 crossing rate를
gaussian / t5 각각 측정. 기준: crossing rate <= 0.000381
(= alpha/J + 3 x MC 표준오차, 사전 확정). 초과 시 STOP.

e-process 정의(phi, 베팅 그리드, envelope 페널티, 혼합)는 frontier.py 커밋
2d7834c의 phi() / frontier_for_n()과 동일 — 파라미터는 하드코딩으로 일치시키고
로직은 수정 없이 옮겨 적음. 메모리 때문에 배치 처리만 추가.
"""
import numpy as np

J = 212
ALPHA = 0.05
M_ENV = 1.3
C_GRID = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
W = np.ones(len(C_GRID)) / len(C_GRID)
N = 360
N_SIM = 100_000
BATCH = 10_000
SEED = 0
THRESHOLD = 0.000381  # alpha/J + 3*MC 표준오차 (사전 확정)

LOG_B = np.log(J / ALPHA)


def phi(x):
    # frontier.py의 phi와 동일 (Catoni influence)
    return np.where(x >= 0,
                    np.log1p(x + 0.5 * x * x),
                    -np.log1p(-x + 0.5 * x * x))


def draw_noise(rng, shape, noise):
    if noise == "gaussian":
        return rng.standard_normal(shape)
    if noise == "t5":
        df = 5
        return rng.standard_t(df, size=shape) / np.sqrt(df / (df - 2))
    raise ValueError(noise)


def null_crossing_rate(noise):
    rng = np.random.default_rng(SEED)
    lam = C_GRID / M_ENV
    logw = np.log(W)
    crossed = 0
    for _ in range(N_SIM // BATCH):
        Y = draw_noise(rng, (BATCH, N), noise)          # delta = 0
        arg = Y[:, :, None] * lam[None, None, :]
        inc = phi(arg) - 0.5 * (lam ** 2) * (M_ENV ** 2)
        logprod = np.cumsum(inc, axis=1)
        a = logprod + logw[None, None, :]
        m = a.max(axis=2, keepdims=True)
        logE = m[:, :, 0] + np.log(np.exp(a - m).sum(axis=2))
        crossed += int((logE.max(axis=1) >= LOG_B).sum())
    return crossed / N_SIM, crossed


print(f"null sanity gate: N_SIM={N_SIM:,}, n={N}, delta=0, "
      f"threshold={THRESHOLD} (alpha/J={ALPHA/J:.6f})\n")
all_pass = True
for noise in ["gaussian", "t5"]:
    rate, k = null_crossing_rate(noise)
    ok = rate <= THRESHOLD
    all_pass &= ok
    print(f"  {noise:>8}: crossing rate = {rate:.6f} ({k}/{N_SIM:,})  "
          f"-> {'PASS' if ok else 'FAIL'}")

print(f"\n{'PASS: 두 케이스 모두 기준 이하' if all_pass else '*** FAIL: STOP — 코드/freeze rule/e-process 점검 필요 ***'}")
