"""
Paper A — Day-1 detectability frontier.

Produces the one figure that decides the paper's narrative:
"minimum detectable monthly effect size (~ monthly Sharpe)" vs post-registration
sample length n, for two curves, plus the OSAP factor scatter on top.

  1) oracle_frontier  : closed-form sub-Gaussian benchmark (Prop. 3, §5.6.2).
                        OPTIMISTIC — drops mixture penalty + finite-sample concentration.
  2) catoni_frontier  : Monte-Carlo frontier of the ACTUAL Catoni-mixture e-process (§4.6)
                        under the registered solo-boundary freeze rule (§4.7).
                        This is the HONEST number you report.

===============  SET THESE TO YOUR REGISTERED VALUES  ===============
The delta -> monthly-Sharpe identity is exact ONLY if your predictable scale
v_{t-1} tracks the conditional sd of (R^net - h), i.e. the standardized score
has conditional sd ~ 1. M_ENV (>1) is the validity/power lever: larger = safer
(more plausibly a true 2nd-moment upper bound) but timider bets, hence a HIGHER
detectable delta. Re-run with YOUR M_ENV, scale rule, grid, and OSAP data.
====================================================================
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- CONFIG (pre-registered choices) -----------------------
J          = 212                      # registered strategies (OSAP family size)
ALPHA      = 0.05                     # target SupFDR level
BETA       = 0.5                      # 1-BETA = target detection power (0.5 = median)
N_GRID     = np.array([60, 120, 240, 360])      # post-registration months to tabulate
M_ENV      = 1.3                      # second-moment envelope multiplier (>1)
C_GRID     = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40])  # dimensionless bets c_k
W          = np.ones(len(C_GRID)) / len(C_GRID)  # mixture weights; c_1=0 keeps E>=w_1>0
NOISE      = os.environ.get("NOISE", "gaussian")  # "gaussian" or "t5" (finite-var fat tail, unit-var)
N_SIM      = 10_000                   # MC paths (raise to ~50k for a publication figure)
DELTA_GRID = np.linspace(0.0, 0.8, 21)           # standardized effect sizes searched
SEED       = 0
# -------------------------------------------------------------------------------

rng    = np.random.default_rng(SEED)
B_SOLO = J / ALPHA                    # solo first-discovery boundary 1/(alpha*gamma), gamma=1/J
LOG_B  = np.log(B_SOLO)


def oracle_frontier(n, log_b=LOG_B, beta=BETA, s=1.0):
    """Closed-form sub-Gaussian sufficient frontier (§5.6.2). s~1 in standardized units."""
    return s * (np.sqrt(2 * np.log(1 / beta) / n) + np.sqrt(2 * log_b / n))


def phi(x):
    """Catoni influence function; exp(phi(x)) <= 1 + x + x^2/2 for all real x.
    Both branches' log1p arguments stay > -1 everywhere, so this is always finite."""
    return np.where(x >= 0,
                    np.log1p(x + 0.5 * x * x),       # x >= 0
                    -np.log1p(-x + 0.5 * x * x))     # x <  0


def draw_noise(shape):
    if NOISE == "gaussian":
        return rng.standard_normal(shape)
    if NOISE == "t5":
        df = 5
        return rng.standard_t(df, size=shape) / np.sqrt(df / (df - 2))   # unit variance
    raise ValueError(NOISE)


def frontier_for_n(n, n_sim=N_SIM, deltas=DELTA_GRID, beta=BETA):
    """Min standardized delta with P_delta(e-process reaches B_SOLO within n) >= 1-beta.
    Common random numbers across deltas -> monotone, low-noise crossing probs."""
    eps  = draw_noise((n_sim, n))                    # shared noise (CRN)
    lam  = C_GRID / M_ENV                            # betting fractions lambda^(k) = c_k / sigma_bar
    logw = np.log(W)
    probs = np.empty(len(deltas))
    for i, d in enumerate(deltas):
        Y       = d + eps                            # standardized score: mean d, sd ~ 1
        arg     = Y[:, :, None] * lam[None, None, :] # (paths, time, K)
        inc     = phi(arg) - 0.5 * (lam ** 2) * (M_ENV ** 2)
        logprod = np.cumsum(inc, axis=1)             # log of each component's running product
        a       = logprod + logw[None, None, :]
        m       = a.max(axis=2, keepdims=True)       # log-sum-exp for the mixture
        logE    = m[:, :, 0] + np.log(np.exp(a - m).sum(axis=2))
        probs[i] = (logE.max(axis=1) >= LOG_B).mean()  # ever crossed within n
    target = 1 - beta
    if probs[-1] < target:
        return np.nan, probs                         # frontier above the searched grid
    idx = int(np.argmax(probs >= target))
    if idx == 0:
        return deltas[0], probs
    d0, d1 = deltas[idx - 1], deltas[idx]
    p0, p1 = probs[idx - 1], probs[idx]
    return d0 + (target - p0) * (d1 - d0) / (p1 - p0), probs


# ----------------------------- compute -----------------------------
oracle = oracle_frontier(N_GRID)
catoni = np.array([frontier_for_n(n)[0] for n in N_GRID])
null_cross = frontier_for_n(N_GRID[-1])[1][0]        # crossing prob at delta=0 (sanity)

print(f"B_solo = J/alpha = {B_SOLO:.0f}   log B_solo = {LOG_B:.2f}   "
      f"M_env = {M_ENV}   noise = {NOISE}   N_sim = {N_SIM}")
print(f"null (delta=0) crossing prob ~ {null_cross:.5f}   "
      f"(Ville bound = alpha/J = {ALPHA/J:.5f})\n")
print(f"{'n (mo)':>7} | {'oracle d*':>10} {'ann.SR':>7} | {'Catoni d*':>10} {'ann.SR':>7}")
print("-" * 56)
for n, o, c in zip(N_GRID, oracle, catoni):
    ann_o = o * np.sqrt(12)
    ann_c = c * np.sqrt(12) if np.isfinite(c) else np.nan
    cs = f"{c:>10.3f} {ann_c:>7.2f}" if np.isfinite(c) else f"{'>0.80':>10} {'n/a':>7}"
    print(f"{n:>7} | {o:>10.3f} {ann_o:>7.2f} | {cs}")


# --------------------------- OSAP overlay ---------------------------
def load_osap_monthly_sharpe():
    """OSAP data release 2025.10 (v2.00), 원논문 구현(op) LS 포트폴리오.
    등록시점 A_j = 출판연도 12월 말; post-registration = Year+1년 1월부터.
    Return (n_j, monthly_sharpe_j): predictor별 post-registration 유효 월 수와
    그 윈도우의 실현 MONTHLY Sharpe (stage2_postpub_sharpe.py가 생성)."""
    import csv
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "osap_postpub_sharpe.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == J, f"expected J={J} predictors, got {len(rows)}"
    n_j  = np.array([int(r["n_j"]) for r in rows])
    sh_m = np.array([float(r["sharpe_m"]) for r in rows])
    return n_j, sh_m


# ------------------------------ plot -------------------------------
nn = np.linspace(48, 372, 200)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(nn, oracle_frontier(nn) * np.sqrt(12), lw=2,
        label="oracle frontier (sub-Gaussian, optimistic)")
ax.plot(N_GRID, catoni * np.sqrt(12), "o-", lw=2,
        label=f"Catoni-mixture frontier (M_env={M_ENV}, {NOISE}) — honest")
n_j, sh_j = load_osap_monthly_sharpe()
ax.scatter(n_j, sh_j * np.sqrt(12), s=14, alpha=0.35, color="crimson",
           label="OSAP factors (v2.00 op LS, post-publication)")
ax.set_xlabel("post-registration sample length n (months)")
ax.set_ylabel("minimum detectable Sharpe (annualized)")
ax.set_title(f"Detectability frontier   (J={J}, α={ALPHA}, power={1-BETA:.0%})")
ax.set_ylim(0, None)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
FIG_NAME = "frontier.png" if NOISE == "gaussian" else f"frontier_{NOISE}.png"
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), FIG_NAME), dpi=140)

below = (sh_j <= np.interp(n_j, N_GRID, catoni)).mean()
print(f"\nsaved -> {FIG_NAME}")
print(f"OSAP v2.00: {below:.0%} of factors fall BELOW the Catoni frontier (undetectable)")
