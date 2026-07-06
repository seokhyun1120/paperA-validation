"""R1b — e-BH aggregation-layer tautness under synthetic worst-case null e-values.

e_j iid with P(e_j = B) = 1/B, else 0 (mean one), B = J/alpha, gamma_j = 1/J.
All-null, so sup_m FDP_m = 1{any discovery}. With two-point e-values the e-BH
rejection condition e_j >= B/k holds automatically for e_j = B, so
SupFDR = P(at least one e_j = B) = 1 - (1 - 1/B)^J ~ 1 - exp(-alpha).
"""
import numpy as np

J, N_RUN, SEED = 212, 200_000, 20260706
for alpha in (0.05, 0.20):
    B = J / alpha
    rng = np.random.default_rng(SEED)
    hits = rng.random((N_RUN, J)) < 1.0 / B          # e_j = B events
    supfdr = hits.any(axis=1).mean()
    se = np.sqrt(supfdr * (1 - supfdr) / N_RUN)
    analytic = 1 - (1 - 1 / B) ** J
    print(f"alpha={alpha}: empirical SupFDR={supfdr:.4f} (SE {se:.4f}), "
          f"analytic={analytic:.4f}, 1-exp(-alpha)={1-np.exp(-alpha):.4f}")
