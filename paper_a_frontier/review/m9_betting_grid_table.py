"""M9 — betting grid 공개 표 (appendix용 LaTeX).

frontier.py에서 상수를 직접 읽어(전사 오류 방지) 사전 등록 구성요소 전체를
LaTeX 표로 출력: c_k, w_k, K, scale rule, v_min, envelope, 보간/외삽, seed 정책.
실행: MPLBACKEND=Agg python3 review/m9_betting_grid_table.py
→ review/appendix_M9_betting_grid.tex
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim import eprocess as ep

fr = ep.frontier
c = ", ".join(f"{x:g}" for x in fr.C_GRID)
w = f"1/{len(fr.C_GRID)}"

TEX = rf"""% M9 — pre-registered betting grid and scale rules (auto-generated,
% review/m9_betting_grid_table.py; values read directly from frontier.py)
\begin{{table}}[htbp]
\centering\small
\caption{{Pre-registered e-process components (Paper A, \S 4.6--4.7)}}
\label{{tab:betting-grid}}
\begin{{tabular}}{{p{{0.30\linewidth}}p{{0.62\linewidth}}}}
\hline
Component & Registered value / rule \\
\hline
Betting grid $c_k$ ($K={len(fr.C_GRID)}$) & $\{{{c}\}}$ (dimensionless; $\lambda_k = c_k/m_{{\mathrm{{env}}}}$) \\
Mixture weights $w_k$ & uniform, $w_k = {w}$; $c_1=0$ keeps $E \ge w_1 > 0$ \\
Envelope $m_{{\mathrm{{env}}}}$ & ${fr.M_ENV}$; coverage rule: block mean$(Y^2) \le m_{{\mathrm{{env}}}}^2$ for $\ge 90\%$ of factors on pre-publication data (adopted: min of \{{1.3, 1.5, 1.7\}}) \\
Scale rule $v_{{t-1}}$ & trailing 36-month sd of raw returns (ddof$=1$), min.\ 24 obs.; strictly predictable (uses data through $t-1$ only) \\
Floor $v_{{\min}}$ & 5\% quantile of the factor's own pre-registration trailing-sd series; no cross-sectional or future data \\
Standardized score & $Y_t = R_t / \max(v_{{t-1}}, v_{{\min}})$ \\
e-process increment & $\log E$ increment $= \phi(\lambda_k Y_t) - \tfrac12 \lambda_k^2 m_{{\mathrm{{env}}}}^2$, Catoni influence $\phi$; mixture via log-sum-exp \\
Boundary / budget & $\gamma_j = 1/J$, $J={fr.J}$, $\alpha={fr.ALPHA}$, $b_{{\mathrm{{solo}}}} = 1/(\alpha\gamma_j) = {fr.B_SOLO:.0f}$ \\
Freeze rule & $E_j$ frozen at first of: solo crossing, deadline $D_j = A_j + 120$, abandonment \\
Frontier grid & $\delta \in$ linspace$(0, 1.2, 31)$; $\delta^*$ = linear interpolation at crossing prob $1-\beta = 0.5$ \\
Frontier interp/extrap in $n_j$ & linear in $n$ on $\{{60,120,240,360\}}$ (extended $\{{480,540,600\}}$); outside: $\sqrt{{n}}$ scaling $f(n) = f(n_0)\sqrt{{n_0/n}}$ \\
Seed policy & frontier MC: SEED$={fr.SEED}$, $N_{{\mathrm{{sim}}}}={fr.N_SIM:,}$, CRN across $\delta$; simulator: \texttt{{default\_rng([20260706, exp, cell, run])}}, CRN across compared cells \\
\hline
\end{{tabular}}
\end{{table}}
"""

out = Path(__file__).parent / "appendix_M9_betting_grid.tex"
out.write_text(TEX)
print(TEX)
print(f"saved -> {out}")
