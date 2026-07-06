"""M8 — audit artifact 표 (appendix용 LaTeX).

전 실험의 git commit hash(실행 기준), seed, N을 한 표로. 데이터 파일 sha256은
실행 시점에 직접 계산. 실행: python3 review/m8_audit_table.py
→ review/appendix_M8_audit.tex
"""
import hashlib
import subprocess
from pathlib import Path

REPO = Path(__file__).parent.parent
HEAD = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                      capture_output=True, text=True).stdout.strip()


def sha(p):
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()[:16]


DATA_FILES = ["data/osap_LS_v200.csv.gz", "data/SignalDoc.csv",
              "data/osap_postpub_sharpe.csv"]

# (실험, 실행 기준 commit, seed, N)
ROWS = [
    ("Frontier MC (Table 2, 4 pts)", "2d7834c", "SEED=0", r"$N_{\rm sim}$=10{,}000"),
    ("Stage 4 envelope diagnostic", "01b5ded", "deterministic", "212 factors"),
    ("Stage 5 null sanity gate", "01b5ded", "SEED=0", r"100{,}000 $\times$ 2"),
    ("Stage 6 grid ext.\\ + margin", "01b5ded", "SEED=0", r"$N_{\rm sim}$=10{,}000"),
    ("E1 validity (live)", "9d091ea", "[20260706,1,$\\cdot$]", r"16 $\times$ 2{,}000"),
    ("E2 power vs frontier", "9d091ea", "[20260706,2,$\\cdot$]", r"12 $\times$ 10{,}000"),
    ("E3 alpha-decay wall", "9d091ea", "[20260706,3,$\\cdot$]", r"18 $\times$ 10{,}000"),
    ("E4 envelope stress", "9d091ea", "[20260706,4,$\\cdot$]", r"10 $\times$ 2{,}000"),
    ("E5 timing (live)", "9d091ea", "[20260706,5,$\\cdot$]", r"3 $\times$ 500"),
    ("E1 baseline reveal + comparators", "b2656c1", "= E1 (CRN reuse)", r"16 $\times$ 2{,}000"),
    ("E5 baseline reveal decomp.", "b2656c1", "[20260706,5,$\\cdot$]", r"3 $\times$ 500"),
    ("R1 mechanism check", "5b9cbac", "[20260706,6,$\\cdot$]", r"8 $\times$ 2{,}000"),
    ("R1b e-BH tautness", "d7b2e7c", "20260706", r"200{,}000 $\times$ 2"),
    ("R2 deflated-SR comparator", "5b9cbac", "= E1 (CRN reuse)", r"16 $\times$ 2{,}000"),
    ("M4/M6/M7/M10/M11 review calcs", HEAD, "SEED=0", "see RUN\\_LOG"),
    ("R3 boundary-near null stress", HEAD, "[20260706,8,$\\cdot$]", r"2 $\times$ 2{,}000"),
]

lines = [r"% M8 — audit artifacts (auto-generated, review/m8_audit_table.py)",
         r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{Audit artifacts: code version, seeds, and sample sizes}",
         r"\label{tab:audit}", r"\begin{tabular}{llll}", r"\hline",
         r"Experiment & Commit & Seed & $N$ \\", r"\hline"]
for name, commit, seed, n in ROWS:
    lines.append(f"{name} & \\texttt{{{commit}}} & {seed} & {n} \\\\")
lines += [r"\hline", r"\multicolumn{4}{l}{\emph{Data (sha256, first 16 hex)}} \\"]
for p in DATA_FILES:
    lines.append(f"\\texttt{{{Path(p).name.replace('_', chr(92)+'_')}}} & "
                 f"\\multicolumn{{3}}{{l}}{{\\texttt{{{sha(p)}}}}} \\\\")
lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

TEX = "\n".join(lines) + "\n"
out = Path(__file__).parent / "appendix_M8_audit.tex"
out.write_text(TEX)
print(TEX)
print(f"saved -> {out}")
