"""Rev3 Fig1 — 600개월 grid로 frontier.png를 명시적으로 갱신한다.

frontier.py는 수정하지 않는다. sim.eprocess를 통해 frontier 커널을 단일 출처로
가져오고, M7과 같은 RNG 재설정 및 호출 순서로 n=600까지 확장한 Fig1을 그린다.
실행: MPLBACKEND=Agg ../../.venv/bin/python3 review/rev3_fig1_600.py
산출: frontier.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sim import eprocess as ep  # noqa: E402  frontier import 및 savefig 보호

N_ALL = np.array([60, 120, 240, 360, 480, 540, 600], dtype=int)


def compute_frontier_600() -> np.ndarray:
    """M7과 동일한 추첨 순서로 7점 Catoni frontier를 계산한다."""
    fr = ep.frontier
    fr.rng = np.random.default_rng(fr.SEED)
    catoni = np.array([fr.frontier_for_n(int(n))[0] for n in N_ALL], dtype=float)
    old = catoni[: len(fr.N_GRID)]
    assert np.array_equal(old, fr.catoni), f"기존 4점 bit-exact 불변 실패: {old} vs {fr.catoni}"
    return catoni


def save_plot(catoni: np.ndarray) -> None:
    """frontier.py의 Fig1 스타일을 600개월 grid에 맞춰 재현한다."""
    fr = ep.frontier
    nn = np.linspace(48, 612, 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        nn,
        fr.oracle_frontier(nn) * np.sqrt(12),
        lw=2,
        label="oracle frontier (sub-Gaussian, optimistic)",
    )
    ax.plot(
        N_ALL,
        catoni * np.sqrt(12),
        "o-",
        lw=2,
        label=f"Catoni-mixture frontier (M_env={fr.M_ENV}, {fr.NOISE}) — honest",
    )
    n_j, sh_j = fr.load_osap_monthly_sharpe()
    ax.scatter(
        n_j,
        sh_j * np.sqrt(12),
        s=14,
        alpha=0.35,
        color="crimson",
        label="OSAP factors (v2.00 op LS, post-publication)",
    )
    ax.set_xlabel("post-registration sample length n (months)")
    ax.set_ylabel("minimum detectable Sharpe (annualized)")
    ax.set_title(f"Detectability frontier   (J={fr.J}, α={fr.ALPHA}, power={1-fr.BETA:.0%})")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPO / "frontier.png", dpi=140)
    plt.close(fig)


def main() -> None:
    """7점 frontier를 계산하고 frontier.png를 갱신한다."""
    catoni = compute_frontier_600()
    print("\nRev3 Fig1 600-grid Catoni frontier:")
    print(f"{'n (mo)':>7} | {'delta*':>14} | {'ann.SR':>10}")
    print("-" * 40)
    for n_months, delta in zip(N_ALL, catoni):
        print(f"{n_months:>7} | {delta:>14.12f} | {delta * np.sqrt(12):>10.6f}")
    print("\n기존 4점 bit-exact 불변 확인: PASS")
    save_plot(catoni)
    print("saved -> frontier.png")


if __name__ == "__main__":
    main()
