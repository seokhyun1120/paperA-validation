"""Rev4 M4 — t(5) 장기 frontier 3점 산출.

frontier.py는 import 시점에 NOISE를 읽고 4점 Catoni frontier를 계산한다.
따라서 이 스크립트는 반드시 NOISE=t5 환경에서 시작하고, import 이후 rng를
SEED로 되감아 n=[60,120,240,360,480,540,600] 순서의 추첨을 재현한다.

실행:
  NOISE=t5 MPLBACKEND=Agg ../../.venv/bin/python3 review/rev4_m4_t5_long.py
"""
import os

assert os.environ["NOISE"] == "t5", "NOISE=t5 환경에서만 실행해야 한다"

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
sys.path.insert(0, str(REPO))
from sim import eprocess as ep  # noqa: E402  frontier import 및 savefig 보호

fr = ep.frontier
assert fr.NOISE == "t5", f"frontier.NOISE 불일치: {fr.NOISE}"

N_ALL = [60, 120, 240, 360, 480, 540, 600]
RUN_LOG_T5_ANN_ROUND = np.array([2.86, 1.84, 1.28, 1.04])

# run_rev3_fig1.log의 gaussian 600-grid 출력값(연율). 월간 값은 재계산하지 않는다.
GAUSSIAN_ANN_REFERENCE = {
    480: 0.895242,
    540: 0.836983,
    600: 0.791898,
}


def check_import_gate() -> None:
    """import 시 계산된 t5 4점이 RUN_LOG 반올림 값과 일치하는지 확인한다."""
    ann = fr.catoni * np.sqrt(12.0)
    got = np.round(ann, 2)
    assert np.array_equal(got, RUN_LOG_T5_ANN_ROUND), (
        f"t5 RUN_LOG 게이트 실패: got={got}, expected={RUN_LOG_T5_ANN_ROUND}"
    )
    assert round(float(fr.catoni[0]), 3) == 0.825, f"n=60 월간 delta* 확인 실패: {fr.catoni[0]}"

    print("t5 import 게이트: PASS")
    for n_months, delta, ann_sr, rounded in zip(fr.N_GRID, fr.catoni, ann, got):
        print(
            f"  n={int(n_months):>3}: delta*={delta:.12f}, "
            f"ann={ann_sr:.6f}, round2={rounded:.2f}"
        )


def main() -> None:
    """t5 장기 frontier를 계산하고 CSV를 저장한다."""
    check_import_gate()

    fr.rng = np.random.default_rng(fr.SEED)
    rows: list[dict[str, float | int | str]] = []

    print("\nRev4 M4 t5 long-horizon Catoni frontier:")
    print(f"{'n (mo)':>7} | {'delta*_t5':>14} | {'ann_t5':>10} | {'gauss_ref':>10} | {'diff':>10} | 비고")
    print("-" * 78)
    for i, n_months in enumerate(N_ALL):
        delta_star, _ = fr.frontier_for_n(n_months)
        if i < len(fr.catoni):
            assert delta_star == fr.catoni[i], (
                f"기존 4점 bit-exact 실패 n={n_months}: {delta_star} vs {fr.catoni[i]}"
            )
            note = "기존 4점 불변"
        else:
            note = "신규 장기점"

        ann_t5 = float(delta_star * np.sqrt(12.0))
        ref = GAUSSIAN_ANN_REFERENCE.get(n_months)
        diff = "" if ref is None else ann_t5 - ref
        ref_cell = "NA" if ref is None else f"{ref:.6f}"
        diff_cell = "NA" if ref is None else f"{diff:+.6f}"
        print(
            f"{n_months:>7} | {delta_star:>14.12f} | {ann_t5:>10.6f} | "
            f"{ref_cell:>10} | {diff_cell:>10} | {note}"
        )
        rows.append(
            {
                "n": n_months,
                "delta_star_t5": float(delta_star),
                "ann_t5": ann_t5,
                "ann_gaussian_ref": "" if ref is None else ref,
                "diff": diff,
            }
        )

    out = DATA / "rev4_m4_t5_long.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n", "delta_star_t5", "ann_t5", "ann_gaussian_ref", "diff"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n기존 4점 == frontier.catoni (bit-exact) 확인 완료")
    print(f"saved -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
