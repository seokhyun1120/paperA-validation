"""Rev5 M5 — envelope-consistent fixed-H 인증의 A_j 관례별 반복.

Rev4 M5의 cutoff/v_min/fixed-H 구성 함수를 재사용하되, 관례별 pre window에서
mean(Y^2)를 다시 계산해 m_env_j=max(1.3, sqrt(meanY2_j))를 적용한다.

실행:
  ../../.venv/bin/python3 review/rev5_m5_adj_sets.py
산출:
  data/rev5_m5_adj_sets.csv
"""
from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

assert os.environ.get("NOISE", "gaussian") == "gaussian", "rev5_m5는 gaussian frontier로 실행해야 한다"

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
REVIEW = REPO / "review"

for path in (REPO, REVIEW):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

# frontier import 때 사용자 홈 아래 matplotlib 캐시 생성을 피한다.
MPLCONFIGDIR = Path("/tmp") / "quant_research_mpl"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import rev4_m5_aj_sets as m5  # noqa: E402
from rev5_m1_full_adj import BASE_M_ENV, TOL, adjusted_single_path, bool_values  # noqa: E402

m1 = m5.m1


def convention_mean_y2(panel: pd.DataFrame, vmin: pd.DataFrame, convention: str) -> pd.DataFrame:
    """관례별 pre window에서 stage4 방식의 mean(Y^2)를 계산한다."""
    vmin_s = vmin.set_index("signalname")["v_min"]
    rows: list[dict[str, object]] = []
    for signalname, g in panel.groupby("signalname", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        pub_year = int(g["pub_year"].iloc[0])
        pre = g.loc[g["date"] < m5.cutoff_date(pub_year, convention), "ret"].to_numpy(dtype=float)
        v = (
            pd.Series(pre)
            .rolling(m1.WINDOW, min_periods=m1.MIN_OBS)
            .std(ddof=1)
            .shift(1)
            .to_numpy()
        )
        valid = ~np.isnan(v)
        v_min = float(vmin_s.loc[signalname])
        if valid.any():
            y = pre[valid] / np.maximum(v[valid], v_min)
            mean_y2 = float(np.mean(y**2))
        else:
            mean_y2 = np.nan
        rows.append(
            {
                "signalname": signalname,
                "n_pre": int(len(pre)),
                "n_Y": int(valid.sum()),
                "v_min": v_min,
                "meanY2": mean_y2,
            }
        )
    return pd.DataFrame(rows).sort_values("signalname").reset_index(drop=True)


def check_primary_mean_y2(primary_mean_y2: pd.DataFrame) -> None:
    """primary meanY2가 stage4 CSV와 1e-9 이내로 같은지 확인한다."""
    saved = pd.read_csv(DATA / "stage4_envelope_check.csv").set_index("signalname").sort_index()
    cur = primary_mean_y2.set_index("signalname").sort_index()
    assert len(cur) == m1.J_BUDGET == len(saved), "primary meanY2 팩터 수 불일치"
    assert cur.index.equals(saved.index), "primary meanY2 signalname 집합 불일치"
    assert np.allclose(
        cur["meanY2"].to_numpy(dtype=float),
        saved["meanY2"].to_numpy(dtype=float),
        rtol=0.0,
        atol=TOL,
        equal_nan=True,
    ), "primary meanY2 stage4 재현 실패"
    print("primary stage4 meanY2 게이트 PASS: 212개 1e-9 일치")


def tau_cal_from_months(tau_months: np.ndarray, date_rows: list[np.ndarray]) -> np.ndarray:
    """tau_months와 날짜 행렬에서 e-BH reveal용 달력 월 정수를 복원한다."""
    tau_cal = np.empty(len(tau_months), dtype=int)
    for i, dates in enumerate(date_rows):
        idx = int(tau_months[i]) - 1
        tau_cal[i] = m1.month_index(pd.Timestamp(dates[idx]))
    return tau_cal


def run_fixed_h_adjusted(
    cert_registered: pd.DataFrame,
    y_mat: np.ndarray,
    date_rows: list[np.ndarray],
    mean_y2: pd.DataFrame,
) -> tuple[pd.DataFrame, int, np.ndarray]:
    """관례별 fixed-H 결과에 m_env_j 조정을 적용하고 e-BH를 재판정한다."""
    mean_s = mean_y2.set_index("signalname")["meanY2"]
    mapped = cert_registered["signalname"].map(mean_s)
    if mapped.isna().any():
        missing = cert_registered.loc[mapped.isna(), "signalname"].tolist()
        raise RuntimeError(f"관례별 meanY2 누락: {missing}")

    m_env_consistent = np.maximum(BASE_M_ENV, np.sqrt(mapped.to_numpy(float)))
    to_adjust = m_env_consistent > BASE_M_ENV + TOL

    e_log_adj = cert_registered["logE_tau"].to_numpy(float).copy()
    tau_months_adj = cert_registered["tau_months"].to_numpy(int).copy()
    solo_adj = cert_registered["solo"].to_numpy(dtype=bool).copy()
    gamma_e_adj = cert_registered["gamma_e"].to_numpy(float).copy()
    tau_cal_adj = tau_cal_from_months(tau_months_adj, date_rows)

    log_b = m1.ep.log_b_solo(1.0 / m1.J_BUDGET)
    for i in np.nonzero(to_adjust)[0]:
        result = adjusted_single_path(y_mat[i], date_rows[i], float(m_env_consistent[i]), log_b)
        e_log_adj[i] = float(result["logE_tau"])
        tau_months_adj[i] = int(result["tau_months"])
        tau_cal_adj[i] = int(result["tau_cal"])
        solo_adj[i] = bool(result["solo"])
        gamma_e_adj[i] = float(result["gamma_e"])

    reveal = m1.direct_reveal(e_log_adj, tau_cal_adj)
    base = m1.ebh.baseline_reveal(
        e_log_adj,
        tau_cal_adj,
        is_null=np.zeros(len(e_log_adj), dtype=bool),
        alpha=m1.ALPHA,
        J_budget=m1.J_BUDGET,
    )
    assert np.array_equal(base["disc"], reveal["disc"]), "adjusted baseline_reveal disc 검산 실패"
    assert np.array_equal(
        base["disc_step"], reveal["disc_step"]
    ), "adjusted baseline_reveal disc_step 검산 실패"

    out = cert_registered.copy()
    out["meanY2_pre"] = mapped.to_numpy(float)
    out["m_env_consistent"] = m_env_consistent
    out["logE_tau_adj"] = e_log_adj
    out["tau_months_adj"] = tau_months_adj
    out["solo_adj"] = solo_adj
    out["gamma_e_adj"] = gamma_e_adj
    out["ebh_thr_gamma_e_adj"] = reveal["thr_gamma"]
    out["rejected_adj"] = reveal["disc"]
    out["disc_step_adj"] = reveal["disc_step"]
    return out, int(reveal["k_199"]), to_adjust


def load_rev5_m1_certified() -> set[str]:
    """Rev5 M1 full adjusted 인증 집합을 읽는다."""
    rev5 = pd.read_csv(DATA / "rev5_m1_full_adj.csv")
    rev5["rejected_adj"] = bool_values(rev5["rejected_adj"])
    return set(rev5.loc[rev5["rejected_adj"], "signalname"])


def report_adjusted_invariance(results: dict[str, dict[str, object]]) -> bool:
    """세 관례의 adjusted certified set 동일성을 보고한다."""
    keys = [key for key, _ in m5.CONVENTIONS]
    sets = {key: set(results[key]["certified_adj"]) for key in keys}
    identical = all(sets[key] == sets[keys[0]] for key in keys[1:])
    print(f"\nadjusted certified set-invariance: {'PASS' if identical else 'FAIL'}")
    if identical:
        print(f"adjusted core {len(sets[keys[0]])}개 불변 확인: {sorted(sets[keys[0]])}")
        return True

    for left, right in combinations(keys, 2):
        only_left = sorted(sets[left] - sets[right])
        only_right = sorted(sets[right] - sets[left])
        if only_left or only_right:
            print(f"  {left} vs {right}")
            print(f"    only {left}: {only_left if only_left else '없음'}")
            print(f"    only {right}: {only_right if only_right else '없음'}")
    return False


def main() -> None:
    """관례별 adjusted certified set을 계산하고 CSV/로그를 저장한다."""
    panel = m1.load_panel()
    assert panel["signalname"].nunique() == m1.J_BUDGET, "Predictor 팩터 수 불일치"
    rev5_m1_certified = load_rev5_m1_certified()

    results: dict[str, dict[str, object]] = {}
    long_rows: list[dict[str, object]] = []

    for key, label in m5.CONVENTIONS:
        vmin = m5.convention_vmin(panel, key)
        mean_y2 = convention_mean_y2(panel, vmin, key)
        cert_base, y_mat, date_rows = m5.build_fixed_h(panel, vmin, key)
        cert_registered, k_registered = m5.run_fixed_h(cert_base, y_mat, date_rows)
        cert_adj, k_adj, to_adjust = run_fixed_h_adjusted(
            cert_registered, y_mat, date_rows, mean_y2
        )
        certified_adj = set(cert_adj.loc[cert_adj["rejected_adj"], "signalname"])

        if key == "primary":
            check_primary_mean_y2(mean_y2)
            if certified_adj != rev5_m1_certified:
                raise AssertionError(
                    "primary adjusted certified set 불일치: "
                    f"{sorted(certified_adj)} vs {sorted(rev5_m1_certified)}"
                )
            print("primary Rev5 M1 full adjusted 정합 게이트 PASS")

        results[key] = {
            "label": label,
            "matured": len(cert_adj),
            "k_registered": k_registered,
            "k_adj": k_adj,
            "n_adjusted": int(to_adjust.sum()),
            "certified_adj": certified_adj,
        }

        for signalname in sorted(certified_adj):
            long_rows.append(
                {
                    "convention": key,
                    "convention_label": label,
                    "set_type": "certified_adj",
                    "signalname": signalname,
                    "k_final": k_adj,
                    "matured": len(cert_adj),
                    "n_m_env_adjusted": int(to_adjust.sum()),
                }
            )
        long_rows.append(
            {
                "convention": key,
                "convention_label": label,
                "set_type": "summary",
                "signalname": "",
                "k_final": k_adj,
                "matured": len(cert_adj),
                "n_m_env_adjusted": int(to_adjust.sum()),
            }
        )

        print(f"\n[{key}] {label}")
        print(f"  matured fixed-H count: {len(cert_adj)} / budget {m1.J_BUDGET}")
        print(f"  registered k_final={k_registered}, adjusted k_final={k_adj}")
        print(f"  m_env_j > 1.3 adjusted count: {int(to_adjust.sum())}")
        print(f"  certified_adj set ({len(certified_adj)}): {sorted(certified_adj)}")

    identical = report_adjusted_invariance(results)
    print(f"\n최종 판정: adjusted certified 동일={identical}")

    out = DATA / "rev5_m5_adj_sets.csv"
    pd.DataFrame(long_rows).to_csv(out, index=False)
    print(f"saved -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
