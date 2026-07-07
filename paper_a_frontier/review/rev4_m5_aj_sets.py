"""Rev4 M5 — A_j 관례별 survivor/certified set 불변성 점검.

세 관례에서 full-horizon frontier survivor와 fixed-H=120 e-BH 인증 집합을
계산한다. full-horizon frontier는 frontier.catoni 정확값을 사용하고, fixed-H
인증은 rev3 M1의 표준화·freeze·등록순 reveal 규칙을 그대로 재사용한다.

실행:
  MPLBACKEND=Agg ../../.venv/bin/python3 review/rev4_m5_aj_sets.py
"""
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

assert os.environ.get("NOISE", "gaussian") == "gaussian", "rev4_m5는 gaussian frontier로 실행해야 한다"

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
REVIEW = REPO / "review"
sys.path.insert(0, str(REVIEW))
import rev3_m1_ebh_cert as m1  # noqa: E402  M1 게이트/프로토콜 재사용

fr = m1.ep.frontier
assert fr.NOISE == "gaussian", f"frontier.NOISE 불일치: {fr.NOISE}"

VMIN_Q = 0.05
N_GRID = np.array([60, 120, 240, 360], dtype=float)
CATONI_ANN_EXACT = fr.catoni * np.sqrt(12.0)
EXPECTED_SURVIVOR_PRIMARY = {"AnalystRevision", "AnnouncementReturn", "DivYieldST"}

CONVENTIONS = [
    ("year_jan", "Year-01-01"),
    ("year_jul", "Year-07-01"),
    ("primary", "(Year+1)-01-01"),
]


def cutoff_date(pub_year: int, convention: str) -> pd.Timestamp:
    """관례별 post 시작일을 반환한다."""
    if convention == "year_jan":
        return pd.Timestamp(year=pub_year, month=1, day=1)
    if convention == "year_jul":
        return pd.Timestamp(year=pub_year, month=7, day=1)
    if convention == "primary":
        return pd.Timestamp(year=pub_year + 1, month=1, day=1)
    raise ValueError(convention)


def frontier_at(n_j: np.ndarray) -> np.ndarray:
    """stage6 보간/외삽 규칙을 frontier.catoni 정확값으로 적용한다."""
    n = np.asarray(n_j, dtype=float)
    f = np.interp(n, N_GRID, CATONI_ANN_EXACT)
    hi = n > N_GRID[-1]
    f[hi] = CATONI_ANN_EXACT[-1] * np.sqrt(N_GRID[-1] / n[hi])
    lo = n < N_GRID[0]
    f[lo] = CATONI_ANN_EXACT[0] * np.sqrt(N_GRID[0] / n[lo])
    return f


def full_horizon(panel: pd.DataFrame, convention: str) -> pd.DataFrame:
    """관례별 전체 post window의 Sharpe와 exact frontier margin을 계산한다."""
    rows: list[dict[str, object]] = []
    for signalname, g in panel.groupby("signalname", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        pub_year = int(g["pub_year"].iloc[0])
        post = g.loc[g["date"] >= cutoff_date(pub_year, convention), "ret"].to_numpy(dtype=float)
        if len(post) < 2:
            raise RuntimeError(f"{convention}: post 표본 부족: {signalname}, n={len(post)}")
        sd_m = float(post.std(ddof=1))
        sharpe_ann = float(np.sqrt(12.0) * post.mean() / sd_m)
        rows.append(
            {
                "signalname": signalname,
                "pub_year": pub_year,
                "n_j": int(len(post)),
                "mean_m": float(post.mean()),
                "sd_m": sd_m,
                "sharpe_ann": sharpe_ann,
            }
        )
    res = pd.DataFrame(rows).sort_values("signalname").reset_index(drop=True)
    res["frontier_at_nj_exact"] = frontier_at(res["n_j"].to_numpy(dtype=float))
    res["margin_exact"] = res["sharpe_ann"] - res["frontier_at_nj_exact"]
    res["survivor"] = res["margin_exact"] >= 0.0
    return res


def convention_vmin(panel: pd.DataFrame, convention: str) -> pd.DataFrame:
    """관례별 pre window만 사용해 stage4 방식의 v_min을 재계산한다."""
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for signalname, g in panel.groupby("signalname", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        pub_year = int(g["pub_year"].iloc[0])
        pre = g.loc[g["date"] < cutoff_date(pub_year, convention), "ret"].to_numpy(dtype=float)
        v = (
            pd.Series(pre)
            .rolling(m1.WINDOW, min_periods=m1.MIN_OBS)
            .std(ddof=1)
            .shift(1)
            .to_numpy()
        )
        valid = ~np.isnan(v)
        v_min = float(np.nanquantile(v, VMIN_Q)) if valid.any() else np.nan
        if np.isnan(v_min):
            missing.append(signalname)
        rows.append(
            {
                "signalname": signalname,
                "n_pre": int(len(pre)),
                "n_v": int(valid.sum()),
                "v_min": v_min,
            }
        )
    if missing:
        print(f"{convention}: v_min NaN 팩터: {', '.join(missing)}")
        raise RuntimeError("관례별 v_min 산출 불가 팩터가 있음")
    return pd.DataFrame(rows).sort_values("signalname").reset_index(drop=True)


def build_fixed_h(
    panel: pd.DataFrame, vmin: pd.DataFrame, convention: str
) -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray]]:
    """관례별 post 첫 120개월 표준화 score 행렬을 만든다."""
    vmin_s = vmin.set_index("signalname")["v_min"]
    rows: list[dict[str, object]] = []
    y_rows: list[np.ndarray] = []
    date_rows: list[np.ndarray] = []

    for signalname, g in panel.groupby("signalname", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        pub_year = int(g["pub_year"].iloc[0])
        post_pos = np.flatnonzero((g["date"] >= cutoff_date(pub_year, convention)).to_numpy())
        if len(post_pos) < m1.H:
            continue

        r_all = g["ret"].to_numpy(dtype=float)
        v_all = (
            pd.Series(r_all)
            .rolling(m1.WINDOW, min_periods=m1.MIN_OBS)
            .std(ddof=1)
            .shift(1)
            .to_numpy()
        )
        pos = post_pos[: m1.H]
        v_h = v_all[pos]
        if np.isnan(v_h).any():
            print(f"{convention}: v_{{t-1}} NaN 포함 팩터: {signalname}")
            raise RuntimeError("post 첫 120개월 표준화에 예측가능 변동성 결측이 있음")

        v_min = float(vmin_s.loc[signalname])
        r_h = r_all[pos]
        y = r_h / np.maximum(v_h, v_min)
        rows.append(
            {
                "signalname": signalname,
                "pub_year": pub_year,
                "n_post": int(len(post_pos)),
                "sr120_raw_ann": float(np.sqrt(12.0) * r_h.mean() / r_h.std(ddof=1)),
                "mean_Y": float(y.mean()),
                "sr120_std_ann": float(np.sqrt(12.0) * y.mean()),
                "array_idx": len(y_rows),
            }
        )
        y_rows.append(y)
        date_rows.append(g.loc[pos, "date"].to_numpy())

    cert = pd.DataFrame(rows)
    cert = cert.sort_values(["pub_year", "signalname"], kind="mergesort").reset_index(drop=True)
    order = cert.pop("array_idx").to_numpy(dtype=int)
    y_mat = np.vstack([y_rows[i] for i in order])
    ordered_dates = [date_rows[i] for i in order]
    return cert, y_mat, ordered_dates


def run_fixed_h(
    cert: pd.DataFrame, y_mat: np.ndarray, date_rows: list[np.ndarray]
) -> tuple[pd.DataFrame, int]:
    """rev3 M1과 같은 freeze 및 등록순 reveal을 실행한다."""
    log_e = m1.ep.log_e_path(y_mat, m_env=m1.M_ENV)
    log_b = m1.ep.log_b_solo(1.0 / m1.J_BUDGET)
    assert log_b == m1.ep.frontier.LOG_B, "solo boundary 불일치"
    log_e_frozen, tau = m1.ep.freeze_at_crossing(log_e, log_b)
    solo = tau >= 0

    tau_cal = np.empty(len(cert), dtype=int)
    tau_months = np.empty(len(cert), dtype=int)
    for i, dates in enumerate(date_rows):
        idx = int(tau[i]) if solo[i] else m1.H - 1
        tau_cal[i] = m1.month_index(pd.Timestamp(dates[idx]))
        tau_months[i] = idx + 1 if solo[i] else m1.H

    e_log = log_e_frozen[:, -1]
    reveal = m1.direct_reveal(e_log, tau_cal)
    base = m1.ebh.baseline_reveal(
        e_log,
        tau_cal,
        is_null=np.zeros(len(e_log), dtype=bool),
        alpha=m1.ALPHA,
        J_budget=m1.J_BUDGET,
    )
    assert np.array_equal(base["disc"], reveal["disc"]), "baseline_reveal disc 검산 실패"

    out = cert.copy()
    out["max_logE"] = log_e.max(axis=1)
    out["logE_tau"] = e_log
    out["tau_months"] = tau_months
    out["solo"] = solo
    out["gamma_e"] = np.exp(e_log) / m1.J_BUDGET
    out["ebh_thr_gamma_e"] = reveal["thr_gamma"]
    out["rejected"] = reveal["disc"]
    out["disc_step"] = reveal["disc_step"]
    return out, int(reveal["k_199"])


def check_primary_stage6(primary_full: pd.DataFrame) -> None:
    """primary full-horizon 원자료와 기존 stage6 CSV의 self-consistency를 검산한다."""
    saved = pd.read_csv(DATA / "osap_postpub_sharpe.csv").set_index("signalname").sort_index()
    cur = primary_full.set_index("signalname").sort_index()
    assert len(cur) == m1.J_BUDGET == len(saved), "primary full-horizon 팩터 수 불일치"
    assert np.array_equal(cur["n_j"].to_numpy(dtype=int), saved["n_j"].to_numpy(dtype=int))
    assert np.allclose(
        cur["sharpe_ann"].to_numpy(dtype=float),
        saved["sharpe_ann"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ), "primary sharpe_ann stage6 재현 실패"

    legacy_margin = cur["sharpe_ann"].to_numpy(dtype=float) - saved["frontier_at_nj"].to_numpy(dtype=float)
    assert np.allclose(
        legacy_margin,
        saved["margin"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    ), "primary margin stage6 CSV self-consistency 실패"

    exact_diff = cur["margin_exact"].to_numpy(dtype=float) - saved["margin"].to_numpy(dtype=float)
    print("primary stage6 게이트: PASS")
    print(f"  n_j/sharpe_ann 및 CSV legacy margin allclose 확인")
    print(f"  exact frontier margin - CSV margin 최대절대차: {np.max(np.abs(exact_diff)):.12f}")


def check_primary_vmin(primary_vmin: pd.DataFrame) -> None:
    """primary v_min이 stage4 산출물과 1e-9 이내로 같은지 확인한다."""
    saved = pd.read_csv(DATA / "stage4_envelope_check.csv").set_index("signalname").sort_index()
    cur = primary_vmin.set_index("signalname").sort_index()
    assert len(cur) == m1.J_BUDGET == len(saved), "primary v_min 팩터 수 불일치"
    assert np.allclose(
        cur.loc[saved.index, "v_min"].to_numpy(dtype=float),
        saved["v_min"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    ), "primary v_min stage4 재현 실패"
    print("primary stage4 v_min 게이트: PASS")


def report_set_invariance(results: dict[str, dict[str, object]], field: str, label: str) -> bool:
    """세 관례의 집합 동일성을 출력하고, 다르면 대칭차를 전부 출력한다."""
    keys = [key for key, _ in CONVENTIONS]
    sets = {key: set(results[key][field]) for key in keys}
    identical = all(sets[key] == sets[keys[0]] for key in keys[1:])
    print(f"\n{label} set-invariance: {'PASS' if identical else 'FAIL'}")
    if identical:
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
    """관례별 집합을 계산하고 CSV/로그를 저장한다."""
    panel = m1.load_panel()
    assert panel["signalname"].nunique() == m1.J_BUDGET, "Predictor 팩터 수 불일치"
    print(f"frontier.catoni exact monthly: {fr.catoni}")
    print(f"frontier.catoni exact annual : {CATONI_ANN_EXACT}")

    results: dict[str, dict[str, object]] = {}
    long_rows: list[dict[str, object]] = []

    for key, label in CONVENTIONS:
        full = full_horizon(panel, key)
        survivor = set(full.loc[full["survivor"], "signalname"])
        vmin = convention_vmin(panel, key)
        cert_base, y_mat, date_rows = build_fixed_h(panel, vmin, key)
        cert, k_final = run_fixed_h(cert_base, y_mat, date_rows)
        certified = set(cert.loc[cert["rejected"], "signalname"])

        if key == "primary":
            check_primary_stage6(full)
            assert survivor == EXPECTED_SURVIVOR_PRIMARY, (
                f"primary survivor set 불일치: {sorted(survivor)}"
            )
            check_primary_vmin(vmin)
            rev3 = pd.read_csv(DATA / "rev3_m1_ebh_cert.csv")
            expected_cert = set(rev3.loc[rev3["rejected"], "signalname"])
            assert certified == expected_cert, (
                f"primary certified set 불일치: {sorted(certified)} vs {sorted(expected_cert)}"
            )
            print("primary rev3 certified 게이트: PASS")

        results[key] = {
            "label": label,
            "matured": len(cert),
            "k_final": k_final,
            "survivor": survivor,
            "certified": certified,
        }
        for signalname in sorted(survivor):
            long_rows.append(
                {
                    "convention": key,
                    "convention_label": label,
                    "set_type": "survivor",
                    "signalname": signalname,
                }
            )
        for signalname in sorted(certified):
            long_rows.append(
                {
                    "convention": key,
                    "convention_label": label,
                    "set_type": "certified",
                    "signalname": signalname,
                }
            )

        print(f"\n[{key}] {label}")
        print(f"  matured fixed-H count: {len(cert)} / budget {m1.J_BUDGET} (k_final={k_final})")
        print(f"  survivor set ({len(survivor)}): {sorted(survivor)}")
        print(f"  certified set ({len(certified)}): {sorted(certified)}")

    survivor_same = report_set_invariance(results, "survivor", "survivor")
    certified_same = report_set_invariance(results, "certified", "certified")
    print(f"\n최종 판정: survivor 동일={survivor_same}, certified 동일={certified_same}")

    out = DATA / "rev4_m5_aj_sets.csv"
    pd.DataFrame(long_rows).to_csv(out, index=False)
    print(f"saved -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
