"""Rev5 M1 — 199개 전체 envelope-consistent 재계산.

Rev3 M1의 fixed-H=120 등록순을 그대로 재현한 뒤, pre-publication mean(Y^2)가
1.3^2를 초과하는 모든 matured 팩터에 대해
m_env_j=max(1.3, sqrt(meanY2_j))로 e-process를 재실행한다.

실행:
  ../../.venv/bin/python3 review/rev5_m1_full_adj.py
산출:
  data/rev5_m1_full_adj.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

import rev3_m1_ebh_cert as m1  # noqa: E402
from sim import ebh, eprocess as ep  # noqa: E402

BASE_M_ENV = 1.3
TOL = 1e-9
EXPECTED_N_ENV_ADJ = 11
EXPECTED_CERTIFIED_ADJ = {
    "AnalystRevision",
    "AnnouncementReturn",
    "EarningsSurprise",
    "STreversal",
}
EXPECTED_FINAL_THR_GAMMA = 1.0 / (m1.ALPHA * len(EXPECTED_CERTIFIED_ADJ))
REV4_CERTIFIED = {
    "AnalystRevision",
    "AnnouncementReturn",
    "DivYieldST",
    "EarningsSurprise",
    "SmileSlope",
    "STreversal",
}


def bool_values(s: pd.Series) -> np.ndarray:
    """CSV bool 또는 문자열 bool 컬럼을 bool 배열로 정규화한다."""
    if pd.api.types.is_bool_dtype(s):
        return s.to_numpy(dtype=bool)
    mapped = s.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        bad = s[mapped.isna()].head().tolist()
        raise AssertionError(f"bool 변환 실패: {bad}")
    return mapped.to_numpy(dtype=bool)


def load_rev3_csv() -> pd.DataFrame:
    """Rev3 M1 CSV를 읽고 판정 컬럼을 명시적으로 bool로 맞춘다."""
    rev3 = pd.read_csv(DATA / "rev3_m1_ebh_cert.csv")
    rev3["solo"] = bool_values(rev3["solo"])
    rev3["rejected"] = bool_values(rev3["rejected"])
    return rev3


def replay_registered(
    cert_base: pd.DataFrame, y_mat: np.ndarray, date_rows: list[np.ndarray]
) -> dict[str, np.ndarray]:
    """registered m_env=1.3 e-process, freeze, e-BH reveal을 재현한다."""
    assert y_mat.shape == (m1.EXPECTED_MATURED, m1.H), f"Y 행렬 shape 불일치: {y_mat.shape}"
    log_e = ep.log_e_path(y_mat, m_env=BASE_M_ENV)
    log_b = ep.log_b_solo(1.0 / m1.J_BUDGET)
    assert log_b == ep.frontier.LOG_B, "log_b_solo(1/212)와 frontier.LOG_B 불일치"
    log_e_frozen, tau = ep.freeze_at_crossing(log_e, log_b)
    solo = tau >= 0

    tau_months = np.empty(len(cert_base), dtype=int)
    tau_cal = np.empty(len(cert_base), dtype=int)
    for i, dates in enumerate(date_rows):
        idx = int(tau[i]) if solo[i] else m1.H - 1
        tau_months[i] = idx + 1 if solo[i] else m1.H
        tau_cal[i] = m1.month_index(pd.Timestamp(dates[idx]))

    e_log = log_e_frozen[:, -1]
    reveal = m1.direct_reveal(e_log, tau_cal)
    base = ebh.baseline_reveal(
        e_log,
        tau_cal,
        is_null=np.zeros(len(e_log), dtype=bool),
        alpha=m1.ALPHA,
        J_budget=m1.J_BUDGET,
    )
    assert np.array_equal(base["disc"], reveal["disc"]), "registered baseline_reveal disc 검산 실패"
    assert np.array_equal(
        base["disc_step"], reveal["disc_step"]
    ), "registered baseline_reveal disc_step 검산 실패"

    return {
        "log_e": log_e,
        "max_logE": log_e.max(axis=1),
        "e_log": e_log,
        "tau_months": tau_months,
        "tau_cal": tau_cal,
        "solo": solo,
        "gamma_e": np.exp(e_log) / m1.J_BUDGET,
        "ebh_thr_gamma_e": reveal["thr_gamma"],
        "rejected": reveal["disc"],
        "disc_step": reveal["disc_step"],
        "k_199": np.asarray([int(reveal["k_199"])], dtype=int),
    }


def assert_rev3_reproduction(
    cert_base: pd.DataFrame, replay: dict[str, np.ndarray], rev3: pd.DataFrame
) -> None:
    """재구성된 registered 결과가 Rev3 CSV 전체와 1e-9 이내로 같은지 확인한다."""
    if cert_base["signalname"].tolist() != rev3["signalname"].tolist():
        raise AssertionError("signalname 등록순 불일치")

    for col in ("pub_year",):
        assert np.array_equal(
            cert_base[col].to_numpy(dtype=int), rev3[col].to_numpy(dtype=int)
        ), f"{col} 재구성 불일치"

    for col in ("sr120_raw_ann", "mean_Y", "sr120_std_ann"):
        assert np.allclose(
            cert_base[col].to_numpy(float),
            rev3[col].to_numpy(float),
            rtol=0.0,
            atol=TOL,
        ), f"{col} 재구성 불일치"

    float_checks = (
        ("max_logE", replay["max_logE"]),
        ("logE_tau", replay["e_log"]),
        ("gamma_e", replay["gamma_e"]),
        ("ebh_thr_gamma_e", replay["ebh_thr_gamma_e"]),
    )
    for col, arr in float_checks:
        assert np.allclose(arr, rev3[col].to_numpy(float), rtol=0.0, atol=TOL), f"{col} 재현 실패"

    int_checks = (
        ("tau_months", replay["tau_months"]),
        ("disc_step", replay["disc_step"]),
    )
    for col, arr in int_checks:
        assert np.array_equal(arr, rev3[col].to_numpy(dtype=int)), f"{col} 재현 실패"

    assert np.array_equal(replay["solo"], rev3["solo"].to_numpy(dtype=bool)), "solo 재현 실패"
    assert np.array_equal(
        replay["rejected"], rev3["rejected"].to_numpy(dtype=bool)
    ), "rejected 재현 실패"
    print("registered 재현 게이트 PASS: Rev3 CSV 전 컬럼 1e-9 검산")


def adjusted_single_path(
    y: np.ndarray, dates: np.ndarray, m_env: float, log_b: float
) -> dict[str, object]:
    """단일 팩터의 envelope-consistent e-process와 freeze 결과를 계산한다."""
    log_e = ep.log_e_path(y[None, :], m_env=m_env)
    log_e_frozen, tau = ep.freeze_at_crossing(log_e, log_b)
    tau_i = int(tau[0])
    solo = tau_i >= 0
    idx = tau_i if solo else m1.H - 1
    e_log = float(log_e_frozen[0, -1])
    return {
        "max_logE": float(log_e.max(axis=1)[0]),
        "logE_tau": e_log,
        "tau_months": int(idx + 1 if solo else m1.H),
        "tau_cal": int(m1.month_index(pd.Timestamp(dates[idx]))),
        "solo": bool(solo),
        "gamma_e": float(np.exp(e_log) / m1.J_BUDGET),
    }


def build_full_adjustment(
    rev3: pd.DataFrame,
    y_mat: np.ndarray,
    date_rows: list[np.ndarray],
    replay: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """199개 전체에 m_env_j를 배정하고 초과 팩터만 개별 재실행한다."""
    stage4 = pd.read_csv(DATA / "stage4_envelope_check.csv")[["signalname", "meanY2"]]
    assert stage4["signalname"].is_unique, "stage4 meanY2 signalname 중복"
    mean_y2 = rev3["signalname"].map(stage4.set_index("signalname")["meanY2"])
    if mean_y2.isna().any():
        missing = rev3.loc[mean_y2.isna(), "signalname"].tolist()
        raise RuntimeError(f"stage4 meanY2 누락: {missing}")

    m_env_consistent = np.maximum(BASE_M_ENV, np.sqrt(mean_y2.to_numpy(float)))
    to_adjust = m_env_consistent > BASE_M_ENV + TOL
    n_adjust = int(to_adjust.sum())
    if n_adjust != EXPECTED_N_ENV_ADJ:
        names = rev3.loc[to_adjust, "signalname"].tolist()
        raise AssertionError(f"m_env 초과 팩터 수 불일치: {n_adjust}, 목록={names}")

    e_log_adj = replay["e_log"].astype(float).copy()
    tau_months_adj = replay["tau_months"].astype(int).copy()
    tau_cal_adj = replay["tau_cal"].astype(int).copy()
    solo_adj = replay["solo"].astype(bool).copy()
    gamma_e_adj = replay["gamma_e"].astype(float).copy()
    max_log_e_adj = replay["max_logE"].astype(float).copy()

    log_b = ep.log_b_solo(1.0 / m1.J_BUDGET)
    for i in np.nonzero(to_adjust)[0]:
        result = adjusted_single_path(y_mat[i], date_rows[i], float(m_env_consistent[i]), log_b)
        e_log_adj[i] = float(result["logE_tau"])
        tau_months_adj[i] = int(result["tau_months"])
        tau_cal_adj[i] = int(result["tau_cal"])
        solo_adj[i] = bool(result["solo"])
        gamma_e_adj[i] = float(result["gamma_e"])
        max_log_e_adj[i] = float(result["max_logE"])

    copy_mask = ~to_adjust
    assert np.allclose(e_log_adj[copy_mask], replay["e_log"][copy_mask], rtol=0.0, atol=TOL)
    assert np.array_equal(tau_months_adj[copy_mask], replay["tau_months"][copy_mask])
    assert np.array_equal(tau_cal_adj[copy_mask], replay["tau_cal"][copy_mask])
    assert np.array_equal(solo_adj[copy_mask], replay["solo"][copy_mask])
    assert np.allclose(gamma_e_adj[copy_mask], replay["gamma_e"][copy_mask], rtol=0.0, atol=TOL)

    out = rev3.copy()
    out["meanY2_pre"] = mean_y2.to_numpy(float)
    out["m_env_consistent"] = m_env_consistent
    out["logE_tau_adj"] = e_log_adj
    out["tau_months_adj"] = tau_months_adj
    out["solo_adj"] = solo_adj
    out["gamma_e_adj"] = gamma_e_adj

    adj_arrays = {
        "e_log": e_log_adj,
        "tau_cal": tau_cal_adj,
        "max_logE": max_log_e_adj,
        "tau_months": tau_months_adj,
        "solo": solo_adj,
        "gamma_e": gamma_e_adj,
        "to_adjust": to_adjust,
    }
    return out, adj_arrays


def run_adjusted_ebh(out: pd.DataFrame, adj_arrays: dict[str, np.ndarray]) -> int:
    """adjusted 동결값으로 등록순 e-BH를 재판정하고 baseline 구현과 대조한다."""
    reveal = m1.direct_reveal(adj_arrays["e_log"], adj_arrays["tau_cal"])
    base = ebh.baseline_reveal(
        adj_arrays["e_log"],
        adj_arrays["tau_cal"],
        is_null=np.zeros(len(out), dtype=bool),
        alpha=m1.ALPHA,
        J_budget=m1.J_BUDGET,
    )
    assert np.array_equal(base["disc"], reveal["disc"]), "adjusted baseline_reveal disc 검산 실패"
    assert np.array_equal(
        base["disc_step"], reveal["disc_step"]
    ), "adjusted baseline_reveal disc_step 검산 실패"

    out["ebh_thr_gamma_e_adj"] = reveal["thr_gamma"]
    out["rejected_adj"] = reveal["disc"]
    out["disc_step_adj"] = reveal["disc_step"]
    print("adjusted e-BH 대조 게이트 PASS: direct_reveal == baseline_reveal")
    return int(reveal["k_199"])


def assert_rev5_gates(out: pd.DataFrame, k_199_adj: int) -> None:
    """과제 1의 STOP 게이트를 모두 검사한다."""
    adjusted = set(out.loc[out["rejected_adj"], "signalname"])
    if adjusted != EXPECTED_CERTIFIED_ADJ:
        raise RuntimeError(
            "게이트(c) 실패: adjusted 인증 집합 불일치 "
            f"{sorted(adjusted)} != {sorted(EXPECTED_CERTIFIED_ADJ)}"
        )
    if k_199_adj != len(EXPECTED_CERTIFIED_ADJ):
        raise RuntimeError(f"게이트(c) 실패: adjusted k_199={k_199_adj}")

    final_thr_gamma = 1.0 / (m1.ALPHA * k_199_adj)
    if not np.isclose(final_thr_gamma, EXPECTED_FINAL_THR_GAMMA, rtol=0.0, atol=TOL):
        raise RuntimeError(f"게이트(c) 실패: final gamma_e threshold={final_thr_gamma}")
    certified_gamma = out.loc[out["rejected_adj"], "gamma_e_adj"].to_numpy(float)
    if not np.all(certified_gamma >= final_thr_gamma - TOL):
        raise RuntimeError("게이트(c) 실패: adjusted 인증 팩터가 final gamma_e 문턱 미만")

    promoted = out.loc[~out["rejected"] & out["rejected_adj"], "signalname"].tolist()
    if promoted:
        raise RuntimeError(f"게이트(d) 실패: registered non-rejected 승격 발생 {promoted}")

    print(
        "게이트(c) PASS: adjusted 인증 집합/k/final gamma_e 문턱 "
        f"{final_thr_gamma:.6f} 확인"
    )
    print("게이트(d) PASS: registered non-rejected -> adjusted rejected 승격 0건")


def assert_rev4_consistency(out: pd.DataFrame) -> None:
    """Rev4 partial 결과와 인증 6개 adjusted 열이 1e-9 이내로 같은지 확인한다."""
    rev4 = pd.read_csv(DATA / "rev4_m1_envelope_adj.csv")
    rev4["solo_adj"] = bool_values(rev4["solo_adj"])
    rev4["rejected_adj"] = bool_values(rev4["rejected_adj"])

    cur = out.set_index("signalname").loc[sorted(REV4_CERTIFIED)]
    old = rev4.set_index("signalname").loc[sorted(REV4_CERTIFIED)]

    float_cols = [
        "meanY2_pre",
        "m_env_consistent",
        "logE_tau_adj",
        "gamma_e_adj",
        "ebh_thr_gamma_e_adj",
    ]
    for col in float_cols:
        assert np.allclose(
            cur[col].to_numpy(float), old[col].to_numpy(float), rtol=0.0, atol=TOL
        ), f"Rev4 정합 실패: {col}"

    int_cols = ["tau_months_adj", "disc_step_adj"]
    for col in int_cols:
        assert np.array_equal(cur[col].to_numpy(int), old[col].to_numpy(int)), f"Rev4 정합 실패: {col}"

    bool_cols = ["solo_adj", "rejected_adj"]
    for col in bool_cols:
        assert np.array_equal(
            cur[col].to_numpy(dtype=bool), old[col].to_numpy(dtype=bool)
        ), f"Rev4 정합 실패: {col}"

    print("Rev4 partial 정합 게이트 PASS: 인증 6개 adjusted 값 1e-9 일치")


def print_summary(out: pd.DataFrame, adj_arrays: dict[str, np.ndarray], k_199_adj: int) -> None:
    """stdout 요구 항목을 등록순 기준으로 출력한다."""
    adjust_rows = out.loc[
        adj_arrays["to_adjust"],
        ["signalname", "m_env_consistent", "logE_tau", "logE_tau_adj", "gamma_e_adj"],
    ]
    print("\n조정 팩터 11개:")
    for _, row in adjust_rows.iterrows():
        print(
            f"  {row['signalname']}: m_env_j={row['m_env_consistent']:.6f}, "
            f"logE_tau {row['logE_tau']:.6f} -> {row['logE_tau_adj']:.6f}, "
            f"gamma_e_adj={row['gamma_e_adj']:.6g}"
        )

    adjusted_names = out.loc[out["rejected_adj"], "signalname"].tolist()
    promoted = out.loc[~out["rejected"] & out["rejected_adj"], "signalname"].tolist()
    print(f"\nadjusted 인증 목록: {', '.join(adjusted_names)}")
    print(f"adjusted k_199: {k_199_adj}")
    print(f"승격 건수: {len(promoted)}")
    print("게이트 전부 PASS")


def main() -> None:
    """Rev5 M1 full envelope-consistent 산출물을 생성한다."""
    cert_base, y_mat, date_rows = m1.build_fixed_h_matrix()
    rev3 = load_rev3_csv()

    replay = replay_registered(cert_base, y_mat, date_rows)
    assert_rev3_reproduction(cert_base, replay, rev3)

    out, adj_arrays = build_full_adjustment(rev3, y_mat, date_rows, replay)
    k_199_adj = run_adjusted_ebh(out, adj_arrays)
    assert_rev5_gates(out, k_199_adj)
    assert_rev4_consistency(out)

    out_csv = DATA / "rev5_m1_full_adj.csv"
    out.to_csv(out_csv, index=False)
    print(f"saved -> {out_csv.relative_to(REPO)}")
    print_summary(out, adj_arrays, k_199_adj)


if __name__ == "__main__":
    main()
