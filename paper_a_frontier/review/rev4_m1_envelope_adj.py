"""Rev4 M1 — 인증 6개 팩터의 envelope-consistent 재검증.

Rev3 M1의 fixed-H=120 데이터 재구성과 등록순 e-BH 프로토콜을 그대로 재사용한다.
조정 범위는 심사 대응 과제의 지시 범위인 Rev3 인증 6개 팩터로 제한한다. 따라서
비인증 193개 팩터는 pre-pub mean(Y^2)와 무관하게 registered 값을 유지하며, 이는
새 전략 탐색이나 grid 변경이 아니라 인증 집합에 대한 사후 일관성 점검이다.

실행: ../../.venv/bin/python3 review/rev4_m1_envelope_adj.py
산출: data/rev4_m1_envelope_adj.csv, review/appendix_rev3_m1.tex
"""
from __future__ import annotations

import sys
import os
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

# frontier import 때 matplotlib가 홈 디렉터리에 캐시를 쓰지 않도록 한다.
MPLCONFIGDIR = Path("/tmp") / "quant_research_mpl"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import rev3_m1_ebh_cert as m1  # noqa: E402
from sim import ebh, eprocess as ep  # noqa: E402

BASE_M_ENV = 1.3
TOL = 1e-9
EXPECTED_ADJUSTED = {
    "AnnouncementReturn",
    "SmileSlope",
    "AnalystRevision",
    "STreversal",
}


def bool_values(s: pd.Series) -> np.ndarray:
    """CSV에서 읽은 bool 또는 문자열 bool을 안전하게 bool 배열로 바꾼다."""
    if pd.api.types.is_bool_dtype(s):
        return s.to_numpy(dtype=bool)
    mapped = s.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        bad = s[mapped.isna()].head().tolist()
        raise AssertionError(f"bool 변환 실패: {bad}")
    return mapped.to_numpy(dtype=bool)


def load_rev3_csv() -> pd.DataFrame:
    """Rev3 M1 인증 CSV를 읽고 bool 컬럼을 명시적으로 정규화한다."""
    rev3 = pd.read_csv(DATA / "rev3_m1_ebh_cert.csv")
    rev3["solo"] = bool_values(rev3["solo"])
    rev3["rejected"] = bool_values(rev3["rejected"])
    return rev3


def replay_registered(
    cert_base: pd.DataFrame, y_mat: np.ndarray, date_rows: list[np.ndarray]
) -> dict[str, np.ndarray]:
    """registered e-process를 직접 재현하고 freeze/e-BH 입력 배열을 반환한다."""
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
    return {
        "log_e": log_e,
        "max_logE": log_e.max(axis=1),
        "e_log": e_log,
        "tau_months": tau_months,
        "tau_cal": tau_cal,
        "solo": solo,
        "gamma_e": np.exp(e_log) / m1.J_BUDGET,
    }


def assert_rev3_reproduction(
    cert_base: pd.DataFrame, replay: dict[str, np.ndarray], rev3: pd.DataFrame
) -> None:
    """Rev3 CSV의 등록순과 registered freeze 결과를 1e-9 허용오차로 검산한다."""
    assert cert_base["signalname"].tolist() == rev3["signalname"].tolist(), "signalname 등록순 불일치"
    assert np.allclose(
        cert_base["sr120_raw_ann"].to_numpy(float),
        rev3["sr120_raw_ann"].to_numpy(float),
        rtol=0.0,
        atol=TOL,
    ), "sr120_raw_ann 재구성 불일치"
    for col, arr in (
        ("logE_tau", replay["e_log"]),
        ("tau_months", replay["tau_months"]),
        ("gamma_e", replay["gamma_e"]),
        ("solo", replay["solo"]),
    ):
        if col == "solo":
            assert np.array_equal(arr, rev3[col].to_numpy(dtype=bool)), "solo 재현 실패"
        elif col == "tau_months":
            assert np.array_equal(arr, rev3[col].to_numpy(dtype=int)), "tau_months 재현 실패"
        else:
            assert np.allclose(arr, rev3[col].to_numpy(float), rtol=0.0, atol=TOL), f"{col} 재현 실패"
    print("registered 재현 assert 통과: logE_tau/tau_months/solo/gamma_e")


def assert_penalty_invariance(m_env_a: float, m_env_b: float) -> None:
    """lambda=c/m_env 파라미터화에서 벌점항 0.5*lambda^2*m_env^2 항등성을 검산한다."""
    c = np.asarray(ep.C_GRID, dtype=float)
    penalty_a = 0.5 * ((c / m_env_a) ** 2) * (m_env_a**2)
    penalty_b = 0.5 * ((c / m_env_b) ** 2) * (m_env_b**2)
    assert np.allclose(penalty_a, penalty_b, rtol=0.0, atol=1e-15), "벌점항 항등성 실패"
    assert np.allclose(penalty_a, 0.5 * c**2, rtol=0.0, atol=1e-15), "0.5*c^2 검산 실패"
    print(f"페널티 항등 assert 통과: m_env={m_env_a:.6f} vs {m_env_b:.6f}")


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


def apply_envelope_adjustment(
    rev3: pd.DataFrame,
    y_mat: np.ndarray,
    date_rows: list[np.ndarray],
    replay: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """인증 6개에 한해 m_env_j=max(1.3,sqrt(meanY2_pre)) 조정을 적용한다."""
    stage4 = pd.read_csv(DATA / "stage4_envelope_check.csv")[["signalname", "meanY2"]]
    assert stage4["signalname"].is_unique, "stage4 meanY2 signalname 중복"
    mean_y2 = rev3["signalname"].map(stage4.set_index("signalname")["meanY2"])
    assert not mean_y2.isna().any(), "stage4 meanY2 누락"
    m_env_consistent = np.maximum(BASE_M_ENV, np.sqrt(mean_y2.to_numpy(float)))

    certified = rev3["rejected"].to_numpy(dtype=bool)
    certified_names = set(rev3.loc[certified, "signalname"])
    assert len(certified_names) == 6, f"Rev3 인증 수 불일치: {len(certified_names)}"

    to_adjust = certified & (m_env_consistent > BASE_M_ENV + TOL)
    adjusted_names = set(rev3.loc[to_adjust, "signalname"])
    assert adjusted_names == EXPECTED_ADJUSTED, f"조정 대상 불일치: {sorted(adjusted_names)}"

    max_m_env = float(m_env_consistent[to_adjust].max())
    assert_penalty_invariance(BASE_M_ENV, max_m_env)

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
    assert np.array_equal(solo_adj[copy_mask], replay["solo"][copy_mask])
    assert np.allclose(gamma_e_adj[copy_mask], replay["gamma_e"][copy_mask], rtol=0.0, atol=TOL)
    print("adjusted=registered 복사 assert 통과: meanY2 적합 인증 2개 + 비인증 193개")

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
    return out, adj_arrays, certified


def run_adjusted_ebh(out: pd.DataFrame, adj_arrays: dict[str, np.ndarray]) -> int:
    """adjusted 동결값과 달력 tau로 등록순 e-BH를 재판정하고 baseline과 대조한다."""
    reveal = m1.direct_reveal(adj_arrays["e_log"], adj_arrays["tau_cal"])
    base = ebh.baseline_reveal(
        adj_arrays["e_log"],
        adj_arrays["tau_cal"],
        is_null=np.zeros(len(out), dtype=bool),
        alpha=m1.ALPHA,
        J_budget=m1.J_BUDGET,
    )
    assert np.array_equal(base["disc"], reveal["disc"]), "baseline_reveal disc 검산 실패"
    assert np.array_equal(base["disc_step"], reveal["disc_step"]), "baseline_reveal disc_step 검산 실패"
    print("baseline_reveal 대조 assert 통과: disc/disc_step")

    out["ebh_thr_gamma_e_adj"] = reveal["thr_gamma"]
    out["rejected_adj"] = reveal["disc"]
    out["disc_step_adj"] = reveal["disc_step"]
    return int(reveal["k_199"])


def write_appendix(cert: pd.DataFrame) -> None:
    """registered 기준 선정 행에 adjusted envelope-consistent 열을 붙여 appendix를 저장한다."""
    rejected = cert.loc[cert["rejected"]].copy()
    top = cert.sort_values("gamma_e", ascending=False).head(15)
    selected = pd.concat([rejected, top], ignore_index=True)
    selected = selected.drop_duplicates("signalname")
    selected = selected.sort_values(["rejected", "gamma_e"], ascending=[False, False])

    lines = [
        r"% Rev4 M1 -- envelope-consistent adjustment of Rev3 certification",
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        r"\caption{Fixed-$H=120$ e-BH certification with reviewer-requested envelope-consistent columns. "
        r"Registered columns keep the Rev3 protocol ($m_{\rm env}=1.3$, $b_{\rm solo}=4240$, "
        r"$J_{\rm budget}=212$, $\alpha=0.05$). Adjusted columns use envelope-consistent "
        r"$m_{{\rm env},j}=\max(1.3,\sqrt{\text{pre-pub mean }Y^2})$ for the six certified factors; "
        r"the betting-grid penalty $0.5\lambda^2m^2=0.5c^2$ is invariant, so any $\log E$ change is "
        r"entirely bet-shrinkage ($\lambda=c/m_{\rm env}$). Full table: "
        r"\texttt{data/rev4\_m1\_envelope\_adj.csv}.}",
        r"\label{tab:rev3-m1-ebh-cert}",
        r"\begin{tabular}{lrrrrrcrrrrc}",
        r"\toprule",
        r"factor & pub. year & $SR^{raw}_{120}$ & $\sqrt{12}\bar{Y}$ & "
        r"$\log E_\tau$ & $\gamma e$ & rej. & mean $Y^2$ & $m_{{\rm env},j}$ & "
        r"$\log E_{\tau,adj}$ & $\gamma e_{adj}$ & rej. adj. \\",
        r"\midrule",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"{m1.tex_escape(row['signalname'])} & "
            f"{int(row['pub_year'])} & "
            f"{m1.fmt_num(float(row['sr120_raw_ann']))} & "
            f"{m1.fmt_num(float(row['sr120_std_ann']))} & "
            f"{m1.fmt_num(float(row['logE_tau']))} & "
            f"{m1.fmt_gamma(float(row['gamma_e']))} & "
            f"{'yes' if bool(row['rejected']) else 'no'} & "
            f"{m1.fmt_num(float(row['meanY2_pre']))} & "
            f"{m1.fmt_num(float(row['m_env_consistent']))} & "
            f"{m1.fmt_num(float(row['logE_tau_adj']))} & "
            f"{m1.fmt_gamma(float(row['gamma_e_adj']))} & "
            f"{'yes' if bool(row['rejected_adj']) else 'no'} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_tex = REVIEW / "appendix_rev3_m1.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"saved -> {out_tex.relative_to(REPO)}")


def print_summary(
    cert: pd.DataFrame,
    adj_arrays: dict[str, np.ndarray],
    certified: np.ndarray,
    k_199_adj: int,
) -> None:
    """stdout 요구 항목을 등록순 기반으로 출력한다."""
    registered_names = cert.loc[cert["rejected"], "signalname"].tolist()
    adjusted_names = cert.loc[cert["rejected_adj"], "signalname"].tolist()
    flipped = cert.loc[cert["rejected"] & ~cert["rejected_adj"], "signalname"].tolist()

    print(f"registered 인증 수: {len(registered_names)} / 목록: {', '.join(registered_names)}")
    print(f"adjusted 인증 수: {len(adjusted_names)} / 목록: {', '.join(adjusted_names) if adjusted_names else '없음'}")
    print(f"adjusted k_199: {k_199_adj}")

    if flipped:
        print("뒤집힌 팩터 상세:")
        for _, row in cert.loc[cert["signalname"].isin(flipped)].iterrows():
            delta_tau = float(row["logE_tau_adj"] - row["logE_tau"])
            delta_max = float(adj_arrays["max_logE"][row.name] - row["max_logE"])
            cause = "logE 변화 없음, adjusted k 축소로 탈락" if abs(delta_tau) <= TOL else "logE 감소 및 adjusted k 축소로 탈락"
            print(
                f"  {row['signalname']}: ΔlogE_tau={delta_tau:.6f}, "
                f"Δ(max logE)={delta_max:.6f}; 페널티 증가분 0 "
                f"(λ 파라미터화상 항등), 전액 λ 축소분; {cause}"
            )
    else:
        print("뒤집힌 팩터: 없음")

    print("meanY2 초과 인증 4개 변화표:")
    show = cert.loc[adj_arrays["to_adjust"], [
        "signalname",
        "m_env_consistent",
        "logE_tau",
        "logE_tau_adj",
        "solo_adj",
        "gamma_e_adj",
    ]]
    for _, row in show.iterrows():
        print(
            f"  {row['signalname']}: m_env_j={row['m_env_consistent']:.6f}, "
            f"logE_tau {row['logE_tau']:.6f} -> {row['logE_tau_adj']:.6f}, "
            f"solo_adj={bool(row['solo_adj'])}, gamma_e_adj={row['gamma_e_adj']:.6g}"
        )

    kept = cert.loc[certified & ~adj_arrays["to_adjust"], "signalname"].tolist()
    print("meanY2 적합 인증 2개는 adjusted=registered 복사: " + ", ".join(kept))


def main() -> None:
    """Rev4 M1 envelope-consistent 재검증 산출물을 생성한다."""
    cert_base, y_mat, date_rows = m1.build_fixed_h_matrix()
    rev3 = load_rev3_csv()

    replay = replay_registered(cert_base, y_mat, date_rows)
    assert_rev3_reproduction(cert_base, replay, rev3)

    out, adj_arrays, certified = apply_envelope_adjustment(rev3, y_mat, date_rows, replay)
    k_199_adj = run_adjusted_ebh(out, adj_arrays)

    out_csv = DATA / "rev4_m1_envelope_adj.csv"
    out.to_csv(out_csv, index=False)
    write_appendix(out)

    print(f"saved -> {out_csv.relative_to(REPO)}")
    print_summary(out, adj_arrays, certified, k_199_adj)


if __name__ == "__main__":
    main()
