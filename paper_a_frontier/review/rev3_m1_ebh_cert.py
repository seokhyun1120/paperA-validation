"""Rev3 M1 — fixed-H=120 실제 e-BH 인증.

n_j >= 120인 matured 팩터의 post-publication 첫 120개월 표준화 수익률로
사전등록 e-process를 실행하고, 등록순 reveal + e-BH 결과를 재현 검산한다.
실행: ../../.venv/bin/python3 review/rev3_m1_ebh_cert.py
산출: data/rev3_m1_ebh_cert.csv, review/appendix_rev3_m1.tex
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
REVIEW = REPO / "review"
sys.path.insert(0, str(REPO))
from sim import eprocess as ep, ebh  # noqa: E402  frontier 커널 단일 출처

H = 120
WINDOW = 36
MIN_OBS = 24
M_ENV = 1.3
ALPHA = 0.05
J_BUDGET = 212
EXPECTED_MATURED = 199


def tex_escape(value: object) -> str:
    """LaTeX 표 셀에 들어갈 문자열의 특수문자를 보호한다."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(value))


def month_index(ts: pd.Timestamp) -> int:
    """달력 월을 year*12+month 정수로 바꾼다."""
    return int(ts.year * 12 + ts.month)


def load_panel() -> pd.DataFrame:
    """LS 수익률과 Predictor 출판연도를 결합한 전체 시계열을 만든다."""
    ls = pd.read_csv(DATA / "osap_LS_v200.csv.gz", parse_dates=["date"])
    doc = pd.read_csv(DATA / "SignalDoc.csv")
    pred = (
        doc.loc[doc["Cat.Signal"] == "Predictor", ["Acronym", "Year"]]
        .rename(columns={"Acronym": "signalname", "Year": "pub_year"})
    )
    assert pred["signalname"].is_unique, "Predictor Acronym 중복"
    panel = ls.merge(pred, on="signalname", how="inner")
    return panel.dropna(subset=["ret"]).sort_values(["signalname", "date"])


def build_fixed_h_matrix() -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray]]:
    """post-pub 첫 120개월의 raw SR, 표준화 Y, 달력 날짜를 구성한다."""
    panel = load_panel()
    vmin = pd.read_csv(DATA / "stage4_envelope_check.csv")[["signalname", "v_min"]]
    assert vmin["signalname"].is_unique, "stage4 v_min signalname 중복"
    vmin_s = vmin.set_index("signalname")["v_min"]
    m4_sr = pd.read_csv(DATA / "m4_pseudolive_sr120.csv").set_index("signalname")

    rows: list[dict[str, object]] = []
    y_rows: list[np.ndarray] = []
    date_rows: list[np.ndarray] = []
    for signalname, g in panel.groupby("signalname", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        pub_year = int(g["pub_year"].iloc[0])
        assert (g["pub_year"] == pub_year).all(), f"{signalname}: pub_year 불일치"
        post = (g["date"] >= pd.Timestamp(year=pub_year + 1, month=1, day=1)).to_numpy()
        post_pos = np.flatnonzero(post)
        if len(post_pos) < H:
            continue

        r_all = g["ret"].to_numpy(dtype=float)
        v_all = (
            pd.Series(r_all)
            .rolling(WINDOW, min_periods=MIN_OBS)
            .std(ddof=1)
            .shift(1)
            .to_numpy()
        )
        pos = post_pos[:H]
        v_h = v_all[pos]
        if np.isnan(v_h).any():
            print(f"v_{{t-1}} NaN 포함 팩터: {signalname}")
            raise RuntimeError("post-pub 첫 120개월 표준화에 예측가능 변동성 결측이 있음")
        if signalname not in vmin_s.index:
            raise RuntimeError(f"stage4 v_min 누락: {signalname}")
        v_min = float(vmin_s.loc[signalname])
        if np.isnan(v_min):
            raise RuntimeError(f"stage4 v_min NaN: {signalname}")

        r_h = r_all[pos]
        sr_raw = float(np.sqrt(12.0) * r_h.mean() / r_h.std(ddof=1))
        if signalname not in m4_sr.index:
            raise RuntimeError(f"m4_pseudolive_sr120 누락: {signalname}")
        expected = float(m4_sr.loc[signalname, "sr120_ann"])
        assert np.isclose(sr_raw, expected, rtol=0.0, atol=1e-9), (
            f"{signalname}: raw SR cross-check 실패 {sr_raw} vs {expected}"
        )

        y = r_h / np.maximum(v_h, v_min)
        rows.append(
            {
                "signalname": signalname,
                "pub_year": pub_year,
                "sr120_raw_ann": sr_raw,
                "mean_Y": float(y.mean()),
                "sr120_std_ann": float(np.sqrt(12.0) * y.mean()),
                "array_idx": len(y_rows),
            }
        )
        y_rows.append(y)
        date_rows.append(g.loc[pos, "date"].to_numpy())

    cert = pd.DataFrame(rows)
    assert len(cert) == EXPECTED_MATURED, f"matured 팩터 수 불일치: {len(cert)}"
    assert set(cert["signalname"]) == set(m4_sr.index), "M4 fixed-H 대상 집합 불일치"

    cert = cert.sort_values(["pub_year", "signalname"], kind="mergesort").reset_index(drop=True)
    order = cert.pop("array_idx").to_numpy(dtype=int)
    y_mat = np.vstack([y_rows[i] for i in order])
    ordered_dates = [date_rows[i] for i in order]
    return cert, y_mat, ordered_dates


def direct_reveal(e_log: np.ndarray, tau_cal: np.ndarray) -> dict[str, np.ndarray | int]:
    """sim/ebh.py baseline_reveal 수식 그대로 등록순 reveal을 직접 계산한다."""
    j = len(e_log)
    b_m = np.maximum.accumulate(tau_cal)
    log_thr = np.log(J_BUDGET / (ALPHA * np.arange(1, j + 1)))
    disc = np.zeros(j, dtype=bool)
    disc_step = np.full(j, -1, dtype=int)
    disc_time = np.full(j, -1, dtype=int)
    thr_gamma = np.full(j, np.nan, dtype=float)
    k_path = np.zeros(j, dtype=int)

    for m in range(1, j + 1):
        prefix = e_log[:m]
        ks = np.arange(1, m + 1)
        counts = (prefix[:, None] >= log_thr[:m][None, :]).sum(axis=0)
        ok = counts >= ks
        if ok.any():
            k_m = int(ks[ok].max())
            in_r = prefix >= log_thr[k_m - 1]
            assert not (disc[:m] & ~in_r).any(), "R_m nested 위반"
            new = in_r & ~disc[:m]
            idx = np.nonzero(new)[0]
            disc[idx] = True
            disc_step[idx] = m
            disc_time[idx] = b_m[m - 1]
            thr_gamma[idx] = 1.0 / (ALPHA * k_m)
        else:
            k_m = 0
            assert not disc[:m].any(), "k_m=0에서 기존 발견 철회 발생"
        k_path[m - 1] = k_m

    k_final = int(k_path[-1])
    final_thr = np.nan if k_final == 0 else 1.0 / (ALPHA * k_final)
    thr_gamma[~disc] = final_thr
    return {
        "disc": disc,
        "disc_step": disc_step,
        "disc_time": disc_time,
        "thr_gamma": thr_gamma,
        "k_path": k_path,
        "k_199": k_final,
    }


def run_protocol(
    cert: pd.DataFrame, y_mat: np.ndarray, date_rows: list[np.ndarray]
) -> tuple[pd.DataFrame, int]:
    """e-process freeze와 등록순 e-BH reveal을 실행하고 baseline 구현과 대조한다."""
    assert y_mat.shape == (EXPECTED_MATURED, H), f"Y 행렬 shape 불일치: {y_mat.shape}"
    log_e = ep.log_e_path(y_mat, m_env=M_ENV)
    log_b = ep.log_b_solo(1.0 / J_BUDGET)
    assert log_b == ep.frontier.LOG_B, "log_b_solo(1/212)와 frontier.LOG_B 불일치"
    log_e_frozen, tau = ep.freeze_at_crossing(log_e, log_b)
    solo = tau >= 0

    tau_cal = np.empty(len(cert), dtype=int)
    tau_months = np.empty(len(cert), dtype=int)
    for i, dates in enumerate(date_rows):
        idx = int(tau[i]) if solo[i] else H - 1
        tau_cal[i] = month_index(pd.Timestamp(dates[idx]))
        tau_months[i] = idx + 1 if solo[i] else H

    e_log = log_e_frozen[:, -1]
    reveal = direct_reveal(e_log, tau_cal)
    base = ebh.baseline_reveal(
        e_log,
        tau_cal,
        is_null=np.zeros(len(e_log), dtype=bool),
        alpha=ALPHA,
        J_budget=J_BUDGET,
    )
    assert np.array_equal(base["disc"], reveal["disc"]), "baseline_reveal disc 검산 실패"

    out = cert.copy()
    out["max_logE"] = log_e.max(axis=1)
    out["logE_tau"] = e_log
    out["tau_months"] = tau_months
    out["solo"] = solo
    out["gamma_e"] = np.exp(e_log) / J_BUDGET
    out["ebh_thr_gamma_e"] = reveal["thr_gamma"]
    out["rejected"] = reveal["disc"]
    out["disc_step"] = reveal["disc_step"]
    return out, int(reveal["k_199"])


def fmt_num(value: float, digits: int = 3) -> str:
    """표와 로그용 고정 소수점 포맷."""
    if np.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_gamma(value: float) -> str:
    """gamma e 값은 크기 차이가 커서 유효숫자로 표시한다."""
    if np.isnan(value):
        return "NA"
    return f"{value:.3g}"


def write_appendix(cert: pd.DataFrame) -> None:
    """기각 팩터와 동결 gamma_e 상위 팩터의 appendix 표를 저장한다."""
    rejected = cert.loc[cert["rejected"]].copy()
    top = cert.sort_values("gamma_e", ascending=False).head(15)
    selected = pd.concat([rejected, top], ignore_index=True)
    selected = selected.drop_duplicates("signalname")
    selected = selected.sort_values(["rejected", "gamma_e"], ascending=[False, False])

    lines = [
        r"% Rev3 M1 -- fixed-H=120 e-BH certification",
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        r"\caption{Fixed-$H=120$ e-BH certification. Protocol parameters: "
        r"$m_{\rm env}=1.3$, $b_{\rm solo}=4240$, $J_{\rm budget}=212$, "
        r"$\alpha=0.05$, registration-order reveal; 13 immature factors remain "
        r"unrevealed but included in the budget. Full table: "
        r"\texttt{data/rev3\_m1\_ebh\_cert.csv}.}",
        r"\label{tab:rev3-m1-ebh-cert}",
        r"\begin{tabular}{lrrrrrrccc}",
        r"\toprule",
        r"factor & pub. year & $SR^{raw}_{120}$ & $\sqrt{12}\bar{Y}$ & "
        r"$\max_t \log E$ & $\log E_\tau$ & tau & solo & $\gamma e$ & rejected \\",
        r"\midrule",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"{tex_escape(row['signalname'])} & "
            f"{int(row['pub_year'])} & "
            f"{fmt_num(float(row['sr120_raw_ann']))} & "
            f"{fmt_num(float(row['sr120_std_ann']))} & "
            f"{fmt_num(float(row['max_logE']))} & "
            f"{fmt_num(float(row['logE_tau']))} & "
            f"{int(row['tau_months'])} & "
            f"{'yes' if bool(row['solo']) else 'no'} & "
            f"{fmt_gamma(float(row['gamma_e']))} & "
            f"{'yes' if bool(row['rejected']) else 'no'} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = REVIEW / "appendix_rev3_m1.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"saved -> {out.relative_to(REPO)}")


def main() -> None:
    """M1 인증 산출물을 생성한다."""
    cert_base, y_mat, date_rows = build_fixed_h_matrix()
    cert, k_199 = run_protocol(cert_base, y_mat, date_rows)
    out_csv = DATA / "rev3_m1_ebh_cert.csv"
    cert.to_csv(out_csv, index=False)
    write_appendix(cert)

    rejected = cert.loc[cert["rejected"], "signalname"].tolist()
    print(f"saved -> {out_csv.relative_to(REPO)}")
    print(f"인증(기각) 수: {len(rejected)} / {len(cert)}")
    print("인증 목록: " + (", ".join(rejected) if rejected else "없음"))
    print(f"k_199: {k_199}")
    print(f"solo 도달 수: {int(cert['solo'].sum())} / {len(cert)}")
    print("max logE 상위 5:")
    for _, row in cert.sort_values("max_logE", ascending=False).head(5).iterrows():
        print(
            f"  {row['signalname']}: max_logE={row['max_logE']:.3f}, "
            f"logE_tau={row['logE_tau']:.3f}, gamma_e={row['gamma_e']:.3g}"
        )


if __name__ == "__main__":
    main()
