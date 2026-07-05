# Paper A — frontier.py 실데이터 교체 RUN LOG

실행일: 2026-07-02
데이터: OSAP data release 2025.10 (v2.00), 원논문 구현(`op`) 포트폴리오, `port == 'LS'`
코드: `frontier.py` (원본 커밋 85a7806 → placeholder 2개만 교체; 통계 로직 무수정)

## Stage 1 — 데이터 확보

- `openassetpricing` 0.0.2로 다운로드. `dl_port('op')` → 1,226,794행 × 7열 (long format).
- `port == 'LS'` 필터 → 173,302행, **시그널 212개** → `data/osap_LS_v200.csv.gz`
- SignalDoc 331행 (Predictor 212 / Placebo 114 / Drop 5) → `data/SignalDoc.csv`
- `data/DATA_VERSION.txt`: "OSAP data release 2025.10 (v2.00), downloaded 2026-07-02"
- 환경 이슈: Python 3.14 + pandas 2.2.3(wrds가 끌어온 구버전) 조합에서 pandas C 확장
  segfault(exit 139) 발생. **pandas 3.0.3으로 업그레이드해 해결.** Stage 1은 polars 경로로
  다운로드 (`dl_port('op','pandas')`의 polars→pandas 변환도 동일 segfault였음). 데이터 내용 무관.
- macOS python.org 빌드 SSL 인증서 문제 → `SSL_CERT_FILE=$(python3 -m certifi)` 필요.

## Stage 2 — post-registration Sharpe (`stage2_postpub_sharpe.py`)

등록시점 관례: A_j = 출판연도(Year) 12월 말, 윈도우 = Year+1년 1월부터 (lookahead 없음).
윈도우 내 NaN drop, sd는 ddof=1. 산출: `data/osap_postpub_sharpe.csv`.

Sanity check:
- **Predictor 수: 212 / 212 (탈락 0)** — 모든 predictor가 post-pub 데이터 보유.
- sharpe_ann 사분위: **Q1 = 0.083, median = 0.258, Q3 = 0.521.**
  McLean–Pontiff 기준(대략 0.3–0.6)보다 median이 약간 낮음. v2.00은 표본이 2024년까지
  연장되어 post-pub decay가 더 길게 반영된 결과로 해석 — 이상 신호 아님.
- n_j: min = 14, median = 216, max = 612 (개월).
- 수익률 단위: percent 확인. mean_m median = 0.226 (Q1 0.060, Q3 0.454), sd_m median = 3.37.
  월평균이 안내 범위(0.3–0.7)보다 다소 낮은 것도 동일하게 post-pub decay 반영으로 일관됨.

## Stage 3 — frontier 재실행 (M_ENV = 1.3, N_SIM = 10,000, SEED = 0)

교체 diff는 (1) `load_osap_monthly_sharpe()` 본문 → CSV 로더, (2) scatter 라벨의
PLACEHOLDER 문구, 그리고 실행에 필요한 최소 변경(NOISE 환경변수 파라미터화,
savefig 로컬 경로)뿐. oracle 닫힌형·phi·Catoni-mixture MC는 그대로.

### Null sanity (δ = 0, n = 360)

두 케이스 모두 crossing prob **0.00000** ≤ Ville bound α/J = 0.05/212 ≈ **0.00024**. 통과.

### Frontier 표 (연율화 Sharpe, √12 × δ*)

| n (mo) | oracle δ*_ann | Catoni δ*_ann (gaussian) | Catoni δ*_ann (t5) | 팩터 below % (gauss) | 팩터 below % (t5) |
|-------:|--------------:|-------------------------:|-------------------:|---------------------:|------------------:|
|     60 |          2.35 | > 2.77 (grid 상한 초과)  | > 2.77             | 100.0%               | 100.0%            |
|    120 |          1.66 | **1.84**                 | 1.84               | 100.0%               | 100.0%            |
|    240 |          1.18 | 1.28                     | 1.28               | 98.6%                | 98.6%             |
|    360 |          0.96 | 1.04                     | 1.04               | 97.2%                | 97.2%             |

- "팩터 below %" = 해당 n의 Catoni frontier보다 낮은 실현 sharpe_ann을 가진 predictor 비율
  (212개 전체 대비). n=60은 frontier가 탐색 grid(월 δ ≤ 0.8) 위라 전부 below.
- 각 팩터를 자기 n_j 위치의 frontier와 비교(보간)하면 below = **98.6%** (양 케이스 동일).
  frontier.py 자체 출력은 92%인데, 이는 n=60 격자값이 NaN이어서 n_j < 120인 팩터들이
  보간 불능(NaN 비교 → False)으로 빠진 값 — 보수적 하한으로 이해하면 됨.
- **placeholder 기준선과의 비교: n=120 Catoni 연율 SR = 1.84로 기준값(1.84 부근)과 일치.**
  frontier 곡선은 OSAP overlay와 무관하게 MC로만 결정되므로(동일 SEED/파라미터) 예상대로
  변화 없음. 조사·중단 조건 해당 없음.
- t5(fat tail, 단위분산) 케이스도 gaussian과 사실상 동일 (δ* 차이 ≤ 0.001) — Catoni
  influence의 강건성과 일관.

### 산출물

- `frontier.png` (gaussian, 실제 OSAP overlay) / `frontier_t5.png`
- `data/osap_postpub_sharpe.csv` (signalname, pub_year, n_j, mean_m, sd_m, sharpe_m, sharpe_ann)
- 실행 로그: `run_gaussian.log`, `run_t5.log`

## t5 스위치 검증 (2026-07-02)

`draw_noise` 직후 `scipy.stats.kurtosis`(excess) print를 임시 추가, N_SIM=1,000으로
gaussian/t5 실행 (SEED=0). draw는 `frontier_for_n` 호출마다 생성되므로 케이스당 5줄
(n=60/120/240/360 + null sanity용 n=360 1회). 실제 출력 그대로:

```
===== NOISE=gaussian, N_SIM=1000 =====
NOISE=gaussian, excess kurtosis = 0.03
NOISE=gaussian, excess kurtosis = 0.00
NOISE=gaussian, excess kurtosis = -0.01
NOISE=gaussian, excess kurtosis = 0.00
NOISE=gaussian, excess kurtosis = 0.01
===== NOISE=t5, N_SIM=1000 =====
NOISE=t5, excess kurtosis = 3.53
NOISE=t5, excess kurtosis = 6.23
NOISE=t5, excess kurtosis = 8.74
NOISE=t5, excess kurtosis = 6.41
NOISE=t5, excess kurtosis = 5.40
```

- **gaussian ≈ 0, t5 ≈ 6 (이론값 6/(df−4) = 6) — NOISE 스위치 정상 동작 확인.**
  t5의 draw별 산포(3.5~8.7)는 t(5)의 표본 kurtosis 추정량 분산이 매우 큰 데서 오는
  정상 현상 (8차 모멘트 발산; 표본이 큰 n=120~360에서 6.2/6.4/5.4로 6에 수렴).
- 따라서 본문 frontier 표에서 "t5가 gaussian과 소수점까지 사실상 동일(δ* 차이 ≤ 0.001)"은
  버그가 아니라 결과다: Catoni 절차의 검정력이 사실상 처음 두 모멘트로 결정된다는
  robustness 진술로 §5.6에 실을 수 있음.
- 검증 후 print 제거, N_SIM=10,000 원복, 그림은 커밋본(N_SIM=10k)으로 복원 —
  frontier.py는 커밋 2d7834c와 동일(git diff 없음).

### 결론 (그림 해석)

실제 OSAP v2.00 overlay에서 팩터의 97~100%가 Catoni frontier 아래: 실현 post-pub Sharpe
(median 연 0.26)로는 J=212, α=0.05, power 50% 설정에서 30년 표본으로도 대부분 검출 불가.
placeholder(연 SR 0.4±0.25 가정)보다 실제 데이터가 더 낮은 Sharpe를 보여 논문 내러티브
("Day-1 detectability는 현실적으로 거의 불가능")가 오히려 강화됨.

## Stage 4 — envelope 진단 (2026-07-05, `stage4_envelope_check.py`)

각 팩터의 **pre-publication 데이터만** 사용 (A_j = Year 12월 말 이전). v_{t-1} = 직전
36개월 trailing sd (ddof=1, 최소 24개월), v_min = 자기 pre-pub trailing-sd 시계열의
하위 5% 분위수 (팩터별 floor, lookahead 없음). Y_t = ret_t / max(v_{t-1}, v_min).

- 212개 팩터 전부 block mean(Y²) 산출 가능 (pre-pub 표본 부족 제외 0).
- block mean(Y²) 분위수: 10% 1.126 / 25% 1.181 / 50% 1.258 / 75% 1.376 /
  90% 1.510 / 95% 1.788 / 99% 2.118.
- coverage (기준 "mean(Y²) ≤ m²인 팩터 ≥ 90%"):
  m=1.3 → **93.9%** (199/212) / m=1.5 → 99.1% / m=1.7 → 100.0%.
- **채택 m_env = 1.3 (최소 후보가 기준 충족). 기존 frontier의 M_ENV=1.3과 동일 →
  frontier 재실행 불필요.** 상세: `data/stage4_envelope_check.csv`, `run_stage4.log`.

## Stage 5 — null sanity gate (2026-07-05, `stage5_null_sanity.py`)

N_SIM=100,000, n=360, δ=0. e-process 로직은 frontier.py 커밋 2d7834c와 동일
(파라미터 하드코딩, 메모리용 배치 처리만 추가). 기준: crossing rate ≤ 0.000381
(= α/J + 3×MC 표준오차).

- gaussian: **0/100,000 = 0.000000 → PASS**
- t5:       **0/100,000 = 0.000000 → PASS**

## Stage 6 — grid 확장 + margin 산출 (2026-07-05)

### Grid 확장 (frontier.py: DELTA_GRID 상한 0.8 → 1.2, 간격 0.04 유지)

허용 변경(탐색 grid 상한)만 수정: `np.linspace(0.0, 0.8, 21)` → `np.linspace(0.0,
1.2, 31)`. 기존 grid점이 보존되고 CRN 추첨 순서 불변이라 n=120/240/360 값은
정의상 동일해야 하며, 실제로도 동일 확인. N_SIM=10,000, SEED=0.

| n (mo) | oracle δ*_ann | Catoni δ*_ann (gaussian) | Catoni δ*_ann (t5) |
|-------:|--------------:|-------------------------:|-------------------:|
|     60 |          2.35 | **2.86** (신규, δ*=0.826) | 2.86 (δ*=0.825)    |
|    120 |          1.66 | 1.84 (불변)              | 1.84               |
|    240 |          1.18 | 1.28 (불변)              | 1.28               |
|    360 |          0.96 | 1.04 (불변)              | 1.04               |

- 기존 세 값(1.84/1.28/1.04) 변동 없음 → STOP 조건 미해당.
- t5 vs gaussian δ* 차이 ≤ 0.001 — Stage 3 robustness 결론 유지.
- frontier.py 자체 below% 출력 99%는 209/212 = 98.6%의 소수점 반올림
  (n=60이 채워지면서 기존 92% 표시의 NaN 보간 문제 해소). 수동 검산으로
  209/212 확인, frontier 위 3개 = AnalystRevision, AnnouncementReturn, DivYieldST.
- frontier.png / frontier_t5.png 갱신 (n=60 점 포함).

### Margin 산출 (`stage6_margin.py`, `data/osap_postpub_sharpe.csv` 컬럼 추가)

- 추가 컬럼: `frontier_at_nj` (60≤n_j≤360 선형 보간; n_j>360은 √n 스케일 외삽
  frontier(360)×√(360/n_j); n_j<60도 동일 방식 위쪽 외삽 — 해당 팩터는 전부
  censored), `margin` (= sharpe_ann − frontier_at_nj), `censored` (= n_j < 120).
- **censored = 13개** (Activism1/2, CBOperProf, ConvDebt, Governance, OperProfRD,
  PatentsRD, ReturnSkew, ReturnSkew3F, TrendFactor, dCPVolSpread, dVolCall,
  dVolPut) — right-censored 케이스로 below/fail 집계에서 제외.
- **matured 199개 margin 분위수 (연율)**: 10% −1.587 / 25% −1.365 / **50% −1.082** /
  **75% −0.802** / 90% −0.568.
- 예비 검산치(median ≈ −1.08, Q3 ≈ −0.80, frontier 위 3개)와 일치 — 이상 없음.
