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

## §8.1 적응탐색 시뮬레이터 (2026-07-06, `sim/`, 엔진 커밋 9d091ea)

ground truth가 있는 세계에서 프로토콜 전체(적응 생성 → 등록 → e-process →
online e-BH)의 validity/power 실측. 잠금값(§13) 준수: α=0.05, m_env=1.3
(E4에서만 {1.2,1.5} 변형), D_j=A_j+120 primary, γ_j=1/J_budget 균등,
Catoni-mixture primary, 월간, gaussian primary + t5, b_solo=1/(αγ).

### 아키텍처·설계 상수

- 모듈 5개: `sim/world.py`(score 생성, 공통요인 ρ), `sim/searcher.py`(적응 탐색:
  L=100 후보, W=36 in-sample 윈도우, honest/adversarial), `sim/registry.py`
  (append-only 원장), `sim/eprocess.py`(**frontier.py 커널 import — 복사 없음**;
  import 중 Figure.savefig만 일시 무력화해 그림 덮어쓰기 방지, 파일 무수정),
  `sim/ebh.py`(online e-BH). 실험 드라이버 `sim/run_E{1..5}.py`.
- **J_budget = 212** (설계 선택): b_solo = 212/0.05 = 4240 = frontier B_SOLO와
  일치 → frontier δ* 캘리브레이션이 E2에 정확히 이전됨. γ=1/212.
- e-BH k_m: 표준 e-BH k_t = max{k : e_(k) ≥ J_budget/(αk)}, k_t=0 → R_t=∅,
  발견 집합은 누적(union), e-값은 solo 도달/데드라인/포기 시 동결.
  주의: proofs final §4 원문 부재로 표준형으로 구현 — 누적 union은 FDP를 크게
  잡는 보수적 방향 (validity 측정에 안전).
- SEED 체계: `default_rng([20260706, exp_id, cell_keys..., run_idx])`,
  코드표 noise{g:0,t5:1}, behavior{hon:0,adv:1}. CRN: E1 세계 잡음은 s_reg/
  behavior 제외, E2는 배수 제외, E3는 전 셀 공유.
- 총 러닝타임 ~6분 (8시간 한도 내, N_RUN 축소 불필요).

### 게이트 1 — 회귀 테스트 (`sim/test_eprocess.py`) : PASS

frontier.frontier_for_n(n=120, 360)의 crossing prob 31개 grid점을 동일 rng
상태·동일 잡음에서 sim.eprocess.log_e_path로 재현 → **np.array_equal
(bit-exact, 허용오차 0)**, δ* 보간값 동일, log_b_solo(1/212)=LOG_B 동일,
freeze rule 불변식 통과.

### E1 — Validity (16셀 × N_RUN=2,000) : **PASS**

전부-null + 적응 등록 + peeking/조기중단. honest 평균 등록 ~114-132건
(파생 ~27-32), adversarial은 예산 212건 소진(클론+뮤턴트 ~143-146, 포기
~199-202건 — 공격적 optional stopping). **전 셀 empirical SupFDR = 0.00000**
(32,000런 중 발견 0건; max logE = 6.34 < log b = 8.35) ≤ 합격선 α+3×SE = 0.0646.
노이즈(g/t5)·ρ(0/0.3)·selection 강도(0.15/0.30)·행동(honest/adversarial) 전부
무관하게 validity 유지 — 적응 등록의 selection bias가 post-A_j e-process를
오염시키지 않음을 실측. `sim_results/E1.parquet`, `sim_E1_supfdr.png`.

### E2 — Power vs frontier (N=10,000/셀) : **PASS**

| 셀 | 0.5× | **1.0× (기대 50%)** | 1.5× | TTD 중앙값@1.0× |
|---|---:|---:|---:|---:|
| gaussian D=60  | 0.10% | **49.2%** | 99.9% | 52mo |
| gaussian D=120 | 0.20% | **50.1%** | 99.9% | 98mo |
| gaussian D=240 | 0.21% | **51.0%** | 99.8% | 187mo |
| t5 D=120       | 0.15% | **50.5%** | 99.9% | 98mo |

1.0×에서 전부 50%±1%p (허용 ±10%p) — frontier가 β=0.5 정의와 정확히 정합하는
설계도구임을 확인. δ*는 frontier.catoni 정확값 사용 (t5 셀도 gaussian δ* 공용,
Stage 3의 δ* 차이 ≤0.001 근거). `sim_results/E2.parquet`, `sim_E2_power.png`.

### E4 — envelope 미스펙 스트레스 (경험적)

(2026-07-06 2차 작업 Task 4로 라벨 재정의: α·ξ_var는 **참조선(reference line)**
이며 Corollary 2의 상계가 아님 — 해당 정리는 지속적 분산 위반을 커버하지 않음.
PASS/FAIL 게이트 제거, empirical degradation curve로 재명명. 코드 로직·seed·
숫자는 1차 실행과 동일.)

전부-null, Var(Y) = ξ_var·m_env_reg², J=212 동시 등록, D=120, gaussian, N_RUN=2,000.

| ξ_var | m=1.2 | m=1.3 | m=1.5 | 참조선 α·ξ_var | 비고 |
|---|---:|---:|---:|---:|---|
| 1.0 | — | 0.0030 | — | 0.050 | 기준셀 |
| 1.2 | 0.0105 | 0.0130 | 0.0080 | 0.060 | 참조선 내 |
| 1.5 | 0.0510 | 0.0480 | 0.0515 | 0.075 | 참조선 내 |
| 2.0 | **0.2710** | **0.2615** | **0.2685** | 0.100 | **참조선 초과** |

- **ξ_var=2.0에서 실현 SupFDR ≈ 0.26-0.27 (참조선 0.10의 ~2.6배).** 코드
  아티팩트 아님 판단 근거:
  (1) 기준셀 ξ=1.0은 0.003으로 정상 (E1·Stage 5 null gate와 일관),
  (2) Y=m√ξ·ε, λ=c/m 구조상 m_env_reg가 정확히 소거되는 스케일 불변성이
  이론 예측인데 실측 세 값(0.271/0.262/0.269)이 MC 오차 내 일치,
  (3) 전략당 crossing율 ~0.17%로 개별 경로 물리와 정합 — 212개 union이 27%를 만듦.
- 해석: Catoni 증분의 supermartingale 성질은 E[Y²]≤m_env²에 의존하는데, 지속적
  ξ_var배 위반 시 E[E_n] 상계가 exp(nλ²m²(ξ_var-1)/2)로 n에 따라 증가 → ξ_var=2,
  n=120에서 Ville-형 보장이 성립할 이유가 없음 (그래서 α·ξ_var는 참조선일 뿐).
  ξ_var≤1.5까지는 degradation이 참조선 내 — envelope 여유가 1.5배 이내 위반까지는
  실무적으로 안전. `sim_results/E4.parquet`, `sim_E4_stress.png`.

### E3 — Alpha decay hard-wall (N=10,000/셀, gaussian)

δ_{A+s} = δ0·ρ_d^(s-1), 누적 edge = δ0/(1-ρ_d) = c×W0, W0 = δ*(120)·120 = 63.7.

- 전이 선명 (예: ρ_d=0.98에서 c=0.5→1.5 구간 검출률 0.2%→99.8%) — "탐지 가능
  ⟺ 누적 edge > 문턱" 경계 실측 확인.
- 경험적 벽(50% 지점): ρ_d=0.99 → 87.5 (1.37×W0) / 0.98 → 62.3 (0.98×W0) /
  0.95 → 47.5 (0.75×W0). ρ_d=0.99의 벽이 높은 것은 D=120 내 실현 edge가
  전체의 69.9%뿐이기 때문 (87.5×0.699=61.2 ≈ W0). D 내 실현 edge 기준으로는
  벽이 0.75~0.96×W0로 수렴 — 빠른 decay(front-loading)일수록 필요 총 edge 감소.
  `sim_results/E3.parquet`, `sim_E3_wall.png`.

### E5 — Queue/timing 3항 분해 (T=360, 수용 1건/월, N_RUN=500/셀)

포아송 도착 λ_arr, 50% alt (δ=δ*(120)), FIFO 큐, 예산 212.

| λ_arr | W_queue | T_grow | T_order | e-BH 조기발견 | k≥2 전용 | 예산 소진(중앙값) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.5mo | 91.7mo | −29.4mo | 45.9% | 53.1% | 357mo |
| 1.0 | 6.8mo | 92.2mo | −34.5mo | 51.0% | 48.4% | 221mo |
| 2.0 | 52.8mo | 92.1mo | −34.4mo | 51.5% | 47.9% | 211mo |

- 3항 분해 재현: 등록 대기(용량 제약 시 급증) + e-process 성장(λ 무관 ~92mo)
  + reveal/다중성 항(T_order ≤ 0 — online e-BH가 동시 활성 e-값 덕에 solo
  도달보다 ~30mo 조기 발견; 발견의 ~절반은 solo 미도달 상태에서 k≥2로 발견).
- 혼합 세계 FDR도 미미 (sup FDP 평균 ≤ 0.0002). `sim_results/E5*.parquet`,
  `sim_E5_timing.png`.

### 종합

- 게이트: 회귀 PASS → E1 PASS → E2 PASS. E4는 경험적 degradation curve로 보고
  (ξ_var≤1.5 참조선 내, ξ_var=2.0 초과 — 라벨 재정의는 2026-07-06 2차 작업 Task 4,
  코드/freeze/normalization 점검 완료). E3·E5는 측정 실험으로 완료.
- frontier.py 무수정 유지 (커널은 import, 회귀 테스트 bit-exact).
- 참고: `sim_results/*.parquet`와 `run_simE*.log`는 repo .gitignore 정책
  (*.parquet, run_*.log — 재생성 가능 산출물 커밋 금지)에 따라 비추적. 셀별
  SEED 체계 + 엔진 커밋(9d091ea) 고정이므로 `python3 -m sim.run_E{1..5}`로
  bit-동일 재생성 가능.

## §8.1 잔여 작업 2차 (2026-07-06, 엔진 커밋 b2656c1 기준 실행)

frontier.py 무수정 유지, 회귀 테스트(bit-exact) 재확인 통과. SEED 체계는 1차와
동일 (`default_rng([20260706, exp_id, ...])`, CRN 재사용).

### Task 1 — baseline reveal 구현 (`sim/ebh.py` 확장)

논문 baseline "동결 e-value의 등록순 reveal"을 `baseline_reveal()`로 구현:
e_j = E_{j,τ_j} (solo/데드라인/포기 중 최선착 동결), B_m = max_{i≤m} τ_i (FIFO),
k_m = max{k≤m : #{j≤m : γ_j e_j ≥ 1/(αk)} ≥ k} (proofs final §4), k_m=0 → R_m=∅.
R_m nested는 매 스텝 assert로 강제 (전체 실행에서 위반 0). 기존 live 변형
(`online_ebh`, 진행 중 e-process에 매월 e-BH)은 secondary로 보존.

### Task 2 — E1 재실행 (baseline reveal) + 비교군 2종 : **PASS**

동일 세계·CRN, 16셀 × 2,000런. 합격선 α + 3×SE(p=α) = 0.0646.

| 절차 | empirical SupFDR (16셀 범위) |
|---|---|
| **full protocol (baseline reveal)** | **전 셀 0.00000** (발견 0/32,000런, 최대 동결 log e = 6.15 < 8.35) → PASS |
| 비교군1 naive full-history t-test | 0.858 – 1.000 (런당 평균 허위발견 7.8~18.0건) |
| 비교군2 fresh-data 반복 t-검정 (peeking) | **전 셀 1.000** (런당 평균 30~58건) |

- naive가 1.000 미만인 셀은 전부 ρ=0.3 (공통요인이 운을 상관시켜 "전부 불운"인
  런 존재). fresh-peek은 uncorrected 반복 검정이라 예외 없이 1.0.
- 그림 `sim_E1_supfdr.png`를 3절차 비교로 교체, `E1.parquet` 교체
  (컬럼: sup_fdp, naive_fdp/naive_n_disc, fresh_fdp/fresh_n_disc, ...).
- naive 창 = [0, min(D_j, T-1)] (포기 무관 — naive 연구자는 abandon하지 않음),
  파생 전략은 잠재 시계열 전체 = 변형 전략의 풀히스토리 백테스트. fresh는
  post-A_j 누적 t, 최소 6개월.

### Task 3 — E5 재설계 (v4 §5.8, baseline reveal) 

R_MAX 제거(도착 즉시 등록, A_j = arrival), λ_arr ∈ {0.5,1,2}, 50% alt
(δ=δ*(120)), J_budget=212, D=A+120, T=360, N_RUN=500/셀. 분해 전 항 ≥ 0 assert 통과.

| λ_arr | T_freeze 평균[중앙] | W_fifo | T_unlock | 합계 | alt 검출률 | k≥2 전용 | 예산소진(중앙) | supFDP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 103.2 [111] | 10.0 [0] | 5.6 [0] | 118.9 | 76.6% | 52.3% | 356mo | 0.00005 |
| 1.0 | 107.5 [119] | 11.7 [0] | 3.6 [0] | 122.8 | 96.0% | 48.1% | 212mo | 0.00000 |
| 2.0 | 107.4 [118] | 12.2 [1] | 1.7 [0] | 121.3 | 95.9% | 47.6% | 106mo | 0.00007 |

- T_freeze가 지배 (발견의 ~절반이 solo 미도달 → 데드라인 120mo 동결 후 k≥2로
  발견되므로 중앙값이 111~119mo). W_fifo는 등록순 prefix 동결 대기, T_unlock은
  잔여 문턱 완화 대기 — 둘 다 평균 한 자릿수~10mo, 중앙값 0.
- live 변형 대비 한 줄: live(1차)는 진행 중 e-값에 매월 e-BH를 적용해 평균
  발견 지연이 ~63-111mo(순)로 더 빨랐음 — baseline reveal은 동결·등록순 규율의
  대가로 평균 ~119-123mo. `sim_E5_timing.png`·E5 parquet 교체.

### Task 4 — E4 라벨 수정

"theory bound α·ξ" → "reference line α·ξ_var (not the Corollary-2 bound)",
PASS/FAIL 게이트 제거, empirical degradation curve로 재명명 (본 로그 E4 섹션
제목·표 라벨도 수정). 코드 로직·seed 불변 — 재실행 결과 1차와 전 셀 동일 확인
(0.0030/0.0105/.../0.2685). `sim_E4_stress.png`·E4.parquet 라벨만 갱신.

### Task 5 — OSAP haircut sensitivity (`stage7_haircut_sensitivity.py`)

월별 hurdle h = (1+a)^(1/12)−1 (percent 단위 ×100), sd 불변이므로 기존
mean_m/sd_m/frontier_at_nj에서 직접 재계산. frontier·A_j 관례 동일.

| a (연율) | h (%/월) | below-frontier % (212) | margin 중앙값 (matured 199) |
|---:|---:|---:|---:|
| 0 (primary) | 0 | 98.6% | −1.082 |
| 0.005 | 0.0416 | 98.6% | −1.125 |
| 0.010 | 0.0830 | 99.1% | −1.193 |

haircut을 줘도 below %·margin이 사실상 불변/악화 방향 — "Day-1 detectability
거의 불가" 내러티브는 hurdle grid에 강건. `data/stage7_haircut_sensitivity.csv`.

## 심사 대응 실험 R1·R2 (2026-07-06, 엔진 커밋 5b9cbac 기준 실행)

frontier.py 무수정, 기존 sim/ 재사용. SEED: R1은 `[20260706, 6, ...]`
(세계 잡음은 (rho, run)만 — alpha/behavior 축 CRN 공유), R2는 E1과 동일
seed 스트림으로 세계·등록 집합을 그대로 재생성.

### R1 — validity mechanism check (M2 대응) : **PASS**

all-null, E1 searcher 재사용, J_budget=20, γ=1/20, α ∈ {0.2, 0.5}
(b_solo = 100, 40), D=120, gaussian, s_reg=0.15, N_RUN=2,000/셀.

| α | ρ | behavior | SupFDR | 발견률(런) | 거짓발견 합 | 합격선 α+3SE |
|---:|---:|---|---:|---:|---:|---:|
| 0.2 | 0.0 | honest | 0.0000 | 0.00% | 0 | 0.2268 |
| 0.2 | 0.0 | adversarial | 0.0000 | 0.00% | 0 | 0.2268 |
| 0.2 | 0.3 | honest | 0.0005 | 0.05% | 1 | 0.2268 |
| 0.2 | 0.3 | adversarial | 0.0000 | 0.00% | 0 | 0.2268 |
| 0.5 | 0.0 | honest | 0.0025 | 0.25% | 5 | 0.5335 |
| 0.5 | 0.0 | adversarial | 0.0025 | 0.25% | 6 | 0.5335 |
| 0.5 | 0.3 | honest | 0.0025 | 0.25% | 5 | 0.5335 |
| 0.5 | 0.3 | adversarial | 0.0015 | 0.15% | 3 | 0.5335 |

- 경계를 낮추면(b_solo=40) 파이프라인이 실제로 거짓발견을 생산함(총 20건)을
  확인 — E1의 SupFDR=0이 "발견이 원천 불가능해서"가 아니라 경계·예산 규율의
  결과임을 보여주는 메커니즘 체크. 전 셀 SupFDR ≪ α+3SE로 **PASS**.
- 발견 0인 셀 3개(α=0.2)는 지침대로 α 추가 상향 없이 그대로 보고.
- 실현 SupFDR이 명목 α보다 훨씬 낮음 (α=0.5에서 ~0.2%). **이 보수성의 귀속은
  R1b로 분해**: e-BH 집계층은 최악 null e-값에서 사실상 타이트하므로, R1의 큰
  마진은 Catoni e-process 층(null crossing이 Ville 예산 1/b_solo를 크게 하회)에서
  발생. `sim_results/R1.parquet`.

### R1b — e-BH 집계층 tautness (합성 최악 null e-값, `sim/run_R1b.py`)

e-process를 우회하고 집계층만 검사 (사용자 작성 스크립트): e_j iid,
P(e_j = B) = 1/B, 그 외 0 (평균 1인 two-point e-값 — Markov/Ville 부등식이
등호가 되는 극단 케이스), B = J/α, γ_j = 1/J, J=212, N_RUN=200,000,
SEED=20260706. two-point 구조에서는 e_j = B가 곧 e-BH 기각 조건(e_j ≥ B/k)
충족이므로 SupFDR = P(어느 하나 e_j = B) = 1−(1−1/B)^J ≈ 1−e^(−α).

| α | empirical SupFDR (SE) | 해석해 1−(1−1/B)^J | 1−e^(−α) |
|---:|---:|---:|---:|
| 0.05 | 0.0486 (0.0005) | 0.0488 | 0.0488 |
| 0.20 | 0.1818 (0.0009) | 0.1813 | 0.1813 |

- 경험치가 해석해와 MC 오차 내 일치 — **e-BH 집계층은 α 예산을 거의 소진하는
  타이트한 층** (α=0.05에서 상한 0.05 대비 실현 0.0488).
- 따라서 R1·E1에서 관측된 SupFDR ≪ α는 집계층 느슨함이 아니라 e-process 층의
  null crossing 보수성에서 오는 것 — 심사 M2의 "α 대비 여유가 왜 큰가"에 대한
  층별 분해 답변으로 사용.

### R2 — deflated Sharpe ratio 비교군 (M3 대응)

E1 16셀 세계·등록 집합 재사용. 런 종료 시점에 등록 전체를 batch family로,
full-history Sharpe([0, min(D_j, T-1)], naive와 동일 창)에 BLdP(2014, JPM 40(5))
DSR 적용. 구현 공식 (논문 그대로, Euler–Mascheroni γ_EM=0.5772…):
  V = Var({SR_i}, ddof=1), N = n_reg,
  SR0 = √V·[(1−γ_EM)·Φ⁻¹(1−1/N) + γ_EM·Φ⁻¹(1−1/(N·e))],
  DSR_i = Φ((SR_i−SR0)·√(n_i−1)/√(1−γ3_i·SR_i+((γ4_i−1)/4)·SR_i²)),
  발견 ⟺ 1−DSR < 0.05. (γ3/γ4는 population 모멘트, SR은 월간·sd ddof=1,
  n_i는 전략별 관측 길이 — BLdP의 공통 n을 전략별로 일반화.)

| 절차 (16셀 범위) | FDP = P(발견≥1) |
|---|---|
| full protocol (baseline reveal) | 전 셀 0.0000 |
| **DSR batch family (BLdP 2014)** | **0.0020 – 0.0185** |
| naive full-history t-test | 0.858 – 1.000 |
| fresh-data 반복 검정 | 전 셀 1.000 |

- DSR은 이 설계에서 FDP를 잘 통제 (전 셀 ≤ 1.9%, α=0.05 이내) — 단 (i) 정확한
  trials N=n_reg를 외생적으로 알려준 이상적 조건이고 (ii) 런 종료 시점의 one-shot
  batch 검정이라 anytime validity·online 발견·등록순 reveal이 없음 — 프로토콜과의
  차별점은 오류 통제율이 아니라 **운용 형태(순차/anytime vs 사후 일괄)**임을 명기.
- FDP가 상대적으로 높은 셀은 강한 selection(s_reg=0.30)·공통요인(ρ=0.3) 조합 —
  SR 횡단면 분산 V가 줄어 SR0가 낮아지는 방향.
- `sim_results/R2.parquet`, `sim_E1_supfdr.png`를 4절차 비교(DSR 4번째 막대)로 교체.

## 심사 대응 2차 — 8건 (2026-07-06, `review/` + `sim/run_R3.py`)

frontier.py 통계 로직 무수정 (M7도 파일 수정 없이 모듈 rng 재설정으로 계산).
STOP 조건 해당 없음 — 기존 수치 전부 불변 확인.

### 1. M4 — fixed-H=120 pseudo-live 인증 (`review/m4_pseudolive.py`)

n_j≥120인 199개 팩터를 post-pub 첫 120개월에서 freeze, raw SR_ann vs
frontier(120)=1.8392: **통과 3/199 = 1.5%** — AnalystRevision 2.70,
EarningsSurprise 2.38, AnnouncementReturn 1.98. 분위수 Q1 0.076 / med 0.283 /
Q3 0.613. → "고정 지평 pseudo-live 인증으로도 사실상 아무도 frontier를 넘지
못함". `data/m4_pseudolive_sr120.csv`.

### 2. M7 — 장기 frontier 직접 MC (`review/m7_longhorizon.py`)

N_GRID 확장 시나리오의 추첨 순서 재현 (rng를 SEED=0으로 재설정 후 60→600 순
호출). **기존 4점 bit-exact 불변 확인** (assert). Table 2 확장값:

| n (mo) | oracle δ*_ann | Catoni δ*_ann |
|---:|---:|---:|
| 480 | 0.83 | **0.90** (δ*=0.2584) |
| 540 | 0.78 | **0.84** (δ*=0.2416) |
| 600 | 0.74 | **0.79** (δ*=0.2286) |

→ 50년(600개월) 관측으로도 요구 Sharpe 연 0.79 — post-pub median 0.26의 3배.

### 3. M6 — survivor envelope-consistent 재계산 (`review/m6_survivor_envelope.py`)

frontier 커널(log_e_path)로 m_env 오버라이드 직접 MC (N_SIM=10,000, SEED=0):

| 팩터 | n | m_env | frontier_ann | 실현 SR_ann | 판정 |
|---|---:|---:|---:|---:|---|
| AnnouncementReturn | 336 | 1.46 | 1.2135 | 1.3520 | **PASS** (1.30 참조: 1.0776) |
| AnalystRevision | 480 | 1.39 | 0.9588 | 1.0549 | **PASS** (1.30 참조: 0.8951) |

→ 두 생존 팩터 모두 자기 envelope-consistent frontier 기준으로도 통과 확정.

### 4. M9 — betting grid 공개 표

`review/appendix_M9_betting_grid.tex` 생성 (frontier.py에서 상수 직접 읽음):
c_k = {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4} (K=7), w_k = 1/7 균등, λ_k = c_k/m_env,
m_env=1.3 + coverage 규칙, scale rule(36개월 trailing sd, ddof=1, 최소 24),
v_min(pre-registration 5% 분위수), freeze rule, 보간(60–600 선형)/외삽(√n),
seed 정책(frontier SEED=0 / sim [20260706, exp, cell, run]).

### 5. M8 — audit artifact 표

`review/appendix_M8_audit.tex` 생성: 전 실험(frontier·Stage4–6·E1–E5 1/2차·
R1·R1b·R2·R3·본 8건)의 실행 기준 commit, seed, N + 데이터 sha256
(osap_LS_v200.csv.gz / SignalDoc.csv / osap_postpub_sharpe.csv, 앞 16 hex).

### 6. M10 — standardized-score Sharpe sensitivity (`review/m10_std_sharpe.py`)

SR_std = √12·mean(Y_t), Y_t = ret/max(v_{t-1}, v_min) (stage4 규칙 동일):
**Pearson corr(raw, std) = 0.962** (Spearman 0.941), median raw 0.258 vs
std 0.287, diff(std−raw) median **+0.022** (Q1 −0.038, Q3 +0.105).
→ 표준화 SR로 바꿔도 그림 불변 (frontier 대비 격차 규모에 비해 미미).
`data/m10_std_sharpe.csv`.

### 7. M11 — A_j 관례 sensitivity (`review/m11_aj_sensitivity.py`)

| A_j 관례 | below % (212) | matured 수 | median margin |
|---|---:|---:|---:|
| Year 1월초 | 98.6% | 201 | −1.041 |
| Year 7월초 | 98.6% | 199 | −1.055 |
| Year 12월말 (primary) | 98.6% | 199 | −1.082 |

→ below-frontier %는 관례와 완전 무관, margin 중앙값 차이 ≤ 0.04.

### 8. R3 — boundary-near null stress (`sim/run_R3.py`)

null 수익률에 GARCH(1,1) (a=0.10, b=0.85, 무조건부 분산 1) + AR(1) φ=0.2
(조건부 평균 0 가정의 명시적 위반 — 무조건부 평균은 0) + ρ=0.3 공통요인 결합.
envelope는 등록 규칙대로 추정 (trailing 36 sd + 등록 전 v_min 5% 분위수 표준화,
m_env=1.3). 등록 = in-sample SR>0.15 첫 돌파 (뮤테이션·포기 없음 — 분포
스트레스에 집중). J_budget=212, D=A+120, N_RUN=2,000, seed [20260706, 8, run].

- n_reg 평균 100.0 (자기상관이 in-sample SR을 부풀려 전 후보 등록),
  max log e = 8.74 (경계 8.35를 실제로 넘는 런 존재 — 스트레스 유효).
- **발견 9건/2,000런 → empirical SupFDR = 0.0045, Wilson 95% CI
  [0.0024, 0.0085]** — α=0.05, 합격선 0.0646 대비 큰 여유로 유지.
- E1(iid null, 발견 0)과 달리 경계 근처에서 실제 crossing이 발생하지만
  SupFDR 통제는 유지 — 분포 가정(iid·gaussian·등분산) 위반 결합에 대한
  프로토콜의 강건성 근거. `sim_results/R3.parquet`.

## 심사 대응 3차 — 6건 (2026-07-07, `review/rev3_*` + `sim/run_R4·R5`, 실행 기준 commit d5d468f)

frontier.py 무수정 (커널은 `sim.eprocess` import 단일 출처). 기존 수치 전부 불변 —
STOP 조건 해당 없음. 실행 환경: 공용 venv `Python/.venv` (pandas 3.0.3).

### 1. 3차 M1 — fixed-H=120 진짜 e-BH 인증 (`review/rev3_m1_ebh_cert.py`)

matured 199개 팩터의 post-pub 첫 120개월 실수익률에 실제 프로토콜 실행. 표준화는
등록 규칙 그대로(36개월 trailing sd ddof=1·최소 24, 팩터별 pre-pub v_min 5% 분위수
floor — stage4 CSV 단일 출처, m_env=1.3), e-process는 frontier 커널 import,
freeze는 solo(b=4240) 도달 또는 t=120, reveal은 등록순(출판연도, 동률 알파벳)
e-BH(γ_j=1/212, α=0.05; immature 13개는 unrevealed지만 예산 포함 — Σγ≤1 유지).
raw SR은 `m4_pseudolive_sr120.csv`와 팩터별 1e-9 이내 일치 assert, reveal 직접
구현은 `sim/ebh.baseline_reveal`과 기각 집합 완전 일치 assert.

- **인증 6/199 (k_199=6, 문턱 γe ≥ 1/(αk)=3.33)**: solo 4건 — AnalystRevision
  (τ=62mo), AnnouncementReturn(46mo), STreversal(73mo), EarningsSurprise(85mo);
  데드라인 동결 후 k≥2 완화로 2건 추가 — DivYieldST(γe 3.70), SmileSlope(γe 9.04).
  7위 γe = 0.35로 경계와 큰 격차 (한계 케이스 아님).
- 주목: **STreversal은 raw SR₁₂₀ = 1.18로 frontier(120)=1.84 미달인데 표준화
  e-process는 73개월에 solo 인증** (√12·mean(Y) = 2.09) — 예측가능 변동성
  표준화가 raw SR 비교보다 검출력이 높은 실증 사례. 반대로 raw 통과 3개
  (M4 2차의 1.5%)는 전부 인증에 포함됨.
- `data/rev3_m1_ebh_cert.csv`(199행 전체), `review/appendix_rev3_m1.tex`
  (기각 6 + γe 상위 15), `run_rev3_m1.log`.

### 2. 3차 M4 — raw vs standardized 분류 안정성 (`review/rev3_m4_raw_std.py`)

| 비교 항목 | raw | standardized (√12·mean Y) |
|---|---|---|
| below-frontier 수 (212) | 209/212 (98.6%) | 208/212 (98.1%) |
| fixed-H=120 상위 3 | AnalystRevision 2.696; EarningsSurprise 2.380; AnnouncementReturn 1.977 | AnalystRevision 2.691; AnnouncementReturn 2.424; STreversal 2.089 |
| 생존 3팩터 status | 전부 above (+0.152/+0.262/+0.582) | 전부 above (+0.420/+0.514/+0.769) |
| matured 199 median margin | −1.082 | −1.034 |

- 분류가 갈리는 팩터는 SmileSlope 1개뿐 (raw −0.229 → std +0.103; 3차 M1의
  e-BH 인증과 정합). 그림 결론 불변. `data/rev3_m4_raw_std.csv`,
  `review/appendix_rev3_m4.tex`.

### 3. 3차 M6 — boundary-near valid-null 스트레스 (`sim/run_R4.py`, EXP_ID R4=9)

유효 null 유지(조건부 평균 0, Y=√0.95·ε, mean(Y²)=0.9502 ≤ m_env²=1.1025)하면서
경계를 낮춰 null e-process가 경계 근처를 자주 방문: J=5, α=0.5, D=600,
m_env=1.05, b_solo=10, N_RUN=20,000, seed [20260706, 9, J, batch].

- **empirical SupFDR = 0.1588, Wilson 95% CI [0.1538, 0.1639]** ≪ α=0.5 —
  발견 3,175런/20,000 전부 solo 경로(발견률 = solo crossing률 정확 일치).
- 경계 근접 P(max logE ≥ log b − 1) = 50.2%, max logE 분포 q50 1.31 / q90 2.67 /
  q99 4.49 / max 9.27 (log b = 2.30) — "경계 근처 빈번 방문" 조건 충족 확인.
  발견이 실제로 발생하므로 J=2 fallback은 미실행 (사전 지정 조건).
- 해석: R3(무효 null 스트레스)와 짝을 이뤄, 유효 envelope 하에서는 경계를
  공격적으로 낮춰도(α=0.5) Ville/e-BH 통제가 큰 마진으로 유지됨을 실측.
  `sim_results/R4.parquet`, `run_simR4.log`.

### 4. 3차 M9 — E5 comparator: DSR batch (`sim/run_R5.py`, E5 seed 재사용)

E5와 동일 mixed world(50% alt, δ=δ*(120))를 CRN으로 재생성 — 저장된
`E5_strategies.parquet`의 A·is_alt와 전 1,500런 완전 일치 assert 통과. 종료 시점
T=360에 BLdP(2014) DSR batch(R2와 동일 공식, trials N=n_reg 외생 제공) 적용:

| λ_arr | protocol power | protocol 평균 지연 | DSR power | DSR 평균 지연 (=T−A_j) |
|---:|---:|---:|---:|---:|
| 0.5 | 76.6% | 118.9mo | 0.20% | 9.3mo |
| 1.0 | 96.0% | 122.8mo | 0.006% | 283.3mo |
| 2.0 | 95.9% | 121.3mo | 0.004% | 272.5mo |

- **DSR power 붕괴는 버그가 아니라 구조**: mixed family에서 SR 횡단면 분산 V가
  참 alt의 스킬 분산(bimodal)까지 흡수 → SR0 ≈ 0.73–0.77 (월간) > 참 alt SR
  0.531 → 문턱이 신호 위. 손계산 기대 power(λ=1, ~1e-5)와 실측 일치 검증.
  λ=0.5의 "발견"은 대부분 표본 짧은 늦은 등록의 행운 SR (null share 19.5%,
  FDP 평균 0.043 — α=0.05 이내로 validity 자체는 유지).
- 발견 가능 시점 T=360 고정(batch)임을 delay 정의(T−A_j)로 기록. SPA·White RC는
  벤치마크 대비 최댓값 검정(per-strategy 인증·anytime 없음)으로 범위 밖 —
  미실행, 표 각주 명시. `sim_results/R5.parquet`, `review/appendix_rev3_m9.tex`.

### 5. 3차 M10 — OSAP idea-level grouping (`review/rev3_m10_clusters.py`)

SignalDoc `Cat.Economic`(Predictor 212개, 결측 0)으로 클러스터 정의(35개),
클러스터 판정 = 소속 팩터 margin 최댓값 (censored 포함 212 전체):

- **primary: below 32/35 = 91.4%** (above 3: valuation→DivYieldST +0.582,
  earnings event→AnnouncementReturn +0.262, earnings forecast→AnalystRevision
  +0.152). above 클러스터의 best factor가 정확히 개별 팩터판 생존 3개와 일치 —
  클러스터 집계는 분모만 바꿀 뿐 결론 불변.
- 민감도: "other"(잡동사니 27개) 싱글턴 분해 시 below 58/61 = 95.1%.
  `data/rev3_m10_clusters.csv`.

### 6. 산출물 위생

- **E2 exact rates** (`review/rev3_e2_rates.py`): `E2.parquet`에서 12셀 정확값
  추출 — RUN_LOG 반올림 표와 전 셀 일치 assert, 1.0× 전 셀 50%±10%p 재확인.
  `data/rev3_e2_exact_rates.csv`(풀 정밀도), `review/appendix_rev3_e2.tex`.
- **Fig1 600 grid 갱신** (`review/rev3_fig1_600.py`): 기존 frontier.png는 4점
  (60–360)이었음 → M7 방식 rng 재현으로 7점(60–600) 재계산, 기존 4점
  bit-exact 불변 assert 후 frontier.py와 동일 스타일로 재렌더.
  연율 δ*: 2.862/1.839/1.282/1.041/0.895/0.837/0.792.
- **manifest.json** (`review/make_manifest.py`, repo 루트): `data/` 전 12개 파일의
  full SHA-256 + bytes (비추적 원본 osap_LS_v200.csv.gz·SignalDoc.csv 포함 —
  다운로드 무결성 확인용). M8 appendix의 16-hex 3건과 대조 assert 통과.

## 심사 대응 4차 — 3건 (2026-07-07, `review/rev4_*`, 실행 기준 commit 4be0a19)

frontier.py 무수정, 기존 수치 전부 불변 (registered 재현 assert 통과). 실행 환경 동일.

### 1. 4차 M1 — certified 6개 envelope-consistent 재검증 (`review/rev4_m1_envelope_adj.py`)

3차 인증 6개 중 4개가 pre-pub mean(Y²) > 1.69 = 1.3² (stage4 확인:
AnnouncementReturn 2.134, SmileSlope 1.959, AnalystRevision 1.939, STreversal
1.788). 이 4개의 e-process를 envelope-consistent m_env_j = max(1.3, √meanY2_j)
= 1.4607/1.3995/1.3925/1.3371로 재실행 (동일 Y·freeze·b_solo=4240), 적합 2개
(EarningsSurprise 1.511, DivYieldST 1.503)와 193개는 registered 유지, adjusted
값으로 e-BH 전면 재판정. 게이트: registered 199개 전량 1e-9 재현, 페널티 항등,
baseline_reveal 대조 — 전부 통과.

- **adjusted 인증 4/199 (k=4)**: AnalystRevision(logE_τ 8.51→8.69, solo 유지),
  AnnouncementReturn(8.53→8.43, solo 유지), STreversal(8.59→8.40, solo 유지),
  EarningsSurprise(불변). **solo 4개는 envelope 조정에 전부 강건.**
- 탈락 2개:
  - **SmileSlope**: ΔlogE_τ = −1.020 (7.558→6.538, γe 9.04→3.26) — 문턱 밖.
    귀속: 커널 파라미터화상 λ_k = c_k/m_env라 페널티 항 0.5λ²m² = 0.5c²는
    m_env 항등(수치 assert) → 감소분 전액이 φ(Y·λ)의 λ 축소분.
  - **DivYieldST**: ΔlogE_τ = 0 (자신은 무조정, γe 3.699 그대로) — SmileSlope
    탈락으로 k가 6→4로 내려가 문턱이 3.33→5.0으로 상승하는 **e-BH 연쇄 효과**로
    탈락 (k=5도 실패: γe≥4.0인 팩터 4개뿐).
- `appendix_rev3_m1.tex`에 열 추가 재생성 (meanY2_pre, m_env_consistent,
  logE_tau_adj, gamma_e_adj, rejected_adj). `data/rev4_m1_envelope_adj.csv`(199행),
  `run_rev4_m1.log`.

### 2. 4차 M4 — t(5) 장기 frontier (`review/rev4_m4_t5_long.py`)

NOISE=t5, N_SIM=10,000, SEED=0, M7 방식 rng 재현 (import 4점 RUN_LOG 게이트 +
기존 4점 bit-exact assert 통과). Table 2 t5 열 3칸:

| n (mo) | t5 δ* (월간) | t5 연율 | gaussian 연율 | 차이 |
|---:|---:|---:|---:|---:|
| 480 | 0.258525 | 0.8956 | 0.8952 | +0.0003 |
| 540 | 0.242966 | 0.8417 | 0.8370 | +0.0047 |
| 600 | 0.229113 | 0.7937 | 0.7919 | +0.0018 |

→ 장기 3점에서도 t5-gaussian 차이 ≤ 0.005 (연율) — Stage 3의 "Catoni 검정력은
사실상 처음 두 모멘트로 결정" robustness가 600개월까지 유지.
`data/rev4_m4_t5_long.csv`, `run_rev4_m4.log`.

### 3. 4차 m5 — A_j 관례 set-invariance (`review/rev4_m5_aj_sets.py`)

세 관례(Year 1월초/7월초/연말 primary)에서 full-horizon survivor set과 fixed-H
e-BH certified set을 집합 비교. frontier는 catoni 정확값 사용 (stage6 CSV는
소수 3자리 반올림 상수 이력 — margin 최대차 0.0019, 비생존 최고 margin −0.033
대비 한 자릿수 밖이라 판정 무영향, self-consistency 게이트는 PASS). v_min은
관례별 pre-window로 재계산 (primary가 stage4와 1e-9 일치 assert). primary
certified가 3차 6개와 일치 assert.

- **survivor set: 세 관례 완전 동일** {AnalystRevision, AnnouncementReturn,
  DivYieldST} → PASS.
- **certified set: primary에서만 DivYieldST 추가** (1월초/7월초 = 5개
  {AnalystRevision, AnnouncementReturn, EarningsSurprise, STreversal, SmileSlope},
  primary = 6개). 원인: DivYieldST 자신의 동결 γe가 창 이동에 민감 —
  1.576(1월초)/2.442(7월초)/3.699(연말)로 k=6 문턱 3.33을 연말에서만 통과.
  나머지 5개는 전 관례에서 γe ≥ 4.0으로 인증 → **비불변은 DivYieldST 1개의
  한계 인증(marginal certification) 문제이고 핵심 집합 5개는 불변**.
- 4차 M1과 종합하면: **solo 4개(AnalystRevision, AnnouncementReturn, STreversal,
  EarningsSurprise)는 envelope 조정·A_j 관례 양쪽에 강건**, SmileSlope는 관례
  불변이나 envelope 조정에 탈락, DivYieldST는 양쪽 모두에서 한계적.
  `data/rev4_m5_aj_sets.csv`, `run_rev4_m5.log`.

### manifest 갱신

4차 신규 CSV 3개 포함 `data/` 전 15개 파일로 manifest.json 재생성
(M8 16-hex 대조 assert 재통과).
