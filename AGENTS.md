# AGENTS.md — quant_research

이 repo에서 코드를 구현하는 에이전트(Codex 포함)를 위한 지시문. 워크스페이스 루트 `AGENTS.md`의 컨벤션에 **추가로** 적용되며, 충돌 시 이 문서가 우선한다.

## 방법론 규율 — 구현 시 절대 준수

이 리서치는 다중검정 오염 방지가 핵심 정체성이다. **규율 위반은 코드 버그보다 더 큰 사고다.**

1. **사전등록 준수**: 지시 없이 lookback/필터/비용/지표/판정 기준을 변경 금지. 결과를 본 뒤 파라미터를 조정하는 것(골대 옮기기)은 어떤 이유로도 금지.
2. **시도 횟수 N 보존**: 지시 없이 그리드 칸을 추가·삭제·변경 금지. 그리드 한 칸도 다중검정 보정의 N에 들어간다.
3. **VAULT(holdout) 봉인**: holdout 데이터를 읽거나 평가하는 코드를 지시 없이 작성·실행 금지. 봉인된 데이터 경로에 접근하지 않는다.
4. **look-ahead 차단**: 신호 정렬은 shift 기반(`x_{i,t} → r_{i,t+1}`)을 유지한다. 함수 내부에서 전역(전체 기간) 통계 사용 금지 — 시점 t의 계산에는 t 이전 데이터만 쓴다.
5. **동결 선언 존중**: `research_log.md`에 🚫로 표시된 종결 프로젝트(8 ETF universe, L1 momentum)에 전략·파라미터 추가 금지.
6. **정직한 보고**: 결과가 나쁘거나 유의하지 않아도 코드로 우회하거나 수치를 숨기지 말고 그대로 보고한다. cherry-pick 금지.
7. **무료 데이터(yfinance) 한계 존중**: L4 결론은 방법론 데모 한정 — 생존편향 때문에 실증 결론을 단정하는 코드·주석·로그 문구를 넣지 않는다.

## 환경

- 공용 venv 사용: `source ../.venv/bin/activate` (Python 3.14, pandas 3.0.3, pyarrow 24.0) — repo 전용 `.venv`에는 pandas가 없으므로 사용하지 않는다.
- numpy 2.x — `np.trapz` 대신 `np.trapezoid`.
- 네트워크 다운로드 시 `SSL_CERT_FILE=$(python3 -m certifi)` 필요 (macOS SSL 인증서 문제).
- parquet 사용 가능 (pyarrow) — 단 `.gitignore` 정책상 `*.parquet`는 비추적(재생성 가능 산출물). 커밋 대상 데이터는 CSV.
- 테스트 프레임워크 없음. 각 프로젝트의 `verify*.py`, `validation_gate_*.py`, `tests/test_*.py`를 직접 실행해 검증한다.

## 공통 규칙

- 요청 범위 밖 파일 수정 금지. 무관한 리포맷·리네이밍 금지.
- git commit / push 금지.
- 주석과 docstring은 한국어로 작성한다.
