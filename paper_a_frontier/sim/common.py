"""sim 공통 유틸 — SEED 체계, parquet 기록, 커밋 해시.

SEED 체계 (§13): 실험셀별 고정, CRN.
  rng = np.random.default_rng([SEED_BASE, exp_id, *cell_keys, run_idx])
문자 파라미터는 아래 정수 코드로 인코딩 (RUN_LOG에 동일 표 기재):
  noise:    gaussian=0, t5=1
  behavior: honest=0, adversarial=1
CRN이 필요한 축(예: E2의 배수, E3의 cumedge)은 noise seed 키에서 제외해
동일 잡음 위에 delta만 바꾼다.
"""
import subprocess
from pathlib import Path

import numpy as np

SEED_BASE = 20260706
EXP_ID = {"E1": 1, "E2": 2, "E3": 3, "E4": 4, "E5": 5, "R1": 6, "R2": 7}
NOISE_CODE = {"gaussian": 0, "t5": 1}
BEHAVIOR_CODE = {"honest": 0, "adversarial": 1}

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "sim_results"


def rng_for(*keys):
    """정수 키 시퀀스로부터 결정론적 rng 생성."""
    return np.random.default_rng([SEED_BASE, *[int(k) for k in keys]])


def commit_hash():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()


def write_parquet(df, name):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    df.to_parquet(path, index=False)
    return path
