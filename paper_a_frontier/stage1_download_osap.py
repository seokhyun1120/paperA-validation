"""Stage 1 — OSAP v2.00 (2025.10) 데이터 다운로드.

원논문 구현(op) 포트폴리오 long format에서 port == 'LS'만 추려 저장하고,
SignalDoc을 함께 저장한다. SSL_CERT_FILE=certifi 필요 (macOS python.org 빌드).

주의: dl_port(..., 'pandas')는 py3.14에서 polars→pandas 변환 중 segfault가 나서
polars로 받아 polars로 바로 CSV를 쓴다 (내용은 동일한 long format).
"""
import datetime
import gzip
from pathlib import Path

import openassetpricing as oap

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

openap = oap.OpenAP()   # 최신 릴리스 v2.00 (2025.10)

df = openap.dl_port('op', 'polars')     # 원논문 구현 포트폴리오, long format
print("port df:", df.shape, df.columns)
print(df['port'].value_counts().sort('port').head(20))

ls = df.filter(df['port'] == 'LS')
print("LS rows:", ls.shape, " signals:", ls['signalname'].n_unique())
with gzip.open(DATA / "osap_LS_v200.csv.gz", "wb") as f:
    ls.write_csv(f)

doc = openap.dl_signal_doc('polars')
print("signal doc:", doc.shape)
doc.write_csv(DATA / "SignalDoc.csv")

today = datetime.date.today().isoformat()
(DATA / "DATA_VERSION.txt").write_text(
    f"OSAP data release 2025.10 (v2.00), downloaded {today}\n"
)
print("saved -> osap_LS_v200.csv.gz, SignalDoc.csv, DATA_VERSION.txt")
