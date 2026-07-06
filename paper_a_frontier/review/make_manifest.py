"""Rev3 — 심사용 데이터 무결성 manifest를 생성한다.

paper_a_frontier/data 아래 모든 파일의 SHA-256과 바이트 크기를 repo 루트 기준
상대경로로 기록하고, M8 appendix에 박제된 원본 3개 파일의 16-hex를 검산한다.
실행: ../../.venv/bin/python3 review/make_manifest.py
산출: ../manifest.json
"""
import hashlib
import json
import re
from pathlib import Path

PAPER_REPO = Path(__file__).resolve().parent.parent
ROOT = PAPER_REPO.parent
DATA = PAPER_REPO / "data"
REVIEW = PAPER_REPO / "review"
OUT = ROOT / "manifest.json"

CHECK_FILES = ["osap_LS_v200.csv.gz", "SignalDoc.csv", "osap_postpub_sharpe.csv"]


def sha256_file(path: Path) -> str:
    """파일을 스트리밍으로 읽어 SHA-256 full hex를 계산한다."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_m8_hashes() -> dict[str, str]:
    """appendix_M8_audit.tex에서 데이터 파일별 16-hex 해시를 읽는다."""
    tex = (REVIEW / "appendix_M8_audit.tex").read_text()
    found: dict[str, str] = {}
    pat = re.compile(r"\\texttt\{(?P<name>[^{}]+)\}.*?\\texttt\{(?P<hex>[0-9a-f]{16})\}")
    for line in tex.splitlines():
        match = pat.search(line)
        if not match:
            continue
        name = match.group("name").replace(r"\_", "_")
        found[name] = match.group("hex")
    missing = [name for name in CHECK_FILES if name not in found]
    assert not missing, f"M8 appendix 해시 누락: {missing}"
    return found


def build_entries() -> list[dict[str, object]]:
    """data 디렉터리의 모든 파일 manifest 항목을 경로 오름차순으로 만든다."""
    files = sorted(
        (path for path in DATA.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    entries: list[dict[str, object]] = []
    for path in files:
        entries.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return entries


def verify_m8(entries: list[dict[str, object]]) -> dict[str, str]:
    """M8 appendix의 16-hex와 현재 파일 해시 prefix를 대조한다."""
    expected = parse_m8_hashes()
    by_path = {entry["path"]: entry for entry in entries}
    prefixes: dict[str, str] = {}
    for name in CHECK_FILES:
        rel = f"paper_a_frontier/data/{name}"
        assert rel in by_path, f"manifest 대상 파일 누락: {rel}"
        prefix = str(by_path[rel]["sha256"])[:16]
        assert prefix == expected[name], f"{name} 16-hex 불일치: {prefix} vs {expected[name]}"
        prefixes[name] = prefix
    return prefixes


def main() -> None:
    """manifest.json을 생성하고 검산 결과를 출력한다."""
    entries = build_entries()
    prefixes = verify_m8(entries)
    data_version = (DATA / "DATA_VERSION.txt").read_text().strip()
    manifest = {
        "purpose": "SHA-256 manifest for Paper A review data files.",
        "data_version": data_version,
        "files": entries,
    }
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"manifest files: {len(entries)}")
    for name in CHECK_FILES:
        print(f"{name}: {prefixes[name]}")
    print("M8 16-hex 대조: PASS")
    print(f"saved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
