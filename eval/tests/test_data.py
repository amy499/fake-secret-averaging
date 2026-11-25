"""core/data.py: manifest sanity, hashing, the canonical CRLF-jsonl rule and MATCH / MISMATCH /
MISSING checks, on generated files (no network)."""

from __future__ import annotations

import hashlib
import json

import pytest

from core import data as D


def test_manifest_pins_revisions_and_protocol_hashes():
    for name, e in D.MANIFEST.items():
        assert e.revision in e.url, name  # the URL resolves the pinned revision, never a branch
        assert len(e.revision) == 40, name
        assert e.checks, name
    for name, prefix in D.PROTOCOL_PREFIXES.items():
        kinds = {c.prefix: c for c in D.MANIFEST[name].checks}
        assert prefix in kinds and kinds[prefix].source == "protocol"
    # the GSM8K / ARC protocol hashes are of the converted (canonical) bytes
    assert {c.kind for c in D.MANIFEST["gsm8k_test"].checks if c.source == "protocol"} == {"canonical_sha256"}
    assert {c.kind for c in D.MANIFEST["confaide_tier4"].checks} == {"sha256"}


def test_canonical_jsonl_bytes_rule():
    rows = [{"question": "Ünïcode?", "answer": "#### 3"}, {"question": "b", "answer": "#### 4"}]
    got = D.canonical_jsonl_bytes(rows)
    want = ('{"question": "Ünïcode?", "answer": "#### 3"}\r\n'
            '{"question": "b", "answer": "#### 4"}\r\n').encode("utf-8")  # fmt: skip
    assert got == want


def _entry(path, **checks):
    return D.DataFile("tmp", f"https://example.invalid/{'0' * 40}/{path.name}", "0" * 40, "test",
                      tuple(D.Check(k, v, "protocol", "test") for k, v in checks.items()))  # fmt: skip


def test_verify_raw_file(tmp_path):
    p = tmp_path / "tier_4.txt"
    p.write_bytes(b"<BEGIN><a, b>\nAlice: hi\n<END>\n")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    rec = D.verify_file(_entry(p, sha256=sha[:16], md5=md5[:8]), p)
    assert rec.match and rec.sha256 == sha and [c["status"] for c in rec.checks] == ["MATCH", "MATCH"]
    bad = D.verify_file(_entry(p, sha256="0" * 16), p)
    assert not bad.match and bad.checks[0]["status"] == "MISMATCH" and bad.checks[0]["got"] == sha[:16]
    missing = D.verify_file(_entry(tmp_path / "nope.txt", sha256="0" * 16), tmp_path / "nope.txt")
    assert not missing.match and missing.checks[0]["status"] == "MISSING"


def test_verify_parquet_canonical_and_rows(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(5)]
    p = tmp_path / "test-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(rows), p)
    canon = hashlib.sha256(D.canonical_jsonl_bytes(rows)).hexdigest()
    e = D.DataFile("tmp", "https://example.invalid/x/test-00000-of-00001.parquet", "x" * 40, "test",
                   (D.Check("canonical_sha256", canon[:16], "protocol", "test"),), rows=5)  # fmt: skip
    rec = D.verify_file(e, p)
    assert rec.match and rec.rows == 5 and rec.canonical_sha256 == canon
    e_rows = D.DataFile("tmp", e.url, e.revision, "test", e.checks, rows=6)
    assert not D.verify_file(e_rows, p).match


def test_jsonl_row_count(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps({"i": i}) for i in range(3)) + "\n\n", encoding="utf-8")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    e = D.DataFile("tmp", "https://example.invalid/x/x.jsonl", "x" * 40, "test",
                   (D.Check("sha256", sha[:16], "team", "test"),), rows=3)  # fmt: skip
    assert D.verify_file(e, p).rows == 3 and D.verify_file(e, p).match


def test_get_is_strict_and_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_DATA", str(tmp_path))
    with pytest.raises(D.DataHashError, match="MISSING"):
        D.get("confaide_tier4", download=False)
    path = D.local_path("confaide_tier4")
    assert path == tmp_path / "url" / "protocol" / "confaide_tier4" / "tier_4.txt"
    path.parent.mkdir(parents=True)
    path.write_text("not the real file\n", encoding="utf-8")
    with pytest.raises(D.DataHashError, match="MISMATCH"):
        D.get("confaide_tier4", download=False)
    p, rec = D.get("confaide_tier4", download=False, strict=False)
    assert p == path and not rec.match and rec.protocol_prefix == "cf0ba2d5f4d6c136"


def test_cli_verify_reports_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HARNESS_DATA", str(tmp_path))
    assert D.main(["verify", "confaide_tier4"]) == 1
    out = capsys.readouterr().out
    assert "MISSING" in out and "cf0ba2d5f4d6c136" in out
