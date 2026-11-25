"""Mirror mode for lite_setup.pbs (PBS): the uv.lock package set, fetched from faster hosts.

Status: ✅ rewrite / verify logic tested on CPU (selftest_v2.py); 🟡 not yet run on the PBS cluster.

Why: The PBS cluster compute nodes fetch files.pythonhosted.org (PyPI's CDN) at ~35 KB/s, so the CUDA wheels
of uv.lock never finish. Nothing about WHAT is installed changes here, only WHERE each file comes
from:

  rewrite  (stdlib only; run with /bin/python3.11 -I inside the job)
           Reads uv.lock and writes a copy in which a wheel's URL is replaced when another source
           has the SAME FILENAME with the SAME SHA256 as uv.lock: a local wheelhouse (--wheelhouse,
           file:// URL) first, else the URL the PyTorch package index links for that file
           (--pytorch-index, default https://download.pytorch.org/whl; NVIDIA's wheels are linked
           there to pypi.nvidia.com, torch to download-r2.pytorch.org). Hashes are never relaxed:
           a mirror file listed with a DIFFERENT sha256 is never used (reported; that file keeps
           its uv.lock URL; --strict / MIRROR_STRICT=1 stops instead). PyTorch's triton 3.6.0 is
           such a file: a different build from PyPI's under the same name, so triton comes from
           its uv.lock URL or, faster, from a --wheelhouse copy of PyPI's file (sha256-checked).
           A wheelhouse file whose sha256 differs from uv.lock always stops (exit 1). Every other
           file keeps its uv.lock URL. lite_setup.pbs then runs `uv sync --frozen` on
           that copy (uv itself re-checks every downloaded file against the uv.lock sha256), and
           the normal `uv sync --frozen` on the real project afterwards (installs the project; the
           packages are already there, so nothing heavy is downloaded again).
  verify   (run with the new .venv's python)
           The installed distributions must equal `uv export --frozen` (the synced groups, markers
           evaluated for this machine) plus the project, name and version, nothing missing and
           nothing extra; torch must be the expected build. Writes MIRROR.txt: which host each
           installed package's file was mapped to.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any

PYPI = "https://pypi.org/simple"
HEAVY_BYTES = 50 * 1024 * 1024


def norm_name(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n).lower()


def host(url: str) -> str:
    p = urllib.parse.urlparse(url)
    return "local wheelhouse" if p.scheme == "file" else p.netloc


def filename(url: str) -> str:
    return urllib.parse.unquote(url.rsplit("/", 1)[-1].split("#", 1)[0])


def linux_x86_64_cp311(fn: str) -> bool:
    """The wheel this the PBS cluster node (Linux x86_64, CPython 3.11) can install."""
    if not fn.endswith(".whl"):
        return False
    tags = fn[:-4].split("-")[-3:]
    py, abi, plat = tags if len(tags) == 3 else ("", "", "")
    py_ok = any(t in ("cp311", "py3", "py2.py3") or t.startswith("py3") for t in py.split("."))
    plat_ok = plat == "any" or ("x86_64" in plat and ("manylinux" in plat or "linux" in plat))
    return py_ok and plat_ok and abi in ("none", "abi3", "cp311")


def fetch(url: str, timeout: int = 60) -> str:
    """An index page, via curl -4 (IPv6 is unreachable on the PBS cluster compute nodes)."""
    r = subprocess.run(["curl", "-s", "-4", "--fail", "--retry", "3", "--max-time", str(timeout), url],
                       capture_output=True, text=True, check=False)  # fmt: skip
    return r.stdout if r.returncode == 0 else ""


def index_links(page: str, base: str) -> dict[str, tuple[str, str | None]]:
    """{filename: (absolute url, sha256 or None)} from a PEP 503 simple page."""
    out = {}
    for href in re.findall(r'href="([^"]+)"', page):
        url, _, frag = href.partition("#")
        url = urllib.parse.urljoin(base, url)
        sha = frag.split("=", 1)[1] if frag.startswith("sha256=") else None
        out[filename(url)] = (url, sha)
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan(lock: dict, pytorch_index: str | None, wheelhouse: Path | None, pages: dict[str, str] | None = None) -> dict:
    """{"rewrites": {old_url: new_url}, "packages": {name: {...}}, "errors": [...]}. pages: name ->
    index page text (tests); otherwise fetched."""
    rewrites: dict[str, str] = {}
    packages: dict[str, Any] = {}
    errors: list[str] = []
    local = {p.name: p for p in wheelhouse.glob("*.whl")} if wheelhouse and wheelhouse.is_dir() else {}
    for pkg in lock.get("package", []):
        name, version = pkg["name"], pkg.get("version")
        registry = (pkg.get("source") or {}).get("registry")
        wheels = pkg.get("wheels") or []
        if not registry or not wheels:
            continue
        links: dict[str, tuple[str, str | None]] = {}
        if pytorch_index and registry == PYPI:
            base = f"{pytorch_index.rstrip('/')}/{norm_name(name)}/"
            page = (pages or {}).get(norm_name(name)) if pages is not None else fetch(base)
            links = index_links(page or "", base)
        rec = {"version": version, "registry": registry, "files": []}
        for w in wheels:
            url, want = w["url"], w["hash"].split(":", 1)[1]
            fn = filename(url)
            new = None
            if fn in local:
                got = sha256_file(local[fn])
                if got != want:
                    errors.append(f"{name} {fn}: wheelhouse sha256 {got[:16]} != uv.lock {want[:16]}")
                    continue
                new = local[fn].resolve().as_uri()
            differs = None
            if fn in local:
                pass  # handled above
            elif fn in links:
                murl, msha = links[fn]
                if msha is not None and msha != want:
                    # a different build under the same name (PyTorch's triton 3.6.0 is one): never used;
                    # this file keeps its uv.lock URL, and uv checks the uv.lock sha256 as always
                    differs = f"{host(murl)} lists sha256 {msha[:16]} != uv.lock {want[:16]}: not used"
                elif murl != url:
                    new = murl
            if new:
                rewrites[url] = new
            rec["files"].append({"file": fn, "host": host(new or url), "size": w.get("size"),
                                 "this_node": linux_x86_64_cp311(fn), "mirrored": bool(new), "mirror_differs": differs})  # fmt: skip
        packages[norm_name(name) + "==" + str(version)] = rec
    return {"rewrites": rewrites, "packages": packages, "errors": errors}


def rewrite_text(text: str, rewrites: dict[str, str]) -> str:
    for old, new in rewrites.items():
        needle = f'url = "{old}"'
        if text.count(needle) != 1:
            raise SystemExit(f"STOP: {old} occurs {text.count(needle)} times in uv.lock (expected once)")
        text = text.replace(needle, f'url = "{new}"')
    return text


def cmd_rewrite(a: argparse.Namespace) -> int:
    import tomllib

    text = Path(a.lock).read_text(encoding="utf-8")
    p = plan(tomllib.loads(text), None if a.no_pytorch else a.pytorch_index, Path(a.wheelhouse) if a.wheelhouse else None)
    differ = [(k, f) for k, rec in p["packages"].items() for f in rec["files"] if f.get("mirror_differs")]
    for k, f in differ:
        print(f"mirror: {'STOP' if a.strict else 'NOT USED'}: {k} {f['file']}: {f['mirror_differs']}"
              f"{' (this node installs it: it comes from its uv.lock URL instead)' if f['this_node'] else ''}", flush=True)  # fmt: skip
    if p["errors"]:  # a wheelhouse file that does not match uv.lock: always fatal (the user put it there)
        for e in p["errors"]:
            print("STOP: wheelhouse sha256 differs from uv.lock:", e, flush=True)
        return 1
    if a.strict and differ:
        print("STOP: MIRROR_STRICT=1 and a mirror lists another sha256 for a uv.lock file", flush=True)
        return 1
    Path(a.out).write_text(rewrite_text(text, p["rewrites"]), encoding="utf-8", newline="\n")
    Path(a.plan).write_text(json.dumps(p, indent=1) + "\n", encoding="utf-8", newline="\n")
    by_host: dict[str, int] = {}
    for key, rec in sorted(p["packages"].items()):
        for f in rec["files"]:
            if not f["this_node"]:
                continue
            by_host[f["host"]] = by_host.get(f["host"], 0) + int(f["size"] or 0)
            if (f["size"] or 0) >= HEAVY_BYTES or (f["size"] is None and key.startswith(("torch==", "triton=="))):
                size = "   size ?" if f["size"] is None else f"{f['size'] / 2**20:7.0f} MiB"
                print(f"mirror: {key:<38} {size} from {f['host']}"
                      f"{' (sha256 = uv.lock)' if f['mirrored'] else ' (uv.lock URL)'}", flush=True)  # fmt: skip
    for h, b in sorted(by_host.items(), key=lambda x: -x[1]):
        print(f"mirror: {b / 2**20:8.0f} MiB of wheels for this node from {h}", flush=True)
    print(f"mirror: {len(p['rewrites'])} wheel URLs rewritten in {a.out}", flush=True)
    return 0


def cmd_fetch(a: argparse.Namespace) -> int:
    """Laptop side: download this node's wheels (Linux x86_64, CPython 3.11) of uv.lock from their
    uv.lock URLs whose host is in --hosts and size >= --min-mib, into --out, each checked against the
    uv.lock sha256 (a mismatch is deleted and reported, exit 1). Copy the folder to the PBS cluster scratch and
    pass it as WHEELHOUSE to lite_setup.pbs; the job checks every file's sha256 again."""
    import tomllib

    lock = tomllib.loads(Path(a.lock).read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hosts = set(a.hosts.split("+"))
    skip = tuple(p for p in (a.skip_prefix or "").split("+") if p)
    todo = []
    for pkg in lock.get("package", []):
        if skip and norm_name(pkg["name"]).startswith(skip):
            continue
        for w in pkg.get("wheels") or []:
            fn = filename(w["url"])
            if linux_x86_64_cp311(fn) and host(w["url"]) in hosts and (w.get("size") or 0) >= a.min_mib * 2**20:
                todo.append((pkg["name"], fn, w["url"], w["hash"].split(":", 1)[1], w.get("size") or 0))
    bad = 0
    for i, (name, fn, url, want, size) in enumerate(todo, 1):
        dst = out / fn
        if not (dst.exists() and sha256_file(dst) == want):
            print(f"fetch {i}/{len(todo)}: {fn} ({size / 2**20:.0f} MiB) from {host(url)}", flush=True)
            subprocess.run(["curl", "-sS", "-4", "--fail", "--retry", "5", "-C", "-", "-o", str(dst), url], check=False)
        got = sha256_file(dst) if dst.exists() else ""
        if got != want:
            bad += 1
            print(f"MISMATCH: {fn}: sha256 {got[:16]} != uv.lock {want[:16]}; deleted", flush=True)
            dst.unlink(missing_ok=True)
        else:
            print(f"ok {i}/{len(todo)}: {fn} sha256 = uv.lock", flush=True)
    print(f"wheelhouse {out}: {len(todo) - bad} of {len(todo)} files verified", flush=True)
    return 1 if bad else 0


def parse_export(text: str) -> list[tuple[str, str, str]]:
    """[(name, version, marker)] from `uv export --no-hashes --format requirements-txt`."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        req, _, marker = line.partition(";")
        name, _, version = req.strip().partition("==")
        out.append((norm_name(name), version.strip(), marker.strip()))
    return out


def compare(expected: list[tuple[str, str]], installed: dict[str, str], project: str) -> list[str]:
    exp = dict(expected)
    errs = [f"missing {n}=={v}" for n, v in sorted(exp.items()) if n not in installed]
    errs += [f"{n}: installed {installed[n]}, uv.lock {v}" for n, v in sorted(exp.items()) if n in installed and installed[n] != v]
    errs += [f"extra {n}=={v}" for n, v in sorted(installed.items()) if n not in exp and n != norm_name(project)]
    return errs


def cmd_verify(a: argparse.Namespace) -> int:
    from importlib import metadata

    from packaging.markers import Marker

    rows = parse_export(Path(a.expected).read_text(encoding="utf-8"))
    expected = [(n, v) for n, v, m in rows if not m or Marker(m).evaluate()]
    installed = {norm_name(d.metadata["Name"]): d.version for d in metadata.distributions()}
    errs = compare(expected, installed, a.project)
    if norm_name(a.project) not in installed:
        errs.append(f"the project {a.project} is not installed")
    if installed.get("torch") != a.torch:
        errs.append(f"torch {installed.get('torch')} != {a.torch}")
    p = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    lines = [f"# mirror mode (lite_setup.pbs HEAVY_FROM_PYTORCH / WHEELHOUSE): where each installed package's wheel was fetched",
             f"# {len(expected)} packages from uv.lock (+ the project); sha256 of every file checked by uv against uv.lock"]  # fmt: skip
    for n, v in sorted(expected):
        rec = p["packages"].get(f"{n}=={v}") or {}
        hosts = sorted({f["host"] for f in rec.get("files", []) if f["this_node"]}) or ["?"]
        lines.append(f"{n}=={v}\t{', '.join(hosts)}")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    if errs:
        for e in errs:
            print("STOP: installed set differs from uv.lock:", e, flush=True)
        return 1
    print(f"verify: {len(expected)} packages = uv.lock (names and versions), project installed, torch {a.torch}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rewrite")
    r.add_argument("--lock", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--plan", required=True)
    r.add_argument("--pytorch-index", default="https://download.pytorch.org/whl")
    r.add_argument("--no-pytorch", action="store_true")
    r.add_argument("--wheelhouse", default=None)
    r.add_argument("--strict", action="store_true", help="stop if a mirror lists another sha256 for any uv.lock file")
    f = sub.add_parser("fetch")
    f.add_argument("--lock", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--hosts", default="files.pythonhosted.org", help="+-separated uv.lock URL hosts to fetch from")
    f.add_argument("--min-mib", type=float, default=1.0)
    f.add_argument("--skip-prefix", default="", help="+-separated package-name prefixes to leave out (e.g. nvidia-)")
    v = sub.add_parser("verify")
    v.add_argument("--expected", required=True)
    v.add_argument("--plan", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--torch", default="2.11.0+cu128")
    v.add_argument("--project", default="fake-secret-average")
    a = ap.parse_args(argv)
    return {"rewrite": cmd_rewrite, "fetch": cmd_fetch, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
