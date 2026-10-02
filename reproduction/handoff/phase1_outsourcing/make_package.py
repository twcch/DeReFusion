"""Create a deterministic Phase 1 source handoff archive from a clean commit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise SystemExit("refusing to export a dirty worktree")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary) / "source.tar"
        with staged.open("wb") as handle:
            subprocess.run(
                ["git", "archive", "--format=tar", f"--prefix=DeReFusion-{commit[:12]}/", commit],
                cwd=ROOT,
                check=True,
                stdout=handle,
            )
        prefix = f"DeReFusion-{commit[:12]}/"
        plain_tar = Path(temporary) / "handoff.tar"
        generated = {
            prefix + "reproduction/handoff/phase1_outsourcing/FROZEN_COMMIT.txt":
                (commit + "\n").encode("utf-8"),
            prefix + "reproduction/handoff/phase1_outsourcing/package_manifest.json":
                (json.dumps({"frozen_commit": commit, "protocol_version": "phase1-v1.1-2026-09-19"},
                            indent=2, sort_keys=True) + "\n").encode("utf-8"),
        }
        with tarfile.open(staged, "r") as source, tarfile.open(plain_tar, "w") as target:
            for member in source.getmembers():
                if member.name in generated:
                    continue
                member.mtime = 0
                member.uid = member.gid = 0
                member.uname = member.gname = ""
                target.addfile(member, source.extractfile(member) if member.isfile() else None)
            for name, content in generated.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mtime = 0
                info.mode = 0o644
                target.addfile(info, io.BytesIO(content))
        with plain_tar.open("rb") as source, args.output.open("wb") as raw_output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, compresslevel=9, mtime=0) as compressed:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    compressed.write(chunk)
    result = {"commit": commit, "archive": str(args.output), "sha256": sha256(args.output)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
