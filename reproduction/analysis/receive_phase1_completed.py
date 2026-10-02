"""Read-only SFTP receiver for completed Phase 1 packages; never fits or loads a model.

Install Paramiko only in an isolated transfer-tools environment. Passwords are
prompted interactively and held in memory. OpenSSH known_hosts is authoritative.
Only atomically published ``attempts`` directories are ready for local intake.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import posixpath
import re
import socket
import stat
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ("checkpoint.pth", "pred.npy", "true.npy", "metrics.npy", "run.log")
REQUIRED_METADATA = ("authorization.json", "environment_fingerprint.json")
OPTIONAL_METADATA = ("batch_meta.json", "batch_progress.json", "manifest.csv")
MARKER = ".phase1-receiver.json"
MAX_JSON_BYTES = 4 * 1024 * 1024


class IntegrityError(RuntimeError):
    """Evidence needs review; do not overwrite any existing evidence."""


def fresh_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex


def remote_root_path(value: str) -> str:
    if not value.startswith("/") or "\\" in value or "\x00" in value or ".." in value.split("/"):
        raise ValueError("remote root must be an absolute POSIX path without traversal")
    result = posixpath.normpath(value)
    if result == "/":
        raise ValueError("remote root cannot be filesystem root")
    return result


def safe_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise IntegrityError("unsafe remote filename")
    return value


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_receipt(raw: bytes, attempt_id: str) -> dict:
    try:
        receipt = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise IntegrityError("invalid receipt JSON") from error
    if not isinstance(receipt, dict):
        raise IntegrityError("receipt must be an object")
    if receipt.get("status") != "completed_unreviewed":
        return receipt
    identity = re.fullmatch(r"(.+)__attempt-([0-9]+)", safe_name(attempt_id))
    if (not identity or int(identity[2]) < 1 or receipt.get("attempt_id") != attempt_id
            or receipt.get("logical_run_id") != identity[1]
            or receipt.get("attempt") != int(identity[2])):
        raise IntegrityError("receipt attempt identity differs from directory")
    hashes = receipt.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(ARTIFACTS):
        raise IntegrityError("receipt must bind exactly the five required artifacts")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in hashes.values()):
        raise IntegrityError("invalid artifact SHA256")
    return receipt


def prepare_local_root(path: Path, remote_root: str) -> Path:
    path = path.resolve()
    # In-repository output is permitted only as a dedicated Phase 1 child.
    if path == ROOT or path in ROOT.parents:
        raise ValueError("local root cannot contain the repository")
    if path.is_relative_to(ROOT) and path.parent != ROOT / "reproduction/results/phase1":
        raise ValueError("local root inside repository must be a new direct Phase 1 child")
    if path.is_relative_to(ROOT) and "intake" not in path.name:
        raise ValueError("receiver root must be an isolated intake directory")
    identity = {"receiver_version": "phase1-sftp-receiver/v1", "remote_root": remote_root}
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        marker = path / MARKER
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise FileExistsError("existing local root is not this receiver's isolated directory")
    else:
        path.mkdir(parents=True, exist_ok=True)
        with (path / MARKER).open("x", encoding="utf-8") as stream:
            json.dump(identity, stream, indent=2)
    for name in ("attempts", "incoming", "quarantine", "snapshots", "reports"):
        directory = path / name
        if directory.is_symlink():
            raise IntegrityError("receiver subdirectory cannot be a symlink")
        directory.mkdir(exist_ok=True)
    return path


class Receiver:
    def __init__(self, local_root: Path, remote_root: str, progress=print):
        self.remote_root = remote_root_path(remote_root)
        self.local_root = prepare_local_root(local_root, self.remote_root)
        self.progress = progress
        self.deadline: float | None = None
        # Only process-lifetime verified receipt bytes are cached. Restart
        # verifies every already-published artifact again, including checkpoints.
        self.verified: dict[str, bytes] = {}

    def remote_file(self, sftp, path: str):
        info = sftp.lstat(path)
        if not stat.S_ISREG(info.st_mode):
            raise IntegrityError("remote input is not a regular file: " + path)
        return info

    def read_small(self, sftp, path: str) -> bytes:
        info = self.remote_file(sftp, path)
        if info.st_size > MAX_JSON_BYTES:
            raise IntegrityError("metadata exceeds bounded size: " + path)
        with sftp.open(path, "rb") as stream:
            result = stream.read(MAX_JSON_BYTES + 1)
        if len(result) > MAX_JSON_BYTES or len(result) != info.st_size:
            raise IntegrityError("metadata changed size while reading: " + path)
        return result

    def snapshot_metadata(self, sftp, entries) -> Path:
        metadata = {}
        names = {entry.filename for entry in entries}
        optional = set(OPTIONAL_METADATA) | {
            safe_name(name) for name in names if name.endswith("manifest.csv")
        }
        for name in (*REQUIRED_METADATA, *sorted(optional - set(REQUIRED_METADATA))):
            if name not in names:
                if name in REQUIRED_METADATA:
                    raise IntegrityError("required root metadata missing: " + name)
                continue
            metadata[name] = self.read_small(sftp, posixpath.join(self.remote_root, name))
        snapshot = self.local_root / "snapshots" / fresh_name()
        snapshot.mkdir()
        for name, raw in metadata.items():
            with (snapshot / name).open("xb") as stream:
                stream.write(raw)
        return snapshot

    def verify_existing(self, directory: Path, raw: bytes, receipt: dict) -> None:
        if directory.is_symlink() or not directory.is_dir():
            raise IntegrityError("published attempt is not a normal directory")
        marker = directory / "receipt.json"
        if marker.is_symlink() or marker.read_bytes() != raw:
            raise IntegrityError("published receipt differs from remote; preserved for review")
        for name in ARTIFACTS:
            path = directory / name
            if path.is_symlink() or not path.is_file() or file_hash(path) != receipt["sha256"][name]:
                raise IntegrityError("published artifact hash mismatch; preserved for review: " + name)

    def download(self, sftp, remote: str, local: Path, expected: str) -> None:
        info = self.remote_file(sftp, remote)
        digest, transferred, next_update = hashlib.sha256(), 0, 25 * 1024 * 1024
        with sftp.open(remote, "rb") as source, local.open("xb") as target:
            while True:
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    raise TimeoutError("authorized receiver duration expired; partial files retained")
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                transferred += len(chunk)
                if local.name == "checkpoint.pth" and transferred >= next_update:
                    self.progress(f"checkpoint bytes {transferred}/{info.st_size}: {local.parent.name}")
                    next_update += 25 * 1024 * 1024
            target.flush()
            os.fsync(target.fileno())
        if transferred != info.st_size or digest.hexdigest() != expected:
            raise IntegrityError("artifact size or SHA256 mismatch: " + local.name)

    def receive_pass(self, sftp) -> dict:
        report = {"status": "running", "published": [], "skipped": [], "issues": [],
                  "remote_root": self.remote_root, "pooling_authorized": False,
                  "training_performed": False, "gate_table": []}
        try:
            entries = sftp.listdir_attr(self.remote_root)
            report["metadata_snapshot"] = str(self.snapshot_metadata(sftp, entries))
            for entry in sorted(entries, key=lambda item: item.filename):
                if not stat.S_ISDIR(entry.st_mode) or "__attempt-" not in entry.filename:
                    continue
                attempt_id = safe_name(entry.filename)
                remote = posixpath.join(self.remote_root, attempt_id)
                # lstat prevents traversing a substituted symlink directory.
                if not stat.S_ISDIR(sftp.lstat(remote).st_mode):
                    raise IntegrityError("attempt directory changed type")
                try:
                    raw = self.read_small(sftp, posixpath.join(remote, "receipt.json"))
                    receipt = parse_receipt(raw, attempt_id)
                except FileNotFoundError:
                    continue
                except IntegrityError as error:
                    report["issues"].append({"attempt_id": attempt_id, "error": str(error)})
                    continue
                if receipt.get("status") != "completed_unreviewed":
                    continue
                published = self.local_root / "attempts" / attempt_id
                if attempt_id in self.verified and self.verified[attempt_id] == raw:
                    report["skipped"].append(attempt_id)
                    continue
                if published.exists():
                    # Any existing mismatch is a hard error: never replace it.
                    self.verify_existing(published, raw, receipt)
                    self.verified[attempt_id] = raw
                    report["skipped"].append(attempt_id)
                    continue
                incoming = self.local_root / "incoming" / (attempt_id + "--" + fresh_name())
                incoming.mkdir()
                try:
                    for name in ARTIFACTS:
                        self.download(sftp, posixpath.join(remote, name), incoming / name,
                                      receipt["sha256"][name])
                    if self.read_small(sftp, posixpath.join(remote, "receipt.json")) != raw:
                        raise IntegrityError("receipt changed during transfer")
                    with (incoming / "receipt.json").open("xb") as stream:
                        stream.write(raw)
                        stream.flush()
                        os.fsync(stream.fileno())
                    self.verify_existing(incoming, raw, receipt)
                    if published.exists():
                        raise IntegrityError("destination appeared during transfer")
                    incoming.rename(published)
                except IntegrityError as error:
                    quarantined = self.local_root / "quarantine" / incoming.name
                    incoming.rename(quarantined)
                    report["issues"].append({"attempt_id": attempt_id, "error": str(error),
                                             "quarantine": str(quarantined)})
                    continue
                self.verified[attempt_id] = raw
                report["published"].append(attempt_id)
                self.progress("published completed package: " + attempt_id)
            report["status"] = "requires_resolution" if report["issues"] else "completed_descriptive_only"
            return report
        except Exception as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            report["gate_table"] = [
                {"check": "transport integrity", "input": "receipt and five SHA256-bound files",
                 "status": report["status"], "evidence": "published and issues records",
                 "next_permitted_action": "local CPU intake and fixed-checkpoint diagnostics"},
                {"check": "scientific use", "input": "received packages", "status": "requires-resolution",
                 "evidence": "transfer only; no retraining, pooling, ranking or equivalence claim",
                 "next_permitted_action": "approved intake and protocol review"}]
            with (self.local_root / "reports" / (fresh_name() + ".json")).open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--host", required=True)
    result.add_argument("--port", type=int, default=22)
    result.add_argument("--user", required=True)
    result.add_argument("--remote-root", default="/data/coding/DeReFusion/reproduction/results/phase1/rtx8000-timesnet-remaining-v1")
    result.add_argument("--local-root", type=Path, required=True)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="default: receive currently completed packages and exit")
    mode.add_argument("--poll-seconds", type=float, help="foreground polling interval, at least 10 seconds")
    result.add_argument("--max-hours", type=float, default=8, help="bounded polling duration; set remaining authorized GPU budget")
    result.add_argument("--known-hosts", type=Path, default=Path.home() / ".ssh/known_hosts")
    return result


def main() -> int:
    args = parser().parse_args()
    if (not 1 <= args.port <= 65535 or not 0 < args.max_hours <= 8
            or args.poll_seconds is not None and args.poll_seconds < 10):
        raise SystemExit("port must be valid, max-hours in (0, 8], polling at least 10 seconds")
    if not args.known_hosts.is_file():
        raise SystemExit("existing OpenSSH known_hosts is required; receiver never accepts new host keys")
    import paramiko  # isolated transfer-tools environment only; no runtime installation
    receiver = Receiver(args.local_root, args.remote_root)
    password = getpass.getpass(f"SSH password for {args.user}@{args.host}: ")
    deadline = time.monotonic() + args.max_hours * 3600
    receiver.deadline = deadline
    result = 0
    try:
        while time.monotonic() < deadline:
            client = paramiko.SSHClient()
            client.load_host_keys(str(args.known_hosts))
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            try:
                client.connect(args.host, port=args.port, username=args.user, password=password,
                               look_for_keys=False, allow_agent=False, timeout=30,
                               banner_timeout=30, auth_timeout=30)
                with client.open_sftp() as sftp:
                    sftp.get_channel().settimeout(60)
                    report = receiver.receive_pass(sftp)
                print(json.dumps({key: report[key] for key in ("status", "published", "skipped", "issues")}))
                if report["issues"]:
                    result = 2
            except (paramiko.AuthenticationException, paramiko.BadHostKeyException, IntegrityError) as error:
                print(f"receiver stopped: {type(error).__name__}: {error}")
                return 2
            except (paramiko.SSHException, socket.timeout, OSError, EOFError) as error:
                print(f"transport interrupted; partial evidence retained: {type(error).__name__}: {error}")
                if args.poll_seconds is None:
                    return 2
            finally:
                client.close()
            if args.poll_seconds is None:
                break
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        print("receiver stopped; all published and partial evidence retained")
        return 130
    finally:
        password = None
    return result


if __name__ == "__main__":
    raise SystemExit(main())
