"""Replay completed bridge checkpoints; diagnostic inference only, never training.

This audit does not call run.py or Exp.test (both write production artifacts).
Receipt CLI values and the reference commit's argparse defaults define every
configuration. Predictions have no equivalence threshold: all differences are
descriptive and authorize neither pooling nor ranking.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
import os
import platform
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE = ROOT / "reproduction/results/phase1"
MODULES = {
    "revin-DLinear": "models.revin-DLinear",
    "DeReFusion": "models.derefusion.DeReFusion",
    "revin-PatchTST": "models.revin-PatchTST",
    "revin-iTransformer": "models.revin-iTransformer",
    "revin-TimesNet": "models.revin-TimesNet",
}


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    """Hash dtype, dimensions and ordered bytes, not merely numeric values."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class BoundInputs:
    """Bind read-only inputs and fail if their bytes change during the audit."""
    def __init__(self):
        self.hashes: dict[str, str] = {}

    def bind(self, path: Path, expected: str | None = None) -> str:
        path = path.resolve()
        observed = file_hash(path)
        if expected is not None and observed != expected:
            raise RuntimeError(f"input hash mismatch: {path}")
        previous = self.hashes.get(str(path))
        if previous is not None and previous != observed:
            raise RuntimeError(f"input changed after binding: {path}")
        self.hashes[str(path)] = observed
        return observed

    def verify(self):
        for name, expected in self.hashes.items():
            if file_hash(Path(name)) != expected:
                raise RuntimeError(f"read-only input changed: {name}")


def parser_from_source(source: str) -> argparse.ArgumentParser:
    """Read literal parser declarations through AST without executing run.py."""
    tree = ast.parse(source)
    blocks = [node for node in tree.body if isinstance(node, ast.If)
              and isinstance(node.test, ast.Compare)
              and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
              and len(node.test.comparators) == 1
              and isinstance(node.test.comparators[0], ast.Constant)
              and node.test.comparators[0].value == "__main__"]
    if len(blocks) != 1:
        raise RuntimeError("run.py must have exactly one main parser block")
    parser = argparse.ArgumentParser(description="Frozen receipt configuration")
    found = 0
    for node in blocks[0].body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not (isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument"
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "parser"):
            continue
        positional = [ast.literal_eval(value) for value in call.args]
        keywords = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                raise RuntimeError("unpacked parser arguments are unsupported")
            if keyword.arg == "type":
                if not isinstance(keyword.value, ast.Name) or keyword.value.id not in {"int", "str", "float", "bool"}:
                    raise RuntimeError("nonliteral parser type is unsupported")
                keywords["type"] = {"int": int, "str": str, "float": float, "bool": bool}[keyword.value.id]
            else:
                keywords[keyword.arg] = ast.literal_eval(keyword.value)
        parser.add_argument(*positional, **keywords)
        found += 1
    if not found:
        raise RuntimeError("no literal argparse declarations found")
    return parser


def receipt_args(command: str, source: str) -> argparse.Namespace:
    # Both recorded POSIX commands and Windows commands use forward slashes in
    # their option paths. The executable is ignored and never executed.
    tokens = shlex.split(command, posix=True)
    run_indices = [index for index, token in enumerate(tokens)
                   if token.replace("\\", "/").split("/")[-1] == "run.py"]
    if len(run_indices) != 1:
        raise RuntimeError("receipt command must contain exactly one run.py")
    return parser_from_source(source).parse_args(tokens[run_indices[0] + 1:])


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def new_output_root(path: Path, protected: list[Path]) -> Path:
    path = path.resolve()
    for source in protected:
        source = source.resolve()
        if path == source or path in source.parents or source in path.parents:
            raise RuntimeError(f"audit output intersects protected input: {source}")
    if path == ROOT or path in ROOT.parents:
        raise RuntimeError("audit output cannot be the workspace or its ancestor")
    # Even an empty existing directory may represent a interrupted/reused run.
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def reference_attempt(track: str, reference: dict, local_root: Path, logical: str) -> Path:
    """Resolve indexed external inputs using the legacy index field names."""
    if track in {"p4", "external"}:
        return ROOT / reference["p4_attempt"]
    if track == "local":
        return local_root / f"{logical}__attempt-01"
    raise ValueError(f"unsupported reference track: {track}")


def reference_provenance(track: str, reference: dict, receipt: dict,
                         label: str | None = None) -> dict:
    execution_track = receipt.get("execution_track")
    source_label = (label or execution_track or "external") if track == "external" else track
    return {"reference_track": source_label,
            "reference_selection_track": track,
            "source_label": source_label,
            "source_execution_track": execution_track,
            "reference_source_commit": (reference["source_commit"] if track in {"p4", "external"}
                                        else receipt["authorization_commit"])}


def git_output(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True, encoding="utf-8")


def verify_source(commits: set[str], bound: BoundInputs) -> dict:
    paths = ["run.py", "data_provider/data_loader.py", "data_provider/data_factory.py",
             "utils/timefeatures.py", "exp/exp_basic.py", "exp/exp_long_term_forecasting.py",
             "reproduction/configs/phase1_modern_baselines.json"]
    paths.extend(module.replace(".", "/") + ".py" for module in MODULES.values())
    paths.extend(git_output("ls-files", "layers/*.py").splitlines())
    paths = sorted(set(paths))
    head = git_output("rev-parse", "HEAD").strip()
    normalized = {}
    for name in paths:
        bound.bind(ROOT / name)
        live = (ROOT / name).read_text(encoding="utf-8")
        head_source = git_output("show", f"{head}:{name}")
        if live != head_source:
            raise RuntimeError(f"source has uncommitted changes: {name}")
        normalized[name] = hashlib.sha256(live.encode()).hexdigest()
    for commit in sorted(commits):
        changed = git_output("diff", "--name-only", commit, head, "--", *paths).splitlines()
        if changed:
            raise RuntimeError(f"reference-to-HEAD source differs ({commit}): {changed}")
    return {"head_commit": head, "reference_commits": sorted(commits),
            "unchanged_paths": paths, "lf_normalized_source_sha256": normalized}


def validate_config(args, row, common, data: Path):
    for key, value in common.items():
        if getattr(args, key) != value:
            raise RuntimeError(f"receipt differs from frozen common: {key}")
    expected = {"task_name": "long_term_forecast", "model": row["model"],
                "data": "custom", "features": "MS", "target": "Close", "freq": "b",
                "pred_len": int(row["horizon"]), "rand_seed": int(row["seed"]),
                "split_mode": row["split_mode"], "data_path": data.name,
                "num_workers": 0, "use_amp": False, "inverse": False,
                "use_multi_gpu": False, "augmentation_ratio": 0}
    for key, value in expected.items():
        if getattr(args, key) != value:
            raise RuntimeError(f"receipt configuration mismatch: {key}")
    for key in ("train_end", "val_end", "test_end"):
        if (getattr(args, key) or "") != row[key]:
            raise RuntimeError(f"receipt split boundary mismatch: {key}")
    # This is the only environment/path adaptation; raw parsed values are saved.
    args.root_path = str(data.parent)


def key_hash_for_dataset(dataset, split) -> str:
    digest = hashlib.sha256()
    for window in range(len(dataset)):
        origin = dataset.data_dates[window + dataset.seq_len - 1].strftime("%Y-%m-%d")
        for lead in range(1, dataset.pred_len + 1):
            identity = {"split_manifest_id": split["split_manifest_id"],
                        "series_id": split["series_id"], "origin_timestamp": origin,
                        "forecast_horizon": dataset.pred_len, "horizon": lead,
                        "target_timestamp": dataset.data_dates[window + dataset.seq_len + lead - 1].strftime("%Y-%m-%d"),
                        "target": split["target"], "partition": "test"}
            digest.update((canonical_hash(identity) + "\n").encode("ascii"))
    return digest.hexdigest()


def differences(pred, reference, true) -> dict:
    if pred.shape != reference.shape or pred.dtype != reference.dtype:
        raise RuntimeError("replay/reference prediction shape or dtype differs")
    if not np.isfinite(pred).all() or not np.isfinite(reference).all():
        raise RuntimeError("nonfinite replay/reference prediction")
    current, old, truth = (array.astype(np.float64) for array in (pred, reference, true))
    delta = current - old
    norm = float(np.linalg.norm(delta.ravel()))
    ref_norm = float(np.linalg.norm(old.ravel()))
    current_mse = float(np.mean((current - truth) ** 2))
    old_mse = float(np.mean((old - truth) ** 2))
    return {"bitwise_equal": pred.tobytes() == reference.tobytes(),
            "max_absolute_prediction_difference": float(np.max(np.abs(delta))),
            "prediction_difference_l2": norm,
            "relative_prediction_difference_l2": norm / ref_norm if ref_norm else None,
            "mean_squared_prediction_difference": float(np.mean(delta ** 2)),
            "replay_mse_float64": current_mse, "reference_mse_float64": old_mse,
            "signed_mse_difference_float64": current_mse - old_mse,
            "numerical_equivalence_threshold": "TBD",
            "numerical_equivalence_claim": False}


def replay(args, checkpoint: Path, device):
    import torch
    from data_provider.data_loader import Dataset_Custom
    from torch.utils.data import DataLoader

    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.rand_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # TF32 flags remain the installed stack's defaults, as in run.py.
    dataset = Dataset_Custom(args, args.root_path, flag="test",
                             size=[args.seq_len, args.label_len, args.pred_len],
                             features=args.features, data_path=args.data_path,
                             target=args.target, timeenc=int(args.embed == "timeF"),
                             freq=args.freq, seasonal_patterns=args.seasonal_patterns)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)
    model = importlib.import_module(MODULES[args.model]).Model(args).float().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    model.requires_grad_(False)
    predictions, truths, batch_records = [], [], []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            cpu = [value.float() for value in batch]
            batch_records.append({"batch_index": index, "samples": int(cpu[0].shape[0]),
                                  "float32_tensors_sha256": [array_hash(value.numpy()) for value in cpu]})
            x, y, x_mark, y_mark = [value.to(device) for value in cpu]
            decoder = torch.cat([y[:, :args.label_len, :],
                                 torch.zeros_like(y[:, -args.pred_len:, :])], dim=1).float().to(device)
            output = model(x, x_mark, decoder, y_mark)
            predictions.append(output[:, -args.pred_len:, -1:].detach().cpu().numpy())
            truths.append(y[:, -args.pred_len:, -1:].detach().cpu().numpy())
    pred, true = np.concatenate(predictions, axis=0), np.concatenate(truths, axis=0)
    scaler = {name: array_hash(np.asarray(getattr(dataset.scaler, name)))
              for name in ("mean_", "scale_", "var_", "n_samples_seen_")}
    metadata = {"input_batches": batch_records, "input_batches_sha256": canonical_hash(batch_records),
                "dataset_arrays_sha256": {name: array_hash(getattr(dataset, name))
                                           for name in ("data_x", "data_y", "data_stamp")},
                "scaler_arrays_sha256": scaler, "scaler_sha256": canonical_hash(scaler),
                "scaler_reconstruction": "Dataset_Custom historical training-prefix StandardScaler; no model fitting",
                "split_metadata": dataset.split_metadata,
                "batch_size": args.batch_size, "shuffle": False, "drop_last": False,
                "samples": len(dataset), "last_batch_samples": batch_records[-1]["samples"],
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "tf32": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                         "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "gradient_mode": "inference_mode", "amp": False, "strict_load": True,
                "weights_only": True}
    del model
    return pred, true, dataset, metadata


def audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-settings", type=int, default=0, help="0 means all fixed manifest settings")
    parser.add_argument("--reference-track", choices=("p4", "local", "both", "external"), default="both")
    parser.add_argument("--reference-label", help="External source label; defaults to the receipt execution_track")
    parser.add_argument("--bridge-manifest", type=Path, default=PHASE / "local-5060-manifests/bridge-v1.csv")
    parser.add_argument("--reference-index", type=Path, default=PHASE / "local-5060-manifests/bridge-p4-reference-index.csv")
    parser.add_argument("--local-root", type=Path, default=PHASE / "local-5060-bridge-v1")
    return parser


def main() -> int:
    parser = audit_parser()
    cli = parser.parse_args()
    if cli.max_settings < 0:
        parser.error("--max-settings must be nonnegative")
    if cli.reference_label is not None:
        if cli.reference_track != "external":
            parser.error("--reference-label requires --reference-track external")
        if not cli.reference_label.strip():
            parser.error("--reference-label must be nonempty")
    import torch
    from reproduction.batches.run_phase1_stages import CONFIG, logical_id, split_for, validate_preflight

    device = torch.device(cli.device)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError(f"requested replay device unavailable or unsupported: {device}")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    bound = BoundInputs()
    for path in (CONFIG, cli.bridge_manifest, cli.reference_index,
                 ROOT / "reproduction/results/dataset_registry.csv",
                 PHASE / "phase1_data_contract.csv", PHASE / "phase1_split_manifest.csv",
                 PHASE / "phase1_prediction_keys.csv.gz", Path(__file__)):
        bound.bind(path)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    selected = read_rows(cli.bridge_manifest)
    if len({logical_id(row) for row in selected}) != len(selected):
        raise RuntimeError("duplicate bridge logical settings")
    if cli.max_settings:
        selected = selected[:cli.max_settings]
    if not selected:
        raise RuntimeError("bridge manifest has no settings")
    index = {row["logical_run_id"]: row for row in read_rows(cli.reference_index)}
    registry, _, issues = validate_preflight(selected)
    if issues:
        raise RuntimeError(f"frozen data/key preflight failed: {issues}")
    tracks = ("p4", "local") if cli.reference_track == "both" else (cli.reference_track,)
    prepared, commits = [], set()
    protected = [ROOT / "dataset", cli.local_root, PHASE / "p4-partial-20260929",
                 cli.bridge_manifest, cli.reference_index, CONFIG]
    sources = {}
    for row in selected:
        logical = logical_id(row)
        data = ROOT / registry[row["asset"]]["file"]
        bound.bind(data, registry[row["asset"]]["sha256"])
        split = split_for(row)
        reference = index[logical]
        for track in tracks:
            attempt = reference_attempt(track, reference, cli.local_root, logical)
            protected.append(attempt)
            receipt_path = attempt / "receipt.json"
            bound.bind(receipt_path, reference["receipt_sha256"] if track in {"p4", "external"} else None)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "completed_unreviewed" or receipt.get("logical_run_id") != logical:
                raise RuntimeError(f"reference receipt identity/status mismatch: {receipt_path}")
            for key, value in {"dataset_sha256": registry[row["asset"]]["sha256"],
                               "split_manifest_id": split["split_manifest_id"],
                               "prediction_keys_sha256": split["prediction_keys_sha256"],
                               "protocol_version": row["protocol_version"]}.items():
                if receipt.get(key) != value:
                    raise RuntimeError(f"reference receipt identity mismatch: {key}")
            for name in ("checkpoint.pth", "pred.npy", "true.npy"):
                bound.bind(attempt / name, receipt["sha256"][name])
            provenance = reference_provenance(track, reference, receipt, cli.reference_label)
            commit = provenance["reference_source_commit"]
            commits.add(commit)
            if commit not in sources:
                sources[commit] = git_output("show", f"{commit}:run.py")
            args = receipt_args(receipt["command"], sources[commit])
            raw_config = vars(args).copy()
            if raw_config != vars(receipt_args(receipt["command"], (ROOT / "run.py").read_text(encoding="utf-8"))):
                raise RuntimeError("current/reference CLI parser defaults differ")
            validate_config(args, row, config["common"], data)
            prepared.append((row, track, attempt, receipt, args, raw_config, split, provenance))
    source_check = verify_source(commits, bound)
    bound.verify()
    output = new_output_root(cli.output_root, protected)
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version(), "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "initial_tf32": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}}
    report = {"audit_version": "phase1-checkpoint-forward/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "Fixed checkpoint inference implementation/stack diagnostic; no training or new model fits",
              "selection": "Fixed bridge manifest order; no score-driven selection",
              "settings": len(selected),
              "reference_tracks": list(dict.fromkeys(item[-1]["reference_track"] for item in prepared)),
              "reference_selection_tracks": list(tracks), "environment": environment,
              "source_provenance": source_check, "input_file_sha256": bound.hashes,
              "numerical_equivalence_threshold": "TBD", "pooling_authorized": False,
              "ranking_authorized": False, "numerical_equivalence_claim": False,
              "results": [], "status": "running"}
    write_json(output / "audit-plan.json", report)
    started = time.monotonic()
    try:
        for ordinal, (row, track, attempt, receipt, args, raw_config, split, provenance) in enumerate(prepared, 1):
            logical = logical_id(row)
            print(f"[{ordinal}/{len(prepared)}] {provenance['reference_track']} {logical}", flush=True)
            begin = time.monotonic()
            pred, true, dataset, metadata = replay(args, attempt / "checkpoint.pth", device)
            keys_hash = key_hash_for_dataset(dataset, split)
            if keys_hash != split["prediction_keys_sha256"]:
                raise RuntimeError(f"rederived prediction keys differ: {logical}")
            ref_pred = np.load(attempt / "pred.npy", allow_pickle=False)
            ref_true = np.load(attempt / "true.npy", allow_pickle=False)
            expected_shape = (int(split["window_count"]), int(row["horizon"]), 1)
            if pred.shape != expected_shape or true.shape != expected_shape or ref_true.shape != expected_shape:
                raise RuntimeError(f"replay/reference shapes differ from frozen keys: {logical}")
            if true.dtype != ref_true.dtype or true.tobytes() != ref_true.tobytes() or not np.isfinite(true).all():
                raise RuntimeError(f"rederived truth is not bitwise identical: {logical}")
            folder = output / f"{logical}__{track}"
            folder.mkdir()
            # New audit arrays only; originals and production receipts are read-only.
            with (folder / "pred.npy").open("xb") as stream:
                np.save(stream, pred, allow_pickle=False)
            result = {"logical_run_id": logical, **provenance,
                      "reference_attempt": str(attempt.resolve()), "receipt_command": receipt["command"],
                      "receipt_config": raw_config, "effective_root_path": args.root_path,
                      "checkpoint_sha256": receipt["sha256"]["checkpoint.pth"],
                      "reference_pred_file_sha256": receipt["sha256"]["pred.npy"],
                      "reference_true_file_sha256": receipt["sha256"]["true.npy"],
                      "reference_pred_array_sha256": array_hash(ref_pred),
                      "replay_pred_array_sha256": array_hash(pred), "replay_true_array_sha256": array_hash(true),
                      "prediction_keys_sha256": keys_hash, "truth_bitwise_equal": True,
                      "replay_pred_file_sha256": file_hash(folder / "pred.npy"),
                      "shape": list(pred.shape), "inference": metadata,
                      "differences": differences(pred, ref_pred, true), "elapsed_seconds": time.monotonic() - begin}
            for name in ("checkpoint.pth", "pred.npy", "true.npy"):
                bound.bind(attempt / name, receipt["sha256"][name])
            write_json(folder / "audit.json", result)
            report["results"].append(result)
            print(f"  truth exact; max_abs={result['differences']['max_absolute_prediction_difference']:.9g}; "
                  f"MSE_delta={result['differences']['signed_mse_difference_float64']:.9g}", flush=True)
        bound.verify()
        report["status"] = "completed_descriptive_only"
        report["input_immutability_verified"] = True
        report["gate_table"] = [
            {"check": "frozen identities, keys, source, strict weights, shapes, truth",
             "input": "fixed bridge manifest, reference receipts/checkpoints, data and prediction keys", "status": "eligible",
             "evidence": "all per-setting audit.json records and input SHA-256 inventory", "next_permitted_action": "descriptive audit review"},
            {"check": "numerical equivalence tolerance", "input": "fixed-checkpoint replay differences",
             "status": "requires-resolution", "evidence": "TBD; none precommitted",
             "next_permitted_action": "research-question/precision analysis and approved protocol required before an equivalence claim"},
            {"check": "pooling/ranking", "input": "existing descriptive bridge authorization",
             "status": "excluded-by-protocol", "evidence": "descriptive bridge authorization",
             "next_permitted_action": "keep execution tracks separate"}]
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "report.json", report)
        print(f"Audit report: {output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
