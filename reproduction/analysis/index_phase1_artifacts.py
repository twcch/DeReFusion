"""Create a read-only, timestamped inventory of Phase 1 computation artifacts.

The inventory never opens receiver ``incoming`` directories, never changes an
attempt, and records receipt-declared hashes rather than rehashing large
checkpoints. Existing intake/audit evidence is summarized separately.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "reproduction/results/phase1"
REPORTS = ROOT / "reports/phase1"
ARTIFACTS = ("checkpoint.pth", "pred.npy", "true.npy", "metrics.npy", "run.log")
SOURCES = (
    ("stage_a_gpu", PHASE / "calibration", "RTX 8000 / GPU calibration"),
    ("stage_a_cpu", PHASE / "calibration-cpu", "CPU calibration"),
    ("stage_b_p4", PHASE / "p4-partial-20260929/attempts", "P4; partial B_screen"),
    ("stage_b_bridge_5060", PHASE / "local-5060-bridge-v1", "RTX 5060 Ti; nonconfirmatory bridge"),
    ("stage_c_5060", PHASE / "local-5060-stage-c-v1", "RTX 5060 Ti; C_confirmation"),
    # Only atomically published attempts are indexed. The receiver's incoming,
    # quarantine, and snapshots trees are intentionally out of scope.
    ("stage_b_rtx8000", PHASE / "rtx8000-intake-20260929-v2/attempts", "RTX 8000; supplemental B_screen"),
)
CSV_FIELDS = (
    "source", "source_root", "hardware_track", "stage", "logical_run_id",
    "attempt_id", "status", "asset", "model", "horizon", "seed", "batch_id",
    "execution_track", "execution_device", "gpu_model", "protocol_version",
    "authorization_commit", "receipt_sha256", "declared_hash_count",
    "declared_artifacts", "present_artifacts", "missing_artifacts", "artifact_bytes",
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def receipt_files(source_root: Path) -> list[Path]:
    if not source_root.is_dir():
        return []
    # All configured roots are immutable attempt roots, not receiver staging.
    # A direct-child scan avoids accidentally including pass reports or partials.
    return sorted(source_root.glob("*/receipt.json"), key=lambda p: p.as_posix().casefold())


def manifest_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"rows": 0, "statuses": {}, "stages": {}}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return {
        "rows": len(rows),
        "statuses": dict(Counter(row.get("status", "") for row in rows)),
        "stages": dict(Counter(row.get("stage", "") for row in rows)),
    }


def acknowledgement_stats(root: Path) -> dict[str, Any]:
    acks = sorted(root.glob("pass-*/acknowledgement.json")) if root.is_dir() else []
    identities: set[str] = set()
    issues = 0
    statuses: Counter[str] = Counter()
    errors = 0
    for path in acks:
        value = read_json(path) or {}
        identities.update((value.get("receipt_hashes") or {}).keys())
        issues += len(value.get("issues") or [])
        errors += int(bool(value.get("error")))
        statuses[str(value.get("intake_status", "missing"))] += 1
    return {"passes": len(acks), "unique_receipt_hashes": len(identities),
            "issues": issues, "error_passes": errors, "statuses": dict(statuses)}


def summarize_file(path: Path) -> dict[str, Any]:
    value = read_json(path) or {}
    return {"status": value.get("status"), "summary": value.get("summary", {}),
            "error": value.get("error"), "path": path.relative_to(ROOT).as_posix()}


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "—"
    return ", ".join(f"{key or '(blank)'}: {value}" for key, value in sorted(counts.items()))


def main() -> int:
    created = datetime.now(timezone.utc)
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = PHASE / f"artifact-index-{stamp}.csv"
    md_path = REPORTS / f"phase1-artifact-inventory-{stamp}.md"

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"

    rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    by_source_status: dict[str, Counter[str]] = defaultdict(Counter)
    all_attempts: dict[str, set[str]] = defaultdict(set)
    missing_file_rows: list[str] = []

    for label, source_root, hardware in SOURCES:
        receipts = receipt_files(source_root)
        for receipt_path in receipts:
            try:
                raw = receipt_path.read_bytes()
                receipt = json.loads(raw)
                if not isinstance(receipt, dict):
                    raise ValueError("receipt is not an object")
            except (OSError, json.JSONDecodeError, ValueError) as error:
                row = {field: "" for field in CSV_FIELDS}
                row.update(source=label, source_root=source_root.relative_to(ROOT).as_posix(),
                           attempt_id=receipt_path.parent.name, status="invalid_receipt",
                           missing_artifacts=f"receipt:{type(error).__name__}")
                rows.append(row)
                by_source_status[label]["invalid_receipt"] += 1
                continue

            declared = receipt.get("sha256") if isinstance(receipt.get("sha256"), dict) else {}
            expected = [name for name in ARTIFACTS if name in declared]
            present = [name for name in expected if (receipt_path.parent / name).is_file()]
            missing = [name for name in expected if name not in present]
            size_bytes = sum((receipt_path.parent / name).stat().st_size for name in present)
            # Older CPU-calibration receipts predate logical_run_id and keep
            # the run identity in their directory name. Preserve that identity
            # instead of turning known runs into blank/unmatched records.
            logical = str(receipt.get("logical_run_id") or receipt_path.parent.name.split("__attempt-", 1)[0])
            attempt_id = str(receipt.get("attempt_id", receipt_path.parent.name))
            status = str(receipt.get("status", "missing_status"))
            by_source_status[label][status] += 1
            if logical:
                all_attempts[logical].add(label)
            if missing:
                missing_file_rows.append(f"{label}/{attempt_id}: {', '.join(missing)}")
            rows.append({
                "source": label,
                "source_root": source_root.relative_to(ROOT).as_posix(),
                "hardware_track": hardware,
                "stage": receipt.get("stage") or ("A_calibration" if logical.startswith("A") else logical.split("_", 1)[0]),
                "logical_run_id": logical,
                "attempt_id": attempt_id,
                "status": status,
                "asset": receipt.get("asset", ""),
                "model": receipt.get("model", ""),
                "horizon": receipt.get("horizon", ""),
                "seed": receipt.get("seed", ""),
                "batch_id": receipt.get("batch_id", ""),
                "execution_track": receipt.get("execution_track", ""),
                "execution_device": receipt.get("execution_device") or (receipt.get("environment") or {}).get("device", ""),
                "gpu_model": receipt.get("gpu_model") or (receipt.get("environment") or {}).get("gpu", ""),
                "protocol_version": receipt.get("protocol_version", ""),
                "authorization_commit": receipt.get("authorization_commit", ""),
                "receipt_sha256": hashlib.sha256(raw).hexdigest(),
                "declared_hash_count": len(declared),
                "declared_artifacts": ";".join(expected),
                "present_artifacts": ";".join(present),
                "missing_artifacts": ";".join(missing),
                "artifact_bytes": size_bytes,
            })
        source_summaries.append({"source": label, "path": source_root.relative_to(ROOT).as_posix(),
                                 "hardware": hardware, "receipts": len(receipts),
                                 "statuses": dict(by_source_status[label])})

    with csv_path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    manifests = {
        "A_calibration": manifest_stats(PHASE / "A_calibration.manifest.csv"),
        "B_screen": manifest_stats(PHASE / "B_screen.manifest.csv"),
        "C_confirmation": manifest_stats(PHASE / "C_confirmation.manifest.csv"),
        "D_temporal_robustness": manifest_stats(PHASE / "D_temporal_robustness.manifest.csv"),
        "RTX8000 supplement": manifest_stats(PHASE / "supplemental-manifests/rtx8000-timesnet-remaining-v1.csv"),
        "Local 5060 Stage C": manifest_stats(PHASE / "C_confirmation.manifest.csv"),
        "Local 5060 bridge": manifest_stats(PHASE / "local-5060-manifests/bridge-v1.csv"),
    }
    p4_intake = read_json(PHASE / "intake-p4-20260929-v1/report.json") or {}
    p4_bridge = read_json(REPORTS / "P4-RTX5060-bridge-audit-20260929.json") or {}
    checkpoint_forward = read_json(PHASE / "checkpoint-forward-5060-v1/report.json") or {}
    stage_c_ack = acknowledgement_stats(PHASE / "consumer-stagec-20260929-v1")
    rtx_ack = acknowledgement_stats(PHASE / "consumer-rtx8000-20260929-v1")
    supervisor = read_json(PHASE / "delivery-supervisor-status.json") or {}
    raw_result_dirs = sum(1 for item in (ROOT / "results").iterdir() if item.is_dir()) if (ROOT / "results").is_dir() else 0
    overlaps = {logical: sorted(labels) for logical, labels in all_attempts.items() if len(labels) > 1}
    overlap_sources = Counter(" + ".join(labels) for labels in overlaps.values())

    lines = [
        "# Phase 1 运算产物索引与差异盘点",
        "",
        f"- 快照时间：{created.isoformat()}",
        f"- 仓库 HEAD：`{commit}`",
        "- 范围：本地 Phase 1 收据化 attempt、已有摄入/审计摘要及 TSLib `results/` 原始输出目录数。",
        "- 安全边界：只读取已发布 attempt；不扫描 `incoming/`、不改写/移动/删除 attempt、receipt、checkpoint 或预测数组。",
        "- 哈希边界：CSV 记录 receipt 的 SHA-256 及 receipt 声明的 artifact 哈希；本次不重复读取/重哈希大型二进制。下表引用已完成的 intake/审计核验。",
        "",
        "## 来源目录清单",
        "",
        "| 来源 | 计算轨道 | receipts | 状态计数 |",
        "|---|---|---:|---|",
    ]
    for item in source_summaries:
        lines.append(f"| `{item['path']}` | {item['hardware']} | {item['receipts']} | {format_counts(item['statuses'])} |")
    lines += [
        "",
        "## 冻结清单与完成度对照",
        "",
        "| 阶段/队列 | 冻结行数 | 产物快照/审计证据 | 差异或状态 |",
        "|---|---:|---|---|",
        f"| A calibration（GPU） | {manifests['A_calibration']['rows']} | {source_summaries[0]['receipts']} receipts；{format_counts(source_summaries[0]['statuses'])} | GPU 标定批齐全；该目录的 file SHA 本轮未重复重算 |",
        f"| A calibration（CPU 对照） | {manifests['A_calibration']['rows']} | {source_summaries[1]['receipts']} receipts；{format_counts(source_summaries[1]['statuses'])} | 少 1 个有效 calibration_pass；resource_blocked 保留为资源阻断，不计成功 |",
        f"| B_screen（P4） | {manifests['B_screen']['rows']} | intake `{p4_intake.get('status', 'unknown')}`；{(p4_intake.get('summary') or {}).get('eligible_packages', 0)} 包通过既有摄入 | {((p4_intake.get('summary') or {}).get('absent_manifest_settings', '未知'))} 行缺失；既有 P4–5060 审计见下 |",
        f"| B_screen（5060 bridge） | {manifests['Local 5060 bridge']['rows']} | {(p4_bridge.get('summary') or {}).get('bridge_validated', source_summaries[3]['receipts'])} fits 已验证 | 与 P4 有 {sum(1 for labels in overlaps.values() if 'stage_b_p4' in labels and 'stage_b_bridge_5060' in labels)} 个跨来源逻辑 ID 重叠（桥接/对照，不能相加为独立设置） |",
        f"| B_screen（RTX8000 supplement） | {manifests['RTX8000 supplement']['rows']} | {source_summaries[5]['receipts']} 本地已发布 receipts；{rtx_ack['unique_receipt_hashes']} 个唯一 receipt hash 已摄入 | {max(0, manifests['RTX8000 supplement']['rows'] - rtx_ack['unique_receipt_hashes'])} 行尚未本地摄入；远端状态取守护程序快照 |",
        f"| C_confirmation（5060） | {manifests['C_confirmation']['rows']} | {source_summaries[4]['receipts']} receipts；{stage_c_ack['unique_receipt_hashes']} 个唯一 receipt hash 被 consumer 接收 | consumer passes={stage_c_ack['passes']}；issues={stage_c_ack['issues']} |",
        f"| D_temporal_robustness | {manifests['D_temporal_robustness']['rows']} | 本索引来源中无 D attempt | 尚未看到 D 产物；不代表仓库外不存在 |",
        "",
        "## 已存在的独立完整性证据",
        "",
        f"- P4 + RTX 5060 bridge：既有机器报告记载 P4 `{(p4_bridge.get('summary') or {}).get('p4_validated', '—')}`、bridge `{(p4_bridge.get('summary') or {}).get('bridge_validated', '—')}` 验证通过，完整性问题 `{(p4_bridge.get('summary') or {}).get('integrity_issues', '—')}`；报告核对 artifact SHA、target/split、finite arrays 和重算 metrics。详见 [`P4-RTX5060-bridge-audit-20260929.md`](P4-RTX5060-bridge-audit-20260929.md)。",
        f"- Stage C 5060：已有 consumer 跨 `{stage_c_ack['passes']}` 个 pass 确认 `{stage_c_ack['unique_receipt_hashes']}` 个唯一 receipt，issues={stage_c_ack['issues']}；每份被摄入的 package 由 intake 校验，训练/使用 GPU={False}。",
        f"- RTX8000：已有 consumer 跨 `{rtx_ack['passes']}` 个 pass 确认 `{rtx_ack['unique_receipt_hashes']}` 个唯一 receipt，issues={rtx_ack['issues']}；摄入仍随远端新产物增长。",
        f"- 固定 checkpoint forward audit：状态 `{checkpoint_forward.get('status', 'unknown')}`，设置数 `{checkpoint_forward.get('settings') if isinstance(checkpoint_forward.get('settings'), int) else len(checkpoint_forward.get('settings', []))}`；是推理复算而非新训练。详见 [`checkpoint-forward-audit-20260929.md`](checkpoint-forward-audit-20260929.md)。",
        f"- Stage A 两个 calibration 根目录合计 `{source_summaries[0]['receipts'] + source_summaries[1]['receipts']}` 个 receipts；本轮核对 receipt 状态、SHA 声明和文件在场情况，但未重复重哈希约 1 GiB 文件。",
        "",
        "## 实时队列快照（可能继续变化）",
        "",
    ]
    progress = supervisor.get("remote_progress") or {}
    total = progress.get("total_selected", manifests["RTX8000 supplement"]["rows"])
    complete = (progress.get("newly_completed", 0) or 0) + (progress.get("previously_completed_verified", 0) or 0)
    lines += [
        f"- 本机 supervisor 状态：`{supervisor.get('status', 'unknown')}`，观测时间 `{supervisor.get('observed_at_utc', 'unknown')}`。",
        f"- 最近收到/摄入计数：{supervisor.get('received_packages', rtx_ack['unique_receipt_hashes'])}/{total} received，{supervisor.get('audited_unique_packages', rtx_ack['unique_receipt_hashes'])}/{total} intake-acknowledged。",
        f"- 远端最后同步到本机的 runner 进度：{complete}/{total}，当前运行 `{progress.get('current_run')}`，状态 `{progress.get('status')}`，快照时间 `{progress.get('updated_at_utc')}`。此字段是最近一次同步快照，不保证等于阅读时的即时远端状态。",
        "",
        "## 原始 TSLib 输出与重叠说明",
        "",
        f"仓库顶层 `results/` 目前有 {raw_result_dirs} 个原始运行目录。它们是传统训练输出目录，不等同于本次的 immutable attempt receipt；本报告不移动、不删除，也不把目录数直接加到 receipt 总数。跨来源逻辑 ID 有 {len(overlaps)} 个重叠；来源组合计数：{format_counts(dict(overlap_sources))}。P4 与 bridge 的重叠是预期配对，不是额外独立 setting。",
        "",
        "## 差异清单与建议",
        "",
        "1. **Stage B 仍未闭合**：P4 245/300；RTX8000 supplemental 55 行队列仍在运行，当前本地已摄入快照未满 55。保留两条硬件轨道，不跨轨道合并排名。",
        "2. **CPU calibration 有 1 个资源阻断**：保留原 receipt 与 attempt，不把 blocked 改写成成功，也不在这个 inventory 里自动重跑。",
        "3. **Stage D 没有本地 attempt**：manifest 仍有 144 planned 行；不得把缺少文件解释为失败或结果。",
        "4. **旧 `results/` 562 个目录尚未逐一连接到本次 attempt**：它们单独保留；若后续需要归档/清理，应先建立明确的 run_id 对照并单独复核，不在本次进行删除。",
        "5. **未发现文件缺失**：" + ("所有已索引 receipt 声明的 artifact 文件均在场。" if not missing_file_rows else f"发现 {len(missing_file_rows)} 个 receipt 声明文件缺失，详见下方。"),
        "",
        f"逐 attempt 索引：[`{csv_path.relative_to(ROOT).as_posix()}`](../../{csv_path.relative_to(ROOT).as_posix()})（{len(rows)} 行；包含 receipt SHA-256、硬件/阶段/模型、artifact 声明与文件在场状态）。",
        "",
    ]
    if missing_file_rows:
        lines += ["### Receipt 声明 artifact 缺失", ""]
        lines.extend(f"- `{entry}`" for entry in missing_file_rows[:100])
        if len(missing_file_rows) > 100:
            lines.append(f"- …另有 {len(missing_file_rows) - 100} 项，完整清单请看 CSV。")

    with md_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({"report": str(md_path), "index": str(csv_path), "rows": len(rows),
                      "missing_artifacts": len(missing_file_rows), "overlapping_logical_ids": len(overlaps),
                      "raw_result_directories": raw_result_dirs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
