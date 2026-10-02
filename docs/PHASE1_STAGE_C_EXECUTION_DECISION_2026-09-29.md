# Phase 1 Stage C execution decision

Status: approved prospectively on 2026-09-29 by the user's explicit decision,
“批准 Stage C 并行运行”. Execution amendment ID:
`phase1-stage-c-parallel-5060-20260929`. Scientific parent:
`phase1-v1.1-2026-09-19`; the Stage A disposition in
[`PHASE1_AMENDMENT_V1.2.md`](PHASE1_AMENDMENT_V1.2.md) remains in force.

This decision prospectively permits Stage C to execute on the local RTX 5060 Ti
while the separately authorized RTX 8000 Stage B technical supplement continues.
It amends the ordering requirement in the outsourcing plan §8 item 8 and the
Stage C execution stack only. Earlier mixed-stack B results remain partial and
nonconfirmatory; this authorization grants no pooling, equivalence, replacement,
or model-ranking use of those results. Stage D remains separately gated.

The full fixed Stage C grid is 30 assets × two primary models
(`revin-DLinear`, `DeReFusion`) × horizons 1 and 24 × seeds 2022, 2023, 2024:
360 fits. Both arms of every pair run on this same GPU and frozen software stack.
No B scores, bridge scores, rankings or pilot measurements select C rows or
change resource allocation. Data bytes, cohort separation, split boundaries,
prediction keys, model implementation, hyperparameters, checkpoint selection,
paired MSE estimand, bootstrap and multiplicity plan remain frozen. The original
scientific manifest and config retain their v1.1 identity and columns.

| Check | Input | Status and evidence | Next permitted action |
|---|---|---|---|
| Explicit stage authority | User decision dated 2026-09-29 | Approved; this prospective amendment | Bind new runtime authorization to reviewed HEAD |
| Fixed scientific grid | `C_confirmation.manifest.csv` | 360 rows; SHA-256 `bda784f0dd816939cdfd02e30109369df6a3e2e76aecd79aa8ee87b3910e6bfa` | Execute all rows without model or limit filters |
| Local environment | Existing `local-5060-manifests/environment_fingerprint.json` | SHA-256 `76b3b80aae7b1addcbc75088331d5d186feb07233b56d323d78bb1e88633c755`; observed values reviewed; historical supplemental track label retained | Reuse immutable observation bytes; new authorization declares C track |
| Source, data and keys | Exact launch HEAD and shared runner preflight | Must pass at launch; not asserted complete by this document | Launch only on matching source/environment and zero preflight issues |
| Coverage and interpretation | Independent C packages | No replacement by B or bridge packages; intake pending | Interpret only after full C coverage, integrity and stage-gate review |

The execution lock is Python 3.11.15, torch 2.7.1+cu128, CUDA runtime 12.8,
NumPy 2.1.2, pandas 2.3.3, scikit-learn 1.7.2, cuDNN 90701; exactly one
`NVIDIA GeForce RTX 5060 Ti`, UUID
`GPU-8b312e09-ea91-9e3a-c5d3-06b9362181b7`, driver 610.62. This is an approved
local Windows execution stack instead of the remote container lock; the saved
platform and complete package inventory are bound by the fingerprint hash.
No immutable container digest is claimed for this local execution. Existing
`run.py` TF32 defaults and deterministic `warn_only=True` settings remain as
frozen; unsupported deterministic operations remain visible in logs.

The new authorization version is `phase1-local-stage-c-auth/v1` and its track is
`confirmatory_local_stage_c`. The runtime credential binds exact source HEAD,
full manifest hash, environment hash, GPU identity, batch, output root and this
amendment. It is generated after the source commit and remains untracked; a
committed credential would invalidate its own source binding. If exact approval
seconds are unavailable, the writer uses actual UTC recording time without
backdating the decision. Existing authorizations and receipts are preserved.

## 中文执行层

本次明确批准 C 与远端 B 并行；C 使用本地 5060 Ti，远端 RTX 8000 原授权及任务继续。
C 输出仅写入 `reproduction/results/phase1/local-5060-stage-c-v1/`，批次名同目录名，
运行标签 `stagec_5060_v1`。命令直接选冻结 `C_confirmation` 清单，不添加科学清单列；
必须保留全部 360 fits，禁止 `--models` 或 `--limit` 筛选。

主任务在提交完成、源码与数据/键预检及测试通过后，运行
`python -m reproduction.batches.write_stage_c_5060_authorization --approved-by "user: 批准 Stage C 并行运行 (2026-09-29)"`。
随后共享 runner 使用独立授权与既有本地指纹启动：`--stage C_confirmation`、
`--device cuda`、上述输出根/批次/标签、`--execute --enable-recovery`、
`--max-hours 8 --fit-timeout-hours 2 --shutdown-buffer-minutes 5 --stop-below-seconds 60`。
这些预算控制属于执行安全边界，不按结果调整；未跑完仍须报告未完成，不作阶段结论。

预算中断保持原有 attempt 与逐 epoch 恢复语义，原中断 receipt 原样归档；真正技术失败
仍须另行重试授权。完成、失败、历史数组及 checkpoint 一律保留，不替换既有结果。
