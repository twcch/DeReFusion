"""Build and verify the result-free stage-1 manifest for the C1 blind audit.

The manifest contains only locked inputs, raw prediction/ground-truth arrays,
the executor manifest, the frozen analysis scripts and their explicit
dependency, and the result-free audit specifications.  It deliberately excludes
metrics and analyst outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


TAGS = (
    "AAPL", "MSFT", "AMZN", "META", "TSLA", "JPM", "XOM", "WMT",
    "N225", "GDAXI", "HSI", "FTSE", "RUT", "GBPUSD", "AUDUSD",
    "USDCAD", "GOLD", "WTI", "GLD", "TLT",
)
LOCKED_CSV_SHA256 = {
    "AAPL": "2d0ec4a90463b9a6bae915a3038d871c061b3d18d813d6572d50d75f4b185916",
    "MSFT": "48139f47b21a71466a60f7db045a20c85c3a75dcaba7afc20d175646f8b0d931",
    "AMZN": "28f348100dc6d056b7bd70aa472ff29923d3776b3eaf9a1385254e08b01e6771",
    "META": "fe0effbe8a541895f60e19c664e7f06791408fd46de31db0396f61b2590fa19d",
    "TSLA": "4993de1e6a0997785a6c40f5397374dcbe9254b5f5bd3f0633edf92c951e5b9d",
    "JPM": "450e3fec5302e7464e1a773e18a4d7f0132b58a926cac014f0e09e2083b1dc5e",
    "XOM": "74ecd45a77d1edb9692cadc61b853a381de1b028f50a290df4057f37f9b9c928",
    "WMT": "927d268a64b42742e6ae0b29c60a178beadf98506c0e6e1bc778c2e08f885481",
    "N225": "7250ac1596365ef894b61bb257789e23fe56a386a90e378901ce651fce4924a3",
    "GDAXI": "6c8cc51c65fa002dc3129fd12b2278c05ae8b9a3e1d33ce3e3bcb076f57dbee5",
    "HSI": "2a04256095a96d541142a426d289c5f173946b287ceefbd0ee2887c10cc25cec",
    "FTSE": "7c69f5dd2b1d6e8f78c228cd607646c1289a688e83e3c6eeef5735e2f276d13e",
    "RUT": "27f21f002ef041815e4a1db4acb8c6a0b2492832b43dfd2e79e0ad016bf1f718",
    "GBPUSD": "90e00d04d481138da937e5316ac8274728d622b30a117e755e33e9057c833c7d",
    "AUDUSD": "cd12448f0624128676d89c3bc7c0d6a12fc34a8b37309e689834804fa15d2a9a",
    "USDCAD": "4f346a1849e213ec83446c6d7cdcd450e4e13f15532ce9787cb26a05e09727ee",
    "GOLD": "76e42c67bcf264a2ce36a12e162a5f61f8807bde3954823ed074e23b77f8f4e5",
    "WTI": "0b7361bb11333d1f6fb889429d8e4869789f260eb6d5c457c1ecec6b936527f1",
    "GLD": "0b77dfa686fce94a851cf19ad1f0f23a45028a85423a5a9f218594b12f69b231",
    "TLT": "4da4b28f0adc0b50a763a4eba39f6e1f0893df85b9f3bac2769576cfb7671da1",
}
MODELS = ("DeReFusion", "revin-DLinear")
SEEDS = ("2021", "2022", "2023")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_dir(root: Path, tag: str, model: str, seed: str) -> Path:
    pattern = (
        f"long_term_forecast_{tag}_96_24_{model}_custom_ftMS_sl96_ll48_pl24_"
        f"dm32_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_"
        f"seed{seed}_0"
    )
    return root / "results" / pattern


def add_entry(rows: list[dict[str, str]], root: Path, role: str, path: Path,
              tag: str = "", model: str = "", seed: str = "") -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows.append(
        {
            "role": role,
            "tag": tag,
            "model": model,
            "seed": seed,
            "relpath": path.relative_to(root).as_posix(),
            "bytes": str(path.stat().st_size),
            "sha256": sha256(path),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reproduction/c1_handoff/BLIND_STAGE1_MANIFEST.csv"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output

    executor_manifest = root / "reproduction/c1_handoff/per_run_manifest.csv"
    with executor_manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))
    by_key = {(r["tag"], r["model"], r["seed"]): r for r in manifest_rows}
    expected = {(t, m, s) for t in TAGS for m in MODELS for s in SEEDS}
    if set(by_key) != expected or len(manifest_rows) != 120:
        raise ValueError("executor manifest is not the exact locked 20 x 2 x 3 grid")
    if any(r["status"] != "ok" for r in manifest_rows):
        raise ValueError("executor manifest contains a non-ok run")

    rows: list[dict[str, str]] = []
    for tag in TAGS:
        path = root / "dataset" / f"{tag}-2016-2025.csv"
        digest = sha256(path)
        if digest != LOCKED_CSV_SHA256[tag]:
            raise ValueError(
                f"locked CSV hash mismatch for {path}: {digest} != {LOCKED_CSV_SHA256[tag]}"
            )
        add_entry(rows, root, "locked_csv", path, tag)

    for tag, model, seed in sorted(expected):
        run = by_key[(tag, model, seed)]
        directory = result_dir(root, tag, model, seed)
        for kind, column in (("pred", "pred_sha256"), ("true", "true_sha256")):
            path = directory / f"{kind}.npy"
            digest = sha256(path)
            if digest != run[column]:
                raise ValueError(f"hash mismatch for {path}: {digest} != {run[column]}")
            add_entry(rows, root, f"raw_{kind}", path, tag, model, seed)

    add_entry(rows, root, "executor_manifest", executor_manifest)
    add_entry(
        rows, root, "frozen_script",
        root / "reproduction/analysis/analyze_volatility_regimes.py",
    )
    add_entry(rows, root, "frozen_script", root / "reproduction/analysis/c1_predictor.py")
    add_entry(
        rows, root, "frozen_dependency",
        root / "reproduction/analysis/structure_routing_experiment.py",
    )
    add_entry(
        rows, root, "audit_spec",
        root / "reports/evidence_closure/24a_c1_blind_recomputation_spec.md",
    )
    add_entry(
        rows, root, "lock_table",
        root / "reports/evidence_closure/24b_c1_blind_handoff_lock_table.md",
    )
    add_entry(
        rows, root, "audit_addendum",
        root / "reports/evidence_closure/24c_c1_blind_dependency_addendum.md",
    )

    if len(rows) != 267:
        raise AssertionError(f"expected 267 files, found {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("role", "tag", "model", "seed", "relpath", "bytes", "sha256"),
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["relpath"]))

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["role"]] = counts.get(row["role"], 0) + 1
    print(f"wrote {output.relative_to(root)} with {len(rows)} verified files")
    print(", ".join(f"{key}={counts[key]}" for key in sorted(counts)))


if __name__ == "__main__":
    main()
