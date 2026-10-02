"""CPU-only safety gates for the fixed-checkpoint inference audit."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproduction.analysis.audit_phase1_checkpoint_forward import (
    ROOT, BoundInputs, array_hash, audit_parser, differences, new_output_root,
    parser_from_source, receipt_args, reference_attempt, reference_provenance, write_json,
)


class CheckpointForwardAuditTest(unittest.TestCase):
    def test_external_cli_keeps_existing_track_defaults(self):
        parser = audit_parser()
        self.assertEqual(parser.parse_args(["--output-root", "audit"]).reference_track, "both")
        for track in ("p4", "local", "both"):
            self.assertEqual(parser.parse_args(["--output-root", "audit", "--reference-track", track]).reference_track, track)
        args = parser.parse_args(["--output-root", "audit", "--reference-track", "external",
                                  "--reference-label", "rtx8000"])
        self.assertEqual((args.reference_track, args.reference_label), ("external", "rtx8000"))

    def test_external_uses_indexed_attempt_and_source_commit(self):
        reference = {"p4_attempt": "received/rtx8000/attempt-01", "source_commit": "indexed-source"}
        receipt = {"authorization_commit": "receipt-authorization",
                   "execution_track": "nonconfirmatory_remote_gpu_supplement"}
        attempt = reference_attempt("external", reference, Path("local"), "fixed")
        self.assertEqual(attempt, ROOT / reference["p4_attempt"])
        self.assertEqual(reference_attempt("p4", reference, Path("local"), "fixed"), attempt)
        self.assertEqual(reference_attempt("local", reference, Path("local"), "fixed"),
                         Path("local") / "fixed__attempt-01")
        provenance = reference_provenance("external", reference, receipt, "rtx8000")
        self.assertEqual(provenance["reference_track"], "rtx8000")
        self.assertEqual(provenance["reference_selection_track"], "external")
        self.assertEqual(provenance["source_execution_track"], receipt["execution_track"])
        self.assertEqual(provenance["reference_source_commit"], "indexed-source")
        self.assertEqual(reference_provenance("p4", reference, receipt)["reference_track"], "p4")
        local = reference_provenance("local", reference, receipt)
        self.assertEqual(local["reference_track"], "local")
        self.assertEqual(local["reference_source_commit"], "receipt-authorization")

    def test_external_default_label_reflects_receipt_execution_track(self):
        reference = {"source_commit": "indexed-source"}
        provenance = reference_provenance("external", reference, {"execution_track": "actual-rtx8000-track"})
        self.assertEqual(provenance["reference_track"], "actual-rtx8000-track")
        self.assertEqual(provenance["source_label"], "actual-rtx8000-track")
        fallback = reference_provenance("external", reference, {})
        self.assertEqual(fallback["reference_track"], "external")
        self.assertIsNone(fallback["source_execution_track"])

    def test_external_indexed_attempt_is_protected_outside_legacy_p4_root(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            reference = {"p4_attempt": str(root / "received" / "rtx8000" / "attempt-01")}
            attempt = reference_attempt("external", reference, root / "local", "fixed")
            attempt.mkdir(parents=True)
            artifact = attempt / "receipt.json"
            artifact.write_text("original", encoding="utf-8")
            for output in (attempt, attempt / "audit", attempt.parent):
                with self.assertRaises(RuntimeError):
                    new_output_root(output, [attempt])
            new_output_root(root / "fresh-audit", [attempt])
            self.assertEqual(artifact.read_text(encoding="utf-8"), "original")

    def test_ast_parser_preserves_frozen_defaults_without_executing_main(self):
        source = (ROOT / "run.py").read_text(encoding="utf-8")
        source += "\nraise AssertionError('source execution is forbidden')\n"
        args = receipt_args("/other/python run.py --task_name long_term_forecast --is_training 1 "
                            "--model_id fixed --model revin-TimesNet --data custom --deterministic", source)
        self.assertEqual((args.top_k, args.num_kernels, args.activation), (5, 6, "gelu"))
        self.assertEqual(args.batch_size, 32)
        self.assertTrue(args.deterministic)
        self.assertFalse(args.use_amp)
        self.assertFalse(args.inverse)

    def test_windows_receipt_command_and_multivalue_parser(self):
        source = (ROOT / "run.py").read_text(encoding="utf-8")
        args = receipt_args(r"C:\env\python.exe run.py --task_name long_term_forecast --is_training 1 "
                            "--model_id fixed --model DeReFusion --data custom "
                            "--root_path C:/repo/dataset/ --p_hidden_dims 64 32 --no_use_gpu", source)
        self.assertEqual(args.p_hidden_dims, [64, 32])
        self.assertEqual(args.root_path, "C:/repo/dataset/")
        self.assertFalse(args.use_gpu)

    def test_parser_rejects_nonliteral_default_without_execution(self):
        source = "if __name__ == '__main__':\n    parser.add_argument('--x', default=perform_training())\n"
        with self.assertRaises(ValueError):
            parser_from_source(source)

    def test_array_hash_binds_dtype_shape_and_order(self):
        array = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.assertEqual(array_hash(array), array_hash(array.copy()))
        self.assertNotEqual(array_hash(array), array_hash(array.reshape(3, 2)))
        self.assertNotEqual(array_hash(array), array_hash(array.astype(np.float64)))
        self.assertNotEqual(array_hash(array), array_hash(array[:, ::-1]))

    def test_bound_inputs_fail_before_and_after_mutation(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "input.bin"
            path.write_bytes(b"frozen")
            bound = BoundInputs()
            digest = bound.bind(path)
            bound.verify()
            path.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                bound.verify()
            with self.assertRaises(RuntimeError):
                bound.bind(path)
            with self.assertRaises(RuntimeError):
                BoundInputs().bind(path, digest)

    def test_output_refuses_reuse_and_intersection_without_touching_inputs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            protected = root / "reference"
            protected.mkdir()
            artifact = protected / "receipt.json"
            artifact.write_text("original", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                new_output_root(protected / "audit", [protected])
            with self.assertRaises(RuntimeError):
                new_output_root(root, [protected])
            output = new_output_root(root / "fresh", [protected])
            write_json(output / "report.json", {"status": "diagnostic"})
            with self.assertRaises(FileExistsError):
                new_output_root(output, [protected])
            with self.assertRaises(FileExistsError):
                write_json(output / "report.json", {"status": "overwrite"})
            self.assertEqual(artifact.read_text(encoding="utf-8"), "original")

    def test_differences_are_descriptive_and_reject_shape_drift(self):
        reference = np.zeros((2, 1, 1), dtype=np.float32)
        pred = np.ones_like(reference)
        result = differences(pred, reference, reference)
        self.assertEqual(result["max_absolute_prediction_difference"], 1)
        self.assertEqual(result["signed_mse_difference_float64"], 1)
        self.assertEqual(result["numerical_equivalence_threshold"], "TBD")
        self.assertFalse(result["numerical_equivalence_claim"])
        with self.assertRaises(RuntimeError):
            differences(pred.reshape(1, 2, 1), reference, reference)


if __name__ == "__main__":
    unittest.main()
