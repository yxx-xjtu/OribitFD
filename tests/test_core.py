from __future__ import annotations

import inspect
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbitfd import (
    OnlineConfig,
    OrbitBank,
    OrbitOnline,
    SourceClassifier,
    benchmark_metrics,
    evaluate_stream,
    fit_bank,
    open_set_metrics,
)
from scripts.demo_synthetic import make_data


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        data, cls.streams, cls.cal_context = make_data(blocks=48, block_size=32)
        cls.data = data
        cls.bank = fit_bank(*data, calibration_context=cls.cal_context)

    def test_orthonormal_basis(self) -> None:
        b = self.bank
        torch.testing.assert_close(
            b.basis.T @ b.basis, torch.eye(b.rank, dtype=torch.float64), atol=1e-9, rtol=1e-9
        )

    def test_source_bank_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bank.npz"
            self.bank.save(path)
            b = OrbitBank.load(path)
            torch.testing.assert_close(b.basis, self.bank.basis)
            torch.testing.assert_close(
                b.predict(self.data[0][:32])["score"],
                self.bank.predict(self.data[0][:32])["score"],
            )

    def test_no_target_labels_in_api(self) -> None:
        self.assertEqual(
            list(inspect.signature(OrbitOnline.step).parameters),
            ["self", "features", "adapt"],
        )

    def test_first_prediction_precedes_update(self) -> None:
        x = self.streams["single_class_known"][0][32 * 8 : 32 * 9]
        engine = OrbitOnline(self.bank)
        expected = self.bank.predict(x)
        out = engine.step(x)
        for key in ("score", "forced_prediction", "label"):
            torch.testing.assert_close(out[key], expected[key])
        torch.testing.assert_close(out["state_used"], torch.zeros(self.bank.rank, dtype=torch.float64))

    def test_source_bank_is_immutable(self) -> None:
        before = {k: v.clone() for k, v in self.bank.__dict__.items() if isinstance(v, torch.Tensor)}
        engine = OrbitOnline(self.bank)
        x = self.streams["single_class_known"][0]
        for i in range(0, 512, 32):
            engine.step(x[i : i + 32])
        for k, v in before.items():
            torch.testing.assert_close(v, getattr(self.bank, k))

    def test_normal_score_invariant_to_orbit_state(self) -> None:
        z = self.bank.normalize(self.data[0][:32])
        a = torch.tensor([1.0, -2.0], dtype=torch.float64)
        b = torch.zeros(2, dtype=torch.float64)
        torch.testing.assert_close(
            self.bank.components(z, a)["normal"], self.bank.components(z, b)["normal"]
        )
        self.assertFalse(
            torch.allclose(self.bank.components(z, a)["parallel"], self.bank.components(z, b)["parallel"])
        )

    def test_unknown_only_rejects_updates_in_off_orbit_case(self) -> None:
        engine = OrbitOnline(self.bank)
        x = self.streams["unknown_only"][0]
        for i in range(0, len(x), 32):
            engine.step(x[i : i + 32])
        self.assertEqual(engine.accepted_updates, 0)

    def test_single_class_context_can_update(self) -> None:
        engine = OrbitOnline(self.bank)
        x = self.streams["single_class_known"][0]
        for i in range(0, len(x), 32):
            engine.step(x[i : i + 32])
        self.assertGreater(engine.accepted_updates, 5)

    def test_large_tangent_jump_is_not_slowly_absorbed(self) -> None:
        b = self.bank
        state = torch.tensor([1.5, 0.0], dtype=torch.float64)
        normalized = b.prototypes[0] + b.basis @ (b.coordinate_scale * state)
        raw = (normalized * b.scale + b.mean)[None].repeat(32, 1)
        engine = OrbitOnline(b)
        for _ in range(4):
            out = engine.step(raw)
        self.assertFalse(out["updated"])
        self.assertEqual(out["reason"], "innovation_exceeds_declared_drift_budget")
        torch.testing.assert_close(engine.state, torch.zeros(2, dtype=torch.float64))

    def test_small_block_holds_state(self) -> None:
        engine = OrbitOnline(self.bank)
        out = engine.step(self.data[0][:2])
        self.assertFalse(out["updated"])
        torch.testing.assert_close(engine.state, torch.zeros(2, dtype=torch.float64))

    def test_bad_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            self.bank.predict(np.full((5, 16), np.nan))
        with self.assertRaises(ValueError):
            fit_bank(*self.data, rank=16)
        with self.assertRaises(ValueError):
            OnlineConfig(max_step=-1)

    def test_encoder_parameter_counts_match_paper(self) -> None:
        m1 = SourceClassifier(channels=1, feature_dim=64, num_classes=3)
        m3 = SourceClassifier(channels=3, feature_dim=64, num_classes=3)
        self.assertEqual(sum(p.numel() for p in m1.encoder.parameters()), 68800)
        self.assertEqual(sum(p.numel() for p in m3.encoder.parameters()), 69760)
        self.assertEqual(sum(p.numel() for p in m1.parameters()), 68995)
        self.assertEqual(sum(p.numel() for p in m3.parameters()), 69955)

    def test_prefix_predictions_do_not_depend_on_future(self) -> None:
        x, y, bids = self.streams["off_orbit_unknown"]
        full, _ = evaluate_stream(self.bank, x, bids)
        prefix, _ = evaluate_stream(self.bank, x[:512], bids[:512])
        np.testing.assert_array_equal(full["score"][:512], prefix["score"])
        np.testing.assert_array_equal(full["label"][:512], prefix["label"])
        bm = benchmark_metrics(y, full["forced_prediction"], full["score"])
        om = open_set_metrics(y, full["forced_prediction"], full["label"], full["score"])
        self.assertAlmostEqual(bm["auroc"], om["auroc_unknown_positive"], places=12)


if __name__ == "__main__":
    unittest.main()
