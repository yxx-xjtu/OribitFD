from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_benchmarks import PUBLIC_GROUPS, role_intervals, validate_task_selection


class ProtocolTests(unittest.TestCase):
    def test_role_intervals_disjoint_source_and_target(self) -> None:
        spans = role_intervals(250_000, g=1024)
        src = [spans[r] for r in ("train", "val", "cal", "audit")]
        for a, b in zip(src, src[1:]):
            self.assertGreaterEqual(b[0] - a[1], 2048)
        self.assertGreaterEqual(spans["eval"][0] - spans["dev"][1], 2048)

    def test_overlap_detection_rejects_leakage(self) -> None:
        bad = {
            "train": [{"record_id": "r0", "start": 0, "end": 1024, "content_sha256": "h1"}],
            "val": [{"record_id": "r0", "start": 512, "end": 1536, "content_sha256": "h2"}],
        }
        with self.assertRaises(ValueError):
            validate_task_selection(bad)

    def test_public_groups_have_disjoint_conditions(self) -> None:
        self.assertEqual(len(PUBLIC_GROUPS), 7)
        for gid, _ds, src, tgt, _desc in PUBLIC_GROUPS:
            self.assertEqual(set(src) & set(tgt), set(), msg=gid)


if __name__ == "__main__":
    unittest.main()
