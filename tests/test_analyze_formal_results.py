import unittest
from pathlib import Path

from analyze_formal_results import phase4_artifact_stem, phase4_artifact_path


class AnalyzeFormalResultsTests(unittest.TestCase):
    def test_phase4_artifact_path_uses_requested_date_tag(self):
        self.assertEqual(
            phase4_artifact_path(Path("/tmp/out"), "formal_phase4_manifest", "20260725", "json"),
            Path("/tmp/out/formal_phase4_manifest_20260725.json"),
        )

    def test_phase4_artifact_stem_uses_requested_prefix(self):
        self.assertEqual(
            phase4_artifact_stem("strengthened_phase4", "manifest"),
            "strengthened_phase4_manifest",
        )


if __name__ == "__main__":
    unittest.main()
