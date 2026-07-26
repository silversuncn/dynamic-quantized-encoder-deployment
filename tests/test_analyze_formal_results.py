import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from analyze_formal_results import FIGURE_OUTPUT_POLICY, FONT_POLICY, phase4_artifact_stem, phase4_artifact_path, write_figures


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

    def test_publication_figure_policy_uses_readable_fonts_and_high_resolution_png(self):
        self.assertGreaterEqual(FONT_POLICY["tick_label_pt"], 7)
        self.assertGreaterEqual(FONT_POLICY["axis_label_pt"], 8)
        self.assertEqual(FIGURE_OUTPUT_POLICY["png_dpi"], 600)
        self.assertEqual(FIGURE_OUTPUT_POLICY["pdf_fonttype"], 42)

    def test_write_figures_emits_pdf_and_high_resolution_png_outputs(self):
        group_rows = [
            {
                "dataset": dataset,
                "backbone": backbone,
                "label_budget_per_class": budget,
                "method": "dynamic_int8_cpu",
                "mean_macro_f1_delta_vs_fp32_cpu": value,
            }
            for dataset, value in [("sst2", -0.02), ("mrpc", -0.11)]
            for backbone in ["albert-base-v2", "bert-base-uncased"]
            for budget in [64, 128]
        ]
        method_rows = [
            {
                "method": "dynamic_int8_cpu",
                "mean_best_deployment_improvement": 0.55,
                "mean_accuracy_delta": -0.07,
            },
            {
                "method": "fp32_gpu",
                "mean_best_deployment_improvement": 32.7,
                "mean_accuracy_delta": 0.0,
            },
            {
                "method": "bf16_gpu",
                "mean_best_deployment_improvement": 29.2,
                "mean_accuracy_delta": 0.0,
            },
        ]
        with TemporaryDirectory() as tmp:
            figures = write_figures(group_rows, method_rows, Path(tmp), figure_prefix="formal")

            self.assertEqual(figures["status"], "PASS")
            generated = figures["generated"]
            self.assertIn("formal_dynamic_int8_macro_f1_delta", generated)
            self.assertIn("formal_method_tradeoff_summary", generated)
            for outputs in generated.values():
                pdf_path = Path(outputs["pdf"]["path"])
                png_path = Path(outputs["png"]["path"])
                self.assertTrue(pdf_path.exists())
                self.assertTrue(png_path.exists())
                self.assertGreater(outputs["pdf"]["size"], 0)
                self.assertGreater(outputs["png"]["size"], 0)
                with Image.open(png_path) as image:
                    dpi = image.info.get("dpi")
                    self.assertIsNotNone(dpi)
                    self.assertGreaterEqual(round(dpi[0]), 600)


if __name__ == "__main__":
    unittest.main()
