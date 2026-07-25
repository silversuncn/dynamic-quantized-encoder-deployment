import unittest

from aggregate_quant_results import matrix_status


class AggregateQuantResultsTests(unittest.TestCase):
    def test_formal_tradeoff_status_is_matrix_specific(self):
        self.assertEqual(matrix_status("formal", "PASS_TRADEOFF_SIGNAL"), "PASS_FORMAL_TRADEOFF_READY")
        self.assertEqual(matrix_status("formal", "PASS_MICRO"), "PASS_FORMAL_MATRIX_READY")

    def test_strengthened_tradeoff_status_is_matrix_specific(self):
        self.assertEqual(matrix_status("strengthened", "PASS_TRADEOFF_SIGNAL"), "PASS_STRENGTHENED_TRADEOFF_READY")
        self.assertEqual(matrix_status("strengthened", "WEAK_TRADEOFF_STOP"), "WEAK_STRENGTHENED_TRADEOFF_STOP")

    def test_existing_micro_and_bounded_status_mapping_is_preserved(self):
        self.assertEqual(matrix_status("micro", "PASS_TRADEOFF_SIGNAL"), "PASS_TRADEOFF_SIGNAL")
        self.assertEqual(matrix_status("bounded", "PASS_TRADEOFF_SIGNAL"), "PASS_BOUNDED_TRADEOFF_READY")


if __name__ == "__main__":
    unittest.main()
