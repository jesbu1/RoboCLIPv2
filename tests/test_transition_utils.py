import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "offline_rl_algorithms"
    / "transition_utils.py"
)
SPEC = importlib.util.spec_from_file_location("transition_utils", MODULE_PATH)
TRANSITION_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRANSITION_UTILS)
build_one_step_next_observations = (
    TRANSITION_UTILS.build_one_step_next_observations
)


class BuildOneStepNextObservationsTest(unittest.TestCase):
    def test_shifts_only_nonterminal_rows(self):
        observations = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
        dones = np.array([0.0, 0.0, 1.0, 0.0, 1.0])

        next_observations = build_one_step_next_observations(observations, dones)

        np.testing.assert_array_equal(
            next_observations,
            np.array([[1.0], [2.0], [2.0], [11.0], [11.0]]),
        )
        np.testing.assert_array_equal(
            observations,
            np.array([[0.0], [1.0], [2.0], [10.0], [11.0]]),
        )

    def test_rejects_final_nonterminal_row(self):
        with self.assertRaisesRegex(ValueError, "final non-terminal"):
            build_one_step_next_observations(
                np.array([[0.0], [1.0]]),
                np.array([0.0, 0.0]),
            )

    def test_rejects_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same number of transitions"):
            build_one_step_next_observations(
                np.array([[0.0], [1.0]]),
                np.array([1.0]),
            )


if __name__ == "__main__":
    unittest.main()
