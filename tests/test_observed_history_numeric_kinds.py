"""Timedeltas require explicit unit conversion, even inside object columns."""
import unittest

import numpy as np
import pandas as pd

from groupbench.observed_history import evaluate_observed_history


class ObservedHistoryNumericKindTests(unittest.TestCase):
    def test_numpy_timedeltas_do_not_become_numeric_readings_or_positions(self):
        for unit in ("ns", "s", "D"):
            for mode in ("float64", "integer_ticks"):
                for field in ("timestamp", "value", "prediction", "target_index"):
                    observations = pd.DataFrame({"trace_id": ["trace"] * 3,
                                                 "timestamp": [0, 1, 2], "value": [1, 2, 3]})
                    origins = pd.DataFrame({"trace_id": ["trace"], "target_index": [2]})
                    predictions = origins.assign(prediction=3)
                    temporal = lambda seq: pd.Series(
                        [np.timedelta64(i, unit) for i in seq], dtype=object)
                    if field in ("timestamp", "value"):
                        observations[field] = temporal(observations[field])
                    elif field == "prediction":
                        predictions[field] = temporal(predictions[field])
                    else:
                        origins[field] = predictions[field] = temporal(origins[field])
                    with self.subTest(unit=unit, mode=mode, field=field):
                        with self.assertRaises(ValueError):
                            evaluate_observed_history(observations, origins=origins,
                                                      saved_predictions=predictions, timestamp_mode=mode)

    def test_explicit_conversion_preserves_numeric_ticks(self):
        elapsed = pd.Series(pd.to_timedelta([0, 1, 3], unit="s"))
        ticks = (elapsed / pd.Timedelta(seconds=1)).astype("int64")
        observations = pd.DataFrame({"trace_id": ["trace"] * 3,
                                     "timestamp": ticks, "value": [1, 2, 3]})
        origins = pd.DataFrame({"trace_id": ["trace"], "target_index": [2]})
        result = evaluate_observed_history(observations, origins=origins,
                                           saved_predictions=origins.assign(prediction=3),
                                           timestamp_mode="integer_ticks")
        self.assertEqual(result.predictions["timestamp"].to_list(), [3])
        self.assertEqual(result.predictions["timestamp"].dtype, np.dtype("int64"))
