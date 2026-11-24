"""Synthetic fixture sizes obey the row cap for Python and NumPy integers."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import make_binary_fixture, make_materials_fixture
from groupbench.data import MAX_ROWS


class FixtureSizeBoundsTests(unittest.TestCase):
    def test_reject_oversized_fixture_before_rng(self):
        for make in (make_materials_fixture, make_binary_fixture):
            for groups, rows in (
                (10001, 2),
                (np.int64(10001), np.int64(2)),
                (np.int32(10001), np.uint64(2)),
            ):
                with self.subTest(make=make.__name__, groups=groups, rows=rows):
                    with patch("numpy.random.default_rng", side_effect=AssertionError("fixture generation started")) as rng:
                        with self.assertRaisesRegex(ValueError, "must not exceed 20,000 rows"):
                            make(groups, rows)
                        rng.assert_not_called()

    def test_accept_row_limit_and_preserve_seeded_output(self):
        for make in (make_materials_fixture, make_binary_fixture):
            with self.subTest(make=make.__name__):
                native = make(200, 100, random_state=12)
                numpy_sizes = make(np.int32(200), np.uint64(100), random_state=np.int64(12))
                self.assertEqual(len(native), MAX_ROWS)
                pd.testing.assert_frame_equal(native, numpy_sizes)
