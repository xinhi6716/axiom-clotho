"""Duration units cannot act as dimensionless numeric model parameters."""

import unittest

import numpy as np
import pandas as pd

from groupbench import fit_predict, plan_prediction
from groupbench.baselines import GroupBalancedRidgeRegressor
from groupbench.kernel import NystroemRidgeRegressor


class DimensionlessModelParameterTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(16, dtype=float).reshape(8, 2)
        self.y = np.arange(8, dtype=float) ** 2
        self.groups = np.repeat(["g0", "g1", "g2", "g3"], 2)

    def test_direct_optional_fit_rejects_temporal_alpha_rank_and_seed(self):
        for duration in (np.timedelta64(3, "ns"), np.timedelta64(3, "us"), np.timedelta64("NaT")):
            for cls, field, fit_kw in (
                    (NystroemRidgeRegressor, "alpha", {}),
                    (NystroemRidgeRegressor, "rank_cap", {}),
                    (NystroemRidgeRegressor, "random_state", {}),
                    (GroupBalancedRidgeRegressor, "alpha", {"groups": self.groups})):
                with self.subTest(estimator=cls.__name__, field=field, duration=str(duration)):
                    estimator = cls(**{field: duration})
                    with self.assertRaisesRegex(ValueError, field):
                        estimator.fit(self.x, self.y, **fit_kw)
                    self.assertFalse(any(name.endswith("_") for name in vars(estimator)))

    def test_public_optional_prediction_and_plan_reject_temporal_seed(self):
        train = pd.DataFrame(self.x, columns=["voltage", "current"]).assign(y=self.y, g=self.groups)
        query = train[["voltage", "current"]].iloc[1:3]
        options = dict(target="y", group="g", baselines=("nystroem_ridge",), inner_splits=2)
        for entry in (fit_predict, plan_prediction):
            for unit in ("ns", "us"):
                with self.subTest(entry=entry.__name__, unit=unit), self.assertRaisesRegex(ValueError, "random_state"):
                    entry(train, query, random_state=np.timedelta64(3, unit), **options)

    def test_ordinary_numpy_numeric_parameters_preserve_predictions_and_details(self):
        for cls, params, fit_kw in (
                (NystroemRidgeRegressor, {"alpha": 3.0, "rank_cap": 3, "random_state": 3}, {}),
                (GroupBalancedRidgeRegressor, {"alpha": 3.0}, {"groups": self.groups})):
            with self.subTest(estimator=cls.__name__):
                numpy_params = {name: (np.float64(value) if name == "alpha" else np.int64(value))
                                for name, value in params.items()}
                numeric = cls(**params).fit(self.x, self.y, **fit_kw)
                typed = cls(**numpy_params).fit(self.x, self.y, **fit_kw)
                self.assertEqual(numeric.fit_details_, typed.fit_details_)
                np.testing.assert_array_equal(numeric.predict(self.x), typed.predict(self.x))
