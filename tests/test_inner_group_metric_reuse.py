"""Check exact candidate selection scores against a full OOF recomputation."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groupbench import fit_predict
from groupbench.data import validate_groups
from groupbench.domain_selection import domain_splits
from groupbench.metrics import grouped_metrics
from groupbench.splitting import GroupKFold


class InnerGroupMetricReuseTests(unittest.TestCase):
    def test_public_fit_predict_keeps_full_oof_and_fold_scores(self):
        rng = np.random.default_rng(1469)
        groups = np.repeat([f"lot-{i:02d}" for i in range(12)], np.arange(2, 14))
        rng.shuffle(groups)
        x = rng.normal(size=len(groups))
        frame = pd.DataFrame({"x": x, "y": .8 * x + rng.normal(size=len(groups)),
                              "g": groups, "site": [f"d{int(g[-2:]) % 3}" for g in groups]})
        for mode in ("group_kfold", "leave_one_domain_out"):
            for objective in ("mae", "group_mae"):
                with self.subTest(mode=mode, objective=objective):
                    y = frame["y"].to_numpy()
                    g = validate_groups(groups, len(groups))
                    if mode == "group_kfold":
                        plan = list(GroupKFold(3, 7).split(groups=g))
                        options = {"inner_splits": 3}
                    else:
                        plan = domain_splits(frame["site"].to_numpy())
                        options = {"inner_splits": None, "domain": "site"}
                    calls = []

                    def observe(observed, predictions, ids):
                        result = grouped_metrics(observed, predictions, ids)
                        calls.append((observed.copy(), predictions.copy(), ids.copy(), result[0][objective]))
                        return result

                    with patch("groupbench.model.grouped_metrics", side_effect=observe):
                        result = fit_predict(frame, frame[["x"]].iloc[:4], target="y", group="g",
                                             features=["x"], random_state=7, inner_strategy=mode,
                                             baselines=("dummy", "ridge"), objective=objective, **options)
                    records = result.estimator.selection_results_
                    # Exactly one grouped call per validation fold/candidate:
                    # no second per-group recomputation over the pooled vector.
                    self.assertEqual(len(calls), len(records) * len(plan))
                    for i, record in enumerate(records):
                        oof, fold_scores = np.empty_like(y), []
                        start = i * len(plan)
                        for (_, test), (observed, prediction, ids, score) in zip(
                                plan, calls[start:start + len(plan)]):
                            np.testing.assert_array_equal(observed, y[test])
                            np.testing.assert_array_equal(ids, g[test])
                            oof[test] = prediction
                            fold_scores.append(score)
                        reference, _ = grouped_metrics(y, oof, g)
                        self.assertEqual(record["inner_fold_scores"], fold_scores)
                        self.assertEqual(record["selection_score"], reference[objective])
