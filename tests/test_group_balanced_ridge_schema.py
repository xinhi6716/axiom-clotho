"""Hand-authored saved declarations, no fits or scientific performance evidence."""
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from groupbench.evaluation_schema import _check, validate_evaluation_bundle
from groupbench.prediction_schema import (
    _metadata, validate_prediction_bundle_schema, _DOMAIN_GRID_V1,
)
from groupbench.verification import verify_bundle

FAMILY = "group_balanced_ridge"
SIZES = {"a": 1, "b": 2, "c": 1, "d": 2, "e": 1, "f": 2}
GRID = {
    "dummy": [("dummy_mean", {"strategy": "mean"})],
    "ridge": [("ridge_alpha_0.1", {"alpha": 0.1}), ("ridge_alpha_1", {"alpha": 1.0}), ("ridge_alpha_10", {"alpha": 10.0})],
    "extra_trees": [("extra_trees_depth_6", {"max_depth": 6, "n_estimators": 64, "min_samples_leaf": 2}),
                    ("extra_trees_depth_none", {"max_depth": None, "n_estimators": 64, "min_samples_leaf": 2})],
    "group_median": [("group_median_lower", {})],
    "nystroem_ridge": [("nystroem_ridge_alpha_0.1", {"alpha": 0.1, "rank_cap": 64}),
                       ("nystroem_ridge_alpha_1", {"alpha": 1.0, "rank_cap": 64}),
                       ("nystroem_ridge_alpha_10", {"alpha": 10.0, "rank_cap": 64})],
    FAMILY: [("group_balanced_ridge_alpha_0.1", {"alpha": 0.1}),
             ("group_balanced_ridge_alpha_1", {"alpha": 1.0}),
             ("group_balanced_ridge_alpha_10", {"alpha": 10.0})],
}
OPTIONAL = ("group_median", "nystroem_ridge", FAMILY)


def details(family, sizes, alpha, seed):
    n, g = sum(sizes.values()), len(sizes)
    base = {"training_rows": n, "transformed_features": 1}
    if family == FAMILY:
        return base | {"implementation": "group-balanced-ridge-v1", "training_groups": g,
            "alpha": alpha, "solver": "lsqr", "tol": 1e-4, "max_iter": None,
            "fit_intercept": True, "copy_X": True, "positive": False, "threads": 1,
            "fitting_loss": "weighted_squared_error", "weight_rule": "equal_training_group_then_equal_row",
            "normalization": "sum_training_rows", "preprocessing_weighting": "unweighted_training_rows",
            "min_training_group_size": min(sizes.values()), "max_training_group_size": max(sizes.values()),
            "min_row_weight": n / (g * max(sizes.values())),
            "max_row_weight": n / (g * min(sizes.values())), "sum_row_weights": float(n)}
    if family == "group_median":
        return base | {"implementation": "group-balanced-lower-median-v1", "training_groups": g,
            "constant": 0.0, "weight_rule": "equal_training_group_then_equal_row", "tie_break": "lower_observed_target"}
    if family == "nystroem_ridge":
        return base | {"implementation": "bounded-nystroem-rbf-ridge-v1", "rank_cap": 64,
            "effective_n_components": min(64, n), "gamma_policy": "inverse_transformed_width", "gamma": 1.0,
            "random_state": seed, "basis_sampling": "uniform_training_rows_without_replacement",
            "basis_indices_sha256": "0" * 64, "alpha": alpha, "solver": "svd", "threads": 1,
            "training_map_bytes": n * min(n, 64) * 8, "map_byte_limit": 16 * 1024 * 1024}
    raise AssertionError("not an optional family")


def config(baselines):
    return {"target": "y", "group": "g", "features": ["x"], "inner_splits": 2,
            "random_state": 0, "baselines": list(baselines), "selection_objective": "group_mae",
            "categorical_encoding": "legacy-v1", "threads": 1,
            "candidates": [{"name": name, "family": family, "params": deepcopy(params)}
                           for family in baselines for name, params in GRID[family]]}


def audit(sizes, test, fold):
    train = sorted(set(sizes) - set(test))
    return {"fold": fold, "train_rows": sum(sizes[g] for g in train),
            "test_rows": sum(sizes[g] for g in test), "train_group_count": len(train),
            "test_group_count": len(test), "overlap_count": 0,
            "train_groups": train, "test_groups": sorted(test)}


def selection(sizes, conf, seed):
    group_ids = sorted(sizes)
    half = len(group_ids) // 2
    folds = [audit(sizes, group_ids[:half], 0), audit(sizes, group_ids[half:], 1)]
    records, final, winners = [], {}, {}
    for index, candidate in enumerate(conf["candidates"]):
        family, params = candidate["family"], candidate["params"]
        record = {"candidate": candidate["name"], "family": family, "params": deepcopy(params),
                  "objective": conf["selection_objective"], "selection_score": float(index + 1),
                  "inner_fold_scores": [float(index + 1), float(index + 1)]}
        if family in OPTIONAL:
            record["inner_fit_details"] = [{"inner_fold": i, "train_rows": f["train_rows"],
                "test_rows": f["test_rows"], "details": details(family,
                    {g: sizes[g] for g in f["train_groups"]}, params.get("alpha"), seed)} for i, f in enumerate(folds)]
        records.append(record)
        if family not in winners:
            winners[family] = candidate["name"]
            if family in OPTIONAL:
                final[family] = details(family, sizes, params.get("alpha"), seed)
    return {"inner_folds": folds, "inner_selection": records, "family_fit_details": final,
            "family_candidates": winners, "selected_candidate": records[0]["candidate"],
            "selected_family": records[0]["family"]}


def prediction_fixture(baselines=(FAMILY,), sizes=None):
    sizes = SIZES if sizes is None else sizes
    n_rows = sum(sizes.values())
    conf = config(baselines)
    return {"mode": "fit_predict", "schema_version": "1.5",
        "protocol": "group-fit-predict-extended-families-v2",
        "capabilities": [f + "-v1" for f in OPTIONAL if f in baselines],
        "config": conf, "n_training_rows": n_rows, "n_training_groups": len(sizes), "n_predictions": 2,
        "feature_schema": [{"name": "x", "dtype": "float64"}],
        "input_support": {"schema_version": "1.0", "reference": "full_training_features_before_imputation",
            "action": "diagnostic_only_no_input_changes", "n_training_rows": n_rows, "n_prediction_rows": 2,
            "rows_with_any_flag": 0, "row_flag_counts": {"missing_value": 0,
                "numeric_outside_training_range": 0, "unseen_category": 0,
                "non_missing_value_in_all_missing_training_feature": 0},
            "features": [{"name": "x", "kind": "numeric", "training_missing_count": 0,
                "training_support": "observed_non_missing_values", "prediction_missing_count": 0,
                "prediction_non_missing_without_training_values_count": 0,
                "below_training_min_count": 0, "above_training_max_count": 0}]},
        **selection(sizes, conf, 0)}


def evaluation_fixture(baselines=(FAMILY,)):
    conf = config(baselines) | {"outer_splits": 3}
    names = list(baselines) + ["selected"]
    folds = [audit(SIZES, pair, fold) for fold, pair in enumerate((("a", "b"), ("c", "d"), ("e", "f")))]
    for fold, record in enumerate(folds):
        record.update(inner_seed=fold + 1, **selection({g: SIZES[g] for g in record["train_groups"]}, conf, fold + 1))
    manifest = {"schema_version": "1.4", "protocol": "nested-group-kfold-extended-families-v2",
                "capabilities": [f + "-v1" for f in OPTIONAL if f in baselines],
                "n_rows": 9, "n_groups": 6, "config": conf, "outer_folds": folds}
    def metrics(n):
        return {"n_rows": n, "mae": 0.0, "rmse": 0.0, "r2": None, "bias": 0.0}
    # Zero observations/predictions keep the metric fixture independently simple.
    def grouped(n, g):
        return metrics(n) | {"n_groups": g, "group_mae": 0.0, "group_rmse": 0.0, "worst_group_mae": 0.0}
    model = {"overall": grouped(9, 6),
             "per_group": [{"group": g, **metrics(n)} for g, n in SIZES.items()],
             "per_fold": [{"fold": i, **grouped(3, 2)} for i in range(3)]}
    summary = {"schema_version": "1.4", "protocol": manifest["protocol"], "capabilities": manifest["capabilities"][:],
               "selection_objective": "group_mae", "primary_entry": "selected",
               "models": {name: deepcopy(model) for name in names}}
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["row_position", "group", "outer_fold", "observed"] + ["prediction_" + name for name in names])
    position = 0
    for group_index, (g, count) in enumerate(SIZES.items()):
        for _ in range(count):
            writer.writerow([position, g, group_index // 2, 0.0] + [0.0] * len(names))
            position += 1
    return manifest, summary, stream.getvalue().encode()


def write_bundle(directory, manifest, summary=None, raw=b"row_position,prediction\n0,0.0\n1,0.0\n"):
    payloads = {"manifest.json": json.dumps(manifest).encode(), "predictions.csv": raw}
    if summary is not None:
        payloads.update({"summary.json": json.dumps(summary).encode(), "report.html": b"<p>Handwritten fixture</p>"})
    for name, content in payloads.items():
        (directory / name).write_bytes(content)
    (directory / "checksums.json").write_text(json.dumps({"algorithm": "sha256", "files": {
        name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}}))


class GroupBalancedRidgeSchemaTests(unittest.TestCase):
    def test_emitted_bundles_match_strict_contracts(self):
        # Tiny fixed integration fixture, not random draws or a model comparison.
        import pandas as pd
        from groupbench import benchmark, fit_predict
        frame = pd.DataFrame({"g": ["a", "b", "b", "c", "d", "d", "e", "f", "f"],
                              "x": [0., 1., 2., 3., 4., 5., 6., 7., 8.],
                              "y": [0., 2., 1., 4., 3., 5., 7., 6., 8.]})
        for families, objective, encoding in [((FAMILY,), "group_mae", "legacy-v1"),
                (("dummy", FAMILY, "ridge", "extra_trees", "group_median", "nystroem_ridge"), "mae", "distinct-v2")]:
            options = dict(target="y", group="g", inner_splits=2, baselines=families,
                           objective=objective, categorical_encoding=encoding)
            result = fit_predict(frame, frame[["x"]].iloc[:2], **options)
            self.assertEqual(_metadata(result.manifest)[0], "1.5")
            result = benchmark(frame, outer_splits=3, **options)
            self.assertEqual(_check(result.manifest, result.summary, result.predictions.to_csv(index=False).encode())[0], "1.4")

    def test_new_versions_combinations_objectives_and_encodings(self):
        for families in [(FAMILY,), ("dummy", FAMILY), (FAMILY, "ridge", "extra_trees", "group_median", "nystroem_ridge")]:
            for encoding in ("legacy-v1", "distinct-v2"):
                pred = prediction_fixture(families)
                pred["config"]["categorical_encoding"] = encoding
                self.assertEqual(_metadata(pred)[0], "1.5")
                manifest, summary, raw = evaluation_fixture(families)
                manifest["config"]["categorical_encoding"] = encoding
                self.assertEqual(_check(manifest, summary, raw)[0], "1.4")
        pred = prediction_fixture()
        pred["config"]["selection_objective"] = "mae"
        for record in pred["inner_selection"]:
            record["objective"] = "mae"
        self.assertEqual(_metadata(pred)[0], "1.5")

    def test_frozen_grid_no_runtime_import_or_domain_widening(self):
        self.assertNotIn(FAMILY, _DOMAIN_GRID_V1)
        with patch("groupbench.model.candidates_for", side_effect=AssertionError("runtime grid forbidden")):
            self.assertEqual(_metadata(prediction_fixture())[0], "1.5")
            self.assertEqual(_check(*evaluation_fixture())[0], "1.4")

    def test_candidate_and_selection_corruption_rejected_both(self):
        changes = [
            lambda a: a["inner_selection"].pop(),
            lambda a: a["inner_selection"].append(deepcopy(a["inner_selection"][0])),
            lambda a: a["inner_selection"].reverse(),
            lambda a: a["inner_selection"][0].update(candidate="unknown"),
            lambda a: a["inner_selection"][0].update(family="unknown"),
            lambda a: a["inner_selection"][0].update(params={"alpha": True}),
            lambda a: a["inner_selection"][0].update(objective="mse"),
            lambda a: a["inner_selection"][0].update(selection_score=True),
            lambda a: a["inner_selection"][0].update(selection_score=-1),
            lambda a: a["inner_selection"][0].update(selection_score=0.0),
            lambda a: a["inner_selection"][0]["inner_fold_scores"].pop(),
            lambda a: a["inner_selection"][0]["inner_fold_scores"].__setitem__(0, False),
            lambda a: a["inner_selection"][0].update(stale_detail={}),
            lambda a: a["inner_selection"][0]["inner_fit_details"].pop(),
            lambda a: a["inner_selection"][0]["inner_fit_details"][0].update(inner_fold=True),
            lambda a: a["inner_selection"][0]["inner_fit_details"][0].update(unknown=1),
            lambda a: a.update(selected_family="dummy"),
            lambda a: a.update(selected_candidate="group_balanced_ridge_alpha_10"),
            lambda a: a["family_candidates"].update(group_balanced_ridge="group_balanced_ridge_alpha_10"),
            lambda a: a["family_fit_details"][FAMILY].update(alpha=1.0),
            lambda a: a["family_fit_details"].update(dummy={}),
            lambda a: a["family_fit_details"].pop(FAMILY),
            lambda a: a["inner_folds"][0].update(train_group_count=True),
            lambda a: a["inner_folds"][0].update(overlap_count=True),
            lambda a: a["inner_folds"][0].update(test_groups=["a", "a", "a"]),
            lambda a: a["inner_folds"][0].update(domain=None),
            lambda a: a["inner_folds"][0].update(unknown=None),
        ]
        for i, mutate in enumerate(changes):
            with self.subTest(change=i):
                pred = prediction_fixture(); mutate(pred)
                with self.assertRaises(ValueError): _metadata(pred)
                manifest, summary, raw = evaluation_fixture(); mutate(manifest["outer_folds"][0])
                with self.assertRaises(ValueError): _check(manifest, summary, raw)

    def test_recipe_and_weight_field_corruptions(self):
        replacements = {"implementation": "unknown", "training_rows": True, "training_groups": True,
            "transformed_features": True, "alpha": True, "solver": "svd", "tol": True,
            "max_iter": 100, "fit_intercept": 1, "copy_X": 1, "positive": 0, "threads": True,
            "fitting_loss": "group_mae", "weight_rule": "equal_domain", "normalization": "sum_groups",
            "preprocessing_weighting": "weighted", "min_training_group_size": True,
            "max_training_group_size": 100, "min_row_weight": True,
            "max_row_weight": 1.0, "sum_row_weights": 6.0}
        for key, replacement in replacements.items():
            for where in ("inner", "final"):
                with self.subTest(key=key, where=where):
                    pred = prediction_fixture()
                    detail = (pred["family_fit_details"][FAMILY] if where == "final" else
                              pred["inner_selection"][0]["inner_fit_details"][0]["details"])
                    detail[key] = replacement
                    with self.assertRaises(ValueError): _metadata(pred)
        for mutation in (lambda d: d.pop("tol"), lambda d: d.update(weights=[1.0])):
            pred = prediction_fixture(); mutation(pred["family_fit_details"][FAMILY])
            with self.assertRaises(ValueError): _metadata(pred)

    def test_capability_grid_configuration_and_old_version_rejections(self):
        changes = [lambda m: m.update(capabilities=[]), lambda m: m.update(capabilities=[FAMILY + "-v1"] * 2),
            lambda m: m.update(capabilities=[FAMILY + "-v2"]),
            lambda m: m["config"].update(baselines=[FAMILY, FAMILY]),
            lambda m: m["config"].update(baselines=["dummy"]),
            lambda m: m["config"]["candidates"][0].update(params={"alpha": True}),
            lambda m: m["config"]["candidates"].reverse(),
            lambda m: m["config"].update(threads=True),
            lambda m: m["config"].update(inner_splits=True),
            lambda m: m["config"].update(random_state=True),
            lambda m: m["config"].update(inner_strategy="leave_one_domain_out"),
            lambda m: m["config"].update(domain="d"),
            lambda m: m.update(schema_version="9.0")]
        for i, mutate in enumerate(changes):
            with self.subTest(change=i):
                pred = prediction_fixture(); mutate(pred)
                with self.assertRaises(ValueError): _metadata(pred)
                manifest, summary, raw = evaluation_fixture(); mutate(manifest)
                with self.assertRaises(ValueError): _check(manifest, summary, raw)
        for version, protocol in [("1.1", None), ("1.2", "group-fit-predict-distinct-v2"),
                                  ("1.3", "group-fit-predict-extended-families-v1"), ("1.4", "domain-inner-fit-predict-v1")]:
            pred = prediction_fixture(); pred.update(schema_version=version)
            if protocol is None: pred.pop("protocol")
            else: pred["protocol"] = protocol
            # Even stripping config/capabilities leaves explicit stale fit declarations.
            pred.pop("capabilities"); pred["config"].pop("candidates"); pred["config"]["baselines"] = ["dummy"]
            with self.assertRaises(ValueError): _metadata(pred)

    def test_ordinary_records_and_optional_final_alpha_are_strict(self):
        pred = prediction_fixture(("dummy", FAMILY, "nystroem_ridge"))
        for mutate in (lambda p: p["inner_selection"][0].update(params={"strategy": "median"}),
                       lambda p: p["inner_selection"][0].update(inner_fit_details=[]),
                       lambda p: p["family_fit_details"]["nystroem_ridge"].update(alpha=10.0)):
            changed = deepcopy(pred); mutate(changed)
            with self.assertRaises(ValueError): _metadata(changed)
        pred["capabilities"].reverse()
        with self.assertRaises(ValueError): _metadata(pred)

    def test_same_training_slice_has_consistent_preprocessing_and_weights(self):
        for change in (
            lambda p: p["inner_selection"][0]["inner_fit_details"][0]["details"].update(transformed_features=2),
            lambda p: p["family_fit_details"]["group_median"].update(transformed_features=2),
        ):
            pred = prediction_fixture((FAMILY, "group_median")); change(pred)
            with self.assertRaisesRegex(ValueError, "widths differ"): _metadata(pred)
        pred = prediction_fixture()
        # Five rows / three groups could have group sizes (1,1,3) or (1,2,2),
        # but different alphas cannot have different weights on the same slice.
        pred["inner_selection"][0]["inner_fit_details"][0]["details"].update(
            max_training_group_size=3, min_row_weight=5 / 9)
        with self.assertRaisesRegex(ValueError, "summaries differ"): _metadata(pred)

    def test_tie_order_and_final_alpha_follow_declared_family_winner(self):
        pred = prediction_fixture((FAMILY, "dummy"))
        for record in pred["inner_selection"]:
            record["selection_score"] = 0.0
            record["inner_fold_scores"] = [0.0, 0.0]
        self.assertEqual(_metadata(pred)[0], "1.5")
        pred.update(selected_candidate="dummy_mean", selected_family="dummy")
        with self.assertRaises(ValueError): _metadata(pred)
        pred = prediction_fixture()
        pred["inner_selection"][1]["selection_score"] = 0.5
        pred["inner_selection"][1]["inner_fold_scores"] = [0.5, 0.5]
        pred.update(selected_candidate="group_balanced_ridge_alpha_1")
        pred["family_candidates"][FAMILY] = "group_balanced_ridge_alpha_1"
        with self.assertRaises(ValueError): _metadata(pred)
        pred["family_fit_details"][FAMILY]["alpha"] = 1.0
        self.assertEqual(_metadata(pred)[0], "1.5")

    def test_inner_group_extrema_are_bounded_by_the_final_population(self):
        pred = prediction_fixture()
        for record in pred["inner_selection"]:
            # Locally feasible (1,1,3), but a whole-group subset cannot contain
            # size3 when the full population declares maximum size2.
            record["inner_fit_details"][0]["details"].update(
                max_training_group_size=3, min_row_weight=5 / 9)
        with self.assertRaisesRegex(ValueError, "final training population bounds"):
            _metadata(pred)
        pred = prediction_fixture(sizes={"a": 2, "b": 3, "c": 4, "d": 2, "e": 3, "f": 4})
        for record in pred["inner_selection"]:
            # Locally feasible (1,4,4), with unchanged maximum4, but violates
            # the full population's declared minimum size2.
            record["inner_fit_details"][0]["details"].update(
                min_training_group_size=1, max_row_weight=3.0)
        with self.assertRaisesRegex(ValueError, "final training population bounds"):
            _metadata(pred)
        pred = prediction_fixture()
        # Every full-training group occurs in an inner training slice, so an
        # overbroad final maximum also contradicts the complete fold coverage.
        pred["family_fit_details"][FAMILY].update(max_training_group_size=3, min_row_weight=0.5)
        with self.assertRaisesRegex(ValueError, "final training population extrema"):
            _metadata(pred)
        pred = prediction_fixture(sizes={"a": 2, "b": 3, "c": 4, "d": 2, "e": 3, "f": 4})
        pred["family_fit_details"][FAMILY].update(min_training_group_size=1, max_row_weight=3.0)
        with self.assertRaisesRegex(ValueError, "final training population extrema"):
            _metadata(pred)

    def test_coherent_alternate_prediction_extrema_remain_unverifiable(self):
        pred = prediction_fixture(sizes={"a": 2, "b": 3, "c": 4, "d": 2, "e": 3, "f": 4})
        self.assertEqual(_metadata(pred)[0], "1.5")
        # Each half could instead be (1,4,4), keeping every saved count.
        # Prediction metadata cannot determine which row population was used.
        for record in pred["inner_selection"]:
            for fit in record["inner_fit_details"]:
                fit["details"].update(min_training_group_size=1, max_row_weight=3.0)
        pred["family_fit_details"][FAMILY].update(min_training_group_size=1, max_row_weight=3.0)
        self.assertEqual(_metadata(pred)[0], "1.5")

    def test_evaluation_summary_selected_column_counts_and_seed(self):
        for mutate in (lambda m, s: s.update(protocol="other"),
                       lambda m, s: s.update(capabilities=[]),
                       lambda m, s: m["outer_folds"][0].update(inner_seed=True),
                       lambda m, s: m["outer_folds"][0].update(inner_seed=2),
                       lambda m, s: m["outer_folds"][0].update(unknown=1)):
            manifest, summary, raw = evaluation_fixture(); mutate(manifest, summary)
            with self.assertRaises(ValueError): _check(manifest, summary, raw)
        manifest, summary, raw = evaluation_fixture()
        altered = raw.replace(b"0,a,0,0.0,0.0,0.0", b"0,a,0,0.0,0.0,1.0", 1)
        with self.assertRaisesRegex(ValueError, "selected predictions"): _check(manifest, summary, altered)
        altered = raw.replace(b"prediction_selected", b"prediction_unknown")
        with self.assertRaisesRegex(ValueError, "columns"): _check(manifest, summary, altered)

    def test_rehashed_corruption_fails_and_validators_do_not_write(self):
        for evaluation in (False, True):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                manifest, summary, raw = evaluation_fixture() if evaluation else (prediction_fixture(), None, b"row_position,prediction\n0,0.0\n1,0.0\n")
                write_bundle(root, manifest, summary, raw)
                validate = validate_evaluation_bundle if evaluation else validate_prediction_bundle_schema
                before = {p.name: p.read_bytes() for p in root.iterdir()}
                checked = validate(root)
                self.assertEqual(checked.schema_version, "1.4" if evaluation else "1.5")
                self.assertIn("extended_family_declarations", checked.checked_scope)
                self.assertIn("weight_execution", checked.unchecked_metadata)
                self.assertIn("actual_inner_predictions_and_score_computation", checked.unchecked_metadata)
                self.assertNotIn("inner_selection", checked.unchecked_metadata)
                self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
                target = manifest["outer_folds"][0] if evaluation else manifest
                target["family_fit_details"][FAMILY]["normalization"] = "sum_groups"
                write_bundle(root, manifest, summary, raw)
                verify_bundle(root)
                with self.assertRaises(ValueError): validate(root)


if __name__ == "__main__":
    unittest.main()
