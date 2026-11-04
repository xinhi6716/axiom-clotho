"""Invented ALD declarations: no scientific data, I/O or model fits."""

from dataclasses import replace
import json

from groupbench.process_contract import (
    ALDExperimentalContract,
    ALDObservation,
    ProcessFeatureDeclaration,
    ThicknessCycleDerivation,
    preview_ald_contract,
)


def main():
    contract = ALDExperimentalContract(
        temperature_role="substrate",
        temperature_basis="setpoint",
        target_method="total_thickness_div_cycles",
        target_column="growth_average",
        features=(ProcessFeatureDeclaration("temperature_setting", "process_setting", "before_process"),),
    )
    recipe = ThicknessCycleDerivation(
        thickness="12.5", thickness_unit="nm", cycle_count=250,
        source_id="invented-source", source_version="invented-v1",
        source_locator="invented table 1 row 1", condition_id="invented-condition-1",
    )
    row = ALDObservation(
        source_id=recipe.source_id, source_version=recipe.source_version,
        source_locator=recipe.source_locator, condition_id=recipe.condition_id,
        temperature="200", temperature_unit="degC", temperature_role="substrate",
        temperature_basis="setpoint", temperature_condition_id=recipe.condition_id,
        target="0.5", target_unit="angstrom/cycle", target_method="total_thickness_div_cycles",
        target_condition_id=recipe.condition_id, observation_kind="experimental",
        evidence_scope="full_text", batch_id="invented-batch-1",
        replicate_id="invented-sample-1", derivation=recipe,
    )
    reports = {
        "invented_consistent_declarations": preview_ald_contract([row], contract=contract),
        "wrong_temperature_role": preview_ald_contract(
            [replace(row, temperature_role="precursor_vessel")], contract=contract),
        "average_mislabeled_as_slope": preview_ald_contract(
            [replace(row, target_method="steady_growth_slope")], contract=contract),
        "incorrect_average_arithmetic": preview_ald_contract(
            [replace(row, target="0.6")], contract=contract),
        "postprocess_feature": preview_ald_contract([row], contract=replace(contract, features=(
            ProcessFeatureDeclaration("thickness_after_growth", "postprocess_measurement", "after_process"),
        ))),
        "unknown_replicate": preview_ald_contract([replace(row, replicate_id=None)], contract=contract),
    }
    expected = {
        "invented_consistent_declarations": set(),
        "wrong_temperature_role": {"declared_role_or_method_mismatch"},
        "average_mislabeled_as_slope": {"declared_role_or_method_mismatch", "derivation_not_applicable_to_method"},
        "incorrect_average_arithmetic": {"derived_target_arithmetic_mismatch"},
        "postprocess_feature": {"feature_role_not_preprocess", "feature_not_available_before_process"},
        "unknown_replicate": {"unknown_independence_metadata"},
    }
    for name, report in reports.items():
        assert set(report["issue_counts"]) == expected[name]
        assert report["training_admission"] == "not_performed"
    print(json.dumps({"fixture": "invented_not_measured", "model_fits": 0, "reports": reports},
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
