"""Invented quantity-local quarantine controls; no measurements or model fits."""

import json

from groupbench.process_contract import (
    ObservationEvidenceKey as Key,
    QuarantineScope as Scope,
    preview_evidence_quarantine,
)


def main():
    scopes = [Scope("invented-source", ("recipe-2",), ("peak_ratio",))]
    clean = [Key("invented-source", "recipe-1", "peak_ratio")]
    reports = {
        "clean_isolation": preview_evidence_quarantine(clean, unresolved_scopes=scopes),
        "injected_quarantined_key": preview_evidence_quarantine(
            clean + [Key("invented-source", "recipe-2", "peak_ratio")], unresolved_scopes=scopes),
        "same_group_other_endpoint": preview_evidence_quarantine(
            clean + [Key("invented-source", "recipe-2", "peak_area")], unresolved_scopes=scopes),
        "unmapped_quantity_family": preview_evidence_quarantine(
            clean, unresolved_scopes=[Scope("invented-source", ("recipe-2",), None)]),
    }
    expected = {
        "clean_isolation": "no_declared_quarantine_matches",
        "injected_quarantined_key": "quarantine_violation_found",
        "same_group_other_endpoint": "no_declared_quarantine_matches",
        "unmapped_quantity_family": "mapping_not_established",
    }
    for name, report in reports.items():
        assert report["status"] == expected[name]
        assert report["training_admission"] == "not_performed"
    assert reports["injected_quarantined_key"]["n_rows_with_quarantine_matches"] == 1
    print(json.dumps({"fixture": "invented_metadata_no_measurements", "model_fits": 0,
                      "reports": reports}, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
