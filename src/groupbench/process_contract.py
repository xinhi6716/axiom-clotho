"""Opt-in process declaration checks, not scientific validation or ML admission.

No I/O, fitting, repairs, row dropping, physical inference or source retrieval.
Only caller-declared roles, identities, availability and bounded exact arithmetic
are checked. Output contains fixed codes and positions, never input values.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import unicodedata

from .quantity_contract import QuantityDeclaration, normalize_decimal_tokens


MAX_RECORDS = 20_000
MAX_FEATURES = 100
MAX_ISSUES = 1_000
MAX_CYCLES = 1_000_000_000
MAX_QUARANTINE_SCOPES = 1_000
MAX_SCOPE_REFERENCES = 20_000
_ROLES = frozenset({"substrate", "stage", "precursor_vessel", "reactor"})
_BASES = frozenset({"measured", "setpoint", "reported_unspecified"})
_METHODS = frozenset({"steady_growth_slope", "total_thickness_div_cycles"})
_FEATURE_ROLES = frozenset({"process_setting", "material_identity", "postprocess_measurement",
                            "computed_target", "identifier", "unknown"})
_AVAILABILITY = frozenset({"before_process", "after_process", "unknown"})
_LENGTH_UNITS = {"angstrom": 1, "nm": 10}
_GROWTH_UNITS = {"angstrom/cycle": 1, "nm/cycle": 10}
_TOKEN = QuantityDeclaration("scalar", "decimal_token", "literal", "caller_declared", "dimensionless", "1")


def _text(value):
    return (type(value) is str and 1 <= len(value) <= 256 and value == value.strip()
            and not any(unicodedata.category(c) in ("Cc", "Cs") for c in value))


def _member(value, choices):
    return type(value) is str and value in choices


def _identity(value):
    return _text(value) and value.casefold() not in {"unknown", "unreported", "not_reported"}


def _fraction(token):
    # Reuse the established 256-character / 64-digit / exponent-200 contract.
    # Fraction preserves exactness regardless of the global Decimal context.
    return Fraction(normalize_decimal_tokens([token], _TOKEN)[0])


@dataclass(frozen=True)
class ProcessFeatureDeclaration:
    column: str
    role: str
    availability: str


@dataclass(frozen=True)
class ALDExperimentalContract:
    temperature_role: str
    temperature_basis: str
    target_method: str
    target_column: str
    features: tuple[ProcessFeatureDeclaration, ...] = ()
    derived_target_tolerance_angstrom_per_cycle: str = "0"

    def __post_init__(self):
        if (not _member(self.temperature_role, _ROLES)
                or not _member(self.temperature_basis, _BASES)
                or not _member(self.target_method, _METHODS) or not _text(self.target_column)):
            raise ValueError("Contract requires explicit supported temperature role/basis, experimental target method and bounded target column.")
        if type(self.features) is not tuple or len(self.features) > MAX_FEATURES:
            raise ValueError("Contract features must be a tuple of at most 100 declarations.")
        seen = set()
        for feature in self.features:
            if (type(feature) is not ProcessFeatureDeclaration or not _text(feature.column)
                    or not _member(feature.role, _FEATURE_ROLES)
                    or not _member(feature.availability, _AVAILABILITY)):
                raise ValueError("Feature declarations require bounded columns and supported role/availability values.")
            if feature.column in seen:
                raise ValueError("Selected feature declarations must have unique columns.")
            seen.add(feature.column)
        try:
            tolerance = _fraction(self.derived_target_tolerance_angstrom_per_cycle)
        except ValueError:
            raise ValueError("Derived-target tolerance must be bounded nonnegative decimal text in angstrom/cycle.") from None
        if tolerance < 0:
            raise ValueError("Derived-target tolerance must be bounded nonnegative decimal text in angstrom/cycle.")


@dataclass(frozen=True)
class ThicknessCycleDerivation:
    thickness: str
    thickness_unit: str
    cycle_count: int
    source_id: str
    source_version: str
    source_locator: str
    condition_id: str


@dataclass(frozen=True)
class ALDObservation:
    source_id: str | None
    source_version: str | None
    source_locator: str | None
    condition_id: str | None
    temperature: str
    temperature_unit: str
    temperature_role: str
    temperature_basis: str
    temperature_condition_id: str | None
    target: str
    target_unit: str
    target_method: str
    target_condition_id: str | None
    observation_kind: str
    evidence_scope: str
    batch_id: str | None
    replicate_id: str | None
    derivation: ThicknessCycleDerivation | None = None


def preview_ald_contract(records, *, contract):
    """Return declaration conflicts for a bounded list/tuple of observations.

    Inputs and frozen declarations are rechecked on every call. Missing or
    unsupported metadata is flagged, never guessed. A conflict-free report is
    not evidence that declarations are true, source-verified or independent.
    No output row subset or model-ready table is produced.
    """
    if type(contract) is not ALDExperimentalContract:
        raise ValueError("An explicit ALDExperimentalContract is required.")
    ALDExperimentalContract.__post_init__(contract)
    if type(records) not in (list, tuple) or not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError("Observations must be a nonempty list or tuple of at most 20,000 records.")
    if any(type(record) is not ALDObservation for record in records):
        raise ValueError("Every observation must be an ALDObservation, not an inferred mapping or arbitrary object.")
    tolerance = _fraction(contract.derived_target_tolerance_angstrom_per_cycle)
    issues, counts, flagged_rows, flagged_features = [], Counter(), set(), set()

    def add(code, field, *, row=None, feature=None):
        counts[code] += 1
        if row is not None:
            flagged_rows.add(row)
        if feature is not None:
            flagged_features.add(feature)
        if len(issues) < MAX_ISSUES:
            issues.append({"code": code, "field": field, "row_position": row,
                           "feature_position": feature})

    for position, feature in enumerate(contract.features):
        if feature.column == contract.target_column:
            add("target_selected_as_feature", "feature.column", feature=position)
        if feature.role not in {"process_setting", "material_identity"}:
            add("feature_role_not_preprocess", "feature.role", feature=position)
        if feature.availability != "before_process":
            add("feature_not_available_before_process", "feature.availability", feature=position)

    identities = defaultdict(list)
    for position, record in enumerate(records):
        valid_ids = {}
        for field in ("source_id", "source_version", "source_locator", "condition_id",
                      "temperature_condition_id", "target_condition_id"):
            valid_ids[field] = _identity(getattr(record, field))
            if not valid_ids[field]:
                add("missing_or_invalid_provenance", field, row=position)
        for field in ("batch_id", "replicate_id"):
            value = getattr(record, field)
            if value is None or (type(value) is str and value.casefold() in {"unknown", "unreported", "not_reported"}):
                add("unknown_independence_metadata", field, row=position)
            elif not _identity(value):
                add("invalid_independence_metadata", field, row=position)
        if _member(record.observation_kind, {"simulation", "theory", "unknown"}):
            add("nonexperimental_or_unknown_observation", "observation_kind", row=position)
        elif not _member(record.observation_kind, {"experimental"}):
            add("unsupported_observation_kind", "observation_kind", row=position)
        if _member(record.evidence_scope, {"abstract", "snippet", "unknown"}):
            add("limited_or_unknown_source_scope", "evidence_scope", row=position)
        elif not _member(record.evidence_scope, {"full_text"}):
            add("unsupported_evidence_scope", "evidence_scope", row=position)
        for field, expected, choices in (
            ("temperature_role", contract.temperature_role, _ROLES),
            ("temperature_basis", contract.temperature_basis, _BASES),
            ("target_method", contract.target_method, _METHODS),
        ):
            value = getattr(record, field)
            if not _member(value, choices):
                add("unknown_or_unsupported_declaration", field, row=position)
            elif value != expected:
                add("declared_role_or_method_mismatch", field, row=position)
        for field in ("temperature_condition_id", "target_condition_id"):
            if valid_ids["condition_id"] and valid_ids[field] and getattr(record, field) != record.condition_id:
                add("condition_identity_mismatch", field, row=position)
        if _member(record.temperature_basis, {"reported_unspecified"}):
            add("unknown_temperature_basis", "temperature_basis", row=position)

        if not _member(record.temperature_unit, {"degC", "K"}):
            add("unsupported_temperature_unit", "temperature_unit", row=position)
        try:
            _fraction(record.temperature)
        except ValueError:
            add("invalid_decimal_token", "temperature", row=position)
        target = None
        if not _member(record.target_unit, _GROWTH_UNITS):
            add("unsupported_growth_unit", "target_unit", row=position)
        try:
            target = _fraction(record.target)
        except ValueError:
            add("invalid_decimal_token", "target", row=position)

        if _member(record.target_method, {"total_thickness_div_cycles"}):
            recipe = record.derivation
            if type(recipe) is not ThicknessCycleDerivation:
                add("missing_or_invalid_derivation", "derivation", row=position)
            else:
                recipe_ids = {}
                for field in ("source_id", "source_version", "source_locator", "condition_id"):
                    recipe_ids[field] = _identity(getattr(recipe, field))
                    if not recipe_ids[field]:
                        add("missing_or_invalid_provenance", "derivation." + field, row=position)
                for field in ("source_id", "source_version", "condition_id"):
                    if valid_ids[field] and recipe_ids[field] and getattr(recipe, field) != getattr(record, field):
                        add("derivation_identity_mismatch", "derivation." + field, row=position)
                unit_ok = _member(recipe.thickness_unit, _LENGTH_UNITS)
                if not unit_ok:
                    add("unsupported_thickness_unit", "derivation.thickness_unit", row=position)
                cycles_ok = type(recipe.cycle_count) is int and 1 <= recipe.cycle_count <= MAX_CYCLES
                if not cycles_ok:
                    add("invalid_cycle_count", "derivation.cycle_count", row=position)
                thickness = None
                try:
                    thickness = _fraction(recipe.thickness)
                except ValueError:
                    add("invalid_decimal_token", "derivation.thickness", row=position)
                if target is not None and _member(record.target_unit, _GROWTH_UNITS) and thickness is not None and unit_ok and cycles_ok:
                    expected = thickness * _LENGTH_UNITS[recipe.thickness_unit] / recipe.cycle_count
                    actual = target * _GROWTH_UNITS[record.target_unit]
                    if abs(actual - expected) > tolerance:
                        add("derived_target_arithmetic_mismatch", "target", row=position)
        elif record.derivation is not None:
            add("derivation_not_applicable_to_method", "derivation", row=position)

        # Locators and source versions do not create a new physical sample.
        # Unknown replicates share a key and remain explicitly flagged above.
        if (valid_ids["source_id"] and valid_ids["condition_id"]
                and (record.batch_id is None or _identity(record.batch_id))
                and (record.replicate_id is None or _identity(record.replicate_id))):
            identities[(record.source_id, record.condition_id, record.batch_id, record.replicate_id)].append(position)
    for positions in identities.values():
        if len(positions) > 1:
            for position in positions:
                add("repeated_declared_observation_identity", "declared_identity", row=position)

    total = sum(counts.values())
    return {
        "schema_version": "1.0", "kind": "ald_declaration_preview",
        "action": "validation_only_no_data_changes", "training_admission": "not_performed",
        "status": "declaration_issues_found" if total else "no_declared_conflicts",
        "n_rows": len(records), "n_selected_features": len(contract.features),
        "n_rows_with_issues": len(flagged_rows),
        "n_feature_declarations_with_issues": len(flagged_features), "issue_count": total,
        "issue_counts": dict(sorted(counts.items())), "issues": issues,
        "issues_truncated": total > len(issues), "max_returned_issues": MAX_ISSUES,
        "limitations": [
            "Caller declarations are not verified against source documents or physical reality.",
            "Availability declarations do not prove freedom from leakage or causal validity.",
            "Exact declared identities do not establish independent samples or detect source aliases.",
            "Arithmetic agreement does not establish steady-state growth, measurement uncertainty or accuracy.",
            "No rows are repaired, dropped, admitted to training, or assigned inferred groups.",
        ],
    }


@dataclass(frozen=True)
class ObservationEvidenceKey:
    """Metadata only; no quantity values or implied specimen identity."""

    source_id: str
    recipe_group_id: str
    endpoint_name: str


@dataclass(frozen=True)
class QuarantineScope:
    """One caller-declared unresolved conflict, not a general source issue.

    The named endpoints apply to every named recipe group in this source.
    None means the endpoint mapping is unknown, never an empty quarantine.
    Split nonrectangular scopes into separate declarations.
    """

    source_id: str
    recipe_group_ids: tuple[str, ...]
    endpoint_names: tuple[str, ...] | None


def preview_evidence_quarantine(observations, *, unresolved_scopes):
    """Check whether exact declared quarantined quantities remain in inventory.

    This checks the supplied inventory regardless of downstream ML admission.
    Scope selection and endpoint mapping are explicit caller decisions. It does
    not read values, resolve evidence, infer aliases, remove rows, or establish
    provenance, paired specimens, independent samples or training eligibility.
    """
    if type(observations) not in (list, tuple) or len(observations) > MAX_RECORDS:
        raise ValueError("Observation metadata must be a list or tuple of at most 20,000 rows.")
    if type(unresolved_scopes) not in (list, tuple) or len(unresolved_scopes) > MAX_QUARANTINE_SCOPES:
        raise ValueError("Unresolved scopes must be an explicit list or tuple of at most 1,000 declarations.")

    # Validate everything before constructing any report. No arbitrary object,
    # iterator, mapping, or dataclass subclass can supply inferred identities.
    missing = object()  # Distinct from an explicitly unknown endpoint mapping.
    for row in observations:
        if (type(row) is not ObservationEvidenceKey
                or not all(_identity(getattr(row, field, missing)) for field in
                           ("source_id", "recipe_group_id", "endpoint_name"))):
            raise ValueError("Observation evidence keys require bounded explicit source, recipe group and endpoint identities.")
    references = 0
    for scope in unresolved_scopes:
        if type(scope) is not QuarantineScope or not _identity(getattr(scope, "source_id", missing)):
            raise ValueError("Each unresolved scope requires a QuarantineScope with an explicit source identity.")
        for field in ("recipe_group_ids", "endpoint_names"):
            values = getattr(scope, field, missing)
            if field == "endpoint_names" and values is None:
                continue
            if (type(values) is not tuple or not 1 <= len(values) <= MAX_SCOPE_REFERENCES
                    or not all(_identity(value) for value in values)):
                raise ValueError("Scope members must be nonempty bounded tuples of explicit identities; only endpoint mapping may be None.")
            if len(set(values)) != len(values):
                raise ValueError("Members within each scope declaration must be unique.")
            references += len(values)
            if references > MAX_SCOPE_REFERENCES:
                raise ValueError("The total declared group and endpoint references must not exceed 20,000.")

    # Index only observed keys. Never expand a group x endpoint Cartesian product.
    indexed = defaultdict(lambda: defaultdict(list))
    for position, row in enumerate(observations):
        indexed[(row.source_id, row.recipe_group_id)][row.endpoint_name].append(position)

    issues, counts = [], Counter()
    matched_rows, potential_rows = set(), set()
    unmapped_scopes = 0

    def add(code, field, scope_position, row_position=None):
        counts[code] += 1
        if len(issues) < MAX_ISSUES:
            issues.append({"code": code, "field": field, "scope_position": scope_position,
                           "row_position": row_position})

    for scope_position, scope in enumerate(unresolved_scopes):
        if scope.endpoint_names is None:
            unmapped_scopes += 1
            add("scope_endpoint_mapping_not_established", "scope.endpoint_names", scope_position)
            for group in scope.recipe_group_ids:
                for positions in indexed.get((scope.source_id, group), {}).values():
                    potential_rows.update(positions)
            continue
        endpoints = set(scope.endpoint_names)
        for group in scope.recipe_group_ids:
            for endpoint, positions in indexed.get((scope.source_id, group), {}).items():
                if endpoint in endpoints:
                    for position in positions:
                        matched_rows.add(position)
                        add("quarantined_observation_present", "observation_evidence_key", scope_position, position)

    total = sum(counts.values())
    status = ("quarantine_violation_found" if matched_rows else
              "mapping_not_established" if unmapped_scopes else "no_declared_quarantine_matches")
    return {
        "schema_version": "1.0", "kind": "evidence_quarantine_preview",
        "action": "audit_only_no_data_changes", "training_admission": "not_performed",
        "status": status, "n_observation_rows": len(observations),
        "n_unresolved_scopes": len(unresolved_scopes), "n_unmapped_scopes": unmapped_scopes,
        "n_rows_with_quarantine_matches": len(matched_rows),
        "n_rows_in_unmapped_scope_groups": len(potential_rows),
        "issue_count": total, "issue_counts": dict(sorted(counts.items())), "issues": issues,
        "issues_truncated": total > len(issues), "max_returned_issues": MAX_ISSUES,
        "limitations": [
            "Only supplied unresolved scopes and exact identities are checked; registry completeness and source truth are not verified.",
            "Scope membership and endpoint mapping are caller declarations, not inferred from text or quantity values.",
            "Unmapped scope groups indicate potential relevance, not confirmed quarantined quantities or safe remaining rows.",
            "Counts describe metadata rows and declared conflict matches, not independent specimens or scientifically admissible samples.",
            "No rows are repaired, removed, admitted to training, or linked as paired physical specimens.",
        ],
    }
