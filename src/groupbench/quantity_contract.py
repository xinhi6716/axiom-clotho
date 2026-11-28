"""Opt-in, no-fit electrical quantity helpers; NOT an ML admission gate.

Declarations are caller assertions, not verified physical facts. A per-column
unit cannot detect secretly mixed-unit rows. No raw-value provenance is retained.
"""
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
import re
import unicodedata

REGISTRY_VERSION = "electrical-decimal-v2"
MAX_VALUES = 20_000
# dimension, canonical unit, exact base-ten exponent
_UNITS = {
    "A": ("electric_current", "A", 0),
    "mA": ("electric_current", "A", -3),
    "uA": ("electric_current", "A", -6),
    "nA": ("electric_current", "A", -9),
    "pA": ("electric_current", "A", -12),
    "V": ("electric_potential", "V", 0),
    "mV": ("electric_potential", "V", -3),
    "uV": ("electric_potential", "V", -6),
    "ohm": ("resistance", "ohm", 0),
    "kohm": ("resistance", "ohm", 3),
    "Mohm": ("resistance", "ohm", 6),
    "S": ("electric_conductance", "S", 0),
    "mS": ("electric_conductance", "S", -3),
    "uS": ("electric_conductance", "S", -6),
    "nS": ("electric_conductance", "S", -9),
    "pS": ("electric_conductance", "S", -12),
    "s": ("elapsed_time", "s", 0),
    "ms": ("elapsed_time", "s", -3),
    "us": ("elapsed_time", "s", -6),
    "ns": ("elapsed_time", "s", -9),
    "ks": ("elapsed_time", "s", 3),
    "1": ("dimensionless", "1", 0),
}
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE]([+-]?[0-9]+))?\Z")


@dataclass(frozen=True)
class QuantityDeclaration:
    column: str
    quantity_id: str
    statistic_id: str
    condition_id: str
    dimension: str
    unit: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (type(value) is not str or not value or len(value) > 256
                    or value != value.strip() or any(unicodedata.category(c) == "Cc" for c in value)):
                raise ValueError("Quantity declarations require bounded unpadded text without control characters.")
        if self.unit not in _UNITS or _UNITS[self.unit][0] != self.dimension:
            raise ValueError("Unsupported unit or inconsistent declared dimension.")


def _declaration(value):
    if type(value) is not QuantityDeclaration:
        raise ValueError("An explicit QuantityDeclaration is required.")
    # Recheck even if a caller circumvented frozen dataclass assignment.
    value.__post_init__()
    return value


def compare_declarations(expected, supplied, *, mode="validate"):
    """Reject different meanings; differing supported units require normalize mode.

    Return a canonical declaration in normalize mode, otherwise the expectation.
    No synonym inference or physical comparability certification is performed.
    """
    expected, supplied = _declaration(expected), _declaration(supplied)
    if type(mode) is not str or mode not in ("validate", "normalize"):
        raise ValueError("mode must be validate or normalize.")
    fields = ("column", "quantity_id", "statistic_id", "condition_id", "dimension")
    if any(getattr(expected, f) != getattr(supplied, f) for f in fields):
        raise ValueError("Quantity, statistic, condition, column or dimension declarations differ.")
    if mode == "validate":
        if expected.unit != supplied.unit:
            raise ValueError("Different units require explicit normalize mode.")
        return expected
    return QuantityDeclaration(*(getattr(expected, f) for f in fields), _UNITS[expected.unit][1])


def _values(values):
    if type(values) not in (list, tuple) or not 1 <= len(values) <= MAX_VALUES:
        raise ValueError("Values must be a nonempty bounded list or tuple.")


def normalize_decimal_tokens(tokens, declaration):
    """Explicitly convert bounded decimal text to exact canonical-unit Decimals.

    Accept only strings, at most 256 characters and 64 coefficient digits, with
    explicit and effective Decimal exponent magnitudes at most 200. Missing
    values are unsupported. Never parse a previously rounded float as raw text.
    """
    declaration = _declaration(declaration)
    _values(tokens)
    shift = _UNITS[declaration.unit][2]
    output = []
    for token in tokens:
        if type(token) is not str or len(token) > 256:
            raise ValueError("Expected bounded decimal text, not typed numeric values.")
        match = _NUMBER.fullmatch(token)
        if match is None or (match.group(1) is not None and abs(int(match.group(1))) > 200):
            raise ValueError("Invalid decimal token or exponent outside bounds.")
        coefficient = re.split("[eE]", token)[0]
        if sum(c.isdigit() for c in coefficient) > 64:
            raise ValueError("Decimal coefficient exceeds 64 digits.")
        value = Decimal(token)
        if abs(value.as_tuple().exponent) > 200:
            raise ValueError("Effective decimal exponent outside bounds.")
        # Tuple construction shifts the exponent without context rounding.
        sign, digits, exponent = value.as_tuple()
        output.append(Decimal((sign, digits, exponent + shift)))
    return tuple(output)


def decimal_to_float64(values):
    """Explicit approximate Python binary64 boundary, with conservative guards.

    Integral values must be exactly representable. Fractional values can round;
    distinct supplied decimals may not collapse within this batch. No claim is
    made about cross-batch collisions or decimal precision already lost upstream.
    """
    _values(values)
    output, seen = [], {}
    for value in values:
        if type(value) is not Decimal or not value.is_finite():
            raise ValueError("Expected finite Decimal values.")
        result = float(value)
        if not math.isfinite(result) or (value != 0 and result == 0):
            raise ValueError("Float64 overflow or nonzero underflow.")
        if value == value.to_integral_value() and Decimal.from_float(result) != value:
            raise ValueError("Integral values must be exactly representable in float64.")
        if result in seen and seen[result] != value:
            raise ValueError("Distinct decimals collapse at the float64 boundary.")
        seen[result] = value
        output.append(result)
    return tuple(output)


def normalize_comparable_pair_to_float64(expected, expected_tokens, supplied, supplied_tokens):
    """Normalize a declared pair and guard the shared binary64 boundary.

    Both batches must be nonempty and together contain at most MAX_VALUES
    entries. Validate matching measurement meaning before parsing either
    batch. Reject collisions across as well as within batches. Return the
    canonical declaration and two immutable tuples in input order.
    """
    canonical = compare_declarations(expected, supplied, mode="normalize")
    _values(expected_tokens)
    _values(supplied_tokens)
    n = len(expected_tokens)
    if n + len(supplied_tokens) > MAX_VALUES:
        raise ValueError("Combined values exceed the float64 comparison limit.")
    a = normalize_decimal_tokens(expected_tokens, expected)
    b = normalize_decimal_tokens(supplied_tokens, supplied)
    converted = decimal_to_float64(a + b)
    return canonical, converted[:n], converted[n:]


def _number(value, role):
    if type(value) is int and value.bit_length() <= 512:
        return Fraction(value)
    if type(value) is float and math.isfinite(value):
        return Fraction(value)
    if type(value) is Decimal and value.is_finite():
        _, digits, exponent = value.as_tuple()
        if len(digits) <= 64 and -200 <= exponent <= 200:
            return Fraction(value)
    raise ValueError(f"{role} needs a bounded finite real number; booleans are not measurements.")


def _three(values, role):
    if type(values) not in (list, tuple) or len(values) != 3:
        raise ValueError(f"{role} must contain low, interior and high values, in that order.")
    return tuple(_number(v, role) for v in values)


@dataclass(frozen=True)
class AffineReadoutComparison:
    """Descriptive checks in readout units, not a calibration certificate."""

    reference_interior_error: Fraction
    comparison_interior_error: Fraction
    low_path_difference: Fraction
    high_path_difference: Fraction
    max_affine_path_difference: Fraction
    interior_pass: bool
    shared_affine_pass: bool


def compare_affine_readouts(standards, reference, comparison, *,
                            max_interior_error, max_path_difference):
    """Fit two endpoint lines, each with a reserved interior check.

    Each arm must measure the same three known, ordered standards in the same
    units. Caller establishes physical path matching, reference accuracy,
    repeats, drift and tolerances. Thresholds are nonnegative, inclusive,
    fixed before comparison, and expressed in response units. A bounded
    model-vs-model difference does not exclude unsampled nonlinearity.
    """
    x0, x1, x2 = _three(standards, "standards")
    r0, r1, r2 = _three(reference, "reference")
    c0, c1, c2 = _three(comparison, "comparison")
    e = _number(max_interior_error, "max_interior_error")
    d = _number(max_path_difference, "max_path_difference")
    if not x0 < x1 < x2 or e < 0 or d < 0:
        raise ValueError("Standards must strictly increase and tolerances must be nonnegative.")
    if not r0 < r2 or not c0 < c2:
        raise ValueError("Each path needs a positive endpoint response slope.")
    t = (x1 - x0) / (x2 - x0)
    er = r1 - (r0 + t * (r2 - r0))
    ec = c1 - (c0 + t * (c2 - c0))
    low, high = c0 - r0, c2 - r2
    maximum = max(abs(low), abs(high))
    interior_pass = abs(er) <= e and abs(ec) <= e
    return AffineReadoutComparison(er, ec, low, high, maximum, interior_pass,
                                  interior_pass and maximum <= d)


@dataclass(frozen=True)
class AnchorChangeDisagreement:
    """Exact finite-roster diagnostics; squared deviations are in squared units."""

    signed_bias: Fraction
    mean_absolute_disagreement: Fraction
    centered_mean_square_disagreement: Fraction


@dataclass(frozen=True)
class AnchorChangeControlAudit:
    """Agreement with two declared comparison paths, not a reliability estimate."""

    n_blocks: int
    candidate_minus_sham: AnchorChangeDisagreement
    candidate_minus_reference: AnchorChangeDisagreement
    sham_minus_reference: AnchorChangeDisagreement


def audit_anchor_change_controls(records, *, expected_blocks) -> AnchorChangeControlAudit:
    """Compare changes on the same declared 0/1 anchor blocks in common units.

    Each record is (block_id, candidate_0, candidate_1, sham_0, sham_1,
    reference_0, reference_1). Both controls must measure the matched condition
    and slots. The sham may share the candidate's acquisition errors; the
    reference is intended to have independent measurement errors. The caller
    must establish these dependencies and the separate measurement paths.
    This function checks a declared roster, not instrument or unit provenance.

    Return signed bias, average absolute disagreement, and centered mean
    squared disagreement of changes, equally weighted by block. Bias and
    centered disagreement separate common offset from block-to-block
    variability descriptively, not as physical drift or noise estimates.
    No predictions or baselines are fit, selected or corrected.
    """
    if not all(type(x) in (list, tuple) and 1 <= len(x) <= MAX_VALUES
               for x in (records, expected_blocks)):
        raise ValueError("Anchor records and frozen blocks must be bounded nonempty sequences.")

    def block_id(value):
        if (type(value) is not str or not 1 <= len(value) <= 256 or value.strip() != value
                or any(unicodedata.category(c) in ("Cc", "Cs") for c in value)):
            raise ValueError("Anchor blocks require plain, nonblank string identifiers.")
        return value

    expected = {block_id(key) for key in expected_blocks}
    if len(expected) != len(expected_blocks):
        raise ValueError("Duplicate block in the frozen anchor roster.")
    seen = set()
    sums = [[Fraction(), Fraction(), Fraction()] for _ in range(3)]
    for row in records:
        if type(row) is not tuple or len(row) != 7:
            raise ValueError("Anchor records require a block and six matched slot values.")
        key = block_id(row[0])
        if key not in expected or key in seen:
            raise ValueError("Anchor records must match the frozen block roster exactly.")
        seen.add(key)
        a0, a1, s0, s1, r0, r1 = (_number(v, "Anchor measurement") for v in row[1:])
        candidate, sham, reference = a1 - a0, s1 - s0, r1 - r0
        for totals, gap in zip(sums, (candidate - sham, candidate - reference, sham - reference)):
            totals[0] += gap
            totals[1] += abs(gap)
            totals[2] += gap * gap
    if seen != expected:
        raise ValueError("Anchor records must match the frozen block roster exactly.")

    n = len(seen)
    comparisons = tuple(AnchorChangeDisagreement(total / n, absolute / n,
                                                   squares / n - (total / n) ** 2)
                        for total, absolute, squares in sums)
    return AnchorChangeControlAudit(n, *comparisons)


@dataclass(frozen=True)
class PolarityMirrorAudit:
    """Exact descriptive even residuals on a frozen mirror-pair roster."""

    n_blocks: int
    n_pairs: int
    n_both_zero_pairs: int
    equal_block_odd_mirror_mae: Fraction
    worst_absolute_even_component: Fraction
    worst_relative_even_component: Fraction | None


def audit_polarity_mirror(records, *, expected_pairs) -> PolarityMirrorAudit:
    """Check an odd current response at positive and negative matched voltages.

    Rows are (block, V_plus, I_plus, V_minus, I_minus); expected pairs are
    (block, strictly positive voltage). Require a complete independent roster,
    exact voltage opposition and caller-established comparable units, state,
    read path, history and conditions. No interpolation or pairing is inferred.
    Odd-mirror error is abs(I_plus + I_minus); average within each block before
    equally weighting blocks. The worst absolute even component is half that
    error. Divide the error by abs(I_plus) + abs(I_minus) for the relative
    component; joint zeros have no relative denominator. These are descriptive
    mismatches, not proof of contact mechanism, independence or transfer.
    """
    _values(records)
    _values(expected_pairs)

    def pair(block, amplitude):
        if (type(block) is not str or not 1 <= len(block) <= 256 or block.strip() != block
                or any(unicodedata.category(c) in ("Cc", "Cs") for c in block)):
            raise ValueError("Polarity blocks require plain, nonblank string identifiers.")
        v = _number(amplitude, "Positive voltage")
        if v <= 0:
            raise ValueError("A polarity pair requires a strictly positive amplitude.")
        return block, v

    expected = set()
    for spec in expected_pairs:
        if type(spec) is not tuple or len(spec) != 2:
            raise ValueError("Expected polarity roster requires block/amplitude pairs.")
        key = pair(*spec)
        if key in expected:
            raise ValueError("Duplicate expected polarity pair.")
        expected.add(key)
    seen, blocks = set(), {}
    worst, relative_worst, zeroes = Fraction(), None, 0
    for row in records:
        if type(row) is not tuple or len(row) != 5:
            raise ValueError("A polarity record requires a block and both voltage/current measurements.")
        block, pos_v, pos_i, neg_v, neg_i = row
        key = pair(block, pos_v)
        if key not in expected or key in seen:
            raise ValueError("Polarity records must match the frozen pair roster exactly.")
        if _number(neg_v, "Negative voltage") != -key[1]:
            raise ValueError("Polarity voltages do not have exactly opposite signs and equal magnitudes.")
        plus, minus = _number(pos_i, "Positive-branch current"), _number(neg_i, "Negative-branch current")
        error = abs(plus + minus)
        denominator = abs(plus) + abs(minus)
        seen.add(key)
        subtotal, count = blocks.get(block, (Fraction(), 0))
        blocks[block] = subtotal + error, count + 1
        worst = max(worst, error)
        if denominator:
            relative = error / denominator
            relative_worst = relative if relative_worst is None else max(relative_worst, relative)
        else:
            zeroes += 1
    if seen != expected:
        raise ValueError("Polarity records must match the frozen pair roster exactly.")
    mae = sum((s / n for s, n in blocks.values()), Fraction()) / len(blocks)
    return PolarityMirrorAudit(len(blocks), len(seen), zeroes, mae, worst / 2, relative_worst)


_PAIRING_MAX_BINS = 128
_PAIRING_MAX_TOTAL = 1_000_000
_PAIRING_MAX_VALUE = (1 << 63) - 1


@dataclass(frozen=True)
class PairingCertificate:
    """Integer primal table and dual potentials in the fixed histogram order."""

    flow: tuple
    origin_potential: tuple
    endpoint_potential: tuple


@dataclass(frozen=True)
class PairingEnvelope:
    """Exact signed total cost extrema within one declared pairing stratum."""

    pair_count: int
    minimum: int
    maximum: int


def _pairing_vector(values, n, low, high):
    if (type(values) is not tuple or len(values) != n
            or any(type(x) is not int or not low <= x <= high for x in values)):
        raise ValueError("Expected a bounded ordered vector of exact integers.")
    return values


def certify_pairing_envelope(origin_counts, endpoint_counts, costs,
                             minimum, maximum) -> PairingEnvelope:
    """Check sharp integer extrema over all within-stratum histogram couplings.

    Supply tuples of positive bin counts and a signed integer cost matrix
    (for example doubled corrected loss minus doubled persistence loss).
    Each certificate needs one feasible integer flow and row/column duals.
    Check every mass, dual inequality, and exact equality of the two
    objectives. Any feasible integer flow can be realized by a pairing of
    repeated rows, so the certified endpoints are attained. The function
    neither computes certificates nor asserts provenance, confidence,
    admissibility of every hypothetical coupling, or the measured pairing.
    """
    if any(type(x) is not tuple or not 1 <= len(x) <= _PAIRING_MAX_BINS
           for x in (origin_counts, endpoint_counts)):
        raise ValueError("Histograms require at most 128 non-empty ordered bins each.")
    n, m = len(origin_counts), len(endpoint_counts)
    a = _pairing_vector(origin_counts, n, 1, _PAIRING_MAX_TOTAL)
    b = _pairing_vector(endpoint_counts, m, 1, _PAIRING_MAX_TOTAL)
    total = sum(a)
    if total != sum(b) or total > _PAIRING_MAX_TOTAL:
        raise ValueError("Pairing histograms must have the same bounded total mass.")
    if type(costs) is not tuple or len(costs) != n:
        raise ValueError("Pairing cost matrix has incorrect origin dimension.")
    for row in costs:
        _pairing_vector(row, m, -_PAIRING_MAX_VALUE, _PAIRING_MAX_VALUE)

    def bound(certificate, direction):
        if (type(certificate) is not PairingCertificate or type(certificate.flow) is not tuple
                or len(certificate.flow) != n):
            raise ValueError("Pairing certificate has incorrect type or flow shape.")
        u = _pairing_vector(certificate.origin_potential, n, -_PAIRING_MAX_VALUE, _PAIRING_MAX_VALUE)
        v = _pairing_vector(certificate.endpoint_potential, m, -_PAIRING_MAX_VALUE, _PAIRING_MAX_VALUE)
        endpoints, primal = [0] * m, 0
        for i, row in enumerate(certificate.flow):
            _pairing_vector(row, m, 0, total)
            if sum(row) != a[i]:
                raise ValueError("Pairing flow does not match origin masses.")
            for j, count in enumerate(row):
                endpoints[j] += count
                price = costs[i][j]
                if direction * (u[i] + v[j] - price) > 0:
                    raise ValueError("Pairing dual violates its bound at a bin pair.")
                primal += count * price
        if tuple(endpoints) != b:
            raise ValueError("Pairing flow does not match endpoint masses.")
        dual = sum(x*y for x,y in zip(a,u)) + sum(x*y for x,y in zip(b,v))
        if primal != dual:
            raise ValueError("Pairing primal and dual objectives must agree exactly.")
        return primal

    lo, hi = bound(minimum, 1), bound(maximum, -1)
    if lo > hi:
        raise ValueError("Minimum exceeds maximum pairing cost.")
    return PairingEnvelope(total, lo, hi)


@dataclass(frozen=True)
class ElapsedCaptureSupport:
    """Descriptive clock support in a prespecified half-open interval."""

    window_id: str
    reference_count: int
    comparison_count: int
    reference_span: Fraction | None
    comparison_span: Fraction | None
    shared_exact_timestamps: int
    identical_observed_grid: bool


def audit_elapsed_capture_support(reference_clock, comparison_clock, *, windows):
    """Support only, never an outcome-based selection or inferred clock alignment.

    The caller declares both monotone clocks in a common unit and known
    common origin. Each (window_id, start, stop) is predeclared, ordered and
    nonoverlapping; membership uses start <= time < stop. Counts include
    empty windows with a null span; one read has span zero. Shared readings
    count exact timestamps only, never interpolated or rounded matches.
    No claims about elapsed-time calibration, device identity or extraction
    causality follow from this description.
    """
    from bisect import bisect_left
    # Source clocks can contain an extended densely sampled trace. Bound
    # them separately from the smaller contract's measurement roster.
    max_clock_rows = 100_000

    def clock(values, role):
        if type(values) not in (tuple, list) or not 1 <= len(values) <= max_clock_rows:
            raise ValueError("Source clocks must be nonempty bounded ordered sequences.")
        times = [_number(x, role) for x in values]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Source clocks must already strictly increase without resets.")
        return times

    a = clock(reference_clock, "Reference timestamp")
    b = clock(comparison_clock, "Comparison timestamp")
    _values(windows)
    known, output, previous_stop = set(), [], None
    for item in windows:
        if type(item) is not tuple or len(item) != 3:
            raise ValueError("Clock windows require ID, start and exclusive stop.")
        key, start, stop = item
        if (type(key) is not str or not 1 <= len(key) <= 256 or key.strip() != key
                or any(unicodedata.category(c) in ("Cc", "Cs") for c in key) or key in known):
            raise ValueError("Clock windows require unique plain nonblank IDs.")
        known.add(key)
        left = _number(start, "Window start")
        right = _number(stop, "Window stop")
        if left >= right or (previous_stop is not None and left < previous_stop):
            raise ValueError("Clock windows must be ordered nonoverlapping positive intervals.")
        previous_stop = right
        i,k = bisect_left(a,left),bisect_left(a,right)
        j,l = bisect_left(b,left),bisect_left(b,right)
        m,n = k-i,l-j
        span_a = a[k-1]-a[i] if m else None
        span_b = b[l-1]-b[j] if n else None
        shared = 0
        while i < k and j < l:
            x,y = a[i],b[j]
            if x == y:
                shared += 1
            if x <= y:
                i += 1
            if y <= x:
                j += 1
        output.append(ElapsedCaptureSupport(key,m,n,span_a,span_b,shared,shared==m==n and m>0))
    return tuple(output)


@dataclass(frozen=True)
class ReadQualificationAudit:
    """Exact shadow replay on a complete roster of attempted episodes."""

    episodes: int
    followup_passes: int
    last_qualified: int
    streak_qualified: int
    last_qualified_followup_passes: int
    streak_qualified_followup_passes: int
    roster_rate: Fraction
    last_selected_rate: Fraction | None
    streak_selected_rate: Fraction | None


def audit_read_qualification(records, *, expected_episodes, streak, band):
    """Compare last-read and full-streak selection without changing exposure.

    Each record is (episode, prequalification_readings, followup_reading).
    The readings must contain exactly the prespecified number of consecutive
    observations before a fresh, common follow-up slot measured on every
    attempted episode. All values have common units; the same closed band
    defines in-band qualification and follow-up. The independent roster,
    acquisition chronology and identical exposures are caller obligations.
    The selected rates are descriptive, not causal policy effects.
    """
    _values(records)
    _values(expected_episodes)
    if type(streak) is not int or not 2 <= streak <= 256:
        raise ValueError("A qualification streak needs 2 to 256 predeclared reads.")
    if type(band) not in (tuple, list) or len(band) != 2:
        raise ValueError("Qualification band requires two closed endpoints.")
    lo, hi = (_number(x, "Band endpoint") for x in band)
    if lo > hi:
        raise ValueError("Qualification band endpoints must be ordered.")

    def plain_id(key):
        if (type(key) is not str or not 1 <= len(key) <= 256 or key.strip() != key
                or any(unicodedata.category(c) in ("Cc", "Cs") for c in key)):
            raise ValueError("Qualifications require plain nonblank episode IDs.")
        return key

    expected = {plain_id(key) for key in expected_episodes}
    if len(expected) != len(expected_episodes):
        raise ValueError("Duplicate episode in the frozen qualification roster.")
    seen = set()
    followups = last = full = last_pass = full_pass = 0
    for row in records:
        if type(row) is not tuple or len(row) != 3:
            raise ValueError("Qualification requires each episode's prior reads and fresh follow-up.")
        key = plain_id(row[0])
        if key not in expected or key in seen:
            raise ValueError("Qualification records must match the frozen episode roster exactly.")
        seen.add(key)
        prior = row[1]
        if type(prior) not in (list, tuple) or len(prior) != streak:
            raise ValueError("Each episode needs the full prespecified prequalification streak.")
        eligible = tuple(lo <= _number(x, "Prequalification read") <= hi for x in prior)
        success = lo <= _number(row[2], "Fresh follow-up read") <= hi
        followups += success
        if eligible[-1]:
            last += 1
            last_pass += success
        if all(eligible):
            full += 1
            full_pass += success
    if seen != expected:
        raise ValueError("Qualification records must match the frozen episode roster exactly.")
    n = len(expected)
    return ReadQualificationAudit(n, followups, last, full, last_pass, full_pass,
                                  Fraction(followups, n),
                                  Fraction(last_pass, last) if last else None,
                                  Fraction(full_pass, full) if full else None)


@dataclass(frozen=True)
class PairedIntervalScoreBounds:
    """Sharp lower and upper score differences on one common target cell."""

    minimum: Fraction
    maximum: Fraction

    @property
    def strict_preference(self) -> str | None:
        if self.maximum < 0:
            return "first"
        if self.minimum > 0:
            return "second"
        return None


def bound_paired_interval_score_difference(first, second, *, target_cell, alpha):
    """Bound interval score(first) minus interval score(second) exactly.

    All three pairs are closed, ordered endpoints in the same units. The
    target cell must independently certify where the ONE common true value
    could be. A negative difference favors the first forecast. An endpoint
    of either prediction interval can be an interior extremum of the score
    difference, even when the two target-cell corners agree on preference.
    No rounding, imputation, physical certification or inference is done.
    Combining cells with joint restrictions requires a separate method.
    """
    def interval(pair, name):
        if type(pair) not in (list, tuple) or len(pair) != 2:
            raise ValueError(f"{name} requires two endpoints.")
        lo, hi = (_number(v, name) for v in pair)
        if lo > hi:
            raise ValueError(f"{name} endpoints must be ordered.")
        return lo, hi

    a, b = interval(first, "First prediction"), interval(second, "Second prediction")
    lo, hi = interval(target_cell, "Target cell")
    level = _number(alpha, "Interval score alpha")
    if not 0 < level < 1:
        raise ValueError("Interval score alpha must be strictly between zero and one.")
    scale = 2 / level

    def score(at, limits):
        left, right = limits
        return right - left + scale * max(left - at, at - right, 0)

    probes = {lo, hi, *(x for x in a + b if lo <= x <= hi)}
    deltas = tuple(score(x, a) - score(x, b) for x in probes)
    return PairedIntervalScoreBounds(min(deltas), max(deltas))
