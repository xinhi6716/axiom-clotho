"""Explicit, target-independent domain-held-out inner selection (opt-in)."""
import hashlib
import json
import numpy as np
import pandas as pd
from .data import prepare_data, validate_groups

GROUP = "group_kfold"
DOMAIN = "leave_one_domain_out"
LIMITATION = ("Inner domains are user-declared, not discovered or scientifically verified. "
              "Selection retains the declared row/group weighting, not equal-domain weighting. "
              "Outer evaluation still holds out groups, not necessarily domains; this is not "
              "a domain-transfer, causal or extrapolation guarantee.")


def validate_strategy(strategy, inner_splits, domains):
    if strategy not in (GROUP, DOMAIN):
        raise ValueError("inner_strategy must be group_kfold or leave_one_domain_out.")
    if strategy == GROUP and domains is not None:
        raise ValueError("domains require explicit leave_one_domain_out strategy.")
    if strategy == DOMAIN and (domains is None or inner_splits is not None):
        raise ValueError("leave_one_domain_out requires declared domains and inner_splits=None.")


def domain_vector(domains, groups, n_rows):
    # Unlike group compatibility normalization, domain identity is exact strings.
    if np.ma.isMaskedArray(domains):
        raise ValueError("Masked domain metadata is unsupported.")
    values = np.asarray(domains, dtype=object)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError("domains must be a positional one-dimensional aligned vector.")
    if any(not isinstance(v, str) or not v or len(v) > 128 or v != v.strip() or
           any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in v) for v in values):
        raise ValueError("Domains must be nonempty unpadded exact strings without controls.")
    groups = validate_groups(groups, n_rows)
    mapping = {}
    for group, domain in zip(groups, values):
        if group in mapping and mapping[group] != domain:
            raise ValueError("Each dependency group must be nested within one domain; crossed groups are unsupported.")
        mapping[group] = domain
    if len(set(values)) < 2:
        raise ValueError("Domain-held-out selection requires at least two training domains.")
    return values.copy()


def domain_splits(domains):
    return tuple((np.flatnonzero(domains != d), np.flatnonzero(domains == d))
                 for d in sorted(set(domains)))


def domain_audit(train, test, domains):
    a, b = sorted(set(domains[train])), sorted(set(domains[test]))
    if set(a) & set(b) or len(b) != 1:
        raise ValueError("Invalid domain-held-out split.")
    return dict(train_domains=a, test_domains=b, train_domain_count=len(a),
                test_domain_count=len(b), domain_overlap_count=0)


def domain_metadata(domains, groups):
    mapping = dict(sorted(set(zip(groups, domains))))
    encoded = json.dumps(list(domains), ensure_ascii=False, separators=(",", ":")).encode()
    return dict(domain_values_sha256=hashlib.sha256(encoded).hexdigest(),
                group_domains=mapping, n_domains=len(set(domains)))


def prepare_domain_data(frame, target, group, features, domain, strategy, inner_splits):
    validate_strategy(strategy, inner_splits, domain)
    if strategy == GROUP:
        return (*prepare_data(frame, target, group, features), None)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Input must be a DataFrame.")
    if not isinstance(domain, str) or domain in (target, group) or domain not in frame.columns:
        raise ValueError("domain must name a separate declared metadata column.")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique.")
    if features is not None and domain in features:
        raise ValueError("Domain metadata cannot be a predictor.")
    X, y, groups = prepare_data(frame.drop(columns=[domain]), target, group, features)
    return X, y, groups, domain_vector(frame[domain], groups, len(X))
