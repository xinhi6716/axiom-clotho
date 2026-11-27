"""Audit the signed ICCAP DC ID-VG block subset without extracting rows."""
import argparse
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path


def number(token):
    value = D(token)
    if not value.is_finite():
        raise ValueError("non-finite input")
    return value


def audit(raw, sha256):
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise ValueError("checksum mismatch")
    fixed = rows = None
    sweeps = {}
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line == "BEGIN_DB":
            if fixed is not None:
                raise ValueError("nested block")
            fixed, rows, declared = {}, [], False
        elif line == "END_DB":
            if fixed is None or set(fixed) != {"vd", "vb", "vs"} or not declared or len(rows) < 2:
                raise ValueError("incomplete block")
            key = tuple(fixed[k] for k in ("vd", "vb", "vs"))
            if key in sweeps:
                raise ValueError("duplicate signed sweep key")
            sweeps[key] = tuple(rows)
            fixed = rows = None
        elif fixed is not None and line:
            if line.startswith("ICCAP_VAR "):
                _, k, v = line.split()
                if declared or k in fixed:
                    raise ValueError("misplaced or duplicate fixed variable")
                fixed[k] = number(v)
            elif line.split() == ["#vg", "id", "ig", "ib", "is"] and not declared:
                declared = True
            else:
                row = tuple(map(number, line.split()))
                if not declared or len(row) != 5:
                    raise ValueError("unexpected data row")
                rows.append(row)
    if fixed is not None or not sweeps:
        raise ValueError("open or absent data block")
    axes = {tuple(row[0] for row in rows) for rows in sweeps.values()}
    if len(axes) != 1:
        raise ValueError("different native gate axes")
    axis, = axes
    steps = {b - a for a, b in zip(axis, axis[1:])}
    step, = steps
    if not step:
        raise ValueError("repeated gate value")
    body_zero = sum(vb == 0 for _, vb, _ in sweeps)
    return {
        "sha256": digest,
        "blocks": len(sweeps),
        "data_rows": sum(map(len, sweeps.values())),
        "body_zero_blocks": body_zero,
        "body_nonzero_blocks": len(sweeps) - body_zero,
        "gate_endpoints": list(map(str, (axis[0], axis[-1]))),
        "native_gate_step": str(step),
        "vd_levels": list(map(str, sorted({k[0] for k in sweeps}))),
        "vb_levels": list(map(str, sorted({k[1] for k in sweeps}))),
        "drain_current_sign_at_last_gate": sorted({
            int(rows[-1][1].compare(D(0))) for rows in sweeps.values()
        }),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.input.read_bytes(), args.sha256), indent=2))
