"""Validate this invented fixture's supplied mapping; not a grouping engine."""
import argparse
import csv
import io
from pathlib import Path

FIELDS = ('specimen_id', 'job_id', 'power', 'speed', 'hatch', 'layer', 'response', 'post_outcome_note')
SETTING = ('power', 'speed', 'hatch', 'layer')
# Manually declared fixture oracle; no graph traversal or inferred types.
EXPECTED = dict(zip(('s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08'),
                    ('C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C3', 'C4')))
HERE = Path(__file__).resolve().parent


def read_table(path, fields):
    with Path(path).open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError('example input exceeds 64 KiB')
    if b'\x00' in raw:
        raise ValueError('NUL is not supported')
    rows = list(csv.reader(io.StringIO(raw.decode('utf-8'), newline=''), strict=True))
    if not rows or tuple(rows[0]) != fields:
        raise ValueError('exact header required')
    if any(len(row) != len(fields) or any(not value for value in row) for row in rows[1:]):
        raise ValueError('ragged or empty cell')
    return [dict(zip(fields, row)) for row in rows[1:]]


def validate_and_join(specimens, mapping):
    """Require unique exact IDs and zero unmatched rows on both sides."""
    def index(rows):
        result = {}
        for row in rows:
            key = row['specimen_id']
            if key in result:
                raise ValueError('duplicate specimen ID')
            result[key] = row
        return result
    source, groups = index(specimens), index(mapping)
    if set(source) != set(groups):
        raise ValueError('unmatched specimen IDs; never use an inner intersection')
    if set(source) != set(EXPECTED):
        raise ValueError('this example requires its exact eight-specimen roster')
    if {key: row['component_id'] for key, row in groups.items()} != EXPECTED:
        raise ValueError('mapping differs from the manually supplied fixture oracle')
    # Check direct dependencies; transitive expectations are supplied, not computed.
    for left in specimens:
        for right in specimens:
            linked = (left['job_id'] == right['job_id'] or
                      tuple(left[f] for f in SETTING) == tuple(right[f] for f in SETTING))
            if linked and groups[left['specimen_id']]['component_id'] != groups[right['specimen_id']]['component_id']:
                raise ValueError('a declared dependency crosses components')
    return [dict(row, component_id=groups[row['specimen_id']]['component_id']) for row in specimens]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional NEW joined CSV; existing paths are never overwritten')
    args = parser.parse_args(argv)
    joined = validate_and_join(read_table(HERE / 'specimens.csv', FIELDS),
                               read_table(HERE / 'expected_components.csv', ('specimen_id', 'component_id')))
    if args.output is not None:
        with args.output.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=(*FIELDS, 'component_id'))
            writer.writeheader()
            writer.writerows(joined)
    print('PASS: 8 unique specimens; 4 supplied components; unmatched source=0, mapping=0; no fits')


if __name__ == '__main__':
    main()
