"""Offline consistency check for the maintained Usage reference version tables.

Not a natural-language linter, Git ancestry check, or installed-wheel capability test.
"""
import argparse
import json
from pathlib import Path
import re
import sys

START = '<!-- mlcalc-version-scope:start -->'
END = '<!-- mlcalc-version-scope:end -->'
FIELDS = {'documentation_scope', 'frozen_tag', 'frozen_source', 'fixes_snapshot'}
SHA = re.compile(r'[0-9a-f]{40}')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def load_contract(path):
    data = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)
    if not isinstance(data, dict) or set(data) != {'schema', 'scope', 'features'}:
        raise ValueError('contract requires schema, scope, features')
    if type(data['schema']) is not int or data['schema'] != 1:
        raise ValueError('unsupported contract schema')
    scope = data['scope']
    if not isinstance(scope, dict) or set(scope) != FIELDS:
        raise ValueError('invalid scope fields')
    if scope['documentation_scope'] != 'current-source':
        raise ValueError('reference must describe current-source')
    if not isinstance(scope['frozen_tag'], str) or not re.fullmatch(r'v\d+\.\d+\.\d+(?:a\d+)?', scope['frozen_tag']):
        raise ValueError('invalid frozen tag')
    for key in ('frozen_source', 'fixes_snapshot'):
        if not isinstance(scope[key], str) or not SHA.fullmatch(scope[key]):
            raise ValueError('invalid revision: ' + key)
    features = data['features']
    if not isinstance(features, dict) or not features:
        raise ValueError('features must be a nonempty object')
    for key, value in features.items():
        if not re.fullmatch(r'[a-z][a-z0-9-]*', key):
            raise ValueError('invalid feature ID')
        if not isinstance(value, dict) or set(value) != {'frozen', 'source'}:
            raise ValueError('feature requires frozen and source states')
        if any(state not in ('present', 'absent') for state in value.values()):
            raise ValueError('invalid capability state')
    return data


def cell(value):
    # Markdown presentation changes do not change field identity.
    return value.strip().strip('`').strip()


def parse_tables(block):
    tables, rows = [], []
    for line in block.splitlines() + ['']:
        if not line.strip():
            if rows:
                tables.append(rows)
                rows = []
            continue
        if not line.strip().startswith('|') or not line.strip().endswith('|'):
            raise ValueError('version block accepts only Markdown tables')
        rows.append([cell(part) for part in line.strip()[1:-1].split('|')])
    if len(tables) != 2:
        raise ValueError('expected scope and capability tables')
    output = []
    for table, header in zip(tables, (['Field', 'Value'], ['Capability', 'Frozen download', 'Reviewed source'])):
        if len(table) < 3 or table[0] != header or len(table[1]) != len(header):
            raise ValueError('invalid table header')
        if not all(re.fullmatch(r':?-{3,}:?', c) for c in table[1]):
            raise ValueError('invalid Markdown separator')
        mapping = {}
        for row in table[2:]:
            if len(row) != len(header) or row[0] in mapping:
                raise ValueError('duplicate or malformed version row')
            mapping[row[0]] = row[1:]
        output.append(mapping)
    return output


def without_fenced_code(text):
    """Ignore Markdown examples, not their content, when locating live status."""
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence is None and match:
            fence = (match[1][0], len(match[1]))
            continue
        if fence is not None:
            if match and match[1][0] == fence[0] and len(match[1]) >= fence[1] and not match[2].strip():
                fence = None
            continue
        lines.append(line)
    return '\n'.join(lines)


def check_text(text, contract):
    text = without_fenced_code(text)
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError('missing or duplicate version block markers')
    before, tail = text.split(START)
    block, after = tail.split(END)
    if END in before or START in after:
        raise ValueError('reversed version markers')
    # This is only the introductory status, not historical sections later on.
    if re.search(r'^\s*\*\*Status:\*\*', before, flags=re.MULTILINE):
        raise ValueError('competing legacy Status paragraph before version block')
    scope, features = parse_tables(block)
    expected_scope = {k: [v] for k, v in contract['scope'].items()}
    expected_features = {k: [v['frozen'], v['source']] for k, v in contract['features'].items()}
    if scope != expected_scope:
        raise ValueError('version scope differs from reviewed contract')
    if features != expected_features:
        raise ValueError('capability ownership differs from reviewed contract')


def check(root):
    contract = load_contract(root / 'tools/doc_version_contract.json')
    check_text((root / 'docs/USAGE_REFERENCE.md').read_text(encoding='utf-8'), contract)
    return {'status': 'pass', 'coverage': 'Usage reference marked version tables and preceding legacy Status paragraph only',
            'capabilities_checked': len(contract['features']), 'network_access': False,
            'runtime_capabilities_verified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='trusted source checkout root')
    args = parser.parse_args(argv)
    try:
        result = check(args.root)
    except (OSError, ValueError, TypeError) as exc:
        print('version documentation check failed: ' + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
