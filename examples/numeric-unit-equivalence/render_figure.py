"""Deterministic editable SVG from reviewed public unit-equivalence summary."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--summary', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--source-ref', required=True)
a = p.parse_args()
if a.out.suffix.lower() != '.svg':
    p.error('--out must name an SVG file')
mapping_path = a.out.with_suffix('.mapping.json')
if a.out.exists() or mapping_path.exists():
    p.error('output SVG or mapping already exists; choose a new output path')
d = json.loads(a.summary.read_text())
rows = []
elements = []

def text(x, y, s, size=20, weight='400', fill='#223d35'):
    elements.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(str(s))}</text>')

def line(x1, y1, x2, y2, color='#d5dfd9'):
    elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}"/>')
text(32, 37, 'SYNTHETIC · SOFTWARE LEARNING ONLY', 17, '700')
text(32, 78, 'Same inputs. Different units.', 30, '700')
text(32, 111, 'A versus nA · no real-device benchmark', 20)
text(32, 143, 'Group MAE · arbitrary response units · lower is better', 18)
models = [('dummy', 'Dummy mean'), ('ridge', 'Ridge · alpha = 1'), ('extra_trees', 'ExtraTrees · 64 trees, depth 6')]
for panel, (arm, title) in enumerate([('raw', '01  Raw-model ablation'), ('calculator_scaled', '02  Calculator train-only scaling')]):
    top = 187 + panel * 370
    text(32, top, title, 24, '700')
    for i, (key, label) in enumerate(models):
        y = top + 37 + i * 94
        text(32, y, label, 20, '600')
        for j, unit in enumerate(['A', 'nA']):
            v = d['scores'][f'{key}__{arm}__{unit}']['group_mae']
            if not math.isfinite(v) or not 0 <= v <= 0.8:
                raise ValueError("MAE outside the frozen 0–0.8 axis; revise the design, never clip values.")
            by = y + 10 + j * 24
            width = v / 0.8 * 400
            text(48, by + 15, unit, 18)
            fill = '#2f6b57' if unit == 'A' else '#fffdf6'
            elements.append(f'<rect x="100" y="{by}" width="{width:.8f}" height="17" fill="{fill}" stroke="#2f6b57"/>')
            text(108 + width, by + 15, f'{v:.4f}', 18)
            rows.append({'key': f'{key}__{arm}__{unit}', 'value': v, 'label': f'{v:.4f}', 'x': 100, 'width': width, 'max_axis': 0.8})
    axis = top + 318
    line(100, axis, 500, axis, '#789186')
    for t in [0, 0.2, 0.4, 0.6, 0.8]:
        x = 100 + t / 0.8 * 400
        line(x, axis, x, axis + 5, '#789186')
        text(x - 10, axis + 26, f'{t:g}', 18)
text(32, 931, f"{d['fit_count']} fixed fits · {len(d['split']['test_groups'])} artificial test groups · seed {d['seed']}", 18)
text(32, 962, 'Raw Ridge: fixed alpha also changes regularization.', 18)
text(32, 989, 'This is not a pure floating-point precision test.', 18)
scaled_gaps = [d['unit_pairs'][f'{m}__calculator_scaled']['max_prediction_gap'] for m, _ in models]
threshold = d['prediction_equivalence_threshold']
if any(not math.isfinite(gap) or gap < 0 for gap in scaled_gaps):
    raise ValueError('Every scaled prediction gap must be finite and nonnegative')
if not math.isfinite(threshold) or threshold <= 0:
    raise ValueError('Prediction tolerance must be finite and positive')
max_gap = max(scaled_gaps)
text(32, 1023, 'Scaled prediction gaps ' + ('meet' if max_gap <= d['prediction_equivalence_threshold'] else 'exceed') + ' the fixed tolerance.', 18, '600')
text(32, 1050, f"Max gap {max_gap:.2e}; tolerance {d['prediction_equivalence_threshold']:.2e}.", 18)
versions = d['versions']
text(32, 1086, f"Python {versions['python'].split()[0]} · NumPy {versions['numpy']} · sklearn {versions['sklearn']}", 16)
text(32, 1111, f'Source {a.source_ref[:12]} · reproduce with run.py', 16)
svg = '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="1140" viewBox="0 0 640 1140" role="img" aria-labelledby="title desc">\n<title id="title">Synthetic A versus nA unit-equivalence check</title>\n<desc id="desc">Paired group mean absolute errors for Dummy, Ridge and ExtraTrees, without scaling and with the existing Calculator train-only scaling. Prediction gaps are compared with a training-derived tolerance below the plots. Raw Ridge also changes regularization under fixed alpha. This is an artificial software fixture, not a real-device accuracy claim.</desc>\n<rect width="640" height="1140" fill="#fffdf6"/>\n<g font-family="Arial, Helvetica, sans-serif">\n' + '\n'.join(elements) + '\n</g>\n</svg>\n'
a.out.parent.mkdir(parents=True, exist_ok=True)
with a.out.open('x', encoding='utf-8') as output:
    output.write(svg)
mapping = {'summary_sha256': hashlib.sha256(a.summary.read_bytes()).hexdigest(), 'source_ref': a.source_ref, 'rows': rows, 'max_scaled_gap': max_gap, 'threshold': d['prediction_equivalence_threshold'], 'svg_sha256': hashlib.sha256(svg.encode()).hexdigest()}
with mapping_path.open('x', encoding='utf-8') as output:
    output.write(json.dumps(mapping, indent=2, allow_nan=False) + '\n')
