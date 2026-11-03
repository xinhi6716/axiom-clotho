"""Access the exact archived generator. Import is no-fit and does not draw RNG."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('optional_study_r2_archive', HERE/'archive/driver_r2_used.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def generate_case(scenario_id, root, *, rng=None):
    """Generate a registered cohort; rng injection supports draw-free fake tests."""
    cases = json.loads((HERE/'artifacts/cases.json').read_text())
    match = [case for case in cases if case['scenario_id'] == scenario_id and case['root'] == root]
    if len(match) != 1:
        raise ValueError('Case is outside the frozen protocol.')
    return archive.generate_case(scenario_id, root, match[0]['fold_ids'], rng=rng)
