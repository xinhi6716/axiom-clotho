"""Invented causal-design software checks, never empirical effect validation."""
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1] / 'examples/causal-design-rehearsal'
spec = importlib.util.spec_from_file_location('causal_design_rehearsal', HERE / 'rehearsal.py')
rehearsal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rehearsal)


def orthogonal_draws():
    # Exactly orthogonal +/-1 arithmetic test, not draws from the Gaussian SCM.
    index = np.arange(32)
    return {key: np.where((index >> bit) & 1, 1., -1.) for key, bit in [('C', 2), ('UA', 0), ('UM', 1), ('UY', 3), ('U', 4)]}


def records(scenario='measured_context', draws=None):
    return rehearsal.observation_rows(rehearsal.structural_values(orthogonal_draws() if draws is None else draws, scenario), 8, 4)


def estimates(rows):
    return {name: rehearsal.estimate_observational(rows, estimand=estimand, adjustment=adjustment) for name, estimand, adjustment in rehearsal.PLANS}


class CausalDesignTests(unittest.TestCase):
    def test_independent_orthogonal_algebra_for_base_and_hidden_cases(self):
        for scenario, expected in [('measured_context', (9.5, 7., 1.)), ('omitted_device_confounder', (28 / 3, 8., 2.))]:
            result = estimates(records(scenario))
            for (name, _, _), target in zip(rehearsal.PLANS, expected):
                self.assertAlmostEqual(result[name]['setting_coefficient'], target, places=12)
                self.assertFalse(result[name]['causal_identification_checked_by_ols'])
        self.assertIn('does not identify', rehearsal.scenario_contract('omitted_device_confounder')['identification_note'])

    def test_independent_residualization_matches_seeded_adjustment(self):
        draws = rehearsal.draw_exogenous(8, 4, 17)
        rows = records(draws=draws)
        c = np.array([r['batch_context'] for r in rows]); c -= c.mean()
        a = np.array([r['process_setting'] for r in rows]); a -= a.mean()
        y = np.array([r['readout'] for r in rows]); y -= y.mean()
        ar = a - c * ((c @ a) / (c @ c)); yr = y - c * ((c @ y) / (c @ c))
        oracle = (ar @ yr) / (ar @ ar)
        actual = rehearsal.estimate_observational(rows, estimand='total_effect', adjustment=('batch_context',))
        self.assertAlmostEqual(actual['setting_coefficient'], oracle, places=12)

    def test_every_paired_total_and_controlled_direct_contrast(self):
        exogenous = orthogonal_draws()
        for scenario in rehearsal.SCENARIOS:
            low = rehearsal.structural_values(exogenous, scenario, do_setting=0)
            high = rehearsal.structural_values(exogenous, scenario, do_setting=1)
            np.testing.assert_array_equal(high['intermediary'] - low['intermediary'], np.full(32, 2.))
            np.testing.assert_array_equal(high['readout'] - low['readout'], np.full(32, 7.))
            dl = rehearsal.structural_values(exogenous, scenario, do_setting=0, do_intermediary=0)
            dh = rehearsal.structural_values(exogenous, scenario, do_setting=1, do_intermediary=0)
            np.testing.assert_array_equal(dh['readout'] - dl['readout'], np.ones(32))

    def test_intervention_surgery_ignores_assignment_noise_not_latent_context(self):
        first = orthogonal_draws(); changed = {k: v.copy() for k, v in first.items()}
        changed['UA'] += 100
        for scenario in rehearsal.SCENARIOS:
            for intervention in (0, 1):
                a = rehearsal.structural_values(first, scenario, do_setting=intervention)
                b = rehearsal.structural_values(changed, scenario, do_setting=intervention)
                for key in a: np.testing.assert_array_equal(a[key], b[key])
        self.assertFalse(np.array_equal(rehearsal.structural_values(first, 'measured_context')['process_setting'], rehearsal.structural_values(changed, 'measured_context')['process_setting']))

    def test_total_effect_request_rejects_post_setting_oracle_and_wrong_roles(self):
        rows = records()
        for adjustment in [(), ('intermediary',), ('batch_context', 'intermediary'), ('batch_id',), ['batch_context']]:
            with patch.object(np.linalg, 'lstsq') as fit:
                with self.assertRaises(rehearsal.InputError):
                    rehearsal.estimate_observational(rows, estimand='total_effect', adjustment=adjustment)
                fit.assert_not_called()
        for column in ['U', 'oracle_readout_do_1', 'UA']:
            changed = copy.deepcopy(rows); changed[0][column] = 3.
            with self.assertRaises(rehearsal.InputError):
                rehearsal.estimate_observational(changed, estimand='total_effect', adjustment=('batch_context',))

    def test_deterministic_assignment_refuses_pseudoinverse(self):
        rows = records('deterministic_assignment')
        self.assertAlmostEqual(estimates(rows)['naive_association']['setting_coefficient'], 12.)
        for estimand, adjustment in [('total_effect', ('batch_context',)), ('controlled_direct_effect', ('batch_context', 'intermediary'))]:
            with patch.object(np.linalg, 'lstsq') as fit:
                result = rehearsal.estimate_observational(rows, estimand=estimand, adjustment=adjustment)
                fit.assert_not_called()
            self.assertEqual(result['status'], 'NUMERICAL_RANK_DEFICIENT')
            self.assertIsNone(result['setting_coefficient'])
            self.assertIsNone(result['coefficients'])
        self.assertFalse(rehearsal.scenario_contract('deterministic_assignment')['assignment_noise_present'])

    def test_near_collinearity_is_numerical_not_claimed_zero_variation(self):
        draws = orthogonal_draws(); draws['UA'] *= 1e-13
        rows = records(draws=draws)
        self.assertTrue(any(r['process_setting'] != r['batch_context'] for r in rows))
        result = rehearsal.estimate_observational(rows, estimand='total_effect', adjustment=('batch_context',))
        self.assertEqual(result['status'], 'NUMERICAL_RANK_DEFICIENT')
        self.assertEqual(result['relative_rank_tolerance'], 1e-12)
        self.assertNotIn('assignment_noise_present', result)
        self.assertGreater(result['singular_values'][-1], 0)

    def test_estimation_isolated_from_intervention_output_changes(self):
        original = rehearsal.structural_values
        files, before = rehearsal.rehearse(batches=8, devices_per_batch=4, seed=17)
        def shifted(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get('do_setting') is not None:
                result['readout'] += 100
            return result
        with patch.object(rehearsal, 'structural_values', side_effect=shifted):
            changed_files, after = rehearsal.rehearse(batches=8, devices_per_batch=4, seed=17)
        for name in rehearsal.SCENARIOS:
            self.assertEqual(before['scenarios'][name]['observational_estimates'], after['scenarios'][name]['observational_estimates'])
            self.assertEqual(files[name + '.observations.csv'], changed_files[name + '.observations.csv'])
            self.assertNotEqual(files[name + '.paired_interventions.csv'], changed_files[name + '.paired_interventions.csv'])

    def test_freezing_observed_intermediary_cannot_pass_total_oracle(self):
        original = rehearsal.structural_values
        def freeze(exogenous, scenario, **kwargs):
            if kwargs.get('do_setting') is not None and kwargs.get('do_intermediary') is None:
                # Wrong total intervention: leave M at observed values.
                observed = original(exogenous, scenario)
                current = original(exogenous, scenario, **kwargs)
                current['readout'] += 3 * (observed['intermediary'] - current['intermediary'])
                current['intermediary'] = observed['intermediary']
                return current
            return original(exogenous, scenario, **kwargs)
        with patch.object(rehearsal, 'structural_values', side_effect=freeze):
            with self.assertRaises(rehearsal.InputError):
                rehearsal.rehearse(batches=8, devices_per_batch=4, seed=17)

    def test_reproducibility_seed_variation_complete_scenarios_and_no_selection(self):
        files, report = rehearsal.rehearse(batches=8, devices_per_batch=4, seed=17)
        self.assertEqual(files, rehearsal.rehearse(batches=8, devices_per_batch=4, seed=17)[0])
        self.assertNotEqual(files, rehearsal.rehearse(batches=8, devices_per_batch=4, seed=18)[0])
        self.assertEqual(tuple(report['scenarios']), rehearsal.SCENARIOS)
        self.assertEqual(len(files), 9)
        for seed in (0, 7, 42):
            _, case = rehearsal.rehearse(batches=2, devices_per_batch=2, seed=seed)
            self.assertEqual(case['configuration']['seed'], seed)
            for scenario in rehearsal.SCENARIOS:
                self.assertEqual(set(case['scenarios'][scenario]['observational_estimates']), {p[0] for p in rehearsal.PLANS})
        # A finite Gaussian result need not equal the base population target.
        self.assertNotEqual(report['scenarios']['measured_context']['observational_estimates']['context_adjusted']['setting_coefficient'], 7.)

    def test_strict_counts_and_observation_validation(self):
        for kwargs in [{'batches': True}, {'batches': 1}, {'batches': 129}, {'devices_per_batch': 1}, {'devices_per_batch': 65}, {'seed': True}, {'seed': -1}, {'seed': 2**32}]:
            with self.assertRaises(rehearsal.InputError): rehearsal.rehearse(**kwargs)
        for name, bad in [('process_setting', float('nan')), ('readout', 1e7), ('readout', True), ('device_id', ' leading'), ('batch_id', 'bad\nvalue')]:
            value = copy.deepcopy(records()); value[0][name] = bad
            with self.assertRaises(rehearsal.InputError): estimates(value)
        inconsistent = records(); inconsistent[1]['batch_context'] += 1
        with self.assertRaises(rehearsal.InputError): estimates(inconsistent)
        duplicate = records(); duplicate[1]['device_id'] = duplicate[0]['device_id']
        with self.assertRaises(rehearsal.InputError): estimates(duplicate)

    def test_actual_cli_private_outputs_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'new'
            command = [sys.executable, str(HERE / 'rehearsal.py'), '--output-dir', str(path), '--accept-simulation-only', '--batches', '8', '--devices-per-batch', '4']
            result = subprocess.run(command, capture_output=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((path / 'COMPLETE.json').read_bytes())
            self.assertEqual(receipt['source_sha256'], hashlib.sha256((HERE / 'rehearsal.py').read_bytes()).hexdigest())
            self.assertFalse(receipt['empirical_effect_identified'])
            for name, digest in receipt['output_files_sha256'].items():
                self.assertEqual(hashlib.sha256((path / name).read_bytes()).hexdigest(), digest)
            if os.name == 'posix':
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
                self.assertTrue(all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in path.iterdir()))
            before = {p.name: p.read_bytes() for p in path.iterdir()}
            again = subprocess.run(command, capture_output=True)
            self.assertEqual(again.returncode, 2)
            self.assertNotIn(str(path).encode(), again.stderr)
            self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})

    def test_existing_and_dangling_output_reject_before_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'dir').mkdir(); (root / 'file').write_bytes(b'keep')
            (root / 'link').symlink_to(root / 'absent', target_is_directory=True)
            for name in ('dir', 'file', 'link'):
                with patch.object(rehearsal, 'rehearse') as compute:
                    with self.assertRaises(rehearsal.InputError): rehearsal.run(root / name, simulation_declared=True)
                    compute.assert_not_called()
            for flag in (False, None, 1, 'true'):
                with self.assertRaises(rehearsal.InputError): rehearsal.run(root / 'new', simulation_declared=flag)

    def test_partial_write_never_has_complete_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'new'; original = rehearsal.write_new
            def fail_report(target, blob):
                if target.name == 'REPORT.json': raise OSError('injected write failure')
                return original(target, blob)
            with patch.object(rehearsal, 'write_new', side_effect=fail_report):
                with self.assertRaises(rehearsal.InputError): rehearsal.run(path, batches=8, devices_per_batch=4, simulation_declared=True)
            self.assertTrue(path.exists())
            self.assertFalse((path / 'COMPLETE.json').exists())

    def test_serialization_bounds_and_actual_maximum_count(self):
        with patch.object(rehearsal, 'MAX_FILE_BYTES', 10):
            with self.assertRaises(rehearsal.InputError): rehearsal.json_bytes({'long': 'x' * 20})
        files, report = rehearsal.rehearse(batches=128, devices_per_batch=64, seed=7)
        self.assertEqual(report['configuration']['devices'], 8192)
        self.assertTrue(all(len(data) <= rehearsal.MAX_FILE_BYTES for data in files.values()))
        parsed = list(csv.DictReader(io.StringIO(files['measured_context.observations.csv'].decode())))
        self.assertEqual(len(parsed), 8192)
        self.assertNotIn('U', parsed[0])
        self.assertNotIn('oracle', parsed[0])


if __name__ == '__main__':
    unittest.main()
