"""Exact transport certificates on tiny matched-mass tables, not physical pairing."""
from dataclasses import replace
from itertools import permutations
import unittest

from groupbench import quantity_contract as p


class PairingEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.l, self.r = (2, 1), (1, 2)
        self.cost = ((0, 4), (3, 1))
        self.low = p.PairingCertificate(((1, 1), (0, 1)), (0, -3), (0, 4))
        self.high = p.PairingCertificate(((0, 2), (1, 0)), (0, 3), (0, 4))

    def test_matches_all_id_assignments_with_repeated_bins(self):
        costs = [sum(self.cost[i][j] for i, j in zip((0, 0, 1), ys))
                 for ys in permutations((0, 1, 1))]
        found = p.certify_pairing_envelope(self.l, self.r, self.cost, self.low, self.high)
        self.assertEqual(found, p.PairingEnvelope(3, min(costs), max(costs)))
        offset = -3
        changed = tuple(tuple(c + offset for c in r) for r in self.cost)
        low = replace(self.low, origin_potential=tuple(v+offset for v in self.low.origin_potential))
        high = replace(self.high, origin_potential=tuple(v+offset for v in self.high.origin_potential))
        self.assertEqual(p.certify_pairing_envelope(self.l, self.r, changed, low, high),
                         p.PairingEnvelope(3, -4, 2))

    def test_invariant_to_constraint_potential_gauge(self):
        low = replace(self.low, origin_potential=(12,9), endpoint_potential=(-12,-8))
        self.assertEqual(p.certify_pairing_envelope(self.l, self.r, self.cost, low, self.high).minimum, 5)

    def test_rejects_bad_primal_dual_and_shape(self):
        for low, high in (
            (replace(self.low, flow=((0, 2), (0, 1))), self.high),  # incorrect endpoints
            (replace(self.low, flow=((2, 0), (-1, 2))), self.high), # negative but correct marginals
            (replace(self.low, origin_potential=(1,-4)), self.high), # objective/dual violation
            (replace(self.low, endpoint_potential=(-1,4)), self.high), # feasible dual with gap
            (self.low, replace(self.high, endpoint_potential=(0,3))), # reverse inequality
            (replace(self.low, flow=((True,1),(0,1))), self.high),
            (replace(self.low, flow=((1,1,0),(0,1,0))), self.high),
            (None, self.high)):
            with self.subTest(low=low,high=high), self.assertRaises(ValueError):
                p.certify_pairing_envelope(self.l,self.r,self.cost,low,high)
        for counts in ((2,2),(2,0),(True,2),(2.0,1),(2,1,1)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                p.certify_pairing_envelope(counts,self.r,self.cost,self.low,self.high)
