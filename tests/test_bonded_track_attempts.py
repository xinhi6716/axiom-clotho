from fractions import Fraction
import itertools
import unittest

from groupbench.process_contract import BondedTrackAttempt as A, summarize_bonded_track_attempts as score


class BondedTrackAttemptsTests(unittest.TestCase):
    def test_geometry_and_full_roster_can_disagree(self):
        ids = ['a','b','c','d']
        Arows = [A('a','bonded','11'),A('b','failed'),A('c','failed'),A('d','bonded','13')]
        Brows = [A(i,'bonded',h) for i,h in zip(ids,('7','8','9','8'))]
        xx,yy=[score(ids,r,height_window_mm=('6','15')) for r in (Arows,Brows)]
        self.assertEqual((xx.mean_measured_bonded_height_mm,yy.mean_measured_bonded_height_mm),(12,8))
        self.assertEqual(xx.qualified_fraction_bounds,(Fraction(1,2),Fraction(1,2)))
        self.assertEqual(yy.qualified_fraction_bounds,(1,1))
        self.assertEqual(score(ids,Arows[::-1],height_window_mm=('6','15')),xx)

    def test_missing_are_unknown_while_observed_zero_is_valid(self):
        rows=[A('a','bonded','0'),A('b','bonded'),A('c','unknown'),A('d','failed')]
        x=score(['a','b','c','d'],rows,height_window_mm=('0','0'))
        self.assertEqual((x.attempted,x.bonded,x.failed_bond,x.unknown_bond,x.measured_bonded_height),(4,2,1,1,1))
        self.assertEqual(x.mean_measured_bonded_height_mm,0)
        self.assertEqual(x.bonding_fraction_bounds,(Fraction(1,2),Fraction(3,4)))
        self.assertEqual(x.qualified_fraction_bounds,(Fraction(1,4),Fraction(3,4)))

    def test_exact_windows_and_missing_all(self):
        self.assertEqual(score(['a'],[A('a','bonded','.3')],height_window_mm=('0.30','0.3')).qualified_fraction_bounds,(1,1))
        x=score(['a'],[A('a','unknown')],height_window_mm=('0','1'))
        self.assertIsNone(x.mean_measured_bonded_height_mm)
        self.assertEqual(x.qualified_fraction_bounds,(0,1))

    def test_roster_and_disposition_disagreement_fails(self):
        for rows in ([A('a','bonded','1')], [A('a','bonded','1'),A('a','failed')],
                     [A('a','failed','0'),A('b','unknown')],[A('a','unknown','0'),A('b','unknown')],
                     [A('a','bonded','-0.01'),A('b','unknown')],[A('a','bonded','1'),A('x','unknown')]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                score(['a','b'],rows,height_window_mm=('0','1'))
        for window in (('2','1'),('-1','1'),('nan','1'),(True,'1')):
            with self.subTest(window=window),self.assertRaises(ValueError):
                score(['a'],[A('a','bonded','1')],height_window_mm=window)

    def test_small_roster_bound_oracle(self):
        cases=(('bonded','1',((1,1),)),('bonded','3',((1,0),)),
               ('bonded',None,((1,1),(1,0))),('failed',None,((0,0),)),
               ('unknown',None,((0,0),(1,0),(1,1))))
        ids=['a','b','c']
        for three in itertools.product(cases,repeat=3):
            inputs=[A(i,status,h) for i,(status,h,_) in zip(ids,three)]
            x=score(ids,inputs,height_window_mm=('1','2'))
            resolved=[tuple(map(sum, zip(*vals))) for vals in itertools.product(*(c[2] for c in three))]
            bounds=[(Fraction(min(col),3),Fraction(max(col),3)) for col in zip(*resolved)]
            self.assertEqual((x.bonding_fraction_bounds,x.qualified_fraction_bounds),tuple(bounds))
