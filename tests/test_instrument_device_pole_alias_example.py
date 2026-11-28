from fractions import Fraction as F
from itertools import product
from pathlib import Path
from runpy import run_path
import unittest

EXAMPLE = Path(__file__).resolve().parents[1]/"examples"/"instrument-device-pole-alias"/"run.py"
TOOLS = run_path(str(EXAMPLE))


class InstrumentDevicePoleAlias(unittest.TestCase):
    def test_all_signed_waveforms_share_the_same_cascade(self):
        for a,b in [(F(3,4),F(1,4)),(F(5,6),0),(F(7,9),F(3,11))]:
            for x in product((-1,0,2),repeat=4):
                self.assertEqual(TOOLS["cascade"](a,b,x),TOOLS["cascade"](b,a,x))

    def test_reference_distinguishes_parameter_attribution(self):
        self.assertEqual(TOOLS["witness"](),{
            "same_training_cascade":["3/16","3/8","135/256","165/256"],
            "two_sensor_only_initial_responses":["3/4","1/4"],
            "new_path_slow_device":["1/8","9/32"],
            "new_path_fast_device":["3/8","21/32"],
            "wrong_device_attribution_mae":"5/16",
        })
        for pole in (True,-1,1,.5):
            with self.assertRaises(ValueError):
                TOOLS["response"](pole,[1])
