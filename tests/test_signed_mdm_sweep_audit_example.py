import hashlib
from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = runpy.run_path(str(ROOT / "examples/signed-mdm-sweep-audit/run.py"))["audit"]


def block(vb):
    return f"""BEGIN_DB
ICCAP_VAR vb {vb}
ICCAP_VAR vd -2
ICCAP_VAR vs 0
#vg id ig ib is
0 -1 0 0 1
-1 -2 0 0 2
END_DB
""".encode()


def verified(raw):
    return AUDIT(raw, hashlib.sha256(raw).hexdigest())


class SignedMdmSweepAuditTests(unittest.TestCase):
    def test_preserves_sign_and_body_bias_blocks(self):
        value = verified(block("0") + block("1"))
        self.assertEqual((value["blocks"], value["data_rows"]), (2, 4))
        self.assertEqual((value["body_zero_blocks"], value["body_nonzero_blocks"]), (1, 1))
        self.assertEqual(value["native_gate_step"], "-1")
        self.assertEqual(value["vd_levels"], ["-2"])
        self.assertEqual(value["drain_current_sign_at_last_gate"], [-1])

    def test_rejects_wrong_hash_and_duplicate_numeric_keys(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            AUDIT(block("0"), "0" * 64)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verified(block("0") + block("-0.0"))
