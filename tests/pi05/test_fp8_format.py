# SPDX-License-Identifier: AGPL-3.0-only
"""CPU checks for the strict FP8 overlay boundary."""

import hashlib
import unittest

from fp8_format import bound_source_hashes, tensor_names, verify_source


class Fp8FormatTests(unittest.TestCase):
    def test_scopes_select_exact_projection_inventories(self):
        vision = tensor_names("vision")
        both = tensor_names("vision-expert")
        expert = tensor_names("expert")
        self.assertEqual(len(expert), 126)
        self.assertEqual(both[163:], expert)
        self.assertEqual(len(vision), 163)
        self.assertEqual(len(both), 289)
        self.assertEqual(len(set(both)), 289)
        self.assertEqual(both[:163], vision)
        self.assertTrue(
            all("norm" not in name and "embed" not in name for name in both)
        )
        with self.assertRaises(ValueError):
            tensor_names("all")

    def test_overlay_cannot_claim_absent_or_changed_base_tensor(self):
        raw = bytes([0, 0, 128, 63])
        digest = hashlib.sha256(raw).hexdigest()
        manifest = {"base_tensor_sha256": {"weight": digest}}
        self.assertEqual(bound_source_hashes(manifest, ["weight"]), {"weight": digest})
        verify_source(raw, digest, "weight")
        with self.assertRaises(ValueError):
            bound_source_hashes(manifest, ["absent"])
        with self.assertRaises(ValueError):
            verify_source(raw + b"x", digest, "weight")


if __name__ == "__main__":
    unittest.main()
