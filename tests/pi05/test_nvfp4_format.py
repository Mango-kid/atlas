# SPDX-License-Identifier: AGPL-3.0-only
"""CPU interchange tests, runnable with `python -m unittest discover ...`."""

import io
import unittest

import numpy as np
from nvfp4_format import append_aligned, swizzle_scales


class FormatTests(unittest.TestCase):
    def test_swizzle_matches_explicit_cublas_offsets_and_padding(self):
        # Cross both row-tile and four-group boundaries, including padding.
        values = np.arange(133 * 9, dtype=np.uint32).astype(np.uint8).reshape(133, 9)
        result = swizzle_scales(values)
        expected = np.zeros(256 * 12, dtype=np.uint8)
        for row in range(133):
            for group in range(9):
                offset = (
                    ((row // 128) * 3 + group // 4) * 512
                    + (row % 32) * 16
                    + ((row % 128) // 32) * 4
                    + group % 4
                )
                expected[offset] = values[row, group]
        np.testing.assert_array_equal(result, expected)

    def test_payload_offsets_align_without_altering_data(self):
        stream = io.BytesIO()
        first, _ = append_aligned(stream, b"abc")
        second, _ = append_aligned(stream, b"defgh")
        self.assertEqual((first, second), (0, 256))
        self.assertEqual(stream.getvalue()[:3], b"abc")
        self.assertEqual(stream.getvalue()[256:], b"defgh")


if __name__ == "__main__":
    unittest.main()
