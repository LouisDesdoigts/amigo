"""Primary dither positions from XOFFSET/YOFFSET, and the fixed rotation offset an exposure adds
to the rotation it reads."""

from copy import copy
from types import SimpleNamespace
import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
from astropy.io import fits

from amigo.misc import primary_offset
from amigo.model_fits import DarkFit, FlatFit, SplineVisFit


def calslope(**cards):
    """A minimal calslope file with extra header cards."""
    slopes = onp.zeros((2, 80, 80))
    slopes[:, 40, 46] = 1.0
    header = fits.Header(
        {
            "ROLL_REF": 10.0,
            "NINTS": 1,
            "FILTER": "F380M",
            "TARGPROP": "STAR",
            "OBSERVTN": "001",
            "PROGRAM": "01234",
            "ACT_ID": "02",
            "VISITGRP": "03",
            "EXPOSURE": "1",
            "IS_PSF": True,
            "FILENAME": "jw01234001001_03102_00001_nis_calslope.fits",
            "DATE-OBS": "2024-07-01",
            **cards,
        }
    )
    hdus = [fits.PrimaryHDU(header=header)]
    for name, data in (
        ("SLOPE", slopes),
        ("SLOPE_COV", onp.ones((2, 2, 80, 80)) + onp.eye(2)[..., None, None]),
        ("RAMP", onp.zeros((3, 80, 80))),
        ("RAMP_COV", onp.ones((3, 3, 80, 80))),
        ("SLOPE_SUP", onp.ones((80, 80), int)),
        ("BADPIX", onp.zeros((80, 80), int)),
    ):
        hdus.append(fits.ImageHDU(data, name=name))
    return fits.HDUList(hdus)


class PrimaryOffsetTests(unittest.TestCase):
    def test_the_primary_positions(self):
        # the five primaries on tauron's program files (2026-11-26)
        cases = {
            (-0.0, -0.0): "x+0p0y+0p0",
            (-0.915, 0.924): "x-1p0y+1p0",
            (-1.829, -1.044): "x-2p0y-1p0",
            (0.915, -1.121): "x+1p0y-1p0",
            (-1.831, 1.127): "x-2p0y+1p0",
        }
        for (x, y), label in cases.items():
            self.assertEqual(primary_offset({"XOFFSET": x, "YOFFSET": y}), label)

    def test_sub_pixel_dithers_keep_their_primary(self):
        for dx, dy in [(0.039, -0.039), (-0.016, 0.017), (0.033, 0.033), (-0.009, -0.009)]:
            self.assertEqual(primary_offset({"XOFFSET": dx, "YOFFSET": dy}), "x+0p0y+0p0")
            self.assertEqual(
                primary_offset({"XOFFSET": -0.915 + dx, "YOFFSET": 0.924 + dy}), "x-1p0y+1p0"
            )

    def test_negative_zero_and_missing(self):
        self.assertEqual(primary_offset({"XOFFSET": -0.0, "YOFFSET": -0.2}), "x+0p0y+0p0")
        self.assertIsNone(primary_offset({"XOFFSET": 0.0}))
        self.assertIsNone(primary_offset({}))

    def test_get_pos_cannot_tell_these_apart(self):
        # the 4-point pattern's second position and the 2-point pattern's
        a = primary_offset({"XOFFSET": -0.915, "YOFFSET": 0.924})
        b = primary_offset({"XOFFSET": -1.831, "YOFFSET": 1.127})
        self.assertNotEqual(a, b)


class RotationOffsetTests(unittest.TestCase):
    def test_exposure_reads_pwcpos_and_position(self):
        exp = SplineVisFit(
            calslope(PWCPOS=204.969559, XOFFSET=-0.915, YOFFSET=0.924), covariance="diagonal"
        )
        self.assertEqual(
            (exp.pwcpos, exp.primary_offset, exp.rotation_offset),
            (204.969559, "x-1p0y+1p0", 0.0),
        )
        bare = SplineVisFit(calslope(), covariance="diagonal")
        self.assertEqual((bare.pwcpos, bare.primary_offset), (None, None))

    def test_rotation_adds_the_offset(self):
        exp = SplineVisFit(calslope(PWCPOS=205.0), covariance="diagonal")
        model = SimpleNamespace(rotations={"01234": np.array(0.0125)})
        self.assertAlmostEqual(float(exp.rotation(model)), 0.0125)
        shifted = copy(exp)
        object.__setattr__(shifted, "rotation_offset", -0.0244)
        self.assertAlmostEqual(float(shifted.rotation(model)), 0.0125 - 0.0244)
        # static: part of the tree structure, so it survives flattening
        leaves, tree = jax.tree.flatten(shifted)
        self.assertEqual(jax.tree.unflatten(tree, leaves).rotation_offset, -0.0244)
        self.assertEqual(exp.rotation_offset, 0.0)  # the original is unchanged

    def test_flats_and_darks_still_construct(self):
        flat = FlatFit(calslope())
        dark = DarkFit(calslope())
        self.assertEqual((flat.rotation_offset, dark.rotation_offset), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
