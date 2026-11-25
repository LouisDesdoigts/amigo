"""Instrument records through amigo alone (no amigo-pipeline): save, load, old layouts, and the
choice of Fourier set by exposure date."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import jax

jax.config.update("jax_enable_x64", True)
import equinox as eqx
import jax.numpy as np
import numpy as onp
import numpy.testing as npt
import zodiax as zdx
from astropy.io import fits

from amigo.detector_models import LinearDetector
from amigo.files import (
    SCHEMA,
    _structure,
    assign_fourier_epochs,
    load_calibration,
    load_record,
    nearest_epoch,
    save_calibration,
    save_record,
)
from amigo.model_fits import Exposure
from amigo.opd import FourierRecipe, OPDCalibration, PTTRecipe, add_filter_ptt
from amigo.optical_models import AMIOptics, DynamicApertureMask
from amigo.ramp_models import NonLinearRamp
from amigo.read_models import ReadModel

EPOCHS = {"04481": "2024-05-05", "08330": "2025-11-11"}


def aperture(static=True):
    pupil = DynamicApertureMask(
        holes=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        npixels=64,
        size=16,
        opds={"ptt": PTTRecipe(), "fourier": FourierRecipe(n_modes=3)},
        opd_sharing={"fourier_coeffs": ("fourier_epoch",)},
    )
    global_opd = zdx.Map(x=zdx.Map(x=np.ones(2), s=1e-9, M=np.eye(2)), M=np.ones((2, 2, 16, 16)))
    pupil = pupil.set("opds", {**pupil.opds, "global": global_opd})
    if static:
        pupil = pupil.to_static()
    return add_filter_ptt(pupil)


def model(static=True):
    pupil = aperture(static)
    n_ptt, n_fourier = pupil.ptt_coeffs.size, pupil.fourier_coeffs.size
    coefficients = {
        "ptt_coeffs": {"001": np.arange(n_ptt) + 1.0},
        "filter_ptt_coeffs": {"F380M": np.linspace(-1, 1, n_ptt)},
        "fourier_coeffs": {"04481": np.ones(n_fourier), "08330": -np.ones(n_fourier)},
    }
    cal = OPDCalibration(pupil, coefficients, filters=("F380M",))
    return SimpleNamespace(
        detector=LinearDetector(rot_angle=0.7, kernel_size=9, kernel_osamp=3),
        ramp_model=NonLinearRamp(time_steps=16, hidden_width=8, n_hidden_layers=2),
        read=ReadModel(ipc=static, per_pixel_dark=static),
        params={"defocus": {"F380M": np.array(0.03)}, "sigmas": {"F380M": np.array(0.019)}},
        opd_calibration=lambda: cal,
    )


def arrays(tree):
    return [onp.asarray(x) for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def exposure(date_obs):
    """A minimal calslope file, enough to construct an Exposure."""
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
            "DATE-OBS": date_obs,
        }
    )
    hdus = [fits.PrimaryHDU(header=header)]
    for name, data in (
        ("SLOPE", slopes),
        ("SLOPE_COV", onp.ones((2, 2, 80, 80))),
        ("RAMP", onp.zeros((3, 80, 80))),
        ("RAMP_COV", onp.ones((3, 3, 80, 80))),
        ("SLOPE_SUP", onp.ones((80, 80), int)),
        ("BADPIX", onp.zeros((80, 80), int)),
    ):
        hdus.append(fits.ImageHDU(data, name=name))
    return Exposure(fits.HDUList(hdus))


class FilesRecordTests(unittest.TestCase):
    def test_record_round_trip(self):
        for static in (True, False):
            saved = model(static)
            with TemporaryDirectory() as root:
                path = Path(root) / "base_model"
                save_calibration(path, saved, fourier_epochs=EPOCHS)
                self.assertEqual(
                    {p.name for p in Path(root).iterdir()}, {"base_model.json", "base_model.npz"}
                )
                record = load_record(path)
                loaded = load_calibration(path)
            self.assertEqual(record["schema"], SCHEMA)
            self.assertEqual(
                record["settings"]["ramp"],
                {
                    "time_steps": 16,
                    "norm": 2**15,
                    "bleed": True,
                    "use_charge": True,
                    "use_SRF": False,
                    "hidden_width": 8,
                    "n_hidden_layers": 2,
                },
            )
            self.assertEqual(record["settings"]["detector"], {"kernel_size": 9, "oversample": 3})
            self.assertEqual(record["settings"]["read"]["per_pixel_dark"], static)
            self.assertEqual("IPC" in loaded["read"].layers, static)
            for key in ("detector", "ramp_model", "read"):
                original, restored = getattr(saved, key), loaded[key]
                self.assertEqual(_structure(original), _structure(restored), key)
                for a, b in zip(arrays(original), arrays(restored), strict=True):
                    npt.assert_array_equal(a, b)
            calibration, restored = saved.opd_calibration(), loaded["opd_calibration"]
            for a, b in zip(arrays(calibration), arrays(restored), strict=True):
                npt.assert_array_equal(a, b)
            self.assertEqual(restored.aperture.opd_sharing, calibration.aperture.opd_sharing)
            self.assertEqual(restored.aperture.alias, calibration.aperture.alias)
            self.assertEqual(loaded["fourier_epochs"], EPOCHS)
            npt.assert_array_equal(loaded["defocus"]["F380M"], 0.03)
            npt.assert_array_equal(loaded["sigmas"]["F380M"], 0.019)

    def test_record_folder_saved_before_2026_11_loads(self):
        # records used to be saved as <folder>/calibration.{json,npz}
        saved = model(False)
        with TemporaryDirectory() as root:
            folder = Path(root) / "old"
            folder.mkdir()
            save_calibration(folder, saved, fourier_epochs=EPOCHS)
            self.assertEqual(
                {p.name for p in folder.iterdir()}, {"calibration.json", "calibration.npz"}
            )
            loaded = load_calibration(folder)
        for a, b in zip(
            arrays(saved.opd_calibration()), arrays(loaded["opd_calibration"]), strict=True
        ):
            npt.assert_array_equal(a, b)

    def test_epoch_sets_need_their_dates(self):
        with TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "one date per Fourier set"):
                save_calibration(Path(root) / "base_model", model())
            with self.assertRaisesRegex(ValueError, "one date per Fourier set"):
                save_calibration(
                    Path(root) / "base_model", model(), fourier_epochs={"04481": "2024-05-05"}
                )

    def test_amigo_0_1_record_is_refused(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "base_model"
            save_record(path, {"schema": "amigo-instrument-v2", "settings": {}})
            with self.assertRaisesRegex(ValueError, "load it with amigo 0.1"):
                load_calibration(path)

    def test_nearest_fourier_epoch(self):
        self.assertEqual(nearest_epoch("2024-07-10", EPOCHS), "04481")
        self.assertEqual(nearest_epoch("2025-06-01", EPOCHS), "08330")
        self.assertEqual(nearest_epoch("2025-02-06", EPOCHS), "04481")  # 277 vs 278 days
        # equidistant: the earlier epoch
        tied = {"B": "2024-01-11", "A": "2024-01-01"}
        self.assertEqual(nearest_epoch("2024-01-06", tied), "A")
        with self.assertRaisesRegex(ValueError, "without a date"):
            nearest_epoch(None, EPOCHS)

    def test_exposures_carry_date_and_epoch(self):
        exp = exposure("2025-11-12")
        self.assertEqual((exp.date, exp.fourier_epoch), ("2025-11-12", None))
        (assigned,) = assign_fourier_epochs([exp], EPOCHS)
        self.assertEqual((assigned.date, assigned.fourier_epoch), ("2025-11-12", "08330"))
        self.assertIsNone(exp.fourier_epoch)  # the original is unchanged
        # static: part of the tree structure, so it survives flattening
        leaves, tree = jax.tree.flatten(assigned)
        self.assertEqual(jax.tree.unflatten(tree, leaves).fourier_epoch, "08330")

    def test_filter_ptt_adds_to_ptt_and_is_held(self):
        pupil = aperture(False)
        a = np.arange(pupil.ptt_coeffs.size) + 1.0
        b = np.linspace(-1, 1, a.size)
        split = pupil.set(["ptt_coeffs", "filter_ptt_coeffs"], [a, b])
        joined = pupil.set(["ptt_coeffs", "filter_ptt_coeffs"], [a + b, np.zeros_like(b)])
        npt.assert_allclose(split.opd, joined.opd, rtol=0, atol=1e-20)
        self.assertGreater(float(np.abs(split.opd).max()), 1e-10)
        optics = AMIOptics(opd_calibration=model(False).opd_calibration(), filters=["F380M"])
        self.assertIn("ptt_coeffs", optics.opd_fits)
        self.assertNotIn("filter_ptt_coeffs", optics.opd_fits)


if __name__ == "__main__":
    unittest.main()
