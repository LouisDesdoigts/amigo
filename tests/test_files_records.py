"""Instrument records through amigo alone (no amigo-pipeline): save, load and old layouts."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy.testing as npt
import zodiax as zdx

from amigo.detector_models import LinearDetector
from amigo.files import load_calibration, load_record, save_calibration
from amigo.opd import FourierRecipe, OPDCalibration, PTTRecipe
from amigo.optical_models import DynamicApertureMask
from amigo.ramp_models import NonLinearRamp
from amigo.read_models import ReadModel


def model(static=True):
    pupil = DynamicApertureMask(
        holes=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        npixels=64,
        size=16,
        opds={"ptt": PTTRecipe(), "fourier": FourierRecipe(n_modes=3)},
    )
    global_opd = zdx.Map(x=zdx.Map(x=np.ones(2), s=1e-9, M=np.eye(2)), M=np.ones((2, 2, 16, 16)))
    pupil = pupil.set("opds", {**pupil.opds, "global": global_opd})
    if static:
        pupil = pupil.to_static()
    pupil = pupil.set("ptt_coeffs", np.arange(pupil.ptt_coeffs.size) + 1.0)
    cal = OPDCalibration(pupil, {"ptt_coeffs": {"001": pupil.ptt_coeffs}}, filters=("F380M",))
    return SimpleNamespace(
        detector=LinearDetector(rot_angle=0.7),
        ramp_model=NonLinearRamp(time_steps=16),
        read=ReadModel(ipc=static),
        params={"defocus": {"F380M": np.array(0.03)}},
        opd_calibration=lambda: cal,
    )


class FilesRecordTests(unittest.TestCase):
    def test_record_round_trip(self):
        for static in (True, False):
            saved = model(static)
            with TemporaryDirectory() as root:
                path = Path(root) / "base_model"
                save_calibration(path, saved)
                self.assertEqual(
                    {p.name for p in Path(root).iterdir()}, {"base_model.json", "base_model.npz"}
                )
                self.assertEqual(load_record(path)["schema"], "amigo-instrument-v2")
                loaded = load_calibration(path)
            self.assertEqual(loaded["ramp_model"].time_steps, 16)
            self.assertEqual(loaded["read"].layers["IPC"] is not None, static)
            npt.assert_allclose(loaded["detector"].resampler.rotation, 0.7)
            npt.assert_array_equal(loaded["ramp_model"].nn_weights, saved.ramp_model.nn_weights)
            original = saved.opd_calibration().aperture
            npt.assert_allclose(loaded["opd_calibration"].aperture.opd, original.opd)
            npt.assert_allclose(loaded["defocus"]["F380M"], 0.03)

    def test_record_folder_saved_before_2026_11_loads(self):
        # records used to be saved as <folder>/calibration.{json,npz}
        saved = model(False)
        with TemporaryDirectory() as root:
            folder = Path(root) / "old"
            folder.mkdir()
            save_calibration(folder, saved)
            self.assertEqual(
                {p.name for p in folder.iterdir()}, {"calibration.json", "calibration.npz"}
            )
            loaded = load_calibration(folder)
        npt.assert_allclose(
            loaded["opd_calibration"].aperture.opd, saved.opd_calibration().aperture.opd
        )


if __name__ == "__main__":
    unittest.main()
