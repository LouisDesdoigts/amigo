import pkg_resources as pkg
import equinox as eqx
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from jax import Array, vmap
from jax.scipy.signal import convolve


# def model_dark_current(dark_current, ngroups):
#     """Models the dark current as a constant background value added cumulatively to
#     each group. For now we assume that the dark current is a float."""
#     return (dark_current * (np.arange(ngroups) + 1))[..., None, None]


class IPC(dl.detector_layers.DetectorLayer):
    ipc: Array

    def __init__(self, ipc):
        self.ipc = np.array(ipc, float)

    def apply(self, ramp):
        conv_fn = lambda x: convolve(x, self.ipc, mode="same")
        return ramp.set("data", vmap(conv_fn)(ramp.data))


class Amplifier(dl.detector_layers.DetectorLayer):
    one_on_fs: Array
    axis: int = eqx.field(staic=True)

    def __init__(self, one_on_fs=None, axis=1):
        if one_on_fs is not None:
            self.one_on_fs = np.array(one_on_fs, float)
        else:
            self.one_on_fs = None
        self.axis = int(axis)

    def apply(self, ramp):

        def read_fn(coeffs):
            xs = np.linspace(-1, 1, coeffs.shape[0])
            return np.rot90(vmap(lambda coeffs: np.polyval(coeffs, xs))(coeffs))

        return ramp.add("data", vmap(read_fn)(self.one_on_fs))


class DarkCurrent(dl.detector_layers.DetectorLayer):
    dark_current: Array

    def __init__(self, dark_current):
        self.dark_current = np.array(dark_current, float)

    def apply(self, ramp):
        dark_current = self.dark_current * (np.arange(len(ramp.data)) + 1)
        # dark_current = model_dark_current(self.dark_current, len(ramp.data))
        return ramp.add("data", dark_current[..., None, None])


class ReadModel(dl.detectors.LayeredDetector):

    def __init__(
        self,
        dark_current=0.0,
        ipc=True,
        one_on_fs=None,
    ):
        layers = []
        layers.append(("read", DarkCurrent(dark_current)))
        if ipc:
            file_path = pkg.resource_filename(__name__, "data/SUB80_ipc.npy")
            ipc = np.load(file_path)
        else:
            ipc = np.array([[1.0]])
        layers.append(("IPC", IPC(ipc)))
        layers.append(("amplifier", Amplifier(one_on_fs)))
        self.layers = dlu.list2dictionary(layers, ordered=True)
