import pkg_resources as pkg
import equinox as eqx
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from jax import Array, vmap
from jax.scipy.signal import convolve


def gen_fourier_signal(single_ramp, coeffs, period=1024):
    orders = np.arange(len(coeffs)) + 1
    xs = vmap(lambda order: order * 2 * np.pi * single_ramp / period)(orders)
    basis = np.vstack([np.sin(xs), np.cos(xs)])
    return np.dot(coeffs.flatten(), basis)


class IPC(dl.detector_layers.DetectorLayer):
    ipc: Array

    def __init__(self, ipc):
        self.ipc = np.array(ipc, float)

    def apply(self, ramp):
        conv_fn = lambda x: convolve(x, self.ipc, mode="same")
        return ramp.set("data", vmap(conv_fn)(ramp.data))


class Amplifier(dl.detector_layers.DetectorLayer):
    one_on_fs: Array
    axis: int = eqx.field(static=True)

    def __init__(self, one_on_fs=None, axis=1):
        if one_on_fs is not None:
            self.one_on_fs = np.array(one_on_fs, float)
        else:
            self.one_on_fs = None
        self.axis = int(axis)

    def apply(self, ramp):

        if self.one_on_fs is None:
            return ramp

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


class LayeredDetector(dl.detectors.LayeredDetector):

    def __getattr__(self, key: str):
        if key in self.layers.keys():
            return self.layers[key]
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")

    def apply(self, psf):
        for layer in list(self.layers.values()):
            if layer is None:
                continue
            psf = layer.apply(psf)
        return psf


class ADC(dl.detector_layers.DetectorLayer):
    ADC_coeffs: Array
    period: int = eqx.field(static=True)

    def __init__(self, ADC_coeffs=None, period=1024):
        if ADC_coeffs is None:
            ADC_coeffs = np.zeros((1, 2))
        if ADC_coeffs[0, 0] == 0:
            ADC_coeffs = ADC_coeffs.at[0, 0].set(1.5)
        self.ADC_coeffs = np.array(ADC_coeffs, float)
        self.period = int(period)

    def apply(self, ramp):
        data = ramp.data
        apply_fn = vmap(lambda x: gen_fourier_signal(x, self.ADC_coeffs, self.period))
        correction = apply_fn(data.reshape(len(data), -1).T).T.reshape(data.shape)
        return ramp.add("data", correction)


class ReadModel(LayeredDetector):

    def __init__(self, dark_current=0.0, ipc=True, one_on_fs=None, ADC_coeffs=np.zeros((4, 2))):
        layers = []
        layers.append(("read", DarkCurrent(dark_current)))
        if ipc:
            file_path = pkg.resource_filename(__name__, "data/SUB80_ipc.npy")
            ipc = np.load(file_path)
        else:
            ipc = np.array([[1.0]])
        layers.append(("IPC", IPC(ipc)))
        layers.append(("amplifier", Amplifier(one_on_fs)))
        layers.append(("ADC", ADC(ADC_coeffs)))
        self.layers = dlu.list2dictionary(layers, ordered=True)
