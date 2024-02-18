import jax.numpy as np
import jax.scipy as jsp
import numpy as onp


def estimate_ramp(data):
    ramp = np.arange(len(data)) + 1
    mean_diff = np.diff(data, axis=0).mean(0)
    ramp = mean_diff[None, ...] * ramp[..., None, None]
    return ramp.at[np.where(np.isnan(ramp))].set(0.0)


def estimate_bias(data):
    mean_diff = np.diff(data, axis=0).mean(0)
    bias = data[0] - mean_diff
    return bias.at[np.where(np.isnan(bias))].set(0.0)


def summarise_files(files, extra_keys=[]):
    main_keys = []

    main_keys += extra_keys
    for key in main_keys:
        values = set([f[0].header[key] for f in files])
        vals_str = ", ".join([f"{val}" for val in values])
        print(f"  {key}: {vals_str}")


def get_files(paths, ext, **kwargs):
    """

    data_path: Path to the data files
    ext: File extension to search for
    """
    import os
    from astropy.io import fits

    if isinstance(paths, str):
        paths = [paths]

    files = []
    for path in paths:
        file_names = os.listdir(path)

        checked = False
        for name in file_names:
            if name.endswith(f"{ext}.fits"):
                file = fits.open(path + name)
                h = file[0].header
                if not checked:
                    if not all([key in h.keys() for key in kwargs.keys()]):
                        raise KeyError(
                            f"Header keys {kwargs.keys()} not found in file {name}"
                        )

                match = True
                for key, val in kwargs.items():
                    if isinstance(val, list):
                        if h[key] not in val:
                            match = False
                    elif h[key] != val:
                        match = False

                if match:
                    files.append(file)
    return files


def get_filter(filter_name: str, filter_dir: str, n_wavels: int = 9):
    if filter_name not in ["F380M", "F430M", "F480M", "F277W"]:
        raise ValueError("Supported filters are F380M, F430M, F480M, F277W.")

    wl_array, throughput_array = np.array(
        onp.loadtxt(filter_dir + "JWST_NIRISS." + filter_name + ".dat", unpack=True)
    )

    edges = np.linspace(wl_array.min(), wl_array.max(), n_wavels + 1)
    wavels = np.linspace(wl_array.min(), wl_array.max(), 2 * n_wavels + 1)[1::2]

    areas = []
    for i in range(n_wavels):
        cond1 = edges[i] < wl_array
        cond2 = wl_array < edges[i + 1]
        throughput = np.where(cond1 & cond2, throughput_array, 0)
        areas.append(jsp.integrate.trapezoid(y=throughput, x=wl_array))

    areas = np.array(areas)
    weights = areas / areas.sum()

    wavels *= 1e-10
    return wavels, weights
