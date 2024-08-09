import jax.numpy as np
import jax.random as jr
from amigo.files import get_files
from amigo.core_models import AmigoModel
from amigo.optical_models import AMIOptics
from amigo.detector_models import LinearDetectorModel
from amigo.read_models import ReadModel
from amigo.files import get_exposures, initialise_params
from amigo.model_fits import PointFit
from amigo.ramp_models import build_conv_layers, build_dense_layers, PolyConv
from amigo.fitting import optimise, sgd, adam
from amigo.fisher import calc_fishers


def load_cal(file_path, dithers=5):

    # Bind file path, type and exposure type
    file_fn = lambda **kwargs: get_files(
        [
            f"{file_path}/CAL04481/calslope/",
        ],
        "calslope",
        EXP_TYPE="NIS_AMI",
        IS_PSF=[True],  # Calibrators
        EXPOSURE=[str(i + 1) for i in range(5)],  # Which sub-pixel position
        **kwargs,
    )

    cal_files = file_fn()

    # Nuke bad edges and new bad pixel
    for file in cal_files:
        file["SCI"].data[:, :, -1:] = np.nan
        file["SCI_VAR"].data[:, :, -1:] = np.nan
        file["ZPOINT"].data[:, -1:] = np.nan
        file["ZPOINT_VAR"].data[:, -1:] = np.nan
        file["SCI"].data[:, 41:43, 1] = np.nan

    return cal_files


def initialise_model(files, conv_hidden=64, n_connect=16, dense_hidden=32, n_terms=6, seed=0):

    # Basic components
    optics = AMIOptics()
    detector = LinearDetectorModel()
    read_model = ReadModel()

    # Bleed model
    keys = jr.split(jr.PRNGKey(seed), 2)
    conv_layers = build_conv_layers(conv_hidden=conv_hidden, n_connect=n_connect, key=keys[0])
    dense_layers = build_dense_layers(
        n_connect=n_connect, n_hidden=dense_hidden, n_terms=n_terms, key=keys[1]
    )

    ramp_model = PolyConv(conv_layers, dense_layers, init_scale=0.5)

    # Prep the model
    exposures = get_exposures(files, PointFit())
    params = initialise_params(exposures, optics, fit_one_on_fs=True)

    # Construct
    model = AmigoModel(
        files,
        params,
        optics=optics,
        detector=detector,
        ramp=ramp_model,
        read=read_model,
    )

    # Return
    return model, exposures


def train(
    model,
    exposures,
    epochs,
    lr,
    batch_size,
    cache="files/fishers",
):

    # Get number of batches
    if batch_size > len(exposures):
        batch_size = len(exposures)
    nbatch = len(exposures) // batch_size
    if len(exposures) % batch_size != 0:
        nbatch += 1

    # Get the fisher lrs
    params = [
        "positions",
        "aberrations",
        "fluxes",
        "dark_current",
        "jitter.r",
        "f2f",
        "distortion",
        "one_on_fs",
        "ADC_coeffs",
    ]
    fishers = calc_fishers(model.set("ramp", None), exposures, params, cache=cache)

    # Define the optimisers
    optimisers = {
        "positions": sgd(5e-1, 0),
        "aberrations": sgd(5e-2, 5),
        "fluxes": sgd(2e-1, 5),
        "dark_current": sgd(2e-1, 5),
        "conv.values": adam(lr, 10 * nbatch),
        "dense.values": adam(lr, 10 * nbatch),
        "jitter.r": sgd(5e-1, 20),
        "f2f": sgd(2e-1, 20),
        "distortion": sgd(1e-1, 20),
        "ADC_coeffs": sgd(1e-1, 25),
        "one_on_fs": sgd(1e-1, 25),
    }

    # Optimise!
    final_model, losses, histories, states = optimise(
        model,
        exposures,
        optimisers,
        epochs=epochs,
        key=jr.PRNGKey(0),
        fishers=fishers,
        batch_size=batch_size,
        batch_params=["conv.values", "dense.values"],
    )

    return final_model, losses, histories, states
