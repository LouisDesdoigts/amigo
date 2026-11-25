"""amigo's files: finding and summarising the calslope data, and instrument records.

An instrument record (``<name>.json`` + ``<name>.npz``) holds a model's instrument
and calibrated optics as plain values and construction settings, never Python
objects. ``save_calibration`` writes one; ``load_calibration`` returns the arguments
to build the model from it, e.g. a versioned base model from the ``amigo_files``
repository: ``load_calibration("amigo_files/v0.2/base_model")``. amigo 0.2 writes and
reads schema ``amigo-instrument-v3``; amigo 0.1's records (v2) load with amigo 0.1.
Custom instruments need their own record adapter; they are not silently coerced into
the standard architecture.
"""

from contextlib import contextmanager
from copy import copy
from datetime import date
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
import inspect
import json
import marshal
import os

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import zodiax as zdx

from .detector_models import LinearDetector
from .optical_models import (
    StaticApertureMask,
    DynamicApertureMask,
    AffineDistortion,
    PolynomialDistortion,
    window_centers,
)
from .opd import BasisOPD, FourierOPD, OPDCalibration
from .ramp_models import NonLinearRamp
from .read_models import ReadModel, PixelNonLinearity, Amplifier
from .vis_models import PlaceholderVisModel


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
                        raise KeyError(f"Header keys {kwargs.keys()} not found in file {name}")

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


# Plain-value records: JSON structure plus a pickle-free array archive


@contextmanager
def atomic(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, record):
    with atomic(path) as temporary:
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def save_arrays(path, **arrays):
    arrays = {key: onp.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise TypeError("Array archives cannot contain Python objects")
    with atomic(path) as temporary:
        with temporary.open("wb") as handle:
            onp.savez_compressed(handle, **arrays)


def load_arrays(path):
    with onp.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def save_record(path, record):
    """Save a plain dictionary as JSON structure plus a pickle-free array archive.

    Only dictionaries, lists, arrays and Python scalar values are accepted.
    Records carry their own explicit schema; classes and callables are rejected.
    """
    arrays = {}

    def encode(value):
        if isinstance(value, (jax.Array, onp.ndarray)):
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"$array": key}
        if isinstance(value, dict):
            if any(not isinstance(key, str) or key == "$array" for key in value):
                raise TypeError("Record keys must be strings other than '$array'")
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise TypeError(f"Not a plain record value: {type(value).__name__}")

    if not isinstance(record, dict):
        raise TypeError("A record must be a dictionary")
    structure = encode(record)
    save_arrays(Path(path).with_suffix(".npz"), **arrays)
    save_json(Path(path).with_suffix(".json"), structure)


def load_record(path):
    """Read only plain values; never import classes or execute saved code."""
    arrays = load_arrays(Path(path).with_suffix(".npz"))

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"$array"}:
                return arrays[value["$array"]]
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value

    return decode(json.loads(Path(path).with_suffix(".json").read_text()))


def fingerprint(*objects):
    """Hash model/exposure values AND static tree structure, without pickle."""
    digest = sha256()
    for obj in objects:
        leaves, tree = jax.tree_util.tree_flatten(obj)
        digest.update(str(tree).encode())
        for leaf in leaves:
            if hasattr(leaf, "dtype"):
                array = onp.asarray(leaf)
                digest.update(str((array.shape, array.dtype)).encode())
                digest.update(array.tobytes())
            elif callable(leaf):
                # Neural ramp activations are leaves too. Their repr contains a
                # process address, which would invalidate every fresh-process resume.
                function = inspect.unwrap(leaf)
                if not inspect.isfunction(function) and not inspect.isbuiltin(function):
                    raise TypeError(f"Cannot fingerprint callable {type(leaf).__name__}")
                digest.update(f"{function.__module__}.{function.__qualname__}".encode())
                if inspect.isfunction(function):
                    digest.update(marshal.dumps(function.__code__))
                    captured = tuple(cell.cell_contents for cell in (function.__closure__ or ()))
                    digest.update(
                        fingerprint(
                            function.__defaults__, function.__kwdefaults__, captured
                        ).encode()
                    )
            else:
                digest.update(repr(leaf).encode())
    return digest.hexdigest()


# Instrument records


def _component(value):
    """Describe the three supported linear OPD evaluators and their bound Maps."""
    if value is None or not isinstance(value, zdx.Base):
        return value
    if type(value) is zdx.Map:
        return {
            "kind": "map",
            "x": _component(value.x),
            "s": value.s,
            "M": value.M,
            "b": value.b,
            "alias": value.alias,
        }
    if type(value) is BasisOPD:
        return {
            "kind": "basis",
            "x": _component(value.x),
            "basis": value.basis,
            "alias": value.alias,
        }
    if type(value) is FourierOPD:
        return {
            "kind": "fourier",
            "x": _component(value.x),
            "kx": value.kx,
            "ky": value.ky,
            "alias": value.alias,
        }
    raise TypeError(f"Unsupported OPD component: {type(value).__name__}")


def _restore_component(record):
    if not isinstance(record, dict):
        return None if record is None else np.asarray(record)
    fields = dict(record)
    kind = fields.pop("kind")
    fields["x"] = _restore_component(fields["x"])
    if kind == "map":
        return zdx.Map(**fields)
    if kind == "basis":
        return BasisOPD(**fields)
    if kind == "fourier":
        kx, ky = fields.pop("kx"), fields.pop("ky")
        result = FourierOPD((kx.shape[1], ky.shape[1]), (kx.shape[0], ky.shape[0]), **fields)
        return result.set(["kx", "ky"], [np.asarray(kx), np.asarray(ky)])
    raise ValueError(f"Unknown OPD component kind: {kind}")


def _aperture(aperture):
    fields = {
        "size": aperture.size,
        "npixels": aperture.npixels,
        "diameter": aperture.diameter,
        "normalise": bool(aperture.normalise),
        "corners": aperture.corners,
        "amp_basis": aperture.amp_basis,
        "amp_coeffs": aperture.amp_coeffs,
        "alias": aperture.alias,
        "sharing": aperture.opd_sharing,
        "opds": {key: _component(value) for key, value in (aperture.opds or {}).items()},
    }
    if type(aperture) is StaticApertureMask:
        return {"kind": "static", "transmission": aperture.transmission, **fields}
    if type(aperture) is DynamicApertureMask:
        distortion = aperture.distortion
        if type(distortion) not in (AffineDistortion, PolynomialDistortion):
            raise TypeError("Custom aperture distortions need an explicit record adapter")
        return {
            "kind": "dynamic",
            "holes": _component(aperture.holes),
            "f2f": aperture.f2f,
            "oversample": aperture.oversample,
            "distortion": {
                "powers": distortion.powers,
                "coefficients": distortion.coefficients,
                "kind": "affine" if type(distortion) is AffineDistortion else "polynomial",
            },
            **fields,
        }
    raise TypeError("Custom apertures need an explicit record adapter")


# Records written before the global Zernike component was renamed (2026-11) call it
# "filter", with coefficients "filter_coeffs" at "opds.filter.x.x". Read them under the
# new names.
_RENAMED = {"filter": "zernike", "filter_coeffs": "zernike_coeffs"}


def _renamed_alias(name, target):
    if target.startswith("opds.filter."):
        target = "opds.zernike." + target[len("opds.filter.") :]
    return _RENAMED.get(name, name), target


def _restore_aperture(record):
    fields = dict(record)
    kind = fields.pop("kind")
    opds = {
        _RENAMED.get(key, key): _restore_component(value)
        for key, value in fields.pop("opds").items()
    }
    aliases, sharing = fields.pop("alias"), fields.pop("sharing")
    if aliases is not None:  # zodiax keeps aliases sorted by name
        aliases = sorted(_renamed_alias(name, target) for name, target in aliases)
    if sharing is not None:
        sharing = [(_RENAMED.get(name, name), group) for name, group in sharing]
    corners = np.asarray(fields.pop("corners"))
    amplitude = {key: fields.pop(key) for key in ("amp_basis", "amp_coeffs")}
    if kind == "static":
        holes = window_centers(corners, fields["size"], fields["npixels"], fields["diameter"])
        aperture = StaticApertureMask(holes=holes, **fields)
    elif kind == "dynamic":
        distortion = fields.pop("distortion")
        cls = {"affine": AffineDistortion, "polynomial": PolynomialDistortion}[distortion["kind"]]
        # Exact sampled powers are retained; never regenerate them from a guessed order.
        shape = distortion["coefficients"].shape
        mapping = cls(n_holes=shape[0]).set(
            ["powers", "coefficients"],
            [np.asarray(distortion["powers"]), np.asarray(distortion["coefficients"])],
        )
        fields["holes"] = _restore_component(fields["holes"])
        aperture = DynamicApertureMask(distortion=mapping, **fields)
    else:
        raise ValueError(f"Unknown aperture kind: {kind}")
    aperture = aperture.set(
        ["corners", "opds", *amplitude],
        [
            corners,
            opds,
            *[None if v is None else np.asarray(v) for v in amplitude.values()],
        ],
    )
    # Static metadata cannot be changed with tree_at; these are construction values.
    object.__setattr__(
        aperture, "alias", None if aliases is None else tuple(tuple(x) for x in aliases)
    )
    object.__setattr__(
        aperture,
        "opd_sharing",
        (None if sharing is None else tuple((key, tuple(group)) for key, group in sharing)),
    )
    aperture.validate_aliases()
    return aperture


# amigo 0.2 writes and reads schema v3. v2 records hold amigo 0.1's instrument (a GaussianJitter
# blur, the 0.1 ramp network and read model), which amigo 0.2 cannot build: amigo 0.1 loads them.
SCHEMA = "amigo-instrument-v3"
SCHEMA_V2 = "amigo-instrument-v2"
RECORD_FIELDS = {
    "schema",
    "settings",
    "instrument",
    "aperture",
    "coefficients",
    "filters",
    "fourier_epochs",
    "defocus",
    "sigmas",
}

# Array paths of the 0.2 detector, ramp and read model, fixed by schema v3. The read model's IPC
# kernel is stored separately, as its layer is absent when IPC is off.
INSTRUMENT_PATHS = (
    ("resampler.rotation", "resampler.anisotropy", "convolution.sigma"),
    (
        "FF",
        "SRF",
        "nn_weights",
        "kernel_model.knots",
        "kernel_model.powers",
        "kernel_model.spatial_encoder.layers.1.weight",
    ),
    ("dark_current", "gain", "non_linearity", "amplifier.one_on_fs"),
)
SETTINGS = {
    "detector": {"kernel_size", "oversample"},
    "ramp": {
        "time_steps",
        "norm",
        "bleed",
        "use_charge",
        "use_SRF",
        "hidden_width",
        "n_hidden_layers",
    },
    "read": {"ipc", "per_pixel_dark", "poly_order", "amplifier_axis"},
}


def _instrument_settings(detector, ramp, read):
    if tuple(type(item) for item in (detector, ramp, read)) != (
        LinearDetector,
        NonLinearRamp,
        ReadModel,
    ):
        raise TypeError("Custom instrument classes need an explicit record adapter")
    blur = detector.layers["convolution"]
    # the bleed network's convolution weights, (out, in, 3, 3): hidden layers, then the output
    shapes = ramp.kernel_model.bleed_encoder.structure.shapes
    return {
        "detector": {"kernel_size": int(blur.kernel_size), "oversample": int(blur.oversample)},
        "ramp": {
            "time_steps": int(ramp.time_steps),
            "norm": int(ramp.norm),
            "bleed": bool(ramp.bleed),
            "use_charge": bool(ramp.use_charge),
            "use_SRF": bool(ramp.ff_model.use_SRF),
            "hidden_width": int(shapes[0][0]),
            "n_hidden_layers": len(shapes) - 1,
        },
        "read": {
            "ipc": "IPC" in read.layers,
            "per_pixel_dark": bool(read.layers["read"].per_pixel),
            "poly_order": read.non_linearity.shape[0] + 1,
            "amplifier_axis": read.amplifier.axis,
        },
    }


def _construct_instrument(settings):
    if set(settings) != set(SETTINGS) or any(
        set(settings[part]) != SETTINGS[part] for part in SETTINGS
    ):
        raise ValueError("Unknown instrument construction settings")
    detector = LinearDetector(
        kernel_size=settings["detector"]["kernel_size"],
        kernel_osamp=settings["detector"]["oversample"],
    )
    ramp_settings = dict(settings["ramp"])
    use_SRF = ramp_settings.pop("use_SRF")
    ramp = NonLinearRamp(**ramp_settings).set("ff_model.use_SRF", use_SRF)
    read_settings = settings["read"]
    read = ReadModel(ipc=read_settings["ipc"], per_pixel_dark=read_settings["per_pixel_dark"])
    read = read.set(
        ["layers.pixel_non_linearity", "layers.amplifier"],
        [
            PixelNonLinearity(poly_order=read_settings["poly_order"]),
            Amplifier(axis=read_settings["amplifier_axis"]),
        ],
    )
    return detector, ramp, read


def _fourier_epochs(aperture, coefficients, epochs):
    """Validate {epoch: "YYYY-MM-DD"} against a Fourier part shared by "fourier_epoch"."""
    by_epoch = tuple(dict(aperture.opd_sharing or ()).get("fourier_coeffs", ())) == (
        "fourier_epoch",
    )
    if not by_epoch:
        if epochs:
            raise ValueError("Fourier epochs need a Fourier part shared by 'fourier_epoch'")
        return None
    sets = coefficients.get("fourier_coeffs")
    if not epochs or not isinstance(sets, dict) or set(epochs) != set(sets):
        raise ValueError(
            "A Fourier part shared by 'fourier_epoch' needs one date per Fourier set "
            "(fourier_epochs, {epoch: 'YYYY-MM-DD'})"
        )
    return {str(epoch): date.fromisoformat(epochs[epoch]).isoformat() for epoch in sorted(epochs)}


def nearest_epoch(day, epochs):
    """The epoch in ``epochs`` ({epoch: "YYYY-MM-DD"}) nearest ``day``; ties go to the earlier."""
    if day is None:
        raise ValueError("An exposure without a date (DATE-OBS) cannot choose a Fourier epoch")
    day = date.fromisoformat(day)
    return min(
        epochs,
        key=lambda epoch: (abs((date.fromisoformat(epochs[epoch]) - day).days), epochs[epoch]),
    )


def assign_fourier_epochs(exposures, epochs):
    """Copies of ``exposures``, each using the Fourier set of the epoch nearest its own date.

    A v3 record holds one Fourier wavefront set per calibrator epoch (``fourier_epochs`` from
    ``load_calibration``); its Fourier part is shared by the exposures' ``fourier_epoch``.
    Build the model and fit with the returned exposures, not the originals.
    """
    assigned = []
    for exp in exposures:
        exp = copy(exp)
        object.__setattr__(exp, "fourier_epoch", nearest_epoch(exp.date, epochs))
        assigned.append(exp)
    return assigned


def _structure(instrument):
    """Validate supported architecture, including activation functions and static flags."""
    return fingerprint(eqx.filter(instrument, lambda value: not eqx.is_array(value)))


def record_path(path):
    """Where the instrument record at ``path`` lives: ``<path>.json`` and ``<path>.npz``.

    A path naming an existing folder means the record ``calibration`` inside it,
    which is how records were saved before 2026-11.
    """
    path = Path(path)
    return path / "calibration" if path.is_dir() else path


def save_calibration(path, model, fourier_epochs=None):
    """Save ``model``'s instrument as the record ``<path>.json`` + ``<path>.npz`` (schema v3).

    For example ``save_calibration("amigo_files/v0.2/base_model", model)``. An existing folder
    gets the record ``calibration`` inside it (``record_path``). The record holds the
    instrument's construction settings and arrays; the calibrated aperture (hole metrology and
    wavefront modes) with its coefficient groups; ``params["defocus"]`` and the per-filter
    detector blurs ``params["sigmas"]``. A Fourier part shared by ``"fourier_epoch"`` needs
    ``fourier_epochs``, the date of each of its sets: ``{epoch: "YYYY-MM-DD"}``.
    """
    path = record_path(path)
    instrument = (model.detector, model.ramp_model, model.read)
    settings = _instrument_settings(*instrument)
    if _structure(instrument) != _structure(_construct_instrument(settings)):
        raise ValueError("Instrument architecture differs from the standard calibration schema")
    values = [
        {key: item.get(key) for key in paths} for item, paths in zip(instrument, INSTRUMENT_PATHS)
    ]
    values[2]["IPC"] = instrument[2].IPC.ipc if settings["read"]["ipc"] else None
    calibration = model.opd_calibration()
    save_record(
        path,
        {
            "schema": SCHEMA,
            "settings": settings,
            "instrument": values,
            "aperture": _aperture(calibration.aperture),
            "coefficients": calibration.coefficients,
            "filters": calibration.filters,
            "fourier_epochs": _fourier_epochs(
                calibration.aperture, calibration.coefficients, fourier_epochs
            ),
            "defocus": model.params["defocus"],
            "sigmas": model.params["sigmas"],
        },
    )


def load_calibration(path):
    """Load the instrument record at ``path`` as model-construction inputs.

    ``path`` names the record (``<path>.json`` + ``<path>.npz``, e.g.
    ``amigo_files/v0.2/base_model``), or a folder holding one saved before 2026-11
    (``record_path``). Keys match the arguments they feed: ``detector``, ``ramp_model``,
    ``read`` and ``vis_model`` for ``AmigoModel``; ``opd_calibration`` for ``AMIOptics``;
    ``defocus`` and ``sigmas`` map keys (filters, in a base model) to starting values for
    ``params.defocus`` and ``params.sigmas``; ``fourier_epochs`` dates each Fourier set
    (``assign_fourier_epochs``), or is None; ``schema`` is the record's. ``vis_model`` is a
    ``PlaceholderVisModel`` (no visibility modulation) until the pipeline builds a basis; it is
    not part of the saved record. amigo 0.1's records (v2) load with amigo 0.1 only.
    """
    record = load_record(record_path(path))
    if record.get("schema") == SCHEMA_V2:
        raise ValueError(
            f"{path} is an amigo 0.1 record ({SCHEMA_V2}): load it with amigo 0.1. amigo 0.2 "
            f"builds only its own instrument ({SCHEMA})"
        )
    if set(record) != RECORD_FIELDS or record["schema"] != SCHEMA:
        raise ValueError(f"Expected a plain-value {SCHEMA} instrument record")
    if len(record["instrument"]) != 3:
        raise ValueError("Calibration requires detector, ramp and read records")
    instrument = list(_construct_instrument(record["settings"]))
    for i, (item, paths, values) in enumerate(
        zip(instrument, INSTRUMENT_PATHS, record["instrument"])
    ):
        if set(values) != set(paths) | ({"IPC"} if i == 2 else set()):
            raise ValueError("Calibration array fields do not match the instrument schema")
        for key in paths:
            value = values[key]
            reference = item.get(key)
            if reference is not None and (value is None or np.shape(reference) != np.shape(value)):
                raise ValueError(f"Calibration shape mismatch for {key}")
            item = item.set(key, None if value is None else np.asarray(value))
        if i == 2 and (values["IPC"] is not None) != record["settings"]["read"]["ipc"]:
            raise ValueError("The IPC kernel does not match the record's IPC setting")
        if i == 2 and values["IPC"] is not None:
            item = item.set("layers.IPC.ipc", np.asarray(values["IPC"]))
        instrument[i] = item
    calibration = OPDCalibration(
        _restore_aperture(record["aperture"]),
        jax.tree.map(
            np.asarray,
            {_RENAMED.get(k, k): v for k, v in record["coefficients"].items()},
        ),
        filters=tuple(record["filters"]),
    )
    epochs = _fourier_epochs(
        calibration.aperture, calibration.coefficients, record["fourier_epochs"]
    )
    detector, ramp_model, read = instrument

    return {
        "schema": record["schema"],
        "detector": detector,
        "ramp_model": ramp_model,
        "read": read,
        "opd_calibration": calibration,
        "defocus": record["defocus"],
        "sigmas": record["sigmas"],
        "fourier_epochs": epochs,
        "vis_model": PlaceholderVisModel(),
    }
