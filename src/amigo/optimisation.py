"""Cholesky residuals, fixed parameter coordinates and fitting routines.

The previous Trainer implementation is kept separately in amigo.legacy.
"""

import json
import equinox as eqx
import jax
import jax.numpy as np
import zodiax as zdx
import optax
import optimistix as optx
from collections.abc import Mapping
from operator import index
from jax import Array, vmap
from jax.scipy.linalg import solve_triangular

# Residuals and objective functions


class Residuals(zdx.Base):
    """Prepared data and fixed covariance factors for a collection of exposures.

    Construct with prepare(exposures). Covariance parameters are not fitted by
    this object: a variable-covariance likelihood needs its own normalization.
    """

    exposures: tuple
    data: tuple
    cholesky: tuple
    size: int = eqx.field(static=True)
    logdet: Array

    def __init__(self, exposures, data, cholesky, size, logdet):
        self.exposures = tuple(exposures)
        self.data = tuple(data)
        self.cholesky = tuple(cholesky)
        self.size = int(size)
        self.logdet = np.asarray(logdet, float)

    def parts(self, model):
        """Yield whitened residuals in pixel/group order, one exposure at a time."""
        solve = vmap(lambda lower, value: solve_triangular(lower, value, lower=True))

        for exp, data, lower in zip(self.exposures, self.data, self.cholesky):
            prediction = exp.to_vec(exp(model))
            yield solve(lower, prediction - data).reshape(-1)

    def __call__(self, model):
        """Return unscaled residuals for statistical information calculations."""
        return np.concatenate(tuple(self.parts(model)))

    def loss(self, model):
        """Mean squared residual over all retained pixels and group differences."""
        return sum(np.vdot(r, r) for r in self.parts(model)) / self.size

    def least_squares(self, model):
        """Residuals for half-squared-norm solvers, matching loss(model)."""
        return np.sqrt(2.0 / self.size) * self(model)

    def nll(self, model):
        """Gaussian negative log likelihood, including fixed normalization."""
        chi_squared = self.size * self.loss(model)
        normalization = self.logdet + self.size * np.log(2 * np.pi)
        return 0.5 * (chi_squared + normalization)


def _cholesky(data, covariance):
    """Validate supported group covariance and factor it without regularization."""
    if data.ndim != 2 or data.size == 0:
        raise ValueError("Data must have shape (pixels, group differences)")

    shape = (data.shape[0], data.shape[1], data.shape[1])
    if covariance.shape != shape:
        raise ValueError("Covariance does not match supported pixels/groups")

    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(covariance)):
        raise ValueError("Data and covariance must be finite")

    # Accept roundoff asymmetry, but do not repair an invalid covariance.
    transpose = covariance.swapaxes(-1, -2)
    scale = np.maximum(np.max(np.abs(covariance)), np.finfo(covariance.dtype).tiny)
    if np.max(np.abs(covariance - transpose)) > 2e-10 * scale:
        raise ValueError("Covariance is not symmetric")

    lower = np.linalg.cholesky(0.5 * (covariance + transpose))

    # JAX returns NaNs for a failed Cholesky instead of raising an exception.
    if not np.all(np.isfinite(lower)):
        raise ValueError("Covariance must be positive definite")

    return lower


def prepare(exposures):
    """Select valid pixels and cache their covariance factors once per fit."""
    exposures = tuple(exposures)
    if not exposures:
        raise ValueError("Select at least one exposure")

    data, factors = [], []
    size, logdet = 0, np.asarray(0.0)

    for exp in exposures:
        values = np.asarray(exp.to_vec(exp.slopes), float)
        covariance = np.asarray(exp.to_vec(exp.cov), float)
        lower = _cholesky(values, covariance)

        data.append(values)
        factors.append(lower)
        size += values.size
        logdet += 2 * np.log(np.diagonal(lower, axis1=-2, axis2=-1)).sum()

    return Residuals(exposures, data, factors, size, logdet)


# Reference information and parameter normalization


def information(residuals, model, paths, *, nbatches=1, checkpoint=False):
    """Sum unscaled residual JTJ by exposure using Zodiax.

    Optimizer curvature for mean(r**2) is 2*JTJ/N. Recalculate JTJ at the fitted
    endpoint for uncertainty work; an optimizer reference is not a covariance.
    """
    layout = zdx.TreeLayout.from_paths(model, paths)
    origin = layout.flatten(model)

    def inject(values):
        leaves = layout.unflatten(values)
        return model.set(list(layout.paths), [leaves[p] for p in layout.paths])

    # Evaluate each exposure separately without storing a full residual Jacobian.
    matrix = np.zeros((layout.size, layout.size), dtype=origin.dtype)

    for exp, data, lower in zip(residuals.exposures, residuals.data, residuals.cholesky):
        part = Residuals((exp,), (data,), (lower,), data.size, 0.0)
        result = zdx.gauss_newton(
            lambda x: part(inject(x)), origin, nbatches=nbatches, checkpoint=checkpoint
        )
        matrix += result.matrix

    return zdx.Fisher(matrix, layout)


class Block(zdx.Base):
    """Parameter paths with none, diagonal or full normalization.

    An optional metric supplies cached unscaled JTJ or its diagonal, labelled
    with matching paths and units. Full normalization does not discard modes.
    """

    paths: tuple = eqx.field(static=True)
    scaling: str = eqx.field(static=True)
    metric: zdx.Fisher | zdx.Hessian | zdx.TreeVector | None

    def __init__(self, paths, scaling="diagonal", metric=None):
        self.paths = (paths,) if isinstance(paths, str) else tuple(paths)
        self.scaling = scaling
        self.metric = metric

        if not self.paths or scaling not in ("none", "diagonal", "full"):
            raise ValueError("Select paths and scaling: none, diagonal or full")
        if scaling == "none" and metric is not None:
            raise ValueError("Unscaled blocks do not use a metric")


def _transform(metric, layout, scaling, factor, rank_rtol):
    """Build a block's optimizer-to-physical coordinate transform."""
    if not layout.compatible(metric.layout):
        raise ValueError("Information has different parameter coordinates")

    values = metric.vector if isinstance(metric, zdx.TreeVector) else metric.matrix
    if not np.all(np.isfinite(values)):
        raise ValueError("Normalization information must be finite")

    # Match the curvature of the canonical mean-squared objective.
    values = factor * values

    if scaling == "diagonal":
        diagonal = values if values.ndim == 1 else np.diag(values)
        if not np.all(diagonal > 0):
            raise ValueError("Diagonal scaling needs positive column information")
        return 1 / np.sqrt(diagonal)

    # Let Zodiax handle equilibration and decomposition of the full block.
    if values.ndim != 2:
        raise ValueError("Full scaling requires a full information matrix")

    projection = zdx.Fisher(values, layout).projection(
        method="eigh", rtol=rank_rtol, equilibrate=True
    )
    if not np.all(projection.retained):
        raise ValueError("Full normalization would remove parameter directions")
    if not np.all(np.isfinite(projection.matrix)):
        raise ValueError("Full normalization produced nonfinite coordinates")

    return projection.matrix


class Coordinates(zdx.Base):
    """Fixed block mapping z -> model, shared by all stages of a fit.

    Combine paths in one full block to include their mutual correlations.
    Different blocks deliberately omit those cross-correlations.
    """

    model: zdx.Base
    layouts: tuple
    origins: tuple
    transforms: tuple
    size: int = eqx.field(static=True)

    def __init__(self, model, blocks, residuals, *, nbatches=1, checkpoint=False, rank_rtol=1e-12):
        blocks = tuple(blocks)
        paths = [p for block in blocks for p in block.paths]
        if not paths:
            raise ValueError("Select at least one block")

        # Validate the joint selection before creating independent block maps.
        selection = zdx.TreeLayout.from_paths(model, paths)
        layouts, origins, transforms = [], [], []

        for block in blocks:
            layout = zdx.TreeLayout.from_paths(model, block.paths)
            origin = layout.flatten(model)
            transform = np.ones(layout.size, dtype=origin.dtype)

            if block.scaling != "none":
                metric = block.metric
                if metric is None:
                    metric = information(
                        residuals, model, block.paths, nbatches=nbatches, checkpoint=checkpoint
                    )
                transform = _transform(
                    metric, layout, block.scaling, 2.0 / residuals.size, rank_rtol
                )

            layouts.append(layout)
            origins.append(origin)
            transforms.append(transform)

        self.model = model
        self.layouts = tuple(layouts)
        self.origins = tuple(origins)
        self.transforms = tuple(transforms)
        self.size = selection.size

    @property
    def zero(self):
        return np.zeros(self.size, dtype=self.origins[0].dtype)

    def __call__(self, z):
        if z.shape != (self.size,):
            raise ValueError("Optimizer coordinate shape differs from selected blocks")

        paths, values = [], []
        start = 0

        for layout, origin, transform in zip(self.layouts, self.origins, self.transforms):
            block = z[start : start + layout.size]
            step = transform * block if transform.ndim == 1 else transform @ block
            leaves = layout.unflatten(origin + step)

            paths.extend(layout.paths)
            values.extend(leaves[p] for p in layout.paths)
            start += layout.size

        return self.model.set(paths, values)


# Cached optimizer information


def save_information(path, metric, *, context):
    """Save information and reference provenance without changing its units.

    context identifies optical state, spectrum, basis, sampling, weighting,
    objective convention and code/data versions. Trainer.normalise expects
    objective-curvature units; information() returns unscaled statistical JTJ.
    No unit conversion or reference equivalence is inferred when loading.
    """
    if not context:
        raise ValueError("Cache context must describe the reference calculation")

    zdx.save(path, {"metric": metric, "context": json.dumps(context, sort_keys=True)})


def load_information(path, *, context, scale=1.0):
    """Load a matching reference with an explicit information scaling factor.

    For shape parameters, photon noise gives flux-ratio scaling; fixed noise
    gives its square. Mixed noise has no universal scaling. A rescaled optimizer
    cache is not endpoint uncertainty information.
    """
    if not context or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Supply reference context and a positive information scale")

    cached = zdx.load(path)
    if cached["context"] != json.dumps(context, sort_keys=True):
        raise ValueError("Cached reference context does not match")

    metric = cached["metric"]
    field = "vector" if isinstance(metric, zdx.TreeVector) else "matrix"
    values = metric.get(field) * scale
    if not np.all(np.isfinite(values)):
        raise ValueError("Cached information is nonfinite")

    return metric.set(field, values)


# Optimization


class FitResult(zdx.Base):
    """Endpoint and termination details; accepted means the convergence gates passed.

    gradient is the maximum absolute gradient in normalized coordinates.
    A capped/failed solve still returns its endpoint for inspection or continuation.
    criteria records requested thresholds; message preserves the solver's reason.
    GD evaluations counts loss-only calls separately from value-and-gradient calls;
    the latter also evaluate the loss. Other solvers leave evaluations unset.
    """

    model: zdx.Base
    coordinates: Coordinates
    z: Array
    loss: Array
    gradient: Array
    accepted: bool
    status: str = eqx.field(static=True)
    history: tuple
    aux: object = None
    opt_state: object = None
    evaluations: dict | None = eqx.field(default=None, kw_only=True)
    criteria: dict | None = None
    message: str = eqx.field(static=True, default="")

    @property
    def parameters(self):
        """Selected physical parameters at the returned endpoint."""
        return {p: self.model.get(p) for layout in self.coordinates.layouts for p in layout.paths}

    @property
    def parameter_history(self):
        """Recorded physical parameters, stacked along the history axis."""
        rows = [row["parameters"] for row in self.history if "parameters" in row]
        return {p: np.stack([row[p] for row in rows]) for p in rows[0]} if rows else {}

    @property
    def losses(self):
        return np.asarray([row["loss"] for row in self.history])


def _line_search(
    objective, z, value, gradient, step_size, growth, contraction, decrease, max_steps
):
    """Shrink to an Armijo step, or grow a valid step while the loss improves.

    Trials share one evaluation budget. Keep the best valid point if expansion
    fails; if no trial passes, return the original point unchanged.
    """
    slope = -np.vdot(gradient, gradient)
    rate = step_size
    best, best_rate, best_value = z, step_size, value
    accepted = False
    shrinking = False

    for evaluation in range(1, max_steps + 1):
        candidate = z - rate * gradient
        trial = objective(candidate)
        valid = np.isfinite(trial) and trial <= value + decrease * rate * slope

        if valid and (not accepted or trial < best_value):
            best, best_rate, best_value = candidate, rate, trial
            accepted = True
            if shrinking or growth == 1:
                break
            rate *= growth
        elif accepted:
            break
        else:
            shrinking = True
            rate *= contraction

        if not np.isfinite(rate) or rate <= 0:
            break

    return best, best_rate, evaluation, accepted


def armijo(residuals, coordinates, **kwargs):
    """Low-level Armijo fit for the canonical prepared residual objective."""
    objective = lambda z: residuals.loss(coordinates(z))
    return _descent(objective, coordinates, **kwargs)


def _descent(
    objective,
    coordinates,
    *,
    start=None,
    max_steps=100,
    tolerance=1e-3,
    step_size=1.0,
    growth=2.0,
    contraction=0.5,
    sufficient_decrease=1e-4,
    max_search_steps=30,
):
    """Two-way Armijo descent, reusing the previous accepted step size.

    growth=1 disables expansion but still reuses the previous step size.
    Evaluation counts separate loss-only trials from value-and-gradient calls.
    """
    if max_steps < 0 or max_search_steps < 1:
        raise ValueError("Invalid iteration or line-search budget")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("Step size must be finite and positive")
    if not np.isfinite(growth) or growth < 1:
        raise ValueError("Step growth must be finite and at least one")
    if not 0 < contraction < 1 or not 0 < sufficient_decrease < 1:
        raise ValueError("Armijo contraction and decrease must lie between zero and one")

    # Compile evaluations in the fixed optimizer coordinates.
    value_gradient = eqx.filter_jit(jax.value_and_grad(objective))
    value_only = eqx.filter_jit(objective)

    z = coordinates.zero if start is None else np.asarray(start)
    value, gradient = value_gradient(z)
    initial = value
    history = [_record(coordinates, z, value, step=0)]
    status = "iteration_cap"
    rate = step_size
    loss_evaluations = 0
    gradient_evaluations = 1

    for iteration in range(max_steps + 1):
        # Reaching the iteration cap alone does not establish convergence.
        score = np.max(np.abs(gradient))
        if not np.isfinite(value) or not np.isfinite(score):
            status = "nonfinite_objective_or_gradient"
            break
        if score <= tolerance:
            status = "gradient_target"
            break
        if iteration == max_steps:
            break

        candidate, rate, evaluations, accepted = _line_search(
            value_only,
            z,
            value,
            gradient,
            rate,
            growth,
            contraction,
            sufficient_decrease,
            max_search_steps,
        )
        loss_evaluations += evaluations
        if not accepted:
            status = "line_search_failed"
            break

        # Only accepted points update the state and step history.
        z = candidate
        value, gradient = value_gradient(z)
        gradient_evaluations += 1
        history.append(
            _record(
                coordinates, z, value, step=iteration + 1, rate=rate,
                search_evaluations=evaluations,
            )
        )

    accepted = status == "gradient_target" and bool(value <= initial)
    return FitResult(
        coordinates(z), coordinates, z, value, score, accepted, status, tuple(history),
        criteria={"gradient_tolerance": tolerance, "max_steps": max_steps},
        evaluations={"loss_only": loss_evaluations, "value_and_grad": gradient_evaluations},
    )


# Notebook-facing training API


def _groups(model, parameters, exposures):
    """Resolve named parameter groups without duplicating shared exposure paths."""
    if isinstance(parameters, str):
        parameters = [parameters]
    if not isinstance(parameters, Mapping):
        parameters = {name: (name,) for name in parameters}

    groups = {}
    for name, names in parameters.items():
        names = (names,) if isinstance(names, str) else tuple(names)
        paths = []

        for parameter in names:
            candidates = [parameter]
            if exposures and "." not in parameter:
                candidates = [exp.map_param(parameter) for exp in exposures]

            for path in candidates:
                value = model.get(path)
                if isinstance(value, Mapping):
                    leaves = zdx.TreeLayout.from_tree(value).paths
                    paths.extend(path + "." + leaf for leaf in leaves)
                else:
                    paths.append(path)

        groups[name] = tuple(dict.fromkeys(paths))

    if not groups or any(not paths for paths in groups.values()):
        raise ValueError("Select nonempty parameter groups")

    zdx.TreeLayout.from_paths(model, [p for paths in groups.values() for p in paths])
    return groups


def _loss_output(output):
    """Accept either a scalar loss or the established (loss, aux) convention."""
    return output if isinstance(output, tuple) else (output, ())


def _record(coordinates, z, loss, **details):
    model = coordinates(z)
    parameters = {p: model.get(p) for layout in coordinates.layouts for p in layout.paths}
    return dict(loss=loss, parameters=parameters, **details)


def _check_objectives(loss, residual_loss, z):
    """Catch mismatched scalar/residual conventions at a reference point."""
    value, gradient = jax.value_and_grad(loss)(z)
    residual_value, residual_gradient = jax.value_and_grad(residual_loss)(z)
    if not (
        np.allclose(value, residual_value, rtol=1e-6, atol=1e-8)
        and np.allclose(gradient, residual_gradient, rtol=1e-6, atol=1e-8)
    ):
        raise ValueError("loss_fn and residual_fn must describe the same objective")


def _residual_diagonal(residual, origin, nbatches, checkpoint):
    """Sum squared Jacobian columns in batches, without building JTJ."""
    if isinstance(nbatches, bool):
        raise TypeError("nbatches must be a positive integer")
    nbatches = index(nbatches)
    if not 1 <= nbatches <= origin.size:
        raise ValueError("nbatches must lie between one and the parameter count")

    # Reuse one linearisation, with the same memory controls as full information.
    function = jax.checkpoint(residual) if checkpoint else residual
    _, derivative = jax.linearize(jax.jit(function), origin)
    batch_size = (origin.size + nbatches - 1) // nbatches
    columns = np.arange(nbatches * batch_size).reshape(nbatches, batch_size)

    # Only column norms survive each batch; padded directions are zero.
    def step(carry, columns):
        directions = jax.nn.one_hot(columns, origin.size, dtype=origin.dtype)
        responses = jax.vmap(derivative)(directions).reshape(batch_size, -1)
        return carry, np.sum(responses ** 2, axis=1)

    _, diagonal = jax.lax.scan(jax.jit(step), None, columns)
    return diagonal.reshape(-1)[:origin.size]


def _objective_information(
    model, layout, loss, residual, source, nbatches, checkpoint, check_loss=False,
    diagonal=False
):
    """Calculate reference curvature in physical parameter coordinates."""
    origin = layout.flatten(model)

    def inject(values):
        leaves = layout.unflatten(values)
        return model.set(list(layout.paths), [leaves[p] for p in layout.paths])

    if source == "residual":
        if residual is None:
            raise ValueError("Residual information requires residual_fn")
        if check_loss:
            _check_objectives(
                lambda x: loss(inject(x))[0],
                lambda x: 0.5 * np.sum(residual(inject(x)) ** 2),
                origin,
            )
        if diagonal:
            values = _residual_diagonal(
                lambda x: residual(inject(x)), origin, nbatches, checkpoint
            )
            return zdx.TreeVector(values, layout)
        result = zdx.gauss_newton(
            lambda x: residual(inject(x)), origin, nbatches=nbatches, checkpoint=checkpoint
        )
        return zdx.Fisher(result.matrix, layout)

    result = zdx.hessian(
        lambda x: loss(inject(x))[0], origin, nbatches=nbatches, checkpoint=checkpoint
    )
    return zdx.Hessian(result.matrix, layout)


def _normalisation_check(metric, transform, method, source):
    """Report objective information in optimizer coordinates, without repairing it."""
    if isinstance(metric, zdx.TreeVector):
        diagonal = transform ** 2 * metric.vector
        return {
            "objective_source": source,
            "method": method,
            "finite": np.all(np.isfinite(diagonal)),
            "diagonal": diagonal,
            "max_diagonal_error": np.max(np.abs(diagonal - 1)),
            "max_identity_error": None,
        }
    if transform.ndim == 1:
        matrix = transform[:, None] * metric.matrix * transform[None, :]
    else:
        matrix = transform.T @ metric.matrix @ transform

    diagonal = np.diag(matrix)
    return {
        "objective_source": source,
        "method": method,
        "finite": np.all(np.isfinite(matrix)),
        "diagonal": diagonal,
        "max_diagonal_error": np.max(np.abs(diagonal - 1)) if method != "none" else None,
        "max_identity_error": (
            np.max(np.abs(matrix - np.eye(matrix.shape[0]))) if method == "full" else None
        ),
    }


def _check_batches(exposures, batches):
    """Batching may reorder the stored data, but may not change its membership."""
    if not exposures:
        if len(batches) != 1 or batches[0]:
            raise ValueError("Without exposures, use args for data rather than exposure batches")
        return

    remaining = list(exposures)
    for batch in batches:
        if not batch:
            raise ValueError("Exposure batches must not be empty")
        for exp in batch:
            for index, reference in enumerate(remaining):
                if eqx.tree_equal(exp, reference):
                    remaining.pop(index)
                    break
            else:
                raise ValueError("Batches must partition the Trainer's stored exposures")

    if remaining:
        raise ValueError("Batches omit stored exposures")


class Trainer(zdx.Base):
    """Custom objectives and optimizers sharing explicit parameter normalization.

    With exposures: loss_fn(model, exposure, args) -> loss or (loss, aux).
    Without exposures: loss_fn(model, args) -> loss or (loss, aux).
    residual_fn follows the same inputs and returns an array whose half squared
    norm is the objective. Exposure losses are summed; residuals concatenated.

    With neither function, prepare the standard Cholesky objective from supplied
    exposures. These exposures and args are reused by normalization and every fit.
    Custom loss+residual pairs must describe the same objective for LM.
    """

    loss_fn: object
    residual_fn: object
    exposures: tuple
    args: object
    groups: dict
    methods: dict
    sources: dict
    checks: dict
    reference: zdx.Base | None
    reference_exposures: tuple
    reference_args: object
    reference_functions: tuple
    prepared: Residuals | None
    information: dict
    transforms: dict
    layouts: dict

    def __init__(self, loss_fn=None, residual_fn=None, *, exposures=(), args=None):
        self.loss_fn = loss_fn
        self.residual_fn = residual_fn
        self.exposures = tuple(exposures)
        self.args = args
        self.groups = {}
        self.methods = {}
        self.sources = {}
        self.checks = {}
        self.reference = None
        self.reference_exposures = ()
        self.reference_args = None
        self.reference_functions = ()
        self.prepared = None
        if loss_fn is None and residual_fn is None and self.exposures:
            self.prepared = prepare(self.exposures)
        self.information = {}
        self.transforms = {}
        self.layouts = {}

    def _objective(self, exposures=None):
        """Adapt the stored objective; an internal subset supports training batches."""
        exposures = self.exposures if exposures is None else tuple(exposures)
        args = self.args

        if self.loss_fn is None and self.residual_fn is None:
            prepared = self.prepared
            if prepared is None or not eqx.tree_equal(prepared.exposures, exposures):
                prepared = prepare(exposures)
            return lambda model: (prepared.loss(model), ()), prepared.least_squares

        residual = None
        if self.residual_fn is not None:

            def residual(model):
                if not exposures:
                    return np.asarray(self.residual_fn(model, args)).reshape(-1)
                return np.concatenate(
                    [
                        np.asarray(self.residual_fn(model, exp, args)).reshape(-1)
                        for exp in exposures
                    ]
                )

        def loss(model):
            if self.loss_fn is None:
                values = residual(model)
                return 0.5 * np.vdot(values, values), ()
            if not exposures:
                return _loss_output(self.loss_fn(model, args))

            outputs = [_loss_output(self.loss_fn(model, exp, args)) for exp in exposures]
            value = sum(item[0] for item in outputs)
            aux = {exp.key: item[1] for exp, item in zip(exposures, outputs)}
            return value, aux

        return loss, residual

    def normalise(
        self,
        model,
        parameters,
        *,
        method="diagonal",
        preconditioner=None,
        source=None,
        check=False,
        nbatches=1,
        checkpoint=False,
        rank_rtol=1e-12,
    ):
        """Prepare fixed normalization for this Trainer's stored objective.

        parameters accepts names/paths or {group: names/paths} for coupled blocks.
        method is none/diagonal/full, globally or per group. source is residual
        JTJ or scalar Hessian; custom losses default to Hessian, others to JTJ.
        Diagonal residual scaling calculates only squared Jacobian column norms;
        full scaling and scalar Hessians retain the full matrix calculation.

        preconditioner={group: metric} supplies APPROXIMATE cached information in
        objective-curvature units. Missing groups are calculated from the objective.
        check=True compares the transform against freshly calculated objective
        information at model. It reports discrepancies without replacing supplied
        metrics. This is a local curvature check, not proof of global equivalence.
        """
        groups = _groups(model, parameters, self.exposures)
        methods = dict(method) if isinstance(method, Mapping) else dict.fromkeys(groups, method)
        supplied = {} if preconditioner is None else dict(preconditioner)

        if methods.keys() != groups.keys() or supplied.keys() - groups.keys():
            raise ValueError("Normalization methods/cache keys must match selected groups")
        if any(value not in ("none", "diagonal", "full") for value in methods.values()):
            raise ValueError("Normalization must be none, diagonal or full")

        source = source or ("hessian" if self.loss_fn is not None else "residual")
        if source not in ("residual", "hessian"):
            raise ValueError("Information source must be residual or hessian")

        calculate = check or any(
            methods[name] != "none" and name not in supplied for name in groups
        )
        loss, residual = self._objective() if calculate else (None, None)
        metrics, transforms, layouts, sources, checks = {}, {}, {}, {}, {}

        for name, paths in groups.items():
            layout = zdx.TreeLayout.from_paths(model, paths)
            layouts[name] = layout
            objective_metric = None

            # The comparison uses the chosen objective source, not the cache.
            if check or (methods[name] != "none" and name not in supplied):
                objective_metric = _objective_information(
                    model, layout, loss, residual, source, nbatches, checkpoint,
                    check_loss=self.loss_fn is not None,
                    diagonal=methods[name] == "diagonal",
                )

            if methods[name] == "none":
                if name in supplied:
                    raise ValueError("Unscaled groups do not use a preconditioner")
                transforms[name] = np.ones(layout.size, dtype=layout.flatten(model).dtype)
                sources[name] = "identity"
            else:
                metric = supplied[name] if name in supplied else objective_metric
                metrics[name] = metric
                sources[name] = "approximate" if name in supplied else "objective:" + source
                transforms[name] = _transform(metric, layout, methods[name], 1.0, rank_rtol)

            if check:
                checks[name] = _normalisation_check(
                    objective_metric, transforms[name], methods[name], source
                )

        return self.set(
            [
                "groups", "methods", "sources", "checks", "reference",
                "reference_exposures", "reference_args", "reference_functions",
                "information", "transforms", "layouts",
            ],
            [
                groups, methods, sources, checks, model,
                self.exposures, self.args, (self.loss_fn, self.residual_fn),
                metrics, transforms, layouts,
            ],
        )

    def _check_reference(self):
        """Do not silently reuse normalization after changing the objective state."""
        if not self.groups:
            return

        current = (self.loss_fn, self.residual_fn, self.exposures, self.args)
        reference = (*self.reference_functions, self.reference_exposures, self.reference_args)
        if not eqx.tree_equal(current, reference):
            raise ValueError(
                "Objective state changed: call normalise again, or explicitly supply "
                "the old information as a preconditioner on a new Trainer"
            )

    def _coordinates(self, model, parameters):
        """Reuse reference scales at the supplied model; never recalculate them."""
        self._check_reference()
        if parameters is None:
            if not self.groups:
                raise ValueError("Select parameters, or call normalise first")
            groups = self.groups
        elif self.groups and not isinstance(parameters, Mapping):
            names = [parameters] if isinstance(parameters, str) else list(parameters)
            if any(name not in self.groups for name in names):
                raise ValueError("Selected groups were not prepared by normalise")
            groups = {name: self.groups[name] for name in names}
        else:
            groups = _groups(model, parameters, self.exposures)

        # Rebase the origin, preserving the fixed transform and all unselected params.
        blocks = [Block(paths, "none") for paths in groups.values()]
        coordinates = Coordinates(model, blocks, None)
        if not self.groups:
            return groups, coordinates

        transforms = []
        for name, layout in zip(groups, coordinates.layouts):
            if name not in self.layouts or not layout.compatible(self.layouts[name]):
                raise ValueError("Fit parameters differ from the prepared normalization")
            transforms.append(self.transforms[name])

        return groups, coordinates.set("transforms", tuple(transforms))

    def gd(
        self,
        model,
        parameters=None,
        *,
        max_steps=100,
        tolerance=1e-3,
        step_size=1.0,
        growth=2.0,
        contraction=0.5,
        sufficient_decrease=1e-4,
        max_search_steps=30,
    ):
        """Two-way Armijo descent with fixed reference scales and step-size reuse.

        step_size is the initial guess; growth=1 disables expansion. Each search
        gets max_search_steps loss evaluations, including rejected trials.
        """
        _, coordinates = self._coordinates(model, parameters)
        loss, _ = self._objective()
        objective = lambda z: loss(coordinates(z))[0]

        result = _descent(
            objective,
            coordinates,
            max_steps=max_steps,
            tolerance=tolerance,
            step_size=step_size,
            growth=growth,
            contraction=contraction,
            sufficient_decrease=sufficient_decrease,
            max_search_steps=max_search_steps,
        )
        # The endpoint evaluation also supplies custom objective auxiliaries.
        counts = dict(result.evaluations)
        counts["loss_only"] += 1
        return result.set(["aux", "evaluations"], [loss(result.model)[1], counts])

    def bfgs(
        self,
        model,
        parameters=None,
        *,
        max_steps=500,
        tolerance=1e-3,
        rtol=1e-6,
        atol=1e-6,
        solver=None,
        options=None,
    ):
        """BFGS with native rtol/atol and an additional normalized-gradient gate.

        tolerance=None accepts successful native termination without that extra
        gate. Step limits and numerical failures return accepted=False, with the
        solver message and partial endpoint. Invalid inputs still raise.
        solver/options allow direct control; a supplied solver owns rtol/atol.
        """
        return self._refine(
            model,
            parameters,
            "bfgs",
            max_steps,
            tolerance,
            solver if solver is not None else optx.BFGS(rtol=rtol, atol=atol),
            options,
        )

    def lm(
        self,
        model,
        parameters=None,
        *,
        max_steps=500,
        tolerance=1e-3,
        rtol=1e-6,
        atol=1e-6,
        solver=None,
        options=None,
    ):
        """LM on explicit residuals, with the same termination controls as bfgs.

        Never derive residuals from a scalar loss. If both functions are supplied,
        check their values and gradients at the initial and successful final point.
        """
        return self._refine(
            model,
            parameters,
            "lm",
            max_steps,
            tolerance,
            solver if solver is not None else optx.LevenbergMarquardt(rtol=rtol, atol=atol),
            options,
        )

    def _refine(
        self, model, parameters, method, max_steps, tolerance, solver, options
    ):
        if max_steps < 0:
            raise ValueError("Refinement budget must be nonnegative")
        if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError("Gradient tolerance must be finite and nonnegative")

        _, coordinates = self._coordinates(model, parameters)
        loss, residual = self._objective()
        objective = lambda z: loss(coordinates(z))[0]

        if method == "lm":
            if residual is None:
                raise ValueError("LM requires residual_fn, or the default Cholesky objective")
            # LM optimizes the residual objective. A supplied scalar loss is not
            # silently substituted for it or assumed to be globally equivalent.
            residual_objective = lambda z: 0.5 * np.sum(residual(coordinates(z)) ** 2)
            if self.loss_fn is not None:
                _check_objectives(objective, residual_objective, coordinates.zero)
            objective = residual_objective
            solution = optx.least_squares(
                lambda z, _: residual(coordinates(z)),
                solver,
                coordinates.zero,
                options=options,
                max_steps=max_steps,
                throw=False,
            )
        else:
            solution = optx.minimise(
                lambda z, _: objective(z),
                solver,
                coordinates.zero,
                options=options,
                max_steps=max_steps,
                throw=False,
            )

        value, gradient = eqx.filter_jit(jax.value_and_grad(objective))(solution.value)
        score = np.max(np.abs(gradient))
        initial = objective(coordinates.zero)
        finite = bool(np.isfinite(value) & np.isfinite(score) & np.isfinite(initial))
        successful = bool(solution.result == optx.RESULTS.successful)
        capped = bool(
            solution.result == optx.RESULTS.max_steps_reached
            or solution.result == optx.RESULTS.nonlinear_max_steps_reached
        )

        # Expected solver failures are returned with their endpoint, not raised.
        # A step cap or numerical failure is never classified as convergence.
        accepted = False
        if not finite:
            status = "nonfinite_objective_or_gradient"
        elif capped:
            status = "max_steps"
        elif not successful:
            status = "solver_error"
        elif value > initial:
            status = "objective_increased"
        elif tolerance is None:
            status, accepted = "solver_converged", True
        elif score <= tolerance:
            status, accepted = "gradient_target", True
        else:
            status = "solver_terminated_above_target"

        if successful and finite and method == "lm" and self.loss_fn is not None:
            _check_objectives(lambda z: loss(coordinates(z))[0], objective, solution.value)

        criteria = {
            "gradient_tolerance": tolerance,
            "rtol": getattr(solver, "rtol", None),
            "atol": getattr(solver, "atol", None),
            "max_steps": max_steps,
        }
        history = (
            _record(coordinates, coordinates.zero, initial, step=0),
            _record(coordinates, solution.value, value, step=int(solution.stats["num_steps"])),
        )
        final_model = coordinates(solution.value)
        return FitResult(
            final_model,
            coordinates,
            solution.value,
            value,
            score,
            accepted,
            status,
            history,
            loss(final_model)[1],
            solution,
            criteria,
            str(optx.RESULTS[solution.result]),
        )

    def train(
        self,
        model,
        optimisers,
        *,
        epochs,
        batches=None,
        tolerance=None,
        callback=None,
    ):
        """Optax training with one optimizer per named group, including schedules.

        As in the previous Trainer, batch gradients accumulate before one update
        per epoch. All optimizer states live in normalized coordinates. callback
        receives (model, history_row, aux) after each update. With tolerance=None,
        completion reports epochs_complete, not certified convergence.
        """
        if epochs < 0 or not optimisers:
            raise ValueError("Select optimizers and a nonnegative epoch budget")
        if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError("Gradient tolerance must be finite and nonnegative")
        batch_values = list(batches.values()) if isinstance(batches, Mapping) else batches
        if batch_values is None:
            batch_values = [self.exposures]
        else:
            batch_values = [tuple(batch) for batch in batch_values]
        if not batch_values:
            raise ValueError("Select at least one batch")

        _check_batches(self.exposures, batch_values)
        exposures = self.exposures
        groups, coordinates = self._coordinates(model, list(optimisers))
        names = tuple(groups)
        initial = {name: np.zeros_like(origin) for name, origin in zip(names, coordinates.origins)}
        vector = lambda values: np.concatenate([values[name] for name in names])
        model_fn = lambda values: coordinates(vector(values))

        # Canonical MSE uses one global sample count, regardless of batch partition.
        if self.loss_fn is None and self.residual_fn is None:
            prepared = [prepare(batch) for batch in batch_values]
            size = sum(part.size for part in prepared)
            losses = [
                lambda m, part=part: (part.loss(m) * part.size / size, ()) for part in prepared
            ]
        else:
            losses = [self._objective(batch)[0] for batch in batch_values]

        def evaluate(values):
            total = np.asarray(0.0)
            gradient = jax.tree.map(np.zeros_like, values)
            aux = []
            for fn in evaluations:
                (value, extra), grads = fn(values)
                total += value
                gradient = jax.tree.map(np.add, gradient, grads)
                aux.append(extra)

            if self.loss_fn is None:
                aux = ()
            elif exposures:
                aux = {key: value for batch_aux in aux for key, value in batch_aux.items()}
            else:
                aux = aux[0]
            return total, aux, gradient

        # Compile each batch separately, then accumulate before updating Optax.
        evaluations = [
            eqx.filter_jit(
                jax.value_and_grad(lambda values, fn=fn: fn(model_fn(values)), has_aux=True)
            )
            for fn in losses
        ]
        optimizer = optax.multi_transform(dict(optimisers), {name: name for name in names})
        state = optimizer.init(initial)
        update = eqx.filter_jit(optimizer.update)
        values = initial
        value, aux, gradient = evaluate(values)

        # Histories contain physical parameters, not opaque optimizer vectors.
        history = [_record(coordinates, vector(values), value, step=0)]
        status = "epochs_complete"

        for epoch in range(epochs + 1):
            score = np.max(np.abs(vector(gradient)))
            if not np.isfinite(value) or not np.isfinite(score):
                status = "nonfinite_objective_or_gradient"
                break
            if tolerance is not None and score <= tolerance:
                status = "gradient_target"
                break
            if epoch == epochs:
                break

            updates, state = update(gradient, state, values)
            values = optax.apply_updates(values, updates)
            value, aux, gradient = evaluate(values)
            row = _record(coordinates, vector(values), value, step=epoch + 1)
            history.append(row)
            if callback is not None:
                callback(model_fn(values), row, aux)

        return FitResult(
            model_fn(values),
            coordinates,
            vector(values),
            value,
            score,
            status == "gradient_target",
            status,
            tuple(history),
            aux,
            state,
            {"gradient_tolerance": tolerance, "epochs": epochs},
        )
