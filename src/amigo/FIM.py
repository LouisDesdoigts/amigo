"""ALL of this will be put into Zodiax down the line"""

import jax
import zodiax as zdx
import jax.numpy as np
from jax import jit, grad, jvp, linearize, lax, vmap


def hessian(f, x, fast=False):
    if fast:
        # print("Running Vmapped")
        # I think this basically just returns np.eye?
        basis = np.eye(x.size).reshape(-1, *x.shape)

        _, hvp = linearize(grad(f), x)
        hvp = jit(hvp)

        # Compile on first input
        first = np.array([hvp(basis[0])])  # Add empty dim for concatenation

        # Vmap others
        others = vmap(hvp)(basis[1:])

        # Recombine
        return np.stack(np.concatenate([first, others], axis=0)).reshape(x.shape + x.shape)
    else:
        # print("Running non-vmapped")
        _, hvp = linearize(grad(f), x)
        # Jit the sub-function here since it is called many times
        # TODO: Test effect on speed
        hvp = jit(hvp)
        # basis = np.eye(np.prod(np.array(x.shape))).reshape(-1, *x.shape)
        basis = np.eye(x.size).reshape(-1, *x.shape)
        return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)


"""
Some code adapted from here: 
https://github.com/google/jax/issues/3801#issuecomment-662131006

More resources:
https://github.com/google/jax/discussions/8456

I believe this efficient hessian diagonal methods only works _correctly_ if the output
hessian is _naturally_ diagonal, else the results are spurious.
"""


def hvp(f, x, v):
    return jvp(grad(f), (x,), (v,))[1]


# TODO: Update this to take the matrix mapper class
def FIM(
    pytree,
    parameters,
    loglike_fn,
    *loglike_args,
    shape_dict={},
    save_ram=True,
    vmapped=False,
    diag=False,
    **loglike_kwargs,
):
    # Build X vec
    pytree = zdx.tree.set_array(pytree, parameters)

    if len(parameters) == 1:
        parameters = [parameters]

    leaves = [pytree.get(p) for p in parameters]
    shapes = [leaf.shape for leaf in leaves]
    lengths = [leaf.size for leaf in leaves]
    N = np.array(lengths).sum()
    X = np.zeros(N)

    # Build function to calculate FIM and calculate
    def loglike_fn_vec(X):
        parametric_pytree = _perturb(X, pytree, parameters, shapes, lengths)
        return loglike_fn(parametric_pytree, *loglike_args, **loglike_kwargs)

    if diag:
        diag = hvp(loglike_fn_vec, X, np.ones_like(X))
        return np.eye(diag.shape[0]) * diag[:, None]

    if save_ram:
        return hessian(loglike_fn_vec, X)

    if vmapped:
        return hessian(loglike_fn_vec, X, fast=True)

    return jax.hessian(loglike_fn_vec)(X)


def _perturb(X, pytree, parameters, shapes, lengths):
    n, xs = 0, []
    if isinstance(parameters, str):
        parameters = [parameters]
    indexes = range(len(parameters))

    for i, param, shape, length in zip(indexes, parameters, shapes, lengths):
        if length == 1:
            xs.append(X[i + n])
        else:
            xs.append(lax.dynamic_slice(X, (i + n,), (length,)).reshape(shape))
            n += length - 1

    return pytree.add(parameters, xs)
