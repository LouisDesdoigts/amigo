"""ALL of this will be put into Zodiax down the line"""

import jax
import zodiax as zdx
import jax.numpy as np
import jax.tree_util as jtu
from jax import jit, grad, jvp, linearize, lax


# def hvp(f, primals, tangents):
#     return jvp(grad(f), primals, tangents)[1]


def hessian(f, x):
    _, hvp = linearize(grad(f), x)
    # Jit the sub-function here since it is called many times
    # TODO: Test effect on speed
    hvp = jit(hvp)
    # basis = np.eye(np.prod(np.array(x.shape))).reshape(-1, *x.shape)
    basis = np.eye(x.size).reshape(-1, *x.shape)
    return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)


# def FIM(
#     pytree,
#     parameters,
#     loglike_fn,
#     *loglike_args,
#     shape_dict={},
#     save_ram=True,
#     **loglike_kwargs,
# ):
#     # Build X vec
#     pytree = zdx.tree.set_array(pytree, parameters)
#     leaves = pytree.get(parameters)

#     # shapes = jtu.tree_map(lambda x: x.shape, leaves)
#     # lengths = jtu.tree_map(lambda x: x.size, leaves)

#     shapes = [leaf.shape for leaf in leaves]
#     lengths = [leaf.size for leaf in leaves]
#     N = np.array(lengths).sum()
#     X = np.zeros(N)

#     # shapes, lengths = zdx.bayes._shapes_and_lengths(pytree, parameters, shape_dict)
#     # X = np.zeros(zdx.bayes._lengths_to_N(lengths))

#     # Build function to calculate FIM and calculate
#     def calc_fim(X):
#         parametric_pytree = _perturb(X, pytree, parameters, shapes, lengths)
#         return loglike_fn(parametric_pytree, *loglike_args, **loglike_kwargs)

#     if save_ram:
#         return hessian(calc_fim, X)
#     return jax.hessian(calc_fim)(X)


# def _perturb(X, pytree, parameters, shapes, lengths):
#     n, xs = 0, []
#     if isinstance(parameters, str):
#         parameters = [parameters]
#     indexes = range(len(parameters))

#     for i, param, shape, length in zip(indexes, parameters, shapes, lengths):
#         if length == 1:
#             xs.append(X[i + n])
#         else:
#             xs.append(lax.dynamic_slice(X, (i + n,), (length,)).reshape(shape))
#             n += length - 1

#     return pytree.add(parameters, xs)


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
