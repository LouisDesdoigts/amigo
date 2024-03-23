import equinox as eqx
import jax
import jax.numpy as np
from .fisher_matrices import calc_local_fisher, calculate_mask_fisher, calculate_bfe_fisher
from tqdm.notebook import tqdm


class MatrixMapper(eqx.Module):
    """Class to map matrices to and across pytree leaves."""

    params: list[str] = eqx.field(static=True)
    step_type: str = eqx.field(static=True)
    fisher_matrix: jax.Array
    step_matrix: jax.Array

    def __init__(self, params, fisher_matrix, step_type):
        self.params = params
        self.fisher_matrix = fisher_matrix

        if step_type not in ["matrix", "vector"]:
            raise ValueError("Step type must be 'matrix' or 'vec'")
        self.step_type = step_type

        if self.step_type == "matrix":
            self.step_matrix = -np.linalg.inv(self.fisher_matrix)
        else:
            self.step_matrix = -1.0 / np.diag(self.fisher_matrix)

    def update(self, model, vec):
        idx = 0
        for param in self.params:
            n = model.get(param).size
            leaf = vec[idx : idx + n].reshape(model.get(param).shape)
            model = model.set(param, leaf)
            idx += n
        return model

    def to_vec(self, model):
        return np.concatenate([model.get(p).flatten() for p in self.params])

    def apply(self, model):
        if self.step_type == "matrix":
            return self.update(model, self.step_matrix @ self.to_vec(model))
        elif self.step_type == "vector":
            return self.update(model, self.step_matrix * self.to_vec(model))
        else:
            raise ValueError("Step type must be 'matrix' or 'vector'")

    def get_cross_terms(self):
        raise NotImplementedError("Method not implemented")

    def get_diagonal_terms(self):
        raise NotImplementedError("Method not implemented")


class LocalStepMapper(MatrixMapper):

    def __init__(self, model, exposure):
        self.params = [
            f"positions.{exposure.key}",
            f"fluxes.{exposure.key}",
            f"aberrations.{exposure.key}",
            f"one_on_fs.{exposure.key}",
        ]

        self.step_type = "matrix"
        self.fisher_matrix = calc_local_fisher(model, exposure, self_fisher=True)
        self.step_matrix = -np.linalg.inv(self.fisher_matrix)

    def recalculate(self, model, exposure):
        fisher_matrix = calc_local_fisher(model, exposure, self_fisher=True)
        step_matrix = -np.linalg.inv(fisher_matrix)
        return self.set(["fisher_matrix", fisher_matrix], ["step_matrix", step_matrix])


class MaskStepMapper(MatrixMapper):
    fisher_matrix: jax.Array

    def __init__(self, model, exposures):

        self.step_type = "matrix"
        self.params = [
            "pupil_mask.holes",
            "pupil_mask.f2f",
            "rotation",
        ]

        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(calculate_mask_fisher(model, exp, self_fisher=True))

        self.fisher_matrix = np.array(fisher_matrices)
        self.step_matrix = -np.linalg.inv(self.fisher_matrix.sum(0))

    def recalculate(self, model, exposures):
        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(calculate_mask_fisher(model, exp, self_fisher=True))

        fisher_matrix = np.array(fisher_matrices)
        step_matrix = -np.linalg.inv(fisher_matrix.sum(0))
        return self.set(["fisher_matrix", fisher_matrix], ["step_matrix", step_matrix])


class BFEStepMapper(MatrixMapper):

    def __init__(self, model, exposures):

        self.step_type = "vector"
        self.params = ["BFE.linear", "BFE.quadratic"]

        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_diag = calculate_bfe_fisher(model, exp, self_fisher=True)
            fisher_diag = fisher_diag.at[np.where(np.abs(fisher_diag) < 1e-16)].set(1e-16)
            fisher_matrices.append(np.eye(fisher_diag.size) * fisher_diag[None, :])

        self.fisher_matrix = np.array(fisher_matrices)
        self.step_matrix = -1 / np.diag(self.fisher_matrix.sum(0))

    def recalculate(self, model, exposures):
        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(calculate_bfe_fisher(model, exp, self_fisher=True))

        fisher_matrix = np.array(fisher_matrices)
        step_matrix = -1 / np.diag(fisher_matrix.sum(0))
        return self.set(["fisher_matrix", fisher_matrix], ["step_matrix", step_matrix])


# TODO: Implement this class
class VisibilityMapper(MatrixMapper):
    pass
