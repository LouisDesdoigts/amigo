import equinox as eqx
import zodiax as zdx
import jax
import jax.numpy as np
from .fisher_matrices import (
    calc_local_fisher,
    calculate_mask_fisher,
    calculate_bfe_fisher,
    calc_visibility_fisher,
    calculate_simple_bfe_fisher,
    calculate_gradient_bfe_fisher,
    calculate_SRF_fisher,
    calculate_anisotropy_fisher,
)
import jax.tree_util as jtu
from tqdm.notebook import tqdm


# class MatrixMapper(eqx.Module):
class MatrixMapper(zdx.Base):
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
        self.fisher_matrix = calc_local_fisher(model, exposure, self_fisher=True, per_pix=True)
        self.step_matrix = -np.linalg.inv(self.fisher_matrix)

    def recalculate(self, model, exposure):
        fisher_matrix = calc_local_fisher(model, exposure, self_fisher=True, per_pix=True)
        step_matrix = -np.linalg.inv(fisher_matrix)
        return self.set(["fisher_matrix", "step_matrix"], [fisher_matrix, step_matrix])


class MaskStepMapper(MatrixMapper):
    fisher_matrix: jax.Array

    def __init__(self, model, exposures):

        self.step_type = "matrix"
        self.params = [
            "pupil_mask.holes",
            # "pupil_mask.f2f",
            "compression",
            "rotation",
            "shear",
        ]

        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(
                calculate_mask_fisher(model, exp, self_fisher=True, per_pix=True)
            )

        self.fisher_matrix = np.array(fisher_matrices)
        self.step_matrix = -np.linalg.inv(self.fisher_matrix.sum(0))

    def recalculate(self, model, exposures):
        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(
                calculate_mask_fisher(model, exp, self_fisher=True, per_pix=True)
            )

        fisher_matrix = np.array(fisher_matrices)
        step_matrix = -np.linalg.inv(fisher_matrix.sum(0))
        return self.set(["fisher_matrix", "step_matrix"], [fisher_matrix, step_matrix])


class BFEStepMapper(MatrixMapper):

    def __init__(self, model, exposures):

        self.step_type = "vector"
        self.params = ["BFE.linear", "BFE.quadratic"]

        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_diag = calculate_bfe_fisher(model, exp, self_fisher=True, per_pix=True)
            fisher_diag = fisher_diag.at[np.where(np.abs(fisher_diag) < 1e-16)].set(1e-16)
            fisher_matrices.append(np.eye(fisher_diag.size) * fisher_diag[None, :])

        self.fisher_matrix = np.array(fisher_matrices)
        self.step_matrix = -1 / np.diag(self.fisher_matrix.sum(0))

    def recalculate(self, model, exposures):
        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(
                calculate_bfe_fisher(model, exp, self_fisher=True, per_pix=True)
            )

        fisher_matrix = np.array(fisher_matrices)
        step_matrix = -1 / np.diag(fisher_matrix.sum(0))
        return self.set(["fisher_matrix", "step_matrix"], [fisher_matrix, step_matrix])


class VisibilityMapper(MatrixMapper):

    def __init__(self, model, exposures, diag=False):

        self.step_type = "matrix"

        visibility_dict = calc_visibility_fisher(model, exposures, self_fisher=True, per_pix=True)
        #
        params = []
        for key in visibility_dict.keys():
            for subkey in visibility_dict[key].keys():
                params.append(f"{key}.{subkey}")
        self.params = params

        self.fisher_matrix = visibility_dict
        if diag:
            step_matrix = jtu.tree_map(lambda x: x * np.eye(x.shape[0]), self.fisher_matrix)
            # self.step_matrix = jtu.tree_map(lambda x: -1.0 / np.diag(x), self.fisher_matrix)
            self.step_matrix = jtu.tree_map(lambda x: -np.linalg.inv(x), step_matrix)
        else:
            self.step_matrix = jtu.tree_map(lambda x: -np.linalg.inv(x), self.fisher_matrix)

    def recalculate(self, model, exposures):

        fisher_dict = calc_visibility_fisher(model, exposures, self_fisher=True, per_pix=True)
        step_matrix = jtu.tree_map(lambda x: -np.linalg.inv(x), fisher_dict)

        return self.set(["fisher_matrix", fisher_dict], ["step_matrix", step_matrix])

    def update(self, model, vec):
        raise NotImplementedError("This method is not implemented for VisibilityMapper")

    def to_vec(self, model):
        raise NotImplementedError("This method is not implemented for VisibilityMapper")

    def apply(self, model):
        apply_fn = lambda step_mat, vals: step_mat @ vals
        new_ampl = jtu.tree_map(apply_fn, self.step_matrix["amplitudes"], model.amplitudes)
        new_phases = jtu.tree_map(apply_fn, self.step_matrix["phases"], model.phases)
        return model.set(["amplitudes", "phases"], [new_ampl, new_phases])


class SimpleBFEStepMapper(MatrixMapper):
    step_type: str

    def __init__(self, model, exposures, step_type="matrix", diag=False, mixed_fisher=True):
        self.step_type = step_type
        self.params = ["BFE.coeffs"]

        fisher_matrix, step_matrix = self.calc_fisher(
            model, exposures, step_type, mixed_fisher, diag
        )
        self.fisher_matrix = fisher_matrix
        self.step_matrix = step_matrix

    def calc_fisher(self, model, exposures, step_type="matrix", diag=False, mixed_fisher=True):
        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_diag = calculate_simple_bfe_fisher(model, exp, self_fisher=True, per_pix=True)
            fisher_matrices.append(fisher_diag)

        fisher_matrix = np.array(fisher_matrices)
        if step_type == "matrix":

            # Diag supersedes all
            if diag:
                I = np.eye(fisher_matrix.shape[-1])
                step_matrix = -np.linalg.inv((I * fisher_matrix.sum(0)))

            elif mixed_fisher:

                step_matrix = -np.linalg.inv(fisher_matrix.sum(0))
            else:
                # Assume 12 pixel kernel for now
                full_fisher_matrix = fisher_matrix.sum(0)
                N = 12**2
                ncoeff = len(full_fisher_matrix) // N

                mask = np.zeros((N * ncoeff, N * ncoeff))
                idx = 0
                for i in range(ncoeff):
                    mask = mask.at[idx : idx + N, idx : idx + N].set(np.ones((N, N)))
                    idx += N
                step_matrix = -np.linalg.inv(full_fisher_matrix * mask)

        else:
            step_matrix = -1 / np.diag(fisher_matrix.sum(0))

        return fisher_matrix, step_matrix

    def recalculate(self, model, exposures, step_type="matrix", diag=False, mixed_fisher=True):
        fisher_matrix, step_matrix = self.calc_fisher(
            model, exposures, step_type, mixed_fisher, diag
        )
        return self.set(
            ["fisher_matrix", "step_matrix", "step_type"], [fisher_matrix, step_matrix, step_type]
        )


class GradientBFEStepMapper(MatrixMapper):
    step_type: str

    def __init__(self, model, exposures, step_type="matrix", diag=False, mixed_fisher=True):
        self.step_type = step_type
        self.params = ["BFE.coeffs"]

        fisher_matrix, step_matrix = self.calc_fisher(
            model, exposures, step_type, mixed_fisher, diag
        )
        self.fisher_matrix = fisher_matrix
        self.step_matrix = step_matrix

    def calc_fisher(self, model, exposures, step_type="matrix", diag=False, mixed_fisher=True):
        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_diag = calculate_gradient_bfe_fisher(model, exp, self_fisher=True, per_pix=True)
            fisher_matrices.append(fisher_diag)

        fisher_matrix = np.array(fisher_matrices)
        if step_type == "matrix":

            # Diag supersedes all
            if diag:
                I = np.eye(fisher_matrix.shape[-1])
                step_matrix = -np.linalg.inv((I * fisher_matrix.sum(0)))

            elif mixed_fisher:
                step_matrix = -np.linalg.inv(fisher_matrix.sum(0))
            else:
                from amigo.fisher_matrices import create_block_diagonal

                n_marginal = model.BFE.oksize**2
                block = create_block_diagonal(fisher_matrix.shape[-1], n_marginal)
                step_matrix = -np.linalg.inv(fisher_matrix.sum(0) * block)

        else:
            step_matrix = -1 / np.diag(fisher_matrix.sum(0))

        return fisher_matrix, step_matrix

    def recalculate(self, model, exposures, step_type="matrix", diag=False, mixed_fisher=True):
        fisher_matrix, step_matrix = self.calc_fisher(
            model, exposures, step_type, mixed_fisher, diag
        )
        return self.set(
            ["fisher_matrix", "step_matrix", "step_type"], [fisher_matrix, step_matrix, step_type]
        )


from amigo.step_mappers import MatrixMapper
from tqdm.notebook import tqdm


class SRFStepMapper(MatrixMapper):
    fisher_matrix: jax.Array

    def __init__(self, model, exposures):

        self.step_type = "matrix"
        self.params = ["sensitivity.SRF"]

        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(
                calculate_SRF_fisher(model, exp, self_fisher=True, per_pix=True)
            )

        self.fisher_matrix = np.array(fisher_matrices)
        self.step_matrix = -np.linalg.inv(self.fisher_matrix.sum(0))

    def recalculate(self, model, exposures):
        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(
                calculate_SRF_fisher(model, exp, self_fisher=True, per_pix=True)
            )
        fisher_matrix = np.array(fisher_matrices)
        step_matrix = -np.linalg.inv(fisher_matrix.sum(0))
        return self.set(["fisher_matrix", "step_matrix"], [fisher_matrix, step_matrix])


class AnisotropyStepMapper(MatrixMapper):
    fisher_matrix: jax.Array

    def __init__(self, model, exposures):

        self.step_type = "matrix"
        self.params = ["anisotropy.compression"]

        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(
                calculate_anisotropy_fisher(model, exp, self_fisher=True, per_pix=True)
            )

        self.fisher_matrix = np.array(fisher_matrices)
        self.step_matrix = -np.linalg.inv(self.fisher_matrix.sum(0))

    def recalculate(self, model, exposures):
        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(
                calculate_anisotropy_fisher(model, exp, self_fisher=True, per_pix=True)
            )
        fisher_matrix = np.array(fisher_matrices)
        step_matrix = -np.linalg.inv(fisher_matrix.sum(0))
        return self.set(["fisher_matrix", "step_matrix"], [fisher_matrix, step_matrix])
