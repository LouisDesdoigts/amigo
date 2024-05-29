import equinox as eqx
import zodiax as zdx
import jax
import jax.numpy as np
from amigo.FIM import FIM
from amigo.stats import variance_model, posterior
import jax.tree_util as jtu
from tqdm.notebook import tqdm
import time


def get_fisher(
    model,
    exp,
    params,
    self_fisher=True,
    read_noise=10.0,
    true_read_noise=False,
    diag=False,  # Return the diagonal of the FIM
    # photon=False,
    per_pix=False,
    save_ram=True,
    vmapped=False,
    **kwargs,
):

    if self_fisher:
        psf, variance = variance_model(
            model, exp, true_read_noise=true_read_noise, read_noise=read_noise
        )
        exp = exp.set(["data", "variance"], [psf, variance])

    return FIM(
        model,
        params,
        posterior,
        exp,
        diag=diag,
        save_ram=save_ram,
        vmapped=vmapped,
        per_pix=per_pix,
        **kwargs,
    )


def recombine(matrices):
    lengths = np.array(jtu.tree_map(lambda x: len(x), matrices))
    mats = np.zeros((lengths.sum(), lengths.sum()))

    idx = 0
    for mat, length in zip(matrices, lengths):
        mats = mats.at[idx : idx + length, idx : idx + length].set(mat)
        idx += length

    return mats


def fix_diag(mat, thresh=1e-16, replace=1.0):
    """
    Some parameters have no effect on the PSF, for example if some pixels are nans. This
    leads to zero gradients on the diagonal of the fisher matrix corresponding to
    those values. If any diagonal entries are zero the inversion return a nan matrix.

    We can fix this by setting the diagonals to one. This should have no effect on the
    result as all correlation terms will also be zero, and the gradients of those
    parameters will also be zero, so they will have no effect
    """
    # Get the indices
    lin_inds = np.arange(len(mat))
    inds = np.array([lin_inds, lin_inds])

    # Fix the diagonal
    diag = np.diag(mat)
    fixed_diag = diag.at[np.where(np.abs(diag) <= thresh)].set(replace)

    # Return the fixed matrix
    return mat.at[*inds].set(fixed_diag)


def create_block_diagonal(size, block_size):
    # Initialize an empty matrix of zeros
    matrix = np.zeros((size, size))

    # Iterate over the matrix in steps of block_size
    for i in range(0, size, block_size):
        # Determine the end of the current block
        end = min(i + block_size, size)

        # Set the elements in the current block to 1
        matrix = matrix.at[i:end, i:end].set(1)

    return matrix


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


def calc_local_fisher(
    model,
    exposure,
    self_fisher=True,
    # photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
    save_ram=True,
    vmapped=False,
    **kwargs,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args,
        model,
        exposure,
        self_fisher=self_fisher,
        # photon=photon,
        per_pix=per_pix,
        read_noise=read_noise,
        true_read_noise=true_read_noise,
        save_ram=save_ram,
        vmapped=vmapped,
        **kwargs,
    )

    # model, exposure, self_fisher=True, per_pix=True):

    # def fisher_fn(*args, **kwargs):
    #     return get_fisher(model, exposure, per_pix=per_pix, *args, **kwargs)

    key = exposure.key

    # Number of aberrations
    nmirror = exposure.aberrations.shape[0]
    nzern = exposure.aberrations.shape[1]
    n = nmirror * nzern

    """Locals"""
    # Position, flux, aberrations ~30 seconds
    exp_params = [
        f"positions.{key}",
        f"fluxes.{key}",
        f"aberrations.{key}",
    ]
    t0 = time.time()
    fisher_matrix = fisher_fn(params=exp_params, **kwargs)
    print(f"Main Time: {time.time() - t0:.2f}")

    # Position and flux
    pos_flux_fisher = fisher_matrix[:3, :3]

    # Aberrations - Only want to marginalise over each mirror separately (covariances)
    aberration_matrix = fisher_matrix[-n:, -n:]
    abb_fisher_matrix = np.zeros((n, n))
    for i in range(nmirror):
        s, e = i * nzern, (i + 1) * nzern
        fmat = aberration_matrix[s:e, s:e]
        abb_fisher_matrix = abb_fisher_matrix.at[s:e, s:e].set(fmat)

    # print(abb_fisher_matrix)

    # # Noise
    t0 = time.time()
    one_on_f_fisher = fisher_fn(params=[f"one_on_fs.{key}"], **kwargs)
    one_on_f_fisher *= np.eye(one_on_f_fisher.shape[0])
    print(f"Noise Time: {time.time() - t0:.2f}")

    # # Coherence
    # coherence_fisher = fisher_fn(params=[f"coherence.{key}"])

    # Recombine
    local_fisher = recombine(
        # [pos_flux_fisher, abb_fisher_matrix, bias_fisher, one_on_f_fisher]
        [pos_flux_fisher, abb_fisher_matrix, one_on_f_fisher]
        # [pos_flux_fisher, abb_fisher_matrix, one_on_f_fisher, coherence_fisher]
    )
    local_fisher = fix_diag(local_fisher)

    return local_fisher


class LocalStepMapper(MatrixMapper):

    def __init__(self, model, exposure, save_ram=True, vmapped=False, **kwargs):
        self.params = [
            f"positions.{exposure.key}",
            f"fluxes.{exposure.key}",
            f"aberrations.{exposure.key}",
            f"one_on_fs.{exposure.key}",
            # f"coherence.{exposure.key}",
        ]

        self.step_type = "matrix"
        self.fisher_matrix = calc_local_fisher(
            model, exposure, self_fisher=True, per_pix=True, save_ram=save_ram, vmapped=vmapped, **kwargs,
        )
        self.step_matrix = -np.linalg.inv(self.fisher_matrix)

    def recalculate(self, model, exposure):
        fisher_matrix = calc_local_fisher(model, exposure, self_fisher=True, per_pix=True)
        step_matrix = -np.linalg.inv(fisher_matrix)
        return self.set(["fisher_matrix", "step_matrix"], [fisher_matrix, step_matrix])


def calculate_mask_fisher(
    model,
    exposure,
    self_fisher=True,
    # photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
    **kwargs,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args,
        model,
        exposure,
        self_fisher=self_fisher,
        # photon=photon,
        per_pix=per_pix,
        read_noise=read_noise,
        true_read_noise=true_read_noise,
        **kwargs,
    )
    global_params = [
        "pupil_mask.holes",
        # "pupil_mask.f2f",
        "compression",
        "rotation",
        "shear",
    ]

    N = np.array(jtu.tree_map(lambda x: x.size, model.get(global_params))).sum()

    # Holes ~1.5 minutes
    t0 = time.time()
    fisher_mask = fisher_fn(params=global_params, **kwargs)
    # mask = np.zeros((N, N))
    mask = np.ones((N, N))
    mask = mask.at[:14, :14].set(np.eye(14))
    # mask = mask.at[14:, 14:].set(1.0)
    fisher_mask *= mask
    print(f"Mask Time: {time.time() - t0:.2f}")

    return fisher_mask


class MaskStepMapper(MatrixMapper):
    fisher_matrix: jax.Array

    def __init__(self, model, exposures, **kwargs):

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
                calculate_mask_fisher(model, exp, self_fisher=True, per_pix=True, **kwargs)
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


def calculate_bfe_fisher(
    model,
    exposure,
    self_fisher=True,
    # photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args,
        model,
        exposure,
        self_fisher=self_fisher,
        # photon=photon,
        per_pix=per_pix,
        read_noise=read_noise,
        true_read_noise=true_read_noise,
        **kwargs,
    )

    # BFE Linear ~10 seconds
    t0 = time.time()
    fisher_linear = fisher_fn(params=["BFE.linear"])
    fisher_linear *= np.eye(fisher_linear.shape[0])
    print(f"BFE Time: {time.time() - t0:.2f}")

    # BFE Qudratic ~10 seconds
    t0 = time.time()
    fisher_quad = fisher_fn(params=["BFE.quadratic"])
    fisher_quad *= np.eye(fisher_quad.shape[0])
    print(f"Quadratic BFE Time: {time.time() - t0:.2f}")

    fisher_bfe = np.concatenate([np.diag(fisher_linear), np.diag(fisher_quad)])
    # fisher_bfe = np.diag(fisher_linear)

    fisher_bfe = fisher_bfe.at[np.where(np.abs(fisher_bfe) < 1e-16)].set(1.0)

    return fisher_bfe


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


def calc_visibility_fisher(
    model,
    exposures,
    self_fisher=True,
    # photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    key_paths = list(model.amplitudes.keys())

    visibility_dict = {"amplitudes": {}, "phases": {}}

    base_fisher_fn = lambda *args, **kwargs: get_fisher(
        *args,
        self_fisher=self_fisher,
        # photon=photon,
        per_pix=per_pix,
        read_noise=read_noise,
        true_read_noise=true_read_noise,
        **kwargs,
    )

    for key_path in tqdm(key_paths):
        exposures_in = []
        for exp in exposures:
            if f"{exp.star}_{exp.filter}" == key_path:
                exposures_in.append(exp)

        amplitudes = []
        phases = []
        for exp in exposures_in:
            fisher_fn = lambda params: base_fisher_fn(model, exp, params)
            ampl_arr = fisher_fn(params=["amplitudes." + key_path])

            amplitudes.append(ampl_arr)
            phases.append(fisher_fn(params=["phases." + key_path]))

        fisher_amplitudes = np.sum(np.array(amplitudes), axis=0)
        fisher_phases = np.sum(np.array(phases), axis=0)

        visibility_dict["amplitudes"][key_path] = fisher_amplitudes
        visibility_dict["phases"][key_path] = fisher_phases

    return visibility_dict


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


def calculate_simple_bfe_fisher(
    model,
    exposure,
    self_fisher=True,
    # photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args,
        model,
        exposure,
        self_fisher=self_fisher,
        # photon=photon,
        per_pix=per_pix,
        read_noise=read_noise,
        true_read_noise=true_read_noise,
        **kwargs,
    )
    t0 = time.time()
    fisher_bfe = fisher_fn(params=["BFE.coeffs"])
    print(f"BFE Time: {time.time() - t0:.2f}")
    return fix_diag(fisher_bfe)


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


def calculate_gradient_bfe_fisher(
    model,
    exposure,
    self_fisher=True,
    # photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args,
        model,
        exposure,
        self_fisher=self_fisher,
        # photon=photon,
        per_pix=per_pix,
        read_noise=read_noise,
        true_read_noise=true_read_noise,
        **kwargs,
    )
    t0 = time.time()
    fisher_bfe = fisher_fn(params=["BFE.coeffs"])
    print(f"BFE Time: {time.time() - t0:.2f}")
    return fix_diag(fisher_bfe)


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


def calculate_SRF_fisher(
    model,
    exposure,
    self_fisher=True,
    # photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args,
        model,
        exposure,
        self_fisher=self_fisher,
        # photon=photon,
        per_pix=per_pix,
        read_noise=read_noise,
        true_read_noise=true_read_noise,
        **kwargs,
    )

    t0 = time.time()
    fisher_SRF = fisher_fn(params=["sensitivity.SRF"])
    fisher_SRF = np.eye(fisher_SRF.shape[0]) * fisher_SRF
    print(f"SRF Time: {time.time() - t0:.2f}")
    return fisher_SRF


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


def calculate_anisotropy_fisher(
    model,
    exposure,
    self_fisher=True,
    # photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args,
        model,
        exposure,
        self_fisher=self_fisher,
        # photon=photon,
        per_pix=per_pix,
        read_noise=read_noise,
        true_read_noise=true_read_noise,
        **kwargs,
    )

    t0 = time.time()
    fisher_ansio = fisher_fn(params=["anisotropy.compression"])
    # fisher_ansio = np.eye(fisher_SRF.shape[0]) * fisher_SRF
    print(f"Anisotropy Time: {time.time() - t0:.2f}")
    return fisher_ansio


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


def calculate_reflectivity_fisher(
    model,
    exposure,
    self_fisher=True,
    # photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args,
        model,
        exposure,
        self_fisher=self_fisher,
        # photon=photon,
        per_pix=per_pix,
        read_noise=read_noise,
        true_read_noise=true_read_noise,
        **kwargs,
    )
    # Holes ~1.5 minutes
    t0 = time.time()
    fisher_reflectivity = fisher_fn(params=["holes.reflectivity"])
    print(f"Reflectivity Time: {time.time() - t0:.2f}")

    return fisher_reflectivity


class ReflectivityStepMapper(MatrixMapper):
    fisher_matrix: jax.Array

    def __init__(self, model, exposures):

        self.step_type = "matrix"
        self.params = ["holes.reflectivity"]

        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(
                calculate_reflectivity_fisher(model, exp, self_fisher=True, per_pix=True)
            )

        self.fisher_matrix = np.array(fisher_matrices)
        self.step_matrix = -np.linalg.inv(self.fisher_matrix.sum(0))

    def recalculate(self, model, exposures):
        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(
                calculate_reflectivity_fisher(model, exp, self_fisher=True, per_pix=True)
            )

        fisher_matrix = np.array(fisher_matrices)
        step_matrix = -np.linalg.inv(fisher_matrix.sum(0))
        return self.set(["fisher_matrix", "step_matrix"], [fisher_matrix, step_matrix])


def calculate_dark_current_fisher(
    model,
    exposure,
    self_fisher=True,
    # photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args,
        model,
        exposure,
        self_fisher=self_fisher,
        # photon=photon,
        per_pix=per_pix,
        read_noise=read_noise,
        true_read_noise=true_read_noise,
        **kwargs,
    )
    # Holes ~1.5 minutes
    t0 = time.time()
    fisher_reflectivity = fisher_fn(params=["dark_current"])
    print(f"Dark current Time: {time.time() - t0:.2f}")

    return fisher_reflectivity


class DarkCurrentStepMapper(MatrixMapper):
    fisher_matrix: jax.Array

    def __init__(self, model, exposures):

        self.step_type = "matrix"
        self.params = ["dark_current"]

        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(
                calculate_dark_current_fisher(model, exp, self_fisher=True, per_pix=True)
            )

        self.fisher_matrix = np.array(fisher_matrices)
        self.step_matrix = -np.linalg.inv(self.fisher_matrix.sum(0))

    def recalculate(self, model, exposures):
        fisher_matrices = []
        for exp in tqdm(exposures):
            fisher_matrices.append(
                calculate_dark_current_fisher(model, exp, self_fisher=True, per_pix=True)
            )

        fisher_matrix = np.array(fisher_matrices)
        step_matrix = -np.linalg.inv(fisher_matrix.sum(0))
        return self.set(["fisher_matrix", "step_matrix"], [fisher_matrix, step_matrix])


#
