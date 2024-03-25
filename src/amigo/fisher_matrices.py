from amigo.FIM import FIM
from amigo.modelling import model_fn
from amigo.stats import variance_model, posterior
from amigo.modelling import model_fn
import jax.numpy as np
from tqdm.notebook import tqdm

# def get_fisher(
#     model,
#     exp,
#     params=None,
#     self_fisher=True,
#     read_noise=10.0,
#     diag=False,  # Return the diagonal of the FIM
#     **kwargs,
# ):

#     if self_fisher:
#         psf, variance = variance_model(model, exp, model_fn)
#         exp = exp.set(["data", "variance"], [psf, variance])

#     if params is not None:
#         return FIM(model, params, posterior, exp, model_fn, diag=diag, **kwargs)
#     return FIM(model, exp.params, posterior, exp, model_fn, diag=diag, **kwargs)


def get_fisher(
    model,
    exp,
    params,
    self_fisher=True,
    read_noise=10.0,
    true_read_noise=False,
    diag=False,  # Return the diagonal of the FIM
    photon=False,
    per_pix=False,
    **kwargs,
):

    # Remove 1/fs for photon model
    if photon:
        model = model.multiply(f'one_on_fs.{exp.key}', 0.)
        psf = model_fn(model, exp, **kwargs)
        variance = np.abs(psf) / exp.nints
        exp = exp.set(["data", "variance"], [psf, variance])


    # if self_fisher or photon:
    if self_fisher:
        psf, variance = variance_model(
            model, exp, true_read_noise=true_read_noise, read_noise=read_noise
        )
        exp = exp.set(["data", "variance"], [psf, variance])

    return FIM(model, params, posterior, exp, diag=diag, per_pix=per_pix, photon=photon, **kwargs)


import jax.tree_util as jtu
import time


def recombine(matrices):
    lengths = np.array(jtu.tree_map(lambda x: len(x), matrices))
    mats = np.zeros((lengths.sum(), lengths.sum()))

    idx = 0
    for mat, length in zip(matrices, lengths):
        mats = mats.at[idx : idx + length, idx : idx + length].set(mat)
        idx += length

    return mats

def fix_diag(mat, thresh=1e-16, replace=1.):
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


def calc_local_fisher(
    model,
    exposure,
    self_fisher=True,
    photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args, 
        model,
        exposure,
        self_fisher=self_fisher, 
        photon=photon, 
        per_pix=per_pix,
        read_noise=read_noise, 
        true_read_noise=true_read_noise, 
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
    fisher_matrix = fisher_fn(params=exp_params)
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
    one_on_f_fisher = fisher_fn(params=[f"one_on_fs.{key}"])
    one_on_f_fisher *= np.eye(one_on_f_fisher.shape[0])
    print(f"Noise Time: {time.time() - t0:.2f}")

    local_fisher = recombine(
        # [pos_flux_fisher, abb_fisher_matrix, bias_fisher, one_on_f_fisher]
        [pos_flux_fisher, abb_fisher_matrix, one_on_f_fisher]
        # [pos_flux_fisher, abb_fisher_matrix]
    )
    local_fisher = fix_diag(local_fisher)

    return local_fisher

def calculate_mask_fisher(
    model,
    exposure,
    self_fisher=True,
    photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args, 
        model,
        exposure,
        self_fisher=self_fisher, 
        photon=photon, 
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
    fisher_mask = fisher_fn(params=global_params)
    # mask = np.zeros((N, N))
    mask = np.ones((N, N))
    mask = mask.at[:14, :14].set(np.eye(14))
    # mask = mask.at[14:, 14:].set(1.0)
    fisher_mask *= mask
    print(f"Mask Time: {time.time() - t0:.2f}")

    return fisher_mask

def calculate_bfe_fisher(
    model,
    exposure,
    self_fisher=True,
    photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args, 
        model,
        exposure,
        self_fisher=self_fisher, 
        photon=photon, 
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

# def calc_visibility_fisher(model, exposures, self_fisher=True):

#     key_paths = model.amplitudes.keys()

#     visibility_dict = {
#         'amplitudes': {},
#         'phases': {}
#     }

#     from tqdm.notebook import tqdm
#     for key_path in tqdm(key_paths):
#         exposures_in = []
#         for exp in exposures:
#             if f"{exp.star}_{exp.filter}" == key_path:
#                 exposures_in.append(exp)

#         amplitudes = []
#         phases = []
#         for exp in exposures_in:
#             fisher_fn = lambda *args, **kwargs: get_fisher(
#                 model, exp, *args, self_fisher=self_fisher, **kwargs
#             )
#             amplitudes.append(fisher_fn(params=["amplitudes." + key_path]))
#             phases.append(fisher_fn(params=["phases." + key_path]))

#         fisher_amplitudes = np.sum(np.array(amplitudes), axis=0)
#         fisher_phases = np.sum(np.array(phases), axis=0)

#         visibility_dict['amplitudes'][key_path] = fisher_amplitudes
#         visibility_dict['phases'][key_path] = fisher_phases

#     return visibility_dict



def calc_visibility_fisher(
    model,
    exposures,
    self_fisher=True,
    photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    key_paths = list(model.amplitudes.keys())

    visibility_dict = {"amplitudes": {}, "phases": {}}

    base_fisher_fn = lambda *args, **kwargs: get_fisher(
        *args, 
        self_fisher=self_fisher, 
        photon=photon, 
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

            # print("Fisher Matrix")
            # plt.imshow(-ampl_arr)
            # plt.colorbar()
            # plt.show()
            amplitudes.append(ampl_arr)
            phases.append(fisher_fn(params=["phases." + key_path]))

        fisher_amplitudes = np.sum(np.array(amplitudes), axis=0)
        fisher_phases = np.sum(np.array(phases), axis=0)

        visibility_dict["amplitudes"][key_path] = fisher_amplitudes
        visibility_dict["phases"][key_path] = fisher_phases

    return visibility_dict


def calculate_simple_bfe_fisher(
    model,
    exposure,
    self_fisher=True,
    photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args,
        model,
        exposure,
        self_fisher=self_fisher,
        photon=photon,
        per_pix=per_pix,
        read_noise=read_noise,
        true_read_noise=true_read_noise,
        **kwargs,
    )
    t0 = time.time()
    fisher_bfe = fisher_fn(params=["BFE.coeffs"])
    print(f"BFE Time: {time.time() - t0:.2f}")
    return fix_diag(fisher_bfe)


def calculate_gradient_bfe_fisher(
    model,
    exposure,
    self_fisher=True,
    photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args,
        model,
        exposure,
        self_fisher=self_fisher,
        photon=photon,
        per_pix=per_pix,
        read_noise=read_noise,
        true_read_noise=true_read_noise,
        **kwargs,
    )
    t0 = time.time()
    fisher_bfe = fisher_fn(params=["BFE.coeffs"])
    print(f"BFE Time: {time.time() - t0:.2f}")
    return fix_diag(fisher_bfe)

def calculate_SRF_fisher(
    model,
    exposure,
    self_fisher=True,
    photon=False,
    per_pix=False,
    read_noise=10.0,
    true_read_noise=False,
):

    fisher_fn = lambda *args, **kwargs: get_fisher(
        *args, 
        model,
        exposure,
        self_fisher=self_fisher, 
        photon=photon, 
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