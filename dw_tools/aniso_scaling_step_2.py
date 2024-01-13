import numpy as np
from scipy.stats import rice, foldnorm
from . import wilson


def make_mnp_index(n):
    """
    Convert between a 1D and 3D array of indices for Fourier components
    """
    mnp_index = np.zeros((n**3, 3))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                mnp_index[i * n**2 + j * n + k] = [i, j, k]
    return mnp_index


def prepare_for_FFT(ds, n, label):
    """
    Put parameters for the gradient function in a dictionary to keep things tidy.
    """
    rlp = np.array(ds.index.to_list())  # [["H","K","L"]].to_numpy() -- so Miller indices
    F = ds[
        label
    ].to_numpy()  # this is likely already pre-normalized, but we'll call it F for clarity
    err_F = ds["SIG" + label].to_numpy()  # same

    hkl_min = np.amin(rlp, axis=0)  # find the range of points
    hkl_max = np.amax(rlp, axis=0)  #
    L = hkl_max - hkl_min + 1  #
    #   L       = power_log(L)                 # round up to next power of 2
    # print(rlp.shape)
    # print(hkl_min.shape)
    hkl_ind = rlp - np.repeat(hkl_min.reshape(-1, 3), len(ds), axis=0)
    params = {}
    params["n"] = n
    params["L"] = L
    params["hkl_ind"] = hkl_ind
    params["F"] = F
    params["err_F"] = err_F
    params["centric"] = ds["CENTRIC"].to_numpy()
    params["mnp_ind"] = make_mnp_index(n)
    # print(rlp)
    # print(hkl_min)
    # print(hkl_max)
    # print(L)
    # print(hkl_ind)
    return params


def ds_dp(hkl_ind, N123, n):
    """
    Derivative of the scale factor with respect to the Fourier coefficients used below.
    Input arguments:
        hkl_ind    (N,3) NumPy array of hkl's for each reflection
        N123       Numpy array of the three dimensions of the Fourier grid
        n          Order of the Fourier series expansion

    Returns: value of the gradient of each reflection's scale factor with respect to each
    Fourier coefficient that is optimized.
    """
    h_sc = hkl_ind / np.repeat(N123.reshape(1, -1), hkl_ind.shape[0], axis=0)
    m = make_mnp_index(n).transpose()
    ds_da = np.cos(
        2 * np.pi * (h_sc @ m)
    )  # the prefactor np.prod(N123) should cancel out in rescaling in the fcn
    ds_db = -np.sin(2 * np.pi * (h_sc @ m))  # don't prepend j. happens elsewhere
    grad = np.concatenate((ds_da, ds_db), axis=1)
    #     print(grad.shape) num obs x num params
    return grad


def dEsc_dE(E, E0, nH):
    """First derivative of the Hill transform used"""
    return (1 + (E / E0) ** nH) ** (-1 - (1 / nH))


def ds_sc_ds(s, lam):
    """First derivative of the transform used to keep scale factors positive"""
    return np.exp(lam * s) / (1 + np.exp(lam * s))


def ds_sc_dp(s, lam, hkl_index, mnp_index, N123, n):
    """
    Calculate d(s_sc)/dp = d(s_sc)/ds * ds/dp
    """
    ds_sc_dp_grad = ds_sc_ds(s, lam).reshape(-1, 1).repeat(2 * n**3, axis=1) * ds_dp(
        hkl_index, N123, n
    )
    #     np.set_printoptions(precision=3)
    return ds_sc_dp_grad


def dE_dp(F, s, lam, hkl_index, N123, n):
    # E = F*s_sc (not inverse)
    return (F * ds_sc_ds(s, lam)).reshape(-1, 1).repeat(2 * n**3, axis=1) * ds_dp(
        hkl_index, N123, n
    )


def dPE_dp_centric(E_sc, E, E0, nH, n_p, dEdp_res):  # np=2*n**3 or p.shape[0]
    """dP(E)/dp = dP(E)/dE * dE/dp, with p the input parameters."""
    return (wilson.dPE_dE_centric(E_sc) * dEsc_dE(E, E0, nH)).reshape(-1, 1).repeat(
        n_p, axis=1
    ) * dEdp_res


def dPE_dp_acentric(E_sc, E, E0, nH, n_p, dEdp_res):
    """dP(E)/dp = dP(E)/dE * dE/dp, with p the input parameters."""
    return (wilson.dPE_dE_acentric(E_sc) * dEsc_dE(E, E0, nH)).reshape(-1, 1).repeat(
        n_p, axis=1
    ) * dEdp_res


def dLn_dp(params, E_sc, E, E0, s, s_sc, lam, nH, n):
    PE_c = (
        wilson.wilson_dist_normalized(E_sc, centric=True, nargout=1)
        .reshape(-1, 1)
        .repeat(2 * n**3, axis=1)
        + 1e-6
    )
    PE_ac = (
        wilson.wilson_dist_normalized(E_sc, centric=False, nargout=1)
        .reshape(-1, 1)
        .repeat(2 * n**3, axis=1)
        + 1e-6
    )

    #     dE_dp(F, s, lam, hkl_index, N123, n)
    dEdp_res = dE_dp(params["F"], s, lam, params["hkl_ind"], params["L"], n)
    return (
        (dPE_dp_centric(E_sc, E, E0, nH, 2 * n**3, dEdp_res) / PE_c)
        * params["centric"].reshape(-1, 1).repeat(2 * n**3, axis=1)
        + (dPE_dp_acentric(E_sc, E, E0, nH, 2 * n**3, dEdp_res) / PE_ac)
        * (1 - params["centric"]).reshape(-1, 1).repeat(2 * n**3, axis=1)
        + (
            ds_sc_dp(s, lam, params["hkl_ind"], params["mnp_ind"], params["L"], n)
            / s_sc.reshape(-1, 1).repeat(2 * n**3, axis=1)
        )
    )


def anisotropic_scaling_to_1_FFT_wilson_loss_fast(
    p_sc, params, bUse=True, correct=True, grad=True, weights=True, nargout=1
):
    """
    Loss function used to calculate corrections to simple anisotropic normalization used above. To do so, we set up a
    grid in reciprocal space (using hkl rather than rlp coordinates) to calculate corrections at each point in
    reciprocal space. The function also calculates the first derivative and can return them as a tuple (f,g) if
    grad=True.

    input arguments:
    p_sc      Input arguments scaled such that p[0] around 1. The left half of p contains the amplitudes of the cosines of
                Fourier components; the right half contains the corresponding amplitudes of the sine components.
    params    Dictionary of parameters prepared by prepare_for_FFT()
    bUse      Boolean array indicating which reflections should be used in evaluating the loss function.
    grad      Boolean, indicates whether to return the gradient as well, in a tuple with the loss function itself
    weights   Boolean, indicates whether to use weights based on input error estimates (before scaling!). Default True.
    nargout   Scalar. If >1, the function will return the value of the loss function, normalized E, the error in those,
                and the scale factors used.

    The code includes two nonlinear transforms: one to keep the scale factors positive (controlled
    by the parameter lam, hardcoded inside, and one to keep the normalized scale factors below E0 (hardcoded).
    The purpose of this second transform is to keep E within the range in which its pdf is sufficiently non-zero.

    """
    E0 = 14  # E0, nH and lam are parameters of the two nonlinear transforms used.
    nH = 6
    lam = 10
    n = params["n"]
    hkl_ind = params["hkl_ind"]
    p = p_sc * np.prod(params["L"])  # to account for IFFT scale
    basis = np.zeros(params["L"], dtype=np.complex_)
    basis[:n, :n, :n] = p[: n**3].reshape(n, n, n) + 1j * p[n**3 :].reshape(n, n, n)

    total_scale_grid = np.fft.ifftn(basis)
    total_scale = (
        total_scale_grid[hkl_ind[:, 0], hkl_ind[:, 1], hkl_ind[:, 2]]
    ).real  # it might be more appropriate to add the negative freq component as a high freq component with neg phase etc.
    total_scale_sc = (1 / lam) * np.log(
        1 + np.exp(lam * total_scale)
    )  # make sure total_scale is positive

    if correct:
        EP_corr = total_scale_sc * params["F"]
        err_EP = total_scale_sc * params["err_F"]
    else:
        EP_corr = params["F"]
        err_EP = params["err_F"]

    if weights == True:
        w = 1 / (1 + params["err_F"] ** 2 / 0.2**2)
    else:
        w = np.ones(params["err_F"].shape)

    EP_corr_sc = (EP_corr**nH / (1 + (EP_corr / E0) ** nH)) ** (
        1 / nH
    )  # we need this nonlinearity to stop running out of the range
    # where the loss functions have meaningfull probability
    # total_scale is the Jacobian dE/dF that comes from
    # the conversion from p(E) to p(F)
    # this is also guaranteed to be >= 0
    loss_E_ac = -2 * rice.logpdf(EP_corr_sc, 0, 0, np.sqrt(0.5)) - 2 * np.log(total_scale)
    loss_E_c = -2 * foldnorm.logpdf(EP_corr_sc, 0, 0, 1) - 2 * np.log(total_scale)
    loss_E_ac = w * loss_E_ac
    loss_E_c = w * loss_E_c

    # be careful with parentheses: [() & bUse]
    loss = np.sum(loss_E_ac[(params["centric"] == False) & bUse]) + np.sum(
        loss_E_c[params["centric"] & bUse]
    )

    if grad:
        per_obs_grad = dLn_dp(
            params,
            E_sc=EP_corr_sc,
            E=EP_corr,
            E0=E0,
            s=total_scale,
            s_sc=total_scale_sc,
            lam=lam,
            nH=nH,
            n=n,
        )
        total_loss_grad = -2 * np.sum(
            np.repeat(w[bUse].reshape(-1, 1), per_obs_grad.shape[1], 1) * per_obs_grad[bUse, :],
            axis=0,
        )
        if not (total_loss_grad.shape[0] == 2 * n**3):
            print(
                "Warning: the output size of the gradients is incorrect. bUse should be an array, not a scalar boolean"
            )
            print(total_loss_grad.shape)

    if (nargout == 1) & grad:
        return (loss, total_loss_grad)
    else:
        if nargout == 1:
            return loss
        else:
            return loss, EP_corr, err_EP, total_scale
