import numpy as np
from scipy.stats import rice, foldnorm
from dw_tools import wilson


def prepare_for_aniso(ds, rlp, F, err_F):
    """
    Put parameters for the gradient function in a dictionary to keep things tidy.
    """
    params = {}
    params["F"] = F
    params["err_F"] = err_F
    params["rlp"] = rlp
    params["centric"] = ds["CENTRIC"].to_numpy()
    params["eps"] = ds["EPSILON"].to_numpy()
    return params


def ds_dp_aniso(p, rlp, eps):
    """
    Derivative of the scale factor with respect to the optimization para,eters used below.
    Input arguments:
        p      (7,) NumPy array of optimization parameters
        rlp    (N,3) NumPy array of reciprocal lattice point coordinates

    Returns: value of the gradient of each reflection's scale factor with respect to each
    parameter that is optimized.
    """
    B = np.asarray([[p[0], p[1], p[2]], [p[1], p[3], p[4]], [p[2], p[4], p[5]]])
    exp_B = np.exp(-np.sum(rlp * (B @ rlp.transpose()).transpose(), axis=1, dtype=float))
    grad = np.zeros((rlp.shape[0], 7))
    grad[:, 0] = -p[6] * exp_B * rlp[:, 0] ** 2
    grad[:, 3] = -p[6] * exp_B * rlp[:, 1] ** 2
    grad[:, 5] = -p[6] * exp_B * rlp[:, 2] ** 2
    grad[:, 1] = -p[6] * exp_B * rlp[:, 0] * rlp[:, 1] * 2
    grad[:, 2] = -p[6] * exp_B * rlp[:, 0] * rlp[:, 2] * 2
    grad[:, 4] = -p[6] * exp_B * rlp[:, 1] * rlp[:, 2] * 2
    grad[:, 6] = exp_B
    grad = grad / np.repeat(np.sqrt(eps.reshape(-1, 1)), 7, axis=1)
    return grad


def dEsc_dE(E, E0, nH):
    """First derivative of the Hill transform used"""
    return (1 + (E / E0) ** nH) ** (-1 - (1 / nH))


def ds_sc_ds(s, lam):
    """First derivative of the transform used to keep scale factors positive"""
    return np.exp(lam * s) / (1 + np.exp(lam * s))


def ds_sc_dp_aniso(s, lam, p, rlp, eps):
    """
    Calculate d(s_sc)/dp = d(s_sc)/ds * ds/dp
    """
    ds_sc_dp_grad = ds_sc_ds(s, lam).reshape(-1, 1).repeat(p.shape[0], axis=1) * ds_dp_aniso(
        p, rlp, eps
    )
    #     np.set_printoptions(precision=3)
    return ds_sc_dp_grad


def dE_dp_aniso(F, s, lam, p, rlp, eps):
    # E = F*s_sc (not inverse)
    return (F * ds_sc_ds(s, lam)).reshape(-1, 1).repeat(p.shape[0], axis=1) * ds_dp_aniso(
        p, rlp, eps
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


def dLn_dp_aniso(params, E_sc, E, E0, s, s_sc, lam, nH, n_p, p):  # n_p as above
    PE_c = (
        wilson.wilson_dist_normalized(E_sc, centric=True, nargout=1)
        .reshape(-1, 1)
        .repeat(n_p, axis=1)
    )
    PE_ac = (
        wilson.wilson_dist_normalized(E_sc, centric=False, nargout=1)
        .reshape(-1, 1)
        .repeat(n_p, axis=1)
    )

    #     dE_dp(F, s, lam, hkl_index, N123, n)
    dEdp_res = dE_dp_aniso(params["F"], s, lam, p, params["rlp"], params["eps"])
    return (
        (dPE_dp_centric(E_sc, E, E0, nH, n_p, dEdp_res) / PE_c)
        * params["centric"].reshape(-1, 1).repeat(n_p, axis=1)
        + (dPE_dp_acentric(E_sc, E, E0, nH, n_p, dEdp_res) / PE_ac)
        * (1 - params["centric"]).reshape(-1, 1).repeat(n_p, axis=1)
        + (
            ds_sc_dp_aniso(s, lam, p, params["rlp"], params["eps"])
            / s_sc.reshape(-1, 1).repeat(n_p, axis=1)
        )
    )


def anisotropic_scaling_to_1_wilson_loss(
    p, ds, label="FP", suffix="", nH=4, E0=20, lam=10, grad=True, weights=False, nargout=1
):
    """
    Scales a dataset with keys FP_1, SIGFP_1, such that EP = a * F* exp(-r*T B r*))/sqrt(eps),
    with r* the reciprocal lattice vectors, encoded as rs_a_1, rs_b_1, rs_c_1,
    epsilon the multiplicity, and {a * exp(-r*T B r*))} playing the role of 1/sqrt(Sigma)
    We'll assume that measurement error is irrelevant to observed spread. With this assumption,
    we can use the Wilson distributions to formulate our loss function. Since the loss function
    should apply to F, not E, we need to take into account the Jacobian dF/dE.

    Input arguments:
        p :       list or vector with parameters such that B = [[p0, p1, p2], [p1, p3, p4], [p2, p4, p5]]
                   and p[6] is a scalar prefactor for structure factor amplitudes
        ds:       data frame with, at least, columns "rs_a_1", "rs_b_1", "rs_c_1", "EPSILON"
        label:    column label for structure factor amplitdes to be scaled. default: "FP".
                   a column with label {"SIG" + label+suffix} should also exist, as should columns
                   "rs_a", "rs_b", "rs_c" + suffix.
        suffix:   default ""
        nH:       Hill coefficient used in transforming E to keep it within range where the Wilson PDFs are significantly nonzero (default: 4)
        E0:       "knee" of the Hill transform (default: 20)
        lam:      parameter of the transform keeping the scale factors positive (default: 20)
        grad:     Boolean: whether to return a (loss, gradient) tuple; default: True
        weights:  Whether to down-weight contributions to the loss function from observations with large error (default: False)
        nargouts: determines whether to only output residuals (1) or all return values (>1)

    Returns:
        loss:     loss function value
        EP_corr:  scaled structure factor amplitudes
        err_EP:   error of EP_corr
    """
    label = label + suffix
    rlp = ds[["rs_a" + suffix, "rs_b" + suffix, "rs_c" + suffix]].to_numpy()
    F = ds[label].to_numpy()
    err_F = ds["SIG" + label].to_numpy()
    params = prepare_for_aniso(ds, rlp, F, err_F)

    B = np.asarray([[p[0], p[1], p[2]], [p[1], p[3], p[4]], [p[2], p[4], p[5]]])
    B_corr = np.exp(-np.sum(rlp * (B @ rlp.transpose()).transpose(), axis=1, dtype=float))
    total_scale = p[6] * B_corr / np.sqrt(ds["EPSILON"].to_numpy())
    total_scale_sc = (1 / lam) * np.log(
        1 + np.exp(lam * total_scale)
    )  # make sure total_scale is positive

    EP_corr = total_scale_sc * F
    err_EP = total_scale_sc * err_F

    EP_corr_sc = (EP_corr**nH / (1 + (EP_corr / E0) ** nH)) ** (
        1 / nH
    )  # we need this nonlinearity to stop running
    # out of the range
    # where the loss functions have meaningfull probability
    # total_scale is the Jacobian dE/dF that comes from
    # the conversion from p(E) to p(F)
    #    print(np.amin(EP_corr_sc))
    #    print(np.amax(EP_corr_sc))
    #    print(np.amin(total_scale_sc))
    #    print(np.amax(total_scale_sc))
    if weights == True:
        w = 1 / (1 + err_F**2 / 0.1**2)
    else:
        w = np.ones(err_F.shape)

    loss_E_ac = -2 * rice.logpdf(EP_corr_sc, 0, 0, np.sqrt(0.5)) - 2 * np.log(total_scale_sc)
    loss_E_c = -2 * foldnorm.logpdf(EP_corr_sc, 0, 0, 1) - 2 * np.log(total_scale_sc)

    loss_E_ac = w * loss_E_ac
    loss_E_c = w * loss_E_c

    loss = np.sum(loss_E_ac[ds["CENTRIC"] == False]) + np.sum(loss_E_c[ds["CENTRIC"] == True])

    grad_per_obs = dLn_dp_aniso(
        params,
        E_sc=EP_corr_sc,
        E=EP_corr,
        E0=E0,
        s=total_scale,
        s_sc=total_scale_sc,
        lam=lam,
        nH=nH,
        n_p=len(p),
        p=p,
    )
    grad_tot = -2 * np.sum(np.repeat(w.reshape(-1, 1), 7, 1) * grad_per_obs, axis=0)
    if (nargout == 1) & grad:
        return (loss, grad_tot)
    else:
        if (nargout == 1) & ~grad:
            return loss
        else:
            return loss, EP_corr_sc, err_EP


def anisotropic_scaling_to_1(p, ds, label="FP", suffix="", mode="F", nargout=1):
    """
    Scales a dataset with keys FP_1, SIGFP_1, to optimally fit F = a * exp(-r*T B r*))*sqrt(eps),
    with r* the reciprocal lattice vectors, encoded as rs_a_1, rs_b_1, rs_c_1,
    epsilon the multiplicity, and {a * exp(-r*T B r*))} playing the role of 1/sqrt(Sigma).
    Residuals are weighted by SIGFP, which may not be so appropriate...

    Input arguments:
        p :       list or vector with parameters such that B = [[p0, p1, p2], [p1, p3, p4], [p2, p4, p5]]
                   and p[6] is a scalar prefactor for structure factor amplitudes
        ds:       data frame with, at least, columns "rs_a_1", "rs_b_1", "rs_c_1", "EPSILON"
        label:    column label for structure factor amplitdes to be scaled. default: "FP".
                   a column with label {"SIG" + label + suffix} should also exist, as should columns
                   with rs_a, rs_b, rs_c + suffix.
        suffix:   dataset suffix for datasets containing multiple.
        mode :    mode="F" (default) will scale structure factor amplitudes to have amean of 1; "I" will do so for
                  intensities, accounting for multiplicity.
        nargouts: determines whether to only output residuals (1) or all return values (>1)

    Returns:
        residual: an error-weighted residual
        EP_corr:  scaled structure factor amplitudes
        err_EP:   error of EP_corr
    """
    label = label + suffix
    rlp = ds[["rs_a" + suffix, "rs_b" + suffix, "rs_c" + suffix]].to_numpy()

    B = np.asarray([[p[0], p[1], p[2]], [p[1], p[3], p[4]], [p[2], p[4], p[5]]])
    B_corr = np.exp(-np.sum(rlp * (B @ rlp.transpose()).transpose(), axis=1))
    F = ds[label].to_numpy()
    total_scale = np.abs(p[6]) * B_corr / np.sqrt(ds["EPSILON"].to_numpy())
    EP_corr = total_scale * F

    err_F = ds["SIG" + label].to_numpy()
    err_EP = err_F * total_scale
    if mode == "F":
        if nargout > 1:
            print("mode: F")
        residual = (
            1.0 - EP_corr
        )  # amplitudes scaled to 1 (not intensities, which would be more appropriate)
        residual = residual / err_EP
    else:  # assume normalize as intensities
        if nargout > 1:
            print("mode: I")
        residual = (
            1.0 - EP_corr**2
        )  # amplitudes scaled to 1 (not intensities, which would be more appropriate)
        err_I = 2 * F * err_F  # error propagation
        err_EP2 = err_I * total_scale**2
        residual = residual / err_EP2

    if nargout == 1:
        return residual
    else:
        return residual, EP_corr, err_EP
