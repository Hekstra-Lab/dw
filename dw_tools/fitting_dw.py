import numpy as np
from scipy.stats import rice, foldnorm
from scipy.optimize import minimize, least_squares


def fit_ab_ds_residual(p, ds, labels, dHKL_label, dHKL_bin_label):
    """
    Calculates residuals for minimization by fit_ab.
    """
    # print(p)
    nbin = ds[dHKL_bin_label].nunique()

    s = 1 / ds[dHKL_label].to_numpy()
    r_DW = p[0] * np.exp(-p[1] * s ** 2)
    rho_DW = r_DW ** 2  # this is approximate as described above

    varW_acentric = rice.var(0, 0, np.sqrt(0.5))
    varW_centric = foldnorm.var(0, 0, 1)
    tmp1 = ds["SIG" + labels[0]].to_numpy() ** 2  # sigE's
    tmp2 = ds["SIG" + labels[1]].to_numpy() ** 2  #
    quad_var_E = tmp1 + tmp2

    rho_w_err_inv2_ac = (
        (1 / rho_DW ** 2)
        + quad_var_E / varW_acentric
        + tmp1 * tmp2 / varW_acentric ** 2
        + ((1 / rho_DW ** 2) - 1) * np.sqrt(tmp1 * tmp2) / varW_acentric
    )
    rho_w_err_inv2_c = (
        (1 / rho_DW ** 2)
        + quad_var_E / varW_centric
        + tmp1 * tmp2 / varW_centric ** 2
        + ((1 / rho_DW ** 2) - 1) * np.sqrt(tmp1 * tmp2) / varW_centric
    )

    rho_w_err_ac = 1 / np.sqrt(rho_w_err_inv2_ac)
    rho_w_err_c = 1 / np.sqrt(rho_w_err_inv2_c)
    centric = (ds["CENTRIC"] == True).to_numpy()
    ds.loc[ds["CENTRIC"], "rho_w_err"] = rho_w_err_c[centric]
    ds.loc[~ds["CENTRIC"], "rho_w_err"] = rho_w_err_ac[~centric]

    # empirical correlation by bin
    g = ds.groupby(dHKL_bin_label)[labels]
    result = g.corr(method="pearson").unstack().loc[:, (labels[0], labels[1])].to_numpy().flatten()

    # error-weighted avg predicted rho (this way of weighting comes from the definition of Pearson r
    #                                   and sample cov being a sum over samples)
    # in what I've seen so far, weighting has a minimal effect.
    rho_avg = np.zeros((nbin, 1))
    for i in range(ds[dHKL_bin_label].min(), ds[dHKL_bin_label].max() + 1):
        boi = ds[dHKL_bin_label] == i
        rho = ds["rho_w_err"][boi].to_numpy()
        w = (ds["SIG" + labels[0]][boi] * ds["SIG" + labels[1]][boi]).to_numpy()
        rho_avg[i - 1] = np.sum(w * rho) / np.sum(w)

    rho_avg = rho_avg.flatten()
    return result - rho_avg


def fit_ab(ds, labels=["EP_1", "EP_2"], dHKL_label="dHKL_1", dHKL_bin_label="dHKL_bin"):
    # a is constrained to be in [0,1], b to be in [0,1e6] (frivolous upper bound)
    result = least_squares(
        fit_ab_ds_residual,
        [0.5, 1],
        args=(ds, labels, dHKL_label, dHKL_bin_label),
        verbose=1,
        bounds=[(0, 0), (1, 1e6)],
        diff_step=0.01,
    )  # diff_step is important for this not to get stuck earlier
    a = result.x[0]
    b = result.x[1]
    return (a, b)


def burp():
    return "burp"


def eff_r_dw_per_hkl(ds, a, b, label, dHKL_label, inplace=True):
    s       = ds[dHKL_label].to_numpy()
    r_DW    = a * np.exp(-b / (s ** 2))
    rho_DW  = r_DW**2   # this is an estimate of the error-free rho(E1,E2)
    var_eta = ds["SIG" + label].to_numpy() ** 2
    varW_acentric = rice.var(0, 0, np.sqrt(0.5))
    varW_centric = foldnorm.var(0, 0, 1)
    centric  = ds["CENTRIC"] == True
    acentric = ds["CENTRIC"] == False
    rho_obs_ac = 1 / np.sqrt(
        rho_DW ** -2 + var_eta.astype(float) / varW_acentric
    )  # not sure why I had to set the data type explicitly; otherwise just "object"
    rho_obs_c = 1 / np.sqrt(rho_DW ** -2 + var_eta.astype(float) / varW_centric)

    r_eff_ac = np.sqrt(rho_obs_ac)
    r_eff_c = np.sqrt(rho_obs_c)
    if inplace:
        ds.loc[acentric, "r_DW_out_" + label] = r_eff_ac[acentric]
        ds.loc[centric, "r_DW_out_" + label] = r_eff_c[centric]
        ds["r_DW_out_" + label] = ds["r_DW_out_" + label].astype("MTZReal")
        return ds
    else:
        ds_out = ds.copy()
        ds_out.loc[acentric, "r_DW_out_" + label] = r_eff_ac[acentric]
        ds_out.loc[centric, "r_DW_out_" + label] = r_eff_c[centric]
        ds_out["r_DW_out_" + label] = ds_out["r_DW_out_" + label].astype("MTZReal")
        return ds_out
