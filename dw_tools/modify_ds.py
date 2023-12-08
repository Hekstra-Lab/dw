# import reciprocalspaceship as rs
import numpy as np
import reciprocalspaceship as rs
import pandas as pd

# import pandas as


def ds_add_rs(ds, force_rs=False, inplace=True):
    """
    Adds three columns to an rs dataframe with the reciprocal space coordinates (in A^-1) for each Miller index.
    Inplace by default!
    """
    if force_rs or (not "rs_a" in ds.keys()):
        orthomat_list = ds.cell.orthogonalization_matrix.tolist()
        orthomat = np.asarray(orthomat_list)

        hkl_array = np.asarray(list(ds.index.to_numpy()))

        orthomat_inv_t = np.linalg.inv(orthomat).transpose()
        S = np.matmul(orthomat_inv_t, hkl_array.transpose())

        if inplace == True:
            ds["rs_a"] = S.transpose()[:, 0]
            ds["rs_b"] = S.transpose()[:, 1]
            ds["rs_c"] = S.transpose()[:, 2]
        else:
            ds_out = ds.copy()
            ds_out["rs_a"] = S.transpose()[:, 0]
            ds_out["rs_b"] = S.transpose()[:, 1]
            ds_out["rs_c"] = S.transpose()[:, 2]
    else:
        if inplace == True:
            pass
        else:
            ds_out = ds.copy()
            ds_out["rs_a"] = ds["rs_a"]
            ds_out["rs_b"] = ds["rs_b"]
            ds_out["rs_c"] = ds["rs_c"]
    if inplace == True:
        return  # already done
    else:
        return ds_out


def ds_high_res_cut(ds, rescut=2, inplace=True):
    if not "dHKL" in ds.keys():
        ds.compute_dHKL(inplace=True)
    if inplace:
        ds = ds.drop(ds[ds['dHKL'] < rescut].index, inplace=True)
        return ds
    else:
        ds_out = ds.drop(ds[ds['dHKL'] < rescut].index, inplace=False)
        return ds_out


def check_col_dtypes(ds):
    df = rs.summarize_mtz_dtypes(print_summary=False)
    dtype_list=ds.dtypes.to_list()
    for i in range(ds.dtypes.shape[0]):
        try:
            if dtype_list[i] in df["Name"].to_numpy():
                pass
            else:
                print(
                    "Column \""
                    + ds.keys()[i]
                    + "\" has a datatype not supported by the MTZ format."
                )
        except:
            print("Column \"" + ds.keys()[i] + "\" has a datatype not supported by the MTZ format.")


def merge_anomalous(ds, inplace=True):
    """
    This is not very good yet. Does not take into account when reflections are their own Bijvoet mates.
    """
    required_cols = ["I(+)", "I(-)", "SIGI(+)", "SIGI(-)"]
    if np.count_nonzero(ds.columns.isin(required_cols)) == 4:
        print("we're good to go")
    else:
        print("We don't have all of the required columns: " + required_cols)
    min_sig = np.percentile(
        ds[["SIGI(+)", "SIGI(-)"]].to_numpy(), 1
    )  # np.percentile flattens by default
    # this finds the smallest 1%
    w = 1 / (min_sig ** 2 + ds[["SIGI(+)", "SIGI(-)"]].to_numpy() ** 2)
    Iw = np.nansum(w * ds[["I(+)", "I(-)"]].to_numpy(), axis=1) / np.nansum(w, axis=1)
    sigIw = (np.nansum(w, axis=1)) ** -0.5
    if inplace:
        ds["I"] = Iw
        ds["SIGI"] = sigIw
        ds_out = ds
    else:
        ds_out = ds.copy()
        ds_out["I"] = Iw
        ds_out["SIGI"] = sigIw
    return ds_out


def to_pickle(ds, pkl_name):
    ds_tmp = ds.copy()
    ds_tmp.attrs["spacegroup"] = ds.spacegroup.xhm()
    ds_tmp.attrs["cell"] = ds.cell.parameters
    ds_tmp.spacegroup = None
    ds_tmp.cell = None
    ds_tmp.to_pickle(pkl_name)
    return


def from_pickle(pkl_name):
    ds = rs.DataSet(pd.read_pickle(pkl_name))
    ds.spacegroup = ds.attrs["spacegroup"]
    ds.cell = ds.attrs["cell"]
    return ds
