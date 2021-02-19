import numpy as np
from matplotlib import pyplot as plt
#import reciprocalspaceship as rs

def compute_meanF_byres(ds, label="FP", nbins=20, sigma_cut=0, median=False):
    '''Calculate mean structure factor amplitude by res. bin.
    Use this function only for graphical inspection, not for scaling.
    '''
    #print(ds.shape)
    if sigma_cut > 0:
        incl_criteria = ds[label].to_numpy().flatten() > sigma_cut * ds["SIG" + label].to_numpy().flatten()
        ds2 = ds[incl_criteria].copy()
    else:
        ds2=ds.copy()
    
    ds2, bin_labels = ds2.assign_resolution_bins(bins=nbins)
    if median:
        print("Average observations per bin: " + str(ds2["bin"].value_counts().median()))
        result = ds2.groupby("bin")[label].median()
    else:
        print("Average observations per bin: " + str(ds2["bin"].value_counts().mean()))
        result = ds2.groupby("bin")[label].mean()
    return result, bin_labels


def compute_cc(ds, labels=["F1","F2"], nbins=20, method="spearman"):
    ds, bin_labels = ds.assign_resolution_bins(bins=nbins) #This adds a column to the input!
    print("Average observations per bin: " + str(ds["bin"].value_counts().mean()))
    g = ds.groupby("bin")[labels]
    result = g.corr(method=method).unstack().loc[:, (labels[0],labels[1])]
    return result, bin_labels


def plot_by_res_bin(result, bin_labels, ylabel=r"$CC_{1/2}$",color='b'):
    plt.plot(result, label="Data", color=color)
    plt.xticks(result.index, bin_labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(ylabel)
    plt.xlabel(r"Resolution Bin ($\AA$)")
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.ylim(0, )
    plt.grid(linestyle='--')
    #plt.tight_layout()
    #plt.show()
    return