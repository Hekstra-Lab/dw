import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import reciprocalspaceship as rs

def compute_cc(half1, half2):
    joined = half1.merge(half2, how="inner", on=["H", "K", "L"],
                         suffixes=("1", "2"))
    joined, bin_labels = joined.assign_resolution_bins()
    print(joined["bin"].value_counts())
    g = joined.groupby("bin")[["F1",  "F2"]]
    result = g.corr(method="pearson").unstack().loc[:, ("F1", "F2")]
    return result, bin_labels

def plot(result, bin_labels):
    plt.plot(result, label="Data")
    plt.xticks(result.index, bin_labels, rotation=45)
    plt.ylabel(r"$CC_{1/2}$")
    plt.xlabel(r"Resolution Bin ($\AA$)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0, 1)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.show()
    return

def main():

    half1 = rs.read_mtz(sys.argv[1])
    half2 = rs.read_mtz(sys.argv[2])

    result, labels = compute_cc(half1, half2)
    plot(result, labels)
    return

if __name__  == "__main__":
    main()
