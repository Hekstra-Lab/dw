import numpy as np
import reciprocalspaceship as rs

def apply_iso_B(ds, col_key, dHKL_key="dHKL", B=0):
    ''' Apply an isotropic B factor to reflections'''
    print("YES")
    ds[col_key + "_B_" + str(B)] = ds[col_key] * np.exp(-0.25*B*(1/ds["dHKL"].to_numpy())**2)
    print(ds.info())
    return 
    