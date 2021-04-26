import numpy as np


def mmh_to_rfl(r, a=256.0, b=1.42):
    """
    wradlib.zr.r2z function
    """
    return a * r ** b


def rfl_to_dbz(z):
    """
    wradlib.trafo.decibel function
    """
    return 10.0 * np.log10(z)


def dbz_to_rfl(d):
    """
    wradlib.trafo.idecibel function
    """
    return 10.0 ** (d / 10.0)


def rfl_to_mmh(z, a=256.0, b=1.42):
    """
    wradlib.zr.z2r function
    """
    return (z / a) ** (1.0 / b)


def mmhr_to_dbz(x, threshold=0.1):
    X_rfl = mmh_to_rfl(x)
    # reflectivity to dBz
    X_rfl[X_rfl == 0] = 0.1  # set 0s to log10(.1) = -1
    X_dbz = rfl_to_dbz(X_rfl)
    X_dbz[~np.isfinite(X_dbz)] = 0
    return X_dbz


def dbz_to_mmhr(x, threshold=0.1):
    X_rfl = dbz_to_rfl(x)
    X_rfl[X_rfl == 1] = 0  # set 0 dbz
    X_threshold = X_rfl <= threshold
    X_mmh = rfl_to_mmh(X_rfl)
    X_mmh[X_threshold] = 0
    X_mmh[~np.isfinite(X_mmh)] = 0
    return X_mmh
