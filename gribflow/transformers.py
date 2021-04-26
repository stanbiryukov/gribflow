from sklearn.base import TransformerMixin

from gribflow.utils import dbz_to_mmhr, mmhr_to_dbz


class prate_per_second_to_dbz(TransformerMixin):
    """
    Given precipitation rate as kg.m-2.s-1, convert to hourly dbz.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return mmhr_to_dbz(X * 3600)

    def inverse_transform(self, X):
        return dbz_to_mmhr(X) / 3600


class per_second_to_hourly(TransformerMixin):
    """
    Given rate data as per second, convert to hourly.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * 3600

    def inverse_transform(self, X):
        return X / 3600


def get_transfomers():
    """
    Expose usable transformer methods in a dictionary.
    """
    return {
        "prate_per_second_to_dbz": prate_per_second_to_dbz(),
        "per_second_to_hourly": per_second_to_hourly(),
    }
