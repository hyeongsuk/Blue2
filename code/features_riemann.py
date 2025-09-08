import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
def concat_and_tangent(windows_cat):
    """windows_cat: ndarray (n_ch, n_samp_concat) -> 1 sample의 탄젠트공간 벡터"""
    X = np.transpose(windows_cat, (1,0))[None, ...]  # (1, n_samp, n_ch)
    cov = Covariances(estimator='oas').fit_transform(X)
    ts = TangentSpace().fit_transform(cov)
    return ts[0]  # 1D
