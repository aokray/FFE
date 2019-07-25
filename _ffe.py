###############################################
## Source Code for Fair Feature Embeddings
###############################################
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


# Calculate eigenvals and eigenvecs from given arguments
def _calc_eigens(K, K_u, K_p, up_idxs, p_idxs, num_features):
    one_u = np.ones((len(up_idxs), 1))
    one_p = np.ones((len(p_idxs), 1))

    # Calculate M
    scale = len(p_idxs) ** 2
    M = (
        (scale / len(up_idxs) ** 2) * K_u.T.dot(K_u)
        - ((2 * scale) / (len(up_idxs) * len(p_idxs)))
        * K_u.T.dot(one_u.dot(one_p.T)).dot(K_p)
        + (scale / len(p_idxs) ** 2) * K_p.T.dot(K_p)
    )
    M = M / np.max(M)

    # Select fairest "num_features" base vector weights satisfying the generalized
    # eigenvalue problem Mv = \lambda Kv
    KinvM = np.linalg.pinv(K).dot(M)
    vals, vecs = np.linalg.eigh(KinvM)

    # Be sure to deallocate these huge memory chunks
    M = None
    KinvM = None

    return vals, vecs


# FFE class design from SkLearn's KRR implementation
# https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/kernel_ridge.py#L16
class FFE:
    def __init__(
        self,
        p_idxs,
        up_idxs,
        kernel,
        gamma=None,
        degree=3,
        coef0=1,
        num_features=None,
        kernel_params=None,
    ):
        self.up_idxs = up_idxs
        self.p_idxs = p_idxs
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.num_features = num_features

    # Function courtesy of SkLearn
    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    def transform(self, sample):
        if self.num_features == None:
            print(
                "Setting to default of n/250 features (with n := number of instances)"
            )
            self.num_features = int(sample.shape[0] / 250)

        # Calculate the 3 required kernels, K_u, K_p, K
        K = self._get_kernel(sample, sample)

        K_u = self._get_kernel(sample[self.up_idxs], sample)

        K_p = self._get_kernel(sample[self.p_idxs], sample)

        vals, vecs = _calc_eigens(
            K, K_u, K_p, self.up_idxs, self.p_idxs, self.num_features
        )

        # Be sure to deallocate these huge memory chunks
        K_u = None
        K_p = None

        self.features = vecs[:,0 : self.num_features] 

        # Construct and return explicit representation of data in kernel space
        return K.dot(self.features)
