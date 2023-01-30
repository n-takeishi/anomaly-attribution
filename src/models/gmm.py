import sklearn.mixture
import numpy as np

class GMM(sklearn.mixture.GaussianMixture):
  def fit(self, X, y=None):
    super().fit(X)
    self.logdet_covs_ = np.zeros(self.n_components)
    for i in range(self.n_components):
      _ , self.logdet_covs_[i] = np.linalg.slogdet(self.covariances_[i])
    return self
