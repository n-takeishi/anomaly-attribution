import math
import numpy as np
from itertools import chain, combinations
import sklearn.linear_model
from scipy.special import comb


def _is_notbadsubset(subset, bad_subsets):
  # subset      : python set
  # bad_subsets : python set of python frozenset
  for bad_subset in bad_subsets:
    if subset >= bad_subset:
      return False
  return True


def _binom(n, k):
  # return int(math.factorial(n) / math.factorial(k) / math.factorial(n - k))
  return comb(n, k, exact=True)


def _powerset(n, min_size, max_size):
  return [s for s in \
    chain.from_iterable(combinations(list(range(n)), k) for k in \
      range(min_size, 1+max_size))]


def _weight(n,k):
  return (n-1)/_binom(n,k)/k/(n-k)


class wls():
  def __init__(self, n, m=2048, seed=0, bignum=1e6):
    self.n = n
    self.seed = seed

    np.random.seed(seed)
    possible_kmax = math.floor((n-1)/2)

    # check max subset-size with which subsets can be completely enumerated
    kmax = 0
    m_needed = 0
    for k in range(1,possible_kmax+1):
      m_needed += 2*_binom(n,k)
      if m>=m_needed: kmax = k

    # completely enumerate subset sizes 1, n-1, ..., kmax, n-kmax
    subsets_small = []; subsets_large = []
    enum_weights = []
    m_enumerate = 0
    for k in range(1,1+kmax):
      subsets_small += _powerset(n, k, k)
      subsets_large += _powerset(n, n-k, n-k)
      enum_weights += [_weight(n,k) for i in range(_binom(n,k))]
      m_enumerate += 2*_binom(n,k)

    self.subsets = []
    if kmax<possible_kmax:
      m_sampled = m - m_enumerate
      # sample subset sizes according to the weight
      sampled_subset_sizes = np.array([i for i in range(kmax+1, n-kmax)])
      sampled_weights = np.array([_weight(n,k) for k in sampled_subset_sizes])
      sampled_prob = sampled_weights/np.sum(sampled_weights)
      chosen_nums = np.random.multinomial(m_sampled, sampled_prob)

      # sample subsets of each size (from small) uniformly
      for i,k in enumerate(sampled_subset_sizes):
        chosen_num = chosen_nums[i]
        for j in range(chosen_num):
          o = tuple(np.random.permutation(n))
          self.subsets.append(o[:k])
    else:
      sampled_weights = 0.0

    self.subsets = subsets_small + self.subsets + subsets_large[::-1]
    m = len(self.subsets)

    # weights;
    # mean weight for sampled part,
    # exact weight for enumerated part,
    # big number for the first and the last
    weights = np.ones(m)*np.mean(sampled_weights)
    weights[:len(subsets_small)] = np.array(enum_weights)
    weights[m-len(subsets_large):] = np.array(enum_weights[::-1])
    weights = np.hstack([bignum, weights, bignum])
    self.weights = weights

    # covariate matrix
    binarymat = np.zeros((m,n))
    for j,subset in enumerate(self.subsets):
      binarymat[j, subset] = 1
    binarymat = np.vstack([np.zeros((1,n)), binarymat, np.ones((1,n))])
    self.binarymat = np.hstack([np.ones((m+2,1)), binarymat])

  def __call__(self, vfunc, do_bound=False):
    bad_subsets = set()

    # valvec must be: first=v(empty), last=v(all); otherwise v(S)
    valvec = np.zeros(len(self.subsets)+2)
    valvec[0], _ = vfunc([])
    for j in range(len(self.subsets)):
      subset = self.subsets[j]
      subset_ = set(subset)
      if _is_notbadsubset(subset_, bad_subsets):
        # if not-bad subset, compute v(subset)
        valvec[j+1], is_bad = vfunc(subset)
        #print('\r', is_bad, subset, valvec[j+1], end='')
        # if bad subset, memorize subset as a bad subset
        if do_bound and is_bad:
          bad_subsets.add(frozenset(subset_))
      else:
        valvec[j+1] = vfunc.e_th
    valvec[-1], _ = vfunc(list(range(self.n)))

    # regression
    #regressor = sklearn.linear_model.LinearRegression(fit_intercept=False,normalize=False)
    regressor = sklearn.linear_model.Ridge(alpha=1e-6,fit_intercept=False,normalize=False)
    result = regressor.fit(self.binarymat, valvec, sample_weight=self.weights)

    shvals = result.coef_
    return shvals[1:], bad_subsets
