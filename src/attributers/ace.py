'''
An (unofficial) implementation of:
Zhang et al., ACE -- An anomaly contribution explainer for cyber-security applications,
Proc. of IEEE BigData, 2019.
'''

import numpy as np
import sklearn.linear_model


def ace(x_expl, ef, gaussian=True, N=64, sigma=None, alpha=1.0, seed=1234567890,
  kl=False, beta=50.0, tolreldif=1e-6, itermax=50000):
  # ef(x) : function that returns x's anomaly score
  assert x_expl.ndim == 1
  dim_x = x_expl.size

  np.random.seed(seed)

  # genearte neighbors
  x_neighbors = np.tile(x_expl, (N,1))
  if gaussian:
    x_neighbors += np.random.randn(*x_neighbors.shape)*0.1
  else:
    flip = np.random.randint(2,size=x_neighbors.shape)
    x_neighbors[flip] = 1-x_neighbors[flip]


  # anomaly scores of neighbors
  As = ef(x_neighbors)

  # weights of neighbors
  if sigma is None:
    sigma = 0.75*np.sqrt(dim_x)
  weights = np.exp(-np.sum(np.power(x_neighbors - x_expl, 2), axis=1) / sigma)

  # weighted least squares
  model = sklearn.linear_model.Ridge(alpha=alpha/N, fit_intercept=False, normalize=False)
  model.fit(x_neighbors, As, weights)
  w = model.coef_

  if kl:
    # use torch
    import torch

    w = w + np.ones(dim_x)

    # set parameters
    w_ = torch.Tensor(w).requires_grad_()
    x_expl_ = torch.from_numpy(x_expl).float().clone()
    As_ = torch.from_numpy(As).float()
    weights_ = torch.from_numpy(weights).float()
    x_neighbors_ = torch.from_numpy(x_neighbors).float()

    # solve optimization problem
    optimizer = torch.optim.Adam([w_,], lr=1e-3)
    loss_value = 1e10
    if beta is None:
      beta = 1.0 / N
    for i in range(itermax):
      optimizer.zero_grad()
      loss = torch.sum(As_*torch.pow(x_neighbors_@w_-As_, 2)) + alpha/N*torch.sum(w_*w_)
      c = 1.0/dim_x
      C = c*np.log(c)
      tmp = torch.log(1.0 + torch.exp(w_*x_expl_))
      loss += beta/N*torch.sum( C - c*torch.log(tmp) + c*torch.log(torch.sum(tmp)) )
      loss.backward()
      optimizer.step()

      loss_value_new = loss.detach().item()
      reldif = abs(loss_value - loss_value_new) / abs(loss_value)
      # if i%100==0:
      #   print('\r%d: loss=%f, reldif=%f' % (i,loss_value_new,reldif), end='')
      loss_value = loss_value_new
      if reldif<tolreldif:
        break

  # contributions
  # contributions = np.log(1.0 + np.exp(x_expl*w))
  contributions = np.logaddexp(0.0, x_expl*w)
  contributions /= np.sum(contributions)

  return contributions
