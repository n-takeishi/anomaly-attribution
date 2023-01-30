'''
An (unofficial) implementation of:
Sundararajan et al., Axiomatic attribution for deep networks,
Proc. of the 34th ICML, 3319-3328, 2017.
'''

import numpy as np
import torch
import numdifftools


def intgrad(x_expl, x_base, ef, m=50):
  if type(x_expl) is np.ndarray:
    x_expl = torch.from_numpy(x_expl).float().clone()
  if type(x_base) is np.ndarray:
    x_base = torch.from_numpy(x_base).float().clone()

  assert x_expl.ndim == 1
  dim_x = x_expl.size(0)

  dif = x_expl - x_base
  accumlated_grad = torch.zeros_like(x_expl)
  for i in range(m):
    x = (x_base + (i+1)*(dif/m)).requires_grad_(True)
    evalue = ef(x)
    grad = torch.autograd.grad(evalue, x)
    accumlated_grad += grad[0] / m
  answer = dif*accumlated_grad
  return answer.detach().numpy()


def intgrad_approx(x_expl, x_base, ef, m=50):
  assert x_expl.ndim == 1
  dim_x = x_expl.size

  grad_ef = numdifftools.Gradient(ef)

  dif = x_expl - x_base
  x = np.copy(x_base)
  accumlated_grad = np.zeros_like(x_expl)
  for i in range(m):
    x += dif / m
    accumlated_grad += grad_ef(x) / m
  return dif*accumlated_grad
