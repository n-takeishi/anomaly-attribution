import numpy as np
from scipy.special import logsumexp
import torch


def compset(S, n):
  if type(S) is list:
    nonS = list(set(range(n)) - set(S))
  elif type(S) is tuple:
    nonS = tuple(set(range(n)) - set(S))
  elif type(S) is np.ndarray:
    nonS = np.array(list(set(range(n)) - set(S)))
  elif type(S) is set:
    nonS = set(range(n)) - S
  else:
    raise ValueError('unknown S type: type(S) = %s' % type(S))
  return nonS


def gaussian_energy(x, dim_x, mean, prec, logdet_cov):
  x = x.reshape(-1, dim_x)

  if type(x) is np.ndarray:
    sum_ = lambda x: np.sum(x, axis=1)
  else:
    sum_ = lambda x: torch.sum(x, dim=1)
  answer = 0.5*sum_((x-mean) * (prec@(x-mean).T).T)
  answer += 0.5*dim_x*1.8378770 + 0.5*logdet_cov
  return answer


def gaussian_marginenergy(S, x, dim_x, mean, cov, prec, logdet_cov):
  x = x.reshape(-1, dim_x)

  if len(S)==dim_x:
    return gaussian_energy(x, dim_x, mean, prec, logdet_cov)

  new_mean = mean[S]
  new_cov = cov[np.ix_(S, S)]
  new_prec = prec[np.ix_(S, S)]
  _, new_logdet_cov = np.linalg.slogdet(new_cov)
  return gaussian_energy(x[:,S], len(S), new_mean, new_prec, new_logdet_cov)


def gaussian_allmarginenergy(x, dim_x, mean, prec):
  x = x.reshape(-1, dim_x)

  diag_prec = np.diag(prec)[np.newaxis,:]
  answer = 0.5*diag_prec * np.power(x-mean, 2)
  answer += 0.5*dim_x*1.8378770 - 0.5*np.log(diag_prec)
  return answer


def gmm_energy(x, dim_x, gmm_model):
  return gmm_marginenergy(range(dim_x), x, dim_x, gmm_model)


def gmm_marginenergy(S, x, dim_x, gmm_model):
  x = x.reshape(-1, dim_x)

  K = gmm_model.weights_.shape[0]
  N = x.shape[0]

  if type(x) is np.ndarray:
    answer = np.zeros((K, N))
    for i in range(K):
      answer[i] = -gaussian_marginenergy(S, x, dim_x,
        gmm_model.means_[i], gmm_model.covariances_[i], gmm_model.precisions_[i],
        gmm_model.logdet_covs_[i])
      answer[i] += np.log(gmm_model.weights_[i])
    return -logsumexp(answer, axis=0, keepdims=False, return_sign=False)
  else:
    answer = torch.zeros(K, N)
    for i in range(K):
      answer[i] = -gaussian_marginenergy(S, x, dim_x,
        torch.from_numpy(gmm_model.means_[i]).float(),
        torch.from_numpy(gmm_model.covariances_[i]).float(),
        torch.from_numpy(gmm_model.precisions_[i]).float(),
        torch.from_numpy(np.array([gmm_model.logdet_covs_[i]])).float())
      answer[i] += torch.log( torch.tensor([gmm_model.weights_[i]]) )
    return -torch.logsumexp(answer, dim=0, keepdim=False)


def gmm_allmarginenergy(x, dim_x, gmm_model):
  x = x.reshape(-1, dim_x)

  K = gmm_model.weights_.shape[0]

  answer = np.zeros((K, x.shape[0], dim_x))
  for i in range(K):
    answer[i] = -gaussian_allmarginenergy(x, dim_x,
      gmm_model.means_[i], gmm_model.precisions_[i])
    answer[i] += np.log(gmm_model.weights_[i])
  return -logsumexp(answer, axis=0, keepdims=False, return_sign=False)


def vae_negelbo(x, dim_x, vae_model):
  x = x.reshape(-1, dim_x)
  vae_model.eval()

  was_numpy = False
  if type(x) is np.ndarray:
    x = torch.from_numpy(x).float()
    was_numpy = True

  mu_x, sigma_sq_x, mu_z, diag_Sigma_z = vae_model(x)

  if vae_model.gaussian:
    REC = 0.5*torch.sum((mu_x-x).pow(2), dim=1) / sigma_sq_x
    REC += 0.5*dim_x*1.8378770
    REC += 0.5*dim_x*torch.log(sigma_sq_x)
  else:
    #REC = torch.nn.functional.binary_cross_entropy(mu_x, x, reduction='none')
    eps = torch.zeros_like(mu_x)+1e-4
    REC = -x*torch.log( torch.max(mu_x, eps) ) - (1-x)*torch.log( torch.max(1-mu_x, eps) )
    REC = torch.sum(REC, dim=1)
  tmp = diag_Sigma_z + mu_z.pow(2) - torch.log(diag_Sigma_z) - 1.0
  KLD = 0.5 * torch.sum(tmp, dim=1)
  negelbo = REC+KLD

  if was_numpy:
    negelbo = negelbo.detach().numpy()
  return negelbo


def vae_recerr(x, dim_x, vae_model):
  return vae_marginrecerr(range(dim_x), x, dim_x, vae_model)


def vae_marginrecerr(S, x, dim_x, vae_model):
  x = x.reshape(-1, dim_x)
  vae_model.eval()

  tmp = vae_allmarginrecerr(x, dim_x, vae_model)
  if type(x) is np.ndarray:
    return np.sum(tmp[:,S], axis=1)
  else:
    return torch.sum(tmp[:,S], dim=1)


def vae_allmarginrecerr(x, dim_x, vae_model):
  x = x.reshape(-1, dim_x)
  vae_model.eval()

  if type(x) is np.ndarray:
    mu_z, diag_Sigma_z = vae_model.encoder(torch.from_numpy(x).float())
  else:
    mu_z, diag_Sigma_z = vae_model.encoder(x)

  #z = dist.Normal(mu_z, diag_Sigma_z).sample()
  z = mu_z
  mu_x, Sigma_z = vae_model.decoder(z)

  if vae_model.gaussian:
    if type(x) is np.ndarray:
      dif = x - mu_x.detach().numpy()
    else:
      dif = x - mu_x
    rec = dif*dif
  else:
    if type(x) is np.ndarray:
      mu_x_ = mu_x.detach().numpy()
      eps = np.zeros_like(mu_x_)+1e-4
      rec = -x*np.log( np.fmax(mu_x_, eps) ) - (1-x)*np.log( np.fmax(1-mu_x_, eps) )
    else:
      eps = torch.zeros_like(mu_x)+1e-4
      rec = -x*torch.log( torch.max(mu_x, eps) ) - (1-x)*torch.log( torch.max(1-mu_x, eps) )
  return rec


def dagmm_energy(x, dim_x, dagmm_model):
  x = x.reshape(-1, dim_x)
  K = dagmm_model.weights_.shape[0]
  N = x.shape[0]
  dagmm_model.eval()

  if type(x) is np.ndarray:
    _, _, latfeat, _ = dagmm_model(torch.from_numpy(x).float().clone())
  else:
    _, _, latfeat, _ = dagmm_model(x)

  answer = torch.zeros(K, N)
  for i in range(K):
    answer[i] = -gaussian_energy(latfeat, latfeat.shape[1],
      dagmm_model.means_[i], dagmm_model.precisions_[i], dagmm_model.logdet_covs_[i])
    answer[i] += torch.log(dagmm_model.weights_[i])
  answer = -torch.logsumexp(answer, dim=0, keepdim=False)

  if type(x) is np.ndarray:
    return answer.detach().numpy()
  else:
    return answer


def _baseliner(S, x, ef, gaussian=True, gamma=1e-2, itermax=2000, learnrate=1e-2, margin=0.0, tolreldif=1e-6):
  assert x.ndim == 1

  # TODO consider device properly

  if type(x) is np.ndarray:
    x_org = torch.from_numpy(x).float().clone().requires_grad_(False)
  elif type(x) is torch.Tensor:
    x_org = x.clone().requires_grad_(False)
  dim_x = x_org.size(0)

  S = list(S)
  nonS = compset(S, dim_x)
  dim_nonS = len(nonS)

  if not gaussian:
    x_org_binary = x_org.clone()
    x_org = x_org*10.0 - 5.0

  x_S = x_org[S] # fixed
  x_nonS = x_org[nonS].clone().requires_grad_(True)

  def loss_function(x_now):
    torch.manual_seed(42); np.random.seed(42)
    if gaussian:
      loss = ef(x_now)[0] / dim_x
      tiny = 1e-4
      normalizer = torch.where(torch.abs(x_org)>tiny, x_org, torch.ones_like(x_org)*tiny)
      reldist = torch.pow((x_now-x_org)/normalizer, 2).sum() / dim_nonS
    else:
      loss = ef(torch.sigmoid(x_now))[0] / dim_x
      reldist = torch.nn.functional.binary_cross_entropy(
        torch.sigmoid(x_now), x_org_binary, reduction='sum') / dim_nonS
    loss += gamma*torch.nn.functional.relu(reldist-margin)
    return loss

  def get_x_now(x_nonS_):
    x_now = torch.zeros(dim_x)
    x_now[S] = x_S
    x_now[nonS] = x_nonS_
    return x_now

  optimizer = torch.optim.Adam([x_nonS,], lr=learnrate)

  loss_value = 1e10
  for i in range(itermax):
    # update
    optimizer.zero_grad()
    loss = loss_function(get_x_now(x_nonS))
    loss.backward()
    optimizer.step()

    # check convergence
    loss_value_new = loss.detach().item()
    reldif = abs(loss_value - loss_value_new) / (abs(loss_value)+1e-6)
    loss_value = loss_value_new
    if reldif<tolreldif:
      break
    if i==0:
      loss_value_init = loss_value

  # print(loss_value_init, loss_value, i)

  # return result
  x_last = torch.zeros(dim_x)
  x_last[S] = x_S.detach()
  x_last[nonS] = x_nonS.detach()

  if not gaussian:
    x_last = torch.sigmoid(x_last)

  if type(x) is np.ndarray:
    return x_last.numpy()
  else:
    return x_last


class Baseliner():
  def __init__(self, x, dim_x, ef, kwargs_to_baseliner):
    self.x = x.reshape(-1, dim_x)
    self.dim_x = dim_x
    self.ef = ef
    self.kwargs_to_baseliner = kwargs_to_baseliner

  def __call__(self, S):
    S = list(S)
    nonS = compset(S, self.dim_x)
    if len(nonS)<1:
      return self.ef(self.x)

    r = np.zeros_like(self.x)
    for i in range(self.x.shape[0]):
      r[i] = _baseliner(S, self.x[i], self.ef, **self.kwargs_to_baseliner)

    ef_avg = np.zeros(self.x.shape[0])
    for j in range(r.shape[0]):
      x_new = np.copy(self.x)
      x_new[:,nonS] = r[j,nonS]
      ef_avg += self.ef(x_new)
    ef_avg /= r.shape[0]
    return ef_avg


class BaselinerRelaxed():
  def __init__(self, x, dim_x, ef, kwargs_to_baseliner):
    assert x.ndim==1
    assert x.size==dim_x
    # currently for only single x

    self.x = x
    self.dim_x = dim_x
    self.ef = ef

    self.x_each = np.zeros((dim_x,dim_x))
    for i in range(dim_x):
      self.x_each[:,i] = _baseliner([i,], x, ef, **kwargs_to_baseliner)
    self.x_empty = _baseliner([], x, ef, **kwargs_to_baseliner)

  def __call__(self, S):
    S = list(S)
    nonS = compset(S, self.dim_x)
    if len(nonS)<1:
      return self.ef(self.x)
    elif len(nonS)==self.dim_x:
      return self.ef(self.x_empty)

    r = np.sum(self.x_each[:,S], axis=1) + self.x_empty
    r /= len(S)+1
    x_new = np.copy(self.x)
    x_new[nonS] = r[nonS]
    ef_ = self.ef(x_new)
    return ef_
