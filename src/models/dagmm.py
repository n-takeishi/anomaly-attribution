import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG2PI = 1.837877066409345483560659472811235279722794947275566825634

######################

# z --> x
class Decoder(nn.Module):
  def __init__(self, x_dim, z_dim, hidden_dim, activation_func, batchnorm=False, gaussian=True):
    super(Decoder, self).__init__()
    self.x_dim = x_dim
    self.z_dim = z_dim
    self.activation_func = activation_func
    self.batchnorm = batchnorm
    self.gaussian = gaussian

    # mu_x
    modules_mu_x = []
    modules_mu_x.append(nn.Linear(z_dim, hidden_dim))
    modules_mu_x.append(self.activation_func)
    if batchnorm:
      modules_mu_x.append(nn.BatchNorm1d(hidden_dim))
    modules_mu_x.append(nn.Linear(hidden_dim, x_dim))
    if not gaussian:
      modules_mu_x.append(nn.Sigmoid())
    self.seq_mu_x = nn.Sequential(*modules_mu_x)

  def forward(self, z):
    # the first dim is batch dim
    z = z.reshape(-1, self.z_dim)
    mu_x = self.seq_mu_x(z)
    return mu_x


# x --> z
class Encoder(nn.Module):
  def __init__(self, x_dim, z_dim, hidden_dim, activation_func, batchnorm=False):
    super(Encoder, self).__init__()
    self.x_dim = x_dim
    self.z_dim = z_dim
    self.activation_func = activation_func
    self.batchnorm = batchnorm

    # common
    modules_common = []
    modules_common.append(nn.Linear(x_dim, hidden_dim))
    modules_common.append(self.activation_func)
    if batchnorm:
      modules_common.append(nn.BatchNorm1d(hidden_dim))
    self.seq_common = nn.Sequential(*modules_common)

    # mu_z
    modules_mu_z = []
    modules_mu_z.append(nn.Linear(hidden_dim, hidden_dim))
    modules_mu_z.append(self.activation_func)
    if batchnorm:
      modules_mu_z.append(nn.BatchNorm1d(hidden_dim))
    modules_mu_z.append(nn.Linear(hidden_dim, z_dim))
    self.seq_mu_z = nn.Sequential(*modules_mu_z)

  def forward(self, x):
    # the first dim is batch dim
    x = x.reshape(-1, self.x_dim)
    common = self.seq_common(x)
    mu_z = self.seq_mu_z(common)
    return mu_z


# latfeat --> gamma
class Estimator(nn.Module):
  def __init__(self, latfeat_dim, hidden_dim, comp_num, activation_func, batchnorm=False):
    super(Estimator, self).__init__()
    self.latfeat_dim = latfeat_dim
    self.comp_num = comp_num
    self.activation_func = activation_func
    self.batchnorm = batchnorm

    # gamma
    modules_gamma = []
    modules_gamma.append(nn.Linear(latfeat_dim, hidden_dim))
    modules_gamma.append(self.activation_func)
    modules_gamma.append(nn.Dropout(p=0.5))
    if batchnorm:
      modules_gamma.append(nn.BatchNorm1d(hidden_dim))
    modules_gamma.append(nn.Linear(hidden_dim, comp_num))
    modules_gamma.append(nn.Softmax(dim=1))
    self.seq_gamma = nn.Sequential(*modules_gamma)

  def forward(self, latfeat):
    # the first dim is batch dim
    latfeat = latfeat.reshape(-1, self.latfeat_dim)
    gamma = self.seq_gamma(latfeat)
    return gamma

######################

# https://github.com/danieltan07/dagmm

def _relative_euclidean_distance(a, b):
  return (a-b).norm(2, dim=1) / a.norm(2, dim=1)


class DAGMM(nn.Module):
  def __init__(self, x_dim, z_dim, hidden_dim, comp_num, activation,
    batchnorm=False, gaussian=True):
    super(DAGMM, self).__init__()
    self.x_dim = x_dim
    self.z_dim = z_dim
    self.comp_num = comp_num
    self.activation = activation
    self.batchnorm = batchnorm
    self.gaussian = gaussian

    if activation=='Softplus':
      self.activation_func = nn.Softplus()
    elif activation=='LeakyReLU':
      self.activation_func = nn.LeakyReLU()
    elif activation=='PReLU':
      self.activation_func = nn.PReLU()
    elif activation=='ReLU':
      self.activation_func = nn.ReLU()
    elif activation=='Tanh':
      self.activation_func = nn.Tanh()
    else:
      raise ValueError('unknown acivation type')

    self.encoder = Encoder(x_dim, z_dim, hidden_dim, self.activation_func, batchnorm)
    self.decoder = Decoder(x_dim, z_dim, hidden_dim, self.activation_func, batchnorm)
    self.estimator = Estimator(z_dim+2, hidden_dim, comp_num, self.activation_func, batchnorm)

    self.register_buffer("weights_", torch.zeros(comp_num))
    self.register_buffer("means_", torch.zeros(comp_num,z_dim+2))
    self.register_buffer("covariances_", torch.zeros(comp_num,z_dim+2,z_dim+2))
    self.register_buffer("precisions_", torch.zeros(comp_num,z_dim+2,z_dim+2))
    self.register_buffer("logdet_covs_", torch.zeros(comp_num))

  def forward(self, x):
    z = self.encoder(x)
    x_rec = self.decoder(z)
    rec_cosine = F.cosine_similarity(x, x_rec, dim=1)
    rec_euclidean = _relative_euclidean_distance(x, x_rec)
    latfeat = torch.cat([z, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
    gamma = self.estimator(latfeat)
    return z, x_rec, latfeat, gamma

  def compute_gmm_params(self, latfeat, gamma):
    K = self.comp_num
    D = self.z_dim+2
    sum_gamma = torch.sum(gamma, dim=0)

    weights = torch.mean(gamma, dim=0)
    if self.training:
      self.weights_ = weights.detach()

    # z = N x D, mu = K x D, gamma = N x K
    means = torch.sum(gamma.unsqueeze(-1)*latfeat.unsqueeze(1), dim=0) \
      / sum_gamma.unsqueeze(-1)
    if self.training:
      self.means_ = means.detach()

    # feat_mu = N x K x D
    feat_center = latfeat.unsqueeze(1) - means.unsqueeze(0)
    # feat_mu_outer = N x K x D x D
    feat_center_outer = feat_center.unsqueeze(-1) * feat_center.unsqueeze(-2)
    # covs = K x D x D
    covs = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * feat_center_outer, dim=0) \
      / sum_gamma.unsqueeze(-1).unsqueeze(-1)
    if self.training:
      self.covariances_ = covs.detach()

    precs = torch.zeros(K, D, D)
    logdet_covs = torch.zeros(K)
    for i in range(K):
      precs[i] = torch.inverse(covs[i])
      _, logdet_covs[i] = torch.slogdet(covs[i])
      # TODO: here can be a little improved
    if self.training:
      self.precisions_ = precs.detach()
      self.logdet_covs_ = logdet_covs.detach()

    return weights, means, covs, precs, logdet_covs

  def compute_energy(self, latfeat, weights, means, covs, precs, logdet_covs):
    K = self.comp_num
    D = self.z_dim+2
    feat_center = latfeat.unsqueeze(1) - means.unsqueeze(0)

    # N x K
    logprob_comps = -0.5*torch.sum(
      torch.sum(feat_center.unsqueeze(-1)*precs.unsqueeze(0), dim=-2)*feat_center, dim=-1)
    logprob_comps -= 0.5*D*LOG2PI
    logprob_comps -= 0.5*logdet_covs
    logprob_comps -= torch.log(weights)

    # N
    energy = -torch.logsumexp(logprob_comps, 1)

    sum_invdiagcov = 0.0
    for i in range(K):
      sum_invdiagcov += torch.sum(1.0 / covs[i].diag())

    return energy, sum_invdiagcov

  def loss_function(self, x, x_rec, latfeat, gamma, lambda_energy, lambda_invdiagcov):
    recerr = torch.sum(torch.pow(x-x_rec,2))
    weights, means, covs, precs, logdet_covs = self.compute_gmm_params(latfeat, gamma)
    energy, sum_invdiagcov = self.compute_energy(latfeat, weights, means, covs, precs, logdet_covs)
    energy = torch.sum(energy)
    loss = recerr + lambda_energy*energy + x.shape[0]*lambda_invdiagcov*sum_invdiagcov
    return loss, recerr, energy, sum_invdiagcov
