import torch
import torch.nn as nn

LOG2PI = 1.837877066409345483560659472811235279722794947275566825634

######################

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

    # Sigma_z
    modules_log_diag_Sigma_z = []
    modules_log_diag_Sigma_z.append(nn.Linear(hidden_dim, hidden_dim))
    modules_log_diag_Sigma_z.append(self.activation_func)
    if batchnorm:
      modules_log_diag_Sigma_z.append(nn.BatchNorm1d(hidden_dim))
    modules_log_diag_Sigma_z.append(nn.Linear(hidden_dim, z_dim))
    self.seq_log_diag_Sigma_z = nn.Sequential(*modules_log_diag_Sigma_z)

  def forward(self, x):
    # the first dim is batch dim
    x = x.reshape(-1, self.x_dim)
    common = self.seq_common(x)
    mu_z = self.seq_mu_z(common)
    log_diag_Sigma_z = self.seq_log_diag_Sigma_z(common)
    diag_Sigma_z = torch.exp(log_diag_Sigma_z)
    return mu_z, diag_Sigma_z


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

    # sigma_sq_x
    if not gaussian:
      self.register_buffer("log_sigma_sq_x", torch.ones(1)*-2)
    else:
      self.log_sigma_sq_x = nn.Parameter(torch.ones(1)*-2)

  def forward(self, z):
    # the first dim is batch dim
    z = z.reshape(-1, self.z_dim)
    mu_x = self.seq_mu_x(z)
    sigma_sq_x = torch.exp(self.log_sigma_sq_x)
    return mu_x, sigma_sq_x # now returns only one sigma_sq_x regardless of batch size

######################

class VAE(nn.Module):
  def __init__(self, x_dim, z_dim, hidden_dim, activation,
    batchnorm=False, gaussian=True):
    super(VAE, self).__init__()
    self.x_dim = x_dim
    self.z_dim = z_dim
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
    self.decoder = Decoder(x_dim, z_dim, hidden_dim, self.activation_func, batchnorm, gaussian=gaussian)

  def reparameterize(self, mu_z, diag_Sigma_z):
      eps = torch.randn_like(diag_Sigma_z)
      return mu_z + eps*diag_Sigma_z

  def forward(self, x):
      mu_z, diag_Sigma_z = self.encoder(x)
      z = self.reparameterize(mu_z, diag_Sigma_z)
      mu_x, sigma_sq_x = self.decoder(z)
      return mu_x, sigma_sq_x, mu_z, diag_Sigma_z

  def reconstruct(self, x):
    mu_z, diag_Sigma_z = self.encoder(x)
    mu_x, _ = self.decoder(mu_z)
    return mu_x


def loss_function(dim_x, x, mu_x, sigma_sq_x, mu_z, diag_Sigma_z, gaussian=True):
  x = x.reshape(-1, dim_x)

  if gaussian:
    REC = 0.5*torch.sum((mu_x-x).pow(2), dim=1) / sigma_sq_x
    REC += 0.5*dim_x*LOG2PI
    REC += 0.5*dim_x*torch.log(sigma_sq_x)
    REC = torch.sum(REC)
  else:
    REC = torch.nn.functional.binary_cross_entropy(mu_x, x, reduction='sum')

  tmp = diag_Sigma_z + mu_z.pow(2) - torch.log(diag_Sigma_z) - 1.0
  KLD = 0.5 * torch.sum(tmp)

  reg_sigma_x = 0.0
  if gaussian:
    reg_sigma_x = -1e-4*(torch.log(sigma_sq_x)-sigma_sq_x)

  return REC+KLD+reg_sigma_x
