import json
import time
import datetime
import numpy as np
import torch

from src.models.vae import VAE

######################

def train(args, set_train, set_valid):
  # args : outdir, suffix, dimx, dimz, dimhidden, noiserate, activation,
  #        batchnorm, gaussian, batchsize, learnrate, epochs,
  #        seed, usecuda, logfreq, testfreq, save
  # set_train : torch Dataset
  # set_valid : torch Dataset

  args.usecuda = args.usecuda and torch.cuda.is_available()
  device = torch.device("cuda" if args.usecuda else "cpu")

  torch.manual_seed(args.seed)

  # determine suffix
  if not args.suffix:
    args.suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

  # output settings
  if args.save:
    with open(args.outdir+'/vae_args_%s.json'%(args.suffix), 'w') as fp:
      json.dump(vars(args), fp, sort_keys=True, indent=2)
  print(vars(args))

  # make data loaders
  kwargs = {'num_workers': 1, 'pin_memory': args.usecuda}
  loader_train = torch.utils.data.DataLoader(dataset=set_train,
    batch_size=args.batchsize, shuffle=True)
  loader_valid = torch.utils.data.DataLoader(dataset=set_valid,
    batch_size=len(set_valid), shuffle=False)

  # make model
  vae_model = VAE(args.dimx, args.dimz, args.dimhidden, args.activation,
    batchnorm=args.batchnorm, gaussian=args.gaussian).to(device)
  # make optimizer
  optimizer = torch.optim.Adam(vae_model.parameters(), lr=args.learnrate)

  if args.gaussian:
    # for non-MNIST (Gaussian VAE), fix log_sigma_sq_x using PCA result
    data = set_train.tensors[0].detach().numpy()
    cov = np.cov(data, rowvar=False)
    cov = 0.5*(cov+cov.T) # make sure symmetry
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    if args.dimx > args.dimz:
      vae_model.decoder.log_sigma_sq_x.data = torch.log(
        torch.ones(1)*np.sum(eigvals[args.dimz:])/(args.dimx-args.dimz)/args.noiserate)
    else:
      vae_model.decoder.log_sigma_sq_x.data = torch.log(torch.ones(1)*1e-2)
    #vae_model.decoder.log_sigma_sq_x.requires_grad = False


  def train_epoch():
    vae_model.train()
    loss_value = 0.
    for x in loader_train:
      if type(x) is list:
        x = x[0]
      x = x.to(device)
      optimizer.zero_grad()
      mu_x, sigma_sq_x, mu_z, diag_Sigma_z = vae_model(x)
      loss = vae.loss_function(x.shape[1], x, mu_x, sigma_sq_x, mu_z, diag_Sigma_z, args.gaussian)
      loss.backward()
      loss_value += loss.item()
      optimizer.step()

    loss_value /= len(loader_train.dataset)
    return loss_value

  def evaluate_epoch():
    vae_model.eval()
    loss_value = 0.
    for x in loader_valid:
      if type(x) is list:
        x = x[0]
      x = x.to(device)
      mu_x, sigma_sq_x, mu_z, diag_Sigma_z = vae_model(x)
      loss = vae.loss_function(x.shape[1], x, mu_x, sigma_sq_x, mu_z, diag_Sigma_z, args.gaussian)
      loss_value += loss.item()

    loss_value /= len(loader_valid.dataset)
    return loss_value


  # creat log file
  if args.save:
    with open(args.outdir+'/vae_info_%s.txt'%(args.suffix), mode='w', encoding='utf-8') as fh:
      fh.write('# epoch  loss_train  runtime_train  loss_valid_best  epoch_best\n');

  # training
  loss_valid_best = 1e10
  epoch_best = 1e10
  runtime_train = 0.
  for epoch in range(args.epochs):
    vae_model.train()
    start = time.time()
    loss_train = train_epoch()
    runtime_train += time.time() - start

    if epoch % args.testfreq == 0:
      vae_model.eval()
      loss_valid = evaluate_epoch()
      # save if best
      if loss_valid < loss_valid_best:
        if args.save:
          torch.save(vae_model.state_dict(), args.outdir+'/vae_model_%s.pt'%(args.suffix))
          # optimizer.save(args.outdir+'/vae_optim_%s.pt'%(args.suffix))
        loss_valid_best = loss_valid
        epoch_best = epoch

    if epoch % args.logfreq == 0:
      print("\r"+"[epoch %03d] training loss: %.4f, best valid loss: %.4f" % (
        epoch, loss_train, loss_valid_best), end="")
      if args.save:
        with open(args.outdir+'/vae_info_%s.txt'%(args.suffix), mode='a', encoding='utf-8') as fh:
          fh.write('%07d  %e  %e  %e  %d' % (epoch,
            loss_train, runtime_train, loss_valid_best, epoch_best) + '\n')

    # stop if valid_best is not updated for 1000 epochs
    if epoch-epoch_best >= 1000:
      break

  print('\nfinish (vae: %s)'%(args.suffix))
