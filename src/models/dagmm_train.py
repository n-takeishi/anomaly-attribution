import json
import time
import datetime
import torch

from src.models.dagmm import DAGMM

######################

def train(args, set_train, set_valid):
  # args : outdir, suffix, dimx, dimz, dimhidden, numcomps, activation,
  #        regparam1, regparam2,
  #        batchnorm, gaussian, batchsize, learnrate, epochs,
  #        seed, usecuda, logfreq, testfreq, save
  # set_train : torch Dataset
  # set_valid : torch Dataset

  args.usecuda = args.usecuda and torch.cuda.is_available()
  device = torch.device("cuda" if args.usecuda else "cpu")

  torch.manual_seed(args.seed)

  lambda_energy = args.regparam1
  lambda_invdiagcov = args.regparam2

  # determine suffix
  if not args.suffix:
    args.suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

  # output settings
  if args.save:
    with open(args.outdir+'/dagmm_args_%s.json'%(args.suffix), 'w') as fp:
      json.dump(vars(args), fp, sort_keys=True, indent=2)
  print(vars(args))

  # make data loaders
  kwargs = {'num_workers': 1, 'pin_memory': args.usecuda}
  loader_train = torch.utils.data.DataLoader(dataset=set_train,
    batch_size=args.batchsize, shuffle=True)
  loader_valid = torch.utils.data.DataLoader(dataset=set_valid,
    batch_size=len(set_valid), shuffle=False)

  # make model
  dagmm_model = dagmm.DAGMM(args.dimx, args.dimz, args.dimhidden, args.numcomps, args.activation,
    batchnorm=args.batchnorm, gaussian=args.gaussian).to(device)
  # make optimizer
  optimizer = torch.optim.Adam(dagmm_model.parameters(), lr=args.learnrate)


  def train_epoch():
    dagmm_model.train()
    loss_value = 0.
    for x in loader_train:
      if type(x) is list:
        x = x[0]
      x = x.to(device)
      optimizer.zero_grad()
      z, x_rec, latfeat, gamma = dagmm_model(x)
      loss, recerr, energy, sum_invdiagcov = dagmm_model.loss_function(
        x, x_rec, latfeat, gamma, lambda_energy, lambda_invdiagcov)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(dagmm_model.parameters(), 5.0)
      loss_value += loss.item()
      optimizer.step()

    loss_value /= len(loader_train.dataset)
    return loss_value

  def evaluate_epoch():
    dagmm_model.eval()
    loss_value = 0.
    for x in loader_valid:
      if type(x) is list:
        x = x[0]
      x = x.to(device)
      z, x_rec, latfeat, gamma = dagmm_model(x)
      loss, recerr, energy, sum_invdiagcov = dagmm_model.loss_function(
        x, x_rec, latfeat, gamma, lambda_energy, lambda_invdiagcov)
      loss_value += loss.item()

    loss_value /= len(loader_valid.dataset)
    return loss_value


  # creat log file
  if args.save:
    with open(args.outdir+'/dagmm_info_%s.txt'%(args.suffix), mode='w', encoding='utf-8') as fh:
      fh.write('# epoch  loss_train  runtime_train  loss_valid_best  epoch_best\n');

  # training
  loss_valid_best = 1e10
  epoch_best = 1e10
  runtime_train = 0.
  for epoch in range(args.epochs):
    start = time.time()
    loss_train = train_epoch()
    runtime_train += time.time() - start

    if epoch % args.testfreq == 0:
      loss_valid = evaluate_epoch()
      # save if best
      if loss_valid < loss_valid_best:
        if args.save:
          torch.save(dagmm_model.state_dict(), args.outdir+'/dagmm_model_%s.pt'%(args.suffix))
          # optimizer.save(args.outdir+'/dagmm_optim_%s.pt'%(args.suffix))
        loss_valid_best = loss_valid
        epoch_best = epoch

    if epoch % args.logfreq == 0:
      print("\r"+"[epoch %03d] training loss: %.4f, best valid loss: %.4f" % (
        epoch, loss_train, loss_valid_best), end="")
      if args.save:
        with open(args.outdir+'/dagmm_info_%s.txt'%(args.suffix), mode='a', encoding='utf-8') as fh:
          fh.write('%07d  %e  %e  %e  %d' % (epoch,
            loss_train, runtime_train, loss_valid_best, epoch_best) + '\n')

    # stop if valid_best is not updated for 1000 epochs
    if epoch-epoch_best >= 1000:
      break

  print('\nfinish (dagmm: %s)'%(args.suffix))
