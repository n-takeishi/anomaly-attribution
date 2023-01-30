from types import SimpleNamespace
import os
import numpy as np
import torch

from src.models import gmm_train, vae_train, dagmm_train
from src import exptutil


def do_gmm_train(outdir, data_train, cand_numcomps):
    args=SimpleNamespace(); args.ntrdata,args.dimx=data_train.shape; args.outdir=outdir; args.save=True

    for i in range(len(cand_numcomps)):
        args.numcomps = cand_numcomps[i]
        args.suffix = 'nc%d' % (args.numcomps)
        gmm_train.train(args, data_train)


def do_vae_train(outdir, data_train, data_valid, cand_params, # [dimz, dimhidden]
                 activation=None, batchnorm=None, gaussian=None,
                 batchsize=None, epochs=None, learnrate=None, testfreq=None, noiserate=None):
    args=SimpleNamespace(); args.ntrdata,args.dimx=data_train.shape; args.outdir=outdir; args.save=True
    args.seed = 1234567890; args.usecuda=False

    args.activation = 'Softplus' if activation is None else activation
    args.batchnorm = False if batchnorm is None else batchnorm
    args.gaussian = True if gaussian is None else gaussian
    args.batchsize = 512 if batchsize is None else batchsize
    args.epochs = round(50000/max(1.0,args.ntrdata/args.batchsize)) if epochs is None else epochs
    args.learnrate = 1e-3 if learnrate is None else learnrate
    args.testfreq = 20 if testfreq is None else testfreq; args.logfreq = args.testfreq
    args.noiserate = 2.0 if noiserate is None else noiserate

    set_train = torch.utils.data.TensorDataset(torch.Tensor(data_train))
    set_valid = torch.utils.data.TensorDataset(torch.Tensor(data_valid))
    for i in range(len(cand_params)):
        args.dimz = cand_params[i][0]
        args.dimhidden = cand_params[i][1]
        args.suffix = 'dz%d_dh%d' % (args.dimz, args.dimhidden)
        vae_train.train(args, set_train, set_valid)


def do_dagmm_train(outdir, data_train, data_valid, cand_params, # [numcomps, dimz, dimhidden]
                   activation=None, batchnorm=None, gaussian=None,
                   batchsize=None, epochs=None, learnrate=None, testfreq=None, reg1=None, reg2=None):
    args=SimpleNamespace(); args.ntrdata,args.dimx=data_train.shape; args.outdir=outdir; args.save=True
    args.seed=1234567890; args.usecuda=False

    args.activation = 'Softplus' if activation is None else activation
    args.batchnorm = False if batchnorm is None else batchnorm
    args.gaussian = True if gaussian is None else gaussian
    args.batchsize = 512 if batchsize is None else batchsize
    args.epochs = round(100000/max(1.0,args.ntrdata/args.batchsize)) if epochs is None else epochs
    args.learnrate = 1e-4 if learnrate is None else learnrate
    args.testfreq = 20 if testfreq is None else testfreq; args.logfreq = args.testfreq
    args.regparam1 = 0.1 if reg1 is None else reg1
    args.regparam2 = 0.005 if reg2 is None else reg2

    set_train = torch.utils.data.TensorDataset(torch.Tensor(data_train))
    set_valid = torch.utils.data.TensorDataset(torch.Tensor(data_valid))
    for i in range(len(cand_params)):
        args.numcomps = cand_params[i][0]
        args.dimz = cand_params[i][1]
        args.dimhidden = cand_params[i][2]
        args.suffix = 'nc%d_dz%d_dh%d' % (args.numcomps, args.dimz, args.dimhidden)
        dagmm_train.train(args, set_train, set_valid)


# train models

for dataname in ['thyroid', 'breastw', 'lympho', 'musk', 'arrhythmia', 'U2R']:
    datadir = os.path.join('data', 'processed', dataname)

    # load data
    data_train, data_valid, data_test = exptutil.load_data(datadir)

    # common settings
    dimx = data_train.shape[1]
    gaussian = True
    batchsize = 512
    if dataname=='lympho':
        gaussian = False
        batchsize = 64
    elif dataname=='arrhythmia':
        batchsize = 256

    # GMM
    outdir = os.path.join('models', dataname, 'gmm')
    os.makedirs(outdir, exist_ok=True)
    do_gmm_train(outdir, data_train, [2,3,4,5])

    # VAE
    outdir = os.path.join('models', dataname, 'vae')
    os.makedirs(outdir, exist_ok=True)
    cand_params = [
        [round(0.2*dimx), round(0.5*dimx)], [round(0.2*dimx), dimx], [round(0.2*dimx), 2*dimx],
        [round(0.4*dimx), round(0.5*dimx)], [round(0.4*dimx), dimx], [round(0.4*dimx), 2*dimx],
        [round(0.6*dimx), round(0.5*dimx)], [round(0.6*dimx), dimx], [round(0.6*dimx), 2*dimx],
        [round(0.8*dimx), round(0.5*dimx)], [round(0.8*dimx), dimx], [round(0.8*dimx), 2*dimx],
    ]
    do_vae_train(outdir, data_train, data_valid, cand_params,
                 gaussian=gaussian, batchsize=batchsize, noiserate=2.0)

    # DAGMM
    outdir = os.path.join('models', dataname, 'dagmm')
    os.makedirs(outdir, exist_ok=True)
    cand_params = [
        [2, max(1,round(0.1*dimx)), round(0.5*dimx)], [2, max(1,round(0.1*dimx)), dimx], [2, max(1,round(0.1*dimx)), 2*dimx],
        [3, max(1,round(0.1*dimx)), round(0.5*dimx)], [3, max(1,round(0.1*dimx)), dimx], [3, max(1,round(0.1*dimx)), 2*dimx],
        [4, max(1,round(0.1*dimx)), round(0.5*dimx)], [4, max(1,round(0.1*dimx)), dimx], [4, max(1,round(0.1*dimx)), 2*dimx],
        [2, round(0.2*dimx), round(0.5*dimx)], [2, round(0.2*dimx), dimx], [2, round(0.2*dimx), 2*dimx],
        [3, round(0.2*dimx), round(0.5*dimx)], [3, round(0.2*dimx), dimx], [3, round(0.2*dimx), 2*dimx],
        [4, round(0.2*dimx), round(0.5*dimx)], [4, round(0.2*dimx), dimx], [4, round(0.2*dimx), 2*dimx],
    ]
    do_dagmm_train(outdir, data_train, data_valid, cand_params,
                   gaussian=gaussian, batchsize=batchsize)

    print('finish dataset:', dataname)
