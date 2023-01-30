import glob
import os
import pickle
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import sklearn.metrics

from src.models.vae import VAE
from src.models.dagmm import DAGMM


def get_pathlist(directory, prefix):
  pathlist_all = glob.glob(directory+'/*')
  if len(pathlist_all)<1:
    return []

  pathlist = []
  for path in pathlist_all:
    filename = os.path.basename(path)
    if filename.startswith(prefix):
      pathlist.append(path)
  return pathlist


def load_data(datadir):
  data_train = np.loadtxt(datadir+'/data_train.txt', ndmin=2)
  data_valid = np.loadtxt(datadir+'/data_valid.txt', ndmin=2)
  data_test = np.loadtxt(datadir+'/data_test.txt', ndmin=2)
  return data_train, data_valid, data_test


def load_model(modelfile, modeltype):
  if modeltype=='gmm':
    with open(modelfile, mode='rb') as f:
      model = pickle.load(f)

  elif modeltype=='vae':
    with open(modelfile.replace('_model_','_args_').replace('.pt','.json'), mode='r') as f:
      args = json.load(f)
    model = VAE(args['dimx'], args['dimz'], args['dimhidden'], args['activation'],
      batchnorm=args['batchnorm'], gaussian=args['gaussian'])
    model.load_state_dict(torch.load(modelfile))

  elif modeltype=='dagmm':
    with open(modelfile.replace('_model_','_args_').replace('.pt','.json'), mode='r') as f:
      args = json.load(f)
    model = DAGMM(args['dimx'], args['dimz'], args['dimhidden'], args['numcomps'], args['activation'],
      batchnorm=args['batchnorm'], gaussian=args['gaussian'])
    model.load_state_dict(torch.load(modelfile))

  else:
    raise ValueError('Unknown model type.')

  return model


def plot_attr(attr, anofeats=None, do_abs=False):
  for i, key in enumerate(attr):
    tmp = attr[key]
    if do_abs:
      tmp = np.abs(tmp)
    #tmp = attr[key] / np.max(np.abs(attr[key]))
    ax = plt.subplot(len(attr),1,i+1)
    ax.bar(range(tmp.size), tmp, label=key)
    if anofeats is not None:
      ax.bar(anofeats, tmp[anofeats])
    ax.legend()
  plt.show()


def hitsatk(attr_all, feat_all, k, do_abs=False):
  correct = 0.0
  total = 0.0

  for i in range(len(attr_all)):
    attr = attr_all[i]
    feat = feat_all[i]

    if attr is None or feat is None:
      continue

    truth = feat_all[i].item()
    attr = np.nan_to_num(attr, nan=0.0)
    if do_abs:
      attr = np.abs(attr)
    detected = np.argsort(attr)[::-1][:k]
    if truth in detected:
      correct += 1.0
    total += 1.0

  return correct/total


def avgrankcorr(attr_all1, attr_all2, do_abs=False):
  assert len(attr_all1) == len(attr_all2)

  corr = []
  for attr1, attr2 in zip(attr_all1, attr_all2):
    attr1 = np.nan_to_num(attr1, nan=0.0)
    attr2 = np.nan_to_num(attr2, nan=0.0)
    if do_abs:
      attr1 = np.abs(attr1)
      attr2 = np.abs(attr2)
    rank1 = score_to_rank(attr1)
    rank2 = score_to_rank(attr2)

    n = len(attr1)
    corr.append(1.0 - (6.0*np.sum((rank1-rank2)**2) / (n*(n**2 - 1.0))))
  corr = np.array(corr)

  z = np.arctanh(np.fmax(np.fmin(corr, 1.0-1e-6), -1.0+1e-6))
  return np.tanh(np.mean(z))


def score_to_rank(score):
  o = np.argsort(score)[::-1]
  rank = np.zeros(score.size)
  c = 1
  for i in range(score.size):
      rank[o[i]] = c
      c += 1
  return rank


def recrank(attr_all, feat_all, do_abs=False):
  recrank = []
  for i in range(len(attr_all)):
    attr = attr_all[i]
    feat = feat_all[i]

    if attr is None or feat is None:
      continue

    truth = feat.item()
    attr = np.nan_to_num(attr, nan=0.0)
    if do_abs:
      attr = np.abs(attr)

    rank = score_to_rank(attr)
    recrank.append(1.0/rank[truth])

  return recrank


def auc(attr_all, feat_all, do_abs=False):
  num_feat = attr_all[0].shape[0]
  auc = []
  for i in range(len(attr_all)):
    attr = attr_all[i]
    feat = feat_all[i]

    if attr is None or feat is None:
      continue

    truth = np.zeros(num_feat)
    truth[feat] = 1
    attr = np.nan_to_num(attr, nan=0.0)
    if do_abs:
      attr = np.abs(attr)

    auc.append(sklearn.metrics.roc_auc_score(truth, attr, average='macro'))

  return auc
