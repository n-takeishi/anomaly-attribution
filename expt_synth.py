import os
import sys
import json
import pickle
import numpy as np

from src.attributers.attributer import Attributer
from src import exptutil

import warnings
warnings.filterwarnings("ignore")


nominal_gamma = 1e-2
seed = 1234


dataname, num_anom, modeltype, scoretype, attrtype = \
    sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]

if len(sys.argv) > 6:
    if attrtype not in ['comp', 'ASH', 'ASHexact']:
        raise ValueError('setting gamma for this attrtype does not make sense')
    gamma = float(sys.argv[6])
else:
    gamma = nominal_gamma

if (modeltype=='dagmm' or (modeltype=='vae' and scoretype=='negelbo')) \
    and (attrtype=='marg' or attrtype=='SFE'):
    raise ValueError('incompatible pair of detector and attributor')

if gamma==nominal_gamma:
    print('start %s na%d_%s_%s_%s' % (dataname, num_anom, modeltype, scoretype, attrtype))
else:
    print('start %s na%d_%s_%s_%s_%0.1e' % (dataname, num_anom, modeltype, scoretype, attrtype, gamma))

datadir = os.path.join('data', 'processed', dataname)

# load data
data_train, _, data_test = exptutil.load_data(datadir)
dim_x = data_test.shape[1]

# use only normal part (first half) of test data
data_test = data_test[:int(data_test.shape[0]/2)]

# load best model
with open(os.path.join('models', dataname, modeltype, 'detection', 'best_model.json'), 'r') as fp:
    best_model = json.load(fp)
model = exptutil.load_model(best_model[scoretype], modeltype)

# some settings
if dataname=='lympho':
    num_cases = 6
    num_tries = 20
else:
    num_cases = 60
    num_tries = 2
ub = 2.0
lb = 1.0

# create perturbed data
np.random.seed(seed)
case_indices = np.random.permutation(data_test.shape[0])
case_all = []
feat_all = []
target_all = []
for i in range(num_cases):
    for j in range(num_tries):
        case_all.append(case_indices[i])

        # target features
        feat_indices = np.random.permutation(data_test.shape[1])
        target_feat = feat_indices[:num_anom]

        # perturb values
        target = np.copy(data_test[case_indices[i]])
        for idx in target_feat:
            if dataname=='lympho':
                target[idx] = 1-target[idx]
            else:
                sign=[-1,1]; sign = sign[np.random.choice(2)]
                perturbation = sign*(np.random.rand()*(ub-lb)+lb)
                target[idx] += perturbation

        feat_all.append(target_feat)
        target_all.append(target)

# set attributer
if modeltype=='gmm':
    detector_type = 'GMM'
elif modeltype=='vae' and scoretype=='recerr':
    detector_type='VAE-r'
elif modeltype=='vae' and scoretype=='negelbo':
    detector_type='VAE-e'
elif modeltype=='dagmm':
    detector_type='DAGMM'
else:
    raise ValueError('unknown model/score type')
AT = Attributer(detector_type, model, data_train, False if dataname=='lympho' else True)

# do attribution
np.random.seed(seed+1)
attr_all = []
info_all = []
for i in range(len(target_all)):
    attr, info = AT.attribute(target_all[i], attrtype, gamma=gamma)
    attr_all.append(attr)
    info_all.append(info)
    # print('\r  attributed #%03d/%d' % (i+1, len(target_all)), end='')

# save
outdir = os.path.join('out', 'expt_synth', dataname)
os.makedirs(outdir, exist_ok=True)
if gamma==nominal_gamma:
    outname = os.path.join(outdir, 'na%d_%s_%s_%s.pickle' % (num_anom, modeltype, scoretype, attrtype))
else:
    outname = os.path.join(outdir, 'na%d_%s_%s_%s_%0.1e.pickle' % (num_anom, modeltype, scoretype, attrtype, gamma))
with open(outname, 'wb') as fp:
    pickle.dump([attr_all, info_all, case_all, feat_all], fp)

if gamma==nominal_gamma:
    print('finish %s na%d_%s_%s_%s' % (dataname, num_anom, modeltype, scoretype, attrtype))
else:
    print('finish %s na%d_%s_%s_%s_%0.1e' % (dataname, num_anom, modeltype, scoretype, attrtype, gamma))
