import os
import sys
import json
import pickle
import numpy as np

from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import shap

from src.attributers.attributer import Attributer
from src import exptutil

import warnings
warnings.filterwarnings("ignore")


dataname, modeltype, scoretype, attrtype = \
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

seed = 1234

if (modeltype=='dagmm' or (modeltype=='vae' and scoretype=='negelbo')) \
    and (attrtype=='marg' or attrtype=='SFE'):
    raise ValueError('incompatible pair of detector and attributor')

print('start %s %s_%s_%s' % (dataname, modeltype, scoretype, attrtype))

datadir = os.path.join('data', 'processed', dataname)

# load data
data_train, data_valid, data_test = exptutil.load_data(datadir)
dim_x = data_test.shape[1]

data_test_norm = data_test[int(data_test.shape[0]/2):]
data_test_anom = data_test[:int(data_test.shape[0]/2)]

if attrtype=='sup':
    # build supervised model
    data_norm = np.concatenate([data_train, data_valid, data_test_norm], axis=0)
    X = np.concatenate([data_norm, data_test_anom], axis=0)
    y = np.concatenate([np.zeros(data_norm.shape[0]), np.ones(data_test_anom.shape[0])]).astype(int)
    X_res, y_res = SMOTE(random_state=12345).fit_resample(X, y)
    model = SVC().fit(X_res, y_res)
    explainer = shap.KernelExplainer(model.predict, shap.kmeans(X, k=32))
else:
    # load best model
    with open(os.path.join('models', dataname, modeltype, 'detection', 'best_model.json'), 'r') as fp:
        best_model = json.load(fp)
    model = exptutil.load_model(best_model[scoretype], modeltype)

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

# anomaly attribution
np.random.seed(seed+1)
attr_all = []
info_all = []
for i in range(data_test_anom.shape[0]):
    if attrtype=='sup':
        attr = np.array(explainer.shap_values(data_test_anom[i][np.newaxis,:], silent=True)[0])
        info = None
    else:
        attr, info = AT.attribute(data_test_anom[i], attrtype)
    attr_all.append(attr)
    info_all.append(info)
    # print('\r  attributed #%03d/%d' % (i+1, data_test_anom.shape[0]), end='')

# save
outdir = os.path.join('out', 'expt_real', dataname)
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, '%s_%s_%s.pickle' % (modeltype, scoretype, attrtype)), 'wb') as fp:
    pickle.dump([attr_all, info_all], fp)

print('finish %s %s_%s_%s' % (dataname, modeltype, scoretype, attrtype))

