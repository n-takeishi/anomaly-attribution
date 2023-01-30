import os
import json
import numpy as np
import sklearn.metrics
from collections import OrderedDict

from src.attributers import anomscore
from src import exptutil


for dataname in ['thyroid', 'breastw', 'lympho', 'musk', 'arrhythmia', 'U2R']:
    datadir = os.path.join('data', 'processed', dataname)

    for modeltype in ['gmm', 'vae', 'dagmm']:
        modeldir = os.path.join('models', dataname, modeltype)
        outdir = os.path.join(modeldir, 'detection')

        # load data
        data_train, data_valid, data_test = exptutil.load_data(datadir)
        dim_x = data_train.shape[1]
        label_test = np.hstack([np.zeros(int(data_test.shape[0]/2)), np.ones(int(data_test.shape[0]/2))])

        # make outdir
        os.makedirs(outdir, exist_ok=True)

        ##### change here #####
        scorefuncs = OrderedDict()
        if modeltype=='gmm':
            scorefuncs['energy'] = lambda data,model: anomscore.gmm_energy(data, dim_x, model)
        elif modeltype=='vae':
            scorefuncs['recerr'] = lambda data,model: anomscore.vae_recerr(data, dim_x, model)
            scorefuncs['negelbo'] = lambda data,model: anomscore.vae_negelbo(data, dim_x, model)
        elif modeltype=='dagmm':
            scorefuncs['energy'] = lambda data,model: anomscore.dagmm_energy(data, dim_x, model)

        # get model list
        modelpaths = exptutil.get_pathlist(modeldir, modeltype+'_model')

        # try each model
        aucs_test = np.zeros((len(modelpaths), len(scorefuncs)))
        for i, modelpath in enumerate(modelpaths):
            # load model
            model = exptutil.load_model(modelpath, modeltype)

            # compute anomaly scores
            for j, scorefuncname in enumerate(scorefuncs):
                scorefunc = scorefuncs[scorefuncname]
                score_train = scorefunc(data_train, model)
                score_valid = scorefunc(data_valid, model)
                score_test = scorefunc(data_test, model)

                # compute AUC
                aucs_test[i,j] = sklearn.metrics.roc_auc_score(label_test, score_test, average='macro')

        # save
        np.savetxt(os.path.join(outdir, 'aucs_test.txt'), aucs_test)
        with open(outdir+'modelpaths.txt', 'w') as fp:
            for modelpath in modelpaths:
                fp.write(modelpath+'\n')
        with open(outdir+'scorenames.txt', 'w') as fp:
            for scorename in scorefuncs:
                fp.write(scorename+'\n')

        # examine best model for each score type
        best_model_idxs = np.argmax(aucs_test, axis=0)
        best_model = {}
        best_model_auc = {}
        for j, scorename in enumerate(scorefuncs):
            best_model[scorename] = modelpaths[best_model_idxs[j]]
            best_model_auc[scorename] = aucs_test[best_model_idxs[j],j]

        # save best model information
        with open(os.path.join(outdir,'best_model.json'), 'w') as fp:
            fp.write(json.dumps(best_model))
        with open(os.path.join(outdir,'best_model_auc.json'), 'w') as fp:
            fp.write(json.dumps(best_model_auc))

        print(best_model, best_model_auc)
