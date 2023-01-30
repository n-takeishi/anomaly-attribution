import json
import datetime
import pickle

from src.models.gmm import GMM

######################

def train(args, data_train):
  # args : outdir, suffix, numcomps, save
  # data_train : ndarray

  # determine suffix
  if not args.suffix:
    args.suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

  # output settings
  if args.save:
    with open(args.outdir+'/gmm_args_%s.json'%(args.suffix), 'w') as fp:
      json.dump(vars(args), fp, sort_keys=True, indent=2)
  print(vars(args))

  # model
  gmm_model = GMM(n_components=args.numcomps, covariance_type='full')

  # training
  gmm_model.fit(data_train)

  # save
  if args.save:
    with open(args.outdir+'/gmm_model_%s.pickle'%(args.suffix), mode='wb') as fp:
      pickle.dump(gmm_model, fp)

  print('finish (gmm: %s)'%(args.suffix))
