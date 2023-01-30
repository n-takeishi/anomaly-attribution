import time
import numpy as np

from sklearn.neighbors import KDTree
import shap

from src.attributers import anomscore
from src.attributers import shapley
from src.attributers import intgrad
from src.attributers import ace
from src.attributers import sfe


class eFunc:
  def __init__(self, e):
    # e : python function, e(*args)
    #     This function can be defined for batch x.
    self.e = e
    self.count = 0

  def __call__(self, *args):
    e_ = self.e(*args)
    self.count+=e_.shape[0]
    return e_

  def zero_count(self):
    self.count=0


class vFunc():
  def __init__(self, e_hat, e_th):
    self.e_hat = e_hat
    self.e_th = e_th
    self.count = 0

  def __call__(self, S):
    self.count += 1
    e_hat_ = self.e_hat(S)
    return np.fmin(e_hat_, self.e_th), e_hat_>=self.e_th

  def zero_count(self):
    self.count=0


class Attributer():
  def __init__(self, detector_type, model, trdata, gaussian):

    dim_x = trdata.shape[1]
    self.dim_x = dim_x
    self.model = model
    self.detector_type = detector_type
    self.trdata = trdata
    self.gaussian = gaussian

    self.marginscorer = None
    self.allmarginscorer = None

    if detector_type=='GMM':
      self.ef = eFunc(lambda x: anomscore.gmm_energy(x, dim_x, model))
      self.marginscorer = lambda S,x: anomscore.gmm_marginenergy(S, x, dim_x, model)
      self.allmarginscorer = lambda x: anomscore.gmm_allmarginenergy(x, dim_x, model)[0]
    elif detector_type=='VAE-r':
      self.ef = eFunc(lambda x: anomscore.vae_recerr(x, dim_x, model))
      self.marginscorer = lambda S,x: anomscore.vae_marginrecerr(S, x, dim_x, model)
      self.allmarginscorer = lambda x: anomscore.vae_allmarginrecerr(x, dim_x, model)[0]
    elif detector_type=='VAE-e':
      self.ef = eFunc(lambda x: anomscore.vae_negelbo(x, dim_x, model))
    elif detector_type=='DAGMM':
      self.ef = eFunc(lambda x: anomscore.dagmm_energy(x, dim_x, model))
    else:
      raise ValueError('Unknown setting: detector_type=%s' % detector_type)

    if trdata is not None:
      self.trdata_mean = np.mean(trdata, axis=0)
      self.trdata_example = shap.kmeans(trdata, 8).data
      self.trdata_kdtree = KDTree(trdata)


  def evalue(self, data):
    data = data.reshape(-1, self.dim_x)
    return self.ef(data)


  def attribute(self,
    datum,
    method,
    seed=1234567890,
    gamma=1e-2
  ):
    assert datum.ndim==1, 'only a single data point can be treated at once'
    assert datum.size==self.dim_x

    info = {}

    # attribution by marginal score
    if method=='marg':
      if self.allmarginscorer is not None:
        self.ef.zero_count()
        start = time.time()
        attr = self.allmarginscorer(datum)
        info['duration'] = time.time()-start
        info['funcount'] = self.ef.count
      else:
        raise ValueError('marg is specified as method, but no allmarginscorer is prepared')

    # attribution by Integrated Gradient
    elif method=='IG':
      if self.trdata_mean is not None:
        self.ef.zero_count()
        start = time.time()
        attr = intgrad.intgrad(datum, self.trdata_mean, self.ef)
        info['duration'] = time.time()-start
        info['funcount'] = self.ef.count
      else:
        raise ValueError('IG is specified as method, but no trdata is prepared')

    # attribution by ACE
    elif method=='ACE':
      self.ef.zero_count()
      start = time.time()
      attr = ace.ace(datum, self.ef, gaussian=self.gaussian, N=32, seed=seed, kl=False)
      info['duration'] = time.time()-start
      info['funcount'] = self.ef.count

    # attribution by SFE
    elif method=='SFE':
      if self.marginscorer is not None:
        tmp = eFunc(lambda S,x: self.marginscorer(S,x))
        self.ef.zero_count()
        start = time.time()
        sfe_order, _ = sfe.seqmarg(datum, tmp)
        info['order'] = sfe_order
        info['duration'] = time.time()-start
        info['funcount_e'] = self.ef.count
        info['funcount_m'] = tmp.count
        point = self.dim_x
        sfe_point = [0 for i in range(self.dim_x)]
        for i in sfe_order:
          sfe_point[i] = point
          point -= 1
        attr = np.array(sfe_point)
      else:
        raise ValueError('SFE is specified as method, but no marginscorer is prepared')

    # attribution by KernelSHAP
    elif method=='KSH':
      if self.trdata_example is not None:
        np.random.seed(seed)
        self.ef.zero_count()
        start = time.time()
        elr = shap.KernelExplainer(lambda x: np.fmin(self.ef(x),1e10), self.trdata_example,
          link='identity' if self.gaussian else 'logit')
        attr = elr.shap_values(datum, nsamples='auto', l1_reg=0)
        info['duration'] = time.time()-start
        info['funcount'] = self.ef.count
      else:
        raise ValueError('KSH is specified as method, but no trdata is provided')

    # attribution by weighted KernelSHAP
    elif method=='wKSH':
      if self.trdata is not None:
        np.random.seed(seed)
        self.ef.zero_count()
        start = time.time()
        _, idx = self.trdata_kdtree.query(datum.reshape(1,-1), k=8)
        background_data = self.trdata[idx][0]
        elr = shap.KernelExplainer(lambda x: np.fmin(self.ef(x),1e10), background_data,
          link='identity' if self.gaussian else 'logit')
        attr = elr.shap_values(datum, nsamples='auto', l1_reg=0)
        info['duration'] = time.time()-start
        info['funcount'] = self.ef.count
      else:
        raise ValueError('wKSH is specified as method, but no trdata is provided')

    # attribution by AnomSHAP
    elif method=='ASH':
      self.ef.zero_count()
      start = time.time()
      baseliner = anomscore.BaselinerRelaxed(datum, self.dim_x, self.ef, {'gaussian':self.gaussian, 'gamma':gamma})
      vf = vFunc(baseliner, 1e10)
      elr = shapley.wls(self.dim_x, self.dim_x*2+2048, seed=seed)
      attr, _ = elr(vf, do_bound=False)
      info['duration'] = time.time()-start
      info['funcount_e'] = self.ef.count
      info['funcount_v'] = vf.count

    # attribution by AnomSHAP (Monte Carlo) without the final relaxation
    elif method=='ASHexact':
      self.ef.zero_count()
      start = time.time()
      baseliner = anomscore.Baseliner(datum, self.dim_x, self.ef, {'gaussian':self.gaussian, 'gamma':gamma})
      vf = vFunc(baseliner, 1e10)
      elr = shapley.wls(self.dim_x, self.dim_x*2+2048, seed=seed)
      attr, _ = elr(vf, do_bound=False)
      info['duration'] = time.time()-start
      info['funcount_e'] = self.ef.count
      info['funcount_v'] = vf.count

    # attribution by AnomSHAP's baseline vector (no Shapley value computation)
    elif method=='comp':
      self.ef.zero_count()
      start = time.time()
      baseliner = anomscore.Baseliner(datum, self.dim_x, self.ef, {'gaussian':self.gaussian, 'gamma':gamma})
      attr = np.sqrt((baseliner([]) - datum)**2)
      info['duration'] = time.time()-start
      info['funcount_e'] = self.ef.count

    # AnomSHAP with 5-NN anomaly score
    elif method=='ASH5nn':
      start = time.time()
      def vf(S):
        S = list(S)
        nonS = anomscore._compset(S, self.dim_x)
        if len(nonS) > 0:
          dists = np.sqrt( np.mean((self.trdata_all[:,nonS] - datum[np.newaxis,nonS])**2, axis=1) )
          idxs = np.argsort(dists)
          min_dist = np.mean(dists[idxs[:5]])
        else:
          min_dist = 0.0
        return min_dist, None
      elr = shapley.wls(self.dim_x, self.dim_x*2+2048, seed=seed)
      attr, _ = elr(vf, do_bound=False)
      info['duration'] = time.time()-start

    return attr, info
