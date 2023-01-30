'''
An (unofficial) implementation of:
Siddiqui et al., Sequential feature explanations for anomaly detection, ACM TKDD 13:1, 2019.
'''

import numpy as np


def seqmarg(x_expl, ef):
  # ef(S, x) : function that returns x's marginal energy
  #            (i.e., negative log marginal density) wrt S
  assert x_expl.ndim == 1
  dim_x = x_expl.size

  E = []
  unused_features = [i for i in range(dim_x)]
  max_energy_old = 0.0
  increased_energy = []
  while(len(unused_features)>0):
    energies = [0.0 for i in range(len(unused_features))]
    for i in range(len(unused_features)):
      new_feature = unused_features[i]
      energies[i] = ef(E+[new_feature,], x_expl)

    max_energy = max(energies)
    chosen_idx = energies.index(max_energy)
    chosen_feature = unused_features[chosen_idx]

    unused_features.pop(chosen_idx)
    E.append(chosen_feature)

    increased_energy.append((max_energy-max_energy_old).item())
    max_energy_old = max_energy

  return E, increased_energy
