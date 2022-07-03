import numpy as np
import math
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif

#! fargh in algorithm ba CNN-Based algorithms chie?
#! cnn ye classifier dare ke baad az feature extractione
#! ama in miad feature haye mokhtalef az image ro extract mikone
#! va mide be T ta weak classifier ke combination ina
#! mishe ye classifier e ghavi (AdaBoost)
#! tedad parameter hash ham az CNN ha kamtare
#! ye concept ham darim taht e onvan e integral image
#! in oomade feature extraction ro be soorat e DP anjam dade
#! manteghesh ham cumulative sum az samt e chap va balast baraye har pixel

class ViolaJones:
    #! in T chie? tedad weak classifier haii ke estefade mishe
    def __init__(self, T = 10):
        self.T = T
        self.alphas = []
        self.clfs = []