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
#! baad to paper oomade ke 4 ta feature dar miarim:
#! 2,3,4-rectangle features ke mishe horizontal and vertical edge, line and diagonal edge


#! in hamoon concept e integral image e ke cumulative sum mizane roo image:
#! in shart e chie? if o else manzoorame. in vase padding e
#! vaghti az gooshe chap bala harkat mikonim miaim ro be paiin rast
#! in shart miad padding ro handle mikone
def integral_image(image):
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii

#! khob in che mikone? miad mohit e region haye mokhtalef ro mohasebe mikone
#! dar vaghe sum of the pixels in the region D tebgh e paper
class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def compute_feature(self, ii):
        return ii[self.y+self.height][self.x+self.width] + ii[self.y][self.x] - (ii[self.y+self.height][self.x]+ii[self.y][self.x+self.width])

#! khob class Viola-Jones asl karie
#! tanha hyper parameter ghabel e tanzim, hamin tedad weak classifier hast
#! idea chie? T ta weak classifier darim harkodoom ye bakhsh(region) az tasvir ro migire
class ViolaJones:
    #! in T chie? tedad weak classifier haii ke estefade mishe
    def __init__(self, T = 10):
        self.T = T
        self.alphas = []
        self.clfs = []

    def train(self, training, pos_num, neg_num):
        weights = np.zeros(len(training))
        training_data = []
        #! agha inja che mikonim? miaim aval integral image ro hesab mikonim
        #! oon training[1] label e image e va training[0] ham ke khod e tasvire
        #! mige hala mikhaim weight initial konim, baraye har class (0 ya 1) biaim
        #! ye vazn e sabet dar nazar begirim:
        #! w[i] = 1/2p for positive class and 1/2n for negative class
        
        for x in range(len(training)):
            training_data.append((integral_image(training[x][0]), training[x][1]))
            if training[x][1] == 1:
                weights[x] = 1.0 / (2 * pos_num)
            else:
                weights[x] = 1.0 / (2 * neg_num)
