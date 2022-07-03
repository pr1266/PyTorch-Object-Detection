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


class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
    #! agha too classifier 2 ta parameter darim:
        """
        polarity va threshold baraye weak Classifier ke
        ye khorooji h(x, f, p, t) bayad hesab konim
        h = 1 if age p f(x) < p th else 0
        agha baad in th va p baiad ba train optimal shan
        yani hamintori alaki nist ye constant bezarim
        """
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity
    
    def classify(self, x):
        
        feature = lambda ii: sum([pos.compute_feature(ii) for pos in self.positive_regions]) - sum([neg.compute_feature(ii) for neg in self.negative_regions])
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0
    

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
        features = self.build_features(training_data[0][0].shape)
        X, y = self.apply_features(features, training_data)

    def train_weak(self, X, y, features, weights):
        #! x ke hamoon feature han y ham label hashe (p ya n)
        #! features hamoon [[sum(pos)], [sum(neg)]] e
        #! w ham ke weight haye assign shode be har training sample e
        
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = []
        total_features = X.shape[0]
        for index, feature in enumerate(X):
            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print("Trained %d classifiers out of %d" % (len(classifiers), total_features))

            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])

            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            
            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers

    def select_best(self, classifiers, weights, training_data):
        #! in daiimoonam ke ye array az weak classifier ha migire
        #! miad behtarinesh ro bar migardoone:
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(training_data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy

    def build_features(self, image_shape):

        #! inja miaim hame oon 2,3,4-rectangle feature haro extract mikonim:
        #! saat 4 sobe va cop zadam injasho mese maaaaaaard
        #! mige element aval mishe positive element dovom mishe negative:
        height, width = image_shape
        features = []
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        #!2 rectangle features
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i+w, j, w, h)
                        if i + 2 * w < width: #!Horizontally Adjacent
                            features.append(([right], [immediate]))

                        bottom = RectangleRegion(i, j+h, w, h)
                        if j + 2 * h < height: #!Vertically Adjacent
                            features.append(([immediate], [bottom]))
                        
                        right_2 = RectangleRegion(i+2*w, j, w, h)
                        #!3 rectangle features
                        if i + 3 * w < width: #!Horizontally Adjacent
                            features.append(([right], [right_2, immediate]))

                        bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        if j + 3 * h < height: #!Vertically Adjacent
                            features.append(([bottom], [bottom_2, immediate]))

                        #!4 rectangle features
                        bottom_right = RectangleRegion(i+w, j+h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        return np.array(features)

    def apply_features(self, features, training_data):
        #! in features hamoon e ke tuple e va element aval pos va element dovom neg e
        #! training data ham ke havi integral image va label e
        """
        Maps features onto the training dataset
          Args:
            features: an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
          Returns:
            X: A numpy array of shape (len(features), len(training_data)). Each row represents the value of a single feature for each training example
            y: A numpy array of shape len(training_data). The ith element is the classification of the ith training example
        """
        X = np.zeros((len(features), len(training_data)))
        y = np.array(list(map(lambda data: data[1], training_data)))
        i = 0
        for positive_regions, negative_regions in features:
            feature = lambda ii: sum([pos.compute_feature(ii) for pos in positive_regions]) - sum([neg.compute_feature(ii) for neg in negative_regions])
            X[i] = list(map(lambda data: feature(data[0]), training_data))
            i += 1
        return X, y