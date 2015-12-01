import math
import numpy as np

#from nflEvaluation import *
def iFunction(y1, y2):
    i = 0.0
    if abs(y1-y2) < 0.0001:
        i = 1.0
    return i
def gFunction(z):
    g = 0
    if z>=8:
        g = 1
    return g
def sFunction(z):
    return 1
#from nflEvaluation import *

def restorePlayType(classType):
    PlayType = math.floor(classType / float(10)) + 1
    Result = classType % 10
    return PlayType, Result


class classifier():
    def classify(self, data, target):
        self.clf.fit(data, target)
        y_pred = self.clf.predict(data)
        return y_pred
    def predict(self, data):
        return self.clf.predict(data)
    def switchPlayType(self, PlayType):
        pType = 0
        if abs(PlayType-1) < 0.0001:
            pType = 2
        if abs(PlayType-2) < 0.0001:
            pType = 1
        return pType
    def recommendationSingle(self,classType):
        PlayType, Result = restorePlayType(classType)
        if gFunction(Result) < 0.0001:
            PlayType = self.switchPlayType(PlayType)
        recommendationCalss = (PlayType-1)*10 + Result
        return recommendationCalss
    def recommendation(self, predict):
        recommendation = np.zeros(np.size(predict))
        for i in range(np.size(predict)):
            recommendation[i] = self.recommendationSingle(predict[i])
            #print predict[i]
            #print recommendation[i]
        return recommendation

class knClassifier(classifier):
    def __init__(self):
        from sklearn.neighbors import KNeighborsClassifier
        self.clf = KNeighborsClassifier(4)
class svmClassifier(classifier):
    def __init__(self):
        from sklearn import svm
        self.clf = svm.SVC(gamma=0.001, C=100.)
    def linear(self, C=0.025):
        from sklearn import svm
        self.clf = svm.SVC(kernel="linear", C=C)
    def gamma(self, gamma=2, C=1):
        from sklearn import svm
        self.clf = svm.SVC(gamma=gamma, C=C)
class BayesClassifier(classifier):
    def __init__(self):
        from sklearn.naive_bayes import GaussianNB
        self.clf = GaussianNB()
class dtClassifier(classifier):
    def __init__(self):
        from sklearn.tree import DecisionTreeClassifier
        self.clf = DecisionTreeClassifier(max_depth=5)
class rfClassifier(classifier):
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
class adaBoostClassifier(classifier):
    def __init__(self):
        from sklearn.ensemble import AdaBoostClassifier
        self.clf = AdaBoostClassifier()
class ldClassifier(classifier):
    def __init__(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.clf =  LinearDiscriminantAnalysis()
class ldClassifier(classifier):
    def __init__(self):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        self.clf = QuadraticDiscriminantAnalysis()
