from csv import DictReader, DictWriter
import numpy as np
from itertools import product

from extractor import NewPbpExtractor
from nflEvaluation import classifierEvaluation
from nflClassifier import *
from sklearn.cross_validation import train_test_split

mc = 2
#--------------------------------------------------#
#training
#data2: combine 2014, first 80% for train, 20% for test
data2013 = list(DictReader(open("pbp-2013.csv", 'r')))
data2014 = list(DictReader(open("pbp-2014.csv", 'r')))
data2015 = list(DictReader(open("pbp-2015.csv", 'r')))
dataList = [data2013, data2014, data2015]
dataName = ["2013","2014","2015"]

o = DictWriter(open("rfClassifier-mc.csv", 'w'), ["dataName", "classifier", "percent", "score", "OmniScore", "Type1-A/A/Good","Type2-A/B/Bad",  "Type3-A/B/Good", "Type4-A/A/Bad"])
o.writeheader()

#train data 2015
data13and14 = data2013+data2014
pbp2014 = NewPbpExtractor()
pbp2014.buildFormationList(data13and14)
feature, target = pbp2014.extract4Classifier(data13and14)
X_train = feature
y_train = target

#test data 2015
pbp2014 = NewPbpExtractor()
pbp2014.buildFormationList(data2015)
feature, target = pbp2014.extract4Classifier(data2015)
X_test = feature
y_test = target

class2014 = classifierEvaluation()

Bscore, Bnum, BtypeNum, BomniScore = class2014.Score(y_test, y_test)
baseline = {'dataName': 2015 ,'classifier': 'baseline', 'percent': Bscore/float(BomniScore),'score': Bscore,'OmniScore': BomniScore, 'Type1-A/A/Good': BtypeNum[0], 'Type2-A/B/Bad': BtypeNum[1], 'Type3-A/B/Good': BtypeNum[2], 'Type4-A/A/Bad': BtypeNum[3]}
o.writerow(baseline)

#--------------------------------------------------#
clf = []
max_depth = 9
splitter = ["best", "random"]
#max_features = np.linspace(7, 16, num = 16-7+1)
#criterion = ["gini", "entropy"]
algorithm = ["SAMME", "SAMME.R"]
method = ["dt", "svm"]

algorithmForkn = ["ball_tree", "brute", "auto"]
n_neighbors  = np.linspace(3, 8, num = 8-3+1)
metric = ["minkowski", "euclidean", "manhattan", "chebyshev"]

#C = [.0001, .001, .01, .1, 1, 10, 100, 1000]
C = [10]
#kernel = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
kernel = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
max_features = [11]
criterion = ["entropy"]

clf.append(rfClassifier())
#for a, b, c in product(max_depth, criterion, max_features):
clf.append( rfClassifier(max_depth=9, criterion="entropy", max_features=10) )

for i in range(len(clf)):
    clf[i].classify(X_train, y_train, needWeight=False)
    temp = clf[i].predict(X_test)
    y_pred = clf[i].recommendation(temp)

    score, num, typeNum, omniScore = class2014.Score(y_test, y_pred)

    clfCVS = {'dataName': '2015', 'classifier': clf[i].name, 'percent': score/float(omniScore),'score': score,'OmniScore': omniScore, 'Type1-A/A/Good': typeNum[0], 'Type2-A/B/Bad': typeNum[1], 'Type3-A/B/Good': typeNum[2], 'Type4-A/A/Bad': typeNum[3]}
    o.writerow(clfCVS)
