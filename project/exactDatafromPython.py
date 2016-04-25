import sys
import csv

from csv import DictReader, DictWriter
import numpy as np
from itertools import product

from extractor import NewPbpExtractor
from nflEvaluation import classifierEvaluation
from nflClassifier import *
from sklearn.cross_validation import train_test_split

mc = 10
#--------------------------------------------------#
#training
#data2: combine 2014, first 80% for train, 20% for test
data2013 = list(DictReader(open("pbp-2013.csv", 'r')))
data2014 = list(DictReader(open("pbp-2014.csv", 'r')))
data2015 = list(DictReader(open("pbp-2015.csv", 'r')))
dataList = [data2013, data2014, data2015]
dataName = ["2013","2014","2015"]

#o = DictWriter(open("rfClassifier-mc.csv", 'w'), ["dataName", "classifier", "percent", "score", "OmniScore", "Type1-A/A/Good","Type2-A/B/Bad",  "Type3-A/B/Good", "Type4-A/A/Bad"])
#o.writeheader()
csvfile = ["2013X.csv","2013Y.csv","2014X.csv","2014Y.csv","2015X.csv","2015Y.csv"]

num = -1
#---------------------------------#
for dataindex in range(len(dataList)):
    pbp2014 = NewPbpExtractor()
    pbp2014.buildFormationList(dataList[dataindex])
    feature, target = pbp2014.extract4Classifier(dataList[dataindex])

    num += 1
    with open(csvfile[num], "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(feature)

    num += 1
    with open(csvfile[num], "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(target)
