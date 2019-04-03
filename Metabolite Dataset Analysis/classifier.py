### File - classifier.py

from sklearn import tree
from datacontainer import *
from gen_combinations import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as NaiveBayes
from sklearn.linear_model import LogisticRegression as LR

''' HYPERPARAMS FOR DECISION TREE
 
 These parameters implement a rudimentary pruning algorithm, would ideally like to use AB pruning'''
enable_pruning = True
# maximum depth of dtree
max_depth = 5
# how many samples your need atleast, at a LEAF node
min_samples = 3


def testDecisionTree(trainx, trainy, testx, testy, cols):
    '''
    Tests a decision tree on given input data.
    :param trainx: training data
    :param trainy: training labels
    :param testx: test data
    :param testy: test labels
    :param cols: the chosen cols to use for the test
    :return: error, cols as a string, auc, roc
    '''

    # fit tree
    if enable_pruning:
        clf = tree.DecisionTreeClassifier(presort=True, max_depth=max_depth, min_samples_leaf=min_samples)
        clf.fit(trainx, trainy)

    else:
        clf = tree.DecisionTreeClassifier(presort=True)
        clf.fit(trainx, trainy)

    # error check
    corr = 0
    total = 0

    pred = clf.predict(testx)

    for p in pred:
        if p == testy[total]: corr += 1
        total += 1

    # do auc and roc here
    auc = roc_auc_score(testy, pred)
    # print(list(pred))
    # print(testy)
    # print()
    # roc = roc_curve(testy, pred)
    return corr / total, ''.join(["%02d" % x for x in cols])


def testRFW(trainx, trainy, testx, testy, cols):
    '''
    Tests a random forest walk classifier on given input data.
    :param trainx: training data
    :param trainy: training labels
    :param testx: test data
    :param testy: test labels
    :param cols: the chosen cols to use for the test
    :return: error, cols as a string
    '''

    # train classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(trainx, trainy)
    # error check
    corr = 0
    total = 0
    pred = list(clf.predict(testx))
    for p in pred:
        if p == testy[total]: corr += 1
        total += 1
    auc = roc_auc_score(testy, pred)
    # roc = roc_curve(testy, pred)
    return corr / total, ''.join(["%02d" % x for x in cols])


def testNaiveBayes(trainx, trainy, testx, testy, cols):
    '''
    Tests a Naive Bayes classifier on given input data.
    :param trainx: training data
    :param trainy: training labels
    :param testx: test data
    :param testy: test labels
    :param cols: the chosen cols to use for the test
    :return: error, cols as a string
    '''

    # train classifier
    clf = NaiveBayes()
    clf.fit(trainx, trainy)
    # error check
    corr = 0
    total = 0
    pred = list(clf.predict(testx))
    for p in pred:
        if p == testy[total]: corr += 1
        total += 1
    auc = roc_auc_score(testy, pred)
    # roc = roc_curve(testy, pred)
    return corr / total, ''.join(["%02d" % x for x in cols])


def testLogisticRegressor(trainx, trainy, testx, testy, cols):
    '''
    Tests a logistic regression classifier on given input data.
    :param trainx: training data
    :param trainy: training labels
    :param testx: test data
    :param testy: test labels
    :param cols: the chosen cols to use for the test
    :return: error, cols as a string
    '''

    # train classifier
    clf = LR(class_weight='balanced', solver='liblinear', multi_class='ovr')
    clf.fit(trainx, trainy)
    # error check
    corr = 0
    total = 0
    pred = list(clf.predict(testx))
    for p in pred:
        if p == testy[total]: corr += 1
        total += 1
    auc = roc_auc_score(testy, pred)
    # roc = roc_curve(testy, pred)
    return corr / total, ''.join(["%02d" % x for x in cols])


def testKNN(trainx, trainy, testx, testy, cols):
    '''
    Tests a KNN classifier on given input data.
    :param trainx: training data
    :param trainy: training labels
    :param testx: test data
    :param testy: test labels
    :param cols: the chosen cols to use for the test
    :return: error, cols as a string
    '''

    # train classifier
    clf = KNN(n_neighbors=5, n_jobs=-1)
    clf.fit(trainx, trainy)
    # error check
    corr = 0
    total = 0
    pred = list(clf.predict(testx))
    for p in pred:
        if p == testy[total]: corr += 1
        total += 1
    auc = roc_auc_score(testy, pred)
    # roc = roc_curve(testy, pred)
    return corr / total, ''.join(["%02d" % x for x in cols])
