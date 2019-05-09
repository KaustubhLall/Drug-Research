### File - main.py

import datetime
from time import time
from itertools import combinations
from tqdm import tqdm

# index where the class is
classind = 0

# distance from first index where data appears
startat = 1
# distance backwards from last index where data appears
stopat = 0

import math


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)


'''
This file will have a method to generate all possible n C k features of a given dataset.
The dataset in question has to be pre-processed and some columns may be dropped. For our set, we drop the first 6 
columns, although only after we extract features from column 3 (index 2).

NOTES:
 To get a column by name from a datacontainer d, use d.header.index('name of column header') and pass that result
 into d.getcol().
'''

from classifier import *
from datacontainer import *
from gen_combinations import *
from sklearn.metrics import roc_auc_score


def preprocess(target, fname='oat1oat3preprocessed.csv'):
    """
    Pre-processing function, specifically for our dataset. Currently NOT used, because dropcol() functionality is not
    correctly working.
    :param target: target csv to write output to.
    :param fname: source csv file to preprocess.
    :return: None. All operations done inside fn call.
    """

    df = DataContainer(fname)

    for e in [5, 4, 1, 0]:
        df.dropcol(e)

    classes = df.getcol(0)
    df.dropcol(0)

    df.writeCSV(target)


def runall(testname, trainname, results, k=5):
    """
    Function to run all classifiers (decision tree, random forest walk, naive bayes, logistic regression, k nearest
    neighbor) through the test data to calculate auc scores.

    Takes in a test file and a training file, and writes to the results file. k specifies the number of features to
    select.

    :param testname: name of the file that contains the test data.
    :param trainname: name of the file that contains the training data.
    :param results: name of the file to write results to.
    :param k: the number of features to target selecting.
    :return: No return value. Reads and writes from disk and prints to stdout.
    """

    '''
    Pre-processing steps: 
    1. load the data from the files
    2. extract the column that contains the labels.
    3. remove the first 6 columns as they are metadata + labels.
    4. construct a list with all feature combinations possible.
    5. make variables for progress, and the best known scores for every classifier, to display as stats.
    6. run all sequences, use a subsetDataContainer(s) for every s in seq (list of all sequences).
    7. rest of description inside loop.
    '''
    # load test and training "main" datacontainers
    testx = DataContainer(testname)
    trainx = DataContainer(trainname)

    # extract labels and drop the columns
    testy = [x[0] for x in subsetDataContainer(testx, [classind]).dataMatrix]
    trainy = [x[0] for x in subsetDataContainer(trainx, [classind]).dataMatrix]

    # clean columns we dont need
    testx = subsetDataContainer(testx, list(range(startat, testx.numcols - stopat)))
    trainx = subsetDataContainer(trainx, list(range(startat, trainx.numcols - stopat)))

    with open('testtable.txt', 'w') as f:
        f.write(testx.__repr__())

    with open('traintable.txt', 'w') as f:
        f.write(trainx.__repr__())

    # generate all feature sequences
    print('generating sequences...')
    # seq = generateSequences(testx.numcols, k)
    # print(seq)
    # print('generated %d sequences' % len(seq))
    # totalseq = len(seq)

    f = open(results, 'w')

    prog = 0
    bestDT = 0, ''
    bestRFW = 0, ''
    bestLR = 0, ''
    bestNB = 0, ''
    bestKNN = 0, ''

    # run the sequences via a decision tree and then repeat for RFW and other classifiers
    print("starting test..., %d choose %d sequences" % (trainx.numcols, k))
    # f.write(','.join(testx.header))
    timeStart = time()

    seq = combinations(list(range(testx.numcols)), k)
    totalseq = nCr(testx.numcols, k)

    for pattern in seq:
        # tqdm(
        # seq,
        # ncols=80,
        # desc='Classifying samples',
        # ):
        '''
        Get a DataContainer of the test and traning data.
        
        First, subset the container, second extract the dataMatrix.
        
        Run all 5 classifiers, and store the results in a variable resClassifierType. The result is a tuple with AUC
        as well as the features used to arrive at a classification.
        
        Check if any of the new scores this iteration beat their respective best previous scores.
        
        Maintain a list of features tried so far, so we can write to file. Since all classifiers run on the same feature
        space each iteration, we really only need one list, however it is nice to have one for each classifier.
        
        Write the results to file, and also display the current bests and the ETA. 
        
        '''
        # Get a DataContainer of the test and traning data.
        # First, subset the container, second extract the dataMatrix.
        subtrainx = subsetDataContainer(trainx, pattern)
        subtestx = subsetDataContainer(testx, pattern)

        header = trainx.header
        subtrainx = subtrainx.dataMatrix
        subtestx = subtestx.dataMatrix

        # Run all 5 classifiers, and store the results in a variable resClassifierType. The result is a tuple with AUC
        # as well as the features used to arrive at a classification.
        resDT = testDecisionTree(subtrainx, trainy, subtestx, testy, pattern)
        resRFW = testRFW(subtrainx, trainy, subtestx, testy, pattern)

        # do auc and roc comparisons here

        # Check if any of the new scores this iteration beat their respective best previous scores.
        if resDT[0] > bestDT[0]:
            bestDT = resDT

        if resRFW[0] > bestRFW[0]:
            bestRFW = resRFW

        # Maintain a list of features tried so far, so we can write to file.
        # Since all classifiers run on the same feature space each iteration, we really only need one list,
        # however it is nice to have one for each classifier.
        # prog tracks how many classifications out of total we have done.
        prog += 1
        featuresDT = []
        featuresRFW = []

        for i in range(0, len(resDT[1]) - 1, 2):
            featuresDT.append(trainx.header[int(resDT[1][i:i + 2])])
            featuresRFW.append(trainx.header[int(resRFW[1][i:i + 2])])

        # Write the results to file, and also display the current bests and the ETA.
        f.write("%s\t%s\t%s\n" % (resDT[0], resRFW[0], ','.join(featuresDT)))

        # bookkeeping for time/ETA
        curtime = time()
        predtime = ((curtime - timeStart) * (totalseq - prog) / prog)
        predtime = str(datetime.timedelta(seconds=predtime)).split('.')[0]

        # print to stdout with current states per iteration
        print("| FEATURES: %100s | PROGRESS: %d/%d %03.02f%% \t | RESULTS: %02.02f/%02.02f(BEST), %02.02f/%02.02f(CURRENT)\t | ETA: %s" % ( ', '.join([header[x] for x in pattern]),
        prog, totalseq, prog / totalseq * 1e2, bestDT[0], bestRFW[0], resDT[0], resRFW[0], predtime ),
              end='\r\r')
    print(end='\r')

    # close file and finish
    f.close()
    print("Finished!")


def enumerate_over_features(ftest, ftrain, f=[2, 3, 4, 5, 6, 7, 8]):
    """
    This function takes in a file each for training and test data, and a list of features to target for classifiers
    (decision tree, random forest walk, naive bayes, logistic regression, k nearest neighbor) and runs them using k many
    features as specified in the list f.

    :param ftest: name of file with test data.
    :param ftrain: name of file with training data.
    :param f: list of features to use.
    :return: None, writes to disk and also prints to stdout.
    """
    for e in f:
        print("Running %s" % e)
        runall(ftest, ftrain, 'results' + str(e), e)


###### Call function over here to run the file
enumerate_over_features('test.csv', 'train.csv', list(range(6, 9)))
