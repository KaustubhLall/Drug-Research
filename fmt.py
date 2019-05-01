# %% md
## Load dataset

# %%
import pandas as pd

# %%
data = pd.read_csv('results6parsed.csv')

# sort by best CA
avg_ca = data.sort_values('Average CA', ascending=False)
avg_rfw = data.sort_values('RFW CA', ascending=False)
avg_dt = data.sort_values('DT CA', ascending=False)

with open('CA Features.csv', 'w')as f:
    f.write('Avg')
    f.write(avg_ca[:5].to_csv())

with open('CA Features.csv', 'a')as f:
    f.write('Rfw')
    f.write(avg_rfw[:5].to_csv())

with open('CA Features.csv', 'a')as f:
    f.write('Dtree')
    f.write(avg_dt[:5].to_csv())

# %% md
Now, let
us
take
the
most
interesting
features
from the csv and write
a
parser
to
analyze
the
feature
importances.
# %%

s = '''
nof_SO3H, nof_PO4, posCharge/Volume, nof_posCharge, molPSA, molLogP
nof_OH, nof_NH2, nof_PO4, C_R0, nof_HBA, PSA/Area
nof_OH, nof_NH2, negCharge/Volume, C_sp3, PSA/Area, molLogS
nof_OH, nof_NH2, C_sp3, nof_HBA, PSA/Area, molLogS
nof_OH, nof_NH2, negCharge/Volume, C_sp3, PSA/Area, molLogS
nof_OH, nof_NH2, posCharge/Volume, C_R0, nof_HBA, PSA/Area
nof_OH, nof_NH2, nof_PO4, C_R0, nof_HBA, PSA/Area
negCharge/Volume, C_sp3, C_R0, nof_posCharge, nof_HBA, molLogP
nof_acetyl, nof_COOH, nof_PO4, posCharge/Volume, C_R2, molLogP
PSA/Area, nof_Rings, Complexity, nof_SO3H, nof_OH, nof_Chirals, C_R0
'''

features = [[x] for x in s.split('\n')]
print(features)

# %%
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from graphviz import Source
from sklearn import tree
from IPython.display import SVG

# %% md
## Project Settings

Specified
here
are
the
paths
for the data and the features to run over in the list of best features.
Each
entry in the
list is a
list
containing
one
single
string
of
the
features
to
try, comma seperated.In this way it is easy to write a script to
add
entries
to
try very easily.
# %%

##### set parameters
path_train_data = 'train.csv'
path_test_data = 'test.csv'
path_all_data = 'Dataset Correlated Removed.csv'

# set features here

best_features = features

best_features = [list(map(str.strip, x[0].split(','))) for x in best_features]

k = len(best_features)

# %% md
## Load Dataset

This
code
loads
dataset
into
the
variables
below and converts
the
labels
to
categorical
0, 1
pairs.
# %%
# load dataset
all_data = pd.DataFrame(pd.read_csv(path_all_data))
all_labels = all_data['SLC'].astype('category').cat.codes
# drop labels
all_data.drop('SLC', axis=1, inplace=True)

train_data = pd.DataFrame(pd.read_csv(path_train_data))
train_labels = train_data['SLC'].astype('category').cat.codes
# drop labels

train_data.drop('SLC', axis=1, inplace=True)

test_data = pd.DataFrame(pd.read_csv(path_test_data))
test_labels = test_data['SLC'].astype('category').cat.codes
# drop labels
test_data.drop('SLC', axis=1, inplace=True)

# %% md
## AUC and Classification Accuracy - Decision Tree

The
code
below
will
find
the
classification
accuracy
using
10 - fold
cross - validation
using
stratified
sampling
to
help


class imbalance.The AUC on the test split is also found.
# %%
# visualize decision tree for input features


''' HYPERPARAMS FOR DECISION TREE

 These parameters implement a rudimentary pruning algorithm, would ideally like to use AB pruning'''
enable_pruning = True
# maximum depth of dtree
max_depth = 5
# how many samples your need atleast, at a LEAF node
min_samples = 3

d_trees = []

# find CA - uses 10-fold cross validation
# with stratified sampling to help with class imbalance
# and simple average over subsets
dt_cas = []

for i in range(k):
    aucs = []
    # make fold
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for trx, tex in skf.split(all_data, all_labels):
        # strip data to required features
        subset_data = all_data.filter(best_features[i], axis=1)

        # find auc
        dtree = DecisionTreeClassifier(presort=True, max_depth=max_depth, min_samples_leaf=min_samples)
        dtree.fit(subset_data.iloc[trx, :], all_labels.iloc[trx])
        pred = dtree.predict(subset_data.iloc[tex, :])
        labels = all_labels.iloc[tex]

        acc = roc_auc_score(labels, pred)
        # record auc to average later
        aucs.append(acc)

    dt_cas.append(np.mean(aucs))

# find AUC
dt_aucs = []
for i in range(k):
    subset_test_data = test_data.filter(best_features[i], axis=1)
    subset_train_data = train_data.filter(best_features[i], axis=1)

    clf = DecisionTreeClassifier(presort=True, max_depth=max_depth, min_samples_leaf=min_samples)
    clf.fit(subset_train_data, train_labels)
    d_trees.append(clf)

    # make its predictions on test data
    pred = d_trees[i].predict(subset_test_data)

    # find auc scores
    auc = roc_auc_score(test_labels, pred)

    # record the scores
    dt_aucs.append(auc)

print('Decision Tree Results:')
print('\tAUC\tAcc\tFeatures')
for i, f in enumerate(zip(dt_aucs, dt_cas)):
    print('\t%05.3f\t%05.3f\t' % tuple(f) + ', '.join(best_features[i]))

# %% md
## AUC and Classification Accuracy - Random Forest Walk

The
code
below
will
find
the
classification
accuracy
using
10 - fold
cross - validation
using
stratified
sampling
to
help


class imbalance.The AUC on the test split is also found.
# %%
# visualize random forest features


rfws = []

# find CA - uses 10-fold cross validation
# with stratified sampling to help with class imbalance
# and simple average over subsets
rfw_cas = []

for i in range(k):
    aucs = []
    # make fold
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for trx, tex in skf.split(all_data, all_labels):
        # strip data to required features
        subset_data = all_data.filter(best_features[i], axis=1)

        # find auc
        rfwtree = RandomForestClassifier(n_estimators=100)
        rfwtree.fit(subset_data.iloc[trx, :], all_labels.iloc[trx])
        pred = rfwtree.predict(subset_data.iloc[tex, :])
        labels = all_labels.iloc[tex]

        acc = roc_auc_score(labels, pred)
        # record auc to average later
        aucs.append(acc)

    rfw_cas.append(np.mean(aucs))

# find AUC
rfw_aucs = []
for i in range(k):
    subset_test_data = test_data.filter(best_features[i], axis=1)
    subset_train_data = train_data.filter(best_features[i], axis=1)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(subset_train_data, train_labels)
    rfws.append(clf)

    # make its predictions on test data
    pred = rfws[i].predict(subset_test_data)

    # find auc scores
    auc = roc_auc_score(test_labels, pred)

    # record the scores
    rfw_aucs.append(auc)

print('Random Forest Results:')
print('\tAUC\tAcc\tFeatures')
for i, f in enumerate(zip(rfw_aucs, rfw_cas)):
    print('\t%05.3f\t%05.3f\t' % tuple(f) + ', '.join(best_features[i]))

# %% md
## Visualizing individual decision trees

The
tree in variable
`dtree` is visualized
by
the
cell
below.We
can
see
how
it is pruned, the
splitting
rule, etc.
# %%
i = 0

dtree = d_trees[8]
graph = Source(tree.export_graphviz(dtree, out_file=None, feature_names=best_features[i][:dtree.n_features_]))
SVG(graph.pipe(format='svg'))
graph = Source(tree.export_graphviz(dtree, out_file=None, feature_names=best_features[i][:dtree.n_features_]))
graph.format = 'png'
graph.render('dtree_render', view=True)
graph = Source(tree.export_graphviz(dtree, out_file=None, feature_names=best_features[i][:dtree.n_features_]))
png_bytes = graph.pipe(format='png')
with open('dtree_pipe.png', 'wb') as f:
    f.write(png_bytes)
Image(png_bytes)
# %% md
## Feature importance

The
feature
importances
are
compared
below
for decision trees and random forests.
Reported
below is code
to
visualize
all
decision
trees.This
requires
the
graphviz
package and has
some
bugs, which
will
be
reported.This
code
visualizes
all
decision
trees and finds
the
feature
importances
for all of them.
# %%
i = 0
# visualization
for dtree in d_trees:
    if i < k:
        print('Feature importances for tree and forest (resp.) %s/%s:' % (i + 1, k))
        for e in zip(dtree.feature_importances_, rfws[i].feature_importances_, best_features[i]):
            print('\t%6f\t%6f\t%s' % e)

        try:
            graph = Source(
                tree.export_graphviz(dtree, out_file=None, feature_names=best_features[i][:dtree.n_features_]))
            SVG(graph.pipe(format='svg'))
            graph = Source(
                tree.export_graphviz(dtree, out_file=None, feature_names=best_features[i][:dtree.n_features_]))
            graph.format = 'png'
            graph.render('dtree_render', view=True)
            graph = Source(
                tree.export_graphviz(dtree, out_file=None, feature_names=best_features[i][:dtree.n_features_]))
            png_bytes = graph.pipe(format='png')
            with open('dtree_pipe.png', 'wb') as f:
                f.write(png_bytes)
            Image(png_bytes)
        except:
            print('Something went wrong with rendering graph')
    else:
        print('Warning, code may be buggy')
    i += 1
# %%

# %%

# %%

# %%
