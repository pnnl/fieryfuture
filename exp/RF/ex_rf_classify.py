import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import scipy.sparse as sps
from sklearn.cross_validation import StratifiedKFold
import random
import sklearn
import os

coverage, ecology, soilage = 15, 100, 10


def get_scores(labels, pred):
    pred = pred[:, 1]
    plabels = (pred > .5).astype(int)
    precision, recall, fscore, _ = sklearn.metrics.precision_recall_fscore_support(labels, plabels, average='binary')
    auc = sklearn.metrics.roc_auc_score(labels, pred)
    accuracy = sklearn.metrics.accuracy_score(labels, plabels)
    return precision, recall, fscore, auc, accuracy


def to_one_hot(X, dim=None):
    """
    :param X: Vector of indices
    :param dim: Dimension of indexing
    :return: A sparse csr_matrix of one hots.
    """

    if dim is None:
        dim = np.amax(X) + 1

    return sps.csr_matrix(([1.0] * X.shape[0], (range(X.shape[0]),
                                                X.astype(int))),
                          shape=(X.shape[0], dim)).toarray()

parser = argparse.ArgumentParser()
parser.add_argument('-partition', type=str, default='d4',
                    help='Code indicating subset of features to use for classification')
parser.add_argument('-outfile', type=str, default='random_forest.csv',
                    help='where to print results')
parser.add_argument('-start', type=int, default=0)
parser.add_argument('-nruns', type=int, default=10)
args = parser.parse_args()

with open('clean_jan_field_data.csv', 'r') as h:
    header = h.readline().strip().split(',')

all_data = np.loadtxt('clean_jan_field_data.csv', skiprows=1, delimiter=',')
soil_class = to_one_hot(all_data[:, 2], dim=10)
coverage_class = to_one_hot(all_data[:, 4], dim=15)
ecoregions = to_one_hot(all_data[:, 5], dim=100)
cov_type = all_data[:, 4]
full_y = all_data[:, header.index('brte_cov')]
full_label = (full_y > 2).astype(int)
train, test = np.load('idxs/cross_val_idxs.npy'), np.load('idxs/test_idxs.npy')
full_y, test_y = full_y[train], full_y[test]
full_label, test_label = full_label[train], full_label[test]
soil_class, test_soil_class = soil_class[train], soil_class[test]
coverage_class, test_coverage_class = coverage_class[train], coverage_class[test]
ecoregions, test_ecoregions = ecoregions[train], ecoregions[test]
cov_type, test_cov_type = cov_type[train], cov_type[test]
test_data = all_data[test]
all_data = all_data[train]
split_labels = all_data[:, 5]
partitions = {'d1': range(6, 56),  # non image variables
              'd2': range(6, 56) + [56] + range(330, 669),  # everything but landsat
              'd3': range(6, 56) + range(57, 330),  # everything but modus
              'd4': range(6, 669)}  # everything

for p in partitions:
    if not os.path.exists(p + args.outfile):
        with open(p + args.outfile, 'w') as f:
            f.write('run,score,stdscore,prec,recall,fscore,auc,accuracy,maxdepth,nestimators,maxfeatures,minsamplesleaf,minsamplessplit,bootstrap,testprc,testrec,testfsc,testauc,testacc\n')

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid


models = [RandomForestClassifier()]
names = ['random_forest']

classifiers = zip(models, names)
folds = StratifiedKFold(split_labels, n_folds=5)

# folds = [(np.load('idxs/train_%s.npy' % i), np.load('idxs/test_%s.npy' %i)) for i in range(5)]

for p in partitions:
    full_x = all_data[:, partitions[p]]
    test_x = test_data[:, partitions[p]]
    for clf, name in classifiers:
        best_score = 0
        for i in range(args.start, args.start+args.nruns):
            scores = []
            predictions = []
            labels = []
            ids = []
            test_scores, test_predictions, test_labels, test_ids = [], [], [], []
            rg = {'n_estimators': random.choice(n_estimators),
                  'max_features': random.choice(max_features),
                  'max_depth': random.choice(max_depth),
                  'min_samples_split': random.choice(min_samples_split),
                  'min_samples_leaf': random.choice(min_samples_leaf),
                  'bootstrap': random.choice(bootstrap)}
            clf = RandomForestClassifier(max_depth=rg['max_depth'],
                                         n_estimators=rg['n_estimators'],
                                         max_features=rg['max_features'],
                                         min_samples_leaf=rg['min_samples_leaf'],
                                         min_samples_split=rg['min_samples_split'],
                                         bootstrap=rg['bootstrap'])
            for i, (train, test) in enumerate(folds):
                mean, std = np.mean(full_x[train], axis=0), np.std(full_x[train], axis=0)
                clf.fit(np.concatenate([soil_class[train], coverage_class[train], ecoregions[train], (full_x[train] - mean)/std], axis=1), full_label[train])
                score = clf.score(np.concatenate([soil_class[test], coverage_class[test], ecoregions[test], (full_x[test] - mean) / std], axis=1), full_label[test])
                prediction = clf.predict_proba(np.concatenate([soil_class[test], coverage_class[test], ecoregions[test], (full_x[test] - mean)/std], axis=1))
                scores.append(score)
                predictions.append(prediction)
                print(prediction.shape)
                labels.append(full_label[test])
                ids.append(all_data[test, 0])
                print(score)

                test_score = clf.score(np.concatenate(
                    [test_soil_class, test_coverage_class, test_ecoregions, (test_x - mean) / std], axis=1),
                                  test_label)
                test_prediction = clf.predict_proba(np.concatenate([test_soil_class, test_coverage_class, test_ecoregions, (test_x - mean) / std], axis=1))
                test_scores.append(test_score)
                test_predictions.append(test_prediction)
                print(prediction.shape)
                test_labels.append(test_label)
                test_ids.append(test_data[:, 0])
            precision, recall, fscore, auc, accuracy = get_scores(np.concatenate(labels), np.concatenate(predictions, axis=0))
            test_precision, test_recall, test_fscore, test_auc, test_accuracy = get_scores(np.concatenate(test_labels), np.concatenate(test_predictions, axis=0))
            print([k.shape for k in labels])

            if accuracy > best_score:
                np.save(p + '_' + str(i) + '_' + 'rf_predictions.npy', np.concatenate([np.concatenate(ids).reshape([-1, 1]),
                                                                        np.concatenate(labels).reshape([-1, 1]),
                                                                        np.concatenate(predictions, axis=0)],
                                                                       axis=1))
                np.save(p + '_' + str(i) + '_' +  'rf_test_predictions.npy', np.concatenate([np.concatenate(test_ids).reshape([-1, 1]),
                                                                        np.concatenate(test_labels).reshape([-1, 1]),
                                                                        np.concatenate(test_predictions, axis=0)],
                                                                       axis=1))
                best_score = accuracy
            with open(p + args.outfile, 'a+') as of:
                of.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (i, np.mean(scores), np.std(scores)*2,
                                                                       precision, recall, fscore, auc, accuracy,
                                                                       rg['max_depth'],
                                                                       rg['n_estimators'],
                                                                       rg['max_features'],
                                                                       rg['min_samples_leaf'],
                                                                       rg['min_samples_split'],
                                                                       rg['bootstrap'],
                                                                       test_precision, test_recall, test_fscore, test_auc, test_accuracy))
            print np.mean(scores), np.std(scores)*2
            print(np.mean(test_scores), np.std(test_scores)*2)
