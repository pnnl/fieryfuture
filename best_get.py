import numpy as np
import argparse
from sklearn import metrics
import glob

def get_scores(res):
    labels = res[:, 1]
    pred = res[:, 3]
    plabels = (pred > .5).astype(int)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(labels, plabels, average='binary')
    auc = metrics.roc_auc_score(labels, pred)
    accuracy = metrics.accuracy_score(labels, plabels)
    return precision, recall, fscore, auc, accuracy


parser = argparse.ArgumentParser()
parser.add_argument('folder')
parser.add_argument('outfile')
args = parser.parse_args()

with open(args.outfile, 'w') as of:
    for f in glob.glob(args.folder + '/*/predictions.npy'):
        res = np.load(f)
        res = get_scores(res)
        res = [str(k) for k in res]
        of.write(f + ',' + ','.join(res) + '\n')