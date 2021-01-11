import tensorflow as tf
import numpy as np
from graph_training_utils import get_feed_dict
import argparse
import tables
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument('field_data', type=str,
                    help='Full path to field data')
parser.add_argument('full_study_data', type=str,
                    help='Path to hdf5 file containing one row of features for each data point in the full study region')
parser.add_argument('outfile', type=str,
                    help='Folder to store results')
parser.add_argument('model', type=str,
                    help='Base name for saved tensorflow model')
parser.add_argument('-debug', action='store_true',
                    help='Use this flag to print feed dictionary contents and dimensions.')
parser.add_argument('-res', type=str, default='250m',
                    help='Code for subset of data to use as input. Can be "30m" or "250m"')
args = parser.parse_args()


def get_scores(labels, pred):
    pred = pred[:, 1]
    plabels = (pred > .5).astype(int)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(labels, plabels, average='binary')
    auc = metrics.roc_auc_score(labels, pred)
    accuracy = metrics.accuracy_score(labels, plabels)
    return precision, recall, fscore, auc, accuracy


class OnePassBatcher:

    def __init__(self, mat, mb):
        self.mat = mat
        self.mb = mb
        self.idx = 0
        self.limit = mat.shape[0]

    def next_batch(self):
        if self.idx == self.limit:
            return None
        if self.idx + self.mb <= self.limit:
            retmat = self.mat[self.idx:self.idx+self.mb]
            self.idx += self.mb
        elif self.idx + self.mb > self.limit:
            retmat = self.mat[self.idx:]
            self.idx = self.limit
        return retmat


class Classifier:

    def __init__(self):
        """
        Restores pre-trained cheat-grass ground cover classifier for prediction.
        """
        saver = tf.train.import_meta_graph(args.model + '.meta')
        self.sess = tf.Session()
        saver.restore(self.sess, args.model)

    def predict(self, datadict):
        """

        :param datadict: Dictionary with keys matching ph_dict
        :return: numpy array (batchsize X 2)
        """
        placeholders = tf.get_collection('placeholders')
        ph_dict = {'x': placeholders[0],
                   'y': placeholders[1],
                   'soil': placeholders[2],
                   'eco': placeholders[3],
                   'cover': placeholders[4]}
        fd = get_feed_dict(datadict, ph_dict, train=0)
        probs = self.sess.run(tf.get_collection('eval')[0], feed_dict=fd)
        return probs

with open(args.field_data, 'r') as h:
    header = h.readline().strip().split(',')

all_data = np.loadtxt(args.field_data, skiprows=1, delimiter=',')
partitions = {'30m': range(6, 669),
              '250m': range(5, 57) + range(60, 74)}

ecoidx = header.index('us_eco_I3_sghr_buff50km_' + args.res)
covidx = header.index('lanfire_sghr_rcls1_' + args.res)
soilidx = header.index('soilmoisttemp_30m_mosaic_rcls_' + args.res)
ecoreg = all_data[:, ecoidx]
cov_type = all_data[:,  covidx]
full_y = all_data[:, header.index('brte_cov')]
full_label = (full_y > 2).astype(int)
full_ids = all_data[:, 0:1]
full_x = all_data[:, partitions[args.res]]

cl = Classifier()
mean, std = np.mean(full_x, axis=0), np.std(full_x, axis=0)
train_datadict = {'x': (full_x - mean)/std,
                  'y': full_label,
                  'soil': all_data[:, 2].astype(int),
                  'cover': all_data[:, 4].astype(int),
                  'eco': all_data[:, 5].astype(int)}
probs = cl.predict(train_datadict)
precision, recall, fscore, auc, accuracy = get_scores(full_label, probs)
print(precision, recall, fscore, auc, accuracy)

# ========================================================================================
# ============================= Evaluate on Full Study Area ==============================
# ========================================================================================
tb = tables.open_file(args.full_study_data, 'r')
full_study_data = OnePassBatcher(tb.root.all_features, 1000)
batch = full_study_data.next_batch()
full_study_probs = np.empty((0, 6), dtype=np.float32)

is30 = int(args.res == '30m')
while batch is not None:
    train_datadict = {'x': batch[:, (3+is30):-2], 'y': (batch[:, 1] >= 2.0).astype(int)}
    train_datadict['x'] = (train_datadict['x'] - mean) / std
    train_datadict.update({'soil': batch[:, 0].astype(int),
                           'cover': batch[:, (1+is30)].astype(int),
                           'eco': batch[:, (2+is30)].astype(int)})
    np_probs = cl.predict(train_datadict)
    lat_long_probs = np.concatenate([batch[:, (3+is30):(5+is30)],  batch[:, -2:], np_probs], axis=1)
    full_study_probs = np.concatenate([full_study_probs, lat_long_probs], axis=0)
    batch = full_study_data.next_batch()
    print(len(full_study_probs))

# # ========================================================================================
# # ============================= Save results and perform check ===========================
# # ========================================================================================
print(full_study_probs.shape)
np.save(args.outfile, full_study_probs)
