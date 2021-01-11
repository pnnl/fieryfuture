"""
Training progress for each cross-validation partition are recorded in:
    args.folder/{0,1,2,3,4}_performance.csv
Full predictions for field study for best cross-validation models are recorded in:
    args.folder/predictions.npy
    with columns: ID,True_label,Probability>0.2Coverage
Best performing Tensorflow models for each cross-validation partition are recorded in:
    args.folder/models/<epoch>_<partition>_<accuracy>*
    <epoch> is the training epoch number of best performance
    <partition> is the cross-validation partition
    <accuracy> is the accuracy of predictions for that model
Global model performance and parameters are saved as a line in the file:
    args.logfile

"""
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
import sklearn

import sys
import os
import argparse

from tf_ops import dnn, ident
from graph_training_utils import ModelRunner
from util import Batcher



parser = argparse.ArgumentParser()
parser.add_argument('-learnrate', type=float, default=0.001,
                    help='Step size for gradient descent.')
parser.add_argument('-mb', type=int, default=128,
                    help='The mini batch size for stochastic gradient descent.')
parser.add_argument('-max_epochs', type=int, default=50,
                    help='Maximum number of epochs to train on.')
parser.add_argument('-debug', action='store_true',
                    help='Use this flag to print feed dictionary contents and dimensions.')
parser.add_argument('-l2', type=float, default=0.0,
                    help='Weight on l2 loss.')
parser.add_argument('-reduce', type=int, default=None)
parser.add_argument('-decay_rate', type=float, default=1.0,
                    help='Decay rate for learnrate decay.')
parser.add_argument('-decay_steps', type=int, default=20,
                    help='Number of steps for learnrate decay.')
parser.add_argument('-partition', type=str, default='d4',
                    help='Code for subset of data to use as input. Can be d1-d4')
parser.add_argument('-folder', type=str, default='test/',
                    help='Folder for all experimental results')
parser.add_argument('-logfile', type=str, default='dnn2log',
                    help='File to print results and hyper-parameters for model.')
parser.add_argument('-random_seed', type=int, default=5,
                    help='Set the numpy and tensorflow random seed')
parser.add_argument('-kp', type=int, default=1)
args = parser.parse_args()

if not args.folder.endswith('/'):
    args.folder += '/'
os.system('mkdir ' + args.folder)
os.system('mkdir ' + args.folder + 'models/')


def get_scores(labels, pred):
    pred = pred[:, 1]
    plabels = (pred > .5).astype(int)
    precision, recall, fscore, _ = sklearn.metrics.precision_recall_fscore_support(labels, plabels, average='binary')
    auc = sklearn.metrics.roc_auc_score(labels, pred)
    accuracy = sklearn.metrics.accuracy_score(labels, plabels)
    return precision, recall, fscore, auc, accuracy

with open('clean_jan_field_data.csv', 'r') as h:
    header = h.readline().strip().split(',')

all_data = np.loadtxt('clean_jan_field_data.csv', skiprows=1, delimiter=',')

ecoreg = all_data[:, 5]
cov_type = all_data[:, 4]

full_y = all_data[:, header.index('brte_cov')]
full_label = (full_y > 2).astype(int)

partitions = {'d1': range(6, 56),  # non image variables
              'd2': range(6, 56) + [56] + range(330, 669),  # everything but landsat
              'd3': range(6, 56) + range(57, 330),  # everything but modus
              'd4': range(6, 669)}  # everything


full_x = all_data[:, partitions[args.partition]]
standard = StandardScaler()
full_x = standard.fit_transform(full_x)
if args.reduce is not None:
    pca = PCA(n_components=args.reduce, svd_solver='full', whiten=True)
    full_x = pca.fit_transform(full_x)
full_ids = all_data[:, 0]

train, test = np.load('idxs/cross_val_idxs.npy'), np.load('idxs/test_idxs.npy')
full_y, test_y = full_y[train], full_y[test]
full_x, test_x = full_x[train], full_x[test]
full_label, test_label = full_label[train], full_label[test]
full_ids, test_ids = full_ids[train], full_ids[test]
test_data, all_data = all_data[test], all_data[train]
split_labels = all_data[:,5] 

# Tensorflow graph
x = tf.placeholder(tf.float32, [None, full_x.shape[1]])
y = tf.placeholder(tf.int64, [None])
soil, cover, eco = tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int32, [None])
ph_dict = {'x': x, 'y': y, 'soil': soil, 'cover': cover, 'eco': eco}
for p in [x, y, soil, cover, eco]:
    tf.add_to_collection('placeholders', p)

soil_embeddings = tf.Variable(0.01 * tf.truncated_normal([8, 64], 0, 1, tf.float32))
cover_embeddings = tf.Variable(0.01 * tf.truncated_normal([15, 64], 0, 1, tf.float32))
eco_embeddings = tf.Variable(0.01 * tf.truncated_normal([100, 64], 0, 1, tf.float32))

x = tf.concat([tf.nn.embedding_lookup(soil_embeddings, soil),
               tf.nn.embedding_lookup(cover_embeddings, cover),
               tf.nn.embedding_lookup(eco_embeddings, eco),
               x], 1)

logits = dnn(x, layers=[2], act=ident, keep_prob=None, norm=None, name='lr')
l2_weight_loss = args.l2*tf.reduce_sum(tf.get_collection('lr_weights')[0]**2)/2.0
l2_bias_loss = args.l2*tf.reduce_sum(tf.get_collection('lr_bias')[0])/2.0
ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
probs = tf.nn.softmax(logits)
tf.add_to_collection('eval', probs)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), y), tf.float32))

model = ModelRunner(ce + l2_weight_loss + l2_bias_loss, ph_dict,
                    learnrate=args.learnrate,
                    debug=args.debug,
                    decay_rate=args.decay_rate,
                    decay_steps=args.decay_steps)

# set up training
scores, predictions, true_classes, point_ids = [], [], [], []
folds = StratifiedKFold(split_labels, n_folds=5)

# folds = [(np.load('idxs/train_%s.npy' % i), np.load('idxs/test_%s.npy' %i)) for i in range(5)]
for i, (train, test) in enumerate(folds):
    performance_file = open(args.folder + str(i) + '_performance.csv', 'w')
    performance_file.write('train_loss train_precision, train_recall, train_fscore, train_auc, train_accuracy '
                           'test_loss test_precision, test_recall, test_fscore, test_auc, test_accuracy\n')

    mean, std = 0.0, 1.0 #np.mean(full_x[train], axis=0), np.std(full_x[train], axis=0)
    data = Batcher(all_data[train], full_label[train])
    batch_x, batch_y = data.next_batch(args.mb)
    batch_num = 0
    current_loss = sys.float_info.max
    test_datadict = {'x': (full_x[test] - mean)/std,
                     'y': full_label[test]}
    full_train_datadict = {'x': (full_x[train] - mean)/std,
                           'y': full_label[train]}
    test_datadict.update({'soil': all_data[:, 2][test].astype(int),
                          'cover': all_data[:, 4][test].astype(int),
                          'eco': all_data[:, 5][test].astype(int)})
    full_train_datadict.update({'soil': all_data[:, 2][train].astype(int),
                                'cover': all_data[:, 4][train].astype(int),
                                'eco': all_data[:, 5][train].astype(int)})
    current_epoch = 0
    test_eval_tensors = [ce, probs, accuracy]
    best_acc = 0.0
    while current_epoch < args.max_epochs:
        train_datadict = {'x': batch_x[:, partitions[args.partition]], 'y': batch_y}
        train_datadict['x'] = standard.transform(train_datadict['x'])#(train_datadict['x'] - mean)/std
        if args.reduce is not None:
            train_datadict['x'] = pca.transform(train_datadict['x'])
        train_datadict.update({'soil': batch_x[:, 2].astype(int),
                               'cover': batch_x[:, 4].astype(int),
                               'eco': batch_x[:, 5].astype(int)})
        model.train_step(train_datadict, [], update=True)
        batch_x, batch_y = data.next_batch(args.mb)
        if data.epoch > current_epoch:
            current_epoch = data.epoch
            _, current_loss, np_probs, np_acc = model.train_step(full_train_datadict,
                                                                 test_eval_tensors,
                                                                 update=False)
            _, test_current_loss, test_np_probs, np_test_acc = model.train_step(test_datadict,
                                                                                test_eval_tensors,
                                                                                update=False)
            test_precision, test_recall, test_fscore, test_auc, test_accuracy = get_scores(test_datadict['y'],
                                                                                           test_np_probs)
            train_precision, train_recall, train_fscore, train_auc, train_accuracy = get_scores(full_train_datadict['y'],
                                                                                                np_probs)
            performance_file.write('%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' % (current_loss,
                                                                                                      train_precision,
                                                                                                      train_recall,
                                                                                                      train_fscore,
                                                                                                      train_auc,
                                                                                                      train_accuracy,
                                                                                                      test_current_loss,
                                                                                                      test_precision,
                                                                                                      test_recall,
                                                                                                      test_fscore,
                                                                                                      test_auc,
                                                                                                      test_accuracy))

            assert np.isclose(test_accuracy, np_test_acc, 0.00001, 0.00001), '%s %s' % (test_accuracy,
                                                                                        np_test_acc)
            print('epoch: %s\tloss: %.5f\ttloss: %.5f acc: %.5f \ttest_acc: %.5f' % (data.epoch,
                                                                                     current_loss,
                                                                                     test_current_loss,
                                                                                     np_acc,
                                                                                     np_test_acc))

            if np_test_acc > best_acc:
                os.system('rm ' + args.folder + 'models/%sfold*' % i)
                thebestfile = model.saver.save(model.sess,
                                 args.folder + 'models/%sfold' % (i))
                best_predictions = test_np_probs
            best_acc = max(np_test_acc, best_acc)
    performance_file.close()
    predictions.append(best_predictions)
    true_classes.append(test_datadict['y'])
    point_ids.append(all_data[:, 0][test])
    scores.append(best_acc)
    model.sess.run(model.init)  # re-initialize all weights to random values

final_precision, final_recall, final_fscore, final_auc, final_accuracy = get_scores(np.concatenate(true_classes),
                                                                                    np.concatenate(predictions))
np.save(args.folder + 'predictions.npy',
        np.vstack([np.concatenate(point_ids),
                   np.concatenate(true_classes),
                   np.concatenate(predictions)[:, 0],
                   np.concatenate(predictions)[:, 1]]).transpose())

test_predictions, test_true_classes, test_point_ids = [], [], []
for fold, (train, test) in enumerate(folds):
    mean, std = np.mean(full_x[train], axis=0), np.std(full_x[train], axis=0)
    test_datadict = {'x': (test_x - mean) / std, 'y': test_label}
    test_datadict.update({'soil': test_data[:, 2].astype(int),
                          'cover': test_data[:, 4].astype(int),
                          'eco': test_data[:, 5].astype(int)})
    model.saver.restore(model.sess, args.folder + 'models/%sfold' % fold)
    _, test_current_loss, test_np_probs, np_test_acc = model.train_step(test_datadict,
                                                                        test_eval_tensors,
                                                                        update=False)
    test_predictions.append(test_np_probs)
    test_true_classes.append(test_datadict['y'])
    test_point_ids.append(test_data[:, 0])

test_precision, test_recall, test_fscore, test_auc, test_accuracy = get_scores(np.concatenate(test_true_classes),
                                                                                    np.concatenate(test_predictions))
np.save(args.folder + 'test_predictions.npy',
        np.vstack([np.concatenate(test_point_ids),
                   np.concatenate(test_true_classes),
                   np.concatenate(test_predictions)[:, 0],
                   np.concatenate(test_predictions)[:, 1]]).transpose())

print(scores)
with open(args.logfile, 'a') as logfile:
    logfile.write('%.5f,%s,%s,%s,%s,%s,%s,%s,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' % (args.learnrate,
                                                                                   args.reduce,
                                                                                    args.mb,
                                                                                    args.max_epochs,
                                                                                    args.l2,
                                                                                    args.random_seed,
                                                                                    args.decay_rate,
                                                                                    args.decay_steps,
                                                                                    final_precision,
                                                                                    final_recall,
                                                                                    final_fscore,
                                                                                    final_auc,
                                                                                    final_accuracy,
                                                                                      test_precision,
                                                                                      test_recall,
                                                                                      test_fscore,
                                                                                      test_auc,
                                                                                      test_accuracy,
                                                                                    np.mean(scores),
                                                                                    np.std(scores) * 2))
print np.mean(scores), np.std(scores)*2

