import sys
import argparse
import os

import tensorflow as tf
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import sklearn

from tf_ops import dnn, batch_normalize
from graph_training_utils import ModelRunner
from util import Batcher



parser = argparse.ArgumentParser()
parser.add_argument('-learnrate', type=float, default=0.0001,
                        help='Step size for gradient descent.')
parser.add_argument('-mb', type=int, default=16,
                    help='The mini batch size for stochastic gradient descent.')
parser.add_argument('-max_epochs', type=int, default=2,
                    help='Maximum number of epochs to train on.')
parser.add_argument('-debug', action='store_true',
                    help='Use this flag to print feed dictionary contents and dimensions.')
parser.add_argument('-kp', type=float, default=.9,
                    help='Keep probability for dropout.')
parser.add_argument('-layers', type=int, nargs='+', default=[256, 128, 128],
                    help='Keep probability for dropout.')
parser.add_argument('-rnn_layers', type=int, nargs='+', default=[256],
                    help='Keep probability for dropout.')
parser.add_argument('-decay_rate', type=float, default=1.0,
                    help='Decay rate for learnrate decay.')
parser.add_argument('-decay_steps', type=int, default=100,
                    help='Number of steps for learnrate decay.')
parser.add_argument('-partition', type=str, default='d4',
                    help='Code for subset of data to use as input. Can be d2-d4')
parser.add_argument('-logfile', type=str, default='dnn2log',
                    help='File to print results and hyper-parameters for model.')
parser.add_argument('-folder', type=str, default='test/',
                    help='Folder for all experimental results')
parser.add_argument('-random_seed', type=int, default=5,
                    help='Set the numpy and tensorflow random seed')
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


def rnn(sequence_input):
    with tf.variable_scope('forward'):
        fw_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units) for num_units in args.rnn_layers]
        fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cells, state_is_tuple=True)
    with tf.variable_scope('backward'):
        bw_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units) for num_units in args.rnn_layers]
        bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cells, state_is_tuple=True)

    hidden_states, fw_cell_state, bw_cell_state = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, sequence_input,
                                                                                 dtype=tf.float32)
    final_hidden = tf.concat((fw_cell_state[-1].h, bw_cell_state[-1].h), 1)
    mean_hidden = tf.reduce_mean(tf.stack(hidden_states, axis=0), axis=0)
    return tf.concat([mean_hidden, final_hidden], 1)


with open('clean_jan_field_data.csv', 'r') as h:
    header = h.readline().strip().split(',')

all_data = np.loadtxt('clean_jan_field_data.csv', skiprows=1, delimiter=',')

ecoreg = all_data[:, 5]
cov_type = all_data[:, 4]
split_labels = ecoreg  # This gives a unique int representation to each of 79 combinations
                                      # of cov_type and ecoreg found in data; used for representative distribution in
                                      # train/test cross-validation folds
full_y = all_data[:, header.index('brte_cov')]
full_label = (full_y > 2).astype(int)

non_sequence = range(6, 59) + range(666, 669)
landsat_range = range(60, 330)
modus_range = range(330, 666)

full_x = all_data[:, non_sequence]
full_ids = all_data[:, 0]

train, test = np.load('idxs/cross_val_idxs.npy'), np.load('idxs/test_idxs.npy')
full_y, test_y = full_y[train], full_y[test]
full_x, test_x = full_x[train], full_x[test]
full_label, test_label = full_label[train], full_label[test]
full_ids, test_ids = full_ids[train], full_ids[test]
split_labels = split_labels[train]
test_data, all_data = all_data[test], all_data[train]

# Tensorflow graph
x = tf.placeholder(tf.float32, [None, full_x.shape[1]])
y = tf.placeholder(tf.int64, [None])
ph_dict = {'x': x, 'y': y}
landsat_input = [tf.placeholder(tf.float32, [None, 27]) for i in range(10)]
modus_input = [tf.placeholder(tf.float32, [None, 21]) for i in range(16)]
ph_dict['landsat_x'] = landsat_input
ph_dict['modus_x'] = modus_input


soil = tf.placeholder(tf.int32, [None])
cover = tf.placeholder(tf.int32, [None])
eco = tf.placeholder(tf.int32, [None])

soil_embeddings = tf.Variable(0.01 * tf.truncated_normal([8, 64], 0, 1, tf.float32))
cover_embeddings = tf.Variable(0.01 * tf.truncated_normal([15, 64], 0, 1, tf.float32))
eco_embeddings = tf.Variable(0.01 * tf.truncated_normal([100, 64], 0, 1, tf.float32))

x = tf.concat([tf.nn.embedding_lookup(soil_embeddings, soil),
               tf.nn.embedding_lookup(cover_embeddings, cover),
               tf.nn.embedding_lookup(eco_embeddings, eco),
               x], 1)
ph_dict.update({'soil': soil, 'cover': cover, 'eco': eco})
for p in [x, y, soil, cover, eco]:
    tf.add_to_collection('placeholders', p)
with tf.variable_scope('modus') as scope:
    modus_uber_ndvi = rnn(modus_input)
with tf.variable_scope('landsat') as scope:
    landsat_uber_ndvi = rnn(landsat_input)

if args.partition == 'd4':
    h = dnn(tf.concat([x, modus_uber_ndvi, landsat_uber_ndvi], 1),
            layers=args.layers, keep_prob=args.kp, norm=batch_normalize)
elif args.partition == 'd2':
    h = dnn(tf.concat([x, modus_uber_ndvi], 1),
            layers=args.layers, keep_prob=args.kp, norm=batch_normalize)
elif args.partition == 'd3':
    h = dnn(tf.concat([x, landsat_uber_ndvi], 1),
            layers=args.layers, keep_prob=args.kp, norm=batch_normalize)

w = tf.Variable(0.01 * tf.truncated_normal([h.get_shape().as_list()[1], 2], 0, 1, tf.float32))
b = tf.Variable(tf.zeros([2]))
logits = tf.matmul(h, w) + b
ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
probs = tf.nn.softmax(logits)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), y), tf.float32))

model = ModelRunner(ce, ph_dict, learnrate=args.learnrate, debug=args.debug,
                    decay_rate=args.decay_rate, decay_steps=args.decay_steps)

#set up training
scores, predictions, true_classes, point_ids = [], [], [], []
folds = StratifiedKFold(split_labels, n_folds=5)
# folds = [(np.load('idxs/train_%s.npy' % i), np.load('idxs/test_%s.npy' %i)) for i in range(5)]
for i, (train, test) in enumerate(folds):


    performance_file = open(args.folder + str(i) + '_performance.csv', 'w')
    performance_file.write('train_loss train_precision, train_recall, train_fscore, train_auc, train_accuracy '
                           'test_loss test_precision, test_recall, test_fscore, test_auc, test_accuracy\n')

    mean, std = np.mean(full_x[train], axis=0), np.std(full_x[train], axis=0)
    landsat_seq_mean, landsat_seq_std = np.mean(all_data[:, landsat_range]), np.std(all_data[:, landsat_range])
    modus_seq_mean, modus_seq_std = np.mean(all_data[:, modus_range]), np.std(all_data[:, modus_range])

    all_landsat_seq = (all_data[:, landsat_range] - landsat_seq_mean) / landsat_seq_std
    all_landsat_seq = np.reshape(all_landsat_seq, [-1, 3, 9, 10])
    all_modus_seq = (all_data[:, modus_range] - modus_seq_mean) / modus_seq_std
    all_modus_seq = np.reshape(all_modus_seq, [-1, 3, 7, 16])

    test_datadict = {'x': (full_x[test] - mean)/std, 'y': full_label[test]}
    landsat_test_seq = np.split(all_landsat_seq[test], 10, axis=3)
    test_datadict['landsat_x'] = [np.reshape(s, [-1, 27]) for s in landsat_test_seq]
    modus_test_seq = np.split(all_modus_seq[test], 16, axis=3)
    test_datadict['modus_x'] = [np.reshape(s, [-1, 21]) for s in modus_test_seq]

    full_train_datadict = {'x': (full_x[train] - mean)/std, 'y': full_label[train]}
    landsat_train_seq = np.split(all_landsat_seq[train], 10, axis=3)
    full_train_datadict['landsat_x'] = [np.reshape(s, [-1, 27]) for s in landsat_train_seq]
    modus_train_seq = np.split(all_modus_seq[train], 16, axis=3)
    full_train_datadict['modus_x'] = [np.reshape(s, [-1, 21]) for s in modus_train_seq]

    data = Batcher(all_data[train], full_label[train])
    batch_x, batch_y = data.next_batch(args.mb)
    batch_num = 0
    current_loss = sys.float_info.max
    test_datadict.update({'soil': all_data[:, 2][test].astype(int),
                          'cover': all_data[:, 4][test].astype(int),
                          'eco': all_data[:, 5][test].astype(int)})
    full_train_datadict.update({'soil': all_data[:, 2][train].astype(int),
                                'cover': all_data[:, 4][train].astype(int),
                                'eco': all_data[:, 5][train].astype(int)})
    current_epoch = 0
    test_eval_tensors = [ce, probs, accuracy]
    best_acc = 0.0
    while current_epoch < args.max_epochs:  # mat is not None and self.badcount < self.badlimit and loss != inf, nan:
        train_datadict = {'x': batch_x[:, non_sequence], 'y': batch_y}  # (batch_x[:, 4] > 2).astype(int)}
        train_datadict['x'] = (train_datadict['x'] - mean)/std
        ls_seq = (batch_x[:, landsat_range] - landsat_seq_mean)/landsat_seq_std
        ls_seq = np.reshape(ls_seq, [-1, 3, 9, 10])
        ls_seq = np.split(ls_seq, 10, axis=3)
        train_datadict['landsat_x'] = [np.reshape(s, [-1, 27]) for s in ls_seq]

        md_seq = (batch_x[:, modus_range] - modus_seq_mean) / modus_seq_std
        md_seq = np.reshape(md_seq, [-1, 3, 7, 16])
        md_seq = np.split(md_seq, 16, axis=3)
        train_datadict['modus_x'] = [np.reshape(s, [-1, 21]) for s in md_seq]
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
            train_precision, train_recall, train_fscore, train_auc, train_accuracy = get_scores(
                full_train_datadict['y'],
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
                os.system('rm %smodels/%s/fold*' % (args.folder, i))
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
    landsat_seq_mean, landsat_seq_std = np.mean(all_data[:, landsat_range]), np.std(all_data[:, landsat_range])
    modus_seq_mean, modus_seq_std = np.mean(all_data[:, modus_range]), np.std(all_data[:, modus_range])
    landsat_test_seq = (test_data[:, landsat_range] - landsat_seq_mean) / landsat_seq_std
    landsat_test_seq = np.reshape(landsat_test_seq, [-1, 3, 9, 10])
    modus_test_seq = (test_data[:, modus_range] - modus_seq_mean) / modus_seq_std
    modus_test_seq = np.reshape(modus_test_seq, [-1, 3, 7, 16])

    test_datadict = {'x': (test_x - mean) / std, 'y': test_label}
    landsat_test_seq = np.split(landsat_test_seq, 10, axis=3)
    test_datadict['landsat_x'] = [np.reshape(s, [-1, 27]) for s in landsat_test_seq]
    modus_test_seq = np.split(modus_test_seq, 16, axis=3)
    test_datadict['modus_x'] = [np.reshape(s, [-1, 21]) for s in modus_test_seq]
    current_loss = sys.float_info.max
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
    logfile.write('%.5f,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' % (args.learnrate,
                                                                                            args.mb,
                                                                                            args.max_epochs,
                                                                                            args.kp,
                                                                                            args.random_seed,
                                                                                            args.layers[0],
                                                                                            args.layers[1],
                                                                                            args.layers[2],
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

