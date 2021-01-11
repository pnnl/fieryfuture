#! /usr/bin/env python

import sys
from string import split, strip
import matplotlib as mpl
import argparse
import numpy as np
from sklearn import metrics

# mpl.use("Agg")
import matplotlib.pyplot as plt

with open('performance/d4/0_performance.csv', 'r') as hline:
    header = hline.readline().strip().split(',')
print(header)
# d4 = np.loadtxt('d4_performance.csv', delimiter=',', skiprows=1)
# d3 = np.loadtxt('d3_performance.csv', delimiter=',', skiprows=1)
# d2 = np.loadtxt('d2_performance.csv', delimiter=',', skiprows=1)
# d1 = np.loadtxt('d1_performance.csv', delimiter=',', skiprows=1)

colors = ['m', 'green', 'blue', 'red']
labels = ['Niche Variables', 'MODIS+Niche', 'LandSat+Niche', 'MODIS+LandSat+Niche']
xvals = range(d4.shape[0])
# for idx, metric in enumerate(header):
#     if idx > 5 or idx == 0:
#         fig, ax = plt.subplots(1)
#         ax.set_ylim(0, 1)
#
#         yvals = d1[:, idx], d2[:, idx], d3[:, idx], d4[:, idx]
#         d1line, d2line, d3line, d4line = [ax.plot(xvals, y, color=c, label=l) for y, c, l in zip(yvals, colors, labels)]
#         plt.xlabel('Epoch')
#         plt.ylabel(metric)
#         plt.legend()
#         ax.set_autoscaley_on(False)
#
#         ax.set_ylim(0, 1)
#
#         plt.savefig('pics/' + metric + '_variables.png')
#

d4mean = np.mean(np.dstack([np.loadtxt('performance/d4/%s_performance.csv' % k, delimiter=',', skiprows=1) for k in range(5)]), axis=2)
d3mean = np.mean(np.dstack([np.loadtxt('performance/d3/%s_performance.csv' % k, delimiter=',', skiprows=1) for k in range(5)]), axis=2)
d2mean = np.mean(np.dstack([np.loadtxt('performance/d2/%s_performance.csv' % k, delimiter=',', skiprows=1) for k in range(5)]), axis=2)
d1mean = np.mean(np.dstack([np.loadtxt('performance/d1/%s_performance.csv' % k, delimiter=',', skiprows=1) for k in range(5)]), axis=2)

d4std = np.std(np.dstack([np.loadtxt('performance/d4/%s_performance.csv' % k, delimiter=',', skiprows=1) for k in range(5)]), axis=2)
d3std = np.std(np.dstack([np.loadtxt('performance/d3/%s_performance.csv' % k, delimiter=',', skiprows=1) for k in range(5)]), axis=2)
d2std = np.std(np.dstack([np.loadtxt('performance/d2/%s_performance.csv' % k, delimiter=',', skiprows=1) for k in range(5)]), axis=2)
d1std = np.std(np.dstack([np.loadtxt('performance/d1/%s_performance.csv' % k, delimiter=',', skiprows=1) for k in range(5)]), axis=2)
#
# for idx, metric in enumerate(header):
#     if idx > 5 or idx == 0:
#
#         fig, ax = plt.subplots(1)
#         ax.set_autoscaley_on(False)
#         ax.set_ylim(0, 1)
#         yvals = d1mean[:, idx], d2mean[:, idx], d3mean[:, idx], d4mean[:, idx]
#         yerr = d1std[:, idx], d2std[:, idx], d3std[:, idx], d4std[:, idx]
#         d1line, d2line, d3line, d4line = [ax.plot(xvals, y, color=c, label=l) for y, c, l in zip(yvals, colors, labels)]
#         [ax.fill_between(xvals, y - e, y + e, alpha=0.3, color=c) for y, e, c in zip(yvals, yerr, colors)]
#         ax.set_autoscaley_on(False)
#         ax.set_ylim(0, 1)
#         plt.xlabel('Epoch')
#         plt.ylabel(metric)
#         # plt.ylim(0, 1)
#         plt.legend()
#         plt.savefig('pics/' + metric + '_mean_variables.png')
        # plt.show()

parts = ['d1', 'd2', 'd3', 'd4']
colors = colors = ['m', 'green']
labels = ['Train', 'Validation']

for idx, metric in enumerate(header[0:1]):
    f, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    # for mean, std, part in zip([d1mean, d2mean, d3mean, d4mean], [d1std, d2std, d3std, d4std], parts):

    ax[0, 0].set_autoscaley_on(False)
    ax[0, 0].set_ylim(0, 1.1)
    ax[0, 0].text(1, 0.1, '$D_1$', fontsize=12)
    ax[0, 0].set_ylabel('Cross Entropy')
    ax[0, 0].yaxis.set_label_coords(-0.175, 0.0)
    yvals = d1mean[:, idx], d1mean[:, idx+6]
    yerr = d1std[:, idx], d1std[:, idx+6]
    trainline, testline = [ax[0, 0].plot(xvals, y, color=c, label=l) for y, c, l in zip(yvals, colors, labels)]
    [ax[0, 0].fill_between(xvals, y - e, y + e, alpha=0.3, color=c) for y, e, c in zip(yvals, yerr, colors)]

    ax[0, 1].set_autoscaley_on(False)
    ax[0, 1].set_ylim(0, 1.1)
    ax[0, 1].text(1, 0.1, '$D_2$', fontsize=12)

    yvals = d2mean[:, idx], d2mean[:, idx + 6]
    yerr = d2std[:, idx], d2std[:, idx + 6]
    trainline, testline = [ax[0, 1].plot(xvals, y, color=c, label=l) for y, c, l in zip(yvals, colors, labels)]
    [ax[0, 1].fill_between(xvals, y - e, y + e, alpha=0.3, color=c) for y, e, c in zip(yvals, yerr, colors)]

    ax[1, 0].set_autoscaley_on(False)
    ax[1, 0].set_ylim(0, 1.1)
    ax[1, 0].text(1, 0.1, '$D_3$', fontsize=12)

    yvals = d3mean[:, idx], d3mean[:, idx + 6]
    yerr = d3std[:, idx], d3std[:, idx + 6]
    trainline, testline = [ax[1, 0].plot(xvals, y, color=c, label=l) for y, c, l in zip(yvals, colors, labels)]
    [ax[1, 0].fill_between(xvals, y - e, y + e, alpha=0.3, color=c) for y, e, c in zip(yvals, yerr, colors)]
    # plt.xlabel('Epoch')
    # plt.ylabel(metric)

    ax[1, 1].set_autoscaley_on(False)
    ax[1, 1].set_ylim(0, 1.1)
    ax[1, 1].text(1, 0.1, '$D_4$', fontsize=12)

    yvals = d4mean[:, idx], d4mean[:, idx + 6]
    yerr = d4std[:, idx], d4std[:, idx + 6]
    trainline, testline = [ax[1, 1].plot(xvals, y, color=c, label=l) for y, c, l in zip(yvals, colors, labels)]
    [ax[1, 1].fill_between(xvals, y - e, y + e, alpha=0.3, color=c) for y, e, c in zip(yvals, yerr, colors)]

    f.subplots_adjust(hspace=0, wspace=0)
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].xaxis.set_label_coords(1.0, -0.175)

    plt.legend()
    # plt.tight_layout()
    plt.savefig('pics/all_part_ce_mean_train_test.png')

# for idx, metric in enumerate(header[:6]):
#     for mean, std, part in zip([d1mean, d2mean, d3mean, d4mean], [d1std, d2std, d3std, d4std], parts):
#
#         fig, ax = plt.subplots(1)
#         ax.set_autoscaley_on(False)
#         ax.set_ylim(0, 1)
#         yvals = mean[:, idx], mean[:, idx+6]
#         yerr = std[:, idx], std[:, idx+6]
#         trainline, testline = [ax.plot(xvals, y, color=c, label=l) for y, c, l in zip(yvals, colors, labels)]
#         [ax.fill_between(xvals, y - e, y + e, alpha=0.3, color=c) for y, e, c in zip(yvals, yerr, colors)]
#         plt.xlabel('Epoch')
#         plt.ylabel(metric)
#
#         plt.legend()
#         plt.savefig('pics/%s_%s_mean_train_test.png' % (part, metric))
#
# for idx, metric in enumerate(header[:6]):
#     for mat, part in zip([d1, d2, d3, d4], parts):
#         fig, ax = plt.subplots(1)
#         ax.set_autoscaley_on(False)
#         ax.set_ylim(0, 1)
#         yvals = mat[:, idx], mat[:,idx+6]
#         trainline, testline = [ax.plot(xvals, y, color=c, label=l) for y, c, l in zip(yvals, colors, labels)]
#         plt.xlabel('Epoch')
#         plt.ylabel(metric)
#         plt.legend()
#
#         ax.set_ylim(0, 1)
#
#         plt.savefig('pics/' + metric + '_best_variables.png')
#
# d4pred = np.load('d4/67/predictions.npy')
# d3pred = np.load('d3/25/predictions.npy')
# d2pred = np.load('d2/19/predictions.npy')
# d1pred = np.load('d1/122/predictions.npy')
# fig, ax = plt.subplots(1)
# true = [d[:, 1] for d in [d1pred, d2pred, d3pred, d4pred]]
# pred = [d[:, 3] for d in [d1pred, d2pred, d3pred, d4pred]]
# rocs = [metrics.roc_curve(t, p) for t, p in zip(true, pred)]
# aucs = [metrics.roc_auc_score(t, p) for t, p in zip(true, pred)]
#
# colors = ['m', 'green', 'blue', 'red']
# labels = ['Niche Variables', 'MODIS+Niche', 'LandSat+Niche', 'MODIS+LandSat+Niche']
# roc1, roc2, roc3, roc4 = [ax.plot(r[0], r[1], color=c, label=l + ' (AUC %.4f)' % a) for a, r, c, l in zip(aucs, rocs, colors, labels)]
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
#
# plt.savefig('roc.png')
