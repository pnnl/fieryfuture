
import matplotlib as mpl
import numpy as np
from sklearn import metrics

# mpl.use("Agg")
import matplotlib.pyplot as plt

d1pred = np.load('dnnd1pred.npy')
d2pred = np.load('dnnd2pred.npy')
d3pred = np.load('dnnd3pred.npy')
d4pred = np.load('dnnd4pred.npy')
fig, ax = plt.subplots(1)
true = [d[:, 1] for d in [d1pred, d2pred, d3pred, d4pred]]
pred = [d[:, 3] for d in [d1pred, d2pred, d3pred, d4pred]]
rocs = [metrics.roc_curve(t, p) for t, p in zip(true, pred)]
aucs = [metrics.roc_auc_score(t, p) for t, p in zip(true, pred)]

colors = ['m', 'green', 'blue', 'red']
labels = ['DNN $D_1$', 'DNN $D_2$', 'DNN $D_3$', 'DNN $D_4$']
roc1, roc2, roc3, roc4 = [ax.plot(r[0], r[1], color=c, label=l + ' (AUC %.4f)' % a) for a, r, c, l in zip(aucs, rocs, colors, labels)]
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
# plt.show()
plt.savefig('dnn_subsets_roc.png')

d1pred = np.load('lrd4pred.npy')
d2pred = np.load('rfd4pred.npy')
d3pred = np.load('rnnd4pred.npy')
d4pred = np.load('dnnd4pred.npy')
fig, ax = plt.subplots(1)
true = [d[:, 1] for d in [d1pred, d2pred, d3pred, d4pred]]
pred = [d[:, 3] for d in [d1pred, d2pred, d3pred, d4pred]]
rocs = [metrics.roc_curve(t, p) for t, p in zip(true, pred)]
aucs = [metrics.roc_auc_score(t, p) for t, p in zip(true, pred)]

colors = ['m', 'green', 'blue', 'red']
labels = ['LR $D_4$', 'RF $D_4$', 'RNN $D_4$', 'DNN $D_4$']
roc1, roc2, roc3, roc4 = [ax.plot(r[0], r[1], color=c, label=l + ' (AUC %.4f)' % a) for a, r, c, l in zip(aucs, rocs, colors, labels)]
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
# plt.show()
plt.savefig('models_d4_roc.png')
