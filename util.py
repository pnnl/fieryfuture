import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import itertools


def confusion(truth, guess):
    """

    :param truth:
    :param guess:
    :return:
    """
    cf = np.zeros((2,2), dtype=np.float32)
    for idx in range(truth.shape[0]):
        cf[truth[idx], np.argmax(guess[idx, :])] += 1.0
    return cf


def confusion_matrix(cm, classes,
                     title=None,
                     cmap=plt.cm.Blues,
                     figname='confusion.png',
                     diag_outline=True,
                     experiment_values=[],
                     verbose=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    :param cm: numpy 2d confusion matrix
    :param classes: List of class labels
    :param title: Title of plot
    :param cmap:  A color map for coloring the matrix according to value
    :param folder: Where to save matrix (include '/' at end)
    :param diag_outline: Whether to print red box around diagonal matrix entries.
    :param verbose: Whether to print values inside confusion matrix
    """

    np.set_printoptions(precision=2)
    fig, ax = plt.subplots(1)
    rects = []
    if diag_outline:

        for i in range(cm.shape[0]):
            rects.append(patches.Rectangle((i + -0.5,i + -0.5),1,1,
                                     linewidth=1,edgecolor='m',
                                     facecolor='none'))
        for i in range(cm.shape[0]):
            ax.add_patch(rects[i])
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title:
        plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=15)
    plt.yticks(tick_marks, classes)
    ax.set_xlim([-0.5,1.5])
    thresh = cm.max() / 2.

    if verbose:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, int(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(figname)
    plt.show()


class Batcher:
    """

    """
    def __init__(self, x, y):
        """

        :param data:
        """
        self.x = x
        self.y = y
        self.index = 0
        self.num = x.shape[0]
        self.epoch = 0
        self.perm = np.arange(self.num)

    def next_batch(self, batchsize):
        """

        :param batchsize:
        :return:
        """
        assert batchsize <= self.num, 'Too large batchsize!!!'
        if self.index + batchsize <= self.num:
            x = self.x[self.index:self.index + batchsize]
            y = self.y[self.index:self.index + batchsize]
            self.index += batchsize
            if self.index == self.num:
                np.random.shuffle(self.perm)
                self.x = self.x[self.perm]
                self.y = self.y[self.perm]
                self.index = 0
                self.epoch += 1
            return x, y
        else:
            np.random.shuffle(self.perm)
            self.x = self.x[self.perm]
            self.y = self.y[self.perm]
            self.index = 0
            x = self.x[self.index:self.index + batchsize]
            y = self.y[self.index:self.index + batchsize]
            self.epoch += 1
            self.index = batchsize
            return x, y

def print_results(probs, ids, filename):
    """

    :param probs:
    :param ids:
    :param filename:
    :return:
    """
    with open(filename, 'w') as outfile:
        outfile.write('OBJECTID,prob_low,prob_high\n')
        for idval, (problow, probhigh) in zip(ids.tolist(), probs.tolist()):
            outfile.write('%s,%s,%s\n' % (idval, problow, probhigh))


def print_results_reg(probs, ids, filename):
    """

    :param probs:
    :param ids:
    :param filename:
    :return:
    """
    with open(filename, 'w') as outfile:
        outfile.write('OBJECTID,cover_estimate\n')
        for idval, cover in zip(ids.tolist(), probs.tolist()):
            outfile.write('%s,%s\n' % (idval, cover))

