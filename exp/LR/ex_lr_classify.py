import os
import random
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

for partition in ['d4', 'd3', 'd2', 'd1']:
    os.system('mkdir %s' % partition)
    for i in range(200):
        os.system('mkdir %s/%s' % (partition, i))
        mb = random.choice([16, 32, 64, 128])
        lr = random.uniform(0.001, 0.03)
        me = 50
        dr = random.uniform(0.0, 1.0)
        ds = random.randint(20, 200)
        l2 = random.uniform(0.0, 0.1)
        rs = random.randint(5, 500)
        reduce = random.randint(10, 400)
        do_reduce = random.choice([0, 1])
        reduce_val = ['', '-reduce %s ' % reduce][do_reduce]
        os.system('python ../../classify_lr.py ' +
                  '-learnrate %s ' % lr +
                  '-mb %s ' % mb +
                  '-max_epochs %s ' % me +
                  '-l2 %s ' % l2 +
                  '%s' % reduce_val +
                  '-decay_rate %s ' % dr +
                  '-decay_steps %s ' % ds +
                  '-partition %s ' % partition +
                  '-logfile %s/experiment_log.txt ' % partition +
                  '-random_seed %s ' % rs +
                  '-folder %s/%s' % (partition, i))
