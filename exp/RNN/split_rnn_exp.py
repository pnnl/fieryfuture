import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-partition')
parser.add_argument('-nruns', type=int, default=25)
parser.add_argument('-start', type=int, default=0)
args = parser.parse_args()

for partition in [args.partition]:
    os.system('mkdir %s' % partition)
    for i in range(args.start, args.start+args.nruns):
        os.system('mkdir %s/%s' % (partition, i))
        mb = random.choice([16, 32, 64, 128])
        lr = random.uniform(0.0, 0.03)
        me = 100
        layers = (random.choice([128, 256, 512]),
                  random.choice([32, 64, 128, 256]),
                  random.choice([64, 128, 256]))
        dr = random.uniform(0.5, 1.0)
        ds = random.randint(20, 200)
        kp = random.uniform(0.5, 1.0)
        rs = random.randint(5, 500)
        rnn_layers = random.choice([128, 256])
        os.system('python ../../classify_rnn.py ' +
                  '-learnrate %s ' % lr +
                  '-mb %s ' % mb +
                  '-max_epochs %s ' % me +
                  '-kp %s ' % kp +
                  '-layers %s %s %s ' % layers +
                  '-rnn_layers %s ' % rnn_layers +
                  '-decay_rate %s ' % dr +
                  '-decay_steps %s ' % ds +
                  '-partition %s ' % partition +
                  '-logfile %s/experiment_log.txt ' % partition +
                  '-random_seed %s ' % rs +
                  '-folder %s/%s' % (partition, i))
