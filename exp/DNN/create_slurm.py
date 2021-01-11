import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-hours', type=int, help='number of gpu hours to request for job', default=24)
parser.add_argument('-allocation', type=str, help='Allocation name for billing', default='deepmpc')
parser.add_argument('-part', type=str, help='datasubset', default='d4')
parser.add_argument('-env', type=str, help='Name of conda environment for running code.', default='fiery')

args = parser.parse_args()

template = '#!/bin/bash\n' +\
           '#SBATCH -A %s\n' % args.allocation +\
           '#SBATCH -t %s:00:00\n' % args.hours +\
           '#SBATCH --gres=gpu:1\n' +\
           '#SBATCH -p dlt\n' +\
           '#SBATCH -N 1\n' +\
           '#SBATCH -n 8\n' +\
           '#SBATCH -o %j.out\n' +\
           '#SBATCH -e %j.err\n' +\
           'source /etc/profile.d/modules.sh\n' +\
           'module purge\n' +\
           'module load cuda/9.2.148\n' +\
           'module load python/anaconda2\n' +\
           'ulimit\n' +\
           'source activate %s\n\n' % args.env

for start in range(0, 200, 25):
    cmd = 'python split_dnn_exp.py -partition %s -start %s ' % (args.part, start)
    with open('%s.slurm' % start, 'w') as cmdfile: # unique name for sbatch script
        cmdfile.write(template + cmd)
