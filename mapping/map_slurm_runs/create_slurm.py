import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-allocation', type=str, help='Allocation name for billing')
parser.add_argument('-mapping_code_path', type=str, help='Allocation name for billing')
parser.add_argument('-data_path', type=str, help='Allocation name for billing')
args = parser.parse_args()

template = '#!/bin/bash\n' +\
           '#SBATCH -A deepmpc\n' +\
           '#SBATCH -t 96:00:00\n'  +\
           '#SBATCH --gres=gpu:1\n' +\
           '#SBATCH -p shared_dlt\n' +\
           '#SBATCH -N 1\n' +\
           '#SBATCH -n 8\n' +\
           '#SBATCH -o logs/%j.out\n' +\
           '#SBATCH -e logs/%j.err\n' +\
           'source /etc/profile.d/modules.sh\n' +\
           'module purge\n' +\
           'module load cuda/9.2.148\n' +\
           'module load python/anaconda2\n' +\
           'ulimit\n' +\
           'source activate fiery\n\n'

for fold in range(5):
    for piece in range(25):
        cmd = 'python %s/map_classify_dnn.py ' % args.mapping_code_path +\
              '%s/clean_jan_field_data.csv ' % args.data_path +\
              '%s/%s_fullextent_30_marianas.hdf5 ' % (args.data_path, piece) +\
              '%s_dnn_30m_results.npy ' % piece +\
              '%s/%sfold ' % (args.data_path, fold) +\
              '-res 30m'
        # cmd = 'python split_dnn_exp.py -partition %s -start %s ' % (args.part, start)
        with open('fold%s/%s_dnn_30m.slurm' % (fold, piece), 'w') as cmdfile: # unique name for sbatch script
            cmdfile.write(template + cmd)
