import os
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-allocation', type=str, help='Allocation name for billing', default='deepmpc')
#
# args = parser.parse_args()

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
        cmd = 'python /people/tuor369/gitland/fiery/code/models/map_classify_dnn.py ' +\
              '/pic/projects/deepscience/data/fieryfuture/fullextent30/clean_feb_field_data.csv ' +\
              '/pic/projects/deepscience/data/fieryfuture/fullextent30/withnan_subset_hdf5/%s_fullextent_30_marianas.hdf5 ' % piece +\
              '%s_dnn_30m_results.npy ' % piece +\
              '/people/tuor369/gitland/fire/mapping/dnn_d4_models/%sfold ' % fold +\
              '-res 30m'
        # cmd = 'python split_dnn_exp.py -partition %s -start %s ' % (args.part, start)
        with open('fold%s/%s_dnn_30m.slurm' % (fold, piece), 'w') as cmdfile: # unique name for sbatch script
            cmdfile.write(template + cmd)
