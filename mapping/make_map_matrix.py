import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('outfolder', type=str,
                    help='Folder to store results')
args = parser.parser_args()

for fold in range(5):
    print('fold%s' % fold)
    mats = [np.load(f) for f in glob.glob('map_slurm_runs/fold%s/*.npy' % fold)]
    dnn_output = np.concatenate(mats)
    matrix = np.zeros((55150, 54267), dtype=np.float32)
    rows = (dnn_output[:, 2]).astype(int)
    cols = (dnn_output[:, 3]).astype(int)
    vals = dnn_output[:, 5]
    matrix[rows, cols] = vals
    print('Save prob matrix')
    np.save(args.outfolder + 'prob/fold%s.npy' % fold, matrix)
    del matrix
    matrix = np.zeros((55150, 54267), dtype=np.float32)
    matrix[matrix > 0.5] = 1
    matrix[matrix <= 0.5] = 0
    print('Save binary matrix')
    np.save(args.outfolder + 'prob/fold%s.npy' % fold, matrix)

    del matrix
    matrix = np.zeros((55150, 54267), dtype=np.float32)
    matrix[:] = np.nan
    matrix[rows, cols] = vals
    print('Save nan matrix')
    np.save(args.outfolder + 'prob/fold%s.npy' % fold, matrix)
