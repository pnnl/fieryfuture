# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:03:26 2018

@author: Kyle Larson

Desc: Convert a numpy array file to PNG image.
"""

import scipy.misc as scm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('infile', type=str,
                    help='path to matrix for mapping to png')
parser.add_argument('outfile', type=str,
                    help='name for output png file')

args = parser.parse_args()

# input numpy filee
npy_file = args.infile

# output raster
out_img = args.outfile

# load numpy file and convert to uint8
npf = np.load(args.infile)
# npf_cast = npf.astype('uint8')

# save array as image
scm.imsave(args.outfile, npf)