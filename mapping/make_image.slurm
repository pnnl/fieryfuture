#!/bin/sh
#SBATCH -A deepscience
#SBATCH -t 4:00:00
##SBATCH --gres=gpu:1
#SBATCH -p all 
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -o ./logs/image.%j.out
#SBATCH -e ./logs/image.%j.err

module purge
module load python/anaconda2
ulimit 

python array_to_img.py test_big.npy test_big.png

