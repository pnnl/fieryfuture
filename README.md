# Deep Learning Classification of Cheatgrass Invasion

This repo is python 2 with an older version of Tensorflow. Details can be found in fiery.yml

## Setup
$ conda env create -f fiery.yml

## Steps to reproduce experiments performed in the publication
1. Run all four experiment scripts contained in exp folder
2. Move experimental output, exp/{DNN, LR, RF, RNN}/{d1, d2, d3, d4}/experiment_log.txt,
   for each model and data partition to results_and_analysis/allruns folder, renaming the logfiles to:
   ex. d1dnn.csv
3. Run results.ipynb in order to determine the best performing runs from the experiments and view/analyze results.   

## Files
+ classify_dnn.py
    - Script for running a single DNN experiment
+ classify_lr.py
    - Script for running a single Logistic Regression experiment
+ classify_rnn.py
    - Script for running a single RNN experiment
+ classify_rf.py
    - Script for running a single Random Forest experiment
+ graph_training_utils.py
    - Helper functions for boilerplate tensorflow training code
+ tf_ops.py
    - Tensorflow neural network modules
+ util.py 
    - There is a Batcher object that gets imported for batch gradient descent
+ exp/
    - {DNN, LR, RF, RNN}/
        + ex_{dnn, rnn, lr, rf}_classify.py: For running all experiments on a single gpu sequentially 
+ mapping/
    - dnn_d4_models/: Best performing DNN models
    - map_slurm_runs/
        + create_slurm.py: Creates slurm dispatch scripts for making the final map. Need to manually create fold{0,1,2,3,4} folders and logs subfolders
    - array_to_img.py: Create displayable image from large numpy array
    - make_image.slurm: Slurm job script for creating map image
    - make_map_matrix.py: Create a matrix of values for mapping from output of prediction model.
+ results_and_analysis/
    - results.ipynb: Digests results from allruns/ to find best performing models and associated metrics
    - allruns/: CSV files recording cross validation, test performance, and hyper parameters of all model runs. 
    - predictions: Predictions for all best performing models for all data subsets. Numpy arrays with shape=(number_data_points, 4):
        + index 1: data point unique id
        + index 2: data point ground truth cheatgrass coverage estimate
        + index 3: model prediction for probability < 2% cheatgrass coverage
        + index 4: model prediction for probability >= 2% cheatgrass coverage
+ idxs: 
    - cross_val_idxs.npy: Indexes for 5-fold cross validation split of 80% of the field data
    - test_idxs.npy: Indexes for held out test data.
+ spatial/
    - kfold_consensus_gte3_masked *.tif | *.clr: GeoTiff (with colormap) depicting predicted cheatgrass distribution based on >=3 k-fold consensus
    - kfold_consensus_masked *.tif | *.clr: GeoTiff (with colormap) depicting predicted cheatgrass distribution for all k-folds
    
## Abstract
Cheatgrass (Bromus tectorum) invasion is driving an emerging cycle of increased fire frequency and irreversible loss of wildlife habitat in the western US. Yet, detailed spatial information about its occurrence is still lacking for much of its presumably invaded range. Here, we provide code and training data used to develop Deep Neural Network, Joint Recurrent Neural Network, Random Forest, and Logistic Regression models  presented in Larson and Tuor (2021; see *Associated Publication*). Resultant maps produced from the best performing model are also included.

## Associated Publication
Larson, K.B. and A.R. Tuor. 2021. Deep learning classification of cheatgrass invasion in the Western United States using biophysical and remote sensing data. *Remote Sensing* 13(7), 1246; [https://doi.org/10.3390/rs13071246](https://doi.org/10.3390/rs13071246).

## Notice
This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
<p align="center">PACIFIC NORTHWEST NATIONAL LABORATORY<br/>operated by<br/> BATTELLE<br/>for the<br/>UNITED STATES DEPARTMENT OF ENERGY<br/>under Contract DE-AC05-76RL01830</p>

## License
See [License.md](https://github.com/pnnl/fieryfuture/blob/master/LICENSE.md)
 
