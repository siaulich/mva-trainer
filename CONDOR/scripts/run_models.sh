#!/bin/bash
# HTCondor submission file for hyperparameter grid search
executable = submitCondor.sh
universe = vanilla

# Resource requirements
RequestGPUs    = 1
RequestMemory  = 60000
+RequestRuntime = 60000

# Output files
output = logs/job_$(Cluster)_$(ModelName).out
error = logs/job_$(Cluster)_$(ModelName).err
log = logs/job_$(Cluster)_$(ModelName).log

arguments = python3 ../TrainScript.py \
    --output_dir /afs/desy.de/user/a/aulich/mva-trainer/models/$(ModelName)/ \
    --load_config train_regression/load_config.yaml \
    --train_config train_regression/train_config.yaml \
    --model_config models/$(ModelName).yaml \
    --event_numbers even 

# Queue from file - reads each line and assigns to variables
queue ModelName from submit_params/model_names.txt