#!/bin/bash
# HTCondor submission file for hyperparameter grid search
executable = submitCondor.sh
universe = vanilla

# Resource requirements
RequestGPUs    = 1
RequestMemory  = 60000
+RequestRuntime = 60000

# Output files
output = logs/job_$(Cluster)_$(EventNumbers).out
error = logs/job_$(Cluster)_$(EventNumbers).err
log = logs/job_$(Cluster)_$(EventNumbers).log

arguments = python3 ../TrainScript.py \
    --output_dir ../models/$(ModelName)/ \
    --load_config train_regression/load_config.yaml \
    --train_config train_regression/train_config.yaml \
    --model_config models/$(ModelName).yaml \
    --event_numbers $(EventNumbers) 

# Queue from file - reads each line and assigns to variables
queue EventNumbers from (
    even
    odd
)