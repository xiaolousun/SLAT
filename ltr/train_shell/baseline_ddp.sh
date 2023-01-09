
#!/usr/bin/env bash

set -x

TRAIN_MOUDLE='SA_Tracker'
TRAIN_NAME='SA_Tracker_ddp'
NPROC_PER_NODE=3

# PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
# python -W ignore -m torch.distributed.launch --master_port 26500 --nproc_per_node ${NPROC_PER_NODE} lib/train/run_training.py \

python -W ignore -m torch.distributed.launch --master_port 26500 --nproc_per_node ${NPROC_PER_NODE} ltr/run_training_multigpu.py \
                                    --train_module ${TRAIN_MOUDLE} --train_name=${TRAIN_NAME}
