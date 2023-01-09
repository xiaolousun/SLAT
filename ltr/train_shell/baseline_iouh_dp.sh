
#!/usr/bin/env bash

set -x
BASE_ROOT_PATH='/data01/xlsun/code/TransT-M/'
TRAIN_MOUDLE='SA_Tracker'
TRAIN_NAME='SA_Tracker_iouh_dp'


# PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \

# save the config
cp -r ${BASE_ROOT_PATH}/ltr/train_settings/${TRAIN_MOUDLE}/${TRAIN_NAME}.py ${BASE_ROOT_PATH}/results/${TRAIN_MOUDLE}

python -u -W ignore ltr/run_training.py --train_module ${TRAIN_MOUDLE} --train_name=${TRAIN_NAME}


