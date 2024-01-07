#!/bin/bash -e
DL_ENV_HOME=/hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/Python

for PYTHON_VERSION in 3
do
    ENV_DIR=$DL_ENV_HOME/conda/miniconda${PYTHON_VERSION}
    if [ -d "$ENV_DIR" ]; then rm -Rf $ENV_DIR; fi
    wget https://repo.continuum.io/miniconda/Miniconda${PYTHON_VERSION}-latest-Linux-x86_64.sh -O $DL_ENV_HOME/miniconda${PYTHON_VERSION}.sh
    echo "Done downloading miniconda installer"
    bash $DL_ENV_HOME/miniconda${PYTHON_VERSION}.sh -b -p $ENV_DIR
done
