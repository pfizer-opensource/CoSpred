#!/bin/bash -e
DL_ENV_HOME=/hpc/grid/wip_adw_compsci/workspace/projects/deep_learning/Python

for PYTHON_VERSION in 2 3
do
    OLD_PATH=$PATH
    PATH=${DL_ENV_HOME}/conda/miniconda${PYTHON_VERSION}/bin:$PATH
    echo "Path is:"
    echo $PATH
    echo " "
    which conda
    ENV_DIR=${DL_ENV_HOME}/py${PYTHON_VERSION}gpu
    ENV_FILE=py${PYTHON_VERSION}gpu_environment.yml
    if [ -d "$ENV_DIR" ]; then rm -Rf $ENV_DIR; fi
    echo "................................................................................"
    echo "conda env create -p ${ENV_DIR} -f ${ENV_FILE}"
    conda env create -p ${ENV_DIR} -f ${ENV_FILE}
    echo "................................................................................"
    echo "Done creating conda${PYTHON_VERSION} dl environment"
    PATH=$OLD_PATH
done

