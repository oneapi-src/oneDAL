#!/bin/bash

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --build-dir)
        BUILD_DIR="$2"
        ;;
        --conda-dir)
        CONDA_DIR="$2"
        ;;
        --oneapi-dir)
        ONEAPI_DIR="$2"
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
    shift
done

# set enviroment
source ${CONDA_DIR}/etc/profile.d/conda.sh
export PATH=${CONDA_DIR}/bin:$PATH
conda create -y -n conf python=3.7
source activate conf
conda install -y -c intel mpich tbb-devel numpy pytest scikit-learn pandas
conda remove -y daal4py --force
conda remove -y daal --force
conda install $HOME/miniconda/envs/CB/conda-bld/linux-64/daal4py*.tar.bz2
conda list
source ${ONEAPI_DIR}/compiler/latest/env/vars.sh
source ${BUILD_DIR}/daal/latest/env/vars.sh intel64

# testing
cd .ci/scripts/conformance-scripts/
python run_tests.py 0.22.1
cd ../../..
