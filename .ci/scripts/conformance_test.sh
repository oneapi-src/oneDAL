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
PYTHON_VERSION="3.7"
source ${CONDA_DIR}/etc/profile.d/conda.sh
export PATH=${CONDA_DIR}/bin:$PATH
conda create -y -n conf python=${PYTHON_VERSION}
source activate conf
conda install -y -c intel mpich numpy pytest pandas
conda install -y -c conda-forge scikit-learn
conda install $HOME/miniconda/envs/CB/conda-bld/linux-64/daal4py*.tar.bz2
conda list
compiler_vars=${ONEAPI_DIR}/compiler/latest/env/vars.sh
if [ ! -e "${compiler_vars}" ]
then
    echo "Can't find compiler vars ${compiler_vars}"
    exit 1
fi
source "${compiler_vars}"
dal_vars=${BUILD_DIR}/daal/latest/env/vars.sh
if [ ! -e "${dal_vars}" ]
then
    echo "Can't find oneDAL vars ${dal_vars}"
    exit 1
fi
source ${dal_vars} intel64
export TBBROOT=${BUILD_DIR}/tbb/latest/lib/intel64
export LD_LIBRARY_PATH=${BUILD_DIR}/tbb/latest/lib/intel64:$LD_LIBRARY_PATH

# testing
cd .ci/scripts/conformance-scripts/ || exit 1
export IDP_SKLEARN_VERBOSE=INFO
python run_tests.py ${PYTHON_VERSION} || exit 1
cd ../../..
