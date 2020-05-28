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
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
    shift
done

export PATH=${CONDA_DIR}/bin:$PATH
conda create -y -n conf python=3.7
source activate base
conda activate conf
conda install -y -c intel mpich tbb-devel numpy pytest scikit-learn
conda remove -y daal4py --force
conda remove -y daal --force
export DAALROOT=${BUILD_DIR}/daal/latest
echo DAALROOT
echo $DAALROOT
echo ${BUILD_DIR}/daal/latest
ls -l ${BUILD_DIR}/daal/latest
conda install $HOME/miniconda/envs/CB/conda-bld/linux-64/daal4py*.tar.bz2
conda list

export DAALROOT=${BUILD_DIR}/daal/latest
echo DAALROOT
echo $DAALROOT
echo ${BUILD_DIR}/daal/latest
ls -l ${BUILD_DIR}/daal/latest
cd .ci/scripts/conformance-scripts/
python run_tests.py 0.22.1
cd ../../..
