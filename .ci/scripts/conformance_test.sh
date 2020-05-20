#!/bin/bash

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --build-dir)
        BUILD_DIR="$2"
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
    shift
done

conda create -y -n conformance python=3.7
source activate conformance
conda install -y -c conda-forge mpich tbb-devel numpy pytest scikit-learn
export DAALROOT=${BUILD_DIR}/daal/latest
conda install $HOME/miniconda/envs/CB/conda-bld/broken/*.tar.bz2

cd .ci/scripts/conformance-scripts/
python run_tests.py
cd ../../..


