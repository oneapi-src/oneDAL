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

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda create -y -n conformance python=3.7
source activate conformance
conda install -y -c intel scikit-learn
conda remove -y daal4py --force
conda install -y -c conda-forge mpich tbb-devel cython jinja2 numpy clang-tools
export TBBROOT=$CONDA_PREFIX
export MPIROOT=$CONDA_PREFIX
source ${BUILD_DIR}/daal/bin/daalvars.sh intel64
git clone https://github.com/IntelPython/daal4py
cd daal4py
python setup.py develop
cd ..
conda install -y -c conda-forge pytest
pwd
cd .ci/conformance/
pwd
python run_tests.py
cd ../..
pwd