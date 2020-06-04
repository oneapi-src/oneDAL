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
        --daal4py-dir)
        DAAL4PY_DIR="$2"
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
source activate conf
conda list
source ${ONEAPI_DIR}/compiler/latest/env/vars.sh
source ${BUILD_DIR}/daal/latest/env/vars.sh intel64
cd ${DAAL4PY_DIR}

# testing
python -c "import daal4py"
mpirun -n 4 python -m unittest discover -v -s tests -p spmd*.py
mpiexec -localonly -n 4 python -m unittest discover -v -s tests -p spmd*.py
python -m unittest discover -v -s tests -p test*.py
python -m daal4py examples/sycl/sklearn_sycl.py
cd examples && python run_examples.py
