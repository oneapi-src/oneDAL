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

# testing
cd .ci/scripts/conformance-scripts/ || exit 1
export IDP_SKLEARN_VERBOSE=INFO
python run_tests.py ${PYTHON_VERSION} || exit 1
cd ../../..
