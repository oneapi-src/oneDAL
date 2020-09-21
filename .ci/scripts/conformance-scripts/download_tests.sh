#!/bin/bash
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --alg-name)
        ALG_NAME="$2"
        ;;
        --python-version)
        PYTHON_VERSION="$2"
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
    shift
done
SKLEARN_PATH="$CONDA_PREFIX/lib/python${PYTHON_VERSION}/site-packages/sklearn/"

case ${ALG_NAME} in
    "dbscan")
        wget -O test_dbscan.py ${SKLEARN_URL_ROOT}cluster/tests/test_dbscan.py
    ;;
    "elastic_net")
        wget -O test_elastic_net.py ${SKLEARN_URL_ROOT}linear_model/tests/test_coordinate_descent.py
    ;;
    "kmeans")
        wget -O test_kmeans.py ${SKLEARN_URL_ROOT}cluster/tests/test_k_means.py
    ;;
    "lin_reg")
        wget -O test_lin_reg.py ${SKLEARN_URL_ROOT}linear_model/tests/test_base.py
    ;;
    "log_reg")
        wget -O test_log_reg.py ${SKLEARN_URL_ROOT}linear_model/tests/test_logistic.py
    ;;
    "pca")
        wget -O test_pca.py ${SKLEARN_URL_ROOT}decomposition/tests/test_pca.py
    ;;
    "ridge_reg")
        wget -O test_ridge_reg.py ${SKLEARN_URL_ROOT}linear_model/tests/test_ridge.py
    ;;
    "svm")
        wget -O test_svm.py ${SKLEARN_URL_ROOT}svm/tests/test_svm.py
    ;;
    "svm_sparse")
        wget -O test_svm_sparse.py ${SKLEARN_URL_ROOT}svm/tests/test_sparse.py
    ;;
    "forest")
        wget -O test_forest.py ${SKLEARN_URL_ROOT}ensemble/tests/test_forest.py
    ;;
    "knn")
        wget -O test_knn.py ${SKLEARN_URL_ROOT}neighbors/tests/test_neighbors.py
    ;;
    *)
        echo "Unknown algorithm: ${ALG_NAME}"
        exit 1
    ;;
esac
