#!/bin/bash
#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --alg-name)
        ALG_NAME="$2"
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
    shift
done
SKLEARN_PATH="$(pip show scikit-learn | grep Location | cut -d ' ' -f 2)/sklearn/"

case ${ALG_NAME} in
    "dbscan")
        cp ${SKLEARN_PATH}cluster/tests/test_dbscan.py test_dbscan.py
    ;;
    "elastic_net")
        cp ${SKLEARN_PATH}linear_model/tests/test_coordinate_descent.py test_elastic_net.py
    ;;
    "kmeans")
        cp ${SKLEARN_PATH}cluster/tests/test_k_means.py test_kmeans.py
    ;;
    "lin_reg")
        cp ${SKLEARN_PATH}linear_model/tests/test_base.py test_lin_reg.py
    ;;
    "log_reg")
        cp ${SKLEARN_PATH}linear_model/tests/test_logistic.py test_log_reg.py
    ;;
    "pca")
        cp ${SKLEARN_PATH}decomposition/tests/test_pca.py test_pca.py
    ;;
    "ridge_reg")
        cp ${SKLEARN_PATH}linear_model/tests/test_ridge.py test_ridge_reg.py
    ;;
    "svm")
        cp ${SKLEARN_PATH}svm/tests/test_svm.py test_svm.py
    ;;
    "svm_sparse")
        cp ${SKLEARN_PATH}svm/tests/test_sparse.py test_svm_sparse.py
    ;;
    "forest")
        cp ${SKLEARN_PATH}ensemble/tests/test_forest.py test_forest.py
    ;;
    "knn")
        cp ${SKLEARN_PATH}neighbors/tests/test_neighbors.py test_knn.py
    ;;
    *)
        echo "Unknown algorithm: ${ALG_NAME}"
        exit 1
    ;;
esac
