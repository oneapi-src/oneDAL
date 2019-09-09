# file: kmeans_csr_batch.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation
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

## <a name="DAAL-EXAMPLE-PY-KMEANS_CSR_BATCH"></a>
## \example kmeans_csr_batch.py

import os
import sys

import daal.algorithms.kmeans.init
from daal.algorithms import kmeans

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, createSparseTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
datasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'kmeans_csr.csv')

# K-Means algorithm parameters
nClusters = 20
nIterations = 5

if __name__ == "__main__":

    # Retrieve the data from the input file
    dataTable = createSparseTable(datasetFileName)

    # Get initial clusters for the K-Means algorithm
    init = kmeans.init.Batch(nClusters, method=kmeans.init.randomDense)

    init.input.set(kmeans.init.data, dataTable)
    res = init.compute()

    centroids = res.get(kmeans.init.centroids)

    # Create an algorithm object for the K-Means algorithm
    algorithm = kmeans.Batch(nClusters, nIterations, method=kmeans.lloydCSR)

    algorithm.input.set(kmeans.data,           dataTable)
    algorithm.input.set(kmeans.inputCentroids, centroids)

    res = algorithm.compute()

    # Print the clusterization results
    printNumericTable(res.get(kmeans.assignments), "First 10 cluster assignments:", 10)
    printNumericTable(res.get(kmeans.centroids), "First 10 dimensions of centroids:", 20, 10)
    printNumericTable(res.get(kmeans.objectiveFunction), "Objective function value:")
