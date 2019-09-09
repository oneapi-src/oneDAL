# file: enable_thread_pinning.py
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

## <a name="DAAL-EXAMPLE-PY-ENABLE_THREAD_PINNING"></a>
## \example enable_thread_pinning.py

import os
import sys

import daal.algorithms.kmeans as kmeans
import daal.algorithms.kmeans.init as init
from daal.data_management import FileDataSource, DataSourceIface
from daal.services import Environment

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable
# Input data set parameters
datasetFileName = os.path.join('..', 'data', 'batch', 'kmeans_dense.csv')

# K-Means algorithm parameters
nClusters = 20
nIterations = 5
nThreads = 2
nThreadsInit = None
nThreadsNew = None

if __name__ == "__main__":

    # Initialize FileDataSource to retrieve the input data from a .csv file
    dataSource = FileDataSource(
        datasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from the input file
    dataSource.loadDataBlock()

    # Get initial clusters for the K-Means algorithm
    initAlg = kmeans.init.Batch(nClusters)

    initAlg.input.set(kmeans.init.data, dataSource.getNumericTable())

    # Enables thread pinning for next algorithm runs
    Environment.getInstance().enableThreadPinning(True)

    res = initAlg.compute()

    # Disables thread pinning for next algorithm runs
    Environment.getInstance().enableThreadPinning(False)

    centroids = res.get(kmeans.init.centroids)

    # Create an algorithm object for the K-Means algorithm
    algorithm = kmeans.Batch(nClusters, nIterations)

    algorithm.input.set(kmeans.data, dataSource.getNumericTable())
    algorithm.input.set(kmeans.inputCentroids, centroids)

    # Run computations
    unused_result = algorithm.compute()

    printNumericTable(unused_result.get(kmeans.assignments), "First 10 cluster assignments:", 10);
    printNumericTable(unused_result.get(kmeans.centroids), "First 10 dimensions of centroids:", 20, 10);
    printNumericTable(unused_result.get(kmeans.objectiveFunction), "Objective function value:");
