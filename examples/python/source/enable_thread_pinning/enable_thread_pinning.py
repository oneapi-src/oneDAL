# file: enable_thread_pinning.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
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
