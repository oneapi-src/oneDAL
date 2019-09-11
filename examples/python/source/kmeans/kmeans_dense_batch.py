# file: kmeans_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-KMEANS_DENSE_BATCH"></a>
## \example kmeans_dense_batch.py

import os
import sys

import daal.algorithms.kmeans.init
from daal.algorithms import kmeans
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
datasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'kmeans_dense.csv')

# K-Means algorithm parameters
nClusters = 20
nIterations = 5

if __name__ == "__main__":

    # Initialize FileDataSource to retrieve the input data from a .csv file
    dataSource = FileDataSource(
        datasetFileName,
        DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from the input file
    dataSource.loadDataBlock()

    # Get initial clusters for the K-Means algorithm
    initAlg = kmeans.init.Batch(nClusters, method=kmeans.init.randomDense)

    initAlg.input.set(kmeans.init.data, dataSource.getNumericTable())

    res = initAlg.compute()
    centroidsResult = res.get(kmeans.init.centroids)

    # Create an algorithm object for the K-Means algorithm
    algorithm = kmeans.Batch(nClusters, nIterations, method=kmeans.lloydDense)

    algorithm.input.set(kmeans.data, dataSource.getNumericTable())
    algorithm.input.set(kmeans.inputCentroids, centroidsResult)

    res = algorithm.compute()

    # Print the clusterization results
    printNumericTable(res.get(kmeans.assignments), "First 10 cluster assignments:", 10)
    printNumericTable(res.get(kmeans.centroids), "First 10 dimensions of centroids:", 20, 10)
    printNumericTable(res.get(kmeans.objectiveFunction), "Objective function value:")
