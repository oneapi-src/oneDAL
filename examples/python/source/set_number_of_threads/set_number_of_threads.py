# file: set_number_of_threads.py
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

## <a name="DAAL-EXAMPLE-PY-SET_NUMBER_OF_THREADS"></a>
## \example set_number_of_threads.py

import os

import daal.algorithms.kmeans as kmeans
import daal.algorithms.kmeans.init as init
from daal.data_management import FileDataSource, DataSourceIface
from daal.services import Environment

# Input data set parameters
datasetFileName = os.path.join('..', 'data', 'batch', 'kmeans_dense.csv')

# K-Means algorithm parameters
nClusters = 20
nIterations = 5
nThreads = 2
nThreadsInit = None
nThreadsNew = None

if __name__ == "__main__":

    # Get the number of threads that is used by the library by default
    nThreadsInit = Environment.getInstance().getNumberOfThreads()

    # Set the maximum number of threads to be used by the library
    Environment.getInstance().setNumberOfThreads(nThreads)

    # Get the number of threads that is used by the library after changing
    nThreadsNew = Environment.getInstance().getNumberOfThreads()

    # Initialize FileDataSource to retrieve the input data from a .csv file
    dataSource = FileDataSource(
        datasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from the input file
    dataSource.loadDataBlock()

    # Get initial clusters for the K-Means algorithm
    initAlg = init.Batch(nClusters)

    initAlg.input.set(init.data, dataSource.getNumericTable())
    res = initAlg.compute()
    centroids = res.get(init.centroids)

    # Create an algorithm object for the K-Means algorithm
    algorithm = kmeans.Batch(nClusters, nIterations)

    algorithm.input.set(kmeans.data, dataSource.getNumericTable())
    algorithm.input.set(kmeans.inputCentroids, centroids)

    # Run computations
    unused_result = algorithm.compute()

    print("Initial number of threads:        {}".format(nThreadsInit))
    print("Number of threads to set:         {}".format(nThreads))
    print("Number of threads after setting:  {}".format(nThreadsNew))
