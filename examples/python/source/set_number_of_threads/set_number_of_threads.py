# file: set_number_of_threads.py
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
