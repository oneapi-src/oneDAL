# file: kmeans_init_dense_batch.py
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

#
# !  Content:
# !    Python example of dense K-Means clustering with different initialization methods
# !    in the batch processing mode
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-KMEANS_INIT_DENSE_BATCH"></a>
## \example kmeans_init_dense_batch.py
#

import os
import numpy as np
from daal.algorithms import kmeans
import daal.algorithms.kmeans.init
from daal.data_management import HomogenNumericTable, FileDataSource, DataSource, BlockDescriptor, readOnly

DAAL_PREFIX = os.path.join('..', 'data')
# Input data set
datasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'kmeans_init_dense.csv')

# K-Means algorithm parameters
nMaxIterations = 1000
cAccuracyThreshold = 0.01
nClusters = 20

def getSingleValue(pTbl, ntype):
    block = BlockDescriptor(ntype=ntype)
    pTbl.getBlockOfRows(0, 1, readOnly, block)
    value = block.getArray().flatten()[0]
    pTbl.releaseBlockOfRows(block)
    return value


def runKmeans(inputData, nClusters, method, methodName, oversamplingFactor = -1.0):
    # Get initial clusters for the K-Means algorithm
    init = kmeans.init.Batch(nClusters, fptype=np.float32, method=method)
    init.input.set(kmeans.init.data, inputData)
    if oversamplingFactor > 0:
        init.parameter.oversamplingFactor = oversamplingFactor
    if method == kmeans.init.parallelPlusDense:
        print("K-means init parameters: method = " + methodName + ", oversamplingFactor = "
              + str(init.parameter.oversamplingFactor) + ", nRounds = " + str(init.parameter.nRounds))
    else:
        print("K-means init parameters: method = " + methodName)

    centroids = init.compute().get(kmeans.init.centroids)

    # Create an algorithm object for the K-Means algorithm
    algorithm = kmeans.Batch(nClusters, nMaxIterations)

    algorithm.input.set(kmeans.data, inputData)
    algorithm.input.set(kmeans.inputCentroids, centroids)
    algorithm.parameter.accuracyThreshold = cAccuracyThreshold
    print("K-means algorithm parameters: maxIterations = " + str(algorithm.parameter.maxIterations)
          + ", accuracyThreshold = " + str(algorithm.parameter.accuracyThreshold))
    res = algorithm.compute()

    # Print the results
    goalFunc = getSingleValue(res.get(kmeans.objectiveFunction), ntype=np.float32)
    nIterations = getSingleValue(res.get(kmeans.nIterations), ntype=np.intc)
    print("K-means algorithm results: Objective function value = " + str(goalFunc*1e-6)
          + "*1E+6, number of iterations = " + str(nIterations) + "\n")


if __name__ == "__main__":
    # Initialize FileDataSource to retrieve the input data from a .csv file
    inputData = HomogenNumericTable(ntype=np.float32)
    dataSource = FileDataSource(datasetFileName,
                                DataSource.notAllocateNumericTable,
                                DataSource.doDictionaryFromContext)

    # Retrieve the data from the input file
    dataSource.loadDataBlock(inputData)

    runKmeans(inputData, nClusters, kmeans.init.deterministicDense, "deterministicDense")
    runKmeans(inputData, nClusters, kmeans.init.randomDense, "randomDense")
    runKmeans(inputData, nClusters, kmeans.init.plusPlusDense, "plusPlusDense")
    runKmeans(inputData, nClusters, kmeans.init.parallelPlusDense, "parallelPlusDense", 0.5)
    runKmeans(inputData, nClusters, kmeans.init.parallelPlusDense, "parallelPlusDense", 2.0)
