# file: kmeans_init_dense_distr.py
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
# !    Python example of dense K-Means clustering in the distributed processing mode
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-KMEANS_INIT_DENSE_DISTRIBUTED"></a>
## \example kmeans_init_dense_distr.py
#

import sys, os
import numpy as np

from daal import step1Local, step2Master, step3Master
import daal.algorithms.kmeans as kmeans
import daal.algorithms.kmeans.init as init
from daal import step1Local, step2Master, step2Local, step3Master, step4Local, step5Master
from daal.data_management import FileDataSource, DataSource, RowMergedNumericTable

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, createSparseTable

# K-Means algorithm parameters
algorithmFPType = np.float32
nClusters = 20
nIterations = 5
nBlocks = 4
nVectorsInBlock = 2500

DAAL_PREFIX = os.path.join('..', 'data', 'distributed')
dataFileNames = [os.path.join(DAAL_PREFIX, 'kmeans_dense_1.csv'),
                 os.path.join(DAAL_PREFIX, 'kmeans_dense_2.csv'),
                 os.path.join(DAAL_PREFIX, 'kmeans_dense_3.csv'),
                 os.path.join(DAAL_PREFIX, 'kmeans_dense_4.csv')]


def loadData(files):
    data = []
    for i in range(nBlocks):
        # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
        dataSource = FileDataSource(files[i],
                                    DataSource.doAllocateNumericTable,
                                    DataSource.doDictionaryFromContext)
        # Retrieve the data from the input file
        dataSource.loadDataBlock()
        data.append(dataSource.getNumericTable())
    return data


def initStep1(data, method):
    for i in range(nBlocks):
        # Create an algorithm object for the K-Means algorithm
        local = kmeans.init.Distributed(step1Local, nClusters, nBlocks*nVectorsInBlock, i*nVectorsInBlock,
                                        fptype=algorithmFPType, method=method)
        local.input.set(kmeans.init.data, data[i])
        pNewCenters = local.compute().get(kmeans.init.partialCentroids)
        if pNewCenters:
            return pNewCenters
    return None


def initStep23(data, localNodeData, step2Input, step3, bFirstIteration, method):
#    kmeans.init.Distributed(nClusters, bFirstIteration, step=step3Master, fptype=algorithmFPType, method=method)
    for i in range(len(data)):
        step2 = kmeans.init.Distributed(step2Local, nClusters, bFirstIteration, fptype=algorithmFPType, method=method)
        step2.input.set(kmeans.init.data, data[i])
        step2.input.setStepInput(kmeans.init.inputOfStep2, step2Input)
        if not bFirstIteration:
            step2.input.setLocal(kmeans.init.internalInput, localNodeData[i])
        res = step2.compute()
        if bFirstIteration:
            localNodeData.append(res.getLocal(kmeans.init.internalResult))
        step3.input.add(kmeans.init.inputOfStep3FromStep2, i, res.getOutput(kmeans.init.outputOfStep2ForStep3))
    return step3.compute()


def initStep4(data, localNodeData, step3res, method):
    aRes = []
    for i in range(0, len(data)):
        # Get an input for step 4 on this node if any
        step3Output = step3res.getOutput(kmeans.init.outputOfStep3ForStep4, i)
        if not step3Output:
            continue

        # Create an algorithm object for the step 4
        step4 = kmeans.init.Distributed(step4Local, nClusters, fptype=algorithmFPType, method=method)
        # Set the input data to the algorithm
        step4.input.setInput(kmeans.init.data, data[i])
        step4.input.setLocal(kmeans.init.internalInput, localNodeData[i])
        step4.input.setStepInput(kmeans.init.inputOfStep4FromStep3, step3Output)
        # Compute and get the result
        step4.compute()
        aRes.append(step4.compute().get(kmeans.init.outputOfStep4))

    if len(aRes) == 0:
        return None
    if len(aRes) == 1:
        return aRes[0]
    # For parallelPlus algorithm
    pMerged = RowMergedNumericTable()
    for r in aRes:
        pMerged.addNumericTable(r)
    return pMerged
#    return NumericTable.cast(pMerged)


def initCentroids_plusPlusDense(data):
    # Internal data to be stored on the local nodes
    localNodeData = []
    # Numeric table to collect the results
    pCentroids = RowMergedNumericTable()
    # First step on the local nodes
    pNewCentroids = initStep1(data, kmeans.init.plusPlusDense)
    pCentroids.addNumericTable(pNewCentroids)

    # Create an algorithm object for the step 3
    step3 = kmeans.init.Distributed(step3Master, nClusters, fptype=algorithmFPType, method=kmeans.init.plusPlusDense)
    for iCenter in range(1, nClusters):
        # Perform steps 2 and 3
        step3res = initStep23(data, localNodeData, pNewCentroids, step3, iCenter == 1, method=kmeans.init.plusPlusDense)
        # Perform steps 4
        pNewCentroids = initStep4(data, localNodeData, step3res, method=kmeans.init.plusPlusDense)
        pCentroids.addNumericTable(pNewCentroids)
    return pCentroids  #NumericTable.cast(pCentroids)


def initCentroids_parallelPlusDense(data):
    # Internal data to be stored on the local nodes
    localNodeData = []
    # First step on the local nodes
    pNewCentroids =  initStep1(data, method=kmeans.init.parallelPlusDense)

    # Create an algorithm object for the step 5
    step5 = kmeans.init.Distributed(step5Master, nClusters, fptype=algorithmFPType, method=kmeans.init.parallelPlusDense)
    step5.input.add(kmeans.init.inputCentroids, pNewCentroids)
    # Create an algorithm object for the step 3
    step3 = kmeans.init.Distributed(step3Master, nClusters, fptype=algorithmFPType, method=kmeans.init.parallelPlusDense)
    for iRound in range(step5.parameter.nRounds):
        # Perform steps 2 and 3
        step3res = initStep23(data, localNodeData, pNewCentroids, step3, iRound == 0, method=kmeans.init.parallelPlusDense)
        # Perform step 4
        pNewCentroids = initStep4(data, localNodeData, step3res, method=kmeans.init.parallelPlusDense)
        step5.input.add(kmeans.init.inputCentroids, pNewCentroids)

    # One more step 2
    for i in range(nBlocks):
        # Create an algorithm object for the step 2
        local = kmeans.init.Distributed(step2Local, nClusters, False, fptype=algorithmFPType, method=kmeans.init.parallelPlusDense)
        local.parameter.outputForStep5Required = True
        # Set the input data to the algorithm
        local.input.setInput(kmeans.init.data, data[i])
        local.input.setLocal(kmeans.init.internalInput, localNodeData[i])
        local.input.setStepInput(kmeans.init.inputOfStep2, pNewCentroids)
        # Compute, get the result and add the result to the input of step 5
        step5.input.add(kmeans.init.inputOfStep5FromStep2, local.compute().getOutput(kmeans.init.outputOfStep2ForStep5))

    step5.input.setStepInput(kmeans.init.inputOfStep5FromStep3, step3res.getStepOutput(kmeans.init.outputOfStep3ForStep5))
    step5.compute()
    return step5.finalizeCompute().get(kmeans.init.centroids)


def initCentroids(data, method):
    if method == kmeans.init.parallelPlusDense:
        return initCentroids_parallelPlusDense(data)
    if method == kmeans.init.plusPlusDense:
        return initCentroids_plusPlusDense(data)
    assert False, "Unknown method for initCentroids"


def calculateCentroids(initialCentroids, data):
    masterAlgorithm = kmeans.Distributed(step2Master, nClusters, fptype=algorithmFPType)

    nRows = initialCentroids.getNumberOfRows()
    nCols = initialCentroids.getNumberOfColumns()

    assignments = []
    centroids = initialCentroids
    objectiveFunction = None

    # Calculate centroids
    for it in range(nIterations):
        for i in range(nBlocks):
            # Create an algorithm object for the K-Means algorithm
            localAlgorithm = kmeans.Distributed(step1Local, nClusters, False, fptype=algorithmFPType)

            # Set the input data to the algorithm
            localAlgorithm.input.set(kmeans.data, data[i])
            localAlgorithm.input.set(kmeans.inputCentroids, centroids)

            masterAlgorithm.input.add(kmeans.partialResults, localAlgorithm.compute())

        masterAlgorithm.compute()
        res = masterAlgorithm.finalizeCompute()

        centroids = res.get(kmeans.centroids)
        objectiveFunction = res.get(kmeans.objectiveFunction)

    # Calculate assignments
    for i in range(nBlocks):
        # Create an algorithm object for the K-Means algorithm
        localAlgorithm = kmeans.Batch(nClusters, 0, fptyep=algorithmFPType)

        # Set the input data to the algorithm
        localAlgorithm.input.set(kmeans.data, data[i])
        localAlgorithm.input.set(kmeans.inputCentroids, centroids)

        assignments.append(localAlgorithm.compute().get(kmeans.assignments))

    # Print the clusterization results
    printNumericTable(assignments[0], "First 10 cluster assignments from 1st node:", 10)
    printNumericTable(centroids, "First 10 dimensions of centroids:", 20, 10)
    printNumericTable(objectiveFunction, "Objective function value:")


def runKMeans(data, method, methodName):
    print("K-means init parameters: method = " + str(methodName))
    centroids = initCentroids(data, method=method)
    calculateCentroids(centroids, data)


if __name__ == "__main__":
    data = loadData(dataFileNames)
    runKMeans(data, kmeans.init.plusPlusDense, "plusPlusDense")
    runKMeans(data, kmeans.init.parallelPlusDense, "parallelPlusDense")
