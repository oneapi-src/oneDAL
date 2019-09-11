# file: pca_metrics_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-PCA_METRICS_DENSE_BATCH"></a>
## \example pca_metrics_dense_batch.py

import os
import sys

import daal.algorithms.pca as pca
import daal.algorithms.pca.quality_metric_set as quality_metric_set
from daal.algorithms.pca.quality_metric import explained_variance
from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable,
    NumericTableIface, BlockDescriptor, readWrite
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

datasetFileName = os.path.join('..', 'data', 'batch', 'pca_normalized.csv')
nVectors = 1000
nComponents = 5

qmsResult = None
eigenData = None

def trainModel():
    global eigenData

    # Initialize FileDataSource to retrieve the input data from a .csv file
    dataSource = FileDataSource(
        datasetFileName,
        DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from the input file
    dataSource.loadDataBlock(nVectors)

    # Create an algorithm for principal component analysis using the SVD method
    algorithm = pca.Batch(method=pca.svdDense)

    # Set the algorithm input data
    algorithm.input.setDataset(pca.data, dataSource.getNumericTable())

    # Compute results of the PCA algorithm
    result = algorithm.compute()
    eigenData = result.get(pca.eigenvalues)

def testPcaQuality():
    global qmsResult

    # Create a quality metric set object to compute quality metrics of the PCA algorithm
    qualityMetricSet = quality_metric_set.Batch(nComponents)
    explainedVariances = explained_variance.Input.downCast(qualityMetricSet.getInputDataCollection().getInput(quality_metric_set.explainedVariancesMetrics))
    explainedVariances.setInput(explained_variance.eigenvalues, eigenData)

    # Compute quality metrics
    qualityMetricSet.compute()

    # Retrieve the quality metrics
    qmsResult = qualityMetricSet.getResultCollection()

def printResults():
    print ("Quality metrics for PCA")
    result = explained_variance.Result.downCast(qmsResult.getResult(quality_metric_set.explainedVariancesMetrics))
    printNumericTable(result.getResult(explained_variance.explainedVariances), "Explained variances:")
    printNumericTable(result.getResult(explained_variance.explainedVariancesRatios), "Explained variances ratios:")
    printNumericTable(result.getResult(explained_variance.noiseVariance), "Noise variance:")

if __name__ == "__main__":
    trainModel()
    testPcaQuality()
    printResults()
