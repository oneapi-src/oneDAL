# file: svm_two_class_metrics_dense_batch.py
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

#
# !  Content:
# !    Python example of two-class support vector machine (SVM) quality metrics
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-SVM_TWO_CLASS_QUALITY_METRIC_SET_BATCH"></a>
## \example svm_two_class_metrics_dense_batch.py
#

import os
import sys

from daal.algorithms import kernel_function
from daal.algorithms.classifier.quality_metric import binary_confusion_matrix
from daal.algorithms import svm
from daal.algorithms import classifier
from daal.data_management import (
    DataSourceIface, FileDataSource, readOnly, BlockDescriptor,
    HomogenNumericTable, NumericTableIface, MergedNumericTable
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTables, printNumericTable

# Input data set parameters
DATA_PREFIX = os.path.join('..', 'data', 'batch')
trainDatasetFileName = os.path.join(DATA_PREFIX, 'svm_two_class_train_dense.csv')
testDatasetFileName = os.path.join(DATA_PREFIX, 'svm_two_class_test_dense.csv')

nFeatures = 20

# Parameters for the SVM kernel function
kernel = kernel_function.linear.Batch()

# Model object for the SVM algorithm
trainingResult = None
predictionResult = None
qualityMetricSetResult = None

predictedLabels = None
groundTruthLabels = None


def trainModel():
    global trainingResult

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for training data and labels
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(trainData, trainGroundTruth)

    # Retrieve the data from the input file
    trainDataSource.loadDataBlock(mergedData)

    # Create an algorithm object to train the SVM model
    algorithm = svm.training.Batch()

    algorithm.parameter.kernel = kernel
    algorithm.parameter.cacheSize = 600000000

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.set(classifier.training.data, trainData)
    algorithm.input.set(classifier.training.labels, trainGroundTruth)

    # Build the SVM model and get the algorithm results
    trainingResult = algorithm.compute()

def testModel():
    global predictionResult, groundTruthLabels

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    testDataSource = FileDataSource(
        testDatasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for testing data and labels
    testData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    groundTruthLabels = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(testData, groundTruthLabels)

    # Retrieve the data from input file
    testDataSource.loadDataBlock(mergedData)

    # Create an algorithm object to predict SVM values
    algorithm = svm.prediction.Batch()

    algorithm.parameter.kernel = kernel

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setTable(classifier.prediction.data, testData)
    algorithm.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))

    # Predict SVM values
    # returns Result class from daal.algorithms.classifier.prediction
    predictionResult = algorithm.compute()


def testModelQuality():
    global predictedLabels, qualityMetricSetResult, groundTruthLabels

    # Retrieve predicted labels
    predictedLabels = predictionResult.get(classifier.prediction.prediction)

    # Create a quality metric set object to compute quality metrics of the SVM algorithm
    qualityMetricSet = svm.quality_metric_set.Batch()

    input = qualityMetricSet.getInputDataCollection().getInput(svm.quality_metric_set.confusionMatrix)

    input.set(binary_confusion_matrix.predictedLabels,   predictedLabels)
    input.set(binary_confusion_matrix.groundTruthLabels, groundTruthLabels)

    # Compute quality metrics and get the quality metrics
    # returns ResultCollection class from svm.quality_metric_set
    qualityMetricSetResult = qualityMetricSet.compute()


def printResults():

    # Print the classification results
    printNumericTables(
        groundTruthLabels, predictedLabels,
        "Ground truth", "Classification results",
        "SVM classification results (first 20 observations):", 20, interval=15, flt64=False
    )

    # Print the quality metrics
    qualityMetricResult = qualityMetricSetResult.getResult(svm.quality_metric_set.confusionMatrix)
    printNumericTable(qualityMetricResult.get(binary_confusion_matrix.confusionMatrix), "Confusion matrix:")

    block = BlockDescriptor()
    qualityMetricsTable = qualityMetricResult.get(binary_confusion_matrix.binaryMetrics)
    qualityMetricsTable.getBlockOfRows(0, 1, readOnly, block)
    qualityMetricsData = block.getArray().flatten()
    print("Accuracy:      {0:.3f}".format(qualityMetricsData[binary_confusion_matrix.accuracy]))
    print("Precision:     {0:.3f}".format(qualityMetricsData[binary_confusion_matrix.precision]))
    print("Recall:        {0:.3f}".format(qualityMetricsData[binary_confusion_matrix.recall]))
    print("F-score:       {0:.3f}".format(qualityMetricsData[binary_confusion_matrix.fscore]))
    print("Specificity:   {0:.3f}".format(qualityMetricsData[binary_confusion_matrix.specificity]))
    print("AUC:           {0:.3f}".format(qualityMetricsData[binary_confusion_matrix.AUC]))
    qualityMetricsTable.releaseBlockOfRows(block)

if __name__ == "__main__":
    trainModel()
    testModel()
    testModelQuality()
    printResults()
