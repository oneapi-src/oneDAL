# file: svm_multi_class_metrics_dense_batch.py
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
# !    Python example of multi-class support vector machine (SVM) quality metrics
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-SVM_MULTI_CLASS_QUALITY_METRIC_SET_BATCH"></a>
## \example svm_multi_class_metrics_dense_batch.py
#

import os
import sys
import numpy as np

from daal.algorithms.classifier.quality_metric import multiclass_confusion_matrix
from daal.algorithms import svm
from daal.algorithms import kernel_function
from daal.algorithms import multi_class_classifier
from daal.algorithms import classifier
from daal.data_management import (
    DataSourceIface, FileDataSource, readOnly, BlockDescriptor, HomogenNumericTable,
    NumericTableIface, MergedNumericTable
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTables, printNumericTable

# Input data set parameters
DATA_PREFIX = os.path.join('..', 'data', 'batch')
trainDatasetFileName = os.path.join(DATA_PREFIX, 'svm_multi_class_train_dense.csv')
testDatasetFileName = os.path.join(DATA_PREFIX, 'svm_multi_class_test_dense.csv')

nFeatures = 20
nClasses = 5

training = svm.training.Batch(fptype=np.float64)
prediction = svm.prediction.Batch(fptype=np.float64)

# Model object for the multi-class classifier algorithm
trainingResult = None
predictionResult = None

# Parameters for the multi-class classifier kernel function
kernel = kernel_function.linear.Batch(fptype=np.float64)

qualityMetricSetResult = None
predictedLabels = None
groundTruthLabels = None


def trainModel():
    global trainingResult

    # Initialize FileDataSource to retrieve the input data from a .csv file
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

    # Create an algorithm object to train the multi-class SVM model
    algorithm = multi_class_classifier.training.Batch(nClasses,fptype=np.float64)

    algorithm.parameter.training = training
    algorithm.parameter.prediction = prediction

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.set(classifier.training.data, trainData)
    algorithm.input.set(classifier.training.labels, trainGroundTruth)

    # Build the multi-class SVM model and get the algorithm results
    trainingResult = algorithm.compute()


def testModel():
    global predictionResult, groundTruthLabels

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file
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

    # Create an algorithm object to predict multi-class SVM values
    algorithm = multi_class_classifier.prediction.Batch(nClasses,fptype=np.float64)

    algorithm.parameter.training = training
    algorithm.parameter.prediction = prediction

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setTable(classifier.prediction.data, testData)
    algorithm.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))

    # Predict multi-class SVM values and get the Result class from daal.algorithms.classifier.prediction
    predictionResult = algorithm.compute()


def testModelQuality():
    global predictedLabels, qualityMetricSetResult

    # Retrieve predicted labels
    predictedLabels = predictionResult.get(classifier.prediction.prediction)

    # Create a quality metric set object to compute quality metrics of the multi-class classifier algorithm
    qualityMetricSet = multi_class_classifier.quality_metric_set.Batch(nClasses)
    input = qualityMetricSet.getInputDataCollection().getInput(multi_class_classifier.quality_metric_set.confusionMatrix)

    input.set(multiclass_confusion_matrix.predictedLabels,   predictedLabels)
    input.set(multiclass_confusion_matrix.groundTruthLabels, groundTruthLabels)

    # Compute quality metrics and get the quality metrics
    # returns ResultCollection class from daal.algorithms.multi_class_classifier.quality_metric_set
    qualityMetricSetResult = qualityMetricSet.compute()

def printResults():

    # Print the classification results
    printNumericTables(
        groundTruthLabels, predictedLabels,
        "Ground truth", "Classification results",
        "SVM classification results (first 20 observations):", 20, interval=15, flt64=False
    )
    # Print the quality metrics
    qualityMetricResult = qualityMetricSetResult.getResult(multi_class_classifier.quality_metric_set.confusionMatrix)
    printNumericTable(qualityMetricResult.get(multiclass_confusion_matrix.confusionMatrix), "Confusion matrix:")

    block = BlockDescriptor()
    qualityMetricsTable = qualityMetricResult.get(multiclass_confusion_matrix.multiClassMetrics)
    qualityMetricsTable.getBlockOfRows(0, 1, readOnly, block)
    qualityMetricsData = block.getArray().flatten()
    print("Average accuracy: {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.averageAccuracy]))
    print("Error rate:       {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.errorRate]))
    print("Micro precision:  {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.microPrecision]))
    print("Micro recall:     {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.microRecall]))
    print("Micro F-score:    {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.microFscore]))
    print("Macro precision:  {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.macroPrecision]))
    print("Macro recall:     {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.macroRecall]))
    print("Macro F-score:    {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.macroFscore]))
    qualityMetricsTable.releaseBlockOfRows(block)

if __name__ == "__main__":
    training.parameter.cacheSize = 100000000
    training.parameter.kernel = kernel
    prediction.parameter.kernel = kernel

    trainModel()
    testModel()
    testModelQuality()
    printResults()
