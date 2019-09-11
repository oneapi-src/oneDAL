# file: lin_reg_metrics_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-LIN_REG_METRICS_DENSE_BATCH"></a>
## \example lin_reg_metrics_dense_batch.py

import os
import sys

import daal.algorithms.linear_regression as linear_regression
import daal.algorithms.linear_regression.quality_metric_set as quality_metric_set
from daal.algorithms.linear_regression import training, prediction
from daal.algorithms.linear_regression.quality_metric import single_beta, group_of_betas
from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable,
    NumericTableIface, BlockDescriptor, readWrite
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

trainDatasetFileName = os.path.join('..', 'data', 'batch', 'linear_regression_train.csv')

nFeatures = 10
nDependentVariables = 2

trainingResult = None
# predictionResult = None
qmsResult = None
trainData = None
trainDependentVariables = None

def trainModel(algorithm):
    global trainingResult, trainData, trainDependentVariables

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.set(training.data, trainData)
    algorithm.input.set(training.dependentVariables, trainDependentVariables)

    # Build the multiple linear regression model and retrieve the algorithm results
    trainingResult = algorithm.compute()
    printNumericTable(trainingResult.get(training.model).getBeta(), "Linear Regression coefficients:")

def predictResults(trainData):
    # Create an algorithm object to predict values of multiple linear regression
    algorithm = prediction.Batch()

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setTable(prediction.data, trainData)
    algorithm.input.setModel(prediction.model, trainingResult.get(training.model))

    # Predict values of multiple linear regression and retrieve the algorithm results
    predictionResult = algorithm.compute()
    return predictionResult.get(prediction.prediction)

def predictReducedModelResults(trainData):
    model = trainingResult.get(training.model)

    betas = model.getBeta()
    nBetas = model.getNumberOfBetas()

    j1 = 2
    j2 = 10
    savedBeta = [[None] * nBetas for _ in range(nDependentVariables)]

    block = BlockDescriptor()
    betas.getBlockOfRows(0, nDependentVariables, readWrite, block)
    pBeta = block.getArray()

    for i in range(0, nDependentVariables):
        savedBeta[i][j1] = pBeta[i][j1]
        savedBeta[i][j2] = pBeta[i][j2]
        pBeta[i][j1] = 0
        pBeta[i][j2] = 0
    betas.releaseBlockOfRows(block)

    predictedResults = predictResults(trainData)

    block = BlockDescriptor()
    betas.getBlockOfRows(0, nDependentVariables, readWrite, block)
    pBeta = block.getArray()

    for i in range(0, nDependentVariables):
        pBeta[i][j1] = savedBeta[i][j1]
        pBeta[i][j2] = savedBeta[i][j2]
    betas.releaseBlockOfRows(block)
    return predictedResults

def testModelQuality():
    global trainingResult, qmsResult

    predictedResults = predictResults(trainData)
    printNumericTable(trainDependentVariables, "Expected responses (first 20 rows):", 20)
    printNumericTable(predictedResults, "Predicted responses (first 20 rows):", 20)

    model = trainingResult.get(linear_regression.training.model)
    predictedReducedModelResults = predictReducedModelResults(trainData)
    printNumericTable(predictedReducedModelResults, "Responses predicted with reduced model (first 20 rows):", 20)

    # Create a quality metric set object to compute quality metrics of the linear regression algorithm
    nBetaReducedModel = model.getNumberOfBetas() - 2
    qualityMetricSet = quality_metric_set.Batch(model.getNumberOfBetas(), nBetaReducedModel)
    singleBeta = single_beta.Input.downCast(qualityMetricSet.getInputDataCollection().getInput(quality_metric_set.singleBeta))
    singleBeta.setDataInput(single_beta.expectedResponses, trainDependentVariables)
    singleBeta.setDataInput(single_beta.predictedResponses, predictedResults)
    singleBeta.setModelInput(single_beta.model, model)

    # Set input for a group of betas metrics algorithm
    groupOfBetas = group_of_betas.Input.downCast(qualityMetricSet.getInputDataCollection().getInput(quality_metric_set.groupOfBetas))
    groupOfBetas.set(group_of_betas.expectedResponses, trainDependentVariables)
    groupOfBetas.set(group_of_betas.predictedResponses, predictedResults)
    groupOfBetas.set(group_of_betas.predictedReducedModelResponses, predictedReducedModelResults)

    # Compute quality metrics
    qualityMetricSet.compute()

    # Retrieve the quality metrics
    qmsResult = qualityMetricSet.getResultCollection()

def printResults():
    # Print the quality metrics for a single beta
    print ("Quality metrics for a single beta")
    result = single_beta.Result.downCast(qmsResult.getResult(quality_metric_set.singleBeta))
    printNumericTable(result.getResult(single_beta.rms), "Root means square errors for each response (dependent variable):")
    printNumericTable(result.getResult(single_beta.variance), "Variance for each response (dependent variable):")
    printNumericTable(result.getResult(single_beta.zScore), "Z-score statistics:")
    printNumericTable(result.getResult(single_beta.confidenceIntervals), "Confidence intervals for each beta coefficient:")
    printNumericTable(result.getResult(single_beta.inverseOfXtX), "Inverse(Xt * X) matrix:")

    coll = result.getResultDataCollection(single_beta.betaCovariances)
    for i in range(0, coll.size()):
        message = "Variance-covariance matrix for betas of " + str(i) + "-th response\n"
        betaCov = result.get(single_beta.betaCovariances, i)
        printNumericTable(betaCov, message)

    # Print quality metrics for a group of betas
    print ("Quality metrics for a group of betas")
    result = group_of_betas.Result.downCast(qmsResult.getResult(quality_metric_set.groupOfBetas))

    printNumericTable(result.get(group_of_betas.expectedMeans), "Means of expected responses for each dependent variable:", 0, 0, 20)
    printNumericTable(result.get(group_of_betas.expectedVariance), "Variance of expected responses for each dependent variable:", 0, 0, 20)
    printNumericTable(result.get(group_of_betas.regSS), "Regression sum of squares of expected responses:", 0, 0, 20)
    printNumericTable(result.get(group_of_betas.resSS), "Sum of squares of residuals for each dependent variable:", 0, 0, 20)
    printNumericTable(result.get(group_of_betas.tSS), "Total sum of squares for each dependent variable:", 0, 0, 20)
    printNumericTable(result.get(group_of_betas.determinationCoeff), "Determination coefficient for each dependent variable:", 0, 0, 20)
    printNumericTable(result.get(group_of_betas.fStatistics), "F-statistics for each dependent variable:", 0, 0, 20)

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(trainDatasetFileName,
                                DataSourceIface.notAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)

    # Create Numeric Tables for data and values for dependent variable
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    trainDependentVariables = HomogenNumericTable(nDependentVariables, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(trainData, trainDependentVariables)

    # Retrieve the data from the input file
    dataSource.loadDataBlock(mergedData)

    for i in range(0, 2):
        if i == 0:
            print ("Train model with normal equation algorithm.")
            algorithm = training.Batch()
            trainModel(algorithm)
        else:
            print ("Train model with QR algorithm.")
            algorithm = training.Batch(method=training.qrDense)
            trainModel(algorithm)
        testModelQuality()
        printResults()
