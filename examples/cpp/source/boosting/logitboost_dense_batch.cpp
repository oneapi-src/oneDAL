/* file: logitboost_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
!  Content:
!    C++ example of LogitBoost classification.
!
!    The program trains the LogitBoost model on a supplied training datasetFileName
!    and then performs classification of previously unseen data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LOGITBOOST_BATCH"></a>
 * \example logitboost_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string trainDatasetFileName     = "../data/batch/logitboost_train.csv";

string testDatasetFileName      = "../data/batch/logitboost_test.csv";

const size_t nFeatures = 20;
const size_t nClasses           = 5;

/* LogitBoost algorithm parameters */
const size_t maxIterations      = 100;    /* Maximum number of terms in additive regression */
const double accuracyThreshold  = 0.01;   /* Training accuracy */

/* Model object for the LogitBoost algorithm */
logitboost::ModelPtr model;

classifier::prediction::ResultPtr predictionResult;

NumericTablePtr testGroundTruth;

void trainModel();
void testModel();
void printResults();

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    trainModel();

    testModel();

    printResults();

    return 0;
}

void trainModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr trainGroundTruth(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));

    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to train the LogitBoost model */
    logitboost::training::Batch<> algorithm(nClasses);
    algorithm.parameter.maxIterations = maxIterations;
    algorithm.parameter.accuracyThreshold = accuracyThreshold;

    /* Pass the training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);

    /* Train the LogitBoost model */
    algorithm.compute();

    /* Retrieve the results of the training algorithm */
    logitboost::training::ResultPtr trainingResult = algorithm.getResult();
    model = trainingResult->get(classifier::training::model);
}

void testModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::notAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    testGroundTruth = NumericTablePtr(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create algorithm objects for LogitBoost prediction with the default method */
    logitboost::prediction::Batch<> algorithm(nClasses);

    /* Pass the testing data set and trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data,  testData);
    algorithm.input.set(classifier::prediction::model, model);

    /* Compute prediction results */
    algorithm.compute();

    /* Retrieve algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults()
{
    printNumericTables<int, int>(testGroundTruth,
                                 predictionResult->get(classifier::prediction::prediction),
                                 "Ground truth", "Classification results",
                                 "LogitBoost classification results (first 20 observations):", 20);
}
