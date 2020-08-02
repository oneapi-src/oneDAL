/* file: svm_regression_thunder_dense_batch.cpp */
/*******************************************************************************
 * Copyright 2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/*
!  Content:
!    C++ example of two-class support vector machine (SVM) classification using
!    the Thunder method
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_REGRESSION_THUNDER_DENSE_BATCH"></a>
 * \example svm_regression_thunder_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
string trainDatasetFileName = "../data/batch/svm_regression_train_dense.csv";
string testDatasetFileName  = "../data/batch/svm_regression_test_dense.csv";

const size_t nFeatures = 20;

/* Parameters for the SVM kernel function */
kernel_function::KernelIfacePtr kernel(new kernel_function::rbf::Batch<>());

/* Model object for the SVM algorithm */
svm::training::ResultPtr trainingResult;
regression::prediction::ResultPtr predictionResult;
NumericTablePtr testGroundTruth;

void trainModel();
void testModel();
void printResults();

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    trainModel();
    testModel();
    printResults();

    return 0;
}

void trainModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data
   * from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData        = HomogenNumericTable<>::create(nFeatures, 0, NumericTable::doNotAllocate);
    NumericTablePtr trainGroundTruth = HomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate);
    NumericTablePtr mergedData       = MergedNumericTable::create(trainData, trainGroundTruth);

    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to train the SVM model using the Thunder method
   */
    svm::regression::training::Batch<float, svm::training::thunder> algorithm;

    algorithm.parameter().kernel  = kernel;
    algorithm.parameter().epsilon = 1e-3;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(regression::training::data, trainData);
    algorithm.input.set(regression::training::dependentVariables, trainGroundTruth);

    /* Build the SVM model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from
   * a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData   = HomogenNumericTable<>::create(nFeatures, 0, NumericTable::doNotAllocate);
    testGroundTruth            = HomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate);
    NumericTablePtr mergedData = MergedNumericTable::create(testData, testGroundTruth);

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict SVM values */
    svm::prediction::Batch<> algorithm;

    algorithm.parameter().kernel = kernel;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(regression::prediction::data, testData);
    algorithm.input.set(regression::prediction::model, trainingResult->get(regression::training::model));

    /* Predict SVM values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults()
{
    printNumericTables<float, float>(testGroundTruth, predictionResult->get(regression::prediction::prediction), "Ground truth", "Regression results",
                                     "SVM regression results (first 20 observations):", 20);
}
