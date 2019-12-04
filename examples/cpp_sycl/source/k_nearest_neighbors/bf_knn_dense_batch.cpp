/* file: bf_knn_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
!    C++ example of k-Nearest Neighbor for GPU in the batch processing mode.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-BF_KNN_DENSE_BATCH"></a>
 * \example bf_knn_dense_batch.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"
#include <cstdio>

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const string trainDatasetFileName            = "../data/batch/k_nearest_neighbors_train.csv";
const string testDatasetFileName             = "../data/batch/k_nearest_neighbors_test.csv";

const size_t nFeatures = 5;
const size_t nClasses  = 5;

void trainModel(bf_knn_classification::training::ResultPtr& trainingResult);
void testModel(bf_knn_classification::training::ResultPtr& trainingResult,
                classifier::prediction::ResultPtr& predictionResult,
                NumericTablePtr& testGroundTruth);
void printResults(NumericTablePtr& testGroundTruth, classifier::prediction::ResultPtr& predictionResult);

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    for (const auto& deviceSelector : getListOfDevices())
    {
        const auto& nameDevice = deviceSelector.first;
        const auto& device = deviceSelector.second;
        if(!device.is_gpu())
            continue;
        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        daal::services::SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        bf_knn_classification::training::ResultPtr trainingResult;
        classifier::prediction::ResultPtr predictionResult;
        NumericTablePtr testGroundTruth;

        trainModel(trainingResult);
        testModel(trainingResult, predictionResult, testGroundTruth);
        printResults(testGroundTruth, predictionResult);
    }
    return 0;
}

void trainModel(bf_knn_classification::training::ResultPtr& trainingResult)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::notAllocate);
    NumericTablePtr trainGroundTruth = SyclHomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate);
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));
    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to train the BF kNN model */
    bf_knn_classification::training::Batch<> algorithm;

    /* Pass the training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);
    algorithm.parameter().nClasses = nClasses;
    /* Train the BF kNN model */
    algorithm.compute();
    /* Retrieve the results of the training algorithm  */
    trainingResult = algorithm.getResult();
}

void testModel(bf_knn_classification::training::ResultPtr& trainingResult,
                classifier::prediction::ResultPtr& predictionResult,
                NumericTablePtr& testGroundTruth)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::notAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::doNotAllocate);
    testGroundTruth = SyclHomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate);
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create algorithm objects for BF kNN prediction with the default method */
    bf_knn_classification::prediction::Batch<> algorithm;

    /* Pass the testing data set and trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data,  testData);
    algorithm.input.set(classifier::prediction::model, trainingResult->get(classifier::training::model));

    /* Compute prediction results */
    algorithm.compute();

    /* Retrieve algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults(NumericTablePtr& testGroundTruth, classifier::prediction::ResultPtr& predictionResult)
{
    printNumericTables<int, int>(testGroundTruth,
                                 predictionResult->get(classifier::prediction::prediction),
                                 "Ground truth", "Classification results",
                                 "BF kNN classification results (first 20 observations):", 20);
}
