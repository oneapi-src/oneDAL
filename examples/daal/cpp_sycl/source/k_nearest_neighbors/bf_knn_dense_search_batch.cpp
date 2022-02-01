/* file: bf_knn_dense_search_batch.cpp */
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
!    C++ example of k-Nearest Neighbor for GPU in the batch processing mode.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-BF_KNN_DENSE_SEARCH_BATCH"></a>
 * \example bf_knn_dense_search_batch.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"
#include <cstdio>

using namespace std;
using namespace daal;
using namespace daal::algorithms;

using daal::data_management::internal::SyclHomogenNumericTable;
using daal::services::internal::SyclExecutionContext;

/* Input data set parameters */
const string trainDatasetFileName         = "../data/batch/k_nearest_neighbors_train.csv";
const string testDatasetFileName          = "../data/batch/k_nearest_neighbors_test.csv";
const string groundTruthDistancesFileName = "../data/batch/k_nearest_neighbors_distances_ground_truth.csv";
const string groundTruthIndicesFileName   = "../data/batch/k_nearest_neighbors_indices_ground_truth.csv";

const size_t nFeatures  = 5;
const size_t nClasses   = 5;
const size_t kNeighbors = 3;

void trainModel(bf_knn_classification::training::ResultPtr & trainingResult);
void testModel(bf_knn_classification::training::ResultPtr & trainingResult, bf_knn_classification::prediction::ResultPtr & predictionResult);
void readGroundTruth(NumericTablePtr & groundTruthIndices, NumericTablePtr & groundTruthDistances);
void printIndicesResults(NumericTablePtr & testGroundTruth, bf_knn_classification::prediction::ResultPtr & predictionResult);
void printDistancesResults(NumericTablePtr & testGroundTruth, bf_knn_classification::prediction::ResultPtr & predictionResult);

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 4, &trainDatasetFileName, &testDatasetFileName, &groundTruthDistancesFileName, &groundTruthIndicesFileName);

    for (const auto & deviceSelector : getListOfDevices())
    {
        const auto & nameDevice = deviceSelector.first;
        const auto & device     = deviceSelector.second;
        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        bf_knn_classification::training::ResultPtr trainingResult;
        trainModel(trainingResult);

        bf_knn_classification::prediction::ResultPtr searchResult;
        testModel(trainingResult, searchResult);

        NumericTablePtr groundTruthDistances, groundTruthIndices;
        readGroundTruth(groundTruthIndices, groundTruthDistances);

        printIndicesResults(groundTruthIndices, searchResult);
        printDistancesResults(groundTruthDistances, searchResult);
    }
    return 0;
}

void trainModel(bf_knn_classification::training::ResultPtr & trainingResult)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data
   * from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData        = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::notAllocate);
    NumericTablePtr trainGroundTruth = SyclHomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate);
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));
    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to train the BF kNN model */
    bf_knn_classification::training::Batch<> algorithm;

    /* Pass the training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);
    algorithm.parameter().k                = kNeighbors;
    algorithm.parameter().nClasses         = nClasses;
    algorithm.parameter().resultsToCompute = bf_knn_classification::computeDistances | bf_knn_classification::computeIndicesOfNeighbors;

    /* Train the BF kNN model */
    algorithm.compute();
    /* Retrieve the results of the training algorithm  */
    trainingResult = algorithm.getResult();
}

void testModel(bf_knn_classification::training::ResultPtr & trainingResult, bf_knn_classification::prediction::ResultPtr & predictionResult)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from
   * a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData        = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::doNotAllocate);
    NumericTablePtr testGroundTruth = SyclHomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate);
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create algorithm objects for BF kNN prediction with the default method */
    bf_knn_classification::prediction::Batch<> algorithm;

    /* Pass the testing data set and trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model, trainingResult->get(classifier::training::model));
    algorithm.parameter().k                = kNeighbors;
    algorithm.parameter().nClasses         = nClasses;
    algorithm.parameter().resultsToCompute = bf_knn_classification::computeDistances | bf_knn_classification::computeIndicesOfNeighbors;

    /* Compute prediction results */
    algorithm.compute();

    /* Retrieve algorithm results */
    predictionResult = algorithm.getResult();
}

void readGroundTruth(NumericTablePtr & groundTruthIndices, NumericTablePtr & groundTruthDistances)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from
   * a .csv file */
    FileDataSource<CSVFeatureManager> indicesDataSource(groundTruthIndicesFileName, DataSource::notAllocateNumericTable,
                                                        DataSource::doDictionaryFromContext);
    FileDataSource<CSVFeatureManager> distancesDataSource(groundTruthDistancesFileName, DataSource::notAllocateNumericTable,
                                                          DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    groundTruthIndices   = SyclHomogenNumericTable<>::create(kNeighbors, 0, NumericTable::doNotAllocate);
    groundTruthDistances = SyclHomogenNumericTable<>::create(kNeighbors, 0, NumericTable::doNotAllocate);

    /* Retrieve the data from input file */
    indicesDataSource.loadDataBlock(groundTruthIndices.get());
    distancesDataSource.loadDataBlock(groundTruthDistances.get());
}

void printIndicesResults(NumericTablePtr & testGroundTruth, bf_knn_classification::prediction::ResultPtr & predictionResult)
{
    auto indicesResult = predictionResult->get(bf_knn_classification::prediction::indices);
    printNumericTable<int>(testGroundTruth, "Indices Ground Truth (first 10 rows):", 10);
    printNumericTable<int>(indicesResult, "Computed Indices (first 10 rows):", 10);
}

void printDistancesResults(NumericTablePtr & testGroundTruth, bf_knn_classification::prediction::ResultPtr & predictionResult)
{
    auto distancesResult = predictionResult->get(bf_knn_classification::prediction::distances);
    printNumericTable(testGroundTruth, "Distances Ground Truth (first 10 rows):", 10);
    printNumericTable(distancesResult, "Computed Distances (first 10 rows):", 10);
}
