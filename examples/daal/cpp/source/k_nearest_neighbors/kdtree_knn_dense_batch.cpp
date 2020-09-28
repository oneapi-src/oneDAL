/* file: kdtree_knn_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
!    C++ example of k-Nearest Neighbor in the batch processing mode.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-KDTREE_KNN_DENSE_BATCH"></a>
 * \example kdtree_knn_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"
#include <cstdio>

#define TIME_MEASUREMENT
#define BINARY_DATASET

#ifdef TIME_MEASUREMENT
    #include "tbb/tick_count.h"
    #include <iostream>
#endif

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

#ifdef BINARY_DATASET
/* Input data set parameters */
string trainDatasetFileName            = "../data/batch/fTrainData_SUSY.bin";
string trainDatasetGroundTruthFileName = "../data/batch/fTrainLabel_SUSY.bin";
string testDatasetFileName             = "../data/batch/fPredictData_SUSY.bin";
string testDatasetLabelsFileName       = "../data/batch/fPredictLabel_SUSY.bin";
#else
/* Input data set parameters */
string trainDatasetFileName = "../data/batch/k_nearest_neighbors_train.csv";
string testDatasetFileName  = "../data/batch/k_nearest_neighbors_test.csv";
#endif

#ifdef BINARY_DATASET
const size_t nClasses = 2;
#else
size_t nFeatures            = 5;
size_t nClasses             = 5;
#endif

kdtree_knn_classification::training::ResultPtr trainingResult;
kdtree_knn_classification::prediction::ResultPtr predictionResult;
NumericTablePtr testGroundTruth;

void trainModel();
void testModel();
void printResults();

int main(int argc, char * argv[])
{
#ifdef BINARY_DATASET
    checkArguments(argc, argv, 4, &trainDatasetFileName, &trainDatasetGroundTruthFileName, &testDatasetFileName, &testDatasetLabelsFileName);
#else
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);
#endif

    trainModel();
    testModel();
    printResults();

    return 0;
}

void trainModel()
{
#ifdef TIME_MEASUREMENT
    const auto t0 = tbb::tick_count::now();
#endif

#ifdef BINARY_DATASET
    DAAL_INT64 observationCount;
    DAAL_INT64 featureCount;

    FILE * f = fopen(trainDatasetFileName, "rb");
    if (!f)
    {
        std::cout << "Training file opening failed!" << std::endl;
        return;
    }
    const size_t readCount1 = fread(&observationCount, sizeof(observationCount), 1, f);
    if (readCount1 != 1)
    {
        std::cout << "Reading number of observations from training file opening failed!" << std::endl;
        return;
    }
    const size_t readCount2 = fread(&featureCount, sizeof(featureCount), 1, f);
    if (readCount2 != 1)
    {
        std::cout << "Reading number of features from training file opening failed!" << std::endl;
        return;
    }

    std::cout << "Training data: " << observationCount << " x " << featureCount << std::endl;

    services::SharedPtr<SOANumericTable> trainData(new SOANumericTable(featureCount, observationCount, DictionaryIface::equal));
    trainData->getDictionarySharedPtr()->setAllFeatures<float>();
    trainData->resize(trainData->getNumberOfRows()); // Just to allocate memory.
    services::SharedPtr<SOANumericTable> trainGroundTruth(new SOANumericTable(1, observationCount));
    trainGroundTruth->getDictionarySharedPtr()->setAllFeatures<int>();
    trainGroundTruth->resize(trainGroundTruth->getNumberOfRows()); // Just to allocate memory.

    for (size_t d = 0; d < featureCount; ++d)
    {
        float * const ptr      = static_cast<float *>(trainData->getArray(d));
        const size_t readCount = fread(ptr, sizeof(float), observationCount, f);
        if (readCount != observationCount)
        {
            std::cout << "ERROR: Only " << readCount << " of " << observationCount << " read for feature " << d << std::endl;
            return;
        }
    }

    fclose(f);
    f = 0;

    {
        DAAL_INT64 observationCount2;
        DAAL_INT64 featureCount2;
        f = fopen(trainDatasetGroundTruthFileName, "rb");
        if (!f)
        {
            std::cout << "Training file with labels opening failed!" << std::endl;
            return;
        }
        const size_t readCount1 = fread(&observationCount2, sizeof(observationCount2), 1, f);
        if (readCount1 != 1)
        {
            std::cout << "Reading number of observations from training file with labels opening failed!" << std::endl;
            return;
        }

        const size_t readCount2 = fread(&featureCount2, sizeof(featureCount2), 1, f);
        if (readCount2 != 1)
        {
            std::cout << "Reading number of features from training file with labels opening failed!" << std::endl;
            return;
        }

        if (observationCount != observationCount2) { std::cout << "Training data and labels must have equal number of rows!" << std::endl; }
        if (featureCount2 != 1) { std::cout << "Training labels file must contain exactly one column!" << std::endl; }

        int * const ptr = static_cast<int *>(trainGroundTruth->getArray(0));
        vector<float> tempLabels(observationCount2);
        float * const tempLabelsPtr = &tempLabels[0];
        const size_t readCount      = fread(tempLabelsPtr, sizeof(float), observationCount2, f);
        if (readCount != observationCount2)
        {
            std::cout << "ERROR: Only " << readCount << " of " << observationCount2 << " read of class labels" << std::endl;
            return;
        }
        for (size_t i = 0, cnt = observationCount2; i != cnt; ++i) { ptr[i] = tempLabelsPtr[i]; }

        fclose(f);
        f = 0;
    }
#else
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr trainGroundTruth(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));

    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());
#endif

#ifdef TIME_MEASUREMENT
    const auto t1 = tbb::tick_count::now();
    std::cout << "Training dataset loading finished in " << (t1 - t0).seconds() << " seconds" << std::endl;
#endif

    /* Create an algorithm object to train the KD-tree based kNN model */
    kdtree_knn_classification::training::Batch<> algorithm;

    /* Pass the training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);
    algorithm.parameter.nClasses = nClasses;

#ifdef TIME_MEASUREMENT
    const auto t2 = tbb::tick_count::now();
    std::cout << "Training parameters setting finished in " << (t2 - t1).seconds() << " seconds" << std::endl;
#endif

    /* Train the KD-tree based kNN model */
    algorithm.compute();

#ifdef TIME_MEASUREMENT
    const auto t3 = tbb::tick_count::now();
    std::cout << "Training finished in " << (t3 - t2).seconds() << " seconds" << std::endl;
#endif

    /* Retrieve the results of the training algorithm  */
    trainingResult = algorithm.getResult();

#ifdef TIME_MEASUREMENT
    const auto t4 = tbb::tick_count::now();
    std::cout << "Training results getting in " << (t4 - t3).seconds() << " seconds" << std::endl;
#endif
}

void testModel()
{
#ifdef TIME_MEASUREMENT
    const auto t0 = tbb::tick_count::now();
#endif

#ifdef BINARY_DATASET
    DAAL_INT64 observationCount;
    DAAL_INT64 featureCount;

    FILE * f = fopen(testDatasetFileName, "rb");
    if (!f)
    {
        std::cout << "Testing file opening failed!" << std::endl;
        return;
    }
    const size_t readCount1 = fread(&observationCount, sizeof(observationCount), 1, f);
    if (readCount1 != 1)
    {
        std::cout << "Reading number of observations from testing file opening failed!" << std::endl;
        return;
    }
    const size_t readCount2 = fread(&featureCount, sizeof(featureCount), 1, f);
    if (readCount2 != 1)
    {
        std::cout << "Reading number of features from testing file opening failed!" << std::endl;
        return;
    }

    std::cout << "Testing data: " << observationCount << " x " << featureCount << std::endl;

    services::SharedPtr<SOANumericTable> testData(new SOANumericTable(featureCount, observationCount, DictionaryIface::equal));
    testData->getDictionarySharedPtr()->setAllFeatures<float>();
    testData->resize(testData->getNumberOfRows()); // Just to allocate memory.
    services::SharedPtr<SOANumericTable> testGroundTruth(new SOANumericTable(1, observationCount));
    testGroundTruth->getDictionarySharedPtr()->setAllFeatures<int>();
    testGroundTruth->resize(testGroundTruth->getNumberOfRows()); // Just to allocate memory.

    for (size_t d = 0; d < featureCount; ++d)
    {
        float * const ptr      = static_cast<float *>(testData->getArray(d));
        const size_t readCount = fread(ptr, sizeof(float), observationCount, f);
        if (readCount != observationCount)
        {
            std::cout << "ERROR: Only " << readCount << " of " << observationCount << " read for feature " << d << std::endl;
            return;
        }
    }

    fclose(f);
    f = 0;

    {
        DAAL_INT64 observationCount2;
        DAAL_INT64 featureCount2;
        f = fopen(testDatasetGroundTruthFileName, "rb");
        if (!f)
        {
            std::cout << "Testing file with labels opening failed!" << std::endl;
            return;
        }
        const size_t readCount1 = fread(&observationCount2, sizeof(observationCount2), 1, f);
        if (readCount1 != 1)
        {
            std::cout << "Reading number of observations from testing file with labels opening failed!" << std::endl;
            return;
        }

        const size_t readCount2 = fread(&featureCount2, sizeof(featureCount2), 1, f);
        if (readCount2 != 1)
        {
            std::cout << "Reading number of features from testing file with labels opening failed!" << std::endl;
            return;
        }

        if (observationCount != observationCount2) { std::cout << "Testing data and labels must have equal number of rows!" << std::endl; }
        if (featureCount2 != 1) { std::cout << "Testing labels file must contain exactly one column!" << std::endl; }

        int * const ptr = static_cast<int *>(testGroundTruth->getArray(0));
        vector<float> tempLabels(observationCount2);
        float * const tempLabelsPtr = &tempLabels[0];
        const size_t readCount      = fread(tempLabelsPtr, sizeof(float), observationCount2, f);
        if (readCount != observationCount2)
        {
            std::cout << "ERROR: Only " << readCount << " of " << observationCount2 << " read of class labels" << std::endl;
            return;
        }
        for (size_t i = 0, cnt = observationCount2; i != cnt; ++i) { ptr[i] = tempLabelsPtr[i]; }

        fclose(f);
        f = 0;
    }
#else
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    testGroundTruth = NumericTablePtr(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());
#endif

#ifdef TIME_MEASUREMENT
    const auto t1 = tbb::tick_count::now();
    std::cout << "Testing dataset loading finished in " << (t1 - t0).seconds() << " seconds" << std::endl;
#endif

    /* Create algorithm objects for KD-tree based kNN prediction with the default method */
    kdtree_knn_classification::prediction::Batch<> algorithm;

    /* Pass the testing data set and trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model, trainingResult->get(classifier::training::model));
    algorithm.parameter.nClasses = nClasses;

#ifdef TIME_MEASUREMENT
    const auto t2 = tbb::tick_count::now();
    std::cout << "Testing parameters setting finished in " << (t2 - t1).seconds() << " seconds" << std::endl;
#endif

    /* Compute prediction results */
    algorithm.compute();

#ifdef TIME_MEASUREMENT
    const auto t3 = tbb::tick_count::now();
    std::cout << "Testing finished in " << (t3 - t2).seconds() << " seconds" << std::endl;
#endif

    /* Retrieve algorithm results */
    predictionResult = algorithm.getResult();

#ifdef TIME_MEASUREMENT
    const auto t4 = tbb::tick_count::now();
    std::cout << "Testing results getting in " << (t4 - t3).seconds() << " seconds" << std::endl;
#endif
}

void printResults()
{
    printNumericTables<int, int>(testGroundTruth, predictionResult->get(kdtree_knn_classification::prediction::prediction), "Ground truth",
                                 "Classification results", "KD-tree based kNN classification results (first 20 observations):", 20);
}
