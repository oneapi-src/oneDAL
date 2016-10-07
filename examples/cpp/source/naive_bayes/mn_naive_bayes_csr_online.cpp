/* file: mn_naive_bayes_csr_online.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
!    C++ example of Naive Bayes classification in the online processing mode.
!
!    The program trains the Naive Bayes model on a supplied training data set in
!    compressed sparse rows (CSR)__format and then performs classification of
!    previously unseen data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-MULTINOMIAL_NAIVE_BAYES_CSR_ONLINE"></a>
 * \example mn_naive_bayes_csr_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::multinomial_naive_bayes;

/* Input data set parameters */
const string trainDatasetFileNames[4]     =
{
    "../data/online/naivebayes_train_csr_1.csv", "../data/online/naivebayes_train_csr_2.csv",
    "../data/online/naivebayes_train_csr_3.csv", "../data/online/naivebayes_train_csr_4.csv"
};
const string trainGroundTruthFileNames[4] =
{
    "../data/online/naivebayes_train_labels_1.csv", "../data/online/naivebayes_train_labels_2.csv",
    "../data/online/naivebayes_train_labels_3.csv", "../data/online/naivebayes_train_labels_4.csv"
};

string testDatasetFileName      = "../data/online/naivebayes_test_csr.csv";
string testGroundTruthFileName  = "../data/online/naivebayes_test_labels.csv";

const size_t nTrainVectorsInBlock = 8000;
const size_t nTestObservations    = 2000;
const size_t nClasses             = 20;
const size_t nBlocks              = 4;

services::SharedPtr<training::Result> trainingResult;
services::SharedPtr<classifier::prediction::Result> predictionResult;
services::SharedPtr<CSRNumericTable> trainData[nBlocks];
services::SharedPtr<CSRNumericTable> testData;

void trainModel();
void testModel();
void printResults();

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 10,
        &trainDatasetFileNames[0], &trainDatasetFileNames[1],
        &trainDatasetFileNames[2], &trainDatasetFileNames[3],
        &trainGroundTruthFileNames[0], &trainGroundTruthFileNames[1],
        &trainGroundTruthFileNames[2], &trainGroundTruthFileNames[3],
        &testDatasetFileName, &testGroundTruthFileName);

    trainModel();

    testModel();

    printResults();

    return 0;
}

void trainModel()
{
    /* Create an algorithm object to train the Naive Bayes model */
    training::Online<double, training::fastCSR> algorithm(nClasses);

    for(size_t i = 0; i < nBlocks; i++)
    {
        /* Read trainDatasetFileNames and create a numeric table to store the input data */
        trainData[i] = services::SharedPtr<CSRNumericTable>(createSparseTable<double>(trainDatasetFileNames[i]));
        FileDataSource<CSVFeatureManager> trainLabelsSource(trainGroundTruthFileNames[i],
                                                        DataSource::doAllocateNumericTable,
                                                        DataSource::doDictionaryFromContext);
        trainLabelsSource.loadDataBlock(nTrainVectorsInBlock);
        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(classifier::training::data,   trainData[i]);
        algorithm.input.set(classifier::training::labels, trainLabelsSource.getNumericTable());

        /* Build the Naive Bayes model */
        algorithm.compute();
    }
    /* Finalize the Naive Bayes model */
    algorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel()
{
    /* Read testDatasetFileName and create a numeric table to store the input data */
    testData = services::SharedPtr<CSRNumericTable>(createSparseTable<double>(testDatasetFileName));

    /* Create an algorithm object to predict Naive Bayes values */
    prediction::Batch<double, prediction::fastCSR> algorithm(nClasses);

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data,  testData);
    algorithm.input.set(classifier::prediction::model, trainingResult->get(classifier::training::model));

    /* Predict Naive Bayes values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults()
{
    FileDataSource<CSVFeatureManager> testGroundTruth(testGroundTruthFileName,
                                                      DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);
    testGroundTruth.loadDataBlock(nTestObservations);

    printNumericTables<int, int>(testGroundTruth.getNumericTable().get(),
                                 predictionResult->get(classifier::prediction::prediction).get(),
                                 "Ground truth", "Classification results",
                                 "NaiveBayes classification results (first 20 observations):", 20);
}
