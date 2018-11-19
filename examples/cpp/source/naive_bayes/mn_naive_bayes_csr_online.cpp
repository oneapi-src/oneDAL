/* file: mn_naive_bayes_csr_online.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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

typedef float algorithmFPType; /* Algorithm floating-point type */

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

training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;
CSRNumericTablePtr trainData[nBlocks];
CSRNumericTablePtr testData;

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
    training::Online<algorithmFPType, training::fastCSR> algorithm(nClasses);

    for(size_t i = 0; i < nBlocks; i++)
    {
        /* Read trainDatasetFileNames and create a numeric table to store the input data */
        trainData[i] = CSRNumericTablePtr(createSparseTable<float>(trainDatasetFileNames[i]));
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
    testData = CSRNumericTablePtr(createSparseTable<float>(testDatasetFileName));

    /* Create an algorithm object to predict Naive Bayes values */
    prediction::Batch<algorithmFPType, prediction::fastCSR> algorithm(nClasses);

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
