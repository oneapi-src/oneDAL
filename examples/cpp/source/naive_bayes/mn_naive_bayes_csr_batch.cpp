/* file: mn_naive_bayes_csr_batch.cpp */
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
!    C++ example of Naive Bayes classification in the batch processing mode.
!
!    The program trains the Naive Bayes model on a supplied training data set in
!    compressed sparse rows (CSR) format and then performs classification of
!    previously unseen data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-MULTINOMIAL_NAIVE_BAYES_CSR_BATCH"></a>
 * \example mn_naive_bayes_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::multinomial_naive_bayes;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
string trainDatasetFileName     = "../data/batch/naivebayes_train_csr.csv";
string trainGroundTruthFileName = "../data/batch/naivebayes_train_labels.csv";

string testDatasetFileName      = "../data/batch/naivebayes_test_csr.csv";
string testGroundTruthFileName  = "../data/batch/naivebayes_test_labels.csv";

const size_t nTrainObservations = 8000;
const size_t nTestObservations  = 2000;
const size_t nClasses           = 20;

training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;

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
    FileDataSource<CSVFeatureManager> trainGroundTruthSource(trainGroundTruthFileName,
                                                             DataSource::doAllocateNumericTable,
                                                             DataSource::doDictionaryFromContext);

    /* Retrieve the data from input files */
    CSRNumericTablePtr trainData(createSparseTable<float>(trainDatasetFileName));
    trainGroundTruthSource.loadDataBlock(nTrainObservations);

    /* Create an algorithm object to train the Naive Bayes model */
    training::Batch<algorithmFPType, training::fastCSR> algorithm(nClasses);

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data,   trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruthSource.getNumericTable());

    /* Build the Naive Bayes model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    CSRNumericTablePtr testData(createSparseTable<float>(testDatasetFileName));

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
