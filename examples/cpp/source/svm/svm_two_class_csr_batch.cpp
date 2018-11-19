/* file: svm_two_class_csr_batch.cpp */
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
!    C++ example of two-class support vector machine (SVM) classification
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_TWO_CLASS_CSR_BATCH"></a>
 * \example svm_two_class_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string trainDatasetFileName     = "../data/batch/svm_two_class_train_csr.csv";
string trainLabelsFileName      = "../data/batch/svm_two_class_train_labels.csv";

string testDatasetFileName      = "../data/batch/svm_two_class_test_csr.csv";
string testLabelsFileName       = "../data/batch/svm_two_class_test_labels.csv";

/* Parameters for the SVM kernel function */
kernel_function::KernelIfacePtr kernel(
    new kernel_function::linear::Batch<float, kernel_function::linear::fastCSR>());

/* Model object for the SVM algorithm */
svm::training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;

void trainModel();
void testModel();
void printResults();

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 4, &trainDatasetFileName, &trainLabelsFileName, &testDatasetFileName, &testLabelsFileName);

    trainModel();

    testModel();

    printResults();

    return 0;
}

void trainModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainLabelsDataSource(trainLabelsFileName,
                                                            DataSource::doAllocateNumericTable,
                                                            DataSource::doDictionaryFromContext);

    /* Create numeric table for training data */
    CSRNumericTablePtr trainData(createSparseTable<float>(trainDatasetFileName));

    /* Retrieve the data from the input file */
    trainLabelsDataSource.loadDataBlock();

    /* Create an algorithm object to train the SVM model */
    svm::training::Batch<> algorithm;

    algorithm.parameter.kernel = kernel;
    algorithm.parameter.cacheSize = 40000000;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainLabelsDataSource.getNumericTable());

    /* Build the SVM model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel()
{
    /* Create Numeric Tables for testing data */
    NumericTablePtr testData(createSparseTable<float>(testDatasetFileName));

    /* Create an algorithm object to predict SVM values */
    svm::prediction::Batch<> algorithm;

    algorithm.parameter.kernel = kernel;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));

    /* Predict SVM values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testLabelsDataSource(testLabelsFileName,
                                                           DataSource::doAllocateNumericTable,
                                                           DataSource::doDictionaryFromContext);
    /* Retrieve the data from input file */
    testLabelsDataSource.loadDataBlock();
    NumericTablePtr testGroundTruth = testLabelsDataSource.getNumericTable();

    printNumericTables<int, float>(testGroundTruth,
                                    predictionResult->get(classifier::prediction::prediction),
                                    "Ground truth\t", "Classification results",
                                    "SVM classification results (first 20 observations):", 20);
}
