/* file: log_reg_dense_batch.cpp */
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
!    C++ example of logistic regression in the batch processing mode.
!
!    The program trains the logistic regression model on a training
!    datasetFileName and computes classification for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LOG_REG_DENSE_BATCH"></a>
 * \example log_reg_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::logistic_regression;

/* Input data set parameters */
const string trainDatasetFileName = "../data/batch/logreg_train.csv";
const string testDatasetFileName = "../data/batch/logreg_test.csv";
const size_t nFeatures = 6;  /* Number of features in training and testing data sets */
const size_t nClasses = 5;  /* Number of classes */

training::ResultPtr trainModel();
void testModel(const training::ResultPtr& res);
void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar);

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    training::ResultPtr trainingResult = trainModel();
    testModel(trainingResult);

    return 0;
}

training::ResultPtr trainModel()
{
    /* Create Numeric Tables for training data and dependent variables */
    NumericTablePtr trainData;
    NumericTablePtr trainDependentVariable;

    loadData(trainDatasetFileName, trainData, trainDependentVariable);

    /* Create an algorithm object to train the logistic regression model */
    training::Batch<> algorithm(nClasses);

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainDependentVariable);
    algorithm.parameter().penaltyL1 = 0.1f;
    algorithm.parameter().penaltyL2 = 0.1f;

    /* Build the logistic regression model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    training::ResultPtr trainingResult = algorithm.getResult();
    logistic_regression::interface1::ModelPtr modelptr = trainingResult->get(classifier::training::model);
    if(modelptr.get())
    {
        printNumericTable(modelptr->getBeta(), "Logistic Regression coefficients:");
    }
    else
    {
        std::cout << "Null model pointer" << std::endl;
    }
    return trainingResult;
}

void testModel(const training::ResultPtr& trainingResult)
{
    /* Create Numeric Tables for testing data and ground truth values */
    NumericTablePtr testData;
    NumericTablePtr testGroundTruth;

    loadData(testDatasetFileName, testData, testGroundTruth);

    /* Create an algorithm object to predict values of logistic regression */
    prediction::Batch<> algorithm(nClasses);
    algorithm.parameter().resultsToCompute |= prediction::computeClassesProbabilities | prediction::computeClassesLogProbabilities;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model, trainingResult->get(classifier::training::model));

    /* Predict values of logistic regression */
    algorithm.compute();

    /* Retrieve the algorithm results */
    logistic_regression::prediction::ResultPtr predictionResult = logistic_regression::prediction::Result::cast(algorithm.getResult());
    printNumericTable(predictionResult->get(classifier::prediction::prediction),
        "Logistic regression prediction results (first 10 rows):", 10);
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10);
    printNumericTable(predictionResult->get(logistic_regression::prediction::probabilities),
        "Logistic regression prediction probabilities (first 10 rows):", 10);
    printNumericTable(predictionResult->get(logistic_regression::prediction::logProbabilities),
        "Logistic regression prediction log probabilities (first 10 rows):", 10);
}

void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(fileName,
        DataSource::notAllocateNumericTable,
        DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and dependent variables */
    pData.reset(new HomogenNumericTable<>(nFeatures, 0, NumericTable::notAllocate));
    pDependentVar.reset(new HomogenNumericTable<>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(pData, pDependentVar));

    /* Retrieve the data from input file */
    trainDataSource.loadDataBlock(mergedData.get());
}
