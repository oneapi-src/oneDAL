/* file: lin_reg_norm_eq_dense_online.cpp */
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
!    C++ example of multiple linear regression in the online processing mode.
!
!    The program trains the multiple linear regression model on a training
!    datasetFileName with the normal equations method and computes regression
!    for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LINEAR_REGRESSION_NORM_EQ_ONLINE"></a>
 * \example lin_reg_norm_eq_dense_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms::linear_regression;

/* Input data set parameters */
string trainDatasetFileName            = "../data/online/linear_regression_train.csv";
string testDatasetFileName             = "../data/online/linear_regression_test.csv";

const size_t nTrainVectorsInBlock = 250;

const size_t nFeatures           = 10;  /* Number of features in training and testing data sets */
const size_t nDependentVariables = 2;   /* Number of dependent variables that correspond to each observation */

void trainModel();
void testModel();

training::ResultPtr trainingResult;
prediction::ResultPtr predictionResult;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    trainModel();
    testModel();

    return 0;
}

void trainModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and dependent variables */
    NumericTablePtr trainData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr trainDependentVariables(new HomogenNumericTable<>(nDependentVariables, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainDependentVariables));

    /* Create an algorithm object to train the multiple linear regression model */
    training::Online<> algorithm;

    while(trainDataSource.loadDataBlock(nTrainVectorsInBlock, mergedData.get()) == nTrainVectorsInBlock)
    {
        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(training::data, trainData);
        algorithm.input.set(training::dependentVariables, trainDependentVariables);

        /* Update the multiple linear regression model */
        algorithm.compute();
    }

    /* Finalize the multiple linear regression model */
    algorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
    printNumericTable(trainingResult->get(training::model)->getBeta(), "Linear Regression coefficients:");
}

void testModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and ground truth values */
    NumericTablePtr testData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr testGroundTruth(new HomogenNumericTable<>(nDependentVariables, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from the input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict values of multiple linear regression */
    prediction::Batch<> algorithm;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(prediction::model, trainingResult->get(training::model));
    algorithm.input.set(prediction::data, testData);

    /* Predict values of multiple linear regression */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(prediction::prediction),
        "Linear Regression prediction results: (first 10 rows):", 10);
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10);
}
