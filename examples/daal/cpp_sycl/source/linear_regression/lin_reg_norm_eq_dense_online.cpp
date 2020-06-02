/* file: lin_reg_norm_eq_dense_online.cpp */
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

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms::linear_regression;

/* Input data set parameters */
string trainDatasetFileName = "../data/online/linear_regression_train.csv";
string testDatasetFileName  = "../data/online/linear_regression_test.csv";

const size_t nVectorsInBlock = 250;

const size_t nFeatures           = 10; /* Number of features in training and testing data sets */
const size_t nDependentVariables = 2;  /* Number of dependent variables that correspond to each observation */

training::ResultPtr trainModel();
void testModel(const training::ResultPtr & trainingResult);

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);
    for (const auto & deviceSelector : getListOfDevices())
    {
        const auto & nameDevice = deviceSelector.first;
        const auto & device     = deviceSelector.second;
        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        services::SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        training::ResultPtr trainingResult = trainModel();
        testModel(trainingResult);
    }

    return 0;
}

training::ResultPtr trainModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(trainDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and dependent variables */
    NumericTablePtr trainData         = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::notAllocate);
    NumericTablePtr trainDependentVar = SyclHomogenNumericTable<>::create(nDependentVariables, 0, NumericTable::notAllocate);
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainDependentVar));

    /* Create an algorithm object to train the multiple linear regression model */
    training::Online<> algorithm;

    while (dataSource.loadDataBlock(nVectorsInBlock, mergedData.get()) == nVectorsInBlock)
    {
        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(training::data, trainData);
        algorithm.input.set(training::dependentVariables, trainDependentVar);

        /* Update the multiple linear regression model */
        algorithm.compute();
    }

    /* Finalize the multiple linear regression model */
    algorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    training::ResultPtr trainingResult = algorithm.getResult();
    printNumericTable(trainingResult->get(training::model)->getBeta(), "Linear Regression coefficients:");

    return trainingResult;
}

void testModel(const training::ResultPtr & trainingResult)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(testDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and dependent variables */
    NumericTablePtr testData        = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::notAllocate);
    NumericTablePtr testGroundTruth = SyclHomogenNumericTable<>::create(nDependentVariables, 0, NumericTable::notAllocate);
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from input file */
    dataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict values of multiple linear regression */
    prediction::Batch<> algorithm;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(prediction::model, trainingResult->get(training::model));
    algorithm.input.set(prediction::data, testData);

    /* Predict values of multiple linear regression */
    algorithm.compute();

    /* Retrieve the algorithm results */
    prediction::ResultPtr predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(prediction::prediction), "Linear Regression prediction results: (first 10 rows):", 10);
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10);
}
