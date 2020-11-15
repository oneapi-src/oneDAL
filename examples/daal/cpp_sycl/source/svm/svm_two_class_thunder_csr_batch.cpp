/* file: svm_two_class_thunder_csr_batch.cpp */
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
!    C++ example of two-class support vector machine (SVM) classification using
!    the Thunder method with DPC++ interfaces
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_TWO_CLASS_THUNDER_DENSE_BATCH"></a>
 * \example svm_two_class_thunder_dense_batch.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

using daal::data_management::internal::SyclHomogenNumericTable;
using daal::services::internal::SyclExecutionContext;

/* Input data set parameters */
string trainDatasetFileName = "../data/batch/svm_two_class_train_csr.csv";
string trainLabelsFileName  = "../data/batch/svm_two_class_train_labels.csv";

string testDatasetFileName = "../data/batch/svm_two_class_test_csr.csv";
string testLabelsFileName  = "../data/batch/svm_two_class_test_labels.csv";

/* Parameters for the SVM kernel function */
kernel_function::KernelIfacePtr kernel(new kernel_function::linear::Batch<float, kernel_function::linear::fastCSR>());

/* Model object for the SVM algorithm */
svm::training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;

void trainModel();
void testModel();
void printResults();

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    for (const auto & deviceSelector : getListOfDevices())
    {
        const auto & nameDevice = deviceSelector.first;
        const auto & device     = deviceSelector.second;

        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        trainModel();
        testModel();
        printResults();
    }

    return 0;
}

void trainModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data
   * from a .csv file */
    FileDataSource<CSVFeatureManager> trainLabelsDataSource(trainLabelsFileName, DataSource::doAllocateNumericTable,
                                                            DataSource::doDictionaryFromContext);

    /* Create numeric table for training data */
    auto trainData = createSyclSparseTable<float>(trainDatasetFileName);

    /* Retrieve the data from the input file */
    trainLabelsDataSource.loadDataBlock();

    svm::training::Batch<float, svm::training::thunder> algorithm;
    algorithm.parameter.kernel            = kernel;
    algorithm.parameter.C                 = 1.0;
    algorithm.parameter.accuracyThreshold = 0.001;
    algorithm.parameter.tau               = 1e-6;
    algorithm.parameter.maxIterations     = 100;

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
    auto testData = createSyclSparseTable<float>(testDatasetFileName);

    /* Create an algorithm object to predict SVM values */
    svm::prediction::Batch<> algorithm;
    algorithm.parameter.kernel = kernel;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model, trainingResult->get(classifier::training::model));

    /* Predict SVM values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from
   * a .csv file */
    FileDataSource<CSVFeatureManager> testLabelsDataSource(testLabelsFileName, DataSource::doAllocateNumericTable,
                                                           DataSource::doDictionaryFromContext);
    /* Retrieve the data from input file */
    testLabelsDataSource.loadDataBlock();

    NumericTablePtr testGroundTruth = testLabelsDataSource.getNumericTable();
    printNumericTables<int, float>(testGroundTruth, predictionResult->get(classifier::prediction::prediction), "Ground truth\t",
                                   "Classification results", "SVM classification results (first 20 observations):", 20);
}
