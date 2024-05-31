/* file: svm_two_class_thunder_dense_batch.cpp */
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
!    the Thunder method
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_TWO_CLASS_THUNDER_DENSE_BATCH"></a>
 * \example svm_two_class_thunder_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"
#include <iostream>
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
std::string trainDatasetFileName = "../data/batch/svm_two_class_train_dense.csv";
std::string testDatasetFileName = "../data/batch/svm_two_class_test_dense.csv";

const size_t nFeatures = 20;

/* Model object for the SVM algorithm */
svm::training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;
NumericTablePtr testGroundTruth;

void trainModel();
void testModel();
void printResults();

int main(int argc, char* argv[]) {
    std::cout << "here 0" << std::endl;
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);
    std::cout << "here 1" << std::endl;
    trainModel();
    std::cout << "here 2" << std::endl;
    testModel();
    std::cout << "here 3" << std::endl;
    printResults();
    std::cout << "here 4" << std::endl;
    return 0;
}

void trainModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);
    std::cout << "here 11" << std::endl;
    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr trainGroundTruth(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));
    std::cout << "here 12" << std::endl;
    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());
    std::cout << "here 13" << std::endl;
    /* Create an algorithm object to train the SVM model using the Thunder method */
    svm::training::Batch<float, svm::training::thunder> algorithm;
    std::cout << "here 14" << std::endl;
    /* Parameters for the SVM kernel function */
    kernel_function::KernelIfacePtr kernel(new kernel_function::linear::Batch<>());
    algorithm.parameter.kernel = kernel;
    std::cout << "here 15" << std::endl;
    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);
    std::cout << "here 16" << std::endl;
    /* Build the SVM model */
    algorithm.compute();
    std::cout << "here 17" << std::endl;
    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::notAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);
    std::cout << "here 18" << std::endl;
    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    testGroundTruth = NumericTablePtr(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));
    std::cout << "here 19" << std::endl;
    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());
    std::cout << "here 20" << std::endl;
    /* Create an algorithm object to predict SVM values */
    svm::prediction::Batch<> algorithm;
    std::cout << "here 21" << std::endl;
    /* Parameters for the SVM kernel function */
    kernel_function::KernelIfacePtr kernel(new kernel_function::linear::Batch<>());
    algorithm.parameter.kernel = kernel;
    std::cout << "here 22" << std::endl;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));
    std::cout << "here 23" << std::endl;
    /* Predict SVM values */
    algorithm.compute();
    std::cout << "here 24" << std::endl;
    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
    std::cout << "here 25" << std::endl;
}

void printResults() {
    printNumericTables<int, float>(testGroundTruth,
                                   predictionResult->get(classifier::prediction::prediction),
                                   "Ground truth\t",
                                   "Classification results",
                                   "SVM classification results (first 20 observations):",
                                   20);
}
