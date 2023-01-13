/* file: df_cls_dense_batch.cpp */
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
!    C++ example of decision forest classification in the batch processing mode.
!
!    The program trains the decision forest classification model on a training
!    datasetFileName and computes classification for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DF_CLS_DENSE_BATCH"></a>
 * \example df_cls_dense_batch.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::decision_forest::classification;

using daal::services::internal::SyclExecutionContext;
using daal::data_management::internal::SyclHomogenNumericTable;

/* Input data set parameters */
const std::string trainDatasetFileName = "../data/batch/df_classification_train.csv";
const std::string testDatasetFileName = "../data/batch/df_classification_test.csv";
const size_t nFeatures = 3; /* Number of features in training and testing data sets */

/* Decision forest parameters */
const size_t nTrees = 10;
const size_t minObservationsInLeafNode = 8;

const size_t nClasses = 5; /* Number of classes */

template <typename algorithmType>
training::ResultPtr trainModel(algorithmType&& algorithm);
void testModel(const training::ResultPtr& res);
void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar);

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    for (const auto& deviceSelector : getListOfDevices()) {
        const auto& nameDevice = deviceSelector.first;
        const auto& device = deviceSelector.second;
        sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        /* Create an algorithm object to train the decision forest classification model */
        training::ResultPtr trainingResult =
            trainModel(training::Batch<float, training::hist>(nClasses));

        testModel(trainingResult);
    }
    return 0;
}

template <typename algorithmType>
training::ResultPtr trainModel(algorithmType&& algorithm) {
    /* Create Numeric Tables for training data and dependent variables */
    NumericTablePtr trainData;
    NumericTablePtr trainDependentVariable;

    loadData(trainDatasetFileName, trainData, trainDependentVariable);

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainDependentVariable);

    algorithm.parameter().nTrees = nTrees;
    algorithm.parameter().featuresPerNode = nFeatures;
    algorithm.parameter().minObservationsInLeafNode = minObservationsInLeafNode;
    algorithm.parameter().varImportance = algorithms::decision_forest::training::MDI;
    algorithm.parameter().resultsToCompute =
        algorithms::decision_forest::training::computeOutOfBagError;

    /* Build the decision forest classification model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    training::ResultPtr trainingResult = algorithm.getResult();
    printNumericTable(trainingResult->get(training::variableImportance),
                      "Variable importance results: ");
    printNumericTable(trainingResult->get(training::outOfBagError), "OOB error: ");
    return trainingResult;
}

void testModel(const training::ResultPtr& trainingResult) {
    /* Create Numeric Tables for testing data and ground truth values */
    NumericTablePtr testData;
    NumericTablePtr testGroundTruth;

    loadData(testDatasetFileName, testData, testGroundTruth);

    /* Create an algorithm object to predict values of decision forest classification */
    prediction::Batch<> algorithm(nClasses);

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));
    algorithm.parameter().votingMethod = prediction::weighted;
    algorithm.parameter().resultsToEvaluate |=
        static_cast<DAAL_UINT64>(classifier::computeClassProbabilities);
    /* Predict values of decision forest classification */
    algorithm.compute();

    /* Retrieve the algorithm results */
    classifier::prediction::ResultPtr predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(classifier::prediction::prediction),
                      "Decision forest prediction results (first 10 rows):",
                      10);
    printNumericTable(predictionResult->get(classifier::prediction::probabilities),
                      "Decision forest probabilities results (first 10 rows):",
                      10);
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10);
}

void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar) {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(fileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and dependent variables */
    pData = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::notAllocate);
    pDependentVar = SyclHomogenNumericTable<>::create(1, 0, NumericTable::notAllocate);
    NumericTablePtr mergedData(new MergedNumericTable(pData, pDependentVar));

    /* Retrieve the data from input file */
    trainDataSource.loadDataBlock(mergedData.get());
}
