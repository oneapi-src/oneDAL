/* file: neural_net_dense_batch.cpp */
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
!    C++ example of neural network training and scoring
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-NEURAL_NETWORK_BATCH"></a>
 * \example neural_net_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"
#include "neural_net_dense_batch.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::services;

/* Input data set parameters */
string trainDatasetFile     = "../data/batch/neural_network_train.csv";
string trainGroundTruthFile = "../data/batch/neural_network_train_ground_truth.csv";
string testDatasetFile      = "../data/batch/neural_network_test.csv";
string testGroundTruthFile  = "../data/batch/neural_network_test_ground_truth.csv";

prediction::ModelPtr predictionModel;
prediction::ResultPtr predictionResult;

void trainModel();
void testModel();
void printResults();

int main()
{
    trainModel();

    testModel();

    printResults();

    return 0;
}

void trainModel()
{
    /* Read training data set from a .csv file and create a tensor to store input data */
    TensorPtr trainingData = readTensorFromCSV(trainDatasetFile);
    TensorPtr trainingGroundTruth = readTensorFromCSV(trainGroundTruthFile);

    /* Create an algorithm to train neural network */
    training::Batch<> net;

    /* Set the batch size for the neural network training */
    net.parameter.batchSize = 10;

    /* Configure the neural network */
    training::TopologyPtr topology = configureNet();
    net.initialize(trainingData->getDimensions(), *topology);

    /* Pass a training data set and dependent values to the algorithm */
    net.input.set(training::data, trainingData);
    net.input.set(training::groundTruth, trainingGroundTruth);

    /* Create stochastic gradient descent (SGD) optimization solver algorithm */
    SharedPtr<optimization_solver::sgd::Batch<float> > sgdAlgorithm(new optimization_solver::sgd::Batch<float>());

    /* Set learning rate for the optimization solver used in the neural network */

    float learningRate = 0.001f;
    sgdAlgorithm->parameter.learningRateSequence = NumericTablePtr(new HomogenNumericTable<double>(1, 1, NumericTable::doAllocate, learningRate));

    /* Set the optimization solver for the neural network training */
    net.parameter.optimizationSolver = sgdAlgorithm;

    /* Run the neural network training */
    net.compute();

    /* Retrieve training and prediction models of the neural network */
    training::ModelPtr trainingModel = net.getResult()->get(training::model);
    predictionModel = trainingModel->getPredictionModel<float>();
}

void testModel()
{
    /* Read testing data set from a .csv file and create a tensor to store input data */
    TensorPtr predictionData = readTensorFromCSV(testDatasetFile);

    /* Create an algorithm to compute the neural network predictions */
    prediction::Batch<> net;

    /* Set input objects for the prediction neural network */
    net.input.set(prediction::model, predictionModel);
    net.input.set(prediction::data, predictionData);

    /* Run the neural network prediction */
    net.compute();

    /* Print results of the neural network prediction */
    predictionResult = net.getResult();
}

void printResults()
{
    /* Read testing ground truth from a .csv file and create a tensor to store the data */
    TensorPtr predictionGroundTruth = readTensorFromCSV(testGroundTruthFile);

    printTensors<int, float>(predictionGroundTruth, predictionResult->get(prediction::prediction),
                             "Ground truth", "Neural network predictions: each class probability",
                             "Neural network classification results (first 20 observations):", 20);
}
