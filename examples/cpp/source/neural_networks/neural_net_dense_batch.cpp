/* file: neural_net_dense_batch.cpp */
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

const size_t batchSize = 10;

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
    TensorPtr trainingGroundTruth = readTensorFromCSV(trainGroundTruthFile, true);

    SharedPtr<optimization_solver::sgd::Batch<> > sgdAlgorithm(new optimization_solver::sgd::Batch<>());
    float learningRate = 0.001f;
    sgdAlgorithm->parameter.learningRateSequence = NumericTablePtr(new HomogenNumericTable<>(1, 1, NumericTable::doAllocate, learningRate));
    /* Set the batch size for the neural network training */
    sgdAlgorithm->parameter.batchSize = batchSize;
    sgdAlgorithm->parameter.nIterations = trainingData->getDimensionSize(0) / sgdAlgorithm->parameter.batchSize;

    /* Create an algorithm to train neural network */
    training::Batch<> net(sgdAlgorithm);

    /* Configure the neural network */
    training::TopologyPtr topology = configureNet();

    services::Collection<size_t> oneBatchDimensions = trainingData->getDimensions();
    oneBatchDimensions[0] = batchSize;
    net.initialize(oneBatchDimensions, *topology);

    /* Pass a training data set and dependent values to the algorithm */
    net.input.set(training::data, trainingData);
    net.input.set(training::groundTruth, trainingGroundTruth);

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

    net.parameter.batchSize = predictionData->getDimensionSize(0);

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
