/* file: neural_net_predict_dense_batch.cpp */
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
!    C++ example of neural network scoring
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-NEURAL_NETWORK_PREDICTION_BATCH"></a>
 * \example neural_net_predict_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"
#include "neural_net_predict_dense_batch.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::services;

/* Input data set parameters */
string testDatasetFile      = "../data/batch/neural_network_test.csv";
string testGroundTruthFile  = "../data/batch/neural_network_test_ground_truth.csv";

/* Weights and biases obtained on the training stage */
string fc1WeightsFile = "../data/batch/fc1_weights.csv";
string fc1BiasesFile  = "../data/batch/fc1_biases.csv";
string fc2WeightsFile = "../data/batch/fc2_weights.csv";
string fc2BiasesFile  = "../data/batch/fc2_biases.csv";

TensorPtr predictionData;
prediction::ModelPtr predictionModel;
prediction::ResultPtr predictionResult;

void createModel();
void testModel();
void printResults();

int main()
{
    createModel();

    testModel();

    printResults();

    return 0;
}

void createModel()
{
    /* Read testing data set from a .csv file and create a tensor to store input data */
    predictionData = readTensorFromCSV(testDatasetFile);

    /* Configure the neural network */
    LayerIds ids;
    prediction::TopologyPtr topology = configureNet(&ids);

    /* Create prediction model of the neural network */
    predictionModel = prediction::ModelPtr(new prediction::Model(*topology));

    /* Read 1st fully-connected layer weights and biases from CSV file */
    /* 1st fully-connected layer weights are a 2D tensor of size 5 x 20 */
    TensorPtr fc1Weights = readTensorFromCSV(fc1WeightsFile);
    /* 1st fully-connected layer biases are a 1D tensor of size 5 */
    TensorPtr fc1Biases = readTensorFromCSV(fc1BiasesFile);

    /* Set weights and biases of the 1st fully-connected layer */
    forward::Input *fc1Input = predictionModel->getLayer(ids.fc1)->getLayerInput();
    fc1Input->set(forward::weights, fc1Weights);
    fc1Input->set(forward::biases, fc1Biases);

    /* Read 2nd fully-connected layer weights and biases from CSV file */
    /* 2nd fully-connected layer weights are a 2D tensor of size 2 x 5 */
    TensorPtr fc2Weights = readTensorFromCSV(fc2WeightsFile);
    /* 2nd fully-connected layer biases are a 1D tensor of size 2 */
    TensorPtr fc2Biases = readTensorFromCSV(fc2BiasesFile);

    /* Set weights and biases of the 2nd fully-connected layer */
    forward::Input *fc2Input = predictionModel->getLayer(ids.fc2)->getLayerInput();
    fc2Input->set(forward::weights, fc2Weights);
    fc2Input->set(forward::biases, fc2Biases);

    /* Allocate memory for prediction model of the neural network */
    predictionModel->allocate<float>(predictionData->getDimensions());
}

void testModel()
{
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
