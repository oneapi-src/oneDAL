# file: neural_net_predict_dense_batch.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

#
# !  Content:
# !    Python example of neural network scoring
# !*****************************************************************************

#
##  <a name="DAAL-EXAMPLE-PY-NEURAL_NET_PREDICTION_DENSE_BATCH"></a>
##  \example  neural_net_predict_dense_batch.py
#

import os
import sys

from daal.algorithms.neural_networks import layers
from daal.algorithms.neural_networks import prediction

import daal.algorithms.neural_networks.layers.fullyconnected.forward
import daal.algorithms.neural_networks.layers.softmax.forward

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensors, readTensorFromCSV

# Input data set parameters
testDatasetFile = os.path.join("..", "data", "batch", "neural_network_test.csv")
testGroundTruthFile = os.path.join("..", "data", "batch", "neural_network_test_ground_truth.csv")

# Weights and biases obtained on the training stage
fc1WeightsFile = os.path.join("..", "data", "batch", "fc1_weights.csv")
fc1BiasesFile = os.path.join("..", "data", "batch", "fc1_biases.csv")
fc2WeightsFile = os.path.join("..", "data", "batch", "fc2_weights.csv")
fc2BiasesFile = os.path.join("..", "data", "batch", "fc2_biases.csv")

fc1 = 0
fc2 = 1
sm1 = 2


def configureNet():
    # Create layers of the neural network
    # Create first fully-connected layer
    fullyConnectedLayer1 = layers.fullyconnected.forward.Batch(5)

    # Create second fully-connected layer
    fullyConnectedLayer2 = layers.fullyconnected.forward.Batch(2)

    # Create softmax layer
    softmaxLayer = layers.softmax.forward.Batch()

    # Create topology of the neural network
    topology = prediction.Topology()

    # Add layers to the topology of the neural network
    topology.push_back(fullyConnectedLayer1)
    topology.push_back(fullyConnectedLayer2)
    topology.push_back(softmaxLayer)
    topology.get(fc1).addNext(fc2)
    topology.get(fc2).addNext(sm1)
    return topology


def createModel():
    # Read testing data set from a .csv file and create a tensor to store input data
    predictionData = readTensorFromCSV(testDatasetFile)

    # Configure the neural network
    topology = configureNet()

    # Create prediction model of the neural network
    predictionModel = prediction.Model(topology)

    # Read 1st fully-connected layer weights and biases from CSV file
    # 1st fully-connected layer weights are a 2D tensor of size 5 x 20
    fc1Weights = readTensorFromCSV(fc1WeightsFile)
    # 1st fully-connected layer biases are a 1D tensor of size 5
    fc1Biases = readTensorFromCSV(fc1BiasesFile)

    # Set weights and biases of the 1st fully-connected layer
    fc1Input = predictionModel.getLayer(fc1).getLayerInput()
    fc1Input.setInput(layers.forward.weights, fc1Weights)
    fc1Input.setInput(layers.forward.biases, fc1Biases)

    # Set flag that specifies that weights and biases of the 1st fully-connected layer are initialized
    fc1Parameter = predictionModel.getLayer(fc1).getLayerParameter()
    fc1Parameter.weightsAndBiasesInitialized = True

    # Read 2nd fully-connected layer weights and biases from CSV file
    # 2nd fully-connected layer weights are a 2D tensor of size 2 x 5
    fc2Weights = readTensorFromCSV(fc2WeightsFile)
    # 2nd fully-connected layer biases are a 1D tensor of size 2
    fc2Biases = readTensorFromCSV(fc2BiasesFile)

    # Set weights and biases of the 2nd fully-connected layer
    fc2Input = predictionModel.getLayer(fc2).getLayerInput()
    fc2Input.setInput(layers.forward.weights, fc2Weights)
    fc2Input.setInput(layers.forward.biases, fc2Biases)

    # Set flag that specifies that weights and biases of the 2nd fully-connected layer are initialized
    fc2Parameter = predictionModel.getLayer(fc2).getLayerParameter()
    fc2Parameter.weightsAndBiasesInitialized = True

    return (predictionData, predictionModel)


def testModel(predictionData, predictionModel):
    # Create an algorithm to compute the neural network predictions
    net = prediction.Batch()

    net.parameter.batchSize = predictionData.getDimensionSize(0)

    # Set input objects for the prediction neural network
    net.input.setModelInput(prediction.model, predictionModel)
    net.input.setTensorInput(prediction.data, predictionData)

    # Run the neural network prediction and
    # get results of the neural network prediction
    return net.compute()


def printResults(predictionResult):
    # Read testing ground truth from a .csv file and create a tensor to store the data
    predictionGroundTruth = readTensorFromCSV(testGroundTruthFile)
    printTensors(predictionGroundTruth, predictionResult.getResult(prediction.prediction),
                 "Ground truth", "Neural network predictions: each class probability",
                 "Neural network classification results (first 20 observations):", 20)


if __name__ == "__main__":
    (predictionData, predictionModel) = createModel()

    predictionResult = testModel(predictionData, predictionModel)

    printResults(predictionResult)
