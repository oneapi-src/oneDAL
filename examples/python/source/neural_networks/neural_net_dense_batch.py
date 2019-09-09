# file: neural_net_dense_batch.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

#
# !  Content:
# !    Python example of neural network training and scoring
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-NEURAL_NET_DENSE_BATCH"></a>
## \example neural_net_dense_batch.py
#

import os
import sys

import numpy as np

from daal.algorithms.neural_networks import initializers
from daal.algorithms.neural_networks import layers
from daal.algorithms import optimization_solver
from daal.algorithms.neural_networks import training, prediction
from daal.data_management import NumericTable, HomogenNumericTable

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensors, readTensorFromCSV

# Input data set parameters
trainDatasetFile = os.path.join("..", "data", "batch", "neural_network_train.csv")
trainGroundTruthFile = os.path.join("..", "data", "batch", "neural_network_train_ground_truth.csv")
testDatasetFile = os.path.join("..", "data", "batch", "neural_network_test.csv")
testGroundTruthFile = os.path.join("..", "data", "batch", "neural_network_test_ground_truth.csv")

fc1 = 0
fc2 = 1
sm1 = 2

batchSize = 10

def configureNet():
    # Create layers of the neural network
    # Create fully-connected layer and initialize layer parameters
    fullyConnectedLayer1 = layers.fullyconnected.Batch(5)
    fullyConnectedLayer1.parameter.weightsInitializer = initializers.uniform.Batch(-0.001, 0.001)
    fullyConnectedLayer1.parameter.biasesInitializer = initializers.uniform.Batch(0, 0.5)

    # Create fully-connected layer and initialize layer parameters
    fullyConnectedLayer2 = layers.fullyconnected.Batch(2)
    fullyConnectedLayer2.parameter.weightsInitializer = initializers.uniform.Batch(0.5, 1)
    fullyConnectedLayer2.parameter.biasesInitializer = initializers.uniform.Batch(0.5, 1)

    # Create softmax layer and initialize layer parameters
    softmaxCrossEntropyLayer = layers.loss.softmax_cross.Batch()

    # Create configuration of the neural network with layers
    topology = training.Topology()

    # Add layers to the topology of the neural network
    topology.push_back(fullyConnectedLayer1)
    topology.push_back(fullyConnectedLayer2)
    topology.push_back(softmaxCrossEntropyLayer)
    topology.get(fc1).addNext(fc2)
    topology.get(fc2).addNext(sm1)
    return topology


def trainModel():
    # Read training data set from a .csv file and create a tensor to store input data
    trainingData = readTensorFromCSV(trainDatasetFile)
    trainingGroundTruth = readTensorFromCSV(trainGroundTruthFile, True)

    sgdAlgorithm = optimization_solver.sgd.Batch(fptype=np.float32)

    # Set learning rate for the optimization solver used in the neural network
    learningRate = 0.001
    sgdAlgorithm.parameter.learningRateSequence = HomogenNumericTable(1, 1, NumericTable.doAllocate, learningRate)
    # Set the batch size for the neural network training
    sgdAlgorithm.parameter.batchSize = batchSize
    sgdAlgorithm.parameter.nIterations = int(trainingData.getDimensionSize(0) / sgdAlgorithm.parameter.batchSize)

    # Create an algorithm to train neural network
    net = training.Batch(sgdAlgorithm)

    sampleSize = trainingData.getDimensions()
    sampleSize[0] = batchSize

    # Configure the neural network
    topology = configureNet()
    net.initialize(sampleSize, topology)

    # Pass a training data set and dependent values to the algorithm
    net.input.setInput(training.data, trainingData)
    net.input.setInput(training.groundTruth, trainingGroundTruth)

    # Run the neural network training and retrieve training model
    trainingModel = net.compute().get(training.model)
    # return prediction model
    return trainingModel.getPredictionModel_Float32()


def testModel(predictionModel):
    # Read testing data set from a .csv file and create a tensor to store input data
    predictionData = readTensorFromCSV(testDatasetFile)

    # Create an algorithm to compute the neural network predictions
    net = prediction.Batch()

    net.parameter.batchSize = predictionData.getDimensionSize(0)

    # Set input objects for the prediction neural network
    net.input.setModelInput(prediction.model, predictionModel)
    net.input.setTensorInput(prediction.data, predictionData)

    # Run the neural network prediction
    # and return results of the neural network prediction
    return net.compute()


def printResults(predictionResult):
    # Read testing ground truth from a .csv file and create a tensor to store the data
    predictionGroundTruth = readTensorFromCSV(testGroundTruthFile)

    printTensors(predictionGroundTruth, predictionResult.getResult(prediction.prediction),
                 "Ground truth", "Neural network predictions: each class probability",
                 "Neural network classification results (first 20 observations):", 20)


topology = ""
if __name__ == "__main__":

    predictionModel = trainModel()

    predictionResult = testModel(predictionModel)

    printResults(predictionResult)
