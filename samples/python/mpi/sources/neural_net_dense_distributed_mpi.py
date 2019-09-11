# file: neural_net_dense_distributed_mpi.py
#===============================================================================
# Copyright 2017-2019 Intel Corporation.
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
#
# License:
# http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
# eement/
#===============================================================================

#
# !  Content:
# !    Python example of neural network training and scoring in the distributed
# !    processing mode
# !*****************************************************************************

#
## <a name"DAAL-SAMPLE-PY-NEURAL_NET_DENSE_DISTRIBUTED"></a>
## \example neural_net_dense_distributed_mpi.py
#

import os
import sys
from os.path import join as jp

import numpy as np
from mpi4py import MPI

from daal import step1Local, step2Master
from daal.algorithms import optimization_solver
from daal.algorithms.neural_networks import initializers
from daal.algorithms.neural_networks import layers
from daal.algorithms.neural_networks.layers import loss
from daal.algorithms.neural_networks import training, prediction
from daal.data_management import (
    OutputDataArchive, InputDataArchive, HomogenNumericTable, NumericTableIface,
    SubtensorDescriptor, readOnly, HomogenTensor
)

utils_folder = os.path.realpath(os.path.abspath(jp(os.environ['DAALROOT'], 'examples', 'python', 'source')))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensors, readTensorFromCSV


# Input data set parameters
trainDatasetFileNames = [
    "./data/distributed/neural_network_train_dense_1.csv",
    "./data/distributed/neural_network_train_dense_2.csv",
    "./data/distributed/neural_network_train_dense_3.csv",
    "./data/distributed/neural_network_train_dense_4.csv"
]

trainGroundTruthFileNames = [
    "./data/distributed/neural_network_train_ground_truth_1.csv",
    "./data/distributed/neural_network_train_ground_truth_2.csv",
    "./data/distributed/neural_network_train_ground_truth_3.csv",
    "./data/distributed/neural_network_train_ground_truth_4.csv"
]

testDatasetFile = "./data/distributed/neural_network_test.csv"
testGroundTruthFile = "./data/distributed/neural_network_test_ground_truth.csv"

nNodes = 4
batchSize = 100
batchSizeLocal = int(batchSize / nNodes)

MPI_ROOT = 0


def getNextSubtensor(inputTensor, startPos, nElements):
    dims = inputTensor.getDimensions()
    dims[0] = nElements

    subtensorBlock = SubtensorDescriptor(ntype=np.float32)
    inputTensor.getSubtensor([], startPos, nElements, readOnly, subtensorBlock)
    subtensorData = np.array(subtensorBlock.getArray(), copy=True, dtype=np.float32)
    inputTensor.releaseSubtensor(subtensorBlock)

    return HomogenTensor(subtensorData, ntype=np.float32)


def configureNet():

    # Create layers of the neural network
    # Create fully-connected layer and initialize layer parameters
    fullyConnectedLayer1 = layers.fullyconnected.Batch(20)

    fullyConnectedLayer1.parameter.weightsInitializer = initializers.uniform.Batch(-0.001, 0.001)

    fullyConnectedLayer1.parameter.biasesInitializer = initializers.uniform.Batch(0, 0.5)

    # Create fully-connected layer and initialize layer parameters
    fullyConnectedLayer2 = layers.fullyconnected.Batch(40)

    fullyConnectedLayer2.parameter.weightsInitializer = initializers.uniform.Batch(0.5, 1)

    fullyConnectedLayer2.parameter.biasesInitializer = initializers.uniform.Batch(0.5, 1)

    # Create fully-connected layer and initialize layer parameters
    fullyConnectedLayer3 = layers.fullyconnected.Batch(2)

    fullyConnectedLayer3.parameter.weightsInitializer = initializers.uniform.Batch(-0.005, 0.005)

    fullyConnectedLayer3.parameter.biasesInitializer = initializers.uniform.Batch(0, 1)

    # Create softmax layer and initialize layer parameters
    softmaxCrossEntropyLayer = loss.softmax_cross.Batch()

    # Create topology of the neural network
    topology = training.Topology()

    # Add layers to the topology of the neural network
    fc1 = topology.add(fullyConnectedLayer1)
    fc2 = topology.add(fullyConnectedLayer2)
    fc3 = topology.add(fullyConnectedLayer3)
    sm = topology.add(softmaxCrossEntropyLayer)
    topology.get(fc1).addNext(fc2)
    topology.get(fc2).addNext(fc3)
    topology.get(fc3).addNext(sm)

    return topology


def initializeNetwork():

    # Read training data set from a .csv file and create tensors to store input data
    trainingData = readTensorFromCSV(trainDatasetFileNames[rankId])
    trainingGroundTruth = readTensorFromCSV(trainGroundTruthFileNames[rankId], True)

    # Create AdaGrad optimization solver algorithm
    solver = optimization_solver.adagrad.Batch(ntpye=np.float32)

    # Set learning rate for the optimization solver used in the neural network
    learningRate = 0.001
    solver.parameter.learningRate = HomogenNumericTable(1, 1, NumericTableIface.doAllocate, learningRate)
    solver.parameter.batchSize = batchSizeLocal
    solver.parameter.optionalResultRequired = True
    trainingModel = None

    # Algorithms to train neural network
    netLocal = training.Distributed(step1Local)
    netMaster = training.Distributed(step2Master, solver)

    sampleSize = trainingData.getDimensions()
    sampleSize[0] = batchSizeLocal

    # Configure the neural network topology
    topology = configureNet()

    if rankId == MPI_ROOT:

        # Set the optimization solver for the neural network training
        netMaster.parameter.optimizationSolver = solver

        # Initialize the neural network on master node
        netMaster.initialize(sampleSize, topology)

        trainingModel = netMaster.getResult().get(training.model)
    else:
        # Configure the neural network on local nodes
        trainingModel = training.Model()
        trainingModel.initialize_Float32(sampleSize, topology)

    # Pass a model from master node to the algorithms on local nodes
    netLocal.input.setStep1LocalInput(training.inputModel, trainingModel)

    return (trainingData, trainingGroundTruth, netLocal, netMaster)


def trainModel(trainingData, trainingGroundTruth, netLocal, netMaster):

    predictionModel = None
    partialResultsArchLength = 0
    partialResultLocalBuffer = np.array([], dtype=np.uint8)
    partialResultMasterBuffer = np.array([], dtype=np.uint8)

    # Run the neural network training
    nSamples = trainingData.getDimensionSize(0)
    for i in range(0, nSamples - batchSizeLocal + 1, batchSizeLocal):
        # Compute weights and biases for the batch of inputs on local nodes
        # Pass a training data set and dependent values to the algorithm
        netLocal.input.setInput(training.data, getNextSubtensor(trainingData, i, batchSizeLocal))
        netLocal.input.setInput(training.groundTruth, getNextSubtensor(trainingGroundTruth, i, batchSizeLocal))

        # Compute weights and biases derivatives on local node
        pres = netLocal.compute()

        partialResults = [0] * nNodes

        gatherPartialResultsFromNodes(pres, partialResults, partialResultsArchLength, partialResultLocalBuffer, partialResultMasterBuffer)

        wb = HomogenNumericTable()
        if rankId == MPI_ROOT:
            for node in range(nNodes):
                # Pass computed weights and biases derivatives to the master algorithm
                netMaster.input.add(training.partialResults, node, partialResults[node])

            # Update weights and biases on master node
            pres = netMaster.compute()
            wbModel = pres.get(training.resultFromMaster).get(training.model)
            wb = wbModel.getWeightsAndBiases()

        # Broadcast updated weights and biases to nodes
        wbLocal = broadcastWeightsAndBiasesToNodes(wb)
        netLocal.input.getStep1LocalInput(training.inputModel).setWeightsAndBiases(wbLocal)

    if rankId == MPI_ROOT:
        # Finalize neural network training on the master node
        res = netMaster.finalizeCompute()

        # Retrieve training and prediction models of the neural network
        trModel = res.get(training.model)
        predictionModel = trModel.getPredictionModel_Float32()

    return predictionModel


def testModel(predictionModel):

    # Read testing data set from a .csv file and create a tensor to store input data
    predictionData = readTensorFromCSV(testDatasetFile)

    # Create an algorithm to compute the neural network predictions
    net = prediction.Batch()

    # Set the batch size for the neural network prediction
    net.parameter.batchSize = predictionData.getDimensionSize(0)

    # Set input objects for the prediction neural network
    net.input.setModelInput(prediction.model, predictionModel)
    net.input.setTensorInput(prediction.data, predictionData)

    # Run the neural network prediction
    return net.compute()


def printResults(predictionResult):

    # Read testing ground truth from a .csv file and create a tensor to store the data
    predictionGroundTruth = readTensorFromCSV(testGroundTruthFile)
    printTensors(predictionGroundTruth, predictionResult.getResult(prediction.prediction),
                 "Ground truth", "Neural network predictions: each class probability",
                 "Neural network classification results (first 20 observations):", 20)


def gatherPartialResultsFromNodes(partialResult, partialResults, partialResultArchLength,
                                  partialResultLocalBuffer, partialResultMasterBuffer):

    dataArch = InputDataArchive()
    partialResult.serialize(dataArch)
    if partialResultArchLength == 0:
        partialResultArchLength = dataArch.getSizeOfArchive()

    # Serialized data is of equal size on each node
    if rankId == MPI_ROOT and len(partialResultMasterBuffer) == 0:
        partialResultMasterBuffer = np.zeros(partialResultArchLength * nNodes, dtype=np.uint8)

    if len(partialResultLocalBuffer) == 0:
        partialResultLocalBuffer = np.zeros(partialResultArchLength, dtype=np.uint8)

    dataArch.copyArchiveToArray(partialResultLocalBuffer)

    # Transfer partial results to step 2 on the root node
    partialResultMasterBuffer = comm.gather(partialResultLocalBuffer)

    if rankId == MPI_ROOT:
        for node in range(nNodes):
            # Deserialize partial results from step 1
            dataArch = OutputDataArchive(partialResultMasterBuffer[node])

            partialResults[node] = training.PartialResult()
            partialResults[node].deserialize(dataArch)


def broadcastWeightsAndBiasesToNodes(wb):

    wbBuffer = None
    # Serialize weights and biases on the root node
    if rankId == MPI_ROOT:
        if not wb:
            # Weights and biases table should be valid and not NULL on master
            return HomogenNumericTable()

        wbDataArch = InputDataArchive()
        wb.serialize(wbDataArch)
        wbBuffer = np.zeros(wbDataArch.getSizeOfArchive(), dtype=np.uint8)
        wbDataArch.copyArchiveToArray(wbBuffer)

    # Broadcast the serialized weights and biases
    wbBuffer = comm.bcast(wbBuffer)

    # Deserialize weights and biases
    wbDataArchLocal = OutputDataArchive(wbBuffer)

    wbLocal = HomogenNumericTable(ntype=np.float32)
    wbLocal.deserialize(wbDataArchLocal)

    return wbLocal


def main():
    init = initializeNetwork()
    predictionModel = trainModel(*init)

    if rankId == MPI_ROOT:
        predictionResult = testModel(predictionModel)
        printResults(predictionResult)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rankId = comm.Get_rank()
    main()
