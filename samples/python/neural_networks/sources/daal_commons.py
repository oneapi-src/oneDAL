# file: daal_commons.py
#===============================================================================
# Copyright 2017-2019 Intel Corporation
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
# !    Common functions for traininig and testing neural networks
# !*****************************************************************************

import numpy as np

from daal.algorithms import optimization_solver
from daal.algorithms.neural_networks import training, prediction
from daal.data_management import HomogenNumericTable, NumericTableIface

from service import ClassificationErrorCounter

batchSize = 1
trainingIterations = 1


# Trains neural network with given dataset reader
def trainClassifier(topology, reader):

    print("Training started with batch size = [{}]".format(batchSize))

    # Get collection of last layer indices from topology
    lastLayerIndices = getLastLayersIndices(topology)

    # Create the neural network training algorithm and set batch size and optimization solver
    net = training.Batch(getDefaultOptimizationSolver(), ntype=np.float32)
    net.parameter.optimizationSolver.getParameter().nIterations = int(trainingIterations / reader.getTotalNumberOfObjects())

    # Initialize neural network with given topology
    net.initialize(reader.getBatchDimensions(), topology)

    batchCounter = 0
    for i in range(trainingIterations):
        # Reset reader's iterator the dataset begining
        reader.reset()

        # Advance dataset reader's iterator to the next batch
        while reader.next():
            batchCounter += 1

            # Set the input data batch to the neural network
            net.input.setInput(training.data, reader.getBatch())

            # Set the input ground truth (labels) batch to the neural network
            setGroundTruthForMultipleOutputs(net.input, lastLayerIndices, reader.getGroundTruthBatch())

            # Compute the neural network forward and backward passes and update
            # weights and biases according to the optimization solver
            trainingResult = net.compute()

            print("{} train batches processed".format(batchCounter))

    # Get prediction model
    trainedModel = trainingResult.get(training.model)
    return trainedModel.getPredictionModel_Float32()


# Tests model with given dataset reader and return top-5 error rate
def testClassifier(predictionModel, reader):
    # Create the neural network prediction algorithm
    net = prediction.Batch(ntyp=np.float32)

    # Set the prediction model retrieved from the training stage
    net.input.setModelInput(prediction.model, predictionModel)

    # Create auxiliary object to compute error rates (defined in services.h)
    errorRateCounter = ClassificationErrorCounter()

    # Reset reader's iterator the dataset begining
    reader.reset()

    batchCounter = 0

    # Advance dataset reader's iterator to the next batch
    while reader.next():
        batchCounter += 1

        # Set the input data batch to the neural network
        net.input.setTensorInput(prediction.data, reader.getBatch())

        # Compute the neural network forward pass
        res = net.compute()

        # Get tensor of predicted probailities for each class and update error rate
        predictionResult = res.getResult(prediction.prediction)
        errorRateCounter.update(predictionResult, reader.getGroundTruthBatch())

        print("{} test batches processed".format(batchCounter))

    return errorRateCounter.getTop5ErrorRate()


def getDefaultOptimizationSolver(learningRate=0.001):
    """Constructs the optimization solver with given learning rate"""

    # Create 1 x 1 NumericTable to store learning rate
    learningRateSequence = HomogenNumericTable(1, 1, NumericTableIface.doAllocate, learningRate, ntype=np.float32)

    # Create SGD optimization solver and set learning rate
    optalg = optimization_solver.sgd.Batch(fptype=np.float32)
    optalg.parameter.learningRateSequence = learningRateSequence
    optalg.parameter.batchSize = batchSize
    return optimization_solver.sgd.Batch(optalg, fptype=np.float32)


def getLastLayersIndices(topology):
    lastLayerIndices = []

    for i in range(topology.size()):
        descriptor = topology[i]
        if descriptor.nextLayers().size() == 0:
            lastLayerIndices.append(descriptor.index())

    return lastLayerIndices


def setGroundTruthForMultipleOutputs(trainNetInput, lastLayerIndices, groundTruth):
    for i in range(len(lastLayerIndices)):
        lastLayerIndex = lastLayerIndices[i]
        trainNetInput.add(training.groundTruthCollection, lastLayerIndex, groundTruth)
