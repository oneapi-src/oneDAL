# file: daal_lenet.py
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
# !    Python sample of LeNet training and prediction.
# !
# !*****************************************************************************

from __future__ import division, print_function

import os
import sys

import numpy as np

from daal.algorithms import optimization_solver
from daal.algorithms.neural_networks import training, prediction
from daal.algorithms.neural_networks import initializers
from daal.algorithms.neural_networks.layers import (
    convolution2d, pooling2d, maximum_pooling2d, fullyconnected, relu, loss
)
from daal.data_management import readOnly, SubtensorDescriptor, HomogenNumericTable

from image_dataset import DatasetReaderMNIST


def printPredictedClasses(_predictionResult, _testingGroundTruth):
    _prediction = _predictionResult.getResult(prediction.prediction)
    predictionDimensions = _prediction.getDimensions()

    predictionBlock = SubtensorDescriptor()
    _prediction.getSubtensor([], 0, predictionDimensions[0], readOnly, predictionBlock)
    predictionPtr = predictionBlock.getArray()

    testGroundTruthBlock = SubtensorDescriptor(ntype=np.intc)
    _testingGroundTruth.getSubtensor([], 0, predictionDimensions[0], readOnly, testGroundTruthBlock)
    testGroundTruthPtr = testGroundTruthBlock.getArray().flatten()

    # Print predicted classes
    maxPIndex = np.argmax(predictionPtr, axis=1)
    for i in range(predictionDimensions[0]):
        for p in predictionPtr[i]:
            print("{:.4f} ".format(p), end="")
        print(" -> {} | {}".format(maxPIndex[i], testGroundTruthPtr[i]))

    _prediction.releaseSubtensor(predictionBlock)
    _testingGroundTruth.releaseSubtensor(testGroundTruthBlock)


def configureNet():
    # Create convolution layer
    convolution1 = convolution2d.Batch()
    convolution1.parameter.kernelSizes = convolution2d.KernelSizes(3, 3)
    convolution1.parameter.strides = convolution2d.Strides(1, 1)
    convolution1.parameter.nKernels = 32
    convolution1.parameter.weightsInitializer = initializers.xavier.Batch()
    convolution1.parameter.biasesInitializer = initializers.uniform.Batch(0, 0)

    # Create pooling layer
    maxpooling1 = maximum_pooling2d.Batch(4)
    maxpooling1.parameter.kernelSizes = pooling2d.KernelSizes(2, 2)
    maxpooling1.parameter.paddings = pooling2d.Paddings(0, 0)
    maxpooling1.parameter.strides = pooling2d.Strides(2, 2)

    # Create convolution layer
    convolution2 = convolution2d.Batch()
    convolution2.parameter.kernelSizes = convolution2d.KernelSizes(5, 5)
    convolution2.parameter.strides = convolution2d.Strides(1, 1)
    convolution2.parameter.nKernels = 64
    convolution2.parameter.weightsInitializer = initializers.xavier.Batch()
    convolution2.parameter.biasesInitializer = initializers.uniform.Batch(0, 0)

    # Create pooling layer
    maxpooling2 = maximum_pooling2d.Batch(4)
    maxpooling2.parameter.kernelSizes = pooling2d.KernelSizes(2, 2)
    maxpooling2.parameter.paddings = pooling2d.Paddings(0, 0)
    maxpooling2.parameter.strides = pooling2d.Strides(2, 2)

    # Create fullyconnected layer
    fullyconnected3 = fullyconnected.Batch(256)
    fullyconnected3.parameter.weightsInitializer = initializers.xavier.Batch()
    fullyconnected3.parameter.biasesInitializer = initializers.uniform.Batch(0, 0)

    # Create ReLU layer
    relu3 = relu.Batch()

    # Create fully connected layer
    fullyconnected4 = fullyconnected.Batch(10)
    fullyconnected4.parameter.weightsInitializer = initializers.xavier.Batch()
    fullyconnected4.parameter.biasesInitializer = initializers.uniform.Batch(0, 0)

    # Create Softmax layer
    softmax = loss.softmax_cross.Batch()

    # Create LeNet Topology
    topology = training.Topology()
    conv1 = topology.add(convolution1)
    pool1 = topology.add(maxpooling1)
    topology.get(conv1).addNext(pool1)
    conv2 = topology.add(convolution2)
    topology.get(pool1).addNext(conv2)
    pool2 = topology.add(maxpooling2)
    topology.get(conv2).addNext(pool2)
    fc3 = topology.add(fullyconnected3)
    topology.get(pool2).addNext(fc3)
    r3 = topology.add(relu3)
    topology.get(fc3).addNext(r3)
    fc4 = topology.add(fullyconnected4)
    topology.get(r3).addNext(fc4)
    sm1 = topology.add(softmax)
    topology.get(fc4).addNext(sm1)

    return topology


# LeNet training
def train(trainingData, trainingGroundTruth):
    batchSize = 10
    learningRate = 0.01

    sgdAlgorithm = optimization_solver.sgd.Batch(fptype=np.float32)
    arr = np.array([[learningRate]], dtype=np.float32)
    sgdAlgorithm.parameter.learningRateSequence = HomogenNumericTable(arr, ntype=np.float32)
    sgdAlgorithm.parameter.batchSize = batchSize
    sgdAlgorithm.parameter.nIterations = int(trainingData.getDimensionSize(0) / sgdAlgorithm.parameter.batchSize)

    topology = configureNet()

    net = training.Batch(sgdAlgorithm)

    sampleSize = trainingData.getDimensions()
    sampleSize[0] = batchSize
    net.initialize(sampleSize, topology)

    net.input.setInput(training.data, trainingData)
    net.input.setInput(training.groundTruth, trainingGroundTruth)

    res = net.compute()

    return res.get(training.model).getPredictionModel_Float64()


# LeNet testing
def test(predictionModel, testingData, testingGroundTruth):
    net = prediction.Batch()

    net.input.setModelInput(prediction.model, predictionModel)
    net.input.setTensorInput(prediction.data, testingData)

    predictionResult = net.compute()

    printPredictedClasses(predictionResult, testingGroundTruth)

    return predictionResult


# check prediction results
def checkResult(predictionResult, testingGroundTruth, TestDataCount):
    pred = predictionResult.getResult(prediction.prediction)
    predictionDimensions = pred.getDimensions()

    predictionBlock = SubtensorDescriptor()
    pred.getSubtensor([], 0, predictionDimensions[0], readOnly, predictionBlock)
    predictionPtr = predictionBlock.getArray()

    testGroundTruthBlock = SubtensorDescriptor(ntype=np.intc)
    testingGroundTruth.getSubtensor([], 0, predictionDimensions[0], readOnly, testGroundTruthBlock)
    testGroundTruthPtr = testGroundTruthBlock.getArray().flatten()
    maxPIndex = 0
    trueCount = 0

    # validation accuracy finding
    maxPIndex = np.argmax(predictionPtr, axis=1)
    trueCount = np.sum(maxPIndex == testGroundTruthPtr)

    pred.releaseSubtensor(predictionBlock)
    testingGroundTruth.releaseSubtensor(testGroundTruthBlock)

    return True if trueCount / TestDataCount > 0.9 else False


def main():
    TrainDataCount = 50000
    TestDataCount = 100

    datasetFileNames = [
        os.path.join('data', 'train-images-idx3-ubyte'),
        os.path.join('data', 'train-labels-idx1-ubyte'),
        os.path.join('data', 't10k-images-idx3-ubyte'),
        os.path.join('data', 't10k-labels-idx1-ubyte')
    ]

    print("Data loading started... ")

    reader = DatasetReaderMNIST()
    reader.setTrainBatch(datasetFileNames[0], datasetFileNames[1], TrainDataCount)
    reader.setTestBatch(datasetFileNames[2], datasetFileNames[3], TestDataCount)
    reader.read()

    print("Data loaded ")

    trainingData = reader._trainData
    trainingGroundTruth = reader._trainGroundTruth
    testingData = reader._testData
    testingGroundTruth = reader._testGroundTruth

    print("LeNet training started... ")

    predictionModel = train(trainingData, trainingGroundTruth)

    print("LeNet training completed ")
    print("LeNet testing started ")

    predictionResult = test(predictionModel, testingData, testingGroundTruth)

    return 0 if checkResult(predictionResult, testingGroundTruth, TestDataCount) else -1

if __name__ == '__main__':
    sys.exit(main())
