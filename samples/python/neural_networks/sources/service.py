# file: service.py
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
# !    Auxiliary functions used in Python neural networks samples
# !*****************************************************************************

from __future__ import division, print_function

import os
import sys
import numpy as np
from daal.data_management import SubtensorDescriptor, readOnly


class ClassificationErrorCounter(object):

    MAX_ERROR_RATE_CLASSES = 5

    def __init__(self, prediction=None, groundTruth=None):
        self.initialize()
        if prediction and groundTruth:
            self.update(prediction, groundTruth)

    def update(self, _prediction, _groundTruth):
        if not _prediction:
            raise RuntimeError("Prediction tensor should not be null")
        if not _groundTruth:
            raise RuntimeError("GroundTruth tensor should not be null")

        dimensions = _prediction.getDimensions()
        if len(dimensions) != 2:
            raise RuntimeError("Predictions tensor should have exactly two dimensions")
        rowsNum = dimensions[0]
        colsNum = dimensions[1]

        if colsNum < ClassificationErrorCounter.MAX_ERROR_RATE_CLASSES:
            raise RuntimeError("Number of classes in prediction result is not enough to compute error rate")

        predictionBlock = SubtensorDescriptor(ntype=np.float32)
        _prediction.getSubtensor([], 0, dimensions[0], readOnly, predictionBlock)
        predictionRows = predictionBlock.getArray()

        groundTruthBlock = SubtensorDescriptor(ntype=np.intc)
        _groundTruth.getSubtensor([], 0, dimensions[0], readOnly, groundTruthBlock)
        groundTruthClasses = groundTruthBlock.getArray()

        for i in range(rowsNum):
            row = predictionRows[i]
            topIndices = self.findTopIndices(row, colsNum, ClassificationErrorCounter.MAX_ERROR_RATE_CLASSES)
            groundTruthClass = groundTruthClasses[0][i]

            self._totalObjects += 1
            if groundTruthClass in topIndices:
                self._top5ClassifiedObjects += 1
                if groundTruthClass == topIndices[0]:
                    self._top1ClassifiedObjects += 1

        _prediction.releaseSubtensor(predictionBlock)
        _groundTruth.releaseSubtensor(groundTruthBlock)

    def getTop1ErrorRate(self):
        return self.getErrorRate(self._top1ClassifiedObjects)

    def getTop5ErrorRate(self):
        return self.getErrorRate(self._top5ClassifiedObjects)

    def initialize(self):
        self._totalObjects = 0
        self._top1ClassifiedObjects = 0
        self._top5ClassifiedObjects = 0

    def getErrorRate(self, topClassifiedObjectsNum):
        if self._totalObjects == 0:
            return 1.0
        return 1.0 - topClassifiedObjectsNum / np.float32(self._totalObjects)

    def findTopIndices(self, rowPtr, colsNum, topSize):
        sorted_indices = sorted(range(len(rowPtr)), key=lambda k: rowPtr[k])
        topIndices = sorted_indices[-topSize:]
        return topIndices


def checkFilesAreAvailable(basePath, datasetFileNames, numberOfFiles):

    for i in range(numberOfFiles):
        if basePath:
            filePath = basePath + "/"

        filePath += datasetFileNames[i]
        if not os.path.exists(filePath):
            return False
    return True


def selectDatasetPathOrExit(defaultDatasetsPath, userDatasetsPath, datasetFileNames, numberOfFiles):

    if userDatasetsPath and checkFilesAreAvailable(userDatasetsPath, datasetFileNames, numberOfFiles):
        return userDatasetsPath

    elif checkFilesAreAvailable(defaultDatasetsPath, datasetFileNames, numberOfFiles):
        if userDatasetsPath:
            print("Warning: Can't open dataset from path provided via command line: {}".format(userDatasetsPath))
            print("         Try to open dataset from default path: {}".format(defaultDatasetsPath))
        return defaultDatasetsPath

    print("Error: Can't open datasets from default path: {}".format(defaultDatasetsPath))
    sys.exit(-1)


def getUserDatasetPath(args):
    userDatasetsPath = ''
    if len(args) > 1:
        userDatasetsPath = args[1]
    return userDatasetsPath
