# file: image_dataset.py
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
# !    Python sample of an image dataset reader
# !
# !*****************************************************************************

import struct

import numpy as np

from daal.data_management import HomogenTensor, TensorIface


class RGBChannelNormalizer(object):
    def __call__(self, value):
        return value / 255.0


class ImageDatasetReader(object):
    def __init__(self, channelsNum, height, width):
        self.numberOfChannels = channelsNum
        self.objectHeight = height
        self.objectWidth = width

        self._normalizer = RGBChannelNormalizer()
        self._trainData = None
        self._trainGroundTruth = None
        self._testData = None
        self._testGroundTruth = None

    def getNumberOfTrainObjects(self):
        raise NotImplementedError()

    def getNumberOfTestObjects(self):
        raise NotImplementedError()

    def allocateTensors(self):
        numberOfObjects = self.getNumberOfTrainObjects()
        numberOfTestObjects = self.getNumberOfTestObjects()

        if numberOfObjects > 0:
            trainDataDims = [numberOfObjects, self.numberOfChannels, self.objectHeight, self.objectWidth]
            self._trainData = HomogenTensor(trainDataDims, TensorIface.doAllocate, 0)

            trainGroundTruthDims = [numberOfObjects, 1]
            self._trainGroundTruth = HomogenTensor(trainGroundTruthDims, TensorIface.doAllocate)

        if numberOfTestObjects > 0:
            testDataDims = [numberOfTestObjects, self.numberOfChannels, self.objectHeight, self.objectWidth]
            self._testData = HomogenTensor(testDataDims, TensorIface.doAllocate, 0)

            testGroundTruthDims = [numberOfTestObjects, 1]
            self._testGroundTruth = HomogenTensor(testGroundTruthDims, TensorIface.doAllocate)

    def normalizeBuffer(self, buffr, normalized, bufferSize):
        normalized[0:bufferSize] = self._normalizer(buffr[0:bufferSize])

    def tensorOffset(self, n, k=0, h=0, w=0):
        return (n * self.numberOfChannels * self.objectHeight * self.objectWidth +
                k * self.objectHeight * self.objectWidth +
                h * self.objectWidth +
                w)


class DatasetReaderMNIST(ImageDatasetReader):
    DATA_MAGIC_NUMBER = 0x00000803
    LABELS_MAGIC_NUMBER = 0x00000801

    def __init__(self, margin=0):
        super(DatasetReaderMNIST, self).__init__(1, 28 + 2 * margin, 28 + 2 * margin)
        self._numOfTrainObjects = 0
        self._numOfTestObjects = 0
        self.originalObjectWidth = 28
        self.originalObjectHeight = 28
        self.margins = margin

    def setTrainBatch(self, pathToBatchData, pathToBatchlabels, numOfObjects):
        self._trainPathData = pathToBatchData
        self._trainPathLabels = pathToBatchlabels
        self._numOfTrainObjects = numOfObjects

    def setTestBatch(self, pathToBatchData, pathToBatchLabels, numOfObjects):
        self._testPathData = pathToBatchData
        self._testPathLabels = pathToBatchLabels
        self._numOfTestObjects = numOfObjects

    def read(self):
        self.objectWidth = self.originalObjectWidth + 2 * self.margins
        self.objectHeight = self.originalObjectHeight + 2 * self.margins
        self.allocateTensors()

        if self._numOfTrainObjects:
            self.readBatchDataFile(self._trainPathData, self._trainData, self._numOfTrainObjects)
            self.readBatchLabelsFile(self._trainPathLabels, self._trainGroundTruth, self._numOfTrainObjects)

        if self._numOfTestObjects > 0:
            self.readBatchDataFile(self._testPathData, self._testData, self._numOfTestObjects)
            self.readBatchLabelsFile(self._testPathLabels, self._testGroundTruth, self._numOfTestObjects)

    def getNumberOfTrainObjects(self):
        return self._numOfTrainObjects

    def getNumberOfTestObjects(self):
        return self._numOfTestObjects

    def readBatchDataFile(self, batchPath, data, numOfObjects):
        with open(batchPath, 'rb') as batchStream:
            dataRaw = data.getArray()
            self.readDataBatch(batchStream, dataRaw, numOfObjects)

    def readBatchLabelsFile(self, batchPath, labels, numOfObjects):
        with open(batchPath, 'rb') as batchStream:
            labelsRaw = labels.getArray()
            self.readLabelsBatch(batchStream, labelsRaw, numOfObjects)

    def readDataBatch(self, stream, tensorData, numOfObjects):
        magicNumber = readDword(stream)
        if magicNumber != DatasetReaderMNIST.DATA_MAGIC_NUMBER:
            raise("Invalid data file format")

        numberOfImages = readDword(stream)
        if numberOfImages < numOfObjects:
            raise("Number of objects too large")

        numberOfRows = readDword(stream)
        if numberOfRows != self.originalObjectWidth:
            raise("Batch contains invalid images")

        numberOfColumns = readDword(stream)
        if numberOfColumns != self.originalObjectHeight:
            raise("Batch contains invalid images")

        bufferSize = self.originalObjectWidth * self.originalObjectHeight

        # Reshape the data in place into a 1D array
        tensorData.shape = (-1)

        for objectCounter in range(numOfObjects):
            channelBuffer = np.fromstring(stream.read(bufferSize), dtype=np.uint8)
            tensorDataView = tensorData[self.tensorOffset(objectCounter):]
            tensorDataView = tensorDataView[self.margins * self.objectWidth:]
            for i in range(self.originalObjectHeight):
                tensorDataView = tensorDataView[self.margins:]
                self.normalizeBuffer(channelBuffer[i * self.originalObjectWidth:], tensorDataView, self.originalObjectWidth)
                tensorDataView = tensorDataView[self.originalObjectWidth + self.margins:]

        # Restore the original shape
        tensorData.shape = (numOfObjects, self.numberOfChannels, self.originalObjectHeight, self.originalObjectWidth)

    def readLabelsBatch(self, stream, labelsData, numOfObjects):
        magicNumber = readDword(stream)
        if magicNumber != DatasetReaderMNIST.LABELS_MAGIC_NUMBER:
            raise("Invalid data file format")

        numberOfItems = readDword(stream)
        if numberOfItems < numOfObjects:
            raise("Number of objects too large")

        for objectCounter in range(numOfObjects):
            labelsData[objectCounter] = struct.unpack('B', stream.read(1))

def readDword(stream):
    dword = struct.unpack('I', stream.read(4))[0]
    return endianDwordConversion(dword)

def endianDwordConversion(dword):
    return (((dword >> 24) & 0x000000FF) |
            ((dword >>  8) & 0x0000FF00) |
            ((dword <<  8) & 0x00FF0000) |
            ((dword << 24) & 0xFF000000))
