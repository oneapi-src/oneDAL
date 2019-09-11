# file: distributed_hdfs_dataset.py
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

import numpy as np

from daal.data_management import (
    InputDataArchive, OutputDataArchive, HomogenNumericTable, CSRNumericTable,
    StringDataSource, DataSourceIface, DataCollection
)


class DistributedHDFSDataSet:

    def __init__(self, filename, labelsfilename=None):
        self.filename = filename
        self.labelsfilename = labelsfilename

    def getAsPairRDD(self, sc):

        dataWithId = sc.wholeTextFiles(self.filename).zipWithIndex()

        def mapper(tup):
            t1, t2 = tup
            data = str(t1[1])
            nVectors = data.count('\n')

            sdds = StringDataSource("",
                                    DataSourceIface.doAllocateNumericTable,
                                    DataSourceIface.doDictionaryFromContext)
            sdds.setData(data)

            sdds.createDictionaryFromContext()
            sdds.allocateNumericTable()
            sdds.loadDataBlock(nVectors)

            dataTable = sdds.getNumericTable()

            serialized_table = serializeNumericTable(dataTable)

            return (t2, serialized_table)

        return dataWithId.map(mapper)

    def getAsPairRDDWithIndex(self, sc):
        dataWithId = sc.wholeTextFiles(self.filename).zipWithIndex()

        def mapper(tup):
            t1, _ = tup
            filename, table = t1

            data = str(table)

            nVectors = data.count('\n')

            sdds = StringDataSource("")
            sdds.setData(data)

            sdds.createDictionaryFromContext()
            sdds.allocateNumericTable()
            sdds.loadDataBlock(nVectors)

            dataTable = sdds.getNumericTable()

            serialized_dataTable = serializeNumericTable(dataTable)

            tokens = filename.split('_')
            tokens[-1] = tokens[-1].split('.')
            tokens = [item for sublist in tokens for item in sublist]

            return (int(tokens[len(tokens) - 2]) - 1, serialized_dataTable)

        return dataWithId.map(mapper)

    def getCSRAsPairRDD(self, sc):
        dataWithId = sc.wholeTextFiles(self.filename).zipWithIndex()

        def mapper(tup):
            t1, t2 = tup
            data = t1[1]

            dataTable = createSparseTable(data)
            serialized_table = serializeNumericTable(dataTable)

            return (t2, serialized_table)

        return dataWithId.map(mapper)

    def getCSRAsPairRDDWithIndex(self, sc):
        dataWithId = sc.wholeTextFiles(self.filename).zipWithIndex()

        def mapper(tup):
            t1, _ = tup
            filename, t12 = t1
            data = str(t12)

            dataTable = createSparseTable(data)
            serialized_dataTable = serializeNumericTable(dataTable)

            tokens = filename.split('_')
            tokens[-1] = tokens[-1].split('.')
            tokens = [item for sublist in tokens for item in sublist]

            return (int(tokens[len(tokens) - 2]) - 1, serialized_dataTable)

        return dataWithId.map(mapper)


def getMergedDataAndLabelsRDD(trainDatafilesPath, trainDataLabelsfilesPath, sc):
    ddTrain = DistributedHDFSDataSet(trainDatafilesPath)
    ddLabels = DistributedHDFSDataSet(trainDataLabelsfilesPath)

    dataRDD = ddTrain.getAsPairRDDWithIndex(sc)
    labelsRDD = ddLabels.getAsPairRDDWithIndex(sc)

    dataAndLablesRDD = dataRDD.cogroup(labelsRDD)

    def mapper(tup):
        key, val = tup
        t1, t2 = val
        dataNT = next(t1.__iter__())
        labelsNT = next(t2.__iter__())

        return (key, (dataNT, labelsNT))

    mergedDataAndLabelsRDD = dataAndLablesRDD.map(mapper)
    mergedDataAndLabelsRDD.count()

    return mergedDataAndLabelsRDD


def getMergedCSRDataAndLabelsRDD(trainDatafilesPath, trainDataLabelsfilesPath, sc):

    ddTrain = DistributedHDFSDataSet(trainDatafilesPath)
    ddLabels = DistributedHDFSDataSet(trainDataLabelsfilesPath)

    dataRDD = ddTrain.getCSRAsPairRDD(sc)
    labelsRDD = ddLabels.getAsPairRDDWithIndex(sc)

    dataAndLablesRDD = dataRDD.cogroup(labelsRDD)

    def mapper(tup):
        key, val = tup
        t1, t2 = val
        dataNT = next(t1.__iter__())
        labelsNT = next(t2.__iter__())

        return (key, (dataNT, labelsNT))

    mergedCSRDataAndLabelsRDD = dataAndLablesRDD.map(mapper)
    mergedCSRDataAndLabelsRDD.count()

    return mergedCSRDataAndLabelsRDD


def createSparseTable(inputData):

    rowIndexLine, columnsLine, valuesLine = inputData.split("\n")[:-1]

    nVectors = getRowLength(rowIndexLine)
    rowOffsets = np.zeros([nVectors], dtype=np.uint64)

    readRow(rowIndexLine, 0, nVectors, rowOffsets)
    nVectors = nVectors - 1

    nCols = getRowLength(columnsLine)

    colIndices = np.zeros([nCols], dtype=np.uint64)
    readRow(columnsLine, 0, nCols, colIndices)

    nNonZeros = getRowLength(valuesLine)

    data = np.zeros([nNonZeros], dtype=np.float64)
    readRow(valuesLine, 0, nNonZeros, data)

    maxCol = 0
    for i in range(nCols):
        if colIndices[i] > maxCol:
            maxCol = colIndices[i]

    nFeatures = int(maxCol)

    if nCols != nNonZeros or nNonZeros != (rowOffsets[nVectors] - 1) or nFeatures == 0 or nVectors == 0:
        raise IOError("Unable to read input data")

    return CSRNumericTable(data, colIndices, rowOffsets, nFeatures, nVectors)


def readRow(line, offset, nCols, data):
    if line is None:
        raise IOError("Unable to read input dataset")

    elements = line.split(",")[:-1]
    for j in range(nCols):
        data[offset + j] = float(elements[j])


def readSparseData(dataset, nVectors, nNonZeroValues, rowOffsets, colIndices, data):
    try:
        readRow(dataset.readline(), 0, nVectors + 1, rowOffsets)
        readRow(dataset.readline(), 0, nNonZeroValues, colIndices)
        readRow(dataset.readline(), 0, nNonZeroValues, data)
    except IOError as e:
        print(e)


def getRowLength(line):
    elements = line.split(",")[:-1]
    return len(elements)


def serializeNumericTable(dataTable):

    #  Create a data archive to serialize the numeric table
    dataArch = InputDataArchive()

    #  Serialize the numeric table into the data archive
    dataTable.serialize(dataArch)

    #  Get the length of the serialized data in bytes
    length = dataArch.getSizeOfArchive()

    #  Store the serialized data in an array
    buffer = np.zeros(length, dtype=np.ubyte)
    dataArch.copyArchiveToArray(buffer)

    return buffer


def deserializeNumericTable(buffer):

    #  Create a data archive to deserialize the numeric table
    dataArch = OutputDataArchive(buffer)

    #  Create a numeric table object
    dataTable = HomogenNumericTable()

    #  Deserialize the numeric table from the data archive
    dataTable.deserialize(dataArch)

    return dataTable


def deserializeCSRNumericTable(buffer):

    dataArch = OutputDataArchive(buffer)

    dataTable = CSRNumericTable(np.zeros(1, dtype=np.float64),
                                np.zeros(1, dtype=np.uint64),
                                np.zeros(1, dtype=np.uint64),
                                0, 0)

    dataTable.deserialize(dataArch)

    return dataTable


def deserializePartialResult(buffer, module, partial=True):
    dataArch = OutputDataArchive(buffer)
    if partial:
        deserialized_pres = module.PartialResult()
    else:
        deserialized_pres = module.Result()
    deserialized_pres.deserialize(dataArch)
    return deserialized_pres


def deserializeDataCollection(buffer):

    #  Create a data archive to deserialize the numeric table
    dataArch = OutputDataArchive(buffer)

    #  Create a data collection object
    collection = DataCollection()

    #  Deserialize the numeric table from the data archive
    collection.deserialize(dataArch)

    return collection
