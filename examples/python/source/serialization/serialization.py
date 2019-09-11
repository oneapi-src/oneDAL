# file: serialization.py
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

## <a name="DAAL-EXAMPLE-PY-SERIALIZATION"></a>
## \example serialization.py

import os
import sys

import numpy as np

from daal.data_management import HomogenNumericTable, FileDataSource, DataSource, InputDataArchive, OutputDataArchive

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

#  Input data set parameters
datasetFileName = os.path.join('..', 'data', 'batch', 'serialization.csv')


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


if __name__ == "__main__":

    #  Initialize FileDataSource_CSVFeatureManager to retrieve the input data from a .csv file
    dataSource = FileDataSource(
        datasetFileName, DataSource.doAllocateNumericTable, DataSource.doDictionaryFromContext
    )

    #  Retrieve the data from the input file
    dataSource.loadDataBlock()

    #  Retrieve a numeric table
    dataTable = dataSource.getNumericTable()

    #  Print the original data
    printNumericTable(dataTable, "Data before serialization:")

    #  Serialize the numeric table into the memory buffer
    buffer = serializeNumericTable(dataTable)

    #  Deserialize the numeric table from the memory buffer
    restoredDataTable = deserializeNumericTable(buffer)

    #  Print the restored data
    printNumericTable(restoredDataTable, "Data after deserialization:")
