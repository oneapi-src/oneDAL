# file: serialization.py
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
