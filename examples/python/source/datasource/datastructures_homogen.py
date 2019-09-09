# file: datastructures_homogen.py
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

## <a name="DAAL-EXAMPLE-PY-DATASTRUCTURES_HOMOGEN"></a>
## @example datastructures_homogen.py

import os
import sys

import numpy as np

from daal.data_management import HomogenNumericTable, BlockDescriptor, readOnly

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printArray


if __name__ == "__main__":

    print("Homogeneous numeric table example\n")

    data = np.array([(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
                     (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2),
                     (2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3),
                     (3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4),
                     (4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5),
                     (5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 1),
                     (6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 2),
                     (7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 3),
                     (8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 4),
                     (9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 5),])

    nObservations = len(data)
    nFeatures = len(data[0])
    firstReadRow = 0
    nRead = 3
    # Construct AOS numericTable for a data array with nFeatures fields and nObservations elements
    # Dictionary will be initialized with type information from ndarray
    dataTable = HomogenNumericTable(data)
    block = BlockDescriptor()
    num_cols = block.getNumberOfColumns()
    dataTable.getBlockOfRows(firstReadRow, nRead, readOnly, block)
    print("%s rows are read" % (block.getNumberOfRows()))
    printArray(
        block.getArray(), nFeatures, block.getNumberOfRows(), 11,
        "Print 3 rows from homogeneous data array as double:"
    )
    dataTable.releaseBlockOfRows(block)

    readFeatureIdx = 2
    dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nObservations, readOnly, block)
    printArray(block.getArray(), 1, 10, 1, "Print the third feature of homogeneous data:")
    dataTable.releaseBlockOfColumnValues(block)

    data[0][0] = 999
    dataFromNumericTable = dataTable.getArray()
    printArray(dataFromNumericTable, nFeatures, nObservations, 11, "Data from getArray:")

    newData = np.array([(1.0, 2.0),
                        (3.0, 4.0),
                        (5.0, 6.0),])

    nNewVectors = len(newData)
    nNewFeatures = len(newData[0])

    # Set new data to HomogenNumericTable. It mush have the same type as the numeric table.
    dataTable = HomogenNumericTable(newData)

    # Set a new number of columns and rows
    dataTable.setNumberOfColumns(nNewFeatures)
    dataTable.setNumberOfRows(nNewVectors)

    # Ensure the data has changed
    readFeatureIdx = 1
    dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nNewVectors, readOnly, block)
    printArray(block.getArray(), 1, 3, 1, "Print the second feature of new data:")
    dataTable.releaseBlockOfColumnValues(block)
