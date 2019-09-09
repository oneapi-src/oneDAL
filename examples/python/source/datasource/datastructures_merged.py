# file: datastructures_merged.py
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

## <a name="DAAL-EXAMPLE-PY-DATASTRUCTURES_MERGED"></a>
## \example datastructures_merged.py

import os
import sys

import numpy as np

from daal.data_management import (
    HomogenNumericTable, MergedNumericTable, BlockDescriptor, readWrite, readOnly
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printArray


if __name__ == "__main__":

    print("Merged numeric table example\n")

    nFeatures1 = 5
    nFeatures2 = 6
    firstReadRow = 3
    nRead = 1

    # Example of using homogeneous numeric table
    data1 = np.array([
        (0.0, 0.1, 0.2, 0.3, 0.4),
        (1.0, 1.1, 1.2, 1.3, 1.4),
        (2.0, 2.1, 2.2, 2.3, 2.4),
        (3.0, 3.1, 3.2, 3.3, 3.4),
        (4.0, 4.1, 4.2, 4.3, 4.4),
    ])

    data2 = np.array([
        (0.5, 0.6, 0.7, 0.8, 0.9, 1),
        (1.5, 1.6, 1.7, 1.8, 1.9, 2),
        (2.5, 2.6, 2.7, 2.8, 2.9, 3),
        (3.5, 3.6, 3.7, 3.8, 3.9, 4),
        (4.5, 4.6, 4.7, 4.8, 4.9, 5),
    ])

    # Create two homogen numeric tables from data arrays
    dataTable1 = HomogenNumericTable(data1)
    dataTable2 = HomogenNumericTable(data2)

    # Create merged numeric table consisting of two homogen numeric tables
    dataTable = MergedNumericTable()
    dataTable.addNumericTable(dataTable1)
    dataTable.addNumericTable(dataTable2)

    block = BlockDescriptor()

    # Read one row from merged numeric table
    dataTable.getBlockOfRows(firstReadRow, nRead, readWrite, block)
    printArray(
        block.getArray(), nFeatures1 + nFeatures2, block.getNumberOfRows(),
        block.getNumberOfColumns(), "Print 1 row from merged numeric table as double:"
    )

    # Modify row of the merged numeric table
    row = block.getArray()
    for i in range(nFeatures1 + nFeatures2):
        row[0][i] *= row[0][i]
    dataTable.releaseBlockOfRows(block)

    # Read the same row from homogen numeric tables
    dataTable1.getBlockOfRows(firstReadRow, nRead, readOnly, block)
    printArray(
        block.getArray(), nFeatures1, block.getNumberOfRows(),
        block.getNumberOfColumns(), "Print 1 row from first homogen numeric table as double:"
    )
    dataTable1.releaseBlockOfRows(block)

    dataTable2.getBlockOfRows(firstReadRow, nRead, readOnly, block)
    printArray(
        block.getArray(), nFeatures2, block.getNumberOfRows(),
        block.getNumberOfColumns(), "Print 1 row from second homogen numeric table as double:"
    )
    dataTable2.releaseBlockOfRows(block)
