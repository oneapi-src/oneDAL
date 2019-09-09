# file: datastructures_packedsymmetric.py
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

#
# !  Content:
# !    Python example of using packed data structures
# !*****************************************************************************

#
## <a name = "DAAL-EXAMPLE-PY-DATASTRUCTURES_PACKEDSYMMETRIC"></a>
## \example datastructures_packedsymmetric.py
#

import os
import sys

import numpy as np

from daal.data_management import PackedSymmetricMatrix, NumericTableIface, BlockDescriptor, readOnly, readWrite

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printArray


if __name__ == "__main__":

    print("Packed symmetric matrix example\n")

    nDim = 5
    firstReadRow = 0
    nRead = 5

    # Example of using a packed symmetric matrix
    data = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float64)

    dataTable = PackedSymmetricMatrix(NumericTableIface.lowerPackedSymmetricMatrix, data)

    block = BlockDescriptor()

    # Read a block of rows
    dataTable.getBlockOfRows(firstReadRow, nRead, readOnly, block)
    print("{} rows are read".format(block.getNumberOfRows()))
    printArray(block.getArray(), nDim, block.getNumberOfRows(), block.getNumberOfColumns(),
               "Print 3 rows from packed symmetric matrix as float:")

    # Read a feature(column) and write into it
    readFeatureIdx = 2
    dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nDim, readWrite, block)
    printArray(block.getArray(), 1, block.getNumberOfRows(), block.getNumberOfColumns(),
               "Print the third feature of packed symmetric matrix:")

    # Set new value to a buffer and release it
    dataBlock = block.getArray()
    dataBlock[readFeatureIdx - 1] = -1
    dataBlock[readFeatureIdx + 1] = -2
    dataTable.releaseBlockOfColumnValues(block)

    # Read a block of rows. Ensure that data has changed
    dataTable.getBlockOfRows(firstReadRow, nRead, readOnly, block)
    print("{} rows are read".format(block.getNumberOfRows()))
    printArray(block.getArray(), nDim, block.getNumberOfRows(), block.getNumberOfColumns(),
               "Print 3 rows from packed symmetric matrix as float:")
