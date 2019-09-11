# file: datastructures_matrix.py
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

#
# !  Content:
# !    Python example of using matrix data structures
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-DATASTRUCTURES_MATRIX">
## \example datastructures_matrix.py
#

import sys, os
import numpy as np
from daal.data_management import Matrix, BlockDescriptor, readOnly

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printArray

if __name__ == "__main__":

    print("Matrix numeric table example\n")

    nObservations = 10
    nFeatures = 11
    firstReadRow = 0
    nRead = 5

    #Example of using a matrix
    data = np.array([(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
                     (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2),
                     (2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3),
                     (3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4),
                     (4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5),
                     (5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 1),
                     (6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 2),
                     (7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 3),
                     (8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 4),
                     (9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 5)], dtype=np.float32)

    dataTable = Matrix(data, ntype=np.float32)

    block = BlockDescriptor()

    # Read a block of rows
    dataTable.getBlockOfRows(firstReadRow, nRead, readOnly, block)
    print(str(block.getNumberOfRows()) + " rows are read")
    printArray(block.getArray(), nFeatures, nRead, block.getNumberOfColumns(), "Print 5 rows from matrix data array as float:")
    dataTable.releaseBlockOfRows(block)

    readFeatureIdx = 2

    # Set new values in Matrix
    npdt = dataTable.getArray()
    npdt[firstReadRow, readFeatureIdx] = -1
    npdt[firstReadRow + 1, readFeatureIdx] = -2
    npdt[firstReadRow + 2, readFeatureIdx] = -3

    # Read a feature(column) and print it
    dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nObservations, readOnly, block)
    printArray(block.getArray(), 1, block.getNumberOfRows(), block.getNumberOfColumns(), "Print the third feature of matrix data:")
    dataTable.releaseBlockOfColumnValues(block)
