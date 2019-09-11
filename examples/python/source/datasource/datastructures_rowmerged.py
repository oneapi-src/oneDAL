# file: datastructures_rowmerged.py
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
# !    Python row merged data structures example.
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-DATASTRUCTURES_ROWMERGED"></a>
## \example datastructures_rowmerged.py
#

import os
import sys

import numpy as np

from daal.data_management import (
    DictionaryIface, HomogenNumericTable, RowMergedNumericTable, BlockDescriptor, convertToHomogen, readWrite, readOnly
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printArray, printNumericTable

if __name__ == "__main__":

    print("Row merged numeric table example\n")

    nObservations1 = 5
    nObservations2 = 6
    nFeatures = 5
    firstReadRow = 3
    nRead = 6
    featureIdx = 2

    # Example of using homogeneous numeric table
    data1 = np.array([(0.0, 0.1, 0.2, 0.3, 0.4,),
                      (1.0, 1.1, 1.2, 1.3, 1.4,),
                      (2.0, 2.1, 2.2, 2.3, 2.4,),
                      (3.0, 3.1, 3.2, 3.3, 3.4,),
                      (4.0, 4.1, 4.2, 4.3, 4.4,),])

    data2 = np.array([(0.5, 0.6, 0.7, 0.8, 0.9,),
                      (1.5, 1.6, 1.7, 1.8, 1.9,),
                      (2.5, 2.6, 2.7, 2.8, 2.9,),
                      (3.5, 3.6, 3.7, 3.8, 3.9,),
                      (4.5, 4.6, 4.7, 4.8, 4.9,),
                      (5.5, 5.6, 5.7, 5.8, 5.9,),])

    # Create row merged numeric table consisting of two homogen numeric tables
    table1 = HomogenNumericTable(DictionaryIface.equal, data1)
    table2 = HomogenNumericTable(DictionaryIface.equal, data2)

    dataTable = RowMergedNumericTable()
    dataTable.addNumericTable(table1)
    dataTable.addNumericTable(table2)

    block = BlockDescriptor()

    # Read one row from merged numeric table
    dataTable.getBlockOfRows(0, nObservations1 + nObservations2, readWrite, block)
    printArray(block.getArray(), nFeatures, block.getNumberOfRows(), block.getNumberOfColumns(),
               "Print rows from row merged numeric table as float:")

    # Modify row of the merged numeric table
    row = block.getArray()
    for i in range(nObservations1 + nObservations2):
        row[i][featureIdx] *= row[i][featureIdx]
    dataTable.releaseBlockOfRows(block)

    dataTable.getBlockOfRows(0, nObservations1 + nObservations2, readOnly, block)
    printArray(block.getArray(), nFeatures, block.getNumberOfRows(), block.getNumberOfColumns(),
               "Print rows from row merged numeric table as float:")
    dataTable.releaseBlockOfRows(block)

    finalizedTable = convertToHomogen(dataTable)

    printNumericTable(finalizedTable, "Row merged table converted to homogen numeric table")
