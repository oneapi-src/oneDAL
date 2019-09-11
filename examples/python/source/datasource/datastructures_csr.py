# file: datastructures_csr.py
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

## <a name="DAAL-EXAMPLE-PY-DATASTRUCTURES_CSR">
## \example datastructures_csr.py

import os
import sys

import numpy as np

from daal.data_management import BlockDescriptor, CSRBlockDescriptor, CSRNumericTable, readOnly, readWrite

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printArray


if __name__ == "__main__":

    print("Compressed spares rows (CSR) numeric table example\n")

    nObservations = 5
    nFeatures = 5
    firstReadRow = 1
    nRead = 3

    #  Example of using CSR numeric table
    values     = np.array([1, -1, -3, -2,  5,  4,  6,  4, -4,  2,  7,  8, -5],     dtype=np.float64)
    colIndices = np.array([1,  2,  4,  1,  2,  3,  4,  5,  1,  3,  4,  2,  5],     dtype=np.uint64)
    rowOffsets = np.array([1,          4,      6,          9,         12,     14], dtype=np.uint64)

    dataTable = CSRNumericTable(values, colIndices, rowOffsets, nFeatures, nObservations)

    #  Read block of rows in dense format
    block = BlockDescriptor(ntype=np.float64)
    dataTable.getBlockOfRows(firstReadRow, nRead, readOnly, block)
    print(str(block.getNumberOfRows()) + " rows are read\n")
    printArray(
        block.getArray(), nFeatures, block.getNumberOfRows(), block.getNumberOfColumns(),
        "Print 3 rows from CSR data array as dense double array:"
    )
    dataTable.releaseBlockOfRows(block)

    #  Read block of rows in CSR format and write into it
    csrBlock = CSRBlockDescriptor(ntpye=np.float32)
    num_cols = csrBlock.getNumberOfColumns()
    dataTable.getSparseBlock(firstReadRow, nRead, readWrite, csrBlock)
    valuesBlock = csrBlock.getBlockValues()
    nValuesInBlock = csrBlock.getDataSize()
    printArray(valuesBlock, nValuesInBlock, 1, num_cols, "Values in 3 rows from CSR data array:")
    printArray(
        csrBlock.getBlockColumnIndices(), nValuesInBlock, 1, num_cols,
        "Columns indices in 3 rows from CSR data array:", flt64=False
    )
    printArray(
        csrBlock.getBlockRowIndices(), nRead + 1, 1, num_cols,
        "Rows offsets in 3 rows from CSR data array:", flt64=False
    )

    for i in range(nValuesInBlock):
        valuesBlock[i] = -(1.0 + i)

    dataTable.releaseSparseBlock(csrBlock)

    #  Read block of rows in dense format
    dataTable.getBlockOfRows(firstReadRow, nRead, readOnly, block)
    print(str(block.getNumberOfRows()) + " rows are read\n")
    printArray(
        block.getArray(), nFeatures, block.getNumberOfRows(), block.getNumberOfColumns(),
        "Print 3 rows from CSR data array as dense double array:"
    )
    dataTable.releaseBlockOfRows(block)
