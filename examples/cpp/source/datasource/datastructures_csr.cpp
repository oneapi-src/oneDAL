/* file: datastructures_csr.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
!  Content:
!    Compressed sparse rows (CSR) data structures example.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_CSR">
 * \example datastructures_csr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;

int main()
{
    std::cout << "Compressed spares rows (CSR) numeric table example" << std::endl << std::endl;

    const size_t nObservations  = 5;
    const size_t nFeatures = 5;
    const size_t firstReadRow = 1;
    const size_t nRead = 3;

    /* Example of using CSR numeric table */
    float  values[]     = {1, -1, -3, -2,  5,  4,  6,  4, -4,  2,  7,  8, -5};
    size_t colIndices[] = {1,  2,  4,  1,  2,  3,  4,  5,  1,  3,  4,  2,  5};
    size_t rowOffsets[] = {1,          4,      6,          9,         12,     14};

    CSRNumericTablePtr dataTable = CSRNumericTable::create(values, colIndices, rowOffsets, nFeatures, nObservations);
    checkPtr(dataTable.get());

    /* Read block of rows in dense format */
    BlockDescriptor<> block;
    dataTable->getBlockOfRows(firstReadRow, nRead, readOnly, block);
    std::cout << block.getNumberOfRows() << " rows are read" << std::endl << std::endl;
    printArray<float>(block.getBlockPtr(), nFeatures, block.getNumberOfRows(),
                       "Print 3 rows from CSR data array as dense float array:");
    dataTable->releaseBlockOfRows(block);

    /* Read block of rows in CSR format and write into it */
    CSRBlockDescriptor<> csrBlock;
    dataTable->getSparseBlock(firstReadRow, nRead, readWrite, csrBlock);
    float *valuesBlock = csrBlock.getBlockValuesPtr();
    size_t nValuesInBlock = csrBlock.getDataSize();
    printArray<float>(valuesBlock, nValuesInBlock, 1,
                      "Values in 3 rows from CSR data array:");
    printArray<size_t>(csrBlock.getBlockColumnIndicesPtr(), nValuesInBlock, 1,
                      "Columns indices in 3 rows from CSR data array:");
    printArray<size_t>(csrBlock.getBlockRowIndicesPtr(), nRead + 1, 1,
                      "Rows offsets in 3 rows from CSR data array:");
    for (size_t i = 0; i < nValuesInBlock; i++)
    {
        valuesBlock[i] = -(1.0f + i);
    }
    dataTable->releaseSparseBlock(csrBlock);

    /* Read block of rows in dense format */
    dataTable->getBlockOfRows(firstReadRow, nRead, readOnly, block);
    std::cout << block.getNumberOfRows() << " rows are read" << std::endl << std::endl;
    printArray<float>(block.getBlockPtr(), nFeatures, block.getNumberOfRows(),
                       "Print 3 rows from CSR data array as dense float array:");
    dataTable->releaseBlockOfRows(block);

    return 0;
}
